"""
turbo_pipeline.py — Turbo-DDCM Video Compression Pipeline for Wan2.1

Encode: ground-truth video → multi-atom codebook indices + signs
Decode: indices + signs → reconstructed video

Key features vs basic SDE-RF pipeline:
  1. Multi-atom selection (M atoms per frame per step) — much better quality
  2. Residual-guided selection: ⟨z_i, x₀ − x̂₀|t⟩ (Eq.13)
  3. DDIM tail: last N steps deterministic (no bits)
  4. Deterministic x_T from seed (no bits for initial noise)
"""

import torch
import time
from typing import List, Tuple
from PIL import Image

from .sde_convert import (
    velocity_to_score,
    diffusion_coeff,
    sde_drift,
    ode_sample_loop,
    shifted_timesteps,
)
from .turbo_codebook import TurboPerFrameCodebook, TurboBitstream
from .wan_wrapper import WanWrapper
from .spectral import SpectralSchedule, IdentitySchedule, apply_spectral_mask
from .spectral_codebook import SpectralBandCodebook, ThreeBandStepSchedule


class TurboDDCMWanPipeline:
    """Turbo-DDCM video compression pipeline for Wan2.1.

    Parameters:
        K: codebook size (atoms per step per frame), 16384+ recommended
        M: atoms selected per frame per step (quality knob)
        num_steps: total sampling steps
        num_ddim_tail: last N steps use ODE (no bits)
        g_scale: SDE diffusion coefficient scale
    """

    def __init__(
        self,
        model: WanWrapper,
        K: int = 16384,
        M: int = 32,
        num_steps: int = 20,
        num_ddim_tail: int = 3,
        guidance_scale: float = 5.0,
        g_scale: float = 3.0,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        M_tail: int = None,
        tail_latent_frames: int = 2,
        spectral_schedule: SpectralSchedule = None,
        spectral_soft_taper: bool = False,
        spectral_codebook_schedule: ThreeBandStepSchedule = None,
        spectral_codebook_preserve_spectrum: bool = False,
        spectral_codebook_mode: str = None,
        vectorized_atom_gen: bool = False,
        skip_model_alternate: bool = False,
        model_call_skip: str = "none",
        model_call_period: int = 2,
        model_call_skip_until: int = -1,
    ):
        self.model = model
        self.K = K
        self.M = M
        self.M_tail = M_tail
        self.tail_latent_frames = tail_latent_frames
        self.num_steps = num_steps
        self.num_ddim_tail = num_ddim_tail
        self.num_sde_steps = num_steps - num_ddim_tail
        self.guidance_scale = guidance_scale
        self.g_scale = g_scale
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.seed = seed
        self.spectral_schedule = spectral_schedule or IdentitySchedule()
        self.spectral_soft_taper = spectral_soft_taper
        self.spectral_codebook_schedule = spectral_codebook_schedule
        self.spectral_codebook_preserve_spectrum = spectral_codebook_preserve_spectrum
        # Resolve effective mode (back-compat: preserve_spectrum bool → "spectrum_preserving")
        if spectral_codebook_mode is not None:
            self.spectral_codebook_mode = spectral_codebook_mode
        elif spectral_codebook_preserve_spectrum:
            self.spectral_codebook_mode = "spectrum_preserving"
        elif spectral_codebook_schedule is not None:
            self.spectral_codebook_mode = "band_masked"
        else:
            self.spectral_codebook_mode = None  # not using V2-a path
        assert self.spectral_codebook_mode in (
            None, "band_masked", "spectrum_preserving",
            "gaussian_control", "selection_only",
        ), f"unknown spectral_codebook_mode={self.spectral_codebook_mode!r}"

        self.device = model.device
        self.latent_shape = model.get_latent_shape(num_frames, height, width)
        self.frame_shape = model.get_frame_shape(height, width)
        self.num_latent_frames = self.latent_shape[1]

        self.timesteps = shifted_timesteps(
            num_steps, shift=model.flow_shift, device=self.device
        )

        self.vectorized_atom_gen = vectorized_atom_gen
        if skip_model_alternate and model_call_skip == "none":
            model_call_skip = "u_cache"
        assert model_call_skip in ("none", "u_cache", "x0_cache")
        assert model_call_period >= 1
        self.model_call_skip = model_call_skip
        self.model_call_period = model_call_period
        self.model_call_skip_until = (
            self.num_sde_steps if model_call_skip_until < 0
            else min(model_call_skip_until, self.num_sde_steps)
        )
        self.codebook = TurboPerFrameCodebook(
            K=K, M=M, frame_shape=self.frame_shape,
            seed=seed, device=self.device,
            vectorized_atom_gen=vectorized_atom_gen,
        )
        self.spectral_codebook = None
        if self.spectral_codebook_schedule is not None:
            self.spectral_codebook = SpectralBandCodebook(
                K=K, M=M, frame_shape=self.frame_shape,
                seed=seed, device=self.device,
            )

        # Print stats
        bits_per_index = self.codebook.bits_per_index
        bits_normal = M * (bits_per_index + 1)
        if M_tail:
            bits_tail = M_tail * (bits_per_index + 1)
            n_tail = min(tail_latent_frames, self.num_latent_frames)
            n_normal = self.num_latent_frames - n_tail
            total_bits = self.num_sde_steps * (n_normal * bits_normal + n_tail * bits_tail)
        else:
            total_bits = self.num_sde_steps * self.num_latent_frames * bits_normal
        # gaussian_control mode transmits no atoms — override to actual cost
        if self.spectral_codebook_mode == "gaussian_control":
            total_bits = 0
        self._total_codebook_bits = total_bits
        total_pixels = num_frames * height * width
        bpp = total_bits / total_pixels
        duration_s = num_frames / 16.0
        bitrate_kbps = total_bits / duration_s / 1000.0 if duration_s else 0.0

        print(f"Turbo-DDCM Pipeline:")
        print(f"  Video: {num_frames}f @ {height}x{width}")
        print(f"  Latent: {self.latent_shape}, frame: {self.frame_shape}")
        if M_tail:
            print(f"  K={K}, M={M}, M_tail={M_tail} (last {tail_latent_frames} lat frames)")
        else:
            print(f"  K={K}, M={M}")
        print(f"  T_sde={self.num_sde_steps}, DDIM_tail={num_ddim_tail}, gen_batch={self.codebook.gen_batch}")
        print(f"  Total: {total_bits} bits ({total_bits/8:.0f} bytes)")
        print(f"  BPP: {bpp:.6f}, Bitrate: {bitrate_kbps:.2f} kbps")
        if not isinstance(self.spectral_schedule, IdentitySchedule):
            print(f"  Spectral schedule: {self.spectral_schedule} (V0, soft={self.spectral_soft_taper})")
        if self.spectral_codebook_schedule is not None:
            mode = (
                "spectrum-preserving replacement"
                if self.spectral_codebook_preserve_spectrum
                else "band-limited negative control"
            )
            print(f"  Spectral codebook schedule: {self.spectral_codebook_schedule} (V2-a {mode})")
            probe_steps = [
                ("early", 0),
                ("middle", self.spectral_codebook_schedule.low_until),
                ("late", self.spectral_codebook_schedule.mid_until),
            ]
            for label, probe_step in probe_steps:
                n_active, n_total, d_band = self.spectral_codebook.active_coefficients(
                    self.spectral_codebook_schedule, probe_step,
                )
                print(
                    f"    {label}: {n_active}/{n_total} spatial DCT coeffs "
                    f"({100.0 * n_active / n_total:.1f}%), D_band={d_band}"
                )

        if self.model_call_skip != "none":
            print(
                f"  Model-call skipping: mode={self.model_call_skip}, "
                f"period={self.model_call_period}, "
                f"until_step={self.model_call_skip_until}"
            )

    def _model_call_or_cache(
        self,
        model_fn,
        x_t: torch.Tensor,
        t_curr: float,
        step_idx: int,
        cached_u: torch.Tensor = None,
        cached_x0: torch.Tensor = None,
    ):
        """Return velocity using either Wan or a deterministic cache.

        `x0_cache` is the preferred skipped-step approximation:

            x0_hat_i ~= cached_x0
            u_i = (x_i - cached_x0) / t_i

        This preserves the current state dependence while avoiding a Wan call.
        """
        should_skip = (
            self.model_call_skip != "none"
            and step_idx < self.model_call_skip_until
            and step_idx < self.num_sde_steps
            and step_idx % self.model_call_period != 0
        )
        if self.model_call_skip == "u_cache":
            should_skip = should_skip and cached_u is not None
        elif self.model_call_skip == "x0_cache":
            should_skip = (
                should_skip
                and cached_x0 is not None
                and cached_x0.shape == x_t.shape
                and t_curr > 1e-6
            )

        if should_skip:
            if self.model_call_skip == "u_cache":
                return cached_u, cached_u, cached_x0, False
            u_t = (x_t - cached_x0) / t_curr
            return u_t, cached_u, cached_x0, False

        u_t = model_fn(x_t, t_curr)
        x0_hat = x_t - t_curr * u_t
        return u_t, u_t.detach(), x0_hat.detach(), True

    def _get_M_for_frame(self, f: int) -> int:
        """Return M for a given latent frame index (tail frames may use M_tail)."""
        if self.M_tail and f >= self.num_latent_frames - self.tail_latent_frames:
            return self.M_tail
        return self.M

    def _band_project_residual(self, r_f, sde_idx, t_curr):
        """For selection_only mode: iDCT(M_B · DCT(r)) — band-masked residual in spatial."""
        from dct_utils import dct_2d, idct_2d
        mask = self.spectral_codebook_schedule.mask(
            sde_idx, t_curr,
            self.frame_shape[1], self.frame_shape[2],
            self.device, torch.float32,
        )
        r_dct = dct_2d(r_f.float().to(self.device))
        r_dct_masked = r_dct * mask.unsqueeze(0)  # broadcast over channel
        return idct_2d(r_dct_masked).to(r_f.dtype)

    def _gaussian_control_noise(self, sde_idx, frame_idx, t_curr):
        """For gaussian_control mode: z* = iDCT(M_B · DCT(eps1) + (1-M_B) · DCT(eps2)).

        Bypass codebook entirely. Returns (empty_idx, empty_sgn, z_spatial).
        Since DCT is orthonormal, eps_i ~ N(0,I) in spatial ↔ DCT(eps_i) ~ N(0,I) in DCT.
        Sampling directly in DCT is equivalent and avoids an extra DCT.
        """
        from .spectral_codebook import idct_2d
        mask = self.spectral_codebook_schedule.mask(
            sde_idx, t_curr,
            self.frame_shape[1], self.frame_shape[2],
            self.device, torch.float32,
        )
        C, H, W = self.frame_shape
        base_seed = self.seed * 100003 + sde_idx * 10007 + frame_idx
        gen1 = torch.Generator(device=self.device).manual_seed(base_seed)
        gen2 = torch.Generator(device=self.device).manual_seed(base_seed + 9001)
        eps1_dct = torch.randn(C, H, W, generator=gen1, device=self.device)
        eps2_dct = torch.randn(C, H, W, generator=gen2, device=self.device)
        z_dct = mask.unsqueeze(0) * eps1_dct + (1.0 - mask.unsqueeze(0)) * eps2_dct
        z_spatial = idct_2d(z_dct)
        std = z_spatial.std()
        if std > 1e-8:
            z_spatial = z_spatial / std
        return [], [], z_spatial

    def _select_noise(self, r_f, sde_idx, frame_idx, M_f, t_curr, encode_mode=True,
                      idx_in=None, sgn_in=None):
        """Unified dispatch for all codebook modes. Returns (idx, sgn, z_f).

        Encode: pass r_f (residual), idx_in/sgn_in=None.
        Decode: pass idx_in, sgn_in; r_f can be None.
        """
        mode = self.spectral_codebook_mode
        if mode is None:
            if encode_mode:
                return self.codebook.select_atoms(r_f, sde_idx, frame_idx, M_override=M_f)
            else:
                z = self.codebook.reconstruct(idx_in, sgn_in, sde_idx, frame_idx)
                return idx_in, sgn_in, z

        if mode == "gaussian_control":
            return self._gaussian_control_noise(sde_idx, frame_idx, t_curr)

        if mode == "selection_only":
            if encode_mode:
                r_band = self._band_project_residual(r_f, sde_idx, t_curr)
                return self.codebook.select_atoms(r_band, sde_idx, frame_idx, M_override=M_f)
            else:
                z = self.codebook.reconstruct(idx_in, sgn_in, sde_idx, frame_idx)
                return idx_in, sgn_in, z

        # band_masked / spectrum_preserving go through SpectralBandCodebook
        preserve = (mode == "spectrum_preserving")
        if encode_mode:
            return self.spectral_codebook.select_atoms(
                r_f, sde_idx, frame_idx,
                schedule=self.spectral_codebook_schedule,
                t=t_curr, M_override=M_f, preserve_spectrum=preserve,
            )
        else:
            z = self.spectral_codebook.reconstruct(
                idx_in, sgn_in, sde_idx, frame_idx,
                schedule=self.spectral_codebook_schedule,
                t=t_curr, preserve_spectrum=preserve,
            )
            return idx_in, sgn_in, z

    def _model_fn(self, embeds: dict, i2v_cond=None):
        """Return CFG velocity callable."""
        def fn(x_t, t):
            return self.model.predict_velocity_cfg(
                x_t, t, embeds, self.guidance_scale, i2v_cond
            )
        return fn

    # ==================================================================
    # Encode: video → codebook indices + signs
    # ==================================================================

    @torch.no_grad()
    def encode(
        self,
        frames: List[Image.Image],
        prompt: str = "",
        ref_image: Image.Image = None,
        precomputed_i2v_cond: dict = None,
    ) -> Tuple[List[List[Tuple[List[int], List[int]]]], torch.Tensor]:
        """Encode video to Turbo-DDCM bitstream.

        At each SDE step:
          1. Predict velocity u_t, compute MMSE estimate x̂₀|t = x_t − t·u_t
          2. Compute residual r = x₀ − x̂₀|t
          3. For each frame: select top-M atoms by |⟨z_i, r_f⟩|
          4. Combine atoms with sign coefficients, normalize
          5. Use as SDE noise for this step

        Args:
            frames: list of PIL Images (ground truth video)
            prompt: text description
            ref_image: reference image for I2V (first frame)
            precomputed_i2v_cond: pre-built conditioning dict (clip_fea + y).
                Use this for FLF2V where conditioning needs both first AND last
                frames (computed externally via model.encode_first_last_frames).
                Takes precedence over ref_image when both are provided.

        Returns:
            step_data: [sde_step][frame] = (indices, signs)
            x0_final: final decoded latent
        """
        embeds = self.model.encode_prompt(prompt)

        # I2V / FLF2V conditioning (CLIP + VAE refs)
        if precomputed_i2v_cond is not None:
            i2v_cond = precomputed_i2v_cond
        elif self.model.is_i2v and ref_image is not None:
            i2v_cond = self.model.encode_image(
                ref_image, self.num_frames, self.height, self.width)
        else:
            i2v_cond = None

        # VAE encode ground-truth video
        x0_true = self.model.encode_video(frames, self.height, self.width)
        self._gt_latent = x0_true  # Store for residual computation

        model_fn = self._model_fn(embeds, i2v_cond)

        # Deterministic initial noise (shared between encoder/decoder)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        step_data = []
        sde_idx = 0
        cached_u, cached_x0 = None, None  # for model_call_skip

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            # Model call or cached-velocity skip — encoder/decoder symmetric.
            u_t, cached_u, cached_x0, _ = self._model_call_or_cache(
                model_fn, x_t, t_curr, i, cached_u, cached_x0,
            )

            # Final step → ODE to t=0
            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break

            # DDIM tail → ODE step, no bits
            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            # --- Turbo-DDCM SDE step ---
            # MMSE estimate: x̂₀|t = x_t − t · u_t
            x0_hat = x_t - t_curr * u_t  # (1, C, F, H, W)

            # Residual: what the model is missing
            residual = (x0_true - x0_hat).squeeze(0)  # (C, F, H, W)

            # SDE components
            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            # Per-frame selection (with optional tail boost)
            frame_entries = []
            noise_frames = []
            for f in range(self.num_latent_frames):
                r_f = residual[:, f, :, :]  # (C, H, W)
                M_f = self._get_M_for_frame(f)
                idx, sgn, z_f = self._select_noise(
                    r_f, sde_idx, f, M_f, t_curr, encode_mode=True,
                )
                frame_entries.append((idx, sgn))
                noise_frames.append(z_f)

            step_data.append(frame_entries)
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)  # (1, C, F, H, W)
            # V0 spectral mask (encoder and decoder must use the SAME schedule)
            if self.spectral_codebook_schedule is None:
                noise_3d = apply_spectral_mask(
                    noise_3d, self.spectral_schedule, t_curr,
                    preserve_per_frame_std=True, soft=self.spectral_soft_taper,
                )
            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d

            sde_idx += 1

            # Progress
            if (i + 1) % 5 == 0 or i == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                print(f"  Encode step {i+1}/{self.num_steps}: "
                      f"residual_MSE={mse:.4f}, noise_coeff={noise_coeff:.4f}")

        return step_data, x_t

    # ==================================================================
    # Decode: codebook indices + signs → video
    # ==================================================================

    @torch.no_grad()
    def decode(
        self,
        step_data: List[List[Tuple[List[int], List[int]]]],
        prompt: str = "",
        ref_image: Image.Image = None,
        latent_correction: torch.Tensor = None,
        precomputed_i2v_cond: dict = None,
    ) -> List[Image.Image]:
        """Decode video from Turbo-DDCM bitstream.

        Replays exact same SDE trajectory as encoder using stored noise.
        Deterministic: same indices + signs + seed + prompt → same video.
        For I2V: ref_image provides the first frame conditioning.
        For FLF2V: use precomputed_i2v_cond (built externally via
            model.encode_first_last_frames) — takes precedence over ref_image.

        Args:
            latent_correction: optional (1, C, F, H, W) tensor added to final
                latent before VAE decode (e.g., residual for tail frame correction).
            precomputed_i2v_cond: pre-built conditioning dict; must match
                what was used during encode for trajectories to align.
        """
        embeds = self.model.encode_prompt(prompt)

        # I2V / FLF2V conditioning
        if precomputed_i2v_cond is not None:
            i2v_cond = precomputed_i2v_cond
        elif self.model.is_i2v and ref_image is not None:
            i2v_cond = self.model.encode_image(
                ref_image, self.num_frames, self.height, self.width)
        else:
            i2v_cond = None

        model_fn = self._model_fn(embeds, i2v_cond)

        # Same deterministic initial noise
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        sde_idx = 0
        cached_u, cached_x0 = None, None  # for model_call_skip

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t, cached_u, cached_x0, _ = self._model_call_or_cache(
                model_fn, x_t, t_curr, i, cached_u, cached_x0,
            )

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break

            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            # Reconstruct noise from stored indices + signs
            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            noise_frames = []
            for f in range(self.num_latent_frames):
                idx, sgn = step_data[sde_idx][f]
                M_f = self._get_M_for_frame(f)
                _, _, z_f = self._select_noise(
                    None, sde_idx, f, M_f, t_curr, encode_mode=False,
                    idx_in=idx, sgn_in=sgn,
                )
                noise_frames.append(z_f)
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
            # V0 spectral mask (same as encoder)
            if self.spectral_codebook_schedule is None:
                noise_3d = apply_spectral_mask(
                    noise_3d, self.spectral_schedule, t_curr,
                    preserve_per_frame_std=True, soft=self.spectral_soft_taper,
                )
            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d

            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{self.num_steps}")

        # Apply latent correction before VAE decode (e.g., tail frame residual)
        if latent_correction is not None:
            x_t = x_t + latent_correction.to(x_t.device, dtype=x_t.dtype)

        frames = self.model.decode_latent(x_t)
        return frames

    # ==================================================================
    # ODE baseline (for quality comparison)
    # ==================================================================

    @torch.no_grad()
    def generate_ode(self, prompt: str, seed: int = 42, ref_image: Image.Image = None) -> List[Image.Image]:
        """Standard ODE generation (deterministic baseline)."""
        embeds = self.model.encode_prompt(prompt)

        i2v_cond = None
        if self.model.is_i2v and ref_image is not None:
            i2v_cond = self.model.encode_image(
                ref_image, self.num_frames, self.height, self.width)

        model_fn = self._model_fn(embeds, i2v_cond)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        x_init = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)

        x_final = ode_sample_loop(model_fn, x_init, self.timesteps)
        frames = self.model.decode_latent(x_final)
        return frames

    # ==================================================================
    # Bitstream I/O
    # ==================================================================

    def save_compressed(self, step_data, filepath: str, prompt: str = ""):
        """Save step_data to binary file."""
        TurboBitstream.save(
            filepath, step_data,
            self.K, self.M,
            self.num_sde_steps, self.num_ddim_tail,
            self.num_latent_frames, self.seed,
            self.frame_shape, prompt,
            self.num_frames, self.height, self.width,
        )

    def load_and_decode(self, filepath: str) -> List[Image.Image]:
        """Load bitstream and decode to video."""
        data = TurboBitstream.load(filepath)
        return self.decode(data["step_data"], data["prompt"])
