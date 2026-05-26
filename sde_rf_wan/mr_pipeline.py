"""
mr_pipeline.py — Multi-Resolution Turbo-DDCM Pipeline (V1 MR-GVCC)

S-stage progressive-resolution GVCC (S >= 2):

  Stage s_i  (i=0..S-1):    steps transition_steps[i-1]..transition_steps[i]-1
                            latent (C, F, H_i, W_i), codebook K_i atoms
                            (with transition_steps[-1]=0 and an implicit
                            num_sde_steps boundary at the end)
  Transition at step k:     spectral DCT expand stage_i → stage_{i+1},
                            fill new high-freq per high_freq_mode
  ODE tail:                 steps num_sde_steps..num_steps-1, top resolution, no bits

High-frequency fill modes at transition:
  "random"         x_H = t_i * randn                              [SPD-style; supports any S]
  "dlc1_pure"      x_H = (1-t_i)*x_hat_0^H + t_i*randn            [DLC1 Method C; supports any S]
  "oracle"         x_H = (1-t_i)*x_0^H + t_i*randn                [encoder upper bound; joint-oracle test]
  "dlc1_codebook"  x_H = (1-t_i)*z_k^H + t_i*randn                [target-aware quantized prior; any S]
                   where z_k ≈ x_0^H via full-rank codebook selection
  "dlc2_subband"   x_H = (1-t_i)*z_k^H + t_i*randn                [DLC2 Version A; any S]
                   where z_k is in low-rank sub-band of DCT high-freq region
  "codebook"       x_H = t_i * z_k^H                              [legacy S=2 simple form, no eps]
  "prior_codebook" x_H = (1-t_i)*x_hat_0^H + t_i*z_k^H            [S=2 only; DLC1+V1 cross]

Transition alignment:
  "variance"  empirical std → sigma_expected = sqrt(1 - 2t + 2t^2)
  "timestep"  SPD kappa scaling
  "none"      no alignment
"""

import sys
import os
import math
import torch
from typing import List, Tuple, Optional, Union
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
_spd = os.path.join(os.path.dirname(_here), "spd_repro")
if _spd not in sys.path:
    sys.path.insert(0, _spd)

from .sde_convert import (
    velocity_to_score,
    diffusion_coeff,
    sde_drift,
    shifted_timesteps,
)
from .turbo_codebook import TurboPerFrameCodebook
from .dlc2_codebook import DLC2HighFreqCodebook
from spectral_noise_expansion import spectral_noise_expand


class MRTurboDDCMPipeline:
    """S-stage multi-resolution Turbo-DDCM pipeline.

    Two construction styles:

    (1) Legacy S=2 (kwargs `low_*`, `full_*`, `K_low`, `K_full`, `transition_step`).
    (2) Multi-stage: pass `resolutions` (ascending list of (h, w)),
        `transition_steps` (length S-1, ascending step indices),
        `K_per_stage` (length S).

    When (2) is provided it overrides (1).
    """

    def __init__(
        self,
        model,
        # Multi-stage interface
        resolutions: Optional[List[Tuple[int, int]]] = None,
        transition_steps_list: Optional[List[int]] = None,
        K_per_stage: Optional[List[int]] = None,
        # Legacy S=2 interface
        low_height: int = 480,
        low_width: int = 832,
        full_height: int = 720,
        full_width: int = 1280,
        K_low: int = 4096,
        K_full: int = 16384,
        transition_step: int = 8,
        # Common
        M: int = 64,
        num_steps: int = 20,
        num_ddim_tail: int = 3,
        guidance_scale: float = 1.0,
        g_scale: float = 3.0,
        num_frames: int = 33,
        seed: int = 42,
        high_freq_mode: str = "random",
        transition_align: str = "variance",
        dlc2_rank_frac: float = 0.5,
        low_target_from_full: bool = False,
        low_target_align: str = "none",
        vectorized_atom_gen: bool = False,
        model_call_skip: str = "none",
        model_call_period: int = 2,
        model_call_skip_stages: str = "all",
        transition_warmup: bool = False,
        adaptive_skip_threshold: float = 0.1,
        max_consecutive_skips: Union[int, str] = 1,
        boundary_link: bool = False,
        boundary_M: int = 64,
        boundary_fade_k: int = 4,
        boundary_lowpass: int = 1,
    ):
        self.model = model
        self.M = M
        self.num_steps = num_steps
        self.num_ddim_tail = num_ddim_tail
        self.num_sde_steps = num_steps - num_ddim_tail
        self.guidance_scale = guidance_scale
        self.g_scale = g_scale
        self.num_frames = num_frames
        self.seed = seed
        self.high_freq_mode = high_freq_mode
        self.transition_align = transition_align
        self.low_target_from_full = low_target_from_full
        self.low_target_align = low_target_align
        # Transition-aware x0_cache skipping. Within a stage, alternate Wan calls
        # per `(i - stage_start_i) % period`. Cache is reset at every transition
        # (resolution changes invalidate cached_x0) and skipping is disabled in
        # the DDIM tail.
        assert model_call_skip in ("none", "x0_cache", "adaptive_x0_cache")
        assert model_call_period >= 1
        self.model_call_skip = model_call_skip
        self.model_call_period = model_call_period
        self.transition_warmup = transition_warmup
        self.adaptive_skip_threshold = float(adaptive_skip_threshold)
        # adaptive_x0_cache: hard cap on consecutive skips per stage. Accepts
        # either a single int (applied to all stages) or a comma-separated
        # str "c0,c1,...,cS-1" for per-stage caps. 1 = at most one skip in a
        # row (matches fixed period=2's cache-staleness bound); 999 effectively
        # disables. Resolved to self.max_consec_per_stage list once S is known.
        self._max_consecutive_skips_raw = max_consecutive_skips
        # adaptive_x0_cache encoder transmits a 1-bit-per-SDE-step mask via
        # this attribute; decoder reads it back. None outside adaptive mode.
        self._encode_skip_mask = None
        # model_call_skip_stages: "all" or comma-separated stage indices.
        # Stages not in this set always run Wan (no skip).
        self._skip_stages_raw = model_call_skip_stages
        self.device = model.device

        # Normalize to multi-stage representation
        if resolutions is None:
            resolutions = [(low_height, low_width), (full_height, full_width)]
            transition_steps_list = [transition_step]
            K_per_stage = [K_low, K_full]
        assert transition_steps_list is not None and K_per_stage is not None
        assert len(resolutions) >= 2
        assert len(transition_steps_list) == len(resolutions) - 1
        assert len(K_per_stage) == len(resolutions)
        # ascending sorted
        for i in range(len(transition_steps_list) - 1):
            assert transition_steps_list[i] < transition_steps_list[i + 1]
        assert transition_steps_list[-1] < self.num_sde_steps

        self.S = len(resolutions)
        self.resolutions = resolutions
        self.transition_steps = transition_steps_list
        self.K_per_stage = K_per_stage

        # Resolve skip_stages now that S is known.
        if self._skip_stages_raw == "all":
            self.skip_stages = set(range(self.S))
        else:
            self.skip_stages = {int(s) for s in self._skip_stages_raw.split(",") if s.strip() != ""}
        assert all(0 <= s < self.S for s in self.skip_stages), (
            f"skip_stages={self.skip_stages} out of range for S={self.S}"
        )

        # Resolve per-stage max_consecutive_skips list.
        raw = self._max_consecutive_skips_raw
        if isinstance(raw, str):
            parts = [int(x) for x in raw.split(",") if x.strip() != ""]
        elif isinstance(raw, (list, tuple)):
            parts = [int(x) for x in raw]
        else:
            parts = [int(raw)]
        if len(parts) == 1:
            parts = parts * self.S
        assert len(parts) == self.S, (
            f"max_consecutive_skips must have 1 or S={self.S} entries, got {parts}"
        )
        assert all(c >= 0 for c in parts), f"max_consecutive_skips must be >=0, got {parts}"
        self.max_consec_per_stage = parts

        # Compute latent / frame shapes per stage
        self.latent_shapes = [
            model.get_latent_shape(num_frames, h, w) for (h, w) in resolutions
        ]
        self.frame_shapes = [
            model.get_frame_shape(h, w) for (h, w) in resolutions
        ]

        # F_lat must be identical across stages
        F_set = set(lat[1] for lat in self.latent_shapes)
        assert len(F_set) == 1, f"F_lat inconsistent across stages: {F_set}"
        self.F_lat = F_set.pop()
        self.C = self.latent_shapes[-1][0]

        # Legacy S=2 aliases (used by codebook/oracle/prior_codebook paths)
        self.low_lat = self.latent_shapes[0]
        self.full_lat = self.latent_shapes[-1]
        self.low_frame = self.frame_shapes[0]
        self.full_frame = self.frame_shapes[-1]
        self.low_h, self.low_w = resolutions[0]
        self.full_h, self.full_w = resolutions[-1]
        self.H_lo, self.W_lo = self.low_lat[2], self.low_lat[3]
        self.H_hi, self.W_hi = self.full_lat[2], self.full_lat[3]
        self.transition_step = self.transition_steps[0]

        self.timesteps = shifted_timesteps(
            num_steps, shift=model.flow_shift, device=self.device
        )

        self.vectorized_atom_gen = vectorized_atom_gen
        # One codebook per stage; per-stage seed offsets avoid inter-stage collision
        self.codebooks = [
            TurboPerFrameCodebook(
                K=K_per_stage[i], M=M, frame_shape=self.frame_shapes[i],
                seed=seed + i, device=self.device,
                vectorized_atom_gen=vectorized_atom_gen,
            )
            for i in range(self.S)
        ]
        # Legacy S=2 aliases
        self.cb_low = self.codebooks[0]
        self.cb_full = self.codebooks[-1]

        # Boundary-link codebook (GOP-link latent correction). Single full-res
        # latent frame per GOP boundary, separate seed to avoid collision with
        # main per-stage codebooks. Variant 2 (fade-in over boundary_fade_k
        # leading latent frames) is applied at decode time via latent_correction.
        self.boundary_link = boundary_link
        self.boundary_M = int(boundary_M)
        self.boundary_fade_k = int(boundary_fade_k)
        self.boundary_lowpass = int(boundary_lowpass)
        assert self.boundary_lowpass >= 1, "boundary_lowpass=1 means full-res (off)"
        self.boundary_codebook = None
        if boundary_link:
            # Boundary residual lives in the FULL-res frame space
            # (z_first_GT - prev_z_last are both full-res latent frames).
            full_frame = self.frame_shapes[-1]
            if self.boundary_lowpass > 1:
                ds = self.boundary_lowpass
                C_, H_, W_ = full_frame
                assert H_ % ds == 0 and W_ % ds == 0, (
                    f"boundary frame {full_frame} not divisible by ds={ds}"
                )
                cb_frame = (C_, H_ // ds, W_ // ds)
            else:
                cb_frame = full_frame
            self._boundary_cb_frame = cb_frame
            self.boundary_codebook = TurboPerFrameCodebook(
                K=K_per_stage[-1], M=self.boundary_M, frame_shape=cb_frame,
                seed=seed + 31337, device=self.device,
                vectorized_atom_gen=vectorized_atom_gen,
            )

        # Transition atom step index for cb_full (S=2 codebook/prior_codebook only).
        # Kept outside [0, num_sde_steps) so it never collides with regular SDE steps.
        self._trans_sde_idx = self.num_sde_steps

        if self.S > 2 and high_freq_mode not in (
            "random", "dlc1_pure", "oracle", "dlc1_codebook", "dlc2_subband",
        ):
            raise NotImplementedError(
                f"high_freq_mode={high_freq_mode!r} is currently S=2 only. "
                f"Got S={self.S}. Use random / dlc1_pure / oracle / "
                f"dlc1_codebook / dlc2_subband for S>2."
            )

        # DLC2 sub-band codebooks (lazy: only construct if used)
        self.dlc2_rank_frac = dlc2_rank_frac
        self.dlc2_codebooks: List[DLC2HighFreqCodebook] = []
        if high_freq_mode == "dlc2_subband":
            for s_idx in range(self.S - 1):
                src_shape = (self.C, self.latent_shapes[s_idx][2],
                             self.latent_shapes[s_idx][3])
                tgt_shape = (self.C, self.latent_shapes[s_idx + 1][2],
                             self.latent_shapes[s_idx + 1][3])
                self.dlc2_codebooks.append(DLC2HighFreqCodebook(
                    K=K_per_stage[s_idx + 1], M=M,
                    src_lat_shape=src_shape, tgt_lat_shape=tgt_shape,
                    rank_frac=dlc2_rank_frac,
                    seed=seed + 100 + s_idx,  # avoid collision with cb_per_stage
                    device=self.device,
                ))

        # Bitrate bookkeeping
        bits = 0
        for s in range(self.S):
            n_steps = self._steps_in_stage(s)
            bits += n_steps * self.F_lat * M * (self.codebooks[s].bits_per_index + 1)
        # S=2 codebook/prior_codebook also transmits transition atoms
        if self.S == 2 and high_freq_mode in ("codebook", "prior_codebook"):
            bits += self.F_lat * M * (self.codebooks[-1].bits_per_index + 1)
        # dlc1_codebook: one transition atom set per transition (any S)
        if high_freq_mode == "dlc1_codebook":
            for s_idx in range(self.S - 1):
                bits += self.F_lat * M * (
                    self.codebooks[s_idx + 1].bits_per_index + 1
                )
        # dlc2_subband: same per-atom bits, possibly different K via DLC2 codebooks
        if high_freq_mode == "dlc2_subband":
            for s_idx, cb_dlc2 in enumerate(self.dlc2_codebooks):
                bits += self.F_lat * M * (cb_dlc2.bits_per_index + 1)
        # boundary_link: 1 boundary residual per GOP boundary (added at runner
        # level, so we record per-GOP overhead here for accounting).
        self._boundary_bits_per_gop = 0
        if self.boundary_link:
            # M_boundary atoms × (bits_per_idx + 1) + 32-bit fp32 scale
            self._boundary_bits_per_gop = (
                self.boundary_M * (self.boundary_codebook.bits_per_index + 1) + 32
            )
        self._total_codebook_bits = bits
        full_h, full_w = resolutions[-1]
        self.bpp = bits / (num_frames * full_h * full_w)

        print(f"MR-Turbo-DDCM ({high_freq_mode}, align={transition_align}, S={self.S}):")
        for i, (h, w) in enumerate(resolutions):
            lat = self.latent_shapes[i]
            print(f"  Stage {i}: {h}×{w}  lat={lat}  K={K_per_stage[i]}")
        edges = [0] + transition_steps_list + [self.num_sde_steps]
        ranges = " | ".join(
            f"{edges[i]}-{edges[i+1]-1} s{i}" for i in range(self.S)
        )
        print(f"  Steps: {ranges} | {num_ddim_tail} ODE tail")
        if self.model_call_skip != "none":
            warmup = " +1-step warmup after transition" if self.transition_warmup else ""
            stages_str = (
                "all" if self.skip_stages == set(range(self.S))
                else ",".join(str(s) for s in sorted(self.skip_stages))
            )
            if self.model_call_skip == "adaptive_x0_cache":
                cap_str = (
                    str(self.max_consec_per_stage[0])
                    if len(set(self.max_consec_per_stage)) == 1
                    else ",".join(str(c) for c in self.max_consec_per_stage)
                )
                print(
                    f"  Model-call skipping: mode=adaptive_x0_cache, "
                    f"threshold={self.adaptive_skip_threshold}, "
                    f"max_consecutive_skips=[{cap_str}], "
                    f"skip_stages={stages_str} "
                    f"(encoder always runs Wan + transmits 1-bit/step skip mask; "
                    f"decoder skips when err=||u_T-u_F||/||u_T|| < threshold)"
                )
            else:
                print(
                    f"  Model-call skipping: mode={self.model_call_skip}, "
                    f"period={self.model_call_period}, "
                    f"skip_stages={stages_str}{warmup} "
                    f"(reset cache at every transition; no skip in DDIM tail)"
                )
        if self.boundary_link:
            ds_note = (
                "" if self.boundary_lowpass == 1
                else f" + lowpass ds={self.boundary_lowpass} → "
                     f"frame={self._boundary_cb_frame}"
            )
            print(
                f"  Boundary link: M={self.boundary_M}, fade_k={self.boundary_fade_k}"
                f"{ds_note} (+{self._boundary_bits_per_gop} bits per GOP boundary)"
            )
        print(f"  M={M}, g_scale={g_scale}, BPP={self.bpp:.6f}")

    # ------------------------------------------------------------------
    # Boundary-link helpers (GOP-link latent correction; variant 2 fade-in)
    # ------------------------------------------------------------------

    def _boundary_downsample(self, x):
        """Avg-pool (C,H,W) → (C,H/ds,W/ds). ds=1 is identity."""
        ds = self.boundary_lowpass
        if ds == 1:
            return x
        return torch.nn.functional.avg_pool2d(
            x.unsqueeze(0).float(), kernel_size=ds, stride=ds,
        ).squeeze(0).to(x.dtype)

    def _boundary_upsample(self, x_low, target_hw):
        """Bilinear (C,H/ds,W/ds) → (C,H,W). ds=1 is identity."""
        if self.boundary_lowpass == 1:
            return x_low
        return torch.nn.functional.interpolate(
            x_low.unsqueeze(0).float(), size=target_hw,
            mode="bilinear", align_corners=False,
        ).squeeze(0).to(x_low.dtype)

    @torch.no_grad()
    def encode_boundary(self, z_target, z_anchor):
        """Encode a GOP-boundary residual r = z_target - z_anchor.

        Args:
            z_target: (C, H, W) — VAE-encoded GT first latent frame of next GOP
            z_anchor: (C, H, W) — last decoded latent frame of previous GOP

        Returns: (indices, signs, scale) — atoms + per-frame MSE-optimal scale
                 in the (possibly downsampled) boundary codebook space.
        """
        assert self.boundary_codebook is not None, "boundary_link not enabled"
        r_full = (z_target.to(self.device).float() - z_anchor.to(self.device).float())
        r_low = self._boundary_downsample(r_full)
        idx, sgn, z_unit = self.boundary_codebook.select_atoms(
            r_low, step_idx=0, frame_idx=0,
        )
        zf = z_unit.float().reshape(-1)
        rf = r_low.float().reshape(-1)
        scale = (zf @ rf / (zf @ zf).clamp_min(1e-8)).item()
        return idx, sgn, scale

    @torch.no_grad()
    def decode_boundary(self, atoms, target_hw):
        """Decode boundary residual from atoms back to full-res latent frame.

        Args:
            atoms: (indices, signs, scale) — output of encode_boundary
            target_hw: (H, W) — full-res latent spatial shape

        Returns: (C, H, W) — reconstructed boundary residual at full-res
        """
        assert self.boundary_codebook is not None, "boundary_link not enabled"
        idx, sgn, scale = atoms
        z_unit_low = self.boundary_codebook.reconstruct(
            idx, sgn, step_idx=0, frame_idx=0,
        )
        r_recon_low = z_unit_low.float() * float(scale)
        r_recon_full = self._boundary_upsample(r_recon_low, target_hw)
        return r_recon_full

    def build_boundary_correction(self, r_recon_full):
        """Build a (1, C, F, H, W) latent_correction tensor with fade-in over
        the first boundary_fade_k latent frames. Weight schedule: linearly
        decaying from 1.0 to 1/(k+1)."""
        k = max(1, min(self.boundary_fade_k, self.F_lat))
        C, H, W = r_recon_full.shape
        out = torch.zeros(
            (1, C, self.F_lat, H, W),
            device=r_recon_full.device, dtype=r_recon_full.dtype,
        )
        for t in range(k):
            w = (k - t) / (k + 1)  # 1.0, 0.75, 0.5, 0.25 for k=4
            out[0, :, t, :, :] = w * r_recon_full
        return out

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _steps_in_stage(self, s: int) -> int:
        edges = [0] + self.transition_steps + [self.num_sde_steps]
        return edges[s + 1] - edges[s]

    def _stage_at_step(self, i: int, transitions: Optional[List[int]] = None) -> int:
        ts = transitions if transitions is not None else self.transition_steps
        for s, k in enumerate(ts):
            if i < k:
                return s
        return self.S - 1

    # ------------------------------------------------------------------
    # Transition: spectral expand stage s → s+1 (random, multi-stage)
    # ------------------------------------------------------------------

    def _expand_one_stage(
        self,
        x_src: torch.Tensor,    # (1, C, F, H_src, W_src)
        t_i: float,
        src_stage: int,
        tgt_stage: int,
    ) -> torch.Tensor:
        """Spectrally expand a single transition: src_stage → tgt_stage.

        Random mode only; new high-freq slots filled with t_i*randn.
        """
        assert tgt_stage == src_stage + 1, "only adjacent transitions"
        H_tgt = self.latent_shapes[tgt_stage][2]
        W_tgt = self.latent_shapes[tgt_stage][3]

        x = x_src.squeeze(0)
        x_hi_frames = []
        for f in range(self.F_lat):
            x_f = x[:, f, :, :]
            frame_seed = (
                self.seed * 1_000_003
                + 77 * f
                + int(t_i * 100_000)
                + 113 * src_stage  # disambiguate per transition
            )
            gen_f = torch.Generator(device=self.device).manual_seed(frame_seed)
            x_f_hi = spectral_noise_expand(
                x_f.float(), H_tgt, W_tgt, t_i, generator=gen_f
            )
            x_hi_frames.append(x_f_hi)
        x_hi = torch.stack(x_hi_frames, dim=1)

        if self.transition_align == "variance":
            sigma_expected = math.sqrt(1.0 - 2.0 * t_i + 2.0 * t_i ** 2)
            current_std = x_hi.std().item()
            if current_std > 1e-8:
                x_hi = x_hi * (sigma_expected / current_std)
        elif self.transition_align == "timestep":
            from timestep_alignment import align as ts_align
            H_src = self.latent_shapes[src_stage][2]
            W_src = self.latent_shapes[src_stage][3]
            r = math.sqrt((H_tgt * W_tgt) / (H_src * W_src))
            _, kappa = ts_align(t_i, r)
            x_hi = x_hi * kappa

        return x_hi.unsqueeze(0).to(x_src.dtype)

    # ------------------------------------------------------------------
    # DLC1 pure (Method C): (1-t)*x_hat_0^H + t*eps — any S, no codebook atom
    # ------------------------------------------------------------------

    def _transition_dlc1_pure(
        self,
        x_src: torch.Tensor,
        t_i: float,
        src_stage: int,
        tgt_stage: int,
        model_fn,
    ) -> torch.Tensor:
        """Transition with DLC1 Method C high-freq fill.

        x_bar  = spectral_expand(x_src) with t*eps in high freq (random mode)
        x_hat_0 = x_bar - t_i * model(x_bar, t_i)
        x_new  = x_bar + (1 - t_i) * x_hat_0^H

        Since x_bar already contains t*eps in its high-freq band (and zero
        low-freq contribution from x_hat_0), adding (1-t)*x_hat_0^H to x_bar
        yields the target form (1-t)*x_hat_0^H + t*eps in the high-freq band
        while leaving the low-freq band untouched.
        """
        from dct_utils import dct_2d, idct_2d

        x_bar = self._expand_one_stage(x_src, t_i, src_stage, tgt_stage)
        u_bar = model_fn(x_bar, t_i)
        x_hat_0 = x_bar - t_i * u_bar

        H_src = self.latent_shapes[src_stage][2]
        W_src = self.latent_shapes[src_stage][3]

        x_b = x_bar.squeeze(0)
        x_hat = x_hat_0.squeeze(0)
        out_frames = []
        for f in range(self.F_lat):
            x_b_f = x_b[:, f, :, :].float()
            x_hat_f = x_hat[:, f, :, :].float()

            x_hat_dct = dct_2d(x_hat_f)
            x_hat_hi_dct = x_hat_dct.clone()
            x_hat_hi_dct[..., :H_src, :W_src] = 0.0
            x_hat_hi = idct_2d(x_hat_hi_dct)

            x_new = x_b_f + (1.0 - t_i) * x_hat_hi
            out_frames.append(x_new)

        x_new_stack = torch.stack(out_frames, dim=1).unsqueeze(0).to(x_src.dtype)
        return x_new_stack

    # ------------------------------------------------------------------
    # Oracle (encoder upper bound): (1-t)*x_0^H + t*eps — any S
    # ------------------------------------------------------------------

    def _transition_oracle(
        self,
        x_src: torch.Tensor,
        t_i: float,
        src_stage: int,
        tgt_stage: int,
        x0_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Oracle high-freq fill with ground-truth x_0 at target stage.

        x_bar = spectral_expand(x_src) with t*eps high freq (random)
        x_new = x_bar + (1 - t_i) * x_0^H  (high-freq band of clean target)

        Encoder-only: decoder cannot reproduce. This measures the ceiling
        achievable if the high-freq prior at transition were perfect.
        """
        from dct_utils import dct_2d, idct_2d

        x_bar = self._expand_one_stage(x_src, t_i, src_stage, tgt_stage)

        H_src = self.latent_shapes[src_stage][2]
        W_src = self.latent_shapes[src_stage][3]

        x_b = x_bar.squeeze(0)
        x0 = x0_tgt.squeeze(0)
        out_frames = []
        for f in range(self.F_lat):
            x_b_f = x_b[:, f, :, :].float()
            x0_f = x0[:, f, :, :].float()

            x0_dct = dct_2d(x0_f)
            x0_hi_dct = x0_dct.clone()
            x0_hi_dct[..., :H_src, :W_src] = 0.0
            x0_hi = idct_2d(x0_hi_dct)

            x_new = x_b_f + (1.0 - t_i) * x0_hi
            out_frames.append(x_new)

        x_new_stack = torch.stack(out_frames, dim=1).unsqueeze(0).to(x_src.dtype)
        return x_new_stack

    # ------------------------------------------------------------------
    # DLC1 codebook: target-aware quantized prior + t*eps — any S
    # ------------------------------------------------------------------

    def _transition_dlc1_codebook(
        self,
        x_src: torch.Tensor,
        t_i: float,
        src_stage: int,
        tgt_stage: int,
        x0_tgt: Optional[torch.Tensor],
        encode_mode: bool,
        atoms: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """Transition with codebook-quantized clean prior + t*eps.

        x_bar  = spectral_expand(x_src) — t·ε in new high freq
        Encoder:
          z*_H = codebook atom selected to approximate x_0^{H_i}
          transmit (idx, sgn) per frame
        Decoder:
          z*_H = reconstruct(idx, sgn)
        x_new = x_bar + (1 - t_i) · z*_H

        Bits per transition: F_lat · M · (log2(K_tgt) + 1).
        """
        from dct_utils import dct_2d, idct_2d

        x_bar = self._expand_one_stage(x_src, t_i, src_stage, tgt_stage)

        H_src = self.latent_shapes[src_stage][2]
        W_src = self.latent_shapes[src_stage][3]
        cb_tgt = self.codebooks[tgt_stage]

        # Per-transition unique step index to avoid codebook noise collision
        transition_step_idx = self.num_sde_steps + src_stage

        x_b = x_bar.squeeze(0)
        out_frames = []
        extra_bits: Optional[List[Tuple]] = [] if encode_mode else None

        for f in range(self.F_lat):
            x_b_f = x_b[:, f, :, :].float()

            if encode_mode:
                assert x0_tgt is not None
                x0_f = x0_tgt.squeeze(0)[:, f, :, :].float()
                x0_dct = dct_2d(x0_f)
                x0_hi_dct = x0_dct.clone()
                x0_hi_dct[..., :H_src, :W_src] = 0.0
                x0_hi = idct_2d(x0_hi_dct)

                idx, sgn, z_f = cb_tgt.select_atoms(
                    x0_hi, transition_step_idx, f,
                )
                assert extra_bits is not None
                extra_bits.append((idx, sgn))
            else:
                assert atoms is not None
                idx, sgn = atoms[f]
                z_f = cb_tgt.reconstruct(idx, sgn, transition_step_idx, f)

            z_dct = dct_2d(z_f.float())
            z_hi_dct = z_dct.clone()
            z_hi_dct[..., :H_src, :W_src] = 0.0
            z_hi = idct_2d(z_hi_dct)

            x_new = x_b_f + (1.0 - t_i) * z_hi
            out_frames.append(x_new)

        x_new_stack = torch.stack(out_frames, dim=1).unsqueeze(0).to(x_src.dtype)
        return x_new_stack, extra_bits

    # ------------------------------------------------------------------
    # DLC2 sub-band: structured low-rank high-freq codebook — any S
    # ------------------------------------------------------------------

    def _transition_dlc2_subband(
        self,
        x_src: torch.Tensor,
        t_i: float,
        src_stage: int,
        tgt_stage: int,
        x0_tgt: Optional[torch.Tensor],
        encode_mode: bool,
        atoms: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """Transition with DLC2 sub-band low-rank codebook prior + t·ε.

        Codebook atoms live in a low-rank sub-band of the DCT high-freq
        region. Residual is projected to sub-band before selection; combined
        atom is iDCTed (lives entirely in sub-band by construction).

        x_new = x_bar + (1 - t_i) · z*_subband  (low-freq band unchanged)
        """
        from dct_utils import dct_2d, idct_2d

        x_bar = self._expand_one_stage(x_src, t_i, src_stage, tgt_stage)
        cb_dlc2 = self.dlc2_codebooks[src_stage]
        transition_step_idx = self.num_sde_steps + src_stage

        x_b = x_bar.squeeze(0)
        out_frames = []
        extra_bits: Optional[List[Tuple]] = [] if encode_mode else None

        for f in range(self.F_lat):
            x_b_f = x_b[:, f, :, :].float()

            if encode_mode:
                assert x0_tgt is not None
                x0_f = x0_tgt.squeeze(0)[:, f, :, :].float()
                # Full DCT of target — codebook will project to sub-band internally
                x0_dct = dct_2d(x0_f)
                idx, sgn, z_dct = cb_dlc2.select_atoms(
                    x0_dct, transition_step_idx, f,
                )
                assert extra_bits is not None
                extra_bits.append((idx, sgn))
            else:
                assert atoms is not None
                idx, sgn = atoms[f]
                z_dct = cb_dlc2.reconstruct(idx, sgn, transition_step_idx, f)

            # Spatial atom (high-freq sub-band content only by construction)
            z_spatial = idct_2d(z_dct)
            x_new = x_b_f + (1.0 - t_i) * z_spatial
            out_frames.append(x_new)

        x_new_stack = torch.stack(out_frames, dim=1).unsqueeze(0).to(x_src.dtype)
        return x_new_stack, extra_bits

    # ------------------------------------------------------------------
    # Transition: spectral expand low-res → full-res (S=2 special modes)
    # ------------------------------------------------------------------

    def _expand_transition(
        self,
        x_low: torch.Tensor,
        t_i: float,
        x0_full: Optional[torch.Tensor] = None,
        encode_mode: bool = True,
        codebook_atoms: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """S=2 oracle / codebook expansion. Kept for backwards compatibility."""
        x = x_low.squeeze(0)
        x_hi_frames = []
        extra_bits: Optional[List[Tuple]] = (
            [] if (self.high_freq_mode == "codebook" and encode_mode) else None
        )

        for f in range(self.F_lat):
            x_f = x[:, f, :, :]
            frame_seed = self.seed * 1_000_003 + 77 * f + int(t_i * 100_000)
            gen_f = torch.Generator(device=self.device).manual_seed(frame_seed)

            if self.high_freq_mode == "random":
                x_f_hi = spectral_noise_expand(
                    x_f.float(), self.H_hi, self.W_hi, t_i, generator=gen_f
                )

            elif self.high_freq_mode == "oracle":
                if not encode_mode:
                    x_f_hi = spectral_noise_expand(
                        x_f.float(), self.H_hi, self.W_hi, t_i, generator=gen_f
                    )
                else:
                    assert x0_full is not None
                    x0_f = x0_full.squeeze(0)[:, f, :, :]
                    x_f_lo_exp = spectral_noise_expand(
                        x_f.float(), self.H_hi, self.W_hi, 0.0
                    )
                    from dct_utils import dct_2d, idct_2d
                    x0_dct = dct_2d(x0_f.float())
                    x0_hi_dct = x0_dct.clone()
                    x0_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
                    x0_hi = idct_2d(x0_hi_dct)
                    eps = torch.randn_like(x0_hi, generator=gen_f)
                    x_f_hi = x_f_lo_exp + (1.0 - t_i) * x0_hi + t_i * eps

            elif self.high_freq_mode == "codebook":
                from dct_utils import dct_2d, idct_2d
                x_f_lo_exp = spectral_noise_expand(
                    x_f.float(), self.H_hi, self.W_hi, 0.0
                )
                if encode_mode and x0_full is not None:
                    x0_f = x0_full.squeeze(0)[:, f, :, :]
                    x0_dct = dct_2d(x0_f.float())
                    x0_hi_dct = x0_dct.clone()
                    x0_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
                    x0_hi = idct_2d(x0_hi_dct)

                    idx, sgn, z_f = self.cb_full.select_atoms(
                        x0_hi, self.trans_sde_idx(), f
                    )
                    assert extra_bits is not None
                    extra_bits.append((idx, sgn))

                    z_dct = dct_2d(z_f.float())
                    z_hi_dct = z_dct.clone()
                    z_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
                    z_hi = idct_2d(z_hi_dct)
                    x_f_hi = x_f_lo_exp + t_i * z_hi
                else:
                    assert codebook_atoms is not None
                    idx, sgn = codebook_atoms[f]
                    z_f = self.cb_full.reconstruct(idx, sgn, self.trans_sde_idx(), f)
                    z_dct = dct_2d(z_f.float())
                    z_hi_dct = z_dct.clone()
                    z_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
                    z_hi = idct_2d(z_hi_dct)
                    x_f_hi = x_f_lo_exp + t_i * z_hi
            else:
                raise ValueError(f"Unknown high_freq_mode: {self.high_freq_mode!r}")

            x_hi_frames.append(x_f_hi)

        x_hi = torch.stack(x_hi_frames, dim=1)

        if self.transition_align == "variance":
            sigma_expected = math.sqrt(1.0 - 2.0 * t_i + 2.0 * t_i ** 2)
            current_std = x_hi.std().item()
            if current_std > 1e-8:
                x_hi = x_hi * (sigma_expected / current_std)
        elif self.transition_align == "timestep":
            from timestep_alignment import align as ts_align
            r = math.sqrt((self.H_hi * self.W_hi) / (self.H_lo * self.W_lo))
            _, kappa = ts_align(t_i, r)
            x_hi = x_hi * kappa

        x_hi = x_hi.unsqueeze(0).to(x_low.dtype)
        return x_hi, extra_bits

    def trans_sde_idx(self) -> int:
        return self._trans_sde_idx

    # ------------------------------------------------------------------
    # Prior+Codebook transition (S=2, V1+DLC1 cross)
    # ------------------------------------------------------------------

    def _spectral_expand_bulk(self, x_low: torch.Tensor, t_i: float) -> torch.Tensor:
        x = x_low.squeeze(0)
        x_hi_frames = []
        for f in range(self.F_lat):
            x_f = x[:, f, :, :]
            frame_seed = self.seed * 1_000_003 + 77 * f + int(t_i * 100_000)
            gen_f = torch.Generator(device=self.device).manual_seed(frame_seed)
            x_f_hi = spectral_noise_expand(
                x_f.float(), self.H_hi, self.W_hi, t_i, generator=gen_f
            )
            x_hi_frames.append(x_f_hi)
        x_hi = torch.stack(x_hi_frames, dim=1)
        if self.transition_align == "variance":
            sigma_expected = math.sqrt(1.0 - 2.0 * t_i + 2.0 * t_i ** 2)
            current_std = x_hi.std().item()
            if current_std > 1e-8:
                x_hi = x_hi * (sigma_expected / current_std)
        elif self.transition_align == "timestep":
            from timestep_alignment import align as ts_align
            r = math.sqrt((self.H_hi * self.W_hi) / (self.H_lo * self.W_lo))
            _, kappa = ts_align(t_i, r)
            x_hi = x_hi * kappa
        return x_hi.unsqueeze(0).to(x_low.dtype)

    def _transition_prior_codebook(
        self,
        x_bar: torch.Tensor,
        x_hat_0_full: torch.Tensor,
        t_i: float,
        x0_full: Optional[torch.Tensor] = None,
        encode_mode: bool = True,
        transition_atoms: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        from dct_utils import dct_2d, idct_2d
        extra_bits: Optional[List[Tuple]] = [] if encode_mode else None
        x_b = x_bar.squeeze(0)
        x_hat = x_hat_0_full.squeeze(0)
        out_frames = []
        for f in range(self.F_lat):
            x_b_f = x_b[:, f, :, :].float()
            x_hat_f = x_hat[:, f, :, :].float()

            x_b_dct = dct_2d(x_b_f)
            x_b_lo_dct = torch.zeros_like(x_b_dct)
            x_b_lo_dct[..., :self.H_lo, :self.W_lo] = (
                x_b_dct[..., :self.H_lo, :self.W_lo]
            )
            x_lo_part = idct_2d(x_b_lo_dct)

            x_hat_dct = dct_2d(x_hat_f)
            x_hat_hi_dct = x_hat_dct.clone()
            x_hat_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
            x_hat_hi = idct_2d(x_hat_hi_dct)

            if encode_mode:
                assert x0_full is not None
                x0_f = x0_full.squeeze(0)[:, f, :, :].float()
                x0_dct = dct_2d(x0_f)
                x0_hi_dct = x0_dct.clone()
                x0_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
                x0_hi = idct_2d(x0_hi_dct)
                r_hi = x0_hi - x_hat_hi
                idx, sgn, z_f = self.cb_full.select_atoms(
                    r_hi, self.trans_sde_idx(), f
                )
                assert extra_bits is not None
                extra_bits.append((idx, sgn))
            else:
                assert transition_atoms is not None
                idx, sgn = transition_atoms[f]
                z_f = self.cb_full.reconstruct(idx, sgn, self.trans_sde_idx(), f)

            z_dct = dct_2d(z_f.float())
            z_hi_dct = z_dct.clone()
            z_hi_dct[..., :self.H_lo, :self.W_lo] = 0.0
            z_hi = idct_2d(z_hi_dct)

            x_f_new = x_lo_part + (1.0 - t_i) * x_hat_hi + t_i * z_hi
            out_frames.append(x_f_new)
        x_new = torch.stack(out_frames, dim=1).unsqueeze(0).to(x_bar.dtype)
        return x_new, extra_bits

    # ------------------------------------------------------------------

    def _model_fn(self, embeds: dict, i2v_cond_per_stage=None):
        """Build model callable. If i2v_cond_per_stage is provided (list of S
        conditioning dicts), the appropriate stage's conditioning is selected
        based on x_t shape (matched to self.latent_shapes)."""
        def fn(x_t, t):
            i2v_cond = None
            if i2v_cond_per_stage is not None:
                # Match by latent spatial shape (H_lat, W_lat)
                target_hw = x_t.shape[-2:]
                for s, shp in enumerate(self.latent_shapes):
                    if shp[2] == target_hw[0] and shp[3] == target_hw[1]:
                        i2v_cond = i2v_cond_per_stage[s]
                        break
            return self.model.predict_velocity_cfg(
                x_t, t, embeds, self.guidance_scale, i2v_cond=i2v_cond,
            )
        return fn

    def _do_transition(
        self,
        x_t: torch.Tensor,
        t_curr: float,
        s_idx: int,
        x0_per_stage: Optional[List[torch.Tensor]],
        model_fn,
        encode_mode: bool,
        codebook_atoms: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple]]]:
        """Dispatch transition s_idx → s_idx+1.

        Supports any S for random / dlc1_pure / oracle / dlc1_codebook.
        Routes S=2-only codebook and prior_codebook to legacy paths.

        Parameters
        ----------
        x0_per_stage : list of clean VAE-encoded latents per stage (encoder
                       only). None for decode. Used by oracle / dlc1_codebook
                       (need x0_per_stage[s_idx+1]) and S=2 codebook variants
                       (need x0_per_stage[-1]).
        """
        if self.high_freq_mode == "dlc1_pure":
            x_new = self._transition_dlc1_pure(
                x_t, t_curr, s_idx, s_idx + 1, model_fn,
            )
            return x_new, None

        if self.high_freq_mode == "oracle":
            # Joint oracle: both encode and decode use x_0^H. Decoder reads
            # x0_per_stage from self._oracle_x0_per_stage which encode() stored.
            # This is a ceiling-only test (decoder cannot have x_0 in practice).
            x0_list = x0_per_stage if x0_per_stage is not None else getattr(
                self, "_oracle_x0_per_stage", None
            )
            if x0_list is None:
                x_new = self._expand_one_stage(x_t, t_curr, s_idx, s_idx + 1)
            else:
                x_new = self._transition_oracle(
                    x_t, t_curr, s_idx, s_idx + 1,
                    x0_tgt=x0_list[s_idx + 1],
                )
            return x_new, None

        if self.high_freq_mode == "dlc1_codebook":
            x0_tgt = (
                x0_per_stage[s_idx + 1]
                if (encode_mode and x0_per_stage is not None)
                else None
            )
            return self._transition_dlc1_codebook(
                x_t, t_curr, s_idx, s_idx + 1,
                x0_tgt=x0_tgt, encode_mode=encode_mode,
                atoms=codebook_atoms,
            )

        if self.high_freq_mode == "dlc2_subband":
            x0_tgt = (
                x0_per_stage[s_idx + 1]
                if (encode_mode and x0_per_stage is not None)
                else None
            )
            return self._transition_dlc2_subband(
                x_t, t_curr, s_idx, s_idx + 1,
                x0_tgt=x0_tgt, encode_mode=encode_mode,
                atoms=codebook_atoms,
            )

        if self.S > 2 or self.high_freq_mode == "random":
            x_new = self._expand_one_stage(x_t, t_curr, s_idx, s_idx + 1)
            return x_new, None

        # S=2 special modes (codebook / prior_codebook)
        x0_top = x0_per_stage[-1] if x0_per_stage is not None else None
        if self.high_freq_mode == "prior_codebook":
            x_bar = self._spectral_expand_bulk(x_t, t_curr)
            u_bar = model_fn(x_bar, t_curr)
            x_hat_0_full = x_bar - t_curr * u_bar
            return self._transition_prior_codebook(
                x_bar, x_hat_0_full, t_curr,
                x0_full=x0_top, encode_mode=encode_mode,
                transition_atoms=codebook_atoms,
            )
        else:
            return self._expand_transition(
                x_t, t_curr,
                x0_full=x0_top, encode_mode=encode_mode,
                codebook_atoms=codebook_atoms,
            )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        frames: List[Image.Image],
        prompt: str = "",
        precomputed_i2v_cond_per_stage: Optional[List[dict]] = None,
    ) -> Tuple[List[List], List[Optional[List]], torch.Tensor]:
        """Encode video frames to MR-Turbo-DDCM bitstream.

        Returns
        -------
        step_data_per_stage : list of S lists; entry s is the per-step bitstream
                              for stage s (length _steps_in_stage(s)).
        transition_atoms_per_transition : list of length S-1; entry t holds
                              [F_lat]=(idx,sgn) for transitions that transmit
                              atoms (codebook / prior_codebook / dlc1_codebook),
                              or None for modes that don't (random / dlc1_pure /
                              oracle).
        x_final             : final latent (1, C, F, H_top, W_top).
        """
        embeds = self.model.encode_prompt(prompt)
        model_fn = self._model_fn(embeds, i2v_cond_per_stage=precomputed_i2v_cond_per_stage)

        # VAE encode at each stage's resolution
        x0_per_stage = [
            self.model.encode_video(frames, h, w) for (h, w) in self.resolutions
        ]
        # Optional: override lower-stage targets with DCT-low projection of
        # full-res latent. Tests whether VAE-mismatch (x0_low vs T_L(x0_full))
        # is the source of V1's structural ceiling. Codebook will then drive
        # the low-res trajectory toward T_L(x0_full) instead of VAE_low.
        if self.low_target_from_full and self.S > 1:
            from dct_utils import dct_2d, idct_2d
            x0_full = x0_per_stage[-1]
            for s in range(self.S - 1):
                Hs, Ws = self.latent_shapes[s][2], self.latent_shapes[s][3]
                B, C, F, Hf, Wf = x0_full.shape
                xf = x0_full.reshape(-1, Hf, Wf).float()
                xf_dct_trunc = dct_2d(xf)[:, :Hs, :Ws].contiguous()
                x0_low_proj = idct_2d(xf_dct_trunc).reshape(B, C, F, Hs, Ws).to(x0_full.dtype)
                x0_vae_low = x0_per_stage[s]
                raw_norm = x0_low_proj.float().norm().item()
                vae_norm = x0_vae_low.float().norm().item()
                raw_delta = (
                    (x0_vae_low.float() - x0_low_proj.float()).norm()
                    / (x0_low_proj.float().norm() + 1e-8)
                ).item()
                cos = torch.nn.functional.cosine_similarity(
                    x0_vae_low.float().reshape(1, -1),
                    x0_low_proj.float().reshape(1, -1),
                    dim=1,
                ).item()
                if self.low_target_align == "std":
                    src_std = x0_low_proj.float().std()
                    tgt_std = x0_vae_low.float().std()
                    if src_std > 1e-8:
                        x0_low_proj = (x0_low_proj.float() * (tgt_std / src_std)).to(x0_full.dtype)
                elif self.low_target_align == "norm":
                    src_norm = x0_low_proj.float().norm()
                    tgt_norm = x0_vae_low.float().norm()
                    if src_norm > 1e-8:
                        x0_low_proj = (x0_low_proj.float() * (tgt_norm / src_norm)).to(x0_full.dtype)
                elif self.low_target_align != "none":
                    raise ValueError(f"unknown low_target_align={self.low_target_align!r}")
                aligned_delta = (
                    (x0_vae_low.float() - x0_low_proj.float()).norm()
                    / (x0_low_proj.float().norm() + 1e-8)
                ).item()
                x0_per_stage[s] = x0_low_proj
                print(
                    f"  [low_target_from_full] stage {s}: "
                    f"||VAE_low||={vae_norm:.2f}, ||T_L(full)||={raw_norm:.2f}, "
                    f"raw_delta={raw_delta:.3f}, cos={cos:.3f}, "
                    f"align={self.low_target_align}, aligned_delta={aligned_delta:.3f}"
                )
            print(f"  [low_target_from_full] overrode x0 for stages 0..{self.S-2} "
                  f"with T_L(x0_full)")
        # Stash for joint-oracle decode (ceiling test only; no bit transmission)
        self._oracle_x0_per_stage = x0_per_stage

        # Initial noise at smallest stage
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shapes[0], generator=gen).to(self.device)

        step_data_per_stage: List[List] = [[] for _ in range(self.S)]
        transition_atoms_per_transition: List[Optional[List]] = (
            [None] * (self.S - 1)
        )
        sde_idx = 0

        current_transitions = list(self.transition_steps)

        # x0_cache skip bookkeeping (per-stage alternation, reset at transitions)
        cached_x0 = None
        stage_start_i = 0
        # adaptive_x0_cache: 1-bit-per-SDE-step decision recorded by encoder.
        skip_mask = [False] * self.num_steps if self.model_call_skip == "adaptive_x0_cache" else None
        consec_skips = 0  # running count of consecutive skips (adaptive cap)

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            # Transitions
            if i in current_transitions:
                s_idx = current_transitions.index(i)
                x_t, atoms = self._do_transition(
                    x_t, t_curr, s_idx, x0_per_stage, model_fn, encode_mode=True,
                )
                transition_atoms_per_transition[s_idx] = atoms
                # Resolution just changed → invalidate cache and anchor stage.
                # transition_warmup shifts stage_start_i by +1 so the first
                # post-transition step (i+1) is also forced to run Wan.
                cached_x0 = None
                stage_start_i = i + 1 if self.transition_warmup else i

            # Skip decision (never on transition, never in DDIM tail, only in
            # stages listed in self.skip_stages, and only when cache is valid).
            stage_for_skip = self._stage_at_step(i, current_transitions)
            skip_eligible = (
                self.model_call_skip != "none"
                and i < self.num_sde_steps
                and i not in current_transitions
                and stage_for_skip in self.skip_stages
                and cached_x0 is not None
                and cached_x0.shape == x_t.shape
                and t_curr > 1e-6
            )
            if self.model_call_skip == "adaptive_x0_cache":
                # Encoder always runs Wan, decides skip from actual error.
                u_teacher = model_fn(x_t, t_curr)
                x0_teacher = (x_t - t_curr * u_teacher).detach()
                if skip_eligible:
                    u_fast = (x_t - cached_x0) / t_curr
                    err = ((u_teacher - u_fast).float().norm()
                           / u_teacher.float().norm().clamp_min(1e-8)).item()
                    skip_now = err < self.adaptive_skip_threshold
                    # Per-stage cap on consecutive skips.
                    cap_here = self.max_consec_per_stage[stage_for_skip]
                    if skip_now and consec_skips >= cap_here:
                        skip_now = False
                else:
                    skip_now = False
                skip_mask[i] = bool(skip_now)
                if skip_now:
                    u_t = u_fast  # cached_x0 anchor unchanged
                    consec_skips += 1
                else:
                    u_t = u_teacher
                    cached_x0 = x0_teacher
                    consec_skips = 0
            else:
                should_skip = (
                    self.model_call_skip == "x0_cache"
                    and skip_eligible
                    and (i - stage_start_i) % self.model_call_period != 0
                )
                if should_skip:
                    u_t = (x_t - cached_x0) / t_curr
                else:
                    u_t = model_fn(x_t, t_curr)
                    cached_x0 = (x_t - t_curr * u_t).detach()

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break

            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            stage = self._stage_at_step(i, current_transitions)
            x0_ref = x0_per_stage[stage]
            cb = self.codebooks[stage]

            x0_hat = x_t - t_curr * u_t
            residual = (x0_ref - x0_hat).squeeze(0)

            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            frame_entries = []
            noise_frames = []
            for f in range(self.F_lat):
                r_f = residual[:, f, :, :]
                idx, sgn, z_f = cb.select_atoms(r_f, sde_idx, f)
                frame_entries.append((idx, sgn))
                noise_frames.append(z_f)

            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d

            step_data_per_stage[stage].append(frame_entries)
            sde_idx += 1

            if (i + 1) % 5 == 0 or i == 0:
                mse = (x0_ref - x0_hat).pow(2).mean().item()
                print(
                    f"  Encode step {i+1}/{self.num_steps} [s{stage}]: "
                    f"residual_MSE={mse:.4f}, noise_coeff={noise_coeff:.4f}"
                )

        # Stash adaptive skip mask + report skip ratio per stage
        if skip_mask is not None:
            self._encode_skip_mask = list(skip_mask)
            # Per-stage skip ratios (over SDE steps only)
            for s in range(self.S):
                edges = [0] + current_transitions + [self.num_sde_steps]
                lo, hi = edges[s], edges[s + 1]
                if hi > lo:
                    n = hi - lo
                    skipped = sum(1 for k in range(lo, hi) if skip_mask[k])
                    print(f"  [adaptive_skip] stage {s} steps {lo}-{hi-1}: "
                          f"{skipped}/{n} skipped ({100.0*skipped/n:.0f}%)")
        else:
            self._encode_skip_mask = None

        return step_data_per_stage, transition_atoms_per_transition, x_t

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        step_data_per_stage: List[List],
        transition_atoms_per_transition: Optional[List[Optional[List]]] = None,
        prompt: str = "",
        latent_correction: Optional["torch.Tensor"] = None,
        return_final_latent: bool = False,
        precomputed_i2v_cond_per_stage: Optional[List[dict]] = None,
    ):
        embeds = self.model.encode_prompt(prompt)
        model_fn = self._model_fn(embeds, i2v_cond_per_stage=precomputed_i2v_cond_per_stage)

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shapes[0], generator=gen).to(self.device)

        per_stage_idx = [0] * self.S
        sde_idx = 0

        current_transitions = list(self.transition_steps)
        cached_x0 = None
        stage_start_i = 0

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            if i in current_transitions:
                s_idx = current_transitions.index(i)
                atoms_at_t = (
                    transition_atoms_per_transition[s_idx]
                    if transition_atoms_per_transition is not None
                    else None
                )
                x_t, _ = self._do_transition(
                    x_t, t_curr, s_idx, None, model_fn,
                    encode_mode=False,
                    codebook_atoms=atoms_at_t,
                )
                cached_x0 = None
                stage_start_i = i + 1 if self.transition_warmup else i

            stage_for_skip = self._stage_at_step(i, current_transitions)
            skip_eligible = (
                self.model_call_skip != "none"
                and i < self.num_sde_steps
                and i not in current_transitions
                and stage_for_skip in self.skip_stages
                and cached_x0 is not None
                and cached_x0.shape == x_t.shape
                and t_curr > 1e-6
            )
            if self.model_call_skip == "adaptive_x0_cache":
                # Read encoder-transmitted decision; ignore mask on ineligible
                # steps (defense-in-depth — encoder shouldn't have flagged them).
                mask = self._encode_skip_mask or []
                skip_now = skip_eligible and i < len(mask) and bool(mask[i])
                if skip_now:
                    u_t = (x_t - cached_x0) / t_curr
                else:
                    u_t = model_fn(x_t, t_curr)
                    cached_x0 = (x_t - t_curr * u_t).detach()
            else:
                should_skip = (
                    self.model_call_skip == "x0_cache"
                    and skip_eligible
                    and (i - stage_start_i) % self.model_call_period != 0
                )
                if should_skip:
                    u_t = (x_t - cached_x0) / t_curr
                else:
                    u_t = model_fn(x_t, t_curr)
                    cached_x0 = (x_t - t_curr * u_t).detach()

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break

            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            stage = self._stage_at_step(i, current_transitions)
            cb = self.codebooks[stage]

            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            entries = step_data_per_stage[stage][per_stage_idx[stage]]
            noise_frames = []
            for f in range(self.F_lat):
                idx, sgn = entries[f]
                z_f = cb.reconstruct(idx, sgn, sde_idx, f)
                noise_frames.append(z_f)
            per_stage_idx[stage] += 1

            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)
            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d

            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{self.num_steps} [s{stage}]")

        # Apply boundary-link / I2V-tail latent correction before VAE decode.
        if latent_correction is not None:
            x_t = x_t + latent_correction.to(x_t.device, dtype=x_t.dtype)

        final_latent = x_t.detach().clone() if return_final_latent else None
        frames = self.model.decode_latent(x_t)
        if return_final_latent:
            return frames, final_latent
        return frames
