"""wan22_fun_pipeline.py — Turbo-DDCM compression on Wan2.2-Fun-5B-InP.

Differences from TurboDDCMWanPipeline:
  1. Both first AND last latent frames are conditioned via FLF2V (frames 0 and F-1).
     F_active = F - 2, no atom bits spent on conditioned frames.
  2. Conditioning passed via `y` tensor (constant across steps) + per-token timestep.
  3. Uses Wan2.2 flow_shift=5.0 via shared shifted_timesteps helper.
"""
import sys
import time
import torch
from typing import List
from PIL import Image

_PROJECT_ROOT = "/home/rog/Desktop/gvcc_turbo/turbogvcc"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from sde_rf_wan.sde_convert import (
    velocity_to_score, diffusion_coeff, sde_drift, shifted_timesteps,
)
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook


class Wan22FunGVCCPipeline:
    def __init__(
        self,
        model,                                # Wan22FunWrapper
        K: int = 16384,
        M: int = 64,
        num_steps: int = 20,
        num_ddim_tail: int = 3,
        guidance_scale: float = 6.0,
        g_scale: float = 3.0,
        num_frames: int = 33,
        height: int = 704,
        width: int = 1280,
        seed: int = 42,
        vectorized_atom_gen: bool = True,
        model_call_skip: str = "none",        # "none" | "u_cache" | "x0_cache"
        model_call_period: int = 2,
        model_call_skip_until: int = -1,
        skip_frame0: bool = True,             # default: pin frame 0, save its atom bits
    ):
        self.model = model
        self.K = K
        self.M = M
        self.num_steps = num_steps
        self.num_ddim_tail = num_ddim_tail
        self.num_sde_steps = num_steps - num_ddim_tail
        self.guidance_scale = guidance_scale
        self.g_scale = g_scale
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.seed = seed

        self.device = model.device
        self.latent_shape = model.get_latent_shape(num_frames, height, width)
        self.frame_shape = model.get_frame_shape(height, width)
        self.num_latent_frames = self.latent_shape[1]

        assert model_call_skip in ("none", "u_cache", "x0_cache")
        assert model_call_period >= 1
        self.model_call_skip = model_call_skip
        self.model_call_period = model_call_period
        self.model_call_skip_until = (
            self.num_sde_steps if model_call_skip_until < 0
            else min(model_call_skip_until, self.num_sde_steps)
        )

        # FlowMatchEulerDiscreteScheduler(shift=5.0) → SDE-equivalent timesteps in [0,1]
        self.timesteps = shifted_timesteps(
            num_steps, shift=model.flow_shift, device=self.device,
        )

        self.codebook = TurboPerFrameCodebook(
            K=K, M=M, frame_shape=self.frame_shape,
            seed=seed, device=self.device,
            vectorized_atom_gen=vectorized_atom_gen,
        )

        # F_active: by default skip frame 0 to save bits (Wan2.2-Fun pins frame 0 via mask).
        # Set self.skip_frame0 = False externally to match original Wan2.1-FLF2V GVCC (atoms for all frames).
        self.skip_frame0 = skip_frame0
        self.F_active = max(self.num_latent_frames - 1, 0) if self.skip_frame0 else self.num_latent_frames
        bits_per_idx = self.codebook.bits_per_index
        bits_per_frame_step = M * (bits_per_idx + 1)
        total_bits = self.num_sde_steps * self.F_active * bits_per_frame_step
        self._total_codebook_bits = total_bits

        total_pixels = num_frames * height * width
        bpp = total_bits / total_pixels
        bitrate_kbps = total_bits / (num_frames / 16.0) / 1000.0

        print(f"Wan2.2-Fun-5B GVCC Pipeline:")
        print(f"  Video: {num_frames}f @ {height}x{width}")
        print(f"  Latent: {self.latent_shape}  (F_active={self.F_active})")
        print(f"  K={K} M={M} T_sde={self.num_sde_steps} ddim_tail={num_ddim_tail}")
        print(f"  Codebook: {total_bits} bits ({total_bits/8:.0f}B)  "
              f"BPP={bpp:.6f}  {bitrate_kbps:.2f} kbps")
        if self.model_call_skip != "none":
            print(f"  Model-call skipping: mode={self.model_call_skip}  "
                  f"period={self.model_call_period}  until_step={self.model_call_skip_until}")

        self._gt_latent = None

    def _model_call_or_cache(self, model_fn, x_t, t_curr, step_idx,
                              cached_u=None, cached_x0=None):
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
                should_skip and cached_x0 is not None
                and cached_x0.shape == x_t.shape and t_curr > 1e-6
            )
        if should_skip:
            if self.model_call_skip == "u_cache":
                return cached_u, cached_u, cached_x0
            u_t = (x_t - cached_x0) / t_curr
            return u_t, cached_u, cached_x0
        u_t = model_fn(x_t, t_curr)
        x0_hat = x_t - t_curr * u_t
        return u_t, u_t.detach(), x0_hat.detach()

    @torch.no_grad()
    def encode(self, frames: List[Image.Image], prompt: str = "",
                first_image: Image.Image = None, last_image: Image.Image = None):
        """Encode video → DDCM bitstream + final latent."""
        assert len(frames) == self.num_frames
        first_image = first_image or frames[0]
        last_image = last_image or frames[-1]

        embeds = self.model.encode_prompt(prompt)
        flf = self.model.encode_first_last_frames(
            first_image, last_image, self.num_frames, self.height, self.width,
        )

        x0_true = self.model.encode_video(frames, self.height, self.width)
        self._gt_latent = x0_true.clone()

        def _model_fn(_x, _t):
            return self.model.predict_velocity_cfg(
                _x, _t, embeds, self.guidance_scale, flf,
            )

        # Initialize x_T per official pipeline:
        #   latents = (1 - pertoken_mask) * masked_video_latents + pertoken_mask * randn
        # After our mask override (mask=0 only at frame 0, mask=1 elsewhere),
        # this effectively only blends frame 0, leaves rest as noise.
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device).to(self.model.weight_dtype)
        m = flf["mask"].to(x_t.dtype)
        x_t = (1.0 - m) * flf["masked_video_latents"] + m * x_t

        step_data = []
        sde_idx = 0
        cached_u, cached_x0 = None, None

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t, cached_u, cached_x0 = self._model_call_or_cache(
                _model_fn, x_t, t_curr, i, cached_u, cached_x0,
            )

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break
            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            x0_hat = x_t - t_curr * u_t
            residual = (x0_true - x0_hat).squeeze(0).float()  # (C, F, H, W)

            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            # Per-frame atom selection — SKIP frames 0 and -1 (conditioned)
            frame_entries = []
            noise_frames = []
            for f in range(self.num_latent_frames):
                if self.skip_frame0 and f == 0:    # frame 0 pinned (no atom bits)
                    noise_frames.append(torch.zeros_like(residual[:, f]))
                    continue
                r_f = residual[:, f, :, :]
                idx, sgn, z_f = self.codebook.select_atoms(r_f, sde_idx, f)
                frame_entries.append((idx, sgn))
                noise_frames.append(z_f)

            step_data.append(frame_entries)
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0).to(x_t.dtype)

            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
            sde_idx += 1

            if (i + 1) % 5 == 0 or i == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                print(f"  Encode step {i+1}/{self.num_steps}: "
                      f"residual_MSE={mse:.4f}  noise_coeff={noise_coeff:.4f}")

        return step_data, x_t

    @torch.no_grad()
    def decode(self, step_data, prompt: str = "",
                first_image: Image.Image = None, last_image: Image.Image = None,
                latent_correction: torch.Tensor = None):
        assert first_image is not None and last_image is not None
        embeds = self.model.encode_prompt(prompt)
        flf = self.model.encode_first_last_frames(
            first_image, last_image, self.num_frames, self.height, self.width,
        )

        def _model_fn(_x, _t):
            return self.model.predict_velocity_cfg(
                _x, _t, embeds, self.guidance_scale, flf,
            )

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device).to(self.model.weight_dtype)
        m = flf["mask"].to(x_t.dtype)
        x_t = (1.0 - m) * flf["masked_video_latents"] + m * x_t

        sde_idx = 0
        cached_u, cached_x0 = None, None
        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t, cached_u, cached_x0 = self._model_call_or_cache(
                _model_fn, x_t, t_curr, i, cached_u, cached_x0,
            )

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                break
            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                continue

            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            entries = step_data[sde_idx]
            noise_frames = []
            entry_idx = 0
            for f in range(self.num_latent_frames):
                if self.skip_frame0 and f == 0:
                    noise_frames.append(torch.zeros(self.frame_shape, device=self.device, dtype=x_t.dtype))
                    continue
                idx, sgn = entries[entry_idx]
                entry_idx += 1
                z_f = self.codebook.reconstruct(idx, sgn, sde_idx, f)
                noise_frames.append(z_f.to(x_t.dtype))
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)

            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{self.num_steps}")

        if latent_correction is not None:
            x_t = x_t + latent_correction.to(x_t.device, dtype=x_t.dtype)

        return self.model.decode_latent(x_t)
