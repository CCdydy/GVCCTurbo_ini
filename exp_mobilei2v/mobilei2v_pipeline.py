"""mobilei2v_pipeline.py — Turbo-DDCM compression on MobileI2V.

Differences from TurboDDCMWanPipeline:
  1. First latent frame is overwritten with ref_latent at every SDE step
     (matches MobileI2V's I2V mechanism in flow_euler_sampler.py:89).
  2. DDCM atoms are NOT selected for frame 0 — first-frame slot is conditioning,
     so encoding noise for it is wasted bits. Frame 0 noise = 0.
  3. Uses MobileI2V's flow_shift=3.0 via the shared shifted_timesteps helper.
  4. Smoke-test scope: no x0_cache / no spectral / no MR — just baseline SDE
     + first-frame overwrite + optional tail_residual.
"""
import sys
import time
import torch
from typing import List, Tuple
from PIL import Image

# turbogvcc shared DDCM utilities
_PROJECT_ROOT = "/home/rog/Desktop/gvcc_turbo/turbogvcc"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from sde_rf_wan.sde_convert import (
    velocity_to_score, diffusion_coeff, sde_drift, shifted_timesteps,
)
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook


class MobileI2VGVCCPipeline:
    def __init__(
        self,
        model,                              # MobileI2VWrapper
        K: int = 16384,
        M: int = 64,
        num_steps: int = 20,
        num_ddim_tail: int = 3,
        guidance_scale: float = 4.5,
        g_scale: float = 3.0,
        seed: int = 42,
        flow_score: float = 2.0,
        vectorized_atom_gen: bool = True,
        model_call_skip: str = "none",      # "none" | "x0_cache" | "u_cache"
        model_call_period: int = 2,
        model_call_skip_until: int = -1,    # SDE step idx; -1 = all SDE steps
    ):
        self.model = model
        self.K = K
        self.M = M
        self.num_steps = num_steps
        self.num_ddim_tail = num_ddim_tail
        self.num_sde_steps = num_steps - num_ddim_tail
        self.guidance_scale = guidance_scale
        self.g_scale = g_scale
        self.seed = seed
        self.flow_score_val = flow_score

        self.device = model.device
        self.latent_shape = (model.latent_dim, model.num_latent_frames,
                              model.latent_h, model.latent_w)
        self.frame_shape = (model.latent_dim, model.latent_h, model.latent_w)
        self.num_latent_frames = model.num_latent_frames
        self.num_frames = model.num_pixel_frames
        self.height = model.H
        self.width = model.W

        # Match MobileI2V's FlowMatchEulerDiscreteScheduler(shift=3.0)
        self.timesteps = shifted_timesteps(
            num_steps, shift=model.flow_shift, device=self.device,
        )

        assert model_call_skip in ("none", "u_cache", "x0_cache"), \
            f"unknown skip mode: {model_call_skip}"
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

        # F_active = F-1 (frame 0 is conditioned, no bits)
        self.F_active = self.num_latent_frames - 1
        bits_per_idx = self.codebook.bits_per_index
        bits_per_frame_step = M * (bits_per_idx + 1)
        total_bits = self.num_sde_steps * self.F_active * bits_per_frame_step
        self._total_codebook_bits = total_bits

        total_pixels = self.num_frames * self.height * self.width
        bpp = total_bits / total_pixels
        bitrate_kbps = total_bits / (self.num_frames / 16.0) / 1000.0

        print(f"MobileI2V-GVCC Pipeline:")
        print(f"  Video: {self.num_frames}f @ {self.height}x{self.width}")
        print(f"  Latent: {self.latent_shape}  (F_active={self.F_active})")
        print(f"  K={K} M={M} T_sde={self.num_sde_steps} ddim_tail={num_ddim_tail}")
        print(f"  Codebook: {total_bits} bits ({total_bits/8:.0f}B)  "
              f"BPP={bpp:.6f}  {bitrate_kbps:.2f} kbps")
        if self.model_call_skip != "none":
            print(f"  Model-call skipping: mode={self.model_call_skip}  "
                  f"period={self.model_call_period}  "
                  f"until_step={self.model_call_skip_until}")

        self._gt_latent = None  # set by encode()

    def _select_noise_for_frame(self, residual_f, sde_idx, frame_idx, encode_mode=True):
        """Per-frame atom selection. Returns (idx_list, sgn_list, z)."""
        return self.codebook.select_atoms(residual_f, sde_idx, frame_idx)

    def _model_call_or_cache(self, model_fn, x_t, t_curr, step_idx,
                              cached_u=None, cached_x0=None):
        """Symmetric encode/decode skip helper (port of TurboDDCMWanPipeline)."""
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
            # x0_cache: derive u_t from cached x0 estimate at current x_t
            u_t = (x_t - cached_x0) / t_curr
            return u_t, cached_u, cached_x0
        u_t = model_fn(x_t, t_curr)
        x0_hat = x_t - t_curr * u_t
        return u_t, u_t.detach(), x0_hat.detach()

    @torch.no_grad()
    def encode(self, frames: List[Image.Image], prompt: str = "",
               ref_image: Image.Image = None):
        """Encode 17 GT frames → DDCM bitstream + final latent x_0."""
        assert ref_image is not None, "MobileI2V pipeline requires ref_image"

        embeds = self.model.encode_prompt(prompt)
        i2v_cond = self.model.encode_image(ref_image)
        ref_latent = i2v_cond["ref_latent"]  # (1, C, 1, H, W)

        # GT latent for residual selection
        x0_true = self.model.encode_video(frames)
        self._gt_latent = x0_true.clone()

        # Initial state: deterministic noise, then overwrite frame 0 = ref
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)
        x_t = x_t.to(self.model.weight_dtype)
        x_t[:, :, :1] = ref_latent

        step_data = []
        sde_idx = 0
        cached_u, cached_x0 = None, None

        def _model_fn(_x, _t):
            return self.model.predict_velocity_cfg(
                _x, _t, embeds, self.guidance_scale, i2v_cond,
                flow_score=self.flow_score_val,
            )

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t, cached_u, cached_x0 = self._model_call_or_cache(
                _model_fn, x_t, t_curr, i, cached_u, cached_x0,
            )

            # Final step → ODE to t=0
            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                x_t[:, :, :1] = ref_latent  # keep frame 0 pinned
                break

            # DDIM tail → ODE step, no bits
            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                x_t[:, :, :1] = ref_latent
                continue

            # MMSE x0_hat
            x0_hat = x_t - t_curr * u_t

            # Residual (encoder-only) for atom selection — codebook is fp32
            residual = (x0_true - x0_hat).squeeze(0).float()  # (C, F, H, W)

            # SDE components
            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            # Per-frame atom selection — SKIP frame 0
            frame_entries = []
            noise_frames = [torch.zeros_like(residual[:, 0])]  # frame 0 noise = 0
            for f in range(1, self.num_latent_frames):
                r_f = residual[:, f, :, :]
                idx, sgn, z_f = self._select_noise_for_frame(r_f, sde_idx, f)
                frame_entries.append((idx, sgn))
                noise_frames.append(z_f)

            step_data.append(frame_entries)
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)  # (1,C,F,H,W)
            noise_3d = noise_3d.to(x_t.dtype)

            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
            x_t[:, :, :1] = ref_latent  # pin frame 0 every step
            sde_idx += 1

            if (i + 1) % 5 == 0 or i == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                print(f"  Encode step {i+1}/{self.num_steps}: "
                      f"residual_MSE={mse:.4f}  noise_coeff={noise_coeff:.4f}")

        return step_data, x_t

    @torch.no_grad()
    def decode(self, step_data, prompt: str = "",
               ref_image: Image.Image = None,
               latent_correction: torch.Tensor = None):
        """Replay SDE trajectory → reconstructed pixel frames."""
        assert ref_image is not None
        embeds = self.model.encode_prompt(prompt)
        i2v_cond = self.model.encode_image(ref_image)
        ref_latent = i2v_cond["ref_latent"]

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device)
        x_t = x_t.to(self.model.weight_dtype)
        x_t[:, :, :1] = ref_latent

        sde_idx = 0
        cached_u, cached_x0 = None, None

        def _model_fn(_x, _t):
            return self.model.predict_velocity_cfg(
                _x, _t, embeds, self.guidance_scale, i2v_cond,
                flow_score=self.flow_score_val,
            )

        for i in range(self.num_steps):
            t_curr = self.timesteps[i].item()
            t_next = self.timesteps[i + 1].item()
            delta_t = t_curr - t_next

            u_t, cached_u, cached_x0 = self._model_call_or_cache(
                _model_fn, x_t, t_curr, i, cached_u, cached_x0,
            )

            if t_next < 1e-6:
                x_t = x_t - u_t * delta_t
                x_t[:, :, :1] = ref_latent
                break
            if i >= self.num_sde_steps:
                x_t = x_t - u_t * delta_t
                x_t[:, :, :1] = ref_latent
                continue

            score = velocity_to_score(u_t, x_t, t_curr)
            g_t = diffusion_coeff(t_curr, self.g_scale)
            f_t = sde_drift(u_t, score, g_t)
            noise_coeff = g_t * (delta_t ** 0.5)

            # Reconstruct noise from stored indices+signs, skipping frame 0
            noise_frames = [torch.zeros(self.frame_shape, device=self.device,
                                         dtype=x_t.dtype)]
            entries = step_data[sde_idx]
            for ent_i, f in enumerate(range(1, self.num_latent_frames)):
                idx, sgn = entries[ent_i]
                z_f = self.codebook.reconstruct(idx, sgn, sde_idx, f)
                noise_frames.append(z_f.to(x_t.dtype))
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)

            x_t = x_t - f_t * delta_t + noise_coeff * noise_3d
            x_t[:, :, :1] = ref_latent
            sde_idx += 1

            if (i + 1) % 5 == 0:
                print(f"  Decode step {i+1}/{self.num_steps}")

        if latent_correction is not None:
            x_t = x_t + latent_correction.to(x_t.device, dtype=x_t.dtype)

        frames = self.model.decode_latent(x_t)
        return frames
