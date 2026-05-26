"""cogvideox_fun_pipeline.py — DDCM compression on CogVideoX-Fun-2b-InP (v-prediction).

Strategy: use the official CogVideoXDDIMScheduler.step() with eta=1.0 and
variance_noise=codebook_atom — this lets us inject codebook noise into the
stochastic part of DDIM without rewriting v-prediction → score math by hand.

Algorithm at each step:
  1. v_pred = transformer(x_t, t, prompt, inpaint_latents)
  2. x0_hat = alpha_t * x_t - sigma_t * v_pred  (for atom selection only)
  3. residual = x0_true - x0_hat
  4. For each non-frame-0 latent slot: select M atoms by |<z, r_f>|
  5. Stack atoms → noise_3d
  6. x_{t-1} = scheduler.step(v_pred, t, x_t, eta=1.0, variance_noise=noise_3d).prev_sample
"""
import sys
import time
import torch
import math
from typing import List
from PIL import Image

_PROJECT_ROOT = "/home/rog/Desktop/gvcc_turbo/turbogvcc"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from sde_rf_wan.turbo_codebook import TurboPerFrameCodebook


class CogVideoXFunGVCCPipeline:
    def __init__(
        self,
        model,                              # CogVideoXFunWrapper
        K: int = 16384,
        M: int = 64,
        num_steps: int = 20,
        guidance_scale: float = 6.0,
        eta: float = 1.0,                    # DDPM-like injection (1.0 = full stochastic)
        num_frames: int = 33,                # 4n+1
        height: int = 480,
        width: int = 720,
        seed: int = 42,
        vectorized_atom_gen: bool = True,
    ):
        self.model = model
        self.K = K
        self.M = M
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.seed = seed

        self.device = model.device
        self.latent_shape = model.get_latent_shape(num_frames, height, width)
        self.frame_shape  = model.get_frame_shape(height, width)
        self.num_latent_frames = self.latent_shape[1]

        # Configure scheduler timesteps
        self.scheduler = model.scheduler
        self.scheduler.set_timesteps(num_steps, device=self.device)
        self.timesteps = self.scheduler.timesteps  # tensor of int timesteps, descending

        # alphas_cumprod for x0_hat reconstruction
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.codebook = TurboPerFrameCodebook(
            K=K, M=M, frame_shape=self.frame_shape,
            seed=seed, device=self.device,
            vectorized_atom_gen=vectorized_atom_gen,
        )

        # F_active: skip frame 0 (first-frame anchor via inpaint). Last frame remains
        # in the atom set since CogVideoX inpaint conditioning is weaker.
        self.F_active = max(self.num_latent_frames - 1, 1)
        bits_per_idx = self.codebook.bits_per_index
        bits_per_frame_step = M * (bits_per_idx + 1)
        total_bits = num_steps * self.F_active * bits_per_frame_step
        self._total_codebook_bits = total_bits

        total_pixels = num_frames * height * width
        bpp = total_bits / total_pixels
        kbps = total_bits / (num_frames / 16.0) / 1000.0

        print(f"CogVideoX-Fun-2b GVCC Pipeline:")
        print(f"  Video: {num_frames}f @ {height}x{width}")
        print(f"  Latent: {self.latent_shape}  (F_active={self.F_active})")
        print(f"  K={K} M={M} steps={num_steps} eta={eta}")
        print(f"  Codebook: {total_bits} bits ({total_bits/8:.0f}B)  "
              f"BPP={bpp:.6f}  {kbps:.2f} kbps")

        self._gt_latent = None

    def _x0_hat_from_v(self, x_t, v, t_int):
        """v-prediction → x0_hat estimate."""
        a = self.alphas_cumprod[t_int]
        sqrt_a = a.sqrt()
        sqrt_1ma = (1 - a).sqrt()
        return sqrt_a * x_t - sqrt_1ma * v

    def _ddim_ddpm_step(self, x_t, v_pred, t_int, noise_3d):
        """Manual DDIM-DDPM mixture step.

        CogVideoXDDIMScheduler ignores eta/variance_noise, so we compute it ourselves.
        Formula (DDIM paper eq 12 + DDPM noise injection):
          x_0_pred = sqrt(a_t) * x_t - sqrt(1-a_t) * v
          eps_pred = sqrt(a_t) * v + sqrt(1-a_t) * x_t
          sigma_t  = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1 - a_t/a_prev)
          dir_xt   = sqrt(1 - a_prev - sigma_t^2) * eps_pred
          x_{t-1}  = sqrt(a_prev) * x_0_pred + dir_xt + sigma_t * noise
        """
        a_t = self.alphas_cumprod[t_int]
        prev_t = t_int - self.scheduler.config.num_train_timesteps // self.num_steps
        if prev_t >= 0:
            a_prev = self.alphas_cumprod[prev_t]
        else:
            a_prev = self.scheduler.final_alpha_cumprod.to(self.device)

        sqrt_a_t   = a_t.sqrt()
        sqrt_1ma_t = (1 - a_t).sqrt()
        sqrt_a_prev   = a_prev.sqrt()
        sqrt_1ma_prev = (1 - a_prev).sqrt()

        x0_pred  = sqrt_a_t * x_t - sqrt_1ma_t * v_pred
        eps_pred = sqrt_a_t * v_pred + sqrt_1ma_t * x_t

        # Be careful at terminal step (a_prev = 1 → 1 - a_t/a_prev could be negative for SNR shift)
        ratio = a_t / a_prev.clamp(min=1e-8)
        sigma2 = (self.eta ** 2) * (1 - a_prev) / (1 - a_t).clamp(min=1e-8) * (1 - ratio).clamp(min=0)
        sigma_t = sigma2.sqrt()

        # Direction term — clamp inside sqrt to avoid neg
        dir_coef_sq = (1 - a_prev - sigma2).clamp(min=0)
        dir_coef = dir_coef_sq.sqrt()

        x_prev = sqrt_a_prev * x0_pred + dir_coef * eps_pred + sigma_t * noise_3d
        return x_prev

    @torch.no_grad()
    def encode(self, frames: List[Image.Image], prompt: str = "",
                first_image: Image.Image = None, last_image: Image.Image = None):
        assert len(frames) == self.num_frames
        first_image = first_image or frames[0]
        last_image  = last_image  or frames[-1]

        embeds = self.model.encode_prompt(prompt)
        flf = self.model.encode_first_last_frames(
            first_image, last_image, self.num_frames, self.height, self.width,
        )

        x0_true = self.model.encode_video(frames, self.height, self.width)
        self._gt_latent = x0_true.clone()

        # Init x_T ~ N(0, I) (deterministic)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device).to(self.model.weight_dtype)

        step_data = []
        for sde_idx, t in enumerate(self.timesteps):
            t_int = int(t.item())

            v_pred = self.model.predict_v_cfg(
                x_t, t_int, embeds, self.guidance_scale, flf,
            )

            # x0_hat for atom selection (encoder-only)
            x0_hat = self._x0_hat_from_v(x_t, v_pred, t_int)
            residual = (x0_true - x0_hat).squeeze(0).float()

            # Per-frame atom selection — skip frame 0 (first-frame anchor)
            frame_entries = []
            noise_frames = []
            for f in range(self.num_latent_frames):
                if f == 0:
                    noise_frames.append(torch.zeros_like(residual[:, f]))
                    continue
                r_f = residual[:, f, :, :]
                idx, sgn, z_f = self.codebook.select_atoms(r_f, sde_idx, f)
                frame_entries.append((idx, sgn))
                noise_frames.append(z_f)

            step_data.append(frame_entries)
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0).to(x_t.dtype)

            # Manual DDIM-DDPM mixture (scheduler ignores eta/variance_noise)
            x_t = self._ddim_ddpm_step(x_t, v_pred, t_int, noise_3d)

            if (sde_idx + 1) % 5 == 0 or sde_idx == 0:
                mse = ((x0_true - x0_hat) ** 2).mean().item()
                print(f"  Encode step {sde_idx+1}/{self.num_steps}: "
                      f"t={t_int}  residual_MSE={mse:.4f}")

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

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        x_t = torch.randn(1, *self.latent_shape, generator=gen).to(self.device).to(self.model.weight_dtype)

        for sde_idx, t in enumerate(self.timesteps):
            t_int = int(t.item())

            v_pred = self.model.predict_v_cfg(
                x_t, t_int, embeds, self.guidance_scale, flf,
            )

            entries = step_data[sde_idx]
            noise_frames = []
            entry_idx = 0
            for f in range(self.num_latent_frames):
                if f == 0:
                    noise_frames.append(torch.zeros(self.frame_shape, device=self.device, dtype=x_t.dtype))
                    continue
                idx, sgn = entries[entry_idx]
                entry_idx += 1
                z_f = self.codebook.reconstruct(idx, sgn, sde_idx, f)
                noise_frames.append(z_f.to(x_t.dtype))
            noise_3d = torch.stack(noise_frames, dim=1).unsqueeze(0)

            x_t = self._ddim_ddpm_step(x_t, v_pred, t_int, noise_3d)

            if (sde_idx + 1) % 5 == 0:
                print(f"  Decode step {sde_idx+1}/{self.num_steps}: t={t_int}")

        if latent_correction is not None:
            x_t = x_t + latent_correction.to(x_t.device, dtype=x_t.dtype)

        return self.model.decode_latent(x_t)
