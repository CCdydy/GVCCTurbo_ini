"""
wan22_ti2v_5b_wrapper.py — Wan 2.2 TI2V-5B Wrapper (Diffusers Format, single transformer)

Wan 2.2-TI2V-5B is the smaller Wan 2.2 variant: 5B params, single transformer
(NOT MoE, unlike A14B). Architecture differs from Wan 2.1/2.2-A14B in two
important ways:

  1. VAE has z_dim=48 (vs 16) and spatial stride 16 (vs 8). So a 720p video
     has a 45 × 80 latent grid (vs 90 × 160 for Wan 2.1).
  2. The pipeline uses `expand_timesteps=True` for per-token timestep
     conditioning during image-to-video mode. For pure T2V (no image
     conditioning, mask=1 everywhere), this reduces to the standard scalar
     timestep path that 1-d timestep input handles correctly.

This wrapper provides T2V-mode operation (matching the WanT2VDiffusersWrapper
interface used by TurboDDCMWanPipeline). I2V mode can be added later by
constructing the conditioning mask and switching to per-token timesteps.
"""

import os
import json
import torch
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image


class WanTI2V5BDiffusersWrapper:
    """Wrapper for Wan 2.2-TI2V-5B in diffusers checkpoint format, T2V mode only."""

    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(
        self,
        checkpoint_dir: str = "./Wan2.2-TI2V-5B-Diffusers",
        flow_shift: float = 3.0,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.flow_shift = flow_shift
        self.is_i2v = False  # this wrapper is T2V-mode only
        self.device = None
        self.dtype = None

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from diffusers import WanTransformer3DModel, AutoencoderKLWan
        from transformers import UMT5EncoderModel, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Wan 2.2-TI2V-5B (diffusers) from {self.checkpoint_dir}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.checkpoint_dir, "tokenizer")
        )

        print("  Loading T5 encoder...")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "text_encoder"),
            torch_dtype=dtype,
        )
        self.text_encoder.eval().requires_grad_(False)

        print("  Loading VAE (z_dim=48, stride=16)...")
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.checkpoint_dir, "vae"),
            torch_dtype=torch.float32,
        )
        self.vae.eval().requires_grad_(False)
        self.vae.to(self.device)

        # Latent normalization constants (48-dim for this VAE)
        mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        std = torch.tensor(self.vae.config.latents_std, dtype=torch.float32)
        self.latent_mean = mean.view(1, -1, 1, 1, 1).to(self.device)
        self.latent_std = std.view(1, -1, 1, 1, 1).to(self.device)

        print("  Loading transformer (single, non-MoE)...")
        self.model = WanTransformer3DModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "transformer"),
            torch_dtype=dtype,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

        # VAE config from the loaded module (do not hardcode — TI2V-5B differs!)
        self.vae_stride = (
            self.vae.config.scale_factor_temporal,
            self.vae.config.scale_factor_spatial,
            self.vae.config.scale_factor_spatial,
        )
        self.vae_temporal_factor = self.vae_stride[0]
        self.vae_spatial_factor = self.vae_stride[1]
        self.latent_channels = self.vae.config.z_dim  # 48 for TI2V-5B
        self.patch_size = tuple(self.model.config.patch_size)
        self.sample_neg_prompt = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        print(f"Wan 2.2-TI2V-5B loaded. Device={device}, dtype={dtype}")
        print(f"  VAE compression: {self.vae_temporal_factor}x{self.vae_spatial_factor}x{self.vae_spatial_factor}")
        print(f"  Latent channels: {self.latent_channels}")

    # ================================================================
    # Latent shape helpers
    # ================================================================

    def get_latent_shape(
        self,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
    ) -> Tuple[int, ...]:
        f_lat = (num_frames - 1) // self.vae_temporal_factor + 1
        h_lat = height // self.vae_spatial_factor
        w_lat = width // self.vae_spatial_factor
        h_lat = h_lat + (h_lat % 2)
        w_lat = w_lat + (w_lat % 2)
        return (self.latent_channels, f_lat, h_lat, w_lat)

    def get_frame_shape(
        self,
        height: int = 480,
        width: int = 832,
    ) -> Tuple[int, ...]:
        h_lat = height // self.vae_spatial_factor
        w_lat = width // self.vae_spatial_factor
        h_lat = h_lat + (h_lat % 2)
        w_lat = w_lat + (w_lat % 2)
        return (self.latent_channels, h_lat, w_lat)

    # ================================================================
    # Text encoding
    # ================================================================

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> dict:
        if not negative_prompt:
            negative_prompt = self.sample_neg_prompt

        max_length = 512  # Wan 2.2 default

        self.text_encoder.to(self.device)

        def _encode_one(text):
            inputs = self.tokenizer(
                [text], padding="max_length", max_length=max_length,
                truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(self.device)
            mask = inputs.attention_mask.to(self.device)
            with torch.no_grad():
                out = self.text_encoder(input_ids, attention_mask=mask)
            hidden = out.last_hidden_state.to(dtype=self.dtype)
            seq_len = mask.gt(0).sum(dim=1).long()
            trimmed = hidden[0, :seq_len[0]]
            padded = torch.cat([
                trimmed,
                trimmed.new_zeros(max_length - trimmed.size(0), trimmed.size(1))
            ]).unsqueeze(0)
            return padded

        prompt_embeds = _encode_one(prompt)
        neg_embeds = _encode_one(negative_prompt)

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": neg_embeds,
        }

    # ================================================================
    # VAE encode / decode
    # ================================================================

    def _normalize_latent(self, raw_z: torch.Tensor) -> torch.Tensor:
        return (raw_z - self.latent_mean) / self.latent_std

    def _denormalize_latent(self, z_norm: torch.Tensor) -> torch.Tensor:
        return z_norm * self.latent_std + self.latent_mean

    @torch.no_grad()
    def encode_video(
        self,
        frames: List[Image.Image],
        height: int = 480,
        width: int = 832,
    ) -> torch.Tensor:
        processed = []
        for frame in frames:
            frame = frame.resize((width, height), Image.LANCZOS)
            arr = np.array(frame).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            processed.append(t)

        video_tensor = torch.stack(processed, dim=1).unsqueeze(0)
        video_tensor = 2.0 * video_tensor - 1.0
        video_tensor = video_tensor.to(device=self.device, dtype=torch.float32)

        posterior = self.vae.encode(video_tensor).latent_dist
        raw_z = posterior.mode()

        z_norm = self._normalize_latent(raw_z).float()

        _, _, _, h_lat, w_lat = z_norm.shape
        self._raw_latent_h = h_lat
        self._raw_latent_w = w_lat
        pad_h = h_lat % 2
        pad_w = w_lat % 2
        if pad_h or pad_w:
            z_norm = torch.nn.functional.pad(z_norm, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
        return z_norm

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> List[Image.Image]:
        if latent.dim() == 4:
            latent = latent.unsqueeze(0)

        raw_h = self._raw_latent_h if hasattr(self, '_raw_latent_h') else latent.shape[3]
        raw_w = self._raw_latent_w if hasattr(self, '_raw_latent_w') else latent.shape[4]
        latent = latent[:, :, :, :raw_h, :raw_w]

        raw_z = self._denormalize_latent(latent.float().to(self.device))

        video = self.vae.decode(raw_z, return_dict=False)[0]
        video = video.squeeze(0)
        video = (video / 2.0 + 0.5).clamp(0, 1)
        video = video.permute(1, 2, 3, 0).cpu().float().numpy()

        frames = []
        for f in range(video.shape[0]):
            img = (video[f] * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        return frames

    # ================================================================
    # Velocity prediction
    # ================================================================

    @torch.no_grad()
    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: float,
        prompt_embeds,
        i2v_cond: Optional[dict] = None,
    ) -> torch.Tensor:
        if x_t.dim() == 4:
            x_t = x_t.unsqueeze(0)
        x_in = x_t.to(self.dtype)

        if isinstance(prompt_embeds, list):
            enc_hidden = prompt_embeds[0].unsqueeze(0) if prompt_embeds[0].dim() == 2 else prompt_embeds[0]
        else:
            enc_hidden = prompt_embeds
        enc_hidden = enc_hidden.to(self.device, self.dtype)

        # Wan 2.2-TI2V-5B trained with expand_timesteps=True: per-token timestep.
        # In T2V mode (no image conditioning, mask=1 everywhere) the per-token value
        # is uniformly t, but the model's AdaLN path is (B, seq_len, inner_dim)
        # rather than (B, inner_dim). Match the training-time code path for fair
        # comparison with the model's own pipeline.
        t_1000 = t * self.NUM_TRAIN_TIMESTEPS
        B, C, F_lat, H_lat, W_lat = x_in.shape
        ph, pw = self.patch_size[1], self.patch_size[2]
        # seq_len after patchify: F_lat * (H_lat//ph) * (W_lat//pw)
        seq_len = F_lat * (H_lat // ph) * (W_lat // pw)
        timestep = torch.full(
            (B, seq_len), t_1000,
            device=self.device, dtype=self.dtype,
        )

        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model(
                hidden_states=x_in,
                timestep=timestep,
                encoder_hidden_states=enc_hidden,
                return_dict=False,
            )[0]
        return out.float()

    @torch.no_grad()
    def predict_velocity_cfg(
        self,
        x_t: torch.Tensor,
        t: float,
        embeds: dict,
        guidance_scale: float = 5.0,
        i2v_cond: Optional[dict] = None,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return self.predict_velocity(x_t, t, embeds["prompt_embeds"])
        v_cond = self.predict_velocity(x_t, t, embeds["prompt_embeds"])
        v_uncond = self.predict_velocity(x_t, t, embeds["negative_prompt_embeds"])
        return v_uncond + guidance_scale * (v_cond - v_uncond)
