"""
wan22_t2v_wrapper.py — Wan2.2 T2V-A14B Wrapper (Diffusers Format, dual-expert MoE)

Wan2.2 introduces a timestep-boundary MoE: two architecturally-identical
WanTransformer3DModel instances (`transformer` and `transformer_2`) that
split the denoising trajectory at `boundary_ratio` of the training timesteps.

Per diffusers WanPipeline:
  - timestep >= boundary_ratio * num_train_timesteps  → use `transformer`     (high-noise)
  - timestep <  boundary_ratio * num_train_timesteps  → use `transformer_2`   (low-noise)

Everything else (VAE, T5, scheduler) is shared with the Wan2.1 diffusers wrapper.
"""

import os
import json
import torch
import numpy as np
from typing import Tuple, List, Optional
from PIL import Image


class WanT2VA14BDiffusersWrapper:
    """Wrapper for Wan2.2 T2V-A14B in diffusers checkpoint format with dual-expert MoE."""

    NUM_TRAIN_TIMESTEPS = 1000  # Wan default

    def __init__(
        self,
        checkpoint_dir: str = "./Wan2.2-T2V-A14B-Diffusers",
        flow_shift: float = 3.0,
        boundary_ratio: Optional[float] = None,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.flow_shift = flow_shift
        self.is_i2v = False
        self.device = None
        self.dtype = None

        # Read boundary_ratio from model_index.json unless overridden
        if boundary_ratio is None:
            mi_path = os.path.join(self.checkpoint_dir, "model_index.json")
            with open(mi_path) as f:
                mi = json.load(f)
            boundary_ratio = mi.get("boundary_ratio", 0.875)
        self.boundary_ratio = boundary_ratio
        self.boundary_timestep = boundary_ratio * self.NUM_TRAIN_TIMESTEPS  # in [0,1000] scale

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from diffusers import WanTransformer3DModel, AutoencoderKLWan
        from transformers import UMT5EncoderModel, AutoTokenizer

        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Wan2.2 T2V-A14B (diffusers) from {self.checkpoint_dir}...")
        print(f"  boundary_ratio={self.boundary_ratio}  (switch at t={self.boundary_timestep:.0f}/1000)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.checkpoint_dir, "tokenizer")
        )

        print("  Loading T5 encoder...")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "text_encoder"),
            torch_dtype=dtype,
        )
        self.text_encoder.eval().requires_grad_(False)

        print("  Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(self.checkpoint_dir, "vae"),
            torch_dtype=torch.float32,
        )
        self.vae.eval().requires_grad_(False)
        self.vae.to(self.device)

        mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        std = torch.tensor(self.vae.config.latents_std, dtype=torch.float32)
        self.latent_mean = mean.view(1, -1, 1, 1, 1).to(self.device)
        self.latent_std = std.view(1, -1, 1, 1, 1).to(self.device)

        print("  Loading transformer (high-noise expert, t >= boundary)...")
        self.model = WanTransformer3DModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "transformer"),
            torch_dtype=dtype,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(self.device)

        print("  Loading transformer_2 (low-noise expert, t < boundary)...")
        self.model_2 = WanTransformer3DModel.from_pretrained(
            os.path.join(self.checkpoint_dir, "transformer_2"),
            torch_dtype=dtype,
        )
        self.model_2.eval().requires_grad_(False)
        self.model_2.to(self.device)

        self.vae_stride = (4, 8, 8)
        self.vae_temporal_factor = self.vae_stride[0]
        self.vae_spatial_factor = self.vae_stride[1]
        self.latent_channels = self.vae.config.z_dim
        self.patch_size = tuple(self.model.config.patch_size)
        self.sample_neg_prompt = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
            "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        )

        print(f"Wan2.2 T2V-A14B loaded. Device={device}, dtype={dtype}")
        print(f"  VAE compression: {self.vae_temporal_factor}x{self.vae_spatial_factor}x{self.vae_spatial_factor}")
        print(f"  Latent channels: {self.latent_channels}")

    def _select_transformer(self, t_scaled_1000: float):
        """Pick transformer based on timestep in [0,1000] scale.
        t >= boundary_timestep → transformer (high noise)
        t <  boundary_timestep → transformer_2 (low noise)
        """
        if t_scaled_1000 >= self.boundary_timestep:
            return self.model
        return self.model_2

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

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> dict:
        if not negative_prompt:
            negative_prompt = self.sample_neg_prompt

        max_length = 512  # Wan2.2 uses longer context than 2.1

        self.text_encoder.to(self.device)

        def _encode_one(text):
            inputs = self.tokenizer(
                [text],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
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

        # t is in [0, 1] scale; Wan native uses [0, 1000]
        t_1000 = t * self.NUM_TRAIN_TIMESTEPS
        timestep = torch.tensor([t_1000], device=self.device, dtype=self.dtype)

        # Dual-expert dispatch
        model = self._select_transformer(t_1000)

        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = model(
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
