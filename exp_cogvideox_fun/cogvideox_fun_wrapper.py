"""cogvideox_fun_wrapper.py — Wrapper around CogVideoX-Fun-2b-InP (2B, v-pred DDIM)."""
import os, sys, time
from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

VFUN_ROOT = "/home/rog/Desktop/gvcc_turbo/VideoX-Fun"
if VFUN_ROOT not in sys.path:
    sys.path.insert(0, VFUN_ROOT)

# Bypass __init__.py optional-deps via direct module imports
from videox_fun.models.cogvideox_transformer3d import CogVideoXTransformer3DModel
from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
from videox_fun.utils.utils import get_image_to_video_latent
from videox_fun.pipeline.pipeline_cogvideox_fun_inpaint import resize_mask
from diffusers.image_processor import VaeImageProcessor


class CogVideoXFunWrapper:
    """Wan-like wrapper around CogVideoX-Fun-2b-InP."""

    DEFAULT_CKPT = "/home/rog/Desktop/gvcc_turbo/checkpoints/CogVideoX-Fun-2b-InP"

    is_i2v = True
    is_flf2v = True
    # DDPM / v-prediction (NOT flow matching)

    def __init__(self, ckpt_dir: str = None,
                 device: str = "cuda", weight_dtype = torch.bfloat16):
        self.device = device
        self.weight_dtype = weight_dtype
        ckpt = ckpt_dir or self.DEFAULT_CKPT

        t0 = time.time()
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            ckpt, subfolder="transformer", torch_dtype=weight_dtype,
        ).to(device).eval()
        print(f"[CogVideoXFunWrapper] transformer "
              f"{sum(p.numel() for p in self.transformer.parameters())/1e9:.2f}B "
              f"({time.time()-t0:.1f}s)")

        t0 = time.time()
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            ckpt, subfolder="vae"
        ).to(device).to(weight_dtype).eval()
        self.latent_dim = self.vae.config.latent_channels  # 16
        self.spatial_compression = 8
        self.temporal_compression = 4
        self.vae_scaling = self.vae.config.scaling_factor  # 1.15258426
        print(f"[CogVideoXFunWrapper] VAE  latent_ch={self.latent_dim}  "
              f"scaling={self.vae_scaling}  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained(ckpt, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            ckpt, subfolder="text_encoder", torch_dtype=weight_dtype,
        ).to(device).eval()
        print(f"[CogVideoXFunWrapper] text_encoder ({time.time()-t0:.1f}s)")

        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(ckpt, subfolder="scheduler")
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.spatial_compression,
            do_normalize=False, do_binarize=True, do_convert_grayscale=True,
        )
        self._null_embeds = None

    # ===================================================================
    # Shape helpers
    # ===================================================================

    def get_latent_shape(self, num_frames: int, height: int, width: int):
        assert (num_frames - 1) % self.temporal_compression == 0, \
            f"num_frames must be 4n+1, got {num_frames}"
        F_lat = (num_frames - 1) // self.temporal_compression + 1
        H_lat = height // self.spatial_compression
        W_lat = width  // self.spatial_compression
        return (self.latent_dim, F_lat, H_lat, W_lat)

    def get_frame_shape(self, height: int, width: int):
        return (self.latent_dim,
                height // self.spatial_compression,
                width  // self.spatial_compression)

    # ===================================================================
    # Encoding
    # ===================================================================

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if prompt == "" and self._null_embeds is not None:
            return self._null_embeds
        ids = self.tokenizer(prompt, max_length=226, padding="max_length",
                              truncation=True, return_tensors="pt").to(self.device)
        embeds = self.text_encoder(ids.input_ids)[0].to(self.weight_dtype)
        if prompt == "":
            self._null_embeds = embeds
        return embeds

    @torch.no_grad()
    def encode_video(self, frames: List[Image.Image], height: int, width: int) -> torch.Tensor:
        import numpy as np
        tensors = []
        for f in frames:
            img = f.resize((width, height), Image.LANCZOS)
            arr = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 127.5 - 1.0
            tensors.append(arr)
        video = torch.stack(tensors, dim=1).unsqueeze(0).to(self.device).to(self.weight_dtype)
        # video: (1, 3, T, H, W) in [-1, 1]
        latent = self.vae.encode(video).latent_dist.sample() * self.vae_scaling
        return latent.to(self.weight_dtype)

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> List[Image.Image]:
        import numpy as np
        frames = self.vae.decode(latent.to(self.weight_dtype) / self.vae_scaling).sample
        frames = (frames / 2 + 0.5).clamp(0, 1).mul(255).round().byte()
        frames = frames.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        return [Image.fromarray(s) for s in frames]

    # ===================================================================
    # FLF2V conditioning (inpaint_latents kwarg)
    # ===================================================================

    @torch.no_grad()
    def encode_first_last_frames(self, first: Image.Image, last: Image.Image,
                                  num_frames: int, height: int, width: int) -> dict:
        """Build inpaint_latents tensor consumed by predict_velocity_cfg."""
        input_video, input_video_mask, _ = get_image_to_video_latent(
            [first], [last], video_length=num_frames, sample_size=[height, width],
        )
        bs, _, T, H, W = input_video.shape

        m_cond = self.mask_processor.preprocess(
            rearrange(input_video_mask, "b c f h w -> (b f) c h w"),
            height=H, width=W,
        ).to(torch.float32)
        m_cond = rearrange(m_cond, "(b f) c h w -> b c f h w", f=T)

        # masked_video: -1 fill (CogVideoX convention)
        m_tile = torch.tile(m_cond, [1, 3, 1, 1, 1])
        masked_video = (input_video.to(self.device).to(self.weight_dtype) * 2.0 - 1.0) * (m_tile.to(self.device) < 0.5) \
                       + torch.ones_like(input_video.to(self.device).to(self.weight_dtype)) * (m_tile.to(self.device) > 0.5) * -1

        # VAE encode masked_video → (B, 16, F_lat, H_lat, W_lat) — using mode (not sample)
        mv_latents = self.vae.encode(masked_video).latent_dist.mode() * self.vae_scaling
        mv_latents = mv_latents.to(self.weight_dtype)

        # Build mask_latents (1 channel at latent res)
        mask_latents = resize_mask(1 - m_cond, mv_latents).to(self.device).to(self.weight_dtype) \
                       * self.vae_scaling

        # Inpaint format: cat along channel dim AFTER rearrange to (B, F_lat, C, H, W)
        mv_in = rearrange(mv_latents, "b c f h w -> b f c h w")
        m_in  = rearrange(mask_latents, "b c f h w -> b f c h w")
        inpaint_latents = torch.cat([m_in, mv_in], dim=2).to(self.weight_dtype)
        # shape: (B, F_lat, 17, H_lat, W_lat) — concatenated INSIDE transformer with hidden_states

        return {
            "inpaint_latents": inpaint_latents,
            "masked_video_latents": mv_latents,
        }

    # ===================================================================
    # Velocity / v-prediction
    # ===================================================================

    @torch.no_grad()
    def predict_v_cfg(
        self,
        x_t: torch.Tensor,                  # (1, 16, F_lat, H_lat, W_lat)
        timestep_int,                       # int or torch scalar in [0, 999]
        embeds: torch.Tensor,
        guidance_scale: float = 6.0,
        flf_cond: dict = None,
    ) -> torch.Tensor:
        """CFG v-prediction. timestep is in scheduler's int domain [0, 999]."""
        null = self.encode_prompt("")
        prompt_embeds = torch.cat([null, embeds], dim=0)

        # Latent format for CogVideoX: (B, F_lat, C, H, W)
        x_in = torch.cat([x_t, x_t], dim=0)
        x_in = rearrange(x_in, "b c f h w -> b f c h w")
        # scale_model_input is no-op for DDIM but be consistent
        x_in = self.scheduler.scale_model_input(x_in, timestep_int)

        ipl = flf_cond["inpaint_latents"]
        ipl_in = torch.cat([ipl, ipl], dim=0)

        timestep_t = torch.tensor([float(timestep_int)], device=self.device,
                                   dtype=self.weight_dtype).expand(x_in.shape[0])

        out = self.transformer(
            hidden_states=x_in.to(self.weight_dtype),
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_t,
            return_dict=False,
            inpaint_latents=ipl_in,
        )[0]
        # Output: (B, F_lat, 16, H_lat, W_lat) — rearrange back
        out = rearrange(out, "b f c h w -> b c f h w").float()

        # CFG
        u_uncond, u_cond = out.chunk(2, dim=0)
        u = u_uncond + guidance_scale * (u_cond - u_uncond)
        return u.to(x_t.dtype)
