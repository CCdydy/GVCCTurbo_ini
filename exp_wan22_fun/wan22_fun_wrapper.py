"""wan22_fun_wrapper.py — Wan-like wrapper around Wan2.2-Fun-5B-InP for GVCC.

Bypasses Wan2_2FunInpaintPipeline and goes directly to the model/VAE/text-encoder
so we can substitute the SDE solver with DDCM. Reuses VideoX-Fun's helpers for
mask construction and the per-token timestep trick.

Run in `wan21` conda env.
"""
import os
import sys
import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange

VFUN_ROOT = "/home/rog/Desktop/gvcc_turbo/VideoX-Fun"
if VFUN_ROOT not in sys.path:
    sys.path.insert(0, VFUN_ROOT)

# Bypass __init__.py optional-deps via direct module imports
from videox_fun.models.wan_vae3_8 import AutoencoderKLWan3_8
from videox_fun.models.wan_transformer3d import Wan2_2Transformer3DModel
from videox_fun.models.wan_text_encoder import WanT5EncoderModel
from videox_fun.utils.utils import get_image_to_video_latent
from videox_fun.pipeline.pipeline_wan2_2_fun_inpaint import resize_mask
from transformers import AutoTokenizer
from diffusers.image_processor import VaeImageProcessor


class Wan22FunWrapper:
    """Wan-like wrapper around Wan2.2-Fun-5B-InP."""

    DEFAULT_CKPT   = "/home/rog/Desktop/gvcc_turbo/checkpoints/Wan2.2-Fun-5B-InP"
    DEFAULT_CONFIG = os.path.join(VFUN_ROOT, "config/wan2.2/wan_civitai_5b.yaml")

    is_i2v = True
    is_flf2v = True
    flow_shift = 5.0   # matches FlowMatchEulerDiscreteScheduler(shift=5.0)

    def __init__(self, ckpt_dir: str = None, config_path: str = None,
                 device: str = "cuda", weight_dtype = torch.bfloat16):
        self.device = device
        self.weight_dtype = weight_dtype
        ckpt = ckpt_dir or self.DEFAULT_CKPT
        cfg_path = config_path or self.DEFAULT_CONFIG
        self.cfg = OmegaConf.load(cfg_path)

        # ----- transformer -----
        t0 = time.time()
        self.transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(ckpt, self.cfg['transformer_additional_kwargs'].get(
                'transformer_low_noise_model_subpath', './')),
            transformer_additional_kwargs=OmegaConf.to_container(self.cfg['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).to(device).eval()
        print(f"[Wan22FunWrapper] transformer {sum(p.numel() for p in self.transformer.parameters())/1e9:.2f}B "
              f"({time.time()-t0:.1f}s)")

        # ----- VAE (Wan2.2-VAE: 48ch, 16x spatial, 4x temporal) -----
        t0 = time.time()
        self.vae = AutoencoderKLWan3_8.from_pretrained(
            os.path.join(ckpt, self.cfg['vae_kwargs'].get('vae_subpath')),
            additional_kwargs=OmegaConf.to_container(self.cfg['vae_kwargs']),
        ).to(device).to(weight_dtype).eval()
        self.latent_dim = self.vae.config.latent_channels  # 48
        self.spatial_compression = self.vae.config.spatial_compression_ratio  # 16
        self.temporal_compression = self.vae.config.temporal_compression_ratio  # 4
        print(f"[Wan22FunWrapper] VAE  latent_ch={self.latent_dim}  "
              f"spatial=1/{self.spatial_compression}  temporal=1/{self.temporal_compression}  "
              f"({time.time()-t0:.1f}s)")

        # ----- text encoder + tokenizer -----
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(ckpt, self.cfg['text_encoder_kwargs'].get('tokenizer_subpath')),
        )
        self.text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(ckpt, self.cfg['text_encoder_kwargs'].get('text_encoder_subpath')),
            additional_kwargs=OmegaConf.to_container(self.cfg['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).to(device).eval()
        print(f"[Wan22FunWrapper] text_encoder ({time.time()-t0:.1f}s)")

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.spatial_compression,
            do_normalize=False, do_binarize=True, do_convert_grayscale=True,
        )
        self._null_embeds = None

    # ===================================================================
    # Shape helpers (Wan-like API)
    # ===================================================================

    def get_latent_shape(self, num_frames: int, height: int, width: int):
        """Returns (C, F, H, W) latent shape."""
        assert (num_frames - 1) % self.temporal_compression == 0, \
            f"num_frames must be 4n+1, got {num_frames}"
        F_lat = (num_frames - 1) // self.temporal_compression + 1
        H_lat = height // self.spatial_compression
        W_lat = width  // self.spatial_compression
        return (self.latent_dim, F_lat, H_lat, W_lat)

    def get_frame_shape(self, height: int, width: int):
        return (self.latent_dim,
                height // self.spatial_compression,
                width // self.spatial_compression)

    # ===================================================================
    # Encoding helpers
    # ===================================================================

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> dict:
        if prompt == "" and self._null_embeds is not None:
            return self._null_embeds
        max_len = self.cfg['text_encoder_kwargs'].get('text_length', 512)
        ids = self.tokenizer(prompt, max_length=max_len, padding="max_length",
                              truncation=True, return_tensors="pt").to(self.device)
        mask = ids.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        # WanT5EncoderModel returns (embeds,) tuple
        out = self.text_encoder(ids.input_ids, attention_mask=mask)
        embeds = out[0] if isinstance(out, tuple) else out
        embeds = embeds.to(self.weight_dtype)
        embeds_list = [u[:v] for u, v in zip(embeds, seq_lens)]
        result = {"embeds": embeds_list, "mask": mask}
        if prompt == "":
            self._null_embeds = result
        return result

    @torch.no_grad()
    def encode_video(self, frames: List[Image.Image], height: int, width: int) -> torch.Tensor:
        """List[PIL] of T frames → latent (1, 48, F_lat, H_lat, W_lat)."""
        tensors = []
        for f in frames:
            img = f.resize((width, height), Image.LANCZOS)
            arr = torch.from_numpy(__import__('numpy').asarray(img)).float()
            arr = arr.permute(2, 0, 1) / 127.5 - 1.0  # to [-1, 1]
            tensors.append(arr)
        video = torch.stack(tensors, dim=1).unsqueeze(0).to(self.device).to(self.weight_dtype)
        # video: (1, 3, T, H, W) in [-1, 1]
        distrib = self.vae.encode(video)[0]
        latents = distrib.mode().to(self.weight_dtype)
        return latents

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> List[Image.Image]:
        """Latent (1, 48, F_lat, H_lat, W_lat) → List[PIL] of T pixel frames."""
        video = self.vae.decode(latent.to(self.weight_dtype)).sample
        video = (video / 2 + 0.5).clamp(0, 1).mul(255).round().byte()
        video = video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
        return [Image.fromarray(s) for s in video]

    # ===================================================================
    # FLF2V conditioning
    # ===================================================================

    @torch.no_grad()
    def encode_first_last_frames(self, first: Image.Image, last: Image.Image,
                                  num_frames: int, height: int, width: int) -> dict:
        """Build the dict consumed by predict_velocity_cfg:
          - mask_latents, masked_video_latents: passed as `y` to transformer
          - mask: per-token timestep mask
          - first_latent, last_latent: for explicit overwrite (extra robustness)
        """
        input_video, input_video_mask, _ = get_image_to_video_latent(
            [first], [last], video_length=num_frames, sample_size=[height, width],
        )
        # input_video: (1, 3, T, H, W) in [0,1]; mask: (1, 1, T, H, W) in {0, 255}
        bs, _, T, H, W = input_video.shape

        # Process mask to {0.0, 1.0} float
        m = self.mask_processor.preprocess(
            rearrange(input_video_mask, "b c f h w -> (b f) c h w"),
            height=H, width=W,
        ).to(torch.float32)
        m = rearrange(m, "(b f) c h w -> b c f h w", f=T)

        # Build masked_video (zero-fill on masked regions per Wan2.2-Fun convention)
        masked_video = input_video.to(self.device).to(self.weight_dtype) \
                       * (torch.tile(m, [1, 3, 1, 1, 1]).to(self.device) < 0.5)

        # Encode masked_video via VAE (preserves first/last frame latent info)
        # Scale to [-1, 1] before VAE encode
        masked_video = masked_video * 2.0 - 1.0
        masked_video_latents = self.vae.encode(masked_video)[0].mode().to(self.weight_dtype)

        # Build mask_latents (4 channel) — match pipeline exactly
        m_repeat = torch.cat([
            torch.repeat_interleave(m[:, :, 0:1], repeats=4, dim=2),
            m[:, :, 1:]
        ], dim=2)
        m_view = m_repeat.view(bs, m_repeat.shape[2] // 4, 4, H, W).transpose(1, 2)
        mask_latents = resize_mask(1 - m_view, masked_video_latents, True).to(self.device).to(self.weight_dtype)

        # Per-token timestep mask — matches pipeline lines 644-647 EXACTLY.
        # Interpolate first of 4 grouped channels to latent shape.
        pertoken_mask = F.interpolate(
            m_view[:, :1], size=masked_video_latents.shape[-3:],
            mode='trilinear', align_corners=True,
        ).to(self.device).to(self.weight_dtype)
        # Pipeline's I2V/FLF2V override: if first latent frame is all visible (any=False),
        # force everything after frame 0 to "generate" (mask=1). Last frame is NOT pinned —
        # it's reconstructed by the model from y conditioning.
        if not pertoken_mask[:, :, 0, :, :].any():
            pertoken_mask[:, :, 1:, :, :] = 1.0

        # For x_T init: blend in masked_video_latents at frame 0 only.
        first_latent_slot = masked_video_latents[:, :, :1]  # (B, 48, 1, H_lat, W_lat)

        return {
            "y": torch.cat([mask_latents, masked_video_latents], dim=1),
            "mask": pertoken_mask,
            "first_latent_slot": first_latent_slot,
            "masked_video_latents": masked_video_latents,  # for x_T init blending
        }

    def _pil_to_tensor(self, img: Image.Image, H: int, W: int) -> torch.Tensor:
        img = img.resize((W, H), Image.LANCZOS)
        arr = torch.from_numpy(__import__('numpy').asarray(img)).float()
        arr = arr.permute(2, 0, 1) / 127.5 - 1.0
        return arr.unsqueeze(0).unsqueeze(2).to(self.device).to(self.weight_dtype)

    # ===================================================================
    # Velocity prediction
    # ===================================================================

    @torch.no_grad()
    def predict_velocity_cfg(
        self,
        x_t: torch.Tensor,                  # (1, 48, F_lat, H_lat, W_lat)
        t_float: float,                     # in [0, 1] (SDE convention)
        embeds: dict,                       # from encode_prompt(prompt)
        guidance_scale: float = 6.0,
        flf_cond: dict = None,              # from encode_first_last_frames
    ) -> torch.Tensor:
        """CFG velocity prediction. guidance_scale=1.0 skips CFG (single forward)."""
        assert flf_cond is not None
        do_cfg = guidance_scale != 1.0

        if do_cfg:
            null = self.encode_prompt("")
            in_prompt = list(null["embeds"]) + list(embeds["embeds"])
            x_in = torch.cat([x_t, x_t], dim=0)
            y_in = torch.cat([flf_cond["y"], flf_cond["y"]], dim=0)
        else:
            in_prompt = list(embeds["embeds"])
            x_in = x_t
            y_in = flf_cond["y"]

        # Per-token timestep — match pipeline_wan2_2_fun_inpaint.py:680-686
        # mask: (1, 1, F_lat, H_lat, W_lat); subsample by patch_size (2,2) on H,W
        m = flf_cond["mask"]
        # timestep in scheduler convention: [0, 1000]
        t_sched = float(t_float * 1000.0)
        temp_ts = (m[0][0][:, ::2, ::2] * t_sched).flatten()
        # Compute seq_len matching transformer's patch tokenization
        target_shape = list(x_t.shape[1:])  # (C, F, H, W)
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2])
            * target_shape[1]
        )
        if temp_ts.size(0) < seq_len:
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_sched,
            ])
        temp_ts = temp_ts.unsqueeze(0).expand(x_in.shape[0], temp_ts.size(0))

        with torch.amp.autocast('cuda', dtype=self.weight_dtype):
            noise_pred = self.transformer(
                x=x_in.to(self.weight_dtype),
                context=in_prompt,
                t=temp_ts,
                seq_len=seq_len,
                y=y_in,
            )
        if do_cfg:
            u_uncond, u_cond = noise_pred.chunk(2, dim=0)
            u = u_uncond + guidance_scale * (u_cond - u_uncond)
        else:
            u = noise_pred
        return u.to(x_t.dtype)
