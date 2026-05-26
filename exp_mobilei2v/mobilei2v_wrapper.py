"""mobilei2v_wrapper.py — Thin Wan-like wrapper around MobileI2V (0.27B SANA + LTX-VAE).

Exposes the subset of API that TurboDDCMWanPipeline needs:
  - is_i2v (always True)
  - latent_shape, frame_shape, num_latent_frames
  - flow_shift (matches MobileI2V's FlowMatchEuler shift=3.0)
  - encode_image(ref_pil, num_frames, H, W) -> dict (the "i2v_cond")
  - encode_prompt(prompt) -> dict
  - encode_video(frames, H, W) -> latent (1,C,F,H_lat,W_lat)
  - decode_latent(latent) -> list[PIL.Image]
  - predict_velocity_cfg(x_t, t, embeds, guidance_scale, i2v_cond) -> velocity

MobileI2V repo lives at /media/rog/data2_SSD/MobileI2V and must be importable
(sys.path inserted at module load). Run in `sana` conda env.
"""
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from PIL import Image
from torchvision import transforms

MOBILEI2V_ROOT = "/media/rog/data2_SSD/MobileI2V"
if MOBILEI2V_ROOT not in sys.path:
    sys.path.insert(0, MOBILEI2V_ROOT)

# These imports require sana env + chdir-style setup
import pyrallis
from diffusion.model.builder import (
    build_model, get_tokenizer_and_text_encoder, get_vae, vae_encode, vae_decode,
)
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.model.utils import get_weight_dtype
from tools.download import find_model


@dataclass
class _SanaInference(SanaConfig):
    """Subset of SanaInference dataclass needed for pyrallis.parse."""
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    sampling_algo: str = "flow_euler"
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False
    save_path: str = ""
    flow_score: float = 1.0
    prompt: str = ""


class MobileI2VWrapper:
    """Wan-like wrapper around MobileI2V."""

    DEFAULT_CONFIG = os.path.join(
        MOBILEI2V_ROOT, "configs/mobilei2v_config/MobileI2V_300M_img512.yaml")
    DEFAULT_CKPT = os.path.join(MOBILEI2V_ROOT, "model/hybrid_371.pth")

    # Wan-like attributes
    is_i2v = True
    is_flf2v = False
    flow_shift = 3.0  # FlowMatchEulerDiscreteScheduler(shift=3.0)

    def __init__(self, config_path: str = None, ckpt_path: str = None,
                 device: str = "cuda", flow_score: float = 2.0):
        self.device = device
        self.flow_score_val = flow_score

        cfg_path = config_path or self.DEFAULT_CONFIG
        ckpt = ckpt_path or self.DEFAULT_CKPT

        # pyrallis requires sys.argv; mock it
        saved_argv = sys.argv
        sys.argv = ["wrapper", "--config", cfg_path]
        try:
            config = pyrallis.parse(config_class=_SanaInference, config_path=cfg_path)
        finally:
            sys.argv = saved_argv
        self.config = config

        # The MobileI2V config has relative paths like ./model/video-vae.
        # Resolve them against MOBILEI2V_ROOT.
        def _abs(p):
            if p and p.startswith("./"):
                return os.path.join(MOBILEI2V_ROOT, p[2:])
            return p
        config.vae.vae_pretrained = _abs(config.vae.vae_pretrained)
        config.model.load_from = _abs(config.model.load_from)
        # NOTE: text_encoder_name is used as a dict KEY in get_tokenizer_and_text_encoder
        # (string match against known names like "./model/Qwen2-0.5B"), so we
        # must NOT absolutify it. The dict lookup then returns the path string,
        # which we resolve via chdir to MOBILEI2V_ROOT.

        # Ensure relative paths inside text_encoder loading work
        self._mobilei2v_root = MOBILEI2V_ROOT
        _saved_cwd = os.getcwd()
        os.chdir(MOBILEI2V_ROOT)

        self.H = config.model.image_height
        self.W = config.model.image_width
        self.ds = config.vae.vae_downsample_rate
        self.latent_h = self.H // self.ds + (1 if self.H % self.ds else 0)
        self.latent_w = self.W // self.ds
        self.num_pixel_frames = config.data.num_frames  # 17
        # 8x temporal compression + 1 for first-frame slot
        self.num_latent_frames = self.num_pixel_frames // 8 + 1  # 3
        self.latent_dim = config.vae.vae_latent_dim  # 128
        self.scale_factor = config.vae.scale_factor  # 0.41407

        self.weight_dtype = get_weight_dtype(config.model.mixed_precision)  # fp16
        self.max_seq_len = config.text_encoder.model_max_length  # 300

        # Load components
        t0 = time.time()
        self.vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device)
        self.vae_type = config.vae.vae_type
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=device)

        model_kwargs = model_init_config(config, latent_size=(self.latent_h, self.latent_w))
        self.model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False),
            **model_kwargs,
        ).to(device)
        state_dict = find_model(ckpt)
        if "pos_embed" in state_dict["state_dict"]:
            del state_dict["state_dict"]["pos_embed"]
        self.model.load_state_dict(state_dict["state_dict"], strict=False)
        self.model.eval().to(self.weight_dtype)
        os.chdir(_saved_cwd)
        print(f"[MobileI2VWrapper] Loaded in {time.time()-t0:.1f}s. "
              f"latent_shape=({self.latent_dim},{self.num_latent_frames},"
              f"{self.latent_h},{self.latent_w})")

        # Image transform
        self._xform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([self.H, self.W]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

        # Cache empty prompt embeds for re-use
        self._null_embeds = None

    # ----------------------- Wan-like accessors -----------------------

    def get_latent_shape(self, num_frames, height, width):
        """Wan API parity. We only support the model's fixed shape."""
        assert num_frames == self.num_pixel_frames, \
            f"MobileI2V is fixed at {self.num_pixel_frames} frames, got {num_frames}"
        assert (height, width) == (self.H, self.W), \
            f"MobileI2V fixed at {self.H}x{self.W}, got {height}x{width}"
        return (self.latent_dim, self.num_latent_frames, self.latent_h, self.latent_w)

    def get_frame_shape(self, height, width):
        """Wan API parity. Per-latent-frame shape (C, H, W)."""
        return (self.latent_dim, self.latent_h, self.latent_w)

    # ----------------------- encoding paths -----------------------

    def _pixel_to_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """List of PIL → (1, C=3, T, H, W) fp32 tensor in [-1, 1]."""
        tensors = [self._xform(f) for f in frames]
        video = torch.stack(tensors, dim=1)  # (3, T, H, W)
        return video.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_image(self, ref_image: Image.Image, num_frames=None,
                     height=None, width=None) -> dict:
        """Encode first-frame ref. Returns dict {ref_latent, cond_mask}.

        This plays the role of Wan's i2v_cond dict — MobileI2V-specific contents.
        """
        if isinstance(ref_image, list):
            ref_image = ref_image[0]
        pix = self._pixel_to_tensor([ref_image])  # (1, 3, 1, H, W)
        ref_latent = vae_encode(
            self.vae_type, self.vae, pix.float(), self.config.vae.sample_posterior,
            device=self.device,
        ).to(self.weight_dtype)  # (1, 128, 1, latent_h, latent_w)

        # cond_mask: ones at first frame slot, zeros elsewhere
        # shape (1, F*H*W) = (1, 3 * 23 * 40)
        cond_mask = torch.zeros(
            (1, self.num_latent_frames * self.latent_h * self.latent_w),
            dtype=torch.float32, device=self.device,
        )
        cond_mask[:, :self.latent_h * self.latent_w] = 1.0

        return {"ref_latent": ref_latent, "cond_mask": cond_mask}

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> dict:
        """Tokenize + encode prompt with Qwen2-0.5B. Returns dict with
        keys 'embeds' (the actual conditioning) and 'mask' (attention mask).
        """
        if prompt == "" and self._null_embeds is not None:
            return self._null_embeds
        tok = self.tokenizer(
            prompt, max_length=self.max_seq_len, padding="max_length",
            truncation=True, return_tensors="pt").to(self.device)
        embeds = self.text_encoder(tok.input_ids, tok.attention_mask)[0][:, None]
        # shape (1, 1, 300, 896)
        embeds = embeds.to(self.weight_dtype)
        result = {"embeds": embeds, "mask": tok.attention_mask}
        if prompt == "":
            self._null_embeds = result
        return result

    @torch.no_grad()
    def encode_video(self, frames: List[Image.Image], height=None,
                     width=None) -> torch.Tensor:
        """Encode pixel video → latent. Returns (1, 128, F, H_lat, W_lat) fp16."""
        assert len(frames) == self.num_pixel_frames, \
            f"need {self.num_pixel_frames} frames, got {len(frames)}"
        pix = self._pixel_to_tensor(frames)  # (1, 3, 17, H, W)
        lat = vae_encode(
            self.vae_type, self.vae, pix.float(), self.config.vae.sample_posterior,
            device=self.device,
        ).to(self.weight_dtype)
        return lat  # (1, 128, 3, 23, 40)

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> List[Image.Image]:
        """Decode latent → list of PIL frames."""
        samples = vae_decode(self.vae_type, self.vae, latent.float())
        # samples: (1, 3, T_pixel, H, W) in [-1, 1]
        samples = samples.clamp(-1, 1).add(1).div(2).mul(255).round().byte()
        samples = samples.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, 3)
        return [Image.fromarray(s) for s in samples]

    # ----------------------- velocity prediction -----------------------

    @torch.no_grad()
    def predict_velocity_cfg(
        self,
        x_t: torch.Tensor,                   # (1, C, F, H, W) in [0,1]-scaled t
        t_float: float,                      # in [0, 1]
        embeds: dict,                        # from encode_prompt(prompt)
        guidance_scale: float = 4.5,
        i2v_cond: dict = None,               # from encode_image
        flow_score: float = None,
    ) -> torch.Tensor:
        """CFG velocity prediction. t_float is in [0,1] (matches Wan SDE math)."""
        assert i2v_cond is not None, "MobileI2V requires i2v_cond"
        flow_score = flow_score if flow_score is not None else self.flow_score_val

        # Force first-frame slot to ref_latent (mirrors flow_euler_sampler line 89)
        x_t = x_t.clone()
        x_t[:, :, :1] = i2v_cond["ref_latent"]

        # Build CFG batch (uncond + cond text); guide image not duplicated
        null = self.encode_prompt("")
        cond_embeds = embeds["embeds"]              # (1,1,300,896)
        null_embeds = null["embeds"]                # (1,1,300,896)
        prompt_embeds = torch.cat([null_embeds, cond_embeds], dim=0)  # (2,1,300,896)
        prompt_embeds = prompt_embeds.squeeze(1).unsqueeze(1)  # match flow_euler_sampler's unsqueeze(1)
        # Reshape: original code does prompt_embeds.unsqueeze(1) on already cat'd
        # tensor of shape (2,300,896) — here we already have (2,1,300,896).
        # Use the cat'd 4D tensor directly.
        prompt_embeds = torch.cat([null_embeds, cond_embeds], dim=0).unsqueeze(1) \
                          if cond_embeds.ndim == 3 else torch.cat([null_embeds, cond_embeds], dim=0)
        # Note: native scripts do `prompt_embeds = torch.cat([uncond, cond], 0).unsqueeze(1)`
        # starting from (1,L,896). Our encode_prompt returns (1,1,L,896) already so:
        prompt_embeds = torch.cat([null_embeds, cond_embeds], dim=0)  # (2,1,L,896)

        # Latent batched
        x_in = torch.cat([x_t, x_t], dim=0)
        B = x_in.shape[0]
        F_lat, h, w = self.num_latent_frames, self.latent_h, self.latent_w

        cond_mask = i2v_cond["cond_mask"].expand(B, -1)
        flow_s = torch.tensor([flow_score], device=self.device, dtype=torch.float32).expand(B)

        # Timestep in [0,1000] — model divides by 1000 internally
        t_model = torch.tensor([t_float * 1000.0], device=self.device,
                               dtype=self.weight_dtype).expand(B)

        # attention mask for text (1,L) — both uncond & cond have full mask
        text_mask = torch.cat([null["mask"], embeds["mask"]], dim=0)

        out = self.model(
            x_in.to(self.weight_dtype),
            t_model,
            i2v_cond["ref_latent"],  # guide_image (model ignores it in this version)
            prompt_embeds.to(self.weight_dtype),
            cond_mask,
            flow_s,
            mask=text_mask,
            data_info={
                "img_hw": torch.tensor([[self.H, self.W]], dtype=torch.float32,
                                       device=self.device).repeat(B, 1),
                "aspect_ratio": torch.tensor([[1.0]], device=self.device).repeat(B, 1),
            },
        )
        # CFG combine
        u_uncond, u_cond = out.chunk(2, dim=0)
        u = u_uncond + guidance_scale * (u_cond - u_uncond)
        return u.to(x_t.dtype)
