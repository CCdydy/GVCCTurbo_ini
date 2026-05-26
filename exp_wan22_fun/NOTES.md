# Wan2.2-Fun-5B-InP × GVCC integration notes

## Goal
Port GVCC compression to the **Wan2.2-Fun-5B-InP** (5B dense DiT + Wan2.2-VAE) FLF2V model.
Compare RD against:
- FLF2V FINAL (Wan2.1-14B): 31.34 dB @ 193 kbps / ~125s decoder
- MobileI2V-GVCC: 32.85 dB @ 770 kbps / ~3s decoder (I2V only, high bit cost)

## Why this candidate

| | Value | Notes |
|---|---|---|
| Params | 5B (vs 14B Wan2.1) | 2.8× smaller |
| FLF2V support | **native** (Inpaint fine-tune) | first+last frame via mask |
| Backbone family | Wan DiT, **single-DiT** (not MoE) | identical math to our Wan2.1 pipeline |
| VAE | **Wan2.2-VAE** `AutoencoderKLWan3_8` | 48 ch latent, 16× spatial, 4× temporal |
| Scheduler | `FlowMatchEulerDiscreteScheduler(shift=5.0)` | same RF family as Wan2.1 / MobileI2V |
| Per-token timestep | yes (visible tokens t=0) | diffusion-forcing trick, like MobileI2V cond_mask |
| Migration cost | **lowest** of all candidates | reuse most of TurboDDCMWanPipeline |

## Key model API (from VideoX-Fun library)

```
Checkpoint: /home/rog/Desktop/gvcc_turbo/checkpoints/Wan2.2-Fun-5B-InP/
Pipeline:    videox_fun.pipeline.Wan2_2FunInpaintPipeline
Transformer: videox_fun.models.Wan2_2Transformer3DModel  (in_dim=100, out_dim=48)
VAE:         videox_fun.models.AutoencoderKLWan3_8       (48ch, 16×16×4)
Text:        WanT5EncoderModel + AutoTokenizer (UMT5-XXL)
```

### Conditioning channel layout

- Latent `x`:     (B, 48, F_lat, H/16, W/16)   ← regular VAE latent
- Mask `y` aux:   (B, 52, F_lat, H/16, W/16)   = cat([mask_4ch, masked_video_48ch], dim=1)
- Inside transformer: `cat([x, y], dim=1)` → 100 channels → patch_embed → DiT

### FLF2V mask construction

```
input_video[:, :, 0]    = first PIL → tensor in [0,1]
input_video[:, :, -1]   = last PIL  → tensor in [0,1]
input_video[:, :, 1:-1] = first frame tiled (placeholder)
input_video_mask[:, :, 0]    = 0       # 0 = keep
input_video_mask[:, :, -1]   = 0
input_video_mask[:, :, 1:-1] = 255     # 255 = generate
```

Helper: `videox_fun.utils.utils.get_image_to_video_latent(first, last, T, [H,W])`.

### Per-token timestep schedule (diffusion forcing)
For tokens corresponding to mask=0 (visible: first/last frame), the effective timestep
is **t=0** (model treats them as already clean). For mask=255 tokens, normal timestep.
This is the same trick MobileI2V uses (cond_mask in mobiledit.py:660-662).

We must REPLICATE this in our GVCC encoder/decoder loop. The pipeline source code
`pipeline_wan2_2_fun_inpaint.py:680-686` shows the construction.

## GVCC integration plan

### Phase A — `Wan22FunWrapper`
- Reuse most of `WanWrapper` from sde_rf_wan (it's already Wan2.1 oriented)
- Override:
  - VAE class: `AutoencoderKLWan3_8` instead of `AutoencoderKLWan`
  - latent_dim: 48 instead of 16
  - spatial DS: 16× instead of 8×
  - in_channels: 100 (model takes cat([x, y], dim=1))
  - is_flf2v=True
  - flow_shift=5.0 (already supported via shifted_timesteps)

### Phase B — `Wan22FunGVCCPipeline`
- Port from TurboDDCMWanPipeline
- Encode FLF2V conditioning to (mask_4ch, masked_video_48ch) once → `y` tensor
- At each model call: pass `x=current_latent` + `y` constant + per-token timestep
- DDCM atom selection on the SAME 48-channel latent x (NOT on y)
- F_active = F_lat - 2 (frames 0 and -1 conditioned, no atom bits)
- Frame 0 overwrite: replace x[:, :, 0] with VAE-encoded first frame each step
- Frame -1 overwrite: replace x[:, :, -1] with VAE-encoded last frame each step

### Phase C — `run_smoke.py`
- 17-frame? 33-frame? — Wan2.2-Fun supports flexible T = 4n+1, default 81 (5s @ 16fps)
- Match Wan2.1-FLF2V baseline at T=33, H×W=720×1280 for direct comparison
- Output: PSNR / MS-SSIM / LPIPS / BPP at K=16384 M=64 steps=20

## Bit budget estimate (T=33, 720×1280)

- Latent shape: (1, 48, 9, 45, 80) = 1,555,200 elements (Wan2.1 was 16ch × 9 × 90 × 160 = 2,073,600 → slightly smaller)
- F_active = 7 (frames 1..7, frames 0 and 8 conditioned)
- K=16384 M=64 steps=20 (T_sde=17): bits = 17 × 7 × 64 × 15 = 114,240 bits = 14,280 bytes
- BPP = 14280 × 8 / (33 × 720 × 1280) = 0.00375 → 55.4 kbps
- Comparable to Wan2.1-FLF2V FINAL's 193 kbps WHY? Because that includes ref bytes + multi-stage residual budget; pure codebook part is also ~30-50 kbps

## Risks

1. **VAE format**: Wan2.2-VAE uses `model.encode(x, scale)` with baked-in mean/std (different API from diffusers). Need adapter.
2. **Per-token timestep**: must construct timestep per latent token (visible=0, generate=t). Bug here = catastrophic quality loss.
3. **Inpaint y tensor**: stays constant across SDE steps (it's the conditioning), but must be re-encoded with VAE for each GOP. Need to cache.
4. **Domain mismatch**: Wan2.2 family is generally strong on natural video. UVG should be in-distribution.

## Files

- [NOTES.md](NOTES.md) (this file)
- `wan22_fun_wrapper.py` (TBD)
- `wan22_fun_pipeline.py` (TBD)
- `run_smoke.py` (TBD)
