# CogVideoX-Fun-2b-InP × GVCC integration notes

## Goal
Port GVCC to **CogVideoX-Fun-2b-InP** (2B DiT + CogVideoX-VAE) FLF2V model.
The smallest open-weights model with native FLF2V support.

## Why this candidate

| | Value | Notes |
|---|---|---|
| Params | 2B (5× smaller than Wan2.2-Fun-5B) | smallest native FLF2V |
| FLF2V | native (Inpaint fine-tune) | first+last via mask |
| Backbone | CogVideoX DiT | independent family (NOT Wan-derived) |
| VAE | `AutoencoderKLCogVideoX` | **16 ch** latent (same as Wan2.1!), 8× spatial, 4× temporal |
| Scheduler | `CogVideoXDDIMScheduler` | **v-prediction**, scaled_linear betas (NOT flow matching!) |
| Migration cost | **HIGH** — different SDE math required | need v-prediction → score conversion |

## ⚠️ Major caveat: NOT a flow matching model

CogVideoX uses **DDPM-style v-prediction with DDIM sampler**:
- `prediction_type="v_prediction"` (NOT velocity field of RF/flow matching)
- `beta_schedule="scaled_linear"`, `beta_start=0.00085`, `beta_end=0.012`
- `snr_shift_scale=3.0`, `rescale_betas_zero_snr=True`

Our existing `sde_convert.py` assumes **linear interpolant** (RF: α_t=1-t, σ_t=t).
For CogVideoX, we need DDPM math:
- α_t = ∏(1 - β_i)
- σ_t = sqrt(1 - α_t²)
- v_pred = α_t · ε - σ_t · x_0
- score = -ε / σ_t

Will need a new `cogvideox_sde_convert.py` module OR derive the equivalent transforms.

## Key model API

```
Checkpoint: /home/rog/Desktop/gvcc_turbo/checkpoints/CogVideoX-Fun-2b-InP/
Pipeline:    videox_fun.pipeline.CogVideoXFunInpaintPipeline
Transformer: videox_fun.models.CogVideoXTransformer3DModel  (in_channels=33)
VAE:         AutoencoderKLCogVideoX (diffusers), latent_dim=16
Text:        T5EncoderModel + T5Tokenizer
```

### Conditioning channel layout

- Latent `hidden_states`: (B, F_lat, 16, H/8, W/8)
- Inpaint aux:             (B, F_lat, 17, H/8, W/8) = cat([mask_1ch, masked_video_16ch])
- Inside transformer:      cat([hidden, inpaint], channel-dim) → 33 ch

Same mask convention as Wan2.2-Fun (`get_image_to_video_latent`).

### Inpaint fill convention
CogVideoX uses **-1 fill** for masked-out regions (vs Wan22-Fun's **0 fill**).

## Integration plan

### Phase A — `CogVideoXFunWrapper`
- Load via diffusers `CogVideoXTransformer3DModel.from_pretrained(model, subfolder="transformer")`
- Standard `AutoencoderKLCogVideoX` from diffusers
- T5 encoder via standard diffusers/transformers loaders
- predict_velocity_cfg → predict_v_cfg (v_prediction)

### Phase B — `CogVideoXFunGVCCPipeline`
- Implement DDPM SDE encoder using v-pred → score conversion
- Encoder loop:
  1. Build initial x_T ~ N(0,I)
  2. At each timestep t (scheduler.timesteps):
     - v_pred = model(x_t, t, prompt_embeds, inpaint_latents)
     - x_0_hat = α_t · x_t - σ_t · v_pred
     - residual = x_0_true - x_0_hat
     - score = -ε / σ_t where ε = σ_t · x_t + α_t · v_pred (DDIM ε form)
     - drift, noise_coeff via DDPM SDE math
     - DDCM atom selection
     - x_{t-1} = drift + noise_coeff * combined_noise
  3. Apply latent_correction (tail residual)
- F_active = F_lat - 2 (first + last conditioned)
- Don't overwrite x[:, :, 0] / x[:, :, -1] explicitly — inpaint_latents conditioning + masked_video should be enough

### Phase C — `run_smoke.py`
- T = 4n+1 (49 default), H×W flexible (384×672 default per their script)
- Match our 33-frame GOP convention? Could use T=33 (4×8+1).
- Output: PSNR / MS-SSIM / LPIPS / BPP

## Bit budget estimate (T=33, 720×1280)

- Latent shape: (1, 16, 9, 90, 160) = 2,073,600 elements (same as Wan2.1-I2V)
- F_active = 7 (frames 1..7, 0 and 8 conditioned)
- K=16384 M=64 steps=20: bits = 17 × 7 × 64 × 15 = 114,240 bits = 14,280 bytes = 55.4 kbps
- Identical to Wan2.2-Fun-5B estimate by coincidence (16ch × 9 latent frames = 48ch × 9/3 = similar volume after spatial)
- Actually different: Wan2.2 spatial 16× → 45×80=3600 per (ch,frame); Cog 8× → 90×160=14400 per (ch,frame). Wan2.2 has 4× fewer elements per (ch,frame) but 3× more channels. Net: Wan2.2 latent ≈ 0.75× Cog volume.

## Risks

1. **v-prediction math is different** — implementation effort 2-3× Wan2.2-Fun
2. **CogVideoX-Fun was V1.1 era** — may have weaker generation than recent models
3. **Author's scheduler choice (`DDIM_Origin` in their script)** — different from `CogVideoXDDIMScheduler`; need to figure out which to match
4. **First/last anchoring** — without explicit overwrite (relying only on inpaint conditioning), frame drift might still occur. May need explicit overwrite of latent slots 0 and -1 like MobileI2V.

## Strategy

Recommend testing Wan2.2-Fun-5B **first** (lower-risk port, same RF math).
If that gives competitive RD, CogVideoX-Fun-2b might not be worth the extra implementation effort.

## Files

- [NOTES.md](NOTES.md) (this file)
- `cogvideox_fun_wrapper.py` (TBD)
- `cogvideox_fun_pipeline.py` (TBD)
- `cogvideox_sde_convert.py` (TBD — v-pred → score)
- `run_smoke.py` (TBD)
