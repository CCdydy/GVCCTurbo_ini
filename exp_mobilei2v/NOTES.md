# MobileI2V × GVCC integration notes

## Goal
Port GVCC compression (DDCM SDE + tail_residual + chained-GOP) onto MobileI2V
(0.27B, LTX-VAE, first-frame I2V) and measure quality gap vs Wan-I2V-14B baseline.

## Reference: how old GVCC solved I2V

Code: `turbogvcc/exp_i2v/run_uvg_chained_gop.py` +
`turbogvcc/sde_rf_wan/turbo_pipeline.py:308` (`latent_correction` kwarg).

Two residual techniques:

1. **Chained-GOP AR ref** — GOP 0: ref = GT first frame (free). GOP k>0:
   ref = decoded last frame from GOP k-1 (0 bytes). No bit cost beyond GOP 0.

2. **Tail latent residual** — encoder takes `(x0_true − x0_enc)` at the last
   latent frame (covers last 4 pixel frames), 8-bit per-channel min/max
   quantize + zlib. Decoder adds it back to final latent before VAE decode.
   Costs ~hundreds of bytes/GOP, pulls the drifted tail back toward GT.

Both techniques are model-agnostic; the only dependency is the pipeline
exposing `latent_correction` to `decode()`.

## MobileI2V interface (verified via smoke_load.py)

- Model forward:
  `model(z, t, guide_image_latent, prompt_embeds, cond_mask, flow_score)`
  → returns velocity prediction (predict_v=True, linear_flow scheduler).
- I2V mechanism: first latent frame slot is **overwritten** with
  `ref_latent` *at every sampler step* (see flow_euler_sampler.py:89:
  `latents[:, :, :1] = lerp(latents[:, :, :1], guide_image, 1.0)`).
- `cond_mask`: shape `(B, F*H*W)`, ones for first-frame slot, zeros elsewhere.
- `flow_score`: scalar conditioning (paper default 2.0).
- CFG: standard `noise_pred_uncond + cfg_scale * (text - uncond)`, cfg_scale=4.5.
- Scheduler: `FlowMatchEulerDiscreteScheduler(shift=3.0)`.
- VAE: LTX-Video, fp32 by default, latent_dim=128, scale_factor=0.41407.
- Text encoder: Qwen2-0.5B.

## Shape summary (720×1280, 17 frames)

| Tensor | Shape | Dtype |
|---|---|---|
| pixel video | (1, 3, 17, 720, 1280) | fp16/fp32 |
| latent video | (1, 128, 3, 23, 40) | fp16 |
| ref latent | (1, 128, 1, 23, 40) | fp16 |
| prompt embed (Qwen2) | (1, 1, 300, 896) | fp16 |
| cond_mask | (B, 3*23*40=2760) | fp32 |

## Integration plan

### Phase A — `MobileI2VWrapper` (Wan-like API)
Lives in `turbogvcc/exp_mobilei2v/mobilei2v_wrapper.py`. Owns model + VAE +
tokenizer. Exposes:

```python
class MobileI2VWrapper:
    is_i2v = True
    latent_shape = (128, 3, 23, 40)         # C, F, H, W
    num_pixel_frames = 17
    height, width = 720, 1280
    scheduler                                # FlowMatchEulerDiscreteScheduler(shift=3.0)
    timesteps                                # cached after set_timesteps(num_steps)

    def encode_image(ref_pil, num_frames, H, W) -> dict:
        # returns dict with ref_latent + cond_mask (matches "i2v_cond" pattern from Wan)
    def encode_prompt(prompt: str) -> Tensor:
        # Qwen2-0.5B → (1, 1, 300, 896)
    def decode_latent(latent: Tensor) -> List[PIL.Image]:
        # LTX VAE decode
    def encode_video_to_latent(frames: List[PIL.Image]) -> Tensor:
        # for GT latent (needed for tail_residual)
    def velocity(x_t, t, i2v_cond, prompt_embed, null_embed,
                 cfg_scale, flow_score) -> Tensor:
        # wraps CFG batching + cond_mask construction
```

### Phase B — `MobileI2VGVCCPipeline` (DDCM SDE)
Lives in `turbogvcc/exp_mobilei2v/mobilei2v_pipeline.py`. Port of
`TurboDDCMWanPipeline` with the following differences:

1. **Use wrapper.scheduler timesteps** instead of FlowMatch Wan timesteps
   (FlowMatchEulerDiscreteScheduler at shift=3.0).
2. **Re-inject first frame at every step**:
   ```python
   x_t = wrapper.scheduler.step(u_t, t, x_t).prev_sample
   x_t[:, :, :1] = ref_latent          # always overwrite
   ```
   This means the **first latent frame is not part of the SDE state** — only
   frames 1..F-1 (i.e., 2 of 3 latent frames here). DDCM atoms should be
   generated for `F_active = F - 1 = 2` latent frames, not F.
3. **velocity-to-score** conversion uses sigma at the *scheduler's* timestep
   (need to verify against Wan's `velocity_to_score`).
4. **latent_correction**: same mechanism as old code, just at different shape.

### Phase C — Smoke test runner
Lives in `turbogvcc/exp_mobilei2v/run_smoke.py`. One UVG sequence × 1 GOP
of 17 frames (NOT 33). Output: PSNR / MS-SSIM / LPIPS / BPP vs the
Wan-I2V-14B baseline number for the same sequence.

## Open risks

1. **DDCM atom dimensionality** — atoms generated at latent_dim=128 may
   need different K to maintain expressivity. Start with same K=16384,
   M=64; if PSNR is bad, sweep.
2. **First-frame overwriting changes SDE encoder formulation** — when
   encoder finds the noise that *would have* led from GT-latent forward
   trajectory, the first frame is **clamped** to ref each step, so the
   noise we solve for shouldn't touch frame 0. Need to mask out frame 0
   in the atom search (otherwise we waste bits on a fixed slot).
3. **17-frame GOP vs 33** — smaller GOP means more frequent ref-frame
   resets in chained mode, but each GOP's ref-decoded-last drift smaller.
   Should be net-positive for chain stability.
4. **Domain mismatch** — MobileI2V trained mostly on talking-head data
   (CelebV-text). UVG sequences (Jockey, HoneyBee, etc) may not have good
   reconstruction; the I2V conditioning may bend to face priors.
   This is the highest-risk unknown.
5. **GOP boundary seam** — first-frame I2V (no last anchor) already shown
   to be visually problematic in our T2V/I2V experiments; chained-GOP
   mostly hides it but expect some residual seam.

## Bit budget back-of-envelope

Same DDCM config as Wan-I2V baseline (T_sde=30, M=64, K=16384):
- bits_per_fs = M * (ceil(log2(K)) + 1) = 64 * 15 = 960 bits/frame-step
- Wan-I2V: 30 * 9 * 960 = 259,200 bits = 32.4 kB / GOP (33 frames)
  → BPP at 720p = 32400*8 / (33*720*1280) = 0.00852, ~127 kbps @ 16fps
- MobileI2V: 30 * 2 (active frames) * 960 = 57,600 bits = 7.2 kB / GOP (17 frames)
  → BPP at 720p = 7200*8 / (17*720*1280) = 0.00367, ~55 kbps @ 16fps

So **same DDCM config → ~3× lower bitrate** simply from smaller latent.
Plus ref-frame for GOP 0 (~few kB) and tail residual (~few kB).

## Existing artifacts

- `smoke_load.py` — verified model+VAE+text encoder load and one forward
  pass in `sana` conda env (267M params, 0.51s/forward CFG fp16).
- This NOTES.md — design doc.

## Next steps (after user OK)

1. Write `mobilei2v_wrapper.py` (~150 lines)
2. Write `mobilei2v_pipeline.py` (port of TurboDDCMWanPipeline, ~300 lines)
3. Write `run_smoke.py` (~100 lines)
4. Run on Jockey GOP 0 (17 frames) → report PSNR / BPP / time
5. If acceptable, run 3-GOP chained for Jockey to validate chain stability
