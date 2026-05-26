# GVCCTurbo Research Notes

This is the single place for the current GVCCTurbo research design. It
consolidates the SPD, DLC1, DLC2, and GVCC/GVCCTurbo notes that were previously
spread across the README and `spd_repro/`.

## Core View

```text
SPD:      timing -- when should a frequency band participate?
GVCC:     coding channel -- how do bits control the generation trajectory?
AsymFlow: subspace structure -- where should the control signal live?
```

The intended GVCCTurbo direction is:

```text
progressive-resolution trajectory
+ frequency-scheduled activation
+ low-rank structured codebook innovation.
```

In short, GVCC should not spend bits trying to approximate full-rank Gaussian
noise. It should spend bits on the effective low-rank, frequency- and
timestep-dependent structure of the residual.

## SPD x GVCC Mapping

| SPD concept | GVCC counterpart | GVCCTurbo use |
| --- | --- | --- |
| Spectral transform `T_Phi` | DCT/DWT on latent video tensors | Decompose `z_t*`, residual `r_t`, or state `x_t`. |
| Low-frequency early / high-frequency late | Codebook injection schedule | Inject low-frequency atoms early, then open middle/high bands. |
| Spectral noise expansion | Codebook spectral injection | Replace random high-frequency noise with encoder-selected codebook innovation. |
| Optimal resolution schedule | Bit allocation / activation schedule | Decide which timesteps and bands receive bits. |
| Timestep alignment | Energy / variance alignment | Renormalize after masking or expansion so the effective `g_t` is not accidentally changed. |

## Why Multi-Resolution Matters

GVCC has a compact bitstream but high compute cost. Each SDE step runs Wan and
performs codebook residual matching:

```text
x_{t-dt} = x_t - f_t(x_t) * dt + g_t * sqrt(dt) * z_t*
```

The main costs are:

1. Wan forward passes at every step.
2. Full-resolution codebook search / residual matching at every step.
3. GOP-level repetition.

SPD attacks exactly this bottleneck by letting early high-noise steps run at
lower token count. The extra GVCC constraint is determinism: encoder and decoder
must reproduce the same resolution transitions and codebook injections from a
fixed schedule and the transmitted indices/signs.

## Preferred Design: Progressive Latent Resolution

A simple fixed-resolution frequency mask is not the best match to SPD. The
closer translation is progressive latent resolution.

Standard GVCC:

```text
x_t in R^{C x T_l x H_l x W_l}
```

Multi-resolution GVCC:

```text
x_t^{s_i} in R^{C x T_l x (s_i H_l) x (s_i W_l)}
s_1 < s_2 < ... < s_S = 1
```

Within each stage:

```text
x_{t-dt}^{s_i}
  = x_t^{s_i}
    - f_theta^{s_i}(x_t^{s_i}, t) * dt
    + g_t * sqrt(dt) * z_t^{*, s_i}
```

At transition:

```text
x_tau^{s_i} -> x_tilde_tau^{s_{i+1}}
```

The transition should use spectral expansion plus alignment, not ordinary
resize.

## Codebook-Guided Spectral Expansion

SPD transition:

```text
low frequencies:  keep the partially denoised low-resolution state
high frequencies: fill with t * epsilon_high
```

GVCC transition:

```text
low frequencies:  keep the low-resolution GVCC trajectory
high frequencies: start from t * epsilon_high, then add codebook residual
```

The trajectory-consistent basic form is:

```text
high frequencies:
  t * epsilon_high + r_hat_t^H

r_t^H = (1 - t) * x_0^H
```

First implementation:

```text
r_hat_t^H ~= lambda_t * T^{-1}(M_new * T(C_k))

x_tilde_tau^{s_{i+1}}
  = Align[
      T^{-1}(
        Embed(T(x_tau^{s_i}))
        + M_new * T(t * epsilon_new + r_hat_t^H)
      )
    ]
```

This is the main upgrade over SPD:

```text
SPD:       random high-frequency noise
GVCCTurbo: transmitted, target-aware high-frequency codebook innovation
```

Note: `z_k^H` is assumed unit-variance on the high-frequency band so that
it can be used as a stable residual direction after masking. The residual
scale is controlled by `lambda_t` or by norm matching to the target residual.
If the codebook is later changed to magnitude-quantized atoms, this scaling
must be re-calibrated.

## First Implementation Target

Use two stages first:

```text
low-resolution latent stage -> full-resolution latent stage
```

For the TI2V-5B iteration backbone:

```text
full 720p latent ~= 9 x 45 x 80
low bucket       ~= 9 x 30 x 52
```

Do not hard-code these for every backbone. Query wrapper shapes with
`get_latent_shape()` and `get_frame_shape()`.

Conservative schedule:

```text
T = 20, N = 3
codebook steps = 17

steps 0-7:    low-resolution GVCC
transition:   low -> full
steps 8-16:   full-resolution GVCC
steps 17-19:  deterministic ODE tail
```

Also test SPD-derived transition timing. With measured `beta ~= 1.99` and
`t* ~= 0.728`, the split may be closer to:

```text
10 low-resolution codebook steps
7 full-resolution codebook steps
3 deterministic ODE tail steps
```

The actual transition index should be derived from the shifted timestep array.

## V1 Implementation Decisions

**Low-resolution ground truth.** Use explicit low-bucket VAE encode:

```text
x_0^low  = VAEEncode(resize(video, low_bucket))
x_0^full = VAEEncode(video at full bucket)

r_t^low = x_0^low - x_hat_{0|t}^low
```

This costs an extra VAE encode but is more defensible than downsampling
`x_0^full`.

**Stage-specific codebook size.**

```text
K_low  = 4096
K_full = 16384

D_low  = 16 * 30 * 52 = 24960
D_full = 16 * 45 * 80 = 57600
```

`K_low` should be smaller because lower dimension saturates faster.

**I2V tail residual.** Multi-resolution should leave I2V tail residual
correction unchanged because the final stage is full-resolution. Speed gains are
expected to be strongest for T2V and FLF2V; I2V gains may be smaller because the
tail residual dominates transmitted bytes.

## Bitstream Design

The resolution schedule should be fixed by codec configuration and not
transmitted. Encoder and decoder both know:

```text
steps 0-7:    low-resolution GVCC
step 8:       transition to full resolution
steps 8-16:   full-resolution GVCC
steps 17-19:  deterministic ODE tail
```

The bitstream still carries only:

```text
codebook indices + signs
```

First version:

```text
low-resolution codebook:  shape = low latent frame shape
full-resolution codebook: shape = full latent frame shape
```

## Transition Codebook Selection

At transition timestep `t_i`, the transition is:

```text
s_i -> s_{i+1}
```

Let `H_i` denote the newly opened high-frequency band. The key point is that
the codec transition happens on the current trajectory, not on clean GT alone.
The decoder does not have `x_0^{s_i}`. It has the current reproducible noisy
state:

```text
x_{t_i}^{s_i}
```

After expansion, the decoder-reproducible base is:

```text
x_bar_{t_i}^{s_{i+1}}
  = T^{-1}(
      Embed(T(x_{t_i}^{s_i}))
      + t_i * epsilon_{H_i}
    )
```

so the newly opened band is:

```text
x_bar_{t_i}^{H_i} = t_i * epsilon_{H_i}
```

The encoder does have the next-level clean latent:

```text
x_0^{s_{i+1}}
```

and therefore the clean component in the newly opened band:

```text
x_0^{H_i} = T_{H_i}(x_0^{s_{i+1}})
```

The oracle high-frequency state at timestep `t_i` is:

```text
x_{t_i,oracle}^{H_i}
  = (1 - t_i) * x_0^{H_i}
    + t_i * epsilon_{H_i}
```

Therefore the trajectory-consistent residual that codebook should transmit is:

```text
r_{t_i}^{H_i}
  = x_{t_i,oracle}^{H_i} - x_bar_{t_i}^{H_i}
  = (1 - t_i) * x_0^{H_i}
```

This is the cleanest DLC1-codebook target: low bands are inherited from the
current trajectory, while the newly opened band receives the current-timestep
clean high-frequency contribution from encoder-selected codebook bits.

### Candidate A: Band Target

Encode the newly opened band directly:

```text
r_{t_i}^{H_i} = (1 - t_i) * x_0^{H_i}
```

Select transition atoms by:

```text
k_i = argmax_k | < T_{H_i}(C_k), r_{t_i}^{H_i} > |
```

For multi-atom GVCC, use top-`M` by absolute inner product and transmit signs.
The decoder reconstructs:

```text
z_{k_i}^{H_i} = T^{-1}(M_{H_i} * T(C_{k_i}))
```

and applies:

```text
x_{t_i}^{H_i}
  = t_i * epsilon_{H_i}
    + lambda_t * z_{k_i}^{H_i}
```

`lambda_t` can first be set by norm matching against the selected residual, or
absorbed into atom normalization. This is closest to the DLC1 theory: SPD's
random high-frequency expansion supplies only `t_i * epsilon_{H_i}`, while
GVCCTurbo transmits the missing target-aware clean contribution.

### Candidate B: Clean Residual Target

This version matches the more traditional residual-coding intuition:

```text
next-level clean GT
- lifted low-level clean GT
= newly needed clean detail
```

Define a clean lift:

```text
x_tilde_0^{s_{i+1}} = Lift(x_0^{s_i})
```

Then encode:

```text
r_0^{H_i}
  = T_{H_i}(x_0^{s_{i+1}} - x_tilde_0^{s_{i+1}})
```

Transition:

```text
x_{t_i}^{H_i}
  = t_i * epsilon_{H_i}
    + (1 - t_i) * r_hat_0^{H_i}
```

This is useful when `Lift(x_0^{s_i})` produces nonzero content in the newly
opened band. If `Lift` is a strict spectral embedding whose new band is zero,
then:

```text
r_0^{H_i} = T_{H_i}(x_0^{s_{i+1}})
```

and Candidate B collapses to Candidate A up to scaling.

For a real codec, `Lift` must be defined from decoder-reproducible quantities.
If the decoder does not have `x_0^{s_i}`, replace the clean lift with a lift of
the decoded low-level reconstruction or current trajectory. Otherwise the
encoder target and decoder state will be mismatched.

### Candidate C: Model Prior Plus Codebook Residual

If a full-grid model call is affordable at transition, first expand the current
trajectory to full grid and estimate clean:

```text
x_bar_{t_i}^{s_{i+1}}
  = SpectralExpand(x_{t_i}^{s_i})

x_hat_{0|t_i}^{s_{i+1}}
  = x_bar_{t_i}^{s_{i+1}}
    - t_i * u_theta(x_bar_{t_i}^{s_{i+1}}, t_i)
```

Then let the model prior carry the predictable high-frequency mean and let the
codebook transmit only the residual:

```text
r_{t_i}^{H_i}
  = (1 - t_i)
    * T_{H_i}(x_0^{s_{i+1}} - x_hat_{0|t_i}^{s_{i+1}})
```

Decoder transition:

```text
x_{t_i}^{H_i}
  = t_i * epsilon_{H_i}
    + (1 - t_i) * x_hat_{0|t_i}^{H_i}
    + r_hat_{t_i}^{H_i}
```

This is the strongest form, but it costs an additional full-resolution model
call and depends on the stability of `x_hat_{0|t_i}^{H_i}` under the temporary
high-frequency initializer.

The key distinction to preserve in implementation is:

```text
Oracle / analysis target:
  clean residual = x_0^{high} - Lift(x_0^{low})

Real codec transition target:
  trajectory residual = x_{t,oracle}^{high} - Expand(x_t^{low})
```

Use clean residuals for ceilings and basis design. Use trajectory-consistent
residuals for actual encoder-decoder codebook selection.

## Transition Alignment

SPD timestep alignment:

```text
t_tilde_i = (r * t_i) / (1 + (r - 1) * t_i)
r = s_{i+1} / s_i
```

Wan buckets are not always exact scale multiples, e.g. `(30, 52) -> (45, 80)`.
First implementation should ablate:

```text
A: spectral expansion + empirical variance alignment, no timestep re-index
B: spectral expansion + SPD timestep re-indexing
```

Empirical variance alignment:

```text
x_expand = spectral_expand(x_low, z_high)
x_expand = x_expand / x_expand.std() * sigma_like_full_t
```

## Minimal Experiment Matrix

| Method | Resolution | Transition high frequency | Codebook use | Purpose |
| --- | --- | --- | --- | --- |
| GVCC full-res | full only | none | full-band | Original quality and speed reference. |
| MR-GVCC-Random | low -> full | `t_i * epsilon_H` | stage codebook | Reproduce SPD-style acceleration inside GVCC. |
| MR-GVCC-Oracle | low -> full | `(1 - t_i) * x_0^H + t_i * epsilon_H` | stage codebook | Upper bound for correct transition high frequencies. |
| MR-GVCC-Codebook | low -> full | `t_i * epsilon_H + r_hat_t^H` | transition codebook | Core target-aware compression version. |
| MR-GVCC-Prior+Codebook | low -> full | `t_i * epsilon_H + (1 - t_i) * xhat0_t^H + r_hat_t^H` | transition codebook | Test conditional mean plus residual codebook correction. |
| MR-GVCC-Codebook + freq mask | low -> full | codebook + scheduled bands | transition codebook | Full V1/V2 bridge. |

Key comparisons:

```text
MR-GVCC-Random vs. MR-GVCC-Codebook
MR-GVCC-Codebook vs. MR-GVCC-Prior+Codebook
```

## Roadmap

```text
V0:   fixed-resolution frequency mask
      Goal: test whether frequency scheduling alone improves quality.

V1:   S=2 multi-resolution GVCC
      Goal: test whether SPD-style acceleration can be attached to GVCC.

V1.5: codebook spectral expansion
      Goal: show that encoder-selected high-frequency codebook information is
      better than random high-frequency noise for compression.

MR-V2: quality-aware multi-resolution acceleration
       Goal: keep multi-resolution as the main speed source, but explicitly
       solve trajectory mismatch with transition/lift/correction modules.

V2-a: fixed-resolution spectral codebook scheduling
      Goal: diagnostic branch; tests whether codebook bits can be
      frequency-aware without changing trajectory. Stage 0a failed.

V2-b: fewer-NFE auxiliary route
      Goal: reduce Wan forward count after MR-V2 identifies a stable quality
      compensation mechanism.
```

## V1 Status (2026-05-22)

Full UVG ablation done (7 seq × 3 GOP, Wan 2.1-T2V-1.3B, 720p, M=64). Detailed
report at `exp_v1_mr/REPORT.md`.

Headline numbers vs `full_res` baseline (PSNR 28.72 dB, LPIPS 0.1138):

- `mr_random`: −1.05 dB PSNR, +0.0208 LPIPS, −6.3% BPP, 1.45× enc / 1.38× dec
- `mr_codebook`: −0.65 dB PSNR, +0.0085 LPIPS, −0.4% BPP, 1.40× enc / 1.38× dec

Hypothesis confirmed direction-wise: target-aware codebook high-freq (current
form `x_H = t_i * z_k^H`) recovers ~40% of the PSNR gap left by random.
Not yet a pure win — paying 0.65 dB for 1.4× speedup. Next candidates:
`transition_step` sweep, trajectory-consistent band target
`x_H = t_i * epsilon_H + r_hat_t^H`, prior+residual form
`x_H = t_i * epsilon_H + (1 - t_i) * x_hat_{0|t_i}^H + r_hat_t^H`, and
transition `M_trans` tuning.

## V1 Final Takeaway (2026-05-23)

Full 8-configuration matrix plus 4 high-frequency modes was run on Bosphorus
and HoneyBee, 3 GOPs each, with the Wan 2.1-T2V-1.3B backbone. The final V1
conclusion is:

```text
V1 is not "more stages is better."
The viable form is progressive acceleration inside the model's reliable
resolution range, plus a small additive codebook correction at transition.
```

Three decisive findings:

1. **Resolution floor matters more than stage count.** Stages below 480p are
   out-of-distribution for Wan 1.3B and cause catastrophic trajectory drift.
   Moving the floor to 480p changes:

```text
S=3 random: 24.88 -> 27.51 dB  (+2.63 dB)
S=4 random: 23.40 -> 27.51 dB  (+4.11 dB)
```

2. **DLC1-codebook is valid.** Additive transition correction works:

```text
x_H = t * epsilon_H + (1 - t) * z_H*
```

It recovers roughly 40-60% of the oracle transition gap, depending on the
stage schedule. This makes it a real building block, not just a conceptual
patch.

3. **Wan 1.3B has a progressive-trajectory ceiling.** Even joint-oracle S=2
   reaches only 28.89 dB, about 1.26 dB below full-resolution on the
   Bosphorus/HoneyBee subset. Later 14B checks suggest this is not just
   backbone capacity; the real problem is trajectory alignment across
   resolutions.

Pareto operating points:

| Use | Configuration | PSNR | Delta vs full_res | Speedup | BPP |
| --- | --- | --- | --- | --- | --- |
| Quality first | S=2 dlc1_codebook | 28.72 | -1.44 dB | 1.34x | +0.4% |
| Speed first | S=3 480p-floor random | 27.51 | -2.64 dB | 1.51x | +0% |
| Balanced | S=3 480p-floor dlc1_codebook | 27.76 | -2.39 dB | 1.41x | +12% |

Next implication: use V1 as a validated acceleration/transition design. Do
not spend the next iteration trying to rescue <480p stages on 1.3B. The next
quality work should keep multi-resolution as the speed source and explicitly
solve trajectory alignment.

## MR-V2: Return to Multi-Resolution as the Main Route

After V2-a Stage 0, the direction should move back to multi-resolution. V2-a
is useful as a falsification: changing codebook frequency while keeping the
same full-resolution Wan trajectory does not create meaningful acceleration,
and direct band-limited innovation breaks the SDE distribution. The only route
that directly reduces DiT token/FLOP cost is still progressive resolution.

The core MR-V2 question is therefore not:

```text
Can we avoid multi-resolution trajectory mismatch?
```

V1 already showed that the mismatch exists. The correct question is:

```text
Can we make the multi-resolution trajectory close enough to full-resolution
reconstruction while preserving most of the token saving?
```

This reframes the problem:

```text
multi-resolution = acceleration backbone
trajectory alignment = quality recovery
codebook correction = compression-side target awareness
```

### MR-V2 Principles

1. **Keep the resolution floor.** The 480p-floor result is decisive. For Wan
   1.3B, stages below 480p are OOD and should not be part of the default
   method.

2. **Do not treat transition error as high-frequency only.** V1 shows that
   DLC1 high-band correction helps, but the remaining gap is a full trajectory
   mismatch. MR-V2 should allow full-state or low/mid/high transition
   correction, not only newly opened high-frequency bands.

3. **Preserve the stochastic carrier.** The DLC1 and V2-a lessons agree:
   target-aware codebook information should be added/replaced inside a noise
   carrier, not replace the noise process with a spectrally degenerate signal.

4. **Make the lift trajectory-aware.** A plain resize/spectral expansion is
   not enough. The transition operator should approximate the full-resolution
   state distribution at the same timestep.

### MR-V2 Candidate Modules

#### A. Trajectory-Consistent Full-State Transition Codebook

Instead of coding only the newly opened high-frequency band, code the residual
between the decoder-reproducible expanded state and a full-resolution target:

```text
x_base^full(t_i) = Expand(x_low(t_i))
r_t^full = x_target^full(t_i) - x_base^full(t_i)
```

Then select a codebook correction on all bands, or on measured high-error
bands:

```text
z_corr = CodebookSelect(r_t^full)
x_transition^full = x_base^full + alpha_t * z_corr
```

The key change from V1 is that the residual is trajectory-level, not only
clean high-frequency detail. This directly attacks the stable PSNR gap.

Possible targets:

```text
cheap target:     x_0^full - xhat_{0|t}^full  converted to state correction
teacher target:   full-resolution GVCC trajectory state at t_i
oracle target:    (1 - t_i) x_0^full + t_i eps_full
```

The teacher target is expensive for the encoder but free for the decoder. It
is acceptable as an experiment because compression encoders are allowed to be
heavier, and it gives a clear upper bound on trajectory alignment.

#### B. Deterministic Learned / Calibrated Lift

Introduce a shared deterministic lift:

```text
L_phi(x_low(t), t, s_i -> s_{i+1}) -> x_lift^full(t)
```

Then transition becomes:

```text
x_base^full(t_i) = L_phi(x_low(t_i), t_i)
x_transition^full = x_base^full(t_i) + codebook_residual
```

The lift can start simple:

```text
1. spectral resize + variance alignment
2. affine per-band calibration
3. PCA / linear residual lift
4. small convolutional latent bridge
```

This is the AsymFlow-style low-res-to-full-res lift idea, but applied as a
shared codec transform rather than changing Wan's flow parameterization.

#### C. Post-Transition Recovery Window

If the first full-resolution step after transition is where mismatch explodes,
allocate a short recovery window:

```text
transition at step k
steps k, k+1: stronger correction / larger M_trans
later steps: normal GVCC
```

This keeps speedup from early low-resolution steps while spending bits exactly
where the trajectory rejoins the full-resolution manifold.

#### D. Transition Schedule Sweep as a Quality-Speed Dial

MR-V2 should treat transition timing as the main Pareto knob:

```text
earlier transition -> more quality, less speed
later transition   -> more speed, more mismatch
```

The next sweep should be S=2 / 480p->720p with transition steps:

```text
k = 4, 6, 8, 10, 12
```

and the same transition correction module. This will identify whether the
quality gap is dominated by transition timing or by the lift/correction
itself.

### MR-V2 Minimal Experiment Matrix

Keep the first MR-V2 matrix small and diagnostic:

| Method | Resolution path | Transition | Purpose |
| --- | --- | --- | --- |
| Full GVCC | 720p only | none | Quality reference. |
| V1 S=2 random | 480p -> 720p | `t eps_H` | Acceleration reference. |
| V1 S=2 dlc1_codebook | 480p -> 720p | `(1-t) z_H + t eps_H` | Current best V1. |
| MR-V2 full-state codebook | 480p -> 720p | full-state residual correction | Test whether the remaining gap is not high-frequency-only. |
| MR-V2 calibrated lift | 480p -> 720p | deterministic lift + residual correction | Test if shared lift reduces bit burden. |
| MR-V2 teacher transition | 480p -> 720p | teacher-state residual correction | Upper bound on trajectory alignment. |

Success criterion:

```text
recover at least half of the remaining S=2 gap beyond V1 dlc1_codebook
while retaining >= 1.3x decode speedup.
```

If MR-V2 can reduce the S=2 gap from roughly 1.4 dB to below 0.7 dB at similar
speed, multi-resolution becomes the main GVCCTurbo story again. If not, the
paper should present multi-resolution as a speed-quality knob rather than a
near-lossless turbo path.

### Diagnostic: Low Stage Target From Full-Resolution Low-Pass Latent

Motivation:

```text
x0_480 = VAE_480(Resize(video))
x0_LP720 = T_L(VAE_720(video))
```

The static diagnostic suggests:

```text
x0_480 != x0_LP720
```

Therefore the low-resolution stage may be optimizing toward the wrong clean
target for a later 720p transition. Test a low-stage target override:

```text
low-stage clean target = x0_LP720
instead of
low-stage clean target = x0_480
```

This does not change token count or the 480p Wan forward shape. It only changes
the encoder-side residual matching target used to select low-stage codebook
atoms.

Hypotheses:

```text
If PSNR improves:
  static VAE mismatch is a major contributor.

If PSNR does not improve:
  dynamic vector-field mismatch dominates; 480p Wan does not follow
  T_L(720p trajectory) even when residual matching uses x0_LP720.
```

Run the diagnostic in three variants:

| Variant | Low-stage target | Purpose |
| --- | --- | --- |
| baseline | native `VAE_480(Resize(video))` | Current V1 behavior. |
| LP raw | `T_L(VAE_720(video))` | Direct test of VAE target mismatch. |
| LP std/norm aligned | scaled `T_L(VAE_720(video))` | Control for DCT-grid / latent-scale mismatch. |

The implementation flag is:

```bash
--low_target_from_full
--low_target_align none|std|norm
```

Suggested first matrix:

```text
method = mr_random
method = mr_dlc1_codebook
transition_step = 8
resolutions = 480x832,720x1280
sequences = Bosphorus HoneyBee
num_gops = 3
low_target_align = none, std, norm
```

Readout:

```text
1. final PSNR / LPIPS / BPP / speed
2. static logs: ||VAE_low||, ||T_L(full)||, raw_delta, cosine, aligned_delta
3. transition residual MSE if available
```

Important interpretation:

```text
This is not a decoder-side bitstream change. It is an encoder diagnostic.
If it helps, MR-V2 should use a shared low-pass latent definition or calibrated
lift. If it fails, MR-V2 must attack the dynamic vector-field mismatch directly.
```

### Result: Full-Latent-Consistent Multi-Resolution GVCC

The low-target diagnostic strongly supports a revised MR design:

```text
low-stage target = T_L(x0_720)
instead of
low-stage target = x0_480 = VAE_480(Resize(video))
```

This is the key change. The low-resolution stage should not reconstruct its own
native 480p VAE latent. It should reconstruct the low-frequency projection of
the final 720p latent. This puts the low-resolution and full-resolution stages
in the same latent coordinate system.

Diagnostic result on Bosphorus + HoneyBee, 3 GOPs each, S=2 DLC1-codebook:

| Variant | PSNR | LPIPS | BPP | Interpretation |
| --- | ---: | ---: | ---: | --- |
| baseline | 28.71 | 0.0931 | 0.004811 | Native 480p VAE target. |
| LP raw | **29.86** | **0.0794** | 0.004811 | Use `T_L(x0_720)` directly. |
| LP std | 29.37 | 0.0831 | 0.004811 | Scale-aligned low-pass target. |
| LP norm | 29.38 | 0.0829 | 0.004811 | Norm-aligned low-pass target. |

The raw low-pass target improves by +1.15 dB over the native-480p baseline at
the same BPP and similar runtime. The std/norm aligned variants are better than
baseline but worse than raw, suggesting that the full-resolution low-pass latent
scale is not a bug to be normalized away.

This revises the interpretation of the old V1 ceiling. A large part of the
multi-resolution gap came from a wrong low-stage clean target:

```text
x0_480 != T_L(x0_720)
```

The old pipeline made the low stage move toward a native 480p latent world and
then tried to connect that state to a 720p trajectory. Even oracle high-frequency
transition correction could not fix the low-frequency base mismatch. With the
new target, spectral expansion is finally a meaningful upsampling operation:

```text
Expand(x_t^low) ~= T_L^{-1}(T_L(x_t^720))
```

The resulting method can be framed as:

```text
Full-latent-consistent multi-resolution GVCC
```

It has three components:

1. Low stages use the full-resolution latent's low-pass projection as the clean
   target, not native low-resolution VAE latents.
2. Spectral expansion connects states that already live in the final latent
   coordinate system.
3. DLC1 additive transition codebook remains useful as high-frequency detail
   compensation, but it is no longer the module responsible for fixing the main
   trajectory gap.

Short version:

```text
Previous multi-resolution failed because the low-resolution stage chased the
wrong target. The revised design lets low resolution chase the low-frequency
part of the final full-resolution latent, so progressive resolution can keep
its speed benefit while staying much closer to full-resolution quality.
```

### Codebook Implementation Fix: Vectorized Atom Generation

The apparent 3D-codebook speedup was traced to an implementation bottleneck in
the original per-frame baseline: deterministic atoms were generated by many
small `torch.randn(D)` calls. The cost was dominated by Python/kernel-launch
overhead, not by a fundamental advantage of joint video atoms.

Vectorizing the original per-frame baseline fixes this directly:

| Mode | PSNR | BPP | Enc(s) | Dec(s) | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline, sequential | 30.08 | 0.00483 | 78.9 | 46.4 | Original per-atom RNG path. |
| baseline, vectorized | 30.15 | 0.00483 | 33.9 | 34.5 | Same method, batched atom generation. |
| 3D codebook, BPP-neutral | 28.50 | 0.00483 | 36.9 | n/a | Lower quality and no speed advantage after the fix. |

Result:

```text
encode speedup: 78.9 -> 33.9 s = 2.33x
decode speedup: 46.4 -> 34.5 s = 1.35x
total speedup: about 1.83x
```

This is a free implementation win: same BPP, same PSNR up to random-codebook
perturbation, and no method-level tradeoff. The temporary 3D codebook direction
is therefore deprecated. Its measured speed came from reducing the number of
small RNG calls, which the vectorized baseline now fixes without sacrificing
quality.

All subsequent MR / GVCCTurbo experiments should use the vectorized per-frame
codebook path as the default baseline.

### Model-Call Skipping Diagnostic

Goal:

```text
Keep the original SDE/codebook step count, but call Wan on only a subset of
timesteps. Intermediate steps reuse or extrapolate model predictions, while the
codebook channel continues to inject one correction per SDE step.
```

This is more conservative than reducing `T` directly:

```text
T=10 fewer steps:
  fewer Wan calls
  fewer trajectory updates
  fewer codebook correction opportunities

T=20 with model-call skipping:
  fewer Wan calls
  same trajectory update density
  same codebook correction density
```

#### Key Result: Cache Clean Prediction, Not Velocity

HoneyBee 1 GOP, vectorized full-resolution baseline:

| Mode | PSNR | LPIPS | BPP | Enc(s) | Dec(s) | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vectorized baseline | 30.15 | 0.0584 | 0.00483 | 33 | 32 | 65 |
| vectorized + `u_cache` p=2 | 28.25 | 0.0808 | 0.00483 | 20 | 19 | 40 |
| vectorized + `x0_cache` p=2 | 29.81 | n/a | 0.00483 | n/a | n/a | 45 |

The model-call skipping speedup is real:

```text
u_cache p=2:
  65 / 40 = 1.63x over vectorized baseline

x0_cache p=2:
  65 / 45 = 1.43x over vectorized baseline
```

but the cached quantity matters. `u_cache` is too crude:

```text
u_cache p=2:
  PSNR:  -1.90 dB
  LPIPS: +38%

x0_cache p=2:
  PSNR:  -0.34 dB
```

Interpretation:

```text
Directly reusing u_t assumes local velocity is constant.
That approximation is too crude for the stochastic GVCC trajectory.
Codebook noise changes x_t between steps, so u_theta(x_t,t) drifts enough for
the error to accumulate across 17 SDE steps.
```

The better approximation is to cache the clean prediction:

```text
x0_cache = x_t - t * u_theta(x_t,t)

u_t(skipped step k)
  = (x_t_current - x0_cache) / t_k
```

This keeps the cached clean endpoint but recomputes the velocity from the
current state and current timestep. It is state-aware; `u_cache` is not.

Flow-matching intuition:

```text
x_t = (1 - t) * x_0 + t * eps
u_t ~= (x_t - x_0) / t
```

If the model's clean prediction is stable across nearby timesteps, then reusing
`x0_hat` and recomputing `u = (x_t - x0_hat) / t` is a good local
approximation. This matches the diagnostic observation that `xhat_0` changes
much more smoothly than velocity.

Period sweep:

| Period | u-cache PSNR | x0-cache PSNR | x0-cache gain |
| ---: | ---: | ---: | ---: |
| 2 | 28.25 | 29.81 | +1.56 |
| 3 | 26.57 | 29.38 | +2.81 |
| 4 | unstable | 28.84 | >3 |

`u_cache` is brittle: when the skipped span increases, cached velocity quickly
becomes stale. `x0_cache` degrades gracefully because every skipped step still
uses the current `x_t`.

Pareto points on HoneyBee 1 GOP:

| Tolerance | Recommended config | PSNR | Speedup |
| --- | --- | ---: | ---: |
| <= -0.5 dB | `x0_cache` p=2 | 29.81 | 1.43x |
| <= -1.0 dB | `x0_cache` p=3 | 29.38 | 1.59x |
| <= -1.5 dB | `x0_cache` p=4 | 28.84 | 1.88x |

Comparison to other speed routes:

| Config | PSNR | Total time | Speedup vs vec | Delta PSNR |
| --- | ---: | ---: | ---: | ---: |
| vectorized full-res | 30.15 | 65 | 1.00x | 0 |
| `x0_cache` p=2 | 29.81 | 45 | 1.43x | -0.34 |
| `u_cache` p=2 | 28.25 | 40 | 1.63x | -1.90 |
| vectorized + T=16 | 28.92 | about 58 | 1.12x | -1.23 |
| vectorized + full-latent-consistent MR | 29.58 | 53 | 1.23x | -0.57 |

```text
x0_cache p=2 beats V1 lp_raw on this GOP:
  higher quality and faster runtime.
```

This also beats direct fewer-NFE at similar runtime:

```text
T=16 reduces Wan calls but also reduces SDE updates and codebook bits.
x0_cache keeps all 17 SDE/codebook steps and only skips model evaluations.
```

Conclusion:

```text
Clean-prediction caching is a strong new GVCCTurbo candidate.
It preserves trajectory/codebook density while sparsifying expensive Wan calls.
```

Remaining caveat: this is one sequence and one GOP. It needs multi-sequence
validation, especially on high-motion sequences.

#### MR + X0-Cache: Transition Warmup Is the Best Candidate

The full-latent-consistent MR path and `x0_cache` model-call skipping were then
combined:

```text
method = mr_dlc1_codebook
low_target_from_full = true
low_target_align = none
model_call_skip = x0_cache
model_call_period = 2
```

Baseline comparison on HoneyBee 1 GOP:

| Config | PSNR | MS-SSIM | LPIPS | BPP | Enc | Dec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full-res baseline | 30.15 | 0.9452 | 0.0584 | 0.00483 | 32 | 33 |
| MR lp_raw | 29.63 | 0.9371 | 0.0612 | 0.00511 | 26 | 26 |
| MR lp_raw + `x0_cache` p=2 | 29.35 | 0.9326 | 0.0660 | 0.00511 | 20 | 19 |
| full-res + `x0_cache` p=2 | 29.81 | 0.9401 | 0.0610 | 0.00483 | 22 | 23 |

The penalties are nearly additive:

```text
MR lp_raw alone:      -0.52 dB
x0_cache alone:       -0.34 dB
expected combined:    -0.86 dB
observed combined:    -0.80 dB
```

This means MR and `x0_cache` mostly act independently:

```text
MR saves latent-token compute.
x0_cache saves Wan-call count.
```

Three transition-aware variants were tested:

| Variant | PSNR | MS-SSIM | LPIPS | Enc | Dec | Delta vs C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C: skip all stages, no warmup | 29.35 | 0.9326 | 0.0660 | 20s | 19s | -- |
| V1: transition warmup | 29.47 | 0.9337 | 0.0658 | 19s | 18s | +0.12 |
| V2: stage 0 only skip | 29.57 | 0.9359 | 0.0620 | 24s | 24s | +0.22 |
| V3: stage 1 only skip | 29.41 | 0.9341 | 0.0643 | 21s | 20s | +0.06 |

Key finding:

```text
transition warmup is effectively free.
```

Without warmup, full stage pattern is:

```text
Wan at 8, 10, 12, 14, 16
skip 9, 11, 13, 15
```

With one-step warmup after transition:

```text
Wan at 8, 9, 11, 13, 15
skip 10, 12, 14, 16
```

The number of Wan calls is unchanged: `5 Wan + 4 skip` in both cases. The only
change is the skip phase. Delaying the first skipped step after resolution
transition gives the clean-prediction cache one full-resolution Wan call to
stabilize, recovering +0.12 dB at no compute cost.

Stage responsibility diagnostic:

```text
stage 0 only skip: 29.57
stage 1 only skip: 29.41
```

Keeping full-res stage exact recovers much more quality than keeping low-res
stage exact. Full-res skipped calls are therefore more expensive per step than
low-res skipped calls, as expected.

Current full-UVG candidate:

```bash
mr_dlc1_codebook
--low_target_from_full
--low_target_align none
--vectorized_atom_gen
--model_call_skip x0_cache
--model_call_period 2
--transition_warmup
```

HoneyBee 1-GOP operating point:

```text
PSNR 29.47 vs 30.15 baseline  (-0.68 dB)
BPP  0.00511 vs 0.00483       (+5.8%)
Dec  18s vs 33s               (1.83x decoder speedup)
Enc  19s vs 32s               (1.68x encoder speedup)
```

This passes the local `29.4+ dB` criterion and should be the next full-UVG run.

#### Full-UVG result (2026-05-24)

The config above was run on full UVG 7-seq × 3 GOPs × 33 frames at 720p:

```text
                          PSNR    LPIPS    BPP       enc   dec
V1 lp_raw alone (no skip) 28.50   0.1161   0.00511   24s   25s
+ x0_cache p=2 + warmup   28.45   0.1193   0.00511   18s   19s  ← FINAL method
```

Combined-vs-paper-baseline (full_res @ 28.72 / 0.00483 / 80s enc / 50s dec):

```text
ΔPSNR  -0.27 dB
ΔBPP   +5.8%
encoder speedup 4.4x  (largely from vectorized atom gen baked in)
decoder speedup 2.6x
```

Per-sequence: skipping is free or slightly positive on Beauty / Bosphorus /
ShakeNDry / YachtRide; costs −0.13 to −0.20 dB on the high-motion sequences
HoneyBee / Jockey / RSG. See `MEMORY.md` for the current result index. The
HoneyBee 1-GOP smoke pessimism (Δ−0.16 dB above) overestimates the aggregate
cost because HoneyBee is one of the most skip-sensitive sequences.

### New Direction: Asymmetric Compute Compression

The recent results suggest a stronger framing than symmetric acceleration:

```text
encoder may spend more compute,
decoder should spend much less Wan compute.
```

This is natural for video compression. Encoding can be offline or server-side,
while decoding is latency- and device-constrained. Therefore GVCCTurbo does not
need encoder and decoder to run the same number of Wan calls. It only needs the
decoder trajectory to be reproducible from the bitstream.

Core question:

```text
Can the encoder use full Wan computation and ground-truth access to transmit a
small number of extra codebook corrections that let the decoder skip many Wan
calls while preserving reconstruction quality?
```

This changes the objective from symmetric solver acceleration to:

```text
rate-assisted decoder compute reduction
```

or:

```text
asymmetric GVCC-Turbo
```

#### Symmetric Skip vs Asymmetric Skip

Symmetric skip:

```text
encoder skips Wan calls
decoder skips Wan calls
encoder selects codebook using cached / approximate predictions
```

Problem: the encoder itself no longer knows the true full-Wan trajectory at the
skipped steps, so the selected residual may be based on a poor approximation.
The skip-alternate result shows this clearly: speed improves, but naive
velocity cache creates a large trajectory error.

Asymmetric skip:

```text
encoder runs full Wan trajectory
decoder runs sparse Wan trajectory
encoder observes the decoder's skipped-step approximation error
encoder sends codebook correction to compensate that error
```

This is more codec-like. The encoder is allowed to be stronger than the
decoder, because the bitstream transmits the missing trajectory information.

#### Candidate Formulation

Let the encoder compute the full trajectory state:

```text
x_i^E, u_i^E = Wan(x_i^E, t_i)
```

The decoder, at a skipped Wan step, uses a cached approximation:

```text
u_i^D = Cache(x_i^D, t_i)
```

without calling Wan. This produces a base decoder update:

```text
x_{i+1,base}^D
  = Step(x_i^D, u_i^D, z_i)
```

The encoder can simulate this decoder base trajectory and compare it against a
better target. The target can be:

1. the encoder full-Wan state `x_{i+1}^E`;
2. the ground-truth clean latent residual `x_0 - xhat_{0|t_i}^D`;
3. a teacher correction between full-Wan and cached-Wan velocity:

   ```text
   delta u_i = u_i^E - u_i^D
   ```

The extra transmitted correction can then be:

```text
c_i^* = CodebookSelect(target_error_i)
```

and the decoder step becomes:

```text
x_{i+1}^D
  = Step(x_i^D, u_i^D, z_i)
    + alpha_i c_i^*
```

or, equivalently, the correction can be injected through the usual stochastic
innovation channel:

```text
z_i^* = z_i + lambda_i c_i^*
```

The important point is that skipped Wan calls are replaced by transmitted
trajectory information, not by a blind cache alone.

#### Why This Has Better Paper Shape

This directly expresses a rate-compute-distortion tradeoff:

```text
reduce decoder Wan calls
increase BPP slightly
preserve reconstruction quality
```

It is more specific to compression than ordinary diffusion acceleration. A
generation sampler cannot see the target video, but a codec encoder can. GVCC's
codebook channel is exactly the mechanism for transmitting this target-aware
trajectory correction.

One-sentence thesis:

```text
GVCCTurbo uses extra transmitted codebook innovations to compensate for
decoder-side model-call skipping, trading a small rate increase for a large
decoder compute reduction.
```

#### Minimal Experiment Matrix

Use HoneyBee 1 GOP first, vectorized per-frame codebook as the baseline:

| Method | Encoder Wan calls | Decoder Wan calls | Extra bits | Purpose |
| --- | ---: | ---: | ---: | --- |
| full GVCC | 17 | 17 | 1.0x | Baseline. |
| symmetric u-cache skip | 9 | 9 | 1.0x | Existing negative result. |
| decoder-only u-cache skip + no correction | 17 | 9 | 1.0x | Isolate asymmetric teacher setup. |
| decoder-only x0-cache skip + no correction | 17 | 9 | 1.0x | Better cache baseline. |
| asymmetric skip + correction M_extra | 17 | 9 | 1.2x / 1.5x / 2.0x | Main hypothesis. |
| asymmetric skip + correction only on skipped steps | 17 | 9 | small | Most efficient version. |

The decisive comparison is:

```text
symmetric skip vs asymmetric skip + correction
```

If asymmetric correction recovers much of the `-1.90 dB` loss with modest BPP,
then the method has a clear codec contribution.

#### Correction Targets to Try

1. **State residual correction.**

   ```text
   r_i^x = x_{i+1}^E - x_{i+1,base}^D
   ```

   Most direct, but it corrects state after the step and may change noise scale.

2. **Clean residual correction.**

   ```text
   r_i^0 = x_0 - xhat_{0|t_i}^D
   ```

   Closest to existing GVCC residual matching; easiest to reuse codebook logic.

3. **Velocity residual correction.**

   ```text
   r_i^u = u_i^E - u_i^D
   ```

   More solver-like. The decoder can use:

   ```text
   u_i^D <- u_i^D + beta_i \hat r_i^u
   ```

4. **Noise-channel correction.**

   Keep the SDE form and add the extra correction as part of `z_i^*`:

   ```text
   z_i^* = z_i + lambda_i \hat r_i
   ```

   This is safest for bitstream compatibility but may be less direct.

#### Open Design Choices

1. **Which steps should skip Wan?**

   Start with every other SDE step, then test less aggressive schedules:

   ```text
   skip 1/2, skip 1/3, early-only, mid-only, late-only
   ```

2. **Where should extra bits go?**

   Only skipped steps should receive correction first. If needed, allocate more
   bits to high-error skipped steps:

   ```text
   M_extra(i) proportional to ||target_error_i||
   ```

3. **Does encoder also transmit normal GVCC atoms on skipped steps?**

   First version: yes, keep original codebook steps unchanged and add a second
   correction stream only for skipped Wan calls. Later versions can merge the
   two streams.

4. **How to maintain decoder determinism?**

   The skip schedule and correction scaling must be fixed or signaled. The
   decoder should not need ground truth or encoder-only quantities.

#### Relationship to Existing Findings

```text
Vectorized codebook:
  free implementation speedup, now the default.

Full-latent-consistent MR:
  reduces token cost but introduces trajectory mismatch.

Symmetric skip:
  reduces Wan calls but creates large trajectory error.

Asymmetric skip + transmitted correction:
  directly targets the remaining bottleneck: decoder Wan calls.
```

This direction may become the main GVCCTurbo story if it can achieve:

```text
decoder speedup >= 1.5x over vectorized baseline
PSNR loss <= 0.3-0.5 dB
BPP increase <= 20-50%
```

### STATUS 2026-05-24 — Asymmetric Direction Closed

Both branches above were tested empirically; the current short index is in
`MEMORY.md`.

**Branch 1: bridge codebook (encoder transmits per-step correction).** Tested with `model_call_skip="x0_correction"`, both `e_x0` and `Δx_t` signals, with full-res / 2×/5×/10× downsampled correction codebook, and M_corr swept 16/64/256. **All variants saturate at +0.08 dB over plain x0_cache for +13–190% BPP.** Random sign-bit codebook capacity √(M/D) is the binding constraint; signal choice and dimension are irrelevant. Code removed; keep as a negative result.

**Branch 2: adaptive skip gate (encoder runs Wan + transmits 1-bit/step skip mask).** Tested with `model_call_skip="adaptive_x0_cache"` + per-stage `--max_consecutive_skips` cap. Three crystallized findings:

- **x0-cache effective radius = 1 step.** Cap≥2 enables consecutive skips, but the moment the cap actually triggers (thr=0.15) quality collapses (−0.39 dB for −1s decoder). Clean-endpoint cache is locally valid for one skipped step only, not for recursive multi-step reuse.
- **Adaptive cap=1 cannot exceed fixed period=2.** With cap=1, adaptive selection degenerates to "gated alternating" — quality ≥ fixed, speed ≤ fixed. Best point (thr=0.30 cap=1 = 29.48 dB / 19s) ties fixed warmup (29.47 / 18s).
- **Single-step err naturally limits consecutive skips.** At thr≤0.10 on HoneyBee, cap=1 / cap=2 / cap=[1,2] give identical results — adaptive never *wants* the 2nd consecutive skip because err crosses threshold. Cap is a safety net against threshold misconfiguration, not a control surface.

**Acceleration target hit by the fixed-schedule cousin instead.** The bullet above ("decoder speedup ≥ 1.5×, PSNR loss ≤ 0.3-0.5 dB, BPP increase ≤ 20-50%") is *over-satisfied* by `mr_dlc1_codebook + low_target_from_full + model_call_skip x0_cache --model_call_period 2 --transition_warmup` (no asymmetric encoder cost, no extra BPP):

```text
decoder: 50s → 19s  = 2.6× vs paper baseline / 1.32× vs lp_raw alone
PSNR:    28.72 → 28.45  = Δ−0.27 dB
BPP:     0.00483 → 0.00511  = +5.8% (MR codebook only, no asymmetric overhead)
```

Full validation is summarized in `MEMORY.md`. Adaptive gate retained as `--model_call_skip adaptive_x0_cache --max_consecutive_skips 1` for robustness mode (paper sidebar, not main table).

**Paper framing:** *"Clean-endpoint caching is locally valid for one skipped step, but not stable under recursive multi-step reuse. The fixed-period-2 alternating schedule + transition warmup is empirically near-optimal under this caching scheme."*

## V2-a: Spectral Codebook Scheduling

### V2-a Stage 0 Result: Band-Limited Innovation Is Falsified

Stage 0 gave a decisive negative result:

| Mode | PSNR | LPIPS | Interpretation |
| --- | ---: | ---: | --- |
| `full_band_baseline` | 30.09 | 0.0786 | Original GVCC. |
| `dct_all_band_control` | 30.19 | 0.0776 | V2-a DCT code path with all bands open; matches baseline. |
| `stage0_masked_random` | 16.48 | 0.6025 | Three-stage band schedule; catastrophic failure. |

This separates implementation from design:

1. The V2-a code path is correct. The all-band DCT control matches the
   baseline, so the failure is not a decoder mismatch or indexing bug.
2. The Stage 0 design is wrong. Selecting and injecting a band-limited
   innovation makes `z_t*` spectrally non-white, and the next Wan forward sees
   an OOD state.

The falsified assumption was:

```text
pre-selection band mask should be safer than V0 post-hoc mask
```

The SDE update does not care how `z_t*` was obtained:

```text
x_{t-dt}
  = x_t
    - f_t(x_t) * dt
    + g_t * sqrt(dt) * z_t*
```

It only sees the distribution of the injected innovation. If `z_t*` is
band-limited, the SDE noise no longer has the full-spectrum structure expected
by the pretrained Wan trajectory. This reproduces the V0 failure mode.

Corrected principle:

```text
codebook bits may be band-aware, but the injected SDE innovation must remain
spectrum-compatible with the pretrained model.
```

The next V2-a design should preserve a full-spectrum noise carrier while using
codebook bits only to control the active band.

V2-a keeps the full-resolution GVCC denoising trajectory unchanged. It does not
change Wan forward resolution and does not perform cross-bucket spectral
expansion. Only the codebook residual matching and injected atom are scheduled
by frequency band.

This branch was originally attractive because it avoids the V1 failure mode:

```text
low-resolution VAE encode
+ spectral expansion
+ intermediate low-bucket model forward
= cross-resolution trajectory mismatch
```

Instead, every SDE step still runs on the full latent grid:

```text
x_t in R^{C x T_l x H_l x W_l}
```

The active band changes with timestep:

```text
early steps:   low
middle steps:  low + mid
late steps:    low + mid + high
```

Let `B_i` be the active band at timestep `t_i`, and let `M_{B_i}` be its
frequency mask. The residual used for codebook selection becomes:

```text
r_t^{B_i}
  = T^{-1}(
      M_{B_i} * T(x_0 - x_hat_{0|t})
    )
```

or equivalently in transform space:

```text
T(r_t^{B_i})
  = M_{B_i} * T(x_0 - x_hat_{0|t})
```

The selected atom is also restricted to the active band:

```text
z_t*
  = T^{-1}(
      M_{B_i} * T(C_k)
    )
```

For multi-atom GVCC, select top-`M` atoms using the band-limited inner product:

```text
k = argmax_k | < M_{B_i} * T(C_k), M_{B_i} * T(r_t) > |
```

The original Stage 0 reconstructed and normalized the combined band-limited
innovation before applying the usual GVCC SDE update:

```text
x_{t-dt}
  = x_t
    - f_t(x_t) * dt
    + g_t * sqrt(dt) * z_t*
```

This exact band-limited injection has now been falsified by experiment. It is
kept here as the negative control, not as the recommended V2-a path.

The goals are different from V1. V2-a is not expected to substantially reduce
decoder Wan forward time, because all steps remain full-resolution. Its goals
are:

1. reduce encoder codebook search cost by comparing only active-band
   coefficients;
2. avoid spending bits on high-frequency atoms during steps where high
   frequencies are noise-dominated;
3. improve reconstruction quality at the same BPP through frequency-aware bit
   allocation.

This is the cleanest SPD-to-GVCC translation after V1:

```text
SPD:  when should each frequency participate?
GVCC: which codebook information controls the trajectory?
V2-a: when should each frequency band receive codebook bits?
```

### V2-a Implementation Stages

Do not start by training a complex codebook. Split the problem into schedule
effect and codebook-learning effect.

```text
Question A: does frequency scheduling itself help?
Question B: does a calibrated band-specialized codebook help beyond masking?
```

The original sequence was:

```text
Stage 0: masked random codebook
Stage 1: calibrated low/mid/high codebooks
Stage 2: timestep-specific band codebooks
```

After the Stage 0 result, update the sequence to:

```text
Stage 0:  all-band DCT control                     [validated]
Stage 0a: band-limited masked random codebook      [falsified]
Stage 0b: spectrum-preserving band-aware codebook  [next]
Stage 1:  calibrated spectrum-preserving band codebooks
Stage 2:  timestep-specific spectrum-preserving codebooks
```

### Stage 0a: Masked Random Codebook (Falsified)

This was the lowest-cost V2-a and keeps GVCC zero-shot. Keep the existing
random codebook:

```text
C_k ~ N(0, I)
```

but restrict both residual and atoms to the current active band:

```text
C_k^B = T^{-1}(M_B * T(C_k))
r_t^B = T^{-1}(M_B * T(r_t))
```

Selection:

```text
k* = argmax_k | < C_k^B, r_t^B > |
```

Injection:

```text
z_t* = C_{k*}^B
```

For multi-atom GVCC, the same masking applies before top-`M` selection and
before reconstructing the summed innovation. This stage answered the cleanest
question:

```text
If the codebook is forced to operate in the right frequency band at the right
timestep, does quality or coding efficiency improve?
```

The answer is no for direct band-limited injection. The reason is not
selection quality; it is trajectory distribution mismatch. Any method that
makes the actual injected `z_t*` band-limited risks pushing `x_t` off the
pretrained SDE manifold.

### Stage 0b: Spectrum-Preserving Band-Aware Codebook

The safe fix is to keep the injected innovation full-spectrum while making the
transmitted information concentrate on the active band.

Let `B_i` be the active band and `bar(B_i)` its complement. Select the codebook
atom only on the active band:

```text
k* = argmax_k | < M_{B_i} * T(C_k), M_{B_i} * T(r_t) > |
```

Build the injected innovation as:

```text
T(z_t*)
  = M_{B_i} * T(C_{k*})
    + M_{bar(B_i)} * eta,

eta ~ N(0, I).
```

Equivalently:

```text
T(z_t*) = T(eps) + M_{B_i} * (T(z_selected) - T(eps)).
```

This second form is the clearest implementation rule: first generate a
full-band Gaussian carrier `eps`, then replace only the active-band
coefficients by the selected codebook atom.

Implementation detail: before replacement, normalize the selected active-band
coefficient vector to unit standard deviation. Otherwise the sum of `M`
selected atoms can dominate the Gaussian carrier and the injected spectrum is
no longer white-noise-like.

For multi-atom GVCC:

```text
T(z_t*)
  = M_{B_i} * T(sum_j s_j C_{k_j})
    + M_{bar(B_i)} * eta.
```

Then normalize the full `z_t*` to unit variance before the usual SDE update.
This keeps the codebook-controlled signal in the scheduled band while
preserving the off-band Gaussian carrier expected by Wan:

```text
z_t* is full-spectrum and approximately white,
but only its active-band component carries transmitted information.
```

This is the preferred next experiment because it does not change `f_t`, `g_t`,
or the pretrained Wan sampler. It only changes how the decoder fills the
uncontrolled frequency complement.

The bitstream does not need to change. The active-band codebook indices/signs
are transmitted as before. The off-band Gaussian carrier is generated from a
shared deterministic seed, so encoder and decoder reproduce it without side
information.

An alternative, more aggressive fix is per-band diffusion scaling:

```text
x_{t-dt}
  = x_t
    - f_t(x_t) * dt
    + sqrt(dt) * sum_B g_{t,B} * z_{t,B}.
```

This is theoretically cleaner but changes the effective SDE and should come
after the spectrum-preserving version.

### Stage 1: Calibrated Spectral Codebook

If spectrum-preserving band scheduling helps, move to calibrated
band-specialized codebooks:

```text
C_low, C_mid, C_high
```

Each codebook should model the residual distribution of its band:

```text
r_t = x_0 - x_hat_{0|t}
r_t^B = M_B * T(r_t)
```

Collect residual statistics from encoder-side calibration runs and build the
codebook from simple to complex:

```text
1. PCA basis + random coefficients
2. k-means / LBG vector quantization
3. residual PCA low-rank Gaussian codebook
4. product quantization per band
5. per-timestep-band codebook
```

The first trained version should avoid full-latent k-means. Prefer:

```text
z_k^B = A_B q_k
q_k ~ N(0, I)
```

where `A_B` is a PCA basis estimated from residuals in band `B`. This is better
described as a **calibrated spectral codebook** rather than a fully trained
codec codebook. It preserves the GVCC flavor: deterministic shared codebooks,
light calibration, no video-specific side information.

### Stage 2: Timestep-Specific Band Codebook

The final extension is to condition codebooks on both band and timestep group:

```text
C_{B,tau}
```

Example groups:

```text
early-low codebook
mid-mid codebook
late-high codebook
```

This is useful because the same frequency band has different residual
statistics at different denoising stages:

```text
early:  high band mostly noise-dominated
middle: edges and motion boundaries emerge
late:   texture and fine details dominate
```

This should be a later V2-a extension, after Stage 0b proves that
spectrum-preserving scheduling is worth doing.

### Band Split

Use DCT radial frequency masks on the full latent spatial grid. For latent size
`H_l x W_l`, define:

```text
rho(u, v) = sqrt((u / W_l)^2 + (v / H_l)^2)
```

Simple first split:

```text
low:   first 1/3 of radial frequency range
mid:   middle 1/3
high:  top 1/3
```

Better split after measurement:

```text
low:   lowest frequencies containing 80% residual energy
mid:   80% -> 95% cumulative residual energy
high:  remaining 5%
```

Before fixing thresholds, measure:

```text
residual spectrum over timestep
```

for `r_t = x_0 - x_hat_{0|t}`. The goal is to see how low/mid/high residual
energy evolves along the GVCC trajectory.

### Band Schedule

Simple manual schedule for `T=20`, `num_ddim_tail=3`, 17 codebook steps:

```text
steps 0-5:    low only
steps 6-11:   low + mid
steps 12-16:  low + mid + high
steps 17-19:  deterministic ODE tail
```

Here step 0 is the high-noise end (`t ~= 1`).

SPD-derived schedule can come later. A simple empirical rule is:

```text
SNR_B(t) = ((1 - t)^2 / t^2) * P_B
open band B when SNR_B(t) > tau_B
```

where `P_B` is the measured residual or clean-latent band energy. This should
naturally open low first, mid later, and high latest.

### V2-a Experiment Matrix

Keep the first matrix small:

| Method | Trajectory | Codebook selection | Purpose |
| --- | --- | --- | --- |
| Full GVCC | full-res | full-band every step | Baseline. |
| DCT all-band control | full-res | all bands open through DCT code path | Verify implementation path. |
| V2-a Stage 0a | full-res | band-limited low -> mid -> high masked random | Negative control; falsified. |
| V2-a Stage 0b | full-res | active-band codebook + off-band Gaussian | Test spectrum-preserving scheduling. |
| V2-a 0b fewer bits | full-res | fewer active-band atoms / smaller K | Test BPP/search reduction after 0b is stable. |
| V2-a calibrated band CB | full-res | calibrated spectrum-preserving band codebooks | Test learned/calibrated codebook benefit. |

The critical first comparison is same total bits:

```text
baseline:
  every codebook step uses M=64 full-band atoms

V2-a Stage 0a:
  every codebook step still uses M=64 atoms
  atoms are selected and injected only in active bands

V2-a Stage 0b:
  every codebook step still uses M=64 atoms
  active band comes from codebook selection
  inactive band is filled by deterministic off-band Gaussian noise
```

Stage 0a has already failed. The next gate is Stage 0b: if
spectrum-preserving scheduling does not recover near-baseline quality, then
training band codebooks is unlikely to be worth the added complexity.

### Speed Expectation

V2-a alone does not substantially speed up the decoder because Wan forward
count is unchanged. It can still speed up the encoder if codebook search is a
meaningful cost:

```text
1. early steps do not search high band;
2. inner products use lower-dimensional masked residuals;
3. each band can use smaller K;
4. high band is enabled only in later steps.
```

The larger speed gain comes from combining V2-a with V2-b:

```text
frequency-scheduled codebook
+ fewer denoising steps
= fewer Wan forwards with denser codebook information per step
```

### Training Band Codebooks: Pros and Risks

Potential benefits:

```text
1. atoms better match residual distribution;
2. low-frequency atoms learn structure, high-frequency atoms learn texture;
3. higher matching efficiency at fixed M/K;
4. less ineffective random atom noise.
```

Risks:

```text
1. weakens zero-shot purity through calibration data;
2. calibration distribution may overfit;
3. V1 DLC2 suggests transition residuals are not always low-rank;
4. codebook versioning and bitstream management become more complex.
```

Therefore the paper language should prefer:

```text
calibrated spectral codebook
```

over "trained codec codebook" unless the training setup becomes a main
contribution.

## V2-b: Step Scheduling / Fewer NFE

V2-b is now an auxiliary branch rather than the main route. It can still reduce
Wan forward count, but the main GVCCTurbo story should return to
multi-resolution because progressive resolution is the direct way to reduce
token count per forward. V2-b becomes useful after MR-V2 identifies a stable
quality-preserving correction.

The V2-b diagnostic baseline can stay full-resolution to isolate fewer-NFE
effects:

```text
T = 20 full-res GVCC baseline
T = 16 full-res spectral-scheduled GVCC
T = 12 full-res spectral-scheduled GVCC
T = 10 full-res spectral-scheduled GVCC
```

The idea is:

```text
fewer Wan forwards
+ more informative spectral codebook bits per step
+ adaptive M per timestep / frequency band
= higher information density per NFE
```

Reducing `T` from 20 to 12 can in principle approach a 1.6x forward-pass
speedup, but it does not reduce per-forward token cost. It should therefore be
combined with MR-V2 rather than replacing the multi-resolution route.

The implementation knobs are:

1. **Step count:** compare `T = 20, 16, 12, 10`.
2. **Timestep allocation:** keep more steps where residual correction is most
   useful, not necessarily uniform spacing.
3. **Band schedule:** early low, middle low+mid, late full band.
4. **Adaptive `M`:** assign more atoms to important timestep/band pairs and
   fewer atoms to weak or noise-dominated pairs.
5. **Bit budget matching:** compare at fixed BPP as well as fixed runtime.

The key V2 question should change from:

```text
Can a low-resolution stage approximate the full-resolution trajectory?
```

to:

```text
On the full-resolution trajectory, which timesteps and frequency bands really
need codebook bits, and which can receive fewer bits, later bits, or no bits?
```

That is the compression-specific version of the SPD prior: GVCC-Turbo should
spend bits where they most affect reconstruction, while using fewer Wan
forwards overall.

## DLC1: SPD-Side Improvement

### Claim

SPD high-frequency expansion is marginally correct but conditionally
suboptimal.

Flow matching in frequency domain:

```text
x_t^{(omega)} = (1 - t) * x_0^{(omega)} + t * epsilon^{(omega)}
```

At transition `t_i`, low-frequency observation:

```text
y_L = x_{t_i}^L = a * x_0^L + b * epsilon_L
a = 1 - t_i
b = t_i
```

True high-frequency state:

```text
x_{t_i}^H = a * x_0^H + b * epsilon_H
```

Minimum-MSE initializer:

```text
x_{t_i}^{H,*}
  = E[a * x_0^H + b * epsilon_H | x_{t_i}^L]
  = a * E[x_0^H | x_{t_i}^L]
```

Under the zero-mean Gaussian/LMMSE approximation:

```text
E[x_0^H | y_L]
  = a * Sigma_HL * (a^2 * Sigma_LL + b^2 * I)^{-1} * y_L
```

so:

```text
x_{t_i}^{H,*}
  = (1 - t_i)^2
    * Sigma_HL
    * ((1 - t_i)^2 * Sigma_LL + t_i^2 * I)^{-1}
    * x_{t_i}^L
```

Note: Method C/D below use nonlinear `v_theta`, so they can in principle
exceed this LMMSE bound when the model captures non-linear low-high coupling.

SPD instead uses:

```text
x_{t_i}^{H,SPD} = t_i * epsilon_H
E[x_{t_i}^{H,SPD} | x_{t_i}^L] = 0
```

So SPD drops the predictable high-frequency mean, behaving as if
`Sigma_HL ~= 0`.

### Conditional Distribution

The more complete target is:

```text
p(x_{t_i}^H | x_{t_i}^L)
```

Under Gaussian assumption:

```text
x_{t_i}^H | y_L
  ~ N(mu_{H|L}(t_i), Sigma_{H|L}(t_i))
```

Mean:

```text
mu_{H|L}(t_i)
  = (1 - t_i)^2
    * Sigma_HL
    * ((1 - t_i)^2 * Sigma_LL + t_i^2 * I)^{-1}
    * y_L
```

Covariance:

```text
Sigma_{H|L}(t_i)
  = (1 - t_i)^2 * Sigma_HH
    + t_i^2 * I
    - (1 - t_i)^4
      * Sigma_HL
      * ((1 - t_i)^2 * Sigma_LL + t_i^2 * I)^{-1}
      * Sigma_LH
```

Natural high-frequency expansion:

```text
x_{t_i}^H
  = mu_{H|L}(t_i) + L_{H|L}(t_i) * eta
L_{H|L}(t_i) * L_{H|L}(t_i)^T = Sigma_{H|L}(t_i)
```

SPD approximation:

```text
mu_{H|L}(t_i) = 0
Sigma_{H|L}(t_i) ~= t_i^2 * I
```

### Limits

```text
t_i -> 1:  mu_{H|L}(t_i) -> 0
t_i -> 0:  mu_{H|L}(t_i) -> Sigma_HL * Sigma_LL^{-1} * x_0^L
```

Thus random padding is reasonable very early, but late transitions should look
more like low-frequency-conditioned super-resolution.

### DLC1 Experiment Design

Fix model, prompt, schedule, seed, and steps. Change only transition
high-frequency initializer:

```text
A: Random      x_{t_i}^H = t_i * epsilon_H
B: Oracle      x_{t_i}^H = (1 - t_i) * x_0^H + t_i * epsilon_H
C: Self-pred   x_{t_i}^H = (1 - t_i) * x_hat_{0|t_i}^H + t_i * epsilon_H
D: Learned     x_{t_i}^H = (1 - t_i) * g_phi(x_{t_i}^L, t_i, c) + t_i * epsilon_H
```

For Method C, computing `x_hat_{0|t_i}` requires a full-grid model call. The
low-resolution state therefore needs a temporary high-frequency initializer
`eta_H` before the self-prediction step. Strictly, this estimates
`E[x_0^H | y_L, eta_H]`, not pure `E[x_0^H | y_L]`; the useful sanity check is
whether `x_hat_{0|t_i}^H` remains stable when `eta_H` changes.

Transition metric:

```text
E_H(t_i)
  = || T_H(x_{t_i}) - T_H(x_{t_i}^{oracle}) ||_2^2
```

Also record final PSNR/LPIPS/MS-SSIM, high-frequency residual decay, runtime,
speedup, temporal MAE, and flicker.

## DLC2: AsymFlow-Inspired Structured Innovation

### Hypothesis

AsymFlow argues that full-rank noise prediction is inefficient in high
dimensions. For GVCCTurbo, the corresponding statement is:

```text
GVCC should not spend codebook bits on full-rank Gaussian innovation.
It should spend bits on a timestep-aware structured subspace.
```

Structured innovation:

```text
z_t* in Im(P_t)
```

where `P_t` may be frequency-aware, motion-aware, spatially adaptive, or
learned.

### Version A: Low-Rank Codebook

Original GVCC:

```text
C_k ~ N(0, I_D)
```

Low-rank codebook:

```text
C_k = A q_k
q_k ~ N(0, I_r)
r << D
```

Selection:

```text
<q_k, A^T r_t>
```

Update:

```text
z_t* = A q_k
```

Possible `A`: residual PCA, DCT/DWT basis, patch PCA, temporal motion basis,
or low-to-full spectral lift basis.

### Version B: Frequency-Band AsymCodebook

For each frequency band `B_m`:

```text
A_{B_m} in R^{D_{B_m} x r_m}
z_k^{B_m} = A_{B_m} q_k
```

Schedule:

```text
early:  P_t = P_low
middle: P_t = P_low + P_mid
late:   P_t = P_low + P_mid + P_high
```

Transition control:

```text
x_t^H = t * epsilon_H + lambda_t * A_H q_k
```

### Version C: Clean Prior + Low-Rank Correction

AsymFlow-style decomposition:

```text
low-rank subspace:      u-style noise/control
orthogonal complement:  x_0-style clean prediction
```

GVCCTurbo transition:

```text
x_t^H
  = t * epsilon_H
    + (1 - t) * x_hat_{0|t}^{H,perp}
    + lambda_t * A_H q_k
```

Summary:

```text
full-rank clean prior + low-rank transmitted innovation
```

### Relation to Existing GVCC Results

GVCC ablations already suggest an effective-rank limit:

- `M=64` gives clear gains.
- `M>128` gives diminishing returns.
- Too many atoms can add codebook noise.
- Too large `g_scale` collapses the trajectory.

This suggests that residuals are structured:

```text
low-frequency structure residual
mid-frequency edge residual
high-frequency texture residual
temporal motion residual
GOP boundary residual
```

The next step is not simply increasing `M` or `K`; it is designing better
`P_t / A_t`.

### DLC2 Experiments

Experiment 1: full-rank random vs PCA low-rank codebook.

```text
C_k ~ N(0, I_D)
C_k = A_PCA q_k
```

Experiment 2: DCT band-limited codebook.

```text
C_k^B = T^{-1}(M_B * T(C_k))
```

Experiment 3: transition-only low-rank high-frequency codebook.

```text
x_t^H = t * epsilon_H
x_t^H = t * epsilon_H + lambda_t * z_k^H
x_t^H = t * epsilon_H + lambda_t * A_H q_k
x_t^H = t * epsilon_H + (1 - t) * x_hat_{0|t}^H + lambda_t * A_H q_k
```

### Latent-to-Pixel Lift Analogy

AsymFlow lifts pretrained latent flow into pixel space with a low-rank linear
lift. GVCCTurbo analogue:

```text
x^{full,L} = A x^{low}
x^{full}   = A x^{low} + A_H q_k
```

Low-resolution trajectory carries semantics/global structure; high-resolution
codebook correction carries low-level detail and frequency compensation.

### Caution

Do not directly apply AsymFlow formulas to Wan. AsymFlow is a training
parameterization:

```text
u_A = P epsilon - x_0
```

Wan still predicts the original flow velocity `u`, not `u_A`. For now, use
AsymFlow only as:

1. codebook subspace design principle,
2. multi-resolution lift design principle,
3. high-frequency innovation low-rank principle,
4. full-rank clean prior + low-rank correction structure.

Training a Wan-style AsymFlow video model is a later project.
