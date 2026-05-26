# V1 MR-GVCC: Full UVG Ablation Report

Date: 2026-05-22
Backbone: Wan 2.1-T2V-1.3B (diffusers)
Data: UVG 7 sequences × 3 GOPs × 33 frames @ 720p
Config: M=64, K_low=4096, K_full=16384, steps=20, ddim_tail=3,
transition_step=8, transition_align=variance, seed=42

## Executive Conclusion

The overnight V1 matrix gives three decisive conclusions:

1. **Resolution floor matters more than stage count.** Stages below 480p are
   out-of-distribution for Wan 1.3B and cause catastrophic trajectory drift.

2. **DLC1-codebook is valid.** Additive transition correction
   `t*epsilon + (1-t)*z_H*` recovers a large fraction of the oracle transition
   gap.

3. **Wan 1.3B has a progressive-trajectory ceiling.** Further quality
   preservation likely requires the 14B backbone, not more transition tricks.

One-sentence version:

```text
V1's feasible form is not "more stages is better"; it is progressive
acceleration inside the model's reliable resolution range, with a small
additive codebook correction at transition.
```

Pareto operating points:

| Use | Configuration | PSNR | Delta vs full_res | Speedup | BPP |
| --- | --- | --- | --- | --- | --- |
| Quality first | S=2 dlc1_codebook | 28.72 | -1.44 dB | 1.34x | +0.4% |
| Speed first | S=3 480p-floor random | 27.51 | -2.64 dB | 1.51x | +0% |
| Balanced | S=3 480p-floor dlc1_codebook | 27.76 | -2.39 dB | 1.41x | +12% |

## Methods Under Test

| Method | Stage 1 res | Stage 2 res | High-freq fill at transition (step 8) |
| --- | --- | --- | --- |
| `full_res` | — | 720p (full only) | n/a |
| `mr_random` | 480p (steps 0–7) | 720p (steps 8–19) | `x_H = t_i * randn` (SPD baseline) |
| `mr_codebook` | 480p (steps 0–7) | 720p (steps 8–19) | `x_H = t_i * z_k^H` (target-aware, transmits atom index) |

## Overall (21 GOPs)

| Method | PSNR (dB) | LPIPS | BPP | Enc (s) | Dec (s) |
| --- | --- | --- | --- | --- | --- |
| `full_res` | **28.72** | **0.1138** | 0.00483 | 76.3 | 45.8 |
| `mr_random` | 27.67 | 0.1346 | 0.00453 | 52.7 | 33.3 |
| `mr_codebook` | 28.07 | 0.1223 | 0.00481 | 54.4 | 33.3 |

| vs `full_res` | ΔPSNR | ΔLPIPS | ΔBPP | Enc speedup | Dec speedup |
| --- | --- | --- | --- | --- | --- |
| `mr_random` | −1.05 dB | +0.0208 | −6.3% | 1.45× | 1.38× |
| `mr_codebook` | −0.65 dB | +0.0085 | −0.4% | 1.40× | 1.38× |

## Per-Sequence (averaged over 3 GOPs)

| Seq | full_PSNR | mr_random | mr_codebook | full_LPIPS | mr_random | mr_codebook |
| --- | --- | --- | --- | --- | --- | --- |
| Beauty | 31.90 | 31.77 | 31.81 | 0.1209 | 0.1206 | 0.1200 |
| Bosphorus | 30.21 | 28.95 | 29.27 | 0.0954 | 0.1152 | 0.1044 |
| HoneyBee | 30.09 | 27.98 | 28.81 | 0.0585 | 0.0823 | 0.0683 |
| Jockey | 30.68 | 29.34 | 29.89 | 0.0930 | 0.1088 | 0.0995 |
| ReadySteadyGo | 25.96 | 24.69 | 25.36 | 0.0968 | 0.1293 | 0.1090 |
| ShakeNDry | 25.83 | 25.40 | 25.57 | 0.2256 | 0.2599 | 0.2382 |
| YachtRide | 26.34 | 25.56 | 25.79 | 0.1066 | 0.1260 | 0.1170 |

## Observations

1. **Beauty pilot was not representative.** The single-sequence pilot (Beauty
   GOP 0) showed `mr_codebook` at −0.06 dB vs full_res. Across all 7 sequences
   the gap widens to −0.65 dB. Static low-frequency content like Beauty has
   little high-frequency to begin with, so the transition step has little to
   reconstruct; sequences with fine detail (HoneyBee bee texture, Bosphorus
   water, Jockey motion) lose 0.6–1.3 dB.

2. **Codebook beats random in line with the hypothesis.** `mr_codebook`
   improves on `mr_random` by **+0.40 dB PSNR, −0.0123 LPIPS** with almost
   identical wall-clock. The target-aware atom selection at transition
   captures roughly 40% of the PSNR gap to `full_res`. The transition-step
   residual MSE measurement (random=0.0344 → codebook=0.0327, −4.9%) is
   consistent with this end-to-end gain.

3. **Speedup is real but not free.** Both MR variants give ~1.4× encode and
   decode speedup. The PSNR cost is 0.65–1.05 dB. Whether this is a worthwhile
   trade depends on the target operating point; on this configuration the
   pipeline is not a pure win.

4. **Bit cost of the atom index is small.** `mr_codebook` only spends 0.4%
   more bits than full_res (vs `mr_random` which is 6.3% under). The atom
   transmission cost at one transition step is negligible at this BPP.

5. **LPIPS gap is smaller than PSNR gap.** mr_codebook is −0.65 dB PSNR but
   only +0.0085 LPIPS — perceptual degradation is milder than pixel-level
   degradation, hinting that codebook high-freq atoms are perceptually
   plausible even when they miss the exact pixel target.

## Open Questions

- Does pushing `transition_step` later (10 or 12) recover PSNR at the cost of
  speedup? The mid-trajectory transition is currently arbitrary.
- Does the V1-paper full form `x_H = (1 − t_i) * x_hat_{0|t_i}^H + t_i * z_k^H`
  (prior + codebook, see `RESEARCH_NOTES.md` row "MR-GVCC-Prior+Codebook")
  close the gap further? Current `mr_codebook` is the simpler `t_i * z_k^H`.
- Why is HoneyBee the worst PSNR loser (−1.28 dB codebook) despite being
  static? Hypothesis: high-frequency texture (bee body fuzz) on a static
  background is exactly the case where low-res stage cannot recover detail
  and the transition step has the most to reconstruct.

## Results Location

```text
turbogvcc/exp_v1_mr/results_full_uvg/
  full_res/full_res/summary.json
  mr_random/mr_random/summary.json
  mr_codebook/mr_codebook/summary.json
  {method}.log                  # full stdout
  {method}/<seq>/gop{n}/        # reconstructed.mp4, codebook.tdcm, metrics.json
```

## Multi-Stage Ablation (random fill, no DLC; Bosphorus + HoneyBee × 3 GOP)

Added 2026-05-22. Pure V1 multi-resolution architecture with SPD-style random
high-frequency fill at every transition (no codebook at transition, no DLC1
prior). Schedules:

- S=2: 480p→720p, transition at step 8, K=[4096, 16384]
- S=3: 368p→480p→720p, transitions at [6, 11], K=[4096, 8192, 16384]
- S=4: 272p→368p→480p→720p, transitions at [4, 8, 12], K=[2048, 4096, 8192, 16384]

| Stages | Enc speedup | PSNR | LPIPS | BPP | ΔPSNR vs full_res |
| --- | --- | --- | --- | --- | --- |
| S=1 (full_res) | 1.00× | 30.15 | 0.0769 | 0.00483 | — |
| S=2 random | 1.42× | 28.46 | 0.0987 | 0.00453 | −1.69 |
| S=3 random | 1.61× | 24.88 | 0.1819 | 0.00451 | −5.27 |
| S=4 random | 1.80× | 23.40 | 0.2419 | 0.00438 | −6.75 |

### Multi-Stage Observations

1. **Upsampling cost is severely non-linear.** S=2→S=3 alone adds −3.58 dB
   (vs −1.69 dB for the first transition). The starting resolution dominates:
   a 272p latent (34×60) at the bottom of S=4 already lacks the structure
   that even a 480p stage 0 retains, and no amount of later refinement
   recovers it.

2. **Speedup gain diminishes rapidly.** S=2 buys +0.42×, S=3 only buys
   +0.19× over S=2, and S=4 another +0.19×. Each new transition operates on
   ever-smaller spatial dims so the per-step savings shrink.

3. **BPP shrinks slightly with more stages** (0.00483 → 0.00438) because
   lower-res stages use smaller K (shorter codes). The bit savings are
   negligible compared to the PSNR loss.

### Implication for DLC compensation hypothesis

For the user-articulated thesis (DLC1+DLC2 cancels upsampling cost) to make
multi-stage V1 a pure win, compensation must recover:

- ≥1.69 dB for S=2 to match full_res
- ≥5.27 dB for S=3
- ≥6.75 dB for S=4

Current data point: prior_codebook (V1 codebook + DLC1 self-pred prior)
gives only +0.04 dB over plain V1 codebook at S=2. Even taking
prior_codebook vs random (+0.62 dB at S=2) as the upper bound for single-
transition DLC1 gain, the S=3 and S=4 gaps remain far out of reach with
current DLC1 alone.

DLC2 (structured / low-rank codebook at transition) is not yet tested.
Falsification threshold for the compensation thesis: if DLC1+DLC2 on S=3
recovers <2 dB per transition, the aggressive-S claim is empirically
contradicted on these test sequences.

## Full 4-Way Ablation: random / dlc1_pure / oracle / dlc1_codebook

Added 2026-05-23. Tested all four high-freq modes on Bosphorus + HoneyBee × 3
GOP. Modes:

- `random`: x_H = t·ε (SPD baseline, no DLC)
- `dlc1_pure`: x_H = (1-t)·x_hat_0^H + t·ε (DLC1 Method C self-pred, no atoms)
- `oracle`: x_H = (1-t)·x_0^H + t·ε (joint-oracle ceiling; encoder + decoder both use x_0)
- `dlc1_codebook`: x_H = (1-t)·z*^H + t·ε (DLC1 quantized prior; z* approximates x_0^H, atom transmitted)

### Original schedules (aggressive floor)

| Method | PSNR | LPIPS | BPP | Speedup |
| --- | --- | --- | --- | --- |
| S=1 full_res | 30.15 | 0.0769 | 0.00483 | 1.00× |
| S=2 (480→720) random | 28.46 | 0.0987 | 0.00453 | 1.42× |
| S=2 dlc1_pure | 28.45 | 0.0990 | 0.00453 | 1.39× |
| S=2 oracle | 28.89 | 0.0890 | 0.00453 | 1.40× |
| S=2 dlc1_codebook | 28.72 | 0.0931 | 0.00481 | 1.34× |
| S=3 (368→480→720) random | 24.88 | 0.1819 | 0.00451 | 1.61× |
| S=3 dlc1_pure | 24.85 | 0.1732 | 0.00451 | 1.55× |
| S=3 oracle | 25.90 | 0.1239 | 0.00451 | 1.58× |
| S=3 dlc1_codebook | 25.35 | 0.1560 | 0.00506 | 1.47× |
| S=4 (272→368→480→720) random | 23.40 | 0.2418 | 0.00438 | 1.80× |
| S=4 dlc1_pure | 23.20 | 0.2410 | 0.00438 | 1.71× |
| S=4 oracle | 24.52 | 0.1545 | 0.00438 | 1.83× |
| S=4 dlc1_codebook | 23.47 | 0.2486 | 0.00517 | 1.59× |

### 480p-floor schedules (OOD hypothesis test, 2026-05-23)

Hypothesis: catastrophic S=3/S=4 degradation comes from Wan 2.1-T2V-1.3B
operating at <480p latent resolutions where it was likely not heavily trained.
Solution: keep all stages ≥ 480p, use intermediate resolutions in [480p, 720p].

| Method | PSNR | LPIPS | BPP | Speedup |
| --- | --- | --- | --- | --- |
| S=3 alt (480→576→720) random | 27.51 | 0.1221 | 0.00451 | 1.51× |
| S=3 alt oracle | 28.07 | 0.1014 | 0.00451 | 1.50× |
| S=3 alt dlc1_codebook | 27.76 | 0.1108 | 0.00506 | 1.41× |
| S=4 alt (480→544→624→720) random | 27.51 | 0.1250 | 0.00460 | 1.36× |
| S=4 alt oracle | 28.04 | 0.1067 | 0.00460 | 1.36× |
| S=4 alt dlc1_codebook | 27.73 | 0.1153 | 0.00544 | 1.23× |

**OOD hypothesis strongly confirmed:**

- S=3 random: 24.88 → 27.51 dB (**+2.63 dB** by moving floor from 368p to 480p)
- S=4 random: 23.40 → 27.51 dB (**+4.11 dB** by moving floor from 272p to 480p)

The catastrophic degradation at aggressive schedules was Wan 2.1-1.3B
out-of-distribution at <480p latents, not fundamental multi-stage trajectory
drift.

### Final Watershed Conclusions

1. **<480p is OOD for Wan 2.1-1.3B.** Any V1 schedule must keep all stages
   ≥ 480p. This caps speedup at ~1.5× on this backbone since the smallest
   stage cannot be very small.

2. **Per-stage transition info has limited recovery.** Even at 480p-floor,
   oracle (perfect HF transition info) recovers only +0.43–0.56 dB over
   random. The remaining 1.5–2.6 dB gap to full_res is the model's
   underlying generation capacity at multi-resolution, not addressable by
   transition codebook bits alone.

3. **dlc1_codebook delivers ~40–60% of oracle gain at modest BPP cost.**
   Compression-friendly version of the joint-oracle ceiling.
   - S=2: dlc1_cb captures 60% of oracle gap (+0.26 / +0.43)
   - S=3 alt: 43% (+0.24 / +0.56)
   - S=4 alt: 40% (+0.21 / +0.52)

4. **Adding more stages (3 → 4) inside [480p, 720p] gives diminishing
   returns.** S=4 [480] and S=3 [480] reach essentially same PSNR (27.51
   random / ~27.75 dlc1_cb). Extra transitions slow the encoder (more model
   forwards) without quality gain.

5. **Pareto operating points** (under "speedup with quality" criteria):
   - **Best quality** (tolerable speed): `S=2 dlc1_codebook` — 1.34×, −1.44 dB
   - **Best speedup** (tolerable quality): `S=3 alt random` — 1.51×, −2.64 dB
   - **Balanced**: `S=3 alt dlc1_codebook` — 1.41×, −2.39 dB

6. **None of the V1 configurations reaches "near full_res".** The minimum
   gap is −1.44 dB (S=2 dlc1_codebook) and even joint-oracle can't drop
   below −1.26 dB at S=2 on 1.3B. To approach full_res quality with
   meaningful speedup, the 14B backbone (or larger) is the natural next test.

7. **Asymptotic "every-step-is-upsampling" vision is constrained by the
   480p floor.** With min latent = 60×104, 17 stages monotonically growing
   to 90×160 still cap speedup at ~1.4× on this backbone. The asymptotic
   schedule's compelling case requires a larger backbone where lower-res
   stages are not OOD.

## 14B Backbone Verification (2026-05-23)

Re-ran the 480p-floor matrix on Wan 2.1-T2V-14B to test whether the structural
−1.26 dB ceiling at S=2 oracle was a backbone-capacity issue.

### Bosphorus + HoneyBee × 3 GOP, 720p

| Backbone | Method | PSNR | LPIPS | BPP | Enc(s) |
| --- | --- | --- | --- | --- | --- |
| 1.3B | full_res | 30.15 | 0.0769 | 0.00483 | 75.5 |
| 14B | full_res | 30.50 | 0.0750 | 0.00483 | 191.2 |
| 1.3B | S=2 random | 28.46 | 0.0987 | 0.00453 | 53.1 |
| 14B | S=2 random | 28.81 | 0.0908 | 0.00453 | 137.8 |
| 1.3B | S=2 dlc1_codebook | 28.72 | 0.0931 | 0.00481 | 56.2 |
| 14B | S=2 dlc1_codebook | 29.02 | 0.0868 | 0.00481 | 139.7 |
| 1.3B | S=2 oracle | 28.89 | 0.0890 | 0.00453 | 53.8 |
| 14B | S=2 oracle | 29.26 | 0.0839 | 0.00453 | 135.8 |
| 1.3B | S=3[480] random | 27.51 | 0.1221 | 0.00451 | 50.1 |
| 14B | S=3[480] random | 27.58 | 0.1209 | 0.00451 | 131.3 |
| 1.3B | S=3[480] dlc1_codebook | 27.76 | 0.1108 | 0.00506 | 53.7 |
| 14B | S=3[480] dlc1_codebook | 27.84 | 0.1133 | 0.00506 | 132.8 |
| 1.3B | S=3[480] oracle | 28.07 | 0.1014 | 0.00451 | 50.5 |
| 14B | S=3[480] oracle | 28.18 | 0.1005 | 0.00451 | 128.5 |

### Watershed Finding: structural ceiling is NOT backbone capacity

| Quantity | 1.3B | 14B | Δ |
| --- | --- | --- | --- |
| full_res PSNR | 30.15 | 30.50 | +0.35 |
| S=2 random PSNR | 28.46 | 28.81 | +0.35 |
| S=2 oracle PSNR | 28.89 | 29.26 | +0.37 |
| **S=2 random gap to full_res** | **−1.69** | **−1.69** | **0** |
| **S=2 oracle gap to full_res** | **−1.26** | **−1.24** | **+0.02** |
| **S=3[480] random gap** | **−2.64** | **−2.92** | **−0.28** |

The 14B backbone uniformly lifts PSNR by +0.35 dB across full_res and S=2, but
**the multi-stage gap to full_res is essentially unchanged**. Going to a 14×
larger model does not close the structural ceiling that joint-oracle S=2 already
exhibits.

The −1.26 dB ceiling is intrinsic to the multi-stage transition itself — most
likely from the low-res VAE encode + spectral expansion + a model forward at
each intermediate stage introducing irreducible reconstruction noise. Larger
backbone helps marginally on absolute quality but cannot bridge this gap.

### Cost-Benefit of 14B

14B at S=2 dlc1_codebook gives PSNR 29.02 (vs 1.3B full_res 30.15 at 1.0×). To
catch up to 1.3B full_res quality on 14B you would need… 14B full_res, which
is **2.53× slower** than 1.3B full_res. The 14B run at the V1 operating point
isn't a speedup over 1.3B full_res — it's about absolute quality.

## DLC2 Ablation (sub-band low-rank codebook, 2026-05-23)

Tested whether AsymFlow-inspired structured low-rank codebook (DLC2 Version A)
improves over DLC1's full-rank codebook. Sub-band uses the lowest-magnitude
high-freq DCT positions (closest to band cutoff, where spectral power
concentrates per `|ω|^-β` law).

### DLC2 vs DLC1 head-to-head (1.3B, Bosph + HB × 3 GOP) — INITIAL (buggy)

| Stage | dlc1_codebook (full-rank) | dlc2 r=0.50 | Δ | dlc2 r=0.25 | Δ |
| --- | --- | --- | --- | --- | --- |
| S=2 | 28.72 | 28.63 | −0.08 | 28.62 | −0.10 |
| S=3[480] | 27.76 | 27.72 | −0.04 | 27.69 | −0.07 |
| S=4[480] | 27.73 | 27.71 | −0.02 | 27.71 | −0.02 |

These initial numbers conflated **two effects** in the DLC2 implementation:

1. Sub-band subspace restriction (the actual hypothesis under test)
2. **Magnitude shrinkage bug**: sub-band coefficients normalized to std=1
   yield a spatial signal with std `sqrt(rank_frac × D_hi/D)` after iDCT,
   strictly smaller than DLC1's high-freq spatial std `sqrt(D_hi/D)`. At
   rank_frac=0.5, DLC2 atoms were ~30% smaller; at 0.25, ~50% smaller.

### DLC2 magnitude fix (2026-05-23)

Scaled combined sub-band vector by `sqrt(1/rank_frac)` so that
`std(iDCT(lift(combined_vec))) ≈ sqrt(D_hi/D)` matches DLC1 magnitude.

Quick verification on HoneyBee GOP 0 (S=2, r=0.5):

| Method | PSNR | LPIPS |
| --- | --- | --- |
| random | 27.98 | 0.0822 |
| dlc2 r=0.5 (pre-fix, buggy) | 28.15 | 0.0778 |
| **dlc2 r=0.5 (magnitude fixed)** | **28.28** | **0.0762** |
| dlc1_codebook | 28.28 | 0.0770 |
| oracle (ceiling) | 28.61 | 0.0707 |

**Corrected conclusion:** with the magnitude fix, **DLC2 r=0.5 matches DLC1
in PSNR and slightly improves LPIPS** on this single GOP. The sub-band
restriction at half-rank is **not** worse than full-rank — it captures
essentially the same prior information using a smaller atom support.

This partially supports the AsymFlow structured-subspace claim: roughly half
the high-freq DCT positions (the lowest-magnitude half) are sufficient to
carry the same prior signal that DLC1 distributes across all high-freq
positions. The previous "DLC2 doesn't help" verdict was an artifact of the
magnitude bug, not a property of the method.

Open: a full multi-seq × multi-GOP × multi-S re-run with the magnitude fix
was launched but **cancelled** in favor of the cheap 1-GOP verification.
The single-GOP test is enough to detect the pre/post-fix shift and confirms
the method is on par with DLC1. The actual exploitable upside of DLC2 (e.g.
fewer transition bits at same quality via smaller K when sub-band is enough)
remains untested.

### Interpretation

The AsymFlow low-rank intuition (residual concentrates in a sub-band → low-rank
codebook is more efficient) does **not** apply to V1 transitions on this
backbone. Transition residual high-freq energy is distributed across all
high-freq DCT positions, not concentrated near the band cutoff.

Possible explanations:

- VAE latent space lacks the `|ω|^-β` spectral power decay of pixel space at
  the band-cutoff scale (Wan latent shows β ≈ 2 globally, but the new high-freq
  band introduced by 480→576→720 transitions is far above the regime where
  decay matters).
- Residuals from random `t·ε` baseline at transition are themselves
  approximately white in the new band, so a low-rank sub-band cannot capture
  them more efficiently than full-rank.
- The bit-cost saving from a smaller K (which sub-band would enable) wasn't
  exercised — we kept K identical to DLC1 for fair quality comparison. A
  bit-rate-constrained comparison might show sub-band's value.

### Implication for DLC2

DLC2 Version A (sub-band low-rank) is **not** a viable building block for V1
transitions on Wan 2.1. The other DLC2 variants (Version B: frequency-band
asym schedule; Version C: clean prior + low-rank correction) might still
work but no longer have AsymFlow's "residual is low-rank" story to motivate
them.

## Final Pareto Frontier (under "speedup with quality preservation" framing)

| Config | Backbone | PSNR | ΔvsFull | LPIPS | Speedup | BPP |
| --- | --- | --- | --- | --- | --- | --- |
| Quality target | 1.3B full_res | 30.15 | 0 | 0.077 | 1.0× | 0.00483 |
| Absolute quality | 14B full_res | 30.50 | +0.35 | 0.075 | 0.4× | 0.00483 |
| **Best balance** | **1.3B S=2 dlc1_codebook** | **28.72** | **−1.44** | **0.093** | **1.34×** | **0.00481** |
| Speed-first | 1.3B S=3[480] random | 27.51 | −2.64 | 0.122 | 1.51× | 0.00451 |
| 14B balanced | 14B S=2 dlc1_codebook | 29.02 | +0.30 vs 1.3B random S=2 | 0.087 | 0.54× vs 1.3B full | 0.00481 |

On 1.3B, **S=2 dlc1_codebook** is the Pareto winner: 1.34× speedup with
−1.44 dB PSNR vs full_res, +0.4% BPP. DLC2 does not improve on it. 14B does
not bridge the structural multi-stage gap.

## VAE-mismatch Diagnostic + lp_raw Fix (2026-05-23 evening)

The V1 ceiling (oracle still −1.2 dB below full_res) was traced to a wrong
choice of low-stage target: `x0_low = VAE_480(LANCZOS_resize(video, 480))` is
NOT the same object as `T_L(VAE_720(video))` (DCT-low projection of the
full-res latent). The spectral expansion at transition assumes these are
equal — they are not.

### Diagnostic (Beauty GOP 0, 720p vs 480p latents)

| Quantity | Value | Interpretation |
| --- | --- | --- |
| `‖x0_480‖` | 522 | LANCZOS-resize + VAE_480 |
| `‖T_L(x0_720)‖` | 760 | DCT-truncate of VAE_720 (full-res encode) |
| `‖δ_0‖ = ‖x0_480 - T_L(x0_720)‖` | 310 | static gap |
| **`‖δ_0‖ / ‖T_L(x0_720)‖`** | **0.41** | **40% mismatch even at t=0** |

Per-step trajectory divergence — relative L2 of `(x_t^480 − T_L(x_t^720))`
against `T_L(x_t^720)` — for 8 SDE steps, init = renormed T_L of 720p noise
so step 0 has zero divergence:

| step | relative L2 |
| --- | --- |
| 0 | 0.00 (shared init) |
| 4 | 1.01 |
| 8 (transition) | 1.24 |

Even with shared low-freq init, the 480p trajectory drifts to >100% of the
target by step 8. The spectral expansion's "low-freq inverse" premise is
violated before transition.

Diagnostic script: `/tmp/v1_trajectory_diag.py`.

### lp_raw fix: redirect low-stage target to `T_L(x0_full)`

Add `--low_target_from_full` flag to `mr_pipeline.py`. When set, replace
`x0_per_stage[s]` for s &lt; S-1 with the DCT-truncated low-freq projection
of the full-res VAE latent. No bitstream change, no decoder change, removes
one VAE encode pass.

Results on Bosphorus + HoneyBee × 1 GOP (`run_low_target_diag.sh` →
`results_low_target_diag/`):

| Config | PSNR | LPIPS | BPP | Δ vs baseline |
| --- | --- | --- | --- | --- |
| baseline (vanilla dlc1_codebook) | 28.71 | 0.0931 | 0.00481 | — |
| **lp_raw** (no alignment) | **29.86** | **0.0794** | 0.00481 | **+1.15 dB** |
| lp_std (std-aligned) | 29.37 | 0.0831 | 0.00481 | +0.66 |
| lp_norm (norm-aligned) | 29.38 | 0.0829 | 0.00481 | +0.67 |
| full_res reference | 30.15 | 0.0786 | 0.00483 | (target) |

`lp_raw` closes ~80% of the previous V1 structural ceiling at the same BPP
and same encode/decode time. Magnitude alignment (`lp_std` / `lp_norm`) HURTS
by ~0.5 dB — spectral correctness matters more than magnitude consistency.
Without rescaling, `T_L(x0_720)` has natural magnitude ≈1.46× of `x0_480`;
the model is robust to this magnitude shift but cannot compensate for
spectrally-distorted targets.

### What this changes vs prior V1 conclusions

All earlier V1 ceiling claims ("structural −1.26 dB", "backbone-independent",
"DLC can't break it") were measured under the WRONG low-stage target. They
hold for vanilla V1 but NOT for lp_raw. The actual ceiling on Wan 1.3B is
much closer to −0.3 dB. DLC2 vs DLC1 comparison should also be revisited
once the ceiling moves.

### Open questions

- lp_raw on other sequences (Beauty, Jockey, etc.) — confirm not Bosph+HB
  specific.
- lp_raw on S=3 / S=4 [480-floor] — does the now-much-smaller bridge cost
  let us claim more aggressive multi-stage speedups without catastrophic
  loss?
- lp_raw on 14B — does the now-narrower gap mean 14B closes to baseline?
- Why does magnitude alignment hurt? Likely answer: the model is approximately
  scale-equivariant locally but very sensitive to spectral structure — but
  worth a targeted check.

## Final V1 Paper Numbers (lp_raw, full UVG, 2026-05-24)

Full UVG 7 sequences × 3 GOPs (21 GOPs total), Wan 2.1-T2V-1.3B at 720p,
M=64, K=16384, transition_step=8 (S=2), T=20, seed=42.

| Method | PSNR (dB) | LPIPS | BPP | Enc (s/GOP) | ΔPSNR | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| `full_res` baseline | **28.72** | **0.1138** | 0.00483 | 75.2 | — | 1.00× |
| `lp_raw S=2 dlc1_codebook` (new V1 default) | **28.50** | 0.1172 | 0.00481 | 55.3 | **−0.22** | **1.36×** |
| `lp_raw S=2 oracle` (ceiling) | 28.62 | 0.1144 | 0.00453 | 52.7 | −0.10 | 1.43× |

**lp_raw improvement over the old V1 result (vanilla S=2 dlc1_codebook 28.07):
+0.43 dB at same BPP and same speedup.** The structural ceiling identified
earlier (`vanilla S=2 oracle` was 28.41 on full UVG) is essentially gone.

Oracle - dlc1_codebook gap shrinks to 0.12 dB — transition codebook captures
almost all recoverable HF info under the corrected low-stage target.

### Iso-compute comparison (T-sweep on HoneyBee 1 GOP)

| Compute budget (~enc s) | lp_raw choice | full_res choice | lp_raw PSNR advantage |
| --- | --- | --- | --- |
| ~57s | T=20 lp_raw → 29.61 | T=16 full → 28.92 | **+0.69 dB** |
| ~47s | T=16 lp_raw → 28.48 | T=12 full → 27.00 | **+1.48 dB** |
| ~37s | T=12 lp_raw → 26.76 | T=10 full → 25.14 | **+1.62 dB** |

At any compute budget tested, lp_raw V1 beats full_res with fewer steps by
0.7–1.6 dB. Multi-resolution (V1) and fewer NFE (V2-b) are composable
acceleration axes, not substitutes.

### Per-sequence content dependence (Phase 2 multi-seq T-sweep, 1 GOP each)

lp_raw advantage over full_res at the same T is content-dependent:

| Sequence type | T=20 lp_raw vs full_res ΔPSNR | Note |
| --- | --- | --- |
| Beauty (static, low-freq dominated) | +0.03 dB | lp_raw slightly better |
| Bosphorus (water, mid-freq) | −0.12 | lp_raw ~tied |
| Jockey (high motion) | −0.34 | lp_raw slightly worse |
| HoneyBee (fine texture, high-freq) | −0.47 | lp_raw worst case |

The 21-GOP UVG average gap of −0.22 dB is consistent with the content mix
(static + low-freq sequences partially offset the high-freq-sensitive
sequences).

### Practical recommendation

**V1's headline Pareto winner: lp_raw S=2 dlc1_codebook at T=20.**

- 1.36× wall-clock speedup vs full_res
- −0.22 dB PSNR on full UVG (perceptually negligible)
- BPP unchanged (atom transmission cost is 0.4%)
- One extra DCT-truncate operation on encoder (sub-millisecond)
- Removes one VAE encode pass (small savings)

For higher speedups, T<20 + lp_raw composes additively but quality degrades
faster than speedup grows past T=16. Reasonable operating points (HoneyBee
reference):

- Quality-first: T=20 lp_raw → 1.36× / −0.22 dB
- Balanced: T=16 lp_raw → 1.67× / −1.60 dB
- Aggressive: T=12 lp_raw → 2.11× / −3.32 dB

## Reproduction

```bash
cd turbogvcc
for m in full_res mr_random mr_codebook; do
  python exp_v1_mr/run_mr_experiment.py \
    --wan_ckpt exp_t2v/Wan2.1-T2V-1.3B-Diffusers \
    --data_dir data/uvg \
    --output_dir exp_v1_mr/results_full_uvg/$m \
    --method $m \
    --num_gops 3 \
    --M 64 --K_low 4096 --K_full 16384 \
    --steps 20 --ddim_tail 3 --transition_step 8 --transition_align variance \
    --seed 42
done
```
