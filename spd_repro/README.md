# SPD Reproduction

Re-implementation of **Spectral Progressive Diffusion** (Xiao et al., arXiv:2605.18736)
for use on Wan 2.1-T2V-1.3B as a self-contained baseline.

This folder is **independent** of the turboddcm research direction —
it reproduces only SPD's *generation* pipeline (training-free spatial
acceleration via progressive resolution growth along the denoising
trajectory). Frequency-Scheduled DDCM (the next paper's hypothesis)
is a separate effort and lives elsewhere.

`DLC1` denotes the SPD-side improvement branch: changes that improve SPD's own
progressive-generation mechanism, such as spectral expansion, transition
alignment, schedule design, or multi-resolution stability. DLC1 stays separate
from GVCC bitstream and codebook-selection changes.

`DLC2` denotes the AsymFlow-inspired structured-innovation branch. It studies
how rank-asymmetric / low-rank noise ideas can guide GVCC codebook subspaces.

The canonical research notes for GVCCTurbo, DLC1, and DLC2 are consolidated in
[`../RESEARCH_NOTES.md`](../RESEARCH_NOTES.md). [`DLC1.md`](DLC1.md) and
[`DLC2.md`](DLC2.md) are pointer files only.

## Paper section → file mapping

| SPD paper | File |
|---|---|
| Sec 3.2 (spectral autoregression observation) | `power_spectrum.py` |
| Sec 4.1 spectral noise expansion (3-step algorithm) | `spectral_noise_expansion.py` |
| Sec 4.1 timestep alignment (Eqs 5–6) | `timestep_alignment.py` |
| Sec 4.2 Prop 1 (per-frequency δ-optimal activation time) | `optimal_schedule.py` |
| Sec 4.2 Prop 2 (per-resolution δ-optimal transition time) | `optimal_schedule.py` |
| Sec 5.3 latent-space video generation experiments | `spd_pipeline.py`, `run_spd_wan.py` |
| Table 3 (Wan 2.1 T2V-1.3B 720p) | `run_spd_wan.py` |
| DCT/DWT/FFT utility | `dct_utils.py` |

## Validation target

Paper Table 3 on Wan 2.1-T2V-1.3B 720p:
- Wan 2.1 (50 steps): 1.00× speedup, baseline
- Ours (S=2): **2.03× speedup**, VBench scores within ±2% of baseline

We aim to match within ~10% of these numbers (re-implementation noise +
single-machine variance).

## Layout

```
spd_repro/
├── README.md                      ← this file
├── dct_utils.py                   ← DCT/IDCT/DWT + radial freq binning
├── power_spectrum.py              ← P_ω measurement on Wan latents
├── spectral_noise_expansion.py    ← SPD Sec 4.1 algorithm
├── timestep_alignment.py          ← Eqs 5-6 rescaling
├── optimal_schedule.py            ← Prop 1, 2 schedules
├── spd_pipeline.py                ← inference pipeline (training-free)
├── run_spd_wan.py                 ← entry, reproduces Table 3 row
└── eval_vbench.py                 ← VBench eval subset for comparison
```

## Status

Re-implementation from paper only — SPD authors have not released code yet
(project page lists "Code: coming soon" as of 2026-05). An unofficial
ComfyUI implementation exists (ruwwww/ComfyUI-SPEED) but supports only
the Anima model, not Wan or FLUX.
