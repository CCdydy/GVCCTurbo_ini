# V2-a Spectral Codebook Scheduling

V2-a keeps the full-resolution GVCC trajectory unchanged and changes only
codebook selection/reconstruction frequency bands.

## Stage 0 Result

| Mode | PSNR | LPIPS | Note |
| --- | ---: | ---: | --- |
| `full_band_baseline` | 30.09 | 0.0786 | Original GVCC. |
| `dct_all_band_control` | 30.19 | 0.0776 | V2-a code path with all bands open; matches baseline. |
| `stage0_masked_random` | 16.48 | 0.6025 | Three-stage band schedule; catastrophic failure. |

Conclusion:

```text
The V2-a DCT implementation path is correct, but direct band-limited
innovation is not compatible with the pretrained Wan SDE trajectory.
```

The failed Stage 0 used a masked random codebook:

```text
steps 0-5:    low only
steps 6-11:   low + mid
steps 12-16:  low + mid + high
```

This reproduces the V0 failure mode because the SDE update only sees the final
innovation spectrum. A band-limited `z_t*` pushes the next state OOD even if
the atom was selected in the intended band.

## Next: Stage 0b Spectrum-Preserving V2-a

Keep the active-band codebook selection, but fill inactive bands with
deterministic Gaussian noise so the injected innovation remains full-spectrum:

```text
T(z_t*) = M_B * T(z_codebook) + M_off * eta
eta ~ N(0, I)
```

Then normalize the full `z_t*` before the usual GVCC SDE update. This keeps
codebook bits focused on the scheduled band without changing the distribution
class expected by Wan.

The older commands below are kept for reproducibility and negative controls.

To launch the first matrix, including Stage 0b:

```bash
bash exp_v2a/run_stage0.sh
```

The launcher defaults to `checkpoints/Wan2.1-T2V-1.3B-Diffusers`,
`Bosphorus HoneyBee`, one GOP, 720p, `T=20`, `M=64`, `K=16384`.
Override any of these with environment variables, for example:

```bash
WAN_CKPT=checkpoints/Wan2.1-T2V-14B-Diffusers NUM_GOPS=3 bash exp_v2a/run_stage0.sh
```

Run Stage 0b on Bosphorus/HoneyBee:

```bash
python exp_v0_spectral/run_v0.py \
  --output_dir exp_v2a/results/stage0b_spectrum_preserving \
  --data_dir data/uvg \
  --wan_ckpt checkpoints/Wan2.1-T2V-1.3B-Diffusers \
  --sequences Bosphorus HoneyBee \
  --num_gops 1 \
  --height 720 --width 1280 \
  --steps 20 --ddim_tail 3 \
  --M 64 --K 16384 \
  --spectral_schedule identity \
  --codebook_mode v2a_spectrum_preserving
```

Run the failed Stage 0a negative control:

```bash
python exp_v0_spectral/run_v0.py \
  --output_dir exp_v2a/results/stage0_masked_random \
  --data_dir data/uvg \
  --wan_ckpt checkpoints/Wan2.1-T2V-1.3B-Diffusers \
  --sequences Bosphorus HoneyBee \
  --num_gops 1 \
  --height 720 --width 1280 \
  --steps 20 --ddim_tail 3 \
  --M 64 --K 16384 \
  --spectral_schedule identity \
  --codebook_mode v2a_masked_random
```

Baseline:

```bash
python exp_v0_spectral/run_v0.py \
  --output_dir exp_v2a/results/full_band_baseline \
  --data_dir data/uvg \
  --wan_ckpt checkpoints/Wan2.1-T2V-1.3B-Diffusers \
  --sequences Bosphorus HoneyBee \
  --num_gops 1 \
  --height 720 --width 1280 \
  --steps 20 --ddim_tail 3 \
  --M 64 --K 16384 \
  --spectral_schedule identity \
  --codebook_mode full
```

DCT-domain all-band control:

```bash
python exp_v0_spectral/run_v0.py \
  --output_dir exp_v2a/results/dct_all_band_control \
  --data_dir data/uvg \
  --wan_ckpt checkpoints/Wan2.1-T2V-1.3B-Diffusers \
  --sequences Bosphorus HoneyBee \
  --num_gops 1 \
  --height 720 --width 1280 \
  --steps 20 --ddim_tail 3 \
  --M 64 --K 16384 \
  --spectral_schedule identity \
  --codebook_mode v2a_masked_random \
  --v2a_low_until 0 \
  --v2a_mid_until 0
```

The control above keeps the V2-a DCT-domain codebook path but opens all
frequencies at every step. It helps separate the schedule effect from the
random realization difference between spatial-domain and DCT-domain Gaussian
codebooks.

Summarize completed runs:

```bash
python exp_v2a/summarize_v2a.py --results_dir exp_v2a/results
```
