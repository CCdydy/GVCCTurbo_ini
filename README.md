# GVCC

Official code release for **[GVCC: Zero-Shot Video Compression via
Codebook-Driven Stochastic Rectified Flow](https://arxiv.org/abs/2603.26571)**
(arXiv:2603.26571).

A short qualitative example is in [`demo/`](demo/). Pipeline-level algorithm
notes are in [`PIPELINE.md`](PIPELINE.md). Ongoing post-paper research design
notes are in [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md).

## GVCCTurbo Snapshot

Current post-paper GVCCTurbo direction:

```text
full-latent-consistent multi-resolution GVCC
+ DLC1 transition codebook
+ vectorized per-frame codebook generation
+ x0-cache Wan-call skipping with transition warmup
```

Current full-UVG headline candidate:

| Method | PSNR | BPP | Dec | Speedup |
| --- | ---: | ---: | ---: | ---: |
| full-res baseline | 28.72 | 0.00483 | ~34s | 1.00x |
| V1 lp_raw | 28.50 | 0.00511 | 25s | 1.36x |
| V1 lp_raw + x0-cache p=2 + warmup | 28.45 | 0.00511 | 19s | 1.79x |

The cache uses clean prediction, not velocity:

```text
x0_cache = x_t - t * u_theta(x_t,t)
u_skipped = (x_t_current - x0_cache) / t
```

This keeps all SDE/codebook correction steps while skipping some Wan calls.
Transition warmup resets the cache at resolution changes and delays the first
post-transition skip by one step without increasing the number of Wan calls.
See [`MEMORY.md`](MEMORY.md) for the current research-state index.

## Method Notes (Post-Paper Turbo)

Two key add-ons over the published GVCC paper. Both are small but non-obvious
modifications grounded in trajectory geometry; each was discovered by chasing a
specific failure mode.

### 1. x0-cache: a one-step-radius skip for the Wan velocity

GVCC's per-step cost is dominated by the Wan velocity evaluation
`u_theta(x_t, t)`. The instinctive way to skip is to **cache the velocity** —
the previous step's direction — and reuse it:

```text
NAIVE:  u_skipped = u_theta(x_prev, t_prev)        # cache the "direction"
```

This silently drifts because `u_theta(x, t)` depends on both arguments and the
trajectory is moving away from `x_prev`. The reused direction is computed at a
stale state, not the current one.

The fix is to cache the **destination** instead — the model's MMSE estimate of
where the clean image lies — and recompute the direction from the current
state at evaluation time:

```text
when we DO call Wan at step k:
  cached_x0 = x_t - t_k * u_theta(x_t, t_k)        # cache the trajectory endpoint
when we SKIP Wan at step k+1:
  u_approx  = (x_{t,k+1} - cached_x0) / t_{k+1}    # rebuild direction from
                                                     # *current* state and *current* t
```

Geometric reading: under the linear flow interpolant
`x_t = (1-t)*x_0 + t*epsilon`, the clean endpoint `x_0` is the same point
regardless of which `t` we sit on. So caching the endpoint and recovering the
direction from `(x_current - endpoint) / t_current` is dimensionally and
geometrically faithful — every skipped step queries the cache *from where we
actually are*, not from where we were one step ago. The full SDE/codebook
correction at the skipped step is preserved; only the `u_theta` query is
replaced.

**Pitfall, validated by ablation.** We attempted an adaptive variant
`adaptive_x0_cache` with a `max_consecutive_skips` cap. Two consecutive skips
already destabilize: on HoneyBee the cap=2 / threshold=0.15 schedule produces
`PSNR -0.39 dB for -1s wall-clock`, and a no-cap variant collapses by ~3 dB.
The mechanism is recursive cache staleness: the second skipped step plugs an
**already-extrapolated** `x_t` back into the same linearization, compounding
its error. Empirical conclusion: **clean-endpoint caching has an effective
radius of one step**; `period=2` + `transition_warmup` is the Pareto-optimal
operating point under this caching scheme. The fixed alternating pattern is
strictly faster than any data-adaptive policy that obeys the same radius.

### 2. Multi-resolution + `lp_raw`: low-stage target replacement

The natural MR formulation runs the SDE at progressively higher resolutions:

```text
stage 0:  T_low  = VAE_low(downsample(video))
stage 1:  T_full = VAE_full(video)
```

Naively, `T_low` is computed by VAE-encoding the spatially-downsampled video.
Empirically this hits a **structural ceiling**: on full UVG 720p with the 1.3B
backbone, S=2 with random or DLC1 transition is capped at PSNR -1.44 dB
vs full-res, and the gap **does not close** with joint oracle (-1.26 dB) or a
14B backbone (-1.24 dB). Two-stage MR appears to leave ~1 dB on the table even
under optimal high-frequency information transfer.

The cause is that the two stages live in **different spatial domains**.
`VAE_low(downsample(video))` and `VAE_full(video)` are two separate
encoder evaluations on different spatial supports, and their outputs are
**not** related by a clean spectral truncation. In particular,

```text
VAE_low( downsample(video) )  is NOT  T_L( VAE_full(video) )
```

even though both produce the same shape `(C, F, H_low, W_low)`. The
non-linearity in VAE and the difference in receptive field on the two pixel
supports yield systematically different low-band content. Measured on UVG:
`|| VAE_low - T_L(VAE_full) || / || T_L(VAE_full) || = 0.41` at t=0, cosine
similarity 0.89.

But the decoder spectrally lifts the low stage into the high domain at
transition:

```text
x_bar^{s_1} = T^{-1}( Embed( T(x^{s_0}) ) + t*epsilon_H )
```

so its low band can only ever equal whatever the low-stage trajectory
converged to. If that's `VAE_low(downsample(video))`, then after lift the low
band is "the wrong low-band of the wrong VAE." The high-band fill (random or
DLC1) has no chance of recovering the mismatch.

**Fix (`low_target_from_full`): unify both stages in the high-resolution
domain.** Encode the video once at full resolution, then project that single
latent down via DCT truncation to serve as the low-stage target:

```text
T_low  <-  T_L( VAE_full(video) )       # high-res latent, DCT-projected down
```

Both stages now refer to the **same latent**, just observed at different
spectral bandwidths. Transition stops being a domain change and becomes a
pure band expansion: the low stage converges to `T_L(z_full)`, the high
stage adds `T_H(z_full)`, and the two sum to `z_full`. No extra bits are
transmitted; only the encoder's codebook-matching target moves to a
spectrally-consistent reference.

On full UVG 720p the structural ceiling collapses from -1.44 dB to -0.22 dB
at the same BPP. The joint-oracle ceiling under `lp_raw` is -0.10 dB, so
DLC1 captures ~95% of the remaining recoverable information.

**Pitfall, validated by ablation.** A natural sub-fix is to magnitude-align
`T_L(VAE_full)` to `VAE_low` (per-frame std-match or norm-match). This
**hurts** by ~0.5 dB on Bosphorus+HoneyBee: raw `T_L` is the spectrally
correct object, and magnitude rescaling re-introduces the original mismatch
in a smoother form. Codebook-driven trajectories prioritize spectral
correctness over magnitude consistency.

### 3. DLC1 transition codebook (target-aware high-freq fill)

At the resolution transition the decoder has only the already-trajected low
stage `x_{t_i}^{s_i}` and a free Gaussian sample `epsilon_{H_i}` in the newly
opened high band. The oracle high-frequency state at timestep `t_i` is

```text
x_{t_i, oracle}^{H_i} = (1 - t_i) * x_0^{H_i} + t_i * epsilon_{H_i}
                                  ^^^^^^^^^^^
                                  encoder-known clean high-freq band
```

so the trajectory-consistent residual the codebook should transmit is

```text
r_{t_i}^{H_i} = (1 - t_i) * x_0^{H_i}     where    x_0^{H_i} = T_{H_i}(x_0^{s_{i+1}})
```

This is the DLC1-codebook payload at the transition: it adds **one** atom set
(per latent frame) to the bitstream, replacing pure-noise high-frequency
expansion with target-aware codebook innovation. On full UVG 1.3B at S=2,
DLC1 recovers +0.43 dB over random transition at <1% BPP overhead.

See [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md) for the full derivation,
falsified alternatives (DLC1-pure no-codebook, DLC2 sub-band low-rank), and
the post-`lp_raw` Pareto table.

## Install

Tested with Python 3.10 and CUDA 13.0. Any environment that runs Wan2.1 should
run GVCC.

```bash
# 1. PyTorch (CUDA 13.0 wheel; adjust the index URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. Remaining dependencies
pip install -r requirements.txt
```

`flash_attn` is optional; `wan/modules/attention.py` falls back to PyTorch SDPA.

## Model Weights

Download the three Wan2.1-14B checkpoints from HuggingFace into the indicated
local paths (real directories or symlinks to a shared cache).

| Method | HuggingFace repo | Local path |
| --- | --- | --- |
| T2V | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `exp_t2v/Wan2.1-T2V-14B-Diffusers/` |
| I2V | `Wan-AI/Wan2.1-I2V-14B-720P` | `exp_i2v/Wan2.1-I2V-14B-720P/` |
| FLF2V | `Wan-AI/Wan2.1-FLF2V-14B-720P` | `exp_flf2v/Wan2.1-FLF2V-14B-720P/` |

Convenience download scripts: `bash exp_{t2v,i2v,flf2v}/download_*.sh`.

For low-VRAM T2V, use `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` at
`exp_t2v/Wan2.1-T2V-1.3B-Diffusers/` and pass `--wan_ckpt
exp_t2v/Wan2.1-T2V-1.3B-Diffusers`. I2V also has a 480P variant.

## Data

Download the seven [UVG-1080p](https://ultravideo.fi/) sequences (Beauty,
Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide) anywhere under
`data/uvg/`. The loader (`uvg_data.py`) recursively scans for `*.yuv`.

## Run

```bash
# T2V: codebook only
bash exp_t2v/run_t2v.sh
bash exp_t2v/run_t2v_1080p.sh

# I2V: autoregressive with tail-residual correction
bash exp_i2v/run.sh
bash exp_i2v/run_1080p.sh

# FLF2V: first/last-frame conditioning
bash exp_flf2v/run_flf2v.sh
bash exp_flf2v/run_flf2v_1080p.sh

# Paper RD / ablation sweeps
bash exp_param_sweep/run_sweep.sh
bash exp_flf2v/run_rd_sweep.sh
bash exp_appendix_rd_sweep/run.sh
```

VRAM scales with the chosen backbone. With Wan2.1-14B: ~48 GB at 720p,
~70 GB at 1080p. The 1.3B T2V variant fits on a single consumer GPU. The
codebook/SDE pipeline adds negligible memory on top of the generator.

Pass `--help` to any `run_*_experiment.py` for the full parameter list
(`M`, `K`, `steps`, `g_scale`, `ddim_tail`, ...).

## Output Layout

```text
exp_{method}/results_{resolution}/
  summary.json
  {sequence}/
    original.mp4
    reconstructed_full.mp4
    gop{N}/
      metrics.json
      reconstructed.mp4
      codebook.tdcm
```

## Citation

```bibtex
@article{zeng2026gvcc,
  title   = {GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow},
  author  = {Zeng, Ziyue and Su, Xun and Liu, Haoyuan and Lu, Bingyu and Tatsumi, Yui and Watanabe, Hiroshi},
  journal = {arXiv preprint arXiv:2603.26571},
  year    = {2026}
}
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). The `wan/` subpackage is vendored from
[Wan2.1](https://github.com/Wan-Video/Wan2.1) (Apache-2.0); upstream copyright
headers are preserved. See [NOTICE](NOTICE) for the full attribution list.
