#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data/uvg}"
WAN_CKPT="${WAN_CKPT:-checkpoints/Wan2.1-T2V-1.3B-Diffusers}"
OUT_DIR="${OUT_DIR:-exp_v2a/results}"
SEQUENCES="${SEQUENCES:-Bosphorus HoneyBee}"
NUM_GOPS="${NUM_GOPS:-1}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
STEPS="${STEPS:-20}"
DDIM_TAIL="${DDIM_TAIL:-3}"
M="${M:-64}"
K="${K:-16384}"
G_SCALE="${G_SCALE:-3.0}"
FLOW_SHIFT="${FLOW_SHIFT:-3.0}"
SEED="${SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a SEQ_ARGS <<< "$SEQUENCES"

common_args=(
  --wan_ckpt "$WAN_CKPT"
  --data_dir "$DATA_DIR"
  --sequences "${SEQ_ARGS[@]}"
  --num_gops "$NUM_GOPS"
  --height "$HEIGHT"
  --width "$WIDTH"
  --steps "$STEPS"
  --ddim_tail "$DDIM_TAIL"
  --M "$M"
  --K "$K"
  --g_scale "$G_SCALE"
  --guidance_scale 1.0
  --flow_shift "$FLOW_SHIFT"
  --seed "$SEED"
  --spectral_schedule identity
)

python exp_v0_spectral/run_v0.py \
  "${common_args[@]}" \
  --output_dir "$OUT_DIR/full_band_baseline" \
  --codebook_mode full \
  $EXTRA_ARGS

python exp_v0_spectral/run_v0.py \
  "${common_args[@]}" \
  --output_dir "$OUT_DIR/dct_all_band_control" \
  --codebook_mode v2a_masked_random \
  --v2a_low_until 0 \
  --v2a_mid_until 0 \
  $EXTRA_ARGS

python exp_v0_spectral/run_v0.py \
  "${common_args[@]}" \
  --output_dir "$OUT_DIR/stage0_masked_random" \
  --codebook_mode v2a_masked_random \
  $EXTRA_ARGS

python exp_v0_spectral/run_v0.py \
  "${common_args[@]}" \
  --output_dir "$OUT_DIR/stage0b_spectrum_preserving" \
  --codebook_mode v2a_spectrum_preserving \
  $EXTRA_ARGS

python exp_v2a/summarize_v2a.py --results_dir "$OUT_DIR"
