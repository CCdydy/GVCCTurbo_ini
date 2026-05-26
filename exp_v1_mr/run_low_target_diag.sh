#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_OUT="exp_v1_mr/results_low_target_diag"
CONDA_BIN="${CONDA_BIN:-/home/rog/miniconda3/condabin/conda}"
COMMON=(
  --method mr_dlc1_codebook
  --sequences Bosphorus HoneyBee
  --num_gops 3
  --transition_step 8
  --transition_align variance
)

"$CONDA_BIN" run -n wan21 python exp_v1_mr/run_mr_experiment.py \
  "${COMMON[@]}" \
  --output_dir "$BASE_OUT/baseline"

"$CONDA_BIN" run -n wan21 python exp_v1_mr/run_mr_experiment.py \
  "${COMMON[@]}" \
  --low_target_from_full \
  --low_target_align none \
  --output_dir "$BASE_OUT/lp_raw"

"$CONDA_BIN" run -n wan21 python exp_v1_mr/run_mr_experiment.py \
  "${COMMON[@]}" \
  --low_target_from_full \
  --low_target_align std \
  --output_dir "$BASE_OUT/lp_std"

"$CONDA_BIN" run -n wan21 python exp_v1_mr/run_mr_experiment.py \
  "${COMMON[@]}" \
  --low_target_from_full \
  --low_target_align norm \
  --output_dir "$BASE_OUT/lp_norm"
