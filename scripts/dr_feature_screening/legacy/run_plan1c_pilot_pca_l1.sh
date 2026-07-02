#!/usr/bin/env bash
set -euo pipefail

EXP="/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
SCRIPT="/home/minhang/mds_project/sc_classification/scripts/comprehensive_run/run_plan1c_supervised_latent_benchmark.py"
OUT_SUBDIR="analysis/plan1c_supervised_latent_k40_pilot_pca_l1_large"
RUN_DIR="$EXP/$OUT_SUBDIR"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"

echo "Starting Plan1C larger pilot: PCA K=40 pooled CV L1"
echo "EXP=$EXP"
echo "OUT=$RUN_DIR"
echo "LOG=$LOG_FILE"

python -u "$SCRIPT" \
  --experiment-dir "$EXP" \
  --k 40 \
  --methods pca \
  --modes pooled \
  --penalties l1 \
  --downsampling-variants none \
  --cv-folds 5 \
  --cv-repeats 5 \
  --alpha-log10-min -4 \
  --alpha-log10-max 5 \
  --alpha-num 10 \
  --output-subdir "$OUT_SUBDIR" \
  2>&1 | tee -a "$LOG_FILE"

echo "Pilot run completed."
