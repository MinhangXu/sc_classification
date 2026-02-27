#!/usr/bin/env bash
set -euo pipefail

# Full Plan1C run: all DR methods, pooled + per-patient CV,
# downsampling none/random, and L1/L2/ElasticNet grids.

EXP="/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
SCRIPT="/home/minhang/mds_project/sc_classification/scripts/comprehensive_run/run_plan1c_supervised_latent_benchmark.py"
OUT_SUBDIR="analysis/plan1c_supervised_latent_k40_full_all"
RUN_DIR="$EXP/$OUT_SUBDIR"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR"

echo "Starting Plan1C full run (K=40, all methods/penalties/modes)"
echo "EXP=$EXP"
echo "OUT=$RUN_DIR"
echo "LOG=$LOG_FILE"

python -u "$SCRIPT" \
  --experiment-dir "$EXP" \
  --k 40 \
  --methods pca,fa,factosig,factosig_promax,cnmf \
  --modes pooled,per_patient \
  --penalties l1,l2,elasticnet \
  --downsampling-variants none,random \
  --cv-folds 5 \
  --cv-repeats 10 \
  --alpha-log10-min -4 \
  --alpha-log10-max 5 \
  --alpha-num 20 \
  --enet-l1-ratios 0.1,0.5,0.9 \
  --low-malignant-threshold 10 \
  --skip-malignant-leq 1 \
  --severe-ratio-after-threshold 20 \
  --output-subdir "$OUT_SUBDIR" \
  2>&1 | tee -a "$LOG_FILE"

echo "Plan1C full run completed."
