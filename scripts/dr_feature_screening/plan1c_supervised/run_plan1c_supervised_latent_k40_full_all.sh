#!/usr/bin/env bash
set -euo pipefail

LOG="/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886/analysis/plan1c_supervised_latent_k40_full_all/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

python -u "/home/minhang/mds_project/sc_classification/scripts/dr_feature_screening/plan1c_supervised/run_plan1c_supervised_latent_benchmark.py" \
  --experiment-dir "/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886" \
  --k 40 \
  --methods "pca,fa,factosig,factosig_promax,cnmf" \
  --modes "pooled,per_patient" \
  --penalties "l1,l2,elasticnet" \
  --downsampling-variants "none,random" \
  --cv-folds 5 --cv-repeats 10 \
  --alpha-log10-min -4 --alpha-log10-max 5 --alpha-num 20 \
  --enet-l1-ratios "0.2,0.5,0.8" \
  --low-malignant-threshold 10 --skip-malignant-leq 1 --severe-ratio-after-threshold 20 \
  --ml-backend gpu --strict-gpu \
  --output-subdir "models/classification/plan1c_supervised_latent_k40_full_all" \
  2>&1 | tee -a "$LOG"
