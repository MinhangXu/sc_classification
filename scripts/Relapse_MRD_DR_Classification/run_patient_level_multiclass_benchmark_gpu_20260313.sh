#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/home/minhang/mds_project/sc_classification/experiments/20260313_192016_relapse_mrd_dr_classification"
LOG="$RUN_ROOT/multiclass_benchmark_gpu_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RUN_ROOT"

python -u "/home/minhang/mds_project/sc_classification/scripts/Relapse_MRD_DR_Classification/run_patient_level_multiclass_benchmark.py" \
  --dr-output-dir "$RUN_ROOT" \
  --patients "P01,P02,P03,P04,P05,P07,P09,P13" \
  --methods "pca,fa,factosig" \
  --cv-folds 5 \
  --cv-repeats 10 \
  --random-state 42 \
  --alpha-log10-min -4 \
  --alpha-log10-max 5 \
  --alpha-num 20 \
  --enet-l1-ratios "0.1,0.5,0.9" \
  --ml-backend gpu \
  --strict-gpu \
  2>&1 | tee -a "$LOG"
