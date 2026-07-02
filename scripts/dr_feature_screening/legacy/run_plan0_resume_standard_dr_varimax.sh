#!/usr/bin/env bash
set -euo pipefail

# Resume/extend Plan 0 standard DR caches (PCA/FA/FactoSig varimax) in place.
# Defaults target the active experiment discussed in chat.

EXP_DIR="/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
KS="20,40,60"
SEEDS="1,2,3,4,5"
METHODS="pca,fa,factosig"
FACTOSIG_ORDER="score_variance"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_SCRIPT="$SCRIPT_DIR/resume_plan0_standard_dr.py"

usage() {
  cat <<'EOF'
Usage:
  run_plan0_resume_standard_dr_varimax.sh [--exp-dir PATH] [--ks LIST] [--seeds LIST]

Options:
  --exp-dir PATH   Existing Plan 0 experiment directory.
  --ks LIST        Comma/space-separated K list (default: 20,40,60).
  --seeds LIST     Comma/space-separated seeds (default: 1,2,3,4,5).
  -h, --help       Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-dir)
      EXP_DIR="$2"
      shift 2
      ;;
    --ks)
      KS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$EXP_DIR" ]]; then
  echo "Experiment directory not found: $EXP_DIR" >&2
  exit 1
fi
if [[ ! -f "$RESUME_SCRIPT" ]]; then
  echo "Resume script not found: $RESUME_SCRIPT" >&2
  exit 1
fi

LOG_DIR="$EXP_DIR/analysis/plan0"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/resume_plan0_standard_dr_varimax_${TS}.log"

echo "[$(date -Iseconds)] Starting varimax resume run"
echo "EXP_DIR=$EXP_DIR"
echo "KS=$KS"
echo "SEEDS=$SEEDS"
echo "LOG_FILE=$LOG_FILE"

python -u "$RESUME_SCRIPT" \
  --experiment-dir "$EXP_DIR" \
  --ks "$KS" \
  --seeds "$SEEDS" \
  --methods "$METHODS" \
  --factosig-order-factors-by "$FACTOSIG_ORDER" \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] Finished varimax resume run"
