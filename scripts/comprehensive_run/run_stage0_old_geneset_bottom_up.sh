#!/usr/bin/env bash
set -euo pipefail

# Stage 0 bottom-up old-geneset screen:
# - materialize one-geneset and one-biology-group panels
# - add HVG/all-filtered data-driven controls
# - run the same DR metric layer used by the top-down pruning analysis

SC_ROOT="/home/minhang/mds_project/sc_classification"
EXP_DIR="$SC_ROOT/experiments/20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0"
HVG_EXP_DIR="$SC_ROOT/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
METHODS="fa,factosig,pca"
KS="5,10,20,40"
PANEL_TYPES="single_geneset_only,single_group_only,hvg_anchor_control,hvg_size_matched_control,full_control,core_only"
OUT_DIR=""
MAKE_UMAPS="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_old_geneset_pruning_metrics.py"

usage() {
  cat <<'EOF'
Usage:
  run_stage0_old_geneset_bottom_up.sh [options]

Options:
  --exp-dir PATH        Old-geneset Plan 0 experiment directory.
  --hvg-exp-dir PATH    HVG Plan 0 experiment directory used for HVG controls.
  --methods LIST        DR methods (default: fa,factosig,pca).
  --ks LIST             K grid (default: 5,10,20,40).
  --panel-types LIST    Panel types to run.
  --out-dir PATH        Optional output directory.
  --make-umaps          Also make gated UMAPs.
  -h, --help            Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --hvg-exp-dir) HVG_EXP_DIR="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --ks) KS="$2"; shift 2 ;;
    --panel-types) PANEL_TYPES="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --make-umaps) MAKE_UMAPS="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$EXP_DIR/analysis/stage0_old_geneset_bottom_up"
fi

CMD=(
  python -u "$RUNNER"
  --sc-root "$SC_ROOT"
  --exp-dir "$EXP_DIR"
  --hvg-exp-dir "$HVG_EXP_DIR"
  --panel-families "top_down,bottom_up,hvg_controls"
  --panel-types "$PANEL_TYPES"
  --methods "$METHODS"
  --ks "$KS"
  --out-dir "$OUT_DIR"
)

if [[ "$MAKE_UMAPS" == "1" ]]; then
  CMD+=(--make-umaps)
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/stage0_bottom_up_${TS}.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date -Iseconds)] Starting Stage 0 bottom-up screen"
echo "EXP_DIR=$EXP_DIR"
echo "HVG_EXP_DIR=$HVG_EXP_DIR"
echo "LOG_FILE=$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date -Iseconds)] Finished Stage 0 bottom-up screen"
