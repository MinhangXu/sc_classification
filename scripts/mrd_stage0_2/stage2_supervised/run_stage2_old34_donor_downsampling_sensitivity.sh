#!/usr/bin/env bash
set -euo pipefail

# Overnight donor-downsampling expansion (v2).
#
# Recommended launch:
#   screen -S donor_ds_v2
#   bash scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.sh
#
# Extra arguments are forwarded to every Python invocation after the script's
# own defaults (for example --gpu-id 1 or --shard-index 0 --shard-count 4).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_stage2_old34_donor_downsampling_sensitivity.py"
EXPERIMENT_DIR="${SC_ROOT}/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
CONDA_ENV="${CONDA_ENV:-sc-classification-rapids-py311}"
GPU_ID="${GPU_ID:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_V1_QA="${SKIP_V1_QA:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  bash $0 [extra python args...]
  bash $0 --merge-only

Environment:
  CONDA_ENV       default: sc-classification-rapids-py311
  GPU_ID          default: 0  (sets --gpu-id)
  SKIP_SMOKE      1 to skip the 2-target smoke test
  SKIP_V1_QA      1 to skip the 7-target v1 overlap QA gate
  PREPARE_ONLY    1 to write manifests and check feature paths only

If compute finished but merge crashed, pick up with:
  bash $0 --merge-only

Screen:
  screen -S donor_ds_v2
  bash ${SCRIPT_DIR}/run_stage2_old34_donor_downsampling_sensitivity.sh
EOF
  exit 0
fi

LOG_DIR="${EXPERIMENT_DIR}/logs/donor_sensitivity_v2"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/donor_downsampling_v2_$(date -u +%Y%m%d_%H%M%S).log"

PYTHON_CMD=(conda run -n "${CONDA_ENV}" --no-capture-output python -u)
COMMON=(
  "${RUNNER}"
  --experiment-dir "${EXPERIMENT_DIR}"
  --gpu-id "${GPU_ID}"
  --backend cuml
  --strict-gpu
)

run_step() {
  echo "[$(date -Iseconds)] $*" | tee -a "${RUN_LOG}"
  "${PYTHON_CMD[@]}" "${COMMON[@]}" "$@" 2>&1 | tee -a "${RUN_LOG}"
}

echo "[$(date -Iseconds)] Donor-downsampling v2 start" | tee -a "${RUN_LOG}"
echo "Runner: ${RUNNER}" | tee -a "${RUN_LOG}"
echo "Log: ${RUN_LOG}" | tee -a "${RUN_LOG}"
echo "GPU_ID=${GPU_ID} CONDA_ENV=${CONDA_ENV}" | tee -a "${RUN_LOG}"

if [[ "${1:-}" == "--merge-only" ]]; then
  shift
  run_step --merge-only --arm both --target-set all_dr_plus_direct "$@"
  echo "[$(date -Iseconds)] Merge-only finished" | tee -a "${RUN_LOG}"
  exit 0
fi

if [[ "${PREPARE_ONLY}" -eq 1 ]]; then
  run_step --prepare-only --target-set all_dr_plus_direct --arm both "$@"
  echo "[$(date -Iseconds)] Prepare-only finished" | tee -a "${RUN_LOG}"
  exit 0
fi

if [[ "${SKIP_SMOKE}" -ne 1 ]]; then
  run_step --smoke-test --arm both --resume "$@"
fi

if [[ "${SKIP_V1_QA}" -ne 1 ]]; then
  run_step --arm both --target-set v1_overlap --qa-against-v1 --resume "$@"
fi

run_step --arm both --target-set all_dr_plus_direct --resume --continue-on-error "$@"

echo "[$(date -Iseconds)] Donor-downsampling v2 finished" | tee -a "${RUN_LOG}"
echo "Tables: ${EXPERIMENT_DIR}/analysis/scorecards/stage2_diagnostic/donor_sensitivity_v2"
echo "Log: ${RUN_LOG}"
