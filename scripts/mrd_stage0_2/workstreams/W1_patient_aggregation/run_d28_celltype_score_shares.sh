#!/usr/bin/env bash
# D28 cell-type-stratified score-energy shares per patient (W1.9), resumable wrapper.
#
# The full run takes about a minute, so it is fine in an agent shell; a screen session is optional:
#   screen -S W1_d28
#   bash scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.sh [extra args]
#   (detach with Ctrl-a d; reattach with screen -r W1_d28)
#
# Default: all 41 biological panels x {pca, fa} x K {5,10,20,40}, --resume, 12 single-thread workers
# (shared-server cap). Extra args pass through, e.g. --merge-only, --workers 4 or --out-root <scratch dir>.
# Python: $SC_PYTHON if set, else the lab conda env under $HOME, else python on PATH.
# Paths resolve through sc_classification.utils.paths (SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$HOME/miniconda3/envs/sc-classification-rapids-py311"
if [[ -n "${SC_PYTHON:-}" ]]; then
  PY=("$SC_PYTHON")
elif [[ -x "$ENV_DIR/bin/python" ]]; then
  PY=(env "CONDA_PREFIX=$ENV_DIR" "$ENV_DIR/bin/python")
else
  PY=(python)
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$REPO"
echo "[$(date -u +%FT%TZ)] D28 start: ${PY[*]} run_d28_celltype_score_shares.py --panels all --resume --workers 12 $*"
status=0
"${PY[@]}" "$HERE/run_d28_celltype_score_shares.py" --panels all --resume --workers 12 "$@" || status=$?
echo "[$(date -u +%FT%TZ)] D28 exit status $status"
exit $status
