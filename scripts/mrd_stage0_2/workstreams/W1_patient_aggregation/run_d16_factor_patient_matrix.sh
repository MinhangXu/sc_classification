#!/usr/bin/env bash
# D16 factor x patient allocation matrix (W1.1), resumable screen wrapper.
#
#   screen -S W1_d16
#   bash scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.sh [extra args]
#   (detach with Ctrl-a d; reattach with screen -r W1_d16)
#
# Default: all 41 biological panels x {pca, fa} x K {5,10,20,40}, --resume, 12 single-thread
# workers (shared-server cap). Extra args are passed through, e.g. --merge-only or --workers 4.
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
echo "[$(date -u +%FT%TZ)] D16 start: ${PY[*]} run_d16_factor_patient_matrix.py --panels all --resume --workers 12 $*"
status=0
"${PY[@]}" "$HERE/run_d16_factor_patient_matrix.py" --panels all --resume --workers 12 "$@" || status=$?
echo "[$(date -u +%FT%TZ)] D16 exit status $status"
exit $status
