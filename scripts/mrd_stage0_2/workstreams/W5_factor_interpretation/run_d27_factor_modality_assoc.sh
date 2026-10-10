#!/usr/bin/env bash
# D27 factor x predicted cell type and factor x CITE-seq surface protein (W5.3), all D18 cards
#   -> analysis/{scorecards,figures,cache}/stage2_diagnostic/factor_modality_assoc_v1/
#
# Reads the D18 factor_cards_v1 bundle, the stored Stage 1 scores and the cohort cache (ADT raw
# counts); nothing is refit. The full run (56 representations, about 1,800 cards) takes a few
# minutes with 4 workers x 3 BLAS threads, so it may run in an agent shell; on a busy machine:
#   screen -S W5_d27
#   bash scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d27_factor_modality_assoc.sh --n-jobs 4 --resume
#   (detach with Ctrl-a d; reattach with screen -r W5_d27)
#
# Options are passed through to run_d27_factor_modality_assoc.py:
#   --shortlist full|smoke   --n-jobs N (<=12; BLAS threads per worker = 12 // N)   --resume
#   --merge-only   --out-root DIR   --experiment-dir DIR
# --shortlist smoke (IFN-alpha PCA40/FA40) requires --out-root DIR:
#   bash .../run_d27_factor_modality_assoc.sh --shortlist smoke --out-root <scratch>/d27_smoke --n-jobs 2
#
# Interpreter: $SC_PYTHON if set, else $CONDA_PREFIX/bin/python if CONDA_PREFIX is set, else python.
# Verify: the run ends with "regression checks passed: True" and "done in ...", and
# d27_regression_checks.csv has passed=True in every row.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../../.." && pwd)"

if [[ -n "${SC_PYTHON:-}" ]]; then
  PY="${SC_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
else
  PY="python"
fi

# Shared server etiquette: at most 12 threads in total; the runner caps BLAS threads per worker
# at 12 // n_jobs through threadpoolctl. These defaults cover imports in the parent process.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

cd "${REPO}"
echo "[run_d27_factor_modality_assoc.sh] repo=${REPO} python=${PY}"
exec "${PY}" "${HERE}/run_d27_factor_modality_assoc.py" "$@"
