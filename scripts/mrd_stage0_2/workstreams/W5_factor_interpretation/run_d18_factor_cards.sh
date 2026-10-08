#!/usr/bin/env bash
# D18 generalized factor cards (W5.1) -> analysis/{scorecards,figures,cache}/stage2_diagnostic/factor_cards_v1/
#
# Reads stored Stage 1 scores/loadings, the globally selected LOPO coefficients and the cohort
# cache; nothing is refit. Full shortlist (56 representations, 1810 cards) takes about 2 minutes
# with 6 workers on the lab server, so it can run in an agent shell; use screen if the machine
# is busy:
#   screen -S W5_d18
#   bash scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.sh --shortlist full --n-jobs 6 --resume
#   (detach with Ctrl-a d; reattach with screen -r W5_d18)
#
# Options are passed through to run_d18_factor_cards.py:
#   --shortlist full|ifna|smoke   --n-jobs N (<=12)   --resume   --merge-only   --out-root DIR
#   --top-genes N   --adt-top N   --no-adt   --no-matching   --experiment-dir DIR
# --shortlist ifna|smoke requires --out-root DIR (outputs then go under DIR/{scorecards,figures,
# cache}/stage2_diagnostic/factor_cards_v1/): the merge rewrites the bundle for the requested
# shortlist only, so a partial shortlist must never land in the canonical factor_cards_v1 slug.
#   bash .../run_d18_factor_cards.sh --shortlist smoke --out-root <scratch>/d18_smoke --n-jobs 6
#
# Interpreter: $SC_PYTHON if set, else $CONDA_PREFIX/bin/python if CONDA_PREFIX is set, else python.
# Paths resolve through sc_classification.utils.paths (SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID, ...).
# Verify: the run ends with "bundle valid; regression checks passed: True", and
# factor_cards_regression_checks.csv has passed=True in every row.
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

# Shared server etiquette: at most 12 threads in total (6 workers x 2 BLAS threads by default).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

cd "${REPO}"
echo "[run_d18_factor_cards.sh] repo=${REPO} python=${PY} threads/worker=${OMP_NUM_THREADS}"
exec "${PY}" "${HERE}/run_d18_factor_cards.py" "$@"
