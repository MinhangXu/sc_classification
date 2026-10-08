#!/usr/bin/env bash
# D17 Stage 1 basis-variant pilot on IFN-alpha (W2.1), resumable screen wrapper.
# Outputs: analysis/{scorecards,figures,cache}/stage2_diagnostic/basis_variant_pilot_ifna_v1/ under the experiment dir.
#
#   screen -S W2_d17
#   bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.sh [extra args]
#   (detach with Ctrl-a d; reattach with screen -r W2_d17)
#
# Default: PCA and FA at K 10, 20 and 40, every arm, --resume, 12 single-thread joblib workers
# (shared-server cap). On the lab server the full plan takes about 7-8 minutes of wall time;
# K = 40 alone (--ks 40) takes about 5 minutes. Finished bases and Stage 2 units are skipped, so
# a rerun after an interruption only does the missing work. Extra args are passed through, e.g.
#   --ks 40                    only the 40-factor bases
#   --parts anchor,align,report   rebuild tables, figures and findings from finished units
#   --smoke --out-root DIR     2 held-out patients, 1 seed, a thinned C grid, outputs under DIR
# Python: $SC_PYTHON if set, else the lab conda env under $HOME, else python on PATH.
# Paths resolve through sc_classification.utils.paths (SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID).
# Verify: the log shows "[anchor] .../84 stored per-patient lifts reproduced" with all rows passing,
# "[stage2] 756 units" and "[done]"; d17_anchor_check.csv has pass=True in every row and
# basis_variant_pilot_ifna_v1_findings.md is regenerated.
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
ARGS=(--ks 10,20,40 --resume --n-jobs 12 "$@")
echo "[$(date -u +%FT%TZ)] D17 start: ${PY[*]} run_d17_basis_variant_pilot.py ${ARGS[*]}"
status=0
"${PY[@]}" "$HERE/run_d17_basis_variant_pilot.py" "${ARGS[@]}" || status=$?
echo "[$(date -u +%FT%TZ)] D17 exit status $status"
exit $status
