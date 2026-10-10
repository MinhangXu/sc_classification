#!/usr/bin/env bash
# D26 leave-high-prevalence-out leverage test on allograft rejection and apoptosis (W2.4), resumable screen wrapper.
# Outputs: analysis/{scorecards,figures,cache}/stage2_diagnostic/leverage_allograft_apoptosis_v1/ under the experiment dir.
#
#   screen -S W2_d26
#   bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.sh [extra args]
#   (detach with Ctrl-a d; reattach with screen -r W2_d26)
#
# Default: both panels, PCA and FA at K 10, 20 and 40, every arm (2,364 bases: per panel x method x K one
# transductive refit, normal-only + 3 random drops, 8 leave-out sets each with 3 size-matched and 20
# composition-matched random drops), --resume, 12 single-thread joblib workers (shared-server cap).
# Smoke-measured cost: about 6,000 CPU-seconds, i.e. about 8.5-10 min of wall time with 12 workers; FA K = 40 is
# the slowest unit (about 6.5 s each). Finished units (saved basis + alignment JSON) are skipped, so a rerun
# after an interruption only does the missing work. Extra args are passed through, e.g.
#   --panels apoptosis           one panel (any Stage 0 panel id or hallmark suffix works)
#   --ks 40                      only the 40-factor bases
#   --merge-only                 rebuild tables, figures and findings from finished units
#   --smoke --out-root DIR       P02+P03+P04 and P09 only, 1 + 2 control seeds, outputs under DIR
# Python: $SC_PYTHON if set, else the lab conda env under $HOME, else python on PATH.
# Paths resolve through sc_classification.utils.paths (SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID).
# Verify: the log ends with "[report] refit sanity: 36/36 rows pass", "[report] tables, 8 figure files, findings,
# manifest and ledger written" and "[done] ... (status 0)"; d26_refit_check.csv has pass=True in every row,
# d26_leverage_contrast.csv has n_label_matched_controls = 20 on all 288 leave_out rows, there is no merge_failed.json,
# and leverage_allograft_apoptosis_v1_findings.md exists. Exit status 3 = units missing (rerun the same command).
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
echo "[$(date -u +%FT%TZ)] D26 start: ${PY[*]} run_d26_leverage_panels.py ${ARGS[*]}"
status=0
"${PY[@]}" "$HERE/run_d26_leverage_panels.py" "${ARGS[@]}" || status=$?
echo "[$(date -u +%FT%TZ)] D26 exit status $status"
exit $status
