#!/usr/bin/env bash
# D15b patient-count learning curve, FULL design (W6.1; slug patient_count_learning_curve_v1).
#
# CPU only, 12 worker processes with 1 BLAS thread each. Run it in screen (AGENTS R1):
#   screen -S W6_d15b
#   bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh
#   (detach with Ctrl-a d; reattach with screen -r W6_d15b)
#
# Always resumable: finished tune units (representation, subset, penalty) and outer units
# (representation, held-out patient, subset) are skipped, and the reduced-design units already in
# the cache are reused. Rerunning the same command after an interruption continues where it stopped.
# Extra arguments are passed to the runner (for example --representations ifna_fa40).
#
# Environment overrides: SC_CONDA_PREFIX (conda env prefix), PYTHON, N_JOBS (<= 12), plus the
# SC_* path variables of sc_classification.utils.paths (SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID).
#
# Outputs (EXP-relative): analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/d15b_full_*.csv,
# analysis/figures/stage2_diagnostic/patient_count_learning_curve_v1/D15b[A-D]_full_*.png|pdf, the findings md,
# manifest and artifact ledger, and the unit cache plus logs/ under analysis/cache/stage2_diagnostic/patient_count_learning_curve_v1/.
# Verify: d15b_full_unit_status.json reports n_units_missing 0 (1764 units = 4 representations x 441),
# and d15b_full_reference_checks.csv has pass = True on every row.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
ENV_PREFIX="${SC_CONDA_PREFIX:-$HOME/miniconda3/envs/sc-classification-rapids-py311}"
PYTHON="${PYTHON:-$ENV_PREFIX/bin/python}"
N_JOBS="${N_JOBS:-12}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EXP="$(env CONDA_PREFIX="$ENV_PREFIX" "$PYTHON" -c "import sys; sys.path.insert(0, sys.argv[1]); from sc_classification.utils.paths import experiment_dir; print(experiment_dir())" "$REPO/src")"
LOG_DIR="$EXP/analysis/cache/stage2_diagnostic/patient_count_learning_curve_v1/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/d15b_full_$(date -u +%Y%m%dT%H%M%SZ).log"

cd "$REPO"
echo "[d15b.sh] repo=$REPO exp=$(basename "$EXP") n_jobs=$N_JOBS log=$LOG" | tee -a "$LOG"
set +e
env CONDA_PREFIX="$ENV_PREFIX" "$PYTHON" scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py \
    --design full --resume --n-jobs "$N_JOBS" "$@" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "[d15b.sh] exit $status" | tee -a "$LOG"
exit "$status"
