#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/minhang/mds_project"
SC_ROOT="${REPO_ROOT}/sc_classification"
DEFAULT_EXPERIMENT_DIR="${SC_ROOT}/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
DEFAULT_BUNDLE_DIR="${SC_ROOT}/scripts/knowledge_driven_embedding/expanded_stage0_genesets"
DEFAULT_BRANCH="expanded_stage0_mrd_manuscript_axes_v1"
DEFAULT_STAGE2_RUN_ID="expanded_stage0_mrd_manuscript_axes_v1_stage2"
CONDA_ENV="sc-classification-rapids-py311"

BUNDLE_DIR="${DEFAULT_BUNDLE_DIR}"
BRANCH_NAME="${DEFAULT_BRANCH}"
EXPERIMENT_DIR="${DEFAULT_EXPERIMENT_DIR}"
GPU_IDS="auto"
GENESETS_GMT=""
MANIFEST_TSV=""
STAGE0_OUTPUT_SUBDIR=""
STAGE2_RUN_ID="${DEFAULT_STAGE2_RUN_ID}"
DRY_RUN=0
SMOKE_TEST=0
RUN_STAGE2=0
SKIP_STAGE2=0
BUILD_BUNDLE=1
BACKEND="gpu"
STRICT_GPU=1
RERUN_STAGE0=0

usage() {
  cat <<'EOF'
Usage:
  run_expanded_stage0_genesets_stage0_to_stage2.sh \
    --bundle-dir <dir> --branch-name <name> --experiment-dir <dir> --gpu-ids <ids> [options]

Required/important options:
  --bundle-dir DIR              Expanded bundle directory (default: knowledge_driven_embedding/expanded_stage0_genesets)
  --branch-name NAME            Namespace for new outputs (default: expanded_stage0_mrd_manuscript_axes_v1)
  --experiment-dir DIR          Existing experiment directory to write namespaced outputs under
  --gpu-ids IDS                 GPU IDs for full Stage 2 sharding, or auto

Inferred/optional:
  --genesets-gmt PATH           Defaults to <bundle-dir>/final_bundle.gmt
  --manifest-tsv PATH           Defaults to <bundle-dir>/final_manifest.tsv
  --stage0-output-subdir NAME   Alias for --branch-name when set
  --stage2-run-id ID            Defaults to <branch-name>_stage2
  --dry-run                     Print commands only
  --smoke-test                  Tiny Stage 0/1 plus minimal Stage 2 path
  --run-stage2                  Run comprehensive Stage 2 (explicit opt-in)
  --skip-stage2                 Run only bundle build + Stage 0/1
  --skip-bundle-build           Do not rebuild/download resources
  --rerun-stage0                Allow replacing existing Stage 0 artifacts in this branch namespace
  --backend auto|gpu|cpu        Stage 2 backend (default: gpu)
  --no-strict-gpu               Do not require cuML when backend is gpu/auto
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --branch-name) BRANCH_NAME="$2"; shift 2 ;;
    --experiment-dir) EXPERIMENT_DIR="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --genesets-gmt) GENESETS_GMT="$2"; shift 2 ;;
    --manifest-tsv) MANIFEST_TSV="$2"; shift 2 ;;
    --stage0-output-subdir) STAGE0_OUTPUT_SUBDIR="$2"; shift 2 ;;
    --stage2-run-id) STAGE2_RUN_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --smoke-test) SMOKE_TEST=1; shift ;;
    --run-stage2) RUN_STAGE2=1; shift ;;
    --skip-stage2) SKIP_STAGE2=1; shift ;;
    --skip-bundle-build) BUILD_BUNDLE=0; shift ;;
    --rerun-stage0) RERUN_STAGE0=1; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --no-strict-gpu) STRICT_GPU=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "${STAGE0_OUTPUT_SUBDIR}" ]]; then
  BRANCH_NAME="${STAGE0_OUTPUT_SUBDIR}"
fi
if [[ -z "${BRANCH_NAME}" ]]; then
  echo "--branch-name must be non-empty to avoid overwriting old outputs." >&2
  exit 2
fi
if [[ "${SKIP_STAGE2}" -eq 1 && "${RUN_STAGE2}" -eq 1 ]]; then
  echo "Choose either --run-stage2 or --skip-stage2, not both." >&2
  exit 2
fi
if [[ "${STAGE2_RUN_ID}" == "${DEFAULT_STAGE2_RUN_ID}" ]]; then
  STAGE2_RUN_ID="${BRANCH_NAME}_stage2"
fi

BUNDLE_DIR="$(realpath "${BUNDLE_DIR}")"
EXPERIMENT_DIR="$(realpath "${EXPERIMENT_DIR}")"
GENESETS_GMT="${GENESETS_GMT:-${BUNDLE_DIR}/final_bundle.gmt}"
MANIFEST_TSV="${MANIFEST_TSV:-${BUNDLE_DIR}/final_manifest.tsv}"

if [[ ! -d "${BUNDLE_DIR}" ]]; then
  echo "Missing bundle directory: ${BUNDLE_DIR}" >&2
  exit 2
fi
if [[ ! -d "${EXPERIMENT_DIR}" ]]; then
  echo "Missing experiment directory: ${EXPERIMENT_DIR}" >&2
  exit 2
fi

LOG_DIR="${EXPERIMENT_DIR}/logs/${BRANCH_NAME}"
RUN_LOG="${LOG_DIR}/expanded_stage0_to_stage2_$(date -u +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
BUILDER="${BUNDLE_DIR}/build_expanded_stage0_bundle.py"
STAGE0_RUNNER="${SC_ROOT}/scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py"
STAGE2_RUNNER="${SC_ROOT}/scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py"

run_cmd() {
  echo "+ $*" | tee -a "${RUN_LOG}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@" 2>&1 | tee -a "${RUN_LOG}"
  fi
}

if [[ "${BUILD_BUNDLE}" -eq 1 ]]; then
  run_cmd "${PYTHON_CMD[@]}" "${BUILDER}" \
    --out-dir "${BUNDLE_DIR}" \
    --cache-dir "${REPO_ROOT}/data/resource_cache/stage0_expanded_genesets"
fi

if [[ ! -f "${GENESETS_GMT}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry-run note: GMT bundle does not exist yet and would be created by the builder: ${GENESETS_GMT}" | tee -a "${RUN_LOG}"
  else
    echo "Missing GMT bundle: ${GENESETS_GMT}" >&2
    exit 2
  fi
fi
if [[ ! -f "${MANIFEST_TSV}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry-run note: manifest does not exist yet and would be created by the builder: ${MANIFEST_TSV}" | tee -a "${RUN_LOG}"
  else
    echo "Missing manifest TSV: ${MANIFEST_TSV}" >&2
    exit 2
  fi
fi

STAGE0_ARGS=(
  "${PYTHON_CMD[@]}" "${STAGE0_RUNNER}"
  --experiment-dir "${EXPERIMENT_DIR}"
  --branch-name "${BRANCH_NAME}"
  --gmt-path "${GENESETS_GMT}"
  --old-manifest-path "${MANIFEST_TSV}"
  --panel-types "atomic_sets,family_union_sets,core_anchor_sets,single_group_only,leave_one_family_out,full_control,core_only,hvg_anchor_control,all_filtered_control"
  --methods "pca,fa,factosig,factosig_promax"
  --ks "5,10,20,40"
  --stage1-scope "across_patient"
  --small-panel-policy "direct_gene"
  --seed 42
  --quick-stage2
)

if [[ "${SMOKE_TEST}" -eq 1 ]]; then
  STAGE0_ARGS+=(
    --panel-types "atomic_sets,family_union_sets,core_anchor_sets,hvg_anchor_control"
    --methods "pca"
    --ks "5"
    --max-panels 4
  )
fi
if [[ "${RERUN_STAGE0}" -eq 1 ]]; then
  STAGE0_ARGS+=(--rerun)
fi

run_cmd "${STAGE0_ARGS[@]}"

STAGE0_SCORECARD="${EXPERIMENT_DIR}/analysis/scorecards/${BRANCH_NAME}/stage0_mrd_old34_broad_scorecard.csv"

if [[ "${SMOKE_TEST}" -eq 1 ]]; then
  STAGE2_ARGS=(
    "${PYTHON_CMD[@]}" "${STAGE2_RUNNER}"
    --experiment-dir "${EXPERIMENT_DIR}"
    --stage0-scorecard "${STAGE0_SCORECARD}"
    --stage2-output-branch "${BRANCH_NAME}"
    --stage2-run-id "${STAGE2_RUN_ID}_smoke"
    --make-shortlist-from-quick-l2
    --run-discovery-full-cohort-fit
    --panel-selection "all_quick_rows"
    --max-selected-representations 1
    --penalties "l2"
    --c-grid-log10-min 0
    --c-grid-log10-max 0
    --c-grid-n 1
    --backend "auto"
    --seed 42
  )
  run_cmd "${STAGE2_ARGS[@]}"
elif [[ "${RUN_STAGE2}" -eq 1 && "${SKIP_STAGE2}" -eq 0 ]]; then
  STAGE2_ARGS=(
    "${PYTHON_CMD[@]}" "${STAGE2_RUNNER}"
    --experiment-dir "${EXPERIMENT_DIR}"
    --stage0-scorecard "${STAGE0_SCORECARD}"
    --stage2-output-branch "${BRANCH_NAME}"
    --stage2-run-id "${STAGE2_RUN_ID}"
    --canonicalize-existing-quick-stage2
    --make-shortlist-from-quick-l2
    --run-discovery-full-cohort-fit
    --run-sharedness-lopo
    --run-patient-specific
    --panel-selection "shortlist_plus_controls"
    --penalties "l1,l2,elasticnet"
    --class-weight "balanced"
    --backend "${BACKEND}"
    --launch-gpu-shards
    --gpu-ids "${GPU_IDS}"
    --seed 42
  )
  if [[ "${STRICT_GPU}" -eq 1 ]]; then
    STAGE2_ARGS+=(--strict-gpu)
  fi
  run_cmd "${STAGE2_ARGS[@]}"
else
  echo "Skipping comprehensive Stage 2. Pass --run-stage2 to launch it, or --smoke-test for the tiny Stage 2 path." | tee -a "${RUN_LOG}"
fi

echo "Run log: ${RUN_LOG}" | tee -a "${RUN_LOG}"
