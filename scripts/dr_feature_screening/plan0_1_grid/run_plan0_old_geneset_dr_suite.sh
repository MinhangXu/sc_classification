#!/usr/bin/env bash
set -euo pipefail

# Plan 0 DR suite on the old-geneset panel space:
# - restrict genes by GMT union (Hallmark/Reactome/KEGG bundle)
# - keep panel genes with all_filtered (no variance filter), not HVG reselection

INPUT_H5AD="/home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad"
GMT_PATH="/home/minhang/mds_project/sc_classification/scripts/knowledge_driven_embedding/older_geneset/genesets_v1.gmt"
EXPERIMENTS_DIR="/home/minhang/mds_project/sc_classification/experiments"

TIMEPOINT_FILTER="MRD"
TECH_FILTER="CITE"
KS="20,40,60"
SEEDS="1,2,3,4,5"
METHODS="fa,factosig,pca,nmf,cnmf"
CNMF_N_ITER="20"
CNMF_DT="0.5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_gene_filter_dr_grid.py"

usage() {
  cat <<'EOF'
Usage:
  run_plan0_old_geneset_dr_suite.sh [options]

Options:
  --input-h5ad PATH      Input full cohort .h5ad (default: project cohort file).
  --gmt PATH             GMT file used to restrict genes by union.
  --experiments-dir PATH Experiments root directory.
  --timepoint VALUE      Timepoint filter (default: MRD).
  --tech VALUE           Tech filter (default: CITE).
  --ks LIST              Comma/space-separated K list (default: 20,40,60).
  --seeds LIST           Comma/space-separated seeds (default: 1,2,3,4,5).
  --methods LIST         DR methods (default: fa,factosig,pca,nmf,cnmf).
  --cnmf-n-iter INT      cNMF iterations (default: 20).
  --cnmf-dt VALUE        cNMF density threshold (default: 0.5).
  -h, --help             Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-h5ad) INPUT_H5AD="$2"; shift 2 ;;
    --gmt) GMT_PATH="$2"; shift 2 ;;
    --experiments-dir) EXPERIMENTS_DIR="$2"; shift 2 ;;
    --timepoint) TIMEPOINT_FILTER="$2"; shift 2 ;;
    --tech) TECH_FILTER="$2"; shift 2 ;;
    --ks) KS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --cnmf-n-iter) CNMF_N_ITER="$2"; shift 2 ;;
    --cnmf-dt) CNMF_DT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$INPUT_H5AD" ]]; then
  echo "Input .h5ad not found: $INPUT_H5AD" >&2
  exit 1
fi
if [[ ! -f "$GMT_PATH" ]]; then
  echo "GMT file not found: $GMT_PATH" >&2
  exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "Runner script not found: $RUNNER" >&2
  exit 1
fi
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Experiments dir not found: $EXPERIMENTS_DIR" >&2
  exit 1
fi

LOG_ROOT="$EXPERIMENTS_DIR"
mkdir -p "$LOG_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_ROOT/plan0_old_geneset_dr_suite_${TS}.log"

echo "[$(date -Iseconds)] Starting Plan 0 old-geneset DR suite"
echo "INPUT_H5AD=$INPUT_H5AD"
echo "GMT_PATH=$GMT_PATH"
echo "EXPERIMENTS_DIR=$EXPERIMENTS_DIR"
echo "LOG_FILE=$LOG_FILE"

python -u "$RUNNER" plan0 \
  --input-h5ad "$INPUT_H5AD" \
  --experiments-dir "$EXPERIMENTS_DIR" \
  --timepoint-filter "$TIMEPOINT_FILTER" \
  --tech-filter "$TECH_FILTER" \
  --reference-selection-method "all_filtered" \
  --reference-allfiltered-min-frac 0.0 \
  --reference-allfiltered-no-variance-filter \
  --restrict-genes-gmt "$GMT_PATH" \
  --ks "$KS" \
  --seeds "$SEEDS" \
  --methods "$METHODS" \
  --cnmf-n-iter "$CNMF_N_ITER" \
  --cnmf-dt "$CNMF_DT" \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date -Iseconds)] Finished Plan 0 old-geneset DR suite"
