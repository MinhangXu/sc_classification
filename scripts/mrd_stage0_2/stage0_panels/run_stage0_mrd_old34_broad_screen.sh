#!/usr/bin/env bash
set -euo pipefail

# Launch the new MRD Stage 0/1/quick-Stage-2 broad screen from the original
# cohort AnnData. With no arguments, this runs the recommended first-pass screen.
# Any supplied arguments are passed directly to the Python runner.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_stage0_mrd_old34_broad_screen.py"

if [[ ! -f "$RUNNER" ]]; then
  echo "Runner not found: $RUNNER" >&2
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec python -u "$RUNNER" --help
fi

if [[ $# -eq 0 ]]; then
  set -- \
    --input-h5ad /home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad \
    --timepoint MRD \
    --tech CITE \
    --panel-types single_geneset_only,single_group_only,full_control,core_only,hvg_anchor_control \
    --hvg-anchor-sizes 500,1000,3000,10000 \
    --methods pca,fa,factosig,factosig_promax \
    --ks 5,10,20,40 \
    --stage1-scope across_patient \
    --small-panel-policy direct_gene \
    --quick-stage2 \
    --out-root /home/minhang/mds_project/sc_classification/experiments
fi

echo "[$(date -Iseconds)] Starting MRD old-34 broad Stage 0 screen"
echo "Runner: $RUNNER"
python -u "$RUNNER" "$@"
echo "[$(date -Iseconds)] Finished MRD old-34 broad Stage 0 screen"
