# dr_feature_screening — DR method / K screening (older lineage)

First-generation dimensionality-reduction and supervised-probe screening on the earlier
experiments (Feb `20260211` HVG, April `20260401` all_filtered). This lineage informed the
K=40 and DR-method choices that the current `scripts/mrd_stage0_2/` study inherited; it is kept
for provenance and reuse, not as the active analysis.

**Plans:** `plans/` (see `plans/active_plan0_plan1.md` and `plans/later_plans2_4.md`).

## Layout

- `plan0_1_grid/`
  - `run_gene_filter_dr_grid.py` — Plan 0 (K sweep + stability per DR method, incl. cNMF) and
    Plan 1 (preprocess-method × DR-method grid).
  - `run_plan0_old_geneset_dr_suite.sh` — April old-geneset Plan 0 DR suite on the GMT-restricted space.
- `plan1c_supervised/`
  - `run_plan1c_supervised_latent_benchmark.py` — Stage 2 supervised benchmark (pooled + per-patient
    repeated CV; L1/L2/elastic-net logistic paths).
  - `run_plan1c_supervised_latent_k40_full_all.sh`, `run_plan1c_full_k40_all.sh` — full K=40 wrappers.
- `skeletons/` — Plan 2-4 exploratory runners (never run): negative controls, representation-first, two-stage selection.
- `legacy/` — one-off recovery/pilot scripts kept for provenance (cNMF/standard-DR resume, cNMF curation,
  k-selection summary, dr-cache attach, pilot/watcher).
- `notebooks/` — analysis notebooks by task: `plan0/`, `plan1c/`, `plan1d/`.

## Important experiment IDs

- Old-geneset Plan 0 reference: `experiments/20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0`
- HVG Plan 0 / Plan 1.C reference: `experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886`

## Run examples

```bash
# Plan 0: K sweep / stability screen
python sc_classification/scripts/dr_feature_screening/plan0_1_grid/run_gene_filter_dr_grid.py plan0 \
  --input-h5ad <path>.h5ad --experiments-dir experiments \
  --ks 20,40,60,80 --seeds 1,2,3,4,5 --methods fa,factosig,pca,nmf,cnmf

# Plan 1: preprocess × DR grid
python sc_classification/scripts/dr_feature_screening/plan0_1_grid/run_gene_filter_dr_grid.py plan1 \
  --input-h5ad <path>.h5ad --experiments-dir experiments \
  --preprocess-set hvg,all_filtered,deg_weak_screen,hybrid --dr-methods pca,fa,nmf,factosig,cnmf

# Plan 1.C: fixed-K supervised latent benchmark
python sc_classification/scripts/dr_feature_screening/plan1c_supervised/run_plan1c_supervised_latent_benchmark.py \
  --experiment-dir <EXP_DIR> --k 40 --methods all --modes pooled,per_patient --penalties l1,l2,elasticnet
```

## Note on the retired bottom-up/pruning work

The old-geneset **bottom-up / pruning** runner and notebooks that lived here were dropped when the
current study's expanded manuscript-axes panel set (`scripts/mrd_stage0_2/`, Set B) superseded them
as the more comprehensive, in-pipeline way to measure each gene space's value added. The design
intent lives on in `scripts/mrd_stage0_2/plans/stage0_geneset_value_added_workflow.md`.
