# Comprehensive Run Workflow Map

This folder keeps runnable experiment code next to the study plan so the current workflow can be rehydrated from code, plans, and experiment IDs.

## Current Entry Points

- `run_gene_filter_dr_grid.py`
  - Plan 0: K sweep, multi-seed loading stability, optional cNMF K selection.
  - Plan 1: preprocess-method x DR-method grid.
- `run_plan0_old_geneset_dr_suite.sh`
  - April 2026 old 34-program Plan 0 DR suite on the GMT-restricted gene space.
- `run_old_geneset_pruning_metrics.py`
  - Top-down strict-ablation pruning metrics and the new Stage 0 bottom-up/HVG-control panel metrics.
- `run_stage0_old_geneset_bottom_up.sh`
  - Reproducible Stage 0 bottom-up screen for single genesets, single biology groups, HVG controls, and core/full controls.
- `run_plan1c_supervised_latent_benchmark.py`
  - Stage 2 supervised benchmark: pooled and per-patient repeated CV with L1/L2/elastic-net logistic paths.
  - Can now read a Stage 0 score manifest through `--feature-manifest-csv`.

## Important Experiment IDs

- Old-geneset Plan 0 / pruning reference:
  - `/home/minhang/mds_project/sc_classification/experiments/20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0`
- HVG Plan 0 / Plan 1.C reference:
  - `/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886`

## Legacy And Recovery Files

- `legacy/` contains one-off watcher and pilot scripts that should not be used as current entry points.
- The cNMF and standard-DR resume scripts remain at top level for now because the active Plan 0 incident notes still reference them:
  - `resume_plan0_cnmf.py`
  - `resume_plan0_standard_dr.py`
  - `run_plan0_resume_standard_dr_varimax.sh`
  - `run_plan0_resume_standard_dr_promax.sh`
- Helper scripts that may become reusable library code:
  - `attach_plan0_dr_cache_to_preprocessed_adata.py`
  - `reorganize_plan0_cnmf_curated.py`
  - `build_plan0_k_selection_summary.py`
- Plan 2-4 skeletons are not current entry points unless explicitly revived.

## Where the “on-paper” plans are

For day-to-day use, use the curated plan docs + index:

- `plans/INDEX.md`
- `plans/active_plan0_plan1.md`
- `plans/stage0_geneset_value_added_workflow.md`
- `plans/comprehensive_run_reorganization_plan.md`
- `plans/later_plans2_4.md`

Raw Cursor snapshots and older drafts live in Cursor's internal plans area (provenance only):

- `.cursor/plans/gene-filtering-eval-plan-iter2_e60076f2.plan.md`
- `.cursor/plans/gene-filtering-eval-plan-iter3-cnmf_a09f862f.plan.md`
- `.cursor/plans/gene-filtering-eval-plans_ed94c3ef.plan.md` (contains the unrefined plan 2–4 ideas)

The intent (iter3) is: **do Plan 0 to pick K per DR method (including cNMF)**, then run the main **2-axis grid** (**DR method × preprocess method**) as Plan 1.A (no-CV) followed by Plan 1.B (classifier-only CV).

## How to run

### Plan 0 (K sweep / stability screen)

Example:

```bash
python sc_classification/scripts/comprehensive_run/run_gene_filter_dr_grid.py plan0 \
  --input-h5ad path/to/input.h5ad \
  --experiments-dir experiments \
  --timepoint-filter MRD \
  --tech-filter CITE \
  --reference-hvg 10000 \
  --ks 20,40,60,80 \
  --seeds 1,2,3,4,5 \
  --methods fa,factosig,pca,nmf,cnmf
```

Key outputs (under the created experiment directory):

- `analysis/plan0/k_selection_summary.csv`: quick table to plot **stability vs variance-proxy** (and consensus silhouette for FA/FactoSig)
- `analysis/plan0/stability/<method>/k_<K>/...`: per-K replicate caches and stability summaries
- `models/cnmf_plan0/` + `analysis/plan0/cnmf/`: cNMF artifacts + consensus stats
  - optional post-processing: `models/cnmf_plan0/curated/` (symlink/copy view with `global/`, `k_<K>/inputs/`, `k_<K>/consensus/`, `MANIFEST.csv`)

Notes:
- **FA rotation**: the current runner uses sklearn FA with no explicit rotation parameter. An in-progress engineering plan adds `--fa-rotation none|varimax|promax` for Plan 0 (and optionally Plan 1). See `plans/plan0rotationseedsplan1stability.md`.
- **Stability/consensusness requires multi-seed**: for FA/FactoSig, consensus clustering caches only run when you provide **2+ seeds**.

### Plan 1 (grid run + multi-seed consensusness)

Example:

```bash
python sc_classification/scripts/comprehensive_run/run_gene_filter_dr_grid.py plan1 \
  --input-h5ad path/to/input.h5ad \
  --experiments-dir experiments \
  --timepoint-filter MRD \
  --tech-filter CITE \
  --preprocess-set hvg,all_filtered,deg_weak_screen,hybrid \
  --hvg-n 3000 \
  --dr-methods pca,fa,nmf,factosig,cnmf \
  --k-by-method pca=60,fa=60,nmf=60,factosig=60,cnmf=60 \
  --seeds 1,2,3,4,5 \
  --cv-folds 0
```

Key outputs:

- `analysis/preprocess_cache/<tag>/adata_with_dr.h5ad`
  - Contains **one embedding per DR method** (from the first seed in `--seeds`) to keep file size reasonable.
- `analysis/plan1_stability/<tag>/<method>/k_<K>/...`
  - Multi-seed replicate caches + stability summary and (FA/FactoSig) consensus clustering cache.
- `analysis/classification_grid/<method>/...`
  - L1-logistic-regression summaries (currently based on the embedding attached to the `.h5ad`).

Note:
- As of now, Plan 1 seeding and `analysis/plan1_stability/...` caches are the intended design but are not yet fully wired in code. See `plans/plan0rotationseedsplan1stability.md`.

### Stage 0 bottom-up old-geneset screen

Example:

```bash
bash sc_classification/scripts/comprehensive_run/run_stage0_old_geneset_bottom_up.sh \
  --methods fa,factosig,pca \
  --ks 5,10,20,40
```

Key outputs:

- `analysis/stage0_old_geneset_bottom_up/panel_manifest.csv`
- `analysis/stage0_old_geneset_bottom_up/panel_dr_metric_rows.csv`
- `analysis/stage0_old_geneset_bottom_up/panel_dr_leaderboard.csv`
- `analysis/stage0_old_geneset_bottom_up/dr_cache_strict_ablation/<panel>/<method>/k_<K>/seed_<seed>/scores.npy`

### Stage 2 on Stage 0 score artifacts

Example:

```bash
python sc_classification/scripts/comprehensive_run/run_plan1c_supervised_latent_benchmark.py \
  --experiment-dir /home/minhang/mds_project/sc_classification/experiments/20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0 \
  --feature-manifest-csv /home/minhang/mds_project/sc_classification/experiments/20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0/analysis/stage0_old_geneset_bottom_up/panel_dr_metric_rows.csv \
  --k 20 \
  --methods all \
  --modes pooled,per_patient \
  --penalties l1,l2,elasticnet \
  --output-subdir analysis/stage0_old_geneset_bottom_up/plan1c_supervised_from_scores_k20
```

## Notes / known gaps vs iter3 plan (and “later” plans)

- The iter3 plan deliberately treats FA/FactoSig “consensusness” as **diagnostic in Plan 0**. This runner also caches multi-seed stability under Plan 1 so you can inspect grid runs without re-running DR, but classification still uses only the first-seed embedding attached to the `.h5ad`.
- The unrefined “later” plans (2–4 in `.cursor/plans/gene-filtering-eval-plans_ed94c3ef.plan.md`) include stricter protocol ideas like **train-only gene selection** and **heldout splits within patient×timepoint_type**. Those are not enforced here yet (current runs operate on the full filtered dataset per preprocess method; CV, when enabled, is classifier-only).

