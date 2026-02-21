# Comprehensive runners: gene filtering × DR grids

This folder is meant to keep **the runnable code next to the study plan** (good for reproducibility + publication handoff).

## What exists today

- `run_gene_filter_dr_grid.py`
  - **plan0**: K-sweep + multi-seed loading stability + optional cNMF K-selection
  - **plan1**: preprocess-method × DR-method grid, plus **multi-seed stability/consensusness caches**
- Helpers:
  - `resume_plan0_cnmf.py`: finish cNMF for an existing (crashed) Plan 0 experiment dir
  - `resume_plan0_standard_dr.py`: append missing PCA/FA/FactoSig replicate caches (multi-seed) into an existing Plan 0 experiment dir (no re-preprocessing)
  - `attach_plan0_dr_cache_to_preprocessed_adata.py`: rehydrate Plan 0 DR caches into `preprocessing/adata_processed*.h5ad`
- Plan 2–4 runner skeletons:
  - `run_gene_filter_dr_plan2_negative_controls.py`
  - `run_gene_filter_dr_plan3_representation_first.py`
  - `run_gene_filter_dr_plan4_two_stage_selection.py`

## Where the “on-paper” plans are

For day-to-day use, use the curated plan docs + index:

- `plans/INDEX.md`
- `plans/active_plan0_plan1.md`
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

## Notes / known gaps vs iter3 plan (and “later” plans)

- The iter3 plan deliberately treats FA/FactoSig “consensusness” as **diagnostic in Plan 0**. This runner also caches multi-seed stability under Plan 1 so you can inspect grid runs without re-running DR, but classification still uses only the first-seed embedding attached to the `.h5ad`.
- The unrefined “later” plans (2–4 in `.cursor/plans/gene-filtering-eval-plans_ed94c3ef.plan.md`) include stricter protocol ideas like **train-only gene selection** and **heldout splits within patient×timepoint_type**. Those are not enforced here yet (current runs operate on the full filtered dataset per preprocess method; CV, when enabled, is classifier-only).

