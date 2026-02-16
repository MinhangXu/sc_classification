# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
pip install -e .
```

No formal test suite; validation is done through notebooks.

## Running Experiments

Main entry point: `scripts/comprehensive_run/run_gene_filter_dr_grid.py`

```bash
# Plan 0: K-sweep + stability screening per DR method
python scripts/comprehensive_run/run_gene_filter_dr_grid.py plan0 \
  --input-h5ad <path>.h5ad --experiments-dir experiments \
  --ks 20,40,60,80 --seeds 1,2,3,4,5 --methods fa,factosig,pca,nmf,cnmf

# Plan 1: preprocess × DR method grid evaluation
python scripts/comprehensive_run/run_gene_filter_dr_grid.py plan1 \
  --input-h5ad <path>.h5ad --experiments-dir experiments \
  --preprocess-set hvg,all_filtered,deg_weak_screen,hybrid \
  --dr-methods pca,fa,nmf,factosig,cnmf
```

Plans 0–1 are implemented; Plans 2–4 have skeleton runners only. Plan specs live in `scripts/comprehensive_run/plans/` (see `INDEX.md` for the mapping).

## Architecture

Uses a `src` layout (`src/sc_classification/`). The pipeline is: **preprocessing → dimensionality reduction → classification**.

### Key modules

- **`pipeline/standardized_pipeline.py`** — `StandardizedPipeline`: unified orchestrator that chains the three stages. This is what runners call.
- **`utils/preprocessing.py`** — Cell/gene filtering, HVG selection, standardization. All data prep starts here.
- **`utils/experiment_manager.py`** — `ExperimentManager`: creates experiment directories, saves configs/models/metrics. Generates experiment IDs as `{timestamp}_{dr_method}_{n_components}_{downsampling}_{config_hash}`.
- **`utils/experiment_analysis.py`** — `ExperimentAnalyzer`: cross-experiment comparison, performance analysis, result export.

### Dimension reduction (`dimension_reduction/`)

All DR methods inherit from `DimensionReductionMethod` in `base.py` and expose `fit_transform`. Implementations: `pca.py`, `factor_analysis.py`, `factor_analysis_R.py`, `nmf.py`, `factosig.py` (wraps the sibling `factosig` package).

### Classification (`classification_methods/`)

All classifiers inherit from `Classifier` in `base.py`. Primary: `lr_lasso.py` (L1-regularized logistic regression).

### Data flow

Data flows as AnnData objects. DR embeddings → `adata.obsm`; loadings → `adata.varm`. Experiment artifacts are saved to `experiments/{experiment_id}/` with subdirs: `preprocessing/`, `models/`, `analysis/`, `logs/`.

## Directory Conventions

- `src/sc_classification/` — Library code only (importable, not runnable directly)
- `scripts/comprehensive_run/` — Active experiment runners
- `scripts/orchestrator/` — Stage orchestration scaffolding
- `scripts/legacy/` — Historical scripts (kept for provenance)
- `notebooks/` — Analysis notebooks, organized by topic subdirectory
- `experiments/` — Generated run artifacts
- `experiments/archive/` — Archived legacy runs
- `.cursor/plans/` — Raw Cursor plan snapshots (provenance only, do not edit)

## Naming Conventions

- Scripts: `run_<study_or_module>.py`
- Notebooks: `<topic>_<purpose>_<YYYYMMDD>.ipynb`
- Experiment dirs: `YYYYMMDD_HHMMSS_<slug>_<hash>`

## Autonomous Runs

Long studies use a staged approach (see `docs/AUTONOMOUS_COMPREHENSIVE_RUNS.md`):
- Stage A (smoke test) → B (pilot) → C (full run), promoting only when validation gates pass.
- Each stage writes `RUN_STATE.json` for interruption/recovery.
- Orchestration scaffold: `scripts/orchestrator/run_goal_orchestrator.py`.

## Key Dependencies

scanpy, anndata, scikit-learn, numpy, pandas, scipy, matplotlib, seaborn, pyyaml.
