# Engineering plan: FA rotation + multi-seed stability + Plan 1 seeding/stability sync

Provenance: copied from Cursor plan `~/.cursor/plans/plan0rotationseedsplan1stability_2d1f91c1.plan.md` so it lives next to the curated comprehensive-run plan docs.

---

## Goals

- Make **FA rotation** an explicit, testable dimension in Plan 0 (and optionally Plan 1), supporting **varimax** and **promax**.
- Enable **multi-seed stability/consensusness** as planned, and clarify where it lives:
  - **Plan 0**: K-sweep + stability/consensusness diagnostics (already implemented, but only activates when `--seeds` has ≥2).
  - **Plan 1**: add back the missing `--seeds` behavior and write the documented `analysis/plan1_stability/...` caches (currently not implemented).
- Ensure **PCA** is consistently supported/used (it is already supported, but needs to be run in Plan 0 experiments).
- For **cNMF**, keep using the native output trees under `models/`, but add a **manifest** that points to the replicate NMF results that feed consensus.
- Update the **plan docs** under `scripts/comprehensive_run/` to match actual runner behavior.

## Key facts from current code

- **Plan 0 FA rotation today**: none. `run_gene_filter_dr_grid.py` imports `FAWrapper` from sklearn-based `src/sc_classification/dimension_reduction/factor_analysis.py` which does not expose rotation.
- An R-based rotated FA implementation exists at `src/sc_classification/dimension_reduction/factor_analysis_R.py` with `rotate="varimax"` default, but it is not used by the comprehensive runner.
- **Plan 0 consensusness**: implemented in `scripts/comprehensive_run/run_gene_filter_dr_grid.py` as `consensus_cluster_components(...)` and only runs for FA/FactoSig when multiple seeds are present (`len(loadings_runs) >= 2`).
- **Plan 1 seeding/stability**: docs mention `--seeds` and `analysis/plan1_stability/...`, but the code currently:
  - has **no `--seeds` argument**
  - runs standard DR with a fixed `seed=42`
  - does **not** write `analysis/plan1_stability/...`

## Proposed design

### 1) Add FA rotation to the sklearn FA path (Python-only)

- Extend `src/sc_classification/dimension_reduction/factor_analysis.py` to accept:
  - `rotation: str` in `{none,varimax,promax}` (default `none` to preserve current behavior)
  - optionally `rotation_kwargs` for promax power, etc.
- Implement rotation post-fit using `factor_analyzer`:
  - rotate loadings matrix \(L\) using the chosen rotation
  - rotate scores \(Z\) consistently so downstream classification uses rotated factors:
    - if rotation matrix is \(R\) and we set \(L' = L R\), then set scores \(Z' = Z (R^{-T})\)
    - for orthogonal \(R\) this reduces to \(Z' = Z R\)
  - store rotation metadata (and the matrix if available) in `adata.uns['fa']`
- If `factor_analyzer` is missing, fail fast with a clear error and/or write a JSON artifact (mirroring the existing `cnmf_missing.json` pattern).

### 2) Expose FA rotation in Plan 0 CLI/config

- Update `scripts/comprehensive_run/run_gene_filter_dr_grid.py` Plan 0 CLI:
  - add `--fa-rotation none|varimax|promax` (default `none`)
- Thread this parameter into `_run_dr_method('fa', ...)` → `FAWrapper.fit_transform(... rotation=...)`.
- Ensure `analysis/plan0/config.json` records the FA rotation choice.

### 3) Multi-seed stability/consensusness: make it first-class in outputs

- Plan 0 already writes per-(method,k) stability summaries at `analysis/plan0/stability/<method>/k_<K>/...`.
- Improve `analysis/plan0/k_selection_summary.csv` generation so it fills:
  - `n_seeds`
  - stability aggregates derived from `pairwise_stability_summary.json`
  - consensus silhouette if `consensus_cache/consensus_metrics.json` exists
  - variance proxy already present via `extras.json` (FA uses `sum_factor_score_variances`)
- Clarify in docs that **stability/consensusness requires `--seeds` ≥ 2**.

### 4) Fix Plan 1 to match docs: add seeds + stability caches

- Update `scripts/comprehensive_run/run_gene_filter_dr_grid.py` Plan 1 CLI:
  - add `--seeds 1,2,3,...`
- For each preprocess tag and method (except cnmf handled separately):
  - run DR for each seed
  - write replicate caches under:
    - `analysis/plan1_stability/<tag>/<method>/k_<K>/replicates/seed_<seed>/{scores.npy,loadings.npy,extras.json}`
  - write stability summaries analogous to Plan 0:
    - `pairwise_stability_summary.json`
    - `consensus_cache/*` for FA/FactoSig when seeds≥2
- Keep the existing behavior for the primary artifact:
  - `analysis/preprocess_cache/<tag>/adata_with_dr.h5ad` contains **one embedding per DR method** (e.g. from the first seed) to avoid bloat.

### 5) PCA in Plan 0 and Plan 1

- No new method implementation needed (PCA is already supported via `PCAWrapper`).
- Ensure docs + examples include PCA and recommend re-running Plan 0 with PCA included if an experiment crashed before reaching PCA.

### 6) cNMF: keep native tree, add manifest for replicate NMF runs

- Plan 0: after factorization, write a manifest under `analysis/plan0/cnmf/` per K, e.g.:
  - `analysis/plan0/cnmf/k_<K>/replicates_manifest.json`
  - Contents: dt, k, n_iter, and a list of file paths inside `models/cnmf_plan0/...` that correspond to individual NMF replicates.
- Plan 1: similarly write a manifest under `analysis/preprocess_cache/<tag>/cnmf_manifest.json` (or under `analysis/plan1_stability/<tag>/cnmf/...` if treating cnmf as part of stability).
- Important: cnmf “replicates” (its internal `n_iter`) are distinct from our outer “seeds”; the manifest should make that explicit.

### 7) Update plan docs under `scripts/comprehensive_run/`

- Update `scripts/comprehensive_run/README.md`:
  - Plan 0: document `--fa-rotation` and that stability/consensusness requires multiple seeds.
  - Plan 1: correct/add `--seeds` and document `analysis/plan1_stability/...` as actually produced once implemented.
- Update `scripts/comprehensive_run/plans/active_plan0_plan1.md`:
  - state where consensusness is computed (Plan 0 for K selection; Plan 1 for grid-run diagnostics once implemented)
  - add FA rotation as an explicit Plan 0 diagnostic dimension

## Non-goals (current iteration)

- Do not add scVI/totalVI/MultiVI DR methods in this iteration.

## Risks / checks

- Rotation correctness: ensure score transformation matches the rotated loading convention (store rotation matrix and verify basic invariances).
- Dependency management: `factor_analyzer` must be available; if not, produce a clear error artifact/instructions.
- Backward compatibility: default rotation remains `none`; existing experiments remain readable.

