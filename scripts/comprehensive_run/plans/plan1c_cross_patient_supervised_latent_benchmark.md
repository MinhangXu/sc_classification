# Plan 1.C — supervised latent benchmark at fixed K=40

This plan adds a supervised evaluation layer on top of the completed Plan 0 DR artifacts.

Scope is intentionally narrow and execution-focused:
- fixed experiment provenance
- fixed latent dimensionality (`K=40`)
- fixed DR method set
- explicit CV protocols
- explicit downsampling variants

## Goal

Benchmark how well latent factors support malignant (`CN.label == cancer`) vs healthy (`normal`) classification:
- in a **pan-patient pooled-cell setting** with repeated stratified CV
- in a **per-patient setting** with repeated stratified CV per patient

while comparing:
- DR method / rotation family
- regularization family (L1, L2, Elastic Net)
- donor downsampling policy (none vs random stratified)

## Inputs (frozen for this run)

- Plan 0 experiment dir:
  - `/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886`
- Cohort filter/provenance inherited from that run:
  - `timepoint_filter=MRD`
  - `tech_filter=CITE`
  - `gene selection = HVG 10k`
- Base AnnData for classification-side preprocessing/metadata:
  - `<EXP_DIR>/preprocessing/adata_processed.h5ad`
  - This is the canonical preprocessed matrix before DR method-specific score attachment.
- Label:
  - `target_column=CN.label`
  - positive class `cancer`, negative class `normal`

## DR methods and features (K=40 only)

Run all of:
- `pca`
- `fa`
- `factosig` (varimax)
- `factosig_promax`
- `cnmf`

Feature source conventions:
- `fa`, `factosig`: from `preprocessing/adata_processed_with_plan0_dr_k40_seed1.h5ad` when available
- `pca`, `factosig_promax`: from Plan0 stability seed-1 replicate caches
  - `analysis/plan0/stability/<method>/k_40/replicates/seed_1/{scores.npy,loadings.npy}`
- `cnmf`: from curated consensus usages
  - `models/cnmf_plan0/curated/k_40/consensus/*usages.k_40.dt_0_5.consensus.df.npz`
- All feature matrices must align to identical `obs_names` ordering before modeling.
- Operationally, modeling metadata/labels/patient/source columns come from
  `preprocessing/adata_processed.h5ad`; DR features are joined by `obs_names`.

## Downsampling variants (both required)

Evaluate both:
- `downsample=none`
- `downsample=random` (stratified donor downsampling)

Random downsampling config (carry-over from prior FA runs):
- `donor_recipient_column=source`
- `cell_type_column=predicted.annotation`
- `target_donor_fraction=0.7`
- `target_donor_recipient_ratio_threshold=10.0`
- `min_cells_per_type=20`
- deterministic seed (`random_state=42`)

Downsampling is applied to the dataset view being scored (pooled or patient subset), and must be logged.

### Adaptive aggressive branch for very low malignant counts

For per-patient runs with very low malignant support, add an explicit aggressive branch:
- define `low_malignant_threshold = 10`
- if a patient has `2 <= n_malignant < low_malignant_threshold`, switch to:
  - keep `target_donor_recipient_ratio_threshold = 10.0` for first pass (legacy-consistent baseline)
  - reduce `min_cells_per_type = 5` (allow stronger reduction while preserving multi-cell-type signal)
  - keep stratification by `predicted.annotation`
- if first pass still leaves severe imbalance (post-downsampling donor/recipient ratio `> 20`), run a second pass with:
  - `target_donor_recipient_ratio_threshold = 5.0`
  - `min_cells_per_type = 5`
- if after aggressive downsampling class support is still insufficient for CV, skip with reason.

This preserves the prior philosophy (donor-focused, cell-type-stratified downsampling), but makes low-malignant handling explicit and reproducible.

## Classifier families and hyperparameter grids

All runs use sklearn `LogisticRegression` with `solver='saga'`, `class_weight='balanced'`, fixed seed.

Families:
- L1 (lasso path): `penalty='l1'`
- L2 (ridge path): `penalty='l2'`
- Elastic Net: `penalty='elasticnet'`

Regularization grid:
- `alpha_grid = logspace(-4, 5, 20)` (shared across families)
- map to `C = 1 / alpha`

Elastic Net mixing grid:
- `l1_ratio in [0.1, 0.5, 0.9]`

## CV protocols (both required)

### A) Pooled-cell CV (pan-patient)

Construct one pooled dataset across patients, run repeated stratified CV:
- splitter: `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`
- stratification target: `CN.label` binary
- run for each `(dr_method, downsampling_variant, penalty[, l1_ratio], alpha)`

Required outputs from pooled-cell CV:
- overall CV metrics (mean/std across folds)
- out-of-fold predictions/probabilities per cell
- **per-patient performance computed from pooled OOF predictions**, including:
  - ROC AUC
  - PR AUC
  - overall accuracy
  - malignant recall / precision
  - healthy recall / precision
  - support counts

### B) Per-patient CV

For each patient independently:
- build patient-specific dataset using same DR features
- apply selected downsampling variant
- run repeated stratified CV with the same defaults (`5x10`) where feasible
- if minority count < 5, reduce `n_splits` to minority count
- if resulting `n_splits < 2` or only one class remains, mark as skipped with reason

Explicit skip policy:
- skip any patient with `n_malignant <= 1` (before modeling)
- also skip if, after optional downsampling, CV feasibility checks fail.

Required outputs from per-patient CV:
- patient-level CV metric table across hyperparameters
- best hyperparameter summary per patient
- coefficient vectors from refit on full patient dataset at selected hyperparameter
- skip log and class/support diagnostics

## Confounding and diagnostics requirements

Given strong known coupling between `source` and `CN.label`, every run must include:
- source composition table pre/post downsampling
- label-by-source contingency table pre/post downsampling
- per-patient class counts pre/post downsampling
- warning flags when label or source imbalance is extreme

These diagnostics are mandatory artifacts (not just console prints).

## Output layout (proposed)

Under:
- `<EXP_DIR>/analysis/plan1c_supervised_latent_k40/`

Proposed structure:
- `config.json` (full run config + provenance)
- `input_diagnostics/` (class/source/patient summaries)
- `pooled_cv/<dr_method>/<downsampling>/<penalty>/...`
- `pooled_cv_per_patient/<dr_method>/<downsampling>/<penalty>/...`
- `per_patient_cv/<dr_method>/<downsampling>/<penalty>/...`
- `summaries/` (cross-method comparison tables and plots)

Minimum machine-readable tables:
- `pooled_grid_metrics.csv`
- `pooled_best_by_objective.csv`
- `pooled_oof_predictions.parquet` (or csv if parquet unavailable)
- `pooled_per_patient_metrics.csv`
- `per_patient_grid_metrics.csv`
- `per_patient_best_by_patient.csv`
- `per_patient_skips_and_warnings.csv`
- `run_manifest.csv` (all run keys + status + runtime)

## Objective selection (default)

Primary selection objective:
- maximize ROC AUC

Tie-breakers:
- PR AUC
- malignant recall
- smaller model norm / stronger regularization preference

Selection is reported separately for:
- pooled mode
- per-patient mode (per patient)

## Operational notes

- This benchmark is classifier-CV only (DR is fixed from Plan 0 artifacts).
- Keep implementation idempotent: skip run units whose output markers already exist.
- If executed as long-running job, always use tee logging and emit stage markers.

## Implementation mapping

Planned runner:
- `../run_plan1c_supervised_latent_benchmark.py` (new)

This runner should not modify Plan 0 raw artifacts; it only reads them and writes analysis outputs.
