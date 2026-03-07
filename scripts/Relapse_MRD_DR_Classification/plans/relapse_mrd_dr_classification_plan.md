# Relapse_MRD_DR_Classification Implementation Notes

This folder implements the approved study direction without modifying the attached Cursor plan file.

## Scope

- paired `MRD` and `Relapse` cells only
- per-patient DR, not pooled DR
- RNA-only inputs
- primary supervised task: 4-class classification
- baseline supervised task: hierarchical binary benchmarks

## Implemented pieces

### Cohort audit

- `audit_relapse_mrd_cohort.py`
- writes patient-level feasibility tables and a markdown summary

### Patient-level DR

- `run_patient_level_dr.py`
- default methods: `pca`, `fa`, `factosig`
- default latent dimensionality: `k=40`
- simple patient-local gene selection:
  - `hvg`: variance-ranked genes using existing HVG flags when present
  - `all`: no gene subsetting

### Multiclass benchmark

- `run_patient_level_multiclass_benchmark.py`
- repeated stratified CV inside patient
- weighted multinomial logistic models through the shared logistic backend
- outputs:
  - grid metrics
  - best model summary
  - OOF predictions
  - confusion matrices
  - refit coefficients

### Hierarchical baseline

- `run_patient_level_hierarchical_benchmark.py`
- tasks:
  - `cancer_vs_normal`
  - `time_within_cancer`
  - `time_within_normal`

## Design choices

- The new code does not reuse `PreprocessingPipeline.filter_data()` because that path assumes one timepoint and two classes.
- Tech diagnostics are saved alongside every benchmark run through latent silhouette and neighbor-homogeneity summaries.
- Class balancing is handled with `class_weight='balanced'` in logistic regression for this first implementation.

## Known limitations

- This version does not yet add a tech-corrected RNA control arm.
- `cNMF` and `NMF` are intentionally not first-class in v1 because the current lightweight runtime avoids the full Scanpy stack.
- The patient-level gene selection is deliberately simple so the paired-timepoint design is easy to reason about before adding more aggressive supervised filtering.
