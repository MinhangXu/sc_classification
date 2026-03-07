# Relapse_MRD_DR_Classification

This study adapts the existing MRD latent-factor benchmarking workflow to a paired longitudinal setting:

- unit of analysis: per patient
- timepoints: coarse `MRD` and `Relapse`
- label space: `MRD_cancer`, `MRD_normal`, `Relapse_cancer`, `Relapse_normal`
- primary benchmark: within-patient 4-class classification
- secondary baseline: hierarchical binary benchmarks

## Why this folder exists

The current `comprehensive_run` code assumes:

- one timepoint filter at a time
- binary `cancer` vs `normal` targets
- pooled or per-patient classification on a shared DR space

This folder changes the design to fit DR separately inside each relapsing patient while keeping both MRD and Relapse cells in the same patient-specific latent space.

## Current cohort audit highlights

The paired cohort audit against `data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad` shows:

- 9 patients with both coarse `MRD` and `Relapse`: `P01`, `P02`, `P03`, `P04`, `P05`, `P07`, `P08`, `P09`, `P13`
- severe class imbalance for several patients, especially `MRD_cancer`
- `P08` has `MRD_cancer = 0`, so it should be skipped for 4-class modeling
- most paired patients follow the expected tech split of MRD=`CITE` and Relapse=`Multi`
- notable exceptions:
  - `P13` has both MRD and Relapse in `CITE`
  - `P01`, `P02`, and `P03` contain mixed-tech MRD cells

The audit script writes machine-readable tables under `cohort_audit/`.

## Main scripts

- `audit_relapse_mrd_cohort.py`
  - lightweight h5py-only cohort audit
- `run_patient_level_dr.py`
  - filters to paired MRD/Relapse cells
  - preprocesses each patient separately
  - fits patient-level DR with `pca`, `fa`, or `factosig`
- `run_patient_level_multiclass_benchmark.py`
  - 4-class within-patient benchmark on saved latents
- `run_patient_level_hierarchical_benchmark.py`
  - binary sanity-check tasks:
    - `cancer_vs_normal`
    - `time_within_cancer`
    - `time_within_normal`

## Suggested execution order

```bash
python sc_classification/scripts/Relapse_MRD_DR_Classification/audit_relapse_mrd_cohort.py \
  --input-h5ad data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad

python sc_classification/scripts/Relapse_MRD_DR_Classification/run_patient_level_dr.py \
  --input-h5ad data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad \
  --methods pca,fa,factosig \
  --k 40 \
  --gene-method hvg \
  --hvg-n 3000

python sc_classification/scripts/Relapse_MRD_DR_Classification/run_patient_level_multiclass_benchmark.py \
  --dr-output-dir sc_classification/experiments/<run_id>

python sc_classification/scripts/Relapse_MRD_DR_Classification/run_patient_level_hierarchical_benchmark.py \
  --dr-output-dir sc_classification/experiments/<run_id>
```

## Output structure

The DR runner creates a new experiment directory containing:

- `input_diagnostics/`
- `patient_dr/<patient>/preprocessed.h5ad`
- `patient_dr/<patient>/<method>/scores.npy`
- `patient_dr/<patient>/<method>/loadings.npy`
- `patient_dr/<patient>/<method>/latent_diagnostics.csv`

The multiclass and hierarchical runners write into the same experiment root:

- `multiclass_cv/<patient>/<method>/`
- `hierarchical_cv/<patient>/<method>/<task>/`

## Important interpretation rule

This first implementation is intentionally RNA-only. Any MRD-vs-Relapse separation must be read together with the saved tech diagnostics, because timepoint and assay are partially entangled in this cohort.
