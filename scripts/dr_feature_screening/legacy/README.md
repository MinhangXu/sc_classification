# Legacy Comprehensive-Run Scripts

This folder keeps one-off recovery, watcher, and pilot launch scripts out of the active runner namespace while preserving provenance.

Do not treat these as current entry points. Prefer the active scripts documented in `../README.md` and `../plans/INDEX.md`.

## Archived Scripts

- `run_plan1c_pilot_pca_l1.sh`: early PCA-only Plan 1.C pilot superseded by the full `../run_plan1c_supervised_latent_benchmark.py` runner and full K=40 launch wrappers.
- `watch_cnmf_resume.sh`: cNMF resume watcher used during the February 2026 Plan 0 recovery incident; retained for audit trail only.

## Still At Top Level For Now

- `resume_plan0_cnmf.py`
- `resume_plan0_standard_dr.py`
- `run_plan0_resume_standard_dr_varimax.sh`
- `run_plan0_resume_standard_dr_promax.sh`

Those files are still referenced by the active Plan 0 incident notes and should only be moved after compatibility wrappers or updated references are added.
