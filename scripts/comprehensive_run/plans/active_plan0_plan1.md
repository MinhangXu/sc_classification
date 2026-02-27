# Active plan (Plan 0–1): K selection → 2-axis grid

This is the **human-readable** version of the active plan that is implemented by the runner(s) in this directory.

Provenance: derived from the raw Cursor snapshot `gene-filtering-eval-plan-iter3-cnmf_a09f862f.plan.md`.

Post-hoc validation/evaluation is tracked in:
- `posthoc_dr_validation_eval_plan.md` (canonical evaluation checklist and chunk status)

## Plan 0 — select \(K\) per DR method (incl. cNMF)

### Goal

Before comparing preprocess methods, select a reasonable \(K\) for each DR method using:

- **diagnostics** (variance proxy / reconstruction proxy)
- **stability** across seeds (consensusness / reproducibility)

### Inputs (default philosophy)

- Use a **single reference preprocess** to decouple K-selection from gene-filter comparisons:
  - typically `hvg` with a large gene set (e.g. 10k) on a fixed cohort filter (e.g. MRD + CITE).

### Methods

- PCA / FA / FactoSig / (optional sklearn NMF)
- cNMF via the `cnmf` package (native replicate outputs; consensus by density threshold `dt`)

### Notes (upcoming updates)

- **FA rotation**: the current comprehensive runner uses sklearn FA (no explicit rotation). We plan to add explicit FA rotation support (varimax/promax) for Plan 0 (and optionally Plan 1). See `plan0rotationseedsplan1stability.md`.
- **Consensusness / multi-seed stability**: in Plan 0, FA/FactoSig consensus clustering is only meaningful when running **multiple seeds** (otherwise stability summaries are empty/degenerate). Plan 1 stability caches are described below but are not yet fully implemented in code; see `plan0rotationseedsplan1stability.md`.
- **Evaluation caveats and interpretation**: metric comparability, rotation diagnostics, K-behavior debugging, and communality comparability are maintained in `posthoc_dr_validation_eval_plan.md` to avoid duplicating interpretation logic here.

### Outputs

- A chosen \(K\) per method (or a shortlist).
- For cNMF: a chosen density threshold `dt`.
- Cached replicate artifacts:
  - cNMF: native `cnmf` output tree under the experiment `models/` folder
  - other DR: arrays-only replicate caches (scores/loadings) and stability summaries

### Implementation

- Runner: `../run_gene_filter_dr_grid.py plan0`
- Key artifavcts:
  - `analysis/plan0/k_selection_summary.csv`
  - `analysis/plan0/stability/<method>/k_<K>/...`
  - `models/cnmf_plan0/` (if installed)

### Current run notes (cNMF resume incident; Feb 2026)

Context: Plan 0 run in experiment dir
`sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886/`
with `K = [20, 40, 60]`, `cnmf_dt=0.5`.

- **Original failure mode (Plan 0 runner)**: `AttributeError: 'cNMF' object has no attribute 'run_nmf'`
  - **Cause**: `cnmf==1.7.0` uses `cNMF.factorize()` (not `run_nmf()`).
  - **Fix**: `run_gene_filter_dr_grid.py` updated to call `factorize()` when present.
- **Resume flow**: use `sc_classification/scripts/comprehensive_run/resume_plan0_cnmf.py` to finish
  `combine_nmf(k)` + `consensus(k)` inside the *same* experiment directory.
- **Important prerequisites for `cnmf==1.7.0 consensus()`**:
  - `cn.paths['tpm']` must exist (expects `.../cnmf_tmp/<name>.tpm.h5ad`)
  - `cn.paths['tpm_stats']` must exist (expects `.../cnmf_tmp/<name>.tpm_stats.df.npz`)
- **Observed failure mode 1**: missing / broken `tpm.h5ad` symlink
  - **Symptom**: `FileNotFoundError` on `...tpm.h5ad` even if `ls` shows a symlink
  - **Cause**: broken symlink (lexists=True, exists=False) created by a prior attempt
  - **Fix**:
    - repair symlink to point to local `.../cnmf_tmp/<name>.norm_counts.h5ad`
    - `resume_plan0_cnmf.py` now auto-removes broken TPM symlinks before re-linking
- **Observed failure mode 2**: missing `tpm_stats.df.npz`
  - **Symptom**: `FileNotFoundError: ...tpm_stats.df.npz` during `cn.consensus()`
  - **Fix**: `resume_plan0_cnmf.py` now reconstructs `tpm_stats` from `tpm.h5ad`
    (same mean/std computation as `cnmf.prepare()`).
  - **Marker file**: `analysis/plan0/cnmf_tpm_stats_created.json`
- **Monitoring protocol (reproducible)**:
  - Always run with a log file under `analysis/plan0/`, e.g.
    `analysis/plan0/resume_plan0_cnmf_<timestamp>.log`
  - Watch for “done markers” written by the resume script:
    `analysis/plan0/cnmf/k_<K>/consensus_stats.json`
  - Prefer `pgrep` on the **absolute script path** to avoid matching the watcher itself.
- **Post-run organization (non-destructive)**:
  - For easier downstream notebook IO, build a curated cNMF view:
    `python sc_classification/scripts/comprehensive_run/reorganize_plan0_cnmf_curated.py --experiment-dir <EXP_DIR> --mode symlink`
  - This creates `models/cnmf_plan0/curated/` with:
    - `global/`
    - `k_<K>/inputs/` and `k_<K>/consensus/`
    - `MANIFEST.csv`, `README.md`
  - Source files under `models/cnmf_plan0/plan0_cnmf/` remain untouched.

Example watcher (uses newest log automatically):

```bash
EXP="/home/minhang/mds_project/sc_classification/experiments/20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
SCRIPT="/home/minhang/mds_project/sc_classification/scripts/comprehensive_run/resume_plan0_cnmf.py"

watch -n 60 "
LOG=\$(ls -1t \$EXP/analysis/plan0/resume_plan0_cnmf_*.log | head -n 1)
date
echo \"LOG=\$LOG\"
pgrep -af \"python .*${SCRIPT}\" || echo '(not running)'
tail -n 20 \"\$LOG\" 2>/dev/null || true
for k in 20 40 60; do
  f=\"\$EXP/analysis/plan0/cnmf/k_\${k}/consensus_stats.json\"
  [ -f \"\$f\" ] && ls -lh \"\$f\" || echo \"missing: \$f\"
done
"
```

## Plan 1 — main 2-axis eval (preprocess × DR)

### Axes

- **Preprocess methods** (gene selection pipelines):
  - `hvg`
  - `all_filtered`
  - `deg_weak_screen`
  - `hybrid` = HVG ∪ all_filtered rescue (implemented as `hvg_plus_rescue_union`)
- **DR methods**:
  - `pca`, `fa`, `nmf`, `factosig`, `cnmf`
  - using the frozen \(K\) mapping decided from Plan 0

### Plan 1.A — no-CV fast pass

For each preprocess method:

- compute DR embeddings for all DR methods and attach to a single `AnnData`
- run per-patient LR-L1 metrics with `cv_folds=0`
- persist a main artifact: one `.h5ad` per preprocess method

### Plan 1.B — CV (classifier-only)

Repeat Plan 1.A with classifier CV enabled (e.g. `cv_folds=5`, `cv_repeats=10`).

Important: this is **classifier-only CV** unless/until DR fitting is nested later.

### Implementation

- Runner: `../run_gene_filter_dr_grid.py plan1`
- Notes:
  - the `.h5ad` keeps only one embedding per method (first seed), to avoid file bloat
  - multi-seed replicate caches + stability/consensus diagnostics are intended to live under `analysis/plan1_stability/...` (see `plan0rotationseedsplan1stability.md`)

## Plan 1.C — fixed-K supervised latent benchmark (next)

### Goal

At fixed `K=40`, evaluate latent-factor classification quality across DR methods and regularization families using:
- pooled-cell repeated stratified CV (pan-patient), with per-patient metrics from OOF predictions
- per-patient repeated stratified CV

### Methods in scope

- DR: `pca`, `fa`, `factosig`, `factosig_promax`, `cnmf`
- penalties: L1, L2, Elastic Net
- grids:
  - `alpha = logspace(-4, 5, 20)` (`C=1/alpha`)
  - elastic-net `l1_ratio in [0.1, 0.5, 0.9]`
- downsampling variants:
  - `none`
  - `random` donor downsampling stratified by `predicted.annotation` with prior FA-era parameters

### Plan doc

- Detailed spec: `plan1c_cross_patient_supervised_latent_benchmark.md`

### Implementation status

- Planned runner: `../run_plan1c_supervised_latent_benchmark.py` *(not implemented yet)*

