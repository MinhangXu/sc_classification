---
id: W2
title: W2_stage1_basis_variants: inductive, leave-out, patient-balanced and subset Stage 1 bases (D17)
status: ready
questions: [Q1, Q3]
---

# W2 — W2_stage1_basis_variants: inductive, leave-out, patient-balanced and subset Stage 1 bases (D17)

Status as of 2026-10-08: ready (P0). No code yet. Template: `run_basis_refit_target()` in the donor runner. `row_weights` in the DR wrappers is silently ignored. D17 waits for W3.1 (the C-tuning API).

## Charter

Verbatim asks (recorded in `research/asks/2026-10-08_team-report.md`):

> "Option 2: exclude the held-out patient from Stage 1 entirely, then project their cells onto the learned factors" (A-20261008-16)

> "Option 1.2: re-weight cells so each patient contributes equally to DR" (A-20261008-14)

> "Option 1.1: subset to the top 4-5 patients with sufficient malignant cells, plus YH's new patient; plan the run after choosing" (A-20261008-13)

> "Option 3: focus only on patients with large cell counts (2, 3, 4) and rerun the full pipeline" (A-20261008-17)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02; shared with W1, routed here through W2.1)

Wording note (AGENTS.md R6): A-20261008-02 and A-20261008-17 are quoted verbatim. P02, P03 and P04 are the highest-prevalence patients; they are neither the largest patients by cell count nor the patients with the most malignant cells (see `research/study_config.json`). Option 3 is therefore read as the leave-high-prevalence-out arm of W2.1 and the subset pipeline of W2.5.

Scope: Q1 and Q3. The user's options 2 (inductive: exclude the held-out patient from Stage 1 and project it), 1.2 (re-weight so each patient contributes equally) and 1.1/3 (patient subsets). This workstream supplies the bases that W1, W3, W6 and W8 consume.

Done means: inductive, leave-high-prevalence-out, patient-balanced and subset Stage 1 bases exist with persisted projection parameters; their Stage 2 transfer is measured with C retuned per fold and compared with the transductive anchor; and the answers feed H-S1-03, H-S1-05 and H-S1-06.

## Builds on

- Registry: D3 and D14 (W1), D4 (W3), D10 and D11 (W4; the donor runner's basis-refit arm). Planned here: D17.
- Hypotheses: H-S1-03, H-S1-04, H-S1-05, H-S1-06.
- Template: `run_basis_refit_target()` and `fit_basis_model()` in `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py`.
- DR wrappers: `src/sc_classification/dimension_reduction/pca.py`, `src/sc_classification/dimension_reduction/factor_analysis.py`, `src/sc_classification/dimension_reduction/factosig.py`. They accept `row_weights`, but `preprocess_data` in `src/sc_classification/dimension_reduction/base.py` ignores it, so weighting must be implemented in `stage1_refit.py` (W2.2 decides whether the wrappers should raise instead, through a type: refactor entry).
- Stage 1 producer and conventions: `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py`; Stage 2 conventions: `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`.
- EXP-relative inputs: `stage1_dr/` (transductive bases; read-only), `stage1_direct_gene/` (covered-gene features), `stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv` (the transductive per-patient lifts the anchor arm must reproduce).
- C policy: W3's `regularization_contract` (wave 2); D17 waits for W3.1.
- Devlog entries: 2026-08-21_donor-sensitivity-d10-d11 (basis-refit arm), 2026-10-02_d14-patient-factor-overlap, 2026-10-08_team-report-devlog-pivot.
- Literature routing (`../W9_literature/README.md`): narrow inductive Stage 1 → W2.1; per-patient refits and centering ablation → W2.3.

## Backlog

{{BACKLOG}}

## Code in this directory

- `stage1_refit.py`: wave 2 (planned). Fit and project PCA and FA with train-only standardization; patient subset and leave-out options; patient weights (weighted covariance plus eigh for PCA, patient-stratified subsampling for FA); persists gene order, mean/sd, components and noise variance/mean.
- `run_d17_basis_variant_pilot.py`: wave 2 (planned). D17 runner with `--resume` per arm/method/fold. Imports W3's contract through `add_study_module_paths(workstreams=("W3_dr_value_regularization",))`.
- `run_d17_basis_variant_pilot.sh`: wave 2 (planned). Screen wrapper.
- `tests/test_stage1_refit.py` (repo-level): wave 2 (planned). Toy matrices for fit/project/persist and for the weighting.
- Screen sessions: `W2_d17` (D17, if it exceeds about 5 minutes). W2.5 and W2.6 are GPU runs in user screen under `W2_<topic>` sessions, each after a campaigns.csv row is added.

## Conventions

- Owned artifact slugs (EXP-relative): `stage1_dr_variants/basis_variant_pilot_ifna_v1/` (bases) and `analysis/{scorecards,figures,cache}/stage2_diagnostic/basis_variant_pilot_ifna_v1/` (D17); `stage1_dr_variants/inductive_lopo_v1/` (W2.6). New bases always go under `stage1_dr_variants/<slug>_v<N>/`, never `stage1_dr/` (R3). A new slug is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Every W2 arm changes the training data, so retune C per fold by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (wave 2), keep L1 at the stored C only as the anchor, and report L2 alongside. Record `c_policy` (per_fold_tuned) and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal".
- Import header for a new script in this directory:

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```

- Screen sessions are named `<W#>_<topic>`. DR/FactoSig grids and GPU Stage 2 runs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with compute: screen-cpu or screen-gpu. New runners implement `--resume` (R2). Add a planned campaigns.csv row before handing over a new GPU campaign (R4).
- Label every basis by fit scope (transductive, inductive, leave-out, balanced, subset); LOPO on the transductive basis is label transfer, not inductive deployment (R6).
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W2.n`.
