---
id: W3
title: W3_dr_value_regularization: what DR adds over direct genes, and the regularization/selection contract (D19, D23, D24)
status: ready
questions: [Q3, Q2]
---

# W3 — W3_dr_value_regularization: what DR adds over direct genes, and the regularization/selection contract (D19, D23, D24)

Status as of 2026-10-08: ready (P0). D19 comes first because it unblocks W2.1, W6.1 and W4.6.

## Charter

Verbatim asks (recorded in `research/asks/2026-10-08_team-report.md`):

> "Q3: what does dimensionality reduction add over direct-gene classification?" (A-20261008-32)

> "DISCUSSED: IFN-alpha response, PCA K=40 gives lift in four patients but FA K=40 captures only P03 and P04" (A-20261008-03)

> "DISCUSSED (a): hypothesis that PCA orthogonality pushes later components beyond P03, while FA keeps allocating factors to the highest-prevalence patient" (A-20261008-04)

> "Ankit: end-to-end semi-supervised analysis with factors learned jointly to discriminate healthy vs malignant (later, once new data arrive)" (A-20261008-27)

Earlier asks, cited by ID (their recorded text is in `research/asks/`): A-20260724-02 (no-DR baseline → D9; capture reconstructed), A-20260917-02 (DR vs direct genes, LCK/ZAP70 → F2, F5), A-20260917-03 (teammate:Ankit, collinear L1 → F3, H-S2-07).

Scope: Q3 (primary) and Q2. Covers the IFN-alpha PCA40-vs-FA40 question (A-20261008-03, A-20261008-04); retuning C whenever n changes (R5); nested selection vs the winner's curse in D9/F4; single-score baselines; and Ankit's semi-supervised model (A-20261008-27; later). The contract module here is imported by W2, W4 and W6.

Done means: one shared regularization contract implements R5 and is used by every refit; the IFN-alpha PCA40-vs-FA40 gap is settled under tuned C (H-S2-08) and team hypothesis (a) is formally retired (H-S2-07); and DR-vs-direct value is restated under nested selection (D23) against single-score baselines (D24), with pooled AP, median lift and P20 lift reported together.

## Builds on

- Registry rows owned here: D1e, D4, D9, D12, D1DD5, D1DD9, F1, F2, F3, F4. Planned: D19, D23, D24.
- Used from other workstreams: D8, F6, F9 (W5; the IFN-alpha PC8/FA7 case), D10 and D11 (W4) and D15 (W6) as fixed-C consumers that the contract will revisit.
- Hypotheses: H-S0-01, H-S0-02, H-S0-04, H-S1-01, H-S1-02, H-S2-07, H-S2-08.
- Stage 2 conventions the contract must match (C grid, solver, balanced class weights, training-fold standardization): `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`, which builds its estimator through `src/sc_classification/utils/logistic_backend.py`.
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_feedback_deep_dives_20260917.ipynb` (F1-F4), `stage2_old34_meeting_followup_metric_interactions.ipynb` (D9), `stage2_old34_layer4_5_interpretability_suite.ipynb` (D4), `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` (D12), `stage2_old34_comprehensive_diagnostic_suite.ipynb` (D1e).
- EXP-relative inputs: `stage1_dr/` (stored transductive scores), `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv`, `analysis/scorecards/stage2_diagnostic/old34_d1e_best_dr_vs_direct_gene_selection.csv`, `analysis/scorecards/stage2_diagnostic/layer4_5/`, `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/` (F1-F4 tables, including `f3_ifna_local_l1_l2_en_refit_on_stored_scores.csv`), `analysis/scorecards/stage2_diagnostic/meeting_followup_v1/` (D9 direct-gene comparisons), Stage 2 namespaces `old34_direct_gene_all_v1` and `old34_controls_direct_gene_v1` (see `research/campaigns.csv`).
- Regression anchor: `anchors/l1_entry_check_scratch_20261008.py` (W3.1 must reproduce it).
- Devlog entries: 2026-10-08_team-report-devlog-pivot (the L1-at-fixed-C correction), 2026-10-08_literature-reviews (knife-edge C sweep), 2026-09-17_team-feedback-f1-f7, 2026-08-31_d6-d9-meeting-followup, 2026-09-03_d12-d13-gsea-hvg-lowsupport, 2026-05-18_old-geneset-bottom-up-postrun, 2026-05-15_stage0-value-added-plan.
- Literature routing (`../W9_literature/README.md`): knife-edge → W3.2; entry-margin audit, Stage 2 retuning → W3.1; nested LOPO / inner-LOPO selection → W3.3; one-score comparators → W3.4; stability selection → W3.6; Ankit ladder → W3.7.
- Dependents: W2.1, W4.6, W6.1 and W6.2 wait on W3.1.

## Backlog

{{BACKLOG}}

## Code in this directory

- `anchors/l1_entry_check_scratch_20261008.py`: frozen regression anchor (balanced-L1 entry threshold 4/(C·n_train) vs the training-fold class-mean gap, IFN-alpha PCA40/FA40). Not a pipeline module; never imported.
- `regularization_contract.py`: wave 2 (planned). The single shared implementation of the C policy (R5); numpy and scikit-learn only; class_weight 'balanced', standardization on training cells, the runner's solver family. Fixed API, imported by W2, W4 and W6:
  - `DEFAULT_C_GRID`: the runner's C grid.
  - `l1_entry_threshold(C, n_train) -> float`: 4/(C·n_train).
  - `l1_entry_margins(Z, y, C) -> ndarray`: |standardized training class-mean gap| / threshold; above 1 the feature can enter at w = 0.
  - `tune_c_inner_lopo(Z, y, patients, *, penalty='l1', l1_ratio=None, c_grid=DEFAULT_C_GRID, min_malignant=10, scoring='average_precision', random_state=42) -> dict` with best_C, scores_by_C, n_inner_folds and fallback (stratified 5-fold CV on the training cells when fewer than 2 supported training patients exist).
  - `fit_probe(Z, y, *, penalty, C, l1_ratio=None, random_state=42, max_iter=5000)`: a fitted estimator.
  - `heldout_lift(scores, y) -> (ap, prevalence, lift)`.
  - `c_policy_record(**kw) -> dict`, for manifests.
  - API changes go through a devlog entry with type: refactor.
- `run_d19_regularization_contract.py`: wave 2 (planned). D19 runner with `--resume` per (representation, fold): the L1 entry-margin audit and the W3.2 C-matched IFN-alpha comparison.
- `tests/test_regularization_contract.py` (repo-level): wave 2 (planned). Threshold formula, margin vs nonzero liblinear coefficient, deterministic tuning, the fallback.
- Screen sessions: `W3_d19` (D19, if a part exceeds about 5 minutes). W3.5 (patient-specific direct-gene grid, GPU) runs in user screen under a `W3_<topic>` session after a campaigns.csv row is added.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `regularization_contract_v1` (D19), `nested_dr_vs_direct_v1` (D23), `single_score_baselines_v1` (D24). New Stage 2 grids (W3.5) use a new `--stage2-output-branch` namespace. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5), owned here: sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Whenever training n changes, retune C per condition by inner LOPO over the training patients with `tune_c_inner_lopo` and report L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal". Flag every L1 "method win" or "driver" whose entry margin sits within the knife-edge band set in W3.1.
- Import header for a new script in this directory (files in `anchors/` sit one level deeper and use `parents[5]`):

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```

- Other workstreams import the contract with `add_study_module_paths(workstreams=("W3_dr_value_regularization",))` and then `import regularization_contract`.
- Screen sessions are named `<W#>_<topic>`. Long or GPU jobs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with compute: screen-cpu or screen-gpu. New runners implement `--resume` (R2).
- Claim language (R6): quote pooled AP, median lift and P20 lift together; for any loading, state the sign orientation and the penalty.
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W3.n`.
