---
id: W6
title: W6_learning_curves: patient-count and cell-depth curves with C retuned (D15b, D15 v2)
status: active
questions: [Q2, meta]
---

# W6 — W6_learning_curves: patient-count and cell-depth curves with C retuned (D15b, D15 v2)

Status as of 2026-10-08: active (P1; D15b is P0 for the grant argument). D15b (W6.1) ran in its reduced design; the full enumeration is handed to user screen `W6_d15b` (devlog 2026-10-08_d15b-patient-count-curve; planned row in `scripts/mrd_stage0_2/research/campaigns.csv`).

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "Plot AUPRC vs number of cells (learning / marginal-benefit curve) ('we can do this for sure, right?')" (A-20261008-24)

> "Learning curve useful for grants to justify sequencing more patients" (A-20261008-25)

> "Randomly subsample to estimate the curve (as Audit AI does with Eclipse)" (A-20261008-26)

Scope: Q2 and grant justification (meta). D15 exists, but it is a cell-depth curve at a fixed patient mix with C not retuned, so it says nothing about adding patients.

Done means: a patient-count learning curve (D15b) with C and penalty retuned per training set and patient-bootstrap CIs; a cell-depth curve (D15 v2) that replaces D15's interpretation; and a grant note that states what the curves can and cannot justify about sequencing more patients (H-S2-06), later extended with the new cohort's patients as external points.

## Builds on

- Registry rows owned here: D15, D15b (partial). D15 v2 (W6.2) is a new version of D15 under a new slug; W6.3 reserves a new ID when it starts.
- Hypotheses: H-S2-06.
- Code: `scripts/mrd_stage0_2/stage2_supervised/cell_count_learning_curve.py` (D15: its four representations and data loading are reused by D15b).
- Notebook: `scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_patient_factor_overlap.ipynb` (D15 section).
- EXP-relative inputs: `analysis/scorecards/stage2_diagnostic/cell_count_learning_curve_v1/` (no findings md yet; W0.5), `stage1_dr/` (frozen transductive bases), the cohort cache (study_config `cohort_cache_relpath`), `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`.
- C policy: W3's `regularization_contract` (D19); `../W3_dr_value_regularization/anchors/l1_entry_check_scratch_20261008.py` prints the fixed-C L1 entry threshold at D15's subsample sizes.
- Devlog entries: 2026-10-02_d15-cell-count-learning-curve (c_policy global_selected with training_n_changed true; kept as history), 2026-10-08_team-report-devlog-pivot, 2026-10-08_literature-reviews.
- Literature routing (`../W9_literature/README.md`): patient-drop curve → W6.1; cell-depth curve retuned → W6.2; 2D design → W6.3. Label remap: the review's "D15b" is D15 v2 (W6.2), its "D15c" is D15b (W6.1), and its "D15d" is W6.3.
- External points: W8 (new cohort) feeds W6.5.

## Backlog

- [W6.1] (P0) D15b patient-count learning curve (slug patient_count_learning_curve_v1). (depends: W3.1)
  - For each held-out supported patient, train on k in 1..5 of the other supported patients, over all combinations. Unsupported patients are always included.
  - Retune C and the penalty for each training set via regularization_contract; at k=1, fall back to within-patient stratified CV.
  - Use the frozen transductive basis, labelled as such, and the four D15 representations.
  - Report patient-bootstrap CIs and fit a weighted inverse power law, validated by predicting k=5 from k<=4.
  - The runner is resumable; the full run goes in user screen.
  - Done when: the curve table, figure and findings exist.
- [W6.2] (P1) D15 v2 cell-depth curve (slug cell_count_learning_curve_v2): C retuned per subsample, more draws, x-axis in training malignant cells and in malignant cells per feature, and a findings md — done when: a devlog entry states that v2 replaces v1's interpretation. (depends: W3.1)
- [W6.3] (P2) 2D design: k training patients × a cap on malignant cells per patient. Reserve a new ID — done when: a surface table exists. (depends: W6.1)
- [W6.4] (P1) Grant note: what D15 and D15b can and cannot justify about sequencing more patients (H-S2-06) — done when: THEME §3 and §10 are updated at a milestone. (depends: W6.1)
- [W6.5] (P1) Add Yu-Hsiang's patients as external points on the D15b curve. (depends: W6.1) (blocked: new data, W8.1)
- [W6.6] (P1) D15b on an inductive Stage 1 basis: refit the basis on each training subset (or at least without the held-out patient), project the held-out patient, and rerun the patient-count curve on the same subsets — done when: transductive and inductive curves are compared in one table and figure. (depends: W6.1, W2.1)

## Code in this directory

- `patient_count_curve.py` is the D15b library. It uses numpy, pandas and scipy only and does no I/O.
  - Design:
    - `enumerate_subsets` and `design_table(design='full'|'reduced')` build the training sets. Each is k of the other supported patients plus every unsupported patient. k = 6 is the ordinary LOPO fold.
    - `cyclic_design_family` and `reduced_subsets` build a deterministic cyclic family of 7 subsets per size. Sizes 3 and 4 are balanced incomplete block designs (sizes 1, 3, 4 and 6 are balanced over training-patient membership; sizes 2 and 5 are not, so reduced-design k = 2 and k = 5 points are skewed), and the reduced units are a subset of the full units.
    - `training_mask` selects the training cells; `select_tuned_penalty` picks the penalty with the best inner score, with ties going to L1.
  - Summaries:
    - `summarize_by_patient_k` gives the lift distribution over subsets for each (held-out patient, k).
    - `curve_by_k` gives the median, P20 and patient-equal mean, with patient-cluster bootstrap CIs.
    - `paired_gain` gives the per-patient gain from a baseline k to each other k. The report writes k = 2 (primary: inner-LOPO tuning at both ends) and k = 1 (reference: k = 1 tuning falls back to stratified cell CV).
    - `patient_inclusion_effect` is descriptive.
  - Power law: `fit_inverse_power_law`, `validate_power_law` and `power_law_table` fit a weighted a − b·k^(−c). The primary validation fits k ≤ 4 and predicts k = 5; the secondary one fits k ≤ 5 and predicts k = 6. A fit is reportable only if the primary validation passes.
- `run_d15b_patient_count_curve.py` is the D15b runner (W6.1; slug `patient_count_learning_curve_v1`).
  - It always uses the frozen transductive Stage 1 basis and labels every output row with it.
  - C policy comes from W3's `regularization_contract`, imported via `add_study_module_paths(workstreams=("W3_dr_value_regularization", ...))`:
    - **tuned** (the headline): penalty L1/L2 and C chosen by `tune_c_inner_lopo` for each training set, using the contract's 5-fold CV fallback at k = 1;
    - **tuned_l1** and **tuned_l2**;
    - **fixed_atlas** (comparable with D15);
    - **scaled_to_n** (C from `c_scaled_to_n`).
  - Parts:
    - `tune` runs once per (representation, subset, penalty), because tuning does not depend on the held-out patient.
    - `units` runs once per (representation, held-out patient, subset).
    - `report` writes design-prefixed tables `d15b_<design>_*.csv`, figures `D15bA–D15bD_<design>_*`, the findings md, the manifest and the artifact ledger.
  - Run options:
    - `--resume` skips finished units; empty, corrupt or out-of-date files are rerun.
    - `--design full|reduced` picks the design.
    - `--smoke` runs held-out P07, k ∈ {1,2}, allograft promax K40. It needs `--out-root`.
  - The k = 6 reference checks compare against the D1 atlas, D15 and D19.
  - Reduced design (fits the agent shell): `python scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py --design reduced --resume --n-jobs 12`.
- `run_d15b_patient_count_curve.sh` is the screen wrapper for the full design.
  - It runs `--design full --resume --n-jobs 12` with 1 BLAS thread per worker and tees a log to `analysis/cache/stage2_diagnostic/patient_count_learning_curve_v1/logs/`.
  - To launch: `screen -S W6_d15b`, then `bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh`, then detach with Ctrl-a d. The planned row is in `../../research/campaigns.csv` (namespace `patient_count_learning_curve_v1`).
  - Overrides: `SC_CONDA_PREFIX`, `PYTHON`, `N_JOBS`.
- `tests/test_patient_count_curve.py` (repo level):
  - synthetic tests cover subset enumeration, design balance, the training mask, the bootstrap, the summaries and the power-law fit (recovery, validation, and rejection on flat or missed curves);
  - two EXP regression tests, skipped when EXP is absent, check the stored reference checks and one recomputed k = 6 fixed-C fold against the atlas.
- D15 v2 (W6.2) code: later; no package assigned yet.
- Screen sessions: `W6_d15b` (full D15b run, CPU). Suggested `W6_d15v2` for W6.2.

Task status (derived from the devlog; see `../../research/STATUS.md`): W6.1 is advanced (status partial) by devlog 2026-10-08_d15b-patient-count-curve. The reduced design ran in the agent shell and the full enumeration is handed to screen `W6_d15b`; a follow-up type: run, status: done entry that supersedes it closes W6.1 once the `d15b_full_*` tables exist. W6.4 can draft from the reduced results now (the k = 1 and k = 6 endpoints are complete); it must state that every subset's basis saw every patient, and that justifying more sequencing needs an inductive basis per subset (W2.1, W6.6) plus new patients as external points (W6.5, W8.1). W6.3 matters more after D15b, because patient count and malignant-cell count are confounded across subsets. W6.6 was added from that entry's Next steps.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `patient_count_learning_curve_v1` (D15b), `cell_count_learning_curve_v2` (D15 v2). `cell_count_learning_curve_v1` is history and is never overwritten. R3: a new slug ends in `_v<N>`; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train), so a fixed C over-penalizes small training sets and flattens learning curves mechanically. Every point on a W6 curve changes training n: retune C and the penalty per training set by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (D19) and report L2 alongside. Record `c_policy` (per_fold_tuned) and `training_n_changed: true` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal".
- Label every curve's basis: the frozen transductive Stage 1 basis means the held-out patient helped define the factors (R6).
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

- Screen sessions are named `<W#>_<topic>`. Long jobs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with status: partial and compute: screen-cpu; a status: done entry supersedes it when the run finishes. New runners implement `--resume` (R2).
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W6.n`.
