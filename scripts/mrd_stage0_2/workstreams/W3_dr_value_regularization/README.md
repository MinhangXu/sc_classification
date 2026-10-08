---
id: W3
title: W3_dr_value_regularization: what DR adds over direct genes, and the regularization/selection contract (D19, D23, D24)
status: active
questions: [Q3, Q2]
---

# W3 — W3_dr_value_regularization: what DR adds over direct genes, and the regularization/selection contract (D19, D23, D24)

Status as of 2026-10-08: active (P0). D19 (W3.1, W3.2) is done (devlog 2026-10-08_d19-regularization-contract): the shared contract module exists and W2, W4 and W6 import it. Next: W3.3 (D23, screen-cpu) and W3.8.

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "Q3: what does dimensionality reduction add over direct-gene classification?" (A-20261008-32)

> "DISCUSSED: IFN-alpha response, PCA K=40 gives lift in four patients but FA K=40 captures only P03 and P04" (A-20261008-03)

> "DISCUSSED (a): hypothesis that PCA orthogonality pushes later components beyond P03, while FA keeps allocating factors to the highest-prevalence patient" (A-20261008-04)

> "Ankit: end-to-end semi-supervised analysis with factors learned jointly to discriminate healthy vs malignant (later, once new data arrive)" (A-20261008-27)

Earlier asks, cited by ID (their recorded text is in `scripts/mrd_stage0_2/research/asks/`): A-20260724-02 (no-DR baseline → D9; capture reconstructed), A-20260917-02 (DR vs direct genes, LCK/ZAP70 → F2, F5), A-20260917-03 (teammate:Ankit, collinear L1 → F3, H-S2-07).

Scope: Q3 (primary) and Q2. Covers the IFN-alpha PCA40-vs-FA40 question (A-20261008-03, A-20261008-04); retuning C whenever n changes (R5); nested selection vs the winner's curse in D9/F4; single-score baselines; and Ankit's semi-supervised model (A-20261008-27; later). The contract module here is imported by W2, W4 and W6.

Done means: one shared regularization contract implements R5 and is used by every refit; the IFN-alpha PCA40-vs-FA40 gap is settled under tuned C (H-S2-08) and team hypothesis (a) is formally retired (H-S2-07); and DR-vs-direct value is restated under nested selection (D23) against single-score baselines (D24), with pooled AP, median lift and P20 lift reported together.

## Builds on

- Registry rows owned here: D1e, D4, D9, D12, D1DD5, D1DD9, F1, F2, F3, F4, D19. Planned: D23, D24.
- Used from other workstreams: D8, F6, F9 (W5; the IFN-alpha PC8/FA7 case), D10 and D11 (W4) and D15 (W6) as fixed-C consumers that the contract will revisit.
- Hypotheses: H-S0-01, H-S0-02, H-S0-04, H-S1-01, H-S1-02, H-S2-07, H-S2-08.
- Stage 2 conventions the contract must match (C grid, solver, balanced class weights, training-fold standardization): `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`, which builds its estimator through `src/sc_classification/utils/logistic_backend.py`.
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_feedback_deep_dives_20260917.ipynb` (F1-F4), `stage2_old34_meeting_followup_metric_interactions.ipynb` (D9), `stage2_old34_layer4_5_interpretability_suite.ipynb` (D4), `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` (D12), `stage2_old34_comprehensive_diagnostic_suite.ipynb` (D1e).
- EXP-relative inputs: `stage1_dr/` (stored transductive scores), `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv`, `analysis/scorecards/stage2_diagnostic/old34_d1e_best_dr_vs_direct_gene_selection.csv`, `analysis/scorecards/stage2_diagnostic/layer4_5/`, `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/` (F1-F4 tables, including `f3_ifna_local_l1_l2_en_refit_on_stored_scores.csv`), `analysis/scorecards/stage2_diagnostic/meeting_followup_v1/` (D9 direct-gene comparisons), Stage 2 namespaces `old34_direct_gene_all_v1` and `old34_controls_direct_gene_v1` (see `scripts/mrd_stage0_2/research/campaigns.csv`).
- Regression anchor: `anchors/l1_entry_check_scratch_20261008.py` (W3.1 must reproduce it).
- Devlog entries: 2026-10-08_team-report-devlog-pivot (the L1-at-fixed-C correction), 2026-10-08_literature-reviews (knife-edge C sweep), 2026-09-17_team-feedback-f1-f7, 2026-08-31_d6-d9-meeting-followup, 2026-09-03_d12-d13-gsea-hvg-lowsupport, 2026-05-18_old-geneset-bottom-up-postrun, 2026-05-15_stage0-value-added-plan.
- Literature routing (`../W9_literature/README.md`): knife-edge → W3.2; entry-margin audit, Stage 2 retuning → W3.1; nested LOPO / inner-LOPO selection → W3.3; one-score comparators → W3.4; stability selection → W3.6; Ankit ladder → W3.7.
- Dependents: W2.1 (D17) and W6.1 (D15b) already use the contract; W4.6 and W6.2 can start on it.

## Backlog

- [W3.1] (P0) D19 regularization contract (slug regularization_contract_v1) — done when: the module, its tests, the audit table and the findings exist.
  - Write regularization_contract.py with the l1_entry_threshold / l1_entry_margins / tune_c_inner_lopo / fit_probe API.
  - Run an entry-margin audit over every L1-selected representation × fold, flagging margins within 10 percent of the threshold.
  - Reproduce the IFN-alpha zero/nonzero outcomes of anchors/l1_entry_check_scratch_20261008.py.
- [W3.2] (P0) C-matched IFN-alpha PCA40 vs FA40, as part of D19: per-fold inner-LOPO C, with L1, L2 and EN side by side, plus a sweep of C around the stored value — done when: the comparison table settles H-S2-08 and a devlog entry formally retires H-S2-07. (depends: W3.1)
- [W3.3] (P0) D23 nested or prespecified DR-vs-direct selection for the 41 biological panels and 6 controls, using stored per-fold metrics. The inner LOPO chooses method, K and C; the outer loop gives held-out lift. Report pooled AP, median lift and P20 lift together — done when: a table compares nested with in-sample selection. (depends: W3.1)
- [W3.4] (P1) D24 single-score baselines: UCell, AUCell, singscore, unnormalized ssGSEA, scanpy score_genes and mean-z, each as a single-feature Stage 2 under the same LOPO and metrics. Note which scorers are transductive — done when: a per-panel baseline lift table exists.
- [W3.5] (P1) Run the patient-specific direct-gene grid, which has never been run. GPU in user screen, on a new branch registered in campaigns.csv first — done when: merged scorecards exist under the new namespace.
- [W3.6] (P2) Stability selection, or cluster-level selection, for correlated factosig_promax factors — done when: driver claims carry selection frequencies.
- [W3.7] (later) Ankit's semi-supervised ladder (supervised PCA, sPLS-DA, linear supervised autoencoder with a lambda sweep) inside nested LOPO, with every supervised step refit per fold. Comparator: the W2 inductive basis plus the W3.1 contract — done when: a pre-registered comparison entry exists. (blocked: new data, W8.2)
- [W3.8] (P1) Per-fold tuned-C re-scoring of the D19 knife-edge atlas cells (L1, L2 and EN via `tune_c_inner_lopo`) — done when: a table compares fixed-C and tuned-C pooled AP, median lift and P20 lift for those cells. (depends: W3.1)

## Code in this directory

- `anchors/l1_entry_check_scratch_20261008.py`: frozen regression anchor (balanced-L1 entry threshold 4/(C·n_train) vs the training-fold class-mean gap, IFN-alpha PCA40/FA40). Not a pipeline module; never imported. D19 reproduces its zero/nonzero outcomes. Known quirk: FA40 has degenerate all-zero trailing factors, so the anchor's FA best-factor columns are NaN-driven; read its PC8/FA7 gap columns instead.
- `regularization_contract.py` is the shared C policy (AGENTS R5). It uses numpy and scikit-learn only. Other workstreams import it with `add_study_module_paths(workstreams=("W3_dr_value_regularization",))` and then `import regularization_contract as rc`. API changes go through a devlog entry with type: refactor.
  - Grid and conventions: `DEFAULT_C_GRID` is exactly the runner grid, 1/logspace(-4, 4, 17). Probes use balanced class weights, impute 0 plus a StandardScaler fitted on each (inner) fold's training cells, and an unpenalised intercept.
  - Entry rule: `l1_entry_threshold(C, n_train, l1_ratio=None)` returns 4·l1_ratio/(C·n_train), with l1_ratio = 1 for L1. The l1_ratio is in the numerator: a smaller l1_ratio makes entry easier. `l1_entry_margins(Z_train, y_train, C)` returns |standardized class-mean gap| / threshold; if every margin is <= 1, the probe is all-zero. `l1_entry_c` gives the per-feature critical C. `c_scaled_to_n(C_ref, n_ref, n_train)` keeps the threshold fixed when training n changes. `c_at_grid_lower_bound` flags C at the grid floor. `kkt_margins` and `optimal_intercept` check a stored solution that has no intercept column. `supported_patients` applies the study support rule.
  - Fitting and tuning: `fit_probe(Z_train, y_train, penalty=..., C=..., l1_ratio=...)` returns a fitted Pipeline; pass it raw Stage 1 scores. `probe_coef` and `probe_scores` read it back. `tune_c_inner_lopo(Z_train, y_train, patients_train, penalty=..., l1_ratio=...)` runs inner leave-one-patient-out over the supported training patients. The criterion is median inner lift; ties go to the sparser probe, then the smaller C. With fewer than 2 supported training patients it falls back to stratified 5-fold CV on patient × label strata (label-only strata when those are too small). `heldout_lift` returns (AP, π, AP − π). `c_policy_record` produces the manifest block.
  - Solvers: L1 and elastic net default to `ProxNewtonLogistic` ('newton_cd', an exact proximal-Newton solver of the sklearn/cuML objective, warm-started along the C path) for up to 200 features (`NEWTON_CD_MAX_FEATURES`). Above that they use saga, the runner's CPU backend. L2 uses lbfgs. All of them minimise the same objective as the stored cuML grids (devlog 2026-10-08_d19-regularization-contract records the backend check).
- `run_d19_regularization_contract.py` is the D19 runner, with parts anchor, audit, ifna, report and the optional GPU-seconds cuml_check.
  - `--resume` skips finished (representation, fold) audit units and finished IFN units. Empty, corrupt or out-of-date unit files are rerun.
  - `--out-root` redirects all outputs; use it for smoke runs. `--smoke` and `--max-cells` give a quick pass.
  - Outputs go to `analysis/{scorecards,figures,cache}/stage2_diagnostic/regularization_contract_v1/` (findings, manifest, ledger, figures D19A-D19C).
  - Command (repo root): `python scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py --parts anchor,audit,ifna,report --resume --n-jobs 12`. It fits the R1 agent-shell budget; add `cuml_check` and `CUDA_VISIBLE_DEVICES=0` only on a GPU host.
- Tests: `tests/test_regularization_contract.py` uses synthetic data. Two regression tests re-check the IFN-alpha anchor against EXP and skip when EXP is absent.
- Screen sessions: `W3_d23` (W3.3, D23: refit through `tune_c_inner_lopo`, screen-cpu with a `--resume` runner). W3.5 (patient-specific direct-gene grid, GPU) runs in user screen under a `W3_<topic>` session after a campaigns.csv row is added.

Task status (derived from the devlog; see `../../research/STATUS.md`): W3.1 and W3.2 are closed by devlog 2026-10-08_d19-regularization-contract, which also re-declares H-S2-08, retires H-S2-07 and marks H-S2-08:ALT-PEN addressed. W3.3 can start; it should refit with `tune_c_inner_lopo` rather than reuse stored per-fold metrics (they are not nested in C), report pooled AP, median lift and P20 lift together, and carry the D19 knife-edge flags. W3.8 was added from that entry's Next steps.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `regularization_contract_v1` (D19), `nested_dr_vs_direct_v1` (D23), `single_score_baselines_v1` (D24). New Stage 2 grids (W3.5) use a new `--stage2-output-branch` namespace. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5), owned here: sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train); under elastic net the threshold is 4·l1_ratio/(C·n_train). Whenever training n changes, retune C per condition by inner LOPO over the training patients with `tune_c_inner_lopo` and report L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal". Flag every L1 "method win" or "driver" whose entry margin sits within the knife-edge band set in W3.1.
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
