---
id: W4
title: W4_evidence_strength: nulls, uncertainty, variance decomposition, robustness (D21, D22)
status: ready
questions: [Q2]
---

# W4 — W4_evidence_strength: nulls, uncertainty, variance decomposition, robustness (D21, D22)

Status as of 2026-10-08: ready (P0). Effect sizes exist but nothing inferential does. D22 is CPU-only from stored tables.

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "Q2: how strong is the evidence (variance explained, effect size, significance)?" (A-20261008-31)

> "We cannot make a causal claim without experimental intervention" (A-20261008-23)

Earlier asks, cited by ID (their recorded text is in `scripts/mrd_stage0_2/research/asks/2026-07-24_team-meeting.md`; capture reconstructed): A-20260724-01 (lift metric → D6), A-20260724-05 (donor downsampling → D10, D11).

Scope: Q2, meaning variance explained, effect size and significance (A-20261008-31). Design spec: `scripts/mrd_stage0_2/plans/stage2_figure_catalog_and_statistical_rigor_v2_jun4.md` (Null Model 1, CIs, AUROC). Set B is parked here (W4.7).

Done means: every headline effect carries a patient-level interval and an AUROC companion (D22); each panel has an empirical p against Null Model 1 with BH adjustment (D21); variance-explained statements name their estimand (W4.4); Stage 1 recurrence is measured (W4.5); and the donor-shortcut test is redone with C retuned (W4.6). Claims stay non-causal (A-20261008-23, R6).

## Builds on

- Registry rows owned here: D6, D10, D11, D1DD7, D1DD8. Planned: D21, D22.
- Used from other workstreams: the D1 atlas and D1f tables (W1) as the per-fold inputs for D22; D16 (W1) for W4.4; D24 (W3) for the Tier 1 one-score nulls; the W3 contract for W4.6.
- Hypotheses: H-S0-01, H-S0-03, H-S2-01, H-S2-04, H-I-01.
- Spec: `scripts/mrd_stage0_2/plans/stage2_figure_catalog_and_statistical_rigor_v2_jun4.md`.
- Code: `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py` and `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.sh` (D10/D11; `--resume` and `--merge-only`, see AGENTS.md R2); `scripts/dr_feature_screening/skeletons/run_gene_filter_dr_plan2_negative_controls.py` (Plan 2 negative-control skeleton, never run; W4.3).
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_meeting_followup_metric_interactions.ipynb` (D6), `stage2_old34_donor_downsampling_sensitivity.ipynb` and `stage2_old34_donor_downsampling_sensitivity_v2.ipynb` (D10, D11), `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` (D1DD7, D1DD8).
- EXP-relative inputs: `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`, `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv`, `analysis/scorecards/stage2_diagnostic/meeting_followup_v1/`, `analysis/scorecards/stage2_diagnostic/donor_sensitivity_v1/`, `analysis/scorecards/stage2_diagnostic/donor_sensitivity_v2/`, `stage2_supervised/multiobjective/sharedness_lopo/`, and Set B under `analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/` (parked; W4.7).
- Devlog entries: 2026-08-31_d6-d9-meeting-followup, 2026-08-21_donor-sensitivity-d10-d11, 2026-08-03_narrative-guide-donor-correction, 2026-07-03_figure-catalog-option-b, 2026-06-22_set-b-expanded-manuscript-axes, 2026-06-22_jaatinen-hsc-up-probe.
- Literature routing (`../W9_literature/README.md`): uncertainty and AUROC → W4.1; Null Model 1 → W4.2; multi-seed cNMF and determinacy → W4.5; C-scaled donor rerun → W4.6.

## Backlog

- [W4.1] (P0) D22 patient-level uncertainty (slug inference_uncertainty_v1) from stored per-fold tables — done when: CI tables exist for the D1 atlas and the focus panels.
  - Patient-cluster bootstrap CIs on median and P20 lift.
  - Per-patient AUPRC intervals.
  - An AUROC companion column.
  - Exact patient-level sign and permutation tests.
- [W4.2] (P0) D21 Null Model 1 — done when: there is an empirical p per panel with BH adjustment. (depends: W3.4)
  - Tier 1: one-score nulls with matched random gene sets (matched on size and expression/detection bins, plus a co-expression-matched variant) for 41 panels.
  - Tier 2: DR + Stage 2 nulls for four focus panels on branch null_model1_v1 (GPU, user screen, campaigns.csv row added first).
- [W4.3] (P1) Within-patient label-permutation negative control (from the dr_feature_screening Plan 2 skeleton) — done when: a permutation distribution of lift exists for the focus panels.
- [W4.4] (P1) Variance decomposition with explicit estimands (label-free Stage 1 variance, per-factor patient eta2, Stage 2 attribution) for the shortlist. Write the estimand note into THEME §1 at the next milestone — done when: a table per factor exists. (depends: W1.1)
- [W4.5] (P2) Stage 1 stability: seeds × patient-stratified bootstrap with component clustering, plus factor determinacy and effective_k, for the focus representations — done when: a recurrence table per factor exists.
- [W4.6] (P2) Donor-shortcut rescue: re-run D10 with condition-specific C through the donor runner (--resume; user screen) — done when: a performance-level donor test is reported. (depends: W3.1)
  - 2026-10-09: overlaps the Stage-2-only proportional-stratified arm of Arm D (W8.7; `../W8_new_cohort/healthy_downsampling_plan.md` §5). The user decided that downsampling runs wait for the new data (devlog 2026-10-09_report-reading-notes), so W4.6 folds into W8.7 unless the user chooses the plan's rehearsal on the current cohort (§12 item 1).
- [W4.7] (later) Read the Set B 294-panel scorecard, after first porting the lift figure contract (H-S0-03). Run expanded_full Stage 2 only if the user asks.

## Code in this directory

- No code yet. D22 (W4.1) and D21 (W4.2) code will land here.
- D21 Tier 1 and D22 reuse W3's `regularization_contract.heldout_lift` and the D24 single-score baselines rather than re-implementing them.
- Screen sessions (planned): `W4_d21` for Null Model 1 Tier 2 (GPU, Stage 2 branch `null_model1_v1`, campaigns.csv row first); `W4_donor_rescue` for the W4.6 donor-runner rerun with `--resume`.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `inference_uncertainty_v1` (D22), `null_model1_v1` (D21; also the Tier 2 `--stage2-output-branch` namespace). New IDs for W4.3-W4.5 come from `python tools/research_index.py reserve-id --workstream W4 --title '...'`. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Whenever training n changes (donor downsampling in W4.6, bootstrap refits in W4.5, null refits in W4.2), retune C per condition by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (wave 2) and report L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal". D22 works from stored per-fold tables and refits nothing.
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

- Screen sessions are named `<W#>_<topic>`. Long or GPU jobs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with compute: screen-cpu or screen-gpu. New runners implement `--resume` (R2). Add a planned campaigns.csv row before any new GPU campaign (R4).
- Metric and claim language (R6, R7): lift = held-out AUPRC − the held-out patient's prevalence π, with normalized lift secondary; Option B patient summaries; quote pooled AP, median lift and P20 lift together; say "constant-score no-skill AP = π"; no causal claims.
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W4.n`.
