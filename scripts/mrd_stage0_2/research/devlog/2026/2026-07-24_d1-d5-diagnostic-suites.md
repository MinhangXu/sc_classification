---
id: 2026-07-24_d1-d5-diagnostic-suites
date: 2026-07-24
type: analysis
title: D1-D3 comprehensive diagnostic suite and D4-D5 interpretability and factor-to-gene grounding
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1, QI]
workstream: W1
analyses: [D1, D1c, D1d, D2, D3, D4, D5]
hypotheses: [H-S0-01, H-S1-03, H-I-03]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: global_selected  # one pooled-LOPO-selected regularization per representation
training_n_changed: false
status: done
commit: b60a6ee  # retro: both notebooks first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # layer4_5 artifacts 2026-07-24 13:42 UTC; D1d stress-arrest example 14:57 UTC
supersedes: []
---

## Question
(reconstructed from dossier §11.2 and the 2026-10-08 chronology reconstruction) Build Layers 1-5 of the diagnostic plan from the 643-representation grid: a representation atlas, a three-way shared/patient-specific taxonomy, regularization pathology, and factor-to-gene grounding. The asker is not recorded; there is no ask ID.

## Why it matters
Q1 (which programs transfer, and for whom) and QI (what the selected factors are). D1 is the atlas every later analysis indexes; D3 is the first shared vs patient-specific triage; D4-D5 set the interpretability gates (H-I-03).

## Estimand / null
- D1: per representation, median and P20 over the 7 supported patients of lift = held-out AUPRC − prevalence, at one global regularization chosen by pooled-cell LOPO AUPRC (not nested).
- D2: per panel, share of representations with P20 > 0 vs best observed P20.
- D3: P20 lift of the pooled-selected representation vs the paired gap (apparent within-patient AUPRC minus the same patient's LOPO AUPRC, same representation), with fixed triage thresholds.
- D4: dense (nonzero fraction > 0.80), sparse and unstable flags over the 65-setting paths; LOPO coefficient stability.
- D5: top positive and negative loading genes for up to 3 selected factors per target.
- No null; descriptive triage.

## Implementation
- Notebooks: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_comprehensive_diagnostic_suite.ipynb (D1-D3) and scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_layer4_5_interpretability_suite.ipynb (D4-D5).
- Inputs: the 2026-06-05 grid, the V2 decision table and v2_lopo_metric_comparison_by_panel_method_k.csv (old34_diagnostic_manifest.json).
- Ran in Jupyter kernels on CPU. The D1-D3 notebook was re-run later: the current D1-D3 tables were regenerated on 2026-09-11 (see 2026-09-11_d1-atlas-783-d1e-d1f) and the notebook mtime is 2026-10-02.

## Artifacts
- analysis/scorecards/stage2_diagnostic/old34_representation_diagnostic_table.csv
- analysis/scorecards/stage2_diagnostic/old34_panel_diagnostic_table.csv
- analysis/scorecards/stage2_diagnostic/old34_provisional_taxonomy_definitions.csv
- analysis/scorecards/stage2_diagnostic/old34_diagnostic_manifest.json
- analysis/scorecards/stage2_diagnostic/layer4_5/
- analysis/figures/stage2_diagnostic/layer4_5/
- analysis/cache/stage2_diagnostic/layer4_5/
- analysis/figures/stage2_diagnostic/D1d_concrete_patient_lift_stress_arrest_fa_K40.png

## Results
- D1 covers 643 representations x 7 supported patients (4,501 representation-patient rows, old34_diagnostic_manifest.json).
- [H-S0-01 supports] D3 provisional taxonomy over 41 biological panels (current 2026-09-11 tables): shared_plus_patient_specific 17, truly_shared 13, null_or_unresolved 7, patient_only 2, cohort_illusion 1, hidden_transfer 1.
- [H-S1-03 supports] D3 compares, within one Stage 1 coordinate system, each patient's apparent within-patient AUPRC with the same patient's LOPO AUPRC on the same representation; the high-opportunity gap threshold is the median gap, 0.381.
- D4 flags over 643 representations: dense 368, sparse 174, unstable 101. In 246 dense discovery maxima a near-best sparse alternative costs a median of 0.0120 AUPRC.
- [H-I-03 supports] D5 grounded 45 factors (843 factor-to-gene rows, 135 patient-overlap rows) as sign-oriented top-10 positive and negative gene lists. Factor IDs are comparable only within one representation, and the interferon signaling, hypoxia, IFN-gamma signaling and p53 top factors are selected in 0 or 1 of 13 folds.

## Claim gate
Supports (descriptive): heterogeneous transfer across 41 programs on a transductive basis, with a provisional triage of shared vs patient-specific panels and readable sign-oriented loadings for selected factors.
Does not support: inferential sharedness (no null, no CI, same-fold selection); patient-specific biology (the patient-specific fits are apparent, with no within-patient CV); naming L1-selected factor indices as unique programs.

## Next steps
- W1.6: within-patient CV to replace the apparent D3 patient-specific fits.
- W4.1: patient-level uncertainty for D1.
- W5.1: generalized factor cards (D18) on top of D5.
