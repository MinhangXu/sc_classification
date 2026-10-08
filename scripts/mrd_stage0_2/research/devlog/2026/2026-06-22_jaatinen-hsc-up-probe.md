---
id: 2026-06-22_jaatinen-hsc-up-probe
date: 2026-06-22
type: analysis
title: Jaatinen HSC-up single-panel strict LOPO case study (the only Set B analysis)
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1]
workstream: W4
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: global_selected  # sklearn default L2 (C=1, balanced) in every fold; no tuning
training_n_changed: false
status: done
commit: dbde125  # retro: notebook first committed in dbde125 (repo-level notebooks/)
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # notebook filename 20260622; outputs 2026-06-22 16:25 UTC
supersedes: []
---

## Question
(reconstructed from the notebook's working_interpretation table) Does the HSC-up stemness prior carry a shared malignant MRD axis beyond cell-type composition? The asker is not recorded; there is no ask ID.

## Why it matters
A single-panel test of the cell-type composition alternative (ALT-CT) for Q1, and the only Set B panel read so far.

## Estimand / null
Patient-equal held-out AUPRC under strict LOPO (one model per held-out patient), on the panel's Stage 1 factors. Comparators: an annotation-only model, annotation-residualized factors, and HSC/MPP(/LMPP)-restricted fits. No null.

## Implementation
- Notebook: notebooks/jaatinen_hsc_up_stage2_probe_20260622.ipynb (repo-level notebooks/, outside scripts/mrd_stage0_2).
- Panel atomic_hspc_lsc_stemness_hsc_mpp_lsc_jaatinen_hematopoietic_stem_cell_up from branch expanded_stage0_mrd_manuscript_axes_v1; logistic regression with class_weight balanced at the sklearn default C.
- Ran in a Jupyter kernel on CPU.

## Artifacts
- analysis/notebook_outputs/jaatinen_hsc_up_stage2_probe_20260622/
- analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/jaatinen_hsc_up_quick_stage2_context.md

## Results
- Patient-equal held-out AUPRC over 9 metric-valid patients (strict_lopo_summary.csv): annotation-only 0.153; all-cell factors 0.203 (AUROC 0.846); annotation-residualized factors 0.086; HSC/MPP/LMPP-only factors 0.216; HSC/MPP-only factors 0.283.

## Claim gate
Supports (descriptive): on this one panel, factor transfer only modestly exceeds an annotation-only model and drops after residualizing on predicted cell type.
Does not support: a shared HSC-up malignant axis independent of composition; any statement about other Set B panels.

## Next steps
- W5.3: factor x predicted cell type associations for the shortlist (ALT-CT).
- W4.7: the Set B scorecard reading.
