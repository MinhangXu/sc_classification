---
id: 2026-05-28_figure3-sharedness-v1
date: 2026-05-28
type: analysis
title: First Figure 3 sharedness suite (Fig 3A-3D)
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1]
workstream: W1
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: not_applicable  # reads stored May-26 and Jun-2 Stage 2 outputs; no refit
training_n_changed: false
status: superseded
commit: dbde125  # retro: notebook first committed in dbde125; stage2_sharedness_plotting.py in 4ada734
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # notebook filename 20260528; outputs regenerated 2026-06-05 05:11-05:15 UTC
supersedes: []
---

## Question
(reconstructed from the notebook and the 2026-10-08 chronology reconstruction) Which Stage 0/1 representations transfer across patients, and which look patient-specific? The asker is not recorded; there is no ask ID.

## Why it matters
The first Q1 figure set. Its failure mode (raw AUPRC reproduces prevalence) motivated lift = AUPRC − prevalence, the study's effect size.

## Estimand / null
Fig 3A discovery vs LOPO AUPRC; Fig 3B per-patient LOPO AUPRC heatmap; Fig 3D patient-specific factor-usage heatmaps. Raw AUPRC; no prevalence baseline subtracted; no null.

## Implementation
- Notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_20260528.ipynb.
- Plotting module: scripts/mrd_stage0_2/stage2_supervised/stage2_sharedness_plotting.py.
- Ran in a Jupyter kernel on CPU over the stored multi-objective outputs; outputs were regenerated on 2026-06-05.

## Artifacts
- analysis/scorecards/stage2_figure3_sharedness/
- analysis/figures/stage2_figure3_sharedness/

## Results
- Produced Fig 3A, 3B and 3D. The 2026-06-04 V2 suite replaced them, and the D1-D11 chain later replaced V2 as claim plots.
- The raw-AUPRC heatmaps reproduced the per-patient malignant prevalence table, which motivated the switch to lift (AUPRC − prevalence).

## Claim gate
Supports (descriptive): nothing current; superseded figures.
Does not support: raw-AUPRC heatmaps as transfer evidence (they track prevalence).

## Next steps
- W4.1: report lift with patient-level uncertainty instead of raw AUPRC.
