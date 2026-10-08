---
id: 2026-06-23_lopo-transfer-audit-tables
date: 2026-06-23
type: analysis
title: Patient-wise LOPO transfer audit tables (Layers 1-2)
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1]
workstream: W1
analyses: [V2]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: agent-cpu  # script run; launch context not recorded
c_policy: not_applicable  # summarizes all 65 settings; no selection
training_n_changed: false
status: done
commit: 4ada734  # retro: script first committed under scripts/comprehensive_run in 4ada734
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # outputs 2026-06-23 18:45-18:47 UTC
supersedes: []
---

## Question
(reconstructed from the script and the 2026-10-08 chronology reconstruction) Per regularization setting, how does held-out lift (AUPRC − prevalence) distribute across patients, and is apparent sharedness carried by one patient? The asker is not recorded; there is no ask ID.

## Why it matters
Layers 1-2 of the diagnostic plan for Q1; these tables feed the D1-D3 representation selection.

## Estimand / null
Per representation and setting: the distribution of held-out lift over patients, read from the 2026-06-05 LOPO by-held-out-patient tables. No refit, no null.

## Implementation
- Script: scripts/mrd_stage0_2/stage2_supervised/build_lopo_transfer_audit_tables.py over the 2026-06-05 LOPO by-patient tables.

## Artifacts
- analysis/scorecards/stage2_figure3_sharedness_v2_jun4/v2_lopo_patient_lift_summary_by_regularization.csv
- analysis/scorecards/stage2_figure3_sharedness_v2_jun4/v2_lopo_metric_comparison_by_panel_method_k.csv
- analysis/figures/stage2_figure3_sharedness_v2_jun4/diagnostic_layer1_layer2/

## Results
- v2_lopo_metric_comparison_by_panel_method_k.csv (written 2026-06-23 18:46 UTC) is the pooled-selected-representation input of D1-D3 (old34_diagnostic_manifest.json, created 2026-09-11).

## Claim gate
Supports (descriptive): per-setting transfer summaries as an input table.
Does not support: any selection-free statement; downstream selection uses the same folds (ALT-SEL).

## Next steps
- W3.3: nested selection instead of pooled in-sample selection.
