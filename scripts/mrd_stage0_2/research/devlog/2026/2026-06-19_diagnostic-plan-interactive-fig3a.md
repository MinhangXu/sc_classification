---
id: 2026-06-19_diagnostic-plan-interactive-fig3a
date: 2026-06-19
type: docs
title: Seven-layer comprehensive diagnostic plan and interactive Fig 3A builder
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1, Q2]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: 91c296b  # retro: plan first committed in 91c296b; Fig 3A builder in 4ada734
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # 'Date: 2026-06-19' in the plan; file mtime 2026-07-02 is the path sweep
supersedes: []
---

## Question
(reconstructed from the plan) Before building a new global shortlist, which systematic diagnostics are needed: Stage 1 stability, patient transfer, taxonomy, regularization pathology, factor-to-gene grounding, controls and theme synthesis? The asker is not recorded; there is no ask ID.

## Why it matters
The plan defines Layers 1-7, which became D1-D5 (Layers 1-5), the control grid (Layer 6) and the still-missing theme synthesis (Layer 7). It treats Stage 2 as a biological probe rather than a deployment classifier.

## Estimand / null
n/a (plan). It states that the V2 103-representation shortlist is not final evidence.

## Implementation
- Plan: scripts/mrd_stage0_2/plans/stage2_comprehensive_diagnostic_plan.md.
- Standalone Plotly Fig 3A builder with regularization-path drill-downs: scripts/mrd_stage0_2/stage2_supervised/stage2_mrd_fig3a_interactive.py. Its smoke output (fig3A_interactive_all_biological_smoke.html under analysis/figures/stage2_figure3_sharedness_v2_jun4/) is dated 2026-06-05 14:30 UTC, before the plan date.

## Artifacts
none

## Results
- Defined 7 diagnostic layers. Layers 1-5 were implemented on 2026-07-24 (D1-D5); Layer 6 was blocked until the 2026-09-04 control grid; Layer 7 is still incomplete on 2026-10-08.

## Claim gate
Supports (descriptive): nothing; this is a plan.
Does not support: the V2 103-representation shortlist as final evidence.

## Next steps
- W4.4: the variance-decomposition estimands that Layer 7 needs.
