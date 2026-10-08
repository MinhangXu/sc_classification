---
id: 2026-06-22_set-b-expanded-manuscript-axes
date: 2026-06-22
type: run
title: Set B expanded manuscript-axes bundle built and run (632 panels; 294-panel Stage 2)
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q3]
workstream: W4
analyses: []
hypotheses: [H-S0-03]
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-gpu  # Stage 0/1 over 3 days plus GPU Stage 2; launch session not recorded
c_policy: not_applicable  # quick L2 triage plus the Stage 2 grid; selection happens downstream
training_n_changed: false
status: done
commit: a2bc955  # retro: bundle builder and tests first committed in a2bc955; wrapper and config in 4ada734
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/expanded_stage0_mrd_manuscript_axes_v1/expanded_stage0_to_stage2_20260622_*.log
supersedes: []
---

## Question
(reconstructed from the 2026-05-15 value-added plan and the 2026-10-08 chronology reconstruction) Which bottom-up dictionaries (atomic sets, family unions, leave-one-family-out) add malignancy signal beyond old34? The asker is not recorded; there is no ask ID.

## Why it matters
Stage 0 value for Q3 (H-S0-03). The compute exists, but the scientific reading is still missing; W4.7 owns it.

## Estimand / null
Quick L2 triage (cell-weighted AUPRC) for 632 panels, then the multi-objective Stage 2 grid for a shortlist plus controls. No null.

## Implementation
- Bundle: scripts/knowledge_driven_embedding/expanded_stage0_genesets/build_expanded_stage0_bundle.py (MSigDB 2026.1.Hs, 606 sets over 12 biological axes), tested by tests/test_expanded_stage0_bundle.py.
- Wrapper: scripts/mrd_stage0_2/stage0_panels/run_expanded_stage0_genesets_stage0_to_stage2.sh with config scripts/mrd_stage0_2/stage0_panels/configs/expanded_stage0_mrd_manuscript_axes.yaml and --branch-name expanded_stage0_mrd_manuscript_axes_v1; Stage 2 panel selection shortlist_plus_controls.
- Launch context: not recorded (the equivalent today is screen -S <W#>_<topic> per AGENTS.md R1).

## Artifacts
- analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/
- analysis/reports/expanded_stage0_mrd_manuscript_axes_v1/postrun_human_review.md
- logs/expanded_stage0_mrd_manuscript_axes_v1/

## Results
- Stage 0/1: 10,487 rows over 632 panels, 8,018 ok (2026-06-22 04:35 to 2026-06-25 15:03 UTC). Stage 2: 294 panels, 24,990 discovery rows and 224,910 patient-specific rows (finished 2026-06-25 16:44 UTC). No cell predictions; expanded_full was not selected for Stage 2.
- Top quick rows (cell-weighted triage): family_union_hspc_lsc_stemness PCA K=40 (AUPRC 0.657, AUROC 0.927); cd54_cd244_adhesion_niche FA K=40 (AUPRC 0.620).
- [H-S0-03 qualifies] 294 panels have Stage 2 scorecards, but as of 2026-10-08 no scorecard-driven analysis of them exists (only the single-panel 2026-06-22 Jaatinen probe).

## Claim gate
Supports (descriptive): run accounting only.
Does not support: any claim that Set B dictionaries outperform old34 (quick AUPRC is cell-weighted triage; no LOPO-lift reading, no null).

## Next steps
- W4.7: port the lift figure contract to Set B and read the 294-panel scorecards; run expanded_full Stage 2 only if the user asks.
