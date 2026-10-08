---
id: 2026-06-05_set-a-all-biological-fullgrid
date: 2026-06-05
type: run
title: Canonical Set A Stage 2 grid over 41 biological panels, 643 representations and 65 settings
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1, Q2, Q3]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-gpu  # 8 GPU shards; launch session not recorded
c_policy: not_applicable  # full 65-setting grid; selection happens downstream
training_n_changed: false
status: done
commit: 4ada734  # retro: runner first committed under scripts/comprehensive_run in 4ada734
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/stage2_mrd_multiobjective_scorecard_20260605_064639.log, 06:46 to 10:11 UTC
supersedes: []
---

## Question
(reconstructed from the V2 section E accounting and the 2026-10-08 chronology reconstruction) Is transfer a property of the prior gene space or of one lucky method and K? That needs every method x K per panel, not one shortlisted representation each. The asker is not recorded; there is no ask ID.

## Why it matters
This is the root Set A grid that D1-D9, D14, D15 and the F-series read. Q1-Q3 numbers all trace back to it.

## Estimand / null
Discovery, sharedness LOPO and patient-specific fits for every representation over 13 L1 + 13 L2 + 39 elastic-net settings (C from 1e-4 to 1e4). No selection inside the run; no null.

## Implementation
- Runner: scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py with --panel-selection all_biological_quick_rows, 8 GPU shards; run id 20260605_stage2_all_biological_fullgrid_gpu8.
- The merged scorecards were copied to the three root Stage 2 scorecards listed in research/study_config.json protected_paths. They must never be overwritten; later grids use --stage2-output-branch namespaces (AGENTS.md R3).
- Launch context: not recorded (the equivalent today is screen -S <W#>_<topic> per AGENTS.md R1).

## Artifacts
- stage2_supervised/multiobjective/runs/20260605_stage2_all_biological_fullgrid_gpu8/
- logs/stage2_mrd_multiobjective_scorecard_20260605_064639.log

## Results
- 643 representations (640 DR + 3 small-panel direct_gene) over 41 biological panels: 41,795 discovery rows and 41,795 LOPO rows (643 x 65), 376,155 patient-specific rows, and 543,335 LOPO by-held-out-patient rows (41,795 x 13).
- Controls (full_34, core_only, HVG anchors) were excluded, which blocked the Layer 6 gene-budget calibration until 2026-09-04. Cell-level predictions were not saved.

## Claim gate
Supports (descriptive): run accounting only.
Does not support: any claim about controls (absent here) or about seeds other than 42; Stage 1 is transductive.

## Next steps
- W3.3: nested DR-vs-direct selection over this grid.
- W4.1: patient-level uncertainty from the stored per-fold tables.
