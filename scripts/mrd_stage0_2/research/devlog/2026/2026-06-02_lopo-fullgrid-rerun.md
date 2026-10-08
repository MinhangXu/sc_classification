---
id: 2026-06-02_lopo-fullgrid-rerun
date: 2026-06-02
type: run
title: LOPO rerun of the 33-representation shortlist on the full 65-setting regularization grid
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1]
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
commit: 4ada734  # retro: runner and the shortlist Fig 3A builder first committed under scripts/comprehensive_run in 4ada734
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/stage2_lopo_fullgrid_20260602_053229.log; run id dated 20260529
supersedes: []
---

## Question
(reconstructed from the run accounting document and the 2026-10-08 chronology reconstruction) The May-26 LOPO used only L2 at C=1. Does transfer hold under the same L1/L2/EN path as discovery? The asker is not recorded; there is no ask ID.

## Why it matters
Q1 transfer must be read on the same regularization path as discovery; a single untuned C confounds transfer with penalty choice (ALT-PEN).

## Estimand / null
Held-out-patient AUPRC for every one of the 65 settings x 33 representations x 13 LOPO folds. No selection inside the run; no null.

## Implementation
- Runner: scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py, run id 20260529_stage2_lopo_fullgrid_gpu8 (launched 2026-05-29). The 2026-06-02 launcher log (05:32 to 05:37 UTC) completes and merges it across 8 GPU shards.
- Interactive Fig 3A: scripts/mrd_stage0_2/stage2_supervised/stage2_mrd_fig3a_interactive_top_stage1_shortlist.py.
- Launch context: not recorded (the equivalent today is screen -S <W#>_<topic> per AGENTS.md R1).

## Artifacts
- stage2_supervised/multiobjective/runs/20260529_stage2_lopo_fullgrid_gpu8/
- logs/stage2_lopo_fullgrid_20260602_053229.log
- stage2_supervised/STAGE2_MULTIOBJECTIVE_RUN_ACCOUNTING.md
- analysis/figures/stage2_figure3_sharedness/fig3A_discovery_vs_lopo_sharedness_interactive.html

## Results
- Merge: 2,145 LOPO rows (33 representations x 65 settings), 27,885 LOPO patient rows (2,145 x 13 folds) and 621,075 LOPO coefficient rows; launcher failures [].
- The accounting document defines R=33 representations, G=65 settings and P=13 folds; canonical analysis switched to the full-grid LOPO.

## Claim gate
Supports (descriptive): run accounting only.
Does not support: biological-panel transfer claims; the 2026-06-05 grid supersedes this shortlist for the 41 biological panels.

## Next steps
- W3.3: nested selection should use the 2026-06-05 grid, not this 33-representation grid.
