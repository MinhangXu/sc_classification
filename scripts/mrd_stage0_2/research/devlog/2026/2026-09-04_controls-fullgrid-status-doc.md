---
id: 2026-09-04_controls-fullgrid-status-doc
date: 2026-09-04
type: run
title: Set A control full-grid Stage 2 (full_34, core_only, HVG 500/1k/3k/10k) and the live status rehydration doc
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q2, Q3]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-gpu  # 8 GPU shards in screen old34_controls_fullgrid
c_policy: not_applicable  # full 65-setting grid; selection happens downstream
training_n_changed: false
status: done
commit: b60a6ee  # retro: --panel-ids and the status doc first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/stage2_mrd_multiobjective_scorecard_20260904_055618.log, 05:56 to 09:36 UTC
supersedes: []
---

## Question
(reconstructed from the status doc and the 2026-10-08 chronology reconstruction) Fill the 2026-06-05 gap (controls excluded) so that Layer 6 gene-budget calibration is possible, without overwriting the root scorecards. The asker is not recorded; there is no ask ID.

## Why it matters
Controls are the technical floor for Q2/Q3 statements about curated priors; without them the D1 atlas had no control rows.

## Estimand / null
Discovery, sharedness LOPO and patient-specific fits over L1/L2/EN (13-point C grid each) for the 16 DR configurations of each of 6 control panels. No selection inside the run; no null.

## Implementation
- Runner: scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py with the new --panel-ids filter and --stage2-output-branch old34_controls_fullgrid_v1.
- The user ran it in screen -S old34_controls_fullgrid (session started 2026-09-04 05:11 UTC). Command as recorded in the status doc, with paths made relative: python -u scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py --experiment-dir <EXP> --stage0-scorecard <EXP>/analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv --stage2-output-branch old34_controls_fullgrid_v1 --stage2-run-id old34_controls_fullgrid_v1 --panel-selection all_quick_rows --panel-ids full_34,core_only,hvg_top_requested_500__available_500,hvg_top_requested_1000__available_1000,hvg_top_requested_3000__available_3000,hvg_top_requested_10000__available_10000 --run-discovery-full-cohort-fit --run-sharedness-lopo --run-patient-specific --penalties l1,l2,elasticnet --c-grid-log10-min -4 --c-grid-log10-max 4 --c-grid-n 13 --class-weight balanced --backend gpu --strict-gpu --launch-gpu-shards --gpu-ids auto --seed 42 (inside conda env sc-classification-rapids-py311).
- Status doc: scripts/mrd_stage0_2/plans/current_experiment_status_rehydration_20260904.md (snapshot 2026-09-04 about 17:55 UTC; it records sessions, namespaces and do-not-relaunch rules). It is now historical.

## Artifacts
- analysis/scorecards/old34_controls_fullgrid_v1/
- stage2_supervised/multiobjective/old34_controls_fullgrid_v1/runs/old34_controls_fullgrid_v1/launcher_manifest.json
- logs/stage2_mrd_multiobjective_scorecard_20260904_055618.log

## Results
- 6 control panels x 16 DR configurations = 96 representations: 6,240 discovery rows and 6,240 LOPO rows (96 x 65) and 56,160 patient-specific rows; completed 09:36:09 UTC with failures [].
- The first launch failed on the invalid flag --save-lopo-coefficients (saving is the default).
- The snapshot recorded 3 detached screen sessions: pooled_stage0_unions_v1 (active), old34_controls_fullgrid and donor_ds_v2 (both finished).

## Claim gate
Supports (descriptive): run accounting only; controls now exist for the D1 atlas.
Does not support: control-based calibration in D2/D3, which still use only the 41 biological panels.

## Next steps
- W4.2: Null Model 1, which the fixed HVG anchors do not replace.
