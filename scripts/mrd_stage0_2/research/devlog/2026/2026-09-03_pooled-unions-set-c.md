---
id: 2026-09-03_pooled-unions-set-c
date: 2026-09-03
type: run
title: Set C pooled old34 plus expanded unions and size-matched HVG; Stage 2 never merged
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q3]
workstream: W0
analyses: []
hypotheses: [H-S0-04]
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-gpu  # Stage 2 on 8 GPU shards in screen pooled_stage0_unions_v1
c_policy: not_applicable  # all_quick_rows full grid; selection happens downstream
training_n_changed: false
status: partial
commit: b60a6ee  # retro: bundle builder, wrapper, config and --full-control-panel-id first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/pooled_stage0_unions_v1/pooled_stage0_unions_20260903_054350.log and _164848.log
supersedes: []
---

## Question
(reconstructed from the run configuration and the 2026-10-08 chronology reconstruction) Do pooled knowledge bags without per-geneset identity still carry shared malignancy, and is a size-matched HVG bag of 11,421 genes the fair bag-level control? The asker is not recorded; there is no ask ID.

## Why it matters
Stage 0 controls for Q3 (H-S0-04). Until the run is merged or abandoned (W0.6), no Set C claim is possible and no new 8-GPU job should start (AGENTS.md R4).

## Estimand / null
Quick L2 triage, then discovery / LOPO / patient-specific Stage 2 over the full grid for 68 representations of 4 bags. No null.

## Implementation
- Bundle: scripts/mrd_stage0_2/stage0_panels/build_pooled_stage0_union_bundle.py with config scripts/mrd_stage0_2/stage0_panels/configs/pooled_stage0_unions.yaml; bundle files under scripts/mrd_stage0_2/stage0_panels/pooled_stage0_unions/.
- Wrapper: scripts/mrd_stage0_2/stage0_panels/run_pooled_stage0_unions_stage0_to_stage2.sh (Stage 0 runner with --full-control-panel-id and --direct-gene-all-panels; Stage 2 all_quick_rows on 8 GPU shards).
- The first launch (05:43 UTC) ran in a Cursor agent shell and died at 62/64 DR fits when the remote host shut down. The user resumed it in screen -S pooled_stage0_unions_v1 at 16:48 UTC with --skip-bundle-build and without --rerun-stage0. Parent command per the 2026-09-04 status doc: bash scripts/mrd_stage0_2/stage0_panels/run_pooled_stage0_unions_stage0_to_stage2.sh --experiment-dir <EXP> --branch-name pooled_stage0_unions_v1 --run-stage2 --gpu-ids auto --skip-bundle-build.
- After the killed launch, the umbrella repo gained the always-apply Cursor rule .cursor/rules/long-runs-in-screen.mdc (mtime 2026-09-03 18:52 UTC).
- No safe resume exists: the wrapper has no --resume and the Stage 2 runner has no shard resume (AGENTS.md R2; campaigns.csv resume_command none).

## Artifacts
- analysis/reports/pooled_stage0_unions_v1/overlap_summary.json
- analysis/scorecards/pooled_stage0_unions_v1/
- stage2_supervised/multiobjective/pooled_stage0_unions_v1/runs/pooled_stage0_unions_v1_stage2/launcher_logs/
- logs/pooled_stage0_unions_v1/

## Results
- Gene counts: old34 covered 2,719, expanded covered 11,358, overlap 2,656, old34-only 63, pooled 11,421. Stage 0/1 plus quick L2: 68 representations over 4 panels, all ok (finished 19:53 UTC).
- [H-S0-04 qualifies] Stage 2 never merged: shards 001 and 004-007 completed on 2026-09-03 (20:10-22:58 UTC); the logs of shards 000, 002 and 003 stop on 2026-09-05 between 22:47 and 23:42 UTC. On 2026-10-08 there is no launcher manifest, no merged scorecard, and no live process or screen session.

## Claim gate
Supports (descriptive): Stage 0/1 bag construction and quick-L2 triage only.
Does not support: any Set C Stage 2 claim (nothing merged); the 2026-09-04 status doc and the 2026-09-08 tracker call this run 'in flight', which is stale.

## Next steps
- W0.6: decide relaunch of shards 000/002/003 under the same run id plus --merge-stage2-run, or abandon; record a decision entry, update campaigns.csv and declare the H-S0-04 verdict. Do not relaunch before reading the runner code (AGENTS.md R2).
