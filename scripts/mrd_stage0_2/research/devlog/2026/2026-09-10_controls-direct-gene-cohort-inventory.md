---
id: 2026-09-10_controls-direct-gene-cohort-inventory
date: 2026-09-10
type: run
title: Control no-DR direct-gene LOPO run and cohort AnnData inventory
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q3]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-gpu  # Stage 2 on 8 GPU shards in screen old34_controls_direct_gene_stage2
c_policy: not_applicable  # full 65-setting grid; selection happens downstream
training_n_changed: false
status: done
commit: b60a6ee  # retro: --direct-gene-all-panels first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/old34_controls_direct_gene_v1/stage0_mrd_old34_broad_screen_20260910_215902.log; Stage 2 log 20260911_035502
supersedes: []
---

## Question
(reconstructed from the run namespace and the 2026-10-08 chronology reconstruction) Give the six D1 control rows a direct_gene column too. Which timepoints, patients and samples exist in the source AnnData (for timing covariates and other timepoints)? The asker is not recorded; there is no ask ID.

## Why it matters
Completes the no-DR comparator for the controls (Q3, H-S1-02) and documents the timing covariates later used descriptively in D1DD13.

## Estimand / null
Discovery and LOPO Stage 2 over the 65-setting grid on standardized control-panel genes. No null.

## Implementation
- Stage 0: scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py with --direct-gene-all-panels on the control panels (branch old34_controls_direct_gene_v1), 2026-09-10 21:59 to 23:09 UTC.
- Stage 2: 2026-09-11 03:55 to 09:41 UTC on 8 GPU shards, run by the user in screen -S old34_controls_direct_gene_stage2 (session name from research/campaigns.csv; the session has finished).
- Cohort inventory: Cursor canvas cohort-adata-inventory.canvas.tsx (outside the repo; not canonical).

## Artifacts
- analysis/scorecards/old34_controls_direct_gene_v1/
- logs/old34_controls_direct_gene_v1/stage0_mrd_old34_broad_screen_20260910_215902.log
- logs/stage2_mrd_multiobjective_scorecard_20260911_035502.log

## Results
- The D1 direct_gene column became 47/47 (41 biological + 6 control panels); discovery and LOPO only, no patient-specific direct-gene fits.
- Inventory: 13 named patients, 4 coarse timepoints (preSCT, MRD, Relapse, unknown), 14 fine Time labels, 53 samples.

## Claim gate
Supports (descriptive): run accounting only.
Does not support: patient-specific direct-gene comparisons (never run).

## Next steps
- W3.5: run the patient-specific direct-gene grid.
- W1.7: descriptive timing plot.
