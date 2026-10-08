---
id: 2026-07-23_last-commit-rehydration-canvas
date: 2026-07-23
type: docs
title: Commit 43f19be locks the Jul 3 catalog; rehydration canvas and V2 rerun
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [meta, Q2]
workstream: W0
analyses: [V2]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: 43f19be  # the commit itself (2026-07-23 23:49 UTC)
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: git  # 43f19be committed 2026-07-23 23:49 UTC; Cursor canvas mtime 19:05 UTC
supersedes: []
---

## Question
(reconstructed from the Cursor rehydration canvas, outside the repo) Rehydrate project state after a gap: what is committed, what is supported, what is risky? The asker is not recorded; there is no ask ID.

## Why it matters
Provenance (meta) and Q2. 43f19be was the last commit for 2.5 months: everything dated 2026-07-24 to 2026-10-02 ran on a dirty tree with HEAD 43f19be and was first committed in b60a6ee on 2026-10-08.

## Estimand / null
n/a (docs).

## Implementation
- Commit 43f19be: 'analysis(mrd): lock Jul 3 figure catalog and patient-summary metrics'.
- Cursor canvas sc-classification-rehydration.canvas.tsx (outside the repo; not canonical): completion matrix, risks, timeline.
- The V2 suite (scripts/mrd_stage0_2/notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb) was re-executed; its decision table was rewritten at 2026-07-24 00:31 UTC.

## Artifacts
- analysis/scorecards/stage2_figure3_sharedness_v2_jun4/v2_decision_table_discovery_lopo_patient_specific.csv

## Results
- 43f19be (2026-07-23) is the last commit before b60a6ee (2026-10-08); all work from 2026-07-24 on was uncommitted until then.
- The canvas risk register flagged: no null model, no uncertainty, controls missing from the grid, a single Stage 1 seed, and an artifact tree of about 112 GB that is not portable.

## Claim gate
Supports (descriptive): nothing; provenance only.
Does not support: treating post-2026-07-23 evidence as clean-tree provenance (dirty_run is true for all of it).

## Next steps
- W0.10: re-run publication-bound analyses from a clean committed tree.
- W0.3: push the backlog commit and the overlay.
