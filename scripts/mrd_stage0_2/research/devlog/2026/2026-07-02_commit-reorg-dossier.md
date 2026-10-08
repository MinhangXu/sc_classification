---
id: 2026-07-02_commit-reorg-dossier
date: 2026-07-02
type: docs
title: Stage 0-2 code committed, repo split into mrd_stage0_2 and dr_feature_screening, rehydration dossier
asked_by: user
asks: []
implemented_by: unknown
questions: [meta]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: db694a3  # retro: first commit holding the dossier, the reorganization plan and the study README at their mrd_stage0_2 paths (dossier first added in 91c296b; reference sweep cc00b4c)
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: git  # 12 commits on 2026-07-02 (7b90649 to cc00b4c); dossier 'Date compiled: 2026-07-02'
supersedes: []
---

## Question
> "the new run wasn't analyzed"

Source: the user's suspicion as quoted in scripts/mrd_stage0_2/plans/experiment_20260525_stage0_2_rehydration_dossier.md §0 (the original prompt was not saved; no ask ID). The provenance gap behind it: the run's code was uncommitted, and the experiment config stamp b381cd5 predates that code.

## Why it matters
Provenance (meta). Before this day none of the Stage 0-2 runners, plans or notebooks were in git; afterwards every pre-July artifact can cite a commit (4ada734, 91c296b, dbde125, a2bc955) even though the runs record b381cd5.

## Estimand / null
n/a (docs and provenance).

## Implementation
- Commits on 2026-07-02, oldest first: 7b90649 (gitignore), ae7c78f (env spec), 4ada734 (Stage 0-2 runners and plotting tools), 78eca56, 91c296b (plans and the rehydration dossier), dbde125 (notebooks), a2bc955 (expanded bundle, builder, tests), 6812c4e, c0146fd, 9210eaa (dossier provenance), db694a3 (split of scripts/comprehensive_run into mrd_stage0_2 and dr_feature_screening), cc00b4c (reference sweep).
- Documents: scripts/mrd_stage0_2/plans/experiment_20260525_stage0_2_rehydration_dossier.md, scripts/mrd_stage0_2/plans/comprehensive_run_reorganization_plan.md, scripts/mrd_stage0_2/README.md.
- The dossier is now historical (Layers 3-5 and D6-D15 post-date it).

## Artifacts
none

## Results
- 12 commits on 2026-07-02 put the Stage 0-2 code under git; the dossier maps runners to artifacts to questions.
- The dossier confirmed that Set B had no scorecard-driven analysis, that the 7-layer diagnostic plan was about 25% built (Layers 3-7 missing), and that the factor-to-gene grounding table was header-only.
- Agreed figure-first work order: catalog old34, then Layers 3-5, then Set B, then controls.

## Claim gate
Supports (descriptive): nothing; provenance and planning only.
Does not support: reproducing pre-July runs from the recorded config hash b381cd5 (the producing code is in 4ada734 and later).

## Next steps
- W0.10: clear dirty-run provenance before any publication verdict.
- W4.7: the Set B reading the dossier flagged.
