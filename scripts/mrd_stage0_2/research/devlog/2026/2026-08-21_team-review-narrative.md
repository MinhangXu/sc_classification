---
id: 2026-08-21_team-review-narrative
date: 2026-08-21
type: docs
title: Curated team-review notebook and team-review talking-points canvas
asked_by: not_recorded
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
commit: b60a6ee  # retro: notebook first committed in b60a6ee
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # notebook 2026-08-21 02:05 UTC; Cursor talking-points canvas 18:50 UTC
supersedes: []
---

## Question
(reconstructed from the notebook and the 2026-10-08 chronology reconstruction) How should D6-D11 be presented to the PI and team so that metric, support and donor caveats come before biological claims? A Cursor transcript attributes the request to the user, but the text is not in the repo; no ask ID.

## Why it matters
Communication of Q1-Q3. It fixed a reading order that puts the AP = π contract and the support caveat before any biology.

## Estimand / null
n/a (presentation of completed artifacts).

## Implementation
- Notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_team_review_narrative.ipynb. It reads completed CSV/PNG artifacts only (no fits) and gives the equations for constant-score AP = π, raw and normalized lift, the within-representation Spearman, the direct-gene logistic model and the factor-stability metrics.
- Talking points: the Cursor canvas d6d-rho-d6e-direct-gene.canvas.tsx (outside the repo; not canonical).
- The meeting date itself is not recorded.

## Artifacts
none

## Results
- Fixed presentation order: cohort, AP = π contract, support, landscape, no-DR baseline, IFN-alpha, donor (D10E first), claim gate.
- Validation: 15 figures, 13 patients and 41 direct-gene comparisons. The notebook predates donor v2 (2026-08-21 16:30 UTC), the F-series (2026-09-17), D14 and D15 (2026-10-02); its IFN-alpha conclusion needs the F3/F6 update.

## Claim gate
Supports (descriptive): nothing new; it curates D6-D11.
Does not support: its own IFN-alpha sentence that FA's sparse model 'retains no factor' as a property of FA (see 2026-09-17_team-feedback-f1-f7).

## Next steps
- W0.8: banner or update docs that present superseded conclusions.
