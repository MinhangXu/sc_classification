---
id: 2026-07-24_team-meeting-six-questions
date: 2026-07-24
type: meeting
title: Team meeting raises six follow-up questions; direct-gene all-panel and donor-v1 runs launched the same day
asked_by: meeting:2026-07-24
asks: [A-20260724-01, A-20260724-02, A-20260724-03, A-20260724-04, A-20260724-05, A-20260724-06]
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
status: done
commit: b60a6ee  # retro: --direct-gene-all-panels (used by the same-day run) first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # direct-gene logs 20260724_203430 and 20260724_205233; meeting date from the narrative guide
supersedes: []
---

## Question
(reconstructed from the narrative guide's numbering of the meeting note's 'Next Steps' and from a Cursor transcript read on 2026-10-08; the asks file research/asks/2026-07-24_team-meeting.md has capture: reconstructed, and W0.7 recovers the verbatim text). Participants are not recorded. The topics, with their ask IDs:

- A-20260724-01: is prevalence the right AP baseline, or should the gray bar be a dumb all-healthy or all-malignant classifier? Leads to D6.
- A-20260724-02: add a no-DR, direct-gene regularized logistic baseline per gene set, to measure what Stage 1 adds. Leads to D9.
- A-20260724-03: factor-by-factor analysis of IFN-alpha PCA40 vs FA40: which PCA factors drive the P02/P07 advantage, and why FA misses it. Leads to D8.
- A-20260724-04: per-patient lift across gene sets and DR methods (targeted exploration, not a grid search). Leads to D7.
- A-20260724-05: donor-healthy downsampling and its effect on lift and on factors. Leads to D10/D11.
- A-20260724-06: malignant vs healthy cell counts per patient (recipient-healthy transparency). Leads to D6.

## Why it matters
These questions set the agenda for D6-D11 (Q2 metric validity and robustness, Q3 value of DR). The note also raised whether higher lift in P02/P03/P04 reflects more malignant cells or biology; those are the highest-prevalence patients.

## Estimand / null
n/a (meeting record). The estimands are defined in the D6-D11 entries.

## Implementation
- Same day: --direct-gene-all-panels was added to scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py and branch old34_direct_gene_all_v1 was run (Stage 0 20:34-20:51 UTC; discovery + LOPO Stage 2 20:52-21:42 UTC on 8 GPU shards).
- Donor-v1 fixed-basis and IFN-alpha basis-refit caches were written to analysis/cache/stage2_diagnostic/donor_sensitivity_v1/ (21:56-21:58 UTC).
- A ChatGPT illustration prompt for lift: scripts/mrd_stage0_2/notebooks/stage0_2/old34_heldout_lift_illustration_prompt.md.

## Artifacts
- analysis/scorecards/old34_direct_gene_all_v1/
- logs/old34_direct_gene_all_v1/stage0_mrd_old34_broad_screen_20260724_203430.log
- logs/stage2_mrd_multiobjective_scorecard_20260724_205233.log

## Results
- Direct-gene branch: 41 panels, 2,665 discovery rows (41 x 65 settings); the patient-specific direct-gene grid was never run.
- Meeting-priority panels (the donor manifest targets): stress arrest, p53 and IFN-alpha.
- The note expected a larger cohort, about double the current patient count, within weeks; on 2026-10-08 the data had not been requested (W8.1).

## Claim gate
Supports (descriptive): nothing; this records questions and same-day runs.
Does not support: answers to any of the six questions here; see the D6-D11 entries.

## Next steps
- W0.7: recover the verbatim 2026-07-24 question set into research/asks/.
- W3.5: run the patient-specific direct-gene grid that was never run.
- W8.1: the larger cohort (email Yu-Hsiang).
