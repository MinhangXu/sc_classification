---
id: 2026-08-03_narrative-guide-donor-correction
date: 2026-08-03
type: fix
title: Old34 inquiry narrative and correction of the donor-factor stability claim
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q2]
workstream: W4
analyses: [D11]
hypotheses: [H-S2-04]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
status: done
commit: b60a6ee  # retro: narrative guide and donor notebook first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # Cursor canvas old34-inquiry-narrative mtime 2026-08-03 16:05 UTC
supersedes: []
---

## Question
(reconstructed from the narrative guide's 'Important correction to the previous summary' and the 2026-10-08 chronology reconstruction) An earlier summary claimed that both IFN-alpha driver factors stay at about 0.99 at 25% donor retention (Jaccard 0.818). Is that right? The asker is not recorded; the 2026-08-03 numbered question set lives only in a Cursor transcript (W0.7). No ask ID.

## Why it matters
Q2 robustness of the IFN-alpha driver factors under donor downsampling (D11). The first report was not a devlog entry, so this entry is the correction of record.

## Estimand / null
Per method, score correlation, loading cosine and top-10 gene Jaccard between the full-donor and downsampled refit, computed only for factors with a nonzero full-donor Stage 2 coefficient (the supervised factors). No null.

## Implementation
- Recomputed in scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_donor_downsampling_sensitivity.ipynb; the narrative is scripts/mrd_stage0_2/notebooks/stage0_2/old34_inquiry_narrative_guide.md (now 'Updated: 2026-10-02').
- Error found: the first donor report took the median over five 'top' factors per method, four of which had zero Stage 2 coefficients.
- The corrected numbers are restated in analysis/scorecards/stage2_diagnostic/donor_sensitivity_v1/donor_sensitivity_findings.md (written 2026-08-21).

## Artifacts
none

## Results
- [H-S2-04 qualifies] Corrected, supervised factors only, at 25% donor retention: FA40 factor 007 has score r 0.952, loading cosine 0.943 and top-10 Jaccard 0.818; PCA40 factor 008 has 0.854, 0.854 and 0.333. Both are about 0.98-0.99 at 50%.
- Do not quote the old sentence 'both driver factors stay at about 0.99 at 25%'.

## Claim gate
Supports (descriptive): FA40 factor 007 is relatively stable at 25% donor retention; PCA40 factor 008 is not comparably stable until 50%.
Does not support: donor-vs-recipient invariance; the stability of non-supervised factors; an inductive test (the refit stays transductive).

## Next steps
- W4.6: donor-shortcut rescue with condition-specific C.
