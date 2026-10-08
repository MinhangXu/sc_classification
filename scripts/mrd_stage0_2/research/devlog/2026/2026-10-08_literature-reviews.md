---
id: 2026-10-08_literature-reviews
date: 2026-10-08
type: docs
title: Two literature reviews homed in W9 (factor interpretability; single-cell transfer and baselines) and routed to tasks
asked_by: user
asks: [A-20261008-22, A-20261008-28]
implemented_by: claude-code
questions: [Q3, QI, Q2]
workstream: W9
analyses: []
hypotheses: [H-S2-08, H-I-03]
verdicts: []
alts: []
closes: [W9.1]
advances: []
claim_level: descriptive
status: done
commit: pending
run_head: b60a6ee
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
(reconstructed from the user's 2026-10-08 report; item texts in research/asks/2026-10-08_team-report.md)
- A-20261008-22: review interpretability frameworks from other fields, for example factor rotation in psychology, and iterate with domain knowledge.
- A-20261008-28: review the single-cell ML literature for existing frameworks.

## Why it matters
Q3 and QI need the established tools for rotation, pattern vs structure, factor determinacy and sparse-probe stability before any factor is named; Q2 and the grant need the field's standards for cross-patient transfer, baselines, nulls, learning curves and external validation. W9.1 routes every recommended action to a W#.n task so the reviews change the backlog, not just the reading list.

## Estimand / null
n/a (literature). The reviews also ran read-only checks on stored IFN-alpha K=40 scores and loadings during the review; those numbers are scratch computations, not repo artifacts, and are reported below as such.

## Implementation
- Review A, factor interpretability (psychometrics, multi-group factor analysis, sparse-probe interpretation, attribution vs causation): scripts/mrd_stage0_2/workstreams/W9_literature/2026-10-08_factor_interpretability_review.md; 72 sources, 15 recommended actions.
- Review B, single-cell ML transfer, baselines, nulls and learning curves: scripts/mrd_stage0_2/workstreams/W9_literature/2026-10-08_scml_transfer_baselines_review.md; 87 sources, 17 recommended actions. It reads Chen et al. 2025 (the cohort's own preprint) in full.
- Routing table: scripts/mrd_stage0_2/workstreams/W9_literature/README.md. Label remap for review B: its 'D16 genotype/label-assay audit' is D25 (W1.4); its 'D15b' cell-count rerun with C retuned is D15 v2 (W6.2); its 'D15c' patient-drop curve is D15b (W6.1); its 'D15d' 2D design is W6.3.
- Citations were checked on 2026-10-08 (review A: DOIs resolved against Crossref; review B: web search of title, venue and DOI).
- Erratum: review B's GSVA link drops the '10.' DOI prefix and does not resolve; the corrected DOI 10.1186/1471-2105-14-7 resolves (checked 2026-10-08). The erratum is recorded in review B's header; its body is unchanged. Review A's body is verbatim except two absolute local paths rewritten as repo-relative paths.

## Artifacts
none

## Results
- [H-S2-08 qualifies] Review A knife-edge check (L1 refit on stored IFN-alpha K=40 scores; scratch): at C=1e-4 the balanced-L1 entry threshold 4/(n·C) is 0.682 SD in the P02 fold (n_train 58,656) and 0.690 SD in the P07 fold (57,933). FA7's entry gradient is 3.4% (P02) and 1.9% (P07) below 1/C, while PC8's is 8.5% and 7.3% above. At C=1.05e-4 FA40-L1 selects FA7 and reaches 18.06 pp in P02 (PCA40-L1 15.94 pp) and 6.97 pp in P07 (PCA40-L1 11.08 pp): the P02 gap is a regularization threshold, while the P07 gap is partly a real PCA advantage that must be shown under tuned C.
- [H-I-03 qualifies] Review A: factosig_promax loadings.npy is the pattern matrix (cov(gene, score) = L·Φ with relative residual 2.4e-4). For IFN-alpha promax K40, pattern and structure top-10 genes overlap with median Jaccard 0.18 (min 0.05), and 136 of 780 score pairs have |r| > 0.5 (max 0.924), so a single promax factor should not be named as a driver.
- [H-I-03 qualifies] Review A: IFN-alpha FA40 has effective_k 36 (factors 37-40 are all zero), and FA7 and FA8 have factor-score determinacy 0.649 and 0.618, below 0.707. PC8 is 81% explained by FA7 + FA8 (R² 0.806) and congruent with neither (Tucker φ −0.691 and +0.577).
- 32 recommended actions (15 in review A, 17 in review B) are each routed to at least one W#.n task in the W9 routing table, which also records the review-label remap.

## Claim gate
Supports (descriptive): the reviews' reading that the IFN-alpha PCA40-vs-FA40 gap at C=1e-4 is a knife-edge L1 threshold effect, that promax loadings are pattern matrices, and that FA beyond effective_k over-extracts.
Does not support: the scratch numbers as repo evidence until W3.1/W3.2 (entry-margin audit, C-matched comparison) and W5.1/W5.4 (determinacy, effective_k, pattern and structure in factor cards) reproduce them; any literature claim not checked against its source.

## Next steps
- W3.1: D19 L1 entry-margin audit (reproduce the knife-edge outcomes).
- W3.2: C-matched IFN-alpha PCA40 vs FA40, including C=1.05e-4.
- W5.4: frameworks note and the upstream factosig issue (normalize factor variances, save Φ, record loading semantics).
- W1.4: D25 genotype/label-provenance audit (review B).
- W9.2: keep THEME §12 current at milestones.
