---
id: 2026-09-17_team-feedback-f1-f7
date: 2026-09-17
type: analysis
title: Team-feedback deep dives F1-F7 (K trends, DR vs direct programs, IFN-alpha PCA vs FA, the collinearity claim)
asked_by: meeting:2026-09-17
asks: [A-20260917-01, A-20260917-02, A-20260917-03]
implemented_by: unknown
questions: [Q3, QI]
workstream: W5
analyses: [F1, F2, F3, F4, F5, F6, F7]
hypotheses: [H-S1-01, H-S1-02, H-S2-08, H-S2-07, H-S2-05]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: fixed_plus_l2  # stored L1 at the global C=1e-4, plus local L1/L2/EN refits at the same C
training_n_changed: false
status: done
commit: b60a6ee  # retro: notebook first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # notebook title dated 20260917 (team review that day); artifacts 2026-09-18 14:01-14:47 UTC
supersedes: []
---

## Question
The 2026-09-17 team review of the D1a heatmap, as recorded in the 'Team claim' paragraphs of Observations 1-3 of the feedback notebook (research/asks/2026-09-17_team-feedback.md, capture verbatim_from_report):

> "Increasing K generally increases median lift on knowledge-driven gene sets. HVGs under PCA are the exception: K=5 is poor, K=10 peaks, K=20/40 drop — interpreted as one discriminative factor appearing between 5 and 10, then noise factors diluting accuracy. Under FA (described as “no orthogonality constraint”), HVG accuracy was said to increase monotonically to K=40." (A-20260917-01)

> "Allograft rejection, apoptosis, and interferon-α have near-zero direct-gene lift, while FactoSig K=40 gives about +12 pp. Hypothesis: nonspecific immune genes (LCK, ZAP70) in the allograft set overlap cytokine / TNF signaling, and DR extracts a latent signal that the gene-wise L1 probe misses." (A-20260917-02)

> "Ankit: FA’s nonorthogonal basis makes the expansion non-unique, so L1 zeroes one of two correlated factors; PCA’s orthogonal basis is unique. Implication: loadings cannot be interpreted biologically — sign is meaningless, and magnitude is unreliable because L1 picks one correlated factor, L2 splits, elastic-net is in between." (A-20260917-03, teammate:Ankit)

## Why it matters
Q3 (what DR adds, and how K matters) and QI (the IFN-alpha PCA-vs-FA case). F3/F6 test the collinearity account and the team's hypothesis (a) (H-S2-07), and establish the L1-at-fixed-C reading (H-S2-08).

## Estimand / null
- F1: D1 atlas medians by method x K; Stage 1 diagnostics energy; PCA independent-K fits vs K=40 prefixes.
- F2/F4/F5: best DR minus direct-gene median lift per panel and per patient (max over 16 DR cells, in-sample); covered-gene overlaps; selected-factor top loadings.
- F3: Hungarian PCA-FA matching; within-method score and loading collinearity; fold-wise logistic refits on stored IFN-alpha K=40 scores at C=1e-4 with L1, L2 and EN (l1_ratio 0.5).
- F6: per fold, univariate training-fold AUPRC per factor, fold-standardized scale, within-patient r(PC8, FA7), and held-out single-factor lift.
- F7: post-hoc rank-matched confusion and ADT FN−TP contrasts.
- No null.

## Implementation
- Notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_feedback_deep_dives_20260917.ipynb (sections 1-7), run in a Jupyter kernel on CPU on stored transductive scores; only small local refits, no new Stage 1 fits.
- The notebook's section 8 scopes the inductive IFN-alpha PCA40/FA40 x 7 refit (CPU minutes) without running it. Section 6 has no written verdict cell; F1-F7 have no findings md (W0.5).

## Artifacts
- analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/
- analysis/figures/stage2_diagnostic/feedback_deep_dive_20260917/
- analysis/cache/stage2_diagnostic/feedback_deep_dive_20260917/

## Results
- [H-S1-01 supports] F1: biological median-of-medians lift rises with K for all 4 methods (K=5/10/20/40, pp): FA −0.00/3.24/3.73/4.64, FactoSig 0.63/2.91/5.77/7.19, promax 0.38/3.00/5.02/5.66, PCA 0.29/3.55/3.98/4.22. The HVG+PCA peak at K=10 holds only for HVG500/HVG1000 (HVG1000 PCA 1.44/12.24/9.14/8.37 pp); HVG3000/10000 peak at K=20; HVG+FA is not monotonic (HVG1000 FA 1.44/13.61/10.88/14.16 pp).
- [H-S1-02 supports] F2/F4: the largest DR-over-direct gaps are allograft (promax K40 12.56 vs 0.23 pp), apoptosis (11.52 vs 0.005 pp) and IFN-alpha (PCA K40 11.08 vs 1.27 pp); best remaining DR minus direct is > 0 in 28/41 biological panels and 2/6 controls. Allograft and Hallmark TNFα/NF-κB share 13 of 147 genes; on allograft promax40 factor 005, TNF, LCK and ZAP70 rank 3, 5 and 8 by |loading|.
- [H-S2-08 supports] F3: sklearn FA scores are uncorrelated (max |off-diagonal r| PCA 0.000, FA 0.003; promax 0.924), so L1 is not choosing between collinear FA columns. At C=1e-4, held-out lift (pp) P02 / P07: FA L1 0 / 0, FA L2 16.14 / 5.07, PCA L1 15.94 / 11.08.
- [H-S2-07 contradicts] F6: FA7 is the strongest training-fold FA coordinate in every fold, and its held-out single-factor lift is P02 18.06 pp and P07 6.97 pp (PC8 15.94 and 11.08); within-patient r(PC8, FA7) runs from −0.575 to −0.722. FA does not allocate the axis to P03 only.
- [H-S2-05 supports] F7: malignant recall at the rank-matched cutoff, P02 PCA L1 0.322 (68/211) vs FA local L2 0.336 (71/211); P07 PCA L1 0.136 (3/22) vs FA L2 0.045 (1/22). ADT FN−TP contrasts are written for P02, P03 and P07.

## Claim gate
Supports (descriptive): lift rises with K for biological panels across methods; DR adds most for allograft, apoptosis and IFN-alpha; the IFN-alpha PCA40-vs-FA40 patient gap at the selected C is an L1 selection effect on uncorrelated FA scores, not missing FA structure.
Does not support: the collinear-L1 account (A-20260917-03; FA scores are uncorrelated); 'FA misses patients' as a property of FA; nested or inductive conclusions (same-fold regularizer selection; transductive Stage 1); rank-defined FN/TP as clinical errors.

## Next steps
- W3.2: C-matched IFN-alpha PCA40 vs FA40 under tuned C (settles H-S2-08 and retires H-S2-07 formally).
- W2.1: the inductive IFN-alpha refit scoped in section 8.
- W0.5: findings md for feedback_deep_dive_20260917.
- W5.3: fix the F7b column label (raw counts, not log1p).
