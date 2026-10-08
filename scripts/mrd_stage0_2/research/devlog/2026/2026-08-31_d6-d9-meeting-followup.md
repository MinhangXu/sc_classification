---
id: 2026-08-31_d6-d9-meeting-followup
date: 2026-08-31
type: analysis
title: D6-D9 meeting follow-up (AP baseline, support association, IFN-alpha attribution, direct-gene baselines)
asked_by: meeting:2026-07-24
asks: [A-20260724-01, A-20260724-02, A-20260724-03, A-20260724-04, A-20260724-06]
implemented_by: unknown
questions: [Q2, Q3]
workstream: W4
analyses: [D6, D7, D8, D9]
hypotheses: [H-I-01, H-S2-01, H-S2-03, H-S1-02, H-S2-07]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: global_selected  # one pooled-LOPO-selected setting per representation and per direct-gene panel
training_n_changed: false
status: done
commit: b60a6ee  # retro: notebook first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # findings 'Generated: 2026-08-31T02:19'; work began 2026-07-24; notebook mtime 2026-09-03
supersedes: []
---

## Question
(reconstructed; the topics of A-20260724-01, -02, -03, -04 and -06, listed in 2026-07-24_team-meeting-six-questions). Is prevalence the right AP baseline; is lift driven by prevalence or cell counts; how do patient x panel x DR interact; why does IFN-alpha PCA40 beat FA40 in P02/P07; what does DR add over a direct-gene probe?

## Why it matters
Q2 (metric validity, support vs biology) and Q3 (value of DR). D6 is the metric contract (H-I-01); D9 is the first DR-vs-direct answer (H-S1-02); D8 is the start of the IFN-alpha case study.

## Estimand / null
- D6: constant-score AP vs malignant prevalence π; per representation, Spearman ρ across the 7 supported patients of a metric vs prevalence, malignant count or normal count, summarized by the median over 643 representations (nominal p-values are not for inference).
- D7: PCA40 − FA40 lift and K=40 method spread per panel x patient.
- D8: exact reconstruction of held-out AP from stored coefficients; fixed-model leave-one-factor-out; Hungarian PCA-to-FA alignment by |score correlation|.
- D9: best DR (chosen in-sample by pooled-cell LOPO AP across method and K) vs a direct-gene probe on standardized panel genes, both at their globally selected settings.

## Implementation
- Notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_meeting_followup_metric_interactions.ipynb, run in a Jupyter kernel on CPU from stored scores, coefficients and per-patient metrics (no refit).
- The direct-gene inputs are branch old34_direct_gene_all_v1 (2026-07-24).

## Artifacts
- analysis/scorecards/stage2_diagnostic/meeting_followup_v1/
- analysis/figures/stage2_diagnostic/meeting_followup_v1/

## Results
- [H-I-01 supports] D6: constant all-healthy and all-malignant scores both reproduce malignant prevalence exactly as AP; they differ only in threshold utility (P06: accuracy 0.9865 vs 0.0135, AP = π = 0.0135 for both).
- [H-S2-01 qualifies] D6: median within-representation Spearman ρ over 643 representations: AP vs prevalence 0.964, raw lift vs prevalence 0.821, raw lift vs malignant count 0.607, AUROC vs prevalence 0.500.
- [H-S2-03 supports] D8 (L1, C=1e-4; AP reconstruction error 1.38e-06): PCA factor_008 is the only active PCA factor, with lift P02 15.94, P03 13.40, P04 16.27, P07 11.08 pp (folds P01 and P09 all-zero); FA factor_007 is active only in P03/P04; PC8 correlates with FA factor_007 at r −0.685 and with FA factor_008 at +0.580.
- [H-S1-02 supports] D9: best DR beats direct genes on pooled LOPO AP in 38/41 panels, on median lift in 16/41 and on P20 lift in 17/41; median DR value added: pooled AP +0.0221, median lift −0.0033, P20 lift −0.0019 (best DR selected in-sample).
- [H-S2-07 contradicts] D7: over 38 panels x 7 patients the median PCA40 − FA40 lift in P02 is −2.76 pp (FA usually better), and MHC class I in P02 gives FA40 47.44 vs PCA40 0.00 pp, so FA is not generally biased toward P03.

## Claim gate
Supports (descriptive): constant-score no-skill AP = π; lift remains rank-associated with prevalence; DR usually improves pooled-cell AP but not consistently patient-equal median or P20 lift; IFN-alpha PC8 is the selected driver in P02/P07 while FA is all-zero there at the chosen L1 setting.
Does not support: that high lift is patient biology rather than support or prevalence; that FA is intrinsically inferior to PCA for IFN-alpha; that DR universally adds value (best DR is an in-sample winner).

## Next steps
- W3.3: nested or prespecified DR-vs-direct selection (D23).
- W1.5: prevalence/support-matched lift (ALT-PREV).
- W4.1: patient-level uncertainty and the AUROC companion column.
