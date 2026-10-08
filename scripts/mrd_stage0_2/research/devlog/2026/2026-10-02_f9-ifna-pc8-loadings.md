---
id: 2026-10-02_f9-ifna-pc8-loadings
date: 2026-10-02
type: analysis
title: F9 IFN-alpha PC8 loadings, within-cell-type PC8 lift, and when P04 joins FA7
asked_by: user
asks: [A-20261008-03, A-20261008-04, A-20261008-05, A-20261008-06]
implemented_by: unknown
questions: [QI, Q3]
workstream: W5
analyses: [F9]
hypotheses: [H-S2-03, H-I-03, H-S2-08]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: global_selected  # selected-model lifts at the atlas L1 C=1e-4, next to univariate single-factor lifts
training_n_changed: false
status: done
commit: b60a6ee  # retro: notebook first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # f9 CSVs and figures 2026-10-02 14:57 UTC
supersedes: []
---

## Question
(reconstructed from the user's report of a team discussion, recorded as A-20261008-03..06; the meeting date is not recorded) PC8 has two opposite-sign gene groups (IFI27, PSME2, PSMA3, CNP, LAP3, USP18 vs TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1): what do they mean? The team hypothesized that PCA orthogonality pushes later components beyond P03 while FA keeps allocating factors to the highest-prevalence patient, and asked to inspect the FA factor-by-patient matrix.

## Why it matters
QI with IFN-alpha as the worked example (H-S2-03, H-I-03) and the mechanism of the PCA40-vs-FA40 coverage gap (H-S2-08, H-S2-07).

## Estimand / null
- F9a: |PC8 loading| vs each gene's own median held-out lift (sign-agnostic per-gene AUPRC on z-scored log1p, the Stage 1 standardization).
- F9b: PC8 and FA7 lift vs a cell-type-mix ceiling (per-patient malignant fraction by predicted type, which uses held-out labels), and PC8 lift within predicted cell types (at least 8 malignant and 8 normal cells).
- F9c: each FA fit at K 5/10/20 matched to K=40 FA7 by score correlation; univariate single-factor lift vs the selected-L1 lift per patient.
- No new Stage 1 fits; no null.

## Implementation
- Section 9 of scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_feedback_deep_dives_20260917.ipynb, run on CPU on stored scores and coefficients.
- The F9 rows are missing from f_artifact_ledger.csv (W0.5).

## Artifacts
- analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f9_ifna_pc8_loading_vs_gene_univariate_summary.csv
- analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f9_ifna_pc8_within_celltype_lift.csv
- analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f9_ifna_pc8_vs_celltype_ceiling.csv
- analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f9_ifna_fa_matched_factor_lift_by_k.csv
- analysis/figures/stage2_diagnostic/feedback_deep_dive_20260917/F9a_ifna_pc8_loading_vs_gene_lift.png
- analysis/figures/stage2_diagnostic/feedback_deep_dive_20260917/F9b_ifna_pc8_within_celltype.png
- analysis/figures/stage2_diagnostic/feedback_deep_dive_20260917/F9c_ifna_fa_matched_factor_across_k.png

## Results
- [H-S2-03 supports] F9a: PC8 contrasts IFI27 +0.315, PSME2 +0.182, PSMA3 +0.169, CNP +0.168, LAP3 +0.166, USP18 +0.161 against TXNIP −0.269, IFITM2 −0.216, TENT5A −0.212, NCOA7 −0.210, TRIM5 −0.187, ELF1 −0.176 (no gene on both sides). |loading| vs single-gene lift has Spearman 0.30 over 92 genes; IFI27 alone gives +0.39 pp, and the best single genes are BST2 4.06, LY6E 2.93 and GBP2 2.77 pp.
- [H-I-03 supports] Orientation: PC8's L1 coefficient is negative in every active fold (−0.040 to −0.064), so the malignant-leaning pole is the negative group (TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1). FA7 (positive coefficient; r −0.685 with PC8) carries the same contrast with flipped raw signs. IFI27 and IFITM2 are both interferon-stimulated genes on opposite poles.
- [H-S2-08 supports] F9c: the FA7-matched axis already exists at K=10 (r +0.932) and K=20 (r −0.774). Single-factor vs selected-L1 lift: P04 13.51/0 (K10), 21.72/0 (K20), 19.87/19.87 (K40); P02 13.35/0, 25.55/0, 18.06/0. L1 at C=1e-4 keeps only P03 at K=10 and K=20, so P04 'joining' at K=40 is L1 starting to keep an existing axis.
- [H-S2-03 qualifies] F9b: the cell-type-mix ceiling exceeds PC8 in P02 (57.33 vs 15.94 pp) and P03 (38.04 vs 13.40 pp) but not in P07 (8.17 vs 11.08 pp); PC8 stays positive inside HSC/MPP2 (P02 +21.8, P03 +12.7, P07 +10.3, P09 +27.1 pp).

## Claim gate
Supports (descriptive): PC8 and FA7 encode one shared IFN-alpha combination axis whose malignant-leaning pole is the TXNIP/IFITM2/TENT5A side; the PCA40-vs-FA40 patient coverage gap at the selected C is an L1 threshold effect on that axis; PC8 keeps signal within HSC/MPP2.
Does not support: naming PC8 gene by gene as 'the IFN-alpha program' (it is a combination axis); per-gene up/down direction (F9a is sign-agnostic); 'FA misses patients' as a property of FA; cell-type-independent biology in P02/P03 (composition alone exceeds PC8 there); inductive or causal statements.

## Next steps
- W5.2: oriented per-patient gene-direction table for PC8 and FA7, plus Haufe activation patterns.
- W3.2: C-matched IFN-alpha PCA40 vs FA40 under tuned C.
- W5.1: factor cards that reproduce the D8/F6/F9 tables.
- W0.5: add the F9 rows to f_artifact_ledger.csv.
