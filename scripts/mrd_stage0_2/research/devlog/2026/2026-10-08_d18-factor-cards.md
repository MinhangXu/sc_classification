---
id: 2026-10-08_d18-factor-cards
date: 2026-10-08
type: analysis
title: D18 generalized factor cards (factor_cards_v1 bundle, schema v1) for the 47 D14 bases, the IFN-alpha PCA40/FA40 pair and pre-declared focus cells
asked_by: user
asks: [A-20261008-19, A-20261008-21]
implemented_by: claude-code
questions: [QI, Q1]
workstream: W5
analyses: [D18]
hypotheses: [H-S2-03, H-I-03]
verdicts: []
alts: [H-S2-03:ALT-ROT=partial, H-I-03:ALT-ROT=partial, H-I-03:ALT-DEGEN=partial]
closes: [W5.1]
advances: [W5.3, W5.5, W7.2]
claim_level: descriptive
compute: agent-cpu
c_policy: global_selected
training_n_changed: false
status: done
commit: pending
run_head: d387eda  # the wave-2 checkpoint commit d387eda was HEAD when the bundle was computed (planned b60a6ee)
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "Factor interpretability to insight, with IFN-alpha as the example; expand beyond IFN-alpha and organize alongside the top priority" (A-20261008-19)

> "Correlate factors with other data modalities (cell types, surface protein expression)" (A-20261008-21)

Source: research/asks/2026-10-08_team-report.md (routes W5.1 and W5.3). This entry closes W5.1 and covers the ADT and cell-type part of W5.3 for the shortlist only.

## Why it matters
QI with IFN-alpha as the worked example (H-S2-03, H-I-03), feeding Q1: before any factor is named as a shared program, it needs an orientation, a selection/sign-stability record, a held-out lift that does not depend on the L1 threshold, and a rotation-aware match across methods and K. The bundle is the fixed interface (schema_version 1) the W7.2 factor explorer builds against, and the input to the W5.5 program-break census.

## Estimand / null
- Unit: one card per factor of one stored Stage 1 representation (panel x method x K, seed 42, transductive), paired with that representation's globally selected sharedness-LOPO Stage 2 probe. No refit of Stage 1 or Stage 2.
- Shortlist, declared before computing (factor_cards_shortlist.csv): the 47 D14 bases (patient_factor_overlap_calls.csv); the IFN-alpha PCA40/FA40 pair; focus cells = IFN-alpha FA K10 and K20 (F9c cross-K anchors) and PCA40 + FA40 for allograft rejection, apoptosis and TNFa/NF-kB. 56 representations.
- Orientation: orientation_sign = sign(beta) when beta is nonzero with the same sign in every supported fold where it is active (positive = toward malignant); 0 with status unoriented_all_zero (never active) or unoriented_sign_flip.
- Per factor x supported patient (P01, P02, P03, P04, P06, P07, P09):
  - univariate held-out lift = AP of the factor alone on the held-out patient minus that patient's prevalence, with the sign fixed on the training folds (training-fold standardized malignant-minus-normal mean gap); the F6 sign-agnostic version is kept for regression;
  - D8 fixed-model attribution: fitted single-factor lift and ablation dAP from the stored fold coefficients on training-fold standardized scores.
- eta2 of patient within label, computed here: normal patients with >=200 cells, malignant patients with >=20 cells (D14 strata, not the support rule).
- Matching within the same panel: Hungarian on |score r| at the same K for the other methods; best |r| across K; many-to-one R2 of the factor on the whole other basis (angle to its span); Tucker phi of loadings with the matched factor, oriented by sign(r); canonical correlations between whole and selected subspaces.
- Stage 1 structure: effective_k = non-degenerate factors; FA determinacy rho from the saved psi (FactoSig: psi approx 1 - h2; promax not computed because Phi is not saved); relative eigengap (PCA explained variance; FA psi-whitened d).
- ADT: obsm['ADT'] raw counts, CLR per cell over all 170 ADT features (161 markers + 9 isotypes); Spearman within label, pooled and patient-stratified (cell-weighted mean of within-patient rho).
- Pre-set flags: dense fit = median supported-fold nonzero fraction > 0.5; match listed when |r| >= 0.5; low determinacy rho < 0.707; small eigengap <= 0.10; degenerate = zero score variance or zero loadings; scale outlier = score SD more than 10x from the representation median (a separate caveat, not degeneracy: Stage 2 standardizes scores and promax column scale is arbitrary); near-duplicate oblique |r| >= 0.9; is_selected_driver = active in >=1 supported fold and the fit is not dense.
- Null: none (descriptive). C policy: the stored global selection per representation (layer4_5 cache; for the six Stage 0 controls the same max cell_weighted_auprc rule as the D1 atlas, rebuilt from the controls' coefficient paths).

## Implementation
- Code: scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/factor_cards.py (pure computation, card markdown, bundle schema and validator); run_d18_factor_cards.py (runner: shortlist, coefficient cache rebuild, per-panel workers, merge, regression checks, slide figures, manifest, findings, ledger); run_d18_factor_cards.sh (wrapper). Tests: tests/test_factor_cards.py (20 synthetic tests plus 2 regression tests against this bundle, skipped without EXP).
- Command (agent shell, 6 workers x 2 BLAS threads):
  `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 bash scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.sh --shortlist full --n-jobs 6`
  Rebuild only the bundle from finished parts: add `--merge-only`. Resume after an interruption: add `--resume` (skips representations whose parts and _done.json marker are complete; a merge failure writes merge_failed in the cache dir).
- Optional screen form for a busy machine: `screen -S W5_d18`, then the command above with `--resume`; detach with Ctrl-a d.
- Inputs: stage1_dr scores/loadings/diagnostics; preprocessing/panels/stage0_panel_genes; analysis/cache/stage2_diagnostic/layer4_5/selected_lopo_coefficients.pkl.gz; stage2_supervised/multiobjective/old34_controls_fullgrid_v1/sharedness_lopo/coefficient_paths.csv (chunked, six control representations) with analysis/scorecards/old34_controls_fullgrid_v1/stage2_sharedness_lopo_scorecard.csv; the cohort cache; stage1_direct_gene z-scored features (promax structure correlations); D1 atlas tables; D8/F3/F6/F9/D14 tables for the regression checks only.
- Runtime: full run 106 s wall (71 s for the 56 representations with 6 workers; controls coefficient rebuild 7 s on the first run). Rerun the same day after the review (degenerate_flag narrowed to zero variance; scale_outlier kept as a caveat): 102 s wall, every part recomputed; only the degenerate_flag column of factor_cards_index.csv changed, and all 27 regression checks still pass.
- Smoke and partial shortlists (`--shortlist smoke|ifna`) require `--out-root <scratch>`: the merge rewrites the bundle for the requested shortlist only, so the runner refuses to merge a partial shortlist into the canonical slug.
- Provenance: run on the d387eda checkpoint with an uncommitted tree; git state recorded in factor_cards_manifest.json.

## Artifacts
- analysis/scorecards/stage2_diagnostic/factor_cards_v1/
  - factor_cards_manifest.json, bundle_schema_v1.md, factor_cards_findings.md, factor_cards_artifact_ledger.csv
  - factor_cards_index.csv, factor_cards_genes.csv, factor_cards_patient_lift.csv, factor_cards_fold_coefs.csv (schema v1 required files)
  - factor_cards_matches.csv, factor_cards_subspace.csv, factor_cards_score_dist.csv, factor_cards_adt.csv, factor_cards_celltype.csv, factor_cards_phi.csv, factor_cards_shortlist.csv, factor_cards_regression_checks.csv
  - cards/<card_id>.md (1810 files)
- analysis/figures/stage2_diagnostic/factor_cards_v1/D18_ifna_pca40_factor008_card.{png,pdf}, D18_ifna_fa40_factor007_card.{png,pdf}
- analysis/cache/stage2_diagnostic/factor_cards_v1/selected_lopo_coefficients_controls.{pkl.gz,metadata.json}, parts/<representation>/ (resume units)

## Results
- Coverage: 56 representations, 1810 cards (1801 non-degenerate; the 9 degenerate cards are zero-variance FA factors). 117 factosig_promax cards carry the separate scale_outlier caveat (score SD more than 10x from the representation median); promax column scale is arbitrary and Stage 2 standardizes scores, so they are not degenerate. Bundle CSVs about 44 MB; card markdown about 13 MB.
- Regression anchors: all 27 rows of factor_cards_regression_checks.csv pass.
  - D8: 560 fold coefficients identical; fixed-model ablation dAP and single-factor lift max abs diff 9.6e-17; active (method, fold, factor) set identical; driver top-12 loading genes for PC8, FA7 and FA8 identical in order.
  - F6: 560 sign-agnostic univariate lifts max abs diff 1.9e-16; F3 Hungarian PCA40-to-FA40 indices identical (r max abs diff 3.3e-16).
  - F9: PC8/FA7 whole-patient lifts max abs diff 1.4e-14 pp; FA K5/K10/K20 matches to FA40 f7 identical; matched-factor univariate lift (21 rows) 1.4e-14 pp.
  - D14: focus eta2 (320 values) max abs diff 6.3e-8; max eta2 per representation over the 47 bases (94 values) 1.5e-7.
  - Atlas: reconstructed held-out AP matches the stored atlas AP in 390 of 392 representation x supported folds (max abs diff 1.8e-6, including the six rebuilt controls). The other 2 folds (IFN-alpha FA K10 and K20, held-out P03) have a single active |beta| of about 7e-6; their stored float32 probabilities tie, so the stored AP differs by up to 1.1e-3 (0.11 pp of F9c selected-model lift).
- [H-S2-03 supports] IFN-alpha PCA40 factor 8 (PC8): orientation sign -1, active in 5 of 7 supported folds (all-zero folds P01, P09), sign consistency 1.00. Malignant pole (oriented loading): TXNIP +0.269, IFITM2 +0.216, TENT5A +0.212, NCOA7 +0.210, TRIM5 +0.187, ELF1 +0.176. Normal pole: IFI27 -0.315, PSME2 -0.182, PSMA3 -0.169, CNP -0.168, LAP3 -0.166, USP18 -0.161. Univariate held-out lift (pp): P01 0.23, P02 15.94, P03 13.40, P04 16.27, P06 0.06, P07 11.08, P09 5.47. eta2 normal 0.010, malignant 0.031.
- [H-S2-03 supports] The PC8 direction lies largely in the span of the other bases: PC8 has R2 0.920 on the whole FA40 basis (Hungarian match FA7, r -0.685), 0.921 on FactoSig40 and promax40, and 0.893 on PCA20 (factor 8, r +0.942, Tucker phi 0.957); at K=5 no basis spans it (best R2 0.020).
- [H-I-03 supports] After orientation by sign(beta), FA7 (sign +1, active only in P03 and P04) lists the same poles as PC8: malignant TENT5A +0.171, BST2 +0.156, IFITM2 +0.155, GBP2 +0.143, IFITM3 +0.135, IFI44L +0.134; normal IFI27 -0.209, PSME2 -0.172, PSMA3 -0.162, IFI30 -0.152, CD47 -0.129, TRIM26 -0.124.
- [H-I-03 qualifies] PC8 and FA7 are not individually identifiable: PC8's relative eigengap is 0.055; FA7's determinacy is 0.649 (below 0.707) with effective_k 36 of 40; Tucker phi between PC8 and FA7 is 0.691, below the 0.85 fair band, although FA7 has R2 0.888 on the PCA40 basis. Across the bundle, 36 of the 119 selected drivers have low determinacy and 61 a small eigengap; none is degenerate, and 13 (all factosig_promax) carry the scale_outlier caveat.
- [H-S2-03 qualifies] FA7's univariate held-out lift is P02 18.06, P03 14.37, P04 19.87, P07 6.97, P09 4.81 pp while its fitted single-factor lift is 0 in the five all-zero folds; the FA7 axis already exists at K10 (factor 7, r +0.932, Tucker phi 0.910) and K20 (factor 7, r -0.774).
- [H-S2-03 qualifies] Per-factor reading is limited by the fits: 46 of 56 representations (41 of the 47 D14 bases) have a dense selected fit, so only 119 cards in 10 representations qualify as selected drivers (113 oriented, 6 sign-flip). Of 1553 cards active in at least one supported fold, 336 flip sign across folds.
- [H-S2-03 qualifies] An L1 zero is not absence of signal: 101 of the 248 never-selected non-degenerate cards reach >=5 pp univariate held-out lift in at least one supported patient (12 in at least three). Example: IFN-alpha PCA40 factor 4 is never selected yet reaches 25.40 pp in P02, 10.50 in P03 and 8.06 in P04, the highest-prevalence patients.
- Training-fold vs held-out sign: in 214 of the 560 IFN-alpha K40 PCA/FA factor x fold rows the held-out patient's best sign differs from the training-fold sign (median optimism of the sign-agnostic F6 lift 0.83 pp, max 21.18 pp); bundle-wide 4920 of 12670 rows (median 1.15 pp, max 81.49 pp). The cards therefore use the training-fold sign as the primary univariate lift.
- ADT (CLR per cell, within-label Spearman, oriented): PC8 in normal cells CD34 +0.251 (patient-stratified +0.263), CD22 -0.221; in malignant cells CD99 +0.222 (+0.192), CD244 +0.209, GPR56 +0.203; isotype max |rho| 0.056 normal, 0.101 malignant. FA7 in normal cells CD22 -0.354, CD112 +0.297, CD34 +0.260; in malignant cells GPR56 +0.215 (+0.155), CD63 -0.181, CD48 -0.180. eta2 of predicted cell type within normal cells: PC8 0.081, FA7 0.175.
- FA determinacy check: rho^2 from the saved psi agrees with the posterior-mean score variance within 0.027 over 581 FA cards.

## Claim gate
Supports (descriptive): on the transductive Stage 1 basis and the globally selected LOPO probes, the IFN-alpha program carries one oriented combination axis (PC8, FA7 at R2 about 0.9 to each other's basis) whose malignant-leaning pole is TXNIP/IFITM2/TENT5A-type genes and whose normal-leaning pole is IFI27/PSME2/PSMA3-type genes (L1, C=1e-4), with every card reproducing the D8, F6, F9 and D14 anchors.
Does not support: that factor 7 or factor 8 is a unique IFN-alpha sub-program (small eigengap, low FA determinacy, Tucker phi below the fair band); per-factor interpretation of the dense fits; any causal reading of ablation dAP; ADT associations free of patient, batch or donor/recipient-origin confounding; a census of program break-up beyond IFN-alpha (W5.5); inductive deployment; recurrence risk.

## Next steps
- W5.5: program-break census from the cards (selected drivers, recurrent sign-consistent factors, never-selected factors with univariate lift), plus the within-span per-patient L2 direction-angle test.
- W5.2: per-patient oriented gene-direction table for PC8 and FA7 (malignant minus normal log1p shift and AUROC per gene), and Haufe activation patterns compared by Tucker phi.
- W5.3: factor x predicted cell type tables stratified by patient, and fix the F7b column label (raw counts). dsb-normalized ADT is blocked: neither the cohort cache nor the raw AnnData holds dsb values or empty-droplet background, so the source is a user decision.
- W5.4: upstream factosig issue (normalize promax factor variances, save Phi) so promax determinacy can be computed; until then promax determinacy stays empty.
- W3.2: show any PC8-over-FA7 coverage difference under tuned C; the cards show FA7's univariate lift is present in P02 and P07.
- W7.2: the factor explorer builds against bundle_schema_v1.md; matched_cards may reference representations outside the shortlist.
- W4.5: seed and bootstrap recurrence for the selected drivers.
