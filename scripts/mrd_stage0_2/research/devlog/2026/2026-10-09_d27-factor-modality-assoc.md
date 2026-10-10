---
id: 2026-10-09_d27-factor-modality-assoc
date: 2026-10-09
type: analysis
title: D27 factor x predicted cell type and factor x CITE-seq surface protein for all 1,801 non-degenerate D18 factor cards, pooled vs within patient vs within patient x cell type
asked_by: user
asks: [A-20261009-08, A-20261008-21]
implemented_by: claude-code
questions: [QI, Q1]
workstream: W5
analyses: [D27]
hypotheses: [H-S2-05, H-S2-03]
verdicts: []
alts: [H-S2-05:ALT-CT=partial]
closes: []
advances: [W5.3]
claim_level: descriptive
compute: agent-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "Correlate factors with cell type and surface protein, let's think about this as a relative high priority, can we do it for the other factors too? And I should be able to explain the rho + 0.251 and rho + 0.222 for CD99 in malignant cell?" (A-20261009-08)

> "Correlate factors with other data modalities (cell types, surface protein expression)" (A-20261008-21)

Source: research/asks/2026-10-09_report-reading-notes.md (the user's reading notes on report v1, item 8; routes W5.3, D27) and research/asks/2026-10-08_team-report.md. D18 (2026-10-08_d18-factor-cards) gave pooled within-label rho for the top 10 markers per card only; this entry extends W5.3 from the shortlist preview to every card, adds the within-patient and within-cell-type views, and explains the two rho values the report quotes.

## Why it matters
QI and Q1 (H-S2-03, H-S2-05, alternative ALT-CT): before a factor is named by its surface-protein or cell-type correlates, the association has to hold inside patients (patients are nested in batch, and the pooled malignant cells are half P03) and has to be separated from predicted cell-type composition. The tables are also the cell-type and ADT layer that the W7 factor explorer and the W5.5 census will read, keyed by card_id.

## Estimand / null
- Unit: one factor card of the D18 bundle (56 representations; 1,801 non-degenerate cards; the 9 degenerate zero-variance FA cards are skipped) x one label (normal or malignant) x one ADT feature (161 markers; the 9 isotypes are kept as a noise reference and never ranked as markers). Stored transductive Stage 1 scores; nothing is refit.
- Score orientation: the D18 orientation_sign (sign of the globally selected LOPO beta; positive = toward malignant). Unoriented cards keep the raw arbitrary sign and carry oriented = False.
- ADT: raw counts from the cohort cache obsm['ADT'], per-cell CLR over all 170 features (log1p minus the cell's mean log1p; the D18 definition). No dsb (not available; user decision pending).
- rho: Spearman within label, computed as rank-standardize then one matrix product per stratum. Strata: (a) pooled = every cell of the label (the D18 number); (b) within each patient with >= 20 cells of the label, summarised over the 7 supported patients as the median rho and the number of patients whose rho has the sign of that median; (c) within each supported patient x predicted cell type stratum with >= 30 cells (median reported); (d) for regression only, the D18 cell-weighted mean of within-patient rho (normal >= 200, malignant >= 20 cells).
- Cell type: eta2 of the raw score by predicted.annotation within label, pooled (cell types with >= 20 cells; the D18 definition) and within each supported patient (cell types with >= 10 cells, at least two types; median reported), with the no-association expectation (g-1)/(n-1); per-cell-type mean of the oriented cohort-z score, pooled and per patient.
- Decision rule fixed before computing (robust association): non-isotype marker, |median within-patient rho| >= 0.20, same sign in >= 5 of the 7 supported patients. Sensitivity thresholds 0.10 and 0.30 are reported alongside. Reference rates (the same rule applied to isotype rows; exceedance of the card's own isotype maximum; retention inside patient x cell type strata) are descriptive and were added after the rule was fixed.
- Null: none (descriptive screen of about 580,000 card x marker x label pairs; no p-values). C policy: not applicable (no Stage 2 fit; orientation is read from D18's stored global selection).

## Implementation
- Code: scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/factor_modality_assoc.py (pure computation: strata, vectorized Spearman, summaries, cell-type eta2 and means, robust table, card summary, table schema); run_d27_factor_modality_assoc.py (runner: per-representation resume units, merge, regression checks, figures, findings, manifest, ledger); run_d27_factor_modality_assoc.sh (wrapper). The runner imports factor_cards.py (D18) for card ids and clr_per_cell; no shared code was edited.
- Command (agent shell, 4 workers x 3 BLAS threads = 12):
  `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 bash scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d27_factor_modality_assoc.sh --n-jobs 4 --resume`
  Rebuild tables, figures and findings from finished parts: `--merge-only`. A merge failure writes merge_failed in the cache dir. Smoke run (IFN-alpha PCA40 and FA40; refuses to run without --out-root): `--shortlist smoke --out-root <scratch>/d27_smoke --n-jobs 2`.
- Optional screen form on a busy machine: `screen -S W5_d27`, then the command above; detach with Ctrl-a d.
- Inputs: the D18 bundle (factor_cards_index.csv, factor_cards_shortlist.csv; factor_cards_adt.csv and factor_cards_celltype.csv for the regression checks); stage1_dr scores.npy for the 56 representations; the cohort cache (patient, CN.label, predicted.annotation, batch, obsm['ADT']). The raw AnnData was not loaded (R8).
- Runtime: smoke 18 s; full run 106 s wall (83 s for the 56 representations); the two later --merge-only rebuilds (figure layout and findings text only) took about 23 s each. Provenance: f5a8f4d with an uncommitted tree; git state in the manifest.
- Tests: tests/test_factor_modality_assoc.py, 16 synthetic tests (Spearman against scipy with ties, CLR and eta2 against factor_cards, strata, a planted within-patient association, a planted Simpson reversal, orientation and min/max swap, robust rule, cell-type tables, merge summaries, resume unit with empty/corrupt files, out-root guard) plus 3 regression tests against the real outputs; run in the lab env (pytest installed 2026-10-09): 19 passed. Full suite: 294 passed, 1 skipped, 1 failed (test_research_index::test_real_repo_check, E14 on H-S2-09 added to hypotheses.md by the concurrent integration work, not by this entry).

## Artifacts
- analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/
  - factor_modality_assoc_v1_findings.md (with the "How to read rho here" section), factor_modality_assoc_v1_manifest.json, factor_modality_assoc_v1_artifact_ledger.csv, factor_modality_assoc_v1_schema.md
  - d27_card_summary.csv, d27_adt_top.csv, d27_adt_robust.csv, d27_celltype_eta2.csv, d27_celltype_means.csv, d27_ifna_pc8_fa7_adt.csv, d27_adt_noise_floor.csv, d27_strata.csv, d27_regression_checks.csv
  - d27_adt_assoc_full.parquet (every card x label x 170 ADT features, per-patient rho for all 13 patients), d27_celltype_means_by_patient.csv.gz
- analysis/figures/stage2_diagnostic/factor_modality_assoc_v1/D27_ifna_pc8_fa7_adt_by_patient.{png,pdf}, D27_pooled_vs_within_patient_rho.{png,pdf}, D27_celltype_vs_patient_eta2.{png,pdf}
- analysis/cache/stage2_diagnostic/factor_modality_assoc_v1/parts/<representation>/ (resume units)

## Results
- Coverage: 56 representations, 1,801 non-degenerate cards, 612,340 card x label x ADT rows. Strata: normal pooled 58,549 cells, malignant pooled 1,569; supported-patient strata normal P01 11,815, P02 1,251, P03 1,569, P04 1,026, P06 3,215, P07 2,163, P09 16,751, malignant P01 45, P02 211, P03 758, P04 102, P06 44, P07 22, P09 375; supported patient x cell type strata with >= 30 cells: 71 normal, 11 malignant (P02 HSC/MPP1, HSC/MPP2, LMPP; P03 HSC/MPP2, MEP-MkP, NP1; P04 HSC/MPP1; P09 HSC/MPP1, HSC/MPP2, LMPP, cDC).
- Regression: all 10 checks pass. D18 pooled rho (36,020 rows) and patient-stratified rho (36,020) max abs diff 5.0e-11; D18 cell-type means (41,423 rows) 5.0e-10; D18 eta2_celltype_normal and isotype maxima (1,801 cards each) at most 5.0e-11 (the D18 CSV precision); the 14 PC8/FA7 values quoted in 2026-10-08_d18-factor-cards reproduce exactly at 3 decimals; the vectorized Spearman matches scipy.stats.spearmanr on 48 sampled stratum x marker values within 1.1e-16.
- [H-S2-03 qualifies] PC8 x CD34 in normal cells (the report's +0.251): pooled rho +0.251; within each supported patient P01 +0.18, P02 +0.14, P03 +0.25, P04 +0.15, P06 +0.24, P07 +0.38, P09 +0.31 (median +0.245, same sign 7/7, meets the robust rule); inside supported patient x cell type strata the median is +0.019. The association is consistent within every patient but is carried by which predicted cell types sit at the malignant-leaning pole (cell-type eta2 of PC8 in normal cells 0.081 pooled, 0.105 median within patient, against a no-association expectation of 0.0047; highest mean EBM +0.79, lowest ncMono -0.73 oriented z), not by variation inside a cell type.
- [H-S2-03 qualifies] PC8 x CD99 in malignant cells (the report's +0.222): pooled rho +0.222; within patient P01 +0.41, P02 +0.16, P03 +0.18, P04 +0.13, P06 +0.16, P07 -0.12, P09 +0.24 (median +0.159, same sign 6/7): the pre-declared robust rule is not met; inside the 11 malignant patient x cell type strata the median is +0.150, so what association there is persists inside cell types. PC8's strongest within-patient malignant correlates are CD48 -0.308 (6/7; pooled -0.192), CD116 -0.287 (7/7), CD33 -0.272 (5/7) and CD146 +0.270 (6/7); P03, the largest malignant patient, sits between -0.03 and +0.07 on all four, which pulls the pooled values toward zero. The card's isotype reference (largest |median within-patient rho| over the 9 isotypes) is 0.227 in malignant cells and 0.109 in normal cells.
- FA7 (IFN-alpha FA40 factor 7): normal cells CD22 -0.312, CD34 +0.298, CD112 +0.282 (each 7/7; patient x cell type medians -0.085, -0.004, +0.035); malignant cells CD116 -0.316 (6/7), CD11a -0.292 (5/7), CLEC12A -0.291 (7/7), with an isotype reference of 0.257 in malignant cells. Cell-type eta2 of FA7 in normal cells 0.175 pooled, 0.173 within patient (highest EBM +0.89, lowest cDC -0.70). Malignant within-patient cell-type eta2 exists for 4 patients only (P02, P03, P04, P09); P01, P06 and P07 lack two cell types with >= 10 malignant cells.
- [H-S2-05 supports] Normal cells, cohort-wide: 20,274 of 289,961 card x marker pairs (7.0%, on 843 cards) meet the robust rule, against 365 of 16,209 isotype rows (2.3%) under the same rule; pooled and within-patient rho agree (for the 23,133 pairs with |pooled| >= 0.20 the median within/pooled ratio is 0.89, IQR 0.76 to 1.02, and 2 reverse sign). But only 729 of the 20,274 robust pairs keep at least half their size inside supported patient x cell type strata (median retention ratio 0.15), and 1,665 of 1,801 cards have a larger within-patient cell-type eta2 than patient eta2. In normal cells, factor x surface-protein associations are mostly predicted cell-type (identity) axes; the 25 strongest are myeloid, erythroid and progenitor markers (CD11a, CD18, CD33, CD200, CLEC12A, CD123, CD71, CD151), led by HVGs (500) FA40 factor 3 x CD11a (median within +0.809, 7/7, pooled +0.829, patient x cell type +0.254).
- [H-S2-05 qualifies] Malignant cells, cohort-wide: 8,474 of 289,961 pairs (2.9%, on 954 cards) meet the robust rule, against 282 of 16,209 isotype rows (1.7%). Pooled malignant rho is a poor guide to within-patient rho: for the 16,190 pairs with |pooled| >= 0.20 the median within/pooled ratio is 0.24 (IQR 0.01 to 0.57), 707 reverse sign with |median within| >= 0.10, and only 2,783 (17.2%) meet the robust rule. Of the 8,474 robust pairs, 7,053 exceed their card's isotype reference and 1,797 keep at least half their size inside patient x cell type strata (median retention 0.26). Pooled malignant associations are therefore largely between-patient (patients nested in batch; P03 holds 758 of the 1,569 malignant cells).
- Isotype noise floor (|median within-patient rho| over all isotype rows): q95 0.163 and q99 0.231 in normal cells (max 0.342), q95 0.150 and q99 0.225 in malignant cells (max 0.359); marker rows q99 0.402 and 0.270.
- Selected drivers (D18, 119 cards): 61 have at least one robust marker in normal cells and 75 in malignant cells; of their robust pairs, 33 of 1,686 (19 cards) in normal and 127 of 680 (53 cards) in malignant cells keep at least half their size inside patient x cell type strata.
- Cell type, cohort-wide: in normal cells the cell-type eta2 has median 0.085 pooled (q90 0.547) and 0.105 within patient (q90 0.506); in malignant cells 0.078 pooled and 0.074 within patient; 1,046 of 1,801 cards have a larger within-patient cell-type eta2 than patient eta2 in malignant cells.
- Reading aid in the findings: a Spearman rho of +0.25 means that of two random cells, the one with more of the marker also has the higher factor score about 58% of the time (50% with no association), with about 6% of rank variance shared.

## Claim gate
Supports (descriptive): on the transductive Stage 1 bases with D18 orientations, CLR ADT without dsb and predicted cell types, PC8's normal-cell CD34 correlate holds inside every supported patient but is carried by predicted cell-type composition; in normal cells most factor x surface-protein associations are cell-identity axes; in malignant cells pooled associations are mostly between-patient, and PC8's CD99 correlate does not meet the pre-declared within-patient rule.
Does not support: any surface protein as a single-cell marker of a factor or of malignancy; a malignant-specific surface phenotype for PC8 or FA7; separating patient from batch (patients are nested in batch) or malignant biology from recipient origin (all malignant cells are recipient-derived); conclusions about dsb-normalized ADT; any causal reading; inductive deployment (Stage 1 is transductive; LOPO is label transfer); recurrence risk. The counts of robust pairs are a screen over related factors, not independent findings.

## Next steps
- W5.3: to close, fix the F7b column label (it holds raw counts, not log1p); dsb-normalized ADT stays blocked on the user's decision about a background source.
- W5.7 (proposed backlog line): held-out lift inside predicted cell types for the 119 selected drivers, plus composition-matched comparisons (the open half of ALT-CT); reuse the D27 strata and d27_celltype_means_by_patient.csv.gz.
- W5.5: let the program-break census read d27_card_summary.csv (eta2_celltype_within_median_*, adt_n_robust_*) to flag sub-axes that are cell-identity axes.
- W7.4 (proposed backlog line): show the D27 layer in the factor explorer, joined on card_id (d27_card_summary.csv, d27_adt_top.csv, d27_celltype_eta2.csv; schema in factor_modality_assoc_v1_schema.md).
- W9.3: CD99 as a literature lead (reported on myeloid disease stem cells); verify the source before any text uses it, and keep it out of claims (the within-patient rule is not met).
