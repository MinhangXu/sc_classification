---
id: 2026-10-08_team-report-devlog-pivot
date: 2026-10-08
type: meeting
title: Team report (aggregation priority, new-patient data, pivot to a devlog and context engineering) and its fact-check
asked_by: user
asks: [A-20261008-01, A-20261008-02, A-20261008-03, A-20261008-04, A-20261008-05, A-20261008-06, A-20261008-07, A-20261008-08, A-20261008-09, A-20261008-10, A-20261008-11, A-20261008-12, A-20261008-13, A-20261008-14, A-20261008-15, A-20261008-16, A-20261008-17, A-20261008-18, A-20261008-19, A-20261008-20, A-20261008-21, A-20261008-22, A-20261008-23, A-20261008-24, A-20261008-25, A-20261008-26, A-20261008-27, A-20261008-28, A-20261008-29, A-20261008-30, A-20261008-31, A-20261008-32]
implemented_by: claude-code
questions: [Q1, Q2, Q3, QI, meta]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: pending
run_head: b60a6ee
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored  # the report was received and fact-checked on 2026-10-08
supersedes: []
---

## Question
The user's 2026-10-08 team report: a TODO list, notes from a team discussion, an inline ask about repository organization, and three scientific questions. Its 32 items are recorded in report order in research/asks/2026-10-08_team-report.md (A-20261008-01..32); this entry records them and does not answer them. The meeting behind the discussion notes is not dated in the report; its content (PC8 genes, IFN-alpha PCA vs FA) matches D8 and F9, so it probably fell between 2026-09-18 and 2026-10-02 (reconstructed).

| Ask | Topic | Routed to |
|---|---|---|
| A-20261008-01 | interactive D1 heatmap explorer (done) | D1, W7.1 |
| A-20261008-02 | P02/P03/P04 'disproportionately shape the factors' | H-S1-05, W1.1, W2.1 |
| A-20261008-03 | IFN-alpha PCA40 lift in 4 patients, FA40 only P03/P04 | H-S2-08, W3.2 |
| A-20261008-04 | hypothesis (a): PCA orthogonality vs FA allocation | H-S2-07 |
| A-20261008-05 | inspect the FA factor-by-patient matrix | W1.1 |
| A-20261008-06 | PC8 opposite-sign gene groups | F9, W5.2 |
| A-20261008-07 | TOP PRIORITY: test whether to aggregate patients | D14, W1.2, H-S1-04 |
| A-20261008-08 | step 1: factors on all patients, project each patient | D14 |
| A-20261008-09 | step 2: per-patient factor distributions | D14 |
| A-20261008-10 | step 3: overlap means shared, disjoint means subtypes | D14, W1.2 |
| A-20261008-11 | inline ask: agent-agnostic context, dated log, subdirs, paper theme | W0.1 |
| A-20261008-12 | new MRD patients from Yu-Hsiang | W8.1 |
| A-20261008-13 | option 1.1: subset to the top patients plus new data | W2.5 |
| A-20261008-14 | option 1.2: equal per-patient weighting in Stage 1 | W2.1, W2.3 |
| A-20261008-15 | option 1.3: re-examine PCA/FA factors on the new basis | W1.3 |
| A-20261008-16 | option 2: exclude the held-out patient from Stage 1 | W2.1, H-S1-06 |
| A-20261008-17 | option 3: P02/P03/P04-only rerun | W2.5 |
| A-20261008-18 | days since transplant / before relapse as covariate | W1.7 |
| A-20261008-19 | factor interpretability beyond IFN-alpha | W5.1 |
| A-20261008-20 | interpretability results in the app | W7.2 |
| A-20261008-21 | factors vs cell types and surface proteins | W5.3 |
| A-20261008-22 | interpretability frameworks from other fields | W5.4, W9.1 |
| A-20261008-23 | no causal claim without intervention | THEME |
| A-20261008-24 | AUPRC vs number of cells | D15, W6.2 |
| A-20261008-25 | learning curve to justify sequencing more patients | W6.1, W6.4 |
| A-20261008-26 | random subsampling for the curve | W6.1, W6.2 |
| A-20261008-27 | Ankit's semi-supervised joint model (later) | W3.7 |
| A-20261008-28 | single-cell ML literature review | W9.1 |
| A-20261008-29 | central publication branch; organize the repo around it | THEME, W0.1 |
| A-20261008-30 | Q1 shared vs patient-specific | THEME |
| A-20261008-31 | Q2 strength of evidence | THEME |
| A-20261008-32 | Q3 what DR adds over direct genes | THEME |

## Why it matters
The report sets the current scientific questions (Q1-Q3) and the backlog. Several of its premises need correcting before they drive new compute; the fact-check below records the corrections, and the routes above put every item on a W#.n task or THEME.

## Estimand / null
n/a (meeting record). The fact-check reads existing artifacts only; no new analysis.

## Implementation
- Fact-check on 2026-10-08: read-only readers of the repository, notebooks, artifacts and git history, plus scratch checks in the session scratchpad that are not repo artifacts (the L1 entry rule, a factor-by-patient sum-of-squares matrix, gene directions). Their productized versions are W3.1 (D19), W1.1 (D16) and W5.2.
- Asks file: research/asks/2026-10-08_team-report.md. The research spine that answers the inline ask is recorded in 2026-10-08_research-spine-adopted; the literature reviews in 2026-10-08_literature-reviews.

## Artifacts
none

## Results
Fact-check corrections (evidence lives in the cited analysis entries):
- 'Patients with more malignant cells (P02, P03, P04)' are the highest-PREVALENCE patients (P03 32.6%, P02 14.4%, P04 9.0%). By malignant count the order is P03 758 > P09 375 > P02 211 > P04 102. Together they hold 1,071/1,569 malignant cells (68.3%; P03 alone 48.3%) but only 4,917/60,118 cells (8.2%) of the label-blind Stage 1 fit, which P09 (28.5%) and P01 (19.7%) dominate.
- 'Disproportionately shape the factors' is untested as stated. They dominate the Stage 2 labels, the cell-weighted pooled-LOPO regularization selection and the D9 best-DR selection, and lift ranks with prevalence (D6 median within-representation ρ 0.821 for lift, 0.964 for AP). D14 finds their factor-score distributions overlapping other patients on 40/41 biological bases; a leave-out or reweighted Stage 1 refit is needed (H-S1-05).
- IFN-alpha PCA40 'lift in four patients' holds for meaningful lift (> 5 pp: P02 15.94, P03 13.40, P04 16.27, P07 11.08); P06 is +0.06 pp, so lift > 0 holds in 5. FA40 keeps FA7 only in P03/P04 (and unsupported P10), where FA beats PCA (14.37 vs 13.40; 19.87 vs 16.27 pp). Both use L1 at C=1e-4.
- Hypothesis (a) is contradicted: FA7 mirrors PC8 (r −0.685; PC8 vs FA8 +0.580); FA7 alone gives P02 18.06 pp (above PC8's 15.94), P04 19.87 and P07 6.97 pp; L2 at the same C recovers P02 16.14 and P07 5.07 pp. A scratch check explains all 14 zero/nonzero L1 outcomes: FA7's standardized class gap (0.66-0.70) sits just under the entry threshold 4/(C·n_train) of about 0.68-0.70.
- L1 entry-rule detail (the same scratch refit on stored IFN-alpha K=40 scores, re-run read-only during integration on 2026-10-08 with identical output; not a repo artifact until W3.1/D19): at C=1e-4 the threshold 4/(C·n_train) is 0.678-0.703 SD in the P02, P03, P04, P06 and P07 folds, and 0.829 (P01 held out, n_train 48,258) and 0.930 (P09 held out, n_train 42,992) in the two smallest training folds. |PC8 gap| is 0.704-0.770 and the FA7 gap 0.658-0.705; comparing gap to threshold reproduces all 14 zero/nonzero outcomes (7 supported folds × 2 methods), including PCA's all-zero P01/P09 folds. Raising C, FA7 enters in P06 (0.65 pp) and P07 (6.97 pp) already at C=1.02e-4 and in P02 (18.06 pp) at 1.05e-4; PCA40-L1 is unchanged up to 1.1e-4. In the stored F6 fold-scale table (f6_ifna_factor7_8_fold_scale_by_patient.csv) the held-out malignant mean z on PC8 is below the healthy mean in all 7 supported patients, and on FA7 it is above in all 7.
- P04 'joining' FA at K=40 is not a new direction: the matched FA7 axis exists at K=10 (r +0.93) and K=20 (r −0.77), with P04 single-factor lift 13.51 and 21.72 pp; L1 only starts keeping it at K=40.
- FA is not 'PCA without orthogonality': sklearn FA scores are essentially uncorrelated (max |r| 0.003 vs PCA 0.000); only the loadings are non-orthogonal (Gram max 0.403). The correlated method is factosig_promax (max |score r| 0.924). This also corrects Ankit's collinear-L1 account (F3).
- 'FA misses patients' is not a property of FA: other IFN-alpha cells reach all 7 patients (e.g. FactoSig40 L2: P02 18.81, P04 19.35, P07 7.02 pp), and across 38 panels the median PCA40 − FA40 lift in P02 is −2.76 pp (MHC-I in P02: FA40 47.44 vs PCA40 0 pp).
- The FA factor-by-patient matrix largely exists: F6 covers 2 methods x 40 factors x 7 patients, and D14 gives FA7 a malignant η² of 0.025 (FA40 maximum 0.121, factor 2). Only a variance-allocation matrix is missing; a scratch version shows P02, not P03, over-represented on most factors (PCA 31/40, FA 27/36).
- PC8 gene lists are the exact top 6 per pole; the trailing 'IFI27, IFITM2' names two interferon-stimulated genes on opposite poles. PC8's Stage 2 coefficient is negative in every active fold (−0.040 to −0.064), so the malignant-leaning pole is the TXNIP/IFITM2/TENT5A/NCOA7/TRIM5/ELF1 side.
- 'Learn factors on all patients, project each patient' is already true of every stored basis (transductive fit on 60,118 cells); steps 1-3 were implemented as D14 on 2026-10-02 (45/47 aggregate_shared, 2 shared_with_location_shift, 0 separate_subtype_candidate). Significance cannot decide the call: the 25-permutation p sits at its floor, 1/26 = 0.038, in every normal stratum.
- Option 1.2 points the other way from what is intended: equal per-patient weighting would raise P02+P03+P04 from 8.2% to 23.1% of the Stage 1 fit. Option 3's 'large cell counts (2, 3, 4)' are among the smallest patients by cell count (1,462 / 2,327 / 1,128 cells; P09 17,126 and P01 11,860 are the largest); a ≥100 malignant-cell cutoff gives P03, P09, P02 and P04.
- 'AUPRC vs number of cells' exists as D15 (2026-10-02), but it is a Stage 2 cell-depth curve at a fixed 12-patient mix with C fixed and 3 draws; it cannot justify sequencing more patients.
- Factor x modality correlation exists only for IFN-alpha (F6c, F7, F9b) and three focus panels at the rank-error level (D1DD10-12); ADT is stored as raw counts (161 markers + 9 isotypes) and the F7b 'log1p' column holds raw count differences. Timing covariates are tabulated per sample (D1DD13); they are collinear with patient except P09.
- The app is validated (--check all PASS; 25/25 tests) but was untracked until b60a6ee and has no factor-level view. Q3 depends on the estimand (D9: 38/41 pooled AP, 16/41 median lift, 17/41 P20 lift; D1e/F4: 30/47 on median lift when chosen by median lift); Q2 has no inferential statistics yet.

## Claim gate
Supports (descriptive): nothing new; this entry records questions and corrections whose evidence is in the cited analysis entries.
Does not support: the report's premises where corrected above ('more malignant cells', 'shape the factors', hypothesis (a), 'FA misses patients', option 1.2's direction, option 3's premise, D15 as a patient curve).

## Next steps
- W1.1: D16 factor x patient variance-allocation matrix (A-20261008-02, -05).
- W1.2: D20 = D14 v2 with calibrated gates (A-20261008-07, -10).
- W3.1 and W3.2: D19 regularization contract and the C-matched IFN-alpha comparison (A-20261008-03, -04).
- W2.1: D17 inductive / leave-out / patient-balanced basis pilot (A-20261008-14, -16).
- W6.1: D15b patient-count learning curve (A-20261008-25, -26).
- W8.1: the user emails Yu-Hsiang (A-20261008-12).
- W0.1: land the research spine (A-20261008-11, -29).
