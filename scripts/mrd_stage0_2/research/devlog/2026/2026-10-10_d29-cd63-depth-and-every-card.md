---
id: 2026-10-10_d29-cd63-depth-and-every-card
date: 2026-10-10
type: analysis
title: D29 - PC8/FA7 x surface protein with each cell's total ADT count held fixed, and a direct malignant-vs-healthy protein contrast (CD63 followed up); the user's questions on full report Figures 20-23 answered in questions report v3.2
asked_by: user
asks: [A-20261010-26, A-20261010-27, A-20261010-28, A-20261010-29, A-20261010-30, A-20261010-31, A-20261010-32, A-20261010-33, A-20261010-34, A-20261010-35, A-20261010-36, A-20261010-37]
implemented_by: claude-code
questions: [QI, Q1]
workstream: W5
analyses: [D29]
hypotheses: [H-S2-05]
verdicts: []
alts: []
closes: []
advances: [W5.3]
claim_level: descriptive
compute: agent-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: pending
run_head: e3ee837
dirty_run: true  # the runner, the figure module, the report and the asks file are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
One message from the user, 2026-10-10 (late evening), written while building slides from the questions report's section 5; recorded verbatim in research/asks/2026-10-10_cd63-every-card-isotypes.md (A-20261010-26..37). The parts that asked for analysis:
> "it looks like the grey mark is very randomly distributed ... but their signals (Rho) are still weak and close to the isotype noise (vertical grey shades). I need Bachireddy Lab's help on whether the isotype protein are a good control/yardstick to say we need proteins have higher rho than isotypes." (A-20261010-26, -27) "So basically we can say look into CD63 and CD82." (A-20261010-29) "Can we see the detailed rank like panel D of figure 20 for CD63?" (A-20261010-30) "And we are saying "Among the malignant cells, cells rank high on PC8 tend to rank low on CD63." And this is built on top of the fact that PC8 is pretty decent at classifying malignant vs. healthy evaluated by the AUPRC score. So if we stretch the conclusion a bit, it's saying that malignant cells in these patients have lower CD63 expression or something, how can I say this better?" (A-20261010-31) "Let's also visualize CD63 like figure 21." (A-20261010-32)

The parts that asked for explanation (answered in the report, numbers below): "I am confused on 20274 of 289961 card x marker pairs, where do those numbers come from?" (A-20261010-33); panel c "it's coming from the rho angle, which means what? Please include my reasoning trace." (A-20261010-34); "We are not including PCA ... I don't see the need for color ... And this doesn't care about stage 2, right?" (A-20261010-35); "Let's have a scientific statement we can make about these" (A-20261010-36); "what's eta^2 of patient? ... If above the diagonal meaning cell type explain a lot then we would narrow the points to below the diagonal to be worthy of investigation." (A-20261010-37); and "We should emphasize the each grey dot is one of the 1569 malignant cells using textbf{} from latex." (A-20261010-28).

## Why it matters
QI and H-S2-05 (ALT-CT): the user is about to tell collaborators to "look into CD63 and CD82" on the strength of weak correlations near the isotype bands. Two things had not been checked: whether the factor-protein correlations run through each cell's total ADT count (CLR subtracts the per-cell mean log count, so the total enters every antibody, isotypes included), and whether malignant cells actually carry less CD63 than healthy cells, which no within-label rho can say. This decides which protein leads go into the reply to Pavan and what to ask the Bachireddy lab about isotypes and dsb.

## Estimand / null
- Factors: IFN-alpha PCA K=40 factor 8 (PC8, D18 sign x -1) and FA K=40 factor 7 (FA7, +1), oriented cohort-z scores. Protein: per-cell CLR over all 170 ADT features (161 markers + 9 isotypes), no dsb, as in D27.
- (1) Total-ADT sensitivity: for every factor x label x feature, D27's Spearman rho pooled, inside each supported patient (median over the 7, same-sign count) and inside each D27 supported patient x predicted type group (>= 30 cells; 71 healthy, 11 malignant; median, same-sign count against the in-type median as in report Table 3), raw and as a rank partial correlation on log1p(total raw ADT count): within the stratum, residualize rank(factor) and rank(protein) on [1, rank(log total ADT)] by least squares and take the Pearson correlation of the residuals. Reference: the largest |rho| over the 9 isotypes for the same statistic (raw and partial; also without RTK2071). The two checks as in D27/report Figure 22: check 1 = |median within| >= 0.20 with >= 5 of 7 of one sign; check 2 = |median in type| beyond the isotype reach.
- (2) Malignant vs healthy, no factor: inside each supported patient x predicted type with >= 10 cells of each label (the 21 groups of report Figure 19), and again inside one sample (only P09 has several samples; 372 of its 375 malignant cells are in P09_MRD_4), the AUC = P(malignant CLR > healthy CLR) + 0.5 P(tie) (Mann-Whitney U / n_m n_h); 0.5 = no difference. Adjusted AUC: the same on rank residuals after log total ADT within the group (both labels together). Reference: the 9 isotypes' AUCs in the same group. Also the AUC of log total ADT itself.
- Also: Spearman of each factor with log total ADT and with log nCount_RNA inside each patient; each isotype's CLR with log total ADT (median within patient) and its healthy-cell type eta2; Spearman of oriented PC8 with oriented FA7 among each label's cells (pooled and per patient).
- No null, no test: descriptive screen; no rule fixed beyond the D27 rule; no Stage 2 refit (c_policy not_applicable).

## Implementation
- Code: scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d29_protein_depth_contrast.py (new; reads D27 and the stored scores, writes the new slug; --resume skips when every output reads back; --out-root for scratch runs; refuses to rewrite a finished slug). Tests: tests/test_d29_protein_depth_contrast.py (8 synthetic tests: raw rho vs scipy, a correlation carried only by the covariate removed by the partial, one beside it kept, constant columns, AUC vs Mann-Whitney with ties, residual ranks, same-sign count, the namespace guard; 1 regression test on the stored outputs, skipped without EXP).
- Command (lab env, agent shell, 44-48 s, BLAS threads 4): `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d29_protein_depth_contrast.py`
- Provenance: the slug was first written at 23:28 UTC; the same task then added the sample-level contrast (P09), deleted that first write (no reader had used it) and rewrote the slug at 23:29 UTC. Regression checks: raw pooled, within-patient median and in-type median reproduce D27 for all 340 factor x label x feature rows to 5e-7 (D27 stores 6 significant digits); same-sign counts (D27's definition: groups against the within-patient median) and stratum counts exact; 21 contrast groups. All 8 pass.
- Report figures: scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_05q_cd63_depth.py (make_figures key 05q; asserts every plotted rho against D27 and D29; prints the PC8 x FA7 agreement and every figure's numbers): fig05n_cd63_ranks, fig05o_cd63_anchors, fig05q_mal_vs_healthy, fig05r_two_checks_depth (full report Figures 24-27). figs_05_pc8.py: fig05f (Figure 23) drops method colours and labels the regions of panels d-e; print_every_card_numbers() prints the "Every card" numbers below from D27/D18 tables.
- Report: questions report version 3.2 (sections/05_what_is_pc8.tex: Figure 20 caption, "Every card" rewritten with the user's reasoning trace and a statement box, new subsection "CD63, followed up (D29)", caveat, team box, next steps, provenance; report.tex revision note; core page 8 one sentence; README). Build: full 106 pages, team 105, core 12; 0 overfull boxes, 0 undefined references.

## Artifacts
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/d29_rho_depth.csv (680 rows: 2 factors x 2 labels x 170 features)
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/d29_rho_depth_strata.csv (33,320 rows: per stratum, raw and partial)
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/d29_isotype_reference.csv
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/d29_depth_axes.csv
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/d29_mal_vs_healthy.csv and d29_mal_vs_healthy_summary.csv (patient and sample levels)
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/d29_regression_checks.csv
- analysis/scorecards/stage2_diagnostic/protein_depth_contrast_v1/protein_depth_contrast_v1_{findings.md,manifest.json,artifact_ledger.csv}

## Results
- [H-S2-05 qualifies] CLR carries each cell's total ADT count into every antibody. Isotype CLR vs log total ADT, median within patient: -0.113 to -0.415 in healthy cells, -0.317 to -0.707 in malignant cells (9 isotypes). Oriented PC8 vs log total ADT: healthy cells -0.096 to -0.349 in every supported patient; malignant cells P01 -0.419, P02 -0.200, P04 -0.358, P09 -0.414, but P03 +0.023, P06 +0.049, P07 +0.001. PC8 vs log nCount_RNA: |rho| <= 0.202 in every patient and label.
- [H-S2-05 qualifies] Isotype reach for PC8 with total ADT held fixed: malignant cells 0.181 within patients and 0.070 inside cell types (both RTK2071; without it 0.048 and 0.064), down from 0.227 and 0.120; healthy cells 0.024 and 0.023, down from 0.109 and 0.043. FA7 malignant: 0.091 and 0.120 (from 0.257 and 0.122).
- [H-S2-05 qualifies] PC8, malignant cells, raw -> total ADT held fixed (median within patient; median inside type): CD63 -0.252 -> -0.200 (6/7), -0.138 -> -0.122 (9/11); CD99 +0.159 -> +0.172 (6/7), +0.150 -> +0.130 (9/11); CD82 +0.238 -> +0.188 (7/7), +0.143 -> +0.067 (9/11); CD48 -0.308 -> -0.177, -0.011 -> +0.029; CD116 -0.287 -> -0.129, -0.037 -> -0.043; CD33 -0.272 -> -0.177, +0.026 -> +0.009; CD146 +0.270 -> +0.057, +0.017 -> -0.026; CD71 -0.094 -> -0.096, -0.152 -> -0.153; CD117 -0.085 -> -0.091, -0.172 -> -0.102; CD244 +0.207 -> +0.201 (5/7), +0.046 -> +0.088. With total ADT held fixed, check 1 passes 1 protein (CD244; 14 raw), 29 proteins are outside the in-type band (12 raw), and CD63 and CD244 are the only proteins outside both isotype bands (raw: CD63 and CD82 passed both checks).
- [H-S2-05 qualifies] PC8, healthy cells: CD63 -0.111 (7/7) -> -0.008 (4/7) within patients, -0.063 -> -0.048 inside types; CD34 +0.245 -> +0.192 within, +0.019 -> +0.024 in type; CD22 -0.246 -> -0.226, -0.046 -> -0.043; CD82 +0.142 -> +0.158, +0.065 -> +0.072.
- CD63 x PC8 per patient (malignant, raw): P01 -0.452 (45 cells), P02 -0.252, P03 -0.001 (758), P04 -0.283, P06 -0.175, P07 +0.077 (22), P09 -0.411 (375); the 11 type groups: P02 HSC/MPP1 +0.005, HSC/MPP2 -0.407, LMPP -0.104; P03 HSC/MPP2 -0.038, MEP-MkP -0.041, NP1 +0.030; P04 HSC/MPP1 -0.354; P09 HSC/MPP1 -0.138, HSC/MPP2 -0.263, LMPP -0.249, cDC -0.207. Rank plot (all 1,569): mean CD63 rank 942 in the lowest tenth of PC8, 643 in the highest (middle 785); P09 258 -> 133 (middle 188); P09 HSC/MPP2 96 -> 51 (middle 65); P03 388 -> 392 (middle 379.5).
- FA7 is not an independent witness: oriented PC8 vs FA7 Spearman 0.620 among malignant cells (0.580-0.777 inside each patient) and 0.677 among healthy cells. FA7 x CD63 malignant: -0.195 (5/7) within, -0.111 (9/11) in type; total ADT fixed -0.133 and -0.092. CD82 x PC8 per patient: +0.107 to +0.684, P03 +0.142.
- Malignant vs healthy, same sample and predicted type (21 groups; no factor): CD63 lower in malignant cells in 15 of 21 (median AUC 0.356; 14 of 21, 0.369 with total ADT held fixed); the 9 isotypes lower in 5-8 of 21 (median AUC 0.531-0.594 raw). Strongest P09 HSC/MPP1 0.066 and HSC/MPP2 0.110; P03 HSC/MPP2 0.485; higher in P04 MEP-MkP (0.784), P09 cDC (0.641), P03 NP1 (0.602). CD99 lower in 14 of 21 (median 0.312) although it rises with PC8 among malignant cells; CD82 lower in 10 of 21 (0.509): higher in P02 and P03 (P03 HSC/MPP2 0.973), lower in P04, P06, P09. CD63 ranks 14th of 161 by median AUC; 20 proteins are lower in >= 15 of 21 groups, led by HLA-DR/DP/DQ (median AUC 0.171), CD38-HIT2 (0.176), CD81 (0.226) and HLA-DR (0.230). Patient level (P09 all samples) gives CD63 16 of 21, 0.356.
- Total ADT counts: malignant cells carry fewer than healthy cells of the same sample and type in 16 of 21 groups; P09 AUC 0.073 (HSC/MPP1), 0.049 (HSC/MPP2), 0.107 (LMPP), 0.068 (NP1) inside P09_MRD_4; P04 is the opposite (0.75-0.76 in HSC/MPP1, MEP-MkP, cDC).
- "Every card" (from D27/D18 tables; printed by figs_05_pc8.print_every_card_numbers): 1,801 cards x 161 markers = 289,961 pairs per label, plus 16,209 isotype rows; cards by method FA 581, FactoSig 380, FactoSig-promax 410, PCA 430; orientation 1,217 oriented, 336 sign flip, 248 never selected; 47 of the 56 bases come from the D14 screen (best Stage 1 configuration per panel by held-out patient-median lift). Robust pairs by method, normal: FA 3,730, FactoSig 2,229, promax 11,559, PCA 2,756; malignant: 2,068, 1,244, 3,785, 1,377. Panels a-b (oriented, best card per marker): no PCA card among the 20 shown; shown |rho| 0.685-0.809 (normal) and 0.526-0.680 (malignant), best PCA card for the same markers 0.552-0.689 and 0.351-0.503. eta2 (cards x label): normal median patient 0.032, cell type 0.105, above the diagonal 1,665, below 136, both < 0.1: 851; malignant 0.065, 0.074, 1,046, 755, 822. PC8 normal (0.010, 0.105), malignant (0.031, 0.120); FA7 normal (0.015, 0.173), malignant (0.025, 0.188).

## Claim gate
Supports (descriptive): among malignant cells of these patients, CD63 is the one PC8 protein correlate that stays beyond the isotype reach within patients and inside predicted cell types when each cell's total ADT count is held fixed (not in P03); separately, malignant cells carry less CD63 than healthy cells of the same sample and predicted type in 15 of 21 groups; part of the within-patient malignant-cell correlations of CD146, CD116, CD48 and CD33 runs through total ADT counts; CD82's inside-type correlation halves under that control.
Does not support: any protein as a malignancy marker; a malignant-vs-healthy contrast from any within-label rho; that FA7 independently confirms PC8's CD63; that holding total counts fixed is the right correction (cell size or activation can be biology); dsb-normalized results; anything separating malignancy from recipient origin, sample or batch; the CD63/CD82 biology (literature unchecked, W9.4).

## Next steps
- W5.8 (new, P2): rerun D27's two checks and D29's contrast on the normalization the Bachireddy lab recommends (dsb if the upstream assay, implied by obs nCount_dsb, or the empty-droplet background can be shared), for all cards.
- W9.4 (new, P2): literature check of CD63 and CD82 (tetraspanins) and of the proteins lower on malignant than on healthy cells of the same sample and type (HLA class II, CD38, CD81) before any text uses them.
- W5.3: the isotype and dsb questions go to the Bachireddy lab in the reply draft team_updates/2026-10-09_protein_association_for_pavan/email_draft_v4.md (MX sends; the agent drafts only). W5.3 stays open (F7b label fix; dsb).
- W0.15: whether the update figures and report_full.pdf are tracked is still the owner's decision; only report_core.pdf is.
