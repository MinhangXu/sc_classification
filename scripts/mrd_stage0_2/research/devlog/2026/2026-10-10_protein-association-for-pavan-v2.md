---
id: 2026-10-10_protein-association-for-pavan-v2
date: 2026-10-10
type: docs
title: v2 of the Pavan protein update - four single-idea PC8 x surface-protein figures (per-patient heatmap, CD34 between vs within cell type, the same-patient same-type comparisons (19 of 21), isotype noise floor), captions and a shorter reply draft
asked_by: user
asks: [A-20261010-03, A-20261010-04, A-20261010-05, A-20261010-06]
implemented_by: claude-code
questions: [QI]
workstream: W5
analyses: [D27]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the update directory, D27 code and the report directory are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
The user's feedback on the v1 figure (2026-10-09_protein-association-for-pavan), quoted verbatim from the 2026-10-10 session. The protein-related part of a longer message; not yet recorded as an A-ID. The related recorded ask is A-20261009-08 (the CD34/CD99 rho, routed to W5.3/D27).
> "So the isotypes are in our data, how do you know they are isotypes, does the data annotate that? Why/how come we decide to use it to define noise? All the single cell is filtered through a CD34 gate. So that's probably the reason to see CD34 there, right? It's still interesting to see CD34 tracks PC8 in all 7 patients but almost not at all within a cell type, okay so "It mostly reflects PC8 differing between cell types." What other surface protein tracks with healthy? The new plot, first panel mixing pooled and mixing within patient is very hard to understand. What does 19 out of 20 within patient comparisons mean? Feel free to expand the figure or add another figure to illustrate it more clearly."

## Why it matters
QI: the reply to Pavan (2026-10-09_protein-association-for-pavan) must be readable cold. v1 put pooled and within-patient estimates in one panel and compressed the malignant-vs-healthy comparison into a count. v2 separates them into four figures with one idea each and answers the user's questions on isotypes, the CD34 sort and the other healthy-cell proteins. These are the cell-type confounds of D27 (H-S2-05 ALT-CT), stated for a collaborator; no verdict changes.

## Estimand / null
n/a (docs). All quantities are for IFN-alpha PCA K=40 factor 8 (PC8), oriented with the D18 sign (x -1, positive = malignant-leaning pole), with per-cell CLR ADT (log1p minus the cell's mean log1p over 170 features), healthy = CN.label normal, and the 7 supported patients.
- Figure 1 restates D27 per-patient within-label Spearman rho (rho_P01..rho_P09), the median and the same-sign count. Rows: the 10 largest |median| proteins per label (either sign) plus any of the questions report's top 3 by |pooled rho| not already among them (asserted; marked with a dagger). Per-patient reference row: max |rho| over the 9 isotypes in that patient; its median column is the median of that row (the Figure 4 floor is in the CSV column floor).
- Figure 2:
  - A: per predicted cell type (healthy cells of the supported patients, types with >= 30 cells), the mean CD34 CLR and mean oriented PC8 (cohort z), with Spearman across types and an unweighted least-squares line.
  - B: Spearman CD34 vs PC8 inside each supported patient x cell type group with >= 30 healthy cells (= D27 PCT_MIN_CELLS strata), next to the per-patient rho with types mixed.
  - C: D27 rho_within_median vs rho_pct_median for the Figure 1 healthy proteins.
- Figure 3: for each supported patient x cell type with >= 10 cells of each label (D27 CELLTYPE_MIN_CELLS_PATIENT), the malignant minus healthy mean oriented PC8, with a percentile bootstrap 95% CI. Cells are resampled within each group (2,000 draws, seed 20261010). All 21 qualifying pairs are counted. The v1 count of 19 of 20 covered only cell types with >= 20 malignant cells pooled (the v1 panel B display rule), which left out pDC2 in P06. Sensitivity: the same difference with healthy cells restricted to the samples (or batches) the pair's malignant cells come from.
- Figure 4: D27 rho_within_median for all 170 features; floor = max |isotype rho_within_median| per label.
- No null and no significance test; the floor is a yardstick. Nothing is refit, so there is no C policy.

## Implementation
- Code: scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figures_v2.py, entry point main().
  - Imports report_style.py from scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/ and factor_cards.clr_per_cell / card_id from W5 by sys.path; neither is edited.
  - Caps BLAS threads at 4.
  - Asserts the per-cell recompute against D27: all 177 per-patient cell-type means within 1e-4, the 7 per-patient CD34 rhos within 1e-5, the 71-group CD34 median within 1e-5, plus the PC8 orientation -1 and the earlier table's markers.
  - Fails if any text falls outside the 7 in canvas.
  - Review fixes (two reviewer passes, same day): all 21 pairs counted in Figure 3; P09 sample/batch sensitivity; PC8's classifier-set sign and the 10-of-21 interval count stated on Figure 3; a cell-type caveat line on Figure 1; dagger instead of bold for report proteins; isotype row shows its own median; Figure 2 uses gray/aqua instead of blue/orange (blue/red now mean only the sign of rho); dashed no-RTK2071 floor in Figure 4; raw CD34 quantiles and per-type medians written to CSV; captions trimmed.
- Command (agent shell, about 8 s): `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figures_v2.py`
- Inputs (EXP-relative, read only):
  - analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_ifna_pc8_fa7_adt.csv, d27_card_summary.csv, d27_strata.csv, d27_celltype_means_by_patient.csv.gz;
  - stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method=pca/k=40/seed=42/scores.npy (column 7);
  - analysis/cache/stage2_diagnostic/deg_hvg_lowsupport_v1/mrd_cite_binary_log1p_15560.h5ad (obs patient, CN.label, predicted.annotation, marker, sample, batch; obsm['ADT'] raw counts). The raw AnnData was not loaded (R8).
- Outputs, all in the update directory:
  - Figures, PNG at 200 dpi and PDF: fig1_proteins_by_patient_v2, fig2_cd34_between_within_v2, fig3_malignant_vs_healthy_same_type_v2, fig4_isotype_noise_floor_v2. Each canvas is exactly 7.0 in wide.
  - The numbers behind them: fig1_proteins_by_patient_v2.csv, fig2a_celltype_means_v2.csv, fig2b_cd34_rho_by_group_v2.csv, fig2c_within_vs_celltype_v2.csv, fig2_cd34_raw_quantiles_v2.csv, fig3_malignant_vs_healthy_same_type_v2.csv, fig4_isotype_noise_floor_v2.csv.
  - Text: caption_v2.md (4 or 5 sentences per figure, 614 words in all), email_draft_v2.md (215 words by wc, including the two bullet markers and the optional bracketed sentence) and a v2 section appended to README.md.
  - The v1 files (make_figure.py, protein_association_pc8.*, caption.txt, email_draft.md, protein_association_*.csv) were not touched.
- Visual check: each PNG was inspected at full size and at a 50% downscale, and figure 2's PDF was rendered. Fixes over three rounds: label collisions in Fig 2A, title and subtitle overflow beyond 7 in, footer overlap in Fig 4, and a clipped CI in Fig 3. A fourth round after review re-checked all four PNGs at full and 50% size (label placement in Fig 3A and Fig 4, spacing in Fig 1 and Fig 2B).

## Artifacts
none (no EXP writes). The outputs listed under Implementation are in the update directory in the repo.

## Results
- Isotypes in the data. The cohort cache obsm['ADT'] has 170 columns.
  - 9 are named Isotype-<clone>: MOPC.173, G0114F7, MOPC.21, MPC.11, RTK4174, RTK4530, HTK888, RTK2071, RTK2758.
  - 161 are named ADT-<target>.
  - D27 identifies isotypes by the name prefix (factor_modality_assoc.is_isotype_name).
  - The isotype noise reference is a D27 design choice. It is not a significance test.
- CD34 sort. obs marker = CD34pos for all 60,118 cells.
  - Raw CD34 counts over all cells: median 41, 5th to 95th percentile 5 to 340.
  - Median raw CD34 counts by predicted type (healthy cells, supported patients): HSC/MPP1 214, HSC/MPP2 205, NP1 95, MEP-MkP 88, pDC1 48, EBM 47, CLP 39, NP2 38, LMPP 37, MEP-EP 26, cDC 19, pDC2 14, ncMono 9.
- Figure 1, healthy cells. Within-patient median rho with same-sign count:
  - CD34 +0.245 (7/7), CD131 +0.190 (7/7), s.CD49f +0.158 (6/7), CD112 +0.154 (7/7);
  - CD85j -0.152 (7/7), CD47 -0.179 (6/7), CD81 -0.201 (7/7), CD39 -0.204 (7/7), CD48 -0.215 (6/7), CD22 -0.246 (6/7).
  - Per-patient max |isotype rho|: 0.08 to 0.19 (P01 0.193); median of that row 0.109. Floor 0.109.
  - Next past the cut at 10: CD43 +0.145 (7/7), CD45RO +0.143 (6/7), CD244 +0.143 (7/7), HLA.DR.DP.DQ -0.142 (5/7).
- Figure 1, malignant cells (22 to 758 per patient). Within-patient median rho with same-sign count:
  - CD146 +0.270 (6/7), CD82 +0.238 (7/7), CD25 +0.229 (7/7), CD244 +0.207 (5/7), CD99 +0.159 (6/7), GPR56 +0.137 (6/7);
  - CD11c -0.221, CD86 -0.243, CLEC12A -0.247, CD63 -0.252 (each 6/7), CD33 -0.272 (5/7), CD116 -0.287 (7/7), CD48 -0.308 (6/7).
  - Per-patient max |isotype rho|: P01 0.538, P02 0.255, P03 0.066, P04 0.306, P06 0.303, P07 0.295, P09 0.310; median of that row 0.303 (the Figure 4 floor is 0.227).
  - In P03 (758 malignant cells) every protein shown has |rho| <= 0.183 (CD99).
- Figure 2:
  - Healthy cells of the supported patients: 37,790, in 13 types. Spearman across type means (mean CD34 CLR vs mean PC8) is +0.599. HSC/MPP1 has CD34 3.88 and PC8 +0.559; ncMono has CD34 0.25 and PC8 -0.745.
  - CD34 vs PC8 within patient with types mixed: median +0.245 (range +0.140 to +0.376).
  - Within the 71 patient x cell type groups: median +0.019 (= D27 rho_pct_median); 43 of 71 groups have |rho| < 0.1; the median excluding MEP-EP is +0.010.
  - Exceptions: MEP-EP, all 7 groups positive (+0.304 to +0.613, median +0.470); CLP, all 6 positive (+0.085 to +0.187, median +0.138).
  - The other nine healthy proteins of Figure 1 have patient x cell type medians between -0.063 and +0.044.
- Figure 3:
  - Malignant > healthy mean PC8 in 19 of 21 pairs; median difference +0.408 cohort SD, range -0.223 to +0.980. In the 20 pairs of the v1 types: 19 of 20, median +0.410.
  - Not higher: NP1 in P09 (13 malignant / 1,256 healthy; -0.223, 95% CI -0.781 to +0.286) and pDC2 in P06 (15 / 213; -0.004, CI -0.397 to +0.406). pDC2 was outside v1's count because its panel showed types with >= 20 malignant cells pooled.
  - Sample/batch. The 14 pairs outside P09 each come from one sample and one batch. All 7 P09 pairs mix samples: P09 healthy cells come from 4 samples in batches E17P and E19P, and P09 malignant cells from E19P only. Restricting healthy cells to the malignant cells' samples gives HSC/MPP1 +0.479, HSC/MPP2 +0.983, LMPP +0.908, MEP-MkP +0.068, NP1 -0.175, NP2 +0.102, cDC +0.421. Restricting them to batch E19P gives +0.610, +0.976, +0.869, +0.248, -0.201, +0.050, +0.400. Either way the count stays 19 of 21.
  - 10 of 21 bootstrap intervals exclude 0.
  - Panel A example: P03 HSC/MPP2, 438 malignant vs 170 healthy; means +0.812 vs +0.248; difference +0.564 (CI +0.428 to +0.700).
- Figure 4:
  - Healthy: floor 0.109 (RTK4530); 33 of 161 proteins outside it.
  - Malignant: floor 0.227 (RTK2071 alone; the next isotypes are MOPC.21 0.158 and RTK4530 0.157); 9 of 161 proteins outside it.
  - All 9 isotype medians are positive in both labels: healthy +0.001 to +0.109, malignant +0.048 to +0.227.

## Claim gate
Supports (descriptive): on the transductive, RNA-only Stage 1 basis with D18 orientation, CLR ADT without dsb and predicted cell types, this entry restates D27 in four parts.
- In healthy cells, about ten surface proteins track PC8 with the same sign in 6 or 7 of 7 supported patients.
- For these proteins, and for CD34 inside a CD34+ sort gate, the association is carried mostly by differences between predicted cell types. The exceptions are MEP-EP (CD34) and, weakly, CLP.
- Malignant cells score higher on PC8 than healthy cells of the same predicted type in the same patient in 19 of 21 comparisons (19 of 20 in v1's types), also with P09's healthy cells restricted to the malignant cells' samples. PC8's sign was set by the malignant-vs-healthy probe, so only the within-patient, within-type persistence is new here.
- Malignant-cell protein correlations sit close to an isotype floor that one isotype sets.

Does not support:
- integration of protein into the factors (there is none);
- any surface protein as a single-cell marker of PC8 or of malignancy;
- separating malignancy from recipient origin, or patient from batch;
- reading the floor or the bootstrap intervals as significance tests;
- conclusions about dsb-normalized ADT;
- causal or mechanistic readings (including the maturation reading of the MEP-EP exception);
- inductive deployment (Stage 1 is transductive; LOPO is label transfer);
- recurrence risk.

## Next steps
- W5.3: record the user's verbatim questions above (isotypes, CD34 sort, other healthy proteins, the 19 of 20 count) and Pavan's question as A-IDs in research/asks/; this task's write scope left the asks files out.
- W5.3: D27 could take over the per-group within-cell-type rho table for selected markers (Figure 2B), the per-patient isotype maxima and the bootstrap same-type comparisons, so a later D27 version computes them once.
- W5.7: the same-patient, same-type comparison of Figure 3 is the score-level counterpart of the planned within-cell-type lift; reuse its pairs (>= 10 cells of each label) when W5.7 runs.
