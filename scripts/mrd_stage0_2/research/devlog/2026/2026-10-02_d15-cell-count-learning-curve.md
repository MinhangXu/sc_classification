---
id: 2026-10-02_d15-cell-count-learning-curve
date: 2026-10-02
type: analysis
title: D15 held-out AUPRC vs number of Stage 2 training cells (frozen basis, fixed C)
asked_by: user
asks: [A-20261008-24, A-20261008-25, A-20261008-26]
implemented_by: unknown
questions: [Q2]
workstream: W6
analyses: [D15]
hypotheses: [H-S2-06]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: global_selected  # atlas-selected penalty and C reused at every training size
training_n_changed: true  # training cells subsampled at fixed C (W11)
status: done
commit: b60a6ee  # retro: first commit containing cell_count_learning_curve.py and the notebook
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # CSVs 2026-10-02 14:49 UTC, figures 14:49-14:50 UTC
supersedes: []
---

## Question
(reconstructed from the user's TODO list as recorded in research/asks/2026-10-08_team-report.md, A-20261008-24..26) Plot AUPRC against the number of cells as a learning or marginal-benefit curve; the user expects it to help grant applications justify sequencing more patients, and suggests random subsampling to estimate it. The original request of about 2026-10-01 was not saved.

## Why it matters
Q2 and grant justification (H-S2-06). The result decides what a learning curve can and cannot say about sequencing more patients; the patient-count version is D15b (W6.1).

## Estimand / null
- For each held-out supported patient: label-stratified random subsets of the training cells (500, 1k, 2k, 5k, 10k, 25k; 3 draws each) plus the full set; StandardScaler and the atlas-selected penalty and C are refit on each subset; the held-out patient is scored in full.
- Cohort summary: median over the 7 patients of each patient's mean over draws.
- Four representations: IFN-alpha PCA40 and FA40 (L1, C=1e-4), apoptosis factosig_promax K40 (L2), allograft factosig_promax K40 (EN).
- C policy: fixed at the atlas selection. Under balanced L1 at fixed C a factor enters only if its class gap exceeds about 4/(C·n_train), so small subsets are over-penalized (AGENTS.md R5). No null; 3 draws, no CIs.

## Implementation
- Module: scripts/mrd_stage0_2/stage2_supervised/cell_count_learning_curve.py, driven from the D15 section of scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_patient_factor_overlap.ipynb.
- Frozen transductive Stage 1 scores; no Stage 1 refit. Ran on CPU in a Jupyter kernel; the notebook's outputs are not saved (W0.5).

## Artifacts
- analysis/scorecards/stage2_diagnostic/cell_count_learning_curve_v1/
- analysis/figures/stage2_diagnostic/cell_count_learning_curve_v1/

## Results
- [H-S2-06 supports] Full-training refits match the D1 atlas within 0.000164 AUPRC. Apoptosis promax K40 (L2) median lift: 7.65 pp at 500 cells (about 13 training malignant cells), 9.28 at 1,000, 11.52 on the full set.
- [H-S2-06 supports] Allograft promax K40 (EN) median lift is 0 through 2,000 cells, then 1.67 / 3.14 / 8.81 / 12.56 pp at 5k / 10k / 25k / full.
- [H-S2-06 qualifies] IFN-alpha PCA40 and FA40 (L1, C=1e-4) stay at prevalence (median lift 0) at every subsample up to 25,000 cells; on the full set PCA reaches 11.08 pp and FA 0. With C fixed, the L1 entry threshold is about 1.6 SD at 25k cells, so these curves are flat by construction.
- Every draw keeps all 12 training patients; full-training n ranges from 42,992 (P09 held out) to 58,990 (P04 held out), with a median of 1,467 training malignant cells.
- Full curve (cell_count_learning_curve_summary.csv; read back on 2026-10-08 for the narrative guide's D15 section): median training malignant cells 13 / 27 / 53 / 134 / 267 / 668 / 1,467 at 500 / 1k / 2k / 5k / 10k / 25k / full; apoptosis median lift 7.65 / 9.28 / 10.21 / 9.63 / 10.92 / 11.72 / 11.52 pp. Per patient (cell_count_learning_curve.csv, mean over draws) the apoptosis curve is not monotone: P02 32.39 pp at 1k vs 39.80 at full, P03 48.03 vs 47.15, P04 20.67 vs 17.93. At C=1e-4 the entry threshold 4/(C·n_train) is 8 SD at 5k cells, and it falls below PC8's full-fold gap (0.704-0.770 SD, devlog 2026-10-08_team-report-devlog-pivot) only at about 52,000-57,000 training cells.

## Claim gate
Supports (descriptive): on a frozen transductive basis, held-out AUPRC of the Stage 2 probe rises with training-cell depth at a fixed patient mix; apoptosis is nearly saturated by about 1,000 cells.
Does not support: that sequencing another patient would move AUPRC (cell depth only; not a patient-count curve); a Stage 1 learning curve; reading the IFN-alpha flat line as absence of signal (fixed-C all-zero probe, AP = π by construction).

## Next steps
- W6.2: D15 v2 with C retuned per subsample, more draws, and malignant cells per feature on the x-axis.
- W6.1: D15b patient-count curve (drop whole patients) with C retuned per training set.
- W0.5: write the D15 findings md.
