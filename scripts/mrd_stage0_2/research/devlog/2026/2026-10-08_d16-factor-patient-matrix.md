---
id: 2026-10-08_d16-factor-patient-matrix
date: 2026-10-08
type: analysis
title: D16 factor x patient allocation matrix on the transductive PCA/FA bases (41 biological panels, IFN-alpha highlighted)
asked_by: user
asks: [A-20261008-02, A-20261008-05]
implemented_by: claude-code
questions: [Q1, Q2]
workstream: W1
analyses: [D16]
hypotheses: [H-S1-05, H-S1-04, H-S2-08]
verdicts: []
alts: []
closes: [W1.1]
advances: []
claim_level: descriptive
compute: agent-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: pending
run_head: d387eda  # checkpoint commit d387eda landed 19:12 UTC, before the pilot and full runs; the smoke run (scratch EXP) was at b60a6ee
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "DISCUSSED (a): 'Need to inspect the factor-by-patient matrix for FA to test this'" (A-20261008-05)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02)

Source: the user's 2026-10-08 team report, as recorded in research/asks/2026-10-08_team-report.md. Per AGENTS R6, P02/P03/P04 are the highest-prevalence patients, not the ones with the most malignant cells.

## Why it matters
This is the score-space half of H-S1-05 (Q1, Q2). It tests whether the highest-prevalence patients take a disproportionate share of each factor's variance on the stored transductive bases, and it gives D20 (W1.2) and the W2 leave-out refits (D17, W2.1/W2.4) a per-factor baseline to compare against. It also completes the FA factor-by-patient matrix the team asked for. A scratch version existed only for IFN-alpha K40.

## Estimand / null
- Unit: stored Stage 1 representation (panel x method x K, seed 42, transductive) x factor x patient. All 13 patients enter the allocation; the 7 supported patients enter the univariate probe.
- Allocation estimand, per factor:
  - SS_p = sum over patient p's cells of (score - cohort mean)^2;
  - SS share = SS_p / sum over patients of SS_p;
  - enrichment = SS share / cell share. It splits exactly into location (squared standardized patient mean) plus spread (squared patient-to-cohort SD ratio).
  - The same quantities are computed within normal cells only and within malignant cells only, still about the cohort mean.
- Representation level: the SS-weighted share = sum_f SS_pf / sum_f SS_f over non-degenerate factors. P02+P03+P04 is compared with their combined cell share.
- Null / comparator: the no-allocation-bias reference, enrichment = 1 (SS share equals cell share). No permutation; this is descriptive.
- Also computed per factor x patient:
  - within-patient malignant - normal shift, in cohort-SD units and as within-patient pooled-SD Cohen's d;
  - eta2 of patient within normal cells (patients with >= 200 normal cells) and within malignant cells (>= 20 malignant cells). These are the D14 strata parameters, not the support rule; each patient's additive part of eta2 is also stored;
  - degenerate flag (cohort SD < 1e-8) and effective_k from the Stage 1 metadata.
- Univariate probe: held-out AP and lift (AP - held-out prevalence) for one factor at a time, LOPO.
  - The sign is fixed on the training folds by the sign of the training-fold class-mean gap; the training folds are all cells outside the held-out patient.
  - The F6 sign-agnostic max(AP(+s), AP(-s)) is kept for regression only.
- Decision rule fixed before looking: the run is accepted only if the IFN-alpha K40 rows reproduce F6 (abs 1e-9), the scratch anchor (2-decimal rounding) and D14 eta2 (abs 1e-6).
- C policy: not applicable (no Stage 2 fit, no refit).

## Implementation
- Code:
  - scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/factor_patient_matrix.py: pure numpy/pandas. `allocation_table`, `univariate_heldout_table`, `factor_patient_table`, the summaries, and an sklearn-equivalent `average_precision` on pre-sorted scores.
  - scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py: the runner. It writes per-unit parts, then merges, runs the regression checks, draws the figures and writes the findings, manifest and ledger. It supports `--resume` and `--merge-only`, and on a merge failure it writes merge_failed.json.
  - scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.sh: screen wrapper (all panels, --resume, 12 single-thread workers).
- Tests: tests/test_factor_patient_matrix.py, 12 tests: synthetic known-allocation tests plus regression tests gated on EXP.
- Inputs (EXP-relative):
  - stage1_dr/panel_id=*/method={pca,fa}/k={5,10,20,40}/seed=42/scores.npy and metadata.json;
  - stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz (patient, CN.label). Its row order was checked identical to the cohort cache obs;
  - the panel list: analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv (atlas_row_block == biological, 41 panels).
- Commands, all from the repo root in the agent shell; with OMP_NUM_THREADS=1 the CPU use stayed under the 12-thread cap:
  - pilot: `python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py --panels pilot --workers 4` (IFN-alpha, allograft rejection, apoptosis, TNFa/NF-kB; 32 units, 18 s wall);
  - full: `bash scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.sh` (= `--panels all --resume --workers 12`; 288 remaining units, 44 s wall);
  - then `--merge-only` after the figure and findings edits. The parts were unchanged.
- Runtime: the summed per-unit compute is 228.6 s over 320 units. Merge plus figures take about 22 s.
- Provenance: a smoke run on a scratch symlinked EXP ran at b60a6ee. The pilot and full runs ran at d387eda with a dirty tree (factor_patient_matrix.py identical to d387eda; the runner modified). The tree was dirty throughout.
- Reproduce or extend in the user's screen. The run is already complete, so `--resume` only re-merges:
  - `screen -S W1_d16`
  - `bash scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.sh`
  - Ctrl-a d
  - Verify: factor_patient_matrix_regression_checks.csv is all pass, the manifest has units.n_merged = 320, and the 8 D16A-D16D files exist.

## Artifacts
- analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1/:
  - factor_patient_matrix_long.csv.gz (76,310 factor x patient rows)
  - factor_patient_matrix_factor_summary.csv (5,870 factors)
  - factor_patient_matrix_representation_summary.csv (320 representations)
  - factor_patient_matrix_representation_patient_summary.csv
  - factor_patient_matrix_top_patient_frequency.csv
  - factor_patient_matrix_ifna_focus_factors.csv (PC8, FA7, FA8)
  - factor_patient_matrix_regression_checks.csv
  - factor_patient_matrix_findings.md
  - factor_patient_matrix_manifest.json
  - factor_patient_matrix_artifact_ledger.csv
- analysis/figures/stage2_diagnostic/factor_patient_matrix_v1/ (.png and .pdf):
  - D16A_ifna_pca40_fa40_enrichment_heatmap
  - D16B_ifna_pca40_fa40_univariate_lift_heatmap
  - D16C_top_patient_frequency
  - D16D_hp_combined_share_by_panel
- analysis/cache/stage2_diagnostic/factor_patient_matrix_v1/parts/: 320 per-unit csv.gz files (resume state).

## Results
- Coverage: 320 representations = 41 biological panels x {pca, fa} x K {5, 10, 20, 40}. 8 units have no stored scores because the panel has fewer genes than K (class I MHC folding K40; NF-kB activation K20/K40; TNFR1 K40).
  - 5,870 factors in total; 32 are degenerate, all of them FA.
- Regression: 18/18 checks pass.
  - F6: 560/560 rows matched; max abs diff 1.0e-16 (train AP), 1.1e-16 (held-out AP), 1.9e-16 (lift).
  - Scratch anchor: degenerate factors FA37-40; top counts PCA P02 31 / P07 7 / P01 2 and FA P02 27 / P07 7 / P01 2; median enrichment; PC8/FA7/FA8 enrichment and shift, all within 2-decimal rounding.
  - D14 eta2: max abs diff 4.4e-9 (normal), 1.8e-8 (malignant).
- [H-S1-05 qualifies] IFN-alpha K40, the factor-by-patient matrix the team asked for (A-20261008-05):
  - P02 is the top-enrichment patient on 31/40 PCA and 27/36 FA factors; P03 tops none.
  - Combined, P02+P03+P04 are 8.2% of cells and hold 9.1% (PCA40; ratio 1.11) and 8.4% (FA40; ratio 1.02) of the SS-weighted score energy.
  - Normal cells only: 6.6% of normal cells, 7.3% and 6.8% of normal-cell SS (ratios 1.11 and 1.03).
  - Malignant cells hold 3.1% (PCA40) and 2.7% (FA40) of the SS vs 2.6% of cells.
- [H-S1-05 qualifies] Across all 320 representations:
  - The P02+P03+P04 SS-weighted share / cell share has median 0.96 (range 0.58-1.14) and is > 1 in 75/320.
  - Normal cells only: median 0.99 (range 0.62-1.18), > 1 in 138/320.
  - Their combined SS share has median 7.8% (range 4.7-9.3%) against 8.2% of cells.
  - Medians by method x K: PCA 0.92/0.94/0.98/1.02 and FA 0.91/0.94/0.95/0.98 for K 5/10/20/40.
  - Lowest: TNFa/NF-kB PCA5 0.58. Highest: TNFR1 PCA20 1.14 and class I MHC folding FA5 1.13; IFN-alpha PCA40 (1.11) is third.
- [H-S1-05 qualifies] Patients' SS-weighted share / cell share (median over representations; representations > 1):
  - P01 1.24 (279/320), P02 1.06 (233/320), P03 0.91 (66/320), P04 0.89 (20/320), P07 0.90 (91/320), P09 0.92 (53/320).
  - Pooled over 5,838 non-degenerate factors, the top-enrichment patient is P01 30.6%, P02 26.6%, P13 11.7%, P03 10.8%, P07 7.4%, P04 0.7%.
  - Normal cells only: P01 33.8%, P02 26.3%, P03 7.6%, P04 1.0%.
  - The modal top patient per representation is P01 in 174/320, P02 in 82, P13 in 20, P07 in 20 and P03 in 13.
- [H-S1-05 qualifies] Where the enrichment sits:
  - P02's frequent top enrichment is excess spread, not a shifted mean. On the 1,550 factors P02 tops, median enrichment 1.34, median spread part 1.28 and median location part 0.010 (the parts add up per factor, not as medians); the spread part is a median 0.99 of the enrichment, and spread > location on 97% of them.
  - P01 shows the same pattern: on 1,788 factors, spread > location on 100%.
  - P02+P03+P04 per-factor enrichment rises along each fit: factor 1 median 0.91, first half (excl. 1) 1.00, second half 1.08.
  - Within the 238 representations with >= 10 factors, the Spearman rho between a factor's share of the representation's SS and its P02+P03+P04 enrichment has median -0.44 and is negative in 221/238. The excess dispersion sits on low-variance trailing factors.
- [H-S1-04 qualifies] Across all 13 patients, the per-patient medians of SS-weighted share / cell share run from 0.87 (P05) to 1.24 (P01). The patient most over-represented in score energy on the shared bases is P01 (19.7% of cells), not a high-prevalence patient.
- [H-S2-08 supports] With the training-fold sign (not the F6 sign-agnostic max), IFN-alpha FA7 alone gives held-out lift P02 +18.1, P03 +14.4, P04 +19.9, P07 +7.0 and P09 +4.8 pp; PC8 gives P02 +15.9, P03 +13.4, P04 +16.3, P07 +11.1 and P09 +5.5 pp.
  - FA7 enrichment: P02 1.19, P03 1.10, P04 0.85, P07 1.34. PC8: P02 1.15, P03 1.10, P04 1.06, P07 1.64.
  - eta2 normal / malignant: FA7 0.015 / 0.025; PC8 0.010 / 0.031.
- Univariate held-out lift over 40,866 supported factor x patient rows: > 5 pp for 11.9% with the training-fold sign vs 17.4% sign-agnostic. By patient (oriented): P01 0.0%, P02 29.1%, P03 31.7%, P04 16.6%, P06 0.4%, P07 3.8%, P09 1.9%.
- Malignant cells' SS-weighted share of score energy: median 2.2% (range 1.3-3.9%) vs 2.6% of cells. It exceeds the cell share in 57/320 representations.

## Claim gate
Supports (descriptive): on the transductive PCA/FA bases of the 41 biological panels, the highest-prevalence patients P02/P03/P04 hold about their cell share of factor score energy. P02's frequent top enrichment is excess within-patient spread concentrated on low-variance trailing factors. IFN-alpha FA7 alone carries held-out univariate signal for P02 and P07 with a training-fold sign, consistent with the L1-at-fixed-C account (F3, F6, F9).
Does not support:
- that high-prevalence patients shape, or do not shape, the Stage 1 loadings. Score-space allocation is not loading leverage; that needs the D17 leave-out / reweighted refits (W2.1, W2.4);
- any inductive or held-out-basis statement. Stage 1 is transductive, and LOPO here is label transfer on that basis;
- significance (no null distribution was computed);
- a cell-type-adjusted allocation. Composition is not controlled, so excess spread may reflect a patient's cell-type mix;
- a Stage 2 model claim. The univariate lifts are one factor at a time, not L1/L2 probes;
- naming P01/P02 dispersion as biology.

## Next steps
- W2.1 / W2.4: run the D17 leave-high-prevalence-out and patient-balanced refits, then compare loadings by principal angles and matched factor correlation. This is the leverage test H-S1-05 needs; D16 is its score-space baseline.
- W1.2: D20 should take the degenerate flags and the factor SS shares from factor_patient_matrix_factor_summary.csv for its variance weights.
- W1.3: rerun D16 on the W2 bases. The runner currently reads stage1_dr/ only, so it needs an additive `--stage1-root` flag for stage1_dr_variants/<slug>/.
- Proposed new task W1.9 (P2): a cell-type-stratified D16 that computes enrichment within predicted.annotation strata. It would separate patient dispersion (P01, P02) from cell-type composition.
