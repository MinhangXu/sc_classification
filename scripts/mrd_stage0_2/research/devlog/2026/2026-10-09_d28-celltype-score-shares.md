---
id: 2026-10-09_d28-celltype-score-shares
date: 2026-10-09
type: analysis
title: D28 cell-type-stratified score-energy shares per patient on the 320 transductive PCA/FA bases (W1.9)
asked_by: user
asks: [A-20261009-02, A-20261009-13, A-20261009-14, A-20261009-15]
implemented_by: claude-code
questions: [Q1, Q2]
workstream: W1
analyses: [D28]
hypotheses: [H-S1-04, H-S1-05]
verdicts: []
alts: []
closes: [W1.9]
advances: []
claim_level: descriptive
compute: agent-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the D28 runner, its wrapper and its test are new, uncommitted files; the report and overlay round are uncommitted too
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored  # written and run on 2026-10-09
supersedes: []
---

## Question
From the user's 2026-10-09 reading notes on report v1 (recorded verbatim in research/asks/2026-10-09_report-reading-notes.md):

> "Shaping the variance in section 2 "do the highest-prevalence patients shape the factors", I like the 1) and 2). especially 2) is very good control in a perturbation like in-silico experiment in this case test how patients "shape" the factors, though I still need to understand the metrics." (A-20261009-13)

> "I don't understand figure 3, I feel like putting score energy with the malignant cells and all cells are a bit weird, but I feel like maybe just not explained very well. There's score energy, there's ratios? What's 1.11 about? Is it shown in Figure 3? In a new run, let's definitely subset the patients to do the whole stage0-2 pipeline, let's definitely not include patients that're just in the cohort and contribute no malignant cells at least. For figure 4, the axis loss x-axis need more explanation." (A-20261009-14)

> "I see the final conclusion is not very conclusive on this. So definitely try to fix the plots and my feedback above, and I like the next step, let's start them." (A-20261009-15)

The next step started here is report v1 section 2's third bullet, "Stratify the score shares by cell type to test whether P01's excess spread is its cell-type mix (proposed W1.9)", which came from devlog 2026-10-08_d16-factor-patient-matrix (Next steps, W1.9). The question as run (reconstructed from that entry and report v1 section 2): is a patient's excess or deficit of factor score energy, P01's in particular, or the P02+P03+P04 share, explained by its predicted cell-type mix? A-20261009-02 routes here as well. A-20261009-14's run decision (exclude patients with no malignant cells from new runs) does not apply: D28 refits nothing. P02/P03/P04 are the highest-prevalence patients, not the ones with the most malignant cells (AGENTS.md R6).

## Why it matters
D16's score-space result for H-S1-05 (the highest-prevalence patients hold about their cell share of score energy) and its H-S1-04 corollary (the patient most over-represented in score energy is P01) were both reported without a cell-type adjustment, and the D16 claim gate says so. D28 separates patient dispersion from cell-type composition, so report v2 can say in plain terms what the ratio is, what 1.11 means and how much of it a patient's cell-type mix explains. The cell-type energy levels also feed the design of the stratified healthy-cell downsampling the user proposed (A-20261009-12; W2.3, waiting on the new data per the user's decision).

## Estimand / null
- Unit: stored Stage 1 representation (41 biological panels x {PCA, FA} x K {5, 10, 20, 40}, seed 42, transductive; the D16 grid of 320 bases) x unit (13 patients, plus P02+P03+P04 combined).
- Score energy of a cell: the sum over the basis's non-degenerate factors of w_f (score - cohort mean of the factor)^2.
  - Primary, ss_weighted: w_f = 1, the D16 definition (raw scores, so a high-variance factor counts more).
  - Sensitivity, standardized: w_f = 1 / cohort variance of factor f (each factor z-scored first, so every factor counts equally).
- Energy share = the unit's summed energy / the cohort's. Cell share = the unit's cells / all cells. Ratio = energy share / cell share = the unit's energy per cell / the cohort's (1 = proportional). This is D16's weighted_enrichment, reproduced exactly.
- Composition-only expectation: every cell gets the average energy of its predicted cell type (`predicted.annotation`, 13 labels) in a reference set, summed over the unit's cells. Composition ratio = expected share / cell share = the unit's cell-type-mix-weighted average of the types' energy levels.
- Residual ratio = observed energy / composition-only expected energy = ratio / composition ratio (indirect standardization, the analogue of a standardized mortality ratio). 1 = the cell-type mix explains the ratio entirely.
- Fraction of the log departure explained = log(composition ratio) / log(ratio), on bases where the ratio is at least 5% from 1.
- References: cohort (all cells of the type, the unit's own included, as defined above) and loo (the other patients' cells only; a type no other patient has falls back to the cohort average, and those cells are counted).
- Scopes: all cells; normal cells only (energy still about the all-cell cohort mean, as in D16). Strata: cell type; or cell type x malignant label (all cells only).
- Per factor (cohort reference, cell-type strata): observed, composition and residual enrichment, and each factor's top patient before and after the adjustment.
- Null / comparator: residual ratio = 1 (composition explains everything) and the D16 observed ratio (composition explains nothing). Descriptive; no permutation and no interval.
- Decision rule fixed before looking: accept the run only if the observed ratios reproduce D16 (representation, group and per-factor, abs 1e-9), the D16 devlog headlines (2-decimal rounding), and the identities (expected shares sum to 1 under the cohort reference; ratio = composition x residual; residual = weighted within-type ratios), all within 1e-9.
- C policy: not applicable (no Stage 2 fit, no refit; training n unchanged).

## Implementation
- Code:
  - scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.py: the runner and the computation, pure numpy/pandas. It reuses the D16 runner's panel list, score paths and cohort-cache order check, and the D16 module's support rule (imported, unchanged).
    - Each unit stores sufficient statistics as one .npz part: per patient x cell type x label x non-degenerate factor sums of squared deviations, plus counts and factor SDs. Every variant is derived at merge time.
    - Supports `--resume` (empty, corrupt or inconsistent parts count as unfinished), `--merge-only` (a failed merge writes merge_failed.json), `--out-root` (outputs outside EXP, for smoke runs), `--panels pilot|all`, `--panel-ids`, `--methods`, `--ks`, `--workers` (capped at 12), `--no-figures`, `--experiment-dir`.
  - scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.sh: wrapper (all panels, `--resume`, 12 single-thread workers; extra args pass through).
- Tests: tests/test_d28_celltype_score_shares.py, 8 tests, run in the lab env (sc-classification-rapids-py311, where pytest is now installed): composition-only cohorts with known residual 1; a dispersed patient whose residual survives; the group-sum path against a brute-force per-cell implementation (both references, both scopes, both weightings, degenerate factor, loo fallback); identities and consistency with the D16 module; the runner end to end on a fake EXP under `--out-root` (resume, corrupt parts, merge failure); and EXP-gated regressions against the D16 tables and the saved D28 outputs. `python -m pytest -q tests/test_d28_celltype_score_shares.py`: 8 passed. Full suite: 294 passed, 1 skipped. `test_research_index.py::test_real_repo_check` was deselected because it fails on the in-progress overlay round (E14 for H-S2-09, stale STATUS/CLAIMS), not on D28.
- Inputs (EXP-relative): stage1_dr/panel_id=*/method={pca,fa}/k={5,10,20,40}/seed=42/scores.npy and metadata.json; stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz (patient, CN.label, predicted.annotation). The row order matches the cohort cache obs (checked: cell_id, patient and CN.label identical). D16 tables for the regression: analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1/.
- Commands, from the repo root in the agent shell, with OMP/OPENBLAS/MKL threads = 1:
  - smoke: `python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.py --panels pilot --workers 4 --out-root <scratchpad>/d28_smoke` (32 units, 12 s, 15/15 checks pass);
  - full: `bash scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.sh` (= `--panels all --resume --workers 12`; 320 units).
- Runtime: 12.4 s summed unit compute, 41 s wall; merge and figures 38 s. No screen session needed. To rebuild the outputs from the parts: `bash scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.sh --merge-only`.
- Provenance: run at f5a8f4d with a dirty tree; the runner, wrapper and test are new, uncommitted files. No shared code was edited.

## Artifacts
- analysis/scorecards/stage2_diagnostic/factor_patient_matrix_celltype_v1/:
  - factor_patient_matrix_celltype_v1_representation_patient.csv.gz (53,760 rows: 320 bases x 14 units x 12 variants)
  - factor_patient_matrix_celltype_v1_distribution_summary.csv (per unit x variant, over bases)
  - factor_patient_matrix_celltype_v1_method_k_summary.csv
  - factor_patient_matrix_celltype_v1_celltype_long.csv.gz (232,960 rows) and factor_patient_matrix_celltype_v1_celltype_summary.csv
  - factor_patient_matrix_celltype_v1_factor_long.csv.gz (163,464 rows)
  - factor_patient_matrix_celltype_v1_top_patient_before_after.csv
  - factor_patient_matrix_celltype_v1_regression_checks.csv
  - factor_patient_matrix_celltype_v1_findings.md, factor_patient_matrix_celltype_v1_manifest.json, factor_patient_matrix_celltype_v1_artifact_ledger.csv
- analysis/figures/stage2_diagnostic/factor_patient_matrix_celltype_v1/ (.png and .pdf):
  - D28A_ratio_before_after_by_patient: per unit, observed vs composition-only vs residual ratio, median and IQR over the 320 bases; all cells and normal cells only
  - D28B_p01_hp_observed_vs_residual: one point per basis, P01 and P02+P03+P04
  - D28C_celltype_mechanism: cell-type energy levels; each patient's cell-type mix; within-type ratios
- analysis/cache/stage2_diagnostic/factor_patient_matrix_celltype_v1/parts/: 320 .npz files (resume state)

## Results
- Coverage and checks: 320 bases (41 panels; the 8 grid points without stored scores are the same as D16's), 13 cell types, every cell labelled. 17/17 regression checks pass.
  - The observed ratios reproduce D16's weighted_enrichment on 4,160 patient rows (max abs diff 4.6e-15, all cells; 4.4e-15, normal cells), its P02+P03+P04 ratio on 320 rows (1.8e-15), its per-factor enrichment on 75,894 rows (1.1e-13), and its headlines (P01 median 1.24; group median 0.96; IFN-alpha PCA40 1.11 and FA40 1.02).
  - Identities hold within 6.7e-16.
- [H-S1-04 qualifies] P01's over-representation is mostly its cell-type mix: median ratio 1.24 before and residual 1.03 after the cell-type adjustment over 320 bases; composition accounts for a median 90% of the log departure. Primary variant (ss_weighted, cohort reference, cell-type strata):
  - observed median 1.24 (IQR 1.10-1.44; > 1 in 279/320); composition-only 1.21 (IQR 1.08-1.37); residual 1.03 (IQR 1.01-1.05; range 0.90-1.16; > 1 in 259/320);
  - the residual is closer to 1 than the observed ratio in 304/320 bases;
  - on the 291 bases with the ratio at least 5% from 1, composition accounts for a median 90% (IQR 82-96%) of the log departure;
  - normal cells only: 1.24 -> 1.20 -> 1.03.
- [H-S1-04 qualifies] The reference matters for P01, because P01 holds 93% of the ncMono cells and 53% of the pDC2 cells. With the other patients' cells as the reference (loo), P01's residual is 1.08 (IQR 1.02-1.15; > 1 in 270/320) and composition accounts for a median 67% (IQR 52-84%) of the log departure.
- Mechanism (median over bases): cell-type energy per cell relative to the cohort's is
  - ncMono 3.25 (IQR 1.88-5.09), pDC2 1.26 and cDC 1.18 at the top;
  - HSC/MPP2 0.65, MEP-MkP 0.73 and HSC/MPP1 0.76 at the bottom.
  P01's cells are 20.5% pDC2, 14.2% cDC and 8.4% ncMono. P01's ncMono cells carry 1.17x the energy per cell of the other patients' ncMono cells (loo).
- [H-S1-05 qualifies] After the cell-type adjustment the P02+P03+P04 group holds slightly more score energy than its mix predicts: median ratio 0.96 before, residual 1.04 after (> 1 in 230/320 bases). In detail:
  - all cells: observed median 0.96 (> 1 in 75/320), composition-only 0.93, residual 1.04 (IQR 0.99-1.08; range 0.82-1.21; > 1 in 230/320);
  - normal cells only: 0.99 -> 0.96 -> 1.03 (> 1 in 208/320);
  - cell type x label strata: 1.02 (> 1 in 215/320; loo 1.01, > 1 in 183/320).
  Their mix is low-energy (29.1% CLP, 16.3% HSC/MPP2). The residual rises with K: PCA 1.01, 1.02, 1.05, 1.08 and FA 1.01, 1.03, 1.04, 1.05 at K = 5, 10, 20, 40.
- [H-S1-05 qualifies] On IFN-alpha K40 the cell-type mix does not explain the P02+P03+P04 excess: PCA40 1.11 -> composition-only 1.00 -> residual 1.11; FA40 1.02 -> 0.95 -> 1.08 (observed -> composition-only -> residual, all cells):
  - P02+P03+P04, normal cells only: PCA40 1.11 -> 1.04 -> 1.07; FA40 1.03 -> 0.98 -> 1.05.
  - P01: PCA40 0.98 -> 1.06 -> 0.92; FA40 1.10 -> 1.17 -> 0.94.
- P02 alone is not explained by composition: observed median 1.06, composition-only 0.96, residual 1.11 (IQR 1.06-1.17; > 1 in 289/320). On IFN-alpha:
  - P02: PCA40 1.43 -> 1.16 -> 1.23; FA40 1.29 -> 1.03 -> 1.26;
  - P07: PCA40 1.31 -> 0.85 -> 1.54; FA40 1.44 -> 0.91 -> 1.58.
- Top patient per factor (largest enrichment among the 13 patients), before -> after the adjustment, pooled over 5,838 non-degenerate factors:
  - P01 30.6% -> 3.2%; P02 26.6% -> 25.8%; P03 10.8% -> 22.0%; P07 7.4% -> 18.4%.
  - IFN-alpha PCA40: P02 31 -> 1 of 40 and P07 7 -> 38. FA40: P02 27 -> 3 of 36 and P07 7 -> 31.
  - A per-factor ratio is noisier for a small patient, so once a systematic excess is removed the smallest patients win more often. The drop of the former winner is the informative part.
- Sensitivity, standardized weighting (each factor counts equally): P01 1.24 -> residual 1.03 (loo 1.07); P02+P03+P04 1.00 -> 1.05.

## Claim gate
Supports (descriptive): on the transductive PCA/FA bases of the 41 biological panels, most of P01's excess factor score energy per cell matches the energy of its predicted cell types (ncMono, pDC2, cDC). After the same adjustment, the highest-prevalence patients P02/P03/P04 hold a few percent more score energy than their cell-type mix predicts. P02 (and P07 on IFN-alpha K40) carry excess energy that composition does not explain.
Does not support: loading leverage (score-space allocation is not influence on the loadings; D17/D26, W2.4); that cell type causes the energy (predicted labels are transferred calls, and batch, sampling time and patient biology remain in the residual); any inductive or held-out-basis statement (Stage 1 is transductive); significance (no null distribution); a Stage 2 or lift claim; naming the P02 or P07 residual as biology.

## Next steps
- W2.4 (D26): the leave-high-prevalence-out leverage test on allograft rejection and apoptosis remains the H-S1-05 test. D28 adds that, after cell-type adjustment, their score-space share is about 1.04x.
- W2.3: the cell-type energy levels (ncMono about 3x the cohort per cell; HSC/MPP2 about 0.65x) mean that a healthy-cell downsampling stratified by cell type preserves the Stage 1 energy budget only if it keeps the type proportions. Any departure reweights the fit toward or away from these types. Use the D28 celltype tables when the downsampling design is fixed (A-20261009-12; waits on the new data).
- W1.3: rerun D28 on the W2 bases. Both the D16 and D28 runners need an additive `--stage1-root` flag for stage1_dr_variants/<slug>/.
- W1.4 (D25) and W5.3: P02's residual 1.11 and P07's IFN-alpha residual of about 1.5 are not composition. Check sample, batch and label provenance, and the factor x cell type association within those patients.
- Proposed new task W1.10 (P2): locate P02's and P07's within-cell-type excess by factor and cell type (D28 factor and celltype tables), and test whether it tracks sample or collection time.
