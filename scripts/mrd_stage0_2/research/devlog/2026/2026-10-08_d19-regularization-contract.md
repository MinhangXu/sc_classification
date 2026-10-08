---
id: 2026-10-08_d19-regularization-contract
date: 2026-10-08
type: analysis
title: D19 regularization contract - L1 entry-margin audit of the D1 atlas and C-matched IFN-alpha PCA40 vs FA40 (W3.1, W3.2)
asked_by: user
asks: [A-20261008-03, A-20261008-04]
implemented_by: claude-code
questions: [Q3, QI, Q2]
workstream: W3
analyses: [D19]
hypotheses: [H-S2-08, H-S2-07, H-S1-01, H-S2-01, H-S1-02]
verdicts: [H-S2-08=supported_descriptive, H-S2-07=retired]
alts: [H-S2-08:ALT-PEN=addressed]
closes: [W3.1, W3.2]
advances: []
claim_level: descriptive
compute: agent-cpu  # plus a seconds-long synthetic cuML check on one GPU (part cuml_check)
c_policy: per_fold_tuned  # audit part reads the stored global_selected C; W3.2 tunes C per outer fold by inner LOPO
training_n_changed: false
status: done
commit: pending
run_head: d387eda  # HEAD moved from b60a6ee to the d387eda wip checkpoint while this session ran
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "DISCUSSED: IFN-alpha response, PCA K=40 gives lift in four patients but FA K=40 captures only P03 and P04" (A-20261008-03)

> "DISCUSSED (a): hypothesis that PCA orthogonality pushes later components beyond P03, while FA keeps allocating factors to the highest-prevalence patient" (A-20261008-04)

Source: research/asks/2026-10-08_team-report.md (condensed wording of the user's 2026-10-08 team report; capture: paraphrase). The task itself is W3.1 (the regularization contract and the entry-margin audit, which AGENTS R5 and the W9 literature review ask for) plus W3.2 (the C-matched IFN-alpha comparison that settles H-S2-08 and retires H-S2-07).

## Why it matters
Q3 and QI: whether the IFN-alpha PCA40-vs-FA40 coverage gap is Stage 1 structure or a Stage 2 fixed-C L1 threshold (H-S2-08, H-S2-07). Q2: how many atlas L1/EN outcomes sit on the entry knife-edge. The module is the shared C policy (R5) that W2.1, W4.6, W6.1 and W3.3 import, so it unblocks them.

## Estimand / null
- Entry rule (fixed before looking; checked on synthetic data and real solvers): sklearn objective C·sum_i s_i·logloss_i + rho·||w||_1 + (1 - rho)/2·||w||^2 with balanced weights and an unpenalised intercept. At w = 0, sigmoid(b) = 1/2 and the gradient of standardized feature j is (n/4)(mean_normal_j - mean_malignant_j), so j can enter only if |standardized training class-mean gap| > 4·rho/(C·n_train) (rho = 1 for L1, rho = l1_ratio for EN). Entry margin = |gap| / threshold; all margins <= 1 means an all-zero probe (KKT, exact up to solver tolerance).
- Audit unit: (D1 atlas DR cell selected with L1 or EN by the D1 rule) × supported held-out fold, on the stored transductive Stage 1 scores at the stored selected C. Comparator: the stored zero/nonzero outcome (layer4_5 coefficient cache for biological cells; old34_controls_fullgrid_v1 by_heldout_patient.csv for controls). Knife-edge = max entry margin within ±10% of 1. Lift at stake (knife-edge folds) = held-out lift just across the threshold (C × 1.01/margin: the largest-margin factor alone; or C × 1/(1.01·margin): all-zero, lift 0) minus the stored atlas lift.
- W3.2: IFN-alpha PCA40 and FA40, 7 supported outer folds. Arms: stored L1 C = 1e-4 (atlas); L1 refit at 0.9/1.0/1.05/1.1/1.25 × the stored C; L2 at the stored C; L1, L2 and EN (l1_ratio 0.5) with C tuned per outer fold by inner LOPO over the 6 supported training patients (criterion median inner lift; ties to the sparser probe, then the smaller C; full 17-value runner grid). Summaries: pooled AP over the supported held-out cells, median lift and P20 lift, together. Decision rule: H-S2-08 holds if FA40 regains P02/P07 lift once the threshold is crossed or C is tuned; any residual PCA advantage is reported per penalty.
- C policy: audit = stored global_selected C (no refit); W3.2 = per_fold_tuned (nested in C only; method and K fixed). Training n is unchanged (standard LOPO folds).

## Implementation
- Module: scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py (numpy + scikit-learn). API: DEFAULT_C_GRID (runner grid 1/logspace(-4,4,17)), l1_entry_threshold, l1_entry_margins, l1_entry_c, c_scaled_to_n, c_at_grid_lower_bound, kkt_margins, optimal_intercept, supported_patients, tune_c_inner_lopo, fit_probe, probe_coef, probe_scores, heldout_lift, c_policy_record, and ProxNewtonLogistic.
- Solver: L1/EN default to ProxNewtonLogistic ('newton_cd', a proximal-Newton solver of the same objective, warm-started along the C path in tune_c_inner_lopo); L2 uses lbfgs; saga (the runner's CPU backend) stays available as solver='saga'. Reason: saga needed up to 257 epochs (about 8 s per fit) on the IFN-alpha FA40 scores, versus 4-6 Newton steps (about 0.15 s) for the same optimum. Agreement with saga at tol 1e-10 is tested (max coefficient difference < 1e-6 on synthetic data).
- Runner: scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py, parts anchor, audit, ifna, cuml_check, report; `--resume` skips finished (representation, fold) audit units and finished ifna units (empty, corrupt or out-of-date unit files rerun); `--out-root` redirects outputs (smoke runs went to the session scratchpad).
- Command (repo root, lab server): `env CUDA_VISIBLE_DEVICES=0 CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py --parts anchor,audit,ifna,cuml_check,report --resume --n-jobs 12` (drop cuml_check and CUDA_VISIBLE_DEVICES without a GPU).
- Inputs (EXP-relative): stage1_dr/panel_id=*/method=*/k=*/seed=42/scores.npy; stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz; analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv and old34_d1_atlas_supported_patient_metrics.csv; analysis/cache/stage2_diagnostic/layer4_5/selected_lopo_coefficients.pkl.gz; the four LOPO scorecards the atlas app joins (root stage2_sharedness_lopo_scorecard.csv read-only, old34_direct_gene_all_v1, old34_controls_fullgrid_v1, old34_controls_direct_gene_v1); stage2_supervised/multiobjective/old34_controls_fullgrid_v1/sharedness_lopo/by_heldout_patient.csv.
- Ran in the agent shell, 12 joblib workers, one BLAS thread each. First full run: 57.9 s wall (126 ifna units in 28.6 s; 322 CPU-s); its report step failed on a non-UTF-8 write, fixed, then resumed. A later resumed run recomputed the 396 audit units (22.7 s) after the lift-at-stake columns were added; the final report-only run regenerated tables, figures, findings, manifest (with run_history) and ledger.
- Backend check: the stored Set A grids ran on cuML (ml_backend_used = gpu in every row of the root LOPO scorecard). cuML 25.6 sets l1_strength = l1_ratio/C and l2_strength = (1 - l1_ratio)/C with penalty normalisation by n, which is the sklearn objective.
- Provenance: code written in this session; uncommitted (dirty tree on d387eda). After the run, the default L1/EN solver was made to switch to saga above 200 features (direct-gene inputs); a scratch rerun of the anchor and all 126 IFN-alpha units with the final code reproduced every stored lift, C and nonzero count exactly.

## Artifacts
- analysis/scorecards/stage2_diagnostic/regularization_contract_v1/
  - d19_anchor_ifna_l1_entry_check.csv
  - d19_atlas_selected_regularization.csv (783 atlas cells with the D1-rule penalty, C, l1_ratio and grid-bound flags)
  - d19_c_grid_bound_by_cell_group.csv
  - d19_entry_margin_audit_by_fold.csv (2,772 cell × fold rows) and d19_entry_margin_audit_summary.csv
  - d19_ifna_units.csv, d19_ifna_inner_lopo_scores_by_C.csv, d19_ifna_pca40_vs_fa40_by_patient.csv, d19_ifna_arm_summary.csv
  - d19_cuml_semantics_check.csv
  - regularization_contract_v1_findings.md, regularization_contract_v1_manifest.json, regularization_contract_v1_artifact_ledger.csv
- analysis/figures/stage2_diagnostic/regularization_contract_v1/ D19A_entry_margin_distribution, D19B_ifna_l1_c_sweep, D19C_ifna_pca40_vs_fa40_by_c_policy (.png and .pdf)
- analysis/cache/stage2_diagnostic/regularization_contract_v1/ audit_units/ (396 per-representation CSVs) and ifna_units/ (126 JSON + held-out probability .npy)

## Results
- Rule checks: on synthetic data an L1 coefficient (liblinear and saga) is zero at 0.97 × and nonzero at 1.03 × the predicted entry C; elastic net enters at 4·l1_ratio/(C·n) (l1_ratio 0.5 and 0.1), not at 4/(C·n·l1_ratio). cuML, saga (tol 1e-10) and newton_cd give the same nonzero count in 6/6 settings around the entry C (final artifact: max |coef difference| cuML vs saga 2.01e-05; cuML logged two OWL-QN line-search stops, which left the zero/nonzero pattern unchanged), so C means the same on the GPU and CPU backends.
- [H-S2-08 supports] Anchor reproduced 14/14 (IFN-alpha PCA40/FA40 × 7 supported folds, L1, C = 1e-4). The largest-margin factor is PC8 in every PCA fold and FA7 in every FA fold. PC8 margins: P01 0.879, P02 1.085, P03 1.112, P04 1.065, P06 1.074, P07 1.073, P09 0.757. FA7 margins: P01 0.851, P02 0.967, P03 1.018, P04 1.006, P06 0.986, P07 0.981, P09 0.707. The threshold is 0.678 SD with P04 held out (n_train 58,990) and 0.930 SD with P09 held out (n_train 42,992).
- [H-S2-08 supports] L1 sweep: at 1.05 × the stored C, FA7 enters in P02 (lift 18.06 pp, vs PCA40 15.94), P07 (6.97 pp, vs 11.08) and P06 (0.65 pp). PCA40 is unchanged from 1.0 × to 1.25 ×, except P01 (0.23 pp at 1.25 ×). At 0.9 × only PCA40 P03 stays nonzero. The newton_cd refit (sklearn objective) at 1.0 × reproduces the stored cuML per-patient lifts within 1.4e-6 AP (0.00014 pp). The stored, refit, 1.05 × and fixed-L2 arms use the atlas C, which the D1 rule selected by pooled LOPO AUPRC over all folds including each held-out patient, so they are not nested in C; part of the stored-vs-tuned gap can be selection optimism.
- [H-S2-08 supports] With C tuned per outer fold by inner LOPO (6 supported inner patients in every fold, no fallback), FA40 has positive lift in 7/7 supported patients under L1, L2 and EN. P02 FA40 vs PCA40: L1 15.05 vs 16.74 pp; L2 15.97 vs 13.19; EN 14.38 vs 15.94. Summaries, PCA40 vs FA40: tuned L1 pooled AP 0.1001 vs 0.0894, median lift 8.71 vs 7.03 pp, P20 1.37 vs 1.25 pp; tuned L2 0.0871 vs 0.1325, 5.60 vs 6.38, 1.17 vs 1.85; tuned EN 0.1209 vs 0.0891, 10.16 vs 7.40, 1.30 vs 1.21. Under the stored fixed C: median 11.08 vs 0.00 pp, P20 0.01 vs 0.00 pp, lift > 0 in 5/7 vs 2/7.
- [H-S2-08 qualifies] A residual PCA40 advantage in P07 (22 malignant cells) remains under tuned C and depends on the penalty: L1 8.71 vs 4.97 pp, EN 10.16 vs 7.40, L2 5.60 vs 5.07. It is a single-patient difference and not identifiable from one patient. Tuned C also gives P09, all-zero at the stored C, lift of 5.63 (PCA) and 7.03 pp (FA) under L1.
- [H-S2-07 contradicts] FA40 covers P02 and P07 as soon as the stored C is raised by 5% or tuned, through FA7, a direction related to PCA40's PC8 in the shared IFN-alpha subspace (stored K40 scores: r(PC8, FA7) −0.685 and r(PC8, FA8) +0.580; PC8 has many-to-one R² 0.920 on the FA40 scores; D18), not the same axis; the gap needs no FA allocation toward the highest-prevalence patient.
- [H-S1-01 supports] Atlas audit: 396 L1/EN-selected DR cells (351 biological, 45 control) × 7 supported folds = 2,772 folds. Predicted vs stored all-zero/nonzero agreement 2,772/2,772. 136 folds in 49 cells are knife-edge (79 on the all-zero side, 57 just entered), and 134 of them sit in cells whose C is at the grid lower bound. Lift at stake in knife-edge folds: 131 evaluable; |change| >= 1 pp in 83 and >= 5 pp in 41; median 1.75 pp, max 39.06 pp (hallmark hypoxia FA K10, P04 held out, all-zero at the stored C). In 2,015/2,023 nonzero folds the stored support contains the largest-margin factor.
- [H-S2-01 qualifies] 749 of 2,772 audited folds are stored all-zero (constant-score no-skill AP = π by construction), and 748 of them are in cells at C = 1e-4. Most all-zero folds are P09 (120) and P01 (116), whose folds have the smallest training sets. 87 cells are all-zero in all 7 supported folds (80 biological), and 39 mix all-zero and nonzero folds (36 biological). Across the whole atlas, 528/783 cells select C at the grid lower bound (EN 185, L1 72, L2 271); among the L1/EN DR cells the count is 245/396.
- [H-S1-02 qualifies] Pooled AP and patient-level lift rank C policies differently. In IFN-alpha, the fixed-C L1 arms have the highest pooled AP over supported held-out cells (PCA40 0.1630, FA40 0.1959, with 2 and 5 all-zero folds), while the tuned arms have positive lift in 7/7 patients but lower pooled AP (L1 0.1001 and 0.0894). Pooled AP mixes fold models and gives every cell of an all-zero fold one tied score. The D1 rule selects C on pooled LOPO AP, so pooled AP, median lift and P20 lift must stay together in W3.3.
- Stored-solution KKT (2,261 biological folds with at least one zero factor): no zero factor exceeds a margin of 1.02, so the stored cuML fits sit at the KKT point. In 655 nonzero folds the next zero factor is within 10% of entry.

## Claim gate
Supports (descriptive): on the transductive Stage 1 basis over seven supported patients, the IFN-alpha PCA40-vs-FA40 coverage gap at the selected C is a Stage 2 L1 entry-threshold effect on related directions in the shared IFN-alpha subspace (PC8 vs FA7 |r| 0.68; PC8 has R² 0.92 on the FA40 scores), not missing FA structure. FA40 covers the same 5 patients as PCA40 under a 5% C increase, and all 7 under per-fold tuned C; the entry rule reproduces every stored atlas all-zero/nonzero fold outcome.
Does not support: "FA misses patients" as a property of FA; FA7 and PC8 as one identical axis (D18: not individually identifiable); a PCA advantage in P07 beyond a single-patient, penalty-dependent difference; the stored-C arms as nested estimates (their C was selected with held-out outcomes); any inductive or deployment statement (Stage 1 is transductive; LOPO is label transfer); a method ranking from fixed-C atlas cells that the audit flags as knife-edge; a fully nested selection (C only; method and K were fixed after inspecting LOPO results).

## Next steps
- W3.3: D23 nested selection should refit through tune_c_inner_lopo (inner LOPO over method, K and C) rather than reuse stored per-fold metrics, which are not nested in C. By the D19 timing (about 7 CPU-s per cell × fold × penalty with K <= 40) the 736 DR cells × 7 folds × 3 penalties come to about 3 hours with 12 workers. Direct-gene cells (up to thousands of features) default to saga and take longer. This is a screen-cpu job: give the user `screen -S W3_d23` with a --resume runner. Report pooled AP, median lift and P20 lift together, and flag the 49 knife-edge cells.
- W2.1 / W4.6 / W6.1 / W6.2: import the contract (add_study_module_paths(workstreams=("W3_dr_value_regularization",))) and use tune_c_inner_lopo per condition, with L2 alongside; c_scaled_to_n gives the fixed-threshold comparison when training n changes.
- W7.1: the per-fold strip can read d19_entry_margin_audit_by_fold.csv (max_entry_margin, knife_edge_entry, lift_at_stake) for knife-edge badges.
- W5.1: name PC8/FA7 from both poles under the tuned-C probes, not only the fixed-C L1 fits.
- Proposed backlog line W3.8 (P1): per-fold tuned-C re-scoring of the 49 knife-edge atlas cells (L1, L2, EN), done when a table compares the fixed-C and tuned-C pooled AP, median lift and P20 lift (depends: W3.1).
