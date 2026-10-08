---
id: 2026-10-08_d17-basis-variant-pilot
date: 2026-10-08
type: analysis
title: D17 Stage 1 basis-variant pilot on IFN-alpha - inductive, leave-patients-out, normal-only and patient-balanced PCA/FA bases with per-fold tuned C (W2.1)
asked_by: user
asks: [A-20261008-16, A-20261008-14, A-20261008-02]
implemented_by: claude-code
questions: [Q1, Q3, QI]
workstream: W2
analyses: [D17]
hypotheses: [H-S1-06, H-S1-05, H-S1-03, H-S2-08]
verdicts: [H-S1-06=supported_descriptive]
alts: [H-S1-06:ALT-TRANS=partial, H-S1-06:ALT-PEN=addressed, H-S1-03:ALT-TRANS=partial, H-S2-08:ALT-TRANS=addressed]
closes: [W2.1]
advances: [W2.2]
claim_level: descriptive
compute: agent-cpu  # five invocations, each under 5 min wall with 12 single-thread workers (see Implementation)
c_policy: per_fold_tuned  # L1/L2 tuned by inner LOPO per outer fold; L1 and L2 at the stored C = 1e-4 reported alongside
training_n_changed: false  # Stage 2 trains on the standard LOPO folds in every arm; only the Stage 1 fit set changes
status: done
commit: pending
run_head: d387eda  # HEAD had moved from b60a6ee to the d387eda wip checkpoint before this session ran
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "Option 2: exclude the held-out patient from Stage 1 entirely, then project their cells onto the learned factors" (A-20261008-16)

> "Option 1.2: re-weight cells so each patient contributes equally to DR" (A-20261008-14)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02)

Source: research/asks/2026-10-08_team-report.md (condensed wording of the user's 2026-10-08 team report; capture: paraphrase). P02, P03 and P04 are the highest-prevalence patients, not the patients with the most malignant cells (AGENTS R6). The task is W2.1: the IFN-alpha pilot of the W2 Stage 1 basis variants.

## Why it matters
Q1 and Q3 rest on a transductive Stage 1 basis: every LOPO result so far is label transfer on a basis that saw the held-out patient (H-S1-03, ALT-TRANS). This pilot measures how much held-out transfer depends on the held-out patient being in the label-blind fit (H-S1-06), whether the highest-prevalence patients steer the discriminative axis (H-S1-05), and whether the IFN-alpha PCA-vs-FA coverage gap survives non-transductive bases (H-S2-08). It decides whether W2.6 (inductive bases for all 47 representations) is a robustness check or a correction.

## Estimand / null
- Unit: (Stage 1 basis variant) × (supported held-out patient) × (C policy); panel hallmark interferon-alpha response (92 covered genes); PCA and FA at K 10, 20 and 40, seed 42.
- Bases (fit scope in parentheses; every basis standardizes genes on its own fit cells and projects all other cells):
  - transductive_stored (all cells; the stored stage1_dr scores, the anchor) and transductive_refit (all cells, refit with stage1_refit; exact projection);
  - inductive (cells of patients ≠ h, one basis per held-out supported patient h);
  - normal_only (normal cells of patients ≠ h, per h) and normal_only_all (normal cells of all patients; alignment only, since it uses every patient's labels);
  - leave_out (all cells except P02, P03, P04 or P02+P03+P04, and as contrasts P01, P09, P06 or P01+P09), each with 3 size-matched random cell drops (alignment only) as the sampling floor; normal_only_all likewise against 3 random drops of 1,569 cells. Added at review (same day): 20 composition-matched random drops per leave-out set (same number of dropped cells and of dropped malignant cells; seeds 101-120; alignment only) with a one-sided empirical rank of the observed axis movement;
  - balanced_weighted (patient-equal weights over 13 patients: weighted covariance + eigh for PCA; FA fitted to the weighted covariance through moment-matched pseudo-data, same randomized solver, tol rescaled to the fit size) and balanced_subsample (patient-stratified equal-n subsample, 1,128 cells per patient, seeds 42-46).
- Stage 2 per (basis, held-out patient), through regularization_contract: L1 and L2 at C = 1e-4 (the stored atlas C for PCA/FA K20 and K40); L1 and L2 with C tuned per outer fold by inner LOPO over the supported training patients (median inner lift; full 17-value runner grid). Training cells are the standard LOPO folds in every arm, so c_scaled_to_n returns the stored C (recorded per unit; no separate scaled-C arm).
- Alignment to the stored transductive basis, reference axis = the factor with the largest all-cell standardized class-mean gap (PC8 and FA7 at every K): Hungarian matching on |r| of scores, many-to-one R² of the reference axis on all variant factors ("axis loss" = 1 − R², on retained cells for leave-out arms), principal angles between loading subspaces (top 8 and all K) and containment of the reference loading.
- Decision rules fixed before the run:
  - anchor: the stored arm must reproduce the stored sharedness_lopo per-patient lifts at L1/L2 C = 1e-4 before anything else is reported (gate: within 0.1 pp with the same nonzero support; strict flag 0.05 pp; see Implementation for when the 0.1 pp gate was set);
  - H-S1-06 holds descriptively if inductive lift under tuned C and L2 stays positive in the patients where the transductive arm is positive and per-patient deltas are small relative to the median lift;
  - H-S1-05 needs leave-high-prevalence-out axis movement clearly beyond size-matched random drops and beyond the contrast patients.
- Null/comparator: the transductive refit (exact projection, same pipeline) for fit-scope deltas; the stored arm for the anchor; size-matched random drops for leverage.

## Implementation
- Code (repo-relative):
  - scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/stage1_refit.py (module): fit_basis (PCA/FA, train-only standardization, optional patient weights), Stage1Basis (transform = sklearn PCA/FA transform from persisted parameters; save/load of gene order, gene mean/SD, components, model mean, FA noise variance, PCA explained variance), patient_balanced_weights, stratified_subsample, random_drop_mask, fit_scope_mask, weighted_covariance, pseudo_data_from_covariance, fa_from_covariance, score_correlation, hungarian_match, many_to_one_r2, principal_angles_deg, subspace_containment, align_scores.
  - scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py (runner; parts anchor, bases, align, stage2, report; `--resume` per basis and per basis × held-out unit; empty, corrupt or out-of-date files are redone; `--out-root` for smoke runs).
  - scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.sh (screen wrapper, session W2_d17).
  - C policy and probes: scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py (D19), imported via add_study_module_paths(workstreams=("W3_dr_value_regularization", "W2_stage1_basis_variants")).
  - Tests: tests/test_stage1_refit.py (25 tests: toy matrices for fit/project/persist, weighting and alignment; 5 EXP regression tests skipped without EXP).
- Inputs (EXP-relative): stage1_direct_gene/old34_direct_gene_all_v1/panel_id=single_geneset__hallmark_interferon_alpha_response/transform=zscore/features.npz (+ feature_names.json); stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method={pca,fa}/k={10,20,40}/seed=42/{scores,loadings}.npy; stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz; stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv (anchor); analysis/scorecards/stage2_diagnostic/regularization_contract_v1/d19_ifna_units.csv (cross-check).
- Settings: PCA = sklearn PCA(svd_solver='randomized', random_state=42); FA = sklearn FactorAnalysis(svd_method='randomized', random_state=42), the stored Stage 1 settings. An unweighted refit on all cells reproduces the stored FA bases and the stored PCA loadings, but not the stored PCA scores: those are randomized-SVD U·S, while the refit projects (X·V; see Results). The pseudo-data FA fit for the weighted arm matches a fit on the original rows up to the sign of each component. Stage 2: balanced class weights, train-fold StandardScaler, newton_cd for L1 and lbfgs for L2 (contract defaults).
- Commands (repo root, agent shell, lab server; each invocation 12 single-thread joblib workers):
  - `python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py --ks 40 --resume --n-jobs 12` (294.0 s wall: anchor 25 s, 114 bases 23 s, 238 Stage 2 units 228 s);
  - `... --ks 10,20 --parts anchor,bases,align,stage2 --resume --n-jobs 12` (first attempt stopped at the anchor gate in 27.7 s, two rows at 0.052 and 0.068 pp against the then 0.05 pp tolerance; after the two-tier gate, 86.6 s: 228 bases 19 s, 476 units 61 s);
  - `... --ks 10,20,40 --resume --n-jobs 12` after giving the refit arm all four policies (61.7 s: 42 units rerun, everything else resumed), plus report-only reruns (about 22 s each).
  - Smoke: `... --smoke --resume --n-jobs 12 --out-root <scratch>` (37-46 s); the screen wrapper was smoke-tested the same way. `--smoke` now refuses to run without `--out-root`.
  - Review (same day): the 960 composition-matched control bases were fitted into the cache with `... --ks 40 --parts bases --resume --n-jobs 12` (66 s) and `... --ks 10,20 --parts bases --resume --n-jobs 12` (50 s). The align and report parts that would add them to the slug tables were not run in the agent session (see Next steps), so every table and number below predates them. basis_variant_pilot_ifna_v1_findings.md was hand-edited at the review instead: the refit section is renamed as a score-convention difference with the stored-score vs reference r, and the leverage section carries a pending-controls note; the artifact ledger's hash for that file is stale until the report reruns.
- Full reproduction in one go (about 9-10 min with the composition-matched controls, so as a screen job): `screen -S W2_d17`, then `bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.sh`.
- Anchor gate: the stored grids ran on cuML (QN, float32); the contract fits in float64. The 0.1 pp gate (with identical nonzero support) was set after the first K10/K20 attempt; K40 passes the original 0.05 pp level. A float32 replay did not reproduce the two stored values, so they are recorded as solver-level differences.
- Bases persist under analysis/cache/stage2_diagnostic/basis_variant_pilot_ifna_v1/bases/ (layout arm=/panel_id=/method=/k=/seed=/variant=), not stage1_dr_variants/ (R3): this session could write only the three analysis slug dirs. `--bases-location stage1_dr_variants` writes the R3 location.
- Provenance: all code written in this session, uncommitted (dirty tree on d387eda). Alignment uses the projection onto the stored PCA loadings as the reference (see Results); Stage 2 units are annotated with the current alignment at report time and the run stops if a matched factor changed (none did).

## Artifacts
- analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/
  - d17_anchor_check.csv (84 rows: PCA/FA × K 10/20/40 × 7 patients × L1/L2 at C = 1e-4)
  - d17_stage2_by_patient.csv (one row per basis × held-out patient × policy; deltas vs the stored and the refit transductive arms)
  - d17_stage2_summary.csv (pooled AP, median and P20 lift, all-zero folds, deltas, per arm × method × K × policy; seed means for the subsample arm)
  - d17_alignment.csv (342 bases), d17_reference_axes.csv, d17_leverage_contrast.csv, d17_pca_vs_fa_by_arm.csv, d17_bases.csv, d17_d19_tuned_crosscheck.csv
  - basis_variant_pilot_ifna_v1_findings.md, basis_variant_pilot_ifna_v1_manifest.json, basis_variant_pilot_ifna_v1_artifact_ledger.csv
- analysis/figures/stage2_diagnostic/basis_variant_pilot_ifna_v1/ D17A_ifna_lift_by_basis_scope, D17B_ifna_leave_out_leverage, D17C_ifna_inductive_vs_transductive (.png and .pdf)
- analysis/cache/stage2_diagnostic/basis_variant_pilot_ifna_v1/ bases/ (342 bases: basis.npz + basis.json), stage2_units/ (756 unit JSON + held-out probability .npy), d17_match_info.json, stored_sharedness_lopo_ifna_pca_fa_rows.csv (67 MB in total)

## Results
- Anchor: 84/84 stored per-patient lifts reproduced within 0.1 pp with the same nonzero support (L1 and L2 at C = 1e-4); 82/84 within 0.05 pp. K40: 28/28 within 0.05 pp (max |difference| 0.0311 pp). K20 holds the two rows above 0.05 pp: FA20 P03 L1 (0.068 pp; one nonzero coefficient of size 0.0006, a knife-edge fold) and PCA20 P06 L2 (0.052 pp). The tuned-C transductive units equal D19's W3.2 units in 28/28 cases (same C; max |difference| below 1e-13 pp).
- Provenance: the stored PCA scores are sklearn's randomized-SVD U·S, not the projection onto the stored loadings. PC8 correlates 0.9913 (K10), 0.9968 (K20) and 0.9990 (K40) with that projection; FA scores are exact. On the exact projection (transductive refit) at fixed L1, PCA10 PC8 enters the P03 fold (15.97 pp vs stored all-zero), and the single-factor lifts move (P02 8.83 → 11.25 pp, P07 3.97 → 7.01 pp). At K40 the refit-vs-stored difference is at most 0.32 pp (L1 fixed), 0.91 pp (L2) and 1.23 pp (L1 tuned).
- [H-S1-06 supports] Inductive Stage 1 (fit without the held-out patient, training-only standardization), C tuned per fold or L2: lift is positive in 7/7 supported patients for PCA and FA at K 10, 20 and 40 (18 method × K × policy summaries). K40 summaries, pooled AP / median lift / P20 lift:
  - PCA40 tuned L1: inductive 0.1164 / 10.55 / 1.29 pp; transductive refit 0.1153 / 8.91 / 1.42; stored 0.1001 / 8.71 / 1.37.
  - FA40 tuned L1: inductive 0.0869 / 6.69 / 1.42 pp; transductive 0.0894 / 7.03 / 1.25.
  - PCA40 tuned L2: inductive 0.0735 / 6.07 / 1.13 pp vs refit 0.0795 / 6.51 / 1.20; FA40 tuned L2: 0.0794 / 5.21 / 0.94 vs 0.1325 / 6.38 / 1.85.
- [H-S1-03 qualifies] Transductive optimism is small but consistent in sign: per-patient inductive-minus-refit median deltas lie between −1.29 and −0.05 pp across the 18 tuned/L2 summaries (all negative). Examples: PCA40 tuned L2 −0.30 pp (7/7 patients lower), PCA20 tuned L1 −0.54 pp (7/7), FA20 tuned L1 −1.29 pp (6/7), FA40 tuned L1 −0.10 pp (4/7; range −3.83 to +3.31). The largest single-patient drops: FA40 P03 tuned L1 −3.83 pp and FA40 P09 L2 −3.38 pp. The P09-out FA basis matches FA7 at |r| 0.75 (many-to-one R² 0.98).
- [H-S1-06 qualifies] At the fixed C = 1e-4, L1 losses track the entry threshold, not the basis. PCA40 P07 drops from 11.08 pp to all-zero inductively, because the PC8 entry margin falls from 1.073 to 0.928; with tuned C (1e-3) the lift is 10.55 pp. FA20 P03 drops by 12.35 pp the same way (FA7 margin 1.001 → 0.964). An all-zero L1 probe scores the constant-score no-skill AP = π.
- Inductive bases keep the reference axis (K40): PC8 matched |r| 0.97-0.9995 for 6/7 held-out patients, and FA7 0.97-0.9997 for 6/7. The exceptions are the P01-out PCA basis (|r| 0.59, many-to-one R² 0.986) and the P09-out FA basis (|r| 0.75, R² 0.981): the axis survives but rotates into neighbouring factors. Sign orientation: malignant cells score lower on PC8 (all-cell standardized class-mean gap −0.74 at K40; the tuned-L1 PC8 coefficient is negative in every inductive fold) and higher on FA7 (gap +0.68 at K40).
- [H-S1-05 qualifies] Dropping the highest-prevalence patients moves the axis more than dropping the same number of random cells, but by little in absolute terms (K40; axis loss = 1 − many-to-one R² on retained cells, size-matched random mean in parentheses):
  - PCA40: drop P02+P03+P04 0.0015 (0.0003), matched |r| 0.9953, top-8 max principal angle 5.0° (2.2°); P02 0.00037 (0.00008), P03 0.00077 (0.00011), P04 0.00011 (0.00005).
  - FA40: drop P02+P03+P04 0.00062 (0.00014), |r| 0.9958, 5.3° (2.2°).
  - Across method × K, the P02+P03+P04 axis loss is 3.1-5.2 times its size-matched random mean (FA20 3.10, FA10 3.53, PCA20 4.18, PCA10 4.33, FA40 4.48, PCA40 5.19) and above the largest of its 3 random controls in all 6 cells.
  - Caveat: the size-matched controls match the number of dropped cells, not of dropped malignant cells (1,071 for P02+P03+P04, about 128 expected in a random drop of 4,917 cells), and dropping only the 1,569 malignant cells already moves the axis 8.4-17.1 times its random control (normal_only_all rows). The excess can therefore reflect how many malignant cells leave the fit rather than these patients, and 3 random seeds give no calibrated reference. The composition-matched controls added at review answer this once the slug tables are regenerated (Next steps).
- [H-S1-05 qualifies] No blanket ordering between the highest-prevalence and the contrast patients holds:
  - P01 moves the axis far more, absolutely and relative to its size-matched control: PCA40 0.0118 (0.0008), matched |r| 0.58, top-8 angle 35.5° (3.6°); ratio to its control 12.0-21.9 at every method × K except FA40 (1.69). P01+P09 also moves it more in absolute terms (PCA40 0.0156 (0.0031), |r| 0.71).
  - P09 moves it more in absolute terms at every method × K (PCA40 0.0037, FA40 0.0188) but is 3.5 times larger than P02+P03+P04; relative to its control it exceeds them only at FA20 (4.43 vs 3.10).
  - P06 moves it less than P02+P03+P04 at every method × K except FA40 (0.0124 vs 0.00062, 114 times its control, a single outlier), e.g. PCA40 0.00042 vs 0.0015.
  - Stage 2 on the bases without P02+P03+P04 keeps positive lift in 7/7 under tuned L1: PCA40 median 6.31, P20 1.31 pp; FA40 15.08, 1.95 pp.
- [H-S1-05 qualifies] Malignant cells barely shape the axis in absolute terms, although more than random cells of the same number. Normal-only bases carry it: PC8 matched |r| 0.9936 (R² 0.9986) and FA7 0.9937 (R² 0.9993) at K40; axis loss 0.0013 (PCA40) and 0.00065 (FA40) vs 0.00008 and 0.000055 for random 1,569-cell drops. Normal-only inductive bases keep positive lift in 7/7 patients under tuned C or L2 at every K (K40 tuned L1: PCA 0.0983 / 6.98 / 1.36; FA 0.1126 / 7.58 / 1.17). At fixed L1 they are all-zero in 5/7 (PCA40) and 7/7 (FA40) folds.
- [H-S2-08 supports] The PCA40-vs-FA40 coverage gap follows the L1 threshold on every basis. With tuned C or L2, both methods have positive lift in 7/7 patients on the stored, refit, inductive, normal-only and balanced bases. Median PCA − FA under tuned L1: stored +1.69, inductive +3.86, normal-only −0.60, balanced weighted +2.40 pp; under tuned L2: −0.78, +0.86, +0.96, −0.82 pp. At fixed L1 the gap moves with the basis. On the patient-balanced weighted basis, FA40 enters P02 (18.12 pp) and P07 (16.50 pp; FA7 margins 1.016 and 1.014), so FA's median (14.28) exceeds PCA's (8.29).
- Patient-balanced bases (W2.3 input; transductive scope): weighting rotates PC8 more than FA7 (matched |r| 0.90 vs 0.98 at K40; subsample seeds 0.76-0.93 vs 0.96-0.98), with many-to-one R² ≥ 0.98. Stage 2 under tuned L1: PCA40 0.1213 / 9.25 / 2.69 pp, FA40 0.0863 / 6.85 / 1.20 pp; FA40 subsample seed mean 0.0836 / 9.74 / 1.15 pp.

## Claim gate
Supports (descriptive): for IFN-alpha PCA and FA at K 10, 20 and 40 over seven supported patients, held-out lift survives an inductive Stage 1 under per-fold tuned C and L2 (positive in every supported patient, median per-patient loss of 0.05-1.3 pp versus the transductive refit). The discriminative axis of each method (PC8 for PCA, FA7 for FA; related, not identical, directions in the shared IFN-alpha subspace) persists without the held-out patient, without the highest-prevalence patients and without malignant cells. The PCA-vs-FA coverage gap stays a Stage 2 L1 threshold effect on non-transductive bases.
Does not support:
- inductive deployment or raw-expression deployment beyond this panel (W2.6 covers the 47 D1 representations; inner C tuning shares the outer fold's basis);
- that the highest-prevalence patients disproportionately shape the Stage 1 loadings (their leverage on the IFN-alpha axis is small in absolute terms; it exceeds size-matched random drops, but those controls do not match malignant composition; P01 has far more leverage, while P06 and P09 do not consistently have more; one panel only, W2.4 pending);
- causal or significance statements (seven patients; no calibrated test of the leverage excess);
- any reading of fixed-C L1 all-zero folds as "no signal";
- single-patient differences (P07, P09, P06) as identifiable effects;
- PCA K10 atlas lifts as exact (the stored randomized-SVD U·S scores differ from the loading projection);
- cell-level lift as recurrence-risk improvement.
Standard closing line (THEME §6): descriptive results over seven supported patients; they do not establish inductive deployment, causal biology or recurrence risk.

## Next steps
- W2.6: inductive Stage 1 for the 47 D1-selected representations with fold-specific storage (GPU or CPU in user screen). This pilot suggests small, sign-consistent transductive optimism; quantify it across panels with tuned C and L2. Give the user a screen command and add a campaigns.csv row first (R4).
- W2.1 follow-up (agent CPU, about 1 min; bases already fitted): add the composition-matched controls to the slug tables with `python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py --ks 10,20,40 --parts align,report --resume --n-jobs 12`. Expected output: "[align] 1302 bases aligned" and "[report] tables, 6 figure files, findings, manifest and ledger written". Verify that d17_leverage_contrast.csv has n_label_matched_controls = 20 on every leave_out row and that the findings have a "Composition-matched controls" table. Then requote the H-S1-05 bullets from the composition-matched columns in a new entry that supersedes this one.
- W2.4: leave-high-prevalence-out refits for allograft and apoptosis with size-matched AND composition-matched random-drop controls (the D17 leverage design), as the remaining test of H-S1-05.
- W2.3: decide the targeted imbalance before more reweighting; on IFN-alpha, patient-equal weighting rotates PC8 (|r| 0.90) without lowering transfer.
- W2.2: write the bases under stage1_dr_variants/basis_variant_pilot_ifna_v1/ (refit with `--bases-location stage1_dr_variants`, or copy with a manifest) and leave the analysis/cache copy in place (R10); add a Stage 2 loader and FactoSig psi/mu. stage1_refit already persists gene order, mean/SD, components and FA noise variance/mean.
- W1.3: rerun D14 on the inductive and normal-only bases from this run.
- W2.7 (new backlog line, added to the W2 README at integration): audit the stored randomized-SVD PCA scores (U·S) against the exact projection onto the stored loadings across the D1 atlas, and flag representations whose stored-score Stage 2 outcome changes. PCA10 IFN-alpha changes at fixed L1 in 4/7 folds by more than 1 pp.
- Screen reproduction of the whole pilot (about 7-8 min): `screen -S W2_d17`, then `bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.sh`. Expected: "[anchor] 84/84 ... reproduced", "[stage2] 756 units", "[done]". Verify that d17_anchor_check.csv has pass=True in every row and that the findings file is regenerated.
