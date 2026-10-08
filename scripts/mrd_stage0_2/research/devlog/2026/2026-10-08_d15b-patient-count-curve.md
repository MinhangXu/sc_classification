---
id: 2026-10-08_d15b-patient-count-curve
date: 2026-10-08
type: run
title: D15b patient-count learning curve with C retuned per training set (reduced design run; full enumeration handed to screen W6_d15b)
asked_by: user
asks: [A-20261008-25, A-20261008-26]
implemented_by: claude-code
questions: [Q2, meta]
workstream: W6
analyses: [D15b]
hypotheses: [H-S2-06, H-S2-08]
verdicts: []
alts: [H-S2-06:ALT-PEN=partial, H-S2-06:ALT-NPAT=partial]
closes: []
advances: [W6.1]
claim_level: descriptive
compute: screen-cpu  # reduced design and smoke ran in the agent shell (12 workers); the full enumeration is the user's screen run
c_policy: per_fold_tuned  # tuned once per training set by inner LOPO; fixed atlas C and c_scaled_to_n are reference arms
training_n_changed: true
status: partial
commit: pending
run_head: d387eda  # HEAD was the d387eda wip checkpoint (one commit after b60a6ee) when this ran
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "Learning curve useful for grants to justify sequencing more patients" (A-20261008-25)

> "Randomly subsample to estimate the curve (as Audit AI does with Eclipse)" (A-20261008-26)

Source: research/asks/2026-10-08_team-report.md. These are that file's item texts, which condense the user's 2026-10-08 report (capture: paraphrase). The task is W6.1: D15b, the patient-count version of D15, with C retuned for every training set (AGENTS R5).

## Why it matters
Q2 and the grant argument (meta). D15 is a cell-depth curve at a fixed 12-patient training mix with C fixed, so it cannot speak to adding patients (H-S2-06 gate). D15b varies the number of training patients and retunes C, so it separates the fixed-C L1 collapse (H-S2-08, ALT-PEN) from how label transfer actually changes with patients. It also feeds the W6.4 grant note on what these curves can and cannot justify.

## Estimand / null
- **Unit.** For each held-out supported patient h and each subset S of size k of the other 6 supported patients, the Stage 2 probe is trained on S plus all 6 unsupported patients (P05, P08, P10-P13; always included) and scored on h.
  - k = 6 is the ordinary LOPO fold, so it reproduces the D1 atlas and D15 full-training point.
  - Estimand per (h, k): held-out lift = AP − π, averaged over the subsets.
  - Estimand per k: the median, P20 and patient-equal mean over held-out patients of those per-(h, k) means.
  - Paired gain: per held-out patient, the mean lift at k minus the mean lift at k = 2 (primary: inner-LOPO tuning at both ends). The gain over k = 1 is kept for reference only, because at k = 1 the tuned arms switch protocol (stratified 5-fold CV over cells instead of inner LOPO; see below).
  - Patient-inclusion effect (descriptive): at matched k, the mean lift of subsets that contain patient j minus those that do not.
- **Basis.** The frozen transductive Stage 1 basis: the stored scores.npy, fit on all 60,118 cells including h and the excluded patients. Every subset's basis has therefore seen every patient. The curve isolates Stage 2 label transfer.
- **C policies (arms).** All arms go through regularization_contract (contract 1.0.0).
  - tuned (headline): for each training set, penalty in {L1, L2} and C from tune_c_inner_lopo. Inner LOPO runs over the supported training patients; k = 1 uses the contract's stratified 5-fold CV fallback. The criterion is median inner lift; ties go to L1.
  - tuned_l1 and tuned_l2: the same tuning with the penalty held fixed.
  - fixed_atlas: the D1 atlas penalty and C, as in D15.
  - scaled_to_n: the atlas penalty with C = c_scaled_to_n(C_atlas, n of h's full LOPO fold, n_train). This keeps the full-fold L1 entry threshold 4ρ/(C·n) fixed.
  - Tuning depends only on the training set, so it ran once per (representation, S, penalty).
- **Representations.** The four D15 representations, K = 40:
  - IFN-alpha PCA40 and FA40 (atlas L1, C = 1e-4);
  - apoptosis factosig_promax K40 (atlas L2, C = 1e-4);
  - allograft factosig_promax K40 (atlas EN, l1_ratio 0.5, C ≈ 4.64e-4).
- **Uncertainty and decision rules, fixed before looking.**
  - Patient-cluster bootstrap: resample the 7 held-out patients, 2,000 draws, percentile 95% CI.
  - A weighted inverse power law, a − b·k^(−c) with weights 1/max(bootstrap SE, 0.25 pp)², is reportable only if all of these hold for the primary validation (fit k ≤ 4, predict k = 5):
    - the fit converges;
    - no parameter sits at a bound;
    - the fitted rise is at least 0.5 pp;
    - the prediction falls inside the observed k = 5 CI.
  - k ≤ 5 → 6 is a secondary validation only.
  - Added at review (same day, before any claim used it): the reduced design's k = 5 target is unbalanced (see Designs), so no power law is reportable from the reduced design at all; the rule applies to the full design.
- **Null.** No permutation null. The comparators are k = 2 (paired gain; k = 1 for reference) and the fixed-C arm. fixed_atlas is not nested: its C was selected by the D1 rule with held-out outcomes.

## Implementation
- **Code** (new, W6 workstream dir):
  - scripts/mrd_stage0_2/workstreams/W6_learning_curves/patient_count_curve.py: design, summaries, bootstrap, power law.
  - scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py: runner with parts tune, units and report.
  - scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh: screen wrapper for the full design.
  - The runner imports regularization_contract (W3) through paths.add_study_module_paths.
- **Inputs** (EXP-relative):
  - the stage1_dr/.../k=40/seed=42/scores.npy files of the four representations (float32, as in D19);
  - stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz (y_true, patient);
  - analysis/scorecards/stage2_diagnostic/old34_representation_diagnostic_table.csv (atlas penalty and C);
  - three reference tables: old34_d1_atlas_supported_patient_metrics.csv, cell_count_learning_curve_v1/cell_count_learning_curve.csv and regularization_contract_v1/d19_ifna_pca40_vs_fa40_by_patient.csv.
- **Designs.**
  - full: every subset, 63 per held-out patient (k = 1..6), 441 outer units and 126 distinct training subsets per representation.
  - reduced (ran here): a deterministic cyclic family of 7 subsets per size over the sorted supported patients, with base blocks {0}, {0,1}, {0,1,3}, {0,1,2,4}, {0,1,2,3,4} and all 6. Sizes 3 and 4 are balanced incomplete block designs. Each held-out patient uses the 7 − k members that exclude it, giving 147 outer units and 42 subsets per representation.
  - k = 1 and k = 6 are complete in the reduced design, and k = 3 and 4 are balanced over training-patient membership. k = 2 and k = 5 are not: the two patients next to the held-out one in sorted order enter 1/5 (k = 2) and 1/2 (k = 5) of its subsets, against 1/3 and 5/6 in the full enumeration (for held-out P02, P03 is in 1 of 5 k = 2 subsets). Reduced-design k = 2 and k = 5 means, including the k = 2 baseline of the primary paired gain, therefore average over a skewed training composition; the full design removes this.
  - The reduced units are a subset of the full units, and the full run resumes from them.
- **Commands run** (repo root; python = the sc-classification-rapids-py311 interpreter with CONDA_PREFIX set; OMP/BLAS threads 1 per worker, 12 workers):
  - Smoke: `python scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py --smoke --out-root <scratch>/d15b_smoke --n-jobs 12`. This is held-out P07, k ∈ {1, 2}, allograft promax K40; 42 tune units and 21 outer units in 46 s wall. A first report crashed on an empty CSV (fixed), and the `--resume` rerun skipped all units.
  - Reduced, step 1: `python scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py --design reduced --parts tune --resume --n-jobs 12`, 336 tune units in 277 s wall.
  - Reduced, step 2: the same command with `--parts tune,units,report`, 588 outer units in 47 s plus a report in 26 s. The report was regenerated twice after findings and figure edits; units were unchanged.
  - Review (same day): `--design reduced --parts report --resume --n-jobs 12` regenerated the report (31 s) after the findings gained the k = 2 paired gain, the k = 1 protocol and reduced-design balance notes, the fixed_atlas not-nested label, and dropped an unsupported 'lower bound' sentence. Units were unchanged; the 84 reference checks still pass.
  - Determinism check: held-out P02 and P07, both IFN representations, recomputed from scratch into `--out-root <scratch>`. All 420 unit × arm rows were identical: lift difference 0.0, same C, same nonzero counts.
- **Full-design cost.** From the reduced run's per-unit timings (d15b_reduced_timing_model.csv), the full design costs about 10,260 CPU-s, about 14 min wall at 12 workers with perfect packing. That exceeds the agent-shell limit (R1), so the full enumeration is handed to the user in screen.
- **Screen command (full design, CPU, resumable).**
  - From the repo root:
    ```
    screen -S W6_d15b
    bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh
    ```
    then detach with Ctrl-a d.
  - The wrapper runs `run_d15b_patient_count_curve.py --design full --resume --n-jobs 12` and tees a log to analysis/cache/stage2_diagnostic/patient_count_learning_curve_v1/logs/.
  - Expected work: 672 new tune units (1,008 total; 336 cached) and 1,176 new outer units (1,764 total; 588 cached), roughly 10-15 min wall. Rerunning the same command after an interruption resumes.
  - Verify that d15b_full_unit_status.json shows n_units_present 1764 and n_units_missing 0, that d15b_full_reference_checks.csv passes on every row, and that the findings md has a "full design" section.

## Artifacts
- analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/:
  - d15b_reduced_design.csv
  - d15b_reduced_unit_results.csv (unit × arm)
  - d15b_reduced_tuning_by_subset.csv
  - d15b_reduced_tuning_scores_by_C.csv
  - d15b_reduced_by_patient_k.csv
  - d15b_reduced_curve_by_k.csv
  - d15b_reduced_gain_vs_k2.csv (primary paired gain) and d15b_reduced_gain_vs_k1.csv (reference)
  - d15b_reduced_patient_inclusion_effect.csv
  - d15b_reduced_power_law_fit.csv
  - d15b_reduced_reference_checks.csv
  - d15b_reduced_timing_model.csv
  - d15b_reduced_unit_status.json
  - patient_count_learning_curve_v1_findings.md
  - patient_count_learning_curve_v1_manifest.json
  - patient_count_learning_curve_v1_artifact_ledger.csv
- analysis/figures/stage2_diagnostic/patient_count_learning_curve_v1/, each as .png and .pdf:
  - D15bA_reduced_median_lift_vs_k_by_C_policy
  - D15bB_reduced_lift_vs_training_malignant_by_patient
  - D15bC_reduced_power_law_validation
  - D15bD_reduced_patient_inclusion_effect
- analysis/cache/stage2_diagnostic/patient_count_learning_curve_v1/ (tune/ and units/ JSON caches, 924 files)
- Pending from the screen run: the same tables and figures with the d15b_full_ / _full_ prefix.

## Results
All numbers are from the reduced design on the frozen transductive basis; lift is in percentage points (pp).

**Reproduction at k = 6** (the ordinary LOPO fold): 84 of 84 checks pass.
- fixed_atlas vs D1 atlas (cuML) held-out AUPRC: 28/28, max |Δ| 1.64e-4.
- fixed_atlas vs D15 full-training refit: 28/28, max |Δ| 1.89e-5.
- tuned_l1 and tuned_l2 vs D19 tuned: 14/14 each, same C, max |Δ lift| 9.6e-17.

**Headline curve, tuned arm.** Median over held-out patients of the per-(h, k) mean lift, for k = 1..6:

| representation | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | P20 range across k |
|---|---|---|---|---|---|---|---|
| IFN-alpha PCA40 | 4.63 | 6.24 | 8.36 | 12.83 | 7.68 | 8.71 | 1.06-1.61 |
| IFN-alpha FA40 | 4.45 | 4.32 | 5.69 | 7.38 | 5.76 | 7.03 | 1.20-1.96 |
| apoptosis promax K40 | 5.34 | 8.77 | 7.27 | 10.72 | 10.10 | 10.26 | 1.25-2.47 |
| allograft promax K40 | 8.79 | 7.06 | 8.13 | 9.03 | 10.31 | 10.12 | 2.23-3.54 |

- Patient-bootstrap 95% CIs of these medians are wide. At k = 1 they are PCA [1.26, 10.38], FA [1.59, 9.31], apoptosis [0.40, 12.03] and allograft [0.92, 18.33].
- Median training malignant cells are 254 / 471 / 740 / 982 / 1,160 / 1,467 at k = 1..6. Per unit the range is 34-770 at k = 1 and 811-1,547 at k = 6.

**Evidence bullets.**
- [H-S2-06 qualifies] With C retuned per training set, the patient-count gain is small next to the spread between patients. Primary paired gain k = 2 → 6, tuned arm (inner-LOPO tuning at both ends; the reduced k = 2 baseline is unbalanced, see Designs), median over patients [95% CI], patients with gain > 0, mean [95% CI]:

  | representation | median gain [95% CI] | patients > 0 | mean gain [95% CI] |
  |---|---|---|---|
  | PCA40 | 2.16 [0.20, 2.48] | 6/7 | 1.95 [0.72, 3.43] |
  | FA40 | 0.75 [-0.64, 4.67] | 5/7 | 1.86 [-0.91, 4.93] |
  | apoptosis | 1.49 [-0.65, 4.57] | 5/7 | 3.49 [0.51, 7.72] |
  | allograft | 3.06 [0.02, 4.43] | 6/7 | 2.29 [0.25, 4.28] |

  For reference only, the gain k = 1 → 6, which mixes the k = 1 tuning-protocol switch with the patient-count effect (all 56 k = 1 tuning runs used the stratified cell-CV fallback; median tuned C at k = 1 is 0.0208 for L1 and 0.0032 for L2, against 0.0010-0.0021 and 0.0001-0.0003 at k = 2-6):

  | representation | median gain [95% CI] | patients > 0 | mean gain [95% CI] |
  |---|---|---|---|
  | PCA40 | 1.67 [0.08, 5.74] | 6/7 | 2.69 [0.67, 4.81] |
  | FA40 | 1.62 [-1.27, 7.35] | 4/7 | 2.45 [-0.92, 6.34] |
  | apoptosis | 3.87 [-0.34, 4.92] | 5/7 | 6.12 [0.67, 14.71] |
  | allograft | 0.72 [-0.07, 9.43] | 5/7 | 2.67 [-0.60, 6.25] |

  The curves are not monotone in k; for example, the PCA40 median is 12.83 at k = 4 and 7.68 at k = 5.
- [H-S2-06 qualifies] No inverse power law is reportable. The primary validation (fit k ≤ 4 → predict k = 5) fails for all four representations on the tuned median: a parameter sits at a bound in all four, and the allograft fit also has no rise. The reduced design's k = 5 target is unbalanced in any case, so the reduced design cannot validate a power law (the findings now report none by rule).
- [H-S2-08 supports] At the fixed atlas C = 1e-4, IFN-alpha FA40 probes are all-zero in every unit at k = 1-4, and in 0.86 and 0.71 of units at k = 5 and 6. Its median lift is 0.00 at every k, the same flat line as D15, by construction. With C tuned, FA40 has no all-zero unit and a median lift of 4.32-7.38 at every k. PCA40 at fixed C is all-zero in every unit at k = 1 and 3.
- [H-S2-06 qualifies] Lift stays concentrated in the highest-prevalence patients at every k. Tuned mean lift, k = 1 → k = 6:
  - PCA40: P02 15.07 → 16.74, P03 10.38 → 15.57, P04 9.14 → 15.48;
  - FA40: P02 19.66 → 15.05, P03 9.31 → 16.66, P04 7.24 → 18.93;
  - P01 and P06 stay below 2.5 for every representation and k.

**Other results.**
- Tuned vs fixed at k = 6. The tuned median is below the fixed-atlas median for PCA40 (8.71 vs 11.08), apoptosis (10.26 vs 11.52) and allograft (10.12 vs 12.56), and above it for FA40 (7.03 vs 0.00). The atlas C was selected on pooled LOPO AP of the same folds; the tuned C is nested.
- Penalty and grid bounds.
  - At k = 1 the tuned arm picks L2 in 42/42 FA40, 36/42 PCA40, 36/42 apoptosis and 30/42 allograft units.
  - Tuned L2 C sits at the grid lower bound (1e-4) in 71% (FA40), 64% (PCA40), 48% (apoptosis) and 12% (allograft) of tuning subsets.
  - All 56 k = 1 tuning runs used the fallback with label-only strata, because the patient × label strata were too small for 5 folds.
- Patient-inclusion effect, tuned arm (descriptive; median over held-out patients; reduced design, so not balanced at k = 2 and 5):
  - including P02: +6.12 allograft, +1.35 PCA40, +0.77 apoptosis, +0.61 FA40;
  - including P09 (the large, low-prevalence patient): −4.55 apoptosis, −2.98 allograft, −0.53 FA40, −0.02 PCA40.

## Claim gate
Supports (descriptive): on a frozen transductive Stage 1 basis, with C and penalty retuned per training set, held-out label transfer for the four D15 representations is already positive with one supported training patient. The median gain from two to six training patients (same tuning protocol at both ends) is a few pp at most, the CIs are wide (7 patient clusters), and the curves are not monotone. The flat IFN-alpha curves at fixed C are all-zero L1 collapse, not absence of signal.
Does not support:
- that sequencing more patients would raise AUPRC (the basis saw every patient, and nothing is extrapolated to new patients); how a basis refit per training subset would change the curve is unknown in either direction (W6.6);
- gains over k = 1 as a patient-count effect (the tuned arms switch from cell-CV to inner-LOPO tuning between k = 1 and k = 2);
- reduced-design k = 2 and k = 5 points as balanced estimates;
- any power-law extrapolation (the primary validation failed);
- inductive deployment;
- a recurrence-risk statement;
- a curve for the full enumeration (that is pending in screen W6_d15b).

## Next steps
- W6.1: the user runs the full design in screen W6_d15b (command and checks under Implementation). A follow-up entry with type run and status done supersedes this one, quotes the d15b_full_* numbers and closes W6.1. The integrator should set the D15b row of research/analyses.csv to current or partial and add a campaigns.csv row (planned, resume_command = the wrapper).
- W6.4: grant note. What would justify sequencing more patients:
  - an inductive Stage 1 basis refit on each training subset, or at least without the held-out patient (W6.6, with the W2 refit code);
  - the same patient-count curve on that basis;
  - new patients (Yu-Hsiang's cohort) scored as external points against the curve's prediction (W6.5, W8.1).
- W6.6 (proposed new backlog line): D15b on an inductive basis, comparing transductive and inductive curves on the same subsets (depends: W6.1, W2.1).
- W6.2: D15 v2 cell-depth curve with the same per-training-set tuning (tune_c_inner_lopo per subsample).
- W6.3: a 2D design (k × a cap on malignant cells per patient) to separate patient count from malignant-cell count, which the subsets confound (P03 and P09 carry 758 and 375 malignant cells).
