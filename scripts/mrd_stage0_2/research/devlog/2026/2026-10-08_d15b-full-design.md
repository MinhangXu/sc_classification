---
id: 2026-10-08_d15b-full-design
date: 2026-10-08
type: analysis
title: D15b patient-count learning curve, full design (all 63 training subsets per held-out patient, C retuned per training set) completed in screen W6_d15b
asked_by: user
asks: [A-20261008-24, A-20261008-25, A-20261008-26]
implemented_by: claude-code
questions: [Q2, meta]
workstream: W6
analyses: [D15b]
hypotheses: [H-S2-06, H-S2-08]
verdicts: []
alts: []
closes: [W6.1]
advances: []
claim_level: descriptive
compute: screen-cpu  # the user's screen session W6_d15b (12 CPU workers); this entry was written from its outputs
c_policy: per_fold_tuned  # penalty and C tuned once per training set by inner LOPO (contract 1.0.0); fixed_atlas and scaled_to_n are reference arms
training_n_changed: true
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # manifest git_dirty true; git status showed only the untracked report directory
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: [2026-10-08_d15b-patient-count-curve]  # completes the reduced-design run entry (AGENTS.md §7); its Results stay valid for the reduced design
seq: 1  # same-day order: after the reduced-design run entry it completes
---

## Question
> "Plot AUPRC vs number of cells (learning / marginal-benefit curve) ('we can do this for sure, right?')" (A-20261008-24)

> "Learning curve useful for grants to justify sequencing more patients" (A-20261008-25)

> "Randomly subsample to estimate the curve (as Audit AI does with Eclipse)" (A-20261008-26)

Source: research/asks/2026-10-08_team-report.md (condensed wording of the user's 2026-10-08 team report; capture: paraphrase). This entry completes W6.1: the full enumeration of the D15b patient-count curve that the reduced-design entry 2026-10-08_d15b-patient-count-curve handed to the user's screen session.

## Why it matters
Q2 and the grant argument (meta; H-S2-06). The reduced design was unbalanced at two and five training patients, so its primary paired gain (k = 2 baseline) and its power-law validation target (k = 5) were not usable. The full design removes that imbalance and gives the patient-count numbers the W6.4 grant note and the 2026-10-08 questions report (section 07) quote.

## Estimand / null
- Unchanged from 2026-10-08_d15b-patient-count-curve (estimand, arms, bootstrap, power-law rule fixed before looking); only the design changes from the cyclic reduced family to every subset.
- Unit: for each held-out supported patient h and each subset S of size k of the other 6 supported patients (k = 1..6; 63 subsets per h), the Stage 2 probe is trained on S plus all 6 unsupported patients and scored on h. 4 representations × 7 held-out patients × 63 subsets = 1,764 units; 126 distinct training subsets per representation.
- Estimand per (h, k): mean held-out lift (AP − π) over subsets. Per k: median, P20 and patient-equal mean over held-out patients. Primary comparison: paired gain over k = 2 (inner-LOPO tuning at both ends); gain over k = 1 kept for reference only (k = 1 tunes by stratified 5-fold cell CV).
- Basis: the frozen transductive Stage 1 scores (fit on all 60,118 cells, including h).
- C policy: tuned arm = penalty in {L1, L2} and C from 1/logspace(−4, 4, 17) by inner LOPO over the supported training patients (criterion: median inner lift; ties to L1), through regularization_contract 1.0.0; tuned_l1, tuned_l2, fixed_atlas (D1 atlas penalty and C; not nested) and scaled_to_n as reference arms.
- Uncertainty: patient-cluster bootstrap over the 7 held-out patients, 2,000 draws, percentile 95% CI. Inverse power law a − b·k^(−c) reportable only if, fitted on k ≤ 4 (weights 1/max(bootstrap SE, 0.25 pp)²), it converges, has no parameter at a bound, rises by at least 0.5 pp and predicts k = 5 inside the observed CI.
- Null: no permutation null; comparators are k = 2 (paired gain) and the fixed-C arm.

## Implementation
- Code (unchanged since the reduced run; committed in f5a8f4d):
  - scripts/mrd_stage0_2/workstreams/W6_learning_curves/patient_count_curve.py
  - scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py
  - scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh (wrapper: `--design full --resume --n-jobs 12`, 1 BLAS thread per worker)
- Where it ran: the user's screen session, from the repo root:
  ```
  screen -S W6_d15b
  bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh
  ```
  Started 2026-10-08 23:26 UTC (log d15b_full_20261008T232605Z.log); finished 23:35 UTC, exit 0.
- Work done: 672 new tune units (336 resumed from the reduced run; 1,008 total) and 1,176 new outer units (588 resumed; 1,764 total). Wall time 565.6 s (tune 448.85 s, units 85.54 s, report 30.94 s).
- Completion checks: d15b_full_unit_status.json has n_units_present 1764 and n_units_missing 0; d15b_full_reference_checks.csv passes 84 of 84 rows; the findings md has a "full design" section.
- Section 07 of the questions report (scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/, figs_07_curves.py) reads the full-design tables; the per-patient gains and CI widths quoted below were read from d15b_full_by_patient_k.csv and d15b_full_curve_by_k.csv with pandas (no recomputation).

## Artifacts
- analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/:
  - d15b_full_design.csv, d15b_full_unit_results.csv, d15b_full_unit_status.json
  - d15b_full_tuning_by_subset.csv, d15b_full_tuning_scores_by_C.csv
  - d15b_full_by_patient_k.csv, d15b_full_curve_by_k.csv
  - d15b_full_gain_vs_k2.csv (primary), d15b_full_gain_vs_k1.csv (reference)
  - d15b_full_patient_inclusion_effect.csv, d15b_full_power_law_fit.csv
  - d15b_full_reference_checks.csv, d15b_full_timing_model.csv
  - patient_count_learning_curve_v1_findings.md (now with a full-design section), _manifest.json, _artifact_ledger.csv (regenerated by the full run)
- analysis/figures/stage2_diagnostic/patient_count_learning_curve_v1/: D15bA_full_median_lift_vs_k_by_C_policy, D15bB_full_lift_vs_training_malignant_by_patient, D15bC_full_power_law_validation, D15bD_full_patient_inclusion_effect (.png and .pdf)
- analysis/cache/stage2_diagnostic/patient_count_learning_curve_v1/ (tune/ and units/ caches; logs/d15b_full_20261008T232605Z.log)

## Results
All numbers are from the full design on the frozen transductive basis; lift in percentage points (pp).

**Reproduction at k = 6**: 84 of 84 reference checks pass (fixed_atlas vs D1 atlas held-out AUPRC 28/28, max |Δ| 1.64e-4; vs D15 full-training refit 28/28, max |Δ| 1.89e-5; tuned_l1 and tuned_l2 vs D19 14/14 each, max |Δ lift| 9.6e-17).

**Headline curve, tuned arm.** Median over held-out patients of the per-(h, k) mean lift, k = 1..6 [k = 1 and k = 6 95% CI]:

| representation | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | P20 range across k |
|---|---|---|---|---|---|---|---|
| IFN-alpha PCA40 | 4.63 [1.26, 10.38] | 5.88 | 7.83 | 9.30 | 8.77 | 8.71 [0.30, 15.57] | 1.18-1.61 |
| IFN-alpha FA40 | 4.45 [1.59, 9.31] | 4.77 | 5.51 | 6.66 | 6.20 | 7.03 [0.32, 16.66] | 1.27-1.96 |
| apoptosis promax K40 | 5.34 [0.40, 12.03] | 8.68 | 8.27 | 9.96 | 10.31 | 10.26 [2.19, 39.80] | 1.26-2.25 |
| allograft promax K40 | 8.79 [0.92, 18.33] | 7.63 | 8.13 | 8.63 | 9.32 | 10.12 [1.64, 27.76] | 2.22-3.29 |

- Median training malignant cells: 254 / 497 / 740 / 982 / 1,224 / 1,467 at k = 1..6.
- Width of the 95% CI of each median, across representations and k: 7.7 to 37.6 pp. The k = 2 → 6 rise of the median curves is 2.83 (PCA40), 2.26 (FA40), 1.58 (apoptosis) and 2.49 pp (allograft).

**Evidence bullets.**
- [H-S2-06 qualifies] With C retuned per training set, lift is positive with one supported training patient for all four representations (median 4.45-8.79 pp), and the primary paired gain k = 2 → 6 is small next to its uncertainty. Median gain [95% CI], patients with gain > 0, mean gain [95% CI]:

  | representation | median gain [95% CI] | patients > 0 | mean gain [95% CI] |
  |---|---|---|---|
  | PCA40 | 1.67 [-0.44, 2.89] | 5/7 | 1.16 [-0.15, 2.43] |
  | FA40 | 0.30 [-0.53, 4.18] | 4/7 | 1.75 [-0.48, 4.42] |
  | apoptosis | 1.14 [-0.53, 2.38] | 5/7 | 2.49 [-0.01, 6.42] |
  | allograft | 2.34 [0.14, 2.73] | 6/7 | 1.27 [-0.15, 2.46] |

  Only allograft's median-gain CI excludes 0. The curves are not monotone: PCA40 peaks at 9.30 at k = 4 and ends at 8.71; FA40 is 6.66 at k = 4 and 6.20 at k = 5.
- [H-S2-06 qualifies] Per-patient gains k = 2 → 6 range from −2.80 (P02, FA40: 17.85 → 15.05) to +13.32 pp (P03, apoptosis: 28.54 → 41.86). The largest gain in each representation is in a highest-prevalence patient (PCA40 P04 +2.94, FA40 P04 +8.30, apoptosis P03 +13.32, allograft P02 +3.09); P02 loses lift on both IFN-alpha representations (−1.81, −2.80). P01 and P06 stay within −0.53 to +1.14 pp. At k = 6, per-patient lift ranges from 0.06 (P06, PCA40) to 41.86 pp (P03, apoptosis).
- [H-S2-06 qualifies] Power law (primary validation, tuned median): reportable for 1 of 4 representations. Apoptosis passes (fit k ≤ 4 predicts 9.51 at k = 5 vs observed 10.31 [1.96, 35.80]; a = 9.80 pp, b = 4.45 pp, c = 1.68); its asymptote lies below the observed k = 5 and 6 medians, so it describes a plateau. PCA40, FA40 and allograft fail with c at its bound (allograft also b at bound and no rise). With CIs this wide the check is weak.
- [H-S2-08 supports] At the fixed atlas C = 1e-4, IFN-alpha FA40 probes are all-zero in every unit at k = 1-4 and in 90% and 71% of units at k = 5 and 6; median lift 0.00 at every k. PCA40 at fixed C is all-zero in 100% / 95% / 91% / 74% / 52% / 29% of units at k = 1..6, and its fixed-C median rises 0.00 → 11.08 only as the entry threshold falls with n. With C tuned, FA40 has no all-zero unit and PCA40 is all-zero in 5% of k = 2 units only.

**Other results.**
- Tuned vs fixed at k = 6: tuned median below fixed-atlas for PCA40 (8.71 vs 11.08), apoptosis (10.26 vs 11.52) and allograft (10.12 vs 12.56), above it for FA40 (7.03 vs 0.00); the atlas C is not nested.
- Paired gain over k = 1 (reference only; protocol switch): median 1.67 (PCA40), 1.62 (FA40), 3.87 (apoptosis), 0.72 pp (allograft) at k = 6.
- Tuning: all 56 k = 1 tuning runs used the stratified 5-fold cell-CV fallback (label strata); none of the 952 runs at k ≥ 2 did. At k = 1 the tuned arm picks L2 in 42/42 FA40, 36/42 PCA40, 36/42 apoptosis and 30/42 allograft units. Median tuned C is 0.0208 (L1) and 0.0032 (L2) at k = 1 against 0.0010 and 0.0001 at every k = 2-6. Tuned L2 C sits at the grid lower bound (1e-4) in 75% (FA40), 63% (PCA40), 60% (apoptosis) and 17% (allograft) of tuning subsets; tuned L1 C in 2% of PCA40 subsets and none of the others.
- Patient-inclusion effect, tuned arm (descriptive; median over held-out patients, matched k; subsets with j also differ in their other members): including P02 +5.54 allograft, +1.08 PCA40, +0.81 FA40, +0.25 apoptosis; including P09 −2.70 apoptosis, −2.48 allograft, −0.38 PCA40, +0.03 FA40.

## Claim gate
Supports (descriptive): on a frozen transductive Stage 1 basis, with C and penalty retuned per training set and every training subset enumerated, held-out label transfer for the four D15 representations is positive with one supported training patient; the median paired gain from two to six training patients is 0.30-2.34 pp, with 95% CIs that include 0 for three of four representations, non-monotone curves and gains concentrated in the highest-prevalence patients. The fixed-C IFN-alpha flat lines are all-zero L1 collapse, not absence of signal.
Does not support:
- that sequencing more patients would raise AUPRC (the basis saw every patient at every k; nothing is extrapolated to new patients); how a basis refit per training subset would change the curve is unknown in either direction (W6.6);
- gains over k = 1 as a patient-count effect (tuning protocol switch);
- the apoptosis power-law fit as a scaling law (one of four passes a weak check; it describes a plateau);
- patient count separated from malignant-cell count (adding a patient adds its malignant cells; W6.3);
- inductive deployment, causal biology or recurrence risk.

## Next steps
- W6.4: grant note from these numbers; the honest grant sentence is drafted in section 07 of the 2026-10-08 questions report.
- W6.6: D15b on an inductive Stage 1 basis refit per training subset, same subsets, compared in one table and figure with this curve (depends: W2.1).
- W6.5: Yu-Hsiang's patients as external points on this curve (blocked: W8.1).
- W6.2 and W6.3: D15 v2 cell-depth curve with C retuned per subsample; 2D design (k × malignant-cell cap).
