---
id: 2026-10-02_d14-patient-factor-overlap
date: 2026-10-02
type: analysis
title: D14 per-patient Stage 1 factor-score overlap test (aggregate vs separate subtypes)
asked_by: user
asks: [A-20261008-07, A-20261008-08, A-20261008-09, A-20261008-10]
implemented_by: unknown
questions: [Q1, Q2]
workstream: W1
analyses: [D14]
hypotheses: [H-S1-04, H-S1-05, H-S1-07]
verdicts: [H-S1-04=supported_descriptive]
alts: []
closes: []
advances: [W1.2]
claim_level: descriptive
compute: notebook-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: b60a6ee  # retro: first commit containing patient_factor_overlap.py and the notebook
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # CSVs 14:28-14:39 UTC, findings 14:39 UTC
supersedes: []
---

## Question
> "TOP PRIORITY: data-driven statistical test for whether to aggregate patients." Step 1: learn factors on all patients and project each patient's cells. Step 2: check per-patient factor distributions. Step 3: if distributions overlap significantly, patients share variation; if distinct, treat them as separate subtypes.

Source: the user's TODO list as recorded in research/asks/2026-10-08_team-report.md (A-20261008-07..10). The original prompt of about 2026-10-01 was not saved, so this text is reconstructed from the report. Stated motivation: P02/P03/P04 seemed to dominate. These are the highest-prevalence patients.

## Why it matters
Q1 needs a data-driven reason to pool patients on one Stage 1 basis before any claim about shared vs patient-specific structure (H-S1-04). The result decides whether the W2 bases (inductive, leave-out, patient-balanced) are a robustness check or a necessity.

## Estimand / null
- **Estimand**, per representation: the variance-weighted mean over factors of the median pairwise 32-bin histogram overlap of factor scores between patients.
  - Computed within CN.label.
  - Normal strata: patients with >=200 cells. Malignant strata: patients with >=20 cells. This is a stratum parameter, not the support rule.
  - Bins are shared over the 1st-99th percentile; tails go into the end bins.
- **Null:** patient labels permuted within label, keeping per-patient counts and fixed bin edges; n_perm = 25.
- **Decision rule** (overlap_decision_rule.json):
  - aggregate_shared when normal weighted overlap >=0.75, normal minimum patient-vs-rest >=0.60, malignant weighted >=0.45 and malignant gap <0.15;
  - separate_subtype_candidate when normal minimum <0.40, or malignant weighted <0.35 with gap >=0.15;
  - otherwise shared_with_location_shift.
- **Single-factor flag:** malignant eta2 >=0.75 AND variance share >=0.15.
- **C policy:** not applicable (no Stage 2 fit).

## Implementation
- Code: scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py (run_screen, focus_factor_table, low_overlap_factors, save_overlap_figures, write_findings).
- Driven from the D14 section of scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_patient_factor_overlap.ipynb.
- Inputs:
  - stored transductive stage1_dr/panel_id=*/method=*/k=*/seed=42/scores.npy, with no DR refit;
  - the D1 atlas selection of one representation per panel (41 biological + 6 controls = 47);
  - the cohort cache for CN.label, patient and predicted.annotation.
- Ran on CPU in a Jupyter kernel, taking minutes.
- Provenance: the code was untracked on 43f19be and first committed in b60a6ee.

## Artifacts
- analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/, containing:
  - patient_factor_overlap_calls.csv
  - patient_factor_overlap_by_stratum.csv
  - patient_factor_overlap_by_patient.csv
  - biological_overlap_summary.csv
  - focus_factor_overlap.csv
  - focus_celltype_overlap.csv
  - low_overlap_high_variance_factors.csv
  - overlap_decision_rule.json
  - patient_factor_overlap_findings.md
- analysis/figures/stage2_diagnostic/patient_factor_overlap_v1/ D14A-D14E (.png and .pdf)

## Results
- [H-S1-04 supports] Calls over 47 representations: 45 aggregate_shared, 2 shared_with_location_shift, 0 separate_subtype_candidate. Biological panels: 40/41 aggregate_shared.
- [H-S1-04 supports] Biological overlap:
  - normal weighted overlap 0.777-0.890 (median 0.843);
  - malignant 0.538-0.657 (median 0.610);
  - median malignant gap to the null 0.060.
- [H-S1-04 qualifies] The permutation p sits at its floor, 1/26 = 0.038, in all 47 normal strata. The calls therefore rest on uncalibrated magnitude gates.
- [H-S1-04 qualifies] Allograft promax K40 is a location shift: P02's normal patient-vs-rest overlap is 0.575, below the 0.60 gate. Promax factor 40 carries 0.71 of the malignant variance weight.
- [H-S1-07 qualifies] P03-private malignant axes appear only in broad bags:

  | Representation | P03-vs-rest overlap | Variance share | eta2 |
  |---|---|---|---|
  | HVG10k FactoSig40 f21 | 0.007 | 0.373 | 0.913 |
  | HVG3k FA40 f27 | 0.013 | 0.196 | 0.907 |
  | full_34 FactoSig40 f30 | 0.009 | 0.299 | 0.898 |

  No biological-program factor meets the flag.
- [H-S1-05 qualifies] P02, P03 and P04 score distributions overlap those of the other patients on 40/41 biological bases. Score overlap is not the same as influence on the loadings.
- IFN-alpha focus:
  - PC8 eta2 0.010 normal / 0.031 malignant;
  - FA7 eta2 0.015 / 0.025;
  - max malignant eta2 0.114 (PCA40) and 0.121 (FA40).
- P07, with 22 malignant cells, is the minimum malignant patient-vs-rest overlap in 38/41 biological panels.

## Claim gate
Supports (descriptive): on the transductive basis, supported patients' factor-score supports overlap for 40/41 biological programs. Pooling them on one Stage 1 basis is reasonable, and no biological program shows a patient-private (subtype) factor.
Does not support:
- that high-prevalence patients leave the loadings unshaped (no leave-out or reweighted refit was done);
- predictive sharedness (that is D3);
- significance (the permutation p is at its floor and the gates are uncalibrated);
- held-out-patient or inductive statements;
- the 13 promax calls (degenerate late factors drive the weights);
- P03's broad-bag axis as a subtype (genotype, label provenance, batch and composition are not excluded);
- small-patient malignant overlaps.

## Next steps
- W1.2: D20 = D14 v2, covering promax weighting, all PCA/FA representations, >=200 permutations, calibrated gates and support-matched subsampling.
- W1.1: D16 factor x patient allocation matrix (H-S1-05).
- W1.3: rerun on the W2 bases after W2.1.
- W1.4: D25 genotype/label-provenance audit of the P03 axis.
- W0.5: re-execute the notebook so its outputs are saved.
