---
id: 2026-10-08_d17-composition-matched-controls
date: 2026-10-08
type: analysis
title: D17 follow-up - composition-matched random-drop controls for the IFN-alpha leave-high-prevalence-out leverage test (W2.1)
asked_by: user
asks: [A-20261008-02]
implemented_by: claude-code
questions: [Q1, Q2]
workstream: W2
analyses: [D17]
hypotheses: [H-S1-05]
verdicts: []
alts: [H-S1-05:ALT-PREV=partial]
closes: []
advances: [W2.4]
claim_level: descriptive
compute: agent-cpu  # align + report parts only; 48 s wall with 12 single-thread workers
c_policy: not_applicable  # Stage 1 alignment only; no Stage 2 refit in this step
training_n_changed: false
status: done
commit: pending
run_head: d387eda
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02)

Source: research/asks/2026-10-08_team-report.md. This entry completes the step listed under Next steps in 2026-10-08_d17-basis-variant-pilot. It does not supersede that entry: its H-S1-06, H-S1-03 and H-S2-08 evidence stands, and its H-S1-05 caveat is answered here.

## Why it matters
The D17 pilot found that dropping P02+P03+P04 moves the IFN-alpha discriminative axis 3-5 times more than dropping the same number of random cells. Those controls did not remove the same number of malignant cells (1,071 versus about 128 expected), so the excess could come from malignant composition rather than from these patients. H-S1-05 (Q1/Q2) cannot be read until that is separated; the result also feeds the W2.3 decision on which imbalance a reweighted Stage 1 should target.

## Estimand / null
Estimand: axis loss = 1 − many-to-one R² of the reference axis (PC8 for PCA, FA7 for FA, matched at K 10/20/40) on the cells retained in the leave-out fit. Comparator: 20 random drops per leave-out set that match both the number of dropped cells and the number of dropped malignant cells (composition-matched controls). Decision rule fixed before looking: a leave-out set has leverage beyond composition only if its axis loss exceeds every composition-matched control (one-sided rank p = 1/21 ≈ 0.048, the floor with 20 controls) at most method × K cells. Rank p values are descriptive and not corrected for the 48 contrasts.

## Implementation
- Code: scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py (control bases were already fitted by the pilot run; this step aligned them and regenerated the report).
- Command (agent shell, 12 single-thread workers): `OMP_NUM_THREADS=1 python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py --ks 10,20,40 --parts align,report --resume --n-jobs 12`
- Output: "[align] 1302 bases aligned in 36.9 s"; "[report] tables, 6 figure files, findings, manifest and ledger written in 9.6 s"; wall 47.7 s.
- Check: n_label_matched_controls = 20 on all 48 leave_out rows of d17_leverage_contrast.csv; the findings gained the "Composition-matched controls" table.

## Artifacts
- analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/d17_leverage_contrast.csv (columns *_label_matched_*)
- analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/basis_variant_pilot_ifna_v1_findings.md (section "Composition-matched controls")
- analysis/figures/stage2_diagnostic/basis_variant_pilot_ifna_v1/ (D17A-D17C, regenerated)

## Results
- [H-S1-05 qualifies] Once the controls remove the same number of malignant cells, the P02+P03+P04 excess disappears for FA and shrinks for PCA. Axis loss divided by the composition-matched control mean (rank p in parentheses):
  - FA: K10 1.00 (0.48), K20 0.88 (0.86), K40 0.36 (0.19).
  - PCA: K10 1.94 (0.048), K20 1.42 (0.048), K40 1.93 (0.048). PCA stays above all 20 controls, but by less than 2 times.
  - Against size-matched controls the same drops were 3.1-5.2 times (2026-10-08_d17-basis-variant-pilot), so most of that excess is malignant composition.
- [H-S1-05 qualifies] In absolute terms the axis barely moves without P02+P03+P04: matched |r| 0.9885 (PCA10), 0.9942 (PCA20), 0.9953 (PCA40), 0.9969 (FA10), 0.9943 (FA20), 0.9958 (FA40).
- [H-S1-05 qualifies] Single patients, ratio to the composition-matched mean: PCA40 P02 3.83, P03 1.98, P04 1.83 (each p 0.048); FA40 P02 0.17, P03 0.15, P04 0.03 (p 0.14-0.33). FA40 controls are noisy (control axis loss up to about 0.013 against a mean near 0.002), so FA40 ratios below 1 mean "within the control range", not "less leverage".
- [H-S1-05 qualifies] The low-prevalence, high-cell-count contrast P01 has by far the most leverage beyond composition: ratio 23.8 / 15.0 / 15.0 (PCA K10/20/40) and 21.0 / 10.7 / 3.74 (FA), all p 0.048; PCA40 matched |r| 0.58. P09: 2.28-3.26 (PCA), 2.90-4.41 (FA). P06: 1.85-2.56 (PCA), 1.89-4.82 (FA).

## Claim gate
Supports (descriptive): on the IFN-alpha panel (PCA and FA, K 10/20/40), the extra axis movement from leaving out P02+P03+P04 is largely what removing that many malignant cells does; a residual of less than 2 times the composition-matched controls remains for PCA, with the axis essentially unchanged (matched |r| ≥ 0.988). P01, not a high-prevalence patient, has the largest leverage on this axis.
Does not support:
- that P02/P03/P04 disproportionately shape the Stage 1 loadings in general (one panel; W2.4 tests allograft and apoptosis);
- that they do not matter for Stage 2 (they still dominate malignant labels, cell-weighted selection and lift summaries);
- significance (descriptive rank p over 20 controls, 48 uncorrected contrasts, seven patients);
- a biological reading of P01's leverage (P01 is 19.7% of the cells; cell-type composition and batch are not separated).
Standard closing line (THEME §6): descriptive results over seven supported patients; they do not establish inductive deployment, causal biology or recurrence risk.

## Next steps
- W2.4: repeat the leave-out design with size- and composition-matched controls on allograft and apoptosis before any H-S1-05 verdict.
- W2.3: the imbalance to target is in Stage 2 (labels, selection), not Stage 1 loadings, at least on IFN-alpha; decide the reweighting target in a decision entry.
- W1: ask why P01 has the most leverage (cell-type composition of its 11,860 cells versus the cohort; D16 already shows P01 is the most over-represented patient in score energy).
