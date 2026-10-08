---
id: 2026-10-08_review-fixes
date: 2026-10-08
type: fix
title: Same-day review fixes for D15b-D19, the team update and the research spine (runner guards, report wording, D18 degenerate flag, D17 composition-matched controls, research_index rules)
asked_by: agent
asks: []
implemented_by: claude-code
questions: [meta, Q1, Q2, Q3]
workstream: W0
analyses: [D15b, D16, D17, D18, D19]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: [W0.1, W2.2]
claim_level: none
status: partial  # the D17 slug tables do not yet include the composition-matched controls (see Next steps)
commit: pending
run_head: d387eda  # HEAD is the d387eda wip checkpoint, one commit after b60a6ee; the tree is dirty
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
(reconstructed from the 2026-10-08 verification pass over the D15b-D19 entries, the team update and the research spine; no user question)

## Why it matters
The same-day entries for D15b-D19 and the team update are the record the team reads first. The review found overclaims (H-S1-05 leverage, PC8 "same axis" as FA7, a "lower bound" on the D15b curve), confounded comparisons (the D15b paired gain over one patient, the D17 random-drop controls), and tooling gaps that could silently overwrite canonical bundles or hide stale THEME text. The original entries were corrected in place (same day, uncommitted); this entry records what changed and what is still pending.

## Estimand / null
n/a (corrections and tooling). The D17 composition-matched control is a new comparator: random Stage 1 cell drops that match both the number of dropped cells and the number of dropped malignant cells of each leave-out set (20 seeds), with a one-sided empirical rank of the observed axis movement among them.

## Implementation
- Runner guards (canonical bundles can no longer be overwritten by partial runs):
  - scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py: new `--out-root`; `--shortlist smoke|ifna` refuses to run without it (the merge rewrites the bundle for the requested shortlist only).
  - scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py: `--smoke` and `--max-cells` need `--out-root`; a resumed tuned unit whose stored c_grid_n differs from the current C grid is rerun.
  - scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py: `--smoke` needs `--out-root`.
- Report fixes:
  - D19: refit labelled newton_cd (sklearn objective); refit-check tolerance printed from the data; anchor lines print "all-zero" instead of "nan"; the stored, refit, 1.05 x and fixed-L2 arms are labelled "C not nested" (C selected with held-out outcomes), with a c_nested column in d19_ifna_arm_summary.csv.
  - D15b: the "lower bound" sentence is replaced by "unknown in either direction"; a k = 2 paired gain (d15b_reduced_gain_vs_k2.csv) is the primary comparison; the k = 1 tuning-protocol switch and the reduced design's imbalance at k = 2 and 5 are stated; no power law is reportable from the reduced design; fixed_atlas is labelled not nested.
  - D16: the spread/location sentence no longer writes medians as an equation; it adds the median spread share.
  - D17: the "Transductive refit floor" section is renamed and reports stored-score vs reference r (PCA stored scores are U·S; the refit projects); new arm leave_out_label_random (stage1_refit.label_matched_drop_mask) and composition-matched columns in the leverage table and findings.
  - D18 (factor_cards.py): degenerate_flag now means zero score variance or zero loadings only; scale_outlier stays a caveat (promax column scale is arbitrary and Stage 2 standardizes scores). The factor explorer chip and the W5 README schema text follow.
- stage1_refit.py docstrings: the unweighted refit reproduces the stored FA bases and PCA loadings, not the stored PCA scores; the pseudo-data FA fit matches up to the sign of each component.
- tools/research_index.py: `--strict` no longer promotes W11 (history); `--strict-artifacts` fails with E21 when the experiment dir does not resolve; lint E07 covers the new tests and the factor explorer app; the THEME review queue includes same-day entries not cited in THEME, or uses the new THEME key reviewed_through; an optional `seq:` front-matter key breaks same-date verdict ties; `reserve-id` takes `--asks`, `--code` and `--findings-md`; CLAIMS Part C cites the gate conditions now written in research/devlog/README.md.
- Tests added: tests/test_research_index.py (strict exemption, E21, review queue, seq, reserve-id columns, E07 on new tests and apps), tests/test_factor_cards.py (scale outlier is not degenerate; D18 partial-shortlist guard), tests/test_stage1_refit.py (label-matched drop; D17 smoke guard), tests/test_regularization_contract.py (D19 smoke guard), tests/test_patient_count_curve.py (reduced-design balance by size).
- Regenerated in the agent shell (12 threads or fewer):
  - D18: `python scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py --shortlist full --n-jobs 6` with OMP_NUM_THREADS=2 (102 s; every part recomputed).
  - D19: `python scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py --parts report --n-jobs 12` (3 s).
  - D15b: `python scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py --design reduced --parts report --resume --n-jobs 12` (31 s).
  - D16: `python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py --merge-only` (25 s).
  - D17: `python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py --ks 40 --parts bases --resume --n-jobs 12` (66 s), then the same with `--ks 10,20` (50 s): 960 composition-matched control bases fitted into the cache. The align and report parts were not run in this session, so the D17 slug tables and findings predate the controls.
- Smoke runs went to the session scratchpad with `--out-root` (D18 smoke 14 s; D17 smoke 45 s).

## Artifacts
- analysis/scorecards/stage2_diagnostic/factor_cards_v1/ (bundle regenerated; only the degenerate_flag column of factor_cards_index.csv changed; findings, manifest and ledger rewritten)
- analysis/scorecards/stage2_diagnostic/regularization_contract_v1/regularization_contract_v1_findings.md, d19_ifna_arm_summary.csv, d19_ifna_pca40_vs_fa40_by_patient.csv, manifest and ledger
- analysis/figures/stage2_diagnostic/regularization_contract_v1/ (figure titles carry the new arm labels)
- analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/d15b_reduced_gain_vs_k2.csv (new), patient_count_learning_curve_v1_findings.md and the other reduced-design tables (rewritten, unchanged values)
- analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1/factor_patient_matrix_findings.md (rewritten)
- analysis/cache/stage2_diagnostic/basis_variant_pilot_ifna_v1/bases/arm=leave_out_label_random/ (960 bases, not yet in the slug tables)
- analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/basis_variant_pilot_ifna_v1_findings.md (hand-edited: refit section renamed with the stored-score vs reference r from d17_reference_axes.csv; pending-controls note; the ledger hash for it is stale until the report reruns)

## Results
- D18 after the flag change: 1810 cards, 1801 non-degenerate (9 zero-variance FA cards), 117 scale-outlier promax cards; 101 of 248 never-selected non-degenerate cards reach at least 5 pp univariate held-out lift in a supported patient (12 in at least three). All 27 regression checks pass.
- D19 refit check: the newton_cd refit reproduces the stored cuML lifts within 0.00014 pp (1.4e-6 AP). Anchor 14/14 unchanged.
- D15b primary paired gain k = 2 → 6 (tuned, median [95% CI]): PCA40 2.16 [0.20, 2.48], FA40 0.75 [−0.64, 4.67], apoptosis 1.49 [−0.65, 4.57], allograft 3.06 [0.02, 4.43] pp. All 56 k = 1 tuning runs used the cell-CV fallback; median tuned L1 C is 0.0208 at k = 1 and 0.0010-0.0021 at k = 2-6. Reference checks 84/84.
- tools: `research_index.py check --strict` now passes on the repo (W11 exempt); 5 new research_index tests plus an extended reserve-id test.

## Claim gate
Supports (descriptive): n/a; this entry changes wording, labels and tooling, not evidence. The corrected claims live in the D15b, D17, D18 and D19 entries, the team update and THEME.
Does not support: any new claim about H-S1-05 from the composition-matched controls until the D17 slug tables are regenerated and requoted in a new entry.

## Next steps
- W2.2 (D17 follow-up, agent CPU, about 1 min): `python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py --ks 10,20,40 --parts align,report --resume --n-jobs 12`. Expected: "[align] 1302 bases aligned" and the report line. Verify n_label_matched_controls = 20 on every leave_out row of d17_leverage_contrast.csv and a "Composition-matched controls" table in the findings, then requote H-S1-05 in a new entry that supersedes 2026-10-08_d17-basis-variant-pilot.
- W0.1: review the overlay and commit it in groups; W0.4: install pytest and dash in the lab env so the suite runs there.
- W6.1: the full D15b design in screen W6_d15b also removes the k = 2 and k = 5 imbalance.
