---
id: 2026-10-10_d26-leverage-allograft-apoptosis-done
date: 2026-10-10
type: analysis
title: D26 done - leaving P02+P03+P04 out of Stage 1 moves the allograft-rejection and apoptosis axes beyond composition-matched controls, mostly through P03, but no more than leaving out the contrast patients does (H-S1-05 not supported on the three panels tested, PCA/FA)
asked_by: user
asks: [A-20261009-02, A-20261009-13, A-20261009-15, A-20261008-02]
implemented_by: claude-code
questions: [Q1, Q2]
workstream: W2
analyses: [D26]
hypotheses: [H-S1-05]
verdicts: [H-S1-05=not_supported]
alts: []
closes: [W2.4]
advances: []
claim_level: descriptive
compute: screen-cpu  # the user's screen W2_d26 (12 single-thread workers); the merge, checks and this entry ran in the agent shell
c_policy: not_applicable  # Stage 1 alignment only; no Stage 2 probe is refit or evaluated held-out
training_n_changed: false
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the D26 runner and the rest of the 2026-10-09 overlay round are uncommitted (manifest git_dirty true)
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: [2026-10-09_d26-leverage-allograft-apoptosis]
---

## Question
> "What's a leverage test? So it sounds like in stage 1 DR, wrt P02, P03 and P04, doesn't affect too much, but their malignant prevalence do dominate in stage 2 classification. Why would it be open until allograft and apoptosis are tested? How about other stage 0 panels? Test what? Should we test them too?" (A-20261009-02)

> "I see the final conclusion is not very conclusive on this. So definitely try to fix the plots and my feedback above, and I like the next step, let's start them." (A-20261009-15)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02)

Sources: research/asks/2026-10-09_report-reading-notes.md (items 2 and 15, verbatim) and research/asks/2026-10-08_team-report.md (condensed wording). The metrics question (A-20261009-13) is answered in report v2 (shaping section). This entry completes the W2.4 run that 2026-10-09_d26-leverage-allograft-apoptosis built and handed to the user. The user started screen W2_d26 on 2026-10-10 with the wrapper as built, so D26 keeps all 13 patients (comparable with D17); the zero-malignant exclusion (A-20261009-14) governs new runs.

## Why it matters
H-S1-05 says the highest-prevalence patients P02/P03/P04 shape the Stage 1 loadings, not only the Stage 2 labels and selection. On IFN-alpha the D17 leave-out test found no leverage beyond malignant composition. Allograft rejection and apoptosis are where DR gains most over direct genes, and where that gain sits in these patients, so they were the panels on which the hypothesis mattered most. The outcome sets the H-S1-05 verdict for PCA/FA and tells W2.3 which imbalance re-weighting should target.

## Estimand / null
Unchanged from 2026-10-09_d26-leverage-allograft-apoptosis (the run entry), restated briefly:
- Unit: one Stage 1 refit × one reference axis; panels hallmark allograft rejection and hallmark apoptosis; PCA and FA at K 10, 20, 40, seed 42; leave-out sets P02, P03, P04, P02+P03+P04 and the contrasts P01, P09, P06, P01+P09; per set 3 size-matched and 20 composition-matched random drops (D17 seeds); normal-only refits; a transductive refit as the sanity floor.
- Axes: top_gap (primary; the largest all-cell standardized malignant-normal gap factor), top_heldout (the largest median univariate held-out lift factor) and probe (the atlas-selected Stage 2 configuration refit on all cells, a multi-factor direction).
- Metric: axis loss = 1 - many-to-one R2 of the reference axis on all K refit factors over the retained cells; ratio = axis loss / mean axis loss of the 20 composition-matched drops; one-sided rank p among those 20, floor 1/21 (descriptive, uncorrected).
- Decision rule, fixed before the full run (run entry, Estimand): (1) leverage beyond composition = P02+P03+P04 above all 20 composition-matched controls in at least 4 of 6 method × K cells; (2) support for H-S1-05 also needs the P02+P03+P04 ratio to exceed every contrast set's (P01, P06, P09, P01+P09) in at least 4 of 6 cells; (3) a verdict change needs the top_gap and probe axes to agree; (4) absolute size is reported regardless (matched |r| >= 0.98 = "beyond controls but small").
- c_policy not_applicable: nothing in Stage 2 is refit or scored held-out.

## Implementation
- Run: the user's screen session, from the repo root: `screen -S W2_d26`, then `bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.sh` (wrapper defaults `--ks 10,20,40 --resume --n-jobs 12`, single-thread BLAS); finished 2026-10-10 01:59 UTC (manifest run_history).
- Code: scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py and .sh (unchanged since the run entry).
- Verification in the agent shell (read-only on EXP): the checklist of the run entry and campaigns.csv row leverage_allograft_apoptosis_v1 (refit check, control counts, no merge_failed.json, findings present); `tests/test_d26_leverage_panels.py` with the lab env's pytest (no skips left); the decision summary recomputed from d26_leverage_contrast.csv in the report module figs_02_shape.py (asserted equal).
- Report: report v2's shaping section reads these tables through figs_02_shape.py (new figure fig02d_leverage_panels, the three panels side by side; IFN-alpha rows from D17).

## Artifacts
- analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/: d26_reference_axes.csv, d26_factor_signal.csv, d26_alignment.csv, d26_refit_check.csv, d26_leverage_contrast.csv, d26_decision_summary.csv, d26_cross_panel_top_gap.csv, leverage_allograft_apoptosis_v1_findings.md, leverage_allograft_apoptosis_v1_manifest.json, leverage_allograft_apoptosis_v1_artifact_ledger.csv
- analysis/figures/stage2_diagnostic/leverage_allograft_apoptosis_v1/: D26A_leverage_top_gap_axis, D26B_leverage_probe_axis, D26C_ratio_heatmap_top_gap_axis, D26D_ratio_heatmap_secondary_axes (.png and .pdf)
- analysis/cache/stage2_diagnostic/leverage_allograft_apoptosis_v1/: bases/ (2,364 refits), align_units/, d26_refs.json, stored_discovery_coefficients_extract.csv

## Results
- Completion: 2,364 bases, 7,092 alignment rows, 324 leverage rows (288 leave-out rows, each with 20 composition-matched and 3 size-matched controls; 36 normal-only rows); refit sanity 36/36 rows pass (minimum many-to-one R2 1, minimum matched |r| 1); no merge_failed.json; wall 482.2 s (refs 6.3 s, bases 468.8 s, report 6.7 s) at 12 workers; tests/test_d26_leverage_panels.py 16 passed, 0 skipped.
- Decision rule (d26_decision_summary.csv; cells = 6 method × K per panel and axis):

  | panel | axis | P02+P03+P04 above all 20 controls | ratio above every contrast | ratio range |
  |---|---|---|---|---|
  | allograft rejection | top_gap | 5/6 | 0/6 | 1.31-4.15 |
  | allograft rejection | top_heldout | 5/6 | 0/6 | 0.82-3.99 |
  | allograft rejection | probe | 5/6 | 0/6 | 1.22-3.24 |
  | apoptosis | top_gap | 6/6 | 0/6 | 2.01-3.98 |
  | apoptosis | top_heldout | 6/6 | 0/6 | 1.76-3.98 |
  | apoptosis | probe | 6/6 | 1/6 | 1.54-4.37 |

  Condition (1), leverage beyond composition, holds on both panels and all three axes; condition (2), more than the contrast patients, fails on all six; top_gap and probe agree on both conditions.
- [H-S1-05 qualifies] Unlike IFN-alpha (3/6 cells, D17), leaving out P02+P03+P04 moves the allograft-rejection and apoptosis primary axes more than all 20 composition-matched drops in 5/6 and 6/6 cells. Absolute size, primary axis: allograft axis loss 0.0038-0.0276 (0.38-2.8% of the axis's variance), matched |r| 0.945-0.970 for PCA (PC8) and 0.988-0.991 for FA (FA8), so the PCA movement is not "small" by rule (4); apoptosis 0.00067-0.0018 (0.07-0.18%), matched |r| 0.994-0.995 for PC6 and FA6, "beyond controls but small".
- [H-S1-05 qualifies] Most of it is P03: P03 alone loses 0.74-0.93 (allograft) and 0.72-0.88 (apoptosis) of the group's primary-axis loss (IFN-alpha 0.45-0.69), and is above all 20 of its own controls in 5/6 and 6/6 cells. P02 alone: 4/6 and 6/6; P04 alone: 3/6 and 1/6.
- [H-S1-05 contradicts] Clearing composition-matched controls is not specific to these patients: on the primary axis P01, P01+P09 are above all 20 in 6/6 cells on both panels, P09 and P06 in 4/6 (allograft) and 6/6 (apoptosis). P01 moves the axes far more than any other set: ratio to its controls 21.2-140.6 (allograft) and 48.1-127.7 (apoptosis), axis loss 0.011-0.090 and 0.011-0.037. On the three panels tested (IFN-alpha from D17), the P02+P03+P04 ratio exceeds every contrast set's in 0 of 6 cells on each primary axis.
- Rotation without loss: leaving P01 out of the apoptosis PCA fit keeps the axis in the refit span (axis loss 0.016-0.026) but no single refit factor matches PC6 (matched |r| 0.010-0.043 at K 10/20/40); for FA6 the match is 0.735-0.756; for allograft PC8 0.667-0.874.
- Probe axis (multi-factor, 8-40 nonzero factors): the axis losses are larger for every set and for the controls alike at K 40 (P02+P03+P04 up to 0.117 for allograft FA40, whose 20 controls average 0.096; up to 0.184 for apoptosis FA40, controls average 0.042, maximum 0.168), so dense K 40 probe directions move under any drop.
- Verdict: by the decision rule fixed before the run, H-S1-05 is not supported for PCA/FA on IFN-alpha, allograft rejection and apoptosis: the highest-prevalence patients move these axes beyond their malignant composition on two of the three panels, but no more than the contrast patients do; P01 has by far the most leverage on all three.

## Claim gate
Supports (descriptive): on transductive PCA/FA bases of IFN-alpha, allograft rejection and apoptosis, removing P02+P03+P04 from the Stage 1 fit moves the malignant-normal axis beyond composition-matched random drops on allograft rejection and apoptosis (mostly through P03), by less than removing P01 does; the H-S1-05 gate ("more than dropping other patients does") is not met on any of the three panels, for both the primary and the probe axis.
Does not support: any statement about FactoSig or FactoSig-promax bases (the atlas winners on both panels; not refit here); the other 38 biological panels; a causal account of why a patient's cells move an axis (in-silico perturbation, uncorrected descriptive rank p over 20 controls, 7 supported patients); reading per-patient axis movement as an effect on held-out lift (no Stage 2 refit); inductive deployment, causal biology or recurrence risk.
Standard closing line (THEME §6): descriptive results over seven supported patients; they do not establish inductive deployment, causal biology or recurrence risk.

## Next steps
- W2.8: its condition (leverage beyond composition on the primary or probe axis) is met, so the proposal is the leverage test on all biological panels (`--panels`, about 3 h in screen); because clearing composition-matched controls turned out to be common to every real patient set, rule (2) against the contrast sets is the informative read-out. Needs a campaigns.csv row and the user's go-ahead (R1, R4).
- W2.9: the cell-type-matched leave-out control for P01, now the main open leverage question (P01 leads on all three panels).
- W2.2: FactoSig/FactoSig-promax refits, the only remaining route to a different H-S1-05 reading on these panels.
- W2.3: the re-weighting target. Stage 1 leverage specific to P02-P04 is not there; their dominance is in the Stage 2 labels (report v2, shaping section, options list).
- W0.13: add this entry to research/paper/OUTLINE.md R4; THEME §6 review of the H-S1-05 verdict at the next milestone (STATUS §9 queue).
