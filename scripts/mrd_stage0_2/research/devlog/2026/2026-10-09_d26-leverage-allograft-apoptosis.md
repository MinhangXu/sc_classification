---
id: 2026-10-09_d26-leverage-allograft-apoptosis
date: 2026-10-09
type: run
title: D26 leave-high-prevalence-out leverage test on allograft rejection and apoptosis (D17 design) - runner, smoke test and reference axes; full run handed to the user's screen W2_d26 (W2.4)
asked_by: user
asks: [A-20261009-02, A-20261009-13, A-20261009-14, A-20261009-15, A-20261008-02]
implemented_by: claude-code
questions: [Q1, Q2]
workstream: W2
analyses: [D26]
hypotheses: [H-S1-05]
verdicts: []
alts: []
closes: []
advances: [W2.4]
claim_level: descriptive
compute: screen-cpu  # full run (2,364 Stage 1 refits) handed to the user; smoke and the 9-s refs part ran in the agent shell
c_policy: not_applicable  # Stage 1 alignment only; no Stage 2 probe is evaluated held-out (the probe axis reuses the stored atlas configuration as a direction)
training_n_changed: false
status: partial
commit: pending
run_head: f5a8f4d
dirty_run: true  # uncommitted D15b-full, report v1 and this session's runner on top of f5a8f4d
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "What's a leverage test? So it sounds like in stage 1 DR, wrt P02, P03 and P04, doesn't affect too much, but their malignant prevalence do dominate in stage 2 classification. Why would it be open until allograft and apoptosis are tested? How about other stage 0 panels? Test what? Should we test them too?" (A-20261009-02)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02)

Sources: research/asks/2026-10-09_report-reading-notes.md (the user's verbatim reading notes on report v1, item 2) and research/asks/2026-10-08_team-report.md (condensed wording; capture: paraphrase). P02, P03 and P04 are the highest-prevalence patients, not the patients with the most malignant cells (AGENTS R6). This entry builds the W2.4 test that the report and the D17 entries left open, and hands its full run to the user.

The same notes ask what the leverage test's metrics mean and why Figure 4's axis-loss x-axis needs explaining (A-20261009-13, A-20261009-14), and ask for the section's next steps to start (A-20261009-15):

> "I see the final conclusion is not very conclusive on this. So definitely try to fix the plots and my feedback above, and I like the next step, let's start them." (A-20261009-15)

The D26 figures carry the axis-loss note (Next steps). A-20261009-14's run decision (new runs exclude patients without malignant cells) is recorded in devlog 2026-10-09_report-reading-notes. As built, D26 keeps all 13 patients: it refits leave-out variants of the canonical transductive bases, which keeps it comparable with D17 and the stored bases. D26 does refit Stage 1, though, so whether the decision covers it is the user's choice before launching W2_d26 (report v2, shaping section, Next step). Applying it would need a runner change (reference bases refit without the zero-malignant patients) and would give up the direct comparison with D17. No recommendation is made.

## Why it matters
H-S1-05 (the highest-prevalence patients shape the Stage 1 loadings, not only the Stage 2 labels and selection) stays open with W2.4 blocking. On IFN-alpha, most of the apparent P02+P03+P04 leverage was malignant composition (2026-10-08_d17-composition-matched-controls). Allograft rejection and apoptosis are the next focus panels, and their atlas-selected probes are multi-factor (8 to 40 nonzero factors). So the test also asks whether a multi-factor discriminative direction moves more than a single factor. The outcome decides the H-S1-05 verdict and which imbalance W2.3 reweighting should target.

## Estimand / null
- Unit: Stage 1 basis variant × reference axis, for panels hallmark allograft rejection (147 covered genes) and hallmark apoptosis (138), PCA and FA at K 10, 20 and 40, seed 42, on the covered-gene z-scored features. FactoSig and FactoSig-promax (the atlas winner on both panels) are out of scope: their refit needs the factosig package's GPU path, which this CPU runner does not use (W2.2 adds FactoSig psi/mu persistence first).
- Arms (all alignment only; this follows the D17 design and uses D17's seeds, so the dropped cells match D17 and both panels):
  - transductive_refit (the sanity floor);
  - leave_out: P02 / P03 / P04 / P02+P03+P04, plus the contrasts P01 / P09 / P06 / P01+P09;
  - 3 size-matched random drops per set (seeds 1-3);
  - 20 composition-matched random drops per set (same number of dropped cells AND dropped malignant cells; seeds 101-120);
  - normal_only_all, with 3 random drops of 1,569 cells.
  - Total: 197 bases per panel × method × K, 2,364 bases.
- Reference axes, all taken from the stored transductive basis:
  - top_gap (PRIMARY): the factor with the largest |all-cell standardized malignant-minus-normal mean gap|. This is the D17 rule. Under balanced L1 it is the factor with the largest entry margin, i.e. the first factor any sparse probe selects. Because the gap is pooled over cells, it is the axis most exposed to the highest-prevalence patients' malignant cells.
  - top_heldout: the factor with the largest median univariate held-out lift over the seven supported folds, with the sign fixed on the training folds (the D18 card statistic).
  - probe: the score of the atlas-selected Stage 2 configuration (penalty, C and l1_ratio from D19 d19_atlas_selected_regularization.csv), refit on all cells of the stored scores. It is a linear combination of all K factors.
- Metrics:
  - axis loss = 1 − many-to-one R² of the reference axis regressed on all K refit factors, over the retained cells (the d17 axis_loss_fit);
  - Hungarian matched |r| (single-factor axes only);
  - top-8 and all-K principal angles;
  - one-sided empirical rank p of the observed axis loss among the 20 composition-matched controls, (1 + #controls ≥ observed) / 21. It is descriptive and uncorrected.
- Decision rule (fixed before the full run, from smoke artifacts only):
  1. "Leverage beyond composition" for a panel × axis: the P02+P03+P04 axis loss lies above all 20 composition-matched controls (rank p = 1/21 ≈ 0.048) in at least 4 of the 6 method × K cells. This is the D17 rule.
  2. Support for H-S1-05 also requires the P02+P03+P04 ratio to its control mean to exceed that of every contrast set (P01, P06, P09, P01+P09) in at least 4 of 6 cells. This is the hypothesis gate: "more than dropping other patients does".
  3. A verdict change needs the top_gap and probe axes to agree; if they disagree, the result is reported as axis-dependent.
  4. Absolute size is reported regardless: an axis that clears the controls but keeps matched |r| ≥ 0.98 is "beyond controls but small".
- c_policy: not_applicable. No Stage 2 probe is refit or evaluated held-out. The probe axis uses the stored atlas configuration on all cells only to define a direction. It reproduces the stored discovery_full_cohort coefficients (Results).

## Implementation
- Code (repo-relative, new in this session):
  - scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py: parts refs, bases (one unit = fit + align of one basis), report. `--resume` skips units that have a saved basis and an alignment JSON with the current reference-axis signature. A saved basis with a missing or stale alignment is only re-aligned. Empty or corrupt files are redone. A report that finds missing units writes merge_failed.json and exits 3; recovery is `--resume` or `--merge-only`. Other flags: `--out-root` (required with `--smoke`), `--panels` (any Stage 0 panel id or hallmark suffix), `--plan-only`, `--bases-location`.
  - scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.sh: screen wrapper, session W2_d26. Defaults: `--ks 10,20,40 --resume --n-jobs 12`, with OMP/OPENBLAS/MKL threads set to 1.
  - tests/test_d26_leverage_panels.py: 16 tests, 11 synthetic and 5 regression against EXP. Among them: leverage_contrast equals d17.leverage_table on a single axis; NaN-safety on the probe axis; the decision rule; plan counts (197 per cell); resume states; refit lossless on toy data; the top-gap rule reproduces D17's PC8/FA7; ref_arrays equal D17's for IFN-alpha PCA10; univariate lifts reproduce the D18 cards for allograft FA40; plus D26 artifact checks that skip until the run has finished.
- Reuse by import, with no edits to shared code:
  - run_d17_basis_variant_pilot (fit_rows, drop sets and seeds, LABEL_MATCHED_METRICS, io helpers);
  - stage1_refit (fit_basis, Stage1Basis, align_scores, many_to_one_r2, score_correlation);
  - regularization_contract (standardize_train, fit_probe, supported_patients);
  - new code is leverage_contrast, which generalizes d17.leverage_table to several panels and axes and is NaN-safe.
- Inputs (EXP-relative):
  - stage1_direct_gene/old34_direct_gene_all_v1/panel_id={single_geneset__hallmark_allograft_rejection,single_geneset__hallmark_apoptosis}/transform=zscore/{features.npz,feature_names.json};
  - stage1_dr/panel_id=<panel>/method={pca,fa}/k={10,20,40}/seed=42/{scores,loadings}.npy;
  - stage2_supervised/multiobjective/discovery_full_cohort/{cell_metadata.csv.gz,coefficient_paths.csv};
  - analysis/scorecards/stage2_diagnostic/regularization_contract_v1/{d19_atlas_selected_regularization.csv,d19_entry_margin_audit_by_fold.csv};
  - analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_index.csv;
  - analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/d17_leverage_contrast.csv (IFN-alpha comparison rows, read-only).
- Smoke tests (agent shell, 12 single-thread workers, `--out-root` in the session scratchpad, nothing in EXP):
  - `--smoke --ks 10`: 44 units in 8 s.
  - `--smoke --ks 10,20,40`: 132 units, 36.7 s wall; refit sanity gate 36/36.
  - After the smoke drop sets became P02+P03+P04 and P09, `--smoke --ks 10,40`: 88 units, 34.4 s, gate 24/24.
  - The wrapper with `--smoke --ks 10 --panels apoptosis`: 22 units, 11.8 s.
  - Resume test: one deleted and one emptied alignment JSON gave merge_failed.json and exit 3; `--resume` then re-aligned the 2 units without refitting and cleared merge_failed.json.
  - Figures D26A-D26D were inspected and the layout fixed: one x-axis note, a 2-line colorbar label, and per-row ratio and rank p text.
- Cost estimate from the smoke unit timings (fit + align, single thread): mean seconds per unit were PCA 0.93 / 1.16 / 1.99 and FA 2.03 / 2.74 / 6.47 at K 10 / 20 / 40. For 2,364 units that is about 6,040 CPU-seconds, or about 8.4 min wall with 12 workers before load imbalance. That exceeds the agent limit (R1), so the full run goes to the user.
- Ran in the agent shell on the canonical slug: the refs part only (8.7 s; 12 reference cells): `python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py --parts refs --n-jobs 12`. The screen run recomputes it with identical, deterministic output.
- Full run for the user (expected about 8.5-10 min wall):
  - `screen -S W2_d26`
  - `bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.sh`
  - detach with Ctrl-a d; reattach with `screen -r W2_d26`. If interrupted, or if exit status 3 appears, rerun the same command (`--resume` is the wrapper default).
  - Verify:
    - the log ends with "[report] refit sanity: 36/36 rows pass", "[report] tables, 8 figure files, findings, manifest and ledger written" and "[done] ... (status 0)";
    - d26_refit_check.csv has pass=True in every row;
    - d26_leverage_contrast.csv has n_label_matched_controls = 20 and n_random_controls = 3 on all 288 leave_out rows;
    - merge_failed.json is absent;
    - `python -m pytest -q tests/test_d26_leverage_panels.py` runs 16 tests with none skipped.
- Tests: `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python -m pytest -q tests/test_d26_leverage_panels.py` gave 14 passed and 2 skipped (D26 run artifacts not yet present) in 2.4 s. The full suite gave 294 passed and 1 skipped, plus one failure: tests/test_research_index.py::test_real_repo_check fails on E14 for H-S2-09, a hypothesis added to hypotheses.md in parallel by the report-notes integration and not yet declared in a devlog entry. This entry does not touch it.
- Provenance: run_head f5a8f4d, dirty tree (uncommitted runner). Bases go to analysis/cache/stage2_diagnostic/leverage_allograft_apoptosis_v1/bases/, as for D17; `--bases-location stage1_dr_variants` writes the R3 location instead.

## Artifacts
- analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/d26_reference_axes.csv (36 rows: panel × method × K × axis; cross-checks against D19, D18 and the stored discovery coefficients)
- analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/d26_factor_signal.csv (every factor: all-cell gap, per-patient univariate held-out lift, probe coefficient)
- analysis/cache/stage2_diagnostic/leverage_allograft_apoptosis_v1/d26_refs.json (reference-axis indices and probe weights; the unit signature)
- analysis/cache/stage2_diagnostic/leverage_allograft_apoptosis_v1/stored_discovery_coefficients_extract.csv
- After the screen run (not yet present), in the scorecards slug: d26_alignment.csv, d26_refit_check.csv, d26_leverage_contrast.csv, d26_decision_summary.csv, d26_cross_panel_top_gap.csv, leverage_allograft_apoptosis_v1_findings.md, _manifest.json and _artifact_ledger.csv. In the figures slug: D26A_leverage_top_gap_axis, D26B_leverage_probe_axis, D26C_ratio_heatmap_top_gap_axis and D26D_ratio_heatmap_secondary_axes (.png and .pdf). In the cache slug: bases/ (2,364 bases) and align_units/.

## Results
Status partial: the leverage contrast has not run, so no H-S1-05 number yet. Reference axes (d26_reference_axes.csv, d26_factor_signal.csv):
- Allograft rejection, top_gap: PC8 at K 10 / 20 / 40 (all-cell standardized gap +0.940, +1.059, +0.950) and FA8 (−1.070, −1.114, −1.107). D19's per-fold largest-margin factor is the same factor in 7/7, 7/7 and 5/7 folds (PCA) and 6/7, 7/7 and 7/7 (FA).
- Allograft rejection, top_heldout:
  - PCA: PC8 at K10 and K20; PC13 at K40 (median 3.40 pp vs PC8 3.37 pp);
  - FA: FA10 at every K (median 5.82 / 6.40 / 7.37 pp vs FA8 2.09 / 2.11 / 2.21 pp).
- Apoptosis, top_gap: PC6 (gap −0.864, −0.858, −0.861) and FA6 (−0.925, −0.876, −0.857).
  - D19 agreement: 6/7 folds at every PCA K and 3/7 for FA20. D19 has no margin rows for FA10 and FA40, whose atlas probes are L2.
- Apoptosis, top_heldout: PC7 at every K (median 12.43 / 12.42 / 12.43 pp vs PC6 2.47 / 2.34 / 2.51 pp); FA6 at K10 and K40, FA13 at K20 (5.88 vs FA6 5.36 pp).
- The pooled-gap factors separate mainly in P03 and one other highest-prevalence patient. Univariate held-out lift at K40 per patient (P01, P02, P03, P04, P06, P07, P09):
  - allograft FA8: 0.37, 5.82, 39.04, 22.68, 0.79, 0.32, 2.21 pp;
  - apoptosis PC6: 0.17, 19.63, 29.27, 0.61, 0.26, 2.51, 5.46 pp;
  - for comparison, apoptosis PC7: 0.74, 26.20, 29.38, 23.11, 1.58, 12.43, 1.19 pp.
  - These numbers describe which patients a factor separates (labels, Stage 2). They say nothing about who shapes the loadings; that is what the screen run measures.
- Probe axis:
  - the atlas configurations have 8-40 nonzero factors: 9 / 15 / 31 (allograft PCA), 8 / 17 / 36 (allograft FA), 8 / 15 / 27 (apoptosis PCA), 10 / 17 / 40 (apoptosis FA);
  - the refit coefficients match the stored discovery_full_cohort coefficients with cosine ≥ 0.99999998 in 12/12 cells;
  - for PCA, the probe score on the alignment reference (projection onto the stored loadings) correlates ≥ 0.9978 with the score on the stored randomized-SVD scores.
- Cross-check against D18: the univariate held-out lift reproduces factor_cards_index.csv median_univariate_lift for all 40 factors at K40 on both panels and methods, within 4.8e-9 pp.

## Claim gate
Supports (descriptive): the three reference axes of the D26 leverage test are defined from stored artifacts and reproduce D17's rule, the D18 cards and the stored discovery probe. On allograft rejection and apoptosis, the factor with the largest pooled malignant-normal gap separates mainly the highest-prevalence patients' cells, and in 7 of 12 method × K cells the best held-out factor is a different one.
Does not support:
- any statement about whether P02/P03/P04 shape the allograft or apoptosis loadings (H-S1-05) before the W2_d26 run and its done entry;
- FactoSig or FactoSig-promax leverage (not refit here);
- reading per-patient univariate lifts as loading leverage or as identifiable single-patient effects;
- significance (descriptive rank p over 20 controls, uncorrected);
- inductive deployment, causal biology or recurrence risk.
Standard closing line (THEME §6): descriptive results over seven supported patients; they do not establish inductive deployment, causal biology or recurrence risk.

## Next steps
- W2.4 (user, screen-cpu, about 8.5-10 min): `screen -S W2_d26`, then `bash scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.sh`, then detach with Ctrl-a d. Verify as listed in Implementation. Then write a type: analysis, status: done entry that supersedes this one:
  - apply the decision rule from d26_decision_summary.csv;
  - quote the P02+P03+P04 and P01 rows of d26_leverage_contrast.csv for each axis;
  - set closes: [W2.4], and change the H-S1-05 verdict only if top_gap and probe agree.
  - For shorter sessions, the wrapper also takes one panel at a time (`--panels allograft_rejection`, then `--panels apoptosis`; about 4-5 min each). The report part then needs a final run of the plain command, which resumes everything.
- W2.4 report (v2 integrator): figures D26A and D26C use report fig02b's encoding (method color, patients on the y axis, both control clouds). d26_leverage_contrast.csv and d26_alignment.csv keep the D17 column names plus panel_id and ref_axis, so figs_02_shape can filter them directly. The axis-loss explanation the user asked for (A-20261009-14) is printed on D26A/D26B as a header note and an x-axis note.
- W2.8 (new, proposed backlog line): the leverage test on other Stage 0 panels with `--panels` (about 3,000 CPU-seconds per panel, about 4-5 min wall at 12 workers; all 41 Set A panels about 3 h in screen). Run it on all panels only if D26 finds leverage beyond composition on the primary or probe axis; otherwise run a stratified sample of panels. This answers "should we test them too?" (A-20261009-02).
- W2.2: FactoSig/FactoSig-promax refits (the atlas winners here) need psi/mu persistence and the GPU path before they can enter this test.
- W2.3: the reweighting target (Stage 1 geometry vs Stage 2 labels) waits for the D26 outcome.
