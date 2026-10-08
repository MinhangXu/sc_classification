---
id: 2026-10-08_w7-atlas-strip-factor-explorer
date: 2026-10-08
type: tool
title: W7.1 atlas per-fold active-coefficient strip, selected penalty and C, labelled leverage caveat; W7.2 read-only factor explorer on factor_cards_v1
asked_by: user
asks: [A-20261008-01, A-20261008-20]
implemented_by: claude-code
questions: [Q1, QI]
workstream: W7
analyses: []
hypotheses: [H-I-01, H-S2-01, H-S2-08]
verdicts: []
alts: []
closes: [W7.1, W7.2]
advances: []
claim_level: descriptive
status: done
commit: pending
run_head: d387eda  # HEAD was the d387eda wip checkpoint when the checks ran (package spec said b60a6ee)
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "DONE: interactive heatmap explorer (patient-specific heatmaps colored by aggregates such as cohort-median lift, or by threshold-based metrics)" (A-20261008-01)

> "Can we view these interpretability results in the interactive app? (or explain why the app should not go to factor specifics)" (A-20261008-20)

Source: research/asks/2026-10-08_team-report.md (condensed wording; routes A-20261008-01 -> W7.1 and A-20261008-20 -> W7.2). The W7.1 items (per-fold strip, selected penalty and C, labelled leverage caveat) are reconstructed from the gap list in devlog 2026-10-01_d1-atlas-explorer-app and the W7 backlog; they are not a user quote.

## Why it matters
Q1 exploration and QI communication. The atlas marked a model as all-zero only when all 13 folds were zero, so cells that are all-zero in some supported folds looked like patient-specific capture; the strip makes the fold-level L1 threshold visible (H-S2-01, H-S2-08, H-I-01). For A-20261008-20 the design answer is: factor specifics stay out of the atlas (a representation x patient view) and live in a separate read-only page over the W5 factor_cards_v1 bundle, reached by a deep link from each atlas cell.

## Estimand / null
n/a (read-only viewers; nothing is refit). The strip shows, per atlas cell x LOPO fold, the nonzero-coefficient count of the stored Stage 2 model at the cell's selected regularization (D1 global rule: max pooled LOPO AUPRC, ties by AUROC, penalty elasticnet < l1 < l2, smaller C). c_policy: global_selected (stored; one penalty and C per cell, the same in all 13 folds). An all-zero fold scores a constant, so its held-out AP equals prevalence and its lift is 0 by construction. The factor explorer displays D18 card quantities only.

## Implementation
- W7.1, additive edits to scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py (R10; defaults unchanged, default colour mode still the cohort median):
  - new constants FOLD_TABLES (the four sharedness_lopo/by_heldout_patient.csv tables), FOLD_COLUMN_PREFIX, FOLD_MANIFEST_KEY, C_POLICY_TEXT, FACTOR_EXPLORER_PORT, FACTOR_CARDS_INDEX, CSS_W71;
  - new Atlas fields (sel_penalty, sel_C, sel_l1_ratio, fold_patients, fold_supported, fold_nonzero, fold_n_malignant, fold_n_normal, factor_card_reps) and `load_atlas(..., with_folds=True)`; `_attach_nonzero` now also fills the selected penalty / C and returns the joined table;
  - new functions build_fold_table, fold_wide_columns, fold_patient_meta, _attach_folds, zero_fold_count, zero_fold_patients, fold_strip, fold_strip_summary, selected_regularization_text, factor_explorer_href, factor_cards_status, leverage_caveat, fold_check;
  - two colour modes appended (supported folds with an all-zero model; selected C with the penalty as cell text), a hover line, an "all-zero supported folds" chip for partial cells, the per-fold strip and the "Open factor cards" deep link in the cell view, the labelled leverage-filter caveat (same texts as before, now with a bold label and a status line when the v2 CSV source is active);
  - --export-bundle carries 13 integer fold_nonzero_<patient> columns in the existing nonzero table and a lopo_folds manifest key (no new file); a pre-W7.1 bundle still loads without the strip;
  - --check: the original lines are unchanged and one line is appended ("per-fold activity joined at selected regularization"); new flag --factor-explorer-url (default: the request's host, port 8051).
- W7.2, new scripts/mrd_stage0_2/apps/old34_factor_explorer.py: read-only Dash page on port 8051. It reads only the factor_cards_v1 CSV/JSON (index at start, every other table on first use), resolves EXP through sc_classification.utils.paths, and does not import the atlas app. It has a persistent caveat banner, URL routing (`?panel=&method=&k=[&factor=]` or `?card=`), the factor list per representation, and a card view (chips, caveat badges with the v1 vocabulary, 13-fold coefficient strip with the "lift = 0 by construction" badge, per-patient univariate lift with the sign-agnostic and fitted single-factor values, both gene poles, score quantiles, ADT and cell-type tables, matches linked only when the matched card is in the bundle). It also has --check (11 lines) and --export-bundle (gzip CSV + manifest + schema doc, at most 20 MB).
- Tests: tests/test_old34_d1_atlas_explorer.py (13 cases appended; the 25 original cases unchanged), tests/test_old34_factor_explorer.py (new; synthetic schema-v1 bundle plus 2 real-bundle tests skipped without D18).
- Commands (repo root): `python scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py --check`; `python scripts/mrd_stage0_2/apps/old34_factor_explorer.py --check`; `python -m pytest -q tests/test_old34_d1_atlas_explorer.py tests/test_old34_factor_explorer.py`; each `--export-bundle DIR` to a new directory outside EXP.
- Ran in the agent shell (CPU seconds): atlas --check 9.5 s, factor explorer --check 5.2 s, the two test files 47 s. Both servers were started on side ports (18050, 18051), answered HTTP 200 and were stopped. The user's d1_atlas screen session (port 8050) was not touched and still runs the pre-W7.1 code.
- Provenance: HEAD d387eda with a dirty tree; the D18 bundle consumed was built at d387eda+dirty (devlog 2026-10-08_d18-factor-cards).

## Artifacts
- analysis/scorecards/stage2_diagnostic/w7_apps_check_v1/ (verification record only; the apps write nothing into EXP):
  - w7_apps_check_findings.md
  - w7_apps_check_manifest.json
  - w7_apps_check_artifact_ledger.csv
  - atlas_check.log
  - factor_explorer_check.log

## Results
- Atlas --check: 12 PASS, 1 INFO, 0 FAIL (exit 0). The original 11 PASS lines and the INFO line are unchanged; the appended line reads 783/783 cells x 13 LOPO folds (7 supported); fold mean / max nonzero counts equal the scorecard values (max |diff| 2.3e-13); supported-fold held-out AUPRC from the per-fold tables equals the D1 patient table (max |diff| 0.0); 4 per-fold tables unchanged by loading.
- [H-I-01 supports] All 757 all-zero supported (cell, fold) pairs at the selected regularization have |lift| <= 5.6e-17, i.e. constant-score AP equals the held-out prevalence.
- [H-S2-01 qualifies] 46 of 783 atlas cells (36 of the 640 biological DR cells) fit an all-zero model in 1-6 of the 7 supported folds, and 87 cells (80 biological DR) in all 7. In the 46 partial cells a patient's zero lift is a fold-level L1 threshold at the selected C, so per-patient counts such as "lift > tau" partly count all-zero folds.
- [H-S2-08 supports] IFN-alpha FA K=40 (selected l1, C = 1e-4) is active in 3 of 13 folds (P03, P04 and the unsupported P10) and all-zero in the other five supported folds (P01, P02, P06, P07, P09); PCA K=40 at the same penalty and C is active in 11 of 13 folds and all-zero only in P01 and P09. The coverage gap appears as whole-model all-zero folds at fixed C.
- Selected regularization over the 783 cells: l2 366, elasticnet 288, l1 129; C = 1e-4 in 528 cells.
- Atlas --export-bundle grows by 27,906 bytes (13 fold columns 27,124 bytes + lopo_folds manifest key 782 bytes; budget 100,000); the four exported tables total 2,155,955 bytes. The exported bundle reloads with identical fold arrays and its fold check passes. Load time in an experiment dir rose from about 0.7 s to about 8 s (reading the per-fold tables).
- Factor explorer --check on the real bundle: 11 PASS, 0 FAIL (exit 0): 1810 cards, 56 representations, 47 panels; 13 manifest row counts agree; no orphan card ids; 1810/1810 cards carry all 7 supported patients; all caveat codes known; 56/56 deep links resolve; views render for one card of every representation; IFN-alpha PC8 (orientation -1) and FA7 (+1) anchor cards present; 27/27 D18 regression checks passed; 14 bundle files unchanged by loading.
- Factor explorer --export-bundle: 6,958,416 bytes (limit 20,000,000); the export passes the same --check.
- Tests: 38 passed in tests/test_old34_d1_atlas_explorer.py (25 original + 13 new) and 23 passed in tests/test_old34_factor_explorer.py (21 synthetic + 2 real bundle).

## Claim gate
Supports (descriptive): exact, read-only views of the D1 atlas with per-fold Stage 2 activity at the stored global-selected regularization, and of the D18 factor cards on the transductive Stage 1 basis; an all-zero fold is constant-score no-skill AP = π.
Does not support: reading an all-zero fold or a zero beta as absence of signal (it is an L1 threshold at a fixed C); any refit or tuned-C statement (W3.2); inductive deployment (LOPO is label transfer); naming a factor as a unique program from the explorer alone (determinacy, eigengap and sign-stability caveats apply); claims from filter hits, which are leads, not tests.

## Next steps
- W7.1 / W7.2 (user, screen): restart the atlas so it serves the W7.1 code, then start the factor explorer next to it. From the sc_classification repo root:
  - `screen -r d1_atlas`, press Ctrl-C, then `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py --host 0.0.0.0 --port 8050`, then detach with Ctrl-a d (use your original flags if they differed).
  - `screen -S W7_factor_explorer`, then `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/apps/old34_factor_explorer.py --host 0.0.0.0 --port 8051`, then detach with Ctrl-a d. The deep link targets the same host on port 8051; through an SSH tunnel, forward both 8050 and 8051.
  - Verify: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8050/` and the same for 8051 return 200. In a browser, click an IFN-alpha FA K=40 cell: the strip shows five red all-zero supported folds, and "Open factor cards" opens the FA K=40 factor list on port 8051.
- W7.3: read EXPERIMENT_ID and EXPECTED from research/study_config.json in the atlas app (the factor explorer already resolves EXP through sc_classification.utils.paths).
- W5.4: promax determinacy shows as empty in the explorer until factosig saves Phi; FactoSig determinacy uses psi approximated as 1-h2 (D18 caveat).
