---
id: 2026-10-01_d1-atlas-explorer-app
date: 2026-10-01
type: tool
title: Interactive Dash D1 atlas explorer (hypothesis-generating heatmap)
asked_by: user
asks: [A-20261008-01]
implemented_by: unknown
questions: [Q1]
workstream: W7
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
status: done
commit: b60a6ee  # retro: app and tests first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # app and test files 2026-10-01 22:33 UTC
supersedes: []
---

## Question
> "I built the interactive plot that would help us to use the heatmap as a hypothesis generating engine. It shows patient specific heatmaps with either aggregated information like cohort-median-lift or threshold-based evaluation metrics."

Source: the user's 2026-10-08 team report, item A-20261008-01 (research/asks/2026-10-08_team-report.md). The original request of about 2026-10-01 was not saved.

## Why it matters
Q1 exploration and team communication: the app lets the team see, per patient, which panel x configuration cells carry the D1 lift, instead of one cohort median.

## Estimand / null
n/a (read-only viewer). It displays the D1/D1f quantities: median, mean and P20 lift over the 7 supported patients, patient counts above lift or AUPRC/π thresholds, one patient's lift or AUPRC, and the Stage 2 nonzero count at the selected regularization.

## Implementation
- App: scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py (Dash/Plotly); tests: tests/test_old34_d1_atlas_explorer.py; documented in scripts/mrd_stage0_2/README.md.
- Inputs (read only): the D1 atlas tables analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv and old34_d1_atlas_supported_patient_metrics.csv, and the D1f threshold matrices.
- Commands: python scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py with --check (validate, no server), --export-bundle DIR (about 2 MB for teammates) or --data-dir DIR.
- Served since 2026-10-02 00:15 UTC from the detached screen session d1_atlas on port 8050.
- At the time, dash/plotly were not in the in-repo env spec and pytest was in no conda env (W0.4).

## Artifacts
none

## Results
- Grid: 47 panels x 17 Stage 1 configurations (783 cells) x 7 supported patients (5,481 patient rows); 7 colour modes, 3 hypothesis filters (median > 0 with P20 ≤ 0; direct genes near zero with a strong DR cell; high single-patient leverage on the median), click-through patient view.
- --check reproduces the saved D1/D1f matrices with 0 differing cells (11 PASS lines on 2026-10-08); 25/25 tests pass with a scratch pytest install.
- Caveat: the high-leverage filter reads a CSV on a different regularization basis (146/164 cells differ, by up to 8.8 pp). There is no factor-level view.

## Claim gate
Supports (descriptive): an exact, read-only view of the D1/D1f atlas for hypothesis generation.
Does not support: claims made from filter hits (they are leads, not tests); any factor-level interpretation (the app stops at representation x patient lift).

## Next steps
- W7.1: per-fold active-coefficient strip, with the leverage-filter caveat labelled.
- W7.2: separate read-only factor explorer on the factor_cards_v1 bundle (A-20261008-20).
- W7.3: read the experiment id from research/study_config.json.
