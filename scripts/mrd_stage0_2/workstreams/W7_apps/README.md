---
id: W7
title: W7_apps: D1 atlas explorer and the separate read-only factor explorer page
status: active
questions: [Q1, QI]
---

# W7 — W7_apps: D1 atlas explorer and the separate read-only factor explorer page

Status as of 2026-10-08: active (P1). W7.1 (atlas per-fold strip) and W7.2 (factor explorer) are done (devlog 2026-10-08_w7-atlas-strip-factor-explorer). The user restarts the `d1_atlas` screen session so it serves the W7.1 code and starts `W7_factor_explorer`. W7.3 is open.

W7's code lives in `../../apps/` (that is, `scripts/mrd_stage0_2/apps/`), not in this directory. Apps go in `apps/` (AGENTS.md R10); this directory holds only this README.

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "DONE: interactive heatmap explorer (patient-specific heatmaps colored by aggregates such as cohort-median lift, or by threshold-based metrics)" (A-20261008-01)

> "Can we view these interpretability results in the interactive app? (or explain why the app should not go to factor specifics)" (A-20261008-20)

Scope: Q1 exploration and QI communication. The interactive heatmap is done (A-20261008-01). The answer to A-20261008-20 is a design decision: factor specifics stay out of the atlas app, which remains a representation × patient view, and a separate read-only factor explorer page reads the W5 factor-card bundle, with a deep link from each atlas cell (W7.2).

Done means: the atlas shows per-fold activity, the selected penalty and C, and the leverage-filter caveat (W7.1); the factor explorer serves the factor_cards_v1 bundle with its own check and export (W7.2); and both apps take the experiment id and the expected accounting from the study config, so they run against a second experiment (W7.3).

## Builds on

- Registry: D1 and D1f (W1; the atlas tables), D1e (W3), D18 (W5; the factor-card bundle).
- Code: `scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py` (read-only Dash app; `--check`, `--export-bundle DIR`, `--data-dir DIR`), `tests/test_old34_d1_atlas_explorer.py`.
- EXP-relative inputs: `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv`, `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`, the four per-fold `stage2_supervised/multiobjective/**/sharedness_lopo/by_heldout_patient.csv` tables for the W7.1 strip, and `analysis/scorecards/stage2_diagnostic/factor_cards_v1/` (D18).
- Bundle contract: `factor_cards_v1`, schema_version 1, specified in `../W5_factor_interpretation/README.md`.
- Study constants: `scripts/mrd_stage0_2/research/study_config.json` (W7.3 replaces the app's hard-coded EXPERIMENT_ID and EXPECTED).
- Environment: dash, plotly and pytest through `pip install -e '.[app,dev]'` (W0.4).
- Devlog entries: 2026-10-01_d1-atlas-explorer-app, 2026-10-08_team-report-devlog-pivot, 2026-10-08_d18-factor-cards, 2026-10-08_w7-atlas-strip-factor-explorer.

## Backlog

- [W7.1] (P1) Add a per-fold active-coefficient strip to apps/old34_d1_atlas_explorer.py — done when: --check passes and pytest passes.
  - Show, per atlas cell × supported fold, whether the fold is active or all-zero.
  - Show the selected penalty and C per cell.
  - Label the leverage-filter caveat.
  - The existing --check lines stay unchanged, plus one new PASS line.
  - The export bundle grows by less than 0.1 MB.
  - Extend the tests.
- [W7.2] (P1) Build a separate read-only factor explorer, apps/old34_factor_explorer.py (port 8051), that reads the factor_cards_v1 bundle — done when: --check passes on the real bundle and pytest passes. (depends: W5.1)
  - Deep link to it from the atlas cell view.
  - Give it its own --check, --export-bundle and --data-dir.
  - Test it on a synthetic bundle.
- [W7.3] (P2) Read EXPERIMENT_ID from research/study_config.json, and parameterize EXPECTED for a new experiment — done when: the app runs against a second experiment dir. (depends: W8.4)

## Code in this directory

None. W7's code lives in `../../apps/` (that is, `scripts/mrd_stage0_2/apps/`).

`../../apps/old34_d1_atlas_explorer.py` is the read-only D1 atlas explorer (Dash, port 8050).
- W7.1 added the following (devlog 2026-10-08_w7-atlas-strip-factor-explorer; additive, defaults unchanged):
  - **Per-fold strip** in the cell view. Supported folds come first, then the unsupported folds, greyed. An all-zero supported fold gets a "lift = 0 by construction" badge.
  - **Selected penalty, C and l1_ratio per cell.** They show in the strip header and the hover. A new "selected C (log10)" colour mode shows the penalty as cell text.
  - **New colour mode:** the number of supported folds with an all-zero model.
  - **Labelled leverage-filter caveat.**
  - **"Open factor cards" deep link** to the factor explorer (`?panel=&method=&k=`). The default target is this request's host on port 8051; `--factor-explorer-url` overrides it.
- New inputs (read-only):
  - the four `stage2_supervised/multiobjective/**/sharedness_lopo/by_heldout_patient.csv` tables. They give the per-fold nonzero counts, matched to each cell's selected penalty, C and l1_ratio;
  - optionally, `analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_index.csv`, for the coverage note next to the deep link.
- `--check` keeps its original lines and appends one: "per-fold activity joined at selected regularization".
- `--export-bundle` carries the per-fold counts as integer `fold_nonzero_<patient>` columns of the existing nonzero table, plus a `lopo_folds` manifest key. No new file is added, and older bundles still load without the strip.
- Startup in an experiment dir now reads the per-fold tables (seconds). An exported bundle starts fast.

`../../apps/old34_factor_explorer.py` (W7.2) is the read-only factor explorer (Dash, port 8051). It reads the W5 `factor_cards_v1` bundle (schema_version 1; contract in `../W5_factor_interpretation/README.md`).
- Data access:
  - It reads only the bundle CSV/JSON, through `--data-dir`. That can be an experiment dir or an exported bundle; the default experiment dir comes from `sc_classification.utils.paths`.
  - The index is read at start; every other table is read on first use.
  - It does not import the atlas app.
- What it shows:
  - a persistent caveat banner;
  - URL routing: `?panel=&method=&k=[&factor=]` or `?card=<card_id>`;
  - a factor list for each representation;
  - a card view: orientation; a 13-fold coefficient strip with the all-zero badge; per-patient univariate lift (training-fold sign), with the sign-agnostic and fitted single-factor values as markers; both gene poles; score quantiles; ADT and cell-type tables; and matches. A match is linked only when the matched card is in the bundle.
- `--check` validates the schema and the manifest counts, referential integrity, that the deep links resolve, that the views render, the IFN-alpha anchor cards, the producer's regression checks, and a read-only fingerprint.
- `--export-bundle DIR` writes gzip CSVs, the manifest and the schema doc, under a size cap.

Tests:
- `tests/test_old34_d1_atlas_explorer.py`: the original cases are unchanged; the W7.1 cases are appended. They use synthetic per-fold tables; the real-data cases skip without EXP.
- `tests/test_old34_factor_explorer.py`: uses a synthetic schema-v1 bundle. The real-bundle cases skip until D18 is present.

Verification record: `analysis/scorecards/stage2_diagnostic/w7_apps_check_v1/` (EXP-relative). It holds check logs, findings, a manifest and a ledger. The agent wrote it; the apps themselves write nothing into EXP.

Screen sessions (servers, started by the user):
- `d1_atlas`: the atlas on port 8050. The name predates the convention and is kept. Restart it so it serves the W7.1 code (commands in the Next steps of devlog 2026-10-08_w7-atlas-strip-factor-explorer).
- `W7_factor_explorer`: the factor explorer on port 8051.
- Through an SSH tunnel, forward both ports.

Task status (derived from the devlog; see `../../research/STATUS.md`): W7.1 and W7.2 are closed by devlog 2026-10-08_w7-atlas-strip-factor-explorer. Follow-up for the user: restart `d1_atlas`, start `W7_factor_explorer`, then click-test one deep link in a browser. W7.3 is open: the atlas app still hard-codes EXPERIMENT_ID and EXPECTED, while the factor explorer already resolves EXP through paths.py and checks against the bundle manifest.

## Conventions

- Owned artifact slugs: only the agent-written verification record `analysis/scorecards/stage2_diagnostic/w7_apps_check_v1/` (check logs, findings, manifest, ledger). The apps themselves are read-only: they read the D1/D1f tables, the per-fold LOPO tables and the W5-owned `factor_cards_v1` bundle, and `--export-bundle` writes a NEW directory outside EXP. Apps never write into the experiment dir (R3).
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train); whenever training n changes, C is retuned per condition by inner LOPO and L2 is reported alongside. The apps refit nothing: they display the stored penalty and C per cell, label the c_policy behind each view, and never present an all-zero L1 cell as "no signal" (its constant-score no-skill AP = π by construction).
- Import header: app code lives in `apps/`, two levels below the study root, so new app code resolves the repo as `Path(__file__).resolve().parents[3]` and then uses `sc_classification.utils.paths`; a new script in a workstream directory uses:

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```

- Screen sessions are named `<W#>_<topic>` (the existing `d1_atlas` session is grandfathered). The agent runs only `--check` in its shell; the user starts and restarts the servers in screen.
- Edits to the existing atlas app are additive only, with defaults unchanged, and named in the devlog entry (R10).
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib, as `tests/test_old34_d1_atlas_explorer.py` does.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W7.n`.
