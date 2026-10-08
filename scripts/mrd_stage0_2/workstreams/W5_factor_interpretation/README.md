---
id: W5
title: W5_factor_interpretation: IFN-alpha template to generalized factor cards and multimodal grounding (D18)
status: active
questions: [QI, Q1]
---

# W5 — W5_factor_interpretation: IFN-alpha template to generalized factor cards and multimodal grounding (D18)

Status as of 2026-10-08: active (P0). D5 and D8 exist, as do F3/F6/F7/F9 and D1DD10-12. The factor-card bundle contract is fixed (see Code in this directory), so W7.2 can be built in parallel.

## Charter

Verbatim asks (recorded in `research/asks/2026-06-04_pi-questions.md` and `research/asks/2026-10-08_team-report.md`):

> "Does each prior program behave as one gene program, or does it contain distinct regulatory groups?" (A-20260604-03, PI; = QI)

> "Factor interpretability to insight, with IFN-alpha as the example; expand beyond IFN-alpha and organize alongside the top priority" (A-20261008-19)

> "DISCUSSED (b): PC8 loadings form two opposite-sign gene groups (IFI27, PSME2, PSMA3, CNP, LAP3, USP18 vs TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1), plus a trailing 'IFI27, IFITM2'" (A-20261008-06)

> "Correlate factors with other data modalities (cell types, surface protein expression)" (A-20261008-21)

> "Review interpretability frameworks from other fields (e.g. factor rotation in psychology; iterate with domain knowledge)" (A-20261008-22)

Earlier asks, cited by ID (their recorded text is in `research/asks/`): A-20260724-03 (IFN-alpha PCA40 vs FA40 → D8; capture reconstructed), A-20260917-02 (DR vs direct genes, LCK/ZAP70 → F2, F5).

Scope: QI (= PI-Q3: does a prior program break into regulatory sub-axes?), feeding Q1. Covers "factor interpretability to insight, with IFN-alpha as the example" (A-20261008-19), the PC8 loadings (A-20261008-06), other modalities (A-20261008-21) and frameworks from other fields (A-20261008-22).

Done means: a factor-card bundle (D18) covers the pre-declared shortlist with oriented genes from both poles, selection and sign stability, per-patient univariate lift, cross-method matching and caveats; the PC8/FA7 cards reproduce D8, F6 and F9; a census answers QI program by program (W5.5, H-S2-03); multimodal associations state their confounds (W5.3, H-S2-05); and the interpretability method questions are settled in a frameworks note (W5.4, H-I-02, H-I-03). W7.2 renders the bundle.

## Builds on

- Registry rows owned here: D5, D8, D1DD10, D1DD11, D1DD12, F5, F6, F7, F9. Planned: D18.
- Used from other workstreams: F3 (W3; IFN-alpha collinearity and L1/L2/EN refits), D14 (W1; the 47 representations of the shortlist), D19 and W3.2 (W3; tuned-C probes for W5.2).
- Hypotheses: H-S2-03, H-S2-05, H-S2-07, H-I-02, H-I-03; H-S2-08 (owned by W3) for the IFN-alpha case.
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_layer4_5_interpretability_suite.ipynb` (D5), `stage2_old34_meeting_followup_metric_interactions.ipynb` (D8), `stage2_old34_feedback_deep_dives_20260917.ipynb` (F5-F7, F9), `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` (D1DD10-D1DD12).
- EXP-relative inputs: `stage1_dr/panel_id=*/method=*/k=*/seed=42/` (`scores.npy`, `loadings.npy`, `metadata.json`, `diagnostics.json`), the cohort cache (study_config `cohort_cache_relpath`), `analysis/scorecards/stage2_diagnostic/layer4_5/` (coefficient stability, selected factors, top loading genes), `analysis/scorecards/stage2_diagnostic/meeting_followup_v1/` (`interferon_alpha_*` tables, D8), `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/` (`f6_*`, `f7_*`, `f9_*` tables), `analysis/scorecards/stage2_diagnostic/d1a_deep_dive_20260910/`.
- Regression anchor: `anchors/gene_direction_scratch_20261008.py` (W5.2 and the PC8 card must reproduce it).
- Devlog entries: 2026-10-02_f9-ifna-pc8-loadings, 2026-09-17_team-feedback-f1-f7, 2026-08-31_d6-d9-meeting-followup, 2026-07-24_d1-d5-diagnostic-suites, 2026-10-08_team-report-devlog-pivot, 2026-10-08_literature-reviews.
- Literature routing (`../W9_literature/README.md`): pattern and structure genes, effective_k and determinacy, many-to-one matching → W5.1; Haufe activation patterns → W5.2; promax pattern matrix → W5.4; span-angle test → W5.5; target rotation and Spectra/PLIER → W5.6.

## Backlog

{{BACKLOG}}

## Code in this directory

- `anchors/gene_direction_scratch_20261008.py`: frozen regression anchor (within-patient malignant−normal mean of z-scored log1p expression for the PC8 top genes). Not a pipeline module; never imported.
- `factor_cards.py`: wave 2 (planned). D18 card builder that writes the factor_cards_v1 bundle below; computes eta2 itself and does not depend on D16.
- `run_d18_factor_cards.py`: wave 2 (planned). D18 runner with `--resume` per card.
- `run_d18_factor_cards.sh`: wave 2 (planned). Screen wrapper.
- `tests/test_factor_cards.py` (repo-level): wave 2 (planned). Synthetic loadings/scores fixture plus schema validation.
- Screen sessions: `W5_d18` (D18, if it exceeds about 5 minutes). Later W5 runs use `W5_<topic>`.

Bundle contract `factor_cards_v1` (schema_version 1; the interface between D18 and `../../apps/old34_factor_explorer.py`, W7.2). EXP-relative directory `analysis/scorecards/stage2_diagnostic/factor_cards_v1/`; the app reads only the CSV and JSON files, and tests use a synthetic bundle with the same schema:
- `factor_cards_manifest.json`: schema_version 1, inputs, git_state, created_by.
- `factor_cards_index.csv`: card_id (`<panel_id>__<method>__k<K>__f<NNN>`), panel_id, method, k, factor, is_selected_driver, n_active_folds, sign_consistency, orientation_sign, eta2_normal, eta2_malignant, degenerate_flag, effective_k, determinacy, matched_cards (';'-separated), caveat_flags.
- `factor_cards_genes.csv`: card_id, gene, loading, oriented_loading, rank, pole, structure_corr (oblique fits only).
- `factor_cards_patient_lift.csv`: card_id, patient, univariate_lift, univariate_auroc, n_malignant, prevalence.
- `factor_cards_fold_coefs.csv`: card_id, heldout_patient, penalty, C, coef.
- `cards/<card_id>.md`: one human-readable card per factor.
Schema changes bump schema_version and go through a devlog entry with type: refactor, and W7.2 must be updated in the same change.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `factor_cards_v1` (D18). New IDs for W5.2-W5.6 come from `python tools/research_index.py reserve-id --workstream W5 --title '...'`. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Cards read stored coefficients at the stored C; any refit (W5.2 probes, W5.5 per-patient directions, W5.6 rotated loadings) retunes C per condition by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (wave 2) and reports L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal".
- Interpretation language (R6): for any loading, state the sign orientation and the penalty (L1/L2/EN) and name genes from both poles; factosig_promax loadings are a pattern matrix, so report structure correlations alongside; descriptive only, no causal claims.
- Import header for a new script in this directory (files in `anchors/` sit one level deeper and use `parents[5]`):

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```

- Screen sessions are named `<W#>_<topic>`. Long or GPU jobs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with compute: screen-cpu or screen-gpu. New runners implement `--resume` (R2).
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W5.n`.
