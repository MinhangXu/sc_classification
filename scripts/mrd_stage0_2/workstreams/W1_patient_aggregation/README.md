---
id: W1
title: W1_patient_aggregation: should patients share one basis, and do high-prevalence patients dominate it? (D16, D20, D25)
status: active
questions: [Q1, Q2]
---

# W1 — W1_patient_aggregation: should patients share one basis, and do high-prevalence patients dominate it? (D16, D20, D25)

Status as of 2026-10-08: active (P0). D14 is done on the transductive basis. D16 comes next and needs no other code (CPU).

## Charter

Verbatim asks (recorded in `research/asks/2026-10-08_team-report.md` and `research/asks/2026-06-04_pi-questions.md`):

> "Q1: shared or patient-specific?" (A-20261008-30)

> "TOP PRIORITY: data-driven statistical test for whether to aggregate patients" (A-20261008-07)

> "Step 1: learn factors on all patients, project each patient's cells ('we should be able to do this already')" (A-20261008-08)

> "Step 2: check per-patient factor distributions (plot from the canonical experiment)" (A-20261008-09)

> "Step 3: if distributions overlap significantly, patients share variation; if distinct, treat them as separate subtypes" (A-20261008-10)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02)

> "DISCUSSED (a): 'Need to inspect the factor-by-patient matrix for FA to test this'" (A-20261008-05)

> "Option 1.3: re-examine the PCA and FA factors to see whether the imbalance problem persists" (A-20261008-15)

> "Use days since transplant and days before relapse per patient to plot variation across patients and consider timing as a covariate (low priority; YH says a similar manuscript plot exists)" (A-20261008-18)

> "Can we build a robust model that separates malignant from non-malignant cells across patients?" (A-20260604-01, PI)

> "Can patient-specific models reveal new biology rather than simply repeat the shared model?" (A-20260604-02, PI)

Wording note (AGENTS.md R6): A-20261008-02 is quoted verbatim. P02, P03 and P04 are the highest-prevalence patients, not the patients with the most malignant cells; per-patient counts live in `research/study_config.json`.

Scope: Q1 (primary) and Q2. The user's top-priority aggregation test (A-20261008-07..10) is already answered by D14 on the transductive basis. The claim that P02/P03/P04 shape the factors (A-20261008-02), the FA factor-by-patient matrix (A-20261008-05), Option 1.3 (A-20261008-15) and timing (A-20261008-18) are open. The single-cell ML review adds a genotype/label-provenance shortcut audit (W1.4).

Done means: whether supported patients can share one Stage 1 basis (H-S1-04) and whether the highest-prevalence patients shape its loadings (H-S1-05) are answered with calibrated tests on the transductive basis and on the W2 bases; the P03-private broad-bag axis is attributed to biology or to a shortcut (H-S1-07); and shared vs patient-specific transfer is compared at matched prevalence and with held-out within-patient probes (H-S2-01, H-S2-02).

## Builds on

- Registry rows owned here: D1, D1c, D1d, D2, D3, D3b, D1f, D7, D13, D14, D1DD1, D1DD2, D1DD3, D1DD4, D1DD6, D1DD13. Planned: D16, D20, D25.
- Used from other workstreams: F6 (W5; univariate factor × patient AP for IFN-alpha PCA40/FA40), D17 (W2; the alternative bases for W1.3), D15 (W6).
- Hypotheses: H-S1-04, H-S1-05, H-S1-07, H-S2-01, H-S2-02.
- Code: `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py` (D14), `scripts/mrd_stage0_2/stage2_supervised/build_lopo_transfer_audit_tables.py` (LOPO transfer audit tables).
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_comprehensive_diagnostic_suite.ipynb` (D1-D3b, D1c-D1f), `stage2_old34_patient_factor_overlap.ipynb` (D14), `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` (D1DD), `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` (D13), `stage2_old34_meeting_followup_metric_interactions.ipynb` (D7).
- EXP-relative inputs: `stage1_dr/panel_id=*/method=*/k=*/seed=42/scores.npy` (transductive bases), the cohort cache (study_config `cohort_cache_relpath`), `analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/`, `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv`, `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`, `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f6_ifna_univariate_factor_ap_by_patient.csv`, `stage2_supervised/multiobjective/sharedness_lopo/`.
- Regression anchor: `anchors/factor_by_patient_scratch_20261008.py` (W1.1 must reproduce it).
- Devlog entries: 2026-10-02_d14-patient-factor-overlap, 2026-09-11_d1-atlas-783-d1e-d1f, 2026-09-10_d1a-descriptive-deep-dives, 2026-07-24_d1-d5-diagnostic-suites, 2026-06-23_lopo-transfer-audit-tables, 2026-06-04_three-pi-questions-v2-suite, 2026-05-28_figure3-sharedness-v1, 2026-10-08_team-report-devlog-pivot (fact-check corrections, including the scratch factor-by-patient check).
- Literature routing (`../W9_literature/README.md`): genotype audit and label provenance → W1.4; gene ablation → W1.8; MOFA+/cNMF sharedness and per-patient centering ablation → W1.3. Label remap: the review's "D16 genotype audit" is D25 here.

## Backlog

{{BACKLOG}}

## Code in this directory

- `anchors/factor_by_patient_scratch_20261008.py`: frozen regression anchor (IFN-alpha PCA40/FA40 factor-by-patient SS enrichment and within-patient malignant−normal shift). Not a pipeline module; never imported.
- `factor_patient_matrix.py`: wave 2 (planned). D16 library: per panel × method × K × factor × patient cell share, SS share and enrichment, standardized malignant−normal shift, eta2 by label, univariate held-out AP and lift with training-fold orientation, degenerate flag and effective_k.
- `run_d16_factor_patient_matrix.py`: wave 2 (planned). D16 runner with `--resume` per panel/method/K.
- `run_d16_factor_patient_matrix.sh`: wave 2 (planned). Screen wrapper for the all-41-panel extension.
- `tests/test_factor_patient_matrix.py` (repo-level): wave 2 (planned). Synthetic scores with a known allocation.
- D20 (`patient_factor_overlap_v2`) and D25 (`genotype_label_audit_v1`) code: later; no package assigned yet.
- Screen sessions: `W1_d16` (D16 all-panel extension, CPU). Later W1 runs use `W1_<topic>`.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `factor_patient_matrix_v1` (D16), `patient_factor_overlap_v2` (D20), `genotype_label_audit_v1` (D25). New IDs for W1.5, W1.6 and W1.8 come from `python tools/research_index.py reserve-id --workstream W1 --title '...'`. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Whenever training n changes (W1.5 malignant-cell subsampling, W1.6 within-patient CV, W1.3 and W1.8 refits), retune C per condition by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (wave 2) and report L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal". D16 and D20 fit no Stage 2 model (c_policy: not_applicable).
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
- Data: read the cohort cache or the cell metadata CSV, never the raw AnnData (R8). Label Stage 1 as transductive unless a W2 basis is used (R6).
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W1.n`.
