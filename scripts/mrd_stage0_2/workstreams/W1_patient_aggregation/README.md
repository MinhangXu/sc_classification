---
id: W1
title: W1_patient_aggregation: should patients share one basis, and do high-prevalence patients dominate it? (D16, D20, D25)
status: active
questions: [Q1, Q2]
---

# W1 — W1_patient_aggregation: should patients share one basis, and do high-prevalence patients dominate it? (D16, D20, D25)

Status as of 2026-10-09: active (P0). D14 is done on the transductive basis, D16 (W1.1) is done (devlog 2026-10-08_d16-factor-patient-matrix), and W1.9 is done (devlog 2026-10-09_d28-celltype-score-shares): D28, the cell-type-stratified score-energy shares, on the D16 grid. Next: D20 (W1.2), then the W1.3 rerun on the D17 bases.

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md` and `scripts/mrd_stage0_2/research/asks/2026-06-04_pi-questions.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

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

Wording note (AGENTS.md R6): A-20261008-02 is quoted as recorded in the asks file. P02, P03 and P04 are the highest-prevalence patients, not the patients with the most malignant cells; per-patient counts live in `scripts/mrd_stage0_2/research/study_config.json`.

Scope: Q1 (primary) and Q2. The user's top-priority aggregation test (A-20261008-07..10) is already answered by D14 on the transductive basis. The claim that P02/P03/P04 shape the factors (A-20261008-02), the FA factor-by-patient matrix (A-20261008-05), Option 1.3 (A-20261008-15) and timing (A-20261008-18) are open. The single-cell ML review adds a genotype/label-provenance shortcut audit (W1.4).

Done means: whether supported patients can share one Stage 1 basis (H-S1-04) and whether the highest-prevalence patients shape its loadings (H-S1-05) are answered with calibrated tests on the transductive basis and on the W2 bases; the P03-private broad-bag axis is attributed to biology or to a shortcut (H-S1-07); and shared vs patient-specific transfer is compared at matched prevalence and with held-out within-patient probes (H-S2-01, H-S2-02).

## Builds on

- Registry rows owned here: D1, D1c, D1d, D2, D3, D3b, D1f, D7, D13, D14, D1DD1, D1DD2, D1DD3, D1DD4, D1DD6, D1DD13, D16, D28. Planned: D20, D25.
- Used from other workstreams: F6 (W5; univariate factor × patient AP for IFN-alpha PCA40/FA40), D17 (W2; the alternative bases for W1.3, and the leave-out loading-leverage test of H-S1-05), D15 (W6).
- Hypotheses: H-S1-04, H-S1-05, H-S1-07, H-S2-01, H-S2-02.
- Code: `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py` (D14), `scripts/mrd_stage0_2/stage2_supervised/build_lopo_transfer_audit_tables.py` (LOPO transfer audit tables).
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_comprehensive_diagnostic_suite.ipynb` (D1-D3b, D1c-D1f), `stage2_old34_patient_factor_overlap.ipynb` (D14), `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` (D1DD), `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` (D13), `stage2_old34_meeting_followup_metric_interactions.ipynb` (D7).
- EXP-relative inputs: `stage1_dr/panel_id=*/method=*/k=*/seed=42/scores.npy` (transductive bases), the cohort cache (study_config `cohort_cache_relpath`), `analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/`, `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv`, `analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`, `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f6_ifna_univariate_factor_ap_by_patient.csv`, `stage2_supervised/multiobjective/sharedness_lopo/`.
- Regression anchor: `anchors/factor_by_patient_scratch_20261008.py` (W1.1 must reproduce it).
- Devlog entries: 2026-10-02_d14-patient-factor-overlap, 2026-09-11_d1-atlas-783-d1e-d1f, 2026-09-10_d1a-descriptive-deep-dives, 2026-07-24_d1-d5-diagnostic-suites, 2026-06-23_lopo-transfer-audit-tables, 2026-06-04_three-pi-questions-v2-suite, 2026-05-28_figure3-sharedness-v1, 2026-10-08_team-report-devlog-pivot (fact-check corrections, including the scratch factor-by-patient check).
- Literature routing (`../W9_literature/README.md`): genotype audit and label provenance → W1.4; gene ablation → W1.8; MOFA+/cNMF sharedness and per-patient centering ablation → W1.3. Label remap: the review's "D16 genotype audit" is D25 here.

## Backlog

- [W1.1] (P0) D16 factor × patient variance-allocation matrix (slug factor_patient_matrix_v1) — done when: the tables, figures D16A-D16D, findings and manifest exist, and the IFN-alpha K40 rows reproduce anchors/factor_by_patient_scratch_20261008.py and F6 f6_ifna_univariate_factor_ap_by_patient.csv.
  - Coverage: PCA and FA at K in {5,10,20,40}, for IFN-alpha plus the focus shortlist (allograft rejection, apoptosis, TNFa/NF-kB). Then all 41 biological panels, as a resumable extension run in user screen.
  - Per factor × patient: cell share, sum-of-squares share and enrichment, standardized malignant−normal shift, eta2 by label, univariate held-out AP/lift with training-fold orientation, degenerate flag, and effective_k.
- [W1.2] (P0) D20 = D14 v2 (slug patient_factor_overlap_v2) — done when: a v1-vs-v2 call table and the calibrated thresholds are written. (depends: W1.1)
  - Handle degenerate or blown-up promax factors in the variance weights.
  - Cover all PCA/FA representations.
  - Add a joint multi-factor overlap.
  - Use ≥200 permutations and simulation-calibrated gates.
  - Subsample malignant cells to match the support of the smallest supported patient.
  - Re-check the allograft call.
- [W1.3] (P0) Option 1.3: re-run D20 and D16 on the W2 bases (inductive, leave-high-prevalence-out, patient-balanced), aligned to the transductive basis. A MOFA+/cNMF sharedness comparator is optional — done when: per-basis call and allocation tables are compared with the transductive results. (depends: W1.2, W2.1)
- [W1.4] (P1) D25 genotype/label-provenance shortcut audit — done when: the findings state which of genotype, label assay, batch or composition can explain the axis (H-S1-07).
  - Read the loadings of the D14 P03-private factors for HLA, ERAP2, chrY genes and XIST.
  - Compare P03 donor cells with P03 malignant cells on those factors.
  - Score the recipient-healthy cells with the stored LOPO models.
  - Build a donor/recipient sex-match table.
  - Add a per-patient label-provenance column (Chen et al. 2025, Table S2) to the D3 and D13 views.
- [W1.5] (P1) Separate prevalence from biology: prevalence/support-matched lift via malignant-cell subsampling, plus normalized lift, for the D1 atlas — done when: matched-lift tables exist for the supported patients and the ALT-PREV state of H-S2-01 can be updated.
- [W1.6] (P2) Test the predictive patient-specific signal: within-patient CV probes vs the shared model. This replaces the apparent, in-sample D3 patient-specific fits — done when: a per-patient held-out comparison with uncertainty exists.
- [W1.7] (P3) Timing, descriptive only: a D6-style scatter of days-from-SCT and days-to-relapse against prevalence and median lift, plus the P09 within-patient MRD series. Request the manuscript plot through W8.1 — done when: a figure and note exist. No covariate model.
- [W1.8] (P2) Ablate sex-chromosome and HLA genes from the HVG3k/HVG10k/full_34 bags, then re-run the matching D1 rows and D14. The Stage 1 refit runs in user screen — done when: ablated and original lift and overlap calls are tabulated side by side. (depends: W1.4)
- [W1.9] (P2) Cell-type-stratified D16: compute the factor × patient enrichment within predicted.annotation strata, to separate patient dispersion (P01, P02) from cell-type composition — done when: a stratified allocation table and a findings note exist. (depends: W1.1)
- [W1.10] (P2) Locate the within-cell-type excess of P02 (all bases) and P07 (IFN-alpha K40) that D28 leaves unexplained, by factor and cell type, from the D28 factor_long and celltype tables. Check whether it tracks sample, collection time or label provenance (with W1.4) — done when: a findings note names the factors and cell types that carry the residual and states whether it follows sample or batch. (depends: W1.9)

## Code in this directory

- `anchors/factor_by_patient_scratch_20261008.py`: frozen regression anchor (IFN-alpha PCA40/FA40 factor-by-patient SS enrichment and within-patient malignant−normal shift). Not a pipeline module; never imported. Its printed IFN-alpha K40 values are embedded in the runner's `ANCHOR` dict; if the anchor is ever re-derived, update `ANCHOR` too.
- `factor_patient_matrix.py`: D16 (W1.1) computation module, pure numpy/pandas, with no experiment paths. For one stored Stage 1 score matrix and per-cell patient/label arrays it returns the factor × patient table:
  - cell share, SS share (about the cohort mean), and enrichment split exactly into location + spread;
  - the same quantities within normal cells only and within malignant cells only;
  - the within-patient malignant − normal shift (cohort-SD units and Cohen's d);
  - eta2 of patient within label (D14 strata) with per-patient contributions;
  - univariate held-out AP and lift with the training-fold sign (the F6 sign-agnostic value is kept as an extra column);
  - the degenerate flag.
  It also provides the summaries `factor_summary`, `representation_patient_summary`, `representation_summary` and `top_patient_frequency`.
- `run_d16_factor_patient_matrix.py`: D16 runner. It reads `stage1_dr/panel_id=*/method={pca,fa}/k=*/seed=42/` and the cell metadata.
  - Writes one part per panel × method × K under `analysis/cache/stage2_diagnostic/factor_patient_matrix_v1/parts/`.
  - Merges the tables, findings, manifest and ledger into `analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1/`, and figures D16A-D16D into `analysis/figures/stage2_diagnostic/factor_patient_matrix_v1/`.
  - Regression checks: F6, the scratch anchor and the D14 eta2.
  - Flags: `--panels pilot|all`, `--panel-ids`, `--methods`, `--ks`, `--workers` (capped at 12), `--resume` (skips complete parts; empty or corrupt parts are recomputed), `--merge-only` (a failed merge writes merge_failed.json), `--no-figures`, `--experiment-dir`.
  - It reads `stage1_dr/` only. Rerunning D16 on the W2 bases (W1.3) needs an additive `--stage1-root` flag (for `stage1_dr_variants/<slug>/` and the D17 bases in the cache).
- `run_d16_factor_patient_matrix.sh`: screen wrapper (`screen -S W1_d16`). Runs all biological panels with `--resume --workers 12` and single-thread BLAS; extra arguments pass through. D16 is complete, so a rerun only re-merges.
- Tests: `tests/test_factor_patient_matrix.py` (synthetic known-allocation tests, a runner resume/merge test, and EXP-gated regression tests).
- `run_d28_celltype_score_shares.py`: D28 (W1.9) runner and computation, pure numpy/pandas. For each stored transductive PCA/FA basis (the D16 grid), it compares each patient's (and the P02+P03+P04 group's) observed score-energy ratio, energy share / cell share as in D16, with the ratio expected if every cell carried its predicted cell type's average energy (`predicted.annotation`).
  - Residual ratio = observed / expected (indirect standardization).
  - Variants: ss_weighted (D16) vs standardized energy; cohort vs leave-own-patient-out reference; cell type vs cell type x label strata; all cells vs normal cells only. Per-factor observed/composition/residual enrichment and top patient before/after.
  - Parts are .npz sufficient statistics under `analysis/cache/stage2_diagnostic/factor_patient_matrix_celltype_v1/parts/`. Tables, findings, manifest and ledger go to `analysis/scorecards/stage2_diagnostic/factor_patient_matrix_celltype_v1/`, and figures D28A-D28C to `analysis/figures/stage2_diagnostic/factor_patient_matrix_celltype_v1/`.
  - Regression: reproduces D16 (representation, group and per-factor) and the D16 headlines, plus the decomposition identities.
  - Flags: `--panels pilot|all` (default all), `--panel-ids`, `--methods`, `--ks`, `--workers` (capped at 12), `--resume` (empty, corrupt or inconsistent parts are recomputed), `--merge-only` (a failed merge writes merge_failed.json), `--out-root` (outputs outside EXP, for smoke runs), `--no-figures`, `--experiment-dir`.
  - It imports `run_d16_factor_patient_matrix.py` (panel list, score paths, cache-order check) and `factor_patient_matrix.py` (support rule), unchanged. It reads `stage1_dr/` only; W1.3 needs the same additive `--stage1-root` flag as D16.
- `run_d28_celltype_score_shares.sh`: wrapper (all panels, `--resume`, 12 single-thread workers; extra args pass through). The full run takes under a minute, so a screen session (`W1_d28`) is optional.
- Tests: `tests/test_d28_celltype_score_shares.py` (known-answer synthetic cohorts, a brute-force per-cell cross-check, D16 consistency, a runner resume/merge/out-root test, and EXP-gated regressions).
- D20 (`patient_factor_overlap_v2`) and D25 (`genotype_label_audit_v1`) code: later; no package assigned yet. D20 (W1.2) can take the degenerate flags and factor SS shares from `factor_patient_matrix_factor_summary.csv`.
- Screen sessions: `W1_d16` (optional D16 re-merge); `W1_d28` (optional D28 re-merge). Later W1 runs use `W1_<topic>`.

Task status (derived from the devlog; see `../../research/STATUS.md`): W1.1 is closed by devlog 2026-10-08_d16-factor-patient-matrix. W1.9 was added from that entry's Next steps and is closed by devlog 2026-10-09_d28-celltype-score-shares, whose Next steps added W1.10.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `factor_patient_matrix_v1` (D16), `factor_patient_matrix_celltype_v1` (D28), `patient_factor_overlap_v2` (D20), `genotype_label_audit_v1` (D25). New IDs for W1.5, W1.6 and W1.8 come from `python tools/research_index.py reserve-id --workstream W1 --title '...'`. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Whenever training n changes (W1.5 malignant-cell subsampling, W1.6 within-patient CV, W1.3 and W1.8 refits), retune C per condition by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (D19) and report L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal". D16 and D20 fit no Stage 2 model (c_policy: not_applicable).
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
