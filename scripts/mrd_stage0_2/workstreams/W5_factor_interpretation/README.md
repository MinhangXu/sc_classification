---
id: W5
title: W5_factor_interpretation: IFN-alpha template to generalized factor cards and multimodal grounding (D18)
status: active
questions: [QI, Q1]
---

# W5 — W5_factor_interpretation: IFN-alpha template to generalized factor cards and multimodal grounding (D18)

Status as of 2026-10-10: active (P1; no P0 task is open since W5.1 closed). D18 (W5.1) is done (devlog 2026-10-08_d18-factor-cards), and the W7.2 factor explorer renders its bundle. D27 (W5.3) ran on every non-degenerate D18 card (devlog 2026-10-09_d27-factor-modality-assoc) and was restated for collaborators in devlog 2026-10-10_protein-association-for-pavan-v2 (A-20261010-01..06); W5.3 can close once the F7b column-label fix lands, and dsb stays blocked. Next: W5.5 (census from the cards), W5.2 and W5.4.

To add a W5 analysis: reserve a D-ID (AGENTS.md §5), put the runner here as `run_<topic>.py` (+ `.sh` for screen) reading the D18 bundle and the cohort cache (never the raw AnnData), write under a new `<slug>_v1` (R3), and test it in `tests/`. `run_d27_factor_modality_assoc.py` is the nearest template (resume units, CLR, isotype floor, the robust rule).

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-06-04_pi-questions.md` and `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "Does each prior program behave as one gene program, or does it contain distinct regulatory groups?" (A-20260604-03, PI; = QI)

> "Factor interpretability to insight, with IFN-alpha as the example; expand beyond IFN-alpha and organize alongside the top priority" (A-20261008-19)

> "DISCUSSED (b): PC8 loadings form two opposite-sign gene groups (IFI27, PSME2, PSMA3, CNP, LAP3, USP18 vs TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1), plus a trailing 'IFI27, IFITM2'" (A-20261008-06)

> "Correlate factors with other data modalities (cell types, surface protein expression)" (A-20261008-21)

> "Review interpretability frameworks from other fields (e.g. factor rotation in psychology; iterate with domain knowledge)" (A-20261008-22)

Earlier asks, cited by ID (their recorded text is in `scripts/mrd_stage0_2/research/asks/`): A-20260724-03 (IFN-alpha PCA40 vs FA40 → D8; capture reconstructed), A-20260917-02 (DR vs direct genes, LCK/ZAP70 → F2, F5).

Scope: QI (= PI-Q3: does a prior program break into regulatory sub-axes?), feeding Q1. Covers "factor interpretability to insight, with IFN-alpha as the example" (A-20261008-19), the PC8 loadings (A-20261008-06), other modalities (A-20261008-21) and frameworks from other fields (A-20261008-22).

Done means: a factor-card bundle (D18) covers the pre-declared shortlist with oriented genes from both poles, selection and sign stability, per-patient univariate lift, cross-method matching and caveats; the PC8/FA7 cards reproduce D8, F6 and F9; a census answers QI program by program (W5.5, H-S2-03); multimodal associations state their confounds (W5.3, H-S2-05); and the interpretability method questions are settled in a frameworks note (W5.4, H-I-02, H-I-03). W7.2 renders the bundle.

## Builds on

- Registry rows owned here: D5, D8, D1DD10, D1DD11, D1DD12, F5, F6, F7, F9, D18, D27, D29.
- Used from other workstreams: F3 (W3; IFN-alpha collinearity and L1/L2/EN refits), D14 (W1; the 47 representations of the shortlist), D19 and W3.2 (W3; tuned-C probes for W5.2).
- Hypotheses: H-S2-03, H-S2-05, H-S2-07, H-I-02, H-I-03; H-S2-08 (owned by W3) for the IFN-alpha case.
- Notebooks (under `scripts/mrd_stage0_2/notebooks/stage0_2/`): `stage2_old34_layer4_5_interpretability_suite.ipynb` (D5), `stage2_old34_meeting_followup_metric_interactions.ipynb` (D8), `stage2_old34_feedback_deep_dives_20260917.ipynb` (F5-F7, F9), `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` (D1DD10-D1DD12).
- Cell-level obs (cohort cache and cell metadata CSV; AGENTS.md R8): patient, CN.label, predicted.annotation(.score), predicted.pseudotime(.score), Time, sample, timepoint_type and others; predicted.* are reference-mapping transfers.
- ADT: the cohort cache `obsm["ADT"]` holds raw counts for 170 features including isotype controls (loaded by `run_d18_factor_cards.load_cohort`; CLR via `factor_cards.clr_per_cell`). Neither the cohort cache nor the raw AnnData holds dsb-normalized ADT or empty-droplet background (their obs carry only nCount_dsb / nFeature_dsb), and the raw AnnData must not be loaded (AGENTS R8).
- EXP-relative inputs: `stage1_dr/panel_id=*/method=*/k=*/seed=42/` (`scores.npy`, `loadings.npy`, `metadata.json`, `diagnostics.json`), the cohort cache (study_config `cohort_cache_relpath`), `analysis/scorecards/stage2_diagnostic/layer4_5/` (coefficient stability, selected factors, top loading genes), `analysis/scorecards/stage2_diagnostic/meeting_followup_v1/` (`interferon_alpha_*` tables, D8), `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/` (`f6_*`, `f7_*`, `f9_*` tables), `analysis/scorecards/stage2_diagnostic/d1a_deep_dive_20260910/`.
- Regression anchor: `anchors/gene_direction_scratch_20261008.py` (W5.2 and the PC8 card must reproduce it).
- Devlog entries: 2026-10-02_f9-ifna-pc8-loadings, 2026-09-17_team-feedback-f1-f7, 2026-08-31_d6-d9-meeting-followup, 2026-07-24_d1-d5-diagnostic-suites, 2026-10-08_team-report-devlog-pivot, 2026-10-08_literature-reviews.
- Literature routing (`../W9_literature/README.md`): pattern and structure genes, effective_k and determinacy, many-to-one matching → W5.1; Haufe activation patterns → W5.2; promax pattern matrix → W5.4; span-angle test → W5.5; target rotation and Spectra/PLIER → W5.6.

## Backlog

- [W5.1] (P0) D18 generalized factor cards (slug factor_cards_v1; bundle schema v1, consumed by W7.2) — done when: the bundle and per-card md exist, and the PC8/FA7 cards reproduce the D8, F6 and F9 tables.
  - Shortlist, declared in advance: the 47 D14 representations, IFN-alpha PCA40/FA40 and the focus cells.
  - Each card holds:
    - oriented genes, sign(beta) × loading, from both poles;
    - per-fold selection and sign stability;
    - univariate held-out lift per patient;
    - per-patient score distribution and eta2;
    - many-to-one matching across method and K (R2, principal angles, Tucker phi);
    - for promax fits, both pattern and structure top genes, plus the factor correlations;
    - FA determinacy;
    - effective_k;
    - a caveat block.
- [W5.2] (P1) Oriented per-patient gene-direction table for PC8 and FA7 (anchors/gene_direction_scratch_20261008.py). Also: Haufe activation patterns for the PCA40-L1, FA40-L2 and FA40-L1-at-knife-edge probes, compared by Tucker phi; and a note contrasting IFN-gamma with type-I genes — done when: the table and note exist. (depends: W3.2)
- [W5.3] (P1) Factor × predicted cell type, and factor × CLR ADT, for the shortlist, stratified by label and patient (dsb is blocked on a background/source decision: ask the user; do not load the raw AnnData, R8). Analysis D27, slug `factor_modality_assoc_v1` (all cards, not only the shortlist), which reads `factor_cards_v1`; do not change the factor_cards_v1 schema. Fix the F7b column label: it holds raw counts, not log1p. State the confounds — done when: the association tables exist.
- [W5.4] (P2) Frameworks note: rotation, congruence, and pattern vs structure (factosig_promax loadings are a pattern matrix). Open an upstream factosig issue: normalize factor variances, save Phi, record the loading semantics. Decide the Aim 2.4 SHAP/LIME question (H-I-02) — done when: the note is written and the issue link is recorded.
- [W5.5] (P2) Census of programs that break into sub-axes, beyond IFN-alpha, using the cards, plus the within-span per-patient L2 direction-angle test — done when: a census table exists (H-S2-03). (depends: W5.1)
- [W5.6] (P2) Target-rotate the FA loadings to IFN sub-signatures, and add a Spectra/PLIER comparator — done when: L1 results are compared between rotated and unrotated loadings.
- [W5.7] (P1) Held-out lift inside predicted cell types for the D18 selected drivers, plus composition-matched comparisons (the open half of ALT-CT for H-S2-05). Reuse the D27 strata and `d27_celltype_means_by_patient.csv.gz` — done when: a within-cell-type lift table exists for the drivers. (depends: W5.3)
- [W5.8] (P2) Rerun D27's two checks and D29's malignant-vs-healthy contrast on the normalization the Bachireddy lab recommends (dsb if the upstream assay, implied by obs nCount_dsb, or its empty-droplet background can be shared; otherwise their preferred isotype or total-count handling), for every card, not only PC8 and FA7; new `_v<N>` slugs (R3) — done when: the two-checks table and the contrast exist on that normalization and the PC8 rows are compared with D29. It waits for the Bachireddy lab's reply to the isotype and dsb questions; added 2026-10-10 by devlog 2026-10-10_d29-cd63-depth-and-every-card. (depends: W5.3)

## Code in this directory

- `anchors/gene_direction_scratch_20261008.py`: frozen regression anchor (within-patient malignant−normal mean of z-scored log1p expression for the PC8 top genes). Not a pipeline module; never imported.
- `factor_cards.py`: D18 card library (pure numpy/pandas/scipy; no experiment paths). Computes, per factor of one stored representation:
  - orientation from sign(beta) of the globally selected LOPO probe (positive = toward malignant; `unoriented_all_zero` / `unoriented_sign_flip` otherwise);
  - per-fold selection and sign stability, with all-zero folds labelled;
  - univariate held-out lift and AUROC per supported patient (sign fixed on the training folds; the F6 sign-agnostic value is kept as an extra column);
  - D8-style fixed-model attribution (single-factor lift, ablation dAP);
  - eta2 of patient within label (computed here; no dependence on D16);
  - score distributions and cell-type means;
  - many-to-one matching within the panel (Hungarian at the same K, best |r| across K, R2 on the whole other basis, angle to its span, Tucker phi, canonical correlations);
  - promax pattern and structure genes plus Phi; FA determinacy; effective_k; eigengaps; degenerate flags;
  - within-label ADT Spearman on CLR-normalized raw counts;
  - the caveat block.
  It also renders `cards/<card_id>.md`, writes the bundle, and validates it (`validate_bundle`).
- `run_d18_factor_cards.py`: D18 runner. Its steps:
  - writes the pre-declared shortlist (the 47 D14 bases, IFN-alpha PCA40/FA40, and focus cells: IFN-alpha FA K10/K20, plus PCA40/FA40 for allograft, apoptosis and TNFa);
  - rebuilds the six controls' selected LOPO coefficients chunked from their coefficient paths, using the D1 max cell_weighted_auprc rule;
  - runs per-panel workers (joblib, at most 12);
  - merges, renders the cards and runs the regression checks (D8, F3, F6, F9, D14 and atlas reconstruction);
  - writes the IFN-alpha PC8/FA7 slide PNG/PDF, the manifest, findings and artifact ledger.
  Flags: `--shortlist full|ifna|smoke` (ifna and smoke require `--out-root DIR`, so a partial shortlist never replaces the canonical bundle), `--out-root`, `--n-jobs`, `--resume` (per representation, not per card; a part counts as done only with all parquet tables plus `_done.json`), `--merge-only` (a merge failure writes `merge_failed` in the cache dir), `--no-adt`, `--no-matching`, `--experiment-dir`.
- `run_d18_factor_cards.sh`: wrapper (interpreter from `$SC_PYTHON` or `$CONDA_PREFIX`; 2 BLAS threads per worker). The full run takes a couple of minutes with 6 workers, so it may run in an agent shell. Optional screen form: `screen -S W5_d18`, then `bash scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.sh --shortlist full --n-jobs 6 --resume`.
- `tests/test_factor_cards.py` (repo-level): synthetic tests (metrics against sklearn, orientation, lift, attribution, eta2, determinacy, matching, ADT, schema validation and corruption) plus regression tests against the real bundle (skipped when EXP or the bundle is absent).
- Outputs (EXP-relative):
  - `analysis/scorecards/stage2_diagnostic/factor_cards_v1/` (bundle plus `bundle_schema_v1.md`, `factor_cards_findings.md`, `factor_cards_artifact_ledger.csv`, `factor_cards_regression_checks.csv`);
  - `analysis/figures/stage2_diagnostic/factor_cards_v1/D18_ifna_{pca40_factor008,fa40_factor007}_card.{png,pdf}`;
  - `analysis/cache/stage2_diagnostic/factor_cards_v1/` (controls coefficient cache, `parts/`).
- Consumers of D18 read `univariate_lift` (training-fold sign), not `univariate_lift_sign_agnostic`; this matches W1's D16 convention.
- `factor_modality_assoc.py`: the D27 library (pure numpy/pandas/scipy; no experiment paths). It computes, for every card, factor x ADT and factor x predicted cell type within label:
  - Orientation: the D18 orientation_sign. Unoriented cards keep the raw sign and are flagged `oriented = False`.
  - ADT: per-cell CLR over all ADT features, the D18 definition. The isotypes are the noise reference and are never ranked as markers.
  - Spearman is vectorized: rank-standardize within the stratum, then one matrix product.
  - Strata:
    - pooled within label (the D18 number);
    - within each patient (>= 20 cells), summarised over the supported patients as the median rho and the number of patients with the same sign;
    - within supported patient x predicted cell type (>= 30 cells);
    - the D18 cell-weighted patient-stratified mean (regression only).
  - Cell type: eta2 pooled (>= 20 cells per type) and within supported patients (>= 10 cells per type; median), with the (g-1)/(n-1) null; pooled and per-patient cell-type means of the oriented cohort-z score.
  - Robust rule (fixed before computing): |median within-patient rho| >= 0.20 and the same sign in >= 5 of 7 supported patients.
  - Also: the ranked robust table, card summary, isotype reference, noise-floor table and the table schema (`SCHEMA_MD`).
- `run_d27_factor_modality_assoc.py`: D27 runner.
  - Reads the D18 bundle (card ids, orientation, degenerate flags; adt/celltype tables for the regression), the stored Stage 1 scores and the cohort cache. The raw AnnData is never loaded (R8).
  - Resume unit = one representation: `parts/<rep>/{adt,celltype_eta2,celltype_means,celltype_means_by_patient}.parquet` plus `_done.json`. Empty or corrupt files count as unfinished.
  - Merge: tables, regression checks against D18, 3 figures, findings (with a 'How to read rho here' section), manifest, ledger and schema.
  - Flags: `--shortlist full|smoke` (smoke = IFN-alpha PCA40/FA40; it needs `--out-root DIR`), `--out-root`, `--n-jobs` (<= 12; BLAS threads per worker = 12 // n_jobs via threadpoolctl), `--resume`, `--merge-only` (a merge failure writes `merge_failed` in the cache dir), `--experiment-dir`.
- `run_d27_factor_modality_assoc.sh`: wrapper (interpreter from `$SC_PYTHON` or `$CONDA_PREFIX`). The full run takes about two minutes with 4 workers x 3 threads, so it may run in an agent shell. Optional screen form: `screen -S W5_d27`, then `bash scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d27_factor_modality_assoc.sh --n-jobs 4 --resume`.
- `tests/test_factor_modality_assoc.py` (repo-level): synthetic tests plus regression tests against the real outputs, skipped when EXP or the outputs are absent:
  - Spearman vs scipy with ties; CLR and eta2 vs factor_cards;
  - strata; a planted within-patient association; a planted Simpson reversal;
  - orientation; the robust rule; cell-type tables; merge summaries;
  - the resume unit; the out-root guard.
- D27 outputs (EXP-relative):
  - `analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/`:
    - compact CSVs keyed by card_id for the explorer: `d27_card_summary.csv`, `d27_adt_top.csv`, `d27_adt_robust.csv`, `d27_celltype_eta2.csv`, `d27_celltype_means.csv`, `d27_ifna_pc8_fa7_adt.csv`, `d27_adt_noise_floor.csv`, `d27_strata.csv`, `d27_regression_checks.csv`;
    - full tables: `d27_adt_assoc_full.parquet` and `d27_celltype_means_by_patient.csv.gz`;
    - `factor_modality_assoc_v1_{findings.md,manifest.json,artifact_ledger.csv,schema.md}`.
  - `analysis/figures/stage2_diagnostic/factor_modality_assoc_v1/D27_{ifna_pc8_fa7_adt_by_patient,pooled_vs_within_patient_rho,celltype_vs_patient_eta2}.{png,pdf}`.
  - `analysis/cache/stage2_diagnostic/factor_modality_assoc_v1/parts/`.
- Consumers of D27 read `rho_within_median` with `n_patients_same_sign`, not the pooled rho, for any claim: pooled malignant rho is mostly between-patient (devlog 2026-10-09_d27-factor-modality-assoc).
- Screen sessions: `W5_d18` and `W5_d27` (both optional). Later W5 runs use `W5_<topic>`.

Task status (derived from the devlog; see `../../research/STATUS.md`): W5.1 is closed by devlog 2026-10-08_d18-factor-cards. W5.3 is advanced by devlog 2026-10-09_d27-factor-modality-assoc (D27: every non-degenerate card, pooled, within-patient and within patient × cell type ADT rho and cell-type eta2), which added W5.7 and the W7.4 explorer layer; still open: the F7b label fix; dsb is blocked (no dsb data or background in the cohort cache or the raw AnnData; a user decision). W5.5 is unblocked: the census reads `factor_cards_index.csv` (is_selected_driver, n_active_folds, orientation_status, max_univariate_lift, caveat_flags). W5.4 input: promax determinacy stays empty until factosig saves Phi, and FactoSig determinacy uses psi approximated as 1-h2.

The D18 bundle contract (`factor_cards_v1`, the interface to the W7.2 factor explorer) is under **Reference** at the end of this file.

## Conventions

- Owned artifact slugs (EXP-relative, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/`): `factor_cards_v1` (D18), `factor_modality_assoc_v1` (D27), `protein_depth_contrast_v1` (D29). New IDs for W5.2-W5.7 come from `python tools/research_index.py reserve-id --workstream W5 --title '...'`. R3: a new slug ends in `_v<N>` and is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Cards read stored coefficients at the stored C; any refit (W5.2 probes, W5.5 per-patient directions, W5.6 rotated loadings) retunes C per condition by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (D19) and reports L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal".
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

## Reference

### Bundle contract `factor_cards_v1` (schema_version 1)

The interface between D18 and `../../apps/old34_factor_explorer.py` (W7.2). The canonical text is generated into the bundle as `bundle_schema_v1.md` by `factor_cards.py`; it is copied below (headings demoted) so the contract is readable without the experiment dir. Schema changes bump schema_version, go through a devlog entry with type: refactor, and must update W7.2 in the same change.

Producer: `scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py` (D18, W5.1).
Consumer: the W7.2 factor explorer. Location: `analysis/scorecards/stage2_diagnostic/factor_cards_v1/`
(EXP-relative). The app reads only the CSV and JSON files; `cards/*.md` are for people and slides.
Required columns are fixed for schema_version 1; files may carry extra columns, and consumers
must ignore columns they do not know. All paths inside the bundle are relative to the bundle dir
or to the experiment dir, never absolute.

#### Identifiers

- `card_id` = `<panel_id>__<method>__k<K>__f<NNN>` (factor 1-based, zero-padded to 3 digits;
  method in pca | fa | factosig | factosig_promax). Regex:
  `^(.+)__(pca|fa|factosig|factosig_promax)__k(\d+)__f(\d{3})$`.
- `representation_id` = `<panel_id>|dr|<method>|<K>|42` (the atlas id).
- `matched_cards` and `matched_card_id` may name cards of representations outside the shortlist;
  such ids have no row in the index and must be shown as plain text, not links.

#### factor_cards_manifest.json

Keys: `schema_version` (1), `slug`, `inputs` ({name: {path (EXP-relative), size_bytes, mtime_utc}}),
`git_state` ({head, dirty}), `created_by` (repo-relative script path), `created_utc`, `parameters`,
`counts` (rows per file, n_cards, n_representations), `runtime_seconds`. Extra keys in v1:
`analysis_id`, `task`, `compute_seconds`, `merge_only`, `part_worker_seconds_sum`,
`regression_checks_passed`.

#### factor_cards_index.csv (one row per card)

Required:

| column | type | meaning |
|---|---|---|
| card_id | str | see above |
| panel_id | str | Stage 0 panel id |
| method | str | Stage 1 method |
| k | int | requested K |
| factor | int | 1-based factor index within this representation only |
| is_selected_driver | bool | beta != 0 in >=1 supported LOPO fold AND the fit is not dense (median supported-fold nonzero fraction <= 0.5) |
| n_active_folds | int | supported folds (of n_folds, normally 7) with beta != 0 |
| sign_consistency | float | share of active supported folds with the majority sign; empty when never active |
| orientation_sign | int | +1 / -1 = sign(beta), stable over active supported folds (positive = toward malignant); 0 = unoriented |
| eta2_normal, eta2_malignant | float | eta2 of patient within label (normal patients >=200 cells, malignant >=20) |
| degenerate_flag | bool | zero score variance / zero loadings (a blown-up or collapsed score scale is the separate caveat `scale_outlier`) |
| effective_k | int | non-degenerate factors in this representation |
| determinacy | float | factor-score determinacy rho (fa: saved psi; factosig: psi approx 1-h2); empty for pca (exact scores) and promax (Phi unsaved) |
| matched_cards | str | ';'-separated card ids with abs(score r) >= 0.5: Hungarian match at the same K for other methods, best-abs(r) match at other K, sorted by abs(r) |
| caveat_flags | str | ';'-separated codes (vocabulary below) |

Extra columns (stable in v1, may be extended): representation_id, panel_label, n_genes,
shortlist_reasons, orientation_status (oriented | unoriented_all_zero | unoriented_sign_flip),
n_folds, n_active_folds_all, n_folds_all, fit_penalty, fit_C, fit_l1_ratio, fit_dense,
median_fold_nonzero_fraction, n_all_zero_folds, all_zero_folds, median_abs_coef,
max_fitted_ap_drop, median_univariate_lift, max_univariate_lift,
n_patients_univariate_lift_ge_0p05, label_direction_sign (pooled malignant-minus-normal sign of
the raw score; descriptive, not the probe), eta2_celltype_normal, determinacy_method, score_sd,
eigengap_rel, eigengap_basis, near_duplicate_max_abs_r, near_duplicate_partner,
adt_isotype_max_abs_rho_normal, adt_isotype_max_abs_rho_malignant,
reconstruction_max_abs_error, loading_semantics, card_md (relative path).

#### factor_cards_genes.csv (top genes per pole)

Required: card_id, gene, loading (raw), oriented_loading (orientation_sign x loading; empty when
unoriented), rank (1-based within card x ranked_by x pole), pole (`malignant` | `normal` when
oriented; `positive` | `negative` by raw sign when unoriented), structure_corr (corr(gene, score),
oriented when possible; promax only, else empty).
Extra: ranked_by (`loading` | `structure`; promax cards carry both lists), loading_semantics.
Degenerate factors have no gene rows.

#### factor_cards_patient_lift.csv (card x supported patient)

Required: card_id, patient, univariate_lift (held-out AP of the factor alone, sign fixed on the
training folds, minus prevalence), univariate_auroc (same sign), n_malignant, prevalence.
Extra: univariate_auprc, univariate_lift_sign_agnostic (F6 definition: max over both signs on the
held-out patient; optimistic), train_orientation_sign, heldout_malignant_mean_z,
heldout_normal_mean_z, n_cells, fitted_coef, fitted_full_model_ap, fitted_full_model_lift,
fitted_ablated_ap, fitted_ap_drop (D8 ablation dAP), fitted_single_factor_lift (D8).

#### factor_cards_fold_coefs.csv (card x LOPO fold, all 13 folds)

Required: card_id, heldout_patient, penalty, C, coef.
Extra: l1_ratio, is_nonzero, supported_fold, fold_all_zero (whole coefficient vector zero),
fold_n_active, fold_nonzero_fraction, fold_model_ap, fold_model_lift (reconstructed; supported folds).

#### Optional files (present in v1)

- factor_cards_matches.csv: card_id, other_representation_id, other_method, other_k, relation
  (`hungarian_same_k` | `best_abs_r`), matched_factor, matched_card_id, score_r, abs_r,
  r2_many_to_one, angle_deg, tucker_phi_aligned, best_factor, best_r, hungarian_factor, hungarian_r.
- factor_cards_subspace.csv: representation_id, other_representation_id, k, other_k, other_method,
  rank, other_rank, n_canonical, cc_min, cc_median, cc_max, n_cc_below_0p9, selected_rank,
  selected_cc_min, selected_cc_median (principal angles as canonical correlations).
- factor_cards_score_dist.csv: card_id, factor, patient, label, n_cells, oriented, mean, sd, q10,
  q25, q50, q75, q90 (cohort-z score, multiplied by orientation_sign when oriented).
- factor_cards_adt.csv: card_id, label, rank, adt_marker, is_isotype, spearman_pooled,
  spearman_patient_stratified, n_cells, oriented (top markers by |pooled rho| excluding isotypes;
  ADT = CLR per cell of raw counts).
- factor_cards_celltype.csv: card_id, factor, label, cell_type, n_cells, mean_oriented_z.
- factor_cards_phi.csv: representation_id, factor_i, factor_j, score_r (oblique fits only).
- factor_cards_shortlist.csv: the pre-declared shortlist with reasons.
- factor_cards_regression_checks.csv: check_id, description, source, n_compared, max_abs_diff,
  tolerance, passed.

#### Caveat flag vocabulary

- `transductive_basis`: Stage 1 was fit on all 13 patients, including each held-out patient; LOPO lift is label transfer on a transductive basis, not inductive deployment.
- `unoriented_all_zero`: beta is 0 in every supported fold, so the probe gives no orientation; loadings are shown with their arbitrary raw sign.
- `unoriented_sign_flip`: beta changes sign across supported folds, so the factor has no stable orientation; loadings are shown with their arbitrary raw sign.
- `has_all_zero_folds`: At least one supported fold fit an all-zero model (no factor selected); its lift is 0 by construction, not evidence of absent signal.
- `dense_fit`: Dense fit (median fold nonzero fraction above 0.5): per-factor betas are partial effects among many correlated contributions and are not interpretable one factor at a time.
- `l1_selected_absent`: The factor is never selected by the fitted probe; check its univariate held-out lift before reading the zero as absence of signal (L1 at a fixed C keeps one of several near-equivalent axes).
- `zero_variance`: Degenerate factor: zero score variance (and loadings), beyond the representation's effective K.
- `scale_outlier`: Blown-up or collapsed factor scale (score SD more than 10x away from the representation median). Not degeneracy: Stage 2 standardizes scores, and oblique (promax) column scale is arbitrary.
- `near_duplicate_oblique`: Oblique fit: this factor's scores correlate |r| >= 0.9 with another factor; beta is a partial effect within a near-duplicate pair.
- `low_determinacy`: Factor-score determinacy below 0.707: equally valid score sets can be uncorrelated, so do not name this factor as a unique program.
- `small_eigengap`: Relative eigengap to a neighbouring factor is 10% or less; the individual axis is not identifiable (rotation within the near-degenerate block).
- `generative_loadings`: FA / FactoSig loadings are generative (how expected expression moves with the factor), not the weights that compute the score.
- `oblique_pattern_loadings`: factosig_promax loadings are a pattern matrix with arbitrary column scale; read them together with the structure correlations and Phi.
- `ablation_is_attribution`: Ablation dAP and single-factor lift are attributions inside the fitted model, not causal effects.
- `small_patient_support`: At least one supported patient has fewer than 25 malignant cells; its lift is noisy.
- `reconstruction_mismatch`: Reconstructed held-out AP differs from the stored atlas AP by more than 1e-4 in at least one supported fold. When the only active coefficients are tiny (|beta| < 1e-4) the stored float32 probabilities tie and shift the stored AP; otherwise the fold coefficients may not match the atlas selection.
