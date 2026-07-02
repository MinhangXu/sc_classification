# MDS Stage 0/1/2 Sharedness Scorecard Plan

Date: 2026-05-25

This note refines the new MRD Stage 0/1/2 experiment plan after the discussion about whether heldout evaluation is necessary. The short answer is: keep heldout evaluation, but do not let it be the only truth. The supervised model should be treated as both a biological probe of a representation and, separately, a sharedness/generalization assay.

## Core Framing

The experiment should preserve three parallel views of the same Stage 0 panel and Stage 1 representation:

1. **Discovery full-cohort fit**
  - Fit the supervised model on all eligible MRD cells and evaluate on the same full cohort.
  - This is an in-sample or apparent fit metric, not a generalization metric.
  - Purpose: ask whether a Stage 0 gene panel and Stage 1 representation contain malignant-vs-non-malignant structure in the observed cohort.
2. **Sharedness patient-heldout evaluation**
  - Hold out patients one at a time, or use patient-aware GroupKFold if leave-one-patient-out is infeasible.
  - Purpose: ask whether the malignant signal learned from other patients transfers to a heldout patient.
  - In a small cohort, leave-one-patient-out should be the default for patients with enough malignant and non-malignant MRD cells to compute meaningful metrics.
3. **Patient-specific modeling**
  - Fit or cross-validate models within each patient on the same across-patient Stage 1 latent basis.
  - Purpose: ask whether each patient uses the same latent axes as the shared model or has patient-specific malignant biology.

These three views answer different questions. A panel can be biologically interesting even if it has weak cross-patient heldout performance, but that result should be labeled as patient-specific, heterogeneous, or cohort-specific rather than broadly shared.

## Stage Definitions

### Stage 0: Prior Gene-Space Subsetting

Stage 0 is the biological hypothesis layer. Each panel asks:

> If the model is only allowed to look at this prior-defined gene space, how much malignant-vs-non-malignant structure can it recover?

Required panel families for the first broad screen:

- old 34-program single genesets
- old biology-group panels
- full old-34 control
- core-only control
- fixed HVG anchors such as 500, 1000, 3000, and 10000
- later: exact or budget-bin size-matched HVG controls for shortlisted panels

Stage 0 comparisons should usually be made within panel family or gene-budget strata. A 10k HVG anchor and a 100-gene curated panel are not the same kind of hypothesis.

### Stage 1: Representation Learning

Stage 1 turns a Stage 0 panel matrix into features:

- `representation_family=dr`: PCA, FA, FactoSig, FactoSig promax, later cNMF where appropriate
- `representation_family=direct_gene`: standardized genes used directly
- `representation_family=summary_score`: optional module scores for tiny panels

The primary first-pass Stage 1 scope should be across-patient DR:

> Fit one representation on the combined eligible MRD cells so all patients share the same latent coordinate system.

This is important because patient-specific classifiers can then be compared in the same factor space.

### Stage 2: Supervised Probes

Stage 2 evaluates how well the Stage 1 features align with malignant status.

It should be split into explicit modeling goals:

- `modeling_goal=discovery_full_cohort_fit`
- `modeling_goal=sharedness_leave_patient_out`
- `modeling_goal=patient_specific`

Do not collapse these into one score. They have different scientific meanings.

## Transductive Shared Representation Analysis

If Stage 1 DR is fit once using all eligible MRD cells, including cells from patients later held out in Stage 2, then the analysis is **transductive**:

- The unsupervised representation has seen the full cohort expression structure.
- The supervised classifier has not necessarily seen the heldout patient's malignant labels during training.
- The heldout evaluation tests label transfer within a shared atlas-like latent space.

This is defensible for the biology goal because the project wants a common MRD latent coordinate system for interpretation.

However, it is not the same as a strict prospective classifier benchmark. A fully inductive benchmark would, for every heldout patient, fit preprocessing, scaling, HVG selection, and DR only on training patients, then project the heldout patient. That is cleaner for deployment-style prediction but harder to compare factors across folds because each fold has a different latent basis.

Recommended reporting:

- Use transductive across-patient Stage 1 as the main biological shared-coordinate analysis.
- Clearly label it as transductive.
- Optionally add an inductive sensitivity analysis later for shortlisted panels.

## Scorecard Metrics

The scorecard should have one row per evaluated representation/model setting, with linked long-form tables for per-patient and per-fold details.

### Identity And Provenance Columns

- `experiment_id`
- `input_h5ad`
- `preprocessing_manifest_path`
- `stage0_panel_id`
- `stage0_panel_type`
- `stage0_panel_family`
- `source_dictionary`
- `n_raw_genes`
- `n_covered_genes`
- `gene_budget_type`
- `matched_control_id`
- `stage0_gene_list_path`

### Stage 1 Columns

- `representation_family`
- `stage1_scope`: `across_patient`, `per_patient`, or other explicit value
- `stage1_method`
- `requested_k`
- `effective_k`
- `stage1_seed`
- `stage1_scores_path`
- `stage1_loadings_path`
- `stage1_metadata_path`
- `stage1_fit_scope_note`: for example `transductive_all_eligible_cells`

### Stage 2 Columns

- `modeling_goal`: `discovery_full_cohort_fit`, `sharedness_leave_patient_out`, or `patient_specific`
- `stage2_mode`
- `classifier`
- `penalty`: `l1`, `l2`, `elasticnet`
- `C`
- `l1_ratio`
- `class_weight`
- `split_policy`
- `split_seed`
- `decision_threshold`
- `positive_class`
- `stage2_status`
- `stage2_predictions_path`
- `stage2_coefficients_path`
- `stage2_metrics_long_path`

### Discovery Full-Cohort Fit Metrics

These are in-sample/apparent fit metrics:

- `full_cohort_fit_auc`
- `full_cohort_fit_auprc`
- `full_cohort_fit_balanced_accuracy`
- `full_cohort_fit_malignant_precision`
- `full_cohort_fit_malignant_recall`
- `full_cohort_fit_healthy_recall_specificity`
- `full_cohort_fit_tp`
- `full_cohort_fit_fn`
- `full_cohort_fit_tn`
- `full_cohort_fit_fp`

Interpretation:

- High values mean the panel/representation separates malignant and non-malignant cells in the observed cohort.
- These metrics should not be described as generalization performance.
- They are useful for discovery, coefficient paths, factor interpretation, and patient-level diagnostics.

### Patient-Heldout Sharedness Metrics

Primary aggregate fields:

- `leave_patient_out_auc_mean`
- `leave_patient_out_auc_min`
- `leave_patient_out_auprc_mean`
- `leave_patient_out_auprc_min`
- `leave_patient_out_balanced_accuracy_mean`
- `leave_patient_out_n_evaluable_patients`
- `leave_patient_out_by_patient_path`

The linked by-patient table should include:

- `heldout_patient_id`
- `n_train_patients`
- `n_test_malignant`
- `n_test_non_malignant`
- `heldout_auc`
- `heldout_auprc`
- `heldout_balanced_accuracy`
- `heldout_malignant_precision`
- `heldout_malignant_recall`
- `heldout_healthy_recall_specificity`
- `heldout_tp`
- `heldout_fn`
- `heldout_tn`
- `heldout_fp`
- `skip_reason`

Interpretation:

- High full-cohort fit and high patient-heldout performance support shared malignant biology.
- High full-cohort fit but weak or patient-dependent heldout performance suggests patient-specific biology, patient subgroup structure, or possible confounding.
- The by-patient vector is often more biologically useful than the mean.

### Patient-Specific Metrics

For each patient:

- `patient_id`
- `n_malignant`
- `n_non_malignant`
- `patient_specific_fit_auc`
- `patient_specific_fit_auprc`
- `patient_specific_cv_auc` if feasible
- `patient_specific_cv_auprc` if feasible
- `selected_factor_ids`
- `selected_factor_count`
- `overlap_with_shared_model_factors`
- `patient_specific_coefficients_path`

Interpretation:

- If patient-specific models select the same factors as the shared model, that supports shared biology.
- If different patients select different factors on the same across-patient latent basis, that is candidate patient-specific biology.

### Regularization-Specific Interpretation

- **L1**: sparse factor selection. Useful for parsimonious biological interpretation, but can arbitrarily choose one factor from a correlated group.
- **L2**: distributed signal. Useful for asking whether the representation carries information even when no small factor subset dominates.
- **Elastic net**: compromise between sparse and grouped selection. Often preferable when latent factors or genes are correlated.

The full regularization path matters. A single fixed `C` is useful for a quick screen, but the final Stage 2 should report coefficient paths, selected factor stability, and performance across `C` values.

## Precision-Recall Interpretation

In a precision-recall plot for the malignant class, the horizontal malignant-prevalence line is the baseline precision expected from random ranking. If malignant prevalence is 0.026, then a random model would have expected precision near 0.026.

Therefore:

- AUPRC should be interpreted relative to malignant prevalence.
- Precision of 0.109 at a selected threshold is roughly 4.2x the 0.026 prevalence baseline.
- AUPRC of 0.549 is a strong ranking/enrichment signal in a rare-positive setting, even if the fixed threshold still produces many false positives.

The fixed threshold, often 0.5, is only one operating point on the curve. It should not be treated as the biological truth unless the probabilities are calibrated and a decision policy has been chosen.

For broad screening, report threshold-free metrics and several thresholded operating points:

- AUROC
- AUPRC
- precision/recall/specificity at threshold 0.5
- precision at fixed malignant recall, for example recall 0.5, 0.7, 0.8
- malignant recall at fixed false-positive budget, for example top 1%, 5%, or 10% highest-scored cells
- optional threshold chosen by maximum balanced accuracy or Youden's J, reported separately

## Interpreting The Example Rows

The example 10k HVG / FactoSig / K=40 row appears to have:

```text
TP = 1075
FN = 494
TN = 49784
FP = 8765
malignant_precision = 0.109
malignant_recall = 0.685
healthy_recall_specificity = 0.850
```

The example cytokine signaling / PCA / K=40 row appears to have:

```text
TP = 1201
FN = 368
TN = 48929
FP = 9620
malignant_precision = 0.111
malignant_recall = 0.765
healthy_recall_specificity = 0.836
```

Important terminology:

- False positive: healthy/non-malignant cell predicted malignant.
- False negative: malignant cell predicted healthy/non-malignant.

So the 10k HVG row has fewer false positives than the cytokine row, but more false negatives. The cytokine row recovers more malignant cells at threshold 0.5, while also calling more healthy cells malignant.

This does not invalidate the Stage 0/1/2 design. It reinforces why the scorecard should separate:

- threshold-free ranking quality, such as AUROC and AUPRC
- threshold-specific confusion counts
- malignant recall
- malignant precision
- healthy specificity
- per-patient behavior

Many healthy cells called malignant may reflect true transcriptomic overlap, class imbalance, label noise, cell-type composition, or patient-specific programs. It is not automatically a failure, but it means thresholded calls need careful interpretation.

## Evaluation Of `stage0_mrd_old34_metric_diagnostics_20260525.ipynb`

Notebook path:

`sc_classification/scripts/mrd_stage0_2/notebooks/stage0_2/stage0_mrd_old34_metric_diagnostics_20260525.ipynb`

Local review status:

- The notebook file is present in this checkout.
- The experiment scorecard path hard-coded inside the notebook was not present on this machine at review time:
`/home/minhang/mds_project/sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093/analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv`
- Therefore this assessment is based on the notebook source and the pasted example rows, not a local rerun of the notebook outputs.

### Still Useful

The notebook is useful as a post-run diagnostic layer for the broad quick screen because it:

- reads the current broad-screen scorecard rather than the old May 18 prototype scorecard
- explains AUROC, AUPRC, and balanced accuracy
- recomputes confusion counts from saved prediction files
- adds malignant precision, malignant recall, and healthy specificity
- plots a representative precision-recall curve with the malignant-prevalence baseline
- compares metric distributions by `stage0_panel_type`
- checks the relationship between gene budget and metrics
- ranks single genesets and biology groups separately from controls
- diagnoses K sensitivity
- diagnoses DR method sensitivity at fixed panel and K

This makes it valuable for human review after the broad Stage 0/1/quick-Stage-2 screen.

### Not Sufficient For The Updated Stage 2 Plan

The notebook should not be treated as the final Stage 2 analysis because it appears to be built around a quick screen:

- fixed L2 logistic regression with `C=1.0`
- one thresholded confusion matrix at threshold 0.5
- no L1 regularization path
- no elastic-net path
- no coefficient stability analysis
- no explicit full-cohort discovery fit vs patient-heldout sharedness vs patient-specific modeling separation
- no clear patient-heldout by-patient vector in the displayed source
- no explicit transductive vs inductive Stage 1 provenance
- hard-coded experiment path
- composite `malignancy_first_score` is useful for triage but should not become a primary scientific endpoint

Recommended role:

- Keep this notebook as a broad-screen diagnostic and human-review aid.
- Update or replace it after the expanded Stage 2 runner writes the new multi-objective scorecard.
- Make it parameterized by experiment directory or manifest rather than hard-coding one experiment path.

## Context-Probing Answers From Current Run Artifacts

These answers refer to experiment `20260525_060508_stage0_mrd_old34_broad_screen_82db5093`.

1. The quick Stage 2 split policy is `GroupKFold_by_patient` with `cv_folds=5`, not leave-one-patient-out, random cell split, or train-on-same-cells evaluation. Representative fold assignment in the metrics artifacts is:
  - fold 1: held out `P09`
  - fold 2: held out `P01`
  - fold 3: held out `P02;P12;P13`
  - fold 4: held out `P03;P05;P08;P10`
  - fold 5: held out `P04;P06;P07;P11`
2. The prediction files are out-of-fold predictions from the quick Stage 2 classifier. Each fold fits the classifier on training patients and predicts the held-out patient group. They are not in-sample predictions from one classifier fit on all cells. However, these are not fully inductive from raw expression because Stage 1 panel standardization and DR representations were computed once before Stage 2 splitting.
3. `included_in_metric` marks cells with finite out-of-fold `y_prob` values that were included in aggregate metric calculation. It would be `False` for cells in skipped folds with missing predictions. In this completed run, every Stage 2 OK scorecard row reports `n_eval_cells=60118`, `n_splits=5`, and `n_valid_folds=5`; a representative prediction file has all `60118` rows marked `True`.
4. Patient identity is defined by the `patient` column. This is the configured `patient_col` and the grouping vector used by `GroupKFold`.
5. The artifacts do not define a formal minimum-count threshold for leave-one-patient-out support. If "support" means at least one malignant and one normal MRD cell in the held-out patient, the eligible patients are:
  - `P01`: 45 malignant, 11815 normal
  - `P02`: 211 malignant, 1251 normal
  - `P03`: 758 malignant, 1569 normal
  - `P04`: 102 malignant, 1026 normal
  - `P05`: 8 malignant, 4351 normal
  - `P06`: 44 malignant, 3215 normal
  - `P07`: 22 malignant, 2163 normal
  - `P09`: 375 malignant, 16751 normal
  - `P13`: 4 malignant, 5770 normal
   Patients `P08`, `P10`, `P11`, and `P12` are normal-only in this MRD/CITE filtered cohort, so they cannot support patient-level malignant-vs-normal metrics alone. Even among patients with both labels, `P05`, `P07`, and especially `P13` have very few malignant cells, so LOPO metrics for those patients would be unstable.
6. Stage 1 DR was fit once on all eligible cells for each panel/method/K/seed, then the resulting scores were passed into quick Stage 2. Stage 1 was not refit within each Stage 2 split.
7. No. The scorecard currently labels this as `stage1_scope=across_patient`, `stage2_mode=shared_cross_patient`, and `split_policy=GroupKFold_by_patient`; it does not explicitly label the Stage 1 provenance as `transductive_all_eligible_cells`. That label should be added in the next schema pass.
8. There are two scaling layers:
  - Before Stage 1, each panel matrix is z-scored once across all eligible cells with `feature_zscore_with_zero_variance_to_zero`; this is global/transductive relative to Stage 2.
  - Inside Stage 2, the logistic-regression pipeline includes `SimpleImputer(strategy="constant", fill_value=0.0)` and `StandardScaler()`, which are fit only on the training cells for each GroupKFold split.
9. Yes. HVG anchors were computed from the shared MRD/CITE cohort produced by this run, after cell filtering, the 1% min-cell gene filter, normalize-total/log1p, and post-filter variance ranking. The run kept `60118` cells and `15560` genes after filtering; HVG panels are labeled with `source_dictionary=shared_mrd_hvg_variance_rank`, not inherited from an old gene-restricted AnnData.
10. The quick classifier is sklearn logistic regression in a pipeline:
  - classifier: `LogisticRegression`
    - penalty: `l2`
    - solver: `liblinear`
    - `C`: `1.0`
    - `class_weight`: `balanced`
    - `max_iter`: `5000`
    - random seed: `42`
    - default decision threshold for balanced accuracy/F1/confusion diagnostics: `0.5`
11. Yes. `y_true=1` corresponds to the configured positive class `CN.label == cancer`, and `y_true=0` corresponds to the configured negative class `CN.label == normal`.
12. Probability scores are raw `predict_proba` outputs from logistic regression. There is no calibration wrapper such as Platt scaling or isotonic calibration in this quick Stage 2 runner.
13. Yes. Direct-gene fallback rows are included when a panel has invalid DR K values under `small_panel_policy=direct_gene`. This run has three direct-gene fallback rows:
  - `single_geneset__reactome_nf_kb_activation`
    - `single_geneset__reactome_tnfr1_induced_nfkb_signaling_pathway`
    - `single_geneset__reactome_antigen_presentation_folding_assembly_and_peptide_loading_of_class_i_mhc`
14. No classifier coefficients or selected-factor summaries are saved for each model row. Stage 1 DR rows save scores/loadings/metadata/diagnostics, and direct-gene rows save feature matrices and feature names, but the fitted Stage 2 logistic models and coefficients are not persisted.
15. Current scorecard metrics are cell-wise aggregate metrics over all `included_in_metric` cells. There is no patient-wise aggregation where each patient contributes equally.
16. The diagnostic confusion-count column order is confirmed at threshold `0.5`:
  - `heldout_malignant_correct_tp`: `y_true == 1` and `y_prob >= 0.5`
    - `heldout_malignant_incorrect_fn`: `y_true == 1` and `y_prob < 0.5`
    - `heldout_healthy_correct_tn`: `y_true == 0` and `y_prob < 0.5`
    - `heldout_healthy_incorrect_fp`: `y_true == 0` and `y_prob >= 0.5`

## Recommended Next Implementation Step

The broad-screen notebook can stay in place, but the runner should next produce a canonical multi-objective scorecard with these three modeling goals:

```text
discovery_full_cohort_fit
sharedness_leave_patient_out
patient_specific
```

The existing quick Stage 2 can be canonicalized as provenance, but it should not be the final sharedness layer. It should be relabeled as:

```text
modeling_goal = quick_sharedness_groupkfold_by_patient
stage1_fit_scope_note = transductive_all_eligible_cells
split_policy = GroupKFold_by_patient
classifier = LogisticRegression
penalty = l2
C = 1.0
```

This makes the existing result useful as a triage screen and audit trail while the next implementation moves to leave-one-patient-out.

## Updated Implementation Direction

Implementation adjustment from artifact review:

- The first implemented runner is Stage-2-only and consumes the existing Stage 0 panels plus Stage 1 feature artifacts. It does not rerun Stage 0/1.
- By default, the expensive discovery and patient-specific regularization paths run on the best quick-L2 representation for each shortlisted panel, rather than every existing `(panel, method, K)` quick row. This keeps the first pass feasible while still covering controls and biologically forced-in panels.
- Exhaustive expansion over all current quick rows remains available via `--panel-selection all_quick_rows`.

### 1. Do Not Rerun Stage 0/1 Unless A Defect Is Found

Use the current Stage 0 panels and Stage 1 artifacts as the input for the next Stage 2 pass. The current Stage 1 artifacts are valid for the biological shared-coordinate analysis, but every downstream scorecard row should explicitly record:

```text
stage1_scope = across_patient
stage1_fit_scope_note = transductive_all_eligible_cells
```

This avoids pretending that the LOPO results are fully inductive from raw expression.

### 2. Canonicalize The Existing Quick L2 Result

Create a canonical copy or transformed view of the existing quick GroupKFold scorecard. This should not require refitting models. The goal is to preserve what was already run with clearer labels:

- `modeling_goal=quick_sharedness_groupkfold_by_patient`
- `split_policy=GroupKFold_by_patient`
- `penalty=l2`
- `C=1.0`
- `stage1_fit_scope_note=transductive_all_eligible_cells`
- `metrics_aggregation=cell_weighted_oof`

Do not spend time probing L2 regularization strength for this provenance block.

### 3. Build A Provisional Shortlist From The Quick L2 Screen

Because the full L1/L2/elastic-net path across every panel, method, K, and patient may be large, create a provisional shortlist before the expensive interpretability pass.

The shortlist can be based on the current quick L2 result, but it should be described as **triage**, not final evidence.

Recommended shortlist rules:

- Rank panels within each `stage0_panel_type`, not globally.
- For each Stage 0 panel, keep the best quick L2 row by a composite rank using AUPRC, AUROC, balanced accuracy, malignant precision, and malignant recall.
- Select the top single genesets within type, for example top 20-30% or top 10-15 panels.
- Select the top biology groups within type, for example top 3-5 groups.
- Always include controls:
  - `full_34`
  - `core_only`
  - HVG anchors 500, 1000, 3000, 10000
- Force-include biologically important panels even if they are not top-ranked:
  - IFN/interferon-related panels
  - NF-kB/TNF-related panels
  - antigen-presentation/MHC-related panels
  - cytokine/immune signaling panels already highlighted by the quick screen
- Include direct-gene fallback panels that are biologically important and too small for valid DR at larger K.

The shortlist table should be written as:

```text
analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv
```

Required columns:

- `stage0_panel_id`
- `stage0_panel_type`
- `n_covered_genes`
- `best_quick_stage1_method`
- `best_quick_requested_k`
- `best_quick_representation_family`
- `best_quick_auroc`
- `best_quick_auprc`
- `best_quick_balanced_accuracy`
- `best_quick_malignant_precision`
- `best_quick_malignant_recall`
- `shortlist_reason`
- `force_include_reason`

If GPU time is abundant, it is acceptable to run discovery full-cohort paths for all broad-screen rows. Patient-specific regularization paths should still be prioritized for the shortlist because they multiply by patient.

### 3a. Current Quick-Stage-2 Shortlist Snapshot And Run Count

The current generated shortlist is:

```text
analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv
```

It contains 47 best-per-panel quick-L2 candidates and 33 selected representations for the first multi-objective pass:

- 6 controls: `full_34`, `core_only`, and HVG anchors 500, 1000, 3000, 10000
- 21 single-geneset panels
- 6 single-biology-group panels

Selected rows from the current quick Stage 2 shortlist:

| panel | type | best quick representation | reason |
| --- | --- | --- | --- |
| `core_only` | `core_only` | `pca`, K=40 | control |
| `full_34` | `full_control` | `fa`, K=10 | control |
| `hvg_top_requested_10000__available_10000` | `hvg_anchor_control` | `factosig`, K=10 | control |
| `hvg_top_requested_3000__available_3000` | `hvg_anchor_control` | `fa`, K=40 | control |
| `hvg_top_requested_1000__available_1000` | `hvg_anchor_control` | `factosig_promax`, K=40 | control |
| `hvg_top_requested_500__available_500` | `hvg_anchor_control` | `fa`, K=40 | control |
| `single_geneset__reactome_cytokine_signaling_in_immune_system` | `single_geneset_only` | `fa`, K=20 | top single geneset; force include cytokine signaling |
| `single_geneset__reactome_mhc_class_i_antigen_presentation` | `single_geneset_only` | `factosig_promax`, K=10 | top single geneset; force include antigen/MHC |
| `single_geneset__hallmark_allograft_rejection` | `single_geneset_only` | `factosig`, K=10 | top single geneset |
| `single_geneset__hallmark_il2_stat5_signaling` | `single_geneset_only` | `fa`, K=10 | top single geneset; force include cytokine signaling |
| `single_geneset__hallmark_tnfa_signaling_via_nfkb` | `single_geneset_only` | `factosig_promax`, K=20 | top single geneset; force include NF-kB/TNF |
| `single_geneset__hallmark_inflammatory_response` | `single_geneset_only` | `fa`, K=20 | top single geneset |
| `single_geneset__hallmark_mtorc1_signaling` | `single_geneset_only` | `fa`, K=20 | top single geneset |
| `single_geneset__kegg_jak_stat_signaling_pathway` | `single_geneset_only` | `factosig_promax`, K=10 | top single geneset; force include cytokine signaling |
| `single_geneset__hallmark_g2m_checkpoint` | `single_geneset_only` | `factosig`, K=20 | top single geneset |
| `single_geneset__hallmark_hypoxia` | `single_geneset_only` | `fa`, K=40 | top single geneset |
| `single_geneset__reactome_interferon_signaling` | `single_geneset_only` | `pca`, K=20 | top single geneset; force include interferon |
| `single_geneset__hallmark_oxidative_phosphorylation` | `single_geneset_only` | `factosig`, K=10 | top single geneset |
| `single_geneset__hallmark_interferon_gamma_response` | `single_geneset_only` | `factosig`, K=20 | force include interferon |
| `single_geneset__reactome_mhc_class_ii_antigen_presentation` | `single_geneset_only` | `pca`, K=10 | force include antigen/MHC |
| `single_geneset__hallmark_il6_jak_stat3_signaling` | `single_geneset_only` | `pca`, K=40 | force include cytokine signaling |
| `single_geneset__hallmark_interferon_alpha_response` | `single_geneset_only` | `factosig_promax`, K=20 | force include interferon |
| `single_geneset__reactome_interferon_gamma_signaling` | `single_geneset_only` | `pca`, K=40 | force include interferon |
| `single_geneset__reactome_antigen_presentation_folding_assembly_and_peptide_loading_of_class_i_mhc` | `single_geneset_only` | `fa`, K=20 | force include antigen/MHC |
| `single_geneset__kegg_antigen_processing_and_presentation` | `single_geneset_only` | `pca`, K=5 | force include antigen/MHC |
| `single_geneset__reactome_nf_kb_activation` | `single_geneset_only` | `fa`, K=10 | force include NF-kB/TNF |
| `single_geneset__reactome_tnfr1_induced_nfkb_signaling_pathway` | `single_geneset_only` | `factosig_promax`, K=20 | force include NF-kB/TNF |
| `single_group__proliferation_metabolism` | `single_group_only` | `factosig_promax`, K=40 | top biology group |
| `single_group__cytokine_jak_stat` | `single_group_only` | `pca`, K=20 | top biology group; force include cytokine signaling |
| `single_group__innate_immune_context` | `single_group_only` | `fa`, K=20 | top biology group |
| `single_group__stress_arrest` | `single_group_only` | `factosig_promax`, K=40 | top biology group |
| `single_group__inflammatory_interferon` | `single_group_only` | `pca`, K=20 | top biology group; force include interferon |
| `single_group__antigen_presentation` | `single_group_only` | `factosig_promax`, K=20 | force include antigen/MHC |

Run-count estimate for the first command I recommend running in screen, using the coarser first pass:

```text
C_grid = 1 / logspace(-3, 3, 13)
       = 1000, 316.23, 100, 31.62, 10, 3.16, 1,
         0.316, 0.1, 0.0316, 0.01, 0.00316, 0.001

l1_ratio_grid = 0.1, 0.5, 0.9
regularization settings per representation =
  13 L1 C values
+ 13 L2 C values
+ 13 * 3 elastic-net C/l1_ratio values
= 65 settings
```

With 33 selected representations, 13 LOPO heldout patients, and 9 patients that have both malignant and normal MRD cells for patient-specific apparent fits:

```text
canonical quick scorecard                  0 fits
shortlist generation                       0 fits
discovery_full_cohort_fit       33 * 65 =  2,145 fits
sharedness_leave_patient_out    33 * 13 =    429 fits
patient_specific             33 * 9 * 65 = 19,305 fits
----------------------------------------------------
total logistic-regression fits                    21,879
```

The row count is not exactly the same as the fit count. `decision_thresholds`, `fixed_recall_targets`, and `top_fraction_thresholds` add metric columns or rows but do not refit models. With the current single threshold `0.5`, expected primary scorecard row counts are approximately:

```text
discovery_full_cohort_scorecard rows   2,145
sharedness_lopo aggregate rows            33
sharedness_lopo by-patient rows          429
patient_specific_scorecard rows       19,305
```

If we instead use the original 17-point grid (`C_grid = 1 / logspace(-4, 4, 17)`), the regularization settings become:

```text
17 L1 + 17 L2 + 17 * 3 elastic-net = 85 settings per representation
```

The same shortlist would then require:

```text
discovery_full_cohort_fit       33 * 85 =  2,805 fits
sharedness_leave_patient_out    33 * 13 =    429 fits
patient_specific             33 * 9 * 85 = 25,245 fits
----------------------------------------------------
total logistic-regression fits                    28,479
```

### 4. Discovery Full-Cohort Fit Should Use Full Regularization Paths

For `modeling_goal=discovery_full_cohort_fit`, fit on all eligible cells and evaluate on all eligible cells. This should use the full regularization path:

- L1 logistic regression over a `C` grid
- L2 logistic regression over a `C` grid
- Elastic net logistic regression over a `C` grid and `l1_ratio` grid

The main output is not just performance. It is the coefficient path and factor/gene selection behavior.

Recommended grids for the first GPU pass:

```text
C_grid = 1 / logspace(-4, 4, 17)
l1_ratio_grid = 0.1, 0.5, 0.9
class_weight = balanced
```

If convergence is slow, use a coarser first pass:

```text
C_grid = 1 / logspace(-3, 3, 13)
l1_ratio_grid = 0.1, 0.5, 0.9
```

Required discovery outputs:

- metrics for every `(panel, representation, penalty, C, l1_ratio)` row
- coefficient vector for every row
- nonzero coefficient count for L1 and elastic net
- selected factor/gene IDs for L1 and elastic net
- coefficient path table per panel/representation
- compressed cell-prediction bundle for downstream cell-level audits:
  - `cell_prediction_matrix.npz`: one probability row per Stage 2 fit and one column per cell
  - `cell_prediction_fit_metadata.csv`: fit metadata keyed to the matrix rows
  - `cell_metadata.csv.gz`: cell IDs, labels, patient, predicted annotation, pseudotime, sample, and timing metadata keyed to the matrix columns
- best rows by AUPRC, balanced accuracy, sparsity-aware score, and biological review score

The discovery full-cohort fit is an apparent/in-sample fit. Label it that way in every output. Do not cache a separate discovery by-patient summary table as a primary artifact; derive patient, cell-type, pseudotime, and other subgroup summaries downstream from the prediction matrix plus cell metadata. This keeps the run artifact compact while preserving the cell-level information needed to ask which cells or cell populations are poorly represented by a given Stage 0 panel and regularized Stage 2 path.

### 5. LOPO Sharedness Should Be The Main Patient-Heldout Layer

For `modeling_goal=sharedness_leave_patient_out`, use leave-one-patient-out rather than GroupKFold-by-patient.

Purpose:

> Ask whether a model trained on all other patients transfers to one heldout patient on the shared transductive Stage 1 basis.

Initial LOPO implementation should be simpler than the discovery path:

- primary baseline: L2 logistic regression with `C=1.0`, matching the quick screen's broad classifier
- optional sparse checks: selected L1/elastic-net settings from the discovery path, not a full nested hyperparameter search

Do not use LOPO as the main regularization-strength search surface in the first implementation. Its main value is the per-patient heldout vector.

Heldout-patient handling:

- Patients with both malignant and non-malignant cells can receive AUROC, AUPRC, balanced accuracy, precision, recall, specificity, TP/FN/TN/FP.
- Normal-only heldout patients cannot receive AUROC/AUPRC for malignant-vs-normal discrimination, but should still be reported as specificity-only / false-positive-burden rows.
- Patients with very few malignant cells, such as fewer than 10, should not be silently dropped. Report them with an instability flag such as `low_malignant_support=true`.

Required LOPO outputs:

- one aggregate scorecard row per model setting
- one long-form by-heldout-patient table per model setting
- cell-weighted aggregate metrics
- patient-equal aggregate metrics where each evaluable patient contributes equally
- explicit skip/instability reasons

LOPO cell-level prediction logging is optional. The main LOPO deliverable is the heldout-patient transfer vector, so per-cell LOPO predictions should only be saved when a specific downstream calibration or per-cell heldout-error analysis needs them.

### 6. Patient-Specific Models Should Also Use Regularization Paths

For `modeling_goal=patient_specific`, fit separate models within each patient using the same shared Stage 1 basis. This should use L1, L2, and elastic-net paths, because the key biological output is which factors each patient uses.

The primary patient-specific result should be:

> Fit on all eligible cells from that patient and evaluate apparent within-patient fit, while saving coefficient paths.

This is analogous to the discovery full-cohort model but scoped to one patient. It is not intended as a strict generalization benchmark.

Within-patient heldout cell splits are optional and should be labeled carefully:

- They can estimate within-patient interpolation and overfitting risk.
- They do not test cross-patient sharedness.
- They may still be optimistic because cells from one patient are correlated.
- They are only meaningful when the patient has enough malignant and non-malignant cells for stratified CV.

Recommended patient-specific policy:

- Always run apparent full-patient fit for patients with both classes and enough cells.
- Run repeated stratified within-patient CV only when the minority class has enough support, for example `n_minority >= 20`.
- For `5 <= n_minority < 20`, optionally run a very small stratified CV or bootstrap stability check, but mark it as unstable.
- For `n_minority < 5`, skip within-patient CV and keep only apparent fit plus support counts.

Required patient-specific outputs:

- patient-level metrics and support counts
- coefficient paths per patient
- selected factors/genes per patient
- overlap with full-cohort discovery selected factors
- overlap with LOPO/shared model selected factors if sparse LOPO settings are run
- instability flags for low malignant support

### 7. Suggested First GPU-Server Command Shape

The other server has cuML/GPU support, so the supervised path runner can be GPU-accelerated where possible. The implementation should still write sklearn-compatible metadata and deterministic manifests so CPU fallback is possible.

Suggested command shape:

```bash
python sc_classification/scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py \
  --experiment-dir sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093 \
  --stage1-fit-scope-note transductive_all_eligible_cells \
  --positive-label cancer \
  --negative-label normal \
  --patient-col patient \
  --canonicalize-existing-quick-stage2 \
  --make-shortlist-from-quick-l2 \
  --shortlist-output analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv \
  --run-discovery-full-cohort-fit \
  --run-sharedness-lopo \
  --run-patient-specific \
  --panel-selection shortlist_plus_controls \
  --penalties l1,l2,elasticnet \
  --c-grid-log10-min -4 \
  --c-grid-log10-max 4 \
  --c-grid-n 17 \
  --l1-ratios 0.1,0.5,0.9 \
  --class-weight balanced \
  --decision-thresholds 0.5 \
  --fixed-recall-targets 0.5,0.7,0.8 \
  --top-fraction-thresholds 0.01,0.05,0.10 \
  --save-discovery-cell-predictions \
  --backend cuml \
  --seed 42
```

The exact script name can change, but the output schema should not.

### 8. Human Review Gate After This Pass

After the multi-objective scorecard is produced, pause before exact size-matched HVG controls and the final downstream benchmark.

Human review should decide:

- Which panels survive because they have strong full-cohort discovery signal?
- Which panels survive because they have LOPO sharedness?
- Which panels are patient-specific but biologically compelling?
- Which sparse coefficients/factors are stable across the regularization path?
- Which exact size-matched HVG controls are needed for the final claims?
- Whether direct-gene small-panel results should be interpreted alongside or separately from DR results.

