# Stage 2 Comprehensive Diagnostic Plan (Pre-Shortlist)

Date: 2026-06-19

Status: planning / discussion draft

This note defines a **systematic diagnostic pass** on the completed full-grid Stage 2 results before building a new global shortlist. It is meant to be read alongside:

- `three_stages_knowledge_prior_mal_classification.md` — conceptual Stage 0/1/2 design and PI questions
- `stage0_mrd_sharedness_scorecard_plan.md` — operational run spec, scorecard schema, and shortlist history
- `scripts/knowledge_driven_embedding/older_geneset/README.md` — the 34 curated geneset inventory

## Motivation

The May 26 provisional shortlist (33 representations) was built from a **quick L2 GroupKFold triage**, not from the full multi-objective Stage 2 grid. On **2026-06-05**, the expensive Stage 2 pass completed for **all biological panels**:

- **41 Stage 0 panels** (34 `single_geneset_only` + 7 `single_group_only`)
- Nominally **16 Stage 1 DR combinations** per panel (`pca`, `fa`, `factosig`, `factosig_promax` × K = 5, 10, 20, 40)
- **643 representation rows** in the v2 decision table: 640 successful DR rows plus 3 small-panel `direct_gene` rows
- **Three Stage 2 goals**: discovery full-cohort, sharedness LOPO, patient-specific
- **~65 regularization settings** per discovery and patient-specific fit

A v2 exploratory notebook (`stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb`) produced decision plots and a provisional 103-representation union shortlist. **That shortlist should not be treated as final evidence** until the diagnostics below are complete.

The scientific goal is not only to rank panels by AUPRC. Stage 2 is a **biological probe**: which prior gene spaces contain malignant structure, which signals transfer across patients, and which Stage 1 latent axes within those prior gene spaces look shared vs patient-specific. In this document, use:

- **Stage 0 panel** for the prior gene space / geneset subset.
- **Stage 1 factor** or **latent axis** for a DR-derived feature.
- **Program** only for a biological interpretation after grounding selected latent axes back to genes.

## Canonical Experiment And Artifacts

Experiment directory:

```text
sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093
```

Stage 2 full biological run:

```text
stage2_supervised/multiobjective/runs/20260605_stage2_all_biological_fullgrid_gpu8
```

Primary merged scorecards (canonical):

| Artifact | Rows (approx.) | Unit of analysis |
| --- | ---: | --- |
| `analysis/scorecards/stage2_discovery_full_cohort_scorecard.csv` | 41,795 | representation × regularization row |
| `analysis/scorecards/stage2_sharedness_lopo_scorecard.csv` | 41,795 | representation × regularization row |
| `analysis/scorecards/stage2_patient_specific_scorecard.csv` | 376,155 | representation × patient × regularization row |
| `stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv` | 543,335 | representation × heldout patient × regularization row |
| `stage2_supervised/multiobjective/*/coefficient_paths.csv` | large | coefficient paths per goal |

Exploratory v2 outputs (starting point, not final):

```text
analysis/scorecards/stage2_figure3_sharedness_v2_jun4/
  v2_decision_table_discovery_lopo_patient_specific.csv   # 643 representation rows
  v2_lopo_patient_best_regularization_long.csv
  v2B1_lopo_patient_best_auprc_matrix.csv
  v2B2_lopo_patient_auprc_enrichment_matrix.csv
  v2C_top_patient_specific_gap_candidates.csv
  top_stage1_shortlist/   # provisional re-shortlist; do not finalize before diagnostics
```

**Gap:** the Jun 5 full-grid run covers **biological panels only**. HVG anchors, `full_34`, and `core_only` controls were not re-run on the same Stage 2 machinery. Layer 6 below addresses this.

## Scope: What A "Representation" Is

A **representation** is one Stage 0 panel plus one Stage 1 setting:

```text
{stage0_panel_id} | dr | {stage1_method} | {K} | {seed}
```

Example: `single_geneset__hallmark_inflammatory_response|dr|fa|20|42`

Here `dr` is the representation family marker, not the DR method; the method is the next field (`pca`, `fa`, `factosig`, or `factosig_promax`). Current full-grid biological rows use `stage1_seed=42`, so the Jun 5 run can measure **method/K sensitivity**, but not seed stability.

Diagnostics should **collapse 643 representation rows → 41 panels** for shortlist decisions, while keeping method/K detail for Stage 1 stability analysis. When counting the intended DR grid, remember that several small panels cannot support larger requested K values, and three small panels also appear as `direct_gene` fallback rows.

## Existing v2 Decision Quadrants (Exploratory Only)

The v2 notebook assigns each of the **643 representations** a `decision_quadrant` by splitting on the **cohort-wide medians** of:

- **Discovery AUPRC** — best regularized full-cohort apparent fit (`select_regularization_rows`, metric = `stage2_auprc`)
- **LOPO AUPRC** — best regularized aggregate held-out fit (metric = `cell_weighted_auprc`)

```text
                        LOPO AUPRC
                     low (<median)     high (≥median)
Discovery AUPRC  high  cohort_apparent_not_shared   shared_candidate
                 low   weak_or_unresolved           heldout_transfer_surprise
```

### Label definitions

| Label | Rule | Plain-language read |
| --- | --- | --- |
| `shared_candidate` | discovery ≥ median AND lopo ≥ median | Above-median cohort signal and above-median cross-patient transfer. |
| `cohort_apparent_not_shared` | discovery ≥ median AND lopo < median | Strong apparent cohort fit that does **not** transfer — cohort-specific or overfit. |
| `heldout_transfer_surprise` | discovery < median AND lopo ≥ median | Transfers on held-out patients better than cohort-average fit suggests — heterogeneous/subgroup biology. |
| `weak_or_unresolved` | both < median | No strong signal on either axis at this resolution. |

Companion fields in the decision table:

- `discovery_minus_lopo` = discovery AUPRC − LOPO AUPRC (overfit / cohort-illusion magnitude)
- `patient_specific_gap` = max patient-specific AUPRC − LOPO AUPRC (patient-specific opportunity)
- `max_patient_auprc`, `median_patient_auprc`, `n_patient_specific`

### Snapshot counts (representation level, Jun 5 v2 table)

| Quadrant | Representations |
| --- | ---: |
| `shared_candidate` | 287 |
| `weak_or_unresolved` | 286 |
| `cohort_apparent_not_shared` | 35 |
| `heldout_transfer_surprise` | 35 |

### Important caveats

1. **Median splits are relative, not absolute.** A `shared_candidate` can still have modest absolute AUPRC if the cohort is globally hard.
2. **643 rows ≠ 41 panels.** Most panels contribute ~16 representations; quadrant counts are inflated at the representation level.
3. **Patient-specific signal is not in the quadrant name.** Many `shared_candidate` rows also have very high `max_patient_auprc` (median gap ≈ 0.79; ~85% with max patient AUPRC > 0.8 in the Jun 5 snapshot).
4. **Quadrants are triage tags, not biological verdicts.**

### Panel-level snapshot (best LOPO per panel, Jun 5)

When each of the **41 panels** is summarized by its **best-LOPO representation**:

| Quadrant | Panels |
| --- | ---: |
| `shared_candidate` | 29 |
| `heldout_transfer_surprise` | 5 |
| `weak_or_unresolved` | 5 |
| `cohort_apparent_not_shared` | 2 |

Illustrative examples from the v2 table:

- **`cohort_apparent_not_shared`:** EMT, apoptosis, TGF plasticity — high discovery, low LOPO, but `max_patient_auprc ≈ 1.0` → strong within-patient biology that does not generalize as a shared axis.
- **`heldout_transfer_surprise`:** interferon alpha, p53, interferon signaling — low discovery, decent LOPO → immune/stress programs with heterogeneous cohort-average fit but transferable held-out signal.
- **`weak_or_unresolved` (panel level):** tiny or direct-gene panels such as NF-κB activation, TNFR1 NF-κB, MHC folding/loading.

## Principle: Diagnose Before Shortlisting

Shortlisting should come **after** answering:

1. Is the signal real or a prevalence / support artifact?
2. Is Stage 1 choice stable or brittle within a panel?
3. Is "shared" actually shared across patients, or driven by one patient?
4. Are winning models interpretable (sparse factors) or dense kitchen-sink fits?
5. Does the prior gene space add value over gene-count-matched controls?

## Diagnostic Layers

### Layer 1: Stage 1 stability within each Stage 0 panel

**Question:** Is the biology in the geneset, or in a lucky DR setting?

**Unit of analysis:** primarily `stage0_panel_id × stage1_method`, with K treated as an ordered capacity curve inside a fixed method. A secondary panel-level summary can collapse over methods, but the four DR methods should not be treated as exchangeable replicates because they encode different inductive biases.

**Metrics:**

- Per method: K curve of aggregate LOPO AUPRC, discovery AUPRC, and patient-specific gap
- Per method: K-stability summary, e.g. IQR, range, max-minus-median, and best-vs-second-best gap across available K values
- Per panel: fraction of available method/K combos landing in each `decision_quadrant`
- Per panel: count of above-median LOPO combos and count of `shared_candidate` combos
- Optional once multi-seed runs exist: seed stability within fixed panel/method/K. The current Jun 5 biological full-grid uses only `stage1_seed=42`, so seed stability cannot be estimated from this run.

**Method-aware stability labels:**

| Label | Operational read |
| --- | --- |
| `method_robust` | Multiple DR methods show above-median LOPO and at least one method is K-stable. Supports a panel-level prior-gene-space claim. |
| `method_specific_k_stable` | One DR method works consistently across K. Not necessarily brittle; interpret as method/inductive-bias specific. |
| `k_sensitive` | Signal appears within a method but changes strongly with K. Treat the required dimensionality as part of the biological hypothesis. |
| `single_spike_brittle` | Only one method/K cell works, with weak neighboring K or method support. Do not make a robust panel claim. |
| `null` | No available method/K setting clears the chosen LOPO/lift threshold. |

K should be interpreted cautiously. Within a method, increasing K asks whether more latent capacity is needed under roughly the same decomposition assumptions. Across methods, however, PCA, FA, FactorSig, and promax-rotated FactorSig have different inductive biases, so method-specific success is not automatically a failure of the Stage 0 panel.

**Outputs:**

```text
analysis/scorecards/stage2_diagnostic/
  panel_method_k_lopo_distribution.csv
  panel_method_stage1_stability_summary.csv
  panel_stage1_stability_summary.csv
```

**Biology:** Stable panels (signal across many methods/K) support a claim about the **prior gene space**. Method-specific but K-stable panels should be labeled as DR-inductive-bias-specific rather than dismissed. Single-spike panels should be labeled DR-sensitive, not robust program evidence.

---

### Layer 2: Patient transfer topology

**Question:** Who drives "sharedness"?

**Unit of analysis:** `stage0_panel_id × heldout_patient_id`, optionally stratified by `stage1_method`. Use the best available method/K row within a panel/method for patient-transfer summaries, rather than averaging arbitrary representation-level heatmap rows.

**Metrics:**

- Per panel or panel/method: median held-out AUPRC and median held-out lift across supported patients
- 20th percentile held-out lift: a more realistic transfer floor than the minimum in a small, heterogeneous cohort
- Number and fraction of evaluable patients with positive lift over malignant prevalence
- Worst supported patient and best supported patient by lift
- Patient leverage: how much the panel's aggregate transfer summary changes when each supported patient is dropped
- Raw AUPRC, **additive lift** over prevalence (`heldout_AUPRC - malignant_prevalence`), and **prevalence-normalized enrichment** (`heldout_AUPRC / malignant_prevalence`)
- Explicit flags: `normal_only`, `low_malignant_support` (per `stage0_mrd_sharedness_scorecard_plan.md`)

**Held-out lift definition:**

For a held-out patient, malignant prevalence is the baseline AUPRC expected from a random ranking:

```text
malignant_prevalence = n_malignant / (n_malignant + n_non_malignant)
heldout_lift = heldout_AUPRC - malignant_prevalence
```

This is an additive improvement over the patient-specific baseline. A held-out AUPRC of 0.10 can be meaningful if malignant prevalence is 0.01 (`lift = +0.09`), but weak if prevalence is 0.30 (`lift = -0.20`). Use lift when asking whether a model ranks malignant cells better than chance within a patient; use raw AUPRC when asking whether absolute performance is high enough to be practically useful. The ratio `heldout_AUPRC / malignant_prevalence` is useful for rare-patient enrichment, but it can explode when prevalence is tiny, so it should be reported alongside raw AUPRC, lift, and support counts.

**Outputs:**

```text
  panel_lopo_patient_summary.csv
  panel_method_lopo_patient_summary.csv
  panel_lopo_patient_transfer_long.csv
  panel_lopo_patient_enrichment_summary.csv
  patient_leverage_on_aggregate_lopo.csv
```

**Biology:**

- Broad positive lift across supported patients → candidate shared malignant signal in that prior gene space
- Single-patient peaks → **patient-specific axis** (see IL2/STAT5 / P02-type examples in `three_stages_...md`)
- High enrichment but low raw AUPRC → rare malignant cells ranked well; interesting but unstable
- "Consistent across all patients" may be too strict for this cohort because several patients are normal-only and some have very low malignant support. Prefer a graded topology: broad-transfer, partial-transfer, single-patient-driven, low-support-only, and no-transfer.

---

### Layer 3: Three-way discovery / LOPO / patient-specific taxonomy

**Question:** Is the panel shared, cohort-illusory, patient-specific, or null?

Extend the 2D quadrant to a **panel-level taxonomy** using best regularized rows per goal:

| Pattern | Discovery | LOPO | Max patient-specific | Interpretation |
| --- | --- | --- | --- | --- |
| Truly shared | high | high | moderate | Cross-patient program |
| Cohort illusion | high | low | high | Patient-specific biology masked as cohort signal |
| Hidden transfer | low | high | variable | Subgroup / heterogeneous program (IFN-type) |
| Patient-only | low | low | high | Strong in one patient, not a shared model |
| Null | low | low | low | Prior space does not separate mal vs non-mal |

**Outputs:**

```text
  panel_three_way_taxonomy.csv
```

**Biology:** Resolves tension where many `shared_candidate` representations also have large `patient_specific_gap`. A panel can be scientifically important without being a shared model.

---

### Layer 4: Regularization and interpretability pathology

**Question:** Are we selecting biology or a dense, unstable kitchen-sink model?

**Checks:**

- Sparsity audit: fraction of winners with `nonzero_coefficient_count ≈ effective_k` (dense L2-like solutions)
- Apply **near-best sparse rule** when max-AUPRC row is dense (tolerance on AUPRC, prefer L1/elastic-net per penalty priority)
- Coefficient path stability across regularization grid
- LOPO coefficient stability across held-out patients (Fig 3C logic in `stage2_sharedness_plotting.py`)

**Outputs:**

```text
  representation_interpretability_flags.csv
  panel_coefficient_stability_summary.csv
```

**Biology:** Stage 2 identifies **which latent factors** carry signal. Dense or fold-unstable factors weaken biological claims even when AUPRC is high.

---

### Layer 5: Factor-to-gene grounding (Stage 1 → biology)

**Question:** Does each prior program decompose into meaningful sub-axes?

**For top panels across quadrants (not only top AUPRC):**

- Map selected Stage 2 coefficients → Stage 1 factor indices → top positive/negative loading genes
- Jaccard overlap of selected factors: discovery vs LOPO vs patient-specific
- Within one geneset: do different patients select different factors on the same Stage 1 basis?

**Outputs:**

```text
  panel_selected_factors_summary.csv
  panel_top_loading_genes.csv
  patient_factor_overlap_vs_discovery.csv
```

**Biology:** Answers PI Q3 — one geneset may be multiple regulatory programs; patient-specific factor usage indicates heterogeneous malignant state within a shared coordinate system.

**Note:** `v2D_top_loading_genes_all_review_representations.csv` was empty as of Jun 5; this layer needs implementation.

---

### Layer 6: Gene-budget and panel-family calibration

**Question:** Does the old-34 prior beat matched gene count?

**Required comparisons (not yet in full-grid scorecards):**

- HVG anchors (500, 1000, 3000, 10000)
- `full_34`, `core_only`
- Size-matched HVG panels per knowledge-driven panel (later)

**Stratify by panel family:**

- `single_geneset_only`
- `single_group_only`
- controls

**Metrics:**

- AUPRC lift over malignant-prevalence baseline (`heldout_AUPRC - malignant_prevalence`)
- Control-adjusted lift:

```text
knowledge_prior_delta = panel_lift - median(size_matched_HVG_control_lift)
```

- Empirical percentile or z-score versus multiple size-matched HVG controls when replicate controls are available
- Discovery vs LOPO gap vs controls at similar gene count
- Optional: gene-count-normalized summaries, but do not use them as a substitute for size-matched controls

**Biology:** A 150-gene curated panel beating a 150-gene HVG control is a knowledge-prior win; beating 10k HVG is a stronger claim. The most defensible evidence is a positive control-adjusted held-out lift with uncertainty across patients and matched-control replicates.

---

### Layer 7: Theme- and panel-family synthesis

**Question:** What are the cross-panel biological patterns?

Collapse diagnostics to **biology themes** (from `infer_biological_theme` / manifest groupings):

- interferon, cytokine/JAK/STAT, antigen presentation, NF-κB/TNF, stress/arrest, cell cycle, metabolism, other
- Also summarize the 7 `single_group_only` biology-group panels

**Per theme:**

- Count of panels in each three-way taxonomy class
- Best LOPO panel, most stable panel, most patient-specific panel
- Whether theme enrichment differs from representation-level quadrant noise

**Outputs:**

```text
  theme_diagnostic_summary.csv
  biology_group_diagnostic_summary.csv
```

## Recommended Workflow (Ordered)

```text
1. Panel summary table (41 rows)
   - best Stage 1 by LOPO/lift, method-aware stability, three-way taxonomy, theme, n evaluable LOPO patients

2. Patient transfer audit
   - panel-level held-out lift table, 20th percentile lift, positive-lift patient count, patient leverage, support flags

3. Three-way taxonomy (discovery / LOPO / patient-specific)
   - tag each panel, not each of 643 representations

4. Interpretability filter
   - down-rank or flag dense / unstable coefficient paths

5. Factor-gene grounding
   - top 10–15 panels across taxonomy classes

6. Control comparison
   - run or merge HVG / full_34 / core_only on same Stage 2 machinery

7. Shortlist (only after 1–6)
   - primary: stable + high LOPO + interpretable
   - secondary: heldout_transfer_surprise + patient-specific peaks for case studies
   - negative: cohort_apparent_not_shared with no patient-specific story
```

## Proposed Primary Deliverable: 41-Row Panel Diagnostic Table

Minimum columns for discussion and shortlist decisions:

| Column | Description |
| --- | --- |
| `stage0_panel_id` | Panel identifier |
| `stage0_panel_type` | `single_geneset_only` or `single_group_only` |
| `biological_theme` | Theme label |
| `n_covered_genes` | Gene budget |
| `best_lopo_representation_id` | Best Stage 1 by aggregate LOPO |
| `best_lopo_auprc` | Aggregate LOPO AUPRC |
| `best_lopo_lift_median` | Median held-out lift across supported patients for the selected panel/method |
| `lopo_lift_p20` | 20th percentile held-out lift across supported patients |
| `n_positive_lift_patients` | Supported held-out patients with AUPRC above malignant-prevalence baseline |
| `best_discovery_auprc` | Discovery AUPRC at matched or best row |
| `max_patient_auprc` | Best patient-specific apparent AUPRC |
| `method_stability_label` | `method_robust`, `method_specific_k_stable`, `k_sensitive`, `single_spike_brittle`, or `null` |
| `stage1_stability_iqr` | LOPO IQR across available method/K combos |
| `n_shared_candidate_combos` | Of 16 Stage 1 settings |
| `three_way_taxonomy` | Layer 3 class |
| `v2_quadrant_at_best_lopo` | Exploratory quadrant tag |
| `n_evaluable_lopo_patients` | Patients with usable held-out metrics |
| `worst_patient_auprc` | Floor of transfer |
| `worst_patient_lift` | Lowest supported-patient lift |
| `best_patient_auprc` | Ceiling of transfer |
| `median_enrichment` | Prevalence-normalized held-out performance |
| `interpretability_flag` | dense / sparse / unstable |
| `diagnostic_notes` | Free-text or enum for case-study priority |

## Relationship To Existing Shortlists

| Shortlist | Source | Count | Role after diagnostics |
| --- | --- | ---: | --- |
| `stage2_provisional_shortlist_from_quick_l2.csv` | Quick L2 triage (May 26) | 33 selected | Historical; superseded for ranking |
| `top_stage1_shortlist/stage2_v2_top_stage1_union_selection.csv` | Full-grid union (Jun 5) | 103 | Provisional; do not finalize |
| `top_stage1_shortlist/stage2_v2_top_stage1_primary_lopo_selection.csv` | Best LOPO per panel + controls | 47 | Starting point for Layer 1–3 |

## Open Questions For Discussion

1. **Thresholds:** Should taxonomy classes use absolute AUPRC cuts, prevalence-adjusted enrichment, or quantiles within panel family?
2. **Stage 1 selection rule:** Best LOPO only, or require stability across ≥N methods?
3. **Patient-specific prominence:** Should high `patient_specific_gap` elevate a panel for case-study follow-up even when LOPO is weak?
4. **Controls timing:** Re-run Stage 2 on controls now, or complete biological diagnostics first?
5. **Shortlist size target:** ~30–40 for figure suite vs ~10–15 for deep factor interpretation?
6. **Representation vs panel claims:** What language is allowed in figures when only 1 of 16 Stage 1 combos works?

## Implementation Notes

- Reuse helpers in `scripts/mrd_stage0_2/stage2_supervised/stage2_sharedness_plotting.py` (`select_regularization_rows`, `infer_biological_theme`, etc.).
- v2 notebook: `scripts/mrd_stage0_2/notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb`
- Suggested output root: `analysis/scorecards/stage2_diagnostic/`
- Suggested next artifact: notebook or script `stage2_comprehensive_diagnostic_suite.ipynb` / `.py` implementing Layers 1–3 and the 41-row table first.

## References

| Doc | Role |
| --- | --- |
| `three_stages_knowledge_prior_mal_classification.md` | PI questions, suggested plots, interpretation framing |
| `stage0_mrd_sharedness_scorecard_plan.md` | Stage 2 goals, metrics, support flags, run commands |
| `stage0_geneset_value_added_workflow.md` | Stage 0 panel families and bottom-up design |
| `older_geneset/README.md` | 34 geneset inventory |
