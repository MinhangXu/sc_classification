# Experiment 20260525 Stage 0–2 Rehydration Dossier

Date compiled: 2026-07-02
Experiment: `experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093`

This is the single master document for the MRD knowledge-prior malignancy-classification
experiment. It rehydrates the pipeline, assigns provenance to every runner / plan /
notebook, maps analysis to scientific questions, and lists what is still missing. It is
meant to be read alongside the three design plans it references:

- `three_stages_knowledge_prior_mal_classification.md` — conceptual Stage 0/1/2 design and the three PI questions
- `stage0_mrd_sharedness_scorecard_plan.md` — operational run spec, scorecard schema, artifact Q&A
- `stage2_comprehensive_diagnostic_plan.md` — the 7-layer post-run diagnostic plan (target state)

---

## 0. TL;DR

- **One experiment directory holds two Stage 0 panel sets**, namespaced by branch:
  - **Set A "old34"** (root namespace) — 47 panels, the *original / older* run (May 25).
  - **Set B "expanded_stage0_mrd_manuscript_axes_v1"** (branch namespace) — 632 panels, the *newer* run (Jun 22–25).
- **Both sets completed the full pipeline**: Stage 0 (panels) → Stage 1 (across-patient DR) → quick Stage 2 (L2 GroupKFold) → multi-objective Stage 2 (discovery / LOPO / patient-specific reg-paths).
- **Analysis is lopsided.** The old34 set has four analysis notebooks (metric diagnostics, scorecard audit, and two Figure-3 sharedness suites). **The expanded set has *no* scorecard-driven analysis notebook** — only a single-panel Jaatinen HSC case study. This confirms the suspicion that "the new run wasn't analyzed."
- **The post-run diagnostic plan (Layers 1–7) is only ~25% built** even for old34: only Layers 1–2 (Stage 1 stability + patient transfer) exist. Layers 3–7 (three-way taxonomy, interpretability, **factor→gene grounding**, gene-budget/control calibration, theme synthesis) are not implemented. The factor→gene grounding table is empty.
- **Provenance risk:** the entire current Stage 0–2 codebase (runners, plans, plotting) is **uncommitted** (git `??`), and `experiment_config.yaml` records `git_commit b381cd5`, which *predates* those runners. The recorded commit does not contain the code that produced the run.

---

## 1. The experiment at a glance

| Field | Value |
| --- | --- |
| Experiment ID | `20260525_060508_stage0_mrd_old34_broad_screen_82db5093` |
| Created / completed (Set A) | 2026-05-25 06:05 → 13:22 |
| Created / completed (Set B) | 2026-06-22 04:35 → 2026-06-25 15:03 |
| Input AnnData | `data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad` (14.2 GB, mtime 2026-03-31) |
| Cohort filter | timepoint = MRD, tech = CITE |
| Target | `CN.label`: positive = `cancer`, negative = `normal` |
| Patient column | `patient` |
| Preprocessing | `min_cells_fraction=0.01`, `normalize_total_log1p` (target_sum 10000), within-panel feature z-score before DR/direct-gene, HVG rank = post-filter log1p variance |
| Filtered cohort | **60,118 cells × 15,560 genes**, 13 patients (P01–P13) |
| Malignant cells | **1,569 total → prevalence ≈ 0.026** |
| Patient support | P08, P10, P11, P12 are **normal-only** (0 malignant); P05 (8), P07 (22), P13 (4) have very low malignant support |
| Recorded git commit | `b381cd5` (see §8 — misleading) |

**Why prevalence matters:** at 0.026 malignant, a random ranker's expected AUPRC ≈ 0.026.
Every AUPRC in this experiment should be read against that baseline (lift = AUPRC − prevalence).

---

## 2. The three-stage pipeline and where its artifacts live

All paths are relative to the experiment directory. Set A ("old34") writes to the root
namespace; Set B writes under a `expanded_stage0_mrd_manuscript_axes_v1/` subfolder in each
tree.

```text
Stage 0  prior gene-space panels
   preprocessing/panels/stage0_panel_manifest.csv                     (Set A)
   preprocessing/panels/stage0_panel_genes/<panel>.json               (Set A)
   preprocessing/panels/expanded_stage0_mrd_manuscript_axes_v1/...    (Set B)
        |
        v
Stage 1  across-patient (transductive) latent basis, seed=42
   stage1_dr/panel_id=<panel>/method={pca,fa,factosig,factosig_promax}/k={5,10,20,40}/seed=42/
        scores.npy, loadings.npy, top_loading_genes.csv, metadata.json
   stage1_direct_gene/panel_id=<panel>/...   (small panels below DR-K, direct_gene fallback)
        |
        v
Stage 2a  quick screen: L2 logistic, GroupKFold-by-patient (5 folds), C=1.0, class_weight balanced
   stage2_supervised/shared_cross_patient/panel_id=<panel>/representation=dr/method=.../k=.../seed=42/
        quick_l2_groupkfold_metrics.json, quick_l2_groupkfold_predictions.csv
        |
        v
Stage 2b  multi-objective: L1/L2/EN regularization paths, 3 modeling goals
   stage2_supervised/multiobjective/{discovery_full_cohort,sharedness_lopo,patient_specific}/
        best_rows.csv, coefficient_paths.csv, by_heldout_patient.csv,
        cell_prediction_matrix.npz, cell_prediction_bundle.npz, cell_metadata.csv.gz
   stage2_supervised/multiobjective/runs/<run_id>/{shards,merged,launcher_manifest.json}
```

**Stage 1 is transductive** (fit once on all 60,118 eligible cells, including cells from
patients later held out in Stage 2). LOPO therefore measures *label transfer on a shared
atlas*, not fully inductive deployment. This is intentional and defensible for the biology
goal, but must be labeled that way in any figure (`stage1_fit_scope_note = transductive_all_eligible_cells`).

---

## 3. The two Stage 0 panel sets

Both sets are the "first run" the user referred to — same cohort, same Stage 1/2 machinery,
different Stage 0 gene-space dictionaries.

| | **Set A — old34** (older) | **Set B — expanded manuscript axes** (newer) |
| --- | --- | --- |
| Branch namespace | root (none) | `expanded_stage0_mrd_manuscript_axes_v1` |
| Stage 0 dictionary | `knowledge_driven_embedding/older_geneset/genesets_v1.gmt` | `knowledge_driven_embedding/expanded_stage0_genesets/final_bundle.gmt` |
| Stage 0 scorecard | `analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv` (755 rows) | `analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/stage0_mrd_old34_broad_scorecard.csv` (10,487 rows) |
| Unique panels | **47** | **632** |
| Panel-type composition | 34 `single_geneset_only`, 7 `single_group_only`, 1 `full_34`, 1 `core_only`, 4 `hvg_anchor` | 566 `atomic_sets`, 26 `family_union_sets`, 12 `single_group_only`, 12 `leave_one_family_out`, 9 `core_anchor_sets`, 1 `full_34`, 1 `core_only`, 1 `all_filtered`, 4 `hvg_anchor` |
| Panel dictionary theme | 34 curated MSigDB programs (IFN, cytokine/JAK-STAT, antigen presentation, NF-κB/TNF, stress, cell cycle, metabolism) | GO/Reactome-derived "atomic" gene sets decomposed by biological family, plus family unions and leave-one-family-out ablations |

**Stage 2 multi-objective coverage differs and this is the key asymmetry:**

| | Set A — old34 | Set B — expanded |
| --- | --- | --- |
| Multi-objective run(s) | `runs/20260526_stage2_*`, `runs/20260529_stage2_lopo_fullgrid`, `runs/20260605_stage2_all_biological_fullgrid` | `runs/expanded_..._stage2` (completed 2026-06-25) |
| Discovery rows | 41,795 | 24,990 |
| Panels in multi-objective | **41 biological only** (34 geneset + 7 group) — **controls excluded** | **294** (254 atomic, 15 family_union, 9 core_anchor, 7 group, 5 LOFO, 4 hvg_anchor) |
| Patient-specific rows | 376,155 | 224,910 |
| Cell-prediction bundles saved | Yes (discovery `cell_prediction_matrix.npz`, patient-specific `cell_prediction_bundle.npz`) | **No** (`--save-*-cell-predictions` not passed; 0 cell rows) |

---

## 4. Provenance map — runners → artifacts

All runners live in `scripts/comprehensive_run/`.

| Runner / wrapper | Produces | Which set |
| --- | --- | --- |
| `run_stage0_mrd_old34_broad_screen.py` | Stage 0 panels + Stage 1 DR + quick Stage 2 + Stage 0 scorecard + `postrun_human_review.md` | Both (Set B via `--branch-name`) |
| `run_stage0_mrd_old34_broad_screen.sh` | thin wrapper that launched Set A | Set A |
| `run_expanded_stage0_genesets_stage0_to_stage2.sh` | builds expanded bundle, then calls the Stage 0 runner with `--branch-name expanded_...`, then optionally the multi-objective Stage 2 runner (GPU-sharded) | Set B |
| `run_stage2_mrd_multiobjective_scorecard.py` | discovery / LOPO / patient-specific reg-path scorecards, coefficient paths, cell-prediction bundles, provisional shortlist, canonical quick scorecard | Both |
| `stage2_sharedness_plotting.py` | shared helpers (`select_regularization_rows`, `infer_biological_theme`, `shorten_panel_label`, factor-usage plotting) used by notebooks + audit scripts | Both |
| `stage2_mrd_fig3a_interactive.py`, `..._top_stage1_shortlist.py` | interactive Fig 3A HTML | Set A |
| `build_lopo_transfer_audit_tables.py` | patient-wise LOPO transfer audit tables/plots (Layer-2-style) | Set A (default path hard-coded) |

**Upstream gene-space builders** (not in `comprehensive_run/`):
- Set A dictionary: `knowledge_driven_embedding/older_geneset/build_gmt.py` → `genesets_v1.gmt`, `manifest.tsv`
- Set B dictionary: `knowledge_driven_embedding/expanded_stage0_genesets/build_expanded_stage0_bundle.py` → `final_bundle.gmt`, `final_manifest.tsv`, `selector_provenance.json`

---

## 5. Provenance map — plans → status → relevance

Full classification in §7. Quick view:

| Plan doc | Implements / describes | Runner | Class |
| --- | --- | --- | --- |
| `three_stages_knowledge_prior_mal_classification.md` | the Stage 0/1/2 design + 3 PI questions | S0/S1 + S2 runners | **CORE** |
| `stage0_mrd_sharedness_scorecard_plan.md` | run spec, scorecard schema, artifact Q&A | S2 runner | **CORE** |
| `stage2_comprehensive_diagnostic_plan.md` | 7-layer post-run diagnostic (target) | notebook/script TBD | **CORE (mostly unbuilt)** |
| `stage0_geneset_value_added_workflow.md` | bottom-up gene-space value-added design | `run_old_geneset_pruning_metrics.py`, expanded wrapper | **SUPPORTING** |
| `posthoc_dr_validation_eval_plan.md` | DR validation playbook; Stage 0 addendum | Plan 0 notebooks | **SUPPORTING** |
| `comprehensive_run_reorganization_plan.md` | repo hygiene / rehydration scaffolding | none | **SUPPORTING** |
| `active_plan0_plan1.md` | older HVG/all_filtered K-sweep + preprocess×DR grid | `run_gene_filter_dr_grid.py` | **SUPERSEDED** |
| `plan1c_cross_patient_supervised_latent_benchmark.md` | fixed-K supervised benchmark on Feb HVG run | `run_plan1c_supervised_latent_benchmark.py` | **SUPERSEDED** |
| `plan0rotationseedsplan1stability.md` | FA-rotation + multi-seed engineering overlay | `run_gene_filter_dr_grid.py` (partial) | **SUPERSEDED** |
| `later_plans2_4.md` | Plan 2/3/4 (permutation, representation-first, two-stage) | skeleton runners only | **SUPERSEDED / never run** |

---

## 6. Provenance map — notebooks → what they analyze

| Notebook | Reads | Set | Goal | Status |
| --- | --- | --- | --- | --- |
| `analysis/stage0_mrd_old34_metric_diagnostics_20260525.ipynb` | Stage 0 scorecard (root) + per-row predictions | A | **(1) run evaluation** — metric intuition, confusion counts, K/DR consistency | polished |
| `analysis/stage2_mrd_multiobjective_scorecard_analysis_20260526.ipynb` | root multi-objective scorecards | A | (1) evaluation + **(2) scientific** review | polished |
| `notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_20260528.ipynb` | root multi-objective scorecards | A | **(2) scientific** — Fig 3A–D sharedness | superseded by v2 |
| `notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` | root multi-objective scorecards + coefficient paths | A | **(2) scientific** — decision quadrants, patient transfer, program decomposition | **newest / authoritative for old34** |
| `analysis/notebook_outputs/jaatinen_hsc_up_stage2_probe_20260622/` (notebook: `notebooks/jaatinen_hsc_up_stage2_probe_20260622.ipynb`) | raw Stage 1/2 artifacts for **one** atomic panel | B | (2) scientific — HSC-up case study | polished but single-panel |
| `notebooks/stage0_bottom_up_postrun_analysis_20260518.ipynb`, `geneset_pruning_triage_refined.ipynb`, `old_geneset_pruning_metrics_20260430.ipynb`, `plan0_old_geneset_diagnosis_april2_2026.ipynb` | **April `20260401` all_filtered** experiment | neither | (1)/(3) old-geneset pruning + Plan 0 diagnosis | different experiment |
| `notebooks/plan1c_*`, `plan1d_*`, `plan0_diagnosis`, `plan0_k_compare_prelim` | **Feb `20260211` HVG** Plan 0 experiment | neither | older lineage | different experiment |

**Read order to rehydrate the old34 analysis end-to-end:**
`stage0_mrd_old34_metric_diagnostics_20260525` → `stage2_mrd_multiobjective_scorecard_analysis_20260526` → `stage2_mrd_figure3_sharedness_suite_v2_jun4`.

**Confirmed:** grepping all notebooks, only the Jaatinen probe references
`expanded_stage0_mrd_manuscript_axes_v1`, and it reads raw artifacts, **not** the merged
expanded scorecards. No notebook reads
`analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/`.

### 6b. Interactive visualization tool (the exploration companion to the figures)

`stage2_mrd_fig3a_interactive.py` (Jun 19) and its earlier variant
`stage2_mrd_fig3a_interactive_top_stage1_shortlist.py` (Jun 5) are **not notebooks**; they are
standalone Plotly builders for the flagship **Figure 3A** decision plot (discovery AUPRC vs
LOPO AUPRC), with per-representation regularization-path drilldowns, hover metadata, and
patient detail. They read the same old34 root scorecards + coefficient paths and write
`analysis/figures/stage2_figure3_sharedness/fig3A_discovery_vs_lopo_sharedness_interactive.html`.

Where it fits: this is the **interactive exploration layer that sits behind the static Figure 3A**
produced by `stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb`. In the figure-based-thinking
work (below), Fig 3A is the primary "which gene spaces are shared vs cohort-illusory vs
patient-specific" decision plot, and this script is the tool for interrogating it before a
static panel is frozen. It is **old34-only** today; an expanded-set analogue is part of the #2
work. Treat it as reusable analysis tooling (it belongs next to `stage2_sharedness_plotting.py`),
not a one-off.

---

## 7. Plan cleanup — implemented vs branched-out, and does it serve the goal?

The core goal: **build and biologically interpret the Stage 0–2 knowledge-prior malignancy
classifier on this experiment, then answer the three PI questions** (robust shared model,
patient-specific biology, breaking apart programs).

### CORE — directly serves the goal
- **`three_stages_knowledge_prior_mal_classification.md`** — the design itself. Implemented by the S0/S1 runner + S2 multi-objective runner. Keep as the canonical concept doc.
- **`stage0_mrd_sharedness_scorecard_plan.md`** — the operational spec and the authoritative artifact Q&A (fold assignments, patient support, quick-classifier settings). Implemented. Keep.
- **`stage2_comprehensive_diagnostic_plan.md`** — the intended post-run analysis (Layers 1–7). **Mostly aspirational**: only Layers 1–2 are partly realized (in the v2 jun4 notebook + `build_lopo_transfer_audit_tables.py`). This is the main "what to build next" doc.

### SUPPORTING — methodology / inputs that feed the core but are not the core analysis
- **`stage0_geneset_value_added_workflow.md`** — bottom-up "value added by a gene space" design. Its *philosophy* (malignancy-first, budget-matched HVG controls, delta-vs-control columns) is exactly right and partly drives Set B's panel families (atomic/family-union/leave-one-family-out) and the size-matched-HVG idea. Its *runner* (`run_old_geneset_pruning_metrics.py`) targets the older April `20260401` experiment, not this one. **Verdict:** the analytical goal genuinely helps; the concrete pruning runner is on a different experiment. The unfinished part that matters here is the budget-matched-HVG control comparison (Layer 6).
- **`posthoc_dr_validation_eval_plan.md`** — DR-validation playbook (Plan 0 lineage). Its **Stage 0 evaluation addendum** (malignancy-first leaderboard, structure metrics secondary) aligns with the current design and justifies DR/K choices inherited from earlier work. Chunks 1–4 done, 5–6 pending. **Verdict:** supporting; only the Stage 0 addendum is on the critical path.
- **`comprehensive_run_reorganization_plan.md`** — repo hygiene (active/reusable/legacy split, README, notebook index). Partly done (README + `legacy/` exist). **Verdict:** supports rehydration/auditability, no biology.

### SUPERSEDED / TANGENTIAL — older lineage or branched ideas, not on the current path
These all predate the knowledge-prior Stage 0–2 design and run on **other experiments**
(Feb `20260211` HVG, April `20260401` all_filtered). They are valuable as methodology
provenance and reusable code patterns, but they do **not** advance analysis of this experiment.
- **`active_plan0_plan1.md`** — Plan 0 K-sweep + Plan 1 preprocess×DR grid. Implemented (`run_gene_filter_dr_grid.py`). Informed K=40 and DR-method choices; superseded by the S0/S1 runner.
- **`plan1c_cross_patient_supervised_latent_benchmark.md`** — fixed-K (40) supervised latent benchmark with pooled + per-patient CV. Implemented and ran on the Feb HVG latents. Conceptual predecessor of Stage 2; superseded by the multi-objective runner's discovery/LOPO/patient-specific goals.
- **`plan0rotationseedsplan1stability.md`** — FA-rotation + multi-seed engineering overlay on Plan 0–1. Partially implemented (cNMF curation + Plan 0 multi-seed done; `--fa-rotation` and Plan 1 `--seeds` not). Relevant only if revisiting DR selection.
- **`later_plans2_4.md`** — Plan 2 (label-permutation negative controls), Plan 3 (representation-first), Plan 4 (two-stage selection). **Skeleton runners only, never run.** Of these, **Plan 2's label-permutation negative control is the one idea worth resurrecting** for this experiment — it directly tests whether AUPRC lift is real vs a support/prevalence artifact (Diagnostic-plan Layer-1 open question). The other two are tangential.

---

## 8. Provenance risk — the code that produced this run is uncommitted

- `experiment_config.yaml` records `git_commit: b381cd5` ("Add Relapse_MRD_DR_Classification pipeline").
- But `run_stage0_mrd_old34_broad_screen.py`, `run_stage2_mrd_multiobjective_scorecard.py`, `run_expanded_stage0_genesets_stage0_to_stage2.sh`, `stage2_sharedness_plotting.py`, and **every current plan doc** are **untracked** (`git status` = `??`). Commit `b381cd5` does not contain them.
- `experiments/` is git-ignored, so the run artifacts are intentionally out of version control.
- **Consequence:** the run cannot currently be reproduced from the recorded commit. Before any manuscript claim, commit the current `scripts/comprehensive_run/` state and re-stamp (or annotate) the true commit hash. This is the highest-value, lowest-effort provenance fix.

---

## 9. What's missing

### Pipeline-side
1. **Controls absent from old34 multi-objective Stage 2.** The 20260605 run covered the 41 biological panels only; `full_34`, `core_only`, and HVG anchors were never pushed through discovery/LOPO/patient-specific reg-paths. Without them there is no gene-budget-matched baseline (Diagnostic-plan **Layer 6**). Set B *did* include the 4 HVG anchors but still lacks **size-matched** HVG controls.
2. **Expanded multi-objective is a shortlist, not the full grid.** 294 of 632 panels ran (top-per-panel `shortlist_plus_controls`); 338 atomic panels have only quick-L2, no reg-path. Fine for a first pass, but the atomic decomposition story is incomplete.
3. **No cell-level bundles for Set B.** Expanded discovery/LOPO ran without `--save-discovery-cell-predictions`, so per-cell / per-annotation / pseudotime audits are not possible for the new set the way they are for old34.
4. **Single seed everywhere (`seed=42`).** Stage 1 seed stability cannot be estimated for either set.
5. **No inductive sensitivity check.** Everything is transductive; a refit-per-fold sensitivity run for the shortlisted panels is not done.

### Analytical-side (higher priority — this is where the asymmetry lives)
1. **The expanded ("new") set has no scorecard analysis at all.** There is no expanded analogue of `stage2_mrd_figure3_sharedness_suite_v2_jun4` and no run-associated write-up. Only the single Jaatinen panel was examined. **This is the biggest gap and the most direct answer to "we didn't analyze the new run."**
2. **Diagnostic Layers 3–7 unbuilt even for old34.** No `analysis/scorecards/stage2_diagnostic/` directory exists. Missing:
   - Layer 3 — three-way discovery/LOPO/patient-specific panel taxonomy (41-row table)
   - Layer 4 — regularization/interpretability pathology (dense-vs-sparse audit, coefficient stability)
   - **Layer 5 — factor→gene grounding** (`v2D_top_loading_genes_all_review_representations.csv` is **empty, header only**). This is the artifact needed for PI Q3 ("does a program break into sub-axes?") and it does not exist yet.
   - Layer 6 — gene-budget / control calibration (blocked by pipeline gap #1)
   - Layer 7 — biology-theme synthesis
3. **No explicit hypothesis-test framing.** Current plots are decision/triage plots. Per §10, each figure destined for the manuscript needs the four-element framing and a stated null.
4. **Set B → Set A reconciliation missing.** No document or plot relates the expanded atomic/family panels back to the old34 curated programs (e.g., does `single_family__cd54_cd244_adhesion_niche` or `hspc_lsc_stemness` beat the best old34 immune panel on LOPO lift?). The expanded Stage 0 scorecard's top rows suggest the new families outperform old34 (LOPO-blind quick AUPRC up to ~0.66 vs ~0.55), but this has not been analyzed with the multi-objective scorecards.

---

## 10. Required framing for every manuscript plot

Per the working agreement, **every plot that leaves this experiment carries four elements**,
and analyses without them risk being "a collection of ML plots without a hypothesis test":

1. **Scientific question** — the biological question in one sentence.
2. **In silico experiment designed** — the exact Stage 0 panel(s), Stage 1 method/K, Stage 2 goal, split policy, and metric.
3. **Results** — what the plot shows, read against the malignant-prevalence baseline (0.026).
4. **Take-home message** — the biological conclusion, scoped to shared / patient-specific / decomposed.

State the **null hypothesis** precisely first, then the SHT follows mechanically. Template:

> H0: *"This Stage 0 gene space is not associated with malignancy"* — i.e., its best regularized
> LOPO AUPRC does not exceed the held-out malignant-prevalence baseline (equivalently, lift ≤ 0),
> and does not exceed a gene-budget-matched HVG control.
> H1: it does.

Mapping the three PI questions to concrete tests on this experiment:

| PI question | Null to reject | Primary evidence (artifact) |
| --- | --- | --- |
| Q1 Robust shared model | LOPO lift ≤ 0 and ≤ size-matched HVG | `stage2_sharedness_lopo_scorecard.csv`, per-patient `by_heldout_patient.csv` |
| Q2 Patient-specific biology | `max_patient_auprc` ≈ LOPO (no patient-specific gap) | `stage2_patient_specific_scorecard.csv`, patient×panel matrices |
| Q3 Breaking apart programs | selected factors are one axis / identical across patients | coefficient paths + **factor→gene grounding (Layer 5, to build)** |

---

## 11. Recommended next actions (ordered)

1. **Commit the current `scripts/comprehensive_run/` code** and correct the recorded provenance commit (§8). Low effort, unblocks reproducibility.
2. **Build the expanded-set analysis notebook** — an analogue of `stage2_mrd_figure3_sharedness_suite_v2_jun4` pointed at `analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/`, plus a run-associated `postrun` write-up. This closes the single largest analytical gap (§9-analytical-1).
3. **Implement Diagnostic Layers 3–5** (three-way taxonomy, interpretability audit, and especially **factor→gene grounding**) for old34, writing to `analysis/scorecards/stage2_diagnostic/`. Layer 5 is required for the Q3 story and currently produces an empty file.
4. **Run controls through multi-objective Stage 2** (`full_34`, `core_only`, HVG anchors, and size-matched HVG for shortlisted panels) to enable Layer 6 gene-budget calibration and the knowledge-prior-delta claim.
5. **Reconcile Set B against Set A** — one comparison table/plot: best LOPO lift per expanded family vs the best old34 curated program, on matched Stage 1/2 settings.
6. **Optional hardening:** resurrect Plan 2 label-permutation negative controls for shortlisted panels; add an inductive (refit-per-fold) sensitivity check; add a second Stage 1 seed for stability.

---

## Appendix A — key artifact paths

```text
# Stage 0 scorecards
analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv                                   # Set A, 755 rows
analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/stage0_mrd_old34_broad_scorecard.csv  # Set B, 10,487 rows

# Multi-objective Stage 2 (Set A, root)
analysis/scorecards/stage2_discovery_full_cohort_scorecard.csv                             # 41,795 rows (41 biological panels)
analysis/scorecards/stage2_sharedness_lopo_scorecard.csv                                   # 41,795 rows
analysis/scorecards/stage2_patient_specific_scorecard.csv                                  # 376,155 rows
analysis/scorecards/stage2_patient_support_counts.csv
analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv

# Multi-objective Stage 2 (Set B, expanded)
analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/stage2_discovery_full_cohort_scorecard.csv  # 24,990 rows (294 panels)
analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/stage2_sharedness_lopo_scorecard.csv
analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1/stage2_patient_specific_scorecard.csv

# v2 exploratory diagnostics (Set A) — Layers 1–2 only
analysis/scorecards/stage2_figure3_sharedness_v2_jun4/
   v2_decision_table_discovery_lopo_patient_specific.csv
   v2_panel_stage1_stability_summary.csv, v2_panel_method_stage1_stability_summary.csv
   v2_panel_lopo_patient_summary.csv, v2_panel_lopo_patient_transfer_long.csv
   v2D_top_loading_genes_all_review_representations.csv        # EMPTY (Layer 5 unbuilt)

# Reports / run manifests
analysis/reports/postrun_human_review.md                                                   # Set A
analysis/reports/expanded_stage0_mrd_manuscript_axes_v1/postrun_human_review.md            # Set B
run_manifest.json                                                                          # Set A
analysis/reports/expanded_stage0_mrd_manuscript_axes_v1/run_manifest.json                  # Set B
```

## Appendix B — patient support (LOPO eligibility)

| patient | n_cells | n_malignant | n_normal | LOPO usable? |
| --- | ---: | ---: | ---: | --- |
| P01 | 11,860 | 45 | 11,815 | yes |
| P02 | 1,462 | 211 | 1,251 | yes (best support) |
| P03 | 2,327 | 758 | 1,569 | yes (best support) |
| P04 | 1,128 | 102 | 1,026 | yes |
| P05 | 4,359 | 8 | 4,351 | low support |
| P06 | 3,259 | 44 | 3,215 | yes |
| P07 | 2,185 | 22 | 2,163 | low support |
| P08 | 1,411 | 0 | 1,411 | normal-only (specificity only) |
| P09 | 17,126 | 375 | 16,751 | yes |
| P10 | 2,715 | 0 | 2,715 | normal-only |
| P11 | 4,024 | 0 | 4,024 | normal-only |
| P12 | 2,488 | 0 | 2,488 | normal-only |
| P13 | 5,774 | 4 | 5,770 | very low support |
