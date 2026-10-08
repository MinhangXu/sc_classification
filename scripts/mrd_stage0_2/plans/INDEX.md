# Plan index — mrd_stage0_2 (current Stage 0-2 study)

This folder is the **plan layer** for `sc_classification/scripts/mrd_stage0_2/` (the knowledge-prior
Stage 0-2 malignancy study). Older DR method/K screening plans live under
`sc_classification/scripts/dr_feature_screening/plans/`.

## How to read this

- **START HERE for the living hypothesis / progress table (canvas)**: `/home/minhang/.cursor/projects/home-minhang-mds-project/canvases/stage0-2-hypothesis-tracker.canvas.tsx` — Stage 0/1/2 + Aim 2 rows, figures, missing pieces, and the ChatGPT inbox. Agent contract: `stage0_2_hypothesis_tracker.md`.
- **START HERE for live experiment / screen / campaign status (2026-09-04)**: `current_experiment_status_rehydration_20260904.md` — what is running, which scorecard namespaces exist, Set A/B/pooled/control-grid dates, and what another agent should not relaunch.
- **START HERE for the current old34 analysis**: `../notebooks/stage0_2/old34_inquiry_narrative_guide.md` — corrected D1–D11 reading order, metric equations, direct-gene/IFN/donor interpretation, and claim boundary.
- **D1a patient/cell deep dive (2026-09-10)**: `../notebooks/stage0_2/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` — grouped panel taxonomy, patient-level K/configuration views for allograft/apoptosis/TNFα, PCA/FA label-free variance, a mean-z one-score baseline, and post-hoc cell-type/ADT rank-error audits.
- **Presentation review**: `../notebooks/stage0_2/stage2_old34_team_review_narrative.ipynb` — curated tables and figures over completed artifacts; no model reruns.
- **Historical run map**: the rehydration dossier — master provenance snapshot compiled before the newer D1–D11 diagnostics.
- **Design / spec (implemented)**: the three-stage design, the run spec, and the gene-space value-added blueprint.
- **Original target**: the comprehensive post-run diagnostic plan (Layers 1-7); Layers 1–5 now have substantial implementations in the D1–D5 notebooks.

## Files

- **Hypothesis tracker (canvas + agent contract)**: `stage0_2_hypothesis_tracker.md` points at `stage0-2-hypothesis-tracker.canvas.tsx`. Use it to place ChatGPT-derived claims and to update rows after an implementation.
- **Live experiment status / agent rehydration (2026-09-04)**: `current_experiment_status_rehydration_20260904.md` — screen sessions, GPU shards, namespaced campaigns (Set A biological, Set A control full-grid, Set B, pooled unions, donor diagnostic), notebook inventory, and next-step constraints.
- **Current old34 inquiry narrative (START HERE for science)**: `../notebooks/stage0_2/old34_inquiry_narrative_guide.md` — current notebook order, AP proof, Spearman interpretation, direct-gene baseline, IFN-α factor attribution, corrected donor-factor stability, exact patient composition, and publication claim gate.
- **D1a descriptive deep dives (2026-09-10)**: `../notebooks/stage0_2/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` — reverses D1a/b aggregation for three focus panels and links selected-DR rankings to predicted cell type, collection timing, QC fields, and 170 CITE-seq ADTs. Rank-defined error groups are descriptive, not operational calls; detailed treatment regimens are unavailable.
- **Curated old34 team-review notebook**: `../notebooks/stage0_2/stage2_old34_team_review_narrative.ipynb` — presentation-ready review of D6–D11 outputs; reads completed CSV/PNG artifacts only.
- **Experiment 20260525 Stage 0-2 rehydration dossier (historical master map)**: `experiment_20260525_stage0_2_rehydration_dossier.md` — the two-panel-set run (old34 + expanded manuscript axes): pipeline artifacts, provenance of runners/plans/notebooks, gaps, and per-plot scientific framing as of 2026-07-02.
- **Three-stage knowledge-prior design (MRD)**: `three_stages_knowledge_prior_mal_classification.md` — the Stage 0/1/2 concept + the three PI questions.
- **Stage 0/1/2 sharedness scorecard run spec**: `stage0_mrd_sharedness_scorecard_plan.md` — run spec, scorecard schema, artifact Q&A.
- **Stage 2 comprehensive diagnostic plan (target)**: `stage2_comprehensive_diagnostic_plan.md` — the intended 7-layer post-run diagnostic; mostly the "what to build next" doc.
- **Stage 2 figure catalog + statistical rigor (v2 Jun 4 suite)**: `stage2_figure_catalog_and_statistical_rigor_v2_jun4.md` — plot-by-plot four-element framing of the old34 sharedness suite (V2A–V2F6), the two cross-patient aggregations (pooled-cell vs patient-summary), the null-model / uncertainty / AUROC design that should gate manuscript plots, and the interactive-tool brainstorm. Output of dossier §11.1.
- **Stage 0 gene-space value-added workflow**: `stage0_geneset_value_added_workflow.md` — bottom-up value-added blueprint that the expanded manuscript-axes panel set (Set B) implements; also carries the not-yet-done roadmap (huCIRA branch, relapse 4-class).
- **Directory reorganization plan (executed)**: `comprehensive_run_reorganization_plan.md` — record of the 2026-07-02 split into `mrd_stage0_2/` + `dr_feature_screening/`.

## Mapping: plan → implementation

| Plan | Goal (one-liner) | Plan doc | Runner |
|---|---|---|---|
| Stage 0/1 | Panel construction → DR representation | `three_stages_...md` | `../stage0_panels/run_stage0_mrd_old34_broad_screen.py`; expanded wrapper `../stage0_panels/run_expanded_stage0_genesets_stage0_to_stage2.sh` |
| Stage 2 | Multi-objective supervised probes (discovery / LOPO / patient-specific) | `stage0_mrd_sharedness_scorecard_plan.md` | `../stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py` |
| Value-added | Which gene space adds malignancy signal (bottom-up + budget-matched controls) | `stage0_geneset_value_added_workflow.md` | expanded panel set (Set B) via the expanded wrapper |
| Diagnostic | 7-layer post-run analysis (target) | `stage2_comprehensive_diagnostic_plan.md` | notebooks in `../notebooks/stage0_2/` (Layers 1-2 partial) |
