# Plan index — mrd_stage0_2 (current Stage 0-2 study)

This folder is the **plan layer** for `sc_classification/scripts/mrd_stage0_2/` (the knowledge-prior
Stage 0-2 malignancy study). Older DR method/K screening plans live under
`sc_classification/scripts/dr_feature_screening/plans/`.

## How to read this

- **START HERE**: the rehydration dossier — master map of the run.
- **Design / spec (implemented)**: the three-stage design, the run spec, and the gene-space value-added blueprint.
- **Target (mostly unbuilt)**: the comprehensive post-run diagnostic plan (Layers 1-7).

## Files

- **⭐ Experiment 20260525 Stage 0-2 rehydration dossier (START HERE)**: `experiment_20260525_stage0_2_rehydration_dossier.md` — the two-panel-set run (old34 + expanded manuscript axes): pipeline artifacts, provenance of runners/plans/notebooks, gaps, and per-plot scientific framing.
- **Three-stage knowledge-prior design (MRD)**: `three_stages_knowledge_prior_mal_classification.md` — the Stage 0/1/2 concept + the three PI questions.
- **Stage 0/1/2 sharedness scorecard run spec**: `stage0_mrd_sharedness_scorecard_plan.md` — run spec, scorecard schema, artifact Q&A.
- **Stage 2 comprehensive diagnostic plan (target)**: `stage2_comprehensive_diagnostic_plan.md` — the intended 7-layer post-run diagnostic; mostly the "what to build next" doc.
- **Stage 0 gene-space value-added workflow**: `stage0_geneset_value_added_workflow.md` — bottom-up value-added blueprint that the expanded manuscript-axes panel set (Set B) implements; also carries the not-yet-done roadmap (huCIRA branch, relapse 4-class).
- **Directory reorganization plan (executed)**: `comprehensive_run_reorganization_plan.md` — record of the 2026-07-02 split into `mrd_stage0_2/` + `dr_feature_screening/`.

## Mapping: plan → implementation

| Plan | Goal (one-liner) | Plan doc | Runner |
|---|---|---|---|
| Stage 0/1 | Panel construction → DR representation | `three_stages_...md` | `../stage0_panels/run_stage0_mrd_old34_broad_screen.py`; expanded wrapper `../stage0_panels/run_expanded_stage0_genesets_stage0_to_stage2.sh` |
| Stage 2 | Multi-objective supervised probes (discovery / LOPO / patient-specific) | `stage0_mrd_sharedness_scorecard_plan.md` | `../stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py` |
| Value-added | Which gene space adds malignancy signal (bottom-up + budget-matched controls) | `stage0_geneset_value_added_workflow.md` | expanded panel set (Set B) via the expanded wrapper |
| Diagnostic | 7-layer post-run analysis (target) | `stage2_comprehensive_diagnostic_plan.md` | notebooks in `../notebooks/stage0_2/` (Layers 1-2 partial) |
