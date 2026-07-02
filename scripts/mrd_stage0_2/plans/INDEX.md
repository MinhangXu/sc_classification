# Plan index (code-adjacent)

This folder is the **plan layer** for `sc_classification/scripts/comprehensive_run/`.

## How to read this

- **Active plans (implemented)**: Plan 0–1 and Plan 1.C supervised latent benchmark at fixed K
- **Planned next active extension**: Stage 0 gene-space value-added workflow
- **Later plans (skeleton runners only)**: Plan 2–4

## Files

- **⭐ Experiment 20260525 Stage 0–2 rehydration dossier (START HERE)**: `experiment_20260525_stage0_2_rehydration_dossier.md` — master map of the two-panel-set experiment (old34 + expanded manuscript axes): pipeline artifacts, provenance of runners/plans/notebooks, gaps, and per-plot scientific framing.
- **Active plan (0–1)**: `active_plan0_plan1.md`
- **Later plans (2–4)**: `later_plans2_4.md`
- **In-progress engineering plan (Feb 2026)**: `plan0rotationseedsplan1stability.md` (FA rotation + multi-seed stability + Plan 1 seeding/stability sync)
- **Post-hoc validation checklist**: `posthoc_dr_validation_eval_plan.md` (chunked DR evaluation status + caveats)
- **Web-LLM briefing memo (data + prompt template)**: `web_llm_research_brief_dr_gene_sets.md`
- **Supervised benchmark (Plan 1.C)**: `plan1c_cross_patient_supervised_latent_benchmark.md`
- **Stage 0 gene-space value-added workflow**: `stage0_geneset_value_added_workflow.md`
- **Stage 2 comprehensive diagnostic plan (pre-shortlist)**: `stage2_comprehensive_diagnostic_plan.md`
- **Three-stage knowledge-prior malignancy design (MRD)**: `three_stages_knowledge_prior_mal_classification.md`
- **Stage 0/1/2 sharedness scorecard run spec**: `stage0_mrd_sharedness_scorecard_plan.md`
- **Directory reorganization plan**: `comprehensive_run_reorganization_plan.md`
- **Operational note**: cNMF outputs can be reorganized non-destructively for analysis using `../reorganize_plan0_cnmf_curated.py` (creates `models/cnmf_plan0/curated/` with a manifest)
- **Raw Cursor snapshots (do not edit; provenance only)**:
  - `.cursor/plans/gene-filtering-eval-plan-iter3-cnmf_a09f862f.plan.md`
  - `.cursor/plans/gene-filtering-eval-plan-iter2_e60076f2.plan.md`

## Mapping: plan → implementation

| Plan | Goal (one-liner) | Plan doc | Runner script |
|---:|---|---|---|
| 0 | Pick \(K\) per DR method; stability/diagnostics; include cNMF | `active_plan0_plan1.md` | `../run_gene_filter_dr_grid.py plan0` |
| 1 | Main 2-axis eval: preprocess × DR; no-CV then CV | `active_plan0_plan1.md` | `../run_gene_filter_dr_grid.py plan1` |
| 1.C | Fixed-K supervised latent benchmark (pooled + per-patient CV; L1/L2/EN) | `plan1c_cross_patient_supervised_latent_benchmark.md` | `../run_plan1c_supervised_latent_benchmark.py` |
| Stage 0 | Gene-space panel value-added: old genesets, huCIRA, HVG/all-filtered controls | `stage0_geneset_value_added_workflow.md` | `../run_old_geneset_pruning_metrics.py`; wrapper `../run_stage0_old_geneset_bottom_up.sh` |
| Stage 2 diagnostic | Pre-shortlist analysis of full-grid biological Stage 2 results | `stage2_comprehensive_diagnostic_plan.md` | *(notebook/script TBD)*; exploratory outputs in experiment `analysis/scorecards/stage2_figure3_sharedness_v2_jun4/` |
| 2 | Negative controls / leakage checks (e.g. label permutation) | `later_plans2_4.md` | `../run_gene_filter_dr_plan2_negative_controls.py` *(skeleton)* |
| 3 | Representation-first evaluation (stability/reconstruction primary) | `later_plans2_4.md` | `../run_gene_filter_dr_plan3_representation_first.py` *(skeleton)* |
| 4 | Two-stage gene selection (unsup prefilter → supervised top-up) | `later_plans2_4.md` | `../run_gene_filter_dr_plan4_two_stage_selection.py` *(skeleton)* |
