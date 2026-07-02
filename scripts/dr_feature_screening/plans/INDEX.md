# Plan index — dr_feature_screening (older DR method/K screening)

Plan layer for `sc_classification/scripts/dr_feature_screening/`. This lineage predates the
current Stage 0-2 study (`scripts/mrd_stage0_2/`) and runs on the Feb `20260211` HVG and April
`20260401` all_filtered experiments.

## Files

- **Active plan (0-1)**: `active_plan0_plan1.md` — Plan 0 K-sweep + Plan 1 preprocess × DR grid.
- **Supervised benchmark (Plan 1.C)**: `plan1c_cross_patient_supervised_latent_benchmark.md` — fixed-K pooled + per-patient CV benchmark.
- **Later plans (2-4, skeleton only)**: `later_plans2_4.md` — negative controls, representation-first, two-stage selection.
- **Engineering overlay**: `plan0rotationseedsplan1stability.md` — FA rotation + multi-seed stability + Plan 1 seeding sync.
- **Post-hoc validation checklist**: `posthoc_dr_validation_eval_plan.md` — chunked DR evaluation status + caveats.

## Mapping: plan → implementation

| Plan | Goal (one-liner) | Plan doc | Runner |
|---:|---|---|---|
| 0 | Pick K per DR method; stability/diagnostics; incl. cNMF | `active_plan0_plan1.md` | `../plan0_1_grid/run_gene_filter_dr_grid.py plan0` |
| 1 | Preprocess × DR grid; no-CV then CV | `active_plan0_plan1.md` | `../plan0_1_grid/run_gene_filter_dr_grid.py plan1` |
| 1.C | Fixed-K supervised latent benchmark (pooled + per-patient; L1/L2/EN) | `plan1c_cross_patient_supervised_latent_benchmark.md` | `../plan1c_supervised/run_plan1c_supervised_latent_benchmark.py` |
| 2 | Negative controls / leakage checks (label permutation) | `later_plans2_4.md` | `../skeletons/run_gene_filter_dr_plan2_negative_controls.py` *(skeleton)* |
| 3 | Representation-first evaluation | `later_plans2_4.md` | `../skeletons/run_gene_filter_dr_plan3_representation_first.py` *(skeleton)* |
| 4 | Two-stage gene selection | `later_plans2_4.md` | `../skeletons/run_gene_filter_dr_plan4_two_stage_selection.py` *(skeleton)* |
