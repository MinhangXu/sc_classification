# Plan index (code-adjacent)

This folder is the **plan layer** for `sc_classification/scripts/comprehensive_run/`.

## How to read this

- **Active plans (implemented)**: Plan 0–1
- **Planned next active extension**: Plan 1.C supervised latent benchmark at fixed K
- **Later plans (skeleton runners only)**: Plan 2–4

## Files

- **Active plan (0–1)**: `active_plan0_plan1.md`
- **Later plans (2–4)**: `later_plans2_4.md`
- **In-progress engineering plan (Feb 2026)**: `plan0rotationseedsplan1stability.md` (FA rotation + multi-seed stability + Plan 1 seeding/stability sync)
- **Planned supervised benchmark (Plan 1.C)**: `plan1c_cross_patient_supervised_latent_benchmark.md`
- **Operational note**: cNMF outputs can be reorganized non-destructively for analysis using `../reorganize_plan0_cnmf_curated.py` (creates `models/cnmf_plan0/curated/` with a manifest)
- **Raw Cursor snapshots (do not edit; provenance only)**:
  - `.cursor/plans/gene-filtering-eval-plan-iter3-cnmf_a09f862f.plan.md`
  - `.cursor/plans/gene-filtering-eval-plan-iter2_e60076f2.plan.md`

## Mapping: plan → implementation

| Plan | Goal (one-liner) | Plan doc | Runner script |
|---:|---|---|---|
| 0 | Pick \(K\) per DR method; stability/diagnostics; include cNMF | `active_plan0_plan1.md` | `../run_gene_filter_dr_grid.py plan0` |
| 1 | Main 2-axis eval: preprocess × DR; no-CV then CV | `active_plan0_plan1.md` | `../run_gene_filter_dr_grid.py plan1` |
| 1.C | Fixed-K supervised latent benchmark (pooled + per-patient CV; L1/L2/EN) | `plan1c_cross_patient_supervised_latent_benchmark.md` | `../run_plan1c_supervised_latent_benchmark.py` *(planned)* |
| 2 | Negative controls / leakage checks (e.g. label permutation) | `later_plans2_4.md` | `../run_gene_filter_dr_plan2_negative_controls.py` *(skeleton)* |
| 3 | Representation-first evaluation (stability/reconstruction primary) | `later_plans2_4.md` | `../run_gene_filter_dr_plan3_representation_first.py` *(skeleton)* |
| 4 | Two-stage gene selection (unsup prefilter → supervised top-up) | `later_plans2_4.md` | `../run_gene_filter_dr_plan4_two_stage_selection.py` *(skeleton)* |
