# Plan index (code-adjacent)

This folder is the **plan layer** for `sc_classification/scripts/comprehensive_run/`.

## How to read this

- **Active plans (implemented)**: Plan 0–1
- **Later plans (skeleton runners only)**: Plan 2–4

## Files

- **Active plan (0–1)**: `active_plan0_plan1.md`
- **Later plans (2–4)**: `later_plans2_4.md`
- **In-progress engineering plan (Feb 2026)**: `plan0rotationseedsplan1stability.md` (FA rotation + multi-seed stability + Plan 1 seeding/stability sync)
- **Raw Cursor snapshots (do not edit; provenance only)**:
  - `.cursor/plans/gene-filtering-eval-plan-iter3-cnmf_a09f862f.plan.md`
  - `.cursor/plans/gene-filtering-eval-plan-iter2_e60076f2.plan.md`

## Mapping: plan → implementation

| Plan | Goal (one-liner) | Plan doc | Runner script |
|---:|---|---|---|
| 0 | Pick \(K\) per DR method; stability/diagnostics; include cNMF | `active_plan0_plan1.md` | `../run_gene_filter_dr_grid.py plan0` |
| 1 | Main 2-axis eval: preprocess × DR; no-CV then CV | `active_plan0_plan1.md` | `../run_gene_filter_dr_grid.py plan1` |
| 2 | Negative controls / leakage checks (e.g. label permutation) | `later_plans2_4.md` | `../run_gene_filter_dr_plan2_negative_controls.py` *(skeleton)* |
| 3 | Representation-first evaluation (stability/reconstruction primary) | `later_plans2_4.md` | `../run_gene_filter_dr_plan3_representation_first.py` *(skeleton)* |
| 4 | Two-stage gene selection (unsup prefilter → supervised top-up) | `later_plans2_4.md` | `../run_gene_filter_dr_plan4_two_stage_selection.py` *(skeleton)* |
