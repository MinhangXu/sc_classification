# Later plans (2–4): protocol hardening + additional study designs

These plans are **not fully implemented yet**. This file is a curated summary of the unrefined Plan 2–4 ideas in:

- workspace copy: `.cursor/plans/gene-filtering-eval-plans_ed94c3ef.plan.md`

Skeleton runner scripts exist (see `INDEX.md`), but they currently only define the CLI + intended outputs.

## Plan 2 — negative controls / leakage checks

### Goal

Quantify how much performance can be driven by label-informed selection or leakage.

### Core idea

- Re-run the same pipeline(s) under **label permutation** within the same strata (e.g. within patient×timepoint_type).
- If you still see high AUC under permutation, you’ve found leakage/confounding.

### Implementation status

- Skeleton runner: `../run_gene_filter_dr_plan2_negative_controls.py`

## Plan 3 — representation-first evaluation

### Goal

Select gene filtering choices based primarily on representation quality/stability; treat CN.label separability as secondary.

### Candidate metrics

- held-out reconstruction / likelihood proxy (method dependent)
- factor stability under bootstrap / refits (alignment + loading stability)
- interpretability proxies (sparsity post-rotation; enrichment consistency)

### Implementation status

- Skeleton runner: `../run_gene_filter_dr_plan3_representation_first.py`

## Plan 4 — two-stage gene selection (fast broad → focused refinement)

### Goal

Avoid hard anchoring on supervision while still allowing tumor signal.

### Core idea

- Stage 1: unsupervised prefilter (e.g. HVG / variance / detection)
- Stage 2: small supervised “top-up” (weak DEGs or rescue genes)
- Evaluate with the Plan 1 metric suite.

### Implementation status

- Skeleton runner: `../run_gene_filter_dr_plan4_two_stage_selection.py`

