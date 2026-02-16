# Repository Organization (Feb 2026)

This document is the working contract for keeping `sc_classification` focused on reproducible single-cell classification and DR studies.

## Canonical layout

- `src/sc_classification/`: reusable library code only.
- `scripts/comprehensive_run/`: active experiment runners for plan-driven studies.
- `scripts/orchestrator/`: stage orchestration and run-state management scaffolds.
- `scripts/legacy/`: historical scripts kept for provenance and reproducibility.
- `notebooks/`: diagnosis and analysis notebooks.
- `experiments/`: generated run artifacts and outputs.
- `.cursor/plans/`: raw Cursor plan snapshots (provenance only; historical trace of how plans evolved).

## Plan files policy

Use one curated source in:

- `scripts/comprehensive_run/plans/INDEX.md`
- `scripts/comprehensive_run/plans/active_plan0_plan1.md`
- `scripts/comprehensive_run/plans/later_plans2_4.md`

Keep raw snapshots only in:

- `.cursor/plans/*.plan.md`

Do not duplicate raw `*.plan.md` files into `scripts/comprehensive_run/plans/`.

## Active workflows

- Current: `scripts/comprehensive_run/run_gene_filter_dr_grid.py` (`plan0`, `plan1`).
- Historical but important context:
  - `scripts/legacy/dr_suite/run_dr_suite.py`
  - `scripts/legacy/replearn_2025/aug19_rep_learn_supervised_filtering.py`
  - `scripts/legacy/replearn_2025/aug21_rep_learning_with_cv.py`
  - `notebooks/dr_suite_analyses/`
  - `notebooks/experiments_eval_YH_prep_Sept2025/`

## Cleanup targets (safe next steps)

1. Keep `src/sc_classification/` code-only:
   - moved notebooks from `src/sc_classification/pipeline/` to `notebooks/` topic subdirs on 2026-02-13
   - moved result folders under `src/sc_classification/pipeline/results_*` into `experiments/archive/pipeline_results_legacy_20260213/` on 2026-02-13
2. Create notebook topic buckets:
   - partial progress: `notebooks/dr/` and `notebooks/multivi/` created on 2026-02-13
   - next: complete `notebooks/factosig/` and `notebooks/archive/` consolidation
3. Add a lightweight run manifest per experiment under `experiments/`:
   - run command, git commit, input dataset hash, key params (`timepoint_filter`, `tech_filter`, DR method, K).
4. Keep one active planning lane:
   - run-facing specs in `scripts/comprehensive_run/plans/`
   - all draft ideation snapshots in `.cursor/plans/`.
5. Consolidate legacy script lineage:
   - moved old one-off runners to `scripts/legacy/` on 2026-02-13
   - moved historical run outputs from `scripts/experiments/` to `experiments/archive/scripts_runs_legacy_20260213/` on 2026-02-13
   - moved remaining top-level FactoSig helper scripts into `scripts/legacy/factosig_pipeline/` and `scripts/legacy/factosig_significance/` on 2026-02-16

## Naming conventions going forward

- Scripts: `run_<study_or_module>.py` for executable entry points.
- Notebooks: `<topic>_<purpose>_<YYYYMMDD>.ipynb`.
- Experiment directories: existing timestamp format is good:
  - `YYYYMMDD_HHMMSS_<slug>_<hash>`.

