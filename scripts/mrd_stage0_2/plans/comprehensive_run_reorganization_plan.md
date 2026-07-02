# Comprehensive Run Reorganization Plan

Status: planning. Do not move or delete files until the inventory below has been checked against notebooks, shell wrappers, and experiment provenance.

## Why Reorganize

`scripts/comprehensive_run` now mixes active runners, one-off recovery scripts, pilot launchers, plan skeletons, and analysis notebooks. This makes it hard to rehydrate what was actually run, which experiment IDs matter, and which files are safe to ignore.

The reorganization should make the directory easier for a human overseer to audit without breaking reproducibility of older experiments.

## Principles

- Separate scientific changes from file-layout cleanup. Implement stage-0 functionality first, then reorganize in a separate change.
- Archive before deleting. Move old scripts into a documented legacy folder first, then delete only after references and provenance are checked.
- Preserve human-facing entry points when possible. If a command appears in notebooks or plans, keep a thin wrapper or update all references.
- Keep experiment provenance explicit. Every legacy file should state which run or incident it supported.

## Proposed File Classes

- `active`: current entry points for Plan 0, Stage 0, Stage 1, Stage 2, scorecards, and analysis.
- `reusable`: helper code that should be imported by active runners or moved into package code.
- `legacy`: one-off resume, recovery, or superseded pilot scripts retained for provenance.
- `skeleton`: planned but not active runners.
- `candidate_delete`: files with newer replacements and no remaining provenance value.

## Proposed Layout

Keep top-level files minimal:

- `README.md`: current workflow map, experiment IDs, active commands, and notebook index.
- active launch wrappers only, if needed for compatibility.

Create organized subfolders:

- `lib/`: reusable panel construction, artifact loading, scorecard utilities, and shared CLI helpers.
- `runs/`: active shell wrappers for reproducible long runs.
- `legacy/`: old resume scripts, recovery wrappers, and superseded pilots.
- `skeletons/`: Plan 2-4 exploratory runners if they are not active.
- `notebooks/`: keep notebooks here, but add a notebook index with status and conclusions.
- `plans/`: keep planning docs, including this document and the stage-0 workflow.

## Candidate Audit List

Likely active:

- `run_gene_filter_dr_grid.py`
- `run_old_geneset_pruning_metrics.py`
- `run_plan1c_supervised_latent_benchmark.py`
- `run_plan0_old_geneset_dr_suite.sh`
- `run_plan1c_supervised_latent_k40_full_all.sh`
- `run_plan1c_full_k40_all.sh`

Likely reusable:

- panel construction and leaderboard logic currently embedded in `run_old_geneset_pruning_metrics.py`
- artifact loading and feature alignment logic currently embedded in `run_plan1c_supervised_latent_benchmark.py`
- cNMF organization logic if still needed by active notebooks

Archived in `../legacy/`:

- `legacy/watch_cnmf_resume.sh`
- `legacy/run_plan1c_pilot_pca_l1.sh`

Likely legacy after documentation:

- `resume_plan0_cnmf.py`
- `resume_plan0_standard_dr.py`
- `run_plan0_resume_standard_dr_varimax.sh`
- `run_plan0_resume_standard_dr_promax.sh`

Likely skeleton or inactive:

- `run_gene_filter_dr_plan2_negative_controls.py`
- `run_gene_filter_dr_plan3_representation_first.py`
- `run_gene_filter_dr_plan4_two_stage_selection.py`

Needs explicit decision:

- `reorganize_plan0_cnmf_curated.py`
- `build_plan0_k_selection_summary.py`
- `attach_plan0_dr_cache_to_preprocessed_adata.py`

## Safe Cleanup Sequence

1. Build an inventory table with file path, class, latest known use, imported-by references, shell references, notebook references, and recommended action.
2. Update `plans/INDEX.md` so it names active, legacy, and skeleton work clearly.
3. Add `README.md` to `scripts/comprehensive_run` with the current workflow and experiment IDs.
4. Move obvious legacy scripts into `legacy/` and add `legacy/README.md`.
5. Move skeleton Plan 2-4 runners into `skeletons/` unless they become active.
6. Extract shared utilities only after the Stage 0 runner proves which abstractions are real.
7. Search and update references in `.py`, `.sh`, `.md`, and notebooks after moves.
8. Only consider deletion after a second pass confirms the file is not referenced and not useful provenance.

## Acceptance Criteria

- A human can answer "what should I run now?" from `scripts/comprehensive_run/README.md`.
- Every active workflow has one plan doc, one runner or wrapper, and one output location convention.
- Legacy scripts are still discoverable but no longer mixed with current entry points.
- The stage-0 bottom-up workflow, HVG baseline workflow, and Plan 1.C supervised benchmark are linked from `plans/INDEX.md`.
- No script is deleted without an inventory note explaining the replacement and why provenance is not needed.
