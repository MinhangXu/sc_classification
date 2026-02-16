# Autonomous comprehensive runs (agent-driven)

This guide describes how to run long studies where you define a goal and let an agent execute with checkpoints, validation, and logging.

## Operating model

Define runs as a loop:

1. **Plan**: declare objective and constraints.
2. **Execute**: run one bounded stage (Plan 0, Plan 1A, Plan 1B, etc.).
3. **Validate**: check metrics and artifact integrity.
4. **Decide**: continue, adjust, or stop based on criteria.
5. **Log**: persist status and next action.

Agents should execute one loop iteration at a time, not an unbounded multi-day run with no checkpoint.

## Goal spec template

For each autonomous run, write a compact spec file (for example in `scripts/comprehensive_run/plans/active_run_specs/`):

- objective: scientific question being tested
- data scope: cohort filters and timepoints
- preprocess grid: gene filtering options
- DR grid: methods and K mapping
- classifier config: CV settings
- stop criteria: what counts as success/failure
- budget: max runtime, memory, retries

## Checkpoint stages

- **Stage A (smoke test)**: tiny data slice and 1-2 seeds.
- **Stage B (pilot)**: reduced grid with full logging.
- **Stage C (full run)**: full grid after pilot passes.

An agent should promote A -> B -> C only when validation gates pass.

## Validation gates (examples)

- files exist:
  - `analysis/plan0/k_selection_summary.csv`
  - `analysis/preprocess_cache/<tag>/adata_with_dr.h5ad`
- metric sanity:
  - no degenerate embeddings
  - no all-constant predictions
  - expected patient-level output coverage
- stability sanity:
  - replicate summary produced for each `(method, K)` in scope

## Logging and observability

Use append-only logs per run directory:

- `logs/agent_run.log`: stage transitions, command, elapsed time, exit status
- `logs/validation.log`: gate checks and pass/fail reasons
- `logs/decision.log`: continue/stop/adjust with rationale

Write a short `RUN_STATE.json` after each stage:

- stage
- status
- last_completed_step
- next_step
- key metrics snapshot
- timestamp

This makes interruption/recovery easy for both humans and agents.

## Failure policy

- Retry transient failures (I/O, environment) with bounded attempts.
- On repeated failure, automatically:
  - mark stage `blocked`
  - save traceback/context
  - propose minimal next action (for example adjust K list or skip a method)

## Practical integration with current repo

- Keep active runners in `scripts/comprehensive_run/`.
- Keep stage orchestration logic in `scripts/orchestrator/`.
- Keep human-readable plans in `scripts/comprehensive_run/plans/`.
- Keep raw planning history in `.cursor/plans/`.
- Store generated outputs in `experiments/` with per-run manifests.

## Current scaffold + next increment

Initial scaffold now lives at:

- `scripts/orchestrator/run_goal_orchestrator.py`
- `scripts/orchestrator/spec_example.yaml`

Next increment:

- wire real stage command execution
- add validation gate hooks
- support explicit resume policy (`--resume-from-state`)

