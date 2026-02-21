# Orchestrator (agent-facing)

This folder is the right home for long-running, goal-driven orchestration logic.

## Why separate from `comprehensive_run/`

- `comprehensive_run/` contains domain runners (`plan0`, `plan1`, ...)
- `orchestrator/` coordinates stages, validation gates, resume, and logging

## Entry point

- `run_goal_orchestrator.py`: stage execution with retries, validation gates, and resume support

## Intended usage

1. Write a run spec (`yaml` or `json`) with objective, commands, and gates.
2. Launch orchestrator with the spec.
3. Let it execute bounded stages and emit `RUN_STATE.json` and logs.
4. Resume safely from last completed stage.

## Key CLI flags

- `--dry-run`: plan/log only, no command execution.
- `--resume-from-state`: continue from existing `RUN_STATE.json`.
- `--state`: choose where state + logs are written.

## Spec notes

- `working_dir` is resolved relative to the spec file location when not absolute.
- Validation types currently supported:
  - `file_exists`
  - `glob_exists`

