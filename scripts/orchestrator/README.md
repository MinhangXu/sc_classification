# Orchestrator (agent-facing)

This folder is the right home for long-running, goal-driven orchestration logic.

## Why separate from `comprehensive_run/`

- `comprehensive_run/` contains domain runners (`plan0`, `plan1`, ...)
- `orchestrator/` coordinates stages, validation gates, resume, and logging

## Initial entry point

- `run_goal_orchestrator.py`: minimal scaffold for stage-based execution

## Intended usage

1. Write a run spec (`yaml` or `json`) with objective, commands, and gates.
2. Launch orchestrator with the spec.
3. Let it execute bounded stages and emit `RUN_STATE.json` and logs.
4. Resume safely from last completed stage.

