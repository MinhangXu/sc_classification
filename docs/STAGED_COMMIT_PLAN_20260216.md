# Staged commit plan (2026-02-16)

This is a concrete commit sequence for the current reorganization state.

## Commit 1: legacy script moves

Message:

- `chore(reorg): move remaining top-level scripts into legacy factosig buckets`

Paths:

- `scripts/legacy/factosig_pipeline/**`
- `scripts/legacy/factosig_significance/**`
- deletions/moves from previous `scripts/*.py` locations

## Commit 2: archive and notebook/script hygiene moves

Message:

- `chore(reorg): archive historical run outputs and notebook helper scripts`

Paths:

- `experiments/archive/scripts_runs_legacy_20260213/**`
- moved helper scripts under `scripts/legacy/notebook_helpers/`
- moved pipeline legacy assets from `src/sc_classification/pipeline/` and old locations

## Commit 3: repository organization docs

Message:

- `docs(repo): update organization map and legacy folder documentation`

Paths:

- `REPO_ORGANIZATION.md`
- `README.md`
- `scripts/legacy/README.md`
- `scripts/legacy/**/README.md`
- `notebooks/**/README.md`
- `experiments/README.md`

## Commit 4: operations docs for github + agents

Message:

- `docs(ops): add github workflow and autonomous run playbooks`

Paths:

- `docs/GITHUB_UPDATE_PLAYBOOK.md`
- `docs/AUTONOMOUS_COMPREHENSIVE_RUNS.md`
- `docs/STAGED_COMMIT_PLAN_20260216.md`

## Commit 5: run metadata/orchestrator scaffolding

Message:

- `feat(ops): add run manifest template and orchestrator scaffold`

Paths:

- `experiments/RUN_MANIFEST_TEMPLATE.yaml`
- `scripts/orchestrator/README.md`
- `scripts/orchestrator/spec_example.yaml`
- `scripts/orchestrator/run_goal_orchestrator.py`

