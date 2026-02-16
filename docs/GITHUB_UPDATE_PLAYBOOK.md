# GitHub update playbook

Use this flow to keep the public repo current while avoiding accidental data bloat.

## 1) Pre-flight checks

- Confirm large generated outputs stay out of commits:
  - `experiments/`
  - `experiments/archive/`
  - large notebook outputs if not needed for reproducibility
- Run:
  - `git status --short`
  - `git diff --stat`
- Validate key docs are updated when structure changes:
  - `README.md`
  - `REPO_ORGANIZATION.md`
  - relevant folder `README.md` files

## 2) Branching strategy

- Keep `main` stable and reviewable.
- Create focused branches per theme:
  - `reorg/layout-cleanup-202602`
  - `feature/comprehensive-run-plan1-cv`
  - `docs/agent-run-playbook`
- Keep PRs narrow: structure/docs changes separated from algorithmic changes when possible.

## 3) Commit strategy

- Commit moves separately from code edits.
- Suggested sequence:
  1. `chore(reorg): move legacy outputs/scripts to archive layout`
  2. `docs(repo): add organization and run operation guides`
  3. `feat(dr): <method or experiment change>`

## 4) PR checklist

- Explain *why* in the PR summary (scientific and operational rationale).
- Include test/validation notes:
  - command used
  - dataset slice
  - expected artifacts generated
- Note any moved paths explicitly to help reviewers.

## 5) Release hygiene for research code

- Keep one canonical run entry point for active studies (`scripts/comprehensive_run/`).
- Maintain one manifest per experiment output directory as `run_manifest.yaml`.
  - Start from `experiments/RUN_MANIFEST_TEMPLATE.yaml`.
  - Include required sections:
    - `run`: objective, command, launched/finished timestamps, status
    - `code`: git commit, branch, repo path
    - `data`: input dataset path/hash and filters
    - `params`: plan, preprocess/dr settings, k config, seeds, CV settings
    - `artifacts`: experiment directory and key outputs
    - `notes`: summary and follow-up
- This manifest is the contract between experiment outputs and orchestrator state/logs:
  - run-level metadata lives in `run_manifest.yaml` (per experiment)
  - orchestration progress lives in `RUN_STATE.json` and `logs/agent_run.log`
- Prefer reproducible reruns over committing bulky derived artifacts.

## 6) Staged commit plan (current reorg wave)

Use this exact order so path moves remain reviewable:

1. `chore(reorg): move scripts into legacy taxonomy`
   - includes moves under `scripts/legacy/`
   - includes moved notebook helper `.py` files
2. `chore(reorg): archive legacy experiment outputs`
   - includes moves to `experiments/archive/*`
3. `docs(repo): add organization and legacy folder READMEs`
   - `REPO_ORGANIZATION.md`, `scripts/legacy/**/README.md`, notebook/experiment docs
4. `docs(ops): add github + autonomous run playbooks`
   - `docs/GITHUB_UPDATE_PLAYBOOK.md`
   - `docs/AUTONOMOUS_COMPREHENSIVE_RUNS.md`
5. `feat(ops): add run manifest template and orchestrator scaffold`
   - `experiments/RUN_MANIFEST_TEMPLATE.yaml`
   - `scripts/orchestrator/README.md`
   - `scripts/orchestrator/spec_example.yaml`
   - `scripts/orchestrator/run_goal_orchestrator.py`

Concrete file-by-file version is tracked in:

- `docs/STAGED_COMMIT_PLAN_20260216.md`

## 7) Agent-executable checklist

Yes, you can assign an agent to execute this plan. Give the agent:

- target branch name
- commit sequence (from section 6)
- include/exclude paths for each commit
- final PR title/body template

Recommended instruction block:

1. create branch `<branch-name>`
2. stage only files for commit 1, commit with message 1
3. repeat for commits 2..N
4. run `git status --short` after each commit
5. open PR with summary + moved-path table + validation notes


