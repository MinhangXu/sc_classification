---
id: W0
title: W0_context_provenance: repo context, provenance, portability (meta)
status: active
questions: [meta]
---

# W0 — W0_context_provenance: repo context, provenance, portability (meta)

Status as of 2026-10-08: active (P0). b60a6ee and the d387eda wip checkpoint (both 2026-10-08) are local and not pushed; the rest of the overlay is uncommitted in the working tree.

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "Inline ask: reorganize the report; review recent files in notebooks/stage0_2 and apps; agent-agnostic context engineering; per-agent dev subdirs; dated log (day, question, implementation); a central theme for the paper" (A-20261008-11)

> "Current scientific questions: central branch for publication; organize the whole repo's development around them" (A-20261008-29)

Scope: agent-agnostic context engineering, a dated devlog, per-workstream subdirs and a central paper theme (A-20261008-11, A-20261008-29). It also covers pushing the backlog, collaborator portability, EXP documentation debt, the pooled Set C decision, and clearing dirty-run provenance before any publication verdict.

Done means: the overlay (AGENTS.md, `scripts/mrd_stage0_2/research/`, `workstreams/`) is committed and pushed and `python tools/research_index.py check` passes in a fresh clone; the open user decisions (W0.2, W0.6) are recorded as decision entries; the EXP documentation debt is paid (W0.5, W0.8); and every hypothesis meant for a supported_publication verdict has evidence from a clean committed tree (W0.10).

## Builds on

- Agent canon and spine: `AGENTS.md` (rules R1-R12), `scripts/mrd_stage0_2/research/` (THEME.md, study_config.json, analyses.csv, campaigns.csv, asks/, devlog/), `scripts/mrd_stage0_2/research/legacy/` (frozen copy of the Cursor canvas).
- Tooling (W0-owned, kept at repo level): `tools/research_index.py`, `src/sc_classification/utils/paths.py`, `.env.example`, `tests/test_research_index.py`, `tests/test_paths.py`.
- Registry rows: V2 (historical Jun-4 sharedness suite), F8 (retired; never assigned). Hypothesis: H-S0-04 (pooled Set C; blocked on W0.6).
- Superseded status and planning docs: `scripts/mrd_stage0_2/plans/INDEX.md`, `scripts/mrd_stage0_2/plans/stage0_2_hypothesis_tracker.md`, `scripts/mrd_stage0_2/plans/current_experiment_status_rehydration_20260904.md`, `scripts/mrd_stage0_2/plans/experiment_20260525_stage0_2_rehydration_dossier.md`.
- Repository audit: kept in the umbrella repo at `reports/mrd_stage0_2_repository_audit/` (outer-repo path; not copied into this repo).
- W0.5 targets: EXP-relative `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/` (F1-F9; `f_artifact_ledger.csv`) and `analysis/scorecards/stage2_diagnostic/cell_count_learning_curve_v1/` (D15); notebooks `scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_patient_factor_overlap.ipynb`, `scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_donor_downsampling_sensitivity.ipynb` and `scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_donor_downsampling_sensitivity_v2.ipynb`.
- W0.6 (pooled Set C): campaign `pooled_stage0_unions_v1` in `scripts/mrd_stage0_2/research/campaigns.csv`; `scripts/mrd_stage0_2/stage0_panels/run_pooled_stage0_unions_stage0_to_stage2.sh`; `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py` (`--merge-stage2-run`; no shard resume, see AGENTS.md R2).
- W0.8 targets: `DESIGN.md`, `README_STANDARDIZED_WORKFLOW.md`, `DOWNSTREAM_ANALYSIS_GUIDE.md`, `REPO_ORGANIZATION.md`, `docs/AUTONOMOUS_COMPREHENSIVE_RUNS.md`.
- W0.9: `find_repo_root` in `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py`; `python tools/research_index.py portability` lists the legacy path debt.
- W0.12: `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py` and its default gene-set file `scripts/knowledge_driven_embedding/older_geneset/genesets_v1.gmt`.
- Devlog entries (W0): 2026-05-25_sharedness-scorecard-design, 2026-05-25_set-a-old34-broad-screen, 2026-05-26_shortlist-first-multiobjective, 2026-06-02_lopo-fullgrid-rerun, 2026-06-05_set-a-all-biological-fullgrid, 2026-06-19_diagnostic-plan-interactive-fig3a, 2026-07-02_commit-reorg-dossier, 2026-07-23_last-commit-rehydration-canvas, 2026-07-24_team-meeting-six-questions, 2026-08-21_team-review-narrative, 2026-08-25_repository-audit, 2026-09-03_pooled-unions-set-c, 2026-09-04_controls-fullgrid-status-doc, 2026-09-08_hypothesis-tracker, 2026-09-10_controls-direct-gene-cohort-inventory, 2026-10-08_team-report-devlog-pivot, 2026-10-08_research-spine-adopted.

## Backlog

- [W0.1] (P0) Land this architecture: integrate the overlay (AGENTS.md and the shims; `tools/` and `paths.py`; `scripts/mrd_stage0_2/research/`; `workstreams/`; the apps and tests), then commit it in reviewable groups after the user's go-ahead — done when: research_index.py check exits 0, pytest passes in the lab env, and a fresh clone contains AGENTS.md, research/ and workstreams/.
- [W0.2] (P0) Decide the .gitattributes '*.ipynb filter=lfs' line before any push. git-lfs is absent and every tracked notebook is raw JSON. Recommended: drop the line — done when: the .gitattributes change is committed, or a decision entry records keeping it with git-lfs installed. (blocked: user decision)
- [W0.3] (P0) Push sc_classification_gpu (b60a6ee, the d387eda wip checkpoint and the follow-up commits; squash d387eda if the user prefers) and bump the outer submodule pointer — done when: origin contains the overlay and the outer repo records the new pointer. (depends: W0.1, W0.2) (blocked: user go-ahead)
- [W0.4] (P1) The user installs the test and app dependencies in sc-classification-rapids-py311 (pip install -e .[app,dev]) — done when: python -m pytest tests -q passes there.
- [W0.5] (P1) Pay the EXP documentation debt — done when: each listed bundle has a findings md and a ledger, and both notebooks run clean.
  - Write a findings md for feedback_deep_dive_20260917 (F1-F9) and for cell_count_learning_curve_v1 (D15).
  - Add the F9 rows to f_artifact_ledger.csv.
  - Re-execute stage2_old34_patient_factor_overlap.ipynb with RECOMPUTE=False so its outputs are saved.
  - Fix the donor v1 plot_contract_rows assertion.
  - Re-run the donor v2 notebook after the 2026-09-03 merge-only (user screen).
- [W0.6] (P1) Decide pooled Set C. The Stage 2 runner has no shard resume, so the options are: after reading the code, relaunch shards 000, 002 and 003 under the same run id and merge with --merge-stage2-run; or abandon. Then record a decision entry, update campaigns.csv and declare the H-S0-04 verdict — done when: the campaigns.csv row says complete or abandoned. (blocked: user decision)
- [W0.7] (P2) Recover the verbatim question sets for 2026-07-24, 2026-08-03, 2026-08-20 and 2026-08-21 from the Cursor transcripts into research/asks/, and settle the meeting-question numbering in THEME §8 — done when: the 2026-07-24 asks file has capture: verbatim.
- [W0.8] (P2) Add HISTORICAL banners to DESIGN.md, README_STANDARDIZED_WORKFLOW.md and DOWNSTREAM_ANALYSIS_GUIDE.md. Fix REPO_ORGANIZATION.md (its .cursor/plans claim) and the references in docs/AUTONOMOUS_COMPREHENSIVE_RUNS.md — done when: no root doc presents the legacy pipeline as current.
- [W0.9] (P2) Optional: make legacy fallbacks check env vars first, e.g. patient_factor_overlap.find_repo_root honours SC_REPO_ROOT. Defaults stay unchanged, and the change goes in a devlog entry with type: refactor — done when: the portability report lists only notebooks and configs.
- [W0.10] (P1) Clear dirty-run provenance for any evidence meant to back a supported_publication verdict: re-run those D/F analyses from a clean committed tree and record run entries — done when: each targeted hypothesis has supports evidence with a hex commit and dirty_run false. (depends: W0.3)
- [W0.11] (P2) Turn on the optional CI workflow after the push — done when: the workflow passes on origin. (depends: W0.3)
- [W0.12] (P2) Document how the old34 gene sets were chosen. If any were informed by enrichment on this cohort, add a Stage 0 selection-bias caveat to THEME §6 — done when: a decision or docs entry records the provenance.

## Code in this directory

- `README.md`: this file. No code lives in this directory. W0's code sits at repo level, where every agent and collaborator finds it:
  - `tools/research_index.py`: stdlib-only generator and validator for the research spine (`check`, `write`, `new-entry`, `reserve-id`, `show`, `portability`).
  - `src/sc_classification/utils/paths.py`: SC_* path resolution and the study_config reader for new code (`python -m sc_classification.utils.paths` prints every resolved path).
  - `tests/test_research_index.py`, `tests/test_paths.py`: fixture tests; they need no experiment data.
- Screen sessions: none planned. W0.10 reruns use the owning workstream's session name, `<W#>_<topic>`.

## Conventions

- Owned artifact slugs: none. W0 writes no new EXP slugs; W0.5 only adds the missing findings and ledger rows to existing bundles, as the task states. Any new slug follows R3: it ends in `_v<N>`, is never overwritten, and its artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Whenever training n changes, retune C per condition by inner LOPO over the training patients (through W3's `regularization_contract.tune_c_inner_lopo`, wave 2) and report L2 alongside. Record `c_policy` and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal". W0.10 reruns keep the c_policy of the entry they re-run.
- Import header: tools and tests resolve locations through `src/sc_classification/utils/paths.py` (research_index.py loads it by file path). A new script inside a workstream directory uses:

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```

- Screen sessions are named `<W#>_<topic>`. Long or GPU jobs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with compute: screen-cpu or screen-gpu. Resume flags are runner-specific (R2).
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W0.n`.
- Never commit, push, stash or reset without the user's explicit go-ahead (R11). Never add absolute or home paths to overlay files (R9, lint E07).
