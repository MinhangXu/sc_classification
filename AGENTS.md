# AGENTS.md: sc_classification

> Canonical for every coding agent and contributor. CLAUDE.md, GEMINI.md, .github/copilot-instructions.md, .cursor/rules and the frozen Cursor canvas only point here. If another doc disagrees, this file wins: fix the other doc in the same task.

## 1. Repo in brief
- `src/sc_classification/`: the library (src layout, installed with `pip install -e .`; importable, not run directly).
- `scripts/` holds the studies: `mrd_stage0_2/` is ACTIVE (knowledge-prior Stage 0-2 malignant-vs-normal probe on the post-transplant MDS MRD cohort); `dr_feature_screening/` historical; `Relapse_MRD_DR_Classification/` paused; `knowledge_driven_embedding/` gene-set builders; `orchestrator/`; `legacy/` provenance only.
- `tools/research_index.py` (research-spine generator and validator, stdlib only) and `tests/`.
- `experiments/` is gitignored. The canonical experiment is about 199 GB. Never commit it.

## 2. Start of every task (stop when you have enough)
`research/`, `workstreams/` and `notebooks/` below are under `scripts/mrd_stage0_2/`.

0. Run `git status --short` and `python tools/research_index.py check`. If check fails before you have touched anything, tell the user. Do not silently "fix" unrelated files.
1. Read this file.
2. Read `scripts/mrd_stage0_2/research/STATUS.md` (generated): cohort, workstreams, open tasks, hypotheses × current verdict, registry, campaigns, unanswered asks.
3. Read `research/THEME.md` §6 (the canonical claim list, i.e. the claim boundary) and the § for your question (§2 Q1, §3 Q2, §4 Q3, §5 QI). Reads 1-3 give you the rules, the current state and the claim boundary.
4. Read your workstream README (`workstreams/<W#_slug>/README.md`). If none fits, ask the user. Never create a W# silently.
5. As needed: `python tools/research_index.py show <ID>` for every ID in the request; `research/campaigns.csv`, always before any compute; `research/CLAIMS.md` for the numbers; `notebooks/stage0_2/old34_inquiry_narrative_guide.md`, only for method mechanics.
6. New questions from the user, PI, a teammate or a meeting: first append them verbatim to a new `research/asks/<date>_<source>.md`, with A-IDs and routes.
7. Before computing: long or GPU job → R1 (screen command, stop); new artifacts → a NEW `_v<N>` slug (R3) under a reserved planned row or `research_index.py reserve-id`; training n changes → plan C retuning (R5); never move files (R10) or add absolute paths (R9).

Docs with a HISTORICAL banner are not current.

## 3. Hard rules R1-R12

R1 Long jobs. Never start any of the following from an agent shell, including background or detached starts:
- Stage 0/1/2 campaigns;
- DR/FactoSig grids;
- GPU Stage 2 shards;
- anything expected to take more than about 5 minutes.
Instead, give the user:
- a screen session name (<W#>_<topic>);
- the full command;
- the expected outputs;
- how to verify completion.
The user runs `screen -S <name>`, then the command, then detaches with Ctrl-a d. Record this in a devlog entry with compute: screen-cpu|screen-gpu and the literal `screen -S` line. CPU-minute work, such as the 92-gene IFN-alpha refits, is fine in the session.

R2 Resume is runner-specific; never assume a --resume flag.
- run_stage2_old34_donor_downsampling_sensitivity.py: --resume and --merge-only.
- run_stage0_mrd_old34_broad_screen.py: skips existing outputs unless --rerun.
- run_stage2_mrd_multiobjective_scorecard.py: no resume. Shards use --shard-count/--shard-index and are merged with --merge-stage2-run. Relaunching means a new --stage2-output-branch, or a per-shard relaunch under the same run id, and only after reading the code.
- run_pooled_stage0_unions_stage0_to_stage2.sh: has no --resume (it exits with 'Unknown argument'). Never pass --rerun-stage0.
- research/campaigns.csv resume_command is authoritative. A value of 'none' means no safe resume exists: ask the user.
- New runners you write must implement --resume:
  - skip finished units;
  - treat empty or corrupt files as unfinished;
  - on a merge failure, write merge_failed and recover with --merge-only.
- Never pass --rerun, --rerun-stage0 or other destructive flags unless the user asked.
- --save-lopo-coefficients is invalid. Saving is the default; opt out with --no-save-lopo-coefficients.

R3 Protected outputs.
- Never write the root Set A scorecards listed in study_config.json protected_paths.
- New grids use --stage2-output-branch or --branch-name namespaces.
- New diagnostics go to EXP/analysis/{scorecards,figures,cache}/stage2_diagnostic/<slug>_v<N>/, with *_findings.md, *_manifest.json and *_artifact_ledger.csv.
- New Stage 1 bases go to EXP/stage1_dr_variants/<slug>_v<N>/, never stage1_dr/.
- Never overwrite an existing slug; bump _vN.
- Combine namespaces only in analysis code.

R4 Campaigns.
- Read research/campaigns.csv before any compute and obey relaunch_policy (never | resume_only | ask_user).
- Add a planned row (through the integrator, or your own edit outside wave 2) before handing the user a new GPU campaign.
- No new 8-GPU job until the pooled Set C decision (W0.6) is made, unless the user overrides.

R5 Regularization.
- sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train).
- Whenever training n changes, retune C per condition by inner LOPO over the training patients, and report L2 alongside. This covers subsets, subsampling, inductive folds, reweighting, donor downsampling, learning curves and new patients.
- Record c_policy and training_n_changed in the devlog.
- An all-zero L1 probe scores AP = π by construction. Never read it as 'no signal'.

R6 Claim-gate language.
- Everything is 'descriptive' unless THEME §6 and a supported_publication verdict say otherwise.
- No causal claims without experimental intervention.
- Say 'constant-score no-skill AP = π'.
- Quote pooled AP, median lift and P20 lift together; never 38/41 alone.
- For any loading, state the sign orientation and the penalty (L1/L2/EN), and name factors from both poles.
- State non-identifiability for single-patient peaks.
- Label Stage 1 as transductive. LOPO is label transfer, not inductive deployment.
- Describe P02/P03/P04 as 'highest-prevalence', not 'most malignant cells'.
- Cell-level lift is not recurrence-risk improvement.
- Keep CRC clinical 'stage 0-2' separate from pipeline Stage 0/1/2.
- The canonical claim list is THEME §6 (link it; do not copy it).

R7 Metric and support.
- lift = held-out AUPRC − the held-out patient's malignant prevalence π. Normalized lift (AP−π)/(1−π) is secondary.
- Use Option B patient summaries.
- Supported patient = at least 10 malignant cells and both classes (study_config.json: P01, P02, P03, P04, P06, P07, P09).
- D14's 20-cell malignant stratum is an analysis parameter, not the support rule.

R8 Data. Never load the 14.2 GB raw AnnData. Use the cohort cache (study_config cohort_cache_relpath) or the cell metadata CSV.

R9 Paths.
- No absolute paths in new files.
- New code resolves locations through sc_classification.utils.paths (SC_REPO_ROOT, SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID, SC_DATA_ROOT, SC_COHORT_H5AD).
- Paths in docs, devlog and registry are relative to the experiment dir.

R10 Overlay, never move.
- Existing code, notebooks and experiment artifacts stay where they are: notebooks sys.path-import stage2_supervised/, tests reference apps/, and experiment configs record script paths.
- New code goes in your workstream dir; apps go in apps/.
- Edits to shared code must be additive (a new function or flag, defaults unchanged) and named in your devlog entry.

R11 Git.
- Never commit, push, amend or rebase without the user's explicit go-ahead. Propose a message instead.
- Never commit experiments/, *.h5ad or data.
- Ask the user to commit runner changes before launching a campaign, so configs record a real hash.

R12 Evidence.
- Numbers live only in devlog Results. THEME, hypotheses.md, analyses.csv notes and READMEs cite IDs.
- If EXP or an artifact is missing, report from devlog entries and CLAIMS.md. Never recompute a campaign, and never invent a number to fill a gap.
- $HOME/.cursor, chat transcripts and ChatGPT threads are never canonical. Copy what matters into research/asks/ or a devlog entry.

## 4. Study facts
- EXP = `$SC_EXPERIMENT_ROOT/<canonical_experiment_id>` from `scripts/mrd_stage0_2/research/study_config.json`; new code gets it from `sc_classification.utils.paths.experiment_dir()`.
- Stage 0: gene-space panels; root Set A = 41 biological panels.
- Stage 1: pca/fa/factosig/factosig_promax at K 5, 10, 20, 40, seed 42, fit transductively on all eligible cells.
- Stage 2: L1/L2/EN logistic paths in three views: discovery, sharedness LOPO and patient-specific.
- Cohort counts, supported patients and per-patient malignant/total counts live only in study_config.json; STATUS.md renders them at the top. They are not restated here.

## 5. IDs and vocabulary
- Question tags: Q1 shared vs patient-specific; Q2 strength of evidence; Q3 what DR adds vs direct genes; QI interpretability lens (= PI-Q3); meta.
- Analysis IDs (`research/analyses.csv`): D1-D15 (+ D1c-f, D3b), D1DD1-13 and F1-F9 are frozen; F8 was never assigned; the D1DD and F series are closed. New analyses use their reserved planned row (D15b, D16-D25) or `python tools/research_index.py reserve-id --workstream W# --title '...'`.
- Other IDs: H-* hypotheses (`research/hypotheses.md`); ALT-* reviewer alternatives (`research/alternatives.csv`); W# workstreams and W#.n tasks; A-YYYYMMDD-NN asks; devlog ids = filename stems. Crosswalk from older labels: THEME §8.

## 6. Where new work goes
- Code and notebooks: `scripts/mrd_stage0_2/workstreams/<W#_slug>/` as `run_<topic>.py`, `run_<topic>.sh` (screen jobs) and `<topic>_<purpose>_<YYYYMMDD>.ipynb`.
- Dash apps: `scripts/mrd_stage0_2/apps/` (owned by W7). Tests: `tests/test_<module>.py`, loading the module by file path with importlib, as `tests/test_old34_d1_atlas_explorer.py` does.
- Import header for a new workstream script (a new notebook instead finds REPO by walking up from `Path.cwd()` to `pyproject.toml`, then calls `add_study_module_paths()`):
```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```
- A module used by 2 or more workstreams is promoted to `stage2_supervised/` only through a devlog entry with type: refactor. Artifacts follow R3.

## 7. End of every task: at most 3 small edits, then 1 command
1. Create the devlog entry: `python tools/research_index.py new-entry <slug> --workstream W# [--type analysis] [--analyses D16]`. Fill all 8 sections: the verbatim question with its ask ID; estimand and null, incl. c_policy; code paths and the exact rerun or screen command; EXP-relative artifacts; headline numbers with [H-ID role] tags on evidence bullets; both claim-gate lines; next steps citing W#.n. Set closes and advances; set verdicts and alts only when they change. Leave commit: pending.
2. Edit only what changed:
   - `research/analyses.csv`: your row's status or notes;
   - your workstream README: new backlog lines;
   - `research/campaigns.csv`: a campaign was planned, launched, finished, stalled or abandoned;
   - `research/hypotheses.md`: only when a statement, alternatives list, blocking list or figures change. Never put verdicts there.
3. Run `python tools/research_index.py write && python tools/research_index.py check` and get exit 0. If code changed, also run `python -m pytest tests -q`.
- THEME.md changes only at milestones (a verdict changes, a claim crosses §6, the user asks for paper text); bump last_reviewed when it does. STATUS.md lists the entries waiting for review.
- Never hand-edit STATUS.md, CLAIMS.md or devlog/INDEX.md; on a merge conflict in any of them, rerun write. Never edit the Results of an old entry: write a new one with supersedes:.
- Long run handed to the user: a type: run entry with status: partial and compute: screen-*; on completion, a status: done entry that supersedes it. Correcting a claim: a new entry with supersedes: [old id].
- Final message to the user: headline numbers and claim level; the devlog path; the files changed; exact screen commands and how to verify them; a proposed commit message, ending with your tool's attribution trailer if your environment requires one; open questions and suspected stale docs. Do not commit.

## 8. Environment and commands
- Lab server: conda env `sc-classification-rapids-py311`. RAPIDS imports need `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python`.
- Collaborators: `conda env create -f env-rapids-factosig.yml` (it pins python 3.10; the lab env is 3.11; any 3.10+ env also works), then `pip install -e '.[app,dev]'`.
- Env vars (SC_REPO_ROOT, SC_EXPERIMENT_ROOT, SC_EXPERIMENT_ID, SC_DATA_ROOT, SC_COHORT_H5AD) are listed in `.env.example`. Path check: `python -m sc_classification.utils.paths`.
- Tests: `python -m pytest tests -q`. App: `python scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py [--check | --export-bundle DIR | --data-dir DIR]`.
- Screen template: `screen -S <W#>_<topic>`, then `bash <runner.sh> ...`, then detach with Ctrl-a d (reattach with `screen -r <W#>_<topic>`).
- Naming: scripts `run_<study_or_module>.py`; notebooks `<topic>_<purpose>_<YYYYMMDD>.ipynb`; experiment dirs `YYYYMMDD_HHMMSS_<slug>_<hash>`.

## 9. If you do not have the experiment dir (EXP), or you are on another machine
- Where EXP lives: on the lab GPU server under `sc_classification/experiments/`. Ask the repo owner (Minhang Xu) for access, then point SC_EXPERIMENT_ROOT at the directory that holds it.
- What works without EXP: devlog Results and `research/CLAIMS.md`, which carry every headline number; notebooks committed with outputs; the D1 atlas app from a bundle made with `--export-bundle` and served with `--data-dir`; `python tools/research_index.py check` without `--strict-artifacts`.
- Never recompute a campaign or invent a number to fill a missing artifact. Report from devlog entries instead, and say so in your entry.
- factosig dependency: a separate repo, `git@github.com:MinhangXu/factosig.git`, which may be private (ask the owner). Clone it as the sibling `../factosig` (the umbrella repo carries it as a submodule there); `env-rapids-factosig.yml` installs `-e ../factosig`. Without it, FactoSig and factosig_promax imports and fits fail; PCA/FA, Stage 2 on stored scores, and analyses that read stored CSVs still work.
- Legacy path debt: older notebooks set REPO to the lab-server repo path in their first cell, and some modules fall back to it (e.g. `patient_factor_overlap.find_repo_root`). Workarounds: pass the runners' `--experiment-dir`; edit a notebook's REPO cell locally and do not commit it; use SC_REPO_ROOT once W0.9 lands; or make a symlink. `python tools/research_index.py portability` lists every affected file.

## 10. Shims and legacy notes
- Pointers only (edit this file, not them): CLAUDE.md, GEMINI.md, `.github/copilot-instructions.md`, `.cursor/rules/agents.mdc`.
- The Cursor canvas is a frozen, non-canonical view; its verbatim copy is in `scripts/mrd_stage0_2/research/legacy/`. The repository audit lives in the umbrella repo at `reports/mrd_stage0_2_repository_audit/`, not in this repo.
- `scripts/mrd_stage0_2/plans/` is design history; `plans/INDEX.md` keeps the plan → runner table.
- StandardizedPipeline and ExperimentManager belong to the legacy pipeline, which the mrd_stage0_2 runners do not import. DESIGN.md, README_STANDARDIZED_WORKFLOW.md and DOWNSTREAM_ANALYSIS_GUIDE.md describe only that legacy pipeline.
- Tracked notebooks are plain JSON, but `.gitattributes` still routes `*.ipynb` through LFS; dropping that line is open user decision W0.2. Do not edit `.gitattributes` without the user.
