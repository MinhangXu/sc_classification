# sc_classification

This repository holds the code and the research record for one active study. The goal is to tell cancer
cells from healthy cells, one cell at a time, in bone-marrow samples from patients with myelodysplastic
syndrome (MDS) after a stem-cell transplant, where only a few cancer cells remain (measurable residual
disease, MRD). The method starts from gene sets with known biology, compresses each set into a few summary
programs, then trains simple penalised models and tests each one on a patient it never saw. The questions are
whether a cancer-cell signal is shared across patients, how strong the evidence is, and what the compression
adds over using the genes directly. All results so far are descriptive, and the cell data are not in git.

## People and roles

- Minhang Xu: repository owner and study lead ("MX" in the core report; "the user" in devlog entries and
  asks). Decides which runs start and what is committed; ask him for data access, FactoSig access and the
  long reports.
- The PI: posed the study's three original questions (asks A-20260604-01..03, `research/asks/2026-06-04_pi-questions.md`).
- Teammates named in asks and reports: Yu-Hsiang (provides the new MRD patients, workstream W8), Pavan
  (asked whether the protein data are integrated; core report "protein" page), Ankit (semi-supervised and FA
  questions). Asks from them are tagged `teammate:<Name>`.
- "The lead agent" in devlog entries is the coding agent that ran a session; `implemented_by` names the tool.

## Start here in 10 minutes

1. Read the core report:
   `scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/report_core.pdf` (12 pages). It is the
   one PDF the repository carries (a `.gitignore` exception); it reaches the remote with the next push (task
   W0.3), so until then it exists only in the lab working tree. Page 1 answers every question in one table;
   each later page is one question with its evidence and one figure. The cohort page has a "Key terms" box.
2. Read `scripts/mrd_stage0_2/research/STATUS.md` for the live status. For a first read, §0 (cohort), §3
   (hypotheses and their current verdicts) and §6 (questions not yet answered) are enough; the rest is for
   maintainers. It is generated, so never edit it by hand.
3. Read `scripts/mrd_stage0_2/research/THEME.md` for the scientific questions (Q1 shared vs patient-specific
   signal, Q2 strength of evidence, Q3 what the compression adds) and, in §6, the claim boundary: what we may
   and may not say. STATUS also tags work QI (the interpretability lens, which is the PI's third question) and
   meta (repository and process); the PI's first two questions fold into Q1. The crosswalk and every ID family
   (H-, D-, F-, ALT-, W#.n, A-) are in THEME §8.

Want more depth? The full report (about 90 pages, reference only: the core is the deliverable) is not in git
and building it needs the experiment data, so ask the owner for `report_full_team.pdf`, the shareable version;
its section numbers match the core's "Detail" pointers. Every number in both reports traces to a dated devlog
entry.

## Using a coding agent? It reads AGENTS.md

`AGENTS.md` is the one instruction file for agents: what to read at the start of a task (§2), the hard rules
(§3: no long jobs from an agent shell, no commits without you, no absolute paths) and how to record a result
(§7). Each tool finds it as follows (the shim files only point, so change `AGENTS.md`, not them):

- Claude Code: `CLAUDE.md` imports `AGENTS.md`.
- OpenAI Codex: reads `AGENTS.md` directly.
- Cursor: `.cursor/rules/agents.mdc` (always applied) points to `AGENTS.md`.
- GitHub Copilot: `.github/copilot-instructions.md` points to `AGENTS.md`.
- Gemini CLI: `GEMINI.md` imports `AGENTS.md`.
- Any other tool: tell it "Read AGENTS.md first".

## Setup

```bash
# 1. Clone this repo and FactoSig side by side (FactoSig is private: ask the owner for access).
#    The study lives on branch sc_classification_gpu (main holds the older pipeline).
git clone -b sc_classification_gpu git@github.com:MinhangXu/sc_classification.git
git clone git@github.com:MinhangXu/factosig.git        # must sit at ../factosig
cd sc_classification
# 2. GPU environment (RAPIDS, Python 3.10). It also installs this repo and ../factosig in editable mode.
conda env create -f env-rapids-factosig.yml
conda activate sc-classification-rapids
pip install -e '.[app,dev]'                             # dash/plotly for the apps, pytest for the tests
# 3. Check that everything resolves.
python -m sc_classification.utils.paths
python tools/research_index.py check
python -m pytest -q tests
```

Without the experiment data, expect `exists=False` on the experiment and data lines of the path check, `check`
to exit 0 with two W11 warnings (recorded history), and pytest to pass with some tests skipped (those that read
the experiment directory). On the lab server, run the path check and pytest with the lab env's python
(`AGENTS.md` §8); `tools/research_index.py` needs only the standard library.

- No FactoSig access: delete the `-e ../factosig` line from `env-rapids-factosig.yml` before creating the
  environment. PCA/FA work; FactoSig fits do not.

- No GPU: use any Python 3.10+ environment with numpy, scipy, pandas, scikit-learn, scanpy, anndata,
  matplotlib, seaborn, pyarrow and pyyaml, then `pip install -e '.[app,dev]'` (`pyproject.toml` declares only these two
  extras). PCA/FA, the classifiers on stored scores and the analyses that read stored tables work; without
  `../factosig`, FactoSig fits do not.
- Paths: every location comes from optional `SC_*` environment variables (`SC_REPO_ROOT`,
  `SC_EXPERIMENT_ROOT`, `SC_EXPERIMENT_ID`, `SC_DATA_ROOT`, `SC_COHORT_H5AD`, and `SC_PYTHON` for the shell
  wrappers). `.env.example` explains each one. On the lab server the defaults work and the environment already
  exists as `sc-classification-rapids-py311` (`AGENTS.md` §8).
- Data: the experiment directory (about 199 GB) and the raw cohort data live on the lab GPU server, not in
  git. Ask the repo owner (Minhang Xu) for access, then point `SC_EXPERIMENT_ROOT` at the directory that holds
  the experiment. Its name is `canonical_experiment_id` in `scripts/mrd_stage0_2/research/study_config.json`.
  Never commit `experiments/`, `*.h5ad` or other data.

## Browse results without the 199 GB experiment

- The core report PDF (above).
- `scripts/mrd_stage0_2/research/CLAIMS.md` lists every headline number with its source entry, and
  `scripts/mrd_stage0_2/research/devlog/INDEX.md` lists every dated entry. The numbers live in each entry's
  Results section.
- Reports and team updates: `scripts/mrd_stage0_2/research/reports/` (the questions report; only its core PDF is
  in git) and `scripts/mrd_stage0_2/research/team_updates/` (dated notes to collaborators, such as the protein
  update for Pavan). Their figures are not in git, only the scripts, tables, captions and drafts (task W0.15).
- The notebooks in `scripts/mrd_stage0_2/notebooks/stage0_2/` are committed with their outputs. The D14
  overlap notebook is the exception: read devlog `2026-10-02_d14-patient-factor-overlap` instead.
- The two exploration apps in `scripts/mrd_stage0_2/apps/` run from an export bundle. Someone with experiment
  access runs `python scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py --export-bundle DIR` (the same for
  `old34_factor_explorer.py`) and shares DIR. You then run the app with `--data-dir DIR`.

## How work is organised (paths under `scripts/mrd_stage0_2/`)

- Workstreams: `workstreams/W0_*` to `W9_*`, one folder per line of inquiry. Each has a README with its charter and numbered tasks (such as W2.8) and holds its new code.
- Devlog: `research/devlog/<YYYY>/`, one dated entry per unit of progress (analysis, run, decision, meeting). Numbers live only in these entries' Results.
- Asks: `research/asks/`, every question from the team, the PI or a meeting, quoted verbatim with an ID (A-YYYYMMDD-NN) and where it is answered.
- Registries: `research/hypotheses.md` (claims, H-*), `research/analyses.csv` (analyses, D-numbers) and `research/campaigns.csv` (long compute runs and how to resume them).
- Paper: `research/paper/` (`OUTLINE.md`, `FIGURES.md`), the living manuscript plan. `THEME.md` §6 stays the claim boundary.

## Add a result in 3 steps

1. `python tools/research_index.py new-entry <short-slug> --workstream W<n> --type analysis` (or `run`,
   `decision`, `meeting`, `tool`, `docs`, `fix`, `refactor`). This creates `research/devlog/<YYYY>/<date>_<slug>.md`.
2. Fill its sections: the question quoted verbatim with its A-ID, what is estimated, the code and the exact
   rerun command, the artifacts, the Results with numbers, the claim-boundary lines and the next steps. Then
   update only what changed: your row in `research/analyses.csv`, your workstream README, `research/campaigns.csv`.
3. Run `python tools/research_index.py write && python tools/research_index.py check`. It regenerates
   `STATUS.md`, `CLAIMS.md` and `devlog/INDEX.md` and must exit 0. Commit only after the owner agrees.

`AGENTS.md` §7 has the details, and `research/devlog/README.md` has the entry format.

## Repository map

- `src/sc_classification/`: the installable library (imported, not run).
- `scripts/mrd_stage0_2/`: the active study. Its `README.md` describes the three stages, the runners and the panel sets.
- `tools/research_index.py`: builds and checks the research record. `tests/`: the pytest suite.
- `experiments/`: gitignored. Run outputs stay on the lab server.

## Older material

- `scripts/dr_feature_screening/` (historical method screening), `scripts/Relapse_MRD_DR_Classification/`
  (paused), `scripts/knowledge_driven_embedding/` (gene-set builders), `scripts/orchestrator/`, and
  `scripts/legacy/` (kept for provenance only).
- `DESIGN.md`, `README_STANDARDIZED_WORKFLOW.md` and `DOWNSTREAM_ANALYSIS_GUIDE.md` describe the legacy
  `StandardizedPipeline`, which the current study does not use.
- `REPO_ORGANIZATION.md` and `docs/` hold the February 2026 cleanup policy and older Git and run playbooks.
  `scripts/mrd_stage0_2/plans/` is design history. Where any of these disagree with `AGENTS.md`, `AGENTS.md` wins.

## Sharing

The results under `scripts/mrd_stage0_2/research/` are unpublished patient-level results: ask the owner before sharing them outside the team.
The GitHub repository is private and must stay so, because the committed core report and the research record
hold patient-level results. Ask the owner before adding collaborators or forking.
