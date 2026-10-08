# sc_classification

**Contributors and coding agents: start at [`AGENTS.md`](AGENTS.md). Current study status: [`scripts/mrd_stage0_2/research/STATUS.md`](scripts/mrd_stage0_2/research/STATUS.md).**

Modular repo for developing  single cell level classification from dimension reduction to semi-/supervised classification algorithm to downstream validations 

## Project Structure

This project follows a standard `src` layout to cleanly separate the installable Python library from run scripts and analysis notebooks.

*   `src/sc_classification/`: This is the core Python package. It contains all the reusable modules for the pipeline, such as preprocessing, dimensionality reduction, and classification methods. Code in here is meant to be imported, not run directly.

*   `scripts/`: This directory contains standalone Python scripts that serve as entry points for running experiments. These scripts import the `sc_classification` package to execute the pipeline with specific configurations.

*   `notebooks/`: This directory is for Jupyter notebooks used for exploratory data analysis, visualization, and interpretation of results.

*   `tools/research_index.py`: stdlib-only generator and validator for the study research spine (`scripts/<study>/research/`).

*   `DESIGN.md`: Provides a high-level overview of the software architecture and pipeline workflow (the legacy `StandardizedPipeline`; the `mrd_stage0_2` runners do not use it).

`experiments/` is gitignored (run artifacts stay on the lab server); the path environment variables are listed in `.env.example` and `AGENTS.md` §8.

## Studies under `scripts/`

- `scripts/mrd_stage0_2/` (active): the knowledge-prior Stage 0-2 malignant-vs-normal study. Start at `AGENTS.md`, then `scripts/mrd_stage0_2/research/STATUS.md`.
- `scripts/dr_feature_screening/` (historical): older DR method/K screening (Plan 0/1/1c/1d).
- `scripts/Relapse_MRD_DR_Classification/` (paused): paired MRD/Relapse per-patient 4-class benchmark.
- `scripts/knowledge_driven_embedding/`: upstream gene-set builders (older and expanded Stage 0 gene sets).
- `scripts/orchestrator/`: stage orchestration scaffolding.
- `scripts/legacy/`: historical scripts, kept for provenance.

## Repository docs

- Repository cleanup policy and archive guidance are documented in `REPO_ORGANIZATION.md`.
- Operational docs:
  - `docs/GITHUB_UPDATE_PLAYBOOK.md`
  - `docs/AUTONOMOUS_COMPREHENSIVE_RUNS.md`
