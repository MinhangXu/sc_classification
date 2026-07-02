# mrd_stage0_2 — knowledge-prior Stage 0-2 malignancy study

Current study: build and biologically interpret a three-stage knowledge-prior malignant-vs-non-malignant
classifier on the MRD cohort, then answer the three PI questions (robust shared model,
patient-specific biology, breaking apart programs).

**Start here:** `plans/INDEX.md`, then the rehydration dossier `plans/experiment_20260525_stage0_2_rehydration_dossier.md`
(master map of the run: pipeline artifacts, provenance, gaps, per-plot framing).

## The three stages

- **Stage 0 — gene-space panels.** Restrict genes to a knowledge-prior panel (single genesets, biology
  groups, family unions, leave-one-family-out, plus HVG/full/core controls).
- **Stage 1 — representation learning.** DR on the panel: `pca`, `fa`, `factosig`, `factosig_promax` at `K in {5,10,20,40}`.
- **Stage 2 — regularized supervised probes.** L1/L2/elastic-net logistic paths under three objectives:
  discovery full-cohort fit, sharedness leave-one-patient-out (LOPO), and patient-specific.

## Layout

- `stage0_panels/` — Stage 0/1 (+ quick Stage 2) runners.
  - `run_stage0_mrd_old34_broad_screen.py` — the Stage 0/1/quick-Stage-2 screen (both panel sets).
  - `run_stage0_mrd_old34_broad_screen.sh` — thin wrapper that launched the old34 set (Set A).
  - `run_expanded_stage0_genesets_stage0_to_stage2.sh` — builds the expanded manuscript-axes bundle
    (Set B) and drives Stage 0 → multi-objective Stage 2.
  - `configs/expanded_stage0_mrd_manuscript_axes.yaml` — expanded panel-set config.
- `stage2_supervised/` — multi-objective Stage 2 + plotting/interactive tools.
  - `run_stage2_mrd_multiobjective_scorecard.py` — discovery / LOPO / patient-specific reg-path scorecards.
  - `stage2_sharedness_plotting.py` — shared plotting/label helpers imported by the notebooks and audit script.
  - `stage2_mrd_fig3a_interactive.py`, `..._top_stage1_shortlist.py` — interactive Fig 3A HTML builders.
  - `build_lopo_transfer_audit_tables.py` — patient-wise LOPO transfer audit tables/plots.
- `notebooks/stage0_2/` — analysis notebooks (authoritative: `stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb`).
- `plans/` — design + run-spec + diagnostic plans and the rehydration dossier (`INDEX.md` maps plan → implementation).

## Two panel sets in the run

The run `experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093/` holds two Stage 0 panel sets:

- **Set A — old34**: curated single genesets / biology groups / controls (`genesets_v1.gmt`).
- **Set B — expanded manuscript axes** (`--branch-name expanded_stage0_mrd_manuscript_axes_v1`): atomic sets,
  family unions, and leave-one-family-out panels — the realization of `plans/stage0_geneset_value_added_workflow.md`.

## Run the pipeline

Old34 broad screen (Stage 0/1 + quick Stage 2):

```bash
bash sc_classification/scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.sh
```

Expanded manuscript-axes set (build bundle → Stage 0 → multi-objective Stage 2):

```bash
bash sc_classification/scripts/mrd_stage0_2/stage0_panels/run_expanded_stage0_genesets_stage0_to_stage2.sh \
  --experiment-dir sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093 \
  --run-stage2 --gpu-ids auto
```

Multi-objective Stage 2 on an existing Stage 0 scorecard:

```bash
python sc_classification/scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py \
  --experiment-dir <EXP_DIR> --stage0-scorecard <scorecard.csv> --stage2-run-id <id> \
  --run-discovery-full-cohort-fit --run-sharedness-lopo --run-patient-specific \
  --penalties l1,l2,elasticnet
```

Generated `experiments/.../` artifacts and notebook `outputs/`/`figures/` are gitignored (regenerable).

## Related

- Older DR method/K screening (Plan 0/1/1c/1d): `scripts/dr_feature_screening/`.
- Upstream gene-space builders: `scripts/knowledge_driven_embedding/`.
