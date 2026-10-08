---
id: 2026-05-25_set-a-old34-broad-screen
date: 2026-05-25
type: run
title: Set A old34 Stage 0/1 plus quick L2 broad screen (canonical experiment created)
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1, Q3]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-cpu  # device and launch session not recorded; about 7 h wall clock
c_policy: global_selected  # quick L2 at one fixed C=1 for every representation (triage)
training_n_changed: false
status: done
commit: 4ada734  # retro: runners first committed under scripts/comprehensive_run in 4ada734; experiment_config records b381cd5, which predates them
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/stage0_mrd_old34_broad_screen_20260525_060508.log, 06:05 to 13:22 UTC
supersedes: []
---

## Question
(reconstructed from the run configuration and the 2026-10-08 chronology reconstruction) Which curated prior gene spaces (34 Hallmark/Reactome/KEGG programs and 7 biology groups), compared with HVG and full-union controls, contain malignant-vs-normal structure after DR? The asker is not recorded; there is no ask ID.

## Why it matters
This run created the canonical experiment and every Stage 1 basis that the Q1-Q3 analyses read. All later transfer, overlap and interpretability results sit on these transductive, seed-42 fits.

## Estimand / null
Triage only: a quick L2 logistic model (C=1, balanced classes) under 5-fold GroupKFold by patient, scored by cell-weighted AUROC/AUPRC per representation. This is not LOPO and has no null.

## Implementation
- Entry point: bash scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.sh with no extra flags (driver scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py). Dictionary: scripts/knowledge_driven_embedding/older_geneset/genesets_v1.gmt.
- Filters: Tech == CITE, timepoint MRD, CN.label in {cancer, normal}. normalize_total + log1p; genes kept if detected in at least 1% of cells; within-panel z-score.
- Stage 1: pca, fa, factosig and factosig_promax at K 5/10/20/40, seed 42, fit transductively on all eligible cells; direct_gene fallback for small panels.
- Launch context: not recorded; no screen session name appears in the logs or docs. A rerun would follow AGENTS.md R1 (screen -S <W#>_<topic>, then the command).
- Provenance: experiment_config.yaml records git b381cd5, which predates the producing code (first committed in 4ada734).

## Artifacts
- experiment_config.yaml
- run_manifest.json
- analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv
- analysis/reports/postrun_human_review.md
- logs/stage0_mrd_old34_broad_screen_20260525_060508.log
- stage1_dr/

## Results
- Cohort after filters: 60,118 cells x 15,560 genes, 13 patients, 1,569 malignant cells (prevalence 0.026).
- 47 panels fit; the scorecard has 755 rows (739 ok, 16 skipped_too_few_genes). The run took 06:05 to 13:22 UTC.
- Top quick row by AUROC: HVG3000 FA K=40 (AUROC 0.889, AUPRC 0.530). Best by AUPRC: HVG10000 FactoSig K=40 (AUPRC 0.549). Best biology group: proliferation_metabolism FactoSig K=40 (AUPRC 0.319).

## Claim gate
Supports (descriptive): run accounting only; quick-L2 metrics are cell-weighted triage.
Does not support: transfer claims (quick L2 is GroupKFold, not LOPO) or any ranking of priors against controls.

## Next steps
- W2.1: the inductive basis variants that test this transductive fit.
- W4.5: Stage 1 seed and bootstrap stability (only seed 42 exists).
