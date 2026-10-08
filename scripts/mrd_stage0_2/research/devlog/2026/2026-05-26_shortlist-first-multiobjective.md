---
id: 2026-05-26_shortlist-first-multiobjective
date: 2026-05-26
type: run
title: Quick-L2 shortlist (33 representations) and first GPU multi-objective Stage 2 with metric-intuition notebooks
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1, Q2]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
compute: screen-gpu  # 8 GPU shards; launch session not recorded
c_policy: not_applicable  # discovery and patient-specific grids; LOPO at a single L2 C=1
training_n_changed: false
status: done
commit: 4ada734  # retro: runner first committed under scripts/comprehensive_run in 4ada734
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/stage2_mrd_multiobjective_scorecard_20260526_*.log; shortlist CSVs 2026-05-26 04:53 UTC
supersedes: []
---

## Question
(reconstructed from the two analysis notebooks and the 2026-10-08 chronology reconstruction) What do AUROC, AUPRC and balanced accuracy mean on this imbalanced cohort, and which shortlisted panels show discovery signal, LOPO transfer or patient-specific structure? The asker is not recorded; there is no ask ID.

## Why it matters
First supervised read of the Set A screen for Q1, and the first look at metric validity (later D6, H-I-01).

## Estimand / null
Discovery (full-cohort apparent fit), LOPO (one setting: L2, C=1) and patient-specific fits on 33 shortlisted representations; cell-prediction bundles for the validation smokes. No null.

## Implementation
- Runner: scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py, 8-GPU sharded discovery, LOPO and patient-specific runs plus validation smokes.
- Shortlist: the best quick representation per panel, ranked within panel type.
- Notebooks inside the gitignored experiment dir: a metric-diagnostics notebook (confusion counts, K/DR consistency) and a multiobjective scorecard analysis notebook.
- Launch context: not recorded (the equivalent today is screen -S <W#>_<topic> per AGENTS.md R1).

## Artifacts
- analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv
- analysis/scorecards/stage2_canonical_quick_groupkfold_scorecard.csv
- stage2_supervised/multiobjective/runs/20260526_stage2_discovery_gpu8/
- stage2_supervised/multiobjective/runs/20260526_stage2_lopo_gpu8/
- stage2_supervised/multiobjective/runs/20260526_stage2_patient_specific_gpu8/
- analysis/stage0_mrd_old34_metric_diagnostics_20260525.ipynb
- analysis/stage2_mrd_multiobjective_scorecard_analysis_20260526.ipynb
- analysis/stage2_mrd_multiobjective_review_20260526/

## Results
- 47 best-per-panel candidates; 33 selected for the multiobjective runs (6 controls, 21 single genesets, 6 groups).
- 65 regularization settings per representation: 13 L1 + 13 L2 + 39 elastic net, over C from 1e-4 to 1e4.
- The May-26 LOPO used a single setting (L2, C=1); the 2026-06-02 full-grid LOPO replaced it.

## Claim gate
Supports (descriptive): run accounting only.
Does not support: transfer statements at a single untuned C, or any final shortlist (the 33-representation matrices are historical; THEME §11 item 9).

## Next steps
- W4.1: AUROC companion column and patient-level uncertainty for the current grids.
