---
id: 2026-05-18_old-geneset-bottom-up-postrun
date: 2026-05-18
type: analysis
title: Precursor old-geneset bottom-up post-run diagnosis on the April experiment
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q3]
workstream: W3
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: dbde125  # retro: notebook first committed in dbde125 under scripts/comprehensive_run/notebooks; removed in db694a3
run_head: b381cd5
dirty_run: true
experiment: 20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0
backfilled: true
date_source: header  # notebook filename stage0_bottom_up_postrun_analysis_20260518
supersedes: []
---

## Question
(reconstructed from the notebook and the 2026-10-08 chronology reconstruction) In the April old-geneset run, how do panels rank under malignancy-first scoring versus structure preservation (cell-type kNN purity, pseudotime smoothness), and do knowledge panels beat the HVG and all_filtered controls? The asker is not recorded; there is no ask ID.

## Why it matters
Precursor for Q3 (does a prior gene space add signal?). It set the 'malignancy first, structure secondary' rule that the canonical 2026-05-25 design carried forward.

## Estimand / null
Panel ranking by malignancy_rank_score (percentile of across-patient AUROC/AUPRC) plus a parsimony bonus, with structure metrics shown separately. Comparator: the HVG and all_filtered controls of the same experiment. No inferential null. C policy: not applicable (reads stored April outputs; no Stage 2 refit).

## Implementation
- Notebook: scripts/comprehensive_run/notebooks/stage0_bottom_up_postrun_analysis_20260518.ipynb. It was removed in the 2026-07-02 reorganization (db694a3) as superseded by Set B; recover it with git show dbde125 on that path.
- Input: the April experiment 20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0, not the canonical experiment.
- Ran in a Jupyter kernel on CPU. Related figures sit in the umbrella repo under figures/geneset_pruning_refined/ (dated 2026-05-11).

## Artifacts
none

## Results
- Established the 'malignancy-first, structure secondary' panel-scoring rule that the 2026-05-25 sharedness design carried forward.
- The notebook was dropped in the 2026-07-02 reorganization as superseded by the Set B expanded panels (run 2026-06-22).

## Claim gate
Supports (descriptive): a panel-ranking rule for the April precursor experiment only.
Does not support: any statement about the canonical 2026-05-25 cohort, or that knowledge panels beat controls (different experiment; no matched null).

## Next steps
- W4.7: the Set B reading that replaced this precursor.
