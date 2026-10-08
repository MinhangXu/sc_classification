---
id: 2026-09-10_d1a-descriptive-deep-dives
date: 2026-09-10
type: analysis
title: D1a descriptive deep dives (taxonomy, K/method winners, focus panels, one-score proxy, cell/ADT rank-error audits)
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1, Q3]
workstream: W1
analyses: [D1DD1, D1DD2, D1DD3, D1DD4, D1DD5, D1DD6, D1DD7, D1DD8, D1DD9, D1DD10, D1DD11, D1DD12, D1DD13]
hypotheses: [H-S1-01, H-S1-02, H-S2-01, H-S2-05]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: global_selected  # D1 atlas selection; the D1DD9 mean-z proxy uses balanced L2 at C=1
training_n_changed: false
status: done
commit: b60a6ee  # retro: notebook first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # notebook title dated 20260910; artifacts regenerated 2026-09-11 14:28-14:29 UTC
supersedes: []
---

## Question
(reconstructed from the notebook and the 2026-10-08 chronology reconstruction) Unpack the D1a all-representation heatmap before any biological claim. Are fits at different K nested? Does a one-score pathway baseline match K=40? Which cell types, collection times and surface proteins carry the errors? The asker is not recorded (user or team review of D1a); there is no ask ID.

## Why it matters
Q1 (who carries the lift) and Q3 (DR vs direct and vs a one-score proxy). D1DD10-12 are the first cell-level error audit (H-S2-05), and D1DD13 the only timing view.

## Estimand / null
- D1DD1-5: atlas lift grouped by construction class, per patient x method x K, and paired selected-DR vs direct for three focus panels (allograft, apoptosis, TNFα/NF-κB).
- D1DD6-9: covered-gene Jaccard; label-free PCA total vs FA common variance; a mean-z one-score proxy in a balanced L2 (C=1) logistic under LOPO.
- D1DD10-13: exact reconstruction of selected-DR held-out rankings, then post-hoc prevalence-matched TP/FN/FP/TN groups joined to predicted cell type, 170 ADTs (log1p raw counts), QC and collection timing.
- No null; the rank cutoff uses held-out labels.

## Implementation
- Notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb, run in a Jupyter kernel on CPU from stored scores, fold scalers and cached coefficients.
- Artifacts were regenerated with the 2026-09-11 D1 rerun.

## Artifacts
- analysis/scorecards/stage2_diagnostic/d1a_deep_dive_20260910/
- analysis/figures/stage2_diagnostic/d1a_deep_dive_20260910/

## Results
- [H-S1-01 supports] Best-median DR winners over the 41 biological panels: K=40 24, K=20 13, K=10 4; by method factosig_promax 12, FA 11, PCA 9, FactoSig 9 (no globally dominant method).
- [H-S1-02 supports] Focus panels, selected-DR vs direct-gene median lift: allograft 12.56 vs 0.23 pp, apoptosis 11.52 vs 0.00 pp, TNFα/NF-κB 18.06 vs 12.78 pp; the mean-z proxy gives 1.68 / 5.30 / 1.97 pp. Allograft and apoptosis share 15 of 270 covered genes (Jaccard 0.056).
- [H-S2-01 qualifies] D1DD2: median lift across the 41 panels for factosig_promax K40 is P02 28.27, P03 29.78, P04 15.58 pp vs P01 0.76, P06 0.52, P07 2.13, P09 4.33 pp; the highest-prevalence patients dominate the summaries.
- [H-S2-05 supports] D1DD10-12: selected-DR rankings reconstruct stored AP within 3.11e-8 over 118,041 cell x focus-panel rows. TNFα recall at the rank-matched cutoff: P03 0.735, P02 0.417, P04 0.343, P09 0.253, P01 0.022, P06 0.045, P07 0.045. P01 false positives carry high platelet/megakaryocyte ADT (CD41 +5.86, CD35 +4.23, CD36 +4.10 log1p vs TN).
- D1DD13: 372/375 P09 malignant cells come from sample P09_MRD_4 (611 days after SCT, 112 days before relapse); days from SCT range from 76 (P06) to 694 (P03).

## Claim gate
Supports (descriptive): strong patient x panel x method x K heterogeneity with no dominant method; for the three focus panels the selected DR beats direct genes and a mean-z proxy on median lift; error groups can be described by cell type, ADT and timing.
Does not support: K trends as a pure capacity effect (rank, refit geometry and penalty all change); mean-z as an AUCell/UCell/ssGSEA comparison; rank-defined TP/FN as clinical misclassification; any treatment-regimen statement (no field).

## Next steps
- W3.4: exact single-score baselines (D24).
- W5.3: factor x cell type and factor x normalized ADT associations.
- W1.7: descriptive timing plot.
