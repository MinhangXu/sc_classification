---
id: 2026-07-03_figure-catalog-option-b
date: 2026-07-03
type: decision
title: Figure catalog and statistical-rigor contract; patient-summary lift (Option B) adopted
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q2]
workstream: W4
analyses: [V2]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: 43f19be  # retro: catalog first committed in 43f19be (2026-07-23)
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # 'Date compiled: 2026-07-03' and 'Decision (2026-07-03)' in the catalog
supersedes: []
---

## Question
(reconstructed from the catalog, which answers dossier §11.1) Which V2 plots are figure-worthy under the four-element framing, and why do pooled-cell and patient-summary numbers disagree? The asker is not recorded; there is no ask ID.

## Why it matters
Q2 effect size. This decision fixed the study's metric contract: patient-summary lift over the supported patients, not pooled-cell AUPRC. It also specified Null Model 1 and the uncertainty work that W4 still owes.

## Estimand / null
- Option A: pooled-cell aggregate AUPRC (support-weighted, inflated by high-malignant-count patients).
- Option B: per-patient lift = held-out AUPRC − that patient's malignant prevalence, summarized by median and P20 over the supported patients.
- Supported patient: at least MIN_MALIGNANT_SUPPORT malignant cells.
- Null Model 1 (specified, not built): random gene sets of matched size through the same Stage 0-2 pipeline, with an empirical p.

## Implementation
- Catalog: scripts/mrd_stage0_2/plans/stage2_figure_catalog_and_statistical_rigor_v2_jun4.md.
- Applied in scripts/mrd_stage0_2/notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb (V2F2/V2F2b/V2F2c switched to Option B; the enrichment metric removed).

## Artifacts
- analysis/scorecards/stage2_figure3_sharedness_v2_jun4/v2B2_lopo_patient_auprc_enrichment_matrix.csv — retired table (V2B2 removed on 2026-07-03)

## Results
- Decision (2026-07-03): Option B everywhere, i.e. median and P20 lift over the 7 supported patients P01, P02, P03, P04, P06, P07, P09 (MIN_MALIGNANT_SUPPORT = 10 malignant cells).
- V2B2, the per-patient AUPRC/prevalence enrichment heatmap, was removed on 2026-07-03.
- Null Model 1 and patient-level uncertainty were specified but are still not implemented on 2026-10-08.

## Claim gate
Supports (descriptive): nothing; this is the metric contract.
Does not support: pooled-cell AUPRC as a headline, or any significance statement before Null Model 1 exists.

## Next steps
- W4.2: Null Model 1 (D21).
- W4.1: patient-level uncertainty and AUROC companion (D22).
