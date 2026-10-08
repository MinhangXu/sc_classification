---
id: 2026-06-04_three-pi-questions-v2-suite
date: 2026-06-04
type: analysis
title: Three-stage design note with the three PI questions, and the V2 sharedness suite
asked_by: PI
asks: [A-20260604-01, A-20260604-02, A-20260604-03]
implemented_by: unknown
questions: [Q1, QI]
workstream: W1
analyses: [V2]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: notebook-cpu
c_policy: not_applicable  # reads stored Stage 2 grids; no refit
training_n_changed: false
status: done
commit: dbde125  # retro: V2 notebook first committed in dbde125; design note in 91c296b
run_head: b381cd5
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # 'Date: 2026-06-04' in the design note; notebook suffix jun4
supersedes: [2026-05-28_figure3-sharedness-v1]
---

## Question
The three PI questions, as recorded in scripts/mrd_stage0_2/plans/three_stages_knowledge_prior_mal_classification.md (section 'How The Design Answers The Three PI Questions'; A-20260604-01..03):

> "Can we build a robust model that separates malignant from non-malignant cells across patients?"

> "Can patient-specific models reveal new biology rather than simply repeat the shared model?"

> "Does each prior program behave as one gene program, or does it contain distinct regulatory groups?"

These are the design note's wording of the PI's questions; the PI's original phrasing is not saved separately.

## Why it matters
PI-Q1 and PI-Q2 together are the user's Q1 (shared vs patient-specific); PI-Q3 is the QI lens (H-S2-03). The V2 suite was the first decision table across the three Stage 2 views, and its section E cost accounting led to the 2026-06-05 all-biological grid.

## Estimand / null
Per representation: discovery apparent AUPRC, per-patient LOPO lift (AUPRC − held-out prevalence) and patient-specific gaps, plus method-aware Stage 1 stability and a coefficient-to-loading view. No null; selection is in-sample.

## Implementation
- Design note: scripts/mrd_stage0_2/plans/three_stages_knowledge_prior_mal_classification.md (schematic plus caption).
- Notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb (V2A-V2F6). It was re-executed on 2026-07-24 after 43f19be; the b60a6ee diff of this notebook is re-execution output only.
- Ran in a Jupyter kernel on CPU over stored scorecards.

## Artifacts
- analysis/scorecards/stage2_figure3_sharedness_v2_jun4/
- analysis/figures/stage2_figure3_sharedness_v2_jun4/

## Results
- The 3 PI questions became the study's organizing questions; they differ from the user's 2026-10-08 Q1-Q3 framing (crosswalk in THEME §8).
- Section E expansion accounting led directly to the 2026-06-05 grid over all 41 biological panels.
- V2 later produced a provisional 103-representation union shortlist, labelled as not confirmatory.
- The V2 decision table (regenerated 2026-07-24 00:31 UTC on the 643-representation grid) is the decision_643 input of D1-D3 (old34_diagnostic_manifest.json).

## Claim gate
Supports (descriptive): the organizing questions and an exploratory decision table.
Does not support: V2F plots as claim plots (they collapse to 41 panels too early; D1 supersedes them per the 2026-08-25 audit), or the 103-representation shortlist as final evidence.

## Next steps
- W1.6: test predictive patient-specific signal (PI-Q2).
- W5.5: census of programs that break into sub-axes (PI-Q3).
