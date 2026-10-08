---
date: 2026-10-08
source: report
asked_by: user
capture: paraphrase
original_location: The user's 2026-10-08 team report (TODO list, discussion notes, inline ask and the three scientific questions), itemized in report order by the 2026-10-08 understanding pass; the item texts are that pass's condensed wording (single-quoted fragments inside an item follow the report's wording); the report itself is not stored in the repo
---

# 2026-10-08 team report: TODO list, discussion items and scientific questions

- [A-20261008-01] "DONE: interactive heatmap explorer (patient-specific heatmaps colored by aggregates such as cohort-median lift, or by threshold-based metrics)" -> D1, W7.1
- [A-20261008-02] "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" -> H-S1-05, W1.1, W2.1
- [A-20261008-03] "DISCUSSED: IFN-alpha response, PCA K=40 gives lift in four patients but FA K=40 captures only P03 and P04" -> H-S2-08, W3.2
- [A-20261008-04] "DISCUSSED (a): hypothesis that PCA orthogonality pushes later components beyond P03, while FA keeps allocating factors to the highest-prevalence patient" -> H-S2-07
- [A-20261008-05] "DISCUSSED (a): 'Need to inspect the factor-by-patient matrix for FA to test this'" -> W1.1
- [A-20261008-06] "DISCUSSED (b): PC8 loadings form two opposite-sign gene groups (IFI27, PSME2, PSMA3, CNP, LAP3, USP18 vs TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1), plus a trailing 'IFI27, IFITM2'" -> F9, W5.2
- [A-20261008-07] "TOP PRIORITY: data-driven statistical test for whether to aggregate patients" -> D14, W1.2, H-S1-04
- [A-20261008-08] "Step 1: learn factors on all patients, project each patient's cells ('we should be able to do this already')" -> D14
- [A-20261008-09] "Step 2: check per-patient factor distributions (plot from the canonical experiment)" -> D14
- [A-20261008-10] "Step 3: if distributions overlap significantly, patients share variation; if distinct, treat them as separate subtypes" -> D14, W1.2
- [A-20261008-11] "Inline ask: reorganize the report; review recent files in notebooks/stage0_2 and apps; agent-agnostic context engineering; per-agent dev subdirs; dated log (day, question, implementation); a central theme for the paper" -> W0.1
- [A-20261008-12] "Incorporate new patient MRD data from Yu-Hsiang (# I need to email him)" -> W8.1
- [A-20261008-13] "Option 1.1: subset to the top 4-5 patients with sufficient malignant cells, plus YH's new patient; plan the run after choosing" -> W2.5
- [A-20261008-14] "Option 1.2: re-weight cells so each patient contributes equally to DR" -> W2.1, W2.3
- [A-20261008-15] "Option 1.3: re-examine the PCA and FA factors to see whether the imbalance problem persists" -> W1.3
- [A-20261008-16] "Option 2: exclude the held-out patient from Stage 1 entirely, then project their cells onto the learned factors" -> W2.1, H-S1-06
- [A-20261008-17] "Option 3: focus only on patients with large cell counts (2, 3, 4) and rerun the full pipeline" -> W2.5
- [A-20261008-18] "Use days since transplant and days before relapse per patient to plot variation across patients and consider timing as a covariate (low priority; YH says a similar manuscript plot exists)" -> W1.7
- [A-20261008-19] "Factor interpretability to insight, with IFN-alpha as the example; expand beyond IFN-alpha and organize alongside the top priority" -> W5.1
- [A-20261008-20] "Can we view these interpretability results in the interactive app? (or explain why the app should not go to factor specifics)" -> W7.2
- [A-20261008-21] "Correlate factors with other data modalities (cell types, surface protein expression)" -> W5.3
- [A-20261008-22] "Review interpretability frameworks from other fields (e.g. factor rotation in psychology; iterate with domain knowledge)" -> W5.4, W9.1
- [A-20261008-23] "We cannot make a causal claim without experimental intervention" -> THEME
- [A-20261008-24] "Plot AUPRC vs number of cells (learning / marginal-benefit curve) ('we can do this for sure, right?')" -> D15, W6.2
- [A-20261008-25] "Learning curve useful for grants to justify sequencing more patients" -> W6.1, W6.4
- [A-20261008-26] "Randomly subsample to estimate the curve (as Audit AI does with Eclipse)" -> W6.1, W6.2
- [A-20261008-27] (teammate:Ankit) "Ankit: end-to-end semi-supervised analysis with factors learned jointly to discriminate healthy vs malignant (later, once new data arrive)" -> W3.7
- [A-20261008-28] "Review single-cell ML literature for existing frameworks" -> W9.1
- [A-20261008-29] "Current scientific questions: central branch for publication; organize the whole repo's development around them" -> THEME, W0.1
- [A-20261008-30] "Q1: shared or patient-specific?" -> THEME, W1.2, W1.5, W2.6
- [A-20261008-31] "Q2: how strong is the evidence (variance explained, effect size, significance)?" -> THEME, W4.1, W4.2, W4.4
- [A-20261008-32] "Q3: what does dimensionality reduction add over direct-gene classification?" -> THEME, W3.3, W3.4
