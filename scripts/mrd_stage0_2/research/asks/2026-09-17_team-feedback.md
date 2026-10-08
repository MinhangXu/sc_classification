---
date: 2026-09-17
source: meeting
asked_by: meeting:2026-09-17
capture: verbatim_from_report
original_location: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_feedback_deep_dives_20260917.ipynb, the Team claim paragraphs of Observations 1 to 3 (the notebook's record of the 2026-09-17 review of the D1a heatmap)
---

# 2026-09-17 team review of the D1a heatmap: team claims

- [A-20260917-01] "Increasing K generally increases median lift on knowledge-driven gene sets. HVGs under PCA are the exception: K=5 is poor, K=10 peaks, K=20/40 drop — interpreted as one discriminative factor appearing between 5 and 10, then noise factors diluting accuracy. Under FA (described as “no orthogonality constraint”), HVG accuracy was said to increase monotonically to K=40." -> F1
- [A-20260917-02] "Allograft rejection, apoptosis, and interferon-α have near-zero direct-gene lift, while FactoSig K=40 gives about +12 pp. Hypothesis: nonspecific immune genes (LCK, ZAP70) in the allograft set overlap cytokine / TNF signaling, and DR extracts a latent signal that the gene-wise L1 probe misses." -> F2, F5
- [A-20260917-03] (teammate:Ankit) "Ankit: FA’s nonorthogonal basis makes the expansion non-unique, so L1 zeroes one of two correlated factors; PCA’s orthogonal basis is unique. Implication: loadings cannot be interpreted biologically — sign is meaningless, and magnitude is unreliable because L1 picks one correlated factor, L2 splits, elastic-net is in between." -> F3, H-S2-07
