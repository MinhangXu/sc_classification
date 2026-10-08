---
date: 2026-07-24
source: meeting
asked_by: meeting:2026-07-24
capture: reconstructed
original_location: Next Steps of the 2026-07-24 meeting summary, pasted into a Cursor agent chat that is not stored in the repo; numbering follows the narrative guide cross-references (questions 1-4 and 6 to the meeting follow-up notebook, question 5 to the donor notebook); wording as relayed by the 2026-10-08 transcript review and not yet re-checked against the transcript (W0.7)
---

# 2026-07-24 team meeting: next steps (reconstructed)

- [A-20260724-01] "Fix the lift metric to an apples-to-apples AUPRC comparison: replace the gray bar (prevalence) with the AUPRC of a dumb baseline classifier (minority- or majority-class guesser) so blue and gray are on the same scale." -> D6
- [A-20260724-02] "Add a no-DR baseline classifier for each gene set: train regularized logistic regression directly on the gene subset to quantify the value added by Stage 1 DR." -> D9
- [A-20260724-03] "Run factor-by-factor matrix analysis for interferon alpha response PCA-40 vs FA-40: identify which PCA factors drive the lift advantage for patients 2 and 7, and why FA fails to capture the equivalent structure." -> D8
- [A-20260724-04] "Generate per-patient lift plots across all gene sets and DR methods: targeted manual exploration of gene set × DR × patient interactions, not a full grid search." -> D7
- [A-20260724-05] "Implement donor cell downsampling and test its effect on lift and factors: 'previously implemented but not applied in the current run'; vary the downsampling level and check whether held-out AUPRC and factor loadings change materially." -> D10, D11
- [A-20260724-06] "Show malignant vs healthy cell counts per patient in MRD data (recipient-healthy transparency)." -> D6
