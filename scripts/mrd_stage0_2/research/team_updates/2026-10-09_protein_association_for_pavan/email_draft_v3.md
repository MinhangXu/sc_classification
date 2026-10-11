Hi Pavan,

Not in the dimensionality reduction: the factors are fit on RNA only, so the CITE-seq protein does not shape them. [Optional, keep or drop: that would need a joint RNA + protein model such as totalVI or MOFA+, which we have not started.]

Instead we compared the factors with protein afterwards: we rank the cells on a factor score and on a protein level (CLR) and correlate the two rankings (Spearman rho), in healthy and malignant cells separately: overall, within patients, and within each patient's cell types. Nine isotype-control antibodies give the noise level. Attached, for PC8 (our report's numbering):

- Figure 20: how one rho is computed, and what its sign means (positive: the two rankings run the same way; negative: in reverse).
- Figure 22 and Table 3: two checks per protein. Healthy-cell correlates (CD34, CD22) mostly reflect cell types. In malignant cells, CD48, CD116, CD33 and CD146 track PC8 within patients (|rho| 0.27 to 0.31) but not within cell types. CD99 is the reverse: weak within patients (+0.16, inside the isotype range) but holds within cell types (+0.15, 9 of 11 groups from 4 patients). We treat it as a lead, not a finding.
- Figure 19: PC8 by predicted cell type, malignant vs healthy cells. Inside one patient and cell type, malignant cells still score higher in 19 of 21 comparisons; PC8 was picked by a malignant-vs-healthy classifier, so what matters is that this holds within cell types.

All descriptive, from 7 patients.

Best,
Minhang
