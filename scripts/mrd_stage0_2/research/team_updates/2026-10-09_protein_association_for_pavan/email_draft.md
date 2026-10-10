Hi Pavan,

Not in the dimensionality reduction itself: the factors in our report are fit on RNA only, so the CITE-seq protein does not shape them. [Optional, keep or drop: integrating it would need a joint RNA + protein model such as totalVI or MOFA+; we have not started that.]

What we have done is check the factors against protein after the fact. For each factor we correlated every cell's score with its surface-protein levels (CLR-normalized per cell, across all 170 antibodies), separately in normal and in malignant cells, within each patient and within predicted cell types, and used the 9 isotype controls as a noise floor. The attached figure updates the PC8 figure and table from our report.

For PC8, CD34 tracks the score in all 7 patients (median Spearman rho +0.24) but barely within a cell type (+0.02), so the link mostly reflects differences between cell types. In malignant cells the correlations are weaker and close to the within-patient isotype floor (0.23). CD99, our earlier top hit, is weak: +0.16 within patients (same sign in 6 of 7, below that floor) and +0.15 within cell types, about what one isotype control also shows, so it is suggestive at most. FA7, the factor-analysis version in the earlier table, gives the same pattern (CD34 +0.30 within patients, 7 of 7, about 0 within cell types).

Happy to share the full table. These are descriptive results from 7 patients.

Best,
Minhang
