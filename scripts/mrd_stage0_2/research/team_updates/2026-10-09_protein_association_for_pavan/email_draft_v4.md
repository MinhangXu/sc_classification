Hi Pavan,

Not in the dimensionality reduction: the factors are fit on RNA only, so the CITE-seq protein does not shape them. [Optional, keep or drop: that would need a joint RNA + protein model such as MOFA+ or totalVI, which we plan to try next.]

Instead we compared the factors with protein afterwards: we rank the cells on a factor score and on a protein level and correlate the two rankings (Spearman rho), in healthy and malignant cells separately: overall, within patients, and within each patient's cell types. Protein levels are CLR-normalized ADT counts (each antibody's log count relative to the cell's average over all 170 antibodies). Isotype controls are antibodies with no human target, so their signal is each cell's non-specific background; we take a factor-protein correlation seriously only if it is larger than any isotype's. Attached, for one interferon-alpha factor (PC8; our report's numbering):

- Figure 20: how one rho is computed, and what its sign means.
- Figure 22: two checks per protein. Most correlates (CD34 and CD22 in healthy cells; CD48, CD116, CD33 and CD146 in malignant cells) track PC8 within patients but not within cell types, so they mostly reflect cell identity.
- Figures 24-26: CD63 is the lead that holds up. Among malignant cells, those further toward the malignant end of PC8 carry less CD63, in 5 of 7 patients and within cell types (not in our largest patient). Separately, malignant cells carry less CD63 than healthy cells of the same sample and cell type in 15 of 21 comparisons. All descriptive, from 7 patients.

Two questions where your lab's experience would help:

1. Isotypes as the yardstick. After CLR, the isotypes mostly track each cell's total ADT count (Figure 28), and so does PC8 in some patients, so part of every correlation runs through that total. Is "beats the isotypes" a fair bar, or would you hold total counts fixed, or use dsb? One isotype (RTK2071) keeps a correlation even after that adjustment. Is it known to bind anything?
2. dsb. Our metadata has nCount_dsb, so a dsb-normalized assay existed upstream. Could you share it (or the empty-droplet background)? dsb removes ambient background and each cell's isotype-estimated technical noise, which is exactly the path above, so it would tell us whether CD63 survives the standard correction.

Best,
Minhang
