Hi Pavan,

Not in the dimensionality reduction: the factors are fit on RNA only, so the CITE-seq protein does not shape them. [Optional, keep or drop: that would need a joint RNA + protein model such as totalVI or MOFA+, which we have not started.]

Instead we checked the factors against protein afterwards: we correlated each cell's factor score with its surface-protein levels (CLR-normalized), within each patient, separately in healthy and malignant cells, using the 9 isotype-control antibodies as a noise yardstick. Attached, for PC8 from our report:

- Figure 1: the proteins that track PC8, patient by patient. In healthy cells the same few recur in 6 or 7 of 7 patients (CD34, CD131, CD112 up; CD22, CD48, CD39, CD81 down; |rho| 0.15 to 0.25), mostly because cell types differ: within one cell type they are near zero (CD34 still varies inside our CD34+ sort gate, by differentiation stage). In malignant cells they vary more and come close to the isotype controls.
- Figure 3: within each patient and predicted cell type (at least 10 cells of each), malignant cells score higher on PC8 than healthy cells in 19 of 21 comparisons.

Optional: Figure 2 shows the cell-type effect, Figure 4 how the isotype floor is set.

All descriptive, from 7 patients.

Best,
Minhang
