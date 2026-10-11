# Table 3, v3: PC8 and FA7 against surface protein at three levels

Spearman ρ between a factor score and a surface protein (CLR), computed among cells of one label only (healthy or malignant). Both factors are oriented: ρ > 0 means cells ranked higher on the factor (toward its malignant-leaning end) tend to rank higher on the protein, ρ < 0 that they tend to rank lower (the two rankings run in reverse). Numbers from D27. In the questions report: full report Table 3 (all rows); core report Table 1 (the PC8 rows). The report's version 3 table had only the unmarked rows. † = in the report v1 table (top 3 by |pooled ρ|). ‡ = added 2026-10-10: CD116, CD33 and CD146, PC8's other strong malignant-cell correlates (asked for), and CD63, CD82 and GPR56, the other proteins the two-checks figure labels. Numbers behind it: `table3_rho_three_levels_v3.csv`.

| Factor, cells | Protein | all cells pooled | median within patient (patients agreeing) | median within patient and cell type (groups agreeing) |
|---|---|---|---|---|
| **PC8, healthy** (58,549 cells) | CD34 | +0.251 | +0.245 (7/7) | +0.019 (40/71) |
|  | CD22 | −0.221 | −0.246 (6/7) | −0.046 (52/71) |
| | *isotype reference* | *0.056* | *0.109* | *0.043* |
| **PC8, malignant** (1,569 cells) | CD99 | +0.222 | +0.159 (6/7) | +0.150 (9/11) |
|  | CD48 | −0.192 | −0.308 (6/7) | −0.011 (6/11) |
|  | CD116 ‡ | −0.108 | −0.287 (7/7) | −0.037 (9/11) |
|  | CD33 ‡ | −0.137 | −0.272 (5/7) | +0.026 (6/11) |
|  | CD146 ‡ | +0.090 | +0.270 (6/7) | +0.017 (6/11) |
|  | CD63 ‡ | −0.183 | −0.252 (6/7) | −0.138 (9/11) |
|  | CD82 ‡ | +0.138 | +0.238 (7/7) | +0.143 (9/11) |
|  | GPR56 ‡ | +0.203 | +0.137 (6/7) | −0.057 (6/11) |
| | *isotype reference* | *0.101* | *0.227* | *0.120* |
| **FA7, healthy** (58,549 cells) | CD22 | −0.354 | −0.312 (7/7) | −0.085 (53/71) |
|  | CD34 | +0.260 | +0.298 (7/7) | −0.004 (36/71) |
|  | CD112 † | +0.297 | +0.282 (7/7) | +0.035 (43/71) |
| | *isotype reference* | *0.080* | *0.071* | *0.045* |
| **FA7, malignant** (1,569 cells) | CD116 | −0.108 | −0.316 (6/7) | −0.035 (6/11) |
|  | CLEC12A | −0.158 | −0.291 (7/7) | −0.041 (8/11) |
|  | GPR56 † | +0.215 | +0.247 (6/7) | −0.004 (6/11) |
|  | CD63 † | −0.181 | −0.195 (5/7) | −0.111 (9/11) |
| | *isotype reference* | *0.122* | *0.257* | *0.122* |

How to read the columns:

- **all cells pooled**: one ρ over every cell of that label in the cohort (58,549 healthy cells from all 13 patients, or 1,569 malignant cells from the 9 patients that have any).
- **median within patient (patients agreeing)**: ρ computed inside each of the 7 patients with at least 10 malignant cells, then the median of the 7; in brackets, how many of the 7 have the same sign as that median (the report's 'within (k/7)').
- **median within patient and cell type (groups agreeing)**: ρ computed inside each patient × predicted cell type group with at least 30 cells of that label (71 healthy groups from all 7 patients; 11 malignant groups from P02, P03, P04 and P09 only), then the median; in brackets, how many groups have the same sign as that median (the report's 'in type' showed the median only). D27's own count (`n_pct_same_sign`) instead uses the sign of the within-patient median; the CSV has both, and they differ only where the two medians differ in sign (here PC8 × CD33 and PC8 × GPR56 in malignant cells, FA7 × CD34 in healthy cells and FA7 × GPR56 in malignant cells).
- **isotype reference**: the largest |ρ| that any of the 9 isotype-control antibodies (no target on human cells) reaches in the same column, for that factor and label. It is a yardstick, not a significance test. For PC8 in malignant cells one isotype (RTK2071) sets both the within-patient and the within-type value.
- D27's 'robust' rule (the old report Figure 20, now Figure 21) reads the within-patient column only: |median| ≥ 0.20 and the same sign in at least 5 of 7 patients. CD48, CD116, CD33, CD146, CD63 and CD82 pass it for PC8 in malignant cells; of these only CD63 and CD82 keep much inside cell types. The rule's 0.20 bar is lower than the malignant-cell isotype reference (0.227), so passing it does not by itself mean leaving the isotype band; these six all do.

Descriptive; Stage 1 is transductive and fit on RNA only; CLR without dsb; predicted cell types are reference-mapping labels.
