---
date: 2026-10-10
source: user
asked_by: user
capture: verbatim
original_location: two chat messages from the user on 2026-10-10 in a claude-code session, about the questions report's Figure 19, Table 3 and Figure 20 (the D27 section) and then about the v3 explainer figures; passed verbatim to the session that wrote devlog 2026-10-10_protein-explainer-v3 and recorded here the same day
---

# 2026-10-10 questions on report Figure 19, Table 3, Figure 20 and the sign of rho (PC8 and FA7 x surface protein)

Each quoted text is verbatim (wording and typos kept), split from two messages; both messages are quoted whole below. The questions are answered by the v3 figures, table and captions in `research/team_updates/2026-10-09_protein_association_for_pavan/` (devlog 2026-10-10_protein-explainer-v3). Report Figure 19 = fig05c_celltype, Table 3 = the PC8/FA7 surface-protein table and Figure 20 = fig05e_rho_anchors, all in section 5 of the 2026-10-08 questions report.

## Figure 19, Table 3 and Figure 20 (D27, W5.3)

- [A-20261010-14] "I feel like figure 19 could be useful to be included in the email, but I am a bit confused about the cell type counts, the numbers on the y-axis doesn't add up to a number that I've seen before, this is all patient's all cells? Why 643 HSC/MPP2?" -> W5.3, D27
- [A-20261010-15] "What's mean oriented PC8 score." -> W5.3, D27
- [A-20261010-16] "I think we should include illustration showing the matrix decomp: PC8's cell score for all malignant cells is a vector of length (1569) and every entry is the PC8's score for that cell. And spearman rho is just the correlation of the ranking of that with CD34's cell score or another surface protein's cell score." -> W5.3, D27
- [A-20261010-17] "Table 3 also looks good to show these rho. The within (k/7), k means the number of patients that has the same rho sign, right? In type is a median value which is an aggregated stats." -> W5.3, D27
- [A-20261010-18] "My observation here would be that CD99's correlation between malignant cell and PC8 seems to not diminish when we check it per patient per cell type (stayed as high as 0.150), which could mean something, but later in Fig 20 when we include the isotype's band, the plot concludes that CD99's malignant association is not robust hmmm? But CD48 is somehow more robust by exceeding the isotype's band." -> W5.3, D27
- [A-20261010-19] "And FA7 malignant cells' surface protein correlation is finding two different surface protein compare to its healthy counter part." -> W5.3, D27
- [A-20261010-20] "Lastly, do these signs mean anything, e.g. CD22 have a negative value for both PC8 and FA7 on the normal cells. And CD34 has a positive value. So does that mean CD22 is less associated with normal cells? Or just that the latent factors' cell score vector is like the reversed rank with CD22?" -> W5.3, D27

## Follow-up on the v3 explainers (second message, same day)

The F5-F7 names in this list are the working names of the v3 files (`fig5_factor_protein_to_rho_v3` and so on), not report figure numbers.

- [A-20261010-21] "I think we should also include "CD116−0.287, CD33−0.272, CD146 +0.270" from PC8's other strong malignant correlates." -> W5.3, D27
- [A-20261010-22] "I don't see the schematics (F5?, it's not figure 5), maybe I am not finding it. I actually don't find all the F figures you just created." -> W0, D27
- [A-20261010-23] "So the sign means how forward and reversed the rank of the factor cell score against the ADT protein cell score? We should include that in the schematics F5." -> W5.3, D27
- [A-20261010-24] "Why not just put those into the report?" -> W0, D27
- [A-20261010-25] "can I say \"CD99 is pretty consistent for within-patient and within cell types, which means that it tracks cell types more. However, CD48 has a high |ρ|, but close to zero (wrt the y-axis), which means that it's not tracking the cell types within patients, which makes it more interesting, right?\" So CD63 and CD82 outside of the isotype vertical area also have high |ρ| on the y-axis, which means that they are also tracking cell types. So it makes CD48, CD33, CD116, GPR56, CD146 more interesting?" -> W5.3, D27

## Where each item is answered

All in devlog 2026-10-10_protein-explainer-v3, the v3 files of the update directory and, since version 3.1 of the questions report (2026-10-10 evening), the report itself: full report section 5 (Figures 19, 20, 22 and Table 3) and the core report's pages 7-8 (Figure 6 bottom, Table 1, Figure 7). Asks routed to W5.3 stay partial until W5.3 closes; A-20261010-22 and -24 (where the figures are) are routed to W0. The `.png`/`.pdf` files and `report_full.pdf` are gitignored: they exist only in the lab working tree, and a clone sees only `report_core.pdf` (which files a clone should carry is W0.15, the owner's decision).

| Ask | Short | Answered in |
|---|---|---|
| A-20261010-14 | Figure 19's counts (all patients? why the HSC/MPP2 count?) | `fig7_celltype_means_v3` and its caption: (malignant / healthy) cells of all patients; which types are left out; the HSC/MPP2 malignant cells split by patient. Full report Figure 19 (relabelled); core page 7, Figure 6 bottom (compact); `email_draft_v3.md` now attaches it |
| A-20261010-15 | "mean oriented PC8 score" | `fig7_celltype_means_v3` subtitle and caption; full report Figure 19 caption |
| A-20261010-16 | illustration: one score per malignant cell, rho = correlation of ranks | `fig5_factor_protein_to_rho_v3` (matrices to vectors, 8-cell worked example, three levels, sign); full report Figure 20; core page 8, Figure 7A (the sign, compact) |
| A-20261010-17 | Table 3's k/7 and "in type" | `table3_rho_three_levels_v3.md` (plain headers, groups agreeing, isotype reference per column); full report Table 3; core page 8, Table 1 (the PC8 malignant-cell rows) |
| A-20261010-18 | CD99 holds within type but is "not robust"; CD48 "robust" | `fig6_two_checks_v3` and its caption (the robust rule reads only the within-patient column); full report Figure 22 and paragraph "Two checks per protein" (which check each protein passes); core page 8, Figure 7B-C |
| A-20261010-19 | FA7's malignant-cell vs healthy-cell proteins | `caption_v3.md`, paragraph "Why FA7's healthy-cell and malignant-cell proteins differ"; full report section 5, paragraph "Why FA7's normal-cell and malignant-cell proteins differ" |
| A-20261010-20 | what the signs mean (CD22 negative, CD34 positive) | `caption_v3.md`, paragraph "What the sign means"; full report box "How to read rho", paragraph "What the sign means" |
| A-20261010-21 | add CD116, CD33, CD146 | Table 3 v3 rows (PC8, malignant), labelled in `fig6_two_checks_v3` and in its reading key; full report Table 3, Figure 22 and paragraph "Two checks per protein"; core page 8, Table 1 and Figure 7B |
| A-20261010-22 | where are the schematic and the F figures? | They were gitignored files in the update directory only (`*.png`, `*.pdf`), so they never reached the remote. Now also in both reports: F5 = full report Figure 20 (pp. 54-55; core Figure 7A), F6 = Figure 22 (p. 57; core Figure 7B-C), F7 = Figure 19 (p. 51; core Figure 6 bottom), table = Table 3 (p. 53; core Table 1). Each update figure prints its report numbers; mapping in `caption_v3.md`, "Where each figure is in the reports". `report_full.pdf` is gitignored too: a clone has the core only (W0.15) |
| A-20261010-23 | the sign = forward or reversed ranking; put it in the schematic | Yes; `fig5_factor_protein_to_rho_v3` panel D (CD99 forward, CD48 reversed); full report Figure 20 panel D and paragraph "What the sign means"; core page 8, Figure 7A |
| A-20261010-24 | put them into the report | Questions report version 3.1: full report section 5 (Figures 19, 20, 22, Table 3) and, compact, core pages 7-8 (Figure 6 bottom, Table 1, Figure 7; the core stays at 12 pages; the full-size appendix, 15 pages, is drafted and offered to MX) |

## The messages (verbatim)

> I am also looking at the "New in this version: factor x cell type and surface protein for every card (D27)". I feel like figure 19 could be useful to be included in the email, but I am a bit confused about the cell type counts, the numbers on the y-axis doesn't add up to a number that I've seen before, this is all patient's all cells? Why 643 HSC/MPP2? What's mean oriented PC8 score. I think we should include illustration showing the matrix decomp: PC8's cell score for all malignant cells is a vector of length (1569) and every entry is the PC8's score for that cell. And spearman rho is just the correlation of the ranking of that with CD34's cell score or another surface protein's cell score. Table 3 also looks good to show these rho. The within (k/7), k means the number of patients that has the same rho sign, right? In type is a median value which is an aggregated stats. My observation here would be that CD99's correlation between malignant cell and PC8 seems to not diminish when we check it per patient per cell type (stayed as high as 0.150), which could mean something, but later in Fig 20 when we include the isotype's band, the plot concludes that CD99's malignant association is not robust hmmm? But CD48 is somehow more robust by exceeding the isotype's band. And FA7 malignant cells' surface protein correlation is finding two different surface protein compare to its healthy counter part. Lastly, do these signs mean anything, e.g. CD22 have a negative value for both PC8 and FA7 on the normal cells. And CD34 has a positive value. So does that mean CD22 is less associated with normal cells? Or just that the latent factors' cell score vector is like the reversed rank with CD22?

> I pushed as you recommended. I think we should also include "CD116−0.287, CD33−0.272, CD146 +0.270" from PC8's other strong malignant correlates. I don't see the schematics (F5?, it's not figure 5), maybe I am not finding it. I actually don't find all the F figures you just created. So the sign means how forward and reversed the rank of the factor cell score against the ADT protein cell score? We should include that in the schematics F5. Why not just put those into the report?
