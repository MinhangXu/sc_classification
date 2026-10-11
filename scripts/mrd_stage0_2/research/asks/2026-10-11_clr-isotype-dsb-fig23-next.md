---
date: 2026-10-11
source: user
asked_by: user
capture: verbatim
original_location: one chat message from the user on 2026-10-11 in a claude-code session, after reading questions report v3.2 (full report section 5, Figures 21-27); recorded here the same day
---

# 2026-10-11 questions on CLR, isotypes, dsb, Figure 21's P03, Figure 23 and how to make the protein results stronger

Each quoted text is verbatim (wording and typos kept), split from one message, which is quoted whole below. Figure numbers are those of questions report version 3.2 (2026-10-10): Figure 21 = fig05e_rho_anchors, Figure 23 = fig05f_modality_all, Figure 27 = fig05r_two_checks_depth.

## CLR, isotypes, dsb and the total-ADT adjustment (D29, W5.3)

- [A-20261011-01] "We should specify what does CLR mean?" -> W5.3, D29
- [A-20261011-02] "Also previous the explanation for Isotype antibody as control was "they bind to nothing specific". I wish for a bit more explanation such that when I email my teammate, I presenting it correctly (succinct)." -> W5.3, D29
- [A-20261011-03] "When you say the isotypes mostly track each cell's total ADT count I tried to look at fig27 and it's not showing isotype's rho with cell total ADT count, can we have a figure for it?" -> W5.3, D29
- [A-20261011-04] "Why would we want to rerun on dsb values? Please explain." -> W5.3, W5.8, D29
- [A-20261011-05] "Then the normalization technique seems to change things up a bit in Fig27, but not RTK2071 so much. Should we use the plots/figures from the new normalization technique?" -> W5.3, W5.8, D29

## Figure 21: P03 near zero (D27, D29)

- [A-20261011-06] "In "The two numbers taken apart" Figure 21, you referenced Table 3, and how P03 sits near 0 on a few of them, how can we interpret this observation?" -> W5.3, D27, D29

## Figure 23 (D27, W5.3)

- [A-20261011-07] "Then talk about Fig23, a, b) shows that: In normal cells, CD123+ has ~0.7 |ρ| wrt "innate immune context"'s promax f7, either on the patient specific level or on the across patients level. * The diamond and median within patient is very close. And within cell type |ρ| drops a lot, which means that CD123+ is more like a cell type identifier. Using CD123+ as an example. But what does that say about promax f7? Is it even useful in classification?" -> W5.3, D27
- [A-20261011-08] "What does "(sign of the within patient median)" on the x-axis mean?" -> W5.3, D27
- [A-20261011-09] "In malignant cells, some protein and factor pairs have very strong patient specific effect that the median has significantly higher |Rho| than pooled. Would you say/show if malignant cells are more often to have this than normal cells? Does this have any useful biological insight?" -> W5.3, W1, D27
- [A-20261011-10] "Panel C sort of answers the question of mal vs. normal by within patient median and pooled rho, should we look at the cell type within patient median vs. within patient median?" -> W5.3, D27
- [A-20261011-11] "What's the SImpson's paradox patter you are talking about?" -> W5.3, D27

## Scope and next steps (W5, W2, W3)

- [A-20261011-12] "I also see your reasoning on "Chaining the two does not work: correlations are not transitive, and the direction in which malignant cells vary among themselves need not be the direction that separates them from healthy cells." And the safest conclusion would be to say "Among malignant cells, the further a cell sits toward the malignant-leaning end of the PC8 axis, the less surface CD63 it tends to carry." this is a statement on whether we can combine the factors' malignancy relationship with the surface proteins. While this is still an appropriate statement with some usefulness for sure. If this is decided to be useful in the future by my team, we should be able to broaden the scope to look into other factors that are selected by L1, ahh the stage 2 story of why FA7 and PC8 isn't fully flushed out in my mind yet, that's okay. Like what we did with Figure 23, but delve into more details. And improve Fig 23 from my comments above." -> W5.3, W5.9, D27
- [A-20261011-13] "I just feel like after stating the results really accurately makes the impact feel super weak, which is okay for a scientific step, but how can we make improvement from this. Definitely include the factors in the DR step as a model that we are going to try?" -> W5, W2, W3, W2.10, W3.9

## Where each item is answered

In devlog 2026-10-11_clr-isotype-p03-fig23-next and questions report version 3.3 (full report section 5). Figures 19-27 kept their numbers; Figure 28 (`fig05t_isotype_depth`) and Table 4 (`tab:05p03`) are new; Figure 23 now runs over two pages (panels a-d, then e-h).

| Ask | Short | Answered in |
|---|---|---|
| A-20261011-01 | what CLR means | box "Three terms in plain words" (before "How to read rho"); glossary "ADT (CLR)"; core page 8 one sentence; email_draft_v4 |
| A-20261011-02 | isotype controls, succinctly and correctly | same box ("Isotype controls" and the email sentence); glossary "Isotype reference"; email_draft_v4 |
| A-20261011-03 | a figure for isotype rho with total ADT | Figure 28 (A: one isotype, raw vs CLR; B: every isotype x patient; C: PC8 vs total ADT; D: isotype x PC8 with and without) |
| A-20261011-04 | why rerun on dsb | same box ("dsb"); glossary "dsb"; email_draft_v4 question 2 |
| A-20261011-05 | switch to the adjusted figures? RTK2071 | paragraph "Should we switch to the total-ADT-adjusted figures?" (no: use as a filter; dsb as primary when available; RTK2071 does not run through total counts) |
| A-20261011-06 | why P03 sits near 0 | paragraph "Why P03 sits near 0" and Table 4 (cell-type share of PC8 in P03 4.6%; no total-count path; CD82/CD99 do show; CD63 the exception) |
| A-20261011-07 | CD123 and promax f7; useful for classification? | "Every card", paragraph "Reading one row of panels a and b" (cell type explains 80% of f7; lift 14.4 pp in P02 only); Figure 23h (cell-type share vs median lift, Spearman +0.07) |
| A-20261011-08 | the x-axis label | same paragraph; new axis label on Figure 23a-b and caption |
| A-20261011-09 | within >> pooled more common in malignant cells? insight? | paragraph "Strong within patients, weak pooled" (39% vs 2.9%; isotypes 32%; large gaps 14% vs 0.4%); Figure 23e |
| A-20261011-10 | inside type vs within patient | new Figure 23d and paragraph "Inside cell types against within patients" |
| A-20261011-11 | Simpson's paradox | paragraph "What Simpson's paradox means here"; Figure 23e |
| A-20261011-12 | broaden to the L1-selected factors; improve Figure 23 | Figure 23 revised (a-h); subsection "Where the protein work goes next", item 4; W5.9 |
| A-20261011-13 | the results feel weak; protein in the DR step? | subsection "Where the protein work goes next" (W3.9 protein in the classifier, W2.10 MOFA+/totalVI, W5.10 contrast with inference, W5.9 drivers) |

## The message (verbatim)

> We should specify what does CLR mean? Also previous the explanation for Isotype antibody as control was "they bind to nothing specific". I wish for a bit more explanation such that when I email my teammate, I presenting it correctly (succinct). When you say the isotypes mostly track each cell's total ADT count I tried to look at fig27 and it's not showing isotype's rho with cell total ADT count, can we have a figure for it? Why would we want to rerun on dsb values? Please explain. Then the normalization technique seems to change things up a bit in Fig27, but not RTK2071 so much. Should we use the plots/figures from the new normalization technique? In "The two numbers taken apart" Figure 21, you referenced Table 3,  and how P03 sits near 0 on a few of them, how can we interpret this observation?
>
> Then talk about Fig23, a, b) shows that: In normal cells, CD123+ has ~0.7 |ρ| wrt "innate immune context"'s promax f7, either on the patient specific level or on the across patients level. * The diamond and median within patient is very close.
>
> And within cell type |ρ| drops a lot, which means that CD123+ is more like a cell type identifier. Using CD123+ as an example. But what does that say about promax f7? Is it even useful in classification? What does "(sign of the within patient median)" on the x-axis mean?
>
> In malignant cells, some protein and factor pairs have very strong patient specific effect that the median has significantly higher |Rho| than pooled. Would you say/show if malignant cells are more often to have this than normal cells? Does this have any useful biological insight? Panel C sort of answers the question of mal vs. normal by within patient median and pooled rho, should we look at the cell type within patient median vs. within patient median? What's the SImpson's paradox patter you are talking about? I also see your reasoning on "Chaining the two does not work: correlations are not transitive,
> and the direction in which malignant cells vary among themselves need not be the direction that
> separates them from healthy cells." And the safest conclusion would be to say "Among malignant cells, the further a cell sits toward the malignant-leaning end of the PC8 axis,
> the less surface CD63 it tends to carry." this is a statement on whether we can combine the factors' malignancy relationship with the surface proteins. While this is still an appropriate statement with some usefulness for sure. If this is decided to be useful in the future by my team, we should be able to broaden the scope to look into other factors that are selected by L1, ahh the stage 2 story of why FA7 and PC8 isn't fully flushed out in my mind yet, that's okay. Like what we did with Figure 23, but delve into more details. And improve Fig 23 from my comments above.
>
> I just feel like after stating the results really accurately makes the impact feel super weak, which is okay for a scientific step, but how can we make improvement from this. Definitely include the factors in the DR step as a model that we are going to try?
