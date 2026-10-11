---
date: 2026-10-10
source: user
asked_by: user
capture: verbatim
original_location: one chat message from the user on 2026-10-10 (late evening) in a claude-code session, written while building slides from the questions report's section 5 (full report Figures 20-23); recorded here the same day
---

# 2026-10-10 questions on the sign panel, CD63, the isotype yardstick and "Every card" (Figure 23)

Each quoted text is verbatim (wording and typos kept), split from one message, which is quoted whole below. Full report figure numbers are those of version 3.1 (2026-10-10): Figure 20 = fig05h/fig05i (from a factor and a protein to one rho; panel D = what the sign means), Figure 21 = fig05e_rho_anchors (one pooled rho, taken apart), Figure 22 = fig05j_two_checks, Figure 23 = fig05f_modality_all (every card).

## Figure 20 panel D and the isotype yardstick (D27, W5.3)

- [A-20261010-26] "On panel D in the full report "What the sign means: do the two ranks run the same way or in reverse?". It says "two rank vectors of panel A" which is helpful, but from just looking, it looks like the grey mark is very randomly distributed, it kinda correspond to something you said earlier, "CD99, and partly CD63 and CD82. Among cells of the same type in the same patient, the ones further toward the malignant end of PC8 still have more CD99 (less CD63, more CD82). Cell type can't explain that, because it's held fixed. So these are the candidates for something beyond cell identity.", but their signals (Rho) are still weak and close to the isotype noise (vertical grey shades)." -> W5.3, D27
- [A-20261010-27] "I need Bachireddy Lab's help on whether the isotype protein are a good control/yardstick to say we need proteins have higher rho than isotypes." -> W5.3, W8
- [A-20261010-28] "We should emphasize the each grey dot is one of the 1569 malignant cells using textbf{} from latex." -> W0, D27

## CD63 (D27, W5.3)

- [A-20261010-29] "I then used the other page which is Figure 22 on the full report in my slides. So basically we can say look into CD63 and CD82." -> W5.3, D27
- [A-20261010-30] "I am going through the FA7 section: I agree with "The proteins that track a factor across a population are largely those that separate the cell types present in it, so different mixes give different lists.". I guess the only interesting thing is that it reinforces on CD63. Basically confirming that CD63 have negative rho on malignant cells doesn't matter the cell type (well as much), Can we see the detailed rank like panel D of figure 20 for CD63?" -> W5.3, D27
- [A-20261010-31] "And we are saying "Among the malignant cells, cells rank high on PC8 tend to rank low on CD63." And this is built on top of the fact that PC8 is pretty decent at classifying malignant vs. healthy evaluated by the AUPRC score. So if we stretch the conclusion a bit, it's saying that malignant cells in these patients have lower CD63 expression or something, how can I say this better?" -> W5.3, D27
- [A-20261010-32] "Let's also visualize CD63 like figure 21." -> W5.3, D27

## "Every card: factor x surface protein and cell type" and Figure 23 (D27, W5.3)

- [A-20261010-33] "Then in Every card: factor x surface protein and cell type section, I am confused on 20274 of 289961 card x marker pairs, where do those numbers come from? I get that in Fig23 we are trying to show a whole cohort level all DR methods with all proteins?" -> W5.3, D27
- [A-20261010-34] "The passage start by describing panel C which shows that x-axis is how patient specificity matters, basically the normal cells are not very patient specific and malignant are more patient specific, but it's coming from the rho angle, which means what? Please include my reasoning trace." -> W5.3, D27
- [A-20261010-35] "We are not including PCA, probably because it's not showing up also I don't see the need for color if the stage 1 name is included on the y-axis label, which is convenient? And this doesn't care about stage 2, right? This is evaluating all the factors from stage 1 with all surface protein?" -> W5.3, D27
- [A-20261010-36] "Let's have a scientific statement we can make about these in addition to what you said in the writing section "Every card: factor × surface protein and cell type"." -> W5.3, W0.13, D27
- [A-20261010-37] "Lastly wrt d) and e) I need to understand what's eta^2 of patient? We are showing all stage 1 possible combination? If above the diagonal meaning cell type explain a lot then we would narrow the points to below the diagonal to be worthy of investigation." -> W5.3, D27

## Where each item is answered

In devlog 2026-10-10_d29-cd63-depth-and-every-card (analysis D29, slug `protein_depth_contrast_v1`) and in the questions report, version 3.2 (full report section 5; figure numbers below are version 3.2's: Figures 19-23 kept their numbers, the new ones are 24-27). `.png`/`.pdf` files and `report_full.pdf` are gitignored (W0.15); a clone has `report_core.pdf` only.

| Ask | Short | Answered in |
|---|---|---|
| A-20261010-26 | the gray cloud looks random; weak rho near the isotype bands | Figure 20 caption (why the cloud looks random: rho^2 about 4-5%); subsection "CD63, followed up", paragraph "What MX read off panel D and Figure 22" |
| A-20261010-27 | ask the Bachireddy lab whether isotypes are a fair yardstick | paragraph "Is the isotype a fair yardstick?" and Figure 27 (total-ADT path, isotype bands shrink when total counts are held fixed, RTK2071); "For the Bachireddy lab" (four questions); `team_updates/2026-10-09_protein_association_for_pavan/email_draft_v4.md` (two of them, for MX to send) |
| A-20261010-28 | bold "each gray dot is one of the 1,569 malignant cells" | Figure 20 caption (\textbf); also Figure 24's caption |
| A-20261010-29 | "look into CD63 and CD82" | CD63 yes, CD82 less so: Figures 25 and 27 (CD82 halves inside cell types once total counts are held fixed) |
| A-20261010-30 | CD63 rank plot like Figure 20D; FA7 reinforces CD63 | Figure 24 (all malignant cells, P09, P09 HSC/MPP2, P03); FA7 is not independent (PC8 and FA7 scores correlate 0.62 among malignant cells) |
| A-20261010-31 | "malignant cells have lower CD63": how to say it better | paragraph "Do malignant cells carry less CD63?", Figure 26 (direct contrast, no factor), the team box sentence; CD99 as the counterexample to the chain |
| A-20261010-32 | CD63 like Figure 21 | Figure 25 (CD63 x PC8, CD63 x FA7, CD82 x PC8, with total-ADT-fixed rows) |
| A-20261010-33 | where 20,274 of 289,961 come from | "Every card", paragraph "Where the numbers come from" (1,801 cards x 161 proteins) and Figure 23 caption |
| A-20261010-34 | panel c "from the rho angle" | "Every card", paragraph "What panel c says" (with MX's reading quoted) |
| A-20261010-35 | PCA missing; colour redundant; Stage 2's role | "Every card" (PCA close behind, not absent; Stage 2 enters only via the bases, the sign and the driver flags); Figure 23 drops the method colours |
| A-20261010-36 | a scientific statement | box "A statement we can make" after "Every card" |
| A-20261010-37 | eta^2 of patient; keep points below the diagonal? | "Every card", paragraph "What panels d and e plot" (definition; below the diagonal = patient axes; look at the bottom-left instead); Figure 23 region labels |

## The message (verbatim)

> I been putting together my reports with the figure from a factor and a protein to one number (Spearman Rho) and what its signs mean? I found panel C to be very useful being able to see all 3 levels of aggregation in the same plot and be able to compare them. On panel D in the full report "What the sign means: do the two ranks run the same way or in reverse?". It says "two rank vectors of panel A" which is helpful, but from just looking, it looks like the grey mark is very randomly distributed, it kinda correspond to something you said earlier, "CD99, and partly CD63 and CD82. Among cells of the same type in the same patient, the ones further toward the malignant end of PC8 still have more CD99 (less CD63, more CD82). Cell type can't explain that, because it's held fixed. So these are the candidates for something beyond cell identity.", but their signals (Rho) are still weak and close to the isotype noise (vertical grey shades). I need Bachireddy Lab's help on whether the isotype protein are a good control/yardstick to say we need proteins have higher rho than isotypes. We should emphasize the each grey dot is one of the 1569 malignant cells using textbf{} from latex.
> I then used the other page which is Figure 22 on the full report in my slides. So basically we can say look into CD63 and CD82. I am going through the FA7 section: I agree with "The proteins that track a factor across a population are largely those that separate the cell types present in it, so different mixes
> give different lists.". I guess the only interesting thing is that it reinforces on CD63. Basically confirming that CD63 have negative rho on malignant cells doesn't matter the cell type (well as much), Can we see the detailed rank like panel D of figure 20 for CD63? And we are saying "Among the malignant cells, cells rank high on PC8 tend to rank low on CD63." And this is built on top of the fact that PC8 is pretty decent at classifying malignant vs. healthy evaluated by the AUPRC score. So if we stretch the conclusion a bit, it's saying that malignant cells in these patients have lower CD63 expression or something, how can I say this better? Let's also visualize CD63 like figure 21.
>
> Then in Every card: factor x surface protein and cell type section, I am confused on 20274 of 289961 card x marker pairs, where do those numbers come from? I get that in Fig23 we are trying to show a whole cohort level all DR methods with all proteins? The passage start by describing panel C which shows that x-axis is how patient specificity matters, basically the normal cells are not very patient specific and malignant are more patient specific, but it's coming from the rho angle, which means what? Please include my reasoning trace. We are not including PCA, probably because it's not showing up also I don't see the need for color if the stage 1 name is included on the y-axis label, which is convenient? And this doesn't care about stage 2, right? This is evaluating all the factors from stage 1 with all surface protein? Let's have a scientific statement we can make about these in addition to what you said in the writing section "Every card: factor × surface protein and cell type". Lastly wrt d) and e) I need to understand what's eta^2 of patient? We are showing all stage 1 possible combination? If above the diagonal meaning cell type explain a lot then we would narrow the points to below the diagonal to be worthy of investigation.
