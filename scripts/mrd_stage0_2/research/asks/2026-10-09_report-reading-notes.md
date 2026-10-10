---
date: 2026-10-09
source: user
asked_by: user
capture: verbatim
original_location: the user's chat message of 2026-10-09 written while reading report v1 (scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/report_v1_2026-10-08.pdf), a record of their trace of thought from the Answers at a glance table onward; items 1-16 of the message map to A-20261009-01..16 in order; the message's opening and closing lines are quoted below the list
---

# 2026-10-09 reading notes on report v1 (the user's trace of thought)

The user asked that these questions NOT be answered in chat: the report itself should let them find the answers (report v2), the plots should be fixed, and the next steps should start. Each line below is the user's text, verbatim (wording and typos kept); figure numbers refer to report v1.

- [A-20261009-01] "In general, what does "descriptive" mean in the "How sure" column?" -> THEME, W4.1, W4.2
- [A-20261009-02] "What's a leverage test? So it sounds like in stage 1 DR, wrt P02, P03 and P04, doesn't affect too much, but their malignant prevalence do dominate in stage 2 classification. Why would it be open until allograft and apoptosis are tested? How about other stage 0 panels? Test what? Should we test them too?" -> W2.4, D26, D17, D28, W1.9, H-S1-05
- [A-20261009-03] "How does re-weight work? And this is just piloted on IFN-alpha, we might want to do more. What does the percentage mean? How do we target stage 1 geometry or stage 2 labels?" -> W2.3, D17, H-S1-05
- [A-20261009-04] ""Aggregate" is a bit abstract, what does it mean quantitatively? That's an interesting observation that P03's only private axis is in the broad gene bags, none of the 41 biological representations." -> D14, W1.2, H-S1-04, H-S1-07, W1.4
- [A-20261009-05] "I will look at the PCA40 vs. FA40 comparison. I will need to understand the L1 entry rule and how that affects the results." -> W3.2, D19, H-S2-08
- [A-20261009-06] "What does "Neither is identifiable alone" in "PC8: two opposite-sign gene groups. What is it?"?" -> W5.2, F9, D18, H-I-03
- [A-20261009-07] "Give an account for 1810 factor cards, they are the factors coming from all stage0 and stage1 combined, right?" -> W5.1, D18, W7.2
- [A-20261009-08] "Correlate factors with cell type and surface protein, let's think about this as a relative high priority, can we do it for the other factors too? And I should be able to explain the rho + 0.251 and rho + 0.222 for CD99 in malignant cell?" -> W5.3, D27
- [A-20261009-09] "The transductive experiment with IFN-alpha is great, I will look into more details." -> W2.6, D17, H-S1-06
- [A-20261009-10] "AUPRC vs. number of cells or patients is a concept that I will need to look at the plot more, for right now, I am primarily confused about why we would retune the C?" -> W6.2, D15, D15b, D19, H-S2-06
- [A-20261009-11] "Looking at the cohort and the yardstick section (I really like these figures here and I also looked at Figure 2, the lift wrt IFN-alpha PCA40 probe, here's my question "We got 4 patients passing the 100 malignant cell rule. P02, P03, P04, P09. We need to think if P01, P06 and P07 are still relevant to use. For some reason P07 still does pretty decent lift in the IFN-alpha case."" -> W8, W8.2, W8.5, W1.5
- [A-20261009-12] "Some of these results really point at the fact that the off-ratio between healthy and malignant cells are a big problem, though we have to think more carefully, if stage 1 is transductive, for P09, then P09's healthy cell contributed quite more to the overall DR space (including variance explained or how each factor is constructed), I did try some donor cell downsampling before, and I feel like there's a real opportunity here. So we just gotta show not only it's the total number of malignant cells giving us a hard time, it's also a ratio problem. So if we do a proper systematic downsampling of the healthy cell stratified by their cell types knowing that they represent the healthy population very well with the trajectory, can we just down-sampling them (and yea more rigorously tested and better code-wise developed/implemented), I guess this whole repo could use an update with the more intelligent agents. The update shouldn't be drastic, but gradual and understandable from my end. Also be super relevant to the tasks/analyses that we are doing. I think "Machine Learning in MRD single cell data, just downsample the healthy cells" could be a great catch line as a little paper. Or a section of the paper. My main concern is that if the downsampling of healthy cells improved the classification, we need to understand why? And speaking of paper, in thinking about repo update, let's have the context ready to develop the paper's results in parallel of all the runs planned and analyses planned." -> W8, W8.2, W8.6, W8.7, W8.8, H-S2-09, W2.3, W4.6, D10, D11, H-S2-04, W0.13, W0.14, THEME
- [A-20261009-13] "Shaping the variance in section 2 "do the highest-prevalence patients shape the factors", I like the 1) and 2). especially 2) is very good control in a perturbation like in-silico experiment in this case test how patients "shape" the factors, though I still need to understand the metrics." -> W2.4, D26, D16, D17, D28, W1.9, H-S1-05
- [A-20261009-14] "I don't understand figure 3, I feel like putting score energy with the malignant cells and all cells are a bit weird, but I feel like maybe just not explained very well. There's score energy, there's ratios? What's 1.11 about? Is it shown in Figure 3? In a new run, let's definitely subset the patients to do the whole stage0-2 pipeline, let's definitely not include patients that're just in the cohort and contribute no malignant cells at least. For figure 4, the axis loss x-axis need more explanation." -> W2.4, D26, D16, D17, D28, W1.9, H-S1-05, W8, W8.2, W8.3, W8.5
- [A-20261009-15] "I see the final conclusion is not very conclusive on this. So definitely try to fix the plots and my feedback above, and I like the next step, let's start them." -> W2.4, D26, W2.3, D28, W1.9, H-S1-05
- [A-20261009-16] "As a result of data driven statistical test for whether to aggregate patients, we say "On the current bases, patients' factor-score distributions overlap enough to pool them", okay. I am just a bit confused about the plots still. I guess we are just figuring out that pretty much all patients contribute to all factors? Except a few cases? How would this help with the downsampling idea above? I guess the detailed methods is something I am confused about unless visualized. How do we compare each patient's score distribution with the other patients' separately for normal and malignant? Then I get you are trying really hard to point out stuff with figure 5 and defining "aggregate, separate-substype candidate, location shift"." -> D14, W1.2, D20, H-S1-04, W8

## Opening and closing lines of the message (verbatim)

> I install pytest on the sc-classification-rapids-py311 env. Great, I am going over the report here are my questions from start reading the "Answers at a glance", which I will not have you answer it as I will then try to read more of the report to potentially answer my own questions so in a way, this is to document my trace of thought:

> Let's work on this while I go over the rest of the report. All the new proposed run idea like downsampling, let's include that when we have the new data from team.

The closing line is a user decision that belongs with A-20261009-12: new proposed runs such as healthy-cell downsampling wait for the new patient data (W8). The second user decision is in A-20261009-14: new runs exclude patients that contribute no malignant cells. Both are recorded in devlog 2026-10-09_report-reading-notes. The opening line closes W0.4 (pytest in the lab env).

## Where report v2 should answer each item

Route tokens above can only name tasks, analyses, hypotheses or THEME; this table adds the report place. Section and figure numbers are those of report v1 (sections/ files in parentheses).

| Ask | Topic (short) | Report v1 place the user was reading | What report v2 needs there | Tasks |
|---|---|---|---|---|
| A-20261009-01 | meaning of "descriptive" | Answers at a glance, How sure column (00) | a one-line definition at the top of the glance table, plus a glossary entry (Appendix A, A1) | THEME §6, W4.1, W4.2 |
| A-20261009-02 | leverage test; why open; other panels | glance row for §2; §2 (02) | define the leverage test where it is first used; say what "open" waits for and which panels are next | W2.4 (D26), W1.9 (D28) |
| A-20261009-03 | how re-weighting works; the percentages; Stage 1 vs Stage 2 target | glance sub-row for §2; §2 (02) | an explainer for equal-patient weighting, what each percentage is a share of, and the two targets | W2.3 |
| A-20261009-04 | "aggregate" quantitatively; P03 private axis | glance row for §3; §3 (03) | the overlap quantity and the gates in one box next to the call | W1.2 (D20), W1.4 (D25) |
| A-20261009-05 | PCA40 vs FA40; L1 entry rule | §4 (04) | the entry rule with one worked example | W3.2 (D19) |
| A-20261009-06 | "Neither is identifiable alone" | glance row for §5; §5 (05) | what identifiability means for a single factor, in words, then the formal statement | W5.2, H-I-03 |
| A-20261009-07 | what the factor cards count | glance sub-row for §5; §5 (05) | the count broken down by panel × method × K | W5.1 (D18), W7.2 |
| A-20261009-08 | factor × cell type and surface protein for all factors; the CD34/CD99 rho | glance sub-row for §5; §5 (05) | the rho definition (which cells, which orientation); D27 results for all cards | W5.3 (D27) |
| A-20261009-09 | transductive vs inductive on IFN-alpha | §6 (06) | (no confusion recorded) | W2.6 |
| A-20261009-10 | why retune C when n changes | glance row for §7; §7 (07); §4 (04) | the C-scaling argument at the first place a curve is shown | W6.2, W3 (D19) |
| A-20261009-11 | the 100-malignant-cell rule; P01, P06, P07 | §1, Figure 2 (01) | where the rule comes from and what keeping or dropping P01, P06, P07 changes | W8.5 (patient-inclusion decision), W8.2, W1.5 |
| A-20261009-12 | healthy:malignant ratio; stratified healthy-cell downsampling; paper | §1-§3 | (new run, waits for the new data) | W8.2 (pre-registration), W2.3, W4.6, W0.13 (paper context), W0.14 |
| A-20261009-13 | the two section-2 tests and their metrics | §2 (02) | metric definitions before the figures | W2.4 (D26), W1.9 (D28) |
| A-20261009-14 | Figure 3 score energy and the ratio; Figure 4 axis loss; new-run patient subset | §2, Figures 3-4 (02) | redrawn Figure 3 with the ratio shown; the axis-loss axis explained; new runs drop patients with no malignant cells | W2.4, W1.9, W8.2, W8.3 |
| A-20261009-15 | §2 conclusion not conclusive; start the next steps | §2 (02) | the next steps run (D26, D28) and the conclusion restated | W2.4 (D26), W2.3, W1.9 (D28) |
| A-20261009-16 | aggregation test plots; normal vs malignant comparison; link to downsampling | §3, Figure 5 (03) | a visual walk-through of one patient-vs-rest comparison, separately for normal and malignant | W1.2 (D20), W8 |
