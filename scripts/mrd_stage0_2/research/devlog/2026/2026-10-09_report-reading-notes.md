---
id: 2026-10-09_report-reading-notes
date: 2026-10-09
type: meeting
title: The user's reading notes on report v1 (16 questions, recorded not answered) routed to report v2 and tasks; two run decisions; pytest in the lab env
asked_by: user
asks: [A-20261009-01, A-20261009-02, A-20261009-03, A-20261009-04, A-20261009-05, A-20261009-06, A-20261009-07, A-20261009-08, A-20261009-09, A-20261009-10, A-20261009-11, A-20261009-12, A-20261009-13, A-20261009-14, A-20261009-15, A-20261009-16]
implemented_by: claude-code
questions: [Q1, Q2, Q3, QI, meta]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: [W0.4]
advances: [W0.13, W0.14]
claim_level: none
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the report directory, the D15b full-design entry and this overlay round are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored  # the notes arrived and were recorded on 2026-10-09
supersedes: []
---

## Question
The user's notes of 2026-10-09, written while reading report v1 from the "Answers at a glance" table onward. All 16 items are recorded verbatim, in order, in `scripts/mrd_stage0_2/research/asks/2026-10-09_report-reading-notes.md` (A-20261009-01..16), together with the message's opening and closing lines. The user framed them as a record of a trace of thought, not as questions to answer in chat:

> "Great, I am going over the report here are my questions from start reading the "Answers at a glance", which I will not have you answer it as I will then try to read more of the report to potentially answer my own questions so in a way, this is to document my trace of thought:"

| Ask | Topic (short) | Routed to |
|---|---|---|
| A-20261009-01 | what "descriptive" means in the How sure column | THEME, W4.1, W4.2 |
| A-20261009-02 | what a leverage test is; why open until allograft and apoptosis; other panels | W2.4, D26, D17, D28, W1.9, H-S1-05 |
| A-20261009-03 | how re-weighting works; the percentages; Stage 1 geometry vs Stage 2 labels | W2.3, D17, H-S1-05 |
| A-20261009-04 | "aggregate" quantitatively; P03's private axis only in broad bags | D14, W1.2, H-S1-04, H-S1-07, W1.4 |
| A-20261009-05 | PCA40 vs FA40; the L1 entry rule | W3.2, D19, H-S2-08 |
| A-20261009-06 | "Neither is identifiable alone" for PC8 | W5.2, F9, D18, H-I-03 |
| A-20261009-07 | an account of the factor-card count | W5.1, D18, W7.2 |
| A-20261009-08 | factors vs cell type and surface protein for all factors (high priority); the CD34/CD99 rho | W5.3, D27 |
| A-20261009-09 | the IFN-alpha transductive vs inductive experiment | W2.6, D17, H-S1-06 |
| A-20261009-10 | why C is retuned on the learning curves | W6.2, D15, D15b, D19, H-S2-06 |
| A-20261009-11 | the malignant-cell rule; whether P01, P06, P07 stay; P07's IFN-alpha lift | W8, W8.2, W8.5, W1.5 |
| A-20261009-12 | healthy:malignant ratio; cell-type-stratified healthy-cell downsampling; paper context; gradual repo updates | W8, W8.2, W8.6, W8.7, W8.8, H-S2-09, W2.3, W4.6, D10, D11, H-S2-04, W0.13, W0.14, THEME |
| A-20261009-13 | the two section-2 tests and their metrics | W2.4, D26, D16, D17, D28, W1.9, H-S1-05 |
| A-20261009-14 | Figure 3 score energy and its ratio; Figure 4 axis loss; new runs drop patients without malignant cells | W2.4, D26, D16, D17, D28, W1.9, H-S1-05, W8, W8.2, W8.3, W8.5 |
| A-20261009-15 | section 2 not conclusive; fix the plots; start the next steps | W2.4, D26, W2.3, D28, W1.9, H-S1-05 |
| A-20261009-16 | the aggregation plots; normal vs malignant comparison; link to downsampling | D14, W1.2, D20, H-S1-04, W8 |

## Why it matters
The user's trace of thought while reading v1 is the specification for report v2: every place where the notes show confusion needs a clearer explanation and a better plot in the report, where the reader looks for it, rather than an answer in chat. The notes also start new work: the section-2 next steps (D26 leverage on allograft and apoptosis, D28 cell-type-stratified score shares) and factor-modality associations for all cards (D27). They also state two decisions that constrain every new run (below).

## Estimand / null
n/a (meeting record; no analysis).

## Implementation
- Asks file: `scripts/mrd_stage0_2/research/asks/2026-10-09_report-reading-notes.md`. Capture is verbatim (wording and typos kept); one line per item, numbered in message order. The front-matter `source` is `user`, because the asks vocabulary has no `report-reading-notes` value; that name lives in the file name.
- Routing, per research/devlog/README.md §4. Route tokens can name only tasks, analyses, hypotheses or THEME, so the asks file adds a table with the report v1 place each item was read at and what report v2 needs there:
  - explanation-only items (A-20261009-01, -03, -04, -06, -07, -10) go to their report sections (glance table, glossary, sections 2-5 and 7) and to the task that owns the underlying analysis;
  - the section-2 cluster (A-20261009-02, -13, -14, -15) goes to W2.4 with D26 (leverage on allograft and apoptosis), W1.9 with D28 (cell-type-stratified score shares) and H-S1-05;
  - re-weighting (A-20261009-03) to W2.3; PCA40 vs FA40 (A-20261009-05) to W3.2 with D19; factor-modality associations (A-20261009-08) to W5.3 with D27; the inductive pilot (A-20261009-09) to W2.6; the aggregation test (A-20261009-16) to D14 and W1.2 (D20);
  - patient inclusion (A-20261009-11) and the new-run patient subset (A-20261009-14) to W8.5 (the patient-inclusion decision: the zero-malignant exclusion is decided; the evaluation threshold and the fate of P01, P06 and P07 are open for the user), to W8.2, the pre-registration that fixes the support rule, and to W8.3 (the opt-in patient filter that applies the exclusion);
  - healthy-cell downsampling (A-20261009-12) to W8.2 (pre-registration) and H-S2-09, then W8.6 (runner), W8.7 (the run, on the new data) and W8.8 (its paper arc), with W2.3, W4.6, D10, D11 and H-S2-04 as the prior donor-downsampling work it builds on. Its paper and repo-update parts go to the new W0.13 (living paper context under scripts/mrd_stage0_2/research/paper/) and W0.14 (gradual, ask-driven updates).
- User decisions stated in the notes (recorded here; W8.2 makes them binding for the new cohort):
  1. New runs exclude patients that contribute no malignant cells (W8.5 records the remaining open choices): "In a new run, let's definitely subset the patients to do the whole stage0-2 pipeline, let's definitely not include patients that're just in the cohort and contribute no malignant cells at least." (A-20261009-14)
  2. New proposed runs, such as cell-type-stratified healthy-cell downsampling, wait for the new patient data: "All the new proposed run idea like downsampling, let's include that when we have the new data from team." (closing line, recorded with A-20261009-12)

  Analyses on the stored EXP artifacts (D26, D27, D28, report v2) are not new proposed runs and start now, as the user asked in A-20261009-15.
- Environment: the user installed pytest in sc-classification-rapids-py311 ("I install pytest on the sc-classification-rapids-py311 env."). This closes W0.4. The full suite was run in the agent shell with that env's python and pytest (`CONDA_PREFIX` set, BLAS threads capped at 4). Before `research_index.py write` the only failure was the real-repo check test, because STATUS.md was stale after the D26-D28 reservations; after `write` the suite passes. Rerun on 2026-10-10 after this round's context edits (BLAS threads 4): 295 passed, 1 skipped.
- Context edits:
  - AGENTS.md §8: tests now run with the lab env's pytest; the "until W0.4 lands" workaround is removed.
  - AGENTS.md §6: a pointer to the living paper context (README.md, OUTLINE.md under research/paper/): when a result changes a claim, its devlog id goes into the matching OUTLINE results subsection.
  - W0 README: a status update, the paper-context pointer, and backlog lines W0.13 and W0.14.

## Artifacts
none

## Results
none (question set)

## Claim gate
Supports (descriptive): nothing; this entry records questions, routes and two user decisions about future runs.
Does not support: answers to any of A-20261009-01..16 (those come from report v2 and the routed analyses); any claim that healthy-cell downsampling improves classification (not yet run; it waits for the new data and needs C retuned per condition, AGENTS.md R5).

## Next steps
- W0.14: report v2 (planned devlog id 2026-10-09_report-v2) adds the explanations and plot fixes at the places listed in the asks file's table: the meaning of "descriptive", the leverage test, re-weighting percentages, the overlap quantity and gates, identifiability, the factor-card count, the CD34/CD99 rho, why C is retuned, Figure 3's score-energy ratio, Figure 4's axis-loss axis, and a walk-through of one patient-vs-rest comparison for normal and malignant cells.
- W2.4: D26 leave-high-prevalence-out leverage test on allograft rejection and apoptosis, with size- and composition-matched random-drop controls (planned devlog id 2026-10-09_d26-leverage-allograft-apoptosis).
- W1.9: D28 cell-type-stratified score-energy shares per patient (planned devlog id 2026-10-09_d28-celltype-score-shares).
- W5.3: D27 factor association with predicted cell type and CITE-seq surface proteins for all factor cards (planned devlog id 2026-10-09_d27-factor-modality-assoc).
- W2.3: decide the re-weighting target (Stage 1 geometry or Stage 2 labels) before more re-weighting runs.
- W8.5: the user decides the open patient-inclusion choices (evaluation threshold; whether P01, P06 and P07 stay as evaluated, training-only or excluded) from the options in W8_new_cohort/integration_plan.md §4; agents present options and do not recommend (A-20261009-11, A-20261009-14).
- W8.2: add to the pre-registered plan (a) the patient-inclusion rule for new runs (no patients without malignant cells; whether P01, P06 and P07 stay, A-20261009-11) and (b) cell-type-stratified healthy-cell downsampling as a pre-registered arm that tests the healthy:malignant ratio separately from malignant count, retunes C per condition, and states in advance how a gain would be explained (A-20261009-12).
- W0.13: keep the paper scaffold under research/paper/ (README.md, OUTLINE.md, FIGURES.md, created by devlog 2026-10-09_w8-healthy-downsampling-plan) current: each result that changes a claim adds its devlog id to the matching OUTLINE results subsection (AGENTS.md §6); the healthy-cell downsampling subsection stays pending until W8.7 (W8.8 writes it).
