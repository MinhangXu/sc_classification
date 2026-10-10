---
id: 2026-10-09_questions-report
date: 2026-10-09
type: docs
title: Working PDF report that answers the 2026-10-08 team questions section by section, with figures, for the lead to filter and share
asked_by: user
asks: []
implemented_by: claude-code
questions: [Q1, Q2, Q3, QI, meta]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the report directory and the D15b full-design entry are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
Verbatim user request (2026-10-09 session; not yet recorded as an A-ID in research/asks/):
> "Maybe you can help create a private shareable pdf? Is that what you mean? I want to see reports on my level and then I learn and filter and then share with my team wrt the questions we asked in the beginning, can I get a report? With whatever plots necessary"

"The questions we asked in the beginning" are the items of the user's 2026-10-08 team report, A-20261008-01..32 (research/asks/2026-10-08_team-report.md).

## Why it matters
The lead needs to understand each result well enough to defend it before the team sees it, and to drop sections that are not ready. The report restates Q1-Q3 and QI evidence from the devlog in one place, one question per section, each with its claim boundary.

## Estimand / null
n/a (docs). The report restates descriptive results from existing devlog entries and EXP artifacts. The figure modules read stored tables and scores only; no Stage 1 or Stage 2 model is refit, so there is no C policy and training n does not change.

## Implementation
- Report directory: scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/ (README.md there explains layout, rebuild and filtering).
- Rebuild (lab server, from the repo root): `bash scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/build.sh` (figures through make_figures.py with the lab env python, then pdflatex until cross-references settle; `--no-figures` compiles only). About 18 s in an agent shell, BLAS threads capped at 4.
- Sections were written by one agent per section (sections/01-09, A1, A2, figs_01-09) against a shared style (report_style.py, preamble.tex); the assembler wrote report.tex, sections/00_answers_at_a_glance.tex, make_figures.py, build.sh and README.md. Each section author kept a number ledger (number, claim, source file, how computed) in the session scratchpad; those ledgers are not committed.
- report.tex layout settings for the assembled PDF only: boxes breakable across pages, figures and tables float (!htbp) inside their own section (\FloatBarrier before each section), a section starts on a new page when less than a third of a page remains. Filtering: one `\input` line per section; the glance table's Section column shows a dash for a dropped section.
- Review and fix pass (same day): three reviewers (numbers, pages, reader) filed 49 findings; each was checked against the EXP artifacts and devlog Results, and all were fixed (several partly, see Results). Main changes: the grant sentence in section 7 (one *supported* training patient, three programs in four representations, the transductive qualifier); section 2 and the glance table no longer state that P02/P03/P04 do not shape Stage 1 (leverage is tested on IFN-alpha only); option 1.2 is "no clear loss under tuned L1; FA40 loses 1.13 pp under L2"; section 3 states both shuffled-label nulls; section 4 has the correct atlas C grid (x4.64 steps, 13 values) and a new factor-by-patient heatmap (figure fig04c, from factor_patient_matrix_long.csv.gz); section 5 adds the cell-type ceiling in P07, the Tucker band and a rotation practice; section 7 adds the allograft all-zero collapse at small n; section 8 is trimmed to evidence only it shows, answers Q1 predictively, adds the cohort-wide eta2 row and the selection-optimism bullet, and the draft paper paragraph names the donor-versus-recipient confound; section 9's repository decisions moved to a new Appendix C (sections/C1_for_you_only.tex) that `build.sh --team` leaves out of report_team.pdf. Layout: no figure wider than the text column, true minus signs and math notation in figures, blue kept for PCA only, consecutive table numbers, correct contents links, unbreakable short-answer and team boxes, no stranded lines. preamble.tex: the explainbox is now a left-rule box so it cannot be confused with the caveat box (report_style.py unchanged).
- Fix outside the owned files, to make `research_index.py check` pass: figs_02_shape.py printed the label "rank p:" followed by a newline escape, which lint E07 reads as a Windows drive letter; the label now reads "rank p (composition-matched):". Output files are unchanged.

## Artifacts
none (no EXP writes). The report's own outputs (figures/*.pdf, figures/*.png, report.pdf) sit in the report directory and are gitignored.

## Results
- Built PDF after the fix pass: 59 pages (report_team.pdf, without Appendix C: 58); 22 figures and 8 numbered tables (numbered 1-8 without gaps); 0 LaTeX errors, 0 overfull boxes, no undefined references; every section also compiles standalone. Section start pages: Answers at a glance 2, 1 cohort and yardstick 5, 2 highest-prevalence patients and the factors 10, 3 aggregation 14, 4 IFN-alpha PCA40 vs FA40 19, 5 PC8 24, 6 inductive Stage 1 (option 2) 29, 7 learning curves 33, 8 Q1-Q3 38, 9 remaining TODOs 45, Appendix A glossary 51, Appendix B where things live 55, Appendix C for you only 59. (First build before review: 55 pages.)
- Filtering check: with sections 3 and 6 commented out the report compiles (47 pages), the remaining sections renumber, and the glance table shows a dash for both.
- Derived numbers first stated in the report (descriptive; computed by the figure modules from stored artifacts, no refits; recorded here under R12):
  - Option 3 pairs (section 9, from patient_count_learning_curve_v1/d15b_full_unit_results.csv, tuned arm): training on the other two of P02/P03/P04 vs the full LOPO fold, median difference +0.13 pp over 12 folds (4 representations x 3 held-out patients), 6/12 higher, range -8.18 to +10.09 pp; P02 lower in all 4 representations, P04 higher in all 4.
  - Timing (section 9, per-sample days from d1a_deep_dive_20260910/d1dd_rank_matched_confusion_by_sample_collection.csv; median raw lift per patient over 643 representations): Spearman rho with days since transplant 0.57 (exact two-sided p 0.20, 5,040 orderings), with days before relapse 0.18 (p 0.71), n = 7; prevalence rho 0.89 is D6 (meeting_followup_v1/old34_patient_level_metric_associations.csv).
  - Lift concentration (section 8, from old34_d1_atlas_supported_patient_metrics.csv, 640 biological DR atlas cells): share of cells with lift > 5 pp P03 80%, P02 71%, P04 64%, P09 23%, P07 16%, P01 0.6%, P06 0.5%; lift > 0 in 69-84% for every supported patient.
  - DR vs direct genes under median-lift selection (section 8, old34_d1e_best_dr_vs_direct_gene_selection.csv): best DR beats direct genes on P20 lift in 24/41 biological panels (0/6 controls), on median lift in 28/41 (30/47 with controls); median biological DR-minus-direct median-lift gap +0.93 pp.
  - Option 3 pair folds include the six unsupported patients (20,771 cells, 12 malignant: P05 8, P13 4): training sets of 23,361-24,560 cells with 325-981 malignant (d15b_full_unit_results.csv, tuned arm, k = 2).
  - Timing: exact two-sided permutation p for the D6 prevalence rho 0.89 (n = 7, 5,040 orderings) is 0.012; with seven patients the three timing/prevalence correlations cannot be told apart.
  - IFN-alpha K = 40 factor-by-patient matrix (factor_patient_matrix_long.csv.gz): P02 tops 31 of 40 PCA40 and 27 of 36 non-degenerate FA40 factors, P07 7 and 7, P01 2 and 2, P03 none (reproduces the D16 findings counts).
  - Cohort-wide per-factor eta2 over the 5,838 non-degenerate PCA/FA factors of the 320 D16 bases (factor_patient_matrix_factor_summary.csv): median 0.024 (normal) and 0.051 (malignant), largest 0.352 and 0.619.
  - Cell-depth curve (cell_count_learning_curve.csv): the allograft elastic-net probe (C 4.64e-4, L1 ratio 0.5) is all-zero in 21 of 21 units at 500, 1,000 and 2,000 training cells and in 3 of 21 at 5,000; its L1 entry bar 4 x 0.5/(C n) is about 2.2 SD at 2,000 cells.
  - Denominators: the 643 representations of D6 are the 640 biological DR atlas cells plus the direct-gene probes of the three small panels; the stored atlas C grid has 13 values per penalty (1e-4 to 1e4, steps of x4.64).
  - Closest biological factor to the D14 single-factor flag (section 3, recomputed from stored scores with the D14 functions): p53 pathway FactoSig K40 factor 9, malignant eta2 0.569, variance share 0.104, P03-vs-rest overlap 0.263.
- Numbers that differ between devlog Results and artifacts, as handled in the report: FA40 normal-only tuned-L1 median lift is 7.5747 pp in d17_pca_vs_fa_by_arm.csv, printed 7.58 in 2026-10-08_d17-basis-variant-pilot; the report uses 7.57. Section 7 and the glance table use the full D15b design (2026-10-08_d15b-full-design), not the reduced-design gains in 2026-10-08_d15b-patient-count-curve and the 2026-10-08 team update (PCA40 gain k = 2 to 6: 1.67 [-0.44, 2.89] pp full vs 2.16 [0.20, 2.48] reduced).
- Data issue found while writing section 9 (not corrected in EXP): d1a_deep_dive_20260910/d1dd_supported_patient_collection_context.csv pairs P09's days before relapse (-112; -294; -399; -586) with days since transplant in the reverse order (137; 324; 429; 611). The per-sample table and the cell metadata pair 137 with -586 and 611 with -112 (relapse at day 723 for every draw); the report uses the per-sample table.

## Claim gate
Supports (descriptive): the report restates descriptive results only, at the claim level of the entries it cites (seven supported patients; transductive Stage 1 except the IFN-alpha inductive pilot; no calibrated significance tests), with each section's "What this does not show" box carrying that entry's claim gate.
Does not support: any claim beyond those entries' claim gates; the derived numbers above as inferential; inductive deployment, causal biology or recurrence risk; "more patients help". The report is a dated snapshot; the devlog and STATUS.md stay canonical.

## Next steps
- W0.1: the user reviews and filters the report, then decides with the rest of the overlay what to commit (sources only; PDFs and PNGs are gitignored).
- W0.2: the .gitattributes LFS decision also covers *.pdf and *.png, which matters only if report figures are ever committed.
- W1.7: record the P09 erratum for d1dd_supported_patient_collection_context.csv in the timing entry when W1.7 is closed.
- W6.1 follow-up: decide whether 2026-10-08_d15b-full-design should list 2026-10-08_d15b-patient-count-curve in supersedes (AGENTS.md section 7 says a completion entry supersedes the partial run entry).
