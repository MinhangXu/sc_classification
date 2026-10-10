---
id: 2026-10-10_core-report-v3
date: 2026-10-10
type: docs
title: Core questions report v3 (12 pages, the deliverable; full report section 2b on allograft rejection and apoptosis), D26 ranges the reports derive, report_core.pdf tracked through a .gitignore exception, and README/AGENTS onboarding for collaborators and their agents
asked_by: user
asks: [A-20261010-11, A-20261010-12, A-20261010-13, A-20261009-02, A-20261009-15]
implemented_by: claude-code
questions: [Q1, Q2, meta]
workstream: W0
analyses: [D26]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: [W0.14, W0.15]
claim_level: descriptive
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the report directory, the D26-D28 code and this overlay round are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
seq: 1  # the last of the day's entries: it records the report built after the D26 and decision entries
---

## Question
> "Definitely trim, even 59 pages are a lot of the final report. I am not going to read 59 pages let along 82." (A-20261010-11)

> "In terms of sharing the repo, let's make sure that we are building something both human can easily start or if my collab also use agents, they can easily start understanding." (A-20261010-12)

> "What did we do with allograft and apoptosis? Please update all that as a new section in the report." (A-20261010-13)

Source: research/asks/2026-10-10_protein-and-decisions.md (verbatim; the three requests were given A-IDs in this round). The D26 asks A-20261009-02 and -15 are cited because the new report section answers them in report form; their analysis is 2026-10-10_d26-leverage-allograft-apoptosis-done.

## Why it matters
The lead will not read an 85-92 page report, so the short core report is the deliverable and the long one is reference only. Collaborators, with or without coding agents, start from a clone, so the clone has to carry the answers (the core PDF) and the entry docs have to say which branch, who is who, what a passing setup looks like and how the record is numbered. The D26 section restates a result that changed the H-S1-05 verdict, so every range it quotes has to trace to a devlog Results line (R12).

## Estimand / null
n/a (docs). The ranges under Results are read from stored D26 and D17 tables (no refit, no Stage 2 probe, no C policy; training n unchanged). They restate the D26 decision-rule run of 2026-10-10_d26-leverage-allograft-apoptosis-done at a finer split (PCA and FA separately, control ranges, normal-only refits).

## Implementation
- Core report (`scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/report_core.tex`, `sections_core/*.tex`): v3 is 12 pages (limit 12). Page 4 is the D26 page (allograft rejection and apoptosis); the former PC8 page is split into "What is PC8?" (page 7) and "Have we integrated the protein data? Does it track PC8?" (page 8, A-20261010-01, -03..-06), so both figures print at their drawn size. "You" is replaced by MX (Minhang Xu, the study lead), defined on page 1; the last box is "Open decisions for the study lead (MX)". The key-terms box adds AUPRC/AP, estimand, leverage test, matched |r|, contrast sets and CLR.
- Core text fixes from two review rounds: PC8-alone lift range; the D26 condition (2) count qualified to the primary axis; rule (4) "small" stated per method; the composition-matched drops' size; "supported patient with the most healthy cells per malignant cell"; PCA40's "four" vs five positive patients; the D15b unit count; the six unsupported patients named; the downsampling cost tied to seven held-out patients; the repository sentence (core PDF tracked, other PDFs and figures rebuilt with build.sh).
- Full report (`report.tex`, `sections/*.tex`): section 2b (`02b_allograft_apoptosis_leverage.tex`) carries the same qualifications; sections 00, 02 and 09 qualify the condition (2) count and the "most healthy cells per malignant cell" wording; the 2b run command sits on its own lines.
- Figures: `figs_02b_d26.py` (tick labels without the P prefix, 7 pt text, filled K markers in the legend so open means only "inside its 20 drops", two-row legend, shorter title), `figs_02_shape.py` (panel a P03 label above the patient-equal line; panel c labels off the dot strips), `figs_03_aggregate.py` (the normal gate stops at the subtype band), `figs_01_cohort.py` (P06 and P01 labels), `figs_04_threshold.py` (shorter panel titles), `figs_05_pc8.py` (fig05a title counts genes; fig05e ticks -0.5/0/+0.5).
- Values under Results were read in the agent shell with pandas from the D26 and D17 tables listed under Artifacts (read-only on EXP; a scratch script, not committed); `figs_02b_d26.py` prints the decision-rule counts and asserts them against d26_decision_summary.csv.
- Repository: `.gitignore` keeps `*.pdf` and `*.png` ignored except `scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/report_core.pdf`; `build.sh` header updated to say so. README.md: people and roles, the branch to clone (`-b sc_classification_gpu`), expected setup output, the no-FactoSig route, pyarrow and pyyaml, reports and team updates, private-repo note. AGENTS.md: R4 campaigns rows for screen runs only, R8 cell-level obs columns, §2 step 6 asks numbering, §5 reserve-id example and new W#.n lines, §7 section list, §8 which python, §9/§10 core-PDF and W0.15 wording. W0, W2 and W5 READMEs updated (W5's bundle contract moved to a closing Reference section). 2026-10-08_d15b-full-design now lists the reduced-design run entry in supersedes (front matter only), so STATUS §7 no longer shows it as pending.
- This entry corrects one statement of 2026-10-10_decisions-new-run-inclusion-reweighting-lfs (Results, Git LFS bullet): the questions report's core PDF is no longer absent from a clone; the full and team PDFs and all figures still are (W0.15).
- Rebuild (lab server, repo root): `bash scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/build.sh --all`.

## Artifacts
- Read only (EXP-relative): analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/d26_leverage_contrast.csv, analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/d26_reference_axes.csv, and D17's d17_leverage_contrast.csv (slug basis_variant_pilot_ifna_v1, recorded by 2026-10-08_d17-composition-matched-controls). Nothing written to EXP. The PDFs and figures sit in the report directory (only report_core.pdf is not gitignored).

## Results
- Built PDFs: report_core.pdf 12 pages (v2 had no core; the first core draft was 11); report_full.pdf 92 pages, report_full_team.pdf 91 (v2: 85 and 84); 0 overfull boxes and no undefined references in all three.
- D26 primary axis, P02+P03+P04 left out (d26_leverage_contrast.csv, arm leave_out, ref_axis top_gap; each composition-matched drop removes 4,917 cells, 1,071 of them malignant):
  - allograft rejection PCA (PC8): axis loss 0.0276, 0.0183, 0.0090 at K 10, 20, 40 (0.9-2.8% of the axis's variance); matched |r| 0.9448, 0.9704, 0.9495; the 20 drops' matched |r| 0.9214-0.9640, 0.9033-0.9696, 0.9221-0.9646 (over all three K 0.90-0.97). The group is inside its drops' range at K 10 and 40 and just above it at K 20 (it rotates PC8 slightly less than every drop there).
  - allograft rejection FA (FA8): axis loss 0.0038-0.0065 (0.4-0.7%), matched |r| 0.9883-0.9910.
  - apoptosis PCA (PC6): matched |r| 0.9941-0.9942, below its drops' minimum 0.9950 at every K (more rotation than every drop, still above 0.99).
- Normal-only refits, no malignant cells in the fit (arm normal_only_all, primary axis), matched |r|: allograft PCA 0.9145, 0.8785, 0.9153 (K 10, 20, 40); allograft FA 0.9786, 0.9729, 0.9790; apoptosis PCA 0.9920-0.9921, FA 0.9947-0.9958; IFN-alpha (d17_leverage_contrast.csv) PCA 0.9921-0.9937, FA 0.9974 (K 10), 0.9523 (K 20), 0.9937 (K 40).
- Largest ratio to its composition-matched drops among the eight leave-out sets, primary axis: P01 in all 12 allograft-rejection and apoptosis cells; with the six IFN-alpha cells (D17) P01 in 17 of 18, the exception IFN-alpha FA40, where P06 leads (4.82).
- P02+P03+P04 against P06 and P09 alone (primary-axis ratios): higher than both only in the three allograft PCA cells (3.73 vs 1.56 and 1.62; 3.99 vs 1.67 and 3.25; 4.15 vs 1.66 and 2.85). Allograft PCA40: P03 alone 10.40 against 4.15 for the three together.
- P01's ratio range by axis: allograft best-held-out 15.96-188.13, probe 7.48-35.07; apoptosis best-held-out 30.72-126.16, probe 0.57-91.93. IFN-alpha (D17, primary axis): P02+P03+P04 0.36-1.94, P01 3.74-23.82.
- Best-held-out reference factor differs from the primary (largest-gap) factor in 8 of the 12 method x K cells (d26_reference_axes.csv, same_factor_as_top_gap true in 4).
- Healthy cells per malignant cell (study_config.json): P01 about 263 (11,815 / 45), the highest of the seven supported patients; the unsupported P05 (about 544) and P13 (about 1,443) are higher.

## Claim gate
Supports (descriptive): the reports restate descriptive results of D26 (transductive PCA/FA bases, three panels, seven supported patients) and the ranges above; nothing here changes a verdict (H-S1-05 not supported is declared in 2026-10-10_d26-leverage-allograft-apoptosis-done).
Does not support: any FactoSig or FactoSig-promax reading, any effect on held-out lift (no Stage 2 refit), any causal account of why P01 moves the axes; inductive deployment, causal biology or recurrence risk.
Standard closing line (THEME §6): descriptive results over seven supported patients; they do not establish inductive deployment, causal biology or recurrence risk.

## Next steps
- W0.3 (user go-ahead): commit and push sc_classification_gpu so that a clone carries README.md, AGENTS.md and report_core.pdf; until then a clone of the remote has none of them.
- W0.15 (user decision): whether the full and team PDFs, the report figures and the team-update figures are tracked.
- W0.14: README "People and roles" names the PI only by role; the owner may add names. pyproject.toml declares requires-python >=3.8 while the docs say 3.10+; align them in a separate change.
- W2.8, W2.9 (user go-ahead): the leverage test on all biological panels and the cell-type-matched control for P01, as in the D26 done entry.
