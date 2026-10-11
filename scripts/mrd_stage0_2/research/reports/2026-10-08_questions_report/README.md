# MRD Stage 0-2: questions from 2026-10-08 (working report)

Answers, question by question, to the user's 2026-10-08 team report (asks A-20261008-01..32) and the
follow-up questions of 2026-10-09/10: whether the highest-prevalence patients shape the factors (IFN-alpha,
then allograft rejection and apoptosis, D26), whether to aggregate patients, the IFN-alpha PCA40-vs-FA40 gap,
what PC8 is (with surface protein), the inductive Stage 1 option, the learning curves, the three scientific
questions (Q1-Q3) and the decisions for the runs on the new patient data. Every result is descriptive: seven
supported patients, mostly a transductive Stage 1 basis, no calibrated significance tests. Numbers trace to
devlog entries (`[Source: ...]` pointers in the PDFs; Appendix B of the full report maps sections to code and
EXP artifacts).

## Which file to read

| File | Pages | Read it when |
|---|---|---|
| `report_core.pdf` | 12 (limit 12) | **Start here; this is the deliverable.** Page 1 is a one-table summary; then one page per question: short answer, 3-5 pieces of evidence with numbers, one figure, what it does not show, and the full-report section that has the detail. Team-safe (no personal appendix; "MX" is the study lead, Minhang Xu). |
| `report_full.pdf` | about 99 | Reference only, not meant to be read through. You want the derivation, every caveat, the explainer boxes or a number's source. Same questions, every section in full; the core's pointers use its section numbers (1, 2, 2b, 3-9). |
| `report_full_team.pdf` | about 98 | You share the long version: `report_full.pdf` without Appendix C (the lead's repository decisions). |
| `report_v1_2026-10-08.pdf` | 59 | Only to follow the 2026-10-09 reading notes, which cite version 1's figure numbers (kept, not rebuilt). |

Only `report_core.pdf` is tracked (a `.gitignore` exception; it reaches the remote with the W0.3 push); the other
PDFs are gitignored, so after a clone build them (next sections; this needs EXP) or ask the owner for
`report_full_team.pdf`. Both reports hold unpublished,
patient-level cohort results: share through a private channel.

**For agents and new collaborators.** The repository entry point is `AGENTS.md` at the repo root. Here, the
sources are `report_core.tex` + `sections_core/*.tex` (core) and `report.tex` + `sections/*.tex` (full); the
figures come from the `figs_NN_*.py` modules. Neither report computes anything: the core copies its numbers
from the full report's sections and from the devlog entries it cites, and the full report from devlog Results
and EXP tables (R12). When a result changes, change the full report's section first, then the matching core
page.

Devlog entries for the report itself: `scripts/mrd_stage0_2/research/devlog/2026/2026-10-09_questions-report.md` (version 1)
`scripts/mrd_stage0_2/research/devlog/2026/2026-10-09_report-v2.md` (version 2) and
`scripts/mrd_stage0_2/research/devlog/2026/2026-10-10_core-report-v3.md` (version 3: the 12-page core, section 2b,
the tracked core PDF, the D26 ranges both reports derive) and
`scripts/mrd_stage0_2/research/devlog/2026/2026-10-10_protein-explainer-v3.md` (version 3.1: the surface-protein
explainers in section 5 and on the core's protein page).

**Core report (2026-10-10).** The user will not read 85-92 pages (A-20261010-11), so `report_core.tex` holds the
answers in 12 pages (limit 12): the glance table on page 1, then cohort and yardstick (with the key terms),
IFN-alpha shaping, allograft rejection and apoptosis (D26), aggregation (D14), the IFN-alpha PCA/FA entry rule,
what PC8 is (D18), the protein data (D27, Pavan's question and the user's 2026-10-10 protein questions
A-20261010-01, -03..-06 from devlog `2026-10-10_protein-association-for-pavan-v2`), the inductive option, the
learning curves, Q1-Q3 with the claim boundary, and the decisions for the new-data runs (A-20261010-07..10). It
calls the study lead "MX" (defined on page 1), so collaborators do not read "you" as addressed to them. It reuses the full report's figures; two
are cropped in LaTeX (see "Core report figures" below).

**Version 3.3 (2026-10-11).** The user's questions on version 3.2 (asks A-20261011-01..13 in
`research/asks/2026-10-11_clr-isotype-dsb-fig23-next.md`; devlog `2026-10-11_clr-isotype-p03-fig23-next`). Section 5
gains a box defining ADT/CLR, isotype controls and dsb (with an email sentence), "Why P03 sits near 0" with a new
Table 4 (`tab:05p03`, typed from `figs_05q_cd63_depth.table4_p03`'s printed rows), Figure 23 split over two pages
(`fig05f_modality_all`: a-d, with a new inside-type / within-patient panel d and a clearer x-axis; `fig05f2_modality_more`:
e a Simpson-type example, f-g the eta2 panels, h lift against cell type), a new Figure 28 (`fig05t_isotype_depth`,
after Figure 27), "Should we switch to the total-ADT-adjusted figures?" and "Where the protein work goes next".
`fig05f2` and `fig05t` are drawn by `figs_05q_cd63_depth.py` (it loads the per-cell data); `figs_05_pc8.py` prints the
new "Every card" numbers. The core page 8 defines CLR and isotypes in one sentence and stays at 12 pages.

**Version 3.2 (2026-10-10, late evening).** The user's questions on full report Figures 20-23 (asks
A-20261010-26..37 in `research/asks/2026-10-10_cd63-every-card-isotypes.md`; devlog
`2026-10-10_d29-cd63-depth-and-every-card`). Section 5 gains the subsection "CD63, followed up" with analysis D29
(`workstreams/W5_factor_interpretation/run_d29_protein_depth_contrast.py`, slug `protein_depth_contrast_v1`) and four
figures drawn by `figs_05q_cd63_depth.py` (make_figures key `05q`): `fig05n_cd63_ranks` (CD63 against PC8 rank by
rank), `fig05o_cd63_anchors` (CD63 and CD82 taken apart), `fig05q_mal_vs_healthy` (protein AUC, malignant vs healthy
cells of the same sample and type) and `fig05r_two_checks_depth` (Figure 22 with each cell's total ADT count held
fixed). They follow Figure 23 (`fig05f`), so Figures 19-23 keep their numbers (new: 24-27; later figures move up by
four). "Every card" explains the 289,961 pairs, Stage 2's role and eta2, with a statement box; `fig05f` drops the
method colours. The core changes by one sentence on page 8.

**Version 3.1 (2026-10-10, evening).** The user's questions on the protein part of section 5 (asks
A-20261010-14..24 in `research/asks/2026-10-10_figure19-and-signs.md`; devlog `2026-10-10_protein-explainer-v3`)
are answered inside both reports instead of only in the Pavan update directory: Figure 19 relabelled, Table 3 with
plain headers and PC8's other strong malignant-cell correlates (CD116, CD33, CD146), a new Figure 20 (how one rho
is computed from two per-cell vectors, and what its sign means) and a new Figure 22 (two checks per protein: within
patient against within cell type). Version 3's Figure 20 is now Figure 21 and its Figure 21 is now Figure 23; every
later figure is two higher. The update directory's working names F5, F6 and F7 are full report Figures 20, 22 and 19.
The core stays at 12 pages and carries compact versions: page 7, Figure 6 bottom = Figure 19 (`fig05m_celltype_core`,
one column per type); page 8, Table 1 = Table 3's PC8 malignant-cell rows, and Figure 7 = `fig05l_protein_core` (A:
what the sign of rho means, Figure 20 panel D; B-C: the two checks of Figure 22). The matrix-to-vector panels of
Figure 20 (A-C) and Table 3's FA7 rows are in the full report only. A three-page core appendix with the full-size
figures was drafted and is not built (15 pages, over the limit; see `sections_core/11_appendix_protein_figures.tex`).

**Version 3 (2026-10-10).** Adds section 2b, *What did we do with allograft rejection and apoptosis?*
(`sections/02b_allograft_apoptosis_leverage.tex`, figures from `figs_02b_d26.py`): the D26 leverage test as run,
the rule fixed before the run, the result, and what it means for re-weighting and healthy-cell downsampling.
Section 2 keeps IFN-alpha and points there; section 2b is numbered "2b" so sections 3-9 keep their version-2
numbers. It records the user's 2026-10-10 decisions (asks A-20261010-07..10, devlog
`2026-10-10_decisions-new-run-inclusion-reweighting-lfs`): new full runs keep only patients with at least 100
malignant cells (section 9, with what that does to median and P20 lift), re-weighting targets the Stage 2 labels
(sections 2 and 2b), and the Git LFS lines are gone (Appendix C). It also applies the findings of two agent reviews
of version 2 (one account of the downsampling mechanism in sections 2 and 9 with the earlier D10/D11 result,
section 8 brought up to date with D26, claim wording held to the devlog gates, figure label collisions, glossary).

**Version 2 (2026-10-09).** The lead's reading notes on version 1 (asks A-20261009-01..16 in
`scripts/mrd_stage0_2/research/asks/2026-10-09_report-reading-notes.md`) are not answered in chat; version 2
changes the report so that it answers them. `report.tex` opens with a revision table that maps each note to the
box, figure or subsection that now covers it. Version 2 adds the 2026-10-09 analyses D27 (factor x surface protein
and cell type for every card) and D28 (cell-type-adjusted score shares), the D26 leverage test on allograft rejection and apoptosis (run 2026-10-10), redrawn
section 2 figures, three worked-example figures in section 3, three figures and a card table in section 5, a
healthy-cell downsampling and patient-inclusion subsection in section 9, and new glossary entries. Figure and table
numbers shifted, so the v1 numbers quoted in the notes no longer match. The version-1 PDF is kept next to the
report as `report_v1_2026-10-08.pdf` (gitignored, like every PDF here except `report_core.pdf`).

## Layout

| Path | What it is |
|---|---|
| `report_core.tex` | Core report: page macros (`\coreq`, `coreanswer`, `\doesnot`, `\fullref`), one `\input` line per page |
| `sections_core/00_title_and_glance.tex` ... `10_decisions_next.tex` | One file per core page; page 1 is the title and the answers-at-a-glance table (page numbers resolve by `\pageref`) |
| `sections_core/11_appendix_protein_figures.tex` | Not built (its `\input` line in `report_core.tex` is commented out): a three-page core appendix A1-A3 with the rho schematic, Table 3's PC8 rows with the two-checks figure, and the relabelled Figure 19. It would take the core to 15 pages; to print it, uncomment the line and restore the glance-table row it names. Its first page carries the label `core:appendix`, after which `build.sh` stops counting the 12-page limit |
| `report.tex` | Full report: title block, how-to-read, contents, one `\input` line per section, layout settings for the assembled report |
| `preamble.tex` | Shared packages and the box macros (`answerbox`, `caveatbox`, `teambox`, `explainbox`, `\src`, `\pp`, `\lift`), used by both reports |
| `sections/00_answers_at_a_glance.tex` | Full report: one-table summary (question, short answer, how sure, section) and what changed since the team report |
| `sections/01_*.tex` ... `sections/09_*.tex` | Full report: one section per question; each stands alone. `02b_allograft_apoptosis_leverage.tex` is section 2b (D26), numbered by `\labelnextsectionb` and `\plainsectionnumbers` in `report.tex` |
| `sections/A1_glossary.tex`, `sections/A2_where_things_live.tex` | Full report appendices: glossary; where code, artifacts and notebooks live |
| `sections/C1_for_you_only.tex` | Full report Appendix C, for the lead only: repository decisions (LFS lines removed, pytest, push, tracked figures); left out of the team build |
| `figs_NN_<topic>.py` | One figure module per full-report section (`figs_02b_d26.py` for section 2b); `main()` reads stored EXP tables (read-only), writes `figures/<name>.pdf` plus a `.png` preview, and prints the numbers its section quotes |
| `figs_05p_protein_explainer.py` | Section 5's surface-protein explainers (make_figures key `05p`): runs `research/team_updates/2026-10-09_protein_association_for_pavan/make_figures_v3.py`, the one source for these figures in the Pavan update and here. Writes `fig05g_celltype_counts` (Figure 19, replacing `fig05c_celltype` in the text), `fig05h_rho_schematic_ab` + `fig05i_rho_schematic_cd` (Figure 20 and its continuation), `fig05j_two_checks` (Figure 22), `fig05k_rho_schematic_core` (for the unbuilt core appendix) and `fig05m_celltype_core` (core page 7, Figure 6 bottom: Figure 19 compact). Table 3 in `sections/05_what_is_pc8.tex` is typed from that script's `table3_rho_three_levels_v3.csv` |
| `figs_05_pc8.py`, `fig05l_protein_core` | The core's protein-page figure (page 8, Figure 7): what the sign of rho means (panel D of Figure 20, compact) and the two checks of Figure 22, drawn with `make_figures_v3.py`'s loaders and panel function (so section 5's figures and this one share code and numbers). Since v3.1 `figs_05_pc8.py` no longer draws `fig05c_celltype` (version 3's Figure 19) |
| `figs_05q_cd63_depth.py` | Section 5's CD63 and total-ADT figures (make_figures key `05q`; D29): `fig05n`, `fig05o`, `fig05q`, `fig05r`, `fig05t` (Figure 28) and `fig05f2` (Figure 23, second page); prints Table 4's rows; asserts every plotted rho against D27 and D29 |
| `report_style.py` | Shared figure style: palette by role (PCA blue, FA orange, FactoSig aqua, direct genes gray), type sizes, `save()`, `exp_dir()`, `diag_dir()` |
| `make_figures.py` | Runs every figure module in section order (or the sections you name) |
| `build.sh` | Figures, then pdflatex until cross-references settle; copies the PDF(s) here (`report_core.pdf` by default) |
| `report_v1_2026-10-08.pdf` | The version-1 PDF the reading notes refer to (kept for reference; not rebuilt) |

## Rebuild

From the repository root, on the lab server (the figure modules need the experiment directory):

```bash
R=scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report
bash $R/build.sh                      # figures + report_core.pdf (default)
bash $R/build.sh --no-figures         # report_core.pdf only, reusing figures/*.pdf
bash $R/build.sh --full               # figures + report_full.pdf
bash $R/build.sh --no-figures --team  # report_full_team.pdf (full report without Appendix C)
bash $R/build.sh --all                # figures + report_core.pdf + report_full.pdf + report_full_team.pdf
bash $R/build.sh --figures 03 07 --full  # regenerate only these sections' figures, then the full PDF
bash $R/build.sh --figures 05 05p --all  # section 5 (05p = its surface-protein explainers), then all three PDFs
```

- Python: `$SC_PYTHON` if set, otherwise `$HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python`
  (with `CONDA_PREFIX` set for the RAPIDS imports), otherwise `python` on PATH. EXP is resolved through
  `sc_classification.utils.paths` (`SC_EXPERIMENT_ROOT`, `SC_EXPERIMENT_ID`); nothing is written to EXP.
- LaTeX: `pdflatex` (TeX Live with tcolorbox, titlesec, longtable, needspace, placeins). Auxiliary files
  (`.aux`, `.log`, `.toc`) go to a temporary directory printed at the end of the build, or to
  `$REPORT_BUILD_DIR` if set, so the report directory only gains the PDFs.
- Without EXP you can still compile with `--no-figures` if `figures/*.pdf` exist on your machine.
- A full build takes about half a minute on the lab server (figures, then up to three pdflatex passes per PDF).
  The script prints each PDF's page count and warns if `report_core.pdf` exceeds 12 pages (pages after an
  optional appendix labelled `core:appendix` are not counted; none is built now).

### Core report figures

Each core page reuses one figure from `figures/` (the yardstick, decision-map, C-policy, inductive,
patient-count and lift-concentration figures as they are). Two are cropped in LaTeX with `\includegraphics`
`trim`, because the core shows only some of their panels: `fig02a_who_dominates` (panels a and c, page 3) and
`fig02b_d26a_ratio_by_set` (panels a-c, page 4). The trim values are in bp and documented at the top of those
two page files; if `figs_02_shape.py` or `figs_02b_d26.py` changes a figure's layout, render the core page and
adjust them. The PC8 page places `fig05a` and `fig05m_celltype_core` at their drawn width (6.5 in) and the protein
page `fig05l_protein_core` at its drawn width (7.0 in; the core's text is 7.1 in wide), so their text prints at its
drawn size; the PC8, protein and D26 pages are full to their last line, and the core is at its 12-page limit: any
text added needs the same amount removed.

## Filter sections or pages before sharing

1. **Core**: comment out (`%`) a page's `\input` line in `report_core.tex`; the glance table's page column
   then shows `??`, so also delete that page's row in `sections_core/00_title_and_glance.tex`.
2. **Full**: comment out the `\input` line of any section in `report.tex`. Each section is one line, so its
   label for the glance table goes with it; section numbers, the contents and the "Answers at a glance" table
   renumber themselves, and a dropped section shows as a dash in the glance table's Section column (delete its
   rows in `sections/00_answers_at_a_glance.tex` if you do not want them). Sections refer to each other by name,
   not number, so any subset compiles. The two appendices are also one line each.
3. Rebuild with `--no-figures` (add `--full` or `--team` for the long versions).
4. Share `report_core.pdf` or the team build, not `report_full.pdf`: `build.sh --team` writes
   `report_full_team.pdf`, the full report without Appendix C (your repository decisions). The team build reads
   the same `report.tex`, so step 2 applies to both full PDFs. To keep a filtered team version apart from your
   own, copy `report.tex` to a new name and compile that copy instead.

## Layout choices in the assembled full report

The section files place figures and tables with `[H]` (fixed position), which suits a standalone
section compile. `report.tex` lets them float (`!htbp`) inside their own section (`\FloatBarrier`
before every section), lets the explainer and caveat boxes break across pages (at least four lines on
each side) but keeps the short answer and the team box whole, forbids single stranded lines, and starts a
section on a new page when less than a third of a page is left. `preamble.tex` and the section files are
unchanged by these settings.

## Git

The repository `.gitignore` ignores `*.pdf` and `*.png`, so `figures/*.pdf`, `figures/*.png`, `report_full.pdf`
and `report_full_team.pdf` are not committed. The one exception is `report_core.pdf`, tracked so that a clone
can read the answers without EXP: rebuild it and commit it together with any change to `report_core.tex` or
`sections_core/*.tex`. (The Git LFS lines that once routed both types were removed on 2026-10-10, W0.2; which
other figures and PDFs a clone should carry is open task W0.15.) The sources are what get committed: `report_core.tex`, `sections_core/*.tex`, `report.tex`,
`preamble.tex`, `sections/*.tex`, `figs_*.py`, `report_style.py`, `make_figures.py`, `build.sh` and this README.
Anyone with EXP access regenerates the figures and the PDFs with `build.sh --all`.
