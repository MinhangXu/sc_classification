# research/paper: the living manuscript context

This directory is the paper's working map. It is developed in parallel with the planned runs and analyses so that, when a result lands, its place in the paper is already known (W0.13; the user's ask A-20261009-12: "let's have the context ready to develop the paper's results in parallel of all the runs planned and analyses planned").

It holds no draft prose yet and **no numbers**. It says which question each part of the paper answers, which hypotheses and analyses feed it, which devlog entries hold the evidence, which figure panel shows it, and what is still missing.

## Files

| File | What it is |
|---|---|
| `README.md` | This file: purpose, relation to the other research files, update rules |
| `OUTLINE.md` | Working titles (including the user's downsampling catch line), an abstract skeleton, the Introduction beats, Results subsections R1-R12 mapped to Q1/Q2/Q3/QI, H-IDs and devlog ids with a status, Methods mapped to code paths, Discussion points, the experiments still needed (mapped to W tasks) and a hypothesis coverage check |
| `FIGURES.md` | Main and supplementary figure plan: panel, message, source analysis ID, status, and the current figure file in a report if one exists |

## How it relates to the rest of `research/`

| File | Role | Canonical for |
|---|---|---|
| `THEME.md` | the central theme (§0), the per-question answers (§2-§5) and the claim boundary (§6, the S and N labels) | **what may be claimed, and in which words.** The OUTLINE cites THEME §6 labels and never goes beyond them |
| `CLAIMS.md` (generated) | each hypothesis's evidence bullets, with numbers, grouped by question | the numbers behind a claim |
| `STATUS.md` (generated) | current verdicts, open tasks, analyses, campaigns, unanswered asks | the current state |
| `devlog/` | the dated trace of every analysis; Results sections are the only place numbers live (R12) | evidence and numbers |
| `hypotheses.md`, `alternatives.csv`, `analyses.csv` | hypothesis statements and gates, reviewer alternatives, the analysis registry | IDs and their definitions |
| `reports/` | lead-facing PDF reports. Their figure modules regenerate figures from EXP, and a paper panel can reuse them | report figures |
| `paper/` (this dir) | the structure and the story: which evidence goes where, in what order, and what is missing | nothing on its own; it cites the files above |

When `paper/` disagrees with THEME §6, THEME wins. Fix `paper/` in the same task.

## Rules for agents and contributors

1. **No numbers here.** Cite devlog ids, analysis IDs (D, F, D1DD), H-IDs, ALT-IDs, THEME §6 labels (S01..., N01...) and W tasks. A number in a draft sentence is written as a slot, for example `[R5: per-patient inductive change, see 2026-10-08_d17-basis-variant-pilot]`. The research_index number-free lint (W04) does not scan this directory yet, so keep it number-free by hand. A lint extension would be a tools change made through a `type: refactor` entry (W0.14).
2. **When your devlog entry changes a claim, add its id to the matching OUTLINE subsection** (the "Feeds" line), and update that subsection's status if it changed (AGENTS.md §6). If your result also needs a figure panel, add or update its row in `FIGURES.md`. Do not paste numbers or rewrite the claim beyond THEME §6.
3. **Status vocabulary** (OUTLINE and FIGURES):
   - `evidence in hand`: the descriptive claim exists in THEME §6;
   - `partial`: some evidence, with a named gap;
   - `running`: an analysis reserved or in progress today;
   - `needs new data`: waits for the new patients (W8);
   - `planned`: a task exists, not started;
   - `not started`: no task yet. Propose one.
4. **Claim level.** Every subsection is descriptive until a `supported_publication` verdict exists (AGENTS R6). Draft sentences use the THEME §6 vocabulary and end, where a summary is needed, with the standard closing line in THEME §6.
5. **Structure changes are the user's call.** Merging or splitting results subsections, promoting the downsampling arc to a companion paper, or choosing a title are user decisions; record them in a `type: docs` or `type: decision` devlog entry.
6. **Figures.** `FIGURES.md` names the panel and its source. The figure file is produced by a report figure module (`reports/<report>/figs_*.py`) or an analysis runner, never drawn by hand here.
7. **No absolute paths** (R9). Paths are repo-relative, or EXP-relative for artifacts.

## How the lead uses it

- **What is the paper today?** Read `OUTLINE.md` top-down. Each Results subsection says what is in hand, what is running and what waits for new data.
- **Which figure shows what?** `FIGURES.md`. Most panels point to a report figure you have already seen.
- **What do we still need to run?** `OUTLINE.md` §7 lists the experiments, mapped to W tasks; `STATUS.md` §2 has their state.
- **Where are the numbers?** Follow a devlog id from `OUTLINE.md`, or open `CLAIMS.md` at the H-ID.

## Maintenance

- Owner: W0.13 (living paper context). It is reviewed at each THEME milestone, together with `THEME.md` §9 (the canonical figure list at milestones, kept consistent with `FIGURES.md`).
- Created 2026-10-09 (README, OUTLINE, FIGURES). The paper-context pointer for agents is AGENTS.md §6.
