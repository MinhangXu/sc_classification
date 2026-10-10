---
id: 2026-10-09_w8-healthy-downsampling-plan
date: 2026-10-09
type: docs
title: W8 pre-registered healthy-cell downsampling plan (Arm D, H-S2-09, ALT-NREG), patient-inclusion section of the integration plan, and the living paper context (research/paper/)
asked_by: user
asks: [A-20261009-11, A-20261009-12, A-20261009-14, A-20261009-16]
implemented_by: claude-code
questions: [Q2, Q3, Q1, meta]
workstream: W8
analyses: []
hypotheses: [H-S2-09]
verdicts: [H-S2-09=open]
alts: []
closes: []
advances: [W8.2, W0.13]
claim_level: none
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # report v2, D26-D28 and this overlay round are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
> "So we just gotta show not only it's the total number of malignant cells giving us a hard time, it's also a ratio problem. So if we do a proper systematic downsampling of the healthy cell stratified by their cell types knowing that they represent the healthy population very well with the trajectory, can we just down-sampling them (and yea more rigorously tested and better code-wise developed/implemented) ... My main concern is that if the downsampling of healthy cells improved the classification, we need to understand why? And speaking of paper, in thinking about repo update, let's have the context ready to develop the paper's results in parallel of all the runs planned and analyses planned." (A-20261009-12, excerpt)

> "In a new run, let's definitely subset the patients to do the whole stage0-2 pipeline, let's definitely not include patients that're just in the cohort and contribute no malignant cells at least." (A-20261009-14, excerpt)

> "We got 4 patients passing the 100 malignant cell rule. P02, P03, P04, P09. We need to think if P01, P06 and P07 are still relevant to use. For some reason P07 still does pretty decent lift in the IFN-alpha case." (A-20261009-11, excerpt)

> "How would this help with the downsampling idea above?" (A-20261009-16, excerpt)

Source: research/asks/2026-10-09_report-reading-notes.md (capture: verbatim). The user's closing line, "All the new proposed run idea like downsampling, let's include that when we have the new data from team.", fixes the timing (recorded in devlog 2026-10-09_report-reading-notes).

## Why it matters
- The healthy-to-malignant ratio idea is a new hypothesis (H-S2-09, Q2/Q3). It needs a pre-registered design that separates count from ratio and explains any gain before it runs, because a fixed-C version (D10) already produced a misleading collapse.
- The inclusion decision (W8.5) changes every new run's training set and the number of LOPO folds. The user needs the facts in one place before the W8.2 lock.
- The paper context (W0.13) lets each new devlog entry land in a known place in the paper.

## Estimand / null
n/a (plans and context; nothing was computed). For the record, the pre-registered Arm D estimands are in `scripts/mrd_stage0_2/workstreams/W8_new_cohort/healthy_downsampling_plan.md`:
- the primary estimand (§5.6, §7): the per-patient paired lift difference between a nested downsampling recipe and the full-data probe, with C retuned per condition by inner LOPO on an extended grid, L1 and L2 (R5), inductive scope;
- the count and ratio effects (§5.3): from the malignant-retention × healthy-retention grid;
- the mechanism read-outs (§6);
- the success and falsification rules (§7).
c_policy for the planned runs: per_fold_tuned (training n changes in every condition).

## Implementation
- New: `scripts/mrd_stage0_2/workstreams/W8_new_cohort/healthy_downsampling_plan.md`, the pre-registered design for Arm D. It covers:
  - what D10, D11 and D19 already show;
  - why balanced class weights plus rank metrics leave only four channels (Stage 1 geometry, negative-class composition, the n-dependent penalty, noise), with a geometric prediction for the Stage 1 channel (small K versus K40);
  - schemes S0-S5 at matched n, placements (Stage 1 only, Stage 2 only, both) × transductive/inductive scope, and the count × ratio grid;
  - the C policy with an extended grid and diagnostic arms, and the nested recipe;
  - read-outs (D16 score energy, D17 axis metrics, D19 entry margins, composition);
  - criteria, a compute estimate and phased screens (W8_hds_*), the runner design (W8.6), the paper angle, risks, and open choices for the lock.
- Edited: `scripts/mrd_stage0_2/workstreams/W8_new_cohort/integration_plan.md`:
  - a 2026-10-09 update banner;
  - §4.0, the user's zero-malignant exclusion (P08, P10, P11, P12 in the current cohort, verified against study_config.json and the cell metadata), with the reading to confirm for Arm P;
  - §4.1, the open inclusion decision (thresholds at least 10, 20 or 100; the training role of sub-threshold patients), with per-patient counts, prevalence, the healthy-per-malignant ratio and fold implications, and no recommendation;
  - §4.2, the old rules, now conditional on W8.5;
  - Arm D in §8, a §10 row, a §11 item, §12 risk rows, §14 items 9-11 and readiness items.
- Edited: `scripts/mrd_stage0_2/workstreams/W8_new_cohort/README.md`:
  - charter additions (A-20261009-11, -12, -14, -16, and the closing-line decision);
  - Builds on;
  - backlog W8.5 (inclusion decision, P0, blocked on the user), W8.6 (runner and tests), W8.7 (the run, blocked on new data), W8.8 (the paper arc);
  - W8.2 now also locks the downsampling plan and depends on W8.5;
  - conventions (slug healthy_downsampling_v1; the patient rule).
- Edited: `scripts/mrd_stage0_2/research/hypotheses.md`, new H-S2-09 (number-free; origin new 2026-10-09).
- Edited: `scripts/mrd_stage0_2/research/alternatives.csv`, new ALT-NREG (n-dependent regularization path; typical tasks W8.7 and W4.6).
- New: `scripts/mrd_stage0_2/research/paper/` (W0.13):
  - `README.md`: purpose, relation to THEME/CLAIMS/STATUS/devlog, rules (number-free, add devlog ids to OUTLINE);
  - `OUTLINE.md`: working titles including the user's catch line, an abstract skeleton, Introduction beats, Results R1-R12 mapped to Q tags, H-IDs, analyses and devlog ids with status, Methods M1-M13 mapped to code, Discussion, experiments still needed mapped to W tasks, and a hypothesis coverage table;
  - `FIGURES.md`: main figures Fig1-Fig9, the HDS panels and supplementary S1-S9, each with message, source ID, status and the current report v1 or EXP figure file.
- Metadata checks (agent shell, seconds, read-only): per-patient malignant and total counts in the cell metadata CSV equal study_config.json; `predicted.annotation` and `predicted.pseudotime` exist in the cell metadata. No probe, AP or lift was computed.
- Follow-up by the W8 author (2026-10-10, after the report-v2 integration; docs only):
  - `research/paper/FIGURES.md`: a crosswalk from this file's figure numbers to THEME §9's coarser numbering; Fig7 panel F (`fig05d_identifiability`, N15); S8 now points to the report v2 card table.
  - `research/paper/OUTLINE.md`: devlog-level results not yet reviewed into THEME §6 (D16, D17 and D28 in R4; D27 in R8) are marked *pending THEME review*, and §0 defines that marker.
  - Re-checked: every devlog id, figure path, THEME §6 label and W task cited in the W8 plans and `research/paper/`; the §4.1 table against `study_config.json`; and, in the cell metadata, the predicted-type × label table behind plan §2.5 and that `predicted.pseudotime` is complete. No probe, AP or lift was computed.
- Integration note (2026-10-09, report-v2 integrator): this entry was staged by the W8 author and written to the devlog by the integrator, unchanged except for this line; it supplies the H-S2-09 verdict declaration (open) that research_index check requires.

## Artifacts
none

## Results
none (plans and context; no analysis)

## Claim gate
Supports (descriptive): nothing; this entry adds a pre-registered design, an open decision laid out for the user, a hypothesis in the open state and the paper scaffold.
Does not support: any claim that healthy-cell downsampling improves classification or that the healthy-to-malignant ratio matters (not yet run; it waits for the new data, needs C retuned per condition, and fixed-C D10 results are all-zero L1 collapses, AP = π by construction); any recommendation on which patients to keep (the user's W8.5 decision).

## Next steps
- W8.5: the user decides the evaluation threshold and the training role of sub-threshold patients (`integration_plan.md` §4.1, §14 items 9-10).
- W8.2: lock `integration_plan.md` and `healthy_downsampling_plan.md` together, with the user's answers to the plan's §12.
- W8.6: write run_healthy_downsampling.py and tests/test_healthy_downsampling.py (both planned); add the full-cell inner-validation option to `tune_c_inner_lopo` in a type: refactor entry; run a permuted-label smoke test for the timing model.
- W8.7: run Arm D with the new data (screens W8_hds_ifna, W8_hds_focus, W8_hds_grid, W8_hds_external), after reserving a D-ID and adding a campaigns.csv row.
- W8.8 / W0.13: keep `research/paper/OUTLINE.md` and `FIGURES.md` current; agents add devlog ids to the matching Results subsection.
- W9.3: a literature addendum on case-control sampling, local case-control subsampling, imbalanced learning and single-cell sketching (`healthy_downsampling_plan.md` §10), checked before citation.
- W0.14: optionally extend the research_index number-free lint (W04) to `research/paper/` through a type: refactor entry.
