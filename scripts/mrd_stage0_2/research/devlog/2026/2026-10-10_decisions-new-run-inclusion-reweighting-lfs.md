---
id: 2026-10-10_decisions-new-run-inclusion-reweighting-lfs
date: 2026-10-10
type: decision
title: User decisions for runs on the new patient data - full Stage 0-2 runs keep patients with at least 100 malignant cells (the 10-cell support rule stays for the current atlas), re-weighting targets the Stage 2 labels, and the Git LFS lines are dropped
asked_by: user
asks: [A-20261010-07, A-20261010-08, A-20261010-09, A-20261010-10, A-20261009-03, A-20261009-11, A-20261009-14, A-20261008-14]
implemented_by: claude-code
questions: [Q1, Q2, meta]
workstream: W8
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: [W8.5, W2.3, W0.2]
advances: [W8.2, W8.6]
claim_level: none
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
The user's decisions, verbatim from two chat messages of 2026-10-10 (`research/asks/2026-10-10_protein-and-decisions.md`, which quotes both messages in full):
> "Let's do the >=100 malignant cells." (A-20261010-07)

> "Reweight should target stage 2 label it sounds like that's more relevant" (A-20261010-08)

> "I am not sure what's up with Git LFS line." (A-20261010-09) ... "Let's change the git LFS thing." (A-20261010-10)

The user's framing of the first two, same message: "These all sounds like decisions relevant for runs when we have the extra patient data."

They answer open questions recorded earlier: which patients stay in new runs (A-20261009-11, "We got 4 patients passing the 100 malignant cell rule. P02, P03, P04, P09. We need to think if P01, P06 and P07 are still relevant to use."; A-20261009-14, the zero-malignant exclusion); "How do we target stage 1 geometry or stage 2 labels?" (A-20261009-03); and Option 1.2, re-weighting so each patient contributes equally (A-20261008-14).

## Why it matters
- W8.5 was the last open input to the W8.2 lock of `workstreams/W8_new_cohort/integration_plan.md` and `healthy_downsampling_plan.md`. It fixes who is fitted, held out and summarized in every new run (Q1, Q2).
- W2.3 asked which imbalance re-weighting should target before any re-weighted refit. D26 (devlog 2026-10-10_d26-leverage-allograft-apoptosis-done; H-S1-05 not supported) and D17 place the highest-prevalence patients' dominance in the Stage 2 labels, not in the Stage 1 geometry; the user chose that target.
- W0.2 blocked the push (W0.3). The LFS lines would have turned files into LFS pointers on any clone with git-lfs installed, which matters now that collaborators and their agents are meant to start from a clone.

## Estimand / null
n/a (decision). The rules fixed here:

1. **Patient inclusion for new runs (W8.5).**
   - **Scope.** Every new full Stage 0-2 run, starting with the runs on the new patient data: the Stage 1 fit set, the Stage 2 training cells, the C-tuning folds and the held-out, summarized patients of Arm I, Arm D and the W2.5 subset pipeline (`integration_plan.md` §4, §8).
   - **Rule.** A patient enters a new full run only if it has at least 100 malignant cells and both classes. In the current cohort that is P02, P03, P04 and P09. A new patient qualifies by the same rule, from the W8.4 inventory counts, before any probe score is computed.
   - **Patients below the rule.** They are left out of every step of a new full run: not fitted, not trained on, not summarized. This is the reading of "do the >=100 malignant cells" together with the earlier "subset the patients to do the whole stage0-2 pipeline" (A-20261009-14); the train-only role (`integration_plan.md` §4.1 (b)) is not chosen. They are listed in the inventory and described by counts only. The W8.2 lock entry asks the user to confirm this reading; if the user wants train-only, that entry supersedes this one before any new-data run.
   - **What does not change.** `research/study_config.json` keeps its support rule (at least 10 malignant cells and both classes; AGENTS.md R7), so every existing result, CLAIMS.md, THEME §6 and any re-analysis of the current atlas (the canonical experiment: W1, W2.8, W2.9, W3, W4, W5, W6 tasks that read stored bases or scorecards) keeps the seven supported patients and its folds. Nothing old changes silently. The new rule lives in each new experiment's configuration and, for the new cohort, in `frozen_config_v1.json` at the W8.2 lock.
   - **Comparisons** between a new run and an existing result name both cohort definitions (the 10-cell, seven-fold atlas and the 100-cell runs) and report side by side; no number crosses from one to the other unlabelled.
   - **Left to the W8.2 lock.** Arm P scores new patients on the stored canonical bases, which are not a new run and stay as they are. Whether Arm P's pre-fitted probes are trained on all current patients' cells, as the canonical LOPO probes were, or on the 100-cell patients only stays `integration_plan.md` §14 item 10. New patients summarized in Arm P follow the 100-cell rule.
2. **Re-weighting target (W2.3).** Re-weighting acts on the Stage 2 labels, not on the Stage 1 geometry.
   - It changes who the probe learns from: per-patient, per-class sample weights in the logistic loss (each training patient's malignant cells sum to the same weight, and likewise its normal cells), or the healthy-cell composition of the Stage 2 training set. The Stage 1 basis stays as fitted.
   - No patient-reweighted Stage 1 refit is run for the focus panels. The IFN-alpha Stage 1 pilot (D17) stays as the record of what Stage 1 re-weighting does.
   - It runs with the new patient data, inside Arm D (`healthy_downsampling_plan.md`): the Stage-2-only placement becomes the primary placement, and a sample-weight scheme (SW) joins the downsampling schemes. The Stage 1 placements stay as read-outs of why a gain happens (M1), not as the target.
   - Training data change, so C is retuned per condition by inner LOPO, with L1 and L2 side by side (AGENTS.md R5).
3. **Git LFS (W0.2).** `.gitattributes` no longer routes any file type through Git LFS. Notebooks are tracked as plain JSON, as they already were.

## Implementation
- No code ran and no Stage 2 refit was made; `c_policy` and `training_n_changed` do not apply.
- `.gitattributes`: the five `filter=lfs diff=lfs merge=lfs -text` lines (for `*.ipynb`, `*.png`, `*.pdf`, `*.pkl`, `*.h5ad`) were removed by the lead agent of this session on 2026-10-10, after A-20261010-10. The file now explains the removal in a comment and keeps the `linguist-generated` lines for the three generated research files. Checked on the lab server the same day: `git lfs` is not an installed git command; no tracked file starts with the LFS pointer header.
- `research/asks/2026-10-10_protein-and-decisions.md` (new): A-20261010-01..10.
- `workstreams/W8_new_cohort/README.md`: W8.5 recorded as decided; the charter, status, W8.2, W8.6 and W8.7 lines and the Patients convention follow the decision.
- `workstreams/W8_new_cohort/integration_plan.md`: header update for 2026-10-10; §4.1 renamed to the decided rule, keeping the evidence table; §4.2 and §14 items 9-10 updated.
- `workstreams/W8_new_cohort/healthy_downsampling_plan.md`: header update; §4 cohort; §5.1 scheme SW; §5.2 placement emphasis; §5.6 recipe placement; §7 criteria 2 and 5 and the note on intervals; §8 size; §9 the additive weight option; §12 items 1 and 2.
- `workstreams/W2_stage1_basis_variants/README.md`: W2.3 closed by this entry.
- `workstreams/W0_context_provenance/README.md`: W0.2 closed by this entry; new W0.15 (which figures and PDFs are tracked).
- `research/THEME.md` §11 items 6 and 10, `research/paper/OUTLINE.md` (R4, R11, R12, discussion point 2, §7) follow the decisions.
- `research/study_config.json` is unchanged on purpose (rule 1, "What does not change").

## Artifacts
none

## Results
Counts are metadata from `research/study_config.json` (cohort 60,118 cells, 1,569 malignant), as in `integration_plan.md` §4.1.
- Kept by the 100-cell rule in the current cohort: P02, P03, P04, P09; 22,043 of 60,118 cells (36.7%) and 1,446 of 1,569 malignant cells (92.2%).
- Left out of new full runs: P01, P05, P06, P07, P13 (27,437 cells, 123 malignant), and P08, P10, P11, P12 (10,638 cells, none malignant; already excluded on 2026-10-09).
- Consequences for the current cohort's part of a new run: 4 held-out LOPO folds instead of 7; 3 training patients per outer fold for inner-LOPO tuning of C instead of 6; P20 lift interpolates between the lowest and second-lowest patient; with every patient positive, the smallest exact one-sided sign-test p is 1/16 (0.0625) instead of 1/128, so patient-level significance needs the new patients.
- Stage 2 label share behind the re-weighting target: P02+P03+P04 hold 1,071 of 1,569 malignant cells (68.3%). On Stage 1, leaving them out moves the allograft-rejection and apoptosis axes beyond composition-matched drops but less than leaving out P01 does, and H-S1-05 is not supported (devlog 2026-10-10_d26-leverage-allograft-apoptosis-done; D17 for IFN-alpha).
- Git LFS: 5 `filter=lfs` lines removed; 0 of the 341 files tracked at HEAD are LFS pointers; 59 tracked notebooks are plain JSON. `*.png` and `*.pdf` are also ignored in `.gitignore`, so the questions report PDF, its figures and the team-update figures are not in a clone (new task W0.15).

## Claim gate
Supports (descriptive): nothing new about the data; this entry fixes the patient-inclusion rule and the re-weighting target for runs on the new patient data, and the repository's file storage.
Does not support: reading the 100-cell rule back into existing results (the 10-cell support rule and the seven supported patients still govern the current atlas and every claim in THEME §6); any claim that P01, P05, P06, P07 or P13 carry no signal (they are left out by a count rule, not by a result); any claim that Stage 2 re-weighting or healthy-cell downsampling improves transfer (not yet run; H-S2-09 open); inductive deployment, causal biology or recurrence risk.

## Next steps
- W8.2: the lock entry writes the 100-cell rule into `frozen_config_v1.json`, asks the user to confirm that patients below it are left out rather than kept as train-only, and settles `integration_plan.md` §14 item 10 (Arm P's pre-fitted probes).
- W8.3: the opt-in `--patients` filter is how a new run applies the rule; defaults stay unchanged.
- W8.6: the runner adds the SW scheme through an additive per-cell weight option in the W3 regularization contract (named in its refactor entry, defaults unchanged).
- W8.7: Arm D runs with the Stage-2-only placement as primary; the Stage 1 placements are attribution read-outs.
- W0.3: the `.gitattributes` change ships with the push, after the user's go-ahead.
- W0.15 (new): decide which figures and PDFs are tracked for collaborators (the `.gitignore` patterns above).
- W0.14: AGENTS.md §10 still describes W0.2 as an open decision; the owner of AGENTS.md updates that line.
