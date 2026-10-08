---
id: 2026-05-25_sharedness-scorecard-design
date: 2026-05-25
type: decision
title: Three-view Stage 2 design (discovery, sharedness LOPO, patient-specific) specified
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q1]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: 91c296b  # retro: plan first committed in 91c296b; the multiobjective runner first committed in 4ada734 (both 2026-07-02)
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # 'Date: 2026-05-25' in the plan; its mtime 2026-07-02 is the reference sweep
supersedes: []
---

## Question
> "This note refines the new MRD Stage 0/1/2 experiment plan after the discussion about whether heldout evaluation is necessary. The short answer is: keep heldout evaluation, but do not let it be the only truth."

Source: the opening paragraph of scripts/mrd_stage0_2/plans/stage0_mrd_sharedness_scorecard_plan.md. The question itself ('is held-out evaluation necessary?') is reconstructed from that sentence. The discussant is not recorded; there is no ask ID.

## Why it matters
This fixed the Stage 2 design behind every later Q1 analysis: one shared Stage 1 basis, read three ways. It is also the origin of the transductive caveat that H-S1-03 carries and that W2 now tests.

## Estimand / null
Three Stage 2 objectives on the same Stage 1 scores:
- discovery: full-cohort apparent fit;
- sharedness LOPO: train on all but one patient and evaluate the held-out patient (label transfer on a transductive atlas, not inductive deployment);
- patient-specific: within-patient fits on the shared basis.

Every scorecard row is tagged stage1_fit_scope_note=transductive_all_eligible_cells. No null was specified at this stage.

## Implementation
- Plan: scripts/mrd_stage0_2/plans/stage0_mrd_sharedness_scorecard_plan.md.
- Runner that implements it: scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py.
- Shortlist rules: always include full_34, core_only and the HVG anchors; force-include the IFN, NF-kB/TNF, MHC and cytokine panels; canonicalize the quick L2 scorecard.

## Artifacts
none

## Results
- Defined the 3 Stage 2 objectives (discovery, sharedness LOPO, patient-specific) that every later run uses.
- Labelled LOPO explicitly as label transfer on a transductive atlas (stage1_fit_scope_note=transductive_all_eligible_cells), not inductive deployment.

## Claim gate
Supports (descriptive): nothing; this is a design decision.
Does not support: reading LOPO lift as inductive deployment performance.

## Next steps
- W2.1: the inductive Stage 1 pilot that tests the transductive caveat (H-S1-06).
