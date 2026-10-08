---
id: 2026-05-15_stage0-value-added-plan
date: 2026-05-15
type: decision
title: Bottom-up Stage 0 gene-space value-added workflow planned
asked_by: not_recorded
asks: []
implemented_by: unknown
questions: [Q3]
workstream: W3
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: 91c296b  # retro: first commit containing the plan (2026-07-02); moved to mrd_stage0_2/plans in db694a3
experiment: none
backfilled: true
date_source: mtime  # plan file mtime 2026-05-15 19:47 UTC; the plan has no date line
supersedes: []
---

## Question
(reconstructed from scripts/mrd_stage0_2/plans/stage0_geneset_value_added_workflow.md) Does a Stage 0 gene space add malignant-vs-healthy signal under matched downstream modeling (bottom-up), rather than only measuring what is lost when the full 34-program panel is pruned (the top-down April pruning analysis)? The asker is not recorded (user planning); there is no ask ID.

## Why it matters
Stage 0 design for Q3. The plan fixed the control logic (size-matched HVG, fixed HVG anchors, all_filtered) that the canonical Set A screen and the Set B manuscript-axes panels later used. It also holds the roadmap for the huCIRA prior and the relapse 4-class task (H-A-02).

## Estimand / null
n/a (planning). The planned comparator for every knowledge panel is size-matched HVG and fixed HVG anchors under the same Stage 1 DR suite and the same L1/L2/EN Stage 2 paths. Ranking is malignancy-first; structure-preservation metrics are reported separately. No C policy (no fit).

## Implementation
- Plan: scripts/mrd_stage0_2/plans/stage0_geneset_value_added_workflow.md (status line 'planning').
- Design elements: a unified panel manifest; old34 single-geneset and group panels; huCIRA as a separate prior; size-matched HVG, fixed HVG anchors and all_filtered controls; a DR suite of PCA, FA, FactoSig varimax/promax and cNMF; L1/L2/EN logistic paths; a secondary MRD+Relapse 4-class task.
- Precursor: the April 2026 old-geneset pruning experiment 20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0, outside the canonical experiment.
- No compute.

## Artifacts
none

## Results
- Planned HVG anchor sizes 500 / 1,000 / 3,000 / 10,000 genes. These are the Set A anchors of the 2026-05-25 screen (experiment_config.yaml, stage0.hvg_anchor_sizes).
- The 2026-07-02 rehydration dossier identifies this plan as the design blueprint for the Set B manuscript-axes panels (2026-06-22).
- The huCIRA branch and the relapse 4-class task were still not started on 2026-10-08.

## Claim gate
Supports (descriptive): nothing; this is a design decision.
Does not support: any claim that curated priors add signal over HVG controls or matched random gene sets.

## Next steps
- W4.7: read the Set B scorecards that this plan motivated (H-S0-03).
- W0.12: document how the old34 gene sets were chosen.
