---
id: 2026-08-25_repository-audit
date: 2026-08-25
type: docs
title: Read-only repository audit and claim-readiness matrix (outer umbrella repo)
asked_by: user
asks: []
implemented_by: unknown
questions: [Q2, meta]
workstream: W0
analyses: []
hypotheses: [H-S1-03]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
status: done
commit: none  # outer-repo documents only; tracked in the umbrella repo commit 3ba49e7
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: header  # 'Audit date: 2026-08-25'; D9 scaler addendum verified 2026-08-26
supersedes: []
---

## Question
> "at what biological resolution MDS MRD malignancy shows transferable phenotypic convergence"

Source: the motivating question as stated in the audit's executive summary (reports/mrd_stage0_2_repository_audit/audit_report.md in the umbrella repo), plus 18 explicit audit questions. The user's prompt text was not saved; no ask ID.

## Why it matters
Q2 and the transductive scope (H-S1-03). The audit's claim-readiness table and open decisions seed THEME §6 and §11.

## Estimand / null
n/a (inspection only: lineage, run inventory, a science-to-repo matrix and open decisions). No rerun, no new fitting, no writes into the experiment.

## Implementation
- Umbrella-repo files under reports/mrd_stage0_2_repository_audit/: audit_report.md, open_decisions.md, run_inventory.tsv, science_to_repo_matrix.tsv, artifact_lineage.md and d9_scaler_cancellation_lopo_folds.tsv (2026-08-26 addendum).
- An experiment-context-map Cursor canvas was made on 2026-08-23 (outside the repo; not canonical).
- These files live outside the sc_classification submodule, so a clone of the submodule does not contain them.

## Artifacts
none

## Results
- 12-claim readiness table: inductive-vs-transductive agreement (claim 9) and IFN-HLA-II decoupling (claim 11) were not implemented; Layer 6 was blocked; 86/643 representations have 0/7 positive lifts.
- [H-S1-03 supports] 2026-08-26 addendum: the D9 LOPO train-fold StandardScaler cancels the all-cell gene z-score, so D9 LOPO is effectively inductive for continuous gene scaling only, while the DR representations remain transductive.

## Claim gate
Supports (descriptive): cohort-conditional transductive label transfer over 7 supported held-out patients, as the audit states.
Does not support: new-patient inductive deployment, a matched-control (Layer 6) claim, or IFN-HLA-II conditional decoupling.

## Next steps
- W2.1: the inductive Stage 1 pilot (audit open decision 1).
- W1.4: the label-provenance and genotype shortcut audit.
