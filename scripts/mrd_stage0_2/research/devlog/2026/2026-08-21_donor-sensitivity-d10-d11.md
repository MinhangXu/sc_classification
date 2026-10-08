---
id: 2026-08-21_donor-sensitivity-d10-d11
date: 2026-08-21
type: analysis
title: Donor-healthy downsampling sensitivity v1 (7 targets) and v2 overnight expansion (681 targets)
asked_by: meeting:2026-07-24
asks: [A-20260724-05]
implemented_by: unknown
questions: [Q2]
workstream: W4
analyses: [D10, D11]
hypotheses: [H-S2-04]
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
compute: screen-gpu  # v2 runner in screen donor_ds_v2 (cuML); v1 from the notebook
c_policy: global_selected  # each target keeps its pooled-LOPO-selected setting at every donor fraction
training_n_changed: true  # donor-healthy training cells removed at fixed C (W11)
status: done
commit: b60a6ee  # retro: notebooks and runner first committed in b60a6ee
run_head: 43f19be
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: log  # logs/donor_sensitivity_v2/donor_downsampling_v2_20260821_042647.log; v1 findings 2026-08-21 03:56 UTC
supersedes: []
---

## Question
(reconstructed from the narrative guide and the 2026-10-08 chronology reconstruction; topic A-20260724-05) With 58,499 donor-healthy cells against 50 recipient-healthy cells, is the probe learning donor-vs-malignant? Are the Stage 1 driver factors stable when donor cells are removed?

## Why it matters
Q2 robustness and the donor-shortcut alternative (ALT-DONOR, H-S2-04). The result decides whether donor abundance needs a dedicated rescue analysis (W4.6).

## Estimand / null
- D10 (fixed basis): change in held-out AP (ΔAP) after removing donor-healthy training cells only, stratified by patient x predicted cell type, at fractions 1, 0.5, 0.25 and 0.1 (3 seeds below 1), with the full unchanged held-out patient.
- D11 (basis refit): transductive PCA/FA refit on the retained cells, Hungarian-aligned to the full-donor refit; score r, loading cosine and top-10 Jaccard of the supervised factors.
- C policy: fixed at each target's globally selected setting. Training n changes, so L1 collapse at fixed C is expected (AGENTS.md R5); this is recorded as history (W11).

## Implementation
- v1: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_donor_downsampling_sensitivity.ipynb, 7 targets, 490 fits; caches from 2026-07-24.
- v2: scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py via run_stage2_old34_donor_downsampling_sensitivity.sh, launched by the user as screen -S donor_ds_v2 (started 2026-08-21 04:26 UTC). 640 DR + 41 direct-gene targets x 7 held-out patients x 10 masks; the D11 arm covers 21 representations.
- The merge crashed on 80 empty supervised CSVs; the user completed it with --merge-only in the same screen on 2026-09-03 18:54 UTC. The resume command is recorded in research/campaigns.csv (row donor_sensitivity_v2).
- v2 notebook: scripts/mrd_stage0_2/notebooks/stage0_2/stage2_old34_donor_downsampling_sensitivity_v2.ipynb (2026-08-21 16:33 UTC), not re-run after the merge-only (W0.5). The v1 notebook's final validation cell raises a stale plot-contract assertion (W0.5).

## Artifacts
- analysis/scorecards/stage2_diagnostic/donor_sensitivity_v1/
- analysis/scorecards/stage2_diagnostic/donor_sensitivity_v2/
- analysis/figures/stage2_diagnostic/donor_sensitivity_v1/
- analysis/figures/stage2_diagnostic/donor_sensitivity_v2/
- logs/donor_sensitivity_v2/

## Results
- [H-S2-04 qualifies] v1 at 25% donors: IFN-alpha PCA40 mean ΔAP −0.0811, FA40 −0.0489, direct genes −0.0357, each with 100% all-zero Stage 2 fits (L1 at C=1e-4), so AP = π by construction.
- [H-S2-04 qualifies] v2: 47,670 fits over 681 targets; 80/681 targets are all-zero already at full donor and 142/681 at 25%. Mean collapse at 25%: L1 0.604, elastic net 0.303, L2 0.000; L2 mean ΔAP −0.0002 (median +0.0001).
- D11 at 25%: median supervised-factor score r 0.791 across PCA targets and 0.845 across FA targets; IFN-alpha FA40 0.952 / 0.943 / 0.818 and PCA40 0.854 / 0.854 / 0.333 (score r / loading cosine / top-10 Jaccard).
- Retained donor-healthy cells: 29,268 (50%), 14,665 (25%) and 5,950 (10%) of 58,499.

## Claim gate
Supports (descriptive): donor downsampling at fixed C pushes sparse probes to the all-zero solution; under L2 the held-out AP barely moves.
Does not support: a donor-identity shortcut (the AP loss is a fixed-C L1 artifact), donor-vs-recipient invariance (only 50 recipient-healthy cells), or inductive stability (D11 stays transductive).

## Next steps
- W4.6: re-run D10 with condition-specific C (--resume; user screen).
- W0.5: re-run the v2 notebook after the merge-only and fix the v1 validation assertion.
