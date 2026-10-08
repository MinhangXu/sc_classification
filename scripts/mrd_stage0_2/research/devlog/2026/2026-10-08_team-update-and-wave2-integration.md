---
id: 2026-10-08_team-update-and-wave2-integration
date: 2026-10-08
type: docs
title: Team update for 2026-10-08 and integration of the wave-2 entries (D15b-D19, W7 apps) into the registry, campaigns, workstream READMEs and generated indexes
asked_by: user
asks: []
implemented_by: claude-code
questions: [meta, Q1, Q2, Q3, QI]
workstream: W0
analyses: []
hypotheses: [H-S1-06]
verdicts: [H-S1-06=supported_descriptive]  # carried forward from 2026-10-08_d17-basis-variant-pilot; see Estimand / null
alts: []
closes: []
advances: [W0.1]
claim_level: descriptive
status: done
commit: pending
run_head: d387eda  # HEAD when written; the user's wip checkpoint d387eda sits on top of b60a6ee
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
(reconstructed from the 2026-10-08 integration session; the request was not recorded verbatim) Integrate the six wave-2 devlog entries and their code into the research spine, and rewrite the user's 2026-10-08 report (research/asks/2026-10-08_team-report.md; A-20261008-01..32) as a team-ready update with Done / Discussed / TODOs / Scientific questions, the new repository layout, the wave-2 results, and the questions teammates are likely to ask.

## Why it matters
Meta: teammates need one dated document that says what changed and where things now live, and the generated STATUS.md and CLAIMS.md must reflect the wave-2 verdicts, task closures and registry states before anyone builds on them (W0.1).

## Estimand / null
n/a (docs and integration; nothing is computed). One derivation rule matters: the current verdict of a hypothesis is taken from the latest declaring entry, ordered by date and then by id. All wave-2 entries share the date 2026-10-08 with the seed entry 2026-10-08_research-spine-adopted, whose id sorts after the d15b-d19 ids. Without a later re-declaration, the seed's H-S1-06=open would therefore override the H-S1-06=supported_descriptive that 2026-10-08_d17-basis-variant-pilot declares. This entry sorts after the seed and re-declares that verdict unchanged; the evidence stays in the D17 entry. The D19 verdicts (H-S2-08=supported_descriptive, H-S2-07=retired) already equal the seed's values and need no carry-forward.

## Implementation
- Moved the six staged wave-2 entries into `scripts/mrd_stage0_2/research/devlog/2026/`: 2026-10-08_d15b-patient-count-curve, 2026-10-08_d16-factor-patient-matrix, 2026-10-08_d17-basis-variant-pilot, 2026-10-08_d18-factor-cards, 2026-10-08_d19-regularization-contract, 2026-10-08_w7-atlas-strip-factor-explorer. Content kept; two wording fixes:
  - D19 Question: the asks file is condensed wording (capture: paraphrase), not verbatim;
  - D17 Next steps: the unnamed proposed backlog line is now W2.7.
- Registry `scripts/mrd_stage0_2/research/analyses.csv`: D16, D17, D18 and D19 planned → current; D15b planned → partial; code paths (including the .sh wrappers) and findings_md filled; D17 notes now say where its bases persist; D18 asks gain A-20261008-21 and D19 asks gain A-20261008-04 (both cited by their entries).
- Campaigns `scripts/mrd_stage0_2/research/campaigns.csv`: new planned row `patient_count_learning_curve_v1` (D15b full design, screen W6_d15b, relaunch_policy resume_only) with the verified resume_command `bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh` (`bash -n` and the runner's `--help` pass; EXP comes from SC_EXPERIMENT_ROOT/SC_EXPERIMENT_ID).
- Workstream READMEs W1, W2, W3, W5, W6 and W7: "Code in this directory" replaced with the packages' text (result numbers removed per R12), status lines updated, front-matter status ready → active for W2, W3 and W6, owned-row lists updated, and four backlog lines added from the entries' Next steps: W1.9, W2.7, W3.8, W6.6. W3 conventions state the elastic-net entry threshold 4·l1_ratio/(C·n_train). W5 now carries the full factor_cards_v1 schema (copied from the bundle's bundle_schema_v1.md). W7 conventions name the agent-written verification slug w7_apps_check_v1.
- Team update: `scripts/mrd_stage0_2/research/team_updates/2026-10-08_team_update.md`. It starts from the understanding pass's corrected report; every wave-2 number in it was read from the six entries and checked against the findings.md files of the five analysis slugs and w7_apps_check_v1.
- `hypotheses.md` and `THEME.md` were not edited (no wave-2 entry proposes a new hypothesis or a change to blocking or alternatives; THEME changes wait for the user's milestone decision).
- Commands (repo root, python = the sc-classification-rapids-py311 interpreter with CONDA_PREFIX set): `python tools/research_index.py write`, `python tools/research_index.py check`, `python tools/research_index.py check --strict-artifacts`, and `python -m pytest -q tests` with pytest on PYTHONPATH from a scratch library (pytest is not installed in the env; W0.4).

## Artifacts
none

## Results
- Devlog: 45 entries after integration (38 before wave 2, 6 wave-2 entries, and this one).
- `research_index.py check` and `check --strict-artifacts`: 0 errors, 0 stale generated files, and the 2 pre-existing W11 history warnings (2026-08-21_donor-sensitivity-d10-d11, 2026-10-02_d15-cell-count-learning-curve). The four W06 warnings (D16-D19 planned but cited by done entries) are gone.
- Full test suite: 242 passed (the earlier failure of test_research_index.py::test_real_repo_check was only the stale generated files).
- Derived task states now include W1.1, W2.1, W3.1, W3.2, W5.1, W7.1 and W7.2 done, and W2.2, W5.3, W5.5 and W6.1 in progress, as rendered in STATUS.md §2. Current verdicts: H-S1-06 supported_descriptive (declared by D17, carried forward here), H-S2-08 supported_descriptive, H-S2-07 retired. STATUS §7 lists the D15b run as pending, next to the stalled Set C run.

## Claim gate
Supports (descriptive): the team update restates the wave-2 devlog Results and the existing claim boundary (THEME §6); it adds no new evidence.
Does not support: anything beyond the cited entries; in particular the verdict carry-forward for H-S1-06 rests entirely on 2026-10-08_d17-basis-variant-pilot (IFN-alpha pilot scope, transductive-vs-inductive on one panel).

## Next steps
- W0.1: user review of the wave-2 code and docs, then the per-package commits proposed for C10+ (one per package, then one docs commit for the registry, campaigns and generated indexes).
- W6.1: the user runs `screen -S W6_d15b`, then `bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh`, detaches with Ctrl-a d, and verifies d15b_full_unit_status.json (no missing units) and d15b_full_reference_checks.csv (all pass); a follow-up type: run, status: done entry supersedes 2026-10-08_d15b-patient-count-curve and the campaigns row becomes complete.
- W7.1 / W7.2: the user restarts the d1_atlas screen session and starts W7_factor_explorer (commands in 2026-10-08_w7-atlas-strip-factor-explorer).
- W0.1: THEME review queue for the user's milestone decision (the generated STATUS §9 queue is empty because these entries share THEME's last_reviewed date): H-S1-06 now supported_descriptive (D17); H-S2-07 retired and H-S2-08 restated under tuned C (D19); H-S1-05 qualified and contradicted on IFN-alpha but still open (D16, D17); H-S2-06 alternatives partial (D15b); H-S2-03 and H-I-03 identifiability caveats (D18); two candidate THEME §7 corrections (the elastic-net entry threshold, and stored PCA scores being randomized-SVD U·S rather than the loading projection, W2.7); candidate figure additions to hypotheses.md (H-S2-08: D19B, D19C, D17A; H-S2-06: D15bA-D15bD).
