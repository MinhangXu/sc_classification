---
id: 2026-10-08_research-spine-adopted
date: 2026-10-08
type: decision
title: In-repo research spine adopted (AGENTS.md, research/, devlog, workstreams); Cursor tracker canvas frozen
asked_by: user
asks: [A-20261008-11, A-20261008-23, A-20261008-29, A-20261008-30, A-20261008-31, A-20261008-32]
implemented_by: claude-code
questions: [meta]
workstream: W0
analyses: []
hypotheses: [H-S0-01, H-S0-02, H-S0-03, H-S0-04, H-S1-01, H-S1-02, H-S1-03, H-S1-04, H-S1-05, H-S1-06, H-S1-07, H-S2-01, H-S2-02, H-S2-03, H-S2-04, H-S2-05, H-S2-06, H-S2-07, H-S2-08, H-I-01, H-I-02, H-I-03, H-A-01, H-A-02]
verdicts: [H-S0-01=supported_descriptive, H-S0-02=supported_descriptive, H-S0-03=open, H-S0-04=open, H-S1-01=supported_descriptive, H-S1-02=supported_descriptive, H-S1-03=supported_descriptive, H-S1-04=supported_descriptive, H-S1-05=open, H-S1-06=open, H-S1-07=open, H-S2-01=open, H-S2-02=open, H-S2-03=supported_descriptive, H-S2-04=not_supported, H-S2-05=supported_descriptive, H-S2-06=supported_descriptive, H-S2-07=retired, H-S2-08=supported_descriptive, H-I-01=supported_descriptive, H-I-02=grant_gap, H-I-03=supported_descriptive, H-A-01=grant_gap, H-A-02=open]
alts: []
closes: []
advances: [W0.1]
claim_level: descriptive
status: done
commit: pending
run_head: b60a6ee
dirty_run: true
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
(reconstructed from the user's 2026-10-08 report; the item texts are recorded in research/asks/2026-10-08_team-report.md) The inline ask (A-20261008-11): reorganize the repository for agent-agnostic context engineering, with a dated development log (day, question, implementation), per-workstream development directories and a central theme for the paper (A-20261008-29), organized around Q1 shared vs patient-specific (A-20261008-30), Q2 strength of evidence (A-20261008-31) and Q3 what DR adds over direct-gene classification (A-20261008-32). The report also states that no causal claim is possible without experimental intervention (A-20261008-23).

## Why it matters
Meta: every later task starts from AGENTS.md and STATUS.md and ends with a devlog entry, so any coding agent or collaborator who clones only sc_classification can see the questions, the evidence, the numbers and the claim boundary. This entry declares the initial verdict of every hypothesis (lint E14) and moves the tracker of record from the Cursor canvas into git.

## Estimand / null
n/a (decision). Verdict semantics: supported_descriptive means descriptive evidence on the transductive basis over the seven supported patients; supported_publication requires every listed reviewer alternative addressed, every blocking task done, clean-tree evidence and figures (lint E13). No hypothesis starts at supported_publication.

## Implementation
- Overlay (wave 1, uncommitted until the user's go-ahead): AGENTS.md (canonical agent instructions) with thin tool shims; scripts/mrd_stage0_2/research/ (study_config.json, THEME.md, hypotheses.md, alternatives.csv, analyses.csv, campaigns.csv, asks/, devlog/, legacy/; generated STATUS.md, CLAIMS.md and devlog/INDEX.md); scripts/mrd_stage0_2/workstreams/W0-W9; tools/research_index.py; src/sc_classification/utils/paths.py; tests.
- Hypothesis migration: the 19 rows of the Cursor canvas stage0-2-hypothesis-tracker.canvas.tsx (SNAPSHOT 2026-10-02b) plus 5 new rows (H-S1-05, H-S1-06, H-S1-07, H-S2-07, H-S2-08) become the 24 static blocks of research/hypotheses.md. Verdicts follow the migration table and live only in devlog front matter (this entry).
- The canvas keeps its data arrays but gains a FROZEN banner; its verbatim pre-freeze copy is scripts/mrd_stage0_2/research/legacy/stage0-2-hypothesis-tracker_2026-10-02b.canvas.tsx.
- The narrative guide's 'Current claim boundary' (scripts/mrd_stage0_2/notebooks/stage0_2/old34_inquiry_narrative_guide.md at b60a6ee) is superseded as the canonical claim list by research/THEME.md §6; its pre-migration text is preserved verbatim below.
- No compute.
- Integration (wave-1 integrator, same day): `python tools/research_index.py write`, `check` and `check --strict-artifacts` exit 0; the two remaining W11 warnings are backfilled history entries (2026-08-21_donor-sensitivity-d10-d11, 2026-10-02_d15-cell-count-learning-curve) that changed training n under global_selected C. Because `--strict` turns every warning into an error, `check --strict` cannot pass while these history entries exist; it is not part of the wave-1 gate.
- Integration fix: research/asks/2026-10-08_team-report.md now declares capture: paraphrase instead of verbatim_from_report, because its item texts are the understanding pass's condensed wording and the report itself is not stored; the workstream README charters say 'quoted as recorded' instead of 'verbatim'.
- Integration fix: 2026-09-03_d12-d13-gsea-hvg-lowsupport now has asked_by: meeting:unknown, matching the D12/D13 rows of analyses.csv and the notebook builder's 'two meeting follow-ups'.
- Integration fix: numbers that the narrative guide cites but no entry held were re-read from the artifacts and added to the cited entries (per-fold L1 thresholds and gaps, the C=1.02e-4 entry and the F6 fold-scale means in 2026-10-08_team-report-devlog-pivot; the full D15 apoptosis curve and per-patient values in 2026-10-02_d15-cell-count-learning-curve); the guide's C5 and F9 orientation citations now name the entries that hold their numbers, and its preamble says that the F1-F9 reading guide also quotes detail values read directly from the tables it lists (spot-checked against the F2, F3 and F6 tables).
- Integration fix: 2026-10-08_literature-reviews records review B's GSVA DOI erratum; AGENTS.md §6 notes that scripts in a workstream's anchors/ directory use parents[5].

## Artifacts
none

## Results
- Initial verdicts for 24 hypotheses: supported_descriptive 12 (H-S0-01, H-S0-02, H-S1-01, H-S1-02, H-S1-03, H-S1-04, H-S2-03, H-S2-05, H-S2-06, H-S2-08, H-I-01, H-I-03), open 8 (H-S0-03, H-S0-04, H-S1-05, H-S1-06, H-S1-07, H-S2-01, H-S2-02, H-A-02), not_supported 1 (H-S2-04), retired 1 (H-S2-07), grant_gap 2 (H-I-02, H-A-01).
- The canvas value in_flight is retired as a verdict: H-S0-04 starts as open, and the stalled Set C run state lives in research/campaigns.csv (status stalled since 2026-09-05).
- Support rule, recorded once in research/study_config.json: a supported patient has at least 10 malignant cells and both classes (P01, P02, P03, P04, P06, P07, P09; MIN_MALIGNANT_SUPPORT = 10 in the 2026-07-03 catalog). D14's 20-cell malignant stratum is an analysis parameter, not the support rule.
- Reversal of the Cursor rule: the umbrella repo's .cursor/rules/stage0-2-hypothesis-tracker.mdc (2026-09-08) said 'Do not create a second tracker markdown that duplicates the table.' From 2026-10-08 the tracker of record is in git (research/hypotheses.md plus devlog verdicts, rendered in STATUS.md and CLAIMS.md); the canvas frozen at SNAPSHOT 2026-10-02b is an optional non-canonical view, and the rule is rewritten as a pointer without that line in the same change set.
- The narrative guide's 'Current claim boundary' as of b60a6ee, verbatim (pre-migration; the canonical list is now THEME §6):

> ### Supported as descriptive findings
>
> - The 41 old34 panels exhibit substantial patient × DR-method × K heterogeneity.
> - Among the 41 biological panels' in-sample best-median DR cells, K=40 wins 24, K=20 wins 13, and K=10 wins 4; method winners are balanced (promax 12, FA 11, PCA 9, FactoSig 9), so there is no global dominant method but no method invariance either.
> - The focused selected-DR median lifts are allograft 12.56 pp, apoptosis 11.52 pp, and TNFα/NF-κB 18.06 pp; their direct-gene medians are 0.23, approximately 0, and 12.78 pp. These summaries remain patient-heterogeneous.
> - Allograft and apoptosis share only 15 covered genes across a 270-gene union (Jaccard 0.056). A mean-z one-score proxy is much weaker than selected DR for allograft/TNF and intermediate for apoptosis; this is not yet an AUCell/UCell/ssGSEA comparison.
> - Selected-DR cell rankings for the three focus panels reconstruct stored patient AP to within 3.2e-8. Cell-type/ADT error audits are descriptive because the rank cutoff uses held-out prevalence and detailed treatment-regimen metadata are absent.
> - Constant-score AP equals patient prevalence exactly under the implemented metric.
> - Lift remains positively rank-associated with prevalence/support across many representations.
> - Direct-gene/no-DR Stage 2 baselines exist for all 41 panels.
> - DR usually improves pooled-cell AP but not consistently patient-equal median/P20 lift.
> - IFN-α PCA factor 008 is the selected driver in P02/P07 while FA is all-zero there at the chosen L1 setting.
> - Donor downsampling can push sparse models to an all-zero solution.
> - FA factor 007 is relatively stable at 25% donor retention; PCA factor 008 is not comparably stable until the 50% condition.
> - The old34 GSEA union and HVG3,000 are HVG-enriched relative to a hypergeometric null (Jaccard 0.199; 65% of old34 is exclusive) and are not interchangeable Stage 0 panels.
> - P01/P06 have within-patient DE (157/232 genes under the triage rule) but best existing held-out lift remains 6.3/5.7 AP points; P07’s EMT panel is both DE-enriched and the best existing transfer (19.0 AP points).
>
> ### Not yet supported as publication-level claims
>
> - high lift is caused by patient-specific biology rather than support/prevalence;
> - FA is intrinsically inferior to PCA for IFN-α;
> - the model is invariant to donor-versus-recipient origin;
> - DR universally adds value over direct genes;
> - selected panels beat matched-size random gene sets;
> - reported “best” configurations are unbiased deployment estimates.
>
> Priority additions for the larger cohort are patient-level uncertainty, matched-size controls, nested/prespecified selection, inductive Stage 1 sensitivity, and a donor-downsampling analysis with condition-specific regularization selection.

## Claim gate
Supports (descriptive): the initial verdict table above as the starting state of the tracker; every supported verdict is descriptive, on a transductive Stage 1 basis over seven supported patients.
Does not support: any publication-level claim (no hypothesis is supported_publication; evidence from 2026-07-24 onwards has dirty_run true); causal biology, inductive deployment or recurrence risk.

## Next steps
- W0.1: integrate the overlay (research_index.py write and check exit 0, tests pass), then commit it in reviewable groups after the user's go-ahead.
- W0.2: the user decides the .gitattributes ipynb LFS line before any push.
- W0.3: push sc_classification_gpu and bump the outer submodule pointer after the go-ahead.
- W0.10: clean-tree reruns before any supported_publication verdict.
- W0.7: recover the verbatim 2026-07-24, 2026-08-03, 2026-08-20 and 2026-08-21 question sets.
