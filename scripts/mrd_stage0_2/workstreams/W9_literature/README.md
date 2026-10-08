---
id: W9
title: Literature reviews and routing of their recommended actions into tasks
status: active
questions: [Q3, QI, Q2]
---

# W9 — Literature reviews and routing of their actions into tasks

## Charter

User asks, as recorded (`research/asks/2026-10-08_team-report.md`, capture: paraphrase):
- [A-20261008-28] "Review single-cell ML literature for existing frameworks"
- [A-20261008-22] "Review interpretability frameworks from other fields (e.g. factor rotation in psychology; iterate with domain knowledge)"

Two reviews answered these asks on 2026-10-08. Both were produced by an agent-run web review and are homed here with a provenance header.

W9 is done when:
- both reviews live in this directory;
- every recommended action is routed to a W#.n task (table below);
- THEME §12 cites the reviews (W9.2);
- any addendum a routed task needs exists, with its own routing (W9.3).

W9 runs no analyses. Numbers inside the reviews are literature values or the reviewer's own scratch checks. They count as evidence only after the routed task reproduces them and records them in a devlog Results section (AGENTS R12).

## Builds on

- Devlog entry `2026-10-08_literature-reviews` (type: docs). It closes W9.1 and tags the findings the reviewer computed.
- `research/THEME.md`:
  - §5, the QI lens;
  - §7, the knife-edge correction to narrative-guide §4;
  - §8, the crosswalk, including the label remap below;
  - §12, related work.
- The hypotheses the routed tasks address: H-S2-08, H-S2-03, H-S2-01, H-S2-06, H-S1-04, H-S1-07, H-S1-02, H-I-03.
- The reviewer alternatives they address: ALT-PEN, ALT-ROT, ALT-SEL, ALT-SEED, ALT-DEGEN, ALT-DONOR, ALT-LABEL, ALT-GENESET, ALT-TRANS, ALT-CT.
- `scripts/mrd_stage0_2/notebooks/stage0_2/old34_inquiry_narrative_guide.md` §4 and §5: the method mechanics that Review A comments on.
- The repository audit, which lives in the outer umbrella repo at `reports/mrd_stage0_2_repository_audit/` and is not copied here.

## Backlog

- [W9.1] (P1) Home the two literature reviews and route every recommended action to a W#.n task, with the routing table in the README — done when: both reviews are committed and every action is mapped.
- [W9.2] (P2) Keep the related-work section (THEME §12) current at milestones — done when: THEME cites the reviews.
- [W9.3] (P2) Write an addendum review if a routed task needs one (e.g. MRD single-cell cohorts, or sample size for external validation) — done when: the addendum and its routing exist.

## Routing table

Row numbers follow each review's recommended-action list. They are local to this table and are not IDs. "§" points to the review section that argues for the action. A "gap" note means the target task's text does not yet name that sub-part. Gap notes are suggestions for the task owner and are not backlog changes.

### Review A — factor interpretability (`2026-10-08_factor_interpretability_review.md`)

| # | Recommended action (paraphrased) | Routed to | § | Notes |
|---|---|---|---|---|
| 1 | The IFN-α PCA40-vs-FA40 L1 gap at the stored C is a threshold knife-edge. Revise narrative-guide §4: P02 is a regularization artifact, and P07 is partly a real PCA advantage. | W3.2 | §3.1 | The C-matched comparison settles H-S2-08. The guide §4 revision is recorded as a THEME §7 correction, citing devlog `2026-10-08_literature-reviews`. |
| 2 | L1 entry-margin audit for every fold × representation. Flag L1 "method wins" and "drivers" whose margin is close to the threshold. | W3.1 | §3.1; §5 A1 | Part of the D19 contract. |
| 3 | Rerun donor downsampling and the D15 learning curve with C scaled to training n, or reselected per condition. | W4.6, W6.2 | §3.1; §5 A2 | AGENTS R5. |
| 4 | factosig_promax `loadings.npy` is a pattern matrix whose column scale is arbitrary. Upstream fixes: unit factor variance, save M or Φ, record `loading_semantics`, order factors by a scale-free criterion. | W5.4 | §1.2; §5 A4 | Upstream factosig issue. |
| 5 | Report pattern **and** structure top genes, plus Φ, for every oblique factor used in gene grounding, including the D14/D15 promax bases. | W5.1 | §1.2 | Factor-card field. |
| 6 | Add `effective_k` (non-zero FA components) to Stage 1 metadata. Treat K beyond it as over-extraction. | W5.1, W4.5 | §1.5; §5 A5 | |
| 7 | Add factor-score determinacy to the diagnostics and flag poorly determined factors. | W5.1, W4.5 | §1.4; §5 A5 | |
| 8 | Replace one-to-one Hungarian matching with many-to-one R², principal angles, and Tucker φ before and after Procrustes alignment. | W5.1 | §1.3 | |
| 9 | Compare Haufe activation patterns of the PCA40-L1, FA40-L2 and FA40-L1 knife-edge probes by Tucker φ. | W5.2 | §4.1; §5 A3 | |
| 10 | Within-span per-patient L2 direction-angle test (span of FA7/FA8, or of PC7–PC9). | W5.5 | §1.1; §5 A6 | |
| 11 | Multi-seed × patient-bootstrap Stage 1 with cNMF-style component clustering. Consider choosing K by stability. | W4.5 | §2.3; §5 A7 | Gap: choosing K by stability is not named in W4.5. |
| 12 | Per-patient Stage 1 refits (Procrustes plus Tucker φ, an invariance check), and per-patient centering of scores before Stage 2. | W2.3, W1.3 | §2.1; §5 A8–A9 | Gap: neither task names the centering ablation. Candidate W1.3 sub-bullet. |
| 13 | Patient-level stability selection along a C path; cluster-level selection or elastic net for promax. | W3.6 | §3.2; §5 A10 | |
| 14 | Nested LOPO for C, method and K. LOCO refit importance alongside fixed-model ablation. | W3.3 | §3.3, §4.2; §5 A11 | Gap: LOCO is not named in W3.3 (candidate for W3.3 or W5.1). |
| 15 | Target-rotate the FA loadings to independent IFN sub-signatures. Spectra comparator with `predicted.annotation`. Within-cell-type AUPRC. | W5.6 | §4.4, §4.3; §5 A12 | Within-cell-type AUPRC overlaps W5.3. |

### Review B — single-cell ML, transfer and baselines (`2026-10-08_scml_transfer_baselines_review.md`)

| # | Recommended action (paraphrased) | Routed to | § | Notes |
|---|---|---|---|---|
| 1 | Genotype/label-assay shortcut audit (the review's "D16"). Read the loadings of the P03-private factors for HLA, ERAP2, chrY and XIST genes. Compare P03 donor cells with P03 malignant cells. Score the recipient-healthy cells. Build a sex-match table. | W1.4 (D25) | §0, §1 | H-S1-07, ALT-DONOR. |
| 2 | Add a per-patient label-provenance column to D3, D7B and D13. Test P02 allograft and IL2/STAT5 lift without the immunophenotype genes. | W1.4 (D25) | §0, §1 | Gap: the immunophenotype-gene ablation is not named in W1.4 or W1.8. Candidate W1.8 sub-bullet. Provenance is confirmed through the W8.1 email. |
| 3 | Ablate sex-chromosome, HLA, ERAP2 and HLA-DRB5 genes from HVG3k, HVG10k and full_34. Rerun the matching D1 rows and D14. | W1.8 | §1 | |
| 4 | Stage 2 retuning: C and penalty by inner LOPO, plus patient-balanced weights. Report AUROC and AUPRC/π next to lift. | W3.1, W4.1 | §2, §3 | Gap: patient-balanced Stage 2 weights are not named in W3.1. |
| 5 | Inner-LOPO selection of DR method and K. Restate the D1 winners under nested selection. | W3.3 | §2, §3 | ALT-SEL. |
| 6 | Narrow inductive Stage 1 (IFN-α PCA40/FA40 × the supported patients), then a patient-balanced inductive variant. | W2.1 | §2 | H-S1-06. |
| 7 | One-score comparators (UCell, AUCell, singscore, unnormalized ssGSEA, control-binned score, mean-z) as single-feature Stage 2. | W3.4 | §4 | D24. |
| 8 | Null Model 1: tier 1 one-score nulls and tier 2 DR + Stage 2 nulls. Add a within-patient label-permutation null. | W4.2 | §4 | The self-contained label-permutation null is W4.3. ALT-GENESET. |
| 9 | The review's "D15b" (the repo's D15 v2): cell-depth curve with C and penalty retuned per subsample. | W6.2 | §5 | |
| 10 | The review's "D15c" (the repo's D15b): patient-drop curve with a power-law fit and patient-level intervals. | W6.1 | §5 | W6.1 uses the frozen transductive basis, labelled as such. The review's inductive refit per subset waits for W2.2. |
| 11 | The review's "D15d": k patients × a cap on malignant cells per patient. | W6.3 | §5 | No ID reserved yet. |
| 12 | Uncertainty: per-patient AUPRC intervals, a patient-cluster bootstrap, exact patient-level tests, and no naive CV standard errors. | W4.1 | §2 | D22. |
| 13 | Model-based sharedness: MOFA+ multi-group, with and without per-group centering, and per-patient cNMF meta-programs. | W1.3 | §2 | An optional comparator in W1.3. |
| 14 | Ankit's ladder inside nested LOPO: supervised PCA, then sPLS-DA, then a linear supervised autoencoder with a λ sweep. | W3.7 | §3 | |
| 15 | Knowledge-guided label-free factorization (Spectra, f-scLVM, PLIER). Name PC8's program by contrasting IFN-γ-specific with type-I-specific genes. | W5.6 | §3, §1 | The IFN-γ vs type-I note is part of W5.2. |
| 16 | Write and lock an external-validation protocol for the collaborator cohort before the data arrive. | W8.2 | §1, §3, §5 | `../W8_new_cohort/integration_plan.md`. |
| 17 | Document why the old34 gene sets were chosen. Add a Stage 0 selection-bias caveat if the choice was informed by this cohort. | W0.12 | §0, §3 | |

### Other suggestions in the review texts (not in either action list)

Lower priority. They are listed so that nothing is lost:
- A §1.1: varimax-rotated PCA and FA Stage 1 variants, after a kurtosis check → candidate for W5.6.
- A §1.1: report eigengaps next to any factor named as a driver → W5.1.
- A §1.2, §4.1: structure coefficients for probes on correlated or L1-zeroed factors → W5.1, W5.2.
- A §1.3: Tucker congruence bands for the donor-downsampling stability metric → W4.6.
- A §4.4: projectR-style inductive projection as a check on the transductive fit → W2.1.
- B §4: VISION autocorrelation per panel and factor → candidate for W5.3.
- B §4: panel lift against the cell-type-only ceiling, and lift within cell types → W5.3.
- B §1: the HLA-II-low immune-escape reading of the missed malignant cells, tested with ADTs → W5.3.
- B §1: frame Stage 2 as label transfer inside prior gene spaces, not as a replacement for the label-generating tools → THEME §6 language (AGENTS R6).
- B §1, §3, §5: require the same label pipeline for new patients, and freeze λ, C, K and gene lists before unblinding → W8.2.

## Label remap note

Review B wrote its own analysis labels before the registry IDs were fixed:

| Label in the review | Repo ID or task |
|---|---|
| "D16" genotype/label-assay audit | **D25**, task W1.4. The repo's D16 is the factor × patient matrix (W1.1). |
| "D15b" cell-depth curve with C retuned | **D15 v2**, task W6.2 (slug `cell_count_learning_curve_v2`) |
| "D15c" patient-drop curve | **D15b**, task W6.1 |
| "D15d" 2D design | task **W6.3** (no ID reserved yet) |

Two more notes:
- Review A's synthesis rows A1–A12 (§5) are local to that file. They are not ask IDs (A-YYYYMMDD-NN) or repository-audit IDs.
- Both reviews use older layer names ("Layer 4/5"). THEME §8 maps them to registry IDs.

## Code in this directory

- `README.md`: this file (charter, backlog, routing).
- `2026-10-08_factor_interpretability_review.md`: Review A (A-20261008-22). The body is verbatim except for two path rewrites, which its header lists.
- `2026-10-08_scml_transfer_baselines_review.md`: Review B (A-20261008-28). The body is verbatim; its header lists one DOI erratum.
- No code and no screen sessions.

## Conventions

- **Reviews are frozen snapshots.** Do not edit a review's body. Put errata in its provenance header with a date, or write an addendum (W9.3).
- **Addenda** are named `<YYYY-MM-DD>_<topic>_review.md`. Use the same flat front-matter keys (title, date, produced_by, scope, routed_tasks), add a provenance header, and add a routing subsection to this README.
- **Artifact slugs:** none. W9 writes nothing to the experiment dir.
- **C policy (AGENTS R5):** does not apply to W9 itself. It binds every routed task that refits a probe.
- **Import header:** not applicable; there is no code here.
- **Paths:** repo-relative for code and experiment-dir-relative for artifacts. No absolute paths (lint E07).
- **Numbers:** cite devlog entries rather than restating numbers here (AGENTS R12).
