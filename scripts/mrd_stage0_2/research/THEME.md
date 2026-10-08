---
study: mrd_stage0_2
last_reviewed: 2026-10-08
reviewed_through: 2026-10-08_w7-atlas-strip-factor-explorer
canonical_experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
---

# THEME: knowledge-prior Stage 0-2 probe of post-transplant MDS MRD malignancy

> No numbers here. Cite H-, D-, F- and devlog IDs. Edit only at milestones (a verdict changes, a claim crosses §6, or paper text is requested) and bump last_reviewed and reviewed_through (the last devlog id reviewed, in INDEX order).

Numbers live only in devlog Results and are rendered by `CLAIMS.md`. Current verdicts, alternative states and open tasks are in the generated `STATUS.md`. Hypothesis statements are in `hypotheses.md`, reviewer alternatives in `alternatives.csv`, analysis IDs in `analyses.csv`, campaigns in `campaigns.csv` and cohort constants in `study_config.json`.

## §0 Working thesis

Classification as a biological probe. At what biological resolution (curated program, learned sub-axis, or patient) does post-transplant MDS MRD malignancy show transferable phenotypic convergence (Q1)? How strong is that evidence once prevalence/support, in-sample selection, fixed-C regularization and transductive fitting are accounted for (Q2)? What does a learned low-dimensional representation add over classifying on the program's genes directly (Q3)?

QI (interpretability, the PI's third question) is the lens through which any shared or patient-specific factor is named.

## §1 Design and estimands

- **Stage 0** restricts the gene space to a prior panel: single curated programs, biology-group unions, and broad controls (full_34, core_only, HVG anchors). A panel is a prior, not a result.
- **Stage 1** decomposes each panel with PCA, FA, FactoSig or FactoSig-promax at several K with a single seed, or keeps the panel genes as a direct-gene (no-DR) representation. Stage 1 is **transductive**: it was fit on every eligible cell, including the cells of the patient later held out. LOPO is therefore label transfer on a transductive atlas, not inductive deployment.
- **Stage 2** fits regularized logistic probes (L1, L2, elastic net) in three views: discovery on the full cohort, sharedness LOPO, and patient-specific fits (apparent, in-sample).
- **Support rule:** a supported held-out patient meets the minimum malignant-cell count and both-class rule in `study_config.json`. Other patients stay in training but are not summarized.
- **Effect size:** lift = held-out AUPRC minus the held-out patient's malignant prevalence π, with constant-score no-skill AP = π (H-I-01, D6). Normalized lift is secondary. Patient summaries follow Option B (patient-equal median and lower-tail P20 lift; decided in devlog 2026-07-03_figure-catalog-option-b).
- **DR versus direct genes** is always reported with three estimands together: pooled LOPO AP, median lift and P20 lift.
- **Variance estimands are kept separate** and never mixed in one sentence:
  1. label-free Stage 1 variance (PCA total variance or FA common variance);
  2. per-factor patient effect (eta-squared by patient within label, D14);
  3. Stage 2 attribution (coefficients, ablation, univariate held-out lift, D8 and F6).

## §2 Q1 Shared vs patient-specific (PI-Q1 + PI-Q2)

**Current answer.** On the transductive basis, supported patients' factor-score distributions overlap on nearly every biological program basis, so pooling them on one Stage 1 basis is reasonable and no biological program shows a patient-private subtype factor (D14, H-S1-04). Held-out lift above π reaches many supported patients for the immune and inflammatory programs, but it is concentrated in the highest-prevalence patients P02, P03 and P04 and ranks with prevalence across representations (D1, D1f, D1DD2, D6). Shared transfer and prevalence are therefore not yet separated, and H-S2-01 stays open. Patient-specific opportunity tags exist (D3) and the low-support patients carry within-patient DE (D13), but the patient-specific fits are in-sample and single-patient peaks are non-identifiable (H-S2-02, open). The only patient-private malignant axis is a P03 axis in broad control bags, not in any program; genotype, label provenance, batch and composition are not yet excluded (H-S1-07). For IFN-alpha, related directions in one shared subspace (PC8 for PCA, FA7 for FA) carry the signal in every patient where the probe is active (F9, D18, H-S2-08). On the IFN-alpha pilot, held-out transfer survives an inductive Stage 1 under per-fold tuned C and L2 in every supported patient, with small, sign-consistent transductive optimism (D17; H-S1-06, H-S1-03); W2.6 extends this beyond one panel. Whether the highest-prevalence patients shape the Stage 1 loadings, and not only the Stage 2 labels and selection, is tested only on IFN-alpha (D16, D17). There their share of score energy is near their cell share (slightly above it for PCA at the largest K), and leaving them out barely moves the discriminative axis; against random drops matched on malignant composition, the excess is gone for FA and small for PCA, while P01 (a low-prevalence, high-cell-count patient) has far more leverage (D17 composition-matched controls). On this panel the imbalance acts mainly through Stage 2 labels and selection. H-S1-05 stays open until W2.4 (allograft, apoptosis) repeats the test.

**What would change the answer.**
- An inductive Stage 1 beyond IFN-alpha (W2.6) on which held-out transfer collapses would turn "shared on a transductive atlas" into "shared only because the held-out patient helped define the basis" (H-S1-06; the IFN-alpha pilot D17 did not collapse).
- Leave-high-prevalence-out refits that move the loadings beyond composition-matched random drops and beyond the contrast patients (D17 controls, W2.4) would support H-S1-05.
- Support-matched lift (W1.5) that removes the patient ordering would favour a prevalence explanation of the patient pattern (ALT-PREV).
- The D25 audit (W1.4) could reclassify the P03 axis as genotype or label provenance.
- New patients (W8.1) are the only external test of the shared axes.

**Blocking tasks:** W1.2, W1.3, W1.4, W1.5, W2.4, W2.6, W4.1, W8.1.

**Candidate figures:** see §9 (Fig2, Fig3).

## §3 Q2 Strength of evidence

- **Effect size is settled:** lift above constant-score no-skill AP = π (H-I-01, D6), with Option B patient summaries.
- **There is no inferential framework yet.**
  - D6 rank correlations with prevalence are nominal, not for inference.
  - The D14 permutation null sits at its floor, so its calls rest on uncalibrated magnitude gates (D20 hardens this).
  - D15 is a cell-depth curve at a fixed patient mix with C not retuned; it says nothing about adding patients (H-S2-06; D15 v2).
  - D15b varies the number of training patients with C retuned on a frozen transductive basis: lift is already positive with one training patient, gains are small with wide patient-cluster intervals, and the curves are not monotone. Gains over one patient mix a tuning-protocol switch, the reduced design is unbalanced at two and five patients, and how a basis refit per training subset would change the curve is unknown (D15b; full design W6.1, inductive version W6.6).
  - The D10 loss of AP under donor downsampling is an L1-at-fixed-C collapse to an all-zero probe, so AP returns to π by construction; it is not a donor test (H-S2-04).
- **Still missing:** patient-level uncertainty and an AUROC companion (D22), matched random gene-set nulls (D21), nested selection (D23), Stage 1 seed and resample stability, and a variance decomposition with the three estimands of §1 kept apart (W4.4).

**Blocking tasks:** W4.1, W4.2, W4.3, W4.4, W4.5, W3.3, W6.1.

## §4 Q3 What DR adds

- **The answer depends on the estimand, and current selection is in-sample.** DR beats direct genes on pooled LOPO AP for most biological panels but far less often on patient-equal median or P20 lift (D9). Choosing the best DR cell by median lift instead changes the count (D1e, F4). Both choices pick a winner on the evaluation metric (winner's curse; ALT-SEL).
- **Controls favour direct genes:** for most of the broad HVG and full-union control bags, the direct-gene probe beats the best DR cell on median lift (D1e, F4).
- **The gains concentrate in the highest-prevalence patients** (F4, D1DD5), so a DR advantage is entangled with prevalence (ALT-PREV).
- **The prior is not an HVG alias** (D12, H-S0-02), so the program-level comparison is not a relabelled HVG comparison.
- **The D1DD9 mean-z one-score proxy is not a fair baseline:** it is not UCell, AUCell or ssGSEA and its Stage 2 is not matched (D24 replaces it).
- **The comparison is asymmetric:** direct-gene LOPO is effectively inductive for gene scaling (the D9 scaler-cancellation audit in the outer-repo audit; summarized in devlog 2026-08-25_repository-audit), while the DR basis is transductive (ALT-TRANS). An inductive DR basis (D17, one panel) and per-fold tuned C (D19) now exist; a fair verdict still needs inductive bases for the selected representations (W2.6) and nested selection (W3.3) (H-S1-02).
- **Many atlas L1 selections sit on the entry knife-edge** at the C-grid floor, and the atlas C was chosen with held-out outcomes, so fixed-C atlas cells do not rank methods (D19; H-S1-01).

**Blocking tasks:** W3.3, W3.4, W3.8, W2.6.

## §5 QI lens, with IFN-alpha as the worked example

- **Related axes, not one identical axis.** IFN-alpha PC8 and FA7 are matched directions with opposite raw signs (F3, F9), but PC8's signal is split across FA7 and FA8, and neither PC8 nor FA7 is individually identifiable (small eigengap, low determinacy; D18, H-I-03). Factor signs are arbitrary. Given the negative Stage 2 coefficient on PC8 in every active fold, the malignant-leaning pole is the **negative** PC8 group (TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1). The opposite pole (IFI27, PSME2, PSMA3, CNP, LAP3, USP18) is normal-leaning. FA7 shows the same contrast with flipped raw signs. PC8 is a combination axis: the best single genes carry much less held-out lift than the axis, and interferon-stimulated genes sit on both poles (F9).
- **The PCA/FA coverage gap is an L1 threshold at fixed C** (H-S2-08).
  - FA7 alone carries held-out lift in the patients the FA probe drops (F6, F9), and L2 on the same FA scores recovers them (F3).
  - The literature-review C sweep shows the gap is a knife-edge: a slightly larger C lets FA7 enter (devlog 2026-10-08_literature-reviews).
  - Under per-fold tuned C, FA reaches every supported patient; a residual PCA advantage remains only in P07, depends on the penalty and is a single-patient, non-identifiable difference (D19). The gap follows the L1 threshold on inductive bases too (D17).
  - The team's Stage 1 allocation hypothesis is retired (H-S2-07).
- **factosig_promax loadings are a pattern matrix** with unnormalized column scale. For oblique fits, report both pattern and structure top genes plus the factor correlations (W5.1, W5.4).
- **Review gates for naming any factor:**
  - name a factor from both poles, with the sign orientation and the Stage 2 penalty (L1, L2 or elastic net) stated;
  - only recurrent, sign-consistent factors support shared claims (W4.5);
  - dense L2 fits are not interpretable per factor;
  - check factor-score determinacy and effective_k before naming an individual factor (poorly determined FA factors are not programs).

**Blocking tasks:** W5.2, W5.3, W5.4, W5.5, W5.6.

## §6 Canonical claim list (supersedes the narrative guide's "Current claim boundary")

This list supersedes the "Current claim boundary" section of `notebooks/stage0_2/old34_inquiry_narrative_guide.md`. Its verbatim numeric pre-migration text is preserved in devlog 2026-10-08_research-spine-adopted. Numbers for each item are in `CLAIMS.md`. Language rules are in AGENTS.md R6. Labels S and N are stable references for talks and drafts.

### Supported as descriptive

- **S01** The old34 biological panels show substantial patient by DR-method by K heterogeneity (D1, D7; H-S1-01).
- **S02** Among the in-sample best-median DR cells, the larger K values win most panels and the method winners are balanced: no method dominates globally, and no method is invariant either (D1, D1DD2; H-S1-01).
- **S03** For the three focus panels (allograft rejection, apoptosis, TNF-alpha via NF-kB), selected DR reaches clearly positive median lift where direct genes are near zero for allograft and apoptosis and lower for TNF-alpha; the summaries remain patient-heterogeneous (D1DD3, D1DD5, F2; H-S1-02).
- **S04** Allograft and apoptosis share only a small part of their covered genes. A mean-z one-score proxy is much weaker than selected DR for allograft and TNF-alpha and intermediate for apoptosis; it is not yet an AUCell, UCell or ssGSEA comparison (D1DD6, D1DD9; H-S1-02).
- **S05** Selected-DR cell rankings for the focus panels reconstruct stored patient AP essentially exactly. The cell-type and ADT error audits are descriptive, because the rank cutoff uses held-out prevalence and treatment-regimen metadata are absent (D1DD10, D1DD11, D1DD12; H-S2-05).
- **S06** Constant-score AP equals patient prevalence exactly under the implemented metric: constant-score no-skill AP = π (D6; H-I-01).
- **S07** Lift remains positively rank-associated with prevalence and support across many representations (D6; H-S2-01).
- **S08** Direct-gene (no-DR) Stage 2 baselines exist for every biological panel and, since the control direct-gene run, for the control panels (D9, D1e; H-S1-02).
- **S09** DR usually improves pooled-cell AP but not consistently patient-equal median or P20 lift; the three estimands are quoted together and the selection is in-sample (D9, D1e, F4; H-S1-02).
- **S10** IFN-alpha PC8 is the selected driver in P02 and P07 while the FA40 probe is all-zero there at the selected L1 C; read with S17, this is a Stage 2 threshold effect, not missing FA structure (D8; H-S2-03, H-S2-08).
- **S11** Donor downsampling can push sparse probes to an all-zero solution at fixed C, which returns AP to π by construction (D10; H-S2-04).
- **S12** Among the genuinely supervised factors, FA7 stays relatively stable when only a quarter of the donor-healthy training cells are kept, while PC8 is comparably stable only when half are kept (D11; H-S2-04; corrected in devlog 2026-08-03_narrative-guide-donor-correction).
- **S13** The old34 GSEA union is HVG-enriched relative to a hypergeometric null, yet most of it lies outside the HVG3k anchor, so the union and the HVG anchors are not interchangeable Stage 0 panels (D12; H-S0-02).
- **S14** Low-support patients have within-patient DE: P01 and P06 have DE genes but weak existing held-out lift, while P07's EMT panel is both DE-enriched and its best existing transfer (D13; H-S2-02).
- **S15** On the transductive basis, supported patients' factor-score supports overlap on nearly all biological programs, so aggregating them on one Stage 1 basis is reasonable and no biological program shows a patient-private subtype factor (D14; H-S1-04).
- **S16** On a frozen transductive basis with C fixed, held-out AUPRC rises with Stage 2 training-cell depth at a fixed patient mix wherever the probe is not collapsed (D15; H-S2-06).
- **S17** The IFN-alpha PCA40 versus FA40 patient-coverage gap is an L1-at-fixed-C effect on related directions (PC8, FA7) in one shared subspace: FA7 alone carries held-out lift in the patients the FA probe drops, L2 on the same FA scores recovers them, and under per-fold tuned C FA reaches every supported patient (F3, F6, F9, D19; H-S2-08).
- **S18** For IFN-alpha PCA and FA, held-out lift above π survives an inductive Stage 1 under per-fold tuned C and L2 in every supported patient, with small, sign-consistent transductive optimism; one panel only (D17; H-S1-06, H-S1-03; W2.6).

### Not supported as publication claims

- **N01** High lift is caused by patient-specific biology rather than support or prevalence (H-S2-02; ALT-PREV).
- **N02** FA is intrinsically inferior to PCA for IFN-alpha (H-S1-01, H-S2-08).
- **N03** The model is invariant to donor-versus-recipient origin (H-S2-04; ALT-DONOR).
- **N04** DR universally adds value over direct genes (H-S1-02).
- **N05** Selected panels beat matched-size random gene sets (H-S0-01; D21).
- **N06** Reported best configurations are unbiased deployment estimates (H-S1-02; D23; ALT-SEL).
- **N07** LOPO equals inductive deployment (H-S1-03, H-S1-06; ALT-TRANS).
- **N08** D15 shows that more patients help (H-S2-06; D15b).
- **N09** Cell-level lift is recurrence-risk improvement (H-A-01).
- **N10** Rank-defined TP and FN groups are clinical misclassifications (H-S2-05).
- **N11** The P03 broad-bag axis is a subtype (H-S1-07; D25).
- **N12** Factor-score overlap is predictive sharedness (H-S1-04; D3, W1.6).
- **N13** "FA misses patients" is a property of FA (H-S2-08).
- **N14** The highest-prevalence patients disproportionately shape the Stage 1 loadings (H-S1-05; D16, D17, W2.4).
- **N15** PC8 or FA7 is a unique, individually identifiable IFN-alpha sub-program (H-I-03; D18).

Standard closing line: "Descriptive results on a transductive Stage 1 basis over seven supported patients; they do not establish inductive deployment, causal biology or recurrence risk."

## §7 Corrections and retired hypotheses

Each item is dated and cites the devlog entry that records it.

- **2026-08-03, donor-factor stability.** An earlier summary over-stated the stability of both IFN-alpha driver factors under donor downsampling; only the genuinely supervised factors count, and FA7 and PC8 differ (devlog 2026-08-03_narrative-guide-donor-correction).
- **2026-09-17, the collinear-L1 account (teammate Ankit) is contradicted.** sklearn FA scores are essentially uncorrelated; L1 is not choosing between collinear FA columns, and the FA failure is signal splitting plus L1 collapse (devlog 2026-09-17_team-feedback-f1-f7).
- **2026-10-02, P04 joining FA at K40 is not a new direction.** The matched FA7 axis already exists at smaller K, and the selected L1 probe only starts keeping it at K40 (devlog 2026-10-02_f9-ifna-pc8-loadings).
- **2026-10-08, "more malignant cells" means highest prevalence.** P02, P03 and P04 are the highest-prevalence patients, not the patients with the most malignant cells, and they are a small share of the cells in the label-blind Stage 1 fit (devlog 2026-10-08_team-report-devlog-pivot).
- **2026-10-08, team hypothesis (a) is retired (H-S2-07).** Stage 1 is label-blind, FA7 carries the missing patients, and L2 recovers them (devlogs 2026-10-08_team-report-devlog-pivot and 2026-10-08_research-spine-adopted).
- **2026-10-08, "FA misses patients" is not a property of FA.** Other IFN-alpha configurations reach every supported patient, and the selected penalties differ across atlas cells (devlog 2026-10-08_team-report-devlog-pivot).
- **2026-10-08, premise of the large-patient rerun option (A-20261008-17).** P02, P03 and P04 are among the smallest patients by cell count; they are the highest-prevalence patients (devlog 2026-10-08_team-report-devlog-pivot).
- **2026-10-08, direction of the equal-weighting option (A-20261008-14).** Equal per-patient weighting in Stage 1 would raise, not lower, the Stage 1 share of the highest-prevalence patients; the targeted imbalance must be stated first (W2.3; devlog 2026-10-08_team-report-devlog-pivot).
- **2026-10-08, D15 is a depth curve, not a patient curve.** Every draw keeps all training patients and C is fixed; the patient-count curve is D15b (devlogs 2026-10-02_d15-cell-count-learning-curve and 2026-10-08_team-report-devlog-pivot).
- **2026-10-08, knife-edge revision of narrative guide §4.** At a slightly larger C the FA40 L1 probe keeps FA7: the P02 gap is a regularization threshold effect, and only P07 may keep a real PCA advantage (devlog 2026-10-08_literature-reviews; W3.2).
- **2026-10-08, PC8 and FA7 are related, not identical, axes.** Earlier text called them one shared axis; their score correlation is moderate, PC8 is split across FA7 and FA8, and neither is identifiable (devlogs 2026-10-08_d18-factor-cards and 2026-10-08_d19-regularization-contract).
- **2026-10-08, leverage of the highest-prevalence patients.** The first D17 reading said the contrast patients have more leverage than P02, P03 and P04 both absolutely and relative to random drops; across method and K this holds for P01 only, and the random drops do not match malignant composition (devlog 2026-10-08_d17-basis-variant-pilot).
- **2026-10-08, the support rule.** The supported-patient rule is the `study_config.json` minimum malignant-cell count with both classes; D14's larger malignant stratum is an analysis parameter, not the support rule (devlog 2026-10-08_research-spine-adopted).

## §8 Crosswalk and glossary

**Questions.**

| User question | PI question | Lens | Main hypotheses |
|---|---|---|---|
| Q1 shared vs patient-specific | PI-Q1 (robust shared model) and PI-Q2 (patient-specific biology) | QI names the axes | H-S2-01, H-S2-02, H-S1-04, H-S1-05, H-S1-06, H-S1-07 |
| Q2 strength of evidence | none (new in the 2026-10-08 report) | none | H-I-01, H-S2-04, H-S2-06, H-S0-01 |
| Q3 what DR adds | none (new in the 2026-10-08 report) | QI for the factors that win | H-S1-01, H-S1-02, H-S0-02, H-S2-08 |
| QI interpretability | PI-Q3 (programs break into sub-axes) | itself | H-S2-03, H-S2-08, H-I-03, H-I-02 |

The PI questions are A-20260604-01, A-20260604-02 and A-20260604-03; the user's questions are A-20261008-30, A-20261008-31 and A-20261008-32.

**2026-07-24 meeting topics** (mapped by topic; the meeting numbering is settled by W0.7).

| Ask | Topic | Analyses |
|---|---|---|
| A-20260724-01 | apples-to-apples lift metric | D6 |
| A-20260724-02 | no-DR baseline | D9 |
| A-20260724-03 | IFN-alpha PCA40 vs FA40 factor analysis | D8 |
| A-20260724-04 | per-patient lift across panels and DR | D7 |
| A-20260724-05 | donor downsampling | D10, D11 |
| A-20260724-06 | per-patient malignant and healthy counts | D6 (panel D6A) |

A second six-item question set from 2026-08-03 produced the narrative guide; it is not yet in `asks/` (W0.7).

**Diagnostic-plan layers** (from `plans/stage2_comprehensive_diagnostic_plan.md`).

| Layer | Question | IDs |
|---|---|---|
| Layer 1 | Stage 1 method and K stability within a panel | D1, D2 (V2F2 historical) |
| Layer 2 | patient transfer topology | D1c, D2 (V2F4 historical) |
| Layer 3 | discovery, LOPO and patient-specific taxonomy | D3, D3b |
| Layer 4 | regularization and sparsity pathology | D4, D19 |
| Layer 5 | factor-to-gene grounding | D5 (sample), D18 |
| Layer 6 | matched control calibration | D1 control rows, D21 |
| Layer 7 | theme synthesis | not done (Set B scope, W4.7) |

**Figure and ID families.**
- **V2 vs F.** V2A to V2F6 are the June sharedness-suite figure IDs (registry row V2). V2F1 to V2F5 are superseded as claim plots, the V2B1 raw AUPRC heatmap and the V2B2 enrichment ratio are retired, and V2F6 theme synthesis was never done. F1 to F9 are the 2026-09-17 feedback deep dives. V2F1 is not F1.
- **D1a and D1b vs D1DD.** D1a and D1b are colour panels of the D1 atlas figure. The D1DD series (D1DD1 to D1DD13) is the separate 2026-09-10 deep-dive notebook that unpacks the D1a panel.
- **F8** was never assigned (registry row retired). Section 8 of the feedback notebook is inductive-scoping text.
- **D1DD and F are closed series.** New analyses use D numbers (planned rows in `analyses.csv`, then `research_index.py reserve-id`).

**Sets and campaigns** (`campaigns.csv`).
- Set A: root, shortlist_20260526_lopo_gpu8, old34_direct_gene_all_v1, old34_controls_fullgrid_v1, old34_controls_direct_gene_v1.
- Set B: expanded_stage0_mrd_manuscript_axes_v1.
- Set C: pooled_stage0_unions_v1 (stalled; W0.6).
- Diagnostics: donor_sensitivity_v1, donor_sensitivity_v2.

**Pipeline Stage vs clinical stage.** Pipeline Stage 0, Stage 1 and Stage 2 (panels, DR, probes) are unrelated to the CRC clinical "stage 0-2" of the grant text in §10.

**Repository audit.** Audit question IDs A1 to G3 are the question_id column of `science_to_repo_matrix.tsv` in the 2026-08-25 repository audit. The audit is tracked by the outer umbrella repo at `reports/mrd_stage0_2_repository_audit/` and is not copied into this repo.

**Literature-review label remap.** The single-cell ML review used provisional labels:
- its "D16 genotype audit" is D25;
- its "D15b" (cell depth with C retuned) is D15 v2 (W6.2);
- its "D15c" (patient-drop curve) is D15b;
- its "D15d" (two-dimensional design) is W6.3.

**ID families.**
- H-S0, H-S1, H-S2, H-I, H-A: hypotheses (`hypotheses.md`).
- ALT-*: reviewer alternatives (`alternatives.csv`).
- V2, D, D1DD, F: analyses (`analyses.csv`).
- W# and W#.n: workstreams and backlog tasks (`../workstreams/`).
- A-YYYYMMDD-NN: asks (`asks/`).
- Devlog ids: the devlog filename stems. EVT- ids in older notes are chronology events, now devlog entries.
- S and N: claim labels in §6.

## §9 Figure plan

| Panel | Question | Hypotheses | Source IDs | Status |
|---|---|---|---|---|
| Fig1 design and metric | What is the probe, and why is lift above π the effect size? | H-I-01 | D6 | draft_exists |
| Fig2 atlas | Which programs transfer to which supported patients? | H-S0-01, H-S1-01, H-S2-01 | D1, D1f, D1DD2 | needs_rework |
| Fig3 aggregation | Can patients share one basis, and do high-prevalence patients dominate it? | H-S1-04, H-S1-05, H-S1-06, H-S1-07 | D14, D16, D20, D17 | needs_rework |
| Fig4 DR vs direct | What does DR add once selection is nested and baselines are fair? | H-S1-02, H-S0-02 | D9, D1e, F4, D23, D24 | needs_rework |
| Fig5 IFN-alpha | Is the PCA/FA gap structure or regularization, and what does the shared axis encode? | H-S2-03, H-S2-08, H-I-03 | D8, F3, F6, F9, D18, D19 | needs_rework |
| Fig6 strength | How strong is the evidence once nulls, uncertainty and C retuning are applied? | H-S2-04, H-S2-06, H-S0-01 | D21, D22, D10, D11, D15, D15b | planned |
| Supplementary | Prior vs HVG, regularization pathology, error audits, low-support DE, K trends | H-S0-02, H-S2-05, H-S2-02, H-S1-01 | D12, D4, D1DD10, D1DD11, D1DD12, D13, F1 | draft_exists |

## §10 Grant context (CRC Aim 2; not this paper's spine)

The CRC Aim 2 text below is the grant's parent hypothesis, copied verbatim from the frozen canvas (`legacy/stage0-2-hypothesis-tracker_2026-10-02b.canvas.tsx`). The pipeline Stages 0-2 are mechanistic evidence that may later feed Aim 2; they are not a substitute for it. Linked hypotheses: H-I-02, H-A-01, H-A-02.

> **WORKING_HYPOTHESIS:** The current classification of stage 0–2 colorectal cancer (CRC) relies on histopathological assessment, which fails to identify patients at high risk of recurrence. We hypothesize that integrating molecular features (genomic and transcriptomic) with standard pathology using a machine-learning framework will improve the prediction of minimal residual disease (MRD) and recurrence risk, enabling more personalized treatment strategies.
>
> **AIM2 2.1 Data integration** (status: partial)
> - text: Consolidate clinicopathological, genomic (e.g. ctDNA, SNVs), and transcriptomic (gene-expression) data from the stage 0–2 cohort.
> - pipelineMap: Stage 0 today is transcriptomic knowledge-prior panels only. No clinicopath, ctDNA, or SNV matrix is in the 20260525 experiment.
>
> **AIM2 2.2 Model training** (status: partial)
> - text: Use ensemble or deep-learning architectures to identify key biomarkers associated with recurrence.
> - pipelineMap: Implemented as Stage 1 DR (PCA/FA/FactoSig) plus Stage 2 L1/L2/elastic-net logistic paths — a biological probe, not an ensemble/DL recurrence model.
>
> **AIM2 2.3 Evaluation** (status: partial)
> - text: Assess AUC-ROC and F1 against traditional staging methods.
> - pipelineMap: Current estimands are cell-level malignant-vs-normal AP / lift vs prevalence, not patient-level recurrence vs TNM. AUROC is specified but not the suite default.
>
> **AIM2 2.4 Interpretability** (status: partial)
> - text: Apply SHAP or LIME to name influential features and disease-progression biology.
> - pipelineMap: Coefficient paths, factor ablation, and Stage 1 loadings→genes are in place (D4–D5, D8). SHAP/LIME are not implemented.

## §11 Open design decisions

Sources: the outer repository audit's `open_decisions.md` (items 1-9 there) plus decisions opened on 2026-10-08. Each is resolved by a devlog entry with type: decision.

1. **Transductive vs inductive estimand** (audit decision on the primary estimand): open. The IFN-alpha pilot found small transductive optimism (D17); resolved through W2.6.
2. **Set B scope** (audit decision on Set B): open. Resolved through W4.7.
3. **Group sparsity** (audit decision on group-level sparsity): open.
4. **preSCT and relapse** (audit decision on other timepoints): open. Tracked by H-A-02.
5. **Pooled vs patient-equal headline** (audit decision on operating points): resolved, Option B (2026-07-03; devlog 2026-07-03_figure-catalog-option-b).
6. **Support rule** (audit decision on supported patients): resolved as the `study_config.json` rule (minimum malignant-cell count and both classes). Revisit for the new cohort in W8.2.
7. **Meaning of donor downsampling** (audit decision): open. Resolved through W4.6 with condition-specific C.
8. **Operational "program"** (audit decision): open. Resolved through W5 (factor cards and the frameworks note).
9. **The May shortlist cell-prediction matrices** (audit decision; campaign shortlist_20260526_lopo_gpu8): open.
10. **Target of the equal-weighting option (A-20261008-14)**, new 2026-10-08: open. Decide which imbalance is targeted before refitting (W2.3).
11. **Pooled Set C** (pooled_stage0_unions_v1), new 2026-10-08: open. Relaunch the unfinished shards and merge, or abandon (W0.6).

## §12 Related work

The two literature reviews of 2026-10-08 are homed in W9 (devlog 2026-10-08_literature-reviews):
- [Factor interpretability review](../workstreams/W9_literature/2026-10-08_factor_interpretability_review.md)
- [Single-cell ML transfer and baselines review](../workstreams/W9_literature/2026-10-08_scml_transfer_baselines_review.md)

Number-free takeaways, with the tasks they inform:

**From the factor interpretability review:**
- The IFN-alpha PCA-over-FA L1 gap at the stored C is a threshold knife-edge, so it is not a method difference (W3.1, W3.2; H-S2-08).
- L1 selection is basis-dependent while L2 on whitened scores is rotation-invariant: compare methods under L2 or elastic net, and use L1 only for parsimony (W3.2).
- factosig_promax loadings are a pattern matrix: report pattern and structure genes and the factor correlations (W5.1, W5.4).
- Report effective_k and factor-score determinacy; poorly determined factors should not be named (W5.1, W4.5).
- Replace one-to-one Hungarian matching with many-to-one matching, principal angles and Tucker congruence (W5.1, W2.1).
- Compare probes in gene space through activation patterns (W5.2), and test patient-specific sub-axes with a within-span per-patient direction test (W5.5).
- Measure Stage 1 recurrence across seeds and resamples (W4.5), use stability selection for correlated promax factors (W3.6), and try target rotation or prior-guided factor models (W5.6).

**From the single-cell ML review:**
- The cohort's own paper derives malignant labels from different assays per patient, so a patient-specific axis needs a label-provenance and genotype audit (W1.4, D25; ALT-LABEL, ALT-DONOR).
- Ablate sex-chromosome and HLA genes from the broad bags before reading their private axes (W1.8).
- Retune C and the penalty by inner LOPO, add an AUROC companion (W3.1, W4.1), and nest the method and K selection (W3.3).
- Run the narrow inductive Stage 1 first (W2.1).
- Use exact single-score comparators (W3.4) and matched random gene-set nulls (W4.2).
- Learning curves: cell depth with C retuned (W6.2), a patient-drop curve (W6.1) and a two-dimensional design (W6.3).
- MOFA+ or cNMF as a sharedness comparator (W1.3); the semi-supervised ladder later (W3.7); a pre-registered external-validation protocol (W8.2); old34 gene-set provenance (W0.12).
