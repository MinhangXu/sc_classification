# Paper outline (living; number-free)

> Maintained under W0.13. **THEME.md §6 is the claim boundary**; the S and N labels below refer to it. Numbers live in the cited devlog entries and in CLAIMS.md, never here. Status as of 2026-10-10 (THEME milestone review: S19-S21, N16-N17 added; H-S1-05 not supported). Update rule: when your devlog entry changes a claim, add its id to the "Feeds" line of the matching subsection and adjust its status (`paper/README.md`).

## 0. How each Results subsection is written

| Field | Content |
|---|---|
| Question | The study question (Q1 shared vs patient-specific, Q2 strength of evidence, Q3 what DR adds, QI interpretability lens) and the PI or user ask behind it |
| Claim now | The current descriptive claim, as THEME §6 words it (S labels), plus what it must not be read as (N labels). A result that exists in a devlog entry but has not yet been reviewed into THEME §6 (STATUS.md §9 lists those entries) is marked *pending THEME review*; it is not yet a paper claim |
| Hypotheses | H-IDs. Their current verdicts are in STATUS.md §3, not here |
| Analyses | D, F and D1DD IDs |
| Feeds | The devlog ids that hold the evidence. Planned ids are marked (planned) |
| Report | Where the questions report (`reports/2026-10-08_questions_report/`; v2 since 2026-10-09, same section numbers as v1) treats it, for the lead's cross-reading |
| Figure | The panel IDs in `FIGURES.md` |
| Status | `evidence in hand`, `partial`, `running`, `needs new data`, `planned` or `not started` (`paper/README.md`) |
| Still needed | W tasks that would change or complete the claim |

## 1. Working titles (the user chooses; record the choice in a docs entry)

1. *What transfers across patients in post-transplant MDS measurable residual disease? Knowledge-prior probes at single-cell resolution.*
2. *Classification as a biological probe: shared and patient-specific malignant programs in post-transplant MDS MRD single-cell data* (the THEME §0 framing).
3. *Knowledge priors, dimensionality reduction and regularization in single-cell MRD classification: what transfers, how strongly, and what DR adds* (methods-forward).
4. **Candidate companion paper or section** (the user's catch line, A-20261009-12): *"Machine learning in MRD single-cell data: just downsample the healthy cells?"* It stays a question until W8.7 reports (R12; `workstreams/W8_new_cohort/healthy_downsampling_plan.md` §10).

## 2. Abstract skeleton (slots only)

- **Background.** In post-transplant MDS, measurable residual disease is a small number of recipient-derived malignant cells among many donor-derived cells. Single-cell CITE-seq with genotype- or CNV-based malignant calls lets us ask whether malignant cells from different patients share a transferable phenotype, and at what biological resolution. [cohort reference; W9 Review B]
- **Approach.**
  - Knowledge-prior gene panels (Stage 0), decomposed by PCA, FA or FactoSig (Stage 1), probed by regularized logistic regression (Stage 2).
  - Transfer to a held-out patient is measured as lift over the constant-score no-skill AP = π.
  - Patient-equal summaries (median, P20) are reported beside pooled AP.
  - Direct-gene probes on the same panel are the no-DR control. [R1]
- **Results** (one slot per arc, each at THEME §6 level):
  - [R2: transfer is common but strong transfer concentrates in the highest-prevalence patients; S01, S07];
  - [R3: patients can share one basis; S15];
  - [R5: transfer survives a basis that never saw the held-out patient, on one panel; S18];
  - [R6: many atlas zeros are L1 thresholds, not absent signal; S11, S17];
  - [R7: what DR adds depends on the estimand; S09, S03];
  - [R8: the IFN-alpha axis, named from both poles; S10, S17];
  - [R10: how strong the evidence is; S16 and the D15b curve];
  - [R11: external patients; needs new data];
  - [R12: healthy-cell downsampling; needs new data].
- **Conclusion.** The descriptive claim boundary, with the THEME §6 standard closing line.

## 3. Introduction beats

1. **Clinical problem.** Relapse after allogeneic transplant in MDS, and why MRD detection at single-cell resolution matters. Malignant labels come from different assays per patient in the source cohort (ALT-LABEL; W1.4).
2. **The ML problem.** Extreme class imbalance, few patients and strong patient heterogeneity. Held-out-patient transfer is the honest test. Under imbalance, AP has a prevalence-dependent no-skill level, hence lift (H-I-01).
3. **Knowledge priors as probes.** Restricting the gene space to curated programs makes a probe interpretable. It must be compared with broad gene bags, with direct-gene probes and, eventually, with matched random gene sets (H-S0-01, H-S0-02; W4.2).
4. **The questions.** Q1 shared vs patient-specific, Q2 strength of evidence, Q3 what DR adds, and the QI lens; mapped to the PI's three questions (A-20260604-01, -02, -03) and the user's (A-20261008-30, -31, -32).
5. **Four traps, and the control each one gets.** These are part of the contribution:
   - a transductive basis versus inductive deployment: R5;
   - fixed-C L1 zeros: R6;
   - in-sample selection of the best representation: R7, W3.3;
   - prevalence confounding of lift: R2, W1.5.
6. **Contributions list.** One line per arc, written last.

## 4. Results

### R1. Design and the yardstick
- **Question:** meta, and Q2. Is lift above π a fair, comparable effect size across patients with very different prevalence?
- **Claim now:** S06 (constant-score no-skill AP = π exactly under the implemented metric); Option B summaries.
- **Hypotheses:** H-I-01.
- **Analyses:** D6, D1d.
- **Feeds:** 2026-08-31_d6-d9-meeting-followup; 2026-07-03_figure-catalog-option-b; 2026-07-24_d1-d5-diagnostic-suites.
- **Report:** v1 section 01 (cohort and yardstick).
- **Figure:** Fig1.
- **Status:** evidence in hand.
- **Still needed:** the AUROC companion (W4.1, D22).

### R2. Which programs transfer to which patients (the atlas)
- **Question:** Q1 (PI-Q1, A-20260604-01).
- **Claim now:** S01, S02, S07. Transfer above π reaches every supported patient in many representations, while strong lift concentrates in P02, P03 and P04, the highest-prevalence patients. Not N01.
- **Hypotheses:** H-S0-01, H-S1-01, H-S2-01. Supplementary: H-S0-03 (Set B), H-S0-04 (pooled Set C).
- **Analyses:** D1, D1c, D1f, D2, D7, D1DD1-D1DD4.
- **Feeds:** 2026-05-25_set-a-old34-broad-screen; 2026-06-05_set-a-all-biological-fullgrid; 2026-07-24_d1-d5-diagnostic-suites; 2026-08-31_d6-d9-meeting-followup; 2026-09-10_d1a-descriptive-deep-dives; 2026-09-11_d1-atlas-783-d1e-d1f; 2026-10-08_d19-regularization-contract (all-zero folds); for the supplementary hypotheses, 2026-06-22_set-b-expanded-manuscript-axes and 2026-09-03_pooled-unions-set-c.
- **Report:** v1 section 08 (Q1).
- **Figure:** Fig2.
- **Status:** partial. It is descriptive, and prevalence is not yet separated from shared transfer.
- **Still needed:**
  - prevalence- and support-matched lift (W1.5);
  - patient-level intervals (W4.1, D22);
  - matched random gene-set nulls (W4.2, D21);
  - tuned-C re-scoring of knife-edge cells (W3.8);
  - Set B scope (W4.7) and the Set C decision (W0.6).

### R3. One shared basis? The aggregation test
- **Question:** Q1 and Q2 (A-20261008-07, A-20261008-10; reading notes A-20261009-04, A-20261009-16).
- **Claim now:** S15. Supported patients' factor-score supports overlap on nearly all biological programs. The only patient-private axis is P03's, in broad control bags (H-S1-07, open; not N11). Not N12. S20: P01's over-representation in score energy is mostly its predicted cell-type mix (D28).
- **Hypotheses:** H-S1-04, H-S1-07.
- **Analyses:** D14, D28, D20 (planned), D25 (planned).
- **Feeds:** 2026-10-02_d14-patient-factor-overlap; 2026-10-09_d28-celltype-score-shares.
- **Report:** v2 section 03 (a worked example of the overlap test on one factor, normal and malignant cells separately, and the three call types).
- **Figure:** Fig3 (panels A-C).
- **Status:** partial. The gates are uncalibrated and the permutation null sits at its floor.
- **Still needed:** D20 hardening (W1.2); the D25 genotype and label audit (W1.4); the test on W2 bases (W1.3); sex-chromosome and HLA ablation of the broad bags (W1.8).

### R4. Do the highest-prevalence patients shape the basis?
- **Question:** Q1 and Q2 (A-20261008-02, A-20261008-05; reading notes A-20261009-02, -03, -13, -14, -15).
- **Claim now:** S19 and S20 (THEME §6, reviewed 2026-10-10); not N14, because H-S1-05 is not supported for PCA/FA on the three panels tested (D26). The evidence, descriptive and transductive:
  - on IFN-alpha, score energy follows cells, not labels (D16);
  - leaving the highest-prevalence patients out barely moves the discriminative axis once controls match malignant composition (D17);
  - the low-prevalence, high-cell-count patient P01 has the most leverage (D17);
  - P01's excess score energy is mostly matched by its cell-type mix (arithmetic on predicted labels, not a cause), while the IFN-alpha excess of the highest-prevalence patients is not (D28);
  - on allograft rejection and apoptosis, leaving the highest-prevalence patients out moves the discriminative axis beyond composition-matched drops, mostly through P03, but never more than leaving out the contrast patients does, so H-S1-05 is not supported for PCA/FA on the three panels tested (D26).
- **Hypotheses:** H-S1-05.
- **Analyses:** D16, D17, D26, D28.
- **Feeds:** 2026-10-08_d16-factor-patient-matrix; 2026-10-08_d17-basis-variant-pilot; 2026-10-08_d17-composition-matched-controls; 2026-10-09_d26-leverage-allograft-apoptosis (run entry); 2026-10-10_d26-leverage-allograft-apoptosis-done; 2026-10-09_d28-celltype-score-shares; 2026-10-10_decisions-new-run-inclusion-reweighting-lfs (the re-weighting target that follows from it).
- **Report:** v2 section 02 (redrawn figures, the cell-type-adjusted figure from D28, the three-panel leverage figure from D26).
- **Figure:** Fig3 (panels D-F).
- **Status:** partial.
- **Still needed:** W2.8 (the leverage test on the other biological panels), W2.2 (FactoSig and FactoSig-promax refits), W2.9 (a cell-type-matched leave-out control for P01) and W1.10 (the excess that cell type does not explain). The re-weighting target is decided (W2.3 closed: the Stage 2 labels); its runs are R12's (W8.7).

### R5. Transductive versus inductive Stage 1
- **Question:** Q1 and Q3 (A-20261008-16; reading note A-20261009-09).
- **Claim now:** S18, one panel. LOPO on the stored bases is label transfer, not deployment (N07).
- **Hypotheses:** H-S1-06, H-S1-03.
- **Analyses:** D17.
- **Feeds:** 2026-10-08_d17-basis-variant-pilot.
- **Report:** v1 section 06.
- **Figure:** Fig4.
- **Status:** evidence in hand (IFN-alpha only).
- **Still needed:** inductive bases for the D1-selected representations (W2.6); persisted projection parameters (W2.2); the PCA score-convention audit (W2.7).

### R6. Zeros are thresholds: the regularization contract
- **Question:** Q2, Q3 and QI (A-20261008-03, A-20261008-04; reading notes A-20261009-05, A-20261009-10).
- **Claim now:** S11, S12 and S17. Under balanced L1 at fixed C, a factor enters only above an n-dependent entry threshold. The IFN-alpha PCA40-vs-FA40 coverage gap is that threshold (not N02, not N13). The fixed-C donor-downsampling losses are all-zero collapses. The team's Stage 1 allocation hypothesis is retired.
- **Hypotheses:** H-S2-08, H-S2-04, H-S2-07 (retired; cited as a correction).
- **Analyses:** D19, D10, D11, F3, F6, D4.
- **Feeds:** 2026-10-08_d19-regularization-contract; 2026-09-17_team-feedback-f1-f7; 2026-08-21_donor-sensitivity-d10-d11; 2026-08-03_narrative-guide-donor-correction; 2026-10-08_literature-reviews; 2026-10-08_team-report-devlog-pivot; 2026-07-24_d1-d5-diagnostic-suites.
- **Report:** v1 section 04; v2 section 07 (why C is retuned).
- **Figure:** Fig5.
- **Status:** evidence in hand.
- **Still needed:** tuned-C re-scoring of the knife-edge atlas cells (W3.8); the donor rescue with condition-specific C (W4.6, which overlaps R12).

### R7. What does dimensionality reduction add over direct genes?
- **Question:** Q3 (A-20261008-32).
- **Claim now:** S03, S04, S08, S09. The answer depends on the estimand and on how the winner is chosen. Large gains are confined to a few programs, and broad bags favour direct genes. Not N04, not N06.
- **Hypotheses:** H-S1-02, H-S0-02.
- **Analyses:** D9, D1e, F4, F2, D12, D1DD5, D1DD6, D1DD9, D23 (planned), D24 (planned).
- **Feeds:** 2026-08-31_d6-d9-meeting-followup; 2026-09-11_d1-atlas-783-d1e-d1f; 2026-09-17_team-feedback-f1-f7; 2026-09-03_d12-d13-gsea-hvg-lowsupport; 2026-09-10_d1a-descriptive-deep-dives.
- **Report:** v1 section 08 (Q3).
- **Figure:** Fig6.
- **Status:** partial. The selection is in-sample, and the comparison is asymmetric while Stage 1 is transductive.
- **Still needed:** nested selection (W3.3, D23); exact single-score baselines (W3.4, D24); inductive bases (W2.6).

### R8. Interpretable axes: what IFN-alpha PC8 and FA7 encode
- **Question:** QI (PI-Q3, A-20260604-03; A-20261008-06, A-20261008-21; reading notes A-20261009-06, -07, -08).
- **Claim now:** S10 and S17. One shared IFN-alpha subspace carries related directions, PC8 for PCA and FA7 for FA. They are named from both poles with the sign orientation and the penalty stated; neither is individually identifiable (N15). S21 (D27): in normal cells most factor × surface-protein correlates are cell-identity axes; in malignant cells pooled correlates are mostly between-patient; all of them are confounded by patient, batch and origin. Not N16: the protein data are not integrated into the factors (A-20261010-01).
- **Hypotheses:** H-S2-03, H-I-03, H-S2-08. H-I-02 (the SHAP/LIME grant gap) goes to the Discussion.
- **Analyses:** D8, D5, F3, F6, F7, F9, D18, D27.
- **Feeds:** 2026-08-31_d6-d9-meeting-followup; 2026-07-24_d1-d5-diagnostic-suites; 2026-09-17_team-feedback-f1-f7; 2026-10-02_f9-ifna-pc8-loadings; 2026-10-08_d18-factor-cards; 2026-10-08_w7-atlas-strip-factor-explorer (the factor explorer); 2026-10-09_d27-factor-modality-assoc; 2026-10-09_protein-association-for-pavan and 2026-10-10_protein-association-for-pavan-v2 (D27 restated for a collaborator: CD34 between versus within cell type, malignant versus healthy cells of the same type, the isotype noise reference).
- **Report:** v2 section 05 (identifiability figure, card accounting table, D27 figures).
- **Figure:** Fig7.
- **Status:** partial. The IFN-alpha example and the modality associations for all cards (D27: normal-cell correlates are mostly cell-identity axes; pooled malignant correlates are mostly between-patient) are in hand; the census of other programs is planned.
- **Still needed:**
  - the oriented gene-direction table and activation patterns (W5.2);
  - held-out lift inside predicted cell types for the selected drivers (W5.7), and the F7b label fix that closes W5.3;
  - the frameworks note (W5.4);
  - the sub-axis census (W5.5);
  - target rotation and a Spectra/PLIER comparator (W5.6);
  - seed and resample stability (W4.5).

### R9. Patient-specific signal and error structure
- **Question:** Q1 (PI-Q2, A-20260604-02).
- **Claim now:** S05 and S14. Patient-specific opportunity tags and within-patient DE exist; the patient-specific fits are in-sample; single-patient peaks are not identifiable. Not N01, not N10.
- **Hypotheses:** H-S2-02, H-S2-05.
- **Analyses:** D3, D3b, D13, D1DD10, D1DD11, D1DD12, D1DD13, F7, D27 (the cell-type confound of the error audits, ALT-CT, partly addressed).
- **Feeds:** 2026-07-24_d1-d5-diagnostic-suites; 2026-09-03_d12-d13-gsea-hvg-lowsupport; 2026-09-10_d1a-descriptive-deep-dives; 2026-09-11_d1-atlas-783-d1e-d1f; 2026-10-09_d27-factor-modality-assoc.
- **Report:** v1 section 08 (low-support patients).
- **Figure:** Supplementary S3 and S4.
- **Status:** partial.
- **Still needed:** within-patient CV probes against the shared model (W1.6); held-out lift inside predicted cell types, the open half of ALT-CT (W5.7).

### R10. How strong is the evidence, and does more data help?
- **Question:** Q2 and meta (A-20261008-24, -25, -26, -31; reading note A-20261009-10).
- **Claim now:**
  - S16, at a fixed patient mix and with C fixed;
  - the D15b patient-count curve with C retuned is positive from one supported training patient, with small gains and wide patient-level intervals (THEME §3);
  - not N08: no claim that sequencing more patients raises AUPRC.
- **Hypotheses:** H-S2-06, H-S0-01 (the null-model part).
- **Analyses:** D15, D15b, D21 (planned), D22 (planned).
- **Feeds:** 2026-10-02_d15-cell-count-learning-curve; 2026-10-08_d15b-patient-count-curve; 2026-10-08_d15b-full-design.
- **Report:** v1 section 07.
- **Figure:** Fig8.
- **Status:** partial.
- **Still needed:**
  - D15 v2 with C retuned (W6.2);
  - the two-dimensional design (W6.3);
  - an inductive patient-count curve (W6.6);
  - patient-level uncertainty (W4.1);
  - matched nulls (W4.2);
  - a within-patient permutation control (W4.3);
  - a variance decomposition with the THEME §1 estimands kept apart (W4.4);
  - the grant note (W6.4).

### R11. External validation on new patients
- **Question:** Q1 and Q2 (A-20261008-12; reading notes A-20261009-11 and A-20261009-14, and the 2026-10-10 decision A-20261010-07, for inclusion).
- **Claim now:** none yet. The pre-registered expectations are in `workstreams/W8_new_cohort/integration_plan.md` §10.
- **Hypotheses:** H-S2-01, H-S1-04, H-S1-06, H-S2-06, H-S2-08; informs H-S1-02.
- **Analyses:** to be reserved at W8.4.
- **Feeds:** none yet. The plan's history is in 2026-10-08_team-report-devlog-pivot, 2026-10-09_report-reading-notes, 2026-10-09_w8-healthy-downsampling-plan (the inclusion section) and 2026-10-10_decisions-new-run-inclusion-reweighting-lfs (the inclusion rule for new full runs; the current atlas keeps the support rule).
- **Report:** v1 section 09; v2 section 09 adds the inclusion decision and the healthy-cell downsampling proposal.
- **Figure:** Fig9 (planned).
- **Status:** needs new data.
- **Still needed:** the email (W8.1); the lock (W8.2), which writes the inclusion rule decided under W8.5 into the frozen configuration; readiness code (W8.3); ingestion and inventory (W8.4); external points on the patient-count curve (W6.5).

### R12. Candidate arc or companion paper: "just downsample the healthy cells?"
- **Question:** Q2 and Q3 (A-20261009-12; A-20261009-16 for the link to R3). Does the healthy-to-malignant ratio limit transfer separately from the malignant count, and if healthy-cell downsampling helps, why?
- **Claim now:**
  - none;
  - what is known is a negative constraint: at fixed C, donor downsampling collapses L1 probes and L2 barely moves (S11, R6);
  - balanced class weights and rank metrics leave only Stage 1 geometry, negative-class composition, the n-dependent penalty and noise as channels (`healthy_downsampling_plan.md` §2.3).
- **Hypotheses:** H-S2-09 (alternatives ALT-NREG, ALT-PEN, ALT-PREV, ALT-CT, ALT-TRANS, ALT-SEL). Neighbours: H-S2-04, H-S1-05, H-S2-06.
- **Analyses:** a D-ID is reserved at W8.7. Background: D10, D11, D14, D16, D17, D19, D28.
- **Feeds:** 2026-08-21_donor-sensitivity-d10-d11 (background); 2026-10-09_report-reading-notes (the decisions); 2026-10-09_w8-healthy-downsampling-plan (the pre-registered design, H-S2-09 open); 2026-10-09_d28-celltype-score-shares (cell-type energy levels for the stratified design); 2026-10-10_d26-leverage-allograft-apoptosis-done (why the target is the Stage 2 labels); 2026-10-10_decisions-new-run-inclusion-reweighting-lfs (the cohort rule, and the Stage 2 target: Stage 2 only becomes the primary placement and Stage 2 sample weights, scheme SW, join the design).
- **Report:** not in v1 (a new idea from the reading notes); v2 sections 02, 03 and 09 (the proposal for the new data).
- **Figure:** HDS panels A-E (planned).
- **Status:** needs new data. It runs with the new cohort, by the user's decision.
- **Still needed:** W8.6, W8.7, W8.8, W4.6 (rehearsal choice) and a W9.3 literature addendum (the case-control and subsampling literature). W8.5 is decided.
- **Two narratives fixed in advance:**
  - supported: "downsample before the decomposition, stratified, and retune C; it helps because of geometry or composition";
  - not supported: "with balanced weights and tuned C, healthy downsampling is a change of C".
  Both are publishable methods messages. Whether this is a main-paper section or a short companion paper is the user's choice (W8.8).

## 5. Methods (mapped to code)

| Subsection | Content | Code (repo-relative) |
|---|---|---|
| M1 Cohort, labels, support rule | MRD CITE-seq cells; `CN.label`; donor and recipient calls; the support rule; the patient inclusion for new runs | `scripts/mrd_stage0_2/research/study_config.json`; `scripts/mrd_stage0_2/notebooks/stage0_2/_build_deg_hvg_lowsupport_notebook.py` (cohort cache); `scripts/mrd_stage0_2/workstreams/W8_new_cohort/integration_plan.md` §4 |
| M2 Stage 0 panels | Curated panels, biology-group unions, broad controls; gene universe; provenance (W0.12) | `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py`; `scripts/knowledge_driven_embedding/` |
| M3 Stage 1 decomposition | PCA, FA, FactoSig, factosig_promax at several K, single seed, transductive; inductive and leave-out variants | `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py` (runs Stage 0 and Stage 1); `scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/stage1_refit.py`; the factosig repo |
| M4 Stage 2 probes and views | Regularized logistic paths (L1, L2, EN), balanced weights; discovery, sharedness LOPO and patient-specific views; direct-gene probes | `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py` |
| M5 Regularization contract | The L1 entry rule; inner-LOPO C tuning; C policies; what fixed C means when n changes | `scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py` |
| M6 Metrics and summaries | Lift over π, normalized lift, AUROC, pooled AP; Option B patient summaries; patient-level uncertainty | `scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/factor_patient_matrix.py` (`average_precision`); D22 runner (planned, W4.1) |
| M7 Aggregation (overlap) test | Per-patient factor-score overlap within label; gates; null | `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py`; D20 runner (planned, W1.2) |
| M8 Allocation and leverage | Score-energy shares (D16); leave-out refits with size- and composition-matched controls (D17, D26) | `scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py`; `scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py` |
| M9 Learning curves | Cell-depth (D15) and patient-count (D15b) curves with C retuned | `scripts/mrd_stage0_2/stage2_supervised/cell_count_learning_curve.py`; `scripts/mrd_stage0_2/workstreams/W6_learning_curves/patient_count_curve.py` |
| M10 Factor cards and interpretation | Sign-oriented loadings from both poles; matching across methods; determinacy; modality associations | `scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/factor_cards.py`; `scripts/mrd_stage0_2/apps/old34_factor_explorer.py` |
| M11 Donor and healthy-cell downsampling | D10/D11 (fixed C); the pre-registered factorial with C retuned (Arm D) | `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py`; `scripts/mrd_stage0_2/workstreams/W8_new_cohort/healthy_downsampling_plan.md` (runner planned, W8.6) |
| M12 External validation | Frozen probes projected onto new patients; inductive extended LOPO; pre-registration | `scripts/mrd_stage0_2/workstreams/W8_new_cohort/integration_plan.md` |
| M13 Reproducibility | Research spine (devlog, registry, claim gates), the experiment layout, the apps | `tools/research_index.py`; `AGENTS.md`; `scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py` |

## 6. Discussion points

1. **Resolution of sharing.** The evidence supports one shared basis, and a shared IFN-alpha axis that reaches low-prevalence patients. Strong transfer is still entangled with prevalence (R2, R3, R8; ALT-PREV).
2. **The highest-prevalence patients dominate labels, not geometry.** On IFN-alpha, allograft rejection and apoptosis (PCA/FA), the imbalance specific to them acts through Stage 2 labels and selection rather than through the Stage 1 loadings; the patient with the most Stage 1 leverage is P01 (S19; D26). Re-weighting therefore targets the Stage 2 labels (W2.3, decided 2026-10-10), and the new-data runs test it (R4, R12).
3. **Methodological lessons for single-cell MRD classification:**
   - zeros at fixed C are thresholds;
   - C must be retuned whenever n changes;
   - transductive optimism is small but has a consistent sign on one panel;
   - pooled AP, median lift and P20 lift rank methods differently (R5, R6, R7).
4. **Interpretability.** Name factors from both poles, with sign and penalty. Single factors are not identifiable programs. The grant's SHAP/LIME aim (H-I-02) is not what the current stack does (R8).
5. **Limitations:**
   - seven supported patients (ALT-NPAT);
   - label provenance differs by patient (ALT-LABEL);
   - donor versus recipient origin (ALT-DONOR);
   - patient nested in batch (ALT-BATCH);
   - in-sample selection until W3.3 (ALT-SEL);
   - a single Stage 1 seed (ALT-SEED);
   - cell-level lift is not recurrence risk (N09);
   - the CRC Aim 2 grant context is separate (H-A-01, H-A-02; THEME §10).
6. **What new patients will and will not settle** (R11; `integration_plan.md` §13).
7. **The downsampling question** as a general lesson for imbalanced single-cell ML (R12).

## 7. Experiments still needed (mapped to W tasks)

| Need | For | Task (analysis ID) | Status |
|---|---|---|---|
| Leverage test on allograft rejection and apoptosis | R4 | W2.4 (D26) | evidence in hand |
| Cell-type-stratified score shares | R4, R12 | W1.9 (D28) | evidence in hand |
| Factor × cell type and factor × surface protein for all cards | R8 | W5.3 (D27) | evidence in hand |
| D14 v2 with calibrated gates | R3 | W1.2 (D20) | planned |
| Genotype and label-provenance audit | R3 | W1.4 (D25) | planned |
| Aggregation and allocation on the W2 bases | R3, R4 | W1.3 | planned |
| Prevalence- and support-matched lift | R2 | W1.5 | planned |
| Within-patient CV probes against the shared model | R9 | W1.6 | planned |
| Inductive bases for the D1-selected representations | R5, R7 | W2.6 (after W2.2) | planned |
| Re-weighting target (decided 2026-10-10: the Stage 2 labels), then Stage 2 re-weighting in Arm D | R4, R12 | W2.3 (closed), W8.6, W8.7 | needs new data |
| Nested DR-vs-direct selection | R7 | W3.3 (D23) | planned |
| Exact single-score baselines | R7 | W3.4 (D24) | planned |
| Tuned-C re-scoring of knife-edge atlas cells | R2, R6 | W3.8 | planned |
| Patient-level uncertainty and the AUROC companion | R1, R2, R10 | W4.1 (D22) | planned |
| Matched random gene-set nulls | R2, R10 | W4.2 (D21) | planned |
| Within-patient label-permutation control | R10 | W4.3 | planned |
| Variance decomposition with separate estimands | R10 | W4.4 | planned |
| Stage 1 seed and resample stability | R8 | W4.5 | planned |
| Donor rescue with condition-specific C | R6, R12 | W4.6 | planned (rehearsal choice in `healthy_downsampling_plan.md` §12) |
| Oriented gene-direction table and activation patterns | R8 | W5.2 | planned |
| Sub-axis census beyond IFN-alpha | R8 | W5.5 | planned |
| Cell-depth curve with C retuned; the two-dimensional design; an inductive patient-count curve | R10 | W6.2, W6.3, W6.6 | planned |
| New patients: email, lock, readiness, ingestion (inclusion rule decided, W8.5) | R11 | W8.1, W8.2, W8.3, W8.4 | needs new data / user |
| Healthy-cell downsampling factorial | R12 | W8.6, W8.7 | needs new data |
| Clean-tree reruns for any publication-level claim | all | W0.10 | planned |

## 8. Hypothesis coverage check (W0.13 done-when)

Every hypothesis with a current verdict has a home:

| Hypothesis | Results subsection |
|---|---|
| H-S0-01 | R2, R10 |
| H-S0-02 | R7 |
| H-S0-03, H-S0-04 | R2 (supplementary) |
| H-S1-01 | R2, R6 |
| H-S1-02 | R7 |
| H-S1-03, H-S1-06 | R5 |
| H-S1-04 | R3 |
| H-S1-05 | R4 |
| H-S1-07 | R3 |
| H-S2-01 | R2, R11 |
| H-S2-02, H-S2-05 | R9 |
| H-S2-03 | R8 |
| H-S2-04 | R6, R12 |
| H-S2-06 | R10 |
| H-S2-07 | R6 (retired; correction) |
| H-S2-08 | R6, R8 |
| H-S2-09 | R12 |
| H-I-01 | R1 |
| H-I-02 | Discussion 4 |
| H-I-03 | R8 |
| H-A-01, H-A-02 | Discussion 5 (grant context, not this paper's spine) |
