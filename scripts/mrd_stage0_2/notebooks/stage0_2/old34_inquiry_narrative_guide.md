# Old34 inquiry narrative and reading guide

Updated: 2026-10-08  
Canonical experiment: `20260525_060508_stage0_mrd_old34_broad_screen_82db5093`

> **Background reading for method mechanics** (AP = π proof, Spearman semantics, direct-gene baseline, IFN-α attribution, donor correction). **Not a START HERE:** see repo-root [`AGENTS.md`](../../../../AGENTS.md), [`../../research/STATUS.md`](../../research/STATUS.md), and the claim list in [`../../research/THEME.md`](../../research/THEME.md) §6.

For campaign and run state (namespaced campaigns such as the pooled unions, the control full-grid and Set B, their status, and what must not be relaunched), see [`../../research/campaigns.csv`](../../research/campaigns.csv). This guide is the method-mechanics reading order, not the process inventory.

The guide covers:

1. historical design and exploratory work;
2. completed diagnostics, including D15 and the F1-F9 team-feedback deep dives;
3. meeting-driven follow-up analyses;
4. dated corrections to earlier statements (see [Corrections (2026-10-08)](#corrections-2026-10-08)).

Which claims are currently supportable is decided in `../../research/THEME.md` §6, not here (see [Current claim boundary](#current-claim-boundary)). Numbers in the sections added on 2026-10-08 cite the devlog entry for their analysis, as `(devlog <id>)`; entries live under `../../research/devlog/2026/<id>.md`. The entries hold the headline numbers; the F1-F9 reading guide also quotes detail values read directly from the EXP-relative tables it lists under each **Artifact** line. Artifact paths in those sections are relative to the experiment dir (EXP).

## Corrections (2026-10-08)

Dated corrections to statements in this guide and in the 2026-10-08 team report. Earlier text stays in place, with short pointer notes inside §4 and §5 where it is now stale.

### C1. §4 knife-edge revision: the IFN-α PCA40 vs FA40 gap is a Stage 2 L1-at-fixed-C effect

§4 explains the P02/P07 PCA advantage as "PCA retains a nonzero factor-008 probe; FA's selected sparse model retains no factor". The fold pattern is right. The mechanism is a Stage 2 regularization threshold, not a missing FA axis.

- **L1 entry threshold.** For balanced-weight L1 logistic regression (sklearn liblinear, `class_weight="balanced"`), the gradient of the weighted loss at β = 0 for a standardized feature j is (n_train/4)·Δ_j, where Δ_j is the training-fold malignant-minus-normal mean of z_j. The coefficient can leave zero only if |Δ_j| > 4/(C·n_train). A fixed C is therefore not a fixed per-cell penalty: the threshold rises as n_train falls.
- At the stored C = 10⁻⁴ the threshold is about 0.68–0.70 SD in most supported folds, and 0.83 (P01 held out) and 0.93 (P09 held out) in the two folds with the fewest training cells. PC8's gap is 0.70–0.77 SD and FA7's is about 0.66–0.70 SD. The rule reproduces all 14 stored zero/nonzero outcomes (7 supported folds × 2 methods), including PCA's all-zero P01/P09 folds (devlog 2026-10-08_team-report-devlog-pivot; devlog 2026-10-08_literature-reviews).
- FA7 carries the P02/P07 signal on its own: single-factor held-out lift is 18.06 pp in P02 and 6.97 pp in P07 (devlog 2026-10-02_f9-ifna-pc8-loadings). L2 on the same stored FA scores at the same C gives 16.14 pp (P02) and 5.07 pp (P07) in the F3d local refit; the D19 contract refit of the same estimand gives 15.97 and 5.07 pp (PCA 13.19 and 5.60 pp against F3d's 13.23 and 5.60). The two refits differ by up to 0.17 pp, presumably through solver and tolerance; D19 is the reference value from here on (devlog 2026-10-08_d19-regularization-contract). FA scores are uncorrelated (max off-diagonal |r| 0.003), so L1 is not choosing between collinear columns (devlog 2026-09-17_team-feedback-f1-f7).
- **Knife-edge.** At C = 1.05×10⁻⁴, 5% above the stored value, FA40-L1 keeps FA7 in P02 (18.06 pp, above PCA40's 15.94 pp) and in P07 (6.97 pp). PCA40-L1 is unchanged (P02 15.94 pp, P07 11.08 pp) (devlog 2026-10-08_literature-reviews). P02 is therefore a regularization knife-edge, not a PCA advantage. At that C, PCA keeps an advantage only in P07.
- What this did not settle when written: a C-matched comparison with C tuned per fold, and whether the transductive basis matters. Both have since run: with C tuned per fold, FA40 has positive lift in all 7 supported patients under L1, L2 and elastic net, and a residual PCA advantage remains only in P07 and depends on the penalty (devlog 2026-10-08_d19-regularization-contract); the gap follows the L1 threshold on inductive bases too (devlog 2026-10-08_d17-basis-variant-pilot). It is not a claim that FA is better than PCA, and PC8 and FA7 are related, not identical, directions (devlog 2026-10-08_d18-factor-cards).

### C2. "More malignant cells" → highest prevalence

P02, P03 and P04 are the **highest-prevalence** patients (P03 32.6%, P02 14.4%, P04 9.0%), not the ones with the most malignant cells. By malignant count the order is P03 758, P09 375, P02 211, P04 102. Together they hold 1,071 of 1,569 malignant cells but only 4,917 of 60,118 cells (8.2%) of the label-blind Stage 1 fit, which is dominated by P09 and P01 (devlog 2026-10-08_team-report-devlog-pivot; composition table in §6 and devlog 2026-08-31_d6-d9-meeting-followup).

- What is established: these patients dominate the Stage 2 positive class and most of the nonzero lift.
- What is untested: whether they shape the label-blind Stage 1 factors. That needs a leave-out or reweighted basis refit with a subspace comparison (W2.4).
- Equal per-patient weighting in Stage 1 (the team's Option 1.2) points the other way from what was intended: across 13 patients it raises the P02+P03+P04 share of the Stage 1 fit from 8.2% to 23.1% (3 of 13 patients), increasing their influence on the unsupervised factors (devlog 2026-10-08_team-report-devlog-pivot). Decide which imbalance is targeted before fitting it (W2.3), and retune C whenever training n changes.

### C3. Hypothesis (a) is retired

The team-report hypothesis (a), "PCA's orthogonality pushes later components to capture variance beyond P03, while FA keeps allocating factors toward the highest-prevalence patient", is retired (H-S2-07, verdict declared in devlog 2026-10-08_research-spine-adopted). Do not cite it as a mechanism.

- The Stage 2 design matrix is orthogonal for both methods; only FA loadings are non-orthogonal (F3).
- The driver is PC8, matched to FA7 with opposite sign, not a late component (F3).
- FA7 separates P02 and P07 on its own, and the patient FA's L1 model misses (P02) is a high-prevalence patient (F6, F9).
- C1 explains the zeros.

Sources: devlog 2026-09-17_team-feedback-f1-f7, devlog 2026-10-02_f9-ifna-pc8-loadings, devlog 2026-10-08_team-report-devlog-pivot. The C-matched comparison in W3.2 is the planned formal confirmation.

Do not confuse this with the "biology/composition vs weaker axis" options inside the feedback notebook's §6 (see [F6](#f6-why-fa-is-zeroed-out-for-p02-and-p07)).

### C4. Support rule is ≥10 malignant cells

A supported patient has at least 10 malignant cells and both classes (`MIN_MALIGNANT_SUPPORT` in `../../plans/stage2_figure_catalog_and_statistical_rigor_v2_jun4.md`, recorded in `../../research/study_config.json`): P01, P02, P03, P04, P06, P07 and P09. D14's 20-cell malignant stratum is a D14 analysis parameter, not the support rule. Any statement of a ≥20 support rule is wrong (devlog 2026-10-08_research-spine-adopted).

### C5. PC8 orientation: the malignant-leaning pole is the negative-loading group

Loading signs are arbitrary. The stored Stage 2 L1 coefficient on PC8 is negative in every active fold (−0.040 to −0.064 across the supported folds), and held-out malignant cells have a lower fold-standardized PC8 mean than healthy cells in every supported patient (devlog 2026-10-02_f9-ifna-pc8-loadings for the coefficients; devlog 2026-10-08_team-report-devlog-pivot for the F6 fold-scale means). The malignant-leaning pole of PC8 is therefore the **negative**-loading group (TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1), not the positive group led by IFI27 (devlog 2026-10-02_f9-ifna-pc8-loadings). FA7 shows the same contrast with flipped raw signs (β > 0). See [F9](#f9-pc8-loadings-within-cell-type-lift-and-fa-k-matching).

### C6. D15 is a cell-depth curve, not a patient-count curve

D15 subsamples cells at a fixed patient mix with C held fixed. It cannot justify sequencing more patients. See [D15 cell-count learning curve](#d15-cell-count-learning-curve) (devlog 2026-10-02_d15-cell-count-learning-curve).

## Important correction to the previous summary

The earlier statement that the IFN-α PCA/FA “driver factors remain highly stable at 25% donor retention (≈0.99; Jaccard 0.818)” was too broad.

The first donor report selected five “top” factors per method even though only one factor per method had a nonzero full-donor Stage 2 coefficient. Four zero-coefficient factors inflated the median stability. The corrected analysis retains only genuinely supervised factors:

- PCA40 factor 008 at 25% donors: median score correlation **0.854**, loading cosine **0.854**, top-10 gene Jaccard **0.333**.
- FA40 factor 007 at 25% donors: median score correlation **0.952**, loading cosine **0.943**, top-10 gene Jaccard **0.818**.
- At 50% donors both supervised factors are much more stable: score/loading correlations are approximately **0.98–0.99**.

Therefore the defensible conclusion is not “both bases are highly stable at 25%.” It is:

- FA factor 007 remains fairly stable at 25%.
- PCA factor 008 is stable at 50% but changes materially at 25%.
- The simultaneous all-zero Stage 2 collapse still prevents the observed AP loss from being interpreted as direct evidence of a donor-origin shortcut.

The corrected source is `stage2_old34_donor_downsampling_sensitivity.ipynb`.

## Recommended reading order



### Pass A: fastest route to the current scientific story

1. **This guide**
  - `old34_inquiry_narrative_guide.md`
  - Purpose: terminology, equations, corrections, and notebook sequence. The claim boundary lives in `../../research/THEME.md` §6.
2. **Curated team-review notebook**
  - `stage2_old34_team_review_narrative.ipynb`
  - Purpose: presentation-ready equations, exact patient table, selected D6–D11 plots, and corrected donor-factor summary. It reads completed artifacts and runs no models.
3. **Canonical all-representation diagnostic**
  - `stage2_old34_comprehensive_diagnostic_suite.ipynb`
  - Read:
    - §1 accounting and regularization-selection policy;
    - §2 D1 all-643 transfer atlas;
    - §3 canonical 41-row panel table and D2;
    - §4 D3 shared-transfer versus patient-specific opportunity.
  - Purpose: moves from 643 Stage 1 representations to panel-level interpretation without discarding method/K heterogeneity.
4. **D1a descriptive deep dives (2026-09-10)**
  - `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb`
  - Read:
    - §1 independent K fits versus a nested-prefix experiment;
    - §2 grouped D1 y-axis taxonomy (34 individual sets, 7 group unions, 2 knowledge unions, 4 HVG controls);
    - §3 patient-level allograft, apoptosis, and TNFα/NF-κB results;
    - §5 PCA variance versus FA common variance;
    - §6 one-score pathway baseline;
    - §7 per-patient rank-error, cell-type, collection-context, and CITE-seq ADT audits.
  - Purpose: reverses D1a/b aggregation before biological interpretation. It reconstructs the selected DR cell rankings exactly, but its prevalence-matched TP/FN/FP/TN labels are post-hoc ranking groups, not deployment classifications.
5. **Interpretability and factor-to-gene grounding**
  - `stage2_old34_layer4_5_interpretability_suite.ipynb`
  - Read:
    - §2–4 regularization and coefficient pathology;
    - §5–6 factor-to-gene grounding;
    - §7 interpretation contracts and validation.
  - Purpose: determines whether selected models are sparse, stable, and biologically groundable.
6. **Meeting follow-up: metric validity, cohort support, interactions, IFN-α, and no-DR baseline**
  - `stage2_old34_meeting_followup_metric_interactions.ipynb`
  - Read in this order:
    - opening “Average precision and the no-skill baseline”;
    - §1 MRD composition and empirical AP verification;
    - §2 D6 metric/prevalence diagnostics;
    - §3 D7 patient × panel × DR interactions;
    - §4 D8 IFN-α PCA40 versus FA40 attribution;
    - §6 D9 direct-gene/no-DR baselines.
  - Chronology warning: D7E records the **pre-follow-up** state (3/41 small-panel fallback baselines). D9E records the completed **post-run** state (41/41). Use D9E for current coverage.
  - Purpose: answers questions 1–4 and 6 from the 2026-07-24 meeting.
7. **Donor-downsampling sensitivity**
  - `stage2_old34_donor_downsampling_sensitivity.ipynb`
  - Read:
    - §1 exact cohort and target manifest;
    - §2 D10 fixed-basis training sensitivity;
    - §3 D11 pre-DR IFN-α basis refits;
    - §4 interpretation boundary.
  - Purpose: answers question 5 while separating coefficient collapse from Stage 1 factor movement.
8. **Whether to aggregate patients on the shared factors (D14)**
  - `stage2_old34_patient_factor_overlap.ipynb`
  - Read the estimand before the figures. Histogram overlap of stored Stage 1 scores, within normal cells and within malignant cells, against a within-label patient-label permutation.
  - Purpose: D3's predictive patient gap is not a test of factor-support overlap. Biological D1-selected bases overlap across patients. P03 malignant cells are nearly disjoint on one factor of HVG 3k, HVG 10k, and the full Stage 0 union.
  - The same notebook holds D15, the Stage 2 cell-count learning curve. Its code cells have no saved outputs; read the [D15 section](#d15-cell-count-learning-curve) of this guide before its figures.
9. **Team-feedback deep dives (F1-F7, F9)**
  - `stage2_old34_feedback_deep_dives_20260917.ipynb`
  - Read with the [F1-F9 reading guide](#f1-f9-reading-guide) below. F8 was never assigned; the notebook's §8 is text-only scoping of an inductive refit.
  - Purpose: answers the 2026-09-17 team feedback (K trends, DR vs direct genes, the IFN-α PCA40 vs FA40 split, cell type and ADT context) and the 2026-10-02 PC8 follow-up.
10. **GSEA vs HVG overlap and low-support patient DE**
  - `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb`
  - Read:
    - §1 D12 Venns and per-program HVG enrichment;
    - §3 per-patient Wilcoxon / AUROC;
    - §4 D13 unique genes, panel enrichment versus existing lift, prerank GSEA.
  - Purpose: confirms that the old34 GSEA dictionary is not an HVG alias, and asks whether P01/P06/P07 have gene-level signals that D7B’s dark columns missed.
11. **Short generated findings reports**
  - `experiments/.../analysis/scorecards/stage2_diagnostic/layer4_5/layer4_5_factor_gene_grounding_report.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/meeting_followup_v1/meeting_followup_findings.md` — metric/IFN/direct-gene findings before the donor sensitivity was completed.
  - `experiments/.../analysis/scorecards/stage2_diagnostic/donor_sensitivity_v1/donor_sensitivity_findings.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/deg_hvg_lowsupport_v1/deg_hvg_lowsupport_findings.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/d1a_deep_dive_20260910/d1a_deep_dive_findings.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/patient_factor_overlap_findings.md` — D14 overlap calls.
  - `cell_count_learning_curve_v1/` (D15) and `feedback_deep_dive_20260917/` (F1-F9) have no findings markdown. Use the D15 and F1-F9 sections of this guide and their devlog entries instead.
  - Purpose: concise result summaries after reading the methods in the notebooks.



### Pass B: historical provenance and why the newer notebooks were needed

Read these before Pass A only if you need to reconstruct the full project history.

1. `../../plans/experiment_20260525_stage0_2_rehydration_dossier.md`
  - Master provenance map as of 2026-07-02.
  - Its statements that Layers 3–5 were absent are now historical; those layers were subsequently implemented.
2. `../../plans/three_stages_knowledge_prior_mal_classification.md`
  - Conceptual Stage 0 → Stage 1 → Stage 2 design and the PI questions.
3. `../../plans/stage0_mrd_sharedness_scorecard_plan.md`
  - Operational run design, scorecard schema, and artifact definitions.
4. `stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb`
  - Historical exploratory sharedness suite.
  - Use it to understand where V2A–V2F6 came from, not as the current final analysis.
5. `../../plans/stage2_figure_catalog_and_statistical_rigor_v2_jun4.md`
  - Explains pooled-cell versus patient-equal aggregation and the required plot framing.
6. `../../plans/stage2_comprehensive_diagnostic_plan.md`
  - The original seven-layer target specification.
  - Compare it against the newer D1–D5 notebooks to see what was implemented.



### Optional conceptual aid

- `old34_heldout_lift_illustration_prompt.md`
- This is a prompt for generating a visual explanation of per-patient AP, prevalence/no-skill AP, median lift, and P20 lift.



### Normally skip

- `stage2_mrd_figure3_sharedness_suite_20260528.ipynb`
- It is retained for provenance but is superseded by the Jun 4 suite and the D1–D11 diagnostic notebooks.



### Referenced historically but absent from the repository

The July dossier names two early notebooks that are no longer present under `notebooks/stage0_2/`:

- `stage0_mrd_old34_metric_diagnostics_20260525.ipynb`;
- `stage2_mrd_multiobjective_scorecard_analysis_20260526.ipynb`.

Do not spend time searching for them. Their useful metric-QA and multi-objective-review roles are superseded by the Jun-4 sharedness suite and the current D1–D11 chain.

## 1. Why constant-score average precision equals prevalence

For one held-out patient, let:

- N_+ be malignant cells;
- N_- be healthy cells;
- N=N_+ + N_-;
- \pi=N_+/N be malignant prevalence.

Scikit-learn average precision is:

# 
\operatorname{AP}

\sum_n (R_n-R_{n-1})P_n,


where P_n and R_n are precision and recall after crossing the n-th distinct score threshold.

If every cell receives the same score c, all cells form one tie group. There are only two meaningful PR states:

1. threshold >c: no predicted-positive cells, recall 0;
2. threshold \le c: every cell predicted positive, recall 1, precision


\frac{N_+}{N_+ + N_-}=\pi.


The full recall increase is 1-0=1, so:


\operatorname{AP}_{\mathrm{constant}}
=(1-0)\pi
=\pi.


This is exact for a constant score under scikit-learn’s tie grouping. It is not merely a statement that AP and prevalence are both dimensionless.

### Important precision

The gray baseline should be named:

> constant-score no-skill AP = malignant prevalence

A finite random permutation produces a realized AP that fluctuates around the no-skill regime and is not exactly equal to prevalence in every finite sample. The exact equality above is for a constant tied score.

### Is a threshold part of AUPRC?

Yes, but not as one chosen operating threshold.

- A PR curve sweeps over every distinct prediction-score threshold.
- AP summarizes performance over that whole sweep.
- AP does not choose a single threshold for clinical use.

“Threshold utility” refers to choosing one operational cutoff t and converting probabilities into hard labels:


\hat y_i(t)=\mathbf 1[s_i\ge t].


At that chosen cutoff, one can optimize accuracy, sensitivity, specificity, or a clinical cost such as missed malignancies.

For constant hard policies:

- all healthy:
  - accuracy =1-\pi;
  - sensitivity =0;
  - specificity =1;
  - every malignant cell is missed;
- all malignant:
  - accuracy =\pi;
  - sensitivity =1;
  - specificity =0;
  - no malignant cell is missed.

Both policies have constant ranking scores and therefore AP =\pi, but their hard-decision utilities are opposite. This is why AP and threshold-specific clinical utility must be reported separately.

### Lift

For held-out patient h:


L_h=\operatorname{AP}_h-\pi_h.


The ceiling-normalized version is:

# 
L_h^{\mathrm{norm}}

\frac{\operatorname{AP}_h-\pi_h}{1-\pi_h}.


Raw lift answers how many AP units the model gains over constant no-skill AP. Normalized lift reports the fraction of the available improvement from \pi_h to 1.

## 2. What “median within-representation Spearman rho” means

The Greek symbol is \rho (“rho”), not a p-value.

For each representation r, there are seven supported held-out patients:


H=\mathrm{P01,P02,P03,P04,P06,P07,P09}.


For a metric M, such as held-out lift, construct:


\mathbf M_r=(M_{r,h}:h\in H).


For a patient covariate C, such as prevalence, construct:


\mathbf C=(C_h:h\in H).


Spearman rho is the Pearson correlation between their ranks:

# 
\rho_{r,M,C}

\operatorname{corr}
\left(
\operatorname{rank}(\mathbf M_r),
\operatorname{rank}(\mathbf C)
\right).


Without ties, it can also be written:

# 
\rho

1-\frac{6\sum_{h=1}^{7}d_h^2}{7(7^2-1)},


where d_h is the difference between the two ranks. SciPy uses average ranks when ties occur.

This calculation is repeated separately for all 643 representations. The reported number is then:


\operatorname{median}*{r=1}^{643}\rho*{r,M,C}.


It is not one correlation over 4,501 pooled rows, and it is not a median p-value.

Verified median within-representation rho values:

- AP versus prevalence: **0.964**;
- raw lift versus prevalence: **0.821**;
- normalized lift versus prevalence: **0.821**;
- raw lift versus malignant count: **0.607**;
- raw lift versus healthy count: **−0.571**.

There is also a different patient-level summary: first take the median metric over 643 representations for each patient, then correlate those seven patient medians with prevalence. That gives rho **0.893** for median raw lift versus prevalence. Do not conflate these two analyses.

### What about the p-value?

`scipy.stats.spearmanr` returns a nominal two-sided p-value for each seven-patient correlation. Those values are stored as `nominal_pvalue_not_for_inference`.

They should not be used as publication-level evidence because:

- n=7 is extremely small;
- the usual SciPy p-value is an asymptotic approximation;
- 643 representations are strongly dependent because they reuse panels, cells, and patients;
- no multiple-testing correction turns these dependent exploratory tests into an identified biological effect.

The D6 conclusion is therefore descriptive: across many fixed representations, high-prevalence patients tend to rank higher in AP and lift.

### Why biology is not yet separable from patient support

The result does not prove that the lift is non-biological. It shows that the current design cannot identify the two explanations.

Potential support-driven mechanisms include:

- more malignant evaluation cells reduce AP uncertainty;
- prevalence itself changes PR geometry;
- pooled-cell regularization selection gives more influence to high-support patients;
- Stage 1 is transductive and includes all eligible patients;
- patient prevalence, malignant count, cell type, and disease biology are confounded in only seven supported patients.

Patient-specific biology may still be real. To separate it from support, use the larger cohort and add matched-support analyses, malignant-cell subsampling, patient bootstrap/permutation procedures, and preferably a prespecified or nested model-selection rule.

## 3. What the direct-gene/no-DR baseline is

“Direct gene” means no Stage 1 latent decomposition.

It does not mean raw count data. The feature path is:

1. library-size normalization and log1p;
2. select the genes in one Stage 0 panel;
3. standardize each gene;
4. fit the same regularized Stage 2 logistic-regression machinery directly on those gene columns.

For panel genes g=1,\ldots,G:

# 
\Pr(y_i=1\mid\mathbf x_i)

\sigma\left(
\beta_0+\sum_{g=1}^{G}\beta_g z_{ig}
\right).


Each \beta_g therefore maps a named gene, rather than a latent factor, to the malignancy score.

This was completed for all **41/41 biological panels**:

- 41 direct-gene representations;
- discovery and LOPO regularization paths;
- 7 selected held-out-patient rows per panel, 287 total.

Only discovery and LOPO were requested for this direct-gene follow-up; a separate patient-specific direct-gene training grid was not run.

The Stage 0 runner gained `--direct-gene-all-panels`; the completed run is under the `old34_direct_gene_all_v1` artifact branch.

### Meaning of the 38/41, 16/41, and 17/41 result

For each panel:

1. choose the DR representation with the highest pooled-cell LOPO AP among its PCA/FA/FactoSig/method-K candidates;
2. compare that selected DR representation with the panel’s direct-gene representation.

Results:

- best DR has higher pooled-cell LOPO AP in **38/41** panels;
- best DR has higher seven-patient median lift in **16/41** panels;
- best DR has higher seven-patient P20 lift in **17/41** panels.

These are three different estimands:

- pooled AP concatenates held-out cells and is cell/support weighted;
- median lift gives each of seven supported patients equal weight;
- P20 lift emphasizes the lower tail.

With seven values sorted x_{(1)}\le\cdots\le x_{(7)}, pandas’ default P20 uses linear interpolation at zero-based position 0.2(7-1)=1.2:


\operatorname{P20}=0.8x_{(2)}+0.2x_{(3)}.


The results say that DR usually improves the pooled ranking but does not consistently improve the typical patient or lower-tail patient.

### Fairness caveat

“Best DR” is selected from many method/K candidates using the same pooled criterion being reported. The direct baseline has one feature representation. The 38/41 count therefore has winner’s-curse/model-selection optimism and is exploratory until selection is nested or evaluated externally.

## 4. How IFN-α PCA factor 008 was identified

The targeted comparison fixes:

- panel: HALLMARK interferon-alpha response, 92 covered genes;
- K: 40;
- Stage 1 seed: 42;
- methods: PCA versus scikit-learn FA;
- Stage 2 regularization: globally selected L1, C=10^{-4}.



### Step 1: reconstruct the fitted LOPO ranking

For held-out patient h, Stage 2-standardized factor score z_{ihj}, and stored coefficient \beta_{hj}:


s_{ih}=\sum_{j=1}^{40}\beta_{hj}z_{ihj}.


The intercept is irrelevant to AP ranking. Reconstructed AP matched stored AP with maximum absolute error below 1.4\times10^{-6}.

### Step 2: fixed-model factor ablation

For factor j:


s_{ih}^{(-j)}=s_{ih}-\beta_{hj}z_{ihj},


# 
\Delta\operatorname{AP}_{h,j}

\operatorname{AP}(s_h)-\operatorname{AP}(s_h^{(-j)}).


A factor is called a driver when removing its already fitted contribution produces the largest cumulative AP loss over the focus patients. This is attribution of the fitted model, not a refit-based causal test.

### Observed fold pattern

- PCA factor 008 is the only active PCA coefficient in P02, P03, P04, P06, and P07.
- Its AP drop equals the full lift in those folds because ablating the only active coefficient leaves a constant score.
- FA factor 007 is active only in P03 and P04.
- FA is all-zero in P02 and P07 at the selected global L1 regularization.

Therefore the concrete P02/P07 PCA advantage is:

> PCA retains a nonzero factor-008 probe; FA’s selected sparse model retains no factor.

> **Stale as a mechanism (2026-10-08).** The fold pattern above is correct, but the cause is the L1 entry threshold 4/(C·n_train) at the single stored C, not a missing FA axis. At a C 5% above the stored value, FA40-L1 keeps FA7 and recovers P02 (P02 is a regularization knife-edge); PCA keeps an advantage only in P07. See [Corrections C1](#c1-4-knife-edge-revision-the-ifn-α-pca40-vs-fa40-gap-is-a-stage-2-l1-at-fixed-c-effect).



### Step 3: compare PCA and FA axes

PCA and FA factors are not compared by factor number. Their all-cell score columns are correlated and matched one-to-one by a Hungarian assignment maximizing absolute correlation.

For PCA factor 008, the two strongest FA score correlations are:

- FA factor 007: -0.685;
- FA factor 008: +0.580.

This supports the narrower statement that PCA factor-008 variation is distributed mainly across FA axes 007 and 008 rather than reproduced by one strongly matching FA axis.

It does not prove that FA lacks the biology. The observed failure can be an interaction among:

- the FA geometry;
- factor splitting/rotation;
- degenerate late FA factors;
- L1 regularization at the globally selected C.

> **Update (2026-10-08).** F3, F6 and F9 isolate the last item. FA scores are uncorrelated, FA7 alone separates P02 and P07, and L2 on the same FA scores recovers both, so the P02/P07 zeros are the L1-at-fixed-C term. Whether the transductive basis contributes is still open (inductive refit, W2.1). See [Corrections C1](#c1-4-knife-edge-revision-the-ifn-α-pca40-vs-fa40-gap-is-a-stage-2-l1-at-fixed-c-effect) and the [F1-F9 reading guide](#f1-f9-reading-guide).

Also, scikit-learn FactorAnalysis is not simply “PCA with relaxed orthogonality.” Its latent factors use a standard independent Gaussian prior and it models feature-specific noise; the loadings need not behave like PCA components, but this is not an oblique-factor model.

Best review plots in `stage2_old34_meeting_followup_metric_interactions.ipynb`:

- D7D: per-patient PCA40 versus FA40 lift;
- D8B: active coefficients by held-out fold;
- D8C: full versus ablated-driver AP;
- D8D: top loading genes;
- D8E: PCA factor-008 correlations to all FA factors.



## 5. Donor downsampling, coefficient collapse, and factor stability



### Design

For donor-healthy retention fractions 1.0, 0.5, 0.25, 0.1:

- remove donor-healthy training cells only;
- stratify masks by patient × predicted cell type;
- preserve all malignant and recipient-healthy cells;
- use three mask seeds below fraction 1;
- evaluate on the full unchanged held-out patient.

Two arms were run:

1. fixed Stage 1 basis and fixed Stage 2 regularization;
2. IFN-α PCA40/FA40 basis refit before Stage 2, retaining the original transductive Stage 1 scope.



### What all-zero sparse-model collapse means

The IFN-α PCA40, FA40, and direct-gene comparators use L1 with C=10^{-4}. At 25% donors, every evaluated fold/seed has:


\boldsymbol\beta=\mathbf 0.


Then:


s_i=\beta_0


is constant for every held-out cell, and:


\operatorname{AP}=\pi.


The apparent AP loss is therefore exactly loss of the previously nonzero sparse solution. It is not evidence that the retained model actively confused donor and malignant cells.

This diagnoses a fixed-regularization-path failure. A donor-shortcut performance test still requires a rescue analysis that reselects or appropriately rescales regularization within each downsampling condition.

> **Update (2026-10-08).** The collapse is expected mechanically. Under balanced-weight L1 a standardized factor enters only if its training class-mean gap exceeds 4/(C·n_train). Donor-healthy cells are most of every training fold, so removing them lowers n_train and raises the threshold in proportion; at fixed C an all-zero probe is what the rule predicts. The rescue is C rescaled with n or reselected within each condition, with L2 reported alongside (W4.6). See [Corrections C1](#c1-4-knife-edge-revision-the-ifn-α-pca40-vs-fa40-gap-is-a-stage-2-l1-at-fixed-c-effect). The factor-stability results below do not depend on Stage 2 and are unaffected.

### Why factor stability can still be measured

Stage 1 factors exist before the Stage 2 coefficient vector is set to zero. The pre-DR arm compares the factor bases directly.

#### Score correlation

After one-to-one factor alignment:


\left|\operatorname{corr}
\left(
\mathbf z_j^{\mathrm{full}},
\mathbf z_{k(j)}^{\mathrm{down}}
\right)\right|.


This measures whether cells retain the same ordering along an aligned latent axis. Absolute correlation ignores arbitrary sign flips.

#### Loading cosine

After orienting signs and expressing loadings in a common gene scale:

# 
\cos(\boldsymbol\lambda,\tilde{\boldsymbol\lambda})

\frac{
\boldsymbol\lambda^\top\tilde{\boldsymbol\lambda}
}{
\boldsymbol\lambda_2
\tilde{\boldsymbol\lambda}_2
}.


This measures similarity of the full gene-weight direction and is insensitive to overall magnitude.

#### Top-10 gene Jaccard

Let A and B be the top 10 genes by absolute loading before and after downsampling:


J(A,B)=\frac{|A\cap B|}{|A\cup B|}.


For two 10-gene sets:

- J=0.818=9/11 means 9 of 10 genes overlap;
- J=0.667=8/12 means 8 of 10 overlap;
- J=0.333=5/15 means 5 of 10 overlap.

Jaccard directly answers whether the named top genes remain the same. It intentionally ignores rank, sign, and magnitude, so it must be interpreted together with loading cosine.

### Corrected 25% result

- FA factor 007:
  - median score correlation 0.952;
  - median loading cosine 0.943;
  - median top-10 Jaccard 0.818.
- PCA factor 008:
  - median score correlation 0.854;
  - median loading cosine 0.854;
  - median top-10 Jaccard 0.333.

The FA driver is comparatively robust at 25%; the PCA driver is only moderately aligned and replaces about half of its top-10 genes. At 50%, both are strongly aligned.

Best review plots in `stage2_old34_donor_downsampling_sensitivity.ipynb`:

- D10C: patient-specific AP changes at 25%;
- D10E: all-zero coefficient frequency;
- D11B: aligned factor-score stability;
- D11C: corrected nonzero-supervised-factor loading and gene stability.



## 6. Exact cohort composition across patients


| Patient   | Malignant | Donor healthy | Recipient healthy | Total      | Malignant prevalence | Included in 7-patient lift summary |
| --------- | --------- | ------------- | ----------------- | ---------- | -------------------- | ---------------------------------- |
| P01       | 45        | 11,815        | 0                 | 11,860     | 0.379%               | Yes                                |
| P02       | 211       | 1,234         | 17                | 1,462      | 14.432%              | Yes                                |
| P03       | 758       | 1,568         | 1                 | 2,327      | 32.574%              | Yes                                |
| P04       | 102       | 1,012         | 14                | 1,128      | 9.043%               | Yes                                |
| P05       | 8         | 4,341         | 10                | 4,359      | 0.184%               | No                                 |
| P06       | 44        | 3,214         | 1                 | 3,259      | 1.350%               | Yes                                |
| P07       | 22        | 2,163         | 0                 | 2,185      | 1.007%               | Yes                                |
| P08       | 0         | 1,411         | 0                 | 1,411      | 0%                   | No                                 |
| P09       | 375       | 16,745        | 6                 | 17,126     | 2.190%               | Yes                                |
| P10       | 0         | 2,715         | 0                 | 2,715      | 0%                   | No                                 |
| P11       | 0         | 4,024         | 0                 | 4,024      | 0%                   | No                                 |
| P12       | 0         | 2,488         | 0                 | 2,488      | 0%                   | No                                 |
| P13       | 4         | 5,769         | 1                 | 5,774      | 0.069%               | No                                 |
| **Total** | **1,569** | **58,499**    | **50**            | **60,118** | **2.610%**           | —                                  |


Source:

- `experiments/.../analysis/scorecards/stage2_diagnostic/meeting_followup_v1/mrd_patient_cell_composition.csv`
- D6A in `stage2_old34_meeting_followup_metric_interactions.ipynb`

Wording note (2026-10-08): rank P02/P03/P04 by the prevalence column, not the malignant column. They are the highest-prevalence patients; P09 has more malignant cells than P02 or P04. See [Corrections C2](#c2-more-malignant-cells--highest-prevalence).

## D15 cell-count learning curve

- Notebook: `stage2_old34_patient_factor_overlap.ipynb` (the D15 cells; the code cells have no saved outputs). Module: `../../stage2_supervised/cell_count_learning_curve.py`.
- Artifacts (EXP-relative): `analysis/scorecards/stage2_diagnostic/cell_count_learning_curve_v1/` (`cell_count_learning_curve.csv`, `cell_count_learning_curve_summary.csv`, `cell_count_learning_curve_atlas_agreement.csv`) and `analysis/figures/stage2_diagnostic/cell_count_learning_curve_v1/` (`D15A_auprc_vs_training_cells`, `D15B_median_lift_vs_training_cells`). There is no findings markdown.
- Devlog: 2026-10-02_d15-cell-count-learning-curve. Next-step tasks live in the W6 workstream README.

### What it is

D15 is a **Stage 2 cell-depth curve at a fixed patient mix**.

- For each of the seven supported held-out patients, it draws label-stratified random subsets of that fold's training cells (500, 1k, 2k, 5k, 10k and 25k cells), plus the full training set. Every draw contains all 12 training patients. The held-out patient is always scored in full. The cohort summary is the median over the seven patients of each patient's mean over draws.
- **Frozen transductive basis.** The Stage 1 scores are the stored all-cell fits (`stage1_dr/.../scores.npy`). No DR is refit per subset, so the held-out patient's cells and the left-out training cells helped define the factors.
- **C is not retuned.** Each subset refits the scaler and the atlas-selected penalty at the atlas-selected C: L1 at C = 10⁻⁴ for IFN-α PCA40 and FA40, L2 at C = 10⁻⁴ for apoptosis promax40, and elastic net (l1_ratio 0.5) for allograft promax40. The full-training refit matches the atlas AUPRC to within 0.000164 (devlog 2026-10-02_d15-cell-count-learning-curve).
- **Three draws** per size, with no confidence intervals.

### How to read it

All values are cohort-median held-out lift (devlog 2026-10-02_d15-cell-count-learning-curve).

| Training cells | 500 | 1k | 2k | 5k | 10k | 25k | full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Median training malignant cells | 13 | 27 | 53 | 134 | 267 | 668 | 1,467 |
| Apoptosis promax40, L2 (pp) | 7.65 | 9.28 | 10.21 | 9.63 | 10.92 | 11.72 | 11.52 |
| Allograft promax40, EN (pp) | 0 | 0 | 0 | 1.67 | 3.14 | 8.81 | 12.56 |
| IFN-α PCA40, L1 (pp) | 0 | 0 | 0 | 0 | 0 | 0 | 11.08 |
| IFN-α FA40, L1 (pp) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- Full-training n differs by fold, from 42,992 (P09 held out) to 58,990 (P04 held out).
- Only the apoptosis (L2) curve is informative about cell depth, and even it is not monotone per patient: P02 is 32.39 pp at 1k versus 39.80 pp at full, while P03 is 48.03 versus 47.15 pp and P04 20.67 versus 17.93 pp.
- The full-set IFN-α FA40 median is 0 because FA40-L1 keeps a factor only in P03 (14.37 pp) and P04 (19.87 pp).

### Why the IFN-α L1 curves are flat by construction

A standardized factor can leave zero under balanced-weight L1 only if its training class-mean gap exceeds 4/(C·n_train) (see [Corrections C1](#c1-4-knife-edge-revision-the-ifn-α-pca40-vs-fa40-gap-is-a-stage-2-l1-at-fixed-c-effect)). Label-stratified subsampling keeps the class balance, so the gaps barely move while n_train shrinks. At C = 10⁻⁴ the threshold is 1.6 SD at 25k cells and 8 SD at 5k (devlog 2026-10-02_d15-cell-count-learning-curve), far above the strongest IFN-α factor gaps (PC8, 0.70–0.77 SD; devlog 2026-10-08_team-report-devlog-pivot). No factor can enter until n_train reaches roughly 54,000 cells (devlog 2026-10-02_d15-cell-count-learning-curve). Below that every IFN-α probe is all-zero, the score is constant, AP = π, and lift is 0. The flat curves therefore say nothing about how much IFN-α signal a smaller training set holds. The allograft elastic net has an L1 part, so its zeros at small n have the same mechanical component.

### What the curve does not show

- It is not a patient-count curve. The x-axis is cells at a fixed 12-patient mix, so it says nothing about the value of sequencing more patients (the grant argument).
- It is not inductive: the basis is frozen and transductive.
- It is not regularization-fair across sizes: with C fixed, the effective penalty grows as 1/n. C was also selected on the full cohort, on the same folds.
- It has no uncertainty estimate (three draws, no CIs).
- It is not evidence that the IFN-α signal needs more than about 54,000 cells.

### Next steps

- **W6.1 (D15b), patient-count learning curve.** For each held-out supported patient, train on k = 1..5 of the other supported patients over all combinations, with unsupported patients always included. Retune C and the penalty for each training set through the W3.1 regularization contract. Report patient-bootstrap CIs and fit a weighted inverse power law.
- **W6.2 (D15 v2), cell-depth curve with C retuned per subsample.** More draws, an x-axis in training malignant cells and malignant cells per feature, and a findings markdown. Its devlog entry states that v2 replaces v1's interpretation.
- Both depend on W3.1 (the D19 regularization contract).

## F1-F9 reading guide

- Notebook: `stage2_old34_feedback_deep_dives_20260917.ipynb`.
- Tables (EXP-relative): `analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/`. Figures: `analysis/figures/stage2_diagnostic/feedback_deep_dive_20260917/`.
- F1-F7 were written on 2026-09-18 in response to the 2026-09-17 team feedback (devlog 2026-09-17_team-feedback-f1-f7). F9 was added on 2026-10-02 (devlog 2026-10-02_f9-ifna-pc8-loadings).
- There is no findings markdown, and `f_artifact_ledger.csv` omits the F9 tables and figures (documentation debt, W0.5).
- The F IDs are a separate namespace from the Jun-4 suite's V2F figure IDs (V2F1 is not F1). The F series is closed.

### F1. Median lift vs independently fitted K

- **Question.** Does lift rise with K for knowledge panels, does HVG+PCA peak at K = 10, and does FA rise monotonically? (Reconstructed from the team feedback.)
- **Artifact.** `f1_biological_median_lift_by_method_k.csv`, `f1_hvg_median_lift_by_method_k.csv`, `f1_k_trend_monotonicity.csv`, `f1_pca_independent_k_vs_k40_prefix.csv`, `f1_scree_*.csv`, `f1_independent_k_cumulative_energy.csv`; figures F1ab, F1b2 and F1c-F1h.
- **How to read it.** Each value is the median over panels of the seven-patient median lift for one method × K (devlog 2026-09-17_team-feedback-f1-f7).
  - Biological panels at K 5/10/20/40: FA −0.00/3.24/3.73/4.64 pp; FactoSig 0.63/2.91/5.77/7.19; promax 0.38/3.00/5.02/5.66; PCA 0.29/3.55/3.98/4.22.
  - The typical panel gains with K, but single panels often do not. Lift is nondecreasing across K in only 18 (PCA), 16 (FactoSig), 14 (FA) and 14 (promax) of the 43 non-HVG panels.
  - F1h: PCA independent-K fits are close to prefixes of the K = 40 fit (IFN-α cumulative explained variance 0.1616/0.2268/0.3358/0.5427 at K 5/10/20/40).
- **Caveat.** Each K is an independent Stage 1 fit, and only PCA is approximately nested. The selected C also changes with K, so a K trend mixes basis capacity with penalty selection. The panel set shrinks at the larger K values (the `n_panels` column).

### F2. Where DR beats direct genes; selected factors and top loading genes

- **Question.** For which panels is the gap between the best DR configuration and the panel's direct-gene probe largest, and which factors and genes do those selected models use?
- **Artifact.** `f2_best_dr_minus_direct_median_gap.csv`, `f2_focus_patient_direct_vs_dr_comparators.csv`, `f2_winner_coefficient_summary.csv`, `f2_winner_selected_factor_top_loading_genes.csv`, `f2_allograft_lck_zap70_loadings_on_active_factors.csv`, `f2_focus_panel_gene_set_overlaps.csv`; figures F2a-F2d.
- **How to read it.** The three largest median-lift gaps are (devlog 2026-09-17_team-feedback-f1-f7):
  - allograft promax40, 12.56 pp vs direct genes 0.23 pp (gap 12.33);
  - apoptosis promax40, 11.52 pp vs direct genes about 0;
  - IFN-α PCA40, 11.08 pp vs direct genes 1.27 pp (gap 9.81).
  The selected models use different penalties: allograft elastic net (30 factors ever nonzero), apoptosis L2 (all 40 factors) and IFN-α L1 (one factor, PC8, with two supported patients all-zero). F2d lists top loading genes per selected factor.
- **Caveat.** The winner is chosen in-sample from 16 DR cells per panel, and regularization is selected on the same folds. Loading signs are arbitrary. factosig_promax `loadings.npy` stores the pattern matrix, whose column scale is arbitrary (devlog 2026-10-08_literature-reviews), so compare promax loadings within a factor, not across factors.

### F3. IFN-α PCA40 vs FA40: matching, collinearity and local refits

- **Question.** Why does PCA40 beat FA40 on IFN-α in P02 and P07? Is L1 choosing between collinear FA columns (the account given at the 2026-09-17 meeting)?
- **Artifact.** `f3_ifna_pca_fa_hungarian_alignment.csv`, `f3_ifna_within_method_collinearity.csv`, `f3_ifna_local_l1_l2_en_refit_on_stored_scores.csv`; figures F3a-F3f.
- **How to read it** (devlog 2026-09-17_team-feedback-f1-f7):
  - F3a: Hungarian matching pairs PC8 with FA7 (r −0.685) and PC7 with FA8 (r +0.692).
  - F3b: the largest off-diagonal score |r| is 0.000 for PCA, 0.003 for FA, 0.403 for FactoSig and 0.924 for promax. FA scores are uncorrelated; only FA loadings are non-orthogonal (loading Gram maximum 0.403).
  - F3c: the stored L1 model keeps PC8 in P02, P03, P04, P06 and P07, and FA7 only in P03 and P04.
  - F3d: a local refit at the same C = 10⁻⁴ with L2 recovers P02 (16.14 pp) and P07 (5.07 pp) for FA, against 13.23 and 5.60 pp for PCA under the same L2. The D19 contract refit of the same L2 arm gives FA 15.97 / 5.07 pp and PCA 13.19 / 5.60 pp; the refits differ by up to 0.17 pp, presumably through solver and tolerance, and D19 is the reference value (devlog 2026-10-08_d19-regularization-contract).
- **Caveat.** The local refits use the stored transductive scores. They are not a new Stage 2 grid, and they use one C. The collinear-L1 account is contradicted, because FA scores are uncorrelated. The mechanism is in [Corrections C1](#c1-4-knife-edge-revision-the-ifn-α-pca40-vs-fa40-gap-is-a-stage-2-l1-at-fixed-c-effect).

### F4. Direct genes vs the best remaining DR configuration

- **Question.** Does the best DR configuration beat direct genes on patient-median lift, per panel and per patient?
- **Artifact.** `f4_direct_vs_best_remaining_dr_median.csv`, `f4_direct_vs_best_remaining_dr_by_patient.csv`; figures F4a and F4b.
- **How to read it.** The best of 16 DR cells beats direct genes on median lift in 28 of 41 biological panels and 2 of 6 controls (devlog 2026-09-17_team-feedback-f1-f7). Read it next to D9 in §3, which picks the best DR configuration by pooled AP instead: higher pooled AP in 38/41 panels, higher median lift in 16/41 and higher P20 lift in 17/41 (devlog 2026-08-31_d6-d9-meeting-followup). Always quote the three estimands together.
- **Caveat.** A maximum over 16 cells is in-sample winner selection (winner's curse), and none of this is nested. Nested selection is W3.3 (D23).

### F5. Covered-gene overlaps among immune programs

- **Question.** How much do the immune and inflammatory programs share covered genes, and do the per-patient winners coincide?
- **Artifact.** `f5_focus_immune_geneset_jaccard.csv`, `f5_focus_immune_geneset_shared_counts.csv`, `f5_immune_program_winner_patient_lift.csv`; figures F5a and F5b.
- **How to read it.** Allograft and TNFα/NF-κB share 13 covered genes, and allograft and apoptosis share 15. The 92 covered IFN-α genes overlap Hallmark IFN-γ in 69 (devlog 2026-09-17_team-feedback-f1-f7). Lift on these panels is therefore not a set of independent tests.
- **Caveat.** The overlaps are on covered Stage 0 genes. A shared gene count says nothing about which genes the selected factors weight.

### F6. Why FA is zeroed out for P02 and P07

- **Question.** Is FA's all-zero model in P02/P07 a biology or cell-composition difference, or a slightly weaker axis that L1 cuts off at this C? (The notebook labels these options (a) and (b); they are not the team-report hypothesis (a) in [Corrections C3](#c3-hypothesis-a-is-retired).)
- **Artifact.** `f6_ifna_univariate_factor_ap_by_patient.csv`, `f6_ifna_factor7_8_fold_scale_by_patient.csv`, `f6_ifna_within_patient_pca8_fa7_correlation.csv`, `f6_ifna_pca8_fa7_by_predicted_celltype_p02_p03_p07.csv`; figures F6a-F6c.
- **How to read it** (devlog 2026-09-17_team-feedback-f1-f7):
  - In every fold the strongest single training factor is FA7 for FA and PC8 for PCA, and PC8's training AP is above FA7's in all seven folds.
  - The fold-standardized held-out SD is close to 1 for both factors, so no scale collapse.
  - Held-out malignant cells sit on FA7: mean z 0.855 in P02 and 1.226 in P07.
  - Within-patient r(PC8, FA7) runs from −0.575 to −0.722.
  - The answer is the weaker-axis option.
- **Caveat.** The notebook's §6 has no written verdict cell; the conclusion is implied in its §3, §7 and the F9c reading. The basis is transductive.

### F7. Rank-matched confusion and ADT context

- **Question.** At a rank cutoff equal to each patient's malignant count, which malignant cells does each IFN-α model miss, and do the missed cells differ in surface protein?
- **Artifact.** `f7_ifna_pca_fa_rank_matched_confusion.csv`, `f7_ifna_pca_fa_rank_error_by_celltype.csv`, `f7_ifna_pca_fa_adt_fn_minus_tp.csv`; figures F7a and F7b.
- **How to read it.** Malignant recall at the cutoff (devlog 2026-09-17_team-feedback-f1-f7):
  - P02: PCA L1 0.322 vs FA local L2 0.336.
  - P03: 0.478 for both L1 models (FA L2 0.512).
  - P04: PCA 0.265 vs FA L1 0.363.
  - P07: PCA 0.136 (3/22) vs FA L2 0.045 (1/22).
  Where the stored FA L1 model is all-zero, F7 substitutes the local L2 refit.
- **Caveat.** The cutoff uses held-out labels: it is a post-hoc ranking group, not an operating threshold. P07 has only 22 malignant cells. The F7b column `median_log1p_fn_minus_tp` holds differences of raw ADT counts, not log1p values, and no CLR/dsb normalization was applied (fix in W5.3).

### F8. Never assigned

F8 was never assigned. The notebook's §8 is text-only scoping of an inductive Stage 1 refit, with no figure and no table. That refit is now W2.1 (D17). Do not reuse the ID.

### F9. PC8 loadings, within-cell-type lift and FA K-matching

- **Question.** What are PC8's two gene groups, and when does P04 join FA7? (User follow-up, 2026-10-02.)
- **Artifact.** `f9_ifna_pc8_loading_vs_gene_univariate.csv`, `f9_ifna_pc8_loading_vs_gene_univariate_summary.csv`, `f9_ifna_pc8_vs_celltype_ceiling.csv`, `f9_ifna_pc8_within_celltype_lift.csv`, `f9_ifna_fa_k_match_to_fa40_factor7.csv`, `f9_ifna_fa_matched_factor_lift_by_k.csv`; figures F9a-F9c.

How to read it (devlog 2026-10-02_f9-ifna-pc8-loadings unless noted):

- **PC8 orientation.** The top positive loadings are IFI27 (+0.315), PSME2, PSMA3, CNP, LAP3 and USP18. The top negative loadings are TXNIP (−0.269), IFITM2, TENT5A, NCOA7, TRIM5 and ELF1. No gene appears on both sides. Loading signs are arbitrary. Stage 2 weights PC8 negatively in every active fold (−0.040 to −0.064), and held-out malignant cells have lower PC8 than healthy cells in every supported patient (F6 fold-scale table; devlog 2026-10-08_team-report-devlog-pivot). The malignant-leaning pole is therefore the **negative** group, TXNIP/IFITM2/TENT5A/NCOA7/TRIM5/ELF1 ([Corrections C5](#c5-pc8-orientation-the-malignant-leaning-pole-is-the-negative-loading-group)). IFI27 and IFITM2 are both interferon-stimulated genes, yet they load on opposite poles.
- **F9a, loading vs single-gene lift.** |PC8 loading| and median single-gene held-out lift have Spearman 0.30 across the 92 genes. IFI27 alone gives 0.39 pp. The best single genes are BST2 (4.06 pp), LY6E (2.93) and GBP2 (2.77), all at modest loadings. PC8 is a combination axis, not a ranking of the best single interferon genes.
- **F9b, within-cell-type lift.** PC8 still separates malignant from normal cells inside a single predicted cell type:
  - HSC/MPP2: P02 +21.8 pp (45 malignant vs 44 normal), P03 +12.7 (438 vs 170), P07 +10.3 (10 vs 164), P09 +27.1 (129 vs 1,536);
  - P04: HSC/MPP1 +24.7 and MEP-EP +23.0;
  - P03 NP1: PC8 +14.4 against FA7 −3.5, so the two axes are related across all cells but not interchangeable in every compartment.
- **F9b, cell-type ceiling.** The ceiling scores each cell by its cell type's malignant fraction in that patient. Ceiling vs PC8: P02 57.33 vs 15.94 pp, P03 38.04 vs 13.40 pp, P07 8.17 vs 11.08 pp. Cell-type mix alone outruns PC8 in P02 and P03 but not in P07. PC8 is not only an HSC proxy.
- **F9c, FA K-matching.** The factor matching K = 40 FA7 is factor 7 at K = 10 (r +0.932) and factor 7 at K = 20 (r −0.774, a sign flip of the same axis); K = 5 has no match (best |r| 0.044). On that axis P04's single-factor lift is 13.51 pp at K = 10, 21.72 at K = 20 and 19.87 at K = 40, but the selected L1 model keeps only P03 at K = 10 and K = 20. At K = 40, FA7 alone gives 18.06 pp in P02 and 6.97 pp in P07. P04 "joining" at K = 40 is the L1 model starting to keep a patient already on this axis, not a new direction.

Caveats:

- Gene AUPRC in F9a is sign-agnostic, so F9a cannot say whether a gene is higher or lower in malignant cells. A per-patient direction table is W5.2.
- The within-cell-type and single-factor lifts orient the score within each patient using that patient's labels. They describe a direction; they are not a deployed model.
- Within-type lift needs at least 8 malignant and 8 normal cells per type. The P07 HSC/MPP2 value rests on 10 malignant cells.
- The cell-type ceiling uses held-out labels and `predicted.annotation`, which is itself a prediction.
- The basis is transductive.



## Suggested team-presentation narrative

1. **Question and pipeline:** prior gene panel → latent or direct-gene representation → regularized malignant probe.
2. **Cohort reality:** show D6A and the patient table; emphasize seven supported patients and prevalence heterogeneity.
3. **Metric contract:** prove constant-score AP =\pi; distinguish AP threshold sweep from one clinical cutoff.
4. **Patient-support diagnostic:** show D6B–D6D and state that support and biology are not identified separately.
5. **Panel/DR landscape:** show D1/D2 or D7A/D7B; method choice matters within panels.
6. **No-DR value added:** show D9A–D9D; pooled DR benefit is broad, patient-equal benefit is mixed.
7. **IFN-α case study:** show D7D, D8B, D8C, and D8E; distinguish retained PCA factor from all-zero FA folds, then show F3d/F9c and state the 2026-10-08 knife-edge correction (Corrections C1).
8. **Donor sensitivity:** show D10E before any AP-loss plot, then D11B/D11C with the corrected factor-specific result.
9. **Claim gate:** separate current descriptive findings from publication claims requiring controls, using the claim list in `../../research/THEME.md` §6.



## Current claim boundary

The canonical claim list lives in [`../../research/THEME.md`](../../research/THEME.md) §6. This guide no longer keeps its own list: link THEME §6 rather than copying it, and use the claim-gate language in repo-root `AGENTS.md` (R6).

The pre-migration list that stood here (as of commit b60a6ee: the "Supported as descriptive findings" and "Not yet supported as publication-level claims" lists, plus the priority additions for the larger cohort) is preserved verbatim in devlog `2026-10-08_research-spine-adopted` ([`../../research/devlog/2026/2026-10-08_research-spine-adopted.md`](../../research/devlog/2026/2026-10-08_research-spine-adopted.md)).
