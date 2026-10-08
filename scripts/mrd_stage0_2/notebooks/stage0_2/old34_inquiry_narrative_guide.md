# Old34 inquiry narrative and reading guide

Updated: 2026-10-02  
Canonical experiment: `20260525_060508_stage0_mrd_old34_broad_screen_82db5093`

For **live screen/GPU status**, namespaced campaigns (pooled unions, control full-grid, Set B), and what another agent should not relaunch, see `../../plans/current_experiment_status_rehydration_20260904.md` (snapshot 2026-09-04). This guide is the scientific reading order, not the process inventory.

This is the current “read me first” document for reconstructing the old34 inquiry. It separates:

1. historical design and exploratory work;
2. completed diagnostics;
3. meeting-driven follow-up analyses;
4. claims that are currently supportable;
5. claims that still require stronger controls.

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
  - Purpose: terminology, equations, corrections, notebook sequence, and claim boundaries.
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
9. **GSEA vs HVG overlap and low-support patient DE**
  - `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb`
  - Read:
    - §1 D12 Venns and per-program HVG enrichment;
    - §3 per-patient Wilcoxon / AUROC;
    - §4 D13 unique genes, panel enrichment versus existing lift, prerank GSEA.
  - Purpose: confirms that the old34 GSEA dictionary is not an HVG alias, and asks whether P01/P06/P07 have gene-level signals that D7B’s dark columns missed.
10. **Short generated findings reports**
  - `experiments/.../analysis/scorecards/stage2_diagnostic/layer4_5/layer4_5_factor_gene_grounding_report.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/meeting_followup_v1/meeting_followup_findings.md` — metric/IFN/direct-gene findings before the donor sensitivity was completed.
  - `experiments/.../analysis/scorecards/stage2_diagnostic/donor_sensitivity_v1/donor_sensitivity_findings.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/deg_hvg_lowsupport_v1/deg_hvg_lowsupport_findings.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/d1a_deep_dive_20260910/d1a_deep_dive_findings.md`
  - `experiments/.../analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/patient_factor_overlap_findings.md` — D14 overlap calls.
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



## Suggested team-presentation narrative

1. **Question and pipeline:** prior gene panel → latent or direct-gene representation → regularized malignant probe.
2. **Cohort reality:** show D6A and the patient table; emphasize seven supported patients and prevalence heterogeneity.
3. **Metric contract:** prove constant-score AP =\pi; distinguish AP threshold sweep from one clinical cutoff.
4. **Patient-support diagnostic:** show D6B–D6D and state that support and biology are not identified separately.
5. **Panel/DR landscape:** show D1/D2 or D7A/D7B; method choice matters within panels.
6. **No-DR value added:** show D9A–D9D; pooled DR benefit is broad, patient-equal benefit is mixed.
7. **IFN-α case study:** show D7D, D8B, D8C, and D8E; distinguish retained PCA factor from all-zero FA folds.
8. **Donor sensitivity:** show D10E before any AP-loss plot, then D11B/D11C with the corrected factor-specific result.
9. **Claim gate:** separate current descriptive findings from publication claims requiring controls.



## Current claim boundary



### Supported as descriptive findings

- The 41 old34 panels exhibit substantial patient × DR-method × K heterogeneity.
- Among the 41 biological panels' in-sample best-median DR cells, K=40 wins 24, K=20 wins 13, and K=10 wins 4; method winners are balanced (promax 12, FA 11, PCA 9, FactoSig 9), so there is no global dominant method but no method invariance either.
- The focused selected-DR median lifts are allograft 12.56 pp, apoptosis 11.52 pp, and TNFα/NF-κB 18.06 pp; their direct-gene medians are 0.23, approximately 0, and 12.78 pp. These summaries remain patient-heterogeneous.
- Allograft and apoptosis share only 15 covered genes across a 270-gene union (Jaccard 0.056). A mean-z one-score proxy is much weaker than selected DR for allograft/TNF and intermediate for apoptosis; this is not yet an AUCell/UCell/ssGSEA comparison.
- Selected-DR cell rankings for the three focus panels reconstruct stored patient AP to within 3.2e-8. Cell-type/ADT error audits are descriptive because the rank cutoff uses held-out prevalence and detailed treatment-regimen metadata are absent.
- Constant-score AP equals patient prevalence exactly under the implemented metric.
- Lift remains positively rank-associated with prevalence/support across many representations.
- Direct-gene/no-DR Stage 2 baselines exist for all 41 panels.
- DR usually improves pooled-cell AP but not consistently patient-equal median/P20 lift.
- IFN-α PCA factor 008 is the selected driver in P02/P07 while FA is all-zero there at the chosen L1 setting.
- Donor downsampling can push sparse models to an all-zero solution.
- FA factor 007 is relatively stable at 25% donor retention; PCA factor 008 is not comparably stable until the 50% condition.
- The old34 GSEA union and HVG3,000 are HVG-enriched relative to a hypergeometric null (Jaccard 0.199; 65% of old34 is exclusive) and are not interchangeable Stage 0 panels.
- P01/P06 have within-patient DE (157/232 genes under the triage rule) but best existing held-out lift remains 6.3/5.7 AP points; P07’s EMT panel is both DE-enriched and the best existing transfer (19.0 AP points).



### Not yet supported as publication-level claims

- high lift is caused by patient-specific biology rather than support/prevalence;
- FA is intrinsically inferior to PCA for IFN-α;
- the model is invariant to donor-versus-recipient origin;
- DR universally adds value over direct genes;
- selected panels beat matched-size random gene sets;
- reported “best” configurations are unbiased deployment estimates.

Priority additions for the larger cohort are patient-level uncertainty, matched-size controls, nested/prespecified selection, inductive Stage 1 sensitivity, and a donor-downsampling analysis with condition-specific regularization selection.