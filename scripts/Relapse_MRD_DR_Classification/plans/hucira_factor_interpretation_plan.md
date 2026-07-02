# huCIRA-Guided Factor Interpretation Plan (partially implemented)

Status: The reusable huCIRA reference export utility and the signature-to-dictionary projection workflow are implemented. Cell-level scoring, direct factor annotation, pseudotime/ecological modeling, and cytokine-prior factorization remain unimplemented.

This note describes how to use `huCIRA` as a cytokine-dictionary interpretation layer for:

- `sc_classification/scripts/comprehensive_run/notebooks/plan1c_k40_full_reg_path_analysis_20260319.ipynb`
- `sc_classification/scripts/comprehensive_run/notebooks/plan1d_k40_per_patient_full_reg_path_analysis_20260319.ipynb`
- `sc_classification/scripts/Relapse_MRD_DR_Classification/notebooks/relapse_mrd_multiclass_analysis_20260319.ipynb`

The core idea is to treat huCIRA less as "another GSEA wrapper" and more as a structured reference atlas of:

- cytokine-response signatures
- cytokine-induced immune programs (CIPs)
- sender-receiver cytokine logic

That makes it useful for factor annotation, cell-state scoring, longitudinal modeling, and residual discovery.

## Why huCIRA is relevant here (implemented)

Status: This motivation now matches the current notebook implementation in `sc_classification/scripts/comprehensive_run/notebooks/plan1c_k40_full_reg_path_analysis_20260319.ipynb`, `sc_classification/scripts/comprehensive_run/notebooks/plan1d_k40_per_patient_full_reg_path_analysis_20260319.ipynb`, and `sc_classification/scripts/Relapse_MRD_DR_Classification/notebooks/relapse_mrd_multiclass_analysis_20260319.ipynb`, where huCIRA is used as a post hoc interpretation layer on top of predictive signatures reconstructed in gene space.

The current Plan1c / Plan1d notebook flow already does several strong things:

1. select predictive models across DR methods and penalties
2. reconstruct predictive signatures in gene space via `S = L @ w`
3. compare selected models through signature similarity rather than naive factor ID matching
4. optionally run Hallmark GSEA as a late interpretation layer

huCIRA can extend this by giving us a cytokine-centered interpretation layer that is closer to:

- per-cell heterogeneity
- per-factor annotation
- MRD to relapse dynamics
- ecosystem-level immune pressure

instead of only "rank genes and call pathway enrichment."

## Working assumption (partially implemented)

Status: The "dictionary provider and biological prior" part is implemented through reusable reference tables and direct signature-to-program projection in `sc_classification/src/sc_classification/utils/hucira_interpretation.py`. The "score cells, factors, or pseudobulks" part is not yet implemented; current usage is limited to predictive-signature scoring rather than cell-level or pseudobulk scoring.

For this project, huCIRA should primarily be used as a **dictionary provider and biological prior**, not necessarily as the main analysis engine.

In practice that means:

- use `huCIRA` to load the human cytokine dictionary
- extract cytokine-response and CIP gene programs
- score cells, factors, or pseudobulks against those programs
- keep huCIRA's own two-group enrichment workflow as a baseline or validation layer, not the only analysis mode

This is appropriate because the notebooks already operate on:

- factor loadings
- selected coefficient vectors
- predictive signatures
- per-cell factor scores

which are richer objects than the simple two-condition contrasts emphasized in the huCIRA README.

## External reference (implemented)

Status: The plan's reading of huCIRA as both a package and a reusable dictionary is consistent with the implemented utility layer. The current code uses huCIRA primarily as a source of cytokine/CIP reference programs rather than only as a packaged enrichment workflow.

- huCIRA README: `https://github.com/theislab/huCIRA`
- huCIRA raw README: `https://raw.githubusercontent.com/theislab/huCIRA/main/README.md`

Key useful points from the README:

- huCIRA provides a human cytokine dictionary for cytokine signaling and cytokine-induced immune program enrichment
- the package supports interpretation of transcriptomic datasets using that dictionary
- under the hood, the published package uses `gseapy`, but the dictionary itself is still valuable outside a preranked GSEA workflow

## Recommended implementation principle (partially implemented)

Status: Layer 1 has been implemented for predictive signatures, not for cells or pseudobulks. Layer 2 and Layer 3 remain planned extensions.

Use huCIRA in **three layers**, from lowest risk to most ambitious:

### Layer 1: dictionary-driven scoring (partially implemented)

Status: Implemented for predictive-signature scoring only. In all three notebooks, the signature matrices (`sig_df` or `sig_df_pp`) are projected onto huCIRA cytokine and CIP reference programs using `compute_signature_program_similarity(...)` and `compute_signature_program_jaccard(...)` from `sc_classification/src/sc_classification/utils/hucira_interpretation.py`. Per-cell and pseudobulk scoring are not yet implemented.

Use cytokine and CIP programs as gene sets for:

- per-cell scoring
- pseudobulk scoring
- predictive-signature scoring

This is the safest first step and integrates naturally with the current notebooks.

### Layer 2: factor-to-dictionary association (not yet implemented)

Status: No current utility or notebook section correlates latent factor scores directly with cytokine/CIP scores. The implemented workflow reconstructs gene-space predictive signatures from factors (`S = L @ w`) and interprets those signatures instead.

Associate latent factors with cytokine/CIP activity using:

- correlation
- sparse or ridge regression
- partial correlation / residualization
- mutual information

This turns the dictionary into an annotation basis for factors and signatures.

### Layer 3: longitudinal and ecosystem modeling (partially implemented)

Status: A limited longitudinal projection is implemented in `relapse_mrd_multiclass_analysis_20260319.ipynb`, where class-specific huCIRA profiles are compared between MRD and relapse and summarized as program deltas. Broader ecosystem modeling, pseudotime modeling, and sender-receiver analysis are not yet implemented.

After score matrices exist, model:

- MRD vs relapse shifts
- patient-specific changes
- pseudotime trends
- sender-receiver cytokine edges
- residual unexplained structure

This is the most biologically ambitious layer and should come after the scoring layer is stable.

## Proposed concrete use cases (partially implemented)

Status: Of the use cases below, the strongest implemented one is signature-to-dictionary projection for selected models. Some longitudinal summary analyses are also implemented in the multiclass notebook, but the cell-level, factor-level, and ecological use cases remain future work.

### 1. Per-cell signature scoring (not yet implemented)

Status: No current notebook writes cytokine/CIP scores into `AnnData.obs` or computes per-cell huCIRA activity by `scanpy.tl.score_genes`, AUCell, UCell, or similar methods. Current implementation operates on reconstructed gene signatures, not on individual cells.

Instead of only running enrichment on ranked genes, compute cell-level cytokine and CIP activity scores.

Candidate scoring methods:

- `scanpy.tl.score_genes`
- AUCell
- UCell
- ssGSEA-like per-cell scoring
- Seurat-style AddModuleScore analogue if needed

Recommended first implementation:

- start with a simple deterministic score such as `scanpy.tl.score_genes`
- optionally add AUCell/UCell later if rank-based robustness is needed

Questions enabled:

- which malignant subpopulations are IFN-like?
- is IL-10-like activity concentrated in a small reservoir?
- does a CIP become broader, more bimodal, or more restricted from MRD to relapse?
- are high-performing classifier regions driven by broad cytokine pressure or narrow rare-cell states?

Recommended outputs:

- `obs`-level score columns for selected cytokine/CIP programs
- per-patient violin plots or ridge plots
- malignant-only UMAP overlays
- patient x timepoint x class summary tables

### 2. Factor-to-dictionary association (not yet implemented)

Status: Not implemented in the current codebase. There is no factor x cytokine or factor x CIP correlation matrix yet. The nearest implemented object is the predictive signature obtained by multiplying DR loadings by classifier coefficients and then projecting that signature into huCIRA space.

Use the cytokine/CIP score matrix as an annotation basis for latent factors.

For each factor:

- correlate factor scores with cell-level cytokine/CIP scores
- regress factor score on multiple cytokine/CIP scores
- quantify explained variance from cytokine/CIP space
- define a residual factor component after dictionary projection

Interpretation categories:

- mostly one cytokine axis
- mixture of multiple cytokine pressures
- weakly explained by the dictionary

The third category is especially valuable because it flags candidate novel malignant biology beyond known cytokine-response structure.

Recommended outputs:

- factor x cytokine correlation matrix
- factor x CIP correlation matrix
- top-associated programs per factor
- explained-vs-residual summary per factor

### 3. Signature-to-dictionary projection for selected models (implemented)

Status: This section is now implemented in all three target notebooks and is the main delivered huCIRA workflow.

Implementation details:

1. A gene-level predictive signature matrix is built first.
   - `plan1c`: `sig_df = build_selected_signatures(...)`
   - `plan1d`: `sig_df_pp = build_selected_signatures_per_patient(...)`
   - `relapse_mrd_multiclass_analysis`: `sig_df, class_sig_meta = build_class_signature_matrix(...)`
2. Each signature is reconstructed in gene space as `S = L @ w`, where:
   - `L` is the gene-loading matrix for a DR method or patient/method pair
   - `w` is the selected classifier coefficient vector aligned to the loading columns
3. Prebuilt huCIRA reference assets are loaded from `data/hucira_reference` via `load_hucira_reference_assets(...)`.
4. A focused subset of programs is selected with `select_reference_programs(...)` using the terms in `DEFAULT_HUCIRA_PROGRAM_SUBSET`:
   - `IFN`
   - `TNF`
   - `IL1`
   - `IL10`
   - `IL15`
   - `IL32`
   - `Antigen`
   - `Myeloid`
5. The notebooks compute:
   - cosine similarity
   - Pearson correlation
   - overlap counts
   - top-200 positive-gene Jaccard overlap
6. Outputs are written to notebook-specific huCIRA directories together with heatmaps and top-match summary tables.

Current interpretation clarifications:

- The main score is cosine similarity between a predictive signature vector and a huCIRA program vector over the shared gene set. Positive values indicate aligned signed weights, values near zero indicate weak alignment, and negative values indicate opposing directions.
- Pearson is a secondary shape-based similarity check on the same shared genes.
- Jaccard is not a weighted score; it measures overlap between the signature's top positive genes and the program's top positive genes.
- The current implementation matches signatures to huCIRA prototypes directly in gene space. It is a reference-projection step, not classical preranked GSEA.
- Program selection is substring-based over `program_name`, then filtered by `min_genes_per_program`.
- The current reference build is not cell-type-specific at scoring time. `data/hucira_reference/reference_build_meta.json` shows `cytokine_aggregation_level = "cytokine"` and `cip_aggregation_level = "cip"`, so the present projection collapses cell types within each cytokine/CIP program instead of scoring `cytokine|celltype` programs separately.
- The attached "blank" heatmap behavior in the multiclass notebook is consistent with many cosine scores being close to zero; the plot is rendering, but the value range is narrow and therefore visually washed out.

This is the most direct extension of the current Plan1c / Plan1d notebooks.

Current notebook behavior:

- build predictive signatures in gene space using `S = L @ w`
- compare selected signatures across DR methods and strategies
- optionally run Hallmark GSEA on those signatures

New huCIRA-based extension:

- compare each selected predictive signature to cytokine/CIP reference programs directly
- compute cosine similarity, Pearson correlation, and top-gene overlap between:
  - selected predictive signatures
  - huCIRA cytokine signatures
  - huCIRA CIPs

This is not classic GSEA. It is a reference-projection step in the same gene space the notebooks already use.

Questions enabled:

- is the selected pooled predictor mostly IFN-related, IL-10-related, or mixed?
- does the per-patient predictor in one patient align with the same cytokine axis as another patient?
- do the recall-first and most-parsimonious models point to the same cytokine program?

Recommended outputs:

- selected-model x cytokine similarity matrix
- selected-model x CIP similarity matrix
- clustered heatmaps
- top-matching reference table per selected model

### 4. Pseudotime / branch dynamics of cytokine activity (not yet implemented)

Status: Not implemented. Current huCIRA usage does not produce per-cell score matrices, pseudotime trajectories, or branch-aware smoothers, so this section cannot be executed without first adding cell-level scoring infrastructure.

After per-cell scores are computed, model score trends along:

- pseudotime
- lineage branch probability
- MRD to relapse continuum
- clone-weighted evolutionary trajectories when available

Candidate models:

- GAMs
- spline regressions
- branch-specific smoothers

Questions enabled:

- does IFN-like activity rise early or late?
- does antigen-presentation-related CIP collapse while inflammatory programs persist?
- are some cytokine programs branch-specific rather than globally changing?

This is closer to disease-evolution analysis than static enrichment.

### 5. Patient-level mixed modeling (not yet implemented)

Status: Not implemented. There are no patient-level mixed models fitted on cytokine/CIP score summaries yet because those score summaries are not currently generated.

Aggregate cytokine/CIP scores at pseudobulk level by:

- patient x timepoint x malignant-status
- patient x timepoint x cell compartment
- patient x timepoint x clone, if clone labels are available later

Then fit:

- `score ~ timepoint + (1 | patient)`
- or expanded models with compartment and interaction terms

Questions enabled:

- which cytokine programs change consistently across patients?
- which programs are highly patient-specific?
- which changes are reproducible but moderate vs dramatic but private?

This is statistically better aligned with longitudinal paired data than pooled enrichment alone.

### 6. Sender-receiver cytokine network reconstruction (infeasible)

Status: Infeasible with the current implementation. The present huCIRA utilities export gene programs and score predictive signatures, but they do not construct ligand-receptor edges, sender populations, receptor expression models, or cross-compartment communication tables. This would require new data products and additional modeling code outside the current huCIRA utility layer.

This is one of the strongest huCIRA-inspired steps.

For each patient and timepoint:

- define sender populations expressing cytokine ligands
- define receiver malignant populations expressing relevant receptors
- annotate receiver state by cytokine/CIP score

Then test whether edges strengthen or weaken from MRD to relapse.

Questions enabled:

- does NK/T-derived IFN pressure rise at relapse?
- do myeloid-derived inflammatory edges strengthen in refractory patients?
- is suppressive IL-10-like context lost or compartmentalized?

This provides ecosystem-level mechanistic interpretation rather than only factor annotation.

### 7. Residual analysis after dictionary projection (not yet implemented)

Status: Not implemented. The current notebooks compute similarity in huCIRA space and summarize top matches, but they do not regress expression or factors onto cytokine/CIP space and study the residual unexplained component.

Project factors or expression profiles onto cytokine/CIP space, then study the residual.

Decompose structure into:

- explained by cytokine scores
- explained by CIPs
- residual unexplained structure

Residual structure may capture:

- clone identity
- quiescent reservoir programs
- lineage priming
- cell cycle
- stress adaptation not represented in PBMC cytokine perturbations

This is a strong framework for quantifying known immune pressure versus novel malignant adaptation.

### 8. Cytokine simplex / cytokine-state embedding (not yet implemented)

Status: Not implemented. Although the notebooks now produce signature x program score matrices, there is no downstream embedding of those score vectors into a cytokine-state coordinate system.

Treat each cell or pseudobulk as a vector of cytokine/CIP scores and embed that vector space directly.

Then:

- visualize MRD vs relapse occupancy in cytokine-state space
- plot patient trajectories through that space
- compare malignant-only and non-malignant-only trajectories

This converts the dictionary into a biologically interpretable coordinate system.

### 9. Semi-supervised factorization with cytokine priors (infeasible)

Status: Infeasible under the current implementation scope. The existing codebase uses standard DR outputs plus downstream interpretation, and there is no guided factorization framework that can accept cytokine priors during model fitting.

Longer-term method development:

- guided NMF
- sparse coding with cytokine priors
- semi-supervised VAE
- dictionary-constrained factorization

This would let some axes be cytokine-guided and others remain free to discover malignant-specific structure.

This is a later-stage modeling project, not the first notebook extension.

### 10. Prototype matching instead of enrichment (implemented)

Status: Implemented for selected predictive signatures. This is exactly what the current huCIRA notebook sections do: they compute direct similarity between reconstructed signatures and cytokine/CIP prototypes instead of relying only on enrichment analysis. This is not yet implemented for cells or pseudobulks.

For each cell state, pseudobulk, or selected predictive signature:

- compute similarity to cytokine perturbation prototypes
- infer which cytokine environment best explains the observed state

This turns the dictionary into a reference atlas for inverse mapping.

## Recommended phased plan (partially implemented)

Status: Phase A and Phase C are implemented in a reusable first pass. Phases B, D, and E remain unimplemented.

## Phase A: build huCIRA-derived reference assets (implemented)

Status: Implemented in `sc_classification/scripts/Relapse_MRD_DR_Classification/build_hucira_reference_sets.py` and `sc_classification/src/sc_classification/utils/hucira_interpretation.py`.

How and where:

- `export_hucira_reference_sets(...)` builds or reloads reference assets.
- `build_cytokine_reference_long(...)` and `build_cip_reference_long(...)` normalize gene symbols, aggregate rows into programs, compute per-gene weights, and enforce a minimum program size.
- The utility exports:
  - `reference_table.csv`
  - `cytokine_reference_long.csv`
  - `cip_reference_long.csv`
  - up/down JSON gene-set mappings
  - optional GMT files
  - `reference_build_meta.json`
- Actual output location in the current implementation is `data/hucira_reference/`, not `analysis/hucira_reference/`.

Important current-state clarification:

- The utility supports both coarse and cell-type-specific aggregation:
  - cytokines: `cytokine` or `cytokine_celltype`
  - CIPs: `cip` or `cip_celltype`
- The currently materialized reference assets were built with the coarse settings `cytokine` and `cip`, so the notebooks are not yet performing cell-type-specific matching even though the utility can support it.

Goal:

- load the human cytokine dictionary once
- export clean gene-set objects for:
  - cytokine-response signatures
  - CIPs

Implementation sketch:

1. Create a small utility script, for example:
   - `scripts/Relapse_MRD_DR_Classification/build_hucira_reference_sets.py`
2. Use `hucira.load_human_cytokine_dict(...)`
3. Normalize gene symbols to the cohort gene naming convention
4. Save:
   - long-form reference table
   - `dict[str, list[str]]` style gene-set mappings
   - optional GMT exports for compatibility

Expected outputs:

- `analysis/hucira_reference/reference_table.csv`
- `analysis/hucira_reference/cytokine_gene_sets.json`
- `analysis/hucira_reference/cip_gene_sets.json`
- optional `*.gmt`

## Phase B: score cells and pseudobulks (not yet implemented)

Status: Not implemented. No current notebook or utility computes cytokine/CIP scores for cells, pseudobulks, or patient-compartment summaries from `AnnData` objects.

Goal:

- generate cytokine/CIP score matrices on your own data

Implementation sketch:

1. For each relevant `AnnData` object:
   - pooled MRD data used in Plan1c
   - per-patient MRD data used in Plan1d
   - MRD + relapse patient-level DR inputs
2. Score cells for selected cytokine/CIP programs
3. Aggregate scores by:
   - patient
   - timepoint
   - class
   - malignant-only compartment

Expected outputs:

- `obs` columns or external score matrix parquet/csv
- summary tables for patient x timepoint x program

## Phase C: integrate with Plan1c / Plan1d notebooks (implemented)

Status: Implemented in the newer March 19 notebooks:

- `sc_classification/scripts/comprehensive_run/notebooks/plan1c_k40_full_reg_path_analysis_20260319.ipynb`
- `sc_classification/scripts/comprehensive_run/notebooks/plan1d_k40_per_patient_full_reg_path_analysis_20260319.ipynb`
- and extended analogously in `sc_classification/scripts/Relapse_MRD_DR_Classification/notebooks/relapse_mrd_multiclass_analysis_20260319.ipynb`

How and where:

1. The notebooks first build selected predictive signatures in gene space.
2. They load huCIRA assets from `data/hucira_reference`.
3. They subset the reference to a focused program panel.
4. They compute signature-to-program similarity tables for cytokines and CIPs.
5. They save:
   - cosine similarity matrices
   - Pearson similarity matrices
   - overlap-count matrices
   - top positive-gene Jaccard matrices
   - top-match summary tables
6. They render heatmaps for quick inspection.

Current output directories:

- pooled: `predictive_signature_hucira_similarity__{SELECTION_MODE}`
- per-patient: `per_patient_hucira_similarity__{SELECTION_MODE}`
- multiclass: `class_signatures/hucira_projection`

Goal:

- extend the current notebooks after signature construction

Natural insertion points:

- `plan1c_k40_regularization_path_analysis_20260305.ipynb`
  - after `build_selected_signatures(...)`
  - parallel to the optional Hallmark GSEA section
- `plan1d_k40_per_patient_regularization_path_analysis_20260306.ipynb`
  - after `build_selected_signatures_per_patient(...)`
  - before or instead of optional per-patient GSEA exports

Implementation sketch:

1. Build selected predictive signatures as already done
2. Load huCIRA reference programs
3. Compute signature-to-program similarity matrices
4. Save:
   - selected-model x cytokine similarity tables
   - selected-model x CIP similarity tables
   - heatmaps and ranked-match summaries

Expected outputs:

- pooled:
  - `predictive_signature_hucira_similarity/`
- per-patient:
  - `per_patient_hucira_similarity/`

## Phase D: factor-level annotation (not yet implemented)

Status: Not implemented. The notebooks do not currently compute factor-score x program relationships, factor residuals, or multi-program regression directly on latent factors.

Goal:

- annotate latent factors directly rather than only selected predictive models

Implementation sketch:

1. For each DR method:
   - obtain cell-level factor score matrix
2. Compute cell-level cytokine/CIP score matrix
3. Correlate factor scores with cytokine/CIP scores
4. Optionally run multi-program regression per factor

Expected outputs:

- factor x program heatmaps
- residual explained-variance tables

## Phase E: longitudinal and ecological analyses (partially implemented)

Status: A narrow longitudinal interpretation is implemented in the multiclass notebook through progression deltas in huCIRA space and cross-patient convergence summaries. Mixed models, pseudotime analyses, and ecological sender-receiver modeling remain unimplemented.

Goal:

- move from interpretation to disease-evolution and ecosystem modeling

Implementation sketch:

1. fit patient-level mixed models
2. model pseudotime trends
3. build sender-receiver cytokine edge tables
4. quantify residual unexplained disease structure

This phase should only begin after the scoring layer is stable and reproducible.

## Minimal first implementation to prioritize now (partially implemented)

Status: Items 1 and 3 are implemented. Items 2 and 4 are not yet implemented.

Implemented now:

1. export huCIRA cytokine and CIP gene sets
3. project selected predictive signatures onto those programs

Still pending:

2. score cells for a focused subset of programs
4. build factor-to-program correlation tables

Do not start with all ten ideas at once.

The highest-value and lowest-risk first pass is:

1. export huCIRA cytokine and CIP gene sets
2. score cells for a focused subset of programs
3. project selected predictive signatures onto those programs
4. build factor-to-program correlation tables

This gives a meaningful interpretation layer beyond Hallmark GSEA while staying close to the existing notebook design.

## Suggested initial program subset (implemented)

Status: Implemented as `DEFAULT_HUCIRA_PROGRAM_SUBSET` in `sc_classification/src/sc_classification/utils/hucira_interpretation.py` and used by all three notebooks via `HUCIRA_FOCUS_TERMS`.

Start with a compact panel instead of the full dictionary.

Candidate first-pass axes:

- type I IFN-like
- IFN-gamma-like
- TNF / IL1 inflammatory
- IL10-like suppressive
- IL15-like cytotoxic support
- IL32-beta-like inflammatory / rewiring axis
- antigen-presentation-related CIPs
- myeloid inflammatory CIPs

The subset can be expanded after checking signal quality and score stability.

## Practical implementation notes (partially implemented)

Status: The current implementation follows the "annotation layer, not proof of mechanism" principle. The malignant/non-malignant split and repeated program scoring across compartments are not yet implemented because per-cell scoring is not yet present.

Additional clarification from the current code:

- The current projection scores predictive signatures against huCIRA reference vectors after intersecting genes shared by the signature matrix and the reference matrix.
- `overlap_n` is computed alongside similarity scores and should be checked when interpreting strong or weak matches.
- Since the current reference build collapses cell types, a high cytokine match should be interpreted as "consistent with this cytokine-associated program family" rather than as direct evidence for a specific sender or receiver cell type.

- huCIRA was developed on immune perturbation data, especially PBMC-like contexts, so transfer to malignant MDS states should be treated as informative but not literal.
- Use dictionary projection as an annotation layer, not as proof of mechanism.
- For malignant cells, receptor expression and sender-receiver context will matter as much as response-score projection.
- Program scoring should be repeated in:
  - all cells
  - malignant-only cells
  - non-malignant immune cells

That separation helps avoid conflating ecosystem shifts with malignant-state shifts.

## Validation strategy (partially implemented)

Status: The notebooks already support some of these checks by comparing pooled versus per-patient results, by comparing methods/downsampling choices, and by saving ranked top-match summaries and overlap tables. Stability across different per-cell scoring methods, ligand-receptor plausibility checks, and residual-explained fractions are not yet implemented.

At every stage, ask:

- are top cytokine matches stable across scoring methods?
- are matches stable across downsampling choices?
- do pooled and per-patient analyses tell a similar story?
- are high-scoring cytokine states biologically plausible given ligand/receptor context?
- what fraction of predictive structure remains unexplained after dictionary projection?

If the residual remains large, that is not failure. It is often exactly where the novel biology is.

## Deliverables (partially implemented)

Status: Near-term deliverables are only partly completed.

Completed deliverables:

- a reusable huCIRA reference export utility
- pooled selected-model x huCIRA similarity tables
- per-patient selected-model x huCIRA similarity tables
- multiclass class-signature x huCIRA similarity tables plus progression/convergence summaries

Pending deliverables:

- per-cell cytokine/CIP score matrices
- factor x cytokine/CIP association heatmaps
- patient-level mixed-model summaries
- pseudotime cytokine-dynamics figures
- sender-receiver cytokine edge tables
- residual-vs-explained decomposition analyses

Near-term deliverables:

- a reusable huCIRA reference export utility
- per-cell cytokine/CIP score matrices
- pooled selected-model x huCIRA similarity tables
- per-patient selected-model x huCIRA similarity tables
- factor x cytokine/CIP association heatmaps

Longer-term deliverables:

- patient-level mixed-model summaries
- pseudotime cytokine-dynamics figures
- sender-receiver cytokine edge tables
- residual-vs-explained decomposition analyses

## Bottom line (implemented)

Status: This summary now matches the current state of the codebase well. huCIRA is presently being used as a biologically informed dictionary/reference atlas layered on top of predictive signatures, not as a replacement for the existing latent-factor workflow and not primarily as another GSEA wrapper.

huCIRA is still useful here even though its packaged workflow is enrichment-centered.

The strongest use for this project is not "run another GSEA," but:

- use the cytokine dictionary as a scoring basis
- use it as a reference atlas for predictive signatures
- use it as an annotation basis for latent factors
- use it to separate known immune-pressure structure from unexplained malignant adaptation

That makes huCIRA a biologically informed layer on top of the current latent-factor and predictive-signature workflow, rather than a replacement for it.
