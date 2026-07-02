# Three-Stage Knowledge-Prior Malignancy Classification

Date: 2026-06-04

This note describes the current MRD malignancy-classification design for a high-level figure or schematic. The goal is not only to build a classifier that separates malignant from non-malignant cells, but to use classification as a biological probe: which prior gene spaces contain malignant structure, which signals transfer across patients, and which programs break into distinct patient-specific regulatory axes?

## Core Idea

The analysis is organized into three stages:

1. **Stage 0: Knowledge-prior gene-space subsetting**
2. **Stage 1: Shared latent representation learning**
3. **Stage 2: Regularized malignant-vs-non-malignant supervised probes**

The old HVG-only pipeline asked whether a broad data-driven feature space could classify malignant cells. The current design asks a more interpretable question:

> If the model is only allowed to look inside a prior-defined biological gene space, how much malignant-vs-non-malignant structure can it recover, and is that structure shared across patients or patient-specific?

## Stage 0: Prior Gene-Space Subsetting

Stage 0 defines the biological hypothesis space before any dimensionality reduction or classifier is fit.

Candidate gene spaces include:

- Single curated gene sets, such as interferon signaling, IL2/STAT5 signaling, antigen presentation, TNF/NF-kB signaling, hypoxia, and cell-cycle programs.
- Biology-group panels, where related gene sets are unioned into broader prior spaces such as cytokine/JAK/STAT, inflammatory interferon, stress arrest, antigen presentation, or proliferation/metabolism.
- Controls, including full old-34 union, core-only union, and HVG anchors.

Interpretation:

- A strong result in a curated gene set means that malignant-vs-non-malignant structure is recoverable within that biological prior.
- A curated panel competing with broad HVG controls is more biologically informative than a black-box HVG-only classifier.
- Comparisons should respect gene-space scale: single gene sets, biology groups, full controls, and HVG anchors answer different kinds of questions.

## Stage 1: Shared Latent Representation Learning

Stage 1 decomposes each Stage 0 gene space into a lower-dimensional representation.

For each Stage 0 panel, the pipeline can fit methods such as:

- PCA
- factor analysis
- FactoSig
- FactoSig promax
- direct-gene fallback for very small panels

The current main analysis uses an across-patient Stage 1 basis: the representation is fit once on all eligible MRD cells so that every patient is embedded in the same latent coordinate system.

Interpretation:

- Stage 1 turns a prior gene set into candidate regulatory axes.
- A single gene set may not be one biological program. It may decompose into multiple latent factors, each reflecting a different regulatory sub-axis.
- Because the latent basis is shared across patients, Stage 2 coefficients can be compared across discovery, LOPO, and patient-specific models within the same representation.

Important caveat:

- This is a transductive shared-coordinate analysis. The unsupervised Stage 1 basis has seen all eligible cells, including cells from patients later held out in Stage 2. LOPO therefore tests label transfer on a common atlas-like basis, not a fully inductive raw-expression deployment benchmark.

## Stage 2: Regularized Supervised Probes

Stage 2 fits regularized logistic-regression probes on the Stage 1 features to separate malignant from non-malignant cells.

The same Stage 0 panel and Stage 1 representation are evaluated under three modeling goals:

| Stage 2 view | Fit scope | Primary question |
| --- | --- | --- |
| Discovery full-cohort fit | Train and score all eligible MRD cells | Does this gene space and latent representation contain malignant structure somewhere in the observed cohort? |
| Sharedness leave-one-patient-out | Train on other patients, score a held-out patient | Does the learned malignant signal transfer across patients? |
| Patient-specific model | Train and score within each patient | Does a patient use the same latent axes as the shared model, or different patient-specific axes? |

Regularization is not just a tuning detail. It is part of the biological assay.

- L1 asks whether a sparse subset of factors is sufficient.
- L2 asks whether information is distributed across many factors.
- Elastic net asks whether correlated groups of factors jointly carry signal.
- The regularization path asks how much performance is retained as the model becomes more parsimonious.

## How The Design Answers The Three PI Questions

### 1. Robust Model Across Patients

Question:

> Can we build a robust model that separates malignant from non-malignant cells across patients?

Stage 2 LOPO answers this directly. A robust shared program should have:

- High discovery apparent AUPRC.
- High aggregate LOPO AUPRC.
- Consistent per-patient LOPO performance across evaluable patients.
- Stable selected factors or coefficient signs across held-out folds.

Added biological value from the new design:

- The model is not only “a classifier works.” Stage 0 tells us which biological gene spaces support the classifier.
- If a curated immune, cytokine, interferon, antigen-presentation, stress, or metabolic panel transfers across patients, that panel becomes a candidate shared malignant program.

Suggested plots:

- Discovery apparent AUPRC vs aggregate LOPO AUPRC.
- Per-patient LOPO AUPRC heatmap, preferably normalized by malignant-prevalence baseline.
- LOPO coefficient stability heatmap across held-out patients.
- Comparison of curated panels against HVG and full-gene-space controls.

### 2. Patient-Specific Biology

Question:

> Can patient-specific models reveal new biology rather than simply repeat the shared model?

Patient-specific Stage 2 models answer this by fitting each patient on the same Stage 1 latent basis.

A patient-specific candidate should show one or more of:

- Strong held-out performance in one patient but weak aggregate/shared performance.
- High patient-specific apparent performance with selected factors that differ from the discovery/shared model.
- A small number of patient-specific factors that separate malignant from non-malignant cells within that patient.
- Biological coherence in the genes loading on those patient-specific factors.

Example interpretation:

- IL2/STAT5 signaling with FA, K=10 has low discovery apparent AUPRC but high held-out P02 AUPRC. This suggests that the IL2/STAT5 gene space is not a broadly strong cohort-wide separator, but can become highly informative for malignant-vs-non-malignant separation in P02.
- Stress arrest has high P02 performance and higher discovery apparent performance, suggesting a more cohort-visible axis with patient-dependent strength.

Suggested plots:

- Patient x panel LOPO heatmap to find patient-specific peaks.
- Patient-specific AUPRC vs aggregate LOPO AUPRC.
- Patient-specific selected-factor heatmap with discovery/shared selected factors annotated.
- For exemplar patients, factor-score distributions by malignant/non-malignant label.

### 3. Breaking Apart Programs

Question:

> Does each prior program behave as one gene program, or does it contain distinct regulatory groups?

The Stage 0 to Stage 1 to Stage 2 chain is designed to answer this.

For a selected gene set or biology group:

1. Stage 0 fixes the biological gene space.
2. Stage 1 decomposes that gene space into latent factors.
3. Stage 2 identifies which factors are predictive, shared, patient-specific, sparse, or distributed.
4. Stage 1 loadings map selected predictive factors back to genes.

Evidence that a prior program breaks into distinct regulatory groups includes:

- Multiple latent factors within the same gene set have interpretable but different loading genes.
- Different patients select different factors from the same Stage 0 gene space.
- Discovery, LOPO, and patient-specific models select partially overlapping but non-identical factors.
- Some factors are stable across held-out patients, while others are specific to one patient or subgroup.

Suggested plots:

- Factor-loading heatmap for genes within a selected Stage 0 panel.
- Stage 2 coefficient path over regularization strength for the same factors.
- Patient-specific coefficient heatmap on the shared factor basis.
- Gene-level loading bars or GSEA results for selected predictive factors.

## Suggested Figure Schematic

A high-level illustration should show the flow:

```text
MRD expression matrix + malignant/non-malignant labels
        |
        v
Stage 0: prior gene-space panels
        single gene sets | biology groups | controls | HVG anchors
        |
        v
Stage 1: shared latent basis per panel
        PCA / FA / FactoSig / FactoSig-promax / direct genes
        |
        v
Stage 2: regularized logistic probes
        discovery full-cohort | LOPO sharedness | patient-specific
        |
        v
Biological readouts
        shared programs | patient-specific programs | decomposed regulatory axes
```

The figure should emphasize that the classifier is not the endpoint. The classifier is a probe applied to prior-constrained gene spaces and shared latent factors. The outputs are both performance metrics and biological interpretability layers.

## Concise Caption Draft

Knowledge-prior three-stage malignancy classification framework. Stage 0 restricts the model to curated gene spaces or controls, turning classification into a biological hypothesis test. Stage 1 learns a shared latent basis within each gene space, allowing prior programs to decompose into candidate regulatory axes. Stage 2 fits regularized malignant-vs-non-malignant logistic probes under discovery, leave-one-patient-out sharedness, and patient-specific goals. Together, the three views distinguish broadly shared malignant programs, patient-specific biology, and distinct regulatory sub-axes within curated programs.
