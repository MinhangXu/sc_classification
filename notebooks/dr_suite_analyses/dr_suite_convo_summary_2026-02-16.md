# DR-suite comparison + Plan0/Plan1 discussion summary (2026-02-16)

This note summarizes two threads:

1) **Software-dev → analysis pipeline**: how to use a Plan 0 experiment directory as a durable cache/artifact hub for downstream analyses (Plan 1–4).
2) **Computational biology DR comparison** (from `dr_result_nov9.ipynb`): how to make DR-method diagnostics comparable; what questions each diagnostic can answer; key mathematical invariances to keep in mind.

---

## A) Experiment-dir audit + downstream plan framing

### A1) Using the experiment dir as a durable cache
- Treat `sc_classification/experiments/<plan0_experiment_id>/` as a **protocol-frozen reference artifact store**.
- Good practice: keep Plan 0 artifacts immutable; write new analyses into new subfolders (e.g. `analysis/plan2_*`, `analysis/plan3_*`) rather than overwriting Plan 0 outputs.

### A2) Snapshot observed in `20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886`
- Present:
  - `analysis/plan0/config.json`
  - `analysis/plan0/k_selection_summary.csv`
  - FA + FactoSig replicate caches and summaries under `analysis/plan0/stability/...`
  - preprocessing artifacts exist (binary): `preprocessing/adata_processed.h5ad`, `hvg_list.pkl`, `scaler.pkl`
  - optional rehydrated output exists (binary): `preprocessing/adata_processed_with_plan0_dr_k60_seed1.h5ad`
- Missing / incomplete at time of audit:
  - **cNMF diagnostics** not yet written: expected `analysis/plan0/cnmf/k_<K>/consensus_stats.json`
  - PCA replicate caches under `analysis/plan0/stability/pca/...` were absent
  - `metadata.json` can be **stale** (said `preprocessing_complete`); file tree under `analysis/plan0/...` is the real source of truth

### A3) Remaining tasks (high level)
- **Plan 0**:
  - finish cNMF and generate consensus stats per K and dt
  - decide whether to run PCA in Plan 0 K-sweep (if PCA is in the comparison set)
  - if “stability” is a key claim: rerun with **multiple seeds** (Plan 0 had `seeds=[1]` → stability is not estimable)
- **Plan 1**:
  - Plan 1.A first (no-CV fast pass), then Plan 1.B (classifier-only CV)
  - decide whether to unblock with a non-cNMF Plan 1.A run vs wait for final cNMF K/dt choice

---

## B) DR comparison: how to think about “science vs art”

### Science (measurement + falsifiability)
- Prefer diagnostics that are **invariant** to known ambiguities:
  - factor permutation
  - sign flips
  - (for some models) rotations
  - (for many factorizations) scaling reparameterizations
- Tie each diagnostic to a decision: choose K, choose method family, interpret factors, assess generalization.
- Use **negative controls** later (Plan 2) to detect leakage/confounding.

### Art (making results decision-useful)
- Put *representation* and *prediction* on the same page:
  - representation: variance/knee + stability + (normalized) communality
  - prediction: patient-level label association + coefficient stability + CV outcomes
- Maintain a small set of canonical plots per run for comparability across iterations.

---

## C) “Variance explained / knee-drop”: comparability across methods

### C1) Why PCA is naturally ordered but FA/NMF/FactoSig aren’t
- **PCA** components are ordered by eigenvalues (EVR decreases by construction).
- **FA** is not uniquely identified without additional constraints: it has **rotational invariance**.
  - If the model is \(x = \Lambda f + \epsilon\) with \(f \sim \mathcal N(0, I)\), then for any orthogonal \(R\):
    \[
    \Lambda f = (\Lambda R)\,(R^\top f)
    \]
  - Therefore “factor 1 vs factor 2” is not inherently meaningful unless you impose an ordering/rotation convention.
- **NMF / FactoSig** factor ordering is generally arbitrary (depends on initialization / optimization path).

### C2) “Chosen proxy” meaning (for FA ordering)
- Any scalar “importance per factor” (e.g. sum of squared loadings, score variance, etc.) is a **chosen diagnostic** rather than something uniquely defined by the objective the way EVR is for PCA.
- Empirically monotone FA SS-loading shares can happen due to a fixed solver/rotation convention, but it is not guaranteed by identifiability.

### C3) Preferred comparable diagnostic: score-variance share
Use the same definition across methods:
- Let scores \(Z \in \mathbb{R}^{n \times k}\) (cells × factors), centered per column.
- Define \(s_j = \mathrm{Var}(Z_{\cdot j})\).
- Plot share \(s_j / \sum_\ell s_\ell\), after sorting \(s_j\) decreasing.

Interpretation:
- A sharp knee suggests “few dominant axes in latent space.”
- Low-variance factors can still be strongly label-aligned (MRD “needle”) and show up in label overlays / corr-matrix analyses.

---

## D) Gene communality: what it asks + why comparability is tricky

### D1) Common proxy used
Given loadings \(L \in \mathbb{R}^{p \times k}\) (genes × factors):
\[
\mathrm{comm}_g = \sum_{j=1}^k L_{g j}^2
\]

### D2) Why comparability is hard
Two major issues:

1) **Scaling non-identifiability** (applies broadly to matrix factorizations):
   - If \(X \approx Z L^\top\), then for any diagonal \(D\) with positive entries:
     \[
     Z L^\top = (Z D)\,(L D^{-1})^\top
     \]
   - So the absolute magnitude of columns of \(L\) is not unique unless you enforce a normalization convention.

2) **Non-orthogonality**:
   - In PCA (orthogonal components), variance attribution is additive-like.
   - With correlated/non-orthogonal factors (common in FA after rotation, NMF-like methods), “variance explained by factor \(j\)” is not uniquely partitioned; overlap can cause “double counting.”

### D3) Example research questions communality can address
- Does a method concentrate representation on a small subset of genes (heavy-tailed communality) vs spread broadly?
- Which genes are consistently high-communality across methods (core programs) vs method-specific?
- Are high-communality genes enriched for technical covariates vs known biology?

### D4) Make communality more comparable (conceptual direction)
- Prefer a normalization or a reconstruction-based gene-wise metric (e.g., gene-wise \(R^2\)) on the **same input scale**.
- If staying with \(\sum L^2\): enforce a consistent column normalization of \(L\) before comparing distributions.

---

## E) Knowledge-driven overlays: GSEA on loadings

### E1) Why it’s comparable
- GSEA on per-factor gene rankings is naturally cross-method (once gene universe and naming are aligned).

### E2) Suggested extension
- Support **multiple GMT collections** (Hallmark, Reactome, GO, TF targets, custom signatures), with caching by output directory.

---

## F) Corr matrices (patient-level): systematic analysis ideas (for later deep dive)

Use patient-stratified analyses to connect representation structure to `CN.label` predictability:
- per-patient factor–label correlations across methods
- per-patient multicollinearity diagnostics among factors (stability of coefficients)
- cancer vs normal “rewiring”: corr matrices computed separately by class within each patient
- cluster patients by corr-structure similarity; relate clusters to performance/biology

---

## G) Math-heavy note-taking template (recommended)

For each analysis you add later (e.g. factor alignment / regression between factor spaces):
- **Definitions**: \(X\), \(Z\), \(L\), dimensions, gene universe, preprocessing scale.
- **Metric formula**: one clean equation + the exact inputs.
- **Invariances**: permutation/sign/rotation/scaling; mark whether the metric is invariant.
- **Bio question**: 1–2 sentences (“what hypothesis does this test?”).
- **Failure modes**: confounders to check (batch, library size, cell cycle, patient imbalance).

