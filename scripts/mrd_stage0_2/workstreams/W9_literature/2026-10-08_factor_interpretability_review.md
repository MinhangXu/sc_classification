---
title: Literature review A — how other fields interpret factors, applied to the old34 Stage 0-2 factor model and sparse probes
date: 2026-10-08
produced_by: agent web review (claude-code session, 2026-10-08); citations checked by the agent at write time; readers should spot-check
scope: psychometric rotation, pattern vs structure, congruence and factor-score indeterminacy; replication of factors across groups; sparse linear probes on correlated or rotated features; attribution vs causation and prior-guided factorization
routed_tasks: [W3.2, W3.1, W4.6, W6.2, W5.4, W5.1, W4.5, W5.2, W5.5, W2.3, W1.3, W3.6, W3.3, W5.6]
---

> **Provenance header (added when this review was homed; not part of the review).**
> - Date: 2026-10-08. Homed by W9.1; the matching devlog entry is `2026-10-08_literature-reviews`. Ask: A-20261008-22.
> - How it was produced: an agent-run web literature review in a claude-code session on 2026-10-08, plus read-only checks against this repo and the canonical experiment. At write time the agent resolved each DOI or URL and read the abstract or full text where the review says so. Readers should spot-check a citation before relying on it, especially those marked *(content not verified here)*.
> - Numbers computed by the reviewer: the "New checks run for this review" table in §0 and the fold table in §3.1 come from scratch scripts in the reviewing agent's temporary scratchpad, which were not kept. Treat them as leads, not registry evidence, until the routed task reproduces them: W3.1 and W3.2 for the L1 entry margins and the knife-edge refit; W5.1 and W5.4 for the promax pattern-vs-structure check, determinacy and effective_k.
> - Edits to the body: two absolute local paths in §0 were rewritten as repo-relative paths (the sibling factosig repo sits at `../factosig` from this repo's root). Nothing else was changed.
> - Labels: the synthesis IDs A1–A12 in §5 are local to this file. They are not ask IDs (A-YYYYMMDD-NN) or repository-audit IDs. Every recommended action is routed to a W#.n task in `README.md` in this directory.

# Literature Review A: Ways other fields interpret factors, applied to the old34 Stage 0–2 factor model and sparse probes

**Date:** 2026-10-08
**Scope.** This review covers four areas:
1. Psychometrics: rotation, pattern vs structure matrices, the congruence coefficient, factor-score indeterminacy, simple structure, and the EFA vs PCA debate.
2. Whether factors replicate across groups: measurement invariance, multi-group and multi-study factor analysis, and consensus or stability methods.
3. How to interpret sparse linear probes on correlated or rotated features: lasso instability, stability selection, elastic net, and post-selection inference.
4. Attribution vs causation, and factor models guided by prior knowledge.

**How citations were checked.** Every DOI below was resolved against Crossref on 2026-10-08. Abstracts or full texts were checked through Europe PMC, PMC, arXiv, JMLR or the publisher page where one was available. When I could confirm a source only at the bibliographic level, the text marks it *(content not verified here)*.

**PI questions.** Q1 = is there a robust shared (LOPO-transferable) malignancy model? Q2 = do patient-specific models find biology the shared model misses? Q3 = does a prior program break into distinct sub-axes? (Sources: `plans/experiment_20260525_stage0_2_rehydration_dossier.md` lines 274–276, and the tracker canvas.)

---

## 0. Repo facts used, plus new read-only checks run for this review

**Facts given in the task:**
- IFN-α PCA factor 8 (PC8) correlates −0.685 with FA factor 7 (FA7) and +0.580 with FA factor 8 (FA8).
- The IFN-α PCA40 vs FA40 gap is a Stage 2 effect of L1 at a single fixed C. FA7 alone gives 18.06 pp lift in P02. L2 on FA recovers P02 to 16.14 pp and P07 to 5.07 pp.
- scikit-learn FA scores are uncorrelated, but its loadings are not orthogonal.
- factosig_promax is oblique (maximum |score r| = 0.924). It was not known whether `loadings.npy` stores the pattern or the structure matrix.
- Factor signs are arbitrary.

**Code facts I read.** All files were read only.
- **sklearn FA wrapper** (`src/sc_classification/dimension_reduction/factor_analysis.py`): no rotation is applied, and `FA_loadings = components_.T`.
- **How sklearn 1.7.2 fits FA:**
  - It sets `W = sqrt(max(s − 1, 0)) · Vt` from the SVD of Ψ^{-1/2}-scaled data, and sets `ψ = max(var − ΣW², SMALL)`.
  - So Λᵀ Ψ⁻¹ Λ is diagonal (the "canonical" identification convention), not ΛᵀΛ.
  - `transform()` returns posterior means.
- **FactoSig promax** (`../factosig/factosig/core/rotation.py:69–128`; `core/fa_torch.py:220–253`, submodule commit `84f2a7c`):
  - It applies varimax, builds the target T = sign(L_v)·|L_v|⁴, solves least squares for P, and sets Λ′ = Λ·M.
  - Scores become Z′ = Z·M^{−T}, so Z′Λ′ᵀ = ZΛᵀ. These are **pattern semantics**.
  - Unlike R `stats::promax`, it does **not** rescale the transform so that the factor correlation matrix has a unit diagonal. R does this with `d <- diag(solve(t(U) %*% U)); U <- U %*% diag(sqrt(d))`.
  - Its varimax step has no Kaiser row normalization (R's default is `normalize = TRUE`).
  - Signs are set so that each column of loadings sums to a positive number. Factors are ordered by sum of squared loadings, computed on these unnormalized columns.

**New checks run for this review (2026-10-08).**

*Inputs (read only):*
- Factor scores and loadings: `stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method={pca,fa,factosig_promax}/k=40/seed=42/{scores,loadings}.npy`
- Z-scored gene matrix (60,118 × 92): `stage1_direct_gene/old34_direct_gene_all_v1/<same panel>/transform=zscore/features.npz`
- Patient and label (obs only): `analysis/cache/stage2_diagnostic/deg_hvg_lowsupport_v1/mrd_cite_binary_log1p_15560.h5ad`

*LOPO setup:* train on all non-held-out cells, StandardScaler, liblinear L1, `class_weight="balanced"`. This copies `scripts/mrd_stage0_2/stage2_supervised/cell_count_learning_curve.py::_fit_auprc`.

*The scripts lived in the session scratchpad, which is temporary.* Each computation is described below in enough detail to rebuild it as a notebook cell.

| Check | Result (IFN-α panel, 92 genes, K=40) |
|---|---|
| FA40 effective rank | FA factors 37–40 have **all-zero loadings and zero-variance scores**. sklearn zeroes a component when its Ψ-whitened eigenvalue is ≤ 1, so the panel supports at most **36** common factors under the FA noise model. |
| FA scores vs loadings | Largest correlation between FA score columns: 0.003. Largest cosine between FA loading columns: 0.115. |
| FA factor-score determinacy ρ_k = √(d_k/(1+d_k)), with d_k = Σ_g λ²_gk/ψ_g | FA5 0.788, FA6 0.709, **FA7 0.648**, **FA8 0.618**, FA9 0.545, FA10 0.477. The empirical score variances equal ρ² to three decimals (posterior-mean shrinkage). |
| How much of PC8 lies in FA space | R²(PC8 ~ FA7 + FA8) = **0.806**. R²(PC8 ~ all 36 nonzero FA factors) = **0.920**. Tucker φ of the loadings: PC8 vs FA7 = −0.691, PC8 vs FA8 = +0.577. |
| PCA eigengaps around PC8 | Score variances of PC7, PC8 and PC9 are 1.357, 1.286 and 1.158. The relative gaps on either side of PC8 are **5.5%** and **10%**. |
| PCA40 vs FA40 score subspaces | Smallest canonical correlation 0.047. **21 of 36** canonical correlations are below 0.9, so the two K=40 bases do **not** span the same subspace. |
| factosig_promax `loadings.npy`: pattern or structure? | **Pattern.** cov(gene, score) = L·Φ fits with relative residual 2.4×10⁻⁴, and the fitted Φ is symmetric (asymmetry 4×10⁻⁵). Pattern and structure columns disagree: cosine median 0.52 (min 0.09), and the top-10 gene Jaccard between ranking by |pattern| and by |corr(gene, score)| has median **0.18** (min 0.05). The model's latent factor correlation peaks at 0.80, while the posterior-mean score correlation peaks at 0.924 (factors 33 and 36). 136 of 780 score pairs have \|r\| > 0.5. Column scale is arbitrary: loading column norms run 0.46 → 0.03 while score variances run 4.7 → 271. |
| L1 entry condition (KKT) at C = 10⁻⁴ | A factor's coefficient leaves zero when \|Σ_i s_i(y_i − p̂)z_ij\| > 1/C = 10,000. Details in §3.1. |
| Refit at C = 1.05×10⁻⁴ (a 5% change) | FA40 L1 now selects **FA7 in P02 (lift 18.06 pp) and P07 (6.97 pp)**. PCA40 is unchanged (P02 15.94 pp, P07 11.08 pp). |

---

## 1. Psychometrics: rotation, oblique solutions, congruence, indeterminacy, EFA vs PCA

### 1.1 Rotational indeterminacy: what "factor 7" means

**The literature.**
- A factor solution can be multiplied by any invertible T: ΛF = (ΛT)(T⁻¹F). Unrotated maximum-likelihood solutions are pinned down only by a convention (Λᵀ Ψ⁻¹ Λ diagonal), and that convention has no substantive meaning.
- Thurstone's **simple structure** asks each variable to load on few factors (Thurstone, 1947, *Multiple-Factor Analysis*, University of Chicago Press; https://wellcomecollection.org/works/jhqr9a62, catalog record).
- Analytic rotations put that idea into practice:
  - Varimax (Kaiser, 1958, *Psychometrika* 23:187–200; https://doi.org/10.1007/BF02289233)
  - Direct oblimin (Jennrich & Sampson, 1966, *Psychometrika* 31:313–323; https://doi.org/10.1007/BF02289465)
  - The general gradient-projection framework (Bernaards & Jennrich, 2005, *Educational and Psychological Measurement* 65:676–696; https://doi.org/10.1177/0013164404272507)
  - Browne (2001, *Multivariate Behavioral Research* 36:111–150; https://doi.org/10.1207/S15327906MBR3601_05) reviews these criteria and shows where they fail for complex patterns.
- Rohe & Zeng (2023, *JRSS-B* 85:1037–1060; https://doi.org/10.1093/jrsssb/qkad029) argue that **varimax "performs statistical inference."** When the factors are leptokurtic (heavy-tailed or sparse), PCA followed by varimax gives a consistent estimator. Rotation is then identifiable from the data rather than cosmetic.
- When eigenvalues are nearly equal, the individual eigenvectors are unstable. Perturbation rotates them at a rate of roughly (perturbation size)/(eigengap). See Davis & Kahan (1970, *SIAM J. Numer. Anal.* 7:1–46; https://doi.org/10.1137/0707001) and the statistician-oriented version by Yu, Wang & Samworth (2015, *Biometrika* 102:315–323; https://doi.org/10.1093/biomet/asv008).

**Implication for this repo**
- **Q3:** Do not read the PC8 → FA7/FA8 split as "IFN-α breaks into two sub-axes." The evidence points to two different orientation conventions inside a nearly degenerate block:
  - PCA orders components by total variance under isotropic noise. sklearn FA orders them by Ψ-whitened signal-to-noise (d₇ = 0.72, d₈ = 0.62).
  - PC7–PC9 are separated by only 5.5% and 10%.
  - FA7 and FA8 together explain 81% of PC8's variance.
- **Q3, a rotation-invariant test:** restrict to the plane span(FA7, FA8), or span(PC7–PC9). In that plane, fit a per-patient L2 logistic direction and compare the angles across the supported patients with a patient-level bootstrap.
  - Same angle: one axis, and the split is a basis artifact.
  - Different angles: patient-specific sub-axes, which is evidence for Q2/Q3.
- Add varimax-rotated PCA40 and FA36 (oblimin optional) as Stage 1 variants. Check the kurtosis of the IFN scores first, because Rohe & Zeng's identifiability argument needs leptokurtic factors.
- Report eigengaps, or Ψ-whitened eigengaps for FA, next to every factor named as a "driver." A gap of 10% or less should block naming that factor individually.

### 1.2 Oblique solutions: pattern vs structure

**The literature.**
- Promax is a fast oblique rotation: varimax, then a power target, then least squares (Hendrickson & White, 1964, *Br. J. Stat. Psychol.* 17:65–70; https://doi.org/10.1111/j.2044-8317.1964.tb00244.x).
- When factors are correlated (correlation matrix Φ):
  - The **pattern** matrix P holds partial regression weights of each variable on the factors.
  - The **structure** matrix S = PΦ holds the zero-order variable–factor correlations.
  - Reading only one of them can mislead (Graham, Guthrie & Thompson, 2003, *Structural Equation Modeling* 10:142–153; https://doi.org/10.1207/S15328007SEM1001_7).
- The same point applies to regression weights vs **structure coefficients** (Courville & Thompson, 2001, *Educational and Psychological Measurement* 61:229–248, "β is not enough"; https://doi.org/10.1177/0013164401612006; Kraha et al., 2012, *Frontiers in Psychology* 3:44; https://doi.org/10.3389/fpsyg.2012.00044).
- R's reference `stats::promax` rescales the transform so that the factors have unit variance (source: https://github.com/wch/r-source/blob/trunk/src/library/stats/R/factanal.R).

**Implication for this repo**
- **Open question resolved.** `factosig_promax/loadings.npy` holds the pattern matrix. The code shows this (Z′Λ′ᵀ = ZΛᵀ), and the empirical check confirms it: cov(gene, score) = L·Φ to a relative residual of 2.4×10⁻⁴. The columns also carry an **arbitrary scale** because the transform is never normalized. Three consequences:
  - Loading magnitudes cannot be compared across factors.
  - The `factor_001…` ordering by sum of squared loadings depends on that arbitrary scale.
  - `top_loading_genes.csv` ranks genes by pattern only.
- **Q3 gene grounding (Layer 5, still to build):** for promax, pattern and structure give different top genes (median top-10 Jaccard 0.18). For every oblique factor, report both, together with Φ. D14 and D15 use `factosig_promax` at K=40 for the apoptosis and allograft panels, so their gene labels need the same treatment.
- **Fix:** scale each promax column so the latent factor variance is 1 (scores divided by their SD, loadings multiplied by it). Save M or Φ in `metadata.json`, and add `loading_semantics: pattern` there. Order factors by a scale-free quantity such as variance explained in the unit-variance parameterization. These are factosig changes; the repo is read-only for me.
- **Stage 2 (Q1):** promax latent correlations reach 0.80 and score correlations reach 0.92. A Stage 2 β on promax scores is therefore a *partial* effect. Report the probe's structure coefficients, corr(z_j, probe score), next to the β values.

### 1.3 Congruence across solutions

**The literature.**
- Tucker's congruence coefficient φ is an uncentered cosine between loading vectors.
- Lorenzo-Seva & ten Berge (2006, *Methodology* 2:57–64; https://doi.org/10.1027/1614-2241.2.2.57) conclude: "a value in the range .85–.94 corresponds to a fair similarity, while a value higher than .95 implies that the two factors… can be considered equal."
- Lovik, Nassiri, Verbeke & Molenberghs (2020, *Methodology* 16:59–74; https://doi.org/10.5964/meth.2813) propose a modified coefficient for matching factors across many solutions.
- Orthogonal Procrustes is the usual step to align one solution to another before computing congruence (Schönemann, 1966, *Psychometrika* 31:1–10; https://doi.org/10.1007/BF02289451).
- Subspaces of different bases are compared through principal angles, or canonical correlations (Björck & Golub, 1973, *Math. Comp.* 27:579–594; https://doi.org/10.1090/S0025-5718-1973-0348991-3).
- The repo's one-to-one matching uses the Hungarian method (Kuhn, 1955, *Naval Res. Logist. Q.* 2:83–97; https://doi.org/10.1002/nav.3800020109).
- For sign ambiguity, see Bro, Acar & Kolda (2008, *J. Chemometrics* 22:135–140; https://doi.org/10.1002/cem.1122).

**Implication for this repo**
- The "loading cosine" used in the donor-downsampling stability analysis (guide §5) *is* Tucker φ. Adopt the .85/.95 bands, and orient signs before computing it.
- One-to-one Hungarian matching is ill-posed when one factor maps onto two. PC8's φ is −0.69 with FA7 and 0.58 with FA8, below 0.85 for both, so neither is "congruent." Replace D8E-style one-to-one tables with:
  - many-to-one R² (each factor regressed on the other basis);
  - principal angles between the full subspaces (here 21 of 36 canonical correlations are below 0.9);
  - principal angles between the subspaces spanned by each probe's selected factors.
- For seed, downsampling and per-patient comparisons, report φ **before and after Procrustes alignment**. The difference is the purely rotational part of the apparent instability.
- The repo matches by score correlation, while psychometric congruence uses loadings. For FA posterior-mean scores the two differ (Ψ weighting and shrinkage), so report both.

### 1.4 Factor-score indeterminacy

**The literature.**
- Common-factor scores are not determined by the data (Guttman, 1955, *Br. J. Stat. Psychol.* 8:65–81; https://doi.org/10.1111/j.2044-8317.1955.tb00321.x; history in Steiger, 1979, *Psychometrika* 44:157–167; https://doi.org/10.1007/BF02293967).
- Determinacy is measured by ρ, the multiple correlation between a factor and its best linear predictor. That predictor is the regression (Thomson) score, which equals sklearn's posterior mean.
- Two equally valid sets of scores can correlate as little as 2ρ² − 1, so they may be uncorrelated or even negatively correlated when ρ < 0.707.
- Grice (2001, *Psychological Methods* 6:430–450; https://doi.org/10.1037/1082-989X.6.4.430) recommends always reporting determinacy and validity coefficients.

**Implication for this repo**
- FA7 has ρ = 0.648 (2ρ²−1 = −0.16) and FA8 has ρ = 0.618 (−0.24). Both are **poorly determined** latent variables. Using their posterior-mean scores for prediction is fine. Claims about "the FA7 program" (Q3) should carry this caveat.
- PCA scores are exactly determined linear combinations of the data. That alone can make PCA look more stable in Stage 2, and it does not show that PCA captures better biology (Velicer & Jackson, §1.5).
- Add ρ_k to `diagnostics.json` for fa, factosig and factosig_promax. For sklearn FA on z-scored genes, ψ_g = 1 − h²_g, so ρ_k can be computed from `loadings.npy` alone. Flag ρ < 0.707.
- d_k grows with the number of genes loading on a factor. Larger panels and smaller K (relative to p) give more determinate factors, which matters when interpretation is the goal (Q3).

### 1.5 EFA vs PCA, and extracting too many factors

**The literature.**
- Fabrigar, Wegener, MacCallum & Strahan (1999, *Psychological Methods* 4:272–299; https://doi.org/10.1037/1082-989X.4.3.272) recommend common-factor EFA when the goal is latent constructs. They also show that choices of extraction, rotation and number of factors change the conclusions.
- Costello & Osborne (2005, *PARE* 10(7); https://doi.org/10.7275/jyj1-4868) give practical guidance and show that EFA solutions replicate poorly even with large samples.
- Velicer & Jackson (1990, *MBR* 25:1–28; https://doi.org/10.1207/s15327906mbr2501_1) and Widaman (1993, *MBR* 28:263–311; https://doi.org/10.1207/s15327906mbr2803_1) debate component vs factor analysis. Widaman shows that PCA inflates loadings when communalities are low.
- Schneeweiss & Mathes (1995, *J. Multivariate Anal.* 55:105–124; https://doi.org/10.1006/jmva.1995.1069) give conditions under which components and factors come close.
- Bentler & Kano (1990, *MBR* 25:67–74; https://doi.org/10.1207/s15327906mbr2501_8) is commonly cited for factors and components becoming equivalent as the number of variables grows *(content not verified here)*.
- On extracting too many factors: Fava & Velicer (1992, *MBR* 27:387–415; https://doi.org/10.1207/s15327906mbr2703_5) and Wood, Tataryn & Gorsuch (1996, *Psychological Methods* 1:354–365; https://doi.org/10.1037/1082-989X.1.4.354) *(content not verified here)*.

**Implication for this repo**
- IFN-α at K=40 has 92 genes, about 2.3 genes per factor, and communalities of at most 0.67. That is far from the regime where PCA and FA converge. FA itself returns only 36 nonzero factors, so **K=40 goes beyond what the FA model supports**. This is the mechanism behind the guide's "degenerate late FA factors."
- Add `effective_k` to the Stage 1 metadata: the number of nonzero FA components, equivalently the number of Ψ-whitened eigenvalues above 1. The current `effective_k` field just repeats the requested K. Treat any K above it as too many factors.
- Describe PCA vs FA wins as differences in representation under a fixed probe, not as statements about latent biology. This supports the guide's existing refusal to claim that "FA is intrinsically inferior to PCA for IFN-α."
- **Q1:** report shared-model results either at K ≤ effective K or as subspace-level summaries.

---

## 2. Whether factors replicate across samples and groups

### 2.1 Measurement invariance as an analogy for "shared across patients"

**The literature.**
- Multi-group factor analysis begins with Jöreskog (1971, *Psychometrika* 36:409–426; https://doi.org/10.1007/BF02291366).
- Meredith (1993, *Psychometrika* 58:525–543; https://doi.org/10.1007/BF02294825) formalized factorial invariance.
- Vandenberg & Lance (2000, *Organizational Research Methods* 3:4–70; https://doi.org/10.1177/109442810031002) and Putnick & Bornstein (2016, *Developmental Review* 41:71–90; https://doi.org/10.1016/j.dr.2016.06.004) codify the levels:
  - **configural:** same factor count and pattern of salient loadings;
  - **metric (weak):** equal loadings;
  - **scalar (strong):** equal intercepts;
  - **strict:** equal residual variances.
- *Partial* invariance is allowed (Byrne, Shavelson & Muthén, 1989, *Psychological Bulletin* 105:456–466; https://doi.org/10.1037/0033-2909.105.3.456).
- With many groups, **alignment** estimates approximate invariance (Asparouhov & Muthén, 2014, *Structural Equation Modeling* 21:495–508; https://doi.org/10.1080/10705511.2014.919210).

| Level | Psychometric meaning | Single-cell analog here | Status in this repo |
|---|---|---|---|
| Configural | Same factors and salient-loading pattern in each group | A per-patient refit finds the same programs (top genes) | Not tested; Stage 1 is a pooled transductive fit |
| Metric | Equal loadings | Per-patient loadings Procrustes-congruent with the pooled ones (φ ≥ .95; .85 = fair) | **Imposed by construction**, not tested |
| Scalar | Equal intercepts, so latent means are comparable | Patient-level gene mean shifts not explained by factor means (patient or batch effects) | Not tested; matters for pooled LOPO training |
| Strict | Equal residual variances | Per-patient Ψ | Not tested |

**Implication for this repo**
- **Q1:** the transductive Stage 1 fit on all 13 patients imposes metric invariance. LOPO lift therefore tests whether the *probe direction* transfers, not whether the factor is *measured* the same way in each patient. A claim of shared biology needs at least a configural and metric check: refit Stage 1 per supported patient (or leave-one-patient-out, i.e., inductively), Procrustes-align to the pooled solution, and report φ per factor and patient against the .85/.95 bands.
- **Q1/Q2:** within-patient AUPRC is unaffected by a patient-specific additive shift in the probe score, so scalar non-invariance cannot hurt the held-out ranking. But LOPO *training* pools 12 patients whose malignant prevalence differs: about 2.3% of training cells in the P02 fold and 2.7% in the P07 fold. Patient-level shifts in factor means can then be learned as malignancy signal (a Simpson-type effect). **Test:** center factor scores per patient (MOFA+-style, §2.2) before Stage 2 and recompare LOPO lift.
- D14 (patient factor overlap) is a descriptive configural-type check on score distributions. Extend it to loadings via per-patient refits.
- Use alignment-style framing: report the fraction of (factor, patient) pairs that fail invariance, rather than a single pass/fail.

### 2.2 Multi-group and multi-study factor models

**The literature.**
- **MOFA** (Argelaguet et al., 2018, *Mol. Syst. Biol.* 14:e8124; https://doi.org/10.15252/msb.20178124).
- **MOFA+** (Argelaguet et al., 2020, *Genome Biology* 21:111; https://doi.org/10.1186/s13059-020-02015-1) adds a group-wise prior hierarchy, so the ARD prior "does not only act on model weights but also on the factor activities." It reports variance explained per factor per group. "Features are centered per group (i.e., intercept effects are regressed out)" because the aim is to find which sources of variability are shared vs group-exclusive, not mean differences.
- **MSFA** (De Vito, Bellio, Trippa & Parmigiani, 2019, *Biometrics* 75:337–346; https://doi.org/10.1111/biom.12974) separately estimates factors common to all studies and study-specific factors. The Bayesian high-dimensional version is De Vito et al. (2021, *Ann. Appl. Stat.* 15:1723–1741; https://doi.org/10.1214/21-AOAS1456).
- **Tetris** (Grabski, De Vito, Trippa & Parmigiani, 2023, *Ann. Appl. Stat.* 17:2212–2235; https://doi.org/10.1214/22-AOAS1715) allows factors shared by *any subset* of studies, using an Indian Buffet Process.

**Implication for this repo**
- The MSFA terms map onto the PI questions: common factors ↔ **Q1**, study-specific factors ↔ **Q2**, and Tetris's subset sharing ↔ **Q3**. An example is an IFN axis used in P02, P03 and P07 but not in the other patients.
- Run MOFA+ with group = patient on IFN-α and two or three shortlisted panels. This gives a model-based variance-explained matrix per factor and patient, which complements D14's overlap.
- Caveat: centering per group removes patient mean shifts, which is right for within-patient questions. Because malignant prevalence is low (about 1–3%), the shift caused by composition differences should be small.
- All of these models treat cells as exchangeable rows. Uncertainty must still come from resampling at the patient level.

### 2.3 Consensus and stability across restarts and resamples

**The literature.**
- **cNMF** (Kotliar et al., 2019, *eLife* 8:e43803; https://doi.org/10.7554/eLife.43803):
  - Runs the factorization many times and filters "outlier components (which tend to represent noise or merges/splits of GEPs)."
  - Clusters the components and takes cluster medians.
  - Chooses K by trading off stability (silhouette) against error.
- **staNMF** chooses K by stability (Wu et al., 2016, *PNAS* 113:4290–4295; https://doi.org/10.1073/pnas.1521171113).
- The stability principle (Yu, 2013, *Bernoulli* 19:1484–1500; https://doi.org/10.3150/13-BEJSP14) and the PCS framework (Yu & Kumbier, 2020, *PNAS* 117:3920–3929; https://doi.org/10.1073/pnas.1901326117) ask that conclusions survive reasonable perturbations.

**Implication for this repo**
- The grid has only `stage1_seed=42` (plan line 79), so seed stability has never been measured. Run R seeds × patient-stratified cell bootstraps, then cluster the components cNMF-style. This gives (a) how often a PC8-like or FA7-like component reappears and (b) flags for components that split or merge. That evidence is what **Q3** needs.
- Choose K per panel by the stability/error trade-off instead of the fixed 5/10/20/40. For IFN-α FA this also catches K > 36.
- **PCS test of the IFN-α headline:** the PCA-over-FA conclusion in P02 flips under a 5% change in C (§3.1), so it fails a basic stability check and should not be reported as a method difference.

---

## 3. Sparse linear probes on correlated or rotated features

### 3.1 Lasso selection depends on the basis and on a threshold

**The literature.**
- The lasso itself: Tibshirani (1996, *JRSS-B* 58:267–288; https://doi.org/10.1111/j.2517-6161.1996.tb02080.x).
- Whether the lasso selects the right variables depends on the **irrepresentable condition**, which mostly concerns predictor covariance (Zhao & Yu, 2006, *JMLR* 7:2541–2563; https://jmlr.org/papers/v7/zhao06a.html).
- Sparse algorithms are not stable (Xu, Caramanis & Mannor, 2012, *IEEE TPAMI* 34:187–193; https://doi.org/10.1109/TPAMI.2011.177).
- Zou & Hastie (2005, *JRSS-B* 67:301–320; https://doi.org/10.1111/j.1467-9868.2005.00503.x) note that within a highly correlated group "the lasso tends to select only one variable from the group and does not care which one is selected." The elastic net instead shows a grouping effect.
- The scikit-learn objective (https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) is (1/S)Σ s_i·loss_i + r(w)/(S·C). In other words, ‖w‖₁/C is set against the *summed* weighted loss.
- From the KKT conditions, an L1 coefficient leaves zero only when |Σ_i s_i(y_i − p̂)z_ij| > 1/C.
- With balanced class weights (S = n, p̂ ≈ ½), this becomes: a standardized feature enters only if its malignant-minus-normal mean difference satisfies **|Δ_j| > 4/(n·C)**. A fixed C is therefore **not** a fixed per-cell penalty.

**Implication for this repo.** The IFN-α PCA40 vs FA40 gap is a threshold knife-edge. The numbers below come from my LOPO refit.

| Held-out fold | n_train | 4/(nC) | PC8: \|grad\| (Δ) | FA7: \|grad\| (Δ) | FA8: \|grad\| (Δ) | Active at C=10⁻⁴ |
|---|---|---|---|---|---|---|
| P02 | 58,656 | 0.682 SD | 10,850 (−0.740), +8.5% over 1/C | **9,665 (+0.659), −3.4% under** | 4,018 (−0.274) | PCA: {8}; FA: ∅ |
| P07 | 57,933 | 0.690 SD | 10,728 (−0.741), +7.3% | **9,806 (+0.677), −1.9%** | 3,815 (−0.263) | PCA: {8}; FA: ∅ |
| P03 | 57,791 | 0.692 SD | 11,122 (−0.770) | 10,176 (+0.704), +1.8% | 3,560 (−0.246) | PCA: {8}; FA: {7} |

- These numbers reproduce the guide's §4 fold pattern exactly. **At C = 1.05×10⁻⁴, FA7 enters in P02 (18.06 pp) and P07 (6.97 pp).** So in P02, once FA7 enters, FA40-L1 *beats* PCA40-L1 (15.94 pp). Only P07 keeps a real PCA advantage (11.08 vs 6.97 pp). This refines the verified fact that "L2 on FA recovers": the P02 gap is pure regularization threshold, and the P07 gap is partly real.
- **Why PC8 crosses the threshold and FA7 does not.**
  - L1 rewards label signal concentrated in a single coordinate.
  - FA's convention spreads PC8's variance over FA7 and FA8 (R² 0.47 and 0.34). The label signal sits mainly on FA7 (Δ 0.66) rather than FA8 (Δ −0.27), and FA7 lands just under the threshold.
  - The opposite signs of PC8 and FA7 (Δ −0.74 vs +0.66) are the sign arbitrariness at work. Probe β signs mean nothing without the loading signs.
- **Basis dependence, Q1 methodology.** L2 logistic regression on whitened features (uncorrelated and standardized, which both PCA scores and sklearn FA scores are) is unchanged by orthogonal rotations. L1 is not. PCA-vs-FA comparisons under **L2** therefore isolate real subspace differences, which exist (21 of 36 canonical correlations are below 0.9). Under **L1** they mix subspace with orientation. Use L2 or elastic net for method comparisons, and L1 only for parsimony.
- **Donor-downsampling collapse (guide §5)** follows from the same formula. Removing donor-healthy cells lowers n, which raises 4/(nC) and pushes every factor below the threshold. The "rescue" analysis the guide calls for is simply C_eff = C·n_full/n_sub, or reselecting C inside each condition.
- **D15 learning curve:** the low-n points use the same fixed C, so they confound signal loss with a stronger effective penalty. Rerun with C scaled to n, or with C selected per n.

### 3.2 Stability selection and remedies for correlated features

**The literature.**
- **Stability selection** (Meinshausen & Bühlmann, 2010, *JRSS-B* 72:417–473; https://doi.org/10.1111/j.1467-9868.2010.00740.x) refits the lasso on random half-samples and keeps features whose selection probability exceeds π_thr. Under an exchangeability assumption it bounds the expected number of false selections at about q²/((2π_thr − 1)p).
- Complementary-pairs stability selection loosens the assumptions (Shah & Samworth, 2013, *JRSS-B* 75:55–80; https://doi.org/10.1111/j.1467-9868.2011.01034.x).
- The cluster lasso selects clusters of correlated predictors (Bühlmann, Rütimann, van de Geer & Zhang, 2013, *J. Stat. Plan. Inference* 143:1835–1858; https://doi.org/10.1016/j.jspi.2013.05.019).
- Model-X knockoffs control FDR (Candès, Fan, Janson & Lv, 2018, *JRSS-B* 80:551–577; https://doi.org/10.1111/rssb.12265). They need a model of the feature distribution, and cells within a patient are not iid.

**Implication for this repo**
- Replace "the driver at the global C" with **selection probabilities** over a C path around the chosen C. Subsample at the **patient** level (half of the training patients, or patient-stratified half-samples), because cells within a patient are not exchangeable.
- For factosig_promax (score correlations up to 0.92; 136 pairs above 0.5), select at the cluster level: cluster factors by |r| or by Φ and report the cluster. Alternatively use elastic net. Do not name a single promax factor as a driver.
- The grid already includes elastic net. Report how often EN keeps FA7 and FA8 together compared with L1.

### 3.3 Post-selection inference and the winner's curse

**The literature.**
- Inference after selection: PoSI (Berk, Brown, Buja, Zhang & Zhao, 2013, *Ann. Stat.* 41:802–837; https://doi.org/10.1214/12-AOS1077), exact lasso inference at fixed λ (Lee, Sun, Sun & Taylor, 2016, *Ann. Stat.* 44:907–927; https://doi.org/10.1214/15-AOS1371), and an overview by Taylor & Tibshirani (2015, *PNAS* 112:7629–7634; https://doi.org/10.1073/pnas.1507583112).
- The cross-validation error of a tuned classifier is biased, and nested CV fixes this (Varma & Simon, 2006, *BMC Bioinformatics* 7:91; https://doi.org/10.1186/1471-2105-7-91).
- Over-fitting the model-selection criterion biases performance estimates (Cawley & Talbot, 2010, *JMLR* 11:2079–2107; https://www.jmlr.org/papers/v11/cawley10a.html).

**Implication for this repo**
- C, method, K and panel are all chosen with the same LOPO criterion that is then reported (the guide's own fairness caveat). Use **nested LOPO**: an inner LOPO over the 6 training patients picks C, method and K, and the outer fold reports. This is affordable for the shortlist.
- Lasso p-values (Lee et al.) assume a Gaussian linear model at fixed λ. That does not fit logistic regression on cells clustered within patients, so use patient-level resampling for uncertainty.

---

## 4. Attribution vs causation, and going back and forth with domain knowledge

### 4.1 Probe weights are backward-model weights

**The literature.**
- Haufe et al. (2014, *NeuroImage* 87:96–110; https://doi.org/10.1016/j.neuroimage.2013.10.067) separate two kinds of model:
  - **Backward models (decoders):** weights can be nonzero on channels statistically independent of the signal (they cancel noise) and zero on related channels.
  - **Forward patterns:** A = Σ_x·W·Σ_ŝ⁻¹, which can be interpreted.
- Lipton (2018, *ACM Queue* 16(3):31–57; https://doi.org/10.1145/3236386.3241340) cautions against treating model parameters as explanations.

**Implication for this repo**
- For every probe, compute the gene-space activation pattern a = cov(X_genes, s)/var(s), where s is the probe score. This pattern does not depend on rotation or sign.
  - Compare the PCA40-L1, FA40-L2 and FA40-L1 (C = 1.05×10⁻⁴) probes by Tucker φ of these patterns.
  - φ ≥ .95 means they encode the same IFN biology whatever the factor labels. That is a direct **Q1/Q3** answer.
- Report factor-level structure coefficients, corr(z_j, s). A factor zeroed by L1 but with a high structure coefficient, for example FA8 or correlated promax factors, is not "irrelevant."

### 4.2 Fixed-model ablation vs refit importance

**The literature.**
- **LOCO** (leave-one-covariate-out) importance refits the model without each feature (Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, 2018, *JASA* 113:1094–1111; https://doi.org/10.1080/01621459.2017.1307116).
- Permutation importance on correlated features forces the model to extrapolate (Hooker, Mentch & Zhou, 2021, *Statistics and Computing* 31:82; https://doi.org/10.1007/s11222-021-10057-z).

**Implication for this repo**
- The guide's ΔAP ablation (§4 step 2) correctly calls itself attribution of the fitted model. Add LOCO, refitting with C reselected and grouped for correlated clusters, to measure how much held-out lift depends on a factor's *information* when other factors can stand in. For promax factors the two will disagree.
- Avoid permutation importance on promax scores.

### 4.3 A causal caveat specific to this cohort

The point below is reasoning, not a cited result. Malignant and normal cells differ in cell-type composition. A factor that tracks a cell type enriched among malignant cells will look predictive without being a malignant-cell program. Two checks, each feeding **Q1/Q2**:
- Evaluate AUPRC within `predicted.annotation` strata. The repo already stratifies the downsampling masks this way.
- Add cell-type indicators as unpenalized covariates and check whether IFN-factor contributions persist.

### 4.4 Rotation and factorization guided by prior knowledge

**The literature.**
- **Target (Procrustes) rotation** to a partially specified target exists in orthogonal and oblique forms (Browne, 1972, *BJMSP* 25:115–120, https://doi.org/10.1111/j.2044-8317.1972.tb00482.x; and 25:207–212, https://doi.org/10.1111/j.2044-8317.1972.tb00492.x).
- **Spectra** (Kunes, Walle, Land, Nawy & Pe'er, 2024, *Nature Biotechnology* 42:1084–1095; https://doi.org/10.1038/s41587-023-01940-3):
  - Turns gene sets into a gene–gene knowledge graph and adds a graph penalty, weighted against reconstruction by λ.
  - Estimates a per-factor dependence on the graph (η) and can detach factors to learn new programs.
  - Fits cell-type-specific as well as global factors.
  - "Disentangles the highly correlated features of CD8+ T cell tumor reactivity and exhaustion."
- **f-scLVM** (Buettner et al., 2017, *Genome Biology* 18:212; https://doi.org/10.1186/s13059-017-1334-8) jointly estimates factor relevance, refines gene-set annotations and infers unannotated factors.
- **PLIER** (Mao, Zaslavsky, Hartmann, Sealfon & Chikina, 2019, *Nature Methods* 16:607–610; https://doi.org/10.1038/s41592-019-0456-1) learns a sparse matrix U that ties each latent variable to prior pathways. Technical factors end up in latent variables with no prior association.
- **expiMap** (Lotfollahi et al., 2023, *Nature Cell Biology* 25:337–350; https://doi.org/10.1038/s41556-022-01072-x) uses soft gene-program membership: an L1 penalty on non-member genes lets them enter. It prunes programs with a group lasso and learns new programs.
- Reviews and transfer: Stein-O'Brien et al. (2018, *Trends in Genetics* 34:790–805; https://doi.org/10.1016/j.tig.2018.07.003) and **projectR** transfer learning (Stein-O'Brien et al., 2019, *Cell Systems* 8:395–411; https://doi.org/10.1016/j.cels.2019.04.004).

**Implication for this repo**
- **Cheap test, no refit (Q1/Q3):**
  - Build a partially specified target for the IFN-α panel from *independently defined* sub-signatures, for example gene modules from an external dataset or curated sub-lists fixed before looking at Stage 2.
  - Rotate the FA36 (effective) loadings orthogonally to that target and rerun Stage 2.
  - Orthogonal rotation of whitened scores leaves L2 unchanged, so any L1 gain is **pure basis alignment**. This cleanly tests whether a domain-aligned basis helps sparse probes.
- **A domain-knowledge loop:**
  1. Data-driven factors.
  2. Gene grounding by pattern **and** structure (§1.2).
  3. Annotation against independent gene sets.
  4. Encode the annotation as a target rotation or a Spectra-style prior (λ/η).
  5. Refit and check that the probe's gene pattern (§4.1) and the lift are preserved.

  Keep "novel"/η-like diagnostics so that departures from the prior stay visible instead of being absorbed.
- **External comparator:** run Spectra with `predicted.annotation` cell types on IFN-α and antigen-presentation panels. It also addresses the cell-type confound in §4.3. Spectra is a different model family (non-negative factorization of expression), so treat it as a sensitivity analysis.
- Use projectR-style projection as an *inductive* check of the transductive fit: learn factors on 12 patients and project the held-out patient.

---

## 5. Synthesis: analyses to consider, ranked

| # | Analysis | Questions | Cost | What it would settle |
|---|---|---|---|---|
| A1 | L1 entry-margin audit: g_j·C per fold and representation; flag folds within ±10% of the threshold | Q1 | Minutes | Which L1 "method wins" are threshold artifacts; IFN-α P02 already shown to be one |
| A2 | Scale C with n (C·n_full/n) for donor downsampling and D15 | Q1 | Hours | Whether collapse is signal loss or penalty scaling |
| A3 | Probe gene activation patterns (Haufe) plus Tucker φ across PCA, FA and promax probes | Q1, Q3 | Hours | Whether different factor bases encode the same biology |
| A4 | Promax: save Φ and M, normalize columns, report pattern **and** structure top genes; rebuild Layer-5 grounding | Q3 | Hours (factosig change) | Gene names for oblique factors |
| A5 | Determinacy ρ_k and effective K (nonzero FA components) in `diagnostics.json` | Q3 | Minutes | Which factors may be interpreted on their own |
| A6 | Rotation-invariant within-plane direction test (FA7/FA8 or PC7–PC9) per patient | Q2, Q3 | Hours | Split as basis artifact vs real patient-specific sub-axes |
| A7 | Multi-seed × patient-bootstrap consensus (cNMF-style) and eigengap report; K chosen by stability | Q1, Q3 | Days | Factor reproducibility; flags for splits and merges |
| A8 | Per-patient Stage 1 refits, Procrustes + Tucker φ (configural/metric invariance) | Q1, Q2 | Days | Whether shared loadings are justified |
| A9 | Per-patient centering of scores before Stage 2 (scalar-invariance sensitivity) | Q1 | Hours | Whether LOPO exploits patient mean shifts |
| A10 | Patient-level stability selection, plus cluster-level selection for promax | Q1, Q3 | Hours | Replaces single-C "driver" claims |
| A11 | Nested LOPO for C, method and K | Q1 | Days | Removes winner's-curse optimism |
| A12 | Target rotation to independent IFN sub-signatures; Spectra comparator with cell types; within-cell-type AUPRC | Q1, Q3 | Days | How domain knowledge and cell-type confounding affect the probes |

---

## References

All URLs were checked; DOIs were resolved via Crossref on 2026-10-08.

- Argelaguet R et al. (2018) Multi-Omics Factor Analysis. *Mol Syst Biol* 14:e8124. https://doi.org/10.15252/msb.20178124
- Argelaguet R et al. (2020) MOFA+. *Genome Biol* 21:111. https://doi.org/10.1186/s13059-020-02015-1
- Asparouhov T, Muthén B (2014) Multiple-group factor analysis alignment. *SEM* 21:495–508. https://doi.org/10.1080/10705511.2014.919210
- Bentler PM, Kano Y (1990) On the equivalence of factors and components. *MBR* 25:67–74. https://doi.org/10.1207/s15327906mbr2501_8 *(content not verified here)*
- Berk R et al. (2013) Valid post-selection inference. *Ann Stat* 41:802–837. https://doi.org/10.1214/12-AOS1077
- Bernaards CA, Jennrich RI (2005) Gradient projection algorithms… *EPM* 65:676–696. https://doi.org/10.1177/0013164404272507
- Björck Å, Golub GH (1973) Angles between linear subspaces. *Math Comp* 27:579–594. https://doi.org/10.1090/S0025-5718-1973-0348991-3
- Bro R, Acar E, Kolda TG (2008) Resolving the sign ambiguity in the SVD. *J Chemometrics* 22:135–140. https://doi.org/10.1002/cem.1122
- Browne MW (1972) Orthogonal rotation to a partially specified target. *BJMSP* 25:115–120. https://doi.org/10.1111/j.2044-8317.1972.tb00482.x
- Browne MW (1972) Oblique rotation to a partially specified target. *BJMSP* 25:207–212. https://doi.org/10.1111/j.2044-8317.1972.tb00492.x
- Browne MW (2001) Overview of analytic rotation in EFA. *MBR* 36:111–150. https://doi.org/10.1207/S15327906MBR3601_05
- Buettner F et al. (2017) f-scLVM. *Genome Biol* 18:212. https://doi.org/10.1186/s13059-017-1334-8
- Bühlmann P, Rütimann P, van de Geer S, Zhang C-H (2013) Correlated variables in regression: clustering and sparse estimation. *JSPI* 143:1835–1858. https://doi.org/10.1016/j.jspi.2013.05.019
- Byrne BM, Shavelson RJ, Muthén B (1989) Partial measurement invariance. *Psychol Bull* 105:456–466. https://doi.org/10.1037/0033-2909.105.3.456
- Candès E, Fan Y, Janson L, Lv J (2018) Model-X knockoffs. *JRSS-B* 80:551–577. https://doi.org/10.1111/rssb.12265
- Cawley GC, Talbot NLC (2010) Over-fitting in model selection. *JMLR* 11:2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html
- Costello AB, Osborne JW (2005) Best practices in EFA. *PARE* 10(7). https://doi.org/10.7275/jyj1-4868
- Courville T, Thompson B (2001) Structure coefficients in multiple regression. *EPM* 61:229–248. https://doi.org/10.1177/0013164401612006
- Davis C, Kahan WM (1970) Rotation of eigenvectors by a perturbation III. *SIAM J Numer Anal* 7:1–46. https://doi.org/10.1137/0707001
- De Vito R, Bellio R, Trippa L, Parmigiani G (2019) Multi-study factor analysis. *Biometrics* 75:337–346. https://doi.org/10.1111/biom.12974
- De Vito R et al. (2021) Bayesian multistudy factor analysis. *Ann Appl Stat* 15:1723–1741. https://doi.org/10.1214/21-AOAS1456
- Fabrigar LR, Wegener DT, MacCallum RC, Strahan EJ (1999) Evaluating EFA use. *Psychol Methods* 4:272–299. https://doi.org/10.1037/1082-989X.4.3.272
- Fava JL, Velicer WF (1992) Overextraction. *MBR* 27:387–415. https://doi.org/10.1207/s15327906mbr2703_5 *(content not verified here)*
- Grabski IN, De Vito R, Trippa L, Parmigiani G (2023) Bayesian combinatorial multi-study factor analysis (Tetris). *Ann Appl Stat* 17:2212–2235. https://doi.org/10.1214/22-AOAS1715
- Graham JM, Guthrie AC, Thompson B (2003) Structure coefficients in CFA. *SEM* 10:142–153. https://doi.org/10.1207/S15328007SEM1001_7
- Grice JW (2001) Computing and evaluating factor scores. *Psychol Methods* 6:430–450. https://doi.org/10.1037/1082-989X.6.4.430
- Guttman L (1955) Determinacy of factor score matrices. *Br J Stat Psychol* 8:65–81. https://doi.org/10.1111/j.2044-8317.1955.tb00321.x
- Haufe S et al. (2014) Interpretation of weight vectors of linear models. *NeuroImage* 87:96–110. https://doi.org/10.1016/j.neuroimage.2013.10.067
- Hendrickson AE, White PO (1964) Promax. *Br J Stat Psychol* 17:65–70. https://doi.org/10.1111/j.2044-8317.1964.tb00244.x
- Hooker G, Mentch L, Zhou S (2021) Unrestricted permutation forces extrapolation. *Stat Comput* 31:82. https://doi.org/10.1007/s11222-021-10057-z
- Jennrich RI, Sampson PF (1966) Rotation for simple loadings. *Psychometrika* 31:313–323. https://doi.org/10.1007/BF02289465
- Jöreskog KG (1971) Simultaneous factor analysis in several populations. *Psychometrika* 36:409–426. https://doi.org/10.1007/BF02291366
- Kaiser HF (1958) The varimax criterion. *Psychometrika* 23:187–200. https://doi.org/10.1007/BF02289233
- Kotliar D et al. (2019) cNMF. *eLife* 8:e43803. https://doi.org/10.7554/eLife.43803
- Kraha A et al. (2012) Interpreting multiple regression under multicollinearity. *Front Psychol* 3:44. https://doi.org/10.3389/fpsyg.2012.00044
- Kuhn HW (1955) The Hungarian method. *Nav Res Logist Q* 2:83–97. https://doi.org/10.1002/nav.3800020109
- Kunes RZ, Walle T, Land M, Nawy T, Pe'er D (2024) Spectra. *Nat Biotechnol* 42:1084–1095. https://doi.org/10.1038/s41587-023-01940-3
- Lee JD, Sun DL, Sun Y, Taylor JE (2016) Exact post-selection inference for the lasso. *Ann Stat* 44:907–927. https://doi.org/10.1214/15-AOS1371
- Lei J et al. (2018) Distribution-free predictive inference for regression (LOCO). *JASA* 113:1094–1111. https://doi.org/10.1080/01621459.2017.1307116
- Lipton ZC (2018) The mythos of model interpretability. *ACM Queue* 16(3):31–57. https://doi.org/10.1145/3236386.3241340
- Lorenzo-Seva U, ten Berge JMF (2006) Tucker's congruence coefficient. *Methodology* 2:57–64. https://doi.org/10.1027/1614-2241.2.2.57
- Lotfollahi M et al. (2023) expiMap. *Nat Cell Biol* 25:337–350. https://doi.org/10.1038/s41556-022-01072-x
- Lovik A, Nassiri V, Verbeke G, Molenberghs G (2020) Modified Tucker's congruence coefficient. *Methodology* 16:59–74. https://doi.org/10.5964/meth.2813
- Mao W et al. (2019) PLIER. *Nat Methods* 16:607–610. https://doi.org/10.1038/s41592-019-0456-1
- Meinshausen N, Bühlmann P (2010) Stability selection. *JRSS-B* 72:417–473. https://doi.org/10.1111/j.1467-9868.2010.00740.x
- Meredith W (1993) Measurement invariance and factorial invariance. *Psychometrika* 58:525–543. https://doi.org/10.1007/BF02294825
- Putnick DL, Bornstein MH (2016) Measurement invariance conventions. *Dev Rev* 41:71–90. https://doi.org/10.1016/j.dr.2016.06.004
- R Core Team, `stats::promax`/`varimax` source. https://github.com/wch/r-source/blob/trunk/src/library/stats/R/factanal.R
- Rohe K, Zeng M (2023) Vintage factor analysis with varimax performs statistical inference. *JRSS-B* 85:1037–1060. https://doi.org/10.1093/jrsssb/qkad029
- Schneeweiss H, Mathes H (1995) Factor analysis and principal components. *J Multivar Anal* 55:105–124. https://doi.org/10.1006/jmva.1995.1069
- Schönemann PH (1966) Orthogonal Procrustes. *Psychometrika* 31:1–10. https://doi.org/10.1007/BF02289451
- scikit-learn: FactorAnalysis docs. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html ; Logistic regression objective. https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Shah RD, Samworth RJ (2013) Complementary-pairs stability selection. *JRSS-B* 75:55–80. https://doi.org/10.1111/j.1467-9868.2011.01034.x
- Steiger JH (1979) Factor indeterminacy in the 1930s and 1970s. *Psychometrika* 44:157–167. https://doi.org/10.1007/BF02293967
- Stein-O'Brien GL et al. (2018) Enter the matrix. *Trends Genet* 34:790–805. https://doi.org/10.1016/j.tig.2018.07.003
- Stein-O'Brien GL et al. (2019) projectR transfer learning. *Cell Syst* 8:395–411. https://doi.org/10.1016/j.cels.2019.04.004
- Taylor J, Tibshirani RJ (2015) Statistical learning and selective inference. *PNAS* 112:7629–7634. https://doi.org/10.1073/pnas.1507583112
- Thurstone LL (1947) *Multiple-Factor Analysis*. University of Chicago Press. https://wellcomecollection.org/works/jhqr9a62 (catalog record)
- Tibshirani R (1996) The lasso. *JRSS-B* 58:267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- Vandenberg RJ, Lance CE (2000) Measurement invariance review. *ORM* 3:4–70. https://doi.org/10.1177/109442810031002
- Varma S, Simon R (2006) Bias in CV error with model selection. *BMC Bioinformatics* 7:91. https://doi.org/10.1186/1471-2105-7-91
- Velicer WF, Jackson DN (1990) Component vs common factor analysis. *MBR* 25:1–28. https://doi.org/10.1207/s15327906mbr2501_1
- Widaman KF (1993) Common factor vs principal component analysis. *MBR* 28:263–311. https://doi.org/10.1207/s15327906mbr2803_1
- Wood JM, Tataryn DJ, Gorsuch RL (1996) Under- and overextraction. *Psychol Methods* 1:354–365. https://doi.org/10.1037/1082-989X.1.4.354 *(content not verified here)*
- Wu S et al. (2016) Stability-driven NMF. *PNAS* 113:4290–4295. https://doi.org/10.1073/pnas.1521171113
- Xu H, Caramanis C, Mannor S (2012) Sparse algorithms are not stable. *IEEE TPAMI* 34:187–193. https://doi.org/10.1109/TPAMI.2011.177
- Yu B (2013) Stability. *Bernoulli* 19:1484–1500. https://doi.org/10.3150/13-BEJSP14
- Yu B, Kumbier K (2020) Veridical data science. *PNAS* 117:3920–3929. https://doi.org/10.1073/pnas.1901326117
- Yu Y, Wang T, Samworth RJ (2015) A useful variant of Davis–Kahan. *Biometrika* 102:315–323. https://doi.org/10.1093/biomet/asv008
- Zhao P, Yu B (2006) Model selection consistency of lasso. *JMLR* 7:2541–2563. https://jmlr.org/papers/v7/zhao06a.html
- Zou H, Hastie T (2005) Elastic net. *JRSS-B* 67:301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
