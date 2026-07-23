# Stage 2 Figure Catalog + Statistical Rigor (old34, v2 Jun 4 suite)

Date compiled: 2026-07-03
Notebook: `../notebooks/stage0_2/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb`
Scorecards: `experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093/analysis/scorecards/stage2_figure3_sharedness_v2_jun4/`

This is the output of dossier action **§11.1 — "walk the v2 suite plot-by-plot with the four-element
framing (§10) to decide which plots are figure-worthy"**, plus the statistical-rigor design that
should gate any plot before it becomes a manuscript panel. It merges (a) the plot-by-plot
walkthrough and (b) the aggregation / null-model / uncertainty decisions taken on 2026-07-03. It is
the reference for porting the suite to the expanded Set B (§11.3).

Malignant-prevalence baseline for the whole experiment ≈ **0.026**. Read every AUPRC against it.

---

## 0. Two "across-patient" aggregations (read this first)

The suite silently used **two different** cross-patient aggregations. Distinguishing them explains
most of the confusion about "why do the numbers look big here and small there."

### Option A — Pooled-cell aggregate AUPRC (the *old* V2A / V2F2 y-axis)

Concatenate held-out predictions from all LOPO folds and compute one AUPRC:

```
A_pool(p,m,K,r) = AUPRC( ∪_{h∈H} { (y_i, s_i) : i ∈ D_h } )
```

Answers: *across all held-out cells from all patients, do malignant cells score higher than normal
cells?* **Problem:** dominated by patients with more cells, more malignant cells, easier structure,
or higher prevalence — here P02 (211 malignant), P03 (758), P09 (375) drive it. In the code this is
`cell_weighted_auprc`.

### Option B — Patient-summary aggregate AUPRC / lift (the *new* y-axis)

Compute each patient's held-out AUPRC first, then summarize across patients:

```
A_patient(p,m,K,r) = median_{h∈H} A_LOPO(p,m,K,r,h)          # AUPRC form
L̃(p,m,K,r)       = median_{h∈H} L(p,m,K,r,h)                # lift form (preferred)
```

Answers: *for a **typical** held-out patient, does this panel/method/K/reg combination enrich
malignant cells above that patient's prevalence baseline?* Weights every patient equally, so it is
not inflated by large-malignant-count patients. **Decision (2026-07-03): use Option B everywhere.**
V2F2/V2F2b now plot the patient-summary median AUPRC (Option B, AUPRC form); V2F2c plots the
patient-summary median lift (Option B, lift form). The enrichment ratio (AUPRC / prevalence) is
**removed** — it explodes for low-prevalence patients and is not interpretable.

**"Supported patients"** = the 7 held-out patients with ≥ `MIN_MALIGNANT_SUPPORT` (=10) malignant
cells and not normal-only: **P01, P02, P03, P04, P06, P07, P09**. P05 (8) and P13 (4) are low-support
(marked `*`); P08/P10/P11/P12 are normal-only (marked `N`, specificity only).

---

## 1. Statistical rigor and null models (target design)

The §10 working agreement requires a **stated null** per figure. The v2 suite currently only uses
Null Model 2. Null Model 1 and the uncertainty estimates are the main "what to build next" for
making these figures defensible, and are the precondition for turning lift into a *p-value* plot.

### Null Model 1 — random gene sets of matched size (preferred)

For each observed panel `p`, draw random gene sets `p*_b` with `|G_{p*}| = |G_p|`, `b = 1..B`, run
the **same Stage 0–2 pipeline**, and compute the null transfer statistic `L̃(p*_b)`. Then:

```
p_empirical = ( 1 + Σ_b 1[ L̃(p*_b) ≥ L̃(p) ] ) / ( B + 1 )
```

This tests whether the panel sits in the right tail of matched random gene-set performance — i.e.
whether the *knowledge prior* adds signal beyond "any N genes of this size." This is the test that
converts the descriptive lift plots into hypothesis tests, and directly answers the §10 second
clause ("does not exceed a gene-budget-matched control").

### Null Model 2 — patient-specific prevalence baseline (already in the suite)

```
π_h = n_malignant,h / (n_malignant,h + n_normal,h)          # expected random-ranker AUPRC in patient h
L(p,m,K,r,h) = A_LOPO(p,m,K,r,h) − π_h                       # additive lift over prevalence
```

Simple and essential, but it does **not** test whether a gene set beats other gene sets of similar
size / expression structure. That is why Null Model 1 is needed on top of it.

### HVG as a null (weak biological null)

Highly variable genes are a useful **technical** benchmark but a poor **biological** null: HVGs may
already be driven by malignant/normal mixture, patient composition, or cell-type composition. Use
HVG controls as a sanity floor, not as the primary null. (This is also dossier pipeline-gap #1: the
size-matched HVG controls were never pushed through multi-objective Stage 2 for old34.)

## 2. Uncertainty estimates for AUPRC / lift

Report standard errors for the headline lift statistics. Three distinct sources:

1. **Cell-level sampling** — bootstrap held-out cells *within each patient*.
2. **Patient-level** — summarize variability across held-out patients, or bootstrap patients.
3. **Algorithmic** — repeat Stage 1 decomposition + Stage 2 training across random seeds
   (currently single-seed `seed=42` everywhere — dossier pipeline-gap #4). This measures method
   stability, **not** biological sampling uncertainty; don't conflate them, but it is what tells you
   whether method-to-method differences (the `method_stability_label` taxonomy) are real.

## 3. Report AUROC and AUPRC together

- AUROC random baseline = **0.5** (patient-independent → easy to compare across patients).
- AUPRC random baseline = **π_h** (patient-specific → the relevant one for rare MRD malignant cells).

A strong MRD classifier should improve **both** AUROC and prevalence-adjusted AUPRC lift. The suite
currently reports AUPRC/lift only; AUROC should be added as a secondary column so cross-patient
comparison has a prevalence-free anchor.

---

## 4. Figure catalog (four-element framing per §10)

Verdict legend: **HEADLINE** (manuscript candidate), **SUPPORTING** (QC / drill-down),
**RETIRE** (superseded).

### V2A — Discovery vs LOPO decision map — **HEADLINE**

- **Question:** which Stage 0 gene spaces carry a *shared* malignant signal that transfers to
  held-out patients, vs one that only looks good in-cohort?
- **Experiment:** 33 representations; x = discovery AUPRC (full-cohort refit, `stage2_auprc`),
  y = aggregate LOPO AUPRC; best regularization per representation. Quadrants cut at cohort medians.
- **Results vs baseline:** everything below x=y (LOPO ≤ discovery, expected); immune / inflammatory /
  proliferation-metabolism panels reach LOPO ≈ 0.35–0.48 vs the 0.026 floor — an order of magnitude
  over baseline.
- **Take-home:** a handful of immune/inflammatory programs are genuinely shared. This is the plot
  that confirms the teammate's DEG (malignant-vs-healthy) result from the representation side.
- **Gaps to fix:** (1) quadrants are relative to the cohort median, not the null — add a
  prevalence/HVG reference; (2) y-axis is pooled-cell (Option A) — consider an Option B twin;
  (3) porting to Set B (632 panels) needs the interactive builder, static is unreadable at that size.

### V2B1 — per-patient LOPO raw-AUPRC heatmap — **RETIRE** (kept for now, superseded by V2F4)

- **Why weak:** raw AUPRC on a fixed 0→1 scale re-draws the prevalence table — bright columns
  (P02/P03/P04) are just the high-prevalence patients; dark columns (P01/P06/P09) are low-prevalence,
  where even a good model can't push raw AUPRC up. The take-home is an artifact of the metric+scale,
  **not** of "best reg per rep-patient."
- **Fix already exists:** V2F4 is the prevalence-corrected version (lift, diverging scale). Keep V2B1
  only as a raw-numbers reference; the manuscript story lives in V2F4.

### V2B2 — per-patient enrichment (AUPRC / prevalence) heatmap — **REMOVED (2026-07-03)**

Ratio-of-AUPRC-to-prevalence explodes for low-prevalence patients and is not interpretable. Deleted
along with the `auprc_enrichment` metric.

### V2F1 — shortlist readiness: transfer floor vs typical lift — **HEADLINE (PI Q1)**

- **Question (PI Q1):** which Stage 0 panels are a *robust shared* malignancy model — transfer above
  each held-out patient's own prevalence baseline, for a typical patient *and* on the weak patients?
- **Null H0:** best regularized LOPO lift ≤ 0 (does not beat prevalence). H1: lift > 0.
- **Experiment:** per panel, over the 7 supported patients: y = **median** held-out lift (typical
  transfer), x = **20th-percentile** held-out lift (near-worst-case floor; with 7 patients ≈ 2nd
  worst). Dot size = # supported patients with positive lift. Color = `method_stability_label`.
- **Reading:** top-right = high typical lift AND high floor = safe shared panel; near/below the y=x
  diagonal = floor much worse than median = a few patients carry it.
- **On "does 'across supported patients' make the signal worse?":** it makes the numbers *smaller on
  purpose*. It keeps the low-prevalence patients (P01/P06/P09) whose lift is genuinely tiny, and it
  weights every patient equally instead of letting P03 dominate (contrast the pooled-cell aggregate,
  which looks bigger). Small honest numbers are the correct behavior for a "robust shared model"
  bar. A strong single-patient program is *supposed* to score low here — that is the patient-specific
  axis (`max_patient_auprc` / `patient_specific_gap`), a different question.
- **`method_stability_label` decision process (now printed on the plot):** bar =
  cohort-median aggregate LOPO AUPRC (`LOPO_HIGH_THRESHOLD` ≈ 0.069). A Stage 1 method "works" if its
  best-K LOPO clears the bar; it is "K-stable" if it also has LOPO IQR ≤ 0.05 and range ≤ 0.10 across
  K = 5/10/20/40. Then: `method_robust` (≥2 work AND ≥1 K-stable) → `method_specific_k_stable`
  (exactly 1 works, K-stable) → `k_sensitive` (clears bar but needs K tuning) → `single_spike_brittle`
  (clears bar in ≤1 method×K cell) → `null` (nothing clears). It scores **robustness of the Stage 1
  choice**, not raw performance — which is why a strong panel can be `k_sensitive`.

### V2F2 / V2F2b / V2F2c — Stage 1 method × K sensitivity — **HEADLINE**

- **Question:** for a given Stage 0 panel, how does transfer depend on Stage 1 method (pca / fa /
  factosig / factosig_promax) and latent dimension K?
- **Experiment:** one facet per panel, one line per method, x = K. V2F2 = priority-ranked panels;
  V2F2b = requested MSigDB pathway list; V2F2c = same panels, per-patient median **lift**.
- **Metric (updated 2026-07-03):** y-axis is now **Option B patient-summary median AUPRC** (median
  across supported patients), replacing the pooled-cell aggregate that was inflated by P02/P03.
  Reference line = typical supported-patient prevalence. V2F2c keeps median **lift** with a 0 baseline.
- **Take-home:** these are the most informative panels because they refuse to aggregate away Stage 1;
  they expose exactly the axis the stability labels compress.
- **Detail tension:** these collapse the *patient* axis. Full method×K×patient is 4-D and cannot be
  shown flat — this is the concrete argument for the interactive tool (§5).

### V2F3 — panel × method transfer-lift heatmap (annotated with best K) — **HEADLINE/SUPPORTING**

Median held-out lift per panel×method, K annotated. Compact companion to V2F2; good for the
manuscript's "which method to use per panel" summary.

### V2F4 — panel-level held-out lift topology — **HEADLINE** (biological insight)

- **Question:** which held-out patients does each panel transfer to?
- **Experiment:** best aggregate-LOPO representation per panel; rows = panels, cols = patients,
  color = held-out lift (diverging, centered 0). Support marks `*`/`N`.
- **Take-home:** **P02/P03/P04 show broad positive lift across almost every program, while P06 (and
  the low-prevalence patients) stay pale** — some patients' malignant cells are "broadly programmed"
  (separable along many axes at once), others are not. This is a real, statable biological
  observation and is more interesting than the panel ranking.
- **Hedge required:** P02/P03/P04 are also highest-prevalence; lift corrects the *baseline* but not
  the *difficulty*, so state the claim as "even after prevalence correction." **This hedge is exactly
  why Null Model 1 (matched-size random gene sets) + per-patient uncertainty (§1–2) are needed** —
  with an empirical p-value per panel×patient this becomes "significant broad transfer" rather than
  "looks bright."

### V2F5 — patient leverage on transfer summaries — **SUPPORTING (QC)**

- **Question:** are the "broad transfer" conclusions real, or propped up by a single held-out patient?
- **Experiment:** per panel×method with ≥3 supported patients, compute median lift, then recompute
  dropping each patient (leave-one-patient-out); y = max |median − LOO-median|, x = full median lift,
  color = which patient's removal drove the swing, marker = stability label.
- **Take-home:** a fragility filter. High-lift + low-leverage = trustworthy; high-lift + high-leverage
  = near-patient-specific masquerading as shared (expect P02/P03 as the usual leverage drivers). Guards
  V2A/V2F1 against the "one patient carries it" failure mode. Supporting, not a headliner.

### V2F6a / V2F6b — theme-level transfer and stability — **SUPPORTING (synthesis; needs Set B)**

- **Question:** are certain biological theme families (IFN, cytokine/JAK-STAT, antigen presentation,
  stress, cell cycle, metabolism …) systematically the robust broad-transfer ones?
- **Experiment:** V2F6a = boxplot of transfer floor (p20 lift) per theme × stability label (0-line =
  null); V2F6b = stacked-bar count of stability labels per theme.
- **Take-home:** first cut at Diagnostic Layer 7 (biology-theme synthesis). With old34's 34 curated
  programs the per-theme counts are thin → suggestive only. **This is the plot that pays off on Set B**
  (632 panels densely populate each theme).

---

## 5. Interactive tool for V2F2 / V2F2b / V2F2c (brainstorm)

The static facets can't show method × K × **patient** simultaneously, and Set B has 632 panels. An
interactive companion (sibling to the existing `stage2_mrd_fig3a_interactive.py` Fig-3A tool) should:

- **Panel picker** (search/multiselect over `short_panel_label`, filter by `stage0_panel_type`,
  `biological_theme`, `method_stability_label`) → renders the method×K facet on demand instead of a
  fixed 12-panel grid.
- **Aggregation toggle**: pooled-cell AUPRC (Option A) ↔ patient-summary median AUPRC (Option B) ↔
  patient-summary median lift — so the viewer sees how the story changes with the aggregation, which
  is the core teaching point of §0.
- **Patient drill-down**: hover / click a (method, K) point to expand the per-patient distribution
  (the collapsed 4th dimension) — box/strip of the 7 supported patients, with P05/P13/normal-only
  greyed and labelled, plus each patient's prevalence baseline.
- **Reference lines**: prevalence baseline (Option B AUPRC), 0 (lift), and — once Null Model 1 exists
  — the matched-random-gene-set null band + empirical p-value badge per panel.
- **Uncertainty ribbons**: cell-level bootstrap CI within patient and, when seeds exist, the
  method-stability spread across seeds.
- **Cross-link**: clicking a panel jumps to its V2A position and its V2F4 patient row, so the
  interactive tool ties the three views together.
- **Reuse**: read the same `stage2_figure3_sharedness_v2_jun4/` CSVs; make it Set-A/Set-B switchable
  by pointing at the root vs `expanded_stage0_mrd_manuscript_axes_v1/` scorecard dirs. Priority:
  medium — it is the natural way to make Set B's 632 panels explorable, but it depends on the Option B
  tables (done) and ideally the null model (to build).

---

## 6. Changes applied to the notebook (2026-07-03)

- **Removed the enrichment metric** (`auprc_enrichment`) and **V2B2** entirely (compute in cell 9,
  plot in cell 10, and the `median_auprc_enrichment` summary columns in cell 12).
- **Added the Option B patient-summary metric** (`build_patient_summary_k`) to the diagnostic cell:
  per (panel, method, K), median across supported patients of held-out AUPRC and lift, plus
  `TYPICAL_SUPPORTED_PREVALENCE`. Written to `v2_panel_method_k_patient_summary_auprc.csv`.
- **V2F2 / V2F2b** y-axis switched from pooled-cell aggregate LOPO AUPRC → **patient-summary median
  AUPRC**; reference line switched from cohort-median LOPO to typical supported-patient prevalence.
  **V2F2c** kept as patient-summary median lift (0 baseline).
- **V2F1** retitled to **PI Q1** with an explicit null, and now prints the `method_stability_label`
  decision process on the plot.

## 7. Implications for §11.3 (expanded Set B)

- Lead the Set B analysis with the **lift-based, non-aggregated-away** figures: V2A (with a null
  line), V2F2/V2F2c, V2F4, and a captioned V2F1. Drop V2B1/V2B2.
- V2F6 (theme synthesis) is where Set B's 632 panels finally pay off — promote it there.
- Build the interactive tool (§5) as the primary way to navigate 632 panels.
- Gate the "broad transfer" biological claims (V2F4) behind **Null Model 1 + uncertainty** (§1–2)
  before they go in the manuscript; that upgrade also applies retroactively to old34.
