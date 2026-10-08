# Prompt: illustrate LOPO AUPRC lift, median lift, and p20 lift

Copy everything below the line into ChatGPT Web's image/illustration workflow.

---

Create a polished scientific explainer illustration for a computational single-cell biology manuscript. Use a clean flat-vector style, white background, restrained blue/orange/red palette, large readable labels, and a 16:9 landscape canvas. The figure must be understandable without a caption and must not imply statistical significance.

## Title

**From patient-held-out AUPRC to median and 20th-percentile lift**

## Scientific setting

- We classify malignant versus non-malignant single cells in an MRD cohort.
- A "representation" is one Stage 0 biological gene panel plus one Stage 1 method and latent dimension K.
- Stage 1 is an unsupervised shared coordinate system fit once on expression from all eligible cells. Label transfer is therefore **transductive**, not a fully inductive deployment test.
- Stage 2 uses leave-one-patient-out (LOPO): train the classifier on all other patients and predict the held-out patient's cells.
- AUPRC is prevalence-sensitive. The expected AUPRC of a random ranking in held-out patient `h` is that patient's malignant prevalence:

  `prevalence_h = malignant cells_h / all cells_h`

- Define additive held-out lift:

  `lift_h = held-out AUPRC_h - prevalence_h`

- A lift of `+0.05` means **+5 AUPRC percentage points over that patient's own random-ranking baseline**. It does not mean 5% sensitivity, 5% accuracy, or a p-value.

## Required three-part layout

### Panel A — one global regularization is chosen from pooled LOPO predictions

Draw 13 patient icons labeled P01–P13 in a leave-one-patient-out loop.

For each candidate regularization setting:

1. hold out one patient;
2. train Stage 2 on the other patients;
3. predict the held-out patient's cells;
4. repeat for all patients;
5. concatenate all held-out predictions and calculate one pooled-cell LOPO AUPRC.

Show several candidate regularization settings as rows. Highlight the single setting with the highest pooled-cell LOPO AUPRC and label it:

**"Choose one global regularization for this representation"**

Add a warning:

**"Pooled selection weights patients by their cell counts and uses the same LOPO folds later summarized; it is not an untouched external test."**

Do not label this as full-cohort in-sample performance. Every Stage 2 prediction contributing to this pooled criterion came from a fold where that patient's labels were excluded from classifier training.

### Panel B — evaluate that fixed setting patient by patient

Show the seven supported held-out patients:

- P01: 45 malignant / 11,860 total, prevalence 0.0038
- P02: 211 / 1,462, prevalence 0.1443
- P03: 758 / 2,327, prevalence 0.3257
- P04: 102 / 1,128, prevalence 0.0904
- P06: 44 / 3,259, prevalence 0.0135
- P07: 22 / 2,185, prevalence 0.0101
- P09: 375 / 17,126, prevalence 0.0219

For each patient, draw:

- a gray baseline bar for malignant prevalence;
- a colored bar for held-out AUPRC using the fixed global regularization;
- a vertical bracket or arrow between them labeled `lift_h`;
- green lift when AUPRC is above prevalence and red lift when it is below.

Include the equation:

`lift_h = AUPRC_h - prevalence_h`

Clearly state:

**"The baseline is patient-specific, not the whole-cohort prevalence."**

Show P05 and P13 separately in gray as low-support exclusions, and P08/P10/P11/P12 as normal-only exclusions. Explain that the Option-B summary uses the seven supported patients above.

### Panel C — summarize the seven patient lifts without cell-count weighting

Draw seven lift dots sorted from lowest to highest on a vertical AUPRC-lift axis centered at zero.

Mark:

- **Median held-out lift**: the middle (4th) ordered patient lift. Label it "typical supported-patient transfer."
- **20th-percentile held-out lift (p20)**: a lower-tail transfer-floor summary. With seven patients and pandas' default linear quantile, it lies 20% of the way from the second-smallest to the third-smallest lift. Label it "near-worst-case transfer floor, not the literal worst patient."
- a dashed zero line labeled "patient-prevalence baseline."

Use this schematic example only, and label it **"illustrative numbers—not study results"**:

Sorted lifts:

`[-0.01, 0.00, 0.02, 0.03, 0.05, 0.08, 0.12]`

- median = `0.03`
- p20 = `0.004`, because it linearly interpolates between `0.00` and `0.02`

Add two interpretation callouts:

1. **High median, low/negative p20:** works for a typical patient but fails in the lower tail.
2. **High median and high p20:** transfers both typically and across the weaker supported patients.

## Footer: guardrails

Include a compact footer with these statements:

- "Positive lift is descriptive evidence above prevalence, not a p-value."
- "Matched-size random-gene controls and uncertainty are still required for a knowledge-prior claim."
- "Patient-specific apparent fits are trained and scored on the same patient's cells; they are case-study signals, not held-out performance."

## Visual constraints

- Spell out AUPRC once as "area under the precision-recall curve."
- Show decimal lift and percentage-point interpretation together once: `0.05 = +5 AUPRC percentage points`.
- Never call lift "percent improvement."
- Do not draw a full-cohort prevalence line as the Option-B baseline.
- Do not depict the patient-specific model as validated or prospective.
- Keep all formulas exactly as written.

