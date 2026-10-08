# Hypotheses: mrd_stage0_2 claim register

Migrated 2026-10-08 from the Cursor canvas SNAPSHOT 2026-10-02b (verbatim copy in legacy/).
Static fields only. Verdicts go in devlog verdicts:, alternative states in devlog alts:, numbers in devlog Results.

Each block is parsed by `tools/research_index.py`: the header is `## H-<S0|S1|S2|I|A>-<NN> — <statement>` and the key lines below it are number-free.
Statements of migrated rows are verbatim from the canvas. Current verdicts, alternative states and evidence are rendered in STATUS.md and CLAIMS.md.
Alternative IDs are defined in alternatives.csv; analysis IDs in analyses.csv; task IDs in the workstream READMEs.

# Stage 0

## H-S0-01 — Malignant-vs-non-malignant structure is recoverable inside curated prior gene spaces (single genesets and biology groups), not only in broad HVG bags.
- stage: s0
- questions: [Q2, Q1]
- analyses: [D1, D3, D12]
- workstreams: [W4, W3]
- alternatives: [ALT-GENESET, ALT-SEL, ALT-NPAT, ALT-TRANS]
- blocking: [W4.2, W4.1, W3.3]
- figures: [V2A, V2F1, D1, D2, D1DD1, D1DD9]
- gate_supports: Descriptive for old34: malignant-vs-normal structure is recoverable inside curated program gene spaces and transfers above constant-score no-skill AP for typical supported patients on the transductive basis.
- gate_not: Not that curated priors beat any N genes (Null Model 1, size-matched HVG and exact single-score comparators are missing); control bags are technical floors, not the biological null.
- origin: canvas 2026-10-02b

## H-S0-02 — The old34 GSEA dictionary is not an alias of HVG 3,000.
- stage: s0
- questions: [Q2, Q3]
- analyses: [D12]
- workstreams: [W3]
- alternatives: [ALT-GENESET]
- blocking: []
- figures: [D12]
- gate_supports: The old34 dictionary and the HVG anchors are not interchangeable Stage 0 panels; the control rows are technical floors shown beside the programs.
- gate_not: Not that the programs beat size-matched HVG or random gene sets run through the same Stage 2 grid.
- origin: canvas 2026-10-02b

## H-S0-03 — Bottom-up manuscript-axis panels (Set B atomic / family union / leave-one-family-out) identify which dictionaries add malignancy signal.
- stage: s0
- questions: [Q3, Q2]
- analyses: []
- workstreams: [W4]
- alternatives: [ALT-SEL, ALT-GENESET]
- blocking: [W4.7]
- figures: [Set B scorecards]
- gate_supports: Compute exists for Set B; no scientific claim yet.
- gate_not: No claim about which bottom-up dictionaries add malignancy signal until Set B is read with the lift figure contract (scope decision W4.7).
- origin: canvas 2026-10-02b

## H-S0-04 — Pooled old34∪expanded bags (no per-geneset identity) still carry shared malignancy, and size-matched HVG 11,421 is the fair bag-level control.
- stage: s0
- questions: [Q3, Q2]
- analyses: []
- workstreams: [W0, W3]
- alternatives: [ALT-GENESET]
- blocking: [W0.6]
- figures: [overlap QC]
- gate_supports: Stage 0 and Stage 1 bookkeeping only (gene-overlap QC of the pooled bags).
- gate_not: No Stage 2 claim until the pooled Stage 2 is merged or the campaign is formally abandoned (W0.6); the canvas value in_flight is retired and run state lives in campaigns.csv.
- origin: canvas 2026-10-02b

# Stage 1

## H-S1-01 — Stage 1 method and K are not interchangeable inside a panel; transfer can flip with geometry × regularization.
- stage: s1
- questions: [Q3, Q2]
- analyses: [D1, D1DD2, D4, F1]
- workstreams: [W3]
- alternatives: [ALT-SEL, ALT-PEN, ALT-SEED]
- blocking: [W3.3, W4.5]
- figures: [V2F2, V2F2c, D7D, D8B–E, D1DD2, D1DD4, D1DD7, D1DD8, F1a–h]
- gate_supports: Descriptive interaction: within a panel, transfer changes with Stage 1 method, K and the Stage 2 penalty.
- gate_not: Not a method ranking and not that FA is intrinsically inferior to PCA; selection and C policy were neither matched nor nested, and independently fitted K are not nested prefixes.
- origin: canvas 2026-10-02b

## H-S1-02 — A no-DR (direct-gene) probe on the same Stage 0 panel is the right Stage 1 control. DR is not universally additive.
- stage: s1
- questions: [Q3]
- analyses: [D9, D1e, F4, D1DD9, D23, D24]
- workstreams: [W3]
- alternatives: [ALT-SEL, ALT-PEN, ALT-TRANS]
- blocking: [W3.3, W3.4, W2.1]
- figures: [D1a, D1b, D1c, D1e, D9A–D, D1DD5, D1DD9, F2a–d, F4a–b, F5a–b]
- gate_supports: Descriptive: the direct-gene probe on the same panel is the right no-DR control; DR helps pooled AP far more often than patient-equal median or P20 lift, and its gains concentrate in the highest-prevalence patients.
- gate_not: Not that DR universally adds value; best-DR selection is in-sample, the comparison is asymmetric while Stage 1 is transductive, and the one-score proxy is not a fair pathway-score baseline. Quote pooled AP, median lift and P20 lift together.
- origin: canvas 2026-10-02b

## H-S1-03 — The across-patient Stage 1 basis lets discovery / LOPO / patient-specific coefficients live in one coordinate system — but LOPO is label transfer on a transductive atlas, not inductive deployment.
- stage: s1
- questions: [Q1]
- analyses: [D3, D4, D17]
- workstreams: [W2]
- alternatives: [ALT-TRANS]
- blocking: [W2.1, W2.6]
- figures: [design, §8]
- gate_supports: Discovery, LOPO and patient-specific coefficients share one transductive Stage 1 coordinate system.
- gate_not: LOPO is label transfer on a transductive atlas, not inductive deployment or a raw-expression deployment benchmark; donor-downsampling refits are not inductive.
- origin: canvas 2026-10-02b

## H-S1-04 — Patients can be aggregated on a shared Stage 1 basis when per-patient factor-score distributions overlap. Disjoint supports would be separate subtypes.
- stage: s1
- questions: [Q1, Q2]
- analyses: [D14, D3, D16, D20]
- workstreams: [W1, W2]
- alternatives: [ALT-TRANS, ALT-NPAT, ALT-DEGEN, ALT-PREV]
- blocking: [W1.2, W1.3]
- figures: [D14A–E]
- gate_supports: Descriptive and transductive: supported patients' factor-score supports overlap on nearly every biological program basis, so pooling them on one Stage 1 basis is reasonable and no biological program shows a patient-private subtype factor.
- gate_not: Not that patient identity does not affect scores; not that D3 patient-specific gaps are subtypes; not that high-prevalence patients leave the loadings unshaped; not significance (the permutation p sits at its floor and the gates are uncalibrated); not an inductive statement.
- origin: canvas 2026-10-02b

## H-S1-05 — High-prevalence patients P02/P03/P04 shape the Stage 1 loadings, not only Stage 2 labels and selection.
- stage: s1
- questions: [Q1, Q2]
- analyses: [D16, D17]
- workstreams: [W1, W2]
- alternatives: [ALT-PREV, ALT-TRANS]
- blocking: [W1.1, W2.1, W2.4]
- figures: [D16, D17]
- gate_supports: Would need loading-leverage evidence: leave-high-prevalence-out or patient-balanced refits that move the loadings (principal angles, matched factor correlation) more than dropping other patients does.
- gate_not: Score overlap (D14) is not influence on the loadings; cell share is not prevalence, and these patients are a small share of the label-blind Stage 1 fit; axis movement beyond size-matched random drops is not leverage while those drops do not match malignant composition (D17); no claim before the composition-matched controls and W2.4.
- origin: new 2026-10-08

## H-S1-06 — Held-out-patient transfer survives an inductive Stage 1.
- stage: s1
- questions: [Q1, Q3]
- analyses: [D17]
- workstreams: [W2]
- alternatives: [ALT-TRANS, ALT-PEN]
- blocking: [W2.1, W2.6]
- figures: [D17]
- gate_supports: Would need per-patient lift on an inductive basis (fit without the held-out patient, training-only standardization, per-fold tuned C) that stays comparable to the transductive anchor; scope: per panel.
- gate_not: Not established beyond the IFN-alpha pilot (W2.6); not inductive deployment (N07); transductive LOPO and donor-downsampling refits do not establish it.
- origin: new 2026-10-08

## H-S1-07 — The P03-private malignant axis seen only in broad control bags reflects patient biology rather than genotype, label provenance, batch or composition.
- stage: s1
- questions: [Q1]
- analyses: [D14, D25]
- workstreams: [W1]
- alternatives: [ALT-DONOR, ALT-LABEL, ALT-BATCH, ALT-CT]
- blocking: [W1.4]
- figures: [D14A–E]
- gate_supports: Would need the D25 audit to exclude genotype (HLA and sex-chromosome genes), label-assay provenance, batch and cell-type composition as explanations of the axis.
- gate_not: Not a P03 subtype and not a shared malignant program; the axis appears only in broad control bags, not in any biological program basis.
- origin: new 2026-10-08

# Stage 2

## H-S2-01 — PI Q1 — a robust shared model exists: high discovery AP, high aggregate LOPO, consistent per-patient lift, stable selected factors.
- stage: s2
- questions: [Q1, Q2]
- analyses: [D1, D1f, D3, D9]
- workstreams: [W1, W4]
- alternatives: [ALT-PREV, ALT-SEL, ALT-NPAT, ALT-TRANS, ALT-GENESET]
- blocking: [W4.1, W4.2, W3.3, W2.1]
- figures: [V2F1, V2A, D1e, D1f, D6B–D, D1DD3, D1DD4, D1DD5]
- gate_supports: Descriptive: immune and inflammatory programs transfer above constant-score no-skill AP for typical supported patients on the transductive basis.
- gate_not: Shared-candidate is a triage tag, not a publication claim; lift still ranks with prevalence; there is no matched-gene-set p-value or patient-level CI; median-split quadrants are not biological verdicts.
- origin: canvas 2026-10-02b

## H-S2-02 — PI Q2 — patient-specific models can reveal biology that the shared model misses, rather than repeating it.
- stage: s2
- questions: [Q1]
- analyses: [D3, D13]
- workstreams: [W1]
- alternatives: [ALT-PREV, ALT-NPAT, ALT-SEL]
- blocking: [W1.5, W1.6]
- figures: [D3, D7B, D13, D14]
- gate_supports: Descriptive: patient-specific opportunity tags (D3) and within-patient DE in low-support patients (D13) exist.
- gate_not: Not that high lift is caused by patient-specific biology; support and prevalence are not separated; D3 patient-specific fits are in-sample; single-patient peaks are non-identifiable.
- origin: canvas 2026-10-02b

## H-S2-03 — PI Q3 — a prior program breaks into distinct regulatory groups: different factors, different patients, partially overlapping discovery/LOPO/patient-specific selections.
- stage: s2
- questions: [QI, Q1]
- analyses: [D5, D8, F6, F9, D18]
- workstreams: [W5]
- alternatives: [ALT-ROT, ALT-PEN, ALT-SEED]
- blocking: [W5.1, W5.5]
- figures: [D8, D4, D5, F3a–f, F6a–c, F7a–b]
- gate_supports: Descriptive, one grounded case: the IFN-alpha program carries one shared combination axis (PC8 matched to FA7, named from both poles) whose patient coverage depends on the Stage 2 penalty.
- gate_not: One grounded case, not a census of programs; factor indices are not unique biological programs; L1 coverage at a fixed C is not Stage 1 structure.
- origin: canvas 2026-10-02b

## H-S2-04 — Donor-healthy abundance can induce a shortcut; downsampling donor-healthy training cells tests that.
- stage: s2
- questions: [Q2]
- analyses: [D10, D11]
- workstreams: [W4]
- alternatives: [ALT-DONOR, ALT-PEN]
- blocking: [W4.6]
- figures: [D10E, D11B, D11C]
- gate_supports: Not supported as stated: at the selected fixed C, donor downsampling collapses L1 probes to all-zero, so AP returns to prevalence by construction; that is a regularization artifact, not a donor test.
- gate_not: Not that the model is invariant to donor-versus-recipient origin, and not that it learns donor identity; no performance-level donor test exists until condition-specific C (W4.6).
- origin: canvas 2026-10-02b

## H-S2-05 — Patient- and cell-level error structure can explain which cells create an aggregate focus-panel lift before mechanistic interpretation.
- stage: s2
- questions: [Q1, QI]
- analyses: [D1DD10, D1DD11, D1DD12, F7]
- workstreams: [W5]
- alternatives: [ALT-CT, ALT-PREV]
- blocking: [W5.3]
- figures: [D1DD10, D1DD11, D1DD12, F7a–b]
- gate_supports: Descriptive: post-hoc audits of prevalence-matched rank errors by cell type, collection timing, QC and ADT can nominate factors and confounds for review.
- gate_not: Rank-defined TP, FN, FP and TN are post-hoc descriptive groups, not clinical misclassifications; the patterns are not mechanistic; no operating threshold has been chosen without held-out labels.
- origin: canvas 2026-10-02b

## H-S2-06 — On a frozen Stage 1 basis, held-out AUPRC rises with the number of Stage 2 training cells. That is a depth curve for this cohort, not a forecast for sequencing a new patient.
- stage: s2
- questions: [Q2]
- analyses: [D15, D15b]
- workstreams: [W6]
- alternatives: [ALT-PEN, ALT-TRANS, ALT-NPAT]
- blocking: [W6.1, W6.2]
- figures: [D15A–B]
- gate_supports: Descriptive: on a frozen transductive basis, held-out AUPRC can rise with Stage 2 training-cell depth at a fixed patient mix where the probe is not collapsed at fixed C.
- gate_not: Not a Stage 1 learning curve and not evidence that sequencing another patient would move AUPRC; cell depth only; not a patient-count curve; flat L1 curves at fixed C are all-zero collapse, not absence of signal.
- origin: canvas 2026-10-02b

## H-S2-07 — Team hypothesis (a): PCA orthogonality pushes later components beyond P03 while FA keeps allocating factors to the highest-prevalence patient.
- stage: s2
- questions: [QI, Q3]
- analyses: [F3, F6, D7]
- workstreams: [W3, W5]
- alternatives: []
- blocking: []
- figures: []
- gate_supports: None: retired. The IFN-alpha PCA versus FA coverage gap is explained by Stage 2 L1 at a fixed C (H-S2-08), and Stage 1 is label-blind.
- gate_not: do not cite as a mechanism
- origin: new 2026-10-08

## H-S2-08 — The IFN-alpha PCA40-vs-FA40 patient-coverage gap at the selected C is a Stage 2 L1 threshold effect on related directions in one shared subspace (PC8, FA7), not missing Stage 1 structure.
- stage: s2
- questions: [QI, Q3]
- analyses: [D8, F3, F6, F9, D19]
- workstreams: [W3]
- alternatives: [ALT-PEN, ALT-TRANS, ALT-ROT]
- blocking: [W3.2, W2.1]
- figures: [D8, F3, F6, F9]
- gate_supports: Descriptive and transductive: at the selected C the coverage gap tracks the L1 entry threshold on related directions (PC8, FA7), and L2 or per-fold tuned C on the same FA scores recovers the patients that L1 drops.
- gate_not: Not that FA misses patients as a property of FA; not a Stage 1 structure difference; not that PC8 and FA7 are one identical axis (D18); not that PCA beats FA in P07: the tuned-C residual there is single-patient and penalty-dependent (D19); not an inductive statement.
- origin: new 2026-10-08

# Interpretability

## H-I-01 — Constant-score average precision equals malignant prevalence exactly under the implemented metric.
- stage: interp
- questions: [Q2]
- analyses: [D6]
- workstreams: [W4]
- alternatives: []
- blocking: []
- figures: [D6A, guide §1]
- gate_supports: Metric contract: constant-score no-skill AP = π under the implemented metric (proved and checked in D6).
- gate_not: A metric contract, not a biology claim; AP and a single clinical cutoff are separate statements.
- origin: canvas 2026-10-02b

## H-I-02 — Influential features can be named with SHAP or LIME (Aim 2.4 as written).
- stage: interp
- questions: [meta]
- analyses: []
- workstreams: [W5]
- alternatives: []
- blocking: [W5.4]
- figures: [D4, D5]
- gate_supports: None yet: the current stack is regularization paths, coefficient ablation and Stage 1 loadings, not SHAP or LIME.
- gate_not: Grant text is not the current method; do not present loadings or coefficients as SHAP or LIME output until W5.4 decides.
- origin: canvas 2026-10-02b

## H-I-03 — Stage 1 loadings can be used as sign-oriented descriptive gene lists, but L1-selected factor indices are not unique biological programs.
- stage: interp
- questions: [QI]
- analyses: [D5, F9, D18]
- workstreams: [W5]
- alternatives: [ALT-ROT, ALT-SEED, ALT-DEGEN]
- blocking: [W5.1, W5.4]
- figures: [F2d, F3]
- gate_supports: Descriptive: sign-oriented within-factor loading ranks are readable gene lists when the sign orientation and the Stage 2 penalty (L1, L2 or elastic net) are stated.
- gate_not: Not that loadings cannot be looked at; not that a factor index is the IFN-alpha gene program; L1-selected factor indices are not unique biological programs; not SHAP or LIME.
- origin: canvas 2026-10-02b

# Grant Aim 2 (CRC; not the MDS spine)

## H-A-01 — Integrating genomic (ctDNA/SNV) and transcriptomic features with standard pathology improves MRD and recurrence prediction over histopathological staging.
- stage: aim2
- questions: [meta]
- analyses: []
- workstreams: []
- alternatives: []
- blocking: []
- figures: []
- gate_supports: None: not tested in the canonical experiment (no recurrence labels and no clinicopathological, ctDNA or TNM data).
- gate_not: Do not present LOPO cell-level lift as recurrence-risk improvement; CRC clinical stage is not pipeline Stage.
- origin: canvas 2026-10-02b

## H-A-02 — Relapse 4-class (MRD_cancer / MRD_normal / Relapse_cancer / Relapse_normal) and huCIRA priors extend the same Stage 0–2 idea.
- stage: aim2
- questions: [meta]
- analyses: []
- workstreams: []
- alternatives: []
- blocking: []
- figures: [value-added]
- gate_supports: None: roadmap only (value-added plan).
- gate_not: Roadmap, not evidence; do not launch unless asked.
- origin: canvas 2026-10-02b
