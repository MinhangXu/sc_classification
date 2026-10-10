# Healthy-cell downsampling in MRD single-cell classification: pre-registered design (W8; H-S2-09)

> **Status: DRAFT (2026-10-09). Not locked. Runs only when the new patient data arrive.** That timing is the user's decision: "All the new proposed run idea like downsampling, let's include that when we have the new data from team." (closing line of the 2026-10-09 reading notes, recorded with A-20261009-12 in devlog 2026-10-09_report-reading-notes).
> - **Lock.** This file is locked together with `integration_plan.md` by the W8.2 `type: decision` entry, which records its sha256, the repo commit and the user's answers to §12. Changes after the lock need a new decision entry with `supersedes:`.
> - **Allowed before the lock:** writing the runner and testing it on toy matrices and on within-patient label-permuted cells (W8.6, §8). No held-out lift, AP or AUROC of a downsampling condition is computed on real labels (canonical or new cohort) before the lock, unless the user chooses the rehearsal in §12 item 1; a rehearsal is labelled as such and never pooled with the pre-registered results.
> - **Asks.** A-20261009-12 (the idea, and "we need to understand why"), A-20261009-14 (no patients without malignant cells in new runs), A-20261009-11 (which patients stay), A-20261009-16 (how the aggregation test bears on downsampling). Background: A-20260724-05 (the first donor downsampling, D10 and D11).
> - **Update 2026-10-10** (`research/asks/2026-10-10_protein-and-decisions.md`; devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs). Two user decisions "for runs when we have the extra patient data":
>   - **Cohort:** new full runs keep patients with at least 100 malignant cells and both classes (A-20261010-07; W8.5 closed; §4).
>   - **Re-weighting targets the Stage 2 labels** (A-20261010-08; W2.3 closed). This plan therefore puts the Stage-2-only placement first (§5.2), adds the Stage 2 sample-weight scheme SW (§5.1), and makes Stage 2 only the proposed primary placement of the recipe (§5.6, §12 item 2). The Stage 1 placements stay: they are the read-outs of why a gain happens (M1) and carry the user's original Stage 1 question about P09's healthy cells (A-20261009-12).
> - **Hypothesis** H-S2-09 (`research/hypotheses.md`). **Alternatives** ALT-NREG (new), ALT-PEN, ALT-PREV, ALT-CT, ALT-TRANS and ALT-SEL (`research/alternatives.csv`).
> - **Paths.** `EXP` is the canonical experiment dir and `NEWEXP` the new-cohort experiment dir (`integration_plan.md` §5). Artifact paths are relative to the named dir; code paths are repo-relative. Existing numbers are cited by devlog id, never restated (R12).

## 1. The idea and the question this plan must answer

The user's idea (A-20261009-12), in short:
- Malignant cells are few, and they are also heavily outnumbered: per patient, healthy (donor) cells outnumber malignant cells by a factor that ranges from a few to over a thousand (table in `integration_plan.md` §4.1).
- Two different things could make classification hard: the **count** of malignant training cells, and the **ratio** of healthy to malignant cells.
- Healthy cells are redundant. A subsample stratified by predicted cell type and trajectory position still represents the healthy population.
- So: downsample healthy cells systematically, show whether the ratio matters separately from the count, and explain **why** any gain happens.

> "My main concern is that if the downsampling of healthy cells improved the classification, we need to understand why?" (A-20261009-12)

The design is built around that sentence:
- §2.3 shows that, for our probe, a gain can come from only a few places;
- §5 is a factorial that switches each of them on and off separately;
- §6 lists the read-outs that say which one is acting;
- §7 fixes, before any data, what counts as success and what counts as falsification.

## 2. What is already known

### 2.1 The first donor downsampling (D10, D11): at fixed C, L1 collapses and L2 does not move

- **D10** removed donor-healthy *training* cells only, stratified by patient × predicted cell type, on the stored transductive basis, with the held-out patient untouched and C fixed at each representation's atlas-selected value (devlog 2026-08-21_donor-sensitivity-d10-d11).
  - Sparse (L1, EN) probes collapsed to the all-zero solution, so AP returned to π by construction.
  - L2 probes, which cannot collapse, barely moved.
  - This is why H-S2-04 is not supported *as a donor test* (THEME §6 S11).
- **D11** refit the transductive PCA/FA bases on the retained cells. The supervised IFN-alpha factors persist, FA7 more stably than PC8 at the deepest cut (same devlog; correction in 2026-08-03_narrative-guide-donor-correction; THEME §6 S12).
- **Reading.** At fixed C, downsampling mostly changes the L1 penalty. The one condition that isolates "Stage 2 sees fewer healthy cells" without collapse (L2) showed no gain.
- **Not yet done:** healthy downsampling with C retuned per condition (W4.6); a scheme that changes the cell-type composition; the malignant count varied separately from the ratio; a Stage 1 refit with Stage 2 retuned.

### 2.2 Why fixed C misleads: the L1 entry rule

- Under balanced L1 at w = 0, a standardized factor enters only if its training class-mean gap exceeds about 4/(C·n_train) (AGENTS R5; derived and checked in devlog 2026-10-08_d19-regularization-contract).
- Removing healthy cells lowers n_train, so at the same C the bar rises. That is the D10 collapse.
- The same audit found that most atlas cells select the lowest C on the runner grid. A full-data comparator tuned on that grid may therefore be unable to reach the effective penalty that a downsampled condition reaches. §5.4 extends the grid for every condition.

### 2.3 What `class_weight='balanced'` already does: the key to the "why"

The Stage 2 objective (sklearn and cuML alike; devlog 2026-10-08_d19-regularization-contract) is

    C · Σ_i s_i · loss_i  +  penalty(w),   with  s_i = n_train / (2 · n_class(i)).

Summed within each class, the data term is C · (n_train / 2) · [mean loss over malignant cells + mean loss over healthy cells]. Three consequences follow.

1. **The classes are already balanced.** Each class carries half the weight, however many healthy cells there are. Downsampling healthy cells does not rebalance the loss.
2. **Random healthy downsampling is a smaller C, plus noise.** If healthy cells are removed at random, or proportionally within strata, the mean healthy loss is unchanged in expectation and only n_train falls. That is the full-data objective at C × (n_after / n_before), plus sampling noise. With C tuned per condition over a grid that reaches that value, Stage-2-only random or proportional downsampling should give no systematic gain. A gain there is a regularization-path effect (ALT-NREG) or noise.
3. **Rank metrics ignore the class prior.** AP and AUROC depend only on how the held-out cells are ranked. The intercept shift that changing class proportions would cause (the case-control result for logistic regression; see the literature list in §10) cannot move them. The held-out patient's prevalence π is the same in every condition, so lift stays comparable across conditions.

So a gain can come only through these channels:

| Channel | What changes | Which conditions switch it on | Read-outs (§6) |
|---|---|---|---|
| M1 Stage 1 geometry | The basis. With fewer healthy cells, malignant cells hold more of the fit's variance, so the malignant-vs-healthy direction moves up the factor order and is more likely to sit inside the top K | Downsampling inside the Stage 1 fit set (transductive or inductive) | §6.2 |
| M2 Effective penalty | C · n_train | Any change of n_train. Neutralized by retuning C on an extended grid; diagnosed by the fixed-C and C·n-matched arms | §6.3 |
| M3 Negative-class composition | Which healthy cells define "normal". A scheme that keeps rare types or trajectory bins and thins abundant ones reweights the healthy mean loss toward the healthy cells that resemble malignant cells ("hard negatives"). This changes the decision function itself, not only its intercept | Composition-changing schemes (S3, S4) and the sample-weight scheme SW in Stage 2, compared with proportional schemes at the same n | §6.4 |
| M4 Sampling noise | Which healthy cells happen to be kept | Every stochastic condition; hurts in expectation | seed spread, §5.5 |
| M5 Tuning interplay | The inner-LOPO choice of C and penalty | Every condition; fixed by the inner-validation rule in §5.4 | §6.3 |

**A geometric prediction for M1 (stated before any data).**
- For PCA, a class-mean difference Δ contributes p(1 − p)·|Δ|² of variance along Δ, where p is the malignant fraction of the fit set.
- While p is small this grows almost in proportion to p. Thinning healthy cells therefore raises the variance on the malignant direction and moves it to earlier components. FA behaves similarly through its common variance.
- Two signatures follow:
  1. the rank of the discriminative factor (the factor with the largest standardized class-mean gap) should move toward factor 1 as p rises;
  2. Stage 1 gains should be largest at small K, where the full-data basis may not contain the direction, and smallest at K40, where it already does (IFN-alpha PC8 and FA7 at K40: F9, D18).
- If neither signature appears, M1 is not the explanation, whatever the lift does.

### 2.4 What the shared-basis analyses add (D14, D16, D17, D28; A-20261009-16)

- **D14 is the licence for stratified downsampling.**
  - Within normal cells, patients' factor-score distributions overlap on nearly every biological panel (devlog 2026-10-02_d14-patient-factor-overlap; H-S1-04). Within a cell type, healthy cells from different patients occupy the same score space, so thinning them loses little of the healthy population.
  - D14's overlap is also the check after downsampling: if the retained healthy cells no longer overlap the full healthy set, part of the healthy population was lost (§6.4).
  - This is how the aggregation test feeds the downsampling idea (A-20261009-16).
- **D16: score energy follows cells, not labels.**
  - Malignant cells hold about their cell share of factor score energy. The patient most over-represented in score energy is P01, a low-prevalence patient with many cells (devlog 2026-10-08_d16-factor-patient-matrix).
  - The user's P09 point (A-20261009-12) is the same mechanism: in the transductive basis, the largest patients' healthy cells set most of the variance, including when that patient is held out.
  - D28 (cell-type-stratified score-energy shares; devlog 2026-10-09_d28-celltype-score-shares) finds that P01's over-representation is mostly its cell-type mix, and that predicted cell types carry very different score energy per cell, so a stratified scheme preserves the Stage 1 energy budget only if it keeps the type proportions. Its celltype tables inform the S2-versus-S3 and S3-versus-S4 contrasts below.
- **D17: on IFN-alpha at K40 the malignant axis does not need malignant cells.**
  - Normal-only bases already carry the IFN-alpha discriminative axis.
  - Removing all malignant cells moves the axis more than random drops of the same size, but little in absolute terms.
  - Patient-equal weighting rotates PC8 without lowering tuned transfer. P01 has the largest leverage.
  - Sources: devlogs 2026-10-08_d17-basis-variant-pilot and 2026-10-08_d17-composition-matched-controls.
  - **Expectation from D17:** on IFN-alpha at K40, the M1 effect is small. Panels and K at which the discriminative direction is not already a leading factor are where M1 can show; hence K10 below.
- **D15 and D15b: count at a fixed ratio only.**
  - D15 subsampled all training cells label-stratified (the count axis at a fixed ratio) at fixed C, so it is confounded by the entry threshold (devlog 2026-10-02_d15-cell-count-learning-curve). W6.2 reruns it with C retuned.
  - D15b varies whole patients (devlog 2026-10-08_d15b-full-design).
  - Neither varies the ratio at a fixed malignant count. §5.3 does.

### 2.5 The metadata fact that makes M3 plausible

- In the current cohort's cell metadata, malignant cells concentrate in a few predicted progenitor types, while the most abundant healthy types carry almost no malignant cells. This was checked against `predicted.annotation` × `CN.label` on 2026-10-09; the W8.4 inventory tabulates it per patient for the new cohort.
- Proportional downsampling keeps that imbalance of types. A type-equalizing scheme moves weight toward the healthy cells that sit where malignant cells sit.

## 3. Hypothesis, alternatives and neighbours

- **H-S2-09** (statement in `research/hypotheses.md`): both the malignant training-cell count and the healthy-to-malignant ratio limit held-out transfer. Stratified healthy downsampling with C retuned per condition raises held-out lift through Stage 1 geometry or negative-class composition, not through the n-dependent penalty.
- **Alternatives** the design must exclude:
  - ALT-NREG (new): the gain is a point on the full-data regularization path, reached because n changed;
  - ALT-PEN: fixed-C L1 artifacts;
  - ALT-PREV: the gain sits only in the highest-prevalence patients;
  - ALT-CT: the gain is a held-out cell-type mix effect, read through within-type lift;
  - ALT-TRANS: the gain appears only when the held-out patient is in the basis;
  - ALT-SEL: the best scheme and fraction were picked in-sample.
- **Neighbours, not the same question:**
  - H-S2-04 asks whether the probe reads donor identity (a shortcut); this plan asks whether fewer healthy cells help.
  - H-S1-05 asks whether patients shape the loadings; the S4 arm and the M1 read-outs extend it.
  - H-S2-06 covers the learning curves (W6).

## 4. Cohort, units and fixed rules

- **Cohort.** As decided under W8.5 (`integration_plan.md` §4; devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs):
  - patients with no malignant cells are excluded from every new run (A-20261009-14);
  - Arm D, like every new full run, keeps patients with at least 100 malignant cells and both classes (A-20261010-07): P02, P03, P04 and P09 in the current cohort, plus each new patient that meets the rule. Patients below it are left out of every step (the reading the W8.2 lock confirms);
  - with four current patients, each outer fold has three training patients for inner-LOPO tuning until new patients join (counts in `integration_plan.md` §4.1).
- **Two arms.**
  - Arm D-L: LOPO over the supported patients of the extended cohort (original plus new patients that meet the rule).
  - Arm D-X: external evaluation of the frozen recipe (§5.6) on the new patients.
- **Held-out patient.** All its cells are scored; they are never downsampled and never relabelled, so its prevalence π is the same in every condition.
- **No label leakage through the fit set.** Deciding which cells are "healthy" uses labels, so only *training* patients are downsampled.
  - In the transductive Stage 1 scope, the held-out patient's cells enter the fit in full and label-blind, as in the canonical design.
  - In the inductive scope they are not in the fit.
  - Downsampling the held-out patient's healthy cells inside the fit set would use its labels, and is not allowed.
  - Note that the downsampled fit set itself is chosen with training labels: Stage 1 is then label-informed for the training patients (like D17's normal-only bases). Each result states this.
- **Healthy cells** are normal cells called donor-derived. Recipient-healthy normal cells are always kept and counted (ALT-DONOR; `integration_plan.md` §3).
- **Strata.** Training patient × `predicted.annotation` × pseudotime bin.
  - Pseudotime bins: quartiles of `predicted.pseudotime` within each predicted type, computed over the training healthy cells, merged until each bin meets the per-stratum floor.
  - If the new cohort lacks `predicted.pseudotime`, types only.
  - Per-stratum floor: as in D10 (`MIN_DONOR_CELLS_PER_STRATUM` in `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py`).
- **Representations** (aligned with the primary family of `integration_plan.md` §7):
  - IFN-alpha, allograft rejection, apoptosis and TNF-alpha/NF-kB;
  - PCA and FA at K10 and K40. K10 tests the M1 prediction; K40 is the atlas setting;
  - the direct-gene probe of each panel, in the Stage-2-only placement and the count × ratio grid (direct genes have no Stage 1);
  - FactoSig and factosig_promax only after W2.2, and only as exploratory.
- **Stage 2.** Balanced class weights, train-fold standardization, L1 and L2 always (R5); EN optional (§12).

## 5. The factorial

### 5.1 Sampling schemes for the healthy training cells

| Scheme | Rule | What it isolates |
|---|---|---|
| S0 | none (all healthy training cells) | the reference |
| S1 | unstratified random fraction of the healthy training cells | n and ratio, plus maximal sampling noise |
| S2 | proportional-stratified: the same fraction in every stratum, with the floor (the user's "representative" scheme; D10's scheme plus pseudotime bins) | n and ratio at preserved composition |
| S3 | type-equalizing: a per-type cap within each patient, so abundant types are thinned more than rare ones; the cap is set to keep the same total number of healthy cells as S2 | composition (M3): weight moves toward rare types and hard negatives |
| S4 | patient-capped: a cap on each training patient's healthy cells, same total as S2; thins the largest patients (P09- and P01-like) most | patient mix in the fit (the user's P09 point; D17's P01 leverage) |
| S5 (optional) | geometric sketching to the same total | coverage-preserving composition; exploratory |
| SW (added 2026-10-10) | no cells dropped; Stage 2 sample weights per training patient and class, so that each training patient's malignant cells sum to the same weight and likewise its healthy cells, multiplied into the balanced class weights; Stage 2 only (the Stage 1 fit is unweighted) | composition (M3) through the labels, with n_train unchanged, so no M2 shift and no M4 noise: the re-weighting target the user chose (A-20261010-08) |

**Matched-n rule.** At each retention level, S1-S4 keep the same number of healthy training cells in each outer fold. M2 is then equal across them, so their differences are composition (M3) or noise (M4). SW keeps every cell, so it is compared with S0 directly; it has no retention level and no seeds.

### 5.2 Placement × Stage 1 scope

| Placement | Stage 1 basis | Stage 2 training cells |
|---|---|---|
| Stage 2 only | the full basis of the scope | downsampled |
| Stage 1 only | refit on the downsampled fit set | all training cells, projected onto that basis |
| Both | refit on the downsampled fit set | the same downsampled cells |

- **Primary placement: Stage 2 only** (since 2026-10-10). The user decided that re-weighting targets the Stage 2 labels (A-20261010-08), so the Stage-2-only placement, where only who the probe learns from changes, is read first, and SW runs only there. Stage 1 only and Both are kept as the decomposition (§6.5) and the geometry read-outs (§6.2): they say whether a gain needs the basis to change.
- **Scope** (every Stage 1 refit is done in both):
  - transductive: the training patients' kept cells plus all of the held-out patient's cells, label-blind;
  - inductive: the training patients' kept cells only, with training-only standardization, then the held-out patient is projected.
- The inductive scope is primary for the recipe (§5.6). The transductive scope answers the user's P09 question directly.
- In the Stage-2-only placement, "the full basis" is the scope's S0 basis.
- Bases are fitted with `fit_basis` (`stage1_refit.py`) and compared with the S0 basis of the same scope and fold.

### 5.3 Count × ratio grid: "not only the count, also the ratio"

- **Grid.** Malignant retention f_M ∈ {1, 1/2, 1/4} × healthy retention f_H ∈ {1, 1/2, 1/4, 1/8}.
  - Healthy cells are thinned by S2.
  - Malignant cells are thinned within each training patient, with a floor, so every training patient keeps its share of malignant labels. Patients at the floor keep all their malignant cells and are flagged.
- **The core 2 × 2:**
  - (1, 1): the baseline;
  - (1/2, 1/2): the count halves at a fixed ratio;
  - (1, 1/2): the ratio halves at a fixed count;
  - (1/2, 1): the count halves and the ratio doubles.
- **Placements:** Stage 2 only, and Both (inductive).
- **Model.** For each held-out patient h:
  - lift = a_h + b · log2(malignant training count) + c · log2(healthy-to-malignant training ratio) + noise, fitted over grid cells and seeds;
  - b is the count effect at a fixed ratio and c the ratio effect at a fixed count. They are identifiable because f_M and f_H vary independently;
  - b and c are summarized over patients as in §6.1, with patient-cluster bootstrap intervals.
- **Relations.** The diagonal f_M = f_H is D15's label-stratified design, now with C retuned (as W6.2 plans). The f_H = 1 column varies the malignant count at full healthy depth, which is the cell-level counterpart of W6.3.

### 5.4 C policy (R5) and the regularization controls

- **Primary policy.**
  - Penalties L1 and L2.
  - C tuned per (condition, seed, outer fold) by inner LOPO over the supported training patients, with `tune_c_inner_lopo` (the W3.1 contract) and its criterion, the median inner lift.
- **Inner folds mirror the outer fold.**
  - Inner-training patients are downsampled with the same scheme and fraction.
  - The inner validation patient keeps all its cells.
  - `tune_c_inner_lopo` currently validates on whatever cells it is given, so W8.6 adds full-cell inner validation as an option, defaults unchanged (R10).
- **Extended grid.** The runner grid is extended two decades below its floor (`_runner_c_grid` with a lower log10 bound) for every condition, S0 included. The full-data comparator can then reach every C · n_train that a downsampled condition reaches. Any tuned C at a grid bound is flagged (`c_at_grid_lower_bound`).
- **Diagnostic arms** (not nested, and labelled as such):
  - the atlas C, which should reproduce the D10 collapse and shows M2 at fixed C;
  - C scaled to keep C · n_train constant (`c_scaled_to_n`);
  - the full-data C path: S0's held-out lift at every grid C. Each downsampled result is placed on that path; if some full-data C matches it, the gain is ALT-NREG.

### 5.5 Retention levels and seeds

| Part | Schemes and healthy retention f_H | Seeds per stochastic condition |
|---|---|---|
| Stage 2 only, DR, inductive basis | S2 at 1/2, 1/4, 1/8, 1/16; S1, S3, S4 at 1/4 and 1/16 | 5 |
| Stage 2 only, DR, transductive basis | S2 at 1/4 and 1/16 | 5 |
| Stage 1 only and Both, DR, both scopes | S1, S2, S3, S4 at 1/4 and 1/16 | 3 |
| Count × ratio grid | §5.3 | 5 (DR); 3 (direct genes, core 2 × 2 only) |
| Direct genes, Stage 2 only | S2 at all four levels; S3 at 1/4 and 1/16 | 3 |
| SW, Stage 2 only, DR (both scopes) and direct genes | SW (no retention level) | 1 (deterministic) |

- Each patient's lift is averaged over seeds before any patient summary (Option B).
- The per-patient seed SD is kept as the M4 read-out.
- Masks are drawn once per (scheme, retention, seed, outer fold) and reused by every placement and penalty, so the placements compare the same kept cells.

### 5.6 The recipe (confirmatory arm)

- **A recipe is a procedure, not a hand-picked fraction.**
  - In each outer fold, the inner LOPO picks (scheme, f_H) from {S0, SW} ∪ ({S2, S3} × {1/4, 1/16}) jointly with C.
  - The choice is made from the stored inner criteria, so it needs no extra fits.
  - Proposed placement (2026-10-10, following the Stage 2 target): Stage 2 only, on the inductive scope's full basis. The same procedure in the Both placement (inductive scope) is pre-declared as the secondary recipe, because the user's original question is about the Stage 1 fit (A-20261009-12); it is read with the same rules and reported beside the primary. The lock confirms both (§12 item 2).
- **Arm D-L.** The recipe's outer lift is compared, patient by patient, with S0 under the same tuning.
- **Arm D-X.** The recipe is frozen on the original cohort and scored on the new patients exactly as Arm P scores its frozen probes, with the same metrics (`integration_plan.md` §8).
- **Limitation, as in D17:** inner tuning shares the outer fold's basis.

## 6. Read-outs: what explains a gain

### 6.1 Outcome (every condition)
- Per-patient lift (primary), normalized lift, AUROC and pooled AP.
- Median and P20 lift over held-out patients. Pooled AP, median lift and P20 lift are always quoted together (R6).
- The paired difference versus S0 per patient (same scope, penalty and tuning), with a patient-cluster bootstrap interval (W4.1 / D22 definitions).

### 6.2 Stage 1 geometry (M1; Stage 1 placements)
- **Malignant share of score energy.** The D16 estimand: the malignant cells' share of the factor score energy about the cohort mean, divided by their cell share. It is computed on all cohort cells projected onto the basis, so conditions are comparable. The malignant fraction p of the fit set is reported beside it.
- **The discriminative factor.** The factor with the largest all-cell standardized class-mean gap: its rank, its share of explained or common variance, and the gap itself (the §2.3 signatures).
- **Movement of the reference axis** versus the S0 basis of the same scope and fold (D17 metrics, `stage1_refit.py`):
  - axis loss = 1 − many-to-one R² of the S0 reference axis on the variant's factors;
  - matched |r|;
  - principal angles (top 8 and all K).
- **Per-patient share of score energy.** Does the largest patients' share fall under S4?

### 6.3 Stage 2 (M2, M5)
- The tuned C, C · n_train, the grid-bound flag and the penalty chosen.
- Entry margins at the tuned C (`l1_entry_margins`; D19), the identity of the largest-margin factor, and the nonzero count.
- The cosine between the condition's probe and the S0 probe in gene space (activation patterns, as W5.2 defines them). This separates "same direction, different strength" (M2) from "a different direction" (M1 or M3).

### 6.4 Composition (M3)
- The retained healthy mix by type and pseudotime bin.
- **Hard-negative share:** the share of retained healthy cells that fall in the types holding the training malignant cells, defined from training labels only.
- Within-type held-out lift, for types with enough held-out malignant cells (ALT-CT).
- D14-style overlap of the retained versus the full healthy score distributions (§2.4).

### 6.5 Decomposition table
For each scheme and retention level, with the seed spread:
- gain(Stage 1 only);
- gain(Stage 2 only);
- gain(Both);
- interaction = Both − Stage 1 only − Stage 2 only.

## 7. Pre-specified success criteria, attribution and falsification

**H-S2-09 is supported (descriptive) only if all six hold:**
1. **Gain.** The recipe's (§5.6) median paired lift gain over S0 is above zero, with its patient-cluster bootstrap 95% interval excluding zero, under tuned L1 and tuned L2. Its P20 paired difference is not below the non-inferiority margin fixed at the lock (§12 item 4). Pooled AP is reported beside both.
2. **Ratio.** In the §5.3 grid, the ratio effect c at a fixed count is positive with its interval excluding zero in the Stage-2-only or the Both placement, and the result names the placement: in Stage 2 only it reads "ratio through the training labels", only in Both it reads "ratio through the Stage 1 fit" (with balanced weights and tuned C, §2.3 predicts no Stage-2-only ratio effect from S2 thinning, so a Stage-2-only effect is the stronger finding). Otherwise the result reads "count only".
3. **Not a penalty artifact.** The gain survives the extended grid, and no point on the full-data C path matches it (§5.4). Otherwise ALT-NREG.
4. **Not transductive only.** The gain is present in the inductive scope.
5. **Not prevalence only.** The gain is positive in at least half of the held-out patients outside the highest-prevalence group. Otherwise ALT-PREV qualifies the claim. Under the 100-cell rule the held-out patients outside P02, P03 and P04 are P09 and the qualifying new patients; with P09 alone the criterion is reported but not read.
6. **External direction.** On the new patients (Arm D-X), the recipe's median paired gain is positive. The exact sign test is reported; with few new patients this is direction only.

**Attribution of the "why"** (each rule fixed now):

| Pattern | Reading |
|---|---|
| Most of the gain (at least half of Both) appears in Stage 1 only; it tracks the malignant score-energy share and the rank of the discriminative factor; it is larger at K10 than at K40 | geometry (M1) |
| The gain appears in Stage 2 only under S3 or S4 but not under S1 or S2 at matched n; the probe direction changes; the hard-negative share rises | composition (M3) |
| S1 equals S2 at matched n | stratification adds nothing beyond n; the "representative subsample" argument is not what produces the gain, although S2 stays the safer default |
| The gain appears only at fixed C, or only against the grid-floor-limited comparator | penalty (M2; ALT-PEN, ALT-NREG): H-S2-09 not supported |

**H-S2-09 is falsified (not_supported) if,** with C tuned on the extended grid and L2 alongside, no scheme × retention × placement beats S0 beyond its seed spread in the inductive scope. That would be the D10 L2 result, generalized. It is still a paper result (§10).

**What the intervals mean.** With the four current patients that meet the 100-cell rule plus the qualifying new patients, nothing here is significance in the inferential sense. The interval rules are decision rules declared in advance and read descriptively (THEME §6).

## 8. Compute estimate and screen plan

**Unit costs.** These are taken from existing timings and are replaced by the smoke run's timing model:
- inner-LOPO tuning costs about 7 CPU-seconds per representation × fold × penalty at full n with K ≤ 40 (devlog 2026-10-08_d19-regularization-contract), less when n is smaller;
- a PCA/FA basis of a focus-sized panel fits in seconds (devlog 2026-10-08_d17-basis-variant-pilot);
- direct-gene probes use saga and are several times slower per unit (same D19 entry).

**Size.** F is the number of held-out patients under W8.5: at least 100 malignant cells, which is 4 in the current cohort plus each qualifying new patient. The CPU-hours below were estimated on 2026-10-09 at F = 7; they scale linearly with F. SW adds about one S0-sized condition per representation and fold. There are 16 DR representations (4 panels × PCA/FA × K10/K40) and 4 direct-gene probes. Units scale linearly with F.

| Part | Conditions per (representation, fold) | Rough CPU-hours at F = 7 |
|---|---|---|
| Stage 2 only, DR, both scopes | about 60 | about 30 |
| Stage 1 only and Both, DR, both scopes | about 25 bases per scope; about 100 Stage 2 units | about 40 (bases a few) |
| Count × ratio grid, DR (Stage 2 only, and Both inductive) | about 56 per placement | about 45 |
| Direct genes (Stage 2 only, plus the core 2 × 2) | about 30 | about 15 |
| Recipe | selection over the stored inner criteria | negligible |
| **Total** | | **about 100-150 CPU-hours, or roughly 8-13 hours of wall time at 12 workers; IFN-alpha alone is about a quarter** |

**Phases.** Every phase uses `--resume` (R2), 12 workers with one BLAS thread each (R1), and a NEWEXP slug (R3).
- **Phase 0 (agent shell, under 5 minutes).**
  - Run the unit tests.
  - Run a smoke on the cohort with labels permuted within each patient, written to a scratch `--out-root`. The timing model is then real, but no real lift is seen before the lock.
- **Phase 1** (`screen -S W8_hds_ifna`): IFN-alpha, every part.
- **Phase 2** (`screen -S W8_hds_focus`): allograft rejection, apoptosis, TNF-alpha/NF-kB.
- **Phase 3** (`screen -S W8_hds_grid`): the count × ratio grid.
- **Phase 4** (`screen -S W8_hds_external`): Arm D-X, after the W8.2 lock and the Arm P reconstruction gate.

**Command template.** The runner is planned (W8.6); its flags are proposals until it exists:

```
screen -S W8_hds_ifna
export SC_EXPERIMENT_ID=<new_experiment_id>
bash scripts/mrd_stage0_2/workstreams/W8_new_cohort/run_healthy_downsampling.sh \
  --experiment-dir "$SC_EXPERIMENT_ROOT/$SC_EXPERIMENT_ID" \
  --plan scripts/mrd_stage0_2/workstreams/W8_new_cohort/healthy_downsampling_plan.md \
  --panels ifna --parts masks,bases,stage2,align,report \
  --slug healthy_downsampling_v1 --resume --n-jobs 12
```

**Verify each phase:**
- the unit-status file shows no missing units;
- the S0 anchor reproduces the Arm I inductive-LOPO tuned lifts for the same representations (a D17-style anchor gate);
- the manifest records this plan's sha256, equal to the one in the lock entry.

**Before Phase 1:**
- a campaigns.csv `planned` row for `healthy_downsampling_v1` (relaunch_policy `resume_only`, the screen names above; R4);
- a reserved D-ID: `python tools/research_index.py reserve-id --workstream W8 --slug healthy_downsampling_v1 --title '...'`;
- the runner committed (R11).

## 9. Implementation (W8.6)

- **Code (planned):**
  - `scripts/mrd_stage0_2/workstreams/W8_new_cohort/run_healthy_downsampling.py` and its `.sh` screen wrapper;
  - parts: `masks`, `bases`, `stage2`, `align`, `report`;
  - `--resume` per unit (empty or corrupt files are redone), `--merge-only` and `merge_failed.json` (R2), `--out-root` for smoke runs, `--experiment-dir`, explicit `--n-jobs`.
- **Reuse, imported via `add_study_module_paths(workstreams=("W3_dr_value_regularization", "W2_stage1_basis_variants", "W1_patient_aggregation"))`:**
  - `regularization_contract.py`: `tune_c_inner_lopo`, `l1_entry_margins`, `c_scaled_to_n`, `c_at_grid_lower_bound`, `_runner_c_grid`;
  - `stage1_refit.py`: `fit_basis`, `Stage1Basis`, `many_to_one_r2`, `principal_angles_deg`, `align_scores`;
  - `factor_patient_matrix.py`: `allocation_table`, for score energy;
  - D10's stratified mask (`donor_keep_mask` in the donor runner), as the reference implementation of S2 (types only).
- **Additive changes to shared code** (named in the W8.6 refactor entry, defaults unchanged):
  - a full-cell inner-validation option for `tune_c_inner_lopo`;
  - a per-cell weight multiplier on top of the balanced class weights, in the contract's fit and in `tune_c_inner_lopo`, for SW (default none);
  - nothing else.
- **Tests** (`tests/test_healthy_downsampling.py`, loading the runner by file path):
  - masks never touch held-out cells, never touch malignant cells unless f_M < 1, and keep recipient-healthy cells;
  - S1-S4 hit the same n at each retention level;
  - S2 preserves composition;
  - masks are deterministic per seed;
  - the inner-validation option leaves `tune_c_inner_lopo`'s default behaviour unchanged;
  - SW weights sum to the same value per training patient within each class, and all-one weights reproduce the default fit exactly;
  - a toy case in which random healthy downsampling under balanced weights equals a C rescale up to noise.
- **Artifacts (NEWEXP-relative; R3):**
  - `analysis/scorecards/stage2_diagnostic/healthy_downsampling_v1/`, with matching `figures/` and `cache/` dirs, plus the findings, manifest and artifact ledger;
  - refit bases under `stage1_dr_variants/healthy_downsampling_v1/`;
  - the mask table with per-stratum counts, as D10 wrote.

## 10. As a paper section, or a companion short paper

- **Working title** (the user's catch line, A-20261009-12): *"Machine learning in MRD single-cell data: just downsample the healthy cells?"* It stays a question until W8.7 reports.
- **Where it lives:** `research/paper/OUTLINE.md` arc R12 and the HDS panels in `research/paper/FIGURES.md`. Main-paper section or companion note is the user's choice (W8.8).
- **The "why" is the spine.** §2.3 (balanced weights plus rank metrics leave only geometry, composition, penalty and noise) is the section's central argument, and the factorial maps onto it one-to-one.
- **Section outline:**
  1. MRD single-cell data are extremely imbalanced, and the healthy-to-malignant ratio varies over orders of magnitude between patients. Downsampling healthy cells is common practice, but its effect is rarely decomposed.
  2. A box: what downsampling can and cannot change for a balanced-weight probe judged by AP.
  3. The count × ratio surface.
  4. Where the gain comes from: the placement decomposition, the geometry read-outs (K10 versus K40), and the composition read-outs.
  5. The fixed-C trap: why naive downsampling looks harmful (D10), and what retuning C does.
  6. External replication on the new patients.
  7. A practical recipe, or a practical warning.
- **Two narratives fixed in advance:**
  - **if supported:** "downsample the healthy cells before the decomposition, stratified by type, and retune C; it helps because the malignant direction enters the basis (M1) or because the negatives become the cells that matter (M3)";
  - **if not:** "with balanced class weights and tuned C, downsampling healthy cells is a change of C; its apparent effects come from fixed-C L1 and the grid floor".
  Either is a useful methods result for machine learning on MRD single-cell data.
- **Claim boundary:** descriptive, cell-level label transfer; inductive scope primary for the recipe; not deployment and not recurrence risk (THEME §6, AGENTS R6).
- **Literature to verify before citing** (W9.3 addendum; cited from memory, not yet checked):
  - case-control sampling and prior correction in logistic regression (Prentice and Pyke, Biometrika 1979; King and Zeng, Political Analysis 2001);
  - local case-control subsampling (Fithian and Hastie, Annals of Statistics 2014);
  - learning from imbalanced data (He and Garcia, IEEE TKDE 2009);
  - geometric sketching of single-cell data (Hie et al., Cell Systems 2019).

## 11. Risks and mitigations

| Risk | ALT | Mitigation |
|---|---|---|
| Forking paths: many schemes × levels × placements, then picking the best | ALT-SEL | nested recipe (§5.6); full curves reported; decision rules fixed here |
| The C-grid floor makes S0 look worse than it is | ALT-NREG | extended grid for every condition; full-data C path |
| A refit changes factor order, so factor indices stop meaning the same thing | ALT-ROT | compare through the reference axis (many-to-one R², principal angles), never by index |
| Small patients run out of malignant cells when f_M < 1 | ALT-PREV | per-patient floor; floored patients flagged and reported separately |
| Labels leak into the basis through the fit set | — | §4 rule; tests in W8.6 |
| The new cohort lacks `predicted.pseudotime` or a donor/recipient call | ALT-DONOR | types-only strata; recipient flag from the inventory; otherwise those patients are excluded from Arm D and listed |
| Rare healthy types vanish under thinning | ALT-CT | per-stratum floor; retained-versus-full overlap check (§6.4) |
| Compute exceeds the estimate | — | phased screens with `--resume`; the smoke timing model sets the phase sizes |

## 12. Open choices for the lock (the user's)

1. **Rehearsal.** Run Phase 1 (IFN-alpha) on the current cohort under the 100-cell rule (P02, P03, P04, P09) before the new data arrive, as a labelled rehearsal; or wait, as the 2026-10-09 decision says (the default is to wait). W4.6 (D10 rerun with condition-specific C) overlaps the Stage-2-only S2 arm, so whether W4.6 runs now or folds into W8.7 is the same choice.
2. **The recipe's placement and scope.** Proposed since 2026-10-10: Stage 2 only on the inductive scope's basis as primary (the decided Stage 2 target), with Both, inductive, as the pre-declared secondary (§5.6). Before 2026-10-10 the proposal was Both, inductive.
3. **Design size.** The retention levels and seeds (§5.5), and whether S5 (sketching) and the pseudotime bins are included.
4. **The P20 non-inferiority margin** for §7 item 1.
5. **Penalties.** Whether EN joins L1 and L2.
6. **Representations.** The full primary family, or IFN-alpha first and the focus panels after.
7. **Lock.** Together with `integration_plan.md` (proposed), or separately.
