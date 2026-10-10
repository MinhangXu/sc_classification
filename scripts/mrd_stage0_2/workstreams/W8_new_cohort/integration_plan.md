# W8 integration and external-validation plan for the new MRD cohort (pre-registration draft)

> **Status: DRAFT (2026-10-08). Not locked.** No new-cohort data were on disk when this was written.
> - **Lock (W8.2).** The plan becomes binding through a `type: decision` devlog entry in workstream W8. That entry records:
>   - the sha256 of this file and of `frozen_config_v1.json` (§7);
>   - the repo commit;
>   - the user's resolution of each item in §14.
> - **Allowed before the lock:** the inventory in §2, which uses counts and metadata only. No probe score, AP or AUROC may be computed on new-cohort cells until the lock entry exists.
> - **Changes after the lock** need a new decision entry with `supersedes:`. Any analysis not listed here is labelled *exploratory (post hoc)* wherever it is reported.
> - Asks: A-20261008-12, plus A-20261008-13 (Option 1.1) and A-20261008-27, which depend on this cohort. Literature basis: W9 Review B §1, §3 and §5 (external-validation protocol, routed to W8.2).
> - Paths: `EXP` is the canonical experiment dir and `NEWEXP` is the new experiment dir (§5). Artifact paths are relative to the named dir; code paths are repo-relative.
> - **Update 2026-10-09** (the user's reading notes on report v1, `research/asks/2026-10-09_report-reading-notes.md`; devlog 2026-10-09_report-reading-notes):
>   - **Decided by the user:** new runs exclude patients with no malignant cells (A-20261009-14; §4.0).
>   - **Open, for the user:** which patients are held out and summarized, and which only train (A-20261009-11; §4.1, §14 item 9). Decided on 2026-10-10, below.
>   - **Added:** healthy-cell downsampling, pre-registered in `healthy_downsampling_plan.md` and run as Arm D (§8) when the new data arrive, as the user decided (A-20261009-12).
> - **Update 2026-10-10** (`research/asks/2026-10-10_protein-and-decisions.md`; devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs). The user framed both decisions as "decisions relevant for runs when we have the extra patient data":
>   - **Decided by the user:** new full Stage 0-2 runs keep patients with at least 100 malignant cells and both classes (A-20261010-07; §4.1). The 10-cell support rule of `research/study_config.json` stays for every re-analysis of the current atlas.
>   - **Decided by the user:** re-weighting targets the Stage 2 labels, not the Stage 1 basis (A-20261010-08; W2.3 closed). It runs inside Arm D (§8; `healthy_downsampling_plan.md` §5.1-§5.2).
>   - **Still for the lock:** confirm that patients below 100 malignant cells are left out rather than kept as train-only (§14 item 9), and Arm P's pre-fitted probes (§14 item 10).

## 1. Purpose, estimand, claim boundary

**Purpose.** This is the study's first test on patients who took no part in any choice: they did not enter Stage 0 panel selection, Stage 1 fitting, Stage 2 training, or the selection of method, K or C. It is the only clean external check of the shared-model claims, and of the winner's-curse risk in the in-sample D1/D9 selections.

**Primary estimand.** For each new supported patient h, the lift of a frozen probe trained on the original cohort:
- lift_h = AP_h − π_h, where π_h is h's malignant prevalence;
- a constant score has no skill and gets AP = π.

These are summarized across new supported patients by the median and the P20 (Option B: one value per patient).

**Questions served.**
- **Q1:** does label transfer replicate in new patients? (H-S2-01, H-S1-04, H-S1-06)
- **Q2:** per-patient intervals, and more supported patients for patient-level inference (H-S2-06)
- **Q3:** frozen DR-vs-direct-gene pairs evaluated out of sample (informs H-S1-02)
- **QI:** the IFN-α threshold prediction (H-S2-08)
- **Q2/Q3, added 2026-10-09:** whether the healthy-to-malignant ratio limits transfer separately from the malignant count, and why downsampling healthy cells would help (H-S2-09; Arm D, `healthy_downsampling_plan.md`)

**Claim boundary (AGENTS R6).** Results are descriptive. They describe label transfer to new patients whose labels come from the collaborator's pipeline. They are not clinical deployment, not recurrence risk and not causal. Each result names the arm (§8); the Arm P basis is "frozen transductive basis of the original cohort, new patients projected".

## 2. Inventory (W8.3 code, run at W8.4)

**Allowed looks before the lock:** counts and metadata. Never predictions.

Per patient × sample, record:
- cells by `Tech`, `timepoint_type` and `CN.label` (cancer / normal / unassigned);
- donor vs recipient calls within normal cells (recipient-healthy) and within cancer cells;
- `Days from SCT`, `Days.from.Relapse`, relapse status, and sequencing batch or run;
- label-evidence type per patient (§3), donor and recipient sex, and whether they match;
- `predicted.annotation` composition (cell-type mix);
- gene coverage against the canonical universe, overall and per panel (covered genes canonical vs new);
- ADT panel overlap with the original object;
- support status under the study rule (§4);
- achievable interval width per patient (§9).

**Outputs** go to `NEWEXP/analysis/scorecards/stage2_diagnostic/new_cohort_inventory_v1/`:
- `new_cohort_inventory_v1_support_table.csv`
- `new_cohort_inventory_v1_panel_coverage.csv`
- `new_cohort_inventory_v1_findings.md`
- `new_cohort_inventory_v1_manifest.json`
- `new_cohort_inventory_v1_artifact_ledger.csv`

These follow AGENTS R3.

**Script (planned, W8.3):** `scripts/mrd_stage0_2/workstreams/W8_new_cohort/run_new_cohort_inventory.py`. It reads the new AnnData in backed mode (obs and var only), never opens the canonical raw AnnData (R8), and implements `--resume` (R2).

## 3. Label harmonization

**Target definition.** It is identical to the canonical cohort cache, built by the cache step in `scripts/mrd_stage0_2/notebooks/stage0_2/_build_deg_hvg_lowsupport_notebook.py`:
- cells are restricted to `Tech == CITE` and `timepoint_type == MRD`;
- y = 1 for `CN.label == cancer` and y = 0 for `CN.label == normal`;
- `unassigned` cells are excluded.

**Required provenance per new patient.** The evidence behind `CN.label`, using the same vocabulary as the W1.4 (D25) provenance column for the original cohort:

| Evidence | Detail |
|---|---|
| Souporcell | singlet status |
| Numbat CNV | donor reference |
| mtDNA variants | |
| immunophenotype | e.g. CD5/CD56 |
| cluster reference | |

The W8.1 email asks for this.

**Harmonized standard (from W9 Review B §1).**
- Keep Souporcell singlets only.
- Call malignant cells with Numbat against a donor reference.
- Add mtDNA variants where available.
- Patients labelled by a different pipeline or rule form a *non-harmonized* stratum. Primary summaries use harmonized patients only (§14, item 4); a sensitivity summary includes everyone.

**Donor/recipient handling.**
- Recipient-normal cells stay in the negative class, as in the canonical cohort, but are flagged and counted.
- A secondary readout scores recipient-normal cells separately with the frozen probes. This mirrors W1.4 and addresses ALT-DONOR.
- The donor/recipient sex-match table from W1.4 is extended to the new patients.

**Label circularity.** If a patient's labels rely on an immunophenotype marker whose gene sits in a panel (for example, CD5/CD7 with allograft rejection), that (patient, panel) pair is flagged as label-circular (ALT-LABEL). It is reported but excluded from that panel's primary summary.

## 4. Inclusion and support rule

### 4.0 Decided by the user (2026-10-09): no patients without malignant cells in new runs

> "In a new run, let's definitely subset the patients to do the whole stage0-2 pipeline, let's definitely not include patients that're just in the cohort and contribute no malignant cells at least." (A-20261009-14; recorded in devlog 2026-10-09_report-reading-notes)

- **Scope.** Every new run:
  - the Stage 1 fit set;
  - the Stage 2 training negatives and the C-tuning folds;
  - Arm I and Arm D (§8);
  - the optional joint grid (§11);
  - the W2.5 patient-subset pipeline.
- **Patients excluded.**
  - In the current cohort: P08, P10, P11 and P12, which have no malignant cells (checked against `research/study_config.json` and the cell metadata on 2026-10-09). Together they hold 10,638 cells, all normal.
  - New patients without malignant cells are excluded the same way; they are listed in the inventory only.
- **Mechanism.** The opt-in `--patients` filter (W8.3), defaults unchanged.
- **Consequences.**
  - Training n changes, so C is retuned (R5).
  - The Stage 1 fit set changes, so new-run bases are not cell-for-cell comparable with the canonical transductive bases. Any comparison names both cohort definitions.
- **Reading to confirm (§14 item 10).** The decision governs *new runs*. Arm P's stored canonical bases (fit on all 13 patients) are not a new run and stay as they are. Arm P's pre-fitted probes (§7) are fitted at lock time, so whether they also drop the zero-malignant patients' cells is the user's choice:
  - **keep:** the probes are trained exactly as the canonical LOPO probes were;
  - **drop:** consistent with the decision, but the Stage 2 training set then differs from every canonical number.
- "At least" left the patients with a few malignant cells to §4.1, decided on 2026-10-10.

### 4.1 Decided by the user (2026-10-10): new full runs keep patients with at least 100 malignant cells (A-20261009-11, A-20261010-07)

> "We got 4 patients passing the 100 malignant cell rule. P02, P03, P04, P09. We need to think if P01, P06 and P07 are still relevant to use. For some reason P07 still does pretty decent lift in the IFN-alpha case." (A-20261009-11)

> "Let's do the >=100 malignant cells." (A-20261010-07; devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs)

**The rule.**
- **(a) T_eval = 100.** A patient is in a new full Stage 0-2 run, as a fitted, trained and held-out, summarized patient, only if it has at least 100 malignant cells and both classes. In the current cohort: P02, P03, P04 and P09. A new patient qualifies by the same rule, from the inventory counts (§2), before any probe score is computed. In this plan, "new supported patient" means a new patient that meets this rule.
- **(b) Patients below it are left out** of every step of a new full run, like the zero-malignant patients (§4.0). This is the reading of the decision together with A-20261009-14 ("subset the patients to do the whole stage0-2 pipeline"); the W8.2 lock asks the user to confirm it against the train-only option (§14 item 9).
- **What does not change.** The support rule in `research/study_config.json` (at least 10 malignant cells and both classes; AGENTS R7) stays, so existing results and any re-analysis of the current atlas keep the seven supported patients and their folds. The new rule is written into `frozen_config_v1.json` (§7) at the lock. A comparison between a new run and an existing result names both cohort definitions.
- **Arm P** keeps the stored canonical bases (§4.0); its summarized new patients follow (a); its pre-fitted probes' training set is §14 item 10.

The tables below are the information the user had when deciding; they stay as the record of what the rule implies.

**The patients who remain after §4.0.** These are metadata values from `research/study_config.json`, rendered in STATUS.md §0 and checked against the cell metadata on 2026-10-09. "Healthy per malignant" is (total − malignant) / malignant.

| Patient | Malignant | Total cells | Prevalence | Healthy per malignant | ≥10 | ≥20 | ≥100 |
|---|---|---|---|---|---|---|---|
| P03 | 758 | 2,327 | 32.57% | 2.1 | held out | held out | held out |
| P02 | 211 | 1,462 | 14.43% | 5.9 | held out | held out | held out |
| P04 | 102 | 1,128 | 9.04% | 10.1 | held out | held out | held out |
| P09 | 375 | 17,126 | 2.19% | 44.7 | held out | held out | held out |
| P06 | 44 | 3,259 | 1.35% | 73.1 | held out | held out | below |
| P07 | 22 | 2,185 | 1.01% | 98.3 | held out | held out | below |
| P01 | 45 | 11,860 | 0.38% | 262.6 | held out | held out | below |
| P05 | 8 | 4,359 | 0.18% | 543.9 | below | below | below |
| P13 | 4 | 5,774 | 0.07% | 1,442.5 | below | below | below |

**Two separate choices.**
- **(a) The evaluation threshold T_eval** (decided: 100): the malignant-cell count at which a patient is a held-out, summarized patient.
  - ≥10 is the current rule (`study_config.json`, AGENTS R7).
  - ≥20 is the D14 stratum parameter.
  - ≥100 is the cut the user named, and the Option 1.1 "top 4-5 patients" idea (A-20261008-13).
- **(b) The training role** of patients that have malignant cells but fall below T_eval (decided: excluded, to be confirmed at the lock):
  - **train-only:** their malignant labels and healthy cells stay in the Stage 1 fit and in Stage 2 training, but they are never summarized. This is how P05 and P13 are used today;
  - **excluded:** removed from every step, like the zero-malignant patients.

**What each threshold implies** (current cohort; each new patient adds to the matching row):

| | ≥10 | ≥20 | ≥100 |
|---|---|---|---|
| Held-out, summarized patients (LOPO folds) | 7: P01, P02, P03, P04, P06, P07, P09 | the same 7 (P07 has 22) | 4: P02, P03, P04, P09 |
| Below T_eval but with malignant cells, role set by (b) | P05, P13 | P05, P13 | P01, P05, P06, P07, P13 |
| Supported training patients per outer fold, i.e. inner-LOPO folds for tuning C | 6 | 6 | 3 |
| Prevalence range of the summarized patients | 0.38-32.57% | 0.38-32.57% | 2.19-32.57%: the three lowest-prevalence supported patients leave the summary |
| Highest-prevalence patients (P02, P03, P04) among the summarized | 3 of 7 | 3 of 7 | 3 of 4 |
| Where P20 lift sits (linear interpolation) | between the 2nd and 3rd lowest patient | same | between the lowest and the 2nd lowest |
| Smallest exact one-sided sign-test p, if every patient has lift > 0 | 1/128 | 1/128 | 1/16: cannot reach 0.05 |
| Cells left if (b) = excluded (from the 49,480 that remain after §4.0) | 39,347, with 1,557 malignant | same | 22,043, with 1,446 malignant |
| Comparability with existing results | the current rule; existing patient summaries stay on the same folds | identical patients here; differs only for new patients with 10-19 malignant cells | every existing patient summary (D1 atlas, D15b, D17, D19, report v1) uses 7 folds; comparing needs a recomputation on 4 folds or side-by-side reporting |

**Further consequences.**
- If (b) = excluded at ≥100, P01 also leaves the Stage 1 fit set, and the bases change. P01 is the patient with the largest leverage on the IFN-alpha axis (devlog 2026-10-08_d17-composition-matched-controls) and the most over-represented in score energy (devlog 2026-10-08_d16-factor-patient-matrix).
- With 3 inner folds, the inner-LOPO choice of C rests on fewer patients. D15b ran three- and four-patient training sets (devlog 2026-10-08_d15b-full-design).

**Evidence pointers on whether P01, P06 and P07 are still relevant** (numbers are in the cited entries):
- **P07.**
  - Positive IFN-alpha PCA40 lift at the atlas setting and under per-fold tuned C (devlog 2026-10-08_d19-regularization-contract; report v1, Figure 2).
  - Positive single-factor PC8 and FA7 lift (devlog 2026-10-08_d16-factor-patient-matrix).
  - Its residual PCA40 advantage over FA40 is a single-patient, penalty-dependent difference (D19).
- **P01 and P06.**
  - Within-patient DE genes, but weak held-out lift (devlog 2026-09-03_d12-d13-gsea-hvg-lowsupport; THEME §6 S14).
  - Near-zero patient-count gains (devlog 2026-10-08_d15b-full-design).
- **Across the atlas.** Lift above π reaches every supported patient in many representations, but strong lift concentrates in P02, P03 and P04 (devlog 2026-09-11_d1-atlas-783-d1e-d1f; report v1 section 08).
- **The substance of the decision.** The low-prevalence patients are the ones closest to a low-burden MRD setting, while the high-count patients carry most of the malignant labels. Which of these the patient summaries should represent is the choice in (a).
- **New patients.** Each new patient's support status and achievable interval width (§9) are known from counts before unblinding, so the threshold can be fixed with that information in hand.

**Where the decision is recorded.**
- The W8.5 `type: decision` entry: devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs.
- The study-wide support rule did not change, so `study_config.json` is unchanged (AGENTS R7).
- The rule for new full runs is written into `frozen_config_v1.json` (§7) at the W8.2 lock.

### 4.2 Other inclusion rules (unchanged)

- **Support rule.** Re-analyses of the current atlas keep the rule in `research/study_config.json` (`support_rule`: minimum malignant cells and both classes; AGENTS R7). New full runs use the 100-cell rule of §4.1 (W8.5, decided 2026-10-10; THEME §11, decision 6).
- **Patients below the chosen threshold** are reported descriptively and never enter lift summaries. Their training role follows §4.1 (b). Patients without malignant cells follow §4.0.
- **Unit of analysis: the patient.** All MRD CITE samples of a patient are pooled, as for P09 in the canonical cohort. Per-sample results are secondary.
- **Known patients.** Extra samples from P01–P13 are not new patients. They form a "new samples, known patients" stratum and are excluded from the external summaries.
- **Multiome-only data** is excluded from the primary analysis, because the canonical study used CITE MRD cells only. It may be analysed as an exploratory stratum.

## 5. Ingestion as a new experiment (W8.4)

**Raw file.**
- Lives under `$SC_DATA_ROOT` (umbrella `data/`, gitignored).
- Never committed (R11).
- Referenced by a path relative to `SC_DATA_ROOT`.

**New experiment dir.** `NEWEXP = $SC_EXPERIMENT_ROOT/<YYYYMMDD_HHMMSS>_stage0_mrd_newcohort_<hash>`, following the experiment-dir naming convention.
- The canonical `EXP` stays read-only. Arm P reads stored bases and scorecards from it and writes only to `NEWEXP`.
- No root Set A scorecard is ever written (R3).

**Filtered cache.** `NEWEXP/analysis/cache/stage2_diagnostic/new_cohort_cache_v1/`. It is built with the §3 filters and the §6 preprocessing.

**campaigns.csv.** Before any compute (R4), add one `planned` row per namespace: `new_cohort_inventory_v1`, `external_projection_v1`, `inductive_extended_lopo_v1`, `healthy_downsampling_v1` (Arm D, added 2026-10-09), and, only if the user asks, `newcohort_joint_grid_v1`. Each row has:
- `experiment_id`: the new id;
- `relaunch_policy`: `resume_only` for the new runners, `ask_user` for any GPU grid;
- `screen_session`: the names in §11;
- `resume_command`: written with `"$SC_EXPERIMENT_ROOT/$SC_EXPERIMENT_ID"`, where `SC_EXPERIMENT_ID` is set to the new id.

The integrator adds these rows in wave 2.

**study_config.json.** W8.4 proposes an entry for the new cohort: experiment id, raw relpath, cache relpath, and per-patient malignant/total counts. Because `study_config.json` changes only through a `type: decision` entry, this rides on the W8.4 entry.

## 6. Gene universe and preprocessing

- **Universe.** Keep the canonical universe: the 15,560 genes in `EXP/preprocessing/panels/stage0_panel_genes/all_filtered_mrd_gene_universe.json`. Panel gene lists come from the same directory. A universe is never re-derived from new cells.
- **Normalization.** As in the canonical cache: `normalize_total(target_sum=1e4)`, then `log1p`, applied to raw counts.
- **Standardization.**
  - Arm P uses the canonical per-gene mean and SD computed on the canonical cohort cache. That is the standardization the stored bases were fit with (`feature_zscore_with_zero_variance_to_zero` in each basis's `metadata.json`).
  - Arm P never re-standardizes on new cells.
  - Arm I standardizes on each fold's training cells.
- **Missing genes.**
  - Listed in the inventory.
  - In Arm P a missing gene is set to the canonical mean (z = 0), and its panel is flagged.
  - A panel that loses more than the pre-set fraction of its covered genes (§14, item 3) is excluded from primary summaries.
- **Readiness gap (for W8.3).** The Stage 0/1 runner (`scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py`) re-derives its universe from `--min-cells-fraction` on the input cells. Any joint rerun therefore needs an opt-in way to pin the canonical universe file. This sits next to the planned `--patients` filter, with defaults unchanged.

## 7. Frozen configuration (`frozen_config_v1.json`, written at lock time in this directory)

All sources are read-only and relative to `EXP`. The file contains:

- **Ids and hashes.** `canonical_experiment_id`, plus the sha256 of the universe file and the panel gene-list files.
- **Panels.** The 47 panels of the D1e/D9 table (41 biological plus 6 controls), from `analysis/scorecards/stage2_diagnostic/old34_d1e_best_dr_vs_direct_gene_selection.csv`. For each panel, the in-sample-selected DR representation (method, K, seed) and the direct-gene representation.
- **Primary (confirmatory) family.** Everything else is secondary and descriptive.
  - IFN-α PCA40 and FA40 (H-S2-08);
  - the D1-selected representations of allograft rejection, apoptosis and TNFα/NF-κB, the focus panels (H-S2-01).
  - Caveat: per the D1e table, all three focus-panel selections are factosig_promax (allograft rejection and apoptosis at K40, TNFα/NF-κB at K20). They can enter Arm P only through the W2.2 route in §8. The lock therefore names a pre-specified Arm P fallback for each focus panel: its best PCA or FA representation under the original in-sample D1 rule, restricted to PCA/FA. If the reconstruction gate fails for a promax representation, the fallback is reported in Arm P and the promax representation is evaluated in Arm I only.
- **Stage 2 setting per representation.** Penalty, C, l1_ratio, `class_weight=balanced` and max_iter, as selected in `analysis/scorecards/stage2_sharedness_lopo_scorecard.csv` (protected; read only). They must be consistent with `analysis/cache/stage2_diagnostic/layer4_5/selected_lopo_coefficients.pkl.gz`.
- **Tuned setting per representation.** C chosen by inner LOPO over the original supported patients with the W3.1 contract (`tune_c_inner_lopo`), reported for L1 and L2. It is fixed before unblinding, at the contract commit recorded in the lock.
- **Pre-fitted external probes.** Arm P trains only on original-cohort cells, so the frozen-setting and tuned probes are fitted at lock time, before any new-cohort cell is scored and ideally before the data arrive. Their coefficients are stored in `NEWEXP/analysis/cache/stage2_diagnostic/external_projection_v1/frozen_probes/` (`NEWEXP` is created at lock time if the data have not arrived yet), and their hashes are recorded in the lock.
- **L1 entry margins.** The margins of these probes on the original training set are recorded too. They pre-specify the H-S2-08 prediction in §10.
- **Seeds and metric code.**
  - Stage 1 seed 42, as stored.
  - Bootstrap seeds are listed.
  - AP uses the same `average_precision_score` call as the canonical Stage 2 runner.

## 8. Analysis arms

### Arm P — transductive projection (primary external test)

- **Stage 1.** The stored canonical bases, fit transductively on the original cohort.
  - New cells are standardized with canonical statistics (§6) and projected.
  - PCA projects through the stored loadings. sklearn FA uses the stored loadings plus the stored per-gene noise variances to compute the posterior mean.
  - FactoSig and factosig_promax enter Arm P only after W2.2 persists ψ/μ **and** a refit reproduces the stored scores. Otherwise those representations are reported in Arm I only.
- **Reconstruction gate.** Before unblinding, re-projecting the original cells must reproduce the stored `scores.npy` within the tolerance fixed by the W8.3 tests. A representation that fails is dropped from Arm P.
- **Stage 2.** The pre-fitted probes from §7 (frozen and tuned) score every new patient. They are trained on all original cells, or on the original cells without the zero-malignant patients if the user chooses that under §14 item 10.
- **Also in Arm P:**
  - D14-style overlap of new-patient factor-score distributions against the original patients on the same bases (H-S1-04). This uses `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py`, or the W1.2 (D20) v2 code if it has landed.
  - The frozen DR-vs-direct pairs on new patients (Q3; informs H-S1-02).
  - The external points for W6.5: probes trained on k original supported patients (the W6.1 subsets), scored on the new patients (H-S2-06).

### Arm I — inductive extended LOPO (secondary; tests H-S1-06)

- **Pooled cohort:** the original patients plus the new patients that meet §4 and §3.
- **Procedure.** For each supported held-out patient h, original or new:
  - fit Stage 1 on all other patients' cells only, with the W2 `stage1_refit`;
  - standardize on the training cells and project h;
  - fit Stage 2 with C tuned by inner LOPO over the training patients (W3.1), reporting L1 and L2 (R5).
- **Scope.** The W2.1 pilot scope (IFN-α PCA40/FA40) plus the primary family. The full 47-representation version is a GPU campaign, run only if the user asks.
- **Comparisons.** Per-patient lift against Arm P for new patients, and against the stored transductive LOPO for original patients.

### Arm D — healthy-cell downsampling (added 2026-10-09; pre-registered in `healthy_downsampling_plan.md`; tests H-S2-09)

- **Question** (A-20261009-12). Does the healthy-to-malignant ratio limit transfer separately from the malignant count? And if downsampling healthy cells helps, *why*: Stage 1 geometry, negative-class composition, or only the n-dependent penalty?
- **Design.** A factorial with these factors:
  - sampling scheme: random, proportional-stratified by patient × predicted type × pseudotime bin, type-equalizing, patient-capped; plus the Stage 2 sample-weight scheme SW, which re-weights the training labels per patient and class without dropping cells (added 2026-10-10: re-weighting targets the Stage 2 labels, A-20261010-08);
  - retention level;
  - placement: Stage 2 only (primary since 2026-10-10, the decided target), Stage 1 only and both (the read-outs of why a gain happens);
  - Stage 1 scope: transductive or inductive;
  - a count × ratio grid that subsamples malignant and healthy training cells independently.
  The held-out patient is never downsampled. C is retuned per condition by inner LOPO on an extended grid, with L1 and L2 (R5).
- **Two parts.**
  - **D-L:** LOPO over the extended cohort, which gives the mechanism read-outs.
  - **D-X:** a nested downsampling recipe, frozen on the original cohort and scored on the new patients like Arm P.
- **Cohort.** §4 (at least 100 malignant cells; W8.5).
- **When.** With the new data (the user's 2026-10-09 decision). No downsampling condition is scored on real labels before the lock.
- Code W8.6, run W8.7, screens `W8_hds_*` (§11).

### Exploratory only

These are reported only as exploratory:
- a joint transductive refit on original plus new cells (the original design, extended);
- timing descriptives (W1.7);
- any analysis chosen after unblinding.

## 9. Metrics and uncertainty

**Primary.** Per-patient lift = AP − π for each new supported patient.

**Cohort summaries** over new supported patients:
- median lift and P20 lift;
- pooled AP, always quoted together with them (R6).

**Secondary.** AUROC, normalized lift (AP − π)/(1 − π), and AUPRC/π.

**Uncertainty.**
- Per patient: a label-stratified cell bootstrap interval for AP and lift, plus a Boyd-type logit interval.
- Cohort: a patient-cluster bootstrap interval for the median and the P20, plus an exact sign test of lift > 0 across new patients.
- Use the W4.1 (D22) definitions and code where available. Never report naive CV standard errors.

**Achievable interval width.** Computed at inventory, before unblinding, using counts only:
- for each new patient with m malignant and n total cells, and for each primary-family representation, the expected 95% interval width of lift;
- obtained from a label-stratified bootstrap of synthetic patients of size (m malignant, n − m normal);
- the synthetic scores are drawn from the original cohort's held-out LOPO scores for that representation;
- reported in the support table, so readers know in advance which new patients can discriminate.

**Multiplicity.**
- The primary family is tested with Holm.
- The 47-panel secondary atlas uses BH.
- Everything else is descriptive.

## 10. Hypotheses tested and pre-specified expectations

The decision rules below are **proposed** and are confirmed in the W8.2 entry (§14, items 1-2).

| Analysis | Arm | Hypothesis | Pre-specified expectation | What would count against |
|---|---|---|---|---|
| Per-patient external lift, primary family (tuned probes primary, frozen reported) | P | H-S2-01 | Median lift across new supported patients > 0, with the patient-cluster interval excluding 0. The P20 is reported as the worst-group summary. | Median ≤ 0, or an interval that includes 0, for most of the primary family. |
| IFN-α PCA40 vs FA40: frozen L1 vs tuned C and L2 | P | H-S2-08 | Under frozen L1, which factor is active is already fixed by the entry margins recorded at lock. Under tuned C or L2, the PCA40–FA40 per-patient lift gap on new patients is smaller than under frozen L1. | A PCA40 advantage that persists under tuned C **and** L2 across new patients, beyond the hypothesis's allowance for a residual advantage like P07's. |
| New-patient factor-score overlap on the stored bases | P | H-S1-04 | On biological panels, new-patient normal and malignant score distributions overlap the original cohort at the D14/D20 gates (aggregate calls). Broad-bag factors private to P03 are not expected to be shared. | Separate-subtype calls for new patients on biological panels. |
| Inductive extended LOPO vs Arm P and the stored transductive LOPO | I | H-S1-06 | Per-patient lift under inductive Stage 1 falls within the patient-level interval of the transductive or projection value. | Systematic drops beyond those intervals. |
| External points on the D15b patient-count curve | P | H-S2-06 | Descriptive. Points fall within the curve's spread at the matching k. | Points far below the curve, which would bound the grant's "more patients" argument. |
| Frozen DR vs direct-gene pairs (informs, not tested) | P | H-S1-02 | Descriptive. Whether the in-sample D1e winners hold up out of sample. | — |
| Healthy-cell downsampling recipe and count × ratio grid (added 2026-10-09) | D | H-S2-09 | The six criteria and the attribution rules in `healthy_downsampling_plan.md` §7: a recipe gain over the full-data probe under tuned L1 and L2 in the inductive scope; a ratio effect at a fixed count; no match on the full-data C path; the same direction on the new patients. | No gain beyond the seed spread with C tuned on the extended grid and L2 alongside, or a gain that appears only at fixed C or only in the transductive scope. |

## 11. Compute and screen-command plan

Rules for every command below:
- They are **templates**. The W8 scripts are planned (W8.3, and a backlog line added after W8.2), so their flags are proposals until the scripts exist.
- Every new runner implements `--resume` (R2).
- No campaign starts until its `campaigns.csv` row is planned (R4) and the runner is committed (R11).
- Agents hand these commands to the user and do not run them (R1).

**1. Inventory** (CPU, expected minutes):
```
screen -S W8_inventory
export SC_EXPERIMENT_ID=<new_experiment_id>   # set on its own line so "$SC_EXPERIMENT_ID" below expands to the new id
python scripts/mrd_stage0_2/workstreams/W8_new_cohort/run_new_cohort_inventory.py \
  --input-h5ad "$SC_DATA_ROOT/<new_cohort_relpath>.h5ad" \
  --experiment-dir "$SC_EXPERIMENT_ROOT/$SC_EXPERIMENT_ID" \
  --canonical-experiment-dir "$SC_EXPERIMENT_ROOT/<canonical_experiment_id>" \
  --slug new_cohort_inventory_v1 --resume
```
Verify: the support table has one row per new patient, and the manifest status is complete.

**2. Arm P** (CPU). This runs only after the lock entry exists:
```
screen -S W8_external_projection
export SC_EXPERIMENT_ID=<new_experiment_id>
python scripts/mrd_stage0_2/workstreams/W8_new_cohort/run_external_projection.py \
  --frozen-config scripts/mrd_stage0_2/workstreams/W8_new_cohort/frozen_config_v1.json \
  --experiment-dir "$SC_EXPERIMENT_ROOT/$SC_EXPERIMENT_ID" \
  --canonical-experiment-dir "$SC_EXPERIMENT_ROOT/<canonical_experiment_id>" \
  --slug external_projection_v1 --resume
```
Verify:
- the reconstruction-gate table passes for every retained representation;
- the frozen-config hash recorded in the manifest matches the lock entry.

**3. Arm I** (CPU for PCA/FA; GPU for FactoSig). Use `screen -S W8_inductive_lopo`. The command is the W2.1/W2.2 runner with the W8.3 `--patients` filter, pointed at `NEWEXP`, with output slug `inductive_extended_lopo_v1` under `stage1_dr_variants/`. The exact line is written in the W8.4 run entry once W2.2 has landed.

**4. Optional joint grid** (GPU campaign, only if the user asks; R4 and W0.6 apply). Use `screen -S W8_joint_grid`. The runners and their resume behaviour:
- `run_stage0_mrd_old34_broad_screen.py`, with `--experiment-dir "$SC_EXPERIMENT_ROOT/$SC_EXPERIMENT_ID" --branch-name newcohort_joint_grid_v1`, the W8.3 universe pin and `--patients`. It skips existing outputs unless `--rerun`, and has no `--resume`.
- `run_stage2_mrd_multiobjective_scorecard.py`, with `--stage2-output-branch newcohort_joint_grid_v1`. It has no resume: use shards with `--shard-count/--shard-index`, then `--merge-stage2-run`.

**5. Arm D, healthy-cell downsampling** (CPU; after the lock, in phases). The screens are `W8_hds_ifna`, `W8_hds_focus`, `W8_hds_grid` and `W8_hds_external`. The command template, the phase plan, the compute estimate and the checks are in `healthy_downsampling_plan.md` §8. The runner (W8.6) is planned.

## 12. Risks and mitigations

| Risk | ALT | Mitigation |
|---|---|---|
| The new cohort's labels come from a different pipeline or rule, so label shift is confounded with transfer | ALT-LABEL | Provenance table (§3); harmonized-only primary; label-circular flags. |
| The negative class is almost all donor cells, so the probe can read genotype (HLA, chrY/XIST, ERAP2) | ALT-DONOR | Sex-match table; secondary scores on recipient-normal cells; the W1.8 gene-ablated panels reported as sensitivity. |
| New patients sit in new batches, or use a different reference or chemistry; patient is nested in batch | ALT-BATCH | Batch and reference recorded in the inventory; gene-coverage gate (§6); batch is not separable from patient, so this is stated per result. |
| Prevalence and support differ from the original patients | ALT-PREV | Lift plus normalized lift plus AUROC; achievable interval width published before unblinding. |
| Few new patients | ALT-NPAT | Patient-level summaries are descriptive; exact sign tests; no forecast beyond the observed patients. |
| A fixed-C L1 knife-edge | ALT-PEN | Tuned arm plus L2 (R5); entry margins locked in advance. |
| The original basis is transductive within the original cohort | ALT-TRANS | Arm P is fully held out for new patients; Arm I removes the transductivity for everyone. |
| In-sample selection inflated the D1/D9 winners | ALT-SEL | Selections frozen before unblinding; no reselection on new patients. |
| Cell-type composition differs between patients | ALT-CT | `predicted.annotation` mix in the inventory; within-cell-type lift as secondary (W5.3 definitions). |
| Stored FactoSig bases cannot project new cells (ψ/μ not saved). This includes all three focus-panel selections | — | Reconstruction gate; pre-specified PCA/FA fallback per focus panel (§7); FactoSig moves to Arm I until W2.2 lands. |
| Forking paths after the data arrive | — | Lock entry with hashes; post hoc labelling; pre-fitted probes. |
| Stage 0 gene sets may have been informed by the original cohort | — | W0.12 provenance. The new cohort is the unbiased test, which is one reason to lock first. |
| Partial or staggered data delivery | — | Inventory per delivery. Primary summaries wait for the cohort the user declares complete in the lock entry, or in a superseding decision entry. |
| Excluding zero-malignant patients (§4.0) changes the Stage 1 fit set and the training n of every new run | ALT-PEN, ALT-TRANS | C retuned per condition (R5); results name the cohort definition; comparisons with canonical numbers are side by side, never pooled. |
| Arm D has many schemes, levels and placements, so the best one could be picked after the fact | ALT-SEL, ALT-NREG | A nested recipe, full curves reported, an extended C grid and decision rules fixed in `healthy_downsampling_plan.md` §5-§7 before the lock. |

## 13. What this plan cannot show

The plan cannot show:
- inductive deployment on raw expression in a clinic;
- recurrence-risk or prognostic value;
- causal biology;
- that the selected Stage 0 panels beat matched random gene sets (that is W4.2);
- any forecast of benefit from sequencing more patients beyond the observed external points (W6.4 sets that boundary).

## 14. Open choices for the W8.2 decision entry

1. Confirm the primary family (§7) and the decision rules (§10).
2. Primary probe for H-S2-01. Proposal: the tuned probe is primary and the frozen setting is reported next to it. The frozen setting stays primary for the H-S2-08 prediction.
3. The gene-loss fraction above which a panel is excluded from primary summaries (§6).
4. Whether the primary analysis is restricted to harmonized-label patients (§3).
5. Whether FactoSig/promax enter Arm P (this depends on W2.2), and the PCA/FA fallback named for each focus panel (§7).
6. Whether to run the optional joint grid (§11, item 4). It is a GPU campaign and waits for the W0.6 decision (R4).
7. Whether the W1.4 provenance column for the original cohort must be complete before the lock.
8. When to lock: before the data arrive (preferred), or on arrival but before any scoring.
9. **Patient inclusion (A-20261009-11, A-20261010-07; W8.5; §4.1).** Decided on 2026-10-10: (a) at least 100 malignant cells. Confirm (b), that patients below it are excluded rather than kept as training-only.
10. **Zero-malignant exclusion and Arm P (§4.0).** Confirm the reading that the decision governs new runs, so Arm P's stored bases are unchanged. Choose whether Arm P's pre-fitted probes keep or drop the zero-malignant patients' cells.
11. **Arm D.** Lock `healthy_downsampling_plan.md` with this plan, and answer its §12 choices: rehearsal or not, the recipe's placement and scope, design size, the P20 non-inferiority margin, EN, representations.

## Readiness checklist (W8.3 and dependencies)

- [ ] An opt-in `--patients` filter on the Stage 0/1 and Stage 2 runners, with defaults unchanged and a `type: refactor` entry (W8.3).
- [ ] An opt-in canonical-universe pin on the Stage 0/1 runner (§6; to be added to W8.3).
- [ ] `run_new_cohort_inventory.py`, with tests showing default behaviour is unchanged (W8.3).
- [ ] Persisted projection parameters and the reconstruction test (W2.2). This is blocking for the focus panels in Arm P, whose selections are factosig_promax.
- [ ] The regularization contract `tune_c_inner_lopo` (W3.1).
- [ ] Uncertainty definitions (W4.1).
- [ ] An app and checks parameterized for a second experiment (W7.3; not blocking for this plan).
- [x] The patient-inclusion decision recorded (W8.5, 2026-10-10; §4.1). The rule is applied through the `--patients` filter (§4.0, §4.1).
- [ ] `run_healthy_downsampling.py` and its tests, plus the additive full-cell inner-validation option for `tune_c_inner_lopo` and the additive per-cell weight option for the SW scheme (W8.6; Arm D).
