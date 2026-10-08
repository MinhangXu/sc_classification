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

- **Support rule.** Read from `scripts/mrd_stage0_2/research/study_config.json` (`support_rule`: minimum malignant cells and both classes). This is the rule that defines the original supported patients (AGENTS R7). It is **not** changed for the new cohort; this answers THEME §11, open decision 6, for W8.2.
- **Patients below the rule** are reported descriptively, like P05, P13 and the normal-only patients. They never enter lift summaries. In Arm I they may serve as training negatives.
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

**campaigns.csv.** Before any compute (R4), add one `planned` row per namespace: `new_cohort_inventory_v1`, `external_projection_v1`, `inductive_extended_lopo_v1`, and, only if the user asks, `newcohort_joint_grid_v1`. Each row has:
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
- **Stage 2.** The pre-fitted probes from §7 (frozen and tuned), trained on all original cells, score every new patient.
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

## Readiness checklist (W8.3 and dependencies)

- [ ] An opt-in `--patients` filter on the Stage 0/1 and Stage 2 runners, with defaults unchanged and a `type: refactor` entry (W8.3).
- [ ] An opt-in canonical-universe pin on the Stage 0/1 runner (§6; to be added to W8.3).
- [ ] `run_new_cohort_inventory.py`, with tests showing default behaviour is unchanged (W8.3).
- [ ] Persisted projection parameters and the reconstruction test (W2.2). This is blocking for the focus panels in Arm P, whose selections are factosig_promax.
- [ ] The regularization contract `tune_c_inner_lopo` (W3.1).
- [ ] Uncertainty definitions (W4.1).
- [ ] An app and checks parameterized for a second experiment (W7.3; not blocking for this plan).
