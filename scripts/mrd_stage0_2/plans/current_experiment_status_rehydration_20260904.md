# Live experiment status and rehydration (2026-09-04)

**Snapshot time:** 2026-09-04 ~17:55 UTC (machine local logs use the same wall clock as `screen -ls`).  
**Workspace:** `/home/minhang/mds_project`  
**Canonical experiment:** `sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093`  
**Conda env for GPU Stage 0–2:** `sc-classification-rapids-py311`  
**Input AnnData:** `/home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad` (MRD / CITE / `CN.label` cancer vs normal)

This note is the **operational** snapshot: what is running, which scorecard namespaces exist, what finished when, and what another agent should (and should not) do next.

**Post-snapshot analysis note (2026-09-10):** the comprehensive D1 notebook now reads the completed `old34_controls_fullgrid_v1` branch, and `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` adds read-only grouped, patient-level, and selected-DR cell/ADT diagnostics. The live `screen` statements below remain the 2026-09-04 operational snapshot.

Scientific reading order for the old34 inquiry is **not** this file. Use:

- `../notebooks/stage0_2/old34_inquiry_narrative_guide.md`

Historical provenance (July 2026, before D6–D13 and the September campaigns) is:

- `experiment_20260525_stage0_2_rehydration_dossier.md`

Prior chat that produced the September runs and this inventory: [Stage0-2 experiment status](4c97693b-9f43-431a-9962-f1f72cd66bb2).

---

## TL;DR for the next agent

1. **Do not launch long jobs from the Cursor agent shell.** Paste them into `screen` on the machine. Rule: `.cursor/rules/long-runs-in-screen.mdc`.
2. **Pooled-union Stage 2 is still running.** Session `pooled_stage0_unions_v1`. Three of eight GPU shards are alive (indices 0, 2, 3). There is **no** merged pooled Stage 2 scorecard yet. Do not kill this session. Do not start another 8-GPU job until it finishes.
3. **Set A control full-grid Stage 2 is done** (2026-09-04 09:36). Session `old34_controls_fullgrid` is an idle leftover bash. Artifacts live under the **namespaced** branch `old34_controls_fullgrid_v1`, not the root biological scorecards.
4. **Donor-downsampling v2 is done** (merge-only 2026-09-03 18:54). Session `donor_ds_v2` is an idle leftover. This is a D10/D11 diagnostic, not Set A/B Stage 0–2.
5. Most notebooks under `notebooks/stage0_2/` analyze **Set A root biological Stage 2** (41 panels). The later comprehensive D1 atlas also reads the six-panel control full-grid; no current notebook reads pooled-union or Set B Stage 2 scorecards.
6. `--save-lopo-coefficients` is **not a valid CLI flag**. Saving coefficients is the default. Opt out with `--no-save-lopo-coefficients`.

---

## Live `screen` and GPU state (this snapshot)

`screen -ls` showed three detached sessions:

| Session | Started | Process tree now | Meaning |
|---|---|---|---|
| `2060358.pooled_stage0_unions_v1` | 2026-09-03 16:48 | `bash` → conda → Stage 2 launcher → **3 live shard Pythons** + 5 zombies | **Active.** Wait for shards 0/2/3, then the launcher will merge. |
| `3075422.old34_controls_fullgrid` | 2026-09-04 05:11 | idle `bash` only | **Finished.** Safe to `screen -X -S old34_controls_fullgrid quit` if you want to clean up. |
| `1447373.donor_ds_v2` | 2026-08-21 04:26 | idle `bash` only | **Finished** after `--merge-only`. Same cleanup optional. |

Pooled Stage 2 parent command (still running):

```text
bash .../stage0_panels/run_pooled_stage0_unions_stage0_to_stage2.sh \
  --experiment-dir .../20260525_060508_stage0_mrd_old34_broad_screen_82db5093 \
  --branch-name pooled_stage0_unions_v1 \
  --run-stage2 --gpu-ids auto --skip-bundle-build
```

Launched Stage 2 with `--backend gpu --strict-gpu --launch-gpu-shards --panel-selection all_quick_rows` (68 quick-ok representations, 8 shards).

| Shard | PID (at snapshot) | Status | Last log mtime | Notes |
|---|---|---|---|---|
| 0 | 2315159 `Rl+` | **running** | 2026-09-04 17:54 | GPU 0, ~194 MiB |
| 1 | 2315160 `Z+` | completed 2026-09-03 22:58 | 22:58 | 9 representations |
| 2 | 2315161 `Rl+` | **running** | 2026-09-04 15:40 | GPU 2, ~730 MiB; long gaps between cuML warnings are expected |
| 3 | 2315162 `Rl+` | **running** | 2026-09-04 15:40 | GPU 3, ~1010 MiB |
| 4 | zombie | completed 2026-09-03 20:11 | 20:11 | 8 representations |
| 5 | zombie | completed 2026-09-03 20:10 | 20:10 | 8 representations |
| 6 | zombie | completed 2026-09-03 20:12 | 20:12 | 8 representations |
| 7 | zombie | completed 2026-09-03 20:11 | 20:11 | 8 representations |

`nvidia-smi` at snapshot: all 8 GPUs at 0% util. GPUs 0/2/3 still hold process memory. That is consistent with long cuML logistic fits on ~11k-gene panels, not with a dead job. Parent has not reaped finished shards because it is waiting on 0/2/3.

**Do not** start `old34_controls_fullgrid` again. It already merged.

How to watch pooled without attaching:

```bash
screen -ls
tail -f .../stage2_supervised/multiobjective/pooled_stage0_unions_v1/runs/pooled_stage0_unions_v1_stage2/launcher_logs/shard_000_gpu_0.log
ls .../analysis/scorecards/pooled_stage0_unions_v1/stage2_sharedness_lopo_scorecard.csv   # appears only after merge
```

When it is truly done you should see a launcher line like `Merged Stage 2 run pooled_stage0_unions_v1_stage2` and these files:

- `analysis/scorecards/pooled_stage0_unions_v1/stage2_discovery_full_cohort_scorecard.csv`
- `analysis/scorecards/pooled_stage0_unions_v1/stage2_sharedness_lopo_scorecard.csv`
- `analysis/scorecards/pooled_stage0_unions_v1/stage2_patient_specific_scorecard.csv`

Those three files were **absent** at this snapshot.

---

## Hard rules (do not violate)

1. Long jobs go in **`screen`**, never the Cursor agent/shell (including background). A previous pooled run was a child of `cursor-server` and died when the extension host exited.
2. Do **not** pass destructive `--rerun` / `--rerun-stage0` unless the user asked to recompute finished artifacts.
3. Do **not** overwrite Set A **root** biological Stage 2 scorecards (`analysis/scorecards/stage2_*.csv`, 41 panels, dated 2026-06-05). New Stage 2 grids use `--stage2-output-branch`.
4. `--resume` must skip finished artifacts. Incomplete/empty/corrupt files are not finished.
5. Merge failure after compute should set `merge_failed` and recover with `--merge-only`, not kill the campaign.

---

## One experiment directory, several namespaced campaigns

Everything below lives in the same experiment dir. Scorecards for later campaigns are **not** mixed into the root CSVs.

```
experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093/
  analysis/scorecards/                          # Set A root + branches
  analysis/reports/pooled_stage0_unions_v1/     # gene-overlap QC
  preprocessing/panels/                         # gene lists, including namespaced branches
  stage1_dr/  stage1_direct_gene/
  stage2_supervised/multiobjective/<branch>/
  logs/<branch>/
```

Gene-count identity that caused confusion:

| Universe | Genes | What it is |
|---|---:|---|
| Set A `full_34` | 2,719 | Union of the original 34 named programs |
| Set B `expanded_full` | 11,358 | Union of the expanded manuscript-axes dictionary |
| Pooled `pooled_old34_expanded_union` | 11,421 | 11,358 + 63 old34-only genes |
| Pooled HVG control | 11,421 | Size-matched HVG bag, same count as pooled union |

Overlap QC: `analysis/reports/pooled_stage0_unions_v1/overlap_summary.json`  
old34 2719, expanded 11358, overlap 2656, old34-only 63, pooled 11421 (~97.7% of old34 nested in expanded).

Pooled **is not** “Set B never ran.” Set B already ran per-pathway in June. Pooled is four giant bags with no IFN-α vs p53 identity.

---

## Campaign inventory

### A. Set A old34 (root namespace) — notebooks read this

**What:** 47 named panels: 34 genesets + 7 groups + `full_34` + `core_only` + HVG 500/1k/3k/10k.

| Stage | Artifact | mtime | Rows / panels | Status |
|---|---|---|---|---|
| Stage 0/1 | `analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv` | 2026-05-25 13:22 | 755 rows / 47 panels | 739 ok, 16 skipped_too_few_genes |
| Quick L2 | `analysis/scorecards/stage2_canonical_quick_groupkfold_scorecard.csv` | 2026-05-26 04:53 | 755 / 47 | all 47 panels that could run |
| Full Stage 2 | `analysis/scorecards/stage2_{discovery_full_cohort,sharedness_lopo,patient_specific}_scorecard.csv` | 2026-06-05 10:00–10:11 | 41,795 discovery/LOPO rows / **41 panels** | **biological genesets+groups only** |

June 5 run used `--panel-selection all_biological_quick_rows`, which **excludes** `full_control`, `hvg_anchor_control`, and `core_only`. Root full Stage 2 therefore **does not** contain:

- `full_34`
- `core_only`
- `hvg_top_requested_{500,1000,3000,10000}__available_*`

Jun 4 figure-3 shortlist (`analysis/scorecards/stage2_figure3_sharedness_v2_jun4/top_stage1_shortlist/`) **does** include those control panels, but only **one best DR spec each**, not the full method×K×penalty grid.

Runner: `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`  
June 5 manifest: `stage2_supervised/multiobjective/runs/20260605_stage2_all_biological_fullgrid_gpu8/`

### B. Set A control full-grid (new, namespaced) — **complete 2026-09-04**

**Branch:** `old34_controls_fullgrid_v1`  
**Why:** fill the June 5 gap without overwriting the 41-panel biological scorecards.

| File | mtime | Rows | Panels |
|---|---|---:|---|
| `analysis/scorecards/old34_controls_fullgrid_v1/stage2_discovery_full_cohort_scorecard.csv` | 09:24 | 6,240 | 6 |
| `.../stage2_sharedness_lopo_scorecard.csv` | 09:32 | 6,240 | 6 |
| `.../stage2_patient_specific_scorecard.csv` | 09:36 | 56,160 | 6 |

Panels: `full_34`, `core_only`, HVG 500/1000/3000/10000.  
Methods: pca, fa, factosig, factosig_promax (all K). Penalties L1/L2/EN, `c-grid-n 13` (65 C×penalty rows × 96 specs = 6240). Patient-specific = 6240 × 9 patients.

Launcher: `stage2_supervised/multiobjective/old34_controls_fullgrid_v1/runs/old34_controls_fullgrid_v1/launcher_manifest.json`  
`status: completed`, `failures: []`, finished 2026-09-04 09:36:09. All 8 shards exited 0.

First launch attempt failed with `unrecognized arguments: --save-lopo-coefficients`. The successful command **omitted** that flag (save-on is the default).

These scorecards are **ready to read**. Notebooks have not been pointed at them yet.

### C. Set B expanded manuscript axes — **complete June 2026**, little notebook analysis

**Branch:** `expanded_stage0_mrd_manuscript_axes_v1/`

| Stage | mtime | Facts |
|---|---|---|
| Stage 0/1 | 2026-06-25 15:03 | 10,487 rows / 632 panels; 8,018 ok |
| Quick L2 | 2026-06-25 15:04 | same |
| Full Stage 2 | 2026-06-25 16:36–16:41 | 24,990 rows / **294 panels** |

Selection was `--panel-selection shortlist_plus_controls` / best-quick-L2 per panel (`c-grid-n` 17). Shortlist: 294 selected, 337 not.

- HVG 500/1k/3k/10k **were** selected (`shortlist_reason=control_panel`).
- `expanded_full` (11,358 genes, best quick = factosig_promax k=10) is **`selected_for_stage2_multiobjective=False`** and is **absent** from the Stage 2 discovery scorecard.
- Hardcoded force-include IDs in `make_shortlist` are Set A names (`full_34`, etc.), so they do not pull in `expanded_full`.

Do **not** rerun the June 294 unless the user asks for `expanded_full` / exhaustive `all_quick_rows` (huge).

### D. Pooled old34∪expanded unions — Stage 0/1 **done**; Stage 2 **in flight**

**Branch:** `pooled_stage0_unions_v1/`  
**Wrapper:** `scripts/mrd_stage0_2/stage0_panels/run_pooled_stage0_unions_stage0_to_stage2.sh`

Four modeled panels:

- `old34_covered_union` (2,719)
- `expanded_covered_union` (11,358)
- `pooled_old34_expanded_union` (11,421)
- `hvg_top_requested_11421__available_11421`

| Stage | Artifact | mtime | Status |
|---|---|---|---|
| Stage 0/1 | `.../pooled_stage0_unions_v1/stage0_mrd_old34_broad_scorecard.csv` | 2026-09-03 19:53 | 68 rows / 4 panels, all ok (includes `direct_gene`) |
| Quick L2 | `.../stage2_canonical_quick_groupkfold_scorecard.csv` | 2026-09-03 19:54 | 68 ok |
| Full Stage 2 | discovery / LOPO / patient-specific CSVs | **missing** | shards 0,2,3 still fitting |

History: another agent started this **not in screen** on 2026-09-03; Cursor remote shutdown killed it at 62/64 DR fits. User resumed in `screen -S pooled_stage0_unions_v1` with `--skip-bundle-build` and **without** `--rerun-stage0`. Stage 0 finished 19:53 the same day; comprehensive Stage 2 launched immediately. Controls full-grid later shared the same 8 GPUs for ~3.5 h (05:56–09:36 on 2026-09-04), which likely slowed the remaining pooled shards.

### E. Direct-gene / no-DR baselines (Set A biological) — **complete 2026-07-24**

**Branch:** `old34_direct_gene_all_v1/`  
Stage 0: 41 panels. Stage 2 discovery: 2,665 rows / 41 panels. This is D9 in the meeting-follow-up notebook (41/41 coverage). Do not confuse D7E (pre-follow-up 3/41) with D9E.

### F. Donor-downsampling diagnostic — **complete 2026-09-03**

**Not** Stage 0–2 of Set A/B. D10/D11 only.

- Runner: `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py`
- Outputs: `analysis/scorecards/stage2_diagnostic/donor_sensitivity_v2/`
- `RUN_STATE.json`: `status: complete`, updated 2026-09-03 18:54 UTC
- Compute finished earlier (681 fixed_basis + 21 basis_refit). Merge originally crashed on pandas `EmptyDataError` from 1-byte `\n` supervised CSVs. 80 targets had all-zero full-donor coefficients (often k=5 L1 collapse); empty supervised files are scientifically valid.
- Code was patched: refuse columnless CSVs, repair empty checkpoints, `concat_checkpoints` no longer crashes, `merge_failed` + `--merge-only`.
- User ran `bash run_stage2_old34_donor_downsampling_sensitivity.sh --merge-only` in the existing `donor_ds_v2` screen. Merge skipped 80 empty supervised files and wrote the seven merged CSVs.

Corrected scientific claim lives in `old34_inquiry_narrative_guide.md` (FA 007 vs PCA 008 at 25% donors). Do not quote the older “both bases ≈0.99 at 25%” sentence.

### G. Other diagnostics already on disk (Set A)

Under `analysis/scorecards/stage2_diagnostic/`:

- `meeting_followup_v1/` — D6–D9
- `layer4_5/` — interpretability / factor→gene
- `deg_hvg_lowsupport_v1/` — D12–D13 (notebook 2026-09-03 05:28)
- `donor_sensitivity_v1/` and `v2/`
- `d1a_deep_dive_20260910/` — grouped D1 taxonomy, three focus-panel patient/K/direct comparisons, label-free PCA/FA variance, mean-z one-score proxy, and selected-DR cell/cell-type/ADT/sample audits
- representation/panel tables used by the comprehensive diagnostic notebook

---

## Notebooks: `scripts/mrd_stage0_2/notebooks/stage0_2/`

They primarily read **Set A root** scorecards and `stage2_diagnostic/` artifacts. The comprehensive D1 atlas additionally reads `old34_controls_fullgrid_v1`; none currently analyze Set B or pooled-union Stage 2.

| File | Role | Last edited |
|---|---|---|
| `old34_inquiry_narrative_guide.md` | **Read me first** for science: equations, claim boundaries, notebook order | 2026-09-03 05:29 |
| `stage2_old34_team_review_narrative.ipynb` | Presentation-ready D6–D11; no model reruns | 2026-08-21 |
| `stage2_old34_comprehensive_diagnostic_suite.ipynb` | D1–D3 over all 643 biological representations | 2026-08-13 |
| `stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` | D1 panel taxonomy; allograft/apoptosis/TNF patient, K, direct, one-score, cell-type, ADT, and collection-time audits | 2026-09-10 |
| `stage2_old34_layer4_5_interpretability_suite.ipynb` | Regularization pathology, factor→gene | 2026-07-24 |
| `stage2_old34_meeting_followup_metric_interactions.ipynb` | D6–D9 (AP baseline, interactions, IFN-α, direct-gene) | 2026-09-03 16:49 |
| `stage2_old34_donor_downsampling_sensitivity.ipynb` | D10/D11 v1 (corrected stability claims) | 2026-08-21 |
| `stage2_old34_donor_downsampling_sensitivity_v2.ipynb` | D10/D11 expansion | 2026-08-21 |
| `stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` | D12/D13 GSEA vs HVG and low-support DE | 2026-09-03 05:28 |
| `_build_deg_hvg_lowsupport_notebook.py` | Generator for the D12 notebook | 2026-09-03 05:27 |
| `stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` | Historical exploratory sharedness suite | 2026-07-24 |
| `stage2_mrd_figure3_sharedness_suite_20260528.ipynb` | Superseded; keep for provenance | 2026-07-23 |
| `old34_heldout_lift_illustration_prompt.md` | Prompt for a lift illustration | 2026-07-24 |

Pass A reading order is in the narrative guide. Skip the 20260528 notebook unless reconstructing history.

---

## Code / policy touched in the September work

- Created: `.cursor/rules/long-runs-in-screen.mdc` (`alwaysApply: true`)
- Edited: `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py` (atomic CSV write, empty-checkpoint repair, merge robustness)
- Edited: `.../run_stage2_old34_donor_downsampling_sensitivity.sh` (`--merge-only`)
- On-disk repair: 80 empty `*.supervised.csv` under `analysis/cache/stage2_diagnostic/donor_sensitivity_v2/checkpoints/fixed_basis/`
- **Did not** start pooled or control Stage 2 from the agent shell; user launched those in `screen`

Stage 2 CLI gotcha:

```text
# INVALID
--save-lopo-coefficients

# default is save-on; opt out with
--no-save-lopo-coefficients
```

(`parser.set_defaults(save_lopo_coefficients=True)` in `run_stage2_mrd_multiobjective_scorecard.py`.)

---

## What is done vs not done

**Done**

- Set A Stage 0/1 + quick L2 (47 panels, May 2026)
- Set A full Stage 2 for 41 biological panels (June 5)
- Set A Jun 4 one-best-DR Stage 2 including `full_34`/HVGs (not a full grid)
- Set A direct-gene 41/41 (July 24)
- Set B Stage 0/1 + Stage 2 for 294 shortlist+control panels (June 22–25)
- Set A D1–D13 diagnostic notebooks and reports
- D1a descriptive deep dive (21 figure pairs; 25 derived tables/reports; no Stage 0/1/2 campaign rerun)
- Donor v2 compute + merge-only
- Pooled Stage 0/1 (68 representations, Sep 3 19:53)
- Set A **control full-grid** Stage 2 for `full_34` / `core_only` / 4 HVGs (Sep 4 09:36), namespaced

**In flight**

- Pooled full Stage 2 (3/8 shards; no merged scorecards yet)

**Not done / not started unless the user asks**

- Pooled Stage 2 notebook/figures; broader control-grid synthesis beyond the D1 atlas
- Set B `expanded_full` full Stage 2
- Set B exhaustive `all_quick_rows` (would be enormous)
- Combining root 41-panel biological scorecards with the new 6-panel control grid into one analysis table (do this in analysis code; do not overwrite root CSVs)
- huCIRA / relapse 4-class (still in the value-added plan)

---

## Suggested next steps (user/agent)

1. **Leave pooled `screen` alone** until merged Stage 2 CSVs appear. If it dies, resume in the **same** screen with `--skip-bundle-build` and without `--rerun-stage0`; shards already completed should be skipped.
2. **Optional analysis (CPU, no GPU contention):** start reading `old34_controls_fullgrid_v1` vs the 41-panel biological grid and vs the Jun 4 one-best-DR control rows.
3. After pooled merge: compare the four bags to Set A `full_34` and Set B `expanded_full` (the latter still has no full Stage 2).
4. Do not relaunch controls full-grid. Do not merge control results into root `analysis/scorecards/stage2_*.csv`.

Corrected controls command (already executed successfully; keep for provenance only):

```bash
screen -S old34_controls_fullgrid   # already finished; do not rerun
conda run --no-capture-output -n sc-classification-rapids-py311 python -u \
  /home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py \
  --experiment-dir /home/minhang/mds_project/sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093 \
  --stage0-scorecard .../analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv \
  --stage2-output-branch old34_controls_fullgrid_v1 \
  --stage2-run-id old34_controls_fullgrid_v1 \
  --panel-selection all_quick_rows \
  --panel-ids full_34,core_only,hvg_top_requested_500__available_500,hvg_top_requested_1000__available_1000,hvg_top_requested_3000__available_3000,hvg_top_requested_10000__available_10000 \
  --run-discovery-full-cohort-fit --run-sharedness-lopo --run-patient-specific \
  --penalties l1,l2,elasticnet --c-grid-log10-min -4 --c-grid-log10-max 4 --c-grid-n 13 \
  --class-weight balanced --backend gpu --strict-gpu --launch-gpu-shards --gpu-ids auto --seed 42
```

---

## Key absolute paths

```
/home/minhang/mds_project/sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093
/home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/notebooks/stage0_2
/home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/plans
/home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py
/home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/stage0_panels/run_pooled_stage0_unions_stage0_to_stage2.sh
/home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py
/home/minhang/mds_project/sc_classification/scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py
```
