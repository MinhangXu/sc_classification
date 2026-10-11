---
id: W2
title: W2_stage1_basis_variants: inductive, leave-out, patient-balanced and subset Stage 1 bases (D17)
status: active
questions: [Q1, Q3]
---

# W2 — W2_stage1_basis_variants: inductive, leave-out, patient-balanced and subset Stage 1 bases (D17)

Status as of 2026-10-10: active (P1; no P0 task is open since W2.4 closed). D17 (W2.1) is done for IFN-alpha (devlog 2026-10-08_d17-basis-variant-pilot); `stage1_refit.py` fits, projects and persists PCA/FA bases, including patient weights. `row_weights` in the src DR wrappers is still silently ignored (W2.2). W2.4 is done: the user ran screen `W2_d26` on 2026-10-10 and D26 is read in devlog 2026-10-10_d26-leverage-allograft-apoptosis-done (campaign leverage_allograft_apoptosis_v1; H-S1-05 not supported for PCA/FA). W2.3 is closed by a decision (devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs): the user chose the Stage 2 labels as the re-weighting target (A-20261010-08), so no patient-reweighted Stage 1 refit is run for the focus panels; Stage 2 re-weighting runs with the new patient data in W8 Arm D. Next: W2.2 and W2.7; W2.8 and W2.9 are proposed.

## Charter

Asks, quoted as recorded (in `scripts/mrd_stage0_2/research/asks/2026-10-08_team-report.md`; the 2026-10-08 team-report items are condensed wording, capture: paraphrase):

> "Option 2: exclude the held-out patient from Stage 1 entirely, then project their cells onto the learned factors" (A-20261008-16)

> "Option 1.2: re-weight cells so each patient contributes equally to DR" (A-20261008-14)

> "Option 1.1: subset to the top 4-5 patients with sufficient malignant cells, plus YH's new patient; plan the run after choosing" (A-20261008-13)

> "Option 3: focus only on patients with large cell counts (2, 3, 4) and rerun the full pipeline" (A-20261008-17)

> "DISCUSSED: patients with more malignant cells (P02, P03, P04) disproportionately shape the factors" (A-20261008-02; shared with W1, routed here through W2.1)

Wording note (AGENTS.md R6): A-20261008-02 and A-20261008-17 are quoted as recorded in the asks file. P02, P03 and P04 are the highest-prevalence patients; they are neither the largest patients by cell count nor the patients with the most malignant cells (see `scripts/mrd_stage0_2/research/study_config.json`). Option 3 is therefore read as the leave-high-prevalence-out arm of W2.1 and the subset pipeline of W2.5.

Scope: Q1 and Q3. The user's options 2 (inductive: exclude the held-out patient from Stage 1 and project it), 1.2 (re-weight so each patient contributes equally) and 1.1/3 (patient subsets). This workstream supplies the bases that W1, W3, W6 and W8 consume.

Done means: inductive, leave-high-prevalence-out, patient-balanced and subset Stage 1 bases exist with persisted projection parameters; their Stage 2 transfer is measured with C retuned per fold and compared with the transductive anchor; and the answers feed H-S1-03, H-S1-05 and H-S1-06.

## Builds on

- Registry: D3 and D14 (W1), D4 (W3), D10 and D11 (W4; the donor runner's basis-refit arm). Owned here: D17.
- Hypotheses: H-S1-03, H-S1-04, H-S1-05, H-S1-06.
- Template: `run_basis_refit_target()` and `fit_basis_model()` in `scripts/mrd_stage0_2/stage2_supervised/run_stage2_old34_donor_downsampling_sensitivity.py`.
- DR wrappers: `src/sc_classification/dimension_reduction/pca.py`, `src/sc_classification/dimension_reduction/factor_analysis.py`, `src/sc_classification/dimension_reduction/factosig.py`. They accept `row_weights`, but `preprocess_data` in `src/sc_classification/dimension_reduction/base.py` ignores it, so weighting must be implemented in `stage1_refit.py` (W2.2 decides whether the wrappers should raise instead, through a type: refactor entry).
- Stage 1 producer and conventions: `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py`; Stage 2 conventions: `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`.
- EXP-relative inputs: `stage1_dr/` (transductive bases; read-only), `stage1_direct_gene/` (covered-gene features), `stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv` (the transductive per-patient lifts the anchor arm must reproduce).
- C policy: W3's `regularization_contract` (D19, devlog 2026-10-08_d19-regularization-contract).
- Devlog entries: 2026-08-21_donor-sensitivity-d10-d11 (basis-refit arm), 2026-10-02_d14-patient-factor-overlap, 2026-10-08_team-report-devlog-pivot.
- Literature routing (`../W9_literature/README.md`): narrow inductive Stage 1 → W2.1; per-patient refits and centering ablation → W2.3.

## Backlog

- [W2.1] (P0) D17 Stage 1 basis-variant pilot (slug basis_variant_pilot_ifna_v1). (depends: W3.1)
  - Scope: IFN-alpha PCA40/FA40 × the 7 supported held-out patients, on the covered genes from the stage1_direct_gene features.
  - Arms:
    - transductive (the anchor);
    - inductive: fit without the held-out patient, standardize on training cells only, then project;
    - leave-high-prevalence-out: drop P02/P03/P04, with drop-P01/P09 as the contrast;
    - patient-balanced: weighted-covariance PCA, and patient-stratified subsampling for FA.
  - Stage 2: L1 at the stored C (the anchor), plus per-fold C tuned by inner LOPO over the training patients (L1 and L2), via regularization_contract.
  - Align to transductive PC8/FA7 with Hungarian matching plus many-to-one R2, and report principal angles.
  - Persist the bases under stage1_dr_variants/basis_variant_pilot_ifna_v1/.
  - Done when: a per-patient lift table (arms × methods × penalties) exists and the transductive arm reproduces the stored per-patient lifts in sharedness_lopo by_heldout_patient.csv.
- [W2.2] (P1) Harden stage1_refit — done when: toy-matrix tests cover the fit/project/persist round trips. (depends: W2.1)
  - Persist projection parameters: gene order, mean/sd, PCA components, FA noise variance and mean, and FactoSig psi/mu.
  - Store fold-specific scores as stage1_dr_variants/<slug>/panel_id=/method=/k=/seed=/heldout=<P>/, with a loader for Stage 2.
  - Change the src DR wrappers (e.g. make row_weights raise instead of being ignored) only through a devlog entry with type: refactor.
- [W2.3] (P1) Option 1.2 beyond IFN-alpha. First decide which imbalance is targeted: equal per-patient weighting raises the Stage 1 share of the high-prevalence patients (THEME §11). Then fit a patient-reweighted Stage 1 for the focus panels, with C retuned — done when: a decision entry and a principal-angle/Hungarian comparison exist. Closed 2026-10-10 by the decision entry alone (devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs): the target is the Stage 2 labels (A-20261010-08), so the Stage 1 refit and its comparison are not run; the IFN-alpha Stage 1 pilot (D17) stays the record, and Stage 2 re-weighting (per-patient, per-class sample weights) runs with the new data as scheme SW of W8 Arm D (W8.6, W8.7). (depends: W2.1)
- [W2.4] (P1) Leave-high-prevalence-out refits for allograft and apoptosis, as the loading-leverage test of H-S1-05, with size-matched and composition-matched (same dropped cells and malignant cells) random-drop controls. D26 (slug leverage_allograft_apoptosis_v1) — done when: angle and matched-|r| tables with control ranks exist. (depends: W2.1)
- [W2.5] (P1) Patient-subset pipeline for Options 1.1/3 — done when: subset scorecards exist under the new namespace. (depends: W8.3) (blocked: new data, W8.1)
  - Subset: supported patients with ≥100 malignant cells, plus the new YH patients, selected with the W8.3 --patients filter.
  - Runs on GPU in user screen, on a new branch registered in campaigns.csv first.
  - C is retuned.
- [W2.6] (P2) Inductive Stage 1 for the 47 D1-selected representations, with fold-specific storage under stage1_dr_variants/inductive_lopo_v1/ (GPU, user screen) — done when: a Stage 2 LOPO run on the inductive bases is merged. (depends: W2.2)
- [W2.7] (P1) Audit the stored randomized-SVD PCA scores (U·S) against the exact projection onto the stored loadings across the D1 atlas, and flag representations whose Stage 2 outcome changes — done when: a per-representation table of score r and stored-vs-projected lift exists. (depends: W2.1)
- [W2.8] (P2) Loading-leverage test on further Stage 0 panels with `run_d26_leverage_panels.py --panels ...` (the D17/D26 design; A-20261009-02 "Should we test them too?"). Run it on all biological panels only if D26 finds leverage beyond composition on the primary or probe axis (met: devlog 2026-10-10_d26-leverage-allograft-apoptosis-done; read the contrast-set condition, which D26 never met); otherwise run a stratified sample of panels. Each panel is a few minutes of CPU in user screen — done when: a cross-panel leverage table with composition-matched rank p exists. (depends: W2.4)
- [W2.9] (P2) Cell-type-matched leave-out control for P01: random drops matched to P01's predicted cell-type mix as well as its size and malignant count, to test whether P01's leverage on the IFN-alpha axis is its cell-type mix (it holds most of the cohort's ncMono cells; devlogs 2026-10-08_d17-composition-matched-controls and 2026-10-09_d28-celltype-score-shares). Extends the D17/D26 drop sets in `run_d26_leverage_panels.py` additively — done when: P01's axis loss is ranked among cell-type-matched controls. (depends: W2.4)
- [W2.10] (P1) Protein in the DR step: joint RNA + protein Stage 1 bases, MOFA+ first (linear, per-factor RNA and protein loadings, readable like FA), totalVI second (VAE with a protein background model), written to `stage1_dr_variants/<slug>_v<N>/` (R3) and scored by the same Stage 2 probe as the RNA-only bases. Needs a campaigns.csv row and screen jobs (R1); the packages (mofapy2, scvi-tools) are not in the lab env and are installed only by the user — done when: lift tables compare joint and RNA-only bases on at least the IFN-alpha panel and one broad panel. MX: "Definitely include the factors in the DR step as a model that we are going to try"; added 2026-10-11 by devlog 2026-10-11_clr-isotype-p03-fig23-next (A-20261011-12, -13).

## Code in this directory

- `stage1_refit.py` (module, numpy/scipy/scikit-learn). It fits and projects Stage 1 bases on a chosen cell set, standardizing genes on the fit cells only.
  - `fit_basis(X, method, k, rows=..., weights=..., seed=42, gene_names=...)` handles PCA and FA with the stored settings (randomized solvers, random_state 42). An unweighted refit on all cells reproduces the stored bases.
  - Patient weights: PCA uses a weighted covariance plus eigh. FA is fitted to the weighted covariance through moment-matched pseudo-data (`fa_from_covariance`), with the same solver and seed and tol rescaled to the fit size. Unweighted, this is identical to sklearn on the rows.
  - Cell selection helpers: `patient_balanced_weights`, `stratified_subsample` (equal n per patient), `random_drop_mask` (size-matched controls), `fit_scope_mask` (leave patients out, normal-only).
  - `Stage1Basis` stores gene order, gene mean/SD, components, model mean, FA noise variance and PCA explained variance (`save` / `load` to basis.npz + basis.json). `transform()` is exactly sklearn's PCA/FA transform.
  - Alignment: `score_correlation`, `hungarian_match`, `many_to_one_r2`, `principal_angles_deg`, `subspace_containment`, `align_scores`.
- `run_d17_basis_variant_pilot.py` is the D17 runner. It imports W3's contract with `add_study_module_paths(workstreams=("W3_dr_value_regularization", "W2_stage1_basis_variants"))`.
  - Parts: anchor, bases, align, stage2, report.
  - `--resume` skips finished bases and Stage 2 units (basis × held-out patient); empty, corrupt or out-of-date files are redone.
  - The anchor gate stops the run unless the stored sharedness_lopo lifts are reproduced (same nonzero support, within the tolerance recorded in the devlog entry).
  - Other flags: `--ks`, `--methods`, `--full-ks`, `--smoke`, `--out-root`.
  - `--bases-location cache|stage1_dr_variants` defaults to cache; D17's bases are under `analysis/cache/stage2_diagnostic/basis_variant_pilot_ifna_v1/bases/`.
  - Outputs: `analysis/{scorecards,figures,cache}/stage2_diagnostic/basis_variant_pilot_ifna_v1/`.
- `run_d17_basis_variant_pilot.sh` is the screen wrapper (`screen -S W2_d17`; default `--ks 10,20,40 --resume --n-jobs 12`; a one-shot reproduction is several minutes, so use screen).
- `run_d26_leverage_panels.py` is the D26 runner (task W2.4; slug leverage_allograft_apoptosis_v1). It runs the D17 leverage design on more panels: leave out P02 / P03 / P04 / P02+P03+P04 and, as contrasts, P01 / P09 / P06 / P01+P09; give each set size-matched and composition-matched random-drop controls (same dropped cells AND dropped malignant cells, D17 seeds); add normal-only plus random drops and a transductive refit as the sanity floor. It covers PCA and FA at K 10/20/40 and is alignment only (no Stage 2 refit). It imports D17 (`fit_rows`, drop sets, seeds, io helpers), `stage1_refit` and W3's contract with `add_study_module_paths(workstreams=("W3_dr_value_regularization", "W2_stage1_basis_variants"))`.
  - Reference axes (part `refs`):
    - `top_gap` (primary): the D17 rule, the largest all-cell standardized malignant-normal gap;
    - `top_heldout`: the D18 card statistic, the largest median univariate held-out lift;
    - `probe`: the atlas-selected Stage 2 configuration from D19, refit on all cells; its coefficients are checked against the stored discovery coefficients.
  - Parts: refs, bases (one unit = fit + align of one basis), report.
    - `--resume` skips units that have a saved basis and an alignment JSON with the current reference-axis signature. A saved basis with a missing or stale alignment is only re-aligned. Empty or corrupt files are redone.
    - A report with missing units writes `merge_failed.json` and exits 3. Recover with `--resume` or `--merge-only`.
  - Other flags: `--panels` (any Stage 0 panel id or hallmark suffix, so further panels can be tested; W2.8), `--methods`, `--ks`, `--smoke` (needs `--out-root`), `--plan-only`, `--bases-location cache|stage1_dr_variants`.
  - Outputs: `analysis/{scorecards,figures,cache}/stage2_diagnostic/leverage_allograft_apoptosis_v1/`.
    - Tables: `d26_reference_axes.csv`, `d26_factor_signal.csv`, `d26_alignment.csv`, `d26_refit_check.csv`, `d26_leverage_contrast.csv`, `d26_decision_summary.csv`, `d26_cross_panel_top_gap.csv`. The alignment and leverage tables keep D17's column names plus `panel_id` and `ref_axis`, so the report's `figs_02_shape.py` can filter them.
    - Figures D26A-D26D.
    - Findings, manifest and ledger.
  - FactoSig and FactoSig-promax are out of scope (no CPU refit; needs W2.2).
- `run_d26_leverage_panels.sh` is the screen wrapper (`screen -S W2_d26`; default `--ks 10,20,40 --resume --n-jobs 12`, single-thread BLAS). A full run takes several minutes, so use screen (R1); the launch and resume commands are the same (campaigns.csv row leverage_allograft_apoptosis_v1). Devlog 2026-10-09_d26-leverage-allograft-apoptosis has the measured smoke cost and the verification steps.
- `tests/test_d26_leverage_panels.py` (repo level) holds two kinds of tests.
  - Synthetic: reference-axis rules, rank p, leverage table equal to `d17.leverage_table` and NaN-safe on the probe axis, decision rule, plan counts, alignment rows, resume states.
  - EXP regression: the D17 PC8/FA7 rule, D17 `ref_arrays`, the D18 univariate lifts, and the D26 slug tables once they exist. These skip without EXP or before the run.
- `tests/test_stage1_refit.py` (repo level): toy matrices for fit/project/persist round trips, weighting equivalences and alignment, plus EXP regression tests that skip when EXP is absent.
- Note: the alignment reference for PCA is the projection onto the stored loadings. The stored PCA scores are randomized-SVD U·S and differ slightly from that projection (devlog 2026-10-08_d17-basis-variant-pilot; audit across the atlas is W2.7).
- Screen sessions: `W2_d17` (optional D17 reproduction); `W2_d26` (the D26 run, finished 2026-10-10). W2.5 and W2.6 are GPU runs in user screen under `W2_<topic>` sessions, each after a campaigns.csv row is added.

Task status (derived from the devlog; see `../../research/STATUS.md`): W2.4 is closed by devlog 2026-10-10_d26-leverage-allograft-apoptosis-done (the screen `W2_d26` run), which supersedes the run entry 2026-10-09_d26-leverage-allograft-apoptosis. W2.8 and W2.9 were added on 2026-10-09 from that entry, D28 and report v2. W2.1 is closed and W2.2 advanced by devlog 2026-10-08_d17-basis-variant-pilot. W2.2 still needs the bases written under `stage1_dr_variants/basis_variant_pilot_ifna_v1/` (refit with `--bases-location stage1_dr_variants`, or copied with a manifest; the analysis/cache copy stays in place, R10), a Stage 2 loader, and FactoSig psi/mu. W2.6 waits on W2.2 (STATUS §2); W2.3 was closed by a decision on 2026-10-10 (devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs); W2.4 reused the D17 leverage design (leave-out plus size-matched AND composition-matched random drops, retained-cell many-to-one R2 and principal angles). The D17 composition-matched controls are now in the slug tables (devlog 2026-10-08_d17-composition-matched-controls). W2.7 was added from that entry's Next steps.

## Conventions

- Owned artifact slugs (EXP-relative): `stage1_dr_variants/basis_variant_pilot_ifna_v1/` (bases; reserved: D17's bases currently persist under `analysis/cache/stage2_diagnostic/basis_variant_pilot_ifna_v1/bases/`, because the D17 run could write only analysis slug dirs; `--bases-location stage1_dr_variants` writes the R3 location, and W2.2 rewrites them there (refit, or copy with a manifest) and leaves the analysis/cache copy in place, R10) and `analysis/{scorecards,figures,cache}/stage2_diagnostic/basis_variant_pilot_ifna_v1/` (D17); `analysis/{scorecards,figures,cache}/stage2_diagnostic/leverage_allograft_apoptosis_v1/` (D26; bases under its cache `bases/`, like D17); `stage1_dr_variants/inductive_lopo_v1/` (W2.6). New bases always go under `stage1_dr_variants/<slug>_v<N>/`, never `stage1_dr/` (R3). A new slug is never overwritten; each artifact dir carries `*_findings.md`, `*_manifest.json` and `*_artifact_ledger.csv`. Paths in devlog entries and `scripts/mrd_stage0_2/research/analyses.csv` are relative to the experiment dir.
- C policy (AGENTS.md R5): sklearn C does not scale with n. Under balanced L1 at fixed C, a standardized feature enters only if its training class-mean gap exceeds about 4/(C·n_train). Every W2 arm changes the training data, so retune C per fold by inner LOPO over the training patients through W3's `regularization_contract.tune_c_inner_lopo` (D19), keep L1 at the stored C only as the anchor, and report L2 alongside. Record `c_policy` (per_fold_tuned) and `training_n_changed` in the devlog entry. An all-zero L1 probe scores the constant-score no-skill AP = π by construction; never read it as "no signal".
- Import header for a new script in this directory:

```python
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache, add_study_module_paths
add_study_module_paths(workstreams=("W3_dr_value_regularization",))  # only if needed
EXP = experiment_dir()
```

- Screen sessions are named `<W#>_<topic>`. DR/FactoSig grids and GPU Stage 2 runs are never started from an agent shell (R1): hand the user `screen -S <W#>_<topic>`, the full command, the expected outputs and how to verify completion, and record a type: run entry with compute: screen-cpu or screen-gpu. New runners implement `--resume` (R2). Add a planned campaigns.csv row before handing over a new GPU campaign (R4).
- Label every basis by fit scope (transductive, inductive, leave-out, balanced, subset); LOPO on the transductive basis is label transfer, not inductive deployment (R6).
- Tests go in `tests/test_<module>.py` and load the module by file path with importlib.
- Cite devlog entries, do not restate numbers. Results live only in devlog Results (R12); this README cites IDs.
- Task state is derived from devlog `closes` and `advances`; never tick boxes. Append new tasks as backlog lines with the next free `W2.n`.
