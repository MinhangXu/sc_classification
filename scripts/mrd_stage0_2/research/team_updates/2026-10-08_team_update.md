# Stage 0-2 MRD malignant-vs-normal classifier: team update (2026-10-08)

Dated communication: numbers here are a snapshot. The living record is the devlog (one entry per result) and the generated `scripts/mrd_stage0_2/research/STATUS.md`. Devlog entry for this update: `scripts/mrd_stage0_2/research/devlog/2026/2026-10-08_team-update-and-wave2-integration.md`.

Conventions:
- Paths are relative to `sc_classification/`. `EXP/` = `experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093/` (local, gitignored, about 199 GB).
- Lift = held-out AUPRC minus the held-out patient's malignant prevalence, in percentage points (pp).
- Supported patients (at least 10 malignant cells and both classes): P01, P02, P03, P04, P06, P07, P09.
- Stage 1 bases are transductive (fit on all 60,118 cells of all 13 patients) unless a result says otherwise. LOPO on them is label transfer, not inductive deployment.

## Where things live now

| What | Where |
|---|---|
| Start here (any person or coding agent): rules R1-R12 (long/GPU jobs only in user screen with `--resume`; never overwrite the root Set A scorecards; retune C whenever training n changes; claim language) | `AGENTS.md` |
| One-page dashboard: workstreams, open tasks, verdicts, analyses, campaigns, unanswered asks (generated) | `scripts/mrd_stage0_2/research/STATUS.md` |
| Central question, Q1-Q3, current claim boundary (§6), corrections (§7); number-free | `scripts/mrd_stage0_2/research/THEME.md` |
| 24 hypotheses (H-IDs) with gates; evidence and verdicts rendered by Q in `CLAIMS.md` | `scripts/mrd_stage0_2/research/hypotheses.md`, `CLAIMS.md` |
| Dated dev trace: question, estimand, implementation, artifacts, results, claim gate, next steps | `scripts/mrd_stage0_2/research/devlog/` (index: `devlog/INDEX.md`) |
| Analysis registry (D/F IDs → code → notebook → EXP artifact dir) and compute campaigns (do-not-relaunch policy, resume commands) | `scripts/mrd_stage0_2/research/analyses.csv`, `campaigns.csv` |
| Questions as asked (this report's items are A-20261008-01..32) | `scripts/mrd_stage0_2/research/asks/` |
| Workstreams W0-W9: charter, backlog (W#.n tasks), code, conventions | `scripts/mrd_stage0_2/workstreams/` |
| Validator / generator for all of the above | `python tools/research_index.py check` / `write` |

The Cursor hypothesis canvas is frozen; its verbatim copy is in `scripts/mrd_stage0_2/research/legacy/`.

## 1. Here's what's done

### New today (wave 2; all descriptive, transductive unless stated)

- **D19 regularization contract and L1 audit** (W3; `workstreams/W3_dr_value_regularization/`). Shared C-policy module that every refit now imports.
  - Balanced L1 at fixed C: a standardized factor enters only if its training class-mean gap exceeds 4/(C·n_train) (elastic net: 4·l1_ratio/(C·n_train)). At C = 1e-4 that threshold is 0.678 SD (P04 held out) to 0.930 SD (P09 held out).
  - The rule reproduces all 14/14 IFN-alpha PCA40/FA40 fold outcomes and 2,772/2,772 zero/nonzero outcomes over the 396 L1/EN-selected atlas DR cells.
  - Knife-edge: 136 folds in 49 cells sit within ±10% of entry (134 of them at the C-grid floor). Crossing the threshold changes lift by ≥5 pp in 41 of 131 evaluable folds (max 39.06 pp, hallmark hypoxia FA K10, P04).
  - 749 of the 2,772 audited folds are stored all-zero (constant score, lift 0 by construction); 528/783 atlas cells select the lowest C on the grid.
- **D16 factor × patient allocation matrix** (W1). 320 representations (41 biological panels × PCA/FA × K 5/10/20/40; 8 panel × K units are missing where the panel has fewer genes than K), 5,870 factors; reproduces the scratch anchor, F6 and the D14 eta2 (18/18 checks).
  - P02+P03+P04 are 8.2% of cells and hold 9.1% (PCA40; ratio 1.11, the third highest of the 320 representations) / 8.4% (FA40) of IFN-alpha score energy. Across all 320 representations the ratio of their share to their cell share has median 0.96 (range 0.58-1.14).
  - The patient most over-represented in score energy is P01 (median ratio 1.24), not a high-prevalence patient. P02's frequent top enrichment is excess spread on low-variance trailing factors, not a shifted mean.
- **D17 Stage 1 basis-variant pilot, IFN-alpha** (W2). PCA/FA at K 10/20/40 refit inductively (without the held-out patient), without P02/P03/P04 (and contrasts P01/P09/P06), normal-only, and patient-balanced; Stage 2 with per-fold tuned C and L2. Anchor: 84/84 stored lifts within 0.1 pp with the same support (82/84 within the original 0.05 pp).
  - Inductive Stage 1 keeps positive lift in 7/7 supported patients under tuned C or L2 at every K. The median per-patient change versus the transductive refit is −1.29 to −0.05 pp across the 18 tuned/L2 summaries; single patients lose up to 3.8 pp. PCA40 tuned L1 (pooled AP / median lift / P20): inductive 0.1164 / 10.55 / 1.29 pp vs transductive refit 0.1153 / 8.91 / 1.42.
  - Leverage: dropping P02+P03+P04 moves the PC8 axis 3-5x more than size-matched random drops (at every method and K) but by little in absolute terms (PCA40 axis loss 0.0015 vs 0.0003; matched |r| 0.995). With 20 composition-matched controls (same number of cells AND malignant cells removed) the excess mostly disappears: FA 0.36-1.00x the control mean (rank p 0.19-0.86), PCA 1.42-1.94x (above all 20 controls). Dropping P01 moves the axis far more (0.0118 vs 0.0008; |r| 0.58; 15-24x its composition-matched control under PCA). Devlog 2026-10-08_d17-composition-matched-controls.
  - Normal-only bases carry the axis (PC8 |r| 0.9936, FA7 |r| 0.9937 at K40).
  - Provenance finding: stored PCA scores are randomized-SVD U·S, not the projection on the stored loadings (PC8 r 0.9913 at K10, 0.9990 at K40); new task W2.7.
- **D18 factor cards** (W5). Bundle `factor_cards_v1` (schema v1): 56 representations (the 47 D14 bases plus IFN-alpha and focus cells), 1,810 cards; 27/27 regression checks against D8, F3, F6, F9, D14 and the atlas.
  - IFN-alpha PC8: orientation −1 (malignant cells score low). Malignant pole TXNIP, IFITM2, TENT5A, NCOA7, TRIM5, ELF1; normal pole IFI27, PSME2, PSMA3, CNP, LAP3, USP18.
  - Identifiability caveats: PC8 relative eigengap 0.055; FA7 determinacy 0.649; PC8-FA7 Tucker phi 0.691.
  - 101 of 248 never-selected non-degenerate factors still reach ≥5 pp univariate held-out lift in at least one supported patient (for example IFN-alpha PCA40 factor 4: 25.40 pp in P02). Degenerate now means zero score variance only (9 FA cards); 117 promax cards carry a separate scale-outlier caveat.
- **D15b patient-count learning curve** (W6), reduced design (588 units; 84/84 reference checks), C and penalty retuned per training set, frozen transductive basis. Tuned median lift from 1 to 6 training patients: PCA40 4.63 → 8.71, FA40 4.45 → 7.03, apoptosis promax40 5.34 → 10.26, allograft promax40 8.79 → 10.12 pp.
  - Median paired gain k = 2 → 6 (same inner-LOPO tuning at both ends): PCA40 2.16 [0.20, 2.48], FA40 0.75 [−0.64, 4.67], apoptosis 1.49 [−0.65, 4.57], allograft 3.06 [0.02, 4.43] pp. Gains over k = 1 mix a tuning-protocol switch (one training patient forces cell-level CV instead of inner LOPO), and the reduced design is unbalanced at k = 2 and 5. The curves are not monotone and no power law is reportable.
  - The full enumeration (1,764 units) is handed to screen `W6_d15b` (CPU, resumable; see campaigns.csv).
- **Apps** (W7).
  - D1 atlas explorer (`scripts/mrd_stage0_2/apps/old34_d1_atlas_explorer.py`, port 8050) now shows a per-fold strip with "lift = 0 by construction" badges for all-zero folds, the selected penalty/C per cell, two new colour modes, and a deep link to the factor cards. `--check`: 12 PASS (original 11 unchanged + 1 new).
  - New read-only factor explorer (`scripts/mrd_stage0_2/apps/old34_factor_explorer.py`, port 8051) over the D18 bundle. `--check`: 11 PASS; `--export-bundle` is about 7 MB, so teammates can browse without EXP.
- **Repo and context engineering** (W0): `AGENTS.md`, the research spine above, the backfilled devlog from 2026-05-15, and W0-W9 workstreams. Full test suite and `research_index.py check` pass (see the devlog entry for the exact counts).

### Already done before today (answers several TODOs below)
- Interactive D1 atlas explorer: 47 Stage 0 panels × 17 Stage 1 configs × the 7 supported patients, with `--check` reproducing the saved D1/D1f matrices and a ~2 MB `--export-bundle`.
- D14 per-patient factor-score overlap (aggregate vs subtype), D15 AUPRC vs number of training cells (both in `stage2_old34_patient_factor_overlap.ipynb`), and F9 (PC8 loadings vs single genes, within-cell-type lift, FA7 across K).

## 2. Here's what's discussed (with corrections)

- **High-prevalence patients dominate.** P03 (32.6%), P02 (14.4%) and P04 (9.0%) are the highest-*prevalence* patients, not the ones with the most cells; together they hold 68% of malignant cells (P03 alone 48%) but only 8.2% of all cells, whereas P09+P01 make up 48% of the label-blind Stage 1 fit. They dominate Stage 2 supervision, cell-weighted selection and lift summaries (lift ranks with prevalence; median within-representation ρ = 0.82; D6, D1DD2).
  - **New:** on IFN-alpha they do not visibly dominate the Stage 1 factors. Their share of score energy is about their cell share (9.1% vs 8.2% for PCA40, ratio 1.11; 8.4% for FA40; the cohort-wide median ratio is 0.96; D16). Leaving them out moves the PC8/FA7 axis by little in absolute terms (matched |r| 0.988-0.997 across method and K). Against random drops that remove the same number of malignant cells, the excess vanishes for FA (0.36-1.00x) and is under 2x for PCA (1.42-1.94x), so it is mostly malignant composition, not these patients. Leaving out P01 moves the axis far more (axis loss 0.0118, |r| 0.58 at PCA40). P09 moves it more in absolute terms but is 3.5x larger; P06 moves it less except at FA40 (D17). One panel; 3 size-matched and 20 composition-matched random controls per drop; W2.4 extends the leverage test to allograft and apoptosis.
- **IFN-alpha, K = 40: PCA covers four patients, FA only P03/P04.** Settled (D19, D17): this is a Stage 2 L1 entry-threshold effect on related directions in the shared IFN-alpha subspace (PC8 vs FA7 |r| 0.68; PC8 has R² 0.92 on the FA40 scores), not missing FA structure.
  - FA7's entry margin is just below 1 in P02 (0.967) and P07 (0.981); PC8's is just above (1.085, 1.073). Raising C by 5% lets FA7 enter: P02 18.06 pp (PCA 15.94), P07 6.97 pp (PCA 11.08).
  - With C tuned per fold, FA40 has positive lift in 7/7 patients under L1, L2 and elastic net. Tuned L1, PCA40 vs FA40: pooled AP 0.1001 vs 0.0894, median lift 8.71 vs 7.03 pp, P20 1.37 vs 1.25 pp; tuned L2 reverses the order (0.0871 vs 0.1325; 5.60 vs 6.38; 1.17 vs 1.85).
  - A residual PCA advantage in P07 (22 malignant cells) depends on the penalty (L1 8.71 vs 4.97, EN 10.16 vs 7.40, L2 5.60 vs 5.07 pp) and is not identifiable from one patient.
  - Team hypothesis (a) (PCA orthogonality vs FA allocating factors to the highest-prevalence patient) is formally retired (H-S2-07).
- **PC8 interpretability.** PC8 is a combination axis with two opposite-sign ISG groups; malignant cells score toward the TXNIP/IFITM2/TENT5A pole (D18 orientation −1; FA7 is the same contrast with sign +1). It is not the best single ISG (single-gene lift vs |loading| ρ = 0.30; F9a), and it is not identifiable as a unique sub-program (small eigengap, low FA determinacy). ADT within label: oriented PC8 (toward malignant, i.e. −1 × raw PC8) tracks CD34 in normal cells (ρ +0.251) and CD99 in malignant cells (ρ +0.222); raw PC8 has the opposite signs — descriptive and confounded by patient and origin.

## 3. Here's the TODOs (old and new), with status

| Item (ask ID) | Status | Where |
|---|---|---|
| Data-driven test for whether to aggregate patients (A-20261008-07..10) | Done on the transductive basis (D14: 45/47 aggregate_shared). Hardening next: promax weighting, all representations, ≥200 permutations, calibrated gates | W1.2 (D20); rerun on D17 bases W1.3 |
| Repo / context engineering (A-20261008-11) | Done; committed locally on `sc_classification_gpu`, not pushed (the `.gitattributes` LFS decision W0.2 must land before any push) | W0 |
| New MRD patients from Yu-Hsiang (A-20261008-12) | Draft email and pre-registered integration plan ready; email not sent | `workstreams/W8_new_cohort/` (W8.1, W8.2) |
| Option 1.1: subset to patients with enough malignant cells (A-20261008-13) | Blocked on new data | W2.5 |
| Option 1.2: re-weight so each patient contributes equally (A-20261008-14) | Piloted on IFN-alpha (D17): patient-balanced weighting rotates PC8 (|r| 0.90 at K40) without lowering transfer. Decide the targeted imbalance before more reweighting | W2.3 |
| Option 1.3: re-examine PCA/FA factors (A-20261008-15) | D16 done (score space); loading leverage piloted (D17) | W1.3, W2.4 |
| Option 2: inductive Stage 1 (A-20261008-16) | Pilot done (D17, IFN-alpha); all 47 D1-selected representations next (GPU, user screen) | W2.6 |
| Option 3: focus on P02/P03/P04 (A-20261008-17) | Folded into the leave-out arm (D17) and the subset pipeline | W2.1 (done), W2.5 |
| Days since transplant / before relapse (A-20261008-18) | Open, low priority | W1.7 |
| Factor interpretability beyond IFN-alpha (A-20261008-19) | Cards done (D18); census of program break-up next | W5.5, W5.2 |
| View interpretability in the app (A-20261008-20) | Done: separate factor explorer, deep-linked from the atlas | W7.2 |
| Correlate factors with cell types / ADT (A-20261008-21) | Partly done (D18: CLR ADT and cell-type tables for the shortlist); dsb and patient-stratified tables open | W5.3 |
| Interpretability frameworks from other fields (A-20261008-22) | Literature review done; frameworks note open | W9 (review A), W5.4 |
| No causal claims without intervention (A-20261008-23) | Built into every claim gate | THEME §6 |
| AUPRC vs number of cells; grant learning curve (A-20261008-24..26) | D15 (cell depth) and D15b reduced (patient count) done; D15b full in screen; D15 v2 and the grant note next | W6.1, W6.2, W6.4 |
| Ankit: semi-supervised joint factor learning (A-20261008-27) | Later, once new data arrive | W3.7 |
| Review single-cell ML literature (A-20261008-28) | Review done (transfer baselines, nested selection, null models) | W9 (review B) |

Screen jobs and servers handed to the user (commands in the devlog entries named):
- `screen -S W6_d15b`, then `bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh` (D15b full design; about 10-15 min on 12 CPU workers; resumable; devlog 2026-10-08_d15b-patient-count-curve).
- Restart `d1_atlas` (port 8050) so it serves the W7.1 code, and start `W7_factor_explorer` (port 8051); devlog 2026-10-08_w7-atlas-strip-factor-explorer. Through an SSH tunnel, forward both ports.

## 4. Current scientific questions (central branch for publication)

1. **Shared or patient-specific?** Do the same factors drive lift across the high-prevalence patients, or different factors in each?
   - *So far:* D14: biological bases can be aggregated, P03 is private only in broad bags. D3 triage over 41 panels (provisional, non-inferential candidate labels): 13 truly shared, 17 shared + patient-specific, 2 patient-only, 7 unresolved, 1 cohort illusion (hypoxia), 1 hidden transfer (UPR). D1f: lift > 5 pp in ≥5 patients for only 84/783 cells. IFN-alpha uses related PC8/FA7 directions across P02/P03/P04/P07, holds within HSC/MPP2, and survives an inductive Stage 1 (D17).
   - *Missing:* inductive bases beyond IFN-alpha (W2.6); support-matched comparisons (W1.5); the program break-up census (W5.5); the new patients as an external test (W8).
2. **How strong is the evidence?** Variance explained, effect size, significance.
   - *So far:* effect sizes (lift, eta2, overlap); label-free variance (IFN-alpha PCA40 explains 54% of gene variance, the first 8 PCs about 21%); robustness to the C policy (D19) and to Stage 1 fit scope (D17); D15b patient-count gains with patient-bootstrap CIs.
   - *Missing:* a matched random gene-set null (D21, W4.2); patient-level uncertainty and an AUROC column (D22, W4.1); nested selection (D23, W3.3); a shared-vs-patient-specific variance decomposition; seed stability (W4.5).
3. **What does DR add over direct genes?**
   - *So far:* best DR beats direct genes on pooled AP in 38/41 panels but on median lift in only 16/41 and on P20 in 17/41 (D9); 30/47 wins when chosen by median lift (D1e/F4); largest gains allograft +12.3, apoptosis +11.5, IFN-alpha +9.8 pp. The old34 prior is not an HVG alias (D12). New: many atlas L1 "wins" sit on the entry knife-edge at the C-grid floor (D19), and pooled AP rewards fixed-C arms with all-zero folds, so pooled AP, median and P20 lift must be read together.
   - *Missing:* nested or prespecified DR selection with tuned C (D23, W3.3, W3.8); single-score baselines such as UCell/AUCell/ssGSEA (D24, W3.4); inductive bases for all representations (W2.6).

**Claim boundary (current; canonical list in THEME §6).**
- *Supported, descriptively:* patient × panel × method/K heterogeneity; lift tracks prevalence; DR improves pooled AP but not consistently median/P20 lift; the IFN-alpha PCA/FA coverage gap is the L1 threshold on related PC8/FA7 directions (holds under tuned C and on inductive bases); biological bases can be aggregated (D14, transductive); IFN-alpha transfer survives an inductive Stage 1 (D17 pilot).
- *Not yet claimable:* biology rather than support/prevalence; FA inferior to PCA; highest-prevalence patients shaping the Stage 1 loadings; DR universally better; panels beating random gene sets; LOPO = inductive deployment in general; "more patients help"; a named factor as a unique program; any causal or recurrence-risk statement.

## 5. Likely questions

- **"So is FA worse than PCA on IFN-alpha?"** No. FA7 is a direction related to PC8 (not the same axis) and sits just under the fixed-C L1 entry threshold in P02 and P07. With a 5% higher C, FA40 covers the same 5 patients as PCA40; with per-fold tuned C it is positive in all 7; under tuned L2 FA40 is ahead on pooled AP, median and P20 lift. Devlog 2026-10-08_d19-regularization-contract (H-S2-08).
- **"Why is a patient's lift exactly 0 for some cells in the atlas?"** The L1 probe in that fold kept no factor, so every cell gets the same score and AP equals prevalence. It is a threshold at the selected C, not absence of signal. The atlas now marks these folds (W7.1); D19 lists the knife-edge cells.
- **"Do P02/P03/P04 shape the factors?"** On IFN-alpha, not visibly. Their score-energy share is close to their cell share (D16). Leaving them out barely moves the PC8/FA7 axis (matched |r| ≥ 0.988), and against random drops that remove the same number of malignant cells the excess is gone for FA and under 2x for PCA. Leaving out P01 moves the axis far more (axis loss 0.0118, |r| 0.58). One panel, 20 composition-matched controls. They do dominate Stage 2 supervision and lift summaries. W2.4 tests allograft and apoptosis.
- **"Isn't LOPO leaking the held-out patient through Stage 1?"** Yes, the stored bases are transductive. On IFN-alpha, under tuned C or L2, the median per-patient change from the transductive refit to the inductive basis is −1.29 to −0.05 pp across the 18 summaries, and single patients lose up to 3.8 pp (D17). Extending this to all 47 D1-selected representations is W2.6 (GPU, user screen).
- **"Can we tell the grant that sequencing more patients helps?"** Not yet. D15b shows lift is already positive with one training patient and gains a few pp by six, with wide CIs, non-monotone curves, a tuning-protocol switch at one patient, and a basis that saw every patient; how a per-subset basis would change the curve is unknown in either direction. What would justify it: an inductive basis per subset (W6.6) and Yu-Hsiang's patients as external points (W6.5, W8). Also W6.4 (grant note).
- **"Which genes drive PC8, and is it 'the IFN program'?"** Malignant cells score toward TXNIP/IFITM2/TENT5A/NCOA7/TRIM5/ELF1 and away from IFI27/PSME2/PSMA3/CNP/LAP3/USP18. It is a combination axis inside the IFN-alpha gene set, not a unique sub-program (small eigengap, FA determinacy 0.649). Open the factor explorer card for `single_geneset__hallmark_interferon_alpha_response`, PCA K40, factor 8.
- **"Is anything significant?"** Not yet tested inferentially. D21 (random gene-set null) and D22 (patient-level uncertainty) are the next W4 tasks; current results are effect sizes on 7 supported patients.
- **"Are the stored atlas numbers still valid after the PCA-score finding?"** Yes for what they are (Stage 2 on the stored scores). The stored PCA scores differ slightly from projecting cells onto the stored loadings, which matters for anyone projecting new cells (W8, W2.6) and for small-K PCA at fixed L1. W2.7 audits the whole atlas.
- **"How do I look at this without the 199 GB experiment?"** Ask for the two export bundles (atlas about 2 MB, factor cards about 7 MB) and run either app with `--data-dir <bundle>`. For everything else, `STATUS.md`, `CLAIMS.md` and the devlog are in git.
- **"Where do I start if I want to pick up a task?"** `AGENTS.md`, then `STATUS.md` §2 (open tasks by priority), then the workstream README. Every task ends with a devlog entry; `python tools/research_index.py new-entry <slug> --workstream W#` scaffolds one.
