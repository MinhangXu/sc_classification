---
id: W8
title: Yu-Hsiang's new MRD patients (email, pre-registered integration plan, readiness, patient inclusion, healthy-cell downsampling)
status: blocked
questions: [Q1, Q2, Q3, meta]
---

# W8 — Yu-Hsiang's new MRD patients: email, pre-registered integration plan, readiness, healthy-cell downsampling

## Charter

User ask, as recorded (`research/asks/2026-10-08_team-report.md`, capture: paraphrase):
- [A-20261008-12] "Incorporate new patient MRD data from Yu-Hsiang (# I need to email him)"

Related asks that depend on this cohort:
- [A-20261008-13] Option 1.1 ("... plus YH's new patient"), task W2.5;
- [A-20261008-18] timing ("YH says a similar manuscript plot exists"), task W1.7;
- [A-20261008-27] Ankit's joint model ("later, once new data arrive"), task W3.7.

Added 2026-10-09 from the user's reading notes on report v1 (`research/asks/2026-10-09_report-reading-notes.md`, capture: verbatim; devlog 2026-10-09_report-reading-notes):
- [A-20261009-11] Which patients stay: "We got 4 patients passing the 100 malignant cell rule. P02, P03, P04, P09. We need to think if P01, P06 and P07 are still relevant to use." Decided on 2026-10-10 (A-20261010-07, below; W8.5 closed). `integration_plan.md` §4.1 keeps the evidence table behind the choice.
- [A-20261009-12] Healthy-cell downsampling: "we just gotta show not only it's the total number of malignant cells giving us a hard time, it's also a ratio problem" ... "if the downsampling of healthy cells improved the classification, we need to understand why?". It is pre-registered in `healthy_downsampling_plan.md` (W8.6, W8.7) and becomes a paper arc (W8.8; `research/paper/OUTLINE.md` R12).
- [A-20261009-14] User decision: "In a new run, ... let's definitely not include patients that're just in the cohort and contribute no malignant cells at least." See `integration_plan.md` §4.0.
- [A-20261009-16] How the aggregation test bears on downsampling. See `healthy_downsampling_plan.md` §2.4.
- User decision, closing line of the same notes: "All the new proposed run idea like downsampling, let's include that when we have the new data from team." Arm D therefore runs with the new data.

Added 2026-10-10 (`research/asks/2026-10-10_protein-and-decisions.md`, capture: verbatim; devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs). The user framed both as "decisions relevant for runs when we have the extra patient data":
- [A-20261010-07] "Let's do the >=100 malignant cells." New full Stage 0-2 runs keep patients with at least 100 malignant cells and both classes: P02, P03, P04 and P09 in the current cohort, and new patients by the same rule. Patients below the rule are left out of every step (the reading the W8.2 lock confirms). The 10-cell support rule in `research/study_config.json` stays for every re-analysis of the current atlas (`integration_plan.md` §4.1).
- [A-20261010-08] "Reweight should target stage 2 label it sounds like that's more relevant". Re-weighting acts on who the probe learns from, not on the Stage 1 basis (W2.3 closed). It runs in Arm D: the Stage-2-only placement is primary, and the sample-weight scheme SW joins the downsampling schemes (`healthy_downsampling_plan.md` §5.1, §5.2).

**Why it matters.** The new patients are the study's only external test:
- **Q1:** do shared-model transfer and factor sharing replicate in patients that played no part in panel, basis, probe or hyperparameter selection?
- **Q2:** more supported patients make patient-level uncertainty possible.
- **Grant:** they become external points on the patient-count curve (W6.5).

**Status.** Blocked until the user sends the email (W8.1).
- The draft email and the pre-registered plan were written on 2026-10-08.
- On 2026-10-09 the plan gained the patient-inclusion section and Arm D. Arm D's own pre-registration is `healthy_downsampling_plan.md`.
- On 2026-10-10 the user decided the inclusion rule (at least 100 malignant cells for new full runs; W8.5 closed) and the re-weighting target (the Stage 2 labels; W2.3 closed), both for the runs on the new data (devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs). The remaining inputs to the lock are the W8.2 items in `integration_plan.md` §14.

**Done means:**
- the plan in `integration_plan.md` is locked by a `type: decision` entry before any new-patient prediction is computed (W8.2);
- the readiness code is tested (W8.3);
- each new patient has a support table (W8.4);
- the patient-inclusion decision is recorded (W8.5; done 2026-10-10);
- the downsampling runner is tested (W8.6) and its plan is locked with the integration plan (W8.2).

The external-validation analysis itself becomes a new backlog line after W8.2.

## Builds on

- Study constants and the support rule: `research/study_config.json` (AGENTS R7).
- The frozen selections:
  - D1 atlas, D1e/D9 (DR vs direct);
  - D19 regularization contract (W3.1);
  - D17 basis variants (W2.1, W2.2);
  - D22 uncertainty (W4.1);
  - D25 label provenance (W1.4);
  - D15b patient-count curve (W6.1, W6.5).
- Hypotheses the plan tests: H-S2-01, H-S2-08, H-S1-04, H-S1-06, H-S2-06 and, through Arm D, H-S2-09. It also informs H-S1-02.
- What Arm D builds on:
  - the first donor downsampling: D10 and D11 (H-S2-04; devlog 2026-08-21_donor-sensitivity-d10-d11);
  - the L1 entry rule and C policy: D19;
  - score energy: D16, plus D28 (cell-type-stratified; devlog 2026-10-09_d28-celltype-score-shares);
  - axis movement under refits: D17, and D26 for allograft rejection and apoptosis (devlog 2026-10-10_d26-leverage-allograft-apoptosis-done), which places the highest-prevalence patients' dominance in the Stage 2 labels;
  - overlap of patients' score distributions: D14;
  - the cell-depth and patient-count curves: D15, D15b, W6.2, W6.3.
- Reviewer alternatives it addresses: ALT-LABEL, ALT-DONOR, ALT-BATCH, ALT-PREV, ALT-NPAT, ALT-PEN, ALT-TRANS, ALT-SEL, ALT-CT and, for Arm D, ALT-NREG (`research/alternatives.csv`).
- Literature: W9 Review B §1, §3 and §5 (`../W9_literature/2026-10-08_scml_transfer_baselines_review.md`). It also covers the cohort preprint and the external-validation protocol.
- Runners that need the opt-in `--patients` filter (W8.3):
  - `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py`;
  - `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`.
- Canonical cohort-cache recipe (filters and normalization): `scripts/mrd_stage0_2/notebooks/stage0_2/_build_deg_hvg_lowsupport_notebook.py`.
- Overlap code reused for H-S1-04: `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py`.

## Backlog

- [W8.1] (P0) The user emails Yu-Hsiang, using workstreams/W8_new_cohort/email_yu-hsiang_draft.md. The agent never sends email — done when: the reply is recorded as an asks file or a meeting entry. (blocked: user sends email)
- [W8.2] (P0) Pre-register the integration and external-validation plan (integration_plan.md), plus a decision entry. Before any data arrive, it fixes the support rule (the W8.5 rule: at least 100 malignant cells for new full runs, written into frozen_config_v1.json, with the user's confirmation that patients below it are left out rather than kept as train-only), Arm P's pre-fitted-probe training set (integration_plan.md §14 item 10), label-pipeline harmonization, the frozen panels, K and C, and the primary metrics. The same entry locks healthy_downsampling_plan.md (Arm D) by its sha256 — done when: the decision entry exists. (depends: W8.5)
- [W8.3] (P1) Readiness code — done when: tests show the default behaviour is unchanged.
  - An opt-in --patients filter on the Stage 0/1 and Stage 2 runners. It is additive, defaults stay unchanged, and it goes in a devlog entry with type: refactor.
  - A new-cohort inventory script.
  - The `--patients` filter is also how every new run applies the user's inclusion rule: no zero-malignant patients (A-20261009-14; integration_plan.md §4.0) and, for new full runs, at least 100 malignant cells (A-20261010-07; §4.1).
- [W8.4] (P1) Ingest the new cohort as a new experiment (a campaigns.csv row and a study_config entry), then run the inventory — done when: a support table exists for each new patient. (depends: W8.2, W8.3) (blocked: new data, W8.1)
- [W8.5] (P0) Patient-inclusion decision for new runs (A-20261009-11, A-20261009-14, A-20261010-07). Decided by the user: patients without malignant cells are excluded from every new run (2026-10-09); new full Stage 0-2 runs keep patients with at least 100 malignant cells and both classes, which is P02, P03, P04 and P09 in the current cohort, and new patients qualify by the same rule (2026-10-10). Patients below the rule are left out of every step; the W8.2 lock asks the user to confirm this reading against the train-only option. The 10-cell support rule in study_config.json is unchanged and governs every re-analysis of the current atlas, so existing results keep their seven folds. Arm P's pre-fitted probes moved to W8.2 (integration_plan.md §14 item 10). (The D26 leverage run, W2.4, ran as built on all patients on 2026-10-10, comparable with D17; devlog 2026-10-10_d26-leverage-allograft-apoptosis-done.) integration_plan.md §4 keeps the counts, prevalence and fold consequences — done when: a type: decision entry records the choice (devlog 2026-10-10_decisions-new-run-inclusion-reweighting-lfs; study_config.json unchanged because the study-wide rule did not change).
- [W8.6] (P1) Healthy-downsampling runner run_healthy_downsampling.py (+ .sh) in this directory, per healthy_downsampling_plan.md §9. It needs --parts, --resume, --merge-only and --out-root. It reuses regularization_contract, stage1_refit and factor_patient_matrix, and adds two additive options, both named in a type: refactor entry with defaults unchanged: full-cell inner validation in tune_c_inner_lopo, and a per-cell weight multiplier on top of the balanced class weights, for the Stage 2 sample-weight scheme SW (the W2.3 re-weighting target, decided 2026-10-10). Tests go in tests/test_healthy_downsampling.py. Before the lock, nothing is run on real labels: smoke runs use within-patient permuted labels — done when: the tests pass and a smoke run with a timing model completes on a scratch --out-root. (depends: W3.1, W2.2)
- [W8.7] (P1) Run the pre-registered healthy-downsampling and Stage 2 re-weighting factorial (Arm D: LOPO part D-L and external part D-X; the Stage-2-only placement is primary and includes the SW scheme; patients per the W8.5 rule) in the user's screens W8_hds_ifna, W8_hds_focus, W8_hds_grid and W8_hds_external. First reserve a D-ID (slug healthy_downsampling_v1) and add a planned campaigns.csv row (R4). C is retuned per condition, with L1 and L2 (R5) — done when: a devlog entry reports the decomposition, the count × ratio grid and the recipe against the criteria in healthy_downsampling_plan.md §7, and declares the H-S2-09 verdict. (depends: W8.2, W8.4, W8.5, W8.6) (blocked: new data, W8.1)
- [W8.8] (P2) Write the downsampling arc into research/paper/OUTLINE.md (R12) and research/paper/FIGURES.md (HDS panels), citing devlog ids only. Decide with the user whether it is a main-paper section or a companion short paper (working title "Machine learning in MRD single-cell data: just downsample the healthy cells?") — done when: OUTLINE R12 cites the W8.7 entry and its status is updated. (depends: W8.7)

## Code in this directory

- `README.md`: this file.
- `email_yu-hsiang_draft.md`: a draft email for the user to edit and send (W8.1). Agents never send it.
- `integration_plan.md`: the pre-registered integration and external-validation protocol (W8.2). It stays a draft until the decision entry locks it with hashes. §4 holds the patient-inclusion rule (decided 2026-10-09 and 2026-10-10) and the evidence table behind it; §8 defines Arm D.
- `healthy_downsampling_plan.md`: the pre-registered design for healthy-cell downsampling (Arm D; H-S2-09). It covers what is known (D10, D11, D19), the "why" channels, the factorial, the read-outs, the success and falsification criteria, the compute plan and the paper angle. It is a draft until the W8.2 lock.
- Planned, not yet written:
  - `run_new_cohort_inventory.py` (W8.3);
  - `frozen_config_v1.json`, written at the W8.2 lock;
  - `run_external_projection.py`, under a backlog line added after W8.2;
  - `run_healthy_downsampling.py` and `run_healthy_downsampling.sh` (W8.6), with `tests/test_healthy_downsampling.py`.
- Screen sessions, from `integration_plan.md` §11:
  - `W8_inventory`
  - `W8_external_projection`
  - `W8_inductive_lopo`
  - `W8_joint_grid`, optional, GPU, only if the user asks.
  - `W8_hds_ifna`, `W8_hds_focus`, `W8_hds_grid` and `W8_hds_external` (Arm D; `healthy_downsampling_plan.md` §8).

## Conventions

- **Artifact slugs owned by W8.** All live in the NEW experiment dir, never the canonical one:
  - `new_cohort_inventory_v1`
  - `new_cohort_cache_v1`
  - `external_projection_v1`
  - `inductive_extended_lopo_v1`, under `stage1_dr_variants/`
  - `healthy_downsampling_v1`, under `analysis/{scorecards,figures,cache}/stage2_diagnostic/` with its refit bases under `stage1_dr_variants/` (Arm D)
  The canonical experiment dir is read-only for W8 (AGENTS R3).
- **C policy (AGENTS R5).** New patients change training n. Every probe is therefore reported at the frozen stored setting and at C tuned by inner LOPO over the training patients (W3.1 contract), with L1 and L2 side by side. Record `c_policy` and `training_n_changed` in each devlog entry.
- **Pre-registration.** No AP, AUROC or probe score may be computed on new-cohort cells before the W8.2 lock entry exists. Anything outside the locked plan is labelled exploratory. Arm D additionally computes nothing on real labels of any cohort before the lock, unless the user chooses the rehearsal in `healthy_downsampling_plan.md` §12 item 1.
- **Patients.** New full runs keep patients with at least 100 malignant cells and both classes (W8.5; A-20261010-07); patients without malignant cells are never in a new run (A-20261009-14). Re-analyses of the current atlas keep the 10-cell support rule of `research/study_config.json` and its seven supported patients. Any comparison between the two names both cohort definitions.
- **Data.** The raw new-cohort file lives under `$SC_DATA_ROOT` and is never committed (R11). New code resolves paths through `sc_classification.utils.paths` (R9).
- **Import header.** Use the one in AGENTS.md §6.
- **Numbers.** Cite devlog entries instead of restating numbers here (R12). The email draft and the plan quote only metadata values checked against the cell metadata.
