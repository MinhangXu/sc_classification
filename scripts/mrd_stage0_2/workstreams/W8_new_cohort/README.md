---
id: W8
title: Yu-Hsiang's new MRD patients (email, pre-registered integration plan, readiness)
status: blocked
questions: [Q1, Q2, meta]
---

# W8 — Yu-Hsiang's new MRD patients: email, pre-registered integration plan, readiness

## Charter

User ask, as recorded (`research/asks/2026-10-08_team-report.md`, capture: paraphrase):
- [A-20261008-12] "Incorporate new patient MRD data from Yu-Hsiang (# I need to email him)"

Related asks that depend on this cohort:
- [A-20261008-13] Option 1.1 ("... plus YH's new patient"), task W2.5;
- [A-20261008-18] timing ("YH says a similar manuscript plot exists"), task W1.7;
- [A-20261008-27] Ankit's joint model ("later, once new data arrive"), task W3.7.

**Why it matters.** The new patients are the study's only external test:
- **Q1:** do shared-model transfer and factor sharing replicate in patients that played no part in panel, basis, probe or hyperparameter selection?
- **Q2:** more supported patients make patient-level uncertainty possible.
- **Grant:** they become external points on the patient-count curve (W6.5).

**Status.** Blocked until the user sends the email (W8.1). The draft email and the pre-registered plan were written on 2026-10-08.

**Done means:**
- the plan in `integration_plan.md` is locked by a `type: decision` entry before any new-patient prediction is computed (W8.2);
- the readiness code is tested (W8.3);
- each new patient has a support table (W8.4).

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
- Hypotheses the plan tests: H-S2-01, H-S2-08, H-S1-04, H-S1-06 and H-S2-06. It also informs H-S1-02.
- Reviewer alternatives it addresses: ALT-LABEL, ALT-DONOR, ALT-BATCH, ALT-PREV, ALT-NPAT, ALT-PEN, ALT-TRANS, ALT-SEL and ALT-CT (`research/alternatives.csv`).
- Literature: W9 Review B §1, §3 and §5 (`../W9_literature/2026-10-08_scml_transfer_baselines_review.md`). It also covers the cohort preprint and the external-validation protocol.
- Runners that need the opt-in `--patients` filter (W8.3):
  - `scripts/mrd_stage0_2/stage0_panels/run_stage0_mrd_old34_broad_screen.py`;
  - `scripts/mrd_stage0_2/stage2_supervised/run_stage2_mrd_multiobjective_scorecard.py`.
- Canonical cohort-cache recipe (filters and normalization): `scripts/mrd_stage0_2/notebooks/stage0_2/_build_deg_hvg_lowsupport_notebook.py`.
- Overlap code reused for H-S1-04: `scripts/mrd_stage0_2/stage2_supervised/patient_factor_overlap.py`.

## Backlog

- [W8.1] (P0) The user emails Yu-Hsiang, using workstreams/W8_new_cohort/email_yu-hsiang_draft.md. The agent never sends email — done when: the reply is recorded as an asks file or a meeting entry. (blocked: user sends email)
- [W8.2] (P0) Pre-register the integration and external-validation plan (integration_plan.md), plus a decision entry. Before any data arrive, it fixes the support rule, label-pipeline harmonization, the frozen panels, K and C, and the primary metrics — done when: the decision entry exists.
- [W8.3] (P1) Readiness code — done when: tests show the default behaviour is unchanged.
  - An opt-in --patients filter on the Stage 0/1 and Stage 2 runners. It is additive, defaults stay unchanged, and it goes in a devlog entry with type: refactor.
  - A new-cohort inventory script.
- [W8.4] (P1) Ingest the new cohort as a new experiment (a campaigns.csv row and a study_config entry), then run the inventory — done when: a support table exists for each new patient. (depends: W8.2, W8.3) (blocked: new data, W8.1)

## Code in this directory

- `README.md`: this file.
- `email_yu-hsiang_draft.md`: a draft email for the user to edit and send (W8.1). Agents never send it.
- `integration_plan.md`: the pre-registered integration and external-validation protocol (W8.2). It stays a draft until the decision entry locks it with hashes.
- Planned, not yet written:
  - `run_new_cohort_inventory.py` (W8.3);
  - `frozen_config_v1.json`, written at the W8.2 lock;
  - `run_external_projection.py`, under a backlog line added after W8.2.
- Screen sessions, from `integration_plan.md` §11:
  - `W8_inventory`
  - `W8_external_projection`
  - `W8_inductive_lopo`
  - `W8_joint_grid`, optional, GPU, only if the user asks.

## Conventions

- **Artifact slugs owned by W8.** All live in the NEW experiment dir, never the canonical one:
  - `new_cohort_inventory_v1`
  - `new_cohort_cache_v1`
  - `external_projection_v1`
  - `inductive_extended_lopo_v1`, under `stage1_dr_variants/`
  The canonical experiment dir is read-only for W8 (AGENTS R3).
- **C policy (AGENTS R5).** New patients change training n. Every probe is therefore reported at the frozen stored setting and at C tuned by inner LOPO over the training patients (W3.1 contract), with L1 and L2 side by side. Record `c_policy` and `training_n_changed` in each devlog entry.
- **Pre-registration.** No AP, AUROC or probe score may be computed on new-cohort cells before the W8.2 lock entry exists. Anything outside the locked plan is labelled exploratory.
- **Data.** The raw new-cohort file lives under `$SC_DATA_ROOT` and is never committed (R11). New code resolves paths through `sc_classification.utils.paths` (R9).
- **Import header.** Use the one in AGENTS.md §6.
- **Numbers.** Cite devlog entries instead of restating numbers here (R12). The email draft and the plan quote only metadata values checked against the cell metadata.
