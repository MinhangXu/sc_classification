# Post-hoc DR Validation and Evaluation Plan

This document is the canonical guide for post-hoc validation of DR outputs from Plan 0/Plan 1 experiments.

Scope:
- validate metric comparability across methods (`pca`, `fa`, `factosig`, `factosig_promax`, `cnmf`)
- debug suspicious patterns before downstream biological interpretation
- keep a reproducible checklist of what has been validated vs pending

Out of scope:
- replacing Plan 0 / Plan 1 execution docs
- implementation details for runner refactors (covered in sibling engineering plans)

## How this plan fits with existing docs

- `active_plan0_plan1.md` remains the execution-status document for Plan 0/Plan 1.
- `plan0rotationseedsplan1stability.md` remains the engineering implementation plan.
- This file is the evaluation/validation playbook both documents should reference.

## Current status snapshot (2026-02)

- Plan 0 DR comparison notebook exists and now includes scale-normalized diagnostics.
- Plan 0 multi-seed stability caches exist in the referenced experiment directory.
- Plan 1 status is not yet consolidated in a single verifiable artifact set in this workspace snapshot.
- Current practical K narrowing from diagnostics: `K=40` is a strong shortlist candidate; `K=60` appears to provide limited incremental gain in current normalized summaries.

## Validation workstreams

## 1) Metric comparability baseline (Chunk 1)

Goal:
- separate true method behavior from scale-convention artifacts.

Actions:
- compute and export scale-normalized factor metrics across all methods
- keep original raw metrics side-by-side (do not overwrite)
- compare raw vs normalized trends for:
  - variance-share curves
  - top label alignment
  - communality distributions
  - loadings/score scale-transfer diagnostics

Deliverables:
- `within_method_factor_metrics_scale_normalized.csv`
- `within_method_gene_communality_scale_normalized.csv`
- `within_method_k_summary_scale_normalized.csv`
- `within_method_scale_transfer_diagnostics.csv`
- `within_method_k_diagnostics_scale_normalized.png`
- `communality_distribution_by_method_k_scale_normalized.png`
- `scale_transfer_diagnostics_by_method_k.png`

## 2) Rotation diagnostics (Chunk 2)

Goal:
- evaluate varimax/promax behavior directly from saved matrices.

Actions:
- for each method/K, quantify:
  - loading orthogonality (`offdiag(L^T L)` magnitude)
  - latent correlation (`offdiag(cov(scores))` magnitude)
  - simple-structure sparsity proxies (e.g., Gini/Hoyer over factor loadings)
  - conditioning / scale spread (per-factor loading norm and score SD range)
- explicitly compare `factosig` (varimax) vs `factosig_promax`.

Deliverables:
- rotation diagnostics table (CSV)
- compact summary figure(s) for method/K comparisons

## 3) K-behavior debugging (Chunk 3)

Goal:
- ensure changes across `K in {20,40,60}` are mathematically expected, not pipeline bugs.

Actions:
- verify top-component raw variance vs normalized share behavior
- for PCA, cross-check with stored `explained_variance_ratio`
- for FA/FactoSig, inspect score variance conventions and factor ordering choices
- summarize expected vs suspicious patterns by method

Deliverables:
- K-debug summary table (CSV/Markdown)
- “expected behavior” notes appended to evaluation outputs

## 4) Communality comparability checks (Chunk 4)

Goal:
- clarify when communalities are comparable vs method-specific proxies.

Actions:
- report communality definitions used per method family
- compare raw and scale-normalized communality distributions
- flag non-comparable interpretations (especially cNMF and oblique rotations)

Deliverables:
- communality interpretation note
- updated figure captions/caveats for downstream reuse

### Chunk 4 audit (current data)

What is already feasible without reruns:
- compare raw vs scale-normalized communality distributions across existing Plan 0 artifacts
- document definition-level caveats:
  - PCA/FA/FactoSig communalities are derived from loadings in a linear latent-factor model family
  - cNMF communality is a spectra-based proxy and not directly equivalent in absolute scale
- add interpretation guardrails to prevent over-reading absolute cross-method communality magnitude

What may require additional runs (optional, for stricter comparability):
- rerun all methods with fully harmonized source conventions (same artifact source path per method, same normalization convention at save-time)
- add method-matched NMF/cNMF variants if needed for direct communality-family comparisons
- extend to Plan 1 preprocess variants to test whether communality conclusions are robust to gene-selection strategy

Current recommendation:
- proceed with Chunk 3 debugging + interpretation finalization first
- treat Chunk 4 as “interpretation caveat + optional hardening reruns” rather than a blocker for narrowing from `K=60` toward `K=40`

## 5) Stability and consensus quality gate (Chunk 5)

Goal:
- establish minimum stability criteria before using factors for biological claims.

Actions:
- aggregate pairwise stability summaries by method/K
- include consensus metrics where available
- define and apply a pass/warn/fail gate per method/K

Deliverables:
- quality-gate table for method/K shortlist

## 6) Plan 1 readiness gate (Chunk 6)

Goal:
- ensure downstream preprocess x DR comparisons inherit validated DR choices.

Actions:
- confirm selected method/K settings from validated Plan 0 diagnostics
- confirm Plan 1 run provenance (preprocess definitions, CV settings, seeds if used)
- ensure evaluation caveats are carried into Plan 1 interpretation

Deliverables:
- Plan 1 readiness checklist
- explicit mapping from Plan 0 decisions to Plan 1 config

## Suggested operating pattern

- Keep this file as the only place that tracks evaluation chunk status.
- Keep execution docs focused on run commands/artifacts, and link here.
- For each completed chunk, append:
  - date
  - experiment directory
  - output files
  - short decision note

## Chunk status tracker

- [x] Chunk 1: metric comparability baseline (scale-normalized diagnostics)
- [x] Chunk 2: rotation diagnostics
- [~] Chunk 3: K-behavior debugging (implemented in notebook; pending final interpretation pass)
- [~] Chunk 4: communality comparability checks (audit + caveats drafted; optional hardening reruns identified)
- [ ] Chunk 5: stability and consensus quality gate
- [ ] Chunk 6: Plan 1 readiness gate
