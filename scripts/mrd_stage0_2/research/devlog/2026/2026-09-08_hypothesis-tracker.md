---
id: 2026-09-08_hypothesis-tracker
date: 2026-09-08
type: docs
title: Living Stage 0-2 hypothesis tracker canvas, agent/ChatGPT contract and Cursor rule
asked_by: user
asks: []
implemented_by: unknown
questions: [meta]
workstream: W0
analyses: []
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: none
status: done
commit: b60a6ee  # retro: tracker contract first committed in b60a6ee
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: true
date_source: mtime  # contract md and outer Cursor rule 2026-09-08 15:43 UTC
supersedes: []
---

## Question
(reconstructed from the tracker contract and the 2026-10-08 chronology reconstruction) The user asked for one progress board where ChatGPT-derived claims and agent implementations land, with pipeline Stage 0/1/2 kept separate from the grant's clinical CRC stage 0-2 (Aim 2). The prompt text was not saved; no ask ID.

## Why it matters
Meta: this was the progress board until 2026-10-08. It lived outside the repo (Cursor-only), and its rule forbade an in-repo tracker; the 2026-10-08 research spine reverses that.

## Estimand / null
n/a (docs).

## Implementation
- Cursor canvas stage0-2-hypothesis-tracker.canvas.tsx (outside the repo) with STAGES / HYPOTHESES / AIM2 arrays, a ChatGPT inbox and a SNAPSHOT bump protocol. Verbatim copy: scripts/mrd_stage0_2/research/legacy/stage0-2-hypothesis-tracker_2026-10-02b.canvas.tsx.
- In-repo pointer: scripts/mrd_stage0_2/plans/stage0_2_hypothesis_tracker.md; plans INDEX updated on 2026-09-10.
- Glob-scoped Cursor rule in the umbrella repo: .cursor/rules/stage0-2-hypothesis-tracker.mdc.

## Artifacts
none

## Results
- The canvas holds 19 hypothesis rows (H-S0-01 to H-A-02) with verdict, evidence, figures, missing, next action and claim gate; the last SNAPSHOT is 2026-10-02b.
- The Aim 2 rows (ctDNA, recurrence, SHAP/LIME) are marked grant_gap or not started.
- The rule's line 'Do not create a second tracker markdown that duplicates the table.' conflicts with an in-repo devlog; it is reversed in 2026-10-08_research-spine-adopted.

## Claim gate
Supports (descriptive): nothing; organizing layer only.
Does not support: reading canvas verdicts after 2026-10-08 as current (the canvas is frozen).

## Next steps
- W0.1: land the in-repo spine that replaces the canvas.
