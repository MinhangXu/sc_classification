---
id: TODO: <YYYY-MM-DD>_<slug>, equal to the file name stem
date: TODO: YYYY-MM-DD (UTC; equals the file name date)
type: TODO: analysis | run | decision | meeting | fix | refactor | tool | docs
title: TODO: one line
asked_by: TODO: user | PI | meeting:YYYY-MM-DD | meeting:unknown | teammate:<Name> | agent | not_recorded
asks: TODO: [A-YYYYMMDD-NN, ...] or []
implemented_by: TODO: human | claude-code | codex | cursor-agent | copilot | gemini-cli | aider | other | unknown
questions: TODO: non-empty subset of [Q1, Q2, Q3, QI, meta]
workstream: TODO: W0..W9 (must exist under workstreams/)
analyses: TODO: [registry IDs created or updated] or []
hypotheses: TODO: [H-IDs whose evidence this entry changes] or []
verdicts: TODO: [H-ID=supported_descriptive|supported_publication|open|not_supported|retired|grant_gap] or [] (only when a verdict changes)
alts: TODO: [H-ID:ALT-ID=open|partial|addressed] or [] (the ALT must be listed in that hypothesis)
closes: TODO: [W<n>.<m> tasks completed] or []
advances: TODO: [W<n>.<m> tasks progressed] or []
claim_level: TODO: descriptive | inferential | publication | none
compute: TODO: none | agent-cpu | notebook-cpu | screen-cpu | screen-gpu (analysis and run entries only)
c_policy: TODO: global_selected | per_fold_tuned | nested | fixed_plus_l2 | not_applicable (analysis and run entries only)
training_n_changed: TODO: true | false (analysis and run entries only; true requires C retuning, R5)
status: TODO: done | partial | blocked | abandoned | superseded
commit: TODO: pending | none | 7-40 hex (leave pending; filled after the user commits)
run_head: TODO: short hex of git HEAD when the computation ran
dirty_run: TODO: true | false (tree dirty when the computation ran)
experiment: TODO: experiment id | none
backfilled: TODO: false (true only for retro entries, which also need date_source)
date_source: TODO: authored | header | git | mtime | log | reconstructed
supersedes: TODO: [earlier devlog ids this entry corrects or completes] or []
---

## Question
TODO: the verbatim question in a '>' blockquote, with its source and ask ID (A-YYYYMMDD-NN). If it was not recorded, write '(reconstructed from <source>)'. Never invent a quote.

## Why it matters
TODO: 1-3 sentences: which Q and H-ID this serves, and which decision it unblocks.

## Estimand / null
TODO: estimand, unit of analysis, null or comparator, and the decision rule fixed before looking. State the C policy whenever Stage 2 is refit.

## Implementation
TODO: repo-relative code paths and the entry point or command; inputs; key parameters (method, K, penalty, C policy); where it ran (agent shell, or the user's screen session with its literal screen -S <W#>_<topic> line); runtime and provenance notes.

## Artifacts
TODO: bullets with paths relative to the experiment dir (analysis/..., stage1_dr_variants/..., stage2_supervised/..., logs/...), or 'none'.

## Results
TODO: headline numbers with units; this is the only place numbers live. Tag evidence bullets as '- [H-ID supports|qualifies|contradicts] text'.

## Claim gate
Supports (descriptive): TODO: what this result supports, in THEME §6 vocabulary (descriptive, inferential or publication).
Does not support: TODO: what it does not support (AGENTS.md R6 language).

## Next steps
TODO: bullets, each citing a W<n>.<m> task (add new tasks to the workstream backlog); include any screen commands the user must run.
