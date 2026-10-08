# research/legacy: frozen provenance

Everything in this directory is **frozen and read-only**. Files here are copies of
context that used to live outside the cloneable repo. They are kept so that a
collaborator who clones only `sc_classification` can see where the current research
spine came from.

- Never edit these files. A correction goes in a new devlog entry (`research/devlog/`),
  and a changed statement goes in `research/hypotheses.md`.
- `tools/research_index.py` does not lint this directory: E07 (absolute or home paths)
  and the other overlay checks skip `research/legacy/**`. The verbatim copies may
  therefore contain lab-server paths; do not copy those paths into overlay files.
- Nothing here is canonical. The canonical tracker is `research/hypotheses.md` plus the
  verdicts declared in devlog front matter. The canonical claim list is `research/THEME.md` §6.

## Contents

| File | Source | Copied | Notes |
|---|---|---|---|
| `stage0-2-hypothesis-tracker_2026-10-02b.canvas.tsx` | Cursor canvas `stage0-2-hypothesis-tracker.canvas.tsx` in the user's Cursor project canvases directory (under `$HOME/.cursor`, outside any repo) | 2026-10-08 | Byte-identical copy of the canvas at `SNAPSHOT = "2026-10-02b"`, made **before** the 2026-10-08 FROZEN banner edit. It holds the 19 migrated hypothesis rows (H-S0-01..H-A-02), the 5 stage rows, the CRC Aim 2 rows and the working-hypothesis text. |

The original canvas outside the repo now carries a comment-only FROZEN banner that points
to `scripts/mrd_stage0_2/research/hypotheses.md`. Its data arrays and `SNAPSHOT` were not
changed, so this copy and the original differ only in that header comment.

## Not copied here

- The 2026-08-25 repository audit (audit report, open decisions, science-to-repo matrix,
  run inventory, artifact lineage and the D9 scaler-cancellation table) is tracked by the
  outer umbrella repo at `reports/mrd_stage0_2_repository_audit/`. It is referenced by that
  outer-repo path (see `research/THEME.md` §8 and §11) rather than duplicated here.
- The canvas sidecar state file (`stage0-2-hypothesis-tracker.canvas.data.json`) held only
  unmerged inbox state and was not copied.
