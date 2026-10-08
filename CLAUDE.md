# CLAUDE.md

@AGENTS.md

Claude Code specifics (everything else is in AGENTS.md):
- Never use background or detached Bash for Stage 0/1/2, DR/FactoSig or GPU work. Give the user a screen command instead (R1).
- Put throwaway scripts in the session scratchpad. The only persistent write-ups are devlog entries and `scripts/mrd_stage0_2/research/` rows.
- Commit and PR attribution follow the harness instructions. Never commit without the user's go-ahead (R11).
