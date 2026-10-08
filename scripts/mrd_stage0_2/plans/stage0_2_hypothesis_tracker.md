# Stage 0–2 hypothesis tracker (agent / ChatGPT contract)

**Open the interactive table:**  
`/home/minhang/.cursor/projects/home-minhang-mds-project/canvases/stage0-2-hypothesis-tracker.canvas.tsx`

That canvas is the working progress board for the knowledge-prior Stage 0–2 study and the parent Aim 2 statement. This note exists so a web-ChatGPT discussion or a later Cursor agent can find the table and update it instead of starting a second tracker.

## Two different “stage 0–2” meanings

- **Grant / Aim 2:** clinical colorectal cancer stages 0–2; endpoint is patient-level MRD / recurrence vs histopathology.
- **This pipeline:** analysis Stage 0 (gene-space panels) → Stage 1 (shared latent basis) → Stage 2 (regularized malignant-vs-normal probes), plus interpretability and an Aim 2 translation row.

Do not present cell-level LOPO lift as recurrence-risk improvement.

## How to place a ChatGPT result

1. Open the canvas beside chat.
2. Go to **Inbox**.
3. Paste the claim / implementation / missing control.
4. Pick the stage row it belongs on (Stage 0 / 1 / 2 / interpretability / Aim 2 translation).
5. **Save to inbox**, then **Place on table**.

Placed rows are candidate overlays in canvas state. They are not compiled findings until an agent merges them into the `STAGES` / `HYPOTHESES` / `AIM2` arrays in the `.canvas.tsx` file and bumps `SNAPSHOT`.

## How a Cursor agent should update the table

After implementing something that changes evidence, figures, or missing pieces:

1. Edit the matching compiled row in the canvas file (`verdict`, `evidence`, `figures`, `missing`, `nextAction`, and the parent stage summaries).
2. Bump `SNAPSHOT`.
3. If the work came from the Inbox, mark that item merged (or delete the placed overlay once it lives in compiled DATA).
4. Do **not** launch GPU Stage 0/1/2 from the agent shell. Long jobs go in `screen`.
5. Do **not** overwrite root Set A biological scorecards (`analysis/scorecards/stage2_*.csv`).

Prompt stub for a new agent chat from the canvas:

> Implement the next missing piece from the Stage 0–2 hypothesis tracker, then update the compiled table in `stage0-2-hypothesis-tracker.canvas.tsx`.

## Canonical experiment

`sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093`

Science reading order: `../notebooks/stage0_2/old34_inquiry_narrative_guide.md`  
Live campaign status: `current_experiment_status_rehydration_20260904.md`
