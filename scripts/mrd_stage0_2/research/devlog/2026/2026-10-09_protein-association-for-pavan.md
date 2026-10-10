---
id: 2026-10-09_protein-association-for-pavan
date: 2026-10-09
type: docs
title: Updated PC8 x surface-protein figure and reply draft for Pavan's question on protein integration (restates D27 for an email; protein is not integrated into Stage 1)
asked_by: teammate:Pavan
asks: [A-20261010-01, A-20261010-02]
implemented_by: claude-code
questions: [QI]
workstream: W5
analyses: [D27]
hypotheses: []
verdicts: []
alts: []
closes: []
advances: []
claim_level: descriptive
status: done
commit: pending
run_head: f5a8f4d
dirty_run: true  # the update directory, D27 code and the report directory are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
Pavan's question, relayed verbatim by the user in the 2026-10-09 session (not yet recorded as an A-ID in research/asks/):
> "have you been able to integrate the protein data?"

The user's request, verbatim (same session):
> "Wrt figure 13, my collaborator Pavan asked "have you been able to integrate the protein data?". I am going to reply that we haven't integrate the protein into the DR, but we can evaluate factors with protein association (not sure if association the correct word to use here), but basically we can email back this screenshot (let's do an update if possible, I need to understand what isotype max |rho| mean and CLR ADT is just the surface protein column from the .h5ad file?"

"Figure 13" and the screenshot are the questions report's PC8 cell-type figure (fig05c_celltype) and its top-ADT table (Table 1 of that screenshot), both from D18 (2026-10-08_d18-factor-cards).

## Why it matters
QI: a collaborator needs a self-contained, correctly bounded answer on protein. The earlier table showed pooled rho only; D27 (2026-10-09_d27-factor-modality-assoc) adds the within-patient, within-cell-type and isotype-floor views that decide how far the surface-protein correlates of PC8 can be read (H-S2-05 confounds), so the email should carry the D27 version, not the D18 one.

## Estimand / null
n/a (docs). The figure restates D27 tables for IFN-alpha PCA K=40 factor 8 (PC8) and FA K=40 factor 7 (FA7): within-label Spearman rho of the D18-oriented score with per-cell CLR ADT, pooled, within each supported patient (median, min-max, same-sign count over 7) and within supported patient x predicted cell type strata (>= 30 cells; median). Two summaries are derived here from the same D27 tables: the isotype floor at each level (largest |rho| over the 9 isotypes for that card, label and level) and, per predicted cell type, the number of supported patients (>= 10 cells of each label in that type) whose malignant mean oriented score exceeds the normal mean. Marker rows shown: the 5 largest |within-patient median rho| per factor x label, the report table's 3 largest |pooled rho| (asserted to equal the report's Table 1), and any marker that meets D27's robust rule (|within-patient median| >= 0.20, same sign in >= 5 of 7) and keeps the sign and at least half of that median inside patient x cell type strata (D27's kept-within-cell-type rule; adds PC8 malignant CD82 and FA7 malignant HLA.DR.DP.DQ). No null; nothing refit; no C policy.

## Implementation
- Code: scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figure.py (reads D27 tables only; imports the palette from scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/report_style.py without editing it; asserts the card-summary isotype maxima, the PC8 orientation -1 and the Table 1 markers).
- Command (agent shell, 2 s): `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figure.py`
- Inputs (EXP-relative): analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_ifna_pc8_fa7_adt.csv, d27_card_summary.csv, d27_strata.csv (pooled n, per-patient n and the patient x cell type strata, read rather than hard-coded), d27_celltype_means.csv, d27_celltype_means_by_patient.csv.gz.
- Outputs in the same update directory: protein_association_pc8.png (200 dpi) and .pdf (renamed from protein_association_pc8_fa7 in the review round, since the figure shows PC8 only; the old files were removed); protein_association_table.csv (PC8 and FA7, selected markers, per-patient rho, the three floors, above-floor flags at each level, robust and kept-within-cell-type flags); protein_association_celltype.csv (panel B); caption.txt (bulleted); email_draft.md (reply in the user's voice; one bracketed optional sentence on joint RNA + protein models); README.md. FA7 is in the CSV and in one caption and one email sentence (a four-sub-panel version was not legible at email size).
- Review round (two reviewers): applied the email rounding fix (CD34 within-patient median 0.2446 is +0.24, not +0.25), the CD99 rewording (weak and suggestive rather than inside the floor), the CD63 exception to the shrinkage sentence, the 19 of 20 / 19 of 21 count, per-level floor wording with the within-cell-type floor printed as not drawn, the malignant strata count (11 strata in 4 patients), an in-figure isotype-floor definition, open squares for pooled values in panel A (panel B keeps open = normal, as in the report), plainer labels and title, a bulleted caption, n read from d27_strata.csv, log(1 + count) in the CLR definition, and the dropped 'yet' in the email. Kept: panel B's glyphs (they match the report figure Pavan saw).
- Index check after the review round: `python tools/research_index.py write` left STATUS, CLAIMS and INDEX unchanged; `check` exits 1 on E14 (H-S2-09 in hypotheses.md has no verdict declaration; added by the concurrent integration work, not by this entry) and warns W06 (D27 is still a planned row while this done entry cites it, until D27 is integrated). No module code changed, so pytest was not rerun.
- Provenance: the D27 devlog entry was staged outside the repo when this entry was written (another job is integrating it); numbers below were re-read from the D27 artifacts.

## Artifacts
none (no EXP writes). The outputs listed under Implementation sit in the update directory in the repo.

## Results
- Definitions for the user. CLR ADT: the raw CITE-seq antibody counts in the cohort cache obsm['ADT'] (161 markers + 9 isotypes; source per AGENTS R8 and the D27 runner's docstring; the cohort cache was the file checked, the raw AnnData was not loaded, per R8), transformed per cell at analysis time as log1p(count) minus the cell's mean log1p over all 170 features; it is not a stored normalized column, and per R8 neither file holds a dsb version. Isotype max |rho|: for one factor and one label, the largest absolute Spearman rho between the factor score and any of the 9 isotype-control antibodies (no specific target), i.e. the size of correlation that non-specific background and the CLR's per-cell centering alone produce.
- PC8, normal cells (58,549 pooled): CD34 pooled +0.251, within-patient median +0.245 (7/7 same sign, range +0.140 to +0.376), patient x cell type median +0.019 (71 strata); CD22 -0.221 / -0.246 (6/7) / -0.046; CD112 +0.213 / +0.154 (7/7) / +0.014; strongest other within-patient markers CD48 -0.215 (6/7), CD39 -0.204 (7/7), CD81 -0.201 (7/7), all with patient x cell type medians between -0.063 and -0.024. Isotype floor: pooled 0.056, within patient 0.109, within patient x cell type 0.043.
- PC8, malignant cells (1,569 pooled; 22 to 758 per supported patient; the 11 patient x cell type strata come from P02 (3), P03 (3), P04 (1) and P09 (4)): CD99 pooled +0.222, within-patient median +0.159 (6/7, range -0.123 to +0.411), patient x cell type +0.150 (same sign in 9 of 11 strata, above that level's 0.120 floor); CD244 +0.209 / +0.207 (5/7) / +0.046; GPR56 +0.203 / +0.137 (6/7) / -0.057. Strongest within-patient markers CD48 -0.308 (6/7), CD116 -0.287 (7/7), CD33 -0.272 (5/7), CD146 +0.270 (6/7), CD63 -0.252 (6/7); their patient x cell type medians are -0.011, -0.037, +0.026, +0.017 and -0.138. Isotype floor: pooled 0.101, within patient 0.227, within patient x cell type 0.120; both within-level floors come from one isotype (RTK2071: +0.227 within patient, +0.120 within patient x cell type), and the next largest within-patient isotype values are 0.158 and 0.157. CD99, CD244 and GPR56 sit below the within-patient floor. Markers meeting the robust rule that keep at least half inside cell type strata: CD63 (-0.252 within, 6/7; -0.138 patient x cell type, 9 of 11) and CD82 (+0.238 within, 7/7; +0.143, 9 of 11; pooled +0.138); CD82 is shown for that reason.
- FA7 (CSV, plus one caption and one email sentence): normal CD22 -0.312, CD34 +0.298, CD112 +0.282 (each 7/7; patient x cell type -0.085, -0.004, +0.035), floors 0.080 pooled, 0.071 within, 0.045 within cell type; malignant strongest CD116 -0.316 (6/7), floors 0.122, 0.257, 0.122 (within-level floors again from RTK2071); the report table's malignant markers GPR56 +0.247 (6/7), CD48 -0.240 (6/7) and CD63 -0.195 (5/7) sit below the within-patient floor; HLA.DR.DP.DQ (-0.222 within, 6/7; -0.120 patient x cell type) is the one FA7 marker kept within cell type.
- Panel B (PC8 cell-type means, oriented cohort z over all 60,118 cells): among normal cells over all 13 types with >= 20 cells, PC8 is highest in EBM +0.793, MEP-MkP +0.676 and HSC/MPP1 +0.577. Malignant cells score higher than normal cells of the same predicted type in 8 of 10 types with patients pooled (exceptions EBM, NP2), and in 19 of 20 within-patient comparisons for those 10 types (supported patients with >= 10 cells of each label; the exception is NP1 in P09); EBM and pDC1 have no qualifying patient. Including pDC2 in P06 (not shown: fewer than 20 malignant cells pooled), 19 of 21.
- Cohort-wide context restated from D27: 729 of 20,274 robust normal-cell card x marker pairs and 1,797 of 8,474 robust malignant-cell pairs keep at least half their within-patient size inside patient x cell type strata.
- Figure checked by eye at full size (2,014 x 1,568 px after the review round) and at a 50% downscale; the PDF renders; no label collisions.

## Claim gate
Supports (descriptive): on the transductive, RNA-only Stage 1 basis with D18 orientation, CLR ADT without dsb and predicted cell types, PC8's normal-cell CD34 association holds within every supported patient but is carried by predicted cell-type composition; PC8's malignant-cell protein associations are weaker and close to its isotype floor at these cell numbers, with CD63, CD82 and CD99 keeping a similar size inside cell types just above the floor that one isotype reaches (restating D27).
Does not support: any claim that protein is integrated into the factors (it is not); any surface protein as a single-cell marker of PC8 or of malignancy; separating patient from batch or malignant biology from recipient origin; conclusions about dsb-normalized ADT; any causal reading; inductive deployment (Stage 1 is transductive; LOPO is label transfer); recurrence risk.

## Next steps
- W5.3: record Pavan's question in research/asks/ with an A-ID (left out here: this task's write scope excluded the asks files); D27 may absorb the two summaries first stated here (the per-level isotype floors and the per-patient cell-type comparison counts); dsb stays blocked on the user's background-source decision.
- Proposed, no task yet (user decision): a joint RNA + protein model (totalVI or MOFA+) as a protein-integrated basis; W1.3 lists MOFA+ only as a sharedness comparator, and nothing has been started.
