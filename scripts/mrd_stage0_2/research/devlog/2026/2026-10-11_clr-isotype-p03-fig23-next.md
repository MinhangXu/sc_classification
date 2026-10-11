---
id: 2026-10-11_clr-isotype-p03-fig23-next
date: 2026-10-11
type: analysis
title: Questions report v3.3 - CLR, isotype controls and dsb explained; how isotypes track each cell's total ADT count (Figure 28); why P03 sits near 0 (Table 4); Figure 23 extended (inside-type/within panel, a Simpson-type example, lift vs cell type); and where the protein work goes next (protein in the classifier, protein in the DR step)
asked_by: user
asks: [A-20261011-01, A-20261011-02, A-20261011-03, A-20261011-04, A-20261011-05, A-20261011-06, A-20261011-07, A-20261011-08, A-20261011-09, A-20261011-10, A-20261011-11, A-20261011-12, A-20261011-13]
implemented_by: claude-code
questions: [QI, Q1]
workstream: W5
analyses: [D27, D29]
hypotheses: [H-S2-05]
verdicts: []
alts: []
closes: []
advances: [W5.3]
claim_level: descriptive
compute: agent-cpu
c_policy: not_applicable
training_n_changed: false
status: done
commit: pending
run_head: 592aa7d
dirty_run: true  # the figure modules, the report and the asks file are uncommitted
experiment: 20260525_060508_stage0_mrd_old34_broad_screen_82db5093
backfilled: false
date_source: authored
supersedes: []
---

## Question
One message from the user, 2026-10-11, after reading questions report v3.2; recorded verbatim in research/asks/2026-10-11_clr-isotype-dsb-fig23-next.md (A-20261011-01..13). Excerpts:
> "We should specify what does CLR mean? Also previous the explanation for Isotype antibody as control was "they bind to nothing specific". I wish for a bit more explanation such that when I email my teammate, I presenting it correctly (succinct)." (A-20261011-01, -02) "When you say the isotypes mostly track each cell's total ADT count I tried to look at fig27 and it's not showing isotype's rho with cell total ADT count, can we have a figure for it? Why would we want to rerun on dsb values? Please explain." (A-20261011-03, -04) "Should we use the plots/figures from the new normalization technique?" (A-20261011-05) "how P03 sits near 0 on a few of them, how can we interpret this observation?" (A-20261011-06) "But what does that say about promax f7? Is it even useful in classification? What does "(sign of the within patient median)" on the x-axis mean?" (A-20261011-07, -08) "Would you say/show if malignant cells are more often to have this than normal cells? Does this have any useful biological insight? ... should we look at the cell type within patient median vs. within patient median? What's the SImpson's paradox patter you are talking about?" (A-20261011-09..-11) "we should be able to broaden the scope to look into other factors that are selected by L1 ... And improve Fig 23 from my comments above." (A-20261011-12) "I just feel like after stating the results really accurately makes the impact feel super weak ... how can we make improvement from this. Definitely include the factors in the DR step as a model that we are going to try?" (A-20261011-13)

## Why it matters
QI and H-S2-05 (ALT-CT): the user is about to explain the protein results to collaborators, so the terms (CLR, isotype controls, dsb) must be stated correctly and briefly, the isotype yardstick's mechanism shown, and the patient pattern behind the pooled values (P03) explained. The scope question (A-20261011-13) decides the next protein work: reading RNA factors with protein afterwards can at best name a factor, so the stronger tests put protein into the classifier and into Stage 1 (proposed W3.9, W2.10).

## Estimand / null
No new estimand beyond D27 and D29; every number is read from their stored tables or recomputed per cell and asserted against them. New summaries: (1) per isotype and supported patient, Spearman of isotype CLR with log1p(total raw ADT count), inside label; (2) for one isotype in one patient (RTK4530, P09 malignant), the same for log1p(raw count) and for CLR; (3) among D27 robust pairs (|median within-patient rho| >= 0.20, >= 5 of 7 same sign), the share with |pooled| < 0.5 |within| and with |within| - |pooled| >= 0.20, proteins and isotype rows, per label; (4) per card, the median over the 7 supported patients of D18's univariate held-out lift (factor alone, sign fixed on training folds), against D27's within-patient cell-type eta2 (normal cells), Spearman over 1,801 cards; (5) per supported patient, D27's per-patient cell-type eta2 of PC8 (malignant); (6) for CD133 x HVGs (3000) FA40 factor 16 (oriented, D18 sign -1), per-patient rho, mean score and mean CLR in malignant cells. No null, no test; c_policy not_applicable (nothing refit).

## Implementation
- scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_05_pc8.py: fig05f_modality_all now panels a-d (new d: inside-type / within-patient ratio for robust pairs; x-axis of a-b relabelled "rho x sign of the within-patient median"; no method colours); print_every_card_numbers() adds (3) and the count of cards with median lift >= 5 pp.
- scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_05q_cd63_depth.py: fig05f2_modality_more (Figure 23 second page: e Simpson-type example, f-g eta2 panels, h median lift vs cell-type eta2), fig05t_isotype_depth (Figure 28), table4_p03 (prints Table 4's LaTeX rows; asserts the raw values against D27). Every per-cell rho is asserted against D27 (tolerance 1e-5).
- Command (lab env, agent shell, about 20 s per section): `CONDA_PREFIX=$HOME/miniconda3/envs/sc-classification-rapids-py311 $HOME/miniconda3/envs/sc-classification-rapids-py311/bin/python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/make_figures.py 05 05q`, then `bash .../build.sh --all --no-figures`.
- Report v3.3 (sections/05_what_is_pc8.tex: box "Three terms in plain words", "Why P03 sits near 0" + Table 4, "Every card" answers, Figure 23 over two pages, Figure 28, "Should we switch", subsection "Where the protein work goes next"; A1 glossary: ADT (CLR), dsb, isotype reference; core page 8: CLR and isotypes in one sentence, still 12 pages). email_draft_v4.md revised with the definitions and the dsb reason.

## Artifacts
- none in EXP (report figures only: questions report figures/fig05f_modality_all, fig05f2_modality_more, fig05t_isotype_depth; gitignored). Inputs: analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/, factor_cards_v1/, protein_depth_contrast_v1/, stage1_dr/panel_id=hvg_top_requested_3000__available_3000/method=fa/k=40/seed=42/scores.npy, the cohort cache.

## Results
- [H-S2-05 qualifies] Isotype CLR vs log total ADT inside a patient (median of 7): healthy -0.113 (G0114F7) to -0.415 (RTK4530); malignant -0.317 (MOPC.173) to -0.707 (RTK4530). One isotype, RTK4530 in P09's 375 malignant cells: log1p(raw count) vs log total ADT +0.049 (the raw count is 0 or 1 in almost every cell), CLR -0.738. Isotype x PC8 median within-patient rho, malignant, CLR -> total ADT fixed: 0.048-0.158 -> -0.009-0.048 for eight isotypes; RTK2071 0.227 -> 0.181.
- [H-S2-05 qualifies] Why P03 sits near 0 (PC8, malignant cells): PC8's cell-type eta2 inside P03 0.046 (P02 0.084, P04 0.156, P09 0.211); PC8 vs log total ADT P03 +0.023 (P01 -0.419, P02 -0.200, P04 -0.358, P09 -0.414); largest isotype |rho| P03 0.07 (P09 0.31, falling to 0.05 with total ADT fixed). P03 per protein (CLR / total ADT fixed): CD48 +0.03/+0.02, CD116 -0.03/-0.03, CD33 +0.07/+0.06, CD146 -0.02/0.00, CD63 0.00/0.00, CD82 +0.14/+0.14, CD99 +0.18/+0.18; CD63 with total ADT fixed in P01, P02, P04, P06, P09: -0.18 to -0.31. In P09 total ADT fixed moves CD33 -0.32 -> -0.07, CD116 -0.31 -> -0.11, CD146 +0.28 -> +0.06.
- Robust pairs whose pooled |rho| is below half the within-patient one: normal proteins 580 of 20,274 (2.9%), isotypes 8 of 365 (2.2%); malignant proteins 3,273 of 8,474 (38.6%), isotypes 91 of 282 (32.3%). Gap >= 0.20: normal 72 (0.4%), isotypes 0; malignant 1,200 (14.2%), isotypes 3 (1.1%).
- Simpson-type example, CD133 x HVGs (3000) FA40 f16, malignant: P01 +0.699 (45 cells), P02 +0.598 (211), P03 +0.020 (758), P04 +0.786 (102), P06 +0.093 (44), P07 +0.409 (22), P09 +0.671 (375); pooled +0.181, median +0.598. Patient means (score, CLR CD133): P03 (0.53, -0.56), P09 (0.14, 1.90), P02 (0.99, 2.19).
- Inside-type / within-patient ratio over robust pairs: normal median 0.149 (729 of 20,274 >= 0.5), malignant 0.264 (1,797 of 8,474).
- [H-S2-05 qualifies] Innate immune context FactoSig-promax K40 f7 (the CD123 row of Figure 23a): CD123 normal cells pooled +0.711, within +0.708 (7/7), inside type +0.112; cell-type eta2 within patient 0.805 (pooled 0.796); highest cDC (+1.32 cohort z), NP2, ncMono, pDC2; lowest CLP (-1.10), MEP-EP; univariate held-out lift P02 14.4 pp, P03 6.6 pp, others <= 1.0 pp, median 0.8 pp; not a selected driver (dense probe). Over 1,801 cards, median lift vs cell-type eta2 (normal): Spearman +0.069; 41 cards have median lift >= 5 pp; PC8 11.1 pp (eta2 0.105), FA7 7.0 pp (0.173); median of the 119 drivers 0.63 pp, of all cards 0.31 pp.

## Claim gate
Supports (descriptive): in these data the isotypes' CLR values mostly follow each cell's total ADT count, so the isotype yardstick mainly measures that path; P03's near-zero PC8 protein correlates are explained by little cell-type variation along PC8 and no total-count path in P03, except CD63; factor-protein relations in malignant cells are often shared in direction but offset by patient; factors with lineage-protein correlates do not give more lift across patients.
Does not support: that holding total counts fixed is the right correction; that any protein marks malignancy; that the patient offsets are clone biology rather than staining batch (patient, sample and batch are nested); that protein would or would not improve classification (untested: W3.9, W2.10).

## Next steps
- W3.9 (new, P1): protein in the classifier (ADT-only and RNA+ADT LOPO probes, C retuned).
- W2.10 (new, P1): protein in the DR step, MOFA+ then totalVI as new Stage 1 bases (packages installed by the user; screen jobs; campaigns row).
- W5.9 (new, P2): the 119 L1-selected drivers in detail (two checks, total-ADT control, protein profile).
- W5.10 (new, P2): the factor-free malignant-vs-healthy protein contrast with patient-level inference.
- W5.8, W9.4 as before; the lab questions are in email_draft_v4.md (MX sends).
