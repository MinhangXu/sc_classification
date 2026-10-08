import {
  Button,
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Select,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
  TextArea,
  TextInput,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
  useMemo,
} from "cursor/canvas";

/**
 * Stage 0–2 hypothesis tracker — compiled scientific record.
 *
 * HOW ANOTHER AGENT UPDATES THIS FILE
 * -----------------------------------
 * 1. Edit the compiled arrays below (STAGES, HYPOTHESES, AIM2). Do not invent
 *    a second source of truth in the git repo.
 * 2. After a ChatGPT discussion produces a new testable claim, figure, or
 *    implementation: either the user Places it from the Inbox (canvas state),
 *    or you merge that inbox item into HYPOTHESES / STAGES and set
 *    `inboxStatus: "merged"`.
 * 3. When you implement a missing piece, update the matching hypothesis:
 *    verdict, evidence, figures, missing, nextAction, and the parent STAGE
 *    summaries. Bump SNAPSHOT.
 * 4. Never relaunch GPU Stage 0/1/2 from the agent shell. Long jobs go in
 *    `screen`. Do not overwrite root Set A biological scorecards.
 * 5. Clinical CRC “stage 0–2” ≠ pipeline Stage 0/1/2. Keep that distinction
 *    in any new row you add.
 *
 * Inbox / placed rows live in canvas state (sidecar .canvas.data.json) until
 * merged here. Display = compiled HYPOTHESES + unmerged placed overlays.
 */

const SNAPSHOT = "2026-10-02b";
const EXP_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093";
const REPO = "/home/minhang/mds_project/sc_classification";
const EXP = `${REPO}/experiments/${EXP_ID}`;
const NB = `${REPO}/scripts/mrd_stage0_2/notebooks/stage0_2`;
const PLANS = `${REPO}/scripts/mrd_stage0_2/plans`;
const S0 = `${REPO}/scripts/mrd_stage0_2/stage0_panels`;
const S2 = `${REPO}/scripts/mrd_stage0_2/stage2_supervised`;
const CANVAS =
  "/home/minhang/.cursor/projects/home-minhang-mds-project/canvases/stage0-2-hypothesis-tracker.canvas.tsx";

type StageId = "s0" | "s1" | "s2" | "interp" | "aim2";
type WorkStatus = "done" | "in_flight" | "partial" | "not_started" | "blocked";
type Verdict =
  | "supported_descriptive"
  | "open"
  | "not_supported"
  | "grant_gap"
  | "in_flight";
type HypKind = "hypothesis" | "implementation" | "figure" | "missing" | "note";
type InboxStatus = "inbox" | "placed" | "merged" | "dropped";

type FileRef = { label: string; path: string };
type FigureRef = { id: string; note: string; path: string };

type StageRow = {
  id: StageId;
  name: string;
  role: string;
  aim2: string;
  status: WorkStatus;
  hypothesesSummary: string;
  evidenceNow: string;
  figuresSummary: string;
  missingSummary: string;
  nextAction: string;
  files: FileRef[];
};

type Hypothesis = {
  id: string;
  stageId: StageId;
  statement: string;
  verdict: Verdict;
  evidence: string;
  figures: FigureRef[];
  missing: string;
  nextAction: string;
  claimGate: string;
};

type InboxItem = {
  id: string;
  createdAt: string;
  stageId: StageId | "unassigned";
  kind: HypKind;
  title: string;
  body: string;
  source: "chatgpt" | "user" | "agent";
  status: InboxStatus;
};

type PlacedHypothesis = Hypothesis & { fromInboxId: string; sourceNote: string };

const WORKING_HYPOTHESIS =
  "The current classification of stage 0–2 colorectal cancer (CRC) relies on histopathological assessment, which fails to identify patients at high risk of recurrence. We hypothesize that integrating molecular features (genomic and transcriptomic) with standard pathology using a machine-learning framework will improve the prediction of minimal residual disease (MRD) and recurrence risk, enabling more personalized treatment strategies.";

const AIM2: { id: string; title: string; text: string; pipelineMap: string; status: WorkStatus }[] = [
  {
    id: "2.1",
    title: "Data integration",
    text: "Consolidate clinicopathological, genomic (e.g. ctDNA, SNVs), and transcriptomic (gene-expression) data from the stage 0–2 cohort.",
    pipelineMap: "Stage 0 today is transcriptomic knowledge-prior panels only. No clinicopath, ctDNA, or SNV matrix is in the 20260525 experiment.",
    status: "partial",
  },
  {
    id: "2.2",
    title: "Model training",
    text: "Use ensemble or deep-learning architectures to identify key biomarkers associated with recurrence.",
    pipelineMap: "Implemented as Stage 1 DR (PCA/FA/FactoSig) plus Stage 2 L1/L2/elastic-net logistic paths — a biological probe, not an ensemble/DL recurrence model.",
    status: "partial",
  },
  {
    id: "2.3",
    title: "Evaluation",
    text: "Assess AUC-ROC and F1 against traditional staging methods.",
    pipelineMap: "Current estimands are cell-level malignant-vs-normal AP / lift vs prevalence, not patient-level recurrence vs TNM. AUROC is specified but not the suite default.",
    status: "partial",
  },
  {
    id: "2.4",
    title: "Interpretability",
    text: "Apply SHAP or LIME to name influential features and disease-progression biology.",
    pipelineMap: "Coefficient paths, factor ablation, and Stage 1 loadings→genes are in place (D4–D5, D8). SHAP/LIME are not implemented.",
    status: "partial",
  },
];

const STAGES: StageRow[] = [
  {
    id: "s0",
    name: "Stage 0 — gene-space panels",
    role: "Fix the biological hypothesis space before DR or classification. A panel is a prior, not a result.",
    aim2: "2.1 transcriptomic slice",
    status: "partial",
    hypothesesSummary:
      "Malignant structure is recoverable inside curated priors; those priors are not just HVG aliases; budget-matched controls are the fair test.",
    evidenceNow:
      "Set A D1 taxonomy is now explicit: 34 individual sets, 7 biology-group unions, 2 knowledge-union controls, 4 HVG controls. D12: old34∩HVG3k Jaccard 0.199. D1DD6: allograft∩apoptosis is 15/270 (Jaccard 0.056). A mean-z one-score proxy is weaker than selected DR for allograft/TNF and intermediate for apoptosis; it is not UCell/AUCell/ssGSEA. Set B 632 panels / 294 Stage 2 shortlist.",
    figuresSummary: "D1DD1 grouped panel taxonomy; D1DD6 focus-set overlap; D1DD9 one-score proxy; D1 control rows; D12 Venns; V2F1 / V2F6",
    missingSummary:
      "Null Model 1 (size-matched random gene sets); per-panel size-matched HVG; exact UCell/AUCell/ssGSEA comparator; HVG 200 never existed as a Stage 0 panel; all_filtered 15,560-gene bag has no Stage 2; huCIRA; Set B / pooled still unread.",
    nextAction:
      "After pooled Stage 2 merges, compare the four bags to Set A full_34. Do not relaunch the 6-panel control grid. D2/D3 stay biological-only.",
    files: [
      { label: "Stage 0 runner", path: `${S0}/run_stage0_mrd_old34_broad_screen.py` },
      { label: "Value-added plan", path: `${PLANS}/stage0_geneset_value_added_workflow.md` },
      { label: "Set A scorecard", path: `${EXP}/analysis/scorecards/stage0_mrd_old34_broad_scorecard.csv` },
    ],
  },
  {
    id: "s1",
    name: "Stage 1 — shared latent basis",
    role: "Decompose each Stage 0 panel into shared axes (PCA / FA / FactoSig / promax, K∈{5,10,20,40}) or a direct-gene fallback.",
    aim2: "2.2 representation half",
    status: "partial",
    hypothesesSummary:
      "A named program is not one axis. Method and K change transfer. Direct genes are the no-DR control. The basis is transductive.",
    evidenceNow:
      "643 Set A biological representations. D1DD: among 41 in-sample best-median DR cells, K40 wins 24, K20 13, K10 4; method winners are promax 12, FA 11, PCA 9, FactoSig 9. Independent K fits are not nested prefixes, especially for FA/rotations. PCA total-variance and FA common-variance diagnostics are now separated. Direct-gene is 47/47 on the D1 atlas (41 biological + 6 controls from old34_controls_direct_gene_v1). On the 41 biological panels, best DR wins pooled LOPO AP in 38/41 but median lift in only 16/41 and P20 in 17/41. D14: on those D1-selected bases, per-patient factor-score overlap is high (normal median 0.84, malignant median 0.61). Patients are not disjoint subtypes. P03 malignant cells are nearly disjoint on one factor of HVG 3k, HVG 10k, and full_34.",
    figuresSummary: "D1DD2/4 patient and focus K trajectories; D1DD7/8 label-free PCA/FA variance; D1DD9 one-score proxy; D1 atlas/e/f; D7A/B; D9A–D; D14A–E patient factor overlap; V2F2/F2c",
    missingSummary:
      "Multi-seed Stage 1; inductive basis that holds patients out of DR; a same-K40 prefix/subspace sensitivity if a nested-rank claim is desired; nested selection so “best DR” is not winner’s-curse; Set B / pooled method×K notebooks.",
    nextAction:
      "Keep D9 as the no-DR contract. Any new “DR adds value” claim must report pooled, median, and P20 together, or nest selection.",
    files: [
      { label: "D1–D3 notebook", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { label: "D1a deep dives", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
      { label: "D9 follow-up", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { label: "D14 patient factor overlap", path: `${NB}/stage2_old34_patient_factor_overlap.ipynb` },
    ],
  },
  {
    id: "s2",
    name: "Stage 2 — regularized probes",
    role: "Three views on the same panel+basis: discovery full-cohort, LOPO sharedness, patient-specific. Regularization is part of the assay.",
    aim2: "2.2 + 2.3 cell-level probe",
    status: "in_flight",
    hypothesesSummary:
      "PI Q1 shared transfer; PI Q2 patient-specific axes; PI Q3 programs split. Lift is still entangled with prevalence/support.",
    evidenceNow:
      "D1DD unpacks allograft, apoptosis, and TNFα before aggregation. Selected median lifts are 12.56/11.52/18.06 pp versus direct 0.23/~0/12.78 pp, but patient patterns differ. Selected-DR cell ranks reconstruct stored AP within 3.2e-8 and now have post-hoc patient/cell-type/ADT audits. These use a prevalence-matched rank cutoff, not an operational threshold. Set A/control/direct/donor grids remain complete; pooled Stage 2 status is separate.",
    figuresSummary: "D1DD3–5 patient focus panels; D1DD9 one-score comparison; D1DD10–12 cell-rank/cell-type/ADT audits; V2A/F1/F4; D1e/f; D3; D6–D11; D15A–B training-cell AUPRC curve",
    missingSummary:
      "Operational threshold chosen without held-out labels; detailed treatment history; conditional cell-type/donor analysis; pooled merge + analysis; AUROC beside AP; patient-level CIs; nested/prespecified C; Set B theme synthesis. Control no-DR is now on D1; controls still lack matched-size random-gene nulls and D2/D3 pairing.",
    nextAction:
      "Leave the pooled screen session alone until merged scorecards appear. Do not relaunch old34_controls_fullgrid_v1 or old34_controls_direct_gene_v1.",
    files: [
      { label: "Stage 2 runner", path: `${S2}/run_stage2_mrd_multiobjective_scorecard.py` },
      { label: "D1a patient/cell audit", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
      { label: "Live status", path: `${PLANS}/current_experiment_status_rehydration_20260904.md` },
      { label: "Set A LOPO scorecard", path: `${EXP}/analysis/scorecards/stage2_sharedness_lopo_scorecard.csv` },
    ],
  },
  {
    id: "interp",
    name: "Interpretability + claim gate",
    role: "Ground selected factors in genes, separate descriptive findings from publication claims, and keep the metric contract honest.",
    aim2: "2.4 (loadings, not SHAP)",
    status: "partial",
    hypothesesSummary:
      "Selected axes can be named. Constant-score AP equals prevalence. Publication claims still need nulls, uncertainty, and nested selection.",
    evidenceNow:
      "D1DD adds exact selected-score reconstruction, focus-panel overlap, a labeled mean-z one-score proxy, and per-patient cell-type/ADT error profiles. F2d grounds allograft selected factors (LCK/ZAP70/TNF co-load on promax factor 005). F3 shows IFN-α PCA8 vs FA7/8 is signal splitting + L1 collapse, not collinear-FA L1 picking. D4–D5 grounding; D8 IFN-α ablation; D6 AP=π; D10 all-zero collapse; corrected D11 stability.",
    figuresSummary: "D1DD6–12; F1–F3 2026-09-17 feedback notebook; D4–D5; D6A–D; D8B–E; D10E; D11B/C",
    missingSummary:
      "Exact rank-based pathway-score comparator; SHAP/LIME if Aim 2.4 is literal; matched-size random-gene p-values; larger-cohort identification of biology vs support; inductive Stage 1 sensitivity. Allograft/apoptosis loadings are now plotted, not yet a biological write-up.",
    nextAction:
      "Use the claim gate as the default close of any presentation. Do not quote the old “both bases ≈0.99 at 25%” sentence.",
    files: [
      { label: "Narrative guide", path: `${NB}/old34_inquiry_narrative_guide.md` },
      { label: "D1a deep dives", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
      { label: "Team review", path: `${NB}/stage2_old34_team_review_narrative.ipynb` },
      { label: "Layer 4–5", path: `${NB}/stage2_old34_layer4_5_interpretability_suite.ipynb` },
      { label: "2026-09-17 team feedback dives", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
  },
  {
    id: "aim2",
    name: "Aim 2 clinical translation",
    role: "Grant endpoint: patient-level MRD / recurrence risk that beats histopathology and informs treatment.",
    aim2: "2.1–2.4 as stated",
    status: "not_started",
    hypothesesSummary:
      "Multi-modal + ML improves MRD and recurrence prediction over pathology staging. The running experiment does not yet test this.",
    evidenceNow:
      "The 20260525 run is a cell-level malignant-vs-normal knowledge-prior probe on CITE/CN.label. Labels are not recurrence. Features are not clinicopath/ctDNA. Models are not ensemble/DL. Metrics are not vs TNM.",
    figuresSummary: "None for recurrence / staging comparison",
    missingSummary:
      "Patient-level recurrence labels; clinicopath + genomic/ctDNA join; model family beyond regularized logistic; evaluation vs traditional staging (AUC-ROC, F1); treatment-utility operating point.",
    nextAction:
      "Do not blur this row with Stage 2 LOPO lift. Any ChatGPT plan that jumps to recurrence belongs here as a new implementation, not as a rewrite of PI Q1–Q3.",
    files: [
      { label: "Three-stage design", path: `${PLANS}/three_stages_knowledge_prior_mal_classification.md` },
      { label: "Value-added (relapse 4-class still planned)", path: `${PLANS}/stage0_geneset_value_added_workflow.md` },
    ],
  },
];

const HYPOTHESES: Hypothesis[] = [
  {
    id: "H-S0-01",
    stageId: "s0",
    statement:
      "Malignant-vs-non-malignant structure is recoverable inside curated prior gene spaces (single genesets and biology groups), not only in broad HVG bags.",
    verdict: "supported_descriptive",
    evidence:
      "Set A 41 biological panels show transferable lift in immune/inflammatory programs. D1DD separates 34 individual sets from 7 biology-group unions and broad controls; a three-panel mean-z one-score proxy is only a preliminary pathway-score comparator.",
    figures: [
      { id: "V2A", note: "discovery vs LOPO map", path: `${NB}/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` },
      { id: "V2F1", note: "PI Q1 transfer floor vs typical lift", path: `${NB}/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` },
      { id: "D1/D2", note: "643-rep atlas → 41-panel view", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { id: "D1DD1/9", note: "panel taxonomy and mean-z one-score proxy", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
    ],
    missing: "Null Model 1 plus per-panel size-matched HVG and exact UCell/AUCell/ssGSEA before claiming prior-specific value.",
    nextAction: "Read D1 control rows as technical floors, not as the biological null. Matched-size random gene sets are still missing.",
    claimGate: "Descriptive for old34. Not “curated priors beat any N genes.”",
  },
  {
    id: "H-S0-02",
    stageId: "s0",
    statement: "The old34 GSEA dictionary is not an alias of HVG 3,000.",
    verdict: "supported_descriptive",
    evidence: "D12: Jaccard 0.199 vs HVG3k; 65% of the old34 union is exclusive. D1 now places full_34, core_only, and HVG 500/1k/3k/10k as a separate top block with both DR and no-DR (89/96 control DR median lift > 0; all 6 control direct-gene median and p20 lifts > 0). They are technical floors, not interchangeable aliases of the 41 programs.",
    figures: [
      { id: "D12", note: "Venns and per-program HVG enrichment", path: `${NB}/stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` },
    ],
    missing: "Per-panel size-matched HVG (and random) controls through the same Stage 2 grid, not only fixed HVG anchors.",
    nextAction: "D1 now shows those six control rows with DR and no-DR. Keep them namespaced; do not merge into root scorecards. Still no HVG 200 or all_filtered Stage 2.",
    claimGate: "Supported as “not interchangeable Stage 0 panels.”",
  },
  {
    id: "H-S0-03",
    stageId: "s0",
    statement:
      "Bottom-up manuscript-axis panels (Set B atomic / family union / leave-one-family-out) identify which dictionaries add malignancy signal.",
    verdict: "open",
    evidence:
      "Set B Stage 0/1 + Stage 2 shortlist+controls finished June 2026 (294 panels). expanded_full (11,358 genes) was not selected for full Stage 2. Almost no notebook analysis.",
    figures: [
      { id: "Set B scorecards", note: "expanded_stage0_mrd_manuscript_axes_v1", path: `${EXP}/analysis/scorecards/expanded_stage0_mrd_manuscript_axes_v1` },
    ],
    missing: "Set B reading of V2A/V2F1/V2F4/V2F6; expanded_full Stage 2 only if explicitly requested (large).",
    nextAction: "Port the lift-based figure contract to Set B before any new Set B GPU grid.",
    claimGate: "Compute exists; scientific claim does not.",
  },
  {
    id: "H-S0-04",
    stageId: "s0",
    statement:
      "Pooled old34∪expanded bags (no per-geneset identity) still carry shared malignancy, and size-matched HVG 11,421 is the fair bag-level control.",
    verdict: "in_flight",
    evidence:
      "Stage 0/1 + quick L2: 68 representations / 4 panels (2026-09-03). Full Stage 2 not merged as of 2026-09-08. Overlap QC: 2,656 shared genes; 63 old34-only; pooled 11,421.",
    figures: [
      { id: "overlap QC", note: "pooled_stage0_unions_v1", path: `${EXP}/analysis/reports/pooled_stage0_unions_v1` },
    ],
    missing: "Merged discovery / LOPO / patient-specific CSVs, then a notebook that does not mix them into root Set A tables.",
    nextAction: "Watch pooled screen; resume in that same session if it dies. Do not start another 8-GPU job until merge.",
    claimGate: "No Stage 2 claim until merge.",
  },
  {
    id: "H-S1-01",
    stageId: "s1",
    statement: "Stage 1 method and K are not interchangeable inside a panel; transfer can flip with geometry × regularization.",
    verdict: "supported_descriptive",
    evidence:
      "D1/D2 and V2F2 show method×K heterogeneity. D1DD shows K40/20/10 win 24/13/4 panels with balanced method winners, and explains that independently refitted FA/rotated factors are not nested prefixes. F1 (2026-09-17/18): biological median-of-medians lift rises with K for all four methods; HVG 500/1,000 PCA peak at K=10 then drop; HVG FA is not monotonic. F1d–f gray bars are share of that K-fit (always sum to 1) and are not comparable across methods as % of data; panel titles now print captured % (HVG PCA K=40 still ~42% of gene variance). HVG relative knees can look similar for PCA vs FA while IFN-α PCA is much flatter — do not mix F1d with F1e. IFN-α PCA40 vs FA40 remains the grounded interaction example.",
    figures: [
      { id: "V2F2/F2c", note: "method × K, patient-summary AUPRC/lift", path: `${NB}/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` },
      { id: "D7D", note: "per-patient PCA40 vs FA40 lift", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "D8B–E", note: "active coefficients, ablation, FA match", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "D1DD2/4/7/8", note: "patient K trends and PCA-total vs FA-common variance", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
      { id: "F1a–h", note: "K-lift, HVG exception, method×K knees, PCA nesting", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
    missing: "Seed repeats and a same-basis prefix/subspace sensitivity; do not conclude FA is intrinsically inferior to PCA.",
    nextAction: "Any new method comparison must keep C / selection policy matched and report collapse separately from geometry.",
    claimGate: "Descriptive interaction, not a method ranking.",
  },
  {
    id: "H-S1-02",
    stageId: "s1",
    statement: "A no-DR (direct-gene) probe on the same Stage 0 panel is the right Stage 1 control. DR is not universally additive.",
    verdict: "supported_descriptive",
    evidence:
      "D1 atlas direct-gene is 47/47 (41 biological + 6 controls). On 41 biological panels, best DR > direct on pooled LOPO AP in 38/41; median lift in 16/41; P20 in 17/41 under D9 selection. F4a (same D1 matrix, column 1 vs max of 16 DR cells): Δ>0 on 28/41 biological + 2/6 controls. Largest median Δ: allograft promax K40 +12.3 pp, apoptosis promax K40 +11.5, IFN-α PCA K40 +9.8. Most negative: HVG 10k −26.0, all Stage 0 −14.3, HVG 3k −12.0. F4b: per-patient median Δ is +12.2 pp in P03, ~+6 in P02/P04, ~0.3–0.7 in P01/P06/P07/P09; P07 has Δ>0 in only 28/47 panels.",
    figures: [
      { id: "D1a/b/c", note: "783-cell atlas; 6 control rows on top with DR + no-DR; direct-gene 47/47", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { id: "D1e", note: "patient × Stage 0; method|K on left y-axis only; D1e-b is direct_gene including 6 controls", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { id: "D9A–D", note: "direct-gene vs selected DR (pooled LOPO AP winner, not D1a median)", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "D1DD5/9", note: "focus patient-paired DR/direct and one-score proxy", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
      { id: "F2a–d / F4a–b / F5a–b", note: "patient DR vs direct; Δ heatmap; immune Jaccard + per-patient maps", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
    missing: "Nested or prespecified DR selection; patient-specific direct-gene grid (never run). D2/D3 still use the June 5 643-row table.",
    nextAction: "Quote all three estimands together. Do not lead with 38/41 alone.",
    claimGate: "Not “DR universally adds value.”",
  },
  {
    id: "H-S1-03",
    stageId: "s1",
    statement:
      "The across-patient Stage 1 basis lets discovery / LOPO / patient-specific coefficients live in one coordinate system — but LOPO is label transfer on a transductive atlas, not inductive deployment.",
    verdict: "supported_descriptive",
    evidence:
      "Design note in the three-stage plan. Stage 1 saw all eligible cells, including later held-out patients. Feedback notebook §8: a full 47×4×4×7 inductive grid is a campaign; IFN-α PCA40/FA40 × 7 patients on 92 genes is notebook-scale and is still not run. Donor-downsampling D11 is not inductive (held-out cells remain in the basis).",
    figures: [
      { id: "design", note: "transductive caveat", path: `${PLANS}/three_stages_knowledge_prior_mal_classification.md` },
      { id: "§8", note: "inductive vs transductive split of compute", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
    missing: "Even the narrow IFN-α PCA/FA K=40 × 7 inductive refit.",
    nextAction: "If we run one inductive check, run IFN-α PCA40/FA40 only. Do not launch the full grid from the agent shell.",
    claimGate: "LOPO ≠ raw-expression deployment benchmark.",
  },
  {
    id: "H-S1-04",
    stageId: "s1",
    statement:
      "Patients can be aggregated on a shared Stage 1 basis when per-patient factor-score distributions overlap. Disjoint supports would be separate subtypes.",
    verdict: "supported_descriptive",
    evidence:
      "D14 uses stored transductive scores, not a refit. On 41 D1-selected biological bases, normal-cell weighted histogram overlap is 0.78–0.89 (median 0.84) and malignant-cell overlap is 0.54–0.66 (median 0.61). Every normal-cell permutation rejects exact exchangeability; the median malignant gap versus that null is 0.06. Zero panels meet the separate-subtype gate. Allograft is a location shift (P02 normal overlap 0.58). P03 malignant cells are nearly disjoint on one high-variance factor of HVG 3k (overlap 0.013), HVG 10k (0.007), and full_34 (0.009).",
    figures: [
      { id: "D14A–E", note: "panel overlap, permutation gap, focus patients, P03 HVG factor, cell-type gap", path: `${NB}/stage2_old34_patient_factor_overlap.ipynb` },
    ],
    missing: "Inductive refit (patients held out of DR). The P03 broad-bag factor is not yet separated from donor, composition, or technical effects.",
    nextAction: "Aggregate biological program bases. Do not read HVG 10k factor 21, HVG 3k factor 27, or full_34 factor 30 as shared malignant programs.",
    claimGate: "Not “patient identity does not affect scores,” and not “D3 patient-specific gaps are subtypes.”",
  },
  {
    id: "H-S2-01",
    stageId: "s2",
    statement:
      "PI Q1 — a robust shared model exists: high discovery AP, high aggregate LOPO, consistent per-patient lift, stable selected factors.",
    verdict: "open",
    evidence:
      "Immune/inflammatory panels transfer above prevalence for typical supported patients, but lift still ranks with prevalence. D1DD3–5 reverse the seven-patient aggregation for allograft/apoptosis/TNF: large median gains are concentrated differently by patient and median-optimal K need not maximize p20. D1e/f remain the atlas-level companions. No matched-gene-set p-value or patient-level CI.",
    figures: [
      { id: "V2F1", note: "median lift vs P20 floor", path: `${NB}/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` },
      { id: "V2A", note: "shared_candidate quadrant (median split, relative)", path: `${NB}/stage2_mrd_figure3_sharedness_suite_v2_jun4.ipynb` },
      { id: "D1e", note: "patient × Stage 0; method|K on left y-axis only; D1e-b is direct_gene including 6 controls", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { id: "D1f", note: "same D1 axes; color = # of 7 patients clearing lift or AUPRC/π cutoff", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { id: "D6B–D", note: "lift vs support", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "D1DD3–5", note: "focus patient/configuration maps and paired direct comparison", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
    ],
    missing: "Null Model 1; patient bootstrap; AUROC column; do not use median-split quadrants as biological verdicts.",
    nextAction: "Manuscript path is lift-based V2F1/V2F4, not raw AUPRC heatmaps (V2B1 retired).",
    claimGate: "Shared-candidate is a triage tag, not a publication claim.",
  },
  {
    id: "H-S2-02",
    stageId: "s2",
    statement:
      "PI Q2 — patient-specific models can reveal biology that the shared model misses, rather than repeating it.",
    verdict: "open",
    evidence:
      "D3 patient-specific gap; IL2/STAT5 FA K=10 is a design example (low discovery, high P02). D13: P01/P06 have DE (157/232 genes) but best existing held-out lift is only 6.3/5.7 AP points; P07 EMT is DE-enriched and the best existing transfer (19.0). D14: those D3 tags are not disjoint factor supports. Biological factor-score distributions still overlap.",
    figures: [
      { id: "D3", note: "shared vs patient-specific opportunity", path: `${NB}/stage2_old34_comprehensive_diagnostic_suite.ipynb` },
      { id: "D7B", note: "patient × panel × DR", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "D13", note: "low-support patient DE vs existing lift", path: `${NB}/stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb` },
      { id: "D14", note: "factor-score overlap is not the D3 gap", path: `${NB}/stage2_old34_patient_factor_overlap.ipynb` },
    ],
    missing: "Cannot yet separate patient biology from support/prevalence. Need matched-support, malignant-cell subsampling, larger cohort.",
    nextAction: "State non-identifiability whenever a single-patient peak is shown.",
    claimGate: "Not “high lift is caused by patient-specific biology.”",
  },
  {
    id: "H-S2-03",
    stageId: "s2",
    statement:
      "PI Q3 — a prior program breaks into distinct regulatory groups: different factors, different patients, partially overlapping discovery/LOPO/patient-specific selections.",
    verdict: "supported_descriptive",
    evidence:
      "IFN-α case study is the worked example: PCA008 is the L1 driver in 5/7 supported folds (not P01/P09). FA L1 C=1e-4 is all-zero in 5/7 folds; L2 on the same stored FA scores recovers P02/P07. F6: FA7 is the strongest train-fold univariate FA coordinate in every patient, including P02/P07; within-patient r(PCA8,FA7)≈−0.69 to −0.70. Held-out FA7 univariate lift is actually higher than PCA8 in P02 (+18.1 vs +15.9 pp) and lower in P07 (+7.0 vs +11.1; n_malignant=22). Train AP of FA7 is ~1 pp weaker than PCA8, which is what L1 sees. F7: stored FA L1 rankings are undefined when all-zero; FA local L2 rank-recall on P02 is 0.34 vs PCA L1 0.32, so P02 is a regularizer artifact. P07 FA L2 recall is 1/22. sklearn FA scores are uncorrelated (max |r|≈0.003); PCA8 splits across FA7/FA8. Promax is the oblique method.",
    figures: [
      { id: "D8", note: "IFN-α factor attribution", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "D4–D5", note: "factor→gene grounding", path: `${NB}/stage2_old34_layer4_5_interpretability_suite.ipynb` },
      { id: "F3a–f / F6a–c / F7a–b", note: "PCA/FA identifiability; P02/P07 univariate+cell type; rank/ADT", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
    missing: "Systematic program-break catalog beyond IFN-α; Set B theme synthesis (V2F6).",
    nextAction: "Treat IFN-α as the template, not the only program.",
    claimGate: "One grounded case, not a census of programs.",
  },
  {
    id: "H-S2-04",
    stageId: "s2",
    statement:
      "Donor-healthy abundance can induce a shortcut; downsampling donor-healthy training cells tests that.",
    verdict: "not_supported",
    evidence:
      "At 25% donors the selected L1 C=1e-4 solution is all-zero, so AP returns to π by construction. That is sparse-path collapse, not a retained donor-vs-malignant confusion. FA007 stays aligned at 25% (score 0.952, Jaccard 0.818); PCA008 moves (0.854, Jaccard 0.333).",
    figures: [
      { id: "D10E", note: "all-zero frequency — show first", path: `${NB}/stage2_old34_donor_downsampling_sensitivity.ipynb` },
      { id: "D11B/C", note: "corrected supervised-factor stability", path: `${NB}/stage2_old34_donor_downsampling_sensitivity.ipynb` },
    ],
    missing: "Rescue analysis with condition-specific regularization selection or rescaling.",
    nextAction: "Never quote the old “both bases ≈0.99 at 25%” line.",
    claimGate: "Not “the model is invariant to donor-versus-recipient origin.”",
  },
  {
    id: "H-S2-05",
    stageId: "s2",
    statement:
      "Patient- and cell-level error structure can explain which cells create an aggregate focus-panel lift before mechanistic interpretation.",
    verdict: "supported_descriptive",
    evidence:
      "D1DD reconstructs all 21 selected-DR focus-panel × supported-patient rankings within 3.2e-8 AP, then reports prevalence-matched rank errors by predicted cell type, collection timing, QC, and 170 CITE-seq ADTs. F7 adds IFN-α PCA40 vs FA40: skip all-zero L1 rankings; P02 PCA L1 and FA L2 both mark FN malignant cells as higher HLA-DR / CD45 / CD38 / CD58 than TP. P07 ADT contrasts are unstable (3 vs 1 TP). Detailed treatment-regimen history is absent.",
    figures: [
      { id: "D1DD10–12", note: "rank-error overview, cell-type map, seven patient ADT audits", path: `${NB}/stage2_old34_d1a_descriptive_deep_dives_20260910.ipynb` },
      { id: "F7a–b", note: "IFN-α PCA L1 vs FA L2 rank-recall and ADT FN−TP", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
    missing:
      "An operating threshold selected without held-out labels; direct-gene cell-rank reconstruction; conditional cell-type/donor analysis; treatment-regimen join; formal protein tests.",
    nextAction:
      "Use error-enriched cells and ADTs to nominate factor/loading review, then test donor/cell-type alternatives rather than calling the patterns mechanistic.",
    claimGate:
      "Rank-defined TP/FN/FP/TN are post-hoc descriptive groups, not clinical misclassifications.",
  },
  {
    id: "H-S2-06",
    stageId: "s2",
    statement:
      "On a frozen Stage 1 basis, held-out AUPRC rises with the number of Stage 2 training cells. That is a depth curve for this cohort, not a forecast for sequencing a new patient.",
    verdict: "supported_descriptive",
    evidence:
      "D15 refits the atlas-selected probe on label-stratified training subsets and scores the held-out patient in full. Full-data refits match the D1 atlas within 0.0002 AUPRC. Apoptosis FactoSig-promax K=40 (L2) median lift is 7.7 pp at 500 cells, 9.3 at 1,000, and 11.5 on the full set. Allograft promax K=40 stays at prevalence through 2,000 cells, then 1.7 / 3.1 / 8.8 / 12.6 pp at 5k / 10k / 25k / full. IFN-α PCA and FA K=40 use L1 C=1e-4 and stay at prevalence through 25,000 cells; the atlas pattern appears only on the full training set (PCA median lift 11.1 pp, FA median 0). Same all-zero collapse as H-S2-04. Every 500-cell draw still contains all 12 training patients.",
    figures: [
      {
        id: "D15A–B",
        note: "per-patient AUPRC and cohort median lift vs training-cell count",
        path: `${NB}/stage2_old34_patient_factor_overlap.ipynb`,
      },
    ],
    missing:
      "C retuned inside each subsample; an inductive Stage 1 refit; a curve that drops whole patients rather than cells.",
    nextAction:
      "Quote apoptosis as nearly saturated by ~1,000 training cells. Quote the IFN flat line as the atlas L1 C staying all-zero, not as absence of signal.",
    claimGate:
      "Not a Stage 1 learning curve and not evidence that sequencing another patient would move AUPRC.",
  },
  {
    id: "H-I-01",
    stageId: "interp",
    statement: "Constant-score average precision equals malignant prevalence exactly under the implemented metric.",
    verdict: "supported_descriptive",
    evidence: "Proved and empirically checked (D6). Gray baseline must be named “constant-score no-skill AP = π,” not a finite random permutation.",
    figures: [
      { id: "D6A", note: "cohort composition", path: `${NB}/stage2_old34_meeting_followup_metric_interactions.ipynb` },
      { id: "guide §1", note: "proof", path: `${NB}/old34_inquiry_narrative_guide.md` },
    ],
    missing: "AUROC (baseline 0.5) as a prevalence-free companion column.",
    nextAction: "Keep AP vs one clinical cutoff as separate sentences.",
    claimGate: "Metric contract, not a biology claim.",
  },
  {
    id: "H-I-02",
    stageId: "interp",
    statement: "Influential features can be named with SHAP or LIME (Aim 2.4 as written).",
    verdict: "grant_gap",
    evidence: "Current stack: regularization paths, coefficient ablation, Stage 1 loadings, GSEA-style gene lists. No SHAP/LIME.",
    figures: [
      { id: "D4–D5", note: "current interpretability", path: `${NB}/stage2_old34_layer4_5_interpretability_suite.ipynb` },
    ],
    missing: "Decide whether Aim 2.4 is satisfied by loadings+coefficients or needs SHAP/LIME on a chosen probe.",
    nextAction: "If ChatGPT proposes SHAP, place it here — do not retrofit it into D8 as if it already exists.",
    claimGate: "Grant text ≠ current method.",
  },
  {
    id: "H-I-03",
    stageId: "interp",
    statement:
      "Stage 1 loadings can be used as sign-oriented descriptive gene lists, but L1-selected factor indices are not unique biological programs.",
    verdict: "supported_descriptive",
    evidence:
      "F3a: leading IFN-α PCA/FA factors match 1–1 with sign flips (PC2↔FA2 r=−0.90; PC8↔FA7 r=−0.68). F3b: sklearn FA scores are uncorrelated; loadings are not. F3d: L2 on stored FA scores recovers folds that L1 zeroes. Within-factor |loading| ranks remain readable after orientation (F2d: allograft factor 005 ranks TNF/LCK/ZAP70 at 3/5/8).",
    figures: [
      { id: "F2d/F3", note: "selected-factor genes + PCA/FA identifiability", path: `${NB}/stage2_old34_feedback_deep_dives_20260917.ipynb` },
    ],
    missing: "Do not treat this as SHAP/LIME. Still no census of programs beyond IFN-α plus three focus panels.",
    nextAction: "When discussing loadings, always state sign orientation and whether Stage 2 was L1, L2, or elastic-net.",
    claimGate: "Not “loadings cannot be looked at”; not “factor 8 is the IFN-α gene.”",
  },
  {
    id: "H-A-01",
    stageId: "aim2",
    statement:
      "Integrating genomic (ctDNA/SNV) and transcriptomic features with standard pathology improves MRD and recurrence prediction over histopathological staging.",
    verdict: "grant_gap",
    evidence:
      "Not tested in 20260525. Current labels are cell CN.label malignant vs normal. No recurrence outcome, no TNM comparator, no ctDNA/clinicopath table in this experiment dir.",
    figures: [],
    missing:
      "Patient-level endpoint definition; multi-modal join; split policy that respects patients; comparison to staging; operating-point utility.",
    nextAction:
      "Keep this as the parent hypothesis. Pipeline Stages 0–2 are mechanistic evidence that can later feed Aim 2, not a substitute for it.",
    claimGate: "Do not present LOPO cell-level lift as recurrence-risk improvement.",
  },
  {
    id: "H-A-02",
    stageId: "aim2",
    statement: "Relapse 4-class (MRD_cancer / MRD_normal / Relapse_cancer / Relapse_normal) and huCIRA priors extend the same Stage 0–2 idea.",
    verdict: "open",
    evidence: "Written in the value-added plan. Not started. Do not launch unless asked.",
    figures: [
      { id: "value-added", note: "huCIRA + relapse roadmap", path: `${PLANS}/stage0_geneset_value_added_workflow.md` },
    ],
    missing: "Panel manifest for huCIRA; patient-local relapse splits; matched HVG on that path.",
    nextAction: "Only after current pooled merge and control-grid reading, and only if requested.",
    claimGate: "Roadmap, not evidence.",
  },
];

const STAGE_FILTERS: { value: string; label: string }[] = [
  { value: "all", label: "All stages" },
  { value: "s0", label: "Stage 0" },
  { value: "s1", label: "Stage 1" },
  { value: "s2", label: "Stage 2" },
  { value: "interp", label: "Interpretability" },
  { value: "aim2", label: "Aim 2 translation" },
];

const VERDICT_FILTERS: { value: string; label: string }[] = [
  { value: "all", label: "All verdicts" },
  { value: "supported_descriptive", label: "Descriptive, supported" },
  { value: "open", label: "Open / not identified" },
  { value: "in_flight", label: "In flight" },
  { value: "not_supported", label: "Not supported as claimed" },
  { value: "grant_gap", label: "Grant gap" },
];

function statusTone(status: WorkStatus): "success" | "warning" | "info" | "danger" | "neutral" {
  if (status === "done") return "success";
  if (status === "in_flight") return "warning";
  if (status === "partial") return "info";
  if (status === "blocked") return "danger";
  return "neutral";
}

function statusLabel(status: WorkStatus): string {
  if (status === "done") return "Done";
  if (status === "in_flight") return "In flight";
  if (status === "partial") return "Partial";
  if (status === "blocked") return "Blocked";
  return "Not started";
}

function verdictLabel(v: Verdict): string {
  if (v === "supported_descriptive") return "Descriptive";
  if (v === "open") return "Open";
  if (v === "in_flight") return "In flight";
  if (v === "not_supported") return "Not supported";
  return "Grant gap";
}

function verdictTone(v: Verdict): "success" | "warning" | "info" | "danger" | "neutral" {
  if (v === "supported_descriptive") return "success";
  if (v === "in_flight") return "warning";
  if (v === "open") return "info";
  if (v === "not_supported") return "danger";
  return "neutral";
}

function stageById(id: StageId): StageRow {
  return STAGES.find((s) => s.id === id) ?? STAGES[0];
}

function FileLink({ file }: { file: FileRef }) {
  const dispatch = useCanvasAction();
  return (
    <Button variant="ghost" onClick={() => dispatch({ type: "openFile", path: file.path })}>
      {file.label}
    </Button>
  );
}

function WorkingHypothesis() {
  const theme = useHostTheme();
  return (
    <Stack gap={10}>
      <H2>Working hypothesis</H2>
      <div
        style={{
          borderLeft: `3px solid ${theme.accent.primary}`,
          paddingLeft: 14,
          paddingTop: 4,
          paddingBottom: 4,
        }}
      >
        <Text>{WORKING_HYPOTHESIS}</Text>
      </div>
      <Text tone="secondary" size="small">
        Source: attached Aim 2 statement · CRC clinical stages 0–2 are the disease setting, not the pipeline
        stages in the table below.
      </Text>
    </Stack>
  );
}

function Aim2Strip() {
  return (
    <Stack gap={10}>
      <H2>Aim 2 breakdown — develop and validate a multi-modal ML model for MRD prediction</H2>
      <Table
        headers={["Aim", "As written", "Where it lands in this pipeline", "Status"]}
        rows={AIM2.map((a) => [
          `${a.id} ${a.title}`,
          a.text,
          a.pipelineMap,
          statusLabel(a.status),
        ])}
        rowTone={AIM2.map((a) => statusTone(a.status))}
        striped
        stickyHeader
      />
    </Stack>
  );
}

function SnapshotStats() {
  const compiledOpen = HYPOTHESES.filter((h) => h.verdict === "open" || h.verdict === "in_flight" || h.verdict === "grant_gap").length;
  return (
    <Grid columns={5} gap={12}>
      <Stat value="5" label="pipeline / translation rows" />
      <Stat value={String(HYPOTHESES.length)} label="compiled testable claims" />
      <Stat value={String(compiledOpen)} label="open, in flight, or grant-gap" tone="warning" />
      <Stat value="7" label="supported held-out patients" />
      <Stat value={SNAPSHOT} label="compiled snapshot date" tone="info" />
    </Grid>
  );
}

function StageTable({
  selected,
  onSelect,
  stageFilter,
}: {
  selected: StageId;
  onSelect: (id: StageId) => void;
  stageFilter: string;
}) {
  const rows = STAGES.filter((s) => stageFilter === "all" || s.id === stageFilter);
  return (
    <Stack gap={10}>
      <Row gap={10} align="end" justify="space-between" wrap>
        <Stack gap={4}>
          <H2>Stage table</H2>
          <Text tone="secondary" size="small">
            One row per analysis stage. Click a stage name to open its hypothesis rows. Columns are the
            working contract for ChatGPT notes and later agents.
          </Text>
        </Stack>
        <Text size="small" tone="tertiary">
          Experiment {EXP_ID}
        </Text>
      </Row>
      <Table
        headers={[
          "Stage",
          "Testable hypotheses",
          "Evidence now",
          "Related figures",
          "Missing pieces",
          "Next implementation",
          "Aim 2",
          "Status",
        ]}
        rows={rows.map((s) => [
          <Button variant={s.id === selected ? "primary" : "ghost"} onClick={() => onSelect(s.id)}>
            {s.name}
          </Button>,
          <Stack gap={4}>
            <Text size="small">{s.hypothesesSummary}</Text>
            <Text size="small" tone="tertiary">
              {s.role}
            </Text>
          </Stack>,
          <Text size="small">{s.evidenceNow}</Text>,
          <Text size="small">{s.figuresSummary}</Text>,
          <Text size="small">{s.missingSummary}</Text>,
          <Text size="small">{s.nextAction}</Text>,
          <Text size="small">{s.aim2}</Text>,
          statusLabel(s.status),
        ])}
        rowTone={rows.map((s) => (s.id === selected ? "info" : statusTone(s.status)))}
        striped
        stickyHeader
      />
    </Stack>
  );
}

function HypothesisTable({
  stageId,
  verdictFilter,
  placed,
}: {
  stageId: StageId;
  verdictFilter: string;
  placed: PlacedHypothesis[];
}) {
  const dispatch = useCanvasAction();
  const rows = [...HYPOTHESES, ...placed].filter((h) => {
    if (h.stageId !== stageId) return false;
    if (verdictFilter !== "all" && h.verdict !== verdictFilter) return false;
    return true;
  });
  const stage = stageById(stageId);

  return (
    <Stack gap={12}>
      <Row gap={8} align="center" wrap>
        <H2>Hypotheses in {stage.name}</H2>
        <Spacer />
        {stage.files.map((f) => (
          <span key={f.path}>
            <FileLink file={f} />
          </span>
        ))}
      </Row>
      <Table
        headers={[
          "ID",
          "Hypothesis",
          "Verdict",
          "Evidence",
          "Figures",
          "Missing pieces",
          "Agent next",
          "Claim gate",
        ]}
        rows={rows.map((h) => [
          h.id,
          <Stack gap={3}>
            <Text size="small">{h.statement}</Text>
            {"sourceNote" in h && h.sourceNote ? (
              <Text size="small" tone="tertiary">
                Placed from inbox — merge into compiled DATA to make this durable
              </Text>
            ) : null}
          </Stack>,
          verdictLabel(h.verdict),
          <Text size="small">{h.evidence}</Text>,
          <Stack gap={4}>
            {h.figures.map((fig) => (
              <span key={`${h.id}-${fig.id}`}>
                <Button variant="ghost" onClick={() => dispatch({ type: "openFile", path: fig.path })}>
                  {fig.id}
                  {fig.note ? ` — ${fig.note}` : ""}
                </Button>
              </span>
            ))}
          </Stack>,
          <Text size="small">{h.missing}</Text>,
          <Text size="small">{h.nextAction}</Text>,
          <Text size="small">{h.claimGate}</Text>,
        ])}
        rowTone={rows.map((h) => verdictTone(h.verdict))}
        striped
        stickyHeader
      />
    </Stack>
  );
}

function Inbox({
  inbox,
  setInbox,
  placed,
  setPlaced,
}: {
  inbox: InboxItem[];
  setInbox: (next: InboxItem[] | ((prev: InboxItem[]) => InboxItem[])) => void;
  placed: PlacedHypothesis[];
  setPlaced: (next: PlacedHypothesis[] | ((prev: PlacedHypothesis[]) => PlacedHypothesis[])) => void;
}) {
  const dispatch = useCanvasAction();
  const [title, setTitle] = useCanvasState("inbox-draft-title", "");
  const [body, setBody] = useCanvasState("inbox-draft-body", "");
  const [stageId, setStageId] = useCanvasState("inbox-draft-stage", "unassigned");
  const [kind, setKind] = useCanvasState("inbox-draft-kind", "hypothesis");
  const [source, setSource] = useCanvasState("inbox-draft-source", "chatgpt");

  const openItems = inbox.filter((i) => i.status === "inbox");

  function place() {
    const trimmedTitle = title.trim();
    const trimmedBody = body.trim();
    if (!trimmedTitle || !trimmedBody) return;
    const id = `IN-${Date.now()}`;
    const item: InboxItem = {
      id,
      createdAt: new Date().toISOString(),
      stageId: stageId as InboxItem["stageId"],
      kind: kind as HypKind,
      title: trimmedTitle,
      body: trimmedBody,
      source: source as InboxItem["source"],
      status: "inbox",
    };
    setInbox((prev) => [item, ...prev]);
    setTitle("");
    setBody("");
  }

  function placeOnTable(item: InboxItem) {
    const target: StageId = item.stageId === "unassigned" ? "interp" : item.stageId;
    const hyp: PlacedHypothesis = {
      id: `P-${item.id}`,
      stageId: target,
      statement: item.title,
      verdict: item.kind === "missing" ? "open" : item.kind === "implementation" ? "open" : "open",
      evidence: item.body,
      figures: [],
      missing: item.kind === "missing" ? item.body : "Not yet merged into compiled DATA; treat as a candidate.",
      nextAction: `Implement or reject, then merge ${item.id} into HYPOTHESES in ${CANVAS}.`,
      claimGate: "Inbox-placed. Not a compiled finding until an agent merges it.",
      fromInboxId: item.id,
      sourceNote: item.source,
    };
    setPlaced((prev) => [hyp, ...prev.filter((p) => p.fromInboxId !== item.id)]);
    setInbox((prev) => prev.map((x) => (x.id === item.id ? { ...x, status: "placed" } : x)));
  }

  function mark(item: InboxItem, status: InboxStatus) {
    setInbox((prev) => prev.map((x) => (x.id === item.id ? { ...x, status } : x)));
  }

  const implementPrompt = [
    "Implement the next missing piece from the Stage 0–2 hypothesis tracker, then update the compiled table.",
    `Canvas file: ${CANVAS}`,
    `Selected-stage default: use the canvas Inbox and any status=placed rows first.`,
    "Rules: do not launch GPU Stage 0/1/2 from the agent shell; use screen. Do not overwrite root Set A biological scorecards. Bump SNAPSHOT. Merge placed inbox rows into the HYPOTHESES / STAGES arrays and set inboxStatus merged.",
    "If the request came from a ChatGPT discussion, treat it as a candidate until it has an id in HYPOTHESES and a real artifact path.",
  ].join(" ");

  return (
    <Stack gap={14}>
      <H2>ChatGPT inbox — place a result on the table</H2>
      <Text tone="secondary">
        Paste a conclusion, a new hypothesis, or an implementation sketch from the web ChatGPT thread.
        Placing it adds a candidate row under the chosen stage. A later Cursor agent should implement
        the work and merge that row into the compiled DATA in this canvas file.
      </Text>
      <Grid columns="1.1fr 1fr" gap={16} align="start">
        <Stack gap={8}>
          <TextInput value={title} onChange={setTitle} placeholder="Short title / hypothesis sentence" />
          <TextArea
            value={body}
            onChange={setBody}
            rows={6}
            placeholder="Paste the ChatGPT result: claim, proposed figure, missing control, or implementation notes."
          />
          <Row gap={8} wrap>
            <Select
              value={stageId}
              onChange={setStageId}
              options={[
                { value: "unassigned", label: "Stage: unassigned" },
                ...STAGES.map((s) => ({ value: s.id, label: s.name })),
              ]}
            />
            <Select
              value={kind}
              onChange={setKind}
              options={[
                { value: "hypothesis", label: "New hypothesis" },
                { value: "implementation", label: "Implementation to build" },
                { value: "figure", label: "Figure / evidence" },
                { value: "missing", label: "Missing piece" },
                { value: "note", label: "Note only" },
              ]}
            />
            <Select
              value={source}
              onChange={setSource}
              options={[
                { value: "chatgpt", label: "From ChatGPT" },
                { value: "user", label: "From me" },
                { value: "agent", label: "From a Cursor agent" },
              ]}
            />
          </Row>
          <Row gap={8}>
            <Button variant="primary" onClick={place} disabled={!title.trim() || !body.trim()}>
              Save to inbox
            </Button>
            <Button
              variant="secondary"
              onClick={() =>
                dispatch({
                  type: "newComposerChat",
                  userPrompt: implementPrompt,
                })
              }
            >
              Ask an agent to implement + update this table
            </Button>
          </Row>
        </Stack>
        <Stack gap={8}>
          <H3>Agent contract</H3>
          <Text size="small">
            Compiled arrays <Code>STAGES</Code>, <Code>HYPOTHESES</Code>, <Code>AIM2</Code> in this
            file are the durable record. Inbox and placed rows persist in canvas state only.
          </Text>
          <Text size="small">
            After implementation: edit the matching compiled row (verdict, evidence, figures,
            missing, nextAction), bump <Code>SNAPSHOT</Code>, and mark the inbox item merged.
          </Text>
          <Text size="small">
            Pointer for agents in-repo: <Code>scripts/mrd_stage0_2/plans/stage0_2_hypothesis_tracker.md</Code>
          </Text>
          <Callout tone="warning" title="Do not collapse the two “stage 0–2” meanings">
            Grant text = CRC clinical stages. Table rows = analysis Stage 0/1/2 plus interpretability
            and Aim 2 translation. A recurrence-model idea belongs on the Aim 2 row.
          </Callout>
        </Stack>
      </Grid>
      {openItems.length > 0 ? (
        <Table
          headers={["Inbox", "Stage", "Kind", "Source", "Actions"]}
          rows={openItems.map((item) => [
            <Stack gap={3}>
              <Text size="small" weight="semibold">
                {item.title}
              </Text>
              <Text size="small" tone="secondary">
                {item.body}
              </Text>
            </Stack>,
            item.stageId,
            item.kind,
            item.source,
            <Row gap={6} wrap>
              <Button variant="primary" onClick={() => placeOnTable(item)}>
                Place on table
              </Button>
              <Button variant="ghost" onClick={() => mark(item, "dropped")}>
                Drop
              </Button>
            </Row>,
          ])}
          striped
        />
      ) : null}
      {placed.length > 0 ? (
        <Callout tone="info" title={`${placed.length} candidate row(s) sitting on the table, not yet compiled`}>
          {placed.map((p) => `${p.id} → ${p.stageId}: ${p.statement}`).join(" · ")}
        </Callout>
      ) : null}
    </Stack>
  );
}

function CampaignNotes() {
  return (
    <Stack gap={10}>
      <H2>Campaign facts the table is allowed to assume</H2>
      <Table
        headers={["Campaign", "What exists", "What the table must not claim"]}
        rows={[
          [
            "Set A old34 root",
            "Stage 0/1 + quick L2 (47). Full Stage 2 for 41 biological panels. Notebooks read this.",
            "That full_34 / HVG anchors are in the root full Stage 2 grid — they are not.",
          ],
          [
            "old34_controls_fullgrid_v1",
            "Full Stage 2 DR for full_34, core_only, HVG 500/1k/3k/10k (2026-09-04). D1 reads it as extra y-axis DR cells.",
            "That it was merged into root CSVs, or that HVG 200 exists.",
          ],
          [
            "old34_controls_direct_gene_v1",
            "No-DR LOPO for the same six control panels (2026-09-11). D1 now fills the direct_gene column (47/47; 783 atlas rows).",
            "That it includes patient-specific Stage 2, D2/D3 pairing, or matched-size random-gene nulls.",
          ],
          [
            "Set B expanded axes",
            "June compute: 632 panels Stage 0/1; 294 Stage 2. expanded_full absent from Stage 2.",
            "That Set B has a D1–D13-quality reading.",
          ],
          [
            "pooled_stage0_unions_v1",
            "Stage 0/1 + quick L2 (4 bags). Stage 2 unmerged as of 2026-09-08.",
            "Any pooled Stage 2 lift or sharedness number.",
          ],
          [
            "Direct-gene + donor v2",
            "D9 41/41; donor v2 merge-only complete. Corrected FA007 / PCA008 stability.",
            "Old 25% ≈0.99 stability sentence; “DR always wins.”",
          ],
        ]}
        striped
      />
    </Stack>
  );
}

export default function Stage02HypothesisTracker() {
  const [view, setView] = useCanvasState("s02-tracker-view", "table");
  const [selected, setSelected] = useCanvasState<StageId>("s02-tracker-stage", "s2");
  const [stageFilter, setStageFilter] = useCanvasState("s02-tracker-stage-filter", "all");
  const [verdictFilter, setVerdictFilter] = useCanvasState("s02-tracker-verdict-filter", "all");
  const [inbox, setInbox] = useCanvasState<InboxItem[]>("s02-tracker-inbox", []);
  const [placed, setPlaced] = useCanvasState<PlacedHypothesis[]>("s02-tracker-placed", []);

  const openInbox = inbox.filter((i) => i.status === "inbox").length;
  const stageHypCount = useMemo(
    () =>
      [...HYPOTHESES, ...placed].filter((h) => h.stageId === selected).length,
    [placed, selected],
  );

  return (
    <Stack gap={22}>
      <Stack gap={6}>
        <H1>Stage 0–2 hypothesis tracker</H1>
        <Text tone="secondary">
          Living map from the Aim 2 working hypothesis to the knowledge-prior malignancy pipeline.
          Primary object is the stage table. Inbox rows are how a ChatGPT discussion becomes a
          table update or an implementation request.
        </Text>
      </Stack>

      <SnapshotStats />

      <Row gap={8} wrap>
        <Pill active={view === "table"} onClick={() => setView("table")}>
          Stage table
        </Pill>
        <Pill active={view === "hypotheses"} onClick={() => setView("hypotheses")}>
          {`Hypotheses (${stageHypCount})`}
        </Pill>
        <Pill active={view === "inbox"} onClick={() => setView("inbox")}>
          {openInbox > 0 ? `Inbox (${openInbox})` : "Inbox"}
        </Pill>
        <Pill active={view === "aim2"} onClick={() => setView("aim2")}>
          Aim 2 + working hypothesis
        </Pill>
        <Pill active={view === "campaigns"} onClick={() => setView("campaigns")}>
          Campaign facts
        </Pill>
      </Row>

      <Row gap={8} wrap align="center">
        {STAGE_FILTERS.map((f) => (
          <span key={f.value}>
            <Pill
              size="sm"
              active={f.value === "all" ? stageFilter === "all" : selected === f.value || stageFilter === f.value}
              onClick={() => {
                setStageFilter(f.value);
                if (f.value !== "all") setSelected(f.value as StageId);
              }}
            >
              {f.label}
            </Pill>
          </span>
        ))}
        <Spacer />
        {view === "hypotheses"
          ? VERDICT_FILTERS.map((f) => (
              <span key={f.value}>
                <Pill size="sm" active={verdictFilter === f.value} onClick={() => setVerdictFilter(f.value)}>
                  {f.label}
                </Pill>
              </span>
            ))
          : null}
      </Row>

      <Divider />

      {view === "table" && (
        <Stack gap={18}>
          <StageTable selected={selected} onSelect={setSelected} stageFilter={stageFilter} />
          <Callout tone="info" title={`Selected: ${stageById(selected).name}`}>
            Switch to Hypotheses to see the testable claims, figure buttons, and claim gates for this
            row. {stageById(selected).nextAction}
          </Callout>
        </Stack>
      )}
      {view === "hypotheses" && (
        <HypothesisTable stageId={selected} verdictFilter={verdictFilter} placed={placed} />
      )}
      {view === "inbox" && (
        <Inbox inbox={inbox} setInbox={setInbox} placed={placed} setPlaced={setPlaced} />
      )}
      {view === "aim2" && (
        <Stack gap={18}>
          <WorkingHypothesis />
          <Aim2Strip />
        </Stack>
      )}
      {view === "campaigns" && <CampaignNotes />}

      <Divider />
      <Text tone="tertiary" size="small">
        Compiled snapshot {SNAPSHOT} · experiment {EXP_ID} · source notebooks and plans under
        scripts/mrd_stage0_2 · placed inbox overlays are local canvas state until merged
      </Text>
    </Stack>
  );
}
