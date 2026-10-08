---
title: Literature review B — single-cell ML frameworks for the MRD Stage 0-2 malignant-vs-normal study
date: 2026-10-08
produced_by: agent web review (claude-code session, 2026-10-08); citations checked by the agent at write time; readers should spot-check
scope: malignant-vs-normal labelling and the cohort paper; cross-patient transfer, evaluation and imbalance; supervised and semi-supervised factor learning; gene-set scoring baselines and matched nulls; learning curves and sample-size planning
routed_tasks: [W1.4, W1.8, W3.1, W4.1, W3.3, W2.1, W3.4, W4.2, W4.3, W6.2, W6.1, W6.3, W1.3, W3.7, W5.6, W5.2, W8.2, W0.12]
---

> **Provenance header (added when this review was homed; not part of the review).**
> - Date: 2026-10-08. Homed by W9.1; the matching devlog entry is `2026-10-08_literature-reviews`. Ask: A-20261008-28.
> - How it was produced: an agent-run web literature review in a claude-code session on 2026-10-08, plus read-only checks against this repo and the canonical experiment. At write time the agent checked each citation's title, authors, venue, year and DOI or URL, mostly from abstracts and records (see "How citations were checked" below). Readers should spot-check a citation before relying on it.
> - Erratum (not edited in the body): the GSVA link in §4.1 lacks the `10.` DOI prefix. The working DOI is https://doi.org/10.1186/1471-2105-14-7 (resolved on 2026-10-08 when this file was homed).
> - Label remap: this review's "D16" genotype/label-assay audit is the repo's **D25** (W1.4); the repo's D16 is the factor × patient matrix (W1.1). The review's "D15b" is **D15 v2** (W6.2), its "D15c" is **D15b** (W6.1), and its "D15d" is task **W6.3** (no ID reserved yet).
> - The "tracker snapshot 2026-10-02b" column refers to the frozen Cursor canvas, now `research/legacy/stage0-2-hypothesis-tracker_2026-10-02b.canvas.tsx`. Current verdicts come from devlog entries; see `research/STATUS.md`.
> - Paths in the body: `preprocessing/panels/...` is relative to the experiment dir (EXP); `scripts/...` is repo-relative. No absolute paths were present, so the body is verbatim.
> - The per-patient label provenance in §0 is the reviewer's reading of the cohort preprint. Confirm it with the authors (W8.1 email) before W1.4 relies on it. Gene-list facts in §0 were read by the reviewer from stored Stage 0 panel files; W1.4 and W5.2 reproduce them before they are cited.
> - Every recommended action is routed to a W#.n task in `README.md` in this directory.

# Literature review B: single-cell ML frameworks for the MRD Stage 0–2 malignant-vs-normal study

Prepared 2026-10-08 for `sc_classification` (branch `sc_classification_gpu`). Canonical experiment: `20260525_060508_stage0_mrd_old34_broad_screen_82db5093`.

## How to read this

**How citations were checked.** I checked every citation by web search on 2026-10-08: title, authors, venue, year, and DOI or stable URL. For most papers I read the abstract, the PubMed Central or publisher record, or the software documentation, not the full text. Where a claim rests on a secondary source or on documentation, I say so. I read one paper in full (Chen et al. 2025), because it describes this cohort.

**Repo anchors used in every section**

| Tag | What it means in this repo | State in tracker snapshot 2026-10-02b |
|---|---|---|
| **Q1** (H-S2-01) | One robust malignant-vs-normal model works across patients (LOPO lift, P20 lift) | Open. Lift still ranks with prevalence (D6). No matched-gene-set null and no patient-level CI. |
| **Q2** (H-S2-02) | Patient-specific models show biology, not just differences in malignant-cell count | Open. D3 gaps, D13 DE, D14 overlap exist, but biology and malignant-cell count are not separated. |
| **Q3** (H-S2-03) | A prior program splits into distinct regulatory axes | Supported descriptively for IFN-α only. The PCA008 vs FA007 gap comes from Stage 2 L1 at a fixed C=1e-4: FA7 alone gives +18.06 pp in P02, and L2 on the FA scores recovers P02 (16.14 pp) and P07 (5.07 pp). |
| **D14** (H-S1-04) | Per-patient overlap of stored transductive factor scores | Biological bases can be aggregated. P03 malignant cells are nearly disjoint on one factor each of HVG 3k (f27), HVG 10k (f21) and full_34 (f30). |
| **D15** (H-S2-06) | Learning curve over training-cell count, on a frozen basis with fixed C | Apoptosis is nearly saturated by about 1,000 cells. IFN-α stays flat through 25k cells because L1 at C=1e-4 gives an all-zero model. Still missing: C re-tuned per subsample, an inductive refit, and a curve that drops patients. |
| **Inductive S1** | Refit Stage 1 without the held-out patient | Not run. First step is IFN-α PCA40/FA40 × 7 patients. |
| **Balanced S1** | Fit Stage 1 so that large patients do not dominate | Planned. P09 + P01 = 28,986 of 60,118 cells (48.2%). |
| **New patients** | Cohort from a collaborator | Not yet received. This is the only true external test. |
| **End-to-end** | Ankit's suggestion to learn the factors with the label | Not implemented. |

Cohort numbers from guide §6: 1,569 malignant cells, 58,499 donor-healthy, 50 recipient-healthy. P03 contributes 758 of the 1,569 malignant cells (48.3%). P07 has 22 malignant cells, P06 44 and P01 45.

---

## 0. The cohort's own paper changes what the classifier is learning

[Chen YH, Bi T, Xu M, …, Patel A, Bachireddy P. 2025, *bioRxiv*, "Molecular and cellular dynamics of measurable residual disease progression in myelodysplastic syndromes"](https://doi.org/10.1101/2025.10.17.682671) describes a post-transplant MDS cohort of 13 patients: 10 relapsed, 3 stayed in remission, 52 samples. The assay, CARAMEL-seq, combines scATAC, scRNA, mtDNA variants and surface epitopes. It appears to be the source of this repo's cohort: the patient IDs match, and relapse-free P10–P12 have 0 malignant cells. I read the Methods.

**How the labels were made**
1. **Donor vs recipient.** Souporcell was run per patient, and only cells with singlet posterior = 1 were kept. Estimated assignment error is 0.06%.
2. **Malignant vs normal recipient.** The evidence differs by patient:
   - myeloid-specific clonal mtDNA variants (P01, P09);
   - Numbat CNVs called from RNA against a donor-cell reference (7/13 patients had clinical CNVs; 2 had chrY loss, e.g. P05);
   - aberrant CD5/CD56 surface protein (P02, P13);
   - a donor-dominant cluster used as the normal reference (P13).

The paper also reports biology that bears on the Stage 0 priors:
- uMRD (MRD below flow-cytometry detection) was found in 9/10 relapsing patients and 0/3 non-relapsing patients.
- HSC/MPP-like uMRD cells vs donor HSC/MPP showed GSEA enrichment of TNF-α/NF-κB, hypoxia, IFN-γ and IFN-α/β, p53 and inflammatory response.
- At relapse, CytoSig activity rose for IFN-γ, IL-1β, TNF-α and TGF-β, but not for type I IFN.
- Most patients already had lower HLA class II surface protein than the donor baseline before transplant.

**Consequence 1: the negative class is 99.9% donor.** 58,499 of 58,549 normal cells are donor-derived. Malignant-vs-normal is therefore almost the same task as recipient-malignant vs donor. Any gene whose measured expression depends on germline genotype can separate the classes with no malignant biology at all. Examples with literature support:
- Per-cell HLA counts from reference alignment are skewed by haplotype ([Darby et al. 2020, *Bioinformatics*, scHLAcount](https://doi.org/10.1093/bioinformatics/btaa264)).
- HLA-DRB5 sits on DR51-group (DR15/DR16) haplotypes, so its expression is a presence/absence trait ([PubMed 16111772, 2005](https://pubmed.ncbi.nlm.nih.gov/16111772/)).
- ERAP2 has a common haplotype whose transcript undergoes nonsense-mediated decay ([Andrés et al. 2010, *PLoS Genetics*](https://doi.org/10.1371/journal.pgen.1001157)).
- chrY genes and XIST separate donor from recipient in sex-mismatched transplants, and they also mark chrY loss.

**Consequence 2: label evidence differs by patient.** A patient-specific lift can partly re-encode the assay that defined the labels rather than malignant biology.

**What the stored Stage 0 gene lists contain** (`preprocessing/panels/stage0_panel_genes/*.json`, read-only):
- **HVG 3k:** RPS4Y1, DDX3Y, UTY, USP9Y, ZFY, PRKY and HLA-DRB5.
- **HVG 10k:** all of the above plus XIST, KDM5D, EIF1AY, ERAP2 and CD7.
- **full_34:** HLA-DRB5, ERAP2, CD7 and 18 HLA genes.
- **Hallmark allograft rejection** (147 covered genes): HLA-A, HLA-DRA, HLA-DQA1, CD7, LCK, ZAP70 and CD3E. Aberrant CD5/CD7 expression on HSPCs is what defined MRD in P02 and P13.
- **IFN-α vs IFN-γ:** 69 of the 92 covered Hallmark IFN-α genes (75%) are also in the covered Hallmark IFN-γ panel (Jaccard 0.347).
- **Why these 34 sets:** the old34 gene-set README (`scripts/knowledge_driven_embedding/older_geneset/README.md`) records how the sets were downloaded but not why they were chosen. If the list was informed by enrichment results on this same cohort, Stage 0 is label-informed (see §3 on selection bias).

**Untested hypothesis about D14's P03 factors.** The P03-private factors appear only in the panels that carry genotype-linked genes (HVG 3k, HVG 10k, full_34). If P03 donor cells overlap the rest of the cohort on the same factor while P03 malignant cells do not, that is the pattern a recipient-genotype factor would produce. Reading the loadings for HLA-DRB5, ERAP2, HLA-DQ*, chrY genes and XIST would test this in minutes.

---

## 1. Malignant vs normal classification in scRNA-seq and MRD

### 1.1 CNV inference from expression
- **inferCNV** ([Trinity CTAT project; Tickle et al. 2019, software](https://github.com/broadinstitute/inferCNV)) smooths expression along the genome relative to reference normal cells. The idea comes from [Patel et al. 2014, *Science*](https://doi.org/10.1126/science.1254257).
- **CopyKAT** ([Gao R et al. 2021, *Nature Biotechnology*](https://doi.org/10.1038/s41587-020-00795-2)) uses Bayesian segmentation at about 5 Mb. It analysed 46,501 cells from 21 tumours and reports about 98% cancer-vs-normal accuracy. It is aimed at solid tumours.
- **SCEVAN** ([De Falco et al. 2023, *Nature Communications*](https://doi.org/10.1038/s41467-023-36790-9)) uses variational multichannel segmentation and separates malignant from non-malignant cells automatically (106 samples, 93,322 cells).
- **Numbat** ([Gao T et al. 2023, *Nature Biotechnology*](https://doi.org/10.1038/s41587-022-01468-y)) adds population-phased haplotypes and allele imbalance and needs no matched DNA (22 tumours, 59,878 cells).
- **Benchmark** ([Song et al. 2025, *Briefings in Bioinformatics*](https://doi.org/10.1093/bib/bbaf076)) used paired scDNA as truth. Numbat performed best overall, CopyKAT is recommended when only counts are available, and SCEVAN was best at breakpoints.

**Limit for MDS.** Chen et al. note that only about 50% of MDS cases carry somatic CNVs, so expression-only callers cannot see normal-karyotype clones. That is why this cohort's labels combine CNV, mtDNA and immunophenotype evidence.

### 1.2 Reference-based and ML annotators
- **ikarus** ([Dohmen et al. 2022, *Genome Biology*](https://doi.org/10.1186/s13059-022-02683-1)): a tumour gene signature learned from several annotated datasets, then logistic scoring, then label propagation over a cell network.
- **scATOMIC** ([Nofech-Mozes et al. 2023, *Nature Communications*](https://doi.org/10.1038/s41467-023-37353-8)): hierarchical classifiers trained on more than 300k cells from 19 cancers and checked on 225 biopsies.
- **Cancer-Finder** ([Zhong et al. 2024, *Nature Communications*](https://doi.org/10.1038/s41467-024-46413-6)): a domain-generalization model with about 95% average accuracy across datasets.
- **AML** ([van Galen et al. 2019, *Cell*](https://doi.org/10.1016/j.cell.2019.01.031)): Seq-Well with targeted mutation genotyping on 3,799 of 38,410 cells, plus a machine-learning classifier (a random forest according to secondary sources) that labels malignant cells along normal bone-marrow cell types.

None of these was built for post-transplant MDS MRD, where the comparison population is donor haematopoiesis.

### 1.3 Genotype-based labels and single-cell MRD assays
- **Genotype demultiplexing:** souporcell ([Heaton et al. 2020, *Nature Methods*](https://doi.org/10.1038/s41592-020-0820-1)) and vireo ([Huang, McCarthy & Stegle 2019, *Genome Biology*](https://doi.org/10.1186/s13059-019-1865-2)).
- **Expressed mutations in AML scRNA** ([Petti et al. 2019, *Nature Communications*](https://doi.org/10.1038/s41467-019-11591-1)): separates normal-karyotype AML cells from normal cells. Detection depends on whether the mutated gene is expressed.
- **scDNA plus immunophenotype MRD** ([Robinson et al. 2023, *Science Advances*](https://doi.org/10.1126/sciadv.adg0488)): sensitivity of about 0.01%.

### 1.4 Post-transplant AML/MDS biology that bears on the Stage 0 priors
- [Sariipek et al. 2025, *bioRxiv*](https://doi.org/10.1101/2025.11.13.688350): in 33 AML/MDS patients, persistent recipient HSPCs predicted later relapse and carried CNVs.
- [Christopher et al. 2018, *NEJM*](https://doi.org/10.1056/NEJMoa1808777): AML that relapses after transplant down-regulates MHC class II (HLA-DPA1/DPB1/DQB1/DRB1), and IFN-γ restores it in vitro.
- [Toffalori et al. 2019, *Nature Medicine*](https://doi.org/10.1038/s41591-019-0400-z): a relapse-specific immune signature (costimulation, antigen presentation) and HLA class II loss.
- [Ganan-Gomez et al. 2022, *Nature Medicine*](https://doi.org/10.1038/s41591-022-01696-4): two MDS stem-cell architectures that depend on different survival signals (BCL-2 vs NF-κB). This is a biological reason for patient heterogeneity.

### Implication for this repo
- **Q1 framing.** Present Stage 2 as label transfer inside prior gene spaces, not as a replacement for Numbat or souporcell, which generate the labels. The natural external comparator is an ikarus-style signature-plus-logistic model or a van Galen-style normal-reference classifier. The D9 direct-gene baseline already plays that role.
- **Q1/Q2: new genotype-shortcut audit ("D16").**
  - (a) Drop chrY genes, XIST, HLA genes, ERAP2 and HLA-DRB5 from HVG 3k, HVG 10k and full_34, then re-run D14 for the P03 factors and the matching D1 rows.
  - (b) Score the 50 recipient-healthy cells (P02 17, P04 14, P05 10) with the LOPO models. If they rank like malignant cells, the probe is partly reading genotype.
  - (c) Record donor/recipient sex match for each patient.
- **Q2: label provenance.** Add a per-patient label-evidence column (mtDNA / CNV / chrY / CD5-CD56 / cluster) to D3, D7B and D13. Test whether P02's allograft and IL2/STAT5 lift survives removing CD5, CD7, LCK, ZAP70 and CD3 genes. The feedback notebook's Observation 2 already lists LCK and ZAP70 among the top loadings.
- **Q3: naming the IFN program.** The two IFN panels share 75% of IFN-α's genes, and Chen et al. find IFN-γ rather than type I IFN rising at relapse. Name PC8's program from a contrast between IFN-γ-specific genes (HLA class II, CIITA) and type-I-specific genes, not from the Hallmark title.
- **Q1 biology.** F7 found that missed malignant cells (false negatives) in P02 have higher HLA-DR, CD45, CD38 and CD58 ADT than detected ones. Together with Christopher 2018, Toffalori 2019 and Chen 2025, this fits a probe that detects the HLA-II-low immune-escape state and misses HLA-II-high malignant cells. The 170-ADT panel can test this directly.
- **New patients.** Require the same label pipeline (souporcell singlet = 1, Numbat with a donor reference, mtDNA where available) or at least a label-provenance table. Otherwise label shift is confounded with model transfer.

---

## 2. Cross-patient transfer and patient heterogeneity

### 2.1 Shared vs patient-specific programs
- **MOFA+** ([Argelaguet et al. 2020, *Genome Biology*](https://doi.org/10.1186/s13059-020-02015-1)): a multi-group factor model whose factors can be active in some groups only, with variance explained broken down by factor × group. The muon wrapper defaults `center_groups=True` ([muon documentation](https://muon.readthedocs.io/en/latest/api/generated/muon.tl.mofa.html)), so in that wrapper per-group means are removed.
- **scITD** ([Mitchel et al. 2024, *Nature Biotechnology*](https://doi.org/10.1038/s41587-024-02411-z); DOI confirmed via an aggregator; [preprint](https://doi.org/10.1101/2022.02.16.480703)): Tucker decomposition of donor × gene × cell-type patterns.
- **GEDI** ([Madrigal et al. 2024, *Nature Communications*](https://doi.org/10.1038/s41467-024-50963-0)): shared reference axes plus sample-specific transformations, with optional gene-set priors on the axes.
- **cNMF** ([Kotliar et al. 2019, *eLife*](https://doi.org/10.7554/eLife.43803)): separates cell-identity programs from activity programs.
- **Pan-cancer meta-programs** ([Gavish et al. 2023, *Nature*](https://doi.org/10.1038/s41586-023-06130-4)): NMF run on each tumour separately (1,163 samples, 77 studies), then clustered into 41 meta-programs. Many of these vary before transformation, reflecting cell of origin.
- **Recurrent states** ([Barkley et al. 2022, *Nature Genetics*](https://doi.org/10.1038/s41588-022-01141-9)): stress, IFN and EMT states recur across 15 cancer types. The IFN state is linked to T cells and macrophages and depended on lymphocytes in mice.

### 2.2 Inductive mapping and integration
- **Reference mapping:** Symphony ([Kang et al. 2021, *Nature Communications*](https://doi.org/10.1038/s41467-021-25957-x)) maps query cells into a fixed reference embedding without re-integrating the reference. scArches ([Lotfollahi et al. 2022, *Nature Biotechnology*](https://doi.org/10.1038/s41587-021-01001-7)) does the same by transfer learning.
- **Integration:** Harmony ([Korsunsky et al. 2019, *Nature Methods*](https://doi.org/10.1038/s41592-019-0619-0)). The benchmark by [Luecken et al. 2022, *Nature Methods*](https://doi.org/10.1038/s41592-021-01336-8) found that scaling pushes methods toward batch removal at the cost of biological signal, while HVG selection helps.

### 2.3 Evaluation: subject-wise splits, leakage, model selection, uncertainty
- **Split by subject:** [Saeb et al. 2017, *GigaScience*](https://doi.org/10.1093/gigascience/gix019) show that splitting by record instead of by subject inflates accuracy.
- **Leakage:** [Kapoor & Narayanan 2023, *Patterns*](https://doi.org/10.1016/j.patter.2023.100804) give a taxonomy of eight leakage types across 329 affected papers.
- **Model selection:** [Gulrajani & Lopez-Paz 2021, *ICLR*](https://arxiv.org/abs/2007.01434) argue that a domain-generalization method without a model-selection rule is incomplete, describe leave-one-domain-out selection, and find well-tuned plain empirical risk minimization (ERM) hard to beat.
- **CV uncertainty:** [Bates, Hastie & Tibshirani 2024, *JASA*](https://doi.org/10.1080/01621459.2023.2197686) show that CV estimates the average error over training sets, that naive CV confidence intervals under-cover, and propose nested-CV variance as a fix.
- **Clustered bootstrap:** [Field & Welsh 2007, *JRSS-B*](https://doi.org/10.1111/j.1467-9868.2007.00593.x) show that a cluster bootstrap gives consistent variance estimates under a random-effects model.
- **AUPRC intervals:** [Boyd, Eng & Page 2013, *ECML PKDD*](https://doi.org/10.1007/978-3-642-40994-3_29) compare AUPRC estimators and confidence intervals and find that some common ones are invalid.
- **AUPRC and prevalence:** [McDermott et al. 2024, *NeurIPS*](https://arxiv.org/abs/2401.06091) prove that AUPRC favours fixing errors in subpopulations with more positives.

### 2.4 Imbalance across patients
- **Pseudoreplication:** cells within a person are not independent observations. Ignoring this inflates type I error ([Zimmerman, Espeland & Langefeld 2021, *Nature Communications*](https://doi.org/10.1038/s41467-021-21038-1)); pseudobulk methods do well ([Squair et al. 2021, *Nature Communications*](https://doi.org/10.1038/s41467-021-25960-2)).
- **Balanced subsampling:** geometric sketching ([Hie et al. 2019, *Cell Systems*](https://doi.org/10.1016/j.cels.2019.05.003)) and Seurat v5 leverage-score sketching ([Hao et al. 2024, *Nature Biotechnology*](https://doi.org/10.1038/s41587-023-01767-y)).
- **Worst-group objective:** group DRO improves worst-group accuracy only when paired with strong regularization ([Sagawa et al. 2020, *ICLR*](https://arxiv.org/abs/1911.08731)).

### Implication for this repo
- **D14 → a model-based version (Q2).** Fit MOFA+ with groups = patients on IFN-α, allograft and HVG 3k. A factor active in only one group is a model-based "patient-private" call that complements D14's histogram overlap. Run it both with and without per-group centering, because centering erases location shifts such as allograft's P02 normal-cell shift.
- **Q2/Q3 bottom-up route.** Following Gavish, run cNMF within each patient for a panel, then cluster the programs across patients into recurrent vs private programs. This answers Q2/Q3 without assuming a shared basis. Fits on malignant cells alone are underpowered for P07, P06 and P01 (22, 44, 45 cells), so fit on all cells.
- **Inductive Stage 1.** Use the Symphony/scArches pattern: compute gene standardization and PCA/FA on training patients only, then project the held-out patient.
  - Expect small changes for biological panels (D14 overlaps 0.54–0.89) and larger ones for HVG 3k, HVG 10k and full_34, where the P03-private factor sits.
  - Report the change in lift per patient and factor alignment by Hungarian matching on scores, as in F3.
  - Under the Kapoor taxonomy, transductive Stage 1 is unsupervised "preprocessing on train+test" leakage, not label leakage.
- **Patient-balanced Stage 1.** Compare three options:
  - (i) the average of per-patient covariances;
  - (ii) a per-patient cap with geometric sketching, which ignores labels and keeps rare states;
  - (iii) per-patient centering on donor cells only.

  Fit on training patients only, so the result can be combined with the inductive test. Do not remove patient effects with Harmony first: prevalence ranges from 0.07% to 33% across patients, so correcting for patient can remove malignant signal (Luecken's batch-vs-biology trade-off).
- **Q1 metrics.** McDermott et al. show that part of D6's link between lift and prevalence is a property of AUPRC itself. Put AUROC and AUPRC/π (D1f) next to lift in every per-patient table, and pick "best" configurations by patient-equal criteria, not pooled AUPRC (pooled AUPRC favours P03 at 33% prevalence).
- **Selection.** Replace the in-sample best-median choice of DR method and K with an inner leave-one-patient-out selection (Gulrajani; Varma & Simon). This addresses the winner's-curse caveat in the tracker.
- **Uncertainty.**
  - Per patient: use Boyd-type logit intervals or a cell bootstrap stratified by label. P07's 22 positives will give wide intervals.
  - For the cohort: use a patient-cluster bootstrap (Field & Welsh), but with only 7 clusters also report exact patient-level sign or permutation tests.
  - Do not quote naive CV standard errors (Bates et al.).
- **Patient-balanced Stage 2.** Weight cells so each training patient contributes equally; otherwise P03 supplies 48% of positives. Treat P20 lift as the worst-group metric (Sagawa). Expect to need stronger regularization, and check how this interacts with the L1 all-zero collapse (H-S2-04).

---

## 3. Supervised and semi-supervised factor learning (Ankit's end-to-end suggestion)

### 3.1 Methods
- **Supervised PCA** ([Bair, Hastie, Paul & Tibshirani 2006, *JASA*](https://doi.org/10.1198/016214505000000628)): screen predictors by univariate association, then run PCA on the retained subset. Consistency results hold when there are many more predictors than samples.
- **HSIC-based supervised PCA** ([Barshan et al. 2011, *Pattern Recognition*](https://doi.org/10.1016/j.patcog.2010.12.015)): components chosen to maximize dependence on the response. Closed form; can be kernelized.
- **sPLS-DA** ([Lê Cao, Boitard & Besse 2011, *BMC Bioinformatics*](https://doi.org/10.1186/1471-2105-12-253)): sparse latent components for classification (mixOmics).
- **Spectra** ([Kunes et al. 2024, *Nature Biotechnology*](https://doi.org/10.1038/s41587-023-01940-3)): factorization that uses gene sets as a gene-gene graph prior plus cell-type labels. A single parameter sets how strongly the prior is used, and the method also finds new programs.
- **f-scLVM / slalom** ([Buettner et al. 2017, *Genome Biology*](https://doi.org/10.1186/s13059-017-1334-8)): factor analysis with factors annotated by gene sets. It refines set membership and adds unannotated sparse and dense factors.
- **PLIER** ([Mao et al. 2019, *Nature Methods*](https://doi.org/10.1038/s41592-019-0456-1)): loadings are pulled toward prior gene sets, and each latent variable gets a gene-set label.
- **expiMap** ([Lotfollahi et al. 2023, *Nature Cell Biology*](https://doi.org/10.1038/s41556-022-01072-x)): a gene-program-constrained VAE for reference mapping that also learns new programs.
- **scANVI** ([Xu et al. 2021, *Molecular Systems Biology*](https://doi.org/10.15252/msb.20209620)): semi-supervised scVI.
- **Supervised autoencoders** ([Le, Patterson & White 2018, *NeurIPS*](https://proceedings.neurips.cc/paper/2018/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html)): a reconstruction loss added to the prediction loss gives a generalization bound in the linear case. In their experiments it never hurt and sometimes helped.
- **Supervised AE as a factor model** ([Talbot, Dunson, Dzirasa & Carlson 2023, *JRSS-C*](https://doi.org/10.1093/jrsssc/qlad035)): fitting joint factor models this way is more robust to misspecification than maximum likelihood, and constraints keep the factors biologically interpretable.
- **Selection bias:**
  - Selecting genes outside cross-validation makes error estimates optimistic ([Ambroise & McLachlan 2002, *PNAS*](https://doi.org/10.1073/pnas.102102699)).
  - Tuning by CV and then reporting that same CV error is biased; nested CV fixes it ([Varma & Simon 2006, *BMC Bioinformatics*](https://doi.org/10.1186/1471-2105-7-91)).
  - Overfitting during model selection can be as large as the differences between algorithms ([Cawley & Talbot 2010, *JMLR*](https://www.jmlr.org/papers/v11/cawley10a.html)).

### 3.2 Synthesis
- **Where the IFN-α bottleneck actually is.** The repo's evidence places it in the Stage 2 sparse penalty, not in the basis: FA7 alone gives +18.06 pp in P02, and L2 recovers P02 and P07. A supervised Stage 1 would mainly change which axis is extracted.
- **What a supervised Stage 1 costs:**
  - each fold gets its own basis, so the common coordinate system used to compare discovery, LOPO and patient-specific models (Q2/Q3) is lost;
  - selection bias, unless every supervised step is refit inside each fold;
  - overfitting risk with only 7 supported patients.
- **A fair test of the idea.** Use a one-parameter family that contains the current pipeline as a limit: a linear supervised autoencoder on panel genes that minimizes reconstruction loss + λ × logistic loss on the latent factors. λ = 0 is roughly "PCA-like Stage 1, then Stage 2"; large λ is end-to-end. Choose λ by inner leave-one-patient-out.

### Implication for this repo
- **Q1 first step.** Before any end-to-end model, finish the cheap Stage 2 fix: choose C and the penalty (L1, L2 or elastic net) by inner LOPO within each fold. If that closes the PCA/FA gaps across panels, end-to-end adds little for Q1.
- **Q1/Q3 ladder.** Try supervised PCA within a panel, then sPLS-DA, then the linear supervised AE λ sweep. Refit every supervised step inside the LOPO folds (Ambroise & McLachlan). Report lift, P20 lift, and loading stability across folds using Riley & Collins-style instability plots.
- **Q3.** Knowledge-guided but label-free factorization (Spectra, f-scLVM, PLIER) fits "programs break apart" better: it lets a Hallmark set split into several factors and edits membership, while Stage 1 stays label-free. Spectra needs cell types; `predicted.annotation` can supply them.
- **Q2.** With supervised factors, patient-specific models no longer share one coordinate system. Either align factors by Hungarian/Procrustes matching as in F3, or keep the unsupervised basis for Q2 and use supervised factors only for Q1 prediction.
- **Nonlinear models.** Try scANVI or deep end-to-end models only after the linear steps, and only with inductive nested LOPO. Seven supported patients are too few to tune deep models without overfitting the selection (Cawley & Talbot).
- **Old34 provenance.** If the list was chosen with knowledge of this cohort's malignant-vs-donor enrichment, the selection-bias argument applies to Stage 0 as well. The matched-null in §4 and the collaborator cohort are then required for Q1 claims.
- **New patients.** Freeze λ, C, K and the gene lists before unblinding the collaborator cohort.

---

## 4. Gene-set scoring baselines and matched random gene-set nulls

### 4.1 Scoring methods
- **AUCell** ([Aibar et al. 2017, *Nature Methods*](https://doi.org/10.1038/nmeth.4463)): ranks genes within each cell and computes the AUC of the gene set among the top-ranked genes. It does not depend on normalization.
- **UCell** ([Andreatta & Carmona 2021, *CSBJ*](https://doi.org/10.1016/j.csbj.2021.06.043)): a Mann–Whitney U statistic on per-cell ranks, robust to dataset size and heterogeneity. A Python version (pyUCell) is listed in 2026 citation metadata.
- **singscore** ([Foroutan et al. 2018, *BMC Bioinformatics*](https://doi.org/10.1186/s12859-018-2435-4)): rank-based single-sample scoring.
- **ssGSEA** ([Barbie et al. 2009, *Nature*](https://doi.org/10.1038/nature08460); credited as the ssGSEA source in the GSVA documentation): GSVA's implementation normalizes scores across samples by default (`ssgsea.norm`).
- **GSVA** ([Hänzelmann, Castelo & Guinney 2013, *BMC Bioinformatics*](https://doi.org/1186/1471-2105-14-7)): estimates each gene's distribution across the whole sample population, so scores depend on the cohort.
- **VISION** ([DeTomaso et al. 2019, *Nature Communications*](https://doi.org/10.1038/s41467-019-12235-0)): signature scores plus local autocorrelation on the cell-cell similarity map.
- **Control-gene scoring** ([Tirosh et al. 2016, *Science*](https://doi.org/10.1126/science.aad0501)): subtracts the mean of control genes drawn from the same expression bins. It is implemented in Seurat AddModuleScore and in [scanpy `score_genes`](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.score_genes.html) (defaults 25 bins, 50 control genes).
- **Benchmarks:**
  - [Zhang et al. 2020, *CSBJ*](https://doi.org/10.1016/j.csbj.2020.10.007): 7 algorithms on 32 datasets.
  - [Holland et al. 2020, *Genome Biology*](https://doi.org/10.1186/s13059-020-1949-z): bulk tools work on single-cell data, and the choice of gene set matters more than the scoring statistic.
  - decoupleR ([Badia-i-Mompel et al. 2022, *Bioinformatics Advances*](https://doi.org/10.1093/bioadv/vbac016)): simple linear models and a consensus of top methods performed best.
- **Hallmark collection:** 50 refined sets built from more than 4,000 founder sets ([Liberzon et al. 2015, *Cell Systems*](https://doi.org/10.1016/j.cels.2015.12.004)).

### 4.2 Theory for the null
- [Venet, Dumont & Detours 2011, *PLoS Computational Biology*](https://doi.org/10.1371/journal.pcbi.1002240): 60% of 47 published breast-cancer signatures were not better than random sets of the same size, and more than 90% of random sets with over 100 genes were significant, because of proliferation.
- [Goeman & Bühlmann 2007, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btm051) distinguish two kinds of null. A competitive null asks whether this gene set is special relative to other genes; a self-contained null asks whether there is any signal at all. They also show that p-values from sampling genes are anti-conservative.
- [Wu & Smyth 2012, *NAR*, CAMERA](https://doi.org/10.1093/nar/gks461): correlation between genes inflates tests that permute genes.

### Implication for this repo
- **Tracker gaps.** These are "exact UCell/AUCell/ssGSEA comparator" and "Null Model 1".
- **One-score comparator.** For each panel, compute UCell, AUCell, singscore, unnormalized ssGSEA, the control-binned score and mean-z. Use each as a single feature in Stage 2 under the same LOPO and metric code; for pure rank scores, take the sign from training patients and compute held-out AUPRC directly.
  - UCell, AUCell and singscore are computed per cell, so they are inductive by construction.
  - GSVA and normalized ssGSEA are transductive, like the current Stage 1.
- **Null Model 1 specification.**
  - Draw random gene sets matched on covered size and on bins of mean expression and detection rate.
  - Add a variant also matched on co-expression (decile of mean pairwise correlation), because Venet and CAMERA show that correlation drives random-set "success".
  - Push each random set through the same selection the real panel received (best of method × K, or the same fixed method|K), so the null includes D1's winner's curse.
  - Empirical p = (1 + #null ≥ observed) / (R + 1) for median lift and P20 lift, with BH correction over the 41 panels.
- **Compute in two tiers.** Tier 1: one-score nulls with R = 1,000 on CPU. Tier 2: DR + Stage 2 nulls with R ≈ 100, only for allograft, apoptosis, TNFα/NF-κB and IFN-α.
- **Self-contained null.** Permute labels within each patient (this keeps prevalence fixed); this is the dossier's Plan 2. It answers "is there any signal", while matched random sets answer "is this prior special" (Goeman & Bühlmann).
- **Composition baseline.** F9b showed that the cell-type-only ceiling beats PC8 in P02 and P03. Random sets will pick up HSC/MPP composition, the analogue of Venet's proliferation effect. Report panel lift against the cell-type ceiling and within cell types (F9b right), not only against random sets.
- **Q3.** Run VISION autocorrelation for each panel and each factor to see which programs vary locally on the cell graph regardless of labels.
- **Claim gate.** "Selected panels beat matched-size random gene sets" stays unsupported until this analysis lands.

---

## 5. Learning curves and sample-size planning

### 5.1 Methods
- **Inverse power-law learning curves:**
  - [Mukherjee et al. 2003, *Journal of Computational Biology*](https://doi.org/10.1089/106652703321825928): power-law curves plus a permutation test of significance at each training size.
  - [Figueroa et al. 2012, *BMC Medical Informatics and Decision Making*](https://doi.org/10.1186/1472-6947-12-8): weighted nonlinear least-squares fits that predict performance with confidence intervals; 568 models.
- **Scaling and curve shape:**
  - [Hestness et al. 2017, *arXiv*](https://arxiv.org/abs/1712.00409): power-law scaling holds across domains, and better models shift the curve without changing its exponent.
  - [Viering & Loog 2023, *IEEE TPAMI*](https://doi.org/10.1109/TPAMI.2022.3220744): learning curves are diverse, there is no universal shape, and badly behaved curves exist.
- **Test-set noise:** [Beleites et al. 2013, *Analytica Chimica Acta*](https://doi.org/10.1016/j.aca.2012.11.007) show that small test sets mask learning curves. About 75–100 test samples are needed to validate a good classifier, and comparing two classifiers needs hundreds.
- **Clinical prediction-model sample size:**
  - Development: [Riley et al. 2020, *BMJ*](https://doi.org/10.1136/bmj.m441).
  - External validation: [Riley et al. 2024, *BMJ*](https://doi.org/10.1136/bmj-2023-074821).
  - Instability plots from bootstrap refits: [Riley & Collins 2023, *Biometrical Journal*](https://doi.org/10.1002/bimj.202200302).
- **Single-cell power:**
  - scPower ([Schmid et al. 2021, *Nature Communications*](https://doi.org/10.1038/s41467-021-26779-7)): power as a function of individuals, cells per individual and depth. For a fixed budget, many shallowly sequenced cells beat fewer deep ones.
  - hierarchicell ([Zimmerman & Langefeld 2021, *BMC Genomics*](https://doi.org/10.1186/s12864-021-07635-w)): calculators that count cells instead of individuals overstate power.
  - [Mandric et al. 2020, *Nature Communications*](https://doi.org/10.1038/s41467-020-19365-w): for cell-type eQTLs, more individuals at lower coverage beat fewer at higher coverage.
- **Scale and few-sample learning:**
  - [DenAdel et al. 2024, *bioRxiv*](https://doi.org/10.1101/2024.12.13.628448): single-cell foundation-model performance plateaus at a fraction of the available pretraining corpus.
  - scDesign3 ([Song et al. 2024, *Nature Biotechnology*](https://doi.org/10.1038/s41587-023-01772-1)) simulates data with interpretable parameters and can produce positive and negative controls.
  - ScRAT ([Mao et al. 2024, *Bioinformatics*](https://doi.org/10.1093/bioinformatics/btae067)) uses sample mixup to predict phenotypes when there are few samples.

### 5.2 What makes a learning curve good enough for a grant
1. The x-axis counts **patients** (and malignant cells per patient), not cells. Cells are pseudo-replicates (Zimmerman; hierarchicell; scPower).
2. The **whole pipeline is refit** at every size: standardization, inductive Stage 1, and choice of C, K and method inside nested folds (Varma & Simon; Cawley & Talbot).
3. Each point has **repeated draws and an interval**: subsets of patients, a cluster bootstrap, nested-CV variance (Bates).
4. A **parametric fit with a check**: an inverse power law fitted by weighted least squares (Figueroa), checked by fitting on small sizes and predicting the largest. The raw points are shown too (Viering & Loog).
5. **Test noise is shown separately** from learning, with the number of test positives per patient and Boyd intervals (Beleites).
6. **Metrics that do not move with prevalence**: lift, AUPRC/π and AUROC (McDermott).
7. **Stability of what is learned**: selected factors and coefficients plotted against training size (Riley & Collins).
8. **A pre-registered external validation** with a stated achievable CI width per new patient (Riley 2024).

### Implication for this repo
- **What D15 currently shows.** D15 measures within-cohort depth at a fixed C, and the H-S2-06 claim gate still holds. At 500 training cells there are about 13 malignant cells for K = 40 features, a regime in which a fixed L1 C collapses to an all-zero model by construction.
- **D15b.** Re-tune C and the penalty inside each subsample by inner LOPO. The IFN-α flat line should disappear, since L2 already recovers P02 and P07.
- **D15c: patient-drop curve.** For each held-out supported patient:
  - train on k = 1 to 5 of the other six supported patients, using all subsets (at most 20 per k);
  - always include the six unsupported patients as negatives;
  - refit Stage 1 inductively on the training patients for the IFN-α pair and two focus panels;
  - plot lift against k, showing the spread across subsets.

  This is the curve a grant reviewer needs, and it answers the "drop whole patients" gap in the tracker.
- **D15d: two-dimensional version.** Cross k patients with a cap on malignant cells per patient (for example 10, 25, 50, all). This shows whether more patients or more cells per patient helps more, the analogue of the scPower and Mandric results.
- **Fitting and extrapolation.** Fit a weighted inverse power law to lift against k, check it by predicting k = 5 from k ≤ 4, and state that with 6 training patients the extrapolation is planning-grade only.
- **Noise floor.** P07 (22), P06 (44) and P01 (45) are below the roughly 75–100 positives Beleites gives for precise validation. Summarize learning curves at the patient level rather than reading single-patient curves.
- **New patients.** Pre-specify the primary metric (per-patient lift, with the median and P20 across patients) and lock the analysis. Use Riley 2024's approach to state the CI width achievable for each new patient given its malignant count. Treat each new patient as one draw at the patient level.

---

## 6. Prioritized next analyses
1. **D16 genotype and label-assay shortcut audit:** sex-chromosome, HLA and ERAP2 gene ablation; scoring of recipient-healthy cells; label-provenance column. It is cheap and is the highest risk to Q1 and Q2.
2. **Stage 2 re-tuning:** C and penalty chosen by inner LOPO, plus patient-balanced weights. Report AUROC and AUPRC/π next to lift.
3. **Narrow inductive refit:** IFN-α PCA40/FA40 × 7 patients, then a balanced-and-inductive variant.
4. **Baselines and nulls:** one-score comparators (UCell, AUCell, singscore, ssGSEA, control-binned score) and tier 1 of Null Model 1.
5. **Learning curves:** D15b, then D15c (patient-drop) and D15d (2D).
6. **Q2/Q3 decompositions:** MOFA+ multi-group and per-patient cNMF meta-programs.
7. **Supervised ladder:** supervised PCA, sPLS-DA, then the linear supervised AE λ sweep, all inside nested LOPO.
8. **External-validation protocol** for the collaborator cohort, locked before the data arrive.
