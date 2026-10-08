# Draft email to Yu-Hsiang: new MRD patients (W8.1)

> **DRAFT for Minhang to edit and send. It has not been sent.** Agents never send email.
> - Task: W8.1. Ask: A-20261008-12 ("Incorporate new patient MRD data from Yu-Hsiang (# I need to email him)"). The draft also covers A-20261008-18 (the manuscript timing plot).
> - Before sending: check the P10–P12 values below against `stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz`, an experiment-dir-relative path with columns `Days from SCT` and `Days.from.Relapse`. Decide whether to cc anyone, and adjust the tone.
> - After the reply arrives, record it verbatim in `research/asks/<date>_yu-hsiang-reply.md` or in a meeting devlog entry, as W8.1's done-when requires. Then update `integration_plan.md` §3 and §4 if anything changes before the plan is locked (W8.2).

---

**Subject:** New MRD patients for the malignant-vs-normal classifier, plus a few metadata questions (P10–P12 relapse status, timing plot, label provenance)

Hi Yu-Hsiang,

I'd like to add the new post-transplant MRD patients to the malignant-vs-normal classifier I've been running on the current 13-patient CITE-seq MRD set, and use them as an external test of the shared model. I want to lock the analysis plan before I look at any new labels or predictions, so a few specifics up front would help a lot.

**1. The new MRD patients**
- How many new patients are there, and which samples does each have (MRD only, or also preSCT or relapse)?
- Which assay: CITE-seq with the same ADT panel as the current set, or Multiome/CARAMEL-seq? Which sequencing batch or run did each sample come from?
- Format: ideally one `.h5ad` laid out like `adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad`, with:
  - raw RNA counts and ADT counts;
  - the same obs columns: `patient`, `sample`, `Time`, `Tech`, `timepoint_type`, `CN.label`, `predicted.annotation` (+ score), `predicted.pseudotime`, `Days from SCT` and `Days.from.Relapse`.
  If the genome reference or Cell Ranger version differs from the current object, please tell me.
- Where will the files be (a server path or a transfer link), and roughly how large are they?
- Are any of the new samples additional timepoints from P01–P13 rather than new patients?

**2. Malignant labels and donor/recipient calls (new patients)**
- For each patient, which evidence defined `CN.label`: Souporcell donor/recipient plus Numbat CNVs against a donor reference, mtDNA variants, immunophenotype (e.g. CD5/CD56), or a cluster-based reference? In other words, the equivalent of the per-patient label table (Table S2) in the bioRxiv preprint.
- Per-cell donor/recipient assignments with the Souporcell singlet status or posterior, so I can keep recipient-normal cells separate from donor cells.
- Donor and recipient sex for each transplant. Sex-mismatched pairs matter here, because chrY genes and XIST sit inside some of the gene panels.
- How "unassigned" cells were defined. They are currently excluded from training and evaluation.

**3. Two confirmations about the current cohort**
- P10, P11 and P12 have no relapse sample and no malignant cells. Their MRD samples have `Days.from.Relapse` of −1329, −1158 and −1382, and `Days from SCT` of 111, 32 and 107. Are these the three patients who stayed in remission, so that the value is time to last follow-up (censoring)? Or did they relapse later? Either way, what does `Days.from.Relapse` mean for them?
- Could you confirm the per-patient label provenance for P01–P13 (Table S2 in the preprint), and that the `CN.label` in my copy of the object is the final version?

**4. The timing plot**
- You mentioned the manuscript has a plot of days from SCT and days before relapse by patient. Could you send that figure, and the per-sample table behind it if that's easy? I'd like to line our per-patient results up against it rather than recreate it.

**What I'll share back:** before analysing the new patients, the locked analysis plan (a frozen model and configuration, with no tuning on the new patients). Afterwards, a short summary of per-patient performance on the new patients: held-out AUPRC minus each patient's malignant fraction, the median and worst-patient summaries with intervals, and a list of what replicated and what didn't. I'm also happy to send per-cell scores or figures if they'd be useful for the manuscript. Nothing will go beyond our group without checking with you first.

Thanks a lot!

Minhang
