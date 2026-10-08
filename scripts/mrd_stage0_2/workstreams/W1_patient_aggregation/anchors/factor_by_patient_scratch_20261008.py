"""Regression anchor, not a pipeline module. Frozen: do not edit, import or extend.

Provenance
- Source: agent scratch script ``factcheck_agent/factor_by_patient.py`` from the
  2026-10-08 fact-check session (session scratchpad; the original file was not kept).
- Copied here on 2026-10-08 (wave-1 package P4_workstreams_core). The only change is
  path resolution: the hard-coded EXP constant and cohort-cache path were replaced by
  ``experiment_dir()`` and ``cohort_cache()`` from ``sc_classification.utils.paths``
  through the standard workstream import header (``parents[5]`` because this file
  sits one level deeper, in ``anchors/``). The computation is unchanged.
- Context: the fact-check corrections recorded in devlog entry
  2026-10-08_team-report-devlog-pivot (asks A-20261008-02, A-20261008-04, A-20261008-05).

Purpose
Label-blind factor-by-patient variance allocation for the IFN-alpha (hallmark
interferon alpha response) PCA40 and FA40 transductive Stage 1 scores (seed 42): each
patient's share of a factor's globally centred sum of squares divided by that
patient's share of cells (enrichment), the patient with the largest enrichment per
non-degenerate factor, and the within-patient malignant-minus-normal mean of the
z-scored factor score for PC8 and FA7/FA8. It tests the team claim that FA keeps
allocating factors to the highest-prevalence patient (H-S2-07, H-S1-05).

Numbers it reproduced on 2026-10-08 (agent session)
- Cell share (%): P09 28.5, P01 19.7, P13 9.6, P05 7.3, P11 6.7, P06 5.4, P10 4.5,
  P12 4.1, P03 3.9, P07 3.6, P02 2.4, P08 2.3, P04 1.9.
- FA40 has 4 degenerate factors (37-40, sd < 1e-8); PCA40 has none.
- Patient with the largest SS enrichment per factor: PCA P02 on 31/40, P07 on 7,
  P01 on 2; FA P02 on 27/36, P07 on 7, P01 on 2. P03 is never the top patient.
- Median enrichment across factors (PCA / FA): P02 1.53 / 1.48; P03 1.06 / 1.04;
  P04 0.94 / 0.87.
- Within-patient malignant-minus-normal mean z (P01, P02, P03, P04, P06, P07, P09):
  PC8 -0.22, -0.78, -0.60, -0.89, -0.05, -0.76, -0.88;
  FA7 0.27, 0.78, 0.62, 0.89, 0.09, 1.33, 0.77.
Caveat: scratch, label-blind variance allocation; descriptive only.

Consumer
W1.1 (D16, slug factor_patient_matrix_v1) must reproduce these IFN-alpha K40 rows
before it reports new numbers.

Inputs (EXP-relative): the cohort cache (study_config cohort_cache_relpath; obs
columns patient and CN.label, read in backed mode) and
stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method={pca,fa}/k=40/seed=42/scores.npy.

Run manually from the repo root with the lab env (CPU, seconds to minutes):
    python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/anchors/factor_by_patient_scratch_20261008.py

Original scratch docstring (verbatim):
Scratch (not a repo artifact): factor-by-patient sum-of-squares share for IFN-alpha PCA40 vs FA40.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[5]  # workstreams/<W#_slug>/anchors/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, cohort_cache
import numpy as np, pandas as pd, anndata as ad
EXP = experiment_dir()
cache = str(cohort_cache())
a = ad.read_h5ad(cache, backed="r"); obs = a.obs[["patient", "CN.label"]].copy(); a.file.close()
pat = obs["patient"].astype(str).to_numpy(); y = (obs["CN.label"].astype(str) == "cancer").to_numpy()
panel = "single_geneset__hallmark_interferon_alpha_response"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
pats = sorted(pd.unique(pat))
cellshare = pd.Series({p: (pat == p).mean() for p in pats})
print("cell share %:", (100 * cellshare).round(1).to_dict())
print("malignant counts:", {p: int(y[pat == p].sum()) for p in pats})
for method in ["pca", "fa"]:
    S = np.load(f"{EXP}/stage1_dr/panel_id={panel}/method={method}/k=40/seed=42/scores.npy")
    sd = S.std(0); print(method, "n factors with sd<1e-8:", int((sd < 1e-8).sum()), "idx", list(np.flatnonzero(sd < 1e-8) + 1))
    Sc = S - S.mean(0)
    ss = pd.DataFrame({p: (Sc[pat == p] ** 2).sum(0) for p in pats}, index=[f"f{j+1}" for j in range(S.shape[1])])
    share = ss.div(ss.sum(1), axis=0)
    enrich = share.div(cellshare, axis=1)
    enrich = enrich[sd >= 1e-8]
    top = enrich.idxmax(axis=1)
    print(f"== {method}: patient with largest SS enrichment per factor (count over {len(enrich)} non-degenerate factors)")
    print(top.value_counts().to_string())
    print("median enrichment across factors:", enrich.median().round(2).to_dict())
    sel = ["f7", "f8"] if method == "fa" else ["f8"]
    print("enrichment for", sel); print(enrich.loc[sel].round(2).to_string())
    # malignant-cell centered mean by patient for those factors (z units)
    Z = (S - S.mean(0)) / np.where(sd < 1e-8, 1, sd)
    m = pd.DataFrame({p: Z[(pat == p) & y].mean(0) for p in ["P01","P02","P03","P04","P06","P07","P09"]}, index=[f"f{j+1}" for j in range(S.shape[1])])
    n = pd.DataFrame({p: Z[(pat == p) & ~y].mean(0) for p in ["P01","P02","P03","P04","P06","P07","P09"]}, index=[f"f{j+1}" for j in range(S.shape[1])])
    print("within-patient malignant-minus-normal mean (z) for", sel); print((m - n).loc[sel].round(2).to_string())
