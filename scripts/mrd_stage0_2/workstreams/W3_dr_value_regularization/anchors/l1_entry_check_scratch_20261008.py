"""Regression anchor, not a pipeline module. Frozen: do not edit, import or extend.

Provenance
- Source: agent scratch script ``factcheck_agent/l1_entry_check.py`` from the
  2026-10-08 fact-check session (session scratchpad; the original file was not kept).
- Copied here on 2026-10-08 (wave-1 package P4_workstreams_core). The only change is
  path resolution: the hard-coded EXP constant and cohort-cache path were replaced by
  ``experiment_dir()`` and ``cohort_cache()`` from ``sc_classification.utils.paths``
  through the standard workstream import header (``parents[5]`` because this file
  sits one level deeper, in ``anchors/``). The computation is unchanged.
- Context: the fact-check corrections recorded in devlog entry
  2026-10-08_team-report-devlog-pivot (asks A-20261008-03, A-20261008-04).

Purpose
Checks the balanced-L1 entry rule behind AGENTS.md R5 on the stored IFN-alpha
(hallmark interferon alpha response) PCA40 and FA40 transductive Stage 1 scores
(seed 42) at the stored C = 1e-4. For each supported held-out patient (LOPO fold) it
standardizes the training-fold scores, computes every factor's training malignant-minus-
normal mean gap, and compares the largest |gap| (and the PC8 / FA7 / FA8 gaps) with the
entry threshold 4/(C*n_train). It also prints that threshold at the subsample sizes used
by the D15 cell-count learning curve. It tests the IFN-alpha PCA40-vs-FA40 coverage gap
(H-S2-08, H-S2-07) and the D15 fixed-C limitation (H-S2-06).

Numbers it reproduced on 2026-10-08 (agent session)
- Threshold 4/(C*n_train): P01 0.829 (n_train 48,258), P02 0.682, P03 0.692,
  P04 0.678, P06 0.703, P07 0.690, P09 0.930 (n_train 42,992).
- |gap| for PC8: P01 0.728, P02 0.740, P03 0.770, P04 0.722, P06 0.755, P07 0.741,
  P09 0.704. PC8 passes in P02/P03/P04/P06/P07 and fails in P01/P09.
- |gap| for FA7: P01 0.705, P02 0.659, P03 0.704, P04 0.682, P06 0.693, P07 0.677,
  P09 0.658. FA7 passes only in P03 (by 0.012) and P04 (by 0.004).
- Compared outside this script with the stored L1 coefficients, the pass/fail calls
  matched all 14 stored IFN-alpha zero/nonzero outcomes (7 folds x 2 methods).
- FA40 on IFN-alpha has 4 degenerate constant factors (37-40, sd < 1e-8).
- Fixed-C entry thresholds at C = 1e-4: 1.6 SD at n = 25,000 and 8 SD at n = 5,000.
Caveats
- The derivation ignores liblinear's intercept penalty, which is harmless at w = 0
  with balanced weights.
- Known quirk, left unchanged: FA40 factors 37-40 are exactly zero, so their
  training-fold standardization is 0/0 = NaN (numpy RuntimeWarning). np.argmax returns
  the first NaN, so for method fa the best_factor / best_abs_gap / passes columns point
  at factor 37 and are not meaningful. Read gap_pc8_or_fa7 and gap_fa8 for FA;
  n_factors_passing ignores the NaN factors. The regularization contract must drop
  zero-variance features before computing margins.

Consumer
W3.1 (D19, slug regularization_contract_v1): ``regularization_contract.l1_entry_threshold``
and ``l1_entry_margins`` must reproduce these IFN-alpha zero/nonzero outcomes before
the entry-margin audit reports new numbers.

Inputs (EXP-relative): the cohort cache (study_config cohort_cache_relpath; obs
columns patient and CN.label, read in backed mode) and
stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method={pca,fa}/k=40/seed=42/scores.npy.

Run manually from the repo root with the lab env (CPU, seconds to minutes):
    python scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/anchors/l1_entry_check_scratch_20261008.py

Original scratch docstring (verbatim):
Scratch check (not a repo artifact): L1 entry threshold vs training-fold class-mean gap.
For balanced-weight logistic L1 (liblinear) at w=0, |dLoss/dw_j| = C*(n/4)*|mean_pos(z_j)-mean_neg(z_j)|,
so feature j can enter only if |gap_j| > 4/(C*n_train).
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
pat = obs["patient"].astype(str).to_numpy(); y = (obs["CN.label"].astype(str) == "cancer").to_numpy().astype(int)
panel = "single_geneset__hallmark_interferon_alpha_response"
C = 1e-4
pd.set_option("display.width", 250)
rows = []
for method in ["pca", "fa"]:
    S = np.load(f"{EXP}/stage1_dr/panel_id={panel}/method={method}/k=40/seed=42/scores.npy")
    for h in ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]:
        tr = pat != h
        Z = S[tr]; Z = (Z - Z.mean(0)) / Z.std(0)
        yt = y[tr]
        gap = Z[yt == 1].mean(0) - Z[yt == 0].mean(0)
        n = tr.sum(); thr = 4.0 / (C * n)
        j = int(np.argmax(np.abs(gap)))
        rows.append(dict(method=method, heldout=h, n_train=int(n), n_train_mal=int(yt.sum()), entry_threshold=thr,
                         best_factor=j + 1, best_abs_gap=abs(gap[j]), passes=abs(gap[j]) > thr,
                         gap_pc8_or_fa7=gap[7] if method == "pca" else gap[6],
                         gap_fa8=gap[7] if method == "fa" else np.nan,
                         n_factors_passing=int((np.abs(gap) > thr).sum())))
df = pd.DataFrame(rows)
print(df.round(3).to_string())
# learning-curve thresholds
for n in [500, 1000, 2000, 5000, 10000, 25000]:
    print(n, "L1 C=1e-4 entry threshold (SD units):", round(4 / (C * n), 2), "| allograft EN C=4.64e-4 rho=0.5 threshold:", round(4 * 0.5 / (4.64e-4 * n), 3))
