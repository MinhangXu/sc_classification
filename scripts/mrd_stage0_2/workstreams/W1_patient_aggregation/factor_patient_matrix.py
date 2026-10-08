"""D16 factor x patient allocation matrix on stored Stage 1 scores (W1.1).

For one stored Stage 1 representation (panel x method x K, seed 42, transductive), this
module describes how each factor's score-space energy is spread over patients:

- cell share and sum-of-squares (SS) share per patient, with SS taken about the cohort
  mean of the factor (the reference point of the label-blind Stage 1 fit);
- enrichment = SS share / cell share, split exactly into a location part (squared
  standardized patient mean) and a spread part (squared patient/cohort SD ratio);
- the same split within each label (normal cells only, malignant cells only);
- the within-patient malignant-minus-normal shift, in cohort-SD units and as a
  within-patient pooled-SD Cohen's d;
- eta2 of patient within each label (D14 strata: normal patients with >=200 normal cells,
  malignant patients with >=20 malignant cells) and each patient's additive share of it;
- univariate held-out AP and lift (AP - held-out prevalence) per supported patient, with
  the factor's sign fixed on the training folds (sign of the training-fold standardized
  class-mean gap). The sign-agnostic max(AP(+s), AP(-s)) of F6 is kept for regression;
- a degenerate flag (cohort SD < 1e-8).

It does not refit anything and reads nothing from disk: callers pass the score matrix and
the per-cell patient/label arrays (same row order). Score-space allocation is not loading
leverage. Whether a patient changes the loadings needs a leave-out or reweighted refit
(D17), which this module does not do.

Pure numpy/pandas; no experiment paths. The runner is run_d16_factor_patient_matrix.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEGENERATE_SD = 1e-8
NORMAL_MIN_CELLS = 200  # D14 normal stratum parameter (not the support rule)
MALIGNANT_MIN_CELLS = 20  # D14 malignant stratum parameter (not the support rule)
HIGH_PREVALENCE = ("P02", "P03", "P04")
REP_KEYS = ["stage0_panel_id", "stage1_method", "k"]
FACTOR_KEYS = REP_KEYS + ["factor_index"]


# --------------------------------------------------------------------------------------
# cohort labels
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cohort:
    """Per-cell patient ids and malignant flags, in Stage 1 score row order."""

    patient: np.ndarray
    malignant: np.ndarray
    supported: tuple = ()
    high_prevalence: tuple = HIGH_PREVALENCE
    patients: tuple = field(default=())

    def __post_init__(self) -> None:
        patient = np.asarray(self.patient).astype(str)
        malignant = np.asarray(self.malignant).astype(bool)
        if patient.shape != malignant.shape or patient.ndim != 1:
            raise ValueError("patient and malignant must be 1-D arrays of equal length")
        object.__setattr__(self, "patient", patient)
        object.__setattr__(self, "malignant", malignant)
        if not self.patients:
            object.__setattr__(self, "patients", tuple(sorted(pd.unique(patient))))
        unknown = set(self.supported) - set(self.patients)
        if unknown:
            raise ValueError(f"supported patients not in the cohort: {sorted(unknown)}")

    @property
    def n_cells(self) -> int:
        return int(self.patient.size)


def supported_patients(
    patient: np.ndarray, malignant: np.ndarray, *, min_malignant: int = 10
) -> tuple:
    """Support rule: at least ``min_malignant`` malignant cells and both classes present."""
    frame = pd.DataFrame({"patient": np.asarray(patient).astype(str), "y": np.asarray(malignant).astype(bool)})
    counts = frame.groupby("patient")["y"].agg(["sum", "size"])
    keep = counts.loc[(counts["sum"] >= min_malignant) & (counts["size"] > counts["sum"])].index
    return tuple(sorted(str(p) for p in keep))


# --------------------------------------------------------------------------------------
# average precision (sklearn-equivalent, on pre-sorted scores)
# --------------------------------------------------------------------------------------


def _ap_desc(scores_desc: np.ndarray, labels_desc: np.ndarray) -> float:
    """AP for scores already sorted in descending order (ties grouped like sklearn).

    Same quantity as sklearn.metrics.average_precision_score: the sum over distinct
    thresholds of (recall_k - recall_{k-1}) * precision_k.
    """
    n = scores_desc.size
    if n == 0:
        return float("nan")
    tps = np.cumsum(labels_desc, dtype=np.int64)
    n_pos = int(tps[-1])
    if n_pos == 0:
        return float("nan")
    idx = np.flatnonzero(np.diff(scores_desc))
    idx = np.append(idx, n - 1)
    tp = tps[idx]
    precision = tp / (idx + 1.0)
    dtp = np.diff(tp, prepend=0)
    return float(np.dot(dtp, precision) / n_pos)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """sklearn-equivalent average precision for binary labels (ties grouped)."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    order = np.argsort(-score, kind="stable")
    return _ap_desc(score[order], y[order])


# --------------------------------------------------------------------------------------
# allocation (score-space energy by patient)
# --------------------------------------------------------------------------------------


def _group_moments(x: np.ndarray, mask: np.ndarray, center: np.ndarray) -> tuple:
    """n, mean, SS about ``center`` and within-group SS for the masked rows of x."""
    sub = x[mask]
    n = int(sub.shape[0])
    if n == 0:
        nan = np.full(x.shape[1], np.nan)
        return 0, nan, np.zeros(x.shape[1]), nan
    mean = sub.mean(axis=0)
    ss_center = ((sub - center) ** 2).sum(axis=0)
    ss_within = ((sub - mean) ** 2).sum(axis=0)
    return n, mean, ss_center, ss_within


def _safe_div(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(num, den, out=out, where=(den != 0) & np.isfinite(den))
    return out


def allocation_table(
    scores: np.ndarray,
    cohort: Cohort,
    *,
    degenerate_sd: float = DEGENERATE_SD,
    normal_min_cells: int = NORMAL_MIN_CELLS,
    malignant_min_cells: int = MALIGNANT_MIN_CELLS,
) -> pd.DataFrame:
    """Long table, one row per factor x patient, of the allocation quantities.

    Enrichment is NaN on degenerate factors (cohort SD < ``degenerate_sd``); the label
    splits are NaN for a patient with no cells of that label.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != cohort.n_cells:
        raise ValueError(f"scores shape {x.shape} does not match {cohort.n_cells} cells")
    n_total, k = x.shape
    mal = cohort.malignant
    pats = cohort.patients
    center = x.mean(axis=0)
    sd = x.std(axis=0)
    degenerate = sd < degenerate_sd
    sd_safe = np.where(degenerate, np.nan, sd)
    ss_total = ((x - center) ** 2).sum(axis=0)
    ss_total = np.where(degenerate, np.nan, ss_total)

    label_masks = {"normal": ~mal, "malignant": mal}
    n_label = {name: int(m.sum()) for name, m in label_masks.items()}
    ss_label = {
        name: np.where(degenerate, np.nan, ((x[m] - center) ** 2).sum(axis=0)) for name, m in label_masks.items()
    }

    # eta2 of patient within label, on D14 strata
    eta2 = {}
    eta2_contrib = {}
    for name, lmask in label_masks.items():
        min_cells = normal_min_cells if name == "normal" else malignant_min_cells
        included = [p for p in pats if int(np.sum(lmask & (cohort.patient == p))) >= min_cells]
        contrib = {p: np.full(k, np.nan) for p in pats}
        if len(included) >= 2:
            inc_mask = lmask & np.isin(cohort.patient, included)
            grand = x[inc_mask].mean(axis=0)
            sst = ((x[inc_mask] - grand) ** 2).sum(axis=0)
            between = np.zeros(k)
            for p in included:
                pm = inc_mask & (cohort.patient == p)
                n_p = int(pm.sum())
                c = n_p * (x[pm].mean(axis=0) - grand) ** 2
                between += c
                contrib[p] = _safe_div(c, sst)
            eta2[name] = np.where(degenerate, np.nan, _safe_div(between, sst))
            for p in included:
                contrib[p] = np.where(degenerate, np.nan, contrib[p])
        else:
            eta2[name] = np.full(k, np.nan)
        eta2_contrib[name] = contrib

    frames = []
    factor_index = np.arange(1, k + 1)
    for p in pats:
        pmask = cohort.patient == p
        n_p, mean_p, ss_p, ssw_p = _group_moments(x, pmask, center)
        n_mal = int(np.sum(pmask & mal))
        n_norm = n_p - n_mal
        row = {
            "factor_index": factor_index,
            "patient": p,
            "supported": p in cohort.supported,
            "high_prevalence": p in cohort.high_prevalence,
            "n_cells": n_p,
            "n_malignant": n_mal,
            "n_normal": n_norm,
            "prevalence": n_mal / n_p if n_p else np.nan,
            "cell_share": n_p / n_total,
            "factor_sd": sd,
            "degenerate": degenerate,
            "ss": np.where(degenerate, np.nan, ss_p),
            "ss_share": _safe_div(ss_p, ss_total),
        }
        row["enrichment"] = row["ss_share"] / row["cell_share"]
        row["enrichment_location"] = ((mean_p - center) / sd_safe) ** 2
        row["enrichment_spread"] = _safe_div(ssw_p / max(n_p, 1), sd_safe**2)
        for name, lmask in label_masks.items():
            m = pmask & lmask
            n_pl, mean_pl, ss_pl, ssw_pl = _group_moments(x, m, center)
            share = _safe_div(ss_pl, ss_label[name]) if n_pl else np.full(k, np.nan)
            cshare = n_pl / n_label[name] if n_label[name] else np.nan
            row[f"ss_{name}"] = np.where(degenerate, np.nan, ss_pl)
            row[f"ss_share_{name}"] = share
            row[f"cell_share_{name}"] = cshare
            row[f"enrichment_{name}"] = share / cshare if n_pl else np.full(k, np.nan)
            row[f"mean_{name}"] = mean_pl
            row[f"ssw_{name}"] = ssw_pl
            row[f"n_{name}_cells"] = n_pl
        diff = row["mean_malignant"] - row["mean_normal"]
        row["shift_z_cohort"] = diff / sd_safe
        if n_mal >= 2 and n_norm >= 2:
            pooled = np.sqrt((row["ssw_malignant"] + row["ssw_normal"]) / (n_p - 2))
            row["cohens_d_within"] = _safe_div(diff, np.where(pooled < degenerate_sd, np.nan, pooled))
        else:
            row["cohens_d_within"] = np.full(k, np.nan)
        row["eta2_normal"] = eta2["normal"]
        row["eta2_malignant"] = eta2["malignant"]
        row["eta2_contrib_normal"] = eta2_contrib["normal"][p]
        row["eta2_contrib_malignant"] = eta2_contrib["malignant"][p]
        for drop in ("mean_normal", "mean_malignant", "ssw_normal", "ssw_malignant", "n_normal_cells", "n_malignant_cells"):
            row.pop(drop)
        frames.append(pd.DataFrame(row))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["factor_index", "patient"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# univariate held-out AP with training-fold orientation
# --------------------------------------------------------------------------------------


def univariate_heldout_table(
    scores: np.ndarray,
    cohort: Cohort,
    patients: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per factor x held-out patient (LOPO label transfer on stored scores).

    Training cells are every cell outside the held-out patient (all 13 patients, as in
    F6). orientation = sign of the training-fold class-mean gap (mean malignant minus mean
    normal, ties -> +1). AP is invariant to the positive affine fold standardization, so
    raw scores give the same AP as fold z-scores.
    """
    x = np.asarray(scores, dtype=np.float64)
    pats = list(cohort.supported if patients is None else patients)
    y = cohort.malignant
    n_total, k = x.shape
    codes = {p: i for i, p in enumerate(cohort.patients)}
    pcode = np.array([codes[p] for p in cohort.patient], dtype=np.int32)
    records = []
    for j in range(k):
        col = x[:, j]
        order = np.argsort(-col, kind="stable")
        xs = col[order]
        ys = y[order]
        ps = pcode[order]
        for p in pats:
            held = ps == codes[p]
            train = ~held
            xt, yt = xs[train], ys[train]
            xh, yh = xs[held], ys[held]
            n_pos_t = int(yt.sum())
            mu1 = xt[yt].mean() if n_pos_t else np.nan
            mu0 = xt[~yt].mean() if n_pos_t < xt.size else np.nan
            sd_t = xt.std()
            gap = (mu1 - mu0) / sd_t if sd_t > 0 else 0.0
            orient = 1 if not np.isfinite(gap) or gap >= 0 else -1
            tr_pos = _ap_desc(xt, yt)
            tr_neg = _ap_desc(xt[::-1], yt[::-1])
            he_pos = _ap_desc(xh, yh)
            he_neg = _ap_desc(xh[::-1], yh[::-1])
            prev = float(yh.mean()) if yh.size else np.nan
            he_or = he_pos if orient > 0 else he_neg
            he_best = max(he_pos, he_neg)
            records.append(
                {
                    "factor_index": j + 1,
                    "patient": p,
                    "train_gap_z": float(gap),
                    "orientation": orient,
                    "train_ap_orientation": 1 if tr_pos >= tr_neg else -1,
                    "train_ap_oriented": tr_pos if orient > 0 else tr_neg,
                    "train_ap_sign_agnostic": max(tr_pos, tr_neg),
                    "heldout_ap_oriented": he_or,
                    "heldout_lift_oriented": he_or - prev,
                    "heldout_ap_sign_agnostic": he_best,
                    "heldout_lift_sign_agnostic": he_best - prev,
                    "heldout_best_orientation": 1 if he_pos >= he_neg else -1,
                    "heldout_prevalence": prev,
                }
            )
    out = pd.DataFrame.from_records(records)
    out["orientation_matches_heldout"] = out["orientation"].eq(out["heldout_best_orientation"]) | np.isclose(
        out["heldout_ap_oriented"], out["heldout_ap_sign_agnostic"], rtol=0, atol=1e-12
    )
    return out


def factor_patient_table(
    scores: np.ndarray,
    cohort: Cohort,
    *,
    with_univariate: bool = True,
    degenerate_sd: float = DEGENERATE_SD,
    normal_min_cells: int = NORMAL_MIN_CELLS,
    malignant_min_cells: int = MALIGNANT_MIN_CELLS,
) -> pd.DataFrame:
    """Allocation table joined with the univariate held-out table (supported rows only)."""
    alloc = allocation_table(
        scores,
        cohort,
        degenerate_sd=degenerate_sd,
        normal_min_cells=normal_min_cells,
        malignant_min_cells=malignant_min_cells,
    )
    if not with_univariate or not cohort.supported:
        return alloc
    uni = univariate_heldout_table(scores, cohort)
    return alloc.merge(uni, on=["factor_index", "patient"], how="left", validate="one_to_one")


# --------------------------------------------------------------------------------------
# summaries over the long table (any number of representations)
# --------------------------------------------------------------------------------------

CARRY_COLUMNS = ("short_panel_label", "n_genes", "effective_k")


def _sum(series: pd.Series) -> float:
    return series.sum(min_count=1)


def _top(frame: pd.DataFrame, value: str, keys: Sequence[str] = FACTOR_KEYS) -> pd.DataFrame:
    """Row of the patient with the largest ``value`` per key group (NaNs ignored)."""
    sub = frame.loc[frame[value].notna()]
    if sub.empty:
        return pd.DataFrame(columns=list(keys) + ["patient", value])
    idx = sub.groupby(list(keys), sort=False)[value].idxmax()
    return sub.loc[idx, list(keys) + ["patient", value]]


def factor_summary(long: pd.DataFrame, high_prevalence: Iterable[str] = HIGH_PREVALENCE) -> pd.DataFrame:
    """One row per representation x factor.

    Columns: degenerate flag, factor SS, eta2 by label, malignant cells' SS share, the
    P02/P03/P04 combined SS share / cell share / enrichment (all cells, normal cells only,
    malignant cells only), the top-enrichment patient (all patients, supported only,
    normal cells only) and the oriented univariate lift summary over supported patients.
    """
    keys = FACTOR_KEYS
    work = long.assign(_hp=long["patient"].isin(set(high_prevalence)))
    g = work.groupby(keys, sort=False)
    out = g.agg(
        degenerate=("degenerate", "first"),
        factor_sd=("factor_sd", "first"),
        factor_ss=("ss", _sum),
        factor_ss_normal=("ss_normal", _sum),
        factor_ss_malignant=("ss_malignant", _sum),
        n_cells=("n_cells", "sum"),
        n_normal=("n_normal", "sum"),
        n_malignant=("n_malignant", "sum"),
        eta2_normal=("eta2_normal", "first"),
        eta2_malignant=("eta2_malignant", "first"),
    )
    for col in CARRY_COLUMNS:
        if col in work.columns:
            out[col] = g[col].first()
    out["malignant_cell_share"] = out["n_malignant"] / out["n_cells"]
    out["malignant_ss_share"] = out["factor_ss_malignant"] / out["factor_ss"]

    hp = work.loc[work["_hp"]].groupby(keys, sort=False).agg(
        hp_ss=("ss", _sum),
        hp_ss_normal=("ss_normal", _sum),
        hp_ss_malignant=("ss_malignant", _sum),
        hp_n=("n_cells", "sum"),
        hp_n_normal=("n_normal", "sum"),
        hp_n_malignant=("n_malignant", "sum"),
    )
    out = out.join(hp)
    out["hp_cell_share"] = out["hp_n"] / out["n_cells"]
    out["hp_ss_share"] = out["hp_ss"] / out["factor_ss"]
    out["hp_enrichment"] = out["hp_ss_share"] / out["hp_cell_share"]
    out["hp_cell_share_normal"] = out["hp_n_normal"] / out["n_normal"]
    out["hp_ss_share_normal"] = out["hp_ss_normal"] / out["factor_ss_normal"]
    out["hp_enrichment_normal"] = out["hp_ss_share_normal"] / out["hp_cell_share_normal"]
    out["hp_cell_share_malignant"] = out["hp_n_malignant"] / out["n_malignant"]
    out["hp_ss_share_malignant"] = out["hp_ss_malignant"] / out["factor_ss_malignant"]
    out["hp_enrichment_malignant"] = out["hp_ss_share_malignant"] / out["hp_cell_share_malignant"]
    out = out.drop(columns=["hp_ss", "hp_ss_normal", "hp_ss_malignant", "hp_n", "hp_n_normal", "hp_n_malignant"])

    for label, frame, value in (
        ("top", work, "enrichment"),
        ("top_supported", work.loc[work["supported"]], "enrichment"),
        ("top_normal", work, "enrichment_normal"),
    ):
        top = _top(frame, value).set_index(keys)
        out[f"{label}_patient"] = top["patient"]
        out[f"{label}_enrichment"] = top[value]

    if "heldout_lift_oriented" in work.columns:
        sup = work.loc[work["supported"] & work["heldout_lift_oriented"].notna()]
        gl = sup.groupby(keys, sort=False)["heldout_lift_oriented"]
        out["median_lift_oriented"] = gl.median()
        out["max_lift_oriented"] = gl.max()
        best = _top(sup, "heldout_lift_oriented").set_index(keys)
        out["best_lift_patient"] = best["patient"]
        out["n_supported_lift_gt_0p05"] = (
            sup.assign(_g=sup["heldout_lift_oriented"] > 0.05).groupby(keys, sort=False)["_g"].sum()
        )
    out = out.reset_index()
    rep_ss = out.groupby(REP_KEYS, sort=False)["factor_ss"].transform(_sum)
    out["factor_ss_share_in_rep"] = out["factor_ss"] / rep_ss
    return out


def representation_patient_summary(long: pd.DataFrame) -> pd.DataFrame:
    """One row per representation x patient over non-degenerate factors.

    weighted_ss_share = sum_f SS_pf / sum_f SS_f (the patient's share of the
    representation's total score-space energy); weighted_enrichment divides it by the
    patient's cell share. The _normal columns use normal cells only.
    """
    keys = REP_KEYS + ["patient"]
    nd = long.loc[~long["degenerate"]]
    g = nd.groupby(keys, sort=False)
    out = g.agg(
        n_cells=("n_cells", "first"),
        n_normal=("n_normal", "first"),
        n_malignant=("n_malignant", "first"),
        cell_share=("cell_share", "first"),
        cell_share_normal=("cell_share_normal", "first"),
        supported=("supported", "first"),
        high_prevalence=("high_prevalence", "first"),
        ss=("ss", "sum"),
        ss_normal=("ss_normal", "sum"),
        median_factor_enrichment=("enrichment", "median"),
        max_factor_enrichment=("enrichment", "max"),
        median_factor_enrichment_normal=("enrichment_normal", "median"),
    ).reset_index()
    rep = out.groupby(REP_KEYS, sort=False)
    out["weighted_ss_share"] = out["ss"] / rep["ss"].transform("sum")
    out["weighted_enrichment"] = out["weighted_ss_share"] / out["cell_share"]
    out["weighted_ss_share_normal"] = out["ss_normal"] / rep["ss_normal"].transform("sum")
    out["weighted_enrichment_normal"] = out["weighted_ss_share_normal"] / out["cell_share_normal"]
    top = _top(nd, "enrichment")
    counts = top.groupby(keys, sort=False).size().rename("n_factors_top").reset_index()
    out = out.merge(counts, on=keys, how="left")
    out["n_factors_top"] = out["n_factors_top"].fillna(0).astype(int)
    n_nd = nd.groupby(REP_KEYS, sort=False)["factor_index"].nunique().rename("n_nondegenerate").reset_index()
    out = out.merge(n_nd, on=REP_KEYS, how="left")
    out["frac_factors_top"] = out["n_factors_top"] / out["n_nondegenerate"]
    return out


def representation_summary(
    fsum: pd.DataFrame,
    rps: pd.DataFrame,
    high_prevalence: Iterable[str] = HIGH_PREVALENCE,
) -> pd.DataFrame:
    """One row per representation: top-patient counts and P02/P03/P04 combined allocation."""
    hp = sorted(set(high_prevalence))
    rows = []
    rps_by_rep = {key: grp for key, grp in rps.groupby(REP_KEYS, sort=False)}
    for key, rep in fsum.groupby(REP_KEYS, sort=False):
        nd = rep.loc[~rep["degenerate"]]
        rec = dict(zip(REP_KEYS, key))
        for col in CARRY_COLUMNS:
            if col in rep.columns:
                rec[col] = rep[col].iloc[0]
        rec["n_factors"] = int(rep.shape[0])
        rec["n_nondegenerate"] = int(nd.shape[0])
        rec["n_degenerate"] = int(rep["degenerate"].sum())
        counts = nd["top_patient"].value_counts()
        rec["modal_top_patient"] = counts.index[0] if len(counts) else None
        rec["modal_top_count"] = int(counts.iloc[0]) if len(counts) else 0
        rec["modal_top_frac"] = rec["modal_top_count"] / rec["n_nondegenerate"] if rec["n_nondegenerate"] else np.nan
        ties = sorted(counts.loc[counts == counts.iloc[0]].index) if len(counts) else []
        rec["modal_top_tied_with"] = ",".join(ties) if len(ties) > 1 else ""
        rec["top_counts"] = ";".join(f"{p}:{int(c)}" for p, c in counts.items())
        for p in hp:
            rec[f"n_top_{p}"] = int(counts.get(p, 0))
        rec["n_top_hp_any"] = int(sum(counts.get(p, 0) for p in hp))
        sup_counts = nd["top_supported_patient"].value_counts()
        rec["modal_top_supported_patient"] = sup_counts.index[0] if len(sup_counts) else None
        rec["modal_top_supported_count"] = int(sup_counts.iloc[0]) if len(sup_counts) else 0
        norm_counts = nd["top_normal_patient"].value_counts()
        rec["modal_top_normal_patient"] = norm_counts.index[0] if len(norm_counts) else None
        rec["modal_top_normal_count"] = int(norm_counts.iloc[0]) if len(norm_counts) else 0

        prow = rps_by_rep.get(key)
        if prow is not None and len(prow):
            hp_rows = prow.loc[prow["patient"].isin(hp)]
            rec["hp_cell_share"] = float(hp_rows["n_cells"].sum() / prow["n_cells"].sum())
            rec["hp_ss_share_weighted"] = float(hp_rows["ss"].sum() / prow["ss"].sum())
            rec["hp_enrichment_weighted"] = rec["hp_ss_share_weighted"] / rec["hp_cell_share"]
            rec["hp_cell_share_normal"] = float(hp_rows["n_normal"].sum() / prow["n_normal"].sum())
            rec["hp_ss_share_normal_weighted"] = float(hp_rows["ss_normal"].sum() / prow["ss_normal"].sum())
            rec["hp_enrichment_normal_weighted"] = rec["hp_ss_share_normal_weighted"] / rec["hp_cell_share_normal"]
        w = nd["factor_ss"]
        rec["malignant_cell_share"] = float(nd["malignant_cell_share"].iloc[0]) if len(nd) else np.nan
        rec["malignant_ss_share_weighted"] = (
            float(nd["factor_ss_malignant"].sum() / w.sum()) if len(nd) and w.sum() > 0 else np.nan
        )
        rec["hp_median_factor_enrichment"] = float(nd["hp_enrichment"].median()) if len(nd) else np.nan
        rec["hp_median_factor_enrichment_normal"] = float(nd["hp_enrichment_normal"].median()) if len(nd) else np.nan
        rec["hp_frac_factors_enriched"] = float((nd["hp_enrichment"] > 1).mean()) if len(nd) else np.nan
        rec["hp_frac_factors_enriched_normal"] = float((nd["hp_enrichment_normal"] > 1).mean()) if len(nd) else np.nan
        if len(nd):
            i = nd["top_enrichment"].idxmax()
            rec["max_factor_enrichment"] = float(nd.loc[i, "top_enrichment"])
            rec["max_enrichment_factor"] = int(nd.loc[i, "factor_index"])
            rec["max_enrichment_patient"] = nd.loc[i, "top_patient"]
            rec["max_eta2_normal"] = float(nd["eta2_normal"].max())
            rec["max_eta2_malignant"] = float(nd["eta2_malignant"].max())
            if "max_lift_oriented" in nd.columns:
                j = nd["max_lift_oriented"].idxmax()
                rec["max_univariate_lift_oriented"] = float(nd.loc[j, "max_lift_oriented"])
                rec["max_univariate_lift_factor"] = int(nd.loc[j, "factor_index"])
                rec["max_univariate_lift_patient"] = nd.loc[j, "best_lift_patient"]
        rows.append(rec)
    return pd.DataFrame(rows)


def top_patient_frequency(fsum: pd.DataFrame, rsum: pd.DataFrame, patients: Sequence[str]) -> pd.DataFrame:
    """Per method x K x patient, pooled over representations (non-degenerate factors only).

    frac_factors_top: share of factors whose top-enrichment patient is this patient (all
    cells); _normal_only: the same with normal-cell enrichment; n_representations_modal_top:
    representations whose most frequent top patient is this patient (ties broken by
    pandas value_counts order; see modal_top_tied_with in the representation summary).
    """
    nd = fsum.loc[~fsum["degenerate"]]
    rows = []
    for (method, k), grp in nd.groupby(["stage1_method", "k"], sort=True):
        reps = rsum.loc[(rsum["stage1_method"] == method) & (rsum["k"] == k)]
        n_f = int(grp.shape[0])
        counts = grp["top_patient"].value_counts()
        normal_counts = grp["top_normal_patient"].value_counts()
        modal = reps["modal_top_patient"].value_counts()
        for p in patients:
            rows.append(
                {
                    "stage1_method": method,
                    "k": int(k),
                    "patient": p,
                    "n_representations": int(reps.shape[0]),
                    "n_nondegenerate_factors": n_f,
                    "n_factors_top": int(counts.get(p, 0)),
                    "frac_factors_top": float(counts.get(p, 0)) / n_f if n_f else np.nan,
                    "n_factors_top_normal_only": int(normal_counts.get(p, 0)),
                    "frac_factors_top_normal_only": float(normal_counts.get(p, 0)) / n_f if n_f else np.nan,
                    "n_representations_modal_top": int(modal.get(p, 0)),
                }
            )
    return pd.DataFrame(rows)
