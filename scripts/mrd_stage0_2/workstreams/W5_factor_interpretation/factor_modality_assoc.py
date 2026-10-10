"""D27 (W5.3) factor x modality associations for every D18 factor card.

For each factor of a stored Stage 1 representation (the 56 representations of the D18
factor_cards_v1 bundle, about 1,810 cards; degenerate cards skipped) this module computes:

1. Surface protein (CITE-seq ADT): within-label Spearman rho between the factor score and each
   ADT feature after a per-cell CLR (log1p(count) minus the cell's mean log1p over all 170 ADT
   features, as in D18). Isotype controls are kept as rows but never ranked as markers: their rho
   is the noise reference. rho is computed
   - pooled over every cell of the label (all patients mixed; the D18 definition),
   - within each patient (one stratum per label x patient with enough cells), summarised over the
     supported patients as the median within-patient rho and the number of supported patients
     whose rho has the same sign as that median,
   - within each supported patient x predicted cell type stratum (a cell-type-adjusted view),
   - and, for the D18 regression only, as the cell-weighted mean of the within-patient rhos over
     the D18 strata (normal >= 200 cells, malignant >= 20 cells, every patient).
2. Predicted cell type: eta^2 of the score by ``predicted.annotation`` within label, pooled and
   within each supported patient (median reported), plus per-cell-type mean oriented scores,
   pooled and per patient.

Orientation: every rho and mean is multiplied by the D18 orientation_sign (sign of the
globally selected LOPO probe's beta; positive = toward malignant). Unoriented cards
(orientation_sign 0) keep the raw, arbitrary sign and carry ``oriented = False``.

Spearman is computed in a vectorized way: rank-transform each column within the stratum
(average ranks for ties), standardize the ranks to mean 0 and population SD 1, then one matrix
product per stratum gives every (marker, factor) rho at once.

Pure numpy / pandas / scipy. Nothing here reads experiment paths; the runner is
run_d27_factor_modality_assoc.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

SLUG = "factor_modality_assoc_v1"
ANALYSIS_ID = "D27"
TASK = "W5.3"
SCHEMA_VERSION = 1
LABELS: Tuple[Tuple[str, int], ...] = (("normal", 0), ("malignant", 1))

# Strata (analysis parameters, fixed before computing; not the support rule).
PATIENT_MIN_CELLS = 20  # label x patient strata; every supported patient passes in both labels
D18_STRATUM_MIN = {"normal": 200, "malignant": 20}  # D18 patient-stratified weighted mean (regression)
PCT_MIN_CELLS = 30  # supported patient x predicted cell type strata (cell-type-adjusted rho)
CELLTYPE_MIN_CELLS_POOLED = 20  # pooled cell-type means and eta2 (= D18)
CELLTYPE_MIN_CELLS_PATIENT = 10  # within-patient cell-type eta2 and per-patient cell-type means

# Robust-association rule (pre-declared): |median within-patient rho| >= ROBUST_ABS_RHO and the
# same sign in at least ROBUST_MIN_SAME_SIGN supported patients; isotypes never qualify.
ROBUST_ABS_RHO = 0.20
ROBUST_MIN_SAME_SIGN = 5
SENSITIVITY_ABS_RHO = (0.10, 0.20, 0.30)
TOP_N_MARKERS = 10

FLOAT_FORMAT = "%.6g"


# --------------------------------------------------------------------------------------
# ranks and Spearman
# --------------------------------------------------------------------------------------


def clr_per_cell(counts: np.ndarray) -> np.ndarray:
    """Centered log-ratio per cell over all ADT features (identical to factor_cards.clr_per_cell)."""
    logged = np.log1p(np.asarray(counts, dtype=np.float64))
    return logged - logged.mean(axis=1, keepdims=True)


def is_isotype_name(name: str) -> bool:
    return str(name).lower().startswith("isotype")


def marker_display(name: str) -> str:
    """'ADT-CD34' -> 'CD34'; isotypes keep their full name."""
    name = str(name)
    return name[4:] if name.startswith("ADT-") else name


def rank_standardize(matrix: np.ndarray) -> np.ndarray:
    """Average ranks per column, centred and scaled to population SD 1.

    Constant columns (no rank variance) become all-NaN, so their correlations are NaN.
    """
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    ranks = rankdata(x, axis=0, method="average")
    mean = ranks.mean(axis=0)
    sd = ranks.std(axis=0)
    out = (ranks - mean) / np.where(sd > 0, sd, 1.0)
    out[:, sd <= 0] = np.nan
    return out


def spearman_from_ranks(ranks_a: np.ndarray, ranks_b: np.ndarray) -> np.ndarray:
    """(m_a, m_b) Spearman rho from two rank_standardize outputs over the same rows."""
    if ranks_a.shape[0] != ranks_b.shape[0]:
        raise ValueError("rank matrices must have the same rows")
    n = ranks_a.shape[0]
    if n == 0:
        return np.full((ranks_a.shape[1], ranks_b.shape[1]), np.nan)
    out = (np.nan_to_num(ranks_a).T @ np.nan_to_num(ranks_b)) / n
    out[np.isnan(ranks_a).all(axis=0), :] = np.nan
    out[:, np.isnan(ranks_b).all(axis=0)] = np.nan
    return np.clip(out, -1.0, 1.0)


# --------------------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------------------


@dataclass
class Stratum:
    key: str  # "<label>|pooled" | "<label>|<patient>" | "<label>|<patient>|<cell type>"
    label: str
    kind: str  # pooled | patient | patient_celltype
    patient: str
    cell_type: str
    rows: np.ndarray
    supported: bool

    @property
    def n_cells(self) -> int:
        return int(self.rows.size)


def build_strata(
    patients: np.ndarray,
    labels: np.ndarray,
    cell_types: Optional[np.ndarray],
    supported: Sequence[str],
    patient_min: int = PATIENT_MIN_CELLS,
    pct_min: int = PCT_MIN_CELLS,
) -> List[Stratum]:
    """Pooled, per-patient (every patient with >= patient_min cells of the label) and supported
    patient x cell type (>= pct_min cells) strata, per label."""
    patients = np.asarray(patients).astype(str)
    labels = np.asarray(labels).astype(int)
    supported = tuple(supported)
    out: List[Stratum] = []
    for label_name, label_value in LABELS:
        lrows = np.flatnonzero(labels == label_value)
        if lrows.size == 0:
            continue
        out.append(Stratum("{}|pooled".format(label_name), label_name, "pooled", "", "", lrows, False))
        for patient in sorted(pd.unique(patients[lrows])):
            prow = lrows[patients[lrows] == patient]
            if prow.size >= patient_min:
                out.append(Stratum("{}|{}".format(label_name, patient), label_name, "patient", patient, "", prow, patient in supported))
            if cell_types is None or patient not in supported:
                continue
            ct = np.asarray(cell_types).astype(str)[prow]
            for ctype in sorted(pd.unique(ct)):
                crow = prow[ct == ctype]
                if crow.size >= pct_min:
                    out.append(Stratum("{}|{}|{}".format(label_name, patient, ctype), label_name, "patient_celltype", patient, ctype, crow, True))
    return out


def strata_table(strata: Sequence[Stratum], cell_types: Optional[np.ndarray] = None, batches: Optional[np.ndarray] = None) -> pd.DataFrame:
    """One row per stratum: population, size, cell-type count and the eta2 null expectation."""
    rows = []
    ct = np.asarray(cell_types).astype(str) if cell_types is not None else None
    bt = np.asarray(batches).astype(str) if batches is not None else None
    for s in strata:
        row = {
            "stratum": s.key,
            "label": s.label,
            "kind": s.kind,
            "patient": s.patient,
            "cell_type": s.cell_type,
            "supported_patient": s.supported,
            "n_cells": s.n_cells,
        }
        if ct is not None and s.kind != "patient_celltype":
            min_cells = CELLTYPE_MIN_CELLS_POOLED if s.kind == "pooled" else CELLTYPE_MIN_CELLS_PATIENT
            counts = pd.Series(ct[s.rows]).value_counts()
            kept = counts[counts >= min_cells]
            row["n_cell_types_eta2"] = int(kept.size)
            row["n_cells_eta2"] = int(kept.sum())
            row["eta2_null_expectation"] = (kept.size - 1) / (kept.sum() - 1) if kept.size >= 2 and kept.sum() > 1 else np.nan
        if bt is not None:
            row["batches"] = ";".join(sorted(pd.unique(bt[s.rows])))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# ADT ranks (precomputed once per cohort)
# --------------------------------------------------------------------------------------


@dataclass
class AdtStrata:
    names: List[str]
    is_isotype: np.ndarray
    strata: List[Stratum]
    ranks: Dict[str, np.ndarray]  # stratum key -> (n_s, m) standardized ranks

    @property
    def n_markers(self) -> int:
        return len(self.names)


def build_adt_strata(adt_clr: np.ndarray, names: Sequence[str], strata: Sequence[Stratum]) -> AdtStrata:
    names = [str(n) for n in names]
    adt_clr = np.asarray(adt_clr, dtype=np.float64)
    if adt_clr.shape[1] != len(names):
        raise ValueError("ADT matrix has {} columns but {} names".format(adt_clr.shape[1], len(names)))
    iso = np.array([is_isotype_name(n) for n in names], dtype=bool)
    ranks = {s.key: rank_standardize(adt_clr[s.rows]) for s in strata}
    return AdtStrata(names, iso, list(strata), ranks)


# --------------------------------------------------------------------------------------
# per-representation ADT association
# --------------------------------------------------------------------------------------


def adt_rho_by_stratum(scores: np.ndarray, adt: AdtStrata) -> Dict[str, np.ndarray]:
    """Raw-sign Spearman rho (m_markers, K) for every stratum."""
    scores = np.asarray(scores, dtype=np.float64)
    return {s.key: spearman_from_ranks(adt.ranks[s.key], rank_standardize(scores[s.rows])) for s in adt.strata}


def same_sign_count(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Count, along axis 0, the finite values whose sign equals sign(reference) (reference != 0)."""
    ref_sign = np.sign(reference)
    agree = (np.sign(values) == ref_sign[None, ...]) & np.isfinite(values) & (ref_sign[None, ...] != 0)
    return agree.sum(axis=0)


def _nan_stat(fn, stack: np.ndarray) -> np.ndarray:
    valid = np.isfinite(stack).any(axis=0)
    out = np.full(stack.shape[1:], np.nan)
    if valid.any():
        out[valid] = fn(stack[:, valid], axis=0)
    return out


def summarize_rho(
    rho: Mapping[str, np.ndarray],
    strata: Sequence[Stratum],
    label: str,
    supported: Sequence[str],
    all_patients: Sequence[str],
    d18_min: Mapping[str, int] = D18_STRATUM_MIN,
) -> Dict[str, np.ndarray]:
    """Summaries (each (m, K)) of the raw-sign rho for one label."""
    lab = [s for s in strata if s.label == label]
    pooled = next((rho[s.key] for s in lab if s.kind == "pooled"), None)
    if pooled is None:
        raise ValueError("no pooled stratum for label {}".format(label))
    shape = pooled.shape
    per_patient = {s.patient: rho[s.key] for s in lab if s.kind == "patient"}
    sup = [p for p in supported if p in per_patient]
    if sup:
        stack = np.stack([per_patient[p] for p in sup])
        median = _nan_stat(np.nanmedian, stack)
        rmin = _nan_stat(np.nanmin, stack)
        rmax = _nan_stat(np.nanmax, stack)
        n_valid = np.isfinite(stack).sum(axis=0)
        n_same = same_sign_count(stack, median)
    else:
        median = rmin = rmax = np.full(shape, np.nan)
        n_valid = n_same = np.zeros(shape, dtype=int)
    # D18 patient-stratified: cell-weighted mean over every patient stratum meeting the D18 minimum
    acc = np.zeros(shape)
    weight = np.zeros(shape)
    for s in lab:
        if s.kind != "patient" or s.n_cells < int(d18_min.get(label, 1)):
            continue
        r = rho[s.key]
        ok = np.isfinite(r)
        acc += np.where(ok, r * s.n_cells, 0.0)
        weight += np.where(ok, s.n_cells, 0.0)
    wmean = np.where(weight > 0, acc / np.where(weight > 0, weight, 1.0), np.nan)
    # supported patient x cell type strata
    pct = [rho[s.key] for s in lab if s.kind == "patient_celltype"]
    if pct:
        pstack = np.stack(pct)
        pct_median = _nan_stat(np.nanmedian, pstack)
        pct_n = np.isfinite(pstack).sum(axis=0)
        pct_same = same_sign_count(pstack, median)
    else:
        pct_median = np.full(shape, np.nan)
        pct_n = pct_same = np.zeros(shape, dtype=int)
    out = {
        "rho_pooled": pooled,
        "rho_within_median": median,
        "rho_within_min": rmin,
        "rho_within_max": rmax,
        "n_patients_valid": n_valid,
        "n_patients_same_sign": n_same,
        "rho_within_wmean_d18": wmean,
        "rho_pct_median": pct_median,
        "n_pct_strata": pct_n,
        "n_pct_same_sign": pct_same,
    }
    for p in all_patients:
        out["rho_{}".format(p)] = per_patient.get(p, np.full(shape, np.nan))
    return out


SIGNED_COLUMNS_PREFIX = ("rho_",)


def adt_long_table(
    rho: Mapping[str, np.ndarray],
    adt: AdtStrata,
    card_ids: Sequence[str],
    orientation: np.ndarray,
    keep: np.ndarray,
    supported: Sequence[str],
    all_patients: Sequence[str],
) -> pd.DataFrame:
    """card x label x ADT feature rows (isotypes included, flagged), oriented by orientation_sign.

    Counts of same-sign patients do not depend on orientation (both the per-patient rho and the
    median flip together).
    """
    orientation = np.asarray(orientation, dtype=int)
    sign = np.where(orientation == 0, 1, orientation).astype(np.float64)
    keep = np.asarray(keep, dtype=bool)
    m = adt.n_markers
    kept = np.flatnonzero(keep)
    frames = []
    for label, _ in LABELS:
        if not any(s.label == label for s in adt.strata):
            continue
        summ = summarize_rho(rho, adt.strata, label, supported, all_patients)
        data: Dict[str, object] = {
            "card_id": np.repeat(np.asarray(card_ids, dtype=object)[kept], m),
            "label": label,
            "adt_marker": np.tile(np.asarray(adt.names, dtype=object), kept.size),
            "is_isotype": np.tile(adt.is_isotype, kept.size),
            "oriented": np.repeat(orientation[kept] != 0, m),
        }
        for col, values in summ.items():
            arr = np.asarray(values)[:, kept]  # (m, n_kept)
            if col.startswith(SIGNED_COLUMNS_PREFIX):
                arr = arr * sign[kept][None, :]
                if col in ("rho_within_min", "rho_within_max"):
                    continue  # handled below (min/max swap under a sign flip)
                data[col] = arr.T.reshape(-1).astype(np.float64)
            else:
                data[col] = arr.T.reshape(-1).astype(np.int64)
        lo = np.asarray(summ["rho_within_min"])[:, kept] * sign[kept][None, :]
        hi = np.asarray(summ["rho_within_max"])[:, kept] * sign[kept][None, :]
        data["rho_within_min"] = np.fmin(lo, hi).T.reshape(-1)
        data["rho_within_max"] = np.fmax(lo, hi).T.reshape(-1)
        frames.append(pd.DataFrame(data))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["adt_marker_short"] = out["adt_marker"].map(marker_display)
    return out


def robust_mask(frame: pd.DataFrame, abs_rho: float = ROBUST_ABS_RHO, min_same_sign: int = ROBUST_MIN_SAME_SIGN) -> pd.Series:
    """The pre-declared robust rule; isotypes never qualify."""
    med = pd.to_numeric(frame["rho_within_median"], errors="coerce")
    return (~frame["is_isotype"].astype(bool)) & (med.abs() >= abs_rho) & (pd.to_numeric(frame["n_patients_same_sign"]) >= min_same_sign)


# --------------------------------------------------------------------------------------
# cell type
# --------------------------------------------------------------------------------------


def _one_hot(groups: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    codes, names = pd.factorize(pd.Series(np.asarray(groups).astype(str)), sort=True)
    n = codes.size
    mat = sp.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(len(names), n))
    counts = np.bincount(codes, minlength=len(names))
    return mat, np.asarray(names, dtype=object), counts


def group_means(x: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(G, K) group means, group names (sorted) and counts."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    mat, names, counts = _one_hot(groups)
    sums = np.asarray(mat @ x)
    return sums / np.maximum(counts, 1)[:, None], names, counts


def eta2_by_group(x: np.ndarray, groups: np.ndarray, min_cells: int = 1) -> np.ndarray:
    """eta2 = between-group SS / total SS per column, over groups with >= min_cells cells.

    Equals factor_cards.eta2_groups (NaN when fewer than two groups qualify; 0 for a constant
    column)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    groups = np.asarray(groups).astype(str)
    counts = pd.Series(groups).value_counts()
    names = counts.index[counts >= min_cells]
    if len(names) < 2:
        return np.full(x.shape[1], np.nan)
    keep = np.isin(groups, np.asarray(names))
    xk = x[keep]
    means, _, cnt = group_means(xk, groups[keep])
    grand = xk.mean(axis=0)
    ss_total = ((xk - grand) ** 2).sum(axis=0)
    ss_between = (cnt[:, None] * (means - grand) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_total > 0, ss_between / ss_total, 0.0)


def oriented_cohort_z(scores: np.ndarray, orientation: np.ndarray, degenerate_sd: float = 1e-8) -> np.ndarray:
    """Cohort z-score (population SD over every cell) times orientation_sign (raw sign when 0)."""
    x = np.asarray(scores, dtype=np.float64)
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    ok = sd > degenerate_sd
    z = (x - mean) / np.where(ok, sd, 1.0)
    z[:, ~ok] = 0.0
    sign = np.where(np.asarray(orientation) == 0, 1, np.asarray(orientation))
    return z * sign[None, :]


def celltype_tables(
    scores: np.ndarray,
    patients: np.ndarray,
    labels: np.ndarray,
    cell_types: np.ndarray,
    card_ids: Sequence[str],
    orientation: np.ndarray,
    keep: np.ndarray,
    supported: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """eta2 (pooled and within supported patients) plus pooled and per-patient cell-type means."""
    scores = np.asarray(scores, dtype=np.float64)
    patients = np.asarray(patients).astype(str)
    labels = np.asarray(labels).astype(int)
    cell_types = np.asarray(cell_types).astype(str)
    orientation = np.asarray(orientation, dtype=int)
    keep = np.asarray(keep, dtype=bool)
    kept = np.flatnonzero(keep)
    cids = np.asarray(card_ids, dtype=object)
    z = oriented_cohort_z(scores, orientation)
    eta_rows = []
    mean_rows = []
    pmean_rows = []
    for label, value in LABELS:
        lmask = labels == value
        if not lmask.any():
            continue
        pooled = eta2_by_group(scores[lmask], cell_types[lmask], CELLTYPE_MIN_CELLS_POOLED)
        counts = pd.Series(cell_types[lmask]).value_counts()
        kept_types = counts[counts >= CELLTYPE_MIN_CELLS_POOLED]
        null_pooled = (kept_types.size - 1) / (kept_types.sum() - 1) if kept_types.size >= 2 else np.nan
        per_patient: Dict[str, np.ndarray] = {}
        nulls: Dict[str, float] = {}
        for p in supported:
            pm = lmask & (patients == p)
            if pm.sum() < 2:
                continue
            per_patient[p] = eta2_by_group(scores[pm], cell_types[pm], CELLTYPE_MIN_CELLS_PATIENT)
            pc = pd.Series(cell_types[pm]).value_counts()
            pk = pc[pc >= CELLTYPE_MIN_CELLS_PATIENT]
            nulls[p] = (pk.size - 1) / (pk.sum() - 1) if pk.size >= 2 else np.nan
        stack = np.stack([per_patient[p] for p in supported if p in per_patient]) if per_patient else np.full((1, scores.shape[1]), np.nan)
        within_median = _nan_stat(np.nanmedian, stack)
        n_valid = np.isfinite(stack).sum(axis=0)
        null_vals = [v for v in nulls.values() if np.isfinite(v)]
        null_median = float(np.median(null_vals)) if null_vals else np.nan
        frame = pd.DataFrame(
            {
                "card_id": cids[kept],
                "label": label,
                "eta2_celltype_pooled": pooled[kept],
                "eta2_null_pooled": null_pooled,
                "n_cell_types_pooled": int(kept_types.size),
                "eta2_celltype_within_median": within_median[kept],
                "eta2_null_within_median": null_median,
                "n_patients_valid": n_valid[kept].astype(int),
            }
        )
        for p in supported:
            frame["eta2_{}".format(p)] = per_patient[p][kept] if p in per_patient else np.nan
        eta_rows.append(frame)
        # pooled means (cell types with >= CELLTYPE_MIN_CELLS_POOLED cells; D18 definition)
        means, names, cnt = group_means(z[lmask], cell_types[lmask])
        for gi, name in enumerate(names):
            if cnt[gi] < CELLTYPE_MIN_CELLS_POOLED:
                continue
            mean_rows.append(pd.DataFrame({"card_id": cids[kept], "label": label, "cell_type": name, "n_cells": int(cnt[gi]), "mean_oriented_z": means[gi, kept]}))
        # per-patient means (every patient; cell types with >= CELLTYPE_MIN_CELLS_PATIENT cells)
        for p in sorted(pd.unique(patients[lmask])):
            pm = lmask & (patients == p)
            means, names, cnt = group_means(z[pm], cell_types[pm])
            pmeans = z[pm].mean(axis=0)
            for gi, name in enumerate(names):
                if cnt[gi] < CELLTYPE_MIN_CELLS_PATIENT:
                    continue
                pmean_rows.append(
                    pd.DataFrame(
                        {
                            "card_id": cids[kept],
                            "label": label,
                            "patient": p,
                            "supported_patient": p in supported,
                            "cell_type": name,
                            "n_cells": int(cnt[gi]),
                            "mean_oriented_z": means[gi, kept],
                            "patient_label_mean_oriented_z": pmeans[kept],
                        }
                    )
                )
    cat = lambda rows: pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()  # noqa: E731
    return {"celltype_eta2": cat(eta_rows), "celltype_means": cat(mean_rows), "celltype_means_by_patient": cat(pmean_rows)}


# --------------------------------------------------------------------------------------
# one representation
# --------------------------------------------------------------------------------------


def representation_tables(
    scores: np.ndarray,
    cards: pd.DataFrame,
    patients: np.ndarray,
    labels: np.ndarray,
    cell_types: Optional[np.ndarray],
    supported: Sequence[str],
    all_patients: Sequence[str],
    adt: Optional[AdtStrata],
) -> Dict[str, pd.DataFrame]:
    """Every D27 table for one representation.

    ``cards`` holds the D18 index rows of this representation (card_id, factor, orientation_sign,
    degenerate_flag); factor f is column f-1 of ``scores``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    k = scores.shape[1]
    cards = cards.sort_values("factor")
    if list(cards["factor"].astype(int)) != list(range(1, k + 1)):
        raise ValueError("cards do not cover factors 1..{} (got {})".format(k, list(cards["factor"])))
    card_ids = list(cards["card_id"].astype(str))
    orientation = cards["orientation_sign"].astype(int).to_numpy()
    sd = scores.std(axis=0)
    keep = (~cards["degenerate_flag"].astype(bool).to_numpy()) & (sd > 1e-8)
    out: Dict[str, pd.DataFrame] = {}
    if adt is not None:
        rho = adt_rho_by_stratum(scores, adt)
        out["adt"] = adt_long_table(rho, adt, card_ids, orientation, keep, supported, all_patients)
    else:
        out["adt"] = pd.DataFrame()
    if cell_types is not None:
        out.update(celltype_tables(scores, patients, labels, cell_types, card_ids, orientation, keep, supported))
    else:
        out.update({"celltype_eta2": pd.DataFrame(), "celltype_means": pd.DataFrame(), "celltype_means_by_patient": pd.DataFrame()})
    return out


PART_TABLES = ("adt", "celltype_eta2", "celltype_means", "celltype_means_by_patient")


# --------------------------------------------------------------------------------------
# merge-time summaries
# --------------------------------------------------------------------------------------


def top_markers(adt: pd.DataFrame, n: int = TOP_N_MARKERS, by: str = "rho_within_median") -> pd.DataFrame:
    """Top-n non-isotype markers per card x label by |by| (ties broken by marker name)."""
    frame = adt.loc[~adt["is_isotype"].astype(bool)].copy()
    frame["_abs"] = pd.to_numeric(frame[by], errors="coerce").abs().fillna(-1.0)
    frame = frame.sort_values(["card_id", "label", "_abs", "adt_marker"], ascending=[True, True, False, True], kind="mergesort")
    frame["rank"] = frame.groupby(["card_id", "label"], sort=False).cumcount() + 1
    return frame.loc[frame["rank"] <= n].drop(columns="_abs")


def isotype_reference(adt: pd.DataFrame) -> pd.DataFrame:
    """Per card x label: the largest |rho| among isotypes (pooled and median within-patient) and
    among markers, the noise reference for that card."""
    frame = adt.copy()
    frame["abs_pooled"] = pd.to_numeric(frame["rho_pooled"], errors="coerce").abs()
    frame["abs_within"] = pd.to_numeric(frame["rho_within_median"], errors="coerce").abs()
    iso = frame.loc[frame["is_isotype"].astype(bool)].groupby(["card_id", "label"])[["abs_pooled", "abs_within"]].max()
    iso.columns = ["isotype_max_abs_rho_pooled", "isotype_max_abs_rho_within_median"]
    mk = frame.loc[~frame["is_isotype"].astype(bool)].groupby(["card_id", "label"])[["abs_pooled", "abs_within"]].max()
    mk.columns = ["marker_max_abs_rho_pooled", "marker_max_abs_rho_within_median"]
    return iso.join(mk, how="outer").reset_index()


def noise_floor_table(adt: pd.DataFrame, quantiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99, 1.0)) -> pd.DataFrame:
    """Quantiles of |rho| (pooled and median within-patient) over isotype rows and marker rows."""
    rows = []
    for label in [lab for lab, _ in LABELS]:
        sub = adt.loc[adt["label"] == label]
        for kind, part in (("isotype", sub.loc[sub["is_isotype"].astype(bool)]), ("marker", sub.loc[~sub["is_isotype"].astype(bool)])):
            for col in ("rho_pooled", "rho_within_median"):
                vals = pd.to_numeric(part[col], errors="coerce").abs().dropna().to_numpy()
                row = {"label": label, "feature_kind": kind, "statistic": "abs_" + col, "n_rows": int(vals.size)}
                for q in quantiles:
                    row["q{:g}".format(100 * q)] = float(np.quantile(vals, q)) if vals.size else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def robust_table(adt: pd.DataFrame, cards: pd.DataFrame, iso_ref: pd.DataFrame, supported: Sequence[str]) -> pd.DataFrame:
    """Ranked robust associations (pre-declared rule) with card context and noise references."""
    rob = adt.loc[robust_mask(adt)].copy()
    rob["abs_rho_within_median"] = rob["rho_within_median"].abs()
    ctx_cols = [c for c in ("card_id", "representation_id", "panel_label", "method", "k", "factor", "orientation_status", "is_selected_driver", "n_active_folds", "max_univariate_lift", "eta2_normal", "eta2_malignant") if c in cards.columns]
    rob = rob.merge(cards[ctx_cols], on="card_id", how="left")
    rob = rob.merge(iso_ref[["card_id", "label", "isotype_max_abs_rho_within_median", "isotype_max_abs_rho_pooled"]], on=["card_id", "label"], how="left")
    rob["exceeds_isotype_reference"] = rob["abs_rho_within_median"] > rob["isotype_max_abs_rho_within_median"].fillna(0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rob["pct_to_within_ratio"] = rob["rho_pct_median"] / rob["rho_within_median"]
        rob["within_to_pooled_ratio"] = rob["rho_within_median"] / rob["rho_pooled"]
    rob = rob.sort_values(["abs_rho_within_median", "n_patients_same_sign", "card_id", "label", "adt_marker"], ascending=[False, False, True, True, True], kind="mergesort").reset_index(drop=True)
    rob.insert(0, "rank", np.arange(1, len(rob) + 1))
    keep = ["rank", "card_id", "panel_label", "method", "k", "factor", "label", "adt_marker", "adt_marker_short",
            "rho_within_median", "n_patients_same_sign", "n_patients_valid", "rho_within_min", "rho_within_max", "rho_pooled",
            "within_to_pooled_ratio", "rho_pct_median", "n_pct_same_sign", "n_pct_strata", "pct_to_within_ratio"]
    keep += ["rho_{}".format(p) for p in supported]
    keep += ["isotype_max_abs_rho_within_median", "isotype_max_abs_rho_pooled", "exceeds_isotype_reference", "oriented",
             "orientation_status", "is_selected_driver", "n_active_folds", "max_univariate_lift", "eta2_normal", "eta2_malignant", "representation_id"]
    return rob[[c for c in keep if c in rob.columns]]


def card_summary(
    cards: pd.DataFrame,
    adt: pd.DataFrame,
    eta: pd.DataFrame,
    means: pd.DataFrame,
    iso_ref: pd.DataFrame,
) -> pd.DataFrame:
    """One row per non-degenerate card: the strongest within-patient ADT association per label,
    robust counts, isotype references and cell-type eta2 / extreme cell types per label."""
    base_cols = [c for c in ("card_id", "representation_id", "panel_id", "panel_label", "method", "k", "factor", "orientation_sign",
                             "orientation_status", "is_selected_driver", "n_active_folds", "max_univariate_lift", "eta2_normal",
                             "eta2_malignant", "eta2_celltype_normal", "caveat_flags") if c in cards.columns]
    out = cards.loc[~cards["degenerate_flag"].astype(bool), base_cols].copy()
    out = out.rename(columns={"eta2_normal": "eta2_patient_normal", "eta2_malignant": "eta2_patient_malignant", "eta2_celltype_normal": "d18_eta2_celltype_normal"})
    out["oriented"] = out["orientation_sign"].astype(int) != 0
    top = top_markers(adt, n=1)
    rob = adt.loc[robust_mask(adt), ["card_id", "label"]]
    for label, _ in LABELS:
        t = top.loc[top["label"] == label].set_index("card_id")
        out["adt_top_marker_{}".format(label)] = out["card_id"].map(t["adt_marker_short"])
        out["adt_top_rho_within_median_{}".format(label)] = out["card_id"].map(t["rho_within_median"])
        out["adt_top_n_same_sign_{}".format(label)] = out["card_id"].map(t["n_patients_same_sign"])
        out["adt_top_rho_pooled_{}".format(label)] = out["card_id"].map(t["rho_pooled"])
        n_rob = rob.loc[rob["label"] == label, "card_id"].value_counts()
        out["adt_n_robust_{}".format(label)] = out["card_id"].map(n_rob).fillna(0).astype(int)
        r = iso_ref.loc[iso_ref["label"] == label].set_index("card_id")
        out["isotype_max_abs_rho_pooled_{}".format(label)] = out["card_id"].map(r["isotype_max_abs_rho_pooled"])
        out["isotype_max_abs_rho_within_median_{}".format(label)] = out["card_id"].map(r["isotype_max_abs_rho_within_median"])
        e = eta.loc[eta["label"] == label].set_index("card_id")
        out["eta2_celltype_pooled_{}".format(label)] = out["card_id"].map(e["eta2_celltype_pooled"])
        out["eta2_celltype_within_median_{}".format(label)] = out["card_id"].map(e["eta2_celltype_within_median"])
        mm = means.loc[means["label"] == label]
        if not mm.empty:
            hi = mm.sort_values(["card_id", "mean_oriented_z"], ascending=[True, False]).drop_duplicates("card_id").set_index("card_id")
            lo = mm.sort_values(["card_id", "mean_oriented_z"], ascending=[True, True]).drop_duplicates("card_id").set_index("card_id")
            out["celltype_highest_{}".format(label)] = out["card_id"].map(hi["cell_type"])
            out["celltype_highest_mean_{}".format(label)] = out["card_id"].map(hi["mean_oriented_z"])
            out["celltype_lowest_{}".format(label)] = out["card_id"].map(lo["cell_type"])
            out["celltype_lowest_mean_{}".format(label)] = out["card_id"].map(lo["mean_oriented_z"])
    return out.reset_index(drop=True)


SCHEMA_MD = """# factor_modality_assoc_v1 (D27) table schema (schema_version 1)

Producer: `scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d27_factor_modality_assoc.py`
(D27, W5.3). Reads the D18 bundle `factor_cards_v1` (card ids, orientation, degenerate flags) and the
stored Stage 1 scores; nothing is refit. Location (EXP-relative):
`analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/`. Keys: `card_id` (D18 format
`<panel_id>__<method>__k<K>__f<NNN>`) and `label` (`normal` | `malignant`). A consumer (the W7.2
factor explorer) can join every table to `factor_cards_index.csv` on `card_id`. Degenerate cards
(D18 `degenerate_flag`) have no rows. Extra columns may be added; consumers ignore unknown columns.

Sign convention: every rho and every mean score is multiplied by the D18 `orientation_sign`
(positive = toward malignant). When `oriented` is False the card is unoriented in D18 and the
sign is the arbitrary raw Stage 1 sign. ADT = per-cell CLR of raw counts over all 170 features
(161 markers + 9 isotypes); no dsb. Spearman rho within label.

## Compact tables (CSV; sized for the factor explorer)

- `d27_card_summary.csv` (one row per non-degenerate card): card context from D18 (orientation,
  is_selected_driver, n_active_folds, max_univariate_lift, eta2_patient_normal/malignant), then per
  label L in {normal, malignant}: `adt_top_marker_L`, `adt_top_rho_within_median_L`,
  `adt_top_n_same_sign_L`, `adt_top_rho_pooled_L` (strongest marker by |median within-patient rho|),
  `adt_n_robust_L` (markers passing the robust rule), `isotype_max_abs_rho_pooled_L`,
  `isotype_max_abs_rho_within_median_L` (noise reference), `eta2_celltype_pooled_L`,
  `eta2_celltype_within_median_L`, `celltype_highest_L` / `celltype_highest_mean_L`,
  `celltype_lowest_L` / `celltype_lowest_mean_L` (pooled cell-type means of the oriented cohort-z score).
- `d27_adt_top.csv` (card x label x rank, top 10 non-isotype markers by |rho_within_median|):
  card_id, label, rank, adt_marker, adt_marker_short, rho_within_median, n_patients_same_sign,
  n_patients_valid, rho_within_min, rho_within_max, rho_pooled, rho_pct_median, n_pct_same_sign,
  n_pct_strata, oriented, rho_<patient> for the 7 supported patients.
- `d27_adt_robust.csv` (ranked): every non-isotype card x label x marker with |rho_within_median|
  >= 0.20 and the same sign in >= 5 supported patients, with card context, the card's isotype
  reference (`isotype_max_abs_rho_within_median`, `exceeds_isotype_reference`),
  `within_to_pooled_ratio` and `pct_to_within_ratio` (how much of the within-patient rho survives
  inside supported patient x cell type strata).
- `d27_celltype_eta2.csv` (card x label): eta2_celltype_pooled (cell types with >= 20 cells),
  eta2_null_pooled ((g-1)/(n-1), the expectation under no association), n_cell_types_pooled,
  eta2_celltype_within_median (median over supported patients; cell types with >= 10 cells in the
  patient), eta2_null_within_median, n_patients_valid, eta2_<patient> (7 supported patients).
- `d27_celltype_means.csv` (card x label x cell type, >= 20 cells; = D18 factor_cards_celltype.csv):
  n_cells, mean_oriented_z (cohort z over all 60,118 cells, times orientation_sign).
- `d27_ifna_pc8_fa7_adt.csv`: every ADT row of IFN-alpha PCA40 factor 8 and FA40 factor 7.
- `d27_adt_noise_floor.csv`: quantiles of |rho| over isotype rows vs marker rows, per label.
- `d27_strata.csv`: one row per stratum (label x pooled / patient / supported patient x cell type):
  n_cells, batches, n_cell_types_eta2, eta2_null_expectation.
- `d27_regression_checks.csv`: check_id, description, n_compared, max_abs_diff, tolerance, passed.

## Full tables (binary or gzip; for analysis, not for the app's first load)

- `d27_adt_assoc_full.parquet` (card x label x all 170 ADT features, isotypes flagged):
  card_id, label, adt_marker, adt_marker_short, is_isotype, oriented, rho_pooled, rho_within_median,
  rho_within_min, rho_within_max, n_patients_valid, n_patients_same_sign, rho_within_wmean_d18
  (cell-weighted mean of within-patient rho over patients with normal >= 200 / malignant >= 20 cells;
  = D18 spearman_patient_stratified), rho_pct_median, n_pct_strata, n_pct_same_sign (supported patient x
  cell type strata with >= 30 cells), rho_<patient> for all 13 patients (NaN when the patient has
  fewer than 20 cells of the label).
- `d27_celltype_means_by_patient.csv.gz` (card x label x patient x cell type with >= 10 cells):
  supported_patient, n_cells, mean_oriented_z, patient_label_mean_oriented_z.
"""
