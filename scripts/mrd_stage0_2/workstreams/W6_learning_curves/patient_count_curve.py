"""D15b patient-count learning curve: design, summaries and curve fits (W6.1).

Library for run_d15b_patient_count_curve.py. Pure numpy / pandas / scipy; no file I/O and no
experiment access, so tests can use synthetic inputs.

Design
- For each held-out supported patient h, training sets are S + U, where S is a subset of size k
  of the OTHER supported patients and U is every unsupported patient (always included).
  k = 6 (all other supported patients) is the ordinary LOPO fold, i.e. the D1 atlas / D15
  full-training point.
- 'full' design: every subset, C(6, k) per held-out patient.
- 'reduced' design: a deterministic cyclic (difference-set) family of 7 subsets per size over the
  sorted supported patients; held-out patient h uses the members that exclude h. Sizes 3 and 4
  are balanced incomplete block designs (every pair of patients co-occurs equally often). The
  reduced units are a subset of the full units, so a later full run resumes from them.
- Basis: the frozen transductive Stage 1 basis. Every subset's basis was fit on all cells,
  including h and the excluded patients. The curve is therefore a Stage 2 (label) learning
  curve only; it says nothing on its own about a basis learned from k patients.

Tuning depends only on the training set S + U (never on h), so the runner tunes once per
(representation, S, penalty) and reuses the result for every h outside S.

Summaries
- per (h, k): distribution of held-out lift over the subsets;
- per k: median / P20 / patient-equal mean over held-out patients of the per-(h, k) mean lift,
  with patient-cluster bootstrap CIs (resample held-out patients with replacement);
- paired gains k vs 1 per held-out patient, bootstrapped the same way;
- patient-inclusion effect: at matched k, mean lift of subsets that contain patient j minus
  those that do not;
- weighted inverse power law y(k) = a - b * k**(-c), validated by predicting the k = 5 summary
  from k <= 4 (and k = 6 from k <= 5). Exploratory; reported only when the checks pass.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "BASIS_LABEL",
    "ARMS",
    "ARM_LABELS",
    "CYCLIC_DIFFERENCE_SETS",
    "subset_tag",
    "parse_subset_tag",
    "enumerate_subsets",
    "cyclic_design_family",
    "reduced_subsets",
    "design_table",
    "training_mask",
    "select_tuned_penalty",
    "bootstrap_patient_stat",
    "summarize_by_patient_k",
    "curve_by_k",
    "paired_gain",
    "patient_inclusion_effect",
    "inverse_power_law",
    "fit_inverse_power_law",
    "validate_power_law",
    "power_law_table",
]

#: Every output row carries this label (AGENTS R6).
BASIS_LABEL = "frozen_transductive_stage1 (basis fit on all 60,118 cells incl. held-out and excluded patients)"

#: C policies ("arms"). 'tuned' is the headline (penalty and C chosen per training set).
ARMS = ("tuned", "tuned_l1", "tuned_l2", "fixed_atlas", "scaled_to_n")
ARM_LABELS = {
    "tuned": "tuned: penalty (L1/L2) and C by inner LOPO per training set",
    "tuned_l1": "L1, C by inner LOPO per training set",
    "tuned_l2": "L2, C by inner LOPO per training set",
    "fixed_atlas": "atlas penalty and C, fixed (D15-comparable; C selected with held-out outcomes, not nested)",
    "scaled_to_n": "atlas penalty, C scaled to keep the full-fold L1 entry threshold",
}

#: Base blocks of the cyclic reduced design (mod 7). {0,1,3} is the Fano plane (each pair once);
#: {0,1,2,4} is its complement design (each pair twice). Sizes 2 and 5 use adjacent pairs and
#: their complements; sizes 1 and 6 are complete. Sizes 1, 3, 4 and 6 are balanced over
#: training-patient membership for every held-out patient; sizes 2 and 5 are NOT: the two
#: patients next to the held-out one in sorted order enter 1/5 (size 2) or 1/2 (size 5) of its
#: subsets, against 1/3 and 5/6 in the full enumeration, so reduced-design k = 2 and k = 5 means
#: average over a skewed training composition (use the full design for those points).
CYCLIC_DIFFERENCE_SETS: Dict[int, Tuple[int, ...]] = {
    1: (0,),
    2: (0, 1),
    3: (0, 1, 3),
    4: (0, 1, 2, 4),
    5: (0, 1, 2, 3, 4),
    6: (0, 1, 2, 3, 4, 5),
}


# --------------------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------------------


def subset_tag(subset: Iterable[str]) -> str:
    """Canonical text id of a training subset: sorted patient ids joined by '+'."""
    items = sorted(str(p) for p in subset)
    if not items:
        raise ValueError("empty subset")
    return "+".join(items)


def parse_subset_tag(tag: str) -> Tuple[str, ...]:
    return tuple(sorted(str(tag).split("+")))


def enumerate_subsets(heldout: str, supported: Sequence[str], k: int) -> List[Tuple[str, ...]]:
    """Every size-k subset of the supported patients other than ``heldout`` (sorted tuples)."""
    others = sorted(str(p) for p in supported if str(p) != str(heldout))
    if str(heldout) not in {str(p) for p in supported}:
        raise ValueError(f"held-out patient {heldout!r} is not a supported patient")
    if not 1 <= int(k) <= len(others):
        raise ValueError(f"k must lie in 1..{len(others)}")
    return [tuple(c) for c in itertools.combinations(others, int(k))]


def cyclic_design_family(supported: Sequence[str], size: int) -> List[Tuple[str, ...]]:
    """The 7 cyclic translates of the base block for ``size`` over the sorted supported patients.

    Defined for exactly 7 supported patients (the study's support rule); raises otherwise.
    """
    pts = sorted(str(p) for p in supported)
    n = len(pts)
    if n != 7:
        raise ValueError("the cyclic reduced design is defined for exactly 7 supported patients")
    if int(size) not in CYCLIC_DIFFERENCE_SETS:
        raise ValueError(f"size must lie in {sorted(CYCLIC_DIFFERENCE_SETS)}")
    base = CYCLIC_DIFFERENCE_SETS[int(size)]
    family = {tuple(sorted(pts[(i + d) % n] for d in base)) for i in range(n)}
    return sorted(family)


def reduced_subsets(heldout: str, supported: Sequence[str], k: int) -> List[Tuple[str, ...]]:
    """Members of the cyclic family of size k that exclude ``heldout`` (7 - k of them)."""
    if str(heldout) not in {str(p) for p in supported}:
        raise ValueError(f"held-out patient {heldout!r} is not a supported patient")
    return [s for s in cyclic_design_family(supported, k) if str(heldout) not in s]


def design_table(
    supported: Sequence[str],
    heldouts: Optional[Sequence[str]] = None,
    ks: Sequence[int] = (1, 2, 3, 4, 5, 6),
    design: str = "full",
) -> pd.DataFrame:
    """One row per (heldout_patient, k, subset) of the requested design."""
    sup = sorted(str(p) for p in supported)
    hs = sup if heldouts is None else [str(h) for h in heldouts]
    rows = []
    for h in hs:
        for k in ks:
            if design == "full":
                subsets = enumerate_subsets(h, sup, int(k))
            elif design == "reduced":
                subsets = reduced_subsets(h, sup, int(k))
            else:
                raise ValueError("design must be 'full' or 'reduced'")
            for s in subsets:
                rows.append({"heldout_patient": h, "k": int(k), "subset": subset_tag(s), "n_subsets_full": math.comb(len(sup) - 1, int(k))})
    return pd.DataFrame(rows, columns=["heldout_patient", "k", "subset", "n_subsets_full"])


def training_mask(patients: Any, subset: Iterable[str], supported: Sequence[str], heldout: Optional[str] = None) -> np.ndarray:
    """Training cells: patients in ``subset`` plus every patient that is not supported.

    ``heldout`` (optional) is checked to be outside the subset; its cells are never included.
    """
    pat = np.asarray(patients).astype(str)
    sub = {str(p) for p in subset}
    sup = {str(p) for p in supported}
    if not sub <= sup:
        raise ValueError("subset must contain supported patients only")
    if heldout is not None and str(heldout) in sub:
        raise ValueError("held-out patient inside the training subset")
    mask = np.isin(pat, sorted(sub)) | ~np.isin(pat, sorted(sup))
    if heldout is not None:
        mask &= pat != str(heldout)
    return mask


def select_tuned_penalty(scores: Mapping[str, float], tie_atol: float = 1e-12, order: Sequence[str] = ("l1", "l2")) -> str:
    """Penalty with the highest inner score (median inner lift); ties go to the earlier entry
    of ``order`` (L1 first: the sparser probe). NaN scores never win."""
    finite = {p: float(scores[p]) for p in order if p in scores and np.isfinite(float(scores[p]))}
    if not finite:
        raise ValueError("no finite inner score")
    top = max(finite.values())
    for p in order:
        if p in finite and abs(finite[p] - top) <= tie_atol:
            return p
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------------------


def _p20(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, 0.2)) if arr.size else float("nan")


def _p80(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, 0.8)) if arr.size else float("nan")


_STATS = {
    "median": lambda a: float(np.median(a)),
    "mean": lambda a: float(np.mean(a)),
    "p20": lambda a: float(np.quantile(a, 0.2)),
}


def bootstrap_patient_stat(
    values: Any,
    stat: str = "median",
    n_boot: int = 2000,
    seed: int = 42,
    level: float = 0.95,
) -> Dict[str, float]:
    """Patient-cluster bootstrap of a statistic over per-patient values.

    ``values`` holds one number per held-out patient (the cluster). Patients are resampled
    with replacement ``n_boot`` times; the percentile interval at ``level`` and the bootstrap
    SD (se) are returned with the point estimate. With few clusters (7 here) the percentile
    interval is coarse: its ends are order statistics of the observed values.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {sorted(_STATS)}")
    fn = _STATS[stat]
    if arr.size == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n_clusters": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    draws = np.apply_along_axis(fn, 1, arr[idx])
    alpha = (1.0 - float(level)) / 2.0
    return {
        "estimate": fn(arr),
        "lo": float(np.quantile(draws, alpha)),
        "hi": float(np.quantile(draws, 1.0 - alpha)),
        "se": float(np.std(draws, ddof=1)) if draws.size > 1 else 0.0,
        "n_clusters": int(arr.size),
    }


def summarize_by_patient_k(units: pd.DataFrame, group_cols: Sequence[str] = ("representation", "arm")) -> pd.DataFrame:
    """Per (group, heldout_patient, k): the lift distribution over training subsets.

    ``units`` needs columns heldout_patient, k, subset, heldout_lift, heldout_ap, prevalence,
    n_train_malignant, n_train, n_nonzero, plus ``group_cols``. Optional: at_grid_lower_bound.
    """
    keys = list(group_cols) + ["heldout_patient", "k"]
    work = units.copy()
    work["all_zero"] = work["n_nonzero"].astype(float) == 0
    if "at_grid_lower_bound" not in work:
        work["at_grid_lower_bound"] = np.nan
    out = (
        work.groupby(keys, as_index=False)
        .agg(
            n_subsets=("subset", "nunique"),
            prevalence=("prevalence", "first"),
            lift_mean=("heldout_lift", "mean"),
            lift_median=("heldout_lift", "median"),
            lift_min=("heldout_lift", "min"),
            lift_max=("heldout_lift", "max"),
            lift_sd=("heldout_lift", "std"),
            ap_mean=("heldout_ap", "mean"),
            n_train_malignant_mean=("n_train_malignant", "mean"),
            n_train_malignant_min=("n_train_malignant", "min"),
            n_train_malignant_max=("n_train_malignant", "max"),
            n_train_mean=("n_train", "mean"),
            frac_all_zero=("all_zero", "mean"),
            frac_C_at_grid_lower_bound=("at_grid_lower_bound", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return out


def curve_by_k(
    by_patient: pd.DataFrame,
    group_cols: Sequence[str] = ("representation", "arm"),
    n_boot: int = 2000,
    seed: int = 42,
    units: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per (group, k): Option B summaries over held-out patients of the per-(h, k) mean lift,
    with patient-cluster bootstrap 95% CIs for the median and the patient-equal mean."""
    rows = []
    for key, frame in by_patient.groupby(list(group_cols) + ["k"], sort=True):
        key = key if isinstance(key, tuple) else (key,)
        vals = frame["lift_mean"].to_numpy(dtype=np.float64)
        med = bootstrap_patient_stat(vals, "median", n_boot=n_boot, seed=seed)
        mean = bootstrap_patient_stat(vals, "mean", n_boot=n_boot, seed=seed)
        row = dict(zip(list(group_cols) + ["k"], key))
        row.update(
            n_heldout=int(frame["heldout_patient"].nunique()),
            n_units=int(frame["n_subsets"].sum()),
            median_lift=med["estimate"],
            median_lift_lo=med["lo"],
            median_lift_hi=med["hi"],
            median_lift_se=med["se"],
            p20_lift=_p20(vals),
            p80_lift=_p80(vals),
            mean_lift=mean["estimate"],
            mean_lift_lo=mean["lo"],
            mean_lift_hi=mean["hi"],
            mean_lift_se=mean["se"],
            n_patients_lift_gt_0=int(np.sum(vals > 0)),
            median_ap=float(np.median(frame["ap_mean"])),
            median_train_malignant=float(np.median(frame["n_train_malignant_mean"])),
            frac_all_zero=float(np.average(frame["frac_all_zero"], weights=frame["n_subsets"])),
        )
        if units is not None:
            mask = np.ones(len(units), dtype=bool)
            for col, value in zip(list(group_cols) + ["k"], key):
                mask &= units[col].to_numpy() == value
            sub = units.loc[mask]
            row["train_malignant_unit_min"] = float(sub["n_train_malignant"].min()) if len(sub) else float("nan")
            row["train_malignant_unit_median"] = float(sub["n_train_malignant"].median()) if len(sub) else float("nan")
            row["train_malignant_unit_max"] = float(sub["n_train_malignant"].max()) if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gain(
    by_patient: pd.DataFrame,
    k_from: int = 1,
    group_cols: Sequence[str] = ("representation", "arm"),
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Per (group, k_to): per held-out patient, mean lift at k_to minus mean lift at k_from;
    median and mean over patients with patient-cluster bootstrap CIs."""
    rows = []
    for key, frame in by_patient.groupby(list(group_cols), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        wide = frame.pivot_table(index="heldout_patient", columns="k", values="lift_mean")
        if k_from not in wide.columns:
            continue
        for k_to in sorted(c for c in wide.columns if c != k_from):
            delta = (wide[k_to] - wide[k_from]).dropna().to_numpy(dtype=np.float64)
            med = bootstrap_patient_stat(delta, "median", n_boot=n_boot, seed=seed)
            mean = bootstrap_patient_stat(delta, "mean", n_boot=n_boot, seed=seed)
            row = dict(zip(list(group_cols), key))
            row.update(
                k_from=int(k_from), k_to=int(k_to), n_heldout=int(delta.size),
                median_gain=med["estimate"], median_gain_lo=med["lo"], median_gain_hi=med["hi"],
                mean_gain=mean["estimate"], mean_gain_lo=mean["lo"], mean_gain_hi=mean["hi"],
                n_patients_gain_gt_0=int(np.sum(delta > 0)), n_patients_gain_lt_0=int(np.sum(delta < 0)),
            )
            rows.append(row)
    return pd.DataFrame(rows)


def patient_inclusion_effect(
    units: pd.DataFrame,
    group_cols: Sequence[str] = ("representation", "arm"),
    ks: Sequence[int] = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    """Per (group, heldout_patient, included_patient j): at each k where subsets with and without
    j both exist, mean lift(with j) - mean lift(without j); the effect is the equal-weight mean
    over those k. Descriptive: subsets with j also differ in their other members."""
    rows = []
    for key, frame in units.groupby(list(group_cols) + ["heldout_patient"], sort=True):
        key = key if isinstance(key, tuple) else (key,)
        frame = frame.loc[frame["k"].isin(list(ks))]
        if frame.empty:
            continue
        members = frame["subset"].map(parse_subset_tag)
        candidates = sorted({p for s in members for p in s})
        for j in candidates:
            has = members.map(lambda s, j=j: j in s).to_numpy()
            deltas, used_k, n_with, n_without = [], [], 0, 0
            for k in sorted(frame["k"].unique()):
                at_k = frame["k"].to_numpy() == k
                with_j = frame["heldout_lift"].to_numpy()[at_k & has]
                without_j = frame["heldout_lift"].to_numpy()[at_k & ~has]
                if with_j.size and without_j.size:
                    deltas.append(float(np.mean(with_j) - np.mean(without_j)))
                    used_k.append(int(k))
                    n_with += int(with_j.size)
                    n_without += int(without_j.size)
            row = dict(zip(list(group_cols) + ["heldout_patient"], key))
            row.update(
                included_patient=j,
                inclusion_effect=float(np.mean(deltas)) if deltas else float("nan"),
                n_k_used=len(used_k),
                ks_used=";".join(str(k) for k in used_k),
                n_units_with=n_with,
                n_units_without=n_without,
            )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# inverse power law
# --------------------------------------------------------------------------------------


def inverse_power_law(k: Any, a: float, b: float, c: float) -> np.ndarray:
    """y(k) = a - b * k**(-c): rises towards the asymptote a when b > 0 and c > 0."""
    return a - b * np.power(np.asarray(k, dtype=np.float64), -c)


DEFAULT_BOUNDS = ((-1.0, 0.0, 0.05), (1.0, 2.0, 5.0))


def fit_inverse_power_law(
    k: Any,
    y: Any,
    se: Optional[Any] = None,
    *,
    se_floor: float = 0.0025,
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = DEFAULT_BOUNDS,
    c_starts: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> Dict[str, Any]:
    """Weighted least squares fit of y = a - b k^-c (scipy curve_fit, bounded, multi-start).

    Weights are 1 / max(se, se_floor)^2 (se = bootstrap SE of each point). Returns a, b, c,
    weighted SSE, converged, which parameters sit at a bound, and the number of points.
    Needs at least 3 points; with 3 the fit is exact and has no residual degrees of freedom.
    """
    from scipy.optimize import curve_fit

    k_arr = np.asarray(k, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(k_arr) & np.isfinite(y_arr)
    k_arr, y_arr = k_arr[ok], y_arr[ok]
    if se is None:
        sigma = np.ones_like(y_arr)
    else:
        sigma = np.maximum(np.asarray(se, dtype=np.float64)[ok], float(se_floor))
        sigma = np.where(np.isfinite(sigma), sigma, float(se_floor))
    out: Dict[str, Any] = {"n_points": int(k_arr.size), "a": float("nan"), "b": float("nan"), "c": float("nan"),
                           "wsse": float("nan"), "converged": False, "at_bound": "", "message": ""}
    if k_arr.size < 3:
        out["message"] = "fewer than 3 points"
        return out
    lo, hi = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    best = None
    for c0 in c_starts:
        b0 = float(np.clip(max(y_arr.max() - y_arr.min(), 1e-3), lo[1] + 1e-9, hi[1] - 1e-9))
        a0 = float(np.clip(y_arr.max(), lo[0] + 1e-9, hi[0] - 1e-9))
        p0 = [a0, b0, float(np.clip(c0, lo[2] + 1e-9, hi[2] - 1e-9))]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(inverse_power_law, k_arr, y_arr, p0=p0, sigma=sigma, bounds=(lo, hi), maxfev=20000)
        except Exception as exc:  # pragma: no cover - scipy failure modes
            out["message"] = f"{type(exc).__name__}: {exc}"
            continue
        resid = (y_arr - inverse_power_law(k_arr, *popt)) / sigma
        wsse = float(np.sum(resid ** 2))
        if best is None or wsse < best[1] - 1e-15:
            best = (popt, wsse)
    if best is None:
        return out
    popt, wsse = best
    span = hi - lo
    at = [name for name, v, l, h, s in zip("abc", popt, lo, hi, span) if v <= l + 1e-6 * s or v >= h - 1e-6 * s]
    out.update(a=float(popt[0]), b=float(popt[1]), c=float(popt[2]), wsse=wsse, converged=True, at_bound=";".join(at), message="ok")
    return out


def validate_power_law(
    k: Any,
    y: Any,
    se: Any,
    lo: Any,
    hi: Any,
    *,
    fit_ks: Sequence[int] = (1, 2, 3, 4),
    target_k: int = 5,
    min_rise: float = 0.005,
    se_floor: float = 0.0025,
) -> Dict[str, Any]:
    """Fit on ``fit_ks`` and predict ``target_k``; compare with the observed summary and its CI.

    supported = converged, no parameter at a bound, the fitted curve rises by at least
    ``min_rise`` (lift units) from the first to the target k, and the prediction lies inside the
    observed bootstrap CI at target_k. Anything else is reported with the failed check.
    """
    k_arr = np.asarray(k, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    se_arr = np.asarray(se, dtype=np.float64)
    lo_arr = np.asarray(lo, dtype=np.float64)
    hi_arr = np.asarray(hi, dtype=np.float64)
    sel = np.isin(k_arr, np.asarray(fit_ks, dtype=np.float64))
    fit = fit_inverse_power_law(k_arr[sel], y_arr[sel], se_arr[sel], se_floor=se_floor)
    tgt = np.flatnonzero(k_arr == float(target_k))
    record: Dict[str, Any] = {
        "fit_ks": ";".join(str(int(v)) for v in fit_ks), "target_k": int(target_k),
        **{f"fit_{key}": value for key, value in fit.items()},
        "pred_target": float("nan"), "obs_target": float("nan"), "obs_target_lo": float("nan"),
        "obs_target_hi": float("nan"), "abs_error": float("nan"), "pred_in_obs_ci": False,
        "fitted_rise": float("nan"), "supported": False, "reason": "",
    }
    if not fit["converged"]:
        record["reason"] = f"fit failed: {fit['message']}"
        return record
    pred = float(inverse_power_law(target_k, fit["a"], fit["b"], fit["c"]))
    rise = float(pred - inverse_power_law(min(fit_ks), fit["a"], fit["b"], fit["c"]))
    record.update(pred_target=pred, fitted_rise=rise)
    if tgt.size:
        i = int(tgt[0])
        record.update(obs_target=float(y_arr[i]), obs_target_lo=float(lo_arr[i]), obs_target_hi=float(hi_arr[i]),
                      abs_error=float(abs(pred - y_arr[i])),
                      pred_in_obs_ci=bool(lo_arr[i] - 1e-12 <= pred <= hi_arr[i] + 1e-12))
    reasons = []
    if fit["at_bound"]:
        reasons.append(f"parameter at bound ({fit['at_bound']})")
    if not rise >= float(min_rise):
        reasons.append(f"fitted rise {rise:.4f} < {min_rise}")
    if not tgt.size:
        reasons.append(f"no observed k={target_k}")
    elif not record["pred_in_obs_ci"]:
        reasons.append("prediction outside the observed CI")
    record["supported"] = not reasons
    record["reason"] = "; ".join(reasons) if reasons else "all checks pass"
    return record


def power_law_table(
    curve: pd.DataFrame,
    group_cols: Sequence[str] = ("representation", "arm"),
    stats: Sequence[str] = ("median", "mean"),
    min_rise: float = 0.005,
) -> pd.DataFrame:
    """Per (group, stat): the primary validation (fit k <= 4, predict k = 5) and a secondary one
    (fit k <= 5, predict k = 6, the ordinary LOPO fold), each with validate_power_law's verdict.
    A fit is reportable only when the primary validation is supported; the secondary row alone
    never makes it reportable."""
    rows = []
    for key, frame in curve.groupby(list(group_cols), sort=True):
        key = key if isinstance(key, tuple) else (key,)
        frame = frame.sort_values("k")
        for stat in stats:
            col = f"{stat}_lift"
            args = (frame["k"], frame[col], frame[f"{col}_se"], frame[f"{col}_lo"], frame[f"{col}_hi"])
            for role, fit_ks, target in (("primary", (1, 2, 3, 4), 5), ("secondary", (1, 2, 3, 4, 5), 6)):
                if not set(fit_ks) <= set(frame["k"].astype(int)):
                    continue
                rec = validate_power_law(*args, fit_ks=fit_ks, target_k=target, min_rise=min_rise)
                row = dict(zip(list(group_cols), key))
                row.update(stat=stat, validation=role, **rec)
                rows.append(row)
    return pd.DataFrame(rows)
