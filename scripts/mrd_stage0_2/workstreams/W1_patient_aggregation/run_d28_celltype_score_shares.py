"""D28 runner: cell-type-stratified score-energy shares per patient (W1.9).

Question: is a patient's excess or deficit of Stage 1 score energy (D16's over-representation
ratio, e.g. P01's median 1.24 over the 320 stored PCA/FA bases, or the P02+P03+P04 share) explained
by the patient's predicted cell-type mix?

Plain-language estimand, for one stored transductive Stage 1 basis (panel x method x K, seed 42):

- score energy of a cell: e_i = sum over the basis's non-degenerate factors f of
  w_f * (x_if - cohort mean_f)^2. Two weightings:
    ss_weighted   w_f = 1           (the D16 definition: raw scores, so high-variance factors count more)
    standardized  w_f = 1 / var_f   (each factor z-scored over the cohort first: every factor counts equally)
- a patient's energy E_p = sum of e_i over its cells; its energy share = E_p / E_total;
- ratio = energy share / cell share (n_p / N). 1 = the patient carries energy in proportion to its
  cells; this is D16's weighted_enrichment.
- composition-only expectation: replace every cell's energy by the average energy of its stratum
  (predicted.annotation; or annotation x malignant label) in a reference set of cells, and sum over
  the patient's cells: X_p = sum_s n_ps * ebar_s. Expected share = X_p / E_total; composition ratio =
  expected share / cell share = the patient's cell-type-mix-weighted average of the strata's
  per-cell energy relative to the cohort's.
- residual ratio = E_p / X_p = ratio / composition ratio (indirect standardization, like a
  standardized mortality ratio). 1 = the patient's energy is exactly what its cell-type mix predicts.
- references: cohort (all cells of the stratum, the patient's own included) and loo (the other
  patients' cells only; strata no other patient has fall back to the cohort average and are counted).
- scopes: all cells; normal cells only (energy still about the all-cell cohort mean, as in D16;
  shares and strata among normal cells).
- units: the 13 patients and the P02+P03+P04 group (the highest-prevalence patients).

Also per factor (cohort reference, cell-type strata): observed, composition and residual enrichment,
and the top-enrichment patient before and after the adjustment.

Nothing is refit and no Stage 2 model is fit (c_policy not_applicable). CPU, about a minute for all
320 bases. Parts hold sufficient statistics (per patient x cell type x label x factor sums of squared
deviations), so the merge recomputes every variant without reloading scores.

Usage (repo root):
  python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.py \
      --panels all --resume --workers 12
  ... --panels pilot                      # IFN-alpha, allograft, apoptosis, TNFa/NF-kB
  ... --merge-only                        # rebuild tables, figures and findings from parts
  ... --out-root <scratch dir>            # write outputs outside EXP (smoke runs)

Outputs (default base EXP/analysis; --out-root replaces it):
  cache/stage2_diagnostic/factor_patient_matrix_celltype_v1/parts/   one .npz per unit (resume state)
  scorecards/stage2_diagnostic/factor_patient_matrix_celltype_v1/    tables, checks, findings, manifest, ledger
  figures/stage2_diagnostic/factor_patient_matrix_celltype_v1/       D28A-D28C (.png, .pdf)

Resume contract: a unit is finished when its part reads back with the expected shapes, cohort and
cell-type order; empty or corrupt parts are recomputed. A merge failure writes merge_failed.json in
the cache dir; fix the cause and rerun with --merge-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, git_state, study_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factor_patient_matrix as fpm  # noqa: E402
import run_d16_factor_patient_matrix as d16  # noqa: E402  (panel list, score paths, cache-order check)

EXP_DEFAULT = experiment_dir(must_exist=False)

ANALYSIS_ID = "D28"
TASK = "W1.9"
SLUG = "factor_patient_matrix_celltype_v1"
PREFIX = SLUG
SEED = 42
METHODS = ("pca", "fa")
KS = (5, 10, 20, 40)
FOCUS_PANEL = d16.FOCUS_PANEL
PILOT_PANELS = d16.PILOT_PANELS
N_CELLS = 60_118
MIN_MALIGNANT_SUPPORT = 10
CELLTYPE_COLUMN = "predicted.annotation"
CELL_METADATA_DEFAULT = d16.CELL_METADATA_DEFAULT
COHORT_CACHE_DEFAULT = d16.COHORT_CACHE_DEFAULT
HP_UNIT = "P02+P03+P04"
D16_SLUG = d16.SLUG
D16_TABLES = f"analysis/scorecards/stage2_diagnostic/{D16_SLUG}"

WEIGHTINGS = ("ss_weighted", "standardized")
REFERENCES = ("cohort", "loo")
# (scope, strata) pairs; annotation x label strata only make sense when both labels are in scope
VARIANTS = (("all", "celltype"), ("all", "celltype_label"), ("normal", "celltype"))
PRIMARY = {"weighting": "ss_weighted", "reference": "cohort", "strata": "celltype"}

# D16 devlog headline values (rounded to 2 decimals there); the merge re-derives them
D16_HEADLINE = {"P01_median_ratio": 1.24, "hp_ifna_pca40": 1.11, "hp_ifna_fa40": 1.02, "hp_median_ratio": 0.96}
ROUND_TOL = 0.005 + 1e-9
D16_TOL = 1e-9
IDENTITY_TOL = 1e-9

_CELLS = None  # worker-global Cells


# --------------------------------------------------------------------------------------
# cohort with cell types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cells:
    """Per-cell integer codes in Stage 1 score row order."""

    patient_code: np.ndarray  # int, index into patients
    celltype_code: np.ndarray  # int, index into celltypes
    malignant: np.ndarray  # bool
    patients: tuple
    celltypes: tuple
    supported: tuple = ()
    high_prevalence: tuple = fpm.HIGH_PREVALENCE

    @property
    def n_cells(self) -> int:
        return int(self.patient_code.size)

    def units(self) -> dict:
        """Unit name -> patient indices: each patient, then the P02+P03+P04 group if present."""
        out = {p: np.array([i]) for i, p in enumerate(self.patients)}
        hp = [self.patients.index(p) for p in self.high_prevalence if p in self.patients]
        if len(hp) == len(self.high_prevalence) and hp:
            out[HP_UNIT] = np.array(sorted(hp))
        return out


def make_cells(patient, celltype, malignant, supported=(), high_prevalence=fpm.HIGH_PREVALENCE) -> Cells:
    patient = np.asarray(patient).astype(str)
    celltype = np.asarray(celltype).astype(str)
    malignant = np.asarray(malignant).astype(bool)
    if not (patient.shape == celltype.shape == malignant.shape) or patient.ndim != 1:
        raise ValueError("patient, celltype and malignant must be 1-D arrays of equal length")
    patients = tuple(sorted(pd.unique(patient)))
    celltypes = tuple(sorted(pd.unique(celltype)))
    pcode = pd.Categorical(patient, categories=patients).codes.astype(np.int64)
    ccode = pd.Categorical(celltype, categories=celltypes).codes.astype(np.int64)
    return Cells(pcode, ccode, malignant, patients, celltypes, tuple(supported), tuple(high_prevalence))


def exp_relpath(key: str, default: str) -> str:
    """EXP-relative path of a study input, from study_config.json (built-in fallback)."""
    return str(study_config().get(key) or default)


def load_cells(exp: Path, celltype_column: str = CELLTYPE_COLUMN) -> tuple:
    """Cell metadata (patient, CN.label, cell type) in Stage 1 row order, plus provenance."""
    meta_path = exp / exp_relpath("cell_metadata_relpath", CELL_METADATA_DEFAULT)
    if not meta_path.is_file():
        raise FileNotFoundError(f"cell metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path, usecols=["cell_id", "patient", "CN.label", celltype_column])
    if len(meta) != N_CELLS:
        raise AssertionError(f"{meta_path} has {len(meta)} rows, expected {N_CELLS}")
    if meta[celltype_column].isna().any():
        raise AssertionError(f"{celltype_column} has {int(meta[celltype_column].isna().sum())} missing values")
    malignant = meta["CN.label"].astype(str).eq("cancer").to_numpy()
    patient = meta["patient"].astype(str).to_numpy()
    supported = fpm.supported_patients(patient, malignant, min_malignant=MIN_MALIGNANT_SUPPORT)
    cfg_supported = study_config().get("supported_patients")
    if cfg_supported and tuple(sorted(cfg_supported)) != supported:
        raise AssertionError(f"support rule gives {supported}, study_config says {cfg_supported}")
    cells = make_cells(patient, meta[celltype_column].astype(str).to_numpy(), malignant, supported)
    provenance = {
        "cell_metadata": d16.rel(exp, meta_path),
        "celltype_column": celltype_column,
        "celltypes": list(cells.celltypes),
        "supported_patients": list(supported),
        "cohort_cache_order_check": d16.verify_cache_order(exp, meta),
    }
    return cells, provenance


# --------------------------------------------------------------------------------------
# sufficient statistics (one unit)
# --------------------------------------------------------------------------------------


def group_sums(scores: np.ndarray, cells: Cells, degenerate_sd: float = fpm.DEGENERATE_SD) -> dict:
    """Per patient x cell type x label x non-degenerate factor sums of (x - cohort mean)^2.

    Returns S (P, T, 2, F_nd; label 0 = normal, 1 = malignant), n (P, T, 2), the cohort SD of
    every factor (population SD, as D16) and the non-degenerate mask (SD >= degenerate_sd).
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != cells.n_cells:
        raise ValueError(f"scores shape {x.shape} does not match {cells.n_cells} cells")
    center = x.mean(axis=0)
    sd = x.std(axis=0)
    nondeg = sd >= degenerate_sd
    d2 = (x[:, nondeg] - center[nondeg]) ** 2
    n_p, n_t = len(cells.patients), len(cells.celltypes)
    code = (cells.patient_code * n_t + cells.celltype_code) * 2 + cells.malignant.astype(np.int64)
    g = n_p * n_t * 2
    n = np.bincount(code, minlength=g).reshape(n_p, n_t, 2)
    s = np.zeros((g, d2.shape[1]))
    for j in range(d2.shape[1]):
        s[:, j] = np.bincount(code, weights=d2[:, j], minlength=g)
    return {"S": s.reshape(n_p, n_t, 2, d2.shape[1]), "n": n, "sd": sd, "nondegenerate": nondeg}


def energy(stats: dict, weighting: str, factor_level: bool = False) -> np.ndarray:
    """Energy array (P, T, 2, F): per factor (factor_level) or summed over factors (F = 1)."""
    s = stats["S"]
    if factor_level:
        return s
    if weighting == "ss_weighted":
        w = np.ones(s.shape[-1])
    elif weighting == "standardized":
        w = 1.0 / stats["sd"][stats["nondegenerate"]] ** 2
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return (s * w).sum(axis=-1, keepdims=True)


def _scope_arrays(e: np.ndarray, n: np.ndarray, scope: str, strata: str) -> tuple:
    """Collapse (P, T, 2, F) energy and (P, T, 2) counts to (P, S, F) and (P, S) for a scope."""
    if scope == "normal":
        if strata != "celltype":
            raise ValueError("normal-cell scope uses cell-type strata only")
        return e[:, :, 0, :], n[:, :, 0]
    if scope != "all":
        raise ValueError(f"unknown scope {scope!r}")
    if strata == "celltype":
        return e.sum(axis=2), n.sum(axis=2)
    if strata == "celltype_label":
        p, t, l, f = e.shape
        return e.reshape(p, t * l, f), n.reshape(p, t * l)
    raise ValueError(f"unknown strata {strata!r}")


def allocate(e: np.ndarray, n: np.ndarray, units: dict, *, scope: str, strata: str, reference: str) -> dict:
    """Observed vs composition-only allocation per unit (arrays over the trailing factor axis).

    Returns {unit: dict of arrays of length F}: energy, expected_energy, n_cells, cell_share,
    obs_share, exp_share, obs_ratio, comp_ratio, resid_ratio, n_fallback_cells.
    """
    es, ns = _scope_arrays(e, n, scope, strata)
    e_tot = es.sum(axis=(0, 1))
    n_tot = ns.sum()
    e_strat = es.sum(axis=0)
    n_strat = ns.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ebar_cohort = np.where(n_strat[:, None] > 0, e_strat / np.maximum(n_strat, 1)[:, None], 0.0)
    all_idx = np.arange(es.shape[0])
    out = {}
    for name, idx in units.items():
        eu = es[idx].sum(axis=(0, 1))
        nus = ns[idx].sum(axis=0)
        nu = nus.sum()
        n_fb = 0
        if reference == "cohort":
            ebar = ebar_cohort
        elif reference == "loo":
            rest = np.setdiff1d(all_idx, idx)
            er = es[rest].sum(axis=0)
            nr = ns[rest].sum(axis=0)
            has = nr > 0
            ebar = np.where(has[:, None], er / np.maximum(nr, 1)[:, None], ebar_cohort)
            n_fb = int(nus[~has].sum())
        else:
            raise ValueError(f"unknown reference {reference!r}")
        x = (nus[:, None] * ebar).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_share = nu / n_tot if n_tot else np.nan
            obs_share = np.where(e_tot > 0, eu / np.where(e_tot > 0, e_tot, 1), np.nan)
            exp_share = np.where(e_tot > 0, x / np.where(e_tot > 0, e_tot, 1), np.nan)
            obs_ratio = obs_share / cell_share if nu else np.full_like(obs_share, np.nan)
            comp_ratio = exp_share / cell_share if nu else np.full_like(exp_share, np.nan)
            resid = np.where(x > 0, eu / np.where(x > 0, x, 1), np.nan)
        out[name] = {
            "energy": eu, "expected_energy": x, "n_cells": int(nu), "cell_share": cell_share,
            "obs_share": obs_share, "exp_share": exp_share, "obs_ratio": obs_ratio,
            "comp_ratio": comp_ratio, "resid_ratio": resid, "n_fallback_cells": n_fb,
        }
    return out


def within_celltype(e: np.ndarray, n: np.ndarray, units: dict, *, scope: str, reference: str) -> dict:
    """Per unit x cell type pieces of the decomposition (F = 1 energy).

    comp_ratio = sum_t comp_share_t * type_level_t (cohort reference) and
    resid_ratio = sum_t weight_t * within_ratio_t, with weight_t = n_ut * ebar_t / X_u.
    """
    es, ns = _scope_arrays(e, n, scope, "celltype")
    es = es[..., 0]
    e_tot, n_tot = es.sum(), ns.sum()
    e_strat, n_strat = es.sum(axis=0), ns.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ebar_cohort = np.where(n_strat > 0, e_strat / np.maximum(n_strat, 1), np.nan)
        level = ebar_cohort / (e_tot / n_tot)
    all_idx = np.arange(es.shape[0])
    out = {}
    for name, idx in units.items():
        eut = es[idx].sum(axis=0)
        nut = ns[idx].sum(axis=0)
        if reference == "cohort":
            ebar = ebar_cohort
            fallback = np.zeros(nut.shape, dtype=bool)
        else:
            rest = np.setdiff1d(all_idx, idx)
            er, nr = es[rest].sum(axis=0), ns[rest].sum(axis=0)
            fallback = nr == 0
            with np.errstate(invalid="ignore", divide="ignore"):
                ebar = np.where(fallback, ebar_cohort, er / np.maximum(nr, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            x_t = np.where(nut > 0, nut * np.nan_to_num(ebar), 0.0)
            x = x_t.sum()
            out[name] = {
                "n_cells": nut,
                "comp_share": nut / nut.sum() if nut.sum() else np.full(nut.shape, np.nan),
                "unit_share_of_type": np.where(n_strat > 0, nut / np.maximum(n_strat, 1), np.nan),
                "type_energy_level": level,
                "reference_energy_level": ebar / (e_tot / n_tot),
                "within_ratio": np.where((nut > 0) & (ebar > 0), eut / np.maximum(nut, 1) / np.where(ebar > 0, ebar, 1), np.nan),
                "weight": x_t / x if x > 0 else np.full(nut.shape, np.nan),
                "fallback": fallback & (nut > 0),
            }
    return out


# --------------------------------------------------------------------------------------
# tables from the parts
# --------------------------------------------------------------------------------------


def rep_patient_rows(key: dict, stats: dict, cells: Cells) -> pd.DataFrame:
    """Representation x unit x scope x weighting x reference x strata rows."""
    units = cells.units()
    rows = []
    for weighting in WEIGHTINGS:
        e = energy(stats, weighting)
        for scope, strata in VARIANTS:
            for reference in REFERENCES:
                res = allocate(e, stats["n"], units, scope=scope, strata=strata, reference=reference)
                for unit, r in res.items():
                    rows.append({
                        **key, "unit": unit, "is_group": unit == HP_UNIT,
                        "supported": unit in cells.supported, "high_prevalence": unit in cells.high_prevalence,
                        "scope": scope, "weighting": weighting, "reference": reference, "strata": strata,
                        "n_cells": r["n_cells"], "cell_share": float(r["cell_share"]),
                        "energy": float(r["energy"][0]), "expected_energy": float(r["expected_energy"][0]),
                        "obs_share": float(r["obs_share"][0]), "exp_share": float(r["exp_share"][0]),
                        "obs_ratio": float(r["obs_ratio"][0]), "comp_ratio": float(r["comp_ratio"][0]),
                        "resid_ratio": float(r["resid_ratio"][0]), "n_fallback_cells": r["n_fallback_cells"],
                    })
    return pd.DataFrame(rows)


def factor_rows(key: dict, stats: dict, cells: Cells) -> pd.DataFrame:
    """Representation x factor x unit x scope rows (cohort reference, cell-type strata)."""
    units = cells.units()
    e = energy(stats, "ss_weighted", factor_level=True)
    fidx = np.flatnonzero(stats["nondegenerate"]) + 1
    frames = []
    for scope in ("all", "normal"):
        res = allocate(e, stats["n"], units, scope=scope, strata="celltype", reference="cohort")
        for unit, r in res.items():
            frames.append(pd.DataFrame({
                **{k: v for k, v in key.items()}, "factor_index": fidx, "unit": unit, "scope": scope,
                "ss_share": r["obs_share"], "exp_share": r["exp_share"], "cell_share": r["cell_share"],
                "obs_enrichment": r["obs_ratio"], "comp_enrichment": r["comp_ratio"],
                "resid_enrichment": r["resid_ratio"],
            }))
    return pd.concat(frames, ignore_index=True)


def celltype_rows(key: dict, stats: dict, cells: Cells) -> pd.DataFrame:
    """Representation x unit x scope x reference x cell type rows (ss_weighted energy)."""
    units = cells.units()
    e = energy(stats, "ss_weighted")
    frames = []
    for scope in ("all", "normal"):
        for reference in REFERENCES:
            res = within_celltype(e, stats["n"], units, scope=scope, reference=reference)
            for unit, r in res.items():
                frames.append(pd.DataFrame({
                    **key, "unit": unit, "scope": scope, "reference": reference, "cell_type": list(cells.celltypes),
                    **{c: r[c] for c in ("n_cells", "comp_share", "unit_share_of_type", "type_energy_level",
                                         "reference_energy_level", "within_ratio", "weight", "fallback")},
                }))
    return pd.concat(frames, ignore_index=True)


def distribution_summary(rp: pd.DataFrame) -> pd.DataFrame:
    """Per unit x variant: distribution of observed, composition and residual ratios over bases."""
    keys = ["unit", "scope", "weighting", "reference", "strata"]
    rows = []
    for k, g in rp.groupby(keys, sort=False):
        rec = dict(zip(keys, k))
        rec["n_representations"] = int(len(g))
        for col in ("obs_ratio", "comp_ratio", "resid_ratio"):
            v = g[col].dropna()
            rec[f"{col}_median"] = float(v.median()) if len(v) else np.nan
            rec[f"{col}_q25"] = float(v.quantile(0.25)) if len(v) else np.nan
            rec[f"{col}_q75"] = float(v.quantile(0.75)) if len(v) else np.nan
            rec[f"{col}_min"] = float(v.min()) if len(v) else np.nan
            rec[f"{col}_max"] = float(v.max()) if len(v) else np.nan
            rec[f"{col}_n_gt1"] = int((v > 1).sum())
        lo = np.log(g["obs_ratio"].to_numpy(dtype=float))
        lr = np.log(g["resid_ratio"].to_numpy(dtype=float))
        ok = np.isfinite(lo) & np.isfinite(lr)
        rec["n_resid_closer_to_1"] = int((np.abs(lr[ok]) < np.abs(lo[ok])).sum())
        # share of the log-scale departure from 1 that the cell-type mix accounts for, on bases where the
        # observed ratio is at least 5% away from 1 (log(obs) = log(comp) + log(resid) per basis)
        lc = np.log(g["comp_ratio"].to_numpy(dtype=float))
        far = ok & (np.abs(lo) >= np.log(1.05))
        frac = lc[far] / lo[far]
        rec["n_far_from_1"] = int(far.sum())
        rec["frac_log_departure_composition_median"] = float(np.median(frac)) if far.any() else np.nan
        rec["frac_log_departure_composition_q25"] = float(np.quantile(frac, 0.25)) if far.any() else np.nan
        rec["frac_log_departure_composition_q75"] = float(np.quantile(frac, 0.75)) if far.any() else np.nan
        rec["n_same_side_of_1"] = int((np.sign(lr[ok]) == np.sign(lo[ok])).sum())
        rec["n_fallback_cells_max"] = int(g["n_fallback_cells"].max())
        rows.append(rec)
    return pd.DataFrame(rows)


def method_k_summary(rp: pd.DataFrame) -> pd.DataFrame:
    keys = ["unit", "scope", "weighting", "reference", "strata", "stage1_method", "k"]
    g = rp.groupby(keys, sort=True)
    out = g.agg(n_representations=("obs_ratio", "size"), obs_ratio_median=("obs_ratio", "median"),
                comp_ratio_median=("comp_ratio", "median"), resid_ratio_median=("resid_ratio", "median"),
                obs_ratio_n_gt1=("obs_ratio", lambda v: int((v > 1).sum())),
                resid_ratio_n_gt1=("resid_ratio", lambda v: int((v > 1).sum())))
    return out.reset_index()


def celltype_summary(ct: pd.DataFrame) -> pd.DataFrame:
    keys = ["unit", "scope", "reference", "cell_type"]
    g = ct.groupby(keys, sort=False)
    out = g.agg(
        n_representations=("within_ratio", "size"),
        n_cells=("n_cells", "first"),
        comp_share=("comp_share", "first"),
        unit_share_of_type=("unit_share_of_type", "first"),
        type_energy_level_median=("type_energy_level", "median"),
        type_energy_level_q25=("type_energy_level", lambda v: v.quantile(0.25)),
        type_energy_level_q75=("type_energy_level", lambda v: v.quantile(0.75)),
        within_ratio_median=("within_ratio", "median"),
        within_ratio_q25=("within_ratio", lambda v: v.quantile(0.25)),
        within_ratio_q75=("within_ratio", lambda v: v.quantile(0.75)),
        weight_median=("weight", "median"),
        fallback=("fallback", "max"),
    )
    return out.reset_index()


def top_patient_before_after(fl: pd.DataFrame, patients: tuple) -> pd.DataFrame:
    """Share of factors whose top patient (largest enrichment among the 13) is each patient,
    before (observed) and after (residual) the cell-type adjustment; per method x K and pooled."""
    pat = fl.loc[fl["unit"].isin(set(patients))]
    keys = ["stage0_panel_id", "stage1_method", "k", "factor_index", "scope"]
    rows = []
    for col, label in (("obs_enrichment", "observed"), ("resid_enrichment", "residual")):
        sub = pat.loc[pat[col].notna()]
        top = sub.loc[sub.groupby(keys, sort=False)[col].idxmax(), keys + ["unit"]]
        for scope, ts in top.groupby("scope"):
            groups = [("pooled", "all", ts)] + [(m, int(k), g) for (m, k), g in ts.groupby(["stage1_method", "k"])]
            for method, k, g in groups:
                counts = g["unit"].value_counts()
                for p in patients:
                    rows.append({"which": label, "scope": scope, "stage1_method": method, "k": k, "patient": p,
                                 "n_factors": int(len(g)), "n_top": int(counts.get(p, 0)),
                                 "frac_top": float(counts.get(p, 0)) / len(g) if len(g) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# paths, parts, workers
# --------------------------------------------------------------------------------------


def out_dirs(base: Path) -> dict:
    """Output dirs under base (EXP/analysis by default, or --out-root)."""
    return {
        "tables": base / "scorecards" / "stage2_diagnostic" / SLUG,
        "figures": base / "figures" / "stage2_diagnostic" / SLUG,
        "cache": base / "cache" / "stage2_diagnostic" / SLUG,
        "parts": base / "cache" / "stage2_diagnostic" / SLUG / "parts",
    }


def relname(exp: Path, base: Path, path: Path) -> str:
    """EXP-relative path; '<out-root>/...' when the output base is outside EXP."""
    try:
        return d16.rel(exp, path)
    except ValueError:
        return "<out-root>/" + Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(base))).as_posix()


def part_path(parts: Path, unit: dict) -> Path:
    return parts / f"{unit['panel']}__{unit['method']}__k{unit['k']}.npz"


def read_part(path: Path, cells: Cells) -> dict | None:
    """Return the part's statistics when complete and consistent with ``cells``, else None."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            out = {k: z[k] for k in ("S", "n", "sd", "nondegenerate", "meta")}
        meta = json.loads(str(out["meta"]))
    except Exception:
        return None
    s, n = out["S"], out["n"]
    shape = (len(cells.patients), len(cells.celltypes), 2)
    if s.ndim != 4 or s.shape[:3] != shape or n.shape != shape:
        return None
    if int(out["nondegenerate"].sum()) != s.shape[3] or out["sd"].shape != out["nondegenerate"].shape:
        return None
    if int(n.sum()) != cells.n_cells or tuple(meta.get("patients", ())) != cells.patients:
        return None
    if tuple(meta.get("celltypes", ())) != cells.celltypes or not np.isfinite(s).all():
        return None
    out["nondegenerate"] = out["nondegenerate"].astype(bool)
    out["meta"] = meta
    return out


def _init_worker(cells: Cells) -> None:
    global _CELLS
    _CELLS = cells


def compute_unit(exp_str: str, unit: dict, parts_str: str) -> dict:
    exp, parts = Path(exp_str), Path(parts_str)
    t0 = time.time()
    d = d16.score_dir(exp, unit["panel"], unit["method"], unit["k"])
    md = json.loads((d / "metadata.json").read_text())
    scores = np.load(d / "scores.npy")
    if scores.shape[0] != _CELLS.n_cells:
        raise AssertionError(f"{d} has {scores.shape[0]} rows, metadata has {_CELLS.n_cells}")
    st = group_sums(scores, _CELLS)
    meta = {
        "stage0_panel_id": unit["panel"], "short_panel_label": unit["label"], "stage1_method": unit["method"],
        "k": int(unit["k"]), "effective_k": int(md.get("effective_k", scores.shape[1])),
        "n_genes": int(md.get("n_genes", -1)), "scores_relpath": d16.rel(exp, d / "scores.npy"),
        "patients": list(_CELLS.patients), "celltypes": list(_CELLS.celltypes),
        "unit_seconds": round(time.time() - t0, 3),
    }
    out = part_path(parts, unit)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, S=st["S"], n=st["n"], sd=st["sd"], nondegenerate=st["nondegenerate"],
                            meta=np.array(json.dumps(meta)))
    os.replace(tmp, out)
    return {"unit": unit, "seconds": time.time() - t0}


def run_units(exp: Path, units: list, parts: Path, cells: Cells, workers: int) -> None:
    if workers <= 1:
        _init_worker(cells)
        for i, unit in enumerate(units, 1):
            res = compute_unit(str(exp), unit, str(parts))
            print(f"[{i}/{len(units)}] {unit['label']} {unit['method']} K{unit['k']}: {res['seconds']:.1f}s", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cells,)) as pool:
        futures = {pool.submit(compute_unit, str(exp), u, str(parts)): u for u in units}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            u = res["unit"]
            print(f"[{i}/{len(units)}] {u['label']} {u['method']} K{u['k']}: {res['seconds']:.1f}s", flush=True)


# --------------------------------------------------------------------------------------
# regression checks
# --------------------------------------------------------------------------------------


def _check(rows: list, check_id: str, source: str, metric: str, diffs, tol: float, note: str = "") -> None:
    diffs = np.asarray(list(diffs), dtype=float)
    n = int(diffs.size)
    worst = float(np.nanmax(diffs)) if n else float("nan")
    status = "pass" if n and np.isfinite(worst) and worst <= tol else ("absent" if not n else "fail")
    rows.append({"check_id": check_id, "source": source, "metric": metric, "n_compared": n,
                 "max_abs_diff": worst, "tolerance": tol, "status": status, "note": note})


def _primary(rp: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    return rp.loc[rp["scope"].eq(scope) & rp["weighting"].eq(PRIMARY["weighting"])
                  & rp["reference"].eq(PRIMARY["reference"]) & rp["strata"].eq(PRIMARY["strata"])]


def regression_checks(exp: Path, rp: pd.DataFrame, fl: pd.DataFrame, ct: pd.DataFrame, cells: Cells) -> pd.DataFrame:
    rows: list = []
    rep = ["stage0_panel_id", "stage1_method", "k"]
    # internal identities
    coh = rp.loc[rp["reference"].eq("cohort") & ~rp["is_group"]]
    sums = coh.groupby(rep + ["scope", "weighting", "strata"])["exp_share"].sum()
    _check(rows, "identity_expected_shares_sum_to_1", "this run", "sum over patients of the expected share (cohort reference)",
           (sums - 1.0).abs(), IDENTITY_TOL)
    _check(rows, "identity_ratio_eq_comp_x_resid", "this run", "observed ratio - composition ratio x residual ratio",
           (rp["obs_ratio"] - rp["comp_ratio"] * rp["resid_ratio"]).abs().dropna(), IDENTITY_TOL)
    prim_all = _primary(rp, "all").set_index(rep + ["unit"])
    for scope in ("all", "normal"):
        cts = ct.loc[ct["scope"].eq(scope) & ct["reference"].eq("cohort")]
        w = (cts.assign(_p=cts["weight"] * cts["within_ratio"]).groupby(rep + ["unit"])["_p"].sum(min_count=1))
        c = (cts.assign(_c=cts["comp_share"] * cts["type_energy_level"]).groupby(rep + ["unit"])["_c"].sum(min_count=1))
        pr = _primary(rp, scope).set_index(rep + ["unit"])
        _check(rows, f"identity_resid_eq_weighted_within_{scope}", "this run",
               "residual ratio - sum over cell types of weight x within-type ratio", (pr["resid_ratio"] - w.reindex(pr.index)).abs().dropna(), IDENTITY_TOL)
        _check(rows, f"identity_comp_eq_mix_x_level_{scope}", "this run",
               "composition ratio - sum over cell types of mix share x type energy level", (pr["comp_ratio"] - c.reindex(pr.index)).abs().dropna(), IDENTITY_TOL)
    # D16 reproduction
    td16 = exp / D16_TABLES
    rps_path = td16 / "factor_patient_matrix_representation_patient_summary.csv"
    if rps_path.is_file():
        rps = pd.read_csv(rps_path).rename(columns={"patient": "unit"}).set_index(rep + ["unit"])
        for scope, col in (("all", "weighted_enrichment"), ("normal", "weighted_enrichment_normal")):
            pr = _primary(rp, scope)
            pr = pr.loc[~pr["is_group"]].set_index(rep + ["unit"])
            m = pr[["obs_ratio"]].join(rps[[col]], how="inner")
            _check(rows, f"d16_{col}", d16.rel(exp, rps_path), f"observed ratio ({scope} cells, ss_weighted) vs D16 {col}",
                   (m["obs_ratio"] - m[col]).abs(), D16_TOL, f"{len(m)} of {len(pr)} rows matched")
    rs_path = td16 / "factor_patient_matrix_representation_summary.csv"
    if rs_path.is_file():
        rs = pd.read_csv(rs_path).set_index(rep)
        for scope, col in (("all", "hp_enrichment_weighted"), ("normal", "hp_enrichment_normal_weighted")):
            pr = _primary(rp, scope)
            pr = pr.loc[pr["unit"].eq(HP_UNIT)].set_index(rep)
            m = pr[["obs_ratio"]].join(rs[[col]], how="inner")
            _check(rows, f"d16_{col}", d16.rel(exp, rs_path), f"P02+P03+P04 observed ratio ({scope}) vs D16 {col}",
                   (m["obs_ratio"] - m[col]).abs(), D16_TOL, f"{len(m)} of {len(pr)} rows matched")
    long_path = td16 / "factor_patient_matrix_long.csv.gz"
    if long_path.is_file():
        lg = pd.read_csv(long_path, usecols=rep + ["factor_index", "patient", "degenerate", "enrichment", "enrichment_normal"])
        lg = lg.loc[~lg["degenerate"]].rename(columns={"patient": "unit"})
        fk = rep + ["factor_index", "unit"]
        for scope, col in (("all", "enrichment"), ("normal", "enrichment_normal")):
            f = fl.loc[fl["scope"].eq(scope) & ~fl["unit"].eq(HP_UNIT)].set_index(fk)
            m = f[["obs_enrichment"]].join(lg.set_index(fk)[[col]], how="inner")
            _check(rows, f"d16_factor_{col}", d16.rel(exp, long_path), f"per-factor observed enrichment ({scope}) vs D16 {col}",
                   (m["obs_enrichment"] - m[col]).abs(), D16_TOL, f"{len(m)} of {len(f)} rows matched")
        mean_enr = lg.groupby(rep + ["unit"])["enrichment"].mean()
        st = rp.loc[rp["scope"].eq("all") & rp["weighting"].eq("standardized") & rp["reference"].eq("cohort")
                    & rp["strata"].eq("celltype") & ~rp["is_group"]].set_index(rep + ["unit"])
        m = st[["obs_ratio"]].join(mean_enr.rename("mean_enrichment"), how="inner")
        _check(rows, "d16_standardized_eq_mean_factor_enrichment", d16.rel(exp, long_path),
               "standardized observed ratio vs mean of D16 per-factor enrichment", (m["obs_ratio"] - m["mean_enrichment"]).abs(),
               D16_TOL, f"{len(m)} rows")
    # D16 devlog headline values (2-decimal rounding)
    pa = _primary(rp, "all")
    if pa["stage0_panel_id"].nunique() >= 41 and "P01" in cells.patients:
        _check(rows, "d16_headline_P01_median", "devlog 2026-10-08_d16-factor-patient-matrix", "P01 median observed ratio over bases",
               [abs(pa.loc[pa["unit"].eq("P01"), "obs_ratio"].median() - D16_HEADLINE["P01_median_ratio"])], ROUND_TOL)
        _check(rows, "d16_headline_hp_median", "devlog 2026-10-08_d16-factor-patient-matrix", "P02+P03+P04 median observed ratio over bases",
               [abs(pa.loc[pa["unit"].eq(HP_UNIT), "obs_ratio"].median() - D16_HEADLINE["hp_median_ratio"])], ROUND_TOL)
    for method, key in (("pca", "hp_ifna_pca40"), ("fa", "hp_ifna_fa40")):
        r = pa.loc[pa["stage0_panel_id"].eq(FOCUS_PANEL) & pa["stage1_method"].eq(method) & pa["k"].eq(40) & pa["unit"].eq(HP_UNIT)]
        if len(r):
            _check(rows, f"d16_headline_{key}", "devlog 2026-10-08_d16-factor-patient-matrix",
                   f"P02+P03+P04 observed ratio, IFN-alpha {method.upper()}40", [abs(r["obs_ratio"].iloc[0] - D16_HEADLINE[key])], ROUND_TOL)
    columns = ["check_id", "source", "metric", "n_compared", "max_abs_diff", "tolerance", "status", "note"]
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------------------
# figures (report palette: dataviz reference instance, light, print)
# --------------------------------------------------------------------------------------

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 8.5, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "axes.facecolor": SURFACE, "axes.grid": True, "axes.grid.axis": "y",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "legend.frameon": False, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
    })
    return plt


def save(fig, figdir: Path, stem: str, written: list) -> None:
    import matplotlib.pyplot as plt

    for ext in ("png", "pdf"):
        path = figdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=170 if ext == "png" else None)
        written.append(path)
    plt.close(fig)


def _unit_order(cells: Cells) -> list:
    return list(cells.patients) + ([HP_UNIT] if HP_UNIT in cells.units() else [])


def fig_d28a(dist: pd.DataFrame, cells: Cells, figdir: Path, written: list) -> None:
    """Per unit: observed ratio vs composition-only expectation vs residual, median and IQR over bases."""
    plt = _plt()
    order = _unit_order(cells)
    prim = dist.loc[dist["weighting"].eq(PRIMARY["weighting"]) & dist["reference"].eq("cohort")
                    & dist["strata"].eq("celltype")].set_index(["scope", "unit"])
    n_rep = int(prim["n_representations"].max())
    cols = [f"{c}_{q}" for c in ("obs_ratio", "comp_ratio", "resid_ratio") for q in ("q25", "q75")]
    lo = float(np.nanmin(prim[cols].to_numpy(float)))
    hi = float(np.nanmax(prim[cols].to_numpy(float)))
    lo, hi = min(lo, 0.9) / 1.06, max(hi, 1.1) * 1.12
    ticks = [t for t in 2.0 ** np.arange(-2, 2.01, 0.25) if lo <= t <= hi]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.4), sharex=True, constrained_layout=True)
    xs = np.arange(len(order), dtype=float)
    if order[-1] == HP_UNIT:
        xs[-1] += 0.6
    series = (("obs_ratio", "observed ratio (D16)", -0.22, dict(marker="o", mfc=SURFACE, mec=MUTED, color=MUTED)),
              ("comp_ratio", "expected from the cell-type mix alone", 0.0, dict(marker="D", mfc=SURFACE, mec=INK_2, color=INK_2, ms=4.5)),
              ("resid_ratio", "residual = observed / expected", 0.22, dict(marker="o", mfc=INK, mec=INK, color=INK)))
    for ax, scope, title in ((axes[0], "all", "All cells"), (axes[1], "normal", "Normal cells only")):
        ax.axhline(1.0, color=INK_2, lw=0.9, zorder=1)
        for col, label, dx, style in series:
            med = np.array([prim.loc[(scope, u), f"{col}_median"] for u in order], dtype=float)
            q25 = np.array([prim.loc[(scope, u), f"{col}_q25"] for u in order], dtype=float)
            q75 = np.array([prim.loc[(scope, u), f"{col}_q75"] for u in order], dtype=float)
            ax.vlines(xs + dx, q25, q75, color=style["color"], lw=1.4, zorder=2)
            ax.plot(xs + dx, med, ls="none", marker=style["marker"], ms=style.get("ms", 5.5), mfc=style["mfc"],
                    mec=style["mec"], mew=1.2, zorder=3, label=label)
        ax.set_yscale("log", base=2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(lo, hi)
        ax.set_ylabel("ratio to cell share")
        ax.set_title(f"{title}: median over the {n_rep} PCA/FA bases (bars: interquartile range)", fontsize=8.5)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([u if u != HP_UNIT else "P02+P03+P04\n(combined)" for u in order])
    for lab, u in zip(axes[1].get_xticklabels(), order):
        if u not in cells.supported and u != HP_UNIT:
            lab.set_color(MUTED)
        if u in cells.high_prevalence or u == HP_UNIT:
            lab.set_fontweight("bold")
    axes[0].legend(loc="upper right", ncol=1, fontsize=7, handletextpad=0.3)
    fig.suptitle("D28A. Score-energy ratio before and after adjusting for predicted cell type\n"
                 "(1 = energy in proportion to cells; ss-weighted energy, cohort cell-type averages; "
                 "gray labels: unsupported patients)", fontsize=8.5, x=0.0, ha="left", fontweight="bold")
    save(fig, figdir, "D28A_ratio_before_after_by_patient", written)


def fig_d28b(rp: pd.DataFrame, figdir: Path, written: list) -> None:
    """P01 and P02+P03+P04: observed vs residual ratio for every base, by method."""
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.2), constrained_layout=True)
    for j, unit in enumerate(("P01", HP_UNIT)):
        both = rp.loc[rp["unit"].eq(unit) & rp["weighting"].eq("ss_weighted") & rp["reference"].eq("cohort")
                      & rp["strata"].eq("celltype")]
        if both.empty:
            continue
        lo = float(min(both["obs_ratio"].min(), both["resid_ratio"].min(), 0.9)) * 0.96
        hi = float(max(both["obs_ratio"].max(), both["resid_ratio"].max(), 1.1)) * 1.04
        for i, scope in enumerate(("all", "normal")):
            ax = axes[i, j]
            ax.grid(True, axis="both")
            sub = both.loc[both["scope"].eq(scope)]
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--", zorder=1)
            ax.axhline(1.0, color=INK_2, lw=0.9, zorder=1)
            ax.axvline(1.0, color=AXIS, lw=0.8, zorder=1)
            for method, color in (("pca", BLUE), ("fa", ORANGE)):
                m = sub.loc[sub["stage1_method"].eq(method)]
                ax.scatter(m["obs_ratio"], m["resid_ratio"], s=14, color=color, alpha=0.75, lw=0.4,
                           edgecolors=SURFACE, zorder=2)
            foc = sub.loc[sub["stage0_panel_id"].eq(FOCUS_PANEL) & sub["k"].eq(40)]
            ax.scatter(foc["obs_ratio"], foc["resid_ratio"], s=60, facecolors="none", edgecolors=INK, lw=1.2, zorder=3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            name = "P01" if unit == "P01" else "P02+P03+P04"
            ax.set_title(f"{name}, {'all cells' if scope == 'all' else 'normal cells only'}", fontsize=8.5)
            ax.set_xlabel("observed ratio (before adjustment)")
            ax.set_ylabel("residual ratio (after)")
    n = rp.loc[rp["unit"].eq(rp["unit"].iloc[0]) & rp["scope"].eq("all") & rp["weighting"].eq("ss_weighted")
               & rp["reference"].eq("cohort") & rp["strata"].eq("celltype")].groupby("stage1_method").size()
    handles = [Line2D([], [], ls="none", marker="o", ms=5, color=BLUE, label=f"PCA basis ({int(n.get('pca', 0))})"),
               Line2D([], [], ls="none", marker="o", ms=5, color=ORANGE, label=f"FA basis ({int(n.get('fa', 0))})"),
               Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec=INK, label="ringed: IFN-alpha K40"),
               Line2D([], [], ls="--", color=MUTED, label="diagonal: cell-type mix explains nothing"),
               Line2D([], [], ls="-", color=INK_2, label="residual 1: cell-type mix explains all")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7)
    fig.suptitle("D28B. One point per stored basis (ss-weighted energy, cohort cell-type averages)",
                 fontsize=8.5, x=0.0, ha="left", fontweight="bold")
    save(fig, figdir, "D28B_p01_hp_observed_vs_residual", written)


def fig_d28c(cts: pd.DataFrame, cells: Cells, figdir: Path, written: list) -> None:
    """Mechanism: cell-type energy level, each patient's cell-type mix, within-type ratio."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    sub = cts.loc[cts["scope"].eq("all") & cts["reference"].eq("cohort")]
    order = _unit_order(cells)
    n_rep = int(sub["n_representations"].max())
    lvl = sub.loc[sub["unit"].eq(order[0])].set_index("cell_type")
    types = list(lvl.sort_values("type_energy_level_median", ascending=False).index)
    comp = sub.pivot(index="unit", columns="cell_type", values="comp_share").reindex(index=order, columns=types)
    within = sub.pivot(index="unit", columns="cell_type", values="within_ratio_median").reindex(index=order, columns=types)
    ncell = sub.pivot(index="unit", columns="cell_type", values="n_cells").reindex(index=order, columns=types)
    fig = plt.figure(figsize=(6.5, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 2.2, 2.2], width_ratios=[1.0, 0.025])
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(types))
    med = lvl.loc[types, "type_energy_level_median"].to_numpy(float)
    q25 = lvl.loc[types, "type_energy_level_q25"].to_numpy(float)
    q75 = lvl.loc[types, "type_energy_level_q75"].to_numpy(float)
    colors = [DIV_POS if v > 1 else DIV_NEG for v in med]
    ax.bar(x, med - 1.0, bottom=1.0, width=0.62, color=colors, zorder=2)
    ax.vlines(x, q25, q75, color=INK, lw=1.1, zorder=3)
    ax.axhline(1.0, color=INK_2, lw=0.9, zorder=1)
    ax.set_yscale("log", base=2)
    yt = [t for t in (0.5, 0.71, 1, 1.41, 2, 4, 8) if min(q25.min(), 0.9) / 1.2 <= t <= max(q75.max(), 1.1) * 1.2]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:g}" for t in yt])
    ax.minorticks_off()
    ax.set_ylim(min(q25.min(), 0.9) / 1.2, max(q75.max(), 1.1) * 1.2)
    ax.set_xlim(-0.5, len(types) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=35, ha="right")
    ax.set_ylabel("energy per cell /\ncohort's (log)")
    ax.set_title(f"(a) Which cell types carry score energy: median over {n_rep} bases (line: IQR)", fontsize=8.5)
    seq = LinearSegmentedColormap.from_list("d28_seq", SEQ_BLUE)
    div = LinearSegmentedColormap.from_list("d28_div", [DIV_NEG, DIV_MID, DIV_POS])
    nc = ncell.to_numpy(float)
    ylabels = [u if u != HP_UNIT else "P02+P03+P04" for u in order]
    for row, mat, cmap, vmin, vmax, fmt, title, logc in (
        (1, 100 * comp.to_numpy(float), seq, 0, 60, lambda v: f"{v:.0f}",
         "(b) Each patient's cell-type mix (% of its cells); composition ratio = this row's average of (a)", False),
        (2, within.to_numpy(float), div, -1, 1, lambda v: f"{v:.2f}",
         "(c) Within-type ratio: patient's energy per cell / cohort's, same cell type", True),
    ):
        ax = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])
        show = np.where(nc >= 20, np.log2(mat), np.nan) if logc else mat
        im = ax.imshow(np.clip(show, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_facecolor(GRID)
        for (r, c), v in np.ndenumerate(mat):
            if not np.isfinite(v) or (logc and nc[r, c] < 20):
                continue
            dark = (not logc and v > 35) or (logc and abs(np.log2(v)) > 0.7)
            ax.text(c, r, fmt(v), ha="center", va="center", fontsize=5.6, color=SURFACE if dark else INK)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=35, ha="right")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(ylabels)
        for lab, u in zip(ax.get_yticklabels(), order):
            if u in cells.high_prevalence or u == HP_UNIT:
                lab.set_fontweight("bold")
            if u not in cells.supported and u != HP_UNIT:
                lab.set_color(MUTED)
        ax.grid(False)
        ax.set_title(title, fontsize=8.5)
        cb = fig.colorbar(im, cax=cax)
        if logc:
            cb.set_ticks([-1, 0, 1])
            cb.set_ticklabels(["0.5x", "1x", "2x"])
            cb.set_label("median over bases; blank: < 20 cells", fontsize=7)
        else:
            cb.set_label("% of the patient's cells", fontsize=7)
    fig.suptitle("D28C. Where the composition expectation comes from (all cells, ss-weighted energy, cohort averages);\n"
                 "the residual ratio is the expected-energy-weighted average of row (c)", fontsize=8.5, x=0.0,
                 ha="left", fontweight="bold")
    save(fig, figdir, "D28C_celltype_mechanism", written)


# --------------------------------------------------------------------------------------
# findings, manifest, ledger
# --------------------------------------------------------------------------------------


def _q(dist: pd.DataFrame, unit: str, scope: str, weighting: str = "ss_weighted", reference: str = "cohort",
       strata: str = "celltype") -> pd.Series | None:
    m = dist.loc[dist["unit"].eq(unit) & dist["scope"].eq(scope) & dist["weighting"].eq(weighting)
                 & dist["reference"].eq(reference) & dist["strata"].eq(strata)]
    return None if m.empty else m.iloc[0]


def _dist_line(r: pd.Series) -> str:
    n = int(r["n_representations"])
    return (f"observed median {r['obs_ratio_median']:.2f} (IQR {r['obs_ratio_q25']:.2f}-{r['obs_ratio_q75']:.2f}; "
            f"> 1 in {int(r['obs_ratio_n_gt1'])}/{n}); composition-only expected {r['comp_ratio_median']:.2f} "
            f"(IQR {r['comp_ratio_q25']:.2f}-{r['comp_ratio_q75']:.2f}); residual {r['resid_ratio_median']:.2f} "
            f"(IQR {r['resid_ratio_q25']:.2f}-{r['resid_ratio_q75']:.2f}; range {r['resid_ratio_min']:.2f}-"
            f"{r['resid_ratio_max']:.2f}; > 1 in {int(r['resid_ratio_n_gt1'])}/{n}); residual closer to 1 than "
            f"observed in {int(r['n_resid_closer_to_1'])}/{n}; on the {int(r['n_far_from_1'])} bases with the observed "
            f"ratio at least 5% from 1, the cell-type mix accounts for a median "
            f"{100 * r['frac_log_departure_composition_median']:.0f}% (IQR {100 * r['frac_log_departure_composition_q25']:.0f}% to "
            f"{100 * r['frac_log_departure_composition_q75']:.0f}%) of the log departure")


def write_findings(path: Path, rp: pd.DataFrame, dist: pd.DataFrame, mk: pd.DataFrame, cts: pd.DataFrame,
                   tops: pd.DataFrame, fl: pd.DataFrame, checks: pd.DataFrame, cells: Cells, missing: list,
                   ctx: dict) -> None:
    L: list = []
    add = L.append
    reps = rp[["stage0_panel_id", "stage1_method", "k"]].drop_duplicates()
    add(f"# D28 cell-type-stratified score-energy shares per patient ({SLUG})")
    add("")
    add(f"- Analysis: D28 ({TASK}). Generated {ctx['created_utc']} by "
        "scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.py "
        f"at git {ctx['git_head']}{' (dirty tree)' if ctx['git_dirty'] else ''}.")
    add(f"- Coverage: {len(reps)} stored transductive Stage 1 bases = {reps['stage0_panel_id'].nunique()} biological "
        f"panels x {{{', '.join(sorted(reps['stage1_method'].unique()))}}} x K in "
        f"{{{', '.join(str(k) for k in sorted(reps['k'].unique()))}}}, seed 42 (the D16 grid; {len(missing)} grid "
        "point(s) have no stored scores because the panel has fewer genes than K).")
    add(f"- Cell types: {CELLTYPE_COLUMN} ({len(cells.celltypes)} labels, every cell labelled). Claim level: descriptive.")
    add("")
    add("## What the numbers mean (read this first)")
    add("- **Score energy** of a cell = the sum, over the basis's non-degenerate factors, of the squared distance of "
        "its factor score from the cohort mean. Primary (ss_weighted, the D16 definition): raw scores, so a "
        "high-variance factor counts more. Sensitivity (standardized): each factor is first divided by its cohort SD, "
        "so every factor counts equally.")
    add("- A patient's **energy share** = its cells' summed energy / the cohort's. Its **cell share** = its cells / all "
        "cells. **Ratio** = energy share / cell share = the patient's average energy per cell / the cohort's. "
        "1 = proportional; 1.24 = its cells sit, on average, 24% further (in squared distance) from the cohort centre.")
    add("- **Composition-only expectation**: give every cell the average energy of its predicted cell type (averaged "
        "over the cohort's cells of that type), and add these up over the patient's cells. Its ratio to the cell "
        "share (the **composition ratio**) is what the patient's cell-type mix alone predicts.")
    add("- **Residual ratio** = observed energy / composition-only expected energy = observed ratio / composition "
        "ratio (indirect standardization, like a standardized mortality ratio). 1 = the cell-type mix explains the "
        "patient's ratio entirely; the observed ratio pulled toward 1 = it explains part of it.")
    add("- **Normal cells only**: the same, using only normal cells (energy still measured from the all-cell cohort "
        "mean, as in D16). This removes the malignant cells, which P02/P03/P04 hold disproportionately.")
    own = cts.loc[cts["scope"].eq("all") & cts["reference"].eq("cohort") & ~cts["unit"].eq(HP_UNIT)]
    own = own.loc[own["unit_share_of_type"] > 0.5].sort_values("unit_share_of_type", ascending=False)
    dominated = "; ".join(f"{r.unit} holds {100 * r.unit_share_of_type:.0f}% of the {r.cell_type} cells"
                          for r in own.itertuples(index=False))
    add("- **Reference**: `cohort` averages each cell type over all cells including the patient's own (the "
        "definition above); `loo` averages over the other patients' cells only (a stratum no other patient has falls "
        "back to the cohort average; counted as fallback cells). They differ when one patient holds most cells of a "
        "type, because the cohort average of that type then largely is that patient"
        + (f" (here: {dominated})." if dominated else "."))
    add("- **Fraction of the log departure explained**: per basis, log(observed) = log(composition) + log(residual); "
        "log(composition) / log(observed) is the share of the departure from 1 that the cell-type mix accounts for "
        "(computed on bases where the observed ratio is at least 5% from 1).")
    add("- **Strata**: `celltype` (13 predicted types) or `celltype_label` (type x malignant label, all cells only).")
    add("- Nothing is refit; no Stage 2 model; c_policy not applicable. Score space, not loading leverage.")
    add("")
    add("## Results")
    add("### Headline (ss_weighted energy, cohort cell-type averages)")
    for unit in ("P01", HP_UNIT):
        for scope in ("all", "normal"):
            r = _q(dist, unit, scope)
            if r is not None:
                add(f"- {unit}, {'all cells' if scope == 'all' else 'normal cells only'}, over {int(r['n_representations'])} "
                    f"bases: {_dist_line(r)}.")
    pa = _primary(rp, "all")
    pn = _primary(rp, "normal")
    foc = []
    for method in METHODS:
        for unit in ("P01", HP_UNIT):
            ra = pa.loc[pa["stage0_panel_id"].eq(FOCUS_PANEL) & pa["stage1_method"].eq(method) & pa["k"].eq(40) & pa["unit"].eq(unit)]
            rn = pn.loc[pn["stage0_panel_id"].eq(FOCUS_PANEL) & pn["stage1_method"].eq(method) & pn["k"].eq(40) & pn["unit"].eq(unit)]
            if len(ra) and len(rn):
                a, b = ra.iloc[0], rn.iloc[0]
                foc.append(f"- IFN-alpha {method.upper()}40, {unit}: all cells {a['obs_ratio']:.2f} -> composition "
                           f"{a['comp_ratio']:.2f} -> residual {a['resid_ratio']:.2f} (energy share {100 * a['obs_share']:.1f}% vs "
                           f"expected {100 * a['exp_share']:.1f}% vs cells {100 * a['cell_share']:.1f}%); normal cells only "
                           f"{b['obs_ratio']:.2f} -> {b['comp_ratio']:.2f} -> {b['resid_ratio']:.2f}.")
    if foc:
        add("### IFN-alpha K40 (observed -> composition-only -> residual)")
        L.extend(foc)
    add("")
    add("### All patients (all cells; normal cells only)")
    for unit in _unit_order(cells):
        ra, rn = _q(dist, unit, "all"), _q(dist, unit, "normal")
        if ra is None or rn is None:
            continue
        add(f"- {unit}: observed {ra['obs_ratio_median']:.2f} -> composition {ra['comp_ratio_median']:.2f} -> residual "
            f"{ra['resid_ratio_median']:.2f}; normal only {rn['obs_ratio_median']:.2f} -> {rn['comp_ratio_median']:.2f} -> "
            f"{rn['resid_ratio_median']:.2f} (medians over bases; medians do not multiply exactly).")
    add("")
    add("### Sensitivity (medians over bases; P01 | P02+P03+P04)")
    for weighting in WEIGHTINGS:
        for scope, strata in VARIANTS:
            for reference in REFERENCES:
                a, b = _q(dist, "P01", scope, weighting, reference, strata), _q(dist, HP_UNIT, scope, weighting, reference, strata)
                if a is None or b is None:
                    continue
                add(f"- {weighting}, {scope} cells, {strata} strata, {reference} reference: P01 observed "
                    f"{a['obs_ratio_median']:.2f} -> residual {a['resid_ratio_median']:.2f} (> 1 in {int(a['resid_ratio_n_gt1'])}/"
                    f"{int(a['n_representations'])}; fallback cells max {int(a['n_fallback_cells_max'])}) | P02+P03+P04 "
                    f"{b['obs_ratio_median']:.2f} -> {b['resid_ratio_median']:.2f} (> 1 in {int(b['resid_ratio_n_gt1'])}/"
                    f"{int(b['n_representations'])}).")
    add("")
    add("### By method x K (primary, all cells; medians)")
    for unit in ("P01", HP_UNIT):
        m = mk.loc[mk["unit"].eq(unit) & mk["scope"].eq("all") & mk["weighting"].eq("ss_weighted")
                   & mk["reference"].eq("cohort") & mk["strata"].eq("celltype")]
        add(f"- {unit}: " + "; ".join(f"{r.stage1_method.upper()}{int(r.k)} {r.obs_ratio_median:.2f} -> {r.resid_ratio_median:.2f}"
                                      for r in m.itertuples(index=False)) + ".")
    add("")
    add("### Where the ratios come from (all cells, medians over bases)")
    lv = cts.loc[cts["unit"].eq(_unit_order(cells)[0]) & cts["scope"].eq("all") & cts["reference"].eq("cohort")]
    lv = lv.sort_values("type_energy_level_median", ascending=False)
    add("- Cell-type energy level (a type's energy per cell / the cohort's energy per cell; median, IQR over bases): "
        + ", ".join(f"{r.cell_type} {r.type_energy_level_median:.2f} ({r.type_energy_level_q25:.2f}-"
                    f"{r.type_energy_level_q75:.2f})" for r in lv.itertuples(index=False)) + ".")
    for unit in ("P01", HP_UNIT):
        c1 = cts.loc[cts["unit"].eq(unit) & cts["scope"].eq("all") & cts["reference"].eq("cohort")]
        c2 = cts.loc[cts["unit"].eq(unit) & cts["scope"].eq("all") & cts["reference"].eq("loo")].set_index("cell_type")
        if c1.empty:
            continue
        c1 = c1.sort_values("comp_share", ascending=False)
        add(f"- {unit} cell-type mix (top 6 by % of its cells; the type's energy level; {unit}'s within-type ratio vs "
            f"the cohort average | vs the other patients' average (loo); {unit}'s share of that type's cells): " + "; ".join(
                f"{r.cell_type} {100 * r.comp_share:.1f}% (level {r.type_energy_level_median:.2f}; within "
                f"{r.within_ratio_median:.2f} | loo {c2.loc[r.cell_type, 'within_ratio_median']:.2f}; holds "
                f"{100 * r.unit_share_of_type:.0f}%)" for r in c1.head(6).itertuples(index=False)) + ".")
    add("")
    add("### Top-enrichment patient per factor, before vs after (cohort reference, cell-type strata)")
    for scope in ("all", "normal"):
        t = tops.loc[tops["scope"].eq(scope) & tops["stage1_method"].eq("pooled")]
        if t.empty:
            continue
        o = t.loc[t["which"].eq("observed")].set_index("patient")
        r = t.loc[t["which"].eq("residual")].set_index("patient")
        shown = [p for p in ("P01", "P02", "P03", "P04", "P07", "P09", "P13") if p in o.index]
        add(f"- Pooled over {int(o['n_factors'].iloc[0])} factors ({scope} cells): " + ", ".join(
            f"{p} {100 * o.loc[p, 'frac_top']:.1f}% -> {100 * r.loc[p, 'frac_top']:.1f}%" for p in shown) + ".")
    pats = set(cells.patients)
    for method in METHODS:
        f = fl.loc[fl["stage0_panel_id"].eq(FOCUS_PANEL) & fl["stage1_method"].eq(method) & fl["k"].eq(40)
                   & fl["scope"].eq("all") & fl["unit"].isin(pats)]
        if f.empty:
            continue
        txt = []
        for col, label in (("obs_enrichment", "before"), ("resid_enrichment", "after")):
            g = f.loc[f[col].notna()]
            top = g.loc[g.groupby("factor_index")[col].idxmax(), "unit"].value_counts()
            txt.append(f"{label} " + ", ".join(f"{p} {int(c)}" for p, c in top.items()))
        add(f"- IFN-alpha {method.upper()}40 ({f['factor_index'].nunique()} non-degenerate factors, all cells), factors "
            f"topped: " + "; ".join(txt) + ".")
    add("- Caveat: a per-factor ratio is noisier for a small patient (fewer cells), and the top patient is a maximum "
        "over 13 noisy ratios, so once a large systematic excess is removed (P01 pooled, P02 on IFN-alpha K40), the "
        "smallest patients win more often. Read the drop of the former winner, not the new winners.")
    add("")
    add("## Regression checks")
    if checks.empty:
        add("- none run")
    else:
        add("| check | metric | n | max abs diff | tol | status |")
        add("|---|---|---|---|---|---|")
        for c in checks.itertuples(index=False):
            add(f"| {c.check_id} | {c.metric} | {c.n_compared} | {c.max_abs_diff:.3g} | {c.tolerance:g} | {c.status} |")
    add("")
    add("## Caveats (claim gate)")
    add("- Descriptive and transductive: one stored seed-42 fit per basis; no null distribution, no interval.")
    add("- Score-space allocation, not loading leverage (that is D17/D26). A residual ratio near 1 says the patient's "
        "spread matches other cells of the same predicted type; it does not say the type caused the spread.")
    add("- Predicted cell types are label-transfer calls (with their own error); malignant cells carry the same "
        "predicted labels as normal cells, so annotation x label strata are sparse for some types.")
    add("- When a patient holds most cells of a type, the cohort reference absorbs part of that patient's own "
        "excess (see the loo rows). Medians over bases do not multiply exactly (observed = composition x residual "
        "holds per basis).")
    add("- Cell-type composition is one confound; batch, sample timing and patient biology remain entangled in the "
        "residual.")
    add("")
    add("## Files")
    for name in ctx["artifacts"]:
        add(f"- {name}")
    path.write_text("\n".join(L) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_ledger(exp: Path, base: Path, path: Path, artifacts: list, created: str) -> None:
    rows = []
    for a in artifacts:
        kind = "figure" if a.suffix in (".png", ".pdf") else ("findings" if a.suffix == ".md" else
                                                              ("manifest" if a.suffix == ".json" else "table"))
        n_rows = int(len(pd.read_csv(a, usecols=[0]))) if a.name.endswith((".csv", ".csv.gz")) else None
        rows.append({"artifact_relpath": relname(exp, base, a), "kind": kind, "analysis_id": ANALYSIS_ID, "slug": SLUG,
                     "n_rows": n_rows, "bytes": a.stat().st_size, "sha256": sha256(a), "created_utc": created})
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------


def merge(exp: Path, base: Path, dirs: dict, cells: Cells, provenance: dict, t_start: float, n_computed: int,
          figures: bool = True) -> dict:
    paths = sorted(dirs["parts"].glob("*.npz"))
    parts, bad = [], []
    for p in paths:
        st = read_part(p, cells)
        (parts.append(st) if st is not None else bad.append(p.name))
    if bad:
        raise RuntimeError(f"incomplete part files (rerun with --resume): {bad}")
    if not parts:
        raise RuntimeError("no part files to merge")
    t_merge = time.time()
    rp_frames, fl_frames, ct_frames = [], [], []
    for st in parts:
        meta = st["meta"]
        key = {"stage0_panel_id": meta["stage0_panel_id"], "short_panel_label": meta["short_panel_label"],
               "stage1_method": meta["stage1_method"], "k": int(meta["k"])}
        rp_frames.append(rep_patient_rows(key, st, cells))
        fl_frames.append(factor_rows({k: key[k] for k in ("stage0_panel_id", "stage1_method", "k")}, st, cells))
        ct_frames.append(celltype_rows({k: key[k] for k in ("stage0_panel_id", "stage1_method", "k")}, st, cells))
    sort = ["stage0_panel_id", "stage1_method", "k"]
    rp = pd.concat(rp_frames, ignore_index=True).sort_values(sort, kind="stable").reset_index(drop=True)
    fl = pd.concat(fl_frames, ignore_index=True).sort_values(sort + ["factor_index"], kind="stable").reset_index(drop=True)
    ct = pd.concat(ct_frames, ignore_index=True).sort_values(sort, kind="stable").reset_index(drop=True)
    _, missing = d16.plan_units(exp, sorted(rp["stage0_panel_id"].unique()), sorted(rp["stage1_method"].unique()),
                                sorted(int(k) for k in rp["k"].unique()))
    dist = distribution_summary(rp)
    mk = method_k_summary(rp)
    cts = celltype_summary(ct)
    tops = top_patient_before_after(fl, cells.patients)
    checks = regression_checks(exp, rp, fl, ct, cells)

    td, fd = dirs["tables"], dirs["figures"]
    artifacts: list = []

    def wcsv(frame: pd.DataFrame, name: str) -> None:
        p = td / name
        frame.to_csv(p, index=False, compression="gzip" if name.endswith(".gz") else None)
        artifacts.append(p)

    wcsv(rp, f"{PREFIX}_representation_patient.csv.gz")
    wcsv(dist, f"{PREFIX}_distribution_summary.csv")
    wcsv(mk, f"{PREFIX}_method_k_summary.csv")
    wcsv(ct, f"{PREFIX}_celltype_long.csv.gz")
    wcsv(cts, f"{PREFIX}_celltype_summary.csv")
    wcsv(fl, f"{PREFIX}_factor_long.csv.gz")
    wcsv(tops, f"{PREFIX}_top_patient_before_after.csv")
    wcsv(checks, f"{PREFIX}_regression_checks.csv")
    if figures:
        fig_d28a(dist, cells, fd, artifacts)
        if rp["unit"].eq("P01").any():
            fig_d28b(rp, fd, artifacts)
        fig_d28c(cts, cells, fd, artifacts)

    head, dirty = git_state(REPO)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    findings = td / f"{PREFIX}_findings.md"
    manifest_path = td / f"{PREFIX}_manifest.json"
    ledger = td / f"{PREFIX}_artifact_ledger.csv"
    names = [relname(exp, base, a) for a in artifacts] + [relname(exp, base, p) for p in (findings, manifest_path, ledger)]
    ctx = {"created_utc": created, "git_head": head, "git_dirty": dirty, "artifacts": names}
    write_findings(findings, rp, dist, mk, cts, tops, fl, checks, cells, missing, ctx)
    artifacts.append(findings)

    unit_seconds = [float(st["meta"].get("unit_seconds", np.nan)) for st in parts]
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "slug": SLUG,
        "workstream_task": TASK,
        "created_utc": created,
        "git_head": head,
        "git_dirty": dirty,
        "experiment_id": exp.name,
        "output_base": "EXP/analysis" if relname(exp, base, td).startswith("analysis/") else "<out-root>",
        "command": "python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.py "
                   + " ".join("<path>" if a.startswith(("/", "~")) else a for a in sys.argv[1:]),
        "inputs": {
            "stage1_scores": "stage1_dr/panel_id=<panel>/method={pca,fa}/k=<K>/seed=42/scores.npy (+ metadata.json)",
            "panel_list": d16.ATLAS_RELPATH + " (atlas_row_block == biological)",
            **provenance,
            "regression_d16": D16_TABLES + "/ (representation_patient_summary, representation_summary, long)",
        },
        "parameters": {
            "methods": sorted(rp["stage1_method"].unique().tolist()),
            "ks": sorted(int(k) for k in rp["k"].unique()),
            "seed": SEED,
            "degenerate_sd": fpm.DEGENERATE_SD,
            "energy_center": "cohort mean of each factor (all cells), as D16",
            "weightings": {"ss_weighted": "w_f = 1 (D16 definition)", "standardized": "w_f = 1 / cohort variance of factor f"},
            "references": {"cohort": "cell-type average over all cells in scope", "loo": "average over the other patients' cells; cohort fallback"},
            "variants": [f"{s}|{t}" for s, t in VARIANTS],
            "primary": PRIMARY,
            "high_prevalence_group": list(cells.high_prevalence),
            "support_rule": f">= {MIN_MALIGNANT_SUPPORT} malignant cells and both classes",
            "c_policy": "not_applicable (no Stage 2 fit)",
        },
        "units": {
            "n_merged": len(parts),
            "n_computed_this_invocation": int(n_computed),
            "missing_inputs": [f"{u['panel']}|{u['method']}|K{u['k']}" for u in missing],
            "panels": int(rp["stage0_panel_id"].nunique()),
        },
        "runtime_seconds": {
            "sum_unit_compute": round(float(np.nansum(unit_seconds)), 1),
            "this_invocation_wall": round(time.time() - t_start, 1),
            "merge_and_figures": round(time.time() - t_merge, 1),
        },
        "rows": {"representation_patient": int(len(rp)), "factor_long": int(len(fl)), "celltype_long": int(len(ct))},
        "regression": {"n_checks": int(len(checks)), "n_pass": int((checks["status"] == "pass").sum()),
                       "n_fail": int((checks["status"] == "fail").sum())},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    history = []
    if manifest_path.is_file():
        try:
            history = list(json.loads(manifest_path.read_text()).get("invocations", []))
        except (ValueError, OSError):
            history = []
    history.append({"utc": created, "command": manifest["command"], "git_head": head, "git_dirty": dirty,
                    "n_units_computed": int(n_computed), "wall_seconds": manifest["runtime_seconds"]["this_invocation_wall"]})
    manifest["invocations"] = history
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    artifacts.append(manifest_path)
    write_ledger(exp, base, ledger, artifacts, created)
    stale = dirs["cache"] / "merge_failed.json"
    if stale.exists():
        stale.unlink()
    return {"checks": checks, "dist": dist, "manifest": manifest}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", choices=("pilot", "all"), default="all",
                    help="all = 41 biological panels (default); pilot = IFN-alpha + allograft + apoptosis + TNFa/NF-kB")
    ap.add_argument("--panel-ids", default="", help="comma-separated stage0_panel_id list (overrides --panels)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--workers", type=int, default=1, help="processes (capped at 12)")
    ap.add_argument("--resume", action="store_true", help="skip units whose part file is complete")
    ap.add_argument("--merge-only", action="store_true", help="only rebuild merged outputs from existing parts")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--experiment-dir", default="", help="override the experiment dir (default: paths.experiment_dir())")
    ap.add_argument("--out-root", default="", help="output base replacing EXP/analysis (smoke runs: a scratch dir)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    exp = Path(args.experiment_dir).resolve() if args.experiment_dir else EXP_DEFAULT
    if not exp.is_dir():
        experiment_dir()  # raises with the SC_EXPERIMENT_ROOT / SC_EXPERIMENT_ID hint
    base = Path(args.out_root).resolve() if args.out_root else exp / "analysis"
    dirs = out_dirs(base)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cells, provenance = load_cells(exp)
    if args.panel_ids:
        panels = [p.strip() for p in args.panel_ids.split(",") if p.strip()]
    elif args.panels == "all":
        panels = d16.biological_panels(exp)["stage0_panel_id"].tolist()
    else:
        panels = list(PILOT_PANELS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    units, missing = d16.plan_units(exp, panels, methods, ks)
    n_computed = 0
    if not args.merge_only:
        finished = [u for u in units if read_part(part_path(dirs["parts"], u), cells) is not None]
        if finished and not args.resume:
            print(f"{len(finished)} unit(s) already have complete parts under {relname(exp, base, dirs['parts'])}; "
                  "pass --resume to skip them or --merge-only to rebuild outputs.", file=sys.stderr)
            return 2
        todo = [u for u in units if u not in finished]
        print(f"units: {len(units)} requested, {len(finished)} finished, {len(todo)} to run, "
              f"{len(missing)} without stored scores", flush=True)
        run_units(exp, todo, dirs["parts"], cells, max(1, min(int(args.workers), 12)))
        n_computed = len(todo)
    try:
        res = merge(exp, base, dirs, cells, provenance, t_start, n_computed, figures=not args.no_figures)
    except Exception:
        (dirs["cache"] / "merge_failed.json").write_text(json.dumps(
            {"failed_utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()}, indent=2))
        print("merge failed; see merge_failed.json in the cache dir, fix, then rerun with --merge-only", file=sys.stderr)
        raise
    checks = res["checks"]
    print(checks.to_string(index=False) if len(checks) else "no regression checks")
    print(f"merged {res['manifest']['units']['n_merged']} units -> {relname(exp, base, dirs['tables'])}")
    return 1 if len(checks) and (checks["status"] == "fail").any() else 0


if __name__ == "__main__":
    sys.exit(main())
