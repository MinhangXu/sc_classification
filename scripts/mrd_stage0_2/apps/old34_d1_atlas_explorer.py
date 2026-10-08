#!/usr/bin/env python3
"""Old34 D1 transfer-atlas explorer: a local, read-only Dash app.

Interactive version of the D1 atlas for experiment
``20260525_060508_stage0_mrd_old34_broad_screen_82db5093``.  Rows are the 47 Stage 0
panels (frozen D1 sort), columns the 17 Stage 1 configurations, and colour is a
Stage 2 leave-one-patient-out (LOPO) result summarised over the seven supported
patients.  The app only *reads* completed scorecards: it never refits Stage 1/2 and
never writes next to the diagnostic CSVs.

Run (needs dash, plotly, pandas, numpy -- all already listed in environment-*.yml)::

    python old34_d1_atlas_explorer.py                       # http://127.0.0.1:8050
    python old34_d1_atlas_explorer.py --debug               # hot reload while iterating
    python old34_d1_atlas_explorer.py --check               # load + validate, no server
    python old34_d1_atlas_explorer.py --export-bundle DIR   # ~2 MB copy to share with teammates
    python old34_d1_atlas_explorer.py --data-dir DIR        # an experiment dir or an exported bundle

Inputs, all under ``<experiment>/analysis/scorecards`` (or flat inside a bundle):

* ``stage2_diagnostic/old34_d1_atlas_representation_table.csv``      783 cells (one per panel x configuration)
* ``stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`` 5481 rows (783 cells x 7 patients)
* ``stage2_figure3_sharedness_v2_jun4/v2_patient_leverage_on_panel_method_median_lift.csv`` (optional)
* four ``stage2_sharedness_lopo_scorecard.csv`` files, only to join the Stage 2 nonzero-coefficient
  count at each cell's *selected* regularization (optional; a bundle carries the joined 783-row table).

Reading guide
-------------
``lift = held-out AUPRC - that patient's own malignant prevalence`` (shown in AUPRC percentage
points, pp).  "Cohort" means the seven supported LOPO patients: median / mean / 20th percentile
of their seven lifts.  Lift is descriptive, not a p-value, and Stage 1 is transductive.

To change the design: colours, row height and default thresholds are constants near the top; each
colour mode is one branch of ``color_spec``; each filter is one entry in ``filter_masks``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape as esc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================================================================
# 1. Constants (design knobs live here)
# ======================================================================================

EXPERIMENT_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
NONZERO_TABLE_NAME = "old34_d1_atlas_stage2_nonzero_at_selected_regularization.csv"
LEVERAGE_SCOPE = "panel_method_best_lopo"

METHODS = ("pca", "fa", "factosig", "factosig_promax")
KS = (5, 10, 20, 40)
CONFIGS = ("direct_gene",) + tuple(f"{m}|K={k}" for m in METHODS for k in KS)
METHOD_LABEL = {
    "direct_gene": "direct gene",
    "pca": "PCA",
    "fa": "FA",
    "factosig": "FactoSig",
    "factosig_promax": "FactoSig-promax",
}
# Vertical method-block rules sit between column 0|1, 4|5, 8|9 and 12|13 (as in D1).
COL_BOUNDARIES = (0.5, 4.5, 8.5, 12.5)

# D1 puts the six non-geneset controls first, in this order.
CONTROL_PANEL_IDS = (
    "full_34",
    "core_only",
    "hvg_top_requested_500__available_500",
    "hvg_top_requested_1000__available_1000",
    "hvg_top_requested_3000__available_3000",
    "hvg_top_requested_10000__available_10000",
)
CLASS_BY_TYPE = {
    "hvg_anchor_control": "HVG",
    "full_control": "UNION",
    "core_only": "UNION",
    "single_group_only": "GROUP",
    "single_geneset_only": "SET",
}
CLASS_ORDER = ("HVG", "UNION", "GROUP", "SET", "OTHER")

# A lift of -2.7e-17 is what an all-zero model gives (AUPRC == prevalence); never "above" a cutoff.
EPS = 1e-9  # lift fraction
EPS_PP = EPS * 100

# Defaults for the editable controls.
DEFAULT_TAU_PP = 5.0  # "lift above" cutoff, AUPRC pp (D1f used 0 and 5)
DEFAULT_RHO = 2.0  # "AUPRC / prevalence above" cutoff (D1f used 2 and 3)
DEFAULT_NEAR_ZERO_PP = 1.0  # direct gene counts as "near zero" when |center lift| <= this
DEFAULT_STRONG_DR_PP = 5.0  # a DR cell is "strong" when center lift >= this
DEFAULT_LEVERAGE_PP = 7.5  # "high single-patient leverage": max |drop-one change in median| >= this
DEFAULT_PATIENT = "P02"

ROW_PX = 19
HEATMAP_MARGIN = dict(l=250, r=10, t=118, b=112)
VMAX_FLOOR_PP = 3.0  # D1: vmax = max(98th percentile of |median, p20|, 3 pp)

# vlag (seaborn) sampled at 17 stops, the D1 diverging map; YlOrRd(8) as in D1f.
VLAG = [
    "#2369bd", "#4a7bbc", "#678bbe", "#829cc3", "#9baecb", "#b5c0d4", "#cfd4e0", "#e9e9ed", "#faf5f4",
    "#f3e2e0", "#e7c8c6", "#ddafad", "#d39794", "#c97f7d", "#bf6765", "#b44f4f", "#a9373b",
]
YLORRD8 = ["#ffffcc", "#ffea9a", "#fece6a", "#fea245", "#fc6932", "#ea2920", "#c20425", "#800026"]
GRAY = "#d9d9d9"
PATIENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
BASELINE_GRAY, HELDOUT_BLUE = "#a6a6a6", "#4472C4"
MEDIAN_ORANGE, MEAN_PURPLE, P20_GREEN = "#D55E00", "#9b59b6", "#009E73"

PATIENT_MODES = ("patient_lift", "patient_auprc")
MODE_LABELS = {
    "center": "Cohort lift (median or mean, pp)",
    "p20": "Cohort p20 lift, lower-tail floor (pp)",
    "count_lift": "# patients with lift > tau",
    "count_ratio": "# patients with AUPRC / prevalence > rho",
    "patient_lift": "Selected patient: lift (pp)",
    "patient_auprc": "Selected patient: raw held-out AUPRC",
    "nonzero": "Stage 2 nonzero coefficients (selected regularization)",
}

# (source tag, path under <experiment>/analysis/scorecards, representation_family filter)
SCORECARDS = (
    ("june5_biological_dr", "stage2_sharedness_lopo_scorecard.csv", "dr"),
    ("direct_gene_all_v1", "old34_direct_gene_all_v1/stage2_sharedness_lopo_scorecard.csv", None),
    ("controls_fullgrid_v1", "old34_controls_fullgrid_v1/stage2_sharedness_lopo_scorecard.csv", None),
    ("controls_direct_gene_v1", "old34_controls_direct_gene_v1/stage2_sharedness_lopo_scorecard.csv", None),
)
SCORECARD_COLUMNS = (
    "stage0_panel_id", "representation_family", "stage1_method", "effective_k", "seed", "penalty", "C",
    "l1_ratio", "cell_weighted_auprc", "cell_weighted_auroc",
    "nonzero_coefficient_count_mean_across_folds", "nonzero_coefficient_count_max_across_folds", "status",
)
PENALTY_RANK = {"elasticnet": 0, "l1": 1, "l2": 2}  # D1 tie-break order

REP_COLUMNS = (
    "stage0_panel_id", "stage0_panel_type", "short_panel_label", "panel_label_with_genes", "biological_theme",
    "n_covered_genes", "stage1_method", "effective_k", "configuration", "patient_median_lift",
    "patient_p20_lift", "atlas_row_block",
)
PAT_COLUMNS = (
    "stage0_panel_id", "configuration", "heldout_patient", "heldout_auprc", "heldout_lift",
    "patient_prevalence", "n_malignant_h", "n_normal_h",
)
LEVERAGE_COLUMNS = (
    "summary_scope", "stage0_panel_id", "selected_stage1_method", "selected_effective_k",
    "dropped_patient_id", "full_median_heldout_lift", "patient_leverage_delta", "patient_leverage_abs_delta",
)

# Expected accounting for *this* experiment (used by --check, never by the app itself).
EXPECTED = dict(panels=47, configs=17, cells=783, unavailable=16, patients=7, leverage_cells=164)

# input key -> (filename, subdir below the experiment dir, required)
INPUT_SPECS = {
    "atlas": ("old34_d1_atlas_representation_table.csv", "analysis/scorecards/stage2_diagnostic", True),
    "patients": ("old34_d1_atlas_supported_patient_metrics.csv", "analysis/scorecards/stage2_diagnostic", True),
    "leverage": (
        "v2_patient_leverage_on_panel_method_median_lift.csv",
        "analysis/scorecards/stage2_figure3_sharedness_v2_jun4",
        False,
    ),
    "nonzero_table": (NONZERO_TABLE_NAME, "", False),
}
# Saved diagnostics used only by --check as ground truth (optional).
CHECK_FILES = {
    "median_matrix": "old34_all_representation_patient_median_lift_matrix.csv",
    "d1f_lift_0": "old34_d1f_lift_gt_0_patient_count_matrix.csv",
    "d1f_lift_5": "old34_d1f_lift_gt_0.05_patient_count_matrix.csv",
    "d1f_ratio_2": "old34_d1f_ratio_gt_2_patient_count_matrix.csv",
    "d1f_ratio_3": "old34_d1f_ratio_gt_3_patient_count_matrix.csv",
}


# ======================================================================================
# 2. Small helpers
# ======================================================================================


class DataError(RuntimeError):
    """Raised with an actionable message when an input table is missing or inconsistent."""


def fmt_pp(x: float | None, digits: int = 2, signed: bool = True) -> str:
    """Format a percentage-point value; float noise around zero prints as plain 0."""
    if x is None or not np.isfinite(x):
        return "n/a"
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    return f"{x:+.{digits}f}" if signed and x != 0 else f"{x:.{digits}f}"


def config_label(config: str) -> str:
    return config.replace("|", " ")


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def drivers_text(drivers: str) -> str:
    """'P06/P07/P09' -> 'any of P06/P07/P09' (patients sharing the maximum drop-one shift)."""
    return f"any of {drivers}" if "/" in str(drivers) else str(drivers)


def shorten(label: str, limit: int = 46) -> str:
    """Shorten a long panel label but keep its trailing ``(n genes)``."""
    if len(label) <= limit:
        return label
    tail_match = re.search(r"\s\(\d+\)$", label)
    tail = tail_match.group(0) if tail_match else ""
    return f"{label[: limit - len(tail) - 1].rstrip()}…{tail}"


def _mix_with_white(color: str, keep: float) -> str:
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(keep * c + (1 - keep) * 255) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in mixed)


def _continuous(colors: list[str]) -> list[list]:
    n = len(colors) - 1
    return [[i / n, c] for i, c in enumerate(colors)]


def _discrete(colors: list[str]) -> list[list]:
    n = len(colors)
    out: list[list] = []
    for i, c in enumerate(colors):
        out += [[i / n, c], [(i + 1) / n, c]]
    return out


def _muted(scale: list[list], keep: float = 0.28) -> list[list]:
    return [[pos, _mix_with_white(col, keep)] for pos, col in scale]


VLAG_SCALE = _continuous(VLAG)
COUNT_SCALE = _discrete(YLORRD8)
VIRIDIS_SCALE = _continuous(
    ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"]
)
BLUES_SCALE = _continuous(["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"])


# ======================================================================================
# 3. Data model and loading
# ======================================================================================


@dataclass(frozen=True)
class RowLayout:
    """One way of ordering the 47 rows.  Orders are computed once at load: they are frozen."""

    key: str
    title: str
    order: np.ndarray  # displayed row -> canonical (D1) row
    labels: list[str]  # displayed tick labels, one per displayed row
    boundaries: tuple[float, ...]  # horizontal rules between displayed rows


@dataclass
class Atlas:
    """Everything the app needs, as plain arrays.  Axes: P patients, R panels (D1 order), C configs."""

    data_dir: Path
    panel_ids: list[str]
    panel_labels: list[str]  # panel_label_with_genes, e.g. "stress arrest (685)"
    panel_classes: list[str]
    panel_themes: list[str]
    panel_genes: np.ndarray
    n_controls: int
    patients: list[str]
    prevalence: np.ndarray  # (P,)
    n_malignant: np.ndarray  # (P,)
    n_normal: np.ndarray  # (P,)
    lift: np.ndarray  # (P, R, C) fraction; NaN where the cell is unavailable
    auprc: np.ndarray  # (P, R, C)
    available: np.ndarray  # (R, C)
    median: np.ndarray  # (R, C) fraction, from the representation table
    p20: np.ndarray
    mean: np.ndarray  # mean of the seven patient lifts
    n_features: np.ndarray
    nonzero_mean: np.ndarray  # mean over LOPO folds at the selected regularization (NaN if unknown)
    nonzero_max: np.ndarray
    reg_label: np.ndarray  # (R, C) object, '' if unknown
    lev_d1: np.ndarray  # leave-one-patient-out change of the median, from this atlas's own patient lifts
    lev_d1_driver: np.ndarray
    lev_d1_delta: np.ndarray
    lev_csv: np.ndarray  # same quantity from the v2 leverage CSV (NaN where the CSV has no row)
    lev_csv_driver: np.ndarray
    lev_csv_delta: np.ndarray
    lev_csv_full_median: np.ndarray
    layouts: dict[str, RowLayout]
    d1_vmax_pp: float
    audit: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.available.shape

    @property
    def has_nonzero(self) -> bool:
        return bool(np.isfinite(self.nonzero_mean).any())

    @property
    def has_leverage_csv(self) -> bool:
        return bool(np.isfinite(self.lev_csv).any())

    def center(self, which: str) -> np.ndarray:
        return self.mean if which == "mean" else self.median


def find_experiment_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "experiments" / EXPERIMENT_ID
        if candidate.is_dir():
            return candidate
    return None


def find_inputs(data_dir: Path) -> dict[str, Path | None]:
    """Resolve each input from a flat bundle directory or from the experiment tree."""
    found: dict[str, Path | None] = {}
    for key, (filename, subdir, required) in INPUT_SPECS.items():
        candidates = [data_dir / filename, data_dir / subdir / filename]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None and required:
            searched = "\n  ".join(str(p) for p in candidates)
            raise DataError(f"required input '{filename}' not found; searched:\n  {searched}")
        found[key] = path
    return found


def scorecard_paths(data_dir: Path) -> list[tuple[str, Path, str | None]]:
    base = data_dir / "analysis" / "scorecards"
    return [(tag, base / rel, family) for tag, rel, family in SCORECARDS]


def input_fingerprint(data_dir: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every input that exists: proves nothing was modified."""
    paths = [p for p in find_inputs(data_dir).values() if p is not None]
    paths += [p for _, p, _ in scorecard_paths(data_dir)]
    stage2_dir = data_dir / "analysis" / "scorecards" / "stage2_diagnostic"
    paths += [stage2_dir / name for name in CHECK_FILES.values()]
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()}


def _read_csv(path: Path, required: tuple[str, ...], what: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(f"{what} ({path}) is missing columns: {missing}")
    return frame


def select_global_regularization(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (panel, configuration, seed): the D1 rule for the frozen global regularizer.

    Highest pooled-cell LOPO AUPRC (``cell_weighted_auprc``); ties by pooled AUROC, then penalty
    (elasticnet, l1, l2), then smaller C.  Reproduces the D1 atlas selection exactly (validated
    against the saved per-patient AUPRCs and the v2 selection table).
    """
    work = frame.copy()
    work["_rank"] = work["penalty"].astype(str).str.lower().map(PENALTY_RANK).fillna(len(PENALTY_RANK))
    keys = ["stage0_panel_id", "configuration", "seed"]
    work = work.sort_values(
        keys + ["cell_weighted_auprc", "cell_weighted_auroc", "_rank", "C"],
        ascending=[True, True, True, False, False, True, True],
        kind="mergesort",
    )
    return work.drop_duplicates(keys, keep="first").drop(columns="_rank")


def build_nonzero_table(data_dir: Path) -> pd.DataFrame | None:
    """Join the nonzero-coefficient count at each cell's selected regularization (read-only)."""
    parts = []
    for tag, path, family in scorecard_paths(data_dir):
        if not path.is_file():
            return None
        frame = pd.read_csv(path, usecols=lambda c: c in SCORECARD_COLUMNS, low_memory=False)
        missing = [c for c in SCORECARD_COLUMNS if c != "status" and c not in frame.columns]
        if missing:
            raise DataError(f"scorecard {path} is missing columns: {missing}")
        if "status" in frame.columns:
            frame = frame.loc[frame["status"].eq("ok")]
        if family is not None:
            frame = frame.loc[frame["representation_family"].eq(family)]
        k = pd.to_numeric(frame["effective_k"], errors="coerce").round().astype("Int64").astype(str)
        frame = frame.assign(
            configuration=np.where(frame["stage1_method"].eq("direct_gene"), "direct_gene", frame["stage1_method"] + "|K=" + k),
            n_features=pd.to_numeric(frame["effective_k"], errors="coerce"),
        )
        chosen = select_global_regularization(frame)
        parts.append(chosen.assign(source_scorecard=tag))
    table = pd.concat(parts, ignore_index=True)
    if table.duplicated(["stage0_panel_id", "configuration"]).any():
        raise DataError("more than one scorecard row (or seed) per panel x configuration after selection")
    return table.rename(
        columns={
            "nonzero_coefficient_count_mean_across_folds": "nonzero_mean_across_folds",
            "nonzero_coefficient_count_max_across_folds": "nonzero_max_across_folds",
        }
    )[
        [
            "stage0_panel_id", "configuration", "penalty", "C", "l1_ratio", "n_features",
            "nonzero_mean_across_folds", "nonzero_max_across_folds", "source_scorecard",
        ]
    ]


def _regularization_text(penalty: str, c_value: float, l1_ratio: float) -> str:
    text = f"{penalty} C={c_value:.3g}"
    if str(penalty).lower() == "elasticnet" and np.isfinite(l1_ratio):
        text += f" l1_ratio={l1_ratio:g}"
    return text


def _d1_leverage(
    lift: np.ndarray, available: np.ndarray, patients: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Largest |change in the median| when one patient is dropped, and which patients attain it.

    With an odd number of patients every patient on the same side of the median moves it by the
    same amount, so the maximum is usually shared by several patients; all of them are returned
    ("P06/P07/P09"), sorted by id.  The signed delta is ``full median - drop-one median`` for the
    first of them.
    """
    n_patients = lift.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full = np.nanmedian(lift, axis=0)
        delta = np.stack([full - np.nanmedian(np.delete(lift, p, axis=0), axis=0) for p in range(n_patients)])
    magnitude = np.abs(delta)
    filled = np.where(np.isfinite(magnitude), magnitude, -1.0)
    best, first = filled.max(axis=0), filled.argmax(axis=0)
    signed = np.take_along_axis(delta, first[None], 0)[0]
    drivers = np.full(available.shape, "", dtype=object)
    for r, c in zip(*np.nonzero(available)):
        drivers[r, c] = "/".join(patients[k] for k in range(n_patients) if filled[k, r, c] >= best[r, c] - EPS)
    lev = np.where(available, best, np.nan)
    return lev, drivers, np.where(available, signed, np.nan)


def load_atlas(data_dir: Path, *, with_nonzero: bool = True) -> Atlas:
    """Load, validate and arrange all inputs (read-only)."""
    data_dir = Path(data_dir)
    files = find_inputs(data_dir)
    notes: list[str] = []
    audit: dict = {}

    rep = _read_csv(files["atlas"], REP_COLUMNS, "D1 representation table")
    pat = _read_csv(files["patients"], PAT_COLUMNS, "D1 supported-patient table")
    unknown = sorted(set(rep["configuration"]) - set(CONFIGS))
    if unknown:
        raise DataError(f"unexpected Stage 1 configurations in the atlas table: {unknown}")
    if rep.duplicated(["stage0_panel_id", "configuration"]).any():
        raise DataError("atlas table has more than one row for some panel x configuration")

    # ---- rows: the frozen D1 sort (controls first, then biology by best p20) -------------
    meta = rep.drop_duplicates("stage0_panel_id").set_index("stage0_panel_id")
    agg = rep.groupby("stage0_panel_id").agg(
        best_p20=("patient_p20_lift", "max"),
        best_median=("patient_median_lift", "max"),
        median_median=("patient_median_lift", "median"),
    )
    agg["label"] = meta["panel_label_with_genes"]
    is_control = meta["atlas_row_block"].eq("non_geneset_control")
    unknown_control_rank = len(CONTROL_PANEL_IDS)
    control_ids = sorted(
        meta.index[is_control],
        key=lambda pid: (CONTROL_PANEL_IDS.index(pid) if pid in CONTROL_PANEL_IDS else unknown_control_rank, pid),
    )
    bio_ids = (
        agg.loc[~is_control]
        .sort_values(["best_p20", "best_median", "median_median", "label"], ascending=[False, False, False, True])
        .index.tolist()
    )
    panel_ids = control_ids + bio_ids
    n_panels, n_configs = len(panel_ids), len(CONFIGS)
    labels = [str(meta.loc[p, "panel_label_with_genes"]) for p in panel_ids]
    classes = [CLASS_BY_TYPE.get(str(meta.loc[p, "stage0_panel_type"]), "OTHER") for p in panel_ids]
    short = [shorten(label) for label in labels]
    if len(set(short)) != len(short):
        short = labels
    if len(set(short)) != len(short):
        raise DataError("panel labels are not unique")

    best_p20 = agg.loc[panel_ids, "best_p20"].to_numpy(float)
    grouped_order = sorted(range(n_panels), key=lambda i: (CLASS_ORDER.index(classes[i]), -best_p20[i], labels[i]))
    grouped_labels = [f"[{classes[i]}] {short[i]}" for i in grouped_order]
    grouped_rules = tuple(
        k - 0.5 for k in range(1, n_panels) if classes[grouped_order[k]] != classes[grouped_order[k - 1]]
    )
    layouts = {
        "d1": RowLayout(
            "d1", "Frozen D1 sort", np.arange(n_panels), short, (len(control_ids) - 0.5,) if control_ids else ()
        ),
        "grouped": RowLayout("grouped", "Grouped by panel construction (D1DD1)", np.array(grouped_order), grouped_labels, grouped_rules),
    }

    # ---- cell arrays -----------------------------------------------------------------
    row_of = {pid: i for i, pid in enumerate(panel_ids)}
    col_of = {cfg: j for j, cfg in enumerate(CONFIGS)}
    ri = rep["stage0_panel_id"].map(row_of).to_numpy(int)
    ci = rep["configuration"].map(col_of).to_numpy(int)
    available = np.zeros((n_panels, n_configs), bool)
    available[ri, ci] = True

    def grid(values) -> np.ndarray:
        out = np.full((n_panels, n_configs), np.nan)
        out[ri, ci] = np.asarray(values, float)
        return out

    median = grid(rep["patient_median_lift"])
    p20 = grid(rep["patient_p20_lift"])
    n_features = grid(rep["effective_k"])

    # ---- patients --------------------------------------------------------------------
    patients = sorted(pat["heldout_patient"].astype(str).unique())
    n_pat = len(patients)
    consts = pat.groupby("heldout_patient")[["patient_prevalence", "n_malignant_h", "n_normal_h"]].nunique()
    if int(consts.to_numpy().max()) != 1:
        raise DataError("patient prevalence / cell counts differ between atlas cells; the app assumes constants")
    first = pat.groupby("heldout_patient").first().reindex(patients)
    prevalence = first["patient_prevalence"].to_numpy(float)
    n_mal = first["n_malignant_h"].to_numpy(int)
    n_norm = first["n_normal_h"].to_numpy(int)

    pi = pat["heldout_patient"].map({p: k for k, p in enumerate(patients)}).to_numpy(int)
    pri = pat["stage0_panel_id"].map(row_of)
    pci = pat["configuration"].map(col_of)
    if pri.isna().any() or pci.isna().any():
        raise DataError("patient table has panels/configurations that are not in the atlas table")
    pri, pci = pri.to_numpy(int), pci.to_numpy(int)
    if pat.duplicated(["stage0_panel_id", "configuration", "heldout_patient"]).any():
        raise DataError("patient table has duplicate panel x configuration x patient rows")
    lift = np.full((n_pat, n_panels, n_configs), np.nan)
    auprc = np.full((n_pat, n_panels, n_configs), np.nan)
    lift[pi, pri, pci] = pat["heldout_lift"].to_numpy(float)
    auprc[pi, pri, pci] = pat["heldout_auprc"].to_numpy(float)
    for k in range(n_pat):
        if not np.array_equal(np.isfinite(lift[k]), available):
            raise DataError(f"patient {patients[k]} is not present in exactly the atlas cells")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(lift, axis=0)
        median_check = np.nanmedian(lift, axis=0)
        p20_check = np.nanquantile(lift, 0.2, axis=0)
    audit["median_vs_table_max_abs_diff"] = float(np.nanmax(np.abs(median_check - median)))
    audit["p20_vs_table_max_abs_diff"] = float(np.nanmax(np.abs(p20_check - p20)))

    d1_values = np.concatenate([100 * median[available], 100 * p20[available]])
    d1_vmax = max(float(np.quantile(np.abs(d1_values), 0.98)), VMAX_FLOOR_PP)

    lev_d1, lev_d1_driver, lev_d1_delta = _d1_leverage(lift, available, patients)

    atlas = Atlas(
        data_dir=data_dir,
        panel_ids=panel_ids,
        panel_labels=labels,
        panel_classes=classes,
        panel_themes=[str(meta.loc[p, "biological_theme"]) for p in panel_ids],
        panel_genes=np.array([int(meta.loc[p, "n_covered_genes"]) for p in panel_ids]),
        n_controls=len(control_ids),
        patients=patients,
        prevalence=prevalence,
        n_malignant=n_mal,
        n_normal=n_norm,
        lift=lift,
        auprc=auprc,
        available=available,
        median=median,
        p20=p20,
        mean=mean,
        n_features=n_features,
        nonzero_mean=np.full((n_panels, n_configs), np.nan),
        nonzero_max=np.full((n_panels, n_configs), np.nan),
        reg_label=np.full((n_panels, n_configs), "", dtype=object),
        lev_d1=lev_d1,
        lev_d1_driver=lev_d1_driver,
        lev_d1_delta=lev_d1_delta,
        lev_csv=np.full((n_panels, n_configs), np.nan),
        lev_csv_driver=np.full((n_panels, n_configs), "", dtype=object),
        lev_csv_delta=np.full((n_panels, n_configs), np.nan),
        lev_csv_full_median=np.full((n_panels, n_configs), np.nan),
        layouts=layouts,
        d1_vmax_pp=d1_vmax,
        audit=audit,
        notes=notes,
    )
    atlas.audit["cells"] = int(available.sum())
    atlas.audit["unavailable_cells"] = int((~available).sum())
    if with_nonzero:
        _attach_nonzero(atlas, files, row_of, col_of)
    _attach_leverage(atlas, files, row_of, col_of)
    return atlas


def _attach_nonzero(atlas: Atlas, files: dict, row_of: dict, col_of: dict) -> None:
    table: pd.DataFrame | None = None
    if files.get("nonzero_table") is not None:
        table = pd.read_csv(files["nonzero_table"], low_memory=False)
        atlas.audit["nonzero_source"] = f"bundle table {files['nonzero_table'].name}"
    else:
        table = build_nonzero_table(atlas.data_dir)
        atlas.audit["nonzero_source"] = "joined from the four stage2_sharedness_lopo_scorecard.csv files"
    if table is None:
        atlas.notes.append(
            "Stage 2 nonzero-coefficient counts are unavailable: neither the LOPO scorecards nor a bundle "
            f"'{NONZERO_TABLE_NAME}' were found."
        )
        atlas.audit["nonzero_cells"] = 0
        return
    ri = table["stage0_panel_id"].map(row_of)
    ci = table["configuration"].map(col_of)
    keep = ri.notna() & ci.notna()
    table, ri, ci = table.loc[keep], ri[keep].to_numpy(int), ci[keep].to_numpy(int)
    atlas.nonzero_mean[ri, ci] = table["nonzero_mean_across_folds"].to_numpy(float)
    atlas.nonzero_max[ri, ci] = table["nonzero_max_across_folds"].to_numpy(float)
    atlas.reg_label[ri, ci] = [
        _regularization_text(p, c, l1) for p, c, l1 in zip(table["penalty"], table["C"].astype(float), table["l1_ratio"].astype(float))
    ]
    atlas.audit["nonzero_cells"] = int(np.isfinite(atlas.nonzero_mean).sum())
    uncovered = int((atlas.available & ~np.isfinite(atlas.nonzero_mean)).sum())
    if uncovered:
        atlas.notes.append(f"{uncovered} atlas cells have no nonzero-coefficient count (no scorecard row).")


def _attach_leverage(atlas: Atlas, files: dict, row_of: dict, col_of: dict) -> None:
    path = files.get("leverage")
    if path is None:
        atlas.notes.append("v2 leverage CSV not found: the leverage filter can only use the D1 patient table.")
        atlas.audit["leverage_csv_cells"] = 0
        return
    lv = _read_csv(path, LEVERAGE_COLUMNS, "v2 patient-leverage table")
    lv = lv.loc[lv["summary_scope"].eq(LEVERAGE_SCOPE)].copy()
    lv["configuration"] = lv["selected_stage1_method"].astype(str) + "|K=" + lv["selected_effective_k"].astype(int).astype(str)
    keys = ["stage0_panel_id", "configuration"]
    lv["dropped_patient_id"] = lv["dropped_patient_id"].astype(str)
    lv = lv.sort_values(keys + ["patient_leverage_abs_delta", "dropped_patient_id"], ascending=[True, True, False, True])
    best = lv.groupby(keys)["patient_leverage_abs_delta"].transform("max")
    tied = lv.loc[lv["patient_leverage_abs_delta"] >= best - EPS]
    drivers = tied.groupby(keys)["dropped_patient_id"].apply(lambda s: "/".join(sorted(s))).rename("drivers")
    top = lv.drop_duplicates(keys).join(drivers, on=keys)
    ri = top["stage0_panel_id"].map(row_of)
    ci = top["configuration"].map(col_of)
    keep = ri.notna() & ci.notna()
    atlas.audit["leverage_csv_unmatched_cells"] = int((~keep).sum())
    top, ri, ci = top.loc[keep], ri[keep].to_numpy(int), ci[keep].to_numpy(int)
    atlas.lev_csv[ri, ci] = top["patient_leverage_abs_delta"].to_numpy(float)
    atlas.lev_csv_delta[ri, ci] = top["patient_leverage_delta"].to_numpy(float)
    atlas.lev_csv_full_median[ri, ci] = top["full_median_heldout_lift"].to_numpy(float)
    atlas.lev_csv_driver[ri, ci] = top["drivers"].to_numpy(object)
    covered = np.isfinite(atlas.lev_csv)
    diff = np.abs(atlas.lev_csv_full_median[covered] - atlas.median[covered])
    atlas.audit["leverage_csv_cells"] = int(covered.sum())
    atlas.audit["leverage_csv_basis_mismatch_cells"] = int((diff > EPS).sum())
    atlas.audit["leverage_csv_basis_max_abs_diff_pp"] = float(100 * diff.max()) if diff.size else 0.0
    atlas.audit["leverage_csv_basis_mean_signed_diff_pp"] = (
        float(100 * (atlas.lev_csv_full_median[covered] - atlas.median[covered]).mean()) if diff.size else 0.0
    )


def audit_lines(atlas: Atlas) -> list[str]:
    a = atlas.audit
    rows, cols = atlas.shape
    lines = [
        f"data: {atlas.data_dir}",
        f"{rows} panels x {cols} configurations = {a['cells']} cells, {a['unavailable_cells']} unavailable (gray); "
        f"{len(atlas.patients)} supported patients",
        f"median / p20 recomputed from the patient table match the representation table "
        f"(max |diff| {a['median_vs_table_max_abs_diff']:.1e} / {a['p20_vs_table_max_abs_diff']:.1e})",
        f"Stage 2 nonzero counts: {a.get('nonzero_cells', 0)}/{a['cells']} cells ({a.get('nonzero_source', 'n/a')})",
        f"v2 leverage CSV: {a.get('leverage_csv_cells', 0)}/{a['cells']} cells (best-LOPO K per panel x method; none for "
        f"controls or direct_gene)",
    ]
    if a.get("leverage_csv_cells"):
        lines.append(
            f"  basis caveat: the CSV's full median differs from the D1 median at {a['leverage_csv_basis_mismatch_cells']}/"
            f"{a['leverage_csv_cells']} covered cells (max {a['leverage_csv_basis_max_abs_diff_pp']:.1f} pp; mean "
            f"{a['leverage_csv_basis_mean_signed_diff_pp']:+.1f} pp). It was built with each patient's own best "
            f"regularization, the atlas with one global regularizer per representation."
        )
    lines += atlas.notes
    return lines


# ======================================================================================
# 4. View state, colour modes, filters (pure functions of Atlas + View)
# ======================================================================================


@dataclass(frozen=True)
class View:
    mode: str = "center"
    center: str = "median"  # "median" | "mean": the cohort statistic behind the 'center' mode and filters 1-2
    tau_pp: float = DEFAULT_TAU_PP
    rho: float = DEFAULT_RHO
    patient: str = DEFAULT_PATIENT
    row_order: str = "d1"
    annotate: bool = False
    vmax_pp: float | None = None
    f1: bool = False  # center > 0 and p20 <= 0
    f2: bool = False  # direct gene near zero with a strong DR cell on the same row
    f2_eps_pp: float = DEFAULT_NEAR_ZERO_PP
    f2_delta_pp: float = DEFAULT_STRONG_DR_PP
    f3: bool = False  # high single-patient leverage
    f3_theta_pp: float = DEFAULT_LEVERAGE_PP
    f3_source: str = "csv"  # "csv" (v2 leverage CSV) | "d1" (this atlas's own patient lifts)
    selected: tuple[int, int] | None = None  # (canonical row, column)

    @staticmethod
    def parse(raw: dict, atlas: Atlas) -> "View":
        """Build a View from raw UI values, replacing blanks/invalid input with defaults."""

        def num(key: str, default: float, lo: float = -1e9, hi: float = 1e9) -> float:
            value = raw.get(key)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            return min(max(value, lo), hi) if np.isfinite(value) else default

        vmax = raw.get("vmax_pp")
        try:
            vmax = float(vmax)
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else None
        except (TypeError, ValueError):
            vmax = None
        mode = raw.get("mode") if raw.get("mode") in MODE_LABELS else "center"
        patient = raw.get("patient") if raw.get("patient") in atlas.patients else (
            DEFAULT_PATIENT if DEFAULT_PATIENT in atlas.patients else atlas.patients[0]
        )
        selected = raw.get("selected")
        if selected is not None:
            selected = (int(selected["r"]), int(selected["c"]))
            rows, cols = atlas.shape
            if not (0 <= selected[0] < rows and 0 <= selected[1] < cols and atlas.available[selected]):
                selected = None
        return View(
            mode=mode,
            center="mean" if raw.get("center") == "mean" else "median",
            tau_pp=num("tau_pp", DEFAULT_TAU_PP, -100, 100),
            rho=num("rho", DEFAULT_RHO, 0, 1000),
            patient=patient,
            row_order=raw.get("row_order") if raw.get("row_order") in atlas.layouts else "d1",
            annotate=bool(raw.get("annotate")),
            vmax_pp=vmax,
            f1=bool(raw.get("f1")),
            f2=bool(raw.get("f2")),
            f2_eps_pp=num("f2_eps_pp", DEFAULT_NEAR_ZERO_PP, 0, 100),
            f2_delta_pp=num("f2_delta_pp", DEFAULT_STRONG_DR_PP, 0, 100),
            f3=bool(raw.get("f3")),
            f3_theta_pp=num("f3_theta_pp", DEFAULT_LEVERAGE_PP, 0, 100),
            f3_source="d1" if raw.get("f3_source") == "d1" or not atlas.has_leverage_csv else "csv",
            selected=selected,
        )


def count_lift(atlas: Atlas, tau_pp: float) -> np.ndarray:
    """Patients (of 7) whose lift is strictly above tau_pp; NaN where the cell is unavailable."""
    with np.errstate(invalid="ignore"):
        passes = (100 * atlas.lift) > (tau_pp + EPS_PP)
    return np.where(atlas.available, passes.sum(axis=0), np.nan)


def count_ratio(atlas: Atlas, rho: float) -> np.ndarray:
    """Patients (of 7) whose AUPRC / prevalence is strictly above rho."""
    with np.errstate(invalid="ignore"):
        passes = (atlas.auprc / atlas.prevalence[:, None, None]) > (rho + EPS)
    return np.where(atlas.available, passes.sum(axis=0), np.nan)


@dataclass
class ColorSpec:
    z: np.ndarray  # (R, C) canonical order; non-finite renders gray
    scale: list
    zmin: float
    zmax: float
    cbar_title: str
    title: str
    primary: Callable[[float], str]  # value -> first hover line
    annotation: Callable[[float], str]  # value -> in-cell text when "annotate" is on
    tickvals: list | None = None
    ticktext: list | None = None


def color_spec(atlas: Atlas, view: View) -> ColorSpec:
    mode = view.mode
    if mode in ("center", "p20"):
        stat = view.center if mode == "center" else "p20"
        base = atlas.center(view.center) if mode == "center" else atlas.p20
        vmax = view.vmax_pp or atlas.d1_vmax_pp
        return ColorSpec(
            100 * base, VLAG_SCALE, -vmax, vmax, f"{stat} lift (AUPRC pp)",
            f"Cohort {stat} held-out lift (AUPRC pp)",
            lambda v, s=stat: f"cohort {s} lift {fmt_pp(v)} pp",
            lambda v: fmt_pp(v, 1, signed=False),
        )
    if mode in ("count_lift", "count_ratio"):
        if mode == "count_lift":
            z, what = count_lift(atlas, view.tau_pp), f"lift > {view.tau_pp:g} pp"
        else:
            z, what = count_ratio(atlas, view.rho), f"AUPRC / prevalence > {view.rho:g}"
        n = len(atlas.patients)
        return ColorSpec(
            z, COUNT_SCALE, -0.5, n - 0.5, f"patients passing (of {n})", f"Supported patients with {what}",
            lambda v, w=what, n=n: f"{int(v)} of {n} patients with {w}",
            lambda v: f"{int(v)}",
            tickvals=list(range(n + 1)), ticktext=[str(i) for i in range(n + 1)],
        )
    if mode == "patient_lift":
        k = atlas.patients.index(view.patient)
        z = 100 * atlas.lift[k]
        finite = np.abs(z[np.isfinite(z)])
        vmax = view.vmax_pp or max(float(np.quantile(finite, 0.98)), 1.0)
        return ColorSpec(
            z, VLAG_SCALE, -vmax, vmax, f"{view.patient} lift (AUPRC pp)",
            f"{view.patient}: held-out lift over own prevalence (AUPRC pp; colour range is patient-specific)",
            lambda v, p=view.patient: f"{p} lift {fmt_pp(v)} pp",
            lambda v: fmt_pp(v, 1, signed=False),
        )
    if mode == "patient_auprc":
        k = atlas.patients.index(view.patient)
        z = atlas.auprc[k]
        vmax = max(float(np.quantile(z[np.isfinite(z)], 0.98)), atlas.prevalence[k] * 1.25, 0.05)
        return ColorSpec(
            z, VIRIDIS_SCALE, 0.0, vmax, f"{view.patient} held-out AUPRC",
            f"{view.patient}: raw held-out AUPRC (random baseline = prevalence {100 * atlas.prevalence[k]:.2f}%)",
            lambda v, p=view.patient: f"{p} AUPRC {100 * v:.1f}%",
            lambda v: f"{v:.2f}",
        )
    # nonzero
    counts = atlas.nonzero_mean
    z = np.log10(1 + counts)
    top = float(np.nanmax(z)) if np.isfinite(z).any() else 1.0
    ticks = [t for t in (0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if np.log10(1 + t) <= top + 0.15]
    return ColorSpec(
        z, BLUES_SCALE, 0.0, top, "nonzero coefficients\n(mean over LOPO folds, log scale)",
        "Stage 2 nonzero coefficients at the selected regularization (mean over LOPO folds)",
        lambda v: f"nonzero coefficients {10 ** v - 1:.1f} (mean over folds)",
        lambda v: f"{10 ** v - 1:.0f}",
        tickvals=[float(np.log10(1 + t)) for t in ticks], ticktext=[str(t) for t in ticks],
    )


def filter_masks(atlas: Atlas, view: View) -> dict[str, np.ndarray]:
    """Boolean (R, C) masks of the enabled filters, in canonical row order."""
    masks: dict[str, np.ndarray] = {}
    center = atlas.center(view.center)
    if view.f1:
        with np.errstate(invalid="ignore"):
            masks["f1"] = atlas.available & (center > EPS) & (atlas.p20 <= EPS)
    if view.f2:
        direct = center[:, 0]
        near_zero = np.isfinite(direct) & (np.abs(direct) <= view.f2_eps_pp / 100 + EPS)
        with np.errstate(invalid="ignore"):
            strong = np.isfinite(center[:, 1:]) & (center[:, 1:] >= view.f2_delta_pp / 100 - EPS)
        qualifying = near_zero & strong.any(axis=1)
        mask = np.zeros_like(atlas.available)
        mask[:, 0] = qualifying
        mask[:, 1:] = strong & qualifying[:, None]
        masks["f2"] = mask & atlas.available
    if view.f3:
        lev = atlas.lev_csv if view.f3_source == "csv" else atlas.lev_d1
        with np.errstate(invalid="ignore"):
            masks["f3"] = np.isfinite(lev) & (100 * lev >= view.f3_theta_pp - EPS_PP)
    return masks


def combined_mask(atlas: Atlas, view: View) -> np.ndarray | None:
    masks = filter_masks(atlas, view)
    if not masks:
        return None
    out = atlas.available.copy()
    for m in masks.values():
        out &= m
    return out


def filter_summary(atlas: Atlas, view: View) -> list[str]:
    """One line per enabled filter plus the combined match count (shown under the heatmap)."""
    masks = filter_masks(atlas, view)
    if not masks:
        return []
    stat = view.center
    lines = []
    if "f1" in masks:
        lines.append(f"{stat} > 0 and p20 <= 0: {plural(int(masks['f1'].sum()), 'cell')}")
    if "f2" in masks:
        rows = int(masks["f2"][:, 0].sum())
        lines.append(
            f"direct gene |{stat}| <= {view.f2_eps_pp:g} pp with a DR cell >= {view.f2_delta_pp:g} pp: "
            f"{plural(rows, 'row')}, {plural(int(masks['f2'].sum()), 'cell')} (direct gene + strong DR cells)"
        )
    if "f3" in masks:
        source = "v2 CSV" if view.f3_source == "csv" else "D1 patient table"
        lev = atlas.lev_csv if view.f3_source == "csv" else atlas.lev_d1
        covered = int(np.isfinite(lev).sum())
        lines.append(
            f"single-patient leverage on the median >= {view.f3_theta_pp:g} pp ({source}): "
            f"{int(masks['f3'].sum())} of {plural(covered, 'covered cell')}"
        )
    combined = combined_mask(atlas, view)
    if len(masks) > 1 and combined is not None:
        lines.append(f"all enabled filters together: {plural(int(combined.sum()), 'cell')}")
    return lines


# ======================================================================================
# 5. Figures
# ======================================================================================


def hover_matrix(atlas: Atlas, view: View, spec: ColorSpec) -> np.ndarray:
    """Hover text for every cell (canonical order)."""
    n_rows, n_cols = atlas.shape
    k = atlas.patients.index(view.patient)
    n_pat = len(atlas.patients)
    c_lift, c_ratio = count_lift(atlas, view.tau_pp), count_ratio(atlas, view.rho)
    out = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        head = f"<b>{esc(atlas.panel_labels[i])}</b>"
        for j, cfg in enumerate(CONFIGS):
            tag = f"{head}<br>{esc(config_label(cfg))}"
            if not atlas.available[i, j]:
                k_value = int(cfg.split("K=")[1]) if "K=" in cfg else None
                why = (
                    f"K={k_value} exceeds the panel's {atlas.panel_genes[i]} covered genes"
                    if k_value is not None and k_value > atlas.panel_genes[i]
                    else "no row for this cell in the atlas table"
                )
                out[i, j] = f"{tag}<br><i>unavailable: {why}</i>"
                continue
            value = spec.z[i, j]
            first = spec.primary(value) if np.isfinite(value) else "no value in this colour mode"
            lines = [
                tag,
                f"<b>{first}</b>",
                f"median {fmt_pp(100 * atlas.median[i, j])} | mean {fmt_pp(100 * atlas.mean[i, j])} | "
                f"p20 {fmt_pp(100 * atlas.p20[i, j])} pp",
                f"lift > {view.tau_pp:g} pp: {int(c_lift[i, j])}/{n_pat} | AUPRC/prev > {view.rho:g}: {int(c_ratio[i, j])}/{n_pat}",
                f"{view.patient}: lift {fmt_pp(100 * atlas.lift[k, i, j])} pp, AUPRC {100 * atlas.auprc[k, i, j]:.1f}%",
            ]
            if np.isfinite(atlas.nonzero_mean[i, j]):
                lines.append(
                    f"nonzero coefs {atlas.nonzero_mean[i, j]:.1f} mean / {atlas.nonzero_max[i, j]:.0f} max of "
                    f"{atlas.n_features[i, j]:.0f} | {esc(str(atlas.reg_label[i, j]))}"
                )
            lev = f"leverage D1 {100 * atlas.lev_d1[i, j]:.1f} pp"
            if atlas.lev_d1[i, j] > EPS:
                lev += f" (drop {drivers_text(atlas.lev_d1_driver[i, j])})"
            if np.isfinite(atlas.lev_csv[i, j]):
                lev += f" | v2 CSV {100 * atlas.lev_csv[i, j]:.1f} pp (drop {drivers_text(atlas.lev_csv_driver[i, j])})"
            lines.append(lev)
            out[i, j] = "<br>".join(lines)
    return out


def _outline_xy(cells: list[tuple[int, int]]) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    for i, j in cells:
        xs += [j - 0.5, j + 0.5, j + 0.5, j - 0.5, j - 0.5, None]
        ys += [i - 0.5, i - 0.5, i + 0.5, i + 0.5, i - 0.5, None]
    return xs, ys


def build_heatmap(atlas: Atlas, view: View) -> go.Figure:
    layout = atlas.layouts[view.row_order]
    order = layout.order
    n_rows, n_cols = atlas.shape
    spec = color_spec(atlas, view)
    z = spec.z[order]
    finite = np.isfinite(z)
    hover = hover_matrix(atlas, view, spec)[order]
    mask = combined_mask(atlas, view)
    xs, ys = list(range(n_cols)), list(range(n_rows))

    annotation_text = np.where(finite, np.vectorize(lambda v: spec.annotation(v) if np.isfinite(v) else "")(z), "")
    common = dict(
        x=xs, y=ys, xgap=1, ygap=1, hoverongaps=False, hoverinfo="text", hovertext=hover,
        zmin=spec.zmin, zmax=spec.zmax,
    )
    text_kwargs = dict(text=annotation_text, texttemplate="%{text}", textfont=dict(size=9)) if (
        view.annotate or view.mode in ("count_lift", "count_ratio")
    ) else {}

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=np.where(finite, np.nan, 1.0), colorscale=[[0, GRAY], [1, GRAY]], showscale=False,
            x=xs, y=ys, xgap=1, ygap=1, hoverongaps=False, hoverinfo="text", hovertext=hover, zmin=0, zmax=1,
        )
    )
    if mask is None:
        main_z, dim_z = z, None
    else:
        shown = mask[order]
        main_z, dim_z = np.where(finite & shown, z, np.nan), np.where(finite & ~shown, z, np.nan)
    colorbar = dict(
        title=dict(text=spec.cbar_title.replace("\n", "<br>"), side="right", font=dict(size=11)),
        thickness=14, len=0.45, y=0.8, yanchor="middle", tickfont=dict(size=10),
    )
    if spec.tickvals is not None:
        colorbar.update(tickmode="array", tickvals=spec.tickvals, ticktext=spec.ticktext)
    fig.add_trace(go.Heatmap(z=main_z, colorscale=spec.scale, colorbar=colorbar, **common, **text_kwargs))
    if dim_z is not None:
        fig.add_trace(go.Heatmap(z=dim_z, colorscale=_muted(spec.scale), showscale=False, **common, **text_kwargs))
        cells = [(int(i), int(j)) for i, j in zip(*np.nonzero(finite & mask[order]))]
        if cells:
            ox, oy = _outline_xy(cells)
            fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="#111111", width=1.6), hoverinfo="skip", showlegend=False))
    if view.selected is not None:
        canonical_row, col = view.selected
        shown_row = int(np.nonzero(order == canonical_row)[0][0])
        ox, oy = _outline_xy([(shown_row, col)])
        fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="#00bcd4", width=3.5), hoverinfo="skip", showlegend=False))

    shapes = [
        dict(type="line", x0=-0.5, x1=n_cols - 0.5, y0=b, y1=b, line=dict(color="black", width=2.0)) for b in layout.boundaries
    ] + [dict(type="line", x0=b, x1=b, y0=-0.5, y1=n_rows - 0.5, line=dict(color="black", width=1.4)) for b in COL_BOUNDARIES]
    centers = [0.0] + [1 + 4 * b + 1.5 for b in range(4)]
    headers = [METHOD_LABEL["direct_gene"]] + [METHOD_LABEL[m] for m in METHODS]
    annotations = [
        dict(x=cx, y=1.0, xref="x", yref="paper", text=f"<b>{h}</b>", showarrow=False, yanchor="bottom", font=dict(size=11))
        for cx, h in zip(centers, headers)
    ]
    fig.update_layout(
        template="plotly_white",
        height=ROW_PX * n_rows + HEATMAP_MARGIN["t"] + HEATMAP_MARGIN["b"],
        margin=HEATMAP_MARGIN,
        shapes=shapes,
        annotations=annotations,
        hovermode="closest",
        hoverlabel=dict(align="left", font=dict(size=12)),
        title=dict(text=f"<b>{esc(spec.title)}</b>", x=0.0, xanchor="left", y=0.985, yanchor="top", font=dict(size=14)),
        xaxis=dict(
            range=[-0.5, n_cols - 0.5], tickmode="array", tickvals=xs, ticktext=[config_label(c) for c in CONFIGS],
            tickangle=-55, showgrid=False, zeroline=False, fixedrange=True, title=dict(text="Stage 1 configuration (gray = unavailable)", font=dict(size=11)),
        ),
        yaxis=dict(
            range=[n_rows - 0.5, -0.5], tickmode="array", tickvals=ys, ticktext=layout.labels,
            showgrid=False, zeroline=False, fixedrange=True, tickfont=dict(size=10),
        ),
        uirevision="atlas",
    )
    return fig


def _sorted_unique_range(values: list[float], pad: float = 0.08, floor: float = 1.0) -> list[float]:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    lo, hi = min(finite + [0.0]), max(finite + [0.0])
    span = max(hi - lo, floor)
    return [lo - pad * span if lo < 0 else 0.0, hi + pad * span]


def patient_row_figure(atlas: Atlas, r: int, c: int, highlight: str | None = None) -> go.Figure:
    """Seven small panels, one per patient, with ONE shared y-axis (AUPRC pp) so they compare honestly."""
    n_pat = len(atlas.patients)
    prev = 100 * atlas.prevalence
    held = 100 * atlas.auprc[:, r, c]
    lifts = held - prev
    titles = []
    for k, p in enumerate(atlas.patients):
        colour = "#2e7d32" if lifts[k] > EPS_PP else "#c62828" if lifts[k] < -EPS_PP else "#666666"
        titles.append(
            f"<b>{p}</b><br><span style='color:{colour}'>lift {fmt_pp(lifts[k], 1)} pp</span><br>"
            f"<span style='font-size:9px;color:#666'>{atlas.n_malignant[k]:,} mal | {atlas.n_normal[k]:,} non</span>"
        )
    fig = make_subplots(rows=1, cols=n_pat, shared_yaxes=True, horizontal_spacing=0.012, subplot_titles=titles)
    for k, p in enumerate(atlas.patients):
        fig.add_trace(
            go.Bar(
                x=["baseline", "held-out"], y=[prev[k], held[k]], marker_color=[BASELINE_GRAY, HELDOUT_BLUE],
                text=[f"{prev[k]:.1f}", f"{held[k]:.1f}"], textposition="outside", cliponaxis=False, showlegend=False,
                hovertemplate=(
                    f"<b>{p}</b><br>%{{x}}: %{{y:.2f}} pp<br>lift {fmt_pp(lifts[k])} pp<br>"
                    f"{atlas.n_malignant[k]:,} malignant / {atlas.n_normal[k]:,} non-malignant cells<extra></extra>"
                ),
            ),
            row=1, col=k + 1,
        )
    for name, colour in (("prevalence (random AUPRC)", BASELINE_GRAY), ("held-out AUPRC", HELDOUT_BLUE)):
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, marker_color=colour, showlegend=True, hoverinfo="skip"), row=1, col=1)
    y_top = max(1.2 * float(max(prev.max(), held.max())), 5.0)
    fig.update_yaxes(range=[0, y_top], matches="y", showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(title_text="AUPRC (percentage points)", row=1, col=1)
    fig.update_xaxes(showticklabels=False, fixedrange=True)  # colours + legend replace per-panel tick labels
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11), yanchor="bottom")
    if highlight in atlas.patients:
        k = atlas.patients.index(highlight) + 1
        fig.add_shape(
            type="rect", xref="x domain" if k == 1 else f"x{k} domain", yref="paper",
            x0=0, x1=1, y0=0, y1=1, line=dict(color="#00bcd4", width=3),
        )
    fig.update_layout(
        template="plotly_white", height=330, margin=dict(l=60, r=10, t=96, b=44), barmode="group",
        title=dict(
            text="<b>Per patient: prevalence (random AUPRC) vs held-out AUPRC</b> - same y-axis in every panel",
            x=0, xanchor="left", font=dict(size=12),
        ),
        legend=dict(orientation="h", x=0.0, xanchor="left", y=-0.02, yanchor="top", font=dict(size=10)),
    )
    return fig


def lift_rank_figure(atlas: Atlas, view: View, r: int, c: int) -> go.Figure:
    """D1d-b: the seven lifts ordered, with 0 / median / mean / p20 references."""
    lifts = 100 * atlas.lift[:, r, c]
    order = np.argsort(lifts, kind="stable")
    ranked = lifts[order]
    ranks = list(range(1, len(ranked) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranks, y=ranked, mode="lines+markers+text", text=[atlas.patients[k] for k in order],
            textposition="top center", marker=dict(size=9, color=HELDOUT_BLUE), line=dict(color=HELDOUT_BLUE),
            hovertemplate="%{text}: %{y:.2f} pp<extra></extra>", showlegend=False,
        )
    )
    median, mean, p20 = 100 * atlas.median[r, c], 100 * atlas.mean[r, c], 100 * atlas.p20[r, c]
    references = [
        (0.0, "#666666", "prevalence baseline (lift 0)", "dash", 1),
        (median, MEDIAN_ORANGE, f"median = {fmt_pp(median)} pp", "solid", 2.4 if view.center == "median" else 1.2),
        (mean, MEAN_PURPLE, f"mean = {fmt_pp(mean)} pp", "dot", 2.4 if view.center == "mean" else 1.2),
        (p20, P20_GREEN, f"p20 = {fmt_pp(p20)} pp", "solid", 1.6),
    ]
    if view.mode == "count_lift":  # tie the cell view to the map being explored: which patients clear tau
        references.append((view.tau_pp, "#c62828", f"tau = {view.tau_pp:g} pp", "dashdot", 1.4))
    for value, colour, name, dash, width in references:
        fig.add_trace(go.Scatter(x=[0.6, len(ranks) + 0.4], y=[value, value], mode="lines", name=name, line=dict(color=colour, dash=dash, width=width)))
    plotted = list(lifts) + [median, mean, p20] + ([view.tau_pp] if view.mode == "count_lift" else [])
    fig.update_layout(
        template="plotly_white", height=330, margin=dict(l=60, r=10, t=44, b=48),
        title=dict(text="<b>Lift ranked across the seven patients</b> (median = typical, p20 = lower-tail floor)", x=0, xanchor="left", font=dict(size=12)),
        xaxis=dict(title="supported patients ordered by held-out lift", tickmode="array", tickvals=ranks, range=[0.4, len(ranks) + 0.6]),
        yaxis=dict(title="held-out lift (AUPRC pp)", range=_sorted_unique_range(plotted, pad=0.18)),
        legend=dict(orientation="h", y=-0.28, font=dict(size=10)),
    )
    return fig


def profile_row_figure(atlas: Atlas, view: View, r: int, c: int) -> go.Figure:
    """Same panel across Stage 1: one small panel per method, lift vs K, with ONE shared y-axis."""
    center = 100 * atlas.center(view.center)
    k_labels = [str(k) for k in KS]
    fig = make_subplots(rows=1, cols=len(METHODS), shared_yaxes=True, horizontal_spacing=0.015, subplot_titles=[METHOD_LABEL[m] for m in METHODS])
    plotted: list[float] = [0.0, center[r, 0]]
    direct = center[r, 0]
    for mi, method in enumerate(METHODS):
        cols = [CONFIGS.index(f"{method}|K={k}") for k in KS]
        for k_idx, patient in enumerate(atlas.patients):
            y = 100 * atlas.lift[k_idx, r, cols]
            plotted += list(y)
            fig.add_trace(
                go.Scatter(
                    x=k_labels, y=y, mode="lines", line=dict(color=PATIENT_COLORS[k_idx % len(PATIENT_COLORS)], width=1),
                    opacity=0.6, name=patient, legendgroup=patient, showlegend=(mi == 0),
                    hovertemplate=f"{patient}: %{{y:.2f}} pp at K=%{{x}}<extra></extra>",
                ),
                row=1, col=mi + 1,
            )
        y_center = center[r, cols]
        plotted += list(y_center)
        fig.add_trace(
            go.Scatter(
                x=k_labels, y=y_center, mode="lines+markers", line=dict(color="#111111", width=2.6), marker=dict(size=7),
                name=f"cohort {view.center}", legendgroup="center", showlegend=(mi == 0),
                hovertemplate=f"cohort {view.center}: %{{y:.2f}} pp at K=%{{x}}<extra></extra>",
            ),
            row=1, col=mi + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=k_labels, y=[direct] * len(KS), mode="lines", line=dict(color="#666666", dash="dash", width=1.4),
                name=f"direct gene ({view.center})", legendgroup="direct", showlegend=(mi == 0),
                hovertemplate=f"direct gene {view.center}: {fmt_pp(direct)} pp<extra></extra>",
            ),
            row=1, col=mi + 1,
        )
        if c in cols:
            fig.add_trace(
                go.Scatter(
                    x=[str(KS[cols.index(c)])], y=[center[r, c]], mode="markers",
                    marker=dict(size=15, symbol="circle-open", color="#00bcd4", line=dict(width=3)),
                    hoverinfo="skip", showlegend=False,
                ),
                row=1, col=mi + 1,
            )
    fig.update_yaxes(range=_sorted_unique_range(plotted, pad=0.1, floor=2.0), matches="y", showgrid=True, gridcolor="#eeeeee", zeroline=True, zerolinecolor="#999999")
    fig.update_yaxes(title_text="held-out lift (AUPRC pp)", row=1, col=1)
    fig.update_xaxes(title_text="K", fixedrange=True)
    fig.update_layout(
        template="plotly_white", height=360, margin=dict(l=60, r=10, t=70, b=90),
        title=dict(text=f"<b>{esc(atlas.panel_labels[r])} across Stage 1</b> - same y-axis in every panel", x=0, xanchor="left", font=dict(size=12)),
        legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
    )
    return fig


def patient_table(atlas: Atlas, view: View, r: int, c: int) -> pd.DataFrame:
    """The seven patients with prevalence, cell counts, AUPRC, lift and drop-one medians."""
    lifts = 100 * atlas.lift[:, r, c]
    held = 100 * atlas.auprc[:, r, c]
    prev = 100 * atlas.prevalence
    drop_one = [float(np.median(np.delete(lifts, k))) for k in range(len(lifts))]
    return pd.DataFrame(
        {
            "patient": atlas.patients,
            "malignant cells": atlas.n_malignant,
            "non-malignant cells": atlas.n_normal,
            "prevalence %": prev,
            "held-out AUPRC %": held,
            "lift pp": lifts,
            "AUPRC / prevalence": atlas.auprc[:, r, c] / atlas.prevalence,
            "median without this patient (pp)": drop_one,
            f"lift > {view.tau_pp:g} pp": lifts > view.tau_pp + EPS_PP,
            f"ratio > {view.rho:g}": atlas.auprc[:, r, c] / atlas.prevalence > view.rho + EPS,
        }
    )


def cell_header(atlas: Atlas, view: View, r: int, c: int) -> tuple[str, str, list[tuple[str, str]]]:
    """Title, subtitle and (label, value) chips for the clicked cell."""
    title = f"[{atlas.panel_classes[r]}] {atlas.panel_labels[r]} | {config_label(CONFIGS[c])}"
    subtitle = f"theme: {atlas.panel_themes[r]} | {atlas.panel_genes[r]} covered genes"
    chips = [
        ("median", f"{fmt_pp(100 * atlas.median[r, c])} pp"),
        ("mean", f"{fmt_pp(100 * atlas.mean[r, c])} pp"),
        ("p20", f"{fmt_pp(100 * atlas.p20[r, c])} pp"),
        ("patients lift > 0", f"{int(count_lift(atlas, 0.0)[r, c])}/{len(atlas.patients)}"),
        (f"lift > {view.tau_pp:g} pp", f"{int(count_lift(atlas, view.tau_pp)[r, c])}/{len(atlas.patients)}"),
        (f"AUPRC/prev > {view.rho:g}", f"{int(count_ratio(atlas, view.rho)[r, c])}/{len(atlas.patients)}"),
    ]
    if np.isfinite(atlas.nonzero_mean[r, c]):
        chips.append(("nonzero coefs", f"{atlas.nonzero_mean[r, c]:.1f} mean / {atlas.nonzero_max[r, c]:.0f} max of {atlas.n_features[r, c]:.0f}"))
        chips.append(("selected regularization", str(atlas.reg_label[r, c])))
        if atlas.nonzero_max[r, c] == 0:
            chips.append(("note", "all-zero model: lift is 0 by construction"))
    lev = f"{100 * atlas.lev_d1[r, c]:.1f} pp"
    if atlas.lev_d1[r, c] > EPS:
        before = 100 * atlas.median[r, c]
        after = before - 100 * atlas.lev_d1_delta[r, c]
        lev += f" (dropping {drivers_text(atlas.lev_d1_driver[r, c])} moves the median {fmt_pp(before)} -> {fmt_pp(after)} pp)"
    chips.append(("leverage, D1 basis", lev))
    if np.isfinite(atlas.lev_csv[r, c]):
        before = 100 * atlas.lev_csv_full_median[r, c]
        after = before - 100 * atlas.lev_csv_delta[r, c]
        chips.append(
            (
                "leverage, v2 CSV",
                f"{100 * atlas.lev_csv[r, c]:.1f} pp (dropping {drivers_text(atlas.lev_csv_driver[r, c])} moves the CSV median "
                f"{fmt_pp(before)} -> {fmt_pp(after)} pp; the heatmap median is {fmt_pp(100 * atlas.median[r, c])} pp)",
            )
        )
    return title, subtitle, chips


# ======================================================================================
# 6. Bundle export (for sharing; never writes inside the experiment)
# ======================================================================================


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(data_dir: Path, out_dir: Path) -> list[Path]:
    """Copy the small inputs and write the joined nonzero table to a NEW directory."""
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    if out_dir == data_dir or data_dir in out_dir.parents:
        raise DataError(f"refusing to write inside the experiment directory: {out_dir}")
    files = find_inputs(data_dir)
    table = build_nonzero_table(data_dir) if files["nonzero_table"] is None else pd.read_csv(files["nonzero_table"])
    if table is None:
        raise DataError("cannot export: the four stage2_sharedness_lopo_scorecard.csv files were not found")
    targets = {key: out_dir / files[key].name for key in ("atlas", "patients", "leverage") if files[key] is not None}
    targets["nonzero_table"] = out_dir / NONZERO_TABLE_NAME
    clash = [str(p) for p in [*targets.values(), out_dir / "bundle_manifest.json"] if p.exists()]
    if clash:
        raise DataError("refusing to overwrite existing files:\n  " + "\n  ".join(clash))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": "python old34_d1_atlas_explorer.py --data-dir <this directory>",
        "files": {},
    }
    for key, target in targets.items():
        if key == "nonzero_table":
            table.to_csv(target, index=False)
            manifest["files"][target.name] = {"derived": "selected-regularization nonzero counts joined from 4 LOPO scorecards"}
        else:
            shutil.copyfile(files[key], target)
            if _sha256(files[key]) != _sha256(target):
                raise DataError(f"copy of {files[key].name} does not match the source")
            manifest["files"][target.name] = {"source": str(files[key]), "sha256": _sha256(target)}
        manifest["files"][target.name]["bytes"] = target.stat().st_size
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [*targets.values(), out_dir / "bundle_manifest.json"]


# ======================================================================================
# 7. --check (validation against your saved diagnostics)
# ======================================================================================


def _saved_matrix(data_dir: Path, key: str) -> pd.DataFrame | None:
    path = data_dir / "analysis" / "scorecards" / "stage2_diagnostic" / CHECK_FILES[key]
    return pd.read_csv(path, index_col=0) if path.is_file() else None


def run_checks(atlas: Atlas, fingerprint_before: dict, fingerprint_after: dict) -> list[tuple[str, str, str]]:
    """(name, PASS|FAIL|SKIP|INFO, detail) rows."""
    rows: list[tuple[str, str, str]] = []
    n_rows, n_cols = atlas.shape
    expected = EXPECTED
    got = dict(panels=n_rows, configs=n_cols, cells=atlas.audit["cells"], unavailable=atlas.audit["unavailable_cells"], patients=len(atlas.patients))
    bad = {k: (got[k], expected[k]) for k in got if got[k] != expected[k]}
    rows.append(("atlas accounting", "FAIL" if bad else "PASS", f"{got}" + (f" expected differs: {bad}" if bad else "")))
    tol = 1e-9
    ok = atlas.audit["median_vs_table_max_abs_diff"] < tol and atlas.audit["p20_vs_table_max_abs_diff"] < tol
    rows.append(("median/p20 recomputed from patient table == representation table", "PASS" if ok else "FAIL",
                 f"max |diff| {atlas.audit['median_vs_table_max_abs_diff']:.1e} / {atlas.audit['p20_vs_table_max_abs_diff']:.1e}"))

    saved = _saved_matrix(atlas.data_dir, "median_matrix")
    if saved is None:
        rows.append(("frozen D1 sort == saved median-matrix order", "SKIP", "old34_all_representation_patient_median_lift_matrix.csv not present"))
    else:
        same_rows = saved.index.tolist() == atlas.panel_labels
        same_cols = saved.columns.tolist() == list(CONFIGS)
        same_values = bool(np.allclose(saved.to_numpy(float), atlas.median, equal_nan=True, atol=1e-12))
        rows.append(("frozen D1 sort == saved median-matrix order", "PASS" if (same_rows and same_cols and same_values) else "FAIL",
                     f"rows {same_rows}, columns {same_cols}, values {same_values}"))

    for key, kind, threshold in [("d1f_lift_0", "lift", 0.0), ("d1f_lift_5", "lift", 5.0), ("d1f_ratio_2", "ratio", 2.0), ("d1f_ratio_3", "ratio", 3.0)]:
        saved = _saved_matrix(atlas.data_dir, key)
        label = f"patient counts reproduce saved D1f {kind} > {threshold:g}{' pp' if kind == 'lift' else ''}"
        if saved is None:
            rows.append((label, "SKIP", f"{CHECK_FILES[key]} not present"))
            continue
        mine = count_lift(atlas, threshold) if kind == "lift" else count_ratio(atlas, threshold)
        theirs = saved.reindex(index=atlas.panel_labels, columns=list(CONFIGS)).to_numpy(float)
        equal = bool(np.array_equal(np.nan_to_num(mine, nan=-1), np.nan_to_num(theirs, nan=-1)))
        rows.append((label, "PASS" if equal else "FAIL", f"{int((mine != theirs).sum() if not equal else 0)} differing cells"))

    cells = atlas.audit["cells"]
    nz = atlas.audit.get("nonzero_cells", 0)
    rows.append(("Stage 2 nonzero count joined at selected regularization", "PASS" if nz == cells else "FAIL", f"{nz}/{cells} cells ({atlas.audit.get('nonzero_source', 'n/a')})"))
    n_zero = int((atlas.nonzero_max == 0).sum())
    zero_lift = bool(np.all(np.abs(atlas.lift[:, atlas.nonzero_max == 0]) < EPS)) if n_zero else True
    rows.append(("all-zero models have exactly zero lift", "PASS" if zero_lift else "FAIL", f"{n_zero} all-zero cells"))
    lv = atlas.audit.get("leverage_csv_cells", 0)
    rows.append(("v2 leverage CSV coverage", "PASS" if lv == expected["leverage_cells"] and atlas.audit.get("leverage_csv_unmatched_cells", 0) == 0 else "INFO",
                 f"{lv}/{cells} cells, unmatched {atlas.audit.get('leverage_csv_unmatched_cells', 0)}"))
    if lv:
        rows.append(("v2 leverage CSV is on the same basis as the heatmap", "INFO",
                     f"{atlas.audit['leverage_csv_basis_mismatch_cells']}/{lv} cells differ from the D1 median "
                     f"(max {atlas.audit['leverage_csv_basis_max_abs_diff_pp']:.1f} pp): the CSV uses per-patient best regularization"))
    rows.append(("inputs unmodified by loading", "PASS" if fingerprint_before == fingerprint_after else "FAIL", f"{len(fingerprint_before)} files fingerprinted (size + mtime)"))
    return rows


# ======================================================================================
# 8. Dash app
# ======================================================================================

CSS = """
body { margin: 0; font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color: #222; }
.page { display: flex; align-items: flex-start; }
.sidebar { width: 296px; flex: none; position: sticky; top: 0; height: 100vh; overflow-y: auto; box-sizing: border-box;
           padding: 14px 16px 24px; background: #f7f8fa; border-right: 1px solid #dde1e6; font-size: 13px; }
.sidebar h1 { font-size: 16px; margin: 0 0 2px; }
.sidebar h2 { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: #667; margin: 18px 0 6px; }
.sub { color: #667; font-size: 11.5px; line-height: 1.35; }
.lbl { display: block; font-weight: 600; margin: 8px 0 3px; }
.help { color: #667; font-size: 11.5px; margin-top: 3px; line-height: 1.35; }
.caveat { color: #8a4b00; background: #fff4e0; border: 1px solid #f0d9b0; border-radius: 4px; padding: 6px 8px; font-size: 11.5px; margin-top: 6px; line-height: 1.35; }
.row2 { display: flex; gap: 8px; } .row2 > div { flex: 1; }
.num { width: 100%; box-sizing: border-box; padding: 4px 6px; }
.main { flex: 1; min-width: 0; padding: 10px 16px 30px; }
.grid { display: grid; grid-template-columns: minmax(620px, 780px) minmax(480px, 1fr); gap: 14px; align-items: start; }
.cellcol { position: sticky; top: 8px; max-height: calc(100vh - 46px); overflow-y: auto; }
@media (max-width: 1450px) { .grid { grid-template-columns: 1fr; } .cellcol { position: static; max-height: none; } }
.card { border: 1px solid #e1e4e8; border-radius: 6px; padding: 8px 12px; background: #fff; }
.status { font-size: 12.5px; color: #333; margin: 4px 2px 6px; line-height: 1.45; }
.chip { display: inline-block; background: #eef2f7; border-radius: 10px; padding: 2px 9px; margin: 2px 4px 2px 0; font-size: 12px; }
.chip b { color: #345; }
button.back { background: #00838f; color: #fff; border: 0; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 13px; }
button.back:disabled { background: #b8c4c8; cursor: default; }
button.ghost { background: #fff; border: 1px solid #c3c9d0; border-radius: 4px; padding: 4px 10px; cursor: pointer; font-size: 12px; margin-top: 8px; }
table.pt { border-collapse: collapse; font-size: 12px; width: 100%; margin: 6px 0; }
table.pt th { text-align: right; font-weight: 600; border-bottom: 2px solid #ccd; padding: 3px 6px; background: #f3f5f8; }
table.pt td { text-align: right; padding: 3px 6px; border-bottom: 1px solid #eee; }
table.pt th:first-child, table.pt td:first-child { text-align: left; }
tr.sel td { background: #e0f7fa; }
details pre { white-space: pre-wrap; font-size: 11px; color: #445; }
.placeholder { color: #667; padding: 40px 18px; text-align: center; line-height: 1.6; }
"""


def create_app(atlas: Atlas):
    """Build the Dash app.  Dash is imported here so --check and the tests do not need it."""
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

    graph_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": "old34_d1_atlas", "scale": 2},
    }
    app = Dash(__name__, title="Old34 D1 atlas explorer")
    app.index_string = (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
        f"<style>{CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"
    )

    default_patient = DEFAULT_PATIENT if DEFAULT_PATIENT in atlas.patients else atlas.patients[0]
    mode_options = [
        {"label": MODE_LABELS[m], "value": m, "disabled": (m == "nonzero" and not atlas.has_nonzero)} for m in MODE_LABELS
    ]

    def number(id_, value, step, lo=None, hi=None):
        return dcc.Input(id=id_, type="number", value=value, step=step, min=lo, max=hi, className="num", debounce=False)

    def control(label, component, help_=None):
        return html.Div([html.Label(label, className="lbl"), component] + ([html.Div(help_, className="help")] if help_ else []))

    radio_style = {"display": "inline-block", "marginRight": "14px"}
    sidebar = html.Div(
        className="sidebar",
        children=[
            html.H1("Old34 D1 atlas explorer"),
            html.Div(
                f"{atlas.shape[0]} Stage 0 panels x {atlas.shape[1]} Stage 1 configurations. Colour = Stage 2 LOPO result "
                f"over {len(atlas.patients)} supported patients. Read-only; nothing is refit.",
                className="sub",
            ),
            html.H2("Colour by"),
            dcc.Dropdown(id="color-mode", options=mode_options, value="center", clearable=False),
            control(
                "Cohort statistic",
                dcc.RadioItems(id="center-stat", options=[{"label": "median", "value": "median"}, {"label": "mean", "value": "mean"}],
                               value="median", labelStyle=radio_style),
                "Applies to the cohort-lift colour mode and to filters 1-2.",
            ),
            html.Div(className="row2", children=[
                html.Div(control("Lift cutoff tau (pp)", number("tau", DEFAULT_TAU_PP, 0.5), "5 pp = 0.05 lift")),
                html.Div(control("AUPRC/prev cutoff rho", number("rho", DEFAULT_RHO, 0.5, 0))),
            ]),
            control("Patient (patient colour modes)", dcc.Dropdown(id="patient", options=atlas.patients, value=default_patient, clearable=False)),
            control("Colour range +/- (pp), blank = D1 auto", number("vmax", None, 1, 0.5), f"D1 auto = {atlas.d1_vmax_pp:.1f} pp for cohort modes."),
            dcc.Checklist(id="annotate", options=[{"label": " show values in cells", "value": "on"}], value=[], style={"marginTop": "8px"}),
            html.H2("Rows"),
            dcc.RadioItems(
                id="row-order",
                options=[
                    {"label": " Frozen D1 sort (controls first, then best p20)", "value": "d1"},
                    {"label": " Grouped by panel construction (D1DD1)", "value": "grouped"},
                ],
                value="d1", labelStyle={"display": "block", "margin": "2px 0"},
            ),
            html.Div("Row order never changes with colour mode, statistic, or filters.", className="help"),
            html.H2("Filters (all enabled filters must match)"),
            dcc.Checklist(id="f1", options=[{"label": " median > 0 with p20 <= 0", "value": "on"}], value=[]),
            html.Div("Typical patient beats own prevalence, but the lower tail does not.", className="help"),
            dcc.Checklist(id="f2", options=[{"label": " direct gene near zero, strong DR cell on the row", "value": "on"}], value=[], style={"marginTop": "10px"}),
            html.Div(className="row2", children=[
                html.Div(control("near zero: |lift| <= (pp)", number("f2-eps", DEFAULT_NEAR_ZERO_PP, 0.5, 0))),
                html.Div(control("strong DR: lift >= (pp)", number("f2-delta", DEFAULT_STRONG_DR_PP, 0.5, 0))),
            ]),
            dcc.Checklist(id="f3", options=[{"label": " high single-patient leverage on the median", "value": "on"}], value=[], style={"marginTop": "10px"}),
            html.Div(className="row2", children=[
                html.Div(control("drop-one change >= (pp)", number("f3-theta", DEFAULT_LEVERAGE_PP, 0.5, 0))),
            ]),
            dcc.RadioItems(
                id="f3-source",
                options=[
                    {"label": " v2 leverage CSV", "value": "csv", "disabled": not atlas.has_leverage_csv},
                    {"label": " D1 patient table (all cells)", "value": "d1"},
                ],
                value="csv" if atlas.has_leverage_csv else "d1", labelStyle={"display": "block", "margin": "2px 0"},
            ),
            html.Div(id="f3-caveat", className="caveat"),
            html.Button("Clear filters", id="btn-clear", className="ghost"),
            html.H2("Data"),
            html.Details([html.Summary("Data audit"), html.Pre("\n".join(audit_lines(atlas)))]),
            html.Div(
                "Lift = held-out AUPRC - the patient's own malignant prevalence. Descriptive, not a p-value; Stage 1 is "
                "fit on all eligible cells (transductive).",
                className="help", style={"marginTop": "12px"},
            ),
        ],
    )

    main = html.Div(
        className="main",
        children=[
            html.Div(id="status", className="status"),
            html.Div(
                className="grid",
                children=[
                    html.Div(className="card", children=[
                        dcc.Graph(id="heatmap", config=graph_config, style={"height": f"{ROW_PX * atlas.shape[0] + HEATMAP_MARGIN['t'] + HEATMAP_MARGIN['b']}px"}),
                    ]),
                    html.Div(className="cellcol", children=[
                        html.Div(className="card", children=[
                            html.Div(style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"}, children=[
                                html.H3("Cell view", style={"margin": "4px 0"}),
                                html.Button("< Back to cohort map", id="btn-back", className="back", disabled=True),
                            ]),
                            html.Div(id="cell-body"),
                        ]),
                    ]),
                ],
            ),
            dcc.Store(id="sel", data=None),
        ],
    )
    app.layout = html.Div(className="page", children=[sidebar, main])

    def raw_view(mode, center, tau, rho, patient, row_order, annotate, vmax, f1, f2, f2_eps, f2_delta, f3, f3_theta, f3_source, sel) -> View:
        return View.parse(
            dict(mode=mode, center=center, tau_pp=tau, rho=rho, patient=patient, row_order=row_order,
                 annotate=bool(annotate), vmax_pp=vmax, f1=bool(f1), f2=bool(f2), f2_eps_pp=f2_eps,
                 f2_delta_pp=f2_delta, f3=bool(f3), f3_theta_pp=f3_theta, f3_source=f3_source, selected=sel),
            atlas,
        )

    view_inputs = [
        Input("color-mode", "value"), Input("center-stat", "value"), Input("tau", "value"), Input("rho", "value"),
        Input("patient", "value"), Input("row-order", "value"), Input("annotate", "value"), Input("vmax", "value"),
        Input("f1", "value"), Input("f2", "value"), Input("f2-eps", "value"), Input("f2-delta", "value"),
        Input("f3", "value"), Input("f3-theta", "value"), Input("f3-source", "value"), Input("sel", "data"),
    ]

    @app.callback(
        Output("heatmap", "figure"), Output("status", "children"), Output("btn-back", "disabled"), Output("f3-caveat", "children"),
        *view_inputs,
    )
    def render_heatmap(*args):
        view = raw_view(*args)
        spec = color_spec(atlas, view)
        figure = build_heatmap(atlas, view)
        finite = int(np.isfinite(spec.z[atlas.available]).sum())
        lines = [html.Div([html.B(spec.title), f"  |  {finite} cells colored, {int((~atlas.available).sum())} unavailable (gray)  |  rows: {atlas.layouts[view.row_order].title}"])]
        for text in filter_summary(atlas, view):
            lines.append(html.Div("filter: " + text))
        if view.mode == "nonzero" and not atlas.has_nonzero:
            lines.append(html.Div("Stage 2 nonzero counts are not available in this data directory."))
        a = atlas.audit
        if view.f3 and view.f3_source == "csv":
            caveat = (
                f"v2 CSV covers {a.get('leverage_csv_cells', 0)} of {a['cells']} cells (best-LOPO K per panel x method; no controls, "
                f"no direct gene). Its medians use each patient's own best regularization and differ from the heatmap's "
                f"median at {a.get('leverage_csv_basis_mismatch_cells', 0)} of those cells. Use 'D1 patient table' for a "
                f"leverage consistent with the colours."
            )
        elif view.f3:
            caveat = "Computed from this atlas's seven lifts per cell: drop one patient, recompute the median. Consistent with the heatmap; covers every cell."
        else:
            caveat = f"v2 CSV: {a.get('leverage_csv_cells', 0)}/{a['cells']} cells; built on a different regularization basis than the heatmap (see Data audit)."
        selected_or_patient_mode = view.selected is not None or view.mode in PATIENT_MODES
        return figure, lines, not selected_or_patient_mode, caveat

    @app.callback(Output("heatmap", "clickData"), Input("btn-back", "n_clicks"), prevent_initial_call=True)
    def clear_click(_):
        return None

    @app.callback(
        Output("sel", "data"), Output("color-mode", "value"),
        Input("heatmap", "clickData"), Input("btn-back", "n_clicks"),
        State("row-order", "value"), State("color-mode", "value"),
        prevent_initial_call=True,
    )
    def on_click_or_back(click, _back, row_order, mode):
        if ctx.triggered_id == "btn-back":
            return None, ("center" if mode in PATIENT_MODES else no_update)
        if not click or not click.get("points"):
            return no_update, no_update
        point = click["points"][0]
        try:
            shown_row, col = int(round(point["y"])), int(round(point["x"]))
            row = int(atlas.layouts[row_order].order[shown_row])
        except (KeyError, TypeError, ValueError, IndexError):
            return no_update, no_update
        if not (0 <= col < atlas.shape[1]) or not atlas.available[row, col]:
            return no_update, no_update
        return {"r": row, "c": col}, no_update

    @app.callback(
        Output("f1", "value"), Output("f2", "value"), Output("f3", "value"),
        Input("btn-clear", "n_clicks"), prevent_initial_call=True,
    )
    def clear_filters(_):
        return [], [], []

    @app.callback(Output("f1", "options"), Output("f2", "options"), Input("center-stat", "value"))
    def relabel_filters(stat):
        stat = "mean" if stat == "mean" else "median"
        return (
            [{"label": f" {stat} > 0 with p20 <= 0", "value": "on"}],
            [{"label": f" direct gene near zero ({stat}), strong DR cell on the row", "value": "on"}],
        )

    def make_table(frame: pd.DataFrame, highlight: str | None):
        def cell(col, value):
            if isinstance(value, (bool, np.bool_)):
                return "yes" if value else "-"
            if col in ("malignant cells", "non-malignant cells"):
                return f"{int(value):,}"
            if col == "patient":
                return value
            if col == "AUPRC / prevalence":
                return f"{value:.2f}x"
            if col == "prevalence %":
                return f"{value:.2f}"
            return fmt_pp(value, 2) if "lift" in col or "median" in col else f"{value:.2f}"

        head = html.Tr([html.Th(c) for c in frame.columns])
        body = [
            html.Tr([html.Td(cell(c, row[c])) for c in frame.columns], className="sel" if row["patient"] == highlight else "")
            for _, row in frame.iterrows()
        ]
        return html.Table([html.Thead(head), html.Tbody(body)], className="pt")

    @app.callback(Output("cell-body", "children"), *view_inputs)
    def render_cell(*args):
        view = raw_view(*args)
        if view.selected is None:
            return html.Div(
                [
                    html.Div("Click any colored cell on the cohort map."),
                    html.Div("You get the seven patient lifts, prevalence, held-out AUPRC and cell counts, plus the same panel across Stage 1."),
                ],
                className="placeholder",
            )
        r, c = view.selected
        title, subtitle, chips = cell_header(atlas, view, r, c)
        highlight = view.patient if view.mode in PATIENT_MODES else None
        return [
            html.Div(title, style={"fontWeight": 700, "fontSize": "15px"}),
            html.Div(subtitle, className="sub", style={"marginBottom": "4px"}),
            html.Div([html.Span([html.B(f"{k}: "), v], className="chip") for k, v in chips]),
            dcc.Graph(figure=patient_row_figure(atlas, r, c, highlight), config=graph_config, style={"height": "330px"}),
            dcc.Graph(figure=lift_rank_figure(atlas, view, r, c), config=graph_config, style={"height": "330px"}),
            make_table(patient_table(atlas, view, r, c), highlight),
            dcc.Graph(figure=profile_row_figure(atlas, view, r, c), config=graph_config, style={"height": "360px"}),
        ]

    return app


# ======================================================================================
# 9. CLI
# ======================================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local, read-only explorer for the old34 D1 transfer atlas.")
    parser.add_argument("--data-dir", type=Path, help="experiment directory or an exported bundle (default: this repo's experiment)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use an SSH tunnel to share)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="Dash hot reload for quick design iteration")
    parser.add_argument("--check", action="store_true", help="load and validate the inputs, print a report, do not serve")
    parser.add_argument("--export-bundle", type=Path, metavar="DIR", help="write a small shareable bundle to a NEW directory and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir or find_experiment_dir()
    if data_dir is None:
        print(f"Could not find experiments/{EXPERIMENT_ID} above this file; pass --data-dir.", file=sys.stderr)
        return 2
    try:
        if args.export_bundle:
            written = export_bundle(data_dir, args.export_bundle)
            print("Wrote bundle:")
            for path in written:
                print(f"  {path}  ({path.stat().st_size / 1e6:.2f} MB)")
            print(f"\nShare that directory, then run:  python {Path(__file__).name} --data-dir {args.export_bundle.name}")
            return 0
        before = input_fingerprint(Path(data_dir))
        started = time.perf_counter()
        atlas = load_atlas(Path(data_dir))
        after = input_fingerprint(Path(data_dir))
    except DataError as error:
        print(f"Data error: {error}", file=sys.stderr)
        return 2
    print(f"[d1-atlas] loaded {atlas.shape[0]}x{atlas.shape[1]} atlas, {atlas.audit['cells']} cells, in {time.perf_counter() - started:.1f}s")
    if args.check:
        failed = 0
        for name, status, detail in run_checks(atlas, before, after):
            failed += status == "FAIL"
            print(f"  [{status:4s}] {name}: {detail}")
        print("\n".join(["", *audit_lines(atlas)]))
        return 1 if failed else 0
    app = create_app(atlas)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    print(f"[d1-atlas] serving http://{args.host if not loopback else '127.0.0.1'}:{args.port}  (read-only; Ctrl-C to stop)")
    if loopback:
        print(f"[d1-atlas] teammates: ssh -L {args.port}:127.0.0.1:{args.port} <user>@<this-host>, then open http://127.0.0.1:{args.port}")
    else:
        print("[d1-atlas] WARNING: bound beyond localhost; the Dash dev server has no authentication.", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
