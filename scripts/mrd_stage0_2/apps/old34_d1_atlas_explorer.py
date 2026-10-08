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
    python old34_d1_atlas_explorer.py --factor-explorer-url URL   # deep-link target (default: this host, port 8051)

Inputs, all under ``<experiment>/analysis/scorecards`` (or flat inside a bundle):

* ``stage2_diagnostic/old34_d1_atlas_representation_table.csv``      783 cells (one per panel x configuration)
* ``stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv`` 5481 rows (783 cells x 7 patients)
* ``stage2_figure3_sharedness_v2_jun4/v2_patient_leverage_on_panel_method_median_lift.csv`` (optional)
* four ``stage2_sharedness_lopo_scorecard.csv`` files, only to join the Stage 2 nonzero-coefficient
  count at each cell's *selected* regularization (optional; a bundle carries the joined 783-row table).
* W7.1 (2026-10-08, additive): the four matching ``<experiment>/stage2_supervised/multiobjective/
  .../sharedness_lopo/by_heldout_patient.csv`` tables, only to read the nonzero-coefficient count of
  every LOPO fold at the selected regularization (the per-fold strip; optional).  A bundle carries
  them as ``fold_nonzero_<patient>`` columns of the joined nonzero table.
* ``analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_index.csv`` (optional): only
  to say whether the D18 factor cards cover a representation before deep-linking to the factor
  explorer (``apps/old34_factor_explorer.py``, port 8051).

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
from urllib.parse import urlencode

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
    # W7.1 additions (2026-10-08)
    "zero_folds": "# supported LOPO folds with an all-zero model (lift = 0 by construction)",
    "selected_c": "Selected Stage 2 C (log10; cell text = penalty)",
}
FOLD_MODES = ("zero_folds", "selected_c")  # need the per-fold table / the selected regularization

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

# ---- W7.1 (2026-10-08): per-fold activity at the selected regularization (additive) -----------
# scorecard tag (as in SCORECARDS) -> per-fold LOPO table below the experiment dir.  Each scorecard
# row has one row per held-out patient (13 folds) there, with that fold's nonzero-coefficient count.
FOLD_TABLES = {
    "june5_biological_dr": "stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv",
    "direct_gene_all_v1": "stage2_supervised/multiobjective/old34_direct_gene_all_v1/sharedness_lopo/by_heldout_patient.csv",
    "controls_fullgrid_v1": "stage2_supervised/multiobjective/old34_controls_fullgrid_v1/sharedness_lopo/by_heldout_patient.csv",
    "controls_direct_gene_v1": "stage2_supervised/multiobjective/old34_controls_direct_gene_v1/sharedness_lopo/by_heldout_patient.csv",
}
FOLD_TABLE_COLUMNS = (
    "stage0_panel_id", "representation_family", "stage1_method", "effective_k", "penalty", "C", "l1_ratio",
    "heldout_patient", "n_malignant", "n_normal", "nonzero_coefficient_count", "stage2_auprc",
)
FOLD_COLUMN_PREFIX = "fold_nonzero_"  # bundle: one integer column per LOPO fold in the nonzero table
FOLD_MANIFEST_KEY = "lopo_folds"  # bundle_manifest.json: {patient: {n_malignant, n_normal}}
C_POLICY_TEXT = (
    "c_policy global_selected: one penalty and C per cell (D1 rule: max pooled LOPO AUPRC), the same in every "
    "fold; nothing is refit"
)
ZERO_FOLD_TEXT = "lift = 0 by construction"
PURPLES8 = ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"]
PENALTY_SHORT = {"l1": "L1", "l2": "L2", "elasticnet": "EN"}
FACTOR_EXPLORER_PORT = 8051
FACTOR_CARDS_INDEX = "analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_index.csv"


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
    # ---- W7.1 additions (2026-10-08); load_atlas fills them, they stay empty when inputs are absent
    sel_penalty: np.ndarray | None = None  # (R, C) object: selected penalty, '' if unknown
    sel_C: np.ndarray | None = None  # (R, C) selected C (NaN if unknown)
    sel_l1_ratio: np.ndarray | None = None
    fold_patients: list[str] = field(default_factory=list)  # every LOPO fold (held-out patient), sorted
    fold_supported: np.ndarray | None = None  # (F,) fold patient is one of the supported patients
    fold_nonzero: np.ndarray | None = None  # (F, R, C) nonzero coefficients per fold (NaN if unknown)
    fold_n_malignant: np.ndarray | None = None  # (F,) NaN if unknown (older bundles)
    fold_n_normal: np.ndarray | None = None
    factor_card_reps: set | None = None  # {(panel, method, K)} with D18 factor cards; None = unknown

    @property
    def shape(self) -> tuple[int, int]:
        return self.available.shape

    @property
    def has_folds(self) -> bool:
        return self.fold_nonzero is not None and bool(np.isfinite(self.fold_nonzero).any())

    @property
    def has_selected_c(self) -> bool:
        return self.sel_C is not None and bool(np.isfinite(self.sel_C).any())

    @property
    def supported_fold_index(self) -> np.ndarray:
        """Positions in ``fold_patients`` of the supported patients, in ``patients`` order."""
        return np.array([self.fold_patients.index(p) for p in self.patients if p in self.fold_patients], dtype=int)

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


def load_atlas(data_dir: Path, *, with_nonzero: bool = True, with_folds: bool = True) -> Atlas:
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
    atlas.sel_penalty = np.full((n_panels, n_configs), "", dtype=object)
    atlas.sel_C = np.full((n_panels, n_configs), np.nan)
    atlas.sel_l1_ratio = np.full((n_panels, n_configs), np.nan)
    table = None
    if with_nonzero:
        table = _attach_nonzero(atlas, files, row_of, col_of)
    _attach_leverage(atlas, files, row_of, col_of)
    if with_folds:
        _attach_folds(atlas, table, files, row_of, col_of)
    _attach_factor_card_index(atlas)
    return atlas


def _attach_nonzero(atlas: Atlas, files: dict, row_of: dict, col_of: dict) -> pd.DataFrame | None:
    """Join the nonzero counts (and, W7.1, the selected penalty / C); returns the joined table."""
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
        return None
    full_table = table
    ri = table["stage0_panel_id"].map(row_of)
    ci = table["configuration"].map(col_of)
    keep = ri.notna() & ci.notna()
    table, ri, ci = table.loc[keep], ri[keep].to_numpy(int), ci[keep].to_numpy(int)
    atlas.nonzero_mean[ri, ci] = table["nonzero_mean_across_folds"].to_numpy(float)
    atlas.nonzero_max[ri, ci] = table["nonzero_max_across_folds"].to_numpy(float)
    atlas.reg_label[ri, ci] = [
        _regularization_text(p, c, l1) for p, c, l1 in zip(table["penalty"], table["C"].astype(float), table["l1_ratio"].astype(float))
    ]
    atlas.sel_penalty[ri, ci] = table["penalty"].astype(str).str.lower().to_numpy(object)
    atlas.sel_C[ri, ci] = pd.to_numeric(table["C"], errors="coerce").to_numpy(float)
    atlas.sel_l1_ratio[ri, ci] = pd.to_numeric(table["l1_ratio"], errors="coerce").to_numpy(float)
    atlas.audit["nonzero_cells"] = int(np.isfinite(atlas.nonzero_mean).sum())
    uncovered = int((atlas.available & ~np.isfinite(atlas.nonzero_mean)).sum())
    if uncovered:
        atlas.notes.append(f"{uncovered} atlas cells have no nonzero-coefficient count (no scorecard row).")
    return full_table


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


# ---- W7.1 (2026-10-08): per-fold activity, selected regularization, factor-card index --------


def fold_table_paths(data_dir: Path) -> dict[str, Path]:
    return {tag: Path(data_dir) / rel for tag, rel in FOLD_TABLES.items()}


def fold_input_fingerprint(data_dir: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of the per-fold LOPO tables that exist (kept apart from input_fingerprint)."""
    paths = fold_table_paths(data_dir).values()
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()}


def _float_key(values) -> np.ndarray:
    """Join key for C / l1_ratio written by two CSV writers: 9 significant digits, 'nan' for blanks."""
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    return np.array(["nan" if not np.isfinite(v) else f"{v:.9g}" for v in numbers], dtype=object)


def _configuration(method: pd.Series, effective_k: pd.Series) -> np.ndarray:
    k = pd.to_numeric(effective_k, errors="coerce").round().astype("Int64").astype(str)
    return np.where(method.eq("direct_gene"), "direct_gene", method.astype(str) + "|K=" + k)


def build_fold_table(data_dir: Path, selected: pd.DataFrame) -> pd.DataFrame | None:
    """Nonzero count of every LOPO fold at each cell's selected regularization (read-only).

    ``selected`` is the joined nonzero table (one row per panel x configuration, with the
    ``source_scorecard`` tag).  Returns one row per cell x held-out patient, or None when a per-fold
    table is missing.  Rows are matched on panel, configuration, penalty, C and l1_ratio.
    """
    paths = fold_table_paths(data_dir)
    if not all(p.is_file() for p in paths.values()) or "source_scorecard" not in selected.columns:
        return None
    families = {tag: family for tag, _, family in SCORECARDS}
    keys = ["stage0_panel_id", "configuration", "_penalty", "_C", "_l1"]
    parts = []
    for tag, path in paths.items():
        chosen = selected.loc[selected["source_scorecard"].eq(tag), ["stage0_panel_id", "configuration", "penalty", "C", "l1_ratio"]]
        if chosen.empty:
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in FOLD_TABLE_COLUMNS, low_memory=False)
        missing = [c for c in FOLD_TABLE_COLUMNS if c not in frame.columns]
        if missing:
            raise DataError(f"per-fold table {path} is missing columns: {missing}")
        if families.get(tag) is not None:
            frame = frame.loc[frame["representation_family"].eq(families[tag])]
        frame = frame.assign(
            configuration=_configuration(frame["stage1_method"], frame["effective_k"]),
            _penalty=frame["penalty"].astype(str).str.lower().to_numpy(object),
            _C=_float_key(frame["C"]),
            _l1=_float_key(frame["l1_ratio"]),
        )
        chosen = chosen.assign(
            _penalty=chosen["penalty"].astype(str).str.lower().to_numpy(object),
            _C=_float_key(chosen["C"]),
            _l1=_float_key(chosen["l1_ratio"]),
        )
        parts.append(chosen[keys].merge(frame, on=keys, how="inner"))
    if not parts:
        return None
    long = pd.concat(parts, ignore_index=True).rename(columns={"nonzero_coefficient_count": "nonzero", "stage2_auprc": "auprc"})
    long["heldout_patient"] = long["heldout_patient"].astype(str)
    return long[["stage0_panel_id", "configuration", "heldout_patient", "nonzero", "auprc", "n_malignant", "n_normal"]]


def fold_wide_columns(long: pd.DataFrame) -> pd.DataFrame:
    """Long per-fold table -> one row per cell with integer ``fold_nonzero_<patient>`` columns (bundle form)."""
    wide = long.pivot(index=["stage0_panel_id", "configuration"], columns="heldout_patient", values="nonzero")
    wide = wide.reindex(columns=sorted(wide.columns)).round().astype("Int64")
    wide.columns = [f"{FOLD_COLUMN_PREFIX}{p}" for p in wide.columns]
    return wide.reset_index()


def fold_patient_meta(long: pd.DataFrame) -> dict[str, dict[str, int]]:
    """{patient: {n_malignant, n_normal}} of every LOPO fold (the held-out patient's cell counts)."""
    first = long.groupby("heldout_patient")[["n_malignant", "n_normal"]].first()
    return {
        str(p): {k: int(v) for k, v in row.items() if pd.notna(v)} for p, row in first.iterrows()
    }


def _bundle_fold_meta(data_dir: Path) -> dict:
    path = Path(data_dir) / "bundle_manifest.json"
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")).get(FOLD_MANIFEST_KEY) or {})
    except (OSError, ValueError, AttributeError):
        return {}


def _attach_folds(atlas: Atlas, table: pd.DataFrame | None, files: dict, row_of: dict, col_of: dict) -> None:
    """Fill ``fold_*``: the nonzero count of every LOPO fold's model at the selected regularization."""
    atlas.audit["fold_cells"] = 0
    if table is None:
        atlas.notes.append("Per-fold strip unavailable: it needs the Stage 2 selection (nonzero table).")
        return
    fold_cols = [c for c in table.columns if str(c).startswith(FOLD_COLUMN_PREFIX)]
    if fold_cols:  # exported bundle: wide integer columns of the nonzero table
        long = table.melt(
            id_vars=["stage0_panel_id", "configuration"], value_vars=fold_cols, var_name="heldout_patient", value_name="nonzero"
        )
        long["heldout_patient"] = long["heldout_patient"].str[len(FOLD_COLUMN_PREFIX):]
        long["auprc"] = np.nan
        meta = _bundle_fold_meta(atlas.data_dir)
        long["n_malignant"] = long["heldout_patient"].map(lambda p: meta.get(p, {}).get("n_malignant", np.nan))
        long["n_normal"] = long["heldout_patient"].map(lambda p: meta.get(p, {}).get("n_normal", np.nan))
        atlas.audit["fold_source"] = f"bundle columns {FOLD_COLUMN_PREFIX}<patient> of {NONZERO_TABLE_NAME}"
    elif files.get("nonzero_table") is None:  # experiment dir: read the per-fold LOPO tables
        before = fold_input_fingerprint(atlas.data_dir)
        try:
            long = build_fold_table(atlas.data_dir, table)
        except (DataError, ValueError, OSError) as error:
            atlas.notes.append(f"Per-fold strip unavailable: {error}")
            return
        after = fold_input_fingerprint(atlas.data_dir)
        if long is None:
            atlas.notes.append("Per-fold strip unavailable: the sharedness_lopo/by_heldout_patient.csv tables were not found.")
            return
        atlas.audit["fold_inputs_files"] = len(before)
        atlas.audit["fold_inputs_unmodified"] = before == after
        atlas.audit["fold_source"] = "joined from the four sharedness_lopo/by_heldout_patient.csv files"
    else:
        atlas.notes.append("Per-fold strip unavailable: this bundle predates W7.1 (no fold_nonzero_* columns).")
        return
    if long.duplicated(["stage0_panel_id", "configuration", "heldout_patient"]).any():
        atlas.notes.append("Per-fold strip unavailable: more than one per-fold row for a selected fit and fold.")
        return
    ri, ci = long["stage0_panel_id"].map(row_of), long["configuration"].map(col_of)
    keep = ri.notna() & ci.notna()
    long, ri, ci = long.loc[keep], ri[keep].to_numpy(int), ci[keep].to_numpy(int)
    patients = sorted(long["heldout_patient"].astype(str).unique())
    if not patients:
        atlas.notes.append("Per-fold strip unavailable: no per-fold row matches an atlas cell.")
        return
    fi = long["heldout_patient"].map({p: i for i, p in enumerate(patients)}).to_numpy(int)
    n_rows, n_cols = atlas.shape
    nonzero = np.full((len(patients), n_rows, n_cols), np.nan)
    nonzero[fi, ri, ci] = pd.to_numeric(long["nonzero"], errors="coerce").to_numpy(float)
    first = long.groupby("heldout_patient")[["n_malignant", "n_normal"]].first().reindex(patients)
    atlas.fold_patients = patients
    atlas.fold_supported = np.array([p in atlas.patients for p in patients])
    atlas.fold_nonzero = nonzero
    atlas.fold_n_malignant = pd.to_numeric(first["n_malignant"], errors="coerce").to_numpy(float)
    atlas.fold_n_normal = pd.to_numeric(first["n_normal"], errors="coerce").to_numpy(float)

    a = atlas.audit
    complete = np.isfinite(nonzero).all(axis=0) & atlas.available
    a["fold_cells"] = int(complete.sum())
    a["fold_count"] = len(patients)
    a["fold_missing_supported"] = [p for p in atlas.patients if p not in patients]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fold_mean, fold_max = np.nanmean(nonzero, axis=0), np.nanmax(nonzero, axis=0)
    both = complete & np.isfinite(atlas.nonzero_mean)
    a["fold_mean_vs_scorecard_max_abs_diff"] = float(np.abs(fold_mean - atlas.nonzero_mean)[both].max()) if both.any() else None
    a["fold_max_vs_scorecard_max_abs_diff"] = float(np.abs(fold_max - atlas.nonzero_max)[both].max()) if both.any() else None
    auprc = pd.to_numeric(long["auprc"], errors="coerce").to_numpy(float)
    a["fold_auprc_vs_patient_table_max_abs_diff"] = None
    if np.isfinite(auprc).any():
        k_of = {p: k for k, p in enumerate(atlas.patients)}
        ki = long["heldout_patient"].map(k_of)
        sup = ki.notna().to_numpy()
        diff = np.abs(auprc[sup] - atlas.auprc[ki[sup].to_numpy(int), ri[sup], ci[sup]])
        a["fold_auprc_vs_patient_table_max_abs_diff"] = float(np.nanmax(diff)) if diff.size else None
    zero = zero_fold_count(atlas)
    n_sup = len(atlas.supported_fold_index)
    a["fold_zero_supported_pairs"] = int(np.nansum(zero))
    a["fold_cells_all_supported_zero"] = int((zero == n_sup).sum())
    a["fold_cells_partially_zero"] = int(((zero > 0) & (zero < n_sup)).sum())


def zero_fold_pairs(atlas: Atlas) -> list[tuple[int, int]]:
    """(index into ``patients``, index into ``fold_patients``) for every supported fold present."""
    return [(k, atlas.fold_patients.index(p)) for k, p in enumerate(atlas.patients) if p in atlas.fold_patients]


def zero_fold_count(atlas: Atlas) -> np.ndarray:
    """(R, C) number of supported folds whose model is all-zero; NaN where unknown or unavailable."""
    out = np.full(atlas.shape, np.nan)
    if not atlas.has_folds:
        return out
    idx = [f for _, f in zero_fold_pairs(atlas)]
    values = atlas.fold_nonzero[idx]
    known = np.isfinite(values).all(axis=0) & atlas.available
    with np.errstate(invalid="ignore"):
        counts = (values == 0).sum(axis=0)
    out[known] = counts[known]
    return out


def zero_fold_patients(atlas: Atlas, r: int, c: int) -> list[str]:
    """Supported patients whose LOPO fold fit an all-zero model at this cell (lift = 0 by construction)."""
    if not atlas.has_folds:
        return []
    return [atlas.patients[k] for k, f in zero_fold_pairs(atlas) if atlas.fold_nonzero[f, r, c] == 0]


def selected_regularization_text(atlas: Atlas, r: int, c: int) -> str:
    """'penalty l1 | C 0.0001' for the cell's selected (global) regularization, '' when unknown."""
    if atlas.sel_C is None or not np.isfinite(atlas.sel_C[r, c]):
        return ""
    penalty = str(atlas.sel_penalty[r, c])
    text = f"penalty {penalty} | C {atlas.sel_C[r, c]:.3g}"
    if penalty == "elasticnet" and np.isfinite(atlas.sel_l1_ratio[r, c]):
        text += f" | l1_ratio {atlas.sel_l1_ratio[r, c]:g}"
    return text


def fold_strip(atlas: Atlas, r: int, c: int) -> list[dict]:
    """One entry per LOPO fold: the supported folds (in ``patients`` order), then the unsupported ones.

    status is 'active' (at least one nonzero coefficient), 'all_zero' (constant score: held-out AP equals
    prevalence, so a supported fold's lift is 0 by construction) or 'unknown'.
    """
    if not atlas.has_folds:
        return []
    order = [f for _, f in zero_fold_pairs(atlas)] + [f for f, s in enumerate(atlas.fold_supported) if not s]
    out = []
    for f in order:
        patient = atlas.fold_patients[f]
        value = float(atlas.fold_nonzero[f, r, c])
        supported = bool(atlas.fold_supported[f])
        status = "unknown" if not np.isfinite(value) else ("active" if value > 0 else "all_zero")
        n_mal = atlas.fold_n_malignant[f] if atlas.fold_n_malignant is not None else np.nan
        model = {
            "active": f"{int(value)} nonzero coefficients",
            "all_zero": "all-zero model (constant score; held-out AP = prevalence)",
            "unknown": "no per-fold row",
        }[status]
        if supported:
            lift = 100 * atlas.lift[atlas.patients.index(patient), r, c]
            title = f"{patient} held out (supported): {model}; held-out lift {fmt_pp(lift)} pp"
            if status == "all_zero":
                title += f" - {ZERO_FOLD_TEXT}, not evidence of absent signal"
        else:
            why = f"{int(n_mal)} malignant cells" if np.isfinite(n_mal) else "below the support rule"
            title = f"{patient} held out (unsupported: {why}; not in the lift summaries): {model}"
        out.append(dict(
            patient=patient, supported=supported, nonzero=value, status=status,
            badge=ZERO_FOLD_TEXT if supported and status == "all_zero" else "", title=title,
        ))
    return out


def fold_strip_summary(atlas: Atlas, r: int, c: int) -> str:
    strip = fold_strip(atlas, r, c)
    if not strip:
        return "per-fold activity unavailable"
    sup = [s for s in strip if s["supported"]]
    uns = [s for s in strip if not s["supported"]]
    zero = [s["patient"] for s in sup if s["status"] == "all_zero"]
    text = f"{sum(s['status'] == 'active' for s in sup)} of {len(sup)} supported folds active"
    if zero:
        text += f"; all-zero ({ZERO_FOLD_TEXT}): {', '.join(zero)}"
    if uns:
        text += f" | unsupported folds: {sum(s['status'] == 'active' for s in uns)} of {len(uns)} active (not in the lift summaries)"
    return text


def factor_explorer_href(panel_id: str, config: str, base_url: str) -> str | None:
    """Deep link into apps/old34_factor_explorer.py (``?panel=&method=&k=``); None for direct genes."""
    if config == "direct_gene" or "|K=" not in config:
        return None
    method, k = config.split("|K=")
    return f"{base_url.rstrip('/')}/?{urlencode({'panel': panel_id, 'method': method, 'k': k})}"


def factor_cards_status(atlas: Atlas, r: int, c: int) -> str:
    """'covered' / 'not_covered' by the D18 factor cards, or 'unknown' when their index is absent."""
    config = CONFIGS[c]
    if atlas.factor_card_reps is None or "|K=" not in config:
        return "unknown"
    method, k = config.split("|K=")
    return "covered" if (atlas.panel_ids[r], method, int(k)) in atlas.factor_card_reps else "not_covered"


def _attach_factor_card_index(atlas: Atlas) -> None:
    for path in (atlas.data_dir / FACTOR_CARDS_INDEX, atlas.data_dir / Path(FACTOR_CARDS_INDEX).name):
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path, usecols=["panel_id", "method", "k"])
        except (ValueError, OSError) as error:
            atlas.notes.append(f"D18 factor-card index unreadable ({error}); deep links are shown without coverage.")
            return
        atlas.factor_card_reps = {(str(p), str(m), int(k)) for p, m, k in frame.drop_duplicates().itertuples(index=False)}
        atlas.audit["factor_card_representations"] = len(atlas.factor_card_reps)
        return


def leverage_caveat(atlas: Atlas, view: "View") -> tuple[str, str]:
    """(label, text) of the leverage-filter caveat shown under the filter (labelled in W7.1)."""
    a = atlas.audit
    if view.f3 and view.f3_source == "csv":
        return "Leverage-filter caveat (v2 CSV source)", (
            f"v2 CSV covers {a.get('leverage_csv_cells', 0)} of {a['cells']} cells (best-LOPO K per panel x method; no controls, "
            f"no direct gene). Its medians use each patient's own best regularization and differ from the heatmap's "
            f"median at {a.get('leverage_csv_basis_mismatch_cells', 0)} of those cells. Use 'D1 patient table' for a "
            f"leverage consistent with the colours."
        )
    if view.f3:
        return "Leverage filter (D1 patient table)", (
            "Computed from this atlas's seven lifts per cell: drop one patient, recompute the median. Consistent with the "
            "heatmap; covers every cell."
        )
    return "Leverage-filter caveat", (
        f"v2 CSV: {a.get('leverage_csv_cells', 0)}/{a['cells']} cells; built on a different regularization basis than the "
        f"heatmap (see Data audit)."
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
    if atlas.has_folds:  # W7.1
        n_sup = len(atlas.supported_fold_index)
        lines.append(
            f"per-fold strip: {a.get('fold_cells', 0)}/{a['cells']} cells x {a.get('fold_count', 0)} LOPO folds "
            f"({n_sup} supported) ({a.get('fold_source', 'n/a')}); all-zero in all {n_sup} supported folds: "
            f"{a.get('fold_cells_all_supported_zero', 0)} cells, in 1-{n_sup - 1}: {a.get('fold_cells_partially_zero', 0)} cells"
        )
    if atlas.factor_card_reps is not None:
        lines.append(f"D18 factor cards: {len(atlas.factor_card_reps)} representations (deep links to the factor explorer)")
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
    cell_text: np.ndarray | None = None  # W7.1: per-cell text (canonical order) instead of annotation(value)


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
    if mode == "zero_folds":  # W7.1
        n = len(atlas.supported_fold_index) if atlas.has_folds else len(atlas.patients)
        colours = [PURPLES8[min(round(i * (len(PURPLES8) - 1) / max(n, 1)), len(PURPLES8) - 1)] for i in range(n + 1)]
        return ColorSpec(
            zero_fold_count(atlas), _discrete(colours), -0.5, n + 0.5,  # one colour per integer 0..n
            f"all-zero supported folds (of {n})",
            f"Supported LOPO folds whose selected model is all-zero ({ZERO_FOLD_TEXT} in those folds)",
            lambda v, n=n: f"{int(v)} of {n} supported folds all-zero ({ZERO_FOLD_TEXT} there)",
            lambda v: f"{int(v)}",
            tickvals=list(range(n + 1)), ticktext=[str(i) for i in range(n + 1)],
        )
    if mode == "selected_c":  # W7.1
        sel_c = atlas.sel_C if atlas.sel_C is not None else np.full(atlas.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(atlas.available & (sel_c > 0), np.log10(sel_c), np.nan)
        finite = z[np.isfinite(z)]
        lo, hi = (float(np.floor(finite.min())), float(np.ceil(finite.max()))) if finite.size else (-4.0, 4.0)
        hi = hi if hi > lo else lo + 1.0
        penalties = atlas.sel_penalty if atlas.sel_penalty is not None else np.full(atlas.shape, "", dtype=object)
        text = np.vectorize(lambda p: PENALTY_SHORT.get(str(p), str(p)), otypes=[object])(penalties)
        ticks = list(range(int(lo), int(hi) + 1))
        return ColorSpec(
            z, VIRIDIS_SCALE, lo, hi, "selected C (log10)",
            f"Selected Stage 2 C per cell (cell text = penalty; {C_POLICY_TEXT})",
            lambda v: f"selected C {10 ** v:.3g}",
            lambda v: f"{10 ** v:.0e}",
            tickvals=[float(t) for t in ticks], ticktext=[f"1e{t}" for t in ticks], cell_text=text,
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
    zero_folds = zero_fold_count(atlas)  # W7.1
    n_sup = len(atlas.supported_fold_index) if atlas.has_folds else 0
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
            if np.isfinite(zero_folds[i, j]):  # W7.1
                folds = f"folds: {n_sup - int(zero_folds[i, j])}/{n_sup} supported active"
                if zero_folds[i, j] > 0:
                    folds += f"; all-zero ({ZERO_FOLD_TEXT}): {', '.join(zero_fold_patients(atlas, i, j))}"
                lines.append(folds)
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
    if spec.cell_text is not None:  # W7.1: e.g. the selected penalty in the selected-C mode
        annotation_text = np.where(finite, spec.cell_text[order], "")
    common = dict(
        x=xs, y=ys, xgap=1, ygap=1, hoverongaps=False, hoverinfo="text", hovertext=hover,
        zmin=spec.zmin, zmax=spec.zmax,
    )
    text_kwargs = dict(text=annotation_text, texttemplate="%{text}", textfont=dict(size=9)) if (
        view.annotate or view.mode in ("count_lift", "count_ratio") + FOLD_MODES
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
    if atlas.has_folds:  # W7.1: partial all-zero folds look like patient-specific capture
        zero = zero_fold_patients(atlas, r, c)
        n_sup = len(atlas.supported_fold_index)
        if zero and len(zero) < n_sup:
            chips.append(("all-zero supported folds", f"{len(zero)}/{n_sup} ({', '.join(zero)}): {ZERO_FOLD_TEXT} there"))
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
    # W7.1: carry the per-fold nonzero counts as integer fold_nonzero_<patient> columns (about 30 KB)
    fold_meta: dict = _bundle_fold_meta(data_dir) if files["nonzero_table"] is not None else {}
    if files["nonzero_table"] is None and not any(str(c).startswith(FOLD_COLUMN_PREFIX) for c in table.columns):
        long = build_fold_table(data_dir, table)
        if long is not None:
            table = table.merge(fold_wide_columns(long), on=["stage0_panel_id", "configuration"], how="left")
            fold_meta = fold_patient_meta(long)
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
    if fold_meta:
        manifest[FOLD_MANIFEST_KEY] = fold_meta
    for key, target in targets.items():
        if key == "nonzero_table":
            table.to_csv(target, index=False)
            derived = "selected-regularization nonzero counts joined from 4 LOPO scorecards"
            if any(str(c).startswith(FOLD_COLUMN_PREFIX) for c in table.columns):
                derived += f"; {FOLD_COLUMN_PREFIX}<patient> = per-fold counts from the 4 sharedness_lopo/by_heldout_patient.csv"
            manifest["files"][target.name] = {"derived": derived}
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
    rows.append(fold_check(atlas))  # W7.1: one new line, after the original eleven
    return rows


def fold_check(atlas: Atlas) -> tuple[str, str, str]:
    """W7.1 check: the per-fold strip is joined at the selected regularization and is self-consistent.

    PASS needs: every available cell has every fold; the fold mean / max equal the scorecard's nonzero
    mean / max; every all-zero supported fold has exactly zero lift; and, in an experiment dir, every
    supported fold's held-out AUPRC equals the D1 patient table and the per-fold tables were not modified.
    """
    name = "per-fold activity joined at selected regularization"
    a = atlas.audit
    if not atlas.has_folds:
        return name, "SKIP", "per-fold LOPO tables (or bundle fold_nonzero_* columns) not present"
    problems = []
    cells, n_folds, n_sup = a["cells"], a.get("fold_count", 0), len(atlas.supported_fold_index)
    if a.get("fold_cells", 0) != cells:
        problems.append(f"only {a.get('fold_cells', 0)}/{cells} cells have every fold")
    if a.get("fold_missing_supported"):
        problems.append(f"supported patients without a fold: {a['fold_missing_supported']}")
    tol = 1e-6
    mean_diff, max_diff = a.get("fold_mean_vs_scorecard_max_abs_diff"), a.get("fold_max_vs_scorecard_max_abs_diff")
    if mean_diff is None or mean_diff > tol or max_diff is None or max_diff > tol:
        problems.append(f"fold mean/max differ from the scorecard (max |diff| {mean_diff} / {max_diff})")
    pairs = zero_fold_pairs(atlas)
    zero_lifts = [
        atlas.lift[k][(atlas.fold_nonzero[f] == 0) & atlas.available] for k, f in pairs
    ]
    zero_lifts = np.concatenate(zero_lifts) if zero_lifts else np.array([])
    max_zero_lift = float(np.abs(zero_lifts).max()) if zero_lifts.size else 0.0
    if max_zero_lift >= EPS:
        problems.append(f"an all-zero supported fold has nonzero lift ({max_zero_lift:.1e})")
    auprc_diff = a.get("fold_auprc_vs_patient_table_max_abs_diff")
    if auprc_diff is not None and auprc_diff > 1e-12:
        problems.append(f"supported-fold AUPRC differs from the patient table (max |diff| {auprc_diff:.1e})")
    if a.get("fold_inputs_unmodified") is False:
        problems.append("a per-fold table changed while loading")
    detail = (
        f"{a.get('fold_cells', 0)}/{cells} cells x {n_folds} folds ({n_sup} supported); fold mean/max == scorecard "
        f"(max |diff| {mean_diff if mean_diff is not None else float('nan'):.1e}); {zero_lifts.size} all-zero supported folds, "
        f"max |lift| {max_zero_lift:.1e}; {a.get('fold_cells_partially_zero', 0)} cells all-zero in 1-{n_sup - 1} supported folds, "
        f"{a.get('fold_cells_all_supported_zero', 0)} in all {n_sup}"
    )
    if auprc_diff is not None:
        detail += f"; supported-fold AUPRC == patient table (max |diff| {auprc_diff:.1e})"
    if "fold_inputs_files" in a:
        detail += f"; {a['fold_inputs_files']} per-fold tables fingerprinted"
    if problems:
        return name, "FAIL", detail + " | " + "; ".join(problems)
    return name, "PASS", detail


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

# W7.1 (2026-10-08): per-fold strip, badge and deep-link styles (appended to CSS).
CSS_W71 = """
.strip-block { border: 1px solid #e1e4e8; border-radius: 6px; padding: 6px 10px; margin: 6px 0 8px; background: #fbfcfd; }
.strip-head { font-size: 12.5px; font-weight: 600; margin-bottom: 2px; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; align-items: stretch; margin: 4px 0; }
.strip-sep { width: 10px; }
.fold { width: 42px; border-radius: 3px; text-align: center; font-size: 10.5px; line-height: 1.25; padding: 2px 0; border: 1px solid transparent; cursor: default; }
.fold b { display: block; font-size: 11px; }
.fold.active { background: #d8eef0; border-color: #00838f; color: #004d55; }
.fold.all_zero { background: #fde4e1; border-color: #c62828; color: #8e1b1b; }
.fold.unknown { background: #f1f1f1; border-color: #bbb; color: #777; }
.fold.unsupported { opacity: .45; border-style: dashed; filter: grayscale(100%); }
.badge { display: inline-block; background: #c62828; color: #fff; border-radius: 9px; padding: 1px 8px; font-size: 11px; font-weight: 600; margin: 2px 6px 2px 0; }
.strip-legend { color: #667; font-size: 11px; line-height: 1.35; }
.deeplink { display: inline-block; margin: 4px 0 6px; font-size: 12.5px; }
.caveat b { color: #6b3a00; }
"""


def _factor_explorer_base(configured: str | None) -> str:
    """The configured factor-explorer URL, else this request's host on port 8051 (works through a tunnel)."""
    if configured:
        return configured
    try:
        from flask import request

        host, scheme = request.host, request.scheme
    except (ImportError, RuntimeError):
        host, scheme = "127.0.0.1", "http"
    if host.startswith("["):  # IPv6 literal
        hostname = host[: host.index("]") + 1]
    else:
        hostname = host.rsplit(":", 1)[0] if ":" in host else host
    return f"{scheme}://{hostname}:{FACTOR_EXPLORER_PORT}/"


def create_app(atlas: Atlas, factor_explorer_url: str | None = None):
    """Build the Dash app.  Dash is imported here so --check and the tests do not need it.

    ``factor_explorer_url`` (W7.1): base URL of apps/old34_factor_explorer.py for the cell-view deep
    link; None means "this page's host, port 8051".
    """
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

    graph_config = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        "toImageButtonOptions": {"format": "png", "filename": "old34_d1_atlas", "scale": 2},
    }
    app = Dash(__name__, title="Old34 D1 atlas explorer")
    app.index_string = (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
        f"<style>{CSS}{CSS_W71}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"
    )

    default_patient = DEFAULT_PATIENT if DEFAULT_PATIENT in atlas.patients else atlas.patients[0]
    unavailable_modes = {
        "nonzero": not atlas.has_nonzero, "zero_folds": not atlas.has_folds, "selected_c": not atlas.has_selected_c,
    }
    mode_options = [
        {"label": MODE_LABELS[m], "value": m, "disabled": unavailable_modes.get(m, False)} for m in MODE_LABELS
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
        if view.mode == "zero_folds" and not atlas.has_folds:  # W7.1
            lines.append(html.Div("Per-fold LOPO activity is not available in this data directory."))
        if view.mode in FOLD_MODES:
            lines.append(html.Div(C_POLICY_TEXT + ". An all-zero fold scores a constant: held-out AP = prevalence, so its lift is 0 by construction, not evidence of absent signal."))
        label, text = leverage_caveat(atlas, view)  # W7.1: same texts as before, now labelled
        if view.f3 and view.f3_source == "csv":
            lines.append(html.Div([html.B(f"{label}: "), text]))
        caveat = [html.B(f"{label}: "), text]
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

    # ---- W7.1: per-fold strip and the deep link to the factor explorer ----------------------
    def fold_strip_block(r: int, c: int):
        reg = selected_regularization_text(atlas, r, c)
        head = [html.Div("Per-fold Stage 2 model at the selected regularization", className="strip-head")]
        head.append(html.Div(
            [html.B("selected: "), reg or "unknown", f"  ({C_POLICY_TEXT})"], className="strip-legend",
        ))
        strip = fold_strip(atlas, r, c)
        if not strip:
            return html.Div(head + [html.Div("Per-fold activity is not available in this data directory.", className="strip-legend")], className="strip-block")
        boxes = []
        for k, item in enumerate(strip):
            if k and item["supported"] != strip[k - 1]["supported"]:
                boxes.append(html.Div(className="strip-sep"))
            value = "-" if item["status"] == "unknown" else ("0" if item["status"] == "all_zero" else f"{int(item['nonzero'])}")
            classes = f"fold {item['status']}" + ("" if item["supported"] else " unsupported")
            boxes.append(html.Div([html.B(item["patient"]), value], className=classes, title=item["title"]))
        zero = [s["patient"] for s in strip if s["badge"]]
        badges = [html.Span(f"{ZERO_FOLD_TEXT}: {', '.join(zero)}", className="badge")] if zero else []
        n_sup = sum(s["supported"] for s in strip)
        legend = (
            f"Left: the {n_sup} supported folds (they enter the lift summaries); right, greyed: the {len(strip) - n_sup} "
            "unsupported folds (fewer than 10 malignant cells or one class; never summarised). Number = nonzero "
            "coefficients of that fold's model. Red 0 = all-zero model: constant score, held-out AP = prevalence, so "
            "that patient's lift is 0 by construction (an L1 threshold at this C, not evidence of absent signal)."
        )
        return html.Div(
            head + [html.Div(boxes, className="strip"), html.Div(badges + [fold_strip_summary(atlas, r, c)], className="strip-legend"),
                    html.Div(legend, className="strip-legend")],
            className="strip-block",
        )

    def factor_link_block(r: int, c: int) -> list:
        href = factor_explorer_href(atlas.panel_ids[r], CONFIGS[c], _factor_explorer_base(factor_explorer_url))
        if href is None:
            return [html.Div("Factor cards: none for direct-gene cells (no Stage 1 factors).", className="strip-legend")]
        status = factor_cards_status(atlas, r, c)
        note = {
            "covered": "D18 factor cards exist for this representation.",
            "not_covered": "Not in the D18 shortlist: the explorer lists the carded representations of this panel.",
            "unknown": "D18 card coverage unknown here (index not in this data directory).",
        }[status]
        return [html.Div([
            html.A(f"Open factor cards: {config_label(CONFIGS[c])} (factor explorer, port {FACTOR_EXPLORER_PORT})",
                   href=href, target="_blank", rel="noopener"),
            html.Span(f"  {note}", className="strip-legend"),
        ], className="deeplink")]

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
            *factor_link_block(r, c),
            fold_strip_block(r, c),
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
    parser.add_argument(
        "--factor-explorer-url", default=None, metavar="URL",
        help=f"base URL of apps/old34_factor_explorer.py for the cell-view deep link (default: this host, port {FACTOR_EXPLORER_PORT})",
    )
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
    app = create_app(atlas, factor_explorer_url=args.factor_explorer_url)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    print(f"[d1-atlas] serving http://{args.host if not loopback else '127.0.0.1'}:{args.port}  (read-only; Ctrl-C to stop)")
    if loopback:
        print(f"[d1-atlas] teammates: ssh -L {args.port}:127.0.0.1:{args.port} <user>@<this-host>, then open http://127.0.0.1:{args.port}")
        if not args.factor_explorer_url:
            print(f"[d1-atlas] cell-view factor links open port {FACTOR_EXPLORER_PORT} on the same host; tunnel it too (-L {FACTOR_EXPLORER_PORT}:127.0.0.1:{FACTOR_EXPLORER_PORT})")
    else:
        print("[d1-atlas] WARNING: bound beyond localhost; the Dash dev server has no authentication.", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
