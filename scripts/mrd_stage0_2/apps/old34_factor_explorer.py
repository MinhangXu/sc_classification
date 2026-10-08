#!/usr/bin/env python3
"""Old34 factor explorer: a local, read-only Dash page over the D18 ``factor_cards_v1`` bundle (W7.2).

It answers "what is this Stage 1 factor?" for the representations the D18 factor cards cover: per
factor, the orientation from the selected LOPO probe, per-fold activity, univariate held-out lift per
supported patient, top genes on both poles, score distributions, ADT / cell-type correlates and
matches to other bases.  It reads ONLY the bundle's CSV/JSON files (never cards/*.md, never the
experiment's Stage 1/2 outputs), refits nothing and never writes into the experiment.  It does not
import the D1 atlas app; the atlas deep-links here with ``?panel=<id>&method=<m>&k=<K>``.

Run (needs dash, plotly, pandas, numpy; ``pip install -e '.[app]'``)::

    python old34_factor_explorer.py                        # http://127.0.0.1:8051
    python old34_factor_explorer.py --check                # validate the bundle, no server
    python old34_factor_explorer.py --export-bundle DIR    # gzip copy of the bundle CSV/JSON (<= 20 MB)
    python old34_factor_explorer.py --data-dir DIR         # an experiment dir or an exported bundle

Deep link: ``http://127.0.0.1:8051/?panel=<stage0_panel_id>&method=<pca|fa|factosig|factosig_promax>&k=<K>``
(optional ``&factor=<n>``).  Inputs: ``<experiment>/analysis/scorecards/stage2_diagnostic/factor_cards_v1/``
(schema: ``bundle_schema_v1.md`` there; producer
``scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py``).  In an experiment
dir only the index is read at start; every other table is read on first use (lazy).

Reading guide: lift = held-out AUPRC - the held-out patient's malignant prevalence (pp).  Stage 1 is
transductive (fit on all 13 patients), so LOPO lift is label transfer, not inductive deployment.  A
factor's sign is arbitrary; "malignant pole" follows sign(beta) of the selected probe.  Descriptive only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parents[3]  # apps/<file>.py -> repo root
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# ======================================================================================
# 1. Constants (schema contract: factor_cards_v1, schema_version 1)
# ======================================================================================

SCHEMA_VERSION = 1
SLUG = "factor_cards_v1"
BUNDLE_SUBDIR = "analysis/scorecards/stage2_diagnostic/factor_cards_v1"  # EXP-relative
MANIFEST = "factor_cards_manifest.json"
EXPORT_MANIFEST = "factor_explorer_export.json"
SCHEMA_DOC = "bundle_schema_v1.md"
DEFAULT_PORT = 8051
EXPORT_LIMIT_BYTES = 20_000_000  # an exported bundle is at most 20 MB

METHODS = ("pca", "fa", "factosig", "factosig_promax")
METHOD_LABEL = {"pca": "PCA", "fa": "FA", "factosig": "FactoSig", "factosig_promax": "FactoSig-promax"}
CARD_ID_RE = re.compile(r"^(.+)__(pca|fa|factosig|factosig_promax)__k(\d+)__f(\d{3})$")
MANIFEST_KEYS = ("schema_version", "slug", "inputs", "git_state", "created_by", "created_utc", "parameters", "counts", "runtime_seconds")

# key -> (file name, required columns, required file)
TABLES: dict[str, tuple[str, tuple[str, ...], bool]] = {
    "index": ("factor_cards_index.csv", (
        "card_id", "panel_id", "method", "k", "factor", "is_selected_driver", "n_active_folds", "sign_consistency",
        "orientation_sign", "eta2_normal", "eta2_malignant", "degenerate_flag", "effective_k", "determinacy",
        "matched_cards", "caveat_flags",
    ), True),
    "genes": ("factor_cards_genes.csv", ("card_id", "gene", "loading", "oriented_loading", "rank", "pole", "structure_corr"), True),
    "patient_lift": ("factor_cards_patient_lift.csv", ("card_id", "patient", "univariate_lift", "univariate_auroc", "n_malignant", "prevalence"), True),
    "fold_coefs": ("factor_cards_fold_coefs.csv", ("card_id", "heldout_patient", "penalty", "C", "coef"), True),
    "matches": ("factor_cards_matches.csv", ("card_id", "relation", "other_method", "other_k", "matched_card_id", "score_r", "abs_r"), False),
    "subspace": ("factor_cards_subspace.csv", ("representation_id", "other_representation_id", "cc_min", "cc_median", "cc_max"), False),
    "score_dist": ("factor_cards_score_dist.csv", ("card_id", "patient", "label", "n_cells", "oriented", "q10", "q25", "q50", "q75", "q90"), False),
    "adt": ("factor_cards_adt.csv", ("card_id", "label", "rank", "adt_marker", "spearman_pooled", "spearman_patient_stratified", "n_cells"), False),
    "celltype": ("factor_cards_celltype.csv", ("card_id", "label", "cell_type", "n_cells", "mean_oriented_z"), False),
    "phi": ("factor_cards_phi.csv", ("representation_id", "factor_i", "factor_j", "score_r"), False),
    "shortlist": ("factor_cards_shortlist.csv", ("representation_id",), False),
    "regression_checks": ("factor_cards_regression_checks.csv", ("check_id", "passed"), False),
}
CARD_TABLES = ("genes", "patient_lift", "fold_coefs", "matches", "score_dist", "adt", "celltype")

# Caveat vocabulary (bundle_schema_v1.md); unknown codes are shown verbatim.
CAVEATS = {
    "transductive_basis": "Stage 1 was fit on all 13 patients, including each held-out patient: LOPO lift is label transfer on a transductive basis, not inductive deployment.",
    "unoriented_all_zero": "beta is 0 in every supported fold, so the probe gives no orientation; loadings keep their arbitrary raw sign.",
    "unoriented_sign_flip": "beta changes sign across supported folds, so the factor has no stable orientation; loadings keep their arbitrary raw sign.",
    "has_all_zero_folds": "at least one supported fold fit an all-zero model; its lift is 0 by construction, not evidence of absent signal.",
    "dense_fit": "dense fit: per-factor betas are partial effects among many correlated contributions and are not interpretable one factor at a time.",
    "l1_selected_absent": "never selected by the fitted probe; check the univariate held-out lift before reading the zero as absence of signal (L1 at a fixed C keeps one of several near-equivalent axes).",
    "zero_variance": "degenerate factor with zero score variance (beyond the representation's effective K).",
    "scale_outlier": "blown-up or collapsed factor scale (score SD more than 10x from the representation median); not degeneracy, since Stage 2 standardizes scores and promax column scale is arbitrary.",
    "near_duplicate_oblique": "oblique fit with a near-duplicate factor (|r| >= 0.9); beta is a partial effect within the pair.",
    "low_determinacy": "factor-score determinacy below 0.707: equally valid score sets can be uncorrelated, so do not name this factor as a unique program.",
    "small_eigengap": "relative eigengap to a neighbouring factor is 10% or less: the individual axis is not identifiable (rotation within the block).",
    "generative_loadings": "FA / FactoSig loadings are generative (how expected expression moves with the factor), not the weights that compute the score.",
    "oblique_pattern_loadings": "promax loadings are a pattern matrix with arbitrary column scale; read them with the structure correlations and Phi.",
    "ablation_is_attribution": "ablation dAP and single-factor lift are attributions inside the fitted model, not causal effects.",
    "small_patient_support": "at least one supported patient has fewer than 25 malignant cells; its lift is noisy.",
    "reconstruction_mismatch": "reconstructed held-out AP differs from the stored atlas AP by more than 1e-4 in a supported fold (tiny-beta float32 ties, or a selection mismatch).",
}
BANNER = (
    "Read-only view of the D18 factor_cards_v1 bundle; descriptive only. Stage 1 is transductive (fit on all 13 "
    "patients), so held-out-patient (LOPO) lift is label transfer, not inductive deployment. A factor's sign is "
    "arbitrary: the 'malignant pole' follows sign(beta) of the globally selected probe, and unoriented factors keep "
    "their raw sign. L1 at a fixed C keeps one of several near-equivalent axes, so a zero beta or an all-zero fold "
    "(lift = 0 by construction) is not absence of signal. Ablation dAP is attribution inside the fitted model, not a "
    "causal effect. Name a factor only from both poles; cell-level lift is not recurrence risk."
)
ZERO_FOLD_TEXT = "lift = 0 by construction"
POLE_ORDER = ("malignant", "normal", "positive", "negative")
LIFT_BLUE, AGNOSTIC_GRAY, FITTED_ORANGE = "#4472C4", "#7f7f7f", "#D55E00"
NORMAL_COLOR, MALIGNANT_COLOR = "#4c9be8", "#d6604d"


class DataError(RuntimeError):
    """Raised with an actionable message when the bundle is missing or inconsistent."""


def fmt_pp(x, digits: int = 2) -> str:
    """Lift fraction -> signed percentage points; float noise prints as 0."""
    try:
        value = 100 * float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:+.{digits}f}" if value else f"{value:.{digits}f}"


def fmt_num(x, digits: int = 3) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:.{digits}g}" if np.isfinite(value) else "n/a"


def as_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    try:
        return bool(x) and not (isinstance(x, float) and np.isnan(x))
    except (TypeError, ValueError):
        return False


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    """A column, or NaN when this (optional, extra) column is absent from the bundle."""
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)


def split_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part for part in str(value).split(";") if part]


def card_id_for(panel: str, method: str, k: int, factor: int) -> str:
    return f"{panel}__{method}__k{int(k)}__f{int(factor):03d}"


def representation_id(panel: str, method: str, k: int) -> str:
    return f"{panel}|dr|{method}|{int(k)}|42"


# ======================================================================================
# 2. Bundle (lazy, read-only)
# ======================================================================================


def default_data_dir() -> Path | None:
    """The canonical experiment dir via sc_classification.utils.paths (SC_* env vars), if it exists."""
    try:
        from sc_classification.utils.paths import experiment_dir

        return experiment_dir(must_exist=True)
    except (ImportError, FileNotFoundError, OSError, ValueError):
        return None


def resolve_bundle_root(data_dir: Path) -> tuple[Path, str]:
    """(directory holding the manifest, 'experiment' | 'bundle')."""
    data_dir = Path(data_dir)
    if (data_dir / BUNDLE_SUBDIR / MANIFEST).is_file():
        return data_dir / BUNDLE_SUBDIR, "experiment"
    if (data_dir / MANIFEST).is_file():
        return data_dir, "bundle"
    raise DataError(
        f"factor_cards_v1 manifest not found; searched:\n  {data_dir / BUNDLE_SUBDIR / MANIFEST}\n  {data_dir / MANIFEST}\n"
        "Has D18 (W5.1, run_d18_factor_cards.py) run for this experiment?"
    )


def table_path(root: Path, filename: str) -> Path | None:
    for candidate in (root / filename, root / f"{filename}.gz"):
        if candidate.is_file():
            return candidate
    return None


def bundle_fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every CSV/JSON in the bundle dir: proves loading changed nothing."""
    files = [p for p in sorted(Path(root).iterdir()) if p.is_file() and (p.suffix in (".csv", ".json", ".gz"))]
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in files}


@dataclass
class CardBundle:
    """The factor_cards_v1 tables.  Only the index is read at start; the rest on first use."""

    data_dir: Path
    root: Path = field(init=False)
    mode: str = field(init=False)
    manifest: dict = field(init=False)
    index: pd.DataFrame = field(init=False)
    _tables: dict = field(default_factory=dict, init=False, repr=False)
    _groups: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.root, self.mode = resolve_bundle_root(self.data_dir)
        try:
            self.manifest = json.loads((self.root / MANIFEST).read_text(encoding="utf-8"))
        except ValueError as error:
            raise DataError(f"{MANIFEST} is not valid JSON: {error}") from error
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise DataError(f"unsupported schema_version {self.manifest.get('schema_version')!r}; this app reads {SCHEMA_VERSION}")
        index = self.table("index")
        index["k"] = pd.to_numeric(index["k"], errors="coerce").astype("Int64")
        index["factor"] = pd.to_numeric(index["factor"], errors="coerce").astype("Int64")
        if index["card_id"].duplicated().any():
            raise DataError("factor_cards_index.csv has duplicate card_id values")
        self.index = index.set_index("card_id", drop=False)
        reps: dict[tuple[str, str, int], list[str]] = {}
        for cid, panel, method, k, factor in index[["card_id", "panel_id", "method", "k", "factor"]].itertuples(index=False):
            reps.setdefault((str(panel), str(method), int(k)), []).append((int(factor), cid))
        self.reps = {key: [cid for _, cid in sorted(values)] for key, values in reps.items()}

    # ---- tables ------------------------------------------------------------------------
    def has_table(self, key: str) -> bool:
        return table_path(self.root, TABLES[key][0]) is not None

    def table(self, key: str) -> pd.DataFrame | None:
        """Read (once) and validate one table; None when an optional table is absent."""
        if key in self._tables:
            return self._tables[key]
        filename, required, is_required = TABLES[key]
        path = table_path(self.root, filename)
        if path is None:
            if is_required:
                raise DataError(f"required table {filename} not found in {self.root}")
            self._tables[key] = None
            return None
        frame = pd.read_csv(path, low_memory=False)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise DataError(f"{filename} is missing required columns: {missing}")
        self._tables[key] = frame
        return frame

    @property
    def loaded(self) -> list[str]:
        return [key for key, value in self._tables.items() if value is not None]

    def load_all(self) -> None:
        """Read every table now (--check); the app itself stays lazy."""
        for key in TABLES:
            self.table(key)

    def rows(self, key: str, card_id: str) -> pd.DataFrame:
        """Rows of a per-card table for one card (empty frame when absent)."""
        frame = self.table(key)
        if frame is None or "card_id" not in frame.columns:
            return pd.DataFrame()
        if key not in self._groups:
            self._groups[key] = frame.groupby("card_id", sort=False).indices
        idx = self._groups[key].get(card_id)
        return frame.iloc[idx] if idx is not None else frame.iloc[0:0]

    def rep_rows(self, key: str, rep_id: str, column: str = "representation_id") -> pd.DataFrame:
        frame = self.table(key)
        if frame is None or column not in frame.columns:
            return pd.DataFrame()
        return frame.loc[frame[column].astype(str).eq(rep_id)]

    # ---- navigation helpers -----------------------------------------------------------
    @property
    def panels(self) -> list[str]:
        return sorted({panel for panel, _, _ in self.reps}, key=lambda p: (self.panel_label(p).lower(), p))

    def panel_label(self, panel: str) -> str:
        if "panel_label" in self.index.columns:
            labels = self.index.loc[self.index["panel_id"].astype(str).eq(panel), "panel_label"].dropna()
            if not labels.empty:
                return str(labels.iloc[0])
        return panel

    def methods_for(self, panel: str) -> list[str]:
        found = {m for p, m, _ in self.reps if p == panel}
        return [m for m in METHODS if m in found] + sorted(found - set(METHODS))

    def ks_for(self, panel: str, method: str) -> list[int]:
        return sorted(k for p, m, k in self.reps if p == panel and m == method)

    def card(self, card_id: str) -> pd.Series:
        return self.index.loc[card_id]

    @property
    def supported_patients(self) -> list[str]:
        listed = (self.manifest.get("parameters") or {}).get("supported_patients")
        if listed:
            return [str(p) for p in listed]
        lift = self.table("patient_lift")
        return sorted(lift["patient"].astype(str).unique())


# ======================================================================================
# 3. Navigation (URL <-> state), pure
# ======================================================================================


@dataclass
class Nav:
    panel: str | None = None
    method: str | None = None
    k: int | None = None
    factor: int | None = None
    status: str = "home"  # home | ok | not_covered | unknown_panel | invalid
    message: str = ""

    @property
    def rep(self) -> tuple[str, str, int] | None:
        return (self.panel, self.method, self.k) if self.status == "ok" else None

    @property
    def card_id(self) -> str | None:
        if self.status != "ok" or self.factor is None:
            return None
        return card_id_for(self.panel, self.method, self.k, self.factor)

    def query(self) -> str:
        if not self.panel:
            return ""
        params = {"panel": self.panel}
        if self.method:
            params["method"] = self.method
        if self.k is not None:
            params["k"] = str(self.k)
        if self.factor is not None:
            params["factor"] = str(self.factor)
        return "?" + urlencode(params)


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower().lstrip("fk")
    try:
        return int(text)
    except ValueError:
        return None


def parse_search(search: str | None) -> dict:
    """'?panel=..&method=..&k=..&factor=..' (or '&card=<card_id>') -> raw dict of strings."""
    params = {k: v[0] for k, v in parse_qs((search or "").lstrip("?"), keep_blank_values=False).items() if v}
    card = params.get("card")
    if card:
        match = CARD_ID_RE.match(card)
        if match:
            params.setdefault("panel", match.group(1))
            params.setdefault("method", match.group(2))
            params.setdefault("k", match.group(3))
            params.setdefault("factor", match.group(4))
    return params


def navigate(bundle: CardBundle, panel=None, method=None, k=None, factor=None, *, strict: bool = True) -> Nav:
    """Resolve requested (panel, method, K, factor) against the cards.

    strict=True (deep links): a representation without cards is reported as not_covered.
    strict=False (dropdowns): an incompatible method / K is replaced by the panel's first carded one.
    """
    if not panel:
        return Nav(status="home")
    panel = str(panel)
    if panel not in {p for p, _, _ in bundle.reps}:
        return Nav(panel=panel, status="unknown_panel", message=(
            f"Panel '{panel}' has no D18 factor cards. The bundle covers {len(bundle.panels)} panels and "
            f"{len(bundle.reps)} representations."
        ))
    k_int = _int_or_none(k)
    if k is not None and k != "" and k_int is None:
        return Nav(panel=panel, status="invalid", message=f"K must be an integer, got '{k}'.")
    methods = bundle.methods_for(panel)
    if method and method not in methods or (method and k_int is not None and k_int not in bundle.ks_for(panel, method)):
        if strict:
            return Nav(panel=panel, method=method, k=k_int, status="not_covered", message=(
                f"{METHOD_LABEL.get(method, method)} K={k_int if k_int is not None else '?'} of this panel is not in the D18 "
                "shortlist (the cards cover selected representations only). Carded representations of this panel are listed below."
            ))
        method = method if method in methods else methods[0]
        ks = bundle.ks_for(panel, method)
        k_int = k_int if k_int in ks else ks[-1]
    if not method:
        method = methods[0]
    ks = bundle.ks_for(panel, method)
    if k_int is None:
        k_int = ks[-1]
    factor_int = _int_or_none(factor)
    message = ""
    if factor_int is not None and card_id_for(panel, method, k_int, factor_int) not in bundle.index.index:
        message = f"factor {factor_int} is not a card of {METHOD_LABEL.get(method, method)} K={k_int}."
        factor_int = None
    return Nav(panel=panel, method=method, k=k_int, factor=factor_int, status="ok", message=message)


# ======================================================================================
# 4. Views (pure data; the Dash layer only renders them)
# ======================================================================================


def orientation_text(card: pd.Series) -> str:
    status = str(card.get("orientation_status", "")) or ("oriented" if int(card["orientation_sign"]) != 0 else "unoriented")
    sign = int(card["orientation_sign"])
    if sign != 0:
        return f"oriented, beta sign {'+' if sign > 0 else '-'} (malignant pole = {'positive' if sign > 0 else 'negative'} raw scores)"
    return {
        "unoriented_all_zero": "unoriented: beta 0 in every supported fold",
        "unoriented_sign_flip": "unoriented: beta sign flips across supported folds",
    }.get(status, "unoriented")


def representation_header(bundle: CardBundle, panel: str, method: str, k: int) -> dict:
    cards = bundle.index.loc[bundle.reps[(panel, method, k)]]
    first = cards.iloc[0]
    out = {
        "panel": panel, "panel_label": bundle.panel_label(panel), "method": method, "k": k, "n_cards": len(cards),
        "n_selected_drivers": int(cards["is_selected_driver"].map(as_bool).sum()),
        "effective_k": int(first["effective_k"]) if pd.notna(first["effective_k"]) else None,
    }
    for key in ("n_genes", "fit_penalty", "fit_C", "fit_l1_ratio", "fit_dense", "median_fold_nonzero_fraction",
                "shortlist_reasons", "n_all_zero_folds", "all_zero_folds", "representation_id"):
        out[key] = first.get(key) if key in cards.columns else None
    return out


def representation_table(bundle: CardBundle, panel: str, method: str, k: int, *, drivers_only: bool = False,
                         hide_degenerate: bool = False) -> pd.DataFrame:
    """One row per factor card of the representation (the factor list)."""
    cards = bundle.index.loc[bundle.reps[(panel, method, k)]].copy()
    if drivers_only:
        cards = cards.loc[cards["is_selected_driver"].map(as_bool)]
    if hide_degenerate:
        cards = cards.loc[~cards["degenerate_flag"].map(as_bool)]
    n_folds = cards["n_folds"] if "n_folds" in cards.columns else pd.Series(len(bundle.supported_patients), index=cards.index)
    return pd.DataFrame({
        "card_id": cards["card_id"],
        "factor": cards["factor"].astype(int),
        "orientation": [("+" if s > 0 else "-" if s < 0 else "0") for s in cards["orientation_sign"].astype(int)],
        "orientation_status": _col(cards, "orientation_status").fillna(""),
        "selected driver": cards["is_selected_driver"].map(as_bool),
        "active folds": [f"{int(a)}/{int(n)}" for a, n in zip(cards["n_active_folds"], n_folds)],
        "sign consistency": cards["sign_consistency"],
        "max lift pp": 100 * pd.to_numeric(_col(cards, "max_univariate_lift"), errors="coerce"),
        "median lift pp": 100 * pd.to_numeric(_col(cards, "median_univariate_lift"), errors="coerce"),
        "patients >= 5 pp": _col(cards, "n_patients_univariate_lift_ge_0p05"),
        "eta2 normal": cards["eta2_normal"],
        "eta2 malignant": cards["eta2_malignant"],
        "determinacy": cards["determinacy"],
        "degenerate": cards["degenerate_flag"].map(as_bool),
        "caveats": cards["caveat_flags"].map(lambda v: len(split_list(v))),
    }).reset_index(drop=True)


def card_chips(bundle: CardBundle, card_id: str) -> list[tuple[str, str]]:
    card = bundle.card(card_id)
    n_folds = int(card["n_folds"]) if "n_folds" in card.index and pd.notna(card["n_folds"]) else len(bundle.supported_patients)
    chips = [
        ("orientation", orientation_text(card)),
        ("selected driver", "yes" if as_bool(card["is_selected_driver"]) else "no"),
        ("active supported folds", f"{int(card['n_active_folds'])}/{n_folds}"),
        ("sign consistency", fmt_num(card["sign_consistency"], 3)),
        ("eta2 patient | label", f"normal {fmt_num(card['eta2_normal'])} / malignant {fmt_num(card['eta2_malignant'])}"),
        ("determinacy", fmt_num(card["determinacy"]) + (f" ({card['determinacy_method']})" if "determinacy_method" in card.index and pd.notna(card.get("determinacy_method")) else "")),
        ("effective K", f"{card['effective_k']} of {card['k']}"),
    ]
    for key, label, fmt in (
        ("max_univariate_lift", "max univariate lift", lambda v: f"{fmt_pp(v)} pp"),
        ("median_univariate_lift", "median univariate lift", lambda v: f"{fmt_pp(v)} pp"),
        ("max_fitted_ap_drop", "max ablation dAP", lambda v: f"{fmt_pp(v)} pp"),
        ("eigengap_rel", "relative eigengap", lambda v: fmt_num(v)),
        ("score_sd", "score SD", lambda v: fmt_num(v)),
        ("loading_semantics", "loadings", str),
    ):
        if key in card.index and pd.notna(card.get(key)):
            chips.append((label, fmt(card[key])))
    if as_bool(card["degenerate_flag"]):
        chips.append(("degenerate", "yes: zero score variance, no genes"))
    return chips


def caveat_badges(flags) -> list[tuple[str, str]]:
    """(code, explanation) for each caveat flag; unknown codes keep their name."""
    return [(code, CAVEATS.get(code, "(code not in the v1 vocabulary)")) for code in split_list(flags)]


def fold_strip(bundle: CardBundle, card_id: str) -> list[dict]:
    """Every LOPO fold of the card: supported folds first (patient order), then unsupported ones.

    status: 'positive' / 'negative' (beta != 0 with that sign; positive = toward malignant), 'not_selected'
    (the fold's model is active but this factor's beta is 0), 'model_zero' (the whole model is all-zero:
    constant score, lift = 0 by construction in a supported fold) or 'unknown'.
    """
    rows = bundle.rows("fold_coefs", card_id)
    if rows.empty:
        return []
    supported = set(bundle.supported_patients)
    out = []
    for _, row in rows.iterrows():
        patient = str(row["heldout_patient"])
        is_supported = as_bool(row["supported_fold"]) if "supported_fold" in rows.columns else patient in supported
        coef = float(row["coef"]) if pd.notna(row["coef"]) else np.nan
        model_zero = as_bool(row.get("fold_all_zero")) if "fold_all_zero" in rows.columns else False
        if not np.isfinite(coef):
            status = "unknown"
        elif coef > 0:
            status = "positive"
        elif coef < 0:
            status = "negative"
        else:
            status = "model_zero" if model_zero else "not_selected"
        text = {
            "positive": f"beta {coef:+.3g} (score up -> malignant)",
            "negative": f"beta {coef:+.3g} (score down -> malignant)",
            "not_selected": "beta 0: the fold's model uses other factors",
            "model_zero": "all-zero model: constant score, held-out AP = prevalence",
            "unknown": "no coefficient",
        }[status]
        if "fold_model_lift" in rows.columns and pd.notna(row.get("fold_model_lift")) and is_supported:
            text += f"; fold model lift {fmt_pp(row['fold_model_lift'])} pp"
        title = f"{patient} held out ({'supported' if is_supported else 'unsupported: not in the lift summaries'}): {text}"
        if is_supported and status == "model_zero":
            title += f" - {ZERO_FOLD_TEXT}"
        out.append(dict(patient=patient, supported=is_supported, coef=coef, status=status, title=title,
                        badge=ZERO_FOLD_TEXT if is_supported and status == "model_zero" else ""))
    order = {p: i for i, p in enumerate(bundle.supported_patients)}
    out.sort(key=lambda s: (not s["supported"], order.get(s["patient"], 999), s["patient"]))
    return out


def gene_poles(bundle: CardBundle, card_id: str, top: int = 15) -> dict[str, pd.DataFrame]:
    """{'<pole> (ranked by <loading|structure>)': rows} in pole order; empty for degenerate cards."""
    rows = bundle.rows("genes", card_id)
    if rows.empty:
        return {}
    rows = rows.copy()
    if "ranked_by" not in rows.columns:
        rows["ranked_by"] = "loading"
    out: dict[str, pd.DataFrame] = {}
    for ranked_by in ("loading", "structure"):
        part = rows.loc[rows["ranked_by"].astype(str).eq(ranked_by)]
        poles = [p for p in POLE_ORDER if p in set(part["pole"].astype(str))] + sorted(set(part["pole"].astype(str)) - set(POLE_ORDER))
        for pole in poles:
            sub = part.loc[part["pole"].astype(str).eq(pole)].sort_values("rank").head(top)
            cols = ["rank", "gene", "loading", "oriented_loading"] + (["structure_corr"] if sub["structure_corr"].notna().any() else [])
            out[f"{pole} pole (ranked by {ranked_by})"] = sub[cols].reset_index(drop=True)
    return out


def patient_lift_table(bundle: CardBundle, card_id: str) -> pd.DataFrame:
    rows = bundle.rows("patient_lift", card_id)
    if rows.empty:
        return rows
    order = {p: i for i, p in enumerate(bundle.supported_patients)}
    rows = rows.assign(_o=rows["patient"].astype(str).map(lambda p: order.get(p, 999))).sort_values(["_o", "patient"]).drop(columns="_o")
    return rows.reset_index(drop=True)


def patient_lift_figure(bundle: CardBundle, card_id: str) -> go.Figure:
    """Univariate held-out lift per supported patient (training-fold sign), with the optimistic
    sign-agnostic value and the fitted single-factor lift (D8 attribution) as markers."""
    table = patient_lift_table(bundle, card_id)
    fig = go.Figure()
    if table.empty:
        fig.update_layout(template="plotly_white", height=120, title="no per-patient lift (degenerate card)")
        return fig
    labels = [f"{p}<br>π {100 * float(v):.1f}%<br>{int(n)} mal" for p, v, n in zip(table["patient"], table["prevalence"], table["n_malignant"])]
    lift = 100 * pd.to_numeric(table["univariate_lift"], errors="coerce")
    fig.add_trace(go.Bar(
        x=labels, y=lift, name="univariate lift (sign fixed on training folds)", marker_color=LIFT_BLUE,
        text=[fmt_pp(v / 100, 1) for v in lift], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>univariate lift %{y:.2f} pp<extra></extra>",
    ))
    plotted = list(lift)
    if "univariate_lift_sign_agnostic" in table.columns:
        agnostic = 100 * pd.to_numeric(table["univariate_lift_sign_agnostic"], errors="coerce")
        plotted += list(agnostic)
        fig.add_trace(go.Scatter(
            x=labels, y=agnostic, mode="markers", name="sign-agnostic (F6; optimistic: best sign on the held-out patient)",
            marker=dict(symbol="circle-open", size=11, color=AGNOSTIC_GRAY, line=dict(width=2)),
            hovertemplate="%{x}<br>sign-agnostic %{y:.2f} pp<extra></extra>",
        ))
    if "fitted_single_factor_lift" in table.columns and table["fitted_single_factor_lift"].notna().any():
        fitted = 100 * pd.to_numeric(table["fitted_single_factor_lift"], errors="coerce")
        plotted += list(fitted)
        fig.add_trace(go.Scatter(
            x=labels, y=fitted, mode="markers", name="single-factor lift in the fitted model (D8; attribution)",
            marker=dict(symbol="diamond", size=9, color=FITTED_ORANGE),
            hovertemplate="%{x}<br>fitted single-factor lift %{y:.2f} pp<extra></extra>",
        ))
    finite = [v for v in plotted if np.isfinite(v)] + [0.0]
    span = max(max(finite) - min(finite), 2.0)
    fig.add_hline(y=0, line=dict(color="#666", dash="dash", width=1))
    fig.update_layout(
        template="plotly_white", height=340, margin=dict(l=60, r=10, t=40, b=60),
        title=dict(text="<b>Held-out lift of this factor alone, per supported patient</b> (AUPRC pp)", x=0, font=dict(size=12)),
        yaxis=dict(title="lift (AUPRC pp)", range=[min(finite) - 0.12 * span, max(finite) + 0.18 * span]),
        legend=dict(orientation="h", y=-0.28, font=dict(size=10)), showlegend=True,
    )
    return fig


def score_dist_figure(bundle: CardBundle, card_id: str) -> go.Figure:
    """Score quantiles (q10/q25/q50/q75/q90) per patient and label, from the bundle (no cell data)."""
    rows = bundle.rows("score_dist", card_id)
    fig = go.Figure()
    if rows.empty:
        fig.update_layout(template="plotly_white", height=120, title="no score distribution in this bundle")
        return fig
    oriented = bool(rows["oriented"].map(as_bool).all())
    order = {p: i for i, p in enumerate(bundle.supported_patients)}
    rows = rows.assign(_o=rows["patient"].astype(str).map(lambda p: order.get(p, 500))).sort_values(["_o", "patient"])
    for label, colour in (("normal", NORMAL_COLOR), ("malignant", MALIGNANT_COLOR)):
        part = rows.loc[rows["label"].astype(str).eq(label)]
        if part.empty:
            continue
        fig.add_trace(go.Box(
            x=part["patient"].astype(str), q1=part["q25"], median=part["q50"], q3=part["q75"],
            lowerfence=part["q10"], upperfence=part["q90"], name=f"{label} cells", marker_color=colour,
            mean=part["mean"] if "mean" in part.columns else None, boxpoints=False,
            hovertext=[f"{label}, n = {int(n):,}" for n in part["n_cells"]],
        ))
    fig.update_layout(
        template="plotly_white", height=330, boxmode="group", margin=dict(l=60, r=10, t=40, b=40),
        title=dict(text="<b>Score distribution per patient</b> (whiskers q10-q90; box q25-q75)", x=0, font=dict(size=12)),
        yaxis=dict(title="oriented score (z; + = malignant pole)" if oriented else "raw score (z; sign arbitrary)", zeroline=True),
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
    )
    return fig


def adt_tables(bundle: CardBundle, card_id: str, top: int = 8) -> dict[str, pd.DataFrame]:
    rows = bundle.rows("adt", card_id)
    out = {}
    for label in ("normal", "malignant"):
        part = rows.loc[rows["label"].astype(str).eq(label)] if not rows.empty else rows
        if not part.empty:
            cols = [c for c in ("rank", "adt_marker", "spearman_pooled", "spearman_patient_stratified", "n_cells") if c in part.columns]
            out[label] = part.sort_values("rank").head(top)[cols].reset_index(drop=True)
    return out


def celltype_table(bundle: CardBundle, card_id: str, top: int = 5) -> dict[str, pd.DataFrame]:
    """Highest and lowest mean (oriented) score per predicted cell type, within each label."""
    rows = bundle.rows("celltype", card_id)
    out = {}
    for label in ("normal", "malignant"):
        part = rows.loc[rows["label"].astype(str).eq(label)] if not rows.empty else rows
        if part.empty:
            continue
        part = part.sort_values("mean_oriented_z", ascending=False)
        chosen = pd.concat([part.head(top), part.tail(top)]).drop_duplicates("cell_type")
        out[label] = chosen[["cell_type", "n_cells", "mean_oriented_z"]].reset_index(drop=True)
    return out


def match_rows(bundle: CardBundle, card_id: str) -> list[dict]:
    """Matches to other bases; ``linked`` only when the matched card exists in this bundle's index."""
    rows = bundle.rows("matches", card_id)
    if rows.empty:
        matched = split_list(bundle.card(card_id)["matched_cards"])
        return [dict(card=m, linked=m in bundle.index.index, relation="matched_cards", abs_r=np.nan, score_r=np.nan,
                     other=f"{m}") for m in matched]
    out = []
    for _, row in rows.sort_values("abs_r", ascending=False).iterrows():
        target = row.get("matched_card_id")
        target = None if target is None or (isinstance(target, float) and np.isnan(target)) else str(target)
        out.append(dict(
            card=target, linked=bool(target) and target in bundle.index.index, relation=str(row["relation"]),
            other=f"{METHOD_LABEL.get(str(row['other_method']), row['other_method'])} K={int(row['other_k'])}",
            score_r=row.get("score_r"), abs_r=row.get("abs_r"), r2=row.get("r2_many_to_one"),
            angle=row.get("angle_deg"), phi=row.get("tucker_phi_aligned"),
        ))
    return out


def card_href(card_id: str) -> str:
    match = CARD_ID_RE.match(card_id)
    if not match:
        return ""
    panel, method, k, factor = match.groups()
    return "?" + urlencode({"panel": panel, "method": method, "k": int(k), "factor": int(factor)})


def audit_lines(bundle: CardBundle) -> list[str]:
    m = bundle.manifest
    counts = m.get("counts") or {}
    git = m.get("git_state") or {}
    return [
        f"data: {bundle.root} ({bundle.mode} mode; tables read so far: {', '.join(bundle.loaded)})",
        f"bundle {m.get('slug')} schema_version {m.get('schema_version')}; {counts.get('n_cards', len(bundle.index))} cards, "
        f"{counts.get('n_representations', len(bundle.reps))} representations, {len(bundle.panels)} panels",
        f"created {m.get('created_utc', 'n/a')} by {m.get('created_by', 'n/a')} (git {git.get('head', 'n/a')}{'+dirty' if git.get('dirty') else ''})",
        f"supported patients: {', '.join(bundle.supported_patients)}",
        f"c_policy: {(m.get('parameters') or {}).get('c_policy', 'n/a')}; univariate lift sign: "
        f"{(m.get('parameters') or {}).get('univariate_lift_sign', 'n/a')}",
    ]


# ======================================================================================
# 5. --check and --export-bundle
# ======================================================================================

ANCHOR_CARDS = {  # this experiment's IFN-alpha pair (D8/F9); SKIP when absent (other bundles)
    "single_geneset__hallmark_interferon_alpha_response__pca__k40__f008": -1,
    "single_geneset__hallmark_interferon_alpha_response__fa__k40__f007": 1,
}


def run_checks(bundle: CardBundle, before: dict, after: dict) -> list[tuple[str, str, str]]:
    """(name, PASS|FAIL|SKIP|INFO, detail) rows.  Loads every table."""
    rows: list[tuple[str, str, str]] = []
    m = bundle.manifest
    missing = [k for k in MANIFEST_KEYS if k not in m]
    rows.append(("manifest keys and schema_version", "FAIL" if missing or m.get("slug") != SLUG else "PASS",
                 f"schema_version {m.get('schema_version')}, slug {m.get('slug')}" + (f"; missing {missing}" if missing else "")))

    index = bundle.index
    bad_ids = []
    for cid, panel, method, k, factor in index[["card_id", "panel_id", "method", "k", "factor"]].itertuples(index=False):
        match = CARD_ID_RE.match(str(cid))
        if not match or match.groups() != (str(panel), str(method), str(int(k)), f"{int(factor):03d}"):
            bad_ids.append(cid)
    bad_orient = int((~index["orientation_sign"].isin([-1, 0, 1])).sum())
    bad_factor = int(((index["factor"] < 1) | (index["factor"] > index["k"])).sum())
    ok = not bad_ids and not bad_orient and not bad_factor
    rows.append(("index card ids, factors and orientation", "PASS" if ok else "FAIL",
                 f"{len(index)} cards; {len(bad_ids)} bad ids, {bad_factor} factors outside 1..K, {bad_orient} bad orientation signs"))

    counts = m.get("counts") or {}
    mismatches, orphans, dupes = [], {}, {}
    known = set(index.index)
    for key, (filename, _, _) in TABLES.items():
        frame = bundle.table(key)
        if frame is None:
            continue
        if filename in counts and int(counts[filename]) != len(frame):
            mismatches.append(f"{filename}: {len(frame)} rows vs manifest {counts[filename]}")
        if key in CARD_TABLES:
            extra = set(frame["card_id"].astype(str)) - known
            if extra:
                orphans[filename] = len(extra)
    if "n_cards" in counts and int(counts["n_cards"]) != len(index):
        mismatches.append(f"n_cards {len(index)} vs manifest {counts['n_cards']}")
    if "n_representations" in counts and int(counts["n_representations"]) != len(bundle.reps):
        mismatches.append(f"n_representations {len(bundle.reps)} vs manifest {counts['n_representations']}")
    for key, keys in (("patient_lift", ["card_id", "patient"]), ("fold_coefs", ["card_id", "heldout_patient"])):
        frame = bundle.table(key)
        dupes[key] = int(frame.duplicated(keys).sum())
    rows.append(("row counts match the manifest", "FAIL" if mismatches else "PASS",
                 "; ".join(mismatches) if mismatches else f"{len(counts)} manifest counts agree"))
    rows.append(("per-card tables reference indexed cards only", "FAIL" if orphans or any(dupes.values()) else "PASS",
                 f"orphans {orphans or 0}; duplicate card x patient rows {dupes}"))

    lift = bundle.table("patient_lift")
    supported = set(bundle.supported_patients)
    extra_patients = sorted(set(lift["patient"].astype(str)) - supported)
    per_card = lift.groupby("card_id").size()
    rows.append(("patient lift covers supported patients only", "FAIL" if extra_patients or (per_card > len(supported)).any() else "PASS",
                 f"{len(supported)} supported patients; {int((per_card == len(supported)).sum())}/{len(index)} cards with all of them"
                 + (f"; unexpected {extra_patients}" if extra_patients else "")))

    unknown = sorted({c for v in index["caveat_flags"] for c in split_list(v)} - set(CAVEATS))
    rows.append(("caveat flags use the v1 vocabulary", "INFO" if unknown else "PASS",
                 f"unknown codes shown verbatim: {unknown}" if unknown else f"{len(CAVEATS)} codes known"))

    unresolved = [rep for rep in bundle.reps if navigate(bundle, *rep).status != "ok"]
    rows.append(("deep links resolve (?panel=&method=&k=)", "FAIL" if unresolved else "PASS",
                 f"{len(bundle.reps) - len(unresolved)}/{len(bundle.reps)} representations" + (f"; failing {unresolved[:3]}" if unresolved else "")))

    errors = []
    for rep, cards in bundle.reps.items():
        cid = cards[0]
        try:
            representation_table(bundle, *rep)
            card_chips(bundle, cid), fold_strip(bundle, cid), gene_poles(bundle, cid), patient_lift_figure(bundle, cid)
            score_dist_figure(bundle, cid), adt_tables(bundle, cid), celltype_table(bundle, cid), match_rows(bundle, cid)
        except Exception as error:  # noqa: BLE001 - reported as a FAIL row
            errors.append(f"{cid}: {type(error).__name__}: {error}")
    rows.append(("views render (first card of every representation)", "FAIL" if errors else "PASS",
                 f"{len(bundle.reps) - len(errors)}/{len(bundle.reps)} representations" + (f"; {errors[:2]}" if errors else "")))

    present = {cid: sign for cid, sign in ANCHOR_CARDS.items() if cid in index.index}
    if not present:
        rows.append(("IFN-alpha PC8 / FA7 anchor cards", "SKIP", "not in this bundle"))
    else:
        wrong = {cid: int(index.loc[cid, "orientation_sign"]) for cid, sign in present.items() if int(index.loc[cid, "orientation_sign"]) != sign}
        rows.append(("IFN-alpha PC8 / FA7 anchor cards", "FAIL" if wrong else "PASS",
                     f"{len(present)}/{len(ANCHOR_CARDS)} present; orientation signs {'differ: ' + str(wrong) if wrong else 'PC8 -1, FA7 +1'}"))

    checks = bundle.table("regression_checks")
    if checks is None:
        rows.append(("producer regression checks passed", "SKIP", "factor_cards_regression_checks.csv not present"))
    else:
        failed = checks.loc[~checks["passed"].map(as_bool), "check_id"].tolist()
        rows.append(("producer regression checks passed", "FAIL" if failed else "PASS",
                     f"{len(checks) - len(failed)}/{len(checks)} D18 checks passed" + (f"; failed {failed}" if failed else "")))
    rows.append(("inputs unmodified by loading", "PASS" if before == after else "FAIL", f"{len(before)} files fingerprinted (size + mtime)"))
    return rows


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_bundle(data_dir: Path, out_dir: Path, *, limit_bytes: int = EXPORT_LIMIT_BYTES) -> list[Path]:
    """Gzip the bundle's CSV tables plus its manifest (and schema doc) into a NEW directory (<= 20 MB)."""
    data_dir, out_dir = Path(data_dir).resolve(), Path(out_dir).resolve()
    root, mode = resolve_bundle_root(data_dir)
    for protected in {data_dir, root.resolve()}:
        if out_dir == protected or protected in out_dir.parents:
            raise DataError(f"refusing to write inside the data directory: {out_dir}")
    payload: dict[str, bytes] = {}
    sources: dict[str, dict] = {}
    for key, (filename, _, required) in TABLES.items():
        path = table_path(root, filename)
        if path is None:
            if required:
                raise DataError(f"cannot export: required table {filename} not found")
            continue
        raw = path.read_bytes()
        csv_bytes = gzip.decompress(raw) if path.suffix == ".gz" else raw
        payload[f"{filename}.gz"] = gzip.compress(csv_bytes, compresslevel=9, mtime=0)
        sources[f"{filename}.gz"] = {"table": key, "csv_sha256": _sha256_bytes(csv_bytes), "csv_bytes": len(csv_bytes)}
    payload[MANIFEST] = (root / MANIFEST).read_bytes()
    if (root / SCHEMA_DOC).is_file():
        payload[SCHEMA_DOC] = (root / SCHEMA_DOC).read_bytes()
    total = sum(len(v) for v in payload.values())
    if total > limit_bytes:
        raise DataError(f"export would be {total / 1e6:.1f} MB, above the {limit_bytes / 1e6:.0f} MB limit")
    clash = [str(out_dir / name) for name in [*payload, EXPORT_MANIFEST] if (out_dir / name).exists()]
    if clash:
        raise DataError("refusing to overwrite existing files:\n  " + "\n  ".join(clash))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in payload.items():
        (out_dir / name).write_bytes(data)
        written.append(out_dir / name)
    export = {
        "slug": SLUG, "schema_version": SCHEMA_VERSION, "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"{BUNDLE_SUBDIR} of experiment {data_dir.name}" if mode == "experiment" else "an exported factor_cards_v1 bundle",
        "run": "python old34_factor_explorer.py --data-dir <this directory>",
        "total_bytes": total, "files": {name: {"bytes": len(data), **sources.get(name, {})} for name, data in payload.items()},
    }
    (out_dir / EXPORT_MANIFEST).write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    written.append(out_dir / EXPORT_MANIFEST)
    return written


# ======================================================================================
# 6. Dash app
# ======================================================================================

CSS = """
body { margin: 0; font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; color: #222; background: #fff; }
.banner { position: sticky; top: 0; z-index: 10; background: #fff4e0; border-bottom: 1px solid #f0d9b0; color: #6b3a00;
          font-size: 12px; line-height: 1.4; padding: 6px 16px; }
.banner b { color: #5a2f00; }
.page { display: flex; align-items: flex-start; }
.sidebar { width: 300px; flex: none; position: sticky; top: 48px; max-height: calc(100vh - 48px); overflow-y: auto; box-sizing: border-box;
           padding: 12px 16px 24px; background: #f7f8fa; border-right: 1px solid #dde1e6; font-size: 13px; }
.sidebar h1 { font-size: 16px; margin: 0 0 2px; }
.sidebar h2 { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: #667; margin: 16px 0 6px; }
.lbl { display: block; font-weight: 600; margin: 8px 0 3px; }
.sub, .help { color: #667; font-size: 11.5px; line-height: 1.35; }
.main { flex: 1; min-width: 0; padding: 10px 18px 30px; }
.card { border: 1px solid #e1e4e8; border-radius: 6px; padding: 8px 12px; background: #fff; margin-bottom: 12px; }
.chip { display: inline-block; background: #eef2f7; border-radius: 10px; padding: 2px 9px; margin: 2px 4px 2px 0; font-size: 12px; }
.chip b { color: #345; }
.flag { display: inline-block; background: #fdf0d5; border: 1px solid #ecd29b; color: #6b4a00; border-radius: 9px; padding: 1px 8px;
        margin: 2px 4px 2px 0; font-size: 11px; cursor: help; }
.badge { display: inline-block; background: #c62828; color: #fff; border-radius: 9px; padding: 1px 8px; font-size: 11px; font-weight: 600; margin: 2px 6px 2px 0; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; margin: 4px 0; }
.strip-sep { width: 10px; }
.fold { width: 44px; border-radius: 3px; text-align: center; font-size: 10.5px; line-height: 1.25; padding: 2px 0; border: 1px solid #ccc; cursor: default; }
.fold b { display: block; font-size: 11px; }
.fold.positive { background: #fbe1dc; border-color: #c0392b; color: #7b1d12; }
.fold.negative { background: #dce9fb; border-color: #2c5fa8; color: #173a6b; }
.fold.not_selected { background: #f4f4f4; border-color: #bbb; color: #666; }
.fold.model_zero { background: #fff; border: 2px solid #c62828; color: #8e1b1b; }
.fold.unsupported { opacity: .45; border-style: dashed; filter: grayscale(100%); }
table.t { border-collapse: collapse; font-size: 12px; margin: 6px 0; }
table.t th { text-align: right; font-weight: 600; border-bottom: 2px solid #ccd; padding: 3px 6px; background: #f3f5f8; }
table.t td { text-align: right; padding: 3px 6px; border-bottom: 1px solid #eee; }
table.t th:first-child, table.t td:first-child, table.t td.l, table.t th.l { text-align: left; }
tr.sel td { background: #e0f7fa; }
.poles { display: flex; gap: 18px; flex-wrap: wrap; align-items: flex-start; }
.msg { background: #f3f5f8; border-left: 3px solid #8899aa; padding: 6px 10px; margin: 6px 0; font-size: 13px; }
details pre { white-space: pre-wrap; font-size: 11px; color: #445; }
.placeholder { color: #667; padding: 30px 18px; line-height: 1.6; }
"""


def _cell(value, kind: str = "") -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "-"
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if kind == "pp":
        return f"{float(value):+.2f}" if float(value) else "0.00"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    return str(value)


def create_app(bundle: CardBundle):
    """Build the Dash app.  Dash is imported here so --check and the tests do not need it."""
    from dash import Dash, Input, Output, ctx, dcc, html

    app = Dash(__name__, title="Old34 factor explorer")
    app.index_string = (
        "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
        f"<style>{CSS}</style></head><body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"
    )
    graph_config = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}

    def table(frame: pd.DataFrame, kinds: dict | None = None, left: tuple = (), selected=None, link_col: str | None = None):
        kinds = kinds or {}
        head = html.Tr([html.Th(c, className="l" if c in left else "") for c in frame.columns if c != "card_id"])
        body = []
        for _, row in frame.iterrows():
            cells = []
            for col in frame.columns:
                if col == "card_id":
                    continue
                text = _cell(row[col], kinds.get(col, ""))
                if link_col == col and "card_id" in frame.columns:
                    cells.append(html.Td(dcc.Link(text, href=card_href(row["card_id"])), className="l"))
                else:
                    cells.append(html.Td(text, className="l" if col in left else ""))
            body.append(html.Tr(cells, className="sel" if selected is not None and row.get("card_id") == selected else ""))
        return html.Table([html.Thead(head), html.Tbody(body)], className="t")

    def rep_links(panel: str) -> list:
        items = []
        for p, m, k in sorted(bundle.reps, key=lambda r: (METHODS.index(r[1]) if r[1] in METHODS else 9, r[2])):
            if p == panel:
                items.append(html.Li(dcc.Link(f"{METHOD_LABEL.get(m, m)} K={k} ({len(bundle.reps[(p, m, k)])} cards)",
                                              href="?" + urlencode({"panel": p, "method": m, "k": k}))))
        return [html.Ul(items)] if items else []

    def strip_view(card_id: str):
        strip = fold_strip(bundle, card_id)
        if not strip:
            return html.Div("No per-fold coefficients for this card.", className="help")
        boxes = []
        for k, item in enumerate(strip):
            if k and item["supported"] != strip[k - 1]["supported"]:
                boxes.append(html.Div(className="strip-sep"))
            value = {"positive": "+", "negative": "-", "not_selected": "0", "model_zero": "Ø", "unknown": "?"}[item["status"]]
            classes = f"fold {item['status']}" + ("" if item["supported"] else " unsupported")
            boxes.append(html.Div([html.B(item["patient"]), value], className=classes, title=item["title"]))
        zero = [s["patient"] for s in strip if s["badge"]]
        return html.Div([
            html.Div(boxes, className="strip"),
            html.Div(([html.Span(f"{ZERO_FOLD_TEXT}: {', '.join(zero)}", className="badge")] if zero else []) + [
                "Left: supported folds; right (greyed): unsupported folds, never summarised. + / - = sign of beta "
                "(positive = toward malignant); 0 = factor not selected in that fold; Ø = the fold's whole model is all-zero."
            ], className="help"),
        ])

    def card_view(card_id: str) -> list:
        card = bundle.card(card_id)
        children = [
            html.H3(f"Factor {int(card['factor'])} of {METHOD_LABEL.get(card['method'], card['method'])} K={int(card['k'])}  "
                    f"({bundle.panel_label(card['panel_id'])})", style={"margin": "4px 0"}),
            html.Div(f"card {card_id}", className="sub"),
            html.Div([html.Span([html.B(f"{k}: "), v], className="chip") for k, v in card_chips(bundle, card_id)]),
            html.Div([html.Span(code, className="flag", title=text) for code, text in caveat_badges(card["caveat_flags"])]),
            html.Details([html.Summary("What the caveat flags mean"),
                          html.Ul([html.Li([html.B(code + ": "), text]) for code, text in caveat_badges(card["caveat_flags"])])]),
            html.H4("Per-fold coefficient (13 LOPO folds at the selected regularization)", style={"marginBottom": "2px"}),
            strip_view(card_id),
            dcc.Graph(figure=patient_lift_figure(bundle, card_id), config=graph_config),
        ]
        lift = patient_lift_table(bundle, card_id)
        if not lift.empty:
            cols = [c for c in ("patient", "n_malignant", "prevalence", "univariate_lift", "univariate_auroc", "univariate_lift_sign_agnostic",
                                "fitted_coef", "fitted_ap_drop", "fitted_single_factor_lift") if c in lift.columns]
            shown = lift[cols].copy()
            if "prevalence" in shown:
                shown["prevalence"] = 100 * shown["prevalence"]
            for col in ("univariate_lift", "univariate_lift_sign_agnostic", "fitted_ap_drop", "fitted_single_factor_lift"):
                if col in shown:
                    shown[col] = 100 * pd.to_numeric(shown[col], errors="coerce")
            shown = shown.rename(columns={"prevalence": "prevalence %", "univariate_lift": "lift pp", "univariate_auroc": "AUROC",
                                          "univariate_lift_sign_agnostic": "sign-agnostic pp", "fitted_ap_drop": "ablation dAP pp",
                                          "fitted_single_factor_lift": "fitted single-factor pp"})
            children.append(table(shown, {c: "pp" for c in shown.columns if c.endswith(" pp")}, left=("patient",)))
        poles = gene_poles(bundle, card_id)
        children.append(html.H4("Top genes on both poles", style={"marginBottom": "2px"}))
        if poles:
            children.append(html.Div("Name a factor from both poles. FA / FactoSig loadings are generative; promax loadings are a "
                                     "pattern matrix (read with the structure correlations).", className="help"))
            children.append(html.Div([html.Div([html.B(name), table(frame, left=("gene",))]) for name, frame in poles.items()], className="poles"))
        else:
            children.append(html.Div("No gene rows (degenerate factor).", className="help"))
        children.append(dcc.Graph(figure=score_dist_figure(bundle, card_id), config=graph_config))
        adt = adt_tables(bundle, card_id)
        if adt:
            floor = ", ".join(f"{label} {fmt_num(card.get(f'adt_isotype_max_abs_rho_{label}'))}" for label in ("normal", "malignant")
                              if f"adt_isotype_max_abs_rho_{label}" in card.index)
            children += [html.H4("Surface protein (ADT, CLR) Spearman within label", style={"marginBottom": "2px"}),
                         html.Div(f"Isotype-control max |rho| (noise floor): {floor or 'n/a'}. Correlation, not mechanism.", className="help"),
                         html.Div([html.Div([html.B(label), table(frame, left=("adt_marker",))]) for label, frame in adt.items()], className="poles")]
        cts = celltype_table(bundle, card_id)
        if cts:
            children += [html.H4("Mean score by predicted cell type (highest and lowest)", style={"marginBottom": "2px"}),
                         html.Div([html.Div([html.B(label), table(frame, left=("cell_type",))]) for label, frame in cts.items()], className="poles")]
        matches = match_rows(bundle, card_id)
        if matches:
            items = []
            for mrow in matches:
                target = mrow["card"]
                link = dcc.Link(target, href=card_href(target)) if mrow["linked"] else html.Span(target or "-")
                items.append(html.Tr([html.Td(mrow["relation"], className="l"), html.Td(mrow["other"], className="l"), html.Td(link, className="l"),
                                      html.Td(fmt_num(mrow.get("score_r"))), html.Td(fmt_num(mrow.get("r2"))), html.Td(fmt_num(mrow.get("phi")))]))
            children += [html.H4("Matches to other bases", style={"marginBottom": "2px"}),
                         html.Div("Linked only when the matched card is in this bundle; other ids are plain text.", className="help"),
                         html.Table([html.Thead(html.Tr([html.Th(h, className="l") for h in ("relation", "basis", "matched card")] +
                                                        [html.Th(h) for h in ("score r", "R2 many-to-one", "Tucker phi")])),
                                     html.Tbody(items)], className="t")]
        return children

    def content(nav: Nav, drivers_only: bool, hide_degenerate: bool) -> list:
        out = []
        if nav.message:
            out.append(html.Div(nav.message, className="msg"))
        if nav.status == "home":
            return out + [html.Div([
                html.Div("Pick a panel, method and K on the left, or open a deep link from the D1 atlas cell view."),
                html.Div(f"{len(bundle.index)} factor cards over {len(bundle.reps)} representations of {len(bundle.panels)} panels."),
            ], className="placeholder")]
        if nav.status in ("unknown_panel", "invalid"):
            return out
        if nav.status == "not_covered":
            return out + rep_links(nav.panel)
        panel, method, k = nav.rep
        head = representation_header(bundle, panel, method, k)
        reg = f"{head.get('fit_penalty')} C={fmt_num(head.get('fit_C'))}" if head.get("fit_penalty") is not None else "n/a"
        out.append(html.Div([
            html.H2(f"{head['panel_label']} | {METHOD_LABEL.get(method, method)} K={k}", style={"margin": "2px 0", "fontSize": "17px"}),
            html.Div(f"{panel}; {head.get('n_genes')} genes; selected probe {reg} (c_policy global_selected; stored, no refit); "
                     f"dense fit: {'yes' if as_bool(head.get('fit_dense')) else 'no'}; {head['n_selected_drivers']} selected drivers of "
                     f"{head['n_cards']} cards; effective K {head.get('effective_k')}; shortlist: {head.get('shortlist_reasons')}", className="sub"),
            html.Div(dcc.Link("permalink to this view", href=nav.query()), className="help"),
        ], className="card"))
        if nav.card_id:
            out.append(html.Div(card_view(nav.card_id), className="card"))
        frame = representation_table(bundle, panel, method, k, drivers_only=drivers_only, hide_degenerate=hide_degenerate)
        kinds = {"max lift pp": "pp", "median lift pp": "pp"}
        out.append(html.Div([
            html.H4(f"Factors ({len(frame)} shown)", style={"margin": "2px 0"}),
            html.Div("Click a factor. Lift = this factor alone on the held-out patient, sign fixed on the training folds (pp).", className="help"),
            table(frame.drop(columns=["orientation_status"]), kinds, left=("factor",), selected=nav.card_id, link_col="factor"),
        ], className="card"))
        return out

    panel_options = [{"label": f"{bundle.panel_label(p)}", "value": p} for p in bundle.panels]
    sidebar = html.Div(className="sidebar", children=[
        html.H1("Old34 factor explorer"),
        html.Div("D18 factor cards (factor_cards_v1). Read-only; nothing is refit.", className="sub"),
        html.H2("Representation"),
        html.Label("Panel", className="lbl"), dcc.Dropdown(id="panel", options=panel_options, value=None, clearable=True),
        html.Label("Method", className="lbl"), dcc.Dropdown(id="method", options=[], value=None, clearable=False),
        html.Label("K", className="lbl"), dcc.Dropdown(id="k", options=[], value=None, clearable=False),
        html.Label("Factor", className="lbl"), dcc.Dropdown(id="factor", options=[], value=None, clearable=True),
        html.H2("Factor list"),
        dcc.Checklist(id="drivers-only", options=[{"label": " selected drivers only", "value": "on"}], value=[]),
        dcc.Checklist(id="hide-degenerate", options=[{"label": " hide degenerate factors", "value": "on"}], value=[]),
        html.H2("Data"),
        html.Details([html.Summary("Data audit"), html.Pre("\n".join(audit_lines(bundle)), id="audit")]),
    ])
    app.layout = html.Div([
        dcc.Location(id="url", refresh="callback-nav"),
        html.Div([html.B("Caveats (always apply): "), BANNER], className="banner", id="banner"),
        html.Div(className="page", children=[sidebar, html.Div(id="content", className="main")]),
    ])

    @app.callback(
        Output("panel", "value"), Output("method", "options"), Output("method", "value"), Output("k", "options"), Output("k", "value"),
        Output("factor", "options"), Output("factor", "value"), Output("content", "children"), Output("url", "search"),
        Input("url", "search"), Input("panel", "value"), Input("method", "value"), Input("k", "value"), Input("factor", "value"),
        Input("drivers-only", "value"), Input("hide-degenerate", "value"),
    )
    def route(search, panel, method, k, factor, drivers_only, hide_degenerate):
        trigger = ctx.triggered_id
        if trigger in (None, "url"):
            raw = parse_search(search)
            nav = navigate(bundle, raw.get("panel"), raw.get("method"), raw.get("k"), raw.get("factor"), strict=True)
        else:
            if trigger in ("panel", "method", "k"):
                factor = None
            nav = navigate(bundle, panel, method, k, factor, strict=False)
        methods = [{"label": METHOD_LABEL.get(m, m), "value": m} for m in (bundle.methods_for(nav.panel) if nav.status in ("ok", "not_covered") else [])]
        ks = [{"label": str(x), "value": x} for x in (bundle.ks_for(nav.panel, nav.method) if nav.status == "ok" else [])]
        factors = []
        if nav.status == "ok":
            for cid in bundle.reps[nav.rep]:
                card = bundle.card(cid)
                tags = ["driver" if as_bool(card["is_selected_driver"]) else "", "degenerate" if as_bool(card["degenerate_flag"]) else ""]
                sign = int(card["orientation_sign"])
                factors.append({"label": f"f{int(card['factor']):03d} ({'+' if sign > 0 else '-' if sign < 0 else 'unoriented'}"
                                         f"{', ' + ', '.join(t for t in tags if t) if any(tags) else ''})", "value": int(card["factor"])})
        body = content(nav, bool(drivers_only), bool(hide_degenerate))
        panel_value = nav.panel if nav.status in ("ok", "not_covered") else None
        url_search = nav.query() if nav.status == "ok" else ("" if nav.status == "home" else (search or ""))
        return (panel_value, methods, nav.method if nav.status == "ok" else None, ks, nav.k if nav.status == "ok" else None,
                factors, nav.factor, body, url_search)

    return app


# ======================================================================================
# 7. CLI
# ======================================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local, read-only explorer of the D18 factor_cards_v1 bundle.")
    parser.add_argument("--data-dir", type=Path, help="experiment directory or an exported bundle (default: SC_* experiment)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use an SSH tunnel to share)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true", help="Dash hot reload for quick design iteration")
    parser.add_argument("--check", action="store_true", help="load and validate the bundle, print a report, do not serve")
    parser.add_argument("--export-bundle", type=Path, metavar="DIR", help="write a gzip copy (<= 20 MB) to a NEW directory and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    if data_dir is None:
        print("Could not resolve the experiment dir (set SC_EXPERIMENT_ROOT / SC_EXPERIMENT_ID) ; pass --data-dir.", file=sys.stderr)
        return 2
    try:
        if args.export_bundle:
            written = export_bundle(data_dir, args.export_bundle)
            total = sum(p.stat().st_size for p in written)
            print("Wrote bundle:")
            for path in written:
                print(f"  {path}  ({path.stat().st_size / 1e6:.2f} MB)")
            print(f"total {total / 1e6:.2f} MB\nShare that directory, then run:  python {Path(__file__).name} --data-dir {args.export_bundle.name}")
            return 0
        root, _ = resolve_bundle_root(Path(data_dir))
        before = bundle_fingerprint(root)
        started = time.perf_counter()
        bundle = CardBundle(Path(data_dir))
        print(f"[factor-explorer] loaded index: {len(bundle.index)} cards, {len(bundle.reps)} representations "
              f"({bundle.mode} mode, lazy tables) in {time.perf_counter() - started:.1f}s")
        if args.check:
            bundle.load_all()
            after = bundle_fingerprint(root)
            rows = run_checks(bundle, before, after)
            failed = 0
            for name, status, detail in rows:
                failed += status == "FAIL"
                print(f"  [{status:4s}] {name}: {detail}")
            print("\n".join(["", *audit_lines(bundle)]))
            return 1 if failed else 0
    except DataError as error:
        print(f"Data error: {error}", file=sys.stderr)
        return 2
    app = create_app(bundle)
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    print(f"[factor-explorer] serving http://{args.host if not loopback else '127.0.0.1'}:{args.port}  (read-only; Ctrl-C to stop)")
    if loopback:
        print(f"[factor-explorer] teammates: ssh -L {args.port}:127.0.0.1:{args.port} <user>@<this-host>, then open http://127.0.0.1:{args.port}")
    else:
        print("[factor-explorer] WARNING: bound beyond localhost; the Dash dev server has no authentication.", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
