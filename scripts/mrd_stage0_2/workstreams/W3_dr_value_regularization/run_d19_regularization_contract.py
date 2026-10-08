"""D19 regularization contract runner (W3.1 entry-margin audit; W3.2 IFN-alpha C-matched comparison).

Parts (--parts, default anchor,audit,ifna,report):
  anchor     Reproduce the scratch L1 entry check (anchors/l1_entry_check_scratch_20261008.py):
             IFN-alpha PCA40 / FA40 x 7 supported LOPO folds at the stored L1 C, predicted
             all-zero vs nonzero from the entry rule against the stored layer4_5 coefficients.
  audit      L1 entry-margin audit over every L1- or elastic-net-selected D1 atlas DR cell x
             supported fold, on the stored transductive Stage 1 scores at the stored selected C.
             Selection = the D1 rule (one global regularizer per representation: highest
             pooled-cell LOPO AUPRC, ties AUROC, then elasticnet > l1 > l2, then smaller C)
             applied to the four LOPO scorecards, cross-checked against the layer4_5 cache.
             Stored zero/nonzero: layer4_5 selected_lopo_coefficients.pkl.gz (biological) and
             old34_controls_fullgrid_v1 by_heldout_patient.csv (controls).
  ifna       W3.2: IFN-alpha PCA40 vs FA40 per supported outer fold with C tuned by inner LOPO
             (L1, L2, EN l1_ratio 0.5), an L1 sweep at 0.9/1.0/1.05/1.1/1.25 x the stored C,
             and L2 at the stored C.
  report     Tables, figures, findings, manifest and artifact ledger.
  cuml_check (not in the default set) seconds-long synthetic check that cuML LogisticRegression
             gives the same zero/nonzero outcome as sklearn around the entry threshold.

Outputs (new slug, R3): EXP/analysis/{scorecards,figures,cache}/stage2_diagnostic/regularization_contract_v1/.
--resume skips finished (representation, fold) audit units and finished ifna units; empty or
corrupt unit files count as unfinished; a tuned unit whose stored c_grid_n differs from the current
C grid is rerun. --out-root redirects every output; --smoke and --max-cells require it.

Example (from the repo root):
  python scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py --resume --n-jobs 12
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker process; joblib supplies the parallelism (<= 12 workers).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import (  # noqa: E402
    add_study_module_paths,
    cell_metadata,
    experiment_dir,
    git_state,
    study_config,
)

add_study_module_paths(subdirs=(), workstreams=("W3_dr_value_regularization",))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import regularization_contract as rc  # noqa: E402

SLUG = "regularization_contract_v1"
IFNA_PANEL = "single_geneset__hallmark_interferon_alpha_response"
IFNA_METHODS = ("pca", "fa")
IFNA_K = 40
SWEEP_MULTIPLIERS = (0.9, 1.0, 1.05, 1.1, 1.25)
TUNE_PENALTIES: Tuple[Tuple[str, Optional[float]], ...] = (("l1", None), ("l2", None), ("elasticnet", 0.5))
KNIFE_EDGE = 0.10
CROSS_FACTOR = 1.01  # "just across" the entry threshold for the lift-at-stake estimate
AUDIT_REQUIRED_COLUMNS = {"argmax_single_feature_lift", "argmax_gap_sign", "fold_agree", "knife_edge_entry"}
KKT_VIOLATION = 1.02  # a zero feature with KKT margin above this means the stored fit stopped early
MAX_WORKERS = 12
DEFAULT_PARTS = ("anchor", "audit", "ifna", "report")
ALL_PARTS = DEFAULT_PARTS + ("cuml_check",)

# (tag, path under EXP/analysis/scorecards, representation_family filter) = the atlas app's SCORECARDS
SCORECARDS = (
    ("june5_biological_dr", "stage2_sharedness_lopo_scorecard.csv", "dr"),
    ("direct_gene_all_v1", "old34_direct_gene_all_v1/stage2_sharedness_lopo_scorecard.csv", None),
    ("controls_fullgrid_v1", "old34_controls_fullgrid_v1/stage2_sharedness_lopo_scorecard.csv", None),
    ("controls_direct_gene_v1", "old34_controls_direct_gene_v1/stage2_sharedness_lopo_scorecard.csv", None),
)
SCORECARD_COLUMNS = (
    "stage0_panel_id", "representation_family", "stage1_method", "effective_k", "seed", "penalty", "C",
    "l1_ratio", "cell_weighted_auprc", "cell_weighted_auroc", "status",
)
PENALTY_RANK = {"elasticnet": 0, "l1": 1, "l2": 2}
ATLAS_TABLE = "analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv"
ATLAS_PATIENT_TABLE = "analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv"
COEF_CACHE = "analysis/cache/stage2_diagnostic/layer4_5/selected_lopo_coefficients.pkl.gz"
CONTROL_BY_FOLD = "stage2_supervised/multiobjective/old34_controls_fullgrid_v1/sharedness_lopo/by_heldout_patient.csv"


# --------------------------------------------------------------------------------------
# context and io helpers
# --------------------------------------------------------------------------------------


class Ctx:
    """Resolved locations, cohort labels and run options."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if (args.smoke or args.max_cells) and not args.out_root:
            # smoke units share names with the full run (thinned C grid) and the report would
            # rebuild the canonical tables, findings and manifest from the smoke subset
            raise SystemExit("--smoke and --max-cells need --out-root (smoke outputs never go into the slug dirs)")
        self.exp = experiment_dir()
        root = Path(args.out_root).resolve() if args.out_root else self.exp / "analysis"
        self.out_root = root
        self.tables = root / "scorecards" / "stage2_diagnostic" / SLUG
        self.figures = root / "figures" / "stage2_diagnostic" / SLUG
        self.cache = root / "cache" / "stage2_diagnostic" / SLUG
        for path in (self.tables, self.figures, self.cache):
            path.mkdir(parents=True, exist_ok=True)
        meta_path = cell_metadata()
        meta = pd.read_csv(meta_path, usecols=["cell_id", "y_true", "patient"])
        self.meta_path = meta_path
        self.y = meta["y_true"].to_numpy().astype(int)
        self.patients = meta["patient"].astype(str).to_numpy()
        self.supported = rc.supported_patients(self.y, self.patients, min_malignant=rc.DEFAULT_MIN_MALIGNANT)
        try:
            configured = list(study_config().get("supported_patients", []))
        except Exception:  # study_config is optional
            configured = []
        if configured and configured != self.supported:
            raise AssertionError(f"support rule gives {self.supported}, study_config lists {configured}")
        self.n_jobs = max(1, min(int(args.n_jobs), MAX_WORKERS))
        self.timings: Dict[str, float] = {}

    def rel(self, path: Path) -> str:
        """EXP-relative path when inside EXP, else the path relative to out_root, tagged."""
        path = Path(path).resolve()
        try:
            return path.relative_to(self.exp.resolve()).as_posix()
        except ValueError:
            return "<out-root>/" + path.relative_to(self.out_root).as_posix()


def score_path(exp: Path, panel: str, method: str, k: int) -> Path:
    return exp / "stage1_dr" / f"panel_id={panel}" / f"method={method}" / f"k={int(k)}" / "seed=42" / "scores.npy"


def load_scores(exp: Path, panel: str, method: str, k: int, n_cells: int) -> np.ndarray:
    scores = np.load(score_path(exp, panel, method, k)).astype(np.float32)
    if scores.shape != (n_cells, int(k)):
        raise AssertionError(f"{panel} {method} K={k}: scores shape {scores.shape}, expected ({n_cells}, {k})")
    return scores


def representation_id(panel: str, method: str, k: int) -> str:
    return f"{panel}|dr|{method}|{int(k)}|42"


def safe_name(text: str) -> str:
    return text.replace("|", "__").replace("/", "_")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    """A unit file that is missing, empty or unparsable counts as unfinished."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        frame = pd.read_csv(path)
        return frame if len(frame) else None
    except Exception:
        return None


def read_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) and obj.get("status") == "done" else None
    except Exception:
        return None


def read_json_any(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def truthy(series: pd.Series) -> pd.Series:
    """Boolean view of a column that may hold True/False/None or their CSV strings."""
    return series.map(lambda v: v is True or v is np.True_ or str(v).strip().lower() in ("true", "1", "1.0"))


def p20(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, 0.2)) if arr.size else float("nan")


def median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def feature_index(feature_id: str) -> int:
    return int(str(feature_id).rsplit("_", 1)[-1]) - 1


# --------------------------------------------------------------------------------------
# selection (D1 rule) and stored coefficients
# --------------------------------------------------------------------------------------


def select_global_regularization(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (panel, configuration, seed): the D1 atlas rule (apps/old34_d1_atlas_explorer.py)."""
    work = frame.copy()
    work["_rank"] = work["penalty"].astype(str).str.lower().map(PENALTY_RANK).fillna(len(PENALTY_RANK))
    keys = ["stage0_panel_id", "configuration", "seed"]
    work = work.sort_values(
        keys + ["cell_weighted_auprc", "cell_weighted_auroc", "_rank", "C"],
        ascending=[True, True, True, False, False, True, True],
        kind="mergesort",
    )
    return work.drop_duplicates(keys, keep="first").drop(columns="_rank")


def atlas_selection(ctx: Ctx) -> pd.DataFrame:
    """783 atlas cells with the D1-selected penalty / C / l1_ratio and the grid-bound flag."""
    parts = []
    for tag, rel, family in SCORECARDS:
        path = ctx.exp / "analysis" / "scorecards" / rel
        frame = pd.read_csv(path, usecols=lambda c: c in SCORECARD_COLUMNS, low_memory=False)
        if "status" in frame.columns:
            frame = frame.loc[frame["status"].eq("ok")]
        if family is not None:
            frame = frame.loc[frame["representation_family"].eq(family)]
        k = pd.to_numeric(frame["effective_k"], errors="coerce").round().astype("Int64").astype(str)
        frame = frame.assign(
            configuration=np.where(frame["stage1_method"].eq("direct_gene"), "direct_gene", frame["stage1_method"] + "|K=" + k)
        )
        parts.append(select_global_regularization(frame).assign(source_scorecard=tag))
    chosen = pd.concat(parts, ignore_index=True)
    if chosen.duplicated(["stage0_panel_id", "configuration"]).any():
        raise AssertionError("more than one selected row per panel x configuration")
    atlas = pd.read_csv(ctx.exp / ATLAS_TABLE)
    out = atlas.merge(
        chosen[["stage0_panel_id", "configuration", "penalty", "C", "l1_ratio", "cell_weighted_auprc", "source_scorecard"]],
        on=["stage0_panel_id", "configuration"],
        how="left",
        validate="one_to_one",
    )
    if out["penalty"].isna().any():
        missing = out.loc[out["penalty"].isna(), "representation_id"].tolist()[:5]
        raise AssertionError(f"atlas cells without a selected regularization, e.g. {missing}")
    out["C_at_grid_lower_bound"] = [rc.c_at_grid_lower_bound(c) for c in out["C"].astype(float)]
    out["C_at_grid_upper_bound"] = np.isclose(out["C"].astype(float), rc.C_GRID_UPPER_BOUND, rtol=1e-9, atol=0)
    out["is_dr"] = out["configuration"].ne("direct_gene")
    out["l1_or_en_selected"] = out["penalty"].isin(["l1", "elasticnet"])
    return out


def load_coef_cache(ctx: Ctx) -> pd.DataFrame:
    cache = pd.read_pickle(ctx.exp / COEF_CACHE)
    cache = cache.loc[cache["selection_roles"].astype(str).str.contains("global_selected")].copy()
    return cache


def stored_bio_coefficients(cache: pd.DataFrame, rep_id: str, k: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """{heldout_patient: coef vector (K,)} and the cached setting for one representation."""
    sub = cache.loc[cache["representation_id"].eq(rep_id)]
    if sub.empty:
        return {}, {}
    setting = {
        "penalty": str(sub["penalty"].iloc[0]),
        "C": float(sub["C"].iloc[0]),
        "l1_ratio": float(sub["l1_ratio"].iloc[0]) if pd.notna(sub["l1_ratio"].iloc[0]) else float("nan"),
        "n_settings": int(sub[["penalty", "C", "l1_ratio"]].drop_duplicates().shape[0]),
    }
    coefs: Dict[str, np.ndarray] = {}
    for patient, grp in sub.groupby("heldout_patient"):
        vec = np.zeros(int(k), dtype=np.float64)
        idx = grp["feature_id"].map(feature_index).to_numpy()
        vec[idx] = grp["coefficient"].to_numpy(dtype=np.float64)
        coefs[str(patient)] = vec
    return coefs, setting


def stored_control_supports(ctx: Ctx, selection: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, List[int]]]:
    """{(panel, configuration): {heldout_patient: [nonzero feature indices]}} at the selected setting."""
    cols = [
        "stage0_panel_id", "representation_family", "stage1_method", "effective_k", "heldout_patient",
        "penalty", "C", "l1_ratio", "threshold", "nonzero_coefficient_count", "selected_feature_ids",
    ]
    frame = pd.read_csv(ctx.exp / CONTROL_BY_FOLD, usecols=cols, low_memory=False)
    frame = frame.loc[frame["representation_family"].eq("dr")].copy()
    frame["configuration"] = frame["stage1_method"] + "|K=" + frame["effective_k"].round().astype(int).astype(str)
    frame["_c_key"] = np.round(np.log10(frame["C"].astype(float)), 6)
    frame["_r_key"] = frame["l1_ratio"].fillna(-1.0).round(6)
    out: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    ctl = selection.loc[selection["atlas_row_block"].ne("biological") & selection["is_dr"] & selection["l1_or_en_selected"]]
    for row in ctl.itertuples(index=False):
        mask = (
            frame["stage0_panel_id"].eq(row.stage0_panel_id)
            & frame["configuration"].eq(row.configuration)
            & frame["penalty"].eq(row.penalty)
            & frame["_c_key"].eq(round(float(np.log10(row.C)), 6))
            & frame["_r_key"].eq(round(-1.0 if pd.isna(row.l1_ratio) else float(row.l1_ratio), 6))
        )
        sub = frame.loc[mask]
        per: Dict[str, List[int]] = {}
        for r in sub.itertuples(index=False):
            ids = [] if pd.isna(r.selected_feature_ids) or str(r.selected_feature_ids) == "" else str(r.selected_feature_ids).split(";")
            idx = sorted(feature_index(f) for f in ids)
            if len(idx) != int(r.nonzero_coefficient_count):
                raise AssertionError(f"{row.representation_id} {r.heldout_patient}: feature ids disagree with the count")
            per[str(r.heldout_patient)] = idx
        out[(row.stage0_panel_id, row.configuration)] = per
    return out


# --------------------------------------------------------------------------------------
# part: anchor
# --------------------------------------------------------------------------------------


def run_anchor(ctx: Ctx, cache: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in IFNA_METHODS:
        rep = representation_id(IFNA_PANEL, method, IFNA_K)
        coefs, setting = stored_bio_coefficients(cache, rep, IFNA_K)
        if setting.get("penalty") != "l1":
            raise AssertionError(f"anchor expects a stored L1 selection for {rep}, got {setting}")
        C = setting["C"]
        scores = load_scores(ctx.exp, IFNA_PANEL, method, IFNA_K, ctx.y.size)
        focus = {"pca": [7], "fa": [6, 7]}[method]  # PC8; FA7 and FA8 (0-based columns)
        for heldout in ctx.supported:
            train = ctx.patients != heldout
            Zs, _ = rc.standardize_train(scores[train])
            yt = ctx.y[train]
            gap = Zs[yt == 1].mean(0) - Zs[yt == 0].mean(0)
            margins = rc.l1_entry_margins(Zs, yt, C, standardize=False)
            thr = rc.l1_entry_threshold(C, int(train.sum()))
            j = int(np.argmax(margins))
            stored = coefs.get(heldout, np.zeros(IFNA_K))
            active = np.flatnonzero(np.abs(stored) > rc.COEF_NONZERO_TOL)
            row = {
                "representation_id": rep,
                "stage1_method": method,
                "heldout_patient": heldout,
                "n_train": int(train.sum()),
                "n_train_malignant": int(yt.sum()),
                "penalty": "l1",
                "C": C,
                "entry_threshold_gap": thr,
                "best_factor": f"factor_{j + 1:03d}",
                "best_abs_gap": float(abs(gap[j])),
                "max_entry_margin": float(margins[j]),
                "critical_C_multiplier": float(1.0 / margins[j]),
                "n_factors_passing": int(np.sum(margins > 1.0)),
                "predicted_nonzero": bool(margins[j] > 1.0),
                "stored_nonzero": bool(active.size > 0),
                "stored_active_factors": ";".join(f"factor_{i + 1:03d}" for i in active),
                "stored_coef_active": ";".join(f"{stored[i]:.6g}" for i in active),
            }
            for col in focus:
                tag = f"{'PC' if method == 'pca' else 'FA'}{col + 1}"
                row[f"gap_{tag}"] = float(gap[col])
                row[f"margin_{tag}"] = float(margins[col])
            row["agree"] = row["predicted_nonzero"] == row["stored_nonzero"]
            rows.append(row)
    table = pd.DataFrame(rows)
    atomic_write_csv(table, ctx.tables / "d19_anchor_ifna_l1_entry_check.csv")
    print(f"[anchor] {int(table['agree'].sum())}/{len(table)} fold outcomes reproduced", flush=True)
    return table


# --------------------------------------------------------------------------------------
# part: audit
# --------------------------------------------------------------------------------------


def _audit_one(
    exp: Path,
    y: np.ndarray,
    patients: np.ndarray,
    folds: Sequence[str],
    cell: Dict[str, Any],
    stored_coefs: Dict[str, np.ndarray],
    stored_support: Dict[str, List[int]],
) -> List[Dict[str, Any]]:
    k = int(cell["effective_k"])
    scores = load_scores(exp, cell["stage0_panel_id"], cell["stage1_method"], k, y.size)
    pen = cell["penalty"]
    ratio = None if pen == "l1" else float(cell["l1_ratio"])
    rho = 1.0 if ratio is None else ratio
    C = float(cell["C"])
    rows = []
    for heldout in folds:
        train = patients != heldout
        Zs, _ = rc.standardize_train(scores[train])
        yt = y[train]
        margins = rc.l1_entry_margins(Zs, yt, C, l1_ratio=ratio, standardize=False)
        order = np.argsort(-margins, kind="mergesort")
        j = int(order[0])
        second = float(margins[order[1]]) if margins.size > 1 else float("nan")
        gap_j = float(Zs[yt == 1, j].mean() - Zs[yt == 0, j].mean())
        test = ~train
        # Just above its entry C only feature j is active, so held-out scores rank as sign(gap_j) * z_j.
        single_lift = rc.heldout_lift(np.sign(gap_j) * scores[test, j].astype(np.float64), y[test])[2]
        if stored_coefs:
            coef = stored_coefs.get(heldout)
            if coef is None:
                raise AssertionError(f"{cell['representation_id']}: no stored coefficients for {heldout}")
            support = sorted(np.flatnonzero(np.abs(coef) > rc.COEF_NONZERO_TOL).tolist())
            source = "layer4_5_cache"
        else:
            if heldout not in stored_support:
                raise AssertionError(f"{cell['representation_id']}: no stored support for {heldout}")
            support = list(stored_support[heldout])
            coef = None
            source = "controls_by_heldout_patient"
        stored_nz = np.zeros(k, dtype=bool)
        stored_nz[support] = True
        predicted_nz = margins > 1.0
        row = {
            "representation_id": cell["representation_id"],
            "stage0_panel_id": cell["stage0_panel_id"],
            "atlas_row_block": cell["atlas_row_block"],
            "configuration": cell["configuration"],
            "stage1_method": cell["stage1_method"],
            "effective_k": k,
            "penalty": pen,
            "C": C,
            "l1_ratio": float("nan") if ratio is None else ratio,
            "C_at_grid_lower_bound": bool(rc.c_at_grid_lower_bound(C)),
            "heldout_patient": heldout,
            "n_train": int(train.sum()),
            "n_train_malignant": int(yt.sum()),
            "entry_threshold_gap": rc.l1_entry_threshold(C, int(train.sum()), ratio),
            "max_entry_margin": float(margins[j]),
            "argmax_feature": f"factor_{j + 1:03d}",
            "second_entry_margin": second,
            "n_features_margin_gt1": int(predicted_nz.sum()),
            "critical_C_multiplier": float(1.0 / margins[j]) if margins[j] > 0 else float("inf"),
            "argmax_gap_sign": int(np.sign(gap_j)),
            "argmax_single_feature_lift": float(single_lift),
            "predicted_all_zero": bool(margins[j] <= 1.0),
            "stored_n_nonzero": int(len(support)),
            "stored_all_zero": bool(len(support) == 0),
            "stored_nonzero_features": ";".join(f"factor_{i + 1:03d}" for i in support),
            "argmax_in_stored_support": bool(stored_nz[j]) if support else None,
            "feature_level_agreement": float(np.mean(predicted_nz == stored_nz)),
            "knife_edge_entry": bool(abs(margins[j] - 1.0) <= KNIFE_EDGE),
            "stored_source": source,
        }
        row["fold_agree"] = row["predicted_all_zero"] == row["stored_all_zero"]
        if coef is not None:
            kkt = rc.kkt_margins(Zs, yt, coef, C, l1_ratio=ratio)
            zero = ~stored_nz
            if zero.any():
                zj = int(np.flatnonzero(zero)[np.argmax(kkt[zero])])
                row["kkt_max_zero_feature_margin"] = float(kkt[zj])
                row["kkt_max_zero_feature"] = f"factor_{zj + 1:03d}"
                row["kkt_n_zero_features_ge_0p9"] = int(np.sum(kkt[zero] >= 1.0 - KNIFE_EDGE))
                row["kkt_n_zero_features_violating"] = int(np.sum(kkt[zero] > KKT_VIOLATION))
            else:
                row["kkt_max_zero_feature_margin"] = float("nan")
                row["kkt_max_zero_feature"] = ""
                row["kkt_n_zero_features_ge_0p9"] = 0
                row["kkt_n_zero_features_violating"] = 0
            row["next_entrant_knife_edge"] = bool(
                support and np.isfinite(row["kkt_max_zero_feature_margin"]) and row["kkt_max_zero_feature_margin"] >= 1.0 - KNIFE_EDGE
            )
        rows.append(row)
    return rows


def run_audit(ctx: Ctx, selection: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    from joblib import Parallel, delayed

    unit_dir = ctx.cache / "audit_units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    cells = selection.loc[selection["is_dr"] & selection["l1_or_en_selected"]].copy()
    cells["effective_k"] = cells["effective_k"].astype(int)
    cells = cells.sort_values(["atlas_row_block", "stage0_panel_id", "stage1_method", "effective_k"]).reset_index(drop=True)
    if ctx.args.max_cells:
        cells = cells.head(int(ctx.args.max_cells))
    controls = stored_control_supports(ctx, selection)
    folds = list(ctx.supported)

    # cross-check the cached biological setting against the D1 rule
    mismatches = []
    jobs = []
    finished: List[pd.DataFrame] = []
    for cell in cells.to_dict("records"):
        rep = cell["representation_id"]
        unit_path = unit_dir / f"{safe_name(rep)}.csv"
        if cell["atlas_row_block"] == "biological":
            coefs, setting = stored_bio_coefficients(cache, rep, int(cell["effective_k"]))
            same = (
                setting
                and setting["penalty"] == cell["penalty"]
                and np.isclose(setting["C"], float(cell["C"]), rtol=1e-9)
                and (cell["penalty"] != "elasticnet" or np.isclose(setting["l1_ratio"], float(cell["l1_ratio"])))
            )
            if not same:
                mismatches.append(rep)
                continue
            support: Dict[str, List[int]] = {}
        else:
            coefs = {}
            support = controls.get((cell["stage0_panel_id"], cell["configuration"]), {})
        done = read_csv_or_none(unit_path) if ctx.args.resume else None
        if done is not None and set(folds) <= set(done["heldout_patient"].astype(str)) and AUDIT_REQUIRED_COLUMNS <= set(done.columns):
            finished.append(done)
            continue
        jobs.append((cell, coefs, support, unit_path))
    if mismatches:
        raise AssertionError(f"{len(mismatches)} biological cells: cache setting != D1 rule, e.g. {mismatches[:3]}")
    print(f"[audit] {len(cells)} L1/EN-selected DR cells; {len(finished)} resumed, {len(jobs)} to run", flush=True)

    results = Parallel(n_jobs=ctx.n_jobs)(
        delayed(_audit_one)(ctx.exp, ctx.y, ctx.patients, folds, cell, coefs, support) for cell, coefs, support, _ in jobs
    )
    for (cell, _, _, unit_path), rows in zip(jobs, results):
        frame = pd.DataFrame(rows)
        atomic_write_csv(frame, unit_path)
        finished.append(frame)
    table = pd.concat(finished, ignore_index=True) if finished else pd.DataFrame()
    table = table.sort_values(["atlas_row_block", "representation_id", "heldout_patient"]).reset_index(drop=True)
    table = add_lift_at_stake(ctx, table)
    atomic_write_csv(table, ctx.tables / "d19_entry_margin_audit_by_fold.csv")
    return table


def add_lift_at_stake(ctx: Ctx, table: pd.DataFrame) -> pd.DataFrame:
    """Stored held-out lift (D1 atlas) and the lift just across the entry threshold.

    Knife-edge fold on the all-zero side: at C * CROSS_FACTOR / max_margin only the largest-margin
    feature is active (valid when second / max margin < 1 / CROSS_FACTOR), so the held-out ranking is
    sign(gap) * z_j and its lift is argmax_single_feature_lift. On the entered side, C * (1 / max_margin)
    / CROSS_FACTOR makes every margin < 1: the probe is all-zero and its lift is 0 by construction.
    lift_at_stake = lift_if_crossed - stored_heldout_lift; filled for every fold, read it for knife-edge folds.
    """
    stored = pd.read_csv(ctx.exp / ATLAS_PATIENT_TABLE, usecols=["representation_id", "heldout_patient", "heldout_lift"])
    stored = stored.rename(columns={"heldout_lift": "stored_heldout_lift"})
    out = table.drop(columns=[c for c in ("stored_heldout_lift",) if c in table.columns]).merge(
        stored, on=["representation_id", "heldout_patient"], how="left", validate="many_to_one"
    )
    if out["stored_heldout_lift"].isna().any():
        raise AssertionError("audit folds without a stored atlas lift")
    single_ok = (out["second_entry_margin"].fillna(0.0) / out["max_entry_margin"]) < 1.0 / CROSS_FACTOR
    crossed = np.where(out["predicted_all_zero"].astype(bool), np.where(single_ok, out["argmax_single_feature_lift"], np.nan), 0.0)
    out["lift_if_crossed"] = crossed
    out["lift_if_crossed_valid"] = np.where(out["predicted_all_zero"].astype(bool), single_ok, True)
    out["lift_at_stake"] = out["lift_if_crossed"] - out["stored_heldout_lift"]
    return out


def audit_summary(table: pd.DataFrame) -> pd.DataFrame:
    def summarise(frame: pd.DataFrame, scope: str, value: str) -> Dict[str, Any]:
        cells = frame.groupby("representation_id")
        row = {
            "scope": scope,
            "group": value,
            "n_cells": int(frame["representation_id"].nunique()),
            "n_folds": int(len(frame)),
            "n_fold_agree": int(frame["fold_agree"].sum()),
            "fold_agreement_rate": float(frame["fold_agree"].mean()),
            "n_predicted_all_zero": int(frame["predicted_all_zero"].sum()),
            "n_stored_all_zero": int(frame["stored_all_zero"].sum()),
            "n_knife_edge_entry": int(frame["knife_edge_entry"].sum()),
            "n_knife_edge_all_zero_side": int((frame["knife_edge_entry"] & frame["predicted_all_zero"]).sum()),
            "n_knife_edge_entered_side": int((frame["knife_edge_entry"] & ~frame["predicted_all_zero"]).sum()),
            "n_cells_with_knife_edge_fold": int(cells["knife_edge_entry"].any().sum()),
            "n_cells_mixed_zero_nonzero_folds": int((cells["stored_all_zero"].any() & ~cells["stored_all_zero"].all()).sum()),
            "n_cells_all_folds_zero": int(cells["stored_all_zero"].all().sum()),
            "n_cells_at_C_lower_bound": int(cells["C_at_grid_lower_bound"].first().sum()),
            "mean_feature_level_agreement": float(frame["feature_level_agreement"].mean()),
            "n_nonzero_folds_argmax_in_support": int(truthy(frame["argmax_in_stored_support"]).sum()),
            "n_nonzero_folds": int((~frame["stored_all_zero"]).sum()),
        }
        if "lift_at_stake" in frame.columns:
            ke = frame.loc[truthy(frame["knife_edge_entry"])]
            stake = ke["lift_at_stake"].abs()
            row["n_knife_edge_stake_valid"] = int(stake.notna().sum())
            row["n_knife_edge_stake_ge_1pp"] = int((stake >= 0.01).sum())
            row["n_knife_edge_stake_ge_5pp"] = int((stake >= 0.05).sum())
            row["median_knife_edge_abs_stake"] = float(stake.median()) if stake.notna().any() else float("nan")
            row["max_knife_edge_abs_stake"] = float(stake.max()) if stake.notna().any() else float("nan")
        if "kkt_max_zero_feature_margin" in frame.columns:
            kk = frame.loc[frame["kkt_max_zero_feature_margin"].notna()]
            row["n_folds_with_kkt"] = int(len(kk))
            row["n_next_entrant_knife_edge"] = int(truthy(frame["next_entrant_knife_edge"]).sum())
            row["n_folds_kkt_violation"] = int((kk["kkt_n_zero_features_violating"] > 0).sum())
        return row

    rows = [summarise(table, "all", "all")]
    for col in ("atlas_row_block", "penalty", "stage1_method", "C_at_grid_lower_bound", "heldout_patient"):
        for value, grp in table.groupby(col):
            rows.append(summarise(grp, col, str(value)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# part: ifna (W3.2)
# --------------------------------------------------------------------------------------


def _ifna_unit(kind: str, method: str, heldout: str, pen: str, ratio: Optional[float], C: Optional[float], mult: Optional[float],
               scores: np.ndarray, y: np.ndarray, patients: np.ndarray, c_grid: Sequence[float]) -> Tuple[Dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    train = patients != heldout
    test = ~train
    record: Dict[str, Any] = {
        "kind": kind, "stage1_method": method, "effective_k": IFNA_K, "heldout_patient": heldout,
        "penalty": pen, "l1_ratio": ratio, "C_multiplier": mult,
        "n_train": int(train.sum()), "n_train_malignant": int(y[train].sum()),
    }
    if kind == "tune":
        tuned = rc.tune_c_inner_lopo(scores[train], y[train], patients[train], penalty=pen, l1_ratio=ratio, c_grid=c_grid, n_jobs=1)
        C = tuned["best_C"]
        record.update(
            best_C=C, inner_median_lift=tuned["best_score"], inner_nonzero_mean=tuned["best_nonzero_mean"],
            n_inner_folds=tuned["n_inner_folds"], inner_folds=";".join(tuned["inner_folds"]), fallback=tuned["fallback"],
            fallback_reason=tuned["fallback_reason"], best_C_at_grid_lower_bound=tuned["at_grid_lower_bound"],
            best_C_at_grid_upper_bound=bool(np.isclose(C, max(c_grid), rtol=1e-9)),
            scores_by_C={repr(k): v for k, v in tuned["scores_by_C"].items()},
            nonzero_by_C={repr(k): v for k, v in tuned["nonzero_by_C"].items()},
            per_fold=tuned["per_fold"],
        )
    est = rc.fit_probe(scores[train], y[train], penalty=pen, C=float(C), l1_ratio=ratio)
    coef = rc.probe_coef(est)
    prob = rc.probe_scores(est, scores[test])
    ap, prevalence, lift = rc.heldout_lift(prob, y[test])
    active = np.flatnonzero(np.abs(coef) > rc.COEF_NONZERO_TOL)
    Zs, _ = rc.standardize_train(scores[train])
    margins = rc.l1_entry_margins(Zs, y[train], float(C), l1_ratio=ratio if pen == "elasticnet" else None, standardize=False) if pen != "l2" else None
    record.update(
        C=float(C), heldout_ap=ap, prevalence=prevalence, heldout_lift=lift,
        n_heldout=int(test.sum()), n_heldout_malignant=int(y[test].sum()),
        n_nonzero=int(active.size), nonzero_factors=";".join(f"factor_{i + 1:03d}" for i in active),
        top_abs_coef_factor=f"factor_{int(np.argmax(np.abs(coef))) + 1:03d}" if np.any(coef) else "",
        coef_PC8_or_FA7=float(coef[7] if method == "pca" else coef[6]),
        max_entry_margin=float(np.max(margins)) if margins is not None else None,
        argmax_entry_factor=f"factor_{int(np.argmax(margins)) + 1:03d}" if margins is not None else None,
        margin_PC8_or_FA7=float(margins[7] if method == "pca" else margins[6]) if margins is not None else None,
        n_iter=int(np.max(est.named_steps["logreg"].n_iter_)), seconds=time.perf_counter() - t0, status="done",
    )
    return record, prob.astype(np.float32)


def ifna_units(ctx: Ctx, stored_C: Dict[str, float]) -> List[Dict[str, Any]]:
    folds = list(ctx.supported)
    mults = SWEEP_MULTIPLIERS
    penalties = TUNE_PENALTIES
    if ctx.args.smoke:
        folds = [f for f in ("P02", "P07") if f in folds]
        mults = (1.0, 1.05)
        penalties = (("l1", None), ("l2", None))
    units = []
    for method in IFNA_METHODS:
        for heldout in folds:
            for pen, ratio in penalties:
                tag = pen if ratio is None else f"{pen}{ratio:g}"
                units.append(dict(unit=f"tune__{method}{IFNA_K}__{heldout}__{tag}", kind="tune", method=method, heldout=heldout,
                                  pen=pen, ratio=ratio, C=None, mult=None))
            for mult in mults:
                units.append(dict(unit=f"sweep__{method}{IFNA_K}__{heldout}__l1__x{mult:g}", kind="sweep_l1", method=method,
                                  heldout=heldout, pen="l1", ratio=None, C=stored_C[method] * mult, mult=mult))
            units.append(dict(unit=f"fixed__{method}{IFNA_K}__{heldout}__l2__x1", kind="fixed_l2", method=method, heldout=heldout,
                              pen="l2", ratio=None, C=stored_C[method], mult=1.0))
    # slow units first for load balance
    order = {"tune": 0, "sweep_l1": 1, "fixed_l2": 2}
    units.sort(key=lambda u: (order[u["kind"]], u["pen"] == "l2", u["unit"]))
    return units


def run_ifna(ctx: Ctx, cache: pd.DataFrame) -> pd.DataFrame:
    from joblib import Parallel, delayed

    unit_dir = ctx.cache / "ifna_units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    stored_C = {}
    for method in IFNA_METHODS:
        _, setting = stored_bio_coefficients(cache, representation_id(IFNA_PANEL, method, IFNA_K), IFNA_K)
        if setting.get("penalty") != "l1":
            raise AssertionError(f"IFN-alpha {method}{IFNA_K}: stored selection is not L1 ({setting})")
        stored_C[method] = setting["C"]
    c_grid = list(rc.DEFAULT_C_GRID)
    if ctx.args.smoke:
        c_grid = c_grid[:: max(1, int(ctx.args.c_grid_stride))]
    scores = {m: load_scores(ctx.exp, IFNA_PANEL, m, IFNA_K, ctx.y.size) for m in IFNA_METHODS}
    units = ifna_units(ctx, stored_C)
    todo = []
    records = []
    for u in units:
        jpath, ppath = unit_dir / f"{u['unit']}.json", unit_dir / f"{u['unit']}.npy"
        done = read_json_or_none(jpath) if ctx.args.resume else None
        if done is not None and u["kind"] == "tune" and done.get("c_grid_n") != len(c_grid):
            done = None  # tuned on a different C grid (e.g. a smoke grid): rerun, never mix
        if done is not None:
            try:
                prob = np.load(ppath)
                if prob.shape == (int(done["n_heldout"]),):
                    records.append(done)
                    continue
            except Exception:
                pass
        todo.append(u)
    print(f"[ifna] {len(units)} units; {len(records)} resumed, {len(todo)} to run (n_jobs={ctx.n_jobs})", flush=True)
    t0 = time.perf_counter()
    results = Parallel(n_jobs=ctx.n_jobs, verbose=0)(
        delayed(_ifna_unit)(u["kind"], u["method"], u["heldout"], u["pen"], u["ratio"], u["C"], u["mult"],
                            scores[u["method"]], ctx.y, ctx.patients, c_grid)
        for u in todo
    )
    for u, (record, prob) in zip(todo, results):
        record["unit"] = u["unit"]
        record["c_grid_n"] = len(c_grid)
        np.save(unit_dir / f"{u['unit']}.npy", prob)
        atomic_write_json(record, unit_dir / f"{u['unit']}.json")
        records.append(record)
    print(f"[ifna] ran {len(todo)} units in {time.perf_counter() - t0:.1f} s", flush=True)
    flat = pd.DataFrame([{k: v for k, v in r.items() if k not in ("scores_by_C", "nonzero_by_C", "per_fold")} for r in records])
    flat = flat.sort_values(["kind", "stage1_method", "penalty", "C_multiplier", "heldout_patient"], na_position="first").reset_index(drop=True)
    atomic_write_csv(flat, ctx.tables / "d19_ifna_units.csv")
    inner = []
    for r in records:
        if r["kind"] != "tune":
            continue
        for f in r["per_fold"]:
            inner.append({"stage1_method": r["stage1_method"], "outer_heldout_patient": r["heldout_patient"], **f})
    if inner:
        atomic_write_csv(pd.DataFrame(inner), ctx.tables / "d19_ifna_inner_lopo_scores_by_C.csv")
    return flat


# --------------------------------------------------------------------------------------
# part: cuml_check (optional, GPU seconds)
# --------------------------------------------------------------------------------------


def run_cuml_check(ctx: Ctx) -> pd.DataFrame:
    from cuml.linear_model import LogisticRegression as CuMLLogisticRegression  # noqa: WPS433

    rng = np.random.default_rng(0)
    n, k = 4000, 6
    y = (rng.random(n) < 0.08).astype(int)
    Z = rng.normal(size=(n, k))
    Z[:, 0] += 0.35 * y
    Z[:, 1] -= 0.2 * y
    Zs, _ = rc.standardize_train(Z)
    rows = []
    for pen, ratio in (("l1", None), ("elasticnet", 0.5)):
        c_entry = float(np.min(rc.l1_entry_c(Zs, y, l1_ratio=ratio, standardize=False)))
        for mult in (0.97, 1.03, 1.2):
            C = c_entry * mult
            kw = dict(penalty=pen, C=C, class_weight="balanced", max_iter=5000, output_type="numpy")
            if ratio is not None:
                kw["l1_ratio"] = ratio
            gpu = np.asarray(CuMLLogisticRegression(**kw).fit(Zs.astype(np.float32), y).coef_).ravel()
            cpu = rc.probe_coef(rc.fit_probe(Zs, y, penalty=pen, C=C, l1_ratio=ratio, standardize=False, solver="saga", tol=1e-10, max_iter=200000))
            newton = rc.probe_coef(rc.fit_probe(Zs, y, penalty=pen, C=C, l1_ratio=ratio, standardize=False))
            rows.append({
                "penalty": pen, "l1_ratio": ratio, "C_over_entry_C": mult, "C": C,
                "max_entry_margin": float(np.max(rc.l1_entry_margins(Zs, y, C, l1_ratio=ratio, standardize=False))),
                "cuml_nonzero": int(np.sum(np.abs(gpu) > rc.COEF_NONZERO_TOL)),
                "sklearn_saga_nonzero": int(np.sum(np.abs(cpu) > rc.COEF_NONZERO_TOL)),
                "cuml_coef_factor1": float(gpu[0]), "sklearn_coef_factor1": float(cpu[0]),
                "newton_cd_nonzero": int(np.sum(np.abs(newton) > rc.COEF_NONZERO_TOL)),
                "max_abs_coef_diff": float(np.max(np.abs(gpu - cpu))),
                "max_abs_coef_diff_newton_vs_saga": float(np.max(np.abs(newton - cpu))),
            })
    table = pd.DataFrame(rows)
    atomic_write_csv(table, ctx.tables / "d19_cuml_semantics_check.csv")
    print(table.to_string(index=False), flush=True)
    return table


# --------------------------------------------------------------------------------------
# part: report
# --------------------------------------------------------------------------------------


#: Arms whose C was chosen with held-out outcomes: the D1 rule picks C by pooled LOPO AUPRC
#: over all folds, including each held-out patient, so these arms are not nested in C and part
#: of any stored-vs-tuned gap can be selection optimism.
NOT_NESTED_ARMS = ("stored_l1_fixed", "refit_l1_fixed", "sweep_l1_x1.05", "fixed_l2")
ARM_LABELS = {
    "stored_l1_fixed": "stored L1 at selected C (cuML, atlas; C not nested)",
    "refit_l1_fixed": "L1 refit at selected C (newton_cd, sklearn objective; C not nested)",
    "sweep_l1_x1.05": "L1 at 1.05 x selected C (C not nested)",
    "fixed_l2": "L2 at selected C (C not nested)",
    "tuned_l1": "L1, C by inner LOPO",
    "tuned_l2": "L2, C by inner LOPO",
    "tuned_elasticnet0.5": "EN (l1_ratio 0.5), C by inner LOPO",
}


def ifna_arm_tables(ctx: Ctx, units: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-patient wide table and per-arm summary (pooled AP over supported held-out cells, median, P20)."""
    stored = pd.read_csv(ctx.exp / ATLAS_PATIENT_TABLE)
    unit_dir = ctx.cache / "ifna_units"
    arms: Dict[Tuple[str, str], pd.DataFrame] = {}
    for method in IFNA_METHODS:
        rep = representation_id(IFNA_PANEL, method, IFNA_K)
        s = stored.loc[stored["representation_id"].eq(rep), ["heldout_patient", "heldout_auprc", "heldout_lift", "patient_prevalence"]]
        arms[(method, "stored_l1_fixed")] = s.rename(columns={"heldout_auprc": "ap", "heldout_lift": "lift", "patient_prevalence": "prevalence"}).assign(C=np.nan, n_nonzero=np.nan)
        sub = units.loc[units["stage1_method"].eq(method)]
        picks = {
            "refit_l1_fixed": sub.loc[sub["kind"].eq("sweep_l1") & np.isclose(sub["C_multiplier"].astype(float), 1.0)],
            "sweep_l1_x1.05": sub.loc[sub["kind"].eq("sweep_l1") & np.isclose(sub["C_multiplier"].astype(float), 1.05)],
            "fixed_l2": sub.loc[sub["kind"].eq("fixed_l2")],
        }
        for pen, ratio in TUNE_PENALTIES:
            tag = pen if ratio is None else f"{pen}{ratio:g}"
            picks[f"tuned_{tag}"] = sub.loc[sub["kind"].eq("tune") & sub["penalty"].eq(pen)]
        for arm, frame in picks.items():
            if frame.empty:
                continue
            arms[(method, arm)] = frame.rename(columns={"heldout_ap": "ap", "heldout_lift": "lift"})[
                ["heldout_patient", "ap", "lift", "prevalence", "C", "n_nonzero", "unit"]
            ]
    # per-arm summary
    summary = []
    for (method, arm), frame in arms.items():
        pooled = float("nan")
        if "unit" in frame.columns and frame["unit"].notna().all() and len(frame) == len(ctx.supported):
            ys, ps = [], []
            for r in frame.itertuples(index=False):
                ps.append(np.load(unit_dir / f"{r.unit}.npy"))
                ys.append(ctx.y[ctx.patients == r.heldout_patient])
            pooled = rc.heldout_lift(np.concatenate(ps), np.concatenate(ys))[0]
        sup = frame.loc[frame["heldout_patient"].isin(ctx.supported)]
        summary.append({
            "stage1_method": method, "arm": arm, "arm_label": ARM_LABELS.get(arm, arm),
            "c_nested": arm not in NOT_NESTED_ARMS,
            "n_patients": int(len(sup)),
            "pooled_ap_supported_heldout_cells": pooled,
            "median_lift": median(sup["lift"]), "p20_lift": p20(sup["lift"]),
            "n_lift_gt_0": int((sup["lift"] > 0).sum()), "n_lift_gt_5pp": int((sup["lift"] > 0.05).sum()),
            "n_all_zero_folds": int((sup["n_nonzero"] == 0).sum()) if sup["n_nonzero"].notna().any() else None,
            "C_values": ";".join(sorted({f"{c:.3g}" for c in sup["C"].dropna()})),
        })
    summary_df = pd.DataFrame(summary)
    # wide per-patient table
    wide = pd.DataFrame({"heldout_patient": list(ctx.supported)})
    prev = arms[(IFNA_METHODS[0], "stored_l1_fixed")].set_index("heldout_patient")["prevalence"]
    wide["prevalence"] = wide["heldout_patient"].map(prev)
    wide["n_malignant"] = [int(ctx.y[ctx.patients == p].sum()) for p in wide["heldout_patient"]]
    for (method, arm), frame in arms.items():
        f = frame.set_index("heldout_patient")
        wide[f"{method}{IFNA_K}_{arm}_lift"] = wide["heldout_patient"].map(f["lift"])
        if f["C"].notna().any():
            wide[f"{method}{IFNA_K}_{arm}_C"] = wide["heldout_patient"].map(f["C"])
            wide[f"{method}{IFNA_K}_{arm}_n_nonzero"] = wide["heldout_patient"].map(f["n_nonzero"])
    for arm in sorted({a for _, a in arms}):
        a, b = f"pca{IFNA_K}_{arm}_lift", f"fa{IFNA_K}_{arm}_lift"
        if a in wide and b in wide:
            wide[f"pca_minus_fa_{arm}_lift"] = wide[a] - wide[b]
    return wide, summary_df


def make_figures(ctx: Ctx, audit: Optional[pd.DataFrame], units: Optional[pd.DataFrame], wide: Optional[pd.DataFrame]) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    })
    written: List[Path] = []

    def save(fig, stem):
        for ext in ("png", "pdf"):
            path = ctx.figures / f"{stem}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    if audit is not None and not audit.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        m = np.log10(audit["max_entry_margin"].clip(lower=1e-3))
        bins = np.linspace(min(-2.0, float(m.min())), max(2.5, float(m.max())), 60)
        ax.axvspan(np.log10(1 - KNIFE_EDGE), np.log10(1 + KNIFE_EDGE), color=grid, alpha=0.9, lw=0, label="within ±10% of threshold")
        ax.hist(m[~audit["stored_all_zero"]], bins=bins, histtype="step", lw=2, color=blue, label="stored probe nonzero")
        ax.hist(m[audit["stored_all_zero"]], bins=bins, histtype="step", lw=2, color=orange, label="stored probe all-zero")
        ax.axvline(0.0, color=ink, lw=1)
        ax.set_xlabel("log10 max entry margin at w = 0  (|class-mean gap| / (4·l1_ratio / (C·n_train)))")
        ax.set_ylabel("atlas cell × supported fold")
        ax.set_title("D19A  L1/EN-selected atlas DR cells: entry margin vs stored outcome", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        save(fig, "D19A_entry_margin_distribution")

    if units is not None and not units.empty:
        sweep = units.loc[units["kind"].eq("sweep_l1")]
        if not sweep.empty:
            pats = [p for p in ctx.supported if p in set(sweep["heldout_patient"])]
            fig, axes = plt.subplots(1, len(pats), figsize=(1.55 * len(pats) + 0.6, 2.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, p in zip(axes, pats):
                for method, color in zip(IFNA_METHODS, (blue, orange)):
                    s = sweep.loc[sweep["heldout_patient"].eq(p) & sweep["stage1_method"].eq(method)].sort_values("C_multiplier")
                    ax.plot(s["C_multiplier"], 100 * s["heldout_lift"], "-o", color=color, lw=2, ms=4, label=f"{method.upper()}{IFNA_K}")
                ax.axhline(0, color=muted, lw=0.8)
                ax.axvline(1.0, color=muted, lw=0.8, ls=":")
                ax.set_title(p, fontsize=9)
                ax.set_xlabel("C / selected C")
            axes[0].set_ylabel("held-out lift (pp)")
            axes[-1].legend(frameon=False, fontsize=7, loc="upper left")
            fig.suptitle("D19B  IFN-α L1 sweep around the selected C (C = 1e-4)", x=0.01, y=1.06, ha="left", fontsize=10)
            save(fig, "D19B_ifna_l1_c_sweep")
    if wide is not None and not wide.empty:
        arms = [a for a in ("stored_l1_fixed", "sweep_l1_x1.05", "fixed_l2", "tuned_l1", "tuned_l2", "tuned_elasticnet0.5")
                if f"pca{IFNA_K}_{a}_lift" in wide and f"fa{IFNA_K}_{a}_lift" in wide]
        fig, axes = plt.subplots(1, len(arms), figsize=(2.2 * len(arms) + 0.4, 2.9), sharey=True)
        axes = np.atleast_1d(axes)
        x = np.arange(len(wide))
        for ax, arm in zip(axes, arms):
            ax.bar(x - 0.2, 100 * wide[f"pca{IFNA_K}_{arm}_lift"], 0.38, color=blue, label=f"PCA{IFNA_K}")
            ax.bar(x + 0.2, 100 * wide[f"fa{IFNA_K}_{arm}_lift"], 0.38, color=orange, label=f"FA{IFNA_K}")
            ax.axhline(0, color=muted, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(wide["heldout_patient"], rotation=90, fontsize=7)
            ax.set_title(ARM_LABELS.get(arm, arm), fontsize=7.5)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("held-out lift (pp)")
        axes[0].legend(frameon=False, fontsize=7, loc="upper left")
        fig.suptitle("D19C  IFN-α PCA40 vs FA40 per supported patient, by C policy", x=0.01, y=1.08, ha="left", fontsize=10)
        save(fig, "D19C_ifna_pca40_vs_fa40_by_c_policy")
    return written


def _fmt_pp(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f} pp"


def write_findings(ctx: Ctx, anchor: Optional[pd.DataFrame], selection: Optional[pd.DataFrame], audit_sum: Optional[pd.DataFrame],
                   audit: Optional[pd.DataFrame], wide: Optional[pd.DataFrame], arm_sum: Optional[pd.DataFrame],
                   units: Optional[pd.DataFrame]) -> Path:
    L: List[str] = [f"# D19 regularization contract ({SLUG})", ""]
    L += [
        "Generated by scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/run_d19_regularization_contract.py.",
        "Stage 1 is the stored transductive basis (seed 42); LOPO is label transfer, not inductive deployment.",
        "An all-zero L1 probe scores the constant-score no-skill AP = prevalence, so its lift is 0 by construction.",
        "",
        "## Entry rule",
        "Balanced class weights, fitted intercept, train-fold StandardScaler: at w = 0 the loss gradient of feature j is",
        "(n/4)(mean_normal_j - mean_malignant_j), so a feature can enter only if |standardized class-mean gap| > 4*l1_ratio/(C*n_train)",
        "(l1_ratio = 1 for L1). The stored Set A grids used cuML (QN, l1_strength = l1_ratio/C, normalised by n): same C semantics.",
        "",
    ]
    if anchor is not None:
        L += ["## Anchor (IFN-alpha PCA40/FA40, stored L1 C)", f"- Fold outcomes reproduced: {int(anchor['agree'].sum())}/{len(anchor)}."]
        for r in anchor.itertuples(index=False):
            L.append(f"  - {r.stage1_method}40 {r.heldout_patient}: n_train {r.n_train}, threshold {r.entry_threshold_gap:.4f}, best {r.best_factor} "
                     f"margin {r.max_entry_margin:.3f} -> predicted {'nonzero' if r.predicted_nonzero else 'all-zero'}, stored "
                     f"{r.stored_active_factors if isinstance(r.stored_active_factors, str) and r.stored_active_factors else 'all-zero'}")
        L.append("")
    if selection is not None:
        lb = selection["C_at_grid_lower_bound"]
        L += ["## Atlas cells at the C-grid lower bound (C = 1e-4)",
              f"- {int(lb.sum())}/{len(selection)} atlas cells select C at the lower bound "
              f"({', '.join(f'{p} {int(n)}' for p, n in selection.loc[lb, 'penalty'].value_counts().sort_index().items())}).",
              f"- L1/EN-selected DR cells: {int((selection['is_dr'] & selection['l1_or_en_selected']).sum())}; of these at the lower bound: "
              f"{int((selection['is_dr'] & selection['l1_or_en_selected'] & lb).sum())}.", ""]
    if audit_sum is not None and audit is not None and not audit.empty:
        a = audit_sum.loc[audit_sum["scope"].eq("all")].iloc[0]
        L += ["## Entry-margin audit (L1/EN-selected atlas DR cells x supported folds)",
              f"- Cells {a['n_cells']}, folds {a['n_folds']}; predicted vs stored all-zero/nonzero agreement {a['n_fold_agree']}/{a['n_folds']} "
              f"({a['fold_agreement_rate']:.4f}).",
              f"- Knife-edge folds (max entry margin within ±10% of the threshold): {a['n_knife_edge_entry']} "
              f"({a['n_knife_edge_all_zero_side']} on the all-zero side, {a['n_knife_edge_entered_side']} just entered), in {a['n_cells_with_knife_edge_fold']} cells.",
              f"- Stored all-zero folds {a['n_stored_all_zero']}; cells with a mix of all-zero and nonzero folds {a['n_cells_mixed_zero_nonzero_folds']}; "
              f"cells all-zero in every supported fold {a['n_cells_all_folds_zero']}.",
              f"- Nonzero folds whose stored support contains the largest-margin feature: {a['n_nonzero_folds_argmax_in_support']}/{a['n_nonzero_folds']}.",
              f"- Mean per-feature agreement of 'margin > 1' with stored nonzero (first-entry heuristic only): {a['mean_feature_level_agreement']:.4f}."]
        if "n_folds_with_kkt" in a and pd.notna(a.get("n_folds_with_kkt")):
            L.append(f"- Biological folds with stored coefficients: {int(a['n_folds_with_kkt'])}; nonzero folds whose next entrant (a zero feature) has KKT margin >= 0.9: "
                     f"{int(a['n_next_entrant_knife_edge'])}; folds with a zero feature above {KKT_VIOLATION} (stored fit not at the KKT point): {int(a['n_folds_kkt_violation'])}.")
        if "n_knife_edge_stake_valid" in a and pd.notna(a.get("n_knife_edge_stake_valid")):
            L.append(f"- Lift at stake in knife-edge folds (held-out lift just across the threshold minus stored lift; all-zero side = the "
                     f"largest-margin factor alone, entered side = 0 by construction): {int(a['n_knife_edge_stake_valid'])} folds evaluable; "
                     f"|change| >= 1 pp in {int(a['n_knife_edge_stake_ge_1pp'])}, >= 5 pp in {int(a['n_knife_edge_stake_ge_5pp'])}; "
                     f"median |change| {_fmt_pp(a['median_knife_edge_abs_stake'])}, max {_fmt_pp(a['max_knife_edge_abs_stake'])}.")
            ke = audit.loc[truthy(audit["knife_edge_entry"]) & audit["lift_at_stake"].notna()].copy()
            ke["_abs"] = ke["lift_at_stake"].abs()
            top = ke.sort_values("_abs", ascending=False).head(12)
            if len(top):
                L += ["", "Largest knife-edge stakes:", "",
                      "| cell | fold | penalty, C | max margin | stored lift | lift just across | change |", "|---|---|---|---|---|---|---|"]
                for r in top.itertuples(index=False):
                    cell = f"{r.stage0_panel_id.replace('single_geneset__', '').replace('single_group__', 'group:')} {r.configuration}"
                    pen = r.penalty if r.penalty == "l1" else f"EN {r.l1_ratio:g}"
                    L.append(f"| {cell} | {r.heldout_patient} | {pen}, {r.C:.3g} | {r.max_entry_margin:.3f} | {_fmt_pp(r.stored_heldout_lift)} | "
                             f"{_fmt_pp(r.lift_if_crossed)} | {100 * r.lift_at_stake:+.2f} pp |")
                L.append("")
        dis = audit.loc[~audit["fold_agree"]]
        if len(dis):
            L.append("- Disagreeing folds:")
            for r in dis.itertuples(index=False):
                L.append(f"  - {r.representation_id} {r.heldout_patient}: margin {r.max_entry_margin:.4f}, stored nonzero {r.stored_n_nonzero}")
        L.append("")
    if arm_sum is not None and wide is not None and not arm_sum.empty:
        L += ["## W3.2 IFN-alpha PCA40 vs FA40 by C policy (7 supported held-out patients)", "",
              "| method | arm | pooled AP (supported held-out cells) | median lift | P20 lift | patients lift>0 | all-zero folds | C |",
              "|---|---|---|---|---|---|---|---|"]
        for r in arm_sum.itertuples(index=False):
            pooled = "n/a (not stored)" if not np.isfinite(r.pooled_ap_supported_heldout_cells) else f"{r.pooled_ap_supported_heldout_cells:.4f}"
            L.append(f"| {r.stage1_method}40 | {r.arm_label} | {pooled} | {_fmt_pp(r.median_lift)} | {_fmt_pp(r.p20_lift)} | {r.n_lift_gt_0}/{r.n_patients} | "
                     f"{'' if r.n_all_zero_folds is None or (isinstance(r.n_all_zero_folds, float) and np.isnan(r.n_all_zero_folds)) else int(r.n_all_zero_folds)} | {r.C_values} |")
        L += ["", "Pooled AP pools probabilities from seven different fold models, so it depends on cross-fold calibration: an all-zero "
              "fold gives every cell of that held-out patient one tied score. Read median and P20 lift first; pooled AP is reported "
              "because R6 asks for all three together.", "",
              "C not nested: the stored, refit, 1.05 x and fixed-L2 arms use the atlas C, which the D1 rule selected by pooled LOPO "
              "AUPRC over all folds, including each held-out patient (C selected with held-out outcomes). Only the tuned arms choose "
              "C without the held-out patient, so part of any stored-vs-tuned gap can be selection optimism.", "",
              "Per-patient lift (pp), PCA40 / FA40:", "", "| patient | prevalence | " + " | ".join(ARM_LABELS[a] for a in ARM_LABELS if f"pca40_{a}_lift" in wide) + " |",
              "|---|---|" + "---|" * sum(1 for a in ARM_LABELS if f"pca40_{a}_lift" in wide)]
        for r in wide.to_dict("records"):
            cells = [f"{100 * r[f'pca40_{a}_lift']:.2f} / {100 * r[f'fa40_{a}_lift']:.2f}" for a in ARM_LABELS if f"pca40_{a}_lift" in wide]
            L.append(f"| {r['heldout_patient']} | {r['prevalence']:.4f} | " + " | ".join(cells) + " |")
        L.append("")
    if wide is not None and not wide.empty:
        diffs = []
        for method in IFNA_METHODS:
            a, b = f"{method}{IFNA_K}_stored_l1_fixed_lift", f"{method}{IFNA_K}_refit_l1_fixed_lift"
            if a in wide and b in wide:
                diffs.append(np.nanmax(np.abs(wide[a] - wide[b])))
        if diffs:
            L += [f"Refit check: the newton_cd L1 refit (sklearn objective) at the selected C reproduces the stored (cuML) per-patient "
                  f"lift within {100 * max(diffs):.2g} pp ({max(diffs):.1e} AP; max over 7 patients x 2 methods), so its pooled AP "
                  f"stands in for the stored arm.", ""]
    cuml_path = ctx.tables / "d19_cuml_semantics_check.csv"
    if cuml_path.is_file():
        cu = pd.read_csv(cuml_path)
        same = int(((cu["cuml_nonzero"] == cu["sklearn_saga_nonzero"]) & (cu["newton_cd_nonzero"] == cu["sklearn_saga_nonzero"])).sum())
        L += ["## Backend check (synthetic, GPU seconds)",
              f"- cuML LogisticRegression, sklearn saga (tol 1e-10) and the contract's newton_cd at 0.97/1.03/1.2 x the entry C, L1 and EN(0.5): "
              f"same nonzero count in {same}/{len(cu)} settings; max |coef difference| cuML vs saga {cu['max_abs_coef_diff'].max():.2e}, "
              f"newton_cd vs saga {cu['max_abs_coef_diff_newton_vs_saga'].max():.2e}. C has the same meaning on both backends.", ""]
    if units is not None and not units.empty:
        tune = units.loc[units["kind"].eq("tune")]
        if not tune.empty:
            L += ["Tuned C per outer fold (inner LOPO over supported training patients; criterion median inner lift; ties sparser then smaller C):", ""]
            for (method, pen), grp in tune.groupby(["stage1_method", "penalty"]):
                L.append(f"- {method}40 {pen}: " + ", ".join(f"{r.heldout_patient} C={r.C:.3g} (nz {r.n_nonzero}{', fallback' if r.fallback else ''})" for r in grp.itertuples(index=False)))
            L.append("")
    L += ["## Claim boundary",
          "Descriptive, on the stored transductive Stage 1 basis over seven supported patients. Tuned-C results are nested only in C",
          "(method and K are fixed to the IFN-alpha PCA40/FA40 pair, which was itself chosen after inspecting LOPO results).",
          "The stored, refit, 1.05 x and fixed-L2 arms are not nested at all: their C was selected with held-out outcomes.", ""]
    path = ctx.tables / f"{SLUG}_findings.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest_and_ledger(ctx: Ctx, started: float, extra: Dict[str, Any]) -> None:
    import sklearn

    head, dirty = git_state(REPO)
    inputs = [ATLAS_TABLE, ATLAS_PATIENT_TABLE, COEF_CACHE, CONTROL_BY_FOLD] + [f"analysis/scorecards/{rel}" for _, rel, _ in SCORECARDS]
    input_rows = []
    for rel in inputs:
        p = ctx.exp / rel
        input_rows.append({"path": rel, "bytes": p.stat().st_size if p.exists() else None,
                           "mtime_ns": p.stat().st_mtime_ns if p.exists() else None})
    input_rows.append({"path": ctx.rel(ctx.meta_path), "bytes": Path(ctx.meta_path).stat().st_size, "mtime_ns": Path(ctx.meta_path).stat().st_mtime_ns})
    manifest = {
        "slug": SLUG,
        "analysis_id": "D19",
        "tasks": ["W3.1", "W3.2"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(a) for a in [Path(sys.argv[0]).name] + sys.argv[1:]),
        "git_head": head,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "n_jobs": ctx.n_jobs,
        "smoke": bool(ctx.args.smoke),
        "parts_runtime_seconds": ctx.timings,
        "total_runtime_seconds": time.perf_counter() - started,
        "supported_patients": ctx.supported,
        "stage1_basis": "stored transductive stage1_dr/panel_id=*/method=*/k=*/seed=42/scores.npy (no DR refit)",
        "parameters": {
            "knife_edge_band": KNIFE_EDGE, "kkt_violation_margin": KKT_VIOLATION, "ifna_panel": IFNA_PANEL,
            "ifna_methods": list(IFNA_METHODS), "ifna_k": IFNA_K, "sweep_multipliers": list(SWEEP_MULTIPLIERS),
            "tune_penalties": [[p, r] for p, r in TUNE_PENALTIES], "selection_rule": "D1: max cell_weighted_auprc, ties AUROC, EN>L1>L2, smaller C",
        },
        "c_policy": {
            "audit": rc.c_policy_record(c_policy="global_selected", training_n_changed=False, notes="stored selected C, no refit"),
            "ifna": rc.c_policy_record(c_policy="per_fold_tuned", training_n_changed=False,
                                       penalties=["l1", "l2", "elasticnet(0.5)"], notes="plus L1 sweep and L2 at the stored C"),
        },
        "inputs": input_rows,
        **extra,
    }
    previous = read_json_any(ctx.tables / f"{SLUG}_manifest.json")
    history = list(previous.get("run_history", [])) if previous else []
    if previous:
        history.append({k: previous.get(k) for k in ("created_utc", "command", "git_head", "git_dirty", "parts_runtime_seconds", "total_runtime_seconds")})
    manifest["run_history"] = history
    atomic_write_json(manifest, ctx.tables / f"{SLUG}_manifest.json")
    ledger = []
    for root, kind in ((ctx.tables, "table"), (ctx.figures, "figure")):
        for p in sorted(root.glob("*")):
            if p.is_file() and not p.name.endswith(".tmp") and p.name != f"{SLUG}_artifact_ledger.csv":
                rows = None
                if p.suffix == ".csv":
                    try:
                        rows = int(sum(1 for _ in open(p)) - 1)
                    except Exception:
                        rows = None
                this_kind = kind if p.suffix in (".csv", ".png", ".pdf") else "doc"
                ledger.append({"path": ctx.rel(p), "kind": this_kind, "rows": rows, "bytes": p.stat().st_size, "sha256": file_sha256(p)})
    unit_files = sorted((ctx.cache).rglob("*"))
    ledger.append({"path": ctx.rel(ctx.cache), "kind": "cache_dir", "rows": None,
                   "bytes": int(sum(p.stat().st_size for p in unit_files if p.is_file())), "sha256": ""})
    ledger_df = pd.DataFrame(ledger)
    ledger_df["rows"] = ledger_df["rows"].astype("Int64")
    atomic_write_csv(ledger_df, ctx.tables / f"{SLUG}_artifact_ledger.csv")


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", default=",".join(DEFAULT_PARTS), help=f"comma list from {','.join(ALL_PARTS)}")
    p.add_argument("--resume", action="store_true", help="skip finished (representation, fold) and ifna units")
    p.add_argument("--n-jobs", type=int, default=MAX_WORKERS, help=f"worker processes (capped at {MAX_WORKERS})")
    p.add_argument("--out-root", default=None, help="write under this dir instead of EXP/analysis (required by --smoke and --max-cells)")
    p.add_argument("--smoke", action="store_true", help="2 folds, coarse C grid, 2 penalties, 2 sweep points; needs --out-root")
    p.add_argument("--max-cells", type=int, default=None, help="audit only the first N cells (smoke); needs --out-root")
    p.add_argument("--c-grid-stride", type=int, default=4, help="with --smoke: keep every Nth C of the grid")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    parts = [x.strip() for x in args.parts.split(",") if x.strip()]
    unknown = sorted(set(parts) - set(ALL_PARTS))
    if unknown:
        raise SystemExit(f"unknown parts: {unknown}")
    started = time.perf_counter()
    ctx = Ctx(args)
    print(f"[d19] EXP={ctx.exp.name} out={ctx.out_root} supported={ctx.supported} n_jobs={ctx.n_jobs}", flush=True)
    cache = load_coef_cache(ctx) if set(parts) & {"anchor", "audit", "ifna"} else None
    anchor = selection = audit = units = None
    extra: Dict[str, Any] = {}

    def timed(name, fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        ctx.timings[name] = round(time.perf_counter() - t0, 2)
        return out

    if "anchor" in parts:
        anchor = timed("anchor", run_anchor, ctx, cache)
    if "audit" in parts:
        selection = timed("selection", atlas_selection, ctx)
        atomic_write_csv(selection, ctx.tables / "d19_atlas_selected_regularization.csv")
        lb = (selection.groupby(["atlas_row_block", "configuration", "penalty"], dropna=False)
              .agg(n_cells=("representation_id", "size"), n_at_C_lower_bound=("C_at_grid_lower_bound", "sum"),
                   n_at_C_upper_bound=("C_at_grid_upper_bound", "sum")).reset_index())
        atomic_write_csv(lb, ctx.tables / "d19_c_grid_bound_by_cell_group.csv")
        audit = timed("audit", run_audit, ctx, selection, cache)
        atomic_write_csv(audit_summary(audit), ctx.tables / "d19_entry_margin_audit_summary.csv")
    if "ifna" in parts:
        units = timed("ifna", run_ifna, ctx, cache)
    if "cuml_check" in parts:
        timed("cuml_check", run_cuml_check, ctx)
    if "report" in parts:
        t0 = time.perf_counter()
        if anchor is None and (ctx.tables / "d19_anchor_ifna_l1_entry_check.csv").is_file():
            anchor = pd.read_csv(ctx.tables / "d19_anchor_ifna_l1_entry_check.csv")
        if selection is None and (ctx.tables / "d19_atlas_selected_regularization.csv").is_file():
            selection = pd.read_csv(ctx.tables / "d19_atlas_selected_regularization.csv")
        if audit is None and (ctx.tables / "d19_entry_margin_audit_by_fold.csv").is_file():
            audit = pd.read_csv(ctx.tables / "d19_entry_margin_audit_by_fold.csv")
        if units is None and (ctx.tables / "d19_ifna_units.csv").is_file():
            units = pd.read_csv(ctx.tables / "d19_ifna_units.csv")
        audit_sum = audit_summary(audit) if audit is not None and not audit.empty else None
        wide = arm_sum = None
        if units is not None and not units.empty:
            wide, arm_sum = ifna_arm_tables(ctx, units)
            atomic_write_csv(wide, ctx.tables / "d19_ifna_pca40_vs_fa40_by_patient.csv")
            atomic_write_csv(arm_sum, ctx.tables / "d19_ifna_arm_summary.csv")
        make_figures(ctx, audit, units, wide)
        write_findings(ctx, anchor, selection, audit_sum, audit, wide, arm_sum, units)
        ctx.timings["report"] = round(time.perf_counter() - t0, 2)
        write_manifest_and_ledger(ctx, started, extra)
    print(f"[d19] done in {time.perf_counter() - started:.1f} s: {ctx.timings}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
