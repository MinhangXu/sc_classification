"""D16 runner: factor x patient allocation matrix on stored Stage 1 scores (W1.1).

Reads stored transductive Stage 1 scores (stage1_dr/panel_id=*/method={pca,fa}/k=*/seed=42)
and the per-cell metadata (patient, CN.label), computes the per factor x patient table of
factor_patient_matrix.py for each panel x method x K unit, and writes:

- EXP/analysis/cache/stage2_diagnostic/factor_patient_matrix_v1/parts/  one csv.gz per unit
- EXP/analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1/   merged tables,
  regression checks, findings, manifest and artifact ledger
- EXP/analysis/figures/stage2_diagnostic/factor_patient_matrix_v1/      D16A-D16D

Nothing is refit. CPU only; the 41-panel x {pca, fa} x K{5,10,20,40} grid takes minutes.

Usage (repo root):
  python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py \
      --panels pilot                       # IFN-alpha, allograft, apoptosis, TNFa/NF-kB
  ... --panels all --resume --workers 8    # all 41 biological panels; skips finished units
  ... --merge-only                         # rebuild tables/figures/findings from parts

Resume contract: a unit is finished when its part file reads back with the expected
rows and columns; empty or corrupt parts are recomputed. A merge failure writes
merge_failed.json in the cache dir; fix the cause and rerun with --merge-only.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, git_state, study_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factor_patient_matrix as fpm  # noqa: E402

EXP_DEFAULT = experiment_dir(must_exist=False)

ANALYSIS_ID = "D16"
SLUG = "factor_patient_matrix_v1"
PREFIX = "factor_patient_matrix"
SEED = 42
METHODS = ("pca", "fa")
KS = (5, 10, 20, 40)
FOCUS_PANEL = "single_geneset__hallmark_interferon_alpha_response"
PILOT_PANELS = (
    FOCUS_PANEL,
    "single_geneset__hallmark_allograft_rejection",
    "single_geneset__hallmark_apoptosis",
    "single_geneset__hallmark_tnfa_signaling_via_nfkb",
)
FOCUS_FACTORS = {"pca": (8,), "fa": (7, 8)}  # PC8 ~ FA7 shared IFN-alpha axis; FA8 for context
N_CELLS = 60_118
MIN_MALIGNANT_SUPPORT = 10

ATLAS_RELPATH = "analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv"
F6_RELPATH = "analysis/scorecards/stage2_diagnostic/feedback_deep_dive_20260917/f6_ifna_univariate_factor_ap_by_patient.csv"
CELL_METADATA_DEFAULT = "stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz"
COHORT_CACHE_DEFAULT = "analysis/cache/stage2_diagnostic/deg_hvg_lowsupport_v1/mrd_cite_binary_log1p_15560.h5ad"
D14_FOCUS_RELPATH = "analysis/scorecards/stage2_diagnostic/patient_factor_overlap_v1/focus_factor_overlap.csv"
ANCHOR_RELPATH = "scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/anchors/factor_by_patient_scratch_20261008.py"

# Printed output of the factor-by-patient scratch anchor (2026-10-08), IFN-alpha K40,
# all 13 patients, cohort-centered SS, values rounded to 2 decimals by the anchor.
ANCHOR = {
    "pca": {
        "degenerate": [],
        "top_counts": {"P02": 31, "P07": 7, "P01": 2},
        "median_enrichment": {"P01": 0.95, "P02": 1.53, "P03": 1.06, "P04": 0.94, "P05": 0.9, "P06": 0.84,
                              "P07": 1.09, "P08": 1.04, "P09": 1.03, "P10": 1.06, "P11": 0.92, "P12": 0.57,
                              "P13": 1.2},
        "enrichment": {8: {"P01": 0.81, "P02": 1.15, "P03": 1.1, "P04": 1.06, "P05": 0.94, "P06": 0.78,
                           "P07": 1.64, "P08": 0.9, "P09": 1.09, "P10": 1.01, "P11": 1.14, "P12": 0.78,
                           "P13": 0.98}},
        "shift_z": {8: {"P01": -0.22, "P02": -0.78, "P03": -0.6, "P04": -0.89, "P06": -0.05, "P07": -0.76,
                        "P09": -0.88}},
    },
    "fa": {
        "degenerate": [37, 38, 39, 40],
        "top_counts": {"P02": 27, "P07": 7, "P01": 2},
        "median_enrichment": {"P01": 0.98, "P02": 1.48, "P03": 1.04, "P04": 0.87, "P05": 0.85, "P06": 0.82,
                              "P07": 1.21, "P08": 1.03, "P09": 1.02, "P10": 1.08, "P11": 0.94, "P12": 0.59,
                              "P13": 1.14},
        "enrichment": {
            7: {"P01": 0.92, "P02": 1.19, "P03": 1.10, "P04": 0.85, "P05": 0.95, "P06": 0.76, "P07": 1.34,
                "P08": 0.87, "P09": 1.03, "P10": 1.15, "P11": 1.16, "P12": 0.79, "P13": 1.01},
            8: {"P01": 1.17, "P02": 1.12, "P03": 0.94, "P04": 0.93, "P05": 0.71, "P06": 0.75, "P07": 2.15,
                "P08": 0.95, "P09": 0.94, "P10": 1.14, "P11": 1.00, "P12": 0.69, "P13": 0.84},
        },
        "shift_z": {
            7: {"P01": 0.27, "P02": 0.78, "P03": 0.62, "P04": 0.89, "P06": 0.09, "P07": 1.33, "P09": 0.77},
            8: {"P01": 0.02, "P02": -0.45, "P03": -0.25, "P04": -0.41, "P06": 0.05, "P07": 0.15, "P09": -0.30},
        },
    },
}
ROUND_TOL = 0.005 + 1e-9  # anchor values are rounded to 2 decimals
F6_TOL = 1e-9
D14_TOL = 1e-6  # D14 summed float32 scores

_COHORT: fpm.Cohort | None = None


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------


def out_dirs(exp: Path) -> dict:
    base = exp / "analysis"
    return {
        "tables": base / "scorecards" / "stage2_diagnostic" / SLUG,
        "figures": base / "figures" / "stage2_diagnostic" / SLUG,
        "cache": base / "cache" / "stage2_diagnostic" / SLUG,
        "parts": base / "cache" / "stage2_diagnostic" / SLUG / "parts",
    }


def score_dir(exp: Path, panel: str, method: str, k: int) -> Path:
    return exp / "stage1_dr" / f"panel_id={panel}" / f"method={method}" / f"k={int(k)}" / f"seed={SEED}"


def rel(exp: Path, path: Path) -> str:
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(exp))).as_posix()


def exp_relpath(key: str, default: str) -> str:
    """EXP-relative path of a study input, from study_config.json (built-in fallback)."""
    return str(study_config().get(key) or default)


def load_cohort(exp: Path) -> tuple:
    """Cell metadata (patient, CN.label) in Stage 1 row order, plus a provenance dict."""
    meta_path = exp / exp_relpath("cell_metadata_relpath", CELL_METADATA_DEFAULT)
    if not meta_path.is_file():
        raise FileNotFoundError(f"cell metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path, usecols=["cell_id", "patient", "CN.label"])
    if len(meta) != N_CELLS:
        raise AssertionError(f"{meta_path} has {len(meta)} rows, expected {N_CELLS}")
    malignant = meta["CN.label"].astype(str).eq("cancer").to_numpy()
    patient = meta["patient"].astype(str).to_numpy()
    supported = fpm.supported_patients(patient, malignant, min_malignant=MIN_MALIGNANT_SUPPORT)
    cfg_supported = study_config().get("supported_patients")
    if cfg_supported and tuple(sorted(cfg_supported)) != supported:
        raise AssertionError(f"support rule gives {supported}, study_config says {cfg_supported}")
    provenance = {
        "cell_metadata": rel(exp, meta_path),
        "supported_patients": list(supported),
        "supported_source": "study_config.json (checked against the support rule)" if cfg_supported
        else "support rule on cell metadata",
        "cohort_cache_order_check": verify_cache_order(exp, meta),
    }
    return fpm.Cohort(patient, malignant, supported=supported), provenance


def verify_cache_order(exp: Path, meta: pd.DataFrame) -> str:
    """Check that the metadata rows follow the cohort cache obs order (D14's source)."""
    try:
        import anndata as ad
    except ImportError:
        return "skipped (anndata not importable)"
    path = exp / exp_relpath("cohort_cache_relpath", COHORT_CACHE_DEFAULT)
    if not path.is_file():
        return "skipped (cohort cache absent)"
    adata = ad.read_h5ad(path, backed="r")
    try:
        obs = adata.obs
        same = (
            adata.n_obs == len(meta)
            and bool((obs.index.astype(str).to_numpy() == meta["cell_id"].astype(str).to_numpy()).all())
            and bool((obs["patient"].astype(str).to_numpy() == meta["patient"].astype(str).to_numpy()).all())
            and bool((obs["CN.label"].astype(str).to_numpy() == meta["CN.label"].astype(str).to_numpy()).all())
        )
    finally:
        adata.file.close()
    if not same:
        raise AssertionError("cell metadata order differs from the cohort cache obs order")
    return "pass (" + rel(exp, path) + ": cell_id, patient and CN.label identical)"


def biological_panels(exp: Path) -> pd.DataFrame:
    atlas = pd.read_csv(exp / ATLAS_RELPATH, usecols=["stage0_panel_id", "short_panel_label", "atlas_row_block", "n_covered_genes"])
    bio = atlas.loc[atlas["atlas_row_block"].eq("biological")].drop_duplicates("stage0_panel_id")
    return bio.sort_values("short_panel_label").reset_index(drop=True)


def plan_units(exp: Path, panels: list, methods: list, ks: list) -> tuple:
    labels = biological_panels(exp).set_index("stage0_panel_id")["short_panel_label"].to_dict()
    units, missing = [], []
    for panel in panels:
        for method in methods:
            for k in ks:
                d = score_dir(exp, panel, method, k)
                unit = {"panel": panel, "label": labels.get(panel, panel), "method": method, "k": int(k)}
                if (d / "scores.npy").is_file() and (d / "metadata.json").is_file():
                    units.append(unit)
                else:
                    missing.append(unit)
    return units, missing


def part_path(parts: Path, unit: dict) -> Path:
    return parts / f"{unit['panel']}__{unit['method']}__k{unit['k']}.csv.gz"


def read_part(path: Path, n_patients: int) -> pd.DataFrame | None:
    """Return the part when it is complete, else None (empty/corrupt parts are unfinished)."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    need = {"stage0_panel_id", "stage1_method", "k", "factor_index", "patient", "enrichment", "unit_seconds"}
    if not need.issubset(frame.columns) or frame.empty:
        return None
    k = int(frame["k"].iloc[0])
    if len(frame) != k * n_patients or frame["factor_index"].nunique() != k:
        return None
    return frame


# --------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------


def _init_worker(cohort: fpm.Cohort) -> None:
    global _COHORT
    _COHORT = cohort


def compute_unit(exp_str: str, unit: dict, parts_str: str) -> dict:
    exp, parts = Path(exp_str), Path(parts_str)
    t0 = time.time()
    d = score_dir(exp, unit["panel"], unit["method"], unit["k"])
    meta = json.loads((d / "metadata.json").read_text())
    scores = np.load(d / "scores.npy")
    if scores.shape[0] != _COHORT.n_cells:
        raise AssertionError(f"{d} has {scores.shape[0]} rows, metadata has {_COHORT.n_cells}")
    table = fpm.factor_patient_table(scores, _COHORT)
    table.insert(0, "stage0_panel_id", unit["panel"])
    table.insert(1, "short_panel_label", unit["label"])
    table.insert(2, "stage1_method", unit["method"])
    table.insert(3, "k", unit["k"])
    table.insert(4, "effective_k", int(meta.get("effective_k", scores.shape[1])))
    table.insert(5, "n_genes", int(meta.get("n_genes", -1)))
    table.insert(6, "scores_relpath", rel(exp, d / "scores.npy"))
    table["unit_seconds"] = round(time.time() - t0, 3)
    out = part_path(parts, unit)
    tmp = out.with_name(out.name + ".tmp")
    table.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, out)
    return {"unit": unit, "seconds": time.time() - t0, "rows": len(table)}


def run_units(exp: Path, units: list, parts: Path, cohort: fpm.Cohort, workers: int) -> list:
    done = []
    if workers <= 1:
        _init_worker(cohort)
        for i, unit in enumerate(units, 1):
            res = compute_unit(str(exp), unit, str(parts))
            done.append(res)
            print(f"[{i}/{len(units)}] {unit['label']} {unit['method']} K{unit['k']}: {res['seconds']:.1f}s", flush=True)
        return done
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cohort,)) as pool:
        futures = {pool.submit(compute_unit, str(exp), u, str(parts)): u for u in units}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            done.append(res)
            u = res["unit"]
            print(f"[{i}/{len(units)}] {u['label']} {u['method']} K{u['k']}: {res['seconds']:.1f}s", flush=True)
    return done


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


def regression_checks(exp: Path, long: pd.DataFrame) -> pd.DataFrame:
    rows: list = []
    focus = long.loc[long["stage0_panel_id"].eq(FOCUS_PANEL) & long["k"].eq(40)]
    for method in METHODS:
        tab = focus.loc[focus["stage1_method"].eq(method)]
        if tab.empty:
            continue
        a = ANCHOR[method]
        nd = tab.loc[~tab["degenerate"]]
        degenerate = sorted(tab.loc[tab["degenerate"], "factor_index"].unique().tolist())
        _check(rows, f"anchor_{method}40_degenerate", ANCHOR_RELPATH, "degenerate factor indices",
               [0.0 if degenerate == a["degenerate"] else 1.0], 0.0, f"got {degenerate}")
        top = nd.loc[nd.groupby("factor_index")["enrichment"].idxmax()]
        counts = top["patient"].value_counts().to_dict()
        _check(rows, f"anchor_{method}40_top_counts", ANCHOR_RELPATH, "top-enrichment patient counts",
               [0.0 if {k: int(v) for k, v in counts.items()} == a["top_counts"] else 1.0], 0.0,
               ";".join(f"{k}:{v}" for k, v in sorted(counts.items())))
        med = nd.groupby("patient")["enrichment"].median()
        _check(rows, f"anchor_{method}40_median_enrichment", ANCHOR_RELPATH, "median enrichment by patient",
               [abs(med[p] - v) for p, v in a["median_enrichment"].items()], ROUND_TOL)
        for f, vals in a["enrichment"].items():
            sub = nd.loc[nd["factor_index"].eq(f)].set_index("patient")["enrichment"]
            _check(rows, f"anchor_{method}40_f{f}_enrichment", ANCHOR_RELPATH, f"factor {f} enrichment",
                   [abs(sub[p] - v) for p, v in vals.items()], ROUND_TOL)
        for f, vals in a["shift_z"].items():
            sub = nd.loc[nd["factor_index"].eq(f)].set_index("patient")["shift_z_cohort"]
            _check(rows, f"anchor_{method}40_f{f}_shift_z", ANCHOR_RELPATH, f"factor {f} malignant-normal shift (cohort z)",
                   [abs(sub[p] - v) for p, v in vals.items()], ROUND_TOL)

    f6_path = exp / F6_RELPATH
    if f6_path.is_file() and not focus.empty:
        f6 = pd.read_csv(f6_path)
        u = focus.loc[focus["supported"]].copy()
        u["feature_id"] = u["factor_index"].map(lambda i: f"factor_{int(i):03d}")
        m = u.merge(f6, left_on=["stage1_method", "patient", "feature_id"],
                    right_on=["stage1_method", "heldout_patient", "feature_id"], how="inner")
        for ours, theirs, metric in (
            ("train_ap_sign_agnostic", "train_univariate_auprc", "train-fold univariate AP (sign-agnostic)"),
            ("heldout_ap_sign_agnostic", "heldout_univariate_auprc", "held-out univariate AP (sign-agnostic)"),
            ("heldout_lift_sign_agnostic", "heldout_univariate_lift", "held-out univariate lift (sign-agnostic)"),
            ("heldout_prevalence", "patient_prevalence", "held-out prevalence"),
        ):
            _check(rows, f"f6_{ours}", F6_RELPATH, metric, (m[ours] - m[theirs]).abs(), F6_TOL,
                   f"{len(m)} of {len(f6)} F6 rows matched (pca, fa x 7 patients x 40 factors)")
    d14_path = exp / D14_FOCUS_RELPATH
    if d14_path.is_file() and not focus.empty:
        d14 = pd.read_csv(d14_path)
        d14 = d14.loc[d14["stage0_panel_id"].eq(FOCUS_PANEL) & d14["effective_k"].eq(40)]
        fe = focus.groupby(["stage1_method", "factor_index"])[["eta2_normal", "eta2_malignant", "degenerate"]].first().reset_index()
        for stratum, col in (("normal", "eta2_normal"), ("cancer", "eta2_malignant")):
            d = d14.loc[d14["stratum"].eq(stratum), ["stage1_method", "factor_index", "eta2"]]
            m = fe.merge(d, on=["stage1_method", "factor_index"])
            m = m.loc[~m["degenerate"]]
            _check(rows, f"d14_eta2_{stratum}", D14_FOCUS_RELPATH, f"eta2 by patient within {stratum} cells",
                   (m[col] - m["eta2"]).abs(), D14_TOL, "non-degenerate factors; D14 summed float32 scores")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

BLUE, MID, RED = "#2a78d6", "#f0efec", "#e34948"
DEGEN_GREY = "#bdbcb8"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    div = LinearSegmentedColormap.from_list("d16_div", [BLUE, MID, RED])
    div.set_bad(DEGEN_GREY)
    seq = LinearSegmentedColormap.from_list("d16_seq", ["#fcfcfb", "#86b6ef", "#256abf", "#0d366b"])
    seq.set_bad(DEGEN_GREY)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.edgecolor": "#52514e"})
    return plt, div, seq


def save(fig, figdir: Path, stem: str, written: list) -> None:
    for ext in ("png", "pdf"):
        path = figdir / f"{stem}.{ext}"
        fig.savefig(path, dpi=180 if ext == "png" else None, bbox_inches="tight")
        written.append(path)


def _annotate(ax, values, fmt, *, fontsize=4.5, threshold=None):
    for (i, j), v in np.ndenumerate(values):
        if not np.isfinite(v):
            continue
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=fontsize, color="#0b0b0b")


def fig_d16a(long: pd.DataFrame, figdir: Path, patients: list, supported: tuple, written: list) -> None:
    plt, div, _ = _plt()
    focus = long.loc[long["stage0_panel_id"].eq(FOCUS_PANEL) & long["k"].eq(40)]
    if focus.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 9.5), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        tab = focus.loc[focus["stage1_method"].eq(method)]
        mat = tab.pivot(index="factor_index", columns="patient", values="enrichment").reindex(columns=patients)
        vals = np.log2(mat.to_numpy(dtype=float))
        im = ax.imshow(np.clip(vals, -2, 2), cmap=div, vmin=-2, vmax=2, aspect="auto")
        _annotate(ax, mat.to_numpy(dtype=float), lambda v: f"{v:.2f}")
        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels([p + ("*" if p in supported else "") for p in patients], rotation=0)
        for lab in ax.get_xticklabels():
            if lab.get_text().rstrip("*") in fpm.HIGH_PREVALENCE:
                lab.set_fontweight("bold")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([f"{'PC' if method == 'pca' else 'FA'}{i}" for i in mat.index])
        for f in FOCUS_FACTORS[method]:
            r = list(mat.index).index(f)
            ax.add_patch(plt.Rectangle((-0.5, r - 0.5), len(patients), 1, fill=False, lw=1.4, ec="#0b0b0b"))
        n_deg = int(tab.loc[tab["degenerate"], "factor_index"].nunique())
        ax.set_title(f"IFN-alpha {method.upper()} K=40: SS share / cell share per factor x patient"
                     + (f"\n(grey rows: {n_deg} degenerate factors, SD < 1e-8)" if n_deg else "\n "))
        ax.set_xlabel("patient (* supported; bold = highest-prevalence P02/P03/P04)")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
    cb.set_label("enrichment (log2 colour scale; cell text = enrichment)")
    fig.suptitle("D16A. Score-space allocation, not loading leverage: transductive Stage 1 scores, cohort-centered SS "
                 "(boxed: PC8, FA7, FA8)", fontsize=10)
    save(fig, figdir, "D16A_ifna_pca40_fa40_enrichment_heatmap", written)
    plt.close(fig)


def fig_d16b(long: pd.DataFrame, figdir: Path, supported: tuple, written: list) -> None:
    plt, div, _ = _plt()
    focus = long.loc[long["stage0_panel_id"].eq(FOCUS_PANEL) & long["k"].eq(40) & long["supported"]]
    if focus.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 9.5), constrained_layout=True)
    lim = 25.0
    for ax, method in zip(axes, METHODS):
        tab = focus.loc[focus["stage1_method"].eq(method)]
        mat = 100 * tab.pivot(index="factor_index", columns="patient", values="heldout_lift_oriented").reindex(columns=list(supported))
        deg = tab.groupby("factor_index")["degenerate"].first().reindex(mat.index).to_numpy()
        show = mat.to_numpy(dtype=float).copy()
        show[deg] = np.nan
        im = ax.imshow(np.clip(show, -lim, lim), cmap=div, vmin=-lim, vmax=lim, aspect="auto")
        _annotate(ax, show, lambda v: f"{v:.1f}", fontsize=5)
        ax.set_xticks(range(len(supported)))
        ax.set_xticklabels(list(supported))
        for lab in ax.get_xticklabels():
            if lab.get_text() in fpm.HIGH_PREVALENCE:
                lab.set_fontweight("bold")
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([f"{'PC' if method == 'pca' else 'FA'}{i}" for i in mat.index])
        for f in FOCUS_FACTORS[method]:
            r = list(mat.index).index(f)
            ax.add_patch(plt.Rectangle((-0.5, r - 0.5), len(supported), 1, fill=False, lw=1.4, ec="#0b0b0b"))
        ax.set_title(f"IFN-alpha {method.upper()} K=40: held-out univariate lift (pp)")
        ax.set_xlabel("held-out supported patient (LOPO label transfer)")
    cb = fig.colorbar(im, ax=axes, shrink=0.5)
    cb.set_label("AP - prevalence, pp (sign fixed on training folds; clipped at +/-25)")
    fig.suptitle("D16B. One factor at a time, sign from the training-fold class-mean gap (grey: degenerate)", fontsize=10)
    save(fig, figdir, "D16B_ifna_pca40_fa40_univariate_lift_heatmap", written)
    plt.close(fig)


def fig_d16c(freq: pd.DataFrame, figdir: Path, patients: list, written: list) -> None:
    plt, _, seq = _plt()
    if freq.empty:
        return
    freq = freq.assign(row=freq["stage1_method"].str.upper() + " K=" + freq["k"].astype(str))
    order = [f"{m.upper()} K={k}" for m in METHODS for k in KS if f"{m.upper()} K={k}" in set(freq["row"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for ax, col, title in (
        (axes[0], "frac_factors_top", "all cells"),
        (axes[1], "frac_factors_top_normal_only", "normal cells only"),
    ):
        mat = 100 * freq.pivot(index="row", columns="patient", values=col).reindex(index=order, columns=patients)
        im = ax.imshow(mat.to_numpy(dtype=float), cmap=seq, vmin=0, vmax=max(50.0, float(np.nanmax(mat.to_numpy()))), aspect="auto")
        _annotate(ax, mat.to_numpy(dtype=float), lambda v: f"{v:.0f}", fontsize=6)
        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels(patients)
        for lab in ax.get_xticklabels():
            if lab.get_text() in fpm.HIGH_PREVALENCE:
                lab.set_fontweight("bold")
        nf = freq.groupby("row")["n_nondegenerate_factors"].first().reindex(order)
        nr = freq.groupby("row")["n_representations"].first().reindex(order)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{r}  ({int(nr[r])} panels, {int(nf[r])} factors)" for r in order])
        ax.set_title(f"% of non-degenerate factors whose top-enrichment patient is ... ({title})")
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("% of factors")
    fig.suptitle("D16C. Which patient tops SS enrichment, pooled over biological panels (bold = highest-prevalence)", fontsize=10)
    save(fig, figdir, "D16C_top_patient_frequency", written)
    plt.close(fig)


def fig_d16d(rsum: pd.DataFrame, figdir: Path, written: list) -> None:
    plt, div, _ = _plt()
    if rsum.empty:
        return
    r = rsum.assign(col=rsum["stage1_method"].str.upper() + " K=" + rsum["k"].astype(str))
    cols = [f"{m.upper()} K={k}" for m in METHODS for k in KS if f"{m.upper()} K={k}" in set(r["col"])]
    rows = sorted(r["short_panel_label"].unique(), key=lambda s: (s != "interferon alpha response", s))
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.0, 0.24 * len(rows) + 1.6)), constrained_layout=True)
    cs_all = float(r["hp_cell_share"].median())
    cs_norm = float(r["hp_cell_share_normal"].median())
    for ax, col, title in (
        (axes[0], "hp_enrichment_weighted", f"all cells (P02+P03+P04 = {100 * cs_all:.1f}% of cells)"),
        (axes[1], "hp_enrichment_normal_weighted", f"normal cells only ({100 * cs_norm:.1f}% of normal cells)"),
    ):
        mat = r.pivot(index="short_panel_label", columns="col", values=col).reindex(index=rows, columns=cols)
        vals = mat.to_numpy(dtype=float)
        im = ax.imshow(np.clip(np.log2(vals), -1, 1), cmap=div, vmin=-1, vmax=1, aspect="auto")
        _annotate(ax, vals, lambda v: f"{v:.2f}", fontsize=5)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows if ax is axes[0] else [])
        ax.set_title(f"combined SS share / cell share, {title}")
    cb = fig.colorbar(im, ax=axes, shrink=0.4, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["0.5x", "1x", "2x"])
    cb.set_label("SS-weighted over non-degenerate factors (log2 colour; text = ratio)")
    fig.suptitle("D16D. Highest-prevalence patients' share of Stage 1 score energy, by panel x method x K "
                 "(score space; leverage on loadings needs D17)", fontsize=10)
    save(fig, figdir, "D16D_hp_combined_share_by_panel", written)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# findings, manifest, ledger
# --------------------------------------------------------------------------------------


def _fmt_counts(counts: dict) -> str:
    return ", ".join(f"{p} {c}" for p, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def write_findings(path: Path, long: pd.DataFrame, fsum: pd.DataFrame, rsum: pd.DataFrame,
                   freq: pd.DataFrame, checks: pd.DataFrame, missing: list, ctx: dict) -> None:
    L: list = []
    add = L.append
    n_rep = len(rsum)
    n_panels = rsum["stage0_panel_id"].nunique()
    add(f"# D16 factor x patient allocation matrix ({SLUG})")
    add("")
    add(f"- Analysis: D16 (W1.1). Generated {ctx['created_utc']} by "
        "scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py "
        f"at git {ctx['git_head']}{' (dirty tree)' if ctx['git_dirty'] else ''}.")
    add(f"- Coverage: {n_rep} representations = {n_panels} biological panels x "
        f"{{{', '.join(sorted(rsum['stage1_method'].unique()))}}} x K in "
        f"{{{', '.join(str(k) for k in sorted(rsum['k'].unique()))}}}, seed 42, stored transductive Stage 1 scores. "
        f"{len(missing)} requested unit(s) have no stored scores (panel smaller than K).")
    add("- Claim level: descriptive. Stage 1 is transductive; the held-out lifts are LOPO label transfer "
        "on that basis, not inductive deployment.")
    add("")
    add("## Questions")
    add("- 'Need to inspect the factor-by-patient matrix for FA' (A-20261008-05).")
    add("- 'Patients with more malignant cells (P02, P03, P04) disproportionately shape the factors' "
        "(A-20261008-02). P02/P03/P04 are the highest-PREVALENCE patients, not the ones with the most malignant cells.")
    add("")
    add("## What was computed")
    add("Per panel x method x K x factor x patient (13 patients; 7 supported):")
    add("- cell share; SS share with SS taken about the cohort mean of the factor; enrichment = SS share / cell share, "
        "split exactly into location (squared standardized patient mean) + spread (squared patient/cohort SD ratio);")
    add("- the same quantities within normal cells and within malignant cells (still about the cohort mean);")
    add("- within-patient malignant - normal mean shift in cohort-SD units, and Cohen's d with the within-patient pooled SD;")
    add("- eta2 of patient within normal cells (patients with >= 200 normal cells) and within malignant cells "
        "(>= 20 malignant cells; D14 strata, not the support rule), and each patient's additive part of it;")
    add("- univariate held-out AP and lift (AP - held-out prevalence) for each supported patient, sign fixed by the "
        "training-fold class-mean gap; the sign-agnostic max(AP(+s), AP(-s)) of F6 is kept as a separate column;")
    add("- degenerate flag (cohort SD < 1e-8) and effective_k from the Stage 1 metadata.")
    add("No refit and no Stage 2 model: c_policy not applicable.")
    add("")
    add("## Results")
    focus = rsum.loc[rsum["stage0_panel_id"].eq(FOCUS_PANEL)]
    ffocus = fsum.loc[fsum["stage0_panel_id"].eq(FOCUS_PANEL) & ~fsum["degenerate"]]
    lfocus = long.loc[long["stage0_panel_id"].eq(FOCUS_PANEL) & long["k"].eq(40)]
    if not focus.empty:
        add("### IFN-alpha (highlighted)")
        for method in METHODS:
            row = focus.loc[focus["stage1_method"].eq(method) & focus["k"].eq(40)]
            if row.empty:
                continue
            row = row.iloc[0]
            sub = ffocus.loc[ffocus["stage1_method"].eq(method) & ffocus["k"].eq(40)]
            counts = sub["top_patient"].value_counts().to_dict()
            ncounts = sub["top_normal_patient"].value_counts().to_dict()
            add(f"- {method.upper()}40: {int(row['n_nondegenerate'])} non-degenerate factors "
                f"({int(row['n_degenerate'])} degenerate). Top-enrichment patient: {_fmt_counts(counts)}. "
                f"Normal cells only: {_fmt_counts(ncounts)}. P03 tops {int(counts.get('P03', 0))} factor(s).")
            add(f"  - P02+P03+P04 = {100 * row['hp_cell_share']:.1f}% of cells and "
                f"{100 * row['hp_ss_share_weighted']:.1f}% of the SS-weighted score energy "
                f"(ratio {row['hp_enrichment_weighted']:.2f}); normal cells only: "
                f"{100 * row['hp_cell_share_normal']:.1f}% of cells, {100 * row['hp_ss_share_normal_weighted']:.1f}% "
                f"of SS (ratio {row['hp_enrichment_normal_weighted']:.2f}). Malignant cells hold "
                f"{100 * row['malignant_ss_share_weighted']:.1f}% of the SS vs {100 * row['malignant_cell_share']:.1f}% of cells.")
            for f in FOCUS_FACTORS[method]:
                t = lfocus.loc[lfocus["stage1_method"].eq(method) & lfocus["factor_index"].eq(f)].set_index("patient")
                if t.empty or bool(t["degenerate"].iloc[0]):
                    continue
                name = f"{'PC' if method == 'pca' else 'FA'}{f}"
                enr = ", ".join(f"{p} {t.loc[p, 'enrichment']:.2f}" for p in ("P02", "P03", "P04", "P07"))
                lifts = ", ".join(
                    f"{p} {100 * t.loc[p, 'heldout_lift_oriented']:+.1f}" for p in ctx["supported"]
                )
                add(f"  - {name}: enrichment {enr}; eta2 normal {t['eta2_normal'].iloc[0]:.3f}, malignant "
                    f"{t['eta2_malignant'].iloc[0]:.3f}; oriented held-out lift (pp) {lifts}.")
        add("")
    add("### Cohort-wide (all panels in this run)")
    for method in METHODS:
        for k in KS:
            fr = freq.loc[freq["stage1_method"].eq(method) & freq["k"].eq(k)]
            if fr.empty:
                continue
            fr = fr.set_index("patient")
            topline = ", ".join(f"{p} {100 * fr.loc[p, 'frac_factors_top']:.0f}%" for p in ("P02", "P03", "P04", "P07"))
            modal = fr["n_representations_modal_top"]
            modal_txt = _fmt_counts({p: int(v) for p, v in modal.items() if v > 0})
            add(f"- {method.upper()} K={k} ({int(fr['n_representations'].iloc[0])} panels, "
                f"{int(fr['n_nondegenerate_factors'].iloc[0])} factors): factors topped by {topline}; "
                f"modal top patient per representation: {modal_txt}.")
    hp = rsum["hp_enrichment_weighted"]
    hpn = rsum["hp_enrichment_normal_weighted"]
    add(f"- P02+P03+P04 SS-weighted share / cell share over {n_rep} representations: median {hp.median():.2f} "
        f"(range {hp.min():.2f}-{hp.max():.2f}; > 1 in {int((hp > 1).sum())}/{n_rep}). Normal cells only: median "
        f"{hpn.median():.2f} (range {hpn.min():.2f}-{hpn.max():.2f}; > 1 in {int((hpn > 1).sum())}/{n_rep}).")
    by_m = rsum.groupby("stage1_method")["hp_enrichment_weighted"].median()
    add("- Median of that ratio by method: " + ", ".join(f"{m.upper()} {v:.2f}" for m, v in by_m.items()) + ".")
    add(f"- Malignant cells' SS-weighted share of score energy: median {100 * rsum['malignant_ss_share_weighted'].median():.1f}% "
        f"vs {100 * rsum['malignant_cell_share'].median():.1f}% of cells.")
    add("")
    add("## Regression checks")
    if checks.empty:
        add("- none run (IFN-alpha K40 not in this run)")
    else:
        add("| check | metric | n | max abs diff | tol | status |")
        add("|---|---|---|---|---|---|")
        for c in checks.itertuples(index=False):
            add(f"| {c.check_id} | {c.metric} | {c.n_compared} | {c.max_abs_diff:.3g} | {c.tolerance:g} | {c.status} |")
    add("")
    add("## Caveats (claim gate)")
    add("- Score-space allocation is not loading leverage. Enrichment says where a factor's score variance sits "
        "on the fixed transductive basis; it does not say that a patient's cells changed the loadings. Testing "
        "H-S1-05 needs D17's leave-out / reweighted Stage 1 refit (W2.1, W2.4).")
    add("- Enrichment is per-cell (mean squared deviation over the cohort variance). A small patient with high "
        "enrichment can still hold a small share of the total SS; the SS share column carries that.")
    add("- No adjustment for cell-type composition; a patient's enrichment can reflect its cell-type mix.")
    add("- Univariate held-out lift fixes the factor's sign on the training folds; the F6 sign-agnostic AP is "
        "optimistic on noise factors. Neither is a Stage 2 model.")
    add("- Descriptive, transductive, seven supported patients; no inferential test.")
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


def write_ledger(exp: Path, path: Path, artifacts: list, created: str) -> None:
    rows = []
    for a in artifacts:
        kind = "figure" if a.suffix in (".png", ".pdf") else ("findings" if a.suffix == ".md" else
                                                              ("manifest" if a.suffix == ".json" else "table"))
        n_rows = None
        if a.name.endswith((".csv", ".csv.gz")):
            n_rows = int(len(pd.read_csv(a, usecols=[0])))
        rows.append({"artifact_relpath": rel(exp, a), "kind": kind, "analysis_id": ANALYSIS_ID, "slug": SLUG,
                     "n_rows": n_rows, "bytes": a.stat().st_size, "sha256": sha256(a), "created_utc": created})
    pd.DataFrame(rows).to_csv(path, index=False)


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------


def merge(exp: Path, dirs: dict, cohort: fpm.Cohort, provenance: dict, missing: list, args, t_start: float,
          n_computed: int, figures: bool = True) -> dict:
    n_pat = len(cohort.patients)
    parts = sorted(dirs["parts"].glob("*.csv.gz"))
    frames, bad = [], []
    for p in parts:
        f = read_part(p, n_pat)
        (frames.append(f) if f is not None else bad.append(p.name))
    if bad:
        raise RuntimeError(f"incomplete part files (rerun with --resume): {bad}")
    if not frames:
        raise RuntimeError("no part files to merge")
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["stage0_panel_id", "stage1_method", "k", "factor_index", "patient"], kind="stable")
    t_merge = time.time()
    fsum = fpm.factor_summary(long)
    rps = fpm.representation_patient_summary(long)
    rsum = fpm.representation_summary(fsum, rps)
    freq = fpm.top_patient_frequency(fsum, rsum, list(cohort.patients))
    checks = regression_checks(exp, long)
    is_focus = pd.Series(False, index=long.index)
    for method, factors in FOCUS_FACTORS.items():
        is_focus |= long["stage1_method"].eq(method) & long["factor_index"].isin(factors)
    focus = long.loc[long["stage0_panel_id"].eq(FOCUS_PANEL) & long["k"].eq(40) & is_focus]

    td, fd = dirs["tables"], dirs["figures"]
    artifacts: list = []

    def wcsv(frame: pd.DataFrame, name: str) -> None:
        path = td / name
        frame.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        artifacts.append(path)

    wcsv(long.drop(columns=["unit_seconds"]), f"{PREFIX}_long.csv.gz")
    wcsv(fsum, f"{PREFIX}_factor_summary.csv")
    wcsv(rsum, f"{PREFIX}_representation_summary.csv")
    wcsv(rps, f"{PREFIX}_representation_patient_summary.csv")
    wcsv(freq, f"{PREFIX}_top_patient_frequency.csv")
    wcsv(focus.drop(columns=["unit_seconds"]), f"{PREFIX}_ifna_focus_factors.csv")
    wcsv(checks, f"{PREFIX}_regression_checks.csv")
    if figures:
        fig_d16a(long, fd, list(cohort.patients), cohort.supported, artifacts)
        fig_d16b(long, fd, cohort.supported, artifacts)
        fig_d16c(freq, fd, list(cohort.patients), artifacts)
        fig_d16d(rsum, fd, artifacts)

    head, dirty = git_state(REPO)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    findings = td / f"{PREFIX}_findings.md"
    manifest_path = td / f"{PREFIX}_manifest.json"
    ledger = td / f"{PREFIX}_artifact_ledger.csv"
    names = [rel(exp, a) for a in artifacts] + [rel(exp, p) for p in (findings, manifest_path, ledger)]
    ctx = {"created_utc": created, "git_head": head, "git_dirty": dirty, "supported": cohort.supported,
           "artifacts": names}
    write_findings(findings, long, fsum, rsum, freq, checks, missing, ctx)
    artifacts.append(findings)

    unit_seconds = long.groupby(["stage0_panel_id", "stage1_method", "k"])["unit_seconds"].first()
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "slug": SLUG,
        "workstream_task": "W1.1",
        "created_utc": created,
        "git_head": head,
        "git_dirty": dirty,
        "experiment_id": exp.name,
        "command": "python scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py "
                   + " ".join("<path>" if a.startswith(("/", "~")) else a for a in sys.argv[1:]),
        "inputs": {
            "stage1_scores": "stage1_dr/panel_id=<panel>/method={pca,fa}/k=<K>/seed=42/scores.npy (+ metadata.json)",
            "panel_list": ATLAS_RELPATH + " (atlas_row_block == biological)",
            **provenance,
            "regression_f6": F6_RELPATH,
            "regression_d14": D14_FOCUS_RELPATH,
            "regression_anchor": ANCHOR_RELPATH + " (repo-relative; printed values embedded in the runner)",
        },
        "parameters": {
            "methods": sorted(long["stage1_method"].unique().tolist()),
            "ks": sorted(int(k) for k in long["k"].unique()),
            "seed": SEED,
            "degenerate_sd": fpm.DEGENERATE_SD,
            "ss_center": "cohort mean of each factor (all 60,118 cells)",
            "eta2_strata": {"normal_min_cells": fpm.NORMAL_MIN_CELLS, "malignant_min_cells": fpm.MALIGNANT_MIN_CELLS},
            "high_prevalence_group": list(fpm.HIGH_PREVALENCE),
            "univariate_orientation": "sign of the training-fold class-mean gap (malignant - normal); ties -> +",
            "univariate_training_cells": "all cells outside the held-out patient (13-patient cohort, as F6)",
            "support_rule": f">= {MIN_MALIGNANT_SUPPORT} malignant cells and both classes",
            "c_policy": "not_applicable (no Stage 2 fit)",
        },
        "units": {
            "n_merged": int(unit_seconds.size),
            "n_computed_this_invocation": int(n_computed),
            "missing_inputs": [f"{u['panel']}|{u['method']}|K{u['k']}" for u in missing],
            "panels": int(long["stage0_panel_id"].nunique()),
        },
        "runtime_seconds": {
            "sum_unit_compute": round(float(unit_seconds.sum()), 1),
            "this_invocation_wall": round(time.time() - t_start, 1),
            "merge_and_figures": round(time.time() - t_merge, 1),
        },
        "rows": {"long": int(len(long)), "factor_summary": int(len(fsum)), "representation_summary": int(len(rsum))},
        "regression": {"n_checks": int(len(checks)), "n_pass": int((checks["status"] == "pass").sum()) if len(checks) else 0,
                       "n_fail": int((checks["status"] == "fail").sum()) if len(checks) else 0},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    artifacts.append(manifest_path)
    write_ledger(exp, ledger, artifacts, created)
    stale = dirs["cache"] / "merge_failed.json"
    if stale.exists():
        stale.unlink()
    return {"checks": checks, "rsum": rsum, "manifest": manifest}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", choices=("pilot", "all"), default="pilot",
                    help="pilot = IFN-alpha + allograft + apoptosis + TNFa/NF-kB; all = 41 biological panels")
    ap.add_argument("--panel-ids", default="", help="comma-separated stage0_panel_id list (overrides --panels)")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--workers", type=int, default=1, help="processes (capped at 12)")
    ap.add_argument("--resume", action="store_true", help="skip units whose part file is complete")
    ap.add_argument("--merge-only", action="store_true", help="only rebuild merged outputs from existing parts")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--experiment-dir", default="", help="override the experiment dir (default: paths.experiment_dir())")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()
    exp = Path(args.experiment_dir).resolve() if args.experiment_dir else EXP_DEFAULT
    if not exp.is_dir():
        experiment_dir()  # raises with the SC_EXPERIMENT_ROOT / SC_EXPERIMENT_ID hint
    dirs = out_dirs(exp)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cohort, provenance = load_cohort(exp)
    if args.panel_ids:
        panels = [p.strip() for p in args.panel_ids.split(",") if p.strip()]
    elif args.panels == "all":
        panels = biological_panels(exp)["stage0_panel_id"].tolist()
    else:
        panels = list(PILOT_PANELS)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    units, missing = plan_units(exp, panels, methods, ks)
    n_pat = len(cohort.patients)
    n_computed = 0
    if not args.merge_only:
        finished = [u for u in units if read_part(part_path(dirs["parts"], u), n_pat) is not None]
        if finished and not args.resume:
            print(f"{len(finished)} unit(s) already have complete parts under {rel(exp, dirs['parts'])}; "
                  "pass --resume to skip them or --merge-only to rebuild outputs.", file=sys.stderr)
            return 2
        todo = [u for u in units if u not in finished]
        print(f"units: {len(units)} requested, {len(finished)} finished, {len(todo)} to run, "
              f"{len(missing)} without stored scores", flush=True)
        run_units(exp, todo, dirs["parts"], cohort, max(1, min(int(args.workers), 12)))
        n_computed = len(todo)
    try:
        res = merge(exp, dirs, cohort, provenance, missing, args, t_start, n_computed, figures=not args.no_figures)
    except Exception:
        (dirs["cache"] / "merge_failed.json").write_text(json.dumps(
            {"failed_utc": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()}, indent=2))
        print("merge failed; see merge_failed.json in the cache dir, fix, then rerun with --merge-only", file=sys.stderr)
        raise
    checks = res["checks"]
    print(checks.to_string(index=False) if len(checks) else "no regression checks (IFN-alpha K40 not merged)")
    print(f"merged {res['manifest']['units']['n_merged']} units -> {rel(exp, dirs['tables'])}")
    return 1 if len(checks) and (checks["status"] == "fail").any() else 0


if __name__ == "__main__":
    sys.exit(main())
