#!/usr/bin/env python
"""D27 (W5.3): factor x predicted cell type and factor x CITE-seq surface protein, all D18 cards.

Reads stored artifacts only; nothing is refit.

Inputs (EXP-relative):
- analysis/scorecards/stage2_diagnostic/factor_cards_v1/ (D18 bundle): factor_cards_index.csv
  (card ids, orientation_sign, degenerate_flag, card context), factor_cards_shortlist.csv (the 56
  representations), and, for the regression checks, factor_cards_adt.csv and
  factor_cards_celltype.csv;
- stage1_dr/panel_id=<panel>/method=<method>/k=<K>/seed=42/scores.npy (row order = cohort cache);
- the cohort cache (study_config cohort_cache_relpath): patient, CN.label, predicted.annotation,
  batch, obsm['ADT'] raw counts (170 features: 161 markers + 9 isotypes). The raw AnnData is never
  loaded (AGENTS R8).

Outputs (new slug factor_modality_assoc_v1; nothing else in EXP is written; with --out-root DIR
every output goes under DIR/{scorecards,figures,cache}/stage2_diagnostic/factor_modality_assoc_v1/
and EXP is only read):
- analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/: d27_*.csv tables,
  d27_adt_assoc_full.parquet, d27_celltype_means_by_patient.csv.gz,
  factor_modality_assoc_v1_{findings.md,manifest.json,artifact_ledger.csv,schema.md};
- analysis/figures/stage2_diagnostic/factor_modality_assoc_v1/: D27_*.{png,pdf};
- analysis/cache/stage2_diagnostic/factor_modality_assoc_v1/parts/<representation>/ (resume units).

Usage:
  python run_d27_factor_modality_assoc.py [--n-jobs 4] [--resume]     # all 56 representations
  python run_d27_factor_modality_assoc.py --merge-only                 # rebuild tables from parts
  python run_d27_factor_modality_assoc.py --shortlist smoke --out-root <scratch>/d27_smoke

--shortlist smoke (IFN-alpha PCA40 and FA40 only) needs --out-root, so a partial run never merges
into the canonical slug.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "3")

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import (  # noqa: E402
    add_study_module_paths,
    cohort_cache,
    experiment_dir,
    git_state,
    study_config,
)

add_study_module_paths(workstreams=("W5_factor_interpretation",))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import factor_cards as fc  # noqa: E402  (D18: card ids, CLR)
import factor_modality_assoc as fma  # noqa: E402

SLUG = fma.SLUG
D18_SLUG = fc.SLUG
SCRIPT_REL = "scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d27_factor_modality_assoc.py"
MAX_THREADS = 12
IFNA = "single_geneset__hallmark_interferon_alpha_response"
PC8 = fc.card_id(IFNA, "pca", 40, 8)
FA7 = fc.card_id(IFNA, "fa", 40, 7)
# D18 values quoted in devlog 2026-10-08_d18-factor-cards (3 decimals): (card, label, marker,
# column, value). The pooled column is spearman_pooled, the stratified one spearman_patient_stratified.
D18_ANCHORS = (
    (PC8, "normal", "ADT-CD34", "rho_pooled", 0.251),
    (PC8, "normal", "ADT-CD34", "rho_within_wmean_d18", 0.263),
    (PC8, "normal", "ADT-CD22", "rho_pooled", -0.221),
    (PC8, "malignant", "ADT-CD99", "rho_pooled", 0.222),
    (PC8, "malignant", "ADT-CD99", "rho_within_wmean_d18", 0.192),
    (PC8, "malignant", "ADT-CD244", "rho_pooled", 0.209),
    (PC8, "malignant", "ADT-GPR56", "rho_pooled", 0.203),
    (FA7, "normal", "ADT-CD22", "rho_pooled", -0.354),
    (FA7, "normal", "ADT-CD112", "rho_pooled", 0.297),
    (FA7, "normal", "ADT-CD34", "rho_pooled", 0.260),
    (FA7, "malignant", "ADT-GPR56", "rho_pooled", 0.215),
    (FA7, "malignant", "ADT-GPR56", "rho_within_wmean_d18", 0.155),
    (FA7, "malignant", "ADT-CD63", "rho_pooled", -0.181),
    (FA7, "malignant", "ADT-CD48", "rho_pooled", -0.180),
)

# --------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------


class Paths:
    """Inputs are read from EXP; outputs go to EXP/analysis or, with out_root, to out_root."""

    def __init__(self, exp: Path, out_root: Optional[Path] = None):
        self.exp = exp
        self.out_root = Path(os.path.abspath(out_root)) if out_root else None
        base = self.out_root if self.out_root else exp / "analysis"
        self.scorecards = base / "scorecards" / "stage2_diagnostic" / SLUG
        self.figures = base / "figures" / "stage2_diagnostic" / SLUG
        self.cache = base / "cache" / "stage2_diagnostic" / SLUG
        self.parts = self.cache / "parts"
        for p in (self.scorecards, self.figures, self.cache):
            if p.name != SLUG or base not in p.parents:
                raise RuntimeError("refusing to write outside the {} namespace: {}".format(SLUG, p))
        self.d18 = exp / "analysis" / "scorecards" / "stage2_diagnostic" / D18_SLUG

    def rel(self, path: Path) -> str:
        path = Path(os.path.abspath(path))
        try:
            return path.relative_to(Path(os.path.abspath(self.exp))).as_posix()
        except ValueError:
            if self.out_root is None:
                raise
            return "<out-root>/" + path.relative_to(self.out_root).as_posix()

    def stage1_scores(self, panel: str, method: str, k: int) -> Path:
        return self.exp / "stage1_dr" / "panel_id={}".format(panel) / "method={}".format(method) / "k={}".format(int(k)) / "seed=42" / "scores.npy"


def log(msg: str) -> None:
    print("[{}] {}".format(datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


def rep_key(panel: str, method: str, k: int) -> str:
    return "{}__{}__k{}".format(panel, method, int(k))


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------


def load_d18(paths: Paths, mode: str) -> Dict[str, pd.DataFrame]:
    index = pd.read_csv(paths.d18 / "factor_cards_index.csv", low_memory=False)
    shortlist = pd.read_csv(paths.d18 / "factor_cards_shortlist.csv")
    if mode == "smoke":
        shortlist = shortlist.loc[(shortlist["stage0_panel_id"] == IFNA) & shortlist["k"].eq(40) & shortlist["stage1_method"].isin(["pca", "fa"])]
    shortlist = shortlist.reset_index(drop=True)
    index = index.loc[index["representation_id"].isin(set(shortlist["representation_id"]))].reset_index(drop=True)
    return {"index": index, "shortlist": shortlist}


_COHORT: Dict[str, Dict[str, object]] = {}


def load_cohort(paths: Paths, with_adt: bool = True) -> Dict[str, object]:
    """Per-process cohort: arrays in Stage 1 score row order, strata and ADT ranks."""
    key = "{}|{}".format(paths.exp, with_adt)
    if key in _COHORT:
        return _COHORT[key]
    import anndata as ad

    cfg = study_config()
    path = paths.exp / cfg.get("cohort_cache_relpath", "")
    if not path.is_file():
        path = cohort_cache(must_exist=True)
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    if adata.n_obs != int(cfg.get("cohort", {}).get("n_cells", 60118)):
        raise AssertionError("unexpected cohort size {}".format(adata.n_obs))
    patients = obs["patient"].astype(str).to_numpy()
    labels = obs["CN.label"].astype(str).eq("cancer").astype(int).to_numpy()
    cell_types = obs["predicted.annotation"].astype(str).to_numpy()
    batches = obs["batch"].astype(str).to_numpy() if "batch" in obs.columns else None
    supported = tuple(cfg["supported_patients"])
    strata = fma.build_strata(patients, labels, cell_types, supported)
    out: Dict[str, object] = {
        "patients": patients,
        "labels": labels,
        "cell_types": cell_types,
        "batches": batches,
        "supported": supported,
        "all_patients": tuple(sorted(pd.unique(patients))),
        "strata": strata,
        "adt": None,
        "cohort_path": path,
    }
    if with_adt:
        adt = adata.obsm["ADT"]
        names = [str(c) for c in adt.columns] if isinstance(adt, pd.DataFrame) else ["ADT_{}".format(i) for i in range(np.asarray(adt).shape[1])]
        counts = np.asarray(adt.to_numpy() if isinstance(adt, pd.DataFrame) else adt, dtype=np.float64)
        clr = fc.clr_per_cell(counts)
        out["adt"] = fma.build_adt_strata(clr, names, strata)
        out["adt_clr"] = clr
        out["adt_names"] = names
    adata.file.close()
    _COHORT[key] = out
    return out


# --------------------------------------------------------------------------------------
# per-representation worker (resume unit)
# --------------------------------------------------------------------------------------


def part_dir(paths: Paths, panel: str, method: str, k: int) -> Path:
    return paths.parts / rep_key(panel, method, k)


def _parquet_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import pyarrow.parquet as pq

        pq.read_metadata(path)
        return True
    except Exception:
        return False


def part_done(paths: Paths, panel: str, method: str, k: int) -> bool:
    """A part counts as done only with a readable _done.json and every readable parquet table."""
    d = part_dir(paths, panel, method, k)
    marker = d / "_done.json"
    if not marker.is_file() or marker.stat().st_size == 0:
        return False
    try:
        info = json.loads(marker.read_text())
    except ValueError:
        return False
    if info.get("schema_version") != fma.SCHEMA_VERSION:
        return False
    return all(_parquet_ok(d / "{}.parquet".format(name)) for name in fma.PART_TABLES)


def _to_parquet(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")
    out.to_parquet(path, index=False)


def run_representation(exp: str, out_root: Optional[str], panel: str, method: str, k: int, cards: pd.DataFrame, threads: int) -> Dict[str, object]:
    from threadpoolctl import threadpool_limits

    t0 = time.time()
    rid = fc.representation_id(panel, method, k)
    try:
        paths = Paths(Path(exp), Path(out_root) if out_root else None)
        with threadpool_limits(limits=max(1, int(threads))):
            cohort = load_cohort(paths)
            scores_path = paths.stage1_scores(panel, method, k)
            if not scores_path.is_file():
                raise FileNotFoundError("missing Stage 1 scores: {}".format(paths.rel(scores_path)))
            scores = np.load(scores_path).astype(np.float64)
            if scores.shape != (len(cohort["patients"]), int(k)):
                raise ValueError("{}: scores shape {} != ({}, {})".format(rid, scores.shape, len(cohort["patients"]), k))
            tables = fma.representation_tables(
                scores, cards, cohort["patients"], cohort["labels"], cohort["cell_types"], cohort["supported"], cohort["all_patients"], cohort["adt"]
            )
        d = part_dir(paths, panel, method, k)
        d.mkdir(parents=True, exist_ok=True)
        marker = d / "_done.json"
        if marker.exists():
            marker.unlink()
        for name in fma.PART_TABLES:
            frame = tables[name]
            if frame.empty and len(frame.columns) == 0:
                frame = pd.DataFrame({"card_id": pd.Series(dtype=str)})
            _to_parquet(frame, d / "{}.parquet".format(name))
        marker.write_text(json.dumps({"schema_version": fma.SCHEMA_VERSION, "representation_id": rid, "seconds": round(time.time() - t0, 2),
                                      "n_cards": int(len(cards)), "n_adt_rows": int(len(tables["adt"]))}))
        return {"representation_id": rid, "status": "done", "seconds": round(time.time() - t0, 2)}
    except Exception as exc:  # keep other representations going
        return {"representation_id": rid, "status": "failed", "error": "{}: {}".format(type(exc).__name__, exc), "trace": traceback.format_exc()}


def read_parts(paths: Paths, shortlist: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    parts: Dict[str, List[pd.DataFrame]] = {name: [] for name in fma.PART_TABLES}
    for _, r in shortlist.iterrows():
        panel, method, k = r["stage0_panel_id"], r["stage1_method"], int(r["k"])
        if not part_done(paths, panel, method, k):
            raise RuntimeError("unfinished part {}".format(rep_key(panel, method, k)))
        d = part_dir(paths, panel, method, k)
        for name in fma.PART_TABLES:
            frame = pd.read_parquet(d / "{}.parquet".format(name))
            if len(frame):
                parts[name].append(frame)
    out = {}
    for name, frames in parts.items():
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for col in frame.columns:
            if str(frame[col].dtype) == "string":
                frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
        out[name] = frame
    return out


# --------------------------------------------------------------------------------------
# regression checks
# --------------------------------------------------------------------------------------


def _check(rows: List[Dict[str, object]], check_id: str, description: str, source: str, diffs, tolerance: float, extra_ok: bool = True) -> None:
    diffs = np.asarray(diffs, dtype=np.float64)
    n = int(diffs.size)
    max_diff = float(np.nanmax(np.abs(diffs))) if n and np.isfinite(diffs).any() else float("nan")
    passed = bool(extra_ok and n > 0 and np.isfinite(diffs).all() and max_diff <= tolerance)
    rows.append({"check_id": check_id, "description": description, "source": source, "n_compared": n, "max_abs_diff": max_diff, "tolerance": tolerance, "passed": passed})


def regression_checks(paths: Paths, tables: Dict[str, pd.DataFrame], d18_index: pd.DataFrame, cohort: Dict[str, object], shortlist: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows: List[Dict[str, object]] = []
    cards = set(d18_index["card_id"])
    adt = tables["adt"]
    d18_adt = pd.read_csv(paths.d18 / "factor_cards_adt.csv")
    d18_adt = d18_adt.loc[d18_adt["card_id"].isin(cards)]
    m = d18_adt.merge(adt, on=["card_id", "label", "adt_marker"], how="left")
    _check(rows, "d18_adt_pooled", "pooled within-label Spearman (oriented) vs D18 factor_cards_adt.spearman_pooled (all top-10 rows)",
           "factor_cards_v1/factor_cards_adt.csv", m["spearman_pooled"] - m["rho_pooled"], 1e-8)
    _check(rows, "d18_adt_patient_stratified", "cell-weighted mean of within-patient rho (D18 strata) vs D18 spearman_patient_stratified",
           "factor_cards_v1/factor_cards_adt.csv", m["spearman_patient_stratified"] - m["rho_within_wmean_d18"], 1e-8)
    iso = fma.isotype_reference(adt)
    for label in ("normal", "malignant"):
        col = "adt_isotype_max_abs_rho_{}".format(label)
        mm = d18_index[["card_id", col, "degenerate_flag"]].loc[~d18_index["degenerate_flag"].astype(bool)].merge(iso.loc[iso["label"] == label], on="card_id", how="left")
        _check(rows, "d18_isotype_max_{}".format(label), "max |pooled rho| over the 9 isotypes vs D18 index {}".format(col), "factor_cards_v1/factor_cards_index.csv",
               mm[col] - mm["isotype_max_abs_rho_pooled"], 1e-8)
    d18_ct = pd.read_csv(paths.d18 / "factor_cards_celltype.csv")
    nondegenerate = set(d18_index.loc[~d18_index["degenerate_flag"].astype(bool), "card_id"])
    d18_ct = d18_ct.loc[d18_ct["card_id"].isin(nondegenerate)]  # D18 lists degenerate cards too; D27 skips them
    mc = d18_ct.merge(tables["celltype_means"], on=["card_id", "label", "cell_type"], how="left", suffixes=("_d18", ""))
    _check(rows, "d18_celltype_means", "pooled cell-type mean oriented z vs D18 factor_cards_celltype (>=20 cells)", "factor_cards_v1/factor_cards_celltype.csv",
           mc["mean_oriented_z_d18"] - mc["mean_oriented_z"], 1e-8, extra_ok=bool((mc["n_cells_d18"] == mc["n_cells"]).all()) and len(mc) == len(tables["celltype_means"]))
    eta = tables["celltype_eta2"]
    me = d18_index.loc[~d18_index["degenerate_flag"].astype(bool), ["card_id", "eta2_celltype_normal"]].merge(eta.loc[eta["label"] == "normal", ["card_id", "eta2_celltype_pooled"]], on="card_id", how="left")
    _check(rows, "d18_eta2_celltype_normal", "pooled cell-type eta2 in normal cells vs D18 index eta2_celltype_normal", "factor_cards_v1/factor_cards_index.csv",
           me["eta2_celltype_normal"] - me["eta2_celltype_pooled"], 1e-8)
    # anchors quoted in the D18 devlog (3 decimals)
    key = adt.set_index(["card_id", "label", "adt_marker"])
    anchor_diffs = []
    for cid, label, marker, col, value in D18_ANCHORS:
        if cid not in cards:
            continue
        anchor_diffs.append(round(float(key.loc[(cid, label, marker), col]), 3) - value)
    if anchor_diffs:
        _check(rows, "d18_devlog_anchors", "IFN-alpha PC8/FA7 rho quoted in devlog 2026-10-08_d18-factor-cards (CD34 +0.251 normal, CD99 +0.222 malignant, ...), rounded to 3 decimals",
               "devlog 2026-10-08_d18-factor-cards", anchor_diffs, 1e-9)
    # coverage
    n_keep = int((~d18_index["degenerate_flag"].astype(bool)).sum())
    n_markers = len(cohort["adt_names"])
    _check(rows, "coverage_adt_rows", "ADT rows = non-degenerate cards x 2 labels x 170 features", "this run",
           [len(adt) - n_keep * 2 * n_markers], 0.0)
    _check(rows, "coverage_cards", "every non-degenerate D18 card has ADT and eta2 rows; degenerate cards have none", "this run",
           [len(set(adt["card_id"]) ^ set(d18_index.loc[~d18_index["degenerate_flag"].astype(bool), "card_id"])), len(set(eta["card_id"]) ^ set(adt["card_id"]))], 0.0)
    # vectorized Spearman vs scipy.stats.spearmanr on sampled strata x markers for PC8 (or the first card)
    probe = PC8 if PC8 in cards else sorted(cards)[0]
    info = fc.parse_card_id(probe)
    scores = np.load(paths.stage1_scores(info["panel_id"], info["method"], info["k"]))[:, info["factor"] - 1].astype(np.float64)
    sign = int(d18_index.set_index("card_id").loc[probe, "orientation_sign"]) or 1
    names = list(cohort["adt_names"])
    clr = cohort["adt_clr"]
    strata = {s.key: s for s in cohort["strata"]}
    rng = np.random.default_rng(27)
    pct_keys = [k for k, s in strata.items() if s.kind == "patient_celltype"]
    picks = ["normal|pooled", "malignant|pooled", "malignant|P03", "malignant|P07", "normal|P09"] + list(rng.choice(pct_keys, size=min(3, len(pct_keys)), replace=False))
    markers = ["ADT-CD34", "ADT-CD99"] + [n for n in names if fma.is_isotype_name(n)][:1] + list(rng.choice(names, size=3, replace=False))
    sub = adt.loc[adt["card_id"] == probe].set_index(["label", "adt_marker"])
    diffs = []
    for skey in picks:
        s = strata[skey]
        for marker in markers:
            ref = spearmanr(scores[s.rows], clr[s.rows, names.index(marker)]).statistic * sign
            if s.kind == "pooled":
                got = sub.loc[(s.label, marker), "rho_pooled"]
            elif s.kind == "patient":
                got = sub.loc[(s.label, marker), "rho_{}".format(s.patient)]
            else:  # patient x cell type rho is summarised, not stored: check the vectorized kernel itself
                got = fma.spearman_from_ranks(fma.rank_standardize(clr[s.rows][:, [names.index(marker)]]), fma.rank_standardize(scores[s.rows]))[0, 0] * sign
            diffs.append(float(got) - float(ref))
    _check(rows, "scipy_spearman_spot", "vectorized rank-product Spearman vs scipy.stats.spearmanr ({}; {} strata x {} markers)".format(probe, len(picks), len(markers)),
           "scipy", diffs, 1e-10)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# figures (static; palette and type scale follow the questions report's report_style.py)
# --------------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
METHOD_COLOR = {"pca": BLUE, "fa": ORANGE, "factosig": AQUA, "factosig_promax": AQUA}
METHOD_LABEL = {"pca": "PCA", "fa": "FA", "factosig": "FactoSig", "factosig_promax": "FactoSig-promax"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 8.5, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
    })
    return plt


def _save(fig, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png.with_suffix(".pdf"))
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def figure_focus_heat(tables: Dict[str, pd.DataFrame], supported: Sequence[str], out_png: Path, n_markers: int = 7) -> Optional[Path]:
    """PC8 / FA7 x normal / malignant: per-patient within-label rho for the top markers.

    Markers are columns (strongest |median within-patient rho| first, plus the D18 anchor);
    rows are the 7 supported patients, then median, pooled and the within-cell-type median."""
    from matplotlib.colors import LinearSegmentedColormap

    adt = tables["adt"]
    if not {PC8, FA7} <= set(adt["card_id"]):
        return None
    plt = _style()
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    rows = list(supported) + ["median", "pooled", "in cell type"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.4), gridspec_kw={"hspace": 0.62, "wspace": 0.22})
    anchors = {(PC8, "normal"): "ADT-CD34", (PC8, "malignant"): "ADT-CD99"}
    for i, (cid, cname) in enumerate(((PC8, "IFN-α PCA40 PC8"), (FA7, "IFN-α FA40 FA7"))):
        for j, label in enumerate(("normal", "malignant")):
            ax = axes[i, j]
            sub = adt.loc[(adt["card_id"] == cid) & (adt["label"] == label) & (~adt["is_isotype"].astype(bool))].copy()
            sub["_a"] = sub["rho_within_median"].abs()
            top = sub.sort_values("_a", ascending=False).head(n_markers)
            anchor = anchors.get((cid, label))
            if anchor and anchor not in set(top["adt_marker"]):
                top = pd.concat([top, sub.loc[sub["adt_marker"] == anchor]])
            mat = np.vstack([top["rho_{}".format(p)].to_numpy(dtype=float) for p in supported]
                            + [top[c].to_numpy(dtype=float) for c in ("rho_within_median", "rho_pooled", "rho_pct_median")])
            ax.imshow(mat, cmap=cmap, vmin=-0.6, vmax=0.6, aspect="auto")
            ax.grid(False)
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    v = mat[r, c]
                    txt = "" if not np.isfinite(v) else "{:+.2f}".format(v).replace("+0.", "+.").replace("-0.", "\u2212.")
                    ax.text(c, r, txt, ha="center", va="center", fontsize=6.2, color=INK if abs(v) < 0.42 else SURFACE,
                            fontweight="bold" if r >= len(supported) else "normal")
            ax.axhline(len(supported) - 0.5, color=SURFACE, lw=2.5)
            ax.axhline(len(supported) + 1.5, color=SURFACE, lw=2.5)
            xlabels = ["{} ({}/7)".format(fma.marker_display(m), int(n)) for m, n in zip(top["adt_marker"], top["n_patients_same_sign"])]
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=40, ha="right", rotation_mode="anchor", fontsize=6.5)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows if j == 0 else [""] * len(rows), fontsize=6.8)
            if anchor:
                pos = list(top["adt_marker"]).index(anchor)
                ax.get_xticklabels()[pos].set_fontweight("bold")
                ax.add_patch(plt.Rectangle((pos - 0.5, -0.5), 1, len(rows), fill=False, ec=INK, lw=1.0))
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title("{} \u00b7 {} cells".format(cname, label), fontsize=8.5)
    fig.text(0.0, -0.01, "Cells: Spearman \u03c1 of the oriented factor score (+ = toward malignant) with CLR surface protein, within label. "
             "Rows P01\u2013P09: within that patient; median: over the 7;\npooled: all cells of the label together (the D18 number); "
             "in cell type: median over supported patient \u00d7 predicted cell type strata (\u226530 cells). Columns: the 7 markers with the largest\n"
             "|median|, plus the D18 anchor (outlined: CD34 in normal, CD99 in malignant). (k/7): patients whose \u03c1 has the sign of the median.",
             fontsize=6.6, color=INK_2, ha="left", va="top")
    _save(fig, out_png)
    return out_png


def figure_pooled_vs_within(tables: Dict[str, pd.DataFrame], out_png: Path, n_cells: Optional[Dict[str, int]] = None) -> Path:
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    adt = tables["adt"]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    n_by_label = n_cells or {}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), gridspec_kw={"wspace": 0.16})
    for ax, label in zip(axes, ("normal", "malignant")):
        sub = adt.loc[adt["label"] == label]
        mk = sub.loc[~sub["is_isotype"].astype(bool)]
        iso = sub.loc[sub["is_isotype"].astype(bool)]
        q99 = float(np.nanquantile(iso["rho_within_median"].abs(), 0.99)) if len(iso) else np.nan
        if np.isfinite(q99):
            ax.axhspan(-q99, q99, color=GRID, alpha=0.9, lw=0, zorder=0, label="isotype 99% range")
        hb = ax.hexbin(mk["rho_pooled"], mk["rho_within_median"], gridsize=60, extent=(-0.9, 0.9, -0.9, 0.9), cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0, zorder=2)
        ax.plot([-0.9, 0.9], [-0.9, 0.9], color=MUTED, lw=0.8, ls="--")
        for y in (-fma.ROBUST_ABS_RHO, fma.ROBUST_ABS_RHO):
            ax.axhline(y, color=ORANGE, lw=0.8, ls=":")
        anchor = {"normal": "ADT-CD34", "malignant": "ADT-CD99"}[label]
        a = sub.loc[(sub["card_id"] == PC8) & (sub["adt_marker"] == anchor)]
        if len(a):
            x, y = float(a["rho_pooled"].iloc[0]), float(a["rho_within_median"].iloc[0])
            ax.scatter([x], [y], s=28, facecolor=ORANGE, edgecolor=SURFACE, lw=1.0, zorder=5)
            ax.annotate("PC8 × {}".format(fma.marker_display(anchor)), (x, y), xytext=(x + 0.12, y - 0.28), fontsize=7, color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.6})
        ax.set_xlim(-0.9, 0.9)
        ax.set_ylim(-0.9, 0.9)
        ax.set_xlabel("pooled ρ (patients mixed)")
        if label == "normal":
            ax.set_ylabel("median within-patient ρ (7 supported)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title("{} cells (n = {:,})".format(label.capitalize(), int(n_by_label.get(label, 0))) if n_by_label else "{} cells".format(label.capitalize()))
        ax.grid(True, axis="both")
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("card × marker pairs", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.text(0.0, -0.06, "Each hexagon: count of factor card × ADT marker pairs (oriented; isotypes excluded). Dashed: equal pooled and within-patient ρ. "
             "Dotted: the ±0.20 robust threshold.\nGray band: the range that 99% of isotype controls reach. Pairs far from the diagonal: the pooled ρ is carried by between-patient (and batch) differences.",
             fontsize=6.6, color=INK_2, ha="left", va="top")
    _save(fig, out_png)
    return out_png


def figure_eta2(tables: Dict[str, pd.DataFrame], cards: pd.DataFrame, out_png: Path) -> Path:
    plt = _style()
    eta = tables["celltype_eta2"]
    ctx = cards[["card_id", "method", "eta2_normal", "eta2_malignant", "is_selected_driver"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={"wspace": 0.3})
    for ax, label in zip(axes, ("normal", "malignant")):
        e = eta.loc[eta["label"] == label].merge(ctx, on="card_id", how="left")
        xcol = "eta2_{}".format(label)
        for method in ("factosig_promax", "factosig", "fa", "pca"):
            s = e.loc[e["method"] == method]
            ax.scatter(s[xcol], s["eta2_celltype_within_median"], s=6, color=METHOD_COLOR[method], alpha=0.55, lw=0,
                       label=METHOD_LABEL[method] if method != "factosig_promax" else None, rasterized=True)
        drv = e.loc[e["is_selected_driver"].astype(bool)]
        ax.scatter(drv[xcol], drv["eta2_celltype_within_median"], s=16, facecolor="none", edgecolor=INK, lw=0.6, label="selected driver")
        for (cid, name), dy in zip(((PC8, "IFN-α PC8"), (FA7, "IFN-α FA7")), (0.0, 0.08)):
            r = e.loc[e["card_id"] == cid]
            if len(r):
                x, y = float(r[xcol].iloc[0]), float(r["eta2_celltype_within_median"].iloc[0])
                ax.scatter([x], [y], s=26, facecolor=ORANGE, edgecolor=INK, lw=0.7, zorder=6)
                ax.annotate(name, (x, y), xytext=(0.55, 0.30 + dy), textcoords="axes fraction", fontsize=7, color=INK, fontweight="bold",
                            bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.9},
                            arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.7}, zorder=7)
        null = float(e["eta2_null_within_median"].dropna().iloc[0]) if e["eta2_null_within_median"].notna().any() else np.nan
        lim = max(0.05, float(np.nanmax([e[xcol].max(), e["eta2_celltype_within_median"].max()])) * 1.05)
        if np.isfinite(null):
            ax.axhline(null, color=MUTED, lw=0.8, ls=":")
            ax.text(lim * 0.98, null, "no-association expectation", fontsize=6.3, color=MUTED, va="bottom", ha="right")
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("η² of patient ({} cells; D18)".format(label))
        ax.set_ylabel("η² of cell type, median within patient")
        ax.set_title("{} cells".format(label.capitalize()))
        ax.grid(True, axis="both")
    axes[0].legend(loc="upper right", markerscale=2, fontsize=6.5)
    fig.text(0.0, -0.06, "Each point: one factor card. Above the dashed line, predicted cell type explains more of the score inside a patient than patient identity explains across patients.\n"
             "Malignant within-patient η² rests on few cells per cell type (P07 has 22 malignant cells); the dotted line is the median (g−1)/(n−1) expectation under no association.",
             fontsize=6.4, color=INK_2, ha="left", va="top")
    _save(fig, out_png)
    return out_png


# --------------------------------------------------------------------------------------
# findings, manifest, ledger
# --------------------------------------------------------------------------------------


def md_table(header: Sequence[str], rows) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _f(v, d: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) else "{:+.{}f}".format(float(v), d)


def _u(v, d: int = 3) -> str:
    return "n/a" if v is None or pd.isna(v) else "{:.{}f}".format(float(v), d)


def write_findings(paths: Paths, tables: Dict[str, pd.DataFrame], derived: Dict[str, pd.DataFrame], checks: pd.DataFrame, cards: pd.DataFrame,
                   cohort: Dict[str, object], runtime: float, head: str) -> Path:
    adt = tables["adt"]
    eta = tables["celltype_eta2"]
    rob = derived["robust"]
    summary = derived["card_summary"]
    noise = derived["noise_floor"]
    strata = derived["strata"]
    supported = list(cohort["supported"])
    L: List[str] = ["# D27 factor x modality associations (factor_modality_assoc_v1): findings", ""]
    L.append("Generated by `{}` (git {}). Descriptive. Reads the D18 factor_cards_v1 bundle and the stored, transductive Stage 1 scores; nothing is refit.".format(SCRIPT_REL, head))
    L.append("")
    L.append("## How to read rho here")
    L.append("")
    L.append("- **Population.** Always one label at a time: normal cells (almost all donor-derived) or malignant cells (all recipient-derived). "
             "*Pooled* rho ranks every cell of that label together, patients mixed (this is the D18 number). *Within-patient* rho ranks the cells of one patient only; "
             "the summary is the median over the 7 supported patients plus the number of them whose rho has the same sign as that median. "
             "*Patient x cell type* rho ranks cells of one supported patient and one predicted cell type (strata with >= {} cells).".format(fma.PCT_MIN_CELLS))
    L.append("- **Normalization.** ADT = raw counts from the cohort cache, per-cell CLR: log1p(count) minus the cell's mean log1p over all 170 ADT features (161 markers + 9 isotypes). "
             "No dsb and no empty-droplet background (neither file holds them). Spearman uses ranks, so any monotone per-marker transform gives the same rho; the CLR matters only through the per-cell centering.")
    L.append("- **Sign.** The factor score is multiplied by the D18 orientation_sign (sign of the globally selected LOPO beta), so rho > 0 means cells with more of the marker sit toward the malignant-leaning pole of the factor. "
             "Unoriented cards (D18 unoriented_all_zero or unoriented_sign_flip) keep the arbitrary raw Stage 1 sign and carry oriented = False.")
    L.append("- **Magnitude.** Spearman rho is the Pearson correlation of ranks. rho = +0.25 means: pick two cells at random; the one with more of the marker also has the higher factor score about 58% of the time "
             "instead of 50% (for a bivariate normal, Kendall tau = (2/pi) arcsin(2 sin(pi rho / 6)) = 0.17 and the concordance probability is (1 + tau)/2). "
             "rho^2 = 0.06: about 6% of the rank variance is shared. That is a small-to-moderate, population-level tendency, not a marker that identifies the factor in single cells.")
    L.append("- **Noise.** With 58,549 normal cells the sampling SE of a pooled rho is about 0.004, so statistical significance is uninformative; within one patient the SE is about 1/sqrt(n-3) "
             "(about 0.22 for P07's 22 malignant cells). Read magnitude, within-patient sign consistency and the isotype reference instead of p-values. About 580,000 card x marker x label pairs are screened; the ranked table is a screen, not inference.")
    L.append("- **Confounds.** (1) Patient and batch: patients are nested in 5 batches (P01-P03 in one, P04 and P06 in another, P07 alone, P09 across two), so a pooled rho can come from between-patient or between-batch shifts in either the score or the antibody staining; within-patient rho removes both, and they cannot be told apart. "
             "(2) Origin: every malignant cell is recipient-derived and almost every normal cell is donor-derived; within-label rho avoids the label contrast itself, but nothing here separates malignant biology from recipient genotype (for example HLA-reactive antibodies). "
             "(3) Cell type: many markers (CD34, CD38, CD71, CD123, ...) and many factors both track predicted cell type, so a within-patient rho can still be a cell-identity axis; the patient x cell type column and the cell-type eta2 tables show how much survives inside a cell type. "
             "(4) CLR is compositional: a cell with more total antibody shifts every feature; the isotype rho (no specific target) measures this non-specific component. "
             "(5) Factors are not independent: matched factors across methods and K (D18 matched_cards) repeat the same association, so counts of robust pairs overstate independent findings.")
    L.append("")
    L.append("## Coverage")
    L.append("")
    n_cards = adt["card_id"].nunique()
    L.append("- Representations: {}; non-degenerate cards: {} (degenerate cards skipped: {}).".format(cards["representation_id"].nunique(), n_cards, int(cards["degenerate_flag"].astype(bool).sum())))
    for label in ("normal", "malignant"):
        sp = strata.loc[(strata["label"] == label) & (strata["kind"] == "pooled")]
        pt = strata.loc[(strata["label"] == label) & (strata["kind"] == "patient") & strata["supported_patient"].astype(bool)]
        pct = strata.loc[(strata["label"] == label) & (strata["kind"] == "patient_celltype")]
        L.append("- {}: pooled n = {}; supported-patient strata: {}; supported patient x cell type strata (>= {} cells): {}.".format(
            label.capitalize(), int(sp["n_cells"].iloc[0]) if len(sp) else 0, ", ".join("{} {}".format(r["patient"], int(r["n_cells"])) for _, r in pt.iterrows()), fma.PCT_MIN_CELLS, len(pct)))
    L.append("")
    L.append("## Regression checks")
    L.append("")
    L.append(md_table(["check", "n", "max abs diff", "tolerance", "passed"], [[r["check_id"], r["n_compared"], "{:.3g}".format(r["max_abs_diff"]), r["tolerance"], r["passed"]] for _, r in checks.iterrows()]))
    L.append("")
    L.append("## Noise reference (isotype controls)")
    L.append("")
    nf = noise.copy()
    L.append(md_table(["label", "features", "statistic", "rows", "median", "q95", "q99", "max"],
                          [[r["label"], r["feature_kind"], r["statistic"], r["n_rows"], _u(r["q50"]), _u(r["q95"]), _u(r["q99"]), _u(r["q100"])] for _, r in nf.iterrows()]))
    L.append("")
    L.append("## Robust associations (rule fixed before computing)")
    L.append("")
    L.append("Rule: non-isotype marker, |median within-patient rho| >= {:.2f}, and the same sign in >= {} of the 7 supported patients.".format(fma.ROBUST_ABS_RHO, fma.ROBUST_MIN_SAME_SIGN))
    L.append("")
    sens_rows = []
    mk = adt.loc[~adt["is_isotype"].astype(bool)]
    for label in ("normal", "malignant"):
        sub = mk.loc[mk["label"] == label]
        for thr in fma.SENSITIVITY_ABS_RHO:
            mask = fma.robust_mask(sub, abs_rho=thr)
            pooled_mask = sub["rho_pooled"].abs() >= thr
            sens_rows.append([label, "{:.2f}".format(thr), int(pooled_mask.sum()), int(mask.sum()), int(sub.loc[mask, "card_id"].nunique()),
                              int((mask & pooled_mask).sum()), "{:.1%}".format((mask & pooled_mask).sum() / max(int(pooled_mask.sum()), 1))])
    L.append(md_table(["label", "|rho| threshold", "pairs with |pooled| >= thr", "robust pairs", "cards with >= 1 robust", "robust among pooled >= thr", "share"], sens_rows))
    L.append("")
    for label in ("normal", "malignant"):
        sub = mk.loc[(mk["label"] == label) & (mk["rho_pooled"].abs() >= fma.ROBUST_ABS_RHO)]
        if len(sub):
            ratio = (sub["rho_within_median"] / sub["rho_pooled"]).replace([np.inf, -np.inf], np.nan)
            flips = int(((np.sign(sub["rho_within_median"]) != np.sign(sub["rho_pooled"])) & (sub["rho_within_median"].abs() >= 0.10)).sum())
            L.append("- {}: for pairs with |pooled rho| >= {:.2f} (n = {}), median within/pooled ratio {:.2f} (IQR {:.2f} to {:.2f}); sign reversals with |within median| >= 0.10: {}.".format(
                label.capitalize(), fma.ROBUST_ABS_RHO, len(sub), float(ratio.median()), float(ratio.quantile(0.25)), float(ratio.quantile(0.75)), flips))
    drivers = cards.loc[cards["is_selected_driver"].astype(bool), "card_id"]
    for label in ("normal", "malignant"):
        n_drv = int(rob.loc[(rob["label"] == label) & rob["card_id"].isin(drivers), "card_id"].nunique())
        L.append("- {}: selected drivers (D18, n = {}) with >= 1 robust marker: {}.".format(label.capitalize(), len(drivers), n_drv))
    L.append("")
    L.append("Reference rates (descriptive, computed after the rule was fixed): the same rule applied to the isotype rows as if they were markers; "
             "robust pairs whose |median within-patient rho| exceeds their own card's largest isotype value; and robust pairs that keep the sign and at least half their size "
             "inside supported patient x cell type strata (|pt x ct median| >= 0.5 |median within|).")
    L.append("")
    ref_rows = []
    iso_rows = adt.loc[adt["is_isotype"].astype(bool)]
    for label in ("normal", "malignant"):
        iso_l = iso_rows.loc[iso_rows["label"] == label]
        iso_pass = (iso_l["rho_within_median"].abs() >= fma.ROBUST_ABS_RHO) & (iso_l["n_patients_same_sign"] >= fma.ROBUST_MIN_SAME_SIGN)
        mk_l = mk.loc[mk["label"] == label]
        mk_pass = fma.robust_mask(mk_l)
        rl = rob.loc[rob["label"] == label]
        exceeds = int(rl["exceeds_isotype_reference"].astype(bool).sum())
        keeps = rl["rho_pct_median"].notna() & (np.sign(rl["rho_pct_median"]) == np.sign(rl["rho_within_median"])) & (rl["rho_pct_median"].abs() >= 0.5 * rl["rho_within_median"].abs())
        ratio = (rl["rho_pct_median"] / rl["rho_within_median"]).replace([np.inf, -np.inf], np.nan)
        ref_rows.append([label, "{} of {} ({:.1%})".format(int(iso_pass.sum()), len(iso_l), iso_pass.mean() if len(iso_l) else np.nan),
                         "{} of {} ({:.1%})".format(int(mk_pass.sum()), len(mk_l), mk_pass.mean() if len(mk_l) else np.nan),
                         "{} of {}".format(exceeds, len(rl)), "{} of {}".format(int(keeps.sum()), len(rl)), _u(ratio.median(), 2)])
    L.append(md_table(["label", "isotype rows passing the rule", "marker rows passing", "robust > own isotype ref", "robust kept within cell type", "median pt x ct / within ratio"], ref_rows))
    by_method = rob.groupby(["label", "method"])["card_id"].nunique().unstack(fill_value=0) if len(rob) else pd.DataFrame()
    if not by_method.empty:
        L.append("")
        L.append("Cards with >= 1 robust marker, by method (cards per method in the bundle: {}):".format(", ".join("{} {}".format(m, int(n)) for m, n in summary.groupby("method").size().items())))
        L.append("")
        L.append(md_table(["label"] + list(by_method.columns), [[i] + [int(v) for v in r] for i, r in by_method.iterrows()]))
    L.append("")
    L.append("### Top 25 robust pairs")
    L.append("")
    show = rob.head(25)
    L.append(md_table(["rank", "panel", "method K factor", "label", "marker", "median within", "same sign", "pooled", "pt x ct median", "isotype ref", "oriented", "driver"],
                          [[int(r["rank"]), r["panel_label"], "{} {} f{}".format(r["method"], int(r["k"]), int(r["factor"])), r["label"], r["adt_marker_short"], _f(r["rho_within_median"]),
                            "{}/{}".format(int(r["n_patients_same_sign"]), int(r["n_patients_valid"])), _f(r["rho_pooled"]), _f(r["rho_pct_median"]), _u(r["isotype_max_abs_rho_within_median"]),
                            bool(r["oriented"]), bool(r["is_selected_driver"])] for _, r in show.iterrows()]))
    L.append("")
    L.append("### Strongest robust pair per panel and label (selected drivers and oriented cards first)")
    L.append("")
    if len(rob):
        rr = rob.assign(_drv=rob["is_selected_driver"].astype(bool), _or=rob["oriented"].astype(bool), _a=rob["rho_within_median"].abs())
        best = rr.sort_values(["_drv", "_or", "_a"], ascending=[False, False, False]).drop_duplicates(["panel_label", "label"]).sort_values(["panel_label", "label"])
        L.append(md_table(["panel", "label", "card", "marker", "median within", "same sign", "pooled", "pt x ct median", "driver", "oriented"],
                          [[r["panel_label"], r["label"], "{} {} f{}".format(r["method"], int(r["k"]), int(r["factor"])), r["adt_marker_short"], _f(r["rho_within_median"]),
                            "{}/{}".format(int(r["n_patients_same_sign"]), int(r["n_patients_valid"])), _f(r["rho_pooled"]), _f(r["rho_pct_median"]),
                            bool(r["is_selected_driver"]), bool(r["oriented"])] for _, r in best.iterrows()]))
    L.append("")
    L.append("## IFN-alpha PC8 and FA7 (the D18 anchors)")
    L.append("")
    if PC8 in set(adt["card_id"]):
        key = adt.set_index(["card_id", "label", "adt_marker"])
        for label, marker in (("normal", "ADT-CD34"), ("malignant", "ADT-CD99")):
            r = key.loc[(PC8, label, marker)]
            per = ", ".join("{} {}".format(p, _f(r["rho_{}".format(p)], 2)) for p in supported)
            L.append("- Anchor reading, PC8 x {} in {} cells: the D18 pooled rho {} ranks all {} cells together. Inside each supported patient the rho is {} "
                     "(median {}, same sign in {}/{}; robust rule {}). Inside supported patient x predicted cell type strata the median is {}: "
                     "{}.".format(fma.marker_display(marker), label, _f(r["rho_pooled"]), label, per, _f(r["rho_within_median"]), int(r["n_patients_same_sign"]),
                                  int(r["n_patients_valid"]), "met" if bool(fma.robust_mask(pd.DataFrame([r])).iloc[0]) else "not met", _f(r["rho_pct_median"]),
                                  "most of the within-patient association is carried by which cell types score high, not by variation inside a cell type"
                                  if abs(float(r["rho_pct_median"])) < 0.5 * abs(float(r["rho_within_median"])) else
                                  "at least half of the within-patient association remains inside cell types"))
        L.append("")
    for cid, name in ((PC8, "PCA40 factor 8 (PC8)"), (FA7, "FA40 factor 7 (FA7)")):
        if cid not in set(adt["card_id"]):
            continue
        for label in ("normal", "malignant"):
            sub = adt.loc[(adt["card_id"] == cid) & (adt["label"] == label) & (~adt["is_isotype"].astype(bool))].copy()
            sub["_a"] = sub["rho_within_median"].abs()
            top = sub.sort_values("_a", ascending=False).head(6)
            anchor = {"normal": "ADT-CD34", "malignant": "ADT-CD99"}[label] if cid == PC8 else None
            if anchor is not None and anchor not in set(top["adt_marker"]):
                top = pd.concat([top, sub.loc[sub["adt_marker"] == anchor]])
            L.append("{} in {} cells (top markers by |median within-patient rho|; columns are the per-patient rho):".format(name, label))
            L.append("")
            L.append(md_table(["marker", "pooled", "median within", "same sign"] + supported + ["pt x ct median"],
                                  [[r["adt_marker_short"], _f(r["rho_pooled"]), _f(r["rho_within_median"]), "{}/{}".format(int(r["n_patients_same_sign"]), int(r["n_patients_valid"]))]
                                   + [_f(r["rho_{}".format(p)], 2) for p in supported] + [_f(r["rho_pct_median"])] for _, r in top.iterrows()]))
            iso = derived["isotype_reference"]
            ir = iso.loc[(iso["card_id"] == cid) & (iso["label"] == label)]
            if len(ir):
                L.append("")
                L.append("Isotype reference for this card and label: max |pooled rho| {}, max |median within-patient rho| {}.".format(_u(ir["isotype_max_abs_rho_pooled"].iloc[0]), _u(ir["isotype_max_abs_rho_within_median"].iloc[0])))
            L.append("")
        e = eta.loc[eta["card_id"] == cid].set_index("label")
        s = summary.loc[summary["card_id"] == cid]
        if len(e) and len(s):
            s = s.iloc[0]
            L.append("- Cell type: eta2 pooled normal {} (within-patient median {}), malignant {} (within-patient median {}); highest normal cell type {} ({}), lowest {} ({}).".format(
                _u(e.loc["normal", "eta2_celltype_pooled"]), _u(e.loc["normal", "eta2_celltype_within_median"]), _u(e.loc["malignant", "eta2_celltype_pooled"]),
                _u(e.loc["malignant", "eta2_celltype_within_median"]), s.get("celltype_highest_normal"), _f(s.get("celltype_highest_mean_normal"), 2),
                s.get("celltype_lowest_normal"), _f(s.get("celltype_lowest_mean_normal"), 2)))
            L.append("")
    L.append("## Cell type (predicted.annotation)")
    L.append("")
    for label in ("normal", "malignant"):
        e = eta.loc[eta["label"] == label]
        L.append("- {}: eta2 of cell type pooled median {} (q90 {}; null expectation {}), within-patient median of medians {} (q90 {}; null median {}).".format(
            label.capitalize(), _u(e["eta2_celltype_pooled"].median()), _u(e["eta2_celltype_pooled"].quantile(0.9)), _u(e["eta2_null_pooled"].iloc[0] if len(e) else np.nan, 4),
            _u(e["eta2_celltype_within_median"].median()), _u(e["eta2_celltype_within_median"].quantile(0.9)), _u(e["eta2_null_within_median"].iloc[0] if len(e) else np.nan, 4)))
        m = e.merge(cards[["card_id", "eta2_normal", "eta2_malignant"]], on="card_id")
        xcol = "eta2_{}".format(label)
        L.append("  Cards whose within-patient cell-type eta2 exceeds their patient eta2 (D18): {} of {}.".format(int((m["eta2_celltype_within_median"] > m[xcol]).sum()), len(m)))
    e = eta.loc[eta["label"] == "normal"].merge(summary[["card_id", "panel_label", "method", "k", "factor", "is_selected_driver", "celltype_highest_normal", "celltype_lowest_normal"]], on="card_id")
    top_ct = e.sort_values("eta2_celltype_within_median", ascending=False).head(10)
    L.append("")
    L.append("Top 10 cards by within-patient cell-type eta2 in normal cells:")
    L.append("")
    L.append(md_table(["panel", "card", "eta2 within (median)", "eta2 pooled", "highest", "lowest", "driver"],
                          [[r["panel_label"], "{} {} f{}".format(r["method"], int(r["k"]), int(r["factor"])), _u(r["eta2_celltype_within_median"]), _u(r["eta2_celltype_pooled"]),
                            r["celltype_highest_normal"], r["celltype_lowest_normal"], bool(r["is_selected_driver"])] for _, r in top_ct.iterrows()]))
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- Descriptive only. Stage 1 is transductive (fit on all 13 patients); nothing here is inductive or causal, and cell-level associations are not recurrence-risk statements.")
    L.append("- predicted.annotation is a label transferred from a healthy reference; for malignant cells it names the nearest healthy state, not a validated identity.")
    L.append("- Malignant within-patient statistics rest on 22 (P07) to 758 (P03) cells; P03 holds 758 of the 1,569 malignant cells, so the pooled malignant rho leans on P03.")
    L.append("- Unoriented cards: sign is arbitrary; compare |rho| only.")
    L.append("")
    L.append("Runtime of this invocation: {:.0f} s.".format(runtime))
    path = paths.scorecards / "{}_findings.md".format(SLUG)
    path.write_text("\n".join(L) + "\n")
    return path


def input_records(paths: Paths, cohort_path: Path) -> Dict[str, Dict[str, object]]:
    def rec(p: Path) -> Dict[str, object]:
        st = p.stat()
        return {"path": paths.rel(p), "size_bytes": int(st.st_size), "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}

    items = {
        "d18_index": paths.d18 / "factor_cards_index.csv",
        "d18_shortlist": paths.d18 / "factor_cards_shortlist.csv",
        "d18_adt": paths.d18 / "factor_cards_adt.csv",
        "d18_celltype": paths.d18 / "factor_cards_celltype.csv",
        "d18_manifest": paths.d18 / fc.MANIFEST_NAME,
        "cohort_cache": cohort_path,
    }
    out = {k: rec(v) for k, v in items.items() if Path(v).is_file()}
    out["stage1_dr"] = {"path": "stage1_dr/panel_id=<panel>/method=<method>/k=<K>/seed=42/scores.npy"}
    return out


def write_ledger(paths: Paths) -> Path:
    rows = []
    kinds = {".csv": "table", ".gz": "table", ".parquet": "table", ".json": "manifest", ".md": "doc", ".png": "figure", ".pdf": "figure"}
    ledger = paths.scorecards / "{}_artifact_ledger.csv".format(SLUG)
    for root in (paths.scorecards, paths.figures):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p != ledger:
                rows.append({"path": paths.rel(p), "kind": kinds.get(p.suffix, "other"), "size_bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(ledger, index=False)
    return ledger


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------


def merge_and_finish(paths: Paths, d18: Dict[str, pd.DataFrame], args: argparse.Namespace, t_start: float, compute_seconds: float) -> int:
    shortlist, cards = d18["shortlist"], d18["index"]
    log("merging {} representation parts".format(len(shortlist)))
    try:
        tables = read_parts(paths, shortlist)
    except Exception as exc:
        (paths.cache / "merge_failed").write_text("{}\n{}".format(exc, traceback.format_exc()))
        log("merge failed: {} (finish the parts, then rerun with --merge-only)".format(exc))
        return 2
    flag = paths.cache / "merge_failed"
    if flag.exists():
        flag.unlink()
    cohort = load_cohort(paths)
    supported = list(cohort["supported"])
    adt = tables["adt"]
    for col in ("is_isotype", "oriented"):
        adt[col] = adt[col].astype(bool)
    iso_ref = fma.isotype_reference(adt)
    derived = {
        "isotype_reference": iso_ref,
        "noise_floor": fma.noise_floor_table(adt),
        "robust": fma.robust_table(adt, cards, iso_ref, supported),
        "card_summary": fma.card_summary(cards, adt, tables["celltype_eta2"], tables["celltype_means"], iso_ref),
        "strata": fma.strata_table(cohort["strata"], cohort["cell_types"], cohort["batches"]),
    }
    eta = tables["celltype_eta2"].merge(cards[["card_id", "eta2_normal", "eta2_malignant"]].rename(columns={"eta2_normal": "eta2_patient_normal_d18", "eta2_malignant": "eta2_patient_malignant_d18"}), on="card_id", how="left")
    paths.scorecards.mkdir(parents=True, exist_ok=True)
    out = paths.scorecards
    sup_cols = ["rho_{}".format(p) for p in supported]
    top = fma.top_markers(adt)
    top_cols = ["card_id", "label", "rank", "adt_marker", "adt_marker_short", "rho_within_median", "n_patients_same_sign", "n_patients_valid", "rho_within_min", "rho_within_max",
                "rho_pooled", "rho_pct_median", "n_pct_same_sign", "n_pct_strata", "oriented"] + sup_cols
    written: Dict[str, int] = {}

    def write_csv(frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(out / name, index=False, float_format=fma.FLOAT_FORMAT)
        written[name] = int(len(frame))

    full_cols = ["card_id", "label", "adt_marker", "adt_marker_short", "is_isotype", "oriented", "rho_pooled", "rho_within_median", "rho_within_min", "rho_within_max",
                 "n_patients_valid", "n_patients_same_sign", "rho_within_wmean_d18", "rho_pct_median", "n_pct_strata", "n_pct_same_sign"] + ["rho_{}".format(p) for p in cohort["all_patients"]]
    full = adt[full_cols].copy()
    for col in full.columns:
        if col.startswith("rho_"):
            full[col] = full[col].astype(np.float32)
    full.to_parquet(out / "d27_adt_assoc_full.parquet", index=False)
    written["d27_adt_assoc_full.parquet"] = int(len(full))
    write_csv(derived["card_summary"], "d27_card_summary.csv")
    write_csv(top[top_cols], "d27_adt_top.csv")
    write_csv(derived["robust"], "d27_adt_robust.csv")
    write_csv(eta, "d27_celltype_eta2.csv")
    write_csv(tables["celltype_means"], "d27_celltype_means.csv")
    tables["celltype_means_by_patient"].to_csv(out / "d27_celltype_means_by_patient.csv.gz", index=False, float_format=fma.FLOAT_FORMAT, compression="gzip")
    written["d27_celltype_means_by_patient.csv.gz"] = int(len(tables["celltype_means_by_patient"]))
    write_csv(adt.loc[adt["card_id"].isin([PC8, FA7]), full_cols], "d27_ifna_pc8_fa7_adt.csv")
    write_csv(derived["noise_floor"], "d27_adt_noise_floor.csv")
    write_csv(derived["strata"], "d27_strata.csv")
    log("tables written")
    checks = regression_checks(paths, tables, cards, cohort, shortlist)
    write_csv(checks, "d27_regression_checks.csv")
    figs = []
    p = figure_focus_heat(tables, supported, paths.figures / "D27_ifna_pc8_fa7_adt_by_patient.png")
    if p:
        figs.append(p)
    n_label = {s.label: s.n_cells for s in cohort["strata"] if s.kind == "pooled"}
    figs.append(figure_pooled_vs_within(tables, paths.figures / "D27_pooled_vs_within_patient_rho.png", n_label))
    figs.append(figure_eta2(tables, cards, paths.figures / "D27_celltype_vs_patient_eta2.png"))
    log("figures: {}".format(", ".join(f.name for f in figs)))
    (out / "{}_schema.md".format(SLUG)).write_text(fma.SCHEMA_MD)
    head, dirty = git_state()
    runtime = time.time() - t_start
    manifest = {
        "schema_version": fma.SCHEMA_VERSION,
        "slug": SLUG,
        "analysis_id": fma.ANALYSIS_ID,
        "task": fma.TASK,
        "inputs": input_records(paths, Path(cohort["cohort_path"])),
        "git_state": {"head": head, "dirty": dirty},
        "created_by": SCRIPT_REL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "shortlist_mode": args.shortlist,
            "out_root": "<out-root>" if args.out_root else None,
            "supported_patients": supported,
            "adt_normalization": "CLR per cell: log1p(count) - mean log1p over all 170 ADT features (161 markers + 9 isotypes); no dsb",
            "correlation": "Spearman (average ranks; rank-standardize then matrix product) within label",
            "orientation": "rho and means multiplied by D18 orientation_sign; unoriented cards keep the raw sign (oriented = False)",
            "patient_min_cells": fma.PATIENT_MIN_CELLS,
            "d18_stratum_min": fma.D18_STRATUM_MIN,
            "patient_celltype_min_cells": fma.PCT_MIN_CELLS,
            "celltype_min_cells_pooled": fma.CELLTYPE_MIN_CELLS_POOLED,
            "celltype_min_cells_patient": fma.CELLTYPE_MIN_CELLS_PATIENT,
            "robust_rule": {"abs_rho_within_median_ge": fma.ROBUST_ABS_RHO, "min_same_sign_supported_patients": fma.ROBUST_MIN_SAME_SIGN, "isotypes_excluded": True},
            "sensitivity_abs_rho": list(fma.SENSITIVITY_ABS_RHO),
            "c_policy": "not_applicable (no Stage 2 refit; orientation read from D18's stored global selection)",
            "n_jobs": args.n_jobs,
        },
        "counts": {**written, "n_cards": int(adt["card_id"].nunique()), "n_representations": int(len(shortlist)), "n_strata": int(len(cohort["strata"]))},
        "regression_checks_passed": bool(checks["passed"].all()) if not checks.empty else None,
        "runtime_seconds": round(runtime, 1),
        "compute_seconds": round(compute_seconds, 1),
        "merge_only": bool(args.merge_only),
    }
    (out / "{}_manifest.json".format(SLUG)).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_findings(paths, tables, derived, checks, cards, cohort, runtime, "{}{}".format(head, "+dirty" if dirty else ""))
    write_ledger(paths)
    log("regression checks passed: {}".format(manifest["regression_checks_passed"]))
    if not checks.empty and not checks["passed"].all():
        log("failed checks: {}".format(list(checks.loc[~checks["passed"], "check_id"])))
        return 3
    log("done in {:.0f}s".format(runtime))
    return 0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", default=None, help="override the experiment dir (default: sc_classification.utils.paths.experiment_dir())")
    parser.add_argument("--shortlist", choices=["full", "smoke"], default="full", help="full = all 56 D18 representations; smoke = IFN-alpha PCA40/FA40 (needs --out-root)")
    parser.add_argument("--out-root", default=None, help="write every output under DIR/{scorecards,figures,cache}/stage2_diagnostic/%s/ instead of EXP/analysis" % SLUG)
    parser.add_argument("--n-jobs", type=int, default=4, help="worker processes over representations; BLAS threads per worker = 12 // n_jobs")
    parser.add_argument("--resume", action="store_true", help="skip representations whose parts are complete and readable")
    parser.add_argument("--merge-only", action="store_true", help="only merge finished parts, write tables, checks, figures and findings")
    args = parser.parse_args(argv)
    if args.shortlist != "full" and not args.out_root:
        raise SystemExit("--shortlist {} needs --out-root DIR: a partial run must never merge into the canonical {} slug".format(args.shortlist, SLUG))
    args.n_jobs = max(1, min(int(args.n_jobs), MAX_THREADS))
    threads = max(1, MAX_THREADS // args.n_jobs)
    t_start = time.time()
    exp = Path(os.path.abspath(args.experiment_dir)) if args.experiment_dir else experiment_dir()
    paths = Paths(exp, Path(args.out_root) if args.out_root else None)
    if not (paths.d18 / "factor_cards_index.csv").is_file():
        raise SystemExit("D18 bundle not found at {}".format(paths.rel(paths.d18)))
    for p in (paths.scorecards, paths.figures, paths.cache, paths.parts):
        p.mkdir(parents=True, exist_ok=True)
    d18 = load_d18(paths, args.shortlist)
    shortlist, cards = d18["shortlist"], d18["index"]
    log("{} representations, {} cards ({}); workers {} x {} threads".format(len(shortlist), len(cards), args.shortlist, args.n_jobs, threads))
    compute_seconds = 0.0
    if not args.merge_only:
        todo = shortlist
        if args.resume:
            done = np.array([part_done(paths, r["stage0_panel_id"], r["stage1_method"], int(r["k"])) for _, r in shortlist.iterrows()], dtype=bool)
            todo = shortlist.loc[~done]
            log("resume: {} done, {} to compute".format(int(done.sum()), len(todo)))
        if len(todo):
            jobs = []
            for _, r in todo.iterrows():
                c = cards.loc[cards["representation_id"] == r["representation_id"], ["card_id", "factor", "orientation_sign", "degenerate_flag"]].copy()
                jobs.append((r["stage0_panel_id"], r["stage1_method"], int(r["k"]), c))
            jobs.sort(key=lambda j: -j[2])
            t0 = time.time()
            out_root = str(paths.out_root) if paths.out_root else None
            if args.n_jobs > 1 and len(jobs) > 1:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(delayed(run_representation)(str(exp), out_root, *j, threads) for j in jobs)
            else:
                results = []
                for j in jobs:
                    results.append(run_representation(str(exp), out_root, *j, threads))
                    log("{} {}".format(results[-1]["representation_id"], results[-1]["status"]))
            compute_seconds = time.time() - t0
            failed = [r for r in results if r["status"] != "done"]
            log("computed {} representations in {:.0f}s; failed {}".format(len(results) - len(failed), compute_seconds, len(failed)))
            if failed:
                for r in failed:
                    log("FAILED {}: {}".format(r["representation_id"], r["error"]))
                    print(r.get("trace", ""))
                return 1
    return merge_and_finish(paths, d18, args, t_start, compute_seconds)


if __name__ == "__main__":
    sys.exit(main())
