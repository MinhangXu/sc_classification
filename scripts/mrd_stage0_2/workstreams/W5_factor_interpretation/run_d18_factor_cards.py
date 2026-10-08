#!/usr/bin/env python
"""D18 (W5.1) generalized factor cards: build the factor_cards_v1 bundle (schema_version 1).

Reads stored artifacts only; nothing is refit.

Inputs (EXP-relative):
- stage1_dr/panel_id=*/method=*/k=*/seed=42/{scores,loadings}.npy and diagnostics.json;
- preprocessing/panels/stage0_panel_genes/<panel>.json (gene order of the loadings);
- analysis/cache/stage2_diagnostic/layer4_5/selected_lopo_coefficients.pkl.gz (globally
  selected LOPO coefficients for the 41 biological panels);
- for the six Stage 0 controls, the same selection rebuilt chunked from
  stage2_supervised/multiobjective/old34_controls_fullgrid_v1/sharedness_lopo/coefficient_paths.csv
  (selection = max cell_weighted_auprc over the control LOPO scorecard, exactly as the D1
  atlas), cached under analysis/cache/stage2_diagnostic/factor_cards_v1/;
- the cohort cache (patient, CN.label, predicted.annotation, obsm['ADT'] raw counts);
- stage1_direct_gene/.../transform=zscore/features.npz for promax structure correlations;
- the D1 atlas tables (shortlist and stored held-out AP for the reconstruction check);
- D8 / F3 / F6 / F9 / D14 tables, only for the regression checks.

Outputs (new namespaced slug factor_cards_v1; nothing else in EXP is written; with
--out-root DIR every output goes under DIR/{scorecards,figures,cache}/stage2_diagnostic/
factor_cards_v1/ instead and EXP is only read):
- analysis/scorecards/stage2_diagnostic/factor_cards_v1/ : bundle CSVs, cards/*.md,
  factor_cards_manifest.json, factor_cards_findings.md, factor_cards_artifact_ledger.csv,
  factor_cards_regression_checks.csv, bundle_schema_v1.md;
- analysis/figures/stage2_diagnostic/factor_cards_v1/ : slide PNG/PDF for IFN-alpha PC8 and FA7;
- analysis/cache/stage2_diagnostic/factor_cards_v1/ : controls coefficient cache and per-
  representation parts (the --resume units).

Usage:
  python run_d18_factor_cards.py [--n-jobs 6] [--resume]          # full shortlist -> the slug
  python run_d18_factor_cards.py --merge-only     # rebuild the bundle from finished parts
  python run_d18_factor_cards.py --shortlist smoke --out-root <scratch>/d18_smoke

The merge rewrites the bundle tables, cards/*.md and the manifest for the requested shortlist
only, so --shortlist ifna|smoke needs --out-root: a partial shortlist never merges into the
canonical factor_cards_v1 bundle that the factor explorer and the atlas deep links read.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

import factor_cards as fc  # noqa: E402

SLUG = fc.SLUG
SCRIPT_REL = "scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py"
METHODS = list(fc.METHODS)
K_VALUES = [5, 10, 20, 40]
IFNA = "single_geneset__hallmark_interferon_alpha_response"
ALLOGRAFT = "single_geneset__hallmark_allograft_rejection"
APOPTOSIS = "single_geneset__hallmark_apoptosis"
TNFA = "single_geneset__hallmark_tnfa_signaling_via_nfkb"
CONTROL_PANELS = (
    "full_34",
    "core_only",
    "hvg_top_requested_500__available_500",
    "hvg_top_requested_1000__available_1000",
    "hvg_top_requested_3000__available_3000",
    "hvg_top_requested_10000__available_10000",
)
# Pre-declared focus cells (beyond the 47 D14 bases and the IFN-alpha PCA40/FA40 pair).
FOCUS_CELLS: Tuple[Tuple[str, str, int, str], ...] = (
    (IFNA, "fa", 10, "focus_cell:F9c_fa7_cross_k"),
    (IFNA, "fa", 20, "focus_cell:F9c_fa7_cross_k"),
    (ALLOGRAFT, "pca", 40, "focus_cell:focus_panel_pca40_fa40"),
    (ALLOGRAFT, "fa", 40, "focus_cell:focus_panel_pca40_fa40"),
    (APOPTOSIS, "pca", 40, "focus_cell:focus_panel_pca40_fa40"),
    (APOPTOSIS, "fa", 40, "focus_cell:focus_panel_pca40_fa40"),
    (TNFA, "pca", 40, "focus_cell:focus_panel_pca40_fa40"),
    (TNFA, "fa", 40, "focus_cell:focus_panel_pca40_fa40"),
)
PART_TABLES = tuple(fc.TABLE_FILES)
MAX_JOBS = 12
TINY_BETA = 1e-4  # folds whose largest |beta| is below this: stored float32 probabilities tie

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
        # read-only fallback for the controls coefficient cache when outputs are redirected
        self.canonical_cache = exp / "analysis" / "cache" / "stage2_diagnostic" / SLUG
        self.parts = self.cache / "parts"
        for p in (self.scorecards, self.figures, self.cache):
            if p.name != SLUG or base not in p.parents:
                raise RuntimeError("refusing to write outside the {} namespace: {}".format(SLUG, p))

    def rel(self, path: Path) -> str:
        path = Path(os.path.abspath(path))
        try:
            return path.relative_to(Path(os.path.abspath(self.exp))).as_posix()
        except ValueError:
            if self.out_root is None:
                raise
            return "<out-root>/" + path.relative_to(self.out_root).as_posix()

    def stage1_dir(self, panel: str, method: str, k: int) -> Path:
        return self.exp / "stage1_dr" / "panel_id={}".format(panel) / "method={}".format(method) / "k={}".format(int(k)) / "seed=42"

    def panel_genes(self, panel: str) -> Path:
        return self.exp / "preprocessing" / "panels" / "stage0_panel_genes" / "{}.json".format(panel)

    def gene_z(self, panel: str) -> Optional[Path]:
        for branch in ("old34_direct_gene_all_v1", "old34_controls_direct_gene_v1"):
            p = self.exp / "stage1_direct_gene" / branch / "panel_id={}".format(panel) / "transform=zscore"
            if (p / "features.npz").is_file():
                return p
        return None

    @property
    def diag(self) -> Path:
        return self.exp / "analysis" / "scorecards" / "stage2_diagnostic"


def rep_key(panel: str, method: str, k: int) -> str:
    return "{}__{}__k{}".format(panel, method, int(k))


def log(msg: str) -> None:
    print("[{}] {}".format(datetime.now(timezone.utc).strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------------------
# shortlist
# --------------------------------------------------------------------------------------


def build_shortlist(paths: Paths, mode: str) -> pd.DataFrame:
    calls = pd.read_csv(paths.diag / "patient_factor_overlap_v1" / "patient_factor_overlap_calls.csv")
    atlas = pd.read_csv(paths.diag / "old34_d1_atlas_representation_table.csv")
    labels = atlas.drop_duplicates("stage0_panel_id").set_index("stage0_panel_id")["short_panel_label"].to_dict()
    blocks = atlas.drop_duplicates("stage0_panel_id").set_index("stage0_panel_id")["atlas_row_block"].to_dict()
    rows: Dict[Tuple[str, str, int], List[str]] = {}

    def add(panel: str, method: str, k: int, reason: str) -> None:
        rows.setdefault((panel, method, int(k)), []).append(reason)

    for _, r in calls.iterrows():
        add(r["stage0_panel_id"], r["stage1_method"], int(r["effective_k"]), "d14_screen")
    add(IFNA, "pca", 40, "ifna_pca40_fa40_pair")
    add(IFNA, "fa", 40, "ifna_pca40_fa40_pair")
    for panel, method, k, reason in FOCUS_CELLS:
        add(panel, method, k, reason)
    out = pd.DataFrame(
        [
            {
                "representation_id": fc.representation_id(p, m, k),
                "stage0_panel_id": p,
                "stage1_method": m,
                "k": k,
                "shortlist_reasons": ";".join(dict.fromkeys(reasons)),
                "panel_label": labels.get(p, p),
                "atlas_row_block": blocks.get(p, ""),
            }
            for (p, m, k), reasons in rows.items()
        ]
    )
    if mode == "smoke":
        out = out.loc[(out["stage0_panel_id"] == IFNA) & out["k"].eq(40) & out["stage1_method"].isin(["pca", "fa"])]
    elif mode == "ifna":
        out = out.loc[out["stage0_panel_id"] == IFNA]
    return out.sort_values(["atlas_row_block", "stage0_panel_id", "stage1_method", "k"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# coefficients
# --------------------------------------------------------------------------------------


def _reg_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["penalty_key"] = out["penalty"].astype(str).str.lower()
    out["C_key"] = pd.to_numeric(out["C"], errors="coerce").round(12)
    out["l1_ratio_key"] = pd.to_numeric(out["l1_ratio"], errors="coerce").fillna(-1.0).round(12)
    return out


def _file_signature(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {"size_bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def control_selection(paths: Paths) -> pd.DataFrame:
    """Globally selected regularization per control representation (D1 atlas rule)."""
    from stage2_sharedness_plotting import select_regularization_rows  # study module

    lopo = pd.read_csv(paths.exp / "analysis" / "scorecards" / "old34_controls_fullgrid_v1" / "stage2_sharedness_lopo_scorecard.csv", low_memory=False)
    lopo = lopo.loc[lopo["stage0_panel_id"].isin(CONTROL_PANELS)].copy()
    lopo["representation_id"] = (
        lopo["stage0_panel_id"].astype(str)
        + "|"
        + lopo["representation_family"].astype(str)
        + "|"
        + lopo["stage1_method"].astype(str)
        + "|"
        + pd.to_numeric(lopo["effective_k"], errors="coerce").round().astype("Int64").astype(str)
        + "|"
        + pd.to_numeric(lopo["seed"], errors="coerce").round().astype("Int64").astype(str)
    )
    selected = select_regularization_rows(
        lopo,
        group_cols=["representation_id"],
        metric_col="cell_weighted_auprc",
        mode="max_metric",
        penalty_priority=("elasticnet", "l1", "l2"),
        min_nonzero=1,
    )
    selected = _reg_keys(selected)
    selected["effective_k"] = pd.to_numeric(selected["effective_k"]).astype(int)
    return selected[["representation_id", "stage0_panel_id", "stage1_method", "effective_k", "penalty", "C", "l1_ratio", "penalty_key", "C_key", "l1_ratio_key"]]


def rebuild_control_coefficients(paths: Paths, wanted_reps: Sequence[str]) -> pd.DataFrame:
    """Chunked read of the controls' LOPO coefficient paths for the selected settings."""
    if not wanted_reps:
        return pd.DataFrame()
    source = paths.exp / "stage2_supervised" / "multiobjective" / "old34_controls_fullgrid_v1" / "sharedness_lopo" / "coefficient_paths.csv"
    cache = paths.cache / "selected_lopo_coefficients_controls.pkl.gz"
    meta_path = cache.with_suffix("").with_suffix(".metadata.json")
    selection = control_selection(paths)
    selection = selection.loc[selection["representation_id"].isin(wanted_reps)].copy()
    wanted_hash = hashlib.sha256(
        selection[["representation_id", "penalty_key", "C_key", "l1_ratio_key"]].sort_values("representation_id").to_csv(index=False).encode()
    ).hexdigest()
    signature = {"source": paths.rel(source), **_file_signature(source), "wanted_sha256": wanted_hash}
    candidates = [cache]
    if paths.out_root is not None:  # redirected run: the canonical cache may be read, never written
        candidates.append(paths.canonical_cache / cache.name)
    for cand in candidates:
        cand_meta = cand.with_suffix("").with_suffix(".metadata.json")
        if not (cand.is_file() and cand_meta.is_file()):
            continue
        try:
            if json.loads(cand_meta.read_text()) == signature:
                frame = pd.read_pickle(cand, compression="gzip")
                if set(frame["representation_id"]) >= set(selection["representation_id"]):
                    log("controls coefficient cache hit ({} rows; {})".format(len(frame), paths.rel(cand)))
                    return frame
        except Exception:  # corrupt cache -> rebuild
            pass
    log("rebuilding control coefficients from {} (chunked)".format(paths.rel(source)))
    usecols = ["stage0_panel_id", "representation_family", "stage1_method", "effective_k", "seed", "heldout_patient", "penalty", "C", "l1_ratio", "feature_id", "coefficient", "is_nonzero"]
    keys = set(zip(selection["stage0_panel_id"], selection["stage1_method"], selection["effective_k"], selection["penalty_key"], selection["C_key"], selection["l1_ratio_key"]))
    panels = set(selection["stage0_panel_id"])
    kept = []
    t0 = time.time()
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=2_000_000, low_memory=False):
        chunk = chunk.loc[chunk["stage0_panel_id"].isin(panels) & chunk["representation_family"].eq("dr")]
        if chunk.empty:
            continue
        chunk = _reg_keys(chunk)
        chunk["effective_k"] = pd.to_numeric(chunk["effective_k"]).astype(int)
        mask = [t in keys for t in zip(chunk["stage0_panel_id"], chunk["stage1_method"], chunk["effective_k"], chunk["penalty_key"], chunk["C_key"], chunk["l1_ratio_key"])]
        if any(mask):
            kept.append(chunk.loc[mask])
    frame = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=usecols)
    frame["representation_id"] = [fc.representation_id(p, m, k) for p, m, k in zip(frame["stage0_panel_id"], frame["stage1_method"], frame["effective_k"])]
    frame["selection_roles"] = "global_selected"
    log("controls rebuilt: {} rows in {:.0f}s".format(len(frame), time.time() - t0))
    paths.cache.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(cache, compression="gzip")
    meta_path.write_text(json.dumps(signature, indent=2))
    return frame


def load_coefficients(paths: Paths, shortlist: pd.DataFrame) -> pd.DataFrame:
    layer = pd.read_pickle(paths.exp / "analysis" / "cache" / "stage2_diagnostic" / "layer4_5" / "selected_lopo_coefficients.pkl.gz", compression="gzip")
    layer = layer.loc[layer["selection_roles"].astype(str).str.contains("global_selected") & layer["representation_family"].eq("dr")]
    wanted = set(shortlist["representation_id"])
    have = layer.loc[layer["representation_id"].isin(wanted)]
    missing = sorted(wanted - set(have["representation_id"]))
    controls = rebuild_control_coefficients(paths, missing)
    frame = pd.concat([have, controls], ignore_index=True) if not controls.empty else have.copy()
    still = sorted(wanted - set(frame["representation_id"]))
    if still:
        raise RuntimeError("no selected LOPO coefficients for {}".format(still))
    cols = ["representation_id", "heldout_patient", "feature_id", "coefficient", "penalty", "C", "l1_ratio"]
    frame = frame[cols].copy()
    dup = frame.duplicated(["representation_id", "heldout_patient", "feature_id"])
    if dup.any():
        raise RuntimeError("duplicate coefficient rows ({})".format(int(dup.sum())))
    return frame


# --------------------------------------------------------------------------------------
# cohort (one per worker process)
# --------------------------------------------------------------------------------------

_COHORT: Dict[str, fc.Cohort] = {}


def load_cohort(paths: Paths, with_adt: bool) -> fc.Cohort:
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
    cohort = fc.Cohort(patients, labels, tuple(cfg["supported_patients"]), cell_types=obs["predicted.annotation"].astype(str).to_numpy())
    if with_adt:
        adt = adata.obsm["ADT"]
        names = [str(c) for c in adt.columns] if isinstance(adt, pd.DataFrame) else ["ADT_{}".format(i) for i in range(np.asarray(adt).shape[1])]
        counts = np.asarray(adt.to_numpy() if isinstance(adt, pd.DataFrame) else adt, dtype=np.float64)
        cohort.adt = fc.build_adt_ranks(fc.clr_per_cell(counts), names, patients, labels)
    adata.file.close()
    _COHORT[key] = cohort
    return cohort


# --------------------------------------------------------------------------------------
# per-panel worker
# --------------------------------------------------------------------------------------


def load_stage1(paths: Paths, panel: str, method: str, k: int, genes: List[str]) -> Optional[Dict[str, object]]:
    d = paths.stage1_dir(panel, method, k)
    if not (d / "scores.npy").is_file() or not (d / "loadings.npy").is_file():
        return None
    scores = np.load(d / "scores.npy").astype(np.float64)
    loadings = np.load(d / "loadings.npy").astype(np.float64)
    if loadings.shape[0] != len(genes) and loadings.shape[1] == len(genes):
        loadings = loadings.T
    diag = json.loads((d / "diagnostics.json").read_text()) if (d / "diagnostics.json").is_file() else {}
    return {"scores": scores, "loadings": loadings, "diagnostics": diag}


def load_gene_z(paths: Paths, panel: str, genes: List[str]) -> Optional[np.ndarray]:
    import scipy.sparse as sp

    d = paths.gene_z(panel)
    if d is None:
        return None
    mat = sp.load_npz(d / "features.npz")
    names = json.loads((d / "feature_names.json").read_text())
    pos = {g: i for i, g in enumerate(names)}
    if any(g not in pos for g in genes):
        return None
    idx = [pos[g] for g in genes]
    dense = mat.tocsc()[:, idx].toarray() if sp.issparse(mat) else np.asarray(mat)[:, idx]
    return np.asarray(dense, dtype=np.float64)


def part_dir(paths: Paths, panel: str, method: str, k: int) -> Path:
    return paths.parts / rep_key(panel, method, k)


def part_done(paths: Paths, panel: str, method: str, k: int) -> bool:
    d = part_dir(paths, panel, method, k)
    marker = d / "_done.json"
    if not marker.is_file():
        return False
    try:
        info = json.loads(marker.read_text())
    except ValueError:
        return False
    for name in PART_TABLES:
        p = d / "{}.parquet".format(name)
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return info.get("schema_version") == fc.SCHEMA_VERSION


def run_panel(
    exp: str,
    panel: str,
    targets: List[Tuple[str, int, str, str]],
    coefs: pd.DataFrame,
    expected: Dict[str, Dict[str, float]],
    options: Dict[str, object],
) -> List[Dict[str, object]]:
    """Compute card tables for the shortlisted representations of one panel."""
    out_root = options.get("out_root")
    paths = Paths(Path(exp), Path(str(out_root)) if out_root else None)
    cohort = load_cohort(paths, with_adt=not options.get("no_adt"))
    genes = list(json.loads(paths.panel_genes(panel).read_text())["genes"])
    bases: List[fc.Basis] = []
    stage1: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not options.get("no_matching"):
        for method in METHODS:
            for k in K_VALUES:
                art = load_stage1(paths, panel, method, k, genes)
                if art is None:
                    continue
                stage1[(method, k)] = art
                bases.append(fc.make_basis(fc.representation_id(panel, method, k), panel, method, k, art["scores"], art["loadings"]))
    gene_z = None
    results = []
    for method, k, reasons, label in targets:
        t0 = time.time()
        try:
            art = stage1.get((method, k)) or load_stage1(paths, panel, method, k, genes)
            if art is None:
                raise FileNotFoundError("missing Stage 1 artifacts for {} {} K={}".format(panel, method, k))
            if method in fc.OBLIQUE_METHODS and gene_z is None:
                gene_z = load_gene_z(paths, panel, genes)
            rid = fc.representation_id(panel, method, k)
            rep = fc.RepInput(
                panel_id=panel,
                method=method,
                k=k,
                scores=art["scores"],
                loadings=art["loadings"],
                genes=genes,
                coefs=coefs.loc[coefs["representation_id"] == rid],
                diagnostics=art["diagnostics"],
                expected_ap=expected.get(rid),
                gene_z=gene_z if method in fc.OBLIQUE_METHODS else None,
                shortlist_reasons=reasons,
                panel_label=label,
            )
            tables = fc.build_representation_tables(rep, cohort, top_genes=int(options["top_genes"]), match_bases=bases, adt_top=int(options["adt_top"]))
            d = part_dir(paths, panel, method, k)
            d.mkdir(parents=True, exist_ok=True)
            marker = d / "_done.json"
            if marker.exists():
                marker.unlink()
            for name in PART_TABLES:
                frame = tables[name]
                frame = frame if not (frame.empty and len(frame.columns) == 0) else pd.DataFrame({"card_id": pd.Series(dtype=str)})
                _to_parquet(frame, d / "{}.parquet".format(name))
            marker.write_text(json.dumps({"schema_version": fc.SCHEMA_VERSION, "representation_id": rid, "seconds": round(time.time() - t0, 2), "n_cards": int(len(tables["index"]))}))
            results.append({"representation_id": rid, "status": "done", "seconds": round(time.time() - t0, 2)})
        except Exception as exc:  # keep other representations going
            results.append({"representation_id": fc.representation_id(panel, method, k), "status": "failed", "error": "{}: {}".format(type(exc).__name__, exc), "trace": traceback.format_exc()})
    return results


def _to_parquet(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda v: v if v is None or isinstance(v, str) or (isinstance(v, float) and np.isnan(v)) else str(v))
            out[col] = out[col].astype("string")
    out.to_parquet(path, index=False)


def read_parts(paths: Paths, shortlist: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    parts = []
    for _, r in shortlist.iterrows():
        d = part_dir(paths, r["stage0_panel_id"], r["stage1_method"], int(r["k"]))
        if not part_done(paths, r["stage0_panel_id"], r["stage1_method"], int(r["k"])):
            raise RuntimeError("unfinished part {}".format(d.name))
        parts.append({name: pd.read_parquet(d / "{}.parquet".format(name)) for name in PART_TABLES})
    tables = fc.concat_tables(parts)
    for key, frame in tables.items():
        for col in frame.columns:
            if str(frame[col].dtype) == "string":
                frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    return tables


# --------------------------------------------------------------------------------------
# regression checks (D8, F3, F6, F9, D14, atlas reconstruction)
# --------------------------------------------------------------------------------------


def regression_checks(paths: Paths, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    lift = tables["patient_lift"]
    index = tables["index"]
    matches = tables["matches"]
    genes = tables["genes"]

    def add(check_id: str, desc: str, source: str, diffs, tol: float, n: Optional[int] = None, exact: bool = False) -> None:
        arr = np.abs(np.asarray(diffs, dtype=np.float64)) if len(diffs) else np.zeros(0)
        maxd = float(np.nanmax(arr)) if arr.size and np.isfinite(arr).any() else (0.0 if exact else float("nan"))
        rows.append(
            {
                "check_id": check_id,
                "description": desc,
                "source": source,
                "n_compared": int(n if n is not None else arr.size),
                "max_abs_diff": maxd,
                "tolerance": tol,
                "passed": bool(arr.size > 0 and np.isfinite(maxd) and maxd <= tol) if not exact else bool(maxd <= tol and (n or arr.size) > 0),
            }
        )

    def cid(method: str, k: int, f: int, panel: str = IFNA) -> str:
        return fc.card_id(panel, method, k, f)

    folds_all = tables["fold_coefs"]
    fold_level = folds_all.assign(rep=[c.rsplit("__f", 1)[0] for c in folds_all["card_id"]], abs_coef=folds_all["coef"].abs())
    fold_level = fold_level.groupby(["rep", "heldout_patient"]).agg(max_abs_coef=("abs_coef", "max"), fold_model_ap=("fold_model_ap", "first"), fold_model_lift=("fold_model_lift", "first")).reset_index()
    tiny = fold_level.loc[(fold_level["max_abs_coef"] > 0) & (fold_level["max_abs_coef"] < TINY_BETA)]
    tiny_keys = set(zip(tiny["rep"], tiny["heldout_patient"]))
    mf = paths.diag / "meeting_followup_v1"
    fb = paths.diag / "feedback_deep_dive_20260917"
    po = paths.diag / "patient_factor_overlap_v1"
    have_ifna = cid("pca", 40, 8) in set(index["card_id"]) and cid("fa", 40, 7) in set(index["card_id"])
    if have_ifna:
        abl = pd.read_csv(mf / "interferon_alpha_pca40_fa40_factor_ablation.csv")
        abl["card_id"] = [cid(m, 40, int(f.split("_")[1])) for m, f in zip(abl["stage1_method"], abl["feature_id"])]
        j = abl.merge(lift, left_on=["card_id", "heldout_patient"], right_on=["card_id", "patient"], how="left")
        src = paths.rel(mf / "interferon_alpha_pca40_fa40_factor_ablation.csv")
        add("D8_coefficient", "IFN-a PCA40/FA40 fold coefficients (560 rows)", src, j["coefficient"] - j["fitted_coef"], 1e-9)
        add("D8_full_model_ap", "reconstructed held-out AP", src, j["full_model_ap"] - j["fitted_full_model_ap"], 1e-4)
        add("D8_ap_drop", "fixed-model ablation dAP", src, j["ap_drop_when_ablated"] - j["fitted_ap_drop"], 1e-4)
        add("D8_single_factor_lift", "fixed-model single-factor lift", src, j["single_factor_lift"] - j["fitted_single_factor_lift"], 1e-4)
        act = pd.read_csv(mf / "interferon_alpha_pca40_fa40_active_factors_by_fold.csv")
        act_set = {(m, p, int(f.split("_")[1])) for m, p, f in zip(act["stage1_method"], act["heldout_patient"], act["feature_id"])}
        mine = lift.loc[lift["card_id"].str.startswith(IFNA + "__") & lift["card_id"].str.contains("__k40__") & (lift["fitted_coef"].abs() > fc.COEF_EPS)]
        mine_set = {(fc.parse_card_id(c)["method"], p, fc.parse_card_id(c)["factor"]) for c, p in zip(mine["card_id"], mine["patient"]) if fc.parse_card_id(c)["method"] in ("pca", "fa")}
        add("D8_active_set", "active (method, fold, factor) set; diff = symmetric-difference size", paths.rel(mf / "interferon_alpha_pca40_fa40_active_factors_by_fold.csv"), [len(act_set ^ mine_set)], 0.0, n=len(act_set), exact=True)
        ali = pd.read_csv(mf / "interferon_alpha_pca40_fa40_factor_alignment.csv")
        mm = matches.loc[matches["other_representation_id"].eq(fc.representation_id(IFNA, "fa", 40))].copy()
        mm = mm.loc[mm["card_id"].str.contains("__pca__k40__")]
        mm["pca_factor_index"] = [fc.parse_card_id(c)["factor"] for c in mm["card_id"]]
        jj = ali.merge(mm, on="pca_factor_index", how="left")
        add("D8_F3_hungarian_index", "PCA40->FA40 Hungarian matched factor index (diff = n mismatches)", paths.rel(mf / "interferon_alpha_pca40_fa40_factor_alignment.csv"), [int((jj["fa_factor_index"] != jj["matched_factor"]).sum())], 0.0, n=len(jj), exact=True)
        add("D8_F3_hungarian_r", "matched score correlation (D8 used float32 scores; F3_hungarian_r is the float64 check)", paths.rel(mf / "interferon_alpha_pca40_fa40_factor_alignment.csv"), jj["score_correlation"] - jj["score_r"], 1e-4)
        add("D8_loading_cosine", "orientation-aligned loading cosine = Tucker phi aligned", paths.rel(mf / "interferon_alpha_pca40_fa40_factor_alignment.csv"), (jj["orientation_aligned_loading_cosine"] - jj["tucker_phi_aligned"]).dropna(), 1e-5)
        f3 = pd.read_csv(fb / "f3_ifna_pca_fa_hungarian_alignment.csv")
        j3 = f3.merge(mm, left_on="pca_factor", right_on="pca_factor_index", how="left")
        add("F3_hungarian_r", "F3 Hungarian PCA->FA score r (float64 recompute)", paths.rel(fb / "f3_ifna_pca_fa_hungarian_alignment.csv"), j3["score_correlation"] - j3["score_r"], 1e-5)
        drv = pd.read_csv(mf / "interferon_alpha_driver_factor_top_loading_genes.csv")
        diffs = []
        mismatched = 0
        for (m, feat), g in drv.groupby(["stage1_method", "feature_id"]):
            c = cid(m, 40, int(feat.split("_")[1]))
            gg = genes.loc[(genes["card_id"] == c) & (genes["ranked_by"] == "loading")]
            for direction, sub in g.groupby("loading_direction"):
                want = sub.sort_values("loading_rank")
                sign = 1 if direction == "positive_loading" else -1
                have = gg.loc[np.sign(gg["loading"]) == sign].sort_values("loading", ascending=(sign < 0)).head(len(want))
                mismatched += int(list(have["gene"]) != list(want["gene"]))
                diffs += list(np.asarray(have["loading"], dtype=float) - np.asarray(want["loading"], dtype=float)) if len(have) == len(want) else [np.inf]
        add("D8_driver_top_loading_genes", "top-12 genes per raw-sign pole for PC8, FA7, FA8 (gene order exact; diff on loadings)", paths.rel(mf / "interferon_alpha_driver_factor_top_loading_genes.csv"), diffs, 1e-6)
        add("D8_driver_top_gene_order", "pole lists with a different gene order (count)", paths.rel(mf / "interferon_alpha_driver_factor_top_loading_genes.csv"), [mismatched], 0.0, n=6, exact=True)
        f6 = pd.read_csv(fb / "f6_ifna_univariate_factor_ap_by_patient.csv")
        f6["card_id"] = [cid(m, 40, int(f.split("_")[1])) for m, f in zip(f6["stage1_method"], f6["feature_id"])]
        j6 = f6.merge(lift, left_on=["card_id", "heldout_patient"], right_on=["card_id", "patient"], how="left")
        add("F6_univariate_lift_sign_agnostic", "F6 held-out univariate lift (560 rows)", paths.rel(fb / "f6_ifna_univariate_factor_ap_by_patient.csv"), j6["heldout_univariate_lift"] - j6["univariate_lift_sign_agnostic"], 1e-9)
        focus = j6.loc[j6["card_id"].isin([cid("pca", 40, 8), cid("fa", 40, 7)])]
        add("F6_PC8_FA7_train_oriented_equals", "PC8/FA7: train-oriented univariate lift equals F6 (sign agrees in every fold)", paths.rel(fb / "f6_ifna_univariate_factor_ap_by_patient.csv"), focus["heldout_univariate_lift"] - focus["univariate_lift"], 1e-9)
        n_disagree = int((np.abs(j6["heldout_univariate_lift"] - j6["univariate_lift"]) > 1e-9).sum())
        rows.append({"check_id": "F6_info_sign_disagreements", "description": "IFN-a K40 factor x fold rows where the training-fold sign differs from the held-out best sign (information only)", "source": paths.rel(fb / "f6_ifna_univariate_factor_ap_by_patient.csv"), "n_compared": int(len(j6)), "max_abs_diff": float(n_disagree), "tolerance": np.nan, "passed": True})
        sc = pd.read_csv(fb / "f6_ifna_factor7_8_fold_scale_by_patient.csv")
        sc["card_id"] = [cid(m, 40, int(f.split("_")[1])) for m, f in zip(sc["stage1_method"], sc["feature_id"])]
        js = sc.merge(lift, left_on=["card_id", "heldout_patient"], right_on=["card_id", "patient"], how="left")
        add("F6_heldout_malignant_mean_z", "fold-standardized held-out malignant mean z (factors 7/8)", paths.rel(fb / "f6_ifna_factor7_8_fold_scale_by_patient.csv"), js["heldout_malignant_mean_z_x"] - js["heldout_malignant_mean_z_y"], 1e-6)
        f9 = pd.read_csv(fb / "f9_ifna_pc8_vs_celltype_ceiling.csv")
        diffs = []
        for _, r in f9.iterrows():
            for col, c in (("pc8_lift_pp", cid("pca", 40, 8)), ("fa7_lift_pp", cid("fa", 40, 7))):
                v = lift.loc[(lift["card_id"] == c) & (lift["patient"] == r["heldout_patient"]), "univariate_lift_sign_agnostic"]
                diffs.append(float(r[col]) - 100 * float(v.iloc[0]) if len(v) else np.inf)
        add("F9_pc8_fa7_whole_patient_lift_pp", "F9b whole-patient PC8/FA7 lift (pp)", paths.rel(fb / "f9_ifna_pc8_vs_celltype_ceiling.csv"), diffs, 1e-6)
        f9s = pd.read_csv(fb / "f9_ifna_pc8_loading_vs_gene_univariate_summary.csv")
        g8 = genes.loc[(genes["card_id"] == cid("pca", 40, 8)) & (genes["ranked_by"] == "loading")].set_index("gene")["loading"]
        jg = f9s.loc[f9s["gene"].isin(g8.index)]
        add("F9_pc8_loadings", "F9a PC8 loadings for the listed top genes", paths.rel(fb / "f9_ifna_pc8_loading_vs_gene_univariate_summary.csv"), jg["pc8_loading"].to_numpy() - g8.reindex(jg["gene"]).to_numpy(), 1e-6)
        km = pd.read_csv(fb / "f9_ifna_fa_k_match_to_fa40_factor7.csv")
        km = km.loc[km["k"] != 40]  # K=40 row is FA7 matched to itself
        diffs_r = []
        idx_mis = 0
        for _, r in km.iterrows():
            m = matches.loc[(matches["card_id"] == cid("fa", 40, 7)) & (matches["other_representation_id"] == fc.representation_id(IFNA, "fa", int(r["k"])))]
            if m.empty:
                diffs_r.append(np.inf)
                continue
            diffs_r.append(float(r["best_r_with_fa40_factor7"]) - float(m["best_r"].iloc[0]))
            idx_mis += int(int(r["best_factor"]) != int(m["best_factor"].iloc[0]))
        add("F9c_fa_k_match_r", "F9c best |r| match of FA40 f7 in FA K5/10/20", paths.rel(fb / "f9_ifna_fa_k_match_to_fa40_factor7.csv"), diffs_r, 1e-5)
        add("F9c_fa_k_match_index", "F9c matched factor index mismatches", paths.rel(fb / "f9_ifna_fa_k_match_to_fa40_factor7.csv"), [idx_mis], 0.0, n=len(km), exact=True)
        kl = pd.read_csv(fb / "f9_ifna_fa_matched_factor_lift_by_k.csv")
        kl = kl.loc[kl["k"].isin([10, 20, 40])]
        d_uni, d_sel, d_sel_tiny = [], [], []
        folds = tables["fold_coefs"]
        card_set = set(index["card_id"])
        for _, r in kl.iterrows():
            c = cid("fa", int(r["k"]), int(r["matched_factor"]))
            if c not in card_set:
                continue
            v = lift.loc[(lift["card_id"] == c) & (lift["patient"] == r["heldout_patient"]), "univariate_lift_sign_agnostic"]
            d_uni.append(float(r["univariate_lift_pp"]) - 100 * float(v.iloc[0]) if len(v) else np.inf)
            fm = folds.loc[(folds["card_id"] == c) & (folds["heldout_patient"] == r["heldout_patient"]), "fold_model_lift"]
            target = d_sel_tiny if (c.rsplit("__f", 1)[0], r["heldout_patient"]) in tiny_keys else d_sel
            target.append(float(r["selected_model_lift_pp"]) - 100 * float(fm.iloc[0]) if len(fm) else np.inf)
        add("F9c_matched_univariate_lift_pp", "F9c univariate lift of the FA7-matched factor at K10/20/40 (pp)", paths.rel(fb / "f9_ifna_fa_matched_factor_lift_by_k.csv"), d_uni, 1e-6)
        add("F9c_selected_model_lift_pp", "F9c selected-model lift at K10/20/40 (pp; reconstructed), folds with max |beta| >= 1e-4", paths.rel(fb / "f9_ifna_fa_matched_factor_lift_by_k.csv"), d_sel, 1e-2)
        if d_sel_tiny:
            rows.append({"check_id": "F9c_info_tiny_beta_folds_pp", "description": "F9c selected-model lift in folds whose only active |beta| < 1e-4: the stored float32 probabilities tie (information only)", "source": paths.rel(fb / "f9_ifna_fa_matched_factor_lift_by_k.csv"), "n_compared": len(d_sel_tiny), "max_abs_diff": float(np.max(np.abs(d_sel_tiny))), "tolerance": np.nan, "passed": True})
    fa_cards = index.loc[(index["method"] == "fa") & index["determinacy"].notna()]
    if not fa_cards.empty:
        diffs = fa_cards["determinacy"].astype(float) ** 2 - fa_cards["score_sd"].astype(float) ** 2
        add("FA_determinacy_vs_score_variance", "FA: rho^2 from saved psi vs the posterior-mean score variance (approximate identity; exact only when the fitted covariance reproduces the sample covariance)", "stage1_dr/.../method=fa/.../diagnostics.json", diffs.to_numpy(), 5e-2)
    focus_path = po / "focus_factor_overlap.csv"
    if focus_path.is_file():
        foc = pd.read_csv(focus_path)
        foc["card_id"] = [fc.card_id(p, m, int(k), int(f)) for p, m, k, f in zip(foc["stage0_panel_id"], foc["stage1_method"], foc["effective_k"], foc["factor_index"])]
        foc = foc.loc[foc["card_id"].isin(set(index["card_id"]))]
        col = np.where(foc["stratum"].eq("normal"), "eta2_normal", "eta2_malignant")
        ix = index.set_index("card_id")
        mine = [ix.loc[c, k] for c, k in zip(foc["card_id"], col)]
        add("D14_focus_eta2", "D14 focus table eta2 (IFN-a PCA40/FA40, allograft/apoptosis promax40)", paths.rel(focus_path), foc["eta2"].to_numpy() - np.asarray(mine, dtype=float), 1e-6)
    calls_path = po / "patient_factor_overlap_calls.csv"
    if calls_path.is_file():
        calls = pd.read_csv(calls_path)
        mx = index.groupby("representation_id")[["eta2_normal", "eta2_malignant"]].max()
        jc = calls.merge(mx, left_on="representation_id", right_index=True, how="inner")
        add("D14_calls_max_eta2", "max eta2 per representation, normal and malignant (47 D14 bases)", paths.rel(calls_path), np.r_[jc["normal_max_eta2"] - jc["eta2_normal"], jc["cancer_max_eta2"] - jc["eta2_malignant"]], 1e-6)
    atlas_path = paths.diag / "old34_d1_atlas_supported_patient_metrics.csv"
    atlas = pd.read_csv(atlas_path)
    rep_to_rid = {fc.card_id(p, m, k, 1).rsplit("__f", 1)[0]: rid for rid, p, m, k in zip(index["representation_id"], index["panel_id"], index["method"], index["k"])}
    fl = fold_level.loc[fold_level["fold_model_ap"].notna()].copy()
    fl["representation_id"] = fl["rep"].map(rep_to_rid)
    fl = fl.merge(atlas[["representation_id", "heldout_patient", "heldout_auprc"]], on=["representation_id", "heldout_patient"], how="inner")
    fl["diff"] = fl["fold_model_ap"] - fl["heldout_auprc"]
    is_tiny = [(r, p) in tiny_keys for r, p in zip(fl["rep"], fl["heldout_patient"])]
    strict = fl.loc[~np.asarray(is_tiny, dtype=bool)]
    add("atlas_reconstruction", "|reconstructed held-out AP - stored atlas AP| over shortlisted representations x supported folds (folds with max |beta| >= 1e-4 or all-zero)", paths.rel(atlas_path), strict["diff"].to_numpy(), 1e-4)
    tiny_rows = fl.loc[np.asarray(is_tiny, dtype=bool)]
    rows.append({"check_id": "atlas_info_tiny_beta_folds", "description": "folds whose only active |beta| < 1e-4: stored float32 probabilities tie, so stored AP differs (information only): " + ";".join("{}:{}".format(r, p) for r, p in zip(tiny_rows["representation_id"], tiny_rows["heldout_patient"])), "source": paths.rel(atlas_path), "n_compared": int(len(tiny_rows)), "max_abs_diff": float(tiny_rows["diff"].abs().max()) if len(tiny_rows) else 0.0, "tolerance": np.nan, "passed": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# slide figure for one card
# --------------------------------------------------------------------------------------

RED = "#e34948"  # diverging pole: toward malignant
BLUE = "#2a78d6"  # diverging pole: toward normal
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
GRAY_LIGHT = "#a3a29c"
GRAY_DARK = "#3d3c39"


def card_figure(tables: Dict[str, pd.DataFrame], cid: str, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = tables["index"].set_index("card_id").loc[cid]
    sign = int(idx["orientation_sign"])
    genes = tables["genes"]
    g = genes.loc[(genes["card_id"] == cid) & (genes["ranked_by"] == "loading")]
    lift = tables["patient_lift"].loc[tables["patient_lift"]["card_id"] == cid].set_index("patient")
    folds = tables["fold_coefs"]
    folds = folds.loc[(folds["card_id"] == cid) & folds["supported_fold"].astype(bool)].set_index("heldout_patient")
    dist = tables["score_dist"].loc[tables["score_dist"]["card_id"] == cid]
    matches = tables["matches"].loc[tables["matches"]["card_id"] == cid]
    adt = tables["adt"].loc[tables["adt"]["card_id"] == cid]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY_LIGHT, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(16, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.45, 1.0], height_ratios=[1, 1], hspace=0.62, wspace=0.30)

    # (a) genes from both poles
    ax = fig.add_subplot(gs[:, 0])
    pole_hi, pole_lo = ("malignant", "normal") if sign != 0 else ("positive", "negative")
    val = "oriented_loading" if sign != 0 else "loading"
    top = g.loc[g["pole"] == pole_hi].sort_values("rank").head(12)
    bot = g.loc[g["pole"] == pole_lo].sort_values("rank").head(12)
    rows = pd.concat([top, bot.iloc[::-1]])
    y = np.arange(len(rows))[::-1]
    colors = [RED if v > 0 else BLUE for v in rows[val]]
    ax.barh(y, rows[val], color=colors, height=0.72, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y, rows["gene"], fontsize=8.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    xlabel = "oriented loading (sign(beta) x loading)" if sign != 0 else "raw loading (unoriented)"
    ax.set_xlabel(xlabel)
    ax.set_title("a. Genes from both poles", pad=22)
    lim = float(np.nanmax(np.abs(rows[val]))) * 1.15 if len(rows) else 1
    ax.set_xlim(-lim, lim)
    ax.text(0.99, 1.01, "toward malignant →" if sign != 0 else "positive →", transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8)
    ax.text(0.01, 1.01, "← toward normal" if sign != 0 else "← negative", transform=ax.transAxes, ha="left", va="bottom", color=INK2, fontsize=8)

    # (b) per-patient lift with beta per fold
    ax = fig.add_subplot(gs[0, 1])
    pats = [p for p in lift.index]
    x = np.arange(len(pats))
    uni = 100 * lift["univariate_lift"].reindex(pats).to_numpy(dtype=float)
    fit = 100 * lift["fitted_single_factor_lift"].reindex(pats).to_numpy(dtype=float)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, uni, width=w, color=GRAY_LIGHT, label="factor alone (univariate)", edgecolor="white", linewidth=1.0)
    ax.bar(x + w / 2 + 0.01, fit, width=w, color=GRAY_DARK, label="inside fitted probe (single-factor)", edgecolor="white", linewidth=1.0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    labels = []
    betas = []
    for p in pats:
        if p in folds.index:
            row = folds.loc[p]
            b = "all-zero" if bool(row["fold_all_zero"]) else ("β = 0" if not bool(row["is_nonzero"]) else "β {:+.3f}".format(float(row["coef"])))
        else:
            b = ""
        betas.append(b)
        labels.append("{}\n{} mal.".format(p, int(lift.loc[p, "n_malignant"])))
    ax.set_xticks(x, labels, fontsize=8)
    top = float(np.nanmax(np.r_[uni, fit, 1.0]))
    bottom = float(min(0.0, np.nanmin(np.r_[uni, fit, 0.0])))
    ax.set_ylim(bottom - 0.04 * (top - bottom), top * 1.16)
    for xi, b in zip(x, betas):
        ax.text(xi, top * 1.10, b, ha="center", va="center", fontsize=7.2, color=INK2)
    ax.set_ylabel("held-out lift (AUPRC pp)")
    ax.set_title("b. Held-out lift per patient (top row: the fold's β)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    # (c) score distributions
    ax = fig.add_subplot(gs[1, 1])
    dd = dist.loc[dist["patient"].isin(pats)]
    for i, p in enumerate(pats):
        for lab, color, off in (("normal", BLUE, -0.17), ("malignant", RED, 0.17)):
            r = dd.loc[(dd["patient"] == p) & (dd["label"] == lab)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([i + off, i + off], [r["q10"], r["q90"]], color=color, lw=1.6, solid_capstyle="round")
            ax.plot([i + off, i + off], [r["q25"], r["q75"]], color=color, lw=5.5, solid_capstyle="round", alpha=0.9)
            ax.plot([i + off], [r["q50"]], marker="o", ms=5.5, color="white", mec=color, mew=1.6)
    ax.set_xticks(np.arange(len(pats)), pats, fontsize=8)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylabel("{} cohort z-score".format("oriented" if sign != 0 else "raw"))
    ax.set_title("c. Score by patient: normal (blue) vs malignant (red)")
    ax.text(0.99, 0.02, "dot = median, bar = IQR, line = 10-90%", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)

    # (d) facts and caveats
    ax = fig.add_subplot(gs[:, 2])
    ax.axis("off")
    best = matches.sort_values("abs_r", ascending=False)
    same_k = best.loc[best["relation"] == "hungarian_same_k"].head(3)
    cross_k = best.loc[(best["relation"] == "best_abs_r") & (best["other_method"] == idx["method"])].head(3)
    short_m = {"pca": "PCA", "fa": "FA", "factosig": "FactoSig", "factosig_promax": "promax"}
    lines = [
        ("Probe", "{} C={} (global LOPO selection)".format(idx["fit_penalty"], idx["fit_C"])),
        ("Orientation", "{} (sign {:+d})".format(idx["orientation_status"], sign)),
        ("Selected", "{} / {} supported folds; sign consistency {}".format(int(idx["n_active_folds"]), int(idx["n_folds"]), "n/a" if pd.isna(idx["sign_consistency"]) else "{:.2f}".format(float(idx["sign_consistency"])))),
        ("All-zero folds", str(idx["all_zero_folds"]) if isinstance(idx["all_zero_folds"], str) and idx["all_zero_folds"] else "none"),
        ("Patient eta2", "normal {:.3f} / malignant {:.3f}".format(float(idx["eta2_normal"]), float(idx["eta2_malignant"]))),
        ("Determinacy", "n/a (exact PCA scores)" if pd.isna(idx["determinacy"]) else "{:.3f}".format(float(idx["determinacy"]))),
        ("Eigengap (rel.)", "n/a" if pd.isna(idx["eigengap_rel"]) else "{:.3f} ({})".format(float(idx["eigengap_rel"]), idx["eigengap_basis"])),
        ("effective_k", "{} of {}".format(int(idx["effective_k"]), int(idx["k"]))),
    ]
    lines.append(("Matches", "score r, R² on the whole basis, Tucker φ"))
    for _, m in same_k.iterrows():
        lines.append(("↔ {} K={}".format(short_m.get(m["other_method"], m["other_method"]), int(m["other_k"])), "f{}  r {:+.2f}  R² {:.2f}  φ {:+.2f}".format(int(m["matched_factor"]), m["score_r"], m["r2_many_to_one"], m["tucker_phi_aligned"])))
    for _, m in cross_k.iterrows():
        lines.append(("↔ {} K={}".format(short_m.get(m["other_method"], m["other_method"]), int(m["other_k"])), "f{}  r {:+.2f}  R² {:.2f}".format(int(m["matched_factor"]), m["score_r"], m["r2_many_to_one"])))
    for lab in ("malignant", "normal"):
        a = adt.loc[adt["label"] == lab].sort_values("rank").head(3)
        if not a.empty:
            lines.append(("ADT ({})".format(lab), ", ".join("{} {:+.2f}".format(str(r["adt_marker"]).replace("ADT-", ""), r["spearman_pooled"]) for _, r in a.iterrows())))
    yy = 0.99
    ax.text(0.0, yy, "d. Card facts", fontsize=10.5, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    yy -= 0.05
    for key, value in lines:
        ax.text(0.0, yy, key, fontsize=8.3, color=INK2, va="top", transform=ax.transAxes)
        ax.text(0.40, yy, value, fontsize=8.3, color=INK, va="top", transform=ax.transAxes)
        yy -= 0.042
    yy -= 0.02
    ax.text(0.0, yy, "Caveats", fontsize=9.5, fontweight="bold", color=INK, va="top", transform=ax.transAxes)
    yy -= 0.045
    short = {
        "transductive_basis": "Transductive Stage 1: LOPO is label transfer, not deployment.",
        "ablation_is_attribution": "Single-factor lift / dAP: attribution in the fitted model, not causal.",
        "has_all_zero_folds": "All-zero folds score AP = prevalence by construction.",
        "small_eigengap": "Small eigengap: axis not identifiable on its own.",
        "low_determinacy": "Low FA determinacy: not a unique program.",
        "generative_loadings": "FA loadings are generative, not score weights.",
        "small_patient_support": "Patients with <25 malignant cells are noisy.",
        "l1_selected_absent": "Never selected: L1 zero is not absence of signal.",
        "unoriented_all_zero": "Unoriented (beta 0 everywhere): raw sign is arbitrary.",
        "unoriented_sign_flip": "Unoriented (beta flips sign).",
    }
    for flag in str(idx["caveat_flags"]).split(";"):
        if flag in short:
            ax.text(0.0, yy, "• " + short[flag], fontsize=8, color=INK2, va="top", transform=ax.transAxes)
            yy -= 0.04
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.945, "card {}  |  descriptive; transductive Stage 1 basis; seven supported patients".format(cid), fontsize=8.5, color=INK2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# findings, manifest, ledger
# --------------------------------------------------------------------------------------


def _pp(v: float) -> str:
    return "n/a" if v is None or pd.isna(v) else "{:+.2f}".format(100 * float(v))


def write_findings(paths: Paths, tables: Dict[str, pd.DataFrame], checks: pd.DataFrame, shortlist: pd.DataFrame, runtime: float, head: str, part_seconds: float = 0.0) -> Path:
    index = tables["index"]
    lift = tables["patient_lift"]
    lines = ["# D18 factor cards (factor_cards_v1): findings", ""]
    lines.append("Generated by `{}` (git {}). Descriptive; Stage 1 is transductive (fit on all 13 patients), so LOPO lift is label transfer on a transductive basis.".format(SCRIPT_REL, head))
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    reps = index.drop_duplicates("representation_id")
    lines.append("- Representations: {} (shortlist reasons: {}).".format(len(reps), ", ".join("{} {}".format(k, v) for k, v in shortlist["shortlist_reasons"].str.split(";").explode().value_counts().items())))
    lines.append("- Cards: {} ({} non-degenerate; degenerate = zero score variance or zero loadings). Scale outliers (score SD >10x from the representation median; a caveat, not degeneracy): {}.".format(
        len(index), int((~index["degenerate_flag"].astype(bool)).sum()), int(index["caveat_flags"].str.contains("scale_outlier").sum())))
    lines.append("- Selected drivers (active in >=1 supported fold, fit not dense): {}; cards in dense fits: {}.".format(int(index["is_selected_driver"].astype(bool).sum()), int(index["fit_dense"].astype(bool).sum())))
    act = index.loc[index["n_active_folds"] > 0]
    lines.append("- Active in >=1 supported fold: {} cards; oriented {}, sign-flip (unoriented) {}.".format(len(act), int((act["orientation_status"] == "oriented").sum()), int((act["orientation_status"] == "unoriented_sign_flip").sum())))
    drv = index.loc[index["is_selected_driver"].astype(bool)]
    lines.append("- Among selected drivers: low determinacy {}, small eigengap {}, degenerate {}, scale outlier {}, near-duplicate oblique {}.".format(
        int(drv["caveat_flags"].str.contains("low_determinacy").sum()),
        int(drv["caveat_flags"].str.contains("small_eigengap").sum()),
        int(drv["degenerate_flag"].astype(bool).sum()),
        int(drv["caveat_flags"].str.contains("scale_outlier").sum()),
        int(drv["caveat_flags"].str.contains("near_duplicate_oblique").sum()),
    ))
    never = index.loc[(index["n_active_folds"] == 0) & (~index["degenerate_flag"].astype(bool))]
    strong_never = never.loc[never["max_univariate_lift"] >= 0.05]
    lines.append("- Never-selected non-degenerate cards whose univariate held-out lift reaches >=5 pp in at least one supported patient: {} of {}.".format(len(strong_never), len(never)))
    lines.append("")
    lines.append("## Regression anchors")
    lines.append("")
    lines.append(fc._md_table(["check", "n", "max abs diff", "tolerance", "passed"], [[r["check_id"], r["n_compared"], "{:.3g}".format(r["max_abs_diff"]), r["tolerance"], r["passed"]] for _, r in checks.iterrows()]))
    lines.append("")
    for name, c in (("IFN-alpha PCA40 factor 8 (PC8)", fc.card_id(IFNA, "pca", 40, 8)), ("IFN-alpha FA40 factor 7 (FA7)", fc.card_id(IFNA, "fa", 40, 7))):
        if c not in set(index["card_id"]):
            continue
        r = index.set_index("card_id").loc[c]
        lines.append("## {}".format(name))
        lines.append("")
        lines.append("- Orientation: {} (sign {:+d}); active in {}/{} supported folds; all-zero folds: {}.".format(r["orientation_status"], int(r["orientation_sign"]), int(r["n_active_folds"]), int(r["n_folds"]), r["all_zero_folds"] if isinstance(r["all_zero_folds"], str) and r["all_zero_folds"] else "none"))
        l = lift.loc[lift["card_id"] == c].set_index("patient")
        lines.append("- Univariate held-out lift (pp): " + ", ".join("{} {}".format(p, _pp(l.loc[p, "univariate_lift"])) for p in l.index) + ".")
        lines.append("- Fitted single-factor lift (pp): " + ", ".join("{} {}".format(p, _pp(l.loc[p, "fitted_single_factor_lift"])) for p in l.index) + ".")
        lines.append("- eta2 normal {:.3f} / malignant {:.3f}; determinacy {}; relative eigengap {}.".format(float(r["eta2_normal"]), float(r["eta2_malignant"]), "n/a" if pd.isna(r["determinacy"]) else "{:.3f}".format(float(r["determinacy"])), "n/a" if pd.isna(r["eigengap_rel"]) else "{:.3f}".format(float(r["eigengap_rel"]))))
        g = tables["genes"]
        g = g.loc[(g["card_id"] == c) & (g["ranked_by"] == "loading")]
        for pole in ("malignant", "normal"):
            gg = g.loc[g["pole"] == pole].sort_values("rank").head(6)
            lines.append("- {} pole (oriented loading): {}.".format(pole.capitalize(), ", ".join("{} {:+.3f}".format(x["gene"], x["oriented_loading"]) for _, x in gg.iterrows())))
        lines.append("- Caveats: {}.".format(r["caveat_flags"].replace(";", ", ")))
        lines.append("")
    lines.append("## Reading rules")
    lines.append("")
    lines.append("- Orientation is sign(beta) of the globally selected LOPO probe; positive = toward malignant. Unoriented cards (beta 0 everywhere, or sign flips) show raw loadings with an arbitrary sign.")
    lines.append("- beta is always read beside the univariate held-out lift and the selection frequency: an L1 zero at a fixed C is not absence of signal.")
    lines.append("- Matches across methods/K use score correlation (Hungarian at the same K), many-to-one R2, principal angles and Tucker phi; never the factor index.")
    lines.append("- FA/FactoSig loadings are generative; promax loadings are a pattern matrix (structure correlations and Phi are listed on promax cards).")
    lines.append("- Ablation dAP and fitted single-factor lift are attributions inside the fitted model, not causal effects.")
    lines.append("- ADT: CLR per cell of raw counts; within-label Spearman. Every malignant cell is recipient-origin and patient is nested in batch.")
    lines.append("")
    lines.append("Runtime of this invocation: {:.0f} s; summed per-representation worker time of the parts: {:.0f} s.".format(runtime, part_seconds))
    path = paths.scorecards / "factor_cards_findings.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def input_records(paths: Paths, shortlist: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    def rec(p: Path) -> Dict[str, object]:
        st = p.stat()
        return {"path": paths.rel(p), "size_bytes": int(st.st_size), "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}

    cfg = study_config()
    items = {
        "layer4_5_selected_lopo_coefficients": paths.exp / "analysis" / "cache" / "stage2_diagnostic" / "layer4_5" / "selected_lopo_coefficients.pkl.gz",
        "controls_lopo_scorecard": paths.exp / "analysis" / "scorecards" / "old34_controls_fullgrid_v1" / "stage2_sharedness_lopo_scorecard.csv",
        "controls_lopo_coefficient_paths": paths.exp / "stage2_supervised" / "multiobjective" / "old34_controls_fullgrid_v1" / "sharedness_lopo" / "coefficient_paths.csv",
        "cohort_cache": paths.exp / cfg.get("cohort_cache_relpath", ""),
        "d1_atlas_representation_table": paths.diag / "old34_d1_atlas_representation_table.csv",
        "d1_atlas_supported_patient_metrics": paths.diag / "old34_d1_atlas_supported_patient_metrics.csv",
        "d14_calls": paths.diag / "patient_factor_overlap_v1" / "patient_factor_overlap_calls.csv",
    }
    out = {k: rec(v) for k, v in items.items() if v.is_file()}
    out["stage1_dr"] = {"path": "stage1_dr/panel_id=<panel>/method=<method>/k=<K>/seed=42/{scores,loadings}.npy", "n_panels": int(shortlist["stage0_panel_id"].nunique())}
    out["panel_genes"] = {"path": "preprocessing/panels/stage0_panel_genes/<panel>.json"}
    out["gene_z"] = {"path": "stage1_direct_gene/{old34_direct_gene_all_v1,old34_controls_direct_gene_v1}/panel_id=<panel>/transform=zscore/features.npz (promax structure correlations)"}
    return out


def write_ledger(paths: Paths) -> Path:
    rows = []
    kinds = {".csv": "table", ".json": "manifest", ".md": "doc", ".png": "figure", ".pdf": "figure"}
    for root in (paths.scorecards, paths.figures):
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.parent.name != "cards" and p.name != "factor_cards_artifact_ledger.csv":
                rows.append({"path": paths.rel(p), "kind": kinds.get(p.suffix, "other"), "size_bytes": p.stat().st_size})
    n_cards = len(list((paths.scorecards / "cards").glob("*.md")))
    rows.append({"path": paths.rel(paths.scorecards / "cards") + "/<card_id>.md", "kind": "card_md ({} files)".format(n_cards), "size_bytes": sum(p.stat().st_size for p in (paths.scorecards / "cards").glob("*.md"))})
    path = paths.scorecards / "factor_cards_artifact_ledger.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def merge_and_finish(paths: Paths, shortlist: pd.DataFrame, args: argparse.Namespace, t_start: float, compute_seconds: float) -> int:
    log("merging {} representation parts".format(len(shortlist)))
    try:
        tables = read_parts(paths, shortlist)
    except Exception as exc:
        (paths.cache / "merge_failed").write_text("{}\n{}".format(exc, traceback.format_exc()))
        log("merge failed: {} (recover with --merge-only after finishing parts)".format(exc))
        return 2
    merge_flag = paths.cache / "merge_failed"
    if merge_flag.exists():
        merge_flag.unlink()
    paths.scorecards.mkdir(parents=True, exist_ok=True)
    counts = fc.write_tables(tables, paths.scorecards)
    shortlist.to_csv(paths.scorecards / "factor_cards_shortlist.csv", index=False)
    cards_dir = paths.scorecards / "cards"
    if cards_dir.exists():
        for old in cards_dir.glob("*.md"):
            old.unlink()
    n_md = fc.render_all_cards(tables, cards_dir)
    log("rendered {} card markdown files".format(n_md))
    checks = regression_checks(paths, tables)
    checks.to_csv(paths.scorecards / "factor_cards_regression_checks.csv", index=False)
    for c, name in ((fc.card_id(IFNA, "pca", 40, 8), "D18_ifna_pca40_factor008_card"), (fc.card_id(IFNA, "fa", 40, 7), "D18_ifna_fa40_factor007_card")):
        if c in set(tables["index"]["card_id"]):
            title = "IFN-α response | {} K=40 | factor {}".format("PCA" if "__pca__" in c else "FA", fc.parse_card_id(c)["factor"])
            card_figure(tables, c, paths.figures / "{}.png".format(name), title)
    (paths.scorecards / "bundle_schema_v1.md").write_text(fc.bundle_schema_markdown())
    head, dirty = git_state()
    runtime = time.time() - t_start
    part_seconds = 0.0
    for _, r in shortlist.iterrows():
        try:
            part_seconds += float(json.loads((part_dir(paths, r["stage0_panel_id"], r["stage1_method"], int(r["k"])) / "_done.json").read_text()).get("seconds", 0.0))
        except (OSError, ValueError):
            pass
    manifest = {
        "schema_version": fc.SCHEMA_VERSION,
        "slug": SLUG,
        "analysis_id": "D18",
        "task": "W5.1",
        "inputs": input_records(paths, shortlist),
        "git_state": {"head": head, "dirty": dirty},
        "created_by": SCRIPT_REL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "shortlist_mode": args.shortlist,
            "out_root": "<out-root>" if args.out_root else None,
            "top_genes_per_pole": args.top_genes,
            "adt_top_markers": args.adt_top,
            "adt_normalization": "CLR per cell: log1p(count) - mean over all 170 ADT features (incl. isotypes)",
            "adt_correlation": "Spearman within label; pooled and patient-stratified (normal >=200, malignant >=20 cells per patient)",
            "eta2_strata": {"normal_min_cells": fc.NORMAL_MIN_CELLS, "malignant_min_cells": fc.MALIGNANT_MIN_CELLS},
            "supported_patients": list(study_config()["supported_patients"]),
            "orientation": "sign(beta) over active supported folds; 0 when never active or sign flips",
            "univariate_lift_sign": "training-fold standardized malignant-minus-normal mean gap",
            "dense_fit_threshold_nonzero_fraction": fc.DENSE_NONZERO_FRACTION,
            "match_abs_r": fc.MATCH_ABS_R,
            "low_determinacy": fc.LOW_DETERMINACY,
            "small_eigengap": fc.SMALL_EIGENGAP,
            "scale_outlier_ratio": fc.SCALE_OUTLIER_RATIO,
            "near_duplicate_r": fc.NEAR_DUPLICATE_R,
            "stage2_selection": "global_selected per representation (layer4_5 cache; controls rebuilt with the D1 max cell_weighted_auprc rule)",
            "c_policy": "global_selected (stored; no refit)",
            "n_jobs": args.n_jobs,
        },
        "counts": {**counts, "n_cards": int(len(tables["index"])), "n_representations": int(tables["index"]["representation_id"].nunique()), "n_card_md": n_md},
        "regression_checks_passed": bool(checks["passed"].all()) if not checks.empty else None,
        "runtime_seconds": round(runtime, 1),
        "compute_seconds": round(compute_seconds, 1),
        "merge_only": bool(args.merge_only),
        "part_worker_seconds_sum": round(part_seconds, 1),
    }
    (paths.scorecards / fc.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_findings(paths, tables, checks, shortlist, runtime, "{}{}".format(head, "+dirty" if dirty else ""), part_seconds)
    write_ledger(paths)
    errors = fc.validate_bundle(paths.scorecards)
    if errors:
        log("bundle validation FAILED: {}".format(errors))
        return 3
    log("bundle valid; regression checks passed: {}".format(manifest["regression_checks_passed"]))
    if not checks.empty and not checks["passed"].all():
        log("failed checks: {}".format(list(checks.loc[~checks["passed"], "check_id"])))
    log("done in {:.0f}s".format(runtime))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", default=None, help="override the experiment dir (default: sc_classification.utils.paths.experiment_dir())")
    parser.add_argument("--shortlist", choices=["full", "ifna", "smoke"], default="full",
                        help="full = the canonical bundle; ifna and smoke need --out-root")
    parser.add_argument("--out-root", default=None,
                        help="write every output under DIR/{scorecards,figures,cache}/stage2_diagnostic/%s/ instead of EXP/analysis (EXP is only read)" % SLUG)
    parser.add_argument("--n-jobs", type=int, default=6, help="worker processes over panels (max {})".format(MAX_JOBS))
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--adt-top", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="skip representations whose parts are complete")
    parser.add_argument("--merge-only", action="store_true", help="only merge finished parts, render, check and validate")
    parser.add_argument("--no-adt", action="store_true")
    parser.add_argument("--no-matching", action="store_true")
    args = parser.parse_args(argv)
    if args.shortlist != "full" and not args.out_root:
        # the merge replaces the bundle tables, cards and manifest with this shortlist only
        raise SystemExit("--shortlist {} needs --out-root DIR: a partial shortlist must never merge into the "
                         "canonical {} bundle".format(args.shortlist, SLUG))
    args.n_jobs = max(1, min(int(args.n_jobs), MAX_JOBS))
    t_start = time.time()
    exp = Path(os.path.abspath(args.experiment_dir)) if args.experiment_dir else experiment_dir()
    paths = Paths(exp, Path(args.out_root) if args.out_root else None)
    for p in (paths.scorecards, paths.figures, paths.cache, paths.parts):
        p.mkdir(parents=True, exist_ok=True)
    shortlist = build_shortlist(paths, args.shortlist)
    log("shortlist: {} representations over {} panels ({})".format(len(shortlist), shortlist["stage0_panel_id"].nunique(), args.shortlist))
    compute_seconds = 0.0
    if not args.merge_only:
        todo = shortlist
        if args.resume:
            done = [part_done(paths, r["stage0_panel_id"], r["stage1_method"], int(r["k"])) for _, r in shortlist.iterrows()]
            todo = shortlist.loc[~np.asarray(done, dtype=bool)]
            log("resume: {} done, {} to compute".format(int(np.sum(done)), len(todo)))
        if len(todo):
            coefs = load_coefficients(paths, todo)
            atlas = pd.read_csv(paths.diag / "old34_d1_atlas_supported_patient_metrics.csv")
            expected = {rid: dict(zip(g["heldout_patient"], g["heldout_auprc"])) for rid, g in atlas.groupby("representation_id")}
            options = {"top_genes": args.top_genes, "adt_top": args.adt_top, "no_adt": args.no_adt, "no_matching": args.no_matching,
                       "out_root": str(paths.out_root) if paths.out_root else None}
            jobs = []
            for panel, g in todo.groupby("stage0_panel_id", sort=False):
                targets = [(r["stage1_method"], int(r["k"]), r["shortlist_reasons"], r["panel_label"]) for _, r in g.iterrows()]
                rids = [fc.representation_id(panel, m, k) for m, k, _, _ in targets]
                jobs.append((panel, targets, coefs.loc[coefs["representation_id"].isin(rids)], {r: expected[r] for r in rids if r in expected}))
            # big panels first
            jobs.sort(key=lambda j: -len(json.loads(paths.panel_genes(j[0]).read_text())["genes"]))
            t0 = time.time()
            results: List[Dict[str, object]] = []
            if args.n_jobs > 1 and len(jobs) > 1:
                from joblib import Parallel, delayed

                out = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(delayed(run_panel)(str(exp), *j, options) for j in jobs)
                for o in out:
                    results += o
            else:
                for j in jobs:
                    results += run_panel(str(exp), *j, options)
                    log("panel {} done".format(j[0]))
            compute_seconds = time.time() - t0
            failed = [r for r in results if r["status"] != "done"]
            log("computed {} representations in {:.0f}s; failed {}".format(len(results) - len(failed), compute_seconds, len(failed)))
            if failed:
                for r in failed:
                    log("FAILED {}: {}".format(r["representation_id"], r["error"]))
                    print(r.get("trace", ""))
                return 1
    return merge_and_finish(paths, shortlist, args, t_start, compute_seconds)


if __name__ == "__main__":
    sys.exit(main())
