"""D17 Stage 1 basis-variant pilot on IFN-alpha (task W2.1; slug basis_variant_pilot_ifna_v1).

Question: does held-out-patient transfer of the IFN-alpha PCA/FA probe depend on the held-out
patient (or the highest-prevalence patients, or the malignant cells) being inside the
label-blind Stage 1 fit? Stage 1 is refit on different cell sets, the other cells are projected,
and Stage 2 LOPO is rerun on each basis through W3's regularization contract.

Arms (per Stage 1 method in {pca, fa} and K; seed 42 throughout):
  transductive_stored  stored stage1_dr scores (the anchor; must reproduce the stored
                       sharedness_lopo per-patient lifts before anything else is reported)
  transductive_refit   refit on all cells with this module (sanity floor vs the stored basis). Its
                       scores are exact projections; the stored PCA scores are randomized-SVD
                       U * S, so for PCA this arm, not the stored one, isolates fit-scope effects
  inductive            per held-out supported patient h: standardize + fit on patients != h,
                       project h (H-S1-06; ALT-TRANS)
  normal_only          per held-out h: fit on the normal cells of patients != h, project all
                       cells (do malignant cells shape the discriminative axis?)
  normal_only_all      normal cells of all patients (alignment only; it uses every patient's
                       labels, so no Stage 2 is run on it)
  leave_out            drop P02 / P03 / P04 / all three (highest prevalence) and, as contrasts,
                       P01 / P09 / P06 / P01+P09 (largest by cell count); refit, project all
                       (loading leverage, H-S1-05)
  leave_out_random     size-matched random cell drops for every leave_out set (alignment only):
                       the sampling floor for the leverage contrast (3 seeds)
  leave_out_label_random  random drops matched on both the number of dropped cells and the
                       number of dropped malignant cells (alignment only, 20 seeds): leverage
                       beyond these is not explained by how many malignant cells leave the fit;
                       the observed leave_out value gets an empirical rank among them
  normal_only_random   random drop of as many cells as there are malignant cells (alignment only)
  balanced_weighted    patient-equal weights over all 13 patients: PCA = weighted covariance +
                       eigh; FA = FA on the weighted covariance (stage1_refit.fa_from_covariance)
  balanced_subsample   patient-stratified equal-n subsample (5 seeds), project all cells

Stage 2 per (basis, held-out supported patient), via regularization_contract (R5):
  l1_fixed  L1 at C = 1e-4 (the stored atlas C for PCA40/FA40 and PCA20/FA20)
  l2_fixed  L2 at C = 1e-4
  l1_tuned  L1 with C tuned per outer fold by inner LOPO over the supported training patients
  l2_tuned  L2, same
Stage 2 training cells are the standard LOPO folds in every arm (only the Stage 1 fit set
changes), so c_scaled_to_n returns the stored C; it is recorded, not run as a separate arm.
Alignment of every basis to the stored transductive basis: Hungarian matching on |r| of scores,
many-to-one R^2 of the reference axis (PC8 / FA7 at K = 40) on all variant factors, principal
angles between loading subspaces (top 8 and all K) and the containment of the reference axis.

Parts (--parts, default anchor,bases,align,stage2,report). --resume skips finished bases and
finished Stage 2 units (one unit = basis x held-out patient, all policies); empty, corrupt or
out-of-date files are redone. --out-root redirects all outputs; --smoke requires it (smoke units
use a thinned C grid and would otherwise land in the canonical slug).

Outputs (new slug, R3), relative to the experiment dir:
  analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/  tables, findings, manifest, ledger
  analysis/figures/stage2_diagnostic/basis_variant_pilot_ifna_v1/     D17A-D17C (.png, .pdf)
  analysis/cache/stage2_diagnostic/basis_variant_pilot_ifna_v1/       stage2_units/, stored-row extract,
                                                                      bases/ (default --bases-location cache)
With --bases-location stage1_dr_variants the bases go to stage1_dr_variants/<slug>/ instead (R3).

Example (repo root):
  python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py --resume --n-jobs 12
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

add_study_module_paths(subdirs=(), workstreams=("W3_dr_value_regularization", "W2_stage1_basis_variants"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import regularization_contract as rc  # noqa: E402
import stage1_refit as s1  # noqa: E402

SLUG = "basis_variant_pilot_ifna_v1"
PANEL = "single_geneset__hallmark_interferon_alpha_response"
FEATURES_REL = f"stage1_direct_gene/old34_direct_gene_all_v1/panel_id={PANEL}/transform=zscore"
STORED_LOPO_REL = "stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv"
METHODS = ("pca", "fa")
DEFAULT_KS = (40,)
FIXED_C = 1e-4
SEED = 42
#: The D19 anchor: the largest-margin factor in every supported fold (0-based column).
EXPECTED_REF_FACTOR = {("pca", 40): 7, ("fa", 40): 6}
HIGH_PREVALENCE = ("P02", "P03", "P04")
DROP_SETS: Tuple[Tuple[str, ...], ...] = (
    ("P02",), ("P03",), ("P04",), ("P02", "P03", "P04"),
    ("P01",), ("P09",), ("P06",), ("P01", "P09"),
)
SUBSAMPLE_SEEDS = (42, 43, 44, 45, 46)
RANDOM_DROP_SEEDS = (1, 2, 3)
#: Composition-matched (cells and malignant cells) random-drop controls for the leverage contrast.
LABEL_MATCHED_DROP_SEEDS = tuple(range(101, 121))
POLICIES_FULL = ("l1_fixed", "l2_fixed", "l1_tuned", "l2_tuned")
POLICIES_FIXED = ("l1_fixed", "l2_fixed")
POLICY_PENALTY = {"l1_fixed": "l1", "l2_fixed": "l2", "l1_tuned": "l1", "l2_tuned": "l2"}
#: Anchor gate: |reproduced - stored lift| <= ANCHOR_GATE_PP with the same nonzero support. Rows above
#: ANCHOR_STRICT_PP are flagged. The stored grids ran on cuML (QN, float32); the contract fits in
#: float64 with sklearn-objective solvers, so knife-edge L1 folds and small-patient L2 folds can
#: differ at the 0.01-0.1 pp level with the same support. (0.1 pp was set after the first K=10/20
#: anchor run showed two rows at 0.052 and 0.068 pp; K = 40 passes the strict 0.05 pp level.)
ANCHOR_STRICT_PP = 0.05
ANCHOR_GATE_PP = 0.10
TOP_M = 8
MAX_WORKERS = 12
DEFAULT_PARTS = ("anchor", "bases", "align", "stage2", "report")
ARM_ORDER = (
    "transductive_stored", "transductive_refit", "inductive", "normal_only", "balanced_weighted",
    "balanced_subsample", "leave_out", "normal_only_all", "normal_only_random", "leave_out_random",
    "leave_out_label_random",
)
ARM_LABELS = {
    "transductive_stored": "transductive (stored)",
    "transductive_refit": "transductive (refit)",
    "inductive": "inductive LOPO",
    "normal_only": "normal-only, inductive",
    "balanced_weighted": "patient-balanced (weighted)",
    "balanced_subsample": "patient-balanced (subsample)",
    "leave_out": "leave patients out",
}
UNIT_SCHEMA = 2


# --------------------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json_done(path: Path) -> Optional[Dict[str, Any]]:
    """A unit file that is missing, empty, unparsable or not marked done counts as unfinished."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) and obj.get("status") == "done" else None
    except Exception:
        return None


def p20(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, 0.2)) if arr.size else float("nan")


def median(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def fid(index: Optional[int]) -> str:
    return "" if index is None or index < 0 else f"factor_{int(index) + 1:03d}"


def factor_label(method: str, index: int) -> str:
    return f"{'PC' if method == 'pca' else 'FA'}{int(index) + 1}"


# --------------------------------------------------------------------------------------
# data (cached per process, so joblib workers load the inputs once)
# --------------------------------------------------------------------------------------

_DATA: Dict[str, Any] = {}


def load_data(exp_str: str) -> Dict[str, Any]:
    if exp_str in _DATA:
        return _DATA[exp_str]
    import scipy.sparse as sp

    exp = Path(exp_str)
    fdir = exp / FEATURES_REL
    X = sp.load_npz(fdir / "features.npz").toarray().astype(np.float64)
    genes = [str(g) for g in json.loads((fdir / "feature_names.json").read_text(encoding="utf-8"))]
    meta = pd.read_csv(cell_metadata(), usecols=["cell_id", "y_true", "patient"])
    y = meta["y_true"].to_numpy().astype(int)
    pat = meta["patient"].astype(str).to_numpy()
    if X.shape != (y.size, len(genes)):
        raise AssertionError(f"features {X.shape} vs metadata {y.size} cells x {len(genes)} genes")
    _DATA[exp_str] = {"X": X, "genes": genes, "y": y, "patients": pat, "stored": {}}
    return _DATA[exp_str]


def stored_dir(exp: Path, method: str, k: int) -> Path:
    return exp / "stage1_dr" / f"panel_id={PANEL}" / f"method={method}" / f"k={int(k)}" / f"seed={SEED}"


def stored_scores(exp_str: str, method: str, k: int) -> np.ndarray:
    data = load_data(exp_str)
    key = (method, int(k))
    if key not in data["stored"]:
        scores = np.load(stored_dir(Path(exp_str), method, k) / "scores.npy").astype(np.float64)
        if scores.shape != (data["y"].size, int(k)):
            raise AssertionError(f"stored {method} K={k} scores have shape {scores.shape}")
        data["stored"][key] = scores
    return data["stored"][key]


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------


class Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.smoke and not args.out_root:
            # smoke units use a thinned C grid under the same unit names as the full run
            raise SystemExit("--smoke needs --out-root (smoke outputs never go into the slug dirs)")
        self.exp = experiment_dir()
        self.exp_str = str(self.exp)
        root = Path(args.out_root).resolve() if args.out_root else self.exp / "analysis"
        self.out_root = root
        self.tables = root / "scorecards" / "stage2_diagnostic" / SLUG
        self.figures = root / "figures" / "stage2_diagnostic" / SLUG
        self.cache = root / "cache" / "stage2_diagnostic" / SLUG
        if args.bases_location == "stage1_dr_variants" and not args.out_root:
            self.bases = self.exp / "stage1_dr_variants" / SLUG
        else:
            self.bases = self.cache / "bases"
        self.units = self.cache / "stage2_units"
        for path in (self.tables, self.figures, self.cache, self.bases, self.units):
            path.mkdir(parents=True, exist_ok=True)
        data = load_data(self.exp_str)
        self.y, self.patients, self.genes = data["y"], data["patients"], data["genes"]
        self.supported = rc.supported_patients(self.y, self.patients, min_malignant=rc.DEFAULT_MIN_MALIGNANT)
        try:
            configured = list(study_config().get("supported_patients", []))
        except Exception:
            configured = []
        if configured and configured != self.supported:
            raise AssertionError(f"support rule gives {self.supported}, study_config lists {configured}")
        self.n_jobs = max(1, min(int(args.n_jobs), MAX_WORKERS))
        self.methods = [m for m in args.methods.split(",") if m]
        self.ks = [int(k) for k in args.ks.split(",") if k]
        self.full_ks = {int(k) for k in args.full_ks.split(",") if k}
        smoke = bool(args.smoke)
        # held-out patients for every arm's Stage 2 (and the inductive / normal-only fits)
        self.heldouts = [p for p in (("P02", "P07") if smoke else self.supported)]
        self.drop_sets = (("P02",), ("P09",)) if smoke else DROP_SETS
        self.sub_seeds = SUBSAMPLE_SEEDS[:1] if smoke else SUBSAMPLE_SEEDS
        self.rand_seeds = RANDOM_DROP_SEEDS[:1] if smoke else RANDOM_DROP_SEEDS
        self.label_seeds = LABEL_MATCHED_DROP_SEEDS[:2] if smoke else LABEL_MATCHED_DROP_SEEDS
        grid = list(rc.DEFAULT_C_GRID)
        self.c_grid = grid[:: max(1, int(args.c_grid_stride))] if smoke else grid
        if smoke and not any(np.isclose(c, FIXED_C) for c in self.c_grid):
            self.c_grid.append(FIXED_C)
        self.timings: Dict[str, float] = {}
        self.session_records: Dict[str, Dict[str, Any]] = {}

    def rel(self, path: Path) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(self.exp.resolve()).as_posix()
        except ValueError:
            return "<out-root>/" + path.relative_to(self.out_root).as_posix()


def build_plan(ctx: Ctx) -> List[Dict[str, Any]]:
    """One spec per Stage 1 basis, with the Stage 2 held-out patients and policies to run on it."""
    specs: List[Dict[str, Any]] = []
    n_mal = int(ctx.y.sum())
    counts = pd.Series(ctx.patients).value_counts()
    for method in ctx.methods:
        for k in ctx.ks:
            full = k in ctx.full_ks
            P4, P2 = list(POLICIES_FULL), list(POLICIES_FIXED)

            def add(arm: str, variant: str, fit: Dict[str, Any], stage2: Sequence[str], policies: Sequence[str]) -> None:
                specs.append({
                    "basis_id": f"{arm}__{method}{k}__{variant}",
                    "arm": arm, "method": method, "k": int(k), "variant": variant, "fit": fit,
                    "stage2_heldouts": list(stage2) if policies else [], "policies": list(policies),
                })

            add("transductive_stored", "stored", {"kind": "stored"}, ctx.heldouts, P4)
            # all four policies: the refit is the exact-projection twin of the stored basis (see _ref_arrays)
            add("transductive_refit", "all", {"kind": "rows"}, ctx.heldouts, P4)
            for h in ctx.heldouts:
                add("inductive", f"heldout-{h}", {"kind": "rows", "exclude": [h]}, [h], P4)
            for h in ctx.heldouts:
                add("normal_only", f"heldout-{h}", {"kind": "rows", "exclude": [h], "label": 0}, [h], P4)
            add("normal_only_all", "all", {"kind": "rows", "label": 0}, [], [])
            for seed in ctx.rand_seeds:
                add("normal_only_random", f"drop{n_mal}-s{seed}", {"kind": "random_drop", "n_drop": n_mal, "seed": seed}, [], [])
            for ds in ctx.drop_sets:
                tag = "-".join(ds)
                add("leave_out", f"drop-{tag}", {"kind": "rows", "exclude": list(ds)}, ctx.heldouts, P4 if full else P2)
                n_drop = int(sum(int(counts.get(p, 0)) for p in ds))
                for seed in ctx.rand_seeds:
                    add("leave_out_random", f"drop-{tag}-random-s{seed}",
                        {"kind": "random_drop", "n_drop": n_drop, "seed": seed, "matches": tag}, [], [])
                n_drop_mal = int(sum(int(ctx.y[ctx.patients == p].sum()) for p in ds))
                for seed in ctx.label_seeds:
                    add("leave_out_label_random", f"drop-{tag}-labelmatched-s{seed}",
                        {"kind": "label_matched_drop", "n_drop": n_drop, "n_drop_malignant": n_drop_mal, "seed": seed,
                         "matches": tag}, [], [])
            add("balanced_weighted", "all", {"kind": "weighted"}, ctx.heldouts, P4)
            for seed in ctx.sub_seeds:
                pol = P4 if (full and method == "fa") else P2
                add("balanced_subsample", f"s{seed}", {"kind": "subsample", "seed": seed}, ctx.heldouts, pol)
    return specs


def fit_rows(fit: Dict[str, Any], y: np.ndarray, patients: np.ndarray) -> np.ndarray:
    """Integer row index of the Stage 1 fit cells for a spec (deterministic)."""
    kind = fit["kind"]
    if kind in ("stored", "weighted"):
        return np.arange(y.size)
    if kind == "rows":
        mask = s1.fit_scope_mask(patients, exclude_patients=fit.get("exclude", ()), labels=y, label_value=fit.get("label"))
        return np.flatnonzero(mask)
    if kind == "random_drop":
        return np.flatnonzero(s1.random_drop_mask(y.size, int(fit["n_drop"]), seed=int(fit["seed"])))
    if kind == "label_matched_drop":
        return np.flatnonzero(s1.label_matched_drop_mask(y, int(fit["n_drop"]), int(fit["n_drop_malignant"]), seed=int(fit["seed"])))
    if kind == "subsample":
        return s1.stratified_subsample(patients, seed=int(fit["seed"]))
    raise ValueError(f"unknown fit kind {kind}")


def basis_dir(ctx: Ctx, spec: Dict[str, Any]) -> Path:
    return (ctx.bases / f"arm={spec['arm']}" / f"panel_id={PANEL}" / f"method={spec['method']}" / f"k={spec['k']}"
            / f"seed={SEED}" / f"variant={spec['variant']}")


# --------------------------------------------------------------------------------------
# part: bases
# --------------------------------------------------------------------------------------


def _fit_one(exp_str: str, spec: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    data = load_data(exp_str)
    y, pat = data["y"], data["patients"]
    rows = fit_rows(spec["fit"], y, pat)
    weights = s1.patient_balanced_weights(pat, rows) if spec["fit"]["kind"] == "weighted" else None
    info = {
        "arm": spec["arm"], "variant": spec["variant"], "panel_id": PANEL, "fit_spec": spec["fit"],
        "fit_patients": sorted(set(pat[rows].tolist())),
        "n_fit_malignant": int(y[rows].sum()),
        "weighting": "patient_equal (1 / (n_patients * n_patient_cells))" if weights is not None else "none",
        "input": FEATURES_REL + "/features.npz",
    }
    basis = s1.fit_basis(data["X"], spec["method"], spec["k"], rows=rows, weights=weights, seed=SEED,
                         gene_names=data["genes"], fit_info=info)
    basis.fit_info["fit_seconds"] = time.perf_counter() - t0
    basis.save(out_dir)
    return {"basis_id": spec["basis_id"], "seconds": time.perf_counter() - t0, "n_fit_cells": int(rows.size),
            "n_iter": basis.fit_info.get("n_iter")}


def run_bases(ctx: Ctx, specs: List[Dict[str, Any]]) -> pd.DataFrame:
    from joblib import Parallel, delayed

    todo = []
    for spec in specs:
        if spec["fit"]["kind"] == "stored":
            continue
        out = basis_dir(ctx, spec)
        if ctx.args.resume and s1.Stage1Basis.is_saved(out):
            continue
        todo.append(spec)
    n_fit = sum(1 for s in specs if s["fit"]["kind"] != "stored")
    print(f"[bases] {n_fit} bases; {n_fit - len(todo)} resumed, {len(todo)} to fit (n_jobs={ctx.n_jobs})", flush=True)
    # FA fits first (slowest) for load balance
    todo.sort(key=lambda s: (s["method"] != "fa", -s["k"], s["basis_id"]))
    t0 = time.perf_counter()
    Parallel(n_jobs=ctx.n_jobs, verbose=0)(delayed(_fit_one)(ctx.exp_str, s, str(basis_dir(ctx, s))) for s in todo)
    print(f"[bases] fitted {len(todo)} in {time.perf_counter() - t0:.1f} s", flush=True)
    return bases_table(ctx, specs)


def bases_table(ctx: Ctx, specs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        if spec["fit"]["kind"] == "stored":
            continue
        b = s1.Stage1Basis.load(basis_dir(ctx, spec))
        fi = b.fit_info
        rows.append({
            "basis_id": spec["basis_id"], "arm": spec["arm"], "stage1_method": spec["method"], "effective_k": spec["k"],
            "variant": spec["variant"], "fit_scope": spec["fit"]["kind"], "n_fit_cells": fi.get("n_fit_cells"),
            "n_fit_malignant": fi.get("n_fit_malignant"), "n_fit_patients": len(fi.get("fit_patients", [])),
            "fit_patients": ";".join(fi.get("fit_patients", [])), "weighting": fi.get("weighting"),
            "solver": fi.get("solver"), "n_iter": fi.get("n_iter"), "fit_seconds": fi.get("fit_seconds"),
            "n_zero_variance_factors_all_cells": None, "basis_dir": ctx.rel(basis_dir(ctx, spec)),
        })
    return pd.DataFrame(rows)


def basis_scores(ctx: Ctx, spec: Dict[str, Any]) -> np.ndarray:
    if spec["fit"]["kind"] == "stored":
        return stored_scores(ctx.exp_str, spec["method"], spec["k"])
    data = load_data(ctx.exp_str)
    return s1.Stage1Basis.load(basis_dir(ctx, spec)).transform(data["X"], data["genes"])


# --------------------------------------------------------------------------------------
# reference axis and part: align
# --------------------------------------------------------------------------------------


def _ref_arrays(exp_str: str, method: str, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alignment reference for the stored transductive basis: exact projection scores, common-scale
    loadings (genes x K) and factor importance order.

    FA: the stored scores (sklearn FA.fit_transform = transform, so they are the exact projection).
    PCA: the projection of the centred input onto the stored loadings. The stored PCA scores are
    sklearn's randomized-SVD U * S, which differ slightly from that projection (recorded in
    d17_reference_axes.csv); Stage 2 anchors still use the stored scores.
    """
    data = load_data(exp_str)
    key = ("ref", method, int(k))
    if key not in data["stored"]:
        L = np.load(stored_dir(Path(exp_str), method, k) / "loadings.npy").astype(np.float64)
        if method == "pca":
            X = data["X"]
            S = (X - X.mean(axis=0)) @ L
            order = np.arange(int(k))
        else:
            S = stored_scores(exp_str, method, k)
            order = np.argsort(-np.sum(L ** 2, axis=0), kind="stable")
        data["stored"][key] = (S, L, order)
    return data["stored"][key]


def reference(ctx: Ctx, method: str, k: int) -> Dict[str, Any]:
    """Stored transductive basis plus its reference axis: the factor with the largest all-cell
    standardized class-mean gap (PC8 / FA7 at K = 40, as in the D19 anchor)."""
    S, L, order = _ref_arrays(ctx.exp_str, method, k)
    stored = stored_scores(ctx.exp_str, method, k)
    Zs, _ = rc.standardize_train(stored)
    gap = Zs[ctx.y == 1].mean(0) - Zs[ctx.y == 0].mean(0)
    ref = int(np.argmax(np.abs(gap)))
    expected = EXPECTED_REF_FACTOR.get((method, int(k)))
    if expected is not None and ref != expected:
        raise AssertionError(f"{method}{k}: largest all-cell class-mean gap is {fid(ref)}, expected {fid(expected)}")
    r_proj = np.array([abs(np.corrcoef(stored[:, i], S[:, i])[0, 1]) if stored[:, i].std() > 0 and S[:, i].std() > 0 else np.nan
                       for i in range(int(k))])
    return {"scores": S, "loadings": L, "order": order, "ref_factor": ref, "gap": gap, "stored": stored, "r_proj": r_proj}


def _align_one(exp_str: str, spec: Dict[str, Any], bdir: str, j: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = load_data(exp_str)
    y, pat = data["y"], data["patients"]
    m, k = spec["method"], spec["k"]
    S, L, order = _ref_arrays(exp_str, m, k)
    basis = s1.Stage1Basis.load(bdir)
    Z = basis.transform(data["X"], data["genes"])
    fit_idx = fit_rows(spec["fit"], y, pat)
    kw = dict(ref_loadings=L, var_loadings=basis.loadings_common(), ref_order=order,
              var_order=basis.factor_variance_order(), top_m=TOP_M)
    on_all = s1.align_scores(S, Z, j, **kw)
    on_fit = s1.align_scores(S, Z, j, rows=fit_idx)
    match = {"ref_factor": j, "matched_factor": on_all["matched_factor"], "matched_r": on_all["matched_r"],
             "r_ref_vs_variant": on_all["r_ref_factor_vs_variant"]}
    subsample = spec["fit"]["kind"] == "subsample"
    row: Dict[str, Any] = {
        "basis_id": spec["basis_id"], "arm": spec["arm"], "stage1_method": m, "effective_k": k, "variant": spec["variant"],
        "matches_drop_set": spec["fit"].get("matches", ""), "n_fit_cells": int(fit_idx.size),
        "n_dropped_cells": None if subsample else int(y.size - fit_idx.size),
        "n_dropped_malignant": None if subsample else int(y.sum() - y[fit_idx].sum()),
        "ref_factor": fid(j), "ref_factor_label": factor_label(m, j),
    }
    for key in ("matched_factor", "best_factor"):
        row[f"{key}_all"] = fid(on_all[key])
    for key in ("matched_r", "matched_abs_r", "best_abs_r", "many_to_one_r2", "mean_matched_abs_r_all", f"mean_matched_abs_r_top{TOP_M}"):
        row[f"{key}_all"] = on_all.get(key)
    row["matched_factor_fit"] = fid(on_fit["matched_factor"])
    for key in ("matched_abs_r", "best_abs_r", "many_to_one_r2", "mean_matched_abs_r_all"):
        row[f"{key}_fit"] = on_fit.get(key)
    for key in (f"angle_top{TOP_M}_max_deg", f"angle_top{TOP_M}_mean_cos2", "angle_allK_max_deg", "angle_allK_mean_cos2",
                f"ref_axis_containment_top{TOP_M}", "ref_axis_containment_allK", "matched_loading_abs_cos",
                "matched_factor_importance_rank"):
        row[key] = on_all.get(key)
    row["axis_loss_fit"] = 1.0 - float(on_fit["many_to_one_r2"])
    row["n_zero_variance_factors"] = int(np.sum(Z.std(0) < 1e-9))
    return row, match


def stored_match_info(ctx: Ctx, spec: Dict[str, Any]) -> Dict[str, Any]:
    ref = reference(ctx, spec["method"], spec["k"])
    j = ref["ref_factor"]
    R = s1.score_correlation(ref["stored"], ref["stored"])
    return {"ref_factor": j, "matched_factor": j, "matched_r": 1.0, "r_ref_vs_variant": [float(v) for v in np.nan_to_num(R[j])]}


def run_align(ctx: Ctx, specs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Alignment of every basis to the stored transductive basis; also the Stage 2 match info."""
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    refs = {(m, k): reference(ctx, m, k) for m in ctx.methods for k in ctx.ks}
    match_info: Dict[str, Dict[str, Any]] = {}
    todo = []
    for spec in specs:
        if spec["fit"]["kind"] == "stored":
            match_info[spec["basis_id"]] = stored_match_info(ctx, spec)
        else:
            todo.append(spec)
    out = Parallel(n_jobs=ctx.n_jobs, verbose=0)(
        delayed(_align_one)(ctx.exp_str, spec, str(basis_dir(ctx, spec)), refs[(spec["method"], spec["k"])]["ref_factor"])
        for spec in todo
    )
    rows = []
    for spec, (row, match) in zip(todo, out):
        rows.append(row)
        match_info[spec["basis_id"]] = match
    table = pd.DataFrame(rows)
    atomic_write_csv(table, ctx.tables / "d17_alignment.csv")
    ref_rows = []
    for (m, k), r in refs.items():
        second = int(np.argsort(-np.abs(r["gap"]))[1])
        ref_rows.append({"stage1_method": m, "effective_k": k, "ref_factor": fid(r["ref_factor"]),
                         "ref_factor_label": factor_label(m, r["ref_factor"]), "ref_all_cell_std_gap": float(r["gap"][r["ref_factor"]]),
                         "second_factor": fid(second), "second_all_cell_std_gap": float(r["gap"][second]),
                         "alignment_reference": "stored scores" if m == "fa" else "projection onto stored loadings",
                         "ref_factor_r_stored_scores_vs_reference": float(r["r_proj"][r["ref_factor"]]),
                         "min_r_stored_scores_vs_reference": float(np.nanmin(r["r_proj"]))})
    atomic_write_csv(pd.DataFrame(ref_rows), ctx.tables / "d17_reference_axes.csv")
    atomic_write_json(match_info, ctx.cache / "d17_match_info.json")
    ctx.timings["align_s"] = time.perf_counter() - t0
    print(f"[align] {len(table)} bases aligned in {ctx.timings['align_s']:.1f} s", flush=True)
    return table, match_info


# --------------------------------------------------------------------------------------
# part: stage2
# --------------------------------------------------------------------------------------


def unit_id(spec: Dict[str, Any], heldout: str) -> str:
    return f"{spec['basis_id']}__{heldout}"


def unit_signature(ctx: Ctx, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema": UNIT_SCHEMA, "policies": list(spec["policies"]), "c_grid_n": len(ctx.c_grid), "fixed_C": FIXED_C,
            "contract_version": rc.CONTRACT_VERSION}


def _stage2_unit(exp_str: str, spec: Dict[str, Any], heldout: str, bdir: str, match: Dict[str, Any],
                 c_grid: Sequence[float], signature: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray]:
    t0 = time.perf_counter()
    data = load_data(exp_str)
    y, pat = data["y"], data["patients"]
    if spec["fit"]["kind"] == "stored":
        Z = stored_scores(exp_str, spec["method"], spec["k"])
    else:
        Z = s1.Stage1Basis.load(bdir).transform(data["X"], data["genes"])
    train = pat != heldout
    test = ~train
    j = int(match["matched_factor"])
    sign = 1.0 if float(match["matched_r"]) >= 0 else -1.0
    r_vec = np.abs(np.asarray(match["r_ref_vs_variant"], dtype=float))
    Zs_train, _ = rc.standardize_train(Z[train])
    n_train = int(train.sum())
    record: Dict[str, Any] = {
        "unit": unit_id(spec, heldout), "basis_id": spec["basis_id"], "arm": spec["arm"], "stage1_method": spec["method"],
        "effective_k": spec["k"], "variant": spec["variant"], "heldout_patient": heldout,
        "n_train": n_train, "n_train_malignant": int(y[train].sum()), "n_heldout": int(test.sum()),
        "n_heldout_malignant": int(y[test].sum()), "ref_factor": fid(int(match["ref_factor"])), "matched_factor": fid(j),
        "matched_r_all_cells": float(match["matched_r"]), "signature": signature,
    }
    policies = []
    probs = []
    for policy in spec["policies"]:
        pen = POLICY_PENALTY[policy]
        rec: Dict[str, Any] = {"policy": policy, "penalty": pen}
        if policy.endswith("_tuned"):
            tuned = rc.tune_c_inner_lopo(Z[train], y[train], pat[train], penalty=pen, c_grid=c_grid, n_jobs=1)
            C = float(tuned["best_C"])
            rec.update(c_source="inner_lopo", inner_median_lift=tuned["best_score"], inner_nonzero_mean=tuned["best_nonzero_mean"],
                       n_inner_folds=tuned["n_inner_folds"], fallback=tuned["fallback"],
                       best_C_at_grid_lower_bound=tuned["at_grid_lower_bound"],
                       best_C_at_grid_upper_bound=bool(np.isclose(C, max(c_grid), rtol=1e-9)))
        else:
            C = FIXED_C
            rec.update(c_source="fixed", C_scaled_to_n=rc.c_scaled_to_n(FIXED_C, n_train, n_train))
        est = rc.fit_probe(Z[train], y[train], penalty=pen, C=C)
        coef = rc.probe_coef(est)
        prob = rc.probe_scores(est, Z[test])
        ap, prevalence, lift = rc.heldout_lift(prob, y[test])
        active = np.flatnonzero(np.abs(coef) > rc.COEF_NONZERO_TOL)
        top = int(np.argmax(np.abs(coef))) if active.size else -1
        rec.update(
            C=C, heldout_ap=ap, prevalence=prevalence, heldout_lift=lift, heldout_auroc=float(roc_auc_score(y[test], prob)),
            n_nonzero=int(active.size), nonzero_factors=";".join(fid(i) for i in active),
            coef_matched_oriented=float(sign * coef[j]), coef_matched_abs_rank=int(np.sum(np.abs(coef) > abs(coef[j])) + 1),
            top_coef_factor=fid(top), top_coef_abs_r_with_ref=float(r_vec[top]) if top >= 0 else float("nan"),
            top_coef_value=float(coef[top]) if top >= 0 else 0.0,
        )
        if pen == "l1":
            margins = rc.l1_entry_margins(Zs_train, y[train], C, standardize=False)
            rec.update(max_entry_margin=float(margins.max()), argmax_entry_factor=fid(int(np.argmax(margins))),
                       entry_margin_matched=float(margins[j]))
        policies.append(rec)
        probs.append(prob.astype(np.float32))
    record.update(policies=policies, seconds=time.perf_counter() - t0, status="done")
    return record, np.vstack(probs)


def run_stage2(ctx: Ctx, specs: List[Dict[str, Any]], match_info: Dict[str, Dict[str, Any]],
               only_arms: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    from joblib import Parallel, delayed

    todo, records = [], []
    for spec in specs:
        if only_arms is not None and spec["arm"] not in only_arms:
            continue
        sig = unit_signature(ctx, spec)
        for h in spec["stage2_heldouts"]:
            uid = unit_id(spec, h)
            if uid in ctx.session_records:
                records.append(ctx.session_records[uid])
                continue
            jpath, ppath = ctx.units / f"{uid}.json", ctx.units / f"{uid}.npy"
            done = read_json_done(jpath) if ctx.args.resume else None
            if done is not None and done.get("signature") == sig:
                try:
                    prob = np.load(ppath)
                    if prob.shape == (len(spec["policies"]), int(done["n_heldout"])):
                        records.append(done)
                        continue
                except Exception:
                    pass
            todo.append((spec, h, sig))
    label = "anchor arm" if only_arms else "stage2"
    print(f"[{label}] {len(records) + len(todo)} units; {len(records)} resumed, {len(todo)} to run (n_jobs={ctx.n_jobs})", flush=True)
    todo.sort(key=lambda t: (-sum(p.endswith("_tuned") for p in t[0]["policies"]), -t[0]["k"], t[0]["basis_id"], t[1]))
    t0 = time.perf_counter()
    results = Parallel(n_jobs=ctx.n_jobs, verbose=0)(
        delayed(_stage2_unit)(ctx.exp_str, spec, h, str(basis_dir(ctx, spec)), match_info[spec["basis_id"]], ctx.c_grid, sig)
        for spec, h, sig in todo
    )
    for (spec, h, _), (record, probs) in zip(todo, results):
        uid = record["unit"]
        np.save(ctx.units / f"{uid}.npy", probs)
        atomic_write_json(record, ctx.units / f"{uid}.json")
        records.append(record)
        ctx.session_records[uid] = record
    elapsed = time.perf_counter() - t0
    ctx.timings[f"{label.replace(' ', '_')}_s"] = ctx.timings.get(f"{label.replace(' ', '_')}_s", 0.0) + elapsed
    print(f"[{label}] ran {len(todo)} units in {elapsed:.1f} s", flush=True)
    return records


def load_records(ctx: Ctx, specs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Finished Stage 2 unit records for the plan (no computation); also the missing unit ids."""
    records, missing = [], []
    for spec in specs:
        sig = unit_signature(ctx, spec)
        for h in spec["stage2_heldouts"]:
            uid = unit_id(spec, h)
            rec = ctx.session_records.get(uid) or read_json_done(ctx.units / f"{uid}.json")
            if rec is not None and rec.get("signature") == sig and (ctx.units / f"{uid}.npy").is_file():
                records.append(rec)
            else:
                missing.append(uid)
    return records, missing


def annotate_records(ctx: Ctx, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Refresh the alignment-dependent fields of Stage 2 units from the current d17_match_info.json.

    A unit stores the coefficient of the factor matched to the reference axis when it ran. The
    current alignment must give the same matched factor; then only the correlation-based fields
    (and, if the match sign flipped, the orientation of that coefficient) are refreshed.
    """
    match_info = json.loads((ctx.cache / "d17_match_info.json").read_text(encoding="utf-8"))
    stale = []
    out = []
    for rec in records:
        info = match_info.get(rec["basis_id"])
        if info is None:
            stale.append(rec["unit"])
            continue
        if fid(int(info["matched_factor"])) != rec["matched_factor"]:
            stale.append(rec["unit"])
            continue
        new = dict(rec)
        flip = (float(info["matched_r"]) >= 0) != (float(rec["matched_r_all_cells"]) >= 0)
        new["matched_r_all_cells"] = float(info["matched_r"])
        r_vec = np.abs(np.asarray(info["r_ref_vs_variant"], dtype=float))
        pols = []
        for p in rec["policies"]:
            q = dict(p)
            if flip:
                q["coef_matched_oriented"] = -float(q["coef_matched_oriented"])
            top = q.get("top_coef_factor") or ""
            q["top_coef_abs_r_with_ref"] = float(r_vec[int(top.rsplit("_", 1)[-1]) - 1]) if top else float("nan")
            pols.append(q)
        new["policies"] = pols
        out.append(new)
    if stale:
        raise SystemExit(f"[report] {len(stale)} Stage 2 units were run against a different matched factor (e.g. {stale[:3]}); "
                         "delete those unit files under stage2_units/ and rerun the stage2 part with --resume")
    return out


def flatten_units(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        base = {k: v for k, v in r.items() if k not in ("policies", "signature", "status")}
        for i, p in enumerate(r["policies"]):
            rows.append({**base, "policy_index": i, **p})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["lift_pp"] = 100.0 * frame["heldout_lift"]
    return frame


# --------------------------------------------------------------------------------------
# part: anchor
# --------------------------------------------------------------------------------------


def stored_rows(ctx: Ctx) -> pd.DataFrame:
    """IFN-alpha PCA/FA rows of the stored sharedness LOPO table (cached extract)."""
    cache = ctx.cache / "stored_sharedness_lopo_ifna_pca_fa_rows.csv"
    if cache.is_file() and cache.stat().st_size > 0:
        try:
            return pd.read_csv(cache)
        except Exception:
            pass
    cols = ["stage0_panel_id", "representation_family", "stage1_method", "effective_k", "seed", "heldout_patient", "penalty",
            "C", "l1_ratio", "n_malignant", "n_normal", "ml_backend_used", "nonzero_coefficient_count", "selected_feature_ids",
            "stage2_auprc", "stage2_auroc"]
    parts = []
    for chunk in pd.read_csv(ctx.exp / STORED_LOPO_REL, usecols=cols, chunksize=250_000, low_memory=False):
        keep = chunk["stage0_panel_id"].eq(PANEL) & chunk["stage1_method"].isin(METHODS) & chunk["representation_family"].eq("dr")
        parts.append(chunk.loc[keep])
    frame = pd.concat(parts, ignore_index=True)
    frame["stored_prevalence"] = frame["n_malignant"] / (frame["n_malignant"] + frame["n_normal"])
    frame["stored_lift"] = frame["stage2_auprc"] - frame["stored_prevalence"]
    atomic_write_csv(frame, cache)
    return frame


def run_anchor(ctx: Ctx, specs: List[Dict[str, Any]], match_info: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, bool]:
    records = run_stage2(ctx, specs, match_info, only_arms=("transductive_stored",))
    units = flatten_units(records)
    stored = stored_rows(ctx)
    rows = []
    for r in units.loc[units["policy"].isin(POLICIES_FIXED)].itertuples(index=False):
        hit = stored.loc[
            stored["stage1_method"].eq(r.stage1_method) & stored["effective_k"].round().astype(int).eq(int(r.effective_k))
            & stored["heldout_patient"].eq(r.heldout_patient) & stored["penalty"].eq(r.penalty)
            & np.isclose(stored["C"].astype(float), FIXED_C, rtol=1e-6)
        ]
        if len(hit) != 1:
            raise AssertionError(f"{len(hit)} stored rows for {r.stage1_method}{r.effective_k} {r.heldout_patient} {r.penalty} C={FIXED_C}")
        s = hit.iloc[0]
        d_pp = 100.0 * (float(r.heldout_lift) - float(s["stored_lift"]))
        rows.append({
            "stage1_method": r.stage1_method, "effective_k": int(r.effective_k), "heldout_patient": r.heldout_patient,
            "penalty": r.penalty, "C": FIXED_C, "stored_backend": s["ml_backend_used"],
            "stored_lift_pp": 100.0 * float(s["stored_lift"]), "reproduced_lift_pp": 100.0 * float(r.heldout_lift),
            "abs_diff_pp": abs(d_pp), "stored_nonzero": int(s["nonzero_coefficient_count"]), "reproduced_nonzero": int(r.n_nonzero),
            "stored_selected": "" if pd.isna(s["selected_feature_ids"]) else str(s["selected_feature_ids"]),
            "reproduced_selected": r.nonzero_factors if r.penalty == "l1" else "",
        })
    table = pd.DataFrame(rows)
    table["same_nonzero"] = table["stored_nonzero"] == table["reproduced_nonzero"]
    table["same_support"] = [(a == b) if pen == "l1" else bool(sn) for a, b, pen, sn in
                             zip(table["stored_selected"], table["reproduced_selected"], table["penalty"], table["same_nonzero"])]
    table["within_strict_tol"] = table["abs_diff_pp"] <= ANCHOR_STRICT_PP
    table["pass"] = (table["abs_diff_pp"] <= ANCHOR_GATE_PP) & table["same_nonzero"] & table["same_support"]
    atomic_write_csv(table, ctx.tables / "d17_anchor_check.csv")
    ok = bool(len(table) and table["pass"].all())
    print(f"[anchor] {int(table['pass'].sum())}/{len(table)} stored per-patient lifts reproduced within {ANCHOR_GATE_PP} pp with the "
          f"same support ({int(table['within_strict_tol'].sum())}/{len(table)} within {ANCHOR_STRICT_PP} pp; "
          f"max |diff| {table['abs_diff_pp'].max():.4f} pp)", flush=True)
    return table, ok


# --------------------------------------------------------------------------------------
# part: report
# --------------------------------------------------------------------------------------


def variant_group(arm: str, variant: str) -> str:
    return "per_fold" if arm in ("inductive", "normal_only") else variant


def pooled_ap(ctx: Ctx, frame: pd.DataFrame) -> float:
    """AP over the concatenated held-out cells of the supported folds (fold models mixed)."""
    if frame["heldout_patient"].nunique() != len(ctx.supported):
        return float("nan")
    ys, ps = [], []
    for r in frame.itertuples(index=False):
        probs = np.load(ctx.units / f"{r.unit}.npy")
        ps.append(probs[int(r.policy_index)])
        ys.append(ctx.y[ctx.patients == r.heldout_patient])
    return rc.heldout_lift(np.concatenate(ps), np.concatenate(ys))[0]


def stage2_tables(ctx: Ctx, units: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    units = units.copy()
    units["variant_group"] = [variant_group(a, v) for a, v in zip(units["arm"], units["variant"])]
    ref = units.loc[units["arm"].eq("transductive_stored")].set_index(["stage1_method", "effective_k", "heldout_patient", "policy"])["heldout_lift"]
    key = list(zip(units["stage1_method"], units["effective_k"], units["heldout_patient"], units["policy"]))
    units["transductive_lift_pp"] = [100.0 * ref.get(k, np.nan) for k in key]
    units["delta_vs_transductive_pp"] = units["lift_pp"] - units["transductive_lift_pp"]
    refit = units.loc[units["arm"].eq("transductive_refit")].set_index(["stage1_method", "effective_k", "heldout_patient", "policy"])["heldout_lift"]
    units["transductive_refit_lift_pp"] = [100.0 * refit.get(k, np.nan) for k in key]
    units["delta_vs_transductive_refit_pp"] = units["lift_pp"] - units["transductive_refit_lift_pp"]
    by_patient_cols = [
        "arm", "stage1_method", "effective_k", "variant", "variant_group", "heldout_patient", "policy", "penalty", "c_source",
        "C", "heldout_ap", "prevalence", "lift_pp", "heldout_auroc", "transductive_lift_pp", "delta_vs_transductive_pp",
        "transductive_refit_lift_pp", "delta_vs_transductive_refit_pp",
        "n_nonzero", "nonzero_factors", "ref_factor", "matched_factor", "matched_r_all_cells", "coef_matched_oriented",
        "coef_matched_abs_rank", "top_coef_factor", "top_coef_abs_r_with_ref", "max_entry_margin", "argmax_entry_factor",
        "entry_margin_matched", "inner_median_lift", "inner_nonzero_mean", "n_inner_folds", "fallback",
        "best_C_at_grid_lower_bound", "best_C_at_grid_upper_bound", "C_scaled_to_n", "n_train", "n_train_malignant",
        "n_heldout", "n_heldout_malignant", "basis_id", "unit", "policy_index", "seconds",
    ]
    by_patient = units.reindex(columns=by_patient_cols)
    by_patient = by_patient.sort_values(["stage1_method", "effective_k", "arm", "variant", "policy", "heldout_patient"]).reset_index(drop=True)
    summary = []
    groups = units.groupby(["arm", "stage1_method", "effective_k", "variant_group", "policy"], sort=False)
    for (arm, m, k, vg, policy), g in groups:
        summary.append(_summ_row(ctx, arm, m, k, vg, policy, g))
    # seed-mean rows for the patient-stratified subsample arm (per-patient mean over seeds)
    sub = units.loc[units["arm"].eq("balanced_subsample")]
    for (m, k, policy), g in sub.groupby(["stage1_method", "effective_k", "policy"], sort=False):
        per_pat = g.groupby("heldout_patient").agg(lift_pp=("lift_pp", "mean"), delta=("delta_vs_transductive_pp", "mean"),
                                                   delta_refit=("delta_vs_transductive_refit_pp", "mean"),
                                                   n_nonzero=("n_nonzero", "mean"), coef=("coef_matched_oriented", "mean"),
                                                   r=("matched_r_all_cells", lambda s: float(np.mean(np.abs(s)))))
        pooled = [v for v in (pooled_ap(ctx, gg) for _, gg in g.groupby("variant")) if np.isfinite(v)]
        summary.append({
            "arm": "balanced_subsample", "stage1_method": m, "effective_k": int(k), "variant_group": f"seed_mean_{g['variant'].nunique()}",
            "policy": policy, "n_folds": int(len(per_pat)), "median_lift_pp": median(per_pat["lift_pp"]), "p20_lift_pp": p20(per_pat["lift_pp"]),
            "mean_lift_pp": float(per_pat["lift_pp"].mean()), "n_lift_gt_0": int((per_pat["lift_pp"] > 1e-9).sum()),
            "n_all_zero_folds": float((g["n_nonzero"] == 0).sum()) / max(1, g["variant"].nunique()),
            "pooled_ap": float(np.mean(pooled)) if pooled else float("nan"),
            "median_delta_vs_transductive_pp": median(per_pat["delta"]),
            "n_folds_below_transductive": int((per_pat["delta"] < -1e-9).sum()),
            "median_delta_vs_refit_pp": median(per_pat["delta_refit"]),
            "n_folds_below_refit": int((per_pat["delta_refit"] < -1e-9).sum()),
            "median_abs_matched_r": median(per_pat["r"]), "median_coef_matched_oriented": median(per_pat["coef"]),
            "C_values": ";".join(sorted({f"{c:.3g}" for c in g["C"]})),
        })
    summary_df = pd.DataFrame(summary)
    summary_df["_arm"] = summary_df["arm"].map({a: i for i, a in enumerate(ARM_ORDER)})
    summary_df["_pol"] = summary_df["policy"].map({p: i for i, p in enumerate(POLICIES_FULL)})
    summary_df = summary_df.sort_values(["effective_k", "stage1_method", "_arm", "variant_group", "_pol"]).drop(columns=["_arm", "_pol"])
    return by_patient, summary_df.reset_index(drop=True)


def _summ_row(ctx: Ctx, arm: str, m: str, k: int, vg: str, policy: str, g: pd.DataFrame) -> Dict[str, Any]:
    return {
        "arm": arm, "stage1_method": m, "effective_k": int(k), "variant_group": vg, "policy": policy,
        "n_folds": int(len(g)), "median_lift_pp": median(g["lift_pp"]), "p20_lift_pp": p20(g["lift_pp"]),
        "mean_lift_pp": float(g["lift_pp"].mean()), "n_lift_gt_0": int((g["lift_pp"] > 1e-9).sum()),
        "n_all_zero_folds": int((g["n_nonzero"] == 0).sum()), "pooled_ap": pooled_ap(ctx, g),
        "median_delta_vs_transductive_pp": median(g["delta_vs_transductive_pp"]),
        "n_folds_below_transductive": int((g["delta_vs_transductive_pp"] < -1e-9).sum()),
        "median_delta_vs_refit_pp": median(g["delta_vs_transductive_refit_pp"]),
        "n_folds_below_refit": int((g["delta_vs_transductive_refit_pp"] < -1e-9).sum()),
        "median_abs_matched_r": median(np.abs(g["matched_r_all_cells"])), "median_coef_matched_oriented": median(g["coef_matched_oriented"]),
        "C_values": ";".join(sorted({f"{c:.3g}" for c in g["C"]})),
    }


def d19_crosscheck(ctx: Ctx, by_patient: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Transductive tuned-C units vs D19's W3.2 units (same contract, data and grid); read-only."""
    path = ctx.exp / "analysis" / "scorecards" / "stage2_diagnostic" / "regularization_contract_v1" / "d19_ifna_units.csv"
    if not path.is_file() or ctx.args.smoke:
        return None
    d19 = pd.read_csv(path)
    d19 = d19.loc[d19["kind"].eq("tune") & d19["penalty"].isin(["l1", "l2"])]
    mine = by_patient.loc[by_patient["arm"].eq("transductive_stored") & by_patient["policy"].isin(["l1_tuned", "l2_tuned"])
                          & by_patient["effective_k"].eq(40)]
    rows = []
    for r in mine.itertuples(index=False):
        hit = d19.loc[d19["stage1_method"].eq(r.stage1_method) & d19["heldout_patient"].eq(r.heldout_patient) & d19["penalty"].eq(r.penalty)]
        if len(hit) != 1:
            continue
        h = hit.iloc[0]
        rows.append({"stage1_method": r.stage1_method, "heldout_patient": r.heldout_patient, "policy": r.policy,
                     "d17_C": r.C, "d19_C": float(h["C"]), "d17_lift_pp": r.lift_pp, "d19_lift_pp": 100.0 * float(h["heldout_lift"]),
                     "abs_diff_pp": abs(r.lift_pp - 100.0 * float(h["heldout_lift"])), "same_C": bool(np.isclose(r.C, float(h["C"])))})
    return pd.DataFrame(rows)


#: (metric, True if a larger value means more axis movement) for the composition-matched controls.
LABEL_MATCHED_METRICS = (("axis_loss_fit", True), ("matched_abs_r_fit", False), (f"angle_top{TOP_M}_max_deg", True))


def leverage_table(align: pd.DataFrame) -> pd.DataFrame:
    """Leave-out (and normal-only) refits vs their size-matched random-drop controls."""
    metrics = ("axis_loss_fit", "matched_abs_r_fit", "many_to_one_r2_fit", f"angle_top{TOP_M}_max_deg", f"angle_top{TOP_M}_mean_cos2",
               "angle_allK_mean_cos2", f"ref_axis_containment_top{TOP_M}", "matched_factor_importance_rank")
    rows = []
    for (m, k), g in align.groupby(["stage1_method", "effective_k"]):
        pairs = [(r, g.loc[g["arm"].eq("leave_out_random") & g["matches_drop_set"].eq(r["variant"][len("drop-"):])])
                 for _, r in g.loc[g["arm"].eq("leave_out")].iterrows()]
        for _, r in g.loc[g["arm"].eq("normal_only_all")].iterrows():
            pairs.append((r, g.loc[g["arm"].eq("normal_only_random")]))
        for r, ctrl in pairs:
            dropped = r["variant"][len("drop-"):] if r["arm"] == "leave_out" else "malignant cells"
            row = {
                "stage1_method": m, "effective_k": int(k), "arm": r["arm"], "dropped": dropped,
                "group": ("highest_prevalence" if r["arm"] == "leave_out" and set(dropped.split("-")) <= set(HIGH_PREVALENCE)
                          else "contrast" if r["arm"] == "leave_out" else "malignant"),
                "n_dropped_cells": r["n_dropped_cells"], "n_dropped_malignant": r["n_dropped_malignant"],
                "n_random_controls": int(len(ctrl)), "ref_factor_label": r["ref_factor_label"],
                "matched_factor_fit": r["matched_factor_fit"],
            }
            for met in metrics:
                val = r.get(met)
                row[met] = val
                if len(ctrl):
                    cv = ctrl[met].astype(float)
                    row[f"{met}_random_mean"] = float(cv.mean())
                    row[f"{met}_random_min"] = float(cv.min())
                    row[f"{met}_random_max"] = float(cv.max())
                    row[f"{met}_excess"] = float(val) - float(cv.mean()) if val is not None and np.isfinite(float(val)) else np.nan
            if len(ctrl):
                row["axis_loss_fit_ratio_to_random_mean"] = float(r["axis_loss_fit"]) / float(ctrl["axis_loss_fit"].astype(float).mean())
            # composition-matched controls (same number of dropped cells AND dropped malignant cells)
            lab = (g.loc[g["arm"].eq("leave_out_label_random") & g["matches_drop_set"].eq(dropped)]
                   if r["arm"] == "leave_out" else g.iloc[0:0])
            row["n_label_matched_controls"] = int(len(lab))
            for met, larger_moves in LABEL_MATCHED_METRICS:
                if not len(lab):
                    continue
                cv = lab[met].astype(float).to_numpy()
                val = float(r[met])
                row[f"{met}_label_matched_mean"] = float(cv.mean())
                row[f"{met}_label_matched_max"] = float(cv.max())
                row[f"{met}_label_matched_min"] = float(cv.min())
                # one-sided empirical rank p: share of controls (plus the observation) moving the axis at least as much
                n_ge = int(np.sum(cv >= val)) if larger_moves else int(np.sum(cv <= val))
                row[f"{met}_label_matched_rank_p"] = (1.0 + n_ge) / (1.0 + cv.size)
            if len(lab):
                row["axis_loss_fit_ratio_to_label_matched_mean"] = float(r["axis_loss_fit"]) / float(lab["axis_loss_fit"].astype(float).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def pca_vs_fa_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not {"pca", "fa"} <= set(summary["stage1_method"]):
        return pd.DataFrame()
    piv = summary.set_index(["arm", "effective_k", "variant_group", "policy", "stage1_method"])
    for (arm, k, vg, policy), _ in summary.groupby(["arm", "effective_k", "variant_group", "policy"], sort=False):
        try:
            a, b = piv.loc[(arm, k, vg, policy, "pca")], piv.loc[(arm, k, vg, policy, "fa")]
        except KeyError:
            continue
        rows.append({"arm": arm, "effective_k": int(k), "variant_group": vg, "policy": policy,
                     "pca_median_lift_pp": a["median_lift_pp"], "fa_median_lift_pp": b["median_lift_pp"],
                     "pca_minus_fa_median_pp": a["median_lift_pp"] - b["median_lift_pp"],
                     "pca_p20_pp": a["p20_lift_pp"], "fa_p20_pp": b["p20_lift_pp"],
                     "pca_pooled_ap": a["pooled_ap"], "fa_pooled_ap": b["pooled_ap"],
                     "pca_n_lift_gt_0": a["n_lift_gt_0"], "fa_n_lift_gt_0": b["n_lift_gt_0"],
                     "pca_n_all_zero": a["n_all_zero_folds"], "fa_n_all_zero": b["n_all_zero_folds"]})
    return pd.DataFrame(rows)


def make_figures(ctx: Ctx, by_patient: pd.DataFrame, leverage: pd.DataFrame) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua, ink, muted, grid = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 8.5, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
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

    k_main = max(ctx.ks)
    pats = list(ctx.supported)
    bp = by_patient.loc[by_patient["effective_k"].eq(k_main)]
    # D17A: per-patient lift by arm, policy and method
    series = (("transductive_stored", "transductive (stored)", ink, "o", "none"),
              ("inductive", "inductive LOPO", blue, "o", blue),
              ("normal_only", "normal-only, inductive", orange, "s", orange),
              ("balanced_weighted", "patient-balanced (weighted)", aqua, "^", aqua))
    pols = [p for p in POLICIES_FULL if p in set(bp["policy"])]
    if not bp.empty and pols:
        fig, axes = plt.subplots(len(ctx.methods), len(pols), figsize=(2.35 * len(pols) + 0.5, 2.3 * len(ctx.methods) + 0.4),
                                 sharey="row", squeeze=False)
        x = np.arange(len(pats))
        for i, m in enumerate(ctx.methods):
            for jx, pol in enumerate(pols):
                ax = axes[i, jx]
                for s_i, (arm, lab, col, mk, face) in enumerate(series):
                    d = bp.loc[bp["stage1_method"].eq(m) & bp["policy"].eq(pol) & bp["arm"].eq(arm)].set_index("heldout_patient")
                    if d.empty:
                        continue
                    ax.plot(x + (s_i - 1.5) * 0.14, [d["lift_pp"].get(p, np.nan) for p in pats], linestyle="none", marker=mk, ms=4.6,
                            mfc=face, mec=col, mew=1.2, label=lab)
                ax.axhline(0, color=muted, lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(pats, rotation=90, fontsize=7)
                ax.grid(axis="x", visible=False)
                if i == 0:
                    ax.set_title(pol.replace("_", " "), fontsize=8.5)
                if jx == 0:
                    ax.set_ylabel(f"{m.upper()}{k_main}\nheld-out lift (pp)")
        axes[0, -1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"D17A  IFN-α {k_main}-factor bases: per-patient held-out lift by Stage 1 fit scope and C policy",
                     x=0.01, y=1.02, ha="left", fontsize=9.5)
        save(fig, "D17A_ifna_lift_by_basis_scope")
    # D17B: leverage (axis loss and top-8 principal angle) vs size-matched random drops
    lv = leverage.loc[leverage["effective_k"].eq(k_main)] if not leverage.empty else leverage
    if lv is not None and not lv.empty:
        mets = (("axis_loss_fit", "1 − many-to-one R² of the\nreference axis (retained cells)", True),
                (f"angle_top{TOP_M}_max_deg", f"largest principal angle,\ntop-{TOP_M} loading subspaces (deg)", False))
        fig, axes = plt.subplots(len(ctx.methods), 2, figsize=(7.6, 2.5 * len(ctx.methods) + 0.4), squeeze=False)
        for i, m in enumerate(ctx.methods):
            d = lv.loc[lv["stage1_method"].eq(m)].reset_index(drop=True)
            xx = np.arange(len(d))
            for jx, (met, lab, logy) in enumerate(mets):
                ax = axes[i, jx]
                cols = [{"highest_prevalence": orange, "contrast": blue}.get(g, aqua) for g in d["group"]]
                if f"{met}_random_min" in d:
                    ax.vlines(xx, d[f"{met}_random_min"], d[f"{met}_random_max"], color=muted, lw=3, alpha=0.45, label="size-matched random drops (range)")
                    ax.plot(xx, d[f"{met}_random_mean"], linestyle="none", marker="_", ms=10, color=muted)
                ax.scatter(xx, d[met], c=cols, s=30, zorder=3, edgecolors="#fcfcfb", linewidths=1.0)
                if logy:
                    ax.set_yscale("log")
                ax.set_xticks(xx)
                ax.set_xticklabels([f"drop {s}" if g != "malignant" else "normal cells only" for s, g in zip(d["dropped"], d["group"])],
                                   rotation=90, fontsize=7)
                ax.set_ylabel(f"{m.upper()}{k_main}\n{lab}", fontsize=7.5)
                ax.grid(axis="x", visible=False)
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], linestyle="none", marker="o", color=orange, label="drop highest-prevalence patient(s)"),
                   Line2D([], [], linestyle="none", marker="o", color=blue, label="drop contrast patient(s)"),
                   Line2D([], [], linestyle="none", marker="o", color=aqua, label="drop all malignant cells (normal-only fit)"),
                   Line2D([], [], color=muted, lw=3, alpha=0.45, label="size-matched random drops (range, mean tick)")]
        axes[0, 1].legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"D17B  IFN-α loading leverage: leave-patients-out refits vs size-matched random drops (K={k_main})",
                     x=0.01, y=1.02, ha="left", fontsize=9.5)
        fig.tight_layout()
        save(fig, "D17B_ifna_leave_out_leverage")
    # D17C: inductive vs transductive per patient, by policy
    ind = bp.loc[bp["arm"].eq("inductive")]
    if not ind.empty:
        fig, axes = plt.subplots(1, len(pols), figsize=(2.3 * len(pols) + 0.6, 2.6), squeeze=False)
        lim_hi = float(np.nanmax([ind["lift_pp"].max(), ind["transductive_lift_pp"].max() if "transductive_lift_pp" in ind else 0])) + 2
        lim_lo = min(-1.0, float(np.nanmin([ind["lift_pp"].min(), ind["transductive_lift_pp"].min()])) - 1)
        for jx, pol in enumerate(pols):
            ax = axes[0, jx]
            for m, col in (("pca", blue), ("fa", orange)):
                d = ind.loc[ind["stage1_method"].eq(m) & ind["policy"].eq(pol)]
                ax.scatter(d["transductive_lift_pp"], d["lift_pp"], s=22, color=col, edgecolors="#fcfcfb", linewidths=1.0,
                           label=f"{m.upper()}{k_main}", zorder=3)
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=muted, lw=0.8, ls=":")
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_title(pol.replace("_", " "), fontsize=8.5)
            ax.set_xlabel("transductive lift (pp)")
            if jx == 0:
                ax.set_ylabel("inductive lift (pp)")
        axes[0, -1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle("D17C  IFN-α inductive vs transductive held-out lift, one point per supported patient",
                     x=0.01, y=1.04, ha="left", fontsize=9.5)
        save(fig, "D17C_ifna_inductive_vs_transductive")
    return written


def _pp(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.2f}"


def write_findings(ctx: Ctx, anchor: pd.DataFrame, align: pd.DataFrame, summary: pd.DataFrame, by_patient: pd.DataFrame,
                   leverage: pd.DataFrame, gap: pd.DataFrame) -> Path:
    k = max(ctx.ks)
    L: List[str] = []
    L.append(f"# D17 basis-variant pilot, IFN-alpha ({SLUG})")
    L.append("")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
             "scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py. "
             "Descriptive results over seven supported patients. Stage 1 fit scopes are labelled per arm; "
             "LOPO on the transductive basis is label transfer, not inductive deployment.")
    L.append("")
    L.append("## Anchor (transductive stored basis)")
    if anchor is not None and not anchor.empty:
        for kk, g in anchor.groupby("effective_k"):
            L.append(f"- K={kk}: {int(g['pass'].sum())}/{len(g)} stored per-patient lifts reproduced within {ANCHOR_GATE_PP} pp with "
                     f"the same nonzero support (L1 and L2 at C = {FIXED_C:g}); {int(g['within_strict_tol'].sum())}/{len(g)} within "
                     f"{ANCHOR_STRICT_PP} pp; max |difference| {g['abs_diff_pp'].max():.4f} pp.")
    L.append("")
    L.append("## Transductive refit vs the alignment reference (score convention)")
    L.append("The alignment reference is the stored FA scores, and for PCA the projection X @ V onto the stored loadings, "
             "not the stored PCA scores: those are sklearn's randomized-SVD U * S. A matched |r| of 1 for the PCA refit "
             "therefore means the refit reproduces the projection, not the stored scores (r of the stored scores vs the "
             "reference below).")
    refax_path = ctx.tables / "d17_reference_axes.csv"
    refax = pd.read_csv(refax_path).set_index(["stage1_method", "effective_k"]) if refax_path.is_file() else None
    tr = align.loc[align["arm"].eq("transductive_refit")]
    for r in tr.itertuples(index=False):
        extra = ""
        if refax is not None and (r.stage1_method, r.effective_k) in refax.index:
            ra = refax.loc[(r.stage1_method, r.effective_k)]
            extra = (f"; stored scores vs reference ({ra['alignment_reference']}): {r.ref_factor_label} r "
                     f"{float(ra['ref_factor_r_stored_scores_vs_reference']):.4f}, min over factors {float(ra['min_r_stored_scores_vs_reference']):.4f}")
        L.append(f"- {r.stage1_method.upper()}{r.effective_k}: refit vs reference {r.ref_factor_label} matched |r| {r.matched_abs_r_all:.4f}, "
                 f"many-to-one R2 {r.many_to_one_r2_all:.4f}, top-{TOP_M} max angle {getattr(r, f'angle_top{TOP_M}_max_deg'):.2f} deg{extra}.")
    L.append("")
    L.append("## Stage 2 summaries (lift in pp; pooled AP over supported held-out cells; median and P20 lift over patients)")
    cols = ["arm", "stage1_method", "effective_k", "variant_group", "policy", "pooled_ap", "median_lift_pp", "p20_lift_pp",
            "n_lift_gt_0", "n_all_zero_folds", "median_delta_vs_transductive_pp", "n_folds_below_transductive",
            "median_delta_vs_refit_pp", "n_folds_below_refit", "median_abs_matched_r"]
    main = summary.loc[summary["arm"].isin(["transductive_stored", "transductive_refit", "inductive", "normal_only", "balanced_weighted"])
                       | (summary["arm"].eq("balanced_subsample") & summary["variant_group"].str.startswith("seed_mean"))]
    L.append("")
    L.append(_md_table(main[cols]))
    L.append("")
    L.append("## Q1: inductive vs transductive (per supported patient, same C policy)")
    L.append("Deltas vs the stored basis (the Stage 2 anchor) and vs the transductive refit (exact projection, same pipeline).")
    ind = by_patient.loc[by_patient["arm"].eq("inductive")]
    for (m, kk, pol), g in ind.groupby(["stage1_method", "effective_k", "policy"]):
        d, dr = g["delta_vs_transductive_pp"], g["delta_vs_transductive_refit_pp"]
        L.append(f"- {m.upper()}{kk} {pol}: vs stored median {_pp(median(d))} pp ({int((d < -1e-9).sum())}/{len(g)} lower, "
                 f"range {_pp(d.min())} to {_pp(d.max())}); vs refit median {_pp(median(dr))} pp ({int((dr < -1e-9).sum())}/{len(g)} lower, "
                 f"range {_pp(dr.min())} to {_pp(dr.max())}).")
    tr = by_patient.loc[by_patient["arm"].eq("transductive_refit")]
    L.append("")
    L.append("Transductive refit vs stored basis. Not a numerical floor for PCA: it is the score-convention difference "
             "(stored randomized-SVD U * S vs the refit's projection X @ V); FA scores are exact, so for FA it is the solver floor:")
    for (m, kk, pol), g in tr.groupby(["stage1_method", "effective_k", "policy"]):
        d = g["delta_vs_transductive_pp"]
        L.append(f"- {m.upper()}{kk} {pol}: max |delta| {_pp(d.abs().max())} pp; folds with |delta| > 1 pp: {int((d.abs() > 1).sum())}/{len(g)}.")
    L.append("")
    L.append("## Q2: loading leverage (leave patients out vs size-matched random drops, retained cells)")
    if leverage is not None and not leverage.empty:
        lcols = ["stage1_method", "effective_k", "dropped", "group", "n_dropped_cells", "n_dropped_malignant", "axis_loss_fit",
                 "axis_loss_fit_random_mean", "axis_loss_fit_random_max", "matched_abs_r_fit", f"angle_top{TOP_M}_max_deg",
                 f"angle_top{TOP_M}_max_deg_random_mean", f"angle_top{TOP_M}_max_deg_random_max", "matched_factor_importance_rank"]
        L.append("")
        L.append(_md_table(leverage[[c for c in lcols if c in leverage.columns]]))
        if "n_label_matched_controls" in leverage.columns and leverage["n_label_matched_controls"].fillna(0).gt(0).any():
            L.append("")
            L.append("Composition-matched controls: random drops with the same number of cells AND the same number of malignant "
                     "cells as each leave-out set (seeds listed in the manifest). rank p = (1 + controls moving the axis at least "
                     "as much) / (1 + controls), one-sided; descriptive, not corrected for the 48 contrasts.")
            L.append("")
            mcols = ["stage1_method", "effective_k", "dropped", "group", "n_dropped_cells", "n_dropped_malignant", "axis_loss_fit",
                     "axis_loss_fit_ratio_to_random_mean", "n_label_matched_controls", "axis_loss_fit_label_matched_mean",
                     "axis_loss_fit_label_matched_max", "axis_loss_fit_ratio_to_label_matched_mean", "axis_loss_fit_label_matched_rank_p",
                     f"angle_top{TOP_M}_max_deg_label_matched_rank_p"]
            lev = leverage.loc[leverage["arm"].eq("leave_out")]
            L.append(_md_table(lev[[c for c in mcols if c in lev.columns]]))
    L.append("")
    L.append("## Q3: normal-only bases")
    nall = align.loc[align["arm"].eq("normal_only_all")]
    for r in nall.itertuples(index=False):
        L.append(f"- {r.stage1_method.upper()}{r.effective_k} normal cells of all patients: {r.ref_factor_label} matched |r| (all cells) "
                 f"{r.matched_abs_r_all:.4f}, many-to-one R2 {r.many_to_one_r2_all:.4f}, matched factor {r.matched_factor_all} "
                 f"(importance rank {r.matched_factor_importance_rank}).")
    L.append("")
    L.append("## Q4: PCA minus FA by arm and C policy")
    if gap is not None and not gap.empty:
        g4 = gap.loc[gap["arm"].isin(["transductive_stored", "inductive", "normal_only", "balanced_weighted"])
                     | (gap["arm"].eq("balanced_subsample") & gap["variant_group"].str.startswith("seed_mean"))]
        L.append("")
        L.append(_md_table(g4))
    L.append("")
    L.append("## Notes")
    L.append("- Stage 2 training cells are the standard LOPO folds in every arm; only the Stage 1 fit set changes, so "
             "c_scaled_to_n returns the stored C (recorded as C_scaled_to_n) and no separate scaled-C arm is run.")
    L.append("- Tuned C: inner LOPO over the supported training patients on the outer fold's basis "
             "(for inductive arms the inner folds share the outer inductive basis).")
    L.append("- An all-zero L1 probe scores the constant-score no-skill AP = pi by construction (lift 0); it is not 'no signal'.")
    L.append("- Coefficients are in train-standardized score units, oriented so that a positive value points the same way as "
             "the stored reference factor (sign of the all-cell score correlation).")
    path = ctx.tables / f"{SLUG}_findings.md"
    atomic_write_text("\n".join(L) + "\n", path)
    return path


def _md_table(frame: pd.DataFrame) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(frame.columns) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest_and_ledger(ctx: Ctx, started: float, extra: Dict[str, Any]) -> None:
    head, dirty = git_state(REPO)
    manifest_path = ctx.tables / f"{SLUG}_manifest.json"
    history: List[Dict[str, Any]] = []
    try:
        history = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("run_history", []))
    except Exception:
        pass
    run = {
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - started, 1), "timings_s": {k: round(v, 1) for k, v in ctx.timings.items()},
        "argv": " ".join(shlex.quote(a) for a in sys.argv[1:]), "git_head": head, "git_dirty": dirty, "n_jobs": ctx.n_jobs,
    }
    history.append(run)
    manifest = {
        "slug": SLUG, "analysis": "D17", "task": "W2.1", "experiment_id": ctx.exp.name,
        "script": "scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py",
        "modules": ["scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/stage1_refit.py",
                    "scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py"],
        "stage1_refit_version": s1.MODULE_VERSION, "git_head": head, "git_dirty": dirty,
        "python": platform.python_version(),
        "inputs": {
            "features": FEATURES_REL + "/features.npz", "feature_names": FEATURES_REL + "/feature_names.json",
            "stored_bases": [f"stage1_dr/panel_id={PANEL}/method={m}/k={k}/seed={SEED}/{{scores,loadings}}.npy" for m in ctx.methods for k in ctx.ks],
            "cell_metadata": ctx.rel(cell_metadata()), "stored_lopo": STORED_LOPO_REL,
        },
        "parameters": {
            "panel": PANEL, "methods": ctx.methods, "ks": ctx.ks, "full_policy_ks": sorted(ctx.full_ks), "seed": SEED,
            "supported_patients": ctx.supported, "heldouts": ctx.heldouts, "drop_sets": ["-".join(d) for d in ctx.drop_sets],
            "subsample_seeds": list(ctx.sub_seeds), "random_drop_seeds": list(ctx.rand_seeds),
            "label_matched_drop_seeds": list(ctx.label_seeds), "fixed_C": FIXED_C,
            "policies": list(POLICIES_FULL), "anchor_gate_pp": ANCHOR_GATE_PP, "anchor_strict_pp": ANCHOR_STRICT_PP, "top_m": TOP_M,
            "pca": "sklearn PCA svd_solver=randomized random_state=42 (weighted: weighted covariance + eigh)",
            "fa": "sklearn FactorAnalysis svd_method=randomized random_state=42 (weighted: pseudo-data with the weighted covariance, tol rescaled)",
            "standardization": "per-gene mean / population SD of the Stage 1 fit cells (weighted for balanced_weighted); zero variance -> 0",
            "bases_location": ctx.rel(ctx.bases), "smoke": bool(ctx.args.smoke), "c_grid_n": len(ctx.c_grid),
        },
        "c_policy": rc.c_policy_record(c_policy="per_fold_tuned", training_n_changed=False,
                                       penalties=["l1", "l2"], fixed_arms=["l1 at C=1e-4", "l2 at C=1e-4"],
                                       notes="Stage 2 training cells are the standard LOPO folds; only the Stage 1 fit set changes."),
        "run_history": history,
        **extra,
    }
    atomic_write_json(manifest, manifest_path)
    ledger = []
    for folder, kind in ((ctx.tables, "table"), (ctx.figures, "figure")):
        for path in sorted(folder.glob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                ledger.append({"path": ctx.rel(path), "kind": kind, "bytes": path.stat().st_size,
                               "sha256": file_sha256(path) if path.name != manifest_path.name else ""})
    n_units = len(list(ctx.units.glob("*.json")))
    n_bases = len(list(ctx.bases.rglob("basis.json")))
    ledger.append({"path": ctx.rel(ctx.units), "kind": "cache_dir", "bytes": None, "sha256": f"{n_units} unit JSON + npy"})
    ledger.append({"path": ctx.rel(ctx.bases), "kind": "bases_dir", "bytes": None, "sha256": f"{n_bases} bases (basis.npz + basis.json)"})
    atomic_write_csv(pd.DataFrame(ledger), ctx.tables / f"{SLUG}_artifact_ledger.csv")


def run_report(ctx: Ctx, specs: List[Dict[str, Any]], align: Optional[pd.DataFrame],
               anchor: Optional[pd.DataFrame], started: float) -> None:
    t0 = time.perf_counter()
    if align is None:
        align = pd.read_csv(ctx.tables / "d17_alignment.csv")
    if anchor is None:
        anchor = pd.read_csv(ctx.tables / "d17_anchor_check.csv")
    if not bool(anchor["pass"].all()) and not ctx.args.allow_anchor_failure:
        raise SystemExit("[report] the anchor check failed; no other numbers are reported")
    records, missing = load_records(ctx, specs)
    if missing:
        raise SystemExit(f"[report] {len(missing)} Stage 2 units are missing (e.g. {missing[:3]}); run the stage2 part with --resume first")
    records = annotate_records(ctx, records)
    bases = bases_table(ctx, specs)
    if not bases.empty:
        zero = align.set_index("basis_id")["n_zero_variance_factors"]
        bases["n_zero_variance_factors_all_cells"] = bases["basis_id"].map(zero)
        atomic_write_csv(bases, ctx.tables / "d17_bases.csv")
    units = flatten_units(records)
    by_patient, summary = stage2_tables(ctx, units)
    atomic_write_csv(by_patient, ctx.tables / "d17_stage2_by_patient.csv")
    atomic_write_csv(summary, ctx.tables / "d17_stage2_summary.csv")
    leverage = leverage_table(align)
    atomic_write_csv(leverage, ctx.tables / "d17_leverage_contrast.csv")
    gap = pca_vs_fa_table(summary)
    atomic_write_csv(gap, ctx.tables / "d17_pca_vs_fa_by_arm.csv")
    cross = d19_crosscheck(ctx, by_patient)
    if cross is not None:
        atomic_write_csv(cross, ctx.tables / "d17_d19_tuned_crosscheck.csv")
    figs = make_figures(ctx, by_patient, leverage)
    findings = write_findings(ctx, anchor, align, summary, by_patient, leverage, gap)
    ctx.timings["report_s"] = time.perf_counter() - t0
    write_manifest_and_ledger(ctx, started, {
        "counts": {"bases": int(len(align)), "stage2_units": int(len(records)), "stage2_rows": int(len(units)),
                   "anchor_rows": int(len(anchor)), "anchor_pass": int(anchor["pass"].sum()) if len(anchor) else 0},
        "outputs": {"findings": ctx.rel(findings), "figures": [ctx.rel(p) for p in figs]},
    })
    print(f"[report] tables, {len(figs)} figure files, findings, manifest and ledger written in {ctx.timings['report_s']:.1f} s", flush=True)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", default=",".join(DEFAULT_PARTS), help="comma list of anchor,bases,align,stage2,report")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS), help="Stage 1 K values (default 40)")
    p.add_argument("--full-ks", default="40", help="K values whose leave_out and FA subsample arms also get tuned-C policies")
    p.add_argument("--n-jobs", type=int, default=12, help="joblib workers (capped at 12)")
    p.add_argument("--resume", action="store_true", help="skip finished bases and Stage 2 units")
    p.add_argument("--smoke", action="store_true", help="2 held-out patients, 2 drop sets, 1 seed, a thinned C grid")
    p.add_argument("--c-grid-stride", type=int, default=4, help="smoke only: keep every n-th C of the runner grid")
    p.add_argument("--out-root", default="", help="write everything under this dir instead of EXP/analysis")
    p.add_argument("--bases-location", choices=("cache", "stage1_dr_variants"), default="cache",
                   help="cache: EXP/analysis/cache/stage2_diagnostic/<slug>/bases (default); stage1_dr_variants: EXP/stage1_dr_variants/<slug> (R3)")
    p.add_argument("--allow-anchor-failure", action="store_true", help="continue past a failed anchor check (debug only)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = set(parts) - set(DEFAULT_PARTS)
    if unknown:
        raise SystemExit(f"unknown parts: {sorted(unknown)}")
    started = time.perf_counter()
    ctx = Ctx(args)
    specs = build_plan(ctx)
    print(f"[plan] {len(specs)} bases ({sum(len(s['stage2_heldouts']) for s in specs)} Stage 2 units); "
          f"methods {ctx.methods}, K {ctx.ks}; outputs under {ctx.rel(ctx.tables)}", flush=True)
    stored_match = {spec["basis_id"]: stored_match_info(ctx, spec) for spec in specs if spec["fit"]["kind"] == "stored"}
    anchor = None
    if "anchor" in parts:
        t0 = time.perf_counter()
        anchor, ok = run_anchor(ctx, specs, stored_match)
        ctx.timings["anchor_s"] = time.perf_counter() - t0
        if not ok and not args.allow_anchor_failure:
            print("[anchor] FAILED: the transductive arm does not reproduce the stored lifts; nothing else is reported.", flush=True)
            return 2
    if "bases" in parts:
        t0 = time.perf_counter()
        run_bases(ctx, specs)
        ctx.timings["bases_s"] = time.perf_counter() - t0
    align = None
    if "align" in parts or "stage2" in parts:
        align, match_info = run_align(ctx, specs)
        match_info.update(stored_match)
        if "stage2" in parts:
            run_stage2(ctx, specs, match_info)
    if "report" in parts:
        run_report(ctx, specs, align, anchor, started)
    print(f"[done] wall {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
