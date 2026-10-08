"""D15b patient-count learning curve runner (W6.1; slug patient_count_learning_curve_v1).

For each held-out supported patient h and each subset S of size k of the other supported
patients (k = 1..5, plus k = 6 = the ordinary LOPO fold), refit the Stage 2 probe on S plus every
unsupported patient and score h. Stage 1 is NOT refit: every representation is the frozen
transductive basis (stage1_dr/.../seed=42/scores.npy), fit on all cells including h and the
excluded patients.

C policies (arms), all through W3's regularization_contract (AGENTS R5):
  tuned        penalty in {L1, L2} and C chosen per training set by tune_c_inner_lopo (median
               inner held-out lift; ties to L1, the sparser probe). Headline arm.
  tuned_l1     L1 with C tuned per training set.
  tuned_l2     L2 with C tuned per training set (reported alongside, R5).
  fixed_atlas  the D1 atlas penalty and C (the D15-comparable reference; that C was selected
               by the D1 rule with held-out outcomes, so this arm is not nested in C).
  scaled_to_n  the atlas penalty with C = c_scaled_to_n(C_atlas, n_full_fold(h), n_train), which
               keeps the full-fold L1 entry threshold 4*rho/(C*n) fixed.
Inner tuning: inner leave-one-patient-out over the supported training patients (k >= 2), or the
contract's stratified 5-fold CV fallback when the training set holds one supported patient (k = 1).
Tuning depends only on the training set, so it runs once per (representation, S, penalty).

Parts (--parts, default tune,units,report):
  tune    rc.tune_c_inner_lopo per (representation, training subset, penalty)   -> cache/tune/
  units   outer fits per (representation, held-out patient, subset)            -> cache/units/
  report  design-prefixed tables, figures D15bA-D, findings, manifest, ledger  -> scorecards/, figures/

Designs (--design): 'full' enumerates every subset (C(6, k) per held-out patient); 'reduced' uses
a deterministic cyclic family of 7 subsets per size (see patient_count_curve.reduced_subsets).
The reduced units are a subset of the full units, so the full run resumes from them.

--resume skips finished units; empty, corrupt or out-of-date unit files count as unfinished.
--out-root redirects every output (use it for smoke runs). --smoke = one held-out patient (P07),
k in {1, 2}, one representation (allograft_promax40), full enumeration; it needs --out-root.

Examples (from the repo root):
  python scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py --smoke --out-root <scratch>/d15b_smoke --n-jobs 12
  python scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py --design reduced --resume --n-jobs 12
  bash scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.sh   # full design, screen W6_d15b
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker process; joblib supplies the parallelism (<= 12 workers).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import math
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

add_study_module_paths(subdirs=(), workstreams=("W3_dr_value_regularization", "W6_learning_curves"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import patient_count_curve as pcc  # noqa: E402
import regularization_contract as rc  # noqa: E402

SLUG = "patient_count_learning_curve_v1"
ANALYSIS_ID = "D15b"
TASK_ID = "W6.1"
DEVLOG_ID = "2026-10-08_d15b-patient-count-curve"
SCHEMA = 1
MAX_WORKERS = 12
DEFAULT_PARTS = ("tune", "units", "report")
DEFAULT_KS = (1, 2, 3, 4, 5, 6)
TUNE_PENALTIES = ("l1", "l2")
N_BOOT = 2000
SEED = 42
#: The four D15 representations (cell_count_learning_curve.CURVE_SPECS), in D15's order.
REPRESENTATIONS: Dict[str, Tuple[str, str, int]] = {
    "ifna_pca40": ("single_geneset__hallmark_interferon_alpha_response", "pca", 40),
    "ifna_fa40": ("single_geneset__hallmark_interferon_alpha_response", "fa", 40),
    "apoptosis_promax40": ("single_geneset__hallmark_apoptosis", "factosig_promax", 40),
    "allograft_promax40": ("single_geneset__hallmark_allograft_rejection", "factosig_promax", 40),
}
SMOKE = {"heldout": ("P07",), "ks": (1, 2), "representations": ("allograft_promax40",)}
ATLAS_DIAG_TABLE = "analysis/scorecards/stage2_diagnostic/old34_representation_diagnostic_table.csv"
ATLAS_PATIENT_TABLE = "analysis/scorecards/stage2_diagnostic/old34_d1_atlas_supported_patient_metrics.csv"
D15_TABLE = "analysis/scorecards/stage2_diagnostic/cell_count_learning_curve_v1/cell_count_learning_curve.csv"
D19_TABLE = "analysis/scorecards/stage2_diagnostic/regularization_contract_v1/d19_ifna_pca40_vs_fa40_by_patient.csv"
REFERENCE_TOL = 1e-3  # |AUPRC difference| allowed against the stored atlas / D15 / D19 values
FIG_ARMS = ("tuned", "tuned_l1", "tuned_l2", "fixed_atlas", "scaled_to_n")
# Categorical slots 1-5 of the reference palette (fixed order), plus line styles as a second cue.
ARM_STYLE = {
    "tuned": ("#2a78d6", "-", "o"),
    "tuned_l1": ("#eb6834", "-", "s"),
    "tuned_l2": ("#1baf7a", "-", "^"),
    "fixed_atlas": ("#eda100", "--", "D"),
    "scaled_to_n": ("#e87ba4", ":", "v"),
}
K_RAMP = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#104281", 6: "#0d366b"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# --------------------------------------------------------------------------------------
# context and io helpers
# --------------------------------------------------------------------------------------


def score_path(exp: Path, panel: str, method: str, k: int) -> Path:
    return exp / "stage1_dr" / f"panel_id={panel}" / f"method={method}" / f"k={int(k)}" / "seed=42" / "scores.npy"


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """A table that is missing, empty or unparsable reads as None."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        frame = pd.read_csv(path)
        return frame if len(frame) else None
    except Exception:
        return None


def read_json_any(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c_grid_signature(grid: Sequence[float]) -> str:
    return hashlib.sha1(",".join(repr(float(c)) for c in grid).encode()).hexdigest()[:12]


C_GRID_SIG = c_grid_signature(rc.DEFAULT_C_GRID)


def tune_file(cache: Path, rep: str, penalty: str, subset: str) -> Path:
    return cache / "tune" / rep / penalty / f"S={subset}.json"


def unit_file(cache: Path, rep: str, heldout: str, subset: str) -> Path:
    return cache / "units" / rep / f"h={heldout}" / f"S={subset}.json"


def valid_tune(obj: Optional[Dict[str, Any]], rep: str, penalty: str, subset: str) -> bool:
    return bool(
        obj
        and obj.get("status") == "done"
        and obj.get("schema") == SCHEMA
        and obj.get("representation") == rep
        and obj.get("penalty") == penalty
        and obj.get("subset") == subset
        and obj.get("c_grid_sig") == C_GRID_SIG
        and np.isfinite(float(obj.get("best_C", float("nan"))))
    )


def valid_unit(obj: Optional[Dict[str, Any]], rep: str, heldout: str, subset: str, tune_c: Dict[str, float]) -> bool:
    if not (obj and obj.get("status") == "done" and obj.get("schema") == SCHEMA and obj.get("representation") == rep
            and obj.get("heldout_patient") == heldout and obj.get("subset") == subset):
        return False
    records = obj.get("records") or []
    arms = {r.get("arm") for r in records}
    if not set(pcc.ARMS) <= arms:
        return False
    # out of date if the tuned C it used differs from the current tune cache
    for r in records:
        if r.get("arm") in ("tuned_l1", "tuned_l2"):
            pen = r["arm"].split("_")[1]
            if not math.isclose(float(r.get("C", float("nan"))), float(tune_c.get(pen, float("nan"))), rel_tol=1e-12):
                return False
    return True


class Ctx:
    """Resolved locations, cohort labels, design and run options."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.exp = experiment_dir()
        if args.smoke and not args.out_root:
            raise SystemExit("--smoke needs --out-root (smoke outputs never go into the slug dirs)")
        root = Path(args.out_root).resolve() if args.out_root else self.exp / "analysis"
        self.out_root = root
        self.tables = root / "scorecards" / "stage2_diagnostic" / SLUG
        self.figures = root / "figures" / "stage2_diagnostic" / SLUG
        self.cache = root / "cache" / "stage2_diagnostic" / SLUG
        for path in (self.tables, self.figures, self.cache):
            path.mkdir(parents=True, exist_ok=True)
        self.meta_path = cell_metadata()
        meta = pd.read_csv(self.meta_path, usecols=["y_true", "patient"])
        self.y = meta["y_true"].to_numpy().astype(int)
        self.patients = meta["patient"].astype(str).to_numpy()
        self.n_cells = int(self.y.size)
        self.supported = rc.supported_patients(self.y, self.patients, min_malignant=rc.DEFAULT_MIN_MALIGNANT)
        try:
            configured = list(study_config().get("supported_patients", []))
        except Exception:
            configured = []
        if configured and configured != self.supported:
            raise AssertionError(f"support rule gives {self.supported}, study_config lists {configured}")
        self.unsupported = sorted(set(self.patients) - set(self.supported))
        self.n_by_patient = {p: int(np.sum(self.patients == p)) for p in sorted(set(self.patients))}
        self.mal_by_patient = {p: int(self.y[self.patients == p].sum()) for p in sorted(set(self.patients))}
        self.n_jobs = max(1, min(int(args.n_jobs), MAX_WORKERS))
        # design
        if args.smoke:
            self.design = "full"
            self.heldouts = list(SMOKE["heldout"])
            self.ks = list(SMOKE["ks"])
            self.reps = list(SMOKE["representations"])
        else:
            self.design = args.design
            self.heldouts = [h.strip() for h in args.heldout.split(",")] if args.heldout else list(self.supported)
            self.ks = [int(k) for k in args.k.split(",")] if args.k else list(DEFAULT_KS)
            self.reps = [r.strip() for r in args.representations.split(",")] if args.representations else list(REPRESENTATIONS)
        bad = sorted(set(self.reps) - set(REPRESENTATIONS))
        if bad:
            raise SystemExit(f"unknown representations {bad}; choose from {list(REPRESENTATIONS)}")
        bad = sorted(set(self.heldouts) - set(self.supported))
        if bad:
            raise SystemExit(f"held-out patients must be supported: {bad}")
        self.design_tag = "smoke" if args.smoke else self.design
        self.design_frame = pcc.design_table(self.supported, self.heldouts, self.ks, design=self.design)
        self.atlas = atlas_selection(self.exp)
        self.timings: Dict[str, float] = {}

    def rel(self, path: Path) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(self.exp.resolve()).as_posix()
        except ValueError:
            return "<out-root>/" + path.relative_to(self.out_root).as_posix()

    def n_full_fold(self, heldout: str) -> int:
        return self.n_cells - self.n_by_patient[heldout]

    def train_counts(self, subset: Sequence[str]) -> Tuple[int, int]:
        members = list(subset) + self.unsupported
        return (int(sum(self.n_by_patient[p] for p in members)), int(sum(self.mal_by_patient[p] for p in members)))


def atlas_selection(exp: Path) -> Dict[str, Dict[str, Any]]:
    """Atlas-selected penalty, C and l1_ratio for each representation (as in D15's curve_targets)."""
    table = pd.read_csv(exp / ATLAS_DIAG_TABLE)
    out: Dict[str, Dict[str, Any]] = {}
    for key, (panel, method, k) in REPRESENTATIONS.items():
        match = table.loc[table["stage0_panel_id"].eq(panel) & table["stage1_method"].eq(method) & table["effective_k"].eq(k)]
        if len(match) != 1:
            raise KeyError(f"expected one diagnostic row for {panel} {method} K={k}, found {len(match)}")
        row = match.iloc[0]
        ratio = pd.to_numeric(pd.Series([row["selected_l1_ratio"]]), errors="coerce").iloc[0]
        pen = str(row["selected_penalty"]).lower()
        out[key] = {
            "panel": panel, "method": method, "K": int(k), "short_label": str(row["short_panel_label"]),
            "curve_label": f"{row['short_panel_label']} | {method} K={k}",
            "penalty": pen, "C": float(row["selected_C_or_alpha"]),
            "l1_ratio": None if (pen != "elasticnet" or pd.isna(ratio)) else float(ratio),
        }
    return out


# --------------------------------------------------------------------------------------
# workers (module level so joblib can ship them to processes)
# --------------------------------------------------------------------------------------

_PROC_CACHE: Dict[str, Any] = {}


def _arrays(meta_path: str, scores_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _PROC_CACHE.get("meta_path") != meta_path:
        meta = pd.read_csv(meta_path, usecols=["y_true", "patient"])
        _PROC_CACHE["meta_path"] = meta_path
        _PROC_CACHE["y"] = meta["y_true"].to_numpy().astype(int)
        _PROC_CACHE["patients"] = meta["patient"].astype(str).to_numpy()
    if _PROC_CACHE.get("scores_file") != scores_file:
        scores = np.load(scores_file).astype(np.float32)  # as D19's load_scores
        if scores.shape[0] != _PROC_CACHE["y"].size:
            raise AssertionError(f"{scores_file}: {scores.shape[0]} rows, metadata has {_PROC_CACHE['y'].size}")
        _PROC_CACHE["scores_file"] = scores_file
        _PROC_CACHE["scores"] = scores
    return _PROC_CACHE["scores"], _PROC_CACHE["y"], _PROC_CACHE["patients"]


def run_tune_unit(task: Dict[str, Any]) -> Dict[str, Any]:
    """rc.tune_c_inner_lopo on one training set (subset + unsupported patients)."""
    t0 = time.perf_counter()
    Z, y, pat = _arrays(task["meta_path"], task["scores_file"])
    subset = pcc.parse_subset_tag(task["subset"])
    train = pcc.training_mask(pat, subset, task["supported"])
    res = rc.tune_c_inner_lopo(Z[train], y[train], pat[train], penalty=task["penalty"], c_grid=rc.DEFAULT_C_GRID, n_jobs=1)
    grid = list(res["c_grid"])
    record = {
        "status": "done", "schema": SCHEMA, "representation": task["rep"], "penalty": res["penalty"],
        "subset": task["subset"], "k": len(subset), "c_grid_sig": C_GRID_SIG,
        "best_C": res["best_C"], "best_score": res["best_score"], "best_nonzero_mean": res["best_nonzero_mean"],
        "at_grid_lower_bound": bool(res["at_grid_lower_bound"]),
        "at_grid_upper_bound": bool(np.isclose(res["best_C"], max(grid), rtol=1e-9)),
        "n_inner_folds": res["n_inner_folds"], "inner_folds": res["inner_folds"], "fallback": res["fallback"],
        "fallback_reason": res["fallback_reason"], "criterion": res["criterion"], "tie_break": res["tie_break"],
        "n_train": res["n_train"], "n_train_malignant": res["n_train_malignant"],
        "scores_by_C": [[float(c), float(res["scores_by_C"][c]), float(res["nonzero_by_C"][c])] for c in grid],
        "per_fold": res["per_fold"], "seconds": time.perf_counter() - t0,
        "basis": pcc.BASIS_LABEL, "contract_version": rc.CONTRACT_VERSION,
    }
    atomic_write_json(record, Path(task["out"]))
    print(f"[d15b tune] {task['rep']} {task['penalty']} S={task['subset']} best_C={res['best_C']:.3g} "
          f"inner={res['best_score']:.4f} folds={res['n_inner_folds']}{' (fallback)' if res['fallback'] else ''} "
          f"{record['seconds']:.1f}s", flush=True)
    return {"out": task["out"], "seconds": record["seconds"]}


def _fit_arm(Z: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, penalty: str, C: float,
             l1_ratio: Optional[float]) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    t0 = time.perf_counter()
    est = rc.fit_probe(Z[train], y[train], penalty=penalty, C=float(C), l1_ratio=l1_ratio)
    coef = rc.probe_coef(est)
    prob = rc.probe_scores(est, Z[test])
    ap, prevalence, lift = rc.heldout_lift(prob, y[test])
    auroc = float(roc_auc_score(y[test], prob)) if 0 < y[test].sum() < y[test].size else float("nan")
    margin = None
    if penalty in ("l1", "elasticnet"):
        margin = float(np.max(rc.l1_entry_margins(Z[train], y[train], float(C), l1_ratio=l1_ratio if penalty == "elasticnet" else None)))
    return {
        "penalty": penalty, "C": float(C), "l1_ratio": l1_ratio, "heldout_ap": ap, "prevalence": prevalence,
        "heldout_lift": lift, "heldout_auroc": auroc, "n_nonzero": int(np.sum(np.abs(coef) > rc.COEF_NONZERO_TOL)),
        "max_entry_margin_at_w0": margin, "fit_seconds": time.perf_counter() - t0,
    }


def run_outer_unit(task: Dict[str, Any]) -> Dict[str, Any]:
    """Outer fits for one (representation, held-out patient, subset): all five arms."""
    t0 = time.perf_counter()
    Z, y, pat = _arrays(task["meta_path"], task["scores_file"])
    subset = pcc.parse_subset_tag(task["subset"])
    h = task["heldout"]
    train = pcc.training_mask(pat, subset, task["supported"], heldout=h)
    test = pat == h
    n_train = int(train.sum())
    atlas = task["atlas"]
    base = {
        "representation": task["rep"], "curve_label": atlas["curve_label"], "stage1_method": atlas["method"],
        "effective_k": atlas["K"], "heldout_patient": h, "k": len(subset), "subset": task["subset"],
        "n_train": n_train, "n_train_malignant": int(y[train].sum()),
        "n_train_patients": int(np.unique(pat[train]).size), "n_test": int(test.sum()),
        "n_test_malignant": int(y[test].sum()), "basis": pcc.BASIS_LABEL,
    }
    records: List[Dict[str, Any]] = []
    tune = task["tune"]
    for pen in TUNE_PENALTIES:
        t = tune[pen]
        rec = dict(base, arm=f"tuned_{pen}", C_source="tune_c_inner_lopo",
                   inner_best_score=t["best_score"], inner_n_folds=t["n_inner_folds"], inner_fallback=t["fallback"],
                   inner_fallback_reason=t["fallback_reason"], at_grid_lower_bound=t["at_grid_lower_bound"],
                   at_grid_upper_bound=t["at_grid_upper_bound"])
        rec.update(_fit_arm(Z, y, train, test, pen, t["best_C"], None))
        records.append(rec)
    chosen = pcc.select_tuned_penalty({p: tune[p]["best_score"] for p in TUNE_PENALTIES})
    sel = dict(next(r for r in records if r["arm"] == f"tuned_{chosen}"))
    sel.update(arm="tuned", selected_penalty=chosen, C_source=f"tune_c_inner_lopo ({chosen} chosen by inner score)")
    records.insert(0, sel)
    fixed = dict(base, arm="fixed_atlas", C_source="D1 atlas selection (global, fixed)",
                 at_grid_lower_bound=rc.c_at_grid_lower_bound(atlas["C"]), at_grid_upper_bound=False)
    fixed.update(_fit_arm(Z, y, train, test, atlas["penalty"], atlas["C"], atlas["l1_ratio"]))
    records.append(fixed)
    n_ref = int(task["n_full_fold"])
    c_scaled = rc.c_scaled_to_n(atlas["C"], n_ref, n_train)
    scaled = dict(base, arm="scaled_to_n", C_source=f"c_scaled_to_n(C_atlas, n_ref={n_ref}, n_train={n_train})",
                  at_grid_lower_bound=rc.c_at_grid_lower_bound(c_scaled), at_grid_upper_bound=False)
    scaled.update(_fit_arm(Z, y, train, test, atlas["penalty"], c_scaled, atlas["l1_ratio"]))
    records.append(scaled)
    obj = {"status": "done", "schema": SCHEMA, "representation": task["rep"], "heldout_patient": h,
           "subset": task["subset"], "k": len(subset), "records": records, "seconds": time.perf_counter() - t0}
    atomic_write_json(obj, Path(task["out"]))
    tuned = records[0]
    print(f"[d15b unit] {task['rep']} h={h} S={task['subset']} tuned={tuned['penalty']} lift={100 * tuned['heldout_lift']:.2f}pp "
          f"fixed={100 * fixed['heldout_lift']:.2f}pp {obj['seconds']:.1f}s", flush=True)
    return {"out": task["out"], "seconds": obj["seconds"]}


def _parallel(fn, tasks: List[Dict[str, Any]], n_jobs: int) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    if n_jobs == 1:
        return [fn(t) for t in tasks]
    from joblib import Parallel, delayed

    return list(Parallel(n_jobs=n_jobs, backend="loky", batch_size=1, pre_dispatch="all")(delayed(fn)(t) for t in tasks))


# --------------------------------------------------------------------------------------
# parts
# --------------------------------------------------------------------------------------


def _expected_cost(rep: str, penalty: str, k: int, n_train: int) -> float:
    """Ordering heuristic only (heaviest first): promax L1 and the k = 1 fallback are slowest."""
    method = REPRESENTATIONS[rep][1]
    per_fold = {("factosig_promax", "l1"): 3.0, ("factosig_promax", "l2"): 1.5}.get((method, penalty), 0.5)
    folds = 5 * (2.5 if (method, penalty) == ("factosig_promax", "l1") else 1.0) if k == 1 else k
    return per_fold * folds * n_train / 40_000


def run_tune(ctx: Ctx) -> Dict[str, Any]:
    subsets = sorted(set(ctx.design_frame["subset"]))
    tasks, skipped = [], 0
    for rep in ctx.reps:
        panel, method, k = REPRESENTATIONS[rep]
        sfile = str(score_path(ctx.exp, panel, method, k))
        for subset in subsets:
            for pen in TUNE_PENALTIES:
                out = tune_file(ctx.cache, rep, pen, subset)
                if ctx.args.resume and valid_tune(read_json_any(out), rep, pen, subset):
                    skipped += 1
                    continue
                n_train, _ = ctx.train_counts(pcc.parse_subset_tag(subset))
                tasks.append({"rep": rep, "penalty": pen, "subset": subset, "supported": ctx.supported,
                              "meta_path": str(ctx.meta_path), "scores_file": sfile, "out": str(out),
                              "cost": _expected_cost(rep, pen, len(pcc.parse_subset_tag(subset)), n_train)})
    tasks.sort(key=lambda t: -t["cost"])
    print(f"[d15b] tune: {len(tasks)} to run, {skipped} finished (resume), n_jobs={ctx.n_jobs}", flush=True)
    t0 = time.perf_counter()
    _parallel(run_tune_unit, tasks, ctx.n_jobs)
    return {"tune_units_run": len(tasks), "tune_units_skipped": skipped, "tune_wall_seconds": time.perf_counter() - t0}


def _tune_summary(ctx: Ctx, rep: str, subset: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for pen in TUNE_PENALTIES:
        obj = read_json_any(tune_file(ctx.cache, rep, pen, subset))
        if not valid_tune(obj, rep, pen, subset):
            raise FileNotFoundError(f"missing tune unit {rep} {pen} S={subset}; run --parts tune first")
        out[pen] = {key: obj[key] for key in ("best_C", "best_score", "n_inner_folds", "fallback", "fallback_reason",
                                               "at_grid_lower_bound", "at_grid_upper_bound")}
    return out


def run_units(ctx: Ctx) -> Dict[str, Any]:
    tasks, skipped = [], 0
    for rep in ctx.reps:
        panel, method, k = REPRESENTATIONS[rep]
        sfile = str(score_path(ctx.exp, panel, method, k))
        atlas = ctx.atlas[rep]
        for row in ctx.design_frame.itertuples(index=False):
            out = unit_file(ctx.cache, rep, row.heldout_patient, row.subset)
            tune = _tune_summary(ctx, rep, row.subset)
            if ctx.args.resume and valid_unit(read_json_any(out), rep, row.heldout_patient, row.subset,
                                              {p: tune[p]["best_C"] for p in TUNE_PENALTIES}):
                skipped += 1
                continue
            n_train, _ = ctx.train_counts(pcc.parse_subset_tag(row.subset))
            tasks.append({"rep": rep, "heldout": row.heldout_patient, "subset": row.subset, "supported": ctx.supported,
                          "meta_path": str(ctx.meta_path), "scores_file": sfile, "out": str(out), "atlas": atlas,
                          "tune": tune, "n_full_fold": ctx.n_full_fold(row.heldout_patient), "cost": n_train})
    tasks.sort(key=lambda t: -t["cost"])
    print(f"[d15b] units: {len(tasks)} to run, {skipped} finished (resume), n_jobs={ctx.n_jobs}", flush=True)
    t0 = time.perf_counter()
    _parallel(run_outer_unit, tasks, ctx.n_jobs)
    return {"outer_units_run": len(tasks), "outer_units_skipped": skipped, "outer_wall_seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def collect(ctx: Ctx) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Unit records (one row per unit x arm) and tune records for the requested design."""
    rows, missing = [], []
    for rep in ctx.reps:
        for row in ctx.design_frame.itertuples(index=False):
            obj = read_json_any(unit_file(ctx.cache, rep, row.heldout_patient, row.subset))
            if not (obj and obj.get("status") == "done" and obj.get("schema") == SCHEMA):
                missing.append(f"{rep}|{row.heldout_patient}|{row.subset}")
                continue
            for rec in obj["records"]:
                rows.append(dict(rec, unit_seconds=obj.get("seconds")))
    units = pd.DataFrame(rows)
    tune_rows = []
    for rep in ctx.reps:
        for subset in sorted(set(ctx.design_frame["subset"])):
            for pen in TUNE_PENALTIES:
                obj = read_json_any(tune_file(ctx.cache, rep, pen, subset))
                if not valid_tune(obj, rep, pen, subset):
                    continue
                tune_rows.append({key: obj[key] for key in (
                    "representation", "penalty", "subset", "k", "best_C", "best_score", "best_nonzero_mean",
                    "at_grid_lower_bound", "at_grid_upper_bound", "n_inner_folds", "fallback", "fallback_reason",
                    "n_train", "n_train_malignant", "seconds")} | {"inner_folds": ";".join(obj["inner_folds"])})
    return units, pd.DataFrame(tune_rows), missing


def tune_scores_long(ctx: Ctx) -> pd.DataFrame:
    rows = []
    for rep in ctx.reps:
        for subset in sorted(set(ctx.design_frame["subset"])):
            for pen in TUNE_PENALTIES:
                obj = read_json_any(tune_file(ctx.cache, rep, pen, subset))
                if not valid_tune(obj, rep, pen, subset):
                    continue
                for C, score, nnz in obj["scores_by_C"]:
                    rows.append({"representation": rep, "penalty": pen, "subset": subset, "k": obj["k"], "C": C,
                                 "inner_median_lift": score, "inner_mean_nonzero": nnz, "is_best": math.isclose(C, obj["best_C"])})
    return pd.DataFrame(rows)


def reference_checks(ctx: Ctx, units: pd.DataFrame) -> pd.DataFrame:
    """k = 6 (ordinary LOPO fold) against stored atlas, D15 and D19 values."""
    rows = []
    full = units.loc[units["k"].eq(len(ctx.supported) - 1)]
    if full.empty:
        return pd.DataFrame(rows)
    atlas = pd.read_csv(ctx.exp / ATLAS_PATIENT_TABLE, usecols=["stage0_panel_id", "configuration", "heldout_patient", "heldout_auprc"])
    d15 = pd.read_csv(ctx.exp / D15_TABLE) if (ctx.exp / D15_TABLE).is_file() else None
    d19 = pd.read_csv(ctx.exp / D19_TABLE) if (ctx.exp / D19_TABLE).is_file() else None
    for r in full.itertuples(index=False):
        rep = r.representation
        meta = ctx.atlas[rep]
        if r.arm == "fixed_atlas":
            m = atlas.loc[atlas["stage0_panel_id"].eq(meta["panel"]) & atlas["configuration"].eq(f"{meta['method']}|K={meta['K']}")
                          & atlas["heldout_patient"].eq(r.heldout_patient)]
            if len(m) == 1:
                ref = float(m["heldout_auprc"].iloc[0])
                rows.append({"check": "fixed_atlas k=6 vs D1 atlas (cuML) heldout AUPRC", "representation": rep,
                             "heldout_patient": r.heldout_patient, "value": r.heldout_ap, "reference": ref,
                             "abs_diff": abs(r.heldout_ap - ref), "pass": abs(r.heldout_ap - ref) <= REFERENCE_TOL})
            if d15 is not None:
                m = d15.loc[d15["curve_label"].eq(meta["curve_label"]) & d15["heldout_patient"].eq(r.heldout_patient)
                            & d15["is_full_training"].astype(bool)]
                if len(m) == 1:
                    ref = float(m["heldout_auprc"].iloc[0])
                    rows.append({"check": "fixed_atlas k=6 vs D15 full-training refit AUPRC", "representation": rep,
                                 "heldout_patient": r.heldout_patient, "value": r.heldout_ap, "reference": ref,
                                 "abs_diff": abs(r.heldout_ap - ref), "pass": abs(r.heldout_ap - ref) <= REFERENCE_TOL})
        if d19 is not None and rep in ("ifna_pca40", "ifna_fa40") and r.arm in ("tuned_l1", "tuned_l2"):
            prefix = "pca40" if rep == "ifna_pca40" else "fa40"
            m = d19.loc[d19["heldout_patient"].eq(r.heldout_patient)]
            if len(m) == 1:
                ref_lift = float(m[f"{prefix}_{r.arm}_lift"].iloc[0])
                ref_C = float(m[f"{prefix}_{r.arm}_C"].iloc[0])
                same_c = math.isclose(ref_C, r.C, rel_tol=1e-9)
                rows.append({"check": f"{r.arm} k=6 vs D19 tuned (same contract)", "representation": rep,
                             "heldout_patient": r.heldout_patient, "value": r.heldout_lift, "reference": ref_lift,
                             "abs_diff": abs(r.heldout_lift - ref_lift), "C": r.C, "reference_C": ref_C,
                             "pass": bool(same_c and abs(r.heldout_lift - ref_lift) <= REFERENCE_TOL)})
    return pd.DataFrame(rows)


def timing_model(ctx: Ctx, tune: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Mean seconds per tune unit by (representation, penalty, k) and per outer unit by
    (representation, k), and the implied full-design CPU and wall time at the current n_jobs."""
    if tune.empty or units.empty:
        return pd.DataFrame()
    n_sup = len(ctx.supported)
    rows = []
    t = tune.groupby(["representation", "penalty", "k"], as_index=False)["seconds"].mean()
    for r in t.itertuples(index=False):
        n_full = math.comb(n_sup, int(r.k))  # distinct training subsets of size k in the full design
        rows.append({"kind": "tune", "representation": r.representation, "penalty": r.penalty, "k": int(r.k),
                     "mean_seconds": r.seconds, "n_units_full_design": n_full, "cpu_seconds_full_design": r.seconds * n_full})
    u = units.loc[units["arm"].eq("tuned")].groupby(["representation", "k"], as_index=False)["unit_seconds"].mean()
    for r in u.itertuples(index=False):
        n_full = n_sup * math.comb(n_sup - 1, int(r.k))
        rows.append({"kind": "outer", "representation": r.representation, "penalty": "", "k": int(r.k),
                     "mean_seconds": r.unit_seconds, "n_units_full_design": n_full, "cpu_seconds_full_design": r.unit_seconds * n_full})
    return pd.DataFrame(rows)


def make_figures(ctx: Ctx, units: pd.DataFrame, by_patient: pd.DataFrame, curve: pd.DataFrame, power: pd.DataFrame,
                 inclusion: pd.DataFrame) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False})
    tag = ctx.design_tag
    reps = [r for r in ctx.reps if r in set(units["representation"])]
    paths: List[Path] = []
    basis_note = "Frozen transductive Stage 1 basis (fit on all cells, incl. held-out and excluded patients)"

    def save(fig, stem):
        png = ctx.figures / f"{stem}.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(ctx.figures / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        paths.append(png)

    def grid_for(n):
        ncols = 2 if n > 1 else 1
        return (n + ncols - 1) // ncols, ncols

    # A: cohort median lift vs k, five C policies
    nrows, ncols = grid_for(len(reps))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.9 * nrows), squeeze=False)
    for ax, rep in zip(axes.ravel(), reps):
        cur = curve.loc[curve["representation"].eq(rep)]
        for arm in FIG_ARMS:
            c = cur.loc[cur["arm"].eq(arm)].sort_values("k")
            if c.empty:
                continue
            color, ls, mk = ARM_STYLE[arm]
            if arm == "tuned":
                ax.fill_between(c["k"], 100 * c["median_lift_lo"], 100 * c["median_lift_hi"], color=color, alpha=0.14, lw=0)
            top = arm == "tuned"
            ax.plot(c["k"], 100 * c["median_lift"], color=color, ls=ls, marker=mk, ms=6 if top else 5, lw=2.6 if top else 1.8,
                    label=arm, zorder=5 if top else 3)
        ax.axhline(0, color=INK2, lw=0.8)
        ks = sorted(cur["k"].unique())
        mal = cur.loc[cur["arm"].eq("tuned")].set_index("k")["median_train_malignant"]
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k}\n({mal.get(k, float('nan')):.0f})" for k in ks])
        ax.set_xlabel("training supported patients k\n(median training malignant cells)")
        ax.set_ylabel("median held-out lift (AP − π, pp)")
        ax.set_title(ctx.atlas[rep]["curve_label"], fontsize=10)
        ax.grid(True, axis="y", color=GRID, lw=0.6)
    for ax in axes.ravel()[len(reps):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, [pcc.ARM_LABELS[l] for l in labels], loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02 - 0.03 * (nrows == 1)))
    fig.suptitle(f"D15b patient-count curve ({tag} design): median over held-out patients; band = patient-bootstrap 95% CI (tuned)\n{basis_note}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, f"D15bA_{tag}_median_lift_vs_k_by_C_policy")

    # B: per held-out patient, lift vs training malignant cells, coloured by k (tuned arm)
    hs = sorted(units["heldout_patient"].unique())
    fig, axes = plt.subplots(len(reps), len(hs), figsize=(2.1 * len(hs) + 0.8, 2.0 * len(reps) + 0.9), squeeze=False, sharey="row")
    tuned = units.loc[units["arm"].eq("tuned")]
    for i, rep in enumerate(reps):
        for j, h in enumerate(hs):
            ax = axes[i, j]
            sub = tuned.loc[tuned["representation"].eq(rep) & tuned["heldout_patient"].eq(h)]
            for k in sorted(sub["k"].unique()):
                s = sub.loc[sub["k"].eq(k)]
                ax.scatter(s["n_train_malignant"], 100 * s["heldout_lift"], s=16, color=K_RAMP.get(int(k), INK2),
                           edgecolor="white", linewidth=0.5, label=f"k={k}", zorder=3)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_xscale("log")
            ax.grid(True, color=GRID, lw=0.5)
            if i == 0:
                ax.set_title(f"held out {h}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{ctx.atlas[rep]['short_label']}\n{ctx.atlas[rep]['method']} K40\nlift (pp)", fontsize=8)
            if i == len(reps) - 1:
                ax.set_xlabel("training malignant cells", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"D15b ({tag}): held-out lift per training subset vs training malignant cells (tuned arm); {basis_note}", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save(fig, f"D15bB_{tag}_lift_vs_training_malignant_by_patient")

    # C: exploratory inverse power-law validation (tuned arm, median)
    if power is not None and not power.empty:
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)
        for ax, rep in zip(axes.ravel(), reps):
            c = curve.loc[curve["representation"].eq(rep) & curve["arm"].eq("tuned")].sort_values("k")
            ax.errorbar(c["k"], 100 * c["median_lift"], yerr=[100 * (c["median_lift"] - c["median_lift_lo"]), 100 * (c["median_lift_hi"] - c["median_lift"])],
                        fmt="o", color=ARM_STYLE["tuned"][0], ms=5, capsize=3, lw=1.2, label="observed median (95% CI)")
            p = power.loc[power["representation"].eq(rep) & power["arm"].eq("tuned") & power["stat"].eq("median") & power["target_k"].eq(5)]
            if len(p) == 1 and bool(p["fit_converged"].iloc[0]):
                rec = p.iloc[0]
                kk = np.linspace(1, max(c["k"].max(), 5), 100)
                ax.plot(kk, 100 * pcc.inverse_power_law(kk, rec["fit_a"], rec["fit_b"], rec["fit_c"]), color=INK, ls="--", lw=1.2,
                        label="fit on k ≤ 4")
                ax.plot([5], [100 * rec["pred_target"]], marker="x", color=INK, ms=8, mew=2, ls="none", label="predicted k = 5")
                verdict = "supported" if bool(rec["supported"]) else f"not supported: {rec['reason']}"
                ax.text(0.98, 0.97, verdict, transform=ax.transAxes, fontsize=7, color=INK2, va="top", ha="right",
                        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.25"}, zorder=6)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_xticks(sorted(c["k"].unique()))
            ax.set_xlabel("training supported patients k")
            ax.set_ylabel("median lift (pp)")
            ax.set_title(ctx.atlas[rep]["curve_label"], fontsize=10)
            ax.grid(True, axis="y", color=GRID, lw=0.6)
        for ax in axes.ravel()[len(reps):]:
            ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.suptitle(f"D15b ({tag}) EXPLORATORY: weighted inverse power law a − b·k^(−c), fit on k ≤ 4, predicting k = 5 (tuned arm)", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        save(fig, f"D15bC_{tag}_power_law_validation")

    # D: patient-inclusion effect heatmaps (tuned arm)
    if inclusion is not None and not inclusion.empty:
        cmap = LinearSegmentedColormap.from_list("bluered", ["#e34948", "#f0efec", "#2a78d6"])
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.1 * nrows), squeeze=False)
        inc = inclusion.loc[inclusion["arm"].eq("tuned")]
        vmax = float(np.nanmax(np.abs(100 * inc["inclusion_effect"]))) if len(inc) else 1.0
        vmax = max(vmax, 0.5)
        for ax, rep in zip(axes.ravel(), reps):
            sub = inc.loc[inc["representation"].eq(rep)]
            mat = sub.pivot_table(index="heldout_patient", columns="included_patient", values="inclusion_effect").reindex(index=hs, columns=hs)
            im = ax.imshow(100 * mat.to_numpy(dtype=float), cmap=cmap, vmin=-vmax, vmax=vmax)
            for a in range(len(hs)):
                for b in range(len(hs)):
                    v = mat.to_numpy(dtype=float)[a, b]
                    if np.isfinite(v):
                        ax.text(b, a, f"{100 * v:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
            ax.set_xticks(range(len(hs)))
            ax.set_xticklabels(hs, rotation=45, fontsize=7)
            ax.set_yticks(range(len(hs)))
            ax.set_yticklabels(hs, fontsize=7)
            ax.set_xlabel("training patient j included")
            ax.set_ylabel("held-out patient")
            ax.set_title(ctx.atlas[rep]["curve_label"], fontsize=9)
        for ax in axes.ravel()[len(reps):]:
            ax.axis("off")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        cbar.set_label("mean lift with j − without j, matched k (pp)")
        fig.suptitle(f"D15b ({tag}): patient-inclusion effect, tuned arm (descriptive)", fontsize=10)
        save(fig, f"D15bD_{tag}_patient_inclusion_effect")
    return paths


def _pp(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.2f}"


def write_findings(ctx: Ctx) -> Path:
    """Findings for every design whose curve table exists in the slug dir (full first)."""
    lines = [
        f"# D15b patient-count learning curve ({SLUG})",
        "",
        f"Analysis {ANALYSIS_ID}, task {TASK_ID}, devlog {DEVLOG_ID}. Generated by "
        "scripts/mrd_stage0_2/workstreams/W6_learning_curves/run_d15b_patient_count_curve.py.",
        "",
        "**Basis: frozen transductive Stage 1.** Every training subset uses the stored Stage 1 scores, fit on all "
        "60,118 cells including the held-out patient and the excluded patients. Only the Stage 2 probe is refit. "
        "The curve therefore measures how Stage 2 label transfer depends on the number of labelled training patients. "
        "An inductive pipeline would also learn its basis from the k training patients. Nothing here varies the Stage 1 "
        "fit set with k, so how a per-subset basis would change the curve's level or slope is unknown in either direction "
        "until the basis is refit per training subset (W6.6, using the W2 refit code). On its own the curve cannot "
        "justify sequencing more patients: that also needs new patients scored as external points (W6.5, W8.1).",
        "",
        "Design: for each held-out supported patient h, training = k of the other 6 supported patients + all 6 "
        "unsupported patients (always included); k = 6 is the ordinary LOPO fold. C policies: tuned (penalty L1/L2 and "
        "C by inner LOPO per training set; k = 1 uses the contract's stratified 5-fold CV fallback), tuned_l1, tuned_l2, "
        "fixed_atlas (D15-comparable; its C was selected by the D1 rule with held-out outcomes, so it is not nested), "
        "scaled_to_n (atlas C scaled to keep the full-fold L1 entry threshold). "
        "lift = held-out AP − held-out prevalence π; an all-zero probe scores the constant-score no-skill AP = π (lift 0). "
        "Per-k summaries are over held-out patients of the per-(h, k) mean lift across subsets; CIs resample held-out "
        "patients (7 clusters, 2,000 draws), so they are coarse and ignore the dependence among subsets that share patients.",
        "",
        "**Tuning protocol changes at k = 1.** With one supported training patient, inner LOPO is impossible, so the tuned "
        "arms fall back to stratified 5-fold CV over cells (label strata), which mixes cells of the same patients across "
        "inner folds and tunes for within-patient fit; from k = 2 on, C is tuned by inner leave-one-patient-out. The k = 1 "
        "point of the tuned curves and any paired gain over k = 1 therefore mix this protocol switch with the patient-count "
        "effect. Paired gains over k = 2 (same protocol at both ends) are the primary comparison; gains over k = 1 are "
        "kept for reference.",
        "",
        "**Reduced-design balance.** The cyclic reduced design is balanced over training-patient membership at k = 1, 3, 4 "
        "and 6 only. At k = 2 and k = 5 the two patients next to the held-out one in sorted order are under-represented "
        "(1/5 and 1/2 of that patient's subsets, against 1/3 and 5/6 in the full enumeration), so reduced-design k = 2 and "
        "k = 5 means average over a skewed training composition, and the k = 5 power-law validation target is not usable "
        "from the reduced design. The full design (all subsets) removes this.",
        "",
    ]
    found = False
    for design in ("full", "reduced", "smoke"):
        curve = read_csv_safe(ctx.tables / f"d15b_{design}_curve_by_k.csv")
        if curve is None:
            continue
        found = True
        lines += [f"## {design} design", ""]
        status_path = ctx.tables / f"d15b_{design}_unit_status.json"
        status = read_json_any(status_path) or {}
        if status:
            lines += [f"Units: {status.get('n_units_present')} of {status.get('n_units_expected')} (representation, held-out patient, subset) "
                      f"units present; missing: {status.get('n_units_missing')}.", ""]
        for rep in [r for r in REPRESENTATIONS if r in set(curve["representation"])]:
            cur = curve.loc[curve["representation"].eq(rep)]
            lines += [f"### {ctx.atlas[rep]['curve_label']} (atlas {ctx.atlas[rep]['penalty']}, C={ctx.atlas[rep]['C']:g})", "",
                      "| k | median training malignant | tuned median lift pp [95% CI] | tuned P20 | tuned_l1 median | tuned_l2 median | fixed_atlas median | scaled_to_n median | tuned frac all-zero |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for k in sorted(cur["k"].unique()):
                row = {a: cur.loc[cur["arm"].eq(a) & cur["k"].eq(k)] for a in pcc.ARMS}

                def g(a, col):
                    return float(row[a][col].iloc[0]) if len(row[a]) else float("nan")

                lines.append(f"| {k} | {g('tuned', 'median_train_malignant'):.0f} | {_pp(g('tuned', 'median_lift'))} "
                             f"[{_pp(g('tuned', 'median_lift_lo'))}, {_pp(g('tuned', 'median_lift_hi'))}] | {_pp(g('tuned', 'p20_lift'))} | "
                             f"{_pp(g('tuned_l1', 'median_lift'))} | {_pp(g('tuned_l2', 'median_lift'))} | {_pp(g('fixed_atlas', 'median_lift'))} | "
                             f"{_pp(g('scaled_to_n', 'median_lift'))} | {g('tuned', 'frac_all_zero'):.2f} |")
            lines.append("")
        bp = read_csv_safe(ctx.tables / f"d15b_{design}_by_patient_k.csv")
        if bp is not None:
            t = bp.loc[bp["arm"].eq("tuned")]
            ks_all = sorted(t["k"].unique())
            lines += ["Per held-out patient, tuned arm: mean lift over training subsets (pp), by k:", "",
                      "| representation | held out | π | " + " | ".join(f"k={k}" for k in ks_all) + " |",
                      "|---|---|---|" + "---|" * len(ks_all)]
            for (rep, h), f in t.groupby(["representation", "heldout_patient"], sort=True):
                vals = f.set_index("k")["lift_mean"]
                lines.append(f"| {rep} | {h} | {f['prevalence'].iloc[0]:.3f} | " + " | ".join(_pp(vals.get(k, float('nan'))) for k in ks_all) + " |")
            lines.append("")
        inc = read_csv_safe(ctx.tables / f"d15b_{design}_patient_inclusion_effect.csv")
        if inc is not None:
            t = inc.loc[inc["arm"].eq("tuned")]
            med = t.groupby(["representation", "included_patient"])["inclusion_effect"].median().unstack()
            cols = list(med.columns)
            lines += ["Patient-inclusion effect, tuned arm (descriptive; median over held-out patients of mean lift with j minus "
                      "without j at matched k, pp; subsets with j also differ in their other members):", "",
                      "| representation | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for rep, row in med.iterrows():
                lines.append(f"| {rep} | " + " | ".join(_pp(row[c]) for c in cols) + " |")
            lines.append("")
        for k_from, note in ((2, "primary: inner-LOPO tuning at both ends"),
                             (1, "reference only: k = 1 uses the stratified 5-fold CV fallback")):
            gain = read_csv_safe(ctx.tables / f"d15b_{design}_gain_vs_k{k_from}.csv")
            if gain is None:
                continue
            gsel = gain.loc[gain["arm"].eq("tuned") & (gain["k_to"] > k_from)]
            if len(gsel):
                caveat = " k = 2 and k = 5 are unbalanced in the reduced design." if design == "reduced" else ""
                lines += [f"Paired gain over k = {k_from} ({note}; tuned arm; per held-out patient, then median over patients).{caveat}", "",
                          "| representation | k_to | median gain pp [95% CI] | patients gain > 0 / < 0 |", "|---|---|---|---|"]
                for r in gsel.itertuples(index=False):
                    lines.append(f"| {r.representation} | {r.k_to} | {_pp(r.median_gain)} [{_pp(r.median_gain_lo)}, {_pp(r.median_gain_hi)}] | "
                                 f"{r.n_patients_gain_gt_0} / {r.n_patients_gain_lt_0} |")
                lines.append("")
        power = read_csv_safe(ctx.tables / f"d15b_{design}_power_law_fit.csv")
        if power is not None:
            psel = power.loc[power["arm"].eq("tuned") & power["stat"].eq("median")]
            if len(psel):
                lines += ["Inverse power law a − b·k^(−c) (EXPLORATORY; weighted by 1/bootstrap SE²; tuned arm, median). "
                          "A fit is reportable only if the primary validation (fit k ≤ 4, predict k = 5) passes; the "
                          "secondary row (fit k ≤ 5, predict k = 6) is shown for completeness. With CIs this wide, "
                          "'prediction inside the observed CI' is a weak check.", "",
                          "| representation | validation | fit k | target k | predicted pp | observed pp [95% CI] | supported | reason |",
                          "|---|---|---|---|---|---|---|---|"]
                for r in psel.itertuples(index=False):
                    lines.append(f"| {r.representation} | {r.validation} | {r.fit_ks} | {r.target_k} | {_pp(r.pred_target)} | {_pp(r.obs_target)} "
                                 f"[{_pp(r.obs_target_lo)}, {_pp(r.obs_target_hi)}] | {r.supported} | {r.reason} |")
                prim = psel.loc[psel["validation"].eq("primary")]
                ok = sorted(prim.loc[prim["supported"].astype(bool), "representation"])
                if design == "reduced":
                    ok = []  # the k = 5 validation target is unbalanced in the reduced design
                lines += ["", f"Power-law verdict (primary validation, tuned median): reportable for {', '.join(ok) if ok else 'none'} "
                          f"of {prim['representation'].nunique()} representations"
                          + (" (none by rule: the reduced design's k = 5 target is unbalanced)." if design == "reduced" else "."), ""]
        tm = read_csv_safe(ctx.tables / f"d15b_{design}_timing_model.csv")
        if tm is not None and set(tm["representation"]) == set(REPRESENTATIONS):
            cpu = float(tm["cpu_seconds_full_design"].sum())
            lines += [f"Full-design cost implied by this run's unit timings (d15b_{design}_timing_model.csv): about {cpu / 60:.0f} CPU-minutes, "
                      f"about {cpu / 60 / MAX_WORKERS:.0f} minutes wall at {MAX_WORKERS} workers with perfect packing.", ""]
        ref = read_csv_safe(ctx.tables / f"d15b_{design}_reference_checks.csv")
        if ref is not None:
            if len(ref):
                summ = ref.groupby("check").agg(n=("pass", "size"), n_pass=("pass", "sum"), max_abs_diff=("abs_diff", "max")).reset_index()
                lines += ["Reference checks at k = 6 (the ordinary LOPO fold):", "", "| check | n | pass | max abs diff |", "|---|---|---|---|"]
                for r in summ.itertuples(index=False):
                    lines.append(f"| {r.check} | {r.n} | {int(r.n_pass)} | {r.max_abs_diff:.2e} |")
                lines.append("")
    if not found:
        lines += ["No curve table yet.", ""]
    lines += [
        "## Claim boundary",
        "",
        "Descriptive results on a transductive Stage 1 basis over seven supported patients; they do not establish "
        "inductive deployment, causal biology or recurrence risk. The patient-count curve is not by itself a "
        "justification for sequencing more patients (see the basis note above). Power-law fits are exploratory.",
        "",
    ]
    path = ctx.tables / f"{SLUG}_findings.md"
    atomic_write_text("\n".join(lines), path)
    return path


def run_report(ctx: Ctx) -> Dict[str, Any]:
    units, tune, missing = collect(ctx)
    tag = ctx.design_tag
    n_expected = len(ctx.design_frame) * len(ctx.reps)
    status = {"design": tag, "n_units_expected": n_expected, "n_units_present": n_expected - len(missing),
              "n_units_missing": len(missing), "missing_examples": missing[:20]}
    atomic_write_json(status, ctx.tables / f"d15b_{tag}_unit_status.json")
    if units.empty:
        print("[d15b] report: no finished units", flush=True)
        return status
    if missing:
        print(f"[d15b] report: {len(missing)} units missing; summaries use the finished ones", flush=True)
    units = units.sort_values(["representation", "arm", "heldout_patient", "k", "subset"]).reset_index(drop=True)
    atomic_write_csv(ctx.design_frame, ctx.tables / f"d15b_{tag}_design.csv")
    atomic_write_csv(units, ctx.tables / f"d15b_{tag}_unit_results.csv")
    atomic_write_csv(tune, ctx.tables / f"d15b_{tag}_tuning_by_subset.csv")
    atomic_write_csv(tune_scores_long(ctx), ctx.tables / f"d15b_{tag}_tuning_scores_by_C.csv")
    by_patient = pcc.summarize_by_patient_k(units)
    atomic_write_csv(by_patient, ctx.tables / f"d15b_{tag}_by_patient_k.csv")
    curve = pcc.curve_by_k(by_patient, n_boot=ctx.args.n_boot, seed=SEED, units=units)
    curve.insert(2, "basis", pcc.BASIS_LABEL)
    atomic_write_csv(curve, ctx.tables / f"d15b_{tag}_curve_by_k.csv")
    gain = pcc.paired_gain(by_patient, k_from=min(ctx.ks), n_boot=ctx.args.n_boot, seed=SEED)
    atomic_write_csv(gain, ctx.tables / f"d15b_{tag}_gain_vs_k{min(ctx.ks)}.csv")
    if 2 in ctx.ks and min(ctx.ks) != 2:
        # primary paired gain: k = 1 tuning falls back to stratified cell CV, k >= 2 uses inner LOPO
        gain2 = pcc.paired_gain(by_patient, k_from=2, n_boot=ctx.args.n_boot, seed=SEED)
        atomic_write_csv(gain2, ctx.tables / f"d15b_{tag}_gain_vs_k2.csv")
    inclusion = pcc.patient_inclusion_effect(units, ks=[k for k in ctx.ks if k < len(ctx.supported) - 1])
    if not inclusion.empty:
        inclusion["design"] = tag
    atomic_write_csv(inclusion, ctx.tables / f"d15b_{tag}_patient_inclusion_effect.csv")
    power = pcc.power_law_table(curve) if set(range(1, 5)) <= set(curve["k"]) else pd.DataFrame()
    atomic_write_csv(power, ctx.tables / f"d15b_{tag}_power_law_fit.csv")
    ref = reference_checks(ctx, units)
    atomic_write_csv(ref, ctx.tables / f"d15b_{tag}_reference_checks.csv")
    timing = timing_model(ctx, tune, units)
    atomic_write_csv(timing, ctx.tables / f"d15b_{tag}_timing_model.csv")
    make_figures(ctx, units, by_patient, curve, power, inclusion)
    write_findings(ctx)
    if len(ref):
        print(f"[d15b] reference checks: {int(ref['pass'].sum())}/{len(ref)} pass, max abs diff {ref['abs_diff'].max():.2e}", flush=True)
    est = {}
    if not timing.empty:
        cpu = float(timing["cpu_seconds_full_design"].sum()) if set(ctx.reps) == set(REPRESENTATIONS) else float("nan")
        est = {"full_design_cpu_seconds_est": cpu, "full_design_wall_seconds_est_at_12": cpu / MAX_WORKERS if np.isfinite(cpu) else None}
    status.update(est)
    return status


def write_manifest_and_ledger(ctx: Ctx, started: float, extra: Dict[str, Any]) -> None:
    import scipy
    import sklearn

    head, dirty = git_state(REPO)
    inputs = [ATLAS_DIAG_TABLE, ATLAS_PATIENT_TABLE, D15_TABLE, D19_TABLE]
    inputs += [score_path(ctx.exp, *REPRESENTATIONS[r]).relative_to(ctx.exp).as_posix() for r in ctx.reps]
    input_rows = []
    for rel in inputs:
        p = ctx.exp / rel
        input_rows.append({"path": rel, "bytes": p.stat().st_size if p.exists() else None, "mtime_ns": p.stat().st_mtime_ns if p.exists() else None})
    mp = Path(ctx.meta_path)
    input_rows.append({"path": ctx.rel(mp), "bytes": mp.stat().st_size, "mtime_ns": mp.stat().st_mtime_ns})
    manifest = {
        "slug": SLUG, "analysis_id": ANALYSIS_ID, "tasks": [TASK_ID], "devlog_id": DEVLOG_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(a) for a in [Path(sys.argv[0]).name] + sys.argv[1:]),
        "git_head": head, "git_dirty": dirty, "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        "n_jobs": ctx.n_jobs, "smoke": bool(ctx.args.smoke), "design": ctx.design_tag,
        "parts_runtime_seconds": ctx.timings, "total_runtime_seconds": time.perf_counter() - started,
        "supported_patients": ctx.supported, "unsupported_patients_always_in_training": ctx.unsupported,
        "stage1_basis": pcc.BASIS_LABEL + "; stored stage1_dr/panel_id=*/method=*/k=40/seed=42/scores.npy, no DR refit",
        "parameters": {
            "representations": {r: {**representation_ids(r), **ctx.atlas[r]} for r in ctx.reps},
            "heldout_patients": ctx.heldouts, "ks": ctx.ks, "design": ctx.design,
            "n_design_rows_per_representation": int(len(ctx.design_frame)),
            "tune_penalties": list(TUNE_PENALTIES), "tuned_arm_penalty_rule": "max inner median lift; ties to l1",
            "arms": pcc.ARM_LABELS, "n_boot": ctx.args.n_boot, "bootstrap": "patient-cluster (held-out patients), percentile 95%",
            "power_law": "y = a - b k^-c, weights 1/max(se, 0.0025)^2, bounds a[-1,1] b[0,2] c[0.05,5]; validate k<=4 -> 5 and k<=5 -> 6",
            "seed": SEED, "reference_tolerance_auprc": REFERENCE_TOL, "scores_dtype": "float32 (as D19)",
        },
        "c_policy": rc.c_policy_record(
            c_policy="per_fold_tuned", training_n_changed=True, penalties=list(TUNE_PENALTIES),
            notes="tuned once per training set (S + unsupported patients); k = 1 uses the stratified 5-fold CV fallback; "
                  "reference arms: fixed atlas C (global_selected) and c_scaled_to_n",
        ),
        "inputs": input_rows,
        **extra,
    }
    previous = read_json_any(ctx.tables / f"{SLUG}_manifest.json")
    history = list(previous.get("run_history", [])) if previous else []
    if previous:
        history.append({k: previous.get(k) for k in ("created_utc", "command", "design", "git_head", "git_dirty",
                                                       "parts_runtime_seconds", "total_runtime_seconds")})
    manifest["run_history"] = history
    atomic_write_json(manifest, ctx.tables / f"{SLUG}_manifest.json")
    ledger = []
    for root, kind in ((ctx.tables, "table"), (ctx.figures, "figure")):
        for p in sorted(root.glob("*")):
            if p.is_file() and not p.name.endswith(".tmp") and p.name != f"{SLUG}_artifact_ledger.csv":
                rows = None
                if p.suffix == ".csv":
                    try:
                        with open(p, encoding="utf-8") as fh:
                            rows = sum(1 for _ in fh) - 1
                    except Exception:
                        rows = None
                this_kind = kind if p.suffix in (".csv", ".png", ".pdf") else "doc"
                ledger.append({"path": ctx.rel(p), "kind": this_kind, "rows": rows, "bytes": p.stat().st_size, "sha256": file_sha256(p)})
    cache_files = [p for p in ctx.cache.rglob("*") if p.is_file()]
    ledger.append({"path": ctx.rel(ctx.cache), "kind": "cache_dir", "rows": len(cache_files),
                   "bytes": int(sum(p.stat().st_size for p in cache_files)), "sha256": ""})
    frame = pd.DataFrame(ledger)
    frame["rows"] = frame["rows"].astype("Int64")
    atomic_write_csv(frame, ctx.tables / f"{SLUG}_artifact_ledger.csv")


def representation_ids(rep: str) -> Dict[str, Any]:
    panel, method, k = REPRESENTATIONS[rep]
    return {"stage0_panel_id": panel, "stage1_method": method, "effective_k": k}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", default=",".join(DEFAULT_PARTS), help="comma list from tune,units,report")
    p.add_argument("--design", choices=("full", "reduced"), default="full", help="full enumeration or the cyclic reduced design")
    p.add_argument("--representations", default=None, help=f"comma list from {','.join(REPRESENTATIONS)} (default: all)")
    p.add_argument("--heldout", default=None, help="comma list of held-out supported patients (default: all)")
    p.add_argument("--k", default=None, help="comma list of training-patient counts (default 1,2,3,4,5,6)")
    p.add_argument("--resume", action="store_true", help="skip finished tune and outer units")
    p.add_argument("--n-jobs", type=int, default=MAX_WORKERS, help=f"worker processes (capped at {MAX_WORKERS})")
    p.add_argument("--n-boot", type=int, default=N_BOOT, help="patient-cluster bootstrap draws")
    p.add_argument("--out-root", default=None, help="write under this dir instead of EXP/analysis (smoke runs)")
    p.add_argument("--smoke", action="store_true", help="held-out P07, k in {1,2}, allograft_promax40; needs --out-root")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    parts = [x.strip() for x in args.parts.split(",") if x.strip()]
    unknown = sorted(set(parts) - set(DEFAULT_PARTS))
    if unknown:
        raise SystemExit(f"unknown parts: {unknown}")
    started = time.perf_counter()
    ctx = Ctx(args)
    print(f"[d15b] EXP={ctx.exp.name} out={ctx.out_root} design={ctx.design_tag} reps={ctx.reps} heldout={ctx.heldouts} "
          f"ks={ctx.ks} rows/rep={len(ctx.design_frame)} distinct subsets={ctx.design_frame['subset'].nunique()} n_jobs={ctx.n_jobs}",
          flush=True)
    extra: Dict[str, Any] = {}
    for name, fn in (("tune", run_tune), ("units", run_units), ("report", run_report)):
        if name in parts:
            t0 = time.perf_counter()
            extra[name] = fn(ctx)
            ctx.timings[name] = round(time.perf_counter() - t0, 2)
    write_manifest_and_ledger(ctx, started, extra)  # every run, so run_history keeps tune-only and units-only runs too
    print(f"[d15b] done in {time.perf_counter() - started:.1f} s: {ctx.timings} {extra.get('report', {})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
