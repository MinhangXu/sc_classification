"""D26 leave-high-prevalence-out leverage test on allograft rejection and apoptosis (task W2.4;
slug leverage_allograft_apoptosis_v1).

Question (H-S1-05): do the highest-prevalence patients (P02, P03, P04) move the discriminative
Stage 1 axes of allograft rejection and apoptosis more than random drops of the same number of
cells AND the same number of malignant cells, and more than the low-prevalence contrast patients?
This is the D17 leverage design (IFN-alpha; run_d17_basis_variant_pilot.py) on two more panels.
Stage 1 is refit without the listed cells, every cell is projected, and the refit basis is
aligned to the stored transductive basis. No Stage 2 probe is refit (alignment only).

Scope: PCA and FA at K 10, 20, 40, seed 42, on the covered-gene z-scored features
(stage1_direct_gene/old34_direct_gene_all_v1/panel_id=<panel>/transform=zscore). FactoSig and
FactoSig-promax (the atlas winner for both panels) are out of scope: their refit needs the
factosig package and its GPU path, which this CPU runner does not use.

Arms per panel x method x K (all alignment only):
  transductive_refit       refit on all cells (sanity floor: must reproduce the stored basis)
  leave_out                drop P02 / P03 / P04 / P02+P03+P04 and, as contrasts, P01 / P09 / P06 /
                           P01+P09 (D17 drop sets); refit; project all cells
  leave_out_random         3 size-matched random cell drops per leave-out set (seeds 1-3, as D17)
  leave_out_label_random   20 composition-matched random drops per set: same number of dropped
                           cells AND dropped malignant cells (seeds 101-120, as D17)
  normal_only_all          normal cells only (all malignant cells dropped)
  normal_only_random       3 random drops of as many cells as there are malignant cells

Reference axes (computed in part 'refs' from the stored transductive basis; all three are
aligned on every basis):
  top_gap      PRIMARY. The factor with the largest |all-cell standardized malignant-minus-normal
               mean gap| on the stored scores: the D17 rule (PC8 / FA7 for IFN-alpha). Under
               balanced L1 this is the factor with the largest entry margin, i.e. the first
               factor any sparse Stage 2 probe selects (D19 argmax_feature per fold is recorded
               as a cross-check). Pooled over cells, so it is the axis most exposed to the
               highest-prevalence patients' malignant cells.
  top_heldout  The factor with the largest median univariate held-out lift over the seven
               supported LOPO folds (sign fixed on the training folds): the D18 factor-card
               statistic (cross-checked against factor_cards_index.csv at K = 40).
  probe        The atlas-selected Stage 2 probe (penalty, C, l1_ratio from D19
               d19_atlas_selected_regularization.csv) refit on all cells of the stored scores
               with regularization_contract; the axis is its standardized linear score, a
               combination of the K reference factors. Cross-checked by cosine against the
               stored discovery_full_cohort coefficients for the same configuration.
Alignment (per basis and axis; s1.align_scores as in D17): axis loss = 1 - many-to-one R^2 of
the reference axis regressed on all variant factors over the retained (fit) cells; Hungarian
matched |r| (single-factor axes only); principal angles between top-8 and all-K loading
subspaces. Leverage beyond composition: one-sided empirical rank p of the observed axis loss
among the 20 composition-matched controls, (1 + #controls >= observed) / 21.

Parts (--parts, default refs,bases,report): refs (reference axes and cross-checks), bases (fit +
align, one unit per basis), report (tables, figures, findings, manifest, ledger). --resume
skips finished units (basis saved and alignment JSON done with the current reference-axis
signature); a saved basis with a missing or stale alignment is only re-aligned; empty or corrupt
files are redone. A report that finds missing units writes merge_failed.json and exits 3;
recover with --resume (or --merge-only once the units exist). --out-root redirects every
output (required with --smoke).

Outputs (EXP-relative, new slug, R3):
  analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/  tables, findings, manifest, ledger
  analysis/figures/stage2_diagnostic/leverage_allograft_apoptosis_v1/     D26A-D26D (.png, .pdf)
  analysis/cache/stage2_diagnostic/leverage_allograft_apoptosis_v1/       bases/, align_units/, d26_refs.json,
                                                                          stored discovery coefficient extract
Example (repo root): python scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py --resume --n-jobs 12
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
from sklearn.metrics import average_precision_score  # noqa: E402

import regularization_contract as rc  # noqa: E402
import run_d17_basis_variant_pilot as d17  # noqa: E402
import stage1_refit as s1  # noqa: E402

SLUG = "leverage_allograft_apoptosis_v1"
ANALYSIS_ID = "D26"
TASK_ID = "W2.4"
DEFAULT_PANELS = ("single_geneset__hallmark_allograft_rejection", "single_geneset__hallmark_apoptosis")
FEATURES_ROOT = "stage1_direct_gene/old34_direct_gene_all_v1"
METHODS = ("pca", "fa")
DEFAULT_KS = (10, 20, 40)
SEED = d17.SEED
TOP_M = d17.TOP_M
MAX_WORKERS = d17.MAX_WORKERS
HIGH_PREVALENCE = d17.HIGH_PREVALENCE
DROP_SETS = d17.DROP_SETS
RANDOM_DROP_SEEDS = d17.RANDOM_DROP_SEEDS
LABEL_MATCHED_DROP_SEEDS = d17.LABEL_MATCHED_DROP_SEEDS
AXES = ("top_gap", "top_heldout", "probe")
PRIMARY_AXIS = "top_gap"
AXIS_LABELS = {
    "top_gap": "largest all-cell class-mean gap factor",
    "top_heldout": "largest median univariate held-out lift factor",
    "probe": "atlas-selected probe direction",
}
SINGLE_FACTOR_AXES = ("top_gap", "top_heldout")
DEFAULT_PARTS = ("refs", "bases", "report")
UNIT_SCHEMA = 1
#: Refit sanity gate on the transductive_refit arm (same floor as tests/test_stage1_refit.py).
REFIT_GATE_R2 = 0.995
REFIT_GATE_ABS_R = 0.998
#: Leave-out sets that test the hypothesis and the decision rule fixed before the run.
TEST_SET = "P02-P03-P04"
CONTRAST_SETS = ("P01", "P06", "P09", "P01-P09")
DECISION_MIN_CELLS = 4  # of the 6 method x K cells per panel
D19_SLUG = "regularization_contract_v1"
D18_SLUG = "factor_cards_v1"
DISCOVERY_COEF_REL = "stage2_supervised/multiobjective/discovery_full_cohort/coefficient_paths.csv"
FALLBACK_PROBE = {"penalty": "l2", "C": 1e-4, "l1_ratio": None, "source": "fallback_l2_C1e-4"}

atomic_write_csv = d17.atomic_write_csv
atomic_write_json = d17.atomic_write_json
atomic_write_text = d17.atomic_write_text
read_json_done = d17.read_json_done
fid = d17.fid
factor_label = d17.factor_label
_md_table = d17._md_table
file_sha256 = d17.file_sha256


# --------------------------------------------------------------------------------------
# names and paths
# --------------------------------------------------------------------------------------


def normalise_panel(name: str) -> str:
    """Full panel id from a full id or a hallmark suffix ('apoptosis' -> single_geneset__hallmark_apoptosis)."""
    name = str(name).strip()
    return name if "__" in name else f"single_geneset__hallmark_{name}"


def panel_tag(panel: str) -> str:
    return panel.split("single_geneset__", 1)[-1]


def panel_label(panel: str) -> str:
    tail = panel.split("__hallmark_", 1)[-1] if "__hallmark_" in panel else panel.split("__")[-1]
    return tail.replace("_", " ").replace("interferon alpha", "IFN-alpha")


def features_rel(panel: str) -> str:
    return f"{FEATURES_ROOT}/panel_id={panel}/transform=zscore"


def stored_dir(exp: Path, panel: str, method: str, k: int) -> Path:
    return exp / "stage1_dr" / f"panel_id={panel}" / f"method={method}" / f"k={int(k)}" / f"seed={SEED}"


def cell_key(panel: str, method: str, k: int) -> str:
    return f"{panel}|{method}|{int(k)}"


# --------------------------------------------------------------------------------------
# data (cached per process, so joblib workers load the inputs once)
# --------------------------------------------------------------------------------------

_DATA: Dict[Any, Any] = {}


def load_meta() -> Tuple[np.ndarray, np.ndarray]:
    if "meta" not in _DATA:
        meta = pd.read_csv(cell_metadata(), usecols=["cell_id", "y_true", "patient"])
        _DATA["meta"] = (meta["y_true"].to_numpy().astype(int), meta["patient"].astype(str).to_numpy())
    return _DATA["meta"]


def load_panel(exp_str: str, panel: str) -> Dict[str, Any]:
    key = ("panel", exp_str, panel)
    if key not in _DATA:
        import scipy.sparse as sp

        fdir = Path(exp_str) / features_rel(panel)
        X = sp.load_npz(fdir / "features.npz").toarray().astype(np.float64)
        genes = [str(g) for g in json.loads((fdir / "feature_names.json").read_text(encoding="utf-8"))]
        y, pat = load_meta()
        if X.shape != (y.size, len(genes)):
            raise AssertionError(f"{panel}: features {X.shape} vs metadata {y.size} cells x {len(genes)} genes")
        _DATA[key] = {"X": X, "genes": genes}
    return _DATA[key]


def stored_scores(exp_str: str, panel: str, method: str, k: int) -> np.ndarray:
    key = ("stored", exp_str, panel, method, int(k))
    if key not in _DATA:
        y, _ = load_meta()
        scores = np.load(stored_dir(Path(exp_str), panel, method, k) / "scores.npy").astype(np.float64)
        if scores.shape != (y.size, int(k)):
            raise AssertionError(f"stored {panel} {method} K={k} scores have shape {scores.shape}")
        _DATA[key] = scores
    return _DATA[key]


def ref_arrays(exp_str: str, panel: str, method: str, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alignment reference of the stored transductive basis, as in D17 (_ref_arrays):
    exact projection scores, common-scale loadings (genes x K) and the factor importance order.

    FA: the stored scores (sklearn FA.fit_transform = transform). PCA: the projection of the
    centred input onto the stored loadings (stored PCA scores are randomized-SVD U * S).
    """
    key = ("ref", exp_str, panel, method, int(k))
    if key not in _DATA:
        L = np.load(stored_dir(Path(exp_str), panel, method, k) / "loadings.npy").astype(np.float64)
        if method == "pca":
            X = load_panel(exp_str, panel)["X"]
            S = (X - X.mean(axis=0)) @ L
            order = np.arange(int(k))
        else:
            S = stored_scores(exp_str, panel, method, k)
            order = np.argsort(-np.sum(L ** 2, axis=0), kind="stable")
        _DATA[key] = (S, L, order)
    return _DATA[key]


def probe_target(exp_str: str, panel: str, method: str, k: int, w: Sequence[float]) -> np.ndarray:
    """The probe axis on the alignment reference: all-cell standardized reference scores @ w."""
    w_arr = np.asarray(w, dtype=np.float64)
    key = ("probe", exp_str, panel, method, int(k), hashlib.sha1(np.round(w_arr, 12).tobytes()).hexdigest())
    if key not in _DATA:
        S, _, _ = ref_arrays(exp_str, panel, method, k)
        _DATA[key] = rc.standardize_train(S)[0] @ w_arr
    return _DATA[key]


# --------------------------------------------------------------------------------------
# reference-axis statistics (pure functions; unit-tested on synthetic data)
# --------------------------------------------------------------------------------------


def all_cell_gap(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Standardized (all cells, population SD) malignant-minus-normal mean gap per factor (D17 rule)."""
    Zs, _ = rc.standardize_train(scores)
    y = np.asarray(y).astype(int)
    return Zs[y == 1].mean(axis=0) - Zs[y == 0].mean(axis=0)


def top_gap_factor(scores: np.ndarray, y: np.ndarray) -> int:
    return int(np.argmax(np.abs(all_cell_gap(scores, y))))


def univariate_heldout_lifts(scores: np.ndarray, y: np.ndarray, patients: np.ndarray,
                             heldouts: Sequence[str]) -> np.ndarray:
    """(n_heldouts, K) univariate held-out lift: AP of one factor on the held-out patient minus its
    prevalence, with each factor's sign fixed on the training folds (sign of the training
    malignant-minus-normal mean difference; >= 0 -> +1), as in D18 factor_cards.heldout_univariate."""
    S = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(int)
    pat = np.asarray(patients).astype(str)
    out = np.full((len(heldouts), S.shape[1]), np.nan)
    for i, h in enumerate(heldouts):
        held = pat == h
        train = ~held
        yh = y[held]
        if yh.size == 0 or yh.min() == yh.max():
            continue
        diff = S[train & (y == 1)].mean(axis=0) - S[train & (y == 0)].mean(axis=0)
        sign = np.where(diff >= 0, 1.0, -1.0)
        prevalence = float(yh.mean())
        for j in range(S.shape[1]):
            out[i, j] = float(average_precision_score(yh, sign[j] * S[held, j])) - prevalence
    return out


def top_heldout_factor(lifts: np.ndarray) -> int:
    med = np.nanmedian(lifts, axis=0)
    return int(np.nanargmax(med))


def probe_weights(scores: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Standardized coefficients of the probe refit on all cells with the contract."""
    est = rc.fit_probe(scores, y, penalty=config["penalty"], C=float(config["C"]),
                       l1_ratio=None if config.get("l1_ratio") is None else float(config["l1_ratio"]))
    return rc.probe_coef(est)


def rank_p(observed: float, controls: Sequence[float], larger_moves: bool = True) -> float:
    """One-sided empirical rank p as in D17: (1 + #controls moving the axis at least as much) / (1 + n)."""
    cv = np.asarray(controls, dtype=float)
    cv = cv[np.isfinite(cv)]
    if not np.isfinite(observed) or cv.size == 0:
        return float("nan")
    n_ge = int(np.sum(cv >= observed)) if larger_moves else int(np.sum(cv <= observed))
    return (1.0 + n_ge) / (1.0 + cv.size)


# --------------------------------------------------------------------------------------
# context and plan
# --------------------------------------------------------------------------------------


class Ctx:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.smoke and not args.out_root:
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
        self.units = self.cache / "align_units"
        for path in (self.tables, self.figures, self.cache, self.bases, self.units):
            path.mkdir(parents=True, exist_ok=True)
        self.y, self.patients = load_meta()
        self.supported = rc.supported_patients(self.y, self.patients, min_malignant=rc.DEFAULT_MIN_MALIGNANT)
        try:
            configured = list(study_config().get("supported_patients", []))
        except Exception:
            configured = []
        if configured and configured != self.supported:
            raise AssertionError(f"support rule gives {self.supported}, study_config lists {configured}")
        self.n_jobs = max(1, min(int(args.n_jobs), MAX_WORKERS))
        self.panels = [normalise_panel(p) for p in args.panels.split(",") if p.strip()]
        self.methods = [m for m in args.methods.split(",") if m]
        self.ks = [int(k) for k in args.ks.split(",") if k]
        smoke = bool(args.smoke)
        # smoke: the test set and one contrast, so the decision rule is exercised
        self.drop_sets = (("P02", "P03", "P04"), ("P09",)) if smoke else DROP_SETS
        self.rand_seeds = RANDOM_DROP_SEEDS[:1] if smoke else RANDOM_DROP_SEEDS
        self.label_seeds = LABEL_MATCHED_DROP_SEEDS[:2] if smoke else LABEL_MATCHED_DROP_SEEDS
        self.timings: Dict[str, float] = {}
        for panel in self.panels:
            for m in self.methods:
                for k in self.ks:
                    d = stored_dir(self.exp, panel, m, k)
                    if not (d / "scores.npy").is_file() or not (d / "loadings.npy").is_file():
                        raise SystemExit(f"stored basis missing: {self.rel(d)}")
            fdir = self.exp / features_rel(panel)
            if not (fdir / "features.npz").is_file():
                raise SystemExit(f"covered-gene features missing: {self.rel(fdir)}")

    def rel(self, path: Path) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(self.exp.resolve()).as_posix()
        except ValueError:
            try:
                return "<out-root>/" + path.relative_to(self.out_root).as_posix()
            except ValueError:
                return path.name


def build_plan(ctx: Ctx) -> List[Dict[str, Any]]:
    """One spec per Stage 1 basis (all alignment only)."""
    specs: List[Dict[str, Any]] = []
    n_mal = int(ctx.y.sum())
    counts = pd.Series(ctx.patients).value_counts()
    for panel in ctx.panels:
        for method in ctx.methods:
            for k in ctx.ks:
                def add(arm: str, variant: str, fit: Dict[str, Any]) -> None:
                    specs.append({
                        "basis_id": f"{arm}__{panel_tag(panel)}__{method}{k}__{variant}",
                        "arm": arm, "panel": panel, "method": method, "k": int(k), "variant": variant, "fit": fit,
                    })

                add("transductive_refit", "all", {"kind": "rows"})
                add("normal_only_all", "all", {"kind": "rows", "label": 0})
                for seed in ctx.rand_seeds:
                    add("normal_only_random", f"drop{n_mal}-s{seed}", {"kind": "random_drop", "n_drop": n_mal, "seed": seed})
                for ds in ctx.drop_sets:
                    tag = "-".join(ds)
                    add("leave_out", f"drop-{tag}", {"kind": "rows", "exclude": list(ds)})
                    n_drop = int(sum(int(counts.get(p, 0)) for p in ds))
                    n_drop_mal = int(sum(int(ctx.y[ctx.patients == p].sum()) for p in ds))
                    for seed in ctx.rand_seeds:
                        add("leave_out_random", f"drop-{tag}-random-s{seed}",
                            {"kind": "random_drop", "n_drop": n_drop, "seed": seed, "matches": tag})
                    for seed in ctx.label_seeds:
                        add("leave_out_label_random", f"drop-{tag}-labelmatched-s{seed}",
                            {"kind": "label_matched_drop", "n_drop": n_drop, "n_drop_malignant": n_drop_mal, "seed": seed,
                             "matches": tag})
    return specs


def basis_dir(ctx: Ctx, spec: Dict[str, Any]) -> Path:
    return (ctx.bases / f"arm={spec['arm']}" / f"panel_id={spec['panel']}" / f"method={spec['method']}" / f"k={spec['k']}"
            / f"seed={SEED}" / f"variant={spec['variant']}")


def unit_path(ctx: Ctx, spec: Dict[str, Any]) -> Path:
    return ctx.units / f"{spec['basis_id']}.json"


# --------------------------------------------------------------------------------------
# part: refs
# --------------------------------------------------------------------------------------


def _atlas_configs(ctx: Ctx) -> Dict[str, Dict[str, Any]]:
    path = ctx.exp / "analysis" / "scorecards" / "stage2_diagnostic" / D19_SLUG / "d19_atlas_selected_regularization.csv"
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    table = pd.read_csv(path)
    for r in table.itertuples(index=False):
        if str(r.stage1_method) not in METHODS:
            continue
        l1r = None if pd.isna(r.l1_ratio) else float(r.l1_ratio)
        out[cell_key(str(r.stage0_panel_id), str(r.stage1_method), int(round(float(r.effective_k))))] = {
            "penalty": str(r.penalty), "C": float(r.C), "l1_ratio": l1r if str(r.penalty) == "elasticnet" else None,
            "source": f"D19 d19_atlas_selected_regularization.csv ({r.source_scorecard})",
            "atlas_patient_median_lift_pp": 100.0 * float(r.patient_median_lift),
        }
    return out


def _d19_argmax(ctx: Ctx) -> pd.DataFrame:
    path = ctx.exp / "analysis" / "scorecards" / "stage2_diagnostic" / D19_SLUG / "d19_entry_margin_audit_by_fold.csv"
    if not path.is_file():
        return pd.DataFrame()
    cols = ["stage0_panel_id", "stage1_method", "effective_k", "heldout_patient", "argmax_feature", "C", "penalty"]
    table = pd.read_csv(path, usecols=cols)
    return table.loc[table["stage0_panel_id"].isin(ctx.panels) & table["stage1_method"].isin(METHODS)]


def _d18_cards(ctx: Ctx) -> pd.DataFrame:
    path = ctx.exp / "analysis" / "scorecards" / "stage2_diagnostic" / D18_SLUG / "factor_cards_index.csv"
    if not path.is_file():
        return pd.DataFrame()
    table = pd.read_csv(path, usecols=["panel_id", "method", "k", "factor", "median_univariate_lift"])
    return table.loc[table["panel_id"].isin(ctx.panels) & table["method"].isin(METHODS)]


def stored_discovery_coefficients(ctx: Ctx) -> pd.DataFrame:
    """Extract of the stored discovery_full_cohort coefficients for the panels (cached)."""
    cache = ctx.cache / "stored_discovery_coefficients_extract.csv"
    if cache.is_file() and cache.stat().st_size > 0:
        try:
            frame = pd.read_csv(cache)
            if set(ctx.panels) <= set(frame["stage0_panel_id"].unique()):
                return frame
        except Exception:
            pass
    src = ctx.exp / DISCOVERY_COEF_REL
    if not src.is_file():
        return pd.DataFrame()
    cols = ["stage0_panel_id", "representation_family", "stage1_method", "effective_k", "seed", "modeling_goal", "penalty",
            "C", "l1_ratio", "feature_id", "coefficient"]
    parts = []
    for chunk in pd.read_csv(src, usecols=cols, chunksize=500_000, low_memory=False):
        keep = (chunk["stage0_panel_id"].isin(ctx.panels) & chunk["representation_family"].eq("dr")
                & chunk["stage1_method"].isin(METHODS))
        parts.append(chunk.loc[keep])
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    atomic_write_csv(frame, cache)
    return frame


def _stored_probe_vector(disc: pd.DataFrame, panel: str, method: str, k: int, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    if disc is None or disc.empty:
        return None
    sel = (disc["stage0_panel_id"].eq(panel) & disc["stage1_method"].eq(method)
           & disc["effective_k"].round().astype(int).eq(int(k)) & disc["penalty"].eq(cfg["penalty"])
           & np.isclose(disc["C"].astype(float), float(cfg["C"]), rtol=1e-6))
    if cfg.get("l1_ratio") is not None:
        sel &= np.isclose(disc["l1_ratio"].astype(float), float(cfg["l1_ratio"]), rtol=1e-6)
    hit = disc.loc[sel]
    if hit.empty:
        return None
    vec = np.zeros(int(k))
    for r in hit.itertuples(index=False):
        idx = int(str(r.feature_id).rsplit("_", 1)[-1]) - 1
        if 0 <= idx < int(k):
            vec[idx] = float(r.coefficient)
    return vec


def run_refs(ctx: Ctx) -> Dict[str, Dict[str, Any]]:
    """Reference axes per panel x method x K, plus cross-checks; writes d26_refs.json and two tables."""
    t0 = time.perf_counter()
    atlas = _atlas_configs(ctx)
    d19 = _d19_argmax(ctx)
    d18 = _d18_cards(ctx)
    disc = stored_discovery_coefficients(ctx)
    refs: Dict[str, Dict[str, Any]] = {}
    ax_rows, sig_rows = [], []
    y, pat = ctx.y, ctx.patients
    for panel in ctx.panels:
        for m in ctx.methods:
            for k in ctx.ks:
                stored = stored_scores(ctx.exp_str, panel, m, k)
                S, _, _ = ref_arrays(ctx.exp_str, panel, m, k)
                gap = all_cell_gap(stored, y)
                j_gap = int(np.argmax(np.abs(gap)))
                lifts = univariate_heldout_lifts(stored, y, pat, ctx.supported)
                med = np.nanmedian(lifts, axis=0)
                j_held = top_heldout_factor(lifts)
                cfg = dict(atlas.get(cell_key(panel, m, k)) or FALLBACK_PROBE)
                w = probe_weights(stored, y, cfg)
                n_nz = int(np.sum(np.abs(w) > rc.COEF_NONZERO_TOL))
                s_ref = probe_target(ctx.exp_str, panel, m, k, w)
                s_stored = rc.standardize_train(stored)[0] @ w
                r_probe = float(np.corrcoef(s_ref, s_stored)[0, 1]) if n_nz else float("nan")
                stored_vec = _stored_probe_vector(disc, panel, m, k, cfg)
                cos = float("nan")
                if stored_vec is not None and np.linalg.norm(stored_vec) > 0 and np.linalg.norm(w) > 0:
                    cos = float(stored_vec @ w / (np.linalg.norm(stored_vec) * np.linalg.norm(w)))
                # D19 per-fold argmax of the atlas-selected configuration (largest L1 entry margin)
                d19_modal, d19_eq, d19_n = "", None, 0
                if not d19.empty:
                    g = d19.loc[d19["stage0_panel_id"].eq(panel) & d19["stage1_method"].eq(m)
                                & d19["effective_k"].round().astype(int).eq(int(k))]
                    if len(g):
                        d19_modal = str(g["argmax_feature"].mode().iloc[0])
                        d19_eq = int((g["argmax_feature"] == fid(j_gap)).sum())
                        d19_n = int(len(g))
                d18_med, d18_diff = float("nan"), float("nan")
                if not d18.empty:
                    g = d18.loc[d18["panel_id"].eq(panel) & d18["method"].eq(m) & d18["k"].eq(int(k))].set_index("factor")
                    if len(g) == int(k):
                        d18_vec = g["median_univariate_lift"].reindex(range(1, int(k) + 1)).to_numpy(dtype=float)
                        d18_med = float(d18_vec[j_held])
                        d18_diff = float(np.nanmax(np.abs(d18_vec - med)))
                rank_gap = np.argsort(np.argsort(-np.abs(gap), kind="stable"), kind="stable") + 1
                rank_held = np.argsort(np.argsort(-med, kind="stable"), kind="stable") + 1
                order_w = np.argsort(-np.abs(w), kind="stable")[:3]
                top_w = ";".join(f"{factor_label(m, int(i))}:{w[i]:+.3f}" for i in order_w if abs(w[i]) > rc.COEF_NONZERO_TOL)
                refs[cell_key(panel, m, k)] = {
                    "panel": panel, "method": m, "k": int(k), "top_gap": j_gap, "top_heldout": j_held,
                    "probe_w": [float(v) for v in w], "probe_config": cfg, "probe_n_nonzero": n_nz,
                }
                base = {"panel_id": panel, "panel_label": panel_label(panel), "n_genes": int(load_panel(ctx.exp_str, panel)["X"].shape[1]),
                        "stage1_method": m, "effective_k": int(k),
                        "alignment_reference": "stored scores" if m == "fa" else "projection onto stored loadings"}
                for axis, j in (("top_gap", j_gap), ("top_heldout", j_held)):
                    ax_rows.append({**base, "ref_axis": axis, "ref_factor": fid(j), "ref_factor_label": factor_label(m, j),
                                    "all_cell_std_gap": float(gap[j]), "median_univariate_lift_pp": 100.0 * float(med[j]),
                                    "n_patients_univariate_lift_ge_5pp": int(np.sum(lifts[:, j] >= 0.05)),
                                    "rank_by_abs_gap": int(rank_gap[j]), "rank_by_median_heldout_lift": int(rank_held[j]),
                                    "same_factor_as_top_gap": bool(j == j_gap),
                                    "d19_modal_argmax_factor": d19_modal, "d19_n_folds_argmax_is_top_gap": d19_eq,
                                    "d19_n_folds": d19_n, "d18_median_univariate_lift_pp": 100.0 * d18_med,
                                    "d18_max_abs_diff_all_factors_pp": 100.0 * d18_diff})
                ax_rows.append({**base, "ref_axis": "probe", "ref_factor": "probe",
                                "ref_factor_label": f"{cfg['penalty']} C={float(cfg['C']):.3g}"
                                + (f" l1_ratio={cfg['l1_ratio']:g}" if cfg.get("l1_ratio") is not None else ""),
                                "probe_penalty": cfg["penalty"], "probe_C": float(cfg["C"]), "probe_l1_ratio": cfg.get("l1_ratio"),
                                "probe_config_source": cfg.get("source"), "probe_n_nonzero": n_nz, "probe_top_coefficients": top_w,
                                "probe_cos_vs_stored_discovery": cos, "probe_r_ref_vs_stored_scores": r_probe,
                                "atlas_patient_median_lift_pp": cfg.get("atlas_patient_median_lift_pp")})
                for j in range(int(k)):
                    row = {"panel_id": panel, "stage1_method": m, "effective_k": int(k), "factor": fid(j),
                           "factor_label": factor_label(m, j), "all_cell_std_gap": float(gap[j]),
                           "median_univariate_lift_pp": 100.0 * float(med[j]),
                           "max_univariate_lift_pp": 100.0 * float(np.nanmax(lifts[:, j])),
                           "probe_coef": float(w[j]), "is_top_gap": j == j_gap, "is_top_heldout": j == j_held}
                    for i, h in enumerate(ctx.supported):
                        row[f"univariate_lift_pp_{h}"] = 100.0 * float(lifts[i, j])
                    sig_rows.append(row)
                print(f"[refs] {panel_tag(panel)} {m.upper()}{k}: top_gap {factor_label(m, j_gap)} (gap {gap[j_gap]:+.3f}), "
                      f"top_heldout {factor_label(m, j_held)} (median {100 * med[j_held]:.2f} pp), probe {cfg['penalty']} "
                      f"C={float(cfg['C']):.3g} ({n_nz} nonzero; cos vs stored {cos:.4f})", flush=True)
    atomic_write_csv(pd.DataFrame(ax_rows), ctx.tables / "d26_reference_axes.csv")
    atomic_write_csv(pd.DataFrame(sig_rows), ctx.tables / "d26_factor_signal.csv")
    atomic_write_json(refs, ctx.cache / "d26_refs.json")
    ctx.timings["refs_s"] = time.perf_counter() - t0
    print(f"[refs] {len(refs)} reference cells in {ctx.timings['refs_s']:.1f} s", flush=True)
    return refs


def load_refs(ctx: Ctx) -> Dict[str, Dict[str, Any]]:
    path = ctx.cache / "d26_refs.json"
    if not path.is_file():
        raise SystemExit("[refs] d26_refs.json is missing; run the refs part first")
    refs = json.loads(path.read_text(encoding="utf-8"))
    missing = [cell_key(p, m, k) for p in ctx.panels for m in ctx.methods for k in ctx.ks if cell_key(p, m, k) not in refs]
    if missing:
        raise SystemExit(f"[refs] d26_refs.json lacks {missing[:3]}; rerun the refs part")
    return refs


def ref_signature(ref: Dict[str, Any]) -> Dict[str, Any]:
    w = np.round(np.asarray(ref["probe_w"], dtype=np.float64), 10)
    return {"schema": UNIT_SCHEMA, "top_gap": int(ref["top_gap"]), "top_heldout": int(ref["top_heldout"]),
            "probe_sha1": hashlib.sha1(w.tobytes()).hexdigest(), "stage1_refit_version": s1.MODULE_VERSION, "top_m": TOP_M}


# --------------------------------------------------------------------------------------
# part: bases (fit + align, one unit per basis)
# --------------------------------------------------------------------------------------


def align_rows(S: np.ndarray, L: np.ndarray, order: np.ndarray, Z: np.ndarray, basis: "s1.Stage1Basis",
               fit_idx: np.ndarray, n_cells: int, n_mal_total: int, y: np.ndarray, spec: Dict[str, Any],
               ref: Dict[str, Any], s_probe: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    """One alignment row per reference axis (D17 _align_one columns plus panel_id and ref_axis)."""
    m, k = spec["method"], spec["k"]
    kw = dict(ref_loadings=L, var_loadings=basis.loadings_common(), ref_order=order,
              var_order=basis.factor_variance_order(), top_m=TOP_M)
    base: Dict[str, Any] = {
        "basis_id": spec["basis_id"], "arm": spec["arm"], "panel_id": spec["panel"], "stage1_method": m, "effective_k": k,
        "variant": spec["variant"], "matches_drop_set": spec["fit"].get("matches", ""), "n_fit_cells": int(fit_idx.size),
        "n_dropped_cells": int(n_cells - fit_idx.size), "n_dropped_malignant": int(n_mal_total - y[fit_idx].sum()),
    }
    rows: List[Dict[str, Any]] = []
    angles: Dict[str, Any] = {}
    for axis in SINGLE_FACTOR_AXES:
        j = int(ref[axis])
        on_all = s1.align_scores(S, Z, j, **kw)
        on_fit = s1.align_scores(S, Z, j, rows=fit_idx)
        row = {**base, "ref_axis": axis, "ref_factor": fid(j), "ref_factor_label": factor_label(m, j)}
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
        row["axis_loss_all"] = 1.0 - float(on_all["many_to_one_r2"])
        rows.append(row)
        if not angles:
            angles = {key: on_all.get(key) for key in (f"angle_top{TOP_M}_max_deg", f"angle_top{TOP_M}_mean_cos2",
                                                       "angle_allK_max_deg", "angle_allK_mean_cos2")}
    if s_probe is not None:
        r2_all = s1.many_to_one_r2(s_probe, Z)
        r2_fit = s1.many_to_one_r2(s_probe, Z, rows=fit_idx)
        r_all = np.abs(np.nan_to_num(s1.score_correlation(s_probe[:, None], Z)[0]))
        r_fit = np.abs(np.nan_to_num(s1.score_correlation(s_probe[:, None], Z, rows=fit_idx)[0]))
        nan = float("nan")
        rows.append({
            **base, "ref_axis": "probe", "ref_factor": "probe", "ref_factor_label": "probe",
            "matched_factor_all": "", "best_factor_all": fid(int(np.argmax(r_all))), "matched_r_all": nan, "matched_abs_r_all": nan,
            "best_abs_r_all": float(r_all.max()), "many_to_one_r2_all": r2_all, "mean_matched_abs_r_all_all": nan,
            f"mean_matched_abs_r_top{TOP_M}_all": nan, "matched_factor_fit": "", "matched_abs_r_fit": nan,
            "best_abs_r_fit": float(r_fit.max()), "many_to_one_r2_fit": r2_fit, "mean_matched_abs_r_all_fit": nan,
            **angles, f"ref_axis_containment_top{TOP_M}": nan, "ref_axis_containment_allK": nan, "matched_loading_abs_cos": nan,
            "matched_factor_importance_rank": nan, "axis_loss_fit": 1.0 - float(r2_fit), "axis_loss_all": 1.0 - float(r2_all),
        })
    nz = int(np.sum(Z.std(0) < 1e-9))
    for row in rows:
        row["n_zero_variance_factors"] = nz
    return rows


def _unit(exp_str: str, spec: Dict[str, Any], bdir: str, upath: str, ref: Dict[str, Any], signature: Dict[str, Any],
          do_fit: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    y, pat = load_meta()
    data = load_panel(exp_str, spec["panel"])
    rows = d17.fit_rows(spec["fit"], y, pat)
    if do_fit:
        info = {"arm": spec["arm"], "variant": spec["variant"], "panel_id": spec["panel"], "fit_spec": spec["fit"],
                "fit_patients": sorted(set(pat[rows].tolist())), "n_fit_malignant": int(y[rows].sum()),
                "weighting": "none", "input": features_rel(spec["panel"]) + "/features.npz", "analysis": ANALYSIS_ID}
        basis = s1.fit_basis(data["X"], spec["method"], spec["k"], rows=rows, seed=SEED, gene_names=data["genes"], fit_info=info)
        basis.fit_info["fit_seconds"] = time.perf_counter() - t0
        basis.save(bdir)
    else:
        basis = s1.Stage1Basis.load(bdir)
    t_fit = time.perf_counter() - t0
    Z = basis.transform(data["X"], data["genes"])
    S, L, order = ref_arrays(exp_str, spec["panel"], spec["method"], spec["k"])
    w = np.asarray(ref["probe_w"], dtype=np.float64)
    s_probe = probe_target(exp_str, spec["panel"], spec["method"], spec["k"], w) if np.any(np.abs(w) > rc.COEF_NONZERO_TOL) else None
    out_rows = align_rows(S, L, order, Z, basis, rows, int(y.size), int(y.sum()), y, spec, ref, s_probe)
    record = {"basis_id": spec["basis_id"], "spec": spec, "signature": signature, "fitted_now": bool(do_fit),
              "fit_seconds": basis.fit_info.get("fit_seconds"), "n_iter": basis.fit_info.get("n_iter"),
              "seconds": time.perf_counter() - t0, "align_seconds": time.perf_counter() - t0 - t_fit,
              "rows": out_rows, "status": "done"}
    atomic_write_json(record, Path(upath))
    return {"basis_id": spec["basis_id"], "seconds": record["seconds"], "fitted": bool(do_fit)}


def unit_state(ctx: Ctx, spec: Dict[str, Any], signature: Dict[str, Any]) -> str:
    """'done', 'align' (basis saved, alignment missing or stale) or 'fit'."""
    if not ctx.args.resume:
        return "fit"
    saved = s1.Stage1Basis.is_saved(basis_dir(ctx, spec))
    if not saved:
        return "fit"
    rec = read_json_done(unit_path(ctx, spec))
    if rec is not None and rec.get("signature") == signature and isinstance(rec.get("rows"), list) and rec["rows"]:
        return "done"
    return "align"


def run_bases(ctx: Ctx, specs: List[Dict[str, Any]], refs: Dict[str, Dict[str, Any]]) -> None:
    from joblib import Parallel, delayed

    todo = []
    n_done = n_align = 0
    for spec in specs:
        ref = refs[cell_key(spec["panel"], spec["method"], spec["k"])]
        sig = ref_signature(ref)
        state = unit_state(ctx, spec, sig)
        if state == "done":
            n_done += 1
            continue
        n_align += state == "align"
        todo.append((spec, ref, sig, state == "fit"))
    print(f"[bases] {len(specs)} units; {n_done} resumed, {n_align} to re-align only, {len(todo) - n_align} to fit + align "
          f"(n_jobs={ctx.n_jobs})", flush=True)
    if not todo:
        return
    # FA at large K first (slowest) for load balance
    todo.sort(key=lambda t: (not t[3], t[0]["method"] != "fa", -t[0]["k"], t[0]["basis_id"]))
    t0 = time.perf_counter()
    gen = Parallel(n_jobs=ctx.n_jobs, return_as="generator_unordered", verbose=0)(
        delayed(_unit)(ctx.exp_str, spec, str(basis_dir(ctx, spec)), str(unit_path(ctx, spec)), ref, sig, fit)
        for spec, ref, sig, fit in todo
    )
    step = max(1, len(todo) // 20)
    for i, _res in enumerate(gen, start=1):
        if i % step == 0 or i == len(todo):
            el = time.perf_counter() - t0
            eta = el / i * (len(todo) - i)
            print(f"[bases] {i}/{len(todo)} units in {el:.0f} s (eta {eta:.0f} s)", flush=True)
    ctx.timings["bases_s"] = ctx.timings.get("bases_s", 0.0) + time.perf_counter() - t0


# --------------------------------------------------------------------------------------
# part: report
# --------------------------------------------------------------------------------------

LEVERAGE_METRICS = ("axis_loss_fit", "matched_abs_r_fit", "many_to_one_r2_fit", f"angle_top{TOP_M}_max_deg",
                    f"angle_top{TOP_M}_mean_cos2", "angle_allK_mean_cos2", f"ref_axis_containment_top{TOP_M}",
                    "matched_factor_importance_rank")
#: (metric, True if a larger value means more axis movement), as d17.LABEL_MATCHED_METRICS.
LABEL_MATCHED_METRICS = d17.LABEL_MATCHED_METRICS


def _group(arm: str, dropped: str) -> str:
    if arm != "leave_out":
        return "malignant"
    return "highest_prevalence" if set(dropped.split("-")) <= set(HIGH_PREVALENCE) else "contrast"


def leverage_contrast(align: pd.DataFrame, group_cols: Sequence[str] = ("panel_id", "ref_axis", "stage1_method", "effective_k")) -> pd.DataFrame:
    """d17.leverage_table generalised to several panels and reference axes, NaN-safe.

    For a single panel and a single-factor axis with finite metrics it returns the same values
    as d17.leverage_table (tests/test_d26_leverage_panels.py). Metrics that are undefined for an
    axis (matched |r| of the multi-factor probe axis) stay NaN and get no rank p.
    """
    rows = []
    for keys, g in align.groupby(list(group_cols), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        pairs = [(r, g.loc[g["arm"].eq("leave_out_random") & g["matches_drop_set"].eq(r["variant"][len("drop-"):])])
                 for _, r in g.loc[g["arm"].eq("leave_out")].iterrows()]
        for _, r in g.loc[g["arm"].eq("normal_only_all")].iterrows():
            pairs.append((r, g.loc[g["arm"].eq("normal_only_random")]))
        for r, ctrl in pairs:
            dropped = r["variant"][len("drop-"):] if r["arm"] == "leave_out" else "malignant cells"
            row: Dict[str, Any] = dict(zip(group_cols, keys))
            row.update({
                "arm": r["arm"], "dropped": dropped, "group": _group(r["arm"], dropped),
                "n_dropped_cells": r["n_dropped_cells"], "n_dropped_malignant": r["n_dropped_malignant"],
                "n_random_controls": int(len(ctrl)), "ref_factor_label": r["ref_factor_label"],
                "matched_factor_fit": r["matched_factor_fit"],
            })
            for met in LEVERAGE_METRICS:
                val = r.get(met)
                row[met] = val
                if len(ctrl):
                    cv = ctrl[met].astype(float)
                    row[f"{met}_random_mean"] = float(cv.mean())
                    row[f"{met}_random_min"] = float(cv.min())
                    row[f"{met}_random_max"] = float(cv.max())
                    fv = float(val) if val is not None and not (isinstance(val, str)) else float("nan")
                    row[f"{met}_excess"] = fv - float(cv.mean()) if np.isfinite(fv) else np.nan
            if len(ctrl):
                row["axis_loss_fit_ratio_to_random_mean"] = float(r["axis_loss_fit"]) / float(ctrl["axis_loss_fit"].astype(float).mean())
            lab = (g.loc[g["arm"].eq("leave_out_label_random") & g["matches_drop_set"].eq(dropped)]
                   if r["arm"] == "leave_out" else g.iloc[0:0])
            row["n_label_matched_controls"] = int(len(lab))
            for met, larger_moves in LABEL_MATCHED_METRICS:
                if not len(lab):
                    continue
                cv = lab[met].astype(float).to_numpy()
                val = float(r[met]) if r[met] is not None else float("nan")
                if not np.isfinite(val) or not np.all(np.isfinite(cv)):
                    continue
                row[f"{met}_label_matched_mean"] = float(cv.mean())
                row[f"{met}_label_matched_max"] = float(cv.max())
                row[f"{met}_label_matched_min"] = float(cv.min())
                row[f"{met}_label_matched_rank_p"] = rank_p(val, cv, larger_moves)
            if len(lab):
                row["axis_loss_fit_ratio_to_label_matched_mean"] = float(r["axis_loss_fit"]) / float(lab["axis_loss_fit"].astype(float).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def refit_check(align: pd.DataFrame) -> pd.DataFrame:
    tr = align.loc[align["arm"].eq("transductive_refit")].copy()
    cols = ["panel_id", "ref_axis", "stage1_method", "effective_k", "ref_factor_label", "matched_factor_all", "matched_abs_r_all",
            "many_to_one_r2_all", f"angle_top{TOP_M}_max_deg", "angle_allK_max_deg"]
    tr = tr[cols].copy()
    single = tr["ref_axis"].isin(SINGLE_FACTOR_AXES)
    tr["pass"] = (tr["many_to_one_r2_all"].astype(float) >= REFIT_GATE_R2) & (
        ~single | (tr["matched_abs_r_all"].astype(float) >= REFIT_GATE_ABS_R))
    return tr.sort_values(["panel_id", "stage1_method", "effective_k", "ref_axis"]).reset_index(drop=True)


def decision_summary(lev: pd.DataFrame, n_controls_expected: int) -> pd.DataFrame:
    """Per panel x axis: the decision rule fixed before the run, evaluated on axis loss."""
    floor = 1.0 / (1.0 + n_controls_expected)
    rows = []
    lo = lev.loc[lev["arm"].eq("leave_out")]
    for (panel, axis), g in lo.groupby(["panel_id", "ref_axis"], sort=False):
        test = g.loc[g["dropped"].eq(TEST_SET)]
        cells = test[["stage1_method", "effective_k"]].drop_duplicates()
        n_beyond = int((np.isclose(test["axis_loss_fit_label_matched_rank_p"].astype(float), floor)).sum())
        n_above_contrasts = 0
        n_cells = 0
        for _, c in cells.iterrows():
            gc = g.loc[g["stage1_method"].eq(c["stage1_method"]) & g["effective_k"].eq(c["effective_k"])]
            t = gc.loc[gc["dropped"].eq(TEST_SET), "axis_loss_fit_ratio_to_label_matched_mean"]
            cr = gc.loc[gc["dropped"].isin(CONTRAST_SETS), "axis_loss_fit_ratio_to_label_matched_mean"]
            if len(t) and len(cr):
                n_cells += 1
                n_above_contrasts += int(float(t.iloc[0]) > float(cr.max()))
        ratios = test["axis_loss_fit_ratio_to_label_matched_mean"].astype(float)
        p01 = g.loc[g["dropped"].eq("P01"), "axis_loss_fit_ratio_to_label_matched_mean"].astype(float)
        rows.append({
            "panel_id": panel, "ref_axis": axis, "n_method_k_cells": int(len(cells)),
            "n_cells_test_set_above_all_composition_controls": n_beyond,
            "n_cells_test_set_ratio_above_every_contrast": n_above_contrasts,
            "test_set_ratio_min": float(ratios.min()) if len(ratios) else np.nan,
            "test_set_ratio_max": float(ratios.max()) if len(ratios) else np.nan,
            "test_set_axis_loss_max": float(test["axis_loss_fit"].astype(float).max()) if len(test) else np.nan,
            "test_set_matched_abs_r_min": float(test["matched_abs_r_fit"].astype(float).min()) if len(test) else np.nan,
            "p01_ratio_min": float(p01.min()) if len(p01) else np.nan, "p01_ratio_max": float(p01.max()) if len(p01) else np.nan,
            "rank_p_floor": floor,
            "leverage_beyond_composition": bool(n_beyond >= min(DECISION_MIN_CELLS, len(cells))) if len(cells) else False,
            "beyond_composition_and_contrasts": bool(n_beyond >= min(DECISION_MIN_CELLS, len(cells))
                                                     and n_above_contrasts >= min(DECISION_MIN_CELLS, n_cells)) if len(cells) else False,
        })
    return pd.DataFrame(rows)


def d17_reference_rows(ctx: Ctx) -> pd.DataFrame:
    """IFN-alpha rows from D17 (same design, top-gap axis), read-only, for the cross-panel view."""
    path = ctx.exp / "analysis" / "scorecards" / "stage2_diagnostic" / d17.SLUG / "d17_leverage_contrast.csv"
    if not path.is_file():
        return pd.DataFrame()
    lev = pd.read_csv(path)
    lev = lev.loc[lev["arm"].eq("leave_out")].copy()
    lev.insert(0, "ref_axis", PRIMARY_AXIS)
    lev.insert(0, "panel_id", d17.PANEL)
    lev["source"] = f"D17 {d17.SLUG}/d17_leverage_contrast.csv"
    return lev


def _fmt_ratio(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 0.95 else f"{v:.2f}")


def make_figures(ctx: Ctx, align: pd.DataFrame, lev: pd.DataFrame, cross: pd.DataFrame) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    surface, ink, ink2, muted, grid, axis_c = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    method_c = {"pca": "#2a78d6", "fa": "#eb6834"}
    div = LinearSegmentedColormap.from_list("d26_div", ["#2a78d6", "#f0efec", "#e34948"])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": axis_c, "axes.labelcolor": ink,
        "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
    })
    written: List[Path] = []

    def save(fig, stem):
        for ext in ("png", "pdf"):
            path = ctx.figures / f"{stem}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    k_main = max(ctx.ks)
    sets = ["-".join(d) for d in ctx.drop_sets]
    set_label = {s: s.replace("-", "+") for s in sets}
    n_hp = sum(1 for s in sets if set(s.split("-")) <= set(HIGH_PREVALENCE))
    floor = 1.0 / (1.0 + len(ctx.label_seeds))
    explain = ("Axis loss = 1 − R² of the stored reference axis regressed on all K factors of the refit basis, over the "
               "cells kept in the fit: the share of the axis's variance the refit cannot reproduce (0.01 = 1%). "
               f"Right of each row: observed ÷ mean of the {len(ctx.label_seeds)} composition-matched drops, and rank p "
               f"among them (floor {floor:.3f} = above every control).")

    # D26A / D26B: dot plots at the largest K (primary axis, probe axis)
    for stem, axis in (("D26A_leverage_top_gap_axis", "top_gap"), ("D26B_leverage_probe_axis", "probe")):
        a = align.loc[align["ref_axis"].eq(axis) & align["effective_k"].eq(k_main)]
        lv = lev.loc[lev["ref_axis"].eq(axis) & lev["effective_k"].eq(k_main)]
        if a.empty or lv.empty:
            continue
        rows_def = [(s, set_label[s]) for s in sets] + [("malignant cells", "all malignant cells")]
        ys, yv = [], 0.0
        for i, _ in enumerate(rows_def):
            if i in (n_hp, len(sets)):
                yv += 0.9
            ys.append(yv)
            yv += 1.0
        fig, axes = plt.subplots(len(ctx.panels), len(ctx.methods), figsize=(6.9, 3.15 * len(ctx.panels) + 0.7),
                                 sharey=True, squeeze=False)
        dy = 0.27
        lo_x, hi_x = np.inf, 0.0
        for i, panel in enumerate(ctx.panels):
            for jx, m in enumerate(ctx.methods):
                ax = axes[i, jx]
                for (tag, _lab), yy in zip(rows_def, ys):
                    if tag == "malignant cells":
                        obs = lv.loc[lv["panel_id"].eq(panel) & lv["stage1_method"].eq(m) & lv["arm"].eq("normal_only_all")]
                        size = a.loc[a["panel_id"].eq(panel) & a["stage1_method"].eq(m) & a["arm"].eq("normal_only_random"), "axis_loss_fit"]
                        comp = pd.Series(dtype=float)
                    else:
                        obs = lv.loc[lv["panel_id"].eq(panel) & lv["stage1_method"].eq(m) & lv["arm"].eq("leave_out") & lv["dropped"].eq(tag)]
                        size = a.loc[a["panel_id"].eq(panel) & a["stage1_method"].eq(m) & a["arm"].eq("leave_out_random")
                                     & a["matches_drop_set"].eq(tag), "axis_loss_fit"]
                        comp = a.loc[a["panel_id"].eq(panel) & a["stage1_method"].eq(m) & a["arm"].eq("leave_out_label_random")
                                     & a["matches_drop_set"].eq(tag), "axis_loss_fit"]
                    if obs.empty:
                        continue
                    o = obs.iloc[0]
                    vals = [float(o["axis_loss_fit"]), *size.astype(float).tolist(), *comp.astype(float).tolist()]
                    vals = [v for v in vals if v > 0]
                    if vals:
                        lo_x, hi_x = min(lo_x, min(vals)), max(hi_x, max(vals))
                    if len(size):
                        ax.plot([size.min(), size.max()], [yy - dy, yy - dy], color=axis_c, lw=1.4, zorder=2, solid_capstyle="round")
                        ax.plot(size, np.full(len(size), yy - dy), ls="none", marker="|", ms=5, mew=1.1, color=muted, zorder=3)
                    if len(comp):
                        ax.plot([comp.min(), comp.max()], [yy + dy, yy + dy], color=muted, lw=1.0, zorder=2, solid_capstyle="round")
                        ax.plot(comp, np.full(len(comp), yy + dy), ls="none", marker="o", ms=2.6, color=ink2, alpha=0.75, mew=0, zorder=3)
                    ax.plot([o["axis_loss_fit"]], [yy], ls="none", marker="o", ms=6.5, color=method_c.get(m, ink),
                            mec=surface, mew=1.2, zorder=5)
                    if tag != "malignant cells":
                        ratio = float(o.get("axis_loss_fit_ratio_to_label_matched_mean", np.nan))
                        p = float(o.get("axis_loss_fit_label_matched_rank_p", np.nan))
                        txt = f"{_fmt_ratio(ratio)}×  p {p:.2f}" if np.isfinite(p) else f"{_fmt_ratio(ratio)}×"
                        weight = "bold" if np.isfinite(p) and np.isclose(p, floor) else "normal"
                    else:
                        ratio = float(o.get("axis_loss_fit_ratio_to_random_mean", np.nan))
                        txt, weight = f"{_fmt_ratio(ratio)}× size-m.", "normal"
                    ax.text(1.015, yy, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.3,
                            color=ink, fontweight=weight, clip_on=False)
                ax.set_xscale("log")
                ax.grid(axis="x", which="major", color=grid, lw=0.6)
                ax.grid(axis="y", visible=False)
                ref_lab = str(a.loc[a["panel_id"].eq(panel) & a["stage1_method"].eq(m), "ref_factor_label"].iloc[0]) if axis != "probe" else "probe score"
                ax.set_title(f"{panel_label(panel)}, {m.upper()} K = {k_main}: axis {ref_lab}", fontsize=8, pad=4, loc="left")
                if i == len(ctx.panels) - 1:
                    ax.set_xlabel("axis loss, 1 − R²")
        for ax in axes.ravel():
            ax.set_xlim(lo_x / 2.0, hi_x * 2.0)
        axes[0, 0].set_ylim(ys[-1] + 0.6, ys[0] - 0.95)
        for axr in axes[:, 0]:
            axr.set_yticks(ys)
            axr.set_yticklabels([lab for _, lab in rows_def], fontsize=7)
            for i0, text in ((0, "highest-prevalence left out"), (n_hp, "contrasts left out"), (len(sets), "normal cells only")):
                axr.text(0.0, ys[i0] - 0.62, text, transform=axr.get_yaxis_transform(), ha="left", va="center",
                         fontsize=6.6, color=ink, fontweight="bold")
        handles = [
            Line2D([], [], ls="none", marker="o", ms=6.5, color=method_c["pca"], mec=surface, mew=1.2, label="leave-out refit, PCA"),
            Line2D([], [], ls="none", marker="o", ms=6.5, color=method_c["fa"], mec=surface, mew=1.2, label="leave-out refit, FA"),
            Line2D([], [], color=axis_c, lw=1.4, marker="|", ms=5, mew=1.1, mec=muted,
                   label=f"{len(ctx.rand_seeds)} size-matched random drops (above the dot)"),
            Line2D([], [], color=muted, lw=1.0, marker="o", ms=2.6, mfc=ink2, mew=0,
                   label=f"{len(ctx.label_seeds)} composition-matched random drops (below the dot)"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.01))
        what = ("reference axis = the stored factor with the largest all-cell malignant−normal gap" if axis == "top_gap"
                else "reference axis = the score of the atlas-selected Stage 2 probe (all K factors)")
        fig.suptitle(f"D26  Leave-patients-out refits vs random drops, K = {k_main}; {what}", x=0.01, ha="left",
                     fontsize=8.8, fontweight="bold", y=0.995)
        fig.text(0.01, 0.955, explain, ha="left", va="top", fontsize=6.4, color=ink2, wrap=True)
        fig.text(0.5, 0.072, "x axis (log scale): further right = the refit basis reproduces less of the reference axis; "
                 "10⁻³ = 0.1% of its variance lost", ha="center", va="center", fontsize=6.6, color=ink2)
        fig.subplots_adjust(left=0.17, right=0.86, top=0.86, bottom=0.15, wspace=0.42, hspace=0.36)
        save(fig, stem)

    # D26C / D26D: ratio heatmaps over every method x K x leave-out set
    def heatmap(ax, frame: pd.DataFrame, row_keys: List[Tuple[str, str, int]], title: str) -> Any:
        mat = np.full((len(row_keys), len(sets)), np.nan)
        pmat = np.full_like(mat, np.nan)
        for i, (panel, m, k) in enumerate(row_keys):
            g = frame.loc[frame["panel_id"].eq(panel) & frame["stage1_method"].eq(m) & frame["effective_k"].eq(k)]
            for j, s in enumerate(sets):
                h = g.loc[g["dropped"].eq(s)]
                if len(h):
                    mat[i, j] = float(h["axis_loss_fit_ratio_to_label_matched_mean"].iloc[0])
                    pmat[i, j] = float(h["axis_loss_fit_label_matched_rank_p"].iloc[0])
        im = ax.imshow(np.log2(np.clip(mat, 1 / 32, 32)), cmap=div, vmin=-4, vmax=4, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isfinite(mat[i, j]):
                    continue
                lg = abs(np.log2(max(mat[i, j], 1e-9)))
                floor_ij = np.isfinite(pmat[i, j]) and pmat[i, j] <= floor + 1e-9
                ax.text(j, i, _fmt_ratio(mat[i, j]) + ("*" if floor_ij else ""), ha="center", va="center", fontsize=6,
                        color="#ffffff" if lg > 2.6 else ink, fontweight="bold" if floor_ij else "normal")
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([set_label[s] for s in sets], rotation=45, ha="right", fontsize=6.6)
        ax.set_yticks(range(len(row_keys)))
        ax.set_yticklabels([f"{panel_label(p)}  {m.upper()} K{k}" for p, m, k in row_keys], fontsize=6.6)
        ax.axvline(n_hp - 0.5, color=surface, lw=2.5)
        prev = None
        for i, (p, _, _) in enumerate(row_keys):
            if prev is not None and p != prev:
                ax.axhline(i - 0.5, color=surface, lw=2.5)
            prev = p
        ax.set_title(title, fontsize=7.8, loc="left", pad=4)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        return im

    cbar_label = (f"axis loss ÷ mean of the {len(ctx.label_seeds)} composition-matched drops\n"
                  "(log scale; red = the refit loses more than its controls)")
    note = (f"* = above all {len(ctx.label_seeds)} composition-matched controls (rank p = {floor:.3f}). Columns left of the white line: "
            "highest-prevalence patients; right: contrast patients.")
    top = cross if not cross.empty else lev.loc[lev["ref_axis"].eq(PRIMARY_AXIS) & lev["arm"].eq("leave_out")]
    keys_top = [(p, m, k) for p in list(dict.fromkeys(top["panel_id"])) for m in ctx.methods for k in sorted(ctx.ks)
                if len(top.loc[top["panel_id"].eq(p) & top["stage1_method"].eq(m) & top["effective_k"].eq(k)])]
    if keys_top:
        fig, ax = plt.subplots(figsize=(5.6, 0.22 * len(keys_top) + 1.6))
        im = heatmap(ax, top, keys_top, "Primary axis (largest all-cell malignant−normal gap factor)")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-4, -2, 0, 2, 4])
        cb.ax.set_yticklabels(["1/16", "1/4", "1", "4", "16"], fontsize=6.5)
        cb.set_label(cbar_label, fontsize=6.5)
        fig.suptitle("D26  Leverage beyond malignant composition, every method × K (IFN-α rows from D17)", x=0.01,
                     ha="left", fontsize=8.8, fontweight="bold")
        fig.text(0.01, -0.01, note, ha="left", va="top", fontsize=6.3, color=ink2)
        save(fig, "D26C_ratio_heatmap_top_gap_axis")
    sec = lev.loc[lev["arm"].eq("leave_out") & lev["ref_axis"].isin(["top_heldout", "probe"])]
    keys_sec = [(p, m, k) for p in ctx.panels for m in ctx.methods for k in sorted(ctx.ks)]
    if not sec.empty:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 0.22 * len(keys_sec) + 1.7), sharey=True)
        for axh, axis, title in ((axs[0], "top_heldout", "Largest median held-out lift factor"),
                                 (axs[1], "probe", "Atlas-selected probe direction")):
            im = heatmap(axh, sec.loc[sec["ref_axis"].eq(axis)], keys_sec, title)
        cb = fig.colorbar(im, ax=list(axs), fraction=0.03, pad=0.02, ticks=[-4, -2, 0, 2, 4])
        cb.ax.set_yticklabels(["1/16", "1/4", "1", "4", "16"], fontsize=6.5)
        cb.set_label(cbar_label, fontsize=6.5)
        fig.suptitle("D26  Secondary reference axes: axis loss relative to composition-matched controls", x=0.01, ha="left",
                     fontsize=8.8, fontweight="bold")
        fig.text(0.01, -0.01, note, ha="left", va="top", fontsize=6.3, color=ink2)
        save(fig, "D26D_ratio_heatmap_secondary_axes")
    return written


def _num(v: Any, fmt: str = "{:.4g}") -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(x) else fmt.format(x)


def write_findings(ctx: Ctx, refax: pd.DataFrame, check: pd.DataFrame, lev: pd.DataFrame, decision: pd.DataFrame,
                   cross: pd.DataFrame) -> Path:
    L: List[str] = []
    L.append(f"# {ANALYSIS_ID} leave-high-prevalence-out leverage test, allograft rejection and apoptosis ({SLUG})")
    L.append("")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
             "scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py (task W2.4; D17 design). "
             "Descriptive: Stage 1 alignment only (no Stage 2 refit), seven supported patients, one-sided empirical rank p "
             f"over {len(ctx.label_seeds)} composition-matched controls, not corrected for the number of contrasts. "
             "FactoSig and FactoSig-promax (the atlas winner on both panels) are out of scope: their refit is not available "
             "in this CPU runner.")
    L.append("")
    L.append("## Reference axes (stored transductive basis)")
    cols = ["panel_id", "stage1_method", "effective_k", "ref_axis", "ref_factor_label", "all_cell_std_gap", "median_univariate_lift_pp",
            "rank_by_abs_gap", "rank_by_median_heldout_lift", "d19_modal_argmax_factor", "d19_n_folds_argmax_is_top_gap",
            "d18_max_abs_diff_all_factors_pp", "probe_n_nonzero", "probe_cos_vs_stored_discovery", "probe_r_ref_vs_stored_scores"]
    L.append("")
    L.append(_md_table(refax[[c for c in cols if c in refax.columns]]))
    L.append("")
    L.append("## Refit sanity (transductive_refit vs the stored basis)")
    L.append(f"Gate: many-to-one R2 >= {REFIT_GATE_R2} for every axis and matched |r| >= {REFIT_GATE_ABS_R} for single-factor axes: "
             f"{int(check['pass'].sum())}/{len(check)} rows pass; min R2 {_num(check['many_to_one_r2_all'].min())}, "
             f"min matched |r| {_num(check.loc[check['ref_axis'].isin(SINGLE_FACTOR_AXES), 'matched_abs_r_all'].min())}.")
    L.append("")
    L.append("## Decision rule (fixed before the run) and outcome")
    L.append(f"Leverage beyond composition for a panel x axis: the {TEST_SET.replace('-', '+')} axis loss lies above all "
             f"{len(ctx.label_seeds)} composition-matched controls (rank p = {1 / (1 + len(ctx.label_seeds)):.3f}) in at least "
             f"{DECISION_MIN_CELLS} of the method x K cells. H-S1-05 additionally needs its ratio to the control mean to exceed "
             f"that of every contrast set ({', '.join(c.replace('-', '+') for c in CONTRAST_SETS)}) in at least {DECISION_MIN_CELLS} cells. "
             f"Primary axis: {PRIMARY_AXIS}.")
    L.append("")
    L.append(_md_table(decision))
    L.append("")
    lcols = ["stage1_method", "effective_k", "dropped", "group", "n_dropped_cells", "n_dropped_malignant", "ref_factor_label",
             "axis_loss_fit", "axis_loss_fit_ratio_to_random_mean", "axis_loss_fit_label_matched_mean", "axis_loss_fit_label_matched_max",
             "axis_loss_fit_ratio_to_label_matched_mean", "axis_loss_fit_label_matched_rank_p", "matched_abs_r_fit",
             f"angle_top{TOP_M}_max_deg", f"angle_top{TOP_M}_max_deg_label_matched_rank_p"]
    for axis in AXES:
        for panel in ctx.panels:
            sub = lev.loc[lev["ref_axis"].eq(axis) & lev["panel_id"].eq(panel)]
            if sub.empty:
                continue
            L.append(f"## {panel_label(panel)}: {axis} axis ({AXIS_LABELS[axis]})")
            L.append("")
            sub = sub.sort_values(["stage1_method", "effective_k", "arm"])
            L.append(_md_table(sub[[c for c in lcols if c in sub.columns]]))
            L.append("")
    if cross is not None and not cross.empty:
        L.append("## Cross-panel view of the primary axis (IFN-alpha rows from D17, read-only)")
        L.append("")
        xc = ["panel_id", "stage1_method", "effective_k", "dropped", "axis_loss_fit", "axis_loss_fit_ratio_to_label_matched_mean",
              "axis_loss_fit_label_matched_rank_p", "matched_abs_r_fit"]
        hp = cross.loc[cross["dropped"].isin([TEST_SET, "P01"])].sort_values(["dropped", "panel_id", "stage1_method", "effective_k"])
        L.append(_md_table(hp[[c for c in xc if c in hp.columns]]))
        L.append("")
    L.append("## Notes")
    L.append("- Axis loss = 1 - many-to-one R2 of the reference axis on all K refit factors over the retained cells "
             "(d17 convention: axis_loss_fit); axis_loss_all also counts the projected left-out cells.")
    L.append("- top_gap is the D17 rule; with balanced L1 it is the factor with the largest entry margin. top_heldout uses the D18 "
             "univariate held-out lift (training-fold sign). The probe axis is the atlas-selected (penalty, C, l1_ratio) refit on "
             "all cells of the stored scores; it is used only to define a direction, not evaluated held-out.")
    L.append("- Size-matched controls (seeds 1-3) drop as many random cells; composition-matched controls (seeds 101-120) also "
             "drop as many malignant cells. Seeds equal D17's, so the dropped cells are the same across panels.")
    L.append("- Matched |r| and containment are undefined for the multi-factor probe axis (NaN); its best single-factor |r| is in "
             "d26_alignment.csv.")
    path = ctx.tables / f"{SLUG}_findings.md"
    atomic_write_text("\n".join(L) + "\n", path)
    return path


def write_manifest_and_ledger(ctx: Ctx, started: float, extra: Dict[str, Any]) -> None:
    head, dirty = git_state(REPO)
    manifest_path = ctx.tables / f"{SLUG}_manifest.json"
    history: List[Dict[str, Any]] = []
    try:
        history = list(json.loads(manifest_path.read_text(encoding="utf-8")).get("run_history", []))
    except Exception:
        pass
    history.append({
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - started, 1), "timings_s": {k: round(v, 1) for k, v in ctx.timings.items()},
        "argv": " ".join(shlex.quote(a) for a in sys.argv[1:]), "git_head": head, "git_dirty": dirty, "n_jobs": ctx.n_jobs,
    })
    manifest = {
        "slug": SLUG, "analysis": ANALYSIS_ID, "task": TASK_ID, "experiment_id": ctx.exp.name,
        "script": "scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py",
        "modules": ["scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/stage1_refit.py",
                    "scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d17_basis_variant_pilot.py",
                    "scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py"],
        "stage1_refit_version": s1.MODULE_VERSION, "contract_version": rc.CONTRACT_VERSION, "git_head": head, "git_dirty": dirty,
        "python": platform.python_version(),
        "inputs": {
            "features": [features_rel(p) + "/features.npz" for p in ctx.panels],
            "stored_bases": [f"stage1_dr/panel_id={p}/method={m}/k={k}/seed={SEED}/{{scores,loadings}}.npy"
                             for p in ctx.panels for m in ctx.methods for k in ctx.ks],
            "cell_metadata": ctx.rel(cell_metadata()),
            "atlas_configs": f"analysis/scorecards/stage2_diagnostic/{D19_SLUG}/d19_atlas_selected_regularization.csv",
            "d19_margin_audit": f"analysis/scorecards/stage2_diagnostic/{D19_SLUG}/d19_entry_margin_audit_by_fold.csv",
            "d18_cards": f"analysis/scorecards/stage2_diagnostic/{D18_SLUG}/factor_cards_index.csv",
            "stored_discovery_coefficients": DISCOVERY_COEF_REL,
            "d17_leverage": f"analysis/scorecards/stage2_diagnostic/{d17.SLUG}/d17_leverage_contrast.csv",
        },
        "parameters": {
            "panels": ctx.panels, "methods": ctx.methods, "ks": ctx.ks, "seed": SEED, "supported_patients": ctx.supported,
            "drop_sets": ["-".join(d) for d in ctx.drop_sets], "random_drop_seeds": list(ctx.rand_seeds),
            "label_matched_drop_seeds": list(ctx.label_seeds), "reference_axes": list(AXES), "primary_axis": PRIMARY_AXIS,
            "top_m": TOP_M, "refit_gate": {"many_to_one_r2": REFIT_GATE_R2, "matched_abs_r": REFIT_GATE_ABS_R},
            "decision_rule": {"test_set": TEST_SET, "contrast_sets": list(CONTRAST_SETS), "min_cells": DECISION_MIN_CELLS},
            "pca": "sklearn PCA svd_solver=randomized random_state=42", "fa": "sklearn FactorAnalysis svd_method=randomized random_state=42",
            "standardization": "per-gene mean / population SD of the Stage 1 fit cells; zero variance -> 0",
            "bases_location": ctx.rel(ctx.bases), "smoke": bool(ctx.args.smoke),
            "out_of_scope": "factosig, factosig_promax (no CPU refit in this runner)",
        },
        "c_policy": rc.c_policy_record(c_policy="not_applicable", training_n_changed=False, penalties=[],
                                       fixed_arms=["probe axis: atlas-selected (penalty, C, l1_ratio) refit on all cells; direction only"],
                                       notes="Stage 1 alignment only; no Stage 2 probe is evaluated held-out."),
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
    ledger.append({"path": ctx.rel(ctx.units), "kind": "cache_dir", "bytes": None, "sha256": f"{n_units} alignment unit JSON"})
    ledger.append({"path": ctx.rel(ctx.bases), "kind": "bases_dir", "bytes": None, "sha256": f"{n_bases} bases (basis.npz + basis.json)"})
    atomic_write_csv(pd.DataFrame(ledger), ctx.tables / f"{SLUG}_artifact_ledger.csv")


def collect_units(ctx: Ctx, specs: List[Dict[str, Any]], refs: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    rows, missing = [], []
    for spec in specs:
        sig = ref_signature(refs[cell_key(spec["panel"], spec["method"], spec["k"])])
        rec = read_json_done(unit_path(ctx, spec))
        if rec is None or rec.get("signature") != sig or not rec.get("rows"):
            missing.append(spec["basis_id"])
            continue
        rows.extend(rec["rows"])
    return pd.DataFrame(rows), missing


def run_report(ctx: Ctx, specs: List[Dict[str, Any]], refs: Dict[str, Dict[str, Any]], started: float) -> int:
    t0 = time.perf_counter()
    failed = ctx.tables / "merge_failed.json"
    align, missing = collect_units(ctx, specs, refs)
    if missing:
        atomic_write_json({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_missing": len(missing),
                           "examples": missing[:10], "recover": "rerun with --resume (parts refs,bases,report) or --merge-only"},
                          failed)
        print(f"[report] {len(missing)} alignment units are missing or stale (e.g. {missing[:3]}); wrote merge_failed.json. "
              "Rerun with --resume.", flush=True)
        return 3
    align["panel_label"] = align["panel_id"].map(panel_label)
    atomic_write_csv(align, ctx.tables / "d26_alignment.csv")
    check = refit_check(align)
    atomic_write_csv(check, ctx.tables / "d26_refit_check.csv")
    ok = bool(len(check) and check["pass"].all())
    print(f"[report] refit sanity: {int(check['pass'].sum())}/{len(check)} rows pass", flush=True)
    if not ok and not ctx.args.allow_anchor_failure:
        atomic_write_json({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reason": "refit sanity gate failed"}, failed)
        print("[report] FAILED: the transductive refit does not reproduce the stored basis; no leverage numbers are reported.", flush=True)
        return 2
    lev = leverage_contrast(align)
    atomic_write_csv(lev, ctx.tables / "d26_leverage_contrast.csv")
    decision = decision_summary(lev, len(ctx.label_seeds))
    atomic_write_csv(decision, ctx.tables / "d26_decision_summary.csv")
    cross = pd.DataFrame()
    if not ctx.args.smoke:
        ifna = d17_reference_rows(ctx)
        mine = lev.loc[lev["ref_axis"].eq(PRIMARY_AXIS) & lev["arm"].eq("leave_out")].copy()
        mine["source"] = f"D26 {SLUG}/d26_leverage_contrast.csv"
        cross = pd.concat([ifna, mine], ignore_index=True, sort=False) if not ifna.empty else mine
        keep = ["panel_id", "ref_axis", "stage1_method", "effective_k", "dropped", "group", "n_dropped_cells", "n_dropped_malignant",
                "ref_factor_label", "axis_loss_fit", "axis_loss_fit_ratio_to_random_mean", "axis_loss_fit_label_matched_mean",
                "axis_loss_fit_ratio_to_label_matched_mean", "axis_loss_fit_label_matched_rank_p", "matched_abs_r_fit",
                f"angle_top{TOP_M}_max_deg", "n_label_matched_controls", "source"]
        cross = cross[[c for c in keep if c in cross.columns]]
        atomic_write_csv(cross, ctx.tables / "d26_cross_panel_top_gap.csv")
    refax = pd.read_csv(ctx.tables / "d26_reference_axes.csv")
    figs = make_figures(ctx, align, lev, cross)
    findings = write_findings(ctx, refax, check, lev, decision, cross)
    ctx.timings["report_s"] = time.perf_counter() - t0
    if failed.is_file():
        failed.unlink()
    write_manifest_and_ledger(ctx, started, {
        "counts": {"bases": int(len(specs)), "alignment_rows": int(len(align)), "leverage_rows": int(len(lev)),
                   "refit_check_pass": int(check["pass"].sum()), "refit_check_rows": int(len(check))},
        "outputs": {"findings": ctx.rel(findings), "figures": [ctx.rel(p) for p in figs]},
    })
    print(f"[report] tables, {len(figs)} figure files, findings, manifest and ledger written in {ctx.timings['report_s']:.1f} s", flush=True)
    return 0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", default=",".join(DEFAULT_PARTS), help="comma list of refs,bases,report")
    p.add_argument("--panels", default=",".join(DEFAULT_PANELS),
                   help="comma list of Stage 0 panel ids (or hallmark suffixes such as 'apoptosis')")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    p.add_argument("--n-jobs", type=int, default=12, help="joblib workers (capped at 12; one BLAS thread each)")
    p.add_argument("--resume", action="store_true", help="skip finished units (saved basis + current alignment JSON)")
    p.add_argument("--merge-only", action="store_true", help="only the report part, from finished units (recovery after merge_failed)")
    p.add_argument("--smoke", action="store_true", help="2 drop sets (P02+P03+P04, P09), 1 size-matched and 2 composition-matched seeds")
    p.add_argument("--out-root", default="", help="write everything under this dir instead of EXP/analysis")
    p.add_argument("--bases-location", choices=("cache", "stage1_dr_variants"), default="cache",
                   help="cache: EXP/analysis/cache/stage2_diagnostic/<slug>/bases (default); stage1_dr_variants: EXP/stage1_dr_variants/<slug>")
    p.add_argument("--allow-anchor-failure", action="store_true", help="continue past a failed refit sanity gate (debug only)")
    p.add_argument("--plan-only", action="store_true", help="print the plan and the unit states, then exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    parts = ["report"] if args.merge_only else [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = set(parts) - set(DEFAULT_PARTS)
    if unknown:
        raise SystemExit(f"unknown parts: {sorted(unknown)}")
    started = time.perf_counter()
    ctx = Ctx(args)
    specs = build_plan(ctx)
    print(f"[plan] {len(specs)} bases; panels {[panel_tag(p) for p in ctx.panels]}, methods {ctx.methods}, K {ctx.ks}; "
          f"outputs under {ctx.rel(ctx.tables)}", flush=True)
    if args.plan_only:
        return 0
    refs = run_refs(ctx) if "refs" in parts else load_refs(ctx)
    if "bases" in parts:
        run_bases(ctx, specs, refs)
    status = 0
    if "report" in parts:
        status = run_report(ctx, specs, refs, started)
    print(f"[done] wall {time.perf_counter() - started:.1f} s (status {status})", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
