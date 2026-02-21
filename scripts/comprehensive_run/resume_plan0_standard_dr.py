#!/usr/bin/env python
"""
Resume / extend Plan 0 *standard* DR caches (PCA / FA / FactoSig) inside an
existing Plan 0 experiment directory, without re-running preprocessing.

Why:
- Plan 0 creates preprocessing outputs at:
    preprocessing/adata_processed.h5ad
- Standard DR methods are cached as arrays under:
    analysis/plan0/stability/<method>/k_<K>/replicates/seed_<seed>/{scores.npy,loadings.npy,extras.json}
- If an experiment crashed mid-run (or you later decide to add more seeds or
  additional method variants like FactoSig rotations), this script appends the
  missing replicate caches *in the same experiment directory*.

Notes:
- This script is intentionally scoped to standardized methods (pca/fa/factosig*).
- It is append-only: existing replicate caches are reused and not overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(p: Path, obj: Any) -> None:
    _ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _split_list_arg(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        s = " ".join(str(x) for x in v)
    else:
        s = str(v)
    s = s.replace(",", " ")
    return [t for t in s.split() if t]


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {"_raw": obj}


def _infer_factosig_defaults(exp_dir: Path, ks: List[int]) -> Tuple[str, Optional[str]]:
    """
    Best-effort: infer (rotation, order_factors_by) from existing FactoSig caches.
    If not found, fall back to ('varimax', 'ss_loadings').
    """
    for k in ks:
        extras_p = (
            exp_dir
            / "analysis"
            / "plan0"
            / "stability"
            / "factosig"
            / f"k_{int(k)}"
            / "replicates"
            / "seed_1"
            / "extras.json"
        )
        if extras_p.exists():
            ex = _read_json(extras_p)
            rot = str(ex.get("rotation", "varimax"))
            order = ex.get("order_factors_by", "ss_loadings")
            order = None if (order is None or str(order).lower() == "none") else str(order)
            return rot, order
    return "varimax", "ss_loadings"


def _run_method(
    *,
    method_token: str,
    adata: sc.AnnData,
    k: int,
    seed: int,
    factosig_device: str,
    factosig_lr: float,
    factosig_max_iter: int,
    factosig_rotation_default: str,
    factosig_order_default: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Return (scores, loadings, extras_uns) where:
    - scores: (n_cells, k)
    - loadings: (n_genes, k) aligned to adata.var
    - extras_uns: dict to save into extras.json
    """
    # Lazy imports to avoid import-path fragility when running as a script
    from sc_classification.dimension_reduction.pca import PCA as PCAWrapper
    from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis as FAWrapper
    from sc_classification.dimension_reduction.factosig import FactoSigDR

    token = method_token.lower().strip()
    base = token.split("_", 1)[0]

    ad = adata.copy()
    if base == "pca":
        model = PCAWrapper()
        ad = model.fit_transform(ad, n_components=int(k), random_state=int(seed), standardize_input=False, svd_solver="randomized")
        return np.asarray(ad.obsm["X_pca"]), np.asarray(ad.varm["PCA_loadings"]), dict(ad.uns.get("pca", {}))

    if base == "fa":
        model = FAWrapper()
        ad = model.fit_transform(
            ad,
            n_components=int(k),
            random_state=int(seed),
            standardize_input=False,
            svd_method="randomized",
            save_fitted_models=False,
        )
        return np.asarray(ad.obsm["X_fa"]), np.asarray(ad.varm["FA_loadings"]), dict(ad.uns.get("fa", {}))

    if base == "factosig":
        rot = (factosig_rotation_default or "varimax").lower()
        if token == "factosig_varimax":
            rot = "varimax"
        elif token == "factosig_promax":
            rot = "promax"
        order = factosig_order_default
        model = FactoSigDR()
        ad = model.fit_transform(
            ad,
            n_components=int(k),
            random_state=int(seed),
            device=str(factosig_device),
            lr=float(factosig_lr),
            max_iter=int(factosig_max_iter),
            verbose=False,
            save_fitted_models=False,
            rotation=str(rot),
            order_factors_by=order,
        )
        return np.asarray(ad.obsm["X_factosig"]), np.asarray(ad.varm["FACTOSIG_loadings"]), dict(ad.uns.get("factosig", {}))

    raise ValueError(f"Unsupported method for standard-DR resume: {method_token}")


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-12
    return float(abs(np.dot(a0, b0) / denom))


def match_components_by_loading_corr(loadings_a: np.ndarray, loadings_b: np.ndarray) -> Dict[str, Any]:
    k = loadings_a.shape[1]
    corr = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            corr[i, j] = _corr_abs(loadings_a[:, i], loadings_b[:, j])
    best_a = corr.max(axis=1).tolist()
    best_b = corr.max(axis=0).tolist()
    return {"corr_matrix": corr, "best_a": best_a, "best_b": best_b}


def consensus_cluster_components(
    all_loadings: np.ndarray,  # (n_runs*K, n_genes)
    k: int,
    *,
    random_state: int = 1,
    outlier_max_sim_threshold: float = 0.2,
) -> Dict[str, Any]:
    # Keep this identical in spirit to the comprehensive runner:
    # L2-normalize, filter out components with max_sim<threshold, cluster via kmeans.
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_comp, _ = all_loadings.shape
    norms = np.linalg.norm(all_loadings, axis=1, keepdims=True) + 1e-12
    X = all_loadings / norms

    Xc = X - X.mean(axis=1, keepdims=True)
    Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    sim = np.abs(Xc @ Xc.T)
    np.fill_diagonal(sim, 0.0)
    max_sim = sim.max(axis=1)
    keep = max_sim >= float(outlier_max_sim_threshold)
    kept_idx = np.where(keep)[0]
    X_keep = X[keep]
    if X_keep.shape[0] < int(k):
        X_keep = X
        kept_idx = np.arange(n_comp)

    km = KMeans(n_clusters=int(k), n_init=20, random_state=int(random_state))
    labels = km.fit_predict(X_keep)

    consensus_loadings = np.zeros((int(k), all_loadings.shape[1]), dtype=float)
    cluster_sizes: List[int] = []
    for c in range(int(k)):
        idx = np.where(labels == c)[0]
        cluster_sizes.append(int(idx.size))
        if idx.size == 0:
            continue
        consensus_loadings[c, :] = np.median(all_loadings[kept_idx[idx], :], axis=0)

    sil = None
    if X_keep.shape[0] >= int(k) + 1 and len(set(labels.tolist())) > 1:
        try:
            sil = float(silhouette_score(X_keep, labels))
        except Exception:
            sil = None

    return {
        "kept_n": int(X_keep.shape[0]),
        "n_components_total": int(n_comp),
        "cluster_sizes": cluster_sizes,
        "silhouette": sil,
        "outlier_max_sim_threshold": float(outlier_max_sim_threshold),
        "consensus_loadings": consensus_loadings,
    }


def resume_plan0_standard_dr(
    *,
    experiment_dir: Path,
    ks: List[int],
    seeds: List[int],
    methods: List[str],
    factosig_rotation: str,
    factosig_order_factors_by: Optional[str],
    factosig_device: str,
    factosig_lr: float,
    factosig_max_iter: int,
    dry_run: bool,
) -> None:
    exp_dir = experiment_dir.resolve()
    adata_path = exp_dir / "preprocessing" / "adata_processed.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"Missing preprocessed AnnData: {adata_path}")

    # Inherit defaults from existing caches to preserve comparability
    inferred_rot, inferred_order = _infer_factosig_defaults(exp_dir, ks)
    if not factosig_rotation:
        factosig_rotation = inferred_rot
    if factosig_order_factors_by is None:
        factosig_order_factors_by = inferred_order

    diag_dir = _ensure_dir(exp_dir / "analysis" / "plan0")
    _write_json(
        diag_dir / "resume_standard_dr_request.json",
        {
            "requested_ks": [int(x) for x in ks],
            "requested_seeds": [int(x) for x in seeds],
            "requested_methods": methods,
            "factosig": {
                "rotation_default": str(factosig_rotation),
                "order_factors_by_default": factosig_order_factors_by,
                "inferred_from_existing": {"rotation": inferred_rot, "order_factors_by": inferred_order},
            },
            "dry_run": bool(dry_run),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )

    if dry_run:
        ad = None
    else:
        ad = sc.read_h5ad(str(adata_path))

    for method in methods:
        method_l = method.lower().strip()
        for k in ks:
            method_dir = _ensure_dir(diag_dir / "stability" / method_l / f"k_{int(k)}")
            rep_root = _ensure_dir(method_dir / "replicates")

            # Fit any missing seeds
            for seed in seeds:
                rep_dir = _ensure_dir(rep_root / f"seed_{int(seed)}")
                scores_p = rep_dir / "scores.npy"
                loadings_p = rep_dir / "loadings.npy"
                extras_p = rep_dir / "extras.json"

                if scores_p.exists() and loadings_p.exists() and extras_p.exists():
                    continue

                if dry_run:
                    print(f"[dry-run] Would run method={method_l} k={k} seed={seed}")
                    continue

                assert ad is not None
                t0 = time.time()
                scores, loadings, extras = _run_method(
                    method_token=method_l,
                    adata=ad,
                    k=int(k),
                    seed=int(seed),
                    factosig_device=str(factosig_device),
                    factosig_lr=float(factosig_lr),
                    factosig_max_iter=int(factosig_max_iter),
                    factosig_rotation_default=str(factosig_rotation),
                    factosig_order_default=factosig_order_factors_by,
                )
                np.save(scores_p, np.asarray(scores, dtype=np.float32))
                np.save(loadings_p, np.asarray(loadings, dtype=np.float32))
                _write_json(extras_p, extras)
                print(f"[resume] Done method={method_l} k={k} seed={seed} ({time.time() - t0:.1f}s)")

            if dry_run:
                continue

            # After fitting, recompute stability summaries using all available seeds for this (method,k)
            seed_dirs = sorted([p for p in rep_root.glob("seed_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
            loadings_runs: List[np.ndarray] = []
            for sd in seed_dirs:
                lp = sd / "loadings.npy"
                if lp.exists():
                    loadings_runs.append(np.load(lp))

            # Pairwise stability (adjacent in seed order)
            pair_stats: List[Dict[str, Any]] = []
            for i in range(len(loadings_runs) - 1):
                mobj = match_components_by_loading_corr(loadings_runs[i], loadings_runs[i + 1])
                best_a = mobj.get("best_a", [])
                pair_stats.append(
                    {
                        "pair": [int(i), int(i + 1)],
                        "best_a_median": float(np.median(best_a)) if best_a else None,
                        "best_a_mean": float(np.mean(best_a)) if best_a else None,
                        "best_a_frac_lt_0p3": float(np.mean(np.array(best_a) < 0.3)) if best_a else None,
                    }
                )
            _write_json(method_dir / "pairwise_stability_summary.json", {"pairs": pair_stats, "n_runs": int(len(loadings_runs))})

            # Consensusness cache for FA + FactoSig variants if multiple runs exist
            if method_l.startswith("fa") or method_l.startswith("factosig"):
                if len(loadings_runs) >= 2:
                    stacked = np.concatenate([L.T for L in loadings_runs], axis=0)  # (runs*k, genes)
                    cons = consensus_cluster_components(stacked, k=int(k), random_state=1, outlier_max_sim_threshold=0.2)
                    cons_dir = _ensure_dir(method_dir / "consensus_cache")
                    np.save(cons_dir / "consensus_loadings.npy", np.asarray(cons["consensus_loadings"], dtype=np.float32))
                    _write_json(
                        cons_dir / "consensus_metrics.json",
                        {
                            "kept_n": cons["kept_n"],
                            "n_components_total": cons["n_components_total"],
                            "cluster_sizes": cons["cluster_sizes"],
                            "silhouette": cons["silhouette"],
                            "outlier_max_sim_threshold": cons["outlier_max_sim_threshold"],
                        },
                    )


def main() -> None:
    p = argparse.ArgumentParser(description="Resume/extend Plan 0 standard DR caches in an existing experiment dir.")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--ks", default="20,40,60", help="Comma/space-separated K list, e.g. '20,40,60'.")
    p.add_argument("--seeds", default="1,2,3,4,5", help="Comma/space-separated seed list.")
    p.add_argument("--methods", default="pca,fa,factosig", help="Methods to resume: pca,fa,factosig,factosig_varimax,factosig_promax")

    p.add_argument("--factosig-rotation", default="", help="Default FactoSig rotation if method token is plain 'factosig'.")
    p.add_argument(
        "--factosig-order-factors-by",
        default="__infer__",
        help="ss_loadings | score_variance | none | __infer__ (default: infer from existing seed_1 extras if present).",
    )
    p.add_argument("--factosig-device", default="cpu")
    p.add_argument("--factosig-lr", type=float, default=1e-2)
    p.add_argument("--factosig-max-iter", type=int, default=300)
    p.add_argument("--dry-run", action="store_true", help="Do not run fits; only report what would run.")

    args = p.parse_args()

    exp_dir = Path(args.experiment_dir)
    ks = [int(x) for x in _split_list_arg(args.ks)]
    seeds = [int(x) for x in _split_list_arg(args.seeds)]
    methods = [m.strip() for m in _split_list_arg(args.methods)]

    factosig_rotation = str(args.factosig_rotation).strip()
    ofb_raw = str(args.factosig_order_factors_by).strip().lower()
    if ofb_raw == "__infer__":
        factosig_order = None
    elif ofb_raw == "none":
        factosig_order = None
    else:
        factosig_order = str(args.factosig_order_factors_by)

    resume_plan0_standard_dr(
        experiment_dir=exp_dir,
        ks=ks,
        seeds=seeds,
        methods=methods,
        factosig_rotation=factosig_rotation,
        factosig_order_factors_by=factosig_order,
        factosig_device=str(args.factosig_device),
        factosig_lr=float(args.factosig_lr),
        factosig_max_iter=int(args.factosig_max_iter),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    # Allow ad-hoc execution without editable install (mirror other scripts)
    try:
        import sc_classification  # noqa: F401
    except Exception:
        here = Path(__file__).resolve()
        pkg_root = here.parents[2]  # .../sc_classification
        src = pkg_root / "src"
        if src.exists():
            sys.path.insert(0, str(src))
    main()

