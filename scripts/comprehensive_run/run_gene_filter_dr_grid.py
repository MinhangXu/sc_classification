#!/usr/bin/env python
"""
Plan 0 + Plan 1 runners for gene-filtering eval:

- Plan 0: K sweep + run-to-run stability + (optional) cNMF K selection.
- Plan 1: preprocess_method × DR_method grid, saving one AnnData per preprocess method
          with multiple DR embeddings attached (.obsm), and writing classification
          metrics in an organized, non-overwriting directory structure.

Design goals:
- Keep replicate-level artifacts compact (arrays / npz); don't bloat a single AnnData.
- Reuse existing sc_classification components where possible.
"""

from __future__ import annotations

import sys
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

# Allow running without an editable install by adding the package src/ to sys.path.
# (Preferred workflow is still `pip install -e sc_classification`, but this makes
# ad-hoc runs on new machines less fragile.)
try:
    import sc_classification  # noqa: F401
except Exception:
    _pkg_root = Path(__file__).resolve().parents[2]  # .../sc_classification
    _src = _pkg_root / "src"
    if _src.exists():
        sys.path.insert(0, str(_src))

from sc_classification.utils.experiment_manager import ExperimentConfig, ExperimentManager
from sc_classification.utils.preprocessing import PreprocessingPipeline, create_preprocessing_config
from sc_classification.dimension_reduction.pca import PCA as PCAWrapper
from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis as FAWrapper
from sc_classification.dimension_reduction.nmf import NMF as NMFWrapper
from sc_classification.dimension_reduction.factosig import FactoSigDR
from sc_classification.classification_methods.lr_lasso import LRLasso


# -----------------------------
# Helpers: IO + conventions
# -----------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(p: Path, obj: Any) -> None:
    _ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _split_list_arg(v: Any) -> List[str]:
    """
    Accept either:
    - a comma-separated string: "20,40,60"
    - a space-separated token list (often with commas): ["20,", "40,", "60"]
    and return clean tokens ["20","40","60"].
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        s = " ".join(str(x) for x in v)
    else:
        s = str(v)
    s = s.replace(",", " ")
    return [t for t in s.split() if t]


def _readable_tag(d: Dict[str, Any]) -> str:
    """Stable-ish identifier for a preprocess method dict."""
    method = d.get("method", "unknown")
    if method == "hvg":
        return f"hvg_{int(d.get('n_top_genes', 3000))}"
    if method == "all_filtered":
        return f"all_filtered_minfrac{d.get('min_cells_fraction', 0.01)}_ratio{d.get('malignant_enrichment_ratio', 'NA')}"
    if method == "deg_weak_screen":
        return f"deg_weak_q{d.get('pval_threshold', 0.1)}_lfc{d.get('lfc_threshold', 0.05)}"
    if method in ("hvg_plus_rescue_union", "hybrid_hvg_rescue"):
        return f"hybrid_hvg{int(d.get('n_top_genes', 3000))}_minfrac{d.get('min_cells_fraction', 0.01)}_ratio{d.get('malignant_enrichment_ratio', 'NA')}"
    return method


def _load_adata(input_h5ad: str) -> sc.AnnData:
    ad = sc.read_h5ad(input_h5ad)
    return ad


def _ensure_timepoint_type(
    adata: sc.AnnData,
    *,
    timepoint_type_key: str = "timepoint_type",
    time_key: str = "Time",
) -> sc.AnnData:
    """
    Ensure `adata.obs[timepoint_type_key]` exists.

    If missing but `adata.obs[time_key]` exists (values like "MRD_1", "preSCT_2"),
    derive coarse type by stripping the trailing underscore + digits.

    Mirrors logic from `lineage_reg_tree/preprocess_adata/inspect_adata.ipynb`.
    """
    if timepoint_type_key in adata.obs.columns:
        return adata
    if time_key not in adata.obs.columns:
        return adata

    # Derive: MRD_1 -> MRD, preSCT_2 -> preSCT, etc.
    tp = (
        adata.obs[time_key]
        .astype(str)
        .str.replace(r"_[0-9]+$", "", regex=True)
        .replace({"unknown": np.nan})
    )
    adata = adata.copy()
    adata.obs[timepoint_type_key] = tp
    return adata


def _extract_counts_for_cnmf(adata: sc.AnnData) -> sc.AnnData:
    """
    Best-effort extraction of a counts matrix for cNMF.

    Priority:
    - adata.raw if present
    - adata.layers['counts'] if present
    - adata.X (assumed counts-like)
    """
    if adata.raw is not None:
        ad = adata.raw.to_adata()
        return ad
    if "counts" in adata.layers:
        ad = adata.copy()
        ad.X = ad.layers["counts"]
        return ad
    return adata.copy()


def _make_tpm_like(counts_adata: sc.AnnData, target_sum: float = 1e6, log1p: bool = True) -> sc.AnnData:
    """
    Create a TPM/CPM-like matrix for cNMF from counts.
    """
    tpm = counts_adata.copy()
    sc.pp.normalize_total(tpm, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(tpm)
    return tpm


# -----------------------------
# Preprocessing
# -----------------------------


def preprocess_adata(
    adata_raw: sc.AnnData,
    timepoint_filter: str,
    tech_filter: Optional[str],
    gene_selection_pipeline: List[Dict[str, Any]],
    standardize: bool,
    n_top_genes: int,
    target_column: str = "CN.label",
    positive_class: str = "cancer",
    negative_class: str = "normal",
) -> Dict[str, Any]:
    preproc_config = create_preprocessing_config(
        n_top_genes=n_top_genes,
        standardize=standardize,
        timepoint_filter=timepoint_filter,
        target_column=target_column,
        positive_class=positive_class,
        negative_class=negative_class,
        tech_filter=tech_filter,
        gene_selection_pipeline=gene_selection_pipeline,
    )
    pipe = PreprocessingPipeline(preproc_config)
    return pipe.run_preprocessing(adata_raw)


# -----------------------------
# DR execution
# -----------------------------


def _run_dr_method(
    method: str,
    adata: sc.AnnData,
    k: int,
    seed: int,
    nmf_handle_negatives: str = "error",
    factosig_device: str = "cpu",
    factosig_lr: float = 1e-2,
    factosig_max_iter: int = 300,
    factosig_rotation: str = "varimax",
    factosig_order_factors_by: Optional[str] = "ss_loadings",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run a DR method on the provided AnnData and return (scores, loadings, extras).
    - scores: (n_cells, k)
    - loadings: (n_genes, k) in the *current* adata.var order
    """
    method_token = method.lower().strip()
    method_base = method_token.split("_", 1)[0]
    ad = adata.copy()

    if method_base == "pca":
        model = PCAWrapper()
        ad = model.fit_transform(ad, n_components=k, random_state=seed, standardize_input=False, svd_solver="randomized")
        scores = np.asarray(ad.obsm["X_pca"])
        loadings = np.asarray(ad.varm["PCA_loadings"])
        extras = {"uns": ad.uns.get("pca", {})}
        return scores, loadings, extras

    if method_base == "fa":
        model = FAWrapper()
        ad = model.fit_transform(ad, n_components=k, random_state=seed, standardize_input=False, svd_method="randomized", save_fitted_models=False)
        scores = np.asarray(ad.obsm["X_fa"])
        loadings = np.asarray(ad.varm["FA_loadings"])
        extras = {"uns": ad.uns.get("fa", {})}
        return scores, loadings, extras

    if method_base == "nmf":
        model = NMFWrapper()
        ad = model.fit_transform(
            ad,
            n_components=k,
            random_state=seed,
            save_model=False,
            save_dir=None,
            use_hvg=False,  # this runner controls gene sets explicitly
            standardize_input=False,
            handle_negative_values=nmf_handle_negatives,
        )
        scores = np.asarray(ad.obsm["X_nmf"])
        loadings = np.asarray(ad.varm["NMF_components"])
        extras = {"uns": ad.uns.get("nmf", {})}
        return scores, loadings, extras

    if method_base == "factosig":
        # FactoSig supports internal rotation and post-hoc factor ordering.
        # We allow method tokens like:
        # - factosig (defaults to varimax)
        # - factosig_varimax
        # - factosig_promax
        rot = (factosig_rotation or "varimax").lower()
        if method_token in ("factosig_varimax",):
            rot = "varimax"
        elif method_token in ("factosig_promax",):
            rot = "promax"
        model = FactoSigDR()
        ad = model.fit_transform(
            ad,
            n_components=k,
            random_state=seed,
            device=factosig_device,
            lr=factosig_lr,
            max_iter=factosig_max_iter,
            verbose=False,
            save_fitted_models=False,
            rotation=rot,
            order_factors_by=factosig_order_factors_by,
        )
        scores = np.asarray(ad.obsm["X_factosig"])
        loadings = np.asarray(ad.varm["FACTOSIG_loadings"])
        extras = {"uns": ad.uns.get("factosig", {})}
        return scores, loadings, extras

    raise ValueError(f"Unknown DR method: {method_token}")


# -----------------------------
# Stability / consensusness
# -----------------------------


def _corr_abs(a: np.ndarray, b: np.ndarray) -> float:
    # pearson correlation with sign-invariance
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Expected 1D vectors.")
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-12
    return float(abs(np.dot(a0, b0) / denom))


def match_components_by_loading_corr(loadings_a: np.ndarray, loadings_b: np.ndarray) -> Dict[str, Any]:
    """
    Greedy max-correlation matching between components in A vs B using |corr|.
    Returns match list + per-factor best correlations.
    """
    k = loadings_a.shape[1]
    corr = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            corr[i, j] = _corr_abs(loadings_a[:, i], loadings_b[:, j])

    # greedy matching
    used_i = set()
    used_j = set()
    matches: List[Tuple[int, int, float]] = []
    for _ in range(k):
        best = (-1, -1, -1.0)
        for i in range(k):
            if i in used_i:
                continue
            for j in range(k):
                if j in used_j:
                    continue
                if corr[i, j] > best[2]:
                    best = (i, j, float(corr[i, j]))
        if best[0] == -1:
            break
        used_i.add(best[0])
        used_j.add(best[1])
        matches.append(best)

    best_a = corr.max(axis=1).tolist()
    best_b = corr.max(axis=0).tolist()
    return {"corr_matrix": corr, "greedy_matches": matches, "best_a": best_a, "best_b": best_b}


def consensus_cluster_components(
    all_loadings: np.ndarray,  # (n_runs*K, n_genes)
    k: int,
    random_state: int = 1,
    outlier_max_sim_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Simple “consensusness” clustering for FA/FactoSig:
    - compute max similarity to any other component (|corr|)
    - filter components with max_sim < threshold
    - kmeans cluster into k clusters on L2-normalized loadings
    - consensus loading = per-cluster median (in original loading space)
    """
    n_comp, n_genes = all_loadings.shape
    # L2 normalize for clustering
    norms = np.linalg.norm(all_loadings, axis=1, keepdims=True) + 1e-12
    X = all_loadings / norms

    # similarity: max |corr| to others (approx via dot product since normalized + mean not centered)
    # We'll use pearson-style centering for robustness:
    Xc = X - X.mean(axis=1, keepdims=True)
    Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    sim = np.abs(Xc @ Xc.T)
    np.fill_diagonal(sim, 0.0)
    max_sim = sim.max(axis=1)
    keep = max_sim >= outlier_max_sim_threshold

    X_keep = X[keep]
    kept_idx = np.where(keep)[0]
    if X_keep.shape[0] < k:
        # not enough to cluster; fall back to using all
        X_keep = X
        kept_idx = np.arange(n_comp)
        keep = np.ones((n_comp,), dtype=bool)

    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X_keep)

    # consensus loadings (median) per cluster, mapped back to original space
    consensus = np.zeros((k, n_genes), dtype=np.float32)
    cluster_sizes: Dict[int, int] = {}
    for cl in range(k):
        members = kept_idx[labels == cl]
        cluster_sizes[int(cl)] = int(len(members))
        if len(members) == 0:
            continue
        consensus[cl, :] = np.median(all_loadings[members, :], axis=0)

    sil = None
    if X_keep.shape[0] >= k + 1 and len(set(labels)) > 1:
        sil = float(silhouette_score(X_keep, labels, metric="euclidean"))

    return {
        "keep_mask": keep.tolist(),
        "kept_n": int(np.sum(keep)),
        "n_components_total": int(n_comp),
        "outlier_max_sim_threshold": float(outlier_max_sim_threshold),
        "kmeans_labels_kept": labels.tolist(),
        "kept_indices": kept_idx.tolist(),
        "cluster_sizes": cluster_sizes,
        "silhouette": sil,
        "consensus_loadings": consensus,  # (k, genes)
    }


# -----------------------------
# Classification (grid-safe saving)
# -----------------------------


def run_lr_l1_per_patient(
    adata_with_scores: sc.AnnData,
    score_key: str,
    target_col: str = "CN.label",
    positive_class: str = "cancer",
    patient_col: str = "patient",
    cv_folds: int = 0,
    cv_repeats: int = 1,
    alphas: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    if alphas is None:
        alphas = np.logspace(-4, 5, 20)

    patients = sorted(adata_with_scores.obs[patient_col].unique())
    all_metrics: List[pd.DataFrame] = []
    all_coefs: List[pd.DataFrame] = []
    correctness_by_patient: Dict[str, pd.DataFrame] = {}

    for patient in patients:
        ad_p = adata_with_scores[adata_with_scores.obs[patient_col] == patient].copy()

        clf = LRLasso(adata=ad_p, target_col=target_col, target_value=positive_class, random_state=random_state)
        X, y, feature_names, _ = clf.prepare_data(use_factorized=True, factorization_method=score_key)

        if len(np.unique(y)) < 2:
            continue

        if cv_folds and cv_folds > 1:
            res = clf.fit_along_regularization_path_cv(
                X, y, feature_names, alphas=alphas, metrics_grouping=patient_col, cv_folds=cv_folds, cv_repeats=cv_repeats
            )
        else:
            res = clf.fit_along_regularization_path(X, y, feature_names, alphas=alphas, metrics_grouping=patient_col)

        metrics_df = pd.DataFrame(res["group_metrics_path"])
        coefs_df = pd.DataFrame(
            res["coefs"],
            index=feature_names,
            columns=[f"alpha_{a:.2e}" for a in alphas],
        )
        correctness_df = res["correctness_df"]
        all_metrics.append(metrics_df.assign(patient=patient))
        all_coefs.append(coefs_df.assign(feature=coefs_df.index, patient=patient))
        correctness_by_patient[str(patient)] = correctness_df

    return {
        "metrics": pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame(),
        "coefs": pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame(),
        "correctness_by_patient": correctness_by_patient,
    }


def save_grid_classification(
    experiment_dir: Path,
    dr_method: str,
    out: Dict[str, Any],
) -> None:
    base = _ensure_dir(experiment_dir / "analysis" / "classification_grid" / dr_method)
    metrics = out["metrics"]
    coefs = out["coefs"]
    metrics.to_csv(base / "metrics.csv", index=False)
    coefs.to_csv(base / "coefs.csv", index=False)
    # correctness per patient can be large; save one file per patient
    corr_dir = _ensure_dir(base / "correctness")
    for patient, df in out["correctness_by_patient"].items():
        df.to_csv(corr_dir / f"{patient}_classification_correctness.csv")


# -----------------------------
# Plan 0 runner
# -----------------------------


def plan0(
    input_h5ad: str,
    experiments_dir: str,
    timepoint_filter: str,
    tech_filter: Optional[str],
    reference_hvg: int,
    ks: List[int],
    seeds: List[int],
    methods: List[str],
    cnmf_n_iter: int,
    cnmf_dt: str,
    factosig_rotation: str = "varimax",
    factosig_order_factors_by: Optional[str] = "ss_loadings",
) -> str:
    """
    Creates one experiment directory and stores:
    - preprocessing (reference HVG)
    - per-method/K single-run diagnostics
    - stability screen across seeds
    - cNMF k-selection outputs if cnmf is available
    """
    config = {
        "stage": "plan0",
        "preprocessing": {
            "timepoint_filter": timepoint_filter,
            "tech_filter": tech_filter,
            "gene_selection_pipeline": [{"method": "hvg", "n_top_genes": int(reference_hvg)}],
            "standardize": True,
        },
        "plan0": {
            "ks": ks,
            "seeds": seeds,
            "methods": methods,
            "cnmf_n_iter": int(cnmf_n_iter),
            "cnmf_dt": str(cnmf_dt),
            "factosig_rotation": str(factosig_rotation),
            "factosig_order_factors_by": factosig_order_factors_by,
        },
        "dimension_reduction": {"method": "plan0_k_sweep", "n_components": int(max(ks))},
        "classification": {"cv_folds": 0},
        "downsampling": {"method": "none"},
    }
    em = ExperimentManager(experiments_dir)
    exp = em.create_experiment(ExperimentConfig(config))
    print(f"[plan0] Experiment directory: {exp.experiment_dir}")

    adata_raw = _ensure_timepoint_type(_load_adata(input_h5ad))

    # Two views:
    # - standardized (PCA/FA/FactoSig)
    # - non-standardized counts-like (NMF/cNMF)
    pre_std = preprocess_adata(
        adata_raw=adata_raw,
        timepoint_filter=timepoint_filter,
        tech_filter=tech_filter,
        gene_selection_pipeline=[{"method": "hvg", "n_top_genes": int(reference_hvg)}],
        standardize=True,
        n_top_genes=int(reference_hvg),
    )
    pre_nstd = preprocess_adata(
        adata_raw=adata_raw,
        timepoint_filter=timepoint_filter,
        tech_filter=tech_filter,
        gene_selection_pipeline=[{"method": "hvg", "n_top_genes": int(reference_hvg)}],
        standardize=False,
        n_top_genes=int(reference_hvg),
    )
    print(f"[plan0] Preprocessing complete. std_shape={pre_std['adata'].shape}, nstd_shape={pre_nstd['adata'].shape}")
    print(f"[plan0] Saving preprocessing outputs under: {exp.experiment_dir / 'preprocessing'}")
    t_save = time.time()
    exp.save_preprocessing_results(
        pre_std["adata"],
        pre_std["hvg_list"],
        pre_std["scaler"],
        {"summary_text": pre_std["summary"], "gene_log": pre_std["info"].get("gene_log", {})},
    )
    print(f"[plan0] Saved preprocessing outputs (elapsed {time.time() - t_save:.1f}s)")

    # single-run diagnostics + replicate caches
    diag_dir = _ensure_dir(exp.experiment_dir / "analysis" / "plan0")
    _write_json(diag_dir / "config.json", config)

    # Run methods in the provided order (supports: fa, factosig, cnmf, pca, nmf)
    for method in methods:
        method_l = method.lower().strip()

        # --- cNMF K selection / stability (optional dependency) ---
        if method_l == "cnmf":
            try:
                from cnmf import cNMF  # type: ignore
            except Exception as e:
                _write_json(diag_dir / "cnmf_missing.json", {"error": str(e)})
                continue

            print("[plan0] Running cNMF pipeline ...")
            cnmf_dir = _ensure_dir(exp.experiment_dir / "models" / "cnmf_plan0")
            counts = _extract_counts_for_cnmf(pre_nstd["adata"])
            tpm = _make_tpm_like(counts, target_sum=1e6, log1p=True)

            cn = cNMF(output_dir=str(cnmf_dir), name="plan0_cnmf")
            norm_counts = cn.get_norm_counts(counts=counts, tpm=tpm, num_highvar_genes=int(reference_hvg))
            cn.save_norm_counts(norm_counts)
            # cnmf consensus() expects TPM on disk (paths['tpm'])
            try:
                tpm.write_h5ad(cn.paths["tpm"])
            except Exception:
                # Best-effort: continue; resume script can salvage by linking to norm_counts if needed
                pass
            rep_params, nmf_kwargs = cn.get_nmf_iter_params(ks=ks, n_iter=int(cnmf_n_iter), random_state_seed=1)
            cn.save_nmf_iter_params(rep_params, nmf_kwargs)
            # cnmf API compatibility:
            # - cnmf>=1.7 uses `factorize(...)`
            # - older versions used `run_nmf(...)`
            if hasattr(cn, "factorize"):
                # worker_i is 0-indexed when total_workers=1
                cn.factorize(worker_i=0, total_workers=1)
            else:
                cn.run_nmf(worker_i=0, total_workers=1)
            for k in ks:
                print(f"[plan0] cNMF combine/consensus k={k} ...")
                cn.combine_nmf(k)
                # cnmf API compatibility: argument name changed across versions
                import inspect

                sig = inspect.signature(cn.consensus)
                kwargs: Dict[str, Any] = {}
                if "density_threshold_str" in sig.parameters:
                    kwargs["density_threshold_str"] = str(cnmf_dt)
                elif "density_threshold" in sig.parameters:
                    kwargs["density_threshold"] = float(cnmf_dt)
                elif "dt" in sig.parameters:
                    kwargs["dt"] = float(cnmf_dt)
                if "show_clustering" in sig.parameters:
                    kwargs["show_clustering"] = True
                stats = cn.consensus(k, **kwargs)
                try:
                    stats_obj = stats.to_dict()  # type: ignore
                except Exception:
                    stats_obj = {"stats_repr": str(stats)}
                _write_json(diag_dir / "cnmf" / f"k_{k}" / "consensus_stats.json", stats_obj)
            cn.k_selection_plot(close_fig=True)
            print("[plan0] cNMF done.")
            continue

        # --- Standard DR methods: cache per (method,k,seed) and allow reruns to reuse cache ---
        for k in ks:
            method_dir = _ensure_dir(diag_dir / "stability" / method_l / f"k_{k}")
            loadings_runs: List[np.ndarray] = []

            for seed in seeds:
                rep_dir = _ensure_dir(method_dir / "replicates" / f"seed_{seed}")
                scores_path = rep_dir / "scores.npy"
                loadings_path = rep_dir / "loadings.npy"
                extras_path = rep_dir / "extras.json"

                if scores_path.exists() and loadings_path.exists() and extras_path.exists():
                    # reuse cached outputs
                    loadings = np.load(loadings_path)
                    loadings_runs.append(loadings)
                    print(f"[plan0] Cached DR={method_l} k={k} seed={seed} found; skipping fit.")
                    continue

                print(f"[plan0] Running DR={method_l} k={k} seed={seed} ...")
                t0 = time.time()
                ad_for_method = pre_nstd["adata"] if method_l == "nmf" else pre_std["adata"]
                if method_l.startswith("factosig"):
                    scores, loadings, extras = _run_dr_method(
                        method_l,
                        ad_for_method,
                        k=k,
                        seed=seed,
                        factosig_rotation=str(factosig_rotation),
                        factosig_order_factors_by=factosig_order_factors_by,
                    )
                else:
                    scores, loadings, extras = _run_dr_method(method_l, ad_for_method, k=k, seed=seed)
                print(f"[plan0] Done DR={method_l} k={k} seed={seed} (elapsed {time.time() - t0:.1f}s)")
                loadings_runs.append(loadings)
                np.save(scores_path, scores.astype(np.float32, copy=False))
                np.save(loadings_path, loadings.astype(np.float32, copy=False))
                _write_json(extras_path, extras.get("uns", {}))

            # pairwise stability (adjacent pairs)
            pair_stats: List[Dict[str, Any]] = []
            for i in range(len(loadings_runs) - 1):
                m = match_components_by_loading_corr(loadings_runs[i], loadings_runs[i + 1])
                pair_stats.append(
                    {
                        "pair": [int(i), int(i + 1)],
                        "best_a_median": float(np.median(m["best_a"])) if m["best_a"] else None,
                        "best_a_mean": float(np.mean(m["best_a"])) if m["best_a"] else None,
                        "best_a_frac_lt_0p3": float(np.mean(np.array(m["best_a"]) < 0.3)) if m["best_a"] else None,
                    }
                )
            _write_json(method_dir / "pairwise_stability_summary.json", {"pairs": pair_stats})

            # consensusness clustering for FA/FactoSig only (diagnostic cache)
            if method_l in ("fa", "factosig") and len(loadings_runs) >= 2:
                stacked = np.concatenate([L.T for L in loadings_runs], axis=0)  # (runs*k, genes)
                cons = consensus_cluster_components(stacked, k=k, random_state=1, outlier_max_sim_threshold=0.2)
                cons_dir = _ensure_dir(method_dir / "consensus_cache")
                np.save(cons_dir / "consensus_loadings.npy", cons["consensus_loadings"])
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

    return exp.experiment_dir.name


# -----------------------------
# Plan 1 runner (no-CV or CV)
# -----------------------------


def plan1(
    input_h5ad: str,
    experiments_dir: str,
    timepoint_filter: str,
    tech_filter: Optional[str],
    preprocess_methods: List[Dict[str, Any]],
    dr_methods: List[str],
    k_by_method: Dict[str, int],
    cv_folds: int,
    cv_repeats: int,
    cnmf_dt: str,
    cnmf_n_iter: int,
    factosig_rotation: str = "varimax",
    factosig_order_factors_by: Optional[str] = "ss_loadings",
) -> str:
    config = {
        "stage": "plan1",
        "preprocessing": {
            "timepoint_filter": timepoint_filter,
            "tech_filter": tech_filter,
            "preprocess_methods": preprocess_methods,
        },
        "dimension_reduction": {"methods": dr_methods, "k_by_method": k_by_method},
        "classification": {"cv_folds": int(cv_folds), "cv_repeats": int(cv_repeats)},
        "downsampling": {"method": "none"},
        "cnmf": {"dt": str(cnmf_dt), "n_iter": int(cnmf_n_iter)},
        "factosig": {"rotation": str(factosig_rotation), "order_factors_by": factosig_order_factors_by},
    }
    em = ExperimentManager(experiments_dir)
    exp = em.create_experiment(ExperimentConfig(config))
    exp_dir = exp.experiment_dir
    _write_json(exp_dir / "analysis" / "plan1_config.json", config)

    adata_raw = _ensure_timepoint_type(_load_adata(input_h5ad))

    # For each preprocess method, create one combined adata with all DR embeddings.
    for pm in preprocess_methods:
        tag = _readable_tag(pm)
        gene_pipeline = [pm]

        # Standardized view for PCA/FA/FactoSig
        pre_std = preprocess_adata(
            adata_raw=adata_raw,
            timepoint_filter=timepoint_filter,
            tech_filter=tech_filter,
            gene_selection_pipeline=gene_pipeline,
            standardize=True,
            n_top_genes=int(pm.get("n_top_genes", 3000) if pm.get("method") == "hvg" else 3000),
        )
        # Non-standardized view for NMF/cNMF
        pre_nstd = preprocess_adata(
            adata_raw=adata_raw,
            timepoint_filter=timepoint_filter,
            tech_filter=tech_filter,
            gene_selection_pipeline=gene_pipeline,
            standardize=False,
            n_top_genes=int(pm.get("n_top_genes", 3000) if pm.get("method") == "hvg" else 3000),
        )
        ad_combined = pre_std["adata"]

        # Write preprocessing cache (per preprocess method)
        out_pre_dir = _ensure_dir(exp_dir / "analysis" / "preprocess_cache" / tag)
        ad_combined.write_h5ad(out_pre_dir / "adata_preprocessed.h5ad")
        _write_json(out_pre_dir / "gene_log.json", pre_std["info"].get("gene_log", {}))

        # Run DR methods (selected K per method)
        for m in dr_methods:
            m_l = m.lower()
            if m_l == "cnmf":
                # cNMF is run on counts-like input, stored under models/cnmf/<tag>/
                try:
                    from cnmf import cNMF, cnmf_load_results  # type: ignore
                except Exception as e:
                    _write_json(out_pre_dir / "cnmf_missing.json", {"error": str(e)})
                    continue

                k = int(k_by_method.get("cnmf", 60))
                cnmf_dir = _ensure_dir(exp_dir / "models" / "cnmf" / tag)
                counts = _extract_counts_for_cnmf(pre_nstd["adata"])
                tpm = _make_tpm_like(counts, target_sum=1e6, log1p=True)

                cn = cNMF(output_dir=str(cnmf_dir), name=f"cnmf_{tag}")
                norm_counts = cn.get_norm_counts(counts=counts, tpm=tpm, num_highvar_genes=counts.n_vars)
                cn.save_norm_counts(norm_counts)
                try:
                    tpm.write_h5ad(cn.paths["tpm"])
                except Exception:
                    pass
                rep_params, nmf_kwargs = cn.get_nmf_iter_params(ks=[k], n_iter=int(cnmf_n_iter), random_state_seed=1)
                cn.save_nmf_iter_params(rep_params, nmf_kwargs)
                if hasattr(cn, "factorize"):
                    cn.factorize(worker_i=0, total_workers=1)
                else:
                    cn.run_nmf(worker_i=0, total_workers=1)
                cn.combine_nmf(k)
                import inspect

                sig = inspect.signature(cn.consensus)
                kwargs: Dict[str, Any] = {}
                if "density_threshold_str" in sig.parameters:
                    kwargs["density_threshold_str"] = str(cnmf_dt)
                elif "density_threshold" in sig.parameters:
                    kwargs["density_threshold"] = float(cnmf_dt)
                elif "dt" in sig.parameters:
                    kwargs["dt"] = float(cnmf_dt)
                if "show_clustering" in sig.parameters:
                    kwargs["show_clustering"] = True
                cn.consensus(k, **kwargs)

                # Attach consensus outputs back to AnnData
                # NOTE: cnmf_load_results edits adata in place and adds:
                # - obsm["cnmf_usages"], varm["cnmf_spectra"], etc.
                cnmf_load_results(ad_combined, cnmf_dir=str(cnmf_dir), name=f"cnmf_{tag}", k=k, dt=str(cnmf_dt), key="cnmf")
                # Normalize to our convention: store a score matrix at obsm["X_cnmf"]
                if "cnmf_usages" in ad_combined.obsm:
                    ad_combined.obsm["X_cnmf"] = np.asarray(ad_combined.obsm["cnmf_usages"])
                continue

            k = int(k_by_method.get(m_l, 60))
            # Choose the appropriate input view for each method
            if m_l == "nmf":
                ad_for_method = pre_nstd["adata"]
            else:
                ad_for_method = ad_combined
            if m_l.startswith("factosig"):
                scores, loadings, extras = _run_dr_method(
                    m_l,
                    ad_for_method,
                    k=k,
                    seed=42,
                    factosig_rotation=str(factosig_rotation),
                    factosig_order_factors_by=factosig_order_factors_by,
                )
            else:
                scores, loadings, extras = _run_dr_method(m_l, ad_for_method, k=k, seed=42)
            # Attach to combined AnnData
            ad_combined.obsm[f"X_{m_l}"] = scores.astype(np.float32, copy=False)
            ad_combined.varm[f"{m_l.upper()}_loadings"] = loadings.astype(np.float32, copy=False)
            ad_combined.uns[f"{m_l}_meta"] = extras.get("uns", {})

            # Also cache arrays-only to experiment models dir
            model_dir = _ensure_dir(exp_dir / "models" / f"{m_l}_{k}" / tag)
            np.save(model_dir / "scores.npy", scores.astype(np.float32, copy=False))
            np.save(model_dir / "loadings.npy", loadings.astype(np.float32, copy=False))
            with open(model_dir / "obs_names.txt", "w") as f:
                f.write("\n".join(list(ad_combined.obs_names)) + "\n")
            with open(model_dir / "var_names.txt", "w") as f:
                f.write("\n".join(list(ad_combined.var_names)) + "\n")
            _write_json(model_dir / "extras.json", extras.get("uns", {}))

        # Save combined AnnData (the main future-proof artifact)
        combined_path = out_pre_dir / "adata_with_dr.h5ad"
        ad_combined.write_h5ad(combined_path)

        # Run classification per DR method and save grid-safe results
        for m in dr_methods:
            m_l = m.lower()
            score_key = "X_cnmf" if m_l == "cnmf" else f"X_{m_l}"
            if score_key not in ad_combined.obsm:
                continue
            out = run_lr_l1_per_patient(
                adata_with_scores=ad_combined,
                score_key=score_key,
                cv_folds=int(cv_folds),
                cv_repeats=int(cv_repeats),
            )
            save_grid_classification(out_pre_dir, dr_method=m_l, out=out)

    return exp_dir.name


def _parse_k_by_method(s: str) -> Dict[str, int]:
    """
    Parse a string like: 'pca=60,fa=60,nmf=40,factosig=60,cnmf=40'
    """
    out: Dict[str, int] = {}
    if not s.strip():
        return out
    for part in s.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        out[k.strip().lower()] = int(v)
    return out


def main():
    p = argparse.ArgumentParser(description="Plan 0/1 runners for gene filtering eval.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("plan0", help="Run Plan 0 (K sweep + stability + optional cNMF K selection).")
    p0.add_argument("--input-h5ad", required=True)
    p0.add_argument("--experiments-dir", default="experiments")
    p0.add_argument("--timepoint-filter", default="MRD")
    p0.add_argument("--tech-filter", default="CITE")
    p0.add_argument("--reference-hvg", type=int, default=10000)
    # Accept both "--ks 20,40,60" and "--ks 20, 40, 60"
    p0.add_argument("--ks", nargs="+", default=["20,40,60"])
    # Default to one seed for fast K screening; you can rerun later with more seeds
    p0.add_argument("--seeds", nargs="+", default=["1"])
    # Default order: FA -> FactoSig -> cNMF -> PCA
    p0.add_argument("--methods", nargs="+", default=["fa,factosig,cnmf,pca"])
    p0.add_argument("--cnmf-n-iter", type=int, default=20)
    p0.add_argument("--cnmf-dt", default="0.5")
    p0.add_argument(
        "--factosig-rotation",
        default="varimax",
        help="Default FactoSig rotation (overridden by method tokens factosig_varimax/factosig_promax).",
    )
    p0.add_argument(
        "--factosig-order-factors-by",
        default="ss_loadings",
        help="FactoSig factor ordering: ss_loadings | score_variance | none",
    )

    p1 = sub.add_parser("plan1", help="Run Plan 1 (preprocess × DR grid; no-CV or CV).")
    p1.add_argument("--input-h5ad", required=True)
    p1.add_argument("--experiments-dir", default="experiments")
    p1.add_argument("--timepoint-filter", default="MRD")
    p1.add_argument("--tech-filter", default="CITE")
    p1.add_argument("--cv-folds", type=int, default=0)
    p1.add_argument("--cv-repeats", type=int, default=10)
    p1.add_argument("--dr-methods", default="pca,fa,nmf,factosig,cnmf")
    p1.add_argument("--k-by-method", default="pca=60,fa=60,nmf=60,factosig=60,cnmf=60")
    p1.add_argument("--cnmf-n-iter", type=int, default=20)
    p1.add_argument("--cnmf-dt", default="0.5")
    p1.add_argument(
        "--factosig-rotation",
        default="varimax",
        help="Default FactoSig rotation (overridden by method tokens factosig_varimax/factosig_promax).",
    )
    p1.add_argument(
        "--factosig-order-factors-by",
        default="ss_loadings",
        help="FactoSig factor ordering: ss_loadings | score_variance | none",
    )

    # Preprocess method definitions as JSON-ish strings (kept simple for now)
    p1.add_argument(
        "--preprocess-set",
        default="hvg,all_filtered,deg_weak_screen,hybrid",
        help="Comma-separated preprocess methods: hvg, all_filtered, deg_weak_screen, hybrid",
    )
    p1.add_argument("--hvg-n", type=int, default=3000)
    p1.add_argument("--allfiltered-min-frac", type=float, default=0.01)
    p1.add_argument("--allfiltered-ratio", type=float, default=20.0)
    p1.add_argument("--deg-p", type=float, default=0.1)
    p1.add_argument("--deg-lfc", type=float, default=0.05)

    args = p.parse_args()

    if args.cmd == "plan0":
        ks = [int(x) for x in _split_list_arg(args.ks)]
        seeds = [int(x) for x in _split_list_arg(args.seeds)]
        methods = [m.strip() for m in _split_list_arg(args.methods)]
        exp_id = plan0(
            input_h5ad=args.input_h5ad,
            experiments_dir=args.experiments_dir,
            timepoint_filter=args.timepoint_filter,
            tech_filter=args.tech_filter,
            reference_hvg=args.reference_hvg,
            ks=ks,
            seeds=seeds,
            methods=methods,
            cnmf_n_iter=args.cnmf_n_iter,
            cnmf_dt=args.cnmf_dt,
            factosig_rotation=str(args.factosig_rotation),
            factosig_order_factors_by=(None if str(args.factosig_order_factors_by).lower() == "none" else str(args.factosig_order_factors_by)),
        )
        print(exp_id)
        return

    if args.cmd == "plan1":
        dr_methods = [m.strip() for m in args.dr_methods.split(",") if m.strip()]
        k_by = _parse_k_by_method(args.k_by_method)
        pre = []
        for m in [x.strip() for x in args.preprocess_set.split(",") if x.strip()]:
            if m == "hvg":
                pre.append({"method": "hvg", "n_top_genes": int(args.hvg_n)})
            elif m == "all_filtered":
                pre.append(
                    {
                        "method": "all_filtered",
                        "min_cells_fraction": float(args.allfiltered_min_frac),
                        "malignant_enrichment_ratio": float(args.allfiltered_ratio),
                    }
                )
            elif m == "deg_weak_screen":
                pre.append(
                    {
                        "method": "deg_weak_screen",
                        "deg_test_method": "wilcoxon",
                        "use_adj_pvals": True,
                        "pval_threshold": float(args.deg_p),
                        "lfc_threshold": float(args.deg_lfc),
                        "min_n_genes": None,
                    }
                )
            elif m == "hybrid":
                pre.append(
                    {
                        "method": "hvg_plus_rescue_union",
                        "n_top_genes": int(args.hvg_n),
                        "min_cells_fraction": float(args.allfiltered_min_frac),
                        "malignant_enrichment_ratio": float(args.allfiltered_ratio),
                    }
                )
            else:
                raise ValueError(f"Unknown preprocess method token: {m}")

        exp_id = plan1(
            input_h5ad=args.input_h5ad,
            experiments_dir=args.experiments_dir,
            timepoint_filter=args.timepoint_filter,
            tech_filter=args.tech_filter,
            preprocess_methods=pre,
            dr_methods=dr_methods,
            k_by_method=k_by,
            cv_folds=int(args.cv_folds),
            cv_repeats=int(args.cv_repeats),
            cnmf_dt=str(args.cnmf_dt),
            cnmf_n_iter=int(args.cnmf_n_iter),
            factosig_rotation=str(args.factosig_rotation),
            factosig_order_factors_by=(None if str(args.factosig_order_factors_by).lower() == "none" else str(args.factosig_order_factors_by)),
        )
        print(exp_id)
        return


if __name__ == "__main__":
    main()

