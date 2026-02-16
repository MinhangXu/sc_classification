#!/usr/bin/env python
"""
Compare dimensionality reduction results between sklearn FactorAnalysis and FactoSig
on the same standardized AnnData matrix, and log outputs with ExperimentManager.

Usage:
  python scripts/compare_dr_factosig_vs_sklearn.py \
    --input-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_standardized_oct7.h5ad \
    --n-components 100 \
    --max-genes 3000 \
    --seed 42 \
    --svd-method randomized \
    --device cpu

Notes:
  - This script only performs DR and caching; it does NOT run supervised classification.
  - To manage memory, set --max-genes to a reasonable value (e.g., 3000 or 6000).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import scanpy as sc

from sklearn.decomposition import FactorAnalysis as SklearnFA

from sc_classification.utils.experiment_manager import ExperimentManager, ExperimentConfig


def _load_adata(h5ad_path: str) -> sc.AnnData:
    adata = sc.read_h5ad(h5ad_path)
    return adata


def _to_numpy(X) -> np.ndarray:
    return X.A if hasattr(X, "A") else np.asarray(X)


def _select_top_var_genes(X: np.ndarray, genes: List[str], max_genes: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if max_genes is None or max_genes <= 0 or X.shape[1] <= max_genes:
        return X, genes, np.arange(X.shape[1], dtype=int)
    vars_ = np.var(X, axis=0)
    idx = np.argpartition(vars_, -max_genes)[-max_genes:]
    idx.sort()
    genes_sel = [genes[i] for i in idx]
    return X[:, idx], genes_sel, idx


def _summarize_dr(loadings: np.ndarray, psi: np.ndarray) -> dict:
    ss_per_factor = (loadings ** 2).sum(axis=0)
    communality = (loadings ** 2).sum(axis=1)
    total_ss = float(ss_per_factor.sum())
    top10_share = float(np.sort(ss_per_factor)[-10:].sum() / (total_ss if total_ss > 0 else 1.0))
    fve = communality / (communality + np.asarray(psi))
    fve = np.clip(fve, 0.0, 1.0)
    q25, q50, q75 = np.quantile(fve, [0.25, 0.5, 0.75])
    return {
        "total_ss": total_ss,
        "top10_ss_share": top10_share,
        "median_fve": float(q50),
        "iqr_fve": [float(q25), float(q75)],
    }


def _save_metrics_json(exp_dir: Path, method: str, n_components: int, metrics: dict) -> None:
    out_dir = exp_dir / "models" / f"{method}_{n_components}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "dr_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)


def run_sklearn_fa(adata_in: sc.AnnData, X: np.ndarray, genes: List[str], n_components: int, seed: int, svd_method: str):
    fa = SklearnFA(n_components=n_components, random_state=seed, svd_method=svd_method)
    scores = fa.fit_transform(X)  # (n, k)
    loadings = fa.components_.T    # (p, k)
    psi = fa.noise_variance_       # (p,)

    adata_out = adata_in.copy()
    adata_out.obsm["X_sklearn_fa"] = np.asarray(scores, dtype=np.float32)
    adata_out.varm["sklearn_fa_loadings"] = np.asarray(loadings, dtype=np.float32)
    adata_out.var["sklearn_fa_psi"] = np.asarray(psi, dtype=np.float32)

    return fa, adata_out, loadings, psi


def run_factosig(adata_in: sc.AnnData, X: np.ndarray, n_components: int, seed: int, device: str, lr: float, max_iter: int):
    from factosig import FactoSig

    fs = FactoSig(
        n_factors=n_components,
        device=device,
        random_state=seed,
        lr=lr,
        max_iter=max_iter,
        verbose=True,
    )
    fs.fit(X)

    # FactoSig applies varimax rotation and sign convention internally
    loadings = np.asarray(fs.loadings_)
    scores = np.asarray(fs.scores_)
    psi = np.asarray(fs.psi_)

    adata_out = adata_in.copy()
    adata_out.obsm["fs_scores"] = np.asarray(scores, dtype=np.float32)
    adata_out.varm["fs_loadings"] = np.asarray(loadings, dtype=np.float32)
    if psi is not None and psi.size == adata_out.n_vars:
        adata_out.var["fs_psi"] = np.asarray(psi, dtype=np.float32)

    return fs, adata_out, loadings, psi


def main():
    parser = argparse.ArgumentParser(description="Compare DR: sklearn FA vs FactoSig (DR only, no classification)")
    parser.add_argument("--input-h5ad", required=True, help="Path to standardized .h5ad (cells × genes)")
    parser.add_argument("--n-components", type=int, default=100, help="Number of factors/components (k)")
    parser.add_argument("--max-genes", type=int, default=0, help="Cap genes by top variance (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--svd-method", default="randomized", choices=["lapack", "randomized"], help="sklearn FA SVD backend")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="FactoSig device")
    parser.add_argument("--lr", type=float, default=1e-2, help="FactoSig Adam learning rate")
    parser.add_argument("--max-iter", type=int, default=300, help="FactoSig max iterations")
    parser.add_argument("--experiments-dir", default="experiments", help="Base directory for ExperimentManager")
    args = parser.parse_args()

    # Load data
    adata = _load_adata(args.input_h5ad)
    X_full = _to_numpy(adata.X).astype(np.float32, copy=False)
    genes_full = list(adata.var_names)
    n, p = X_full.shape
    print(f"[compare-dr] Loaded matrix: n={n}, p={p}, dtype={X_full.dtype}")

    # Gene cap by variance (shared for both methods)
    X, genes_sel, idx_sel = _select_top_var_genes(X_full, genes_full, args.max_genes)
    if X.shape[1] != p:
        print(f"[compare-dr] Gene cap applied: p -> {X.shape[1]}")
    adata_sel = adata[:, genes_sel].copy()

    # Create experiment
    config = {
        'preprocessing': {
            'used_input_h5ad': args.input_h5ad,
            'max_genes': int(args.max_genes),
        },
        'dimension_reduction': {
            'method': 'compare_fa',
            'n_components': int(args.n_components),
        },
        'compare': {
            'methods': ['sklearn_fa', 'factosig'],
            'seed': int(args.seed),
            'svd_method': args.svd_method,
            'device': args.device,
            'lr': float(args.lr),
            'max_iter': int(args.max_iter),
        }
    }
    exp_cfg = ExperimentConfig(config)
    em = ExperimentManager(args.experiments_dir)
    exp = em.create_experiment(exp_cfg)
    print(f"[compare-dr] Experiment directory: {exp.experiment_dir}")

    # --- sklearn FA ---
    print("[compare-dr] Running sklearn FactorAnalysis...")
    fa_model, adata_fa, L_skl, psi_skl = run_sklearn_fa(
        adata_in=adata_sel,
        X=X,
        genes=genes_sel,
        n_components=args.n_components,
        seed=args.seed,
        svd_method=args.svd_method,
    )
    summ_skl = _summarize_dr(L_skl, psi_skl)
    summ_skl.update({
        'n_cells': int(X.shape[0]),
        'n_genes': int(X.shape[1]),
        'k': int(args.n_components),
        'method': 'sklearn_fa',
        'svd_method': args.svd_method,
        'seed': int(args.seed),
    })
    skl_summary_text = (
        f"sklearn FA (n={X.shape[0]}, p={X.shape[1]}, k={args.n_components})\n"
        f"top10_ss_share={summ_skl['top10_ss_share']:.3f}, median_FVE={summ_skl['median_fve']:.3f}, "
        f"IQR_FVE=({summ_skl['iqr_fve'][0]:.3f}, {summ_skl['iqr_fve'][1]:.3f})\n"
    )
    exp.save_dr_results(
        model=fa_model,
        transformed_adata=adata_fa,
        dr_method='sklearn_fa',
        n_components=args.n_components,
        summary=skl_summary_text,
    )
    _save_metrics_json(exp.experiment_dir, 'sklearn_fa', args.n_components, summ_skl)

    # --- FactoSig ---
    print("[compare-dr] Running FactoSig...")
    fs_model, adata_fs, L_fs, psi_fs = run_factosig(
        adata_in=adata_sel,
        X=X,
        n_components=args.n_components,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        max_iter=args.max_iter,
    )
    if psi_fs is None or psi_fs.shape[0] != X.shape[1]:
        # Fallback: estimate psi via residual variance if not available (rare)
        # X ≈ Z @ L^T; residual variance per gene
        Z = np.asarray(fs_model.scores_)
        R = X - Z @ np.asarray(fs_model.loadings_).T
        dof = max(X.shape[0] - args.n_components, 1)
        psi_fs = (R ** 2).sum(axis=0) / float(dof)

    summ_fs = _summarize_dr(L_fs, psi_fs)
    summ_fs.update({
        'n_cells': int(X.shape[0]),
        'n_genes': int(X.shape[1]),
        'k': int(args.n_components),
        'method': 'factosig',
        'device': args.device,
        'seed': int(args.seed),
        'rotation': 'varimax',
    })
    fs_summary_text = (
        f"FactoSig (n={X.shape[0]}, p={X.shape[1]}, k={args.n_components})\n"
        f"top10_ss_share={summ_fs['top10_ss_share']:.3f}, median_FVE={summ_fs['median_fve']:.3f}, "
        f"IQR_FVE=({summ_fs['iqr_fve'][0]:.3f}, {summ_fs['iqr_fve'][1]:.3f})\n"
    )
    exp.save_dr_results(
        model=fs_model,
        transformed_adata=adata_fs,
        dr_method='factosig',
        n_components=args.n_components,
        summary=fs_summary_text,
    )
    _save_metrics_json(exp.experiment_dir, 'factosig', args.n_components, summ_fs)

    print("[compare-dr] Completed. Models and transformed data saved under:")
    print(f"  - {exp.experiment_dir / 'models' / f'sklearn_fa_{args.n_components}'}")
    print(f"  - {exp.experiment_dir / 'models' / f'factosig_{args.n_components}'}")


if __name__ == "__main__":
    main()


