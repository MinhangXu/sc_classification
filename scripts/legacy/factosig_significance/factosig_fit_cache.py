#!/usr/bin/env python
"""
Fit FactoSig on an AnnData matrix with low memory usage, cache results for later bootstrap.

Usage:
  python factosig_fit_cache.py \
    --input-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_standardized_oct7.h5ad \
    --output-npz /home/minhang/mds_project/data/cohort_adata/factosig_fit_k100_oct7.npz \
    --k 100 \
    --device cpu \
    --max-genes 6000 \
    --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _load_adata_X(h5ad_path: str) -> tuple[np.ndarray, list[str]]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X.A if hasattr(adata.X, "A") else adata.X
    X = np.asarray(X, dtype=np.float32)
    genes = adata.var_names.tolist()
    return X, genes



def _select_top_var_genes(X: np.ndarray, genes: list[str], max_genes: int) -> tuple[np.ndarray, list[str]]:
    '''
    Select the top max_genes genes by variance.
    '''
    if max_genes is None or max_genes <= 0 or X.shape[1] <= max_genes:
        return X, genes
    vars_ = np.var(X, axis=0)
    idx = np.argpartition(vars_, -max_genes)[-max_genes:]
    idx.sort()
    return X[:, idx], [genes[i] for i in idx]


def main():
    parser = argparse.ArgumentParser(description="Fit FactoSig and cache results for later bootstrap.")
    parser.add_argument("--input-h5ad", required=True, help="Path to standardized .h5ad (cells × genes)")
    parser.add_argument("--output-npz", required=True, help="Path to save cached fit arrays (.npz)")
    parser.add_argument("--k", type=int, required=True, help="Number of factors")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device")
    parser.add_argument("--max-genes", type=int, default=0, help="Cap genes by top variance (0 to disable)")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Adam optimizer")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum optimization iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t0 = time.time()
    X, genes = _load_adata_X(args.input_h5ad)
    print(f"[factosig-fit] Loaded matrix: n={X.shape[0]}, p={X.shape[1]}, dtype={X.dtype}")
    approx_mem_mb = X.size * X.dtype.itemsize / (1024**2)
    print(f"[factosig-fit] Approx host RAM for X: {approx_mem_mb:.1f} MB")
    X, genes = _select_top_var_genes(X, genes, args.max_genes)
    if args.max_genes and args.max_genes > 0:
        print(f"[factosig-fit] Gene cap applied: p -> {X.shape[1]}")
    print(f"[factosig-fit] Config: k={args.k}, device={args.device}, lr={args.lr}, max_iter={args.max_iter}, seed={args.seed}")

    # Import here to avoid importing torch when only inspecting
    from factosig import FactoSig

    fs = FactoSig(
        n_factors=args.k,
        device=args.device,
        random_state=args.seed,
        max_iter=args.max_iter,
        lr=args.lr,
        verbose=True,
    )
    t_fit0 = time.time()
    fs.fit(X)
    t_fit1 = time.time()
    print(f"[factosig-fit] Fit wall time: {t_fit1 - t_fit0:.1f}s")
    print(f"[factosig-fit] Loadings shape: {None if fs.loadings_ is None else fs.loadings_.shape}")
    print(f"[factosig-fit] Scores shape: {None if fs.scores_ is None else fs.scores_.shape}")

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        loadings=fs.loadings_,
        scores=fs.scores_,
        mu=fs.mu_,
        psi=fs.psi_,
        V=fs.posterior_cov_,
        genes=np.asarray(genes),
        n_factors=np.int32(args.k),
        device=np.array(args.device),
        seed=np.int32(args.seed),
    )

    meta_path = out.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(
            {
                "input_h5ad": args.input_h5ad,
                "output_npz": str(out),
                "n_factors": args.k,
                "device": args.device,
                "max_genes": args.max_genes,
                "seed": args.seed,
                "n_cells": int(X.shape[0]),
                "n_genes": int(X.shape[1]),
            },
            f,
            indent=2,
        )

    print(f"Saved cached fit to: {out}")
    print(f"[factosig-fit] Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


