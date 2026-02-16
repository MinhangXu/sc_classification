#!/usr/bin/env python
"""
Load cached FactoSig fit arrays and compute bootstrap-based significance with low memory.

Usage:
  python factosig_bootstrap_from_cache.py \
    --input-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_standardized_oct7.h5ad \
    --fit-npz /home/minhang/mds_project/data/cohort_adata/factosig_fit_k100_oct7.npz \
    --output-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_factosig_bootstrap_oct7.h5ad \
    --k 100 \
    --B 200 \
    --device cpu \
    --seed 42
"""

import argparse
import time
from pathlib import Path

import numpy as np


def _load_adata_X(h5ad_path: str) -> tuple[object, np.ndarray]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X.A if hasattr(adata.X, "A") else adata.X
    X = np.asarray(X, dtype=np.float32)
    return adata, X


def main():
    parser = argparse.ArgumentParser(description="Bootstrap significance from cached FactoSig fit.")
    parser.add_argument("--input-h5ad", required=True, help="Standardized .h5ad used for fitting")
    parser.add_argument("--fit-npz", required=True, help="Cached fit arrays (.npz) from factosig_fit_cache.py")
    parser.add_argument("--output-h5ad", required=True, help="Output .h5ad with significance written")
    parser.add_argument("--k", type=int, required=True, help="Number of factors (must match cached fit)")
    parser.add_argument("--B", type=int, default=200, help="Number of bootstraps")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load X and AnnData
    t0 = time.time()
    adata, X = _load_adata_X(args.input_h5ad)
    print(f"[factosig-bootstrap] Loaded matrix: n={X.shape[0]}, p={X.shape[1]}, dtype={X.dtype}")

    # Load cached fit, from factosig_fit_cache.py 
    data = np.load(args.fit_npz)
    L = data["loadings"]
    Z = data["scores"]
    mu = data["mu"]
    psi = data["psi"]
    V = data["V"]
    genes_fit = data.get("genes")

    # Subset and reorder genes to match the cached fit
    if genes_fit is not None:
        genes_fit = [g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else str(g) for g in genes_fit]   # decode utf-8 if bytes, otherwise convert to string
        # Build index map from adata.var_names -> column index
        name_to_idx = {g: i for i, g in enumerate(list(adata.var_names))}
        missing = [g for g in genes_fit if g not in name_to_idx]
        if missing:
            raise RuntimeError(f"Input AnnData is missing {len(missing)} genes from cached fit, e.g., {missing[:5]}")
        idx = np.asarray([name_to_idx[g] for g in genes_fit], dtype=int)  # convert gene names to indices
        X = X[:, idx]  # subset X to keep dimensions consistent for to_anndata
        # also subset adata to keep dimensions consistent for to_anndata
        adata = adata[:, genes_fit].copy()
    print(f"[factosig-bootstrap] Using k={args.k}, B={args.B}, device={args.device}, seed={args.seed}")

    # Build a FactoSig instance and inject cached parameters
    from factosig import FactoSig

    fs = FactoSig(
        n_factors=args.k,
        device=args.device,
        random_state=args.seed,
        verbose=True,
    )
    fs.loadings_ = np.asarray(L)
    fs.scores_ = np.asarray(Z)
    fs.mu_ = np.asarray(mu)
    fs.psi_ = np.asarray(psi)
    fs.posterior_cov_ = np.asarray(V)

    # Compute bootstrap significance on chosen device
    t_boot0 = time.time()
    sig = fs.significance(X, B=args.B)
    t_boot1 = time.time()
    print(f"[factosig-bootstrap] Bootstrap wall time: {t_boot1 - t_boot0:.1f}s")
    # Write results to AnnData and save
    fs.to_anndata(adata)
    out = Path(args.output_h5ad)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out)
    print(f"Saved bootstrap significance to: {out}")
    print(f"[factosig-bootstrap] Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


