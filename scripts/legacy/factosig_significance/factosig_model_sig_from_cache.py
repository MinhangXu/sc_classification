#!/usr/bin/env python
"""
Compute model-based (OLS) significance from a cached FactoSig fit and write to .h5ad.

Usage:
  python factosig_model_sig_from_cache.py \
    --input-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_standardized_oct7.h5ad \
    --fit-npz /home/minhang/mds_project/data/cohort_adata/factosig_fit_k100_oct7.npz \
    --output-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_factosig_modelsig_oct7.h5ad
"""

import argparse
import time
from pathlib import Path

import numpy as np


def _load_adata_X(h5ad_path: str):
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X.A if hasattr(adata.X, "A") else adata.X
    X = np.asarray(X, dtype=np.float32)
    return adata, X


def main():
    parser = argparse.ArgumentParser(description="Model-based significance from cached FactoSig fit.")
    parser.add_argument("--input-h5ad", required=True, help="Standardized .h5ad used for fitting")
    parser.add_argument("--fit-npz", required=True, help="Cached fit arrays (.npz) from factosig_fit_cache.py")
    parser.add_argument("--output-h5ad", required=True, help="Output .h5ad with model-based significance written")
    args = parser.parse_args()

    t0 = time.time()
    adata, X = _load_adata_X(args.input_h5ad)
    print(f"[model-sig] Loaded matrix: n={X.shape[0]}, p={X.shape[1]}, dtype={X.dtype}")

    from factosig import FactoSig

    data = np.load(args.fit_npz)
    L = data["loadings"]
    Z = data["scores"]
    mu = data["mu"]
    genes_fit = data.get("genes")

    if genes_fit is not None:
        genes_fit = [g.decode("utf-8") if isinstance(g, (bytes, bytearray)) else str(g) for g in genes_fit]
        name_to_idx = {g: i for i, g in enumerate(list(adata.var_names))}
        missing = [g for g in genes_fit if g not in name_to_idx]
        if missing:
            raise RuntimeError(f"Input AnnData is missing {len(missing)} genes from cached fit, e.g., {missing[:5]}")
        idx = np.asarray([name_to_idx[g] for g in genes_fit], dtype=int)
        X = X[:, idx]
        adata = adata[:, genes_fit].copy()
    print(f"[model-sig] Using k={L.shape[1]}")

    fs = FactoSig(n_factors=L.shape[1], device="cpu", verbose=True)
    fs.loadings_ = np.asarray(L)
    fs.scores_ = np.asarray(Z)
    fs.mu_ = np.asarray(mu)

    t_sig0 = time.time()
    res = fs.significance(X, method="model")
    t_sig1 = time.time()
    print(f"[model-sig] Model-based significance wall time: {t_sig1 - t_sig0:.1f}s")

    # Write results using the module-level writer to avoid fit-state checks.
    from factosig.io.anndata_io import to_anndata as fs_to_anndata

    meta = {
        "writer": "factosig_model_sig_from_cache",
        "method": "model",
        "n_factors": int(L.shape[1]),
    }
    fs_to_anndata(
        adata=adata,
        L=np.asarray(L),
        Z=np.asarray(Z),
        z=res.get("z"),
        q=res.get("q"),
        stab=None,
        meta=meta,
        z_key="fs_loading_z_model",
        q_key="fs_loading_q_model",
        p=res.get("p"),
        se=res.get("se"),
        p_key="fs_loading_p_model",
        se_key="fs_loading_se_model",
        meta_key="fs_meta",
    )
    out = Path(args.output_h5ad)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out)
    print(f"Saved model-based significance to: {out}")
    print(f"[model-sig] Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


