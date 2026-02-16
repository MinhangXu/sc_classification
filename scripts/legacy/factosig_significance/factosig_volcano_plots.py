#!/usr/bin/env python
"""
Generate per-factor volcano plots comparing loadings vs significance.

Outputs two sets of plots (if inputs available):
  - loading_model_based_sig/: x = loading, y = -log10(model {q|p})
  - loading_bootstrap_based_sig/: x = loading, y = -log10(bootstrap q)

Usage example:
  python factosig_volcano_plots.py \
    --bootstrap-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_factosig_bootstrap_oct7.h5ad \
    --model-h5ad /home/minhang/mds_project/data/cohort_adata/adata_mrd_factosig_modelsig_oct7.h5ad \
    --out-root /home/minhang/mds_project/data/cohort_adata/factosig_volcano_oct7 \
    --model-metric q --q-threshold 0.05 --label-top 15
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _read_adata(path: Optional[str]):
    if not path:
        return None
    import anndata as ad  # imported lazily to avoid hard dep if not used

    return ad.read_h5ad(path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_volcano(
    x_loading: np.ndarray,
    y_sig: np.ndarray,
    gene_names: np.ndarray,
    out_png: Path,
    title: str,
    highlight_mask: Optional[np.ndarray] = None,
    label_top: int = 0,
    dpi: int = 150,
) -> None:
    import matplotlib.pyplot as plt

    # Compute symmetric x limits for nicer comparisons
    max_abs = float(np.nanmax(np.abs(x_loading))) if x_loading.size else 1.0
    xlim = (-max_abs, max_abs)

    # Robust y-limit (avoid outliers exploding the scale)
    finite_y = y_sig[np.isfinite(y_sig)]
    ymax = float(np.percentile(finite_y, 99.5)) if finite_y.size else 1.0
    ymax = max(ymax, 1.0)

    plt.figure(figsize=(6, 5), constrained_layout=True)
    # Base scatter
    plt.scatter(x_loading, y_sig, s=4, alpha=0.6, c="#7f7f7f", edgecolor="none")
    # Highlight significant if provided
    if highlight_mask is not None and highlight_mask.any():
        plt.scatter(
            x_loading[highlight_mask],
            y_sig[highlight_mask],
            s=6,
            alpha=0.8,
            c="#d62728",
            edgecolor="none",
            label="significant",
        )

    # Label top points by y value
    if label_top and label_top > 0 and gene_names is not None and gene_names.size:
        idx = np.argsort(-y_sig)[: int(label_top)]
        for i in idx:
            # Slight offset for visibility
            plt.text(
                float(x_loading[i]),
                float(y_sig[i]) + 0.02 * ymax,
                str(gene_names[i]),
                fontsize=6,
                ha="center",
                va="bottom",
            )

    plt.title(title)
    plt.xlabel("loading")
    plt.ylabel("significance (y)")
    plt.xlim(xlim)
    plt.ylim(0.0, ymax * 1.05)
    if highlight_mask is not None and highlight_mask.any():
        plt.legend(loc="upper right", fontsize=7, frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def _volcano_model(
    adata,
    out_dir: Path,
    metric: str,
    q_threshold: float,
    p_threshold: float,
    label_top: int,
    dpi: int,
) -> Tuple[int, int]:
    L = np.asarray(adata.varm["fs_loadings"])  # (p, k)
    genes = adata.var_names.to_numpy()
    p, k = L.shape
    if metric == "q":
        M = np.asarray(adata.varm["fs_loading_q_model"])  # (p, k)
        thr = float(q_threshold)
    elif metric == "p":
        M = np.asarray(adata.varm["fs_loading_p_model"])  # (p, k)
        thr = float(p_threshold)
    else:
        raise ValueError("metric must be 'q' or 'p'")

    M = np.clip(M, 1e-300, 1.0)
    Y = -np.log10(M)
    for j in range(k):
        x = L[:, j]
        y = Y[:, j]
        highlight = (M[:, j] < thr)
        out_png = out_dir / f"factor_{j:03d}.png"
        title = f"Factor {j} — -log10(model {metric}) vs loading"
        _plot_volcano(x, y, genes, out_png, title, highlight, label_top, dpi)
    return p, k


def _volcano_bootstrap(
    adata,
    out_dir: Path,
    q_threshold: float,
    label_top: int,
    dpi: int,
) -> Tuple[int, int]:
    L = np.asarray(adata.varm["fs_loadings"])  # (p, k)
    Q = np.asarray(adata.varm["fs_loading_q"])  # (p, k)
    genes = adata.var_names.to_numpy()
    p, k = L.shape
    Qc = np.clip(Q, 1e-300, 1.0)
    Y = -np.log10(Qc)
    thr = float(q_threshold)
    for j in range(k):
        x = L[:, j]
        y = Y[:, j]
        highlight = (Q[:, j] < thr)
        out_png = out_dir / f"factor_{j:03d}.png"
        title = f"Factor {j} — -log10(bootstrap q) vs loading"
        _plot_volcano(x, y, genes, out_png, title, highlight, label_top, dpi)
    return p, k


def main():
    parser = argparse.ArgumentParser(description="Generate per-factor volcano plots from FactoSig results.")
    parser.add_argument("--bootstrap-h5ad", default="", help=".h5ad with bootstrap significance (fs_loading_q)")
    parser.add_argument("--model-h5ad", default="", help=".h5ad with model-based significance (fs_loading_{p,q}_model)")
    parser.add_argument("--out-root", required=True, help="Output root directory for volcano plots")
    parser.add_argument("--model-metric", choices=["q", "p"], default="q", help="Model-based significance metric")
    parser.add_argument("--q-threshold", type=float, default=0.05, help="Q-value threshold for highlighting")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="P-value threshold for highlighting")
    parser.add_argument("--label-top", type=int, default=15, help="Annotate top-N points by y value")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = parser.parse_args()

    if not args.bootstrap_h5ad and not args.model_h5ad:
        raise SystemExit("Provide at least one of --bootstrap-h5ad or --model-h5ad")

    out_root = Path(args.out_root)
    # Directories
    dir_model = out_root / "loading_model_based_sig"
    dir_boot = out_root / "loading_bootstrap_based_sig"

    ad_boot = _read_adata(args.bootstrap_h5ad)
    ad_model = _read_adata(args.model_h5ad)

    # 1) model-based
    if ad_model is not None:
        _ensure_dir(dir_model)
        p2, k2 = _volcano_model(
            ad_model,
            dir_model,
            metric=args.model_metric,
            q_threshold=args.q_threshold,
            p_threshold=args.p_threshold,
            label_top=args.label_top,
            dpi=args.dpi,
        )
    # 2) bootstrap-based
    if ad_boot is not None:
        _ensure_dir(dir_boot)
        p3, k3 = _volcano_bootstrap(
            ad_boot,
            dir_boot,
            q_threshold=args.q_threshold,
            label_top=args.label_top,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()


