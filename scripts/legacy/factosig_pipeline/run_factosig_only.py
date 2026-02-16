#!/usr/bin/env python
"""
Run FactoSig only (Gaussian FA with varimax + factor re-ordering) on the MRD cohort.

This mirrors the preprocessing and caching structure of `run_dr_suite.py` but
only fits FactoSig. Outputs are saved under the same ExperimentManager
hierarchy in the `experiments` subdirectory by default.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler

from sc_classification.utils.experiment_manager import ExperimentManager, ExperimentConfig
from sc_classification.dimension_reduction.factosig import FactoSigDR


# --------------------------------------------------------------------------- #
# Helpers (copied from run_dr_suite.py to keep preprocessing identical)      #
# --------------------------------------------------------------------------- #


def _detect_label_col(adata: sc.AnnData) -> str:
    candidates = ["CN.label", "cnLabel", "cn_label", "cnlabel"]
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError("Could not find a label column among: " + ", ".join(candidates))


def _filter_mrd(adata: sc.AnnData) -> sc.AnnData:
    # Prefer explicit timepoint column
    if "timepoint_type" in adata.obs.columns:
        mask = adata.obs["timepoint_type"] == "MRD"
        return adata[mask].copy()
    # Fallback to sample substring as in notebooks
    for c in ["sample", "Sample", "SAMPLE"]:
        if c in adata.obs.columns:
            mask = adata.obs[c].astype(str).str.contains("MRD", case=False, na=False)
            return adata[mask].copy()
    # If neither exists, return as is
    return adata.copy()


def _filter_cohort_scope(adata_mrd: sc.AnnData, scope: str) -> sc.AnnData:
    if scope == "mrd_all_patients":
        return adata_mrd
    if scope == "mrd_only_patients_with_malignant":
        label_col = _detect_label_col(adata_mrd)
        # find patient column
        patient_col = "patient" if "patient" in adata_mrd.obs.columns else None
        if patient_col is None:
            raise KeyError("Expected 'patient' column in obs to filter cohort by patients.")
        labels = adata_mrd.obs[label_col].astype(str)
        malignant_patients = set(
            adata_mrd.obs.loc[labels.str.lower().isin(["cancer", "tumor", "malignant"]), patient_col]
        )
        mask = adata_mrd.obs[patient_col].isin(malignant_patients)
        return adata_mrd[mask].copy()
    raise ValueError(f"Unknown cohort scope: {scope}")


def _select_hvg(adata: sc.AnnData, n_top_genes: int) -> sc.AnnData:
    ad = adata.copy()
    try:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor="seurat_v3")
    except Exception:
        # fallback simple variance
        X = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
        vars_ = np.var(X, axis=0)
        idx = np.argsort(vars_)[-n_top_genes:]
        ad.var["highly_variable"] = False
        ad.var.iloc[idx, ad.var.columns.get_loc("highly_variable")] = True
    return ad[:, ad.var["highly_variable"]].copy()


def _standardize_by_gene(adata: sc.AnnData) -> Tuple[sc.AnnData, StandardScaler]:
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = StandardScaler(with_mean=True)
    Xz = scaler.fit_transform(X)
    ad = adata.copy()
    ad.X = Xz
    return ad, scaler


def _top10_ss_share(ss_per_factor: np.ndarray) -> float:
    if ss_per_factor.size == 0:
        return 0.0
    s = np.sort(ss_per_factor)
    denom = float(np.sum(s))
    top10 = float(np.sum(s[-10:])) if s.size >= 10 else float(np.sum(s))
    return top10 / (denom if denom > 0 else 1.0)


def _summarize_common(ss_per_factor: np.ndarray) -> Dict[str, Any]:
    total_ss = float(np.sum(ss_per_factor))
    return {
        "total_ss": total_ss,
        "top10_ss_share": _top10_ss_share(ss_per_factor),
    }


# --------------------------------------------------------------------------- #
# Main CLI                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FactoSig only on MRD cohort.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument(
        "--cohort-scope",
        default="mrd_only_patients_with_malignant",
        choices=["mrd_only_patients_with_malignant", "mrd_all_patients"],
    )
    parser.add_argument(
        "--gene-selection",
        default="all",
        choices=["hvg", "all"],
        help="Use 'hvg' to subset to top-variable genes, or 'all' to use all genes.",
    )
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments-dir", default="experiments")

    # FactoSig-specific options
    parser.add_argument("--fs-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fs-lr", type=float, default=1e-2)
    parser.add_argument("--fs-max-iter", type=int, default=300)

    args = parser.parse_args()

    # Load
    adata = sc.read_h5ad(args.input_h5ad)
    print(f"[factosig-only] Loaded: {adata.shape}")

    # MRD filter and cohort scope
    adata_mrd = _filter_mrd(adata)
    # restrict to cancer/normal labels for stability (matches run_dr_suite)
    adata_mrd = adata_mrd[adata_mrd.obs[_detect_label_col(adata_mrd)].isin(["cancer", "normal"])].copy()
    adata_cohort = _filter_cohort_scope(adata_mrd, args.cohort_scope)
    print(f"[factosig-only] Cohort after filtering: {adata_cohort.shape} (scope={args.cohort_scope})")

    # Gene selection
    if args.gene_selection == "hvg":
        adata_sel = _select_hvg(adata_cohort, n_top_genes=args.hvg)
        print(f"[factosig-only] HVG selected: {adata_sel.n_vars} genes")
    else:
        adata_sel = adata_cohort.copy()
        print(f"[factosig-only] Using all genes: {adata_sel.n_vars}")

    # Standardized view for FactoSig
    print("[factosig-only] Standardizing input for FactoSig...")
    adata_std, _ = _standardize_by_gene(adata_sel)

    # Experiment config & creation (mirrors run_dr_suite)
    config = {
        "preprocessing": {
            "used_input_h5ad": args.input_h5ad,
            "cohort_scope": args.cohort_scope,
            "gene_selection": args.gene_selection,
            "hvg": int(args.hvg),
        },
        "dimension_reduction": {
            "method": "factosig_only",
            "n_components": int(args.n_components),
            "methods": ["factosig"],
        },
        "params": {
            "factosig": {
                "device": args.fs_device,
                "lr": float(args.fs_lr),
                "max_iter": int(args.fs_max_iter),
            }
        },
        "random_state": int(args.seed),
    }
    exp_cfg = ExperimentConfig(config)
    em = ExperimentManager(args.experiments_dir)
    exp = em.create_experiment(exp_cfg)
    print(f"[factosig-only] Experiment directory: {exp.experiment_dir}")

    # Run FactoSig
    print("[factosig-only] Running FactoSig...")
    fs_model = FactoSigDR()
    ad_work = adata_std.copy()
    ad_work = fs_model.fit_transform(
        ad_work,
        n_components=args.n_components,
        random_state=args.seed,
        device=args.fs_device,
        lr=args.fs_lr,
        max_iter=args.fs_max_iter,
        verbose=True,
        save_fitted_models=False,
    )

    fs = ad_work.uns.get("_temp_factosig_model_obj")
    scores = ad_work.obsm["X_factosig"]
    loadings = ad_work.varm["FACTOSIG_loadings"]
    psi = ad_work.var["FACTOSIG_psi"].values if "FACTOSIG_psi" in ad_work.var.columns else None

    # Variance explained proxy via SS loadings per factor (after rotation + ordering)
    ss = np.sum(loadings**2, axis=0)
    metrics = _summarize_common(ss)

    summary = (
        f"FactoSig (n={ad_work.n_obs}, p={ad_work.n_vars}, k={args.n_components})\n"
        f"top10_ss_share={metrics['top10_ss_share']:.3f}\n"
    )

    exp.save_dr_arrays(
        dr_method="factosig",
        n_components=args.n_components,
        obs_names=list(ad_work.obs_names),
        var_names=list(ad_work.var_names),
        scores=scores,
        loadings=loadings,
        summary_text=summary,
        extras={"psi": psi, "dr_metrics": metrics},
        model=fs,
        keys={"obsm_key": "X_factosig", "varm_key": "FACTOSIG_loadings", "var_psi_key": "FACTOSIG_psi"},
    )
    print("[factosig-only] FactoSig completed.")
    print("[factosig-only] Array outputs saved under:")
    print(f"  - {exp.experiment_dir / 'models' / f'factosig_{args.n_components}'}")


if __name__ == "__main__":
    main()


