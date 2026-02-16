#!/usr/bin/env python
"""
Run a DR suite: PCA, NMF, FA, FactoSig on MRD cohort with configurable scope.
Saves compact arrays + identifiers for downstream re-attachment, avoiding large h5ad files.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import time

# Absolute imports from package
from sc_classification.utils.experiment_manager import ExperimentManager, ExperimentConfig
from sc_classification.dimension_reduction.pca import PCA as PCAWrapper
from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis as FAWrapper
from sc_classification.dimension_reduction.nmf import NMF as NMFWrapper
from sc_classification.dimension_reduction.factosig import FactoSigDR


def _detect_label_col(adata: sc.AnnData) -> str:
    candidates = ['CN.label', 'cnLabel', 'cn_label', 'cnlabel']
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise KeyError("Could not find a label column among: " + ", ".join(candidates))


def _filter_mrd(adata: sc.AnnData) -> sc.AnnData:
    # Prefer explicit timepoint column
    if 'timepoint_type' in adata.obs.columns:
        mask = adata.obs['timepoint_type'] == 'MRD'
        return adata[mask].copy()
    # Fallback to sample substring as in notebooks
    for c in ['sample', 'Sample', 'SAMPLE']:
        if c in adata.obs.columns:
            mask = adata.obs[c].astype(str).str.contains('MRD', case=False, na=False)
            return adata[mask].copy()
    # If neither exists, return as is
    return adata.copy()


def _filter_cohort_scope(adata_mrd: sc.AnnData, scope: str) -> sc.AnnData:
    if scope == "mrd_all_patients":
        return adata_mrd
    if scope == "mrd_only_patients_with_malignant":
        label_col = _detect_label_col(adata_mrd)
        # find patient column
        patient_col = 'patient' if 'patient' in adata_mrd.obs.columns else None
        if patient_col is None:
            raise KeyError("Expected 'patient' column in obs to filter cohort by patients.")
        labels = adata_mrd.obs[label_col].astype(str)
        malignant_patients = set(adata_mrd.obs.loc[labels.str.lower().isin(['cancer','tumor','malignant']), patient_col])
        mask = adata_mrd.obs[patient_col].isin(malignant_patients)
        return adata_mrd[mask].copy()
    raise ValueError(f"Unknown cohort scope: {scope}")


def _select_hvg(adata: sc.AnnData, n_top_genes: int) -> sc.AnnData:
    ad = adata.copy()
    try:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor='seurat_v3')
    except Exception:
        # fallback simple variance
        X = ad.X.toarray() if hasattr(ad.X, 'toarray') else ad.X
        vars_ = np.var(X, axis=0)
        idx = np.argsort(vars_)[-n_top_genes:]
        ad.var['highly_variable'] = False
        ad.var.iloc[idx, ad.var.columns.get_loc('highly_variable')] = True
    return ad[:, ad.var['highly_variable']].copy()


def _standardize_by_gene(adata: sc.AnnData) -> Tuple[sc.AnnData, StandardScaler]:
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    scaler = StandardScaler(with_mean=True)
    Xz = scaler.fit_transform(X)
    ad = adata.copy()
    ad.X = Xz
    return ad, scaler


def _prepare_nmf_input(
    adata_base: sc.AnnData,
    genes_keep: List[str],
    normalize: bool,
    handle_negatives: str
) -> sc.AnnData:
    """
    Prepare a nonnegative matrix for NMF:
    - Prefer .raw if present, otherwise layer 'counts', otherwise X
    - Subset to selected genes_keep
    - Optionally library-normalize and log1p
    - Handle negatives by 'clip' or 'error'
    """
    # pick source
    if adata_base.raw is not None:
        ad_src = adata_base.raw.to_adata()
        ad_src = ad_src[:, [g for g in genes_keep if g in ad_src.var_names]].copy()
    elif 'counts' in adata_base.layers:
        ad_src = adata_base.copy()
        ad_src.X = ad_src.layers['counts']
        ad_src = ad_src[:, [g for g in genes_keep if g in ad_src.var_names]].copy()
    else:
        ad_src = adata_base[:, [g for g in genes_keep if g in adata_base.var_names]].copy()
    if normalize:
        sc.pp.normalize_total(ad_src, target_sum=1e4)
        sc.pp.log1p(ad_src)
    # Negative handling
    X = ad_src.X.toarray() if hasattr(ad_src.X, 'toarray') else ad_src.X
    if np.any(X < 0):
        if handle_negatives == 'clip':
            X = np.clip(X, 0, None)
            ad_src.X = X
        else:
            raise ValueError("Negative values found for NMF input. Use --nmf-handle-negatives clip or provide nonnegative input.")
    return ad_src


def _top10_ss_share(ss_per_factor: np.ndarray) -> float:
    if ss_per_factor.size == 0:
        return 0.0
    s = np.sort(ss_per_factor)
    denom = float(np.sum(s))
    top10 = float(np.sum(s[-10:])) if s.size >= 10 else float(np.sum(s))
    return top10 / (denom if denom > 0 else 1.0)


def main():
    parser = argparse.ArgumentParser(description="Run DR suite (PCA, NMF, FA, FactoSig) on MRD cohort.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--cohort-scope", default="mrd_only_patients_with_malignant",
                        choices=["mrd_only_patients_with_malignant", "mrd_all_patients"])
    parser.add_argument("--methods", nargs="+", default=["pca", "nmf", "fa", "factosig"],
                        choices=["pca", "nmf", "fa", "factosig"])
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--hvg", type=int, default=3000)
    parser.add_argument("--gene-selection", default="hvg", choices=["hvg", "all"],
                        help="Use 'hvg' to subset to top-variable genes, or 'all' to use all genes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiments-dir", default="experiments")
    # PCA
    parser.add_argument("--pca-svd-solver", default="randomized", choices=["auto", "full", "arpack", "randomized"])
    parser.add_argument("--pca-whiten", action="store_true")
    # FA
    parser.add_argument("--fa-svd-method", default="randomized", choices=["lapack", "randomized"])
    # NMF
    parser.add_argument("--nmf-beta-loss", default="kullback-leibler",
                        choices=["kullback-leibler", "frobenius", "itakura-saito"])
    parser.add_argument("--nmf-solver", default="mu", choices=["mu", "cd"])
    parser.add_argument("--nmf-handle-negatives", default="error", choices=["error", "clip"])
    parser.add_argument("--nmf-normalize", action="store_true", help="Apply library normalize + log1p before NMF")
    parser.add_argument("--nmf-max-iter", type=int, default=500)
    # FactoSig
    parser.add_argument("--fs-device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fs-lr", type=float, default=1e-2)
    parser.add_argument("--fs-max-iter", type=int, default=300)
    args = parser.parse_args()

    # Load
    adata = sc.read_h5ad(args.input_h5ad)
    print(f"[dr-suite] Loaded: {adata.shape}")

    # MRD filter and cohort scope
    adata_mrd = _filter_mrd(adata)
    adata_mrd = adata_mrd[adata_mrd.obs[_detect_label_col(adata_mrd)].isin(['cancer', 'normal'])].copy()
    adata_cohort = _filter_cohort_scope(adata_mrd, args.cohort_scope)
    print(f"[dr-suite] Cohort after filtering: {adata_cohort.shape} (scope={args.cohort_scope})")

    # Gene selection (shared gene set)
    if args.gene_selection == "hvg":
        t0 = time.time()
        adata_sel = _select_hvg(adata_cohort, n_top_genes=args.hvg)
        print(f"[dr-suite] HVG selected: {adata_sel.n_vars} genes (elapsed {time.time()-t0:.1f}s)")
    else:
        adata_sel = adata_cohort.copy()
        print(f"[dr-suite] Using all genes: {adata_sel.n_vars}")
    genes_sel = list(adata_sel.var_names)

    # Standardized view for PCA/FA/FactoSig
    print("[dr-suite] Standardizing input for PCA/FA/FactoSig...")
    t0 = time.time()
    adata_std, scaler = _standardize_by_gene(adata_sel)
    print(f"[dr-suite] Standardization complete (elapsed {time.time()-t0:.1f}s)")

    # NMF input prep (nonnegative)
    print("[dr-suite] Preparing nonnegative input for NMF...")
    t0 = time.time()
    adata_nmf_input = _prepare_nmf_input(
        adata_base=adata_cohort,
        genes_keep=genes_sel,
        normalize=args.nmf_normalize,
        handle_negatives=args.nmf_handle_negatives
    )
    print(f"[dr-suite] NMF input ready: {adata_nmf_input.shape} (elapsed {time.time()-t0:.1f}s)")
    # Mark HVGs only if using HVG selection (NMF wrapper can map back if use_hvg=True)
    if args.gene_selection == "hvg":
        adata_nmf_input.var['highly_variable'] = adata_nmf_input.var_names.isin(genes_sel)

    # Experiment config & creation
    config = {
        'preprocessing': {
            'used_input_h5ad': args.input_h5ad,
            'cohort_scope': args.cohort_scope,
            'gene_selection': args.gene_selection,
            'hvg': int(args.hvg)
        },
        'dimension_reduction': {
            'method': 'dr_suite',
            'n_components': int(args.n_components),
            'methods': args.methods,
        },
        'params': {
            'pca': {'svd_solver': args.pca_svd_solver, 'whiten': bool(args.pca_whiten)},
            'fa': {'svd_method': args.fa_svd_method},
            'nmf': {
                'beta_loss': args.nmf_beta_loss,
                'solver': args.nmf_solver,
                'handle_negatives': args.nmf_handle_negatives,
                'normalize': bool(args.nmf_normalize),
                'max_iter': int(args.nmf_max_iter),
            },
            'factosig': {'device': args.fs_device, 'lr': float(args.fs_lr), 'max_iter': int(args.fs_max_iter)}
        },
        'random_state': int(args.seed)
    }
    exp_cfg = ExperimentConfig(config)
    em = ExperimentManager(args.experiments_dir)
    exp = em.create_experiment(exp_cfg)
    print(f"[dr-suite] Experiment directory: {exp.experiment_dir}")

    # Helpers for saving summaries/metrics
    def summarize_common(ss_per_factor: np.ndarray) -> Dict[str, Any]:
        total_ss = float(np.sum(ss_per_factor))
        return {
            "total_ss": total_ss,
            "top10_ss_share": _top10_ss_share(ss_per_factor),
        }

    # Run PCA
    if "pca" in args.methods:
        print("[dr-suite] Running PCA...")
        t0 = time.time()
        pca_model = PCAWrapper()
        ad_work = adata_std.copy()
        ad_work = pca_model.fit_transform(
            ad_work,
            n_components=args.n_components,
            random_state=args.seed,
            standardize_input=False,  # already standardized
            svd_solver=args.pca_svd_solver,
            whiten=args.pca_whiten,
            save_fitted_models=False
        )
        pca = ad_work.uns.get("_temp_pca_model_obj")
        scores = ad_work.obsm["X_pca"]
        loadings = ad_work.varm["PCA_loadings"]
        ev = pca.explained_variance_
        evr = pca.explained_variance_ratio_
        sv = pca.singular_values_
        ss = np.sum(pca.components_**2, axis=1)
        metrics = summarize_common(ss)
        summary = (
            f"PCA (n={ad_work.n_obs}, p={ad_work.n_vars}, k={args.n_components})\n"
            f"top10_ss_share={metrics['top10_ss_share']:.3f}\n"
        )
        exp.save_dr_arrays(
            dr_method="pca",
            n_components=args.n_components,
            obs_names=list(ad_work.obs_names),
            var_names=list(ad_work.var_names),
            scores=scores,
            loadings=loadings,
            summary_text=summary,
            extras={
                "explained_variance": ev,
                "explained_variance_ratio": evr,
                "singular_values": sv,
                "dr_metrics": metrics
            },
            model=pca,
            keys={"obsm_key": "X_pca", "varm_key": "PCA_loadings"}
        )
        print(f"[dr-suite] PCA completed (elapsed {time.time()-t0:.1f}s)")

    # Run FA (sklearn)
    if "fa" in args.methods:
        print("[dr-suite] Running sklearn FactorAnalysis...")
        t0 = time.time()
        fa_model = FAWrapper()
        ad_work = adata_std.copy()
        ad_work = fa_model.fit_transform(
            ad_work,
            n_components=args.n_components,
            random_state=args.seed,
            standardize_input=False,
            svd_method=args.fa_svd_method,
            save_fitted_models=False
        )
        fa = ad_work.uns.get("_temp_fa_model_obj")
        scores = ad_work.obsm["X_fa"]
        loadings = ad_work.varm["FA_loadings"]
        psi = ad_work.uns.get("fa", {}).get("noise_variance_per_feature", None)
        ss = np.sum(fa.components_**2, axis=1)
        metrics = summarize_common(ss)
        summary = (
            f"sklearn FA (n={ad_work.n_obs}, p={ad_work.n_vars}, k={args.n_components})\n"
            f"top10_ss_share={metrics['top10_ss_share']:.3f}\n"
        )
        exp.save_dr_arrays(
            dr_method="fa",
            n_components=args.n_components,
            obs_names=list(ad_work.obs_names),
            var_names=list(ad_work.var_names),
            scores=scores,
            loadings=loadings,
            summary_text=summary,
            extras={"psi": psi, "dr_metrics": metrics},
            model=fa,
            keys={"obsm_key": "X_fa", "varm_key": "FA_loadings", "var_psi_key": "fa_psi"}
        )
        print(f"[dr-suite] FA completed (elapsed {time.time()-t0:.1f}s)")

    # Run NMF
    if "nmf" in args.methods:
        print("[dr-suite] Running NMF...")
        t0 = time.time()
        nmf_model = NMFWrapper()
        ad_work = adata_nmf_input.copy()
        ad_work = nmf_model.fit_transform(
            ad_work,
            n_components=args.n_components,
            random_state=args.seed,
            save_model=False,
            save_dir=None,
            use_hvg=(args.gene_selection == "hvg"),
            beta_loss=args.nmf_beta_loss,
            standardize_input=False,
            handle_negative_values=args.nmf_handle_negatives
        )
        scores = ad_work.obsm["X_nmf"]  # W
        loadings = ad_work.varm["NMF_components"]  # gene × k
        # Reconstruction error (approximate Frobenius)
        # Build H for genes_used only in the order of ad_work.var_names
        genes_used = ad_work.uns.get("nmf", {}).get("genes_used", list(ad_work.var_names))
        gene_index = {g: i for i, g in enumerate(ad_work.var_names)}
        used_idx = [gene_index[g] for g in genes_used if g in gene_index]
        H_used = loadings[used_idx, :].T  # k × |genes_used|
        X_used = ad_work[:, genes_used].X
        X_used = X_used.toarray() if hasattr(X_used, 'toarray') else X_used
        recon = scores @ H_used
        rec_err = float(np.linalg.norm(X_used - recon, ord='fro'))
        # Metrics
        ss = np.sum(loadings**2, axis=0)
        metrics = summarize_common(ss)
        summary = (
            f"NMF (n={ad_work.n_obs}, p={ad_work.n_vars}, k={args.n_components})\n"
            f"top10_ss_share={metrics['top10_ss_share']:.3f}, recon_fro={rec_err:.3f}\n"
        )
        exp.save_dr_arrays(
            dr_method="nmf",
            n_components=args.n_components,
            obs_names=list(ad_work.obs_names),
            var_names=list(ad_work.var_names),
            scores=scores,
            loadings=loadings,
            summary_text=summary,
            extras={
                "reconstruction_error": rec_err,
                "explained_variance_ratio": ad_work.uns.get("nmf", {}).get("explained_variance_ratio", None),
                "dr_metrics": metrics
            },
            model=None,
            keys={"obsm_key": "X_nmf", "varm_key": "NMF_components"}
        )
        print(f"[dr-suite] NMF completed (elapsed {time.time()-t0:.1f}s)")

    # Run FactoSig
    if "factosig" in args.methods:
        print("[dr-suite] Running FactoSig...")
        t0 = time.time()
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
            save_fitted_models=False
        )
        fs = ad_work.uns.get("_temp_factosig_model_obj")
        scores = ad_work.obsm["X_factosig"]
        loadings = ad_work.varm["FACTOSIG_loadings"]
        psi = ad_work.var["FACTOSIG_psi"].values if "FACTOSIG_psi" in ad_work.var.columns else None
        ss = np.sum(loadings**2, axis=0)
        metrics = summarize_common(ss)
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
            keys={"obsm_key": "X_factosig", "varm_key": "FACTOSIG_loadings", "var_psi_key": "FACTOSIG_psi"}
        )
        print(f"[dr-suite] FactoSig completed (elapsed {time.time()-t0:.1f}s)")

    print("[dr-suite] Completed. Array outputs saved per method under:")
    for m in args.methods:
        print(f"  - {exp.experiment_dir / 'models' / f'{m}_{args.n_components}'}")


if __name__ == "__main__":
    main()


