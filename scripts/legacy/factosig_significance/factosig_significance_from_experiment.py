#!/usr/bin/env python
"""
Compute FactoSig loading significance (model-based and bootstrap) for a given
experiment directory and cache the results as compact .npz files.

This is analogous in spirit to `factosig_model_sig_from_cache.py` and
`factosig_bootstrap_from_cache.py`, but operates directly on the Experiment
Manager layout (scores/loadings saved under models/factosig_k).

Usage example:

  python factosig_significance_from_experiment.py \
    --exp-dir /home/minhang/mds_project/sc_classification/scripts/experiments/20251115_054547_factosig_only_100_none__ec10d437 \
    --method factosig \
    --k 100 \
    --B 200 \
    --device cpu \
    --seed 42

The script:
  1. Reconstructs the standardized matrix X used for FactoSig by applying the
     same cohort filtering and per-gene z-scoring as in `run_factosig_only.py`.
  2. Loads scores/loadings (and var_names) from the experiment's model folder.
  3. Builds a lightweight FactoSig object and computes:
       - model-based significance (OLS)   -> model_sig.npz
       - bootstrap significance + stability -> bootstrap_sig_B{B}_seed{seed}.npz
  4. Stores these in a `significance/` subdirectory under the model folder.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scanpy as sc
import yaml
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Helpers (closely mirror run_factosig_only.py)                              #
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


def _load_experiment_matrix(exp_dir: Path, method: str, k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reconstruct the standardized matrix X used in the FactoSig fit and
    return (X_sub, meta) where X_sub has genes matched to the model's var_names.
    """
    # Read config.yaml for preprocessing config
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found under {exp_dir}")
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    input_h5ad = cfg.get("preprocessing", {}).get("used_input_h5ad")
    cohort_scope = cfg.get("preprocessing", {}).get("cohort_scope", "mrd_only_patients_with_malignant")
    gene_selection = cfg.get("preprocessing", {}).get("gene_selection", "all")
    hvg = int(cfg.get("preprocessing", {}).get("hvg", 3000))

    print(f"[sig-from-exp] Loading AnnData from: {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    print(f"[sig-from-exp] Raw shape: {adata.shape}")

    # Apply same cohort filtering as in run_factosig_only.py
    adata_mrd = _filter_mrd(adata)
    adata_mrd = adata_mrd[adata_mrd.obs[_detect_label_col(adata_mrd)].isin(["cancer", "normal"])].copy()
    adata_cohort = _filter_cohort_scope(adata_mrd, cohort_scope)
    print(f"[sig-from-exp] Cohort after filtering: {adata_cohort.shape} (scope={cohort_scope})")

    # Gene selection
    if gene_selection == "hvg":
        adata_sel = _select_hvg(adata_cohort, n_top_genes=hvg)
        print(f"[sig-from-exp] HVG selected: {adata_sel.n_vars} genes")
    else:
        adata_sel = adata_cohort.copy()
        print(f"[sig-from-exp] Using all genes: {adata_sel.n_vars}")

    # Standardize by gene
    print("[sig-from-exp] Standardizing by gene ...")
    adata_std, _ = _standardize_by_gene(adata_sel)

    # Subset / reorder genes to match the model var_names
    model_dir = exp_dir / "models" / f"{method}_{k}"
    var_names_path = model_dir / "var_names.txt"
    if not var_names_path.exists():
        raise FileNotFoundError(f"var_names.txt not found under {model_dir}")
    with var_names_path.open("r") as f:
        genes_model = [ln.strip() for ln in f]

    name_to_idx = {g: i for i, g in enumerate(list(adata_std.var_names))}
    missing = [g for g in genes_model if g not in name_to_idx]
    if missing:
        raise RuntimeError(
            f"Standardized AnnData is missing {len(missing)} genes from model, e.g., {missing[:5]}"
        )
    idx = np.asarray([name_to_idx[g] for g in genes_model], dtype=int)
    X = adata_std.X.toarray() if hasattr(adata_std.X, "toarray") else adata_std.X
    X_sub = np.asarray(X[:, idx], dtype=np.float32)
    print(f"[sig-from-exp] Standardized matrix for significance: {X_sub.shape}")

    meta = {
        "input_h5ad": input_h5ad,
        "cohort_scope": cohort_scope,
        "gene_selection": gene_selection,
        "hvg": hvg,
        "genes_model": genes_model,
    }
    return X_sub, meta


# --------------------------------------------------------------------------- #
# Main CLI                                                                   #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute FactoSig significance (model + bootstrap) from an experiment directory."
    )
    parser.add_argument("--exp-dir", required=True, help="Path to a single experiment directory")
    parser.add_argument("--method", default="factosig", help="DR method name (default: factosig)")
    parser.add_argument("--k", type=int, required=True, help="Number of factors (must match experiment)")
    parser.add_argument("--B", type=int, default=200, help="Number of bootstraps for stability-based significance")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for bootstrap refits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-model", action="store_true", help="Skip model-based (OLS) significance computation"
    )
    parser.add_argument(
        "--skip-bootstrap", action="store_true", help="Skip bootstrap-based significance computation"
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    method = args.method
    k = int(args.k)
    model_dir = exp_dir / "models" / f"{method}_{k}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    t0 = time.time()

    # Load X used for significance + meta
    X, meta = _load_experiment_matrix(exp_dir, method, k)

    # Load loadings/scores from experiment cache
    loadings = np.load(model_dir / "loadings.npy")
    scores = np.load(model_dir / "scores.npy")
    print(f"[sig-from-exp] Loaded loadings {loadings.shape}, scores {scores.shape}")

    from factosig import FactoSig

    sig_dir = model_dir / "significance"
    sig_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Model-based significance ------------------------ #
    if not args.skip_model:
        print("[sig-from-exp] Computing model-based significance (OLS) ...")
        fs_model = FactoSig(
            n_factors=k,
            device="cpu",
            random_state=args.seed,
            verbose=True,
        )
        fs_model.loadings_ = np.asarray(loadings)
        fs_model.scores_ = np.asarray(scores)
        # fs_model.mu_ left as None -> compute_model_significance will center X by column means

        t_m0 = time.time()
        res_model = fs_model.significance(X, method="model")
        t_m1 = time.time()
        print(f"[sig-from-exp] Model-based significance wall time: {t_m1 - t_m0:.1f}s")

        out_model = sig_dir / "model_significance.npz"
        np.savez_compressed(
            out_model,
            se=res_model.get("se"),
            z=res_model.get("z"),
            p=res_model.get("p"),
            q=res_model.get("q"),
            meta=json.dumps(
                {
                    "type": "model",
                    "n_factors": k,
                    "exp_dir": str(exp_dir),
                    "method": method,
                    "seed": args.seed,
                    **meta,
                }
            ),
        )
        print(f"[sig-from-exp] Saved model-based significance to: {out_model}")

    # -------------------- Bootstrap-based significance ---------------------- #
    if not args.skip_bootstrap:
        print(
            f"[sig-from-exp] Computing bootstrap significance (B={args.B}, device={args.device}, seed={args.seed}) ..."
        )
        fs_boot = FactoSig(
            n_factors=k,
            device=args.device,
            random_state=args.seed,
            verbose=True,
        )
        fs_boot.loadings_ = np.asarray(loadings)

        t_b0 = time.time()
        res_boot = fs_boot.significance(X, B=args.B, method="bootstrap")
        t_b1 = time.time()
        print(f"[sig-from-exp] Bootstrap significance wall time: {t_b1 - t_b0:.1f}s")

        out_boot = sig_dir / f"bootstrap_significance_B{args.B}_seed{args.seed}.npz"
        np.savez_compressed(
            out_boot,
            se=res_boot.get("se"),
            z=res_boot.get("z"),
            p=res_boot.get("p"),
            q=res_boot.get("q"),
            stability=res_boot.get("stability"),
            meta=json.dumps(
                {
                    "type": "bootstrap",
                    "B": args.B,
                    "n_factors": k,
                    "exp_dir": str(exp_dir),
                    "method": method,
                    "device": args.device,
                    "seed": args.seed,
                    **meta,
                }
            ),
        )
        print(f"[sig-from-exp] Saved bootstrap significance to: {out_boot}")

    print(f"[sig-from-exp] Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


