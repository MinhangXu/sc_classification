#!/usr/bin/env python
"""
Plan 1.C supervised latent benchmark at fixed K=40.

Runs:
- pooled-cell repeated stratified CV (pan-patient) with per-patient OOF metrics
- per-patient repeated stratified CV

Across:
- DR methods: pca, fa, factosig, factosig_promax, cnmf
- penalties: L1, L2, Elastic Net
- downsampling: none, random
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_div(n: float, d: float) -> float:
    return float(n / d) if d else np.nan


def split_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def split_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    out = {
        "n": int(len(y_true)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "acc": float(accuracy_score(y_true, y_pred)),
        "mal_recall": safe_div(tp, tp + fn),
        "mal_precision": safe_div(tp, tp + fp),
        "norm_recall": safe_div(tn, tn + fp),
        "norm_precision": safe_div(tn, tn + fn),
    }
    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auc"] = np.nan
        out["ap"] = np.nan
    return out


@dataclass
class DownsamplingConfig:
    method: str = "none"
    donor_col: str = "source"
    donor_label: str = "donor"
    strat_col: str = "predicted.annotation"
    target_donor_fraction: float = 0.7
    ratio_threshold: float = 10.0
    min_cells_per_type: int = 20
    random_state: int = 42


def random_downsample_stratified_indices(
    meta: pd.DataFrame,
    cfg: DownsamplingConfig,
) -> Tuple[np.ndarray, Dict]:
    details: Dict = {
        "downsampling_method": cfg.method,
        "n_cells_before": int(meta.shape[0]),
        "scenario": "none",
        "n_donor_original": 0,
        "n_recipient_original": 0,
        "final_fraction_used": 1.0,
        "n_donor_after_downsampling": 0,
        "n_cells_after": int(meta.shape[0]),
        "parameters": {
            "donor_col": cfg.donor_col,
            "strat_col": cfg.strat_col,
            "target_donor_fraction": cfg.target_donor_fraction,
            "ratio_threshold": cfg.ratio_threshold,
            "min_cells_per_type": cfg.min_cells_per_type,
            "random_state": cfg.random_state,
        },
        "per_stratum_log": {},
    }

    if cfg.method == "none":
        return np.arange(meta.shape[0]), details

    src = meta[cfg.donor_col].astype(str).values
    donor_mask = src == cfg.donor_label
    n_donor = int(donor_mask.sum())
    n_recip = int((~donor_mask).sum())
    details["n_donor_original"] = n_donor
    details["n_recipient_original"] = n_recip

    if n_donor == 0:
        details["scenario"] = "no_donor_cells"
        return np.arange(meta.shape[0]), details

    target_fraction = float(cfg.target_donor_fraction)
    if n_recip > 0:
        current_ratio = n_donor / float(n_recip)
        if current_ratio <= cfg.ratio_threshold:
            target_fraction = 1.0
            details["scenario"] = "ratio_below_threshold"
        else:
            target_n_donor = int(n_recip * cfg.ratio_threshold)
            target_fraction = max(0.0, min(1.0, target_n_donor / float(n_donor)))
            details["scenario"] = "ratio_above_threshold"
    else:
        details["scenario"] = "no_recipient_cells"

    details["final_fraction_used"] = target_fraction
    rng = np.random.default_rng(cfg.random_state)

    recipient_idx = np.where(~donor_mask)[0]
    donor_idx_global = np.where(donor_mask)[0]

    strata = meta.iloc[donor_idx_global][cfg.strat_col].astype(str).values
    kept_donor_idx_local: List[int] = []
    for g in np.unique(strata):
        g_local_idx = np.where(strata == g)[0]
        n_group = len(g_local_idx)
        n_keep_frac = int(n_group * target_fraction)
        n_keep = max(cfg.min_cells_per_type, n_keep_frac)
        n_keep = min(n_keep, n_group)
        details["per_stratum_log"][g] = {"before": int(n_group), "after": int(n_keep)}
        if n_keep > 0:
            chosen_local = rng.choice(g_local_idx, size=n_keep, replace=False)
            kept_donor_idx_local.extend(chosen_local.tolist())

    kept_donor_idx_global = donor_idx_global[np.asarray(kept_donor_idx_local, dtype=int)] if kept_donor_idx_local else np.array([], dtype=int)
    kept_idx = np.sort(np.concatenate([recipient_idx, kept_donor_idx_global]))

    details["n_donor_after_downsampling"] = int(len(kept_donor_idx_global))
    details["n_cells_after"] = int(len(kept_idx))
    return kept_idx, details


def apply_downsampling_with_low_malignant_policy(
    meta: pd.DataFrame,
    cfg: DownsamplingConfig,
    n_malignant: int,
    low_malignant_threshold: int = 10,
    severe_ratio_after_threshold: float = 20.0,
) -> Tuple[np.ndarray, Dict]:
    if cfg.method == "none":
        return np.arange(meta.shape[0]), {"downsampling_method": "none", "policy": "none"}

    cfg1 = DownsamplingConfig(**cfg.__dict__)
    if 2 <= n_malignant < low_malignant_threshold:
        cfg1.min_cells_per_type = 5
    idx1, info1 = random_downsample_stratified_indices(meta, cfg1)

    src = meta.iloc[idx1][cfg.donor_col].astype(str).values
    n_d = int(np.sum(src == cfg.donor_label))
    n_r = int(np.sum(src != cfg.donor_label))
    ratio_after = float(n_d / n_r) if n_r > 0 else np.inf
    info1["policy"] = "pass1"
    info1["donor_to_recipient_ratio_after"] = ratio_after

    if 2 <= n_malignant < low_malignant_threshold and ratio_after > severe_ratio_after_threshold:
        cfg2 = DownsamplingConfig(**cfg.__dict__)
        cfg2.min_cells_per_type = 5
        cfg2.ratio_threshold = 5.0
        idx2, info2 = random_downsample_stratified_indices(meta, cfg2)
        src2 = meta.iloc[idx2][cfg.donor_col].astype(str).values
        n_d2 = int(np.sum(src2 == cfg.donor_label))
        n_r2 = int(np.sum(src2 != cfg.donor_label))
        info2["policy"] = "pass2_ratio5_min5"
        info2["donor_to_recipient_ratio_after"] = float(n_d2 / n_r2) if n_r2 > 0 else np.inf
        info2["first_pass_ratio_after"] = ratio_after
        return idx2, info2

    return idx1, info1


def eval_cv_grid(
    X: np.ndarray,
    y: np.ndarray,
    combos: List[Dict],
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    splitter = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state)
    for combo in combos:
        fold_rows: List[Dict] = []
        l1_ratio = combo.get("l1_ratio", None)
        if isinstance(l1_ratio, float) and np.isnan(l1_ratio):
            l1_ratio = None
        if combo["penalty"] != "elasticnet":
            l1_ratio = None
        for tr, te in splitter.split(X, y):
            clf = LogisticRegression(
                penalty=combo["penalty"],
                solver="saga",
                C=float(combo["C"]),
                l1_ratio=l1_ratio,
                class_weight="balanced",
                random_state=random_state,
                max_iter=5000,
                n_jobs=-1,
            )
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_rows.append(binary_metrics(y[te], prob, pred))

        fdf = pd.DataFrame(fold_rows)
        row = {k: combo.get(k) for k in ["penalty", "alpha", "C", "l1_ratio"]}
        for c in ["auc", "ap", "acc", "mal_recall", "mal_precision", "norm_recall", "norm_precision"]:
            row[f"{c}_mean"] = float(fdf[c].mean()) if c in fdf else np.nan
            row[f"{c}_std"] = float(fdf[c].std()) if c in fdf else np.nan
        row["folds_total"] = int(len(fdf))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_full_path_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    combos: List[Dict],
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coef_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    for combo in combos:
        l1_ratio = combo.get("l1_ratio", None)
        if isinstance(l1_ratio, float) and np.isnan(l1_ratio):
            l1_ratio = None
        if combo["penalty"] != "elasticnet":
            l1_ratio = None

        clf = LogisticRegression(
            penalty=combo["penalty"],
            solver="saga",
            C=float(combo["C"]),
            l1_ratio=l1_ratio,
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
            n_jobs=-1,
        )
        clf.fit(X, y)
        c = clf.coef_.ravel()
        nnz = int(np.sum(np.abs(c) > 0))
        summary_rows.append(
            {
                "penalty": combo["penalty"],
                "alpha": float(combo["alpha"]),
                "C": float(combo["C"]),
                "l1_ratio": combo.get("l1_ratio", np.nan),
                "nonzero_coef_count_full_refit": nnz,
                "coef_l1_norm_full_refit": float(np.sum(np.abs(c))),
            }
        )
        for i, coef in enumerate(c):
            coef_rows.append(
                {
                    "penalty": combo["penalty"],
                    "alpha": float(combo["alpha"]),
                    "C": float(combo["C"]),
                    "l1_ratio": combo.get("l1_ratio", np.nan),
                    "feature": feature_names[i] if i < len(feature_names) else f"feature_{i+1}",
                    "coef": float(coef),
                    "abs_coef": float(abs(coef)),
                    "is_nonzero": int(abs(coef) > 0),
                }
            )
    return pd.DataFrame(coef_rows), pd.DataFrame(summary_rows)


def collect_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state)
    probs = np.full((len(y),), np.nan, dtype=float)
    counts = np.zeros((len(y),), dtype=int)
    for tr, te in splitter.split(X, y):
        l1_ratio_eff = l1_ratio
        if penalty != "elasticnet":
            l1_ratio_eff = None
        clf = LogisticRegression(
            penalty=penalty,
            solver="saga",
            C=float(C),
            l1_ratio=l1_ratio_eff,
            class_weight="balanced",
            random_state=random_state,
            max_iter=5000,
            n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        for i, idx in enumerate(te):
            if np.isnan(probs[idx]):
                probs[idx] = p[i]
            else:
                probs[idx] += p[i]
            counts[idx] += 1
    mask = counts > 0
    probs[mask] = probs[mask] / counts[mask]
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def make_hyperparam_grid(alpha_grid: np.ndarray, enet_l1_ratios: List[float]) -> List[Dict]:
    grid: List[Dict] = []
    for a in alpha_grid:
        grid.append({"penalty": "l1", "alpha": float(a), "C": float(1.0 / a), "l1_ratio": np.nan})
    for a in alpha_grid:
        grid.append({"penalty": "l2", "alpha": float(a), "C": float(1.0 / a), "l1_ratio": np.nan})
    for l1r in enet_l1_ratios:
        for a in alpha_grid:
            grid.append({"penalty": "elasticnet", "alpha": float(a), "C": float(1.0 / a), "l1_ratio": float(l1r)})
    return grid


def choose_best(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["auc_mean"] = d["auc_mean"].fillna(-np.inf)
    d["ap_mean"] = d["ap_mean"].fillna(-np.inf)
    d["mal_recall_mean"] = d["mal_recall_mean"].fillna(-np.inf)
    d["C"] = d["C"].fillna(np.inf)
    d = d.sort_values(["auc_mean", "ap_mean", "mal_recall_mean", "C"], ascending=[False, False, False, True])
    return d.iloc[0]


def load_df_npz(path: Path) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame(z["data"], index=z["index"], columns=z["columns"])


def load_features(exp_dir: Path, k: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, List[str]]]:
    base = exp_dir / "preprocessing" / "adata_processed.h5ad"
    ad = sc.read_h5ad(base, backed="r")
    meta = ad.obs[["patient", "CN.label", "source", "predicted.annotation"]].copy()
    meta.index = ad.obs_names.copy()
    obs_names = np.asarray(ad.obs_names).astype(str)
    del ad

    features: Dict[str, np.ndarray] = {}
    feature_names: Dict[str, List[str]] = {}

    # FA + FactoSig from rehydrated seed1 h5ad
    h5 = exp_dir / "preprocessing" / f"adata_processed_with_plan0_dr_k{k}_seed1.h5ad"
    adk = sc.read_h5ad(h5, backed="r")
    for m, key in [("fa", "X_fa"), ("factosig", "X_factosig")]:
        if key in adk.obsm.keys():
            X = np.asarray(adk.obsm[key]).astype(np.float32, copy=False)
            features[m] = X
            feature_names[m] = [f"{m}_{i+1}" for i in range(X.shape[1])]
    del adk

    # PCA + factosig_promax from stability caches
    for m in ["pca", "factosig_promax"]:
        p = exp_dir / "analysis" / "plan0" / "stability" / m / f"k_{k}" / "replicates" / "seed_1" / "scores.npy"
        X = np.load(p).astype(np.float32, copy=False)
        features[m] = X
        feature_names[m] = [f"{m}_{i+1}" for i in range(X.shape[1])]

    # cNMF from curated consensus usages
    up = sorted((exp_dir / "models" / "cnmf_plan0" / "curated" / f"k_{k}" / "consensus").glob(f"*usages.k_{k}.dt_0_5.consensus.df.npz"))
    if not up:
        raise FileNotFoundError("cNMF usage file not found in curated consensus directory.")
    U = load_df_npz(up[0])
    if not np.array_equal(obs_names, U.index.astype(str).values):
        raise ValueError("cNMF usage index does not match base obs_names order.")
    features["cnmf"] = U.values.astype(np.float32, copy=False)
    feature_names["cnmf"] = [str(c) for c in U.columns]

    # Validate method matrices
    n = len(meta)
    for m, X in features.items():
        if X.shape[0] != n or X.shape[1] != k:
            raise ValueError(f"Unexpected shape for {m}: {X.shape}; expected ({n}, {k})")

    return meta, features, feature_names


def write_input_diagnostics(meta: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_patient_label = meta.groupby(["patient", "CN.label"], observed=False).size().unstack(fill_value=0)
    by_patient_source = meta.groupby(["patient", "source"], observed=False).size().unstack(fill_value=0)
    by_source_label = meta.groupby(["source", "CN.label"], observed=False).size().unstack(fill_value=0)
    by_patient_label.to_csv(out_dir / "counts_patient_by_label.csv")
    by_patient_source.to_csv(out_dir / "counts_patient_by_source.csv")
    by_source_label.to_csv(out_dir / "counts_source_by_label.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Plan 1.C supervised latent benchmark at K=40.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--low-malignant-threshold", type=int, default=10)
    parser.add_argument("--skip-malignant-leq", type=int, default=1)
    parser.add_argument("--severe-ratio-after-threshold", type=float, default=20.0)
    parser.add_argument("--methods", default="pca,fa,factosig,factosig_promax,cnmf")
    parser.add_argument("--modes", default="pooled,per_patient")
    parser.add_argument("--downsampling-variants", default="none,random")
    parser.add_argument("--penalties", default="l1,l2,elasticnet")
    parser.add_argument("--alpha-log10-min", type=float, default=-4.0)
    parser.add_argument("--alpha-log10-max", type=float, default=5.0)
    parser.add_argument("--alpha-num", type=int, default=20)
    parser.add_argument("--enet-l1-ratios", default="0.1,0.5,0.9")
    parser.add_argument("--output-subdir", default="analysis/plan1c_supervised_latent_k40")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    out_root = exp_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    log(f"Loading metadata + DR features from {exp_dir}")
    meta, features, feat_names = load_features(exp_dir, k=int(args.k))
    meta = meta.copy()
    meta["y"] = (meta["CN.label"].astype(str) == "cancer").astype(int)

    write_input_diagnostics(meta, out_root / "input_diagnostics")

    methods = split_csv_arg(args.methods)
    modes = set(split_csv_arg(args.modes))
    downsample_variants = split_csv_arg(args.downsampling_variants)
    penalties = split_csv_arg(args.penalties)
    alpha_grid = np.logspace(float(args.alpha_log10_min), float(args.alpha_log10_max), int(args.alpha_num))
    enet_l1_ratios = split_csv_floats(args.enet_l1_ratios)
    grid_all = make_hyperparam_grid(alpha_grid, enet_l1_ratios)
    run_manifest: List[Dict] = []
    pooled_grid_rows: List[pd.DataFrame] = []
    pooled_best_rows: List[Dict] = []
    pooled_per_patient_rows: List[Dict] = []
    per_patient_grid_rows: List[pd.DataFrame] = []
    per_patient_best_rows: List[Dict] = []
    per_patient_skip_rows: List[Dict] = []
    coef_rows: List[Dict] = []

    for method in methods:
        if method not in features:
            raise ValueError(f"Method '{method}' is not available in loaded features. Available: {sorted(features.keys())}")
        X_full = features[method]
        log(f"Starting method={method}")
        for ds in downsample_variants:
            log(f"  Downsampling variant={ds}")
            cfg = DownsamplingConfig(method=ds)

            # ---- pooled mode ----
            if "pooled" in modes:
                n_mal_full = int(meta["y"].sum())
                idx_keep, ds_info = apply_downsampling_with_low_malignant_policy(
                    meta=meta,
                    cfg=cfg,
                    n_malignant=n_mal_full,
                    low_malignant_threshold=int(args.low_malignant_threshold),
                    severe_ratio_after_threshold=float(args.severe_ratio_after_threshold),
                )
                meta_pool = meta.iloc[idx_keep].copy()
                X_pool = X_full[idx_keep]
                y_pool = meta_pool["y"].values
                pool_mode_dir = out_root / "pooled_cv" / method / ds
                pool_mode_dir.mkdir(parents=True, exist_ok=True)
                with open(pool_mode_dir / "downsampling_info.json", "w") as f:
                    json.dump(ds_info, f, indent=2, default=str)

                for pen in penalties:
                    grid = [g for g in grid_all if g["penalty"] == pen]
                    if not grid:
                        continue
                    log(f"    pooled penalty={pen} grid_size={len(grid)}")
                    gdf = eval_cv_grid(
                        X=X_pool,
                        y=y_pool,
                        combos=grid,
                        cv_folds=int(args.cv_folds),
                        cv_repeats=int(args.cv_repeats),
                        random_state=int(args.random_state),
                    )
                    coef_path_df, coef_summary_df = fit_full_path_coefficients(
                        X=X_pool,
                        y=y_pool,
                        feature_names=feat_names[method],
                        combos=grid,
                        random_state=int(args.random_state),
                    )
                    if not coef_summary_df.empty:
                        gdf = gdf.merge(
                            coef_summary_df[["penalty", "alpha", "C", "l1_ratio", "nonzero_coef_count_full_refit", "coef_l1_norm_full_refit"]],
                            on=["penalty", "alpha", "C", "l1_ratio"],
                            how="left",
                        )
                    gdf.insert(0, "mode", "pooled")
                    gdf.insert(1, "method", method)
                    gdf.insert(2, "downsampling", ds)
                    gdf.to_csv(pool_mode_dir / f"grid_metrics_{pen}.csv", index=False)
                    if not coef_path_df.empty:
                        coef_path_df.insert(0, "mode", "pooled")
                        coef_path_df.insert(1, "method", method)
                        coef_path_df.insert(2, "downsampling", ds)
                        coef_path_df.to_csv(pool_mode_dir / f"coefficient_path_{pen}.csv", index=False)
                    pooled_grid_rows.append(gdf)

                    best = choose_best(gdf)
                    best_row = best.to_dict()
                    best_row.update({"mode": "pooled", "method": method, "downsampling": ds})
                    pooled_best_rows.append(best_row)

                    best_l1_ratio = best.get("l1_ratio", np.nan)
                    if pd.isna(best_l1_ratio):
                        best_l1_ratio = None
                    probs, preds = collect_oof_predictions(
                        X=X_pool,
                        y=y_pool,
                        cv_folds=int(args.cv_folds),
                        cv_repeats=int(args.cv_repeats),
                        random_state=int(args.random_state),
                        penalty=str(best["penalty"]),
                        C=float(best["C"]),
                        l1_ratio=best_l1_ratio,
                    )

                    oof = meta_pool.copy()
                    oof["method"] = method
                    oof["downsampling"] = ds
                    oof["penalty"] = str(best["penalty"])
                    oof["alpha"] = float(best["alpha"])
                    oof["C"] = float(best["C"])
                    oof["l1_ratio"] = best.get("l1_ratio", np.nan)
                    oof["y_prob"] = probs
                    oof["y_pred"] = preds
                    oof.to_csv(pool_mode_dir / f"oof_predictions_best_{pen}.csv")

                    for patient, sub in oof.groupby("patient", observed=False):
                        if sub["y"].nunique() < 2:
                            m = {"auc": np.nan, "ap": np.nan, "acc": np.nan, "mal_recall": np.nan, "mal_precision": np.nan, "norm_recall": np.nan, "norm_precision": np.nan, "n": len(sub)}
                        else:
                            m = binary_metrics(sub["y"].values.astype(int), sub["y_prob"].values.astype(float), sub["y_pred"].values.astype(int))
                        pooled_per_patient_rows.append(
                            {
                                "method": method,
                                "downsampling": ds,
                                "penalty_family": pen,
                                "patient": patient,
                                **m,
                                "n_malignant": int(np.sum(sub["y"].values == 1)),
                                "n_healthy": int(np.sum(sub["y"].values == 0)),
                            }
                        )

                    run_manifest.append(
                        {
                            "mode": "pooled",
                            "method": method,
                            "downsampling": ds,
                            "penalty_family": pen,
                            "status": "completed",
                            "n_cells": int(len(meta_pool)),
                            "n_malignant": int(np.sum(y_pool == 1)),
                        }
                    )

            # ---- per-patient mode ----
            if "per_patient" in modes:
                per_mode_dir = out_root / "per_patient_cv" / method / ds
                per_mode_dir.mkdir(parents=True, exist_ok=True)
                for patient, pm in meta.groupby("patient", observed=False):
                    mask = meta["patient"].astype(str).values == str(patient)
                    idx_patient = np.where(mask)[0]
                    meta_p = meta.iloc[idx_patient].copy()
                    X_p = X_full[idx_patient]
                    n_mal = int(meta_p["y"].sum())
                    if n_mal <= int(args.skip_malignant_leq):
                        per_patient_skip_rows.append(
                            {
                                "method": method,
                                "downsampling": ds,
                                "patient": patient,
                                "reason": f"n_malignant <= {int(args.skip_malignant_leq)}",
                                "n_cells_before": int(meta_p.shape[0]),
                                "n_malignant_before": n_mal,
                            }
                        )
                        continue

                    idx_keep_p, ds_info_p = apply_downsampling_with_low_malignant_policy(
                        meta=meta_p,
                        cfg=cfg,
                        n_malignant=n_mal,
                        low_malignant_threshold=int(args.low_malignant_threshold),
                        severe_ratio_after_threshold=float(args.severe_ratio_after_threshold),
                    )
                    meta_pd = meta_p.iloc[idx_keep_p].copy()
                    X_pd = X_p[idx_keep_p]
                    y_pd = meta_pd["y"].values

                    n_pos = int(np.sum(y_pd == 1))
                    n_neg = int(np.sum(y_pd == 0))
                    minority = min(n_pos, n_neg)
                    folds = min(int(args.cv_folds), minority)
                    if minority < 2 or folds < 2:
                        per_patient_skip_rows.append(
                            {
                                "method": method,
                                "downsampling": ds,
                                "patient": patient,
                                "reason": "insufficient minority after downsampling",
                                "n_cells_before": int(meta_p.shape[0]),
                                "n_malignant_before": int(meta_p["y"].sum()),
                                "n_cells_after": int(meta_pd.shape[0]),
                                "n_malignant_after": int(np.sum(y_pd == 1)),
                                "downsampling_policy": ds_info_p.get("policy", ""),
                            }
                        )
                        continue

                    with open(per_mode_dir / f"{patient}_downsampling_info.json", "w") as f:
                        json.dump(ds_info_p, f, indent=2, default=str)

                    for pen in penalties:
                        grid = [g for g in grid_all if g["penalty"] == pen]
                        if not grid:
                            continue
                        gdf = eval_cv_grid(
                            X=X_pd,
                            y=y_pd,
                            combos=grid,
                            cv_folds=folds,
                            cv_repeats=int(args.cv_repeats),
                            random_state=int(args.random_state),
                        )
                        coef_path_df, coef_summary_df = fit_full_path_coefficients(
                            X=X_pd,
                            y=y_pd,
                            feature_names=feat_names[method],
                            combos=grid,
                            random_state=int(args.random_state),
                        )
                        if not coef_summary_df.empty:
                            gdf = gdf.merge(
                                coef_summary_df[["penalty", "alpha", "C", "l1_ratio", "nonzero_coef_count_full_refit", "coef_l1_norm_full_refit"]],
                                on=["penalty", "alpha", "C", "l1_ratio"],
                                how="left",
                            )
                        gdf.insert(0, "mode", "per_patient")
                        gdf.insert(1, "method", method)
                        gdf.insert(2, "downsampling", ds)
                        gdf.insert(3, "patient", str(patient))
                        gdf.insert(4, "cv_folds_used", int(folds))
                        gdf.to_csv(per_mode_dir / f"{patient}_grid_metrics_{pen}.csv", index=False)
                        if not coef_path_df.empty:
                            coef_path_df.insert(0, "mode", "per_patient")
                            coef_path_df.insert(1, "method", method)
                            coef_path_df.insert(2, "downsampling", ds)
                            coef_path_df.insert(3, "patient", str(patient))
                            coef_path_df.to_csv(per_mode_dir / f"{patient}_coefficient_path_{pen}.csv", index=False)
                        per_patient_grid_rows.append(gdf)

                        best = choose_best(gdf)
                        best_row = best.to_dict()
                        best_row.update(
                            {
                                "mode": "per_patient",
                                "method": method,
                                "downsampling": ds,
                                "patient": str(patient),
                                "cv_folds_used": int(folds),
                                "n_cells_after": int(meta_pd.shape[0]),
                                "n_malignant_after": int(np.sum(y_pd == 1)),
                                "n_healthy_after": int(np.sum(y_pd == 0)),
                                "downsampling_policy": ds_info_p.get("policy", ""),
                            }
                        )
                        per_patient_best_rows.append(best_row)

                        best_l1_ratio = best.get("l1_ratio", np.nan)
                        if pd.isna(best_l1_ratio):
                            best_l1_ratio = None
                        clf = LogisticRegression(
                            penalty=str(best["penalty"]),
                            solver="saga",
                            C=float(best["C"]),
                            l1_ratio=best_l1_ratio if str(best["penalty"]) == "elasticnet" else None,
                            class_weight="balanced",
                            random_state=int(args.random_state),
                            max_iter=5000,
                            n_jobs=-1,
                        )
                        clf.fit(X_pd, y_pd)
                        coefs = clf.coef_.ravel()
                        fnames = feat_names[method]
                        for i, c in enumerate(coefs):
                            coef_rows.append(
                                {
                                    "method": method,
                                    "downsampling": ds,
                                    "patient": str(patient),
                                    "penalty_family": pen,
                                    "feature": fnames[i] if i < len(fnames) else f"{method}_{i+1}",
                                    "coef": float(c),
                                    "alpha": float(best["alpha"]),
                                    "C": float(best["C"]),
                                    "l1_ratio": best.get("l1_ratio", np.nan),
                                }
                            )

                        run_manifest.append(
                            {
                                "mode": "per_patient",
                                "method": method,
                                "downsampling": ds,
                                "patient": str(patient),
                                "penalty_family": pen,
                                "status": "completed",
                                "n_cells": int(meta_pd.shape[0]),
                                "n_malignant": int(np.sum(y_pd == 1)),
                                "cv_folds_used": int(folds),
                            }
                        )

    # Write final consolidated outputs
    log("Writing consolidated output tables")
    if pooled_grid_rows:
        pd.concat(pooled_grid_rows, ignore_index=True).to_csv(out_root / "pooled_grid_metrics.csv", index=False)
    if pooled_best_rows:
        pd.DataFrame(pooled_best_rows).to_csv(out_root / "pooled_best_by_objective.csv", index=False)
    if pooled_per_patient_rows:
        pd.DataFrame(pooled_per_patient_rows).to_csv(out_root / "pooled_per_patient_metrics.csv", index=False)
    if per_patient_grid_rows:
        pd.concat(per_patient_grid_rows, ignore_index=True).to_csv(out_root / "per_patient_grid_metrics.csv", index=False)
    if per_patient_best_rows:
        pd.DataFrame(per_patient_best_rows).to_csv(out_root / "per_patient_best_by_patient.csv", index=False)
    if per_patient_skip_rows:
        pd.DataFrame(per_patient_skip_rows).to_csv(out_root / "per_patient_skips_and_warnings.csv", index=False)
    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(out_root / "per_patient_refit_coefficients.csv", index=False)
    pd.DataFrame(run_manifest).to_csv(out_root / "run_manifest.csv", index=False)

    config = {
        "experiment_dir": str(exp_dir),
        "k": int(args.k),
        "cv_folds": int(args.cv_folds),
        "cv_repeats": int(args.cv_repeats),
        "random_state": int(args.random_state),
        "methods": methods,
        "modes": sorted(list(modes)),
        "penalties": penalties,
        "alpha_grid": {
            "log10_min": float(args.alpha_log10_min),
            "log10_max": float(args.alpha_log10_max),
            "num": int(args.alpha_num),
        },
        "enet_l1_ratios": enet_l1_ratios,
        "downsampling_variants": downsample_variants,
        "low_malignant_threshold": int(args.low_malignant_threshold),
        "skip_malignant_leq": int(args.skip_malignant_leq),
        "severe_ratio_after_threshold": float(args.severe_ratio_after_threshold),
        "low_malignant_policy": {
            "pass1": {"ratio_threshold": 10.0, "min_cells_per_type": 5},
            "pass2_if_ratio_gt_20": {"ratio_threshold": 5.0, "min_cells_per_type": 5},
        },
    }
    with open(out_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    log(f"Done. Outputs at: {out_root}")


if __name__ == "__main__":
    main()
