#!/usr/bin/env python
"""
Elastic Net classification on cached DR outputs (sklearn FA or FactoSig),
integrated with ExperimentManager for logging. Supports per-patient or pan-patient
training, repeated stratified CV, and donor downsampling.

Examples:
  python sc_classification/scripts/elastic_net_from_dr_cache.py \
    --compare-exp-dir /home/minhang/mds_project/experiments/20251019_003140_compare_fa_100_none__f8a59d1c \
    --dr-method factosig \
    --patient-mode per \
    --cv-folds 5 --cv-repeats 3 \
    --l1-ratios 0.1 0.5 0.9 \
    --Cs 0.1 0.3 1.0 3.0 10.0 \
    --downsample random \
    --target-donor-fraction 0.7 \
    --donor-recipient-ratio-threshold 10.0 \
    --min-cells-per-type 20

Notes:
  - LogisticRegression with solver='saga', penalty='elasticnet'.
  - Score keys: 'X_sklearn_fa' for sklearn, 'fs_scores' for FactoSig.
  - Writes classification results under the existing experiment directory using
    ExperimentManager.save_classification_results(...).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

from sc_classification.utils.experiment_manager import ExperimentManager


def _find_transformed_h5ad(compare_exp_dir: Path, dr_method: str) -> Path:
    models_dir = compare_exp_dir / "models"
    if dr_method == "sklearn_fa":
        candidates = sorted(models_dir.glob("sklearn_fa_*/transformed_data.h5ad"))
    elif dr_method == "factosig":
        candidates = sorted(models_dir.glob("factosig_*/transformed_data.h5ad"))
    else:
        raise ValueError(f"Unknown dr_method: {dr_method}")
    if not candidates:
        raise FileNotFoundError(f"No transformed_data.h5ad found for {dr_method} under {models_dir}")
    return candidates[-1]


def _score_key_for_method(dr_method: str) -> str:
    return "X_sklearn_fa" if dr_method == "sklearn_fa" else "fs_scores"


def _downsample_donors(
    adata: sc.AnnData,
    donor_col: str = "source",
    recipient_label: str = "recipient",
    donor_label: str = "donor",
    ratio_thresh: float = 10.0,
) -> sc.AnnData:
    if donor_col not in adata.obs.columns:
        return adata
    src = adata.obs[donor_col].astype(str).values
    n_rec = int(np.sum(src == recipient_label))
    n_don = int(np.sum(src == donor_label))
    if n_rec == 0 or n_don == 0:
        return adata
    current_ratio = n_don / float(n_rec)
    if current_ratio <= ratio_thresh:
        return adata
    # target donor count
    target_don = int(np.floor(ratio_thresh * n_rec))
    keep_don = target_don
    # indices
    idx_don = np.where(src == donor_label)[0]
    rng = np.random.default_rng(0)
    keep_idx_don = rng.choice(idx_don, size=keep_don, replace=False)
    idx_rec = np.where(src == recipient_label)[0]
    keep_idx = np.sort(np.concatenate([idx_rec, keep_idx_don]))
    return adata[keep_idx, :].copy()


def _min_cells_filter(adata: sc.AnnData, cell_type_col: str, min_cells: int) -> sc.AnnData:
    if cell_type_col not in adata.obs.columns or min_cells <= 0:
        return adata
    counts = adata.obs[cell_type_col].value_counts()
    ok_types = counts[counts >= min_cells].index
    return adata[adata.obs[cell_type_col].isin(ok_types)].copy()


def _prepare_group(adata: sc.AnnData,
                   score_key: str,
                   target_col: str,
                   pos_label: str,
                   downsample: str,
                   donor_col: str,
                   ratio_thresh: float,
                   min_cells: int,
                   cell_type_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(adata.obsm[score_key])
    y = (adata.obs[target_col].astype(str).values == pos_label).astype(int)
    barcodes = adata.obs_names.to_numpy()
    # Downsampling
    if downsample == "random":
        adata2 = _downsample_donors(adata, donor_col=donor_col, ratio_thresh=ratio_thresh)
        adata2 = _min_cells_filter(adata2, cell_type_col=cell_type_col, min_cells=min_cells)
        X = np.asarray(adata2.obsm[score_key])
        y = (adata2.obs[target_col].astype(str).values == pos_label).astype(int)
        barcodes = adata2.obs_names.to_numpy()
    return X, y, barcodes


def _cv_grid_scores(X: np.ndarray,
                    y: np.ndarray,
                    l1_ratios: List[float],
                    Cs: List[float],
                    folds: int,
                    repeats: int,
                    random_state: int) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)
    records = []
    preds_store: Dict[str, np.ndarray] = {}
    for l1 in l1_ratios:
        for C in Cs:
            y_true_all = []
            y_prob_all = []
            y_pred_all = []
            for tr, te in rskf.split(X, y):
                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=float(l1),
                    C=float(C),
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=random_state,
                )
                clf.fit(X[tr], y[tr])
                prob = clf.predict_proba(X[te])[:, 1]
                pred = (prob >= 0.5).astype(int)
                y_true_all.append(y[te])
                y_prob_all.append(prob)
                y_pred_all.append(pred)
            y_true = np.concatenate(y_true_all)
            y_prob = np.concatenate(y_prob_all)
            y_pred = np.concatenate(y_pred_all)
            key = f"l1_{l1}_C_{C}"
            preds_store[key] = y_prob
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = np.nan
            ap = average_precision_score(y_true, y_prob)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            records.append({"l1_ratio": l1, "C": C, "auc": auc, "ap": ap, "acc": acc, "f1": f1, "precision": prec, "recall": rec})
    return pd.DataFrame(records), preds_store


def _refit_best(X: np.ndarray,
                y: np.ndarray,
                l1_ratio: float,
                C: float,
                random_state: int) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=float(l1_ratio),
        C=float(C),
        max_iter=4000,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    return clf


def main():
    p = argparse.ArgumentParser(description="Elastic Net classification from DR cache with ExperimentManager logging")
    p.add_argument("--compare-exp-dir", required=True, help="Path to compare DR experiment directory")
    p.add_argument("--dr-method", required=True, choices=["sklearn_fa", "factosig"], help="Which DR to use")
    p.add_argument("--patient-mode", default="per", choices=["per", "pan"], help="Per-patient or pan-patient training")
    p.add_argument("--target-col", default="CN.label", help="Obs column for labels")
    p.add_argument("--positive-class", default="cancer", help="Positive class label")
    p.add_argument("--patient-col", default="patient", help="Obs column for patient id")
    p.add_argument("--l1-ratios", nargs="*", type=float, default=[0.1, 0.5, 0.9], help="Grid of l1_ratio values")
    p.add_argument("--Cs", nargs="*", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0], help="Grid of C values")
    p.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds")
    p.add_argument("--cv-repeats", type=int, default=2, help="CV repeats")
    p.add_argument("--random-state", type=int, default=42)
    # Downsampling
    p.add_argument("--downsample", choices=["none", "random"], default="random")
    p.add_argument("--donor-col", default="source")
    p.add_argument("--donor-recipient-ratio-threshold", type=float, default=10.0)
    p.add_argument("--target-donor-fraction", type=float, default=0.7)  # reserved for future
    p.add_argument("--min-cells-per-type", type=int, default=20)
    p.add_argument("--cell-type-col", default="predicted.annotation")
    args = p.parse_args()

    compare_exp_dir = Path(args.compare_exp_dir)
    exp_id = compare_exp_dir.name

    # Load compare experiment via ExperimentManager
    em = ExperimentManager(compare_exp_dir.parent.as_posix())
    exp = em.load_experiment(exp_id)

    h5ad_path = _find_transformed_h5ad(compare_exp_dir, args.dr_method)
    score_key = _score_key_for_method(args.dr_method)
    ad = sc.read_h5ad(h5ad_path)
    if score_key not in ad.obsm:
        raise KeyError(f"Score key '{score_key}' not found in {h5ad_path}")

    # Determine groups
    if args.patient_mode == "pan":
        groups = [("all_patients", np.ones(ad.n_obs, dtype=bool))]
    else:
        pats = pd.unique(ad.obs[args.patient_col].astype(str).values)
        groups = [(pid, (ad.obs[args.patient_col].astype(str).values == pid)) for pid in sorted(pats)]

    # For each group/patient, run ENet with CV and refit best; save via ExperimentManager
    for pid, mask in groups:
        if mask.sum() < max(args.cv_folds * 2, 30):  # minimal size for CV stability
            continue
        ad_p = ad[mask].copy()
        X, y, barcodes = _prepare_group(
            adata=ad_p,
            score_key=score_key,
            target_col=args.target_col,
            pos_label=args.positive_class,
            downsample=args.downsample,
            donor_col=args.donor_col,
            ratio_thresh=args.donor_recipient_ratio_threshold,
            min_cells=args.min_cells_per_type,
            cell_type_col=args.cell_type_col,
        )
        if len(np.unique(y)) < 2:
            continue

        cv_df, _ = _cv_grid_scores(
            X=X,
            y=y,
            l1_ratios=args.l1_ratios,
            Cs=args.Cs,
            folds=args.cv_folds,
            repeats=args.cv_repeats,
            random_state=args.random_state,
        )
        # pick best by AUC, then AP
        cv_df = cv_df.sort_values(["auc", "ap"], ascending=[False, False])
        best = cv_df.iloc[0]
        clf = _refit_best(X, y, l1_ratio=float(best["l1_ratio"]), C=float(best["C"]), random_state=args.random_state)

        # coefficients
        coef = clf.coef_.ravel()
        k = X.shape[1]
        coef_df = pd.DataFrame({"coef": coef}, index=[f"factor_{i+1}" for i in range(k)])

        # group-level metrics from refit
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y, prob)
        except ValueError:
            auc = np.nan
        ap = average_precision_score(y, prob)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        prec = precision_score(y, pred)
        rec = recall_score(y, pred)
        metrics_df = pd.DataFrame([
            {
                "dr_method": args.dr_method,
                "l1_ratio": float(best["l1_ratio"]),
                "C": float(best["C"]),
                "auc": auc,
                "ap": ap,
                "acc": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "cv_auc": float(best["auc"]),
                "cv_ap": float(best["ap"]),
            }
        ])

        # correctness per cell
        correctness_df = pd.DataFrame({
            "barcode": barcodes,
            "y_true": y,
            "y_prob": prob,
            "y_pred": pred,
        }).set_index("barcode")

        # downsampling info
        downsampling_info = {
            "method": args.downsample,
            "donor_col": args.donor_col,
            "donor_recipient_ratio_threshold": args.donor_recipient_ratio_threshold,
            "min_cells_per_type": args.min_cells_per_type,
            "cell_type_col": args.cell_type_col,
        }

        exp.save_classification_results(
            patient_id=str(pid),
            coefficients=coef_df,
            metrics=metrics_df,
            correctness=correctness_df,
            downsampling_info=downsampling_info,
        )

    print("Elastic Net classification complete. Results saved under:")
    print(compare_exp_dir / "models" / "classification")


if __name__ == "__main__":
    main()



