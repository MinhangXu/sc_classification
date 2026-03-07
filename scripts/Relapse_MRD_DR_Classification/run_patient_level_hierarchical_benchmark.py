#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (
    binary_metrics,
    choose_best,
    fit_logistic_model,
    latent_diagnostics,
    load_patient_dr_artifacts,
    log,
    make_hyperparam_grid,
    split_csv_arg,
    split_csv_floats,
    write_json,
)


def evaluate_binary_grid(
    X: np.ndarray,
    y: np.ndarray,
    *,
    combos: List[Dict],
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
    positive_name: str,
    negative_name: str,
    ml_backend: str,
    strict_gpu: bool,
) -> pd.DataFrame:
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv_folds),
        n_repeats=int(cv_repeats),
        random_state=int(random_state),
    )
    rows: List[Dict] = []
    for combo in combos:
        fold_rows: List[Dict] = []
        backend_used = "cpu"
        l1_ratio = combo.get("l1_ratio", np.nan)
        if pd.isna(l1_ratio) or combo["penalty"] != "elasticnet":
            l1_ratio = None
        for tr, te in splitter.split(X, y):
            clf = fit_logistic_model(
                penalty=str(combo["penalty"]),
                C=float(combo["C"]),
                l1_ratio=l1_ratio,
                random_state=int(random_state),
                ml_backend=str(ml_backend),
                strict_gpu=bool(strict_gpu),
            )
            clf.fit(X[tr], y[tr])
            backend_used = clf.backend_used
            prob = clf.predict_proba(X[te])[:, 1]
            pred = clf.predict(X[te]).astype(int)
            fold_rows.append(
                binary_metrics(
                    y[te],
                    pred,
                    prob,
                    positive_name=positive_name,
                    negative_name=negative_name,
                )
            )
        fold_df = pd.DataFrame(fold_rows)
        row = {
            "penalty": combo["penalty"],
            "alpha": float(combo["alpha"]),
            "C": float(combo["C"]),
            "l1_ratio": combo.get("l1_ratio", np.nan),
            "folds_total": int(len(fold_df)),
            "ml_backend_requested": ml_backend,
            "ml_backend_used": backend_used,
        }
        for column in fold_df.columns:
            row[f"{column}_mean"] = float(fold_df[column].mean())
            row[f"{column}_std"] = float(fold_df[column].std())
        rows.append(row)
    return pd.DataFrame(rows)


def collect_binary_oof(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
    ml_backend: str,
    strict_gpu: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    splitter = RepeatedStratifiedKFold(
        n_splits=int(cv_folds),
        n_repeats=int(cv_repeats),
        random_state=int(random_state),
    )
    prob_sum = np.zeros((len(y),), dtype=float)
    counts = np.zeros((len(y),), dtype=int)
    for tr, te in splitter.split(X, y):
        clf = fit_logistic_model(
            penalty=penalty,
            C=float(C),
            l1_ratio=l1_ratio,
            random_state=int(random_state),
            ml_backend=str(ml_backend),
            strict_gpu=bool(strict_gpu),
        )
        clf.fit(X[tr], y[tr])
        prob_sum[te] += clf.predict_proba(X[te])[:, 1]
        counts[te] += 1
    counts[counts == 0] = 1
    probs = prob_sum / counts
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def fit_binary_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    feature_names: List[str],
    random_state: int,
    ml_backend: str,
    strict_gpu: bool,
) -> Tuple[pd.DataFrame, Dict]:
    clf = fit_logistic_model(
        penalty=penalty,
        C=float(C),
        l1_ratio=l1_ratio,
        random_state=int(random_state),
        ml_backend=str(ml_backend),
        strict_gpu=bool(strict_gpu),
    )
    clf.fit(X, y)
    coef = np.asarray(clf.coef_).ravel()
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    )
    summary = {
        "ml_backend_used": clf.backend_used,
        "coef_l1_norm": float(np.abs(coef).sum()),
        "nonzero_coef_count": int(np.sum(np.abs(coef) > 0)),
    }
    return coef_df, summary


def task_datasets(meta: pd.DataFrame, scores: np.ndarray) -> List[Dict]:
    out: List[Dict] = []
    tasks = [
        (
            "cancer_vs_normal",
            meta["CN.label"].isin(["cancer", "normal"]).to_numpy(),
            (meta["CN.label"].astype(str) == "cancer").to_numpy(dtype=int),
            "cancer",
            "normal",
        ),
        (
            "time_within_cancer",
            (meta["CN.label"].astype(str) == "cancer").to_numpy(),
            (meta.loc[meta["CN.label"].astype(str) == "cancer", "timepoint_type"].astype(str) == "Relapse").to_numpy(dtype=int),
            "Relapse",
            "MRD",
        ),
        (
            "time_within_normal",
            (meta["CN.label"].astype(str) == "normal").to_numpy(),
            (meta.loc[meta["CN.label"].astype(str) == "normal", "timepoint_type"].astype(str) == "Relapse").to_numpy(dtype=int),
            "Relapse",
            "MRD",
        ),
    ]
    for task_name, mask, y, positive_name, negative_name in tasks:
        mask = np.asarray(mask, dtype=bool)
        out.append(
            {
                "task": task_name,
                "meta": meta.loc[mask].copy(),
                "X": scores[mask],
                "y": np.asarray(y, dtype=int),
                "positive_name": positive_name,
                "negative_name": negative_name,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hierarchical binary baselines on saved DR embeddings.")
    parser.add_argument("--dr-output-dir", required=True)
    parser.add_argument("--patients", default="")
    parser.add_argument("--methods", default="pca,fa,factosig")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alpha-log10-min", type=float, default=-4.0)
    parser.add_argument("--alpha-log10-max", type=float, default=5.0)
    parser.add_argument("--alpha-num", type=int, default=20)
    parser.add_argument("--enet-l1-ratios", default="0.1,0.5,0.9")
    parser.add_argument("--ml-backend", choices=["cpu", "gpu", "auto"], default="cpu")
    parser.add_argument("--strict-gpu", action="store_true")
    args = parser.parse_args()

    root = Path(args.dr_output_dir).resolve()
    patient_root = root / "patient_dr"
    out_root = root / "hierarchical_cv"
    out_root.mkdir(parents=True, exist_ok=True)

    methods = split_csv_arg(args.methods)
    if split_csv_arg(args.patients):
        patients = split_csv_arg(args.patients)
    else:
        patients = sorted([p.name for p in patient_root.iterdir() if p.is_dir()])

    alpha_grid = np.logspace(float(args.alpha_log10_min), float(args.alpha_log10_max), int(args.alpha_num))
    grid_all = make_hyperparam_grid(alpha_grid, split_csv_floats(args.enet_l1_ratios))
    write_json(
        out_root / "config.json",
        {
            "dr_output_dir": str(root),
            "patients": patients,
            "methods": methods,
            "tasks": ["cancer_vs_normal", "time_within_cancer", "time_within_normal"],
            "cv_folds": int(args.cv_folds),
            "cv_repeats": int(args.cv_repeats),
            "alpha_grid": [float(x) for x in alpha_grid],
            "enet_l1_ratios": split_csv_floats(args.enet_l1_ratios),
            "ml_backend": str(args.ml_backend),
            "strict_gpu": bool(args.strict_gpu),
            "balancing_strategy": "class_weight=balanced",
        },
    )

    grid_rows: List[pd.DataFrame] = []
    best_rows: List[Dict] = []
    skip_rows: List[Dict] = []
    coef_rows: List[pd.DataFrame] = []
    run_manifest: List[Dict] = []

    for patient in patients:
        for method in methods:
            method_dir = patient_root / str(patient) / method
            if not method_dir.exists():
                skip_rows.append({"patient": str(patient), "method": method, "task": "all", "reason": "missing_dr_artifacts"})
                continue

            meta, scores, _loadings, feature_names, _gene_names = load_patient_dr_artifacts(method_dir)
            for task in task_datasets(meta, scores):
                task_name = task["task"]
                task_meta = task["meta"]
                X = task["X"]
                y = task["y"]
                counts = pd.Series(y).value_counts().reindex([0, 1], fill_value=0)
                folds = min(int(args.cv_folds), int(counts.min()))
                if counts.min() == 0 or folds < 2:
                    skip_rows.append(
                        {
                            "patient": str(patient),
                            "method": method,
                            "task": task_name,
                            "reason": "insufficient_support_for_cv",
                            "n_negative": int(counts[0]),
                            "n_positive": int(counts[1]),
                            "cv_folds_used": int(folds),
                        }
                    )
                    continue

                log(f"Running hierarchical task patient={patient} method={method} task={task_name}")
                task_out = out_root / str(patient) / method / task_name
                task_out.mkdir(parents=True, exist_ok=True)

                gdf = evaluate_binary_grid(
                    X,
                    y,
                    combos=grid_all,
                    cv_folds=folds,
                    cv_repeats=int(args.cv_repeats),
                    random_state=int(args.random_state),
                    positive_name=str(task["positive_name"]),
                    negative_name=str(task["negative_name"]),
                    ml_backend=str(args.ml_backend),
                    strict_gpu=bool(args.strict_gpu),
                )
                gdf.insert(0, "patient", str(patient))
                gdf.insert(1, "method", method)
                gdf.insert(2, "task", task_name)
                gdf.insert(3, "cv_folds_used", int(folds))
                gdf.to_csv(task_out / "grid_metrics.csv", index=False)
                grid_rows.append(gdf)

                best = choose_best(
                    gdf,
                    primary=["balanced_accuracy_mean", "f1_mean", "auc_mean", "recall_mean"],
                )
                best_row = best.to_dict()
                best_row.update(
                    {
                        "patient": str(patient),
                        "method": method,
                        "task": task_name,
                        "cv_folds_used": int(folds),
                        "n_negative": int(counts[0]),
                        "n_positive": int(counts[1]),
                    }
                )

                best_l1_ratio = best.get("l1_ratio", np.nan)
                if pd.isna(best_l1_ratio) or str(best["penalty"]) != "elasticnet":
                    best_l1_ratio = None

                probs, preds = collect_binary_oof(
                    X,
                    y,
                    penalty=str(best["penalty"]),
                    C=float(best["C"]),
                    l1_ratio=best_l1_ratio,
                    cv_folds=folds,
                    cv_repeats=int(args.cv_repeats),
                    random_state=int(args.random_state),
                    ml_backend=str(args.ml_backend),
                    strict_gpu=bool(args.strict_gpu),
                )
                oof = task_meta.copy()
                oof["y_true"] = y
                oof["y_pred"] = preds
                oof["y_prob"] = probs
                oof.to_csv(task_out / "oof_predictions.csv")

                confusion = (
                    pd.crosstab(
                        pd.Series(y, name="true"),
                        pd.Series(preds, name="pred"),
                        dropna=False,
                    )
                    .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
                )
                confusion.to_csv(task_out / "oof_confusion_matrix.csv")

                best_row.update(
                    {
                        f"oof_{k}": v
                        for k, v in binary_metrics(
                            y,
                            preds,
                            probs,
                            positive_name=str(task["positive_name"]),
                            negative_name=str(task["negative_name"]),
                        ).items()
                    }
                )

                coef_df, coef_summary = fit_binary_coefficients(
                    X,
                    y,
                    penalty=str(best["penalty"]),
                    C=float(best["C"]),
                    l1_ratio=best_l1_ratio,
                    feature_names=feature_names,
                    random_state=int(args.random_state),
                    ml_backend=str(args.ml_backend),
                    strict_gpu=bool(args.strict_gpu),
                )
                coef_df.insert(0, "patient", str(patient))
                coef_df.insert(1, "method", method)
                coef_df.insert(2, "task", task_name)
                coef_df.insert(3, "penalty", str(best["penalty"]))
                coef_df.insert(4, "alpha", float(best["alpha"]))
                coef_df.insert(5, "C", float(best["C"]))
                coef_df.insert(6, "l1_ratio", best.get("l1_ratio", np.nan))
                coef_df.to_csv(task_out / "refit_coefficients.csv", index=False)
                coef_rows.append(coef_df)

                diag_df = pd.DataFrame(latent_diagnostics(task_meta, X))
                diag_df.to_csv(task_out / "latent_diagnostics.csv", index=False)

                best_row.update(coef_summary)
                best_rows.append(best_row)
                write_json(task_out / "best_model.json", best_row)

                run_manifest.append(
                    {
                        "patient": str(patient),
                        "method": method,
                        "task": task_name,
                        "status": "completed",
                        "cv_folds_used": int(folds),
                        "n_negative": int(counts[0]),
                        "n_positive": int(counts[1]),
                    }
                )

    if grid_rows:
        pd.concat(grid_rows, ignore_index=True).to_csv(out_root / "hierarchical_grid_metrics.csv", index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(out_root / "hierarchical_best_by_patient_task.csv", index=False)
    if skip_rows:
        pd.DataFrame(skip_rows).to_csv(out_root / "hierarchical_skips_and_warnings.csv", index=False)
    if coef_rows:
        pd.concat(coef_rows, ignore_index=True).to_csv(out_root / "hierarchical_refit_coefficients.csv", index=False)
    pd.DataFrame(run_manifest).to_csv(out_root / "run_manifest.csv", index=False)
    log(f"Done. Hierarchical outputs written to {out_root}")


if __name__ == "__main__":
    main()
