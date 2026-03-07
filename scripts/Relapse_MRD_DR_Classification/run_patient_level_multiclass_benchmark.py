#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (
    DEFAULT_MULTICLASS_ORDER,
    align_probability_output,
    choose_best,
    fit_logistic_model,
    latent_diagnostics,
    load_patient_dr_artifacts,
    log,
    make_hyperparam_grid,
    multiclass_metrics,
    split_csv_arg,
    split_csv_floats,
    write_json,
)


CLASS_NAMES = list(DEFAULT_MULTICLASS_ORDER)
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASS_NAMES)}
CLASS_LABELS = np.asarray(list(range(len(CLASS_NAMES))), dtype=int)


def evaluate_grid(
    X: np.ndarray,
    y: np.ndarray,
    *,
    combos: List[Dict],
    cv_folds: int,
    cv_repeats: int,
    random_state: int,
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
            prob = align_probability_output(clf, clf.predict_proba(X[te]), class_labels=CLASS_LABELS)
            pred = clf.predict(X[te]).astype(int)
            fold_rows.append(
                multiclass_metrics(
                    y[te],
                    pred,
                    prob,
                    class_labels=CLASS_LABELS,
                    class_names=CLASS_NAMES,
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


def collect_oof_predictions(
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
    prob_sum = np.zeros((len(y), len(CLASS_NAMES)), dtype=float)
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
        prob = align_probability_output(clf, clf.predict_proba(X[te]), class_labels=CLASS_LABELS)
        prob_sum[te] += prob
        counts[te] += 1
    counts[counts == 0] = 1
    probs = prob_sum / counts[:, None]
    preds = np.argmax(probs, axis=1).astype(int)
    return probs, preds


def fit_refit_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    feature_names: Sequence[str],
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
    coef = np.asarray(clf.coef_)
    rows: List[Dict] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for feat_idx, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "class_label": class_name,
                    "feature": str(feature_name),
                    "coef": float(coef[class_idx, feat_idx]),
                    "abs_coef": float(abs(coef[class_idx, feat_idx])),
                }
            )
    summary = {
        "ml_backend_used": clf.backend_used,
        "coef_l1_norm": float(np.abs(coef).sum()),
        "nonzero_coef_count": int(np.sum(np.abs(coef) > 0)),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run within-patient 4-class benchmarks on saved DR embeddings.")
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
    out_root = root / "multiclass_cv"
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
            "class_order": CLASS_NAMES,
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
                skip_rows.append({"patient": str(patient), "method": method, "reason": "missing_dr_artifacts"})
                continue

            meta, scores, _loadings, feature_names, _gene_names = load_patient_dr_artifacts(method_dir)
            counts = meta["relapse_mrd_label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
            min_support = int(counts.min())
            if min_support == 0:
                skip_rows.append(
                    {
                        "patient": str(patient),
                        "method": method,
                        "reason": "missing_one_or_more_classes",
                        **{f"n_{cls}": int(counts[cls]) for cls in CLASS_NAMES},
                    }
                )
                continue

            folds = min(int(args.cv_folds), min_support)
            if folds < 2:
                skip_rows.append(
                    {
                        "patient": str(patient),
                        "method": method,
                        "reason": "insufficient_support_for_cv",
                        "cv_folds_used": int(folds),
                        **{f"n_{cls}": int(counts[cls]) for cls in CLASS_NAMES},
                    }
                )
                continue

            log(f"Running multiclass benchmark patient={patient} method={method} folds={folds}")
            patient_out = out_root / str(patient) / method
            patient_out.mkdir(parents=True, exist_ok=True)

            y = meta["relapse_mrd_label"].map(CLASS_TO_INT).to_numpy(dtype=int)
            gdf = evaluate_grid(
                scores,
                y,
                combos=grid_all,
                cv_folds=folds,
                cv_repeats=int(args.cv_repeats),
                random_state=int(args.random_state),
                ml_backend=str(args.ml_backend),
                strict_gpu=bool(args.strict_gpu),
            )
            gdf.insert(0, "patient", str(patient))
            gdf.insert(1, "method", method)
            gdf.insert(2, "cv_folds_used", int(folds))
            gdf.to_csv(patient_out / "grid_metrics.csv", index=False)
            grid_rows.append(gdf)

            best = choose_best(
                gdf,
                primary=["macro_f1_mean", "balanced_accuracy_mean", "macro_recall_mean", "macro_auc_ovr_mean"],
            )
            best_row = best.to_dict()
            best_row.update(
                {
                    "patient": str(patient),
                    "method": method,
                    "cv_folds_used": int(folds),
                    **{f"n_{cls}": int(counts[cls]) for cls in CLASS_NAMES},
                }
            )

            best_l1_ratio = best.get("l1_ratio", np.nan)
            if pd.isna(best_l1_ratio) or str(best["penalty"]) != "elasticnet":
                best_l1_ratio = None

            probs, preds = collect_oof_predictions(
                scores,
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
            oof = meta.copy()
            oof["y_true"] = y
            oof["y_pred"] = preds
            oof["predicted_label"] = [CLASS_NAMES[idx] for idx in preds]
            for idx, class_name in enumerate(CLASS_NAMES):
                oof[f"prob_{class_name}"] = probs[:, idx]
            oof.to_csv(patient_out / "oof_predictions.csv")

            confusion = (
                pd.crosstab(
                    pd.Series(meta["relapse_mrd_label"], name="true"),
                    pd.Series(oof["predicted_label"], name="pred"),
                    dropna=False,
                )
                .reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
            )
            confusion.to_csv(patient_out / "oof_confusion_matrix.csv")

            metric_snapshot = multiclass_metrics(
                y,
                preds,
                probs,
                class_labels=CLASS_LABELS,
                class_names=CLASS_NAMES,
            )
            best_row.update({f"oof_{k}": v for k, v in metric_snapshot.items()})

            refit_coef_df, refit_summary = fit_refit_coefficients(
                scores,
                y,
                penalty=str(best["penalty"]),
                C=float(best["C"]),
                l1_ratio=best_l1_ratio,
                feature_names=feature_names,
                random_state=int(args.random_state),
                ml_backend=str(args.ml_backend),
                strict_gpu=bool(args.strict_gpu),
            )
            refit_coef_df.insert(0, "patient", str(patient))
            refit_coef_df.insert(1, "method", method)
            refit_coef_df.insert(2, "penalty", str(best["penalty"]))
            refit_coef_df.insert(3, "alpha", float(best["alpha"]))
            refit_coef_df.insert(4, "C", float(best["C"]))
            refit_coef_df.insert(5, "l1_ratio", best.get("l1_ratio", np.nan))
            refit_coef_df.to_csv(patient_out / "refit_coefficients.csv", index=False)
            coef_rows.append(refit_coef_df)

            diag_df = pd.DataFrame(latent_diagnostics(meta, scores))
            diag_df.to_csv(patient_out / "latent_diagnostics.csv", index=False)

            best_row.update(refit_summary)
            best_row["label_by_tech_neighbor_fraction"] = float(
                diag_df.loc[diag_df["label_column"] == "Tech", "same_label_neighbor_fraction"].iloc[0]
            ) if (diag_df["label_column"] == "Tech").any() else float("nan")
            best_rows.append(best_row)
            write_json(patient_out / "best_model.json", best_row)

            run_manifest.append(
                {
                    "patient": str(patient),
                    "method": method,
                    "status": "completed",
                    "cv_folds_used": int(folds),
                    **{f"n_{cls}": int(counts[cls]) for cls in CLASS_NAMES},
                }
            )

    if grid_rows:
        pd.concat(grid_rows, ignore_index=True).to_csv(out_root / "multiclass_grid_metrics.csv", index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(out_root / "multiclass_best_by_patient.csv", index=False)
    if skip_rows:
        pd.DataFrame(skip_rows).to_csv(out_root / "multiclass_skips_and_warnings.csv", index=False)
    if coef_rows:
        pd.concat(coef_rows, ignore_index=True).to_csv(out_root / "multiclass_refit_coefficients.csv", index=False)
    pd.DataFrame(run_manifest).to_csv(out_root / "run_manifest.csv", index=False)
    log(f"Done. Multiclass outputs written to {out_root}")


if __name__ == "__main__":
    main()
