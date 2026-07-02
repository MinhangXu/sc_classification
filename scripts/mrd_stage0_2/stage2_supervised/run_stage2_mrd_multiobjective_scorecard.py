#!/usr/bin/env python
"""
Stage 2 multi-objective scorecard for the MRD old-34 broad screen.

This runner consumes the existing Stage 0/1 artifacts from
run_stage0_mrd_old34_broad_screen.py. It does not refit Stage 0 panels or
Stage 1 DR. All outputs therefore record the Stage 1 basis as transductive
when --stage1-fit-scope-note is set that way.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("stage2_mrd_multiobjective_scorecard")
DEFAULT_EXPERIMENT_DIR = Path(
    "/home/minhang/mds_project/sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
)
DEFAULT_SHORTLIST_OUTPUT = "analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv"


def add_src_path() -> None:
    sc_root = Path(__file__).resolve().parents[3]
    src = sc_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


add_src_path()
from sc_classification.utils.logistic_backend import make_logistic_regression  # noqa: E402


@dataclass(frozen=True)
class RepresentationSpec:
    source_scorecard_row_id: int
    stage0_panel_id: str
    stage0_panel_family: str
    stage0_panel_subfamily: str
    stage0_panel_type: str
    interpretation_layer: str
    n_covered_genes: int
    representation_family: str
    stage1_method: str
    requested_k: str
    effective_k: int | None
    seed: int
    scores_path: str
    features_path: str
    feature_names_path: str
    stage1_metadata_path: str
    gene_list_path: str
    best_quick_auroc: float
    best_quick_auprc: float
    best_quick_balanced_accuracy: float
    best_quick_malignant_precision: float
    best_quick_malignant_recall: float
    shortlist_reason: str
    force_include_reason: str


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default))


def split_csv(value: str | None, cast=str) -> list[Any]:
    if value is None:
        return []
    return [cast(x.strip()) for x in str(value).replace(",", " ").split() if x.strip()]


def split_comma_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def configure_logging(experiment_dir: Path, verbose: bool) -> Path:
    log_dir = experiment_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stage2_mrd_multiobjective_scorecard_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    LOGGER.info("Logging to %s", log_path)
    return log_path


def backend_alias(value: str) -> str:
    norm = str(value).strip().lower()
    if norm in {"cuml", "cuda", "gpu"}:
        return "gpu"
    if norm in {"sklearn", "cpu"}:
        return "cpu"
    if norm == "auto":
        return "auto"
    raise ValueError(f"Unknown backend {value!r}; expected cpu, sklearn, gpu, cuml, or auto")


def package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except Exception:
        return "not-installed"


def preflight_backend(args: argparse.Namespace) -> str:
    """Fail before writing large scorecards if the requested ML backend is unusable."""
    requested = backend_alias(args.backend)
    LOGGER.info(
        "Backend versions before preflight: sklearn=%s cuml=%s requested=%s strict_gpu=%s",
        package_version("scikit-learn"),
        package_version("cuml"),
        args.backend,
        bool(args.strict_gpu),
    )
    try:
        clf = make_logistic_regression(
            penalty="l2",
            C=1.0,
            l1_ratio=None,
            random_state=int(args.seed),
            max_iter=10,
            n_jobs=1,
            backend=requested,
            strict_gpu=bool(args.strict_gpu),
            class_weight=args.class_weight,
        )
        x = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y = np.asarray([0, 1, 0, 1], dtype=int)
        clf.fit(x, y)
        LOGGER.info("Backend preflight passed: requested=%s used=%s", args.backend, clf.backend_used)
        if requested == "auto" and clf.backend_used != "gpu":
            LOGGER.warning(
                "Backend auto did not use cuML/GPU; falling back to CPU. "
                "Versions: sklearn=%s, cuml=%s",
                package_version("scikit-learn"),
                package_version("cuml"),
            )
        return clf.backend_used
    except Exception as exc:
        versions = f"sklearn={package_version('scikit-learn')}, cuml={package_version('cuml')}"
        msg = (
            f"Requested backend {args.backend!r} failed preflight before model fitting. "
            f"Installed versions: {versions}. Original error: {exc!r}. "
            "If you need to proceed now, rerun with '--backend auto' without '--strict-gpu' "
            "or with '--backend cpu'. To require GPU, fix the RAPIDS/cuML environment so cuML "
            "imports cleanly with the installed scikit-learn version."
        )
        LOGGER.exception(msg)
        raise RuntimeError(msg) from exc


def c_grid_from_args(args: argparse.Namespace) -> list[float]:
    alpha_grid = np.logspace(float(args.c_grid_log10_min), float(args.c_grid_log10_max), int(args.c_grid_n))
    return [float(1.0 / alpha) for alpha in alpha_grid]


def make_model_grid(args: argparse.Namespace, *, penalties: list[str] | None = None) -> list[dict[str, Any]]:
    selected_penalties = penalties or split_csv(args.penalties, str)
    l1_ratios = split_csv(args.l1_ratios, float)
    rows: list[dict[str, Any]] = []
    for penalty in selected_penalties:
        penalty = str(penalty).lower()
        if penalty not in {"l1", "l2", "elasticnet"}:
            raise ValueError(f"Unsupported penalty: {penalty}")
        if penalty == "elasticnet":
            for l1_ratio in l1_ratios:
                for c_val in c_grid_from_args(args):
                    rows.append({"penalty": penalty, "C": float(c_val), "l1_ratio": float(l1_ratio)})
        else:
            for c_val in c_grid_from_args(args):
                rows.append({"penalty": penalty, "C": float(c_val), "l1_ratio": np.nan})
    return rows


def normalize_requested_k(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    try:
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
    except Exception:
        pass
    return str(value)


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def truth_columns(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    bal_acc = np.nan
    if np.isfinite(recall) and np.isfinite(specificity):
        bal_acc = (recall + specificity) / 2.0
    return {
        "threshold": float(threshold),
        "heldout_malignant_correct_tp": tp,
        "heldout_malignant_incorrect_fn": fn,
        "heldout_healthy_correct_tn": tn,
        "heldout_healthy_incorrect_fp": fp,
        "heldout_malignant_total": int((y_true == 1).sum()),
        "heldout_healthy_total": int((y_true == 0).sum()),
        "predicted_malignant_total": int((y_pred == 1).sum()),
        "malignant_precision": float(precision) if np.isfinite(precision) else np.nan,
        "malignant_recall": float(recall) if np.isfinite(recall) else np.nan,
        "healthy_recall_specificity": float(specificity) if np.isfinite(specificity) else np.nan,
        "balanced_accuracy": float(bal_acc) if np.isfinite(bal_acc) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
    }


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    out = truth_columns(y_true, y_prob, threshold)
    out["n_eval_cells"] = int(len(y_true))
    if len(y_true) and len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
        out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)))
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
        out["log_loss"] = np.nan
    return out


def top_fraction_metrics(y_true: np.ndarray, y_prob: np.ndarray, fractions: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(y_true)
    if n == 0:
        return out
    order = np.argsort(-y_prob)
    for frac in fractions:
        k = max(1, int(math.ceil(float(frac) * n)))
        idx = order[:k]
        key = str(frac).replace(".", "p")
        out[f"top_fraction_{key}_n"] = int(k)
        out[f"top_fraction_{key}_precision"] = float(np.mean(y_true[idx] == 1))
        out[f"top_fraction_{key}_malignant_recall"] = float(np.sum(y_true[idx] == 1) / max(np.sum(y_true == 1), 1))
    return out


def fixed_recall_metrics(y_true: np.ndarray, y_prob: np.ndarray, targets: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(targets) == 0:
        return out

    y_arr = np.asarray(y_true, dtype=int)
    prob_arr = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(prob_arr)
    if not np.all(valid):
        y_arr = y_arr[valid]
        prob_arr = prob_arr[valid]

    n = len(y_arr)
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n == 0 or n_pos == 0:
        return out

    order = np.argsort(-prob_arr, kind="mergesort")
    sorted_prob = prob_arr[order]
    sorted_pos = (y_arr[order] == 1).astype(np.int64)
    is_group_end = np.r_[sorted_prob[1:] != sorted_prob[:-1], True]
    group_end = np.flatnonzero(is_group_end)
    thresholds = sorted_prob[group_end]
    tp = np.cumsum(sorted_pos)[group_end]
    predicted = group_end + 1
    fp = predicted - tp
    tn = n_neg - fp
    recall = tp / float(n_pos)

    for target in targets:
        key = str(target).replace(".", "p")
        idx = int(np.searchsorted(recall, float(target), side="left"))
        if idx >= len(thresholds):
            out[f"fixed_recall_{key}_threshold"] = np.nan
            out[f"fixed_recall_{key}_precision"] = np.nan
            out[f"fixed_recall_{key}_specificity"] = np.nan
            out[f"fixed_recall_{key}_predicted_malignant_fraction"] = np.nan
        else:
            precision = tp[idx] / predicted[idx] if predicted[idx] else np.nan
            specificity = tn[idx] / float(n_neg) if n_neg else np.nan
            out[f"fixed_recall_{key}_threshold"] = float(thresholds[idx])
            out[f"fixed_recall_{key}_precision"] = float(precision) if np.isfinite(precision) else np.nan
            out[f"fixed_recall_{key}_specificity"] = float(specificity) if np.isfinite(specificity) else np.nan
            out[f"fixed_recall_{key}_predicted_malignant_fraction"] = float(predicted[idx] / n)
    return out


def load_stage0_scorecard(experiment_dir: Path, stage0_scorecard: Path | None = None) -> pd.DataFrame:
    path = stage0_scorecard or experiment_dir / "analysis" / "scorecards" / "stage0_mrd_old34_broad_scorecard.csv"
    if not path.is_absolute():
        path = experiment_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 0 scorecard: {path}")
    df = pd.read_csv(path)
    df.insert(0, "source_scorecard_row_id", np.arange(len(df)))
    return df


def load_confusion_diagnostics(experiment_dir: Path) -> pd.DataFrame:
    path = (
        experiment_dir
        / "analysis"
        / "scorecards"
        / "metric_diagnostics_20260525"
        / "stage2_confusion_counts_by_scorecard_row.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def canonicalize_existing_quick_stage2(
    experiment_dir: Path,
    scorecard: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = scorecard.copy()
    out["modeling_goal"] = "quick_sharedness_groupkfold_by_patient"
    out["question_id"] = "Q0_quick_sharedness_groupkfold_by_patient"
    out["stage1_fit_scope_note"] = args.stage1_fit_scope_note
    out["metrics_aggregation"] = "cell_weighted_oof"
    out["evaluation_fit_scope"] = "out_of_fold_by_patient_group"
    out["probability_calibration"] = "raw_logistic_regression_predict_proba"
    out["implementation_note"] = (
        "Canonical copy of existing quick Stage 2 artifacts; no model refitting was performed."
    )
    out_path = analysis_scorecard_root(experiment_dir, args) / "stage2_canonical_quick_groupkfold_scorecard.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    LOGGER.info("Wrote canonical quick scorecard: %s", out_path)
    return out


def merge_quick_with_confusion(scorecard: pd.DataFrame, confusion: pd.DataFrame) -> pd.DataFrame:
    quick = scorecard[scorecard.get("stage2_status", "").eq("ok")].copy()
    if not confusion.empty:
        keep_cols = [
            "scorecard_row_id",
            "malignant_precision",
            "malignant_recall",
            "healthy_recall_specificity",
            "heldout_malignant_correct_tp",
            "heldout_malignant_incorrect_fn",
            "heldout_healthy_correct_tn",
            "heldout_healthy_incorrect_fp",
        ]
        keep_cols = [c for c in keep_cols if c in confusion.columns]
        quick = quick.merge(
            confusion[keep_cols],
            left_on="source_scorecard_row_id",
            right_on="scorecard_row_id",
            how="left",
        )
    if "malignant_precision" not in quick:
        quick["malignant_precision"] = np.nan
    if "malignant_recall" not in quick:
        quick["malignant_recall"] = np.nan
    return quick


def composite_rank_best_rows(quick: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "stage2_auprc",
        "stage2_auroc",
        "stage2_balanced_accuracy",
        "malignant_precision",
        "malignant_recall",
    ]
    for col in metrics:
        if col not in quick:
            quick[col] = np.nan
        quick[f"{col}_rank"] = quick.groupby("stage0_panel_type")[col].rank(ascending=False, method="average")
    quick["quick_composite_rank"] = quick[[f"{col}_rank" for col in metrics]].mean(axis=1, skipna=True)
    best = (
        quick.sort_values(["stage0_panel_id", "quick_composite_rank", "stage2_auprc", "stage2_auroc"], ascending=[True, True, False, False])
        .groupby("stage0_panel_id", as_index=False)
        .head(1)
        .copy()
    )
    return best


def force_include_reason(panel_id: str) -> str:
    text = str(panel_id).lower()
    reasons: list[str] = []
    if "interferon" in text or "ifn" in text:
        reasons.append("interferon_related")
    if "nfkb" in text or "nf_kb" in text or "tnfa" in text or "tnfr" in text:
        reasons.append("nfkb_tnf_related")
    if "antigen" in text or "mhc" in text:
        reasons.append("antigen_presentation_mhc_related")
    if "cytokine" in text or "jak_stat" in text or "il6" in text or "il2" in text:
        reasons.append("cytokine_immune_signaling")
    return ";".join(reasons)


def make_shortlist(experiment_dir: Path, scorecard: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    quick = merge_quick_with_confusion(scorecard, load_confusion_diagnostics(experiment_dir))
    best = composite_rank_best_rows(quick)

    selected = pd.Series(False, index=best.index)
    reasons = pd.Series("", index=best.index, dtype=object)

    controls = best["stage0_panel_id"].isin(
        [
            "full_34",
            "core_only",
            "hvg_top_requested_500__available_500",
            "hvg_top_requested_1000__available_1000",
            "hvg_top_requested_3000__available_3000",
            "hvg_top_requested_10000__available_10000",
        ]
    )
    selected |= controls
    reasons.loc[controls] = "control_panel"

    single = best["stage0_panel_type"].isin(["single_geneset_only", "atomic_sets", "core_anchor_sets"])
    single_order = best.loc[single].sort_values("quick_composite_rank").index.tolist()
    n_single = min(int(args.shortlist_single_geneset_top_n), len(single_order))
    if n_single:
        idx = single_order[:n_single]
        selected.loc[idx] = True
        reasons.loc[idx] = reasons.loc[idx].mask(reasons.loc[idx].eq(""), "top_single_geneset")
        reasons.loc[idx] = reasons.loc[idx].mask(reasons.loc[idx].ne("top_single_geneset"), reasons.loc[idx] + ";top_single_geneset")

    groups = best["stage0_panel_type"].isin(["single_group_only", "family_union_sets"])
    group_order = best.loc[groups].sort_values("quick_composite_rank").index.tolist()
    n_groups = min(int(args.shortlist_group_top_n), len(group_order))
    if n_groups:
        idx = group_order[:n_groups]
        selected.loc[idx] = True
        reasons.loc[idx] = reasons.loc[idx].mask(reasons.loc[idx].eq(""), "top_biology_group")
        reasons.loc[idx] = reasons.loc[idx].mask(reasons.loc[idx].ne("top_biology_group"), reasons.loc[idx] + ";top_biology_group")

    force_reasons = best["stage0_panel_id"].map(force_include_reason)
    force = force_reasons.ne("")
    selected |= force
    reasons.loc[force] = [
        ";".join(x for x in [old, "force_include_biology"] if x)
        for old in reasons.loc[force].astype(str).tolist()
    ]

    out = pd.DataFrame(
        {
            "stage0_panel_id": best["stage0_panel_id"],
            "stage0_panel_type": best["stage0_panel_type"],
            "n_covered_genes": best["n_covered_genes"].astype(int),
            "best_quick_stage1_method": best["stage1_method"],
            "best_quick_requested_k": best["requested_k"],
            "best_quick_representation_family": best["representation_family"],
            "best_quick_auroc": best["stage2_auroc"],
            "best_quick_auprc": best["stage2_auprc"],
            "best_quick_balanced_accuracy": best["stage2_balanced_accuracy"],
            "best_quick_malignant_precision": best["malignant_precision"],
            "best_quick_malignant_recall": best["malignant_recall"],
            "shortlist_reason": reasons,
            "force_include_reason": force_reasons,
            "selected_for_stage2_multiobjective": selected,
            "quick_composite_rank": best["quick_composite_rank"],
            "source_scorecard_row_id": best["source_scorecard_row_id"].astype(int),
            "scores_path": best.get("scores_path", ""),
            "features_path": best.get("features_path", ""),
            "feature_names_path": best.get("feature_names_path", ""),
            "stage1_metadata_path": best.get("stage1_metadata_path", ""),
            "gene_list_path": best.get("gene_list_path", ""),
            "effective_k": best.get("effective_k", np.nan),
            "seed": best.get("seed", 42),
        }
    ).sort_values(["selected_for_stage2_multiobjective", "stage0_panel_type", "quick_composite_rank"], ascending=[False, True, True])

    output = experiment_dir / args.shortlist_output
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    LOGGER.info("Wrote provisional shortlist: %s", output)
    return out


def resolve_selected_specs(experiment_dir: Path, scorecard: pd.DataFrame, shortlist: pd.DataFrame, args: argparse.Namespace) -> list[RepresentationSpec]:
    if args.panel_selection == "all_quick_rows":
        selected = merge_quick_with_confusion(scorecard, load_confusion_diagnostics(experiment_dir))
    elif args.panel_selection == "all_biological_quick_rows":
        selected = merge_quick_with_confusion(scorecard, load_confusion_diagnostics(experiment_dir))
        selected = selected.loc[
            selected["stage0_panel_type"].isin(
                ["single_geneset_only", "single_group_only", "atomic_sets", "family_union_sets", "core_anchor_sets", "leave_one_family_out"]
            )
        ].copy()
    elif args.panel_selection == "shortlist":
        selected_ids = set(shortlist.loc[shortlist["selected_for_stage2_multiobjective"].astype(bool), "source_scorecard_row_id"].astype(int))
        selected = scorecard[scorecard["source_scorecard_row_id"].isin(selected_ids)].copy()
    elif args.panel_selection == "shortlist_plus_controls":
        selected_ids = set(shortlist.loc[shortlist["selected_for_stage2_multiobjective"].astype(bool), "source_scorecard_row_id"].astype(int))
        selected = scorecard[scorecard["source_scorecard_row_id"].isin(selected_ids)].copy()
    else:
        raise ValueError(f"Unknown panel selection: {args.panel_selection}")

    if args.max_selected_representations:
        selected = selected.head(int(args.max_selected_representations)).copy()

    shortlist_by_row = shortlist.set_index("source_scorecard_row_id", drop=False)
    specs: list[RepresentationSpec] = []
    for _, row in selected.iterrows():
        if row.get("stage2_status") != "ok":
            continue
        source_id = int(row["source_scorecard_row_id"])
        sl = shortlist_by_row.loc[source_id] if source_id in shortlist_by_row.index else {}
        specs.append(
            RepresentationSpec(
                source_scorecard_row_id=source_id,
                stage0_panel_id=str(row["stage0_panel_id"]),
                stage0_panel_family=str(row.get("stage0_panel_family", "")),
                stage0_panel_subfamily=str(row.get("stage0_panel_subfamily", "")),
                stage0_panel_type=str(row["stage0_panel_type"]),
                interpretation_layer=str(row.get("interpretation_layer", "")),
                n_covered_genes=int(row["n_covered_genes"]),
                representation_family=str(row["representation_family"]),
                stage1_method=str(row["stage1_method"]),
                requested_k=normalize_requested_k(row.get("requested_k")),
                effective_k=safe_int(row.get("effective_k")),
                seed=safe_int(row.get("seed"), 42) or 42,
                scores_path=str(row.get("scores_path", "")),
                features_path=str(row.get("features_path", "")),
                feature_names_path=str(row.get("feature_names_path", "")),
                stage1_metadata_path=str(row.get("stage1_metadata_path", "")),
                gene_list_path=str(row.get("gene_list_path", "")),
                best_quick_auroc=safe_float(row.get("stage2_auroc")),
                best_quick_auprc=safe_float(row.get("stage2_auprc")),
                best_quick_balanced_accuracy=safe_float(row.get("stage2_balanced_accuracy")),
                best_quick_malignant_precision=safe_float(sl.get("best_quick_malignant_precision", np.nan)),
                best_quick_malignant_recall=safe_float(sl.get("best_quick_malignant_recall", np.nan)),
                shortlist_reason=str(sl.get("shortlist_reason", "")),
                force_include_reason=str(sl.get("force_include_reason", "")),
            )
        )
    return specs


def read_first_prediction_table(experiment_dir: Path, scorecard: pd.DataFrame) -> pd.DataFrame:
    ok = scorecard[scorecard.get("stage2_status", "").eq("ok")]
    for rel in ok.get("stage2_predictions_path", []):
        path = experiment_dir / str(rel)
        if path.exists():
            return pd.read_csv(path, index_col=0)
    raise FileNotFoundError("Could not find any existing quick Stage 2 prediction table to recover obs labels")


def labels_from_predictions(pred: pd.DataFrame, positive_label: str, negative_label: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if "y_true" in pred:
        y = pred["y_true"].astype(int).to_numpy()
    else:
        label_col = "CN.label" if "CN.label" in pred else "label"
        labels = pred[label_col].astype(str)
        y = np.full(len(labels), -1, dtype=int)
        y[labels.eq(str(positive_label)).to_numpy()] = 1
        y[labels.eq(str(negative_label)).to_numpy()] = 0
        if np.any(y < 0):
            raise ValueError("Unexpected labels in prediction table")
    if "patient" not in pred:
        raise ValueError("Prediction table must include a patient column")
    groups = pred["patient"].astype(str).to_numpy()
    obs = pred[[c for c in ["patient", "CN.label"] if c in pred.columns]].copy()
    return y, groups, obs


def load_input_h5ad_path(experiment_dir: Path) -> Path | None:
    config_path = experiment_dir / "experiment_config.yaml"
    if not config_path.exists():
        return None
    for line in config_path.read_text().splitlines():
        if line.startswith("input_h5ad:"):
            value = line.split(":", 1)[1].strip().strip("'\"")
            return Path(value) if value else None
    return None


def build_cell_metadata(experiment_dir: Path, pred: pd.DataFrame, y: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    cell_ids = pred.index.astype(str)
    meta = pd.DataFrame({"cell_id": cell_ids, "y_true": y})
    for col in ["patient", "CN.label"]:
        if col in pred.columns:
            meta[col] = pred[col].to_numpy()

    requested = split_comma_csv(args.discovery_cell_metadata_cols)
    h5ad_path = load_input_h5ad_path(experiment_dir)
    if not requested or h5ad_path is None or not h5ad_path.exists():
        return meta

    try:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path, backed="r")
        obs_cols = [col for col in requested if col in adata.obs.columns and col not in meta.columns]
        if obs_cols:
            available = adata.obs.loc[adata.obs_names.intersection(cell_ids), obs_cols].copy()
            available.insert(0, "cell_id", available.index.astype(str))
            meta = meta.merge(available, on="cell_id", how="left")
    except Exception:
        LOGGER.warning("Could not enrich discovery cell metadata from input AnnData", exc_info=True)
    return meta


def load_feature_names(experiment_dir: Path, spec: RepresentationSpec, n_features: int) -> list[str]:
    if spec.feature_names_path and str(spec.feature_names_path) != "nan":
        path = experiment_dir / spec.feature_names_path
        if path.exists():
            names = json.loads(path.read_text())
            return [str(x) for x in names]
    if spec.stage1_metadata_path:
        meta_path = experiment_dir / spec.stage1_metadata_path
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            feature_names_path = meta.get("feature_names_path")
            if feature_names_path:
                path = experiment_dir / feature_names_path
                if path.exists():
                    return [str(x) for x in json.loads(path.read_text())]
    prefix = "factor" if spec.representation_family == "dr" else "feature"
    return [f"{prefix}_{i + 1:03d}" for i in range(n_features)]


def load_features(experiment_dir: Path, spec: RepresentationSpec) -> tuple[np.ndarray, list[str]]:
    if spec.representation_family == "dr":
        path = experiment_dir / spec.scores_path
        x = np.load(path)
    else:
        path = experiment_dir / spec.features_path
        if path.suffix == ".npz":
            x = load_npz(path).toarray()
        else:
            x = np.load(path)
    x = np.asarray(x, dtype=np.float32)
    return x, load_feature_names(experiment_dir, spec, x.shape[1])


def fit_preprocess_transform(x_train: np.ndarray, x_eval: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    x_train_t = imputer.fit_transform(x_train)
    x_train_t = scaler.fit_transform(x_train_t)
    x_eval_t = None
    if x_eval is not None:
        x_eval_t = scaler.transform(imputer.transform(x_eval))
    return np.asarray(x_train_t, dtype=np.float32), None if x_eval_t is None else np.asarray(x_eval_t, dtype=np.float32)


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    setting: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    l1_ratio = setting.get("l1_ratio")
    if setting["penalty"] != "elasticnet" or pd.isna(l1_ratio):
        l1_ratio = None
    clf = make_logistic_regression(
        penalty=str(setting["penalty"]),
        C=float(setting["C"]),
        l1_ratio=l1_ratio,
        random_state=int(args.seed),
        max_iter=int(args.max_iter),
        n_jobs=int(args.n_jobs),
        backend=backend_alias(args.backend),
        strict_gpu=bool(args.strict_gpu),
        class_weight=args.class_weight,
    )
    clf.fit(x_train, y_train)
    return clf


def validate_shard_args(args: argparse.Namespace) -> None:
    if int(args.shard_count) < 1:
        raise ValueError("--shard-count must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.shard_count):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")
    if int(args.shard_count) > 1 and not args.stage2_run_id and not args.launch_gpu_shards:
        raise ValueError("--stage2-run-id is required when --shard-count > 1")


def apply_spec_shard(specs: list[RepresentationSpec], args: argparse.Namespace) -> list[RepresentationSpec]:
    if int(args.shard_count) <= 1:
        return specs
    shard_index = int(args.shard_index)
    shard_count = int(args.shard_count)
    sharded = [spec for idx, spec in enumerate(specs) if idx % shard_count == shard_index]
    LOGGER.info(
        "Selected %d/%d representations for shard %d/%d",
        len(sharded),
        len(specs),
        shard_index,
        shard_count,
    )
    return sharded


def safe_branch(value: str | None) -> str:
    if not value:
        return ""
    out = str(value).lower()
    for old, new in [(" ", "_"), ("/", "_"), ("-", "_"), (".", "_"), (":", "_")]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def stage2_base_root(experiment_dir: Path, branch_name: str = "") -> Path:
    root = experiment_dir / "stage2_supervised" / "multiobjective"
    branch = safe_branch(branch_name)
    return root / branch if branch else root


def stage2_run_root(experiment_dir: Path, run_id: str, branch_name: str = "") -> Path:
    return stage2_base_root(experiment_dir, branch_name) / "runs" / run_id


def analysis_scorecard_root(experiment_dir: Path, args: argparse.Namespace) -> Path:
    branch = safe_branch(getattr(args, "stage2_output_branch", ""))
    root = experiment_dir / "analysis" / "scorecards"
    return root / branch if branch else root


def stage2_output_root(experiment_dir: Path, args: argparse.Namespace) -> Path:
    if args.stage2_run_id:
        return (
            stage2_run_root(experiment_dir, args.stage2_run_id, getattr(args, "stage2_output_branch", ""))
            / "shards"
            / f"shard_{int(args.shard_index):03d}_of_{int(args.shard_count):03d}"
        )
    return stage2_base_root(experiment_dir, getattr(args, "stage2_output_branch", ""))


def stage2_scorecard_dir(experiment_dir: Path, args: argparse.Namespace) -> Path:
    if args.stage2_run_id:
        return stage2_output_root(experiment_dir, args) / "scorecards"
    return analysis_scorecard_root(experiment_dir, args)


def goal_artifact_dir(experiment_dir: Path, args: argparse.Namespace, goal: str) -> Path:
    return stage2_output_root(experiment_dir, args) / goal


def base_spec_row(spec: RepresentationSpec, args: argparse.Namespace) -> dict[str, Any]:
    row = {
        "source_scorecard_row_id": spec.source_scorecard_row_id,
        "stage0_panel_id": spec.stage0_panel_id,
        "stage0_panel_family": spec.stage0_panel_family,
        "stage0_panel_subfamily": spec.stage0_panel_subfamily,
        "stage0_panel_type": spec.stage0_panel_type,
        "interpretation_layer": spec.interpretation_layer,
        "n_covered_genes": spec.n_covered_genes,
        "representation_family": spec.representation_family,
        "stage1_method": spec.stage1_method,
        "requested_k": spec.requested_k,
        "effective_k": spec.effective_k,
        "seed": spec.seed,
        "stage1_scope": "across_patient",
        "stage1_fit_scope_note": args.stage1_fit_scope_note,
        "shortlist_reason": spec.shortlist_reason,
        "force_include_reason": spec.force_include_reason,
        "best_quick_auroc": spec.best_quick_auroc,
        "best_quick_auprc": spec.best_quick_auprc,
        "best_quick_balanced_accuracy": spec.best_quick_balanced_accuracy,
    }
    if getattr(args, "stage2_run_id", None):
        row.update(
            {
                "stage2_run_id": args.stage2_run_id,
                "shard_index": int(args.shard_index),
                "shard_count": int(args.shard_count),
            }
        )
    return row


def selected_feature_ids(coef: np.ndarray, feature_names: list[str], tol: float) -> str:
    selected = [feature_names[i] for i, value in enumerate(coef) if abs(float(value)) > tol]
    return ";".join(selected)


def write_coefficient_annotation_summary(coef_rows: list[dict[str, Any]], out_dir: Path) -> str:
    if not coef_rows:
        return ""
    coef = pd.DataFrame(coef_rows)
    for col in ["stage0_panel_family", "stage0_panel_subfamily", "interpretation_layer"]:
        if col not in coef:
            coef[col] = ""
    coef["abs_coefficient"] = coef["coefficient"].abs()
    group_cols = [
        col
        for col in [
            "modeling_goal",
            "stage0_panel_family",
            "stage0_panel_subfamily",
            "interpretation_layer",
            "stage0_panel_type",
            "penalty",
            "C",
            "l1_ratio",
        ]
        if col in coef.columns
    ]
    summary = (
        coef.groupby(group_cols, dropna=False)
        .agg(
            n_feature_coefficients=("coefficient", "size"),
            n_nonzero_coefficients=("is_nonzero", "sum"),
            coefficient_l1_sum=("abs_coefficient", "sum"),
            coefficient_abs_mean=("abs_coefficient", "mean"),
        )
        .reset_index()
        .sort_values(["modeling_goal", "coefficient_l1_sum"], ascending=[True, False])
    )
    path = out_dir / "coefficient_summary_by_panel_annotation.csv"
    summary.to_csv(path, index=False)
    return str(path)


def run_discovery_full_cohort(
    experiment_dir: Path,
    specs: list[RepresentationSpec],
    y: np.ndarray,
    cell_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    prediction_rows: list[np.ndarray] = []
    prediction_fit_rows: list[dict[str, Any]] = []
    grid = make_model_grid(args)
    thresholds = split_csv(args.decision_thresholds, float) or [0.5]
    top_fractions = split_csv(args.top_fraction_thresholds, float)
    fixed_recalls = split_csv(args.fixed_recall_targets, float)

    for spec_i, spec in enumerate(specs, start=1):
        LOGGER.info("Discovery full-cohort [%d/%d]: %s", spec_i, len(specs), spec.stage0_panel_id)
        x, feature_names = load_features(experiment_dir, spec)
        x_train, _ = fit_preprocess_transform(x)
        for setting in grid:
            try:
                clf = fit_model(x_train, y, setting, args)
                y_prob = clf.predict_proba(x_train)[:, 1]
                coef = np.asarray(clf.coef_).ravel()
                fit_id = (
                    f"row={spec.source_scorecard_row_id}|panel={spec.stage0_panel_id}|"
                    f"method={spec.stage1_method}|k={spec.requested_k}|penalty={setting['penalty']}|"
                    f"C={float(setting['C']):.12g}|l1_ratio={setting.get('l1_ratio', np.nan)}"
                )
                if args.save_discovery_cell_predictions:
                    prediction_rows.append(np.asarray(y_prob, dtype=args.discovery_cell_prediction_dtype))
                    prediction_fit_rows.append(
                        {
                            **base_spec_row(spec, args),
                            "fit_id": fit_id,
                            "modeling_goal": "discovery_full_cohort_fit",
                            "evaluation_fit_scope": "apparent_in_sample_cell_predictions",
                            "classifier": "LogisticRegression",
                            "penalty": setting["penalty"],
                            "C": float(setting["C"]),
                            "l1_ratio": setting.get("l1_ratio", np.nan),
                            "class_weight": args.class_weight,
                            "ml_backend_requested": args.backend,
                            "ml_backend_used": clf.backend_used,
                            "n_cells": int(len(y_prob)),
                            "prediction_dtype": args.discovery_cell_prediction_dtype,
                        }
                    )
                for threshold in thresholds:
                    metrics = binary_metrics(y, y_prob, threshold)
                    row = {
                        **base_spec_row(spec, args),
                        "modeling_goal": "discovery_full_cohort_fit",
                        "evaluation_fit_scope": "apparent_in_sample",
                        "metrics_aggregation": "cell_weighted_apparent",
                        "classifier": "LogisticRegression",
                        "penalty": setting["penalty"],
                        "C": float(setting["C"]),
                        "l1_ratio": setting.get("l1_ratio", np.nan),
                        "fit_id": fit_id,
                        "class_weight": args.class_weight,
                        "threshold": threshold,
                        "ml_backend_requested": args.backend,
                        "ml_backend_used": clf.backend_used,
                        "n_features": int(x.shape[1]),
                        "nonzero_coefficient_count": int(np.sum(np.abs(coef) > float(args.coef_nonzero_tol))),
                        "coef_l1_norm": float(np.sum(np.abs(coef))),
                        "selected_feature_ids": selected_feature_ids(coef, feature_names, float(args.coef_nonzero_tol))
                        if setting["penalty"] in {"l1", "elasticnet"}
                        else "",
                        **{f"stage2_{k}": v for k, v in metrics.items()},
                        **top_fraction_metrics(y, y_prob, top_fractions),
                        **fixed_recall_metrics(y, y_prob, fixed_recalls),
                        "status": "ok",
                    }
                    rows.append(row)
                for feature, value in zip(feature_names, coef):
                    coef_rows.append(
                        {
                            **base_spec_row(spec, args),
                            "modeling_goal": "discovery_full_cohort_fit",
                            "penalty": setting["penalty"],
                            "C": float(setting["C"]),
                            "l1_ratio": setting.get("l1_ratio", np.nan),
                            "feature_id": feature,
                            "coefficient": float(value),
                            "is_nonzero": bool(abs(float(value)) > float(args.coef_nonzero_tol)),
                        }
                    )
            except Exception as exc:
                LOGGER.exception("Discovery failed for %s %s", spec.stage0_panel_id, setting)
                rows.append(
                    {
                        **base_spec_row(spec, args),
                        "modeling_goal": "discovery_full_cohort_fit",
                        "penalty": setting["penalty"],
                        "C": float(setting["C"]),
                        "l1_ratio": setting.get("l1_ratio", np.nan),
                        "status": "failed",
                        "reason": repr(exc),
                    }
                )

    out_dir = goal_artifact_dir(experiment_dir, args, "discovery_full_cohort")
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_path = out_dir / "coefficient_paths.csv"
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    coef_summary_path = write_coefficient_annotation_summary(coef_rows, out_dir)
    prediction_matrix_path = out_dir / "cell_prediction_matrix.npz"
    prediction_fit_path = out_dir / "cell_prediction_fit_metadata.csv"
    cell_metadata_path = out_dir / "cell_metadata.csv.gz"
    if args.save_discovery_cell_predictions:
        if prediction_rows:
            y_prob_matrix = np.vstack(prediction_rows).astype(args.discovery_cell_prediction_dtype, copy=False)
        else:
            y_prob_matrix = np.empty((0, len(cell_metadata)), dtype=args.discovery_cell_prediction_dtype)
        np.savez_compressed(
            prediction_matrix_path,
            y_prob=y_prob_matrix,
            fit_id=np.asarray([row["fit_id"] for row in prediction_fit_rows], dtype=np.str_),
            cell_id=np.asarray(cell_metadata["cell_id"].astype(str).tolist(), dtype=np.str_),
            y_true=np.asarray(y, dtype=np.int8),
        )
        pd.DataFrame(prediction_fit_rows).to_csv(prediction_fit_path, index=False)
        cell_metadata.to_csv(cell_metadata_path, index=False, compression="gzip")
    scorecard = pd.DataFrame(rows)
    scorecard["coefficient_path"] = str(coef_path.relative_to(experiment_dir))
    if args.save_discovery_cell_predictions:
        scorecard["cell_prediction_matrix_path"] = str(prediction_matrix_path.relative_to(experiment_dir))
        scorecard["cell_prediction_fit_metadata_path"] = str(prediction_fit_path.relative_to(experiment_dir))
        scorecard["cell_metadata_path"] = str(cell_metadata_path.relative_to(experiment_dir))
    scorecard_path = stage2_scorecard_dir(experiment_dir, args) / "stage2_discovery_full_cohort_scorecard.csv"
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    if coef_summary_path:
        scorecard["coefficient_summary_by_panel_annotation_path"] = str(Path(coef_summary_path).relative_to(experiment_dir))
    scorecard.to_csv(scorecard_path, index=False)

    if not scorecard.empty and "status" in scorecard:
        ok = scorecard[scorecard["status"].eq("ok")].copy()
        if not ok.empty:
            best_parts: list[pd.DataFrame] = []
            for metric in ["stage2_auprc", "stage2_balanced_accuracy"]:
                best_parts.append(
                    ok.sort_values(metric, ascending=False)
                    .groupby(["stage0_panel_id", "representation_family"], as_index=False)
                    .head(1)
                    .assign(best_by=metric)
                )
            sparse = ok.copy()
            sparse["sparsity_aware_score"] = sparse["stage2_auprc"] / np.log1p(sparse["nonzero_coefficient_count"].clip(lower=1))
            best_parts.append(
                sparse.sort_values("sparsity_aware_score", ascending=False)
                .groupby(["stage0_panel_id", "representation_family"], as_index=False)
                .head(1)
                .assign(best_by="sparsity_aware_score")
            )
            pd.concat(best_parts, ignore_index=True).to_csv(out_dir / "best_rows.csv", index=False)
    LOGGER.info("Wrote discovery scorecard: %s", scorecard_path)
    return scorecard


def patient_support_table(y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for patient in sorted(np.unique(groups)):
        mask = groups == patient
        rows.append(
            {
                "patient": patient,
                "n_cells": int(mask.sum()),
                "n_malignant": int((y[mask] == 1).sum()),
                "n_normal": int((y[mask] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def lopo_fold_fit_id(spec: RepresentationSpec, setting: dict[str, Any], heldout_patient: str) -> str:
    return (
        f"row={spec.source_scorecard_row_id}|panel={spec.stage0_panel_id}|"
        f"method={spec.stage1_method}|k={spec.requested_k}|heldout={heldout_patient}|"
        f"penalty={setting['penalty']}|C={float(setting['C']):.12g}|"
        f"l1_ratio={setting.get('l1_ratio', np.nan)}"
    )


def lopo_setting_fit_id(spec: RepresentationSpec, setting: dict[str, Any]) -> str:
    return (
        f"row={spec.source_scorecard_row_id}|panel={spec.stage0_panel_id}|"
        f"method={spec.stage1_method}|k={spec.requested_k}|"
        f"penalty={setting['penalty']}|C={float(setting['C']):.12g}|"
        f"l1_ratio={setting.get('l1_ratio', np.nan)}"
    )


def run_lopo_sharedness(
    experiment_dir: Path,
    specs: list[RepresentationSpec],
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    thresholds = split_csv(args.decision_thresholds, float) or [0.5]
    fixed_recalls = split_csv(args.fixed_recall_targets, float)
    grid = make_model_grid(args)
    rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    predictions_rows: list[pd.DataFrame] = []
    patients = sorted(np.unique(groups))

    for spec_i, spec in enumerate(specs, start=1):
        LOGGER.info(
            "LOPO sharedness [%d/%d]: %s (%d regularization settings)",
            spec_i,
            len(specs),
            spec.stage0_panel_id,
            len(grid),
        )
        x, feature_names = load_features(experiment_dir, spec)
        for setting in grid:
            y_prob = np.full(len(y), np.nan, dtype=float)
            backend_used = ""
            fold_nonzero_counts: list[int] = []
            fold_coef_l1_norms: list[float] = []
            setting_patient_rows: list[dict[str, Any]] = []

            for patient in patients:
                test_mask = groups == patient
                train_mask = ~test_mask
                if len(np.unique(y[train_mask])) < 2:
                    continue
                x_train, x_test = fit_preprocess_transform(x[train_mask], x[test_mask])
                clf = fit_model(x_train, y[train_mask], setting, args)
                backend_used = clf.backend_used
                y_prob[test_mask] = clf.predict_proba(x_test)[:, 1]
                coef = np.asarray(clf.coef_).ravel()
                fold_nonzero = int(np.sum(np.abs(coef) > float(args.coef_nonzero_tol)))
                fold_nonzero_counts.append(fold_nonzero)
                fold_coef_l1_norms.append(float(np.sum(np.abs(coef))))
                fold_fit_id = lopo_fold_fit_id(spec, setting, str(patient))

                if args.save_lopo_coefficients:
                    for feature, value in zip(feature_names, coef):
                        coef_rows.append(
                            {
                                **base_spec_row(spec, args),
                                "modeling_goal": "sharedness_leave_patient_out",
                                "fit_id": fold_fit_id,
                                "heldout_patient": patient,
                                "penalty": setting["penalty"],
                                "C": float(setting["C"]),
                                "l1_ratio": setting.get("l1_ratio", np.nan),
                                "feature_id": feature,
                                "coefficient": float(value),
                                "is_nonzero": bool(abs(float(value)) > float(args.coef_nonzero_tol)),
                            }
                        )

                mask = test_mask & np.isfinite(y_prob)
                y_p = y[mask]
                prob_p = y_prob[mask]
                n_mal = int((y_p == 1).sum())
                n_norm = int((y_p == 0).sum())
                for threshold in thresholds:
                    metrics = binary_metrics(y_p, prob_p, threshold) if len(y_p) else {}
                    setting_patient_rows.append(
                        {
                            **base_spec_row(spec, args),
                            "fit_id": fold_fit_id,
                            "modeling_goal": "sharedness_leave_patient_out",
                            "patient": patient,
                            "heldout_patient": patient,
                            "classifier": "LogisticRegression",
                            "penalty": setting["penalty"],
                            "C": float(setting["C"]),
                            "l1_ratio": setting.get("l1_ratio", np.nan),
                            "threshold": threshold,
                            "n_malignant": n_mal,
                            "n_normal": n_norm,
                            "has_both_classes": bool(n_mal > 0 and n_norm > 0),
                            "normal_only": bool(n_mal == 0 and n_norm > 0),
                            "low_malignant_support": bool(0 < n_mal < int(args.low_malignant_support_threshold)),
                            "ml_backend_requested": args.backend,
                            "ml_backend_used": backend_used,
                            "nonzero_coefficient_count": fold_nonzero,
                            "coef_l1_norm": float(np.sum(np.abs(coef))),
                            "selected_feature_ids": selected_feature_ids(coef, feature_names, float(args.coef_nonzero_tol))
                            if setting["penalty"] in {"l1", "elasticnet"}
                            else "",
                            **{f"stage2_{k}": v for k, v in metrics.items()},
                            **fixed_recall_metrics(y_p, prob_p, fixed_recalls),
                        }
                    )

            patient_rows.extend(setting_patient_rows)
            valid = np.isfinite(y_prob)
            if args.save_lopo_cell_predictions:
                pred_df = pd.DataFrame(
                    {
                        "fit_id": lopo_setting_fit_id(spec, setting),
                        "patient": groups,
                        "y_true": y,
                        "y_prob": y_prob,
                        "included_in_metric": valid,
                        "stage0_panel_id": spec.stage0_panel_id,
                        "representation_family": spec.representation_family,
                        "stage1_method": spec.stage1_method,
                        "requested_k": spec.requested_k,
                        "penalty": setting["penalty"],
                        "C": setting["C"],
                        "l1_ratio": setting["l1_ratio"],
                    }
                )
                predictions_rows.append(pred_df)

            setting_fit_id = lopo_setting_fit_id(spec, setting)
            for threshold in thresholds:
                cell_metrics = binary_metrics(y[valid], y_prob[valid], threshold) if valid.any() else {}
                evaluable = pd.DataFrame(setting_patient_rows)
                if not evaluable.empty:
                    evaluable = evaluable.loc[evaluable["threshold"].eq(threshold) & evaluable["has_both_classes"].astype(bool)]
                patient_equal = {
                    f"patient_equal_{metric}": float(evaluable[f"stage2_{metric}"].mean())
                    for metric in ["auroc", "auprc", "balanced_accuracy", "malignant_precision", "malignant_recall", "healthy_recall_specificity"]
                    if not evaluable.empty and f"stage2_{metric}" in evaluable
                }
                rows.append(
                    {
                        **base_spec_row(spec, args),
                        "fit_id": setting_fit_id,
                        "modeling_goal": "sharedness_leave_patient_out",
                        "evaluation_fit_scope": "leave_one_patient_out_on_transductive_stage1_basis",
                        "metrics_aggregation": "cell_weighted_and_patient_equal",
                        "classifier": "LogisticRegression",
                        "penalty": setting["penalty"],
                        "C": float(setting["C"]),
                        "l1_ratio": setting.get("l1_ratio", np.nan),
                        "threshold": threshold,
                        "class_weight": args.class_weight,
                        "ml_backend_requested": args.backend,
                        "ml_backend_used": backend_used,
                        "n_features": int(x.shape[1]),
                        "nonzero_coefficient_count_mean_across_folds": float(np.mean(fold_nonzero_counts))
                        if fold_nonzero_counts
                        else np.nan,
                        "nonzero_coefficient_count_max_across_folds": int(np.max(fold_nonzero_counts))
                        if fold_nonzero_counts
                        else 0,
                        "coef_l1_norm_mean_across_folds": float(np.mean(fold_coef_l1_norms)) if fold_coef_l1_norms else np.nan,
                        "n_valid_cells": int(valid.sum()),
                        "n_patients": int(len(patients)),
                        "n_evaluable_patients_with_both_classes": int(evaluable["patient"].nunique()) if not evaluable.empty else 0,
                        **{f"cell_weighted_{k}": v for k, v in cell_metrics.items()},
                        **(
                            {f"cell_weighted_{k}": v for k, v in fixed_recall_metrics(y[valid], y_prob[valid], fixed_recalls).items()}
                            if valid.any()
                            else {}
                        ),
                        **patient_equal,
                        "status": "ok" if valid.any() else "skipped_no_predictions",
                    }
                )

    out_dir = goal_artifact_dir(experiment_dir, args, "sharedness_lopo")
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_path = out_dir / "by_heldout_patient.csv"
    pred_path = out_dir / "predictions.csv"
    coef_path = out_dir / "coefficient_paths.csv"
    pd.DataFrame(patient_rows).to_csv(patient_path, index=False)
    if args.save_lopo_coefficients:
        pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    coef_summary_path = write_coefficient_annotation_summary(coef_rows, out_dir) if args.save_lopo_coefficients else ""
    if predictions_rows:
        pd.concat(predictions_rows, ignore_index=True).to_csv(pred_path, index=False)
    scorecard = pd.DataFrame(rows)
    scorecard["by_heldout_patient_path"] = str(patient_path.relative_to(experiment_dir))
    if args.save_lopo_coefficients and coef_rows:
        scorecard["coefficient_path"] = str(coef_path.relative_to(experiment_dir))
    if coef_summary_path:
        scorecard["coefficient_summary_by_panel_annotation_path"] = str(Path(coef_summary_path).relative_to(experiment_dir))
    if args.save_lopo_cell_predictions and predictions_rows:
        scorecard["predictions_path"] = str(pred_path.relative_to(experiment_dir))
    scorecard_path = stage2_scorecard_dir(experiment_dir, args) / "stage2_sharedness_lopo_scorecard.csv"
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(scorecard_path, index=False)
    LOGGER.info("Wrote LOPO scorecard: %s", scorecard_path)
    return scorecard


def run_patient_specific(
    experiment_dir: Path,
    specs: list[RepresentationSpec],
    y: np.ndarray,
    groups: np.ndarray,
    cell_metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    thresholds = split_csv(args.decision_thresholds, float) or [0.5]
    fixed_recalls = split_csv(args.fixed_recall_targets, float)
    grid = make_model_grid(args)
    rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    prediction_values: list[np.ndarray] = []
    prediction_cell_indices: list[np.ndarray] = []
    prediction_fit_rows: list[dict[str, Any]] = []

    support = patient_support_table(y, groups)
    patients = sorted(support.loc[(support["n_malignant"] > 0) & (support["n_normal"] > 0), "patient"].astype(str))
    if args.max_patient_specific_patients:
        patients = patients[: int(args.max_patient_specific_patients)]

    for spec_i, spec in enumerate(specs, start=1):
        LOGGER.info("Patient-specific [%d/%d]: %s", spec_i, len(specs), spec.stage0_panel_id)
        x, feature_names = load_features(experiment_dir, spec)
        for patient in patients:
            mask = groups == patient
            cell_indices = np.flatnonzero(mask).astype(np.int64)
            y_p = y[mask]
            n_mal = int((y_p == 1).sum())
            n_norm = int((y_p == 0).sum())
            if n_mal == 0 or n_norm == 0:
                rows.append(
                    {
                        **base_spec_row(spec, args),
                        "modeling_goal": "patient_specific",
                        "patient": patient,
                        "status": "skipped_one_class",
                        "n_malignant": n_mal,
                        "n_normal": n_norm,
                    }
                )
                continue
            x_p, _ = fit_preprocess_transform(x[mask])
            for setting in grid:
                try:
                    clf = fit_model(x_p, y_p, setting, args)
                    y_prob = clf.predict_proba(x_p)[:, 1]
                    coef = np.asarray(clf.coef_).ravel()
                    fit_id = (
                        f"{spec.stage0_panel_id}|family={spec.representation_family}|"
                        f"method={spec.stage1_method}|k={spec.requested_k}|patient={patient}|"
                        f"penalty={setting['penalty']}|C={float(setting['C']):.12g}|"
                        f"l1_ratio={setting.get('l1_ratio', np.nan)}"
                    )
                    if args.save_patient_specific_cell_predictions:
                        prediction_values.append(np.asarray(y_prob, dtype=args.patient_specific_cell_prediction_dtype))
                        prediction_cell_indices.append(cell_indices)
                        prediction_fit_rows.append(
                            {
                                **base_spec_row(spec, args),
                                "fit_id": fit_id,
                                "modeling_goal": "patient_specific",
                                "evaluation_fit_scope": "apparent_within_patient_cell_predictions",
                                "patient": patient,
                                "classifier": "LogisticRegression",
                                "penalty": setting["penalty"],
                                "C": float(setting["C"]),
                                "l1_ratio": setting.get("l1_ratio", np.nan),
                                "class_weight": args.class_weight,
                                "ml_backend_requested": args.backend,
                                "ml_backend_used": clf.backend_used,
                                "n_cells": int(len(y_prob)),
                                "n_malignant": n_mal,
                                "n_normal": n_norm,
                                "prediction_dtype": args.patient_specific_cell_prediction_dtype,
                            }
                        )
                    for threshold in thresholds:
                        metrics = binary_metrics(y_p, y_prob, threshold)
                        rows.append(
                            {
                                **base_spec_row(spec, args),
                                "fit_id": fit_id,
                                "modeling_goal": "patient_specific",
                                "evaluation_fit_scope": "apparent_within_patient",
                                "metrics_aggregation": "cell_weighted_apparent_within_patient",
                                "patient": patient,
                                "classifier": "LogisticRegression",
                                "penalty": setting["penalty"],
                                "C": float(setting["C"]),
                                "l1_ratio": setting.get("l1_ratio", np.nan),
                                "threshold": threshold,
                                "n_malignant": n_mal,
                                "n_normal": n_norm,
                                "low_malignant_support": bool(n_mal < int(args.low_malignant_support_threshold)),
                                "skip_within_patient_cv_reason": "not_implemented_first_pass",
                                "ml_backend_requested": args.backend,
                                "ml_backend_used": clf.backend_used,
                                "nonzero_coefficient_count": int(np.sum(np.abs(coef) > float(args.coef_nonzero_tol))),
                                "coef_l1_norm": float(np.sum(np.abs(coef))),
                                "selected_feature_ids": selected_feature_ids(coef, feature_names, float(args.coef_nonzero_tol))
                                if setting["penalty"] in {"l1", "elasticnet"}
                                else "",
                                **{f"stage2_{k}": v for k, v in metrics.items()},
                                **fixed_recall_metrics(y_p, y_prob, fixed_recalls),
                                "status": "ok",
                            }
                        )
                    for feature, value in zip(feature_names, coef):
                        coef_rows.append(
                            {
                                **base_spec_row(spec, args),
                                "modeling_goal": "patient_specific",
                                "patient": patient,
                                "penalty": setting["penalty"],
                                "C": float(setting["C"]),
                                "l1_ratio": setting.get("l1_ratio", np.nan),
                                "feature_id": feature,
                                "coefficient": float(value),
                                "is_nonzero": bool(abs(float(value)) > float(args.coef_nonzero_tol)),
                            }
                        )
                except Exception as exc:
                    LOGGER.exception("Patient-specific failed for %s %s %s", spec.stage0_panel_id, patient, setting)
                    rows.append(
                        {
                            **base_spec_row(spec, args),
                            "modeling_goal": "patient_specific",
                            "patient": patient,
                            "penalty": setting["penalty"],
                            "C": float(setting["C"]),
                            "l1_ratio": setting.get("l1_ratio", np.nan),
                            "status": "failed",
                            "reason": repr(exc),
                            "n_malignant": n_mal,
                            "n_normal": n_norm,
                        }
                    )

    out_dir = goal_artifact_dir(experiment_dir, args, "patient_specific")
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_path = out_dir / "coefficient_paths.csv"
    pd.DataFrame(coef_rows).to_csv(coef_path, index=False)
    coef_summary_path = write_coefficient_annotation_summary(coef_rows, out_dir)
    prediction_bundle_path = out_dir / "cell_prediction_bundle.npz"
    prediction_fit_path = out_dir / "cell_prediction_fit_metadata.csv"
    cell_metadata_path = out_dir / "cell_metadata.csv.gz"
    if args.save_patient_specific_cell_predictions:
        prediction_lengths = [len(values) for values in prediction_values]
        prediction_indptr = np.r_[0, np.cumsum(prediction_lengths, dtype=np.int64)]
        y_prob_values = (
            np.concatenate(prediction_values).astype(args.patient_specific_cell_prediction_dtype, copy=False)
            if prediction_values
            else np.empty(0, dtype=args.patient_specific_cell_prediction_dtype)
        )
        cell_index_values = np.concatenate(prediction_cell_indices) if prediction_cell_indices else np.empty(0, dtype=np.int64)
        np.savez_compressed(
            prediction_bundle_path,
            y_prob=y_prob_values,
            cell_index=cell_index_values,
            fit_indptr=prediction_indptr,
            fit_id=np.asarray([row["fit_id"] for row in prediction_fit_rows], dtype=np.str_),
            cell_id=np.asarray(cell_metadata["cell_id"].astype(str).tolist(), dtype=np.str_),
            y_true=np.asarray(y, dtype=np.int8),
        )
        pd.DataFrame(prediction_fit_rows).to_csv(prediction_fit_path, index=False)
        cell_metadata.to_csv(cell_metadata_path, index=False, compression="gzip")
    scorecard = pd.DataFrame(rows)
    scorecard["coefficient_path"] = str(coef_path.relative_to(experiment_dir))
    if coef_summary_path:
        scorecard["coefficient_summary_by_panel_annotation_path"] = str(Path(coef_summary_path).relative_to(experiment_dir))
    if args.save_patient_specific_cell_predictions:
        scorecard["cell_prediction_bundle_path"] = str(prediction_bundle_path.relative_to(experiment_dir))
        scorecard["cell_prediction_fit_metadata_path"] = str(prediction_fit_path.relative_to(experiment_dir))
        scorecard["cell_metadata_path"] = str(cell_metadata_path.relative_to(experiment_dir))
    scorecard_path = stage2_scorecard_dir(experiment_dir, args) / "stage2_patient_specific_scorecard.csv"
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(scorecard_path, index=False)
    LOGGER.info("Wrote patient-specific scorecard: %s", scorecard_path)
    return scorecard


def read_csv_many(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                LOGGER.warning("Skipping empty CSV during merge: %s", path)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_merged_csv(df: pd.DataFrame, run_path: Path, canonical_path: Path) -> int:
    if df.empty and len(df.columns) == 0:
        LOGGER.info("No shard CSVs found for merge target %s; leaving canonical output untouched", canonical_path)
        return 0
    run_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path, index=False)
    df.to_csv(canonical_path, index=False)
    LOGGER.info("Wrote merged CSV: %s and %s", run_path, canonical_path)
    return int(len(df))


def merge_stage2_run(experiment_dir: Path, run_id: str, branch_name: str = "") -> dict[str, Any]:
    run_root = stage2_run_root(experiment_dir, run_id, branch_name)
    merged_root = run_root / "merged"
    outputs: dict[str, Any] = {"stage2_run_id": run_id, "run_root": str(run_root.relative_to(experiment_dir))}
    if not run_root.exists():
        raise FileNotFoundError(f"Missing Stage 2 run root: {run_root}")

    scorecard_root = (experiment_dir / "analysis" / "scorecards" / safe_branch(branch_name)) if safe_branch(branch_name) else (experiment_dir / "analysis" / "scorecards")
    canonical_root = stage2_base_root(experiment_dir, branch_name)

    discovery_coef = read_csv_many(list(run_root.glob("shards/shard_*/discovery_full_cohort/coefficient_paths.csv")))
    discovery_coef_canonical = canonical_root / "discovery_full_cohort" / "coefficient_paths.csv"
    outputs["discovery_coefficient_rows"] = write_merged_csv(
        discovery_coef,
        merged_root / "discovery_full_cohort" / "coefficient_paths.csv",
        discovery_coef_canonical,
    )
    discovery_fit_meta = read_csv_many(list(run_root.glob("shards/shard_*/discovery_full_cohort/cell_prediction_fit_metadata.csv")))
    discovery_fit_meta_canonical = canonical_root / "discovery_full_cohort" / "cell_prediction_fit_metadata.csv"
    outputs["discovery_cell_prediction_fit_rows"] = write_merged_csv(
        discovery_fit_meta,
        merged_root / "discovery_full_cohort" / "cell_prediction_fit_metadata.csv",
        discovery_fit_meta_canonical,
    )
    matrix_paths = sorted(run_root.glob("shards/shard_*/discovery_full_cohort/cell_prediction_matrix.npz"))
    if matrix_paths:
        matrices: list[np.ndarray] = []
        fit_ids: list[np.ndarray] = []
        cell_ids: np.ndarray | None = None
        y_true_values: np.ndarray | None = None
        for path in matrix_paths:
            data = np.load(path)
            matrices.append(data["y_prob"])
            fit_ids.append(data["fit_id"].astype(str))
            if cell_ids is None:
                cell_ids = data["cell_id"].astype(str)
                y_true_values = data["y_true"]
            elif not np.array_equal(cell_ids, data["cell_id"].astype(str)):
                raise ValueError(f"Cell order mismatch while merging prediction matrix: {path}")
        merged_matrix = np.vstack(matrices)
        merged_fit_ids = np.concatenate(fit_ids)
        prediction_matrix_run = merged_root / "discovery_full_cohort" / "cell_prediction_matrix.npz"
        prediction_matrix_canonical = canonical_root / "discovery_full_cohort" / "cell_prediction_matrix.npz"
        prediction_matrix_run.parent.mkdir(parents=True, exist_ok=True)
        prediction_matrix_canonical.parent.mkdir(parents=True, exist_ok=True)
        for path in [prediction_matrix_run, prediction_matrix_canonical]:
            np.savez_compressed(
                path,
                y_prob=merged_matrix,
                fit_id=merged_fit_ids,
                cell_id=cell_ids,
                y_true=y_true_values,
            )
        outputs["discovery_cell_prediction_matrix_shape"] = list(merged_matrix.shape)
    cell_metadata_paths = sorted(run_root.glob("shards/shard_*/discovery_full_cohort/cell_metadata.csv.gz"))
    if cell_metadata_paths:
        cell_metadata = pd.read_csv(cell_metadata_paths[0])
        cell_metadata_run = merged_root / "discovery_full_cohort" / "cell_metadata.csv.gz"
        cell_metadata_canonical = canonical_root / "discovery_full_cohort" / "cell_metadata.csv.gz"
        cell_metadata_run.parent.mkdir(parents=True, exist_ok=True)
        cell_metadata_canonical.parent.mkdir(parents=True, exist_ok=True)
        cell_metadata.to_csv(cell_metadata_run, index=False, compression="gzip")
        cell_metadata.to_csv(cell_metadata_canonical, index=False, compression="gzip")
        outputs["discovery_cell_metadata_rows"] = int(len(cell_metadata))
    discovery = read_csv_many(list(run_root.glob("shards/shard_*/scorecards/stage2_discovery_full_cohort_scorecard.csv")))
    if not discovery.empty:
        discovery["coefficient_path"] = str(discovery_coef_canonical.relative_to(experiment_dir))
        if matrix_paths:
            discovery["cell_prediction_matrix_path"] = str((canonical_root / "discovery_full_cohort" / "cell_prediction_matrix.npz").relative_to(experiment_dir))
            discovery["cell_prediction_fit_metadata_path"] = str(discovery_fit_meta_canonical.relative_to(experiment_dir))
            discovery["cell_metadata_path"] = str((canonical_root / "discovery_full_cohort" / "cell_metadata.csv.gz").relative_to(experiment_dir))
    outputs["discovery_rows"] = write_merged_csv(
        discovery,
        merged_root / "scorecards" / "stage2_discovery_full_cohort_scorecard.csv",
        scorecard_root / "stage2_discovery_full_cohort_scorecard.csv",
    )
    ok_discovery = discovery[discovery.get("status", pd.Series(dtype=object)).eq("ok")].copy() if not discovery.empty else pd.DataFrame()
    if not ok_discovery.empty:
        best_parts: list[pd.DataFrame] = []
        for metric in ["stage2_auprc", "stage2_balanced_accuracy"]:
            best_parts.append(
                ok_discovery.sort_values(metric, ascending=False)
                .groupby(["stage0_panel_id", "representation_family"], as_index=False)
                .head(1)
                .assign(best_by=metric)
            )
        sparse = ok_discovery.copy()
        sparse["sparsity_aware_score"] = sparse["stage2_auprc"] / np.log1p(sparse["nonzero_coefficient_count"].clip(lower=1))
        best_parts.append(
            sparse.sort_values("sparsity_aware_score", ascending=False)
            .groupby(["stage0_panel_id", "representation_family"], as_index=False)
            .head(1)
            .assign(best_by="sparsity_aware_score")
        )
        best_rows = pd.concat(best_parts, ignore_index=True)
        write_merged_csv(
            best_rows,
            merged_root / "discovery_full_cohort" / "best_rows.csv",
            canonical_root / "discovery_full_cohort" / "best_rows.csv",
        )

    lopo_coef = read_csv_many(list(run_root.glob("shards/shard_*/sharedness_lopo/coefficient_paths.csv")))
    lopo_coef_canonical = canonical_root / "sharedness_lopo" / "coefficient_paths.csv"
    outputs["lopo_coefficient_rows"] = write_merged_csv(
        lopo_coef,
        merged_root / "sharedness_lopo" / "coefficient_paths.csv",
        lopo_coef_canonical,
    )
    lopo_patient = read_csv_many(list(run_root.glob("shards/shard_*/sharedness_lopo/by_heldout_patient.csv")))
    lopo_patient_canonical = canonical_root / "sharedness_lopo" / "by_heldout_patient.csv"
    outputs["lopo_patient_rows"] = write_merged_csv(
        lopo_patient,
        merged_root / "sharedness_lopo" / "by_heldout_patient.csv",
        lopo_patient_canonical,
    )
    lopo_predictions = read_csv_many(list(run_root.glob("shards/shard_*/sharedness_lopo/predictions.csv")))
    lopo_predictions_canonical = canonical_root / "sharedness_lopo" / "predictions.csv"
    outputs["lopo_prediction_rows"] = write_merged_csv(
        lopo_predictions,
        merged_root / "sharedness_lopo" / "predictions.csv",
        lopo_predictions_canonical,
    )
    lopo = read_csv_many(list(run_root.glob("shards/shard_*/scorecards/stage2_sharedness_lopo_scorecard.csv")))
    if not lopo.empty:
        lopo["by_heldout_patient_path"] = str(lopo_patient_canonical.relative_to(experiment_dir))
        if not lopo_coef.empty:
            lopo["coefficient_path"] = str(lopo_coef_canonical.relative_to(experiment_dir))
        if not lopo_predictions.empty:
            lopo["predictions_path"] = str(lopo_predictions_canonical.relative_to(experiment_dir))
    outputs["lopo_rows"] = write_merged_csv(
        lopo,
        merged_root / "scorecards" / "stage2_sharedness_lopo_scorecard.csv",
        scorecard_root / "stage2_sharedness_lopo_scorecard.csv",
    )

    patient_coef = read_csv_many(list(run_root.glob("shards/shard_*/patient_specific/coefficient_paths.csv")))
    patient_coef_canonical = canonical_root / "patient_specific" / "coefficient_paths.csv"
    outputs["patient_specific_coefficient_rows"] = write_merged_csv(
        patient_coef,
        merged_root / "patient_specific" / "coefficient_paths.csv",
        patient_coef_canonical,
    )
    patient_fit_meta = read_csv_many(list(run_root.glob("shards/shard_*/patient_specific/cell_prediction_fit_metadata.csv")))
    patient_fit_meta_canonical = canonical_root / "patient_specific" / "cell_prediction_fit_metadata.csv"
    outputs["patient_specific_cell_prediction_fit_rows"] = write_merged_csv(
        patient_fit_meta,
        merged_root / "patient_specific" / "cell_prediction_fit_metadata.csv",
        patient_fit_meta_canonical,
    )
    patient_bundle_paths = sorted(run_root.glob("shards/shard_*/patient_specific/cell_prediction_bundle.npz"))
    if patient_bundle_paths:
        y_prob_parts: list[np.ndarray] = []
        cell_index_parts: list[np.ndarray] = []
        fit_ids: list[np.ndarray] = []
        indptr_parts: list[np.ndarray] = [np.asarray([0], dtype=np.int64)]
        data_offset = 0
        cell_ids: np.ndarray | None = None
        y_true_values: np.ndarray | None = None
        for path in patient_bundle_paths:
            data = np.load(path, allow_pickle=False)
            y_prob = data["y_prob"]
            cell_index = data["cell_index"]
            indptr = data["fit_indptr"].astype(np.int64)
            y_prob_parts.append(y_prob)
            cell_index_parts.append(cell_index.astype(np.int64))
            fit_ids.append(data["fit_id"].astype(str))
            indptr_parts.append(indptr[1:] + data_offset)
            data_offset += int(len(y_prob))
            if cell_ids is None:
                cell_ids = data["cell_id"].astype(str)
                y_true_values = data["y_true"]
            elif not np.array_equal(cell_ids, data["cell_id"].astype(str)):
                raise ValueError(f"Cell order mismatch while merging patient-specific prediction bundle: {path}")
        merged_y_prob = np.concatenate(y_prob_parts) if y_prob_parts else np.empty(0, dtype=np.float32)
        merged_cell_index = np.concatenate(cell_index_parts) if cell_index_parts else np.empty(0, dtype=np.int64)
        merged_indptr = np.concatenate(indptr_parts).astype(np.int64)
        merged_fit_ids = np.concatenate(fit_ids) if fit_ids else np.asarray([], dtype=np.str_)
        prediction_bundle_run = merged_root / "patient_specific" / "cell_prediction_bundle.npz"
        prediction_bundle_canonical = canonical_root / "patient_specific" / "cell_prediction_bundle.npz"
        prediction_bundle_run.parent.mkdir(parents=True, exist_ok=True)
        prediction_bundle_canonical.parent.mkdir(parents=True, exist_ok=True)
        for path in [prediction_bundle_run, prediction_bundle_canonical]:
            np.savez_compressed(
                path,
                y_prob=merged_y_prob,
                cell_index=merged_cell_index,
                fit_indptr=merged_indptr,
                fit_id=merged_fit_ids,
                cell_id=cell_ids,
                y_true=y_true_values,
            )
        outputs["patient_specific_cell_prediction_fit_count"] = int(len(merged_fit_ids))
        outputs["patient_specific_cell_prediction_values"] = int(len(merged_y_prob))
    patient_cell_metadata_paths = sorted(run_root.glob("shards/shard_*/patient_specific/cell_metadata.csv.gz"))
    if patient_cell_metadata_paths:
        patient_cell_metadata = pd.read_csv(patient_cell_metadata_paths[0])
        patient_cell_metadata_run = merged_root / "patient_specific" / "cell_metadata.csv.gz"
        patient_cell_metadata_canonical = canonical_root / "patient_specific" / "cell_metadata.csv.gz"
        patient_cell_metadata_run.parent.mkdir(parents=True, exist_ok=True)
        patient_cell_metadata_canonical.parent.mkdir(parents=True, exist_ok=True)
        patient_cell_metadata.to_csv(patient_cell_metadata_run, index=False, compression="gzip")
        patient_cell_metadata.to_csv(patient_cell_metadata_canonical, index=False, compression="gzip")
        outputs["patient_specific_cell_metadata_rows"] = int(len(patient_cell_metadata))
    patient_specific = read_csv_many(list(run_root.glob("shards/shard_*/scorecards/stage2_patient_specific_scorecard.csv")))
    if not patient_specific.empty:
        patient_specific["coefficient_path"] = str(patient_coef_canonical.relative_to(experiment_dir))
        if patient_bundle_paths:
            patient_specific["cell_prediction_bundle_path"] = str((canonical_root / "patient_specific" / "cell_prediction_bundle.npz").relative_to(experiment_dir))
            patient_specific["cell_prediction_fit_metadata_path"] = str(patient_fit_meta_canonical.relative_to(experiment_dir))
            patient_specific["cell_metadata_path"] = str((canonical_root / "patient_specific" / "cell_metadata.csv.gz").relative_to(experiment_dir))
    outputs["patient_specific_rows"] = write_merged_csv(
        patient_specific,
        merged_root / "scorecards" / "stage2_patient_specific_scorecard.csv",
        scorecard_root / "stage2_patient_specific_scorecard.csv",
    )

    manifest_path = merged_root / "merge_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "completed",
            "completed_at": pd.Timestamp.now().isoformat(),
            "experiment_dir": str(experiment_dir),
            "stage2_run_id": run_id,
            "stage2_output_branch": safe_branch(branch_name),
            "outputs": outputs,
        },
    )
    outputs["merge_manifest_path"] = str(manifest_path.relative_to(experiment_dir))
    LOGGER.info("Merged Stage 2 run %s", run_id)
    return outputs


def discover_gpu_ids() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=True,
        )
        gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return gpu_ids or ["0"]
    except Exception:
        LOGGER.warning("Could not discover GPUs with nvidia-smi; defaulting to GPU 0", exc_info=True)
        return ["0"]


def worker_command_from_args(args: argparse.Namespace, shard_index: int, shard_count: int, run_id: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--experiment-dir",
        str(args.experiment_dir),
        "--stage2-output-branch",
        args.stage2_output_branch,
        "--stage1-fit-scope-note",
        args.stage1_fit_scope_note,
        "--positive-label",
        args.positive_label,
        "--negative-label",
        args.negative_label,
        "--patient-col",
        args.patient_col,
        "--shortlist-output",
        args.shortlist_output,
        "--shortlist-single-geneset-top-n",
        str(args.shortlist_single_geneset_top_n),
        "--shortlist-group-top-n",
        str(args.shortlist_group_top_n),
        "--panel-selection",
        args.panel_selection,
        "--penalties",
        args.penalties,
        "--c-grid-log10-min",
        str(args.c_grid_log10_min),
        "--c-grid-log10-max",
        str(args.c_grid_log10_max),
        "--c-grid-n",
        str(args.c_grid_n),
        "--l1-ratios",
        args.l1_ratios,
        "--class-weight",
        args.class_weight,
        "--decision-thresholds",
        args.decision_thresholds,
        "--fixed-recall-targets",
        args.fixed_recall_targets,
        "--top-fraction-thresholds",
        args.top_fraction_thresholds,
        "--backend",
        args.backend,
        "--seed",
        str(args.seed),
        "--max-iter",
        str(args.max_iter),
        "--n-jobs",
        str(args.n_jobs),
        "--coef-nonzero-tol",
        str(args.coef_nonzero_tol),
        "--low-malignant-support-threshold",
        str(args.low_malignant_support_threshold),
        "--discovery-cell-prediction-dtype",
        args.discovery_cell_prediction_dtype,
        "--patient-specific-cell-prediction-dtype",
        args.patient_specific_cell_prediction_dtype,
        "--discovery-cell-metadata-cols",
        args.discovery_cell_metadata_cols,
        "--stage2-run-id",
        run_id,
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]
    if args.stage0_scorecard:
        cmd.extend(["--stage0-scorecard", str(args.stage0_scorecard)])
    if args.canonicalize_existing_quick_stage2:
        cmd.append("--canonicalize-existing-quick-stage2")
    if args.make_shortlist_from_quick_l2:
        cmd.append("--make-shortlist-from-quick-l2")
    if args.run_discovery_full_cohort_fit:
        cmd.append("--run-discovery-full-cohort-fit")
    if args.run_sharedness_lopo:
        cmd.append("--run-sharedness-lopo")
    if args.run_patient_specific:
        cmd.append("--run-patient-specific")
    if args.max_selected_representations is not None:
        cmd.extend(["--max-selected-representations", str(args.max_selected_representations)])
    if args.max_patient_specific_patients is not None:
        cmd.extend(["--max-patient-specific-patients", str(args.max_patient_specific_patients)])
    if args.strict_gpu:
        cmd.append("--strict-gpu")
    if args.save_discovery_cell_predictions:
        cmd.append("--save-discovery-cell-predictions")
    if args.save_lopo_cell_predictions:
        cmd.append("--save-lopo-cell-predictions")
    if not args.save_lopo_coefficients:
        cmd.append("--no-save-lopo-coefficients")
    if args.save_patient_specific_cell_predictions:
        cmd.append("--save-patient-specific-cell-predictions")
    if args.verbose:
        cmd.append("--verbose")
    return cmd


def launch_gpu_shards(args: argparse.Namespace, log_path: Path) -> dict[str, Any]:
    gpu_ids = discover_gpu_ids() if args.gpu_ids == "auto" else [x.strip() for x in args.gpu_ids.replace(",", " ").split() if x.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids available for --launch-gpu-shards")
    run_id = args.stage2_run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_stage2_gpu_shards"
    shard_count = int(args.shard_count) if int(args.shard_count) > 1 else len(gpu_ids)
    run_root = stage2_run_root(args.experiment_dir, run_id, getattr(args, "stage2_output_branch", ""))
    launcher_log_dir = run_root / "launcher_logs"
    launcher_log_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Launching %d Stage 2 shards across GPUs %s with run_id=%s", shard_count, gpu_ids, run_id)

    processes: list[tuple[int, str, subprocess.Popen[Any], Any, Path]] = []
    for shard_index in range(shard_count):
        gpu_id = gpu_ids[shard_index % len(gpu_ids)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        shard_log_path = launcher_log_dir / f"shard_{shard_index:03d}_gpu_{gpu_id}.log"
        handle = shard_log_path.open("w")
        cmd = worker_command_from_args(args, shard_index, shard_count, run_id)
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        process = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((shard_index, gpu_id, process, handle, shard_log_path))
        LOGGER.info("Launched shard %d/%d on GPU %s pid=%s log=%s", shard_index, shard_count, gpu_id, process.pid, shard_log_path)

    failures: list[dict[str, Any]] = []
    for shard_index, gpu_id, process, handle, shard_log_path in processes:
        return_code = process.wait()
        handle.close()
        LOGGER.info("Shard %d/%d on GPU %s exited with code %s", shard_index, shard_count, gpu_id, return_code)
        if return_code != 0:
            failures.append({"shard_index": shard_index, "gpu_id": gpu_id, "return_code": return_code, "log_path": str(shard_log_path)})

    launcher_manifest = run_root / "launcher_manifest.json"
    outputs: dict[str, Any] = {
        "stage2_run_id": run_id,
        "shard_count": shard_count,
        "gpu_ids": gpu_ids,
        "launcher_log_path": str(log_path.relative_to(args.experiment_dir)),
        "failures": failures,
    }
    if failures:
        write_json(launcher_manifest, {"status": "failed", "outputs": outputs})
        raise RuntimeError(f"{len(failures)} Stage 2 GPU shard(s) failed; see {launcher_manifest}")
    if not args.no_merge_after_launch:
        outputs["merge"] = merge_stage2_run(args.experiment_dir, run_id, getattr(args, "stage2_output_branch", ""))
    write_json(launcher_manifest, {"status": "completed", "completed_at": pd.Timestamp.now().isoformat(), "outputs": outputs})
    LOGGER.info("Wrote launcher manifest: %s", launcher_manifest)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--stage0-scorecard", type=Path, default=None, help="Stage 0 scorecard to consume, absolute or relative to --experiment-dir.")
    parser.add_argument("--stage2-output-branch", default="", help="Namespace for Stage 2 artifacts and analysis scorecards.")
    parser.add_argument("--stage1-fit-scope-note", default="transductive_all_eligible_cells")
    parser.add_argument("--positive-label", default="cancer")
    parser.add_argument("--negative-label", default="normal")
    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--canonicalize-existing-quick-stage2", action="store_true")
    parser.add_argument("--make-shortlist-from-quick-l2", action="store_true")
    parser.add_argument("--shortlist-output", default=DEFAULT_SHORTLIST_OUTPUT)
    parser.add_argument("--shortlist-single-geneset-top-n", type=int, default=12)
    parser.add_argument("--shortlist-group-top-n", type=int, default=5)
    parser.add_argument("--run-discovery-full-cohort-fit", action="store_true")
    parser.add_argument("--run-sharedness-lopo", action="store_true")
    parser.add_argument("--run-patient-specific", action="store_true")
    parser.add_argument(
        "--panel-selection",
        choices=["shortlist_plus_controls", "shortlist", "all_quick_rows", "all_biological_quick_rows"],
        default="shortlist_plus_controls",
        help=(
            "Representation selection for Stage 2. all_biological_quick_rows keeps successful "
            "single_geneset_only and single_group_only quick rows while excluding controls/HVG anchors."
        ),
    )
    parser.add_argument("--max-selected-representations", type=int, default=None)
    parser.add_argument("--max-patient-specific-patients", type=int, default=None)
    parser.add_argument("--penalties", default="l1,l2,elasticnet")
    parser.add_argument("--c-grid-log10-min", type=float, default=-4.0)
    parser.add_argument("--c-grid-log10-max", type=float, default=4.0)
    parser.add_argument("--c-grid-n", type=int, default=17)
    parser.add_argument("--l1-ratios", default="0.1,0.5,0.9")
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--decision-thresholds", default="0.5")
    parser.add_argument("--fixed-recall-targets", default="0.5,0.7,0.8")
    parser.add_argument("--top-fraction-thresholds", default="0.01,0.05,0.10")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--strict-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--coef-nonzero-tol", type=float, default=1e-8)
    parser.add_argument("--low-malignant-support-threshold", type=int, default=10)
    parser.add_argument("--stage2-run-id", default=None)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-stage2-run", action="store_true")
    parser.add_argument("--launch-gpu-shards", action="store_true")
    parser.add_argument("--gpu-ids", default="auto")
    parser.add_argument("--no-merge-after-launch", action="store_true")
    parser.add_argument("--save-discovery-cell-predictions", action="store_true")
    parser.add_argument("--save-lopo-cell-predictions", action="store_true")
    parser.add_argument(
        "--no-save-lopo-coefficients",
        action="store_false",
        dest="save_lopo_coefficients",
        help="Skip fold-level LOPO coefficient_paths.csv (default saves coefficients for Fig 3C stability).",
    )
    parser.set_defaults(save_lopo_coefficients=True)
    parser.add_argument("--save-patient-specific-cell-predictions", action="store_true")
    parser.add_argument("--discovery-cell-prediction-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--patient-specific-cell-prediction-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--discovery-cell-metadata-cols",
        default=(
            "patient,CN.label,predicted.annotation,predicted.annotation.score,"
            "predicted.pseudotime,predicted.pseudotime.score,Time,Tech,sample,"
            "timepoint_type,Days from SCT,Days.from.Relapse"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.experiment_dir = args.experiment_dir.expanduser().resolve()
    if args.stage0_scorecard is not None:
        args.stage0_scorecard = args.stage0_scorecard.expanduser()
        if not args.stage0_scorecard.is_absolute():
            args.stage0_scorecard = args.experiment_dir / args.stage0_scorecard
        args.stage0_scorecard = args.stage0_scorecard.resolve()
    args.stage2_output_branch = safe_branch(args.stage2_output_branch)
    if args.stage2_output_branch and args.shortlist_output == DEFAULT_SHORTLIST_OUTPUT:
        args.shortlist_output = f"analysis/scorecards/{args.stage2_output_branch}/stage2_provisional_shortlist_from_quick_l2.csv"
    log_path = configure_logging(args.experiment_dir, args.verbose)
    validate_shard_args(args)

    if args.merge_stage2_run:
        if not args.stage2_run_id:
            raise ValueError("--stage2-run-id is required with --merge-stage2-run")
        outputs = merge_stage2_run(args.experiment_dir, args.stage2_run_id, args.stage2_output_branch)
        LOGGER.info("Completed merge for Stage 2 run %s: %s", args.stage2_run_id, outputs)
        return

    if args.launch_gpu_shards:
        outputs = launch_gpu_shards(args, log_path)
        LOGGER.info("Completed GPU shard launcher: %s", outputs)
        return

    preflight_backend(args)

    scorecard = load_stage0_scorecard(args.experiment_dir, args.stage0_scorecard)
    outputs: dict[str, Any] = {
        "log_path": str(log_path.relative_to(args.experiment_dir)),
        "stage0_scorecard": str((args.stage0_scorecard or (args.experiment_dir / "analysis" / "scorecards" / "stage0_mrd_old34_broad_scorecard.csv")).relative_to(args.experiment_dir)),
        "stage2_output_branch": args.stage2_output_branch,
        "stage2_run_id": args.stage2_run_id,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
    }

    if args.canonicalize_existing_quick_stage2:
        canonical = canonicalize_existing_quick_stage2(args.experiment_dir, scorecard, args)
        outputs["canonical_quick_rows"] = int(len(canonical))

    shortlist_path = args.experiment_dir / args.shortlist_output
    if args.make_shortlist_from_quick_l2:
        shortlist = make_shortlist(args.experiment_dir, scorecard, args)
    elif args.run_discovery_full_cohort_fit or args.run_sharedness_lopo or args.run_patient_specific:
        if shortlist_path.exists():
            shortlist = pd.read_csv(shortlist_path)
            LOGGER.info("Loaded existing shortlist: %s", shortlist_path)
        else:
            shortlist = make_shortlist(args.experiment_dir, scorecard, args)
    else:
        shortlist = pd.read_csv(shortlist_path) if shortlist_path.exists() else pd.DataFrame()

    all_specs = resolve_selected_specs(args.experiment_dir, scorecard, shortlist, args) if not shortlist.empty else []
    specs = apply_spec_shard(all_specs, args)
    outputs["selected_representations_total"] = len(all_specs)
    outputs["selected_representations_this_shard"] = len(specs)

    if args.run_discovery_full_cohort_fit or args.run_sharedness_lopo or args.run_patient_specific:
        pred = read_first_prediction_table(args.experiment_dir, scorecard)
        y, groups, _obs = labels_from_predictions(pred, args.positive_label, args.negative_label)
        cell_metadata = build_cell_metadata(args.experiment_dir, pred, y, args)
        support = patient_support_table(y, groups)
        support_path = analysis_scorecard_root(args.experiment_dir, args) / "stage2_patient_support_counts.csv"
        support_path.parent.mkdir(parents=True, exist_ok=True)
        support.to_csv(support_path, index=False)
        outputs["patient_support_counts_path"] = str(support_path.relative_to(args.experiment_dir))
        outputs["cell_metadata_columns"] = list(cell_metadata.columns)

    if args.run_discovery_full_cohort_fit:
        discovery = run_discovery_full_cohort(args.experiment_dir, specs, y, cell_metadata, args)
        outputs["discovery_rows"] = int(len(discovery))

    if args.run_sharedness_lopo:
        lopo = run_lopo_sharedness(args.experiment_dir, specs, y, groups, args)
        outputs["lopo_rows"] = int(len(lopo))

    if args.run_patient_specific:
        patient_specific = run_patient_specific(args.experiment_dir, specs, y, groups, cell_metadata, args)
        outputs["patient_specific_rows"] = int(len(patient_specific))

    manifest_path = stage2_output_root(args.experiment_dir, args) / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "status": "completed",
            "completed_at": pd.Timestamp.now().isoformat(),
            "experiment_dir": str(args.experiment_dir),
            "stage0_scorecard": str(args.stage0_scorecard) if args.stage0_scorecard else None,
            "stage2_output_branch": args.stage2_output_branch,
            "stage1_fit_scope_note": args.stage1_fit_scope_note,
            "args": vars(args),
            "outputs": outputs,
            "implementation_adjustment_note": (
                "This first implementation consumes the best quick-L2 representation per shortlisted panel by default "
                "instead of expanding every panel/method/K row. Use --panel-selection all_quick_rows for exhaustive "
                "Stage 2 path fitting over all existing quick rows."
            ),
        },
    )
    LOGGER.info("Wrote multiobjective manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
