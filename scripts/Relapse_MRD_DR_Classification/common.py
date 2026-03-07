from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis as FAWrapper
from sc_classification.dimension_reduction.factosig import FactoSigDR
from sc_classification.dimension_reduction.pca import PCA as PCAWrapper
from sc_classification.utils.logistic_backend import make_logistic_regression

warnings.filterwarnings(
    "ignore",
    message=r".*'penalty' was deprecated.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Inconsistent values: penalty=.*with l1_ratio=None.*",
    category=UserWarning,
    module=r"sklearn\.linear_model\._logistic",
)
warnings.filterwarnings(
    "ignore",
    message=r".*n_jobs.*has no effect.*",
    category=FutureWarning,
    module=r"sklearn\.linear_model\._logistic",
)


FOCUS_TIMEPOINTS = {"MRD", "Relapse"}
VALID_TARGETS = {"cancer", "normal"}
DEFAULT_MULTICLASS_ORDER = [
    "MRD_cancer",
    "MRD_normal",
    "Relapse_cancer",
    "Relapse_normal",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def split_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def split_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def coarse_timepoint(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return re.sub(r"_[0-9]+$", "", str(value))


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def ensure_timepoint_type(
    adata: AnnData,
    *,
    time_key: str = "Time",
    timepoint_type_key: str = "timepoint_type",
) -> AnnData:
    if timepoint_type_key in adata.obs.columns:
        return adata
    if time_key not in adata.obs.columns:
        raise KeyError(f"Column '{time_key}' not found in adata.obs.")
    adata = adata.copy()
    adata.obs[timepoint_type_key] = adata.obs[time_key].astype(str).map(coarse_timepoint)
    return adata


def load_relapse_mrd_adata(
    input_h5ad: str | Path,
    *,
    target_col: str = "CN.label",
    patient_col: str = "patient",
    time_key: str = "Time",
    timepoint_type_key: str = "timepoint_type",
    composite_col: str = "relapse_mrd_label",
) -> AnnData:
    adata = read_h5ad(str(input_h5ad))
    adata = ensure_timepoint_type(adata, time_key=time_key, timepoint_type_key=timepoint_type_key)
    obs = adata.obs.copy()
    keep = (
        obs[target_col].astype(str).isin(VALID_TARGETS)
        & obs[timepoint_type_key].astype(str).isin(FOCUS_TIMEPOINTS)
        & obs[patient_col].notna()
        & (obs[patient_col].astype(str) != "unknown")
    )
    adata = adata[keep].copy()
    adata.obs[composite_col] = (
        adata.obs[timepoint_type_key].astype(str) + "_" + adata.obs[target_col].astype(str)
    )
    return adata


def patient_class_summary(
    adata: AnnData,
    *,
    patient_col: str = "patient",
    composite_col: str = "relapse_mrd_label",
    timepoint_type_key: str = "timepoint_type",
    tech_col: str = "Tech",
) -> pd.DataFrame:
    rows: List[Dict] = []
    for patient, sub in adata.obs.groupby(patient_col, observed=False):
        patient = str(patient)
        counts = sub[composite_col].value_counts()
        row = {
            "patient": patient,
            "n_cells": int(sub.shape[0]),
            "has_MRD": bool((sub[timepoint_type_key].astype(str) == "MRD").any()),
            "has_Relapse": bool((sub[timepoint_type_key].astype(str) == "Relapse").any()),
            "n_timepoints": int(sub[timepoint_type_key].astype(str).nunique()),
            "tech_values": ",".join(sorted(sub[tech_col].astype(str).dropna().unique().tolist()))
            if tech_col in sub.columns
            else "",
        }
        for cls in DEFAULT_MULTICLASS_ORDER:
            row[f"n_{cls}"] = int(counts.get(cls, 0))
        row["min_four_class_support"] = int(min(row[f"n_{cls}"] for cls in DEFAULT_MULTICLASS_ORDER))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["min_four_class_support", "patient"])


def eligible_relapse_patients(
    summary_df: pd.DataFrame,
    *,
    require_all_four_classes: bool = False,
) -> List[str]:
    df = summary_df.copy()
    df = df[df["has_MRD"] & df["has_Relapse"]]
    if require_all_four_classes:
        df = df[df["min_four_class_support"] > 0]
    return df["patient"].astype(str).tolist()


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def variance_select_genes(adata: AnnData, n_top_genes: int) -> Tuple[AnnData, List[str]]:
    if "highly_variable" in adata.var.columns and int(adata.var["highly_variable"].sum()) >= n_top_genes:
        keep = adata.var["highly_variable"].to_numpy(dtype=bool)
        selected = adata.var_names[keep].tolist()
        return adata[:, keep].copy(), selected

    X = adata.X
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
        sq_means = np.asarray(X.power(2).mean(axis=0)).ravel()
        variances = sq_means - np.square(means)
    else:
        variances = np.var(np.asarray(X), axis=0)
    n_keep = min(int(n_top_genes), adata.n_vars)
    top_idx = np.argsort(np.asarray(variances).ravel())[-n_keep:]
    top_idx = np.sort(top_idx)
    selected = adata.var_names[top_idx].tolist()
    return adata[:, selected].copy(), selected


def preprocess_patient_adata(
    adata: AnnData,
    *,
    gene_method: str = "hvg",
    n_top_genes: int = 3000,
    standardize: bool = True,
) -> Tuple[AnnData, Dict]:
    adata_proc = adata.copy()
    info: Dict[str, object] = {
        "input_shape": [int(adata.n_obs), int(adata.n_vars)],
        "gene_method": gene_method,
        "n_top_genes_requested": int(n_top_genes),
        "standardize": bool(standardize),
    }

    if gene_method == "all":
        selected_genes = adata_proc.var_names.tolist()
    elif gene_method == "hvg":
        adata_proc, selected_genes = variance_select_genes(adata_proc, n_top_genes=n_top_genes)
    else:
        raise ValueError(f"Unsupported gene_method '{gene_method}'. Expected one of: hvg, all.")

    info["n_selected_genes"] = int(len(selected_genes))
    info["selected_genes_preview"] = selected_genes[:20]

    if standardize:
        X_scaled = StandardScaler(with_mean=True).fit_transform(to_dense(adata_proc.X))
        adata_proc.X = X_scaled.astype(np.float32, copy=False)
    else:
        adata_proc.X = to_dense(adata_proc.X).astype(np.float32, copy=False)

    info["output_shape"] = [int(adata_proc.n_obs), int(adata_proc.n_vars)]
    return adata_proc, info


def run_dr_method(
    adata: AnnData,
    *,
    method: str,
    k: int,
    random_state: int,
    factosig_device: str = "cpu",
    factosig_lr: float = 1e-2,
    factosig_max_iter: int = 300,
    factosig_rotation: str = "varimax",
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    method = str(method).strip().lower()
    ad = adata.copy()

    if method == "pca":
        model = PCAWrapper()
        ad = model.fit_transform(ad, n_components=k, random_state=random_state, standardize_input=False, svd_solver="randomized")
        scores = np.asarray(ad.obsm["X_pca"]).astype(np.float32, copy=False)
        loadings = np.asarray(ad.varm["PCA_loadings"]).astype(np.float32, copy=False)
        feature_names = [f"pca_{i+1}" for i in range(scores.shape[1])]
        info = ad.uns.get("pca", {})
    elif method == "fa":
        model = FAWrapper()
        ad = model.fit_transform(ad, n_components=k, random_state=random_state, standardize_input=False, svd_method="randomized")
        scores = np.asarray(ad.obsm["X_fa"]).astype(np.float32, copy=False)
        loadings = np.asarray(ad.varm["FA_loadings"]).astype(np.float32, copy=False)
        feature_names = [f"fa_{i+1}" for i in range(scores.shape[1])]
        info = ad.uns.get("fa", {})
    elif method == "factosig":
        model = FactoSigDR()
        ad = model.fit_transform(
            ad,
            n_components=k,
            random_state=random_state,
            device=factosig_device,
            lr=factosig_lr,
            max_iter=factosig_max_iter,
            verbose=False,
            rotation=factosig_rotation,
        )
        scores = np.asarray(ad.obsm["X_factosig"]).astype(np.float32, copy=False)
        loadings = np.asarray(ad.varm["FACTOSIG_loadings"]).astype(np.float32, copy=False)
        feature_names = [f"factosig_{i+1}" for i in range(scores.shape[1])]
        info = ad.uns.get("factosig", {})
    else:
        raise ValueError(f"Unsupported DR method '{method}'. Expected one of: pca, fa, factosig.")

    return scores, loadings, feature_names, info


def save_patient_dr_artifacts(
    method_dir: Path,
    *,
    metadata: pd.DataFrame,
    scores: np.ndarray,
    loadings: np.ndarray,
    feature_names: Sequence[str],
    gene_names: Sequence[str],
    dr_info: Dict,
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(method_dir / "metadata.csv")
    pd.DataFrame({"feature": list(feature_names)}).to_csv(method_dir / "feature_names.csv", index=False)
    pd.DataFrame({"gene": list(gene_names)}).to_csv(method_dir / "gene_names.csv", index=False)
    np.save(method_dir / "scores.npy", scores)
    np.save(method_dir / "loadings.npy", loadings)
    write_json(method_dir / "dr_info.json", dr_info)


def load_patient_dr_artifacts(method_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    metadata = pd.read_csv(method_dir / "metadata.csv", index_col=0)
    scores = np.load(method_dir / "scores.npy")
    loadings = np.load(method_dir / "loadings.npy")
    feature_names = pd.read_csv(method_dir / "feature_names.csv")["feature"].astype(str).tolist()
    gene_names = pd.read_csv(method_dir / "gene_names.csv")["gene"].astype(str).tolist()
    return metadata, scores, loadings, feature_names, gene_names


def safe_silhouette(X: np.ndarray, labels: Sequence[object]) -> float:
    labels_arr = pd.Series(labels).astype(str)
    if labels_arr.nunique() < 2 or len(labels_arr) < 3:
        return float("nan")
    return float(silhouette_score(X, labels_arr))


def tech_neighbor_mixing(X: np.ndarray, labels: Sequence[object], n_neighbors: int = 15) -> float:
    labels_arr = pd.Series(labels).astype(str).to_numpy()
    if len(labels_arr) < 3 or len(np.unique(labels_arr)) < 2:
        return float("nan")
    k = min(int(n_neighbors) + 1, len(labels_arr))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    idx = nn.kneighbors(return_distance=False)
    same = []
    for i in range(len(labels_arr)):
        nbr_idx = idx[i][1:]
        if len(nbr_idx) == 0:
            continue
        same.append(np.mean(labels_arr[nbr_idx] == labels_arr[i]))
    return float(np.mean(same)) if same else float("nan")


def latent_diagnostics(metadata: pd.DataFrame, scores: np.ndarray) -> List[Dict]:
    rows: List[Dict] = []
    for column in ["Tech", "timepoint_type", "CN.label", "relapse_mrd_label"]:
        if column not in metadata.columns:
            continue
        labels = metadata[column].astype(str)
        rows.append(
            {
                "label_column": column,
                "n_labels": int(labels.nunique()),
                "silhouette": safe_silhouette(scores, labels),
                "same_label_neighbor_fraction": tech_neighbor_mixing(scores, labels),
            }
        )
    return rows


def make_hyperparam_grid(alpha_grid: np.ndarray, enet_l1_ratios: Sequence[float]) -> List[Dict]:
    grid: List[Dict] = []
    for alpha in alpha_grid:
        grid.append({"penalty": "l1", "alpha": float(alpha), "C": float(1.0 / alpha), "l1_ratio": np.nan})
    for alpha in alpha_grid:
        grid.append({"penalty": "l2", "alpha": float(alpha), "C": float(1.0 / alpha), "l1_ratio": np.nan})
    for l1_ratio in enet_l1_ratios:
        for alpha in alpha_grid:
            grid.append(
                {
                    "penalty": "elasticnet",
                    "alpha": float(alpha),
                    "C": float(1.0 / alpha),
                    "l1_ratio": float(l1_ratio),
                }
            )
    return grid


def choose_best(df: pd.DataFrame, primary: Sequence[str]) -> pd.Series:
    ranked = df.copy()
    ascending = []
    for column in primary:
        ranked[column] = ranked[column].fillna(-np.inf)
        ascending.append(False)
    ranked["C"] = ranked["C"].fillna(np.inf)
    ranked = ranked.sort_values(list(primary) + ["C"], ascending=ascending + [True])
    return ranked.iloc[0]


def _align_probabilities(prob: np.ndarray, model_classes: Sequence[int], target_classes: Sequence[int]) -> np.ndarray:
    out = np.zeros((prob.shape[0], len(target_classes)), dtype=float)
    class_to_idx = {int(c): i for i, c in enumerate(target_classes)}
    for j, cls in enumerate(model_classes):
        out[:, class_to_idx[int(cls)]] = prob[:, j]
    return out


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    class_labels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(class_labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(class_labels), average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=list(class_labels), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=list(class_labels), average="macro", zero_division=0)),
    }

    per_precision = precision_score(y_true, y_pred, labels=list(class_labels), average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, labels=list(class_labels), average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, labels=list(class_labels), average=None, zero_division=0)
    supports = pd.Series(y_true).value_counts().reindex(class_labels, fill_value=0)
    for idx, class_name in enumerate(class_names):
        out[f"support_{class_name}"] = int(supports.iloc[idx])
        out[f"precision_{class_name}"] = float(per_precision[idx])
        out[f"recall_{class_name}"] = float(per_recall[idx])
        out[f"f1_{class_name}"] = float(per_f1[idx])

    y_bin = label_binarize(y_true, classes=list(class_labels))
    auc_values: List[float] = []
    ap_values: List[float] = []
    for idx, class_name in enumerate(class_names):
        if y_bin[:, idx].sum() == 0 or y_bin[:, idx].sum() == y_bin.shape[0]:
            out[f"auc_ovr_{class_name}"] = float("nan")
            out[f"ap_ovr_{class_name}"] = float("nan")
            continue
        auc_val = float(roc_auc_score(y_bin[:, idx], y_prob[:, idx]))
        ap_val = float(average_precision_score(y_bin[:, idx], y_prob[:, idx]))
        out[f"auc_ovr_{class_name}"] = auc_val
        out[f"ap_ovr_{class_name}"] = ap_val
        auc_values.append(auc_val)
        ap_values.append(ap_val)
    out["macro_auc_ovr"] = float(np.mean(auc_values)) if auc_values else float("nan")
    out["macro_ap_ovr"] = float(np.mean(ap_values)) if ap_values else float("nan")
    return out


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    positive_name: str,
    negative_name: str,
) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    out: Dict[str, float] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"recall_{positive_name}": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f"precision_{positive_name}": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f"recall_{negative_name}": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        f"precision_{negative_name}": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    return out


def fit_logistic_model(
    *,
    penalty: str,
    C: float,
    l1_ratio: Optional[float],
    random_state: int,
    ml_backend: str,
    strict_gpu: bool,
):
    return make_logistic_regression(
        penalty=penalty,
        C=float(C),
        l1_ratio=l1_ratio if penalty == "elasticnet" else None,
        random_state=int(random_state),
        max_iter=5000,
        n_jobs=-1,
        backend=ml_backend,
        strict_gpu=strict_gpu,
    )


def align_probability_output(
    clf,
    prob: np.ndarray,
    *,
    class_labels: Sequence[int],
) -> np.ndarray:
    model_classes = getattr(clf.model, "classes_", np.asarray(class_labels))
    return _align_probabilities(prob, model_classes=model_classes, target_classes=class_labels)
