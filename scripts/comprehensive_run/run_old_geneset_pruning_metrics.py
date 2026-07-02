#!/usr/bin/env python
"""
Run the expensive old-geneset pruning DR grid outside the notebook.

This script builds the strict-ablation panel manifest, runs DR methods over
candidate panels, computes quantitative metrics, writes a leaderboard, and
optionally writes gated UMAP PNGs. It is designed for screen/tmux execution and
restartable runs via per-panel score caches.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


LOGGER = logging.getLogger("old_geneset_pruning")

REQUIRED_OBS = ["predicted.annotation", "predicted.pseudotime", "CN.label", "patient", "timepoint_type"]
CN_LABEL_COL = "CN.label"
CELL_TYPE_COL = "predicted.annotation"
PSEUDOTIME_COL = "predicted.pseudotime"
PATIENT_COL = "patient"
TIMEPOINT_COL = "timepoint_type"
MRD_VALUES = {"MRD"}
MALIGNANT_VALUES = {"cancer"}
NORMAL_VALUES = {"normal"}


def split_csv(value: str | None, cast=str) -> list[Any]:
    if value is None:
        return []
    return [cast(x.strip()) for x in str(value).replace(",", " ").split() if x.strip()]


def configure_logging(out_dir: Path, verbose: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"old_geneset_pruning_metrics_{time.strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    LOGGER.info("Logging to %s", log_path)
    return log_path


def add_import_paths(sc_root: Path) -> None:
    for p in (sc_root / "src", sc_root / "scripts" / "comprehensive_run"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def parse_gmt(path: Path) -> dict[str, set[str]]:
    genesets: dict[str, set[str]] = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genesets[parts[0]] = {g.strip().upper() for g in parts[2:] if g.strip()}
    return genesets


def assign_biology_group(why_include: str) -> str:
    text = str(why_include).lower()
    if "antigen presentation" in text or "mhc" in text:
        return "antigen_presentation"
    if "il6" in text or "jak" in text or "stat" in text or "il2" in text:
        return "cytokine_jak_stat"
    if "ifng" in text or "ifna" in text or "tnf" in text or "nfkb" in text or "inflammation" in text:
        return "inflammatory_interferon"
    if "p53" in text or "apoptosis" in text or "hypoxia" in text or "upr" in text or "ros" in text or "dna repair" in text:
        return "stress_arrest"
    if "e2f" in text or "g2m" in text or "myc" in text or "oxphos" in text or "glycolysis" in text or "mtorc1" in text:
        return "proliferation_metabolism"
    if "tgfbeta" in text or "emt" in text or "plasticity" in text:
        return "tgf_plasticity"
    if "tlr" in text or "complement" in text or "allograft" in text:
        return "innate_immune_context"
    return "other"


def safe_id(s: str) -> str:
    return (
        str(s)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


def covered_genes_for_sets(
    selected_sets: list[str],
    geneset_to_covered: dict[str, set[str]],
    excluded_genes: set[str] | None = None,
) -> list[str]:
    genes: set[str] = set()
    for gs in selected_sets:
        genes |= geneset_to_covered[gs]
    if excluded_genes:
        genes -= set(excluded_genes)
    return sorted(genes)


def panel_row(
    panel_id: str,
    panel_type: str,
    selected_sets: list[str],
    description: str,
    geneset_to_covered: dict[str, set[str]],
    excluded_genes: set[str] | None = None,
    excluded_gene_sets: list[str] | None = None,
) -> dict[str, Any]:
    excluded_genes = set(excluded_genes or set())
    genes_before_exclusion = covered_genes_for_sets(selected_sets, geneset_to_covered)
    genes = covered_genes_for_sets(selected_sets, geneset_to_covered, excluded_genes=excluded_genes)
    return {
        "panel_id": panel_id,
        "panel_type": panel_type,
        "panel_family": "knowledge_old_geneset",
        "source_dictionary": "old_34_programs",
        "description": description,
        "genesets": selected_sets,
        "excluded_gene_sets": list(excluded_gene_sets or []),
        "excluded_genes": sorted(excluded_genes),
        "n_gene_sets": len(selected_sets),
        "n_excluded_genes": len(excluded_genes),
        "n_shared_genes_removed": len(set(genes_before_exclusion) & excluded_genes),
        "genes": genes,
        "n_genes": len(genes),
    }


def direct_gene_panel_row(
    panel_id: str,
    panel_type: str,
    genes: list[str],
    description: str,
    *,
    panel_family: str,
    source_dictionary: str,
    matched_panel_id: str | None = None,
) -> dict[str, Any]:
    genes = sorted(dict.fromkeys(str(g) for g in genes))
    return {
        "panel_id": panel_id,
        "panel_type": panel_type,
        "panel_family": panel_family,
        "source_dictionary": source_dictionary,
        "description": description,
        "genesets": [],
        "excluded_gene_sets": [],
        "excluded_genes": [],
        "n_gene_sets": 0,
        "n_excluded_genes": 0,
        "n_shared_genes_removed": 0,
        "genes": genes,
        "n_genes": len(genes),
        "matched_panel_id": matched_panel_id,
    }


def ordered_available_genes(candidates: list[str], available: set[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for gene in candidates:
        g = str(gene)
        if g in available and g not in seen:
            out.append(g)
            seen.add(g)
    return out


def panel_family_enabled(args: argparse.Namespace, family: str) -> bool:
    return family in set(args.panel_families)


def _sample_indices(n: int, max_n: int, random_state: int = 42) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def make_binary_labels(obs_df: pd.DataFrame) -> pd.Series:
    s = obs_df[CN_LABEL_COL].astype(str)
    y = pd.Series(index=obs_df.index, dtype="float")
    y[s.isin(NORMAL_VALUES)] = 0
    y[s.isin(MALIGNANT_VALUES)] = 1
    return y.dropna().astype(int)


def make_logreg_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def compute_knn_label_metrics(scores: np.ndarray, labels: pd.Series, n_neighbors: int) -> dict[str, float]:
    labels = labels.astype(str).reset_index(drop=True)
    n = scores.shape[0]
    if n < 3 or labels.nunique() < 2:
        return {"knn_label_purity_micro": np.nan, "knn_label_purity_macro": np.nan}
    nn = min(n_neighbors + 1, n)
    nbrs = NearestNeighbors(n_neighbors=nn, metric="euclidean")
    nbrs.fit(scores)
    indices = nbrs.kneighbors(scores, return_distance=False)[:, 1:]
    same = labels.to_numpy()[indices] == labels.to_numpy()[:, None]
    purity_per_cell = same.mean(axis=1)
    per_label = pd.DataFrame({"label": labels, "purity": purity_per_cell}).groupby("label")["purity"].mean()
    return {
        "knn_label_purity_micro": float(np.mean(purity_per_cell)),
        "knn_label_purity_macro": float(per_label.mean()),
    }


def compute_silhouette(scores: np.ndarray, labels: pd.Series, max_cells: int) -> float:
    idx = _sample_indices(scores.shape[0], max_cells)
    y = labels.astype(str).to_numpy()[idx]
    if len(np.unique(y)) < 2:
        return np.nan
    counts = pd.Series(y).value_counts()
    valid_labels = set(counts[counts >= 2].index)
    keep = np.array([v in valid_labels for v in y])
    if keep.sum() < 3 or len(valid_labels) < 2:
        return np.nan
    x_scaled = StandardScaler().fit_transform(scores[idx][keep])
    return float(silhouette_score(x_scaled, y[keep], metric="euclidean"))


def compute_cell_type_linear_probe(scores: np.ndarray, labels: pd.Series, max_cells: int) -> dict[str, float]:
    idx = _sample_indices(scores.shape[0], max_cells)
    y_raw = labels.astype(str).to_numpy()[idx]
    counts = pd.Series(y_raw).value_counts()
    keep_labels = set(counts[counts >= 5].index)
    keep = np.array([v in keep_labels for v in y_raw])
    if keep.sum() < 20 or len(keep_labels) < 2:
        return {"celltype_probe_balanced_accuracy": np.nan, "celltype_probe_macro_f1": np.nan}

    x_probe = scores[idx][keep]
    y_probe = LabelEncoder().fit_transform(y_raw[keep])
    min_class = int(pd.Series(y_probe).value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return {"celltype_probe_balanced_accuracy": np.nan, "celltype_probe_macro_f1": np.nan}

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, x_probe, y_probe, cv=cv, method="predict")
    return {
        "celltype_probe_balanced_accuracy": float(balanced_accuracy_score(y_probe, y_pred)),
        "celltype_probe_macro_f1": float(f1_score(y_probe, y_pred, average="macro")),
    }


def compute_pseudotime_metrics(
    scores: np.ndarray,
    pseudotime: pd.Series,
    n_neighbors: int,
    max_pair_sample: int,
) -> dict[str, float]:
    pt = pd.to_numeric(pseudotime, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(pt)
    if valid.sum() < 10:
        return {
            "pseudotime_knn_absdiff": np.nan,
            "pseudotime_knn_smoothness": np.nan,
            "pseudotime_distance_spearman": np.nan,
        }
    x_valid = scores[valid]
    pt = pt[valid]
    n = x_valid.shape[0]
    nn = min(n_neighbors + 1, n)
    nbrs = NearestNeighbors(n_neighbors=nn, metric="euclidean")
    nbrs.fit(x_valid)
    indices = nbrs.kneighbors(x_valid, return_distance=False)[:, 1:]
    neighbor_absdiff = np.abs(pt[indices] - pt[:, None]).mean()

    rng = np.random.default_rng(42)
    pair_n = min(max_pair_sample, max(1000, n * 5))
    a = rng.integers(0, n, size=pair_n)
    b = rng.integers(0, n, size=pair_n)
    random_absdiff = np.abs(pt[a] - pt[b]).mean()
    smoothness = 1.0 - (neighbor_absdiff / (random_absdiff + 1e-12))

    pair_n_spearman = min(max_pair_sample, n * (n - 1) // 2)
    a = rng.integers(0, n, size=pair_n_spearman)
    b = rng.integers(0, n, size=pair_n_spearman)
    nonself = a != b
    a = a[nonself]
    b = b[nonself]
    latent_dist = np.linalg.norm(x_valid[a] - x_valid[b], axis=1)
    pt_dist = np.abs(pt[a] - pt[b])
    rho = spearmanr(latent_dist, pt_dist, nan_policy="omit").correlation
    return {
        "pseudotime_knn_absdiff": float(neighbor_absdiff),
        "pseudotime_knn_smoothness": float(smoothness),
        "pseudotime_distance_spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def evaluate_group_cv_binary(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int | None = None) -> dict[str, float]:
    groups = groups.astype(str)
    common = X.index.intersection(y.index).intersection(groups.index)
    X = X.loc[common]
    y = y.loc[common]
    groups = groups.loc[common]
    if y.nunique() < 2 or groups.nunique() < 2:
        return {
            "across_patient_auroc": np.nan,
            "across_patient_auprc": np.nan,
            "across_patient_brier": np.nan,
            "across_patient_log_loss": np.nan,
            "across_patient_n_folds": 0,
        }
    if n_splits is None:
        n_splits = min(5, groups.nunique())
    n_splits = max(2, min(n_splits, groups.nunique()))
    cv = GroupKFold(n_splits=n_splits)
    pred = pd.Series(index=X.index, dtype=float)
    used_folds = 0
    for tr, te in cv.split(X, y, groups=groups):
        y_tr = y.iloc[tr]
        y_te = y.iloc[te]
        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue
        model = make_logreg_pipeline()
        model.fit(X.iloc[tr], y_tr)
        pred.iloc[te] = model.predict_proba(X.iloc[te])[:, 1]
        used_folds += 1
    valid = pred.notna()
    if valid.sum() == 0 or y.loc[valid].nunique() < 2:
        return {
            "across_patient_auroc": np.nan,
            "across_patient_auprc": np.nan,
            "across_patient_brier": np.nan,
            "across_patient_log_loss": np.nan,
            "across_patient_n_folds": used_folds,
        }
    yv = y.loc[valid]
    pv = pred.loc[valid]
    return {
        "across_patient_auroc": float(roc_auc_score(yv, pv)),
        "across_patient_auprc": float(average_precision_score(yv, pv)),
        "across_patient_brier": float(brier_score_loss(yv, pv)),
        "across_patient_log_loss": float(log_loss(yv, np.clip(pv, 1e-6, 1 - 1e-6))),
        "across_patient_n_folds": int(used_folds),
    }


def evaluate_per_patient_binary(
    X: pd.DataFrame,
    y: pd.Series,
    patients: pd.Series,
    min_cells: int = 50,
    min_class_cells: int = 10,
) -> dict[str, float]:
    common = X.index.intersection(y.index).intersection(patients.index)
    X = X.loc[common]
    y = y.loc[common]
    patients = patients.loc[common].astype(str)
    rows = []
    for pid, idx in patients.groupby(patients).groups.items():
        idx = list(idx)
        xp = X.loc[idx]
        yp = y.loc[idx]
        if len(idx) < min_cells or yp.nunique() < 2 or yp.value_counts().min() < min_class_cells:
            continue
        n_splits = min(5, int(yp.value_counts().min()))
        if n_splits < 2:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pred = pd.Series(index=xp.index, dtype=float)
        for tr, te in cv.split(xp, yp):
            model = make_logreg_pipeline()
            model.fit(xp.iloc[tr], yp.iloc[tr])
            pred.iloc[te] = model.predict_proba(xp.iloc[te])[:, 1]
        rows.append(
            {
                "patient": pid,
                "auroc": roc_auc_score(yp, pred),
                "auprc": average_precision_score(yp, pred),
                "brier": brier_score_loss(yp, pred),
                "log_loss": log_loss(yp, np.clip(pred, 1e-6, 1 - 1e-6)),
            }
        )
    if not rows:
        return {
            "within_patient_mean_auroc": np.nan,
            "within_patient_mean_auprc": np.nan,
            "within_patient_mean_brier": np.nan,
            "within_patient_mean_log_loss": np.nan,
            "within_patient_n_patients": 0,
        }
    df = pd.DataFrame(rows)
    return {
        "within_patient_mean_auroc": float(df["auroc"].mean()),
        "within_patient_mean_auprc": float(df["auprc"].mean()),
        "within_patient_mean_brier": float(df["brier"].mean()),
        "within_patient_mean_log_loss": float(df["log_loss"].mean()),
        "within_patient_n_patients": int(df["patient"].nunique()),
    }


def evaluate_latent_scores(scores: np.ndarray, obs_df: pd.DataFrame, panel_meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if scores.shape[0] != obs_df.shape[0]:
        raise ValueError(f"scores rows {scores.shape[0]} != obs rows {obs_df.shape[0]}")
    scores = np.asarray(scores, dtype=float)
    metrics: dict[str, Any] = {
        "n_cells": int(scores.shape[0]),
        "n_components": int(scores.shape[1]),
        "n_gene_sets": int(panel_meta["n_gene_sets"]),
        "n_genes": int(panel_meta["n_genes"]),
    }

    labels = obs_df[CELL_TYPE_COL]
    metrics.update(compute_knn_label_metrics(scores, labels, args.knn_neighbors))
    metrics["celltype_silhouette"] = compute_silhouette(scores, labels, args.max_silhouette_cells)
    if args.run_cell_type_linear_probe:
        metrics.update(compute_cell_type_linear_probe(scores, labels, args.max_linear_probe_cells))

    metrics.update(compute_pseudotime_metrics(scores, obs_df[PSEUDOTIME_COL], args.knn_neighbors, args.max_pair_sample))

    mrd_mask = obs_df[TIMEPOINT_COL].astype(str).isin(MRD_VALUES)
    y = make_binary_labels(obs_df.loc[mrd_mask])
    if not y.empty:
        x_binary = pd.DataFrame(scores[mrd_mask.to_numpy()], index=obs_df.index[mrd_mask])
        x_binary = x_binary.loc[y.index]
        metrics.update(evaluate_group_cv_binary(x_binary, y, obs_df.loc[y.index, PATIENT_COL]))
        metrics.update(evaluate_per_patient_binary(x_binary, y, obs_df.loc[y.index, PATIENT_COL]))
    else:
        metrics.update(evaluate_group_cv_binary(pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=str)))
        metrics.update(evaluate_per_patient_binary(pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=str)))

    return metrics


def write_metadata_contract(adata: sc.AnnData, out_dir: Path) -> pd.DataFrame:
    missing = [c for c in REQUIRED_OBS if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required obs columns: {missing}")

    rows = []
    for col in REQUIRED_OBS:
        s = adata.obs[col]
        row = {"column": col, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()), "n_unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            row.update({"min": float(s.min()), "median": float(s.median()), "max": float(s.max())})
        else:
            top = s.astype(str).value_counts(dropna=False).head(8)
            row["top_values"] = "; ".join(f"{idx}: {val}" for idx, val in top.items())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata_contract.csv", index=False)
    LOGGER.info("Wrote metadata contract: %s", out_dir / "metadata_contract.csv")
    return df


def build_panels(adata: sc.AnnData, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], pd.DataFrame]:
    gmt = parse_gmt(args.gmt_path)
    manifest = pd.read_csv(args.manifest_path, sep="\t")
    manifest["biology_group"] = manifest["why_include"].map(assign_biology_group)

    var_upper_to_name = {str(g).upper(): str(g) for g in adata.var_names}
    geneset_to_covered: dict[str, set[str]] = {}
    coverage_rows = []
    for geneset_name, raw_genes in gmt.items():
        covered_upper = raw_genes & set(var_upper_to_name)
        covered_original = {var_upper_to_name[g] for g in covered_upper}
        geneset_to_covered[geneset_name] = covered_original
        coverage_rows.append(
            {
                "geneset_name": geneset_name,
                "raw_n_genes": len(raw_genes),
                "covered_n_genes": len(covered_original),
                "hit_fraction": len(covered_original) / max(len(raw_genes), 1),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows).merge(manifest, on="geneset_name", how="left")
    coverage_df = coverage_df.sort_values(["priority", "biology_group", "geneset_name"])
    coverage_df.to_csv(args.out_dir / "geneset_coverage.csv", index=False)

    with args.old_gene_lists_path.open() as f:
        old_gene_lists = json.load(f)
    with args.hvg_gene_lists_path.open() as f:
        hvg_gene_lists = json.load(f)
    old_final_genes = {str(g) for g in old_gene_lists.get("final_gene_set", adata.var_names.tolist())}
    hvg_gene_order = [str(g) for g in hvg_gene_lists["final_gene_set"]]
    hvg_final_genes = set(hvg_gene_order)
    old_hvg_overlap = old_final_genes & hvg_final_genes
    old_vs_hvg_summary = pd.DataFrame(
        [
            {
                "old_n_genes": len(old_final_genes),
                "hvg_n_genes": len(hvg_final_genes),
                "overlap_n_genes": len(old_hvg_overlap),
                "old_fraction_in_hvg": len(old_hvg_overlap) / max(len(old_final_genes), 1),
                "hvg_fraction_in_old": len(old_hvg_overlap) / max(len(hvg_final_genes), 1),
                "jaccard": len(old_hvg_overlap) / max(len(old_final_genes | hvg_final_genes), 1),
                "old_only_n_genes": len(old_final_genes - hvg_final_genes),
                "hvg_only_n_genes": len(hvg_final_genes - old_final_genes),
            }
        ]
    )
    old_vs_hvg_summary.to_csv(args.out_dir / "old_vs_hvg_overlap_summary.csv", index=False)

    per_geneset_hvg_overlap_rows = []
    for geneset_name, covered_genes in geneset_to_covered.items():
        covered = set(covered_genes)
        overlap = covered & hvg_final_genes
        per_geneset_hvg_overlap_rows.append(
            {
                "geneset_name": geneset_name,
                "covered_n_genes": len(covered),
                "overlap_hvg_n_genes": len(overlap),
                "covered_fraction_in_hvg": len(overlap) / max(len(covered), 1),
                "covered_only_old_n_genes": len(covered - hvg_final_genes),
                "overlap_hvg_genes": ";".join(sorted(overlap)),
                "covered_only_old_genes": ";".join(sorted(covered - hvg_final_genes)),
            }
        )
    per_geneset_hvg_overlap_df = (
        pd.DataFrame(per_geneset_hvg_overlap_rows)
        .merge(manifest, on="geneset_name", how="left")
        .sort_values(["covered_fraction_in_hvg", "covered_n_genes"], ascending=[True, False])
    )
    per_geneset_hvg_overlap_df.to_csv(args.out_dir / "per_geneset_hvg_overlap.csv", index=False)

    all_sets = list(manifest["geneset_name"])
    unknown_sets = sorted(set(all_sets) - set(gmt))
    if unknown_sets:
        raise ValueError(f"Manifest genesets missing from GMT: {unknown_sets}")

    panels: list[dict[str, Any]] = []
    panels.append(panel_row("full_34", "full_control", all_sets, "Full 34-gene-set union control.", geneset_to_covered))

    core_sets = manifest.loc[manifest["priority"].astype(str).str.lower().eq("core"), "geneset_name"].tolist()
    panels.append(panel_row("core_only", "core_only", core_sets, "Manifest-priority Core genesets only.", geneset_to_covered))

    ablation_audit_rows = []
    for group, group_df in manifest.groupby("biology_group", sort=True):
        drop_sets = sorted(group_df["geneset_name"])
        keep_sets = [gs for gs in all_sets if gs not in set(drop_sets)]
        excluded_genes = covered_genes_for_sets(drop_sets, geneset_to_covered)
        panels.append(
            panel_row(
                f"drop_group__{safe_id(group)}",
                "group_dropout_strict_gene_ablation",
                keep_sets,
                f"Full panel minus biology group: {group}; dropped group genes are removed even if shared.",
                geneset_to_covered,
                excluded_genes=set(excluded_genes),
                excluded_gene_sets=drop_sets,
            )
        )

    for gs in all_sets:
        keep_sets = [x for x in all_sets if x != gs]
        excluded_genes = set(geneset_to_covered[gs])
        panel_id = f"drop_geneset__{safe_id(gs)}"
        panels.append(
            panel_row(
                panel_id,
                "leave_one_geneset_out_strict_gene_ablation",
                keep_sets,
                f"Full panel minus geneset: {gs}; dropped genes are removed even if shared with retained sets.",
                geneset_to_covered,
                excluded_genes=excluded_genes,
                excluded_gene_sets=[gs],
            )
        )
        for gene in sorted(excluded_genes):
            retained_sets_with_gene = [retained for retained in keep_sets if gene in geneset_to_covered[retained]]
            if retained_sets_with_gene:
                ablation_audit_rows.append(
                    {
                        "panel_id": panel_id,
                        "dropped_geneset": gs,
                        "removed_gene": gene,
                        "also_present_in_retained_genesets": ";".join(retained_sets_with_gene),
                        "n_retained_genesets_with_gene": len(retained_sets_with_gene),
                    }
                )
    pd.DataFrame(ablation_audit_rows).to_csv(args.out_dir / "strict_ablation_shared_gene_audit.csv", index=False)

    if panel_family_enabled(args, "bottom_up"):
        for gs in all_sets:
            panels.append(
                panel_row(
                    f"single_geneset__{safe_id(gs)}",
                    "single_geneset_only",
                    [gs],
                    f"Bottom-up panel with only geneset: {gs}.",
                    geneset_to_covered,
                )
            )

        for group, group_df in manifest.groupby("biology_group", sort=True):
            group_sets = sorted(group_df["geneset_name"])
            panels.append(
                panel_row(
                    f"single_group__{safe_id(group)}",
                    "single_group_only",
                    group_sets,
                    f"Bottom-up panel with only biology group: {group}.",
                    geneset_to_covered,
                )
            )

    jaccard_threshold = args.jaccard_threshold
    selected_redundancy: list[str] = []
    priority_rank = {"Core": 0, "Optional": 1}
    ordered_for_redundancy = (
        coverage_df.assign(priority_rank=coverage_df["priority"].map(priority_rank).fillna(9))
        .sort_values(["priority_rank", "covered_n_genes"], ascending=[True, False])["geneset_name"]
        .tolist()
    )
    redundancy_decisions = []
    for gs in ordered_for_redundancy:
        covered = geneset_to_covered[gs]
        priority = str(manifest.loc[manifest["geneset_name"].eq(gs), "priority"].iloc[0])
        max_j = 0.0
        max_with = None
        for kept in selected_redundancy:
            kept_genes = geneset_to_covered[kept]
            denom = len(covered | kept_genes)
            j = len(covered & kept_genes) / denom if denom else 0.0
            if j > max_j:
                max_j = j
                max_with = kept
        keep = priority == "Core" or max_j < jaccard_threshold
        if keep:
            selected_redundancy.append(gs)
        redundancy_decisions.append(
            {
                "geneset_name": gs,
                "priority": priority,
                "kept": keep,
                "max_jaccard_to_kept": max_j,
                "most_similar_kept": max_with,
            }
        )
    pd.DataFrame(redundancy_decisions).to_csv(args.out_dir / "redundancy_pruning_decisions.csv", index=False)
    panels.append(
        panel_row(
            f"redundancy_pruned_jaccard{int(jaccard_threshold * 100):02d}",
            "redundancy_pruned",
            selected_redundancy,
            f"Core-preserving optional-set pruning with Jaccard threshold {jaccard_threshold:.2f}.",
            geneset_to_covered,
        )
    )

    if panel_family_enabled(args, "hvg_controls"):
        available_genes = set(map(str, adata.var_names))
        hvg_available = ordered_available_genes(hvg_gene_order, available_genes)
        panels.append(
            direct_gene_panel_row(
                "all_filtered_current_adata",
                "all_filtered_control",
                list(map(str, adata.var_names)),
                "All genes available in the current preprocessed AnnData; use as a broad data-driven stage-0 control.",
                panel_family="data_driven",
                source_dictionary="current_adata_all_filtered",
            )
        )

        for n in args.hvg_anchor_sizes:
            if n <= 0:
                continue
            genes = hvg_available[: min(n, len(hvg_available))]
            panels.append(
                direct_gene_panel_row(
                    f"hvg_top_requested_{n}__available_{len(genes)}",
                    "hvg_anchor_control",
                    genes,
                    f"Top {len(genes)} available HVG genes from the HVG Plan 0 reference.",
                    panel_family="data_driven",
                    source_dictionary="hvg_plan0_reference",
                )
            )

        size_match_types = set(args.hvg_size_match_panel_types)
        match_sizes = sorted({int(p["n_genes"]) for p in panels if p["panel_type"] in size_match_types and int(p["n_genes"]) > 0})
        for n in match_sizes:
            genes = hvg_available[: min(n, len(hvg_available))]
            panels.append(
                direct_gene_panel_row(
                    f"hvg_size_matched_requested_{n}__available_{len(genes)}",
                    "hvg_size_matched_control",
                    genes,
                    f"Top {len(genes)} available HVG genes; size-matched to at least one knowledge-driven panel.",
                    panel_family="data_driven",
                    source_dictionary="hvg_plan0_reference",
                )
            )

    panel_manifest = pd.DataFrame(
        [
            {
                "panel_id": p["panel_id"],
                "panel_type": p["panel_type"],
                "panel_family": p.get("panel_family", ""),
                "source_dictionary": p.get("source_dictionary", ""),
                "description": p["description"],
                "n_gene_sets": p["n_gene_sets"],
                "n_genes": p["n_genes"],
                "n_excluded_genes": p.get("n_excluded_genes", 0),
                "n_shared_genes_removed": p.get("n_shared_genes_removed", 0),
                "matched_panel_id": p.get("matched_panel_id", ""),
                "hvg_overlap_n_genes": len(set(p["genes"]) & hvg_final_genes),
                "hvg_overlap_fraction": len(set(p["genes"]) & hvg_final_genes) / max(len(p["genes"]), 1),
                "full34_overlap_n_genes": len(set(p["genes"]) & set(covered_genes_for_sets(all_sets, geneset_to_covered))),
                "full34_overlap_fraction": len(set(p["genes"]) & set(covered_genes_for_sets(all_sets, geneset_to_covered))) / max(len(p["genes"]), 1),
                "genesets": ";".join(p["genesets"]),
                "excluded_gene_sets": ";".join(p.get("excluded_gene_sets", [])),
                "excluded_genes": ";".join(p.get("excluded_genes", [])),
                "genes": ";".join(p["genes"]),
            }
            for p in panels
        ]
    ).sort_values(["panel_type", "panel_id"])
    panel_manifest.to_csv(args.out_dir / "panel_manifest.csv", index=False)

    LOGGER.info("Wrote %d panels to %s", len(panel_manifest), args.out_dir / "panel_manifest.csv")
    return panels, {p["panel_id"]: p for p in panels}, panel_manifest


def filter_panels(panels: list[dict[str, Any]], panel_ids: list[str], panel_types: list[str]) -> list[dict[str, Any]]:
    selected = panels
    if panel_ids:
        wanted = set(panel_ids)
        selected = [p for p in selected if p["panel_id"] in wanted]
    if panel_types:
        wanted_types = set(panel_types)
        selected = [p for p in selected if p["panel_type"] in wanted_types]
    return selected


def existing_full_control_scores_path(args: argparse.Namespace, method: str, k: int, seed: int) -> Path:
    return args.stability_dir / method / f"k_{k}" / "replicates" / f"seed_{seed}" / "scores.npy"


def cache_paths(args: argparse.Namespace, panel_id: str, method: str, k: int, seed: int) -> tuple[Path, Path]:
    base = args.dr_cache_dir / panel_id / method / f"k_{k}" / f"seed_{seed}"
    return base / "scores.npy", base / "metadata.json"


def load_or_run_scores(
    adata: sc.AnnData,
    panel: dict[str, Any],
    method: str,
    k: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    from run_gene_filter_dr_grid import _run_dr_method

    panel_id = panel["panel_id"]
    score_cache, meta_cache = cache_paths(args, panel_id, method, k, seed)

    if panel_id == "full_34":
        source_path = existing_full_control_scores_path(args, method, k, seed)
        if source_path.exists() and not args.rerun_dr:
            scores = np.load(source_path)
            return scores, {"score_source": str(source_path), "reused_existing_plan0": True, "status": "ok"}

    if score_cache.exists() and meta_cache.exists() and not args.rerun_dr:
        scores = np.load(score_cache)
        meta = json.loads(meta_cache.read_text())
        meta["score_source"] = str(score_cache)
        meta["reused_existing_plan0"] = False
        meta["status"] = "ok"
        return scores, meta

    genes = [g for g in panel["genes"] if g in adata.var_names]
    if len(genes) <= k:
        return None, {
            "status": "skipped_too_few_genes",
            "reason": f"panel has {len(genes)} covered genes for k={k}",
            "score_source": None,
            "reused_existing_plan0": False,
        }
    if adata.n_obs <= k:
        return None, {
            "status": "skipped_too_few_cells",
            "reason": f"adata has {adata.n_obs} cells for k={k}",
            "score_source": None,
            "reused_existing_plan0": False,
        }

    ad_sub = adata[:, genes].copy()
    scores, _loadings, extras = _run_dr_method(
        method=method,
        adata=ad_sub,
        k=k,
        seed=seed,
        factosig_max_iter=args.factosig_max_iter,
    )
    score_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(score_cache, scores)
    meta = {
        "panel_id": panel_id,
        "method": method,
        "k": int(k),
        "seed": int(seed),
        "n_cells": int(scores.shape[0]),
        "n_genes": int(len(genes)),
        "n_gene_sets": int(panel["n_gene_sets"]),
        "panel_type": panel.get("panel_type"),
        "panel_family": panel.get("panel_family"),
        "source_dictionary": panel.get("source_dictionary"),
        "matched_panel_id": panel.get("matched_panel_id"),
        "n_excluded_genes": int(panel.get("n_excluded_genes", 0)),
        "n_shared_genes_removed": int(panel.get("n_shared_genes_removed", 0)),
        "excluded_gene_sets": panel.get("excluded_gene_sets", []),
        "excluded_genes": panel.get("excluded_genes", []),
        "genes": genes,
        "score_source": str(score_cache),
        "reused_existing_plan0": False,
        "status": "ok",
        "extras_keys": sorted(list(extras.keys())),
    }
    meta_cache.write_text(json.dumps(meta, indent=2, default=str))
    return scores, meta


def evaluate_panel_grid(adata: sc.AnnData, obs: pd.DataFrame, panels_to_run: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(panels_to_run) * len(args.methods) * len(args.ks)
    completed = 0
    partial_path = args.out_dir / "panel_dr_metric_rows.partial.csv"
    for panel in panels_to_run:
        for method in args.methods:
            for k in args.ks:
                completed += 1
                started = time.time()
                label = f"[{completed}/{total}] {panel['panel_id']} | {method} | k={k}"
                LOGGER.info("Starting %s", label)
                row = {
                    "panel_id": panel["panel_id"],
                    "panel_type": panel["panel_type"],
                    "panel_family": panel.get("panel_family", ""),
                    "source_dictionary": panel.get("source_dictionary", ""),
                    "matched_panel_id": panel.get("matched_panel_id", ""),
                    "method": method,
                    "k": int(k),
                    "seed": int(args.seed),
                    "description": panel["description"],
                    "n_gene_sets": int(panel["n_gene_sets"]),
                    "n_genes": int(panel["n_genes"]),
                }
                try:
                    scores, meta = load_or_run_scores(adata, panel, method, k, args.seed, args)
                    row.update(meta)
                    if scores is None:
                        LOGGER.warning("Skipped %s: %s", label, meta.get("reason"))
                        rows.append(row)
                        continue
                    metric_row = evaluate_latent_scores(scores, obs, panel, args)
                    row.update(metric_row)
                    row["status"] = "ok"
                    LOGGER.info("Finished %s in %.1fs", label, time.time() - started)
                except Exception as exc:
                    row.update({"status": "failed", "error": repr(exc)})
                    LOGGER.exception("Failed %s", label)
                rows.append(row)
                pd.DataFrame(rows).to_csv(partial_path, index=False)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(args.out_dir / "panel_dr_metric_rows.csv", index=False)
    LOGGER.info("Wrote metrics: %s", args.out_dir / "panel_dr_metric_rows.csv")
    return metrics_df


def build_leaderboard(metrics_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ok = metrics_df.loc[metrics_df["status"].eq("ok")].copy()
    if ok.empty:
        ok.to_csv(out_dir / "panel_dr_leaderboard.csv", index=False)
        return ok

    baseline_cols = [
        "method",
        "k",
        "across_patient_auroc",
        "across_patient_auprc",
        "knn_label_purity_macro",
        "pseudotime_knn_smoothness",
        "n_genes",
        "n_gene_sets",
    ]
    baseline = ok.loc[ok["panel_id"].eq("full_34"), baseline_cols].rename(
        columns={
            "across_patient_auroc": "baseline_across_patient_auroc",
            "across_patient_auprc": "baseline_across_patient_auprc",
            "knn_label_purity_macro": "baseline_knn_label_purity_macro",
            "pseudotime_knn_smoothness": "baseline_pseudotime_knn_smoothness",
            "n_genes": "baseline_n_genes",
            "n_gene_sets": "baseline_n_gene_sets",
        }
    )
    out = ok.merge(baseline, on=["method", "k"], how="left")
    for metric in ["across_patient_auroc", "across_patient_auprc", "knn_label_purity_macro", "pseudotime_knn_smoothness"]:
        out[f"delta_{metric}"] = out[metric] - out[f"baseline_{metric}"]
    out["gene_fraction_vs_full"] = out["n_genes"] / out["baseline_n_genes"]
    out["geneset_fraction_vs_full"] = out["n_gene_sets"] / out["baseline_n_gene_sets"]

    score_cols = [
        "across_patient_auroc",
        "across_patient_auprc",
        "knn_label_purity_macro",
        "pseudotime_knn_smoothness",
    ]
    rank_frame = out[score_cols].copy()
    for c in score_cols:
        rank_frame[c] = rank_frame[c].rank(pct=True, na_option="bottom")
    out["metric_rank_score"] = rank_frame.mean(axis=1)
    out["parsimony_bonus"] = 1.0 - out["gene_fraction_vs_full"].clip(upper=1.0)
    out["triage_score"] = out["metric_rank_score"] + 0.15 * out["parsimony_bonus"]
    out = out.sort_values(["triage_score", "metric_rank_score"], ascending=False)
    out.to_csv(out_dir / "panel_dr_leaderboard.csv", index=False)
    LOGGER.info("Wrote leaderboard: %s", out_dir / "panel_dr_leaderboard.csv")
    return out


def write_leaderboard_plot(leaderboard_df: pd.DataFrame, out_dir: Path) -> None:
    if leaderboard_df.empty:
        return
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plot_df = leaderboard_df.loc[leaderboard_df["status"].eq("ok")].copy()
    if plot_df.empty:
        return
    top_panels = plot_df.groupby("panel_id", as_index=False)["triage_score"].max().nlargest(20, "triage_score")["panel_id"]
    compact = plot_df.loc[plot_df["panel_id"].isin(top_panels)]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    sns.scatterplot(data=compact, x="gene_fraction_vs_full", y="across_patient_auroc", hue="panel_type", style="method", ax=axes[0, 0])
    axes[0, 0].set_title("Across-patient malignant-vs-normal AUROC")
    sns.scatterplot(data=compact, x="gene_fraction_vs_full", y="knn_label_purity_macro", hue="panel_type", style="method", ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("Cell-type kNN purity")
    sns.scatterplot(data=compact, x="gene_fraction_vs_full", y="pseudotime_knn_smoothness", hue="panel_type", style="method", ax=axes[1, 0], legend=False)
    axes[1, 0].set_title("Pseudotime neighborhood smoothness")
    sns.boxplot(data=plot_df, x="panel_type", y="triage_score", ax=axes[1, 1])
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_title("Triage score by pruning family")
    out_path = out_dir / "leaderboard_metric_summary.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Wrote plot: %s", out_path)


def _metric_gate_value(row: pd.Series, metric: str, min_delta: float) -> bool:
    value = row.get(metric)
    base = row.get(f"baseline_{metric}")
    if pd.isna(value) or pd.isna(base):
        return True
    return (value - base) >= min_delta


def passes_umap_gate(row: pd.Series, args: argparse.Namespace) -> bool:
    if row["panel_id"] == "full_34":
        return True
    if row.get("status") != "ok":
        return False
    metric_ok = (
        _metric_gate_value(row, "across_patient_auroc", args.auroc_delta_min)
        and _metric_gate_value(row, "across_patient_auprc", args.auprc_delta_min)
        and _metric_gate_value(row, "knn_label_purity_macro", args.knn_purity_delta_min)
        and _metric_gate_value(row, "pseudotime_knn_smoothness", args.pseudotime_smoothness_delta_min)
    )
    parsimony_ok = row.get("gene_fraction_vs_full", 1.0) <= args.max_gene_fraction_for_pruned_panel
    return bool(metric_ok and parsimony_ok)


def select_umap_rows(leaderboard_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if leaderboard_df.empty:
        selection = leaderboard_df.copy()
        selection.to_csv(args.out_dir / "umap_selection.csv", index=False)
        return selection
    candidates = leaderboard_df.copy()
    candidates["passes_umap_gate"] = candidates.apply(lambda row: passes_umap_gate(row, args), axis=1)
    passed_pruned = candidates.loc[candidates["passes_umap_gate"] & ~candidates["panel_id"].eq("full_34")].sort_values("triage_score", ascending=False)
    selected_pruned = passed_pruned.head(args.max_umap_rows)

    selected_controls = []
    for _, row in selected_pruned.iterrows():
        control = candidates.loc[
            candidates["panel_id"].eq("full_34")
            & candidates["method"].eq(row["method"])
            & candidates["k"].eq(row["k"])
        ]
        selected_controls.append(control)
    if selected_controls:
        selected_controls_df = pd.concat(selected_controls, axis=0).drop_duplicates(subset=["panel_id", "method", "k"])
    else:
        selected_controls_df = candidates.loc[candidates["panel_id"].eq("full_34")].head(0)
    selection = pd.concat([selected_controls_df, selected_pruned], axis=0, ignore_index=True)
    selection.to_csv(args.out_dir / "umap_selection.csv", index=False)
    LOGGER.info("Wrote UMAP selection with %d rows: %s", len(selection), args.out_dir / "umap_selection.csv")
    return selection


def make_umap_for_row(adata: sc.AnnData, obs: pd.DataFrame, panel_lookup: dict[str, dict[str, Any]], row: pd.Series, args: argparse.Namespace) -> ad.AnnData:
    panel = panel_lookup[row["panel_id"]]
    scores, meta = load_or_run_scores(adata, panel, row["method"], int(row["k"]), int(row.get("seed", args.seed)), args)
    if scores is None:
        raise ValueError(f"No scores available for {row['panel_id']} {row['method']} k={row['k']}: {meta}")
    ad_lat = ad.AnnData(X=np.asarray(scores), obs=obs.copy())
    sc.pp.neighbors(ad_lat, n_neighbors=15, use_rep="X")
    sc.tl.umap(ad_lat, random_state=42)
    return ad_lat


def write_gated_umaps(adata: sc.AnnData, obs: pd.DataFrame, panel_lookup: dict[str, dict[str, Any]], selection_df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.umap_dir.mkdir(parents=True, exist_ok=True)
    for _, row in selection_df.iterrows():
        panel_id = row["panel_id"]
        method = row["method"]
        k = int(row["k"])
        LOGGER.info("Writing UMAP: %s | %s | k=%d", panel_id, method, k)
        ad_lat = make_umap_for_row(adata, obs, panel_lookup, row, args)
        fig, axes = plt.subplots(1, 4, figsize=(28, 6.5), constrained_layout=True)
        sc.pl.umap(ad_lat, color=CN_LABEL_COL, ax=axes[0], show=False, title=f"{panel_id}\n{method} k={k} | CN.label")
        sc.pl.umap(ad_lat, color=CELL_TYPE_COL, ax=axes[1], show=False, title="predicted.annotation", legend_loc="right margin")
        sc.pl.umap(ad_lat, color=PSEUDOTIME_COL, ax=axes[2], show=False, title="predicted.pseudotime", color_map="viridis")
        sc.pl.umap(ad_lat, color=PATIENT_COL, ax=axes[3], show=False, title="patient", legend_loc="right margin")
        for ax in axes:
            ax.set_aspect("equal", adjustable="datalim")
        out_path = args.umap_dir / f"{panel_id}__{method}__k{k}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_sc_root = Path("/home/minhang/mds_project/sc_classification")
    default_exp = default_sc_root / "experiments" / "20260401_023024_plan0_k_sweep_60_none_all_filtered_8f5363e0"
    default_hvg = default_sc_root / "experiments" / "20260211_212806_plan0_k_sweep_60_none_hvg_c06f4886"
    parser.add_argument("--sc-root", type=Path, default=default_sc_root)
    parser.add_argument("--exp-dir", type=Path, default=default_exp)
    parser.add_argument("--hvg-exp-dir", type=Path, default=default_hvg)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--methods", default="fa,factosig,pca")
    parser.add_argument("--ks", default="20,40,60")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--panel-families",
        default="top_down,bottom_up,hvg_controls",
        help=(
            "Comma/space-separated panel families to materialize. "
            "top_down is always retained for backward compatibility; optional additions are bottom_up and hvg_controls."
        ),
    )
    parser.add_argument("--panel-ids", default="", help="Optional comma/space-separated panel IDs to run.")
    parser.add_argument("--panel-types", default="", help="Optional comma/space-separated panel types to run.")
    parser.add_argument("--hvg-anchor-sizes", default="500,1000,3000,10000")
    parser.add_argument(
        "--hvg-size-match-panel-types",
        default="single_geneset_only,single_group_only,full_control,core_only",
        help="Panel types whose n_genes should define size-matched HVG controls.",
    )
    parser.add_argument("--rerun-dr", action="store_true")
    parser.add_argument("--make-umaps", action="store_true", help="Also generate gated UMAP PNGs after the leaderboard.")
    parser.add_argument("--skip-grid", action="store_true", help="Only rebuild metadata/panel files and downstream summaries from existing metrics.")
    parser.add_argument("--run-cell-type-linear-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--factosig-max-iter", type=int, default=300)
    parser.add_argument("--max-silhouette-cells", type=int, default=5000)
    parser.add_argument("--max-linear-probe-cells", type=int, default=12000)
    parser.add_argument("--max-pair-sample", type=int, default=60000)
    parser.add_argument("--knn-neighbors", type=int, default=30)
    parser.add_argument("--jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--auroc-delta-min", type=float, default=-0.02)
    parser.add_argument("--auprc-delta-min", type=float, default=-0.02)
    parser.add_argument("--knn-purity-delta-min", type=float, default=-0.03)
    parser.add_argument("--pseudotime-smoothness-delta-min", type=float, default=-0.03)
    parser.add_argument("--max-gene-fraction-for-pruned-panel", type=float, default=0.95)
    parser.add_argument("--max-umap-rows", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.methods = split_csv(args.methods, str)
    args.ks = split_csv(args.ks, int)
    args.panel_families = split_csv(args.panel_families, str)
    args.panel_ids = split_csv(args.panel_ids, str)
    args.panel_types = split_csv(args.panel_types, str)
    args.hvg_anchor_sizes = split_csv(args.hvg_anchor_sizes, int)
    args.hvg_size_match_panel_types = split_csv(args.hvg_size_match_panel_types, str)
    args.preprocessed = args.exp_dir / "preprocessing" / "adata_processed.h5ad"
    args.plan0_dir = args.exp_dir / "analysis" / "plan0"
    args.stability_dir = args.plan0_dir / "stability"
    args.gmt_path = args.sc_root / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "genesets_v1.gmt"
    args.manifest_path = args.sc_root / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "manifest.tsv"
    args.old_gene_lists_path = args.exp_dir / "preprocessing" / "gene_lists_at_each_filtering_steps.json"
    args.hvg_gene_lists_path = args.hvg_exp_dir / "preprocessing" / "gene_lists_at_each_filtering_steps.json"
    if args.out_dir is None:
        args.out_dir = args.exp_dir / "analysis" / "old_geneset_pruning_metrics"
    args.dr_cache_dir = args.out_dir / "dr_cache_strict_ablation"
    args.umap_dir = args.out_dir / "umaps"
    for p in (args.out_dir, args.dr_cache_dir, args.umap_dir):
        p.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    parser = build_parser()
    args = normalize_args(parser.parse_args())
    configure_logging(args.out_dir, verbose=args.verbose)
    add_import_paths(args.sc_root)

    LOGGER.info("Arguments: %s", vars(args))
    LOGGER.info("Loading AnnData: %s", args.preprocessed)
    adata = sc.read_h5ad(args.preprocessed)
    obs = adata.obs.copy()
    LOGGER.info("AnnData shape: %s", adata.shape)

    write_metadata_contract(adata, args.out_dir)
    panels, panel_lookup, _panel_manifest = build_panels(adata, args)
    panels_to_run = filter_panels(panels, args.panel_ids, args.panel_types)
    LOGGER.info("Selected %d/%d panels for grid", len(panels_to_run), len(panels))
    if not panels_to_run:
        raise ValueError("No panels selected. Check --panel-ids or --panel-types.")

    metrics_path = args.out_dir / "panel_dr_metric_rows.csv"
    if args.skip_grid:
        LOGGER.info("Skipping grid and reading existing metrics: %s", metrics_path)
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = evaluate_panel_grid(adata, obs, panels_to_run, args)

    leaderboard_df = build_leaderboard(metrics_df, args.out_dir)
    write_leaderboard_plot(leaderboard_df, args.out_dir)
    selection_df = select_umap_rows(leaderboard_df, args)
    if args.make_umaps:
        write_gated_umaps(adata, obs, panel_lookup, selection_df, args)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
