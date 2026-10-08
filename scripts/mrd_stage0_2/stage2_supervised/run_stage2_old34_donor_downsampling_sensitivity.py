#!/usr/bin/env python
"""Overnight expansion of old34 donor-healthy downsampling sensitivity (v2).

Copies the frozen D10/D11 method from
``stage2_old34_donor_downsampling_sensitivity.ipynb``. It does not overwrite
``donor_sensitivity_v1`` and does not dump full coefficient paths.

Claim boundary: this still diagnoses fixed-regularization collapse versus
abundance sensitivity. It is not a condition-specific C-rescue / donor-shortcut
test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("stage2_old34_donor_downsampling_sensitivity")
_STAGE2_DIR = Path(__file__).resolve().parent
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))
from run_stage2_mrd_multiobjective_scorecard import fit_model, fit_preprocess_transform  # noqa: E402

EXPERIMENT_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
DIRECT_BRANCH = "old34_direct_gene_all_v1"
OUTPUT_BRANCH = "donor_sensitivity_v2"
IFNA_PANEL = "single_geneset__hallmark_interferon_alpha_response"
P53_PANEL = "single_geneset__hallmark_p53_pathway"
STRESS_PANEL = "single_group__stress_arrest"
SUPPORTED_PATIENTS = ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]
DONOR_FRACTIONS = [1.0, 0.5, 0.25, 0.1]
DOWNSAMPLE_SEEDS = [42, 43, 44]
SMOKE_CONDITIONS = [(1.0, 42), (0.25, 42)]
MIN_DONOR_CELLS_PER_STRATUM = 5
COEF_TOLERANCE = 1e-8
N_CELLS = 60_118
N_DONOR_HEALTHY = 58_499
N_RECIPIENT_HEALTHY = 50
DEFAULT_INPUT_H5AD = Path(
    "/home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad"
)
QA_MAX_ABS_AP_ERROR = 1e-3
CHECKPOINT_VERSION = 1

V1_FIXED_ALIASES = [
    {
        "target_id": "stress_best_dr",
        "target_label": "Stress arrest | best DR",
        "representation_id": "single_group__stress_arrest|dr|factosig|40|42",
    },
    {
        "target_id": "stress_direct",
        "target_label": "Stress arrest | direct genes",
        "representation_id": "single_group__stress_arrest|direct_gene|direct_gene|685|42",
    },
    {
        "target_id": "p53_best_dr",
        "target_label": "p53 | best DR",
        "representation_id": "single_geneset__hallmark_p53_pathway|dr|fa|10|42",
    },
    {
        "target_id": "p53_direct",
        "target_label": "p53 | direct genes",
        "representation_id": "single_geneset__hallmark_p53_pathway|direct_gene|direct_gene|164|42",
    },
    {
        "target_id": "ifna_pca40",
        "target_label": "IFN-α | PCA40",
        "representation_id": "single_geneset__hallmark_interferon_alpha_response|dr|pca|40|42",
    },
    {
        "target_id": "ifna_fa40",
        "target_label": "IFN-α | FA40",
        "representation_id": "single_geneset__hallmark_interferon_alpha_response|dr|fa|40|42",
    },
    {
        "target_id": "ifna_direct",
        "target_label": "IFN-α | direct genes",
        "representation_id": "single_geneset__hallmark_interferon_alpha_response|direct_gene|direct_gene|92|42",
    },
]
V1_BASIS_TARGET_IDS = ["ifna_pca40", "ifna_fa40"]
METHOD_LABELS = {
    "pca": "PCA",
    "fa": "FA",
    "factosig": "FactoSig",
    "factosig_promax": "FactoSig-Promax",
    "direct_gene": "direct genes",
}
SUPERVISED_COLUMNS = [
    "target_id",
    "representation_id",
    "heldout_patient",
    "feature_id",
    "coefficient",
]
CHECKPOINT_CSV_SCHEMAS = {
    "supervised": SUPERVISED_COLUMNS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_repo_root() -> Path:
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "sc_classification").exists():
            return candidate
    raise FileNotFoundError("Could not locate sc_classification repository root")


def condition_grid(smoke: bool = False) -> list[tuple[float, int]]:
    if smoke:
        return list(SMOKE_CONDITIONS)
    return [(1.0, 42)] + [
        (fraction, seed)
        for fraction in DONOR_FRACTIONS
        if fraction < 1.0
        for seed in DOWNSAMPLE_SEEDS
    ]


def safe_checkpoint_stem(target_id: str) -> str:
    return str(target_id).replace("|", "__").replace("/", "_")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    elif frame.columns.empty:
        raise ValueError(f"Refusing to write a columnless CSV that pandas cannot re-read: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def read_checkpoint_csv(path: Path, key: str = "") -> pd.DataFrame:
    """Read a checkpoint CSV. Headerless/empty files are treated as zero-row tables."""
    schema = CHECKPOINT_CSV_SCHEMAS.get(key, [])
    if not path.exists() or path.stat().st_size <= 1:
        if schema and path.exists():
            repaired = pd.DataFrame(columns=schema)
            write_csv_atomic(repaired, path, columns=schema)
            LOGGER.warning("Repaired unreadable empty checkpoint CSV: %s", path)
            return repaired
        return pd.DataFrame(columns=schema)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        if schema:
            repaired = pd.DataFrame(columns=schema)
            write_csv_atomic(repaired, path, columns=schema)
            LOGGER.warning("Repaired unreadable empty checkpoint CSV: %s", path)
            return repaired
        LOGGER.warning("Skipping unreadable empty checkpoint CSV: %s", path)
        return pd.DataFrame()


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-15:
        return np.nan
    return float(np.dot(left, right) / denominator)


def jaccard_active(left: np.ndarray, right: np.ndarray, tolerance: float = COEF_TOLERANCE) -> float:
    left_set = set(np.flatnonzero(np.abs(left) > tolerance))
    right_set = set(np.flatnonzero(np.abs(right) > tolerance))
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def score_correlation_matrix(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    current_centered = current - current.mean(axis=0, keepdims=True)
    reference_norm = np.linalg.norm(reference_centered, axis=0, keepdims=True)
    current_norm = np.linalg.norm(current_centered, axis=0, keepdims=True)
    return (reference_centered / np.maximum(reference_norm, 1e-12)).T @ (
        current_centered / np.maximum(current_norm, 1e-12)
    )


def top_gene_jaccard(left: np.ndarray, right: np.ndarray, top_n: int = 10) -> float:
    left_set = set(np.argsort(np.abs(left))[-top_n:])
    right_set = set(np.argsort(np.abs(right))[-top_n:])
    return len(left_set & right_set) / len(left_set | right_set)


def setting_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "penalty": str(row["penalty"]),
        "C": float(row["C"]),
        "l1_ratio": pd.to_numeric(pd.Series([row["l1_ratio"]]), errors="coerce").iloc[0],
    }


def method_display(method: str, k: int | None = None) -> str:
    label = METHOD_LABELS.get(str(method), str(method))
    if method == "direct_gene" or k is None:
        return label
    return f"{label}{int(k)}"


def load_cohort(input_h5ad: Path) -> dict[str, Any]:
    adata_backed = ad.read_h5ad(input_h5ad, backed="r")
    obs = adata_backed.obs
    mrd_mask = (
        obs["Tech"].astype(str).eq("CITE")
        & obs["timepoint_type"].astype(str).eq("MRD")
        & obs["CN.label"].astype(str).isin(["cancer", "normal"])
    )
    mrd_obs = obs.loc[mrd_mask, ["patient", "CN.label", "source", "predicted.annotation"]].copy()
    mrd_obs.insert(0, "cell_id", mrd_obs.index.astype(str))
    for column in ["patient", "CN.label", "source", "predicted.annotation"]:
        mrd_obs[column] = mrd_obs[column].astype(str)
    adata_backed.file.close()

    cell_labels = mrd_obs["CN.label"].eq("cancer").astype(int).to_numpy()
    cell_patients = mrd_obs["patient"].to_numpy()
    donor_healthy = (mrd_obs["CN.label"].eq("normal") & mrd_obs["source"].eq("donor")).to_numpy()
    recipient_healthy = (mrd_obs["CN.label"].eq("normal") & mrd_obs["source"].eq("recipient")).to_numpy()
    if len(mrd_obs) != N_CELLS:
        raise AssertionError(f"Expected {N_CELLS} MRD cells, found {len(mrd_obs)}")
    if int(donor_healthy.sum()) != N_DONOR_HEALTHY:
        raise AssertionError(f"Expected {N_DONOR_HEALTHY} donor-healthy cells")
    if int(recipient_healthy.sum()) != N_RECIPIENT_HEALTHY:
        raise AssertionError(f"Expected {N_RECIPIENT_HEALTHY} recipient-healthy cells")
    return {
        "mrd_obs": mrd_obs,
        "cell_labels": cell_labels,
        "cell_patients": cell_patients,
        "donor_healthy": donor_healthy,
        "recipient_healthy": recipient_healthy,
    }


def donor_keep_mask(
    mrd_obs: pd.DataFrame,
    donor_healthy: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")
    keep = np.ones(len(mrd_obs), dtype=bool)
    if fraction == 1:
        return keep
    rng = np.random.default_rng(seed)
    candidate_indices = np.flatnonzero(donor_healthy)
    strata = mrd_obs.iloc[candidate_indices][["patient", "predicted.annotation"]].copy()
    strata["row_index"] = candidate_indices
    for _, group in strata.groupby(["patient", "predicted.annotation"], sort=True, observed=True):
        indices = group["row_index"].to_numpy(dtype=int)
        n_keep = min(len(indices), max(MIN_DONOR_CELLS_PER_STRATUM, int(round(len(indices) * fraction))))
        keep[indices] = False
        chosen = rng.choice(indices, size=n_keep, replace=False)
        keep[chosen] = True
    return keep


def precompute_masks(
    cohort: dict[str, Any],
    conditions: list[tuple[float, int]],
) -> tuple[dict[tuple[float, int], np.ndarray], pd.DataFrame]:
    mrd_obs = cohort["mrd_obs"]
    cell_labels = cohort["cell_labels"]
    cell_patients = cohort["cell_patients"]
    donor_healthy = cohort["donor_healthy"]
    recipient_healthy = cohort["recipient_healthy"]
    condition_masks: dict[tuple[float, int], np.ndarray] = {}
    mask_count_records: list[dict[str, Any]] = []
    for fraction, downsample_seed in conditions:
        keep_mask = donor_keep_mask(mrd_obs, donor_healthy, fraction, downsample_seed)
        condition_masks[(fraction, downsample_seed)] = keep_mask
        for patient_id in sorted(mrd_obs["patient"].unique()):
            patient_mask = cell_patients == patient_id
            mask_count_records.append(
                {
                    "donor_fraction": fraction,
                    "downsample_seed": downsample_seed,
                    "patient_id": patient_id,
                    "donor_healthy_before": int((patient_mask & donor_healthy).sum()),
                    "donor_healthy_retained": int((patient_mask & donor_healthy & keep_mask).sum()),
                    "recipient_healthy_retained": int((patient_mask & recipient_healthy & keep_mask).sum()),
                    "malignant_retained": int((patient_mask & (cell_labels == 1) & keep_mask).sum()),
                }
            )
    return condition_masks, pd.DataFrame(mask_count_records).drop_duplicates()


def paths_for_experiment(experiment_dir: Path) -> dict[str, Path]:
    diagnostic = experiment_dir / "analysis" / "scorecards" / "stage2_diagnostic"
    return {
        "experiment_dir": experiment_dir,
        "diagnostic": diagnostic,
        "representation_table": diagnostic / "old34_representation_diagnostic_table.csv",
        "patient_metrics": diagnostic / "old34_supported_patient_representation_metrics.csv",
        "direct_selected_patient_metrics": diagnostic
        / "meeting_followup_v1"
        / "old34_direct_gene_selected_patient_metrics.csv",
        "direct_vs_best_dr": diagnostic / "meeting_followup_v1" / "old34_direct_gene_vs_best_dr_summary.csv",
        "v1_table_dir": diagnostic / "donor_sensitivity_v1",
        "table_dir": diagnostic / OUTPUT_BRANCH,
        "cache_dir": experiment_dir / "analysis" / "cache" / "stage2_diagnostic" / OUTPUT_BRANCH,
        "v1_fixed_metrics": diagnostic / "donor_sensitivity_v1" / "fixed_basis_donor_downsampling_patient_metrics.csv",
        "v1_basis_metrics": diagnostic / "donor_sensitivity_v1" / "ifna_basis_refit_patient_metrics.csv",
    }


def load_target_features(row: pd.Series) -> tuple[np.ndarray, list[str]]:
    path = Path(row["features_path"])
    if path.suffix == ".npz":
        matrix = load_npz(path).toarray().astype(np.float32)
        names = json.loads(Path(row["feature_names_path"]).read_text())
    else:
        matrix = np.load(path).astype(np.float32)
        names = [f"factor_{index:03d}" for index in range(1, matrix.shape[1] + 1)]
    expected = (N_CELLS, int(row["n_features"]))
    if matrix.shape != expected:
        raise AssertionError(f"{path} has shape {matrix.shape}, expected {expected}")
    if len(names) != matrix.shape[1]:
        raise AssertionError(f"Feature-name count {len(names)} != {matrix.shape[1]} for {path}")
    return matrix, [str(name) for name in names]


def dr_feature_paths(experiment_dir: Path, panel_id: str, method: str, k: int) -> tuple[str, str]:
    scores = (
        experiment_dir
        / "stage1_dr"
        / f"panel_id={panel_id}"
        / f"method={method}"
        / f"k={k}"
        / "seed=42"
        / "scores.npy"
    )
    return str(scores), ""


def direct_feature_paths(experiment_dir: Path, panel_id: str) -> tuple[str, str]:
    root = experiment_dir / "stage1_direct_gene" / DIRECT_BRANCH / f"panel_id={panel_id}" / "transform=zscore"
    return str(root / "features.npz"), str(root / "feature_names.json")


def representation_id_for(panel_id: str, family: str, method: str, k: int) -> str:
    return f"{panel_id}|{family}|{method}|{k}|42"


def apply_v1_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    alias_id = {row["representation_id"]: row["target_id"] for row in V1_FIXED_ALIASES}
    alias_label = {row["representation_id"]: row["target_label"] for row in V1_FIXED_ALIASES}
    out = frame.copy()
    out["target_id"] = [alias_id.get(rid, rid) for rid in out["representation_id"]]
    aliased_labels = [alias_label.get(rid) for rid in out["representation_id"]]
    out["target_label"] = [
        aliased if aliased else label for aliased, label in zip(aliased_labels, out["target_label"])
    ]
    return out


def build_dr_row(
    experiment_dir: Path,
    panel_id: str,
    method: str,
    k: int,
    penalty: str,
    c_value: float,
    l1_ratio: Any,
    short_panel_label: str,
    representation_id: str | None = None,
) -> dict[str, Any]:
    features_path, feature_names_path = dr_feature_paths(experiment_dir, panel_id, method, int(k))
    rid = representation_id or representation_id_for(panel_id, "dr", method, int(k))
    return {
        "target_id": rid,
        "target_label": f"{short_panel_label} | {method_display(method, int(k))}",
        "stage0_panel_id": panel_id,
        "short_panel_label": short_panel_label,
        "representation_id": rid,
        "representation_family": "dr",
        "stage1_method": method,
        "effective_k": int(k),
        "n_features": int(k),
        "penalty": penalty,
        "C": float(c_value),
        "l1_ratio": l1_ratio,
        "features_path": features_path,
        "feature_names_path": feature_names_path,
        "d11_inclusion_reason": "",
    }


def build_direct_row(
    experiment_dir: Path,
    panel_id: str,
    n_features: int,
    penalty: str,
    c_value: float,
    l1_ratio: Any,
    short_panel_label: str,
    representation_id: str | None = None,
) -> dict[str, Any]:
    features_path, feature_names_path = direct_feature_paths(experiment_dir, panel_id)
    rid = representation_id or representation_id_for(panel_id, "direct_gene", "direct_gene", int(n_features))
    return {
        "target_id": rid,
        "target_label": f"{short_panel_label} | direct genes",
        "stage0_panel_id": panel_id,
        "short_panel_label": short_panel_label,
        "representation_id": rid,
        "representation_family": "direct_gene",
        "stage1_method": "direct_gene",
        "effective_k": int(n_features),
        "n_features": int(n_features),
        "penalty": penalty,
        "C": float(c_value),
        "l1_ratio": l1_ratio,
        "features_path": features_path,
        "feature_names_path": feature_names_path,
        "d11_inclusion_reason": "",
    }


def load_source_tables(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    required = ["representation_table", "direct_selected_patient_metrics", "direct_vs_best_dr"]
    for name in required:
        if not paths[name].exists():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    representation = pd.read_csv(paths["representation_table"], low_memory=False)
    direct_selected = pd.read_csv(paths["direct_selected_patient_metrics"], low_memory=False)
    direct_vs_best_dr = pd.read_csv(paths["direct_vs_best_dr"], low_memory=False)
    return {
        "representation": representation,
        "direct_selected": direct_selected,
        "direct_vs_best_dr": direct_vs_best_dr,
    }


def unique_direct_settings(direct_selected: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        direct_selected.groupby("stage0_panel_id", as_index=False)
        .agg(
            penalty=("penalty", "first"),
            C=("C", "first"),
            l1_ratio=("l1_ratio", "first"),
            n_covered_genes=("n_covered_genes", "first"),
            representation_id=("representation_id", "first"),
            n_penalty=("penalty", "nunique"),
            n_c=("C", "nunique"),
        )
    )
    if not grouped["n_penalty"].eq(1).all() or not grouped["n_c"].eq(1).all():
        raise AssertionError("Direct-gene panels do not have a unique global regularization setting")
    return grouped


def build_fixed_basis_manifest(
    experiment_dir: Path,
    sources: dict[str, pd.DataFrame],
    target_set: str,
) -> pd.DataFrame:
    representation = sources["representation"]
    direct_settings = unique_direct_settings(sources["direct_selected"])
    short_lookup = (
        representation.drop_duplicates("stage0_panel_id").set_index("stage0_panel_id")["short_panel_label"].to_dict()
    )
    extra_short = sources["direct_vs_best_dr"].set_index("stage0_panel_id")["short_panel_label"].to_dict()
    short_lookup.update(extra_short)

    dr_rows = representation.loc[representation["stage1_method"].ne("direct_gene")].copy()
    records: list[dict[str, Any]] = []
    for _, row in dr_rows.iterrows():
        records.append(
            build_dr_row(
                experiment_dir,
                str(row["stage0_panel_id"]),
                str(row["stage1_method"]),
                int(row["effective_k"]),
                str(row["selected_penalty"]),
                float(row["selected_C_or_alpha"]),
                row["selected_l1_ratio"],
                str(row.get("short_panel_label", short_lookup.get(row["stage0_panel_id"], row["stage0_panel_id"]))),
                str(row["representation_id"]),
            )
        )
    for _, row in direct_settings.iterrows():
        panel_id = str(row["stage0_panel_id"])
        records.append(
            build_direct_row(
                experiment_dir,
                panel_id,
                int(row["n_covered_genes"]),
                str(row["penalty"]),
                float(row["C"]),
                row["l1_ratio"],
                str(short_lookup.get(panel_id, panel_id)),
                str(row["representation_id"]),
            )
        )
    manifest = apply_v1_aliases(pd.DataFrame(records).drop_duplicates("representation_id", keep="first"))
    alias_order = {row["target_id"]: index for index, row in enumerate(V1_FIXED_ALIASES)}
    manifest["v1_order"] = manifest["target_id"].map(alias_order)
    manifest["family_order"] = np.where(manifest["representation_family"].eq("dr"), 0, 1)
    manifest = manifest.sort_values(
        ["v1_order", "family_order", "stage0_panel_id", "stage1_method", "effective_k"],
        na_position="last",
    ).drop(columns=["v1_order", "family_order"])
    manifest = manifest.reset_index(drop=True)

    if target_set == "v1_overlap":
        keep = [row["target_id"] for row in V1_FIXED_ALIASES]
        manifest = manifest.loc[manifest["target_id"].isin(keep)].copy()
        manifest["target_id"] = pd.Categorical(manifest["target_id"], categories=keep, ordered=True)
        manifest = manifest.sort_values("target_id").reset_index(drop=True)
        manifest["target_id"] = manifest["target_id"].astype(str)
    elif target_set == "smoke":
        keep = ["ifna_pca40", "ifna_direct"]
        manifest = manifest.loc[manifest["target_id"].isin(keep)].copy()
        manifest["target_id"] = pd.Categorical(manifest["target_id"], categories=keep, ordered=True)
        manifest = manifest.sort_values("target_id").reset_index(drop=True)
        manifest["target_id"] = manifest["target_id"].astype(str)
    elif target_set != "all_dr_plus_direct":
        raise ValueError(f"Unknown target set: {target_set}")
    return manifest


def build_basis_refit_manifest(
    experiment_dir: Path,
    sources: dict[str, pd.DataFrame],
    target_set: str,
) -> pd.DataFrame:
    representation = sources["representation"]
    best = sources["direct_vs_best_dr"].copy()
    short_lookup = best.set_index("stage0_panel_id")["short_panel_label"].to_dict()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_from_rep(panel_id: str, method: str, k: int, reason: str) -> None:
        match = representation.loc[
            representation["stage0_panel_id"].eq(panel_id)
            & representation["stage1_method"].eq(method)
            & representation["effective_k"].eq(int(k))
        ]
        if match.empty:
            raise KeyError(f"Missing representation {panel_id}|dr|{method}|{k}|42")
        row = match.iloc[0]
        record = build_dr_row(
            experiment_dir,
            panel_id,
            method,
            int(k),
            str(row["selected_penalty"]),
            float(row["selected_C_or_alpha"]),
            row["selected_l1_ratio"],
            str(row.get("short_panel_label", short_lookup.get(panel_id, panel_id))),
            str(row["representation_id"]),
        )
        record["d11_inclusion_reason"] = reason
        if record["representation_id"] not in seen:
            records.append(record)
            seen.add(str(record["representation_id"]))

    pca_fa_best = best.loc[best["best_dr_method"].isin(["pca", "fa"])]
    for _, row in pca_fa_best.iterrows():
        add_from_rep(str(row["stage0_panel_id"]), str(row["best_dr_method"]), int(row["best_dr_k"]), "best_dr_pca_fa")
    add_from_rep(IFNA_PANEL, "pca", 40, "ifna_case_study")
    add_from_rep(IFNA_PANEL, "fa", 40, "ifna_case_study")

    manifest = apply_v1_aliases(pd.DataFrame(records))
    if target_set == "v1_overlap":
        manifest = manifest.loc[manifest["target_id"].isin(V1_BASIS_TARGET_IDS)].copy()
    elif target_set == "smoke":
        manifest = manifest.loc[manifest["target_id"].eq("ifna_pca40")].copy()
    elif target_set != "all_dr_plus_direct":
        raise ValueError(f"Unknown target set: {target_set}")
    preferred = V1_BASIS_TARGET_IDS
    manifest["v1_order"] = manifest["target_id"].map({tid: index for index, tid in enumerate(preferred)})
    manifest = manifest.sort_values(["v1_order", "stage0_panel_id", "stage1_method", "effective_k"], na_position="last")
    return manifest.drop(columns=["v1_order"]).reset_index(drop=True)


def assert_feature_paths(manifest: pd.DataFrame) -> None:
    missing = [path for path in manifest["features_path"] if not Path(path).exists()]
    extra_missing = [
        path
        for path in manifest["feature_names_path"]
        if path and not Path(path).exists()
    ]
    if missing or extra_missing:
        preview = (missing + extra_missing)[:8]
        raise FileNotFoundError(f"Missing {len(missing) + len(extra_missing)} feature files, e.g. {preview}")


def shard_frame(frame: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    if shard_count <= 1:
        return frame.reset_index(drop=True)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    keep = [idx for idx in range(len(frame)) if idx % shard_count == shard_index]
    return frame.iloc[keep].reset_index(drop=True)


def attach_reference_metrics(results: pd.DataFrame, ap_col: str = "heldout_auprc") -> pd.DataFrame:
    out = results.copy()
    reference = out.loc[out["donor_fraction"].eq(1.0)].set_index(["target_id", "heldout_patient"])
    out["reference_heldout_auprc"] = [
        reference.loc[(target_id, patient_id), ap_col]
        for target_id, patient_id in zip(out["target_id"], out["heldout_patient"])
    ]
    if "heldout_lift" in out.columns:
        out["reference_heldout_lift"] = [
            reference.loc[(target_id, patient_id), "heldout_lift"]
            for target_id, patient_id in zip(out["target_id"], out["heldout_patient"])
        ]
        out["delta_heldout_lift_vs_full_donor"] = out["heldout_lift"] - out["reference_heldout_lift"]
    out["delta_heldout_auprc_vs_full_donor"] = out[ap_col] - out["reference_heldout_auprc"]
    return out


def collapse_summary(results: pd.DataFrame) -> pd.DataFrame:
    work = results.copy()
    work["zero_coefficient_solution"] = work["active_coefficient_count"].eq(0)
    summary = (
        work.groupby(["target_id", "target_label", "donor_fraction"], as_index=False)
        .agg(
            n_fits=("heldout_patient", "size"),
            n_zero_fits=("zero_coefficient_solution", "sum"),
            zero_solution_fraction=("zero_coefficient_solution", "mean"),
            median_active_coefficients=("active_coefficient_count", "median"),
            mean_heldout_lift=("heldout_lift", "mean"),
            n_features=("n_features", "first"),
        )
    )
    return summary.sort_values(["target_id", "donor_fraction"]).reset_index(drop=True)


def evaluate_lopo_condition(
    features: np.ndarray,
    feature_names: list[str],
    setting: dict[str, Any],
    model_args: Any,
    cohort: dict[str, Any],
    keep_mask: np.ndarray,
    target: pd.Series,
    fraction: float,
    downsample_seed: int,
    extra: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cell_labels = cohort["cell_labels"]
    cell_patients = cohort["cell_patients"]
    donor_healthy = cohort["donor_healthy"]
    recipient_healthy = cohort["recipient_healthy"]
    records: list[dict[str, Any]] = []
    coefficients: dict[str, np.ndarray] = {}
    for heldout_patient in SUPPORTED_PATIENTS:
        test_mask = cell_patients == heldout_patient
        train_mask = (cell_patients != heldout_patient) & keep_mask
        x_train, x_test = fit_preprocess_transform(features[train_mask], features[test_mask])
        clf = fit_model(x_train, cell_labels[train_mask], setting, model_args)
        probability = np.asarray(clf.predict_proba(x_test))[:, 1]
        coefficient = np.asarray(clf.coef_).ravel().astype(float)
        y_test = cell_labels[test_mask]
        prevalence = float(y_test.mean())
        ap_value = float(average_precision_score(y_test, probability))
        record = {
            "target_id": target["target_id"],
            "target_label": target["target_label"],
            "stage0_panel_id": target["stage0_panel_id"],
            "short_panel_label": target.get("short_panel_label", ""),
            "representation_id": target["representation_id"],
            "representation_family": target["representation_family"],
            "stage1_method": target["stage1_method"],
            "effective_k": int(target["effective_k"]),
            "n_features": int(target["n_features"]),
            "penalty": target["penalty"],
            "C": float(target["C"]),
            "l1_ratio": target["l1_ratio"],
            "donor_fraction": fraction,
            "downsample_seed": downsample_seed,
            "heldout_patient": heldout_patient,
            "n_train_cells": int(train_mask.sum()),
            "n_train_donor_healthy": int((train_mask & donor_healthy).sum()),
            "n_train_recipient_healthy": int((train_mask & recipient_healthy).sum()),
            "n_train_malignant": int((train_mask & (cell_labels == 1)).sum()),
            "n_test_cells_full": int(test_mask.sum()),
            "patient_prevalence": prevalence,
            "heldout_auprc": ap_value,
            "heldout_auroc": float(roc_auc_score(y_test, probability)),
            "heldout_lift": ap_value - prevalence,
            "active_coefficient_count": int((np.abs(coefficient) > COEF_TOLERANCE).sum()),
            "coefficient_l1_norm": float(np.abs(coefficient).sum()),
        }
        if extra:
            record.update(extra)
        records.append(record)
        coefficients[heldout_patient] = coefficient
        del clf, x_train, x_test
    return records, coefficients


def run_fixed_basis_target(
    target: pd.Series,
    cohort: dict[str, Any],
    condition_masks: dict[tuple[float, int], np.ndarray],
    conditions: list[tuple[float, int]],
    model_args: Any,
) -> dict[str, pd.DataFrame]:
    features, feature_names = load_target_features(target)
    setting = setting_from_row(target)
    metric_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    supervised_records: list[dict[str, Any]] = []
    reference_coefs: dict[str, np.ndarray] | None = None
    try:
        for fraction, downsample_seed in conditions:
            keep_mask = condition_masks[(fraction, downsample_seed)]
            records, coefficients = evaluate_lopo_condition(
                features,
                feature_names,
                setting,
                model_args,
                cohort,
                keep_mask,
                target,
                fraction,
                downsample_seed,
            )
            metric_records.extend(records)
            if reference_coefs is None:
                if fraction != 1.0:
                    raise AssertionError("The first fixed-basis condition must be fraction 1")
                reference_coefs = coefficients
                for heldout_patient, coefficient in coefficients.items():
                    for feature_name, value in zip(feature_names, coefficient):
                        if abs(value) > COEF_TOLERANCE:
                            supervised_records.append(
                                {
                                    "target_id": target["target_id"],
                                    "representation_id": target["representation_id"],
                                    "heldout_patient": heldout_patient,
                                    "feature_id": feature_name,
                                    "coefficient": float(value),
                                }
                            )
            for heldout_patient, coefficient in coefficients.items():
                reference = reference_coefs[heldout_patient]
                stability_records.append(
                    {
                        "target_id": target["target_id"],
                        "representation_id": target["representation_id"],
                        "donor_fraction": fraction,
                        "downsample_seed": downsample_seed,
                        "heldout_patient": heldout_patient,
                        "coefficient_cosine_vs_full_donor": cosine_similarity(coefficient, reference),
                        "active_feature_jaccard_vs_full_donor": jaccard_active(coefficient, reference),
                        "active_feature_count": int((np.abs(coefficient) > COEF_TOLERANCE).sum()),
                        "reference_active_feature_count": int((np.abs(reference) > COEF_TOLERANCE).sum()),
                        "target_label": target["target_label"],
                        "representation_family": target["representation_family"],
                        "stage1_method": target["stage1_method"],
                        "n_features": int(target["n_features"]),
                    }
                )
    finally:
        del features
    metrics = attach_reference_metrics(pd.DataFrame(metric_records))
    return {
        "metrics": metrics,
        "stability": pd.DataFrame(stability_records),
        "supervised": pd.DataFrame(supervised_records, columns=SUPERVISED_COLUMNS),
    }


def fit_basis_model(method_name: str, n_components: int, retained_gene_matrix: np.ndarray):
    if method_name == "pca":
        return PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    if method_name == "fa":
        return FactorAnalysis(n_components=n_components, svd_method="randomized", random_state=42)
    raise ValueError(f"D11 basis refit is implemented only for pca/fa, not {method_name}")


def run_basis_refit_target(
    target: pd.Series,
    gene_matrix: np.ndarray,
    gene_names: list[str],
    cohort: dict[str, Any],
    condition_masks: dict[tuple[float, int], np.ndarray],
    conditions: list[tuple[float, int]],
    model_args: Any,
) -> dict[str, pd.DataFrame]:
    method_name = str(target["stage1_method"])
    n_components = int(target["effective_k"])
    setting = setting_from_row(target)
    metric_records: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    supervised_by_factor: dict[str, float] = {}
    reference_scores: np.ndarray | None = None
    reference_loadings_common: np.ndarray | None = None
    for fraction, downsample_seed in conditions:
        keep_mask = condition_masks[(fraction, downsample_seed)]
        panel_scaler = StandardScaler(with_mean=True, with_std=True)
        retained_gene_matrix = panel_scaler.fit_transform(gene_matrix[keep_mask])
        all_gene_matrix = panel_scaler.transform(gene_matrix)
        basis_model = fit_basis_model(method_name, n_components, retained_gene_matrix)
        retained_scores = basis_model.fit_transform(retained_gene_matrix)
        all_scores = basis_model.transform(all_gene_matrix).astype(np.float32)
        all_scores[keep_mask] = np.asarray(retained_scores, dtype=np.float32)
        loadings_standardized = np.asarray(basis_model.components_).T
        loadings_common = loadings_standardized / np.asarray(panel_scaler.scale_)[:, None]
        if reference_scores is None:
            if fraction != 1.0:
                raise AssertionError("The first basis condition must be fraction 1")
            reference_scores = all_scores.copy()
            reference_loadings_common = loadings_common.copy()
        correlations = score_correlation_matrix(reference_scores, all_scores)
        reference_indices, current_indices = linear_sum_assignment(-np.abs(correlations))
        for reference_factor, current_factor in zip(reference_indices, current_indices):
            signed_score_correlation = float(correlations[reference_factor, current_factor])
            orientation = 1.0 if signed_score_correlation >= 0 else -1.0
            oriented_loading = orientation * loadings_common[:, current_factor]
            reference_loading = reference_loadings_common[:, reference_factor]
            reference_name = f"factor_{reference_factor + 1:03d}"
            stability_records.append(
                {
                    "target_id": target["target_id"],
                    "target_label": target["target_label"],
                    "stage0_panel_id": target["stage0_panel_id"],
                    "representation_id": target["representation_id"],
                    "stage1_method": method_name,
                    "donor_fraction": fraction,
                    "downsample_seed": downsample_seed,
                    "reference_factor": reference_name,
                    "current_factor": f"factor_{current_factor + 1:03d}",
                    "absolute_score_correlation": abs(signed_score_correlation),
                    "loading_cosine_common_gene_scale": cosine_similarity(reference_loading, oriented_loading),
                    "top10_gene_jaccard": top_gene_jaccard(reference_loading, oriented_loading),
                    "reference_top10_genes": ";".join(
                        np.asarray(gene_names)[np.argsort(np.abs(reference_loading))[-10:][::-1]]
                    ),
                    "current_top10_genes": ";".join(
                        np.asarray(gene_names)[np.argsort(np.abs(oriented_loading))[-10:][::-1]]
                    ),
                    "d11_inclusion_reason": target.get("d11_inclusion_reason", ""),
                }
            )
        records, coefficients = evaluate_lopo_condition(
            all_scores,
            [f"factor_{index:03d}" for index in range(1, n_components + 1)],
            setting,
            model_args,
            cohort,
            keep_mask,
            target,
            fraction,
            downsample_seed,
            extra={"n_stage1_retained_cells": int(keep_mask.sum())},
        )
        metric_records.extend(records)
        if fraction == 1.0:
            stacked = np.vstack([coefficients[patient] for patient in SUPPORTED_PATIENTS])
            mean_abs = np.mean(np.abs(stacked), axis=0)
            for index, value in enumerate(mean_abs):
                supervised_by_factor[f"factor_{index + 1:03d}"] = float(value)
        del basis_model, retained_gene_matrix, all_gene_matrix, all_scores
    metrics = attach_reference_metrics(pd.DataFrame(metric_records))
    metrics = metrics.rename(columns={"delta_heldout_auprc_vs_full_donor": "delta_heldout_auprc_vs_full_donor_basis"})
    factor_stability = pd.DataFrame(stability_records)
    factor_stability["mean_absolute_full_donor_coefficient"] = factor_stability["reference_factor"].map(
        supervised_by_factor
    )
    factor_stability["is_supervised_factor"] = factor_stability["mean_absolute_full_donor_coefficient"].gt(
        COEF_TOLERANCE
    )
    driver = (
        pd.Series(supervised_by_factor, name="mean_absolute_coefficient")
        .sort_values(ascending=False)
    )
    driver = driver.loc[driver.gt(COEF_TOLERANCE)]
    driver_factor = str(driver.index[0]) if len(driver) else ""
    factor_stability["is_driver_factor"] = factor_stability["reference_factor"].eq(driver_factor)
    return {"metrics": metrics, "factor_stability": factor_stability}


def checkpoint_paths(cache_dir: Path, arm: str, target_id: str) -> dict[str, Path]:
    stem = safe_checkpoint_stem(str(target_id))
    root = cache_dir / "checkpoints" / arm
    paths = {
        "complete": root / f"{stem}.complete.json",
        "metrics": root / f"{stem}.metrics.csv",
    }
    if arm == "fixed_basis":
        paths["stability"] = root / f"{stem}.stability.csv"
        paths["supervised"] = root / f"{stem}.supervised.csv"
    else:
        paths["factor_stability"] = root / f"{stem}.factor_stability.csv"
    return paths


def checkpoint_is_complete(paths: dict[str, Path], signature: dict[str, Any]) -> bool:
    if not paths["complete"].exists() or not paths["metrics"].exists():
        return False
    try:
        payload = json.loads(paths["complete"].read_text())
    except json.JSONDecodeError:
        return False
    if payload.get("status") != "complete" or payload.get("signature") != signature:
        return False
    try:
        metrics = pd.read_csv(paths["metrics"])
    except pd.errors.EmptyDataError:
        return False
    if metrics.empty:
        return False
    for key in ("stability", "supervised", "factor_stability"):
        csv_path = paths.get(key)
        if csv_path is None:
            continue
        if not csv_path.exists():
            return False
        frame = read_checkpoint_csv(csv_path, key)
        if key != "supervised" and csv_path.stat().st_size > 1 and frame.empty:
            return False
    return True


def save_target_checkpoint(
    arm: str,
    target_id: str,
    outputs: dict[str, pd.DataFrame],
    cache_dir: Path,
    signature: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Path]:
    paths = checkpoint_paths(cache_dir, arm, target_id)
    write_csv_atomic(outputs["metrics"], paths["metrics"])
    if arm == "fixed_basis":
        write_csv_atomic(outputs["stability"], paths["stability"])
        write_csv_atomic(outputs["supervised"], paths["supervised"], columns=SUPERVISED_COLUMNS)
    else:
        write_csv_atomic(outputs["factor_stability"], paths["factor_stability"])
    write_json(
        paths["complete"],
        {
            "status": "complete",
            "target_id": target_id,
            "arm": arm,
            "completed_utc": utc_now(),
            "elapsed_seconds": elapsed_seconds,
            "signature": signature,
            "rows": {name: int(len(frame)) for name, frame in outputs.items()},
        },
    )
    return paths


def concat_checkpoints(cache_dir: Path, arm: str, key: str) -> pd.DataFrame:
    root = cache_dir / "checkpoints" / arm
    if not root.exists():
        return pd.DataFrame()
    frames = []
    n_empty = 0
    n_bad_complete = 0
    for complete_path in sorted(root.glob("*.complete.json")):
        try:
            payload = json.loads(complete_path.read_text())
        except json.JSONDecodeError:
            n_bad_complete += 1
            LOGGER.warning("Skipping corrupt complete marker: %s", complete_path)
            continue
        if payload.get("status") != "complete":
            continue
        csv_path = complete_path.with_name(complete_path.name.replace(".complete.json", f".{key}.csv"))
        frame = read_checkpoint_csv(csv_path, key)
        if frame.empty:
            n_empty += 1
            continue
        frames.append(frame)
    if n_empty:
        LOGGER.info("Skipped %d empty %s/%s checkpoint CSVs during merge", n_empty, arm, key)
    if n_bad_complete:
        LOGGER.warning("Skipped %d corrupt %s complete markers during merge", n_bad_complete, arm)
    if not frames:
        return pd.DataFrame(columns=CHECKPOINT_CSV_SCHEMAS.get(key, []))
    return pd.concat(frames, ignore_index=True)


def finalize_merge(
    paths: dict[str, Path],
    cache_dir: Path,
    table_dir: Path,
    state: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str]]:
    state_path = table_dir / "RUN_STATE.json"
    existing: dict[str, Any] = dict(state or {})
    if not existing and state_path.exists():
        try:
            existing = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    try:
        written = merge_outputs(paths, cache_dir, table_dir)
    except Exception as error:  # noqa: BLE001 - preserve compute; merge must be retryable
        LOGGER.exception(
            "Checkpoints are on disk but merge failed (%s). Re-run with --merge-only; "
            "do not recompute finished targets.",
            error,
        )
        existing["status"] = "merge_failed"
        existing["merge_error"] = str(error)
        write_run_state(state_path, existing)
        return 3, {}
    existing["status"] = "complete"
    existing["written"] = written
    write_run_state(state_path, existing)
    LOGGER.info("Merged outputs: %s", written)
    return 0, written


def append_failure(table_dir: Path, arm: str, target_id: str, error: BaseException) -> None:
    path = table_dir / "failures.csv"
    record = pd.DataFrame(
        [
            {
                "arm": arm,
                "target_id": target_id,
                "failed_utc": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        ]
    )
    if path.exists() and path.stat().st_size > 1:
        try:
            record = pd.concat([pd.read_csv(path), record], ignore_index=True)
        except pd.errors.EmptyDataError:
            pass
    write_csv_atomic(record, path)


def write_run_state(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_utc"] = utc_now()
    write_json(path, payload)


def compare_to_v1(current: pd.DataFrame, reference: pd.DataFrame, label: str) -> dict[str, Any]:
    keys = ["target_id", "donor_fraction", "downsample_seed", "heldout_patient"]
    merged = current.merge(
        reference[keys + ["heldout_auprc", "active_coefficient_count"]].rename(
            columns={
                "heldout_auprc": "v1_heldout_auprc",
                "active_coefficient_count": "v1_active_coefficient_count",
            }
        ),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"No overlapping {label} rows for v1 QA")
    merged["abs_ap_error"] = (merged["heldout_auprc"] - merged["v1_heldout_auprc"]).abs()
    merged["collapse_mismatch"] = merged["active_coefficient_count"].eq(0) != merged["v1_active_coefficient_count"].eq(
        0
    )
    report = {
        "label": label,
        "n_compared_rows": int(len(merged)),
        "max_abs_ap_error": float(merged["abs_ap_error"].max()),
        "n_collapse_mismatches": int(merged["collapse_mismatch"].sum()),
        "passed": bool(
            float(merged["abs_ap_error"].max()) <= QA_MAX_ABS_AP_ERROR and int(merged["collapse_mismatch"].sum()) == 0
        ),
    }
    return report


def merge_outputs(paths: dict[str, Path], cache_dir: Path, table_dir: Path) -> dict[str, str]:
    written: dict[str, str] = {}
    fixed_metrics = concat_checkpoints(cache_dir, "fixed_basis", "metrics")
    if not fixed_metrics.empty:
        write_csv_atomic(fixed_metrics, table_dir / "fixed_basis_patient_metrics.csv")
        write_csv_atomic(collapse_summary(fixed_metrics), table_dir / "fixed_basis_collapse_summary.csv")
        written["fixed_basis_patient_metrics"] = str(table_dir / "fixed_basis_patient_metrics.csv")
        written["fixed_basis_collapse_summary"] = str(table_dir / "fixed_basis_collapse_summary.csv")
    fixed_stability = concat_checkpoints(cache_dir, "fixed_basis", "stability")
    if not fixed_stability.empty:
        write_csv_atomic(fixed_stability, table_dir / "fixed_basis_coefficient_stability.csv")
        written["fixed_basis_coefficient_stability"] = str(table_dir / "fixed_basis_coefficient_stability.csv")
    supervised = concat_checkpoints(cache_dir, "fixed_basis", "supervised")
    if not supervised.empty:
        write_csv_atomic(supervised, table_dir / "fixed_basis_full_donor_supervised_features.csv")
        written["fixed_basis_full_donor_supervised_features"] = str(
            table_dir / "fixed_basis_full_donor_supervised_features.csv"
        )
    basis_metrics = concat_checkpoints(cache_dir, "basis_refit", "metrics")
    if not basis_metrics.empty:
        write_csv_atomic(basis_metrics, table_dir / "basis_refit_patient_metrics.csv")
        written["basis_refit_patient_metrics"] = str(table_dir / "basis_refit_patient_metrics.csv")
        performance = (
            basis_metrics.groupby(["target_id", "target_label", "stage1_method", "donor_fraction"], as_index=False)
            .agg(
                mean_delta_ap=("delta_heldout_auprc_vs_full_donor_basis", "mean"),
                median_delta_ap=("delta_heldout_auprc_vs_full_donor_basis", "median"),
                worst_delta_ap=("delta_heldout_auprc_vs_full_donor_basis", "min"),
            )
        )
        write_csv_atomic(performance, table_dir / "basis_refit_performance_summary.csv")
        written["basis_refit_performance_summary"] = str(table_dir / "basis_refit_performance_summary.csv")
    factor_stability = concat_checkpoints(cache_dir, "basis_refit", "factor_stability")
    if not factor_stability.empty:
        write_csv_atomic(factor_stability, table_dir / "basis_refit_factor_stability.csv")
        written["basis_refit_factor_stability"] = str(table_dir / "basis_refit_factor_stability.csv")
    return written


def run_arm(
    arm: str,
    manifest: pd.DataFrame,
    cohort: dict[str, Any],
    condition_masks: dict[tuple[float, int], np.ndarray],
    conditions: list[tuple[float, int]],
    model_args: Any,
    paths: dict[str, Path],
    resume: bool,
    continue_on_error: bool,
    gene_cache: dict[str, tuple[np.ndarray, list[str]]],
    state: dict[str, Any],
) -> None:
    signature = {
        "version": CHECKPOINT_VERSION,
        "arm": arm,
        "conditions": [list(condition) for condition in conditions],
        "supported_patients": SUPPORTED_PATIENTS,
        "min_donor_cells_per_stratum": MIN_DONOR_CELLS_PER_STRATUM,
    }
    cache_dir = paths["cache_dir"]
    table_dir = paths["table_dir"]
    completed_seconds: list[float] = []
    n_failed = 0
    for index, target in manifest.iterrows():
        target_id = str(target["target_id"])
        target_signature = {
            **signature,
            "target_id": target_id,
            "representation_id": str(target["representation_id"]),
            "penalty": str(target["penalty"]),
            "C": float(target["C"]),
            "l1_ratio": None if pd.isna(target["l1_ratio"]) else float(target["l1_ratio"]),
        }
        ckpt = checkpoint_paths(cache_dir, arm, target_id)
        if resume and checkpoint_is_complete(ckpt, target_signature):
            LOGGER.info("[%s] skip completed %s (%d/%d)", arm, target_id, index + 1, len(manifest))
            continue
        LOGGER.info("[%s] %d/%d %s", arm, index + 1, len(manifest), target["target_label"])
        started = time.time()
        try:
            if arm == "fixed_basis":
                outputs = run_fixed_basis_target(target, cohort, condition_masks, conditions, model_args)
            else:
                panel_id = str(target["stage0_panel_id"])
                if panel_id not in gene_cache:
                    gene_row = target.copy()
                    gene_path, gene_names_path = direct_feature_paths(paths["experiment_dir"], panel_id)
                    gene_row["features_path"] = gene_path
                    gene_row["feature_names_path"] = gene_names_path
                    gene_row["n_features"] = len(json.loads(Path(gene_names_path).read_text()))
                    gene_cache[panel_id] = load_target_features(gene_row)
                gene_matrix, gene_names = gene_cache[panel_id]
                outputs = run_basis_refit_target(
                    target,
                    gene_matrix,
                    gene_names,
                    cohort,
                    condition_masks,
                    conditions,
                    model_args,
                )
            elapsed = time.time() - started
            save_target_checkpoint(arm, target_id, outputs, cache_dir, target_signature, elapsed)
            completed_seconds.append(elapsed)
            mean_seconds = float(np.mean(completed_seconds))
            remaining = int(manifest.shape[0] - (index + 1))
            state[arm] = {
                "n_total": int(len(manifest)),
                "n_completed": int(index + 1 - n_failed),
                "n_failed": n_failed,
                "last_target_id": target_id,
                "last_seconds": elapsed,
                "mean_seconds_per_target": mean_seconds,
                "eta_seconds": remaining * mean_seconds,
            }
            write_run_state(table_dir / "RUN_STATE.json", state)
            LOGGER.info("[%s] finished %s in %.1fs", arm, target_id, elapsed)
        except Exception as error:  # noqa: BLE001 - overnight job must log and optionally continue
            n_failed += 1
            LOGGER.exception("[%s] failed %s", arm, target_id)
            append_failure(table_dir, arm, target_id, error)
            state[arm] = {
                "n_total": int(len(manifest)),
                "n_failed": n_failed,
                "last_target_id": target_id,
                "last_error": str(error),
            }
            write_run_state(table_dir / "RUN_STATE.json", state)
            if not continue_on_error:
                raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Old34 donor-downsampling sensitivity expansion (v2)")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT_H5AD)
    parser.add_argument("--arm", choices=["fixed_basis", "basis_refit", "both"], default="both")
    parser.add_argument(
        "--target-set",
        choices=["all_dr_plus_direct", "v1_overlap", "smoke"],
        default="all_dr_plus_direct",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qa-against-v1", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    parser.add_argument("--backend", default="cuml")
    parser.add_argument("--strict-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    if args.smoke_test:
        args.target_set = "smoke"
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    repo_root = find_repo_root()
    experiment_dir = args.experiment_dir or (repo_root / "experiments" / EXPERIMENT_ID)
    paths = paths_for_experiment(experiment_dir)
    paths["table_dir"].mkdir(parents=True, exist_ok=True)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)

    smoke = args.target_set == "smoke"
    conditions = condition_grid(smoke=smoke)
    sources = load_source_tables(paths)
    fixed_manifest = build_fixed_basis_manifest(experiment_dir, sources, args.target_set)
    basis_manifest = build_basis_refit_manifest(experiment_dir, sources, args.target_set)
    if args.max_targets is not None:
        fixed_manifest = fixed_manifest.head(args.max_targets).copy()
        basis_manifest = basis_manifest.head(args.max_targets).copy()
    fixed_manifest = shard_frame(fixed_manifest, args.shard_index, args.shard_count)
    basis_manifest = shard_frame(basis_manifest, args.shard_index, args.shard_count)
    write_csv_atomic(fixed_manifest, paths["table_dir"] / "target_manifest.csv")
    write_csv_atomic(basis_manifest, paths["table_dir"] / "basis_refit_target_manifest.csv")
    if args.arm in {"fixed_basis", "both"}:
        assert_feature_paths(fixed_manifest)
    if args.arm in {"basis_refit", "both"}:
        assert_feature_paths(basis_manifest)
        gene_paths = []
        for panel_id in basis_manifest["stage0_panel_id"].unique():
            gene_path, gene_names_path = direct_feature_paths(experiment_dir, str(panel_id))
            gene_paths.extend([gene_path, gene_names_path])
        missing = [path for path in gene_paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing D11 gene files: {missing[:8]}")

    manifest_payload = {
        "analysis_id": "old34_donor_sensitivity_v2",
        "experiment_id": EXPERIMENT_ID,
        "generated_utc": utc_now(),
        "arm": args.arm,
        "target_set": args.target_set,
        "conditions": [list(condition) for condition in conditions],
        "supported_patients": SUPPORTED_PATIENTS,
        "n_fixed_basis_targets": int(len(fixed_manifest)),
        "n_basis_refit_targets": int(len(basis_manifest)),
        "fixed_basis_target_ids": fixed_manifest["target_id"].tolist(),
        "basis_refit_target_ids": basis_manifest["target_id"].tolist(),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "regularization": "held fixed at globally selected pooled-LOPO setting per representation",
        "heldout_evaluation": "full_unchanged_patient",
        "stage1_refit_scope": "transductive_all_retained_patients",
        "output_branch": OUTPUT_BRANCH,
    }
    write_json(paths["table_dir"] / "run_manifest.json", manifest_payload)
    LOGGER.info(
        "Prepared manifests: D10=%d targets, D11=%d targets, conditions=%s",
        len(fixed_manifest),
        len(basis_manifest),
        conditions,
    )
    if args.prepare_only:
        write_run_state(
            paths["table_dir"] / "RUN_STATE.json",
            {"status": "prepared", "run_manifest": str(paths["table_dir"] / "run_manifest.json")},
        )
        return 0
    if args.merge_only:
        merge_rc, _written = finalize_merge(paths, paths["cache_dir"], paths["table_dir"])
        return merge_rc

    cohort = load_cohort(args.input_h5ad)
    condition_masks, mask_counts = precompute_masks(cohort, conditions)
    write_csv_atomic(mask_counts, paths["table_dir"] / "fixed_basis_donor_downsampling_retained_counts.csv")
    model_args = SimpleNamespace(
        backend=args.backend,
        strict_gpu=bool(args.strict_gpu),
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        n_jobs=int(args.n_jobs),
        class_weight=args.class_weight,
    )
    state: dict[str, Any] = {
        "status": "running",
        "analysis_id": "old34_donor_sensitivity_v2",
        "started_utc": utc_now(),
        "arm": args.arm,
        "target_set": args.target_set,
        "conditions": [list(condition) for condition in conditions],
    }
    write_run_state(paths["table_dir"] / "RUN_STATE.json", state)
    gene_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    if args.arm in {"fixed_basis", "both"}:
        run_arm(
            "fixed_basis",
            fixed_manifest,
            cohort,
            condition_masks,
            conditions,
            model_args,
            paths,
            args.resume,
            args.continue_on_error or args.target_set == "all_dr_plus_direct",
            gene_cache,
            state,
        )
    if args.arm in {"basis_refit", "both"}:
        run_arm(
            "basis_refit",
            basis_manifest,
            cohort,
            condition_masks,
            conditions,
            model_args,
            paths,
            args.resume,
            args.continue_on_error or args.target_set == "all_dr_plus_direct",
            gene_cache,
            state,
        )
    merge_rc, written = finalize_merge(paths, paths["cache_dir"], paths["table_dir"], state)
    if merge_rc != 0:
        return merge_rc
    qa_reports: list[dict[str, Any]] = []
    if args.qa_against_v1:
        if args.arm in {"fixed_basis", "both"}:
            current = concat_checkpoints(paths["cache_dir"], "fixed_basis", "metrics")
            reference = pd.read_csv(paths["v1_fixed_metrics"])
            qa_reports.append(compare_to_v1(current, reference, "fixed_basis"))
        if args.arm in {"basis_refit", "both"}:
            current = concat_checkpoints(paths["cache_dir"], "basis_refit", "metrics")
            current = current.loc[current["target_id"].isin(V1_BASIS_TARGET_IDS)].copy()
            if not current.empty:
                reference = pd.read_csv(paths["v1_basis_metrics"])
                qa_reports.append(compare_to_v1(current, reference, "basis_refit"))
        write_json(paths["table_dir"] / "v1_qa_report.json", {"reports": qa_reports})
        failed = [report for report in qa_reports if not report["passed"]]
        if failed:
            state["status"] = "failed_qa"
            state["qa"] = qa_reports
            write_run_state(paths["table_dir"] / "RUN_STATE.json", state)
            LOGGER.error("v1 QA failed: %s", failed)
            return 2
    state["status"] = "complete"
    state["written"] = written
    state["qa"] = qa_reports
    write_run_state(paths["table_dir"] / "RUN_STATE.json", state)
    LOGGER.info("Run complete. Outputs in %s", paths["table_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
