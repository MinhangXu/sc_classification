"""Per-patient overlap of Stage 1 factor scores.

The Stage 1 basis is already fit on every eligible cell. This module does not
refit PCA, factor analysis, or FactoSig. It asks whether patients occupy the
same score distributions on that shared basis, or sit on disjoint supports
that should be treated as separate subtypes.

The decision quantity is histogram overlap, not a cell-level p-value. Cells
inside a patient are dependent, so a two-sample KS test on cells rejects
trivial shifts. A within-label permutation of patient labels calibrates how
far the observed overlap sits from exchangeable patients with the same
per-patient counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
N_BINS = 32
NORMAL_MIN_CELLS = 200
CANCER_MIN_CELLS = 20
N_PERM_DEFAULT = 25
RANDOM_SEED = 42

# Absolute-overlap gates. They are magnitude rules, not p-value cutoffs.
NORMAL_SHARED_OVERLAP = 0.75
NORMAL_SHARED_MIN_PATIENT = 0.60
NORMAL_SUBTYPE_MIN_PATIENT = 0.40
CANCER_SHARED_OVERLAP = 0.45
CANCER_SUBTYPE_OVERLAP = 0.35
CANCER_SUBTYPE_GAP = 0.15
CANCER_SHARED_MAX_GAP = 0.15


@dataclass(frozen=True)
class StratumResult:
    stratum: str
    n_cells: int
    n_patients: int
    patients: list[str]
    patient_n: dict[str, int]
    weighted_overlap: float
    unweighted_mean_median_overlap: float
    min_factor_median_overlap: float
    min_factor_index: int
    min_factor_var_share: float
    min_patient_vs_rest_overlap: float
    min_patient_id: str
    patient_vs_rest: dict[str, float]
    max_eta2: float
    max_eta2_factor_index: int
    factor_median_overlap: np.ndarray
    factor_var_share: np.ndarray
    factor_eta2: np.ndarray
    null_weighted_overlap: np.ndarray

    @property
    def null_mean(self) -> float:
        if self.null_weighted_overlap.size == 0:
            return float("nan")
        return float(np.mean(self.null_weighted_overlap))

    @property
    def null_std(self) -> float:
        return float(np.std(self.null_weighted_overlap, ddof=1)) if self.null_weighted_overlap.size > 1 else 0.0

    @property
    def overlap_gap(self) -> float:
        return self.null_mean - self.weighted_overlap

    @property
    def permutation_p_lower(self) -> float:
        n_perm = int(self.null_weighted_overlap.size)
        n_as_small = int(np.sum(self.null_weighted_overlap <= self.weighted_overlap))
        return (1.0 + n_as_small) / (n_perm + 1.0)


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "experiments").exists():
            return candidate
    fallback = Path("/home/minhang/mds_project/sc_classification")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not locate the sc_classification repository root")


def experiment_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    return root / "experiments" / EXPERIMENT_ID


def output_dirs(repo_root: Path | None = None) -> tuple[Path, Path]:
    exp = experiment_dir(repo_root)
    table_dir = exp / "analysis" / "scorecards" / "stage2_diagnostic" / "patient_factor_overlap_v1"
    figure_dir = exp / "analysis" / "figures" / "stage2_diagnostic" / "patient_factor_overlap_v1"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def load_cell_metadata(repo_root: Path | None = None) -> pd.DataFrame:
    """Row order matches Stage 1 ``scores.npy`` (asserted in the D1a notebook)."""
    import anndata as ad

    path = (
        experiment_dir(repo_root)
        / "analysis"
        / "cache"
        / "stage2_diagnostic"
        / "deg_hvg_lowsupport_v1"
        / "mrd_cite_binary_log1p_15560.h5ad"
    )
    adata = ad.read_h5ad(path, backed="r")
    if adata.n_obs != 60_118:
        raise AssertionError(f"Expected 60118 cells, found {adata.n_obs}")
    frame = adata.obs[["patient", "CN.label", "predicted.annotation"]].copy()
    frame.index = frame.index.astype(str)
    frame["patient"] = frame["patient"].astype(str)
    frame["CN.label"] = frame["CN.label"].astype(str)
    frame["predicted.annotation"] = frame["predicted.annotation"].astype(str)
    return frame


def select_screen_representations(repo_root: Path | None = None) -> pd.DataFrame:
    """One D1-atlas DR representation per panel: highest patient-median lift."""
    atlas_path = (
        experiment_dir(repo_root)
        / "analysis"
        / "scorecards"
        / "stage2_diagnostic"
        / "old34_d1_atlas_representation_table.csv"
    )
    taxonomy_path = (
        experiment_dir(repo_root)
        / "analysis"
        / "scorecards"
        / "stage2_diagnostic"
        / "old34_panel_diagnostic_table.csv"
    )
    atlas = pd.read_csv(atlas_path)
    dr = atlas.loc[atlas["configuration"].ne("direct_gene")].copy()
    dr["patient_median_lift"] = pd.to_numeric(dr["patient_median_lift"], errors="raise")
    dr["patient_p20_lift"] = pd.to_numeric(dr["patient_p20_lift"], errors="raise")
    dr["effective_k"] = pd.to_numeric(dr["effective_k"], errors="raise").astype(int)
    chosen_index = dr.groupby(["atlas_row_block", "stage0_panel_id"], observed=True)["patient_median_lift"].idxmax()
    chosen = dr.loc[chosen_index].copy()
    taxonomy = pd.read_csv(
        taxonomy_path,
        usecols=["stage0_panel_id", "provisional_three_way_taxonomy"],
    )
    chosen = chosen.merge(taxonomy, on="stage0_panel_id", how="left")
    chosen["representation_id"] = chosen.apply(
        lambda row: f"{row.stage0_panel_id}|dr|{row.stage1_method}|{int(row.effective_k)}|42",
        axis=1,
    )
    return chosen.sort_values(["atlas_row_block", "short_panel_label"]).reset_index(drop=True)


def score_path(repo_root: Path, panel_id: str, method: str, k: int) -> Path:
    return (
        experiment_dir(repo_root)
        / "stage1_dr"
        / f"panel_id={panel_id}"
        / f"method={method}"
        / f"k={int(k)}"
        / "seed=42"
        / "scores.npy"
    )


def _edges_for_matrix(x: np.ndarray, n_bins: int = N_BINS) -> list[np.ndarray]:
    edges: list[np.ndarray] = []
    for j in range(x.shape[1]):
        column = x[:, j]
        lo, hi = np.quantile(column, [0.01, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-8:
            edges.append(np.array([-0.5, 0.5], dtype=float))
        else:
            edges.append(np.linspace(float(lo), float(hi), n_bins + 1))
    return edges


def _densities(column: np.ndarray, groups: list[np.ndarray], edges: np.ndarray) -> np.ndarray:
    densities = np.empty((len(groups), edges.size - 1), dtype=np.float64)
    for i, group in enumerate(groups):
        values = column[group]
        hist, _ = np.histogram(values, bins=edges)
        hist = hist.astype(np.float64, copy=False)
        hist[0] += np.count_nonzero(values < edges[0])
        hist[-1] += np.count_nonzero(values > edges[-1])
        total = hist.sum()
        if total <= 0:
            densities[i] = 0.0
        else:
            densities[i] = hist / total
    return densities


def _factor_eta2(column: np.ndarray, groups: list[np.ndarray]) -> float:
    counts = np.array([group.size for group in groups], dtype=np.float64)
    means = np.array([column[group].mean() for group in groups], dtype=np.float64)
    grand = np.average(means, weights=counts)
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_total = float(np.sum((column - grand) ** 2))
    if ss_total <= 0:
        return 0.0
    return ss_between / ss_total


def _overlap_from_labels(
    x: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    edges: list[np.ndarray],
    variance: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    groups = [np.flatnonzero(labels == name) for name in names]
    n_factors = x.shape[1]
    median_overlap = np.empty(n_factors, dtype=np.float64)
    patient_vs_rest = np.empty((len(names), n_factors), dtype=np.float64)
    counts = np.array([group.size for group in groups], dtype=np.float64)
    upper = np.triu_indices(len(names), 1)
    for j in range(n_factors):
        densities = _densities(x[:, j], groups, edges[j])
        pair_matrix = np.minimum(densities[:, None, :], densities[None, :, :]).sum(axis=-1)
        median_overlap[j] = float(np.median(pair_matrix[upper]))
        for i in range(len(names)):
            rest_weights = counts.copy()
            rest_weights[i] = 0.0
            rest = np.average(densities, axis=0, weights=rest_weights)
            patient_vs_rest[i, j] = float(np.minimum(densities[i], rest).sum())
    weights = variance / variance.sum()
    weighted = float(np.dot(weights, median_overlap))
    patient_summary = patient_vs_rest @ weights
    return weighted, median_overlap, patient_summary


def evaluate_stratum(
    scores: np.ndarray,
    patient: np.ndarray,
    mask: np.ndarray,
    *,
    stratum: str,
    min_cells: int,
    n_perm: int,
    rng: np.random.Generator,
) -> StratumResult | None:
    idx = np.flatnonzero(mask)
    labels_full = patient[idx]
    names = [name for name in pd.unique(labels_full) if int(np.sum(labels_full == name)) >= min_cells]
    names = sorted(names)
    if len(names) < 3:
        return None
    keep = np.isin(labels_full, names)
    x = np.ascontiguousarray(scores[idx[keep]])
    labels = labels_full[keep]
    groups = [np.flatnonzero(labels == name) for name in names]
    included = np.concatenate(groups)
    variance = x[included].var(axis=0)
    if float(variance.sum()) <= 0:
        variance = np.ones(x.shape[1], dtype=np.float64)
    edges = _edges_for_matrix(x[included])
    weighted, median_overlap, patient_summary = _overlap_from_labels(x, labels, names, edges, variance)
    eta2 = np.array([_factor_eta2(x[:, j], groups) for j in range(x.shape[1])], dtype=np.float64)
    if n_perm > 0:
        nulls = np.empty(n_perm, dtype=np.float64)
        for perm_index in range(n_perm):
            shuffled = rng.permutation(labels)
            nulls[perm_index], _, _ = _overlap_from_labels(x, shuffled, names, edges, variance)
    else:
        nulls = np.empty(0, dtype=np.float64)
    var_share = variance / variance.sum()
    min_factor = int(np.argmin(median_overlap))
    min_patient = int(np.argmin(patient_summary))
    patient_n = {name: int(group.size) for name, group in zip(names, groups)}
    return StratumResult(
        stratum=stratum,
        n_cells=int(x.shape[0]),
        n_patients=len(names),
        patients=names,
        patient_n=patient_n,
        weighted_overlap=weighted,
        unweighted_mean_median_overlap=float(median_overlap.mean()),
        min_factor_median_overlap=float(median_overlap[min_factor]),
        min_factor_index=min_factor + 1,
        min_factor_var_share=float(var_share[min_factor]),
        min_patient_vs_rest_overlap=float(patient_summary[min_patient]),
        min_patient_id=names[min_patient],
        patient_vs_rest={name: float(value) for name, value in zip(names, patient_summary)},
        max_eta2=float(eta2.max()),
        max_eta2_factor_index=int(np.argmax(eta2)) + 1,
        factor_median_overlap=median_overlap,
        factor_var_share=var_share,
        factor_eta2=eta2,
        null_weighted_overlap=nulls,
    )


def aggregation_call(normal: StratumResult, cancer: StratumResult | None) -> str:
    """Map overlap magnitudes to aggregate / shift / separate-subtype.

    A small permutation p-value is not enough to split patients. With tens of
    thousands of cells, exact exchangeability is rejected by location shifts
    whose supports still overlap. Separate-subtype requires a low absolute
    overlap, and for malignant cells that low overlap must also sit well
    below the small-sample null.
    """
    normal_shared = (
        normal.weighted_overlap >= NORMAL_SHARED_OVERLAP
        and normal.min_patient_vs_rest_overlap >= NORMAL_SHARED_MIN_PATIENT
    )
    normal_split = normal.min_patient_vs_rest_overlap < NORMAL_SUBTYPE_MIN_PATIENT
    if cancer is None:
        cancer_shared = True
        cancer_split = False
    else:
        cancer_shared = (
            cancer.weighted_overlap >= CANCER_SHARED_OVERLAP
            and cancer.overlap_gap < CANCER_SHARED_MAX_GAP
        )
        cancer_split = (
            cancer.weighted_overlap < CANCER_SUBTYPE_OVERLAP
            and cancer.overlap_gap >= CANCER_SUBTYPE_GAP
        )
    if normal_split or cancer_split:
        return "separate_subtype_candidate"
    if normal_shared and cancer_shared:
        return "aggregate_shared"
    return "shared_with_location_shift"


def _stratum_row(representation: pd.Series, result: StratumResult) -> dict:
    return {
        "representation_id": representation.representation_id,
        "stage0_panel_id": representation.stage0_panel_id,
        "short_panel_label": representation.short_panel_label,
        "atlas_row_block": representation.atlas_row_block,
        "stage1_method": representation.stage1_method,
        "effective_k": int(representation.effective_k),
        "patient_median_lift": float(representation.patient_median_lift),
        "patient_p20_lift": float(representation.patient_p20_lift),
        "provisional_three_way_taxonomy": representation.provisional_three_way_taxonomy,
        "stratum": result.stratum,
        "n_cells": result.n_cells,
        "n_patients": result.n_patients,
        "patients": ",".join(result.patients),
        "weighted_overlap": result.weighted_overlap,
        "unweighted_mean_median_overlap": result.unweighted_mean_median_overlap,
        "min_factor_median_overlap": result.min_factor_median_overlap,
        "min_factor_index": result.min_factor_index,
        "min_factor_var_share": result.min_factor_var_share,
        "min_patient_vs_rest_overlap": result.min_patient_vs_rest_overlap,
        "min_patient_id": result.min_patient_id,
        "max_eta2": result.max_eta2,
        "max_eta2_factor_index": result.max_eta2_factor_index,
        "null_mean_weighted_overlap": result.null_mean,
        "null_std_weighted_overlap": result.null_std,
        "overlap_gap": result.overlap_gap,
        "n_perm": int(result.null_weighted_overlap.size),
        "permutation_p_lower": result.permutation_p_lower,
    }


def run_screen(
    repo_root: Path | None = None,
    *,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    root = repo_root or find_repo_root()
    metadata = load_cell_metadata(root)
    patient = metadata["patient"].to_numpy()
    label = metadata["CN.label"].to_numpy()
    representations = select_screen_representations(root)
    rng = np.random.default_rng(seed)
    stratum_rows: list[dict] = []
    patient_rows: list[dict] = []
    call_rows: list[dict] = []
    for representation in representations.itertuples(index=False):
        path = score_path(root, representation.stage0_panel_id, representation.stage1_method, representation.effective_k)
        scores = np.load(path)
        if scores.shape[0] != patient.size:
            raise AssertionError(f"{path} has {scores.shape[0]} rows, metadata has {patient.size}")
        series = pd.Series(representation._asdict())
        normal = evaluate_stratum(
            scores,
            patient,
            label == "normal",
            stratum="normal",
            min_cells=NORMAL_MIN_CELLS,
            n_perm=n_perm,
            rng=rng,
        )
        cancer = evaluate_stratum(
            scores,
            patient,
            label == "cancer",
            stratum="cancer",
            min_cells=CANCER_MIN_CELLS,
            n_perm=n_perm,
            rng=rng,
        )
        if normal is None:
            raise RuntimeError(f"Normal-cell stratum failed for {representation.stage0_panel_id}")
        call = aggregation_call(normal, cancer)
        stratum_rows.append(_stratum_row(series, normal))
        for name, overlap in normal.patient_vs_rest.items():
            patient_rows.append(
                {
                    "representation_id": representation.representation_id,
                    "short_panel_label": representation.short_panel_label,
                    "stage1_method": representation.stage1_method,
                    "effective_k": int(representation.effective_k),
                    "stratum": "normal",
                    "patient": name,
                    "n_cells": normal.patient_n[name],
                    "vs_rest_weighted_overlap": overlap,
                }
            )
        if cancer is not None:
            stratum_rows.append(_stratum_row(series, cancer))
            for name, overlap in cancer.patient_vs_rest.items():
                patient_rows.append(
                    {
                        "representation_id": representation.representation_id,
                        "short_panel_label": representation.short_panel_label,
                        "stage1_method": representation.stage1_method,
                        "effective_k": int(representation.effective_k),
                        "stratum": "cancer",
                        "patient": name,
                        "n_cells": cancer.patient_n[name],
                        "vs_rest_weighted_overlap": overlap,
                    }
                )
        call_rows.append(
            {
                "representation_id": representation.representation_id,
                "stage0_panel_id": representation.stage0_panel_id,
                "short_panel_label": representation.short_panel_label,
                "atlas_row_block": representation.atlas_row_block,
                "stage1_method": representation.stage1_method,
                "effective_k": int(representation.effective_k),
                "patient_median_lift": float(representation.patient_median_lift),
                "patient_p20_lift": float(representation.patient_p20_lift),
                "provisional_three_way_taxonomy": representation.provisional_three_way_taxonomy,
                "aggregation_call": call,
                "normal_weighted_overlap": normal.weighted_overlap,
                "normal_overlap_gap": normal.overlap_gap,
                "normal_min_patient_vs_rest": normal.min_patient_vs_rest_overlap,
                "normal_min_patient_id": normal.min_patient_id,
                "normal_max_eta2": normal.max_eta2,
                "normal_permutation_p_lower": normal.permutation_p_lower,
                "cancer_weighted_overlap": None if cancer is None else cancer.weighted_overlap,
                "cancer_overlap_gap": None if cancer is None else cancer.overlap_gap,
                "cancer_min_patient_vs_rest": None if cancer is None else cancer.min_patient_vs_rest_overlap,
                "cancer_min_patient_id": None if cancer is None else cancer.min_patient_id,
                "cancer_max_eta2": None if cancer is None else cancer.max_eta2,
                "cancer_max_eta2_factor_index": None if cancer is None else cancer.max_eta2_factor_index,
                "cancer_permutation_p_lower": None if cancer is None else cancer.permutation_p_lower,
                "cancer_n_patients": 0 if cancer is None else cancer.n_patients,
            }
        )
    return {
        "representations": representations,
        "by_stratum": pd.DataFrame(stratum_rows),
        "by_patient": pd.DataFrame(patient_rows),
        "calls": pd.DataFrame(call_rows),
    }


def focus_factor_table(
    repo_root: Path | None = None,
    *,
    n_perm: int = N_PERM_DEFAULT,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor-level overlap for the meeting focus bases, including both IFN-α geometries."""
    root = repo_root or find_repo_root()
    metadata = load_cell_metadata(root)
    patient = metadata["patient"].to_numpy()
    label = metadata["CN.label"].to_numpy()
    annotation = metadata["predicted.annotation"].to_numpy()
    specs = [
        ("allograft rejection", "single_geneset__hallmark_allograft_rejection", "factosig_promax", 40),
        ("apoptosis", "single_geneset__hallmark_apoptosis", "factosig_promax", 40),
        ("interferon alpha", "single_geneset__hallmark_interferon_alpha_response", "pca", 40),
        ("interferon alpha", "single_geneset__hallmark_interferon_alpha_response", "fa", 40),
    ]
    rng = np.random.default_rng(seed)
    factor_rows: list[dict] = []
    celltype_rows: list[dict] = []
    for label_name, panel_id, method, k in specs:
        scores = np.load(score_path(root, panel_id, method, k))
        for stratum, mask, min_cells in (
            ("normal", label == "normal", NORMAL_MIN_CELLS),
            ("cancer", label == "cancer", CANCER_MIN_CELLS),
        ):
            result = evaluate_stratum(
                scores,
                patient,
                mask,
                stratum=stratum,
                min_cells=min_cells,
                n_perm=n_perm,
                rng=rng,
            )
            if result is None:
                continue
            for index in range(result.factor_median_overlap.size):
                factor_rows.append(
                    {
                        "panel": label_name,
                        "stage0_panel_id": panel_id,
                        "stage1_method": method,
                        "effective_k": k,
                        "stratum": stratum,
                        "factor_index": index + 1,
                        "median_pairwise_overlap": float(result.factor_median_overlap[index]),
                        "var_share": float(result.factor_var_share[index]),
                        "eta2": float(result.factor_eta2[index]),
                    }
                )
        for cell_type in (
            "CLP",
            "MEP-EP",
            "HSC/MPP2",
            "cDC",
            "NP1",
            "NP2",
            "pDC2",
            "LMPP",
        ):
            type_mask = (label == "normal") & (annotation == cell_type)
            result = evaluate_stratum(
                scores,
                patient,
                type_mask,
                stratum=f"normal|{cell_type}",
                min_cells=80,
                n_perm=min(n_perm, 15),
                rng=rng,
            )
            if result is None:
                continue
            celltype_rows.append(
                {
                    "panel": label_name,
                    "stage1_method": method,
                    "effective_k": k,
                    "cell_type": cell_type,
                    "n_patients": result.n_patients,
                    "n_cells": result.n_cells,
                    "weighted_overlap": result.weighted_overlap,
                    "null_mean_weighted_overlap": result.null_mean,
                    "overlap_gap": result.overlap_gap,
                    "min_patient_vs_rest_overlap": result.min_patient_vs_rest_overlap,
                    "min_patient_id": result.min_patient_id,
                }
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(celltype_rows)


def write_screen(tables: dict[str, pd.DataFrame], repo_root: Path | None = None) -> Path:
    table_dir, _ = output_dirs(repo_root)
    tables["by_stratum"].to_csv(table_dir / "patient_factor_overlap_by_stratum.csv", index=False)
    tables["by_patient"].to_csv(table_dir / "patient_factor_overlap_by_patient.csv", index=False)
    tables["calls"].to_csv(table_dir / "patient_factor_overlap_calls.csv", index=False)
    return table_dir


def low_overlap_factors(
    repo_root: Path | None = None,
    *,
    min_var_share: float = 0.05,
    max_median_overlap: float = 0.50,
) -> pd.DataFrame:
    """Factors whose own support separates patients, hiding inside an average.

    The panel call is variance-weighted. A single high-variance factor can still
    be patient-private. This table keeps every factor with variance share at
    least ``min_var_share`` and median pairwise overlap below ``max_median_overlap``.
    """
    root = repo_root or find_repo_root()
    metadata = load_cell_metadata(root)
    patient = metadata["patient"].to_numpy()
    label = metadata["CN.label"].to_numpy()
    representations = select_screen_representations(root)
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for representation in representations.itertuples(index=False):
        scores = np.load(
            score_path(root, representation.stage0_panel_id, representation.stage1_method, int(representation.effective_k))
        )
        for stratum, mask, min_cells in (
            ("normal", label == "normal", NORMAL_MIN_CELLS),
            ("cancer", label == "cancer", CANCER_MIN_CELLS),
        ):
            result = evaluate_stratum(
                scores,
                patient,
                mask,
                stratum=stratum,
                min_cells=min_cells,
                n_perm=0,
                rng=rng,
            )
            if result is None:
                continue
            for index in range(result.factor_median_overlap.size):
                var_share = float(result.factor_var_share[index])
                overlap = float(result.factor_median_overlap[index])
                if var_share < min_var_share or overlap >= max_median_overlap:
                    continue
                rows.append(
                    {
                        "representation_id": representation.representation_id,
                        "short_panel_label": representation.short_panel_label,
                        "atlas_row_block": representation.atlas_row_block,
                        "stage1_method": representation.stage1_method,
                        "effective_k": int(representation.effective_k),
                        "stratum": stratum,
                        "factor_index": index + 1,
                        "median_pairwise_overlap": overlap,
                        "var_share": var_share,
                        "eta2": float(result.factor_eta2[index]),
                    }
                )
    return pd.DataFrame(rows)


def save_overlap_figures(repo_root: Path | None = None) -> Path:
    """Write D14 figures from the tables produced by ``run_screen``."""
    import matplotlib.pyplot as plt

    root = repo_root or find_repo_root()
    table_dir, figure_dir = output_dirs(root)
    calls = pd.read_csv(table_dir / "patient_factor_overlap_calls.csv")
    by_patient = pd.read_csv(table_dir / "patient_factor_overlap_by_patient.csv")
    celltype = pd.read_csv(table_dir / "focus_celltype_overlap.csv")
    metadata = load_cell_metadata(root)
    patient = metadata["patient"].to_numpy()
    label = metadata["CN.label"].to_numpy()

    call_colors = {
        "aggregate_shared": "#2c7fb8",
        "shared_with_location_shift": "#e6550d",
        "separate_subtype_candidate": "#d7301f",
    }
    patient_colors = {
        "P01": "#1b9e77",
        "P02": "#d95f02",
        "P03": "#7570b3",
        "P04": "#e7298a",
        "P06": "#66a61e",
        "P07": "#e6ab02",
        "P09": "#a6761d",
    }

    def _save(fig: plt.Figure, stem: str) -> None:
        fig.savefig(figure_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # D14A — panel-level overlap. Dashed lines are the absolute gates, not p-values.
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for call_name, color in call_colors.items():
        subset = calls.loc[calls["aggregation_call"].eq(call_name)]
        if subset.empty:
            continue
        biological = subset["atlas_row_block"].eq("biological")
        ax.scatter(
            subset.loc[biological, "normal_weighted_overlap"],
            subset.loc[biological, "cancer_weighted_overlap"],
            s=46,
            color=color,
            label=call_name.replace("_", " "),
            zorder=3,
        )
        ax.scatter(
            subset.loc[~biological, "normal_weighted_overlap"],
            subset.loc[~biological, "cancer_weighted_overlap"],
            s=54,
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            zorder=4,
        )
    ax.axvline(NORMAL_SHARED_OVERLAP, color="0.45", ls="--", lw=0.8)
    ax.axhline(CANCER_SHARED_OVERLAP, color="0.45", ls="--", lw=0.8)
    for panel_name in (
        "allograft rejection",
        "interferon alpha response",
        "apoptosis",
        "il2 stat5 signaling",
        "HVGs (10000)",
        "nf kb activation",
    ):
        row = calls.loc[calls["short_panel_label"].eq(panel_name)].iloc[0]
        ax.annotate(
            panel_name,
            (row.normal_weighted_overlap, row.cancer_weighted_overlap),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    ax.set_xlabel("Normal-cell weighted overlap")
    ax.set_ylabel("Malignant-cell weighted overlap")
    ax.set_title("D14A. Patients share factor support on the D1-selected bases")
    ax.set_xlim(0.74, 0.92)
    ax.set_ylim(0.47, 0.70)
    ax.scatter([], [], s=54, facecolors="none", edgecolors="0.2", linewidths=1.4, label="Open marker = control")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "D14A_panel_factor_overlap")

    # D14B — permutation gap. Every biological panel is detectable; few are large.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for axis, stratum_column, title in (
        (axes[0], "normal_overlap_gap", "Normal cells"),
        (axes[1], "cancer_overlap_gap", "Malignant cells"),
    ):
        biological = calls.loc[calls["atlas_row_block"].eq("biological"), stratum_column]
        control = calls.loc[~calls["atlas_row_block"].eq("biological"), stratum_column]
        axis.hist(biological, bins=12, color="#2c7fb8", alpha=0.85, label="41 biological")
        axis.hist(control, bins=8, color="#e6550d", alpha=0.75, label="6 controls")
        axis.set_xlabel("Null overlap minus observed overlap")
        axis.set_ylabel("Panels")
        axis.set_title(title)
        axis.legend(frameon=False)
    axes[1].axvline(CANCER_SUBTYPE_GAP, color="0.3", ls="--", lw=0.8)
    axes[1].text(CANCER_SUBTYPE_GAP + 0.004, axes[1].get_ylim()[1] * 0.92, "subtype gap gate", fontsize=8, color="0.25")
    fig.suptitle("D14B. Exchangeability is rejected; the overlap deficit is small", y=1.02)
    fig.tight_layout()
    _save(fig, "D14B_overlap_gap_versus_permutation")

    # D14C — meeting focus panels, each patient against the other patients.
    focus = by_patient.loc[
        by_patient["short_panel_label"].isin(["allograft rejection", "apoptosis", "interferon alpha response"])
        & (
            (
                by_patient["short_panel_label"].eq("interferon alpha response")
                & by_patient["stage1_method"].isin(["pca", "fa"])
                & by_patient["effective_k"].eq(40)
            )
            | (
                by_patient["short_panel_label"].isin(["allograft rejection", "apoptosis"])
                & by_patient["stage1_method"].eq("factosig_promax")
                & by_patient["effective_k"].eq(40)
            )
        )
    ].copy()
    # The screen stores the D1-selected row. IFN-α's selected method is PCA, so FA is absent
    # from by_patient. Rebuild the four focus contrasts from stored scores below if needed.
    if focus.loc[focus["stage1_method"].eq("fa")].empty:
        focus_fa_rows = []
        scores = np.load(
            score_path(root, "single_geneset__hallmark_interferon_alpha_response", "fa", 40)
        )
        rng = np.random.default_rng(0)
        for stratum, mask, min_cells in (
            ("normal", label == "normal", NORMAL_MIN_CELLS),
            ("cancer", label == "cancer", CANCER_MIN_CELLS),
        ):
            result = evaluate_stratum(
                scores, patient, mask, stratum=stratum, min_cells=min_cells, n_perm=0, rng=rng
            )
            for name, overlap in result.patient_vs_rest.items():
                focus_fa_rows.append(
                    {
                        "short_panel_label": "interferon alpha response",
                        "stage1_method": "fa",
                        "effective_k": 40,
                        "stratum": stratum,
                        "patient": name,
                        "vs_rest_weighted_overlap": overlap,
                    }
                )
        focus = pd.concat([focus, pd.DataFrame(focus_fa_rows)], ignore_index=True)
    focus["contrast"] = focus["short_panel_label"].map(
        {
            "allograft rejection": "Allograft promax K40",
            "apoptosis": "Apoptosis promax K40",
            "interferon alpha response": "IFN-α",
        }
    )
    focus.loc[focus["stage1_method"].eq("pca"), "contrast"] = "IFN-α PCA K40"
    focus.loc[focus["stage1_method"].eq("fa"), "contrast"] = "IFN-α FA K40"
    order = ["Allograft promax K40", "Apoptosis promax K40", "IFN-α PCA K40", "IFN-α FA K40"]
    patients_order = ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=True)
    for axis, stratum in zip(axes, ("normal", "cancer")):
        subset = focus.loc[focus["stratum"].eq(stratum)]
        width = 0.11
        x = np.arange(len(order))
        for offset, patient_id in enumerate(patients_order):
            heights = []
            for contrast in order:
                value = subset.loc[
                    subset["contrast"].eq(contrast) & subset["patient"].eq(patient_id),
                    "vs_rest_weighted_overlap",
                ]
                heights.append(float(value.iloc[0]) if len(value) else np.nan)
            axis.bar(
                x + (offset - 3) * width,
                heights,
                width=width,
                color=patient_colors[patient_id],
                label=patient_id,
            )
        axis.set_ylim(0.35, 1.0)
        axis.set_ylabel("Overlap with other patients")
        axis.set_title("Normal cells" if stratum == "normal" else "Malignant cells (n ≥ 20)")
        axis.axhline(0.60 if stratum == "normal" else 0.45, color="0.4", ls="--", lw=0.8)
    axes[1].set_xticks(np.arange(len(order)), order)
    axes[0].legend(ncol=7, frameon=False, loc="upper right")
    fig.suptitle("D14C. No supported patient is disjoint from the others on the focus bases", y=1.01)
    fig.tight_layout()
    _save(fig, "D14C_focus_patient_vs_rest_overlap")

    # D14D — one shared biological factor, the closest biological dip, and the P03 broad-bag factor.
    histogram_specs = [
        (
            "IFN-α PCA K40, factor 1\nmalignant cells",
            "single_geneset__hallmark_interferon_alpha_response",
            "pca",
            40,
            0,
        ),
        (
            "NF-κB FA K10, factor 2\nmalignant cells",
            "single_geneset__reactome_nf_kb_activation",
            "fa",
            10,
            1,
        ),
        (
            "HVG 10k FactoSig K40, factor 21\nmalignant cells",
            "hvg_top_requested_10000__available_10000",
            "factosig",
            40,
            20,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    malignant_patients = ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]
    for axis, (title, panel_id, method, k, factor_index) in zip(axes, histogram_specs):
        scores = np.load(score_path(root, panel_id, method, k))
        column = scores[:, factor_index]
        mask = label == "cancer"
        values = [column[(patient == patient_id) & mask] for patient_id in malignant_patients]
        # Common bins across patients so a shifted cloud stays visible.
        shown = column[mask & np.isin(patient, malignant_patients)]
        if "HVG" in title:
            pad = 0.04 * float(shown.max() - shown.min())
            bins = np.linspace(float(shown.min()) - pad, float(shown.max()) + pad, 40)
        else:
            lo, hi = np.quantile(shown, [0.01, 0.99])
            bins = np.linspace(float(lo), float(hi), 28)
        for patient_id, sample in zip(malignant_patients, values):
            plotted = sample[(sample >= bins[0]) & (sample <= bins[-1])]
            axis.hist(
                plotted,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.5,
                color=patient_colors[patient_id],
                label=f"{patient_id} (n={sample.size})",
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Factor score" if "HVG" in title else "Factor score (central 98%)")
        axis.set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("D14D. Curated factors overlap; one HVG factor isolates P03 malignant cells", y=1.03)
    fig.tight_layout()
    _save(fig, "D14D_factor_score_distributions")

    # D14E — within cell type, the normal-cell gap shrinks further.
    heat = celltype.pivot_table(index=["panel", "stage1_method"], columns="cell_type", values="overlap_gap")
    column_order = ["CLP", "MEP-EP", "HSC/MPP2", "cDC", "NP1", "NP2", "pDC2", "LMPP"]
    heat = heat.reindex(columns=column_order)
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    image = ax.imshow(heat.to_numpy(), cmap="YlOrRd", vmin=0, vmax=0.12, aspect="auto")
    ax.set_xticks(np.arange(heat.shape[1]), heat.columns, rotation=40, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]), [f"{panel} / {method}" for panel, method in heat.index])
    for row_index in range(heat.shape[0]):
        for column_index in range(heat.shape[1]):
            value = heat.to_numpy()[row_index, column_index]
            if np.isfinite(value):
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Null overlap − observed overlap")
    ax.set_title("D14E. Within normal cell types the patient-overlap gap stays under 0.10")
    fig.tight_layout()
    _save(fig, "D14E_celltype_overlap_gap")
    return figure_dir


def write_findings(repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    table_dir, figure_dir = output_dirs(root)
    calls = pd.read_csv(table_dir / "patient_factor_overlap_calls.csv")
    flags = pd.read_csv(table_dir / "low_overlap_high_variance_factors.csv")
    biological = calls.loc[calls["atlas_row_block"].eq("biological")]
    private = flags.loc[flags["eta2"].ge(0.75) & flags["var_share"].ge(0.15)].sort_values("eta2", ascending=False)
    allograft = calls.loc[calls["short_panel_label"].eq("allograft rejection")].iloc[0]
    lines = [
        "# Patient factor-overlap findings",
        "",
        "Stage 1 factors were already learned on all 60,118 eligible cells. This note projects those stored scores back onto patients. It does not refit DR.",
        "",
        f"Screen: {len(calls)} D1-selected representations ({biological.shape[0]} biological panels + {len(calls) - biological.shape[0]} controls). One representation per panel, the D1 atlas DR cell with the highest patient-median lift.",
        "",
        "## Calls",
        "",
        calls["aggregation_call"].value_counts().to_string(),
        "",
        (
            f"Biological normal-cell weighted overlap ranges from {biological['normal_weighted_overlap'].min():.3f} "
            f"to {biological['normal_weighted_overlap'].max():.3f} "
            f"(median {biological['normal_weighted_overlap'].median():.3f}). "
            f"Malignant-cell weighted overlap ranges from {biological['cancer_weighted_overlap'].min():.3f} "
            f"to {biological['cancer_weighted_overlap'].max():.3f} "
            f"(median {biological['cancer_weighted_overlap'].median():.3f}). "
            f"Median malignant overlap gap versus the within-label permutation null is "
            f"{biological['cancer_overlap_gap'].median():.3f}."
        ),
        "",
        (
            f"Allograft rejection is the only biological panel not labeled aggregate_shared. "
            f"Its panel-average overlaps are normal {allograft.normal_weighted_overlap:.3f} and "
            f"malignant {allograft.cancer_weighted_overlap:.3f}. The label changes because "
            f"{allograft.normal_min_patient_id} normal cells overlap the other patients at "
            f"{allograft.normal_min_patient_vs_rest:.3f}, just under the 0.60 minimum-patient gate. "
            "That is a location shift, not a disjoint subtype."
        ),
        "",
        "## High-variance factors that separate one patient",
        "",
        "A variance-weighted average can hide one patient-private factor. Factors below have malignant-cell eta-squared at least 0.75 and variance share at least 0.15.",
        "",
        private.to_string(index=False) if len(private) else "None.",
        "",
        "Patient-versus-rest overlap on each of those factors (malignant cells, seven supported patients):",
        "",
    ]
    metadata = load_cell_metadata(root)
    patient_ids = metadata["patient"].to_numpy()
    labels = metadata["CN.label"].to_numpy()
    supported = ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]
    for row in private.itertuples(index=False):
        scores = np.load(score_path(root, row.representation_id.split("|dr|")[0], row.stage1_method, int(row.effective_k)))
        mask = (labels == "cancer") & np.isin(patient_ids, supported)
        column = scores[mask, int(row.factor_index) - 1][:, None]
        edges = _edges_for_matrix(column)
        names = sorted(supported)
        _, _, versus_rest = _overlap_from_labels(
            column,
            patient_ids[mask],
            names,
            edges,
            np.array([1.0]),
        )
        worst = int(np.argmin(versus_rest))
        lines.append(
            f"- {row.short_panel_label} factor {int(row.factor_index)}: "
            f"{names[worst]} overlap with the other malignant cells is {float(versus_rest[worst]):.3f}."
        )
    lines.extend(
        [
            "",
            "No biological Hallmark, Reactome, or group-panel factor in the screened representations meets that eta-squared and variance-share pair. Those programs can stay on one shared basis.",
            "",
            f"Figures: `{figure_dir}`.",
            "",
        ]
    )
    path = table_dir / "patient_factor_overlap_findings.md"
    path.write_text("\n".join(lines))
    return path


def write_rule_card(repo_root: Path | None = None) -> None:
    table_dir, _ = output_dirs(repo_root)
    card = {
        "estimand": (
            "Variance-weighted mean of the median pairwise histogram-overlap "
            "of Stage 1 factor scores, within CN.label."
        ),
        "basis": "Existing transductive Stage 1 scores.npy. No DR refit.",
        "bins": N_BINS,
        "bin_grid": "Shared equal-width bins on the included cells' 1st-99th percentile; tails join the end bins.",
        "normal_min_cells": NORMAL_MIN_CELLS,
        "cancer_min_cells": CANCER_MIN_CELLS,
        "n_perm": N_PERM_DEFAULT,
        "permutation": "Shuffle patient labels inside the stratum, preserving per-patient counts. Bin edges stay fixed.",
        "aggregate_shared": {
            "normal_weighted_overlap_at_least": NORMAL_SHARED_OVERLAP,
            "normal_min_patient_vs_rest_at_least": NORMAL_SHARED_MIN_PATIENT,
            "cancer_weighted_overlap_at_least": CANCER_SHARED_OVERLAP,
            "cancer_gap_below": CANCER_SHARED_MAX_GAP,
        },
        "separate_subtype_candidate": {
            "normal_min_patient_vs_rest_below": NORMAL_SUBTYPE_MIN_PATIENT,
            "or_cancer_weighted_overlap_below": CANCER_SUBTYPE_OVERLAP,
            "and_cancer_gap_at_least": CANCER_SUBTYPE_GAP,
        },
        "shared_with_location_shift": "Neither gate. Permutation may reject exchangeability while supports still overlap.",
        "not_this_test": (
            "D3 paired patient-specific AUPRC gaps are predictive opportunity, not factor-support overlap. "
            "A cell-level KS p-value is not used."
        ),
    }
    (table_dir / "overlap_decision_rule.json").write_text(json.dumps(card, indent=2))
