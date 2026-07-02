"""Utilities for Stage 2 MRD multi-objective sharedness analyses.

The helpers in this module are intentionally filename-tolerant.  The Stage 2
runner writes canonical artifacts today, but the analysis notebook should keep
working when future runs add optional coefficient or prediction outputs.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # Plotting is optional for table-only artifact audits.
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - environment-dependent notebook import
    plt = None
    sns = None

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist
except Exception:  # pragma: no cover - notebook dependency fallback
    leaves_list = None
    linkage = None
    pdist = None


BASIS_MATCH_COLUMNS = [
    "stage0_panel_id",
    "representation_family",
    "stage1_method",
    "effective_k",
    "stage1_seed",
    "stage1_scores_path",
    "stage1_loadings_path",
]

PLOT_CONTEXT = {
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

ARTIFACT_PATTERNS = (
    "*multiobjective*",
    "*stage2*",
    "*discovery*",
    "*lopo*",
    "*heldout*",
    "*sharedness*",
    "*leave_patient*",
    "*patient_specific*",
    "*coefficient*",
    "*coef*",
    "*scorecard*",
    "*metrics*",
    "*by_patient*",
    "*prediction*",
    "*metadata*",
    "*loadings*",
)


def _require_plotting() -> tuple[object, object]:
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install them in the notebook kernel environment before running figure cells."
        )
    return plt, sns


def resolve_col(
    df: pd.DataFrame | None,
    candidates: Sequence[str],
    required: bool = False,
) -> str | None:
    """Return the first matching column, allowing case-insensitive lookup."""

    if df is None:
        if required:
            raise KeyError(f"Cannot resolve {candidates}: dataframe is None")
        return None
    columns = list(df.columns)
    exact = {c: c for c in columns}
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise KeyError(f"None of {list(candidates)} found in dataframe columns")
    return None


def _ensure_unique_columns(df: pd.DataFrame, context: str = "dataframe") -> pd.DataFrame:
    """Return a copy with unique column labels, preserving repeated columns with suffixes."""

    if df.columns.is_unique:
        return df
    out = df.copy()
    seen: dict[str, int] = {}
    columns: list[str] = []
    duplicated: list[str] = []
    for column in out.columns:
        name = str(column)
        count = seen.get(name, 0)
        if count == 0:
            columns.append(name)
        else:
            columns.append(f"{name}__duplicate_{count}")
            duplicated.append(name)
        seen[name] = count + 1
    out.columns = columns
    duplicate_preview = ", ".join(sorted(set(duplicated))[:8])
    more = "..." if len(set(duplicated)) > 8 else ""
    warnings.warn(
        f"{context} had duplicate column labels; suffixed repeated labels: {duplicate_preview}{more}",
        RuntimeWarning,
    )
    return out


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = path.suffixes
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix == ".json":
        with path.open() as handle:
            data = json.load(handle)
        return pd.json_normalize(data)
    raise ValueError(f"Unsupported table type: {path}")


def _read_header(path: Path) -> list[str]:
    try:
        if path.suffix == ".parquet":
            return list(pd.read_parquet(path, columns=[]).columns)
        if path.suffix == ".json":
            return list(_read_table(path).columns)
        sep = "\t" if path.suffix == ".tsv" else ","
        return list(pd.read_csv(path, sep=sep, nrows=0).columns)
    except Exception:
        return []


def _count_table_shape(path: Path, max_count_bytes: int = 750_000_000) -> tuple[int | float, int | float, str]:
    """Count table rows/columns without materializing large CSVs when possible."""

    if path.suffix not in {".csv", ".tsv", ".parquet", ".json"} and path.suffixes[-2:] != [".csv", ".gz"]:
        return (np.nan, np.nan, "")
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            return (len(df), len(df.columns), "")
        if path.suffix == ".json":
            df = _read_table(path)
            return (len(df), len(df.columns), "")
        cols = _read_header(path)
        if path.stat().st_size > max_count_bytes:
            return (np.nan, len(cols), "row_count_skipped_large_file")
        if path.suffixes[-2:] == [".csv", ".gz"]:
            import gzip

            with gzip.open(path, "rt", errors="ignore") as handle:
                n_lines = sum(1 for _ in handle)
        else:
            with path.open("rt", errors="ignore") as handle:
                n_lines = sum(1 for _ in handle)
        return (max(n_lines - 1, 0), len(cols), "")
    except Exception as exc:
        return (np.nan, np.nan, f"row_count_failed: {exc!r}")


def infer_biological_theme(stage0_panel_id: str, stage0_panel_type: str | None = None) -> str:
    """Map a Stage 0 panel id/type to a deterministic biological theme."""

    text = f"{stage0_panel_id or ''} {stage0_panel_type or ''}".lower()
    rules = [
        ("interferon", ("interferon", "ifn")),
        ("nfkb_tnf", ("nfkb", "nf_kb", "tnfa", "tnfr", "tnf")),
        ("antigen_presentation_mhc", ("mhc", "antigen", "hla", "allograft")),
        ("cytokine_jak_stat", ("cytokine", "jak_stat", "il2", "il6", "stat3", "stat5")),
        ("inflammatory_immune", ("inflammatory", "immune", "complement", "tcr", "bcr")),
        ("cell_cycle", ("g2m", "e2f", "cell_cycle", "mitotic", "mitosis")),
        ("metabolism", ("oxidative_phosphorylation", "mtorc", "metabolism", "glycolysis")),
        ("hypoxia_stress", ("hypoxia", "stress", "apoptosis", "unfolded_protein")),
        ("stemness_quiescence", ("stem", "quiescence", "dormancy", "self_renewal")),
        ("epigenetic_chromatin", ("epigen", "chromatin", "polycomb", "hdac", "methyl")),
        ("hvg_control", ("hvg",)),
        ("full_control", ("full_34", "full old", "full_control")),
        ("core_control", ("core",)),
    ]
    for theme, needles in rules:
        if any(needle in text for needle in needles):
            return theme
    return "other"


def shorten_panel_label(stage0_panel_id: str) -> str:
    """Short, readable label for plotting."""

    label = str(stage0_panel_id)
    for prefix in (
        "single_geneset__",
        "single_group__",
        "hvg_top_requested_",
        "reactome_",
        "hallmark_",
        "kegg_",
    ):
        label = re.sub(f"^{re.escape(prefix)}", "", label)
    label = re.sub(r"__available_\d+$", "", label)
    label = label.replace("_", " ")
    return re.sub(r"\s+", " ", label).strip()


def safe_filename(value: str, max_len: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text[:max_len] or "value"


def infer_candidate_role(path: Path, columns: Sequence[str] | None = None) -> str:
    """Infer an artifact role from path and lightweight column signatures."""

    p = path.as_posix().lower()
    name = path.name.lower()
    cols = {c.lower() for c in (columns or [])}
    has_goal = "modeling_goal" in cols
    has_coef = bool({"feature_id", "factor_id", "variable"} & cols) and bool({"coefficient", "coef"} & cols)
    if "stage2_provisional_shortlist" in name:
        return "stage1_metadata"
    if name.startswith("stage0_mrd") or "quick_groupkfold" in name or "metric_diagnostics" in p:
        return "unknown"
    if "stage1_dr" in p and path.name == "metadata.json":
        return "stage1_metadata"
    if "loadings" in p:
        return "stage1_loadings"
    if "scores" in p:
        return "stage1_scores"
    if "cell_prediction_fit_metadata" in p:
        return "cell_prediction_fit_metadata"
    if "cell_metadata" in p:
        return "cell_metadata"
    if "cell_prediction_matrix" in p or "cell_prediction_bundle" in p or p.endswith("/predictions.csv"):
        return "cell_prediction_matrix"
    if "by_heldout_patient" in p or "heldout_patient" in cols:
        return "lopo_by_patient_metrics"
    if has_coef:
        if "patient_specific" in p or "patient" in cols:
            return "patient_specific_coefficients"
        if "sharedness" in p or "lopo" in p or "leave_patient" in p:
            return "lopo_coefficients"
        if "discovery" in p:
            return "discovery_coefficients"
    if "stage2_discovery_full_cohort_scorecard" in p:
        return "discovery_metrics"
    if "stage2_sharedness_lopo_scorecard" in p:
        return "lopo_aggregate_metrics"
    if "stage2_patient_specific_scorecard" in p:
        return "patient_specific_metrics"
    if has_goal and "stage2_auprc" in cols:
        return "patient_specific_metrics" if "patient" in cols else "discovery_metrics"
    if "scorecard" in p or "metrics" in p:
        return "unknown"
    return "unknown"


def build_artifact_inventory(experiment_dir: Path, include_all_matches: bool = True) -> pd.DataFrame:
    """Recursively inventory likely Stage 2 analysis artifacts."""

    experiment_dir = Path(experiment_dir)
    paths: set[Path] = set()
    if include_all_matches:
        for pattern in ARTIFACT_PATTERNS:
            paths.update(experiment_dir.rglob(pattern))
    else:
        paths.update(experiment_dir.rglob("*stage2*"))
    rows = []
    for path in sorted(paths):
        if path.is_dir():
            continue
        rel = path.relative_to(experiment_dir)
        columns = _read_header(path)
        n_rows, n_cols, note = _count_table_shape(path)
        role = infer_candidate_role(path, columns)
        file_type = "".join(path.suffixes) if path.suffixes else "unknown"
        rows.append(
            {
                "artifact_name": path.name,
                "path": rel.as_posix(),
                "exists": path.exists(),
                "file_type": file_type,
                "n_rows_if_csv": n_rows,
                "n_cols_if_csv": n_cols,
                "candidate_role": role,
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def _preferred_artifact_path(inventory: pd.DataFrame, role: str, experiment_dir: Path) -> Path | None:
    subset = inventory.loc[inventory["candidate_role"].eq(role)].copy()
    if subset.empty:
        return None
    path_text = subset["path"].astype(str)
    canonical_scorecards = {
        "discovery_metrics": "analysis/scorecards/stage2_discovery_full_cohort_scorecard.csv",
        "lopo_aggregate_metrics": "analysis/scorecards/stage2_sharedness_lopo_scorecard.csv",
        "patient_specific_metrics": "analysis/scorecards/stage2_patient_specific_scorecard.csv",
    }
    subset["_rank"] = 100
    if role in canonical_scorecards:
        is_canonical = path_text.eq(canonical_scorecards[role])
        subset.loc[is_canonical, "_rank"] = 0
        subset.loc[path_text.str.startswith("analysis/scorecards/") & ~is_canonical, "_rank"] = 10
    else:
        subset.loc[path_text.str.startswith("analysis/scorecards/"), "_rank"] = 0
    subset.loc[path_text.str.startswith(f"stage2_supervised/multiobjective/{role.split('_')[0]}"), "_rank"] = 1
    subset.loc[path_text.str.startswith("stage2_supervised/multiobjective/discovery_full_cohort/"), "_rank"] = 1
    subset.loc[path_text.str.startswith("stage2_supervised/multiobjective/sharedness_lopo/"), "_rank"] = 1
    subset.loc[path_text.str.startswith("stage2_supervised/multiobjective/patient_specific/"), "_rank"] = 1
    subset.loc[path_text.str.contains("/merged/"), "_rank"] = subset.loc[path_text.str.contains("/merged/"), "_rank"].clip(upper=2)
    subset.loc[path_text.str.contains("/validation_"), "_rank"] = 50
    subset.loc[path_text.str.contains("/shards/"), "_rank"] = 20
    subset = subset.sort_values(["_rank", "path"])
    return experiment_dir / str(subset.iloc[0]["path"])


def _standardize_stage2_columns(df: pd.DataFrame, source_file: Path | None = None) -> pd.DataFrame:
    out = _ensure_unique_columns(df.copy(), context=f"stage2 table {source_file}" if source_file is not None else "stage2 table")
    if source_file is not None:
        out["source_file"] = str(source_file)
    if "seed" in out.columns and "stage1_seed" not in out.columns:
        out["stage1_seed"] = out["seed"]
    if "patient" in out.columns and "patient_id" not in out.columns:
        out["patient_id"] = out["patient"]
    if "heldout_patient" in out.columns and "heldout_patient_id" not in out.columns:
        out["heldout_patient_id"] = out["heldout_patient"]
    if "n_malignant" in out.columns and "n_test_malignant" not in out.columns:
        out["n_test_malignant"] = out["n_malignant"]
    if "n_normal" in out.columns and "n_test_non_malignant" not in out.columns:
        out["n_test_non_malignant"] = out["n_normal"]
    if "stage2_auprc" in out.columns and "auprc" not in out.columns:
        out["auprc"] = out["stage2_auprc"]
    if "stage2_auroc" in out.columns and "auroc" not in out.columns:
        out["auroc"] = out["stage2_auroc"]
    if "cell_weighted_auprc" in out.columns and "leave_patient_out_auprc_mean" not in out.columns:
        out["leave_patient_out_auprc_mean"] = out["cell_weighted_auprc"]
    if "patient_equal_auprc" in out.columns and "leave_patient_out_patient_equal_auprc_mean" not in out.columns:
        out["leave_patient_out_patient_equal_auprc_mean"] = out["patient_equal_auprc"]
    if "stage0_panel_id" in out.columns:
        stage0_type = out["stage0_panel_type"] if "stage0_panel_type" in out.columns else None
        out["biological_theme"] = [
            infer_biological_theme(panel, None if stage0_type is None else stage0_type.iloc[i])
            for i, panel in enumerate(out["stage0_panel_id"].astype(str))
        ]
        out["short_panel_label"] = out["stage0_panel_id"].astype(str).map(shorten_panel_label)
    out = add_stage2_ids(out)
    return out


def _attach_stage1_paths(df: pd.DataFrame, experiment_dir: Path) -> pd.DataFrame:
    if df.empty or "source_scorecard_row_id" not in df.columns:
        return df
    shortlist = experiment_dir / "analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv"
    if not shortlist.exists():
        return df
    try:
        meta = pd.read_csv(shortlist)
    except Exception as exc:
        warnings.warn(f"Could not load shortlist metadata: {exc!r}")
        return df
    keep = [
        "source_scorecard_row_id",
        "scores_path",
        "loadings_path",
        "features_path",
        "stage1_metadata_path",
        "feature_names_path",
        "gene_list_path",
    ]
    keep = [c for c in keep if c in meta.columns]
    if "source_scorecard_row_id" not in keep:
        return df
    meta = meta[keep].drop_duplicates("source_scorecard_row_id")
    meta = meta.rename(
        columns={
            "scores_path": "stage1_scores_path",
            "loadings_path": "stage1_loadings_path",
            "features_path": "stage1_features_path",
            "stage1_metadata_path": "stage1_metadata_path",
        }
    )
    out = df.merge(meta, on="source_scorecard_row_id", how="left", suffixes=("", "_from_shortlist"))
    return add_stage2_ids(out)


def add_stage2_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable representation/model identifiers when the required columns exist."""

    out = df.copy()
    required = ["stage0_panel_id", "representation_family", "stage1_method", "effective_k"]
    if all(c in out.columns for c in required):
        seed = out["stage1_seed"] if "stage1_seed" in out.columns else "seedNA"
        seed = pd.Series(seed, index=out.index).fillna("seedNA").astype(str)
        out["representation_id"] = (
            out["stage0_panel_id"].astype(str)
            + "|"
            + out["representation_family"].astype(str)
            + "|"
            + out["stage1_method"].astype(str)
            + "|"
            + out["effective_k"].astype(str)
            + "|"
            + seed
        )
    if "representation_id" in out.columns and "modeling_goal" in out.columns:
        penalty = out["penalty"].astype(str) if "penalty" in out.columns else "penaltyNA"
        c_value = out["C"].astype(str) if "C" in out.columns else "CNA"
        l1 = out["l1_ratio"].astype(str) if "l1_ratio" in out.columns else "l1NA"
        out["model_id"] = out["representation_id"].astype(str) + "|" + out["modeling_goal"].astype(str) + "|" + penalty + "|" + c_value + "|" + l1
        if "heldout_patient_id" in out.columns:
            out["model_id"] = out["model_id"] + "|heldout=" + out["heldout_patient_id"].astype(str)
        if "patient_id" in out.columns and out["modeling_goal"].astype(str).eq("patient_specific").any():
            out["model_id"] = out["model_id"] + "|patient=" + out["patient_id"].astype(str)
    return out


def load_stage2_artifacts(experiment_dir: Path) -> dict[str, pd.DataFrame | Path | None]:
    """Return canonical Stage 2 dataframes and paths for a multi-objective run."""

    experiment_dir = Path(experiment_dir)
    inventory = build_artifact_inventory(experiment_dir)
    result: dict[str, pd.DataFrame | Path | None] = {"artifact_inventory": inventory}
    role_to_key = {
        "discovery_metrics": "discovery_metrics",
        "discovery_coefficients": "discovery_coef",
        "lopo_aggregate_metrics": "lopo_metrics",
        "lopo_by_patient_metrics": "lopo_by_patient",
        "lopo_coefficients": "lopo_coef",
        "patient_specific_metrics": "patient_specific_metrics",
        "patient_specific_coefficients": "patient_specific_coef",
    }
    for role, key in role_to_key.items():
        path = _preferred_artifact_path(inventory, role, experiment_dir)
        result[f"{key}_path"] = path
        if path is None or not path.exists():
            result[key] = pd.DataFrame()
            continue
        try:
            df = _read_table(path)
            df = _standardize_stage2_columns(df, source_file=path.relative_to(experiment_dir))
            df = _attach_stage1_paths(df, experiment_dir)
            result[key] = df
        except Exception as exc:
            warnings.warn(f"Could not load {role} from {path}: {exc!r}")
            result[key] = pd.DataFrame()
    stage1 = _load_stage1_metadata_index(experiment_dir)
    result["stage1_metadata"] = stage1
    return result


def _load_stage1_metadata_index(experiment_dir: Path) -> pd.DataFrame:
    shortlist = experiment_dir / "analysis/scorecards/stage2_provisional_shortlist_from_quick_l2.csv"
    if shortlist.exists():
        df = pd.read_csv(shortlist)
        df = _standardize_stage2_columns(df, source_file=shortlist.relative_to(experiment_dir))
        if "scores_path" in df.columns and "stage1_scores_path" not in df.columns:
            df["stage1_scores_path"] = df["scores_path"]
        if "loadings_path" in df.columns and "stage1_loadings_path" not in df.columns:
            df["stage1_loadings_path"] = df["loadings_path"]
        return add_stage2_ids(df)
    rows = []
    for path in experiment_dir.rglob("stage1_dr/**/metadata.json"):
        try:
            with path.open() as handle:
                payload = json.load(handle)
        except Exception:
            payload = {}
        payload["stage1_metadata_path"] = path.relative_to(experiment_dir).as_posix()
        rows.append(payload)
    return pd.DataFrame(rows)


def print_artifact_summary(inventory: pd.DataFrame) -> None:
    """Print a compact, notebook-friendly artifact summary."""

    if inventory.empty:
        print("No candidate artifacts found.")
        return
    counts = inventory["candidate_role"].value_counts().sort_index()
    print("Artifact roles found:")
    for role, count in counts.items():
        print(f"  - {role}: {count}")
    available = set(inventory["candidate_role"])
    immediate = []
    missing = []
    if {"discovery_metrics", "lopo_aggregate_metrics"} <= available:
        immediate.append("Fig 3A discovery-vs-LOPO scatter")
    else:
        missing.append("Fig 3A requires discovery and LOPO aggregate metrics")
    if "lopo_by_patient_metrics" in available:
        immediate.append("Fig 3B LOPO per-patient heatmap")
    else:
        missing.append("Fig 3B requires LOPO by-heldout-patient metrics")
    if "lopo_coefficients" in available:
        immediate.append("Fig 3C LOPO coefficient stability")
    else:
        missing.append("Fig 3C requires LOPO fold coefficient paths")
    if {"patient_specific_coefficients", "discovery_coefficients"} <= available:
        immediate.append("Fig 3D discovery-vs-patient-specific factor usage")
    else:
        missing.append("Fig 3D requires discovery and patient-specific coefficients")
    print("\nPlots available now:")
    for item in immediate:
        print(f"  - {item}")
    print("\nMissing/optional inputs:")
    for item in missing:
        print(f"  - {item}")


def _metric_series(df: pd.DataFrame, metric_col: str, fallback: Sequence[str] = ()) -> tuple[str, pd.Series]:
    col = resolve_col(df, [metric_col, *fallback], required=True)
    return col, pd.to_numeric(df[col], errors="coerce")


def _penalty_rank(series: pd.Series, priority: Sequence[str]) -> pd.Series:
    ranks = {penalty: i for i, penalty in enumerate(priority)}
    return series.astype(str).str.lower().map(ranks).fillna(len(ranks)).astype(int)


def select_regularization_rows(
    metrics_df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str = "auprc",
    mode: str = "max_metric",
    penalty_priority: tuple[str, ...] = ("elasticnet", "l1", "l2"),
    min_nonzero: int | None = 1,
    max_nonzero: int | None = None,
    delta_mode: bool = False,
    zero_coef_policy: str = "exclude_all_zero",
    target_nonzero: int | None = None,
) -> pd.DataFrame:
    """Select one regularization row per group.

    Smaller sklearn ``C`` means stronger regularization.  Path-based modes sort
    explicitly by ``C`` so the direction of regularization is visible.
    """

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    df = _standardize_stage2_columns(metrics_df)
    metric_name, metric = _metric_series(
        df,
        metric_col,
        fallback=("stage2_auprc", "full_cohort_fit_auprc", "patient_equal_auprc", "cell_weighted_auprc"),
    )
    auroc_col = resolve_col(df, ["auroc", "stage2_auroc", "cell_weighted_auroc", "patient_equal_auroc"])
    nz_col = resolve_col(df, ["nonzero_coefficient_count", "selected_factor_count", "n_nonzero"])
    df["_selection_metric"] = metric
    df["_selection_auroc"] = pd.to_numeric(df[auroc_col], errors="coerce") if auroc_col else np.nan
    df["_selection_nonzero"] = pd.to_numeric(df[nz_col], errors="coerce") if nz_col else np.nan
    df["_penalty_rank"] = _penalty_rank(df["penalty"], penalty_priority) if "penalty" in df.columns else 0
    df["_C_numeric"] = pd.to_numeric(df["C"], errors="coerce") if "C" in df.columns else np.nan
    if mode.startswith("target_sparsity_") and target_nonzero is None:
        target_nonzero = int(mode.rsplit("_", 1)[-1])
        mode = "target_sparsity"
    filtered = df.copy()
    if zero_coef_policy == "exclude_all_zero" and min_nonzero is not None and nz_col:
        filtered = filtered.loc[filtered["_selection_nonzero"].fillna(min_nonzero) >= min_nonzero].copy()
    if max_nonzero is not None and nz_col:
        filtered = filtered.loc[filtered["_selection_nonzero"].fillna(max_nonzero) <= max_nonzero].copy()
    if filtered.empty:
        filtered = df.copy()

    selected_parts = []
    for _, group in filtered.groupby(group_cols, dropna=False):
        chosen = _select_one_group(
            group,
            mode=mode,
            penalty_priority=penalty_priority,
            target_nonzero=target_nonzero,
        )
        selected_parts.append(chosen.to_dict())
    if not selected_parts:
        return pd.DataFrame()
    selected = pd.DataFrame.from_records(selected_parts).reset_index(drop=True)
    selected["selection_mode"] = mode if target_nonzero is None else f"{mode}_{target_nonzero}"
    selected["selection_metric"] = metric_name
    selected["selected_metric_value"] = selected["_selection_metric"]
    if "model_id" in selected.columns:
        selected["source_model_id"] = selected["model_id"]
    keep_front = [
        "representation_id",
        "stage0_panel_id",
        "stage0_panel_type",
        "biological_theme",
        "short_panel_label",
        "representation_family",
        "stage1_method",
        "effective_k",
        "stage1_seed",
        "penalty",
        "C",
        "l1_ratio",
        "selection_mode",
        "selection_metric",
        "selected_metric_value",
        "nonzero_coefficient_count",
        "selected_factor_count",
        "source_model_id",
    ]
    ordered = [c for c in keep_front if c in selected.columns]
    remaining = [c for c in selected.columns if c not in ordered and not c.startswith("_selection") and c not in {"_penalty_rank", "_C_numeric"}]
    return selected[ordered + remaining]


def _select_one_group(
    group: pd.DataFrame,
    mode: str,
    penalty_priority: Sequence[str],
    target_nonzero: int | None,
) -> pd.Series:
    if mode == "last_before_all_zero":
        sparse = group.loc[group["penalty"].astype(str).str.lower().isin(["l1", "elasticnet"])].copy()
        if not sparse.empty and "_selection_nonzero" in sparse:
            sparse = sparse.sort_values(["_penalty_rank", "_C_numeric"], ascending=[True, False])
            positive = sparse.loc[sparse["_selection_nonzero"].fillna(0) > 0]
            if not positive.empty:
                # C decreases along the path as regularization strengthens; the
                # smallest C with nonzero coefficients is the last row before all-zero.
                candidates = positive.sort_values(["_penalty_rank", "_C_numeric"], ascending=[True, True])
                return candidates.iloc[0]
    if mode == "largest_delta_metric":
        best = None
        best_delta = -np.inf
        for _, path_df in group.groupby("penalty", dropna=False):
            path_df = path_df.sort_values("_C_numeric", ascending=True)
            metric = path_df["_selection_metric"].astype(float)
            delta = metric.diff()
            if delta.notna().any():
                idx = delta.idxmax()
                if float(delta.loc[idx]) > best_delta:
                    best_delta = float(delta.loc[idx])
                    best = path_df.loc[idx]
        if best is not None and np.isfinite(best_delta) and best_delta > 0:
            return best
    if mode == "target_sparsity" and target_nonzero is not None and "_selection_nonzero" in group:
        target_df = group.copy()
        target_df["_target_distance"] = (target_df["_selection_nonzero"].fillna(np.inf) - target_nonzero).abs()
        return (
            target_df.sort_values(
                ["_target_distance", "_selection_metric", "_selection_auroc", "_penalty_rank", "_C_numeric"],
                ascending=[True, False, False, True, True],
            )
            .iloc[0]
            .drop(labels=["_target_distance"], errors="ignore")
        )
    return group.sort_values(
        ["_selection_metric", "_selection_auroc", "_selection_nonzero", "_penalty_rank", "_C_numeric"],
        ascending=[False, False, True, True, True],
    ).iloc[0]


def choose_interpretable_row_near_best(
    df: pd.DataFrame,
    best_metric_col: str,
    tolerance: float = 0.02,
    prefer_penalties: tuple[str, ...] = ("elasticnet", "l1"),
    max_nonzero_fraction: float = 0.5,
) -> pd.Series:
    """Choose a sparse/interpretable model within tolerance of the best row."""

    if df.empty:
        return pd.Series(dtype=object)
    metric_col = resolve_col(df, [best_metric_col, "stage2_auprc", "auprc"], required=True)
    nz_col = resolve_col(df, ["nonzero_coefficient_count", "selected_factor_count"])
    k_col = resolve_col(df, ["effective_k", "n_features"])
    tmp = df.copy()
    tmp["_metric"] = pd.to_numeric(tmp[metric_col], errors="coerce")
    best_value = tmp["_metric"].max()
    near = tmp.loc[tmp["_metric"] >= best_value - tolerance].copy()
    if near.empty:
        near = tmp.copy()
    if nz_col and k_col:
        frac = pd.to_numeric(near[nz_col], errors="coerce") / pd.to_numeric(near[k_col], errors="coerce").replace(0, np.nan)
        sparse = near.loc[frac <= max_nonzero_fraction].copy()
        if not sparse.empty:
            near = sparse
    if "penalty" in near.columns:
        near["_penalty_rank"] = _penalty_rank(near["penalty"], prefer_penalties)
    else:
        near["_penalty_rank"] = 0
    if nz_col:
        near["_nonzero"] = pd.to_numeric(near[nz_col], errors="coerce")
    else:
        near["_nonzero"] = np.nan
    return near.sort_values(["_penalty_rank", "_nonzero", "_metric"], ascending=[True, True, False]).iloc[0]


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix:
        stem = output_path.with_suffix("")
    else:
        stem = output_path
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_discovery_vs_lopo_sharedness(
    discovery_selected_df: pd.DataFrame,
    lopo_metrics_df: pd.DataFrame,
    *,
    x_metric: str = "full_cohort_fit_auprc",
    y_metric: str = "leave_patient_out_auprc_mean",
    y_metric_preference: tuple[str, ...] = (
        "leave_patient_out_patient_equal_auprc_mean",
        "patient_equal_auprc_mean",
        "leave_patient_out_auprc_mean",
        "cell_weighted_auprc",
        "heldout_auprc_mean",
    ),
    color_by: str | None = "biological_theme",
    shape_by: str | None = None,
    size_by: str | None = None,
    label_top_n: int = 10,
    label_include_patterns: list[str] | None = None,
    facet_by: str | None = None,
    title: str | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Plot discovery apparent performance against heldout-patient LOPO performance."""

    _plt, _sns = _require_plotting()
    if discovery_selected_df.empty or lopo_metrics_df.empty:
        warnings.warn("Missing discovery or LOPO metrics; Fig 3A cannot be plotted.")
        return pd.DataFrame()
    discovery = _standardize_stage2_columns(discovery_selected_df)
    lopo = _standardize_stage2_columns(lopo_metrics_df)
    x_col = resolve_col(discovery, [x_metric, "stage2_auprc", "auprc"], required=True)
    y_col = resolve_col(lopo, [y_metric, *y_metric_preference], required=True)
    lopo_cols = ["representation_id", y_col]
    for col in ["penalty", "C", "l1_ratio", "cell_weighted_auprc", "patient_equal_auprc", "source_file"]:
        if col in lopo.columns and col not in lopo_cols:
            lopo_cols.append(col)
    rows = []
    for _, row in discovery.iterrows():
        candidates = lopo.loc[lopo["representation_id"].eq(row["representation_id"])].copy()
        if candidates.empty:
            continue
        same = candidates.copy()
        for col in ["penalty", "C", "l1_ratio"]:
            if col in same.columns and col in row.index:
                same = same.loc[same[col].astype(str).eq(str(row[col]))]
        if not same.empty:
            chosen = same.iloc[0]
            strategy = "same_penalty_C_l1_ratio"
        else:
            candidates["_y_metric"] = pd.to_numeric(candidates[y_col], errors="coerce")
            chosen = candidates.sort_values("_y_metric", ascending=False).iloc[0]
            strategy = "representation_level_best_available_lopo"
        combined = row.to_dict()
        for col in lopo_cols:
            combined[f"lopo_{col}" if col in combined and col != "representation_id" else col] = chosen.get(col, np.nan)
        combined["lopo_join_strategy"] = strategy
        rows.append(combined)
    plotted = pd.DataFrame(rows)
    if plotted.empty:
        warnings.warn("No overlapping representation_id values between discovery and LOPO metrics.")
        return plotted
    plotted["discovery_plot_metric"] = pd.to_numeric(plotted[x_col], errors="coerce")
    plotted["lopo_plot_metric"] = pd.to_numeric(plotted[y_col], errors="coerce")
    plotted["discovery_minus_lopo"] = plotted["discovery_plot_metric"] - plotted["lopo_plot_metric"]

    with _sns.plotting_context("notebook", rc=PLOT_CONTEXT):
        if facet_by and facet_by in plotted.columns:
            grid = _sns.relplot(
                data=plotted,
                x="discovery_plot_metric",
                y="lopo_plot_metric",
                hue=color_by if color_by in plotted.columns else None,
                style=shape_by if shape_by and shape_by in plotted.columns else None,
                size=size_by if size_by and size_by in plotted.columns else None,
                col=facet_by,
                kind="scatter",
                height=4,
                aspect=1,
            )
            fig = grid.fig
            axes = grid.axes.flat
        else:
            fig, ax = _plt.subplots(figsize=(6.5, 5.5))
            axes = [ax]
            _sns.scatterplot(
                data=plotted,
                x="discovery_plot_metric",
                y="lopo_plot_metric",
                hue=color_by if color_by in plotted.columns else None,
                style=shape_by if shape_by and shape_by in plotted.columns else None,
                size=size_by if size_by and size_by in plotted.columns else None,
                s=70,
                edgecolor="white",
                linewidth=0.5,
                ax=ax,
            )
        for ax in axes:
            min_value = np.nanmin([plotted["discovery_plot_metric"].min(), plotted["lopo_plot_metric"].min()])
            max_value = np.nanmax([plotted["discovery_plot_metric"].max(), plotted["lopo_plot_metric"].max()])
            ax.plot([min_value, max_value], [min_value, max_value], color="0.5", linestyle="--", linewidth=1)
            ax.axhline(plotted["lopo_plot_metric"].median(), color="0.85", linewidth=0.8)
            ax.axvline(plotted["discovery_plot_metric"].median(), color="0.85", linewidth=0.8)
            ax.set_xlabel(f"Discovery apparent AUPRC ({x_col})")
            ax.set_ylabel(f"LOPO sharedness AUPRC ({y_col})")
            ax.set_title(title or "Figure 3A: Discovery vs LOPO Sharedness")
        label_df = plotted.sort_values(["lopo_plot_metric", "discovery_minus_lopo"], ascending=[False, False]).head(label_top_n)
        if label_include_patterns:
            pattern = re.compile("|".join(label_include_patterns), flags=re.IGNORECASE)
            label_df = pd.concat(
                [label_df, plotted.loc[plotted["stage0_panel_id"].astype(str).str.contains(pattern, na=False)]],
                ignore_index=True,
            ).drop_duplicates("representation_id")
        if len(axes) == 1:
            ax = axes[0]
            for _, row in label_df.iterrows():
                ax.text(
                    row["discovery_plot_metric"],
                    row["lopo_plot_metric"],
                    str(row.get("short_panel_label", row.get("stage0_panel_id", "")))[:35],
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )
        fig.tight_layout()
        _save_figure(fig, output_path)
    return plotted


def plot_lopo_patient_heatmap(
    lopo_by_patient_df: pd.DataFrame,
    selected_representations_df: pd.DataFrame | None = None,
    *,
    value_col_preference: tuple[str, ...] = ("heldout_auprc", "lopo_auprc", "auprc"),
    row_label_col: str = "short_panel_label",
    col_label_col: str = "heldout_patient_id",
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    annotate_support: bool = True,
    low_support_threshold: int = 10,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Plot heldout-patient LOPO AUPRC by selected representation."""

    _plt, _sns = _require_plotting()
    if lopo_by_patient_df.empty:
        warnings.warn("Missing LOPO by-patient metrics; Fig 3B cannot be plotted.")
        return pd.DataFrame()
    lopo = _standardize_stage2_columns(lopo_by_patient_df)
    if selected_representations_df is not None and not selected_representations_df.empty:
        selected_ids = set(selected_representations_df["representation_id"].astype(str))
        lopo = lopo.loc[lopo["representation_id"].astype(str).isin(selected_ids)].copy()
        reg_parts: list[pd.DataFrame] = []
        for _, sel in selected_representations_df.iterrows():
            rid = str(sel.get("representation_id", ""))
            if not rid:
                continue
            sub = lopo.loc[lopo["representation_id"].astype(str).eq(rid)].copy()
            reg_map = {
                "penalty": sel.get("lopo_penalty", sel.get("penalty", np.nan)),
                "C": sel.get("lopo_C", sel.get("C", np.nan)),
                "l1_ratio": sel.get("lopo_l1_ratio", sel.get("l1_ratio", np.nan)),
            }
            for reg_col, reg_val in reg_map.items():
                if reg_col in sub.columns and pd.notna(reg_val):
                    sub = sub.loc[sub[reg_col].astype(str).eq(str(reg_val))]
            if not sub.empty:
                reg_parts.append(sub)
        if reg_parts:
            lopo = pd.concat(reg_parts, ignore_index=True)
    value_col = resolve_col(lopo, list(value_col_preference) + ["stage2_auprc"], required=True)
    if col_label_col not in lopo.columns:
        col_label_col = resolve_col(lopo, ["heldout_patient", "patient", "patient_id"], required=True)
    if row_label_col not in lopo.columns:
        row_label_col = "representation_id"
    lopo["heatmap_row_label"] = lopo[row_label_col].astype(str)
    duplicated = lopo.groupby("heatmap_row_label")["representation_id"].transform("nunique") > 1
    lopo.loc[duplicated, "heatmap_row_label"] = (
        lopo.loc[duplicated, "heatmap_row_label"].astype(str)
        + " | "
        + lopo.loc[duplicated, "stage1_method"].astype(str)
        + " k="
        + lopo.loc[duplicated, "effective_k"].astype(str)
    )
    lopo["heatmap_value"] = pd.to_numeric(lopo[value_col], errors="coerce")
    mal_col = resolve_col(lopo, ["n_test_malignant", "n_malignant"])
    norm_col = resolve_col(lopo, ["n_test_non_malignant", "n_normal"])
    if mal_col and norm_col:
        malignant = pd.to_numeric(lopo[mal_col], errors="coerce")
        normal = pd.to_numeric(lopo[norm_col], errors="coerce")
        prevalence = malignant / (malignant + normal).replace(0, np.nan)
        lopo["malignant_prevalence"] = prevalence
        lopo["auprc_enrichment"] = lopo["heatmap_value"] / prevalence
        lopo["normal_only"] = malignant.fillna(0).eq(0) & normal.fillna(0).gt(0)
        lopo["low_malignant_support"] = malignant.gt(0) & malignant.lt(low_support_threshold)
    matrix = lopo.pivot_table(index="heatmap_row_label", columns=col_label_col, values="heatmap_value", aggfunc="mean")
    matrix = _cluster_matrix(matrix, axis=0) if cluster_rows else matrix
    matrix = _cluster_matrix(matrix, axis=1) if cluster_cols else matrix
    height = max(4, min(18, 0.28 * len(matrix.index) + 1.8))
    width = max(6, min(16, 0.42 * len(matrix.columns) + 4))
    with _sns.plotting_context("notebook", rc=PLOT_CONTEXT):
        fig, ax = _plt.subplots(figsize=(width, height))
        _sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=max(1.0, np.nanmax(matrix.values)), linewidths=0.2, linecolor="white", ax=ax)
        ax.set_xlabel("Heldout patient")
        ax.set_ylabel("Stage 0/1 representation")
        ax.set_title("Figure 3B: LOPO per-patient AUPRC")
        if annotate_support and {"low_malignant_support", "normal_only"} & set(lopo.columns):
            support_lookup = lopo.set_index(["heatmap_row_label", col_label_col])
            for y, row_label in enumerate(matrix.index):
                for x, patient in enumerate(matrix.columns):
                    mark = ""
                    if (row_label, patient) in support_lookup.index:
                        rec = support_lookup.loc[(row_label, patient)]
                        if isinstance(rec, pd.DataFrame):
                            rec = rec.iloc[0]
                        if bool(rec.get("normal_only", False)):
                            mark = "N"
                        elif bool(rec.get("low_malignant_support", False)):
                            mark = "*"
                    if mark:
                        ax.text(x + 0.5, y + 0.5, mark, ha="center", va="center", color="white", fontsize=7, fontweight="bold")
        fig.tight_layout()
        _save_figure(fig, output_path)
    return lopo


def _cluster_matrix(matrix: pd.DataFrame, axis: int) -> pd.DataFrame:
    if matrix.shape[axis] <= 2 or leaves_list is None or linkage is None or pdist is None:
        return matrix
    filled = matrix.fillna(matrix.mean().mean())
    values = filled.values if axis == 0 else filled.values.T
    try:
        order = leaves_list(linkage(pdist(values), method="average"))
    except Exception:
        return matrix
    return matrix.iloc[order, :] if axis == 0 else matrix.iloc[:, order]


def plot_lopo_coefficient_stability(
    lopo_coef_df: pd.DataFrame,
    representation_id: str,
    *,
    coef_col: str | None = None,
    feature_col: str | None = None,
    fold_col: str | None = None,
    selected_regularization_row: pd.Series | None = None,
    top_n_features: int = 30,
    min_selection_frequency: float = 0.0,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Plot LOPO coefficient stability within one fixed Stage 0/1 basis."""

    if lopo_coef_df is None or lopo_coef_df.empty:
        warnings.warn("No LOPO coefficient table was found; skipping Fig 3C.")
        return pd.DataFrame()
    df = _standardize_stage2_columns(lopo_coef_df)
    df = df.loc[df["representation_id"].astype(str).eq(str(representation_id))].copy()
    if df.empty:
        warnings.warn(f"No LOPO coefficients for representation_id={representation_id}")
        return pd.DataFrame()
    if selected_regularization_row is not None and not selected_regularization_row.empty:
        for col in ["penalty", "C", "l1_ratio"]:
            if col in df.columns and col in selected_regularization_row.index:
                df = df.loc[df[col].astype(str).eq(str(selected_regularization_row[col]))].copy()
    coef_col = coef_col or resolve_col(df, ["coefficient", "coef"], required=True)
    feature_col = feature_col or resolve_col(df, ["feature_id", "factor_id", "variable"], required=True)
    fold_col = fold_col or resolve_col(df, ["heldout_patient_id", "heldout_patient", "fold_id", "patient_id"], required=True)
    df["_coef"] = pd.to_numeric(df[coef_col], errors="coerce").fillna(0.0)
    matrix = df.pivot_table(index=feature_col, columns=fold_col, values="_coef", aggfunc="mean").fillna(0.0)
    summary = _coefficient_stability_summary(matrix).reset_index().rename(columns={feature_col: "feature_id", "index": "feature_id"})
    summary["representation_id"] = representation_id
    summary = summary.loc[summary["selection_frequency"] >= min_selection_frequency].copy()
    top_features = summary.head(top_n_features)["feature_id"].astype(str)
    matrix = matrix.loc[matrix.index.astype(str).isin(set(top_features))]
    matrix = matrix.loc[top_features]
    _plot_coef_heatmap(matrix, f"Figure 3C: LOPO coefficient stability\n{representation_id}", output_path)
    return summary


def _coefficient_stability_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    nonzero = matrix.ne(0)
    signs = np.sign(matrix.where(nonzero))
    summary = pd.DataFrame(index=matrix.index)
    summary["selection_frequency"] = nonzero.mean(axis=1)
    summary["sign_stability"] = signs.mean(axis=1, skipna=True).abs().fillna(0)
    summary["mean_abs_coef"] = matrix.abs().mean(axis=1)
    summary["mean_coef"] = matrix.mean(axis=1)
    summary["n_folds_selected"] = nonzero.sum(axis=1)
    summary["n_folds_total"] = matrix.shape[1]
    return summary.sort_values(["selection_frequency", "sign_stability", "mean_abs_coef"], ascending=[False, False, False])


def plot_patient_specific_factor_usage(
    patient_coef_df: pd.DataFrame,
    discovery_coef_df: pd.DataFrame,
    representation_id: str,
    *,
    selected_discovery_row: pd.Series | None = None,
    patient_selected_df: pd.DataFrame | None = None,
    lopo_feature_summary_df: pd.DataFrame | None = None,
    top_n_features: int = 40,
    patient_col: str = "patient_id",
    coef_col: str | None = None,
    feature_col: str | None = None,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare patient-specific factor usage with discovery coefficients."""

    if patient_coef_df.empty or discovery_coef_df.empty:
        warnings.warn("Missing patient-specific or discovery coefficients; Fig 3D cannot be plotted.")
        return pd.DataFrame(), pd.DataFrame()
    patient = _standardize_stage2_columns(patient_coef_df)
    discovery = _standardize_stage2_columns(discovery_coef_df)
    patient = patient.loc[patient["representation_id"].astype(str).eq(str(representation_id))].copy()
    discovery = discovery.loc[discovery["representation_id"].astype(str).eq(str(representation_id))].copy()
    if patient.empty or discovery.empty:
        warnings.warn(f"Missing coefficients for representation_id={representation_id}")
        return pd.DataFrame(), pd.DataFrame()
    if selected_discovery_row is not None and not selected_discovery_row.empty:
        discovery = _filter_like_row(discovery, selected_discovery_row, ["penalty", "C", "l1_ratio"])
    if patient_selected_df is not None and not patient_selected_df.empty:
        selected = _standardize_stage2_columns(patient_selected_df)
        selected = selected.loc[selected["representation_id"].astype(str).eq(str(representation_id))]
        patient = patient.merge(
            selected[["patient_id", "penalty", "C", "l1_ratio"]].drop_duplicates(),
            on=["patient_id", "penalty", "C", "l1_ratio"],
            how="inner",
        )
    coef_col = coef_col or resolve_col(patient, ["coefficient", "coef"], required=True)
    feature_col = feature_col or resolve_col(patient, ["feature_id", "factor_id", "variable"], required=True)
    if patient_col not in patient.columns:
        patient_col = resolve_col(patient, ["patient_id", "patient"], required=True)
    patient["_coef"] = pd.to_numeric(patient[coef_col], errors="coerce").fillna(0.0)
    discovery["_coef"] = pd.to_numeric(discovery[coef_col], errors="coerce").fillna(0.0)
    discovery_by_feature = discovery.groupby(feature_col)["_coef"].mean()
    if selected_discovery_row is not None and str(selected_discovery_row.get("penalty", "")).lower() == "l2":
        discovery_selected = set(discovery_by_feature.abs().sort_values(ascending=False).head(top_n_features).index.astype(str))
        discovery_selection_rule = "l2_top_abs_coef"
    else:
        discovery_selected = set(discovery_by_feature.loc[discovery_by_feature.ne(0)].index.astype(str))
        discovery_selection_rule = "coef_nonzero"
    patient_matrix = patient.pivot_table(index=patient_col, columns=feature_col, values="_coef", aggfunc="mean").fillna(0.0)
    recurrent = patient_matrix.ne(0).mean(axis=0).sort_values(ascending=False)
    feature_scores = pd.DataFrame(
        {
            "patient_selection_frequency": recurrent,
            "discovery_abs_coef": discovery_by_feature.abs().reindex(recurrent.index).fillna(0),
        }
    )
    feature_scores["is_discovery_selected"] = feature_scores.index.astype(str).isin(discovery_selected)
    feature_scores = feature_scores.sort_values(
        ["is_discovery_selected", "patient_selection_frequency", "discovery_abs_coef"],
        ascending=[False, False, False],
    )
    top_features = feature_scores.head(top_n_features).index
    matrix = patient_matrix.reindex(columns=top_features).fillna(0.0)
    overlap_rows = []
    for patient_id, vals in patient_matrix.iterrows():
        patient_selected = set(vals.index[vals.ne(0)].astype(str))
        union = patient_selected | discovery_selected
        inter = patient_selected & discovery_selected
        overlap_rows.append(
            {
                "representation_id": representation_id,
                "patient_id": patient_id,
                "jaccard_with_discovery": len(inter) / len(union) if union else np.nan,
                "overlap_count": len(inter),
                "patient_selected_count": len(patient_selected),
                "discovery_selected_count": len(discovery_selected),
                "discovery_selection_rule": discovery_selection_rule,
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    _plot_coef_heatmap(matrix, f"Figure 3D: Patient-specific factor usage\n{representation_id}", output_path)
    matrix_out = matrix.reset_index()
    matrix_out["representation_id"] = representation_id
    if lopo_feature_summary_df is not None and not lopo_feature_summary_df.empty:
        pass
    return matrix_out, overlap


def _filter_like_row(df: pd.DataFrame, row: pd.Series, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and col in row.index:
            out = out.loc[out[col].astype(str).eq(str(row[col]))].copy()
    return out


def _plot_coef_heatmap(matrix: pd.DataFrame, title: str, output_path: Path | None) -> None:
    if matrix.empty:
        return
    _plt, _sns = _require_plotting()
    height = max(3.5, min(14, 0.28 * len(matrix.index) + 1.8))
    width = max(6.5, min(18, 0.28 * len(matrix.columns) + 3))
    vmax = np.nanpercentile(np.abs(matrix.values), 98) if matrix.size else 1
    vmax = max(float(vmax), 1e-9)
    with _sns.plotting_context("notebook", rc=PLOT_CONTEXT):
        fig, ax = _plt.subplots(figsize=(width, height))
        _sns.heatmap(matrix, cmap="vlag", center=0, vmin=-vmax, vmax=vmax, linewidths=0.2, linecolor="white", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Factor / feature")
        ax.set_ylabel("Fold / patient")
        fig.tight_layout()
        _save_figure(fig, output_path)


def select_representations_for_coefficient_heatmaps(
    fig3a_source_df: pd.DataFrame,
    *,
    n_top_shared: int = 5,
    n_top_patient_specific_gap: int = 5,
    force_include_patterns: Sequence[str] = ("interferon", "ifn", "cytokine", "tnfa", "nfkb", "antigen", "mhc", "full_34", "core", "hvg"),
) -> pd.DataFrame:
    """Choose a compact set of representations for coefficient heatmaps."""

    if fig3a_source_df.empty:
        return pd.DataFrame()
    df = fig3a_source_df.copy()
    rows = []
    top_shared = df.sort_values(["lopo_plot_metric", "discovery_plot_metric"], ascending=False).head(n_top_shared)
    for _, row in top_shared.iterrows():
        rows.append({**row.to_dict(), "selection_reason": "top_high_discovery_high_lopo"})
    gap = df.sort_values("discovery_minus_lopo", ascending=False).head(n_top_patient_specific_gap)
    for _, row in gap.iterrows():
        rows.append({**row.to_dict(), "selection_reason": "high_discovery_low_lopo_gap"})
    pattern = re.compile("|".join(force_include_patterns), re.IGNORECASE)
    forced = df.loc[df["stage0_panel_id"].astype(str).str.contains(pattern, na=False)]
    for _, row in forced.iterrows():
        rows.append({**row.to_dict(), "selection_reason": "force_include_interest_or_control"})
    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected
    reason = selected.groupby("representation_id")["selection_reason"].agg(lambda x: ";".join(sorted(set(x))))
    selected = selected.drop_duplicates("representation_id").drop(columns=["selection_reason"]).merge(reason, on="representation_id")
    out_cols = [
        "representation_id",
        "stage0_panel_id",
        "short_panel_label",
        "biological_theme",
        "stage1_method",
        "effective_k",
        "discovery_plot_metric",
        "lopo_plot_metric",
        "discovery_minus_lopo",
        "selection_reason",
    ]
    out = selected[[c for c in out_cols if c in selected.columns]].copy()
    return out.rename(columns={"discovery_plot_metric": "discovery_auprc", "lopo_plot_metric": "lopo_auprc"})


def write_markdown_report(
    path: Path,
    *,
    artifact_inventory: pd.DataFrame,
    selection_mode: str,
    lopo_metric: str,
    fig3a_source: pd.DataFrame,
    selected_representations: pd.DataFrame,
    missing_caveats: Sequence[str],
    figure_paths: Sequence[Path],
) -> None:
    """Write a short, portable markdown report for Figure 3 outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    top_shared = fig3a_source.sort_values("lopo_plot_metric", ascending=False).head(10) if not fig3a_source.empty else pd.DataFrame()
    high_gap = fig3a_source.sort_values("discovery_minus_lopo", ascending=False).head(10) if not fig3a_source.empty else pd.DataFrame()
    lines = [
        "# Stage 2 Figure 3 Sharedness Visualization Report",
        "",
        "## Inputs",
        "",
        f"- Candidate artifacts inventoried: {len(artifact_inventory)}",
        f"- Selection mode: `{selection_mode}`",
        f"- LOPO metric used: `{lopo_metric}`",
        "",
        "## Top Shared Candidates",
        "",
        _markdown_candidate_list(top_shared, "lopo_plot_metric"),
        "",
        "## High Discovery / Low LOPO Candidates",
        "",
        _markdown_candidate_list(high_gap, "discovery_minus_lopo"),
        "",
        "## Coefficient Heatmap Representations",
        "",
        _markdown_candidate_list(selected_representations, "lopo_auprc"),
        "",
        "## Caveats",
        "",
    ]
    lines.extend([f"- {item}" for item in missing_caveats] or ["- No caveats recorded."])
    lines.extend(["", "## Figure Files", ""])
    lines.extend([f"- `{p}`" for p in figure_paths])
    path.write_text("\n".join(lines) + "\n")


def _markdown_candidate_list(df: pd.DataFrame, sort_col: str) -> str:
    if df.empty:
        return "- None available."
    rows = []
    for _, row in df.head(10).iterrows():
        label = row.get("short_panel_label", row.get("stage0_panel_id", row.get("representation_id", "")))
        value = row.get(sort_col, np.nan)
        rows.append(f"- {label}: {sort_col}={value:.4g}" if isinstance(value, (int, float, np.floating)) else f"- {label}")
    return "\n".join(rows)

