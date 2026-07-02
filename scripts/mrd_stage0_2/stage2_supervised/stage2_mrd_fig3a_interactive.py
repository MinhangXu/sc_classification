#!/usr/bin/env python
"""Interactive Discovery-vs-LOPO sharedness plot for Stage 2 MRD Figure 3A."""

from __future__ import annotations

import argparse
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stage2_sharedness_plotting import _standardize_stage2_columns, resolve_col


EXPERIMENT_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parents[3] / "experiments" / EXPERIMENT_ID
DEFAULT_SOURCE_REL = Path("analysis/scorecards/stage2_figure3_sharedness/fig3A_discovery_vs_lopo_sharedness_source.csv")
DEFAULT_DISCOVERY_REL = Path("analysis/scorecards/stage2_discovery_full_cohort_scorecard.csv")
DEFAULT_LOPO_REL = Path("analysis/scorecards/stage2_sharedness_lopo_scorecard.csv")
DEFAULT_LOPO_BY_PATIENT_REL = Path("stage2_supervised/multiobjective/sharedness_lopo/by_heldout_patient.csv")
DEFAULT_PATIENT_SPECIFIC_REL = Path("analysis/scorecards/stage2_patient_specific_scorecard.csv")
DEFAULT_DISCOVERY_COEF_REL = Path("stage2_supervised/multiobjective/discovery_full_cohort/coefficient_paths.csv")
DEFAULT_LOPO_COEF_REL = Path("stage2_supervised/multiobjective/sharedness_lopo/coefficient_paths.csv")
DEFAULT_PATIENT_SPECIFIC_COEF_REL = Path("stage2_supervised/multiobjective/patient_specific/coefficient_paths.csv")
DEFAULT_HTML_REL = Path("analysis/figures/stage2_figure3_sharedness/fig3A_discovery_vs_lopo_sharedness_interactive.html")
REG_PATH_TOP_K = 12

PANEL_TYPE_ORDER = [
    "hvg_anchor_control",
    "core_only",
    "full_control",
    "single_geneset_only",
    "single_group_only",
]
PANEL_TYPE_COLORS = {
    "hvg_anchor_control": "#4C78A8",
    "core_only": "#F58518",
    "full_control": "#54A24B",
    "single_geneset_only": "#B279A2",
    "single_group_only": "#E45756",
}
PANEL_TYPE_NOTES = {
    "full_control": "Full union of the old 34-program dictionary.",
    "core_only": "Union of old-34 manifest entries marked priority=Core.",
    "hvg_anchor_control": "Top-N highly variable genes from the shared MRD/CITE gene universe.",
    "single_geneset_only": "One old-34 gene set in isolation.",
    "single_group_only": "Union of old-34 gene sets within one biology-group label.",
}
MODE_LABELS = {
    "matched_regularization": "Matched discovery regularization",
    "best_lopo": "Best LOPO regularization",
    "best_patient_specific": "Best patient-specific apparent regularization",
}
REG_METRIC_COLORS = {
    "discovery": "#1f77b4",
    "lopo": "#d62728",
    "patient_specific": "#c51b7d",
}
ELASTICNET_DASHES = {
    "0.1": "dash",
    "0.5": "dot",
    "0.9": "dashdot",
}
REG_METRIC_OPTIONS = {
    "auprc": {
        "label": "AUPRC",
        "discovery": ["stage2_auprc", "full_cohort_fit_auprc", "auprc"],
        "lopo": ["leave_patient_out_auprc_mean", "cell_weighted_auprc", "patient_equal_auprc", "stage2_auprc", "auprc"],
        "patient_specific": ["stage2_auprc", "auprc"],
    },
    "auroc": {
        "label": "AUROC",
        "discovery": ["stage2_auroc", "full_cohort_fit_auroc", "auroc"],
        "lopo": ["cell_weighted_auroc", "patient_equal_auroc", "stage2_auroc", "auroc"],
        "patient_specific": ["stage2_auroc", "auroc"],
    },
    "balanced_accuracy": {
        "label": "Balanced accuracy",
        "discovery": ["stage2_balanced_accuracy", "balanced_accuracy"],
        "lopo": ["cell_weighted_balanced_accuracy", "patient_equal_balanced_accuracy", "stage2_balanced_accuracy", "balanced_accuracy"],
        "patient_specific": ["stage2_balanced_accuracy", "balanced_accuracy"],
    },
    "specificity": {
        "label": "Specificity",
        "discovery": ["stage2_healthy_recall_specificity", "specificity"],
        "lopo": ["cell_weighted_healthy_recall_specificity", "patient_equal_healthy_recall_specificity", "stage2_healthy_recall_specificity", "specificity"],
        "patient_specific": ["stage2_healthy_recall_specificity", "specificity"],
    },
    "precision": {
        "label": "Malignant precision",
        "discovery": ["stage2_malignant_precision", "malignant_precision"],
        "lopo": ["cell_weighted_malignant_precision", "patient_equal_malignant_precision", "stage2_malignant_precision", "malignant_precision"],
        "patient_specific": ["stage2_malignant_precision", "malignant_precision"],
    },
    "recall": {
        "label": "Malignant recall",
        "discovery": ["stage2_malignant_recall", "malignant_recall"],
        "lopo": ["cell_weighted_malignant_recall", "patient_equal_malignant_recall", "stage2_malignant_recall", "malignant_recall"],
        "patient_specific": ["stage2_malignant_recall", "malignant_recall"],
    },
    "f1": {
        "label": "F1",
        "discovery": ["stage2_f1", "f1"],
        "lopo": ["cell_weighted_f1", "stage2_f1", "f1"],
        "patient_specific": ["stage2_f1", "f1"],
    },
    "log_loss": {
        "label": "Log loss",
        "discovery": ["stage2_log_loss", "log_loss"],
        "lopo": ["cell_weighted_log_loss", "stage2_log_loss", "log_loss"],
        "patient_specific": ["stage2_log_loss", "log_loss"],
    },
}
CUSTOM_DATA = [
    "row_key",
    "lopo_target_label",
    "stage0_panel_id",
    "stage0_panel_type",
    "geneset_name",
    "n_covered_genes_display",
    "stage1_method",
    "effective_k_display",
    "discovery_reg_display",
    "lopo_reg_display",
    "lopo_join_strategy",
    "discovery_minus_lopo_display",
    "patient_support_display",
    "specificity_display",
    "genes_preview",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch an interactive Dash version of Figure 3A. Each point is one "
            "shortlisted Stage 0 panel + Stage 1 method/K/seed representation."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--source-table", type=Path, default=None, help="Fig 3A source CSV. Defaults under experiment-dir.")
    parser.add_argument("--discovery-scorecard", type=Path, default=None, help="Full-grid discovery scorecard CSV.")
    parser.add_argument("--lopo-scorecard", type=Path, default=None, help="Full-grid LOPO aggregate scorecard CSV.")
    parser.add_argument("--lopo-by-patient", type=Path, default=None, help="LOPO by-heldout-patient scorecard CSV.")
    parser.add_argument("--patient-specific-scorecard", type=Path, default=None, help="Patient-specific apparent scorecard CSV.")
    parser.add_argument("--discovery-coefficients", type=Path, default=None, help="Discovery coefficient path CSV.")
    parser.add_argument("--lopo-coefficients", type=Path, default=None, help="LOPO fold-level coefficient path CSV.")
    parser.add_argument("--patient-specific-coefficients", type=Path, default=None, help="Patient-specific coefficient path CSV.")
    parser.add_argument(
        "--lopo-metric",
        default="leave_patient_out_auprc_mean",
        help=(
            "LOPO metric for the y-axis. The default matches the current notebook "
            "helper and resolves to the cell-weighted aggregate when present."
        ),
    )
    parser.add_argument(
        "--comparison-mode",
        choices=["matched_regularization", "best_lopo", "both"],
        default="both",
        help="Initial view shown in the app and used for HTML export.",
    )
    parser.add_argument(
        "--lopo-target",
        default="aggregate_mean",
        help="Initial LOPO target: aggregate_mean or a heldout patient id such as P05.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the Dash app URL in your default browser after the server starts.",
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        default=None,
        help="Optional standalone Plotly HTML export path. Defaults under experiment-dir when --no-server is used.",
    )
    parser.add_argument("--no-server", action="store_true", help="Build data/export HTML, then exit without launching Dash.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    source_table = _resolve_path(args.source_table, experiment_dir, DEFAULT_SOURCE_REL)
    discovery_scorecard = _resolve_path(args.discovery_scorecard, experiment_dir, DEFAULT_DISCOVERY_REL)
    lopo_scorecard = _resolve_path(args.lopo_scorecard, experiment_dir, DEFAULT_LOPO_REL)
    lopo_by_patient = _resolve_path(args.lopo_by_patient, experiment_dir, DEFAULT_LOPO_BY_PATIENT_REL)
    patient_specific_scorecard = _resolve_path(args.patient_specific_scorecard, experiment_dir, DEFAULT_PATIENT_SPECIFIC_REL)
    discovery_coefficients = _resolve_path(args.discovery_coefficients, experiment_dir, DEFAULT_DISCOVERY_COEF_REL)
    lopo_coefficients = _resolve_path(args.lopo_coefficients, experiment_dir, DEFAULT_LOPO_COEF_REL)
    patient_specific_coefficients = _resolve_path(args.patient_specific_coefficients, experiment_dir, DEFAULT_PATIENT_SPECIFIC_COEF_REL)

    plot_data, y_col, y_label = build_plot_data(
        experiment_dir=experiment_dir,
        source_table=source_table,
        lopo_scorecard=lopo_scorecard,
        lopo_by_patient=lopo_by_patient,
        patient_specific_scorecard=patient_specific_scorecard,
        lopo_metric=args.lopo_metric,
    )
    print(f"Loaded {plot_data['representation_id'].nunique()} representations")
    print(f"Matched-reg rows: {(plot_data['comparison_mode'] == 'matched_regularization').sum()}")
    print(f"Best-LOPO rows: {(plot_data['comparison_mode'] == 'best_lopo').sum()}")
    print(f"LOPO targets: {', '.join(target_options(plot_data).keys())}")
    print(f"LOPO metric: {y_col}")

    if args.export_html is not None or args.no_server:
        html_path = args.export_html or experiment_dir / DEFAULT_HTML_REL
        html_path = html_path if html_path.is_absolute() else Path.cwd() / html_path
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig = make_figure(filter_mode(plot_data, args.comparison_mode, args.lopo_target), y_label=y_label)
        fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Wrote {html_path}")

    if args.no_server:
        return

    reg_data = LazyRegData(
        discovery_scorecard=discovery_scorecard,
        lopo_scorecard=lopo_scorecard,
        lopo_by_patient=lopo_by_patient,
        patient_specific_scorecard=patient_specific_scorecard,
        discovery_coefficients=discovery_coefficients,
        lopo_coefficients=lopo_coefficients,
        patient_specific_coefficients=patient_specific_coefficients,
        lopo_metric_col=y_col,
    )
    app = build_app(
        plot_data,
        y_label=y_label,
        initial_mode=args.comparison_mode,
        initial_target=args.lopo_target,
        reg_data=reg_data,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Starting Dash server at {url}")
    print("Open that URL in your browser if a window does not appear automatically.")
    print("Regularization-path tables load on demand when you click a point (large CSVs may take a few minutes).")
    if args.open_browser:
        webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=args.debug)


def _resolve_path(path: Path | None, experiment_dir: Path, default_rel: Path) -> Path:
    if path is None:
        return experiment_dir / default_rel
    return path if path.is_absolute() else Path.cwd() / path


def build_plot_data(
    *,
    experiment_dir: Path,
    source_table: Path,
    lopo_scorecard: Path,
    lopo_by_patient: Path | None,
    patient_specific_scorecard: Path | None,
    lopo_metric: str,
) -> tuple[pd.DataFrame, str, str]:
    if not source_table.exists():
        raise FileNotFoundError(
            f"Missing Fig 3A source table: {source_table}. Run the Figure 3 notebook first."
        )
    if not lopo_scorecard.exists():
        raise FileNotFoundError(f"Missing LOPO scorecard: {lopo_scorecard}")

    matched = pd.read_csv(source_table, low_memory=False)
    matched = prepare_hover_fields(matched, experiment_dir=experiment_dir)
    matched["comparison_mode"] = "matched_regularization"
    matched["comparison_mode_label"] = MODE_LABELS["matched_regularization"]
    matched["lopo_target"] = "aggregate_mean"
    matched["lopo_target_label"] = "LOPO mean across held-out patients"
    matched["row_key"] = matched["representation_id"].astype(str) + "||aggregate_mean||matched_regularization"

    lopo = pd.read_csv(lopo_scorecard, low_memory=False)
    lopo = _standardize_stage2_columns(lopo)
    y_col = resolve_col(
        lopo,
        [
            lopo_metric,
            "leave_patient_out_auprc_mean",
            "cell_weighted_auprc",
            "patient_equal_auprc",
            "heldout_auprc_mean",
            "stage2_auprc",
            "auprc",
        ],
        required=True,
    )
    best_lopo = build_best_lopo_rows(matched, lopo, y_col=y_col)
    best_lopo = prepare_hover_fields(best_lopo, experiment_dir=experiment_dir)
    best_lopo["comparison_mode"] = "best_lopo"
    best_lopo["comparison_mode_label"] = MODE_LABELS["best_lopo"]
    best_lopo["lopo_target"] = "aggregate_mean"
    best_lopo["lopo_target_label"] = "LOPO mean across held-out patients"
    best_lopo["row_key"] = best_lopo["representation_id"].astype(str) + "||aggregate_mean||best_lopo"

    parts = [matched, best_lopo]
    per_patient = pd.DataFrame()
    if lopo_by_patient is not None and lopo_by_patient.exists():
        per_patient = pd.read_csv(lopo_by_patient, low_memory=False)
        per_patient = _standardize_stage2_columns(per_patient)
        patient_rows = build_patient_lopo_rows(matched, per_patient)
        if not patient_rows.empty:
            patient_rows = prepare_hover_fields(patient_rows, experiment_dir=experiment_dir)
            parts.append(patient_rows)

    patient_specific = pd.DataFrame()
    if patient_specific_scorecard is not None and patient_specific_scorecard.exists():
        patient_specific = _standardize_stage2_columns(pd.read_csv(patient_specific_scorecard, low_memory=False))
        best_patient_specific = build_best_patient_specific_rows(matched, patient_specific)
        if not best_patient_specific.empty:
            best_patient_specific = prepare_hover_fields(best_patient_specific, experiment_dir=experiment_dir)
            best_patient_specific["lopo_reg_display"] = "not used"
            parts.append(best_patient_specific)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    combined = attach_aggregate_cell_counts(combined, per_patient)
    if not patient_specific.empty:
        combined = add_patient_specific_apparent_metrics(combined, patient_specific)
    else:
        combined["patient_specific_plot_metric"] = np.nan
        combined["patient_specific_join_strategy"] = ""
        combined["patient_specific_reg_display"] = ""
    combined["stage0_panel_type"] = combined["stage0_panel_type"].fillna("unknown")
    combined["discovery_plot_metric"] = pd.to_numeric(combined["discovery_plot_metric"], errors="coerce")
    combined["lopo_plot_metric"] = pd.to_numeric(combined["lopo_plot_metric"], errors="coerce")
    combined["patient_specific_plot_metric"] = pd.to_numeric(combined["patient_specific_plot_metric"], errors="coerce")
    combined["discovery_minus_lopo"] = combined["discovery_plot_metric"] - combined["lopo_plot_metric"]
    combined["discovery_minus_lopo_display"] = combined["discovery_minus_lopo"].map(lambda x: _fmt_number(x, digits=3))
    y_label = metric_label(y_col)
    return combined, y_col, y_label


def build_regularization_data(
    *,
    discovery_scorecard: Path,
    lopo_scorecard: Path,
    lopo_by_patient: Path | None,
    patient_specific_scorecard: Path | None,
    discovery_coefficients: Path,
    lopo_coefficients: Path,
    patient_specific_coefficients: Path,
    lopo_metric_col: str,
) -> dict[str, Any]:
    loader = LazyRegData(
        discovery_scorecard=discovery_scorecard,
        lopo_scorecard=lopo_scorecard,
        lopo_by_patient=lopo_by_patient,
        patient_specific_scorecard=patient_specific_scorecard,
        discovery_coefficients=discovery_coefficients,
        lopo_coefficients=lopo_coefficients,
        patient_specific_coefficients=patient_specific_coefficients,
        lopo_metric_col=lopo_metric_col,
    )
    return {key: loader.get(key) for key in loader.table_keys} | {"lopo_metric_col": lopo_metric_col}


class LazyRegData:
    """Defer loading multi-GB Stage 2 tables until the reg-path panel needs them."""

    _TABLE_PATHS: dict[str, str] = {
        "discovery_metrics": "discovery_scorecard",
        "lopo_metrics": "lopo_scorecard",
        "lopo_by_patient": "lopo_by_patient",
        "patient_specific_metrics": "patient_specific_scorecard",
        "discovery_coef": "discovery_coefficients",
        "lopo_coef": "lopo_coefficients",
        "patient_specific_coef": "patient_specific_coefficients",
    }

    def __init__(
        self,
        *,
        discovery_scorecard: Path,
        lopo_scorecard: Path,
        lopo_by_patient: Path | None,
        patient_specific_scorecard: Path | None,
        discovery_coefficients: Path,
        lopo_coefficients: Path,
        patient_specific_coefficients: Path,
        lopo_metric_col: str,
    ) -> None:
        self._paths = {
            "discovery_scorecard": discovery_scorecard,
            "lopo_scorecard": lopo_scorecard,
            "lopo_by_patient": lopo_by_patient,
            "patient_specific_scorecard": patient_specific_scorecard,
            "discovery_coefficients": discovery_coefficients,
            "lopo_coefficients": lopo_coefficients,
            "patient_specific_coefficients": patient_specific_coefficients,
        }
        self._lopo_metric_col = lopo_metric_col
        self._cache: dict[str, pd.DataFrame] = {}
        self.table_keys = list(self._TABLE_PATHS)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "lopo_metric_col":
            return self._lopo_metric_col
        path_attr = self._TABLE_PATHS.get(key)
        if path_attr is None:
            return default
        if key not in self._cache:
            path = self._paths[path_attr]
            label = path.name if path is not None else key
            print(f"Loading {key} from {label} ...", flush=True)
            if key.endswith("_coef"):
                self._cache[key] = _read_coefficient_table(path)
            else:
                self._cache[key] = _read_optional_stage2_table(path)
            rows = len(self._cache[key])
            print(f"Loaded {key}: {rows:,} rows", flush=True)
        return self._cache[key]


def _read_coefficient_table(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        return _standardize_stage2_columns(pd.read_csv(path, low_memory=False))
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc!r}")
        return pd.DataFrame()


def _read_optional_stage2_table(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    try:
        return _standardize_stage2_columns(pd.read_csv(path, low_memory=False))
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc!r}")
        return pd.DataFrame()


def build_best_lopo_rows(source: pd.DataFrame, lopo: pd.DataFrame, *, y_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        representation_id = str(row["representation_id"])
        candidates = lopo.loc[lopo["representation_id"].astype(str).eq(representation_id)].copy()
        if candidates.empty:
            continue
        candidates["_lopo_metric"] = pd.to_numeric(candidates[y_col], errors="coerce")
        auroc_col = resolve_col(candidates, ["stage2_auroc", "auroc", "cell_weighted_auroc", "patient_equal_auroc"])
        nz_col = resolve_col(candidates, ["nonzero_coefficient_count", "selected_factor_count", "n_nonzero"])
        candidates["_tie_auroc"] = pd.to_numeric(candidates[auroc_col], errors="coerce") if auroc_col else np.nan
        candidates["_tie_nonzero"] = pd.to_numeric(candidates[nz_col], errors="coerce") if nz_col else np.nan
        candidates["_penalty_rank"] = candidates.get("penalty", pd.Series("", index=candidates.index)).map(
            {"elasticnet": 0, "l1": 1, "l2": 2}
        ).fillna(3)
        chosen = candidates.sort_values(
            ["_lopo_metric", "_tie_auroc", "_tie_nonzero", "_penalty_rank"],
            ascending=[False, False, True, True],
        ).iloc[0]

        combined = row.to_dict()
        combined["lopo_plot_metric"] = chosen["_lopo_metric"]
        combined["lopo_penalty"] = chosen.get("penalty", np.nan)
        combined["lopo_C"] = chosen.get("C", np.nan)
        combined["lopo_l1_ratio"] = chosen.get("l1_ratio", np.nan)
        combined["lopo_join_strategy"] = "best_lopo_for_representation"
        combined["cell_weighted_auprc"] = chosen.get("cell_weighted_auprc", np.nan)
        combined["patient_equal_auprc"] = chosen.get("patient_equal_auprc", np.nan)
        combined["leave_patient_out_auprc_mean"] = chosen.get("leave_patient_out_auprc_mean", np.nan)
        combined["discovery_minus_lopo"] = pd.to_numeric(combined["discovery_plot_metric"], errors="coerce") - chosen["_lopo_metric"]
        rows.append(combined)
    return pd.DataFrame(rows)


def build_patient_lopo_rows(source: pd.DataFrame, per_patient: pd.DataFrame) -> pd.DataFrame:
    y_col = resolve_col(per_patient, ["stage2_auprc", "heldout_auprc", "lopo_auprc", "auprc"], required=True)
    patient_col = resolve_col(per_patient, ["heldout_patient_id", "heldout_patient", "patient"], required=True)
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        representation_id = str(row["representation_id"])
        rep = per_patient.loc[per_patient["representation_id"].astype(str).eq(representation_id)].copy()
        if rep.empty:
            continue
        for patient_id, patient_df in rep.groupby(patient_col, dropna=False):
            patient_id = str(patient_id)
            same = filter_same_regularization(patient_df, row)
            if not same.empty:
                chosen = same.iloc[0]
                rows.append(patient_row(row, chosen, y_col=y_col, patient_id=patient_id, mode="matched_regularization"))
            else:
                chosen = choose_best_patient_candidate(patient_df, y_col=y_col)
                rows.append(
                    patient_row(
                        row,
                        chosen,
                        y_col=y_col,
                        patient_id=patient_id,
                        mode="matched_regularization",
                        strategy="heldout_patient_best_available_lopo_fallback",
                    )
                )

            best = choose_best_patient_candidate(patient_df, y_col=y_col)
            rows.append(patient_row(row, best, y_col=y_col, patient_id=patient_id, mode="best_lopo"))
    return pd.DataFrame(rows)


def attach_aggregate_cell_counts(plot_data: pd.DataFrame, per_patient: pd.DataFrame) -> pd.DataFrame:
    out = plot_data.copy()
    if per_patient is None or per_patient.empty or "representation_id" not in per_patient.columns:
        return out
    counts = aggregate_counts_by_representation(per_patient)
    if counts.empty:
        return out
    for count_col in ["n_malignant", "n_normal"]:
        if count_col not in out.columns:
            out[count_col] = np.nan
    aggregate_mask = out["lopo_target"].astype(str).eq("aggregate_mean")
    for idx, row in out.loc[aggregate_mask].iterrows():
        rid = str(row.get("representation_id", ""))
        if rid not in counts.index:
            continue
        out.at[idx, "n_malignant"] = counts.at[rid, "n_malignant"]
        out.at[idx, "n_normal"] = counts.at[rid, "n_normal"]
    return out


def aggregate_counts_by_representation(per_patient: pd.DataFrame) -> pd.DataFrame:
    patient_col = resolve_col(per_patient, ["heldout_patient_id", "heldout_patient", "patient"])
    malignant_col, healthy_col = resolve_cell_count_columns(per_patient)
    if patient_col is None or malignant_col is None or healthy_col is None:
        return pd.DataFrame()
    sub = per_patient[["representation_id", patient_col, malignant_col, healthy_col]].copy()
    sub["_malignant"] = pd.to_numeric(sub[malignant_col], errors="coerce")
    sub["_healthy"] = pd.to_numeric(sub[healthy_col], errors="coerce")
    per_patient_counts = (
        sub.groupby(["representation_id", patient_col], dropna=False)
        .agg(n_malignant=("_malignant", "max"), n_normal=("_healthy", "max"))
        .reset_index()
    )
    return per_patient_counts.groupby("representation_id", dropna=False).agg(
        n_malignant=("n_malignant", "sum"),
        n_normal=("n_normal", "sum"),
    )


def build_best_patient_specific_rows(source: pd.DataFrame, patient_specific: pd.DataFrame) -> pd.DataFrame:
    y_col = resolve_col(patient_specific, ["stage2_auprc", "auprc"])
    patient_col = resolve_col(patient_specific, ["patient_id", "patient"])
    if y_col is None or patient_col is None or "representation_id" not in patient_specific.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        rid = str(row.get("representation_id", ""))
        candidates = patient_specific.loc[patient_specific["representation_id"].astype(str).eq(rid)].copy()
        if candidates.empty:
            continue

        aggregate_metric, aggregate_reg_display = aggregate_best_patient_specific_apparent(
            candidates,
            y_col=y_col,
            patient_col=patient_col,
        )
        if not _is_missing(aggregate_metric):
            combined = row.to_dict()
            combined["comparison_mode"] = "best_patient_specific"
            combined["comparison_mode_label"] = MODE_LABELS["best_patient_specific"]
            combined["lopo_target"] = "aggregate_mean"
            combined["lopo_target_label"] = "Patient-specific apparent mean across patients"
            combined["row_key"] = f"{combined['representation_id']}||aggregate_mean||best_patient_specific"
            combined["lopo_plot_metric"] = np.nan
            combined["lopo_join_strategy"] = ""
            combined["lopo_penalty"] = np.nan
            combined["lopo_C"] = np.nan
            combined["lopo_l1_ratio"] = np.nan
            combined["patient_specific_plot_metric"] = aggregate_metric
            combined["patient_specific_join_strategy"] = "best_patient_specific_cell_weighted_mean"
            combined["patient_specific_reg_display"] = aggregate_reg_display
            rows.append(combined)

        for patient_id, patient_df in candidates.groupby(patient_col, dropna=False):
            patient_id = str(patient_id)
            chosen = choose_best_patient_candidate(patient_df, y_col=y_col)
            combined = row.to_dict()
            combined["comparison_mode"] = "best_patient_specific"
            combined["comparison_mode_label"] = MODE_LABELS["best_patient_specific"]
            combined["lopo_target"] = patient_id
            combined["lopo_target_label"] = f"Patient-specific apparent {patient_id}"
            combined["row_key"] = f"{combined['representation_id']}||{patient_id}||best_patient_specific"
            combined["lopo_plot_metric"] = np.nan
            combined["lopo_join_strategy"] = ""
            combined["lopo_penalty"] = np.nan
            combined["lopo_C"] = np.nan
            combined["lopo_l1_ratio"] = np.nan
            combined["patient_specific_plot_metric"] = pd.to_numeric(chosen.get(y_col), errors="coerce")
            combined["patient_specific_join_strategy"] = "best_patient_specific_apparent"
            combined["patient_specific_reg_display"] = format_reg(chosen.get("penalty"), chosen.get("C"), chosen.get("l1_ratio"))
            for col in [
                "patient",
                "patient_id",
                "n_test_malignant",
                "n_test_non_malignant",
                "n_malignant",
                "n_normal",
                "low_malignant_support",
                "stage2_auroc",
                "stage2_healthy_recall_specificity",
                "stage2_predicted_malignant_total",
            ]:
                if col in chosen.index:
                    combined[col] = chosen.get(col, np.nan)
            rows.append(combined)
    return pd.DataFrame(rows)


def aggregate_best_patient_specific_apparent(
    candidates: pd.DataFrame,
    *,
    y_col: str,
    patient_col: str,
) -> tuple[float, str]:
    chosen_rows: list[pd.Series] = []
    for _, patient_df in candidates.groupby(patient_col, dropna=False):
        chosen_rows.append(choose_best_patient_candidate(patient_df, y_col=y_col))
    if not chosen_rows:
        return np.nan, ""
    chosen_df = pd.DataFrame(chosen_rows)
    values = pd.to_numeric(chosen_df[y_col], errors="coerce")
    malignant_col, healthy_col = resolve_cell_count_columns(chosen_df)
    if malignant_col is not None and healthy_col is not None:
        weights = pd.to_numeric(chosen_df[malignant_col], errors="coerce").fillna(0) + pd.to_numeric(chosen_df[healthy_col], errors="coerce").fillna(0)
        valid = values.notna() & weights.gt(0)
        metric = float(np.average(values.loc[valid], weights=weights.loc[valid])) if valid.any() else float(values.mean())
    else:
        metric = float(values.mean())
    return metric, "per-patient best apparent"


def add_patient_specific_apparent_metrics(plot_data: pd.DataFrame, patient_specific: pd.DataFrame) -> pd.DataFrame:
    out = plot_data.copy()
    if "patient_specific_plot_metric" not in out.columns:
        out["patient_specific_plot_metric"] = np.nan
    if "patient_specific_join_strategy" not in out.columns:
        out["patient_specific_join_strategy"] = ""
    if "patient_specific_reg_display" not in out.columns:
        out["patient_specific_reg_display"] = ""
    if patient_specific is None or patient_specific.empty or "representation_id" not in patient_specific.columns:
        return out
    y_col = resolve_col(patient_specific, ["stage2_auprc", "auprc"])
    patient_col = resolve_col(patient_specific, ["patient_id", "patient"])
    if y_col is None or patient_col is None:
        return out

    patient_specific = patient_specific.copy()
    patient_specific["_patient_specific_metric"] = pd.to_numeric(patient_specific[y_col], errors="coerce")
    for idx, row in out.iterrows():
        if not _is_missing(row.get("patient_specific_plot_metric")):
            continue
        rid = str(row.get("representation_id", ""))
        candidates = patient_specific.loc[patient_specific["representation_id"].astype(str).eq(rid)].copy()
        if candidates.empty:
            continue
        preference = patient_specific_regularization_preference(row)
        target = str(row.get("lopo_target", "aggregate_mean"))
        if target == "aggregate_mean":
            metric, strategy, reg_display = aggregate_patient_specific_apparent(
                candidates,
                row,
                y_col=y_col,
                patient_col=patient_col,
                regularization_preference=preference,
            )
        else:
            patient_candidates = candidates.loc[candidates[patient_col].astype(str).eq(target)].copy()
            if patient_candidates.empty:
                continue
            chosen, strategy = choose_same_regularization_or_best_patient_specific(
                patient_candidates,
                row,
                y_col=y_col,
                regularization_preference=preference,
            )
            metric = pd.to_numeric(chosen.get(y_col), errors="coerce")
            reg_display = format_reg(chosen.get("penalty"), chosen.get("C"), chosen.get("l1_ratio"))
        out.at[idx, "patient_specific_plot_metric"] = metric
        out.at[idx, "patient_specific_join_strategy"] = strategy
        out.at[idx, "patient_specific_reg_display"] = reg_display
    return out


def aggregate_patient_specific_apparent(
    candidates: pd.DataFrame,
    row: pd.Series,
    *,
    y_col: str,
    patient_col: str,
    regularization_preference: str,
) -> tuple[float, str, str]:
    chosen_rows: list[pd.Series] = []
    strategies: list[str] = []
    for _, patient_df in candidates.groupby(patient_col, dropna=False):
        chosen, strategy = choose_same_regularization_or_best_patient_specific(
            patient_df,
            row,
            y_col=y_col,
            regularization_preference=regularization_preference,
        )
        chosen_rows.append(chosen)
        strategies.append(strategy)
    if not chosen_rows:
        return np.nan, "", ""
    chosen_df = pd.DataFrame(chosen_rows)
    values = pd.to_numeric(chosen_df[y_col], errors="coerce")
    malignant_col, healthy_col = resolve_cell_count_columns(chosen_df)
    if malignant_col is not None and healthy_col is not None:
        weights = pd.to_numeric(chosen_df[malignant_col], errors="coerce").fillna(0) + pd.to_numeric(chosen_df[healthy_col], errors="coerce").fillna(0)
        valid = values.notna() & weights.gt(0)
        metric = float(np.average(values.loc[valid], weights=weights.loc[valid])) if valid.any() else float(values.mean())
    else:
        metric = float(values.mean())
    strategy = "patient_specific_cell_weighted_mean"
    if any(item.endswith("fallback") for item in strategies):
        strategy += "_with_best_available_fallback"
    reg_display = format_selected_regularization(row, regularization_preference=regularization_preference)
    return metric, strategy, reg_display


def choose_same_regularization_or_best_patient_specific(
    candidates: pd.DataFrame,
    row: pd.Series,
    *,
    y_col: str,
    regularization_preference: str,
) -> tuple[pd.Series, str]:
    same = filter_same_regularization(candidates, row, regularization_preference=regularization_preference)
    if not same.empty:
        return same.iloc[0], f"patient_specific_same_{regularization_preference}_penalty_C_l1_ratio"
    return choose_best_patient_candidate(candidates, y_col=y_col), "patient_specific_best_available_fallback"


def patient_specific_regularization_preference(row: pd.Series) -> str:
    return "discovery" if str(row.get("comparison_mode", "")) == "matched_regularization" else "lopo"


def format_selected_regularization(row: pd.Series, *, regularization_preference: str) -> str:
    penalty, c_value, l1_ratio = regularization_values(row, regularization_preference=regularization_preference)
    return format_reg(penalty, c_value, l1_ratio)


def regularization_values(row: pd.Series, *, regularization_preference: str) -> tuple[Any, Any, Any]:
    if regularization_preference == "discovery":
        return row.get("penalty", np.nan), row.get("C", np.nan), row.get("l1_ratio", np.nan)
    return (
        first_present(row.get("lopo_penalty"), row.get("penalty")),
        first_present(row.get("lopo_C"), row.get("C")),
        first_present(row.get("lopo_l1_ratio"), row.get("l1_ratio")),
    )


def filter_same_regularization(
    candidates: pd.DataFrame,
    source_row: pd.Series,
    *,
    regularization_preference: str = "lopo",
) -> pd.DataFrame:
    out = candidates.copy()
    penalty, c_value, l1_ratio = regularization_values(source_row, regularization_preference=regularization_preference)
    reg_pairs = [
        ("penalty", penalty),
        ("C", c_value),
        ("l1_ratio", l1_ratio),
    ]
    for col, value in reg_pairs:
        if col not in out.columns:
            continue
        if col == "C":
            numeric = pd.to_numeric(out[col], errors="coerce")
            try:
                value_numeric = float(value)
            except Exception:
                out = out.loc[out[col].astype(str).eq(str(value))]
            else:
                out = out.loc[np.isclose(numeric, value_numeric, rtol=1e-8, atol=1e-12)]
        elif _is_missing(value):
            out = out.loc[out[col].isna() | out[col].astype(str).str.lower().isin({"", "nan", "none"})]
        else:
            out = out.loc[out[col].astype(str).eq(str(value))]
    return out


def choose_best_patient_candidate(candidates: pd.DataFrame, *, y_col: str) -> pd.Series:
    tmp = candidates.copy()
    tmp["_lopo_metric"] = pd.to_numeric(tmp[y_col], errors="coerce")
    auroc_col = resolve_col(tmp, ["stage2_auroc", "auroc"])
    specificity_col = resolve_col(tmp, ["stage2_healthy_recall_specificity", "specificity"])
    predicted_col = resolve_col(tmp, ["stage2_predicted_malignant_total"])
    nz_col = resolve_col(tmp, ["nonzero_coefficient_count", "selected_factor_count", "n_nonzero"])
    tmp["_tie_auroc"] = pd.to_numeric(tmp[auroc_col], errors="coerce") if auroc_col else np.nan
    tmp["_tie_specificity"] = pd.to_numeric(tmp[specificity_col], errors="coerce") if specificity_col else np.nan
    tmp["_tie_predicted_malignant"] = pd.to_numeric(tmp[predicted_col], errors="coerce") if predicted_col else np.nan
    tmp["_tie_nonzero"] = pd.to_numeric(tmp[nz_col], errors="coerce") if nz_col else np.nan
    tmp["_penalty_rank"] = tmp.get("penalty", pd.Series("", index=tmp.index)).map(
        {"elasticnet": 0, "l1": 1, "l2": 2}
    ).fillna(3)
    return tmp.sort_values(
        ["_lopo_metric", "_tie_auroc", "_tie_specificity", "_tie_predicted_malignant", "_tie_nonzero", "_penalty_rank"],
        ascending=[False, False, False, True, True, True],
        na_position="last",
    ).iloc[0]


def patient_row(
    source_row: pd.Series,
    chosen: pd.Series,
    *,
    y_col: str,
    patient_id: str,
    mode: str,
    strategy: str | None = None,
) -> dict[str, Any]:
    combined = source_row.to_dict()
    combined["lopo_plot_metric"] = pd.to_numeric(chosen.get(y_col), errors="coerce")
    combined["lopo_penalty"] = chosen.get("penalty", np.nan)
    combined["lopo_C"] = chosen.get("C", np.nan)
    combined["lopo_l1_ratio"] = chosen.get("l1_ratio", np.nan)
    combined["lopo_join_strategy"] = strategy or (
        "heldout_patient_same_penalty_C_l1_ratio" if mode == "matched_regularization" else "heldout_patient_best_lopo"
    )
    combined["comparison_mode"] = mode
    combined["comparison_mode_label"] = MODE_LABELS[mode]
    combined["lopo_target"] = patient_id
    combined["lopo_target_label"] = f"Held-out patient {patient_id}"
    combined["row_key"] = f"{combined['representation_id']}||{patient_id}||{mode}"
    for col in [
        "heldout_patient_id",
        "heldout_patient",
        "patient",
        "n_test_malignant",
        "n_test_non_malignant",
        "n_malignant",
        "n_normal",
        "has_both_classes",
        "normal_only",
        "low_malignant_support",
        "stage2_auroc",
        "stage2_healthy_recall_specificity",
        "stage2_predicted_malignant_total",
    ]:
        if col in chosen.index:
            combined[col] = chosen.get(col, np.nan)
    return combined


def prepare_hover_fields(df: pd.DataFrame, *, experiment_dir: Path) -> pd.DataFrame:
    out = df.copy()
    if "lopo_target" not in out.columns:
        out["lopo_target"] = "aggregate_mean"
    if "lopo_target_label" not in out.columns:
        out["lopo_target_label"] = "LOPO mean across held-out patients"
    out["short_panel_label"] = out.get("short_panel_label", out["stage0_panel_id"]).fillna(out["stage0_panel_id"])
    out["geneset_name"] = [
        single_geneset_name(panel_id, panel_type)
        for panel_id, panel_type in zip(out["stage0_panel_id"], out.get("stage0_panel_type", pd.Series("", index=out.index)))
    ]
    gene_summaries = [
        gene_summary(row, experiment_dir=experiment_dir)
        for _, row in out.iterrows()
    ]
    out["n_covered_genes_display"] = [item["n_genes"] for item in gene_summaries]
    out["genes_preview"] = [item["preview"] for item in gene_summaries]
    out["genesets_preview"] = [item["genesets_preview"] for item in gene_summaries]
    out["stage0_panel_type_note"] = out["stage0_panel_type"].map(lambda x: PANEL_TYPE_NOTES.get(str(x), ""))
    out["effective_k_display"] = out.get("effective_k", pd.Series(np.nan, index=out.index)).map(lambda x: _fmt_number(x, digits=0))
    out["discovery_reg_display"] = [
        format_reg(row.get("penalty"), row.get("C"), row.get("l1_ratio"))
        for _, row in out.iterrows()
    ]
    out["lopo_reg_display"] = [
        format_reg(
            first_present(row.get("lopo_penalty"), row.get("penalty")),
            first_present(row.get("lopo_C"), row.get("C")),
            first_present(row.get("lopo_l1_ratio"), row.get("l1_ratio")),
        )
        for _, row in out.iterrows()
    ]
    out["patient_support_display"] = [patient_support_display(row) for _, row in out.iterrows()]
    out["specificity_display"] = [
        _fmt_number(row.get("stage2_healthy_recall_specificity"), digits=3)
        for _, row in out.iterrows()
    ]
    return out


def patient_support_display(row: pd.Series) -> str:
    malignant, healthy = cell_count_values_from_row(row)
    if str(row.get("lopo_target", "aggregate_mean")) == "aggregate_mean":
        if not _is_missing(malignant) or not _is_missing(healthy):
            return format_cell_count_display(malignant, healthy, prefix="total")
        return "aggregate across held-out patients"
    parts = [
        f"held-out malignant={_fmt_number(malignant, digits=0) or 'NA'}",
        f"held-out healthy={_fmt_number(healthy, digits=0) or 'NA'}",
    ]
    if _is_true(row.get("normal_only", False)):
        parts.append("normal-only: AUPRC/AUROC undefined")
    elif _is_true(row.get("low_malignant_support", False)):
        parts.append("low malignant support")
    return "; ".join(parts)


def resolve_cell_count_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    malignant_col = resolve_col(df, ["n_test_malignant", "n_malignant", "stage2_heldout_malignant_total"])
    healthy_col = resolve_col(df, ["n_test_non_malignant", "n_normal", "stage2_heldout_healthy_total"])
    return malignant_col, healthy_col


def cell_count_values_from_row(row: pd.Series) -> tuple[Any, Any]:
    malignant = first_non_missing(
        row.get("n_test_malignant"),
        row.get("n_malignant"),
        row.get("stage2_heldout_malignant_total"),
    )
    healthy = first_non_missing(
        row.get("n_test_non_malignant"),
        row.get("n_normal"),
        row.get("stage2_heldout_healthy_total"),
    )
    return malignant, healthy


def format_cell_count_display(malignant: Any, healthy: Any, *, prefix: str) -> str:
    return (
        f"{prefix} malignant={_fmt_number(malignant, digits=0) or 'NA'}; "
        f"{prefix} healthy={_fmt_number(healthy, digits=0) or 'NA'}"
    )


def single_geneset_name(panel_id: Any, panel_type: Any) -> str:
    if str(panel_type) != "single_geneset_only":
        return "not a single-gene-set panel"
    text = str(panel_id)
    for prefix in ("single_geneset__", "hallmark_", "reactome_", "kegg_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.replace("_", " ")


def gene_summary(row: pd.Series, *, experiment_dir: Path, preview_n: int = 20) -> dict[str, str]:
    n_genes = row.get("n_covered_genes", np.nan)
    n_display = _fmt_number(n_genes, digits=0)

    gene_list_path = row.get("gene_list_path")
    if not isinstance(gene_list_path, str) or not gene_list_path:
        return {"n_genes": n_display, "preview": "", "genesets_preview": ""}
    path = Path(gene_list_path)
    path = path if path.is_absolute() else experiment_dir / path
    try:
        payload = json.loads(path.read_text())
        genes = payload.get("genes", [])
        genesets = payload.get("genesets", [])
    except Exception as exc:
        return {"n_genes": n_display, "preview": f"could not read genes: {exc!r}", "genesets_preview": ""}

    genesets_preview = ""
    if isinstance(genesets, list) and genesets:
        genesets_preview = ", ".join(map(str, genesets[:preview_n]))
        if len(genesets) > preview_n:
            genesets_preview += f", ... (+{len(genesets) - preview_n} more)"
    if str(row.get("stage0_panel_type", "")) != "single_geneset_only":
        return {"n_genes": n_display, "preview": "", "genesets_preview": genesets_preview}
    if isinstance(genes, list):
        if not n_display:
            n_display = str(len(genes))
        preview = ", ".join(map(str, genes[:preview_n]))
        if len(genes) > preview_n:
            preview += f", ... (+{len(genes) - preview_n} more)"
        return {"n_genes": n_display, "preview": preview, "genesets_preview": genesets_preview}
    return {"n_genes": n_display, "preview": "gene list has unexpected format", "genesets_preview": genesets_preview}


def format_reg(penalty: Any, c_value: Any, l1_ratio: Any) -> str:
    penalty_text = "NA" if _is_missing(penalty) else str(penalty)
    c_text = _fmt_number(c_value, digits=4)
    parts = [penalty_text]
    if c_text:
        parts.append(f"C={c_text}")
    if not _is_missing(l1_ratio):
        parts.append(f"l1={_fmt_number(l1_ratio, digits=3)}")
    return " ".join(parts)


def first_present(primary: Any, fallback: Any) -> Any:
    return fallback if _is_missing(primary) else primary


def first_non_missing(*values: Any) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return np.nan


def metric_label(metric_col: str) -> str:
    labels = {
        "leave_patient_out_auprc_mean": "LOPO AUPRC (cell-weighted mean)",
        "cell_weighted_auprc": "LOPO AUPRC (cell-weighted mean)",
        "patient_equal_auprc": "LOPO AUPRC (patient-equal mean)",
        "leave_patient_out_patient_equal_auprc_mean": "LOPO AUPRC (patient-equal mean)",
        "stage2_auprc": "LOPO AUPRC",
        "auprc": "LOPO AUPRC",
    }
    return labels.get(metric_col, f"LOPO AUPRC ({metric_col})")


def target_options(plot_data: pd.DataFrame) -> dict[str, str]:
    options = {"aggregate_mean": "LOPO mean"}
    targets = sorted(
        str(target)
        for target in plot_data["lopo_target"].dropna().astype(str).unique()
        if str(target) != "aggregate_mean"
    )
    for target in targets:
        label = f"Patient {target}"
        normal_only = plot_data.loc[plot_data["lopo_target"].eq(target), "normal_only"].map(_is_true).all()
        if normal_only:
            label = f"{label} (normal-only; AUPRC unavailable)"
        options[target] = label
    return options


def filter_mode(plot_data: pd.DataFrame, mode: str, target: str = "aggregate_mean") -> pd.DataFrame:
    target = target if target in set(plot_data["lopo_target"].astype(str)) else "aggregate_mean"
    out = plot_data.loc[plot_data["lopo_target"].astype(str).eq(str(target))].copy()
    if mode == "both":
        return out.loc[out["comparison_mode"].isin(["matched_regularization", "best_lopo"])].copy()
    return out.loc[out["comparison_mode"].eq(mode)].copy()


def make_figure(df: pd.DataFrame, *, y_label: str, y_axis_source: str = "lopo") -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Discovery vs LOPO Sharedness")
        fig.add_annotation(text="No rows are available for this selection.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    y_axis_source = y_axis_source if y_axis_source in {"lopo", "patient_specific_apparent"} else "lopo"
    df = df.copy()
    df["comparison_mode_label"] = [
        comparison_mode_display_label(mode, y_axis_source)
        for mode in df.get("comparison_mode", pd.Series("", index=df.index))
    ]
    y_col = "patient_specific_plot_metric" if y_axis_source == "patient_specific_apparent" else "lopo_plot_metric"
    symbol = "comparison_mode_label" if df["comparison_mode"].nunique() > 1 else None
    target_value = str(df["lopo_target"].dropna().iloc[0]) if df["lopo_target"].notna().any() else "aggregate_mean"
    target_label = str(df["lopo_target_label"].dropna().iloc[0]) if df["lopo_target_label"].notna().any() else "LOPO"
    if y_axis_source == "patient_specific_apparent":
        target_label = "Patient-specific apparent mean across patients" if target_value == "aggregate_mean" else f"Patient {target_value}"
        display_y_label = (
            "Patient-specific apparent AUPRC (cell-weighted mean)"
            if df["lopo_target"].astype(str).eq("aggregate_mean").all()
            else "Patient-specific apparent AUPRC"
        )
        title_prefix = "Discovery vs Patient-Specific Apparent"
    else:
        display_y_label = y_label if df["lopo_target"].astype(str).eq("aggregate_mean").all() else "Held-out patient AUPRC"
        title_prefix = "Discovery vs LOPO Sharedness"
    fig = px.scatter(
        df,
        x="discovery_plot_metric",
        y=y_col,
        color="stage0_panel_type",
        symbol=symbol,
        hover_name="short_panel_label",
        custom_data=CUSTOM_DATA,
        category_orders={"stage0_panel_type": PANEL_TYPE_ORDER},
        color_discrete_map=PANEL_TYPE_COLORS,
        labels={
            "discovery_plot_metric": "Discovery apparent AUPRC",
            y_col: display_y_label,
            "stage0_panel_type": "Stage 0 panel type",
            "comparison_mode_label": "Y-axis regularization setting",
        },
        title=f"{title_prefix}: {target_label}",
    )
    fig.update_traces(
        marker={"size": 12, "line": {"width": 0.8, "color": "white"}},
        hovertemplate=(
            "Panel id: %{customdata[2]}<br>"
            "Stage 0 genes: %{customdata[5]}<br>"
            "Stage 1: %{customdata[6]}, K=%{customdata[7]}<br>"
            "Cell counts: %{customdata[12]}"
            "<extra></extra>"
        ),
    )

    finite = pd.concat([df["discovery_plot_metric"], df[y_col]]).replace([np.inf, -np.inf], np.nan).dropna()
    upper = max(0.05, float(finite.max()) * 1.08) if not finite.empty else 1.0
    upper = min(1.0, upper)
    fig.update_xaxes(range=[0, upper], showgrid=True, zeroline=False)
    fig.update_yaxes(range=[0, upper], showgrid=True, zeroline=False, scaleanchor="x", scaleratio=1)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=upper,
        y1=upper,
        line={"color": "rgba(80,80,80,0.65)", "width": 1, "dash": "dash"},
    )
    if df["discovery_plot_metric"].notna().any():
        fig.add_vline(x=float(df["discovery_plot_metric"].median()), line_width=1, line_color="rgba(0,0,0,0.18)")
    if df[y_col].notna().any():
        fig.add_hline(y=float(df[y_col].median()), line_width=1, line_color="rgba(0,0,0,0.18)")
    else:
        missing_text = (
            "No patient-specific apparent AUPRC values are available for this selection."
            if y_axis_source == "patient_specific_apparent"
            else "No AUPRC values for this held-out patient because the fold has no malignant cells."
        )
        fig.add_annotation(
            text=missing_text,
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            bgcolor="rgba(255,255,255,0.85)",
        )
    fig.update_layout(
        template="plotly_white",
        width=900,
        height=720,
        legend_title_text="Stage 0 panel type",
        margin={"l": 70, "r": 30, "t": 70, "b": 70},
    )
    return fig


def comparison_mode_display_label(mode: Any, y_axis_source: str) -> str:
    mode_text = str(mode)
    if y_axis_source == "patient_specific_apparent" and mode_text == "best_lopo":
        return "LOPO-selected regularization"
    return MODE_LABELS.get(mode_text, mode_text)


def available_regularization_families(rec: pd.Series, reg_data: dict[str, Any]) -> list[str]:
    rid = str(rec.get("representation_id", ""))
    families: set[str] = set()
    for key in ["discovery_metrics", "lopo_metrics", "lopo_by_patient", "patient_specific_metrics"]:
        df = reg_data.get(key, pd.DataFrame())
        if df is None or df.empty or "representation_id" not in df.columns:
            continue
        sub = df.loc[df["representation_id"].astype(str).eq(rid)]
        if sub.empty or "penalty" not in sub.columns:
            continue
        l1_values = sub["l1_ratio"] if "l1_ratio" in sub.columns else pd.Series(np.nan, index=sub.index)
        for penalty, l1_ratio in zip(sub["penalty"], l1_values):
            families.add(regularization_family_key(penalty, l1_ratio))
    return sorted(families, key=regularization_family_sort_key)


def default_regularization_families(families: list[str]) -> list[str]:
    return list(families)


def default_coefficient_family(rec: pd.Series, families: list[str]) -> str | None:
    candidates = [
        regularization_family_key(first_present(rec.get("lopo_penalty"), rec.get("penalty")), first_present(rec.get("lopo_l1_ratio"), rec.get("l1_ratio"))),
        regularization_family_key(rec.get("penalty"), rec.get("l1_ratio")),
    ]
    for candidate in candidates:
        if candidate in families:
            return candidate
    return families[0] if families else None


def selected_coefficient_family(selected: Any, available: list[str]) -> str | None:
    selected_values: list[str]
    if isinstance(selected, str):
        selected_values = [selected]
    elif isinstance(selected, (list, tuple, set)):
        selected_values = [str(item) for item in selected]
    else:
        selected_values = []
    allowed = set(available)
    for value in selected_values:
        if value in allowed:
            return value
    return available[0] if available else None


def selected_regularization_families(selected: Any, available: list[str]) -> list[str]:
    if isinstance(selected, str):
        selected_values = [selected]
    elif isinstance(selected, (list, tuple, set)):
        selected_values = [str(item) for item in selected]
    else:
        selected_values = []
    allowed = set(available)
    out = [family for family in selected_values if family in allowed]
    return out or list(available)


def regularization_strength(c_value: Any) -> float | None:
    try:
        c_float = float(c_value)
    except Exception:
        return None
    if not math.isfinite(c_float) or c_float <= 0:
        return None
    return 1.0 / c_float


def regularization_family_key(penalty: Any, l1_ratio: Any) -> str:
    penalty_text = "NA" if _is_missing(penalty) else str(penalty).lower()
    return f"{penalty_text}|{l1_ratio_key(l1_ratio)}"


def l1_ratio_key(value: Any) -> str:
    if _is_missing(value):
        return "NA"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.12g}"


def regularization_family_label(family_key: str | None) -> str:
    if family_key is None:
        return "selected regularization family"
    penalty, l1_ratio = split_regularization_family_key(family_key)
    if penalty == "elasticnet" and l1_ratio != "NA":
        return f"elasticnet l1_ratio={l1_ratio}"
    return penalty


def split_regularization_family_key(family_key: str) -> tuple[str, str]:
    if "|" not in str(family_key):
        return str(family_key), "NA"
    penalty, l1_ratio = str(family_key).split("|", 1)
    return penalty, l1_ratio


def regularization_family_sort_key(family_key: str) -> tuple[int, float, str]:
    penalty, l1_ratio = split_regularization_family_key(family_key)
    penalty_order = {"l2": 0, "l1": 1, "elasticnet": 2}
    try:
        l1_number = float(l1_ratio)
    except Exception:
        l1_number = -1.0
    return penalty_order.get(penalty, 9), l1_number, family_key


def filter_regularization_family(df: pd.DataFrame, family_key: str) -> pd.DataFrame:
    if df.empty or "penalty" not in df.columns:
        return df.iloc[0:0].copy()
    penalty, l1_ratio = split_regularization_family_key(family_key)
    out = df.loc[df["penalty"].astype(str).str.lower().eq(penalty)].copy()
    if "l1_ratio" not in out.columns:
        return out if l1_ratio == "NA" else out.iloc[0:0].copy()
    if l1_ratio == "NA":
        return out.loc[out["l1_ratio"].map(_is_missing)].copy()
    return out.loc[out["l1_ratio"].map(l1_ratio_key).eq(l1_ratio)].copy()


def filter_regularization_families(df: pd.DataFrame, family_keys: list[str] | None) -> pd.DataFrame:
    if df.empty or not family_keys:
        return df.copy()
    if "reg_family" in df.columns:
        return df.loc[df["reg_family"].astype(str).isin(set(family_keys))].copy()
    parts = [filter_regularization_family(df, family_key) for family_key in family_keys]
    parts = [part for part in parts if not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def regularization_family_summary(family_keys: list[str] | None) -> str:
    if not family_keys:
        return "all regularization families"
    labels = [regularization_family_label(family) for family in family_keys]
    if len(labels) <= 3:
        return ", ".join(labels)
    return ", ".join(labels[:3]) + f", ... (+{len(labels) - 3} more)"


def source_label(source: str) -> str:
    labels = {
        "discovery": "Discovery",
        "lopo": "LOPO",
        "patient_specific": "Patient-specific",
    }
    return labels.get(source, str(source))


def make_regularization_path_figure(
    rec: pd.Series,
    reg_data: dict[str, Any],
    family_keys: list[str] | None,
    *,
    coefficient_family_key: str | None = None,
    path_scope: str = "cross_patient",
    patient_id: str | None = None,
    metric_key: str = "auprc",
    show_coefficients: bool = False,
    show_feature_count: bool = False,
) -> tuple[go.Figure, str]:
    metric_key = metric_key if metric_key in REG_METRIC_OPTIONS else "auprc"
    path_scope = path_scope if path_scope in {"cross_patient", "heldout_patient_lopo", "individual_patient"} else "cross_patient"
    if path_scope in {"heldout_patient_lopo", "individual_patient"} and not patient_id:
        patient_id = first_patient_option(reg_data, path_scope=path_scope)

    discovery_path = build_metric_path_frame(
        reg_data.get("discovery_metrics", pd.DataFrame()),
        rec,
        source="discovery",
        metric_key=metric_key,
    )
    comparison_source = "patient_specific" if path_scope == "individual_patient" else "lopo"
    if comparison_source == "patient_specific":
        comparison_path = build_patient_specific_metric_path_frame(reg_data, rec, patient_id, metric_key)
        comparison_label = f"Patient-specific apparent ({patient_id or 'patient unavailable'})"
    elif path_scope == "heldout_patient_lopo":
        comparison_path = build_lopo_metric_path_frame(reg_data, rec, metric_key, heldout_patient_id=patient_id)
        comparison_label = f"Held-out patient LOPO ({patient_id or 'patient unavailable'})"
    else:
        comparison_path = build_lopo_metric_path_frame(reg_data, rec, metric_key)
        comparison_label = "LOPO mean across held-out patients"
    available_families = sorted(
        set(discovery_path.get("reg_family", [])) | set(comparison_path.get("reg_family", [])),
        key=regularization_family_sort_key,
    )
    selected_families = selected_regularization_families(family_keys, available_families)
    coefficient_family = selected_coefficient_family(coefficient_family_key, available_families)
    coefficient_families = [coefficient_family] if coefficient_family else []
    discovery_path = filter_regularization_families(discovery_path, selected_families)
    comparison_path = filter_regularization_families(comparison_path, selected_families)

    row_specs = [("metric", REG_METRIC_OPTIONS[metric_key]["label"])]
    if show_feature_count:
        row_specs.append(("nonzero", "Nonzero coefficient count"))
    if show_coefficients:
        row_specs.append(
            (
                "coefficients",
                coefficient_panel_title(
                    path_scope,
                    patient_id,
                    coefficient_families,
                    lopo_target_label=str(rec.get("lopo_target_label", "LOPO")),
                ),
            )
        )

    n_rows = len(row_specs)
    row_heights = row_height_weights(row_specs)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1 if n_rows == 2 else 0.08,
        row_heights=row_heights,
        subplot_titles=[title for _, title in row_specs],
    )

    for source, path_df in [("discovery", discovery_path), (comparison_source, comparison_path)]:
        add_metric_path_traces(fig, path_df, source=source, y_col="metric_value", row=1)

    row_lookup = {kind: i + 1 for i, (kind, _) in enumerate(row_specs)}
    if "nonzero" in row_lookup:
        nonzero_row = row_lookup["nonzero"]
        for source, path_df in [("discovery", discovery_path), (comparison_source, comparison_path)]:
            add_metric_path_traces(fig, path_df, source=source, y_col="nonzero_value", row=nonzero_row, showlegend=False)
        fig.update_yaxes(title_text="nonzero coefficients", row=nonzero_row, col=1)

    if "coefficients" in row_lookup:
        coefficient_path = (
            build_coefficient_path_frame(reg_data, rec, coefficient_families, path_scope=path_scope, patient_id=patient_id)
            if coefficient_families
            else pd.DataFrame()
        )
        coef_row = row_lookup["coefficients"]
        if coefficient_path.empty:
            fig.add_annotation(
                text=f"No coefficient path rows found for {regularization_family_summary(coefficient_families)}.",
                x=0.5,
                y=0.5,
                xref=f"x{coef_row}" if coef_row > 1 else "x",
                yref=f"y{coef_row}" if coef_row > 1 else "y",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.85)",
            )
        else:
            add_coefficient_path_traces(fig, coefficient_path, row=coef_row)
        fig.add_hline(y=0, row=coef_row, col=1, line_width=1, line_color="rgba(0,0,0,0.35)")
        fig.update_yaxes(title_text="coefficient", row=coef_row, col=1)

    add_path_selection_markers(
        fig,
        rec=rec,
        discovery_path=discovery_path,
        comparison_path=comparison_path,
        comparison_source=comparison_source,
        metric_key=metric_key,
        n_rows=n_rows,
    )

    for row_idx in range(1, n_rows + 1):
        fig.update_xaxes(type="log", row=row_idx, col=1)
    fig.update_xaxes(title_text="regularization strength (1/C, log scale)", row=n_rows, col=1)
    fig.update_yaxes(title_text=REG_METRIC_OPTIONS[metric_key]["label"], row=1, col=1)

    title = f"Metric path over regularization<br>{representation_title(rec)}"
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        height=figure_height(row_specs),
        width=1040,
        margin={"l": 70, "r": 30, "t": 95, "b": 70},
        legend={"orientation": "v", "x": 1.02, "y": 1.0},
    )
    note_parts = [
        f"Top panel compares Discovery apparent {REG_METRIC_OPTIONS[metric_key]['label']} with {comparison_label}.",
    ]
    cell_count_note = regularization_scope_cell_count_display(reg_data, rec, path_scope=path_scope, patient_id=patient_id)
    if cell_count_note:
        note_parts.append(cell_count_note)
    if show_feature_count:
        if path_scope == "cross_patient" and str(rec.get("lopo_target", "aggregate_mean")) == "aggregate_mean":
            note_parts.append("The count panel uses scorecard nonzero-coefficient counts; aggregate LOPO counts are fold means.")
        else:
            note_parts.append("The count panel uses scorecard nonzero-coefficient counts.")
    if show_coefficients:
        if path_scope == "individual_patient":
            coef_scope = "patient-specific apparent"
        elif path_scope == "heldout_patient_lopo":
            coef_scope = "held-out-patient LOPO"
        else:
            coef_scope = "cross-patient LOPO fold-mean"
        note_parts.append(
            f"Visible regularization families: {regularization_family_summary(selected_families)}. "
            f"Coefficient panel shows {coef_scope} paths for {regularization_family_summary(coefficient_families)}."
        )
    else:
        note_parts.append(f"Visible regularization families: {regularization_family_summary(selected_families)}.")
    return fig, " ".join(note_parts)


def representation_title(rec: pd.Series) -> str:
    label = str(rec.get("short_panel_label", rec.get("stage0_panel_id", "")))
    n_genes = rec.get("n_covered_genes_display", "")
    method = rec.get("stage1_method", "")
    k_value = rec.get("effective_k_display", "")
    details = []
    if n_genes:
        details.append(f"{n_genes} genes")
    if method or k_value:
        details.append(f"{method}, k = {k_value}".strip(", "))
    return f"{label} ({' | '.join(details)})" if details else label


def row_height_weights(row_specs: list[tuple[str, str]]) -> list[float]:
    weights = {"metric": 0.5, "nonzero": 0.25, "coefficients": 0.45}
    raw = [weights.get(kind, 0.3) for kind, _ in row_specs]
    total = sum(raw) or 1.0
    return [item / total for item in raw]


def figure_height(row_specs: list[tuple[str, str]]) -> int:
    height = 430
    if any(kind == "nonzero" for kind, _ in row_specs):
        height += 260
    if any(kind == "coefficients" for kind, _ in row_specs):
        height += 390
    return height


def coefficient_panel_title(
    path_scope: str,
    patient_id: str | None,
    family_keys: list[str] | None = None,
    *,
    lopo_target_label: str = "LOPO",
) -> str:
    family_text = regularization_family_summary(family_keys)
    if path_scope == "individual_patient":
        return f"Patient-specific coefficient path ({patient_id or 'patient unavailable'} | {family_text})"
    if path_scope == "heldout_patient_lopo":
        return f"Held-out patient LOPO coefficient path ({patient_id or 'patient unavailable'} | {family_text})"
    return f"Cross-patient LOPO coefficient path ({family_text})"


def build_metric_path_frame(metrics: pd.DataFrame, rec: pd.Series, *, source: str, metric_key: str) -> pd.DataFrame:
    if metrics is None or metrics.empty or "representation_id" not in metrics.columns:
        return pd.DataFrame()
    rid = str(rec.get("representation_id", ""))
    sub = metrics.loc[metrics["representation_id"].astype(str).eq(rid)].copy()
    if sub.empty:
        return pd.DataFrame()
    candidates = REG_METRIC_OPTIONS.get(metric_key, REG_METRIC_OPTIONS["auprc"]).get(source, [])
    metric_col = resolve_col(sub, candidates)
    if metric_col is None:
        return pd.DataFrame()
    nz_col = resolve_col(sub, nonzero_candidates(source))
    return finalize_regularization_metric_frame(sub, metric_col=metric_col, nonzero_col=nz_col, source=source)


def build_lopo_metric_path_frame(
    reg_data: dict[str, Any],
    rec: pd.Series,
    metric_key: str,
    *,
    heldout_patient_id: str | None = None,
) -> pd.DataFrame:
    if not heldout_patient_id:
        metrics = reg_data.get("lopo_metrics", pd.DataFrame())
    else:
        metrics = reg_data.get("lopo_by_patient", pd.DataFrame())
        if metrics is None or metrics.empty:
            return pd.DataFrame()
        patient_col = resolve_col(metrics, ["heldout_patient_id", "heldout_patient", "patient"])
        if patient_col is None:
            return pd.DataFrame()
        metrics = metrics.loc[metrics[patient_col].astype(str).eq(str(heldout_patient_id))].copy()
    return build_metric_path_frame(metrics, rec, source="lopo", metric_key=metric_key)


def build_patient_specific_metric_path_frame(
    reg_data: dict[str, Any],
    rec: pd.Series,
    patient_id: str | None,
    metric_key: str,
) -> pd.DataFrame:
    metrics = reg_data.get("patient_specific_metrics", pd.DataFrame())
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    patient_col = resolve_col(metrics, ["patient_id", "patient"], required=False)
    if patient_col is None or not patient_id:
        return pd.DataFrame()
    metrics = metrics.loc[metrics[patient_col].astype(str).eq(str(patient_id))].copy()
    return build_metric_path_frame(metrics, rec, source="patient_specific", metric_key=metric_key)


def finalize_regularization_metric_frame(
    sub: pd.DataFrame,
    *,
    metric_col: str,
    nonzero_col: str | None,
    source: str,
) -> pd.DataFrame:
    out = sub.copy()
    if "C" not in out.columns or "penalty" not in out.columns:
        return pd.DataFrame()
    out["C"] = pd.to_numeric(out["C"], errors="coerce")
    out["regularization_strength"] = 1.0 / out["C"]
    out["metric_value"] = pd.to_numeric(out[metric_col], errors="coerce")
    out["reg_family"] = [
        regularization_family_key(penalty, l1_ratio)
        for penalty, l1_ratio in zip(out["penalty"], out.get("l1_ratio", pd.Series(np.nan, index=out.index)))
    ]
    if nonzero_col is not None:
        out["nonzero_value"] = pd.to_numeric(out[nonzero_col], errors="coerce")
    else:
        out["nonzero_value"] = np.nan
    out["nonzero_display"] = out["nonzero_value"].map(lambda x: _fmt_number(x, digits=2))
    out["metric_label"] = f"{source_label(source)} {metric_col}"
    return out.loc[out["regularization_strength"].gt(0) & out["metric_value"].notna()].copy()


def nonzero_candidates(source: str) -> list[str]:
    if source == "lopo":
        return ["nonzero_coefficient_count_mean_across_folds", "nonzero_coefficient_count", "selected_factor_count", "n_nonzero"]
    return ["nonzero_coefficient_count", "selected_factor_count", "n_nonzero"]


def add_metric_path_traces(
    fig: go.Figure,
    path_df: pd.DataFrame,
    *,
    source: str,
    y_col: str,
    row: int,
    showlegend: bool = True,
) -> None:
    if path_df.empty or y_col not in path_df.columns:
        return
    plot_df = path_df.loc[path_df[y_col].notna()].copy()
    if plot_df.empty:
        return
    for family, group in plot_df.groupby("reg_family", dropna=False):
        group = group.sort_values("regularization_strength")
        style = regularization_line_style(source, family)
        fig.add_trace(
            go.Scatter(
                x=group["regularization_strength"],
                y=group[y_col],
                mode="lines+markers",
                name=f"{source_label(source)} | {regularization_family_label(family)}",
                legendgroup=f"{source}-{family}",
                showlegend=showlegend,
                line={"color": style["color"], "dash": style["dash"], "width": style["width"]},
                marker={"size": 5, "color": style["color"]},
                customdata=np.stack(
                    [
                        group["C"].map(lambda x: _fmt_number(x, digits=4)),
                        group["nonzero_display"],
                        group["metric_label"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{customdata[2]}<br>"
                    "1/C=%{x:.4g}<br>"
                    "C=%{customdata[0]}<br>"
                    "value=%{y:.4g}<br>"
                    "nonzero=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def regularization_line_style(source: str, family_key: str) -> dict[str, Any]:
    penalty, l1_ratio = split_regularization_family_key(family_key)
    palettes = {
        "discovery": {"l2": "#08519c", "l1": "#3182bd", "elasticnet": "#6baed6", "default": "#1f77b4"},
        "lopo": {"l2": "#99000d", "l1": "#cb181d", "elasticnet": "#fb6a4a", "default": "#d62728"},
        "patient_specific": {"l2": "#7a0177", "l1": "#c51b8a", "elasticnet": "#f768a1", "default": "#c51b7d"},
    }
    color = palettes.get(source, palettes["lopo"]).get(penalty, palettes.get(source, palettes["lopo"])["default"])
    dash = "solid"
    if penalty == "elasticnet":
        dash = ELASTICNET_DASHES.get(l1_ratio, "dash")
    return {"color": color, "dash": dash, "width": 2.4 if penalty in {"l1", "l2"} else 2.0}


def build_coefficient_path_frame(
    reg_data: dict[str, Any],
    rec: pd.Series,
    family_keys: list[str] | None,
    *,
    path_scope: str,
    patient_id: str | None,
) -> pd.DataFrame:
    coef_key = "patient_specific_coef" if path_scope == "individual_patient" else "lopo_coef"
    coef = reg_data.get(coef_key, pd.DataFrame())
    if coef is None or coef.empty or "representation_id" not in coef.columns:
        return pd.DataFrame()
    rid = str(rec.get("representation_id", ""))
    sub = coef.loc[coef["representation_id"].astype(str).eq(rid)].copy()
    sub["reg_family"] = [
        regularization_family_key(penalty, l1_ratio)
        for penalty, l1_ratio in zip(sub["penalty"], sub.get("l1_ratio", pd.Series(np.nan, index=sub.index)))
    ]
    sub = filter_regularization_families(sub, family_keys)
    if sub.empty:
        return pd.DataFrame()

    patient_col = resolve_col(sub, ["patient_id", "patient", "heldout_patient_id", "heldout_patient", "fold_id"])
    if path_scope in {"individual_patient", "heldout_patient_lopo"} and patient_col is not None and patient_id:
        sub = sub.loc[sub[patient_col].astype(str).eq(str(patient_id))].copy()
    feature_col = resolve_col(sub, ["feature_id", "factor_id", "variable"], required=True)
    coef_col = resolve_col(sub, ["coefficient", "coef"], required=True)
    sub["C"] = pd.to_numeric(sub["C"], errors="coerce")
    sub["regularization_strength"] = 1.0 / sub["C"]
    sub["_coef"] = pd.to_numeric(sub[coef_col], errors="coerce").fillna(0.0)
    sub["_is_nonzero"] = sub["is_nonzero"].map(_is_true) if "is_nonzero" in sub.columns else sub["_coef"].ne(0)

    if path_scope == "cross_patient" and patient_col is not None:
        grouped = (
            sub.groupby(["reg_family", feature_col, "C", "regularization_strength"], dropna=False)
            .agg(plot_coef=("_coef", "mean"), selection_frequency=("_is_nonzero", "mean"), n_folds=(patient_col, "nunique"))
            .reset_index()
        )
    else:
        grouped = (
            sub.groupby(["reg_family", feature_col, "C", "regularization_strength"], dropna=False)
            .agg(plot_coef=("_coef", "mean"), selection_frequency=("_is_nonzero", "mean"))
            .reset_index()
        )
        grouped["n_folds"] = 1

    grouped = grouped.rename(columns={feature_col: "feature_id"})
    scores = (
        grouped.assign(abs_coef=grouped["plot_coef"].abs())
        .groupby("feature_id", dropna=False)
        .agg(max_abs_coef=("abs_coef", "max"), max_selection_frequency=("selection_frequency", "max"))
        .sort_values(["max_abs_coef", "max_selection_frequency"], ascending=False)
        .head(REG_PATH_TOP_K)
    )
    top_features = list(scores.index.astype(str))
    out = grouped.loc[grouped["feature_id"].astype(str).isin(top_features)].copy()
    out["feature_id"] = pd.Categorical(out["feature_id"].astype(str), categories=top_features, ordered=True)
    return out.sort_values(["reg_family", "feature_id", "regularization_strength"])


def add_coefficient_path_traces(fig: go.Figure, coefficient_path: pd.DataFrame, *, row: int = 2) -> None:
    group_cols = ["reg_family", "feature_id"] if "reg_family" in coefficient_path.columns else ["feature_id"]
    for group_key, group in coefficient_path.groupby(group_cols, sort=False, observed=True):
        if isinstance(group_key, tuple):
            family_key, feature_id = group_key
            trace_name = f"{regularization_family_label(str(family_key))} | {feature_id}"
        else:
            family_key, feature_id = "", group_key
            trace_name = str(feature_id)
        group = group.sort_values("regularization_strength")
        max_freq = float(group["selection_frequency"].max()) if "selection_frequency" in group else 1.0
        line_style = regularization_line_style("patient_specific", str(family_key)) if family_key else {"dash": "solid", "color": None}
        fig.add_trace(
            go.Scatter(
                x=group["regularization_strength"],
                y=group["plot_coef"],
                mode="lines+markers",
                name=trace_name,
                legendgroup=f"coef-{family_key}-{feature_id}",
                line={
                    "width": 1.1 + 1.8 * max_freq,
                    "dash": line_style.get("dash", "solid"),
                },
                marker={"size": 4},
                customdata=np.stack(
                    [
                        group["C"].map(lambda x: _fmt_number(x, digits=4)),
                        group["selection_frequency"].map(lambda x: _fmt_number(x, digits=3)),
                        group["n_folds"].map(lambda x: _fmt_number(x, digits=0)),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"{trace_name}<br>"
                    "1/C=%{x:.4g}<br>"
                    "C=%{customdata[0]}<br>"
                    "coef=%{y:.4g}<br>"
                    "selection_frequency=%{customdata[1]}<br>"
                    "folds=%{customdata[2]}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def add_path_selection_markers(
    fig: go.Figure,
    *,
    rec: pd.Series,
    discovery_path: pd.DataFrame,
    comparison_path: pd.DataFrame,
    comparison_source: str,
    metric_key: str,
    n_rows: int,
) -> None:
    discovery_chosen = choose_best_metric_row(discovery_path, metric_key)
    if discovery_chosen is not None:
        label = f"selected discovery: {format_reg(discovery_chosen.get('penalty'), discovery_chosen.get('C'), discovery_chosen.get('l1_ratio'))}"
        add_marker_at_path_row(fig, discovery_chosen, label=label, source="discovery", symbol="diamond", n_rows=n_rows)
    comparison_chosen = choose_best_metric_row(comparison_path, metric_key)
    if comparison_chosen is not None:
        comparison_label = "best LOPO" if comparison_source == "lopo" else "best patient-specific"
        label = f"{comparison_label}: {format_reg(comparison_chosen.get('penalty'), comparison_chosen.get('C'), comparison_chosen.get('l1_ratio'))}"
        add_marker_at_path_row(
            fig,
            comparison_chosen,
            label=label,
            source=comparison_source,
            symbol="diamond-open",
            n_rows=n_rows,
        )


def add_marker_at_path_row(fig: go.Figure, row: pd.Series, *, label: str, source: str, symbol: str, n_rows: int) -> None:
    strength = row.get("regularization_strength")
    y_value = row.get("metric_value")
    try:
        strength_float = float(strength)
        y_float = float(y_value)
    except Exception:
        return
    if not math.isfinite(strength_float) or not math.isfinite(y_float):
        return
    style = regularization_line_style(source, row.get("reg_family", ""))
    fig.add_trace(
        go.Scatter(
            x=[strength_float],
            y=[y_float],
            mode="markers",
            name=label,
            marker={"size": 13, "color": style["color"], "symbol": symbol, "line": {"width": 2, "color": style["color"]}},
            hovertemplate=f"{label}<br>1/C=%{{x:.4g}}<br>metric=%{{y:.4g}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for row_idx in range(1, n_rows + 1):
        fig.add_vline(x=strength_float, line_dash="dot", line_color=style["color"], line_width=1, row=row_idx, col=1)


def choose_best_metric_row(path_df: pd.DataFrame, metric_key: str) -> pd.Series | None:
    if path_df.empty or "metric_value" not in path_df.columns:
        return None
    candidates = path_df.loc[path_df["metric_value"].notna()].copy()
    if candidates.empty:
        return None
    ascending = metric_key == "log_loss"
    return candidates.sort_values("metric_value", ascending=ascending).iloc[0]


def regularization_scope_cell_count_display(
    reg_data: dict[str, Any],
    rec: pd.Series,
    *,
    path_scope: str,
    patient_id: str | None,
) -> str:
    rid = str(rec.get("representation_id", ""))
    if path_scope == "individual_patient":
        malignant, healthy = cell_counts_from_table(
            reg_data.get("patient_specific_metrics", pd.DataFrame()),
            rid,
            patient_col_candidates=["patient_id", "patient"],
            patient_id=patient_id,
        )
        return format_cell_count_display(malignant, healthy, prefix=f"patient {patient_id}") if not (_is_missing(malignant) and _is_missing(healthy)) else ""
    if path_scope == "heldout_patient_lopo":
        malignant, healthy = cell_counts_from_table(
            reg_data.get("lopo_by_patient", pd.DataFrame()),
            rid,
            patient_col_candidates=["heldout_patient_id", "heldout_patient", "patient"],
            patient_id=patient_id,
        )
        return format_cell_count_display(malignant, healthy, prefix=f"held-out {patient_id}") if not (_is_missing(malignant) and _is_missing(healthy)) else ""
    malignant, healthy = cell_counts_from_table(
        reg_data.get("lopo_by_patient", pd.DataFrame()),
        rid,
        patient_col_candidates=["heldout_patient_id", "heldout_patient", "patient"],
        patient_id=None,
    )
    if _is_missing(malignant) and _is_missing(healthy):
        malignant, healthy = cell_count_values_from_row(rec)
    return format_cell_count_display(malignant, healthy, prefix="total") if not (_is_missing(malignant) and _is_missing(healthy)) else ""


def cell_counts_from_table(
    df: pd.DataFrame,
    representation_id: str,
    *,
    patient_col_candidates: list[str],
    patient_id: str | None,
) -> tuple[Any, Any]:
    if df is None or df.empty or "representation_id" not in df.columns:
        return np.nan, np.nan
    sub = df.loc[df["representation_id"].astype(str).eq(str(representation_id))].copy()
    if sub.empty:
        return np.nan, np.nan
    patient_col = resolve_col(sub, patient_col_candidates)
    if patient_id is not None and patient_col is not None:
        sub = sub.loc[sub[patient_col].astype(str).eq(str(patient_id))].copy()
        if sub.empty:
            return np.nan, np.nan
    malignant_col, healthy_col = resolve_cell_count_columns(sub)
    if malignant_col is None or healthy_col is None:
        return np.nan, np.nan
    sub["_malignant"] = pd.to_numeric(sub[malignant_col], errors="coerce")
    sub["_healthy"] = pd.to_numeric(sub[healthy_col], errors="coerce")
    if patient_id is None and patient_col is not None:
        per_patient = sub.groupby(patient_col, dropna=False).agg(
            malignant=("_malignant", "max"),
            healthy=("_healthy", "max"),
        )
        return per_patient["malignant"].sum(), per_patient["healthy"].sum()
    return sub["_malignant"].dropna().max(), sub["_healthy"].dropna().max()


def first_patient_option(reg_data: dict[str, Any], *, path_scope: str = "individual_patient") -> str | None:
    options = patient_options_for_path_scope(reg_data, path_scope)
    return str(options[0]["value"]) if options else None


def patient_options_from_table(reg_data: dict[str, Any], table_key: str, candidates: list[str]) -> list[str]:
    df = reg_data.get(table_key, pd.DataFrame())
    if df is None or df.empty:
        return []
    patient_col = resolve_col(df, candidates)
    if patient_col is None:
        return []
    return sorted(df[patient_col].dropna().astype(str).unique())


def patient_specific_options(reg_data: dict[str, Any]) -> list[dict[str, str]]:
    patients = sorted(
        set(patient_options_from_table(reg_data, "patient_specific_metrics", ["patient_id", "patient"]))
        | set(patient_options_from_table(reg_data, "patient_specific_coef", ["patient_id", "patient"]))
    )
    return [{"label": patient, "value": patient} for patient in patients]


def heldout_lopo_patient_options(reg_data: dict[str, Any]) -> list[dict[str, str]]:
    patients = sorted(
        set(patient_options_from_table(reg_data, "lopo_by_patient", ["heldout_patient_id", "heldout_patient", "patient"]))
        | set(patient_options_from_table(reg_data, "lopo_coef", ["heldout_patient_id", "heldout_patient", "patient", "fold_id"]))
    )
    return [{"label": patient, "value": patient} for patient in patients]


def patient_options_for_path_scope(reg_data: dict[str, Any], path_scope: str) -> list[dict[str, str]]:
    if path_scope == "heldout_patient_lopo":
        return heldout_lopo_patient_options(reg_data)
    if path_scope == "individual_patient":
        return patient_specific_options(reg_data)
    patients = sorted(
        {str(option["value"]) for option in heldout_lopo_patient_options(reg_data)}
        | {str(option["value"]) for option in patient_specific_options(reg_data)}
    )
    return [{"label": patient, "value": patient} for patient in patients]


def scatter_regularization_options(y_axis_source: str) -> list[dict[str, str]]:
    if y_axis_source == "patient_specific_apparent":
        return [
            {"label": "Matched discovery regularization", "value": "matched_regularization"},
            {"label": "LOPO-selected regularization", "value": "best_lopo"},
            {"label": "Best patient-specific apparent regularization", "value": "best_patient_specific"},
        ]
    return [
        {"label": "Matched discovery regularization", "value": "matched_regularization"},
        {"label": "Best LOPO regularization", "value": "best_lopo"},
        {"label": "Both LOPO settings", "value": "both"},
    ]


def build_app(
    plot_data: pd.DataFrame,
    *,
    y_label: str,
    initial_mode: str,
    initial_target: str,
    reg_data: dict[str, Any],
):
    from dash import Dash, Input, Output, State, dcc, html

    mode_options = scatter_regularization_options("lopo")
    lopo_target_options = [
        {"label": label, "value": value}
        for value, label in target_options(plot_data).items()
    ]
    if initial_target not in {option["value"] for option in lopo_target_options}:
        initial_target = "aggregate_mean"
    patient_options: list[dict[str, str]] = []
    initial_patient = None
    reg_metric_options = [{"label": item["label"], "value": key} for key, item in REG_METRIC_OPTIONS.items()]
    scatter_y_source_options = [
        {"label": "LOPO held-out / mean", "value": "lopo"},
        {"label": "Patient-specific apparent", "value": "patient_specific_apparent"},
    ]
    app = Dash(__name__)
    app.title = "Stage 2 Fig 3A Sharedness"
    app.layout = html.Div(
        [
            html.H2("Discovery vs LOPO Sharedness"),
            html.P(
                "Each point is one shortlisted Stage 0 gene-space panel with its chosen Stage 1 DR method/K/seed. "
                "Use the dropdowns to compare LOPO at the discovery-selected regularization versus the best LOPO row, "
                "and to switch from the aggregate LOPO mean to a single held-out patient."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("LOPO target"),
                            dcc.Dropdown(
                                id="lopo-target",
                                options=lopo_target_options,
                                value=initial_target,
                                clearable=False,
                            ),
                        ],
                        style={"width": "420px", "display": "inline-block", "verticalAlign": "top", "marginRight": "16px"},
                    ),
                    html.Div(
                        [
                            html.Label("Y-axis regularization setting"),
                            dcc.Dropdown(
                                id="comparison-mode",
                                options=mode_options,
                                value=initial_mode,
                                clearable=False,
                            ),
                        ],
                        style={"width": "360px", "display": "inline-block", "verticalAlign": "top", "marginRight": "16px"},
                    ),
                    html.Div(
                        [
                            html.Label("Scatter Y-axis source"),
                            dcc.Dropdown(
                                id="scatter-y-source",
                                options=scatter_y_source_options,
                                value="lopo",
                                clearable=False,
                            ),
                        ],
                        style={"width": "260px", "display": "inline-block", "verticalAlign": "top"},
                    ),
                ],
                style={"marginBottom": "14px"},
            ),
            dcc.Graph(id="sharedness-scatter", style={"height": "760px"}),
            html.Div(id="point-details", style={"maxWidth": "900px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"}),
            html.Div(
                [
                    dcc.Checklist(
                        id="reg-path-toggle",
                        options=[
                            {"label": "Show regularization path for selected point", "value": "show"},
                            {"label": "Show nonzero coefficient count", "value": "nonzero"},
                            {"label": "Show coefficient paths", "value": "coefficients"},
                        ],
                        value=[],
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"fontWeight": "bold", "marginRight": "18px"},
                    ),
                    html.Div(
                        [
                            html.Label("Regularization families"),
                            dcc.Dropdown(
                                id="reg-family",
                                options=[],
                                value=[],
                                clearable=False,
                                disabled=True,
                                multi=True,
                                placeholder="Select one or more families",
                            ),
                        ],
                        style={"width": "360px", "display": "inline-block", "verticalAlign": "top", "marginRight": "16px", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("Coefficient family"),
                            dcc.Dropdown(
                                id="coef-family",
                                options=[],
                                value=None,
                                clearable=False,
                                disabled=True,
                                multi=False,
                                placeholder="Pick one family",
                            ),
                        ],
                        style={"width": "260px", "display": "inline-block", "verticalAlign": "top", "marginRight": "16px", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("Path scope"),
                            dcc.Dropdown(
                                id="reg-path-scope",
                                options=[
                                    {"label": "Cross-patient LOPO", "value": "cross_patient"},
                                    {"label": "Held-out patient LOPO", "value": "heldout_patient_lopo"},
                                    {"label": "Individual patient apparent", "value": "individual_patient"},
                                ],
                                value="cross_patient",
                                clearable=False,
                            ),
                        ],
                        style={"width": "240px", "display": "inline-block", "verticalAlign": "top", "marginRight": "16px", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("Patient"),
                            dcc.Dropdown(
                                id="reg-patient",
                                options=patient_options,
                                value=initial_patient,
                                clearable=False,
                                disabled=True,
                            ),
                        ],
                        style={"width": "160px", "display": "inline-block", "verticalAlign": "top", "marginRight": "16px", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Label("Y-axis metric"),
                            dcc.Dropdown(id="reg-y-metric", options=reg_metric_options, value="auprc", clearable=False),
                        ],
                        style={"width": "220px", "display": "inline-block", "verticalAlign": "top", "marginTop": "8px"},
                    ),
                ],
                style={"maxWidth": "1100px", "marginTop": "16px", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    dcc.Graph(id="reg-path-graph", style={"height": "880px"}),
                    html.Div(id="reg-path-note", style={"maxWidth": "900px", "fontSize": "13px", "color": "#444"}),
                ],
                id="reg-path-wrap",
                style={"display": "none"},
            ),
        ],
        style={"fontFamily": "Arial, sans-serif", "margin": "24px"},
    )

    @app.callback(
        Output("comparison-mode", "options"),
        Output("comparison-mode", "value"),
        Input("scatter-y-source", "value"),
        State("comparison-mode", "value"),
    )
    def update_scatter_regularization_options(y_axis_source: str, current_mode: str):
        options = scatter_regularization_options(y_axis_source)
        valid_values = {option["value"] for option in options}
        value = current_mode if current_mode in valid_values else options[0]["value"]
        return options, value

    @app.callback(
        Output("sharedness-scatter", "figure"),
        Input("comparison-mode", "value"),
        Input("lopo-target", "value"),
        Input("scatter-y-source", "value"),
    )
    def update_figure(mode: str, target: str, y_axis_source: str) -> go.Figure:
        return make_figure(filter_mode(plot_data, mode, target), y_label=y_label, y_axis_source=y_axis_source)

    @app.callback(
        Output("point-details", "children"),
        Input("sharedness-scatter", "clickData"),
        Input("scatter-y-source", "value"),
    )
    def update_details(click_data: dict[str, Any] | None, y_axis_source: str):
        if not click_data:
            return "Click a point to pin its details here."
        row_key = click_data["points"][0]["customdata"][0]
        row = plot_data.loc[plot_data["row_key"].eq(row_key)]
        if row.empty:
            return "Selected point was not found."
        rec = row.iloc[0]
        quick_line = (
            "quick Stage 1/quick-L2 diagnosis: "
            f"AUPRC={_fmt_number(rec.get('best_quick_auprc'), digits=4)}, "
            f"AUROC={_fmt_number(rec.get('best_quick_auroc'), digits=4)}, "
            f"balanced_accuracy={_fmt_number(rec.get('best_quick_balanced_accuracy'), digits=4)}"
        )
        display_mode_label = comparison_mode_display_label(rec.get("comparison_mode", ""), y_axis_source)
        lines = [
            f"{rec.get('short_panel_label', '')} [{display_mode_label}]",
            f"LOPO target: {rec.get('lopo_target_label', '')}",
            f"stage0_panel_id: {rec.get('stage0_panel_id', '')}",
            f"stage0_panel_type: {rec.get('stage0_panel_type', '')}",
            f"panel type note: {rec.get('stage0_panel_type_note', '')}",
            f"geneset: {rec.get('geneset_name', '')}",
            f"n_covered_genes: {rec.get('n_covered_genes_display', '')}",
            f"stage1: {rec.get('stage1_method', '')}, K={rec.get('effective_k_display', '')}",
            quick_line,
            f"shortlist reason: {rec.get('shortlist_reason', '')}",
            f"patient support: {rec.get('patient_support_display', '')}",
            f"specificity @ 0.5: {rec.get('specificity_display', '')}",
            f"discovery AUPRC: {_fmt_number(rec.get('discovery_plot_metric'), digits=4)}",
            f"LOPO AUPRC: {_fmt_number(rec.get('lopo_plot_metric'), digits=4)}",
            f"patient-specific apparent AUPRC: {_fmt_number(rec.get('patient_specific_plot_metric'), digits=4)}",
            f"discovery regularization: {rec.get('discovery_reg_display', '')}",
            f"LOPO regularization: {rec.get('lopo_reg_display', '')}",
            f"patient-specific regularization: {rec.get('patient_specific_reg_display', '')}",
            f"LOPO strategy: {rec.get('lopo_join_strategy', '')}",
            f"patient-specific strategy: {rec.get('patient_specific_join_strategy', '')}",
            f"current scatter y-axis: {'Patient-specific apparent AUPRC' if y_axis_source == 'patient_specific_apparent' else 'LOPO AUPRC'}",
        ]
        genes_preview = rec.get("genes_preview", "")
        if isinstance(genes_preview, str) and genes_preview:
            lines.extend(["", f"genes preview: {genes_preview}"])
        genesets_preview = rec.get("genesets_preview", "")
        if isinstance(genesets_preview, str) and genesets_preview:
            lines.extend(["", f"gene sets included: {genesets_preview}"])
        return "\n".join(lines)

    @app.callback(
        Output("reg-family", "options"),
        Output("reg-family", "value"),
        Output("reg-family", "disabled"),
        Output("coef-family", "options"),
        Output("coef-family", "value"),
        Output("coef-family", "disabled"),
        Input("sharedness-scatter", "clickData"),
    )
    def update_regularization_family_options(click_data: dict[str, Any] | None):
        if not click_data:
            return [], [], True, [], None, True
        row_key = click_data["points"][0]["customdata"][0]
        row = plot_data.loc[plot_data["row_key"].eq(row_key)]
        if row.empty:
            return [], [], True, [], None, True
        rec = row.iloc[0]
        families = available_regularization_families(rec, reg_data)
        options = [{"label": regularization_family_label(family), "value": family} for family in families]
        return options, default_regularization_families(families), not bool(options), options, default_coefficient_family(rec, families), not bool(options)

    @app.callback(
        Output("reg-patient", "options"),
        Output("reg-patient", "value"),
        Output("reg-patient", "disabled"),
        Input("reg-path-scope", "value"),
    )
    def update_patient_dropdown(path_scope: str):
        scoped_options = patient_options_for_path_scope(reg_data, path_scope)
        value = first_patient_option(reg_data, path_scope=path_scope)
        return scoped_options, value, path_scope == "cross_patient" or not bool(scoped_options)

    @app.callback(
        Output("reg-path-wrap", "style"),
        Output("reg-path-graph", "figure"),
        Output("reg-path-note", "children"),
        Input("reg-path-toggle", "value"),
        Input("reg-family", "value"),
        Input("coef-family", "value"),
        Input("reg-path-scope", "value"),
        Input("reg-patient", "value"),
        Input("reg-y-metric", "value"),
        Input("sharedness-scatter", "clickData"),
    )
    def update_regularization_path(
        toggle_values: list[str],
        family_keys: list[str] | None,
        coefficient_family_key: str | None,
        path_scope: str,
        patient_id: str | None,
        metric_key: str,
        click_data: dict[str, Any] | None,
    ):
        hidden_style = {"display": "none"}
        if "show" not in (toggle_values or []) or not click_data:
            return hidden_style, go.Figure(), ""
        row_key = click_data["points"][0]["customdata"][0]
        row = plot_data.loc[plot_data["row_key"].eq(row_key)]
        if row.empty:
            fig = go.Figure()
            fig.add_annotation(text="Selected point was not found.", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
            return {"display": "block"}, fig, ""
        rec = row.iloc[0]
        family_keys = selected_regularization_families(family_keys, available_regularization_families(rec, reg_data))
        toggle_values = toggle_values or []
        fig, note = make_regularization_path_figure(
            rec,
            reg_data,
            family_keys,
            coefficient_family_key=coefficient_family_key,
            path_scope=path_scope,
            patient_id=patient_id,
            metric_key=metric_key,
            show_coefficients="coefficients" in toggle_values,
            show_feature_count="nonzero" in toggle_values,
        )
        return {"display": "block"}, fig, note

    return app


def _fmt_number(value: Any, *, digits: int = 3) -> str:
    if _is_missing(value):
        return ""
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return ""
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}g}"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_true(value: Any) -> bool:
    if _is_missing(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


if __name__ == "__main__":
    main()
