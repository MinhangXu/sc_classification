#!/usr/bin/env python3
"""Build patient-wise LOPO transfer audit tables for Stage 2 MRD analyses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2_sharedness_plotting import (  # noqa: E402
    infer_biological_theme,
    select_regularization_rows,
    shorten_panel_label,
)

DEFAULT_EXPERIMENT_DIR = Path(
    "/home/minhang/mds_project/sc_classification/experiments/20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
)


def add_representation_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty or "representation_id" in out.columns:
        return out
    needed = ["stage0_panel_id", "representation_family", "stage1_method", "effective_k", "seed"]
    if set(needed).issubset(out.columns):
        out["representation_id"] = (
            out[needed[0]].astype(str)
            + "|"
            + out[needed[1]].astype(str)
            + "|"
            + out[needed[2]].astype(str)
            + "|"
            + pd.to_numeric(out[needed[3]], errors="coerce").round().astype("Int64").astype(str)
            + "|"
            + pd.to_numeric(out[needed[4]], errors="coerce").round().astype("Int64").astype(str)
        )
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = add_representation_id(df)
    if "short_panel_label" not in out.columns and "stage0_panel_id" in out.columns:
        out["short_panel_label"] = out["stage0_panel_id"].map(shorten_panel_label)
    if "biological_theme" not in out.columns and "stage0_panel_id" in out.columns:
        stage0_type = out["stage0_panel_type"] if "stage0_panel_type" in out.columns else pd.Series("", index=out.index)
        out["biological_theme"] = [infer_biological_theme(p, t) for p, t in zip(out["stage0_panel_id"], stage0_type)]
    return out


def regularization_label(df: pd.DataFrame) -> pd.Series:
    c_values = pd.to_numeric(df["C"], errors="coerce")
    l1_values = pd.to_numeric(df["l1_ratio"], errors="coerce") if "l1_ratio" in df else pd.Series(np.nan, index=df.index)
    return (
        df["penalty"].astype(str)
        + "|C="
        + c_values.map(lambda value: "nan" if pd.isna(value) else f"{value:.12g}")
        + "|l1_ratio="
        + l1_values.map(lambda value: "nan" if pd.isna(value) else f"{value:.12g}")
    )


def prepare_lopo_inputs(experiment_dir: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    lopo_scorecard = experiment_dir / "analysis" / "scorecards" / "stage2_sharedness_lopo_scorecard.csv"
    lopo_patient = experiment_dir / "stage2_supervised" / "multiobjective" / "sharedness_lopo" / "by_heldout_patient.csv"
    lopo_score = add_labels(pd.read_csv(lopo_scorecard, low_memory=False))
    lopo_patient_df = add_labels(pd.read_csv(lopo_patient, low_memory=False))

    if "threshold" in lopo_score.columns:
        lopo_score = lopo_score.loc[pd.to_numeric(lopo_score["threshold"], errors="coerce").eq(threshold)].copy()
    if "threshold" in lopo_patient_df.columns:
        lopo_patient_df = lopo_patient_df.loc[
            pd.to_numeric(lopo_patient_df["threshold"], errors="coerce").eq(threshold)
        ].copy()

    for df in (lopo_score, lopo_patient_df):
        df["stage0_id"] = df["stage0_panel_id"].astype(str)
        df["stage1_method_k"] = pd.to_numeric(df["effective_k"], errors="coerce").round().astype("Int64").astype(str)
        df["C_or_alpha"] = pd.to_numeric(df["C"], errors="coerce")
        df["stage2_regularization"] = regularization_label(df)
    return lopo_score, lopo_patient_df


def build_patient_wise(lopo_patient: pd.DataFrame) -> pd.DataFrame:
    malignant = pd.to_numeric(lopo_patient["n_malignant"], errors="coerce")
    normal = pd.to_numeric(lopo_patient["n_normal"], errors="coerce")
    prevalence = malignant / (malignant + normal).replace(0, np.nan)
    heldout_auprc = pd.to_numeric(lopo_patient["stage2_auprc"], errors="coerce")
    heldout_lift = heldout_auprc - prevalence
    normalized_lift = (heldout_lift / (1 - prevalence)).where(lambda values: np.isfinite(values), np.nan)

    patient_wise = pd.DataFrame(
        {
            "stage0_id": lopo_patient["stage0_id"],
            "stage0_panel_id": lopo_patient["stage0_panel_id"],
            "short_panel_label": lopo_patient["short_panel_label"],
            "biological_theme": lopo_patient.get("biological_theme", np.nan),
            "stage1_method": lopo_patient["stage1_method"],
            "stage1_method_k": lopo_patient["stage1_method_k"],
            "stage2_model": lopo_patient["classifier"],
            "stage2_regularization": lopo_patient["stage2_regularization"],
            "penalty": lopo_patient["penalty"],
            "C_or_alpha": lopo_patient["C_or_alpha"],
            "l1_ratio": pd.to_numeric(lopo_patient["l1_ratio"], errors="coerce"),
            "heldout_patient": lopo_patient["heldout_patient"],
            "n_malignant_h": malignant,
            "n_normal_h": normal,
            "patient_prevalence_pi_h": prevalence,
            "heldout_auprc": heldout_auprc,
            "heldout_auroc": pd.to_numeric(lopo_patient["stage2_auroc"], errors="coerce"),
            "heldout_lift": heldout_lift,
            "heldout_normalized_lift": normalized_lift,
            "has_both_classes": lopo_patient.get("has_both_classes", np.nan),
            "normal_only": lopo_patient.get("normal_only", np.nan),
            "low_malignant_support": lopo_patient.get("low_malignant_support", np.nan),
            "nonzero_coefficient_count": pd.to_numeric(lopo_patient.get("nonzero_coefficient_count", np.nan), errors="coerce"),
            "coef_l1_norm": pd.to_numeric(lopo_patient.get("coef_l1_norm", np.nan), errors="coerce"),
            "fit_id": lopo_patient["fit_id"],
        }
    )
    patient_wise["supported_patient"] = (
        patient_wise["has_both_classes"].astype("boolean").fillna(False)
        & ~patient_wise["normal_only"].astype("boolean").fillna(False)
        & ~patient_wise["low_malignant_support"].astype("boolean").fillna(False)
        & patient_wise["heldout_lift"].notna()
    )
    return patient_wise


def summarize_by_regularization(patient_wise: pd.DataFrame) -> pd.DataFrame:
    supported = patient_wise.loc[patient_wise["supported_patient"]].copy()
    keys = [
        "stage0_id",
        "stage0_panel_id",
        "short_panel_label",
        "biological_theme",
        "stage1_method",
        "stage1_method_k",
        "stage2_model",
        "stage2_regularization",
        "penalty",
        "C_or_alpha",
        "l1_ratio",
    ]
    return (
        supported.groupby(keys, dropna=False)
        .agg(
            median_heldout_auprc=("heldout_auprc", "median"),
            lopo_auprc_patient_median=("heldout_auprc", "median"),
            median_heldout_lift=("heldout_lift", "median"),
            lopo_lift_patient_median=("heldout_lift", "median"),
            lopo_lift_patient_p20=("heldout_lift", lambda values: values.quantile(0.20)),
            mean_heldout_lift=("heldout_lift", "mean"),
            n_supported_patients=("heldout_patient", "nunique"),
            n_positive_lift_patients=("heldout_lift", lambda values: int((values > 0).sum())),
            frac_positive_lift_patients=("heldout_lift", lambda values: float((values > 0).mean())),
            mean_patient_prevalence=("patient_prevalence_pi_h", "mean"),
        )
        .reset_index()
    )


def build_current_pooled_comparison(lopo_score: pd.DataFrame, regularization_summary: pd.DataFrame) -> pd.DataFrame:
    current_selected = select_regularization_rows(
        lopo_score,
        group_cols=["representation_id"],
        metric_col="cell_weighted_auprc",
        mode="max_metric",
        penalty_priority=("elasticnet", "l1", "l2"),
        min_nonzero=1,
    )
    current_selected["stage0_id"] = current_selected["stage0_panel_id"].astype(str)
    current_selected["stage1_method_k"] = (
        pd.to_numeric(current_selected["effective_k"], errors="coerce").round().astype("Int64").astype(str)
    )
    current_selected["C_or_alpha"] = pd.to_numeric(current_selected["C"], errors="coerce")
    current_selected["stage2_regularization"] = regularization_label(current_selected)
    current_selected = current_selected.rename(columns={"selected_metric_value": "lopo_auprc_pooled"})
    return current_selected[
        [
            "stage0_id",
            "stage0_panel_id",
            "short_panel_label",
            "biological_theme",
            "stage1_method",
            "stage1_method_k",
            "penalty",
            "C_or_alpha",
            "l1_ratio",
            "selection_metric",
            "lopo_auprc_pooled",
            "patient_equal_auprc",
            "stage2_regularization",
        ]
    ].merge(
        regularization_summary[
            [
                "stage0_id",
                "stage1_method",
                "stage1_method_k",
                "stage2_regularization",
                "lopo_auprc_patient_median",
                "lopo_lift_patient_median",
                "lopo_lift_patient_p20",
                "n_supported_patients",
                "n_positive_lift_patients",
                "frac_positive_lift_patients",
            ]
        ],
        on=["stage0_id", "stage1_method", "stage1_method_k", "stage2_regularization"],
        how="left",
    )


def build_heatmap(patient_wise: pd.DataFrame, best_reg_summary: pd.DataFrame, table_dir: Path, fig_dir: Path) -> None:
    panel_best_method = (
        best_reg_summary.sort_values(["stage0_id", "median_heldout_lift", "lopo_lift_patient_p20"], ascending=[True, False, False])
        .drop_duplicates("stage0_id")
    )
    shortlist = panel_best_method.sort_values(["lopo_lift_patient_p20", "median_heldout_lift"], ascending=False).head(12).copy()
    heat_rows = []
    for _, row in shortlist.iterrows():
        mask = (
            patient_wise["stage0_id"].eq(row["stage0_id"])
            & patient_wise["stage1_method"].eq(row["stage1_method"])
            & patient_wise["stage1_method_k"].eq(str(row["stage1_method_k"]))
            & patient_wise["stage2_regularization"].eq(row["stage2_regularization"])
            & patient_wise["supported_patient"]
        )
        sub = patient_wise.loc[mask].copy()
        sub["heatmap_row_label"] = row["short_panel_label"] + " | " + row["stage1_method"] + " K" + str(row["stage1_method_k"])
        sub["best_method"] = row["stage1_method"]
        sub["best_k"] = row["stage1_method_k"]
        sub["best_penalty"] = row["penalty"]
        sub["best_C_or_alpha"] = row["C_or_alpha"]
        heat_rows.append(sub)
    heat_input = pd.concat(heat_rows, ignore_index=True) if heat_rows else pd.DataFrame()
    heat_input.to_csv(table_dir / "v2_lopo_compact_patient_lift_heatmap_input.csv", index=False)
    if heat_input.empty:
        return

    row_order = (
        shortlist.assign(
            heatmap_row_label=shortlist["short_panel_label"]
            + " | "
            + shortlist["stage1_method"]
            + " K"
            + shortlist["stage1_method_k"].astype(str)
        )["heatmap_row_label"]
        .tolist()
    )
    matrix = heat_input.pivot_table(
        index="heatmap_row_label", columns="heldout_patient", values="heldout_lift", aggfunc="mean"
    ).reindex(row_order)
    vmax = float(np.nanpercentile(np.abs(matrix.values), 95)) if np.isfinite(matrix.values).any() else 0.1
    vmax = max(vmax, 0.05)
    fig, ax = plt.subplots(figsize=(9.8, max(5.5, 0.42 * len(matrix) + 1.8)))
    sns.heatmap(
        matrix,
        cmap="vlag",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "held-out lift over patient prevalence"},
        ax=ax,
    )
    ax.set_xlabel("Held-out patient")
    ax.set_ylabel("Panel | best Stage 1 method K")
    ax.set_title("Compact patient-wise LOPO transfer: best method/K by median held-out lift")
    fig.tight_layout()
    fig.savefig(fig_dir / "v2_lopo_compact_patient_lift_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "v2_lopo_compact_patient_lift_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def build_outputs(experiment_dir: Path, threshold: float) -> None:
    table_dir = experiment_dir / "analysis" / "scorecards" / "stage2_figure3_sharedness_v2_jun4"
    fig_dir = experiment_dir / "analysis" / "figures" / "stage2_figure3_sharedness_v2_jun4" / "diagnostic_layer1_layer2"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    lopo_score, lopo_patient = prepare_lopo_inputs(experiment_dir, threshold)
    patient_wise = build_patient_wise(lopo_patient)
    regularization_summary = summarize_by_regularization(patient_wise)

    patient_wise.to_csv(table_dir / "v2_lopo_patient_wise_transfer_by_regularization.csv", index=False)
    regularization_summary.to_csv(table_dir / "v2_lopo_patient_lift_summary_by_regularization.csv", index=False)

    score_metric_rows = lopo_score[
        [
            "stage0_id",
            "stage1_method",
            "stage1_method_k",
            "stage2_regularization",
            "cell_weighted_auprc",
            "patient_equal_auprc",
            "cell_weighted_auroc",
            "patient_equal_auroc",
            "n_valid_cells",
            "n_evaluable_patients_with_both_classes",
            "fit_id",
        ]
    ].rename(
        columns={
            "cell_weighted_auprc": "lopo_auprc_pooled",
            "patient_equal_auprc": "lopo_auprc_patient_mean_from_scorecard",
        }
    )
    regularization_summary.merge(
        score_metric_rows,
        on=["stage0_id", "stage1_method", "stage1_method_k", "stage2_regularization"],
        how="left",
    ).to_csv(table_dir / "v2_lopo_patient_lift_summary_by_regularization_with_pooled.csv", index=False)

    comparison = build_current_pooled_comparison(lopo_score, regularization_summary)
    comparison.to_csv(table_dir / "v2_lopo_metric_comparison_by_panel_method_k.csv", index=False)

    best_reg_summary = (
        regularization_summary.sort_values(
            ["stage0_id", "stage1_method", "stage1_method_k", "median_heldout_lift", "lopo_lift_patient_p20"],
            ascending=[True, True, True, False, False],
        )
        .drop_duplicates(["stage0_id", "stage1_method", "stage1_method_k"])
        .sort_values(["median_heldout_lift", "lopo_lift_patient_p20"], ascending=False)
    )
    best_reg_summary.to_csv(table_dir / "v2_lopo_patient_lift_summary_best_regularization_by_panel_method_k.csv", index=False)

    selected_keys = comparison[["stage0_id", "stage1_method", "stage1_method_k", "stage2_regularization", "selection_metric"]]
    selected_patient_table = patient_wise.merge(
        selected_keys,
        on=["stage0_id", "stage1_method", "stage1_method_k", "stage2_regularization"],
        how="inner",
    )
    selected_patient_table[
        [
            "stage0_id",
            "stage1_method",
            "stage1_method_k",
            "penalty",
            "C_or_alpha",
            "l1_ratio",
            "selection_metric",
            "heldout_patient",
            "heldout_auprc",
            "patient_prevalence_pi_h",
            "heldout_lift",
            "stage2_model",
            "stage2_regularization",
            "n_malignant_h",
            "n_normal_h",
            "heldout_auroc",
            "heldout_normalized_lift",
            "supported_patient",
        ]
    ].to_csv(table_dir / "v2_lopo_selected_regularization_patient_metrics.csv", index=False)

    build_heatmap(patient_wise, best_reg_summary, table_dir, fig_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    build_outputs(args.experiment_dir, args.threshold)


if __name__ == "__main__":
    main()
