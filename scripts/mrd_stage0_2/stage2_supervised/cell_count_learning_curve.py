"""Held-out AUPRC against the number of Stage 2 training cells.

The Stage 1 basis stays the stored transductive ``scores.npy``. Each draw keeps
the held-out patient intact and refits only the atlas-selected logistic probe
on a label-stratified subset of the other cells. This is a probe learning
curve, not a refit of PCA, factor analysis, or FactoSig, and not a forecast
for sequencing a new patient.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import patient_factor_overlap as pfo

SUPPORTED_PATIENTS = ("P01", "P02", "P03", "P04", "P06", "P07", "P09")
TRAIN_SIZES = (500, 1_000, 2_000, 5_000, 10_000, 25_000)
N_SEEDS = 3
RANDOM_SEED = 42
# Four atlas cells the meeting already reads: the IFN-α PCA/FA contrast, and the
# two FactoSig-promax programs whose median lift is carried by several patients.
CURVE_SPECS = (
    ("single_geneset__hallmark_interferon_alpha_response", "pca", 40),
    ("single_geneset__hallmark_interferon_alpha_response", "fa", 40),
    ("single_geneset__hallmark_apoptosis", "factosig_promax", 40),
    ("single_geneset__hallmark_allograft_rejection", "factosig_promax", 40),
)


def output_dirs(repo_root: Path | None = None) -> tuple[Path, Path]:
    exp = pfo.experiment_dir(repo_root)
    table_dir = exp / "analysis" / "scorecards" / "stage2_diagnostic" / "cell_count_learning_curve_v1"
    figure_dir = exp / "analysis" / "figures" / "stage2_diagnostic" / "cell_count_learning_curve_v1"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def curve_targets(repo_root: Path | None = None) -> pd.DataFrame:
    """Atlas-selected penalty and C for the four curve representations."""
    path = (
        pfo.experiment_dir(repo_root)
        / "analysis"
        / "scorecards"
        / "stage2_diagnostic"
        / "old34_representation_diagnostic_table.csv"
    )
    table = pd.read_csv(path)
    rows = []
    for panel_id, method, k in CURVE_SPECS:
        match = table.loc[
            table["stage0_panel_id"].eq(panel_id)
            & table["stage1_method"].eq(method)
            & table["effective_k"].eq(k)
        ]
        if len(match) != 1:
            raise KeyError(f"Expected one diagnostic row for {panel_id} {method} K={k}, found {len(match)}")
        row = match.iloc[0]
        rows.append(
            {
                "stage0_panel_id": panel_id,
                "short_panel_label": str(row["short_panel_label"]),
                "stage1_method": method,
                "effective_k": int(k),
                "penalty": str(row["selected_penalty"]),
                "C": float(row["selected_C_or_alpha"]),
                "l1_ratio": pd.to_numeric(pd.Series([row["selected_l1_ratio"]]), errors="coerce").iloc[0],
                "curve_label": f"{row['short_panel_label']} | {method} K={k}",
            }
        )
    return pd.DataFrame(rows)


def _draw_stratified(y_train: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Label-stratified indices. The malignant fraction stays at the training prevalence."""
    n_total = int(y_train.size)
    if n >= n_total:
        return np.arange(n_total)
    positive = np.flatnonzero(y_train == 1)
    negative = np.flatnonzero(y_train == 0)
    n_pos = int(round(n * (positive.size / n_total)))
    n_pos = int(np.clip(n_pos, 1, positive.size))
    n_neg = int(n - n_pos)
    if n_neg > negative.size:
        n_neg = int(negative.size)
        n_pos = int(min(positive.size, n - n_neg))
    if n_neg < 1:
        n_neg = 1
        n_pos = int(min(positive.size, max(n - 1, 1)))
    chosen_pos = rng.choice(positive, n_pos, replace=False)
    chosen_neg = rng.choice(negative, n_neg, replace=False)
    return np.concatenate([chosen_pos, chosen_neg])


def _fit_auprc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    penalty: str,
    C: float,
    l1_ratio: float,
    seed: int,
) -> float:
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    x_train_t = scaler.fit_transform(imputer.fit_transform(x_train))
    x_test_t = scaler.transform(imputer.transform(x_test))
    penalty_name = str(penalty).lower()
    kwargs = {
        "C": float(C),
        "class_weight": "balanced",
        "max_iter": 200,
        "random_state": int(seed),
    }
    if penalty_name == "elasticnet":
        kwargs.update(solver="saga", penalty="elasticnet", l1_ratio=float(l1_ratio))
    elif penalty_name == "l1":
        kwargs.update(solver="liblinear", penalty="l1")
    else:
        kwargs.update(solver="lbfgs", penalty="l2")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        clf = LogisticRegression(**kwargs)
        clf.fit(x_train_t, y_train)
        probability = clf.predict_proba(x_test_t)[:, 1]
    return float(average_precision_score(y_test, probability))


def run_curve(
    repo_root: Path | None = None,
    *,
    train_sizes: tuple[int, ...] = TRAIN_SIZES,
    n_seeds: int = N_SEEDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    root = repo_root or pfo.find_repo_root()
    metadata = pfo.load_cell_metadata(root)
    patient = metadata["patient"].to_numpy()
    malignant = metadata["CN.label"].to_numpy() == "cancer"
    targets = curve_targets(root)
    records: list[dict] = []
    for target in targets.itertuples(index=False):
        print(f"[cell-count] {target.curve_label} ({target.penalty}, C={target.C:g})", flush=True)
        scores = np.load(pfo.score_path(root, target.stage0_panel_id, target.stage1_method, target.effective_k))
        if scores.shape[0] != patient.size:
            raise AssertionError(f"{target.curve_label} scores have {scores.shape[0]} rows, metadata has {patient.size}")
        for heldout in SUPPORTED_PATIENTS:
            test = patient == heldout
            train = ~test
            y_test = malignant[test].astype(int)
            prevalence = float(y_test.mean())
            y_train_full = malignant[train].astype(int)
            x_test = scores[test]
            x_train_full = scores[train]
            train_patients = patient[train]
            schedules = [(int(y_train_full.size), 0, True)]
            schedules += [(int(n), int(draw), False) for n in train_sizes if n < y_train_full.size for draw in range(n_seeds)]
            for n_requested, draw, is_full in schedules:
                rng = np.random.default_rng(seed + 1000 * draw + n_requested)
                if is_full:
                    chosen = np.arange(y_train_full.size)
                else:
                    chosen = _draw_stratified(y_train_full, n_requested, rng)
                y_train = y_train_full[chosen]
                auprc = _fit_auprc(
                    x_train_full[chosen],
                    y_train,
                    x_test,
                    y_test,
                    penalty=target.penalty,
                    C=float(target.C),
                    l1_ratio=0.0 if pd.isna(target.l1_ratio) else float(target.l1_ratio),
                    seed=seed + draw,
                )
                records.append(
                    {
                        "curve_label": target.curve_label,
                        "short_panel_label": target.short_panel_label,
                        "stage0_panel_id": target.stage0_panel_id,
                        "stage1_method": target.stage1_method,
                        "effective_k": int(target.effective_k),
                        "penalty": target.penalty,
                        "C": float(target.C),
                        "l1_ratio": None if pd.isna(target.l1_ratio) else float(target.l1_ratio),
                        "heldout_patient": heldout,
                        "patient_prevalence": prevalence,
                        "n_test_cells": int(test.sum()),
                        "n_test_malignant": int(y_test.sum()),
                        "is_full_training": bool(is_full),
                        "n_requested": int(n_requested),
                        "draw": int(draw),
                        "n_train_cells": int(chosen.size),
                        "n_train_malignant": int(y_train.sum()),
                        "n_train_patients": int(pd.unique(train_patients[chosen]).size),
                        "heldout_auprc": auprc,
                        "heldout_lift": auprc - prevalence,
                    }
                )
    return pd.DataFrame.from_records(records)


def atlas_agreement(curve: pd.DataFrame, repo_root: Path | None = None) -> pd.DataFrame:
    """Full-training refit versus the saved D1 atlas patient AUPRCs."""
    path = (
        pfo.experiment_dir(repo_root)
        / "analysis"
        / "scorecards"
        / "stage2_diagnostic"
        / "old34_d1_atlas_supported_patient_metrics.csv"
    )
    atlas = pd.read_csv(path, usecols=["stage0_panel_id", "configuration", "heldout_patient", "heldout_auprc"])
    full = curve.loc[curve["is_full_training"]].copy()
    full["configuration"] = full["stage1_method"] + "|K=" + full["effective_k"].astype(str)
    merged = full.merge(atlas, on=["stage0_panel_id", "configuration", "heldout_patient"], how="left", suffixes=("", "_atlas"))
    merged["abs_auprc_diff"] = (merged["heldout_auprc"] - merged["heldout_auprc_atlas"]).abs()
    return merged[
        ["curve_label", "heldout_patient", "heldout_auprc", "heldout_auprc_atlas", "abs_auprc_diff"]
    ]


def _per_patient_means(curve: pd.DataFrame) -> pd.DataFrame:
    """One row per representation, held-out patient, and nominal training size.

    Full-training folds do not share a cell count: holding out P09 leaves about
    43,000 training cells and holding out P04 leaves about 59,000. Those seven
    folds are one condition, not seven points on the x-axis.
    """
    work = curve.copy()
    work["size_id"] = np.where(work["is_full_training"], -1, work["n_requested"].astype(int))
    return (
        work.groupby(
            ["curve_label", "heldout_patient", "size_id", "is_full_training", "patient_prevalence"],
            as_index=False,
        )
        .agg(
            n_train_cells=("n_train_cells", "mean"),
            n_train_malignant=("n_train_malignant", "mean"),
            n_train_patients=("n_train_patients", "mean"),
            heldout_auprc=("heldout_auprc", "mean"),
            heldout_lift=("heldout_lift", "mean"),
            auprc_draw_sd=("heldout_auprc", "std"),
        )
    )


def summarize_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """Median across the seven patients of the per-patient mean over draws."""
    per_patient = _per_patient_means(curve)
    summary = (
        per_patient.groupby(["curve_label", "size_id", "is_full_training"], as_index=False)
        .agg(
            n_train_cells=("n_train_cells", "median"),
            n_train_malignant=("n_train_malignant", "median"),
            n_train_patients=("n_train_patients", "median"),
            median_auprc=("heldout_auprc", "median"),
            p20_auprc=("heldout_auprc", lambda s: float(np.quantile(s, 0.2))),
            p80_auprc=("heldout_auprc", lambda s: float(np.quantile(s, 0.8))),
            median_lift=("heldout_lift", "median"),
            p20_lift=("heldout_lift", lambda s: float(np.quantile(s, 0.2))),
            p80_lift=("heldout_lift", lambda s: float(np.quantile(s, 0.8))),
        )
        .sort_values(["curve_label", "n_train_cells"])
        .reset_index(drop=True)
    )
    return summary


def save_curve_figures(curve: pd.DataFrame, repo_root: Path | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    _, figure_dir = output_dirs(repo_root)
    patient_colors = {
        "P01": "#1b9e77",
        "P02": "#d95f02",
        "P03": "#7570b3",
        "P04": "#e7298a",
        "P06": "#66a61e",
        "P07": "#e6ab02",
        "P09": "#a6761d",
    }
    labels = list(dict.fromkeys(curve["curve_label"]))
    per_patient = _per_patient_means(curve)
    summary = summarize_curve(curve)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), sharex=True)
    for ax, label in zip(axes.ravel(), labels):
        subset = per_patient.loc[per_patient["curve_label"].eq(label)]
        cohort = summary.loc[summary["curve_label"].eq(label)].sort_values("n_train_cells")
        for patient_id, color in patient_colors.items():
            one = subset.loc[subset["heldout_patient"].eq(patient_id)].sort_values("n_train_cells")
            ax.plot(one["n_train_cells"], one["heldout_auprc"], color=color, lw=1.1, alpha=0.9, label=patient_id)
        ax.plot(cohort["n_train_cells"], cohort["median_auprc"], color="black", lw=2.2, label="median of 7")
        ax.fill_between(cohort["n_train_cells"], cohort["p20_auprc"], cohort["p80_auprc"], color="0.75", alpha=0.45, lw=0)
        ax.set_xscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Held-out AUPRC")
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", color="#eeeeee")
    axes[0, 1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("Stage 2 training cells (label-stratified; test patient unchanged)")
    fig.suptitle("Held-out AUPRC vs training-cell count on the frozen Stage 1 basis", fontsize=13)
    fig.tight_layout()
    auprc_path = figure_dir / "D15A_auprc_vs_training_cells.png"
    fig.savefig(auprc_path, dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "D15A_auprc_vs_training_cells.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for color, label in zip(colors, labels):
        cohort = summary.loc[summary["curve_label"].eq(label)].sort_values("n_train_cells")
        ax.plot(cohort["n_train_cells"], 100 * cohort["median_lift"], color=color, lw=2.2, label=label)
        ax.fill_between(
            cohort["n_train_cells"],
            100 * cohort["p20_lift"],
            100 * cohort["p80_lift"],
            color=color,
            alpha=0.15,
            lw=0,
        )
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Stage 2 training cells")
    ax.set_ylabel("Median held-out lift (AUPRC pp)")
    ax.set_title("Cohort median lift vs training-cell count")
    ax.grid(True, which="both", color="#eeeeee")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    lift_path = figure_dir / "D15B_median_lift_vs_training_cells.png"
    fig.savefig(lift_path, dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "D15B_median_lift_vs_training_cells.pdf", bbox_inches="tight")
    plt.close(fig)
    return [auprc_path, lift_path]


def write_curve(curve: pd.DataFrame, repo_root: Path | None = None) -> Path:
    table_dir, _ = output_dirs(repo_root)
    curve_path = table_dir / "cell_count_learning_curve.csv"
    summary_path = table_dir / "cell_count_learning_curve_summary.csv"
    agreement_path = table_dir / "cell_count_learning_curve_atlas_agreement.csv"
    curve.to_csv(curve_path, index=False)
    summarize_curve(curve).to_csv(summary_path, index=False)
    atlas_agreement(curve, repo_root).to_csv(agreement_path, index=False)
    return curve_path
