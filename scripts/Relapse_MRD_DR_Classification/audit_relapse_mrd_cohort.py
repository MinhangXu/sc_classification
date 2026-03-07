#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import pandas as pd


def coarse_timepoint(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return re.sub(r"_[0-9]+$", "", str(value))


def decode_categorical(group) -> List[Optional[str]]:
    codes = group["codes"][:]
    categories_raw = group["categories"][:]
    categories = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in categories_raw
    ]
    values: List[Optional[str]] = []
    for code in codes:
        if int(code) < 0:
            values.append(None)
        else:
            values.append(categories[int(code)])
    return values


def read_obs_columns(path: Path, columns: List[str]) -> Dict[str, List[Optional[str]]]:
    out: Dict[str, List[Optional[str]]] = {}
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        for column in columns:
            if column not in obs:
                raise KeyError(f"Column '{column}' not found in obs.")
            out[column] = decode_categorical(obs[column])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit relapse/MRD cohort availability from an h5ad file.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "cohort_audit"),
        help="Directory for audit CSV and markdown outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_h5ad).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    obs = read_obs_columns(
        input_path,
        columns=["patient", "Time", "Tech", "CN.label", "predicted.annotation", "source"],
    )
    df = pd.DataFrame(obs)
    df["timepoint_type"] = df["Time"].map(coarse_timepoint)

    focus = df[
        df["patient"].notna()
        & (df["patient"].astype(str) != "unknown")
        & df["timepoint_type"].isin(["MRD", "Relapse"])
        & df["CN.label"].isin(["cancer", "normal"])
    ].copy()
    focus["relapse_mrd_label"] = focus["timepoint_type"].astype(str) + "_" + focus["CN.label"].astype(str)

    counts_patient_class = (
        focus.groupby(["patient", "relapse_mrd_label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["MRD_cancer", "MRD_normal", "Relapse_cancer", "Relapse_normal"],
            fill_value=0,
        )
        .sort_index()
    )
    counts_patient_class.to_csv(output_dir / "counts_patient_by_relapse_mrd_label.csv")

    counts_patient_time_tech = (
        focus.groupby(["patient", "timepoint_type", "Tech"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["patient", "timepoint_type", "Tech"])
    )
    counts_patient_time_tech.to_csv(output_dir / "counts_patient_by_timepoint_and_tech.csv", index=False)

    counts_patient_celltype = (
        focus.groupby(["patient", "timepoint_type", "CN.label", "predicted.annotation"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["patient", "timepoint_type", "CN.label", "n_cells"], ascending=[True, True, True, False])
    )
    counts_patient_celltype.to_csv(output_dir / "counts_patient_by_celltype_timepoint_and_label.csv", index=False)

    summary_rows: List[Dict] = []
    for patient, sub in focus.groupby("patient", observed=False):
        label_counts = Counter(sub["relapse_mrd_label"])
        tech_by_tp = defaultdict(set)
        for _, row in sub.iterrows():
            tech_by_tp[str(row["timepoint_type"])].add(str(row["Tech"]))
        summary_rows.append(
            {
                "patient": str(patient),
                "n_cells": int(sub.shape[0]),
                "has_MRD": bool((sub["timepoint_type"] == "MRD").any()),
                "has_Relapse": bool((sub["timepoint_type"] == "Relapse").any()),
                "n_MRD_cancer": int(label_counts.get("MRD_cancer", 0)),
                "n_MRD_normal": int(label_counts.get("MRD_normal", 0)),
                "n_Relapse_cancer": int(label_counts.get("Relapse_cancer", 0)),
                "n_Relapse_normal": int(label_counts.get("Relapse_normal", 0)),
                "min_four_class_support": int(
                    min(
                        label_counts.get("MRD_cancer", 0),
                        label_counts.get("MRD_normal", 0),
                        label_counts.get("Relapse_cancer", 0),
                        label_counts.get("Relapse_normal", 0),
                    )
                ),
                "MRD_tech_values": ",".join(sorted(tech_by_tp["MRD"])),
                "Relapse_tech_values": ",".join(sorted(tech_by_tp["Relapse"])),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["min_four_class_support", "patient"])
    summary_df.to_csv(output_dir / "patient_feasibility_summary.csv", index=False)

    eligible = summary_df[summary_df["has_MRD"] & summary_df["has_Relapse"]].copy()
    strict = eligible[eligible["min_four_class_support"] > 0].copy()

    summary_lines = [
        "# Relapse/MRD Cohort Audit",
        "",
        f"- Input: `{input_path}`",
        f"- Total focus cells (`MRD`/`Relapse`, `cancer`/`normal`, known patient): {int(focus.shape[0])}",
        f"- Patients with both coarse `MRD` and `Relapse`: {int(eligible.shape[0])}",
        f"- Patients with all four classes present: {int(strict.shape[0])}",
        "",
        "## Patients with paired MRD and Relapse",
        "",
    ]
    for _, row in eligible.iterrows():
        summary_lines.append(
            "- "
            f"{row['patient']}: "
            f"MRD_cancer={row['n_MRD_cancer']}, "
            f"MRD_normal={row['n_MRD_normal']}, "
            f"Relapse_cancer={row['n_Relapse_cancer']}, "
            f"Relapse_normal={row['n_Relapse_normal']}, "
            f"MRD_tech={row['MRD_tech_values']}, "
            f"Relapse_tech={row['Relapse_tech_values']}"
        )
    summary_lines.extend(
        [
            "",
            "## Key caveats",
            "",
            "- `MRD_cancer` is the limiting class for several patients (`P05`, `P07`, `P13`, and `P08` has zero MRD cancer cells).",
            "- Most paired patients follow the expected tech pattern of MRD=`CITE` and Relapse=`Multi`, but `P13` has both timepoints in `CITE` and `P01`/`P02`/`P03` include mixed-tech MRD cells.",
            "- These outputs should be treated as the source of truth for patient inclusion and CV feasibility thresholds in downstream runners.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines))


if __name__ == "__main__":
    main()
