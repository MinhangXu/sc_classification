#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from common import (
    eligible_relapse_patients,
    latent_diagnostics,
    load_relapse_mrd_adata,
    log,
    patient_class_summary,
    preprocess_patient_adata,
    run_dr_method,
    save_patient_dr_artifacts,
    split_csv_arg,
    write_json,
)


def make_output_root(experiments_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (experiments_dir / f"{stamp}_relapse_mrd_dr_classification").resolve()


def write_input_diagnostics(adata, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = patient_class_summary(adata)
    summary_df.to_csv(out_dir / "patient_feasibility_summary.csv", index=False)

    counts_by_class = (
        adata.obs.groupby(["patient", "relapse_mrd_label"], observed=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts_by_class.to_csv(out_dir / "counts_patient_by_relapse_mrd_label.csv")

    counts_by_tech = (
        adata.obs.groupby(["patient", "timepoint_type", "Tech"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["patient", "timepoint_type", "Tech"])
    )
    counts_by_tech.to_csv(out_dir / "counts_patient_by_timepoint_and_tech.csv", index=False)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit per-patient DR on paired MRD and Relapse cells.")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--experiments-dir", default=str(Path(__file__).resolve().parents[2] / "experiments"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patients", default="")
    parser.add_argument("--methods", default="pca,fa,factosig")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--gene-method", choices=["hvg", "all"], default="hvg")
    parser.add_argument("--hvg-n", type=int, default=3000)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--factosig-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--factosig-lr", type=float, default=1e-2)
    parser.add_argument("--factosig-max-iter", type=int, default=300)
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir).resolve()
    output_root = make_output_root(experiments_dir, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "config.json",
        {
            "input_h5ad": str(Path(args.input_h5ad).resolve()),
            "methods": split_csv_arg(args.methods),
            "k_requested": int(args.k),
            "gene_method": str(args.gene_method),
            "hvg_n": int(args.hvg_n),
            "standardize": not bool(args.no_standardize),
            "random_state": int(args.random_state),
            "factosig_device": str(args.factosig_device),
            "factosig_lr": float(args.factosig_lr),
            "factosig_max_iter": int(args.factosig_max_iter),
        },
    )

    log(f"Loading relapse/MRD subset from {args.input_h5ad}")
    adata = load_relapse_mrd_adata(args.input_h5ad)
    summary_df = write_input_diagnostics(adata, output_root / "input_diagnostics")

    requested_patients = split_csv_arg(args.patients)
    if requested_patients:
        patients = requested_patients
    else:
        patients = eligible_relapse_patients(summary_df, require_all_four_classes=False)
    methods = split_csv_arg(args.methods)
    run_manifest: List[Dict] = []

    for patient in patients:
        patient_mask = adata.obs["patient"].astype(str).values == str(patient)
        patient_adata = adata[patient_mask].copy()
        if patient_adata.n_obs == 0:
            run_manifest.append({"patient": str(patient), "status": "missing_after_filter"})
            continue

        patient_root = output_root / "patient_dr" / str(patient)
        patient_root.mkdir(parents=True, exist_ok=True)
        log(f"Processing patient={patient} cells={patient_adata.n_obs}")

        proc_adata, preprocess_info = preprocess_patient_adata(
            patient_adata,
            gene_method=str(args.gene_method),
            n_top_genes=int(args.hvg_n),
            standardize=not bool(args.no_standardize),
        )
        preprocess_info["patient"] = str(patient)
        preprocess_info["class_counts"] = patient_adata.obs["relapse_mrd_label"].value_counts().to_dict()
        preprocess_info["tech_counts"] = patient_adata.obs["Tech"].astype(str).value_counts().to_dict()
        write_json(patient_root / "preprocess_info.json", preprocess_info)
        proc_adata.write_h5ad(patient_root / "preprocessed.h5ad")
        proc_adata.obs.to_csv(patient_root / "metadata.csv")

        effective_k = int(min(int(args.k), max(proc_adata.n_obs - 1, 1), proc_adata.n_vars))
        if effective_k < 2:
            run_manifest.append(
                {
                    "patient": str(patient),
                    "status": "skipped_low_rank",
                    "n_cells": int(proc_adata.n_obs),
                    "n_genes": int(proc_adata.n_vars),
                    "effective_k": int(effective_k),
                }
            )
            continue

        for method in methods:
            method_dir = patient_root / method
            log(f"  DR method={method} effective_k={effective_k}")
            try:
                scores, loadings, feature_names, dr_info = run_dr_method(
                    proc_adata,
                    method=method,
                    k=effective_k,
                    random_state=int(args.random_state),
                    factosig_device=str(args.factosig_device),
                    factosig_lr=float(args.factosig_lr),
                    factosig_max_iter=int(args.factosig_max_iter),
                )
                dr_info = dict(dr_info)
                dr_info["patient"] = str(patient)
                dr_info["k_effective"] = int(effective_k)
                dr_info["n_cells"] = int(proc_adata.n_obs)
                dr_info["n_genes"] = int(proc_adata.n_vars)
                save_patient_dr_artifacts(
                    method_dir,
                    metadata=proc_adata.obs.copy(),
                    scores=scores,
                    loadings=loadings,
                    feature_names=feature_names,
                    gene_names=proc_adata.var_names.astype(str).tolist(),
                    dr_info=dr_info,
                )
                pd.DataFrame(latent_diagnostics(proc_adata.obs, scores)).to_csv(
                    method_dir / "latent_diagnostics.csv",
                    index=False,
                )
                run_manifest.append(
                    {
                        "patient": str(patient),
                        "method": method,
                        "status": "completed",
                        "n_cells": int(proc_adata.n_obs),
                        "n_genes": int(proc_adata.n_vars),
                        "k_effective": int(effective_k),
                    }
                )
            except Exception as exc:
                run_manifest.append(
                    {
                        "patient": str(patient),
                        "method": method,
                        "status": "error",
                        "error": str(exc),
                        "n_cells": int(proc_adata.n_obs),
                        "n_genes": int(proc_adata.n_vars),
                        "k_effective": int(effective_k),
                    }
                )

    pd.DataFrame(run_manifest).to_csv(output_root / "patient_dr" / "run_manifest.csv", index=False)
    log(f"Done. Patient-level DR outputs written to {output_root}")


if __name__ == "__main__":
    main()
