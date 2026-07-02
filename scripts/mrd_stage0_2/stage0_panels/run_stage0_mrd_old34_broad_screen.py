#!/usr/bin/env python
"""
Broad MRD Stage 0/1/quick-Stage-2 screen from the original cohort AnnData.

This runner intentionally does not reuse older preprocessed experiment folders.
It creates a fresh experiment directory, builds a shared MRD/CITE gene universe,
materializes old-34 and HVG-anchor Stage 0 panels, writes across-patient Stage 1
artifacts, and optionally evaluates a lightweight patient-aware shared classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger("stage0_mrd_old34_broad_screen")

DEFAULT_INPUT_H5AD = Path("/home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad")
DEFAULT_OUT_ROOT = Path("/home/minhang/mds_project/sc_classification/experiments")
DEFAULT_SC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GMT = DEFAULT_SC_ROOT / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "genesets_v1.gmt"
DEFAULT_OLD_MANIFEST = DEFAULT_SC_ROOT / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "manifest.tsv"

MALIGNANT_VALUES = {"cancer"}
NORMAL_VALUES = {"normal"}


@dataclass
class ArtifactPaths:
    experiment_dir: Path
    logs_dir: Path
    preprocessing_dir: Path
    gene_universe_dir: Path
    panels_dir: Path
    panel_genes_dir: Path
    panel_gene_annotations_dir: Path
    stage1_dr_dir: Path
    stage1_direct_gene_dir: Path
    stage2_shared_dir: Path
    scorecards_dir: Path
    reports_dir: Path


def split_csv(value: str | list[str] | None, cast=str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    return [cast(x.strip()) for x in str(value).replace(",", " ").split() if x.strip()]


def safe_id(value: str) -> str:
    out = str(value).lower()
    for old, new in [(" ", "_"), ("/", "_"), ("-", "_"), (".", "_"), (":", "_")]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Series):
        return value.to_dict()
    return str(value)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default))


def write_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        path.write_text(yaml.safe_dump(obj, sort_keys=False))
    except Exception:
        # JSON is valid YAML and avoids requiring PyYAML in minimal environments.
        path.write_text(json.dumps(obj, indent=2, default=json_default))


def configure_logging(logs_dir: Path, verbose: bool) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"stage0_mrd_old34_broad_screen_{time.strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    LOGGER.info("Logging to %s", log_path)
    return log_path


def add_import_paths(sc_root: Path) -> None:
    for candidate in (sc_root / "src", sc_root / "scripts" / "dr_feature_screening" / "plan0_1_grid"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as f:
        sha.update(f.read(1024 * 1024))
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256_first_1mb": sha.hexdigest(),
    }


def get_git_commit(sc_root: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(["git", "-C", str(sc_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def make_experiment_id(slug: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"{stamp}_{slug}".encode()).hexdigest()[:8]
    return f"{stamp}_{slug}_{digest}"


def make_paths(experiment_dir: Path, branch_name: str = "") -> ArtifactPaths:
    branch = safe_id(branch_name) if branch_name else ""
    if branch:
        return ArtifactPaths(
            experiment_dir=experiment_dir,
            logs_dir=experiment_dir / "logs" / branch,
            preprocessing_dir=experiment_dir / "preprocessing" / branch,
            gene_universe_dir=experiment_dir / "preprocessing" / "gene_universe" / branch,
            panels_dir=experiment_dir / "preprocessing" / "panels" / branch,
            panel_genes_dir=experiment_dir / "preprocessing" / "panels" / branch / "stage0_panel_genes",
            panel_gene_annotations_dir=experiment_dir / "preprocessing" / "panels" / branch / "stage0_panel_gene_annotations",
            stage1_dr_dir=experiment_dir / "stage1_dr" / branch,
            stage1_direct_gene_dir=experiment_dir / "stage1_direct_gene" / branch,
            stage2_shared_dir=experiment_dir / "stage2_supervised" / "shared_cross_patient" / branch,
            scorecards_dir=experiment_dir / "analysis" / "scorecards" / branch,
            reports_dir=experiment_dir / "analysis" / "reports" / branch,
        )
    return ArtifactPaths(
        experiment_dir=experiment_dir,
        logs_dir=experiment_dir / "logs",
        preprocessing_dir=experiment_dir / "preprocessing",
        gene_universe_dir=experiment_dir / "preprocessing" / "gene_universe",
        panels_dir=experiment_dir / "preprocessing" / "panels",
        panel_genes_dir=experiment_dir / "preprocessing" / "panels" / "stage0_panel_genes",
        panel_gene_annotations_dir=experiment_dir / "preprocessing" / "panels" / "stage0_panel_gene_annotations",
        stage1_dr_dir=experiment_dir / "stage1_dr",
        stage1_direct_gene_dir=experiment_dir / "stage1_direct_gene",
        stage2_shared_dir=experiment_dir / "stage2_supervised" / "shared_cross_patient",
        scorecards_dir=experiment_dir / "analysis" / "scorecards",
        reports_dir=experiment_dir / "analysis" / "reports",
    )


def ensure_dirs(paths: ArtifactPaths) -> None:
    for path in asdict(paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)


def derive_timepoint_type(adata: ad.AnnData, timepoint_col: str, fallback_time_col: str = "Time") -> None:
    if timepoint_col in adata.obs.columns or fallback_time_col not in adata.obs.columns:
        return
    coarse = (
        adata.obs[fallback_time_col]
        .astype(str)
        .str.replace(r"_[0-9]+$", "", regex=True)
        .replace({"unknown": np.nan})
    )
    adata.obs[timepoint_col] = coarse


def parse_gmt(path: Path) -> dict[str, list[str]]:
    genesets: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seen: set[str] = set()
            genes: list[str] = []
            for gene in parts[2:]:
                upper = gene.strip().upper()
                if upper and upper not in seen:
                    genes.append(upper)
                    seen.add(upper)
            genesets[parts[0]] = genes
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


def matrix_to_csr(x: Any) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


def filter_cells(adata: ad.AnnData, args: argparse.Namespace) -> tuple[ad.AnnData, pd.DataFrame]:
    derive_timepoint_type(adata, args.timepoint_col)
    rows: list[dict[str, Any]] = []

    def apply_mask(label: str, mask: pd.Series | np.ndarray) -> None:
        nonlocal adata
        before = adata.n_obs
        mask_arr = np.asarray(mask, dtype=bool)
        adata = adata[mask_arr].copy()
        rows.append({"filter": label, "n_before": int(before), "n_after": int(adata.n_obs), "n_removed": int(before - adata.n_obs)})
        LOGGER.info("%s: %d -> %d cells", label, before, adata.n_obs)

    if args.tech and args.tech_col in adata.obs.columns:
        apply_mask(f"{args.tech_col} == {args.tech}", adata.obs[args.tech_col].astype(str).eq(args.tech))
    elif args.tech:
        LOGGER.warning("Tech filter requested but obs column %s was not found", args.tech_col)

    if args.timepoint_col in adata.obs.columns:
        apply_mask(f"{args.timepoint_col} == {args.timepoint}", adata.obs[args.timepoint_col].astype(str).eq(args.timepoint))
    else:
        LOGGER.warning("Timepoint filter requested but obs column %s was not found", args.timepoint_col)

    if args.target_col not in adata.obs.columns:
        raise ValueError(f"Missing target column: {args.target_col}")
    valid_labels = {args.positive_class, args.negative_class}
    apply_mask(f"{args.target_col} in {sorted(valid_labels)}", adata.obs[args.target_col].astype(str).isin(valid_labels))

    if args.patient_col not in adata.obs.columns:
        raise ValueError(f"Missing patient column: {args.patient_col}")
    if adata.n_obs == 0:
        raise ValueError("No cells remain after MRD/tech/label filtering")
    return adata, pd.DataFrame(rows)


def filter_gene_universe(adata: ad.AnnData, args: argparse.Namespace, paths: ArtifactPaths) -> tuple[ad.AnnData, dict[str, Any]]:
    x = adata.X
    if sparse.issparse(x):
        expressed_counts = np.asarray((x > 0).sum(axis=0)).ravel()
    else:
        expressed_counts = np.sum(np.asarray(x) > 0, axis=0)

    min_cells = args.min_cells
    if min_cells is None:
        min_cells = max(1, int(math.ceil(float(args.min_cells_fraction) * adata.n_obs)))
    keep_mask = expressed_counts >= int(min_cells)
    kept = adata.var_names[keep_mask].astype(str).tolist()
    removed = adata.var_names[~keep_mask].astype(str).tolist()

    gene_filter = pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "expressed_cells": expressed_counts.astype(int),
            "kept_all_filtered_gene_universe": keep_mask,
        }
    )
    gene_filter.to_csv(paths.preprocessing_dir / "cohort_filter_manifest.csv", index=False)
    write_json(paths.gene_universe_dir / "all_filtered_gene_list.json", {"genes": kept, "n_genes": len(kept)})
    write_json(paths.gene_universe_dir / "genes_removed_min_cells.json", {"genes": removed, "n_genes": len(removed), "min_cells": int(min_cells)})

    LOGGER.info("Gene universe min-cells filter kept %d / %d genes", len(kept), adata.n_vars)
    return adata[:, kept].copy(), {"min_cells": int(min_cells), "n_genes_before": int(adata.n_vars), "n_genes_after": len(kept)}


def normalize_log1p(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    import scanpy as sc

    if args.counts_layer:
        if args.counts_layer not in adata.layers:
            raise ValueError(f"Requested counts layer {args.counts_layer!r} not found")
        adata = adata.copy()
        adata.X = adata.layers[args.counts_layer].copy()

    if args.skip_normalize_log1p:
        LOGGER.info("Skipping normalize_total/log1p by request")
        return adata.copy()

    out = adata.copy()
    sc.pp.normalize_total(out, target_sum=float(args.target_sum))
    sc.pp.log1p(out)
    return out


def rank_hvgs_by_variance(adata: ad.AnnData, paths: ArtifactPaths) -> list[str]:
    x = adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.power(2).mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        arr = np.asarray(x)
        var = np.var(arr, axis=0)

    order = np.argsort(-var)
    ranked = [str(adata.var_names[i]) for i in order]
    hvg_df = pd.DataFrame({"rank": np.arange(1, len(ranked) + 1), "gene": ranked, "variance": var[order]})
    hvg_df.to_csv(paths.gene_universe_dir / "hvg_ranked_genes.csv", index=False)
    write_json(paths.gene_universe_dir / "hvg_ranked_genes.json", {"ranking_method": "post_filter_log1p_variance", "genes": ranked})
    return ranked


def ordered_available_genes(candidates: list[str], available_upper_to_name: dict[str, str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for gene in candidates:
        upper = str(gene).upper()
        if upper in available_upper_to_name:
            actual = available_upper_to_name[upper]
            if actual not in seen:
                out.append(actual)
                seen.add(actual)
    return out


def make_panel(
    *,
    experiment_id: str,
    panel_id: str,
    panel_type: str,
    panel_family: str,
    panel_subfamily: str = "",
    interpretation_layer: str = "",
    priority: str = "",
    rationale: str = "",
    circularity_flag: str = "",
    source_dictionary: str,
    panel_scope: str,
    genes: list[str],
    raw_genes: list[str],
    missing_genes: list[str],
    gene_budget_type: str,
    matched_control_id: str = "",
    genesets: list[str] | None = None,
    description: str = "",
    gene_annotation_path: str = "",
) -> dict[str, Any]:
    covered_unique = sorted(dict.fromkeys(str(g) for g in genes))
    raw_unique = sorted(dict.fromkeys(str(g) for g in raw_genes))
    missing_unique = sorted(dict.fromkeys(str(g) for g in missing_genes))
    return {
        "experiment_id": experiment_id,
        "panel_id": panel_id,
        "panel_type": panel_type,
        "panel_family": panel_family,
        "panel_subfamily": panel_subfamily,
        "interpretation_layer": interpretation_layer,
        "priority": priority,
        "rationale": rationale,
        "circularity_flag": circularity_flag,
        "source_dictionary": source_dictionary,
        "panel_scope": panel_scope,
        "description": description,
        "genesets": genesets or [],
        "genes": covered_unique,
        "raw_genes": raw_unique,
        "missing_genes": missing_unique,
        "n_gene_sets": len(genesets or []),
        "n_raw_genes": len(raw_unique),
        "n_covered_genes": len(covered_unique),
        "n_missing_genes": len(missing_unique),
        "gene_budget_type": gene_budget_type,
        "matched_control_id": matched_control_id,
        "gene_annotation_path": gene_annotation_path,
        "eligible_for_dr": len(covered_unique) >= 2,
        "eligible_for_direct_gene": len(covered_unique) >= 1,
    }


def split_gene_string(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x)]
    return [x for x in str(value).replace(",", ";").split(";") if x]


def is_expanded_manifest(manifest: pd.DataFrame) -> bool:
    required = {"set_id", "family", "subfamily", "interpretation_layer"}
    return required.issubset(set(manifest.columns))


def write_panel_gene_annotations(
    paths: ArtifactPaths,
    experiment_dir: Path,
    panel_id: str,
    annotation_rows: list[dict[str, Any]],
) -> str:
    if not annotation_rows:
        return ""
    out = pd.DataFrame(annotation_rows).drop_duplicates()
    grouped = (
        out.groupby("gene", as_index=False)
        .agg(
            original_gene_sets=("set_id", lambda x: ";".join(sorted(set(map(str, x))))),
            source_collections=("source_collection", lambda x: ";".join(sorted(set(map(str, x))))),
            families=("family", lambda x: ";".join(sorted(set(map(str, x))))),
            subfamilies=("subfamily", lambda x: ";".join(sorted(set(map(str, x))))),
            interpretation_layers=("interpretation_layer", lambda x: ";".join(sorted(set(map(str, x))))),
            circularity_flags=("circularity_flag", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values("gene")
    )
    path = paths.panel_gene_annotations_dir / f"{panel_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(path, index=False)
    return str(path.relative_to(experiment_dir))


def build_expanded_stage0_panels(
    adata: ad.AnnData,
    hvg_ranked: list[str],
    args: argparse.Namespace,
    paths: ArtifactPaths,
    experiment_id: str,
    gmt: dict[str, list[str]],
    manifest: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    manifest = manifest.copy()
    if "geneset_name" not in manifest.columns:
        manifest["geneset_name"] = manifest["set_id"]
    if "why_include" not in manifest.columns:
        manifest["why_include"] = manifest.get("rationale", "")
    if "source" not in manifest.columns:
        manifest["source"] = manifest.get("source_collection", "")
    if "priority" not in manifest.columns:
        manifest["priority"] = "support"
    manifest["set_id"] = manifest["set_id"].astype(str)

    all_sets = manifest["set_id"].tolist()
    unknown_sets = sorted(set(all_sets) - set(gmt))
    if unknown_sets:
        raise ValueError(f"Expanded manifest sets missing from GMT: {unknown_sets[:20]} (n={len(unknown_sets)})")

    available_upper_to_name = {str(g).upper(): str(g) for g in adata.var_names}
    geneset_coverage: dict[str, dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        set_id = str(row["set_id"])
        raw_upper = gmt[set_id]
        covered = ordered_available_genes(raw_upper, available_upper_to_name)
        covered_upper = {g.upper() for g in covered}
        missing = sorted(set(raw_upper) - covered_upper)
        meta = row.to_dict()
        geneset_coverage[set_id] = {"raw": raw_upper, "covered": covered, "missing": missing, "meta": meta}
        coverage_rows.append(
            {
                "set_id": set_id,
                "original_name": meta.get("original_name", set_id),
                "family": meta.get("family", ""),
                "subfamily": meta.get("subfamily", ""),
                "interpretation_layer": meta.get("interpretation_layer", ""),
                "raw_n_genes": len(raw_upper),
                "covered_n_genes": len(covered),
                "missing_n_genes": len(missing),
                "hit_fraction": len(covered) / max(len(raw_upper), 1),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows).merge(manifest, on="set_id", how="left", suffixes=("", "_manifest"))
    coverage_df.sort_values(["family", "interpretation_layer", "subfamily", "set_id"]).to_csv(
        paths.gene_universe_dir / "expanded_stage0_coverage.csv", index=False
    )

    def collect_sets(sets: list[str]) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
        raw: list[str] = []
        covered: list[str] = []
        missing: list[str] = []
        annotation_rows: list[dict[str, Any]] = []
        for set_id in sets:
            info = geneset_coverage[set_id]
            meta = info["meta"]
            raw.extend(info["raw"])
            covered.extend(info["covered"])
            missing.extend(info["missing"])
            for gene in info["covered"]:
                annotation_rows.append(
                    {
                        "gene": gene,
                        "set_id": set_id,
                        "source_collection": meta.get("source_collection", meta.get("source", "")),
                        "family": meta.get("family", ""),
                        "subfamily": meta.get("subfamily", ""),
                        "interpretation_layer": meta.get("interpretation_layer", ""),
                        "circularity_flag": meta.get("circularity_flag", ""),
                    }
                )
        return covered, raw, missing, annotation_rows

    def make_expanded_panel(
        *,
        panel_id: str,
        panel_type: str,
        panel_family: str,
        panel_subfamily: str,
        interpretation_layer: str,
        gene_budget_type: str,
        genesets: list[str],
        description: str,
        priority: str = "",
        rationale: str = "",
        circularity_flag: str = "",
    ) -> dict[str, Any]:
        covered, raw, missing, annotation_rows = collect_sets(genesets)
        annotation_path = write_panel_gene_annotations(paths, paths.experiment_dir, panel_id, annotation_rows)
        return make_panel(
            experiment_id=experiment_id,
            panel_id=panel_id,
            panel_type=panel_type,
            panel_family=panel_family,
            panel_subfamily=panel_subfamily,
            interpretation_layer=interpretation_layer,
            priority=priority,
            rationale=rationale,
            circularity_flag=circularity_flag,
            source_dictionary="expanded_stage0_gene_sets",
            panel_scope=panel_type,
            genes=covered,
            raw_genes=raw,
            missing_genes=missing,
            gene_budget_type=gene_budget_type,
            genesets=genesets,
            description=description,
            gene_annotation_path=annotation_path,
        )

    panels: list[dict[str, Any]] = []
    biological_sets = manifest.loc[~manifest["interpretation_layer"].astype(str).eq("core_anchor"), "set_id"].tolist()
    panels.append(
        make_expanded_panel(
            panel_id="expanded_full",
            panel_type="full_control",
            panel_family="knowledge_expanded_stage0",
            panel_subfamily="all",
            interpretation_layer="expanded_union",
            gene_budget_type="knowledge_union",
            genesets=biological_sets or all_sets,
            description="Full union of expanded public-prior Stage 0 gene sets.",
        )
    )

    core_sets = manifest.loc[manifest["interpretation_layer"].astype(str).eq("core_anchor"), "set_id"].tolist()
    if core_sets:
        panels.append(
            make_expanded_panel(
                panel_id="expanded_core_anchor_only",
                panel_type="core_only",
                panel_family="knowledge_expanded_stage0",
                panel_subfamily="core_anchor",
                interpretation_layer="core_anchor",
                gene_budget_type="knowledge_core_anchor",
                genesets=core_sets,
                description="Union of expert-curated core-anchor sets for interpretation and sanity checks.",
            )
        )

    layer_to_panel_type = {
        "family_union": "family_union_sets",
        "core_anchor": "core_anchor_sets",
    }
    for _, row in manifest.iterrows():
        set_id = str(row["set_id"])
        layer = str(row.get("interpretation_layer", "atomic_sets"))
        if layer == "family_union":
            panel_type = "family_union_sets"
        elif layer == "core_anchor":
            panel_type = "core_anchor_sets"
        else:
            panel_type = layer_to_panel_type.get(layer, "atomic_sets")
        panels.append(
            make_expanded_panel(
                panel_id=safe_id(set_id, ),
                panel_type=panel_type,
                panel_family=str(row.get("family", "")),
                panel_subfamily=str(row.get("subfamily", "")),
                interpretation_layer=layer,
                gene_budget_type=f"knowledge_{panel_type}",
                genesets=[set_id],
                priority=str(row.get("priority", "")),
                rationale=str(row.get("rationale", row.get("why_include", ""))),
                circularity_flag=str(row.get("circularity_flag", "")),
                description=f"Expanded Stage 0 panel for resolved set: {set_id}.",
            )
        )

    for family, family_df in manifest.groupby("family", sort=True):
        family_sets = family_df.loc[~family_df["interpretation_layer"].astype(str).eq("core_anchor"), "set_id"].tolist()
        if not family_sets:
            continue
        panels.append(
            make_expanded_panel(
                panel_id=f"single_family__{safe_id(family)}",
                panel_type="single_group_only",
                panel_family=str(family),
                panel_subfamily="all_non_anchor",
                interpretation_layer="family_group",
                gene_budget_type="knowledge_family",
                genesets=family_sets,
                description=f"Union of all non-anchor expanded sets for biological family: {family}.",
            )
        )

    family_names = sorted(manifest["family"].dropna().astype(str).unique())
    for family in family_names:
        keep_sets = manifest.loc[
            (~manifest["family"].astype(str).eq(family)) & (~manifest["interpretation_layer"].astype(str).eq("core_anchor")),
            "set_id",
        ].tolist()
        if not keep_sets:
            continue
        panels.append(
            make_expanded_panel(
                panel_id=f"leave_one_family_out__without_{safe_id(family)}",
                panel_type="leave_one_family_out",
                panel_family="knowledge_expanded_stage0",
                panel_subfamily=f"without_{family}",
                interpretation_layer="fairness_control",
                gene_budget_type="leave_one_family_out",
                genesets=keep_sets,
                description=f"Expanded full union leaving out biological family: {family}.",
            )
        )

    all_filtered_genes = [str(g) for g in adata.var_names]
    panels.append(
        make_panel(
            experiment_id=experiment_id,
            panel_id="all_filtered_mrd_gene_universe",
            panel_type="all_filtered_control",
            panel_family="data_driven_all_filtered",
            panel_subfamily="all",
            interpretation_layer="control",
            source_dictionary="shared_mrd_preprocessing",
            panel_scope="all_filtered_gene_universe",
            genes=all_filtered_genes,
            raw_genes=all_filtered_genes,
            missing_genes=[],
            gene_budget_type="all_filtered",
            description="All genes remaining after shared MRD/CITE cell and gene filters.",
        )
    )

    available_upper_to_name = {str(g).upper(): str(g) for g in adata.var_names}
    hvg_available = ordered_available_genes(hvg_ranked, available_upper_to_name)
    for n in args.hvg_anchor_sizes:
        genes = hvg_available[: min(int(n), len(hvg_available))]
        panels.append(
            make_panel(
                experiment_id=experiment_id,
                panel_id=f"hvg_top_requested_{int(n)}__available_{len(genes)}",
                panel_type="hvg_anchor_control",
                panel_family="data_driven_hvg",
                panel_subfamily=f"top_{int(n)}",
                interpretation_layer="control",
                source_dictionary="shared_mrd_hvg_variance_rank",
                panel_scope="fixed_hvg_anchor",
                genes=genes,
                raw_genes=genes,
                missing_genes=[],
                gene_budget_type=f"hvg_anchor_{int(n)}",
                description=f"Top {len(genes)} available HVGs from the shared MRD/CITE gene universe.",
            )
        )

    for panel in panels:
        recommended = [k for k in args.ks if int(panel["n_covered_genes"]) > k]
        panel["recommended_k_values"] = recommended
        panel["eligible_for_dr"] = bool(recommended)
        gene_path = paths.panel_genes_dir / f"{panel['panel_id']}.json"
        panel["gene_list_path"] = str(gene_path.relative_to(paths.experiment_dir))
        write_json(
            gene_path,
            {
                "experiment_id": experiment_id,
                "panel_id": panel["panel_id"],
                "panel_type": panel["panel_type"],
                "panel_family": panel["panel_family"],
                "panel_subfamily": panel.get("panel_subfamily", ""),
                "interpretation_layer": panel.get("interpretation_layer", ""),
                "genes": panel["genes"],
                "raw_genes": panel["raw_genes"],
                "missing_genes": panel["missing_genes"],
                "genesets": panel["genesets"],
                "gene_annotation_path": panel.get("gene_annotation_path", ""),
            },
        )

    manifest_rows = []
    for panel in panels:
        row = {k: v for k, v in panel.items() if k not in {"genes", "raw_genes", "missing_genes", "genesets"}}
        row["genesets"] = ";".join(panel["genesets"])
        row["recommended_k_values"] = ",".join(str(k) for k in panel["recommended_k_values"])
        manifest_rows.append(row)

    panel_manifest = pd.DataFrame(manifest_rows).sort_values(["panel_type", "panel_family", "panel_id"])
    panel_manifest.to_csv(paths.panels_dir / "stage0_panel_manifest.csv", index=False)
    LOGGER.info("Wrote %d expanded Stage 0 panels", len(panel_manifest))
    return panels, panel_manifest


def build_stage0_panels(
    adata: ad.AnnData,
    hvg_ranked: list[str],
    args: argparse.Namespace,
    paths: ArtifactPaths,
    experiment_id: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    gmt = parse_gmt(args.gmt_path)
    manifest = pd.read_csv(args.old_manifest_path, sep="\t")
    if is_expanded_manifest(manifest):
        return build_expanded_stage0_panels(adata, hvg_ranked, args, paths, experiment_id, gmt, manifest)
    manifest["biology_group"] = manifest["why_include"].map(assign_biology_group)
    all_sets = manifest["geneset_name"].tolist()
    unknown_sets = sorted(set(all_sets) - set(gmt))
    if unknown_sets:
        raise ValueError(f"Manifest genesets missing from GMT: {unknown_sets}")

    available_upper_to_name = {str(g).upper(): str(g) for g in adata.var_names}
    geneset_coverage: dict[str, dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []
    for geneset_name in all_sets:
        raw_upper = gmt[geneset_name]
        covered = ordered_available_genes(raw_upper, available_upper_to_name)
        missing = sorted(set(raw_upper) - set(g.upper() for g in covered))
        geneset_coverage[geneset_name] = {"raw": raw_upper, "covered": covered, "missing": missing}
        coverage_rows.append(
            {
                "geneset_name": geneset_name,
                "raw_n_genes": len(raw_upper),
                "covered_n_genes": len(covered),
                "missing_n_genes": len(missing),
                "hit_fraction": len(covered) / max(len(raw_upper), 1),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows).merge(manifest, on="geneset_name", how="left")
    coverage_df.sort_values(["priority", "biology_group", "geneset_name"]).to_csv(paths.gene_universe_dir / "old34_coverage.csv", index=False)

    def collect_sets(sets: list[str]) -> tuple[list[str], list[str], list[str]]:
        raw: list[str] = []
        covered: list[str] = []
        missing: list[str] = []
        for geneset in sets:
            raw.extend(geneset_coverage[geneset]["raw"])
            covered.extend(geneset_coverage[geneset]["covered"])
            missing.extend(geneset_coverage[geneset]["missing"])
        return covered, raw, missing

    panels: list[dict[str, Any]] = []
    full_covered, full_raw, full_missing = collect_sets(all_sets)
    panels.append(
        make_panel(
            experiment_id=experiment_id,
            panel_id="full_34",
            panel_type="full_control",
            panel_family="knowledge_old_geneset",
            source_dictionary="old_34_programs",
            panel_scope="full_old34_union",
            genes=full_covered,
            raw_genes=full_raw,
            missing_genes=full_missing,
            gene_budget_type="knowledge_union",
            genesets=all_sets,
            description="Full union of the old 34-program dictionary.",
        )
    )

    core_sets = manifest.loc[manifest["priority"].astype(str).str.lower().eq("core"), "geneset_name"].tolist()
    core_covered, core_raw, core_missing = collect_sets(core_sets)
    panels.append(
        make_panel(
            experiment_id=experiment_id,
            panel_id="core_only",
            panel_type="core_only",
            panel_family="knowledge_old_geneset",
            source_dictionary="old_34_programs",
            panel_scope="core_old34_union",
            genes=core_covered,
            raw_genes=core_raw,
            missing_genes=core_missing,
            gene_budget_type="knowledge_union",
            genesets=core_sets,
            description="Manifest-priority Core old-34 genesets only.",
        )
    )

    for geneset in all_sets:
        info = geneset_coverage[geneset]
        panels.append(
            make_panel(
                experiment_id=experiment_id,
                panel_id=f"single_geneset__{safe_id(geneset)}",
                panel_type="single_geneset_only",
                panel_family="knowledge_old_geneset",
                source_dictionary="old_34_programs",
                panel_scope="single_geneset",
                genes=info["covered"],
                raw_genes=info["raw"],
                missing_genes=info["missing"],
                gene_budget_type="knowledge_single",
                genesets=[geneset],
                description=f"Bottom-up panel with only geneset: {geneset}.",
            )
        )

    for group, group_df in manifest.groupby("biology_group", sort=True):
        group_sets = sorted(group_df["geneset_name"])
        group_covered, group_raw, group_missing = collect_sets(group_sets)
        panels.append(
            make_panel(
                experiment_id=experiment_id,
                panel_id=f"single_group__{safe_id(group)}",
                panel_type="single_group_only",
                panel_family="knowledge_old_geneset",
                source_dictionary="old_34_programs",
                panel_scope="single_biology_group",
                genes=group_covered,
                raw_genes=group_raw,
                missing_genes=group_missing,
                gene_budget_type="knowledge_group",
                genesets=group_sets,
                description=f"Bottom-up panel with only biology group: {group}.",
            )
        )

    all_filtered_genes = [str(g) for g in adata.var_names]
    panels.append(
        make_panel(
            experiment_id=experiment_id,
            panel_id="all_filtered_mrd_gene_universe",
            panel_type="all_filtered_control",
            panel_family="data_driven_all_filtered",
            source_dictionary="shared_mrd_preprocessing",
            panel_scope="all_filtered_gene_universe",
            genes=all_filtered_genes,
            raw_genes=all_filtered_genes,
            missing_genes=[],
            gene_budget_type="all_filtered",
            description="All genes remaining after shared MRD/CITE cell and gene filters.",
        )
    )

    available_upper_to_name = {str(g).upper(): str(g) for g in adata.var_names}
    hvg_available = ordered_available_genes(hvg_ranked, available_upper_to_name)
    for n in args.hvg_anchor_sizes:
        genes = hvg_available[: min(int(n), len(hvg_available))]
        panels.append(
            make_panel(
                experiment_id=experiment_id,
                panel_id=f"hvg_top_requested_{int(n)}__available_{len(genes)}",
                panel_type="hvg_anchor_control",
                panel_family="data_driven_hvg",
                source_dictionary="shared_mrd_hvg_variance_rank",
                panel_scope="fixed_hvg_anchor",
                genes=genes,
                raw_genes=genes,
                missing_genes=[],
                gene_budget_type=f"hvg_anchor_{int(n)}",
                description=f"Top {len(genes)} available HVGs from the shared MRD/CITE gene universe.",
            )
        )

    for panel in panels:
        recommended = [k for k in args.ks if int(panel["n_covered_genes"]) > k]
        panel["recommended_k_values"] = recommended
        panel["eligible_for_dr"] = bool(recommended)
        gene_path = paths.panel_genes_dir / f"{panel['panel_id']}.json"
        panel["gene_list_path"] = str(gene_path.relative_to(paths.experiment_dir))
        write_json(
            gene_path,
            {
                "experiment_id": experiment_id,
                "panel_id": panel["panel_id"],
                "panel_type": panel["panel_type"],
                "genes": panel["genes"],
                "raw_genes": panel["raw_genes"],
                "missing_genes": panel["missing_genes"],
                "genesets": panel["genesets"],
            },
        )

    manifest_rows = []
    for panel in panels:
        row = {k: v for k, v in panel.items() if k not in {"genes", "raw_genes", "missing_genes", "genesets"}}
        row["genesets"] = ";".join(panel["genesets"])
        row["recommended_k_values"] = ",".join(str(k) for k in panel["recommended_k_values"])
        manifest_rows.append(row)

    panel_manifest = pd.DataFrame(manifest_rows).sort_values(["panel_type", "panel_id"])
    panel_manifest.to_csv(paths.panels_dir / "stage0_panel_manifest.csv", index=False)
    LOGGER.info("Wrote %d Stage 0 panels", len(panel_manifest))
    return panels, panel_manifest


def select_panels(panels: list[dict[str, Any]], panel_types: list[str], panel_ids: list[str]) -> list[dict[str, Any]]:
    selected = panels
    if panel_types:
        wanted = set(panel_types)
        selected = [p for p in selected if p["panel_type"] in wanted]
    if panel_ids:
        wanted_ids = set(panel_ids)
        selected = [p for p in selected if p["panel_id"] in wanted_ids]
    return selected


def get_dense_panel_matrix(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    sub = adata[:, genes].X
    if sparse.issparse(sub):
        arr = sub.toarray()
    else:
        arr = np.asarray(sub)
    return np.asarray(arr, dtype=np.float32)


def standardize_matrix(x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mean = np.nanmean(x, axis=0, dtype=np.float64)
    std = np.nanstd(x, axis=0, dtype=np.float64)
    zero_std = std <= 1e-12
    std_safe = std.copy()
    std_safe[zero_std] = 1.0
    x = (x - mean.astype(np.float32)) / std_safe.astype(np.float32)
    x = np.nan_to_num(x, copy=False).astype(np.float32, copy=False)
    return x, {
        "standardization": "feature_zscore_with_zero_variance_to_zero",
        "n_zero_variance_genes": int(np.sum(zero_std)),
    }


def make_panel_adata(adata: ad.AnnData, panel: dict[str, Any]) -> tuple[ad.AnnData, dict[str, Any]]:
    genes = [g for g in panel["genes"] if g in adata.var_names]
    x = get_dense_panel_matrix(adata, genes)
    x, std_info = standardize_matrix(x)
    out = ad.AnnData(X=x, obs=adata.obs.copy(), var=pd.DataFrame(index=pd.Index(genes, name=adata.var_names.name)))
    return out, std_info


def write_top_loading_gene_annotations(
    loadings: np.ndarray,
    genes: list[str],
    panel: dict[str, Any],
    artifact_dir: Path,
    experiment_dir: Path,
    top_n: int = 50,
) -> str:
    rows: list[dict[str, Any]] = []
    annotations = pd.DataFrame({"gene": genes})
    rel_annotation_path = panel.get("gene_annotation_path", "")
    if rel_annotation_path:
        annotation_path = experiment_dir / rel_annotation_path
        if annotation_path.exists():
            annotations = annotations.merge(pd.read_csv(annotation_path), on="gene", how="left")
    for factor_idx in range(loadings.shape[1]):
        values = np.asarray(loadings[:, factor_idx], dtype=float)
        order = np.argsort(-np.abs(values))[: min(top_n, len(values))]
        for rank, gene_idx in enumerate(order, start=1):
            rows.append(
                {
                    "factor_id": f"factor_{factor_idx + 1:03d}",
                    "rank_abs_loading": rank,
                    "gene": genes[gene_idx],
                    "loading": float(values[gene_idx]),
                    "abs_loading": float(abs(values[gene_idx])),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty and not annotations.empty:
        out = out.merge(annotations, on="gene", how="left")
    path = artifact_dir / "top_loading_genes.csv"
    out.to_csv(path, index=False)
    return str(path.relative_to(experiment_dir))


def run_dr_representation(
    panel_adata: ad.AnnData,
    panel: dict[str, Any],
    method: str,
    k: int,
    seed: int,
    paths: ArtifactPaths,
    args: argparse.Namespace,
    std_info: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    from run_gene_filter_dr_grid import _run_dr_method

    artifact_dir = paths.stage1_dr_dir / f"panel_id={panel['panel_id']}" / f"method={method}" / f"k={k}" / f"seed={seed}"
    scores_path = artifact_dir / "scores.npy"
    loadings_path = artifact_dir / "loadings.npy"
    metadata_path = artifact_dir / "metadata.json"
    diagnostics_path = artifact_dir / "diagnostics.json"

    base_meta = {
        "representation_family": "dr",
        "stage1_scope": args.stage1_scope,
        "panel_id": panel["panel_id"],
        "stage1_method": method,
        "requested_k": int(k),
        "effective_k": int(k),
        "seed": int(seed),
        "n_cells": int(panel_adata.n_obs),
        "n_genes": int(panel_adata.n_vars),
        "scores_path": str(scores_path.relative_to(paths.experiment_dir)),
        "loadings_path": str(loadings_path.relative_to(paths.experiment_dir)),
        "metadata_path": str(metadata_path.relative_to(paths.experiment_dir)),
        "diagnostics_path": str(diagnostics_path.relative_to(paths.experiment_dir)),
        "gene_annotation_path": panel.get("gene_annotation_path", ""),
        **std_info,
    }

    if panel_adata.n_vars <= k:
        base_meta.update({"status": "skipped_too_few_genes", "reason": f"n_genes={panel_adata.n_vars} <= k={k}"})
        return None, base_meta
    if panel_adata.n_obs <= k:
        base_meta.update({"status": "skipped_too_few_cells", "reason": f"n_cells={panel_adata.n_obs} <= k={k}"})
        return None, base_meta

    if scores_path.exists() and loadings_path.exists() and metadata_path.exists() and not args.rerun:
        scores = np.load(scores_path)
        meta = json.loads(metadata_path.read_text())
        meta["status"] = "ok"
        meta["reused_existing_artifact"] = True
        return scores, meta

    artifact_dir.mkdir(parents=True, exist_ok=True)
    scores, loadings, extras = _run_dr_method(
        method=method,
        adata=panel_adata,
        k=k,
        seed=seed,
        factosig_max_iter=args.factosig_max_iter,
        factosig_rotation=args.factosig_rotation,
    )
    np.save(scores_path, scores.astype(np.float32, copy=False))
    np.save(loadings_path, loadings.astype(np.float32, copy=False))
    top_loading_path = write_top_loading_gene_annotations(
        loadings=loadings,
        genes=[str(g) for g in panel_adata.var_names],
        panel=panel,
        artifact_dir=artifact_dir,
        experiment_dir=paths.experiment_dir,
    )
    diagnostics = {
        "score_variance": np.var(scores, axis=0).tolist(),
        "loading_ss": np.sum(np.square(loadings), axis=0).tolist(),
        "extras": extras,
    }
    meta = {**base_meta, "status": "ok", "reused_existing_artifact": False, "top_loading_genes_path": top_loading_path}
    write_json(metadata_path, meta)
    write_json(diagnostics_path, diagnostics)
    return scores, meta


def make_logreg_pipeline(seed: int) -> Pipeline:
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
                    random_state=seed,
                ),
            ),
        ]
    )


def binary_labels(obs: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    labels = obs[args.target_col].astype(str)
    y = np.full(labels.shape[0], -1, dtype=int)
    y[labels.eq(args.negative_class).to_numpy()] = 0
    y[labels.eq(args.positive_class).to_numpy()] = 1
    if np.any(y < 0):
        raise ValueError("Unexpected labels after preprocessing")
    return y


def safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    out: dict[str, Any] = {
        "n_eval_cells": int(len(y_true)),
        "stage2_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "stage2_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) >= 2:
        out["stage2_auroc"] = float(roc_auc_score(y_true, y_prob))
        out["stage2_auprc"] = float(average_precision_score(y_true, y_prob))
        clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
        out["stage2_log_loss"] = float(log_loss(y_true, clipped))
    else:
        out["stage2_auroc"] = np.nan
        out["stage2_auprc"] = np.nan
        out["stage2_log_loss"] = np.nan
    return out


def quick_shared_stage2(
    features: np.ndarray,
    obs: pd.DataFrame,
    rep_meta: dict[str, Any],
    paths: ArtifactPaths,
    args: argparse.Namespace,
) -> dict[str, Any]:
    y = binary_labels(obs, args)
    groups = obs[args.patient_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    out_dir = (
        paths.stage2_shared_dir
        / f"panel_id={rep_meta['panel_id']}"
        / f"representation={rep_meta['representation_family']}"
        / f"method={rep_meta.get('stage1_method', rep_meta['representation_family'])}"
        / f"k={rep_meta.get('requested_k', 'NA')}"
        / f"seed={rep_meta['seed']}"
    )
    metrics_path = out_dir / "quick_l2_groupkfold_metrics.json"
    predictions_path = out_dir / "quick_l2_groupkfold_predictions.csv"

    stage2_base = {
        "stage2_mode": "shared_cross_patient",
        "classifier": "logistic_regression",
        "penalty": "l2",
        "C": 1.0,
        "l1_ratio": np.nan,
        "split_policy": "GroupKFold_by_patient",
        "stage2_metrics_path": str(metrics_path.relative_to(paths.experiment_dir)),
        "stage2_predictions_path": str(predictions_path.relative_to(paths.experiment_dir)),
    }

    if len(unique_groups) < 2:
        return {**stage2_base, "stage2_status": "skipped_too_few_groups", "stage2_reason": "Need at least two patients"}
    if len(np.unique(y)) < 2:
        return {**stage2_base, "stage2_status": "skipped_one_class", "stage2_reason": "Need both malignant and normal labels"}

    n_splits = min(int(args.cv_folds), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    y_prob = np.full(y.shape[0], np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, y, groups), start=1):
        train_classes = np.unique(y[train_idx])
        test_classes = np.unique(y[test_idx])
        if len(train_classes) < 2 or len(test_classes) < 1:
            fold_rows.append({"fold": fold, "status": "skipped_one_class", "n_train": len(train_idx), "n_test": len(test_idx)})
            continue
        clf = make_logreg_pipeline(rep_meta["seed"])
        clf.fit(features[train_idx], y[train_idx])
        y_prob[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
        fold_rows.append(
            {
                "fold": fold,
                "status": "ok",
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "heldout_patients": ";".join(sorted(set(groups[test_idx]))),
            }
        )

    valid = np.isfinite(y_prob)
    if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
        return {**stage2_base, "stage2_status": "skipped_no_valid_folds", "stage2_reason": "No valid GroupKFold predictions"}

    metrics = {
        **stage2_base,
        **safe_binary_metrics(y[valid], y_prob[valid]),
        "stage2_status": "ok",
        "n_splits": int(n_splits),
        "n_valid_folds": int(sum(row["status"] == "ok" for row in fold_rows)),
        "folds": fold_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = obs[[args.patient_col, args.target_col]].copy()
    pred_df["y_true"] = y
    pred_df["y_prob"] = y_prob
    pred_df["included_in_metric"] = valid
    pred_df.to_csv(predictions_path, index=True)
    write_json(metrics_path, metrics)
    return metrics


def write_direct_gene_representation(
    panel_adata: ad.AnnData,
    panel: dict[str, Any],
    paths: ArtifactPaths,
    args: argparse.Namespace,
    std_info: dict[str, Any],
    representation_family: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    transform = "zscore"
    if representation_family == "summary_score":
        features = np.asarray(panel_adata.X.mean(axis=1)).reshape(-1, 1).astype(np.float32)
        feature_names = [f"{panel['panel_id']}__mean_zscore"]
        out_dir = paths.stage1_direct_gene_dir / f"panel_id={panel['panel_id']}" / "transform=summary_mean_zscore"
    else:
        features = np.asarray(panel_adata.X, dtype=np.float32)
        feature_names = [str(g) for g in panel_adata.var_names]
        out_dir = paths.stage1_direct_gene_dir / f"panel_id={panel['panel_id']}" / f"transform={transform}"

    features_path = out_dir / "features.npz"
    feature_names_path = out_dir / "feature_names.json"
    metadata_path = out_dir / "metadata.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(features_path, sparse.csr_matrix(features))
    write_json(feature_names_path, feature_names)
    meta = {
        "representation_family": representation_family,
        "stage1_scope": args.stage1_scope,
        "panel_id": panel["panel_id"],
        "stage1_method": representation_family,
        "requested_k": np.nan,
        "effective_k": int(features.shape[1]),
        "seed": int(args.seed),
        "n_cells": int(features.shape[0]),
        "n_genes": int(panel_adata.n_vars),
        "status": "ok",
        "features_path": str(features_path.relative_to(paths.experiment_dir)),
        "feature_names_path": str(feature_names_path.relative_to(paths.experiment_dir)),
        "metadata_path": str(metadata_path.relative_to(paths.experiment_dir)),
        "gene_annotation_path": panel.get("gene_annotation_path", ""),
        **std_info,
    }
    write_json(metadata_path, meta)
    return features, meta


def scorecard_row(
    experiment_id: str,
    panel: dict[str, Any],
    rep_meta: dict[str, Any],
    stage2_meta: dict[str, Any] | None,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    row = {
        "experiment_id": experiment_id,
        "modeling_goal": "broad_stage0_panel_ranking",
        "question_id": "Q1_shared_cross_patient_screen",
        "patient_scope": "across_patient",
        "stage0_panel_id": panel["panel_id"],
        "stage0_panel_family": panel["panel_family"],
        "stage0_panel_subfamily": panel.get("panel_subfamily", ""),
        "stage0_panel_type": panel["panel_type"],
        "interpretation_layer": panel.get("interpretation_layer", ""),
        "priority": panel.get("priority", ""),
        "rationale": panel.get("rationale", ""),
        "circularity_flag": panel.get("circularity_flag", ""),
        "source_dictionary": panel["source_dictionary"],
        "n_raw_genes": int(panel["n_raw_genes"]),
        "n_covered_genes": int(panel["n_covered_genes"]),
        "n_missing_genes": int(panel["n_missing_genes"]),
        "gene_budget_type": panel["gene_budget_type"],
        "matched_control_id": panel["matched_control_id"],
        "representation_family": rep_meta.get("representation_family", ""),
        "stage1_method": rep_meta.get("stage1_method", ""),
        "requested_k": rep_meta.get("requested_k", np.nan),
        "effective_k": rep_meta.get("effective_k", np.nan),
        "stage1_scope": rep_meta.get("stage1_scope", ""),
        "seed": rep_meta.get("seed", np.nan),
        "stage1_seed": rep_meta.get("seed", np.nan),
        "status": status,
        "reason": reason or rep_meta.get("reason", ""),
        "scores_path": rep_meta.get("scores_path", ""),
        "loadings_path": rep_meta.get("loadings_path", ""),
        "top_loading_genes_path": rep_meta.get("top_loading_genes_path", ""),
        "features_path": rep_meta.get("features_path", ""),
        "feature_names_path": rep_meta.get("feature_names_path", ""),
        "stage1_metadata_path": rep_meta.get("metadata_path", ""),
        "gene_list_path": panel["gene_list_path"],
        "gene_annotation_path": panel.get("gene_annotation_path", rep_meta.get("gene_annotation_path", "")),
    }
    if stage2_meta:
        row.update({k: v for k, v in stage2_meta.items() if k != "folds"})
    return row


def write_report(experiment_id: str, paths: ArtifactPaths, args: argparse.Namespace) -> Path:
    scorecard_path = paths.scorecards_dir / "stage0_mrd_old34_broad_scorecard.csv"
    panel_manifest_path = paths.panels_dir / "stage0_panel_manifest.csv"
    report_path = paths.reports_dir / "postrun_human_review.md"
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    parts = [
        f"# Stage 0 MRD Old-34 Broad Screen Review\n",
        f"- Experiment ID: `{experiment_id}`",
        f"- Scorecard: `{scorecard_path.relative_to(paths.experiment_dir)}`",
        f"- Panel manifest: `{panel_manifest_path.relative_to(paths.experiment_dir)}`",
        "",
    ]
    if scorecard_path.exists():
        df = pd.read_csv(scorecard_path)
        ok = df[df["stage2_status"].eq("ok")] if "stage2_status" in df else pd.DataFrame()
        parts.append(f"## Run Summary\n")
        parts.append(f"- Scorecard rows: {len(df)}")
        parts.append(f"- Stage 2 OK rows: {len(ok)}")
        if not ok.empty and "stage2_auroc" in ok:
            top = ok.sort_values(["stage2_auroc", "stage2_auprc"], ascending=False).head(int(args.report_top_n))
            parts.append("")
            parts.append("## Top Quick Shared Cross-Patient Rows")
            parts.append("")
            cols = [
                "stage0_panel_id",
                "stage0_panel_type",
                "representation_family",
                "stage1_method",
                "requested_k",
                "stage2_auroc",
                "stage2_auprc",
                "stage2_balanced_accuracy",
                "n_covered_genes",
            ]
            cols = [c for c in cols if c in top.columns]
            parts.append("```text")
            parts.append(top[cols].to_string(index=False))
            parts.append("```")
    parts.extend(
        [
            "",
            "## Human Review Gate",
            "",
            "Before launching the full downstream Stage 2 regularization benchmark, review:",
            "",
            "- Which single geneset and biology-group panels deserve shortlisting.",
            "- Whether biologically important panels should be force-included even if not top-ranked.",
            "- Which exact size-matched HVG controls are needed for fair follow-up comparisons.",
            "- Whether direct-gene small-panel rows should be trusted, ignored, or treated separately.",
            "- Whether this broad quick Stage 2 screen shows enough signal for the full L1/L2/elastic-net benchmark.",
            "",
        ]
    )
    report_path.write_text("\n".join(parts))
    return report_path


def run_screen(args: argparse.Namespace, paths: ArtifactPaths, experiment_id: str) -> pd.DataFrame:
    add_import_paths(args.sc_root)
    LOGGER.info("Loading cohort AnnData: %s", args.input_h5ad)
    adata_raw = ad.read_h5ad(args.input_h5ad)
    adata_cells, filter_df = filter_cells(adata_raw, args)
    filter_df.to_csv(paths.preprocessing_dir / "cell_filter_manifest.csv", index=False)
    del adata_raw

    adata_gene_filtered, gene_filter_info = filter_gene_universe(adata_cells, args, paths)
    del adata_cells
    adata_norm = normalize_log1p(adata_gene_filtered, args)
    del adata_gene_filtered

    hvg_ranked = rank_hvgs_by_variance(adata_norm, paths)
    panels, _panel_manifest = build_stage0_panels(adata_norm, hvg_ranked, args, paths, experiment_id)
    panels_to_run = select_panels(panels, args.panel_types, args.panel_ids)
    if args.max_panels:
        panels_to_run = panels_to_run[: int(args.max_panels)]
    LOGGER.info("Selected %d panels for Stage 1/quick Stage 2", len(panels_to_run))

    rows: list[dict[str, Any]] = []
    partial_scorecard = paths.scorecards_dir / "stage0_mrd_old34_broad_scorecard.partial.csv"
    total = len(panels_to_run) * max(1, len(args.methods)) * max(1, len(args.ks))
    completed = 0

    for panel in panels_to_run:
        LOGGER.info("Materializing standardized panel matrix: %s (%d genes)", panel["panel_id"], panel["n_covered_genes"])
        if panel["n_covered_genes"] == 0:
            rows.append(scorecard_row(experiment_id, panel, {}, None, "skipped_no_covered_genes", "No covered genes"))
            continue
        try:
            panel_adata, std_info = make_panel_adata(adata_norm, panel)
        except Exception as exc:
            LOGGER.exception("Failed to build panel matrix: %s", panel["panel_id"])
            rows.append(scorecard_row(experiment_id, panel, {}, None, "failed_panel_matrix", repr(exc)))
            continue

        invalid_k_exists = any(panel_adata.n_vars <= k for k in args.ks)
        for method in args.methods:
            for k in args.ks:
                completed += 1
                label = f"[{completed}/{total}] {panel['panel_id']} | {method} | k={k}"
                LOGGER.info("Starting %s", label)
                try:
                    scores, rep_meta = run_dr_representation(panel_adata, panel, method, k, args.seed, paths, args, std_info)
                    if scores is None:
                        rows.append(scorecard_row(experiment_id, panel, rep_meta, None, rep_meta["status"], rep_meta.get("reason", "")))
                        continue
                    stage2_meta = quick_shared_stage2(scores, panel_adata.obs, rep_meta, paths, args) if args.quick_stage2 else None
                    rows.append(scorecard_row(experiment_id, panel, rep_meta, stage2_meta, "ok"))
                    LOGGER.info("Finished %s", label)
                except Exception as exc:
                    LOGGER.exception("Failed %s", label)
                    rep_meta = {
                        "representation_family": "dr",
                        "stage1_method": method,
                        "requested_k": int(k),
                        "effective_k": np.nan,
                        "stage1_scope": args.stage1_scope,
                        "seed": int(args.seed),
                    }
                    rows.append(scorecard_row(experiment_id, panel, rep_meta, None, "failed", repr(exc)))
                pd.DataFrame(rows).to_csv(partial_scorecard, index=False)

        if args.small_panel_policy in {"direct_gene", "direct_gene_and_summary"} and invalid_k_exists:
            try:
                features, rep_meta = write_direct_gene_representation(panel_adata, panel, paths, args, std_info, "direct_gene")
                stage2_meta = quick_shared_stage2(features, panel_adata.obs, rep_meta, paths, args) if args.quick_stage2 else None
                rows.append(scorecard_row(experiment_id, panel, rep_meta, stage2_meta, "ok"))
            except Exception as exc:
                LOGGER.exception("Failed direct-gene representation for %s", panel["panel_id"])
                rows.append(scorecard_row(experiment_id, panel, {"representation_family": "direct_gene", "seed": args.seed}, None, "failed", repr(exc)))
            if args.small_panel_policy == "direct_gene_and_summary":
                try:
                    features, rep_meta = write_direct_gene_representation(panel_adata, panel, paths, args, std_info, "summary_score")
                    stage2_meta = quick_shared_stage2(features, panel_adata.obs, rep_meta, paths, args) if args.quick_stage2 else None
                    rows.append(scorecard_row(experiment_id, panel, rep_meta, stage2_meta, "ok"))
                except Exception as exc:
                    LOGGER.exception("Failed summary representation for %s", panel["panel_id"])
                    rows.append(scorecard_row(experiment_id, panel, {"representation_family": "summary_score", "seed": args.seed}, None, "failed", repr(exc)))
            pd.DataFrame(rows).to_csv(partial_scorecard, index=False)

        del panel_adata

    scorecard = pd.DataFrame(rows)
    scorecard_path = paths.scorecards_dir / "stage0_mrd_old34_broad_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)
    LOGGER.info("Wrote scorecard: %s", scorecard_path)
    write_json(paths.preprocessing_dir / "preprocessing_summary.json", {"gene_filter": gene_filter_info, "n_cells": int(adata_norm.n_obs), "n_genes": int(adata_norm.n_vars)})
    return scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT_H5AD)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--experiment-dir", type=Path, default=None, help="Existing experiment directory to reuse with a branch namespace.")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--branch-name", default="", help="Namespace for branch outputs under an existing experiment directory.")
    parser.add_argument("--sc-root", type=Path, default=DEFAULT_SC_ROOT)
    parser.add_argument("--gmt-path", type=Path, default=DEFAULT_GMT)
    parser.add_argument("--old-manifest-path", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--timepoint", default="MRD")
    parser.add_argument("--timepoint-col", default="timepoint_type")
    parser.add_argument("--tech", default="CITE")
    parser.add_argument("--tech-col", default="Tech")
    parser.add_argument("--target-col", default="CN.label")
    parser.add_argument("--positive-class", default="cancer")
    parser.add_argument("--negative-class", default="normal")
    parser.add_argument("--patient-col", default="patient")
    parser.add_argument("--counts-layer", default=None)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--skip-normalize-log1p", action="store_true")
    parser.add_argument("--min-cells", type=int, default=None)
    parser.add_argument("--min-cells-fraction", type=float, default=0.01)
    parser.add_argument("--panel-types", default="single_geneset_only,single_group_only,full_control,core_only,hvg_anchor_control")
    parser.add_argument("--panel-ids", default="")
    parser.add_argument("--hvg-anchor-sizes", default="500,1000,3000,10000")
    parser.add_argument("--methods", default="pca,fa,factosig,factosig_promax")
    parser.add_argument("--ks", default="5,10,20,40")
    parser.add_argument("--stage1-scope", choices=["across_patient"], default="across_patient")
    parser.add_argument("--small-panel-policy", choices=["skip", "direct_gene", "direct_gene_and_summary"], default="direct_gene")
    parser.add_argument("--quick-stage2", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factosig-max-iter", type=int, default=300)
    parser.add_argument("--factosig-rotation", default="varimax")
    parser.add_argument("--max-panels", type=int, default=None, help="Debug/smoke limit on the selected panel list.")
    parser.add_argument("--report-top-n", type=int, default=25)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Create config/manifests only; skip Stage 1/2 execution.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.input_h5ad = args.input_h5ad.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if args.experiment_dir is not None:
        args.experiment_dir = args.experiment_dir.expanduser().resolve()
    args.sc_root = args.sc_root.expanduser().resolve()
    args.gmt_path = args.gmt_path.expanduser().resolve()
    args.old_manifest_path = args.old_manifest_path.expanduser().resolve()
    args.panel_types = split_csv(args.panel_types)
    args.panel_ids = split_csv(args.panel_ids)
    args.hvg_anchor_sizes = split_csv(args.hvg_anchor_sizes, int)
    args.methods = split_csv(args.methods)
    args.ks = split_csv(args.ks, int)
    return args


def main() -> None:
    parser = build_parser()
    args = normalize_args(parser.parse_args())

    branch_name = safe_id(args.branch_name) if args.branch_name else ""
    if args.experiment_dir is not None:
        experiment_dir = args.experiment_dir
        experiment_id = args.experiment_id or (f"{experiment_dir.name}__{branch_name}" if branch_name else experiment_dir.name)
    else:
        experiment_id = args.experiment_id or make_experiment_id("stage0_mrd_old34_broad_screen")
        experiment_dir = args.out_root / experiment_id
        if experiment_dir.exists() and not args.rerun:
            raise FileExistsError(f"Experiment directory already exists. Pass --rerun to reuse: {experiment_dir}")
    paths = make_paths(experiment_dir, branch_name)
    existing_scorecard = paths.scorecards_dir / "stage0_mrd_old34_broad_scorecard.csv"
    if branch_name and existing_scorecard.exists() and not args.rerun:
        raise FileExistsError(f"Branch scorecard already exists. Pass --rerun to reuse: {existing_scorecard}")
    ensure_dirs(paths)
    log_path = configure_logging(paths.logs_dir, args.verbose)
    config_path = (paths.reports_dir / "experiment_config.yaml") if branch_name else (paths.experiment_dir / "experiment_config.yaml")
    run_manifest_path = (paths.reports_dir / "run_manifest.json") if branch_name else (paths.experiment_dir / "run_manifest.json")

    config = {
        "experiment_id": experiment_id,
        "branch_name": branch_name,
        "created_at": datetime.now().isoformat(),
        "purpose": "Broad Stage 0 old-34 and HVG-anchor screen from original cohort AnnData.",
        "input_h5ad": str(args.input_h5ad),
        "timepoint": args.timepoint,
        "tech": args.tech,
        "target": {"column": args.target_col, "positive_class": args.positive_class, "negative_class": args.negative_class},
        "patient_col": args.patient_col,
        "preprocessing": {
            "min_cells": args.min_cells,
            "min_cells_fraction": args.min_cells_fraction,
            "normalization": "normalize_total_log1p" if not args.skip_normalize_log1p else "input_matrix_as_is",
            "target_sum": args.target_sum,
            "standardization_policy": "within_panel_feature_zscore_before_dr_or_direct_gene",
            "hvg_ranking": "post_filter_log1p_variance",
        },
        "stage0": {
            "panel_types_to_run": args.panel_types,
            "hvg_anchor_sizes": args.hvg_anchor_sizes,
            "include_hucira": False,
            "dictionary": str(args.gmt_path),
            "old_manifest": str(args.old_manifest_path),
        },
        "stage1": {
            "scope": args.stage1_scope,
            "methods": args.methods,
            "ks": args.ks,
            "small_panel_policy": args.small_panel_policy,
            "seed": args.seed,
        },
        "stage2": {"quick_stage2": bool(args.quick_stage2), "split_policy": "GroupKFold_by_patient", "cv_folds": args.cv_folds},
        "code": {"sc_root": str(args.sc_root), "git_commit": get_git_commit(args.sc_root)},
    }
    write_yaml(config_path, config)
    source_meta = file_metadata(args.input_h5ad)
    if not branch_name:
        write_json(paths.experiment_dir / "input_source.json", source_meta)
    write_json(paths.preprocessing_dir / "input_source.json", source_meta)
    write_json(
        run_manifest_path,
        {
            "experiment_id": experiment_id,
            "branch_name": branch_name,
            "status": "started",
            "started_at": datetime.now().isoformat(),
            "log_path": str(log_path.relative_to(paths.experiment_dir)),
            "paths": {k: str(Path(v).relative_to(paths.experiment_dir)) for k, v in asdict(paths).items() if Path(v) != paths.experiment_dir},
            "config_path": str(config_path.relative_to(paths.experiment_dir)),
        },
    )

    try:
        if args.dry_run:
            LOGGER.info("Dry-run requested; building preprocessing and panel manifests only.")
            add_import_paths(args.sc_root)
            adata_raw = ad.read_h5ad(args.input_h5ad)
            adata_cells, filter_df = filter_cells(adata_raw, args)
            filter_df.to_csv(paths.preprocessing_dir / "cell_filter_manifest.csv", index=False)
            adata_gene_filtered, gene_filter_info = filter_gene_universe(adata_cells, args, paths)
            adata_norm = normalize_log1p(adata_gene_filtered, args)
            hvg_ranked = rank_hvgs_by_variance(adata_norm, paths)
            build_stage0_panels(adata_norm, hvg_ranked, args, paths, experiment_id)
            scorecard = pd.DataFrame()
            write_json(paths.preprocessing_dir / "preprocessing_summary.json", {"gene_filter": gene_filter_info, "n_cells": int(adata_norm.n_obs), "n_genes": int(adata_norm.n_vars)})
        else:
            scorecard = run_screen(args, paths, experiment_id)

        report_path = write_report(experiment_id, paths, args)
        write_json(
            run_manifest_path,
            {
                "experiment_id": experiment_id,
                "branch_name": branch_name,
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "log_path": str(log_path.relative_to(paths.experiment_dir)),
                "scorecard_path": str((paths.scorecards_dir / "stage0_mrd_old34_broad_scorecard.csv").relative_to(paths.experiment_dir)),
                "report_path": str(report_path.relative_to(paths.experiment_dir)),
                "n_scorecard_rows": int(len(scorecard)),
            },
        )
        LOGGER.info("Completed experiment: %s", paths.experiment_dir)
    except Exception:
        LOGGER.exception("Run failed")
        write_json(
            run_manifest_path,
            {
                "experiment_id": experiment_id,
                "branch_name": branch_name,
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "log_path": str(log_path.relative_to(paths.experiment_dir)),
            },
        )
        raise


if __name__ == "__main__":
    main()
