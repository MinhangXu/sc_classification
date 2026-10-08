#!/usr/bin/env python3
"""Build a two-set Stage 0 GMT from existing covered gene lists.

Reads the already-materialized old34 `full_34` and expanded `expanded_full`
gene JSONs (no GMT rediscovery) and writes:

- pooled_unions.gmt
- pooled_unions_manifest.tsv
- qc/overlap_summary.json
- qc/overlap_summary.csv
- qc/old34_only_genes.csv
- qc/expanded_only_genes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_DIR = Path(
    "/home/minhang/mds_project/sc_classification/experiments/"
    "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
)
DEFAULT_OLD34_JSON = (
    DEFAULT_EXPERIMENT_DIR / "preprocessing" / "panels" / "stage0_panel_genes" / "full_34.json"
)
DEFAULT_EXPANDED_JSON = (
    DEFAULT_EXPERIMENT_DIR
    / "preprocessing"
    / "panels"
    / "expanded_stage0_mrd_manuscript_axes_v1"
    / "stage0_panel_genes"
    / "expanded_full.json"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "pooled_stage0_unions"

OLD34_SET_ID = "old34_covered_union"
EXPANDED_SET_ID = "expanded_covered_union"
FAMILY = "pooled_stage0_unions"


def load_gene_list(path: Path) -> list[str]:
    obj: dict[str, Any] = json.loads(path.read_text())
    genes = obj.get("genes")
    if not isinstance(genes, list) or not genes:
        raise ValueError(f"No genes list in {path}")
    seen: set[str] = set()
    out: list[str] = []
    for gene in genes:
        symbol = str(gene).strip()
        if not symbol:
            continue
        key = symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(symbol)
    return out


def write_gmt(path: Path, sets: list[tuple[str, str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for set_id, description, genes in sets:
        lines.append("\t".join([set_id, description, *genes]))
    path.write_text("\n".join(lines) + "\n")


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "set_id",
        "source_collection",
        "source_url",
        "source_version",
        "original_name",
        "family",
        "subfamily",
        "interpretation_layer",
        "priority",
        "rationale",
        "genes_raw",
        "genes_present",
        "n_raw",
        "n_present",
        "circularity_flag",
        "geneset_name",
        "source",
        "why_include",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_gene_csv(path: Path, genes: list[str], column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column])
        for gene in genes:
            writer.writerow([gene])


def manifest_row(
    *,
    set_id: str,
    subfamily: str,
    genes: list[str],
    source_url: str,
    original_name: str,
    rationale: str,
) -> dict[str, Any]:
    joined = ";".join(genes)
    return {
        "set_id": set_id,
        "source_collection": "existing_stage0_covered_gene_lists",
        "source_url": source_url,
        "source_version": "covered_lists_from_20260525_experiment",
        "original_name": original_name,
        "family": FAMILY,
        "subfamily": subfamily,
        "interpretation_layer": "atomic_sets",
        "priority": "core",
        "rationale": rationale,
        "genes_raw": joined,
        "genes_present": joined,
        "n_raw": len(genes),
        "n_present": len(genes),
        "circularity_flag": "public_prior_union",
        "geneset_name": set_id,
        "source": "existing_stage0_covered_gene_lists",
        "why_include": rationale,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old34-gene-json", type=Path, default=DEFAULT_OLD34_JSON)
    parser.add_argument("--expanded-gene-json", type=Path, default=DEFAULT_EXPANDED_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    old34_path = args.old34_gene_json.expanduser().resolve()
    expanded_path = args.expanded_gene_json.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    qc_dir = out_dir / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    old34_genes = load_gene_list(old34_path)
    expanded_genes = load_gene_list(expanded_path)
    old34_upper = {g.upper(): g for g in old34_genes}
    expanded_upper = {g.upper(): g for g in expanded_genes}

    overlap_upper = sorted(set(old34_upper) & set(expanded_upper))
    old34_only_upper = sorted(set(old34_upper) - set(expanded_upper))
    expanded_only_upper = sorted(set(expanded_upper) - set(old34_upper))
    pooled: list[str] = []
    seen: set[str] = set()
    for gene in list(old34_genes) + list(expanded_genes):
        key = gene.upper()
        if key in seen:
            continue
        seen.add(key)
        pooled.append(gene)

    overlap_genes = [old34_upper[g] for g in overlap_upper]
    old34_only = [old34_upper[g] for g in old34_only_upper]
    expanded_only = [expanded_upper[g] for g in expanded_only_upper]

    gmt_path = out_dir / "pooled_unions.gmt"
    manifest_path = out_dir / "pooled_unions_manifest.tsv"
    write_gmt(
        gmt_path,
        [
            (
                OLD34_SET_ID,
                "Covered-gene union of the old 34-program Stage 0 dictionary.",
                old34_genes,
            ),
            (
                EXPANDED_SET_ID,
                "Covered-gene union of the expanded manuscript-axes Stage 0 dictionary.",
                expanded_genes,
            ),
        ],
    )
    write_manifest(
        manifest_path,
        [
            manifest_row(
                set_id=OLD34_SET_ID,
                subfamily="old34",
                genes=old34_genes,
                source_url=str(old34_path),
                original_name="full_34",
                rationale="Pooled old34 covered genes; no per-geneset Stage 0 filtering.",
            ),
            manifest_row(
                set_id=EXPANDED_SET_ID,
                subfamily="expanded",
                genes=expanded_genes,
                source_url=str(expanded_path),
                original_name="expanded_full",
                rationale="Pooled expanded covered genes; no per-geneset Stage 0 filtering.",
            ),
        ],
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "old34_gene_json": str(old34_path),
        "expanded_gene_json": str(expanded_path),
        "n_old34": len(old34_genes),
        "n_expanded": len(expanded_genes),
        "n_overlap": len(overlap_genes),
        "n_old34_only": len(old34_only),
        "n_expanded_only": len(expanded_only),
        "n_pooled": len(pooled),
        "old34_fraction_nested_in_expanded": round(len(overlap_genes) / max(len(old34_genes), 1), 6),
        "gmt_path": str(gmt_path),
        "manifest_path": str(manifest_path),
        "full_control_panel_id": "pooled_old34_expanded_union",
        "atomic_set_ids": [OLD34_SET_ID, EXPANDED_SET_ID],
    }
    (qc_dir / "overlap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (qc_dir / "overlap_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    write_gene_csv(qc_dir / "old34_only_genes.csv", old34_only, "gene")
    write_gene_csv(qc_dir / "expanded_only_genes.csv", expanded_only, "gene")
    write_gene_csv(qc_dir / "overlap_genes.csv", overlap_genes, "gene")
    write_gene_csv(qc_dir / "pooled_genes.csv", pooled, "gene")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
