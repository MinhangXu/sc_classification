#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sc_classification.utils.hucira_interpretation import export_hucira_reference_sets


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Export reusable huCIRA reference assets for downstream notebook interpretation.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "hucira_reference",
        help="Directory where huCIRA reference tables and gene sets will be written.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-download and rebuild even if cached exports already exist.",
    )
    parser.add_argument(
        "--cytokine-aggregation-level",
        choices=["cytokine", "cytokine_celltype"],
        default="cytokine",
        help="How to aggregate cytokine-response rows into reference programs.",
    )
    parser.add_argument(
        "--cip-aggregation-level",
        choices=["cip", "cip_celltype"],
        default="cip",
        help="How to aggregate CIP rows into reference programs.",
    )
    parser.add_argument(
        "--cytokine-adj-p-value-max",
        type=float,
        default=0.05,
        help="Maximum adjusted p-value kept for cytokine dictionary rows.",
    )
    parser.add_argument(
        "--cytokine-min-abs-log-fc",
        type=float,
        default=0.0,
        help="Minimum absolute log fold-change kept for cytokine dictionary rows.",
    )
    parser.add_argument(
        "--cip-min-abs-effect-size",
        type=float,
        default=0.0,
        help="Minimum absolute effect size kept for CIP rows.",
    )
    parser.add_argument(
        "--min-genes-per-program",
        type=int,
        default=10,
        help="Drop reference programs with fewer than this many genes after aggregation.",
    )
    parser.add_argument(
        "--no-gmt",
        action="store_true",
        help="Skip GMT export files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = export_hucira_reference_sets(
        args.output_dir,
        force_rebuild=bool(args.force_rebuild),
        cytokine_aggregation_level=str(args.cytokine_aggregation_level),
        cip_aggregation_level=str(args.cip_aggregation_level),
        cytokine_adj_p_value_max=float(args.cytokine_adj_p_value_max),
        cytokine_min_abs_log_fc=float(args.cytokine_min_abs_log_fc),
        cip_min_abs_effect_size=float(args.cip_min_abs_effect_size),
        min_genes_per_program=int(args.min_genes_per_program),
        write_gmt=not bool(args.no_gmt),
    )

    summary = {
        "output_dir": str(assets["output_dir"]),
        "reference_rows": int(assets["reference_table"].shape[0]),
        "cytokine_programs": int(assets["cytokine_long"]["program_name"].nunique()),
        "cip_programs": int(assets["cip_long"]["program_name"].nunique()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
