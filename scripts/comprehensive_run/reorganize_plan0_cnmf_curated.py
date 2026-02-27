#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Entry:
    src: Path
    dst: Path
    role: str
    k: Optional[int]


def infer_role_and_k(name: str) -> tuple[str, Optional[int]]:
    k = None
    for kk in (20, 40, 60):
        if f".k_{kk}." in name:
            k = kk
            break

    if "iter_" in name and "spectra" in name:
        return "iter_spectra", k
    if "merged" in name and "spectra" in name:
        return "merged_spectra", k
    if "local_density_cache" in name:
        return "local_density_cache", k
    if "consensus" in name and "usages" in name:
        return "consensus_usages", k
    if "consensus" in name and "spectra" in name:
        return "consensus_spectra", k
    if "gene_spectra_tpm" in name:
        return "gene_spectra_tpm", k
    if "gene_spectra_score" in name:
        return "gene_spectra_score", k
    if "starcat_spectra" in name:
        return "starcat_spectra", k
    if "nmf_params" in name:
        return "nmf_params", None
    if "nmf_idvrun_params" in name:
        return "nmf_run_params_yaml", None
    if "norm_counts.h5ad" in name:
        return "norm_counts_h5ad", None
    if "tpm.h5ad" in name:
        return "tpm_h5ad", None
    if "tpm_stats" in name:
        return "tpm_stats", None
    if "overdispersed_genes" in name:
        return "overdispersed_genes", None
    return "other", k


def target_subdir(role: str, k: Optional[int]) -> str:
    if k is None:
        return "global"
    if role in {"iter_spectra", "merged_spectra", "local_density_cache"}:
        return f"k_{k}/inputs"
    return f"k_{k}/consensus"


def build_entries(raw_dir: Path, curated_dir: Path) -> list[Entry]:
    entries: list[Entry] = []
    for src in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        role, k = infer_role_and_k(src.name)
        sub = target_subdir(role, k)
        dst = curated_dir / sub / src.name
        entries.append(Entry(src=src, dst=dst, role=role, k=k))
    return entries


def ensure_parent(paths: Iterable[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def write_manifest(curated_dir: Path, entries: list[Entry], exp_dir: Path) -> None:
    out = curated_dir / "MANIFEST.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "src_rel_to_experiment",
                "dst_rel_to_experiment",
                "role",
                "k",
                "size_mb",
            ]
        )
        for e in entries:
            w.writerow(
                [
                    str(e.src.relative_to(exp_dir)),
                    str(e.dst.relative_to(exp_dir)),
                    e.role,
                    "" if e.k is None else e.k,
                    round(e.src.stat().st_size / (1024 * 1024), 3),
                ]
            )


def write_readme(curated_dir: Path) -> None:
    text = """# Curated cNMF output view (non-destructive)

This folder provides a sequence-oriented view of `plan0_cnmf` artifacts.
Source files remain in `../plan0_cnmf/` and are linked here.

## Layout
- `global/`: run-wide files (`nmf_params`, `norm_counts`, `tpm`, stats, hvg list)
- `k_<K>/inputs/`: per-K pre-consensus artifacts (iter spectra, merged spectra, local-density cache)
- `k_<K>/consensus/`: per-K consensus artifacts (usages, spectra, gene-level exports)
- `MANIFEST.csv`: source/destination mapping and file roles

## Sequence (high-level)
1. `iter_spectra` are generated from repeated NMF runs.
2. `merged_spectra` combines all replicate components for each K.
3. `local_density_cache` supports outlier filtering during consensus.
4. `consensus_spectra` + `consensus_usages` are final program and usage matrices.
5. `gene_spectra_*`/`starcat_spectra` are derived gene-level exports.
"""
    (curated_dir / "README.md").write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Build non-destructive curated cNMF layout.")
    p.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory containing models/cnmf_plan0/plan0_cnmf",
    )
    p.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Materialize curated files as symlinks (default) or copies.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing curated links/files if needed.",
    )
    args = p.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    raw_dir = exp_dir / "models" / "cnmf_plan0" / "plan0_cnmf"
    curated_dir = exp_dir / "models" / "cnmf_plan0" / "curated"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw cNMF dir not found: {raw_dir}")

    entries = build_entries(raw_dir, curated_dir)
    ensure_parent([e.dst for e in entries])

    for e in entries:
        if e.dst.exists() or e.dst.is_symlink():
            if args.force:
                e.dst.unlink()
            else:
                continue

        if args.mode == "symlink":
            e.dst.symlink_to(e.src)
        else:
            e.dst.write_bytes(e.src.read_bytes())

    write_manifest(curated_dir, entries, exp_dir)
    write_readme(curated_dir)
    print(f"curated_dir={curated_dir}")
    print(f"entries={len(entries)} mode={args.mode}")


if __name__ == "__main__":
    main()

