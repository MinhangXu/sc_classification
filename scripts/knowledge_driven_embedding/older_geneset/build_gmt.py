#!/usr/bin/env python3
"""
Build an MSigDB-based older-generation gene set bundle.

Outputs (in the same directory as this script):
  - manifest.tsv
  - genesets_v1.gmt
  - README.md
"""

from __future__ import annotations

import csv
import datetime as dt
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import requests


MSIGDB_RELEASE = "2026.1.Hs"
MSIGDB_BASE = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{MSIGDB_RELEASE}"
SOURCE_URLS = {
    "H": f"{MSIGDB_BASE}/h.all.v{MSIGDB_RELEASE}.symbols.gmt",
    "Reactome": f"{MSIGDB_BASE}/c2.cp.reactome.v{MSIGDB_RELEASE}.symbols.gmt",
    "KEGG": f"{MSIGDB_BASE}/c2.cp.kegg_legacy.v{MSIGDB_RELEASE}.symbols.gmt",
}

DESC_BY_SOURCE = {
    "H": f"MSigDB H (Hallmark) release {MSIGDB_RELEASE}",
    "Reactome": f"MSigDB C2:CP:REACTOME release {MSIGDB_RELEASE}",
    "KEGG": f"MSigDB C2:CP:KEGG (kegg_legacy) release {MSIGDB_RELEASE}",
}

# Some requested Reactome names are not present verbatim in modern MSigDB releases.
# We map them to the closest current canonical set names, then emit requested names.
ALIASES: Dict[str, str] = {
    "REACTOME_NF_KB_ACTIVATION": "REACTOME_NF_KB_IS_ACTIVATED_AND_SIGNALS_SURVIVAL",
    "REACTOME_TNFR1_INDUCED_NFKB_SIGNALING_PATHWAY": "REACTOME_TNFR1_INDUCED_NF_KAPPA_B_SIGNALING_PATHWAY",
    "REACTOME_MHC_CLASS_I_ANTIGEN_PRESENTATION": "REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION",
}

HALLMARK_SETS: List[str] = [
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_IL2_STAT5_SIGNALING",
    "HALLMARK_COMPLEMENT",
    "HALLMARK_ALLOGRAFT_REJECTION",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_HYPOXIA",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_GLYCOLYSIS",
    "HALLMARK_MTORC1_SIGNALING",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
]

REACTOME_SETS: List[str] = [
    "REACTOME_INTERFERON_SIGNALING",
    "REACTOME_INTERFERON_GAMMA_SIGNALING",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM",
    "REACTOME_NF_KB_ACTIVATION",
    "REACTOME_TNFR1_INDUCED_NFKB_SIGNALING_PATHWAY",
    "REACTOME_ANTIGEN_PRESENTATION_FOLDING_ASSEMBLY_AND_PEPTIDE_LOADING_OF_CLASS_I_MHC",
    "REACTOME_MHC_CLASS_I_ANTIGEN_PRESENTATION",
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION",
]

KEGG_SETS: List[str] = [
    "KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION",
    "KEGG_JAK_STAT_SIGNALING_PATHWAY",
    "KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY",
]

CORE_SETS = {
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_HYPOXIA",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    "HALLMARK_TGF_BETA_SIGNALING",
    "REACTOME_INTERFERON_SIGNALING",
    "REACTOME_INTERFERON_GAMMA_SIGNALING",
    "REACTOME_NF_KB_ACTIVATION",
    "REACTOME_ANTIGEN_PRESENTATION_FOLDING_ASSEMBLY_AND_PEPTIDE_LOADING_OF_CLASS_I_MHC",
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION",
    "KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION",
}

WHY_INCLUDE_BY_SET: Dict[str, str] = {
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "HALLMARK_INTERFERON_ALPHA_RESPONSE": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "HALLMARK_INFLAMMATORY_RESPONSE": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING": "IL6/JAK/STAT, IL2/STAT5: cytokine signaling axes",
    "HALLMARK_IL2_STAT5_SIGNALING": "IL6/JAK/STAT, IL2/STAT5: cytokine signaling axes",
    "HALLMARK_COMPLEMENT": "TLR/complement/allograft rejection: innate/immune activation context",
    "HALLMARK_ALLOGRAFT_REJECTION": "TLR/complement/allograft rejection: innate/immune activation context",
    "HALLMARK_APOPTOSIS": "p53/apoptosis/hypoxia/UPR/ROS/DNA repair: stress/arrest programs",
    "HALLMARK_P53_PATHWAY": "p53/apoptosis/hypoxia/UPR/ROS/DNA repair: stress/arrest programs",
    "HALLMARK_HYPOXIA": "p53/apoptosis/hypoxia/UPR/ROS/DNA repair: stress/arrest programs",
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": "p53/apoptosis/hypoxia/UPR/ROS/DNA repair: stress/arrest programs",
    "HALLMARK_DNA_REPAIR": "p53/apoptosis/hypoxia/UPR/ROS/DNA repair: stress/arrest programs",
    "HALLMARK_E2F_TARGETS": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_G2M_CHECKPOINT": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_MYC_TARGETS_V1": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_MYC_TARGETS_V2": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_GLYCOLYSIS": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_MTORC1_SIGNALING": "E2F/G2M/MYC/OXPHOS/glycolysis/mTORC1: proliferation/metabolism counter-axis",
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY": "p53/apoptosis/hypoxia/UPR/ROS/DNA repair: stress/arrest programs",
    "HALLMARK_TGF_BETA_SIGNALING": "TGFbeta/EMT: immunosuppression / plasticity axis",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": "TGFbeta/EMT: immunosuppression / plasticity axis",
    "REACTOME_INTERFERON_SIGNALING": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "REACTOME_INTERFERON_GAMMA_SIGNALING": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM": "IL6/JAK/STAT, IL2/STAT5: cytokine signaling axes",
    "REACTOME_NF_KB_ACTIVATION": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "REACTOME_TNFR1_INDUCED_NFKB_SIGNALING_PATHWAY": "IFNg/IFNa/TNF/NFkB/inflammation: core immune-pressure axis / inflammatory signaling",
    "REACTOME_ANTIGEN_PRESENTATION_FOLDING_ASSEMBLY_AND_PEPTIDE_LOADING_OF_CLASS_I_MHC": "Antigen presentation (Reactome/KEGG): required for IFNg-MHC decoupling tests",
    "REACTOME_MHC_CLASS_I_ANTIGEN_PRESENTATION": "Antigen presentation (Reactome/KEGG): required for IFNg-MHC decoupling tests",
    "REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION": "Antigen presentation (Reactome/KEGG): required for IFNg-MHC decoupling tests",
    "KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION": "Antigen presentation (Reactome/KEGG): required for IFNg-MHC decoupling tests",
    "KEGG_JAK_STAT_SIGNALING_PATHWAY": "IL6/JAK/STAT, IL2/STAT5: cytokine signaling axes",
    "KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY": "TLR/complement/allograft rejection: innate/immune activation context",
}


def _clean_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    return "" if (not s or s == "NA") else s


def _parse_gmt_text(gmt_text: str) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for line in gmt_text.splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        genes = [_clean_symbol(g) for g in parts[2:]]
        genes = [g for g in genes if g]
        parsed[name] = genes
    return parsed


def _download_gmt(url: str) -> Dict[str, List[str]]:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return _parse_gmt_text(resp.text)


def _ordered_rows() -> List[Tuple[str, str, str, str]]:
    ordered = (
        [(name, "H") for name in HALLMARK_SETS]
        + [(name, "Reactome") for name in REACTOME_SETS]
        + [(name, "KEGG") for name in KEGG_SETS]
    )
    out: List[Tuple[str, str, str, str]] = []
    for name, source in ordered:
        out.append(
            (
                name,
                source,
                WHY_INCLUDE_BY_SET[name],
                "Core" if name in CORE_SETS else "Optional",
            )
        )
    return out


def _get_pip_freeze_snippet() -> str:
    try:
        cp = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            text=True,
            capture_output=True,
        )
        lines = [ln for ln in cp.stdout.splitlines() if ln.strip()]
        return "\n".join(lines[:30]) if lines else "(no packages reported)"
    except Exception:
        return "(pip freeze unavailable)"


def _fetch_requested_sets(rows: List[Tuple[str, str, str, str]]) -> Dict[str, List[str]]:
    source_maps: Dict[str, Dict[str, List[str]]] = {}
    for source, url in SOURCE_URLS.items():
        source_maps[source] = _download_gmt(url)

    genes_by_requested: Dict[str, List[str]] = {}
    for requested_name, source, _, _ in rows:
        source_map = source_maps[source]
        actual_name = ALIASES.get(requested_name, requested_name)
        if actual_name not in source_map:
            raise RuntimeError(
                f"Requested set '{requested_name}' not found in source '{source}'. "
                f"Expected actual name '{actual_name}'."
            )
        cleaned_sorted = sorted(set(_clean_symbol(g) for g in source_map[actual_name] if _clean_symbol(g)))
        genes_by_requested[requested_name] = cleaned_sorted
    return genes_by_requested


def _write_manifest(path: Path, rows: List[Tuple[str, str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["geneset_name", "source", "why_include", "priority"])
        writer.writerows(rows)


def _write_gmt(
    path: Path,
    rows: List[Tuple[str, str, str, str]],
    genes_by_set: Dict[str, List[str]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="") as f:
        for name, source, _, _ in rows:
            genes = genes_by_set[name]
            counts[name] = len(genes)
            f.write("\t".join([name, DESC_BY_SOURCE[source], *genes]) + "\n")
    return counts


def _write_readme(
    path: Path,
    rows: List[Tuple[str, str, str, str]],
    counts: Dict[str, int],
    started_utc: dt.datetime,
    pip_snippet: str,
) -> None:
    hallmarks = [name for name, source, _, _ in rows if source == "H"]
    reactome = [name for name, source, _, _ in rows if source == "Reactome"]
    kegg = [name for name, source, _, _ in rows if source == "KEGG"]
    low_count = [name for name, n in counts.items() if n < 5]
    vals = list(counts.values())

    md: List[str] = []
    md.append("# Older Generation Gene Set Bundle (MSigDB)")
    md.append("")
    md.append("This directory contains a uniform MSigDB-based bundle for requested older-generation gene sets.")
    md.append("")
    md.append("## Requested set list (exact names)")
    md.append("")
    md.append("### Hallmark (H)")
    for x in hallmarks:
        md.append(f"- `{x}`")
    md.append("")
    md.append("### Reactome (C2:CP:REACTOME)")
    for x in reactome:
        md.append(f"- `{x}`")
    md.append("")
    md.append("### KEGG (C2:CP:KEGG)")
    for x in kegg:
        md.append(f"- `{x}`")
    md.append("")
    md.append("## How sets were retrieved")
    md.append("")
    md.append(f"- Source: MSigDB release `{MSIGDB_RELEASE}` GMT files downloaded over HTTPS.")
    md.append("- Files used:")
    md.append(f"  - Hallmark: `{SOURCE_URLS['H']}`")
    md.append(f"  - Reactome: `{SOURCE_URLS['Reactome']}`")
    md.append(f"  - KEGG: `{SOURCE_URLS['KEGG']}`")
    md.append("- Exact-name output preserved; for 3 Reactome entries, current canonical MSigDB names were aliased:")
    for req_name, actual in ALIASES.items():
        md.append(f"  - `{req_name}` <= `{actual}`")
    md.append("")
    md.append("## Gene symbol harmonization rules")
    md.append("")
    md.append("- Uppercase symbols.")
    md.append("- Remove empty / NA symbols.")
    md.append("- Deduplicate symbols within each set.")
    md.append("- Sort symbols alphabetically within each set.")
    md.append("- Keep sets with fewer than 5 genes and flag them as warnings.")
    md.append("")
    md.append("## QA")
    md.append("")
    md.append(f"- Number of sets written: **{len(rows)}** (expected 34).")
    md.append(
        f"- Gene counts summary (min / median / max): **{min(vals)} / {statistics.median(vals)} / {max(vals)}**."
    )
    md.append("")
    md.append("### Gene counts per set")
    md.append("")
    md.append("| geneset_name | n_genes |")
    md.append("|---|---:|")
    for name, _, _, _ in rows:
        md.append(f"| `{name}` | {counts[name]} |")
    md.append("")
    md.append("### Warnings")
    md.append("")
    if low_count:
        for name in low_count:
            md.append(f"- `{name}` has {counts[name]} genes after cleaning (<5). Included by request.")
    else:
        md.append("- None.")
    md.append("")
    md.append("## Run metadata")
    md.append("")
    md.append(f"- Build timestamp (UTC): `{started_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}`")
    md.append(f"- Python: `{platform.python_version()}`")
    md.append(f"- Platform: `{platform.platform()}`")
    md.append("")
    md.append("### pip freeze (first 30 lines)")
    md.append("")
    md.append("```text")
    md.append(pip_snippet)
    md.append("```")
    md.append("")
    md.append("## Future extension note")
    md.append("")
    md.append(
        "Tier 2 cytokine-dictionary gene sets should live in "
        "`knowledge_driven_embedding/cytokine_dictionary/`."
    )
    md.append("")

    path.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    started = dt.datetime.now(dt.timezone.utc)
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _ordered_rows()
    requested_names = [r[0] for r in rows]
    if len(requested_names) != 34:
        raise RuntimeError(f"Expected 34 sets but got {len(requested_names)}")
    if len(set(requested_names)) != len(requested_names):
        raise RuntimeError("Duplicate geneset names detected in requested list")

    genes_by_set = _fetch_requested_sets(rows)
    missing_after_fetch = [n for n in requested_names if n not in genes_by_set]
    if missing_after_fetch:
        raise RuntimeError(f"Missing sets after retrieval: {missing_after_fetch}")

    manifest_path = out_dir / "manifest.tsv"
    gmt_path = out_dir / "genesets_v1.gmt"
    readme_path = out_dir / "README.md"

    _write_manifest(manifest_path, rows)
    counts = _write_gmt(gmt_path, rows, genes_by_set)
    _write_readme(readme_path, rows, counts, started_utc=started, pip_snippet=_get_pip_freeze_snippet())

    print(f"Wrote {manifest_path}")
    print(f"Wrote {gmt_path}")
    print(f"Wrote {readme_path}")
    print(f"Set count: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
