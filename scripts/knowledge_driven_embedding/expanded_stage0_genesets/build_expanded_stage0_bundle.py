#!/usr/bin/env python3
"""Build the expanded Stage 0 gene-set bundle for MRD malignant classification.

The builder downloads/caches public GMT resources, resolves regex selectors into
frozen gene-set names, adds family-union and expert anchor panels, and writes
manifest/QC artifacts that can be consumed by the existing Stage 0 runner.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
SC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_H5AD = REPO_ROOT / "data" / "cohort_adata" / "adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad"
DEFAULT_OLD_BUNDLE = SC_ROOT / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "genesets_v1.gmt"
DEFAULT_HVG_RANKED = (
    SC_ROOT
    / "experiments"
    / "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
    / "preprocessing"
    / "gene_universe"
    / "hvg_ranked_genes.csv"
)
DEFAULT_CACHE = REPO_ROOT / "data" / "resource_cache" / "stage0_expanded_genesets"
MSIGDB_RELEASE = "2026.1.Hs"
MSIGDB_BASE = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{MSIGDB_RELEASE}"


@dataclass(frozen=True)
class ResourceSpec:
    source_collection: str
    source_version: str
    url: str
    cache_name: str
    resource_type: str = "gmt"
    local_path: str = ""


@dataclass(frozen=True)
class SelectorSpec:
    selector_id: str
    family: str
    subfamily: str
    interpretation_layer: str
    priority: str
    source_collections: tuple[str, ...]
    regexes: tuple[str, ...]
    rationale: str
    circularity_flag: str = "public_prior"


MSIGDB_RESOURCES: tuple[ResourceSpec, ...] = (
    ResourceSpec("MSigDB_H", MSIGDB_RELEASE, f"{MSIGDB_BASE}/h.all.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_h.all.symbols.gmt"),
    ResourceSpec("MSigDB_C2_REACTOME", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c2.cp.reactome.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c2.cp.reactome.symbols.gmt"),
    ResourceSpec("MSigDB_C2_KEGG_LEGACY", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c2.cp.kegg_legacy.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c2.cp.kegg_legacy.symbols.gmt"),
    ResourceSpec("MSigDB_C5_GO_BP", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c5.go.bp.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c5.go.bp.symbols.gmt"),
    ResourceSpec("MSigDB_C5_GO_CC", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c5.go.cc.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c5.go.cc.symbols.gmt"),
    ResourceSpec("MSigDB_C5_GO_MF", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c5.go.mf.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c5.go.mf.symbols.gmt"),
    ResourceSpec("MSigDB_C3_TFT_GTRD", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c3.tft.gtrd.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c3.tft.gtrd.symbols.gmt"),
    ResourceSpec("MSigDB_C3_TFT", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c3.tft.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c3.tft.symbols.gmt"),
    ResourceSpec("MSigDB_C7_IMMUNESIGDB", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c7.immunesigdb.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c7.immunesigdb.symbols.gmt"),
    ResourceSpec("MSigDB_C8_CELL_TYPE", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c8.all.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c8.all.symbols.gmt"),
    ResourceSpec("MSigDB_C2_CGP", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c2.cgp.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c2.cgp.symbols.gmt"),
    ResourceSpec("MSigDB_C2_WIKIPATHWAYS", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c2.cp.wikipathways.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c2.cp.wikipathways.symbols.gmt"),
    ResourceSpec("MSigDB_C4_3CA", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c4.3ca.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c4.3ca.symbols.gmt"),
    ResourceSpec("MSigDB_C6_ONCOGENIC", MSIGDB_RELEASE, f"{MSIGDB_BASE}/c6.all.v{MSIGDB_RELEASE}.symbols.gmt", f"msigdb_{MSIGDB_RELEASE}_c6.all.symbols.gmt"),
)


HUCIRA_UP = REPO_ROOT / "data" / "hucira_reference" / "cytokine_gene_sets_up.gmt"
HUCIRA_DOWN = REPO_ROOT / "data" / "hucira_reference" / "cytokine_gene_sets_down.gmt"
LOCAL_RESOURCES: tuple[ResourceSpec, ...] = (
    ResourceSpec("huCIRA_CYTOKINE_UP", "local_hucira_reference", "", "cytokine_gene_sets_up.gmt", local_path=str(HUCIRA_UP)),
    ResourceSpec("huCIRA_CYTOKINE_DOWN", "local_hucira_reference", "", "cytokine_gene_sets_down.gmt", local_path=str(HUCIRA_DOWN)),
)


SELECTORS: tuple[SelectorSpec, ...] = (
    SelectorSpec(
        "hspc_lsc_stemness_public",
        "hspc_lsc_stemness",
        "hsc_mpp_lsc",
        "cell_state",
        "core",
        ("MSigDB_C8_CELL_TYPE", "MSigDB_C5_GO_BP", "MSigDB_C2_CGP"),
        ("HSC", "HEMATOPOIETIC[_ ]STEM", "MULTIPOTENT[_ ]PROGENITOR", r"\bMPP\b", "LMPP", "STEM[_ ]CELL", "LEUKEMIC[_ ]STEM", r"\bLSC\b", "PROGENITOR"),
        "Stem/progenitor and LSC-like transcriptional programs missing from the old-34 immune/stress panel.",
    ),
    SelectorSpec(
        "hematopoietic_lineage_controls_public",
        "hematopoietic_lineage_controls",
        "lineage_composition",
        "lineage_control",
        "core",
        ("MSigDB_C8_CELL_TYPE", "MSigDB_C5_GO_BP", "MSigDB_C4_3CA"),
        ("ERYTHROID", "MEGAKARYOCYTE", "GRANULOCYTE", "MONOCYTE", "DENDRITIC", "LYMPHOID", r"\bGMP\b", r"\bCMP\b", r"\bMEP\b", r"\bCLP\b", "MYELOID[_ ]DIFFERENTIATION"),
        "Lineage controls to distinguish malignancy biology from composition or maturation effects.",
    ),
    SelectorSpec(
        "quiescence_dormancy_growth_arrest_public",
        "quiescence_dormancy_growth_arrest",
        "cell_cycle_arrest",
        "cell_state",
        "core",
        ("MSigDB_C5_GO_BP", "MSigDB_C2_REACTOME", "MSigDB_C2_CGP"),
        ("QUIESCENCE", "DORMANCY", "CELL[_ ]CYCLE[_ ]ARREST", "NEGATIVE[_ ]REGULATION[_ ]OF[_ ]CELL[_ ]CYCLE", r"\bG0\b", r"\bG1\b", "CDK[_ ]INHIBITOR", "CHECKPOINT"),
        "Dormancy/quiescence axis for relapse MRD programs with low proliferation.",
    ),
    SelectorSpec(
        "senescence_sasp_public",
        "senescence_sasp_control",
        "senescence_sasp",
        "cell_state",
        "support",
        ("MSigDB_C5_GO_BP", "MSigDB_C2_REACTOME", "MSigDB_C2_CGP"),
        ("SENESCENCE", r"\bSASP\b", "SENESCENCE[_ ]ASSOCIATED[_ ]SECRETORY", "CELLULAR[_ ]SENESCENCE"),
        "Senescence/SASP contrast so quiescence is not overinterpreted as senescence.",
    ),
    SelectorSpec(
        "ifng_stat_irf_targets_public",
        "ifng_decomposed_response",
        "ifng_stat_irf_targets",
        "target_footprint",
        "core",
        ("MSigDB_C7_IMMUNESIGDB", "MSigDB_C5_GO_BP", "MSigDB_C2_REACTOME", "huCIRA_CYTOKINE_UP", "huCIRA_CYTOKINE_DOWN"),
        ("INTERFERON[_ ]GAMMA", r"\bIFNG\b", "IFN[_ -]GAMMA", "STAT1", "IRF1", "IRF7", "IRF8", "GBP", "IFITM", "ISG"),
        "Decomposed IFN-gamma target footprint beyond the broad hallmark set.",
    ),
    SelectorSpec(
        "ifng_chemokine_output_public",
        "ifng_decomposed_response",
        "ifng_chemokine_output",
        "effector",
        "core",
        ("MSigDB_C5_GO_BP", "MSigDB_C7_IMMUNESIGDB", "huCIRA_CYTOKINE_UP"),
        ("CXCL9", "CXCL10", "CXCL11", "CHEMOKINE", "INTERFERON[_ ]GAMMA.*CHEMOKINE"),
        "IFN-gamma chemokine output axis.",
    ),
    SelectorSpec(
        "ifng_feedback_public",
        "ifng_decomposed_response",
        "ifng_feedback",
        "relay",
        "support",
        ("MSigDB_C5_GO_BP", "MSigDB_C7_IMMUNESIGDB", "huCIRA_CYTOKINE_UP", "huCIRA_CYTOKINE_DOWN"),
        ("SOCS1", "SOCS3", "USP18", "NEGATIVE[_ ]REGULATION.*INTERFERON", "INTERFERON.*FEEDBACK"),
        "IFN feedback and desensitization controls.",
    ),
    SelectorSpec(
        "antigen_presentation_ciita_public",
        "antigen_presentation_ciita_hla_escape",
        "mhc_i_mhc_ii_ciita",
        "effector",
        "core",
        ("MSigDB_C2_REACTOME", "MSigDB_C5_GO_BP", "MSigDB_C3_TFT_GTRD", "MSigDB_C3_TFT"),
        ("ANTIGEN[_ ]PROCESS", "ANTIGEN[_ ]PRESENT", "MHC[_ ]CLASS[_ ]I", "MHC[_ ]CLASS[_ ]II", r"\bHLA\b", "CIITA", r"\bB2M\b", r"\bTAP\b", "IMMUNE[_ ]EVASION", "RFX5", "RFXANK"),
        "MHC-I/MHC-II/CIITA axis for IFNG-HLA decoupling and immune-escape checks.",
    ),
    SelectorSpec(
        "tnf_nfkb_survival_adhesion_public",
        "tnf_nfkb_decomposed_survival_adhesion",
        "canonical_tnf_nfkb_survival",
        "relay",
        "core",
        ("MSigDB_C2_REACTOME", "MSigDB_C5_GO_BP", "MSigDB_C3_TFT_GTRD", "MSigDB_C3_TFT", "huCIRA_CYTOKINE_UP"),
        ("TNF", "NFKB", "NF[_ ]KB", "TNFR", "RELA", "NFKB1", "CANONICAL[_ ]NF", "APOPTOSIS", "ANTI[_ ]APOPTOSIS", "ADHESION"),
        "Canonical TNF/NF-kB survival and adhesion axis.",
    ),
    SelectorSpec(
        "cd54_cd244_adhesion_niche_public",
        "cd54_cd244_adhesion_niche",
        "adhesion_niche",
        "effector",
        "core",
        ("MSigDB_C5_GO_BP", "MSigDB_C2_WIKIPATHWAYS", "MSigDB_C2_REACTOME"),
        ("CELL[_ ]ADHESION", "LEUKOCYTE[_ ]ADHESION", "CELL[_ ]CELL[_ ]ADHESION", "INTEGRIN", r"\bICAM\b", r"\bSLAM\b", "IMMUNOREGULATORY[_ ]INTERACTIONS", "EXTRACELLULAR[_ ]MATRIX", "CXCR4"),
        "CD54/CD244 and niche-interaction biology that can coordinate with TNF/NF-kB.",
    ),
    SelectorSpec(
        "tf_regulon_public",
        "regulon_ifn_nfkb_ap1_gata_myc_e2f_p53_smad",
        "tf_targets",
        "tf_regulon",
        "core",
        ("MSigDB_C3_TFT_GTRD", "MSigDB_C3_TFT"),
        ("STAT1", "STAT2", "IRF1", "IRF2", "IRF4", "IRF7", "IRF8", "IRF9", "NFKB1", "NFKB2", "RELA", "RELB", r"\bREL\b", r"\bJUN\b", r"\bFOS\b", "BATF", "GATA1", "GATA2", r"\bMYC\b", "E2F1", "E2F2", "E2F4", "TP53", "SMAD2", "SMAD3", "SMAD4", "CIITA", "RFX5", "RFXANK", "RUNX1", r"\bERG\b", "HOXA9", "MEIS1"),
        "Transcription-factor target/regulon priors for immune, stemness, proliferation, p53, and TGF-beta axes.",
    ),
    SelectorSpec(
        "typeI_ifn_inflammasome_noncanonical_nfkb_public",
        "typeI_ifn_inflammasome_noncanonical_nfkb_contrast",
        "typeI_ifn_inflammasome_noncanonical_nfkb",
        "target_footprint",
        "core",
        ("MSigDB_C5_GO_BP", "MSigDB_C2_REACTOME", "MSigDB_C7_IMMUNESIGDB"),
        ("TYPE[_ ]I[_ ]INTERFERON", "INTERFERON[_ ]ALPHA", "INTERFERON[_ ]BETA", "INFLAMMASOME", "NLRP3", "AIM2", "PYROPTOSIS", "INTERLEUKIN[_ ]1", r"\bIL1\b", "IL18", "CASPASE[_ ]1", "NONCANONICAL[_ ]NF", "NFKB2", "RELB"),
        "Inflammasome/type-I IFN/noncanonical NF-kB contrast for inflammatory-state specificity.",
    ),
    SelectorSpec(
        "mitochondrial_metabolism_fine_public",
        "mitochondrial_metabolism_fine",
        "mitochondrial_energy_metabolism",
        "cell_state",
        "core",
        ("MSigDB_C5_GO_BP", "MSigDB_C5_GO_CC", "MSigDB_C2_REACTOME"),
        ("MITOCHONDRIAL", "RESPIRATORY[_ ]CHAIN", "ELECTRON[_ ]TRANSPORT", "FATTY[_ ]ACID", "ACYL[_ ]COA", "COENZYME[_ ]A", r"\bTCA\b", "TRICARBOXYLIC", "OXIDATIVE[_ ]PHOSPHORYLATION", "MITOCHONDRIAL[_ ]TRANSPORT", "MITOCHONDRIAL[_ ]INNER[_ ]MEMBRANE"),
        "Fine mitochondrial and oxidative-metabolism contrasts beyond hallmark OXPHOS.",
    ),
    SelectorSpec(
        "tgfb_niche_quiescence_public",
        "tgfb_niche_quiescence",
        "tgfb_smad_ecm_quiescence",
        "relay",
        "core",
        ("MSigDB_H", "MSigDB_C2_REACTOME", "MSigDB_C5_GO_BP"),
        ("TGF[_ ]BETA", r"\bSMAD\b", "EXTRACELLULAR[_ ]MATRIX", "HEMATOPOIESIS", "QUIESCENCE", "FIBROSIS"),
        "TGF-beta/SMAD/niche/quiescence axis for marrow microenvironment and dormancy interpretations.",
    ),
)


CORE_ANCHORS: dict[str, dict[str, Any]] = {
    "hspc_lsc_stemness": {
        "subfamily": "core_hsc_mpp_lsc",
        "genes": "CD34 KIT MPL PROM1 HLF GATA2 MEIS1 HOXA9 ERG LMO2 RUNX1 MECOM ADGRG1 ITGA6 THY1".split(),
        "layer": "cell_state",
        "rationale": "Expert anchors for HSC/MPP/LSC interpretation and sanity checks.",
    },
    "quiescence_dormancy_growth_arrest": {
        "subfamily": "core_quiescence_arrest",
        "genes": "CDKN1A CDKN1B CDKN1C CDKN2C BTG1 BTG2 KLF2 KLF4 EGR1 FOXO3 GADD45A".split(),
        "layer": "cell_state",
        "rationale": "Expert anchors for G0/quiescence/growth-arrest interpretation.",
    },
    "ifng_decomposed_response": {
        "subfamily": "core_ifng_ligand_receptor_stat_irf_output_feedback",
        "genes": "IFNG IFNGR1 IFNGR2 JAK1 JAK2 STAT1 STAT2 IRF1 IRF2 IRF7 IRF8 IRF9 GBP1 GBP2 GBP5 IFI30 IFITM1 ISG15 ISG20 CXCL9 CXCL10 CXCL11 SOCS1 SOCS3 USP18".split(),
        "layer": "ligand_receptor_relay_tf_effector",
        "rationale": "Expert anchors decomposing ligand, receptor, relay, TF, targets, chemokines, and feedback.",
    },
    "antigen_presentation_ciita_hla_escape": {
        "subfamily": "core_mhc_ciita",
        "genes": "HLA-A HLA-B HLA-C B2M TAP1 TAP2 PSMB8 PSMB9 CIITA CD74 HLA-DRA HLA-DRB1 HLA-DPA1 HLA-DPB1 HLA-DQA1 HLA-DQB1 RFX5 RFXANK".split(),
        "layer": "effector_tf",
        "rationale": "Expert anchors for MHC-I/MHC-II, CIITA, and peptide processing.",
    },
    "tnf_nfkb_decomposed_survival_adhesion": {
        "subfamily": "core_tnf_nfkb_survival",
        "genes": "TNF TNFRSF1A TNFRSF1B NFKB1 RELA NFKBIA TNFAIP3 BCL2A1 BIRC2 BIRC3 CFLAR ICAM1".split(),
        "layer": "ligand_receptor_relay_effector",
        "rationale": "Expert anchors for canonical TNF/NF-kB survival and adhesion output.",
    },
    "cd54_cd244_adhesion_niche": {
        "subfamily": "core_cd54_cd244_niche",
        "genes": "ICAM1 SLAMF4 ADGRG1 ITGA6 THY1 CXCR4 CD44 VCAM1 SELL SELPLG ITGAL ITGB2 LGALS3".split(),
        "layer": "effector",
        "rationale": "Expert anchors for CD54/CD244/niche adhesion biology.",
    },
    "regulon_ifn_nfkb_ap1_gata_myc_e2f_p53_smad": {
        "subfamily": "core_tf_nodes",
        "genes": "STAT1 STAT2 IRF1 IRF2 IRF4 IRF7 IRF8 IRF9 NFKB1 NFKB2 RELA RELB REL JUN FOS BATF GATA1 GATA2 MYC E2F1 E2F2 E2F4 TP53 SMAD2 SMAD3 SMAD4 CIITA RFX5 RFXANK RUNX1 ERG HOXA9 MEIS1".split(),
        "layer": "tf",
        "rationale": "Expert TF-node anchors; primary regulon benchmark uses public target sets, not these nodes alone.",
    },
    "mitochondrial_metabolism_fine": {
        "subfamily": "core_mitochondrial_metabolism",
        "genes": "ACSM3 SLC25A21 NDUF* SDH* UQCR* COX* ATP5* CPT* ACAD* SLC25*".split(),
        "layer": "cell_state",
        "rationale": "Expert mitochondrial anchors with family-prefix expansion against the expression gene universe.",
    },
    "tgfb_niche_quiescence": {
        "subfamily": "core_tgfb_smad_ecm",
        "genes": "TGFB1 TGFB2 TGFB3 TGFBR1 TGFBR2 SMAD2 SMAD3 SMAD4 SMAD7 SERPINE1 CTGF PMEPA1 COL* VIM".split(),
        "layer": "ligand_receptor_relay_effector",
        "rationale": "Expert TGF-beta/SMAD/ECM anchors for niche/quiescence interpretation.",
    },
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    return str(value)


def safe_id(value: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).lower()).strip("_")
    text = re.sub(r"_+", "_", text)
    if len(text) <= max_len:
        return text or "set"
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{text[: max_len - 9].rstrip('_')}_{digest}"


def clean_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value or value in {"NA", "N/A", "NULL", "NONE"}:
        return ""
    return value


def unique_clean_genes(genes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for gene in genes:
        clean = clean_symbol(gene)
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def resource_metadata(spec: ResourceSpec, path: Path, retrieval_status: str, retrieval_date: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source_collection": spec.source_collection,
        "source_version": spec.source_version,
        "resource_type": spec.resource_type,
        "url": spec.url,
        "retrieval_date_utc": retrieval_date,
        "retrieval_status": retrieval_status,
        "file_size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
        "local_path": str(path),
    }


def materialize_resource(spec: ResourceSpec, cache_dir: Path, *, overwrite: bool = False) -> tuple[Path, dict[str, Any]]:
    retrieval_date = dt.datetime.now(dt.timezone.utc).isoformat()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / spec.cache_name
    if spec.local_path:
        source = Path(spec.local_path)
        if not source.exists():
            raise FileNotFoundError(f"Missing local resource for {spec.source_collection}: {source}")
        return source, resource_metadata(spec, source, "local_existing", retrieval_date)
    if path.exists() and not overwrite:
        return path, resource_metadata(spec, path, "cached_existing", retrieval_date)
    if path.exists() and overwrite:
        backup = path.with_suffix(path.suffix + f".bak_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}")
        shutil.move(str(path), str(backup))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(spec.url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(tmp, path)
    return path, resource_metadata(spec, path, "downloaded", retrieval_date)


def parse_gmt(path: Path, source_collection: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "source_collection": source_collection,
                    "original_name": parts[0].strip(),
                    "description": parts[1].strip(),
                    "genes": unique_clean_genes(parts[2:]),
                    "source_line": line_no,
                }
            )
    return rows


def load_gene_universe(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    try:
        return {clean_symbol(g) for g in adata.var_names.astype(str)}
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


def load_hvg(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        return set()
    return {clean_symbol(g) for g in df["gene"].astype(str)}


def load_old_bundle(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    genes: set[str] = set()
    for row in parse_gmt(path, "old_bundle"):
        genes.update(row["genes"])
    return genes


def intersect_present(genes: list[str], universe: set[str]) -> list[str]:
    if not universe:
        return list(genes)
    return [gene for gene in genes if clean_symbol(gene) in universe]


def compile_selector_regex(selector: SelectorSpec) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{pattern})" for pattern in selector.regexes), flags=re.IGNORECASE)


def make_set_id(kind: str, family: str, subfamily: str, name: str) -> str:
    stem = "__".join([kind, safe_id(family, 36), safe_id(subfamily, 36), safe_id(name, 80)])
    if len(stem) <= 150:
        return stem
    digest = hashlib.sha1(stem.encode()).hexdigest()[:10]
    return f"{stem[:139].rstrip('_')}__{digest}"


def serialize_genes(genes: list[str]) -> str:
    return ";".join(genes)


def manifest_row(
    *,
    set_id: str,
    source_collection: str,
    source_url: str,
    source_version: str,
    original_name: str,
    family: str,
    subfamily: str,
    interpretation_layer: str,
    priority: str,
    rationale: str,
    genes_raw: list[str],
    genes_present: list[str],
    old_bundle_genes: set[str],
    hvg_genes: set[str],
    circularity_flag: str,
    selector_id: str,
    selector_regex: str,
    broad_threshold: int,
    very_broad_threshold: int,
) -> dict[str, Any]:
    n_raw = len(genes_raw)
    n_present = len(genes_present)
    broad_flag = ""
    if n_raw > very_broad_threshold:
        broad_flag = f"very_broad_gt_{very_broad_threshold}"
    elif n_raw > broad_threshold:
        broad_flag = f"broad_gt_{broad_threshold}"
    return {
        "set_id": set_id,
        "source_collection": source_collection,
        "source_url": source_url,
        "source_version": source_version,
        "original_name": original_name,
        "family": family,
        "subfamily": subfamily,
        "interpretation_layer": interpretation_layer,
        "priority": priority,
        "rationale": rationale,
        "genes_raw": serialize_genes(genes_raw),
        "genes_present": serialize_genes(genes_present),
        "n_raw": n_raw,
        "n_present": n_present,
        "overlap_old_bundle": len(set(genes_raw) & old_bundle_genes),
        "overlap_hvg": len(set(genes_raw) & hvg_genes),
        "circularity_flag": circularity_flag,
        "selector_id": selector_id,
        "selector_regex": selector_regex,
        "broad_set_flag": broad_flag,
        # Backward-compatible aliases consumed by the existing Stage 0 runner.
        "geneset_name": set_id,
        "source": source_collection,
        "why_include": rationale,
    }


def anchor_genes(raw: list[str], universe: set[str]) -> list[str]:
    expanded: list[str] = []
    for gene in raw:
        clean = clean_symbol(gene)
        if clean.endswith("*"):
            prefix = clean[:-1]
            expanded.extend(sorted(g for g in universe if g.startswith(prefix)))
        else:
            expanded.append(clean)
    return unique_clean_genes(expanded)


def write_gmt(path: Path, manifest: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for _, row in manifest.iterrows():
            genes = [g for g in str(row["genes_raw"]).split(";") if g]
            desc = (
                f"{row['source_collection']}|{row['source_version']}|"
                f"{row['family']}|{row['subfamily']}|{row['original_name']}"
            )
            handle.write("\t".join([str(row["set_id"]), desc, *genes]) + "\n")


def build_bundle(
    *,
    resources: list[dict[str, Any]],
    gene_universe: set[str],
    old_bundle_genes: set[str],
    hvg_genes: set[str],
    min_present: int,
    broad_threshold: int,
    very_broad_threshold: int,
    max_atomic_per_selector: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source_meta = {row["source_collection"]: row for row in resources}
    all_sets: list[dict[str, Any]] = []
    for meta in resources:
        all_sets.extend(parse_gmt(Path(meta["local_path"]), str(meta["source_collection"])))

    selected_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    seen_atomic: set[tuple[str, str, str, str]] = set()
    selector_provenance: dict[str, Any] = {}

    for selector in SELECTORS:
        regex = compile_selector_regex(selector)
        matches = [
            row
            for row in all_sets
            if row["source_collection"] in selector.source_collections
            and regex.search(str(row["original_name"]).replace("-", "_"))
        ]
        matches = sorted(matches, key=lambda row: (str(row["source_collection"]), str(row["original_name"])))
        if max_atomic_per_selector and len(matches) > max_atomic_per_selector:
            for row in matches[max_atomic_per_selector:]:
                dropped_rows.append(
                    {
                        "set_id": make_set_id("atomic", selector.family, selector.subfamily, row["original_name"]),
                        "original_name": row["original_name"],
                        "source_collection": row["source_collection"],
                        "family": selector.family,
                        "subfamily": selector.subfamily,
                        "reason": f"selector_cap_gt_{max_atomic_per_selector}",
                        "n_raw": len(row["genes"]),
                        "n_present": len(intersect_present(row["genes"], gene_universe)),
                    }
                )
            matches = matches[:max_atomic_per_selector]
        selector_provenance[selector.selector_id] = {
            **asdict(selector),
            "regex": "|".join(selector.regexes),
            "n_resolved_before_filter": len(matches),
            "resolved_names": [row["original_name"] for row in matches],
        }
        for row in matches:
            key = (selector.family, selector.subfamily, row["source_collection"], row["original_name"])
            if key in seen_atomic:
                continue
            seen_atomic.add(key)
            genes_raw = sorted(unique_clean_genes(row["genes"]))
            genes_present = sorted(intersect_present(genes_raw, gene_universe))
            set_id = make_set_id("atomic", selector.family, selector.subfamily, row["original_name"])
            if len(genes_present) < min_present:
                dropped_rows.append(
                    {
                        "set_id": set_id,
                        "original_name": row["original_name"],
                        "source_collection": row["source_collection"],
                        "family": selector.family,
                        "subfamily": selector.subfamily,
                        "reason": f"n_present_lt_{min_present}",
                        "n_raw": len(genes_raw),
                        "n_present": len(genes_present),
                    }
                )
                continue
            meta = source_meta[row["source_collection"]]
            selected_rows.append(
                manifest_row(
                    set_id=set_id,
                    source_collection=row["source_collection"],
                    source_url=meta.get("url", ""),
                    source_version=meta.get("source_version", ""),
                    original_name=row["original_name"],
                    family=selector.family,
                    subfamily=selector.subfamily,
                    interpretation_layer=selector.interpretation_layer,
                    priority=selector.priority,
                    rationale=selector.rationale,
                    genes_raw=genes_raw,
                    genes_present=genes_present,
                    old_bundle_genes=old_bundle_genes,
                    hvg_genes=hvg_genes,
                    circularity_flag=selector.circularity_flag,
                    selector_id=selector.selector_id,
                    selector_regex="|".join(selector.regexes),
                    broad_threshold=broad_threshold,
                    very_broad_threshold=very_broad_threshold,
                )
            )

    atomic_df = pd.DataFrame(selected_rows)
    union_rows: list[dict[str, Any]] = []
    if not atomic_df.empty:
        for (family, subfamily), group in atomic_df.groupby(["family", "subfamily"], sort=True):
            genes = sorted(set().union(*(set(str(x).split(";")) for x in group["genes_raw"])))
            genes = [g for g in genes if g]
            genes_present = sorted(intersect_present(genes, gene_universe))
            if len(genes_present) < min_present:
                dropped_rows.append(
                    {
                        "set_id": make_set_id("family_union", family, subfamily, "selected_atomic_sets"),
                        "original_name": "selected_atomic_sets",
                        "source_collection": "derived_union",
                        "family": family,
                        "subfamily": subfamily,
                        "reason": f"n_present_lt_{min_present}",
                        "n_raw": len(genes),
                        "n_present": len(genes_present),
                    }
                )
                continue
            union_rows.append(
                manifest_row(
                    set_id=make_set_id("family_union", family, subfamily, "selected_atomic_sets"),
                    source_collection="derived_union",
                    source_url="derived_from_final_manifest_atomic_sets",
                    source_version=MSIGDB_RELEASE,
                    original_name=f"{family}:{subfamily}:selected_atomic_sets_union",
                    family=family,
                    subfamily=subfamily,
                    interpretation_layer="family_union",
                    priority="core",
                    rationale=f"Union of selected atomic sets for {family}/{subfamily}.",
                    genes_raw=genes,
                    genes_present=genes_present,
                    old_bundle_genes=old_bundle_genes,
                    hvg_genes=hvg_genes,
                    circularity_flag="derived_public_prior_union",
                    selector_id="derived_family_subfamily_union",
                    selector_regex="resolved_names_frozen_in_manifest",
                    broad_threshold=broad_threshold,
                    very_broad_threshold=very_broad_threshold,
                )
            )
        for family, group in atomic_df.groupby("family", sort=True):
            genes = sorted(set().union(*(set(str(x).split(";")) for x in group["genes_raw"])))
            genes = [g for g in genes if g]
            genes_present = sorted(intersect_present(genes, gene_universe))
            union_rows.append(
                manifest_row(
                    set_id=make_set_id("family_union", family, "all", "selected_atomic_sets"),
                    source_collection="derived_union",
                    source_url="derived_from_final_manifest_atomic_sets",
                    source_version=MSIGDB_RELEASE,
                    original_name=f"{family}:all:selected_atomic_sets_union",
                    family=family,
                    subfamily="all",
                    interpretation_layer="family_union",
                    priority="core",
                    rationale=f"Union of all selected atomic sets for {family}.",
                    genes_raw=genes,
                    genes_present=genes_present,
                    old_bundle_genes=old_bundle_genes,
                    hvg_genes=hvg_genes,
                    circularity_flag="derived_public_prior_union",
                    selector_id="derived_family_union",
                    selector_regex="resolved_names_frozen_in_manifest",
                    broad_threshold=broad_threshold,
                    very_broad_threshold=very_broad_threshold,
                )
            )

    anchor_rows: list[dict[str, Any]] = []
    for family, info in CORE_ANCHORS.items():
        genes_raw = sorted(anchor_genes(info["genes"], gene_universe))
        genes_present = sorted(intersect_present(genes_raw, gene_universe))
        set_id = make_set_id("core_anchor", family, str(info["subfamily"]), "expert_curated")
        if len(genes_present) < min_present:
            dropped_rows.append(
                {
                    "set_id": set_id,
                    "original_name": "expert_curated_core_anchors",
                    "source_collection": "expert_anchor",
                    "family": family,
                    "subfamily": info["subfamily"],
                    "reason": f"n_present_lt_{min_present}",
                    "n_raw": len(genes_raw),
                    "n_present": len(genes_present),
                }
            )
            continue
        anchor_rows.append(
            manifest_row(
                set_id=set_id,
                source_collection="expert_anchor",
                source_url="curated_in_build_expanded_stage0_bundle.py",
                source_version="manual_2026_06",
                original_name=f"{family}:expert_curated_core_anchors",
                family=family,
                subfamily=str(info["subfamily"]),
                interpretation_layer="core_anchor",
                priority="support",
                rationale=str(info["rationale"]),
                genes_raw=genes_raw,
                genes_present=genes_present,
                old_bundle_genes=old_bundle_genes,
                hvg_genes=hvg_genes,
                circularity_flag="expert_anchor_interpretation_support",
                selector_id="expert_core_anchor",
                selector_regex="not_regex_selected",
                broad_threshold=broad_threshold,
                very_broad_threshold=very_broad_threshold,
            )
        )

    manifest = pd.DataFrame(selected_rows + union_rows + anchor_rows)
    if not manifest.empty:
        manifest = manifest.sort_values(["family", "interpretation_layer", "subfamily", "source_collection", "original_name"]).reset_index(drop=True)
    dropped = pd.DataFrame(dropped_rows)
    return manifest, dropped, selector_provenance


def write_qc_reports(out_dir: Path, manifest: pd.DataFrame, dropped: pd.DataFrame, old_bundle_genes: set[str], hvg_genes: set[str]) -> None:
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    if manifest.empty:
        pd.DataFrame().to_csv(qc_dir / "gene_set_size_distribution.csv", index=False)
        return
    manifest[["set_id", "family", "subfamily", "interpretation_layer", "n_raw", "n_present", "broad_set_flag"]].to_csv(
        qc_dir / "gene_set_size_distribution.csv", index=False
    )
    manifest[["set_id", "family", "subfamily", "interpretation_layer", "overlap_old_bundle", "n_raw", "n_present"]].to_csv(
        qc_dir / "overlap_with_current_bundle.csv", index=False
    )
    manifest[["set_id", "family", "subfamily", "interpretation_layer", "overlap_hvg", "n_raw", "n_present"]].to_csv(
        qc_dir / "overlap_with_hvg.csv", index=False
    )
    coverage_rows = []
    for family, group in manifest.groupby("family", sort=True):
        genes = set()
        present = set()
        for _, row in group.iterrows():
            genes.update(g for g in str(row["genes_raw"]).split(";") if g)
            present.update(g for g in str(row["genes_present"]).split(";") if g)
        coverage_rows.append(
            {
                "family": family,
                "n_sets": int(len(group)),
                "n_unique_raw_genes": len(genes),
                "n_unique_present_genes": len(present),
                "overlap_old_bundle": len(genes & old_bundle_genes),
                "overlap_hvg": len(genes & hvg_genes),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(qc_dir / "family_gene_coverage.csv", index=False)
    all_genes = set()
    for genes in manifest["genes_raw"]:
        all_genes.update(g for g in str(genes).split(";") if g)
    pd.DataFrame({"gene": sorted(all_genes - old_bundle_genes)}).to_csv(qc_dir / "genes_new_to_expanded_bundle.csv", index=False)
    dropped.to_csv(qc_dir / "dropped_gene_sets.csv", index=False)


def write_readme(out_dir: Path, manifest: pd.DataFrame, resource_meta: list[dict[str, Any]], min_present: int) -> None:
    families = []
    if not manifest.empty:
        fam = manifest.groupby("family").agg(n_sets=("set_id", "count"), n_present=("n_present", "sum")).reset_index()
        families = [f"- `{row.family}`: {int(row.n_sets)} sets" for row in fam.itertuples(index=False)]
    lines = [
        "# Expanded Stage 0 Gene-Set Bundle",
        "",
        "This bundle expands the old-34 MSigDB Hallmark/Reactome/KEGG panel with public and local-prior gene sets for MRD malignant-vs-normal classification.",
        "",
        "## Biological Axes",
        "",
        *(families or ["- No sets were selected."]),
        "",
        "## Reproducibility",
        "",
        f"- MSigDB release: `{MSIGDB_RELEASE}`.",
        f"- Minimum expression-universe overlap: `n_present >= {min_present}`.",
        "- Regex selectors are frozen in `selector_provenance.json`; exact resolved names are frozen in `final_manifest.tsv`.",
        "- Public prior gene sets are primary evaluation resources; expert anchors are support/interpretation panels.",
        "- Same-cohort DE-derived sets are not used. Local huCIRA cytokine resources are treated as external perturbation priors when present.",
        "",
        "## Outputs",
        "",
        "- `final_manifest.tsv` / `final_manifest.csv`: full manifest with Stage 0 compatibility columns.",
        "- `final_bundle.gmt`: GMT keyed by `set_id`.",
        "- `resource_metadata.csv` / `.json`: URL/local path, retrieval date, file size, and SHA256.",
        "- `qc/`: set-size, overlap, coverage, novelty, and dropped-set reports.",
        "",
        "## Interpretation Guardrails",
        "",
        "Classifier performance should be described as a transcriptional footprint consistent with a biological process, not as causal evidence. Pooled-cell performance alone is not biological validation; sharedness and patient-specific results should be reviewed separately.",
        "",
        "## Resource Summary",
        "",
    ]
    for meta in resource_meta:
        lines.append(f"- `{meta['source_collection']}`: `{meta['local_path']}` (`sha256={meta['sha256']}`)")
    lines.append("")
    out_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT_H5AD)
    parser.add_argument("--old-bundle-gmt", type=Path, default=DEFAULT_OLD_BUNDLE)
    parser.add_argument("--hvg-ranked-csv", type=Path, default=DEFAULT_HVG_RANKED)
    parser.add_argument("--min-present", type=int, default=10)
    parser.add_argument("--broad-gene-threshold", type=int, default=750)
    parser.add_argument("--very-broad-gene-threshold", type=int, default=1000)
    parser.add_argument("--max-atomic-per-selector", type=int, default=80)
    parser.add_argument("--include-hucira", action="store_true", default=True)
    parser.add_argument("--no-include-hucira", action="store_false", dest="include_hucira")
    parser.add_argument("--overwrite-resources", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Use cached resources only; fail if a public file is absent.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    resource_specs = list(MSIGDB_RESOURCES)
    if args.include_hucira:
        resource_specs.extend(LOCAL_RESOURCES)

    resource_rows: list[dict[str, Any]] = []
    for spec in resource_specs:
        expected = cache_dir / spec.cache_name
        if args.skip_download and not spec.local_path and not expected.exists():
            raise FileNotFoundError(f"Missing cached resource with --skip-download: {expected}")
        path, meta = materialize_resource(spec, cache_dir, overwrite=bool(args.overwrite_resources))
        resource_rows.append(meta)

    gene_universe = load_gene_universe(args.input_h5ad.expanduser().resolve() if args.input_h5ad else None)
    old_bundle_genes = load_old_bundle(args.old_bundle_gmt.expanduser().resolve() if args.old_bundle_gmt else None)
    hvg_genes = load_hvg(args.hvg_ranked_csv.expanduser().resolve() if args.hvg_ranked_csv else None)

    manifest, dropped, selector_provenance = build_bundle(
        resources=resource_rows,
        gene_universe=gene_universe,
        old_bundle_genes=old_bundle_genes,
        hvg_genes=hvg_genes,
        min_present=int(args.min_present),
        broad_threshold=int(args.broad_gene_threshold),
        very_broad_threshold=int(args.very_broad_gene_threshold),
        max_atomic_per_selector=int(args.max_atomic_per_selector),
    )

    manifest_path = out_dir / "final_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    manifest.to_csv(out_dir / "final_manifest.csv", index=False)
    write_gmt(out_dir / "final_bundle.gmt", manifest)
    pd.DataFrame(resource_rows).to_csv(out_dir / "resource_metadata.csv", index=False)
    (out_dir / "resource_metadata.json").write_text(json.dumps(resource_rows, indent=2), encoding="utf-8")
    (out_dir / "selector_provenance.json").write_text(json.dumps(selector_provenance, indent=2), encoding="utf-8")
    write_qc_reports(out_dir, manifest, dropped, old_bundle_genes, hvg_genes)
    write_readme(out_dir, manifest, resource_rows, int(args.min_present))

    print(f"Wrote {manifest_path}")
    print(f"Wrote {out_dir / 'final_bundle.gmt'}")
    print(f"Selected sets: {len(manifest)}")
    print(f"Dropped candidate sets: {len(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
