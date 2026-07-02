from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import hucira as hc
except Exception:
    hc = None

HUCIRA_CYTOKINE_DICT_URL = "https://cdn.parsebiosciences.com/gigalab/10m/DEGs.csv"
HUCIRA_CIP_URL = "https://raw.githubusercontent.com/theislab/huCIRA/main/src/hucira/data/df_cips_genesets.csv"
HUCIRA_CYTOKINE_INFO_URL = (
    "https://raw.githubusercontent.com/theislab/huCIRA/main/src/hucira/data/"
    "20250125_cytokine_info_with_functional_classification_LV.xlsx"
)


DEFAULT_HUCIRA_PROGRAM_SUBSET = [
    "IFN",
    "TNF",
    "IL1",
    "IL10",
    "IL15",
    "IL32",
    "Antigen",
    "Myeloid",
]


def normalize_gene_symbol(value: object) -> str:
    text = str(value).strip()
    return text.upper() if text and text.lower() != "nan" else ""


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _make_gene_sets_from_long(reference_long: pd.DataFrame, positive_only: bool = True) -> dict[str, list[str]]:
    df = reference_long.copy()
    if positive_only:
        df = df[df["weight"] > 0].copy()
    else:
        df = df[df["weight"] < 0].copy()

    gene_sets: dict[str, list[str]] = {}
    for program_name, sub in df.groupby("program_name", observed=False):
        ordered = (
            sub.sort_values(["abs_weight", "gene"], ascending=[False, True])["gene"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if ordered:
            gene_sets[str(program_name)] = ordered
    return gene_sets


def _write_gmt(path: Path, gene_sets: dict[str, list[str]]) -> None:
    lines = []
    for program_name in sorted(gene_sets):
        genes = [g for g in gene_sets[program_name] if g]
        if not genes:
            continue
        lines.append("\t".join([program_name, "huCIRA"] + genes))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_cytokine_reference_long(
    cytokine_dict: pd.DataFrame,
    *,
    aggregation_level: str = "cytokine",
    adj_p_value_max: Optional[float] = 0.05,
    min_abs_log_fc: float = 0.0,
    exclude_well_biased: bool = True,
    min_genes_per_program: int = 10,
) -> pd.DataFrame:
    _require_columns(
        cytokine_dict,
        ["gene", "cytokine", "celltype", "log_fc"],
        "huCIRA cytokine dictionary",
    )

    d = cytokine_dict.copy()
    d["gene"] = d["gene"].map(normalize_gene_symbol)
    d["cytokine"] = d["cytokine"].astype(str).str.strip()
    d["celltype"] = d["celltype"].astype(str).str.strip()
    d["log_fc"] = pd.to_numeric(d["log_fc"], errors="coerce")
    if "adj_p_value" in d.columns and adj_p_value_max is not None:
        d["adj_p_value"] = pd.to_numeric(d["adj_p_value"], errors="coerce")
        d = d[d["adj_p_value"] <= float(adj_p_value_max)].copy()
    if exclude_well_biased and "well_biased" in d.columns:
        d = d[~d["well_biased"].fillna(False)].copy()
    if min_abs_log_fc > 0:
        d = d[d["log_fc"].abs() >= float(min_abs_log_fc)].copy()

    d = d[(d["gene"] != "") & d["cytokine"].ne("")].copy()
    if aggregation_level == "cytokine":
        d["program_name"] = d["cytokine"]
    elif aggregation_level == "cytokine_celltype":
        d["program_name"] = d["cytokine"] + "|" + d["celltype"]
    else:
        raise ValueError("aggregation_level must be 'cytokine' or 'cytokine_celltype'")

    out = (
        d.groupby(["program_name", "gene"], observed=False)
        .agg(
            weight=("log_fc", "mean"),
            abs_weight=("log_fc", lambda s: float(np.mean(np.abs(s)))),
            n_rows=("log_fc", "size"),
            n_celltypes=("celltype", "nunique"),
        )
        .reset_index()
    )
    out["source"] = "cytokine"
    out["program_type"] = "cytokine"
    out["aggregation_level"] = aggregation_level
    out = out.groupby("program_name", observed=False).filter(lambda x: len(x) >= min_genes_per_program).reset_index(drop=True)
    return out[
        [
            "source",
            "program_type",
            "aggregation_level",
            "program_name",
            "gene",
            "weight",
            "abs_weight",
            "n_rows",
            "n_celltypes",
        ]
    ]


def build_cip_reference_long(
    cip_df: pd.DataFrame,
    *,
    aggregation_level: str = "cip",
    min_abs_effect_size: float = 0.0,
    min_genes_per_program: int = 10,
) -> pd.DataFrame:
    _require_columns(cip_df, ["gene", "CIP", "celltype", "effect_size"], "huCIRA CIP signatures")

    d = cip_df.copy()
    d["gene"] = d["gene"].map(normalize_gene_symbol)
    d["CIP"] = d["CIP"].astype(str).str.strip()
    d["celltype"] = d["celltype"].astype(str).str.strip()
    d["effect_size"] = pd.to_numeric(d["effect_size"], errors="coerce")
    if min_abs_effect_size > 0:
        d = d[d["effect_size"].abs() >= float(min_abs_effect_size)].copy()
    d = d[(d["gene"] != "") & d["CIP"].ne("")].copy()

    if aggregation_level == "cip":
        d["program_name"] = d["CIP"]
    elif aggregation_level == "cip_celltype":
        d["program_name"] = d["CIP"] + "|" + d["celltype"]
    else:
        raise ValueError("aggregation_level must be 'cip' or 'cip_celltype'")

    out = (
        d.groupby(["program_name", "gene"], observed=False)
        .agg(
            weight=("effect_size", "mean"),
            abs_weight=("effect_size", lambda s: float(np.mean(np.abs(s)))),
            n_rows=("effect_size", "size"),
            n_celltypes=("celltype", "nunique"),
        )
        .reset_index()
    )
    out["source"] = "cip"
    out["program_type"] = "cip"
    out["aggregation_level"] = aggregation_level
    out = out.groupby("program_name", observed=False).filter(lambda x: len(x) >= min_genes_per_program).reset_index(drop=True)
    return out[
        [
            "source",
            "program_type",
            "aggregation_level",
            "program_name",
            "gene",
            "weight",
            "abs_weight",
            "n_rows",
            "n_celltypes",
        ]
    ]


def _fetch_hucira_tables_without_package(output_dir: Path, force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cytokine_dict_path = output_dir / "human_cytokine_dict.csv"
    cip_path = output_dir / "CIP_signatures.csv"
    cytokine_info_path = output_dir / "cytokine_info.xlsx"

    if force_rebuild or not cytokine_dict_path.exists():
        cytokine_dict = pd.read_csv(HUCIRA_CYTOKINE_DICT_URL, index_col=0).reset_index(drop=True)
        cytokine_dict.to_csv(cytokine_dict_path)
    else:
        cytokine_dict = pd.read_csv(cytokine_dict_path, index_col=0).reset_index(drop=True)

    if force_rebuild or not cip_path.exists():
        cip_df = pd.read_csv(HUCIRA_CIP_URL, index_col=0)
        cip_df.to_csv(cip_path, index=False)
    else:
        cip_df = pd.read_csv(cip_path)

    if force_rebuild or not cytokine_info_path.exists():
        cytokine_info = pd.read_excel(HUCIRA_CYTOKINE_INFO_URL, sheet_name="all_cytokines", engine="openpyxl")
        cytokine_info.to_excel(cytokine_info_path, index=False)
    else:
        cytokine_info = pd.read_excel(cytokine_info_path)

    return cytokine_dict, cip_df, cytokine_info


def export_hucira_reference_sets(
    output_dir: str | Path,
    *,
    force_rebuild: bool = False,
    cytokine_aggregation_level: str = "cytokine",
    cip_aggregation_level: str = "cip",
    cytokine_adj_p_value_max: Optional[float] = 0.05,
    cytokine_min_abs_log_fc: float = 0.0,
    cip_min_abs_effect_size: float = 0.0,
    min_genes_per_program: int = 10,
    write_gmt: bool = True,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_table_path = output_dir / "reference_table.csv"
    cytokine_long_path = output_dir / "cytokine_reference_long.csv"
    cip_long_path = output_dir / "cip_reference_long.csv"
    cytokine_up_json = output_dir / "cytokine_gene_sets_up.json"
    cytokine_down_json = output_dir / "cytokine_gene_sets_down.json"
    cip_up_json = output_dir / "cip_gene_sets_up.json"
    cip_down_json = output_dir / "cip_gene_sets_down.json"
    meta_json = output_dir / "reference_build_meta.json"

    if (
        not force_rebuild
        and reference_table_path.exists()
        and cytokine_long_path.exists()
        and cip_long_path.exists()
    ):
        return load_hucira_reference_assets(output_dir)

    if hc is not None:
        cytokine_dict = hc.load_human_cytokine_dict(save_dir=str(output_dir), force_download=force_rebuild)
        cip_df = hc.load_CIP_signatures(save_dir=str(output_dir), force_download=force_rebuild)
        cytokine_info = hc.load_cytokine_info(save_dir=str(output_dir), force_download=force_rebuild)
    else:
        cytokine_dict, cip_df, cytokine_info = _fetch_hucira_tables_without_package(
            output_dir,
            force_rebuild=force_rebuild,
        )

    cytokine_long = build_cytokine_reference_long(
        cytokine_dict,
        aggregation_level=cytokine_aggregation_level,
        adj_p_value_max=cytokine_adj_p_value_max,
        min_abs_log_fc=cytokine_min_abs_log_fc,
        min_genes_per_program=min_genes_per_program,
    )
    cip_long = build_cip_reference_long(
        cip_df,
        aggregation_level=cip_aggregation_level,
        min_abs_effect_size=cip_min_abs_effect_size,
        min_genes_per_program=min_genes_per_program,
    )

    reference_table = pd.concat([cytokine_long, cip_long], ignore_index=True)
    reference_table.to_csv(reference_table_path, index=False)
    cytokine_long.to_csv(cytokine_long_path, index=False)
    cip_long.to_csv(cip_long_path, index=False)
    cytokine_info.to_csv(output_dir / "cytokine_info.csv", index=False)

    cytokine_up = _make_gene_sets_from_long(cytokine_long, positive_only=True)
    cytokine_down = _make_gene_sets_from_long(cytokine_long, positive_only=False)
    cip_up = _make_gene_sets_from_long(cip_long, positive_only=True)
    cip_down = _make_gene_sets_from_long(cip_long, positive_only=False)

    _write_json(cytokine_up_json, cytokine_up)
    _write_json(cytokine_down_json, cytokine_down)
    _write_json(cip_up_json, cip_up)
    _write_json(cip_down_json, cip_down)

    if write_gmt:
        _write_gmt(output_dir / "cytokine_gene_sets_up.gmt", cytokine_up)
        _write_gmt(output_dir / "cytokine_gene_sets_down.gmt", cytokine_down)
        _write_gmt(output_dir / "cip_gene_sets_up.gmt", cip_up)
        _write_gmt(output_dir / "cip_gene_sets_down.gmt", cip_down)

    _write_json(
        meta_json,
        {
            "cytokine_aggregation_level": cytokine_aggregation_level,
            "cip_aggregation_level": cip_aggregation_level,
            "cytokine_adj_p_value_max": cytokine_adj_p_value_max,
            "cytokine_min_abs_log_fc": cytokine_min_abs_log_fc,
            "cip_min_abs_effect_size": cip_min_abs_effect_size,
            "min_genes_per_program": min_genes_per_program,
            "reference_rows": int(reference_table.shape[0]),
            "cytokine_programs": int(cytokine_long["program_name"].nunique()),
            "cip_programs": int(cip_long["program_name"].nunique()),
        },
    )

    return load_hucira_reference_assets(output_dir)


def load_hucira_reference_assets(output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    reference_table = pd.read_csv(output_dir / "reference_table.csv")
    cytokine_long = pd.read_csv(output_dir / "cytokine_reference_long.csv")
    cip_long = pd.read_csv(output_dir / "cip_reference_long.csv")

    return {
        "output_dir": output_dir,
        "reference_table": reference_table,
        "cytokine_long": cytokine_long,
        "cip_long": cip_long,
        "cytokine_gene_sets_up": json.loads((output_dir / "cytokine_gene_sets_up.json").read_text())
        if (output_dir / "cytokine_gene_sets_up.json").exists()
        else {},
        "cytokine_gene_sets_down": json.loads((output_dir / "cytokine_gene_sets_down.json").read_text())
        if (output_dir / "cytokine_gene_sets_down.json").exists()
        else {},
        "cip_gene_sets_up": json.loads((output_dir / "cip_gene_sets_up.json").read_text())
        if (output_dir / "cip_gene_sets_up.json").exists()
        else {},
        "cip_gene_sets_down": json.loads((output_dir / "cip_gene_sets_down.json").read_text())
        if (output_dir / "cip_gene_sets_down.json").exists()
        else {},
    }


def select_reference_programs(
    reference_long: pd.DataFrame,
    *,
    include_terms: Optional[Iterable[str]] = None,
    min_genes_per_program: Optional[int] = None,
) -> pd.DataFrame:
    out = reference_long.copy()
    if include_terms:
        patterns = [str(term).lower() for term in include_terms]
        mask = out["program_name"].astype(str).str.lower().apply(lambda x: any(term in x for term in patterns))
        out = out[mask].copy()
    if min_genes_per_program is not None:
        out = out.groupby("program_name", observed=False).filter(lambda x: len(x) >= int(min_genes_per_program)).reset_index(drop=True)
    return out


def _prepare_signature_matrix(signatures_df: pd.DataFrame) -> pd.DataFrame:
    sig = signatures_df.copy()
    sig.index = sig.index.map(normalize_gene_symbol)
    sig = sig[sig.index != ""].copy()
    sig = sig.groupby(level=0).mean()
    return sig


def reference_long_to_matrix(reference_long: pd.DataFrame) -> pd.DataFrame:
    ref = reference_long.copy()
    ref["gene"] = ref["gene"].map(normalize_gene_symbol)
    ref = ref[ref["gene"] != ""].copy()
    mat = ref.pivot_table(index="gene", columns="program_name", values="weight", aggfunc="mean", fill_value=0.0)
    return mat.sort_index(axis=0).sort_index(axis=1)


def compute_signature_program_similarity(
    signatures_df: pd.DataFrame,
    reference_long: pd.DataFrame,
    *,
    metrics: Iterable[str] = ("cosine", "pearson"),
    min_overlap: int = 10,
) -> dict[str, pd.DataFrame]:
    sig = _prepare_signature_matrix(signatures_df)
    ref = reference_long_to_matrix(reference_long)
    common = sig.index.intersection(ref.index)
    if len(common) < min_overlap:
        raise ValueError(f"Only {len(common)} overlapping genes found; need at least {min_overlap}.")

    sig = sig.loc[common]
    ref = ref.loc[common]

    overlap = (
        pd.DataFrame((sig.ne(0)).astype(int), index=common, columns=sig.columns).T @
        pd.DataFrame((ref.ne(0)).astype(int), index=common, columns=ref.columns)
    )
    overlap = overlap.astype(int)

    results: dict[str, pd.DataFrame] = {"overlap_n": overlap}
    requested = {metric.lower() for metric in metrics}

    if "cosine" in requested:
        sig_vals = sig.to_numpy(dtype=float)
        ref_vals = ref.to_numpy(dtype=float)
        sig_norm = np.linalg.norm(sig_vals, axis=0, keepdims=True)
        ref_norm = np.linalg.norm(ref_vals, axis=0, keepdims=True)
        sig_norm[sig_norm == 0] = 1.0
        ref_norm[ref_norm == 0] = 1.0
        cosine = (sig_vals.T @ ref_vals) / (sig_norm.T @ ref_norm)
        results["cosine"] = pd.DataFrame(cosine, index=sig.columns, columns=ref.columns)

    if "pearson" in requested:
        pearson = pd.DataFrame(index=sig.columns, columns=ref.columns, dtype=float)
        for sig_name in sig.columns:
            for prog_name in ref.columns:
                pearson.loc[sig_name, prog_name] = sig[sig_name].corr(ref[prog_name], method="pearson")
        results["pearson"] = pearson

    return results


def compute_signature_program_jaccard(
    signatures_df: pd.DataFrame,
    reference_long: pd.DataFrame,
    *,
    top_n: int = 200,
    direction: str = "positive",
) -> pd.DataFrame:
    sig = _prepare_signature_matrix(signatures_df)
    ref = reference_long.copy()

    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be 'positive' or 'negative'")

    if direction == "positive":
        sig_sets = {col: set(sig[col].nlargest(top_n).index.astype(str)) for col in sig.columns}
        ref = ref[ref["weight"] > 0].copy()
    else:
        sig_sets = {col: set(sig[col].nsmallest(top_n).index.astype(str)) for col in sig.columns}
        ref = ref[ref["weight"] < 0].copy()

    ref_sets = {}
    for program_name, sub in ref.groupby("program_name", observed=False):
        ranked = sub.sort_values("abs_weight", ascending=False)["gene"].astype(str).drop_duplicates().head(top_n)
        ref_sets[str(program_name)] = set(ranked)

    out = pd.DataFrame(index=sig.columns, columns=sorted(ref_sets), dtype=float)
    for sig_name, sig_set in sig_sets.items():
        for prog_name, prog_set in ref_sets.items():
            union_size = len(sig_set | prog_set)
            out.loc[sig_name, prog_name] = len(sig_set & prog_set) / union_size if union_size > 0 else np.nan
    return out


def summarize_top_program_matches(score_df: pd.DataFrame, *, top_k: int = 5, score_name: str = "score") -> pd.DataFrame:
    rows = []
    for model_uid, row in score_df.iterrows():
        ranked = row.dropna().sort_values(ascending=False).head(top_k)
        for rank, (program_name, value) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "model_uid": model_uid,
                    "rank": rank,
                    "program_name": str(program_name),
                    score_name: float(value),
                }
            )
    return pd.DataFrame(rows)
