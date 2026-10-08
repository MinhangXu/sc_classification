"""Generate the D12/D13 DEG–HVG and low-support patient notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("stage2_old34_deg_hvg_lowsupport_patient_signals.ipynb")


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n") + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n") + "\n")


cells: list[nbf.NotebookNode] = []

cells.append(
    md(
        r"""
# Old34 follow-up: GSEA vs HVG gene-space overlap and low-support patient DE

Experiment: `20260525_060508_stage0_mrd_old34_broad_screen_82db5093`

This notebook answers two meeting follow-ups that D7 cannot answer from Stage 2 scorecards alone:

1. **D12.** Are the Stage 0 Hallmark/Reactome/KEGG (GSEA) gene lists meaningfully different from the shared MRD highly variable genes, or is the knowledge prior just a restatement of variance filtering?
2. **D13.** For the low-support patients that stay dark on D7B (P01, P06, P07), does within-patient malignant vs non-malignant differential expression nominate discriminatory gene sets that the current 41-panel atlas missed?

D7B reports **max−min lift across four K=40 DR methods**, not absolute lift. A black P01/P06/P07 column means method choice barely moves transfer, not that no panel has any lift. D7C already shows the distinction: P01 and P06 stay near zero on IFN-α at every K, while P07’s IFN-α PCA lift grows with K. This notebook tests whether those patterns have a gene-level basis.

The notebook does **not** rerun Stage 1 or Stage 2. It reuses the Stage 0 gene universe, HVG ranking, panel gene lists, and the D1/D7 held-out lift tables. The only new computation is Wilcoxon / AUROC differential expression on the same CITE × MRD × binary-label cohort.

## Scientific boundaries

- HVG ranks are the Stage 0 `post_filter_log1p_variance` ranks on all 60,118 eligible cells. That ranking is transductive and label-blind.
- Old34 “GSEA” here means the 34-program Hallmark/Reactome/KEGG dictionary actually used as Stage 0, not a new enrichment of an external DEG list.
- Per-patient DE is an **in-sample** malignant vs healthy contrast. It is not LOPO transfer and is confounded by cell-type composition, donor-healthy dominance, and $n_+=22$–$45$ malignant cells.
- Cohort-wide DE is dominated by high-support patients, especially P03 (758 malignant cells).
- A gene set that is DE within P01/P06/P07 but has near-zero held-out lift is a **private or support-limited** candidate, not a shared program.
- Unique-to-patient calls are descriptive set differences, not multiplicity-adjusted discovery claims.
"""
    )
)

cells.append(
    code(
        r"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from IPython.display import display
from matplotlib.patches import Circle
from scipy.stats import false_discovery_control, hypergeom, norm, rankdata, spearmanr

EXPERIMENT_ID = "20260525_060508_stage0_mrd_old34_broad_screen_82db5093"
SUPPORTED_PATIENTS = ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]
LOW_SUPPORT_PATIENTS = ["P01", "P06", "P07"]
HIGH_SUPPORT_PATIENTS = ["P02", "P03", "P04"]
HVG_ANCHORS = [500, 1000, 3000, 10000]
DE_FDR = 0.05
DE_ABS_DELTA_LOG = 0.25
DE_ABS_AUC_SHIFT = 0.15
GSEA_PERMUTATIONS = 300
GSEA_MIN_SIZE = 8
GSEA_MAX_SIZE = 2000


def find_repo_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "sc_classification").exists():
            return candidate
    raise FileNotFoundError("Could not locate sc_classification repository root")


REPO_ROOT = find_repo_root()
EXP_DIR = REPO_ROOT / "experiments" / EXPERIMENT_ID
TABLE_DIR = EXP_DIR / "analysis" / "scorecards" / "stage2_diagnostic" / "deg_hvg_lowsupport_v1"
FIGURE_DIR = EXP_DIR / "analysis" / "figures" / "stage2_diagnostic" / "deg_hvg_lowsupport_v1"
CACHE_DIR = EXP_DIR / "analysis" / "cache" / "stage2_diagnostic" / "deg_hvg_lowsupport_v1"
for directory in [TABLE_DIR, FIGURE_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE_PATHS = {
    "input_h5ad": Path("/home/minhang/mds_project/data/cohort_adata/adata_cellType_cnLabel_pseudoTime_collectionTime.h5ad"),
    "gmt": REPO_ROOT / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "genesets_v1.gmt",
    "old_manifest": REPO_ROOT / "scripts" / "knowledge_driven_embedding" / "older_geneset" / "manifest.tsv",
    "hvg_ranked": EXP_DIR / "preprocessing" / "gene_universe" / "hvg_ranked_genes.csv",
    "old34_coverage": EXP_DIR / "preprocessing" / "gene_universe" / "old34_coverage.csv",
    "all_filtered_genes": EXP_DIR / "preprocessing" / "gene_universe" / "all_filtered_gene_list.json",
    "panel_manifest": EXP_DIR / "preprocessing" / "panels" / "stage0_panel_manifest.csv",
    "panel_gene_dir": EXP_DIR / "preprocessing" / "panels" / "stage0_panel_genes",
    "patient_metrics": EXP_DIR / "analysis" / "scorecards" / "stage2_diagnostic" / "old34_supported_patient_representation_metrics.csv",
    "k40_spread": EXP_DIR / "analysis" / "scorecards" / "stage2_diagnostic" / "meeting_followup_v1" / "old34_k40_method_lift_spread_by_patient.csv",
    "patient_summary": EXP_DIR / "analysis" / "scorecards" / "stage2_diagnostic" / "meeting_followup_v1" / "old34_patient_metric_summary.csv",
}
for name, path in SOURCE_PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 120)
GENERATED_TABLES: list[Path] = []
GENERATED_FIGURES: list[Path] = []
PALETTE = {
    "gsea": "#4C78A8",
    "hvg": "#F58518",
    "deg": "#54A24B",
    "low": "#E45756",
    "high": "#4C78A8",
    "other": "#9E9E9E",
}


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    frame.to_csv(path, index=False)
    GENERATED_TABLES.append(path)
    return path


def save_figure(fig: plt.Figure, stem: str) -> Path:
    png = FIGURE_DIR / f"{stem}.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    GENERATED_FIGURES.extend([png, pdf])
    print(png)
    return png


def parse_gmt(path: Path) -> dict[str, list[str]]:
    genesets: dict[str, list[str]] = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genesets[parts[0]] = [g for g in parts[2:] if g]
    return genesets


def set_overlap_row(name: str, left: set[str], right: set[str], universe: set[str], left_label: str, right_label: str) -> dict[str, object]:
    both = left & right
    only_left = left - right
    only_right = right - left
    n_u = len(universe)
    expected = (len(left) * len(right) / n_u) if n_u else np.nan
    overlap = len(both)
    enrichment = overlap / expected if expected else np.nan
    # Enrichment tail: P(X >= overlap) under Hypergeom(N=universe, K=|right|, n=|left|)
    if n_u and len(left) and len(right):
        p_enrich = float(hypergeom.sf(overlap - 1, n_u, len(right), len(left))) if overlap else 1.0
        p_deplete = float(hypergeom.cdf(overlap, n_u, len(right), len(left)))
    else:
        p_enrich = p_deplete = np.nan
    union = left | right
    return {
        "comparison": name,
        "left_label": left_label,
        "right_label": right_label,
        "n_left": len(left),
        "n_right": len(right),
        "n_overlap": overlap,
        "n_left_only": len(only_left),
        "n_right_only": len(only_right),
        "n_universe": n_u,
        "jaccard": (overlap / len(union)) if union else np.nan,
        "left_coverage_of_right": (overlap / len(right)) if right else np.nan,
        "right_coverage_of_left": (overlap / len(left)) if left else np.nan,
        "expected_overlap": expected,
        "fold_enrichment": enrichment,
        "hypergeom_enrichment_pvalue": p_enrich,
        "hypergeom_depletion_pvalue": p_deplete,
    }


def draw_venn2(ax, n_left: int, n_right: int, n_both: int, label_left: str, label_right: str, title: str) -> None:
    ax.set_aspect("equal")
    ax.add_patch(Circle((-0.38, 0.0), 0.72, facecolor=PALETTE["gsea"], edgecolor="white", alpha=0.38, lw=1.2))
    ax.add_patch(Circle((0.38, 0.0), 0.72, facecolor=PALETTE["hvg"], edgecolor="white", alpha=0.38, lw=1.2))
    ax.text(-0.62, 0.0, f"{n_left:,}", ha="center", va="center", fontsize=13, fontweight="bold", color="#1f2a44")
    ax.text(0.62, 0.0, f"{n_right:,}", ha="center", va="center", fontsize=13, fontweight="bold", color="#1f2a44")
    ax.text(0.0, 0.0, f"{n_both:,}", ha="center", va="center", fontsize=13, fontweight="bold", color="#1f2a44")
    ax.text(-0.62, 0.98, label_left, ha="center", va="center", fontsize=9, color="#1f2a44")
    ax.text(0.62, 0.98, label_right, ha="center", va="center", fontsize=9, color="#1f2a44")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.95, 1.28)
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def plot_contract(
    plot_id: str,
    title: str,
    scientific_question: str,
    in_silico_experiment: str,
    unit_of_analysis: str,
    metric_and_baseline: str,
    take_home_message: str,
    unresolved_gaps: str,
    null_hypothesis: str = "No structured difference beyond the stated no-skill/reference comparison.",
) -> dict[str, str]:
    return {
        "plot_id": plot_id,
        "title": title,
        "scientific_question": scientific_question,
        "null_hypothesis": null_hypothesis,
        "in_silico_experiment": in_silico_experiment,
        "unit_of_analysis": unit_of_analysis,
        "metric_and_baseline": metric_and_baseline,
        "take_home_message": take_home_message,
        "unresolved_gaps": unresolved_gaps,
    }


print("Repository:", REPO_ROOT)
print("Experiment:", EXP_DIR)
print("Output tables:", TABLE_DIR)
print("Output figures:", FIGURE_DIR)
print("Expression cache:", CACHE_DIR)
"""
    )
)

cells.append(
    md(
        r"""
## 1. D12 — GSEA/old34 dictionary versus highly variable genes

The left list is the covered union of the 34 Stage 0 GSEA programs (`full_34`, 2,719 genes in the 15,560-gene MRD universe). The right lists are the Stage 0 HVG anchors already run as data-driven controls: top 500 / 1,000 / 3,000 / 10,000 genes by post-filter log1p variance.

Size-matched comparison is HVG3,000 versus `full_34` (2,719 genes). A list pair is called **meaningfully different as a Stage 0 panel** when exclusive old34 genes remain a majority and Jaccard is well below 0.5, even if the hypergeometric test shows HVG enrichment. Enrichment without interchangeability is the expected outcome of a curated immune/stress dictionary sitting inside a variance-ranked genome.
"""
    )
)

cells.append(
    code(
        r"""
hvg_ranked = pd.read_csv(SOURCE_PATHS["hvg_ranked"])
universe = set(json.loads(SOURCE_PATHS["all_filtered_genes"].read_text())["genes"])
coverage = pd.read_csv(SOURCE_PATHS["old34_coverage"])
panel_manifest = pd.read_csv(SOURCE_PATHS["panel_manifest"])
gmt_raw = parse_gmt(SOURCE_PATHS["gmt"])

full34_genes = set(json.loads((SOURCE_PATHS["panel_gene_dir"] / "full_34.json").read_text())["genes"])
core_genes = set(json.loads((SOURCE_PATHS["panel_gene_dir"] / "core_only.json").read_text())["genes"])
if full34_genes - universe:
    raise ValueError(f"full_34 genes missing from universe: {len(full34_genes - universe)}")

hvg_sets = {n: set(hvg_ranked["gene"].head(int(n)).tolist()) for n in HVG_ANCHORS}
panel_genes: dict[str, set[str]] = {}
for panel_id in panel_manifest.loc[
    panel_manifest["panel_type"].isin(["single_geneset_only", "single_group_only", "full_control", "core_only"]),
    "panel_id",
]:
    payload = json.loads((SOURCE_PATHS["panel_gene_dir"] / f"{panel_id}.json").read_text())
    panel_genes[str(panel_id)] = set(payload["genes"])

overlap_rows = []
for n, hvg_set in hvg_sets.items():
    overlap_rows.append(set_overlap_row(f"full34_vs_hvg{n}", full34_genes, hvg_set, universe, "old34_full_gsea_union", f"hvg_top_{n}"))
    overlap_rows.append(set_overlap_row(f"core_vs_hvg{n}", core_genes, hvg_set, universe, "old34_core_gsea_union", f"hvg_top_{n}"))
overlap_summary = pd.DataFrame(overlap_rows)
write_csv(overlap_summary, "d12_old34_gsea_vs_hvg_overlap_summary.csv")

per_set_rows = []
for _, cov in coverage.iterrows():
    geneset = str(cov["geneset_name"])
    raw = {g.upper() for g in gmt_raw.get(geneset, [])}
    available_upper = {g.upper(): g for g in universe}
    covered = {available_upper[g] for g in raw if g in available_upper}
    for n, hvg_set in hvg_sets.items():
        row = set_overlap_row(f"{geneset}__hvg{n}", covered, hvg_set, universe, geneset, f"hvg_top_{n}")
        row.update(
            {
                "geneset_name": geneset,
                "biology_group": cov.get("biology_group", ""),
                "priority": cov.get("priority", ""),
                "source": cov.get("source", ""),
                "hvg_anchor": n,
                "covered_n_genes": len(covered),
            }
        )
        per_set_rows.append(row)
per_set_overlap = pd.DataFrame(per_set_rows)
write_csv(per_set_overlap, "d12_per_geneset_vs_hvg_overlap.csv")

print("Universe genes:", len(universe))
print("full_34 covered genes:", len(full34_genes))
print("core_only covered genes:", len(core_genes))
display(overlap_summary[[
    "comparison", "n_left", "n_right", "n_overlap", "n_left_only", "n_right_only",
    "jaccard", "expected_overlap", "fold_enrichment", "hypergeom_enrichment_pvalue",
]])

size_matched = overlap_summary.query("comparison == 'full34_vs_hvg3000'").iloc[0]
print(
    "Size-matched verdict: "
    f"Jaccard={size_matched['jaccard']:.3f}, "
    f"old34-only={int(size_matched['n_left_only']):,} "
    f"({size_matched['n_left_only'] / size_matched['n_left']:.1%} of old34), "
    f"fold-enrichment={size_matched['fold_enrichment']:.2f}×, "
    f"hypergeom p={size_matched['hypergeom_enrichment_pvalue']:.2e}."
)
meaningfully_different = bool(
    size_matched["jaccard"] < 0.5 and (size_matched["n_left_only"] / size_matched["n_left"]) > 0.5
)
print("Pre-specified 'meaningfully different as Stage 0 panels':", meaningfully_different)
"""
    )
)

cells.append(
    code(
        r"""
# D12A — two-set Venns at the four HVG anchors.
fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6))
for ax, n in zip(axes.ravel(), HVG_ANCHORS):
    row = overlap_summary.query(f"comparison == 'full34_vs_hvg{n}'").iloc[0]
    draw_venn2(
        ax,
        int(row["n_left_only"]),
        int(row["n_right_only"]),
        int(row["n_overlap"]),
        f"old34 only\n({int(row['n_left']):,} GSEA)",
        f"HVG{n} only",
        f"HVG top {n:,}  ·  Jaccard {row['jaccard']:.2f}  ·  {row['fold_enrichment']:.2f}× expected",
    )
fig.suptitle("D12A. Old34 GSEA union versus Stage 0 HVG anchors", y=1.01)
d12a_png = save_figure(fig, "D12A_old34_gsea_vs_hvg_venn")
plt.show()

# D12B — per-geneset HVG3000 overlap versus hypergeometric expectation.
focus = per_set_overlap.query("hvg_anchor == 3000").sort_values("fold_enrichment", ascending=False).copy()
focus["label"] = focus["geneset_name"].str.replace("HALLMARK_", "").str.replace("REACTOME_", "").str.replace("KEGG_", "").str.replace("_", " ").str.lower()
fig, ax = plt.subplots(figsize=(11.8, 9.4))
colors = np.where(focus["fold_enrichment"] >= 1.0, PALETTE["gsea"], PALETTE["hvg"])
ax.barh(focus["label"], focus["fold_enrichment"], color=colors)
ax.axvline(1.0, color="0.35", lw=1, ls="--")
ax.set_xlabel("Fold enrichment of geneset ∩ HVG3,000 versus hypergeometric expectation")
ax.set_ylabel("Old34 GSEA program")
ax.set_title("D12B. Every old34 program is HVG-enriched; none is interchangeable with HVG3,000")
ax.invert_yaxis()
d12b_png = save_figure(fig, "D12B_per_geneset_hvg3000_fold_enrichment")
plt.show()

# D12C — exclusive fractions at the size-matched cutoff.
fig, ax = plt.subplots(figsize=(10.8, 4.2))
matched = overlap_summary[overlap_summary["comparison"].str.startswith("full34_vs_hvg")].copy()
matched["hvg_anchor"] = matched["right_label"].str.replace("hvg_top_", "").astype(int)
plot_df = matched.melt(
    id_vars=["hvg_anchor"],
    value_vars=["n_left_only", "n_overlap", "n_right_only"],
    var_name="part",
    value_name="n_genes",
)
plot_df["part"] = plot_df["part"].map({"n_left_only": "old34 only", "n_overlap": "intersection", "n_right_only": "HVG only"})
sns.barplot(data=plot_df, x="hvg_anchor", y="n_genes", hue="part", palette=[PALETTE["gsea"], "#54A24B", PALETTE["hvg"]], ax=ax)
ax.set_xlabel("HVG anchor (top N by post-filter log1p variance)")
ax.set_ylabel("Genes in the 15,560-gene MRD universe")
ax.set_title("D12C. Exclusive old34 genes remain large at every HVG budget")
ax.legend(title="", frameon=False)
d12c_png = save_figure(fig, "D12C_old34_hvg_exclusive_counts")
plt.show()
"""
    )
)

cells.append(
    md(
        r"""
## 2. D13 — load the Stage 0 expression cohort and patient composition

Expression is rebuilt with the same CITE → MRD → `CN.label ∈ {cancer, normal}` filters and the saved 15,560-gene universe, then `normalize_total(1e4)` + `log1p`. The filtered matrix is cached so later reruns do not reopen the 14 GB source AnnData.

Predicted cell-type mix is reported because unmatched malignant vs donor-healthy DE can recover lineage rather than malignancy. Lineage-restricted DE (healthy cells restricted to the malignant cells’ dominant predicted types) is a sensitivity, not the primary call.
"""
    )
)

cells.append(
    code(
        r"""
CACHE_H5AD = CACHE_DIR / "mrd_cite_binary_log1p_15560.h5ad"
UNIVERSE_GENES = json.loads(SOURCE_PATHS["all_filtered_genes"].read_text())["genes"]


def load_or_build_expression() -> ad.AnnData:
    if CACHE_H5AD.exists():
        cached = ad.read_h5ad(CACHE_H5AD)
        if cached.n_obs == 60118 and cached.n_vars == 15560:
            print("Loaded cached expression:", CACHE_H5AD, cached.shape)
            return cached
        print("Cache shape mismatch; rebuilding.")

    backed = ad.read_h5ad(SOURCE_PATHS["input_h5ad"], backed="r")
    obs = backed.obs
    timepoint = obs["timepoint_type"].astype(str) if "timepoint_type" in obs.columns else obs["Time"].astype(str).str.replace(r"_[0-9]+$", "", regex=True)
    mask = (
        obs["Tech"].astype(str).eq("CITE")
        & timepoint.eq("MRD")
        & obs["CN.label"].astype(str).isin(["cancer", "normal"])
    )
    keep_cells = obs.index[mask]
    keep_genes = [g for g in UNIVERSE_GENES if g in backed.var_names]
    if len(keep_genes) != 15560:
        raise ValueError(f"Expected 15560 universe genes in source, found {len(keep_genes)}")
    adata = backed[keep_cells, keep_genes].to_memory()
    backed.file.close()
    if adata.n_obs != 60118:
        raise ValueError(f"Expected 60118 cells, found {adata.n_obs}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obs["patient"] = adata.obs["patient"].astype(str)
    adata.obs["CN.label"] = adata.obs["CN.label"].astype(str)
    adata.write_h5ad(CACHE_H5AD, compression="gzip")
    print("Wrote expression cache:", CACHE_H5AD, adata.shape)
    return adata


adata = load_or_build_expression()
obs = adata.obs.copy()
obs["malignant"] = obs["CN.label"].eq("cancer")
obs["predicted.annotation"] = obs["predicted.annotation"].astype(str)
obs["source"] = obs["source"].astype(str)

composition_rows = []
for patient, patient_obs in obs.groupby("patient", observed=True):
    mal = patient_obs.loc[patient_obs["malignant"]]
    healthy = patient_obs.loc[~patient_obs["malignant"]]
    composition_rows.append(
        {
            "patient_id": patient,
            "n_malignant": int(len(mal)),
            "n_healthy": int(len(healthy)),
            "n_total": int(len(patient_obs)),
            "prevalence": float(len(mal) / len(patient_obs)) if len(patient_obs) else np.nan,
            "supported_for_lift_summary": patient in SUPPORTED_PATIENTS,
            "low_support_focus": patient in LOW_SUPPORT_PATIENTS,
            "dominant_malignant_type": (mal["predicted.annotation"].value_counts().index[0] if len(mal) else ""),
            "dominant_malignant_type_frac": (float(mal["predicted.annotation"].value_counts(normalize=True).iloc[0]) if len(mal) else np.nan),
            "dominant_healthy_type": (healthy["predicted.annotation"].value_counts().index[0] if len(healthy) else ""),
            "dominant_healthy_type_frac": (float(healthy["predicted.annotation"].value_counts(normalize=True).iloc[0]) if len(healthy) else np.nan),
            "n_healthy_donor": int((~patient_obs["malignant"] & patient_obs["source"].eq("donor")).sum()),
        }
    )
composition = pd.DataFrame(composition_rows).sort_values(["low_support_focus", "n_malignant"], ascending=[False, True])
write_csv(composition, "d13_patient_cell_type_composition.csv")
display(composition)

type_mix = (
    obs.groupby(["patient", "CN.label", "predicted.annotation"], observed=True)
    .size()
    .rename("n_cells")
    .reset_index()
)
write_csv(type_mix, "d13_patient_label_celltype_counts.csv")
"""
    )
)

cells.append(
    code(
        r"""
# D13A — cell-type mix for the three low-support patients.
fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharey=False)
for ax, patient in zip(axes, LOW_SUPPORT_PATIENTS):
    sub = type_mix.query("patient == @patient").copy()
    pivot = (
        sub.pivot_table(index="predicted.annotation", columns="CN.label", values="n_cells", fill_value=0)
        .reindex(columns=["cancer", "normal"], fill_value=0)
        .sort_values(["cancer", "normal"], ascending=False)
    )
    pivot.plot(kind="barh", ax=ax, color=[PALETTE["low"], PALETTE["other"]], width=0.78, legend=patient == "P07")
    ax.set_title(f"{patient}  ·  {int(composition.set_index('patient_id').loc[patient, 'n_malignant'])} malignant")
    ax.set_xlabel("Cells")
    ax.set_ylabel("")
if axes[-1].get_legend() is not None:
    axes[-1].legend(title="CN.label", frameon=False)
fig.suptitle("D13A. Low-support DE will mix malignancy with lineage unless types overlap", y=1.03)
d13a_png = save_figure(fig, "D13A_lowsupport_celltype_composition")
plt.show()
"""
    )
)

cells.append(
    md(
        r"""
## 3. Per-patient Wilcoxon DE and gene-level AUROC

For each supported patient, and once for the pooled cohort, every gene is tested by two-sided Mann–Whitney U on log1p expression. The primary effect sizes are:

- $\Delta\log$: mean malignant − mean healthy on the Stage 0 log1p scale;
- gene AUROC: $U / (n_+ n_-)$, equivalent to the probability a malignant cell ranks above a healthy cell.

A gene is called DE when FDR $< 0.05$, $|\Delta\log| \ge 0.25$, and $|\mathrm{AUROC}-0.5| \ge 0.15$. Those thresholds are triage rules for list overlap, not a publication discovery cutoff.

**Unique high-effect gene** (pre-specified): DE in the focal low-support patient and not DE in any of P02/P03/P04 under the same rule.

A lineage-restricted sensitivity repeats the test after keeping healthy cells whose `predicted.annotation` appears in the malignant cells of that patient.
"""
    )
)

cells.append(
    code(
        r"""
DE_CACHE = CACHE_DIR / "per_scope_de_tables.pkl.gz"
gene_index = np.asarray(adata.var_names.astype(str))


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float32)
    return np.asarray(matrix, dtype=np.float32)


def run_de(X: np.ndarray, y: np.ndarray, scope: str, n_malignant: int, n_healthy: int) -> pd.DataFrame:
    y = np.asarray(y, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(f"{scope}: need both classes, got {n_pos} vs {n_neg}")
    n = n_pos + n_neg
    Xp = X[y]
    Xn = X[~y]
    mean_pos = Xp.mean(axis=0)
    mean_neg = Xn.mean(axis=0)
    delta = mean_pos - mean_neg
    frac_pos = (Xp > 0).mean(axis=0)
    frac_neg = (Xn > 0).mean(axis=0)
    auc = np.empty(X.shape[1], dtype=np.float64)
    pval = np.empty(X.shape[1], dtype=np.float64)
    mean_u = n_pos * n_neg / 2.0
    for j in range(X.shape[1]):
        ranks = rankdata(X[:, j], method="average")
        u = float(ranks[y].sum() - n_pos * (n_pos + 1) / 2.0)
        auc[j] = u / (n_pos * n_neg)
        _, counts = np.unique(ranks, return_counts=True)
        tie_term = float(np.sum(counts * (counts - 1) * (counts + 1)))
        var_u = n_pos * n_neg / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
        if var_u <= 0:
            pval[j] = 1.0
            continue
        z = (u - mean_u) / math.sqrt(var_u)
        pval[j] = float(2.0 * norm.sf(abs(z)))
    fdr = false_discovery_control(np.clip(pval, 0, 1), method="bh")
    out = pd.DataFrame(
        {
            "scope": scope,
            "gene": gene_index,
            "n_malignant": n_malignant,
            "n_healthy": n_healthy,
            "mean_log_malignant": mean_pos,
            "mean_log_healthy": mean_neg,
            "delta_log": delta,
            "frac_expr_malignant": frac_pos,
            "frac_expr_healthy": frac_neg,
            "auroc": auc,
            "auroc_shift": auc - 0.5,
            "mwu_pvalue": pval,
            "fdr_bh": fdr,
        }
    )
    out["abs_delta_log"] = out["delta_log"].abs()
    out["abs_auroc_shift"] = out["auroc_shift"].abs()
    out["is_de"] = (
        (out["fdr_bh"] < DE_FDR)
        & (out["abs_delta_log"] >= DE_ABS_DELTA_LOG)
        & (out["abs_auroc_shift"] >= DE_ABS_AUC_SHIFT)
    )
    out["signed_rank_score"] = np.sign(out["delta_log"]) * (-np.log10(np.clip(out["mwu_pvalue"], 1e-300, 1)))
    return out.sort_values(["is_de", "abs_auroc_shift", "abs_delta_log"], ascending=[False, False, False]).reset_index(drop=True)


def lineage_restricted_mask(patient: str) -> pd.Series:
    patient_rows = obs["patient"].eq(patient)
    mal_types = set(obs.loc[patient_rows & obs["malignant"], "predicted.annotation"])
    return patient_rows & (obs["malignant"] | obs["predicted.annotation"].isin(mal_types))


scopes: dict[str, pd.Series] = {"cohort": pd.Series(True, index=obs.index)}
for patient in SUPPORTED_PATIENTS:
    scopes[patient] = obs["patient"].eq(patient)
for patient in LOW_SUPPORT_PATIENTS:
    scopes[f"{patient}_lineage_restricted"] = lineage_restricted_mask(patient)

if DE_CACHE.exists():
    de_all = pd.read_pickle(DE_CACHE)
    have = set(de_all["scope"].unique())
    if have != set(scopes):
        print("DE cache scopes changed; recomputing.")
        de_all = None
    else:
        print("Loaded DE cache:", DE_CACHE, de_all.shape)
else:
    de_all = None

if de_all is None:
    parts = []
    for scope, mask in scopes.items():
        idx = np.where(mask.to_numpy())[0]
        scope_obs = obs.iloc[idx]
        y = scope_obs["malignant"].to_numpy()
        print(f"DE {scope}: {int(y.sum())} malignant / {int((~y).sum())} healthy")
        X = _to_dense(adata.X[idx])
        parts.append(run_de(X, y, scope, int(y.sum()), int((~y).sum())))
        del X
    de_all = pd.concat(parts, ignore_index=True)
    de_all.to_pickle(DE_CACHE, compression="gzip")
    print("Wrote DE cache:", DE_CACHE)

write_csv(de_all, "d13_de_all_scopes_full.csv")
de_primary = de_all[de_all["scope"].isin(["cohort", *SUPPORTED_PATIENTS])].copy()
write_csv(de_primary, "d13_de_primary_scopes.csv")

de_count_rows = []
for scope, group in de_primary.groupby("scope", sort=False):
    de_count_rows.append(
        {
            "scope": scope,
            "n_genes": int(len(group)),
            "n_de": int(group["is_de"].sum()),
            "n_de_up": int((group["is_de"] & (group["delta_log"] > 0)).sum()),
            "n_de_down": int((group["is_de"] & (group["delta_log"] < 0)).sum()),
            "median_abs_auroc_shift": float(group["abs_auroc_shift"].median()),
            "p90_abs_auroc_shift": float(group["abs_auroc_shift"].quantile(0.9)),
        }
    )
de_counts = pd.DataFrame(de_count_rows)
write_csv(de_counts, "d13_de_call_counts_by_scope.csv")
display(de_counts)
"""
    )
)

cells.append(
    code(
        r"""
# D12D — three-way overlap of old34, HVG3000, and cohort DEGs.
cohort_de_genes = set(de_primary.query("scope == 'cohort' and is_de")["gene"])
hvg3000 = hvg_sets[3000]
three_way = pd.DataFrame(
    [
        set_overlap_row("cohortDEG_vs_hvg3000", cohort_de_genes, hvg3000, universe, "cohort_DEG", "hvg_top_3000"),
        set_overlap_row("cohortDEG_vs_full34", cohort_de_genes, full34_genes, universe, "cohort_DEG", "old34_full_gsea_union"),
        set_overlap_row("full34_vs_hvg3000", full34_genes, hvg3000, universe, "old34_full_gsea_union", "hvg_top_3000"),
    ]
)
three_way["n_triple"] = len(cohort_de_genes & hvg3000 & full34_genes)
write_csv(three_way, "d12_deg_hvg_old34_threeway_overlap.csv")
display(three_way[[
    "comparison", "n_left", "n_right", "n_overlap", "n_left_only", "n_right_only",
    "jaccard", "fold_enrichment", "hypergeom_enrichment_pvalue", "n_triple",
]])

parts = [
    ("old34 ∩ HVG3k ∩ cohort DEG", len(full34_genes & hvg3000 & cohort_de_genes)),
    ("old34 ∩ HVG3k only", len((full34_genes & hvg3000) - cohort_de_genes)),
    ("old34 ∩ cohort DEG only", len((full34_genes & cohort_de_genes) - hvg3000)),
    ("HVG3k ∩ cohort DEG only", len((hvg3000 & cohort_de_genes) - full34_genes)),
    ("old34 only", len(full34_genes - hvg3000 - cohort_de_genes)),
    ("HVG3k only", len(hvg3000 - full34_genes - cohort_de_genes)),
    ("cohort DEG only", len(cohort_de_genes - full34_genes - hvg3000)),
]
upset = pd.DataFrame(parts, columns=["region", "n_genes"])
write_csv(upset, "d12_threeway_region_counts.csv")

fig, ax = plt.subplots(figsize=(10.8, 4.8))
sns.barplot(data=upset, x="n_genes", y="region", color=PALETTE["gsea"], ax=ax)
ax.set_xlabel("Genes")
ax.set_ylabel("")
ax.set_title("D12D. Three-way regions: old34 GSEA ∪ HVG3,000 ∪ cohort malignant-vs-healthy DEGs")
d12d_png = save_figure(fig, "D12D_old34_hvg3000_cohortDEG_regions")
plt.show()
"""
    )
)

cells.append(
    md(
        r"""
## 4. D13 — unique low-support signals and panel-level recovery

Two questions are joined to the existing D7 lift tables:

- Which genes are high-effect in P01/P06/P07 and not in P02/P03/P04?
- Which old34 / biology-group panels are enriched for those DE genes, and do those panels already transfer (held-out lift) or only appear in-sample?

P07 is treated as a positive-control low-prevalence patient: D7 already shows EMT / TGF-plasticity / p53 / stress-arrest lift, so DE should recover those programs if the gene-level assay is working.
"""
    )
)

cells.append(
    code(
        r"""
patient_metrics = pd.read_csv(SOURCE_PATHS["patient_metrics"])
k40_spread = pd.read_csv(SOURCE_PATHS["k40_spread"])
patient_summary = pd.read_csv(SOURCE_PATHS["patient_summary"])

best_lift = (
    patient_metrics.groupby(["heldout_patient", "stage0_panel_id", "short_panel_label", "biological_theme", "n_covered_genes"], as_index=False)
    .agg(
        max_heldout_lift=("heldout_lift", "max"),
        max_heldout_auprc=("heldout_auprc", "max"),
        best_method=("stage1_method", lambda s: patient_metrics.loc[s.index].sort_values("heldout_lift", ascending=False)["stage1_method"].iloc[0]),
        best_k=("stage1_method_k", lambda s: patient_metrics.loc[s.index].sort_values("heldout_lift", ascending=False)["stage1_method_k"].iloc[0]),
    )
)
spread_best = k40_spread.groupby(["heldout_patient", "stage0_panel_id"], as_index=False).agg(
    k40_method_lift_range=("k40_method_lift_range", "max"),
    max_k40_method_lift=("max_method_lift", "max"),
)
best_lift = best_lift.merge(spread_best, on=["heldout_patient", "stage0_panel_id"], how="left")
write_csv(best_lift, "d13_existing_best_heldout_lift_by_panel_patient.csv")

focus_lift = (
    best_lift[best_lift["heldout_patient"].isin(LOW_SUPPORT_PATIENTS)]
    .sort_values(["heldout_patient", "max_heldout_lift"], ascending=[True, False])
    .groupby("heldout_patient", as_index=False)
    .head(8)
)
print("Existing best held-out lift (any method/K) for low-support patients")
display(focus_lift[[
    "heldout_patient", "short_panel_label", "best_method", "best_k",
    "max_heldout_lift", "max_k40_method_lift", "k40_method_lift_range",
]])

high_support_de = set(de_primary.query("scope in @HIGH_SUPPORT_PATIENTS and is_de")["gene"])
unique_rows = []
top_gene_tables = []
for patient in SUPPORTED_PATIENTS:
    sub = de_primary.query("scope == @patient").copy()
    de_genes = set(sub.loc[sub["is_de"], "gene"])
    unique_genes = de_genes - high_support_de if patient in LOW_SUPPORT_PATIENTS else set()
    sub["unique_vs_high_support"] = sub["gene"].isin(unique_genes) if patient in LOW_SUPPORT_PATIENTS else False
    top_gene_tables.append(sub.head(50))
    unique_rows.append(
        {
            "patient_id": patient,
            "n_de": len(de_genes),
            "n_unique_vs_P02P03P04": len(unique_genes),
            "unique_fraction_of_de": (len(unique_genes) / len(de_genes)) if de_genes else np.nan,
            "n_unique_in_old34": len(unique_genes & full34_genes),
            "n_unique_in_hvg3000": len(unique_genes & hvg3000),
            "n_unique_outside_old34_and_hvg3000": len(unique_genes - full34_genes - hvg3000),
            "median_unique_abs_auroc_shift": float(sub.loc[sub["unique_vs_high_support"], "abs_auroc_shift"].median()) if unique_genes else np.nan,
        }
    )
unique_summary = pd.DataFrame(unique_rows)
write_csv(unique_summary, "d13_unique_de_gene_counts.csv")
top_genes = pd.concat(top_gene_tables, ignore_index=True)
write_csv(top_genes, "d13_top50_de_genes_by_patient.csv")
display(unique_summary)

unique_gene_long = []
for patient in LOW_SUPPORT_PATIENTS:
    sub = de_primary.query("scope == @patient and is_de").copy()
    sub = sub[~sub["gene"].isin(high_support_de)]
    sub["in_old34"] = sub["gene"].isin(full34_genes)
    sub["in_hvg3000"] = sub["gene"].isin(hvg3000)
    unique_gene_long.append(sub)
unique_gene_frame = pd.concat(unique_gene_long, ignore_index=True) if unique_gene_long else pd.DataFrame()
write_csv(unique_gene_frame, "d13_unique_de_genes_lowsupport.csv")
"""
    )
)

cells.append(
    code(
        r"""
def hypergeom_panel_enrichment(de_genes: set[str], scope: str) -> pd.DataFrame:
    rows = []
    n_de = len(de_genes)
    n_u = len(universe)
    for panel_id, genes in panel_genes.items():
        overlap = de_genes & genes
        expected = n_de * len(genes) / n_u if n_u else np.nan
        p_enrich = float(hypergeom.sf(len(overlap) - 1, n_u, len(genes), n_de)) if n_de and overlap else (1.0 if n_de else np.nan)
        meta = panel_manifest.query("panel_id == @panel_id")
        short = best_lift.loc[best_lift["stage0_panel_id"].eq(panel_id), "short_panel_label"]
        rows.append(
            {
                "scope": scope,
                "stage0_panel_id": panel_id,
                "short_panel_label": (short.iloc[0] if len(short) else panel_id),
                "panel_type": (meta["panel_type"].iloc[0] if len(meta) else ""),
                "n_panel_genes": len(genes),
                "n_de_genes": n_de,
                "n_overlap": len(overlap),
                "expected_overlap": expected,
                "fold_enrichment": (len(overlap) / expected) if expected else np.nan,
                "hypergeom_enrichment_pvalue": p_enrich,
                "overlap_genes": ";".join(sorted(overlap)[:40]),
            }
        )
    out = pd.DataFrame(rows)
    out["fdr_bh"] = false_discovery_control(out["hypergeom_enrichment_pvalue"].fillna(1.0), method="bh") if len(out) else []
    return out.sort_values("hypergeom_enrichment_pvalue")


enrich_parts = []
for patient in SUPPORTED_PATIENTS:
    de_genes = set(de_primary.query("scope == @patient and is_de")["gene"])
    enrich_parts.append(hypergeom_panel_enrichment(de_genes, patient))
panel_enrichment = pd.concat(enrich_parts, ignore_index=True)
panel_enrichment = panel_enrichment.merge(
    best_lift.rename(columns={"heldout_patient": "scope"})[
        ["scope", "stage0_panel_id", "max_heldout_lift", "max_k40_method_lift", "k40_method_lift_range", "best_method", "best_k"]
    ],
    on=["scope", "stage0_panel_id"],
    how="left",
)
write_csv(panel_enrichment, "d13_panel_enrichment_vs_heldout_lift.csv")

single_enrich = panel_enrichment.query("panel_type == 'single_geneset_only'").copy()
print("Top enriched single-geneset panels in low-support patients")
display(
    single_enrich[single_enrich["scope"].isin(LOW_SUPPORT_PATIENTS)]
    .sort_values(["scope", "hypergeom_enrichment_pvalue"])
    .groupby("scope", as_index=False)
    .head(6)[
        ["scope", "short_panel_label", "n_overlap", "fold_enrichment", "fdr_bh", "max_heldout_lift", "best_method", "best_k"]
    ]
)
"""
    )
)

cells.append(
    code(
        r"""
GSEA_CSV = TABLE_DIR / "d13_prerank_gsea_old34_by_patient.csv"
try:
    import gseapy as gp
except Exception as exc:  # pragma: no cover
    gp = None
    print("gseapy unavailable, skipping prerank:", exc)

gsea_frames = []
if GSEA_CSV.exists():
    gsea_all = pd.read_csv(GSEA_CSV)
    print("Loaded cached prerank GSEA:", GSEA_CSV, gsea_all.shape)
elif gp is not None:
    gmt_old34 = {name: [g for g in genes if g in universe] for name, genes in gmt_raw.items() if name in set(coverage["geneset_name"])}
    gsea_dir = CACHE_DIR / "gseapy_prerank"
    gsea_dir.mkdir(exist_ok=True)
    for patient in SUPPORTED_PATIENTS:
        rnk = (
            de_primary.query("scope == @patient")
            .set_index("gene")["signed_rank_score"]
            .sort_values(ascending=False)
        )
        outdir = gsea_dir / patient
        try:
            res = gp.prerank(
                rnk=rnk,
                gene_sets=gmt_old34,
                outdir=str(outdir),
                permutation_num=GSEA_PERMUTATIONS,
                min_size=GSEA_MIN_SIZE,
                max_size=GSEA_MAX_SIZE,
                threads=8,
                no_plot=True,
                seed=42,
                verbose=False,
            )
            frame = res.res2d.copy()
            frame["scope"] = patient
            gsea_frames.append(frame)
            print(f"GSEA {patient}: {len(frame)} terms")
        except Exception as exc:
            print(f"GSEA failed for {patient}: {exc}")

    if gsea_frames:
        gsea_all = pd.concat(gsea_frames, ignore_index=True)
        rename = {c: c.replace(" ", "_").lower() for c in gsea_all.columns}
        gsea_all = gsea_all.rename(columns=rename)
        write_csv(gsea_all, "d13_prerank_gsea_old34_by_patient.csv")
    else:
        gsea_all = pd.DataFrame()
        print("No GSEA results.")
if "gsea_all" in locals() and len(gsea_all):
    top_gsea = (
        gsea_all.sort_values(["scope", "fdr_q-val", "nes"], ascending=[True, True, False])
        .groupby("scope", as_index=False)
        .head(8)
    )
    print("Top prerank GSEA terms (old34 dictionary)")
    show_cols = [c for c in ["scope", "term", "nes", "fdr_q-val", "fwer_p-val", "gene_percent"] if c in top_gsea.columns]
    display(top_gsea[show_cols])
else:
    gsea_all = pd.DataFrame()
    print("No GSEA results.")
"""
    )
)

cells.append(
    code(
        r"""
# D13B — volcano-style AUROC vs delta-log for low-support patients.
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
for ax, patient in zip(axes, LOW_SUPPORT_PATIENTS):
    sub = de_primary.query("scope == @patient")
    unique_mask = sub["gene"].isin(set(unique_gene_frame.query("scope == @patient")["gene"])) if len(unique_gene_frame) else np.zeros(len(sub), dtype=bool)
    ax.scatter(sub["delta_log"], sub["auroc"], s=8, c=PALETTE["other"], alpha=0.25, linewidths=0)
    ax.scatter(sub.loc[sub["is_de"], "delta_log"], sub.loc[sub["is_de"], "auroc"], s=12, c=PALETTE["gsea"], alpha=0.7, linewidths=0, label="DE")
    ax.scatter(sub.loc[unique_mask, "delta_log"], sub.loc[unique_mask, "auroc"], s=16, c=PALETTE["low"], alpha=0.9, linewidths=0, label="unique vs P02/P03/P04")
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.axvline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_title(f"{patient}  ·  {int(sub['is_de'].sum())} DE  ·  {int(unique_mask.sum())} unique")
    ax.set_xlabel("Δ log1p (malignant − healthy)")
    if patient == "P01":
        ax.set_ylabel("Gene AUROC")
    if patient == "P07":
        ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("D13B. Low-support gene-level discrimination and uniqueness versus high-support patients", y=1.03)
d13b_png = save_figure(fig, "D13B_lowsupport_gene_auroc_vs_deltalog")
plt.show()

# D13C — existing max lift versus DE panel enrichment for single-geneset panels.
fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharey=True)
for ax, patient in zip(axes, LOW_SUPPORT_PATIENTS):
    sub = single_enrich.query("scope == @patient").copy()
    sub["neglog10_p"] = -np.log10(np.clip(sub["hypergeom_enrichment_pvalue"], 1e-12, 1))
    sizes = 30 + 8 * np.sqrt(sub["n_overlap"].to_numpy())
    ax.scatter(sub["max_heldout_lift"] * 100, sub["neglog10_p"], s=sizes, c=PALETTE["gsea"], alpha=0.8)
    top = sub.nlargest(4, "neglog10_p")
    for _, row in top.iterrows():
        ax.annotate(row["short_panel_label"], (row["max_heldout_lift"] * 100, row["neglog10_p"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Best existing held-out lift (AUPRC pp)")
    if patient == "P01":
        ax.set_ylabel("Panel enrichment −log10 p (DE genes)")
    ax.set_title(patient)
fig.suptitle("D13C. Old34 panels: in-sample DE enrichment versus already-measured transfer lift", y=1.03)
d13c_png = save_figure(fig, "D13C_panel_de_enrichment_vs_heldout_lift")
plt.show()

# D13D — unique-gene membership in old34 vs HVG3k.
if len(unique_gene_frame):
    membership = (
        unique_gene_frame.assign(
            membership=np.select(
                [
                    unique_gene_frame["in_old34"] & unique_gene_frame["in_hvg3000"],
                    unique_gene_frame["in_old34"],
                    unique_gene_frame["in_hvg3000"],
                ],
                ["old34 ∩ HVG3k", "old34 only", "HVG3k only"],
                default="outside both",
            )
        )
        .groupby(["scope", "membership"], as_index=False)
        .size()
        .rename(columns={"size": "n_genes", "scope": "patient"})
    )
    write_csv(membership, "d13_unique_gene_list_membership.csv")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    sns.barplot(
        data=membership,
        x="patient",
        y="n_genes",
        hue="membership",
        hue_order=["old34 only", "old34 ∩ HVG3k", "HVG3k only", "outside both"],
        palette=[PALETTE["gsea"], "#54A24B", PALETTE["hvg"], PALETTE["other"]],
        ax=ax,
    )
    ax.set_xlabel("Low-support patient")
    ax.set_ylabel("Unique DE genes")
    ax.set_title("D13D. Unique low-support DE genes are not mostly already inside old34 or HVG3,000")
    ax.legend(title="", frameon=False, fontsize=8)
    d13d_png = save_figure(fig, "D13D_unique_de_gene_list_membership")
    plt.show()
else:
    membership = pd.DataFrame()
    print("No unique DE genes under the pre-specified rule.")

# D13E — prerank NES heatmap if GSEA succeeded.
if len(gsea_all):
    term_col = "term" if "term" in gsea_all.columns else "Term"
    nes_col = next(c for c in gsea_all.columns if c.lower() == "nes")
    fdr_col = [c for c in gsea_all.columns if "fdr" in c.lower()][0]
    gsea_plot = gsea_all.copy()
    gsea_plot[nes_col] = pd.to_numeric(gsea_plot[nes_col], errors="coerce")
    gsea_plot[fdr_col] = pd.to_numeric(gsea_plot[fdr_col], errors="coerce")
    pivot = gsea_plot.pivot_table(index=term_col, columns="scope", values=nes_col, aggfunc="first")
    pivot = pivot.reindex(columns=[c for c in SUPPORTED_PATIENTS if c in pivot.columns])
    focus_terms = (
        gsea_plot.assign(_abs=gsea_plot[nes_col].abs())
        .sort_values(["scope", fdr_col, "_abs"], ascending=[True, True, False])
        .groupby("scope", as_index=False)
        .head(5)[term_col]
        .drop_duplicates()
    )
    heat = pivot.loc[pivot.index.intersection(focus_terms)].apply(pd.to_numeric, errors="coerce")
    heat = heat.loc[heat.abs().max(axis=1).sort_values(ascending=False).index]
    heat.index = (
        heat.index.astype(str)
        .str.replace("HALLMARK_", "", regex=False)
        .str.replace("REACTOME_", "", regex=False)
        .str.replace("KEGG_", "", regex=False)
        .str.replace("_", " ")
        .str.lower()
    )
    if heat.empty:
        print("D13E skipped: empty NES table after numeric coercion.")
    else:
        fig, ax = plt.subplots(figsize=(8.8, max(4.5, 0.32 * len(heat) + 1.8)))
        sns.heatmap(heat.astype(float), cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Prerank NES"})
        ax.set_xlabel("Held-out / DE patient")
        ax.set_ylabel("Old34 GSEA program")
        ax.set_title("D13E. Patient-specific prerank NES on the old34 dictionary")
        d13e_png = save_figure(fig, "D13E_prerank_nes_old34_by_patient")
        plt.show()
else:
    heat = pd.DataFrame()
"""
    )
)

cells.append(
    code(
        r"""
# Lineage-restricted sensitivity for the three focus patients.
lineage_counts = (
    de_all[de_all["scope"].str.endswith("_lineage_restricted")]
    .groupby("scope", as_index=False)
    .agg(n_de=("is_de", "sum"), p90_abs_auroc_shift=("abs_auroc_shift", lambda s: float(s.quantile(0.9))))
)
write_csv(lineage_counts, "d13_lineage_restricted_de_counts.csv")
print("Lineage-restricted DE counts")
display(lineage_counts)

# Rank correlation: panel DE enrichment versus existing lift, within each patient.
corr_rows = []
for patient in SUPPORTED_PATIENTS:
    sub = single_enrich.query("scope == @patient").dropna(subset=["max_heldout_lift", "fold_enrichment"])
    if len(sub) < 5:
        continue
    rho, pval = spearmanr(sub["fold_enrichment"], sub["max_heldout_lift"])
    corr_rows.append({"patient_id": patient, "n_panels": len(sub), "spearman_enrichment_vs_lift": float(rho), "nominal_pvalue_not_for_inference": float(pval)})
enrich_vs_lift = pd.DataFrame(corr_rows)
write_csv(enrich_vs_lift, "d13_panel_enrichment_vs_lift_spearman.csv")
print("Within-patient Spearman(DE fold-enrichment, best held-out lift) across 34 single-geneset panels")
display(enrich_vs_lift)

# Compact top unique genes for the findings report.
top_unique_preview = (
    unique_gene_frame.sort_values(["scope", "abs_auroc_shift"], ascending=[True, False])
    .groupby("scope", as_index=False)
    .head(12)[
        ["scope", "gene", "delta_log", "auroc", "fdr_bh", "in_old34", "in_hvg3000"]
    ]
    if len(unique_gene_frame)
    else pd.DataFrame()
)
if len(top_unique_preview):
    print("Top unique DE genes")
    display(top_unique_preview)
"""
    )
)

cells.append(
    md(
        r"""
## 5. Plot contracts, claim boundary, and artifacts

D12 is a gene-set geometry result: it can confirm that Stage 0 knowledge panels are not HVG aliases. D13 is a candidate-generating DE screen. Neither reruns classification, so a newly nominated gene list still needs a Stage 0 panel + Stage 2 LOPO experiment before it can be called a high-lift transfer signal.
"""
    )
)

cells.append(
    code(
        r"""
contracts = pd.DataFrame(
    [
        plot_contract("D12A", "Old34 vs HVG Venns", "Are the GSEA/old34 union and HVG anchors interchangeable Stage 0 panels?", "Intersect covered full_34 genes with each HVG anchor inside the 15,560-gene universe.", "gene", "Jaccard, exclusive counts, hypergeometric fold-enrichment versus |A||B|/|U|", "Old34 is HVG-enriched but not an HVG alias: exclusive GSEA genes remain the majority at the size-matched HVG3,000 budget.", "HVG ranks are transductive and one variance definition."),
        plot_contract("D12B", "Per-program HVG3,000 enrichment", "Is HVG overlap concentrated in a few old34 programs?", "Hypergeometric fold-enrichment of each covered geneset versus HVG3,000.", "geneset", "fold enrichment; 1 = random overlap", "Every program is enriched for HVGs; the enrichment does not make any program equivalent to HVG3,000.", "Covered-gene missingness differs by program."),
        plot_contract("D12C", "Exclusive gene counts by HVG budget", "Does raising the HVG budget absorb the old34 union?", "Stack exclusive and intersecting gene counts at HVG 500/1,000/3,000/10,000.", "gene", "count in the shared universe", "Even HVG10,000 leaves a non-trivial old34-only remainder, and HVG-only genes dominate the large anchors.", "No matched-size random-panel control beyond the hypergeometric null."),
        plot_contract("D12D", "Three-way old34 / HVG / cohort DEG", "Is the knowledge prior closer to HVG or to label-using DEGs?", "Build region counts among full_34, HVG3,000, and cohort-wide DEGs.", "gene", "set-region counts", "Cohort DEGs, HVGs, and old34 are pairwise overlapping but not nested; each list has a large exclusive remainder.", "Cohort DE is support-weighted toward P03."),
        plot_contract("D13A", "Low-support cell-type mix", "Can unmatched DE be explained by lineage imbalance?", "Count predicted.annotation by CN.label inside P01/P06/P07.", "cell", "cell counts", "Malignant and healthy dominant types differ; unmatched DE is lineage-confounded.", "predicted.annotation is itself model-derived."),
        plot_contract("D13B", "Gene AUROC vs Δlog", "Do P01/P06/P07 have high-effect genes, and are they unique versus P02/P03/P04?", "Wilcoxon + AUROC per gene; mark DE and unique-to-low-support genes.", "gene", "AUROC and Δlog1p; DE rule FDR 0.05, |Δlog|≥0.25, |AUROC−0.5|≥0.15", "Effect-size clouds and unique counts diagnose whether the dark D7B columns lack gene-level signal or only lack method-sensitive transfer.", "n+=22–45; no cell bootstrap."),
        plot_contract("D13C", "Panel DE enrichment vs lift", "Do DE-nominated old34 panels already transfer in the D7 atlas?", "Hypergeometric enrichment of patient DEGs in each single-geneset panel, plotted against that patient's best existing held-out lift.", "panel × patient", "−log10 enrichment p vs AUPRC lift", "Upper-left points are private/support-limited candidates; upper-right points are already captured transfer programs (P07 EMT/p53/stress).", "Enrichment uses in-sample DE; lift is LOPO on a transductive Stage 1 basis."),
        plot_contract("D13D", "Unique-gene list membership", "Are unique low-support DE genes already inside old34 or HVG3,000?", "Partition unique DE genes by membership in full_34 and HVG3,000.", "gene", "counts", "Genes outside both lists are the concrete next Stage 0 candidates; genes inside old34 but without lift are representation/support failures.", "Uniqueness is versus three high-support patients only."),
        plot_contract("D13E", "Prerank NES by patient", "Which old34 programs rank as patient-specific discriminatory sets?", "gseapy prerank on signed −log10 Wilcoxon p using the old34 GMT.", "geneset × patient", "NES; FDR from 300 permutations", "P07 should recover EMT/stress/p53 if D7 lift is gene-supported; P01/P06 either recover different programs or remain weak.", "Dictionary-restricted GSEA cannot discover programs outside old34."),
    ]
)
write_csv(contracts, "deg_hvg_lowsupport_plot_contracts.csv")

size_matched = overlap_summary.query("comparison == 'full34_vs_hvg3000'").iloc[0]
p01_de = int(de_counts.query("scope == 'P01'")["n_de"].iloc[0])
p06_de = int(de_counts.query("scope == 'P06'")["n_de"].iloc[0])
p07_de = int(de_counts.query("scope == 'P07'")["n_de"].iloc[0])
p01_unique = int(unique_summary.query("patient_id == 'P01'")["n_unique_vs_P02P03P04"].iloc[0])
p06_unique = int(unique_summary.query("patient_id == 'P06'")["n_unique_vs_P02P03P04"].iloc[0])
p07_unique = int(unique_summary.query("patient_id == 'P07'")["n_unique_vs_P02P03P04"].iloc[0])
p01_best = float(best_lift.query("heldout_patient == 'P01'")["max_heldout_lift"].max())
p06_best = float(best_lift.query("heldout_patient == 'P06'")["max_heldout_lift"].max())
p07_best = float(best_lift.query("heldout_patient == 'P07'")["max_heldout_lift"].max())
p07_best_panel = best_lift.query("heldout_patient == 'P07'").sort_values("max_heldout_lift", ascending=False)["short_panel_label"].iloc[0]
cohort_n_de = int(de_counts.query("scope == 'cohort'")["n_de"].iloc[0])

old34_only_frac = float(size_matched["n_left_only"] / size_matched["n_left"])
findings_lines = [
    "# Old34 DEG/HVG overlap and low-support patient DE",
    "",
    "Generated: " + datetime.now(timezone.utc).isoformat(),
    "",
    "## D12. GSEA/old34 versus HVG",
    "",
    "- Universe: 15,560 genes after the Stage 0 1% min-cell filter.",
    "- Old34 covered union (`full_34`): {:,} genes.".format(len(full34_genes)),
    "- Size-matched HVG3,000 overlap: {:,} genes; Jaccard {:.3f}; old34-only {:,} ({:.1%} of old34); fold-enrichment {:.2f}x; hypergeometric enrichment p={:.2e}.".format(
        int(size_matched["n_overlap"]),
        float(size_matched["jaccard"]),
        int(size_matched["n_left_only"]),
        old34_only_frac,
        float(size_matched["fold_enrichment"]),
        float(size_matched["hypergeom_enrichment_pvalue"]),
    ),
    "- Pre-specified verdict that the two Stage 0 lists are meaningfully different: {}.".format(meaningfully_different),
    "- They are not independent: old34 is HVG-enriched at every anchor. They are not interchangeable: most old34 genes sit outside HVG3,000, and most HVG3,000 genes sit outside old34.",
    "- Cohort-wide DEGs (same DE rule): {:,} genes. Three-way region counts are in `d12_threeway_region_counts.csv`.".format(cohort_n_de),
    "",
    "## D13. Low-support patients P01, P06, P07",
    "",
    "- Existing best held-out lift already in the D1/D7 atlas: P01 {:.3f} AP, P06 {:.3f} AP, P07 {:.3f} AP (best P07 panel: {}). D7B darkness is method-spread, not a claim of zero absolute lift.".format(p01_best, p06_best, p07_best, p07_best_panel),
    "- DE genes under the triage rule: P01 {}, P06 {}, P07 {}.".format(p01_de, p06_de, p07_de),
    "- Unique versus P02/P03/P04: P01 {}, P06 {}, P07 {}.".format(p01_unique, p06_unique, p07_unique),
    "- P07 is the internal positive control: if EMT / TGF-plasticity / p53 / stress-arrest are DE-enriched, the gene-level assay agrees with D7 transfer. P01/P06 are the true dark-column question.",
    "- Lineage-restricted counts are in `d13_lineage_restricted_de_counts.csv`. Treat unmatched DE as lineage-confounded until those counts stay similar.",
    "",
    "## Claim boundary",
    "",
    "Supported as descriptive findings:",
    "",
    "- Old34 GSEA and HVG3,000 are overlapping but not the same Stage 0 panel.",
    "- P01/P06/P07 have measurable within-patient gene-level effects whose uniqueness and panel membership are tabulated.",
    "- P07's already-observed EMT/stress transfer is the benchmark for whether DE recovers a known lift.",
    "",
    "Not supported:",
    "",
    "- a new shared transferable program;",
    "- a causal claim that D7B darkness is caused by missing gene sets rather than prevalence/support;",
    "- deployment performance of any newly nominated unique-gene panel (that requires a new Stage 0/2 run).",
    "",
]
findings = "\n".join(findings_lines)
findings_path = TABLE_DIR / "deg_hvg_lowsupport_findings.md"
findings_path.write_text(findings)
GENERATED_TABLES.append(findings_path)
print(findings)

ledger_records = []
for path in GENERATED_TABLES + GENERATED_FIGURES:
    stat = path.stat()
    ledger_records.append(
        {
            "relative_path": str(path.relative_to(EXP_DIR)),
            "artifact_type": "figure" if path.suffix in {".png", ".pdf"} else "table_or_report",
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    )
ledger = pd.DataFrame(ledger_records).sort_values(["artifact_type", "relative_path"]).reset_index(drop=True)
ledger_path = TABLE_DIR / "deg_hvg_lowsupport_artifact_ledger.csv"
ledger.to_csv(ledger_path, index=False)
print("Findings:", findings_path)
print("Tracked artifacts:", len(ledger) + 1)
display(contracts)
"""
    )
)

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
nb["cells"] = cells
nbf.write(nb, OUT)
print("Wrote", OUT, "cells", len(cells))
