#!/usr/bin/env python
"""D30 (W5.11): the feature-space explainer figure, "The same cells can be represented in different feature
spaces" (asks A-20261011-14, -15, -16 in research/asks/2026-10-11_feature-space-figure.md).

The same cells are drawn as rows of the Stage 1 input X~ (the 92 Hallmark IFN-alpha genes, log1p, z-scored per
gene over all 60,118 cells) and of the stored Stage 1 scores Z (IFN-alpha PCA K = 40, seed 42: one basis fit on
the cells of all 13 patients without labels, so shared across patients and transductive). Each cell keeps its
patient, label (CN.label), predicted type and origin (source). Two versions:
  core      X~ and Z with panels A-D (one cell and one gene in gene space; the same cell and one factor in
            factor space). The candidate for paper Fig1 panel A.
  extended  adds the stored FA K = 40 scores (Z'), nine surface proteins (per-cell CLR, as D27) and one
            recipient-origin healthy cell per patient where one exists, with panels E (protein) and F (FA).
            For review.

Rows, fixed before drawing (seed 42): for P01, P02, P03, P04 and P09, the D18 comparison groups (patient x
predicted type with >= 10 cells of each label), at most 2 per patient with the most malignant cells, 2 malignant
and 2 healthy cells from each, then 1 healthy cell of the patient's most abundant healthy type not already
shown. The highlighted cell is example cell 4 of full report Figure 20B (P03, malignant, HSC/MPP2). Labels only
stratify and order the rows; no gene, factor or protein is chosen with held-out outcomes (PC8, and with it IFI27
and the poles, was chosen and oriented by the supervised probe, which the figure says). Genes: all 92, in a
label-free order (average-linkage clustering of gene-gene correlation over all cells). Factors: all 40.

Reads (EXP-relative, read only): the cohort cache (study_config cohort_cache_relpath: X, obs, obsm['ADT']);
stage2_supervised/multiobjective/discovery_full_cohort/cell_metadata.csv.gz (row order);
preprocessing/panels/stage0_panel_genes/<panel>.json; stage1_dr/panel_id=<panel>/method={pca,fa}/k=40/seed=42/
{scores.npy, loadings.npy, diagnostics.json, metadata.json}; analysis/scorecards/stage2_diagnostic/
factor_cards_v1/ (D18) and factor_modality_assoc_v1/ (D27, through make_figures_v3's loaders, which assert
against it). The raw AnnData is never loaded (AGENTS R8). Nothing is refit; no Stage 2 (c_policy:
not_applicable).

Writes (new slug feature_space_explainer_v1; with --out-root DIR everything goes under DIR, EXP is only read):
  analysis/figures/stage2_diagnostic/<slug>/feature_space_explainer_{core,extended}.{pdf,png}
  analysis/scorecards/stage2_diagnostic/<slug>/selected_cells.csv, displayed_features.csv, panel_values.csv,
    regression_checks.csv, <slug>_{findings.md,manifest.json,artifact_ledger.csv}

Usage (lab env; about half a minute on CPU, BLAS threads capped at 4):
  python scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_feature_space_explainer.py
  ... --version core|extended|both   (default both)
  ... --resume                       skip if every output exists and reads back; refuse otherwise (R3)
  ... --out-root <scratch>/fsx       write under a scratch root instead of EXP
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[4]  # workstreams/<W#_slug>/<file>.py -> repo root
sys.path.insert(0, str(REPO / "src"))
from sc_classification.utils.paths import experiment_dir, git_state, study_config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SLUG = "feature_space_explainer_v1"
ANALYSIS_ID = "D30"
SCRIPT_REL = "scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_feature_space_explainer.py"
PANEL = "single_geneset__hallmark_interferon_alpha_response"
REPORT_DIR = REPO / "scripts" / "mrd_stage0_2" / "research" / "reports" / "2026-10-08_questions_report"
PAVAN_DIR = REPO / "scripts" / "mrd_stage0_2" / "research" / "team_updates" / "2026-10-09_protein_association_for_pavan"
W5_DIR = Path(__file__).resolve().parent

SEED = 42
PATIENTS = ("P01", "P02", "P03", "P04", "P09")
HIGH_PREVALENCE = ("P02", "P03", "P04")
MIN_EACH_LABEL = 10  # D18 comparison groups: >= 10 cells of each label inside one patient x predicted type
TYPES_PER_PATIENT = 2
CELLS_PER_LABEL = 2  # per label inside each shown type
N_COMPOSITION = 1  # healthy cells of the patient's most abundant healthy type not already shown
HIGHLIGHT_EXAMPLE_CELL = 4  # full report Figure 20B worked example (P03, malignant, HSC/MPP2)
HIGHLIGHT_GENE = "IFI27"  # PC8's largest |loading| (core report page 7)
LABELLED_GENES = ("IFI27", "PSME2", "PSMA3", "TXNIP", "IFITM2", "TENT5A")  # 3 per pole (Figure 5a)
PC8_COL, FA7_COL = 7, 6  # 0-based columns of scores.npy
PC8_CARD = f"{PANEL}__pca__k40__f008"
FA7_CARD = f"{PANEL}__fa__k40__f007"
PROTEINS = ("CD34", "CD99", "CD48", "CD116", "CD33", "CD146", "CD63", "CD82")  # core report Table 1 (+ CD34)
LINK_PROTEIN = "CD99"
CLIP = 3.0
MIN_PROJECTION_R = 0.997
DETERMINACY_LINE = 0.707
OUTPUT_TABLES = ("selected_cells.csv", "displayed_features.csv", "panel_values.csv", "regression_checks.csv")
FIGURES = {"core": "feature_space_explainer_core", "extended": "feature_space_explainer_extended"}


# ----------------------------------------------------------------------------------------------
# Pure helpers (no experiment paths; tested in tests/test_feature_space_explainer.py)
# ----------------------------------------------------------------------------------------------

def zscore_columns(x: np.ndarray) -> np.ndarray:
    """Stage 0 standardization: z-score every column over all rows (ddof 0); zero-variance columns become 0."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    zero = std <= 1e-12
    z = (x - mean) / np.where(zero, 1.0, std)
    z[:, zero] = 0.0
    return z


def label_free_gene_order(xz: np.ndarray) -> np.ndarray:
    """Column order from average-linkage clustering of the column correlations (labels never used)."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    r = np.corrcoef(np.asarray(xz, dtype=np.float64), rowvar=False)
    r = np.nan_to_num(r, nan=0.0)
    d = np.clip(1.0 - r, 0.0, 2.0)
    np.fill_diagonal(d, 0.0)
    cond = squareform((d + d.T) / 2.0, checks=False)
    tree = optimal_leaf_ordering(linkage(cond, method="average"), cond)
    return leaves_list(tree)


def select_cells(meta: pd.DataFrame, patients: Sequence[str], highlight_id: str, seed: int = SEED,
                 min_each: int = MIN_EACH_LABEL, types_per_patient: int = TYPES_PER_PATIENT,
                 per_label: int = CELLS_PER_LABEL, n_composition: int = N_COMPOSITION,
                 recipient_extras: bool = True) -> pd.DataFrame:
    """The displayed rows, in display order.

    meta: one row per cell with idx (row in the cohort cache), cell_id, patient, label ('malignant' or
    'healthy'), cell_type and source ('donor' or 'recipient'). Core rows are drawn first for every patient, so
    the recipient-origin extras (extended version only) never change them. Within a patient: the shown types in
    order of malignant count (ties: healthy count, then name), malignant rows before healthy rows, cells sorted
    by cell_id inside a stratum (no score is used), then the composition row, then the extra.
    """
    rng = np.random.default_rng(seed)
    hl = meta.loc[meta["cell_id"] == highlight_id]
    if len(hl) != 1:
        raise ValueError(f"highlight cell {highlight_id!r} not found exactly once")
    hl = hl.iloc[0]
    rows: List[dict] = []
    plan: Dict[str, List[str]] = {}
    for p in patients:
        m = meta[meta["patient"] == p]
        counts = (pd.crosstab(m["cell_type"], m["label"]).reindex(columns=["malignant", "healthy"], fill_value=0)
                  .reset_index().rename(columns={"index": "cell_type"}))
        ok = counts[(counts["malignant"] >= min_each) & (counts["healthy"] >= min_each)]
        ok = ok.sort_values(["malignant", "healthy", "cell_type"], ascending=[False, False, True], kind="mergesort")
        types = list(ok["cell_type"].iloc[:types_per_patient])
        if not types:
            raise ValueError(f"{p}: no predicted type with >= {min_each} cells of each label")
        if hl["patient"] == p and hl["cell_type"] not in types:
            raise ValueError(f"highlight cell type {hl['cell_type']} is not a shown type of {p}")
        plan[p] = types
        for t in types:
            for lab in ("malignant", "healthy"):
                pool = m[(m["cell_type"] == t) & (m["label"] == lab)].sort_values("cell_id")
                forced: List[str] = []
                if hl["patient"] == p and hl["cell_type"] == t and hl["label"] == lab:
                    forced = [str(hl["cell_id"])]
                    pool = pool[pool["cell_id"] != hl["cell_id"]]
                k = per_label - len(forced)
                picked = list(rng.choice(pool["cell_id"].to_numpy(), size=k, replace=False)) if k > 0 else []
                for cid in sorted(forced + [str(c) for c in picked]):
                    rows.append({"cell_id": cid, "patient": p, "block_type": t, "stratum": "type_pair",
                                 "in_core": True})
        healthy_other = m[(m["label"] == "healthy") & (~m["cell_type"].isin(types))]
        vc = healthy_other["cell_type"].value_counts()
        vc = vc.reindex(sorted(vc.index)).sort_values(ascending=False, kind="mergesort")
        comp_type = str(vc.index[0])
        pool = healthy_other[healthy_other["cell_type"] == comp_type].sort_values("cell_id")
        for cid in sorted(str(c) for c in rng.choice(pool["cell_id"].to_numpy(), size=n_composition, replace=False)):
            rows.append({"cell_id": cid, "patient": p, "block_type": comp_type, "stratum": "composition",
                         "in_core": True})
    if recipient_extras:
        chosen = {r["cell_id"] for r in rows}
        for p in patients:
            m = meta[(meta["patient"] == p) & (meta["label"] == "healthy") & (meta["source"] == "recipient")
                     & (~meta["cell_id"].isin(chosen))]
            if len(m):
                cid = str(rng.choice(m.sort_values("cell_id")["cell_id"].to_numpy(), size=1)[0])
                rows.append({"cell_id": cid, "patient": p, "block_type": None, "stratum": "recipient_healthy",
                             "in_core": False})
    sel = pd.DataFrame(rows)
    sel = sel.merge(meta, on=["cell_id", "patient"], how="left", validate="one_to_one")
    sel.loc[sel["stratum"] == "recipient_healthy", "block_type"] = sel["cell_type"]
    order = {p: i for i, p in enumerate(patients)}
    stratum_rank = {"type_pair": 0, "composition": 1, "recipient_healthy": 2}
    sel["_p"] = sel["patient"].map(order)
    sel["_s"] = sel["stratum"].map(stratum_rank)
    sel["_t"] = [plan[p].index(t) if s == "type_pair" else 0 for p, t, s in
                 zip(sel["patient"], sel["block_type"], sel["stratum"])]
    sel["_l"] = (sel["label"] != "malignant").astype(int)
    sel = sel.sort_values(["_p", "_s", "_t", "_l", "cell_id"], kind="mergesort").drop(columns=["_p", "_s", "_t", "_l"])
    sel["highlighted"] = sel["cell_id"] == highlight_id
    return sel.reset_index(drop=True)


def wrap(text: str, width_in: float, size: float) -> str:
    """Wrap text to a width in inches at a font size in points (DejaVu Sans averages about 0.55 em)."""
    n = max(12, int(width_in * 72.0 / (size * 0.55)))
    return "\n".join(textwrap.fill(par, n) for par in text.split("\n"))


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_signed(v: float, nd: int = 3) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


def fmt_pct(v: float) -> str:
    """Prevalence as the report prints it: one decimal from 1%, two below."""
    return f"{v:.1f}%" if v >= 1 else f"{v:.2f}%"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def _helpers():
    """make_figures_v3 (D27 loaders), figs_05_pc8 (D18 readers) and factor_cards (D18 library), read-only."""
    import importlib.util

    sys.path.insert(0, str(REPORT_DIR))
    sys.path.insert(0, str(PAVAN_DIR))
    import figs_05_pc8 as f5  # noqa: E402
    import make_figures_v3 as mx  # noqa: E402

    spec = importlib.util.spec_from_file_location("factor_cards_d30", W5_DIR / "factor_cards.py")
    fc = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fc)
    return mx, f5, fc


def load_stage1(exp: Path, method: str) -> dict:
    base = exp / "stage1_dr" / f"panel_id={PANEL}" / f"method={method}" / "k=40" / "seed=42"
    scores = np.asarray(np.load(base / "scores.npy"), dtype=np.float64)
    loadings = np.asarray(np.load(base / "loadings.npy"), dtype=np.float64)
    meta = json.loads((base / "metadata.json").read_text())
    if loadings.shape[0] != meta["n_genes"]:
        loadings = loadings.T
    return {"base": base, "scores": scores, "loadings": loadings, "meta": meta,
            "diag": json.loads((base / "diagnostics.json").read_text())}


def display_z(scores: np.ndarray) -> np.ndarray:
    """Per-column z over all cells for display; zero-variance (degenerate) columns become NaN."""
    sd = scores.std(axis=0)
    z = (scores - scores.mean(axis=0)) / np.where(sd > 1e-12, sd, 1.0)
    z[:, sd <= 1e-12] = np.nan
    return z


class Checks:
    def __init__(self) -> None:
        self.rows: List[dict] = []

    def add(self, name: str, value, expected=None, tol: Optional[float] = None, passed: Optional[bool] = None) -> None:
        if passed is None:
            if tol is not None:
                passed = bool(abs(float(value) - float(expected)) <= tol)
            elif expected is not None:
                passed = bool(value == expected)
            else:
                passed = True
        self.rows.append({"check": name, "value": value, "expected": expected, "tolerance": tol, "passed": bool(passed)})
        if not passed:
            raise AssertionError(f"check failed: {name}: {value!r} vs {expected!r} (tol {tol})")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def load_all(exp: Path, cfg: dict, checks: Checks) -> dict:
    import anndata as ad

    mx, f5, fc = _helpers()
    d27 = mx.load_d27()
    cohort = mx.load_cells(d27["orient"])
    fig20 = mx.fig5_data(d27, cohort)  # asserts PC8 x CD99 rho at every level against D27
    cells = cohort["cells"]

    adata = ad.read_h5ad(exp / cfg["cohort_cache_relpath"])
    genes = json.loads((exp / "preprocessing" / "panels" / "stage0_panel_genes" / f"{PANEL}.json").read_text())["genes"]
    x = adata[:, genes].X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    obs = adata.obs
    cell_ids = np.asarray(adata.obs_names).astype(str)
    del adata
    meta = pd.DataFrame({
        "idx": np.arange(len(cell_ids)),
        "cell_id": cell_ids,
        "patient": obs["patient"].astype(str).to_numpy(),
        "cn_label": obs["CN.label"].astype(str).to_numpy(),
        "cell_type": obs["predicted.annotation"].astype(str).to_numpy(),
        "source": obs["source"].astype(str).to_numpy(),
        "batch": obs["batch"].astype(str).to_numpy(),
        "sample": obs["sample"].astype(str).to_numpy(),
    })
    meta["label"] = np.where(meta["cn_label"] == "cancer", "malignant", "healthy")

    cm = pd.read_csv(exp / cfg["cell_metadata_relpath"], usecols=["cell_id"])
    checks.add("row order: cell_metadata.csv.gz cell_id == cohort cache obs_names",
               bool((cm["cell_id"].astype(str).to_numpy() == cell_ids).all()), True)
    checks.add("row order: make_figures_v3.load_cells == cohort cache",
               bool((cells["cell_id"].to_numpy() == cell_ids).all()), True)
    checks.add("load_cells label == CN.label mapping",
               bool((np.where(cells["label"] == "malignant", "malignant", "healthy") == meta["label"]).all()), True)
    coh = cfg["cohort"]
    checks.add("n_cells", len(meta), int(coh["n_cells"]))
    checks.add("n_malignant", int((meta["label"] == "malignant").sum()), int(coh["n_malignant"]))
    xt = pd.crosstab(meta["label"], meta["source"])
    checks.add("healthy donor cells", int(xt.loc["healthy", "donor"]), int(coh["donor_healthy"]))
    checks.add("healthy recipient cells", int(xt.loc["healthy", "recipient"]), int(coh["recipient_healthy"]))
    checks.add("malignant donor cells", int(xt.loc["malignant"].get("donor", 0)), 0)
    for p in PATIENTS:
        m = meta[meta["patient"] == p]
        checks.add(f"{p} malignant / total",
                   f"{int((m['label'] == 'malignant').sum())}/{len(m)}",
                   f"{cfg['patients'][p]['malignant']}/{cfg['patients'][p]['total']}")

    pca, fa = load_stage1(exp, "pca"), load_stage1(exp, "fa")
    for name, s in (("pca", pca), ("fa", fa)):
        checks.add(f"{name} scores shape", str(s["scores"].shape), str((len(meta), 40)))
        checks.add(f"{name} loadings shape", str(s["loadings"].shape), str((len(genes), 40)))
        checks.add(f"{name} stage1_scope", s["meta"]["stage1_scope"], "across_patient")
        checks.add(f"{name} standardization", s["meta"]["standardization"], "feature_zscore_with_zero_variance_to_zero")
    checks.add("panel genes", len(genes), int(pca["meta"]["n_genes"]))

    xz = zscore_columns(x)
    proj = xz @ pca["loadings"]
    r = np.array([np.corrcoef(proj[:, k], pca["scores"][:, k])[0, 1] for k in range(40)])
    checks.add("PCA: min over 40 factors of r(stored score, Xz @ loadings)", round(float(r.min()), 5),
               passed=bool(r.min() >= MIN_PROJECTION_R))
    checks.add("PCA: PC8 r(stored score, Xz @ loadings)", round(float(r[PC8_COL]), 5),
               passed=bool(r[PC8_COL] >= MIN_PROJECTION_R))

    orient = {"PC8": f5.orientation(PC8_CARD), "FA7": f5.orientation(FA7_CARD)}
    checks.add("D18 orientation sign PC8", orient["PC8"], -1)
    checks.add("D18 orientation sign FA7", orient["FA7"], 1)
    pc8_z = display_z(pca["scores"])[:, PC8_COL] * orient["PC8"]
    checks.add("oriented cohort-z PC8 == make_figures_v3.load_cells PC8",
               float(np.abs(pc8_z - cells["PC8"].to_numpy()).max()), 0.0, tol=1e-6)

    ct = pd.read_csv(f5.cards_dir() / "factor_cards_celltype.csv")
    ct = ct[ct["card_id"] == PC8_CARD]
    d18_label = np.where(meta["label"] == "malignant", "malignant", "normal")
    mine = (pd.DataFrame({"label": d18_label, "cell_type": meta["cell_type"], "z": pc8_z})
            .groupby(["label", "cell_type"])["z"].agg(["mean", "size"]).reset_index())
    cmp_ = ct.merge(mine, on=["label", "cell_type"], how="left")
    checks.add("D18 factor_cards_celltype: PC8 groups reproduced", int(cmp_["mean"].notna().sum()), len(ct))
    checks.add("D18 factor_cards_celltype: max |mean oriented PC8 z diff|",
               float((cmp_["mean"] - cmp_["mean_oriented_z"]).abs().max()), 0.0, tol=1e-4)
    checks.add("D18 factor_cards_celltype: n_cells equal", bool((cmp_["size"] == cmp_["n_cells"]).all()), True)

    idx = pd.read_csv(f5.cards_dir() / "factor_cards_index.csv").set_index("card_id")
    matches = pd.read_csv(f5.cards_dir() / "factor_cards_matches.csv")
    phi_card = float(matches[(matches["card_id"] == PC8_CARD) & (matches["matched_card_id"] == FA7_CARD)]
                     ["tucker_phi_aligned"].iloc[0])
    phi = abs(float(fc.tucker_phi(pca["loadings"][:, PC8_COL], fa["loadings"][:, FA7_COL])))
    checks.add("PC8-FA7 Tucker phi (recomputed from loadings) vs D18", round(phi, 5), round(phi_card, 5), tol=1e-4)
    psi = np.asarray(fa["diag"]["extras"]["uns"]["noise_variance_per_feature"], dtype=np.float64)
    det = np.asarray(fc.factor_determinacy(fa["loadings"], psi), dtype=np.float64)
    det_card = float(idx.loc[FA7_CARD, "determinacy"])
    checks.add("FA7 determinacy (recomputed from loadings and psi) vs D18", round(float(det[FA7_COL]), 5),
               round(det_card, 5), tol=1e-4)
    fa_sd = fa["scores"].std(axis=0)
    degenerate = fa_sd <= 1e-12
    checks.add("FA degenerate factors (zero-variance scores)", int(degenerate.sum()),
               int(40 - int(idx.loc[FA7_CARD, "effective_k"])))

    poles = f5.pc8_pole_table()
    gpos = {g: i for i, g in enumerate(genes)}
    pc8_or = pca["loadings"][:, PC8_COL] * orient["PC8"]
    fa7_or = fa["loadings"][:, FA7_COL] * orient["FA7"]
    same = int(sum(np.sign(pc8_or[gpos[g]]) == np.sign(fa7_or[gpos[g]]) for g in poles["gene"]))
    checks.add("PC8 pole genes (20) with the same FA7 sign, recomputed", same, int(poles["same_sign"].sum()))
    checks.add("highlight gene has PC8's largest |loading|", genes[int(np.argmax(np.abs(pc8_or)))], HIGHLIGHT_GENE)

    ex = fig20["example"].set_index("example_cell")
    hl_id = str(ex.loc[HIGHLIGHT_EXAMPLE_CELL, "cell_id"])
    hl = meta.set_index("cell_id").loc[hl_id]
    checks.add("highlight cell: Figure 20B example cell 4 is P03 malignant HSC/MPP2",
               f"{hl['patient']}/{hl['label']}/{hl['cell_type']}", "P03/malignant/HSC/MPP2")
    checks.add("Figure 20 pooled malignant rho(PC8, CD99) (asserted against D27 by make_figures_v3)",
               round(float(fig20["rho_pooled"]), 4), passed=True)

    ev = np.asarray(pca["diag"]["extras"]["uns"]["explained_variance"], dtype=np.float64)
    evr = np.asarray(pca["diag"]["extras"]["uns"]["explained_variance_ratio"], dtype=np.float64)

    names = cohort["names"]
    iso = mx.isotype_refs(d27["adt"]).loc[("PC8", "malignant")]
    clone = str(iso["iso_within_patient_set_by"])
    iso_name = [n for n in names if "sotype" in n and n.endswith(clone)]
    if len(iso_name) != 1:
        raise ValueError(f"isotype {clone!r}: {iso_name}")
    prot_cols = [names.index(f"ADT-{p}") for p in PROTEINS] + [names.index(iso_name[0])]
    prot_labels = list(PROTEINS) + [f"isotype ({clone})"]
    clr = cohort["clr"][:, prot_cols]

    return {"meta": meta, "genes": genes, "xz": xz, "pca": pca, "fa": fa, "orient": orient, "pc8_z": pc8_z,
            "proj_r": r, "ev": ev, "evr": evr, "det": det, "degenerate": degenerate, "phi": phi,
            "same_sign": same, "poles": poles, "pc8_or": pc8_or, "fa7_or": fa7_or, "hl_id": hl_id,
            "rho_cd99": float(fig20["rho_pooled"]), "n_mal_all": int(fig20["n_mal"]), "clr": clr,
            "prot_labels": prot_labels, "iso_clone": clone, "fig20_example": ex.reset_index()}


# ----------------------------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------------------------

PATIENT_FILL = {"P01": "#e2eaf5", "P02": "#e1efdf", "P03": "#ebe3f3", "P04": "#f6e6d6", "P09": "#efecd0"}
PATIENT_EDGE = {"P01": "#9db4d6", "P02": "#97c091", "P03": "#b7a1d3", "P04": "#deae80", "P09": "#c9c07b"}
GENE_FILL, GENE_EDGE, GENE_TITLE = "#f3f1ec", "#d3d0c5", "#3d3c38"
PCA_FILL, PCA_EDGE, PCA_TITLE = "#e7f0fb", "#b4cfef", "#1d5aa6"
FA_FILL, FA_EDGE, FA_TITLE = "#fcede5", "#f1c3ab", "#b3461c"
PROT_FILL, PROT_EDGE, PROT_TITLE = "#e3f5ee", "#a6dcc6", "#127a55"


class Canvas:
    """One full-figure axes in inches; positions are given from the top edge."""

    def __init__(self, w: float, h: float):
        import matplotlib.pyplot as plt

        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_autoscale_on(False)
        self.ax.axis("off")

    def y(self, top: float) -> float:
        return self.h - top

    def text(self, x: float, top: float, s: str, **kw):
        kw.setdefault("va", "baseline")
        return self.ax.text(x, self.y(top), s, **kw)

    def rect(self, x: float, top: float, w: float, h: float, **kw):
        from matplotlib.patches import Rectangle

        return self.ax.add_patch(Rectangle((x, self.y(top + h)), w, h, **kw))

    def rbox(self, x: float, top: float, w: float, h: float, fc: str, ec: str, lw: float = 0.8, r: float = 0.07,
             **kw):
        from matplotlib.patches import FancyBboxPatch

        return self.ax.add_patch(FancyBboxPatch((x, self.y(top + h)), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                                facecolor=fc, edgecolor=ec, lw=lw, **kw))

    def line(self, xs: Sequence[float], tops: Sequence[float], **kw):
        return self.ax.plot(list(xs), [self.y(t) for t in tops], **kw)

    def image(self, m: np.ndarray, x: float, top: float, w: float, h: float, cmap, vmin: float, vmax: float, z: int = 2):
        return self.ax.imshow(m, extent=(x, x + w, self.y(top + h), self.y(top)), origin="upper", cmap=cmap,
                              vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto", zorder=z)

    def arrow(self, x0: float, t0: float, x1: float, t1: float, color: str, lw: float = 1.0, style: str = "-|>",
              ms: float = 9, ls: str = "-", rad: float = 0.0, z: int = 5):
        return self.ax.annotate("", xy=(x1, self.y(t1)), xytext=(x0, self.y(t0)), zorder=z,
                                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                                mutation_scale=ms, linestyle=ls,
                                                connectionstyle=f"arc3,rad={rad}"))


def heat_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    rs = _style()
    cmap = LinearSegmentedColormap.from_list("d30_div", [rs.DIV_NEG, rs.DIV_MID, rs.DIV_POS], N=256)
    cmap.set_bad("#e6e5df")
    return cmap


def _style():
    sys.path.insert(0, str(REPORT_DIR))
    import report_style as rs  # noqa: E402

    return rs


class Rows:
    """Vertical geometry of the displayed rows: patient blocks separated by a gap."""

    def __init__(self, sel: pd.DataFrame, top: float, rh: float, gap: float):
        self.sel = sel
        self.rh, self.gap = rh, gap
        tops, t, prev = [], top, None
        for p in sel["patient"]:
            if prev is not None and p != prev:
                t += gap
            tops.append(t)
            t += rh
            prev = p
        self.tops = np.asarray(tops)
        self.top, self.bottom = top, t
        self.patients = list(dict.fromkeys(sel["patient"]))
        self.span = {p: (float(self.tops[(sel["patient"] == p).to_numpy()].min()),
                         float(self.tops[(sel["patient"] == p).to_numpy()].max()) + rh) for p in self.patients}

    def mid(self, i: int) -> float:
        return float(self.tops[i] + self.rh / 2)

    def blocks(self) -> List[Tuple[str, str, int, int]]:
        """(patient, block label, first row, last row) for consecutive rows of one patient and type/stratum."""
        out, start = [], 0
        keys = list(zip(self.sel["patient"], self.sel["block_type"], self.sel["stratum"]))
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                out.append((keys[start][0], keys[start][1], start, i - 1))
                start = i
        return out


def draw_heat_block(c: Canvas, m: np.ndarray, x: float, rows: Rows, cw: float, cmap, vmin: float, vmax: float) -> float:
    """Heatmap of the displayed rows (one image per patient so gaps stay empty); returns the right edge."""
    for p in rows.patients:
        ii = np.where((rows.sel["patient"] == p).to_numpy())[0]
        t0, t1 = rows.span[p]
        c.image(m[ii], x, t0, cw * m.shape[1], t1 - t0, cmap, vmin, vmax)
    rs = _style()
    for (_, _, a, b) in rows.blocks()[:-1]:
        if b + 1 < len(rows.sel) and rows.sel["patient"].iloc[b + 1] == rows.sel["patient"].iloc[b]:
            yy = rows.tops[b + 1]
            c.line([x, x + cw * m.shape[1]], [yy, yy], color=rs.SURFACE, lw=0.9, zorder=3)
    return x + cw * m.shape[1]


def draw_annotations(c: Canvas, rows: Rows, x: Dict[str, float], detailed: bool, cfg: dict, header_top: float,
                     fs: float = 6.4) -> None:
    """Cell number, patient band, label, predicted type and origin for every displayed row."""
    rs = _style()
    sel = rows.sel
    hdr = dict(fontsize=6.6, color=rs.INK_2, fontweight="bold", va="baseline")
    c.text(x["cell"] + x["cell_w"], header_top, "Cell", ha="right", **hdr)
    c.text(x["pat"] + x["pat_w"] / 2, header_top, "Patient", ha="center", **hdr)
    c.text(x["lab"] + 0.03, header_top, "Label", ha="left", **hdr)
    c.text(x["type"] + 0.03, header_top, "Predicted type", ha="left", **hdr)
    c.text(x["orig"] + x["orig_w"] / 2, header_top, "Origin", ha="center", **hdr)
    for p in rows.patients:
        t0, t1 = rows.span[p]
        c.rect(x["pat"], t0, x["pat_w"], t1 - t0, facecolor=PATIENT_FILL[p], edgecolor=PATIENT_EDGE[p], lw=0.8, zorder=2)
        if detailed:
            n_mal, n_tot = cfg["patients"][p]["malignant"], cfg["patients"][p]["total"]
            batch = "/".join(sorted(set(sel.loc[sel["patient"] == p, "batch"])))
            lines = [(p, 8.6, "bold", rs.INK), (f"{fmt_int(n_mal)} malignant", 5.9, "normal", rs.INK_2),
                     (f"of {fmt_int(n_tot)} cells", 5.9, "normal", rs.INK_2),
                     (f"π {fmt_pct(100 * n_mal / n_tot)} · {batch}", 5.9, "normal", rs.INK_2)]
            if p in HIGH_PREVALENCE:
                lines.append(("highest prevalence", 5.7, "normal", rs.INK))
            gaps = [0.0, 0.112] + [0.086] * (len(lines) - 2)
            span = sum(gaps) + 0.07
            tt = (t0 + t1) / 2 - span / 2 + 0.075
            for (s, size, weight, col), g in zip(lines, gaps):
                tt += g
                c.text(x["pat"] + x["pat_w"] / 2, tt, s, ha="center", fontsize=size, fontweight=weight, color=col,
                       zorder=3, style="italic" if s == "highest prevalence" else "normal")
        else:
            c.text(x["pat"] + x["pat_w"] / 2, (t0 + t1) / 2 + 0.035, p, ha="center", fontsize=8.0, fontweight="bold",
                   color=rs.INK, zorder=3)
    for i, r in sel.iterrows():
        ym = rows.mid(i)
        c.text(x["cell"] + x["cell_w"], ym + 0.024, str(i + 1), ha="right", fontsize=5.6, color=rs.MUTED)
        mal = r["label"] == "malignant"
        c.ax.plot([x["lab"] + 0.07], [c.y(ym)], marker="o", markersize=3.6, linestyle="none", zorder=4,
                  markerfacecolor=rs.INK if mal else rs.SURFACE, markeredgecolor=rs.INK if mal else rs.MUTED,
                  markeredgewidth=0.8)
        c.text(x["lab"] + 0.14, ym + 0.024, "malignant" if mal else "healthy", ha="left", fontsize=fs,
               color=rs.INK if mal else rs.INK_2, fontweight="bold" if mal else "normal")
        rec = r["source"] == "recipient"
        c.text(x["orig"] + x["orig_w"] / 2, ym + 0.024, "R" if rec else "D", ha="center", fontsize=fs,
               color=rs.INK if rec else rs.MUTED, fontweight="bold" if rec else "normal")
    for (p, t, a, b) in rows.blocks():
        y0, y1 = rows.tops[a], rows.tops[b] + rows.rh
        if b > a:
            c.line([x["type"] + 0.012, x["type"] + 0.012], [y0 + 0.012, y1 - 0.012], color=rs.AXIS, lw=0.8)
        c.text(x["type"] + 0.045, (y0 + y1) / 2 + 0.024, str(t), ha="left", fontsize=fs, color=rs.INK)


def outline_row(c: Canvas, rows: Rows, i: int, x0: float, x1: float, color: str, lw: float = 1.3) -> None:
    c.rect(x0 - 0.01, rows.tops[i] - 0.004, x1 - x0 + 0.02, rows.rh + 0.008, facecolor="none", edgecolor=color, lw=lw,
           zorder=8)


def outline_col(c: Canvas, rows: Rows, x: float, cw: float, color: str, lw: float = 1.1) -> None:
    c.rect(x - 0.004, rows.top - 0.012, cw + 0.008, rows.bottom - rows.top + 0.024, facecolor="none", edgecolor=color,
           lw=lw, zorder=7)


def colorbar(c: Canvas, cmap, x: float, top: float, w: float, h: float, ticks: Sequence[Tuple[float, str]],
             vmin: float, vmax: float, label: str, label_w: float) -> None:
    rs = _style()
    c.image(np.linspace(vmin, vmax, 256)[None, :], x, top, w, h, cmap, vmin, vmax)
    c.rect(x, top, w, h, facecolor="none", edgecolor=rs.AXIS, lw=0.5, zorder=3)
    for v, s in ticks:
        xx = x + w * (v - vmin) / (vmax - vmin)
        c.line([xx, xx], [top + h, top + h + 0.03], color=rs.MUTED, lw=0.6)
        c.text(xx, top + h + 0.13, s, ha="center", fontsize=6.0, color=rs.INK_2)
    c.text(x + w + 0.08, top + h * 0.85, wrap(label, label_w, 6.1), ha="left", va="top", fontsize=6.1, color=rs.INK_2,
           linespacing=1.15)


def bracket(c: Canvas, x0: float, x1: float, top: float, label: str, color: str, fs: float = 7.2) -> None:
    """A bracket under a block; the label wraps to the block width (at least 1.5 in)."""
    c.line([x0, x0, x1, x1], [top, top + 0.045, top + 0.045, top], color=color, lw=0.9)
    c.line([(x0 + x1) / 2] * 2, [top + 0.045, top + 0.08], color=color, lw=0.9)
    c.text((x0 + x1) / 2, top + 0.1, wrap(label, max(1.5, x1 - x0), fs), ha="center", va="top", fontsize=fs,
           color=color, linespacing=1.15)


def header_box(c: Canvas, x: float, top: float, w: float, title: str, sub: str, fc: str, ec: str, tc: str,
               h: float = 0.5) -> None:
    """A rounded title box; the type shrinks to fit narrow boxes."""
    rs = _style()
    c.rbox(x, top, w, h, fc, ec, lw=0.9)
    fs_t = min(10.6, 0.92 * w * 72 / (len(title) * 0.62))
    plain = len(sub.replace("$", "").replace("\\tilde", "").replace("\\times", "x").replace("{", "").replace("}", ""))
    fs_s = min(7.6, 0.95 * w * 72 / (plain * 0.53))
    c.text(x + w / 2, top + 0.2, title, ha="center", fontsize=fs_t, fontweight="bold", color=tc)
    c.text(x + w / 2, top + 0.39, sub, ha="center", fontsize=fs_s, color=rs.INK_2)


def gene_labels(c: Canvas, order_genes: List[str], x_x: float, cw: float, strip_top: float, label_top: float,
                labels: Sequence[str], highlight: str) -> None:
    """Names for a few genes above the loading strip, spread apart, with leader lines to their columns."""
    rs = _style()
    pos = sorted(((order_genes.index(g), g) for g in labels), key=lambda t: t[0])
    xs = [x_x + cw * (j + 0.5) for j, _ in pos]
    widths = [0.065 * len(g) + 0.06 for _, g in pos]
    lx = list(xs)
    for _ in range(200):
        moved = False
        for k in range(1, len(lx)):
            need = (widths[k - 1] + widths[k]) / 2
            if lx[k] - lx[k - 1] < need:
                push = (need - (lx[k] - lx[k - 1])) / 2
                lx[k - 1] -= push
                lx[k] += push
                moved = True
        if not moved:
            break
    for (j, g), x0, xl in zip(pos, xs, lx):
        hl = g == highlight
        c.text(xl, label_top, g, ha="center", fontsize=6.4, color=rs.INK, fontweight="bold" if hl else "normal")
        c.line([xl, xl, x0], [label_top + 0.03, label_top + 0.075, strip_top - 0.012], color=rs.MUTED, lw=0.55)


def panel_box(c: Canvas, x: float, top: float, w: float, h: float, letter: str, title: str, fc: str, ec: str) -> None:
    rs = _style()
    c.rbox(x, top, w, h, fc, ec, lw=0.9)
    c.text(x + 0.12, top + 0.23, letter, ha="left", fontsize=11, fontweight="bold", color=rs.INK)
    c.text(x + 0.36, top + 0.22, title, ha="left", fontsize=8.4, fontweight="bold", color=rs.INK)


def bullets(c: Canvas, x: float, top: float, w: float, items: Sequence[str], fs: float = 6.7,
            line: float = 0.112) -> float:
    rs = _style()
    t = top
    for s in items:
        txt = wrap(s, w - 0.12, fs).split("\n")
        c.text(x, t, "•", ha="left", fontsize=fs, color=rs.INK_2)
        for k, ln in enumerate(txt):
            c.text(x + 0.1, t, ln, ha="left", fontsize=fs, color=rs.INK)
            t += line
        t += 0.02
    return t


def used_in(c: Canvas, x: float, top: float, w: float, s: str, fs: float = 6.3) -> None:
    rs = _style()
    c.text(x, top, wrap("Used in: " + s, w, fs), ha="left", va="top", fontsize=fs, color=rs.INK_2, style="italic",
           linespacing=1.2)


def strip_h(c: Canvas, vals: np.ndarray, x: float, top: float, cw: float, h: float, cmap, vmin: float, vmax: float,
            hl: Optional[int] = None, hl_color: str = "#0b0b0b") -> float:
    """A horizontal vector of colored cells; returns its right edge."""
    rs = _style()
    c.image(np.asarray(vals, dtype=float)[None, :], x, top, cw * len(vals), h, cmap, vmin, vmax)
    c.rect(x, top, cw * len(vals), h, facecolor="none", edgecolor=rs.AXIS, lw=0.5, zorder=3)
    if hl is not None:
        c.rect(x + cw * hl - 0.003, top - 0.012, cw + 0.006, h + 0.024, facecolor="none", edgecolor=hl_color, lw=1.1,
               zorder=4)
    return x + cw * len(vals)


def strip_v(c: Canvas, vals: np.ndarray, rows_patients: Sequence[str], x: float, top: float, w: float, rh: float,
            cmap, vmin: float, vmax: float, hl: Optional[int] = None) -> float:
    """A vertical vector (one value per displayed row) with patient ticks on its left; returns its bottom."""
    rs = _style()
    n = len(vals)
    c.image(np.asarray(vals, dtype=float)[:, None], x, top, w, rh * n, cmap, vmin, vmax)
    c.rect(x, top, w, rh * n, facecolor="none", edgecolor=rs.AXIS, lw=0.5, zorder=3)
    for i, p in enumerate(rows_patients):
        c.rect(x - 0.055, top + rh * i, 0.04, rh, facecolor=PATIENT_EDGE[p], edgecolor="none", zorder=3)
    if hl is not None:
        c.rect(x - 0.012, top + rh * hl - 0.004, w + 0.024, rh + 0.008, facecolor="none", edgecolor=rs.INK, lw=1.1,
               zorder=4)
    return top + rh * n


def footer(c: Canvas, top: float, text: str, w: float) -> None:
    rs = _style()
    c.text(0.3, top, wrap(text, w, 6.3), ha="left", va="top", fontsize=6.3, color=rs.MUTED, linespacing=1.25)


def mathtext_ok() -> bool:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    try:
        fig.text(0, 0, r"$\mathbb{R}^{1\times 92}$")
        fig.canvas.draw()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        plt.close(fig)


def draw(data: dict, sel_all: pd.DataFrame, cfg: dict, version: str, out_pdf: Path, out_png: Path) -> dict:
    """Draw one version; returns the numbers it printed (for panel_values.csv)."""
    import matplotlib.pyplot as plt

    rs = _style()
    rs.apply()
    plt.rcParams["axes.grid"] = False
    extended = version == "extended"
    sel = sel_all if extended else sel_all[sel_all["in_core"]].reset_index(drop=True)
    cmap = heat_cmap()
    RR = r"\mathbb{R}" if mathtext_ok() else "R"
    genes = data["genes"]
    order = data["gene_order"]
    order_genes = [genes[j] for j in order]
    xz_rows = data["xz"][sel["idx"].to_numpy()][:, order]
    z_disp = display_z(data["pca"]["scores"])
    z_disp[:, PC8_COL] *= data["orient"]["PC8"]
    z_rows = z_disp[sel["idx"].to_numpy()]
    hl_i = int(np.where(sel["highlighted"].to_numpy())[0][0])
    g_hl = order_genes.index(HIGHLIGHT_GENE)
    n_rows, n_mal_rows = len(sel), int((sel["label"] == "malignant").sum())
    vals: Dict[str, object] = {}

    cw = 0.043 if not extended else 0.04
    rh, gap = 0.098, 0.075
    # ---------------- horizontal layout
    x0 = 0.3
    L = {"cell": x0, "cell_w": 0.24, "pat": x0 + 0.28, "pat_w": 0.92}
    L.update({"lab": L["pat"] + L["pat_w"] + 0.05, "lab_w": 0.64})
    L.update({"type": L["lab"] + L["lab_w"], "type_w": 0.74})
    L.update({"orig": L["type"] + L["type_w"], "orig_w": 0.28})
    x_x = L["orig"] + L["orig_w"] + 0.07
    x_x_end = x_x + cw * len(genes)
    if not extended:
        gap_w = 1.36
        R = {"cell": x_x_end + gap_w, "cell_w": 0.24}
        R.update({"pat": R["cell"] + 0.28, "pat_w": 0.42})
        R.update({"lab": R["pat"] + R["pat_w"] + 0.05, "lab_w": 0.64})
        R.update({"type": R["lab"] + R["lab_w"], "type_w": 0.74})
        R.update({"orig": R["type"] + R["type_w"], "orig_w": 0.28})
        x_z = R["orig"] + R["orig_w"] + 0.07
        x_z_end = x_z + cw * 40
        W = x_z_end + 1.45
    else:
        gap_w = 1.22
        x_z = x_x_end + gap_w + 0.12
        x_z_end = x_z + cw * 40
        x_f = x_z_end + 0.55
        x_f_end = x_f + cw * 40
        pw = 0.17
        x_p = x_f_end + 0.75
        x_p_end = x_p + pw * len(data["prot_labels"])
        W = x_p_end + 0.45
    # ---------------- vertical layout
    title_top, hdr_top = 0.36, 0.7
    strip_top, label_top, ann_hdr = 1.57, 1.36, 1.81
    rows = Rows(sel, 1.88, rh, gap)
    below = rows.bottom
    n_panels = 6 if extended else 4
    panel_rows = 2 if extended else 1
    panel_h = 2.08 if not extended else 1.86
    panels_top = below + 1.02
    H = panels_top + panel_rows * panel_h + (panel_rows - 1) * 0.16 + 0.62
    c = Canvas(W, H)

    # ---------------- title
    c.text(W / 2, title_top, "The same cells can be represented in different feature spaces", ha="center",
           fontsize=16, fontweight="bold", color=rs.INK)
    sub = (f"Real data: post-transplant MDS MRD CITE-seq · Hallmark IFN-α panel ({len(genes)} genes) → "
           + ("PCA, K = 40" if not extended else "PCA and FA, K = 40; surface protein alongside")
           + f" · {n_rows} of {fmt_int(len(data['meta']))} cells shown (a stratified illustration sample, seed {SEED})")
    c.text(W / 2, title_top + 0.25, sub, ha="center", fontsize=8.6, color=rs.INK_2)

    # ---------------- gene space block
    header_box(c, L["cell"], hdr_top, x_x_end - L["cell"], "Cell × gene matrix (gene space)",
               r"$\tilde{X}$ ($N \times G$):  N = " + fmt_int(len(data["meta"])) + " cells × G = " + str(len(genes))
               + " IFN-α genes (Stage 0 panel; log1p, z-scored per gene over all cells)",
               GENE_FILL, GENE_EDGE, GENE_TITLE)
    load_or = data["pc8_or"][order]
    lim = float(np.ceil(np.abs(load_or).max() * 20) / 20)
    strip_h(c, load_or, x_x, strip_top, cw, 0.12, cmap, -lim, lim, hl=g_hl)
    pole_set = set(data["poles"]["gene"])
    for j, g in enumerate(order_genes):
        if g in pole_set:
            c.line([x_x + cw * (j + 0.5)] * 2, [strip_top + 0.125, strip_top + 0.16], color=rs.INK_2, lw=0.6)
    c.text(x_x - 0.06, strip_top + 0.095, "PC8 loading (×−1)", ha="right", fontsize=6.2, color=rs.INK_2)
    gene_labels(c, order_genes, x_x, cw, strip_top, label_top, LABELLED_GENES, HIGHLIGHT_GENE)
    draw_annotations(c, rows, L, True, cfg, ann_hdr)
    draw_heat_block(c, xz_rows, x_x, rows, cw, cmap, -CLIP, CLIP)
    outline_col(c, rows, x_x + cw * g_hl, cw, rs.INK, lw=1.0)
    outline_row(c, rows, hl_i, L["cell"], x_x_end, rs.INK)
    bracket(c, x_x, x_x_end, below + 0.07, f"Genes (G = {len(genes)}; label-free clustered order; ticks under the "
            "strip: the 20 PC8 pole genes of Figure 5a)", rs.INK_2)
    colorbar(c, cmap, x_x, below + 0.42, 1.25, 0.075, [(-3, "−3"), (0, "0"), (3, "+3")], -CLIP, CLIP,
             "z-score of log1p expression (the Stage 1 input; colour clipped at ±3; most negative values are zero "
             "counts)", 2.5)
    vals["gene_strip_loading_limit"] = lim

    # legend under the annotation columns
    leg_top = below + 0.2
    c.text(L["cell"], leg_top, wrap("Label: CN.label (cancer = malignant, normal = healthy). Origin: source "
                                    "(R recipient, D donor). Predicted type: reference-mapped "
                                    "(for a malignant cell, its nearest healthy state).", 2.85, 6.1),
           ha="left", va="top", fontsize=6.1, color=rs.INK_2, linespacing=1.2)

    # ---------------- Stage 1 arrow
    hl_y = rows.mid(hl_i)
    mid = (rows.top + rows.bottom) / 2
    a_top = max(mid - 0.15, hl_y + 0.55)
    ax0, ax1 = x_x_end + 0.12, (R["cell"] if not extended else x_z - 0.12) - 0.1
    from matplotlib.patches import FancyArrow

    c.ax.add_patch(FancyArrow(ax0, c.y(a_top), ax1 - ax0, 0, width=0.2, head_width=0.38, head_length=0.22,
                              length_includes_head=True, facecolor="#cfcec7", edgecolor="none", zorder=4))
    cx = (ax0 + ax1) / 2
    c.text(cx, a_top - 0.3, "Stage 1", ha="center", fontsize=10, fontweight="bold", color=rs.INK)
    if not extended:
        lines = [("PCA, K = 40, seed 42", PCA_TITLE, "bold"), ("fit once on all", rs.INK_2, "normal"),
                 (f"{fmt_int(len(data['meta']))} cells of", rs.INK_2, "normal"), ("13 patients;", rs.INK_2, "normal"),
                 ("labels never used", rs.INK_2, "normal"), ("(transductive)", rs.INK_2, "normal")]
    else:
        lines = [("PCA → Z, FA → Z′", rs.INK, "bold"), ("K = 40, seed 42; both", rs.INK_2, "normal"),
                 ("fit once on all", rs.INK_2, "normal"), (f"{fmt_int(len(data['meta']))} cells of", rs.INK_2, "normal"),
                 ("13 patients; labels", rs.INK_2, "normal"), ("never used", rs.INK_2, "normal"),
                 ("(transductive)", rs.INK_2, "normal")]
    tt = a_top + 0.36
    for s, col, wgt in lines:
        c.text(cx, tt, s, ha="center", fontsize=7.0, color=col, fontweight=wgt)
        tt += 0.125
    c.text(cx, tt + 0.05, r"$Z \approx \tilde{X}W$", ha="center", fontsize=8.6, color=rs.INK)
    c.text(cx, tt + 0.2, "W: 92 × 40 loadings", ha="center", fontsize=6.4, color=rs.INK_2)

    # ---------------- factor space block (PCA)
    if not extended:
        header_box(c, R["cell"], hdr_top, x_z_end - R["cell"], "Cell × factor matrix (factor space)",
                   r"$Z$ ($N \times K$):  K = 40 PCA factors (stored Stage 1 scores)", PCA_FILL, PCA_EDGE, PCA_TITLE)
        draw_annotations(c, rows, R, False, cfg, ann_hdr)
    else:
        header_box(c, x_z - 0.12, hdr_top, x_f_end - x_z + 0.12, "Cell × factor matrices (factor space)",
                   r"$Z$: PCA, K = 40    $Z'$: FA, K = 40 (same cells, same order)", PCA_FILL, PCA_EDGE, PCA_TITLE)
        for xx in (x_z, x_f, x_p):
            for p in rows.patients:
                t0, t1 = rows.span[p]
                c.rect(xx - 0.1, t0, 0.06, t1 - t0, facecolor=PATIENT_EDGE[p], edgecolor="none", zorder=2)
    evr = data["evr"]
    bar_bottom, bar_h = strip_top + 0.16, 0.42
    vmax_bar = float(evr.max())
    for k in range(40):
        hk = bar_h * evr[k] / vmax_bar
        col = rs.METHOD["pca"] if k == PC8_COL else ("#9a9890" if k in (PC8_COL - 1, PC8_COL + 1) else "#c9c8bf")
        c.rect(x_z + cw * k + cw * 0.12, bar_bottom - hk, cw * 0.76, hk, facecolor=col, edgecolor="none", zorder=3)
    c.text(x_z + cw * (PC8_COL + 0.5), bar_bottom - bar_h * evr[PC8_COL] / vmax_bar - 0.05,
           f"PC8 {100 * evr[PC8_COL]:.2f}%", ha="left", fontsize=6.0, color=PCA_TITLE)
    c.text(x_z - 0.06, bar_bottom - 0.22, "% of panel\nvariance", ha="right", va="center", fontsize=6.0,
           color=rs.INK_2, linespacing=1.1)
    vals["pc8_variance_pct"] = 100 * float(evr[PC8_COL])
    vals["pc7_over_pc8_minus_1_pct"] = 100 * float(data["ev"][PC8_COL - 1] / data["ev"][PC8_COL] - 1)
    vals["pca40_total_variance_pct"] = 100 * float(evr.sum())
    draw_heat_block(c, z_rows, x_z, rows, cw, cmap, -CLIP, CLIP)
    outline_col(c, rows, x_z + cw * PC8_COL, cw, rs.METHOD["pca"], lw=1.5)
    outline_row(c, rows, hl_i, (R["cell"] if not extended else x_z), x_z_end, rs.INK)
    for k, s in ((0, "PC1"), (PC8_COL, "PC8"), (19, "PC20"), (39, "PC40")):
        c.text(x_z + cw * (k + 0.5), below + 0.13, s, ha="center", fontsize=6.0,
               color=PCA_TITLE if k == PC8_COL else rs.INK_2, fontweight="bold" if k == PC8_COL else "normal")
    bracket(c, x_z, x_z_end, below + 0.18, "Factors (K = 40, variance order)", rs.INK_2)
    colorbar(c, cmap, x_z, below + 0.55, 0.85, 0.075, [(-3, "−3"), (0, "0"), (3, "+3")], -CLIP, CLIP,
             "factor score, z over all cells (display only); PC8 ×(−1), other signs arbitrary", 1.25)
    c.text(x_z + cw * (PC8_COL + 0.5), rows.top - 0.035, "×(−1)", ha="center", fontsize=5.6, color=PCA_TITLE)

    # connector: the same cell in both blocks
    xa = x_x_end + 0.03
    xb = (R["cell"] if not extended else x_z) - 0.03
    c.line([xa, xb], [hl_y, hl_y], color=rs.INK, lw=0.8, ls=(0, (3, 2)), zorder=6)
    c.text((xa + xb) / 2, hl_y - 0.045, f"same cell (row {hl_i + 1})", ha="center", fontsize=6.3, color=rs.INK,
           zorder=6)

    if not extended:
        # Stage 2 pointer
        sx = x_z_end + 0.22
        st = mid - 0.55
        c.arrow(x_z_end + 0.05, mid, sx - 0.02, mid, color=rs.INK_2, lw=1.0)
        c.rbox(sx, st, 1.12, 1.1, "#f6f5f1", "#d6d4cb", lw=0.8)
        txt = ("Stage 2 reads rows of Z: a penalized logistic probe trained on 12 patients ranks the held-out "
               "patient's cells (LOPO: label transfer, not deployment)")
        c.text(sx + 0.08, st + 0.18, "Stage 2", ha="left", fontsize=7.6, fontweight="bold", color=rs.INK)
        c.text(sx + 0.08, st + 0.28, wrap(txt.replace("Stage 2 reads", "Reads"), 1.0, 6.3), ha="left", va="top",
               fontsize=6.3, color=rs.INK_2, linespacing=1.2)
    else:
        # FA block
        det = data["det"]
        degenerate = data["degenerate"]
        f_disp = display_z(data["fa"]["scores"])
        f_disp[:, FA7_COL] *= data["orient"]["FA7"]
        f_rows = f_disp[sel["idx"].to_numpy()]
        for k in range(40):
            if degenerate[k]:
                c.text(x_f + cw * (k + 0.5), bar_bottom - 0.02, "·", ha="center", fontsize=6, color=rs.MUTED)
                continue
            hk = bar_h * det[k]
            col = rs.METHOD["fa"] if k == FA7_COL else "#c9c8bf"
            c.rect(x_f + cw * k + cw * 0.12, bar_bottom - hk, cw * 0.76, hk, facecolor=col, edgecolor="none", zorder=3)
        yl = bar_bottom - bar_h * DETERMINACY_LINE
        c.line([x_f, x_f_end], [yl, yl], color=rs.INK_2, lw=0.6, ls=(0, (2, 2)))
        c.text(x_f_end + 0.03, yl + 0.02, "0.707", ha="left", fontsize=5.6, color=rs.INK_2)
        c.text(x_f - 0.06, bar_bottom - 0.22, "determi-\nnacy", ha="right", va="center", fontsize=6.0, color=rs.INK_2,
               linespacing=1.1)
        c.text(x_f + cw * (FA7_COL + 0.5), bar_bottom - bar_h * det[FA7_COL] - 0.05, f"FA7 {det[FA7_COL]:.2f}",
               ha="left", fontsize=6.0, color=FA_TITLE)
        draw_heat_block(c, f_rows, x_f, rows, cw, cmap, -CLIP, CLIP)
        for k in np.where(degenerate)[0]:
            c.rect(x_f + cw * k, rows.top, cw, rows.bottom - rows.top, facecolor="none", edgecolor=rs.AXIS,
                   hatch="////", lw=0.0, zorder=3)
        outline_col(c, rows, x_f + cw * FA7_COL, cw, rs.METHOD["fa"], lw=1.5)
        outline_row(c, rows, hl_i, x_f, x_f_end, rs.INK)
        for k, s in ((0, "FA1"), (FA7_COL, "FA7"), (19, "FA20"), (39, "FA40")):
            c.text(x_f + cw * (k + 0.5), below + 0.13, s, ha="center", fontsize=6.0,
                   color=FA_TITLE if k == FA7_COL else rs.INK_2, fontweight="bold" if k == FA7_COL else "normal")
        bracket(c, x_f, x_f_end, below + 0.18, f"Factors (K = 40; {int(degenerate.sum())} degenerate, hatched)",
                rs.INK_2)
        c.text(x_f + cw * (FA7_COL + 0.5), rows.top - 0.035, "+1", ha="center", fontsize=5.6, color=FA_TITLE)
        # protein block
        header_box(c, x_p - 0.12, hdr_top, x_p_end - x_p + 0.24, "Surface protein (measured)",
                   r"$P$: 170 antibodies, CLR (no dsb); 9 shown", PROT_FILL, PROT_EDGE, PROT_TITLE)
        clr = data["clr"]
        p_disp = zscore_columns(clr)
        p_rows = p_disp[sel["idx"].to_numpy()]
        for p in rows.patients:
            ii = np.where((rows.sel["patient"] == p).to_numpy())[0]
            t0, t1 = rows.span[p]
            c.image(p_rows[ii], x_p, t0, pw * p_rows.shape[1], t1 - t0, cmap, -CLIP, CLIP)
        j_link = data["prot_labels"].index(LINK_PROTEIN)
        outline_col(c, rows, x_p + pw * j_link, pw, rs.AQUA, lw=1.5)
        outline_row(c, rows, hl_i, x_p, x_p_end, rs.INK)
        for j, s in enumerate(data["prot_labels"]):
            c.ax.text(x_p + pw * (j + 0.5), c.y(rows.top - 0.06), s, rotation=60, ha="left", va="bottom",
                      fontsize=6.1, color=PROT_TITLE if s == LINK_PROTEIN else rs.INK_2,
                      fontweight="bold" if s == LINK_PROTEIN else "normal", rotation_mode="anchor")
        bracket(c, x_p, x_p_end, below + 0.18, "Proteins (CLR z, display only)", rs.INK_2)
        # dashed link: read with rho, not fit
        xa_l, xb_l = x_z + cw * (PC8_COL + 0.5), x_p + pw * (j_link + 0.5)
        c.arrow(xb_l, rows.top - 0.42, xa_l, rows.top - 0.06, color=rs.AQUA, lw=1.1, style="-", ls=(0, (3, 2)),
                rad=0.32)
        c.text((xa_l + xb_l) / 2 + 0.25, rows.top - 0.83, f"read with ρ, not fit (Spearman, among cells of one "
               f"label): all {fmt_int(data['n_mal_all'])} malignant cells, PC8 vs CD99: ρ = "
               f"{fmt_signed(data['rho_cd99'])}", ha="center", fontsize=6.3, color=PROT_TITLE)
        vals["rho_pc8_cd99_malignant_pooled"] = data["rho_cd99"]
        vals["fa7_determinacy"] = float(det[FA7_COL])
        vals["pc8_fa7_tucker_phi"] = data["phi"]
        vals["pc8_fa7_same_sign_pole_genes"] = data["same_sign"]

    # ---------------- panels
    pw_ = (W - 0.6 - (n_panels // panel_rows - 1) * 0.15) / (n_panels // panel_rows)
    pos = []
    for k in range(n_panels):
        rr, cc = divmod(k, n_panels // panel_rows)
        pos.append((0.3 + cc * (pw_ + 0.15), panels_top + rr * (panel_h + 0.16)))
    hl_row = sel.iloc[hl_i]
    hl_desc = (f"row {hl_i + 1}: {hl_row['patient']} · {hl_row['label']} · {hl_row['cell_type']} · "
               f"{hl_row['source']}")

    # A
    xA, tA = pos[0]
    panel_box(c, xA, tA, pw_, panel_h, "A", r"Gene space · one cell (a row of $\tilde{X}$)", GENE_FILL, GENE_EDGE)
    c.text(xA + 0.14, tA + 0.47, r"$\tilde{x}_i = (\tilde{x}_{i,1}, \ldots, \tilde{x}_{i,G}) \in " + RR
           + r"^{1 \times 92}$", ha="left", fontsize=8.4, color=rs.INK)
    cwA = (pw_ - 0.3) / len(genes)
    strip_h(c, xz_rows[hl_i], xA + 0.14, tA + 0.58, cwA, 0.16, cmap, -CLIP, CLIP, hl=g_hl)
    c.text(xA + 0.14, tA + 0.9, hl_desc, ha="left", fontsize=6.6, color=rs.INK, fontweight="bold")
    c.text(xA + 0.14, tA + 1.02, f"{hl_row['cell_id']} (Figure 20B, example cell {HIGHLIGHT_EXAMPLE_CELL})",
           ha="left", fontsize=6.0, color=rs.MUTED)
    t = bullets(c, xA + 0.14, tA + 1.18, pw_ - 0.26, [
        "Its 92 z-scored IFN-α genes are its coordinates in gene space; Stage 0 kept these 92 of "
        f"{fmt_int(cfg['cohort']['n_genes'])} genes.",
        "Patient, label, predicted type and origin belong to the cell, not to the coordinates.",
    ])
    used_in(c, xA + 0.14, t + 0.0, pw_ - 0.26, "the direct-gene probe, Stage 2 on these 92 coordinates (the no-DR "
            "control of Q3; D9).")

    # B
    xB, tB = pos[1]
    panel_box(c, xB, tB, pw_, panel_h, "B", r"Gene space · one gene (a column of $\tilde{X}$)", GENE_FILL, GENE_EDGE)
    c.text(xB + 0.14, tB + 0.47, r"$\tilde{X}_{:,g} = (\tilde{x}_{1,g}, \ldots, \tilde{x}_{N,g})^{\top} \in "
           + RR + r"^{N \times 1}$,  g = " + HIGHLIGHT_GENE, ha="left", fontsize=8.4, color=rs.INK)
    rhB = min(0.026, (panel_h - 0.72) / n_rows)
    vb = strip_v(c, xz_rows[:, g_hl], list(sel["patient"]), xB + 0.24, tB + 0.6, 0.12, rhB, cmap, -CLIP, CLIP,
                 hl=hl_i)
    c.text(xB + 0.30, vb + 0.12, f"{n_rows} rows", ha="center", fontsize=5.8, color=rs.MUTED)
    # mean IFI27 z per patient x label over all of each patient's cells
    gi = genes.index(HIGHLIGHT_GENE)
    meta = data["meta"]
    dx0, dw = xB + 0.62, 1.15
    dtop = tB + 0.66
    c.text(dx0, dtop - 0.04, f"mean {HIGHLIGHT_GENE} z, all cells", ha="left", fontsize=6.1, color=rs.INK_2)
    lo, hi = -1.0, 1.0
    allv = []
    for p in PATIENTS:
        for lab in ("healthy", "malignant"):
            mm = (meta["patient"] == p).to_numpy() & (meta["label"] == lab).to_numpy()
            allv.append(float(data["xz"][mm, gi].mean()))
    lo, hi = min(lo, min(allv) - 0.1), max(hi, max(allv) + 0.1)
    for k, p in enumerate(PATIENTS):
        yy = dtop + 0.1 + k * 0.17
        vh = float(data["xz"][(meta["patient"] == p).to_numpy() & (meta["label"] == "healthy").to_numpy(), gi].mean())
        vm = float(data["xz"][(meta["patient"] == p).to_numpy() & (meta["label"] == "malignant").to_numpy(), gi].mean())
        vals[f"mean_z_{HIGHLIGHT_GENE}_{p}_healthy"] = vh
        vals[f"mean_z_{HIGHLIGHT_GENE}_{p}_malignant"] = vm
        X = lambda v: dx0 + 0.3 + dw * (v - lo) / (hi - lo)  # noqa: E731
        c.text(dx0 + 0.24, yy + 0.022, p, ha="right", fontsize=6.0, color=rs.INK_2)
        c.line([X(vh), X(vm)], [yy, yy], color=rs.AXIS, lw=1.2)
        c.ax.plot([X(vh)], [c.y(yy)], marker="o", markersize=3.8, markerfacecolor=rs.SURFACE,
                  markeredgecolor=rs.MUTED, markeredgewidth=0.9, zorder=5)
        c.ax.plot([X(vm)], [c.y(yy)], marker="o", markersize=3.8, markerfacecolor=rs.INK, markeredgecolor=rs.INK,
                  zorder=5)
    X = lambda v: dx0 + 0.3 + dw * (v - lo) / (hi - lo)  # noqa: E731
    ybase = dtop + 0.1 + 4 * 0.17 + 0.09
    c.line([X(lo), X(hi)], [ybase, ybase], color=rs.AXIS, lw=0.6)
    for v in (-1, 0, 1):
        if lo <= v <= hi:
            c.line([X(v), X(v)], [dtop + 0.02, ybase], color=rs.GRID, lw=0.5, zorder=1)
            c.text(X(v), ybase + 0.11, f"{v:+d}".replace("+0", "0").replace("-", "−"), ha="center", fontsize=5.6,
                   color=rs.INK_2)
    c.text(X(hi) + 0.05, dtop + 0.12, "○ healthy\n● malignant", ha="left", va="top", fontsize=5.8, color=rs.INK_2,
           linespacing=1.3)
    tb = ybase + 0.24
    t = bullets(c, xB + 0.62, tb, pw_ - 0.74, [
        f"{HIGHLIGHT_GENE} is PC8's largest loading (normal-leaning pole), fixed in advance.",
        "A gene varies with patient, predicted type and label at once; a pole gene is a loading on one axis, not a "
        "differential-expression result.",
    ], fs=6.5, line=0.105)
    used_in(c, xB + 0.14, max(t, vb + 0.24), pw_ - 0.26, "Stage 0 panels; within-patient DE (D13); single genes "
            "against the axis (F9).")

    # C
    xC, tC = pos[2]
    panel_box(c, xC, tC, pw_, panel_h, "C", "Factor space · the same cell (a row of Z)", PCA_FILL, PCA_EDGE)
    c.text(xC + 0.14, tC + 0.47, r"$z_i = (z_{i,1}, \ldots, z_{i,K}) \in " + RR + r"^{1 \times 40}$,   "
           r"$z_i \approx \tilde{x}_i W$", ha="left", fontsize=8.4, color=rs.INK)
    cwC = min(cw * 1.4, (pw_ - 0.3) / 40)
    strip_h(c, z_rows[hl_i], xC + 0.14, tC + 0.58, cwC, 0.16, cmap, -CLIP, CLIP, hl=PC8_COL, hl_color=rs.METHOD["pca"])
    c.text(xC + 0.14 + cwC * (PC8_COL + 0.5), tC + 0.88, "PC8", ha="center", fontsize=6.0, color=PCA_TITLE,
           fontweight="bold")
    c.text(xC + 0.14 + cwC * 40 + 0.1, tC + 0.7, wrap(hl_desc + " (unchanged)", pw_ - cwC * 40 - 0.4, 6.4),
           ha="left", va="center", fontsize=6.4, color=rs.INK, fontweight="bold", linespacing=1.2)
    t = bullets(c, xC + 0.14, tC + 1.08, pw_ - 0.26, [
        "The same cell: same patient, label, predicted type and origin. Only its coordinates changed: 40 PCA scores "
        "summarize its 92 genes (W = the 92 × 40 loadings).",
        "Shown per factor as z over all cells; the stored scores keep PCA's decreasing variance (bars above Z).",
    ])
    used_in(c, xC + 0.14, t, pw_ - 0.26, "the Stage 2 probe, which scores rows of Z (LOPO: trained on 12 patients, "
            "it ranks the held-out patient's cells).")

    # D
    xD, tD = pos[3]
    panel_box(c, xD, tD, pw_, panel_h, "D", "Factor space · one factor (a column of Z)", PCA_FILL, PCA_EDGE)
    c.text(xD + 0.14, tD + 0.47, r"$Z_{:,8} \approx \tilde{X} w_8$,   $w_8$ = the 92 PC8 loadings, shown $\times(-1)$",
           ha="left", fontsize=8.4, color=rs.INK)
    srt = np.argsort(data["pc8_or"])
    bw = (pw_ - 0.3) / len(genes)
    b0, bh = tD + 0.86, 0.26
    c.line([xD + 0.14, xD + 0.14 + bw * len(genes)], [b0, b0], color=rs.AXIS, lw=0.6)
    for k, j in enumerate(srt):
        v = data["pc8_or"][j]
        hh = bh * abs(v) / lim
        col = rs.DIV_POS if v > 0 else rs.DIV_NEG
        if v > 0:
            c.rect(xD + 0.14 + bw * k, b0 - hh, bw * 0.8, hh, facecolor=col, edgecolor="none", zorder=3)
        else:
            c.rect(xD + 0.14 + bw * k, b0, bw * 0.8, hh, facecolor=col, edgecolor="none", zorder=3)
    neg = [genes[j] for j in srt[:3]]
    pos3 = [genes[j] for j in srt[::-1][:3]]
    c.text(xD + 0.16, b0 + bh + 0.1, "normal-leaning: " + ", ".join(neg) + " …", ha="left", fontsize=6.2,
           color=rs.DIV_NEG)
    c.text(xD + 0.14 + bw * len(genes), b0 - bh - 0.03, "… " + ", ".join(reversed(pos3)) + " :malignant-leaning",
           ha="right", fontsize=6.2, color=rs.DIV_POS)
    t = bullets(c, xD + 0.14, tD + 1.36, pw_ - 0.26, [
        "Raw PCA signs are arbitrary; PC8 is shown ×(−1) so + is the malignant-leaning end (the sign of its "
        "coefficient in the L1 probe at C = 10⁻⁴).",
        f"'Factor 8' is an index, not an identity: PC8's variance is within "
        f"{vals['pc7_over_pc8_minus_1_pct']:.1f}% of PC7's, and a refit can spread it over several factors. What "
        "it tracks (label, type, patient, batch, origin) has to be tested.",
    ], fs=6.5, line=0.105)
    used_in(c, xD + 0.14, t, pw_ - 0.26, "D14 overlap, D16/D28 variance shares, D18 card, D27 protein ρ, D17/D26 "
            "leave-out refits.")

    if extended:
        # E: protein
        xE, tE = pos[4]
        panel_box(c, xE, tE, pw_, panel_h, "E", "A measured feature space · one protein (a column of P)", PROT_FILL,
                  PROT_EDGE)
        j_link = data["prot_labels"].index(LINK_PROTEIN)
        p_rows = zscore_columns(data["clr"])[sel["idx"].to_numpy()]
        rhE = min(0.024, (panel_h - 0.62) / n_rows)
        vb = strip_v(c, p_rows[:, j_link], list(sel["patient"]), xE + 0.24, tE + 0.42, 0.12, rhE, cmap, -CLIP, CLIP,
                     hl=hl_i)
        c.text(xE + 0.30, vb + 0.12, LINK_PROTEIN, ha="center", fontsize=6.0, color=PROT_TITLE, fontweight="bold")
        t = bullets(c, xE + 0.62, tE + 0.5, pw_ - 0.74, [
            "170 surface proteins measured on the same cells (CLR of raw ADT counts; no dsb). Stage 1 never sees "
            "them: they are used to read the factors, not to fit them.",
            f"One Spearman ρ compares the PC8 column with a protein column among cells of one label: all "
            f"{fmt_int(data['n_mal_all'])} malignant cells, PC8 vs {LINK_PROTEIN}: ρ = {fmt_signed(data['rho_cd99'])} "
            "(Figure 20).",
            "PC8's strongest protein correlates mostly mark predicted cell type rather than persisting inside it "
            "(core report page 8).",
        ], fs=6.5, line=0.105)
        used_in(c, xE + 0.62, t, pw_ - 0.74, "D27 (every card), D29 (total-ADT control).")
        # F: FA
        xF, tF = pos[5]
        panel_box(c, xF, tF, pw_, panel_h, "F", "Another Stage 1 for the same cells (Z′: FA, K = 40)", FA_FILL, FA_EDGE)
        f_disp = display_z(data["fa"]["scores"])
        f_rows = f_disp[sel["idx"].to_numpy()] * np.where(np.arange(40) == FA7_COL, data["orient"]["FA7"], 1)
        vb = strip_v(c, f_rows[:, FA7_COL], list(sel["patient"]), xF + 0.24, tF + 0.42, 0.12, rhE, cmap, -CLIP, CLIP,
                     hl=hl_i)
        c.text(xF + 0.30, vb + 0.12, "FA7", ha="center", fontsize=6.0, color=FA_TITLE, fontweight="bold")
        t = bullets(c, xF + 0.62, tF + 0.5, pw_ - 0.74, [
            f"FA7 is related to PC8, not identical: same sign on {data['same_sign']} of the 20 pole genes, loading "
            f"similarity (Tucker φ) {data['phi']:.2f}.",
            f"FA scores are estimates, not projections (Z′ ≠ X̃W): FA7's determinacy is {data['det'][FA7_COL]:.2f}, "
            f"below the {DETERMINACY_LINE} line; {int(data['degenerate'].sum())} of FA40's factors are degenerate.",
            "The same cells get different coordinates under another method; the cell and its labels do not change.",
        ], fs=6.5, line=0.105)
        used_in(c, xF + 0.62, t, pw_ - 0.74, "D18 (cards), D19 (FA7 and the L1 entry rule), D17 (inductive FA).")

    # ---------------- footer
    pct_rows = 100 * n_mal_rows / n_rows
    pct_all = 100 * cfg["cohort"]["n_malignant"] / cfg["cohort"]["n_cells"]
    rec_h = cfg["cohort"]["recipient_healthy"]
    n_h = cfg["cohort"]["n_cells"] - cfg["cohort"]["n_malignant"]
    note = (f"Rows: a stratified illustration sample (seed {SEED}), not the cohort's composition: {n_mal_rows} of "
            f"{n_rows} rows ({pct_rows:.0f}%) are malignant, against {fmt_int(cfg['cohort']['n_malignant'])} of "
            f"{fmt_int(cfg['cohort']['n_cells'])} cells ({pct_all:.1f}%) in the cohort. Within each patient: the predicted "
            f"types with ≥ {MIN_EACH_LABEL} cells of each label (at most {TYPES_PER_PATIENT}, most malignant cells "
            f"first), {CELLS_PER_LABEL} malignant and {CELLS_PER_LABEL} healthy cells each, then {N_COMPOSITION} cell of "
            "the most abundant other healthy type"
            + ("; last row per patient: one recipient-origin healthy cell, where one exists. " if extended else ". ")
            + f"Every malignant cell is recipient-derived and {100 * (n_h - rec_h) / n_h:.1f}% of healthy cells are "
            "donor-derived, so malignant versus healthy is almost recipient versus donor; patients are nested in "
            "batches. PC8 (and with it IFI27 and the poles) was chosen and oriented by the supervised probe. "
            "Descriptive; transductive Stage 1; nothing here is evidence of transfer across patients or of a malignant "
            f"program. Source: {ANALYSIS_ID} ({SLUG}).")
    footer(c, H - 0.5, note, W - 0.6)
    vals["n_rows"] = n_rows
    vals["n_malignant_rows"] = n_mal_rows
    c.fig.savefig(out_pdf, dpi=300)
    c.fig.savefig(out_png, dpi=300)
    plt.close(c.fig)
    vals["width_in"], vals["height_in"] = W, H
    return vals


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------

class Paths:
    def __init__(self, exp: Path, out_root: Optional[Path]):
        self.exp = exp
        self.out_root = out_root
        if out_root is None:
            self.tab = exp / "analysis" / "scorecards" / "stage2_diagnostic" / SLUG
            self.fig = exp / "analysis" / "figures" / "stage2_diagnostic" / SLUG
        else:
            self.tab = out_root / "scorecards" / "stage2_diagnostic" / SLUG
            self.fig = out_root / "figures" / "stage2_diagnostic" / SLUG

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.exp))
        except ValueError:
            return str(p)

    def outputs(self, versions: Sequence[str]) -> List[Path]:
        out = [self.tab / n for n in OUTPUT_TABLES]
        out += [self.tab / f"{SLUG}_findings.md", self.tab / f"{SLUG}_manifest.json"]
        for v in versions:
            out += [self.fig / f"{FIGURES[v]}.pdf", self.fig / f"{FIGURES[v]}.png"]
        return out


def complete(paths: Paths, versions: Sequence[str]) -> bool:
    for p in paths.outputs(versions) + [paths.tab / f"{SLUG}_artifact_ledger.csv"]:
        if not p.is_file() or p.stat().st_size == 0:
            return False
        if p.suffix == ".csv":
            try:
                pd.read_csv(p)
            except Exception:  # noqa: BLE001
                return False
        if p.suffix == ".json":
            try:
                json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                return False
    return True


def feature_table(data: dict) -> pd.DataFrame:
    genes, order = data["genes"], data["gene_order"]
    pole = dict(zip(data["poles"]["gene"], data["poles"]["pole"]))
    rows = []
    for k, j in enumerate(order):
        g = genes[j]
        rows.append({"block": "X~ genes", "feature": g, "display_order": k + 1, "panel_index": j + 1,
                     "pc8_loading_raw": float(data["pca"]["loadings"][j, PC8_COL]),
                     "pc8_loading_oriented": float(data["pc8_or"][j]), "pc8_pole": pole.get(g, ""),
                     "labelled": g in LABELLED_GENES, "highlighted": g == HIGHLIGHT_GENE})
    for k in range(40):
        rows.append({"block": "Z PCA K40", "feature": f"PC{k + 1}", "display_order": k + 1, "panel_index": k + 1,
                     "variance_pct": 100 * float(data["evr"][k]),
                     "orientation": "x(-1) (D18)" if k == PC8_COL else "raw (sign arbitrary)",
                     "highlighted": k == PC8_COL,
                     "projection_r": float(data["proj_r"][k])})
    for k in range(40):
        rows.append({"block": "Z' FA K40 (extended)", "feature": f"FA{k + 1}", "display_order": k + 1,
                     "panel_index": k + 1, "determinacy": float(data["det"][k]) if not data["degenerate"][k] else np.nan,
                     "degenerate": bool(data["degenerate"][k]),
                     "orientation": "+1 (D18)" if k == FA7_COL else "raw (sign arbitrary)", "highlighted": k == FA7_COL})
    for k, s in enumerate(data["prot_labels"]):
        rows.append({"block": "P protein CLR (extended)", "feature": s, "display_order": k + 1,
                     "is_isotype": s.startswith("isotype"), "highlighted": s == LINK_PROTEIN})
    return pd.DataFrame(rows)


def findings_md(data: dict, sel: pd.DataFrame, checks: pd.DataFrame, vals: Dict[str, dict], paths: Paths,
                cfg: dict) -> str:
    core = sel[sel["in_core"]]
    by_p = []
    for p in PATIENTS:
        s = sel[sel["patient"] == p]
        types = list(dict.fromkeys(s.loc[s["stratum"] == "type_pair", "block_type"]))
        comp = list(s.loc[s["stratum"] == "composition", "block_type"])
        extra = list(s.loc[s["stratum"] == "recipient_healthy", "cell_type"])
        by_p.append(f"| {p} | {cfg['patients'][p]['malignant']} / {fmt_int(cfg['patients'][p]['total'])} | "
                    f"{', '.join(types)} | {', '.join(comp)} | {', '.join(extra) if extra else 'none'} | "
                    f"{int(s['in_core'].sum())} |")
    r = data["proj_r"]
    vc = vals.get("core", {})
    ve = vals.get("extended", {})
    lines = [
        f"# {SLUG} findings ({ANALYSIS_ID}, W5.11)",
        "",
        "The feature-space explainer figure: the same cells as rows of the Stage 1 input and of the stored Stage 1 "
        "scores, with their annotations unchanged. Asks A-20261011-14, -15, -16 "
        "(`research/asks/2026-10-11_feature-space-figure.md`). Explanatory, not a performance figure; descriptive.",
        "",
        "## Data and model",
        "",
        f"- Cells: the cohort cache (`{cfg['cohort_cache_relpath']}`), {fmt_int(len(data['meta']))} cells x "
        f"{fmt_int(cfg['cohort']['n_genes'])} genes after normalize_total (target 1e4) and log1p, the Stage 0 "
        "preprocessing. Row order = `cell_metadata.csv.gz` cell_id = the rows of every Stage 1 `scores.npy` "
        "(the Stage 1 directories store no cell IDs).",
        "- Annotations: `patient`; `CN.label` (cancer -> malignant, normal -> healthy); `predicted.annotation` "
        "(reference-mapped; for a malignant cell its nearest healthy state); `source` (donor / recipient); `batch` "
        "(patients are nested in batches). Timepoint columns are not shown: `timepoint_type`, `Tech` and `marker` are "
        "constant, and `sample`, `Time` and `Days from SCT` vary only within P09.",
        f"- Gene space X~: the Hallmark IFN-alpha panel ({len(data['genes'])} genes, "
        f"`preprocessing/panels/stage0_panel_genes/{PANEL}.json`), each gene z-scored over all cells "
        "(`feature_zscore_with_zero_variance_to_zero`): the Stage 1 input itself.",
        f"- Factor space Z: `stage1_dr/panel_id={PANEL}/method=pca/k=40/seed=42/scores.npy` as stored "
        "(randomized-SVD U.S; `stage1_scope = across_patient`): one basis fit once on all cells of the 13 patients "
        "without labels, so shared across patients (stacking patients' rows is legitimate) and transductive.",
        "- Extended only: Z' = the stored FA K=40 scores (same directory layout, method=fa); P = per-cell CLR of the raw "
        "ADT counts (cohort cache `obsm['ADT']`, as D27; no dsb), "
        f"{', '.join(data['prot_labels'])}.",
        "",
        "## Alignment and regression checks (all passed)",
        "",
        f"- r(stored PCA score, X~ @ loadings) over the 40 factors: min {r.min():.4f}, median {np.median(r):.4f}, PC8 "
        f"{r[PC8_COL]:.4f} (the randomized-SVD convention, W2.7; threshold {MIN_PROJECTION_R}).",
        "- The full list is `regression_checks.csv` "
        f"({len(checks)} checks): row order, cohort counts and the label x origin crosstab against study_config, "
        "D18 orientation signs (PC8 -1, FA7 +1), the D18 mean oriented PC8 z of every type x label, the PC8-FA7 Tucker "
        "phi and FA7 determinacy recomputed from the stored loadings, and the Figure 20 rho values (asserted against D27 "
        "by make_figures_v3).",
        "",
        "## Rows (seed 42)",
        "",
        f"Rule: for each of {', '.join(PATIENTS)}, the patient x predicted type groups with >= {MIN_EACH_LABEL} cells "
        f"of each label (the D18 comparison groups), at most {TYPES_PER_PATIENT} with the most malignant cells; "
        f"{CELLS_PER_LABEL} malignant and {CELLS_PER_LABEL} healthy cells each (sorted by cell_id inside a stratum); "
        f"then {N_COMPOSITION} healthy cell of the most abundant healthy type not shown. The highlighted cell "
        f"`{data['hl_id']}` is example cell {HIGHLIGHT_EXAMPLE_CELL} of full report Figure 20B (P03, malignant, "
        "HSC/MPP2), forced into its stratum. The extended version appends one recipient-origin healthy cell per "
        "patient where one exists (drawn after every core row, so the core rows do not depend on it).",
        "",
        "| Patient | malignant / cells | shown types (malignant + healthy) | composition type | recipient healthy (ext.) "
        "| core rows |",
        "|---|---|---|---|---|---|",
        *by_p,
        "",
        f"Core: {len(core)} rows, {int((core['label'] == 'malignant').sum())} malignant; extended: {len(sel)} rows. The "
        "malignant share of the rows is a design choice; the patient bands give the true counts and prevalence.",
        "",
        "## Displayed features",
        "",
        f"- Genes: all {len(data['genes'])}, in a label-free order (average-linkage clustering of 1 - r, r = gene-gene "
        "correlation over all cells, optimal leaf order). Strip above X~: each gene's PC8 loading x(-1); ticks: the "
        f"20 pole genes of Figure 5a; named: {', '.join(LABELLED_GENES)}. Highlighted gene: {HIGHLIGHT_GENE}, PC8's "
        "largest |loading|.",
        f"- Factors: all 40 PCA factors in variance order; bars above Z: % of panel variance (PC8 "
        f"{vc.get('pc8_variance_pct', float('nan')):.2f}%; PC7 is {vc.get('pc7_over_pc8_minus_1_pct', float('nan')):.1f}% "
        f"above it; the 40 together {vc.get('pca40_total_variance_pct', float('nan')):.1f}%).",
        "",
        "## Scaling (display only unless stated)",
        "",
        "- X~ is the Stage 1 input as it is; colour clipped at +-3. Most negative values are zero counts after "
        "z-scoring.",
        "- Z: each factor z-scored over all cells (display only, the report's cohort z); PC8 x(-1) (D18 orientation: "
        "the sign of its coefficient in the L1 probe at C = 1e-4), other factors keep the raw stored sign (arbitrary). "
        "Z' likewise (FA7 orientation +1; degenerate factors hatched). P: each protein's CLR z-scored over all cells.",
        "",
        "## Numbers printed in the figures",
        "",
        "`panel_values.csv` lists them by version, e.g. the mean IFI27 z per patient and label over all of each "
        "patient's cells (panel B)"
        + (f"; the extended panels: rho(PC8, CD99) over all malignant cells {ve.get('rho_pc8_cd99_malignant_pooled', 0):+.3f}, "
           f"PC8-FA7 Tucker phi {ve.get('pc8_fa7_tucker_phi', 0):.3f}, FA7 determinacy {ve.get('fa7_determinacy', 0):.3f}, "
           f"{ve.get('pc8_fa7_same_sign_pole_genes', 0)} of 20 pole genes with the same sign." if ve else "."),
        "",
        "## Limitations",
        "",
        "- Transductive basis: the displayed patients' cells helped fit Z (the inductive refit of D17 is not drawn).",
        "- A stratified illustration sample: patterns in these rows are not evidence of transfer across patients, of a "
        "malignant program, or of differential expression.",
        "- Malignant versus healthy is almost recipient versus donor (every malignant cell is recipient-derived); "
        "patients are nested in batches; predicted types are transferred labels.",
        "- PC8, and with it IFI27 and the poles, was chosen and oriented by the supervised probe (labels); PC8 and FA7 "
        "are not individually identifiable (D18).",
        "- Protein: CLR without dsb; isotype references as in D27.",
        "",
        "## Files",
        "",
        *[f"- `{paths.rel(p)}`" for p in paths.outputs(list(vals.keys()))],
        f"- `{paths.rel(paths.tab / (SLUG + '_artifact_ledger.csv'))}`",
        "",
    ]
    return "\n".join(lines)


def write_ledger(paths: Paths, versions: Sequence[str]) -> Path:
    ledger = paths.tab / f"{SLUG}_artifact_ledger.csv"
    rows = []
    for p in paths.outputs(versions):
        kind = "figure" if p.suffix in (".pdf", ".png") else ("table" if p.suffix == ".csv" else "doc")
        rows.append({"path": paths.rel(p), "kind": kind, "size_bytes": p.stat().st_size})
    pd.DataFrame(rows).to_csv(ledger, index=False)
    return ledger


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", choices=("core", "extended", "both"), default="both")
    ap.add_argument("--resume", action="store_true", help="skip if every output exists and reads back")
    ap.add_argument("--out-root", type=Path, default=None, help="write under this scratch root; EXP is only read")
    args = ap.parse_args(argv)
    t0 = time.time()
    versions = ["core", "extended"] if args.version == "both" else [args.version]
    exp = experiment_dir()
    cfg = study_config()
    paths = Paths(exp, args.out_root)
    if paths.tab.is_dir() and any(paths.tab.iterdir()):
        if args.resume and complete(paths, versions):
            print(f"[{SLUG}] complete; nothing to do ({paths.rel(paths.tab)})")
            return 0
        if not args.resume or (paths.tab / f"{SLUG}_manifest.json").is_file():
            raise SystemExit(f"{paths.tab} already holds outputs: a new slug (_v2) is needed (R3), or --resume")
    paths.tab.mkdir(parents=True, exist_ok=True)
    paths.fig.mkdir(parents=True, exist_ok=True)

    checks = Checks()
    data = load_all(exp, cfg, checks)
    data["gene_order"] = label_free_gene_order(data["xz"])
    sel = select_cells(data["meta"], PATIENTS, data["hl_id"])
    sel["row_core"] = np.where(sel["in_core"], sel["in_core"].cumsum(), 0)
    sel["row_extended"] = np.arange(1, len(sel) + 1)
    sel["pc8_oriented_z"] = data["pc8_z"][sel["idx"].to_numpy()]
    cols = ["row_core", "row_extended", "cell_id", "idx", "patient", "label", "cn_label", "cell_type", "source",
            "batch", "sample", "stratum", "block_type", "in_core", "highlighted", "pc8_oriented_z"]
    sel = sel[cols]
    sel.to_csv(paths.tab / "selected_cells.csv", index=False)
    back = pd.read_csv(paths.tab / "selected_cells.csv", keep_default_na=False)
    back["in_core"] = back["in_core"].astype(str).eq("True")
    back["highlighted"] = back["highlighted"].astype(str).eq("True")
    same = all((back[k].astype(str).to_numpy() == sel[k].astype(str).to_numpy()).all()
               for k in ("cell_id", "patient", "label", "cell_type", "source", "batch", "stratum"))
    checks.add("selected_cells.csv round trip (labels drawn from the re-read file)", same, True)
    m = data["meta"].set_index("cell_id").loc[back["cell_id"]]
    checks.add("selected rows: annotations equal the cohort cache",
               bool((m["patient"].to_numpy() == back["patient"].to_numpy()).all()
                    and (m["label"].to_numpy() == back["label"].to_numpy()).all()
                    and (m["cell_type"].to_numpy() == back["cell_type"].to_numpy()).all()
                    and (m["source"].to_numpy() == back["source"].to_numpy()).all()), True)
    checks.add("selected rows: idx indexes the same cells in X~, Z and Z'",
               bool((m["idx"].to_numpy() == back["idx"].to_numpy()).all()), True)
    checks.add("highlight cell in its stratum and in the core rows",
               bool(back.loc[back["highlighted"], "in_core"].all() and back["highlighted"].sum() == 1), True)
    checks.add("recipient-healthy extras are recipient and healthy",
               bool(((back["stratum"] != "recipient_healthy")
                     | ((back["source"] == "recipient") & (back["label"] == "healthy"))).all()), True)

    vals: Dict[str, dict] = {}
    for v in versions:
        vals[v] = draw(data, back, cfg, v, paths.fig / f"{FIGURES[v]}.pdf", paths.fig / f"{FIGURES[v]}.png")
    feature_table(data).to_csv(paths.tab / "displayed_features.csv", index=False)
    pv = [{"version": v, "name": k, "value": val} for v, d in vals.items() for k, val in d.items()]
    pd.DataFrame(pv).to_csv(paths.tab / "panel_values.csv", index=False)
    ck = checks.frame()
    ck.to_csv(paths.tab / "regression_checks.csv", index=False)
    (paths.tab / f"{SLUG}_findings.md").write_text(findings_md(data, back, ck, vals, paths, cfg))
    head, dirty = git_state()
    manifest = {
        "slug": SLUG, "analysis_id": ANALYSIS_ID, "created_by": SCRIPT_REL,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_state": {"head": head, "dirty": dirty},
        "inputs": {
            "cohort_cache": cfg["cohort_cache_relpath"], "cell_metadata": cfg["cell_metadata_relpath"],
            "panel_genes": f"preprocessing/panels/stage0_panel_genes/{PANEL}.json",
            "stage1": f"stage1_dr/panel_id={PANEL}/method={{pca,fa}}/k=40/seed=42/",
            "d18": "analysis/scorecards/stage2_diagnostic/factor_cards_v1",
            "d27": "analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1",
        },
        "parameters": {
            "seed": SEED, "patients": list(PATIENTS), "min_each_label": MIN_EACH_LABEL,
            "types_per_patient": TYPES_PER_PATIENT, "cells_per_label": CELLS_PER_LABEL,
            "n_composition": N_COMPOSITION, "highlight": f"Figure 20B example cell {HIGHLIGHT_EXAMPLE_CELL}",
            "highlight_gene": HIGHLIGHT_GENE, "gene_order": "average-linkage clustering of 1 - r over all cells",
            "factor_display": "per-factor z over all cells; PC8 x(-1), FA7 +1 (D18)", "clip": CLIP,
            "proteins": data["prot_labels"], "versions": versions, "c_policy": "not_applicable",
        },
        "out_root": None if args.out_root is None else str(args.out_root),
        "regression_checks_passed": bool(ck["passed"].all()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (paths.tab / f"{SLUG}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_ledger(paths, versions)
    print(f"[{SLUG}] {len(ck)} checks passed; rows core {int(back['in_core'].sum())}, extended {len(back)}; "
          f"{round(time.time() - t0, 1)} s")
    for v in versions:
        print(f"  {paths.rel(paths.fig / (FIGURES[v] + '.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
