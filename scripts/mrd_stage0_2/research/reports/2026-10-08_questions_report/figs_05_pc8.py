"""Figures for section 05 (what PC8 is, and how to read a factor).

fig05a_pc8_poles     IFN-alpha PCA40 factor 8 (PC8): top 10 genes of each pole as oriented
                     loadings (positive = toward malignant, orientation = sign of the L1 probe
                     coefficient), with FA40 factor 7 (FA7) oriented loadings for the same genes.
fig05b_axis_lift     Univariate held-out lift of PC8 and FA7 per supported patient (the factor
                     alone, sign fixed on the training folds), plus a strip showing whether the
                     stored L1 probe (C = 1e-4) kept the factor in that LOPO fold.
fig05c_celltype      Mean oriented PC8 score by predicted cell type, normal vs malignant cells
                     (types with malignant rows in the D18 cell-type table). Version 3's Figure 19; since
                     version 3.1 (2026-10-10) main() no longer draws it: both reports show the relabelled
                     fig05g_celltype_counts (make_figures section 05p) instead.

Added for report v2 (2026-10-09 reading notes A-20261009-06, -07, -08):
fig05d_identifiability  What "not identifiable" means for PC8 and FA7. (a) PCA40 variance per component:
                     PC8 is within 10% of both neighbours. (b) FA40 factor-score determinacy: FA7 is below
                     0.707. (c) D17 refits of Stage 1 (leave-outs, patient-equal weights, equal-n subsamples,
                     normal cells only): the reference axis stays in the refit span (many-to-one R^2) while
                     its best single match (matched |r|) can drop.
fig05e_rho_anchors   D27: the report's PC8 x ADT numbers taken apart. PC8 x CD34 in normal cells (+0.251),
                     PC8 x CD99 and PC8 x CD48 in malignant cells: the rho inside each supported patient, the
                     median of the seven, the pooled (D18) value and the median inside patient x cell type
                     strata, against the card's isotype reference.
fig05f_modality_all  D27 over all 1,801 non-degenerate cards. (a, b) The strongest robust factor x ADT pair per
                     marker (oriented cards), normal and malignant cells. (c) How much of a pooled rho holds
                     within patients (|pooled| >= 0.20). (d, e) Cell-type eta2 within patient vs patient eta2.

Added for report v3.1 (2026-10-10; asks A-20261010-18, -21, -23, -24):
fig05l_protein_core  The core report's surface-protein page: (A) what the sign of one Spearman rho means (the
                     1,569 malignant cells ranked on oriented PC8 and on CD99 or CD48; mean protein rank per
                     tenth of the PC8 ranking), (B, C) the two checks per protein of full report Figure 22.
                     Drawn with the loaders and panel function of the Pavan update's make_figures_v3.py, which
                     also draws section 5's other explainer figures (fig05g-fig05k, make_figures section 05p).
                     Reads the cohort cache (obs and obsm['ADT'] only; never the raw AnnData) and the stored
                     IFN-alpha PCA40 scores; asserts every recomputed rho against D27.

Data (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_genes.csv
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_index.csv
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_patient_lift.csv
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_fold_coefs.csv
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_celltype.csv
  analysis/scorecards/stage2_diagnostic/factor_cards_v1/factor_cards_shortlist.csv
  analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv
  analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/d17_alignment.csv
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/ (D27: d27_ifna_pc8_fa7_adt.csv,
      d27_card_summary.csv, d27_adt_robust.csv, d27_adt_assoc_full.parquet, d27_celltype_eta2.csv,
      d27_strata.csv, d27_regression_checks.csv)
  stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method=fa/k=40/seed=42/loadings.npy
  stage1_dr/panel_id=single_geneset__hallmark_interferon_alpha_response/method=pca/k=40/seed=42/diagnostics.json
  preprocessing/panels/stage0_panel_genes/single_geneset__hallmark_interferon_alpha_response.json

Run from the repo root or the report dir:
  python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_05_pc8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import report_style as rs  # noqa: E402

PANEL = "single_geneset__hallmark_interferon_alpha_response"
PC8 = f"{PANEL}__pca__k40__f008"
FA7 = f"{PANEL}__fa__k40__f007"
N_PER_POLE = 10
# The two gene groups named in the team report (2026-10-08), by raw PC8 sign.
TEAM_GENES = {
    "IFI27", "PSME2", "PSMA3", "CNP", "LAP3", "USP18",
    "TXNIP", "IFITM2", "TENT5A", "NCOA7", "TRIM5", "ELF1",
}


def cards_dir() -> Path:
    return rs.diag_dir("factor_cards_v1")


def _read(name: str, ids=(PC8, FA7)) -> pd.DataFrame:
    df = pd.read_csv(cards_dir() / name)
    return df[df["card_id"].isin(ids)].copy()


def orientation(card_id: str) -> int:
    idx = _read("factor_cards_index.csv", ids=(card_id,))
    assert len(idx) == 1
    return int(idx["orientation_sign"].iloc[0])


def fa7_oriented_loadings() -> pd.Series:
    """Full FA40 factor-7 loadings (all 92 genes), oriented by the D18 orientation sign."""
    exp = rs.exp_dir()
    genes = json.loads((exp / "preprocessing" / "panels" / "stage0_panel_genes" / f"{PANEL}.json").read_text())["genes"]
    base = exp / "stage1_dr" / f"panel_id={PANEL}" / "method=fa" / "k=40" / "seed=42"
    load = np.load(base / "loadings.npy").astype(float)
    if load.shape[0] != len(genes):
        load = load.T
    s = pd.Series(load[:, 6], index=genes) * orientation(FA7)
    # Cross-check against the card's own top-gene table.
    card = _read("factor_cards_genes.csv", ids=(FA7,)).set_index("gene")["oriented_loading"]
    assert np.allclose(s.loc[card.index].values, card.values, atol=1e-6)
    return s


def pc8_pole_table() -> pd.DataFrame:
    g = _read("factor_cards_genes.csv", ids=(PC8,))
    g = g[(g["ranked_by"] == "loading") & (g["rank"] <= N_PER_POLE)]
    mal = g[g["pole"] == "malignant"].sort_values("oriented_loading", ascending=False)
    nor = g[g["pole"] == "normal"].sort_values("oriented_loading", ascending=False)
    assert len(mal) == N_PER_POLE and len(nor) == N_PER_POLE
    out = pd.concat([mal, nor], ignore_index=True)[["gene", "pole", "loading", "oriented_loading"]]
    fa7 = fa7_oriented_loadings()
    out["fa7_oriented"] = fa7.loc[out["gene"]].values
    out["same_sign"] = np.sign(out["oriented_loading"]) == np.sign(out["fa7_oriented"])
    return out


def fig05a_pc8_poles() -> pd.DataFrame:
    tab = pc8_pole_table()
    n_same = int(tab["same_sign"].sum())
    gap = 0.7  # visual gap between the two poles
    y = np.array([i if i < N_PER_POLE else i + gap for i in range(len(tab))], dtype=float)

    fig, ax = plt.subplots(figsize=(rs.FULL_W, 2.85))
    ax.barh(y, tab["oriented_loading"], height=0.58, color=rs.METHOD["pca"], linewidth=0, zorder=2,
            label="PC8 (PCA, K = 40) weight")
    ax.plot(tab["fa7_oriented"], y, linestyle="none", marker="o", markersize=rs.MARKER,
            markerfacecolor=rs.METHOD["fa"], markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING,
            zorder=3, label="FA7 (FA, K = 40) loading")
    ax.axvline(0, color=rs.INK_2, lw=0.8, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(tab["gene"])
    for lab, gene in zip(ax.get_yticklabels(), tab["gene"]):
        if gene in TEAM_GENES:
            lab.set_fontweight("bold")
            lab.set_color(rs.INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-0.36, 0.36)
    ax.set_xticks(np.arange(-0.3, 0.31, 0.1))
    ax.set_xticklabels([f"{v:+.1f}".replace("-", "\u2212") if abs(v) > 1e-9 else "0"
                        for v in np.arange(-0.3, 0.31, 0.1)])
    ax.set_xlabel("Oriented loading (unitless; positive = pushes the score toward malignant)")
    ax.tick_params(axis="y", length=0)

    # Pole brackets left of the gene names (the bars themselves show the direction: right = malignant).
    tr = ax.get_yaxis_transform()
    xb = -0.135
    for rows, text in ((y[:N_PER_POLE], "malignant-\nleaning pole"), (y[N_PER_POLE:], "normal-\nleaning pole")):
        lo, hi = rows.min() - 0.35, rows.max() + 0.35
        ax.plot([xb + 0.012, xb, xb, xb + 0.012], [lo, lo, hi, hi], color=rs.INK_2, lw=0.9, transform=tr,
                clip_on=False, solid_capstyle="butt")
        ax.text(xb - 0.012, (lo + hi) / 2, text, rotation=90, ha="right", va="center", fontsize=7.5,
                color=rs.INK, transform=tr, clip_on=False, linespacing=1.05)

    ax.set_title(f"PC8 contrasts two IFN-\u03b1 gene groups; FA7 has the same sign on {n_same} of these {len(tab)} genes")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in sorted(labels, key=lambda t: not t.startswith("PC8"))]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    handles.append(Line2D([], [], linestyle="none"))
    labels.append("bold = named in the team report")
    ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, borderaxespad=0.3,
              handlelength=1.4, fontsize=7.2)
    rs.save(fig, "fig05a_pc8_poles")
    return tab


def axis_lift_table() -> pd.DataFrame:
    lift = _read("factor_cards_patient_lift.csv")
    coefs = _read("factor_cards_fold_coefs.csv")
    coefs = coefs.rename(columns={"heldout_patient": "patient"})[["card_id", "patient", "coef", "is_nonzero", "fold_all_zero"]]
    out = lift.merge(coefs, on=["card_id", "patient"], how="left")
    out = out[out["patient"].isin(rs.SUPPORTED)]
    out["factor_name"] = np.where(out["card_id"] == PC8, "PC8", "FA7")
    out["univariate_lift_pp"] = 100 * out["univariate_lift"]
    out["fitted_single_factor_lift_pp"] = 100 * out["fitted_single_factor_lift"]
    assert len(out) == 14
    return out


def fig05b_axis_lift() -> pd.DataFrame:
    tab = axis_lift_table()
    pats = list(rs.SUPPORTED)
    x = np.arange(len(pats), dtype=float)
    w = 0.28
    series = [("PC8", "pca", -w / 2 - 0.015), ("FA7", "fa", w / 2 + 0.015)]

    fig, (ax, sx) = plt.subplots(2, 1, figsize=(rs.FULL_W, 2.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.0, 0.24], "hspace": 0.08})
    for name, method, dx in series:
        d = tab[tab["factor_name"] == name].set_index("patient").loc[pats]
        ax.bar(x + dx, d["univariate_lift_pp"], width=w, color=rs.METHOD[method], linewidth=0, zorder=2,
               label=f"{name} alone ({rs.METHOD_LABEL[method]}, K = 40)")
    ax.axhline(0, color=rs.INK_2, lw=0.8, zorder=1)
    ax.set_ylabel("Univariate held-out lift (pp)")
    ax.set_ylim(-1, 24.5)
    ymax = tab["univariate_lift_pp"].max()
    # Bracket over the highest-prevalence patients.
    i0, i1 = pats.index("P02"), pats.index("P04")
    yb = ymax + 2.0
    ax.plot([i0 - 0.35, i0 - 0.35, i1 + 0.35, i1 + 0.35], [yb - 0.8, yb, yb, yb - 0.8], color=rs.INK_2, lw=0.9,
            solid_capstyle="butt", clip_on=False)
    ax.text((i0 + i1) / 2, yb + 0.4, "highest prevalence", ha="center", va="bottom", fontsize=7.5, color=rs.INK_2)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], linestyle="none", marker="s", markersize=rs.MARKER - 1, markerfacecolor=rs.INK_2,
               markeredgecolor=rs.INK_2),
        Line2D([], [], linestyle="none", marker="s", markersize=rs.MARKER - 1, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.INK_2, markeredgewidth=1.2),
    ]
    labels += ["strip: the L1 probe keeps the factor in that fold", "strip: coefficient zero (whole probe all-zero)"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=7.2,
              handlelength=1.4, columnspacing=1.6, borderaxespad=0.2)
    ax.set_title("Alone, each factor gives 4.8\u201319.9 pp of held-out lift\n"
                 "in P02, P03, P04, P07 and P09, and under 1 pp in P01 and P06", pad=34)

    # Selection strip: did the stored L1 probe keep the factor in this LOPO fold?
    rows = [("PC8", "pca", 1.0), ("FA7", "fa", 0.0)]
    for name, method, yy in rows:
        d = tab[tab["factor_name"] == name].set_index("patient").loc[pats]
        on = d["is_nonzero"].astype(bool).values
        col = rs.METHOD[method]
        sx.plot(x[on], np.full(on.sum(), yy), linestyle="none", marker="s", markersize=rs.MARKER,
                markerfacecolor=col, markeredgecolor=col, markeredgewidth=1.0)
        sx.plot(x[~on], np.full((~on).sum(), yy), linestyle="none", marker="s", markersize=rs.MARKER,
                markerfacecolor=rs.SURFACE, markeredgecolor=col, markeredgewidth=1.2)
    sx.set_ylim(-0.6, 1.6)
    sx.set_yticks([1.0, 0.0])
    sx.set_yticklabels(["PC8 in PCA40-L1", "FA7 in FA40-L1"], fontsize=7.2)
    sx.tick_params(axis="y", length=0)
    sx.grid(False)
    for side in ("left", "bottom"):
        sx.spines[side].set_visible(False)
    sx.tick_params(axis="x", length=0)

    meta = tab[tab["factor_name"] == "PC8"].set_index("patient").loc[pats]
    labels = [f"{p}\n{int(n)} mal.\n\u03c0 {100 * pi:.1f}%" for p, n, pi in
              zip(pats, meta["n_malignant"], meta["prevalence"])]
    sx.set_xticks(x)
    sx.set_xticklabels(labels, fontsize=7.2)
    sx.set_xlim(-0.6, len(pats) - 0.4)
    rs.save(fig, "fig05b_axis_lift")
    return tab


def celltype_table() -> pd.DataFrame:
    ct = _read("factor_cards_celltype.csv", ids=(PC8,))
    wide = ct.pivot_table(index="cell_type", columns="label", values=["mean_oriented_z", "n_cells"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["mean_oriented_z_malignant", "mean_oriented_z_normal"]).reset_index()
    wide["n_cells_malignant"] = wide["n_cells_malignant"].astype(int)
    wide["n_cells_normal"] = wide["n_cells_normal"].astype(int)
    wide["malignant_higher"] = wide["mean_oriented_z_malignant"] > wide["mean_oriented_z_normal"]
    return wide.sort_values("n_cells_malignant", ascending=False).reset_index(drop=True)


def fig05c_celltype() -> pd.DataFrame:
    tab = celltype_table()
    n_hi = int(tab["malignant_higher"].sum())
    y = np.arange(len(tab), dtype=float)
    fig, ax = plt.subplots(figsize=(rs.HALF_W, 2.75))
    for yi, (_, r) in zip(y, tab.iterrows()):
        ax.plot([r["mean_oriented_z_normal"], r["mean_oriented_z_malignant"]], [yi, yi], color=rs.AXIS, lw=rs.LINE_W,
                zorder=1, solid_capstyle="butt")
    ax.plot(tab["mean_oriented_z_normal"], y, linestyle="none", marker="o", markersize=rs.MARKER,
            markerfacecolor=rs.SURFACE, markeredgecolor=rs.MUTED, markeredgewidth=1.4, zorder=3, label="normal cells")
    ax.plot(tab["mean_oriented_z_malignant"], y, linestyle="none", marker="o", markersize=rs.MARKER,
            markerfacecolor=rs.INK_2, markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, zorder=4,
            label="malignant cells")
    ax.axvline(0, color=rs.INK_2, lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({n})" for c, n in zip(tab["cell_type"], tab["n_cells_malignant"])], fontsize=7.2)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-0.45, 1.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "+0.5", "+1.0"])
    ax.set_xlabel("Mean oriented PC8 score (cohort z, SD)")
    fig.subplots_adjust(left=0.31, right=0.97)
    ax.set_title(f"Malignant cells score higher\nin {n_hi} of {len(tab)} predicted cell types", pad=20)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, handletextpad=0.3,
              borderaxespad=0.1, columnspacing=1.2, fontsize=7.2)
    rs.save(fig, "fig05c_celltype")
    return tab


# ---------------------------------------------------------------------------------------------
# Report v2 (2026-10-09): identifiability, the 1,810 cards, and D27 factor x modality associations
# ---------------------------------------------------------------------------------------------

D17_SLUG = "basis_variant_pilot_ifna_v1"
D27_SLUG = "factor_modality_assoc_v1"
LABELS = ("normal", "malignant")
DETERMINACY_BAR = 0.707  # rho^2 = 0.5
EIGENGAP_BAR = 0.10
ROBUST_ABS = 0.20  # D27 rule, fixed before computing: |median within-patient rho| >= 0.20 ...
ROBUST_SAME_SIGN = 5  # ... and the same sign in >= 5 of the 7 supported patients
PANEL_SHORT = {
    "mhc class i antigen presentation": "MHC class I antigen pres.",
    "mhc class ii antigen presentation": "MHC class II antigen pres.",
    "all Stage 0 genesets": "all Stage 0 gene sets",
    "interferon alpha response": "IFN-\u03b1 response",
    "tnfa signaling via nfkb": "TNF-\u03b1 via NF-kB",
}
METHOD_SHORT = {"pca": "PCA", "fa": "FA", "factosig": "FactoSig", "factosig_promax": "promax"}


def _minus(s: str) -> str:
    return s.replace("-", "\u2212")


def _fmt_signed(v: float, nd: int = 2) -> str:
    return _minus(f"{v:+.{nd}f}")


def d27_dir() -> Path:
    return rs.diag_dir(D27_SLUG)


# ---- the 1,810 cards ------------------------------------------------------------------------

def card_accounting() -> dict:
    """Where the 1,810 D18 cards come from: representations, shortlist reasons, K, degenerate factors."""
    idx = pd.read_csv(cards_dir() / "factor_cards_index.csv")
    sl = pd.read_csv(cards_dir() / "factor_cards_shortlist.csv")
    rep = (idx.groupby("representation_id")
           .agg(n_cards=("card_id", "size"), k=("k", "first"), effective_k=("effective_k", "first"),
                method=("method", "first"), n_degenerate=("degenerate_flag", "sum"))
           .reset_index()
           .merge(sl[["representation_id", "shortlist_reasons", "atlas_row_block", "panel_label"]],
                  on="representation_id", validate="one_to_one"))
    assert len(rep) == 56 and int(rep["n_cards"].sum()) == 1810
    assert (rep["n_cards"] == rep["k"]).all(), "one card per requested factor"
    assert int(rep["effective_k"].sum()) == 1801 == int((~idx["degenerate_flag"]).sum())

    def group(row) -> str:
        reasons = row["shortlist_reasons"]
        if "d14_screen" in reasons:
            return "screen_biological" if row["atlas_row_block"] == "biological" else "screen_control"
        if "focus_panel" in reasons:
            return "focus_pca40_fa40"
        if "F9c" in reasons:
            return "ifna_fa_cross_k"
        return "ifna_fa40_pair"

    rep["group"] = rep.apply(group, axis=1)
    by_group = rep.groupby("group").agg(n_rep=("n_cards", "size"), n_cards=("n_cards", "sum"),
                                        n_nondegenerate=("effective_k", "sum")).to_dict("index")
    atlas = pd.read_csv(rs.diag_dir("") / "old34_d1_atlas_representation_table.csv")
    dr = atlas[atlas["configuration"].ne("direct_gene")]
    out = {
        "rep": rep,
        "by_group": by_group,
        "by_method": rep.groupby("method")["n_cards"].sum().to_dict(),
        "by_method_rep": rep.groupby("method").size().to_dict(),
        "by_k": rep.groupby("k").agg(n_rep=("n_cards", "size"), n_cards=("n_cards", "sum")).to_dict("index"),
        "degenerate": rep.loc[rep["n_degenerate"] > 0, ["representation_id", "n_degenerate", "effective_k"]],
        "atlas_dr_cells": int(len(dr)),
        "atlas_dr_cells_biological": int((dr["atlas_row_block"] == "biological").sum()),
        "atlas_dr_factors": int(dr["effective_k"].sum()),
        "n_drivers": int(idx["is_selected_driver"].sum()),
        "n_dense": int(idx["fit_dense"].sum()),
        "orientation": idx["orientation_status"].value_counts().to_dict(),
    }
    return out


# ---- identifiability ------------------------------------------------------------------------

def pca40_spectrum() -> pd.DataFrame:
    diag = json.loads((rs.exp_dir() / "stage1_dr" / f"panel_id={PANEL}" / "method=pca" / "k=40" / "seed=42"
                       / "diagnostics.json").read_text())
    uns = diag["extras"]["uns"]
    ev = np.asarray(uns["explained_variance"], dtype=float)
    evr = np.asarray(uns["explained_variance_ratio"], dtype=float)
    gaps = np.full(ev.size, np.nan)
    for i in range(ev.size):
        nb = [abs(ev[i] - ev[j]) for j in (i - 1, i + 1) if 0 <= j < ev.size]
        gaps[i] = min(nb) / ev[i]
    tab = pd.DataFrame({"pc": np.arange(1, ev.size + 1), "explained_variance": ev, "pct": 100 * evr, "eigengap_rel": gaps})
    card = pd.read_csv(cards_dir() / "factor_cards_index.csv")
    card = card[(card["panel_id"] == PANEL) & (card["method"] == "pca") & (card["k"] == 40)].sort_values("factor")
    assert np.allclose(card["eigengap_rel"].values, tab["eigengap_rel"].values, atol=1e-6), "eigengap = D18 card"
    return tab


def fa40_determinacy() -> pd.DataFrame:
    card = pd.read_csv(cards_dir() / "factor_cards_index.csv")
    card = card[(card["panel_id"] == PANEL) & (card["method"] == "fa") & (card["k"] == 40)].sort_values("factor")
    return card[["factor", "determinacy", "degenerate_flag"]].reset_index(drop=True)


# Stage 1 perturbations of the IFN-alpha K = 40 basis (D17), in display order: (arm, variant, label).
PERTURBATIONS = [
    ("leave_out", "drop-P02", "P02 left out"),
    ("leave_out", "drop-P03", "P03 left out"),
    ("leave_out", "drop-P04", "P04 left out"),
    ("leave_out", "drop-P02-P03-P04", "P02+P03+P04 left out"),
    ("leave_out", "drop-P06", "P06 left out"),
    ("leave_out", "drop-P09", "P09 left out"),
    ("leave_out", "drop-P01", "P01 left out"),
    ("leave_out", "drop-P01-P09", "P01+P09 left out"),
    ("balanced_weighted", "all", "patient-equal weights"),
    ("balanced_subsample", None, "1,128 cells per patient\n(5 random draws)"),
    ("normal_only_all", "all", "normal cells only"),
]


def perturbation_table() -> pd.DataFrame:
    al = pd.read_csv(rs.diag_dir(D17_SLUG) / "d17_alignment.csv")
    al = al[al["stage1_method"].isin(["pca", "fa"]) & (al["effective_k"] == 40)]
    rows = []
    for order, (arm, variant, label) in enumerate(PERTURBATIONS):
        sel = al[al["arm"] == arm]
        if variant is not None:
            sel = sel[sel["variant"] == variant]
        for _, r in sel.iterrows():
            rows.append({"order": order, "label": label, "arm": arm, "variant": r["variant"],
                         "method": r["stage1_method"], "ref": r["ref_factor_label"],
                         "matched_abs_r": float(r["matched_abs_r_fit"]), "r2": float(r["many_to_one_r2_fit"]),
                         "axis_loss": float(r["axis_loss_fit"])})
    tab = pd.DataFrame(rows)
    n_expected = {"balanced_subsample": 5}
    for (arm, variant, _label) in PERTURBATIONS:
        for m in ("pca", "fa"):
            n = int(((tab["arm"] == arm) & (tab["method"] == m) &
                     ((tab["variant"] == variant) if variant is not None else True)).sum())
            assert n == n_expected.get(arm, 1), (arm, variant, m, n)
    assert set(tab.loc[tab["method"] == "pca", "ref"]) == {"PC8"} and set(tab.loc[tab["method"] == "fa", "ref"]) == {"FA7"}
    return tab


def fig05d_identifiability() -> dict:
    spec = pca40_spectrum()
    det = fa40_determinacy()
    pert = perturbation_table()

    fig = plt.figure(figsize=(rs.FULL_W, 3.75))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.18], height_ratios=[1, 1], wspace=0.62, hspace=0.95)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])

    # (a) PCA40 spectrum around PC8.
    show = spec[spec["pc"] <= 16]
    pc8 = float(spec.loc[spec["pc"] == 8, "pct"].iloc[0])
    ax_a.axhspan(0.9 * pc8, 1.1 * pc8, color=rs.GRID, zorder=0, lw=0)
    ax_a.plot(show["pc"], show["pct"], color=rs.AXIS, lw=1.0, zorder=1)
    nb = show["pc"].isin([7, 9])
    ax_a.plot(show.loc[~nb & (show["pc"] != 8), "pc"], show.loc[~nb & (show["pc"] != 8), "pct"], linestyle="none",
              marker="o", markersize=4.2, markerfacecolor=rs.SURFACE, markeredgecolor=rs.MUTED, markeredgewidth=1.0,
              zorder=2)
    ax_a.plot(show.loc[nb, "pc"], show.loc[nb, "pct"], linestyle="none", marker="o", markersize=5.2,
              markerfacecolor=rs.SURFACE, markeredgecolor=rs.METHOD["pca"], markeredgewidth=1.4, zorder=3)
    ax_a.plot([8], [pc8], linestyle="none", marker="o", markersize=6.2, markerfacecolor=rs.METHOD["pca"],
              markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, zorder=4)
    g7 = float(spec.loc[spec["pc"] == 8, "eigengap_rel"].iloc[0])
    gap9 = abs(spec.loc[spec["pc"] == 8, "explained_variance"].iloc[0] - spec.loc[spec["pc"] == 9, "explained_variance"].iloc[0]) \
        / spec.loc[spec["pc"] == 8, "explained_variance"].iloc[0]
    ax_a.annotate(f"PC8 {pc8:.2f}%\nPC7 is {100 * g7:.1f}% above,\nPC9 {100 * gap9:.1f}% below",
                  xy=(8, pc8), xytext=(9.0, 3.9), fontsize=7.0, color=rs.INK, ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.7, shrinkA=1, shrinkB=3))
    ax_a.text(16.3, 0.45, "gray band: PC8 \u00b1 10%", fontsize=6.6, color=rs.INK_2, ha="right", va="center")
    ax_a.set_xlim(0.3, 16.5)
    ax_a.set_xticks([1, 4, 8, 12, 16])
    ax_a.set_ylim(0, 5.8)
    ax_a.set_xlabel("PCA component (IFN-\u03b1, K = 40)")
    ax_a.set_ylabel("% of panel variance")
    ax_a.set_title("(a) PC8 is nearly tied with\nits neighbours", fontsize=8.6)

    # (b) FA40 determinacy.
    d = det[~det["degenerate_flag"]]
    fa7 = float(d.loc[d["factor"] == 7, "determinacy"].iloc[0])
    ax_b.axhline(DETERMINACY_BAR, color=rs.INK_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    other = d["factor"] != 7
    ax_b.plot(d.loc[other, "factor"], d.loc[other, "determinacy"], linestyle="none", marker="o", markersize=3.8,
              markerfacecolor=rs.SURFACE, markeredgecolor=rs.MUTED, markeredgewidth=1.0, zorder=2)
    ax_b.plot([7], [fa7], linestyle="none", marker="o", markersize=6.2, markerfacecolor=rs.METHOD["fa"],
              markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, zorder=3)
    n_below = int((d["determinacy"] < DETERMINACY_BAR).sum())
    ax_b.annotate(f"FA7 {fa7:.3f}", xy=(7, fa7), xytext=(13.0, 0.60), fontsize=7.0, color=rs.INK, ha="left",
                  va="center", arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.7, shrinkA=1, shrinkB=3))
    ax_b.text(36.5, DETERMINACY_BAR + 0.03, f"0.707 line: {n_below} of {len(d)} below", fontsize=6.6,
              color=rs.INK_2, ha="right", va="bottom")
    ax_b.set_xlim(0, 37.5)
    ax_b.set_xticks([1, 7, 12, 18, 24, 30, 36])
    ax_b.set_ylim(0, 1.02)
    ax_b.set_xlabel(f"FA factor (IFN-\u03b1, K = 40; {len(d)} non-degenerate)")
    ax_b.set_ylabel("Determinacy \u03c1")
    ax_b.set_title("(b) FA7's scores are estimates", fontsize=8.6)

    # (c) Perturbation dot plot.
    labels = [p[2] for p in PERTURBATIONS]
    ys = np.arange(len(labels), dtype=float)
    for m, dy in (("pca", -0.17), ("fa", 0.17)):
        col = rs.METHOD[m]
        sub = pert[pert["method"] == m]
        for _, r in sub.iterrows():
            y = r["order"] + dy
            ax_c.plot([r["matched_abs_r"], r["r2"]], [y, y], color=col, lw=0.8, alpha=0.45, zorder=1,
                      solid_capstyle="butt")
            ax_c.plot([r["r2"]], [y], linestyle="none", marker="o", markersize=4.4, markerfacecolor=rs.SURFACE,
                      markeredgecolor=col, markeredgewidth=1.2, zorder=2)
            ax_c.plot([r["matched_abs_r"]], [y], linestyle="none", marker="o", markersize=4.8, markerfacecolor=col,
                      markeredgecolor=rs.SURFACE, markeredgewidth=0.6, zorder=3)
    ax_c.set_yticks(ys)
    ax_c.set_yticklabels(labels, fontsize=7.0, linespacing=1.0)
    ax_c.invert_yaxis()
    ax_c.tick_params(axis="y", length=0)
    ax_c.grid(axis="y", visible=False)
    ax_c.grid(axis="x", visible=True)
    ax_c.set_xlim(0.55, 1.005)
    ax_c.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_c.set_xlabel("Agreement with the stored axis")
    # Separators between groups of perturbations.
    for yb in (3.5, 7.5):
        ax_c.axhline(yb, color=rs.GRID, lw=0.8, zorder=0)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4.8, markerfacecolor=rs.METHOD["pca"],
               markeredgecolor=rs.SURFACE, label="PC8: best single match, |r|"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.4, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.METHOD["pca"], markeredgewidth=1.2, label="PC8: whole refit basis, R\u00b2"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.8, markerfacecolor=rs.METHOD["fa"],
               markeredgecolor=rs.SURFACE, label="FA7: best single match, |r|"),
        Line2D([], [], linestyle="none", marker="o", markersize=4.4, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.METHOD["fa"], markeredgewidth=1.2, label="FA7: whole refit basis, R\u00b2"),
    ]
    ax_c.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=6.6,
                handletextpad=0.2, columnspacing=0.8, borderaxespad=0.15)
    ax_c.set_title("(c) Refit Stage 1 differently: the axis stays in\nthe basis (R\u00b2), its single factor can rotate (|r|)",
                   fontsize=8.6, pad=27)
    rs.save(fig, "fig05d_identifiability")
    n_tied = int((spec["eigengap_rel"] <= EIGENGAP_BAR).sum())
    return {"spec": spec, "det": det, "pert": pert, "n_pcs_tied": n_tied, "fa_n_below": n_below,
            "fa_n_nondeg": int(len(d))}


# ---- D27: rho anchors -----------------------------------------------------------------------

# (card, label, marker, panel title). Titles are short (PC8 x ... is in the caption): with "PC8 x" the titles of
# (b) and (c) ran together (v3.1 review).
ANCHORS = [
    (PC8, "normal", "CD34", "(a) CD34, normal"),
    (PC8, "malignant", "CD99", "(b) CD99, malignant"),
    (PC8, "malignant", "CD48", "(c) CD48, malignant"),
]


def d27_patient_counts() -> pd.DataFrame:
    st = pd.read_csv(d27_dir() / "d27_strata.csv")
    st = st[(st["kind"] == "patient") & st["supported_patient"].astype(bool)]
    return st.pivot_table(index="patient", columns="label", values="n_cells").astype(int)


def anchor_table() -> pd.DataFrame:
    a = pd.read_csv(d27_dir() / "d27_ifna_pc8_fa7_adt.csv")
    cs = pd.read_csv(d27_dir() / "d27_card_summary.csv").set_index("card_id")
    rows = []
    for card, label, marker, title in ANCHORS:
        r = a[(a["card_id"] == card) & (a["label"] == label) & (a["adt_marker_short"] == marker)]
        assert len(r) == 1
        r = r.iloc[0]
        rows.append({
            "card_id": card, "label": label, "marker": marker, "title": title,
            "rho_pooled": float(r["rho_pooled"]), "rho_within_median": float(r["rho_within_median"]),
            "n_same_sign": int(r["n_patients_same_sign"]), "rho_pct_median": float(r["rho_pct_median"]),
            "n_pct_strata": int(r["n_pct_strata"]),
            "isotype_ref": float(cs.loc[card, f"isotype_max_abs_rho_within_median_{label}"]),
            **{f"rho_{p}": float(r[f"rho_{p}"]) for p in rs.SUPPORTED},
        })
    tab = pd.DataFrame(rows)
    # The D18 anchors quoted in report v1 (and the D27 regression check that reproduces them).
    pooled = dict(zip(zip(tab["label"], tab["marker"]), tab["rho_pooled"]))
    assert round(pooled[("normal", "CD34")], 3) == 0.251 and round(pooled[("malignant", "CD99")], 3) == 0.222
    reg = pd.read_csv(d27_dir() / "d27_regression_checks.csv")
    assert bool(reg["passed"].all()) and len(reg) == 10
    return tab


def _is_robust(median: float, n_same: int) -> bool:
    return abs(median) >= ROBUST_ABS and n_same >= ROBUST_SAME_SIGN


def fig05e_rho_anchors() -> pd.DataFrame:
    tab = anchor_table()
    counts = d27_patient_counts()
    pats = list(rs.SUPPORTED)
    summary_rows = ["median of the 7", "pooled: all cells", "inside cell types"]
    ys_p = np.arange(len(pats), dtype=float)
    ys_s = len(pats) + 0.6 + np.arange(len(summary_rows), dtype=float)
    xlim = (-0.8, 0.8)
    ticks = (-0.5, 0.0, 0.5)  # three ticks: five ran together at the panel width

    fig = plt.figure(figsize=(rs.FULL_W, 3.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08, left=0.135, right=0.995, top=0.74, bottom=0.15)
    axes = [fig.add_subplot(gs[0, 0])]
    # (a) normal cells has its own y labels (normal-cell counts); (b) and (c) share the malignant labels.
    gs_bc = gs[0, 1:].subgridspec(1, 2, wspace=0.16)
    axes.append(fig.add_subplot(gs_bc[0, 0]))
    axes.append(fig.add_subplot(gs_bc[0, 1], sharey=axes[1]))
    for i, (ax, (_, r)) in enumerate(zip(axes, tab.iterrows())):
        ref = r["isotype_ref"]
        ax.axvspan(-ref, ref, color=rs.GRID, lw=0, zorder=0)
        ax.axvline(0, color=rs.INK_2, lw=0.8, zorder=1)
        for xv in (-ROBUST_ABS, ROBUST_ABS):
            ax.axvline(xv, color=rs.MUTED, lw=0.7, ls=(0, (1.5, 2)), zorder=1)
        rho = np.array([r[f"rho_{p}"] for p in pats])
        for y, p, v in zip(ys_p, pats, rho):
            big = r["label"] == "malignant" and p == "P03"
            ax.plot([0, v], [y, y], color=rs.AXIS, lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([v], [y], linestyle="none", marker="o", markersize=5.8 if big else 4.6,
                    markerfacecolor=rs.INK if big else rs.INK_2, markeredgecolor=rs.SURFACE,
                    markeredgewidth=0.8, zorder=3)
        # Summary rows; values printed in a column at the right edge.
        ax.axhline(len(pats) - 0.2, color=rs.AXIS, lw=0.8, zorder=1)
        vals = [r["rho_within_median"], r["rho_pooled"], r["rho_pct_median"]]
        marks = [("o", rs.INK, 6.4), ("D", rs.SURFACE, 5.2), ("s", rs.SURFACE, 5.0)]
        for y, v, (mk, fc, ms) in zip(ys_s, vals, marks):
            ax.plot([0, v], [y, y], color=rs.AXIS, lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([v], [y], linestyle="none", marker=mk, markersize=ms, markerfacecolor=fc, markeredgecolor=rs.INK,
                    markeredgewidth=1.1, zorder=3)
            ax.text(xlim[1] - 0.02, y, _fmt_signed(v, 3), fontsize=6.6, color=rs.INK, ha="right", va="center",
                    zorder=4)
        n = counts[r["label"]]
        ylabels = [f"{p} ({n.loc[p]:,})" for p in pats] + summary_rows
        ax.set_yticks(np.concatenate([ys_p, ys_s]))
        if i < 2:
            ax.set_yticklabels(ylabels, fontsize=6.6)
            for lab in ax.get_yticklabels()[len(pats):]:
                lab.set_color(rs.INK)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylim(ys_s[-1] + 0.6, -0.7)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_minus(f"{v:+.1f}") if v else "0" for v in ticks], fontsize=6.8)
        ax.set_xlabel(f"\u03c1 (oriented); band: isotype \u00b1{ref:.2f}", fontsize=6.8)
        # Title: panel name and verdict (bold), then both conditions of the rule with this pair's values
        # (regular), so the verdict is never read off the sign count alone (CD99: 6/7 agree, yet not robust
        # because its |median| is below 0.20). Since v3.1 the verdict names the check it is (A-20261010-18): the
        # rule reads the within-patient values only ("across patients"); whether a correlation persists inside
        # cell types is the square, the second check of Figure 22 (fig05j).
        med, k = float(r["rho_within_median"]), int(r["n_same_sign"])
        robust = _is_robust(med, k)
        verdict = "across patients: passes" if robust else "across patients: fails"
        size_ok = abs(med) >= ROBUST_ABS
        sign_ok = k >= ROBUST_SAME_SIGN
        ax.set_title(f"{r['title']}\n{verdict}", fontsize=7.4, pad=22)
        ge = "\u2265"  # no backslash inside f-string expressions (Python 3.11)
        cond = (f"|median| {abs(med):.3f} {ge if size_ok else '<'} {ROBUST_ABS:.2f}"
                f"{'' if size_ok else '  (fails)'}\n"
                f"{k}/7 same sign {ge if sign_ok else '<'} {ROBUST_SAME_SIGN}/7"
                f"{'' if sign_ok else '  (fails)'}")
        ax.text(0.0, 1.015, cond, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3,
                color=rs.INK_2, linespacing=1.15)
    # Space between (a) and (b) for (b)'s y labels.
    pos_a = axes[0].get_position()
    axes[0].set_position([pos_a.x0, pos_a.y0, pos_a.width * 0.80, pos_a.height])
    for ax in axes[1:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + 0.115 if ax is axes[1] else pos.x0 + 0.06, pos.y0,
                         pos.width - (0.06 if ax is axes[1] else 0.0), pos.height])
    rs.save(fig, "fig05e_rho_anchors")
    return tab


# ---- D27: all cards -------------------------------------------------------------------------

def _card_label(row) -> str:
    panel = PANEL_SHORT.get(row["panel_label"], row["panel_label"])
    return f"{panel} \u00b7 {METHOD_SHORT[row['method']]}{int(row['k'])} f{int(row['factor'])}"


def top_robust_pairs(label: str, n: int = 10) -> pd.DataFrame:
    rob = pd.read_csv(d27_dir() / "d27_adt_robust.csv")
    r = rob[(rob["label"] == label) & rob["oriented"].astype(bool)].copy()
    assert ((r["rho_within_median"].abs() >= ROBUST_ABS) & (r["n_patients_same_sign"] >= ROBUST_SAME_SIGN)).all()
    r["abs_within"] = r["rho_within_median"].abs()
    best = (r.sort_values("abs_within", ascending=False).drop_duplicates("adt_marker_short").head(n).copy())
    best["card_label"] = best.apply(_card_label, axis=1)
    return best.reset_index(drop=True)


def robust_counts() -> dict:
    rob = pd.read_csv(d27_dir() / "d27_adt_robust.csv")
    full = pd.read_parquet(d27_dir() / "d27_adt_assoc_full.parquet",
                           columns=["card_id", "label", "is_isotype", "rho_pooled", "rho_within_median",
                                    "n_patients_same_sign"])
    out = {}
    for lab in LABELS:
        m = full[(full["label"] == lab) & ~full["is_isotype"]]
        iso = full[(full["label"] == lab) & full["is_isotype"]]
        rr = rob[rob["label"] == lab]
        iso_pass = int(((iso["rho_within_median"].abs() >= ROBUST_ABS)
                        & (iso["n_patients_same_sign"] >= ROBUST_SAME_SIGN)).sum())
        big = m[m["rho_pooled"].abs() >= 0.20]
        ratio = big["rho_within_median"] / big["rho_pooled"]
        rev = int(((np.sign(big["rho_within_median"]) != np.sign(big["rho_pooled"]))
                   & (big["rho_within_median"].abs() >= 0.10)).sum())
        out[lab] = {
            "n_marker_rows": int(len(m)), "n_isotype_rows": int(len(iso)), "n_robust": int(len(rr)),
            "n_robust_cards": int(rr["card_id"].nunique()), "iso_pass": iso_pass,
            "n_exceeds_iso": int(rr["exceeds_isotype_reference"].astype(bool).sum()),
            "n_kept_in_ct": int((rr["pct_to_within_ratio"] >= 0.5).sum()),
            "n_big": int(len(big)), "ratio_median": float(ratio.median()),
            "ratio_q25": float(ratio.quantile(0.25)), "ratio_q75": float(ratio.quantile(0.75)), "n_reversed": rev,
            "ratio": ratio.to_numpy(),
        }
    return out


def celltype_vs_patient_eta2() -> pd.DataFrame:
    eta = pd.read_csv(d27_dir() / "d27_celltype_eta2.csv")
    eta["eta2_patient"] = np.where(eta["label"] == "normal", eta["eta2_patient_normal_d18"],
                                   eta["eta2_patient_malignant_d18"])
    return eta


def _pairs_panel(ax, best: pd.DataFrame, title: str, iso_q99: float) -> None:
    ys = np.arange(len(best), dtype=float)
    ax.axvspan(0, iso_q99, color=rs.GRID, lw=0, zorder=0)
    for y, (_, r) in zip(ys, best.iterrows()):
        sgn = np.sign(r["rho_within_median"])
        w, pl, ct = abs(r["rho_within_median"]), sgn * r["rho_pooled"], sgn * r["rho_pct_median"]
        lo, hi = min(w, pl, ct), max(w, pl, ct)
        ax.plot([lo, hi], [y, y], color=rs.AXIS, lw=1.0, zorder=1, solid_capstyle="butt")
        ax.plot([ct], [y], linestyle="none", marker="s", markersize=4.2, markerfacecolor=rs.SURFACE,
                markeredgecolor=rs.INK_2, markeredgewidth=1.0, zorder=2)
        ax.plot([pl], [y], linestyle="none", marker="D", markersize=4.2, markerfacecolor=rs.SURFACE,
                markeredgecolor=rs.INK, markeredgewidth=1.0, zorder=3)
        # One ink colour: the row label already names the Stage 1 method (A-20261010-35: colour was redundant, and its
        # missing blue read as "no PCA card").
        ax.plot([w], [y], linestyle="none", marker="o", markersize=5.6, markerfacecolor=rs.INK,
                markeredgecolor=rs.SURFACE, markeredgewidth=0.8, zorder=4)
        ax.text(0.99, y, f"{int(r['n_patients_same_sign'])}/7", transform=ax.get_yaxis_transform(), fontsize=6.4,
                color=rs.INK_2, ha="right", va="center")
    labels = []
    for _, r in best.iterrows():
        sign = "+" if r["rho_within_median"] > 0 else "\u2212"
        labels.append(f"{r['adt_marker_short']} ({sign})\n{r['card_label']}")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=6.3, linespacing=1.0)
    for lab in ax.get_yticklabels():
        lab.set_color(rs.INK)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("|\u03c1| (sign of the within-patient median)", fontsize=7.4)
    ax.set_title(title, fontsize=8.4)


def fig05f_modality_all() -> dict:
    counts = robust_counts()
    noise = pd.read_csv(d27_dir() / "d27_adt_noise_floor.csv")
    q99 = {lab: float(noise[(noise["label"] == lab) & (noise["feature_kind"] == "isotype")
                            & (noise["statistic"] == "abs_rho_within_median")]["q99"].iloc[0]) for lab in LABELS}
    best = {lab: top_robust_pairs(lab) for lab in LABELS}
    eta = celltype_vs_patient_eta2()

    fig = plt.figure(figsize=(rs.FULL_W, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.75, 1.0], hspace=0.42, wspace=0.55, top=0.915, bottom=0.075,
                          left=0.21, right=0.985)
    # Two dot plots side by side in the top row (nested grid keeps the y labels inside the figure).
    top = gs[0, :].subgridspec(1, 2, wspace=1.3)  # room for (a)'s k/7 column before (b)'s labels
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2], sharey=ax_d)

    _pairs_panel(ax_a, best["normal"], "(a) Normal cells: strongest pair per marker", q99["normal"])
    _pairs_panel(ax_b, best["malignant"], "(b) Malignant cells: strongest pair per marker", q99["malignant"])
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.4, markerfacecolor=rs.INK,
               markeredgecolor=rs.SURFACE, label="median within patient"),
        Line2D([], [], linestyle="none", marker="D", markersize=4.2, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.INK, label="pooled (patients mixed)"),
        Line2D([], [], linestyle="none", marker="s", markersize=4.2, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.INK_2, label="inside patient \u00d7 cell type"),
        plt.Rectangle((0, 0), 1, 1, color=rs.GRID, label="isotype 99% range"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=6.8,
               handletextpad=0.3, columnspacing=1.1)

    # (c) within / pooled ratio.
    bins = np.linspace(-1.0, 2.0, 61)
    spots = {"normal": (1.13, 0.098, "left"), "malignant": (-0.12, 0.080, "right")}
    for lab, col in (("normal", rs.MUTED), ("malignant", rs.INK)):
        rr = np.clip(counts[lab]["ratio"], bins[0] + 1e-9, bins[-1] - 1e-9)
        h, _ = np.histogram(rr, bins=bins)
        h = h / h.sum()
        ax_c.stairs(h, bins, color=col, lw=1.3)
        med = counts[lab]["ratio_median"]
        ax_c.axvline(med, color=col, lw=0.9, ls=(0, (3, 2)))
        x, y, ha = spots[lab]
        ax_c.text(x, y, f"{lab}\nmedian {med:.2f}", fontsize=6.4, color=col, ha=ha, va="center", linespacing=1.0)
    ax_c.axvline(1.0, color=rs.AXIS, lw=0.8)
    ax_c.axvline(0.0, color=rs.AXIS, lw=0.8)
    ax_c.set_xlim(-1.0, 2.0)
    ax_c.set_ylim(0, 0.13)
    ax_c.set_xticks([-1, 0, 1, 2])
    ax_c.set_yticks([0, 0.05, 0.10])
    ax_c.set_yticklabels(["0", "0.05", "0.10"])
    ax_c.set_xlabel("within-patient median \u00f7 pooled \u03c1\n(pairs with |pooled \u03c1| \u2265 0.20)", fontsize=7.0)
    ax_c.set_ylabel("share of pairs", fontsize=7.4)
    ax_c.set_title("(c) Does a pooled \u03c1\nhold within patients?", fontsize=8.0)

    # (d, e) eta2: cell type within patient vs patient.
    for ax, lab, title in ((ax_d, "normal", "(d) Normal cells"), (ax_e, "malignant", "(e) Malignant cells")):
        e = eta[eta["label"] == lab]
        ax.plot([0, 1], [0, 1], color=rs.AXIS, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.plot(e["eta2_patient"], e["eta2_celltype_within_median"], linestyle="none", marker="o", markersize=2.0,
                markerfacecolor=rs.MUTED, markeredgecolor="none", alpha=0.45, zorder=1)
        for cid, name, m in ((PC8, "PC8", "pca"), (FA7, "FA7", "fa")):
            rr = e[e["card_id"] == cid].iloc[0]
            ax.plot([rr["eta2_patient"]], [rr["eta2_celltype_within_median"]], linestyle="none", marker="o",
                    markersize=5.0, markerfacecolor=rs.INK, markeredgecolor=rs.SURFACE, markeredgewidth=0.8,
                    zorder=3)
            ax.annotate(name, xy=(rr["eta2_patient"], rr["eta2_celltype_within_median"]),
                        xytext=(0.30, 0.55 if name == "FA7" else 0.42), fontsize=6.6, color=rs.INK,
                        arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.6, shrinkA=1, shrinkB=2))
        above = int((e["eta2_celltype_within_median"] > e["eta2_patient"]).sum())
        # Region labels (A-20261010-37): what above and below the diagonal mean, with the card counts.
        ax.text(0.97, 0.97, f"above: cell type explains\nmore ({above:,} of {len(e):,})", transform=ax.transAxes,
                fontsize=6.2, color=rs.INK, ha="right", va="top", linespacing=1.05)
        ax.text(0.97, 0.03, f"below: patient explains\nmore ({len(e) - above:,})", transform=ax.transAxes,
                fontsize=6.2, color=rs.INK, ha="right", va="bottom", linespacing=1.05)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("\u03b7\u00b2 of patient", fontsize=7.4)
        ax.set_title(title, fontsize=8.0)
        ax.grid(axis="x", visible=True)
    ax_d.set_ylabel("\u03b7\u00b2 of cell type\n(median within patient)", fontsize=7.4)
    plt.setp(ax_e.get_yticklabels(), visible=False)
    rs.save(fig, "fig05f_modality_all")
    return {"counts": counts, "best": best, "q99": q99, "eta": eta}


# ---------------------------------------------------------------------------------------------
# Report v3.1 (2026-10-10): the core report's surface-protein page (asks A-20261010-18, -21, -23, -24)
# ---------------------------------------------------------------------------------------------

# The explainer figures of section 5 (full report Figures 19, 20 and 22: fig05g-fig05k) are drawn by the Pavan
# update's make_figures_v3.py (make_figures section "05p"). The core page's one figure reuses that script's
# loaders, per-cell checks and two-checks panel, so both reports show the same numbers from the same code.
EXPLAINER = HERE.parents[1] / "team_updates" / "2026-10-09_protein_association_for_pavan" / "make_figures_v3.py"


def _explainer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_figures_v3_core", EXPLAINER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fig05l_protein_core() -> dict:
    """Core report, surface-protein page: (A) what the sign of one Spearman rho means, compact (panel D of full
    report Figure 20: the 1,569 malignant cells ranked on oriented PC8 and on CD99 or CD48, as the mean protein
    rank in each tenth of the PC8 ranking); (B, C) the two checks per protein of full report Figure 22 (fig05j),
    malignant and healthy cells, drawn with the explainer's own panel function. Drawn 7.0 in wide (the core's
    text width is 7.1 in), so the core places it at its drawn size."""
    mx = _explainer()
    d27 = mx.load_d27()
    iso = mx.isotype_refs(d27["adt"])
    cohort = mx.load_cells(d27["orient"])
    d5 = mx.fig5_data(d27, cohort)  # asserts the per-cell recomputation against D27
    n = d5["n_mal"]
    mid = (n + 1) / 2

    W, H = 7.0, 3.1  # v3.1 review: 0.4 in shorter (panels 1.9 in, one legend row) so Table 1 fits the page
    c = mx.Canvas(W, H)
    fs_small, fs_note = mx.FS_SMALL, mx.FS_NOTE
    top, side = 0.62, 1.9  # B and C: axes top (in) and side
    # ---- A: the sign, compact
    c.panel_title(0.1, 0.2, "A", "What the sign of ρ means")
    c.text(0.3, 0.3, f"Each of the {n:,} malignant cells is ranked\non oriented PC8 (x) and on one protein (y);\n"
                     "ρ = the Pearson correlation of the two\nrank vectors. Line: mean protein rank in\n"
                     "each tenth of the PC8 ranking. Dotted:\nthe middle rank, where it stays if ρ = 0.",
           fontsize=fs_note - 0.6, color=rs.INK_2, linespacing=1.25)
    a_side = 1.45
    axA = c.ax(0.52, top + side - a_side, a_side, a_side)
    axA.axhline(mid, color=rs.INK_2, lw=0.8, ls=(0, (1, 2)), zorder=1)
    styles = {mx.SIGN_EXAMPLES[0]: dict(color=rs.INK, marker="o"), mx.SIGN_EXAMPLES[1]: dict(color=rs.MUTED, marker="s")}
    for prot in mx.SIGN_EXAMPLES:
        s = d5["sign"][prot]
        axA.plot(s["bin_x"], s["bin_y"], lw=1.5, ms=3.0, mec=rs.SURFACE, mew=0.5, zorder=3, **styles[prot])
    s_pos, s_neg = (d5["sign"][p] for p in mx.SIGN_EXAMPLES)
    axA.text(n * 0.97, s_pos["bin_y"][-1] + 95, f"{mx.SIGN_EXAMPLES[0]}  ρ {mx.signed(s_pos['rho'], 3)}\nforward",
             ha="right", va="bottom", fontsize=fs_small - 0.8, color=rs.INK, fontweight="bold", linespacing=1.1)
    axA.text(n * 0.97, s_neg["bin_y"][-1] - 95, f"{mx.SIGN_EXAMPLES[1]}  ρ {mx.signed(s_neg['rho'], 3)}\nreversed",
             ha="right", va="top", fontsize=fs_small - 0.8, color=rs.INK_2, fontweight="bold", linespacing=1.1)
    ticks = [1, int(round(mid)), n]
    axA.set_xlim(-30, n + 31)
    axA.set_ylim(-30, n + 31)
    for setter in (axA.set_xticks, axA.set_yticks):
        setter(ticks)
    axA.set_xticklabels([f"{t:,}" for t in ticks], fontsize=fs_small - 1.2)
    axA.set_yticklabels([f"{t:,}" for t in ticks], fontsize=fs_small - 1.2)
    axA.tick_params(length=2, pad=1.5)
    axA.grid(axis="both", visible=False)
    axA.set_xlabel("PC8 rank (1 = lowest)", fontsize=fs_small - 0.6, color=rs.INK_2, labelpad=2)
    axA.set_ylabel("protein rank", fontsize=fs_small - 0.6, color=rs.INK_2, labelpad=1)
    # ---- B, C: the two checks (as fig05j, smaller)
    mx.LABEL_OFFSETS.update({("malignant", "CD82"): (5, 4), ("malignant", "GPR56"): (5, -1),
                             ("normal", "CD112"): (5, -7)})
    # CD146 sits near the right edge of the smaller panel: its label goes below it on a leader, into empty space
    # (to its right it would run into panel C; to its left it sat on the isotype diamonds).
    mx.LABEL_AT[("malignant", "CD146")] = (0.232, -0.118)
    adt = d27["adt"]
    r99 = mx.d27_row(adt, "PC8", "malignant", "CD99")
    r34 = mx.d27_row(adt, "PC8", "normal", "CD34")
    xB, xC = 2.72, 4.98
    c.panel_title(xB - 0.42, 0.2, "B", "Two checks: malignant cells")
    c.text(xB - 0.22, 0.3, f"{d27['pooled_n']['malignant']:,} cells; {int(r99['n_pct_strata'])} type groups from "
                           "P02, P03, P04, P09", fontsize=fs_small - 1.2, color=rs.INK_2)
    c.panel_title(xC + 0.02, 0.2, "C", "Healthy cells")
    c.text(xC + 0.22, 0.3, f"{d27['pooled_n']['normal']:,} cells; {int(r34['n_pct_strata'])} type groups\n"
                           "from 7 patients", fontsize=fs_small - 1.2, color=rs.INK_2, linespacing=1.15)
    axB = c.ax(xB, top, side, side)
    axC = c.ax(xC, top, side, side)
    tB = mx.fig6_panel(axB, adt, iso.loc[("PC8", "malignant")], "malignant", key=False)
    tC = mx.fig6_panel(axC, adt, iso.loc[("PC8", "normal")], "normal", key=False)
    for ax in (axB, axC):
        ax.set_xlabel("x: within patient (median of 7)", fontsize=fs_small - 0.6, color=rs.INK_2, labelpad=2)
        ax.tick_params(labelsize=fs_small - 1.4, length=2, pad=1.5)
    axB.set_ylabel("y: within patient and cell type (median)", fontsize=fs_small - 0.6, color=rs.INK_2, labelpad=1)
    axC.set_ylabel("")
    axC.set_yticklabels([])
    handles = [Line2D([], [], ls="none", marker="o", ms=5.0, mfc=rs.INK_2, mec=rs.SURFACE),
               Line2D([], [], ls="none", marker="o", ms=4.0, mfc=mx.LIGHT_DOT, mec="none"),
               Line2D([], [], ls="none", marker="D", ms=4.0, mfc=mx.ISOTYPE_C, mec=rs.SURFACE),
               Line2D([], [], color=rs.MUTED, lw=0.9, ls=(0, (4, 2.5))),
               Line2D([], [], color=rs.INK_2, lw=0.8, ls=(0, (1, 2))),
               Rectangle((0, 0), 1, 1, facecolor=mx.BAND, edgecolor="none")]
    c.fig.legend(handles, ["named protein", "other", "isotype (9)",
                           "y = x: holds in type; y ≈ 0: cell type", f"|x| = {mx.ROBUST_ABS:.2f}: rule's bar",
                           "isotype reach"],
                 loc="upper left", bbox_to_anchor=(0.3 / W, 1 - (top + side + 0.36) / H), ncol=6,
                 frameon=False, fontsize=fs_small - 1.2, handletextpad=0.4, columnspacing=0.8, labelspacing=0.35)
    bad = c.overflow()
    assert not bad, f"fig05l_protein_core: text outside the canvas: {bad}"
    c.save("fig05l_protein_core", report=True)
    both = pd.concat([tB, tC], ignore_index=True)
    return {"sign": {p: (round(d5["sign"][p]["rho"], 3), np.round(d5["sign"][p]["bin_y"]).astype(int).tolist())
                     for p in mx.SIGN_EXAMPLES},
            "checks": both[both["labelled"]][["cells", "adt_marker_short", "rho_within_median", "n_patients_same_sign",
                                               "rho_pct_median", "n_pct_same_sign", "meets_fig20_rule",
                                               "outside_x_band", "outside_y_band"]],
            "iso": iso.loc[[("PC8", "malignant"), ("PC8", "normal")], ["iso_within_patient", "iso_within_type"]]}


def adt_table_rows() -> pd.DataFrame:
    """Rows of the section's ADT table: PC8 and FA7, two markers per label, pooled vs within patient."""
    a = pd.read_csv(d27_dir() / "d27_ifna_pc8_fa7_adt.csv")
    picks = [(PC8, "normal", "CD34"), (PC8, "normal", "CD22"), (PC8, "malignant", "CD99"), (PC8, "malignant", "CD48"),
             (FA7, "normal", "CD22"), (FA7, "normal", "CD34"), (FA7, "malignant", "CD116"), (FA7, "malignant", "CLEC12A")]
    rows = []
    for card, lab, mk in picks:
        r = a[(a["card_id"] == card) & (a["label"] == lab) & (a["adt_marker_short"] == mk)].iloc[0]
        rows.append({"factor": "PC8" if card == PC8 else "FA7", "label": lab, "marker": mk,
                     "pooled": round(float(r["rho_pooled"]), 3), "within": round(float(r["rho_within_median"]), 3),
                     "same_sign": int(r["n_patients_same_sign"]), "in_celltype": round(float(r["rho_pct_median"]), 3)})
    cs = pd.read_csv(d27_dir() / "d27_card_summary.csv").set_index("card_id")
    for card, name in ((PC8, "PC8"), (FA7, "FA7")):
        for lab in LABELS:
            rows.append({"factor": name, "label": lab, "marker": "isotype ref.",
                         "within": round(float(cs.loc[card, f"isotype_max_abs_rho_within_median_{lab}"]), 3),
                         "pooled": round(float(cs.loc[card, f"isotype_max_abs_rho_pooled_{lab}"]), 3)})
    return pd.DataFrame(rows)


def print_every_card_numbers() -> None:
    """The numbers of section 5's "Every card" answers (version 3.2; asks A-20261010-33..37), from D27 and D18 tables."""
    cs = pd.read_csv(d27_dir() / "d27_card_summary.csv")
    rob = pd.read_csv(d27_dir() / "d27_adt_robust.csv")
    sl = pd.read_csv(cards_dir() / "factor_cards_shortlist.csv")
    eta = celltype_vs_patient_eta2()
    n_markers = 161
    print("[every card] cards", len(cs), "x markers", n_markers, "=", len(cs) * n_markers, "pairs per label; isotype rows",
          len(cs) * 9)
    print("[every card] cards by method", cs.groupby("method").size().to_dict(), "; orientation",
          cs["orientation_status"].value_counts().to_dict())
    print("[every card] bases: D14 screen", int(sl["shortlist_reasons"].str.contains("d14_screen").sum()), "of", len(sl))
    print("[every card] robust pairs by method", {lab: g.groupby("method").size().to_dict() for lab, g in rob.groupby("label")})
    for lab in LABELS:
        r = rob[(rob["label"] == lab) & rob["oriented"].astype(bool)].copy()
        r["a"] = r["rho_within_median"].abs()
        best = r.sort_values("a", ascending=False).drop_duplicates("adt_marker_short").head(10)
        pca = r[r["method"] == "pca"].groupby("adt_marker_short")["a"].max()
        shown = best.set_index("adt_marker_short")["a"]
        print(f"[every card] {lab}: shown |rho| {shown.min():.3f}-{shown.max():.3f}; best PCA card for the same markers "
              f"{pca.reindex(shown.index).min():.3f}-{pca.reindex(shown.index).max():.3f}; PCA among shown: "
              f"{int((best['method'] == 'pca').sum())}")
        e = eta[eta["label"] == lab]
        x, y = e["eta2_patient"], e["eta2_celltype_within_median"]
        print(f"[every card] {lab}: eta2 patient median {x.median():.3f}, cell type median {y.median():.3f}; above "
              f"{int((y > x).sum())}, below {int((y <= x).sum())}; both < 0.1: {int(((x < 0.1) & (y < 0.1)).sum())}")
        for cid, name in ((PC8, "PC8"), (FA7, "FA7")):
            rr = e[e["card_id"] == cid].iloc[0]
            print(f"[every card] {lab} {name}: eta2 patient {rr['eta2_patient']:.3f}, cell type {rr['eta2_celltype_within_median']:.3f}")


def main() -> None:
    rs.apply()
    a = fig05a_pc8_poles()
    b = fig05b_axis_lift()
    # fig05c_celltype (version 3's Figure 19) is no longer drawn: both reports show the relabelled fig05g
    # (make_figures section 05p). Its table still gives the "8 of 10 types" the text quotes.
    c = celltype_table()
    pd.set_option("display.width", 220)
    print(a.to_string())
    print(b[["factor_name", "patient", "univariate_lift_pp", "fitted_single_factor_lift_pp", "coef", "is_nonzero",
             "fold_all_zero", "n_malignant", "prevalence"]].to_string())
    print(c.to_string())
    print_v2_numbers()
    print_every_card_numbers()


def print_v2_numbers() -> None:
    """Every number the v2 text quotes from the new figures and the card accounting."""
    acc = card_accounting()
    print("\n== 1,810 cards")
    print({k: v for k, v in acc.items() if k not in ("rep", "degenerate")})
    print(acc["degenerate"].to_string())
    d = fig05d_identifiability()
    spec = d["spec"]
    print("\n== identifiability")
    print(spec.head(10).round(4).to_string())
    print("PCs with relative eigengap <= 0.10:", d["n_pcs_tied"], "of", len(spec))
    print("FA40 determinacy < 0.707:", d["fa_n_below"], "of", d["fa_n_nondeg"])
    print(d["pert"].round(4).to_string())
    e = fig05e_rho_anchors()
    print("\n== rho anchors")
    print(e.round(3).to_string())
    print(d27_patient_counts().to_string())
    f = fig05f_modality_all()
    print("\n== all cards")
    for lab in LABELS:
        print(lab, {k: v for k, v in f["counts"][lab].items() if k != "ratio"})
        print(f["best"][lab][["adt_marker_short", "card_label", "rho_within_median", "n_patients_same_sign",
                               "rho_pooled", "rho_pct_median", "isotype_max_abs_rho_within_median",
                               "is_selected_driver"]].round(3).to_string())
    print("isotype q99 |median within|:", f["q99"])
    eta = f["eta"]
    for lab in LABELS:
        e2 = eta[eta["label"] == lab]
        print(lab, "eta2 celltype within median (median over cards)", round(float(e2["eta2_celltype_within_median"].median()), 3),
              "eta2 patient median", round(float(e2["eta2_patient"].median()), 3),
              "cards ct > patient", int((e2["eta2_celltype_within_median"] > e2["eta2_patient"]).sum()))
        for cid, nm in ((PC8, "PC8"), (FA7, "FA7")):
            rr = e2[e2["card_id"] == cid].iloc[0]
            print("  ", nm, "eta2 ct pooled", round(float(rr["eta2_celltype_pooled"]), 3), "ct within", round(float(rr["eta2_celltype_within_median"]), 3),
                  "patient", round(float(rr["eta2_patient"]), 4), "null within", round(float(rr["eta2_null_within_median"]), 4))
    print("\n== ADT table")
    print(adt_table_rows().to_string())
    core = fig05l_protein_core()
    print("\n== core protein page (fig05l): sign examples (pooled rho; mean protein rank per PC8 tenth)")
    print(core["sign"])
    print(core["iso"].round(3).to_string())
    print(core["checks"].round(3).to_string())


if __name__ == "__main__":
    main()
