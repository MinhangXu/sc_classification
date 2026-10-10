"""Updated report Figure 13 + Table 1 for a collaborator email: PC8 vs CITE-seq surface protein.

One figure, two panels, for IFN-alpha PCA K=40 factor 8 (PC8; Stage 1 is RNA only):
  A  Which surface proteins track the factor? Spearman rho of the oriented PC8 score with per-cell
     CLR ADT, within normal and within malignant cells, at three levels: all cells of the label
     pooled (the report's Table 1 number), within each supported patient (median, min-max range and
     same-sign count over the 7 supported patients) and within supported patient x predicted cell
     type groups (>= 30 cells; median). Isotype noise floor = the largest |rho| over the 9 isotype
     controls at the same level, for this factor and label.
  B  Where the factor sits by cell type (the report's Figure 13): mean oriented PC8 score per
     predicted cell type, normal vs malignant, plus the number of supported patients (>= 10 cells
     of both labels in that type) in which the malignant mean is higher.

Proteins shown per factor x label: the N_TOP largest |within-patient median rho|, the report's
Table 1 proteins (top N_TABLE1 by |pooled rho|) and any protein that meets D27's robust rule
(|within-patient median| >= 0.20, same sign in >= 5 of 7) and keeps at least half of it, with the
same sign, inside patient x cell type groups (D27's "kept within cell type" rate).

Outputs (this directory): protein_association_pc8.{png,pdf}; protein_association_table.csv
(PC8 and FA7: the proteins selected by the rule above); protein_association_celltype.csv (panel B).

Reads only D27 artifacts (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_ifna_pc8_fa7_adt.csv
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_card_summary.csv
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_strata.csv
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_celltype_means.csv
  analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/d27_celltype_means_by_patient.csv.gz

Palette and type scale come from the questions report's report_style.py (imported, not edited).

Run (lab env), from anywhere:
  python scripts/mrd_stage0_2/research/team_updates/2026-10-09_protein_association_for_pavan/make_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # team_updates/<dir>/ -> research -> mrd_stage0_2 -> scripts -> repo
STYLE_DIR = REPO / "scripts" / "mrd_stage0_2" / "research" / "reports" / "2026-10-08_questions_report"
sys.path.insert(0, str(STYLE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import report_style as rs  # noqa: E402

SLUG = "factor_modality_assoc_v1"
PANEL = "single_geneset__hallmark_interferon_alpha_response"
CARDS = {"PC8": f"{PANEL}__pca__k40__f008", "FA7": f"{PANEL}__fa__k40__f007"}
FACTOR_LONG = {"PC8": "IFN-alpha PCA K=40 factor 8", "FA7": "IFN-alpha FA K=40 factor 7"}
LABELS = ("normal", "malignant")
SUPPORTED = list(rs.SUPPORTED)
N_TOP = 5  # proteins with the largest |within-patient median rho| per factor x label
N_TABLE1 = 3  # the report's Table 1 lists the top 3 proteins by |pooled rho|
ROBUST_RHO, ROBUST_SAME_SIGN = 0.20, 5  # D27 robust rule (fixed before D27 computed anything)
KEPT_FRACTION = 0.5  # D27 "kept within cell type": same sign and |pt x ct median| >= 0.5 |within median|
# Report Table 1 (2026-10-08 questions report, section 5), used only as a regression check.
TABLE1_EXPECTED = {
    ("PC8", "normal"): ["CD34", "CD22", "CD112"],
    ("PC8", "malignant"): ["CD99", "CD244", "GPR56"],
    ("FA7", "normal"): ["CD22", "CD112", "CD34"],
    ("FA7", "malignant"): ["GPR56", "CD63", "CD48"],
}
OUT_STEM = "protein_association_pc8"


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def d27_dir() -> Path:
    return rs.diag_dir(SLUG)


def load_adt() -> pd.DataFrame:
    df = pd.read_csv(d27_dir() / "d27_ifna_pc8_fa7_adt.csv")
    inv = {v: k for k, v in CARDS.items()}
    df["factor"] = df["card_id"].map(inv)
    assert df["factor"].notna().all()
    assert (df["n_patients_valid"] == len(SUPPORTED)).all()
    return df


def load_card_summary() -> pd.DataFrame:
    cs = pd.read_csv(d27_dir() / "d27_card_summary.csv")
    cs = cs[cs["card_id"].isin(CARDS.values())].copy()
    assert len(cs) == 2 and cs["oriented"].all()
    return cs.set_index("card_id")


def load_strata() -> dict:
    """Cell counts behind each level, per label (pooled n, per-patient n, patient x cell type groups)."""
    st = pd.read_csv(d27_dir() / "d27_strata.csv")
    out = {}
    for lab in LABELS:
        s = st[st["label"] == lab]
        pooled = s[s["kind"] == "pooled"]
        assert len(pooled) == 1
        pat = s[(s["kind"] == "patient") & s["supported_patient"]]
        assert sorted(pat["patient"]) == SUPPORTED
        pct = s[s["kind"] == "patient_celltype"]
        assert pct["supported_patient"].all() and (pct["n_cells"] >= 30).all()
        out[lab] = {
            "n_pooled": int(pooled["n_cells"].iloc[0]),
            "n_patient_min": int(pat["n_cells"].min()),
            "n_patient_max": int(pat["n_cells"].max()),
            "n_groups": len(pct),
            "group_patients": sorted(pct["patient"].unique()),
        }
    return out


def isotype_floors(adt: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    """Largest |rho| over the 9 isotype controls, per factor x label, at each of the three levels."""
    iso = adt[adt["is_isotype"]]
    assert iso.groupby(["factor", "label"]).size().eq(9).all()
    rows = []
    for (fac, lab), g in iso.groupby(["factor", "label"]):
        floor = {
            "factor": fac,
            "label": lab,
            "floor_pooled": g["rho_pooled"].abs().max(),
            "floor_within": g["rho_within_median"].abs().max(),
            "floor_celltype": g["rho_pct_median"].abs().max(),
            "floor_within_isotype": g.loc[g["rho_within_median"].abs().idxmax(), "adt_marker_short"],
            "floor_within_second": g["rho_within_median"].abs().nlargest(2).iloc[-1],
            "floor_celltype_isotype": g.loc[g["rho_pct_median"].abs().idxmax(), "adt_marker_short"],
        }
        # Cross-check against the D27 card summary (the numbers quoted in its findings).
        card = cs.loc[CARDS[fac]]
        assert abs(floor["floor_pooled"] - card[f"isotype_max_abs_rho_pooled_{lab}"]) < 1e-9
        assert abs(floor["floor_within"] - card[f"isotype_max_abs_rho_within_median_{lab}"]) < 1e-9
        rows.append(floor)
    return pd.DataFrame(rows).set_index(["factor", "label"])


def select_markers(adt: pd.DataFrame) -> pd.DataFrame:
    """Top N_TOP by |median within-patient rho|, the report's Table 1 proteins, and robust proteins
    that keep at least half their within-patient rho inside patient x cell type groups."""
    m = adt[~adt["is_isotype"]].copy()
    m["abs_within"] = m["rho_within_median"].abs()
    m["rank_within"] = m.groupby(["factor", "label"])["abs_within"].rank(ascending=False, method="first").astype(int)
    m["rank_pooled"] = (m["rho_pooled"].abs().groupby([m["factor"], m["label"]])
                        .rank(ascending=False, method="first").astype(int))
    m["in_earlier_table"] = m["rank_pooled"] <= N_TABLE1
    for key, expected in TABLE1_EXPECTED.items():
        got = (m[(m["factor"] == key[0]) & (m["label"] == key[1]) & m["in_earlier_table"]]
               .sort_values("rank_pooled")["adt_marker_short"].tolist())
        assert got == expected, (key, got, expected)
    m["meets_robust_rule"] = (m["abs_within"] >= ROBUST_RHO) & (m["n_patients_same_sign"] >= ROBUST_SAME_SIGN)
    m["keeps_half_within_celltype"] = (
        m["meets_robust_rule"]
        & (np.sign(m["rho_pct_median"]) == np.sign(m["rho_within_median"]))
        & (m["rho_pct_median"].abs() >= KEPT_FRACTION * m["abs_within"]))
    m["shown"] = (m["rank_within"] <= N_TOP) | m["in_earlier_table"] | m["keeps_half_within_celltype"]
    return m[m["shown"]].copy()


def build_table(sel: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sel.iterrows():
        fl = floors.loc[(r["factor"], r["label"])]
        row = {
            "factor": r["factor"],
            "factor_description": FACTOR_LONG[r["factor"]],
            "cells": r["label"],
            "marker": r["adt_marker_short"],
            "adt_feature": r["adt_marker"],
            "in_earlier_table": bool(r["in_earlier_table"]),
            "rank_by_within_patient_rho": int(r["rank_within"]),
            "rho_pooled": r["rho_pooled"],
            "rho_within_patient_median": r["rho_within_median"],
            "rho_within_patient_min": r["rho_within_min"],
            "rho_within_patient_max": r["rho_within_max"],
            "n_patients_same_sign": int(r["n_patients_same_sign"]),
            "n_patients": int(r["n_patients_valid"]),
        }
        for p in SUPPORTED:
            row[f"rho_{p}"] = r[f"rho_{p}"]
        row.update({
            "rho_within_patient_celltype_median": r["rho_pct_median"],
            "n_patient_celltype_groups": int(r["n_pct_strata"]),
            "n_patient_celltype_same_sign": int(r["n_pct_same_sign"]),
            "isotype_floor_pooled": fl["floor_pooled"],
            "isotype_floor_within_patient": fl["floor_within"],
            "isotype_floor_within_patient_celltype": fl["floor_celltype"],
            "pooled_above_isotype_floor": bool(abs(r["rho_pooled"]) > fl["floor_pooled"]),
            "within_patient_above_isotype_floor": bool(abs(r["rho_within_median"]) > fl["floor_within"]),
            "within_celltype_above_isotype_floor": bool(abs(r["rho_pct_median"]) > fl["floor_celltype"]),
            "meets_robust_rule": bool(r["meets_robust_rule"]),
            "keeps_half_within_celltype": bool(r["keeps_half_within_celltype"]),
        })
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["_f"] = tab["factor"].map({"PC8": 0, "FA7": 1})
    tab["_l"] = tab["cells"].map({"normal": 0, "malignant": 1})
    tab = tab.sort_values(["_f", "_l", "rho_within_patient_median"], ascending=[True, True, False])
    return tab.drop(columns=["_f", "_l"]).reset_index(drop=True)


def celltype_table() -> pd.DataFrame:
    """Panel B: pooled cell-type means (types with >= 20 cells of both labels) and per-patient counts."""
    card = CARDS["PC8"]
    ct = pd.read_csv(d27_dir() / "d27_celltype_means.csv")
    ct = ct[ct["card_id"] == card]
    wide = ct.pivot_table(index="cell_type", columns="label", values=["mean_oriented_z", "n_cells"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["mean_oriented_z_malignant", "mean_oriented_z_normal"]).reset_index()
    for lab in LABELS:
        wide[f"n_cells_{lab}"] = wide[f"n_cells_{lab}"].astype(int)

    bp = pd.read_csv(d27_dir() / "d27_celltype_means_by_patient.csv.gz")
    bp = bp[(bp["card_id"] == card) & bp["supported_patient"]]
    pw = bp.pivot_table(index=["cell_type", "patient"], columns="label", values="mean_oriented_z").dropna()
    pw["malignant_higher"] = pw["malignant"] > pw["normal"]
    per_type = pw.groupby("cell_type")["malignant_higher"].agg(n_patients_compared="count",
                                                               n_patients_malignant_higher="sum")
    wide = wide.merge(per_type, left_on="cell_type", right_index=True, how="left")
    wide[["n_patients_compared", "n_patients_malignant_higher"]] = (
        wide[["n_patients_compared", "n_patients_malignant_higher"]].fillna(0).astype(int))
    wide["malignant_higher_pooled"] = wide["mean_oriented_z_malignant"] > wide["mean_oriented_z_normal"]
    wide = wide.sort_values("n_cells_malignant", ascending=False).reset_index(drop=True)
    # Comparisons in supported-patient types that are not shown (fewer than 20 malignant cells pooled).
    hidden = per_type.drop(index=[c for c in wide["cell_type"] if c in per_type.index])
    wide.attrs["hidden_compared"] = int(hidden["n_patients_compared"].sum())
    wide.attrs["hidden_higher"] = int(hidden["n_patients_malignant_higher"].sum())
    wide.attrs["hidden_types"] = ", ".join(hidden.index)
    # Normal-cell ranking over every type with >= 20 normal cells (for the caption).
    nrm = ct[ct["label"] == "normal"].sort_values("mean_oriented_z", ascending=False)
    wide.attrs["normal_top3"] = list(zip(nrm["cell_type"].head(3), nrm["mean_oriented_z"].head(3).round(3)))
    return wide


# ----------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------

FS = 8.6  # base text size (the figure is read at email size)
FS_SMALL = 8.2
FS_NOTE = 7.8
POOLED_STYLE = dict(marker="s", markersize=6.4, markerfacecolor=rs.SURFACE, markeredgecolor=rs.INK_2,
                    markeredgewidth=1.2, linestyle="none")
WITHIN_STYLE = dict(marker="o", markersize=5.8, markerfacecolor=rs.INK, markeredgecolor=rs.SURFACE,
                    markeredgewidth=0.8, linestyle="none")
CT_STYLE = dict(marker="D", markersize=4.2, markerfacecolor=rs.MUTED, markeredgecolor=rs.SURFACE,
                markeredgewidth=0.6, linestyle="none")
RANGE_COLOR = "#a9a89f"
BAND_COLOR = "#e6e5df"
LANE = 0.33  # vertical offset (in rows) of the pooled (up) and cell-type (down) lanes
XLIM = (-0.8, 0.7)
GROUP_HEADER = {"normal": "Normal cells", "malignant": "Malignant cells"}


def ax_height_in(ax) -> float:
    """Axes height in inches (to place header text a fixed distance above an axes)."""
    return ax.get_position().height * ax.figure.get_figheight()


def above(ax, inches: float) -> float:
    """Axes-fraction y that sits `inches` above the top of the axes."""
    return 1.0 + inches / ax_height_in(ax)


def draw_assoc(ax, g: pd.DataFrame, fl: pd.Series, strata: dict, lab: str, show_xlabel: bool) -> None:
    g = g.sort_values("rho_within_patient_median", ascending=False).reset_index(drop=True)
    y = np.arange(len(g), dtype=float)
    ax.axvspan(-fl["floor_within"], fl["floor_within"], color=BAND_COLOR, lw=0, zorder=0)
    for s in (-1, 1):
        ax.axvline(s * fl["floor_pooled"], color=rs.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0, color=rs.INK_2, lw=0.7, zorder=1)
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.plot([r["rho_within_patient_min"], r["rho_within_patient_max"]], [yi, yi], color=RANGE_COLOR,
                lw=1.3, solid_capstyle="butt", zorder=2)
    ax.plot(g["rho_pooled"], y - LANE, zorder=3, **POOLED_STYLE)
    ax.plot(g["rho_within_patient_median"], y, zorder=4, **WITHIN_STYLE)
    ax.plot(g["rho_within_patient_celltype_median"], y + LANE, zorder=3, **CT_STYLE)
    ax.set_yticks(y)
    ax.set_yticklabels(g["marker"], fontsize=FS)
    for tick, bold in zip(ax.get_yticklabels(), g["in_earlier_table"]):
        tick.set_fontweight("bold" if bold else "normal")
        tick.set_color(rs.INK if bold else rs.INK_2)
    ax.set_ylim(len(g) - 0.45, -0.62)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(*XLIM)
    ticks = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0" if t == 0 else f"{t:+.1f}" for t in ticks], fontsize=FS_SMALL)
    if show_xlabel:
        ax.set_xlabel("Spearman ρ, protein level (CLR) vs PC8 score\n"
                      "ρ > 0: cells with more of the protein sit toward the malignant-leaning pole",
                      fontsize=FS, color=rs.INK_2, labelpad=4)
    st = strata[lab]
    ax.text(0.0, above(ax, 0.44), f"{GROUP_HEADER[lab]} (n = {st['n_pooled']:,})", transform=ax.transAxes,
            fontsize=FS + 0.6, fontweight="bold", color=rs.INK, ha="left", va="bottom")
    ax.text(0.0, above(ax, 0.25),
            (f"isotype floor: ±{fl['floor_within']:.2f} within patient (band), "
             f"±{fl['floor_pooled']:.2f} pooled (dashed)"),
            transform=ax.transAxes, fontsize=FS_NOTE, color=rs.INK_2, ha="left", va="bottom")
    if len(st["group_patients"]) == len(SUPPORTED):
        where = f"{len(SUPPORTED)} patients"
    else:
        where = "/".join(st["group_patients"]) + " only"
    ax.text(0.0, above(ax, 0.08),
            (f"diamonds: {st['n_groups']} groups, {where}; "
             f"floor ±{fl['floor_celltype']:.2f}, not drawn"),
            transform=ax.transAxes, fontsize=FS_NOTE, color=rs.INK_2, ha="left", va="bottom")
    # Same-sign column, right of the axis.
    for yi, (_, r) in zip(y, g.iterrows()):
        ax.text(1.04, yi, f"{r['n_patients_same_sign']}/{r['n_patients']}", transform=ax.get_yaxis_transform(),
                fontsize=FS_SMALL, color=rs.INK_2, ha="left", va="center")
    ax.text(1.04, above(ax, 0.08), "patients\nagreeing\nin sign", transform=ax.transAxes,
            fontsize=FS_SMALL - 0.8, color=rs.MUTED, ha="left", va="bottom", linespacing=0.95)


def draw_celltype(ax, ct: pd.DataFrame) -> None:
    y = np.arange(len(ct), dtype=float)
    for yi, (_, r) in zip(y, ct.iterrows()):
        ax.plot([r["mean_oriented_z_normal"], r["mean_oriented_z_malignant"]], [yi, yi], color=rs.AXIS,
                lw=rs.LINE_W, zorder=1, solid_capstyle="butt")
    ax.plot(ct["mean_oriented_z_normal"], y, linestyle="none", marker="o", markersize=7.0,
            markerfacecolor=rs.SURFACE, markeredgecolor=rs.MUTED, markeredgewidth=1.4, zorder=3)
    ax.plot(ct["mean_oriented_z_malignant"], y, linestyle="none", marker="o", markersize=7.0,
            markerfacecolor=rs.INK_2, markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, zorder=4)
    ax.axvline(0, color=rs.INK_2, lw=0.7, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ({n})" for c, n in zip(ct["cell_type"], ct["n_cells_malignant"])], fontsize=FS)
    ax.set_ylim(len(ct) - 0.4, -0.6)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(-0.45, 1.3)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "+0.5", "+1.0"], fontsize=FS_SMALL)
    ax.set_xlabel("Mean PC8 score (z-scored over all cells)\npositive = malignant-leaning pole", fontsize=FS,
                  color=rs.INK_2, labelpad=4)
    for yi, (_, r) in zip(y, ct.iterrows()):
        txt = (f"{r['n_patients_malignant_higher']}/{r['n_patients_compared']}"
               if r["n_patients_compared"] > 0 else "–")
        ax.text(1.03, yi, txt, transform=ax.get_yaxis_transform(), fontsize=FS_SMALL, color=rs.INK_2,
                ha="left", va="center")
    ax.text(1.03, -0.62, "patients where\nmalignant\n> normal", transform=ax.get_yaxis_transform(),
            fontsize=FS_SMALL - 0.8, color=rs.MUTED, ha="left", va="bottom", linespacing=0.95)
    ax.text(-0.03, -0.62, "predicted cell type\n(n malignant cells)", transform=ax.get_yaxis_transform(),
            fontsize=FS_SMALL - 0.8, color=rs.MUTED, ha="right", va="bottom", linespacing=0.95)


def make_figure(table: pd.DataFrame, floors: pd.DataFrame, strata: dict, ct: pd.DataFrame) -> plt.Figure:
    pc8 = table[table["factor"] == "PC8"]
    n_rows = {lab: int((pc8["cells"] == lab).sum()) for lab in LABELS}
    W, H = 10.4, 7.9
    fig = plt.figure(figsize=(W, H))

    def fy(inch_from_top: float) -> float:
        return 1.0 - inch_from_top / H

    def fx(inch_from_left: float) -> float:
        return inch_from_left / W

    # Geometry in inches (from the top / left edge).
    plot_top, plot_bottom = 2.86, H - 0.86
    sub_gap = 0.98  # normal x ticks + malignant header block (3 lines)
    unit = (plot_bottom - plot_top - sub_gap) / (n_rows["normal"] + n_rows["malignant"])
    a_left, a_width = 0.85, 3.9
    ax_n = fig.add_axes([fx(a_left), fy(plot_top + unit * n_rows["normal"]), fx(a_width),
                         unit * n_rows["normal"] / H])
    ax_m = fig.add_axes([fx(a_left), fy(plot_bottom), fx(a_width), unit * n_rows["malignant"] / H])
    b_left, b_width, b_top = 6.9, 2.3, 2.6
    ax_b = fig.add_axes([fx(b_left), fy(plot_bottom), fx(b_width), (plot_bottom - b_top) / H])

    for ax, lab in ((ax_n, "normal"), (ax_m, "malignant")):
        draw_assoc(ax, pc8[pc8["cells"] == lab], floors.loc[("PC8", lab)], strata, lab,
                   show_xlabel=(lab == "malignant"))
    draw_celltype(ax_b, ct)

    # Title block.
    fig.text(fx(0.12), fy(0.08), ("Surface-protein correlations with the RNA-only PC8 score are small "
                                  "and mostly reflect cell type"),
             fontsize=13.0, fontweight="bold", color=rs.INK, ha="left", va="top")
    fig.text(fx(0.12), fy(0.40), ("CD34 rises with PC8 in all 7 patients but barely within a cell type; "
                                  "in malignant cells the correlations are weaker and near the isotype noise floor"),
             fontsize=FS + 0.8, color=rs.INK_2, ha="left", va="top")

    # Panel headers.
    for x0, letter, text in ((0.12, "A", "Which surface proteins track the factor?"),
                             (5.75, "B", "Where the factor sits by cell type")):
        fig.text(fx(x0), fy(0.88), letter, fontsize=12, fontweight="bold", color=rs.INK, ha="left",
                 va="baseline")
        fig.text(fx(x0 + 0.24), fy(0.88), text, fontsize=FS + 1.6, fontweight="bold", color=rs.INK,
                 ha="left", va="baseline")

    # Panel A legend (shared by both sub-panels): marks in column 1, references in column 2.
    handles = [
        Line2D([], [], **POOLED_STYLE),
        Line2D([], [], color=RANGE_COLOR, lw=1.3, marker="o", markersize=5.8, markerfacecolor=rs.INK,
               markeredgecolor=rs.SURFACE),
        Line2D([], [], **CT_STYLE),
        Patch(facecolor=BAND_COLOR, edgecolor="none"),
        Line2D([], [], color=rs.MUTED, lw=0.8, ls=(0, (3, 2))),
        Line2D([], [], linestyle="none"),
    ]
    labels = [
        "pooled, all cells (as in earlier table)",
        "within patient: median;\nline = range over the 7 patients",
        "within patient × cell type: median",
        "isotype floor for dots (within patient)",
        "isotype floor for squares (pooled;\n= earlier table's isotype max |ρ|)",
        "bold protein = in earlier table",
    ]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(fx(0.30), fy(0.98)), ncol=2, frameon=False,
               fontsize=FS_SMALL - 0.2, handlelength=1.6, handletextpad=0.5, columnspacing=1.2,
               labelspacing=0.45, borderaxespad=0.0)
    iso_n = 9
    fig.text(fx(0.30), fy(1.80),
             (f"Isotype floor: the largest |ρ| reached by any of the {iso_n} isotype-control antibodies "
              "(no specific target).\nInside it = indistinguishable from non-specific antibody background; "
              "a rough guide, not a significance test."),
             fontsize=FS_NOTE, color=rs.INK_2, ha="left", va="top", linespacing=1.25)

    # Panel B legend and summary lines.
    n_hi = int(ct["malignant_higher_pooled"].sum())
    hb = [Line2D([], [], linestyle="none", marker="o", markersize=7.0, markerfacecolor=rs.SURFACE,
                 markeredgecolor=rs.MUTED, markeredgewidth=1.4),
          Line2D([], [], linestyle="none", marker="o", markersize=7.0, markerfacecolor=rs.INK_2,
                 markeredgecolor=rs.SURFACE)]
    fig.legend(hb, ["normal cells", "malignant cells"], loc="upper left", bbox_to_anchor=(fx(5.93), fy(0.98)),
               ncol=2, frameon=False, fontsize=FS_SMALL, handletextpad=0.3, columnspacing=1.4,
               borderaxespad=0.0)
    n_cmp = int(ct["n_patients_compared"].sum())
    n_cmp_hi = int(ct["n_patients_malignant_higher"].sum())
    all_cmp = n_cmp + ct.attrs["hidden_compared"]
    all_hi = n_cmp_hi + ct.attrs["hidden_higher"]
    fig.text(fx(5.99), fy(1.27),
             (f"Malignant cells score higher than normal cells of the\n"
              f"same type in {n_hi} of {len(ct)} types (patients pooled) and in\n"
              f"{n_cmp_hi} of {n_cmp} within-patient comparisons (≥10 cells of each;\n"
              f"{all_hi} of {all_cmp} including {ct.attrs['hidden_types']}, not shown)"),
             fontsize=FS_NOTE, color=rs.INK_2, ha="left", va="top", linespacing=1.25)

    fig.text(fx(0.12), fy(H - 0.06), ("Source: D27 factor_modality_assoc_v1. CITE-seq MRD cohort, 13 patients; "
                                      "within-patient values use the 7 patients with ≥10 malignant cells. "
                                      "PC8 = IFN-alpha PCA K=40 factor 8, fit on RNA only."),
             fontsize=FS_SMALL - 1.0, color=rs.MUTED, ha="left", va="bottom")
    return fig


def main() -> None:
    rs.apply()
    adt = load_adt()
    cs = load_card_summary()
    assert int(cs.loc[CARDS["PC8"], "orientation_sign"]) == -1  # D18 orientation, as in the report
    floors = isotype_floors(adt, cs)
    strata = load_strata()
    table = build_table(select_markers(adt), floors)
    ct = celltype_table()

    fig = make_figure(table, floors, strata, ct)
    fig.savefig(HERE / f"{OUT_STEM}.png", dpi=200)
    fig.savefig(HERE / f"{OUT_STEM}.pdf")
    plt.close(fig)

    num = table.select_dtypes("float").columns
    table[num] = table[num].round(3)
    table.to_csv(HERE / "protein_association_table.csv", index=False)
    ct_out = ct.rename(columns={"mean_oriented_z_normal": "mean_pc8_score_normal",
                                "mean_oriented_z_malignant": "mean_pc8_score_malignant"})
    ct_out = ct_out[["cell_type", "n_cells_normal", "n_cells_malignant", "mean_pc8_score_normal",
                     "mean_pc8_score_malignant", "malignant_higher_pooled", "n_patients_compared",
                     "n_patients_malignant_higher"]]
    ct_out[["mean_pc8_score_normal", "mean_pc8_score_malignant"]] = (
        ct_out[["mean_pc8_score_normal", "mean_pc8_score_malignant"]].round(3))
    ct_out.to_csv(HERE / "protein_association_celltype.csv", index=False)

    pd.set_option("display.width", 250)
    print(floors.round(3).to_string())
    print(strata)
    print(table.drop(columns=["factor_description", "adt_feature"]).to_string())
    print(ct_out.to_string())
    print("types not shown (fewer than 20 malignant cells pooled):", ct.attrs["hidden_types"],
          f"{ct.attrs['hidden_higher']}/{ct.attrs['hidden_compared']} malignant higher")
    print("normal-cell PC8 means, highest 3 of all types:", ct.attrs["normal_top3"])


if __name__ == "__main__":
    main()
