"""Figures for section 02 (do the highest-prevalence patients shape the factors?).

Report v2 separates the two ideas that v1's Figure 3 put in one chart (A-20261009-14) and explains
the leave-out axis (A-20261009-13/14):

fig02a_who_dominates     (a) per patient: share of all cells (= share of the unweighted Stage 1 fit)
                         vs share of malignant cells (= share of the Stage 2 positive labels), with
                         the 1/13 patient-equal weight line and the P02+P03+P04 column;
                         (b) per patient: score-energy share / cell share on IFN-alpha PCA40 and FA40
                         (D16), 1 = proportional, P02+P03+P04 combined ratio labelled (1.11, 1.02);
                         (c) the P02+P03+P04 ratio on all 320 stored PCA/FA bases, IFN-alpha K40
                         marked.
fig02b_axis_leverage     (a, b) IFN-alpha K40 leave-out refits (D17): axis loss of the reference
                         axis (PC8 / FA7) per leave-out set with its 3 size-matched and 20
                         composition-matched random-drop controls; bottom axis in percent of the
                         axis's variance, top axis as R = sqrt(1 - loss); matched |r| on key rows.
                         (c) ratio to the composition-matched control mean at K 10/20/40, filled
                         when the set is above all 20 controls (rank p = 1/21), with the count of
                         such method x K cells per set (the D26 decision rule, applied to D17).
fig02c_celltype_adjusted (a) D28: per patient and for P02+P03+P04, the score-energy ratio before
                         (observed, D16) and after the cell-type adjustment (residual = observed /
                         composition-only expected), median and IQR over the 320 bases, plus the
                         P02+P03+P04 IFN-alpha PCA40 / FA40 values; (b) energy per cell of each
                         predicted cell type relative to the cohort's (median, IQR over bases),
                         with the cell-type mix of P01 and of P02+P03+P04 printed underneath.
The D26 leave-out figures (allograft rejection, apoptosis) moved to figs_02b_d26.py (section 2b).
main() prints every number the section quotes that it can recompute from the same artifacts, so the
section's ledger can be checked against them.

Data (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/factor_patient_matrix_v1/                 (D16)
      factor_patient_matrix_representation_patient_summary.csv
      factor_patient_matrix_representation_summary.csv
  analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/              (D17)
      d17_alignment.csv, d17_leverage_contrast.csv
  analysis/scorecards/stage2_diagnostic/factor_patient_matrix_celltype_v1/        (D28)
      factor_patient_matrix_celltype_v1_distribution_summary.csv
      factor_patient_matrix_celltype_v1_representation_patient.csv.gz
      factor_patient_matrix_celltype_v1_celltype_summary.csv

Run from the repo root or the report dir:
  python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_02_shape.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

import report_style as rs  # noqa: E402

IFNA = "single_geneset__hallmark_interferon_alpha_response"
PATIENTS = tuple(f"P{i:02d}" for i in range(1, 14))
HP = rs.HIGH_PREVALENCE
GROUP = "P02+P03+P04"
N_CELLS = 60118
N_MALIGNANT = 1569
N_PATIENTS = 13

# Two neutral grays for the label-free (cells) and the label (malignant cells) series.
CELLS_FILL = rs.AXIS
MALIGNANT_FILL = rs.INK_2

D28_SLUG = "factor_patient_matrix_celltype_v1"
# D28 primary variant (= the D16 definition): raw cohort-centred scores, cohort cell-type averages,
# predicted cell type as the stratum.
D28_PRIMARY = {"weighting": "ss_weighted", "reference": "cohort", "strata": "celltype"}

RANK_P_FLOOR = 1 / 21  # one-sided rank p with 20 controls when the set is above all of them


# ----------------------------------------------------------------------------- data


def load_rep_patient() -> pd.DataFrame:
    path = rs.diag_dir("factor_patient_matrix_v1") / "factor_patient_matrix_representation_patient_summary.csv"
    df = pd.read_csv(path)
    n_rep = df.groupby(["stage0_panel_id", "stage1_method", "k"]).ngroups
    assert n_rep == 320, n_rep
    assert set(df["patient"]) == set(PATIENTS)
    return df


def load_rep_summary() -> pd.DataFrame:
    path = rs.diag_dir("factor_patient_matrix_v1") / "factor_patient_matrix_representation_summary.csv"
    df = pd.read_csv(path)
    assert len(df) == 320, len(df)
    return df


def ifna_k40_shares(rp: pd.DataFrame) -> pd.DataFrame:
    """Per patient (+ the P02+P03+P04 row): % of cells, % of malignant cells, % of PCA40 / FA40
    SS-weighted score energy and the ratio energy share / cell share."""
    q = rp[(rp["stage0_panel_id"] == IFNA) & (rp["k"] == 40)]
    pca = q[q["stage1_method"] == "pca"].set_index("patient").loc[list(PATIENTS)]
    fa = q[q["stage1_method"] == "fa"].set_index("patient").loc[list(PATIENTS)]
    assert (pca["n_cells"].to_numpy() == fa["n_cells"].to_numpy()).all()
    assert int(pca["n_cells"].sum()) == N_CELLS and int(pca["n_malignant"].sum()) == N_MALIGNANT
    out = pd.DataFrame(
        {
            "n_cells": pca["n_cells"],
            "n_malignant": pca["n_malignant"],
            "cells_pct": 100 * pca["n_cells"] / pca["n_cells"].sum(),
            "malignant_pct": 100 * pca["n_malignant"] / pca["n_malignant"].sum(),
            "pca40_energy_pct": 100 * pca["weighted_ss_share"],
            "fa40_energy_pct": 100 * fa["weighted_ss_share"],
            "pca40_ratio": pca["weighted_enrichment"],
            "fa40_ratio": fa["weighted_enrichment"],
        }
    )
    assert np.allclose(out["cells_pct"], 100 * pca["cell_share"])
    # ratio = energy share / cell share (D16 weighted_enrichment)
    assert np.allclose(out["pca40_ratio"], out["pca40_energy_pct"] / out["cells_pct"])
    hp = out.loc[list(HP)].sum()
    grp = {
        "n_cells": hp["n_cells"], "n_malignant": hp["n_malignant"], "cells_pct": hp["cells_pct"],
        "malignant_pct": hp["malignant_pct"], "pca40_energy_pct": hp["pca40_energy_pct"],
        "fa40_energy_pct": hp["fa40_energy_pct"],
        "pca40_ratio": hp["pca40_energy_pct"] / hp["cells_pct"],
        "fa40_ratio": hp["fa40_energy_pct"] / hp["cells_pct"],
    }
    out.loc[GROUP] = pd.Series(grp)
    # Cross-check against the D16 findings (8.2% / 9.1% / 8.4% / 68.3%; ratios 1.11 / 1.02).
    g = out.loc[GROUP]
    assert round(g["cells_pct"], 1) == 8.2 and round(g["pca40_energy_pct"], 1) == 9.1
    assert round(g["fa40_energy_pct"], 1) == 8.4 and round(g["malignant_pct"], 1) == 68.3
    assert round(g["pca40_ratio"], 2) == 1.11 and round(g["fa40_ratio"], 2) == 1.02
    return out


def load_alignment() -> pd.DataFrame:
    return pd.read_csv(rs.diag_dir("basis_variant_pilot_ifna_v1") / "d17_alignment.csv")


def load_leverage() -> pd.DataFrame:
    df = pd.read_csv(rs.diag_dir("basis_variant_pilot_ifna_v1") / "d17_leverage_contrast.csv")
    lo = df[df["arm"] == "leave_out"]
    assert (lo["n_label_matched_controls"] == 20).all()
    return df


def load_d28():
    """D28 tables (primary variant): distribution over the 320 bases, IFN-alpha K40 rows, cell types."""
    base = rs.diag_dir(D28_SLUG)
    dist = pd.read_csv(base / f"{D28_SLUG}_distribution_summary.csv")
    sel = np.ones(len(dist), bool)
    for key, val in D28_PRIMARY.items():
        sel &= dist[key] == val
    dist_all = dist[sel & (dist["scope"] == "all")].set_index("unit")
    assert len(dist_all) == 14 and (dist_all["n_representations"] == 320).all()
    rp = pd.read_csv(base / f"{D28_SLUG}_representation_patient.csv.gz")
    sel = (rp["stage0_panel_id"] == IFNA) & (rp["k"] == 40) & (rp["scope"] == "all")
    for key, val in D28_PRIMARY.items():
        sel &= rp[key] == val
    ifna = rp[sel].set_index(["stage1_method", "unit"])
    assert len(ifna) == 28
    ct = pd.read_csv(base / f"{D28_SLUG}_celltype_summary.csv")
    ct = ct[(ct["scope"] == "all") & (ct["reference"] == "cohort")]
    return dist_all, ifna, ct


# ----------------------------------------------------------------------------- helpers


def _patient_positions():
    """x positions: 13 patients, a gap, then the P02+P03+P04 column."""
    x = {p: float(i) for i, p in enumerate(PATIENTS)}
    x[GROUP] = len(PATIENTS) + 0.55
    return x


def _patient_axis(ax, x, labels=None, *, group_label="P02+P03+P04\n(combined)"):
    keys = list(PATIENTS) + [GROUP]
    ax.set_xticks([x[k] for k in keys])
    ax.set_xticklabels([group_label if k == GROUP else k for k in keys] if labels is None else labels)
    for lab, k in zip(ax.get_xticklabels(), keys):
        if k in HP or k == GROUP:
            lab.set_fontweight("bold")
            lab.set_color(rs.INK)
        elif k not in rs.SUPPORTED:
            lab.set_color(rs.MUTED)
    ax.set_xlim(-0.6, x[GROUP] + 0.75)
    ax.axvline(len(PATIENTS) - 0.2, color=rs.GRID, lw=0.8, zorder=1)


def _panel_title(ax, text, *, pad=4):
    ax.set_title(text, fontsize=8.5, pad=pad, loc="left")


# ----------------------------------------------------------------------------- fig02a


def fig02a_who_dominates(shares: pd.DataFrame, rsum: pd.DataFrame):
    rs.apply()
    fig = plt.figure(figsize=(rs.FULL_W, 6.15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.1, 0.62], hspace=0.7)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))
    x = _patient_positions()
    keys = list(PATIENTS) + [GROUP]
    xs = np.array([x[k] for k in keys])

    # (a) cells vs malignant cells -------------------------------------------------------------
    w, off = 0.36, 0.19
    ax_a.bar(xs - off, shares.loc[keys, "cells_pct"], width=w, color=CELLS_FILL, lw=0, zorder=3,
             label="share of all cells = share of the Stage 1 fit (no labels, every cell weight 1)")
    ax_a.bar(xs + off, shares.loc[keys, "malignant_pct"], width=w, color=MALIGNANT_FILL, lw=0, zorder=3,
             label="share of malignant cells = share of the Stage 2 positive labels")
    eq = 100 / N_PATIENTS
    ax_a.plot([-0.6, len(PATIENTS) - 0.4], [eq, eq], color=rs.INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=4)
    ax_a.text(3.7, eq + 1.3, f"patient-equal weighting:\nevery patient 1/13 = {eq:.1f}%",
              fontsize=6.7, color=rs.INK_2, va="bottom", ha="left", linespacing=1.15)
    gx = x[GROUP]
    ax_a.plot([gx - off - w / 2 - 0.05, gx - off + w / 2 + 0.05], [3 * eq, 3 * eq], color=rs.INK_2, lw=0.9,
              ls=(0, (4, 2.5)), zorder=4)
    g = shares.loc[GROUP]
    ax_a.text(gx - off - w / 2 - 0.1, 3 * eq, f"{3 * eq:.1f}% under\nequal weights", fontsize=6.4,
              color=rs.INK_2, ha="right", va="center", linespacing=1.15)
    ax_a.text(gx - off + w / 2 - 0.02, g["cells_pct"] + 1.0, f"{g['cells_pct']:.1f}%", ha="right", va="bottom",
              fontsize=6.6, color=rs.INK_2)
    ax_a.text(gx + off, g["malignant_pct"] + 1.0, f"{g['malignant_pct']:.1f}%", ha="center", va="bottom",
              fontsize=6.6, color=rs.INK_2)
    p03 = shares.loc["P03"]
    ax_a.text(x["P03"] + off + 0.25, p03["malignant_pct"] - 1.5, f"{p03['malignant_pct']:.1f}%", ha="left",
              va="top", fontsize=6.6, color=rs.INK_2)
    # Above the dashed patient-equal line, so the line does not strike through the label.
    ax_a.text(x["P03"] - off, max(p03["cells_pct"], eq) + 1.0, f"{p03['cells_pct']:.1f}%", ha="center",
              va="bottom", fontsize=6.2, color=rs.INK_2,
              bbox=dict(boxstyle="square,pad=0.1", fc=rs.SURFACE, ec="none"))
    zero = [p for p in PATIENTS if shares.loc[p, "n_malignant"] == 0]
    for p in zero:
        ax_a.text(x[p] + off, 0.8, "0", ha="center", va="bottom", fontsize=6.2, color=rs.INK_2)
    z_pct = shares.loc[zero, "cells_pct"].sum()
    ax_a.text(9.5, 46, f"No malignant cells: {', '.join(zero)}\n"
              f"({z_pct:.1f}% of the cells in the Stage 1 fit)", ha="center", va="center", fontsize=6.8,
              color=rs.INK_2, linespacing=1.2)
    ax_a.set_ylim(0, 78)
    ax_a.set_yticks([0, 20, 40, 60])
    ax_a.set_ylabel("% of cohort total")
    _patient_axis(ax_a, x)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), fontsize=6.8, handlelength=1.0,
                handleheight=0.8, borderaxespad=0.1, labelspacing=0.3)
    _panel_title(ax_a, "(a) Cells (what Stage 1 sees) vs malignant labels (what Stage 2 learns from)")

    # (b) energy share / cell share, IFN-alpha K40 ---------------------------------------------
    d = 0.17
    for col, m, dx in (("pca40_ratio", "pca", -d), ("fa40_ratio", "fa", d)):
        vals = shares.loc[keys, col].to_numpy()
        for xi, v in zip(xs + dx, vals):
            ax_b.plot([xi, xi], [1.0, v], color=rs.METHOD[m], lw=1.3, zorder=3, solid_capstyle="butt")
        ax_b.plot(xs + dx, vals, linestyle="none", marker="o", ms=rs.MARKER - 0.5, color=rs.METHOD[m],
                  markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING * 0.8, zorder=4,
                  label=f"{rs.METHOD_LABEL[m]}40 (IFN-α)")
    ax_b.axhline(1.0, color=rs.INK_2, lw=0.9, zorder=2)
    # Label the combined values next to their own dots (PCA above, FA to the right), in the method
    # colour, so they cannot be read as P13's.
    v_pca, v_fa = shares.loc[GROUP, "pca40_ratio"], shares.loc[GROUP, "fa40_ratio"]
    ax_b.text(gx - d, v_pca + 0.045, f"{v_pca:.2f}", ha="center", va="bottom", fontsize=7.2,
              color=rs.METHOD["pca"], fontweight="bold")
    ax_b.text(gx + d + 0.11, v_fa, f"{v_fa:.2f}", ha="left", va="center", fontsize=7.2,
              color=rs.METHOD["fa"], fontweight="bold")
    ax_b.set_ylim(0.55, 1.56)
    ax_b.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax_b.set_ylabel("energy share ÷ cell share\n(1 = proportional)")
    _patient_axis(ax_b, x)
    ax_b.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=2, fontsize=6.8, handlelength=1.0,
                borderaxespad=0.1, columnspacing=1.2)
    _panel_title(ax_b, "(b) Score-energy ratio, IFN-α, K = 40: share of score energy ÷ share of cells")

    # (c) the P02+P03+P04 ratio on all 320 bases ------------------------------------------------
    rng = np.random.default_rng(0)
    rows = {"pca": 1.0, "fa": 0.0}
    for m, y0 in rows.items():
        v = rsum.loc[rsum["stage1_method"] == m, "hp_enrichment_weighted"].to_numpy()
        jit = rng.uniform(-0.2, 0.2, size=v.size)
        ax_c.plot(v, y0 + jit, linestyle="none", marker="o", ms=2.6, color=rs.METHOD[m], alpha=0.45,
                  markeredgewidth=0, zorder=3)
        ifna = rsum[(rsum["stage1_method"] == m) & (rsum["stage0_panel_id"] == IFNA) & (rsum["k"] == 40)]
        iv = float(ifna["hp_enrichment_weighted"].iloc[0])
        ax_c.plot([iv], [y0], linestyle="none", marker="o", ms=rs.MARKER + 0.5, markerfacecolor="none",
                  markeredgecolor=rs.INK, markeredgewidth=1.2, zorder=5)
        rank = int((rsum["hp_enrichment_weighted"] > iv).sum()) + 1
        lab = f"IFN-α {rs.METHOD_LABEL[m]}40: {iv:.2f}"
        if m == "pca":
            lab += f" ({rank}rd highest of 320)" if rank == 3 else f" (rank {rank} of 320)"
        # Labels sit outside the jittered strips (PCA above its strip, FA between the strips), on an opaque box.
        ax_c.text(iv + (-0.012 if m == "pca" else 0.012), y0 + (0.5 if m == "pca" else 0.42), lab, fontsize=6.6,
                  color=rs.INK, va="center", ha="right" if m == "pca" else "left",
                  bbox=dict(boxstyle="square,pad=0.1", fc=rs.SURFACE, ec="none"), zorder=6)
    allv = rsum["hp_enrichment_weighted"]
    med = float(allv.median())
    ax_c.axvline(1.0, color=rs.INK_2, lw=0.9, zorder=2)
    ax_c.plot([med, med], [-0.42, 1.42], color=rs.INK_2, lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax_c.text(med - 0.008, -0.62, f"median {med:.2f}", fontsize=6.6, color=rs.INK_2, ha="right", va="center")
    ax_c.set_yticks([1.0, 0.0])
    ax_c.set_yticklabels(["PCA bases (160)", "FA bases (160)"])
    ax_c.set_ylim(-0.8, 1.75)
    ax_c.set_xlim(0.55, 1.2)
    ax_c.grid(axis="y", visible=False)
    ax_c.grid(axis="x", color=rs.GRID, lw=0.6)
    ax_c.set_xlabel("P02+P03+P04 energy share ÷ cell share (one dot per basis: 41 panels × K 5/10/20/40)",
                    fontsize=7.6)
    _panel_title(ax_c, "(c) P02+P03+P04 ratio on all 320 stored bases: IFN-α PCA40 is near the top")
    return fig


# ----------------------------------------------------------------------------- fig02b

LEAVE_OUT_ROWS = (
    ("P02", "P02"),
    ("P03", "P03"),
    ("P04", "P04"),
    ("P02-P03-P04", "P02+P03+P04"),
    ("P01", "P01"),
    ("P06", "P06"),
    ("P09", "P09"),
    ("P01-P09", "P01+P09"),
)
K_MARKERS = {10: "^", 20: "s", 40: "o"}
# rows whose matched |r| is printed next to the observed dot (method -> set tags)
R_LABEL_ROWS = {"pca": ("P02-P03-P04", "P01", "P01-P09"), "fa": ("P02-P03-P04", "P09")}


def _row_positions():
    ys, y = [], 0.0
    for i, _ in enumerate(LEAVE_OUT_ROWS):
        if i == 4:
            y += 0.9  # gap (and header room) between the highest-prevalence and the contrast block
        ys.append(y)
        y += 1.0
    return np.array(ys)


def _loss_axis(ax):
    """Log axis loss with percent labels below and R = sqrt(1 - loss) on a top axis."""
    ax.set_xscale("log")
    ax.set_xlim(1.5e-5, 0.06)
    major = [1e-4, 1e-3, 1e-2]
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.set_xticklabels(["0.01%", "0.1%", "1%"])
    ax.grid(axis="x", which="major", color=rs.GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    top = ax.secondary_xaxis("top")
    top.xaxis.set_major_locator(FixedLocator(major))
    top.xaxis.set_minor_locator(NullLocator())
    top.xaxis.set_minor_formatter(NullFormatter())
    r_labels = ["0.99995", "0.9995", "0.995"]  # R = sqrt(1 - loss) at the three major ticks
    assert np.allclose([np.sqrt(1 - v) for v in major], [float(t) for t in r_labels], atol=2e-5)
    top.set_xticklabels(r_labels, fontsize=6.4, color=rs.INK_2)
    top.tick_params(axis="x", length=2.5, pad=1.5, colors=rs.MUTED, labelcolor=rs.INK_2)
    top.spines["top"].set_color(rs.AXIS)
    return top


def fig02b_axis_leverage(align: pd.DataFrame, lev: pd.DataFrame, k: int = 40):
    rs.apply()
    fig = plt.figure(figsize=(rs.FULL_W, 4.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.92], wspace=0.09)
    ax_p = fig.add_subplot(gs[0])
    ax_f = fig.add_subplot(gs[1], sharey=ax_p)
    ax_r = fig.add_subplot(gs[2], sharey=ax_p)
    ys = _row_positions()
    dy = 0.27
    lo = lev[lev["arm"] == "leave_out"]
    for ax, m in ((ax_p, "pca"), (ax_f, "fa")):
        colour = rs.METHOD[m]
        for (tag, _), y in zip(LEAVE_OUT_ROWS, ys):
            obs = lo[(lo["stage1_method"] == m) & (lo["effective_k"] == k) & (lo["dropped"] == tag)]
            assert len(obs) == 1
            obs = obs.iloc[0]
            size = align[(align["arm"] == "leave_out_random") & (align["stage1_method"] == m)
                          & (align["effective_k"] == k) & (align["matches_drop_set"] == tag)]["axis_loss_fit"]
            comp = align[(align["arm"] == "leave_out_label_random") & (align["stage1_method"] == m)
                          & (align["effective_k"] == k) & (align["matches_drop_set"] == tag)]["axis_loss_fit"]
            assert len(size) == 3 and len(comp) == 20
            assert np.isclose(comp.mean(), obs["axis_loss_fit_label_matched_mean"])
            assert np.isclose(size.mean(), obs["axis_loss_fit_random_mean"])
            ss = np.sort(size.to_numpy())
            for seg in np.split(ss, np.where(ss[1:] / ss[:-1] > 4.0)[0] + 1):
                ax.plot([seg.min(), seg.max()], [y - dy, y - dy], color=rs.AXIS, lw=1.4, zorder=2,
                        solid_capstyle="round")
            ax.plot(size, np.full(3, y - dy), linestyle="none", marker="|", ms=5, mew=1.1,
                    color=rs.MUTED, zorder=3)
            # one segment per mode: FA40's controls are bimodal (a second mode at 1.1-1.8%), and one line
            # from the lowest to the highest control made P01 and P09 look inside their control range
            cs = np.sort(comp.to_numpy())
            cuts = np.where(cs[1:] / cs[:-1] > 4.0)[0]
            for seg in np.split(cs, cuts + 1):
                ax.plot([seg.min(), seg.max()], [y + dy, y + dy], color=rs.MUTED, lw=1.0, zorder=2,
                        solid_capstyle="round")
            ax.plot(comp, np.full(20, y + dy), linestyle="none", marker="o", ms=2.6,
                    color=rs.INK_2, alpha=0.75, markeredgewidth=0, zorder=3)
            ax.plot([obs["axis_loss_fit"]], [y], linestyle="none", marker="o", ms=rs.MARKER + 0.5,
                    color=colour, markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, zorder=5)
            if tag in R_LABEL_ROWS[m]:
                xo = obs["axis_loss_fit"]
                right = xo < 6e-3
                ax.text(xo * (1.45 if right else 1 / 1.45), y, f"|r| {obs['matched_abs_r_fit']:.3f}"
                        if obs["matched_abs_r_fit"] > 0.9 else f"|r| {obs['matched_abs_r_fit']:.2f}",
                        ha="left" if right else "right", va="center", fontsize=6.5, color=rs.INK,
                        zorder=6, bbox=dict(boxstyle="square,pad=0.08", fc=rs.SURFACE, ec="none"))
        top = _loss_axis(ax)
        top.set_xlabel("R = √(1 − loss)", fontsize=6.6, labelpad=2, color=rs.INK_2)
        ax.set_xlabel("Axis loss = 1 − R² (log)\n(% of the axis's variance lost)", fontsize=7.4)
        ref = "PC8" if m == "pca" else "FA7"
        ax.set_title(f"({'a' if m == 'pca' else 'b'}) {rs.METHOD_LABEL[m]}, K = {k}: axis {ref}",
                     fontsize=8.3, pad=20, loc="left")
    plt.setp(ax_f.get_yticklabels(), visible=False)
    plt.setp(ax_r.get_yticklabels(), visible=False)

    # (c) ratio to the composition-matched mean at every K, filled = above all 20 controls ------
    ry = {"pca": -0.15, "fa": 0.15}
    for (tag, _), y in zip(LEAVE_OUT_ROWS, ys):
        n_above = 0
        for m in ("pca", "fa"):
            for kk, mk in K_MARKERS.items():
                r = lo[(lo["stage1_method"] == m) & (lo["effective_k"] == kk) & (lo["dropped"] == tag)].iloc[0]
                ratio = r["axis_loss_fit_ratio_to_label_matched_mean"]
                above = r["axis_loss_fit_label_matched_rank_p"] <= RANK_P_FLOOR + 1e-9
                n_above += int(above)
                ax_r.plot([ratio], [y + ry[m]], linestyle="none", marker=mk, ms=4.6,
                          markerfacecolor=rs.METHOD[m] if above else rs.SURFACE,
                          markeredgecolor=rs.METHOD[m], markeredgewidth=1.0, zorder=4)
        ax_r.text(1.03, y, f"{n_above}/6", transform=blended_transform_factory(ax_r.transAxes, ax_r.transData),
                  ha="left", va="center", fontsize=7, clip_on=False,
                  color=rs.INK, fontweight="bold" if tag == "P02-P03-P04" else "normal")
    ax_r.axvline(1.0, color=rs.INK_2, lw=0.9, zorder=2)
    ax_r.set_xscale("log")
    ax_r.set_xlim(0.02, 40)
    ax_r.xaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    ax_r.set_xticklabels(["0.1×", "1×", "10×"])
    ax_r.grid(axis="x", which="major", color=rs.GRID, lw=0.6)
    ax_r.grid(axis="y", visible=False)
    ax_r.set_xlabel("Axis loss ÷ mean of its 20\ncomposition-matched controls (log)", fontsize=7.4)
    ax_r.set_title("(c) All K: ratio to controls", fontsize=8.3, pad=20, loc="left")
    ax_r.text(1.03, ys[0] - 0.55, "above\nall 20", transform=blended_transform_factory(ax_r.transAxes, ax_r.transData),
              ha="left", va="bottom", fontsize=6.2, color=rs.INK_2, clip_on=False,
              linespacing=1.1)

    ax_p.set_yticks(ys)
    labels = []
    for (tag, lab) in LEAVE_OUT_ROWS:
        r = lo[(lo["stage1_method"] == "pca") & (lo["effective_k"] == k) & (lo["dropped"] == tag)].iloc[0]
        labels.append(f"{lab}\n{int(r['n_dropped_cells']):,} cells, {int(r['n_dropped_malignant']):,} mal.")
    ax_p.set_yticklabels(labels, fontsize=7)
    ax_p.set_ylim(ys[-1] + 0.6, ys[0] - 0.95)
    for i, text in ((0, "Highest-prevalence sets"), (4, "Contrast sets")):
        ax_p.text(1.7e-5, ys[i] - 0.68, text, ha="left", va="center", fontsize=7.0,
                  color=rs.INK, fontweight="bold")
    ax_f.text(3.6e-2, ys[0] - 0.68, "controls' second mode\n(1.1–1.8%)", ha="right", va="center", fontsize=5.9,
              color=rs.INK_2, linespacing=1.0)
    handles = [
        Line2D([], [], linestyle="none", marker="o", ms=rs.MARKER + 0.5, color=rs.METHOD["pca"],
               markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, label="refit without the set (PCA)"),
        Line2D([], [], linestyle="none", marker="o", ms=rs.MARKER + 0.5, color=rs.METHOD["fa"],
               markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, label="refit without the set (FA)"),
        Line2D([], [], color=rs.AXIS, lw=1.4, marker="|", ms=5, mew=1.1, markeredgecolor=rs.MUTED,
               label="3 size-matched drops (above dot)"),
        Line2D([], [], color=rs.MUTED, lw=1.0, marker="o", ms=2.6, markerfacecolor=rs.INK_2,
               markeredgewidth=0, label="20 composition-matched drops (below dot)"),
        Line2D([], [], linestyle="none", marker="^", ms=4.6, markerfacecolor=rs.INK_2,
               markeredgecolor=rs.INK_2, label="K = 10"),
        Line2D([], [], linestyle="none", marker="s", ms=4.6, markerfacecolor=rs.INK_2,
               markeredgecolor=rs.INK_2, label="K = 20"),
        Line2D([], [], linestyle="none", marker="o", ms=4.6, markerfacecolor=rs.INK_2,
               markeredgecolor=rs.INK_2, label="K = 40"),
        Line2D([], [], linestyle="none", marker="o", ms=4.6, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.INK_2, label="open: within control range"),
    ]
    leg_kw = dict(loc="lower center", ncol=2, fontsize=6.6, frameon=False, handlelength=1.6, columnspacing=1.0,
                  borderaxespad=0.0)
    fig.legend(handles=handles[:4], bbox_to_anchor=(0.36, -0.01), **leg_kw)
    leg_c = fig.legend(handles=handles[4:], bbox_to_anchor=(0.84, -0.01), title="(c)", title_fontsize=6.6,
                       **leg_kw)
    leg_c._legend_box.align = "left"
    hp_n = sum(int(lo[(lo["stage1_method"] == m) & (lo["effective_k"] == kk)
                      & (lo["dropped"] == "P02-P03-P04")].iloc[0]["axis_loss_fit_label_matched_rank_p"]
                   <= RANK_P_FLOOR + 1e-9) for m in ("pca", "fa") for kk in K_MARKERS)
    single = [sum(int(lo[(lo["stage1_method"] == m) & (lo["effective_k"] == kk) & (lo["dropped"] == t)]
                      .iloc[0]["axis_loss_fit_label_matched_rank_p"] <= RANK_P_FLOOR + 1e-9)
                  for m in ("pca", "fa") for kk in K_MARKERS) for t in ("P01", "P02", "P03", "P04", "P06", "P09")]
    fig.suptitle("Without P02+P03+P04 in the fit, the IFN-α malignant axis barely moves;\n"
                 f"it beats its matched controls in {hp_n} of 6 method × K cells "
                 f"(single patients: {min(single)}–{max(single)} of 6)",
                 x=0.01, ha="left", fontsize=9.2, fontweight="bold", y=1.0)
    fig.subplots_adjust(left=0.165, right=0.945, top=0.835, bottom=0.235)
    return fig


# ----------------------------------------------------------------------------- fig02c


def fig02c_celltype_adjusted(dist_all: pd.DataFrame, ifna: pd.DataFrame, ct: pd.DataFrame):
    rs.apply()
    fig = plt.figure(figsize=(rs.FULL_W, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.45)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) observed vs residual, median + IQR over 320 bases; then the IFN-alpha K40 group values ----
    x = {p: float(i) for i, p in enumerate(PATIENTS)}
    x[GROUP] = len(PATIENTS) + 0.55
    x_ifna = {"pca": len(PATIENTS) + 2.2, "fa": len(PATIENTS) + 3.55}
    d = 0.17
    obs_kw = dict(linestyle="none", marker="o", ms=rs.MARKER - 0.3, markerfacecolor=rs.SURFACE,
                  markeredgecolor=rs.MUTED, markeredgewidth=1.3, zorder=4)
    res_kw = dict(linestyle="none", marker="o", ms=rs.MARKER - 0.3, color=rs.INK, markeredgecolor=rs.SURFACE,
                  markeredgewidth=rs.RING * 0.7, zorder=5)
    for u in list(PATIENTS) + [GROUP]:
        r = dist_all.loc[u]
        ax_a.plot([x[u] - d] * 2, [r["obs_ratio_q25"], r["obs_ratio_q75"]], color=rs.MUTED, lw=1.2, zorder=3)
        ax_a.plot([x[u] - d], [r["obs_ratio_median"]], **obs_kw)
        ax_a.plot([x[u] + d] * 2, [r["resid_ratio_q25"], r["resid_ratio_q75"]], color=rs.INK, lw=1.2, zorder=3)
        ax_a.plot([x[u] + d], [r["resid_ratio_median"]], **res_kw)
    for m, xx in x_ifna.items():
        r = ifna.loc[(m, GROUP)]
        ax_a.plot([xx - d], [r["obs_ratio"]], **{**obs_kw, "markeredgecolor": rs.METHOD[m]})
        ax_a.plot([xx + d], [r["resid_ratio"]], **{**res_kw, "color": rs.METHOD[m]})
        ax_a.text(xx, 0.735, f"{r['obs_ratio']:.2f}\n→ {r['resid_ratio']:.2f}", ha="center", va="center",
                  fontsize=6.6, color=rs.INK)
    p01, grp = dist_all.loc["P01"], dist_all.loc[GROUP]
    ax_a.annotate(f"P01: {p01['obs_ratio_median']:.2f} → {p01['resid_ratio_median']:.2f}\n"
                  "(mostly matched by its mix)",
                  xy=(x["P01"] - d, p01["obs_ratio_median"]), xytext=(x["P01"] + 0.55, 1.39),
                  fontsize=6.7, color=rs.INK, va="center", ha="left",
                  arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.7, shrinkA=1, shrinkB=3))
    ax_a.text(x[GROUP], 0.735, f"{grp['obs_ratio_median']:.2f}\n→ {grp['resid_ratio_median']:.2f}",
              ha="center", va="center", fontsize=6.6, color=rs.INK)
    p02 = dist_all.loc["P02"]
    ax_a.annotate(f"P02: {p02['obs_ratio_median']:.2f} → {p02['resid_ratio_median']:.2f}\n(not its mix)",
                  xy=(x["P02"] + d, p02["resid_ratio_median"]), xytext=(x["P02"] + 1.0, 1.29),
                  fontsize=6.7, color=rs.INK, va="center", ha="left",
                  arrowprops=dict(arrowstyle="-", color=rs.INK_2, lw=0.7, shrinkA=1, shrinkB=3))
    ax_a.axhline(1.0, color=rs.INK_2, lw=0.9, zorder=2)
    ax_a.axvline(len(PATIENTS) - 0.2, color=rs.GRID, lw=0.8, zorder=1)
    ax_a.axvline(len(PATIENTS) + 1.4, color=rs.GRID, lw=0.8, zorder=1)
    ticks = list(PATIENTS) + [GROUP]
    ax_a.set_xticks([x[t] for t in ticks] + list(x_ifna.values()))
    ax_a.set_xticklabels(ticks[:-1] + ["P02+P03\n+P04"] + ["IFN-α\nPCA40", "IFN-α\nFA40"], fontsize=7)
    for lab, t in zip(ax_a.get_xticklabels(), ticks + ["i1", "i2"]):
        if t in HP or t == GROUP:
            lab.set_fontweight("bold")
            lab.set_color(rs.INK)
        elif t in PATIENTS and t not in rs.SUPPORTED:
            lab.set_color(rs.MUTED)
    ax_a.text((x_ifna["pca"] + x_ifna["fa"]) / 2, 1.505, "P02+P03+P04\non IFN-α", ha="center", va="top",
              fontsize=6.6, color=rs.INK_2, fontweight="bold", linespacing=1.1)
    ax_a.set_xlim(-0.6, x_ifna["fa"] + 0.6)
    ax_a.set_ylim(0.66, 1.52)
    ax_a.set_yticks([0.8, 1.0, 1.2, 1.4])
    ax_a.set_ylabel("energy share ÷ cell share")
    handles = [Line2D([], [], **{**obs_kw, "zorder": None}, label="observed ratio (no adjustment)"),
               Line2D([], [], **{**res_kw, "zorder": None},
                      label="adjusted: observed \u00f7 expected from the cell-type mix alone")]
    ax_a.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.795, 1.0), ncol=1, fontsize=6.7,
                handlelength=1.0, borderaxespad=0.15, labelspacing=0.3)
    _panel_title(ax_a, "(a) Observed vs cell-type-adjusted energy ratio (median, IQR over 320 bases)")

    # (b) cell-type energy levels and the two mixes ----------------------------------------------
    p01ct = ct[ct["unit"] == "P01"].set_index("cell_type")
    hpct = ct[ct["unit"] == GROUP].set_index("cell_type")
    order = p01ct.sort_values("type_energy_level_median", ascending=False).index.tolist()
    assert np.allclose(p01ct.loc[order, "type_energy_level_median"], hpct.loc[order, "type_energy_level_median"])
    xb = np.arange(len(order))
    lv = p01ct.loc[order, "type_energy_level_median"].to_numpy()
    q1 = p01ct.loc[order, "type_energy_level_q25"].to_numpy()
    q3 = p01ct.loc[order, "type_energy_level_q75"].to_numpy()
    colours = [rs.INK_2 if v >= 1 else rs.AXIS for v in lv]
    ax_b.bar(xb, lv - 1.0, bottom=1.0, width=0.62, color=colours, lw=0, zorder=3)
    ax_b.vlines(xb, q1, q3, color=rs.INK, lw=0.9, zorder=4)
    ax_b.axhline(1.0, color=rs.INK_2, lw=0.9, zorder=2)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.5, 6.0)
    ax_b.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax_b.yaxis.set_minor_locator(NullLocator())
    ax_b.set_yticklabels(["0.5×", "1×", "2×", "4×"])
    ax_b.set_ylabel("energy per cell ÷\ncohort's (log)")
    ax_b.set_xlim(-0.6, len(order) - 0.4)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([])
    tr = blended_transform_factory(ax_b.transData, ax_b.transAxes)
    rows = (("cell type", None, -0.07), ("P01: % of cells", p01ct, -0.24), ("P02+P03+P04: %", hpct, -0.39))
    for name, tab, yy in rows:
        ax_b.text(-0.75, yy, name, transform=tr, ha="right", va="center", fontsize=6.6, color=rs.INK_2,
                  fontweight="bold" if tab is None else "normal")
        for xi, c in zip(xb, order):
            if tab is None:
                ax_b.text(xi, yy, c.replace("HSC/", "HSC/\n"), transform=tr, ha="center", va="center",
                          fontsize=6.4, color=rs.INK, linespacing=1.0)
            else:
                v = 100 * tab.loc[c, "comp_share"]
                ax_b.text(xi, yy, f"{v:.1f}" if v < 10 else f"{v:.0f}", transform=tr, ha="center",
                          va="center", fontsize=6.6, color=rs.INK if v >= 10 else rs.INK_2,
                          fontweight="bold" if v >= 10 else "normal")
    _panel_title(ax_b, "(b) Why: energy per cell differs by predicted cell type (median, IQR over 320 bases)")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.95, bottom=0.14)
    return fig




# ----------------------------------------------------------------------------- text checks


def print_text_numbers(shares, rp, rsum, align, lev, dist_all, ifna, ct) -> None:
    """Recompute the numbers the section quotes from the same artifacts (for the ledger)."""
    print("== IFN-alpha K40 shares and ratios (D16)")
    print(shares.round(3).to_string())
    zero = [p for p in PATIENTS if shares.loc[p, "n_malignant"] == 0]
    uns = [p for p in PATIENTS if p not in rs.SUPPORTED]
    print(f"no-malignant patients {zero}: {shares.loc[zero, 'n_cells'].sum():.0f} cells "
          f"= {shares.loc[zero, 'cells_pct'].sum():.2f}%; unsupported {uns}: "
          f"{shares.loc[uns, 'n_cells'].sum():.0f} cells = {shares.loc[uns, 'cells_pct'].sum():.2f}%")
    print(f"patient-equal weight 1/13 = {100 / 13:.2f}%; P02+P03+P04 3/13 = {300 / 13:.2f}%")
    hp = rsum["hp_enrichment_weighted"]
    print(f"P02+P03+P04 ratio over 320 bases: median {hp.median():.2f}, range {hp.min():.2f}-{hp.max():.2f}, "
          f">1 in {(hp > 1).sum()}/320")
    for m in ("pca", "fa"):
        iv = float(rsum[(rsum.stage1_method == m) & (rsum.stage0_panel_id == IFNA) & (rsum.k == 40)]
                   ["hp_enrichment_weighted"].iloc[0])
        print(f"  IFN-alpha {m}40 {iv:.4f}, rank {(hp > iv).sum() + 1} of 320")
    print("PCA median by K:", rsum[rsum["stage1_method"] == "pca"].groupby("k")["hp_enrichment_weighted"]
          .median().round(2).to_dict())
    med = rp.groupby("patient")["weighted_enrichment"].agg(["median", lambda v: int((v > 1).sum())])
    print("per-patient median ratio (> 1 count):", med.round(2).to_dict("index"))

    print("== D17 leverage (IFN-alpha)")
    lo = lev[lev["arm"] == "leave_out"]
    for col in ("axis_loss_fit", "matched_abs_r_fit", "axis_loss_fit_label_matched_mean",
                "axis_loss_fit_label_matched_min", "axis_loss_fit_label_matched_max",
                "axis_loss_fit_ratio_to_label_matched_mean", "axis_loss_fit_label_matched_rank_p",
                "axis_loss_fit_random_mean", "axis_loss_fit_ratio_to_random_mean"):
        tab = lo.pivot_table(index="dropped", columns=["stage1_method", "effective_k"], values=col)
        print(f"-- {col}\n{tab.round(5).to_string()}")
    above = lo.assign(above=lo["axis_loss_fit_label_matched_rank_p"] <= RANK_P_FLOOR + 1e-9)
    print("method x K cells above all 20 composition-matched controls:",
          above.groupby("dropped")["above"].sum().to_dict())
    rat = lo.pivot_table(index="dropped", columns=["stage1_method", "effective_k"],
                         values="axis_loss_fit_ratio_to_label_matched_mean")
    n_rule2 = int((rat.loc["P02-P03-P04"] > rat.loc[["P01", "P06", "P09", "P01-P09"]].max()).sum())
    print(f"D26 rule 2 on IFN-alpha: P02+P03+P04 ratio above every contrast set in {n_rule2}/6 cells")
    c = align[(align["arm"] == "leave_out_label_random") & (align["effective_k"] == 40)
              & (align["stage1_method"] == "fa")]
    hi = c[c["axis_loss_fit"] > 0.005]
    print("FA40 controls in the upper mode per set:", hi.groupby("matches_drop_set").size().to_dict(),
          f"range {hi['axis_loss_fit'].min():.4f}-{hi['axis_loss_fit'].max():.4f}")
    nm = lev[lev["arm"] != "leave_out"][["stage1_method", "effective_k", "dropped", "axis_loss_fit",
                                         "axis_loss_fit_ratio_to_random_mean"]]
    print("normal-only (all malignant cells dropped):\n", nm.round(5).to_string())
    bw = align[align["arm"].isin(["balanced_weighted", "balanced_subsample"]) & (align["effective_k"] == 40)]
    print("balanced K40:\n", bw[["arm", "stage1_method", "variant", "n_fit_cells", "matched_abs_r_all",
                                 "many_to_one_r2_all"]].round(4).to_string())

    print("== D28 cell-type adjustment (primary variant, all cells)")
    cols = ["obs_ratio_median", "obs_ratio_q25", "obs_ratio_q75", "obs_ratio_n_gt1", "comp_ratio_median",
            "resid_ratio_median", "resid_ratio_q25", "resid_ratio_q75", "resid_ratio_n_gt1",
            "n_resid_closer_to_1", "frac_log_departure_composition_median"]
    print(dist_all[cols].round(3).to_string())
    print("IFN-alpha K40 (obs -> comp -> resid):")
    for (m, u), r in ifna.iterrows():
        if u in ("P01", "P02", "P07", GROUP):
            print(f"  {m} {u}: {r['obs_ratio']:.3f} -> {r['comp_ratio']:.3f} -> {r['resid_ratio']:.3f}")
    p01 = ct[ct["unit"] == "P01"].set_index("cell_type")
    hpc = ct[ct["unit"] == GROUP].set_index("cell_type")
    print("cell-type energy level (median, q25, q75); % of P01; P01 share of type; % of group:")
    for c_ in p01.sort_values("type_energy_level_median", ascending=False).index:
        print(f"  {c_:9s} {p01.loc[c_, 'type_energy_level_median']:.2f} ({p01.loc[c_, 'type_energy_level_q25']:.2f}-"
              f"{p01.loc[c_, 'type_energy_level_q75']:.2f}); P01 {100 * p01.loc[c_, 'comp_share']:.1f}% "
              f"(holds {100 * p01.loc[c_, 'unit_share_of_type']:.0f}%); group {100 * hpc.loc[c_, 'comp_share']:.1f}%")


# ----------------------------------------------------------------------------- main


def main() -> None:
    rp = load_rep_patient()
    rsum = load_rep_summary()
    shares = ifna_k40_shares(rp)
    align, lev = load_alignment(), load_leverage()
    dist_all, ifna, ct = load_d28()
    # D28's observed ratios are D16's (regression-checked in D28); re-check the IFN-alpha K40 rows here.
    for m in ("pca", "fa"):
        assert np.isclose(ifna.loc[(m, GROUP), "obs_ratio"], shares.loc[GROUP, f"{m}40_ratio"])
        for p in PATIENTS:
            assert np.isclose(ifna.loc[(m, p), "obs_ratio"], shares.loc[p, f"{m}40_ratio"])
    rs.save(fig02a_who_dominates(shares, rsum), "fig02a_who_dominates")
    rs.save(fig02b_axis_leverage(align, lev), "fig02b_axis_leverage")
    rs.save(fig02c_celltype_adjusted(dist_all, ifna, ct), "fig02c_celltype_adjusted")
    print_text_numbers(shares, rp, rsum, align, lev, dist_all, ifna, ct)


if __name__ == "__main__":
    main()
