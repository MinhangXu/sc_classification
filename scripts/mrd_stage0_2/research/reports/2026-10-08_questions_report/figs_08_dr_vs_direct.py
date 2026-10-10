"""Figures for section 08 (the three scientific questions).

fig08a_patient_concentration: per supported patient, the share of the 640 biological
    DR atlas cells whose held-out lift is > 0 and > 5 pp (Q1: where the lift sits).
fig08b_dr_vs_direct: per Stage 0 panel, best-DR median lift (best of up to 16 DR cells,
    chosen in-sample by median lift) against the direct-gene median lift (Q3).

Run from the report directory or the repo root:
    python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_08_dr_vs_direct.py
Inputs are read-only tables under EXP/analysis/scorecards/stage2_diagnostic/ (resolved via
report_style.diag_dir). main() also prints the numbers quoted in section 08.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_style as rs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# All-zero L1 folds score lift = constant-score AP - pi, which is ~1e-17, not exactly 0.
LIFT_EPS = 1e-9
LIFT_STRONG = 0.05  # 5 pp, the D1f cutoff

ATLAS_PATIENT = "old34_d1_atlas_supported_patient_metrics.csv"
D1E_TABLE = "old34_d1e_best_dr_vs_direct_gene_selection.csv"
D9_TABLE = "meeting_followup_v1/old34_direct_gene_vs_best_dr_summary.csv"

# Panels to label in fig08b: stage0_panel_id -> short label.
FOCUS = {
    "single_geneset__hallmark_allograft_rejection": "allograft rejection",
    "single_geneset__hallmark_apoptosis": "apoptosis",
    "single_geneset__hallmark_interferon_alpha_response": "IFN-\u03b1 response",
    "single_geneset__hallmark_tnfa_signaling_via_nfkb": "TNF-\u03b1 via NF-\u03baB",
}
FOCUS_LABELLED = (
    "single_geneset__hallmark_allograft_rejection",
    "single_geneset__hallmark_apoptosis",
    "single_geneset__hallmark_interferon_alpha_response",
)
CONTROL_LABEL = {
    "full_34": "full_34 union",
    "core_only": "core-only union",
    "hvg_top_requested_500__available_500": "HVG 500",
    "hvg_top_requested_1000__available_1000": "HVG 1,000",
    "hvg_top_requested_3000__available_3000": "HVG 3,000",
    "hvg_top_requested_10000__available_10000": "HVG 10,000",
}
METHOD_OF_CONFIG = {
    "pca": "pca",
    "fa": "fa",
    "factosig": "factosig",
    "factosig_promax": "factosig_promax",
}


def _ring(**kw):
    """Marker style with the surface ring."""
    base = dict(markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING, linestyle="none")
    base.update(kw)
    return base


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def patient_concentration() -> pd.DataFrame:
    """Per supported patient: share of biological DR atlas cells with lift > 0 and > 5 pp."""
    m = pd.read_csv(rs.diag_dir("") / ATLAS_PATIENT)
    bio_dr = m[(m["atlas_row_block"] == "biological") & (m["stage1_method"] != "direct_gene")]
    g = bio_dr.groupby("heldout_patient").agg(
        prevalence=("patient_prevalence", "first"),
        n_malignant=("n_malignant_h", "first"),
        n_cells=("representation_id", "nunique"),
        n_pos=("heldout_lift", lambda s: int((s > LIFT_EPS).sum())),
        n_strong=("heldout_lift", lambda s: int((s > LIFT_STRONG).sum())),
        median_lift=("heldout_lift", "median"),
    )
    g["share_pos"] = g["n_pos"] / g["n_cells"]
    g["share_strong"] = g["n_strong"] / g["n_cells"]
    return g.sort_values("prevalence", ascending=False)


def atlas_cell_patient_counts() -> pd.Series:
    """Per atlas cell (all 783): number of supported patients with lift > 5 pp (D1f check)."""
    m = pd.read_csv(rs.diag_dir("") / ATLAS_PATIENT)
    return m.groupby("representation_id")["heldout_lift"].apply(lambda s: int((s > LIFT_STRONG).sum()))


def dr_vs_direct() -> pd.DataFrame:
    """D1e: best DR (by median lift, in-sample, best of 16) vs direct genes, per panel, in pp."""
    d = pd.read_csv(rs.diag_dir("") / D1E_TABLE)
    d = d.assign(
        best_pp=100 * d["best_dr_median_lift"],
        direct_pp=100 * d["direct_median_lift"],
        best_p20_pp=100 * d["best_dr_p20_lift"],
        direct_p20_pp=100 * d["direct_p20_lift"],
        is_control=d["atlas_row_block"] != "biological",
    )
    return d


def estimand_counts() -> dict:
    """Counts of panels where best DR beats direct genes, per selection rule and estimand."""
    d9 = pd.read_csv(rs.diag_dir("") / D9_TABLE)
    e = dr_vs_direct()
    bio = e[~e["is_control"]]
    ctrl = e[e["is_control"]]
    return {
        "d9_n": len(d9),
        "d9_pooled_ap": int((d9["dr_value_added_pooled_lopo_ap"] > 0).sum()),
        "d9_median": int((d9["dr_value_added_median_lift"] > 0).sum()),
        "d9_p20": int((d9["dr_value_added_p20_lift"] > 0).sum()),
        "d9_median_value_added": (
            float(d9["dr_value_added_pooled_lopo_ap"].median()),
            float(d9["dr_value_added_median_lift"].median()),
            float(d9["dr_value_added_p20_lift"].median()),
        ),
        "d1e_bio_n": len(bio),
        "d1e_bio_median": int((bio["best_pp"] > bio["direct_pp"]).sum()),
        "d1e_bio_p20": int((bio["best_p20_pp"] > bio["direct_p20_pp"]).sum()),
        "d1e_ctrl_n": len(ctrl),
        "d1e_ctrl_median": int((ctrl["best_pp"] > ctrl["direct_pp"]).sum()),
        "d1e_ctrl_p20": int((ctrl["best_p20_pp"] > ctrl["direct_p20_pp"]).sum()),
        "d1e_all_median": int((e["best_pp"] > e["direct_pp"]).sum()),
        "d1e_all_n": len(e),
        "d1e_bio_median_gap_pp": float((bio["best_pp"] - bio["direct_pp"]).median()),
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def fig08a_patient_concentration() -> Path:
    g = patient_concentration()
    pats = list(g.index)
    x = np.arange(len(pats))
    w = 0.3  # two bars per slot, 0.6 of the slot in total
    c_pos, c_strong = rs.AXIS, rs.INK_2  # neutral pair: blue is reserved for PCA in this report

    fig, ax = plt.subplots(figsize=(rs.FULL_W, 2.75))
    gap = 0.012  # surface gap between the paired bars
    ax.bar(x - w / 2 - gap, 100 * g["share_pos"], width=w, color=c_pos, linewidth=0, label="lift > 0 (AP above \u03c0)")
    ax.bar(x + w / 2 + gap, 100 * g["share_strong"], width=w, color=c_strong, linewidth=0,
           label="lift > 5 pp")
    for xi, v in zip(x, g["share_strong"]):
        ax.text(xi + w / 2 + gap, 100 * v + 1.5, f"{100 * v:.0f}%" if v >= 0.01 else f"{100 * v:.1f}%",
                ha="center", va="bottom", fontsize=7, color=rs.INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nπ {100 * prev:.1f}%" for p, prev in zip(pats, g["prevalence"])])
    ax.set_xlim(-0.6, len(pats) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Share of the 640 biological\nDR atlas cells (%)")
    ax.set_xlabel("Held-out supported patient, ordered by malignant prevalence π")
    ax.set_title("Lift above π reaches every supported patient;\n"
                 "lift above 5 pp is concentrated in the three highest-prevalence patients", loc="left")

    # Bracket over the highest-prevalence patients (P03, P02, P04 are the first three).
    hp = [i for i, p in enumerate(pats) if p in rs.HIGH_PREVALENCE]
    y0 = 101
    ax.plot([min(hp) - 0.35, min(hp) - 0.35, max(hp) + 0.35, max(hp) + 0.35], [y0 - 2, y0, y0, y0 - 2],
            color=rs.INK_2, linewidth=0.8, solid_capstyle="butt")
    ax.text(np.mean(hp), y0 + 1.5, "highest prevalence (P02, P03, P04)", ha="center", va="bottom",
            fontsize=7.5, color=rs.INK_2)

    ax.legend(loc="upper right", ncol=2, handlelength=1.0, columnspacing=1.2, bbox_to_anchor=(1.0, 1.0))
    return rs.save(fig, "fig08a_patient_concentration")


def fig08b_dr_vs_direct() -> Path:
    d = dr_vs_direct()
    bio = d[~d["is_control"]]
    ctrl = d[d["is_control"]]
    counts = estimand_counts()

    fig, ax = plt.subplots(figsize=(rs.FULL_W, 4.35))
    lo, hi_x, hi_y = -1.5, 40.5, 22.0
    ax.set_xlim(lo, hi_x)
    ax.set_ylim(lo, hi_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, axis="both")

    # y = x reference and the "DR better" side.
    ax.fill_between([lo, hi_y], [lo, hi_y], [hi_y, hi_y], color=rs.GRID, alpha=0.35, linewidth=0,
                    zorder=0)
    ax.plot([lo, hi_x], [lo, hi_x], color=rs.MUTED, linewidth=1.0, zorder=1)
    ax.text(31.0, 4.0, "direct genes better\n(below y = x)", ha="left", va="center", fontsize=7.5,
            color=rs.INK_2)
    ax.text(-1.0, 21.0, "best DR better (above y = x)", ha="left", va="center", fontsize=7.5,
            color=rs.INK_2)

    # Biological panels: neutral dark; the labelled focus panels take their best-DR method colour.
    other = bio[~bio["stage0_panel_id"].isin(FOCUS_LABELLED)]
    ax.plot(other["direct_pp"], other["best_pp"], marker="o", markersize=rs.MARKER - 0.5,
            markerfacecolor=rs.INK_2, alpha=0.85, zorder=3, **_ring())
    ax.plot(ctrl["direct_pp"], ctrl["best_pp"], marker="s", markersize=rs.MARKER - 0.5,
            markerfacecolor=rs.SURFACE, markeredgecolor=rs.MUTED, markeredgewidth=1.3, linestyle="none",
            zorder=3)

    focus = bio[bio["stage0_panel_id"].isin(FOCUS_LABELLED)]
    label_xy = {  # absolute text anchor (x, y) in data units, left-aligned
        "single_geneset__hallmark_allograft_rejection": (1.6, 14.9),
        "single_geneset__hallmark_apoptosis": (-1.2, 8.9),
        "single_geneset__hallmark_interferon_alpha_response": (3.0, 12.5),
    }
    for _, r in focus.iterrows():
        method = METHOD_OF_CONFIG[r["best_dr_method"]]
        ax.plot(r["direct_pp"], r["best_pp"], marker="o", markersize=rs.MARKER + 1.5,
                markerfacecolor=rs.METHOD[method], zorder=5, **_ring())
        tx, ty = label_xy[r["stage0_panel_id"]]
        k = int(r["best_dr_k"])
        txt = f"{FOCUS[r['stage0_panel_id']]}\n{rs.METHOD_LABEL[method]} K{k}"
        ax.annotate(txt, xy=(r["direct_pp"], r["best_pp"]), xytext=(tx, ty), ha="left", va="center",
                    fontsize=7, color=rs.INK,
                    arrowprops=dict(arrowstyle="-", color=rs.MUTED, linewidth=0.7, shrinkA=1, shrinkB=4))

    ctrl_xy = {  # absolute text anchor (x, y, ha)
        "full_34": (27.1, 11.6, "center"),
        "core_only": (14.9, 9.3, "left"),
        "hvg_top_requested_500__available_500": (11.1, 18.3, "right"),
        "hvg_top_requested_1000__available_1000": (18.3, 21.1, "center"),
        "hvg_top_requested_3000__available_3000": (28.6, 17.8, "center"),
        "hvg_top_requested_10000__available_10000": (37.9, 10.6, "center"),
    }
    for _, r in ctrl.iterrows():
        tx, ty, ha = ctrl_xy[r["stage0_panel_id"]]
        ax.text(tx, ty, CONTROL_LABEL[r["stage0_panel_id"]], ha=ha, va="center", fontsize=6.8, color=rs.INK_2)

    ax.set_xlabel("Direct-gene probe on the same panel genes: median held-out lift (pp)")
    ax.set_ylabel("Best of up to 16 DR cells, chosen in-sample\nby median lift: median lift (pp)")
    ax.set_title(
        f"Best DR sits above y = x for {counts['d1e_bio_median']} of {counts['d1e_bio_n']} programs, "
        f"far above it for only a few;\ncontrol bags mostly favour direct genes "
        f"({counts['d1e_ctrl_median']} of {counts['d1e_ctrl_n']} above)",
        loc="left",
    )

    handles = [
        Line2D([], [], marker="o", markersize=rs.MARKER - 0.5, markerfacecolor=rs.INK_2,
               **_ring(color=rs.INK_2)),
        Line2D([], [], marker="s", markersize=rs.MARKER - 0.5, markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.MUTED, markeredgewidth=1.3, linestyle="none"),
        Line2D([], [], color=rs.MUTED, linewidth=1.0),
        Line2D([], [], marker="o", markersize=rs.MARKER + 1, markerfacecolor=rs.METHOD["factosig_promax"],
               **_ring()),
        Line2D([], [], marker="o", markersize=rs.MARKER + 1, markerfacecolor=rs.METHOD["pca"], **_ring()),
    ]
    labels = [
        "biological program (41)",
        "control bag, HVG or union (6)",
        "y = x (no DR gain)",
        "labelled: best DR is FactoSig-promax",
        "labelled: best DR is PCA",
    ]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=3, fontsize=6.8,
              handletextpad=0.4, columnspacing=1.2)
    return rs.save(fig, "fig08b_dr_vs_direct")


def report_numbers() -> None:
    g = patient_concentration()
    print("fig08a per patient (biological DR atlas cells):")
    print(g.to_string())
    c = atlas_cell_patient_counts()
    print(f"atlas cells: {len(c)}; >=5 patients with lift > 5 pp: {(c >= 5).sum()}; all 7: {(c == 7).sum()}")
    hp = g.loc[list(rs.HIGH_PREVALENCE)]
    print("high-prevalence malignant cells:", int(hp["n_malignant"].sum()), "of", int(g["n_malignant"].sum()),
          "supported malignant cells")
    print("estimand counts:", estimand_counts())
    d = dr_vs_direct()
    cols = ["stage0_panel_id", "best_dr_configuration", "best_pp", "direct_pp", "best_p20_pp", "direct_p20_pp"]
    print(d[d["stage0_panel_id"].isin(list(FOCUS) + list(CONTROL_LABEL))][cols].to_string())


def main() -> None:
    rs.apply()
    print(fig08a_patient_concentration())
    print(fig08b_dr_vs_direct())
    report_numbers()


if __name__ == "__main__":
    main()
