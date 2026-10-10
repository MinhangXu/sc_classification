"""Figures for section 01 (the cohort and the yardstick).

fig01a_cohort    13 patients on the x-axis: total cells, malignant cells (log scale),
                 malignant prevalence; supported patients in dark ink (blue is reserved for PCA),
                 P02/P03/P04 bracketed.
fig01b_yardstick IFN-alpha PCA40 (L1, C = 1e-4) held-out AP versus the constant-score
                 no-skill AP = prevalence for the 7 supported patients; lift is the gap.

Data (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/meeting_followup_v1/mrd_patient_cell_composition.csv
  analysis/scorecards/stage2_diagnostic/meeting_followup_v1/interferon_alpha_pca40_fa40_patient_comparison.csv

Run from the repo root or the report dir:
  python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_01_cohort.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import report_style as rs  # noqa: E402

UNSUPPORTED_FILL = rs.AXIS  # recessive gray for patients that stay in training only
SUPPORTED_FILL = rs.INK_2  # dark neutral: blue is the PCA colour everywhere else in the report


def load_composition() -> pd.DataFrame:
    path = rs.diag_dir("meeting_followup_v1") / "mrd_patient_cell_composition.csv"
    df = pd.read_csv(path)
    df = df.sort_values("patient_id").reset_index(drop=True)
    df["supported"] = df["patient_id"].isin(rs.SUPPORTED)
    # Cross-check the support flag stored with the table against the study rule.
    assert (df["supported"] == df["supported_for_lift_summary"].astype(bool)).all()
    assert int(df["n_total_binary_cells"].sum()) == 60118
    assert int(df["n_malignant_recipient"].sum()) == 1569
    return df


def load_ifna_pca40() -> pd.DataFrame:
    path = rs.diag_dir("meeting_followup_v1") / "interferon_alpha_pca40_fa40_patient_comparison.csv"
    df = pd.read_csv(path)
    df = df[(df["stage1_method"] == "pca") & (df["stage1_method_k"] == 40)]
    df = df[df["supported_patient"].astype(bool)].sort_values("heldout_patient").reset_index(drop=True)
    assert list(df["heldout_patient"]) == list(rs.SUPPORTED)
    return df


def _bracket(ax, x0, x1, y, text, *, h, text_dy=0.0, fontsize=7.5, transform=None, text_ha="center"):
    """Horizontal bracket from x0 to x1 at height y (data coords unless transform given)."""
    tr = transform if transform is not None else ax.transData
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=rs.INK_2, lw=0.9,
            transform=tr, clip_on=False, solid_capstyle="butt")
    tx = (x0 + x1) / 2 if text_ha == "center" else x0
    ax.text(tx, y + text_dy, text, ha=text_ha, va="bottom", fontsize=fontsize,
            color=rs.INK_2, transform=tr, clip_on=False)


def _fmt_prev(p: float) -> str:
    pct = 100 * p
    if pct == 0:
        return "0"
    return f"{pct:.1f}%" if pct >= 1 else f"{pct:.2f}%"


def fig01a_cohort() -> Path:
    rs.apply()
    df = load_composition()
    n = len(df)
    x = np.arange(n)
    colors = [SUPPORTED_FILL if s else UNSUPPORTED_FILL for s in df["supported"]]
    total = df["n_total_binary_cells"].to_numpy()
    mal = df["n_malignant_recipient"].to_numpy()
    prev = df["malignant_prevalence"].to_numpy()
    n_all = total.sum()
    idx = {p: i for i, p in enumerate(df["patient_id"])}
    hp = [idx[p] for p in rs.HIGH_PREVALENCE]
    width = 0.56

    fig, axes = plt.subplots(3, 1, figsize=(rs.FULL_W, 3.85), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 0.9, 1.0], "hspace": 0.68})
    ax_t, ax_m, ax_p = axes

    # (a) total cells, linear, thousands.
    ax_t.bar(x, total / 1000, width=width, color=colors, linewidth=0)
    ax_t.set_ylabel("cells per patient\n(thousands)")
    ax_t.set_ylim(0, 21.5)
    ax_t.set_yticks([0, 5, 10, 15, 20])
    for p in ("P01", "P09"):
        i = idx[p]
        ax_t.text(i - 0.28, total[i] / 1000 + 0.5, f"{total[i]:,} cells ({100 * total[i] / n_all:.1f}%)",
                  ha="left", va="bottom", fontsize=7, color=rs.INK_2)
    hp_cells = total[hp].sum()
    _bracket(ax_t, min(hp) - 0.3, max(hp) + 0.3, 5.2,
             f"P02\u2013P04: {hp_cells:,} cells ({100 * hp_cells / n_all:.1f}%)", h=0.8, text_dy=0.4,
             fontsize=7, text_ha="left")
    ax_t.set_title("P09 + P01 are 48% of all cells; P02 + P03 + P04 are 8.2%", pad=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SUPPORTED_FILL, lw=0),
               plt.Rectangle((0, 0), 1, 1, color=UNSUPPORTED_FILL, lw=0)]
    ax_p.legend(handles, ["supported (lift reported)", "not supported (training only)"],
                loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.0, borderaxespad=0.1,
                fontsize=7)

    # (b) malignant cells, log scale, dot plot (bars on a log axis have no true baseline).
    floor = 1.0
    ax_m.set_yscale("log")
    ax_m.set_ylim(floor, 3000)
    ax_m.axhline(10, color=rs.INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax_m.text(n - 0.45, 10 * 1.12, "support rule: 10 malignant cells", ha="right", va="bottom",
              fontsize=7, color=rs.INK_2)
    for i in range(n):
        if mal[i] == 0:
            ax_m.text(i, floor * 1.25, "0", ha="center", va="bottom", fontsize=7.5, color=rs.MUTED)
            continue
        ax_m.vlines(i, floor, mal[i], color=colors[i], lw=rs.LINE_W, zorder=2)
        ax_m.plot(i, mal[i], "o", ms=rs.MARKER, color=colors[i], mec=rs.SURFACE, mew=rs.RING, zorder=3)
        if mal[i] < 10:  # below the support line: label under the dot so the dashed line stays clear
            ax_m.text(i + 0.2, mal[i] * 0.8, f"{mal[i]:,}", ha="left", va="top", fontsize=7, color=rs.INK_2)
        else:
            ax_m.text(i + 0.2, mal[i], f"{mal[i]:,}", ha="left", va="center", fontsize=7, color=rs.INK_2)
    ax_m.set_ylabel("malignant cells\n(log scale)")
    ax_m.set_yticks([1, 10, 100, 1000])
    ax_m.set_yticklabels(["1", "10", "100", "1,000"])
    ax_m.minorticks_off()
    ax_m.set_title("Seven patients pass the 10-malignant-cell support rule", pad=6)

    # (c) prevalence, linear percent.
    ax_p.bar(x, 100 * prev, width=width, color=colors, linewidth=0)
    for i in range(n):
        if prev[i] > 0:
            ax_p.text(i, 100 * prev[i] + 0.7, _fmt_prev(prev[i]), ha="center", va="bottom",
                      fontsize=7, color=rs.INK_2)
    ax_p.set_ylabel("malignant prevalence\n(% of patient's cells)")
    ax_p.set_ylim(0, 49)
    ax_p.set_yticks([0, 10, 20, 30, 40])
    _bracket(ax_p, min(hp) - 0.3, max(hp) + 0.3, 41.5, "highest prevalence", h=2.0, text_dy=0.8,
             fontsize=7)
    ax_p.set_title("P03, P02 and P04 have by far the highest malignant prevalence", pad=6)

    ax_p.set_xticks(x)
    ax_p.set_xticklabels(df["patient_id"])
    for lab, s in zip(ax_p.get_xticklabels(), df["supported"]):
        lab.set_color(rs.INK if s else rs.MUTED)
        lab.set_fontweight("bold" if s else "normal")
    ax_p.set_xlabel("patient (bold = supported)")
    ax_p.set_xlim(-0.6, n - 0.4)
    return rs.save(fig, "fig01a_cohort")


def fig01b_yardstick() -> Path:
    rs.apply()
    df = load_ifna_pca40()
    x = np.arange(len(df))
    ap = 100 * df["heldout_auprc"].to_numpy()
    pi = 100 * df["patient_prevalence"].to_numpy()
    lift = df["heldout_lift_percentage_points"].to_numpy()
    idx = {p: i for i, p in enumerate(df["heldout_patient"])}
    hp = [idx[p] for p in rs.HIGH_PREVALENCE]

    fig, ax = plt.subplots(figsize=(rs.FULL_W, 2.25))
    ax.vlines(x, pi, ap, color=rs.BLUE, lw=rs.LINE_W, zorder=2)
    ax.plot(x, pi, "o", ms=rs.MARKER, color=rs.MUTED, mec=rs.SURFACE, mew=rs.RING, zorder=3,
            label="no-skill AP = prevalence $\\pi$ (constant score)")
    ax.plot(x, ap, "o", ms=rs.MARKER, color=rs.BLUE, mec=rs.SURFACE, mew=rs.RING, zorder=4,
            label="IFN-$\\alpha$ PCA40 probe, held-out AP")
    for i in range(len(df)):
        if abs(lift[i]) < 1e-9:
            # The first column's label is nudged right so it clears the y-axis spine.
            ax.annotate("all-zero probe:\nAP = $\\pi$, lift 0", xy=(i, ap[i] + 0.9), xytext=(i + (0.18 if i == 0 else 0), 10),
                        ha="center", va="bottom", fontsize=6.8, color=rs.MUTED, linespacing=1.05,
                        arrowprops=dict(arrowstyle="-", color=rs.MUTED, lw=0.7, shrinkA=0, shrinkB=0))
            continue
        if lift[i] < 1.0:  # dots nearly coincide: one label above them (clear of the next column's bar)
            ax.text(i, ap[i] + 1.6, f"AP {ap[i]:.1f}%\nlift +{lift[i]:.2f} pp", ha="center",
                    va="bottom", fontsize=6.8, color=rs.INK_2, linespacing=1.05)
            continue
        ax.text(i, ap[i] + 1.6, f"AP {ap[i]:.1f}%", ha="center", va="bottom", fontsize=6.8,
                color=rs.INK_2)
        ax.text(i + 0.1, (ap[i] + pi[i]) / 2, f"lift\n+{lift[i]:.2f} pp", ha="left", va="center",
                fontsize=7, color=rs.INK, linespacing=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["heldout_patient"])
    ax.set_xlim(-0.5, len(df) - 0.1)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("held-out AUPRC (%)")
    ax.set_xlabel("held-out supported patient")
    _bracket(ax, min(hp) - 0.25, max(hp) + 0.25, 55, "highest prevalence", h=2.0, text_dy=0.6,
             fontsize=7)
    ax.set_title("P03 has the highest AP, but P04 and P02 have more lift above $\\pi$", pad=6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.0, borderaxespad=0.1,
              fontsize=7)
    return rs.save(fig, "fig01b_yardstick")


def main() -> None:
    for fn in (fig01a_cohort, fig01b_yardstick):
        print(fn())


if __name__ == "__main__":
    main()
