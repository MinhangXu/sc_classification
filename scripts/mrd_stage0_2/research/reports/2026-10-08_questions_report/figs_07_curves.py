"""Figures for section 07 (learning curves): D15 cell-depth curve and D15b patient-count curve.

Run from the repo root or this directory with the lab env python:
    python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_07_curves.py

Reads (EXP-relative, through report_style.diag_dir):
  analysis/scorecards/stage2_diagnostic/cell_count_learning_curve_v1/cell_count_learning_curve_summary.csv
  analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/d15b_full_curve_by_k.csv
  analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/d15b_full_by_patient_k.csv
  analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/d15b_full_gain_vs_k2.csv
Writes figures/fig07a_cell_depth.{pdf,png}, fig07b_patient_count.{pdf,png},
fig07c_patient_gain.{pdf,png}.
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
from matplotlib.patches import Patch  # noqa: E402

D15_SLUG = "cell_count_learning_curve_v1"
D15B_SLUG = "patient_count_learning_curve_v1"

# (D15b representation id, D15 curve label, display name, Stage 2 penalty at the atlas C, method role)
REPS = [
    ("ifna_pca40", "interferon alpha response | pca K=40", "IFN-\u03b1 PCA40", "L1", "pca"),
    ("ifna_fa40", "interferon alpha response | fa K=40", "IFN-\u03b1 FA40", "L1", "fa"),
    ("apoptosis_promax40", "apoptosis | factosig_promax K=40", "Apoptosis FactoSig-promax40", "L2", "factosig_promax"),
    ("allograft_promax40", "allograft rejection | factosig_promax K=40", "Allograft FactoSig-promax40", "EN", "factosig_promax"),
]
MARKERS = {"ifna_pca40": "o", "ifna_fa40": "s", "apoptosis_promax40": "o", "allograft_promax40": "D"}

C_ATLAS_L1 = 1e-4  # D1 atlas C of both IFN-alpha L1 probes
PC8_GAP_SD = (0.704, 0.770)  # |PC8| full-fold class-mean gap, devlog 2026-10-08_team-report-devlog-pivot


def _ring(**kw):
    """Marker kwargs with the surface ring."""
    base = dict(markersize=rs.MARKER, markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING)
    base.update(kw)
    return base


def fig07a_cell_depth() -> Path:
    """D15: median lift vs training-cell depth (fixed C) and the fixed-C L1 entry threshold."""
    summ = pd.read_csv(rs.diag_dir(D15_SLUG) / "cell_count_learning_curve_summary.csv")
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(rs.FULL_W, 2.65), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.5}
    )

    ref = summ[summ.curve_label == REPS[0][1]].sort_values("n_train_cells")
    xs = ref.n_train_cells.to_numpy()
    mal = ref.n_train_malignant.to_numpy()
    for rid, label, name, pen, role in REPS:
        d = summ[summ.curve_label == label].sort_values("n_train_cells")
        assert np.allclose(d.n_train_cells.to_numpy(), xs)
        ax.plot(
            xs, d.median_lift * 100, color=rs.METHOD[role], lw=rs.LINE_W, marker=MARKERS[rid],
            label=f"{name} ({pen})", zorder=3 if role != "pca" else 2.5, **_ring()
        )
    # Direct labels at the right end (full training set), with leaders where the ends crowd.
    end = {r[0]: summ[(summ.curve_label == r[1]) & summ.is_full_training].median_lift.iloc[0] * 100 for r in REPS}
    xe, xr = xs[-1], xs[-1] * 1.55
    for rid, text, ylab in (("allograft_promax40", "Allograft", 14.6), ("apoptosis_promax40", "Apoptosis", 12.1),
                            ("ifna_pca40", "IFN PCA", 9.9), ("ifna_fa40", "IFN FA", 1.3)):
        ax.annotate(text, xy=(xe * 1.06, end[rid]), xytext=(xr, ylab), fontsize=7.2, color=rs.INK_2, va="center",
                    arrowprops=dict(arrowstyle="-", color=rs.MUTED, lw=0.6, shrinkA=0, shrinkB=2))
    ax.text(
        420, -1.0,
        "All-zero probe on the 0 line (lift 0 by\n"
        "construction): IFN-\u03b1 PCA40 and FA40\n"
        "through 25k cells, allograft through 2k",
        fontsize=7, color=rs.INK_2, va="top", ha="left", linespacing=1.15,
    )
    ax.set_xscale("log")
    ax.set_xlim(380, xs[-1] * 3.6)
    ax.set_ylim(-6.2, 22.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x) // 1000}k\n({int(m):,})" if x >= 1000 else f"{int(x)}\n({int(m):,})"
                        for x, m in zip(xs[:-1], mal[:-1])] + [f"full\n({int(mal[-1]):,})"])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_xlabel("Training cells, log scale (malignant training cells)")
    ax.set_ylabel("Median held-out lift (pp)")
    ax.set_title("(a) Median held-out lift vs training cells", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6, borderaxespad=0.1, ncol=1)

    # Right panel: L1 entry threshold 4/(C*n) at the atlas C, in SD of a standardized factor.
    n = np.geomspace(380, xs[-1] * 2.2, 200)
    thr = 4.0 / (C_ATLAS_L1 * n)
    ax2.axhspan(*PC8_GAP_SD, color=rs.BLUE, alpha=0.18, lw=0, zorder=1)
    ax2.plot(n, thr, color=rs.INK_2, lw=rs.LINE_W, zorder=3)
    ax2.plot(xs, 4.0 / (C_ATLAS_L1 * xs), ls="none", marker="o", color=rs.INK_2, zorder=4, **_ring(markersize=5))
    ax2.text(420, 0.80, "PC8 gap 0.70\u20130.77 SD", fontsize=7, color=rs.INK_2, va="bottom")
    ax2.annotate("8 SD at 5,000 cells", xy=(5000, 8.0), xytext=(6500, 22), fontsize=7, color=rs.INK_2,
                 arrowprops=dict(arrowstyle="-", color=rs.MUTED, lw=0.7))
    ax2.annotate("1.6 SD at\n25,000 cells", xy=(25000, 1.6), xytext=(480, 2.6), fontsize=7, color=rs.INK_2,
                 va="center",
                 arrowprops=dict(arrowstyle="-", color=rs.MUTED, lw=0.7))
    x_lo, x_hi = 4.0 / (C_ATLAS_L1 * PC8_GAP_SD[1]), 4.0 / (C_ATLAS_L1 * PC8_GAP_SD[0])
    ax2.axvspan(x_lo, x_hi, color=rs.MUTED, alpha=0.25, lw=0, zorder=1)
    ax2.annotate("crosses at about\n52,000\u201357,000 cells", xy=(x_lo, 0.74), xytext=(1100, 0.27), fontsize=7,
                 color=rs.INK_2, arrowprops=dict(arrowstyle="-", color=rs.MUTED, lw=0.7))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(380, xs[-1] * 2.2)
    ax2.set_ylim(0.2, 150)
    ax2.set_xticks([1e3, 1e4, 5e4])
    ax2.set_xticklabels(["1,000", "10,000", "50,000"])
    ax2.set_yticks([0.3, 1, 3, 10, 30, 100])
    ax2.set_yticklabels(["0.3", "1", "3", "10", "30", "100"])
    ax2.minorticks_off()
    ax2.grid(True, axis="both")
    ax2.set_xlabel("Training cells (log scale)")
    ax2.set_ylabel("L1 entry threshold $4/(C\\,n)$,\nSD units (log scale)")
    ax2.set_title("(b) L1 entry threshold at $C = 10^{-4}$", fontsize=8.5)
    fig.suptitle("Fixed C: apoptosis (L2) gains with depth; allograft (EN) and both IFN-\u03b1 L1 probes\n"
                 "are all-zero at small n, because the entry bar falls only as n grows",
                 x=0.005, y=1.0, ha="left", va="bottom", fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig07a_cell_depth")


def fig07b_patient_count() -> Path:
    """D15b full design: median tuned lift vs number of training patients, 4 small multiples."""
    curve = pd.read_csv(rs.diag_dir(D15B_SLUG) / "d15b_full_curve_by_k.csv")
    fig, axes = plt.subplots(2, 2, figsize=(rs.FULL_W, 3.55), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.16, "wspace": 0.08})
    ks = np.arange(1, 7)
    mal_ref = None
    for ax, (rid, _label, name, pen, role) in zip(axes.flat, REPS):
        col = rs.METHOD[role]
        t = curve[(curve.representation == rid) & (curve.arm == "tuned")].sort_values("k")
        f = curve[(curve.representation == rid) & (curve.arm == "fixed_atlas")].sort_values("k")
        assert list(t.k) == list(ks)
        mal = t.median_train_malignant.to_numpy()
        if mal_ref is None:
            mal_ref = mal
        ax.fill_between(ks, t.median_lift_lo * 100, t.median_lift_hi * 100, color=col, alpha=0.13, lw=0, zorder=1)
        ax.plot(ks, f.median_lift * 100, color=rs.MUTED, lw=1.1, marker="s", zorder=2, **_ring(markersize=4.2))
        ax.plot(ks[1:], t.median_lift.iloc[1:] * 100, color=col, lw=rs.LINE_W, zorder=3)
        ax.plot(ks[:2], t.median_lift.iloc[:2] * 100, color=col, lw=rs.LINE_W, ls=(0, (2, 1.5)), zorder=3)
        ax.plot(ks[1:], t.median_lift.iloc[1:] * 100, ls="none", marker="o", color=col, zorder=4, **_ring())
        ax.plot([1], [t.median_lift.iloc[0] * 100], ls="none", marker="o", color=rs.SURFACE, zorder=4,
                markersize=rs.MARKER - 0.6, markeredgecolor=col, markeredgewidth=1.4)
        y6 = t.median_lift.iloc[-1] * 100
        ax.text(6.2, y6, f"{y6:.2f}", fontsize=7, color=rs.INK, va="center")
        ax.text(0.03, 0.96, name, transform=ax.transAxes, fontsize=7.8, color=rs.INK, va="top", fontweight="bold")
        ax.text(0.03, 0.85, f"atlas penalty {pen}; tuned arm picks L1 or L2", transform=ax.transAxes, fontsize=6.8,
                color=rs.INK_2, va="top")
        ax.set_xlim(0.6, 6.8)
        ax.set_ylim(0, 44)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_xticks(ks)
    for ax in axes[1]:
        ax.set_xticklabels([str(k) for k in ks])
    fig.supxlabel("Supported training patients k (the 6 unsupported patients are always in training)",
                  fontsize=8, color=rs.INK_2, y=0.0)
    fig.supylabel("Median held-out lift over 7 held-out patients (pp)", fontsize=8, color=rs.INK_2, x=0.055)
    handles = [  # neutral ink: colour marks the representation, not a method of the legend
        Line2D([], [], color=rs.INK_2, lw=rs.LINE_W, marker="o", **_ring()),
        Patch(color=rs.INK_2, alpha=0.13, lw=0),
        Line2D([], [], color=rs.INK_2, lw=rs.LINE_W, ls=(0, (2, 1.5)), marker="o", markerfacecolor=rs.SURFACE,
               markeredgecolor=rs.INK_2, markersize=rs.MARKER - 0.6, markeredgewidth=1.4),
        Line2D([], [], color=rs.MUTED, lw=1.1, marker="s", **_ring(markersize=4.2)),
    ]
    labels = [
        "C and penalty retuned per training set (median)",
        "95% patient-bootstrap CI of that median",
        "k = 1: C tuned by cell-level CV (protocol switch)",
        "fixed atlas C, as in the cell-depth curve (not nested)",
    ]
    fig.subplots_adjust(top=0.81)
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.085, 0.825), ncol=2, fontsize=7,
               handlelength=2.2, columnspacing=1.4, frameon=False)
    fig.suptitle("With C retuned, lift is positive with one supported training patient;\n"
                 "from 2 to 6 patients the median rises by a few pp, and the 95% CI is far wider",
                 x=0.085, y=0.935, va="bottom", ha="left", fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig07b_patient_count")


def _signed(v: float, fmt: str = "+.1f") -> str:
    """Signed number with a true minus sign (U+2212)."""
    return format(v, fmt).replace("-", "\u2212")


def fig07c_patient_gain() -> Path:
    """D15b full design: per-patient paired gain from k = 2 to k = 6, tuned arm."""
    byp = pd.read_csv(rs.diag_dir(D15B_SLUG) / "d15b_full_by_patient_k.csv")
    gain = pd.read_csv(rs.diag_dir(D15B_SLUG) / "d15b_full_gain_vs_k2.csv")
    pats = list(rs.SUPPORTED)
    fig, axes = plt.subplots(1, 4, figsize=(rs.FULL_W, 2.55), sharey=True, gridspec_kw={"wspace": 0.10})
    ypos = np.arange(len(pats))[::-1]
    gmin, gmax = np.inf, -np.inf
    for ax, (rid, _label, name, _pen, role) in zip(axes, REPS):
        col = rs.METHOD[role]
        t = byp[(byp.representation == rid) & (byp.arm == "tuned")].set_index(["heldout_patient", "k"]).lift_mean
        g = np.array([(t[(p, 6)] - t[(p, 2)]) * 100 for p in pats])
        gmin, gmax = min(gmin, g.min()), max(gmax, g.max())
        gr = gain[(gain.representation == rid) & (gain.arm == "tuned") & (gain.k_to == 6)].iloc[0]
        ax.axvspan(gr.median_gain_lo * 100, gr.median_gain_hi * 100, color=col, alpha=0.13, lw=0, zorder=0)
        ax.axvline(0, color=rs.AXIS, lw=0.8, zorder=1)
        ax.axvline(gr.median_gain * 100, color=col, lw=1.0, zorder=2)
        ax.hlines(ypos, 0, g, color=col, lw=rs.LINE_W, zorder=3)
        ax.plot(g, ypos, ls="none", marker="o", color=col, zorder=4, **_ring(markersize=5.2))
        # Value label just above each dot, on a surface patch so lines and bands never run through it.
        for y, v in zip(ypos, g):
            ax.text(v, y + 0.24, _signed(v), fontsize=7.2, color=rs.INK_2, ha="center", va="bottom", zorder=5,
                    bbox=dict(facecolor=rs.SURFACE, edgecolor="none", pad=0.4))
        ax.set_xlim(-4.5, 15.5)
        ax.set_xticks([0, 5, 10, 15])
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        program, rep = name.split(" ", 1)
        ax.set_title(f"{program}\n{rep}", fontsize=7.6, color=rs.INK, pad=13, loc="center")
        ax.text(0.5, 1.02, f"{_signed(gr.median_gain * 100, '+.2f')} [{_signed(gr.median_gain_lo * 100, '.2f')}, "
                f"{_signed(gr.median_gain_hi * 100, '.2f')}]", transform=ax.transAxes, fontsize=7.2,
                color=rs.INK_2, ha="center", va="bottom")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(pats)
    axes[0].set_ylim(-0.6, len(pats) - 0.05)
    fig.supxlabel("Gain in held-out lift from k = 2 to k = 6 training patients (pp); "
                  "above each panel: median gain [95% CI]", fontsize=8, color=rs.INK_2, y=-0.06)
    # Bracket the highest-prevalence patients (P02, P03, P04) left of the tick labels.
    hp = [ypos[pats.index(p)] for p in rs.HIGH_PREVALENCE]
    trans = axes[0].get_yaxis_transform()
    xb = -0.30
    lo, hi = min(hp) - 0.3, max(hp) + 0.3
    axes[0].plot([xb + 0.05, xb, xb, xb + 0.05], [lo, lo, hi, hi], transform=trans, color=rs.INK_2, lw=0.9,
                 clip_on=False)
    axes[0].text(xb - 0.05, np.mean(hp), "highest\nprevalence", transform=trans, rotation=90, fontsize=7.2,
                 color=rs.INK_2, ha="right", va="center")
    fig.subplots_adjust(top=0.72)
    fig.suptitle(f"Per-patient gains run from {_signed(gmin)} to {_signed(gmax)} pp;\n"
                 "both extremes are highest-prevalence patients",
                 x=0.0, y=0.91, va="bottom", ha="left", fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig07c_patient_gain")


def main() -> None:
    rs.apply()
    for fn in (fig07a_cell_depth, fig07b_patient_count, fig07c_patient_gain):
        print(fn())


if __name__ == "__main__":
    main()
