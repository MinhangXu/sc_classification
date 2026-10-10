"""Section 04 figures: the IFN-alpha PCA40-vs-FA40 coverage gap is a Stage 2 L1 entry threshold.

Reads only stored tables (no refits):
  regularization_contract_v1/d19_anchor_ifna_l1_entry_check.csv   (fig04a)
  regularization_contract_v1/d19_ifna_pca40_vs_fa40_by_patient.csv (fig04b)
  factor_patient_matrix_v1/factor_patient_matrix_long.csv.gz       (fig04c, D16 factor x patient matrix)

Run: python <report dir>/figs_04_threshold.py (any cwd; EXP resolves through report_style.exp_dir())
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import report_style as rs

D19 = "regularization_contract_v1"
PATIENTS = list(rs.SUPPORTED)
ZERO_TOL = 1e-9  # stored all-zero folds carry |lift| ~ 1e-17


def _anchor() -> pd.DataFrame:
    return pd.read_csv(rs.diag_dir(D19) / "d19_anchor_ifna_l1_entry_check.csv")


def _by_patient() -> pd.DataFrame:
    return pd.read_csv(rs.diag_dir(D19) / "d19_ifna_pca40_vs_fa40_by_patient.csv")


def _bracket(ax, x0, x1, y, text, *, transform=None, down=True, tick=0.04):
    """Thin bracket from x0 to x1 at height y (data x, given y transform)."""
    tr = transform if transform is not None else ax.transData
    d = -tick if down else tick
    ax.plot([x0, x0, x1, x1], [y + d, y, y, y + d], color=rs.INK_2, lw=0.8,
            transform=tr, clip_on=False, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - d * 0.6, text, ha="center", va="bottom" if down else "top",
            fontsize=7, color=rs.INK_2, transform=tr, clip_on=False)


# ---------------------------------------------------------------------------
# fig04a: entry margin per held-out patient, PC8 vs FA7, at the atlas C
# ---------------------------------------------------------------------------
def fig04a_entry_margin() -> None:
    a = _anchor()
    pca = a[a.stage1_method == "pca"].set_index("heldout_patient").loc[PATIENTS]
    fa = a[a.stage1_method == "fa"].set_index("heldout_patient").loc[PATIENTS]
    x = np.arange(len(PATIENTS))
    dx = 0.14

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(rs.FULL_W, 3.55), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    fig.subplots_adjust(left=0.13, right=0.885, top=0.815, bottom=0.14, hspace=0.52)

    # (a) margin = |gap| / threshold, line at 1.
    ax.axhspan(0.9, 1.1, color=rs.GRID, alpha=0.55, lw=0, zorder=0)
    ax.axhline(1.0, color=rs.INK_2, lw=1.0, zorder=1)
    ax.plot(x - dx, pca.margin_PC8.values, "o", color=rs.METHOD["pca"], ms=rs.MARKER,
            mec=rs.SURFACE, mew=rs.RING, zorder=3, linestyle="none")
    ax.plot(x + dx, fa.margin_FA7.values, "o", color=rs.METHOD["fa"], ms=rs.MARKER,
            mec=rs.SURFACE, mew=rs.RING, zorder=3, linestyle="none")
    # Selective value labels: the contested folds P02, P06, P07.
    for p in ("P02", "P06", "P07"):
        i = PATIENTS.index(p)
        ax.text(i - dx - 0.09, pca.margin_PC8.iloc[i], f"{pca.margin_PC8.iloc[i]:.3f}",
                ha="right", va="center", fontsize=7, color=rs.INK_2)
        ax.text(i + dx + 0.09, fa.margin_FA7.iloc[i], f"{fa.margin_FA7.iloc[i]:.3f}",
                ha="left", va="center", fontsize=7, color=rs.INK_2)
    ax.text(-0.42, 1.185, "above 1: the factor enters the probe", fontsize=7,
            color=rs.INK_2, va="top")
    ax.text(-0.42, 0.675, "below 1: all-zero probe, lift = 0 by construction", fontsize=7,
            color=rs.INK_2, va="bottom")
    ax.text(6.6, 1.0, "entry\nthreshold", fontsize=7, color=rs.INK_2, va="center", ha="left")
    ax.text(6.6, 1.1, "+10%", fontsize=7, color=rs.MUTED, va="center", ha="left")
    ax.text(6.6, 0.9, "\u221210%", fontsize=7, color=rs.MUTED, va="center", ha="left")
    ax.set_ylim(0.66, 1.2)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    ax.set_ylabel("entry margin\n(|gap| / threshold)")
    ax.set_title("(a) PC8 clears the L1 bar in 5 folds, FA7 in 2", loc="left")
    ax.set_xlim(-0.5, 6.5)

    # (b) gap vs threshold in SD units: the bar moves, the gaps barely do.
    thr = pca.entry_threshold_gap.values  # same per fold for both methods
    assert np.allclose(thr, fa.entry_threshold_gap.values)
    ntr = pca.n_train.values
    bx.bar(x, thr, width=0.56, color=rs.GRID, edgecolor="none", zorder=1)
    bx.plot(x - dx, np.abs(pca.gap_PC8.values), "o", color=rs.METHOD["pca"], ms=rs.MARKER - 1,
            mec=rs.SURFACE, mew=rs.RING, zorder=3)
    bx.plot(x + dx, np.abs(fa.gap_FA7.values), "o", color=rs.METHOD["fa"], ms=rs.MARKER - 1,
            mec=rs.SURFACE, mew=rs.RING, zorder=3)
    for i in range(len(PATIENTS)):
        bx.text(x[i], 0.05, f"{ntr[i]:,}", ha="center", va="bottom", fontsize=7,
                color=rs.INK_2, zorder=4)
    bx.set_ylim(0, 1.12)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_ylabel("class-mean gap\n(training SD)")
    bx.set_title("(b) The gaps barely move; the bar rises when a large patient is held out", loc="left")
    bx.set_xticks(x)
    bx.set_xticklabels(PATIENTS)
    bx.set_xlabel(r"held-out supported patient; number in bar = training cells $n_{\mathrm{train}}$", labelpad=4)
    bx.tick_params(axis="x", length=0)
    _bracket(bx, 0.72, 3.28, 0.84, "highest prevalence", down=True, tick=0.05)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=rs.METHOD["pca"], ms=rs.MARKER,
               mec=rs.SURFACE, mew=rs.RING, label="PC8, PCA40's best factor"),
        Line2D([], [], marker="o", linestyle="none", color=rs.METHOD["fa"], ms=rs.MARKER,
               mec=rs.SURFACE, mew=rs.RING, label="FA7, FA40's best factor"),
        Patch(facecolor=rs.GRID, edgecolor="none", label=r"(b) threshold $4/(C\,n_{\mathrm{train}})$"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.11, 0.87), ncol=3,
               frameon=False, handletextpad=0.3, columnspacing=1.5)
    fig.suptitle(r"IFN-$\alpha$ $K$ = 40, L1 at the atlas $C = 10^{-4}$: one fixed $C$ is a different bar per fold",
                 x=0.012, ha="left", fontsize=9.5, fontweight="bold", y=0.985)
    rs.save(fig, "fig04a_entry_margin")


# ---------------------------------------------------------------------------
# fig04b: per-patient lift under four C policies (small multiples, patients on y)
# ---------------------------------------------------------------------------
POLICIES = (
    ("stored_l1_fixed", "refit_l1_fixed", r"L1, atlas $C = 10^{-4}$"),
    ("sweep_l1_x1.05", "sweep_l1_x1.05", "L1, 1.05 \u00d7 atlas C"),
    ("tuned_l1", "tuned_l1", "L1, tuned C"),
    ("tuned_l2", "tuned_l2", "L2, tuned C"),
)


def fig04b_policy_lift() -> None:
    b = _by_patient().set_index("heldout_patient").loc[PATIENTS]
    y = np.arange(len(PATIENTS))
    dy = {"pca": -0.17, "fa": 0.17}

    fig, axes = plt.subplots(1, len(POLICIES), figsize=(rs.FULL_W, 2.85), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.68, bottom=0.16, wspace=0.12)
    for ax, (lift_key, nz_key, title) in zip(axes, POLICIES):
        stats = {}
        ax.grid(axis="x", color=rs.GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        for method in ("pca", "fa"):
            lift = 100 * b[f"{method}40_{lift_key}_lift"].values
            nz = b[f"{method}40_{nz_key}_n_nonzero"].values
            zero = nz == 0
            lift = np.where(zero, 0.0, lift)
            ys = y + dy[method]
            col = rs.METHOD[method]
            ax.hlines(ys[~zero], 0, lift[~zero], color=col, lw=1.0, zorder=2)
            ax.plot(lift[~zero], ys[~zero], "o", color=col, ms=rs.MARKER - 0.5, mec=rs.SURFACE,
                    mew=rs.RING, zorder=3, linestyle="none")
            ax.plot(lift[zero], ys[zero], "o", ms=rs.MARKER - 1.5, mfc=rs.SURFACE, mec=col,
                    mew=1.3, zorder=3, linestyle="none", clip_on=False)
            stats[method] = (int((lift > ZERO_TOL).sum()), float(np.median(lift)))
        ax.axvline(0, color=rs.AXIS, lw=0.8, zorder=1)
        ax.set_title(title, loc="left", fontsize=7.8, pad=25)
        ax.text(0.0, 1.015,
                f"lift > 0: {stats['pca'][0]} / {stats['fa'][0]} of 7\n"
                f"median: {stats['pca'][1]:.2f} / {stats['fa'][1]:.2f} pp",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=rs.INK_2)
        ax.set_xlim(-1.5, 23)
        ax.set_xticks([0, 10, 20])
        ax.set_xlabel("held-out lift (pp)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylim(len(PATIENTS) - 0.45, -0.55)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{p}  \u03c0 {100 * v:.1f}%" for p, v in zip(PATIENTS, b.prevalence.values)])
    # Vertical bracket for the highest-prevalence patients, left of the tick labels.
    tr = axes[0].get_yaxis_transform()
    xb = -0.77
    axes[0].plot([xb + 0.05, xb, xb, xb + 0.05], [0.75, 0.75, 3.25, 3.25], color=rs.INK_2, lw=0.8,
                 transform=tr, clip_on=False, solid_capstyle="butt")
    axes[0].text(xb - 0.04, 2.0, "highest\nprevalence", rotation=90, ha="right", va="center",
                 fontsize=7, color=rs.INK_2, transform=tr, clip_on=False)
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=rs.METHOD["pca"], ms=rs.MARKER - 0.5,
               mec=rs.SURFACE, mew=rs.RING, label="PCA40"),
        Line2D([], [], marker="o", linestyle="none", color=rs.METHOD["fa"], ms=rs.MARKER - 0.5,
               mec=rs.SURFACE, mew=rs.RING, label="FA40"),
        Line2D([], [], marker="o", linestyle="none", ms=rs.MARKER - 1.5, mfc=rs.SURFACE,
               mec=rs.MUTED, mew=1.3, label="hollow = all-zero probe (lift 0 by construction)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.855), ncol=3,
               frameon=False, handletextpad=0.3, columnspacing=1.4)
    fig.text(0.985, 0.875, "panel numbers: PCA40 / FA40", ha="right", va="bottom", fontsize=7,
             color=rs.INK_2)
    fig.suptitle("FA40 matches PCA40's patients at +5% C and reaches all 7 once C is tuned per fold",
                 x=0.005, ha="left", fontsize=9.5, fontweight="bold", y=0.995)
    rs.save(fig, "fig04b_policy_lift")


# ---------------------------------------------------------------------------
# fig04c: the factor x patient matrix the team asked for (D16), IFN-alpha PCA40 and FA40
# ---------------------------------------------------------------------------
D16 = "factor_patient_matrix_v1"
IFNA = "single_geneset__hallmark_interferon_alpha_response"
ALL_PATIENTS = [f"P{i:02d}" for i in range(1, 14)]
AXIS_FACTOR = {"pca": 8, "fa": 7}  # the reference malignant axis of each basis (PC8, FA7)


def fig04c_factor_patient_matrix() -> dict:
    from matplotlib.colors import LinearSegmentedColormap

    long = pd.read_csv(rs.diag_dir(D16) / "factor_patient_matrix_long.csv.gz")
    q = long[(long.stage0_panel_id == IFNA) & (long.k == 40) & (~long.degenerate.astype(bool))]
    cmap = LinearSegmentedColormap.from_list("div", [rs.DIV_NEG, rs.DIV_MID, rs.DIV_POS])
    lim = 1.5  # log2 ratio shown: 0.35x to 2.8x
    n_rows = 40
    fig, axes = plt.subplots(1, 2, figsize=(rs.FULL_W, 4.75), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.87, top=0.755, bottom=0.08, wspace=0.08)
    out = {}
    im = None
    for ax, method in zip(axes, ("pca", "fa")):
        m = q[q.stage1_method == method].pivot(index="factor_index", columns="patient", values="enrichment")
        m = m[ALL_PATIENTS].sort_index()
        n_f = len(m)
        z = np.log2(m.to_numpy())
        im = ax.imshow(np.clip(z, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                       extent=(-0.5, len(ALL_PATIENTS) - 0.5, n_f + 0.5, 0.5), interpolation="nearest")
        top = m.idxmax(axis=1)
        counts = top.value_counts()
        out[method] = (n_f, counts.to_dict())
        # Dot = the patient that tops this factor (largest ratio in the row).
        ax.plot([ALL_PATIENTS.index(p) for p in top.values], m.index.to_numpy(), ls="none", marker="o",
                ms=2.4, color=rs.INK, mec="none", zorder=3)
        # Factor-topping counts above the columns.
        for j, pid in enumerate(ALL_PATIENTS):
            c = int(counts.get(pid, 0))
            ax.text(j, -0.4, f"{c}" if c else "\u00b7", ha="center", va="bottom", fontsize=7.2,
                    color=rs.INK if c else rs.MUTED, fontweight="bold" if c >= 10 else "normal")
        if method == "pca":
            ax.text(-0.75, -0.4, "tops:", ha="right", va="bottom", fontsize=7.2, color=rs.INK_2)
        # Highest-prevalence bracket above the counts.
        i0, i1 = ALL_PATIENTS.index("P02"), ALL_PATIENTS.index("P04")
        yb = -2.6
        ax.plot([i0 - 0.42, i0 - 0.42, i1 + 0.42, i1 + 0.42], [yb + 0.6, yb, yb, yb + 0.6], color=rs.INK_2,
                lw=0.8, clip_on=False, solid_capstyle="butt")
        ax.text((i0 + i1) / 2, yb - 0.25, "highest prevalence", ha="center", va="bottom", fontsize=7,
                color=rs.INK_2, clip_on=False)
        ax.set_xticks(range(len(ALL_PATIENTS)))
        ax.set_xticklabels([p[1:] for p in ALL_PATIENTS], fontsize=7.2)
        for lab, pid in zip(ax.get_xticklabels(), ALL_PATIENTS):
            if pid not in rs.SUPPORTED:
                lab.set_color(rs.MUTED)
        ax.set_xlabel("patient (P01\u2013P13; gray = not supported)", fontsize=7.6)
        ax.set_ylim(n_rows + 0.5, 0.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        name, axis_name = ("PCA40", "PC8") if method == "pca" else ("FA40", "FA7")
        extra = "" if n_f == n_rows else f", {n_rows - n_f} degenerate omitted"
        ax.set_title(f"{name}: {n_f} factors{extra};\noutlined row = {axis_name}", loc="left", fontsize=8, pad=30,
                     linespacing=1.1)
        # Outline the reference malignant axis row (PC8 or FA7).
        fa = AXIS_FACTOR[method]
        ax.add_patch(plt.Rectangle((-0.5, fa - 0.5), len(ALL_PATIENTS), 1.0, fill=False, edgecolor=rs.INK,
                                   lw=1.1, zorder=4, clip_on=False))
    ticks = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([str(t) for t in ticks], fontsize=7.2)
    axes[0].set_ylabel("factor index (variance order within the fit)", fontsize=7.6)
    cax = fig.add_axes([0.905, 0.25, 0.018, 0.45])
    cb = fig.colorbar(im, cax=cax)
    rt = [0.5, 0.71, 1.0, 1.41, 2.0, 2.83]
    cb.set_ticks(np.log2(rt))
    cb.set_ticklabels([f"{r:g}\u00d7" for r in rt], fontsize=7.2)
    cb.outline.set_visible(False)
    cax.set_title("ratio", fontsize=7.2, color=rs.INK_2, pad=4)
    fig.suptitle("IFN-$\\alpha$, $K$ = 40: P02 tops most factors in both bases and P03 none,\n"
                 "so FA40 does not favour the highest-prevalence patients more than PCA40",
                 x=0.005, ha="left", fontsize=9.5, fontweight="bold", y=0.995)
    rs.save(fig, "fig04c_factor_patient_matrix")
    return out


def main() -> None:
    rs.apply()
    fig04a_entry_margin()
    fig04b_policy_lift()
    print("fig04c factor-topping counts:", fig04c_factor_patient_matrix())


if __name__ == "__main__":
    main()
