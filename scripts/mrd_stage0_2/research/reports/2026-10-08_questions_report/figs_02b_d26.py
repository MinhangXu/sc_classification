"""Figures for section 2b (what did we do with allograft rejection and apoptosis? the D26 leverage test).

D26 repeats the D17 leave-out design on the two panels where DR gains most over direct genes. The IFN-alpha
rows come from D17 so that the three panels tested sit side by side. Primary axis = top_gap (the factor
with the largest all-cell standardized malignant-normal gap): PC8/FA7 on IFN-alpha, PC8/FA8 on allograft
rejection, PC6/FA6 on apoptosis.

fig02b_d26a_ratio_by_set  (a-c) per leave-out set (P02, P03, P04, P02+P03+P04; contrasts P01, P09, P06,
                          P01+P09): axis loss divided by the mean of its 20 composition-matched random drops
                          (same number of cells and of malignant cells), one marker per method x K, filled
                          when above all 20 (rank p = 1/21); the P01 column is shaded; the count of cells
                          above all 20 is printed over each column.
                          (d-f) absolute size: matched |r| of the primary axis after leaving P02+P03+P04
                          out (method colour), the range over its 20 composition-matched drops (gray bar)
                          and after leaving P01 out (ring), on a 1 - |r| log scale labelled in |r|; the
                          dashed line is the decision rule's |r| = 0.98 ("beyond controls but small").
(Report v2's heatmap of every primary-axis cell, fig02d_leverage_panels, is dropped in version 3; the
per-cell ratios are printed by main() and tabulated in the run's findings file.)

main() also recomputes the decision-rule counts from d26_leverage_contrast.csv (asserted equal to
d26_decision_summary.csv) and prints every number section 2b quotes.

Data (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/leverage_allograft_apoptosis_v1/   (D26)
      d26_leverage_contrast.csv, d26_decision_summary.csv, d26_reference_axes.csv
  analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/       (D17, the IFN-alpha rows)
      d17_leverage_contrast.csv

Run from the repo root or the report dir:
  python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_02b_d26.py
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
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

import report_style as rs  # noqa: E402

D26_SLUG = "leverage_allograft_apoptosis_v1"
D17_SLUG = "basis_variant_pilot_ifna_v1"
IFNA = "single_geneset__hallmark_interferon_alpha_response"
ALLO = "single_geneset__hallmark_allograft_rejection"
APOP = "single_geneset__hallmark_apoptosis"
PANELS = ((IFNA, "IFN-$\\alpha$"), (ALLO, "allograft rejection"), (APOP, "apoptosis"))
SHORT = {IFNA: "IFN-$\\alpha$", ALLO: "allograft rej.", APOP: "apoptosis"}
GROUP = "P02-P03-P04"
# Tick labels drop the "P" prefix so the eight columns stay legible at 7 pt (the axis label says "patient").
SETS = (("P02", "02"), ("P03", "03"), ("P04", "04"), (GROUP, "02+\n03+\n04"),
        ("P01", "01"), ("P09", "09"), ("P06", "06"), ("P01-P09", "01+\n09"))
CONTRASTS = ("P01", "P06", "P09", "P01-P09")
HIGH = ("P02", "P03", "P04", GROUP)
CELLS = tuple((m, k) for m in ("pca", "fa") for k in (10, 20, 40))
K_MARKERS = {10: "^", 20: "s", 40: "o"}
RANK_P_FLOOR = 1 / 21  # one-sided rank p with 20 controls when the set is above all of them
RULE_R = 0.98  # decision rule (4): matched |r| >= 0.98 means "beyond controls but small"
COLS = ["panel_id", "stage1_method", "effective_k", "dropped", "n_dropped_cells", "n_dropped_malignant",
        "ref_factor_label", "axis_loss_fit", "axis_loss_fit_label_matched_mean", "axis_loss_fit_label_matched_min",
        "axis_loss_fit_label_matched_max", "axis_loss_fit_ratio_to_label_matched_mean",
        "axis_loss_fit_label_matched_rank_p", "matched_abs_r_fit", "matched_abs_r_fit_label_matched_min",
        "matched_abs_r_fit_label_matched_max", "n_label_matched_controls"]


# ----------------------------------------------------------------------------- data


def load_primary() -> pd.DataFrame:
    """Primary-axis (top_gap) leave-out rows: D26 (allograft, apoptosis) plus D17 (IFN-alpha)."""
    d26 = pd.read_csv(rs.diag_dir(D26_SLUG) / "d26_leverage_contrast.csv")
    d26 = d26[(d26["arm"] == "leave_out") & (d26["ref_axis"] == "top_gap")].copy()
    d17 = pd.read_csv(rs.diag_dir(D17_SLUG) / "d17_leverage_contrast.csv")
    d17 = d17[d17["arm"] == "leave_out"].copy()
    d17["panel_id"] = IFNA
    out = pd.concat([d17[COLS], d26[COLS]], ignore_index=True)
    assert (out["n_label_matched_controls"] == 20).all()
    assert len(out) == 3 * 6 * 8, len(out)
    out["above_all"] = out["axis_loss_fit_label_matched_rank_p"] <= RANK_P_FLOOR + 1e-9
    return out


def load_all_axes() -> pd.DataFrame:
    d26 = pd.read_csv(rs.diag_dir(D26_SLUG) / "d26_leverage_contrast.csv")
    lo = d26[d26["arm"] == "leave_out"].copy()
    assert (lo["n_label_matched_controls"] == 20).all() and len(lo) == 2 * 3 * 6 * 8
    lo["above_all"] = lo["axis_loss_fit_label_matched_rank_p"] <= RANK_P_FLOOR + 1e-9
    return d26.assign(above_all=d26["axis_loss_fit_label_matched_rank_p"] <= RANK_P_FLOOR + 1e-9)


def _ratios(sub: pd.DataFrame) -> pd.DataFrame:
    return sub.pivot_table(index="dropped", columns=["stage1_method", "effective_k"],
                           values="axis_loss_fit_ratio_to_label_matched_mean")


def n_above_every_contrast(sub: pd.DataFrame, contrasts=CONTRASTS) -> int:
    rat = _ratios(sub)
    return int((rat.loc[GROUP] > rat.loc[list(contrasts)].max()).sum())


def check_decision_summary(all_axes: pd.DataFrame) -> pd.DataFrame:
    """Recompute the stored decision summary (every panel x axis) and assert it matches."""
    summ = pd.read_csv(rs.diag_dir(D26_SLUG) / "d26_decision_summary.csv")
    lo = all_axes[all_axes["arm"] == "leave_out"]
    for _, r in summ.iterrows():
        sub = lo[(lo["panel_id"] == r["panel_id"]) & (lo["ref_axis"] == r["ref_axis"])]
        grp = sub[sub["dropped"] == GROUP]
        assert len(grp) == 6
        assert int(grp["above_all"].sum()) == int(r["n_cells_test_set_above_all_composition_controls"])
        assert n_above_every_contrast(sub) == int(r["n_cells_test_set_ratio_above_every_contrast"])
        assert np.isclose(grp["axis_loss_fit_ratio_to_label_matched_mean"].min(), r["test_set_ratio_min"])
        assert np.isclose(grp["axis_loss_fit_ratio_to_label_matched_mean"].max(), r["test_set_ratio_max"])
        p01 = sub[sub["dropped"] == "P01"]["axis_loss_fit_ratio_to_label_matched_mean"]
        assert np.isclose(p01.min(), r["p01_ratio_min"]) and np.isclose(p01.max(), r["p01_ratio_max"])
    return summ


# ----------------------------------------------------------------------------- figure helpers


def _title(ax, text, *, pad=4):
    ax.set_title(text, fontsize=8.3, pad=pad, loc="left")


SET_X = np.array([0, 1, 2, 3.15, 4.6, 5.6, 6.6, 7.6])  # a gap before the contrast block
OFFSET = {("pca", 10): -0.27, ("pca", 20): -0.17, ("pca", 40): -0.07,
          ("fa", 10): 0.07, ("fa", 20): 0.17, ("fa", 40): 0.27}


def fig02b_d26a_ratio_by_set(prim: pd.DataFrame):
    rs.apply()
    fig = plt.figure(figsize=(rs.FULL_W, 5.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0], hspace=0.72, wspace=0.1,
                          left=0.085, right=0.995, top=0.87, bottom=0.16)
    top = [fig.add_subplot(gs[0, 0])]
    top += [fig.add_subplot(gs[0, i], sharey=top[0]) for i in (1, 2)]
    bot = [fig.add_subplot(gs[1, 0])]
    bot += [fig.add_subplot(gs[1, i], sharey=bot[0]) for i in (1, 2)]
    sets = [s for s, _ in SETS]

    # (a-c) ratio to the composition-matched control mean, per set
    for i, ((pid, plab), ax) in enumerate(zip(PANELS, top)):
        sub = prim[prim["panel_id"] == pid]
        p01x = SET_X[sets.index("P01")]
        ax.axvspan(p01x - 0.45, p01x + 0.45, color=rs.DIV_POS, alpha=0.09, lw=0, zorder=0)
        ax.axhline(1.0, color=rs.INK_2, lw=0.9, zorder=1)
        for xx, st in zip(SET_X, sets):
            s = sub[sub["dropped"] == st]
            for (m, k) in CELLS:
                r = s[(s["stage1_method"] == m) & (s["effective_k"] == k)].iloc[0]
                above = bool(r["above_all"])
                ax.plot([xx + OFFSET[(m, k)]], [r["axis_loss_fit_ratio_to_label_matched_mean"]],
                        linestyle="none", marker=K_MARKERS[k], ms=3.6,
                        markerfacecolor=rs.METHOD[m] if above else rs.SURFACE,
                        markeredgecolor=rs.METHOD[m], markeredgewidth=0.9, zorder=3)
            n_above = int(s["above_all"].sum())
            ax.text(xx, 1.0, f"{n_above}/6", transform=blended_transform_factory(ax.transData, ax.transAxes),
                    ha="center", va="bottom", fontsize=6.8, color=rs.INK if st in (GROUP, "P01") else rs.INK_2,
                    fontweight="bold" if st in (GROUP, "P01") else "normal")
        grp = sub[sub["dropped"] == GROUP]["axis_loss_fit_ratio_to_label_matched_mean"]
        p01 = sub[sub["dropped"] == "P01"]["axis_loss_fit_ratio_to_label_matched_mean"]
        gx = SET_X[sets.index(GROUP)]
        fmt = lambda v: f"{v:.1f}" if v < 10 else f"{v:.0f}"  # noqa: E731
        ax.text(gx, 0.028, f"{fmt(grp.min())}–{fmt(grp.max())}×", ha="center", va="bottom", fontsize=6.8,
                color=rs.INK, fontweight="bold")
        ax.text(p01x, 330, f"{fmt(p01.min())}–{fmt(p01.max())}×", ha="center", va="top", fontsize=6.8,
                color=rs.INK, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(0.02, 420)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["0.1×", "1×", "10×", "100×"])
        ax.set_xlim(-0.55, SET_X[-1] + 0.55)
        ax.set_xticks(SET_X)
        ax.set_xticklabels([lab for _, lab in SETS], fontsize=7.0, linespacing=0.9)
        for lab, st in zip(ax.get_xticklabels(), sets):
            if st in HIGH:
                lab.set_fontweight("bold")
                lab.set_color(rs.INK)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlabel("patient(s) left out", fontsize=7.0, labelpad=2)
        ax.grid(axis="y", color=rs.GRID, lw=0.6)
        src = " (D17)" if pid == IFNA else ""
        _title(ax, f"({'abc'[i]}) {plab}{src}", pad=11)
        if i:
            plt.setp(ax.get_yticklabels(), visible=False)
    top[0].set_ylabel("axis loss ÷ mean of its\n20 matched drops (log)", fontsize=7.0)
    top[0].text(-0.55, 1.0, "above\nall 20:", transform=blended_transform_factory(top[0].transData, top[0].transAxes),
                ha="right", va="bottom", fontsize=6.8, color=rs.INK_2, linespacing=0.95)

    # (d-f) matched |r| of the primary axis: P02+P03+P04 vs its controls and P01
    xs = np.arange(6, dtype=float)
    for i, ((pid, plab), ax) in enumerate(zip(PANELS, bot)):
        sub = prim[prim["panel_id"] == pid]
        ax.axhline(1 - RULE_R, color=rs.INK_2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for xx, (m, k) in zip(xs, CELLS):
            c = sub[(sub["stage1_method"] == m) & (sub["effective_k"] == k)].set_index("dropped")
            g = c.loc[GROUP]
            lo_r, hi_r = g["matched_abs_r_fit_label_matched_min"], g["matched_abs_r_fit_label_matched_max"]
            ax.plot([xx, xx], [max(1 - hi_r, 2.5e-4), 1 - lo_r], color=rs.AXIS, lw=3.4, solid_capstyle="butt",
                    zorder=2)
            ax.plot([xx - 0.12], [max(1 - g["matched_abs_r_fit"], 2.5e-4)], "o", ms=4.4, color=rs.METHOD[m],
                    mec=rs.SURFACE, mew=0.6, zorder=4)
            ax.plot([xx + 0.12], [max(1 - c.loc["P01", "matched_abs_r_fit"], 2.5e-4)], "D", ms=3.6, mfc="none",
                    mec=rs.INK, mew=0.9, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1.3, 2e-4)  # inverted: |r| = 1 at the top
        ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1.0]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["0.999", "0.99", "0.9", "0"])
        ax.set_xlim(-0.6, 5.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{rs.METHOD_LABEL[m]}\n{k}" for m, k in CELLS], fontsize=7.0, linespacing=0.9)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.axvline(2.5, color=rs.GRID, lw=0.8, zorder=0)
        ax.grid(axis="y", color=rs.GRID, lw=0.6)
        _title(ax, f"({'def'[i]}) {SHORT[pid]}: matched |r|", pad=4)
        if i:
            plt.setp(ax.get_yticklabels(), visible=False)
    bot[0].set_ylabel("matched |r| of the\nprimary axis (log 1 − |r|)", fontsize=7.0)
    bot[2].text(5.55, 1 - RULE_R, "|r| = 0.98", ha="right", va="bottom", fontsize=6.8, color=rs.INK_2)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.2, color=rs.METHOD["pca"], mec=rs.METHOD["pca"], label="PCA"),
        Line2D([], [], marker="o", ls="none", ms=4.2, color=rs.METHOD["fa"], mec=rs.METHOD["fa"], label="FA"),
        Line2D([], [], marker="o", ls="none", ms=4.2, mfc=rs.SURFACE, mec=rs.INK_2,
               label="open: inside its 20 drops"),
        Line2D([], [], marker="^", ls="none", ms=4.2, mfc=rs.INK_2, mec=rs.INK_2, label="K 10"),
        Line2D([], [], marker="s", ls="none", ms=4.2, mfc=rs.INK_2, mec=rs.INK_2, label="K 20"),
        Line2D([], [], marker="o", ls="none", ms=4.2, mfc=rs.INK_2, mec=rs.INK_2, label="K 40"),
        Line2D([], [], color=rs.AXIS, lw=3.4, label="(d–f) P02+P03+P04's 20 drops"),
        Line2D([], [], marker="D", ls="none", ms=3.6, mfc="none", mec=rs.INK, label="(d–f) P01 left out"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.0, bbox_to_anchor=(0.53, 0.0),
               handletextpad=0.35, handlelength=1.3, columnspacing=0.9)
    fig.suptitle("Leaving patient sets out of Stage 1 (primary axis): P02+P03+P04 against the contrast sets",
                 fontsize=8.2, fontweight="bold", x=0.01, ha="left", y=0.995)
    return fig


# ----------------------------------------------------------------------------- text checks


def print_numbers(prim: pd.DataFrame, all_axes: pd.DataFrame, summ: pd.DataFrame) -> None:
    """Every number section 2b quotes, recomputed from the stored tables."""
    print("== decision summary (stored, checked against the recomputation)")
    print(summ[["panel_id", "ref_axis", "n_cells_test_set_above_all_composition_controls",
                "n_cells_test_set_ratio_above_every_contrast", "test_set_ratio_min", "test_set_ratio_max",
                "test_set_axis_loss_max", "test_set_matched_abs_r_min", "p01_ratio_min", "p01_ratio_max"]]
          .round(4).to_string(index=False))
    p01_max = 0
    for pid, plab in PANELS:
        sub = prim[prim["panel_id"] == pid]
        rat = _ratios(sub)
        print(f"-- {plab} primary axis ({sorted(sub['ref_factor_label'].unique())})")
        print("   ratio:\n" + rat.round(2).to_string())
        print("   cells above all 20 per set:", sub.groupby("dropped")["above_all"].sum().astype(int).to_dict())
        n_c = n_above_every_contrast(sub)
        n_np01 = n_above_every_contrast(sub, ("P06", "P09"))
        cells_np01 = [c for c in rat.columns if rat.loc[GROUP, c] > rat.loc[["P06", "P09"], c].max()]
        largest = rat.idxmax()
        p01_max += int((largest == "P01").sum())
        print(f"   group above every contrast {n_c}/6; above both P06 and P09 {n_np01}/6 {cells_np01}; "
              f"set with the largest ratio per cell {largest.to_dict()}")
        g = sub[sub["dropped"] == GROUP].set_index(["stage1_method", "effective_k"])
        p03 = sub[sub["dropped"] == "P03"].set_index(["stage1_method", "effective_k"])
        p01 = sub[sub["dropped"] == "P01"].set_index(["stage1_method", "effective_k"])
        for m in ("pca", "fa"):
            gm = g.loc[m]
            print(f"   {m}: group loss {gm['axis_loss_fit'].min():.5f}-{gm['axis_loss_fit'].max():.5f}, "
                  f"matched |r| {gm['matched_abs_r_fit'].min():.4f}-{gm['matched_abs_r_fit'].max():.4f}, "
                  f"controls' matched |r| {gm['matched_abs_r_fit_label_matched_min'].min():.4f}-"
                  f"{gm['matched_abs_r_fit_label_matched_max'].max():.4f}; "
                  f"group |r| inside controls' |r| range in "
                  f"{int((gm['matched_abs_r_fit'] >= gm['matched_abs_r_fit_label_matched_min']).sum())}/3; "
                  f"P01 matched |r| {p01.loc[m, 'matched_abs_r_fit'].min():.3f}-{p01.loc[m, 'matched_abs_r_fit'].max():.3f}")
        share = (p03["axis_loss_fit"] / g["axis_loss_fit"])
        print(f"   P03 loss / group loss {share.min():.2f}-{share.max():.2f}; "
              f"P01 ratio {p01['axis_loss_fit_ratio_to_label_matched_mean'].min():.1f}-"
              f"{p01['axis_loss_fit_ratio_to_label_matched_mean'].max():.1f}, "
              f"P01 loss {p01['axis_loss_fit'].min():.4f}-{p01['axis_loss_fit'].max():.4f}")
        if pid == ALLO:
            print(f"   allograft PCA40: P03 ratio {rat.loc['P03', ('pca', 40)]:.2f}, "
                  f"group {rat.loc[GROUP, ('pca', 40)]:.2f}; group malignant dropped "
                  f"{int(sub[sub['dropped'] == GROUP]['n_dropped_malignant'].iloc[0])}")
    print(f"P01 has the largest ratio of the 8 sets in {p01_max}/18 primary-axis cells")
    no = all_axes[(all_axes["arm"] == "normal_only_all") & (all_axes["ref_axis"] == "top_gap")]
    print("normal-only refits (all 1,569 malignant cells dropped), primary axis matched |r|:")
    print(no.groupby(["panel_id", "stage1_method"])["matched_abs_r_fit"].agg(["min", "max"]).round(4).to_string())
    ref = pd.read_csv(rs.diag_dir(D26_SLUG) / "d26_reference_axes.csv")
    th = ref[ref["ref_axis"] == "top_heldout"]
    print(f"top_heldout differs from top_gap in {int((~th['same_factor_as_top_gap'].astype(bool)).sum())}/{len(th)} "
          f"cells: {th[~th['same_factor_as_top_gap'].astype(bool)][['panel_label', 'stage1_method', 'effective_k', 'ref_factor_label']].values.tolist()}")
    pr = ref[ref["ref_axis"] == "probe"]
    print(f"probe nonzero factors {int(pr['probe_n_nonzero'].min())}-{int(pr['probe_n_nonzero'].max())}")
    lo = all_axes[all_axes["arm"] == "leave_out"]
    for pid in (ALLO, APOP):
        pk = lo[(lo["panel_id"] == pid) & (lo["ref_axis"] == "probe") & (lo["dropped"] == GROUP)
                & (lo["stage1_method"] == "fa") & (lo["effective_k"] == 40)].iloc[0]
        print(f"   probe FA40 {pid[-10:]}: group loss {pk['axis_loss_fit']:.4f}, controls mean "
              f"{pk['axis_loss_fit_label_matched_mean']:.4f}, max {pk['axis_loss_fit_label_matched_max']:.4f}")


# ----------------------------------------------------------------------------- main


def main() -> None:
    prim = load_primary()
    all_axes = load_all_axes()
    summ = check_decision_summary(all_axes)
    # IFN-alpha (D17) under the same rule: 3/6 above all 20 controls, 0/6 above every contrast
    ifna = prim[prim["panel_id"] == IFNA]
    assert int(ifna[ifna["dropped"] == GROUP]["above_all"].sum()) == 3
    assert n_above_every_contrast(ifna) == 0
    # the verdict does not hinge on P01: without the P01 sets, only the three allograft PCA cells clear P06 and P09
    assert [n_above_every_contrast(prim[prim["panel_id"] == p], ("P06", "P09")) for p, _ in PANELS] == [0, 3, 0]
    rs.save(fig02b_d26a_ratio_by_set(prim), "fig02b_d26a_ratio_by_set")
    print_numbers(prim, all_axes, summ)


if __name__ == "__main__":
    main()
