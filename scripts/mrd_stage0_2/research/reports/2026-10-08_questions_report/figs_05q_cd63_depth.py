"""Section 5 follow-up figures for the user's 2026-10-10 late-evening questions (asks A-20261010-26..32): CD63
in detail, the total-ADT sensitivity of the two checks and a direct malignant-vs-healthy contrast (D29).

  fig05n_cd63_ranks       CD63 against PC8 as rank plots, like Figure 20 panel D: all 1,569 malignant cells, P09
                          (375), P09's predicted HSC/MPP2 (129) and P03 (758, where the link is absent).
  fig05o_cd63_anchors     CD63 (PC8, FA7) and CD82 (PC8) in malignant cells taken apart, like Figure 21: each
                          supported patient, the median of the seven, the pooled value and the median inside patient
                          x cell type (with the 11 groups as ticks), then the same two medians with each cell's total
                          ADT count held fixed. Each row's gray band is the largest |rho| any isotype reaches there.
  fig05q_mal_vs_healthy   No factor: inside one sample and predicted type (21 groups with >= 10 cells of each label),
                          the AUC of CD63, CD99 and CD82 (malignant vs healthy), raw and with total ADT held fixed, the
                          9 isotypes' range as a gray band, and the AUC of total ADT counts itself.
  fig05r_two_checks_depth Figure 22 redrawn with each cell's total ADT count held fixed (partial Spearman), malignant
                          and healthy cells; named proteins carry an arrow from their Figure 22 position.

Reads D27 and D29 (EXP-relative, read only: analysis/scorecards/stage2_diagnostic/factor_modality_assoc_v1/ and
.../protein_depth_contrast_v1/), the stored IFN-alpha PCA/FA K=40 scores and the cohort cache (obs and obsm['ADT'] only,
never the raw AnnData), through the loaders of the Pavan update's make_figures_v3.py. Every per-cell rho drawn is
asserted against D27 (raw) or D29 (partial). About 20 s.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report_style as rs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

D29_SLUG = "protein_depth_contrast_v1"
SOURCE = HERE.parents[1] / "team_updates" / "2026-10-09_protein_association_for_pavan" / "make_figures_v3.py"
TOL = 1e-5


def _mx():
    spec = importlib.util.spec_from_file_location("make_figures_v3_d29", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mx = _mx()
FS, FS_SMALL, FS_NOTE = mx.FS, mx.FS_SMALL, mx.FS_NOTE
signed = mx.signed


def load() -> dict:
    d27 = mx.load_d27()
    cohort = mx.load_cells(d27["orient"])
    d = rs.diag_dir(D29_SLUG)
    rho = pd.read_csv(d / "d29_rho_depth.csv")
    strata = pd.read_csv(d / "d29_rho_depth_strata.csv")
    iso = pd.read_csv(d / "d29_isotype_reference.csv")
    mvh = pd.read_csv(d / "d29_mal_vs_healthy.csv")
    checks = pd.read_csv(d / "d29_regression_checks.csv")
    assert bool(checks["passed"].all()), checks
    return {"d27": d27, "cohort": cohort, "rho": rho, "strata": strata, "iso": iso, "mvh": mvh}


def d29_row(rho: pd.DataFrame, factor: str, label: str, protein: str) -> pd.Series:
    r = rho[(rho["factor"] == factor) & (rho["label"] == label) & (rho["feature_short"] == protein)]
    assert len(r) == 1, (factor, label, protein)
    return r.iloc[0]


def iso_ref(iso: pd.DataFrame, factor: str, label: str, level: str, col: str) -> float:
    r = iso[(iso["factor"] == factor) & (iso["label"] == label) & (iso["level"] == level)]
    assert len(r) == 1
    return float(r.iloc[0][f"iso_max_{col}"])


# ----------------------------------------------------------------------------------------------
# fig05n: CD63 rank plots (like Figure 20 panel D)
# ----------------------------------------------------------------------------------------------

def fig05n_cd63_ranks(D: dict) -> pd.DataFrame:
    cells, clr, names = D["cohort"]["cells"], D["cohort"]["clr"], D["cohort"]["names"]
    j = names.index("ADT-CD63")
    mal = (cells["label"] == "malignant").to_numpy()
    subsets = [
        ("All malignant cells", mal),
        ("P09", mal & (cells["patient"] == "P09").to_numpy()),
        ("P09, predicted HSC/MPP2", mal & (cells["patient"] == "P09").to_numpy()
         & (cells["cell_type"] == "HSC/MPP2").to_numpy()),
        ("P03", mal & (cells["patient"] == "P03").to_numpy()),
    ]
    row = mx.d27_row(D["d27"]["adt"], "PC8", "malignant", "CD63")
    strata = D["strata"]
    W, side, gap, top = 7.0, 1.30, 0.33, 0.62
    c = mx.Canvas(W, top + side + 0.55)
    out = []
    for k, (name, m) in enumerate(subsets):
        f, p = cells.loc[m, "PC8"].to_numpy(), clr[m, j]
        n = int(m.sum())
        rho = float(spearmanr(f, p)[0])
        # every rho drawn equals the stored one: pooled and patients from D27, the type group from D29
        if name == "All malignant cells":
            assert abs(rho - row["rho_pooled"]) < TOL
        elif "," not in name:
            assert abs(rho - row[f"rho_{name}"]) < TOL
        else:
            s = strata[(strata["factor"] == "PC8") & (strata["label"] == "malignant") & (strata["kind"] == "patient_celltype")
                       & (strata["patient"] == "P09") & (strata["cell_type"] == "HSC/MPP2") & (strata["feature"] == "ADT-CD63")]
            assert len(s) == 1 and abs(rho - float(s["rho_raw"].iloc[0])) < TOL and int(s["n_cells"].iloc[0]) == n
        rf, rp = rankdata(f), rankdata(p)
        bins = np.array_split(np.argsort(rf, kind="mergesort"), mx.N_BINS)
        bx = np.array([rf[b].mean() for b in bins])
        by = np.array([rp[b].mean() for b in bins])
        x0 = 0.40 + k * (side + gap + 0.03)
        ax = c.ax(x0, top, side, side)
        ax.scatter(rf, rp, s=1.2 if n > 500 else 2.6, color=mx.LIGHT_DOT, linewidth=0, zorder=1, rasterized=True)
        ax.axhline((n + 1) / 2, color=rs.INK_2, lw=0.9, ls=(0, (1, 2)), zorder=2)
        ax.plot(bx, by, color=rs.INK, lw=1.5, marker="o", ms=2.8, mec=rs.SURFACE, mew=0.5, zorder=3)
        pad = n * 0.02
        ax.set_xlim(1 - pad, n + pad)
        ax.set_ylim(1 - pad, n + pad)
        ticks = [1, n]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{t:,}" for t in ticks], fontsize=FS_SMALL - 1.4)
        ax.set_yticklabels([f"{t:,}" for t in ticks], fontsize=FS_SMALL - 1.4)
        ax.tick_params(length=2, pad=1.5)
        ax.grid(axis="both", visible=False)
        ax.set_aspect("equal")
        ax.set_xlabel("PC8 rank", fontsize=FS_SMALL - 0.8, color=rs.INK_2, labelpad=1)
        if k == 0:
            ax.set_ylabel("CD63 rank", fontsize=FS_SMALL - 0.8, color=rs.INK_2, labelpad=-6)
        c.text(x0, 0.12, name, fontsize=FS_SMALL, fontweight="bold", color=rs.INK)
        c.text(x0, 0.32, f"{n:,} cells   ρ = {signed(rho, 3)}", fontsize=FS_SMALL - 0.6, color=rs.INK_2)
        out.append({"subset": name, "n_cells": n, "rho": rho, "decile_mean_cd63_rank_first": by[0],
                    "decile_mean_cd63_rank_last": by[-1], "middle_rank": (n + 1) / 2})
    c.save("fig05n_cd63_ranks", report=True)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# fig05o: CD63 and CD82 taken apart (like Figure 21)
# ----------------------------------------------------------------------------------------------

ANCHORS_O = [("PC8", "CD63", "(a) CD63 × PC8"), ("FA7", "CD63", "(b) CD63 × FA7"), ("PC8", "CD82", "(c) CD82 × PC8")]


def fig05o_cd63_anchors(D: dict) -> pd.DataFrame:
    pats = list(rs.SUPPORTED)
    n_mal = D["d27"]["patient_n"]["malignant"]
    strata, rho29, iso = D["strata"], D["rho"], D["iso"]
    # rows: 7 patients; three summaries (raw); two summaries with total ADT held fixed
    y_p = np.arange(len(pats), dtype=float)
    y_s = len(pats) + 0.5 + np.arange(3, dtype=float)
    y_q = y_s[-1] + 1.5 + np.arange(2, dtype=float)
    labels = ([f"{p} ({n_mal.loc[p]:,})" for p in pats]
              + ["median of the 7", "pooled: all cells", "inside cell types"]
              + ["median of the 7, ADT fixed", "inside types, ADT fixed"])
    fig = plt.figure(figsize=(rs.FULL_W, 3.55))
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.215, right=0.995, top=0.79, bottom=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    out = []
    xlim = (-0.75, 0.75)
    for i, (ax, (fac, prot, title)) in enumerate(zip(axes, ANCHORS_O)):
        r27 = mx.d27_row(D["d27"]["adt"], fac, "malignant", prot)
        r29 = d29_row(rho29, fac, "malignant", prot)
        assert abs(r29["rho_within_median_raw"] - r27["rho_within_median"]) < TOL
        g = strata[(strata["factor"] == fac) & (strata["label"] == "malignant") & (strata["feature_short"] == prot)]
        gp = g[g["kind"] == "patient"].set_index("patient")
        gt = g[g["kind"] == "patient_celltype"]
        assert len(gt) == 11
        bands = {"within": iso_ref(iso, fac, "malignant", "within_patient", "raw"),
                 "pooled": iso_ref(iso, fac, "malignant", "pooled", "raw"),
                 "type": iso_ref(iso, fac, "malignant", "within_type", "raw"),
                 "within_p": iso_ref(iso, fac, "malignant", "within_patient", "partial"),
                 "type_p": iso_ref(iso, fac, "malignant", "within_type", "partial")}
        row_band = ([bands["within"]] * (len(pats) + 1) + [bands["pooled"], bands["type"], bands["within_p"], bands["type_p"]])
        ys = list(y_p) + list(y_s) + list(y_q)
        for y, b in zip(ys, row_band):
            ax.add_patch(Rectangle((-b, y - 0.5), 2 * b, 1.0, facecolor=rs.GRID, edgecolor="none", zorder=0))
        ax.axvline(0, color=rs.INK_2, lw=0.8, zorder=1)
        for xv in (-mx.ROBUST_ABS, mx.ROBUST_ABS):
            ax.axvline(xv, color=rs.MUTED, lw=0.7, ls=(0, (1.5, 2)), zorder=1)
        for y, p in zip(y_p, pats):
            v = float(gp.loc[p, "rho_raw"])
            assert abs(v - r27[f"rho_{p}"]) < TOL
            big = p == "P03"
            ax.plot([0, v], [y, y], color=rs.AXIS, lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([v], [y], ls="none", marker="o", ms=5.6 if big else 4.4, mfc=rs.INK if big else rs.INK_2,
                    mec=rs.SURFACE, mew=0.8, zorder=3)
        raw_vals = [r29["rho_within_median_raw"], r29["rho_pooled_raw"], r29["rho_pct_median_raw"]]
        part_vals = [r29["rho_within_median_partial"], r29["rho_pct_median_partial"]]
        marks = [("o", rs.INK, 6.2), ("D", rs.SURFACE, 5.0), ("s", rs.SURFACE, 5.0)]
        for y, v, (mk, fc, ms) in zip(y_s, raw_vals, marks):
            ax.plot([0, v], [y, y], color=rs.AXIS, lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([v], [y], ls="none", marker=mk, ms=ms, mfc=fc, mec=rs.INK, mew=1.1, zorder=4)
            ax.text(xlim[1] - 0.02, y, signed(v, 3), fontsize=6.4, color=rs.INK, ha="right", va="center", zorder=5)
        for y, v, mk in zip(y_q, part_vals, ("o", "s")):
            ax.plot([0, v], [y, y], color=rs.AXIS, lw=1.0, zorder=2, solid_capstyle="butt")
            ax.plot([v], [y], ls="none", marker=mk, ms=5.4, mfc=rs.SURFACE, mec=rs.INK, mew=1.1, zorder=4)
            ax.text(xlim[1] - 0.02, y, signed(v, 3), fontsize=6.4, color=rs.INK, ha="right", va="center", zorder=5)
        # the 11 patient x type groups as ticks on the two in-type rows (raw and total ADT fixed)
        for y, col in ((y_s[2], "rho_raw"), (y_q[1], "rho_partial")):
            ax.plot(gt[col], np.full(len(gt), y), ls="none", marker="|", ms=7, mew=0.9, color=rs.INK_2, zorder=3)
        ax.set_yticks(ys)
        if i == 0:
            ax.set_yticklabels(labels, fontsize=6.5)
            for lab in ax.get_yticklabels()[len(pats):]:
                lab.set_color(rs.INK)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(ys[-1] + 0.7, -0.7)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.set_xlim(*xlim)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticklabels([signed(-0.5, 1), "0", signed(0.5, 1)], fontsize=6.6)
        ax.set_xlabel("ρ (oriented; malignant cells)", fontsize=6.8)
        ax.axhline(len(pats) - 0.25, color=rs.AXIS, lw=0.8)
        ax.axhline(y_q[0] - 0.75, color=rs.AXIS, lw=0.8)
        ax.set_title(title, fontsize=7.6, pad=24, loc="left", fontweight="bold")
        ax.text(0.0, 1.012, f"within patient {signed(r29['rho_within_median_raw'], 3)} ({int(r29['n_patients_same_sign_raw'])}/7)\n"
                f"inside cell types {signed(r29['rho_pct_median_raw'], 3)} ({int(r29['n_pct_same_sign_raw'])}/11)",
                transform=ax.transAxes, fontsize=6.2, color=rs.INK_2, ha="left", va="bottom", linespacing=1.1)
        out.append({"factor": fac, "protein": prot, **{f"band_{k}": v for k, v in bands.items()},
                    "within_raw": raw_vals[0], "pooled_raw": raw_vals[1], "type_raw": raw_vals[2],
                    "within_partial": part_vals[0], "type_partial": part_vals[1],
                    **{f"rho_{p}": float(gp.loc[p, "rho_raw"]) for p in pats}})
    rs.save(fig, "fig05o_cd63_anchors")
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# fig05q: malignant vs healthy, same sample and predicted type
# ----------------------------------------------------------------------------------------------

Q_PROTEINS = ("CD63", "CD99", "CD82")


def fig05q_mal_vs_healthy(D: dict) -> pd.DataFrame:
    m = D["mvh"]
    m = m[m["level"] == "sample"].copy()
    groups = (m[["patient", "sample", "cell_type", "n_malignant", "n_healthy", "auc_log_total_adt"]]
              .drop_duplicates(["patient", "sample", "cell_type"]).sort_values(["patient", "cell_type"])
              .reset_index(drop=True))
    assert len(groups) == 21
    iso = m[m["is_isotype"]].groupby(["patient", "sample", "cell_type"])["auc_raw"].agg(["min", "max"])
    ys = np.arange(len(groups), dtype=float)
    fig = plt.figure(figsize=(rs.FULL_W, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.10, left=0.215, right=0.985, top=0.905,
                          bottom=0.145)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    out = []
    for ax, prot in zip(axes[:3], Q_PROTEINS):
        g = m[m["feature_short"] == prot].set_index(["patient", "sample", "cell_type"])
        for y, (_, r) in zip(ys, groups.iterrows()):
            key = (r["patient"], r["sample"], r["cell_type"])
            lo, hi = iso.loc[key, "min"], iso.loc[key, "max"]
            ax.add_patch(Rectangle((lo, y - 0.38), hi - lo, 0.76, facecolor=rs.GRID, edgecolor="none", zorder=0))
            raw, adj = float(g.loc[key, "auc_raw"]), float(g.loc[key, "auc_adjusted"])
            ax.plot([raw, adj], [y, y], color=rs.AXIS, lw=0.9, zorder=1)
            ax.plot([adj], [y], ls="none", marker="o", ms=4.0, mfc=rs.SURFACE, mec=rs.INK_2, mew=0.9, zorder=2)
            ax.plot([raw], [y], ls="none", marker="o", ms=4.2, mfc=rs.INK, mec=rs.SURFACE, mew=0.6, zorder=3)
            out.append({"protein": prot, "patient": key[0], "sample": key[1], "cell_type": key[2], "auc_raw": raw,
                        "auc_adjusted": adj, "isotype_auc_min": lo, "isotype_auc_max": hi})
        ax.axvline(0.5, color=rs.INK_2, lw=0.8, zorder=1)
        low = int((g["auc_raw"] < 0.5).sum())
        low_adj = int((g["auc_adjusted"] < 0.5).sum())
        ax.set_title(f"{prot}", fontsize=8.0, fontweight="bold", loc="left", pad=15)
        ax.text(0.0, 1.01, f"malignant lower in {low} of 21\n({low_adj} with total ADT fixed)", transform=ax.transAxes,
                fontsize=6.2, color=rs.INK_2, ha="left", va="bottom", linespacing=1.05)
    ax = axes[3]
    for y, (_, r) in zip(ys, groups.iterrows()):
        v = float(r["auc_log_total_adt"])
        ax.plot([0.5, v], [y, y], color=rs.AXIS, lw=0.9, zorder=1)
        ax.plot([v], [y], ls="none", marker="o", ms=4.2, mfc=rs.MUTED, mec=rs.SURFACE, mew=0.6, zorder=3)
    ax.axvline(0.5, color=rs.INK_2, lw=0.8, zorder=1)
    ax.set_title("total ADT", fontsize=8.0, fontweight="bold", loc="left", pad=15)
    ax.text(0.0, 1.01, f"lower in {int((groups['auc_log_total_adt'] < 0.5).sum())} of 21\n(count per cell)",
            transform=ax.transAxes, fontsize=6.2, color=rs.INK_2, ha="left", va="bottom", linespacing=1.05)
    ylabels = [f"{r['patient']} {r['cell_type']} ({r['n_malignant']} / {r['n_healthy']:,})" for _, r in groups.iterrows()]
    for k, ax in enumerate(axes):
        ax.set_ylim(ys[-1] + 0.7, -0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels(ylabels if k == 0 else [], fontsize=6.3)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"], fontsize=6.6)
        prev = None
        for y, p in zip(ys, groups["patient"]):
            if prev is not None and p != prev:
                ax.axhline(y - 0.5, color=rs.GRID, lw=0.8, zorder=0)
            prev = p
    axes[0].set_xlabel("AUC: P(malignant cell > healthy cell)", fontsize=6.8)
    for ax in axes[1:3]:
        ax.set_xlabel("AUC", fontsize=6.8)
    axes[3].set_xlabel("AUC", fontsize=6.8)
    handles = [Line2D([], [], ls="none", marker="o", ms=4.2, mfc=rs.INK, mec=rs.SURFACE),
               Line2D([], [], ls="none", marker="o", ms=4.0, mfc=rs.SURFACE, mec=rs.INK_2),
               Rectangle((0, 0), 1, 1, facecolor=rs.GRID, edgecolor="none")]
    fig.legend(handles, ["CLR, as in every figure", "with total ADT held fixed", "range of the 9 isotypes (raw)"],
               loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=3, frameon=False, fontsize=6.6, handletextpad=0.3,
               columnspacing=1.2)
    rs.save(fig, "fig05q_mal_vs_healthy")
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# fig05r: Figure 22 with total ADT held fixed
# ----------------------------------------------------------------------------------------------

R_NAMED = {"malignant": ["CD99", "CD63", "CD82", "CD48", "CD116", "CD33", "CD146", "CD71", "CD117"],
           "normal": ["CD34", "CD22", "CD48", "CD63", "CD82"]}
R_OFFSETS = {("malignant", "CD99"): (-5, 7), ("malignant", "CD63"): (-5, -6), ("malignant", "CD82"): (5, -6),
             ("malignant", "CD48"): (5, 6), ("malignant", "CD116"): (5, -6), ("malignant", "CD33"): (5, -5),
             ("malignant", "CD146"): (5, -7), ("malignant", "CD71"): (-5, -4), ("malignant", "CD117"): (-5, 5),
             ("normal", "CD34"): (5, 6), ("normal", "CD22"): (-5, 6), ("normal", "CD48"): (0, -9),
             ("normal", "CD63"): (5, -7), ("normal", "CD82"): (5, 6)}


def _r_panel(ax, D: dict, lab: str) -> pd.DataFrame:
    r = D["rho"]
    a = r[(r["factor"] == "PC8") & (r["label"] == lab)]
    prot, isos = a[~a["is_isotype"]], a[a["is_isotype"]]
    bx = iso_ref(D["iso"], "PC8", lab, "within_patient", "partial")
    by = iso_ref(D["iso"], "PC8", lab, "within_type", "partial")
    lim = 0.36
    ax.axvspan(-bx, bx, color=mx.BAND, lw=0, zorder=0)
    ax.axhspan(-by, by, color=mx.BAND, lw=0, zorder=0)
    ax.add_patch(Rectangle((-bx, -by), 2 * bx, 2 * by, facecolor=mx.BAND_BOTH, edgecolor="none", zorder=0))
    ax.plot([-lim, lim], [-lim, lim], color=rs.MUTED, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    ax.axhline(0, color=rs.INK_2, lw=0.6, zorder=1)
    ax.axvline(0, color=rs.INK_2, lw=0.4, zorder=1)
    for xv in (-mx.ROBUST_ABS, mx.ROBUST_ABS):
        ax.axvline(xv, color=rs.INK_2, lw=0.7, ls=(0, (1, 2)), zorder=1)
    named = prot["feature_short"].isin(R_NAMED[lab])
    ax.scatter(prot.loc[~named, "rho_within_median_partial"], prot.loc[~named, "rho_pct_median_partial"], s=10,
               color=mx.LIGHT_DOT, edgecolor="none", zorder=2)
    ax.scatter(isos["rho_within_median_partial"], isos["rho_pct_median_partial"], s=16, color=mx.ISOTYPE_C,
               edgecolor=rs.SURFACE, linewidth=0.6, zorder=3, marker="D")
    for _, q in prot[named].iterrows():
        x0, y0 = q["rho_within_median_raw"], q["rho_pct_median_raw"]
        x1, y1 = q["rho_within_median_partial"], q["rho_pct_median_partial"]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=rs.MUTED, lw=0.7, mutation_scale=6, shrinkA=2, shrinkB=2.5))
        ax.scatter([x0], [y0], s=14, facecolor="none", edgecolor=rs.MUTED, linewidth=0.7, zorder=3)
        ax.scatter([x1], [y1], s=24, color=rs.INK, edgecolor=rs.SURFACE, linewidth=0.7, zorder=4)
        dx, dy = R_OFFSETS.get((lab, q["feature_short"]), (5, 4))
        ax.annotate(q["feature_short"], (x1, y1), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"), va="center", fontsize=FS_SMALL - 0.8, color=rs.INK, zorder=5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ticks = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    lab_t = [signed(t, 1) if t else "0" for t in ticks]
    ax.set_xticklabels(lab_t, fontsize=FS_SMALL - 1.2)
    ax.set_yticklabels(lab_t, fontsize=FS_SMALL - 1.2)
    ax.grid(False)
    ax.set_xlabel("x: median within-patient ρ, total ADT fixed", fontsize=FS_SMALL - 0.8, color=rs.INK_2)
    ax.set_ylabel("y: median within patient-and-type ρ,\ntotal ADT fixed", fontsize=FS_SMALL - 0.8, color=rs.INK_2)
    setter_x = D["iso"][(D["iso"]["factor"] == "PC8") & (D["iso"]["label"] == lab)
                        & (D["iso"]["level"] == "within_patient")]["iso_max_partial_set_by"].iloc[0]
    ax.text(0.98, 0.02, f"isotype ref.: x ±{bx:.3f}, y ±{by:.3f}" + (f"\n(x set by {setter_x})" if lab == "malignant" else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_SMALL - 1.4, color=rs.INK_2, linespacing=1.1)
    t = a[["feature_short", "is_isotype", "rho_within_median_raw", "rho_within_median_partial",
           "n_patients_same_sign_partial", "rho_pct_median_raw", "rho_pct_median_partial", "n_pct_same_sign_partial",
           "n_pct_strata"]].copy()
    t.insert(0, "cells", mx.SHOW[lab])
    t["iso_ref_x_partial"], t["iso_ref_y_partial"] = bx, by
    return t


def fig05r_two_checks_depth(D: dict) -> pd.DataFrame:
    fig = plt.figure(figsize=(rs.FULL_W, 3.65))
    axA = fig.add_axes([0.085, 0.17, 0.40, 0.40 * rs.FULL_W / 3.65])
    axB = fig.add_axes([0.585, 0.17, 0.40, 0.40 * rs.FULL_W / 3.65])
    tA = _r_panel(axA, D, "malignant")
    tB = _r_panel(axB, D, "normal")
    axA.set_title("(A) Malignant cells", fontsize=8.0, fontweight="bold", loc="left")
    axB.set_title("(B) Healthy cells", fontsize=8.0, fontweight="bold", loc="left")
    handles = [Line2D([], [], ls="none", marker="o", ms=4.6, mfc=rs.INK, mec=rs.SURFACE),
               Line2D([], [], ls="none", marker="o", ms=3.8, mfc="none", mec=rs.MUTED),
               Line2D([], [], ls="none", marker="o", ms=4.0, mfc=mx.LIGHT_DOT, mec="none"),
               Line2D([], [], ls="none", marker="D", ms=4.0, mfc=mx.ISOTYPE_C, mec=rs.SURFACE),
               Line2D([], [], color=rs.MUTED, lw=0.9, ls=(0, (4, 2.5)))]
    fig.legend(handles, ["named protein, total ADT fixed", "its Figure 22 position (arrow: the move)", "other protein",
                         "isotype control (9)", "y = x"], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
               frameon=False, fontsize=6.4, handletextpad=0.3, columnspacing=1.0)
    rs.save(fig, "fig05r_two_checks_depth")
    return pd.concat([tA, tB], ignore_index=True)


def pc8_fa7_agreement(D: dict) -> pd.DataFrame:
    """Spearman of the oriented PC8 and FA7 scores among the cells of one label: pooled and inside each supported
    patient (FA7's CD63 is not an independent confirmation of PC8's when the two scores rank cells alike)."""
    cells = D["cohort"]["cells"]
    rows = []
    for lab in ("malignant", "normal"):
        m = (cells["label"] == lab).to_numpy()
        rows.append({"label": lab, "patient": "pooled", "n_cells": int(m.sum()),
                     "rho_pc8_fa7": float(spearmanr(cells.loc[m, "PC8"], cells.loc[m, "FA7"])[0])})
        for p in rs.SUPPORTED:
            mm = m & (cells["patient"] == p).to_numpy()
            rows.append({"label": lab, "patient": p, "n_cells": int(mm.sum()),
                         "rho_pc8_fa7": float(spearmanr(cells.loc[mm, "PC8"], cells.loc[mm, "FA7"])[0])})
    return pd.DataFrame(rows)


def main() -> None:
    D = load()
    print("PC8 x FA7 (oriented scores)\n", pc8_fa7_agreement(D).round(3).to_string(index=False))
    n = fig05n_cd63_ranks(D)
    o = fig05o_cd63_anchors(D)
    q = fig05q_mal_vs_healthy(D)
    r = fig05r_two_checks_depth(D)
    if mx.OVERFLOW:
        raise SystemExit("\n".join(mx.OVERFLOW))
    pd.set_option("display.width", 200)
    print("fig05n\n", n.round(3).to_string(index=False))
    print("fig05o\n", o.round(3).T.to_string())
    qq = q.pivot_table(index=["patient", "cell_type"], columns="protein", values=["auc_raw", "auc_adjusted"]).round(3)
    print("fig05q\n", qq.to_string())
    named = r[r["feature_short"].isin(set(R_NAMED["malignant"]) | set(R_NAMED["normal"]))]
    print("fig05r\n", named.round(3).to_string(index=False))
    for lab, g in r.groupby("cells"):
        p = g[~g["is_isotype"]]
        c1 = (p["rho_within_median_partial"].abs() >= mx.ROBUST_ABS) & (p["n_patients_same_sign_partial"] >= mx.ROBUST_SAME_SIGN)
        c2 = p["rho_pct_median_partial"].abs() > p["iso_ref_y_partial"]
        xo = p["rho_within_median_partial"].abs() > p["iso_ref_x_partial"]
        print(f"{lab}: total ADT fixed: check 1 {int(c1.sum())} {sorted(p.loc[c1, 'feature_short'])}; "
              f"outside y band {int(c2.sum())}; outside both bands {int((xo & c2).sum())} "
              f"{sorted(p.loc[xo & c2, 'feature_short'])}")


if __name__ == "__main__":
    main()
