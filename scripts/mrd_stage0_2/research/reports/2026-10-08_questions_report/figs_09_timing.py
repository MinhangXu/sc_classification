"""Figures for section 09 (remaining TODOs): MRD sample timing for the supported patients.

fig09a_timeline  one row per supported patient on a days-since-transplant axis: the
                 transplant-to-relapse span, each MRD draw with its malignant / total cell
                 count, and the gap from the malignant-bearing draw to relapse.
fig09b_timing    per-patient median lift (D6, median over 643 representations) against
                 days since transplant, days before relapse, and malignant prevalence; the
                 last panel is the D6 reference association. n = 7, descriptive only.

Data (EXP-relative, read only):
  analysis/scorecards/stage2_diagnostic/d1a_deep_dive_20260910/
      d1dd_rank_matched_confusion_by_sample_collection.csv  (D1DD13; per-sample timing,
      malignant and normal counts; one row per panel x sample, the counts repeat per panel)
  analysis/scorecards/stage2_diagnostic/meeting_followup_v1/old34_patient_metric_summary.csv
      (D6; per-patient prevalence and median raw lift)
  analysis/scorecards/stage2_diagnostic/meeting_followup_v1/old34_patient_level_metric_associations.csv
      (D6; Spearman rho of median lift vs prevalence, n = 7)
  analysis/scorecards/stage2_diagnostic/patient_count_learning_curve_v1/d15b_full_unit_results.csv
      (D15b full design; main() also prints the Option 3 pair-vs-LOPO table of section 09 and the
      median / P20 lift over the seven supported patients vs the four with >= 100 malignant cells)

Note: d1dd_supported_patient_collection_context.csv lists P09's days_from_relapse in an
order that does not pair with its days_from_sct column, so it is not used here. Each
draw's days from transplant minus its days from relapse gives the same relapse day for
P09 (723) in the per-sample table, which this module asserts.

Run from the repo root or the report dir:
  python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_09_timing.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

import report_style as rs  # noqa: E402

EXPECTED_TOTAL = {"P01": 11860, "P02": 1462, "P03": 2327, "P04": 1128, "P06": 3259,
                  "P07": 2185, "P09": 17126}
EXPECTED_MAL = {"P01": 45, "P02": 211, "P03": 758, "P04": 102, "P06": 44, "P07": 22, "P09": 375}


DOT = rs.INK_2  # one series per panel, not a method: dark neutral (blue is reserved for PCA)


def load_samples() -> pd.DataFrame:
    """One row per supported MRD sample: timing, malignant and total cells, relapse day."""
    path = rs.diag_dir("d1a_deep_dive_20260910") / "d1dd_rank_matched_confusion_by_sample_collection.csv"
    df = pd.read_csv(path)
    cols = ["patient", "sample", "Time", "Days from SCT", "Days.from.Relapse", "n_malignant", "n_normal"]
    per_panel = [g[cols].sort_values("sample").reset_index(drop=True) for _, g in df.groupby("panel")]
    for other in per_panel[1:]:  # counts and timing are sample properties: identical across panels
        pd.testing.assert_frame_equal(per_panel[0], other)
    s = per_panel[0].rename(columns={"Days from SCT": "days_sct", "Days.from.Relapse": "days_rel"})
    s["n_total"] = s["n_malignant"] + s["n_normal"]
    s["relapse_day"] = s["days_sct"] - s["days_rel"]
    assert set(s["patient"]) == set(rs.SUPPORTED)
    tot = s.groupby("patient")[["n_total", "n_malignant"]].sum()
    assert tot["n_total"].to_dict() == EXPECTED_TOTAL
    assert tot["n_malignant"].to_dict() == EXPECTED_MAL
    # One relapse day per patient (the P09 pairing check).
    assert (s.groupby("patient")["relapse_day"].nunique() == 1).all()
    return s


def load_patient_summary() -> pd.DataFrame:
    path = rs.diag_dir("meeting_followup_v1") / "old34_patient_metric_summary.csv"
    df = pd.read_csv(path).set_index("heldout_patient").loc[list(rs.SUPPORTED)]
    assert (df["n_representations"] == 643).all()
    return df


def d6_rho_lift_prevalence() -> float:
    path = rs.diag_dir("meeting_followup_v1") / "old34_patient_level_metric_associations.csv"
    df = pd.read_csv(path)
    row = df[(df["metric"] == "median_raw_lift") & (df["covariate"] == "prevalence")]
    assert len(row) == 1 and int(row["n_patients"].iloc[0]) == 7
    return float(row["spearman_rho_n7"].iloc[0])


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rho and its exact two-sided permutation p (all n! orderings; n = 7 -> 5,040)."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    hits = 0
    total = 0
    for perm in itertools.permutations(ry):
        total += 1
        if abs(np.corrcoef(rx, perm)[0, 1]) >= abs(rho) - 1e-12:
            hits += 1
    return rho, hits / total


def main_draw(s: pd.DataFrame) -> pd.DataFrame:
    """Per patient, the draw holding most of its malignant cells (P09: MRD_4, 372 of 375)."""
    idx = s.groupby("patient")["n_malignant"].idxmax()
    return s.loc[idx].set_index("patient").loc[list(rs.SUPPORTED)]


def timing_table() -> pd.DataFrame:
    """Per-patient values behind fig09b (also used for the numbers ledger)."""
    s = load_samples()
    m = main_draw(s)
    ps = load_patient_summary()
    out = pd.DataFrame({
        "days_sct_main_draw": m["days_sct"],
        "days_before_relapse_main_draw": -m["days_rel"],
        "relapse_day": m["relapse_day"],
        "prevalence_pct": 100 * ps["prevalence"],
        "median_lift_pp": 100 * ps["median_raw_lift"],
    })
    return out


def fig09a_timeline() -> Path:
    rs.apply()
    s = load_samples()
    m = main_draw(s)
    patients = list(rs.SUPPORTED)
    ypos = {p: i for i, p in enumerate(patients)}  # P01 at top after inversion

    fig, ax = plt.subplots(figsize=(rs.FULL_W, 2.75))
    fig.subplots_adjust(left=0.07, right=0.74)  # keep the count column and bracket inside FULL_W
    ax.grid(False)
    ax.grid(True, axis="x", color=rs.GRID, lw=0.6)
    for p in patients:
        y = ypos[p]
        rel = float(m.loc[p, "relapse_day"])
        ax.plot([0, rel], [y, y], color=rs.AXIS, lw=rs.LINE_W, zorder=1, solid_capstyle="butt")
        ax.plot(rel, y, marker="|", ms=9, mew=1.6, color=rs.INK_2, zorder=2)
        # Gap from the malignant-bearing draw to relapse, labelled above the segment.
        d = float(m.loc[p, "days_sct"])
        gap = rel - d
        gap_lab = f"{gap:.0f} d to relapse" if p == "P01" else f"{gap:.0f} d"
        ax.text((d + rel) / 2, y - 0.2, gap_lab, ha="center", va="bottom", fontsize=6.8,
                color=rs.INK_2)
        rows = s[s["patient"] == p].sort_values("days_sct")
        for _, r in rows.iterrows():
            main = r["sample"] == m.loc[p, "sample"]
            ax.plot(r["days_sct"], y, "o", ms=rs.MARKER,
                    color=DOT if main else rs.SURFACE, mec=DOT if not main else rs.SURFACE,
                    mew=rs.RING if main else 1.3, zorder=3)
            if p == "P09":
                above = r["Time"] == "MRD_3"  # stagger the two close labels
                ax.text(r["days_sct"], y - 0.2 if above else y + 0.3,
                        f"{int(r['n_malignant'])} malignant", ha="center",
                        va="bottom" if above else "top", fontsize=6.8, color=rs.INK_2)
        n_mal = int(rows["n_malignant"].sum())
        n_tot = int(rows["n_total"].sum())
        ax.text(1.005, y, f"{n_mal:,} / {n_tot:,}  ({100 * n_mal / n_tot:.1f}%)",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=7,
                color=rs.INK_2)
    ax.text(1.005, -0.62, "malignant / all cells (prevalence)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6.8, color=rs.MUTED)

    # Bracket the highest-prevalence patients to the right of the count column.
    hp = [ypos[p] for p in rs.HIGH_PREVALENCE]
    xb = 1.245
    tr = ax.get_yaxis_transform()
    ax.plot([xb - 0.008, xb, xb, xb - 0.008], [min(hp) - 0.3, min(hp) - 0.3, max(hp) + 0.3, max(hp) + 0.3],
            color=rs.INK_2, lw=0.9, transform=tr, clip_on=False, solid_capstyle="butt")
    ax.text(xb + 0.01, (min(hp) + max(hp)) / 2, "highest\nprevalence", transform=tr, ha="left",
            va="center", fontsize=7, color=rs.INK_2, linespacing=1.0)

    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels(patients)
    ax.set_ylim(len(patients) - 0.35, -0.85)
    ax.set_xlim(0, 860)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])
    ax.set_xlabel("days since stem-cell transplant (SCT)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    handles = [
        Line2D([], [], color=rs.AXIS, lw=rs.LINE_W, label="SCT to relapse"),
        Line2D([], [], ls="", marker="o", ms=rs.MARKER, color=DOT, mec=rs.SURFACE, mew=rs.RING,
               label="draw holding the malignant cells"),
        Line2D([], [], ls="", marker="o", ms=rs.MARKER, color=rs.SURFACE, mec=DOT, mew=1.3,
               label="earlier P09 draw"),
        Line2D([], [], ls="", marker="|", ms=9, mew=1.6, color=rs.INK_2, label="relapse"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4,
              handlelength=1.4, columnspacing=1.4, borderaxespad=0.2, fontsize=7)
    ax.set_title("Every draw holding a patient's malignant cells sits 28\u2013118 days before relapse", pad=6)
    return rs.save(fig, "fig09a_timeline")


def fig09b_timing() -> Path:
    rs.apply()
    s = load_samples()
    t = timing_table()
    rho_prev_d6 = d6_rho_lift_prevalence()
    lift = t["median_lift_pp"].to_numpy()

    rho_sct, p_sct = exact_spearman(t["days_sct_main_draw"].to_numpy(), lift)
    rho_rel, p_rel = exact_spearman(t["days_before_relapse_main_draw"].to_numpy(), lift)
    rho_prev, p_prev = exact_spearman(t["prevalence_pct"].to_numpy(), lift)
    print(f"timing rho/p: since SCT {rho_sct:.3f}/{p_sct:.4f}; before relapse {rho_rel:.3f}/{p_rel:.4f}; "
          f"prevalence {rho_prev:.3f}/{p_prev:.4f}")
    assert abs(rho_prev - rho_prev_d6) < 1e-9  # the reference panel reproduces D6

    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL_W, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.11, right=0.985)
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, axis="both", color=rs.GRID, lw=0.6)
    axes[0].set_ylim(0.13, 40)
    axes[0].yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].set_yticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
    axes[0].set_ylabel("median lift per patient\n(pp, log scale)")

    xcols = ["days_sct_main_draw", "days_before_relapse_main_draw", "prevalence_pct"]
    # Label offsets (points) per panel to keep patient labels clear of each other.
    offsets = [
        {"P01": (5, -2), "P02": (5, 0), "P03": (-5, 0, "right"), "P04": (5, 0), "P06": (5, -5),
         "P07": (5, 2), "P09": (0, 8, "center")},
        {"P01": (5, -3), "P02": (5, 0), "P03": (-5, 2, "right"), "P04": (-5, 0, "right"),
         "P06": (-5, -3, "right"), "P07": (5, 3), "P09": (5, 0)},
        {"P01": (5, -2), "P02": (-5, 3, "right"), "P03": (-5, 0, "right"), "P04": (5, -4),
         "P06": (5, -3), "P07": (5, 3), "P09": (5, 0)},
    ]
    for k, (ax, col) in enumerate(zip(axes, xcols)):
        x = t[col].to_numpy()
        ax.plot(x, lift, "o", ls="", ms=rs.MARKER, color=DOT, mec=rs.SURFACE, mew=rs.RING, zorder=3)
        for p, xi, yi in zip(t.index, x, lift):
            dx, dy, *ha = offsets[k][p]
            ax.annotate(p, (xi, yi), xytext=(dx, dy), textcoords="offset points",
                        ha=ha[0] if ha else "left", va="center", fontsize=7,
                        color=rs.INK if p in rs.HIGH_PREVALENCE else rs.INK_2,
                        fontweight="bold" if p in rs.HIGH_PREVALENCE else "normal")

    # Panel 1: P09's earlier draws as hollow markers at the same lift.
    p09 = s[(s["patient"] == "P09")].sort_values("days_sct")
    y09 = float(t.loc["P09", "median_lift_pp"])
    axes[0].plot(p09["days_sct"], [y09] * len(p09), color=rs.AXIS, lw=0.9, zorder=1)
    early = p09[p09["days_sct"] < float(t.loc["P09", "days_sct_main_draw"])]
    axes[0].plot(early["days_sct"], [y09] * len(early), "o", ls="", ms=rs.MARKER - 1, color=rs.SURFACE,
                 mec=DOT, mew=1.2, zorder=2)
    axes[0].text(float(early["days_sct"].mean()), y09 / 1.3, "P09 earlier draws", ha="center", va="top",
                 fontsize=6.6, color=rs.MUTED)
    axes[0].set_xlim(0, 760)
    axes[0].set_xticks([0, 200, 400, 600])
    axes[0].set_xlabel("days since SCT\n(malignant-bearing draw)")
    # Panel 2: earlier P09 draws are off this axis; say so.
    axes[1].set_xlim(0, 135)
    axes[1].set_xticks([0, 30, 60, 90, 120])
    axes[1].set_xlabel("days before relapse\n(malignant-bearing draw)")
    axes[1].text(0.03, 0.03, "P09's earlier draws:\n294\u2013586 d (off axis)", transform=axes[1].transAxes,
                 ha="left", va="bottom", fontsize=6.6, color=rs.MUTED, linespacing=1.0)
    # Panel 3: prevalence, log scale (0.4% to 33%).
    axes[2].set_xscale("log")
    axes[2].set_xlim(0.25, 60)
    axes[2].xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
    axes[2].xaxis.set_minor_locator(NullLocator())
    axes[2].set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    axes[2].set_xlabel("malignant prevalence\n(%, log scale)")

    for ax, rho, pv in ((axes[0], rho_sct, p_sct), (axes[1], rho_rel, p_rel), (axes[2], rho_prev_d6, p_prev)):
        ptxt = f"{pv:.3f}" if pv < 0.1 else f"{pv:.2f}"
        ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.2f}\nexact p = {ptxt}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=rs.INK_2, linespacing=1.05)
    fig.suptitle("Across 7 patients, lift tracks prevalence more closely than timing (descriptive)",
                 x=0.015, y=1.02, ha="left", fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig09b_timing")


def option3_pairs() -> pd.DataFrame:
    """Table 09opt3: D15b full design, tuned arm. For each highest-prevalence patient held out,
    lift when training on exactly the other two (k = 2) vs the ordinary LOPO fold (k = 6)."""
    path = rs.diag_dir("patient_count_learning_curve_v1") / "d15b_full_unit_results.csv"
    u = pd.read_csv(path)
    u = u[u["arm"] == "tuned"]
    hp = set(rs.HIGH_PREVALENCE)
    rows = []
    for (rep, h), g in u[u["heldout_patient"].isin(hp)].groupby(["representation", "heldout_patient"]):
        pair = g[(g["k"] == 2) & (g["subset"].str.split("+").apply(set) == hp - {h})]
        full = g[g["k"] == 6]
        assert len(pair) == 1 and len(full) == 1
        assert int(pair["inner_n_folds"].iloc[0]) == 2  # inner LOPO over the two training patients
        rows.append({"representation": rep, "heldout_patient": h,
                     "n_train_pair": int(pair["n_train"].iloc[0]), "n_train_lopo": int(full["n_train"].iloc[0]),
                     "lift_pair_pp": 100 * float(pair["heldout_lift"].iloc[0]),
                     "lift_lopo_pp": 100 * float(full["heldout_lift"].iloc[0])})
    out = pd.DataFrame(rows)
    out["diff_pp"] = out["lift_pair_pp"] - out["lift_lopo_pp"]
    assert len(out) == 12
    return out


AT_LEAST_100 = ("P02", "P03", "P04", "P09")  # supported patients with >= 100 malignant cells (study_config counts)


def inclusion_summaries() -> pd.DataFrame:
    """Table 09incl, last rows: median and P20 lift over the seven supported held-out patients vs over the
    four with >= 100 malignant cells (the user's 2026-10-10 choice for the new-data runs), from the stored
    LOPO folds (k = 6) of the D15b patient-count curve, tuned arm. Nothing is refit; P20 = pandas
    quantile(0.20), as in the stage2 audit tables."""
    assert all(EXPECTED_MAL[p] >= 100 for p in AT_LEAST_100)
    assert all(EXPECTED_MAL[p] < 100 for p in rs.SUPPORTED if p not in AT_LEAST_100)
    path = rs.diag_dir("patient_count_learning_curve_v1") / "d15b_full_unit_results.csv"
    u = pd.read_csv(path)
    u = u[(u["arm"] == "tuned") & (u["k"] == 6)]
    rows = []
    for rep, g in u.groupby("representation"):
        lift = 100 * g.set_index("heldout_patient")["heldout_lift"]
        assert sorted(lift.index) == sorted(rs.SUPPORTED)
        four = lift.loc[list(AT_LEAST_100)]
        rows.append({"representation": rep, "median_7": lift.median(), "p20_7": lift.quantile(0.20),
                     "median_4": four.median(), "p20_4": four.quantile(0.20),
                     "median_patient_7": lift.sub(lift.median()).abs().idxmin()})
    out = pd.DataFrame(rows).set_index("representation")
    out["median_ratio"] = out["median_4"] / out["median_7"]
    return out


def main() -> None:
    for fn in (fig09a_timeline, fig09b_timing):
        print(fn())
    print(timing_table().round(4).to_string())
    o3 = option3_pairs()
    print(o3.round(2).to_string(index=False))
    d = o3["diff_pp"]
    print(f"Option 3 pairs: median diff {d.median():+.2f} pp, {int((d > 0).sum())}/12 higher, "
          f"range {d.min():+.2f} to {d.max():+.2f} pp")
    inc = inclusion_summaries()
    print("Inclusion >= 10 (7 patients) vs >= 100 (P02, P03, P04, P09): LOPO fold, tuned arm")
    print(inc.round(4).to_string())


if __name__ == "__main__":
    main()
