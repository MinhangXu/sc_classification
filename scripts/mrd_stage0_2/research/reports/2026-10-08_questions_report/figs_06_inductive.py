"""Section 06 figures: option 2, an inductive Stage 1 (D17 basis-variant pilot, IFN-alpha).

Reads only the D17 slug tables (EXP/analysis/scorecards/stage2_diagnostic/basis_variant_pilot_ifna_v1/):
  d17_stage2_by_patient.csv  one row per basis x held-out patient x C policy
  d17_stage2_summary.csv     pooled AP / median / P20 lift per arm x method x K x policy

Figures (figures/<name>.pdf + .png):
  fig06a_inductive_lift    per-patient lift at K 40, tuned L1: transductive refit vs inductive vs normal-only
  fig06b_paired_change     per-patient change vs the transductive refit over the 18 tuned/L2 summaries,
                           inductive and normal-only arms

Run: python figs_06_inductive.py   (prints the numbers quoted in sections/06_inductive_stage1.tex)
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

SLUG = "basis_variant_pilot_ifna_v1"
PATIENTS = list(rs.SUPPORTED)
TUNED_OR_L2 = ("l1_tuned", "l2_fixed", "l2_tuned")
POLICY_LABEL = {"l1_tuned": "tuned L1", "l2_fixed": "fixed L2", "l2_tuned": "tuned L2", "l1_fixed": "fixed L1"}

# Arm encodings (shape/fill carry the arm; hue carries the method).
ARM_STYLE = {
    "transductive_refit": dict(marker="o", filled=False, label="transductive refit: Stage 1 fit on all 13 patients"),
    "inductive": dict(marker="o", filled=True, label="inductive: Stage 1 fit without the held-out patient"),
    "normal_only": dict(marker="D", filled=True, label="normal-only: Stage 1 fit on the other patients' normal cells"),
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = rs.diag_dir(SLUG)
    by_patient = pd.read_csv(d / "d17_stage2_by_patient.csv")
    summary = pd.read_csv(d / "d17_stage2_summary.csv")
    return by_patient, summary


def _sel(df: pd.DataFrame, arm: str, method: str, k: int, policy: str) -> pd.DataFrame:
    out = df[(df.arm == arm) & (df.stage1_method == method) & (df.effective_k == k) & (df.policy == policy)]
    out = out.set_index("heldout_patient").loc[PATIENTS]
    return out


def _marker(ax, x, y, color, arm, size=rs.MARKER, z=3):
    st = ARM_STYLE[arm]
    face = color if st["filled"] else rs.SURFACE
    ms = size * (0.85 if st["marker"] == "D" else 1.0)
    ax.plot(
        x, y, linestyle="none", marker=st["marker"], markersize=ms, markerfacecolor=face,
        markeredgecolor=color if not st["filled"] else rs.SURFACE,
        markeredgewidth=rs.LINE_W * 0.9 if not st["filled"] else rs.RING, zorder=z,
    )


def _patient_ticks(ax, by_patient: pd.DataFrame, with_prevalence: bool = True) -> None:
    prev = (
        by_patient[(by_patient.arm == "transductive_refit")]
        .drop_duplicates("heldout_patient")
        .set_index("heldout_patient")
        .loc[PATIENTS, "prevalence"]
    )
    labels = [f"{p}\n{100 * prev[p]:.1f}%" if with_prevalence else p for p in PATIENTS]
    ax.set_xticks(range(len(PATIENTS)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(PATIENTS) - 0.4)


def _bracket(ax, y, text="highest prevalence") -> None:
    i0, i1 = PATIENTS.index("P02"), PATIENTS.index("P04")
    tick = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([i0 - 0.3, i0 - 0.3, i1 + 0.3, i1 + 0.3], [y - tick, y, y, y - tick], color=rs.INK_2,
            linewidth=0.8, solid_capstyle="butt", clip_on=False, zorder=1)
    ax.text((i0 + i1) / 2, y + tick * 0.4, text, ha="center", va="bottom", fontsize=7.5, color=rs.INK_2)


def _fmt_pp(v: float) -> str:
    return f"{v:+.2f}".replace("-", "−") + " pp"


# --------------------------------------------------------------------------------------------- fig06a
def fig06a(by_patient: pd.DataFrame, numbers: list) -> None:
    """K 40, tuned L1: per-patient lift for the transductive refit, inductive and normal-only bases."""
    policy = "l1_tuned"
    offs = {"transductive_refit": -0.22, "inductive": 0.0, "normal_only": 0.22}
    fig, axes = plt.subplots(1, 2, figsize=(rs.FULL_W, 3.05), sharey=True, layout="constrained")
    for c, method in enumerate(("pca", "fa")):
        ax = axes[c]
        color = rs.METHOD[method]
        vals = {arm: _sel(by_patient, arm, method, 40, policy) for arm in offs}
        x = np.arange(len(PATIENTS))
        for i in x:
            ax.plot([i + o for o in offs.values()], [vals[a].lift_pp.iloc[i] for a in offs], color=rs.MUTED,
                    linewidth=1.0, zorder=2)
        for arm, o in offs.items():
            _marker(ax, x + o, vals[arm].lift_pp.values, color, arm, size=rs.MARKER * 0.95)
        tr = vals["transductive_refit"].lift_pp
        sub = []
        for arm in ("inductive", "normal_only"):
            lift = vals[arm].lift_pp
            delta = lift - tr
            n_lower = int((delta < 0).sum())
            med = float(np.median(delta))
            sub.append(f"{'inductive' if arm == 'inductive' else 'normal-only'}: median change {_fmt_pp(med)}, {n_lower}/7 lower")
            numbers.append((f"{med:.2f}", f"fig06a {method.upper()}40 {policy} {arm} median per-patient change vs refit (pp); {n_lower}/7 lower",
                            "d17_stage2_by_patient.csv", "median over 7 supported patients of lift_pp(arm) - lift_pp(transductive_refit)"))
            numbers.append((f"{int((lift > 0).sum())}/7", f"fig06a {method.upper()}40 {policy} {arm} lift>0",
                            "d17_stage2_by_patient.csv", "count lift_pp>0 over supported patients"))
            if arm != "inductive":
                continue
            # Label inductive single-patient changes of at least 1.5 pp.
            for j in np.flatnonzero(np.abs(delta.values) >= 1.5):
                yi = lift.iloc[j]
                down = delta.iloc[j] < 0
                ax.annotate(_fmt_pp(delta.iloc[j]), xy=(j, yi), xytext=(j + 0.5, yi - 4.5 if down else yi + 2.6),
                            fontsize=7.5, color=rs.INK, ha="left", va="center",
                            arrowprops=dict(arrowstyle="-", color=rs.INK_2, linewidth=0.7, shrinkA=0, shrinkB=3))
                numbers.append((f"{delta.iloc[j]:.2f}", f"fig06a {method.upper()}40 {policy} inductive change >= 1.5 pp ({PATIENTS[j]})",
                                "d17_stage2_by_patient.csv", "inductive minus refit lift_pp for that patient"))
        ax.set_title(f"{rs.METHOD_LABEL[method]}, K = 40 \u00b7 C tuned per fold, L1", loc="left", fontsize=8.5,
                     fontweight="bold", color=rs.INK, pad=24)
        ax.text(0.0, 1.015, "\n".join(sub), transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.5, color=rs.INK_2, linespacing=1.15)
        _patient_ticks(ax, by_patient, with_prevalence=True)
        ax.set_xlabel("held-out patient (malignant prevalence \u03c0)", fontsize=8)
        if c == 0:
            ax.set_ylabel("held-out lift (pp)")
        ax.set_ylim(-1, 26)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        _bracket(ax, 23.0)
    handles = []
    for arm in offs:
        st = ARM_STYLE[arm]
        handles.append(Line2D([], [], linestyle="none", marker=st["marker"],
                              markersize=rs.MARKER * (0.8 if st["marker"] == "D" else 0.95),
                              markerfacecolor=rs.INK_2 if st["filled"] else rs.SURFACE,
                              markeredgecolor=rs.SURFACE if st["filled"] else rs.INK_2,
                              markeredgewidth=rs.RING if st["filled"] else rs.LINE_W * 0.9,
                              label=st["label"]))
    fig.legend(handles=handles, loc="outside lower left", ncol=2, frameon=False, fontsize=7.5,
               handletextpad=0.4, columnspacing=1.2)
    fig.suptitle("Stage 1 fit without the held-out patient, or on normal cells only,\n"
                 "keeps lift positive in all 7 patients",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=rs.INK)
    rs.save(fig, "fig06a_inductive_lift")


# --------------------------------------------------------------------------------------------- fig06b
def fig06b(by_patient: pd.DataFrame, numbers: list) -> None:
    groups = [(m, k) for m in ("pca", "fa") for k in (10, 20, 40)]
    xs, xlabels, xinfo = [], [], []
    pos = 0.0
    for g, (m, k) in enumerate(groups):
        for p in TUNED_OR_L2:
            xs.append(pos)
            xlabels.append(POLICY_LABEL[p])
            xinfo.append((m, k, p))
            pos += 1.0
        pos += 0.6
    arms = (("inductive", "Inductive (all cells of the other 12 patients)"),
            ("normal_only", "Normal-only (normal cells of the other 12 patients)"))
    fig, axes = plt.subplots(2, 1, figsize=(rs.FULL_W, 4.1), sharex=True, sharey=True, layout="constrained")
    jit = np.linspace(-0.22, 0.22, len(PATIENTS))
    all_meds = {}
    for r, (arm, title) in enumerate(arms):
        ax = axes[r]
        ax.axhline(0, color=rs.INK_2, linewidth=0.8, zorder=1)
        meds = []
        rows = []
        for x, (m, k, p) in zip(xs, xinfo):
            s = _sel(by_patient, arm, m, k, p)
            dlt = s.delta_vs_transductive_refit_pp.values
            ax.plot(x + jit, dlt, linestyle="none", marker="o", markersize=rs.MARKER * 0.72,
                    markerfacecolor=rs.MUTED, markeredgecolor=rs.SURFACE, markeredgewidth=rs.RING * 0.8, zorder=2)
            med = float(np.median(dlt))
            meds.append(med)
            ax.plot([x - 0.36, x + 0.36], [med, med], color=rs.METHOD[m], linewidth=rs.LINE_W * 1.6,
                    solid_capstyle="butt", zorder=3)
            for pat, v in zip(PATIENTS, dlt):
                rows.append((m, k, p, pat, v, x))
            numbers.append((f"{med:.3f}", f"fig06b {arm} {m.upper()}{k} {p} median per-patient change vs refit (pp)",
                            "d17_stage2_by_patient.csv", "median of delta_vs_transductive_refit_pp over 7 supported patients"))
        all_meds[arm] = meds
        rows = pd.DataFrame(rows, columns=["m", "k", "p", "pat", "v", "x"])
        # Label the most negative and most positive single patients, inside their own (method, K) column:
        # the label sits below the lowest dot and above the highest one, centred on the column.
        for idx, below in ((rows.v.idxmin(), True), (rows.v.idxmax(), False)):
            rr = rows.loc[idx]
            xj = rr.x + jit[PATIENTS.index(rr.pat)]
            g = groups.index((rr.m, rr.k))
            xc = float(np.mean(xs[3 * g: 3 * g + 3]))
            lab = f"{rs.METHOD_LABEL[rr.m]}{rr.k} {rr.pat}\n{_fmt_pp(rr.v)}"
            ty = rr.v - 1.0 if below else rr.v + 1.0
            ax.annotate(lab, xy=(xj, rr.v), xytext=(xc, ty), fontsize=7.2, color=rs.INK, ha="center",
                        va="top" if below else "bottom", linespacing=1.05,
                        arrowprops=dict(arrowstyle="-", color=rs.INK_2, linewidth=0.7, shrinkA=0, shrinkB=3))
            numbers.append((f"{rr.v:.2f}", f"fig06b {arm} extreme single-patient change {rr.m.upper()}{rr.k} {rr.p} {rr.pat}",
                            "d17_stage2_by_patient.csv", "min/max of delta_vs_transductive_refit_pp over 18 summaries x 7 patients"))
        lo, hi = min(meds), max(meds)
        numbers.append((f"{lo:.2f} to {hi:.2f}", f"fig06b {arm} range of the 18 per-summary medians (pp)",
                        "d17_stage2_by_patient.csv", "min and max over 18 (method,K,policy) of the per-patient median change"))
        n_pos = int((by_patient[(by_patient.arm == arm) & by_patient.policy.isin(TUNED_OR_L2)].lift_pp > 0).sum())
        numbers.append((f"{n_pos}/126", f"fig06b {arm} patient x summary units with lift>0 (tuned/L2)",
                        "d17_stage2_by_patient.csv", "count lift_pp>0 over 18 summaries x 7 patients"))
        ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=rs.INK)
        ax.set_title(f"medians {_fmt_pp(lo).replace(' pp', '')} to {_fmt_pp(hi)}", loc="right", fontsize=7.5,
                     fontweight="normal", color=rs.INK_2)
        ax.set_ylim(-8.4, 11.6)
        ax.set_yticks([-8, -4, 0, 4, 8])
    ax = axes[1]
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=90)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    centers = [float(np.mean(xs[3 * g: 3 * g + 3])) for g in range(len(groups))]
    for a in axes:
        for g in range(1, len(groups)):
            a.axvline((xs[3 * g - 1] + xs[3 * g]) / 2, color=rs.GRID, linewidth=0.8, zorder=0)
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(centers)
    top.set_xticklabels([f"{rs.METHOD_LABEL[m]}, K = {k}" for m, k in groups], fontsize=7.5, color=rs.INK)
    top.tick_params(length=0, pad=2)
    top.spines["top"].set_visible(False)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=rs.MARKER * 0.72, markerfacecolor=rs.MUTED,
               markeredgecolor=rs.SURFACE, label="one supported patient (7 per column)"),
        Line2D([], [], color=rs.METHOD["pca"], linewidth=rs.LINE_W * 1.6, solid_capstyle="butt", label="median, PCA"),
        Line2D([], [], color=rs.METHOD["fa"], linewidth=rs.LINE_W * 1.6, solid_capstyle="butt", label="median, FA"),
    ]
    fig.supylabel("change in held-out lift vs transductive refit (pp)", fontsize=8.5, color=rs.INK_2)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("Leaving the held-out patient out of Stage 1 lowers its lift\n"
                 "by 0.05\u20131.3 pp at the median (top row)",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=rs.INK)
    rs.save(fig, "fig06b_paired_change")


def fixed_c_numbers(by_patient: pd.DataFrame, numbers: list) -> None:
    """Fixed-C (1e-4) L1 entry margins and all-zero folds at K 40 (quoted in the text, no figure)."""
    for method in ("pca", "fa"):
        for arm in ("transductive_refit", "inductive", "normal_only"):
            s = _sel(by_patient, arm, method, 40, "l1_fixed")
            numbers.append((f"{int((s.n_nonzero == 0).sum())}/7", f"{method.upper()}40 {arm} all-zero folds at C=1e-4 (L1)",
                            "d17_stage2_by_patient.csv", "count n_nonzero==0, policy l1_fixed"))
            numbers.append((f"{int((s.lift_pp > 0).sum())}/7 ; median {np.median(s.lift_pp):.2f}",
                            f"{method.upper()}40 {arm} lift>0 and median lift at C=1e-4 (L1)",
                            "d17_stage2_by_patient.csv", "policy l1_fixed"))
            numbers.append((";".join(f"{p}:{v:.3f}" for p, v in s.max_entry_margin.items()),
                            f"{method.upper()}40 {arm} largest entry margin per patient at C=1e-4",
                            "d17_stage2_by_patient.csv", "max_entry_margin, policy l1_fixed"))
    j = "P07"
    for arm in ("transductive_refit", "transductive_stored", "inductive"):
        for pol in ("l1_fixed", "l1_tuned"):
            s = _sel(by_patient, arm, "pca", 40, pol)
            numbers.append((f"lift {s.loc[j, 'lift_pp']:.2f}; margin {s.loc[j, 'max_entry_margin']:.3f}; C {s.loc[j, 'C']:g}",
                            f"PCA40 {j} {arm} {pol}", "d17_stage2_by_patient.csv", "lift_pp, max_entry_margin, C"))


def table_numbers(by_patient: pd.DataFrame, summary: pd.DataFrame, numbers: list) -> pd.DataFrame:
    """K40 tuned-L1 per-patient lifts (refit / inductive / normal-only) and the summary triplets."""
    rows = []
    for method in ("pca", "fa"):
        for arm in ("transductive_refit", "inductive", "normal_only"):
            s = _sel(by_patient, arm, method, 40, "l1_tuned")
            for p in PATIENTS:
                rows.append((method, arm, p, s.loc[p, "lift_pp"], s.loc[p, "prevalence"], s.loc[p, "C"]))
                numbers.append((f"{s.loc[p, 'lift_pp']:.2f}", f"table K40 tuned L1 {method.upper()} {arm} {p} lift (pp)",
                                "d17_stage2_by_patient.csv", "lift_pp, policy l1_tuned"))
    tab = pd.DataFrame(rows, columns=["method", "arm", "patient", "lift_pp", "prevalence", "C"])
    for method in ("pca", "fa"):
        for arm in ("transductive_refit", "inductive", "normal_only", "balanced_weighted", "transductive_stored"):
            for policy in ("l1_tuned", "l2_tuned", "l1_fixed"):
                r = summary[(summary.arm == arm) & (summary.stage1_method == method) & (summary.effective_k == 40)
                            & (summary.policy == policy)]
                if r.empty:
                    continue
                r = r.iloc[0]
                numbers.append((f"{r.pooled_ap:.4f} / {r.median_lift_pp:.2f} / {r.p20_lift_pp:.2f}",
                                f"summary {method.upper()}40 {arm} {policy} pooled AP / median lift / P20 lift (pp); "
                                f"lift>0 {int(r.n_lift_gt_0)}/7; all-zero {r.n_all_zero_folds:g}; median delta vs refit {r.median_delta_vs_refit_pp:.3f}",
                                "d17_stage2_summary.csv", "read directly"))
    return tab


def write_ledger(numbers: list) -> None:
    import os

    scratch = os.environ.get("REPORT_NUMBERS_DIR")
    if not scratch:
        return
    out = Path(scratch) / "06_figs_autogen.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("number\tclaim\tsource_file\thow_computed\n")
        for row in numbers:
            fh.write("\t".join(str(x) for x in row) + "\n")


def main() -> None:
    rs.apply()
    by_patient, summary = load()
    numbers: list = []
    fig06a(by_patient, numbers)
    fig06b(by_patient, numbers)
    fixed_c_numbers(by_patient, numbers)
    tab = table_numbers(by_patient, summary, numbers)
    pd.set_option("display.width", 200)
    print(tab.pivot_table(index=["patient"], columns=["method", "arm"], values="lift_pp").round(2))
    for row in numbers:
        print("\t".join(str(x) for x in row))
    write_ledger(numbers)


if __name__ == "__main__":
    main()
