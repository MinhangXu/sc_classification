"""Figures for section 03: should we aggregate patients on one Stage 1 basis?

Evidence is D14 (patient_factor_overlap_v1). The section opens with three method figures and
then shows the screen-wide results:

fig03w1_one_factor        Worked example on one real factor (IFN-alpha PCA K40, PC8): each
                          patient's score histogram on the shared 32-bin grid, in normal cells and
                          in malignant cells separately; one pair's shared area shaded (the median
                          pair); one patient against the pooled rest; the 21 malignant pairs.
fig03w2_factors_to_call   The same representation, all 40 factors: per-factor median pairwise
                          overlap (dot area = variance share) -> variance-weighted overlap; each
                          patient's variance-weighted patient-vs-rest overlap; the four gates.
fig03w3_call_types        Schematic (synthetic shapes): the three calls and their exact gates.
fig03a_decision_map       One point per representation (47): normal vs malignant weighted overlap.
fig03b_patient_vs_rest    Each patient's patient-vs-rest overlap over the 41 biological panels.
fig03c_factor_spectrum    Three single factors from shared to P03-private (malignant cells).

Panel-level and patient-level numbers are read from the D14 tables. The single-factor views
(w1, w2 per-factor dots, 03c) are recomputed from the stored transductive Stage 1 scores with the
D14 functions themselves (patient_factor_overlap.py: same strata, bins, overlap and weights),
and every recomputed summary is checked against the D14 tables before anything is drawn.
The gates are read from overlap_decision_rule.json and checked against the D14 module constants.

Run from the repo root or this directory:
    python scripts/mrd_stage0_2/research/reports/2026-10-08_questions_report/figs_03_aggregate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import report_style as rs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

SLUG = "patient_factor_overlap_v1"

# D14 decision gates (overlap_decision_rule.json); checked against the JSON and the D14 module in load_tables().
NORMAL_WEIGHTED_GATE = 0.75
NORMAL_MIN_PATIENT_GATE = 0.60
NORMAL_SUBTYPE_GATE = 0.40
MALIGNANT_WEIGHTED_GATE = 0.45
MALIGNANT_GAP_GATE = 0.15  # aggregate needs gap < 0.15; subtype needs gap >= 0.15
MALIGNANT_SUBTYPE_GATE = 0.35

GATE_INK = rs.INK_2

# Worked example: the representation the user knows best (IFN-alpha PCA K40) and its Stage 2 factor PC8.
WORKED = {
    "panel_id": "single_geneset__hallmark_interferon_alpha_response",
    "method": "pca",
    "k": 40,
    "factor": 8,
    "name": "IFN-$\\alpha$ PCA K40",
    "vs_rest_patient": "P03",  # malignant patient shown against the pooled rest (panel d)
}

# Three single factors that span the range from shared to patient-private (malignant cells).
SPECTRUM = (
    {
        "panel_id": "single_geneset__hallmark_interferon_alpha_response",
        "method": "pca",
        "k": 40,
        "factor": 8,
        "title": "IFN-$\\alpha$ PCA K40, PC8",
        "verdict": "shared",
        "role": "pca",
    },
    {
        "panel_id": "single_geneset__hallmark_p53_pathway",
        "method": "factosig",
        "k": 40,
        "factor": 9,
        "title": "p53 pathway FactoSig K40, f9",
        "verdict": "P03 shifted",
        "role": "factosig",
    },
    {
        "panel_id": "hvg_top_requested_10000__available_10000",
        "method": "factosig",
        "k": 40,
        "factor": 21,
        "title": "HVG 10k bag FactoSig K40, f21",
        "verdict": "P03 disjoint",
        "role": "direct_gene",  # muted gray: a broad-bag control, not a biological program
    },
)

STRATUM_TITLE = {"normal": "Normal cells", "cancer": "Malignant cells"}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def _pfo():
    """The D14 module (stage2_supervised/patient_factor_overlap.py)."""
    from sc_classification.utils.paths import add_study_module_paths

    add_study_module_paths()
    import patient_factor_overlap as pfo

    return pfo


def load_tables() -> dict:
    d = rs.diag_dir(SLUG)
    rule = json.loads((d / "overlap_decision_rule.json").read_text())
    agg, sub = rule["aggregate_shared"], rule["separate_subtype_candidate"]
    assert agg["normal_weighted_overlap_at_least"] == NORMAL_WEIGHTED_GATE
    assert agg["normal_min_patient_vs_rest_at_least"] == NORMAL_MIN_PATIENT_GATE
    assert agg["cancer_weighted_overlap_at_least"] == MALIGNANT_WEIGHTED_GATE
    assert agg["cancer_gap_below"] == MALIGNANT_GAP_GATE
    assert sub["normal_min_patient_vs_rest_below"] == NORMAL_SUBTYPE_GATE
    assert sub["or_cancer_weighted_overlap_below"] == MALIGNANT_SUBTYPE_GATE
    assert sub["and_cancer_gap_at_least"] == MALIGNANT_GAP_GATE
    assert rule["bins"] == 32 and rule["normal_min_cells"] == 200 and rule["cancer_min_cells"] == 20
    pfo = _pfo()
    assert (pfo.N_BINS, pfo.NORMAL_MIN_CELLS, pfo.CANCER_MIN_CELLS) == (32, 200, 20)
    assert (pfo.NORMAL_SHARED_OVERLAP, pfo.NORMAL_SHARED_MIN_PATIENT, pfo.NORMAL_SUBTYPE_MIN_PATIENT) == (
        NORMAL_WEIGHTED_GATE, NORMAL_MIN_PATIENT_GATE, NORMAL_SUBTYPE_GATE)
    assert (pfo.CANCER_SHARED_OVERLAP, pfo.CANCER_SHARED_MAX_GAP, pfo.CANCER_SUBTYPE_OVERLAP, pfo.CANCER_SUBTYPE_GAP) == (
        MALIGNANT_WEIGHTED_GATE, MALIGNANT_GAP_GATE, MALIGNANT_SUBTYPE_GATE, MALIGNANT_GAP_GATE)
    calls = pd.read_csv(d / "patient_factor_overlap_calls.csv")
    by_stratum = pd.read_csv(d / "patient_factor_overlap_by_stratum.csv")
    by_patient = pd.read_csv(d / "patient_factor_overlap_by_patient.csv")
    assert len(calls) == 47
    return {"rule": rule, "calls": calls, "by_stratum": by_stratum, "by_patient": by_patient}


def load_cohort() -> tuple[np.ndarray, np.ndarray]:
    """Patient and CN.label per cell, in the row order of every Stage 1 scores.npy."""
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)  # cupy (pulled in by anndata) needs it
    import anndata as ad

    from sc_classification.utils.paths import cohort_cache

    adata = ad.read_h5ad(cohort_cache(must_exist=True), backed="r")
    assert adata.n_obs == 60_118
    patient = adata.obs["patient"].astype(str).to_numpy()
    label = adata.obs["CN.label"].astype(str).to_numpy()
    return patient, label


def _scores(panel_id: str, method: str, k: int) -> np.ndarray:
    path = rs.exp_dir() / "stage1_dr" / f"panel_id={panel_id}" / f"method={method}" / f"k={k}" / "seed=42" / "scores.npy"
    return np.load(path)


def worked_example(tables: dict, patient: np.ndarray, label: np.ndarray) -> dict:
    """Every quantity of the D14 test for one representation, plus one factor in detail.

    Calls pfo.evaluate_stratum (the D14 code path, no permutations) for both strata, then rebuilds
    the single-factor histograms on exactly the cells and bin grid that evaluate_stratum used.
    The summaries are checked against the stored D14 tables.
    """
    pfo = _pfo()
    spec = WORKED
    scores = _scores(spec["panel_id"], spec["method"], spec["k"])
    assert scores.shape[0] == patient.size
    rep_id = f"{spec['panel_id']}|dr|{spec['method']}|{spec['k']}|42"
    call = tables["calls"].set_index("representation_id").loc[rep_id]
    strat = tables["by_stratum"].set_index(["representation_id", "stratum"])
    bp = tables["by_patient"]
    j = spec["factor"] - 1
    out: dict = {"call": call, "strata": {}}
    for stratum, min_cells in (("normal", pfo.NORMAL_MIN_CELLS), ("cancer", pfo.CANCER_MIN_CELLS)):
        mask = label == stratum
        res = pfo.evaluate_stratum(scores, patient, mask, stratum=stratum, min_cells=min_cells, n_perm=0,
                                   rng=np.random.default_rng(0))
        idx = np.flatnonzero(mask)
        keep = np.isin(patient[idx], res.patients)
        x = np.ascontiguousarray(scores[idx[keep]])
        labs = patient[idx][keep]
        edges = pfo._edges_for_matrix(x)
        groups = [np.flatnonzero(labs == name) for name in res.patients]
        dens = pfo._densities(x[:, j], groups, edges[j])
        pairs = np.minimum(dens[:, None, :], dens[None, :, :]).sum(axis=-1)
        _, med, vs_rest_f = pfo._overlap_from_labels(x[:, [j]], labs, res.patients, [edges[j]], np.array([1.0]))
        counts = np.array([g.size for g in groups], dtype=float)
        row = strat.loc[(rep_id, stratum)]
        # Checks against the stored D14 tables.
        assert abs(res.weighted_overlap - row.weighted_overlap) < 1e-5, (stratum, res.weighted_overlap)
        assert res.n_cells == row.n_cells and res.n_patients == row.n_patients
        assert res.min_patient_id == row.min_patient_id
        assert abs(res.min_patient_vs_rest_overlap - row.min_patient_vs_rest_overlap) < 1e-5
        stored = bp[bp.representation_id.eq(rep_id) & bp.stratum.eq(stratum)].set_index("patient")
        for name, value in res.patient_vs_rest.items():
            assert abs(value - stored.loc[name, "vs_rest_weighted_overlap"]) < 1e-5, (stratum, name)
        assert abs(med[0] - res.factor_median_overlap[j]) < 1e-12
        upper = np.triu_indices(len(res.patients), 1)
        out["strata"][stratum] = {
            "res": res,
            "names": list(res.patients),
            "counts": counts,
            "edges": edges[j],
            "dens": dens,
            "pairs": pairs,
            "pair_values": pairs[upper],
            "upper": upper,
            "factor_median": float(med[0]),
            "factor_vs_rest": dict(zip(res.patients, vs_rest_f.tolist())),
            "null_mean": float(row.null_mean_weighted_overlap),
            "gap": float(row.overlap_gap),
        }
    # The malignant pair whose overlap IS the factor's median (21 pairs: an odd count, so one pair).
    c = out["strata"]["cancer"]
    k_med = int(np.argmin(np.abs(c["pair_values"] - c["factor_median"])))
    assert abs(c["pair_values"][k_med] - c["factor_median"]) < 1e-12
    i, k = c["upper"][0][k_med], c["upper"][1][k_med]
    c["median_pair"] = (c["names"][i], c["names"][k])
    # The call this representation received, recomputed from the four gate quantities.
    n, m = out["strata"]["normal"], out["strata"]["cancer"]
    gates = {
        "normal_weighted": n["res"].weighted_overlap,
        "normal_min_patient": n["res"].min_patient_vs_rest_overlap,
        "normal_min_patient_id": n["res"].min_patient_id,
        "malignant_weighted": m["res"].weighted_overlap,
        "malignant_gap": m["gap"],
    }
    aggregate = (
        gates["normal_weighted"] >= NORMAL_WEIGHTED_GATE
        and gates["normal_min_patient"] >= NORMAL_MIN_PATIENT_GATE
        and gates["malignant_weighted"] >= MALIGNANT_WEIGHTED_GATE
        and gates["malignant_gap"] < MALIGNANT_GAP_GATE
    )
    assert aggregate and call.aggregation_call == "aggregate_shared"
    out["gates"] = gates
    return out


def single_factor_profiles(patient: np.ndarray, label: np.ndarray) -> list[dict]:
    """Per-patient malignant-cell histograms and patient-vs-rest overlap on single factors.

    Uses the D14 binning (32 shared bins over the 1st-99th percentile of the included
    malignant cells, tails folded into the end bins) and D14's own overlap and eta2
    functions, so each number is the quantity the D14 test computes.
    """
    pfo = _pfo()
    names = sorted(rs.SUPPORTED)
    mask = (label == "cancer") & np.isin(patient, names)
    labels = patient[mask]
    out = []
    for spec in SPECTRUM:
        scores = _scores(spec["panel_id"], spec["method"], spec["k"])
        assert scores.shape[0] == patient.size
        column = scores[mask, spec["factor"] - 1][:, None]
        edges = pfo._edges_for_matrix(column)
        _, median_overlap, vs_rest = pfo._overlap_from_labels(column, labels, names, edges, np.array([1.0]))
        groups = [np.flatnonzero(labels == name) for name in names]
        densities = pfo._densities(column[:, 0], groups, edges[0])
        eta2 = pfo._factor_eta2(column[:, 0], groups)
        # D14 variance weight of this factor: its share of score variance over the stratum's cells.
        variance = scores[mask].var(axis=0)
        var_share = float(variance[spec["factor"] - 1] / variance.sum())
        out.append(
            {
                **spec,
                "patients": names,
                "n": [int(g.size) for g in groups],
                "edges": edges[0],
                "densities": densities,
                "vs_rest": dict(zip(names, vs_rest.tolist())),
                "median_pairwise": float(median_overlap[0]),
                "eta2": float(eta2),
                "var_share": var_share,
            }
        )
    # Regression checks against D14 (findings.md: HVG10k f21 P03 overlap 0.007, eta2 0.913).
    hvg = out[2]
    assert round(hvg["vs_rest"]["P03"], 3) == 0.007, hvg["vs_rest"]["P03"]
    assert abs(hvg["eta2"] - 0.913262) < 1e-5, hvg["eta2"]
    assert abs(hvg["median_pairwise"] - 0.413270) < 1e-5, hvg["median_pairwise"]
    assert abs(hvg["var_share"] - 0.373127) < 1e-5, hvg["var_share"]
    return out


# --------------------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------------------


def _steps(edges: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x, y of a step outline over bins (one flat segment per bin)."""
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def _bold_high_prevalence(labels) -> None:
    for tick in labels:
        if tick.get_text().split(" ")[0].split("\n")[0] in rs.HIGH_PREVALENCE:
            tick.set_fontweight("bold")
            tick.set_color(rs.INK)


def _ridgeline(ax, st: dict, tags: dict[str, str], title: str) -> None:
    """One row per patient: the patient's histogram (proportions) on the stratum's 32-bin grid."""
    names, dens, edges = st["names"], st["dens"], st["edges"]
    n_rows = len(names)
    # One height scale per panel, set by the patients with >= 100 cells so that a 22-cell
    # patient's spikes (one cell = 4.5% of its histogram) do not flatten everyone else;
    # those spikes are clipped at 1.4 row heights.
    big = st["counts"] >= 100
    scale = 0.92 / float(dens[big].max() if big.any() else dens.max())
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for i, pid in enumerate(names):
        base = n_rows - 1 - i
        xs, ys = _steps(edges, np.minimum(dens[i] * scale, 1.4))
        ax.fill_between(xs, base, base + ys, color=rs.BLUE, alpha=0.22, lw=0, zorder=2)
        ax.plot(xs, base + ys, color=rs.BLUE, lw=0.9, zorder=3)
        ax.plot([edges[0], edges[-1]], [base, base], color=rs.AXIS, lw=0.6, zorder=1)
        value = st["factor_vs_rest"][pid]
        ax.text(1.03, base + 0.28, f"{value:.3f}", transform=trans, fontsize=6.8, color=rs.INK,
                va="center", ha="left", clip_on=False)
        if pid in tags:
            ax.text(1.27, base + 0.28, tags[pid], transform=trans, fontsize=6.6, color=rs.BLUE,
                    fontweight="bold", va="center", ha="left", clip_on=False)
    ax.text(1.03, n_rows - 0.05, "vs rest", transform=trans, fontsize=6.8, color=rs.INK_2, ha="left",
            va="bottom", clip_on=False)
    ax.set_yticks([n_rows - 1 - i + 0.28 for i in range(n_rows)])
    ax.set_yticklabels([f"{p} ({int(c):,})" for p, c in zip(names, st["counts"])], fontsize=6.8)
    _bold_high_prevalence(ax.get_yticklabels())
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(-0.25, n_rows + 0.1)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # The 32 bins, as ticks under the baseline.
    ax.set_xticks(edges, minor=True)
    ax.tick_params(axis="x", which="minor", length=2.2, color=rs.AXIS)
    ax.set_xlabel("PC8 score (raw sign); minor ticks = the 32 bins", fontsize=7.0)
    ax.set_title(title, fontsize=7.6, fontweight="normal", color=rs.INK, loc="left")


def _overlay(ax, edges, a, b, label_a, label_b, color_b, ovl, title, note) -> None:
    """Two histograms (proportions) on one grid with their shared area shaded."""
    xs, ya = _steps(edges, a)
    _, yb = _steps(edges, b)
    _, ys = _steps(edges, np.minimum(a, b))
    ax.fill_between(xs, 0, ys, color=rs.SEQ_BLUE[1], lw=0, zorder=1)
    ax.plot(xs, ya, color=rs.INK, lw=1.0, zorder=3)
    ax.plot(xs, yb, color=color_b, lw=1.0, zorder=2)
    handles = [
        Line2D([], [], color=rs.INK, lw=1.0, label=label_a),
        Line2D([], [], color=color_b, lw=1.0, label=label_b),
        Patch(facecolor=rs.SEQ_BLUE[1], edgecolor="none", label=f"shared area = {ovl:.3f}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6.3, handlelength=1.2, borderaxespad=0.1,
              labelspacing=0.25, handletextpad=0.4)
    ax.set_xlim(edges[0], edges[-1])
    top = max(a.max(), b.max())
    ax.set_ylim(0, top * 1.75)
    ax.set_xlabel("PC8 score (raw sign)", fontsize=7.0)
    ax.set_ylabel("share of cells in bin", fontsize=7.0)
    ax.set_title(title, fontsize=7.6, fontweight="normal", color=rs.INK, loc="left")
    ax.text(0.98, 0.5, note, transform=ax.transAxes, fontsize=6.3, color=rs.INK_2, ha="right", va="center")


# --------------------------------------------------------------------------------------
# Method figures (worked example)
# --------------------------------------------------------------------------------------


def fig03w1_one_factor(w: dict) -> Path:
    n, c = w["strata"]["normal"], w["strata"]["cancer"]
    pa, pb = c["median_pair"]
    pr = WORKED["vs_rest_patient"]
    fig = plt.figure(figsize=(rs.FULL_W, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.75], hspace=0.5)
    top = outer[0].subgridspec(1, 2, wspace=0.78)
    bot = outer[1].subgridspec(1, 3, wspace=0.5, width_ratios=[1.0, 1.0, 0.82])

    ax_n = fig.add_subplot(top[0, 0])
    _ridgeline(ax_n, n, {}, f"(a) Normal cells: {len(n['names'])} patients with $\\geq$ 200 normal\n"
               f"cells ({int(n['counts'].sum()):,}); median of {len(n['pair_values'])} pairs {n['factor_median']:.3f}")
    ax_c = fig.add_subplot(top[0, 1])
    _ridgeline(ax_c, c, {pa: "(c)", pb: "(c)", pr: "(d)"},
               f"(b) Malignant cells: {len(c['names'])} patients with $\\geq$ 20 malignant\n"
               f"cells ({int(c['counts'].sum()):,}); median of {len(c['pair_values'])} pairs {c['factor_median']:.3f}")

    names = c["names"]
    ia, ib, ir = names.index(pa), names.index(pb), names.index(pr)
    ax = fig.add_subplot(bot[0, 0])
    _overlay(ax, c["edges"], c["dens"][ia], c["dens"][ib],
             f"{pa} ({int(c['counts'][ia])} cells)", f"{pb} ({int(c['counts'][ib])} cells)", rs.BLUE,
             c["pairs"][ia, ib], f"(c) One pair: {pa} vs {pb}\n(the median of the 21 pairs)", "")
    rest_w = c["counts"].copy()
    rest_w[ir] = 0.0
    rest = np.average(c["dens"], axis=0, weights=rest_w)
    ax = fig.add_subplot(bot[0, 1])
    _overlay(ax, c["edges"], c["dens"][ir], rest, f"{pr} ({int(c['counts'][ir])} cells)",
             f"rest ({int(rest_w.sum())} cells)", rs.MUTED, c["factor_vs_rest"][pr],
             f"(d) One patient vs the rest:\n{pr} vs the other {len(names) - 1} pooled", "")

    # (e) all 21 malignant pairs on this factor.
    ax = fig.add_subplot(bot[0, 2])
    cmap = LinearSegmentedColormap.from_list("seq_blue", rs.SEQ_BLUE)
    m = len(names)
    # Lower triangle without the empty first row and last column: rows names[1:], columns names[:-1].
    grid = np.full((m - 1, m - 1), np.nan)
    for i in range(1, m):
        for k in range(i):
            grid[i - 1, k] = c["pairs"][i, k]
    ax.imshow(grid, cmap=cmap, vmin=0.3, vmax=1.0, aspect="equal")
    for i in range(m - 1):
        for k in range(i + 1):
            v = grid[i, k]
            ax.text(k, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=5.8,
                    color="white" if v > 0.72 else rs.INK)
    r, q = max(ia, ib), min(ia, ib)
    ax.add_patch(Rectangle((q - 0.5, r - 1 - 0.5), 1, 1, fill=False, edgecolor=rs.INK, lw=1.3))
    m -= 1
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(names[:-1], fontsize=6.0, rotation=90)
    ax.set_yticklabels(names[1:], fontsize=6.0)
    _bold_high_prevalence(ax.get_xticklabels())
    _bold_high_prevalence(ax.get_yticklabels())
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_title(f"(e) All {len(c['pair_values'])} malignant pairs;\nmedian {c['factor_median']:.3f} (boxed)", fontsize=7.6,
                 fontweight="normal", color=rs.INK, loc="left")

    fig.suptitle(f"How the test reads one factor: {WORKED['name']}, PC8, normal and malignant cells kept apart",
                 x=0.005, ha="left", y=0.995, fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig03w1_one_factor")


def _beeswarm_y(x: np.ndarray, spread: float = 0.36) -> np.ndarray:
    """Deterministic vertical offsets so dots at similar x do not sit on top of each other."""
    order = np.argsort(x)
    y = np.zeros_like(x, dtype=float)
    pattern = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25, 0.75, -0.75])
    for rank, idx in enumerate(order):
        y[idx] = pattern[rank % pattern.size] * spread
    return y


def fig03w2_factors_to_call(w: dict) -> Path:
    fig, axes = plt.subplots(
        2, 2, figsize=(rs.FULL_W, 3.55), sharex=True,
        gridspec_kw={"width_ratios": [1.35, 1.0], "hspace": 0.42, "wspace": 0.08},
    )
    x0, x1 = 0.30, 1.0
    j = WORKED["factor"] - 1
    g = w["gates"]
    for row, stratum in enumerate(("normal", "cancer")):
        st = w["strata"][stratum]
        res = st["res"]
        med, share = res.factor_median_overlap, res.factor_var_share
        # Left: one dot per factor (area = variance share); the solid line is their weighted mean.
        ax = axes[row, 0]
        y = _beeswarm_y(med)
        ax.scatter(med, y, s=1500 * share, color=rs.BLUE, alpha=0.55, edgecolors=rs.SURFACE, linewidths=0.6, zorder=3)
        ax.scatter([med[j]], [y[j]], s=1500 * share[j], facecolors="none", edgecolors=rs.INK, linewidths=1.2, zorder=4)
        # PC8 is labelled at its ring with a leader, so the label cannot be read as the null value
        # printed next to the dotted null line (a v2 reading-review fix).
        ax.annotate(f"PC8 {med[j]:.3f}", xy=(med[j], y[j]), xytext=(med[j] - 0.04, 1.02), fontsize=6.4,
                    color=rs.INK, ha="right", va="bottom",
                    arrowprops={"arrowstyle": "-", "color": rs.INK, "lw": 0.6, "shrinkB": 5})
        top_f = int(np.argmax(share))
        ax.annotate(f"PC{top_f + 1} (top weight {share[top_f]:.3f})", xy=(med[top_f], y[top_f]),
                    xytext=(x0 + 0.01, -1.32), fontsize=6.3, color=rs.INK_2, ha="left", va="center",
                    arrowprops={"arrowstyle": "-", "color": rs.MUTED, "lw": 0.6, "shrinkB": 4})
        wo = res.weighted_overlap
        ax.plot([wo, wo], [-0.85, 0.85], color=rs.INK, lw=1.6, zorder=5)
        ax.text(wo, -0.9, f"{wo:.3f}", fontsize=6.8, color=rs.INK, ha="center", va="top", fontweight="bold")
        gate = NORMAL_WEIGHTED_GATE if stratum == "normal" else MALIGNANT_WEIGHTED_GATE
        ax.axvline(gate, color=GATE_INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.text(gate - 0.007, 0.98, f"{gate:.2f}", fontsize=6.4, color=GATE_INK, ha="right", va="bottom")
        nm = st["null_mean"]
        ax.plot([nm, nm], [-0.85, 0.85], color=rs.MUTED, lw=1.0, ls=(0, (1, 1.3)), zorder=2)
        if nm > 0.9:  # near the panel edge: two lines inside the axes, right of the null line
            ax.text(nm + 0.007, 0.98, f"null\n{nm:.3f}", fontsize=6.4, color=rs.MUTED, ha="left", va="top",
                    linespacing=1.0)
        else:
            ax.text(nm + 0.007, 0.98, f"null {nm:.3f}", fontsize=6.4, color=rs.MUTED, ha="left", va="bottom")
        gap_note = f"gap = {nm:.3f} $-$ {wo:.3f} = {st['gap']:.3f}"
        if stratum == "cancer":
            gap_note += f"\n(aggregate needs < {MALIGNANT_GAP_GATE:.2f})"
            ax.axvspan(x0, MALIGNANT_SUBTYPE_GATE, color=rs.DIV_POS, alpha=0.06, lw=0, zorder=0)
            ax.text(x0 + 0.008, 0.55, "< 0.35:\nsubtype\nzone", fontsize=6.0, color=rs.DIV_POS, ha="left", va="center")
        if stratum == "normal":  # the left of this row is empty
            ax.text(x0 + 0.01, 0.4, gap_note, fontsize=6.3, color=rs.INK_2, ha="left", va="center")
        else:  # the bottom right of this row is empty
            ax.text(0.995, -1.3, gap_note, transform=blended_transform_factory(ax.transAxes, ax.transData),
                    fontsize=6.3, color=rs.INK_2, ha="right", va="center")
        ax.set_ylim(-1.55, 1.3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_ylabel(f"{STRATUM_TITLE[stratum]}\n({len(st['names'])} patients)", fontsize=7.4, color=rs.INK)
        # Right: one dot per patient (its patient-vs-rest overlap, same weights).
        ax = axes[row, 1]
        vals = np.array([res.patient_vs_rest[p] for p in st["names"]])
        yp = _beeswarm_y(vals, spread=0.32)
        ax.scatter(vals, yp, s=16, color=rs.INK_2, edgecolors=rs.SURFACE, linewidths=0.6, zorder=3)
        lo = int(np.argmin(vals))
        left = vals[lo] - x0 > 0.35  # room for the label on the left of the dot
        ax.annotate(f"lowest: {st['names'][lo]} {vals[lo]:.3f}", xy=(vals[lo], yp[lo]),
                    xytext=(vals[lo] + (-0.05 if left else 0.05), -1.15), fontsize=6.4, color=rs.INK,
                    ha="right" if left else "left", va="center",
                    arrowprops={"arrowstyle": "-", "color": rs.INK_2, "lw": 0.6, "shrinkB": 3})
        if stratum == "normal":
            ax.axvline(NORMAL_MIN_PATIENT_GATE, color=GATE_INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
            ax.text(NORMAL_MIN_PATIENT_GATE + 0.007, 0.98, "0.60: every patient", fontsize=6.4, color=GATE_INK,
                    ha="left", va="bottom")
            ax.axvline(NORMAL_SUBTYPE_GATE, color=rs.DIV_POS, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
            ax.text(NORMAL_SUBTYPE_GATE - 0.007, 0.98, "0.40", fontsize=6.4, color=rs.DIV_POS, ha="right", va="bottom")
            ax.text(x0 + 0.008, 0.2, "any\npatient\nbelow:\nsubtype", fontsize=6.0, color=rs.DIV_POS, ha="left",
                    va="center")
        else:
            ax.text(x0 + 0.01, 1.0, "no per-patient gate in malignant cells\n(22 to 758 cells per patient)",
                    fontsize=6.3, color=rs.MUTED, ha="left", va="center")
        ax.set_ylim(-1.55, 1.3)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_title("(a) One dot per factor: median pairwise overlap\n(dot area = share of score variance)",
                         fontsize=7.6, fontweight="normal", color=rs.INK, loc="left", pad=6)
    axes[0, 1].set_title("(b) One dot per patient: overlap with the pooled\nrest, averaged over factors (same weights)",
                         fontsize=7.6, fontweight="normal", color=rs.INK, loc="left", pad=6)
    for ax in axes[1]:
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Overlap (0 = disjoint, 1 = identical)", fontsize=7.2)
    handles = [
        Line2D([], [], color=rs.INK, lw=1.6, label="weighted overlap = $\\Sigma_j\\, w_j m_j$"),
        Line2D([], [], color=rs.MUTED, lw=1.0, ls=(0, (1, 1.3)), label="label-shuffle null (mean of 25)"),
        Line2D([], [], color=GATE_INK, lw=0.9, ls=(0, (4, 2)), label="aggregate gate"),
        Line2D([], [], color=rs.DIV_POS, lw=0.9, ls=(0, (1, 1.5)), label="subtype gate"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.955), ncol=4, fontsize=6.6, frameon=False,
               handlelength=2.0, columnspacing=1.2)
    tick = "\u2713"  # check mark
    check = (
        f"Call for {WORKED['name']}: all four aggregate gates pass, so the call is aggregate.\n"
        f"Normal weighted {g['normal_weighted']:.3f} $\\geq$ 0.75 {tick};   "
        f"lowest normal patient ({g['normal_min_patient_id']}) {g['normal_min_patient']:.3f} $\\geq$ 0.60 {tick};   "
        f"malignant weighted {g['malignant_weighted']:.3f} $\\geq$ 0.45 {tick};\n"
        f"malignant gap {g['malignant_gap']:.3f} < 0.15 {tick}.   No subtype gate fires "
        f"(lowest normal patient $\\geq$ 0.40; malignant weighted $\\geq$ 0.35)."
    )
    fig.text(0.005, -0.04, check, fontsize=6.8, color=rs.INK, ha="left", va="top", linespacing=1.4)
    fig.suptitle(f"From 40 factors to one call: {WORKED['name']}", x=0.005, ha="left", y=1.065,
                 fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig03w2_factors_to_call")


def fig03w3_call_types(calls: pd.DataFrame) -> Path:
    counts = calls.aggregation_call.value_counts()
    n_total = len(calls)
    shifts = calls.loc[calls.aggregation_call.eq("shared_with_location_shift"), "short_panel_label"].tolist()
    assert sorted(shifts) == ["HVGs (10000)", "allograft rejection"], shifts
    x = np.linspace(-3.6, 7.6, 1200)
    dx = x[1] - x[0]

    def bump(mu: float) -> np.ndarray:
        return np.exp(-0.5 * (x - mu) ** 2) / np.sqrt(2 * np.pi)

    others = (0.0, 0.3, -0.25)
    rest = np.mean([bump(mu) for mu in others], axis=0)
    specs = (
        ("Aggregate", 0.12,
         "All four hold:\n"
         "  normal weighted overlap $\\geq$ 0.75\n"
         "  every patient's normal vs rest $\\geq$ 0.60\n"
         "  malignant weighted overlap $\\geq$ 0.45\n"
         "  malignant gap to the null < 0.15",
         f"In this screen: {counts.get('aggregate_shared', 0)} of {n_total}"),
        ("Location shift", 1.2,
         "Neither of the other two, e.g.\n"
         "  one patient's normal vs rest is\n"
         "  between 0.40 and 0.60, or the\n"
         "  malignant gap is $\\geq$ 0.15 while\n"
         "  malignant overlap stays $\\geq$ 0.35",
         f"In this screen: {counts.get('shared_with_location_shift', 0)} of {n_total} (allograft\nrejection; HVG 10k bag)"),
        ("Separate-subtype candidate", 4.9,
         "Either one (checked first):\n"
         "  any patient's normal vs rest < 0.40\n"
         "  or malignant weighted overlap < 0.35\n"
         "  together with a gap $\\geq$ 0.15",
         f"In this screen: {counts.get('separate_subtype_candidate', 0)} of {n_total}"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL_W, 2.1), sharey=True, gridspec_kw={"wspace": 0.2})
    peak = float(bump(0.0).max())
    for ax, (name, odd, rule, here) in zip(axes, specs):
        for mu in others:
            ax.fill_between(x, 0, bump(mu), color=rs.BLUE, alpha=0.10, lw=0)
            ax.plot(x, bump(mu), color=rs.BLUE, lw=0.9, alpha=0.8)
        one = bump(odd)
        ovl = float(np.minimum(one, rest).sum() * dx)  # shared area of this synthetic patient and the pooled others
        ax.plot(x, one, color=rs.INK, lw=1.4)
        ax.text(odd + 0.55, peak * 0.93, "one\npatient", fontsize=6.2, color=rs.INK, ha="left", va="center")
        ax.text(-1.25, peak * 0.93, "the\nothers", fontsize=6.2, color=rs.BLUE, ha="right", va="center")
        ax.text(0.0, 1.0, f"its vs-rest overlap $\\approx$ {ovl:.2f}", transform=ax.transAxes, fontsize=6.4,
                color=rs.INK_2, ha="left", va="bottom")
        ax.set_ylim(0, peak * 1.08)
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(name, fontsize=8.2, fontweight="bold", color=rs.INK, loc="left", pad=12)
        ax.text(0.0, -0.07, rule, transform=ax.transAxes, fontsize=6.2, color=rs.INK, ha="left", va="top",
                linespacing=1.25)
        ax.text(0.0, -0.57, here, transform=ax.transAxes, fontsize=6.2, color=rs.INK_2, ha="left", va="top",
                fontstyle="italic")
    axes[0].set_ylabel("score density\n(schematic)", fontsize=7.0)
    fig.suptitle("The three calls: schematic shapes (not data) and the exact gates", x=0.005, ha="left", y=1.13,
                 fontsize=9.5, fontweight="bold")
    return rs.save(fig, "fig03w3_call_types")


# --------------------------------------------------------------------------------------
# Screen-wide figures
# --------------------------------------------------------------------------------------


def _method_style(method: str) -> dict:
    marker = "D" if method == "factosig_promax" else "o"
    size = 4.6 if marker == "D" else rs.MARKER
    return {
        "marker": marker,
        "markersize": size,
        "markerfacecolor": rs.METHOD[method],
        "markeredgecolor": rs.SURFACE,
        "markeredgewidth": rs.RING,
        "linestyle": "none",
    }


def fig03a_decision_map(calls: pd.DataFrame, by_stratum: pd.DataFrame) -> Path:
    null = by_stratum.pivot_table(
        index="representation_id", columns="stratum", values="null_mean_weighted_overlap"
    )
    df = calls.set_index("representation_id").join(null.rename(columns=lambda c: f"null_{c}"))

    fig, ax = plt.subplots(figsize=(rs.FULL_W, 2.6))
    x0, x1, y0, y1 = 0.70, 0.985, 0.30, 0.77

    # Decision regions (weighted-overlap gates only; the min-patient gate is in fig03b).
    ax.add_patch(
        Rectangle(
            (NORMAL_WEIGHTED_GATE, MALIGNANT_WEIGHTED_GATE),
            x1 - NORMAL_WEIGHTED_GATE,
            y1 - MALIGNANT_WEIGHTED_GATE,
            facecolor=rs.SEQ_BLUE[0],
            alpha=0.35,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        Rectangle((x0, y0), x1 - x0, MALIGNANT_SUBTYPE_GATE - y0, facecolor=rs.DIV_POS, alpha=0.07, edgecolor="none", zorder=0)
    )
    # The normal gate stops at the top of the subtype band, so it does not run through the band's label.
    ax.axvline(NORMAL_WEIGHTED_GATE, ymin=(MALIGNANT_SUBTYPE_GATE - y0) / (y1 - y0), color=GATE_INK, lw=0.9,
               ls=(0, (4, 2)), zorder=1)
    ax.axhline(MALIGNANT_WEIGHTED_GATE, color=GATE_INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.axhline(MALIGNANT_SUBTYPE_GATE, color=rs.DIV_POS, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    ax.text(0.7535, 0.762, "aggregate region (weighted-overlap gates)", color=rs.SEQ_BLUE[5], fontsize=7.5, fontweight="bold", va="top")
    ax.text(x0 + 0.004, MALIGNANT_SUBTYPE_GATE - 0.012, "subtype zone: malignant < 0.35 with gap $\\geq$ 0.15",
            color=rs.DIV_POS, fontsize=7, va="top")
    ax.text(NORMAL_WEIGHTED_GATE - 0.003, 0.60, "normal gate 0.75", color=GATE_INK, fontsize=7,
            rotation=90, ha="right", va="center")
    ax.text(x1 - 0.004, MALIGNANT_WEIGHTED_GATE - 0.006, "malignant gate 0.45", color=GATE_INK, fontsize=7,
            va="top", ha="right")

    # Exchangeable null: each representation's mean over 25 within-label shuffles.
    ax.plot(df.null_normal, df.null_cancer, "o", ms=3.2, color=rs.MUTED, mec="none", zorder=2)
    ax.annotate(
        "label-shuffle null:\nexchangeable patients",
        xy=(0.9445, 0.655),
        xytext=(0.983, 0.567),
        fontsize=7,
        color=rs.INK_2,
        ha="right",
        va="top",
        arrowprops={"arrowstyle": "-", "color": rs.MUTED, "lw": 0.6},
    )

    bio = df[df.atlas_row_block.eq("biological")]
    ctrl = df[~df.atlas_row_block.eq("biological")]
    for method in ("pca", "fa", "factosig", "factosig_promax"):
        sub = bio[bio.stage1_method.eq(method)]
        ax.plot(sub.normal_weighted_overlap, sub.cancer_weighted_overlap, zorder=4, **_method_style(method))
    ax.plot(
        ctrl.normal_weighted_overlap,
        ctrl.cancer_weighted_overlap,
        "s",
        ms=5.0,
        mfc=rs.SURFACE,
        mec=rs.MUTED,
        mew=1.2,
        zorder=4,
    )

    def _row(label: str) -> pd.Series:
        return df[df.short_panel_label.eq(label)].iloc[0]

    allo = _row("allograft rejection")
    ax.annotate(
        "allograft rejection (FactoSig-promax K40): location shift,\nP02 normal cells overlap the rest at 0.575 (< 0.60 gate)",
        xy=(allo.normal_weighted_overlap, allo.cancer_weighted_overlap),
        xytext=(0.7545, 0.713),
        fontsize=7,
        color=rs.INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": rs.INK_2, "lw": 0.7, "shrinkB": 4},
    )
    hvg = _row("HVGs (10000)")
    ax.annotate(
        "HVG 10k bag (control): location shift\n(malignant gap 0.182 $\\geq$ 0.15)",
        xy=(hvg.normal_weighted_overlap, hvg.cancer_weighted_overlap),
        xytext=(0.858, 0.478),
        fontsize=7,
        color=rs.INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": rs.INK_2, "lw": 0.7, "shrinkB": 4},
    )
    ifn = _row("interferon alpha response")
    ax.annotate(
        "IFN-$\\alpha$ PCA K40\n(worked example)",
        xy=(ifn.normal_weighted_overlap, ifn.cancer_weighted_overlap),
        xytext=(0.897, 0.603),
        fontsize=7,
        color=rs.INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": rs.INK_2, "lw": 0.7, "shrinkB": 4},
    )

    handles = [Line2D([], [], **_method_style(m), label=rs.METHOD_LABEL[m]) for m in ("pca", "fa", "factosig", "factosig_promax")]
    handles.append(Line2D([], [], marker="s", ms=5.0, mfc=rs.SURFACE, mec=rs.MUTED, mew=1.2, ls="none", label="broad-bag control"))
    handles.append(Line2D([], [], marker="o", ms=3.2, color=rs.MUTED, mec="none", ls="none", label="label-shuffle null"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.185, 0.305), frameon=False, handletextpad=0.3,
              borderaxespad=0.2, ncol=3, columnspacing=1.2, fontsize=7.2)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Normal-cell weighted overlap (0 = disjoint, 1 = identical)")
    ax.set_ylabel("Malignant-cell weighted overlap")
    ax.grid(axis="both")
    ax.set_title("40 of 41 programs pass every aggregate gate; none nears a subtype gate")
    return rs.save(fig, "fig03a_decision_map")


def _fmt_n(n: int) -> str:
    if n >= 10000:  # two digits are enough for the ordering, and keep neighbouring labels apart
        return f"{n / 1000:.0f}k"
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def fig03b_patient_vs_rest(calls: pd.DataFrame, by_patient: pd.DataFrame) -> Path:
    bio_ids = set(calls.loc[calls.atlas_row_block.eq("biological"), "representation_id"])
    bp = by_patient[by_patient.representation_id.isin(bio_ids)].copy()
    rng = np.random.default_rng(3)

    fig, axes = plt.subplots(
        1, 2, figsize=(rs.FULL_W + 0.25, 1.8), sharey=True, gridspec_kw={"width_ratios": [14.2, 6.8], "wspace": 0.05}
    )
    for ax, stratum, title in (
        (axes[0], "normal", "Normal cells: 13 patients with $\\geq$ 200 normal cells"),
        (axes[1], "cancer", "Malignant: 7 supported patients"),
    ):
        sub = bp[bp.stratum.eq(stratum)]
        n_cells = sub.groupby("patient").n_cells.first().sort_values(kind="stable")
        order = list(n_cells.index)
        for i, pid in enumerate(order):
            vals = sub.loc[sub.patient.eq(pid), "vs_rest_weighted_overlap"].to_numpy()
            assert vals.size == 41
            jitter = rng.uniform(-0.2, 0.2, vals.size)
            ax.plot(i + jitter, vals, "o", ms=2.6, color=rs.MUTED, alpha=0.75, mec="none", zorder=2)
            med = float(np.median(vals))
            ax.plot([i - 0.32, i + 0.32], [med, med], color=rs.INK, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{p}\n{_fmt_n(int(n_cells[p]))}" for p in order], fontsize=7.0)
        for tick, pid in zip(ax.get_xticklabels(), order):
            if pid in rs.HIGH_PREVALENCE:
                tick.set_fontweight("bold")
                tick.set_color(rs.INK)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(title, fontsize=8, fontweight="normal", color=rs.INK_2)
        ax.grid(axis="y")

    ax = axes[0]
    ax.axhline(NORMAL_MIN_PATIENT_GATE, color=GATE_INK, lw=0.9, ls=(0, (4, 2)), zorder=1)
    ax.axhline(NORMAL_SUBTYPE_GATE, color=rs.DIV_POS, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
    right = len(axes[0].get_xticks()) - 0.5
    ax.text(right, NORMAL_MIN_PATIENT_GATE - 0.012, "every patient must be $\\geq$ 0.60 to aggregate",
            fontsize=6.8, color=GATE_INK, va="top", ha="right")
    ax.text(right, NORMAL_SUBTYPE_GATE - 0.012, "any patient < 0.40 = subtype candidate", fontsize=6.8,
            color=rs.DIV_POS, va="top", ha="right")
    # The one value under the 0.60 gate: allograft rejection, P02.
    normal = bp[bp.stratum.eq("normal")]
    low = normal.loc[normal.vs_rest_weighted_overlap.idxmin()]
    order_n = list(normal.groupby("patient").n_cells.first().sort_values(kind="stable").index)
    ax.annotate(
        f"allograft, {low.patient}: {low.vs_rest_weighted_overlap:.3f}",
        xy=(order_n.index(low.patient), low.vs_rest_weighted_overlap),
        xytext=(order_n.index(low.patient) + 1.3, 0.50),
        fontsize=6.8,
        color=rs.INK,
        arrowprops={"arrowstyle": "-", "color": rs.INK_2, "lw": 0.7},
    )
    axes[1].annotate(
        "fewest malignant cells,\nlowest overlap",
        xy=(0, 0.60),
        xytext=(-0.3, 0.88),
        fontsize=6.8,
        color=rs.INK_2,
        arrowprops={"arrowstyle": "-", "color": rs.INK_2, "lw": 0.7},
    )
    axes[0].set_ylabel("Patient-vs-rest overlap\n(variance-weighted)")
    axes[0].set_ylim(0.30, 1.0)
    fig.text(0.5, -0.115, "Patient, ordered by cells in the stratum (second line = cells; bold = highest prevalence)",
             ha="center", fontsize=7.5, color=rs.INK_2)
    handles = [
        Line2D([], [], marker="o", ms=2.6, color=rs.MUTED, ls="none", label="one panel"),
        Line2D([], [], color=rs.INK, lw=2.0, label="median of 41"),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=False, fontsize=7.0, handlelength=1.4,
                   borderaxespad=0.2)
    fig.suptitle(
        "No patient sits apart; malignant overlap rises with malignant cell count",
        x=0.005, ha="left", y=1.065, fontsize=9.5, fontweight="bold",
    )
    return rs.save(fig, "fig03b_patient_vs_rest")


def fig03c_factor_spectrum(profiles: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(rs.FULL_W - 0.6, 1.9), sharey=True, gridspec_kw={"wspace": 0.36})
    names = profiles[0]["patients"]
    n_rows = len(names)
    for ax, prof in zip(axes, profiles):
        color = rs.METHOD[prof["role"]]
        dens = prof["densities"]
        scale = 0.85 / float(dens.max())
        edges = prof["edges"]
        for i, pid in enumerate(names):
            base = n_rows - 1 - i
            xs, ys = _steps(edges, dens[i] * scale)
            ax.fill_between(xs, base, base + ys, color=color, alpha=0.28, lw=0, zorder=2)
            ax.plot(xs, base + ys, color=color, lw=1.0, zorder=3)
            ax.plot([edges[0], edges[-1]], [base, base], color=rs.AXIS, lw=0.6, zorder=1)
            value = prof["vs_rest"][pid]
            weight = "bold" if value < 0.1 else "normal"
            ax.text(1.04, base + 0.3, f"{value:.3f}", transform=blended_transform_factory(ax.transAxes, ax.transData),
                    fontsize=7, color=rs.INK, fontweight=weight, va="center", ha="left", clip_on=False)
        ax.text(1.04, n_rows - 0.1, "overlap\nvs rest", transform=blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=7, color=rs.INK_2, ha="left", va="bottom", clip_on=False, linespacing=1.0)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(-0.2, n_rows + 0.05)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Factor score (raw sign)", fontsize=7.2)
        ax.set_title(
            f"{prof['title']}\n{prof['verdict']}, $\\eta^2$ = {prof['eta2']:.3f}",
            fontsize=7.2,
            fontweight="normal",
            color=rs.INK,
            loc="left",
            pad=20,
        )
    axes[0].set_yticks([n_rows - 1 - i + 0.3 for i in range(n_rows)])
    axes[0].set_yticklabels([f"{p} ({n})" for p, n in zip(names, profiles[0]["n"])], fontsize=7.2)
    _bold_high_prevalence(axes[0].get_yticklabels())
    fig.suptitle(
        "Single factors run from shared to P03-private; only a broad gene bag isolates P03",
        x=0.005, ha="left", y=1.25, fontsize=9.5, fontweight="bold",
    )
    return rs.save(fig, "fig03c_factor_spectrum")


def _print_worked(w: dict) -> None:
    """The worked-example numbers quoted in the section text (for the report ledger)."""
    for stratum in ("normal", "cancer"):
        st = w["strata"][stratum]
        res = st["res"]
        j = WORKED["factor"] - 1
        n_pairs = len(st["pair_values"])
        print(
            f"[w] {stratum}: n_patients {res.n_patients}, n_cells {res.n_cells}, pairs {n_pairs}; "
            f"PC8 median pair {st['factor_median']:.4f} (pairs {st['pair_values'].min():.4f}-{st['pair_values'].max():.4f}), "
            f"PC8 var share {res.factor_var_share[j]:.4f}; weighted {res.weighted_overlap:.4f}; "
            f"null {st['null_mean']:.4f}; gap {st['gap']:.4f}; min patient {res.min_patient_id} "
            f"{res.min_patient_vs_rest_overlap:.4f}; top weight PC{int(np.argmax(res.factor_var_share)) + 1} "
            f"{res.factor_var_share.max():.4f}; per-factor median range {res.factor_median_overlap.min():.4f}-"
            f"{res.factor_median_overlap.max():.4f}"
        )
        print("    PC8 vs rest: " + ", ".join(f"{p} {v:.3f}" for p, v in st["factor_vs_rest"].items()))
    c = w["strata"]["cancer"]
    pa, pb = c["median_pair"]
    print(f"[w] median malignant pair {pa}-{pb}; rest of {WORKED['vs_rest_patient']} = "
          f"{int(c['counts'].sum() - c['counts'][c['names'].index(WORKED['vs_rest_patient'])])} cells")


def main() -> None:
    rs.apply()
    tables = load_tables()
    patient, label = load_cohort()
    worked = worked_example(tables, patient, label)
    _print_worked(worked)
    paths = [
        fig03w1_one_factor(worked),
        fig03w2_factors_to_call(worked),
        fig03w3_call_types(tables["calls"]),
        fig03a_decision_map(tables["calls"], tables["by_stratum"]),
        fig03b_patient_vs_rest(tables["calls"], tables["by_patient"]),
    ]
    profiles = single_factor_profiles(patient, label)
    for prof in profiles:
        print(
            f"{prof['panel_id']} {prof['method']} K{prof['k']} f{prof['factor']}: eta2 {prof['eta2']:.4f}, "
            f"var_share {prof['var_share']:.4f}, "
            f"median pairwise {prof['median_pairwise']:.4f}, vs_rest "
            + ", ".join(f"{p} {v:.3f}" for p, v in prof["vs_rest"].items())
        )
    paths.append(fig03c_factor_spectrum(profiles))
    for p in paths:
        print(p.relative_to(HERE))


if __name__ == "__main__":
    main()
