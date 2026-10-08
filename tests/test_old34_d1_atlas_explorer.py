"""Tests for the read-only old34 D1 atlas explorer (scripts/mrd_stage0_2/apps).

Most tests use a tiny synthetic experiment whose cohort statistics, filter matches and leverage
were derived by hand (comments show the arithmetic).  One integration test re-checks the real
experiment against the saved D1/D1f diagnostics and is skipped when the data is not present.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APP_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "mrd_stage0_2" / "apps" / "old34_d1_atlas_explorer.py"
)


def load_app():
    spec = importlib.util.spec_from_file_location("old34_d1_atlas_explorer", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


explorer = load_app()

# --------------------------------------------------------------------------------------
# Synthetic experiment
# --------------------------------------------------------------------------------------

PATIENTS = ["P01", "P02", "P03", "P04", "P05"]
PREVALENCE = [0.10, 0.20, 0.05, 0.01, 0.30]
N_MALIGNANT = [100, 200, 50, 10, 300]
N_NORMAL = [900, 800, 950, 990, 700]
NOISE = 2.7e-17  # what an all-zero model gives instead of exact 0
D, P5, P10, F5 = "direct_gene", "pca|K=5", "pca|K=10", "factosig|K=5"
HVG = "hvg_top_requested_500__available_500"

PANELS = {  # panel -> (stage0_panel_type, atlas_row_block, n_covered_genes)
    "full_34": ("full_control", "non_geneset_control", 100),
    HVG: ("hvg_anchor_control", "non_geneset_control", 500),
    "bio_a": ("single_group_only", "biological", 60),
    "bio_b": ("single_geneset_only", "biological", 40),
    "bio_c": ("single_geneset_only", "biological", 8),
}
LIFTS = {  # (panel, configuration) -> lift (fraction) for P01..P05; bio_c has no P10 / F5 cell
    ("bio_a", D): [0.003, -0.002, 0.001, 0.0, 0.002],  # median 0.001, p20 -0.0004
    ("bio_a", P5): [0.20, 0.12, 0.08, 0.06, 0.10],  # median 0.10, p20 0.076
    ("bio_a", P10): [0.02, 0.01, 0.0, 0.01, 0.015],  # median 0.01, p20 0.008
    ("bio_a", F5): [0.25, 0.20, 0.15, 0.01, 0.18],  # median 0.18, p20 0.122
    ("bio_b", D): [0.05, 0.04, 0.03, 0.02, 0.035],  # median 0.035
    ("bio_b", P5): [0.50, 0.01, 0.0, -0.01, 0.30],  # median 0.01, p20 -0.002
    ("bio_b", P10): [0.10, 0.08, 0.06, 0.04, 0.02],  # median 0.06, p20 0.036
    ("bio_b", F5): [-NOISE, NOISE, -NOISE, NOISE, NOISE],  # all-zero model: median +2.7e-17, p20 -2.7e-17
    ("bio_c", D): [0.02, 0.01, 0.0, -0.01, 0.005],  # median 0.005, p20 -0.002
    ("bio_c", P5): [0.30, 0.20, 0.10, 0.05, 0.15],  # median 0.15, p20 0.09
}
for _panel, _base in (("full_34", 0.05), (HVG, 0.04)):
    for _k, _cfg in enumerate((D, P5, P10, F5)):
        LIFTS[(_panel, _cfg)] = [_base + 0.01 * _k + 0.002 * j for j in range(5)]
CELLS = sorted(LIFTS)
ZERO_MODEL_CELL = ("bio_b", F5)
TIE_BREAK_CELL = ("bio_a", P5)  # three scorecard candidates tie on everything except penalty / C


def _method_and_k(cfg: str) -> tuple[str, int]:
    return ("direct_gene", 50) if cfg == D else (cfg.split("|")[0], int(cfg.split("K=")[1]))


def _scorecard_rows(panel: str, cfg: str, family: str, nz_mean: float, nz_max: float) -> list[dict]:
    method, k = _method_and_k(cfg)
    base = dict(stage0_panel_id=panel, representation_family=family, stage1_method=method, effective_k=k, seed=42, status="ok", l1_ratio=0.5)
    rows = [
        dict(base, penalty="elasticnet", C=0.001, cell_weighted_auprc=0.50, cell_weighted_auroc=0.70,
             nonzero_coefficient_count_mean_across_folds=nz_mean, nonzero_coefficient_count_max_across_folds=nz_max),
        dict(base, penalty="l2", C=0.1, cell_weighted_auprc=0.30, cell_weighted_auroc=0.99,  # lower AUPRC: loses
             nonzero_coefficient_count_mean_across_folds=999.0, nonzero_coefficient_count_max_across_folds=999),
    ]
    if (panel, cfg) == TIE_BREAK_CELL:
        rows += [
            dict(base, penalty="l2", C=0.0001, cell_weighted_auprc=0.50, cell_weighted_auroc=0.70,  # ties; penalty rank loses
                 nonzero_coefficient_count_mean_across_folds=99.0, nonzero_coefficient_count_max_across_folds=99),
            dict(base, penalty="elasticnet", C=0.01, cell_weighted_auprc=0.50, cell_weighted_auroc=0.70,  # ties; larger C loses
                 nonzero_coefficient_count_mean_across_folds=77.0, nonzero_coefficient_count_max_across_folds=77),
            dict(base, penalty="l1", C=0.1, cell_weighted_auprc=0.40, cell_weighted_auroc=0.90,
                 nonzero_coefficient_count_mean_across_folds=88.0, nonzero_coefficient_count_max_across_folds=88),
            dict(base, representation_family="other", penalty="l1", C=0.1, cell_weighted_auprc=0.90, cell_weighted_auroc=0.9,
                 nonzero_coefficient_count_mean_across_folds=55.0, nonzero_coefficient_count_max_across_folds=55),
            dict(base, status="failed", penalty="l1", C=0.1, cell_weighted_auprc=0.95, cell_weighted_auroc=0.9,
                 nonzero_coefficient_count_mean_across_folds=44.0, nonzero_coefficient_count_max_across_folds=44),
        ]
    return rows


def _nonzero_truth() -> dict[tuple[str, str], tuple[float, float]]:
    truth = {cell: (1.5 + i, float(2 + i)) for i, cell in enumerate(CELLS)}
    truth[ZERO_MODEL_CELL] = (0.0, 0.0)
    truth[TIE_BREAK_CELL] = (7.5, 9.0)
    return truth


def _leverage_rows() -> list[dict]:
    rows = []

    def add(panel, method, k, scope, full_median, deltas):
        for pid, delta in zip(PATIENTS, deltas):
            rows.append(dict(
                summary_scope=scope, stage0_panel_id=panel, selected_stage1_method=method, selected_effective_k=k,
                dropped_patient_id=pid, full_median_heldout_lift=full_median,
                leave_one_patient_out_median_lift=full_median - delta, patient_leverage_delta=delta,
                patient_leverage_abs_delta=abs(delta),
            ))

    add("bio_b", "pca", 5, explorer.LEVERAGE_SCOPE, 0.02, [0.005, -0.01, -0.20, -0.20, 0.005])  # P03/P04 tie; basis differs
    add("bio_a", "factosig", 5, explorer.LEVERAGE_SCOPE, 0.18, [0.01, -0.005, 0.0, 0.005, -0.01])  # P01/P05 tie; same basis
    add("bio_a", "pca", 10, "some_other_scope", 0.01, [0.9] * 5)  # must be ignored
    return rows


def build_experiment(root: Path, *, scorecards: bool = True, leverage: bool = True) -> Path:
    diag = root / "analysis" / "scorecards" / "stage2_diagnostic"
    diag.mkdir(parents=True)
    rep_rows, pat_rows = [], []
    for (panel, cfg), lifts in LIFTS.items():
        ptype, block, genes = PANELS[panel]
        method, k = _method_and_k(cfg)
        rep_rows.append(dict(
            representation_id=f"{panel}|{cfg}", stage0_panel_id=panel, stage0_panel_type=ptype,
            short_panel_label=panel, panel_label_with_genes=f"{panel} ({genes})", biological_theme=f"theme_{panel}",
            n_covered_genes=genes, stage1_method=method, effective_k=k, configuration=cfg,
            patient_median_lift=float(np.median(lifts)), patient_p20_lift=float(np.quantile(lifts, 0.2)), atlas_row_block=block,
        ))
        for j, pid in enumerate(PATIENTS):
            pat_rows.append(dict(
                stage0_panel_id=panel, configuration=cfg, heldout_patient=pid, heldout_auprc=PREVALENCE[j] + lifts[j],
                heldout_lift=lifts[j], patient_prevalence=PREVALENCE[j], n_malignant_h=N_MALIGNANT[j], n_normal_h=N_NORMAL[j],
            ))
    pd.DataFrame(rep_rows).to_csv(diag / "old34_d1_atlas_representation_table.csv", index=False)
    pd.DataFrame(pat_rows).to_csv(diag / "old34_d1_atlas_supported_patient_metrics.csv", index=False)
    if leverage:
        lev_dir = root / "analysis" / "scorecards" / "stage2_figure3_sharedness_v2_jun4"
        lev_dir.mkdir(parents=True)
        pd.DataFrame(_leverage_rows()).to_csv(lev_dir / "v2_patient_leverage_on_panel_method_median_lift.csv", index=False)
    if scorecards:
        truth = _nonzero_truth()
        groups = {rel: [] for _, rel, _ in explorer.SCORECARDS}
        rels = [rel for _, rel, _ in explorer.SCORECARDS]
        for (panel, cfg) in CELLS:
            control = PANELS[panel][1] == "non_geneset_control"
            direct = cfg == D
            rel = rels[{(False, False): 0, (False, True): 1, (True, False): 2, (True, True): 3}[(control, direct)]]
            family = "dr" if rel == rels[0] else ("direct_gene" if direct else "dr")
            groups[rel] += _scorecard_rows(panel, cfg, family, *truth[(panel, cfg)])
        for rel, rows in groups.items():
            path = root / "analysis" / "scorecards" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, index=False)
    return root


@pytest.fixture()
def data_dir(tmp_path):
    return build_experiment(tmp_path / "exp")


@pytest.fixture()
def atlas(data_dir):
    return explorer.load_atlas(data_dir)


def cell(atlas, panel: str, cfg: str) -> tuple[int, int]:
    return atlas.panel_ids.index(panel), explorer.CONFIGS.index(cfg)


def matched(atlas, mask) -> set[tuple[str, str]]:
    return {(atlas.panel_ids[r], explorer.CONFIGS[c]) for r, c in zip(*np.nonzero(mask))}


def brute_force_leverage(lifts: list[float]) -> float:
    full = np.median(lifts)
    return max(abs(full - np.median(np.delete(lifts, k))) for k in range(len(lifts)))


# --------------------------------------------------------------------------------------
# Rows, statistics, counts
# --------------------------------------------------------------------------------------


def test_frozen_d1_row_order_and_grouped_layout(atlas):
    # controls first in the fixed D1 order, then biology by best p20 (bio_a 0.122 > bio_c 0.09 > bio_b 0.036)
    assert atlas.panel_ids == ["full_34", HVG, "bio_a", "bio_c", "bio_b"]
    assert atlas.n_controls == 2
    assert atlas.layouts["d1"].boundaries == (1.5,)
    grouped = atlas.layouts["grouped"]
    assert [atlas.panel_ids[i] for i in grouped.order] == [HVG, "full_34", "bio_a", "bio_c", "bio_b"]  # HVG, UNION, GROUP, SET
    assert grouped.boundaries == (0.5, 1.5, 2.5)
    assert grouped.labels[0].startswith("[HVG] ") and grouped.labels[2].startswith("[GROUP] ")
    assert grouped.labels[3].startswith("[SET] bio_c")


def test_cohort_statistics_match_independent_numpy(atlas):
    for (panel, cfg), lifts in LIFTS.items():
        r, c = cell(atlas, panel, cfg)
        assert atlas.median[r, c] == pytest.approx(np.median(lifts))
        assert atlas.mean[r, c] == pytest.approx(np.mean(lifts))
        assert atlas.p20[r, c] == pytest.approx(np.quantile(lifts, 0.2))
    assert atlas.available.sum() == len(LIFTS)
    expected = np.concatenate([[np.median(v), np.quantile(v, 0.2)] for v in LIFTS.values()])
    assert atlas.d1_vmax_pp == pytest.approx(max(np.quantile(np.abs(100 * expected), 0.98), 3.0))


def test_counts_are_strict_and_ignore_float_noise(atlas):
    for tau in (0.0, 5.0, 10.0):
        counts = explorer.count_lift(atlas, tau)
        for (panel, cfg), lifts in LIFTS.items():
            r, c = cell(atlas, panel, cfg)
            assert counts[r, c] == sum(100 * v > tau + 1e-7 for v in lifts), (panel, cfg, tau)
    assert np.isnan(explorer.count_lift(atlas, 0.0)[cell(atlas, "bio_c", P10)])  # unavailable stays NaN
    r, c = cell(atlas, *ZERO_MODEL_CELL)
    assert explorer.count_lift(atlas, 0.0)[r, c] == 0  # +2.7e-17 is not "above 0"
    assert explorer.count_ratio(atlas, 1.0)[r, c] == 0
    # exactly at the cutoff is NOT above it: bio_c direct gene has a lift of exactly 0.01 for P02
    assert explorer.count_lift(atlas, 1.0)[cell(atlas, "bio_c", D)] == 1  # only P01 (2 pp)
    # AUPRC/prevalence > 1 is the same statement as lift > 0
    assert np.array_equal(np.nan_to_num(explorer.count_ratio(atlas, 1.0), nan=-1), np.nan_to_num(explorer.count_lift(atlas, 0.0), nan=-1))
    # bio_a|pca5: P01 sits exactly on the cutoff ((0.10 + 0.20) / 0.10 == 3.0000000000000004 in floats), so it is not
    # "above 3"; P04 (7x) is; P02/P03/P05 are 1.6x / 2.6x / 1.33x.
    r, c = cell(atlas, "bio_a", P5)
    assert explorer.count_ratio(atlas, 3.0)[r, c] == 1
    assert explorer.count_ratio(atlas, 2.5)[r, c] == 3  # P01 3x, P03 2.6x, P04 7x


# --------------------------------------------------------------------------------------
# Filters and leverage
# --------------------------------------------------------------------------------------


def test_filter_median_positive_with_p20_nonpositive(atlas):
    mask = explorer.filter_masks(atlas, explorer.View(f1=True))["f1"]
    # the all-zero model (+2.7e-17 median, -2.7e-17 p20) must not match; bio_a|P10 has p20 0.008 > 0
    assert matched(atlas, mask) == {("bio_a", D), ("bio_b", P5), ("bio_c", D)}
    mean_mask = explorer.filter_masks(atlas, explorer.View(f1=True, center="mean"))["f1"]
    expected = {
        key for key, lifts in LIFTS.items() if np.mean(lifts) > 1e-9 and np.quantile(lifts, 0.2) <= 1e-9
    }
    assert matched(atlas, mean_mask) == expected


def test_filter_direct_gene_near_zero_with_strong_dr_cell(atlas):
    mask = explorer.filter_masks(atlas, explorer.View(f2=True))["f2"]
    # bio_a: direct 0.1 pp, DR >= 5 pp at pca5 and factosig5 (pca10 is 1 pp); bio_c: direct 0.5 pp, DR pca5 15 pp
    assert matched(atlas, mask) == {("bio_a", D), ("bio_a", P5), ("bio_a", F5), ("bio_c", D), ("bio_c", P5)}
    assert mask[:, 0].sum() == 2
    loose = explorer.filter_masks(atlas, explorer.View(f2=True, f2_eps_pp=4.0, f2_delta_pp=0.5))["f2"]
    assert ("bio_b", D) in matched(atlas, loose)  # direct 3.5 pp is "near zero" at 4 pp
    strict = explorer.filter_masks(atlas, explorer.View(f2=True, f2_eps_pp=0.05))["f2"]
    assert not strict.any()


def test_d1_leverage_matches_brute_force_and_reports_all_tied_drivers(atlas):
    for (panel, cfg), lifts in LIFTS.items():
        r, c = cell(atlas, panel, cfg)
        assert atlas.lev_d1[r, c] == pytest.approx(brute_force_leverage(lifts), abs=1e-12), (panel, cfg)
    r, c = cell(atlas, "bio_b", P5)  # lifts 0.50 0.01 0.0 -0.01 0.30 -> dropping P03 or P04 moves the median 0.01 -> 0.155
    assert atlas.lev_d1[r, c] == pytest.approx(0.145)
    assert atlas.lev_d1_driver[r, c] == "P03/P04"
    assert atlas.lev_d1_delta[r, c] == pytest.approx(0.01 - 0.155)
    assert explorer.drivers_text("P03/P04") == "any of P03/P04" and explorer.drivers_text("P03") == "P03"


def test_filter_high_leverage_from_v2_csv_and_from_d1_table(atlas):
    csv_mask = explorer.filter_masks(atlas, explorer.View(f3=True, f3_source="csv", f3_theta_pp=7.5))["f3"]
    assert matched(atlas, csv_mask) == {("bio_b", P5)}  # CSV: 20 pp; bio_a|F5 is only 1 pp
    d1_mask = explorer.filter_masks(atlas, explorer.View(f3=True, f3_source="d1", f3_theta_pp=7.5))["f3"]
    expected = {key for key, lifts in LIFTS.items() if 100 * brute_force_leverage(lifts) >= 7.5 - 1e-7}
    assert matched(atlas, d1_mask) == expected and ("bio_b", P5) in expected
    assert not explorer.filter_masks(atlas, explorer.View(f3=True, f3_theta_pp=99.0))["f3"].any()
    both = explorer.combined_mask(atlas, explorer.View(f1=True, f3=True))
    assert matched(atlas, both) == {("bio_b", P5)}  # intersection of F1 and the CSV leverage filter
    assert explorer.combined_mask(atlas, explorer.View()) is None


def test_leverage_csv_join_and_basis_audit(atlas):
    r, c = cell(atlas, "bio_b", P5)
    assert atlas.lev_csv[r, c] == pytest.approx(0.20) and atlas.lev_csv_driver[r, c] == "P03/P04"
    assert atlas.lev_csv_full_median[r, c] == pytest.approx(0.02) and atlas.median[r, c] == pytest.approx(0.01)
    r2, c2 = cell(atlas, "bio_a", F5)
    assert atlas.lev_csv[r2, c2] == pytest.approx(0.01) and atlas.lev_csv_driver[r2, c2] == "P01/P05"
    assert np.isfinite(atlas.lev_csv).sum() == 2  # the other-scope rows are ignored
    assert atlas.audit["leverage_csv_cells"] == 2
    assert atlas.audit["leverage_csv_basis_mismatch_cells"] == 1  # only bio_b|pca5 disagrees with the D1 median
    assert atlas.audit["leverage_csv_basis_max_abs_diff_pp"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Stage 2 regularization selection and nonzero join
# --------------------------------------------------------------------------------------


def test_regularization_rule_breaks_ties_like_d1():
    frame = pd.DataFrame(
        [
            dict(stage0_panel_id="p", configuration="c", seed=1, penalty="l2", C=0.0001, cell_weighted_auprc=0.5, cell_weighted_auroc=0.7),
            dict(stage0_panel_id="p", configuration="c", seed=1, penalty="elasticnet", C=0.01, cell_weighted_auprc=0.5, cell_weighted_auroc=0.7),
            dict(stage0_panel_id="p", configuration="c", seed=1, penalty="elasticnet", C=0.001, cell_weighted_auprc=0.5, cell_weighted_auroc=0.7),
            dict(stage0_panel_id="p", configuration="c", seed=1, penalty="l1", C=0.1, cell_weighted_auprc=0.4, cell_weighted_auroc=0.99),
            dict(stage0_panel_id="q", configuration="c", seed=1, penalty="l2", C=1.0, cell_weighted_auprc=0.6, cell_weighted_auroc=0.6),
            dict(stage0_panel_id="q", configuration="c", seed=1, penalty="l1", C=1.0, cell_weighted_auprc=0.6, cell_weighted_auroc=0.7),
        ]
    )
    chosen = explorer.select_global_regularization(frame).set_index("stage0_panel_id")
    assert (chosen.loc["p", "penalty"], chosen.loc["p", "C"]) == ("elasticnet", 0.001)  # AUPRC tie -> AUROC tie -> penalty -> smaller C
    assert chosen.loc["q", "penalty"] == "l1"  # higher AUROC wins the AUPRC tie


def test_nonzero_counts_are_joined_at_the_selected_regularization(atlas):
    truth = _nonzero_truth()
    assert atlas.audit["nonzero_cells"] == len(LIFTS)
    for (panel, cfg), (mean, maximum) in truth.items():
        r, c = cell(atlas, panel, cfg)
        assert atlas.nonzero_mean[r, c] == pytest.approx(mean), (panel, cfg)
        assert atlas.nonzero_max[r, c] == pytest.approx(maximum), (panel, cfg)
    r, c = cell(atlas, *TIE_BREAK_CELL)
    assert atlas.reg_label[r, c] == "elasticnet C=0.001 l1_ratio=0.5"
    r, c = cell(atlas, *ZERO_MODEL_CELL)
    assert atlas.nonzero_max[r, c] == 0 and abs(atlas.lift[:, r, c]).max() < 1e-9  # all-zero model <=> zero lift
    spec = explorer.color_spec(atlas, explorer.View(mode="nonzero"))
    assert spec.z[cell(atlas, "bio_a", P5)] == pytest.approx(np.log10(1 + 7.5))


def test_missing_scorecards_disable_nonzero_without_breaking_the_atlas(tmp_path):
    atlas = explorer.load_atlas(build_experiment(tmp_path / "exp2", scorecards=False))
    assert not atlas.has_nonzero and any("nonzero" in note for note in atlas.notes)
    figure = explorer.build_heatmap(atlas, explorer.View(mode="nonzero"))  # must not crash even if forced
    assert len(figure.data) >= 2
    assert explorer.build_heatmap(atlas, explorer.View(mode="center")).layout.height > 0


def test_missing_leverage_csv_falls_back_to_the_d1_table(tmp_path):
    atlas = explorer.load_atlas(build_experiment(tmp_path / "exp3", leverage=False))
    assert not atlas.has_leverage_csv
    view = explorer.View.parse({"f3": True, "f3_source": "csv"}, atlas)
    assert view.f3_source == "d1" and explorer.filter_masks(atlas, view)["f3"].any()


# --------------------------------------------------------------------------------------
# Figures and cell view
# --------------------------------------------------------------------------------------


def _y_ranges(figure):
    axes = [figure.layout[key] for key in figure.layout if key.startswith("yaxis")]
    return axes, [tuple(axis.range) for axis in axes]


def test_row_figures_use_one_shared_y_axis(atlas):
    r, c = cell(atlas, "bio_a", F5)
    patients_fig = explorer.patient_row_figure(atlas, r, c, highlight="P02")
    axes, ranges = _y_ranges(patients_fig)
    assert len(axes) == len(PATIENTS) and len(set(ranges)) == 1
    assert all(axis.matches in (None, "y") for axis in axes)
    top = ranges[0][1]
    assert top >= max(max(PREVALENCE), max(100 * v + 100 * p for v, p in zip(LIFTS[("bio_a", F5)], PREVALENCE))) - 1e-9
    profile_fig = explorer.profile_row_figure(atlas, explorer.View(), r, c)
    axes, ranges = _y_ranges(profile_fig)
    assert len(axes) == len(explorer.METHODS) and len(set(ranges)) == 1
    assert all(axis.matches in (None, "y") for axis in axes)
    lo, hi = ranges[0]
    plotted = [y for trace in profile_fig.data if trace.y is not None for y in trace.y if y is not None]
    assert lo <= min(plotted) and hi >= max(plotted)  # nothing is clipped by the shared range


def test_cell_view_shows_all_patients_with_prevalence_and_cell_counts(atlas):
    r, c = cell(atlas, "bio_a", P5)
    view = explorer.View(tau_pp=5.0, rho=2.0, selected=(r, c))
    table = explorer.patient_table(atlas, view, r, c)
    assert table["patient"].tolist() == PATIENTS
    assert table["malignant cells"].tolist() == N_MALIGNANT and table["non-malignant cells"].tolist() == N_NORMAL
    assert table["prevalence %"].tolist() == pytest.approx([100 * p for p in PREVALENCE])
    assert table["lift pp"].tolist() == pytest.approx([100 * v for v in LIFTS[("bio_a", P5)]])
    assert table["held-out AUPRC %"].tolist() == pytest.approx([100 * (p + v) for p, v in zip(PREVALENCE, LIFTS[("bio_a", P5)])])
    title, subtitle, chips = explorer.cell_header(atlas, view, r, c)
    chip = dict(chips)
    assert "bio_a (60)" in title and "pca K=5" in title and "60 covered genes" in subtitle
    assert chip["median"] == "+10.00 pp" and chip["lift > 5 pp"] == "5/5" and chip["nonzero coefs"].startswith("7.5 mean / 9 max")
    assert "elasticnet C=0.001" in chip["selected regularization"] and "leverage, D1 basis" in chip
    fig = explorer.patient_row_figure(atlas, r, c)
    text = " ".join(str(a.text) for a in fig.layout.annotations)
    assert all(pid in text for pid in PATIENTS) and "100 mal" in text and "900 non" in text


def test_rank_figure_shows_the_tau_cutoff_only_in_the_count_mode(atlas):
    r, c = cell(atlas, "bio_a", P5)

    def names(view):
        return [t.name for t in explorer.lift_rank_figure(atlas, view, r, c).data]

    assert "tau = 5 pp" in names(explorer.View(mode="count_lift", tau_pp=5.0))
    assert not any(n and n.startswith("tau") for n in names(explorer.View(mode="center")))
    far = explorer.lift_rank_figure(atlas, explorer.View(mode="count_lift", tau_pp=50.0), r, c)
    assert far.layout.yaxis.range[1] >= 50  # an off-scale cutoff stays visible


def test_heatmap_marks_missing_cells_and_dims_non_matches(atlas):
    plain = explorer.build_heatmap(atlas, explorer.View())
    gray = plain.data[0]
    assert np.isfinite(gray.z).sum() == (~atlas.available).sum() and gray.colorscale[0][1] == explorer.GRAY
    main = plain.data[1]
    assert np.isfinite(main.z).sum() == atlas.available.sum()
    assert list(plain.layout.yaxis.ticktext) == atlas.layouts["d1"].labels
    assert [s.y0 for s in plain.layout.shapes if s.y0 == s.y1] == [1.5]  # control separator
    assert sorted(s.x0 for s in plain.layout.shapes if s.x0 == s.x1) == list(explorer.COL_BOUNDARIES)  # method blocks
    filtered = explorer.build_heatmap(atlas, explorer.View(f1=True))
    outline = [t for t in filtered.data if t.type == "scatter"][0]
    assert sum(v is None for v in outline.x) == 3  # three matching cells outlined
    main_z, dim_z = filtered.data[1].z, filtered.data[2].z
    assert np.isfinite(main_z).sum() == 3 and np.isfinite(dim_z).sum() == atlas.available.sum() - 3
    selected = explorer.build_heatmap(atlas, explorer.View(selected=cell(atlas, "bio_a", P5)))
    assert any(t.type == "scatter" and t.line.width == 3.5 for t in selected.data)


def test_color_modes_use_the_documented_scales(atlas):
    median = explorer.color_spec(atlas, explorer.View(mode="center"))
    assert median.zmax == pytest.approx(atlas.d1_vmax_pp) and median.zmin == -median.zmax
    mean = explorer.color_spec(atlas, explorer.View(mode="center", center="mean"))
    r, c = cell(atlas, "bio_a", P5)
    assert mean.z[r, c] == pytest.approx(100 * np.mean(LIFTS[("bio_a", P5)]))
    assert explorer.color_spec(atlas, explorer.View(mode="p20")).z[r, c] == pytest.approx(100 * np.quantile(LIFTS[("bio_a", P5)], 0.2))
    k = PATIENTS.index("P02")
    pl = explorer.color_spec(atlas, explorer.View(mode="patient_lift", patient="P02"))
    assert pl.z[r, c] == pytest.approx(100 * LIFTS[("bio_a", P5)][k])
    pa = explorer.color_spec(atlas, explorer.View(mode="patient_auprc", patient="P02"))
    assert pa.z[r, c] == pytest.approx(PREVALENCE[k] + LIFTS[("bio_a", P5)][k])
    for tau in (2.0, 12.0):
        spec = explorer.color_spec(atlas, explorer.View(mode="count_lift", tau_pp=tau))
        assert spec.z[r, c] == sum(100 * v > tau for v in LIFTS[("bio_a", P5)]) and (spec.zmin, spec.zmax) == (-0.5, len(PATIENTS) - 0.5)
    hover = explorer.hover_matrix(atlas, explorer.View(), median)[r, c]
    assert "bio_a (60)" in hover and "nonzero coefs 7.5" in hover and "p20" in hover
    gone = explorer.hover_matrix(atlas, explorer.View(), median)[cell(atlas, "bio_c", P10)]
    assert "unavailable" in gone and "K=10 exceeds the panel's 8 covered genes" in gone


def test_view_parse_replaces_blank_and_invalid_input_with_defaults(atlas):
    view = explorer.View.parse(
        {"tau_pp": None, "rho": "abc", "vmax_pp": -3, "patient": "P99", "mode": "nonsense", "row_order": "zzz",
         "selected": {"r": 99, "c": 99}, "f2_eps_pp": "", "f3_theta_pp": 1e9},
        atlas,
    )
    assert view.tau_pp == explorer.DEFAULT_TAU_PP and view.rho == explorer.DEFAULT_RHO and view.vmax_pp is None
    assert view.patient == explorer.DEFAULT_PATIENT and view.mode == "center" and view.row_order == "d1" and view.selected is None
    assert view.f2_eps_pp == explorer.DEFAULT_NEAR_ZERO_PP and view.f3_theta_pp == 100
    unavailable = explorer.View.parse({"selected": {"r": cell(atlas, "bio_c", P10)[0], "c": cell(atlas, "bio_c", P10)[1]}}, atlas)
    assert unavailable.selected is None


def test_small_formatters():
    assert explorer.fmt_pp(-2.7e-15) == "0.00" and explorer.fmt_pp(1.234) == "+1.23" and explorer.fmt_pp(float("nan")) == "n/a"
    assert explorer.shorten("x" * 80 + " (685)", 30).endswith("… (685)") and explorer.shorten("short (3)") == "short (3)"
    assert explorer.plural(1, "cell") == "1 cell" and explorer.plural(0, "cell") == "0 cells"


# --------------------------------------------------------------------------------------
# Read-only guarantees and sharing
# --------------------------------------------------------------------------------------


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


def test_loading_and_figures_never_modify_or_add_files(data_dir):
    before = _snapshot(data_dir)
    atlas = explorer.load_atlas(data_dir)
    explorer.build_heatmap(atlas, explorer.View(mode="count_lift", f1=True, f2=True, f3=True))
    explorer.patient_row_figure(atlas, 2, 1)
    assert _snapshot(data_dir) == before
    assert before == explorer.input_fingerprint(data_dir) | {k: v for k, v in before.items() if k not in explorer.input_fingerprint(data_dir)}


def test_export_bundle_is_self_contained_and_never_writes_into_the_experiment(data_dir, tmp_path):
    with pytest.raises(explorer.DataError, match="inside the experiment"):
        explorer.export_bundle(data_dir, data_dir / "analysis" / "bundle")
    before = _snapshot(data_dir)
    out = tmp_path / "bundle"
    written = explorer.export_bundle(data_dir, out)
    assert {p.name for p in written} == {
        "old34_d1_atlas_representation_table.csv", "old34_d1_atlas_supported_patient_metrics.csv",
        "v2_patient_leverage_on_panel_method_median_lift.csv", explorer.NONZERO_TABLE_NAME, "bundle_manifest.json",
    }
    manifest = json.loads((out / "bundle_manifest.json").read_text())
    assert manifest["experiment_id"] == explorer.EXPERIMENT_ID and all("bytes" in v for v in manifest["files"].values())
    assert _snapshot(data_dir) == before  # source untouched
    with pytest.raises(explorer.DataError, match="refusing to overwrite"):
        explorer.export_bundle(data_dir, out)
    original, bundled = explorer.load_atlas(data_dir), explorer.load_atlas(out)
    assert bundled.panel_ids == original.panel_ids
    for name in ("median", "p20", "mean", "nonzero_mean", "nonzero_max", "lev_csv"):
        assert np.array_equal(getattr(bundled, name), getattr(original, name), equal_nan=True), name
    assert list(bundled.reg_label.ravel()) == list(original.reg_label.ravel())


def test_missing_required_input_gives_an_actionable_error(tmp_path):
    with pytest.raises(explorer.DataError, match="old34_d1_atlas_representation_table.csv"):
        explorer.load_atlas(tmp_path)


# --------------------------------------------------------------------------------------
# Dash wiring (needs dash; exercised through Flask's test client, no server)
# --------------------------------------------------------------------------------------

VIEW_INPUTS = [
    ("color-mode", "value", "center"), ("center-stat", "value", "median"), ("tau", "value", 5), ("rho", "value", 2),
    ("patient", "value", "P02"), ("row-order", "value", "d1"), ("annotate", "value", []), ("vmax", "value", None),
    ("f1", "value", []), ("f2", "value", []), ("f2-eps", "value", 1), ("f2-delta", "value", 5), ("f3", "value", []),
    ("f3-theta", "value", 7.5), ("f3-source", "value", "csv"), ("sel", "data", None),
]


def _post(app, output_key, inputs, state=(), changed=None):
    parts = output_key.strip(".").split("...")
    outputs = [{"id": p.rsplit(".", 1)[0], "property": p.rsplit(".", 1)[1]} for p in parts]
    payload = {
        "output": output_key,
        "outputs": outputs if len(outputs) > 1 else outputs[0],
        "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
        "state": [{"id": i, "property": p, "value": v} for i, p, v in state],
        "changedPropIds": [changed] if changed else [],
    }
    return app.server.test_client().post("/_dash-update-component", data=json.dumps(payload), content_type="application/json")


def _key(app, text, *, startswith=False):
    return next(k for k in app.callback_map if (k.startswith(text) if startswith else text in k))


def _view(**overrides):
    return [(i, p, overrides.get(i.replace("-", "_"), v)) for i, p, v in VIEW_INPUTS]


def test_dash_click_cell_then_back_roundtrip(atlas):
    pytest.importorskip("dash")
    app = explorer.create_app(atlas)
    click_key = _key(app, "..sel.data", startswith=True)
    state = [("row-order", "value", "d1"), ("color-mode", "value", "center")]

    response = _post(app, click_key, [("heatmap", "clickData", {"points": [{"x": 1, "y": 2}]}), ("btn-back", "n_clicks", None)], state, "heatmap.clickData")
    assert response.status_code == 200 and response.get_json()["response"]["sel"]["data"] == {"r": 2, "c": 1}  # bio_a x pca K=5

    grouped = [("row-order", "value", "grouped"), ("color-mode", "value", "center")]
    response = _post(app, click_key, [("heatmap", "clickData", {"points": [{"x": 1, "y": 0}]}), ("btn-back", "n_clicks", None)], grouped, "heatmap.clickData")
    assert response.get_json()["response"]["sel"]["data"] == {"r": 1, "c": 1}  # grouped row 0 is the HVG control (canonical row 1)

    response = _post(app, click_key, [("heatmap", "clickData", {"points": [{"x": 2, "y": 3}]}), ("btn-back", "n_clicks", None)], state, "heatmap.clickData")
    assert response.status_code == 204 or "sel" not in (response.get_json() or {}).get("response", {})  # unavailable cell: ignored

    patient_mode = [("row-order", "value", "d1"), ("color-mode", "value", "patient_lift")]
    response = _post(app, click_key, [("heatmap", "clickData", None), ("btn-back", "n_clicks", 1)], patient_mode, "btn-back.n_clicks")
    body = response.get_json()["response"]
    assert body["sel"]["data"] is None and body["color-mode"]["value"] == "center"  # Back returns to the cohort map

    stay = [("row-order", "value", "d1"), ("color-mode", "value", "count_lift")]
    response = _post(app, click_key, [("heatmap", "clickData", None), ("btn-back", "n_clicks", 1)], stay, "btn-back.n_clicks")
    body = response.get_json()["response"]
    assert body["sel"]["data"] is None and "color-mode" not in body  # clearing a selection keeps the current map


def test_dash_heatmap_and_cell_callbacks_cover_every_mode_and_filter(atlas):
    pytest.importorskip("dash")
    app = explorer.create_app(atlas)
    heat_key, cell_key = _key(app, "heatmap.figure"), _key(app, "cell-body.children")
    for mode in explorer.MODE_LABELS:
        for center in ("median", "mean"):
            for order in ("d1", "grouped"):
                for selected in (None, {"r": 2, "c": 1}):
                    inputs = _view(color_mode=mode, center_stat=center, row_order=order, f1=["on"], f2=["on"], f3=["on"], sel=selected)
                    for key in (heat_key, cell_key):
                        assert _post(app, key, inputs).status_code == 200, (mode, center, order, selected)
    inputs = _view(tau=None, rho="abc", vmax="x", patient=None, f2_eps=None, f3_theta=None)  # blank / invalid controls
    assert _post(app, heat_key, inputs).status_code == 200
    body = _post(app, heat_key, _view(sel={"r": 2, "c": 1})).get_json()["response"]
    assert body["btn-back"]["disabled"] is False
    assert _post(app, heat_key, _view()).get_json()["response"]["btn-back"]["disabled"] is True
    text = json.dumps(_post(app, cell_key, _view(sel={"r": 2, "c": 1})).get_json())
    assert all(pid in text for pid in PATIENTS) and "bio_a (60)" in text
    assert "Click any colored cell" in json.dumps(_post(app, cell_key, _view()).get_json())


# --------------------------------------------------------------------------------------
# Integration: the real experiment must reproduce the saved D1 / D1f diagnostics
# --------------------------------------------------------------------------------------


def test_real_experiment_reproduces_saved_d1_and_d1f_diagnostics():
    experiment = explorer.find_experiment_dir()
    if experiment is None or not (experiment / "analysis/scorecards/stage2_diagnostic/old34_d1_atlas_representation_table.csv").is_file():
        pytest.skip("old34 experiment data is not available here")
    before = explorer.input_fingerprint(experiment)
    atlas = explorer.load_atlas(experiment)
    after = explorer.input_fingerprint(experiment)
    assert atlas.shape == (47, 17) and atlas.audit["cells"] == 783 and atlas.audit["unavailable_cells"] == 16
    failures = [row for row in explorer.run_checks(atlas, before, after) if row[1] == "FAIL"]
    assert not failures, failures
