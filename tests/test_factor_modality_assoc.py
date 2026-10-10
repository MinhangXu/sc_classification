"""Tests for the D27 factor x modality module (scripts/mrd_stage0_2/workstreams/W5_factor_interpretation).

Synthetic tests plant associations with known answers (a Simpson-type patient shift, a
within-patient marker association, a cell-type effect, a degenerate factor) and check the
vectorized Spearman against scipy. Regression tests read the real factor_modality_assoc_v1
outputs and are skipped when the experiment dir (or the outputs) are absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
W5 = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W5_factor_interpretation"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, W5 / "{}.py".format(name))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fma = load("factor_modality_assoc")
fc = load("factor_cards")

PATIENTS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08"]
SUPPORTED = ("P01", "P02", "P03", "P04", "P05", "P06", "P07")  # P08: unsupported (5 malignant cells)


# --------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------


def test_rank_spearman_matches_scipy_with_ties_and_constant_column():
    rng = np.random.default_rng(0)
    n = 400
    a = rng.integers(0, 6, size=(n, 4)).astype(float)  # heavy ties
    a[:, 3] = 2.0  # constant
    b = np.column_stack([a[:, 0] + rng.normal(size=n), rng.normal(size=n), -a[:, 1] + 0.5 * rng.normal(size=n)])
    rho = fma.spearman_from_ranks(fma.rank_standardize(a), fma.rank_standardize(b))
    for i in range(3):
        for j in range(3):
            assert rho[i, j] == pytest.approx(spearmanr(a[:, i], b[:, j]).statistic, abs=1e-12)
    assert np.isnan(rho[3]).all()


def test_clr_matches_factor_cards():
    counts = np.random.default_rng(1).poisson(5, size=(50, 12))
    np.testing.assert_allclose(fma.clr_per_cell(counts), fc.clr_per_cell(counts))


def test_eta2_by_group_matches_factor_cards():
    rng = np.random.default_rng(2)
    groups = np.array(["A"] * 50 + ["B"] * 40 + ["C"] * 8 + ["D"] * 30)
    x = rng.normal(size=(groups.size, 3))
    x[groups == "A", 0] += 2.0
    x[:, 2] = 1.0  # constant column -> 0
    for min_cells in (1, 10, 45):
        np.testing.assert_allclose(fma.eta2_by_group(x, groups, min_cells), fc.eta2_groups(x, groups, min_cells), equal_nan=True)
    assert np.isnan(fma.eta2_by_group(x, groups, 60)).all()  # fewer than two groups qualify


# --------------------------------------------------------------------------------------
# synthetic cohort
# --------------------------------------------------------------------------------------


def make_cohort(seed: int = 0, n_per_patient: int = 300):
    rng = np.random.default_rng(seed)
    patients = np.repeat(PATIENTS, n_per_patient)
    labels = np.zeros(patients.size, dtype=int)
    for i, p in enumerate(PATIENTS):
        idx = np.flatnonzero(patients == p)
        labels[idx[: 40 + 10 * i]] = 1 if p != "P08" else 0
    labels[np.flatnonzero(patients == "P08")[:5]] = 1  # P08: 5 malignant (below the stratum minimum)
    cell_types = np.array(["HSC", "Mono", "T"])[rng.integers(0, 3, patients.size)]
    return patients, labels, cell_types, rng


def make_scores_and_adt(patients, labels, cell_types, rng):
    """factor 1: tracks marker M1 within every patient (same sign);
    factor 2: Simpson - patient means of factor and M2 rise together, within-patient slope negative;
    factor 3: degenerate; factor 4: cell-type effect (HSC high)."""
    n = patients.size
    offset = {p: i for i, p in enumerate(PATIENTS)}
    off = np.array([offset[p] for p in patients], dtype=float)
    m1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    scores = np.zeros((n, 4))
    scores[:, 0] = m1 + 0.8 * rng.normal(size=n)
    scores[:, 1] = 3.0 * off + noise2
    scores[:, 3] = 2.0 * (cell_types == "HSC") + rng.normal(size=n)
    # ADT counts: M1 follows m1, M2 = patient offset minus the within-patient factor-2 noise, M3 noise, 2 isotypes
    lam = np.column_stack([
        np.exp(2.0 + 0.6 * m1),
        np.exp(1.0 + 0.5 * off - 0.4 * noise2),
        np.full(n, 6.0),
        np.full(n, 2.0),
        np.full(n, 2.0),
    ])
    counts = rng.poisson(lam)
    names = ["ADT-M1", "ADT-M2", "ADT-M3", "Isotype-A", "Isotype-B"]
    return scores, counts, names


@pytest.fixture(scope="module")
def synthetic():
    patients, labels, cell_types, rng = make_cohort()
    scores, counts, names = make_scores_and_adt(patients, labels, cell_types, rng)
    strata = fma.build_strata(patients, labels, cell_types, SUPPORTED, patient_min=20, pct_min=30)
    adt = fma.build_adt_strata(fma.clr_per_cell(counts), names, strata)
    cards = pd.DataFrame({
        "card_id": [fc.card_id("toy", "pca", 4, f) for f in range(1, 5)],
        "factor": [1, 2, 3, 4],
        "orientation_sign": [1, -1, 0, 0],
        "degenerate_flag": [False, False, True, False],
    })
    tables = fma.representation_tables(scores, cards, patients, labels, cell_types, SUPPORTED, tuple(PATIENTS), adt)
    return {"patients": patients, "labels": labels, "cell_types": cell_types, "scores": scores, "counts": counts, "names": names,
            "strata": strata, "adt": adt, "cards": cards, "tables": tables}


def test_build_strata(synthetic):
    strata = {s.key: s for s in synthetic["strata"]}
    assert "normal|pooled" in strata and "malignant|pooled" in strata
    assert strata["malignant|pooled"].n_cells == int(synthetic["labels"].sum())
    assert "malignant|P08" not in strata  # 5 malignant cells < patient_min
    assert "normal|P08" in strata and not strata["normal|P08"].supported
    pct = [s for s in synthetic["strata"] if s.kind == "patient_celltype"]
    assert pct and all(s.patient in SUPPORTED and s.n_cells >= 30 for s in pct)
    for s in pct:
        assert set(synthetic["cell_types"][s.rows]) == {s.cell_type}
        assert set(synthetic["patients"][s.rows]) == {s.patient}


def test_degenerate_card_skipped_and_row_count(synthetic):
    adt = synthetic["tables"]["adt"]
    assert fc.card_id("toy", "pca", 4, 3) not in set(adt["card_id"])
    assert len(adt) == 3 * 2 * len(synthetic["names"])
    assert set(adt.loc[adt["is_isotype"], "adt_marker"]) == {"Isotype-A", "Isotype-B"}


def test_within_patient_association_is_robust(synthetic):
    adt = synthetic["tables"]["adt"].set_index(["card_id", "label", "adt_marker"])
    row = adt.loc[(fc.card_id("toy", "pca", 4, 1), "normal", "ADT-M1")]
    assert row["rho_within_median"] > 0.4
    assert row["n_patients_same_sign"] == len(SUPPORTED) == row["n_patients_valid"]
    assert row["rho_within_min"] <= row["rho_within_median"] <= row["rho_within_max"]
    flat = synthetic["tables"]["adt"]
    mask = fma.robust_mask(flat)
    hit = (flat["card_id"] == fc.card_id("toy", "pca", 4, 1)) & (flat["label"] == "normal") & (flat["adt_marker"] == "ADT-M1")
    assert bool(mask[hit].iloc[0])


def test_simpson_pooled_vs_within(synthetic):
    """Factor 2: pooled rho > 0 (patient means move together), within-patient rho < 0.

    Orientation -1 flips both signs, so in the table pooled < 0 and within > 0."""
    adt = synthetic["tables"]["adt"].set_index(["card_id", "label", "adt_marker"])
    row = adt.loc[(fc.card_id("toy", "pca", 4, 2), "normal", "ADT-M2")]
    assert row["rho_pooled"] < -0.3  # raw pooled > 0, oriented by -1
    assert row["rho_within_median"] > 0.2  # raw within < 0, oriented by -1
    assert row["n_patients_same_sign"] == len(SUPPORTED)
    raw = fma.adt_rho_by_stratum(synthetic["scores"], synthetic["adt"])
    m2 = synthetic["names"].index("ADT-M2")
    assert raw["normal|pooled"][m2, 1] > 0.3
    assert all(raw["normal|{}".format(p)][m2, 1] < 0 for p in SUPPORTED)


def test_orientation_and_unoriented_flag(synthetic):
    adt = synthetic["tables"]["adt"]
    raw = fma.adt_rho_by_stratum(synthetic["scores"], synthetic["adt"])
    m1 = synthetic["names"].index("ADT-M1")
    c2 = adt.set_index(["card_id", "label", "adt_marker"]).loc[(fc.card_id("toy", "pca", 4, 2), "normal", "ADT-M1")]
    assert c2["rho_pooled"] == pytest.approx(-raw["normal|pooled"][m1, 1])
    per = [raw["normal|{}".format(p)][m1, 1] for p in SUPPORTED]
    assert c2["rho_within_min"] == pytest.approx(-max(per))
    assert c2["rho_within_max"] == pytest.approx(-min(per))
    assert c2["rho_P01"] == pytest.approx(-per[0])
    ori = adt.groupby("card_id")["oriented"].first()
    assert bool(ori[fc.card_id("toy", "pca", 4, 1)]) and not bool(ori[fc.card_id("toy", "pca", 4, 4)])


def test_scipy_agreement_on_strata(synthetic):
    clr = fma.clr_per_cell(synthetic["counts"])
    raw = fma.adt_rho_by_stratum(synthetic["scores"], synthetic["adt"])
    for s in synthetic["strata"][:6]:
        for mi in range(len(synthetic["names"]) - 2):
            for f in (0, 1, 3):
                ref = spearmanr(synthetic["scores"][s.rows, f], clr[s.rows, mi]).statistic
                assert raw[s.key][mi, f] == pytest.approx(ref, abs=1e-12)


def test_d18_weighted_mean_definition(synthetic):
    """rho_within_wmean_d18 = cell-weighted mean over patient strata meeting the D18 minimum."""
    raw = fma.adt_rho_by_stratum(synthetic["scores"], synthetic["adt"])
    summ = fma.summarize_rho(raw, synthetic["strata"], "normal", SUPPORTED, PATIENTS, d18_min={"normal": 200, "malignant": 20})
    pats = [s for s in synthetic["strata"] if s.label == "normal" and s.kind == "patient" and s.n_cells >= 200]
    expect = sum(raw[s.key] * s.n_cells for s in pats) / sum(s.n_cells for s in pats)
    np.testing.assert_allclose(summ["rho_within_wmean_d18"], expect)


def test_robust_rule_thresholds():
    frame = pd.DataFrame({
        "is_isotype": [False, False, False, True, False],
        "rho_within_median": [0.25, -0.31, 0.19, 0.5, 0.4],
        "n_patients_same_sign": [5, 7, 7, 7, 4],
    })
    assert list(fma.robust_mask(frame)) == [True, True, False, False, False]


def test_celltype_tables(synthetic):
    eta = synthetic["tables"]["celltype_eta2"].set_index(["card_id", "label"])
    hi = eta.loc[(fc.card_id("toy", "pca", 4, 4), "normal")]
    lo = eta.loc[(fc.card_id("toy", "pca", 4, 1), "normal")]
    assert hi["eta2_celltype_pooled"] > 0.3 and hi["eta2_celltype_within_median"] > 0.3
    assert lo["eta2_celltype_pooled"] < 0.05
    assert hi["n_patients_valid"] == len(SUPPORTED)
    means = synthetic["tables"]["celltype_means"]
    z = fma.oriented_cohort_z(synthetic["scores"], np.array([1, -1, 0, 0]))
    lab = synthetic["labels"] == 0
    ref = pd.Series(z[lab, 1]).groupby(synthetic["cell_types"][lab]).mean()
    got = means.loc[(means["card_id"] == fc.card_id("toy", "pca", 4, 2)) & (means["label"] == "normal")].set_index("cell_type")["mean_oriented_z"]
    np.testing.assert_allclose(got.sort_index().to_numpy(), ref.sort_index().to_numpy())
    byp = synthetic["tables"]["celltype_means_by_patient"]
    assert set(byp["patient"]) <= set(PATIENTS) and (byp["n_cells"] >= fma.CELLTYPE_MIN_CELLS_PATIENT).all()


def test_merge_summaries(synthetic):
    adt = synthetic["tables"]["adt"]
    cards = synthetic["cards"].assign(representation_id="toy|dr|pca|4|42", panel_id="toy", panel_label="toy", method="pca", k=4,
                                      orientation_status=["oriented", "oriented", "unoriented_all_zero", "unoriented_all_zero"],
                                      is_selected_driver=[True, False, False, False], n_active_folds=[3, 1, 0, 0], max_univariate_lift=0.1,
                                      eta2_normal=0.01, eta2_malignant=0.02)
    iso = fma.isotype_reference(adt)
    assert set(iso.columns) >= {"isotype_max_abs_rho_pooled", "isotype_max_abs_rho_within_median", "marker_max_abs_rho_within_median"}
    rob = fma.robust_table(adt, cards, iso, SUPPORTED)
    assert (rob["abs_rho_within_median"] if "abs_rho_within_median" in rob else rob["rho_within_median"].abs()).is_monotonic_decreasing
    assert not rob["adt_marker"].str.startswith("Isotype").any()
    summ = fma.card_summary(cards, adt, synthetic["tables"]["celltype_eta2"], synthetic["tables"]["celltype_means"], iso)
    assert len(summ) == 3 and fc.card_id("toy", "pca", 4, 3) not in set(summ["card_id"])
    s1 = summ.set_index("card_id").loc[fc.card_id("toy", "pca", 4, 1)]
    assert s1["adt_top_marker_normal"] == "M1" and s1["adt_n_robust_normal"] >= 1
    top = fma.top_markers(adt, n=2)
    assert top.groupby(["card_id", "label"]).size().max() == 2
    nf = fma.noise_floor_table(adt)
    assert set(nf["feature_kind"]) == {"isotype", "marker"}


# --------------------------------------------------------------------------------------
# runner: resume units and the out-root guard
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    return load("run_d27_factor_modality_assoc")


def test_runner_smoke_requires_out_root(runner):
    with pytest.raises(SystemExit, match="--out-root"):
        runner.main(["--shortlist", "smoke"])


def test_part_done_treats_empty_or_corrupt_as_unfinished(runner, tmp_path):
    paths = runner.Paths(tmp_path / "exp", tmp_path / "out")
    d = runner.part_dir(paths, "toy", "pca", 4)
    d.mkdir(parents=True)
    assert not runner.part_done(paths, "toy", "pca", 4)
    for name in fma.PART_TABLES:
        pd.DataFrame({"card_id": ["a"], "x": [1.0]}).to_parquet(d / "{}.parquet".format(name))
    assert not runner.part_done(paths, "toy", "pca", 4)  # no marker yet
    (d / "_done.json").write_text(json.dumps({"schema_version": fma.SCHEMA_VERSION}))
    assert runner.part_done(paths, "toy", "pca", 4)
    (d / "adt.parquet").write_bytes(b"")
    assert not runner.part_done(paths, "toy", "pca", 4)  # empty file
    (d / "adt.parquet").write_bytes(b"not a parquet file")
    assert not runner.part_done(paths, "toy", "pca", 4)  # corrupt file
    pd.DataFrame({"card_id": ["a"]}).to_parquet(d / "adt.parquet")
    (d / "_done.json").write_text("{broken")
    assert not runner.part_done(paths, "toy", "pca", 4)  # corrupt marker


def test_paths_refuse_foreign_namespace(runner, tmp_path):
    paths = runner.Paths(tmp_path / "exp", tmp_path / "out")
    assert paths.scorecards.name == fma.SLUG and paths.cache.name == fma.SLUG
    assert paths.rel(paths.scorecards / "x.csv").startswith("<out-root>/scorecards/stage2_diagnostic/{}".format(fma.SLUG))


# --------------------------------------------------------------------------------------
# regression against the real experiment (skipped without EXP / outputs)
# --------------------------------------------------------------------------------------


def _experiment():
    sys.path.insert(0, str(REPO / "src"))
    from sc_classification.utils.paths import experiment_dir

    exp = experiment_dir(must_exist=False)
    return exp if exp.is_dir() else None


EXP = _experiment()
OUT = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / fma.SLUG if EXP is not None else None
needs_outputs = pytest.mark.skipif(OUT is None or not (OUT / "{}_manifest.json".format(fma.SLUG)).is_file(), reason="experiment dir or factor_modality_assoc_v1 outputs not present")
IFNA = "single_geneset__hallmark_interferon_alpha_response"


@needs_outputs
def test_real_regression_checks_pass():
    checks = pd.read_csv(OUT / "d27_regression_checks.csv")
    assert checks["passed"].astype(bool).all(), checks.loc[~checks["passed"].astype(bool), "check_id"].tolist()
    assert {"d18_adt_pooled", "d18_adt_patient_stratified", "d18_celltype_means", "d18_devlog_anchors", "scipy_spearman_spot"} <= set(checks["check_id"])


@needs_outputs
def test_real_pc8_anchor_values():
    focus = pd.read_csv(OUT / "d27_ifna_pc8_fa7_adt.csv").set_index(["card_id", "label", "adt_marker"])
    pc8 = fc.card_id(IFNA, "pca", 40, 8)
    assert round(float(focus.loc[(pc8, "normal", "ADT-CD34"), "rho_pooled"]), 3) == 0.251
    assert round(float(focus.loc[(pc8, "malignant", "ADT-CD99"), "rho_pooled"]), 3) == 0.222


@needs_outputs
def test_real_tables_are_consistent():
    rob = pd.read_csv(OUT / "d27_adt_robust.csv")
    assert (rob["rho_within_median"].abs() >= fma.ROBUST_ABS_RHO - 1e-6).all()
    assert (rob["n_patients_same_sign"] >= fma.ROBUST_MIN_SAME_SIGN).all()
    assert not rob["adt_marker"].str.lower().str.startswith("isotype").any()
    summary = pd.read_csv(OUT / "d27_card_summary.csv")
    index = pd.read_csv(EXP / "analysis" / "scorecards" / "stage2_diagnostic" / fc.SLUG / "factor_cards_index.csv")
    assert set(summary["card_id"]) == set(index.loc[~index["degenerate_flag"].astype(bool), "card_id"])
