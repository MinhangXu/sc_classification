"""Tests for the D16 factor x patient allocation matrix (W1.1).

Module: scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/factor_patient_matrix.py
Runner: scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d16_factor_patient_matrix.py

Synthetic tests build scores with a known allocation (the arithmetic is in the comments).
The regression tests re-check the canonical experiment read-only (IFN-alpha K40 against F6,
the 2026-10-08 scratch anchor and D14) and are skipped when the experiment dir is absent.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
W1 = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W1_patient_aggregation"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fpm = _load("factor_patient_matrix", W1 / "factor_patient_matrix.py")
runner = _load("run_d16_factor_patient_matrix", W1 / "run_d16_factor_patient_matrix.py")


# --------------------------------------------------------------------------------------
# synthetic cohort with a known allocation
# --------------------------------------------------------------------------------------

N = {"PA": 100, "PB": 300, "PC": 600}  # cell shares 0.1, 0.3, 0.6
N_MAL = {"PA": 20, "PB": 30, "PC": 0}  # PC has no malignant cells -> unsupported


def _alt(n: int, amp: float) -> np.ndarray:
    """+amp, -amp, ... : mean 0, variance amp**2 (n even)."""
    return amp * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)


@pytest.fixture(scope="module")
def synth():
    patient = np.concatenate([[p] * n for p, n in N.items()])
    malignant = np.concatenate([np.arange(n) < N_MAL[p] for p, n in N.items()])
    cols = []
    # f1: PA at amplitude 3, PB/PC at 1. SS = 100*9 + 300 + 600 = 1800 about mean 0.
    #     shares PA 0.5, PB 1/6, PC 1/3 -> enrichment PA 5, PB 5/9, PC 5/9.
    cols.append(np.concatenate([_alt(100, 3.0), _alt(300, 1.0), _alt(600, 1.0)]))
    # f2: constant -> degenerate
    cols.append(np.zeros(1000))
    # f3: same +-1 pattern in every patient -> every enrichment is exactly 1
    cols.append(np.concatenate([_alt(100, 1.0), _alt(300, 1.0), _alt(600, 1.0)]))
    # f4: patient means PA +2, PB 0, PC -1/3 (cohort mean 0), no within-patient spread:
    #     eta2 within normal cells = 1; enrichment PA = 4 / var, var = (100*4 + 600/9)/1000
    cols.append(np.concatenate([np.full(100, 2.0), np.zeros(300), np.full(600, -1.0 / 3.0)]))
    # f5: malignant cells score +1 and normal cells -1 in PA/PB; PC normal cells alternate.
    f5 = np.where(malignant, 1.0, -1.0)
    f5[patient == "PC"] = _alt(600, 1.0)
    cols.append(f5)
    scores = np.column_stack(cols)
    cohort = fpm.Cohort(patient, malignant, supported=("PA", "PB"), high_prevalence=("PA",))
    return scores, cohort


def test_supported_patients_rule():
    patient = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 9 + ["D"] * 10)
    malignant = np.array([True] * 10 + [False] * 10 + [True] * 9 + [False] * 11 + [True] * 9 + [True] * 10)
    # A: 10 malignant + 10 normal -> supported; B: 9 malignant -> no; C: all 9 malignant -> no;
    # D: 10 malignant but no normal cells -> no (both classes required)
    assert fpm.supported_patients(patient, malignant, min_malignant=10) == ("A",)


def test_average_precision_matches_sklearn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(20, 400))
        y = rng.random(n) < rng.uniform(0.02, 0.5)
        if not y.any():
            y[0] = True
        s = np.round(rng.normal(size=n) + y * rng.uniform(0, 2), int(rng.integers(0, 3)))  # many ties
        assert fpm.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)
        assert fpm.average_precision(y, -s) == pytest.approx(sk.average_precision_score(y, -s), abs=1e-12)
    assert fpm.average_precision(np.array([0, 1, 0, 1]), np.zeros(4)) == pytest.approx(0.5)  # constant score -> pi


def test_known_allocation(synth):
    scores, cohort = synth
    t = fpm.allocation_table(scores, cohort, normal_min_cells=10, malignant_min_cells=5).set_index(
        ["factor_index", "patient"]
    )
    # shares sum to 1 per non-degenerate factor; cell shares are n/N
    for f in (1, 3, 4, 5):
        assert t.loc[f, "ss_share"].sum() == pytest.approx(1.0)
    assert t.loc[(1, "PA"), "cell_share"] == pytest.approx(0.1)
    assert t.loc[(1, "PA"), "enrichment"] == pytest.approx(5.0)
    assert t.loc[(1, "PB"), "enrichment"] == pytest.approx(5.0 / 9.0)
    assert t.loc[(1, "PC"), "enrichment"] == pytest.approx(5.0 / 9.0)
    # f1 is pure spread, f4 pure location; the split is exact
    np.testing.assert_allclose(t["enrichment"], t["enrichment_location"] + t["enrichment_spread"], rtol=1e-10)
    assert t.loc[(1, "PA"), "enrichment_location"] == pytest.approx(0.0, abs=1e-12)
    assert t.loc[(4, "PB"), "enrichment_spread"] == pytest.approx(0.0, abs=1e-12)
    var4 = (100 * 4 + 600 / 9) / 1000
    assert t.loc[(4, "PA"), "enrichment"] == pytest.approx(4 / var4)
    # f3: identical distribution in every patient
    np.testing.assert_allclose(t.loc[3, "enrichment"], 1.0)
    # f2: degenerate
    assert t.loc[2, "degenerate"].all() and t.loc[2, "enrichment"].isna().all()
    assert not t.loc[1, "degenerate"].any()


def test_label_split_shift_and_eta2(synth):
    scores, cohort = synth
    t = fpm.allocation_table(scores, cohort, normal_min_cells=10, malignant_min_cells=5).set_index(
        ["factor_index", "patient"]
    )
    # label shares: per label they sum to 1 over patients with cells of that label
    for f in (1, 4, 5):
        assert t.loc[f, "ss_share_normal"].sum() == pytest.approx(1.0)
        assert t.loc[f, "ss_share_malignant"].sum(min_count=1) == pytest.approx(1.0)
    assert np.isnan(t.loc[(1, "PC"), "enrichment_malignant"])  # PC has no malignant cells
    # absolute label SS add up to the patient SS
    np.testing.assert_allclose(t["ss_normal"] + t["ss_malignant"].fillna(0), t["ss"], rtol=1e-10)
    # f5: malignant +1 vs normal -1 inside PA/PB: shift = 2 cohort SDs; no within-class spread
    sd5 = scores[:, 4].std()
    assert t.loc[(5, "PA"), "shift_z_cohort"] == pytest.approx(2 / sd5)
    assert np.isnan(t.loc[(5, "PA"), "cohens_d_within"])  # pooled within SD is 0
    assert np.isnan(t.loc[(5, "PC"), "shift_z_cohort"])
    # f4: patient means differ, no spread -> eta2 normal = 1; contributions add up to eta2
    assert t.loc[(4, "PA"), "eta2_normal"] == pytest.approx(1.0)
    assert t.loc[4, "eta2_contrib_normal"].sum() == pytest.approx(1.0)
    # f3: same pattern everywhere -> eta2 0
    assert t.loc[(3, "PA"), "eta2_normal"] == pytest.approx(0.0, abs=1e-12)
    # malignant stratum (>=5 malignant cells): PA and PB only
    assert np.isnan(t.loc[(4, "PC"), "eta2_contrib_malignant"])


def test_univariate_orientation_from_training_folds():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    pats = np.repeat(["T1", "T2", "H"], [400, 400, 200])
    y = np.zeros(1000, dtype=bool)
    for p in ("T1", "T2", "H"):
        idx = np.flatnonzero(pats == p)
        y[idx[:40]] = True
    x = rng.normal(size=1000)
    x[(pats != "H") & y] += 2.0  # training patients: malignant high
    x[(pats == "H") & y] -= 2.0  # held-out patient: malignant low (sign flips)
    cohort = fpm.Cohort(pats, y, supported=("T1", "T2", "H"))
    u = fpm.univariate_heldout_table(x[:, None], cohort).set_index("patient")
    h = u.loc["H"]
    hm = pats == "H"
    # held-out H is trained on T1+T2, where the gap is positive -> orientation +1
    assert h["orientation"] == 1 and h["train_gap_z"] > 0
    assert h["heldout_ap_oriented"] == pytest.approx(sk.average_precision_score(y[hm], x[hm]))
    assert h["heldout_ap_sign_agnostic"] == pytest.approx(sk.average_precision_score(y[hm], -x[hm]))
    assert h["heldout_lift_oriented"] < 0 < h["heldout_lift_sign_agnostic"]
    assert not h["orientation_matches_heldout"]
    assert h["heldout_prevalence"] == pytest.approx(0.2)
    # training-fold AP for H: all cells outside H
    tm = ~hm
    assert h["train_ap_oriented"] == pytest.approx(sk.average_precision_score(y[tm], x[tm]))


def test_summaries(synth):
    scores, cohort = synth
    long = fpm.factor_patient_table(scores, cohort, normal_min_cells=10, malignant_min_cells=5)
    long.insert(0, "stage0_panel_id", "panel_x")
    long.insert(1, "short_panel_label", "panel x")
    long.insert(2, "stage1_method", "pca")
    long.insert(3, "k", 5)
    long.insert(4, "effective_k", 5)
    fs = fpm.factor_summary(long, high_prevalence=("PA",)).set_index("factor_index")
    assert fs.loc[1, "top_patient"] == "PA" and fs.loc[1, "top_enrichment"] == pytest.approx(5.0)
    assert fs.loc[1, "hp_ss_share"] == pytest.approx(0.5) and fs.loc[1, "hp_enrichment"] == pytest.approx(5.0)
    assert fs.loc[2, "degenerate"] and pd.isna(fs.loc[2, "top_patient"])
    assert fs.loc[1, "factor_ss"] == pytest.approx(1800.0)
    assert fs.loc[1, "malignant_cell_share"] == pytest.approx(0.05)
    rps = fpm.representation_patient_summary(long)
    assert rps.groupby("stage0_panel_id")["weighted_ss_share"].sum().iloc[0] == pytest.approx(1.0)
    rs = fpm.representation_summary(fs.reset_index(), rps, high_prevalence=("PA",))
    assert len(rs) == 1
    r = rs.iloc[0]
    assert r["n_nondegenerate"] == 4 and r["n_degenerate"] == 1
    # PA tops f1 (5x), f4 (spike) and f5; f3 ties at 1.0 for all patients -> first patient PA
    assert r["modal_top_patient"] == "PA"
    assert r["hp_cell_share"] == pytest.approx(0.1)
    freq = fpm.top_patient_frequency(fs.reset_index(), rs, list(cohort.patients))
    assert freq["frac_factors_top"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# runner end to end on a synthetic experiment dir
# --------------------------------------------------------------------------------------


def _write_fake_exp(root: Path, scores_by_unit: dict, patient: np.ndarray, malignant: np.ndarray) -> Path:
    exp = root / "exp_fake"
    for (panel, method, k), scores in scores_by_unit.items():
        d = exp / "stage1_dr" / f"panel_id={panel}" / f"method={method}" / f"k={k}" / "seed=42"
        d.mkdir(parents=True)
        np.save(d / "scores.npy", scores.astype(np.float32))
        (d / "metadata.json").write_text(json.dumps({"effective_k": k, "n_genes": 50}))
    meta_dir = exp / "stage2_supervised" / "multiobjective" / "discovery_full_cohort"
    meta_dir.mkdir(parents=True)
    pd.DataFrame(
        {"cell_id": [f"c{i}" for i in range(patient.size)], "patient": patient,
         "CN.label": np.where(malignant, "cancer", "normal")}
    ).to_csv(meta_dir / "cell_metadata.csv.gz", index=False)
    atlas = exp / "analysis" / "scorecards" / "stage2_diagnostic"
    atlas.mkdir(parents=True)
    panels = sorted({u[0] for u in scores_by_unit})
    pd.DataFrame(
        {"stage0_panel_id": panels, "short_panel_label": [p.replace("_", " ") for p in panels],
         "atlas_row_block": "biological", "n_covered_genes": 50}
    ).to_csv(atlas / "old34_d1_atlas_representation_table.csv", index=False)
    return exp


def test_runner_resume_and_merge(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    patient = np.repeat(["P02", "P03", "P04", "P09"], [60, 60, 60, 220])
    malignant = np.zeros(400, dtype=bool)
    for p in ("P02", "P03", "P04", "P09"):
        malignant[np.flatnonzero(patient == p)[:12]] = True
    units = {
        ("panel_a", "pca", 5): rng.normal(size=(400, 5)),
        ("panel_a", "fa", 5): np.column_stack([rng.normal(size=(400, 4)), np.zeros(400)]),
        ("panel_b", "pca", 5): rng.normal(size=(400, 5)),
    }
    exp = _write_fake_exp(tmp_path, units, patient, malignant)
    monkeypatch.setattr(runner, "N_CELLS", 400)
    monkeypatch.setattr(runner, "study_config", lambda *a, **k: {})
    base = ["--experiment-dir", str(exp), "--panel-ids", "panel_a,panel_b", "--ks", "5,10", "--no-figures"]
    assert runner.main(base) == 0
    dirs = runner.out_dirs(exp)
    parts = sorted(dirs["parts"].glob("*.csv.gz"))
    assert len(parts) == 3
    # a second plain run refuses to overwrite finished parts
    assert runner.main(base) == 2
    # an empty part and a truncated part count as unfinished and are recomputed by --resume
    parts[0].write_bytes(b"")
    with gzip.open(parts[1], "wt") as fh:
        fh.write("stage0_panel_id,k\nx,5\n")
    assert runner.main(base + ["--resume"]) == 0
    assert all(runner.read_part(p, 4) is not None for p in parts)
    assert runner.main(["--experiment-dir", str(exp), "--merge-only", "--no-figures"]) == 0
    td = dirs["tables"]
    long = pd.read_csv(td / "factor_patient_matrix_long.csv.gz")
    assert len(long) == 3 * 5 * 4
    rs = pd.read_csv(td / "factor_patient_matrix_representation_summary.csv")
    assert set(rs["n_degenerate"]) == {0, 1}
    manifest = json.loads((td / "factor_patient_matrix_manifest.json").read_text())
    assert manifest["units"]["n_merged"] == 3
    # missing units are reported within the merged panel x method x K grid
    assert manifest["units"]["missing_inputs"] == ["panel_b|fa|K5"]
    ledger = pd.read_csv(td / "factor_patient_matrix_artifact_ledger.csv")
    assert not ledger["artifact_relpath"].str.startswith("/").any()
    assert (td / "factor_patient_matrix_findings.md").read_text().count("not loading leverage") >= 1
    # a corrupt part makes --merge-only fail loudly and leaves merge_failed.json
    parts[2].write_bytes(b"")
    with pytest.raises(RuntimeError):
        runner.main(["--experiment-dir", str(exp), "--merge-only", "--no-figures"])
    assert (dirs["cache"] / "merge_failed.json").is_file()


# --------------------------------------------------------------------------------------
# regression against the canonical experiment (skipped without EXP)
# --------------------------------------------------------------------------------------

EXP = runner.EXP_DEFAULT
HAS_EXP = (EXP / runner.CELL_METADATA_DEFAULT).is_file() and (
    runner.score_dir(EXP, runner.FOCUS_PANEL, "pca", 40) / "scores.npy"
).is_file()
needs_exp = pytest.mark.skipif(not HAS_EXP, reason="canonical experiment dir not available")


@pytest.fixture(scope="module")
def ifna_k40():
    meta = pd.read_csv(EXP / runner.CELL_METADATA_DEFAULT, usecols=["patient", "CN.label"])
    y = meta["CN.label"].eq("cancer").to_numpy()
    sup = fpm.supported_patients(meta["patient"], y)
    cohort = fpm.Cohort(meta["patient"].to_numpy(), y, supported=sup)
    out = {}
    for method in ("pca", "fa"):
        scores = np.load(runner.score_dir(EXP, runner.FOCUS_PANEL, method, 40) / "scores.npy")
        out[method] = fpm.factor_patient_table(scores, cohort)
    return cohort, out


@needs_exp
def test_regression_supported_patients(ifna_k40):
    cohort, _ = ifna_k40
    assert cohort.supported == ("P01", "P02", "P03", "P04", "P06", "P07", "P09")


@needs_exp
def test_regression_f6_univariate(ifna_k40):
    f6_path = EXP / runner.F6_RELPATH
    if not f6_path.is_file():
        pytest.skip("F6 table absent")
    f6 = pd.read_csv(f6_path)
    _, tabs = ifna_k40
    for method, tab in tabs.items():
        u = tab.loc[tab["supported"]].assign(feature_id=lambda d: d["factor_index"].map(lambda i: f"factor_{i:03d}"))
        m = u.merge(f6.loc[f6["stage1_method"].eq(method)], left_on=["patient", "feature_id"],
                    right_on=["heldout_patient", "feature_id"])
        assert len(m) == 7 * 40
        np.testing.assert_allclose(m["train_ap_sign_agnostic"], m["train_univariate_auprc"], atol=1e-9)
        np.testing.assert_allclose(m["heldout_ap_sign_agnostic"], m["heldout_univariate_auprc"], atol=1e-9)
        np.testing.assert_allclose(m["heldout_lift_sign_agnostic"], m["heldout_univariate_lift"], atol=1e-9)
        # the oriented lift never exceeds the sign-agnostic one
        assert (m["heldout_lift_oriented"] <= m["heldout_lift_sign_agnostic"] + 1e-12).all()


@needs_exp
def test_regression_scratch_anchor(ifna_k40):
    _, tabs = ifna_k40
    for method, tab in tabs.items():
        a = runner.ANCHOR[method]
        assert sorted(tab.loc[tab["degenerate"], "factor_index"].unique().tolist()) == a["degenerate"]
        nd = tab.loc[~tab["degenerate"]]
        top = nd.loc[nd.groupby("factor_index")["enrichment"].idxmax(), "patient"].value_counts().to_dict()
        assert top == a["top_counts"]
        med = nd.groupby("patient")["enrichment"].median()
        for p, v in a["median_enrichment"].items():
            assert abs(med[p] - v) <= runner.ROUND_TOL
        for f, vals in a["enrichment"].items():
            sub = nd.loc[nd["factor_index"].eq(f)].set_index("patient")["enrichment"]
            for p, v in vals.items():
                assert abs(sub[p] - v) <= runner.ROUND_TOL
        for f, vals in a["shift_z"].items():
            sub = nd.loc[nd["factor_index"].eq(f)].set_index("patient")["shift_z_cohort"]
            for p, v in vals.items():
                assert abs(sub[p] - v) <= runner.ROUND_TOL


@needs_exp
def test_regression_d14_eta2(ifna_k40):
    d14_path = EXP / runner.D14_FOCUS_RELPATH
    if not d14_path.is_file():
        pytest.skip("D14 focus table absent")
    d14 = pd.read_csv(d14_path)
    _, tabs = ifna_k40
    for method, tab in tabs.items():
        fe = tab.groupby("factor_index")[["eta2_normal", "eta2_malignant", "degenerate"]].first()
        for stratum, col in (("normal", "eta2_normal"), ("cancer", "eta2_malignant")):
            d = d14.loc[d14["stage0_panel_id"].eq(runner.FOCUS_PANEL) & d14["stage1_method"].eq(method)
                        & d14["stratum"].eq(stratum)].set_index("factor_index")["eta2"]
            z = fe.join(d.rename("d14"), how="inner")
            z = z.loc[~z["degenerate"]]
            assert len(z) > 30
            np.testing.assert_allclose(z[col], z["d14"], atol=runner.D14_TOL)


@needs_exp
def test_saved_outputs_consistent():
    td = runner.out_dirs(EXP)["tables"]
    checks_path = td / "factor_patient_matrix_regression_checks.csv"
    if not checks_path.is_file():
        pytest.skip("D16 outputs not written yet")
    checks = pd.read_csv(checks_path)
    assert (checks["status"] == "pass").all(), checks.loc[checks["status"] != "pass"]
    manifest = json.loads((td / "factor_patient_matrix_manifest.json").read_text())
    rs = pd.read_csv(td / "factor_patient_matrix_representation_summary.csv")
    assert manifest["rows"]["representation_summary"] == len(rs)
    assert rs["hp_cell_share"].round(3).eq(0.082).all()  # P02+P03+P04 = 8.2% of cells
