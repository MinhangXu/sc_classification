"""Tests for the D18 factor-card module (scripts/mrd_stage0_2/workstreams/W5_factor_interpretation).

Synthetic tests build a tiny cohort with planted factors, so orientation, lift, attribution,
eta2, determinacy, matching and the bundle schema can be checked against hand-derivable
answers. Regression tests compare the real factor_cards_v1 bundle with the D8 / F6 tables and
are skipped when the experiment dir (or the bundle) is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W5_factor_interpretation" / "factor_cards.py"


def load_module():
    spec = importlib.util.spec_from_file_location("factor_cards", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fc = load_module()

PATIENTS = ["P01", "P02", "P03", "P04", "P05"]
SUPPORTED = ("P01", "P02", "P03", "P04")


# --------------------------------------------------------------------------------------
# synthetic fixture
# --------------------------------------------------------------------------------------


def make_cohort(seed: int = 0, n_per_patient: int = 300):
    rng = np.random.default_rng(seed)
    patients = np.repeat(PATIENTS, n_per_patient)
    labels = np.zeros(patients.size, dtype=int)
    for i, p in enumerate(PATIENTS):
        idx = np.flatnonzero(patients == p)
        labels[idx[: 25 + 10 * i]] = 1  # 25..65 malignant per patient
    cell_types = np.array(["HSC", "Mono", "T"])[rng.integers(0, 3, patients.size)]
    return fc.Cohort(patients, labels, SUPPORTED, cell_types=cell_types), rng


def make_rep(cohort, rng, method="pca", k=5):
    """Factor 1 carries the label (malignant high); factor 2 noise; factor 3 sign-flip probe;
    factor 4 patient-offset axis; factor 5 degenerate (zero)."""
    n = cohort.n_cells
    y = cohort.labels
    scores = rng.normal(size=(n, k))
    scores[:, 0] += 1.6 * y
    offsets = {p: o for p, o in zip(PATIENTS, [-2.0, -1.0, 0.0, 1.0, 2.0])}
    scores[:, 3] = np.array([offsets[p] for p in cohort.patients]) + 0.1 * rng.normal(size=n)
    scores[:, 4] = 0.0
    genes = ["G{:02d}".format(i) for i in range(12)]
    loadings = rng.normal(scale=0.1, size=(12, k))
    loadings[0, 0] = 0.9  # gene G00 rises with factor 1
    loadings[1, 0] = -0.8  # gene G01 falls with factor 1
    loadings[:, 4] = 0.0
    rows = []
    for p in PATIENTS:
        for j in range(k):
            coef = 0.0
            if j == 0 and p != "P01":  # P01 fold is all-zero
                coef = -0.05  # negative beta: the probe uses -factor1
            if j == 2 and p in ("P02", "P03"):
                coef = 0.01
            if j == 2 and p == "P04":
                coef = -0.01
            rows.append({"heldout_patient": p, "feature_id": "factor_{:03d}".format(j + 1), "coefficient": coef, "penalty": "l1", "C": 1e-4, "l1_ratio": np.nan})
    coefs = pd.DataFrame(rows)
    return fc.RepInput("panelX", method, k, scores, loadings, genes, coefs, {}, None, None, "synthetic", "panel X")


@pytest.fixture(scope="module")
def synthetic():
    cohort, rng = make_cohort()
    adt_counts = rng.poisson(20, size=(cohort.n_cells, 6)).astype(float)
    names = ["ADT-A", "ADT-B", "ADT-C", "ADT-D", "Isotype-X", "ADT-E"]
    cohort.adt = fc.build_adt_ranks(fc.clr_per_cell(adt_counts), names, cohort.patients, cohort.labels)
    rep = make_rep(cohort, rng)
    perm = [1, 0, 2, 3, 4]
    other_scores = rep.scores[:, perm] * np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    other_load = rep.loadings[:, perm] * np.array([1.0, -1.0, 1.0, 1.0, 1.0])
    other = fc.make_basis("panelX|dr|fa|5|42", "panelX", "fa", 5, other_scores, other_load)
    tables = fc.build_representation_tables(rep, cohort, top_genes=4, match_bases=[other], adt_top=3)
    return cohort, rep, tables


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def test_average_precision_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(5, 400))
        y = rng.random(n) < rng.uniform(0.05, 0.6)
        y[0] = True
        score = np.round(rng.normal(size=n), int(rng.integers(0, 3)))
        assert fc.average_precision(y, score) == pytest.approx(average_precision_score(y, score), abs=1e-12)


def test_average_precision_constant_score_is_prevalence():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    assert fc.average_precision(y, np.zeros(8)) == pytest.approx(y.mean())
    assert np.isnan(fc.average_precision(np.zeros(4), np.arange(4)))


def test_auroc_matches_sklearn():
    rng = np.random.default_rng(2)
    y = rng.random(300) < 0.2
    s = np.round(rng.normal(size=(300, 3)), 1)
    got = fc.auroc_columns(y, s)
    for j in range(3):
        assert got[j] == pytest.approx(roc_auc_score(y, s[:, j]), abs=1e-12)


def test_card_id_roundtrip_and_errors():
    cid = fc.card_id("single_geneset__x", "factosig_promax", 40, 7)
    assert cid == "single_geneset__x__factosig_promax__k40__f007"
    assert fc.parse_card_id(cid) == {"panel_id": "single_geneset__x", "method": "factosig_promax", "k": 40, "factor": 7}
    with pytest.raises(ValueError):
        fc.parse_card_id("panel__ica__k5__f001")


# --------------------------------------------------------------------------------------
# orientation, lift and attribution
# --------------------------------------------------------------------------------------


def test_orientation_statuses(synthetic):
    _, _, tables = synthetic
    idx = tables["index"].set_index("factor")
    assert idx.loc[1, "orientation_status"] == "oriented" and idx.loc[1, "orientation_sign"] == -1
    assert idx.loc[1, "n_active_folds"] == 3  # P01 fold is all-zero
    assert idx.loc[3, "orientation_status"] == "unoriented_sign_flip" and idx.loc[3, "orientation_sign"] == 0
    assert idx.loc[3, "sign_consistency"] == pytest.approx(2 / 3)
    assert idx.loc[2, "orientation_status"] == "unoriented_all_zero"
    assert np.isnan(idx.loc[2, "sign_consistency"])
    for f in (2, 3, 4, 5):
        assert "l1_selected_absent" in idx.loc[f, "caveat_flags"] or idx.loc[f, "n_active_folds"] > 0
    assert "has_all_zero_folds" in idx.loc[1, "caveat_flags"]
    assert idx.loc[1, "all_zero_folds"] == "P01"


def test_oriented_genes_follow_sign_beta(synthetic):
    _, _, tables = synthetic
    genes = tables["genes"]
    g1 = genes.loc[genes["card_id"] == fc.card_id("panelX", "pca", 5, 1)]
    # beta < 0, so the oriented loading of G00 (+0.9 raw) is negative -> normal pole
    g00 = g1.loc[g1["gene"] == "G00"].iloc[0]
    assert g00["oriented_loading"] == pytest.approx(-0.9)
    assert g00["pole"] == "normal"
    assert g1.loc[g1["gene"] == "G01", "pole"].iloc[0] == "malignant"
    assert set(g1["pole"]) == {"malignant", "normal"}
    g3 = genes.loc[genes["card_id"] == fc.card_id("panelX", "pca", 5, 3)]
    assert set(g3["pole"]) <= {"positive", "negative"}
    assert g3["oriented_loading"].isna().all()
    assert genes.loc[genes["card_id"] == fc.card_id("panelX", "pca", 5, 5)].empty  # degenerate


def test_univariate_lift_train_oriented(synthetic):
    _, _, tables = synthetic
    lift = tables["patient_lift"]
    f1 = lift.loc[lift["card_id"] == fc.card_id("panelX", "pca", 5, 1)]
    assert set(f1["patient"]) == set(SUPPORTED)
    assert (f1["train_orientation_sign"] == 1).all()  # malignant cells are high on factor 1
    assert np.allclose(f1["univariate_lift"], f1["univariate_lift_sign_agnostic"])
    assert (f1["univariate_lift"] > 0.1).all()
    assert (lift["univariate_lift_sign_agnostic"] >= lift["univariate_lift"] - 1e-12).all()
    assert np.allclose(f1["prevalence"], f1["n_malignant"] / f1["n_cells"])


def test_fitted_attribution(synthetic):
    _, _, tables = synthetic
    lift = tables["patient_lift"].set_index(["card_id", "patient"])
    c1 = fc.card_id("panelX", "pca", 5, 1)
    # P01: all-zero fold -> AP = prevalence, single-factor lift 0
    assert lift.loc[(c1, "P01"), "fitted_full_model_lift"] == pytest.approx(0.0, abs=1e-12)
    assert lift.loc[(c1, "P01"), "fitted_single_factor_lift"] == pytest.approx(0.0, abs=1e-12)
    # P02-P04: negative beta on a malignant-high factor -> the fitted single-factor lift is the
    # reversed-sign AP minus prevalence (negative), and equals the full model when alone
    for p in ("P02", "P03", "P04"):
        assert lift.loc[(c1, p), "fitted_single_factor_lift"] < 0
    folds = tables["fold_coefs"]
    f = folds.loc[(folds["card_id"] == c1) & (folds["heldout_patient"] == "P01")].iloc[0]
    assert bool(f["fold_all_zero"]) and not bool(f["is_nonzero"])
    assert len(folds) == 5 * 5  # 5 folds (incl. unsupported P05) x 5 factors


def test_single_active_factor_ablation_identity():
    cohort, rng = make_cohort(seed=3)
    rep = make_rep(cohort, rng)
    rep.coefs.loc[rep.coefs["feature_id"] == "factor_003", "coefficient"] = 0.0
    rep.coefs.loc[rep.coefs["feature_id"] == "factor_001", "coefficient"] = 0.07
    tables = fc.build_representation_tables(rep, cohort, top_genes=3)
    lift = tables["patient_lift"]
    row = lift.loc[(lift["card_id"] == fc.card_id("panelX", "pca", 5, 1)) & (lift["patient"] == "P02")].iloc[0]
    assert row["fitted_single_factor_lift"] == pytest.approx(row["fitted_full_model_lift"], abs=1e-12)
    assert row["fitted_ap_drop"] == pytest.approx(row["fitted_full_model_lift"], abs=1e-12)
    assert row["fitted_single_factor_lift"] == pytest.approx(row["univariate_lift"], abs=1e-12)


# --------------------------------------------------------------------------------------
# eta2, determinacy, structure
# --------------------------------------------------------------------------------------


def test_eta2_detects_patient_axis(synthetic):
    _, _, tables = synthetic
    idx = tables["index"].set_index("factor")
    assert idx.loc[4, "eta2_normal"] > 0.95
    assert idx.loc[2, "eta2_normal"] < 0.05


def test_eta2_by_label_exact():
    patients = np.array(["A"] * 300 + ["B"] * 300)
    labels = np.zeros(600, dtype=int)
    x = np.r_[np.zeros(300), np.ones(300)][:, None] + np.r_[np.tile([-1.0, 1.0], 150), np.tile([-1.0, 1.0], 150)][:, None]
    cohort = fc.Cohort(patients, labels, ("A", "B"))
    out = fc.eta2_by_label(x, cohort)
    # between SS = 600 * 0.25, total SS = 600 * (0.25 + 1)
    assert out["normal"][0] == pytest.approx(0.25 / 1.25)
    assert np.isnan(out["malignant"][0])


def test_determinacy_closed_form():
    lam = np.zeros((6, 2))
    lam[:3, 0] = 0.8
    lam[3:, 1] = 0.5
    psi = 1.0 - (lam ** 2).sum(axis=1)
    d = (lam ** 2 / psi[:, None]).sum(axis=0)
    rho = fc.factor_determinacy(lam, psi)
    assert np.allclose(rho, np.sqrt(d / (1 + d)))
    assert np.allclose(fc.psi_whitened_d(lam, psi), d)


def test_determinacy_approximates_fa_score_variance():
    from sklearn.decomposition import FactorAnalysis

    rng = np.random.default_rng(4)
    f = rng.normal(size=(4000, 2))
    lam = rng.normal(scale=0.6, size=(15, 2))
    x = f @ lam.T + rng.normal(scale=0.7, size=(4000, 15))
    x = (x - x.mean(0)) / x.std(0)
    fa = FactorAnalysis(n_components=2, random_state=0).fit(x)
    scores = fa.transform(x)
    rho = fc.factor_determinacy(fa.components_.T, fa.noise_variance_)
    assert np.allclose(rho ** 2, scores.var(axis=0), atol=0.02)


def test_structure_flags():
    sd_scores = np.c_[np.random.default_rng(5).normal(size=(500, 3)), np.zeros(500), 50 * np.random.default_rng(6).normal(size=500)]
    load = np.ones((4, 5))
    load[:, 3] = 0.0
    z, _, _, _ = fc.zscore_columns(sd_scores)
    st = fc.stage1_structure("factosig_promax", sd_scores, load, {}, z.T @ z / z.shape[0])
    assert st.effective_k == 4
    assert st.zero_variance.tolist() == [False, False, False, True, False]
    assert st.scale_outlier.tolist() == [False, False, False, False, True]
    assert st.determinacy_method == "not_computed_oblique_phi_unsaved"
    gaps = fc.relative_eigengap(np.array([4.0, 2.0, 1.9, 1.0]))
    assert gaps[1] == pytest.approx(0.05) and gaps[2] == pytest.approx(0.1 / 1.9)


def test_scale_outlier_is_a_caveat_not_degenerate():
    """degenerate_flag = zero variance only; a blown-up score scale stays the scale_outlier caveat."""
    cohort, rng = make_cohort(seed=3)
    rep = make_rep(cohort, rng, method="factosig_promax")
    rep.scores[:, 1] *= 50.0  # factor 2: score SD 50x the representation median
    tables = fc.build_representation_tables(rep, cohort, top_genes=4, match_bases=[], adt_top=3)
    idx = tables["index"].set_index("factor")
    assert not bool(idx.loc[2, "degenerate_flag"])
    assert "scale_outlier" in idx.loc[2, "caveat_flags"].split(";")
    assert bool(idx.loc[5, "degenerate_flag"]) and "zero_variance" in idx.loc[5, "caveat_flags"].split(";")
    assert int(idx.loc[2, "effective_k"]) == 4


# --------------------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------------------


def test_matching_recovers_permutation_and_sign(synthetic):
    _, _, tables = synthetic
    m = tables["matches"].set_index("factor")
    assert m.loc[1, "relation"] == "hungarian_same_k"
    assert m.loc[1, "matched_factor"] == 2 and m.loc[1, "score_r"] == pytest.approx(-1.0)
    assert m.loc[1, "tucker_phi_aligned"] == pytest.approx(1.0)
    assert m.loc[2, "matched_factor"] == 1 and m.loc[2, "score_r"] == pytest.approx(1.0)
    assert np.allclose(m["r2_many_to_one"], 1.0)
    assert 5 not in m.index  # degenerate factor is not matched
    idx = tables["index"].set_index("factor")
    assert idx.loc[1, "matched_cards"] == fc.card_id("panelX", "fa", 5, 2)


def test_many_to_one_split_axis():
    rng = np.random.default_rng(7)
    b = rng.normal(size=(3000, 2))
    a_scores = np.c_[(b[:, 0] + b[:, 1]) / np.sqrt(2), rng.normal(size=3000)]
    a = fc.make_basis("a", "p", "pca", 2, a_scores, np.eye(2))
    bb = fc.make_basis("b", "p", "fa", 3, np.c_[b, rng.normal(size=3000)], np.eye(2, 3))
    m = fc.match_factors(a, bb).set_index("factor")
    assert m.loc[1, "relation"] == "best_abs_r"
    assert abs(m.loc[1, "best_r"]) == pytest.approx(1 / np.sqrt(2), abs=0.03)
    assert m.loc[1, "r2_many_to_one"] == pytest.approx(1.0, abs=1e-6)
    assert m.loc[1, "angle_deg"] == pytest.approx(0.0, abs=0.1)
    sub = fc.subspace_summary(a, bb, selected=[1])
    assert sub["selected_cc_min"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# ADT
# --------------------------------------------------------------------------------------


def test_clr_and_within_label_spearman():
    rng = np.random.default_rng(8)
    counts = rng.poisson(30, size=(400, 4)).astype(float)
    clr = fc.clr_per_cell(counts)
    assert np.allclose(clr.sum(axis=1), 0.0)
    patients = np.repeat(["A", "B"], 200)
    labels = np.r_[np.zeros(150), np.ones(50), np.zeros(150), np.ones(50)].astype(int)
    adt = fc.build_adt_ranks(clr, ["ADT-a", "ADT-b", "ADT-c", "Isotype-d"], patients, labels, {"normal": 100, "malignant": 20})
    assert adt.is_isotype.tolist() == [False, False, False, True]
    score = clr[:, [0]] * 3 + 1  # monotone in marker a
    rho = fc.adt_spearman(score, adt)
    assert rho["normal"]["pooled"][0, 0] == pytest.approx(1.0)
    assert rho["malignant"]["stratified"][0, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# bundle schema
# --------------------------------------------------------------------------------------


def write_bundle(tables, out: Path) -> Path:
    fc.write_tables(tables, out)
    fc.render_all_cards(tables, out / "cards")
    manifest = {
        "schema_version": fc.SCHEMA_VERSION,
        "slug": fc.SLUG,
        "inputs": {"stage1_dr": {"path": "stage1_dr/panel_id=panelX"}},
        "git_state": {"head": None, "dirty": None},
        "created_by": "tests",
        "created_utc": "2026-10-08T00:00:00+00:00",
        "parameters": {},
        "counts": {},
    }
    (out / fc.MANIFEST_NAME).write_text(json.dumps(manifest))
    return out


def test_synthetic_bundle_validates(synthetic, tmp_path):
    _, _, tables = synthetic
    out = write_bundle(tables, tmp_path / "bundle")
    assert fc.validate_bundle(out) == []
    md = (out / "cards" / "{}.md".format(fc.card_id("panelX", "pca", 5, 1))).read_text()
    for heading in ("## Orientation", "## Selection, beta and held-out lift", "## Genes from both poles", "## Cross-method and cross-K matches", "## Surface protein", "## Caveats"):
        assert heading in md
    assert "0 (all-zero fold)" in md
    for name, cols in fc.BUNDLE_FILES.items():
        frame = pd.read_csv(out / name)
        assert list(cols) == [c for c in frame.columns if c in cols]


def test_validation_catches_schema_breaks(synthetic, tmp_path):
    _, _, tables = synthetic
    out = write_bundle(tables, tmp_path / "broken")
    idx = pd.read_csv(out / "factor_cards_index.csv")
    idx.drop(columns=["determinacy"]).to_csv(out / "factor_cards_index.csv", index=False)
    genes = pd.read_csv(out / "factor_cards_genes.csv")
    raw = genes["pole"].isin(["positive", "negative"])
    genes.loc[raw, "oriented_loading"] = 1.0
    genes.to_csv(out / "factor_cards_genes.csv", index=False)
    lift = pd.read_csv(out / "factor_cards_patient_lift.csv")
    lift.loc[0, "card_id"] = "not_a_card"
    lift.to_csv(out / "factor_cards_patient_lift.csv", index=False)
    manifest = json.loads((out / fc.MANIFEST_NAME).read_text())
    manifest["schema_version"] = 2
    (out / fc.MANIFEST_NAME).write_text(json.dumps(manifest))
    errors = " | ".join(fc.validate_bundle(out))
    assert "missing columns ['determinacy']" in errors
    assert "unoriented genes carry oriented_loading" in errors
    assert "card ids not in the index" in errors
    assert "schema_version 2" in errors


def test_schema_markdown_lists_required_columns():
    text = fc.bundle_schema_markdown()
    for cols in fc.BUNDLE_FILES.values():
        for col in cols:
            assert col in text
    for flag in fc.CAVEAT_TEXT:
        assert "`{}`".format(flag) in text


# --------------------------------------------------------------------------------------
# regression against the real experiment (skipped without EXP / bundle)
# --------------------------------------------------------------------------------------


def _experiment():
    sys.path.insert(0, str(REPO / "src"))
    from sc_classification.utils.paths import experiment_dir

    exp = experiment_dir(must_exist=False)
    return exp if exp.is_dir() else None


def test_d18_runner_refuses_partial_shortlist_without_out_root():
    """--shortlist smoke|ifna would rewrite the canonical bundle with a partial shortlist."""
    path = MODULE_PATH.parent / "run_d18_factor_cards.py"
    spec = importlib.util.spec_from_file_location("run_d18_factor_cards", path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    for mode in ("smoke", "ifna"):
        with pytest.raises(SystemExit, match="--out-root"):
            runner.main(["--shortlist", mode])


EXP = _experiment()
BUNDLE = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / fc.SLUG if EXP is not None else None
needs_bundle = pytest.mark.skipif(BUNDLE is None or not (BUNDLE / fc.MANIFEST_NAME).is_file(), reason="experiment dir or factor_cards_v1 bundle not present")
IFNA = "single_geneset__hallmark_interferon_alpha_response"


@needs_bundle
def test_real_bundle_validates_and_checks_pass():
    assert fc.validate_bundle(BUNDLE, check_cards=False) == []
    checks = pd.read_csv(BUNDLE / "factor_cards_regression_checks.csv")
    assert checks["passed"].all(), checks.loc[~checks["passed"], "check_id"].tolist()
    manifest = json.loads((BUNDLE / fc.MANIFEST_NAME).read_text())
    assert manifest["schema_version"] == 1
    for info in manifest["inputs"].values():
        assert not str(info.get("path", "")).startswith("/")


@needs_bundle
def test_real_pc8_fa7_reproduce_d8_and_f6():
    lift = pd.read_csv(BUNDLE / "factor_cards_patient_lift.csv")
    mf = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / "meeting_followup_v1"
    fb = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / "feedback_deep_dive_20260917"
    abl = pd.read_csv(mf / "interferon_alpha_pca40_fa40_factor_ablation.csv")
    f6 = pd.read_csv(fb / "f6_ifna_univariate_factor_ap_by_patient.csv")
    for method, factor in (("pca", 8), ("fa", 7)):
        cid = fc.card_id(IFNA, method, 40, factor)
        mine = lift.loc[lift["card_id"] == cid].set_index("patient")
        d8 = abl.loc[(abl["stage1_method"] == method) & (abl["feature_id"] == "factor_{:03d}".format(factor))].set_index("heldout_patient")
        f = f6.loc[(f6["stage1_method"] == method) & (f6["feature_id"] == "factor_{:03d}".format(factor))].set_index("heldout_patient")
        assert np.allclose(mine.loc[d8.index, "fitted_coef"], d8["coefficient"], atol=1e-9)
        assert np.allclose(mine.loc[d8.index, "fitted_ap_drop"], d8["ap_drop_when_ablated"], atol=1e-6)
        assert np.allclose(mine.loc[d8.index, "fitted_single_factor_lift"], d8["single_factor_lift"], atol=1e-6)
        assert np.allclose(mine.loc[f.index, "univariate_lift_sign_agnostic"], f["heldout_univariate_lift"], atol=1e-6)
        assert np.allclose(mine.loc[f.index, "univariate_lift"], f["heldout_univariate_lift"], atol=1e-6)
    idx = pd.read_csv(BUNDLE / "factor_cards_index.csv").set_index("card_id")
    assert idx.loc[fc.card_id(IFNA, "pca", 40, 8), "orientation_sign"] == -1
    assert idx.loc[fc.card_id(IFNA, "fa", 40, 7), "orientation_sign"] == 1
    assert idx.loc[fc.card_id(IFNA, "fa", 40, 7), "n_active_folds"] == 2
