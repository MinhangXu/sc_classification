"""Tests for the W2 Stage 1 refit module (D17, tasks W2.1 / W2.2).

Module: scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/stage1_refit.py, loaded by
file path with importlib. All tests use small synthetic matrices except the last three, which
re-check the canonical experiment read-only (refit reproduces the stored transductive IFN-alpha
bases; the stored per-patient L1 lifts are reproduced) and are skipped when it is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA, FactorAnalysis

REPO = Path(__file__).resolve().parents[1]
WORKSTREAMS = REPO / "scripts" / "mrd_stage0_2" / "workstreams"
MODULE_FILE = WORKSTREAMS / "W2_stage1_basis_variants" / "stage1_refit.py"
CONTRACT_FILE = WORKSTREAMS / "W3_dr_value_regularization" / "regularization_contract.py"
PATHS_FILE = REPO / "src" / "sc_classification" / "utils" / "paths.py"


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s1 = _load("stage1_refit", MODULE_FILE)


def _toy(n=600, p=12, k_true=3, seed=0, patients=("A", "B", "C"), sizes=(300, 200, 100)):
    """Low-rank-plus-noise matrix with a per-patient mean shift and unequal patient sizes."""
    rng = np.random.default_rng(seed)
    assert sum(sizes) == n
    W = rng.normal(size=(k_true, p))
    Z = rng.normal(size=(n, k_true))
    X = Z @ W + 0.5 * rng.normal(size=(n, p))
    pat = np.repeat(np.asarray(patients), sizes)
    for i, name in enumerate(patients):
        X[pat == name] += 0.3 * i
    X[:, 1] *= 4.0  # unequal gene scales
    return X, pat


def _abs_corr_cols(A, B):
    return np.array([abs(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(A.shape[1])])


# --------------------------------------------------------------------------------------
# standardization, weights and scopes
# --------------------------------------------------------------------------------------


def test_standardizer_population_sd_and_zero_variance_gene():
    X, _ = _toy()
    X[:, 3] = 7.0  # constant gene
    mean, scale = s1.fit_standardizer(X)
    np.testing.assert_allclose(mean, X.mean(0))
    assert scale[3] == 1.0
    Xs = s1.apply_standardizer(X, mean, scale)
    np.testing.assert_allclose(Xs[:, 3], 0.0)
    keep = np.arange(X.shape[1]) != 3
    np.testing.assert_allclose(Xs[:, keep].std(0), 1.0, atol=1e-12)


def test_weighted_standardizer_matches_row_duplication():
    X, pat = _toy()
    w = s1.patient_balanced_weights(pat)
    # duplicating rows to equal patient sizes (300 each) gives the same moments
    reps = np.where(pat == "A", 2, np.where(pat == "B", 3, 6))
    Xdup = np.repeat(X, reps, axis=0)
    m1, s1_ = s1.fit_standardizer(X, w)
    m2, s2_ = s1.fit_standardizer(Xdup)
    np.testing.assert_allclose(m1, m2, atol=1e-12)
    np.testing.assert_allclose(s1_, s2_, atol=1e-12)


def test_patient_balanced_weights_sum_to_one_per_patient_share():
    _, pat = _toy()
    w = s1.patient_balanced_weights(pat)
    assert np.isclose(w.sum(), 1.0)
    for name in np.unique(pat):
        assert np.isclose(w[pat == name].sum(), 1.0 / 3.0)
    rows = pat != "C"
    w2 = s1.patient_balanced_weights(pat, rows)
    assert w2.size == rows.sum() and np.isclose(w2[pat[rows] == "A"].sum(), 0.5)


def test_stratified_subsample_equal_n_deterministic_no_duplicates():
    _, pat = _toy()
    idx = s1.stratified_subsample(pat, seed=3)
    assert np.array_equal(idx, np.unique(idx))
    names, counts = np.unique(pat[idx], return_counts=True)
    assert counts.tolist() == [100, 100, 100]
    assert np.array_equal(idx, s1.stratified_subsample(pat, seed=3))
    assert not np.array_equal(idx, s1.stratified_subsample(pat, seed=4))
    with pytest.raises(ValueError):
        s1.stratified_subsample(pat, n_per_patient=101)


def test_fit_scope_mask_and_random_drop():
    X, pat = _toy()
    y = (np.arange(pat.size) % 10 == 0).astype(int)
    m = s1.fit_scope_mask(pat, exclude_patients=["B"])
    assert not m[pat == "B"].any() and m[pat != "B"].all()
    m2 = s1.fit_scope_mask(pat, labels=y, label_value=0)
    assert np.array_equal(m2, y == 0)
    keep = s1.random_drop_mask(pat.size, 50, seed=1)
    assert keep.sum() == pat.size - 50
    keep_b = s1.random_drop_mask(pat.size, 20, seed=1, rows=pat == "B")
    assert keep_b[pat != "B"].all() and (~keep_b).sum() == 20


# --------------------------------------------------------------------------------------
# fit / project / persist
# --------------------------------------------------------------------------------------


def test_pca_fit_project_matches_sklearn_on_fit_and_new_rows():
    X, pat = _toy()
    rows = pat != "C"
    basis = s1.fit_basis(X, "pca", 4, rows=rows, pca_solver="full")
    mean, scale = X[rows].mean(0), X[rows].std(0)
    ref = PCA(n_components=4, svd_solver="full").fit((X[rows] - mean) / scale)
    expected = ref.transform((X - mean) / scale)
    np.testing.assert_allclose(basis.transform(X), expected, atol=1e-10)
    # held-out cells were projected, not fitted: the fit-cell scores are centred, theirs need not be
    assert np.allclose(basis.transform(X[rows]).mean(0), 0.0, atol=1e-10)
    assert basis.fit_info["n_fit_cells"] == int(rows.sum())


def test_fa_fit_project_matches_sklearn_transform():
    X, pat = _toy()
    rows = pat != "A"
    basis = s1.fit_basis(X, "fa", 3, rows=rows, fa_svd_method="lapack")
    mean, scale = X[rows].mean(0), X[rows].std(0)
    ref = FactorAnalysis(n_components=3, svd_method="lapack", random_state=42).fit((X[rows] - mean) / scale)
    np.testing.assert_allclose(basis.transform(X), ref.transform((X - mean) / scale), atol=1e-10)
    np.testing.assert_allclose(basis.noise_variance, ref.noise_variance_)


@pytest.mark.parametrize("method", ["pca", "fa"])
def test_persist_round_trip(tmp_path, method):
    X, pat = _toy()
    genes = [f"G{j}" for j in range(X.shape[1])]
    basis = s1.fit_basis(X, method, 3, rows=pat != "B", gene_names=genes, fit_info={"arm": "inductive", "heldout": "B"})
    out = basis.save(tmp_path / "b")
    assert s1.Stage1Basis.is_saved(out)
    back = s1.Stage1Basis.load(out)
    assert back.gene_names == genes and back.method == method and back.k == 3
    assert back.fit_info["arm"] == "inductive"
    np.testing.assert_array_equal(back.transform(X), basis.transform(X))
    with pytest.raises(ValueError):
        back.transform(X, gene_names=genes[::-1])


def test_corrupt_or_partial_basis_is_not_saved(tmp_path):
    X, _ = _toy()
    out = s1.fit_basis(X, "pca", 2).save(tmp_path / "b")
    (out / "basis.json").write_text("", encoding="utf-8")
    assert not s1.Stage1Basis.is_saved(out)
    assert not s1.Stage1Basis.is_saved(tmp_path / "missing")


def test_uniform_weights_give_unweighted_pca():
    X, _ = _toy()
    n = X.shape[0]
    a = s1.fit_basis(X, "pca", 4, pca_solver="full")
    b = s1.fit_basis(X, "pca", 4, weights=np.ones(n))
    # weighted fits report population eigenvalues; sklearn uses the n - 1 convention
    np.testing.assert_allclose(b.explained_variance * n / (n - 1), a.explained_variance, rtol=1e-9)
    assert _abs_corr_cols(a.transform(X), b.transform(X)).min() > 1 - 1e-9


def test_patient_weighted_pca_equals_pca_on_duplicated_rows():
    X, pat = _toy()
    w = s1.patient_balanced_weights(pat)
    reps = np.where(pat == "A", 2, np.where(pat == "B", 3, 6))
    Xdup = np.repeat(X, reps, axis=0)
    weighted = s1.fit_basis(X, "pca", 4, weights=w)
    dup = s1.fit_basis(Xdup, "pca", 4, pca_solver="full")
    nd = Xdup.shape[0]
    np.testing.assert_allclose(weighted.explained_variance * nd / (nd - 1), dup.explained_variance, rtol=1e-8)
    assert _abs_corr_cols(weighted.transform(X), dup.transform(X)).min() > 1 - 1e-8
    unweighted = s1.fit_basis(X, "pca", 4, pca_solver="full")
    assert not np.allclose(weighted.explained_variance, unweighted.explained_variance)


def test_pseudo_data_has_exact_covariance():
    X, _ = _toy()
    C = s1.weighted_covariance(s1.apply_standardizer(X, *s1.fit_standardizer(X)))
    Y = s1.pseudo_data_from_covariance(C)
    np.testing.assert_allclose(Y.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cov(Y.T, bias=True), C, atol=1e-10)


@pytest.mark.parametrize("svd_method", ["lapack", "randomized"])
def test_fa_from_covariance_equals_fa_on_data(svd_method):
    """Same iterates as a fit on the rows once tol is rescaled to the number of rows."""
    X, _ = _toy()
    Xs = s1.apply_standardizer(X, *s1.fit_standardizer(X))
    direct = FactorAnalysis(n_components=3, svd_method=svd_method, random_state=0).fit(Xs)
    via_cov = s1.fa_from_covariance(s1.weighted_covariance(Xs), 3, seed=0, svd_method=svd_method, n_samples=Xs.shape[0])
    assert via_cov.n_iter_ == direct.n_iter_
    np.testing.assert_allclose(via_cov.noise_variance_, direct.noise_variance_, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.abs(via_cov.components_), np.abs(direct.components_), rtol=1e-6, atol=1e-9)
    assert _abs_corr_cols(direct.transform(Xs), via_cov.transform(Xs)).min() > 1 - 1e-9


def test_uniform_weights_give_unweighted_fa():
    X, _ = _toy()
    a = s1.fit_basis(X, "fa", 3)
    b = s1.fit_basis(X, "fa", 3, weights=np.ones(X.shape[0]))
    assert b.fit_info["n_iter"] == a.fit_info["n_iter"]
    np.testing.assert_allclose(b.noise_variance, a.noise_variance, rtol=1e-8, atol=1e-10)
    assert _abs_corr_cols(a.transform(X), b.transform(X)).min() > 1 - 1e-9


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------


def test_hungarian_recovers_permutation_and_sign():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 5))
    perm = np.array([3, 0, 4, 1, 2])
    B = A[:, perm] * np.array([1, -1, 1, -1, 1])
    match = s1.hungarian_match(s1.score_correlation(A, B))
    for ref, (var, r) in match.items():
        assert perm[var] == ref and abs(abs(r) - 1) < 1e-12
    assert match[0][1] < 0  # column 1 of B (= A[:, 0]) was sign-flipped


def test_many_to_one_r2_is_one_for_a_rotated_basis_and_zero_column_is_safe():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(400, 4))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    B = np.column_stack([A @ Q, np.zeros(400)])
    assert s1.many_to_one_r2(A[:, 2], B) > 1 - 1e-12
    R = s1.score_correlation(A, B)
    assert np.all(R[:, -1] == 0.0)
    assert s1.many_to_one_r2(rng.normal(size=400), A) < 0.05


def test_principal_angles_and_containment():
    e = np.eye(6)
    assert np.allclose(s1.principal_angles_deg(e[:, :2], e[:, [1, 0]]), 0.0, atol=1e-6)
    assert np.allclose(s1.principal_angles_deg(e[:, :2], e[:, 2:4]), 90.0)
    assert np.isclose(s1.subspace_containment(e[:, 0] + e[:, 3], e[:, :2]), np.sqrt(0.5))
    assert np.isclose(s1.subspace_containment(e[:, 1], np.column_stack([e[:, :2], np.zeros(6)])), 1.0)


def test_align_scores_identity_and_leave_out():
    X, pat = _toy()
    full = s1.fit_basis(X, "pca", 4, pca_solver="full")
    out = s1.align_scores(
        full.transform(X), full.transform(X), 1,
        ref_loadings=full.loadings_common(), var_loadings=full.loadings_common(),
        ref_order=full.factor_variance_order(), var_order=full.factor_variance_order(), top_m=2,
    )
    assert out["matched_factor"] == 1 and out["matched_abs_r"] > 1 - 1e-12
    assert out["angle_allK_max_deg"] < 1e-4 and out["ref_axis_containment_allK"] > 1 - 1e-12
    assert out["matched_factor_importance_rank"] == 2
    drop = s1.fit_basis(X, "pca", 4, rows=pat != "C", pca_solver="full")
    out2 = s1.align_scores(full.transform(X), drop.transform(X), 0, rows=pat != "C",
                           ref_loadings=full.loadings_common(), var_loadings=drop.loadings_common())
    assert 0.0 <= out2["angle_allK_mean_cos2"] <= 1.0 and out2["matched_abs_r"] > 0.9


# --------------------------------------------------------------------------------------
# regression against the canonical experiment (read-only; skipped when absent)
# --------------------------------------------------------------------------------------

IFNA = "single_geneset__hallmark_interferon_alpha_response"


def _exp():
    try:
        paths = _load("sc_classification_utils_paths_for_tests", PATHS_FILE)
        exp = paths.experiment_dir(must_exist=False)
    except Exception:
        return None
    feats = exp / "stage1_direct_gene" / "old34_direct_gene_all_v1" / f"panel_id={IFNA}" / "transform=zscore" / "features.npz"
    return exp if feats.is_file() else None


EXP = _exp()
needs_exp = pytest.mark.skipif(EXP is None, reason="canonical experiment dir not available")


def _features():
    import scipy.sparse as sp

    d = EXP / "stage1_direct_gene" / "old34_direct_gene_all_v1" / f"panel_id={IFNA}" / "transform=zscore"
    return sp.load_npz(d / "features.npz").toarray().astype(np.float64)


def _stored(method):
    return np.load(EXP / "stage1_dr" / f"panel_id={IFNA}" / f"method={method}" / "k=40" / "seed=42" / "scores.npy")


@needs_exp
def test_transductive_refit_reproduces_stored_ifna_bases():
    X = _features()
    assert X.shape == (60118, 92)
    for method, ref in (("pca", 7), ("fa", 6)):  # PC8, FA7 (0-based)
        Z = s1.fit_basis(X, method, 40).transform(X)
        out = s1.align_scores(_stored(method), Z, ref)
        assert out["matched_factor"] == ref
        # floor set by re-standardizing on all cells plus sklearn's randomized solvers
        assert out["matched_abs_r"] > 0.998
        assert out["many_to_one_r2"] > 0.995


@needs_exp
def test_stored_ifna_pc8_l1_lift_reproduced_for_p02():
    import pandas as pd

    rc = _load("regularization_contract", CONTRACT_FILE)
    meta = pd.read_csv(EXP / "stage2_supervised" / "multiobjective" / "discovery_full_cohort" / "cell_metadata.csv.gz",
                       usecols=["y_true", "patient"])
    y = meta["y_true"].to_numpy().astype(int)
    pat = meta["patient"].astype(str).to_numpy()
    S = _stored("pca")
    train = pat != "P02"
    est = rc.fit_probe(S[train], y[train], penalty="l1", C=1e-4)
    _, _, lift = rc.heldout_lift(rc.probe_scores(est, S[~train]), y[~train])
    assert np.flatnonzero(np.abs(rc.probe_coef(est)) > rc.COEF_NONZERO_TOL).tolist() == [7]
    assert abs(100 * lift - 15.94) < 0.01  # stored sharedness_lopo L1 C=1e-4 lift for P02 (pp)


@needs_exp
def test_inductive_fit_excludes_heldout_patient():
    import pandas as pd

    meta = pd.read_csv(EXP / "stage2_supervised" / "multiobjective" / "discovery_full_cohort" / "cell_metadata.csv.gz",
                       usecols=["patient"])
    pat = meta["patient"].astype(str).to_numpy()
    X = _features()
    basis = s1.fit_basis(X, "pca", 10, rows=s1.fit_scope_mask(pat, exclude_patients=["P02"]))
    assert basis.fit_info["n_fit_cells"] == int((pat != "P02").sum())
    np.testing.assert_allclose(basis.gene_mean, X[pat != "P02"].mean(0), atol=1e-10)


@needs_exp
def test_stored_pca_scores_are_randomized_svd_us_not_exact_projection():
    """Provenance (D17): stored PCA scores are sklearn's randomized-SVD U * S, which differ from the
    projection onto the stored loadings; FA scores are exact projections."""
    X = _features()
    d = EXP / "stage1_dr" / f"panel_id={IFNA}" / "method=pca" / "k=10" / "seed=42"
    S, L = np.load(d / "scores.npy").astype(np.float64), np.load(d / "loadings.npy").astype(np.float64)
    proj = (X - X.mean(0)) @ L
    r_pc8 = abs(np.corrcoef(S[:, 7], proj[:, 7])[0, 1])
    assert 0.98 < r_pc8 < 0.999
    basis = s1.fit_basis(X, "pca", 10)
    assert abs(np.corrcoef(basis.transform(X)[:, 7], proj[:, 7])[0, 1]) > 0.9999


@needs_exp
def test_d17_anchor_artifact_reproduces_stored_lifts():
    import pandas as pd

    path = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / "basis_variant_pilot_ifna_v1" / "d17_anchor_check.csv"
    if not path.is_file():
        pytest.skip("D17 artifact not present")
    table = pd.read_csv(path)
    assert bool(table["pass"].all())
    k40 = table.loc[table["effective_k"].eq(40)]
    assert len(k40) == 28 and bool(k40["within_strict_tol"].all())


def test_label_matched_drop_mask_matches_both_counts():
    y = np.r_[np.ones(50, dtype=int), np.zeros(450, dtype=int)]
    keep = s1.label_matched_drop_mask(y, 100, 30, seed=3)
    assert keep.size == 500 and int((~keep).sum()) == 100 and int(y[~keep].sum()) == 30
    assert np.array_equal(keep, s1.label_matched_drop_mask(y, 100, 30, seed=3))
    assert not np.array_equal(keep, s1.label_matched_drop_mask(y, 100, 30, seed=4))
    with pytest.raises(ValueError):
        s1.label_matched_drop_mask(y, 100, 60)
    with pytest.raises(ValueError):
        s1.label_matched_drop_mask(y, 20, 30)


def test_d17_runner_refuses_smoke_without_out_root():
    runner = _load("run_d17_basis_variant_pilot", WORKSTREAMS / "W2_stage1_basis_variants" / "run_d17_basis_variant_pilot.py")
    with pytest.raises(SystemExit, match="--out-root"):
        runner.main(["--smoke", "--parts", "report"])

