"""Tests for the W3 regularization contract (D19, task W3.1).

Module: scripts/mrd_stage0_2/workstreams/W3_dr_value_regularization/regularization_contract.py,
loaded by file path with importlib. All tests use small synthetic data except the last two,
which re-check the IFN-alpha anchor against the canonical experiment read-only and are
skipped when that directory is absent.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
MODULE_FILE = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W3_dr_value_regularization" / "regularization_contract.py"
PATHS_FILE = REPO / "src" / "sc_classification" / "utils" / "paths.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Loaded by file path under its real name, with its directory on sys.path, so joblib worker
# processes (tune_c_inner_lopo n_jobs > 1) can re-import it the way W2/W4/W6 code does.
if str(MODULE_FILE.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_FILE.parent))
rc = _load("regularization_contract", MODULE_FILE)


def _synthetic(n=3000, k=6, prevalence=0.08, shifts=(0.35, -0.2, 0.05), seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    Z = rng.normal(size=(n, k))
    for j, shift in enumerate(shifts):
        Z[:, j] += shift * y
    return Z, y


def _gap(Zs, y):
    return Zs[y == 1].mean(0) - Zs[y == 0].mean(0)


def _cohort(seed=0, n_per=400, malignant=(60, 40, 30, 25, 3, 0)):
    """Six synthetic patients; the last two are below the support rule."""
    rng = np.random.default_rng(seed)
    Z, y, pats = [], [], []
    for i, n_mal in enumerate(malignant):
        yi = np.zeros(n_per, dtype=int)
        yi[:n_mal] = 1
        zi = rng.normal(size=(n_per, 5))
        zi[:, 0] += 0.9 * yi + 0.2 * i
        zi[:, 1] -= 0.5 * yi
        Z.append(zi)
        y.append(yi)
        pats += [f"Q{i + 1:02d}"] * n_per
    return np.vstack(Z), np.concatenate(y), np.asarray(pats)


# --------------------------------------------------------------------------------------
# grid and threshold formula
# --------------------------------------------------------------------------------------


def test_default_c_grid_matches_runner():
    grid = rc.DEFAULT_C_GRID
    assert len(grid) == 17
    assert grid[0] == pytest.approx(1e4) and grid[-1] == pytest.approx(1e-4)
    assert rc.C_GRID_LOWER_BOUND == min(grid)
    assert np.allclose(np.diff(np.log10(grid)), -0.5)
    expected = [float(1.0 / a) for a in np.logspace(-4.0, 4.0, 17)]  # runner c_grid_from_args
    assert list(grid) == expected
    assert rc.c_at_grid_lower_bound(1e-4) and not rc.c_at_grid_lower_bound(10 ** -3.5)


def test_threshold_formula():
    assert rc.l1_entry_threshold(1e-4, 50_000) == pytest.approx(0.8)
    assert rc.l1_entry_threshold(1e-4, 58_656) == pytest.approx(4.0 / (1e-4 * 58_656))
    # elastic net: 4 * l1_ratio / (C n), so a smaller l1_ratio lowers the bar
    assert rc.l1_entry_threshold(4.64e-4, 10_000, l1_ratio=0.5) == pytest.approx(4 * 0.5 / (4.64e-4 * 10_000))
    assert rc.l1_entry_threshold(1.0, 100, l1_ratio=0.1) < rc.l1_entry_threshold(1.0, 100)
    with pytest.raises(ValueError):
        rc.l1_entry_threshold(0.0, 10)


def test_margins_equal_gap_over_threshold():
    Z, y = _synthetic()
    Zs, _ = rc.standardize_train(Z)
    C = 2e-3
    expected = np.abs(_gap(Zs, y)) / rc.l1_entry_threshold(C, y.size)
    assert np.allclose(rc.l1_entry_margins(Z, y, C), expected)
    assert np.allclose(rc.l1_entry_margins(Zs, y, C, standardize=False), expected)
    en = rc.l1_entry_margins(Z, y, C, l1_ratio=0.5)
    assert np.allclose(en, expected * 2.0)
    assert np.allclose(rc.l1_entry_c(Z, y), C / rc.l1_entry_margins(Z, y, C))


def test_c_scaled_to_n_keeps_threshold_fixed():
    C_ref, n_ref = 1e-4, 58_000
    for n in (500, 5_000, 25_000):
        C_n = rc.c_scaled_to_n(C_ref, n_ref, n)
        assert C_n == pytest.approx(C_ref * n_ref / n)
        assert rc.l1_entry_threshold(C_n, n) == pytest.approx(rc.l1_entry_threshold(C_ref, n_ref))


def test_standardize_matches_runner_and_keeps_constant_columns_zero():
    Z, _ = _synthetic(n=200)
    Z[:, 3] = 7.0  # degenerate factor, like IFN-alpha FA40 factors 37-40
    Z[5, 2] = np.nan
    Zs, Ze = rc.standardize_train(Z, Z[:10])
    assert np.allclose(Zs[:, 3], 0.0)
    assert np.allclose(Zs.mean(0), 0.0, atol=1e-12)
    assert np.allclose(Ze, Zs[:10])


# --------------------------------------------------------------------------------------
# entry rule against real solvers
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["liblinear", "saga"])
@pytest.mark.parametrize("mult,expect_nonzero", [(0.97, False), (1.03, True)])
def test_margin_above_one_matches_nonzero_l1_coefficient(solver, mult, expect_nonzero):
    Z, y = _synthetic()
    Zs, _ = rc.standardize_train(Z)
    c_entry = float(np.min(rc.l1_entry_c(Zs, y, standardize=False)))
    C = c_entry * mult
    margins = rc.l1_entry_margins(Zs, y, C, standardize=False)
    assert (margins.max() > 1.0) == expect_nonzero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(penalty="l1", C=C, solver=solver, class_weight="balanced", tol=1e-10, max_iter=50_000, random_state=0)
        model.fit(Zs, y)
    nonzero = np.flatnonzero(np.abs(model.coef_.ravel()) > 1e-10)
    if expect_nonzero:
        assert nonzero.tolist() == [int(np.argmax(margins))]
    else:
        assert nonzero.size == 0


@pytest.mark.parametrize("ratio", [0.5, 0.1])
def test_elastic_net_threshold_scales_with_l1_ratio(ratio):
    Z, y = _synthetic()
    Zs, _ = rc.standardize_train(Z)
    c_entry = float(np.min(rc.l1_entry_c(Zs, y, l1_ratio=ratio, standardize=False)))
    for mult, expect in ((0.97, 0), (1.03, 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(penalty="elasticnet", l1_ratio=ratio, C=c_entry * mult, solver="saga", class_weight="balanced",
                                   tol=1e-10, max_iter=50_000, random_state=0).fit(Zs, y)
        assert int(np.sum(np.abs(m.coef_) > 1e-10)) == expect
    # the alternative reading 4 / (C n l1_ratio) would put entry at c_entry / ratio**2: far off
    assert c_entry / ratio**2 > 1.5 * c_entry


@pytest.mark.parametrize("penalty,ratio", [("l1", None), ("elasticnet", 0.5), ("l2", None)])
def test_prox_newton_matches_tight_saga(penalty, ratio):
    Z, y = _synthetic(n=2000, k=5, shifts=(0.6, -0.4, 0.2))
    Zs, _ = rc.standardize_train(Z)
    for C in (3e-3, 0.1, 10.0):
        ours = rc.ProxNewtonLogistic(penalty=penalty, C=C, l1_ratio=ratio).fit(Zs, y)
        kw = dict(penalty=penalty, C=C, solver="saga", class_weight="balanced", tol=1e-10, max_iter=200_000, random_state=0)
        if ratio is not None:
            kw["l1_ratio"] = ratio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = LogisticRegression(**kw).fit(Zs, y)
        assert np.max(np.abs(ours.coef_ - ref.coef_)) < 1e-6
        assert (np.abs(ours.coef_) > 1e-8).sum() == (np.abs(ref.coef_) > 1e-8).sum()


def test_default_solver_switches_for_wide_inputs():
    Z, y = _synthetic(n=600, k=4)
    narrow = rc.fit_probe(Z, y, penalty="l1", C=0.1)
    assert isinstance(narrow.named_steps["logreg"], rc.ProxNewtonLogistic)
    wide = np.random.default_rng(0).normal(size=(600, rc.NEWTON_CD_MAX_FEATURES + 1))
    est = rc.fit_probe(wide, y, penalty="l1", C=0.1, max_iter=200)
    assert isinstance(est.named_steps["logreg"], LogisticRegression)
    assert est.named_steps["logreg"].solver == "saga"
    assert isinstance(rc.fit_probe(Z, y, penalty="l2", C=0.1).named_steps["logreg"], LogisticRegression)


def test_kkt_margins_at_zero_and_at_solution():
    Z, y = _synthetic()
    Zs, _ = rc.standardize_train(Z)
    C = 1.5 * float(np.min(rc.l1_entry_c(Zs, y, standardize=False)))
    at_zero = rc.kkt_margins(Zs, y, np.zeros(Zs.shape[1]), C)
    assert np.allclose(at_zero, rc.l1_entry_margins(Zs, y, C, standardize=False), rtol=1e-8)
    est = rc.fit_probe(Zs, y, penalty="l1", C=C, standardize=False)
    coef = rc.probe_coef(est)
    kkt = rc.kkt_margins(Zs, y, coef, C)
    zero = np.abs(coef) <= rc.COEF_NONZERO_TOL
    assert zero.any() and (~zero).any()
    assert np.all(kkt[zero] <= 1.0 + 1e-6)
    assert np.allclose(kkt[~zero], 1.0, atol=1e-5)
    b = rc.optimal_intercept(Zs, y, coef)
    assert b == pytest.approx(float(est.named_steps["logreg"].intercept_[0]), abs=1e-6)


def test_all_zero_probe_has_zero_lift():
    Z, y = _synthetic()
    est = rc.fit_probe(Z, y, penalty="l1", C=1e-6)
    assert np.all(rc.probe_coef(est) == 0.0)
    ap, prevalence, lift = rc.heldout_lift(rc.probe_scores(est, Z), y)
    assert ap == pytest.approx(prevalence) and lift == pytest.approx(0.0)
    assert np.isnan(rc.heldout_lift(np.ones(5), np.zeros(5))[2])


# --------------------------------------------------------------------------------------
# nested tuning
# --------------------------------------------------------------------------------------


def test_supported_patients_rule():
    _, y, pats = _cohort()
    assert rc.supported_patients(y, pats) == ["Q01", "Q02", "Q03", "Q04"]
    assert rc.supported_patients(y, pats, min_malignant=30) == ["Q01", "Q02", "Q03"]


@pytest.mark.parametrize("penalty,ratio", [("l1", None), ("elasticnet", 0.5), ("l2", None)])
def test_tune_c_inner_lopo_is_deterministic(penalty, ratio):
    Z, y, pats = _cohort()
    grid = rc.DEFAULT_C_GRID[::2]
    a = rc.tune_c_inner_lopo(Z, y, pats, penalty=penalty, l1_ratio=ratio, c_grid=grid)
    b = rc.tune_c_inner_lopo(Z, y, pats, penalty=penalty, l1_ratio=ratio, c_grid=grid, n_jobs=2)
    assert a["best_C"] == b["best_C"]
    assert a["scores_by_C"] == b["scores_by_C"]
    assert a["n_inner_folds"] == 4 and a["inner_folds"] == ["Q01", "Q02", "Q03", "Q04"]
    assert not a["fallback"]
    assert a["best_C"] in grid
    assert a["best_score"] == max(v for v in a["scores_by_C"].values() if np.isfinite(v))
    assert len(a["per_fold"]) == 4 * len(grid)


def test_tune_tie_break_prefers_sparser_then_smaller_c():
    Z, y, pats = _cohort()
    # every C at or below 1e-4 is all-zero (lift exactly 0); with a signal-free feature set
    # only all-zero probes tie, so the smallest C must win
    rng = np.random.default_rng(3)
    noise = rng.normal(size=Z.shape)
    out = rc.tune_c_inner_lopo(noise, y, pats, penalty="l1", c_grid=(1e-6, 1e-5, 1e-4))
    assert all(v == 0.0 for v in out["scores_by_C"].values())
    assert out["best_C"] == 1e-6


def test_tune_fallback_with_fewer_than_two_supported_patients():
    Z, y, pats = _cohort(malignant=(60, 5, 3, 0, 0, 0))
    out = rc.tune_c_inner_lopo(Z, y, pats, penalty="l1", c_grid=rc.DEFAULT_C_GRID[::4])
    assert out["fallback"] is True
    assert out["n_inner_folds"] == 5
    assert "stratified" in out["fallback_reason"]
    again = rc.tune_c_inner_lopo(Z, y, pats, penalty="l1", c_grid=rc.DEFAULT_C_GRID[::4])
    assert again["best_C"] == out["best_C"] and again["scores_by_C"] == out["scores_by_C"]


def test_c_policy_record_overrides():
    rec = rc.c_policy_record(c_policy="nested", training_n_changed=True)
    assert rec["c_policy"] == "nested" and rec["training_n_changed"] is True
    assert rec["c_grid_lower_bound"] == rc.C_GRID_LOWER_BOUND
    assert "4*l1_ratio/(C*n_train)" in rec["l1_entry_rule"]


# --------------------------------------------------------------------------------------
# regression against the canonical experiment (skipped without EXP)
# --------------------------------------------------------------------------------------

IFNA = "single_geneset__hallmark_interferon_alpha_response"
SUPPORTED = ("P01", "P02", "P03", "P04", "P06", "P07", "P09")


def _exp_or_skip() -> Path:
    try:
        paths = _load("sc_paths_for_contract_test", PATHS_FILE)
        exp = paths.experiment_dir(must_exist=False)
        meta = paths.cell_metadata()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"paths unavailable: {exc}")
    needed = [meta, exp / "stage1_dr" / f"panel_id={IFNA}" / "method=pca" / "k=40" / "seed=42" / "scores.npy",
              exp / "analysis" / "cache" / "stage2_diagnostic" / "layer4_5" / "selected_lopo_coefficients.pkl.gz"]
    if not all(p.exists() for p in needed):
        pytest.skip("canonical experiment not available")
    return exp


def _anchor_rows(exp: Path):
    import pandas as pd

    paths = sys.modules["sc_paths_for_contract_test"]
    meta = pd.read_csv(paths.cell_metadata(), usecols=["y_true", "patient"])
    y = meta["y_true"].to_numpy().astype(int)
    pats = meta["patient"].astype(str).to_numpy()
    cache = pd.read_pickle(exp / "analysis" / "cache" / "stage2_diagnostic" / "layer4_5" / "selected_lopo_coefficients.pkl.gz")
    out = {}
    for method in ("pca", "fa"):
        rep = f"{IFNA}|dr|{method}|40|42"
        sub = cache.loc[cache["representation_id"].eq(rep) & cache["selection_roles"].str.contains("global_selected")]
        assert set(sub["penalty"]) == {"l1"} and np.allclose(sub["C"], 1e-4)
        scores = np.load(exp / "stage1_dr" / f"panel_id={IFNA}" / f"method={method}" / "k=40" / "seed=42" / "scores.npy")
        for heldout in SUPPORTED:
            train = pats != heldout
            margins = rc.l1_entry_margins(scores[train], y[train], 1e-4)
            stored = sub.loc[sub["heldout_patient"].eq(heldout), "is_nonzero"].any()
            out[(method, heldout)] = (float(margins.max()), int(np.argmax(margins)), bool(stored))
    return out


def test_ifna_anchor_zero_nonzero_pattern_reproduced():
    exp = _exp_or_skip()
    rows = _anchor_rows(exp)
    assert all((m > 1.0) == stored for m, _, stored in rows.values())
    nonzero = {k for k, (m, _, _) in rows.items() if m > 1.0}
    assert nonzero == {("pca", p) for p in ("P02", "P03", "P04", "P06", "P07")} | {("fa", "P03"), ("fa", "P04")}
    assert {j for (meth, _), (_, j, _) in rows.items() if meth == "pca"} == {7}  # PC8 everywhere
    assert {j for (meth, _), (_, j, _) in rows.items() if meth == "fa"} == {6}  # FA7 everywhere


def test_ifna_fa7_is_a_knife_edge_in_p02_and_p07():
    exp = _exp_or_skip()
    rows = _anchor_rows(exp)
    for heldout in ("P02", "P07"):
        margin = rows[("fa", heldout)][0]
        assert 0.95 < margin < 1.0  # within 5% below the threshold: C x 1.05 lets FA7 enter


def test_d19_runner_refuses_smoke_without_out_root():
    runner = _load("run_d19_regularization_contract", MODULE_FILE.parent / "run_d19_regularization_contract.py")
    with pytest.raises(SystemExit, match="--out-root"):
        runner.main(["--smoke", "--parts", "report"])
    with pytest.raises(SystemExit, match="--out-root"):
        runner.main(["--max-cells", "3", "--parts", "report"])

