"""Tests for the D29 runner (scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d29_protein_depth_contrast.py).

Synthetic tests check the statistics with known answers: the raw Spearman against scipy, a correlation planted only
through a shared depth covariate (the partial removes it), one planted beside the covariate (the partial keeps it),
the AUC against Mann-Whitney U, and the same-sign count. Regression tests read the real protein_depth_contrast_v1
outputs and are skipped when the experiment dir (or the outputs) are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[1]
W5 = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W5_factor_interpretation"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, W5 / "{}.py".format(name))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


d29 = load("run_d29_protein_depth_contrast")


def test_raw_rho_matches_scipy():
    rng = np.random.default_rng(0)
    f = rng.normal(size=400)
    Y = np.column_stack([f + rng.normal(size=400), rng.normal(size=400), -f + 0.5 * rng.normal(size=400)])
    raw, _ = d29.spearman_raw_partial(f, Y, rng.normal(size=400))
    for j in range(Y.shape[1]):
        assert raw[j] == pytest.approx(spearmanr(f, Y[:, j])[0], abs=1e-10)


def test_partial_removes_a_correlation_carried_by_depth_only():
    rng = np.random.default_rng(1)
    z = rng.normal(size=3000)  # the shared covariate (total counts)
    f = z + rng.normal(size=3000)
    y = -z + rng.normal(size=3000)  # tied to f only through z
    raw, part = d29.spearman_raw_partial(f, y[:, None], z)
    assert raw[0] < -0.3
    assert abs(part[0]) < 0.06


def test_partial_keeps_a_correlation_beside_depth():
    rng = np.random.default_rng(2)
    z = rng.normal(size=3000)
    f = rng.normal(size=3000)
    y = f + rng.normal(size=3000)  # independent of z
    raw, part = d29.spearman_raw_partial(f, y[:, None], z)
    assert raw[0] > 0.5 and part[0] == pytest.approx(raw[0], abs=0.03)


def test_constant_column_is_nan():
    f = np.arange(50, dtype=float)
    Y = np.column_stack([np.ones(50), f])
    raw, part = d29.spearman_raw_partial(f, Y, np.linspace(0, 1, 50) ** 2)
    assert np.isnan(raw[0]) and np.isnan(part[0])
    assert raw[1] == pytest.approx(1.0)


def test_auc_matches_mann_whitney_with_ties():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 5, size=40).astype(float)
    b = rng.integers(1, 6, size=60).astype(float)
    u = mannwhitneyu(a, b).statistic
    assert d29.auc(a, b) == pytest.approx(u / (len(a) * len(b)), abs=1e-12)
    assert d29.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert d29.auc(np.array([1.0]), np.array([1.0])) == 0.5


def test_residual_ranks_remove_the_covariate():
    rng = np.random.default_rng(4)
    z = rng.normal(size=500)
    y = 2 * z + 0.1 * rng.normal(size=500)
    res = d29.residual_ranks(y, z)
    assert abs(np.corrcoef(res, pd.Series(z).rank().to_numpy())[0, 1]) < 1e-10


def test_same_sign_counts_against_the_reference():
    v = np.array([0.2, 0.1, -0.3, np.nan, 0.05])
    assert d29.same_sign(v, 0.1) == 3
    assert d29.same_sign(v, -0.1) == 1
    assert d29.same_sign(v, 0.0) == 0


def test_paths_refuse_to_write_outside_the_slug(tmp_path):
    p = d29.Paths(tmp_path / "exp", tmp_path / "out")
    assert p.out.name == d29.SLUG
    assert p.rel(p.out).startswith("<out-root>/")


# ---- regression against the stored outputs (skipped without EXP) ---------------------------------------------


def _out_dir():
    try:
        sys.path.insert(0, str(REPO / "src"))
        from sc_classification.utils.paths import experiment_dir

        d = experiment_dir() / "analysis" / "scorecards" / "stage2_diagnostic" / d29.SLUG
    except Exception:
        return None
    return d if (d / "d29_regression_checks.csv").is_file() else None


@pytest.mark.skipif(_out_dir() is None, reason="protein_depth_contrast_v1 outputs not available")
def test_stored_outputs_pass_their_checks_and_hold_the_reported_numbers():
    d = _out_dir()
    checks = pd.read_csv(d / "d29_regression_checks.csv")
    assert len(checks) == 8 and checks["passed"].all()
    rho = pd.read_csv(d / "d29_rho_depth.csv")
    assert len(rho) == 2 * 2 * 170
    cd63 = rho[(rho["factor"] == "PC8") & (rho["label"] == "malignant") & (rho["feature_short"] == "CD63")].iloc[0]
    assert round(cd63["rho_within_median_raw"], 3) == -0.252 and round(cd63["rho_within_median_partial"], 3) == -0.200
    assert round(cd63["rho_pct_median_partial"], 3) == -0.122
    summ = pd.read_csv(d / "d29_mal_vs_healthy_summary.csv")
    s = summ[(summ["level"] == "sample") & (summ["feature_short"] == "CD63")].iloc[0]
    assert (s["n_lower_raw"], s["n_groups"]) == (15, 21)
