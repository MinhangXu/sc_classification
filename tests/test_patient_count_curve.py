"""Tests for the D15b patient-count learning curve (W6.1).

Module: scripts/mrd_stage0_2/workstreams/W6_learning_curves/patient_count_curve.py, loaded by file
path with importlib. Synthetic tests cover subset enumeration, the reduced cyclic design, the
training mask, penalty selection, the patient-cluster bootstrap, the summaries and the inverse
power-law fit. The last tests read the canonical experiment read-only and are skipped when it is
absent: they re-check the stored D15b reference checks and recompute one k = 6 fixed-C fold
against the D1 atlas.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
W6 = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W6_learning_curves"
W3 = REPO / "scripts" / "mrd_stage0_2" / "workstreams" / "W3_dr_value_regularization"
MODULE_FILE = W6 / "patient_count_curve.py"
PATHS_FILE = REPO / "src" / "sc_classification" / "utils" / "paths.py"
SUPPORTED = ("P01", "P02", "P03", "P04", "P06", "P07", "P09")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


for _dir in (W6, W3):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
pcc = _load("patient_count_curve", MODULE_FILE)


# --------------------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_enumerate_subsets_counts_and_exclusion(k):
    for h in SUPPORTED:
        subsets = pcc.enumerate_subsets(h, SUPPORTED, k)
        assert len(subsets) == math.comb(6, k)
        assert len(set(subsets)) == len(subsets)
        for s in subsets:
            assert h not in s
            assert len(s) == k
            assert list(s) == sorted(s)


def test_enumerate_subsets_rejects_bad_input():
    with pytest.raises(ValueError):
        pcc.enumerate_subsets("P05", SUPPORTED, 2)  # unsupported held-out patient
    with pytest.raises(ValueError):
        pcc.enumerate_subsets("P01", SUPPORTED, 0)
    with pytest.raises(ValueError):
        pcc.enumerate_subsets("P01", SUPPORTED, 7)


def test_full_design_table_size():
    design = pcc.design_table(SUPPORTED, ks=(1, 2, 3, 4, 5, 6), design="full")
    assert len(design) == 7 * 63  # 441 (held-out, subset) rows per representation
    assert design.groupby("k").size().to_dict() == {k: 7 * math.comb(6, k) for k in range(1, 7)}
    assert design["subset"].nunique() == 2 ** 7 - 2  # every nonempty proper subset of the 7


def test_cyclic_design_balance():
    for size in range(1, 7):
        family = pcc.cyclic_design_family(SUPPORTED, size)
        assert len(family) == 7
        counts = pd.Series([p for s in family for p in s]).value_counts()
        assert set(counts.index) == set(SUPPORTED)
        assert set(counts.values) == {size}  # every patient appears equally often
    for size, lam in ((3, 1), (4, 2)):  # balanced incomplete block designs
        family = pcc.cyclic_design_family(SUPPORTED, size)
        pair_counts = {pair: sum(set(pair) <= set(s) for s in family) for pair in itertools.combinations(SUPPORTED, 2)}
        assert set(pair_counts.values()) == {lam}


def test_reduced_design_is_subset_of_full():
    full = pcc.design_table(SUPPORTED, design="full")
    reduced = pcc.design_table(SUPPORTED, design="reduced")
    key = ["heldout_patient", "k", "subset"]
    merged = reduced[key].merge(full[key], how="left", indicator=True)
    assert (merged["_merge"] == "both").all()
    assert reduced.groupby("k").size().to_dict() == {k: 7 * (7 - k) for k in range(1, 7)}
    assert reduced["subset"].nunique() == 42
    for h in SUPPORTED:
        assert not reduced.loc[reduced["heldout_patient"].eq(h), "subset"].str.contains(h).any()


def test_cyclic_design_needs_seven_patients():
    with pytest.raises(ValueError):
        pcc.cyclic_design_family(SUPPORTED[:6], 3)


def test_subset_tag_roundtrip():
    assert pcc.subset_tag(["P04", "P01"]) == "P01+P04"
    assert pcc.parse_subset_tag("P04+P01") == ("P01", "P04")
    with pytest.raises(ValueError):
        pcc.subset_tag([])


def test_training_mask_includes_unsupported_and_excludes_heldout():
    patients = np.array(["P01", "P02", "P03", "P05", "P08", "P13", "P02", "P07"])
    mask = pcc.training_mask(patients, ("P02",), SUPPORTED, heldout="P07")
    assert mask.tolist() == [False, True, False, True, True, True, True, False]
    with pytest.raises(ValueError):
        pcc.training_mask(patients, ("P07",), SUPPORTED, heldout="P07")
    with pytest.raises(ValueError):
        pcc.training_mask(patients, ("P05",), SUPPORTED)


def test_select_tuned_penalty():
    assert pcc.select_tuned_penalty({"l1": 0.10, "l2": 0.12}) == "l2"
    assert pcc.select_tuned_penalty({"l1": 0.12, "l2": 0.10}) == "l1"
    assert pcc.select_tuned_penalty({"l1": 0.12, "l2": 0.12}) == "l1"  # tie -> sparser
    assert pcc.select_tuned_penalty({"l1": float("nan"), "l2": 0.01}) == "l2"
    with pytest.raises(ValueError):
        pcc.select_tuned_penalty({"l1": float("nan"), "l2": float("nan")})


# --------------------------------------------------------------------------------------
# bootstrap and summaries
# --------------------------------------------------------------------------------------


def test_bootstrap_patient_stat_properties():
    vals = np.array([0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.30])
    a = pcc.bootstrap_patient_stat(vals, "median", n_boot=500, seed=1)
    b = pcc.bootstrap_patient_stat(vals, "median", n_boot=500, seed=1)
    assert a == b  # deterministic given the seed
    assert a["estimate"] == pytest.approx(0.05)
    assert vals.min() <= a["lo"] <= a["estimate"] <= a["hi"] <= vals.max()
    const = pcc.bootstrap_patient_stat(np.full(7, 0.04), "mean", n_boot=200)
    assert const["lo"] == const["hi"] == pytest.approx(0.04)
    assert const["se"] == pytest.approx(0.0)
    empty = pcc.bootstrap_patient_stat([], "median")
    assert empty["n_clusters"] == 0 and math.isnan(empty["estimate"])


def _synthetic_units(curve_fn, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    offsets = dict(zip(SUPPORTED, np.linspace(0.0, 0.12, 7)))
    rows = []
    for row in pcc.design_table(SUPPORTED, design="full").itertuples(index=False):
        lift = offsets[row.heldout_patient] + curve_fn(row.k) + noise * rng.normal()
        rows.append({
            "representation": "rep", "arm": "tuned", "heldout_patient": row.heldout_patient, "k": row.k,
            "subset": row.subset, "heldout_lift": lift, "heldout_ap": lift + 0.05, "prevalence": 0.05,
            "n_train_malignant": 100 * row.k, "n_train": 20000 + 5000 * row.k, "n_nonzero": 3,
            "at_grid_lower_bound": False,
        })
    return pd.DataFrame(rows)


def test_summaries_on_synthetic_curve():
    curve_fn = lambda k: pcc.inverse_power_law(k, 0.10, 0.08, 0.9)  # noqa: E731
    units = _synthetic_units(curve_fn)
    bp = pcc.summarize_by_patient_k(units)
    assert len(bp) == 7 * 6
    assert (bp["n_subsets"] == bp["k"].map(lambda k: math.comb(6, k))).all()
    curve = pcc.curve_by_k(bp, n_boot=300, units=units)
    median_offset = float(np.median(np.linspace(0.0, 0.12, 7)))
    for r in curve.itertuples(index=False):
        assert r.median_lift == pytest.approx(median_offset + curve_fn(r.k))
        assert r.median_lift_lo <= r.median_lift <= r.median_lift_hi
        assert r.n_heldout == 7
    gain = pcc.paired_gain(bp, k_from=1, n_boot=300)
    for r in gain.itertuples(index=False):
        assert r.median_gain == pytest.approx(curve_fn(r.k_to) - curve_fn(1))
        assert r.n_patients_gain_gt_0 == 7


def test_patient_inclusion_effect_detects_a_helpful_patient():
    units = _synthetic_units(lambda k: 0.0)
    helps = units["subset"].str.contains("P03")
    units.loc[helps, "heldout_lift"] += 0.05
    inc = pcc.patient_inclusion_effect(units)
    p03 = inc.loc[inc["included_patient"].eq("P03")]
    assert len(p03) == 6  # every held-out patient except P03
    assert np.allclose(p03["inclusion_effect"], 0.05)
    others = inc.loc[~inc["included_patient"].eq("P03"), "inclusion_effect"]
    assert others.abs().max() < 0.05


# --------------------------------------------------------------------------------------
# inverse power law
# --------------------------------------------------------------------------------------


def test_power_law_fit_recovers_parameters():
    k = np.arange(1, 7, dtype=float)
    y = pcc.inverse_power_law(k, 0.15, 0.10, 0.8)
    fit = pcc.fit_inverse_power_law(k, y, np.full(k.size, 0.01))
    assert fit["converged"] and fit["at_bound"] == ""
    assert fit["a"] == pytest.approx(0.15, abs=1e-4)
    assert fit["b"] == pytest.approx(0.10, abs=1e-4)
    assert fit["c"] == pytest.approx(0.8, abs=1e-3)


def test_power_law_validation_predicts_k5_from_k4():
    k = np.arange(1, 7, dtype=float)
    rng = np.random.default_rng(3)
    y = pcc.inverse_power_law(k, 0.12, 0.08, 1.0) + rng.normal(0, 0.001, k.size)
    se = np.full(k.size, 0.004)
    rec = pcc.validate_power_law(k, y, se, y - 0.02, y + 0.02, fit_ks=(1, 2, 3, 4), target_k=5)
    assert rec["supported"], rec["reason"]
    assert rec["abs_error"] < 0.01
    assert rec["pred_in_obs_ci"]


def test_power_law_not_supported_on_flat_curve():
    k = np.arange(1, 7, dtype=float)
    y = np.full(k.size, 0.05)
    se = np.full(k.size, 0.01)
    rec = pcc.validate_power_law(k, y, se, y - 0.02, y + 0.02, fit_ks=(1, 2, 3, 4), target_k=5)
    assert not rec["supported"]
    assert "rise" in rec["reason"] or "bound" in rec["reason"]


def test_power_law_not_supported_when_prediction_misses():
    k = np.arange(1, 7, dtype=float)
    y = pcc.inverse_power_law(k, 0.12, 0.08, 1.0)
    y[4] = 0.30  # k = 5 jumps far above the extrapolation
    rec = pcc.validate_power_law(k, y, np.full(k.size, 0.004), y - 0.01, y + 0.01, fit_ks=(1, 2, 3, 4), target_k=5)
    assert not rec["supported"]
    assert "outside the observed CI" in rec["reason"]


def test_power_law_table_roles():
    units = _synthetic_units(lambda k: pcc.inverse_power_law(k, 0.10, 0.08, 0.9), noise=0.0005)
    curve = pcc.curve_by_k(pcc.summarize_by_patient_k(units), n_boot=300)
    table = pcc.power_law_table(curve)
    assert set(table["validation"]) == {"primary", "secondary"}
    prim = table.loc[table["validation"].eq("primary") & table["stat"].eq("mean")].iloc[0]
    assert prim["target_k"] == 5 and prim["fit_ks"] == "1;2;3;4"
    assert prim["supported"], prim["reason"]


def test_too_few_points():
    fit = pcc.fit_inverse_power_law([1, 2], [0.1, 0.2])
    assert not fit["converged"] and fit["message"] == "fewer than 3 points"


# --------------------------------------------------------------------------------------
# regression against the canonical experiment (skipped when absent)
# --------------------------------------------------------------------------------------


def _exp_dir():
    try:
        paths = _load("sc_paths_for_d15b_tests", PATHS_FILE)
        return paths.experiment_dir(must_exist=False)
    except Exception:
        return None


EXP = _exp_dir()
SLUG_TABLES = None if EXP is None else EXP / "analysis" / "scorecards" / "stage2_diagnostic" / "patient_count_learning_curve_v1"
needs_exp = pytest.mark.skipif(EXP is None or not EXP.is_dir(), reason="canonical experiment dir not available")


@needs_exp
def test_stored_reference_checks_pass():
    found = sorted(SLUG_TABLES.glob("d15b_*_reference_checks.csv")) if SLUG_TABLES.is_dir() else []
    if not found:
        pytest.skip("no D15b reference checks written yet")
    for path in found:
        ref = pd.read_csv(path)
        assert len(ref) > 0
        assert ref["pass"].astype(bool).all(), path.name
        d19 = ref.loc[ref["check"].str.contains("D19")]
        if len(d19):
            assert (d19["abs_diff"] < 1e-9).all()  # same contract, same rows: identical tuning


@needs_exp
def test_k6_fixed_fold_matches_atlas():
    """IFN-alpha PCA40, held-out P02, all other patients, atlas L1 C: the ordinary LOPO fold."""
    rc = sys.modules.get("regularization_contract") or _load("regularization_contract", W3 / "regularization_contract.py")
    meta_path = EXP / "stage2_supervised" / "multiobjective" / "discovery_full_cohort" / "cell_metadata.csv.gz"
    scores_path = EXP / "stage1_dr" / "panel_id=single_geneset__hallmark_interferon_alpha_response" / "method=pca" / "k=40" / "seed=42" / "scores.npy"
    atlas_path = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / "old34_d1_atlas_supported_patient_metrics.csv"
    if not (meta_path.is_file() and scores_path.is_file() and atlas_path.is_file()):
        pytest.skip("inputs missing")
    meta = pd.read_csv(meta_path, usecols=["y_true", "patient"])
    y = meta["y_true"].to_numpy().astype(int)
    pat = meta["patient"].astype(str).to_numpy()
    Z = np.load(scores_path).astype(np.float32)
    others = tuple(p for p in SUPPORTED if p != "P02")
    train = pcc.training_mask(pat, others, SUPPORTED, heldout="P02")
    assert train.sum() == (pat != "P02").sum()  # k = 6 is the LOPO fold
    est = rc.fit_probe(Z[train], y[train], penalty="l1", C=1e-4)
    ap, _, _ = rc.heldout_lift(rc.probe_scores(est, Z[pat == "P02"]), y[pat == "P02"])
    atlas = pd.read_csv(atlas_path)
    ref = atlas.loc[atlas["stage0_panel_id"].eq("single_geneset__hallmark_interferon_alpha_response")
                    & atlas["configuration"].eq("pca|K=40") & atlas["heldout_patient"].eq("P02"), "heldout_auprc"]
    assert len(ref) == 1
    assert abs(ap - float(ref.iloc[0])) < 1e-3


def test_reduced_design_balance_by_size():
    """Sizes 1, 3, 4 and 6 are balanced over training-patient membership for every held-out
    patient; sizes 2 and 5 are not (the findings and the devlog caveat those points)."""
    for k, balanced in ((1, True), (2, False), (3, True), (4, True), (5, False), (6, True)):
        for h in SUPPORTED:
            subs = pcc.reduced_subsets(h, SUPPORTED, k)
            counts = {p: sum(p in s for s in subs) for p in SUPPORTED if p != h}
            assert (len(set(counts.values())) == 1) == balanced, (k, h, counts)
    # held-out P02, size 2: its sorted neighbours P01 and P03 enter 1 of the 5 subsets
    subs = pcc.reduced_subsets("P02", SUPPORTED, 2)
    assert len(subs) == 5 and sum("P03" in s for s in subs) == 1 and sum("P01" in s for s in subs) == 1

