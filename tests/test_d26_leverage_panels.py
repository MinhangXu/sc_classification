"""Tests for the D26 leave-high-prevalence-out leverage runner (task W2.4).

Module: scripts/mrd_stage0_2/workstreams/W2_stage1_basis_variants/run_d26_leverage_panels.py,
loaded by file path with importlib. The synthetic tests are fast and need no experiment dir; the
regression tests at the end re-check the canonical experiment read-only (reference-axis rules
against D17 and D18 artifacts, and the D26 slug tables once they exist) and are skipped when it
or the artifact is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
WORKSTREAMS = REPO / "scripts" / "mrd_stage0_2" / "workstreams"
MODULE_FILE = WORKSTREAMS / "W2_stage1_basis_variants" / "run_d26_leverage_panels.py"
D17_FILE = WORKSTREAMS / "W2_stage1_basis_variants" / "run_d17_basis_variant_pilot.py"
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


d26 = _load("run_d26_leverage_panels", MODULE_FILE)
d17 = sys.modules.get("run_d17_basis_variant_pilot") or _load("run_d17_basis_variant_pilot", D17_FILE)
s1 = d26.s1


# --------------------------------------------------------------------------------------
# synthetic helpers
# --------------------------------------------------------------------------------------


def _cohort(seed=0):
    """Toy cohort: 4 patients with different prevalence, 8 genes, a planted malignant direction."""
    rng = np.random.default_rng(seed)
    sizes = {"P01": 300, "P02": 120, "P03": 150, "P09": 330}
    n_mal = {"P01": 6, "P02": 40, "P03": 60, "P09": 30}
    pat = np.concatenate([[p] * n for p, n in sizes.items()])
    y = np.zeros(pat.size, dtype=int)
    start = 0
    for p, n in sizes.items():
        y[start:start + n_mal[p]] = 1
        start += n
    X = rng.normal(size=(pat.size, 8))
    X[:, 2] += 1.8 * y  # gene 2 carries the label
    X[:, 0] *= 3.0
    return X, y, pat


# --------------------------------------------------------------------------------------
# names and reference-axis statistics
# --------------------------------------------------------------------------------------


def test_panel_names():
    assert d26.normalise_panel("apoptosis") == "single_geneset__hallmark_apoptosis"
    full = "single_geneset__hallmark_allograft_rejection"
    assert d26.normalise_panel(full) == full
    assert d26.panel_tag(full) == "hallmark_allograft_rejection"
    assert d26.panel_label(full) == "allograft rejection"
    assert d26.features_rel(full).endswith(f"panel_id={full}/transform=zscore")


def test_top_gap_factor_finds_planted_factor():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(50, dtype=int), np.zeros(450, dtype=int)]
    S = rng.normal(size=(500, 5))
    S[:, 3] -= 2.0 * y  # negative gap, largest in absolute value
    S[:, 1] += 0.5 * y
    gap = d26.all_cell_gap(S, y)
    assert d26.top_gap_factor(S, y) == 3
    assert gap[3] < 0 and gap[1] > 0
    # scale invariance: standardization removes column scale
    S2 = S * np.array([1, 100, 1, 0.01, 1])
    assert d26.top_gap_factor(S2, y) == 3


def test_univariate_heldout_lifts_sign_from_training_and_nan_for_one_class():
    rng = np.random.default_rng(2)
    pat = np.repeat(np.array(["A", "B", "C"]), 200)
    y = np.zeros(600, dtype=int)
    y[:40] = 1
    y[200:230] = 1  # C has no malignant cell
    S = rng.normal(size=(600, 3))
    S[:, 1] -= 3.0 * y  # malignant cells score low: oriented by the training sign, the lift is high
    lifts = d26.univariate_heldout_lifts(S, y, pat, ["A", "B", "C"])
    assert lifts.shape == (3, 3)
    assert np.all(np.isnan(lifts[2]))
    assert lifts[0, 1] > 0.5 and lifts[1, 1] > 0.5
    assert abs(lifts[0, 0]) < 0.15
    assert d26.top_heldout_factor(lifts) == 1


def test_rank_p_matches_d17_definition():
    controls = np.linspace(0.0, 1.0, 20)
    assert d26.rank_p(2.0, controls) == pytest.approx(1 / 21)
    assert d26.rank_p(-1.0, controls) == pytest.approx(1.0)
    assert d26.rank_p(1.0, controls) == pytest.approx(2 / 21)  # a tie counts as "at least as much"
    assert d26.rank_p(-1.0, controls, larger_moves=False) == pytest.approx(1 / 21)
    assert np.isnan(d26.rank_p(np.nan, controls))
    assert d26.rank_p(0.5, [np.nan, 0.1, 0.9]) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------------------
# leverage table, decision summary
# --------------------------------------------------------------------------------------


def _toy_alignment(seed=0, n_label=20, with_probe=False):
    """Synthetic alignment rows in the D17 schema for one panel x method x K."""
    rng = np.random.default_rng(seed)
    rows = []

    def row(arm, variant, matches="", loss=None, rank=None, axis="top_gap"):
        loss = float(rng.uniform(1e-4, 1e-3)) if loss is None else loss
        rec = {
            "panel_id": "single_geneset__hallmark_apoptosis", "ref_axis": axis, "stage1_method": "pca", "effective_k": 40,
            "basis_id": f"{arm}__{variant}__{axis}", "arm": arm, "variant": variant, "matches_drop_set": matches,
            "n_dropped_cells": 1000, "n_dropped_malignant": 100, "ref_factor_label": "PC6" if axis != "probe" else "probe",
            "matched_factor_fit": "factor_006" if axis != "probe" else "", "axis_loss_fit": loss,
            "matched_abs_r_fit": float(1 - loss) if axis != "probe" else np.nan, "many_to_one_r2_fit": 1 - loss,
            "angle_top8_max_deg": float(rng.uniform(1, 5)), "angle_top8_mean_cos2": float(rng.uniform(0.98, 1)),
            "angle_allK_mean_cos2": float(rng.uniform(0.99, 1)),
            "ref_axis_containment_top8": float(rng.uniform(0.99, 1)) if axis != "probe" else np.nan,
            "matched_factor_importance_rank": float(rank if rank is not None else 6) if axis != "probe" else np.nan,
        }
        rows.append(rec)

    for axis in (("top_gap", "probe") if with_probe else ("top_gap",)):
        for tag, loss in (("P02-P03-P04", 5e-3), ("P01", 2e-2), ("P06", 2e-4)):
            row("leave_out", f"drop-{tag}", loss=loss, axis=axis)
            for s in range(3):
                row("leave_out_random", f"drop-{tag}-random-s{s}", matches=tag, axis=axis)
            for s in range(n_label):
                row("leave_out_label_random", f"drop-{tag}-labelmatched-s{s}", matches=tag, axis=axis)
        row("normal_only_all", "all", loss=3e-3, axis=axis)
        for s in range(3):
            row("normal_only_random", f"drop1569-s{s}", axis=axis)
    return pd.DataFrame(rows)


def test_leverage_contrast_equals_d17_leverage_table_for_one_axis():
    align = _toy_alignment()
    mine = d26.leverage_contrast(align)
    theirs = d17.leverage_table(align.drop(columns=["panel_id", "ref_axis"]))
    assert len(mine) == len(theirs) == 4
    for col in theirs.columns:
        a, b = mine[col].to_numpy(), theirs[col].to_numpy()
        if np.issubdtype(np.asarray(b).dtype, np.number):
            np.testing.assert_allclose(a.astype(float), b.astype(float), rtol=1e-12, equal_nan=True, err_msg=col)
        else:
            assert list(a) == list(b), col
    hp = mine.loc[mine["dropped"].eq("P02-P03-P04")].iloc[0]
    assert hp["group"] == "highest_prevalence"
    assert hp["axis_loss_fit_label_matched_rank_p"] == pytest.approx(1 / 21)


def test_leverage_contrast_is_nan_safe_for_the_probe_axis():
    align = _toy_alignment(with_probe=True)
    lev = d26.leverage_contrast(align)
    probe = lev.loc[lev["ref_axis"].eq("probe") & lev["arm"].eq("leave_out")]
    assert len(probe) == 3
    assert probe["axis_loss_fit_label_matched_rank_p"].notna().all()
    # matched |r| is undefined for the multi-factor axis: no rank p, never a spurious floor value
    assert "matched_abs_r_fit_label_matched_rank_p" not in probe.columns or probe["matched_abs_r_fit_label_matched_rank_p"].isna().all()
    top = lev.loc[lev["ref_axis"].eq("top_gap") & lev["arm"].eq("leave_out")]
    assert top["matched_abs_r_fit_label_matched_rank_p"].notna().all()


def test_decision_summary_counts_cells_and_contrasts():
    frames = []
    for k in (10, 20, 40):
        a = _toy_alignment(seed=k)
        a["effective_k"] = k
        frames.append(a)
    lev = d26.leverage_contrast(pd.concat(frames, ignore_index=True))
    dec = d26.decision_summary(lev, 20)
    r = dec.iloc[0]
    assert r["n_method_k_cells"] == 3
    assert r["n_cells_test_set_above_all_composition_controls"] == 3  # 5e-3 exceeds every control in (1e-4, 1e-3)
    assert r["n_cells_test_set_ratio_above_every_contrast"] == 0  # P01 (2e-2) moves the axis more
    assert bool(r["leverage_beyond_composition"]) and not bool(r["beyond_composition_and_contrasts"])


# --------------------------------------------------------------------------------------
# plan, alignment rows and resume
# --------------------------------------------------------------------------------------


def _fake_ctx(tmp_path, X=None, y=None, pat=None, resume=True):
    if y is None:
        X, y, pat = _cohort()
    return SimpleNamespace(
        y=y, patients=pat, panels=["single_geneset__hallmark_apoptosis"], methods=["pca", "fa"], ks=[10, 20, 40],
        drop_sets=d26.DROP_SETS, rand_seeds=d26.RANDOM_DROP_SEEDS, label_seeds=d26.LABEL_MATCHED_DROP_SEEDS,
        bases=tmp_path / "bases", units=tmp_path / "units", args=SimpleNamespace(resume=resume),
    )


def test_build_plan_counts_per_cell(tmp_path):
    rng = np.random.default_rng(0)
    names = ["P01", "P02", "P03", "P04", "P06", "P07", "P09", "P11"]
    pat = np.repeat(names, 60)
    y = (rng.random(pat.size) < 0.2).astype(int)
    ctx = _fake_ctx(tmp_path, y=y, pat=pat)
    specs = d26.build_plan(ctx)
    per_cell = 1 + 1 + 3 + 8 * (1 + 3 + 20)
    assert per_cell == 197
    assert len(specs) == per_cell * 2 * 3
    assert len({s["basis_id"] for s in specs}) == len(specs)
    lab = [s for s in specs if s["arm"] == "leave_out_label_random" and s["fit"]["matches"] == "P02-P03-P04"]
    assert len(lab) == 20 * 6
    n_mal = int(y[np.isin(pat, ["P02", "P03", "P04"])].sum())
    assert {s["fit"]["n_drop_malignant"] for s in lab} == {n_mal}


def test_align_rows_refit_is_lossless_and_leave_out_counts():
    X, y, pat = _cohort()
    k = 4
    ref = s1.fit_basis(X, "pca", k, seed=42)
    S = ref.transform(X)
    L = ref.loadings_common()
    order = np.arange(k)
    j = d26.top_gap_factor(S, y)
    lifts = d26.univariate_heldout_lifts(S, y, pat, ["P01", "P02", "P03", "P09"])
    w = d26.probe_weights(S, y, {"penalty": "l2", "C": 1.0, "l1_ratio": None})
    s_probe = d26.rc.standardize_train(S)[0] @ w
    refd = {"top_gap": j, "top_heldout": d26.top_heldout_factor(lifts), "probe_w": list(w)}
    spec = {"basis_id": "b", "arm": "transductive_refit", "panel": "p", "method": "pca", "k": k, "variant": "all", "fit": {"kind": "rows"}}
    rows_all = np.arange(y.size)
    out = d26.align_rows(S, L, order, s1.fit_basis(X, "pca", k, seed=42).transform(X), ref, rows_all, y.size, int(y.sum()), y,
                         spec, refd, s_probe)
    assert [r["ref_axis"] for r in out] == ["top_gap", "top_heldout", "probe"]
    for r in out:
        assert r["axis_loss_fit"] < 1e-9 and r["n_dropped_cells"] == 0
    keep = np.flatnonzero(pat != "P02")
    var = s1.fit_basis(X, "pca", k, rows=keep, seed=42)
    spec2 = dict(spec, arm="leave_out", variant="drop-P02", fit={"kind": "rows", "exclude": ["P02"]})
    out2 = d26.align_rows(S, L, order, var.transform(X), var, keep, y.size, int(y.sum()), y, spec2, refd, s_probe)
    assert out2[0]["n_dropped_cells"] == int((pat == "P02").sum())
    assert out2[0]["n_dropped_malignant"] == int(y[pat == "P02"].sum())
    assert np.isnan(out2[2]["matched_abs_r_fit"]) and 0 <= out2[2]["axis_loss_fit"] <= 1


def test_unit_state_resume_semantics(tmp_path):
    X, y, pat = _cohort()
    ctx = _fake_ctx(tmp_path, X, y, pat)
    spec = {"basis_id": "leave_out__x__pca4__drop-P02", "arm": "leave_out", "panel": "p", "method": "pca", "k": 4,
            "variant": "drop-P02", "fit": {"kind": "rows", "exclude": ["P02"]}}
    sig = {"schema": 1, "top_gap": 2}
    assert d26.unit_state(ctx, spec, sig) == "fit"
    s1.fit_basis(X, "pca", 4, seed=42).save(d26.basis_dir(ctx, spec))
    assert d26.unit_state(ctx, spec, sig) == "align"  # basis saved, alignment missing
    upath = d26.unit_path(ctx, spec)
    upath.parent.mkdir(parents=True, exist_ok=True)
    upath.write_text("", encoding="utf-8")
    assert d26.unit_state(ctx, spec, sig) == "align"  # empty file = unfinished
    upath.write_text(json.dumps({"status": "done", "signature": sig, "rows": [{"a": 1}]}), encoding="utf-8")
    assert d26.unit_state(ctx, spec, sig) == "done"
    assert d26.unit_state(ctx, spec, {"schema": 1, "top_gap": 3}) == "align"  # reference axes changed
    upath.write_text("{not json", encoding="utf-8")
    assert d26.unit_state(ctx, spec, sig) == "align"
    ctx.args.resume = False
    assert d26.unit_state(ctx, spec, sig) == "fit"


def test_runner_refuses_smoke_without_out_root():
    with pytest.raises(SystemExit, match="--out-root"):
        d26.main(["--smoke", "--parts", "report"])


# --------------------------------------------------------------------------------------
# regression against the canonical experiment (read-only; skipped when absent)
# --------------------------------------------------------------------------------------

IFNA = "single_geneset__hallmark_interferon_alpha_response"
ALLO = "single_geneset__hallmark_allograft_rejection"


def _exp():
    try:
        paths = _load("sc_classification_utils_paths_for_d26_tests", PATHS_FILE)
        exp = paths.experiment_dir(must_exist=False)
    except Exception:
        return None
    feats = exp / "stage1_direct_gene" / "old34_direct_gene_all_v1" / f"panel_id={ALLO}" / "transform=zscore" / "features.npz"
    return exp if feats.is_file() else None


EXP = _exp()
needs_exp = pytest.mark.skipif(EXP is None, reason="canonical experiment dir not available")


@needs_exp
def test_top_gap_rule_reproduces_d17_reference_axes():
    y, _ = d26.load_meta()
    for (method, k), expected in d17.EXPECTED_REF_FACTOR.items():
        assert d26.top_gap_factor(d26.stored_scores(str(EXP), IFNA, method, k), y) == expected


@needs_exp
def test_ref_arrays_match_d17_for_ifna_pca10():
    S, L, order = d26.ref_arrays(str(EXP), IFNA, "pca", 10)
    S17, L17, order17 = d17._ref_arrays(str(EXP), "pca", 10)
    np.testing.assert_allclose(S, S17, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(L, L17)
    np.testing.assert_array_equal(order, order17)


@needs_exp
def test_univariate_lift_reproduces_d18_cards_allograft_fa40():
    cards = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / "factor_cards_v1" / "factor_cards_index.csv"
    if not cards.is_file():
        pytest.skip("D18 factor cards not present")
    idx = pd.read_csv(cards, usecols=["panel_id", "method", "k", "factor", "median_univariate_lift"])
    g = idx.loc[idx["panel_id"].eq(ALLO) & idx["method"].eq("fa") & idx["k"].eq(40)].set_index("factor").sort_index()
    y, pat = d26.load_meta()
    supported = ["P01", "P02", "P03", "P04", "P06", "P07", "P09"]
    lifts = d26.univariate_heldout_lifts(d26.stored_scores(str(EXP), ALLO, "fa", 40), y, pat, supported)
    np.testing.assert_allclose(np.nanmedian(lifts, axis=0), g["median_univariate_lift"].to_numpy(), atol=1e-9)


def _slug_table(name):
    if EXP is None:
        return None
    path = EXP / "analysis" / "scorecards" / "stage2_diagnostic" / d26.SLUG / name
    return pd.read_csv(path) if path.is_file() else None


@needs_exp
def test_d26_reference_axes_artifact():
    ref = _slug_table("d26_reference_axes.csv")
    if ref is None:
        pytest.skip("D26 reference axes not written yet")
    probe = ref.loc[ref["ref_axis"].eq("probe")]
    assert len(probe) == len(ref) // 3
    assert (probe["probe_cos_vs_stored_discovery"].astype(float) > 0.999).all()
    assert (probe["probe_n_nonzero"].astype(int) > 0).all()
    top = ref.loc[ref["ref_axis"].eq("top_gap")]
    assert (top["rank_by_abs_gap"].astype(int) == 1).all()


@needs_exp
def test_d26_full_run_artifacts():
    lev = _slug_table("d26_leverage_contrast.csv")
    chk = _slug_table("d26_refit_check.csv")
    if lev is None or chk is None:
        pytest.skip("D26 full run not finished yet")
    assert bool(chk["pass"].all())
    lo = lev.loc[lev["arm"].eq("leave_out")]
    assert (lo["n_label_matched_controls"] == 20).all() and (lo["n_random_controls"] == 3).all()
    assert len(lo) == 2 * 3 * 2 * 3 * 8  # panels x axes x methods x K x drop sets
