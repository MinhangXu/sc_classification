"""Tests for D28, cell-type-stratified score-energy shares per patient (W1.9).

Runner: scripts/mrd_stage0_2/workstreams/W1_patient_aggregation/run_d28_celltype_score_shares.py

Synthetic tests build cohorts whose answer is known by construction (each patient's cells of a cell
type are copies of one template block, so composition explains everything; then one patient is
made more dispersed). They also check the group-sum path against a brute-force per-cell
implementation and against the D16 module. The regression tests read the canonical experiment
read-only and are skipped when it is absent.
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
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fpm = _load("factor_patient_matrix", W1 / "factor_patient_matrix.py")
d28 = _load("run_d28_celltype_score_shares", W1 / "run_d28_celltype_score_shares.py")


# --------------------------------------------------------------------------------------
# synthetic cohorts
# --------------------------------------------------------------------------------------

TYPES = ("tA", "tB", "tC")
# copies of each type's template block per patient: very different mixes
COPIES = {"P02": (4, 1, 0), "P03": (1, 1, 1), "P04": (0, 2, 3), "P09": (1, 3, 1)}
BLOCK = 10  # rows per template block


def _templates(rng, k: int) -> dict:
    """Template score blocks; tA is far from the centre (high energy), tC close (low energy)."""
    return {"tA": rng.normal(3.0, 2.0, size=(BLOCK, k)), "tB": rng.normal(0.0, 1.0, size=(BLOCK, k)),
            "tC": rng.normal(-0.5, 0.3, size=(BLOCK, k))}


def _cohort_from_copies(templates: dict, copies: dict, malignant_frac: float = 0.2, scale: dict | None = None):
    rows, pats, types, mal = [], [], [], []
    for p, counts in copies.items():
        for t, c in zip(TYPES, counts):
            for _ in range(c):
                block = templates[t] * (scale or {}).get(p, 1.0)
                rows.append(block)
                pats += [p] * BLOCK
                types += [t] * BLOCK
                m = np.zeros(BLOCK, dtype=bool)
                m[: int(round(malignant_frac * BLOCK))] = True
                mal.append(m)
    return np.vstack(rows), np.array(pats), np.array(types), np.concatenate(mal)


@pytest.fixture(scope="module")
def composition_only():
    rng = np.random.default_rng(0)
    scores, pats, types, mal = _cohort_from_copies(_templates(rng, 4), COPIES)
    cells = d28.make_cells(pats, types, mal, supported=("P02", "P03", "P04", "P09"))
    return scores, cells


def _brute_force(scores, cells, unit_idx, *, scope="all", reference="cohort", weighting="ss_weighted"):
    """Per-cell reference implementation: replace each cell's energy by its type's reference average."""
    x = scores.astype(float)
    mu, sd = x.mean(axis=0), x.std(axis=0)
    keep = sd >= fpm.DEGENERATE_SD
    w = np.ones(keep.sum()) if weighting == "ss_weighted" else 1.0 / sd[keep] ** 2
    e = (((x[:, keep] - mu[keep]) ** 2) * w).sum(axis=1)
    in_scope = np.ones(x.shape[0], bool) if scope == "all" else ~cells.malignant
    in_unit = np.isin(cells.patient_code, unit_idx) & in_scope
    ref_pool = in_scope & (~np.isin(cells.patient_code, unit_idx) if reference == "loo" else True)
    expected = 0.0
    for i in np.flatnonzero(in_unit):
        same = ref_pool & (cells.celltype_code == cells.celltype_code[i])
        if not same.any():
            same = in_scope & (cells.celltype_code == cells.celltype_code[i])
        expected += e[same].mean()
    obs_share = e[in_unit].sum() / e[in_scope].sum()
    cell_share = in_unit.sum() / in_scope.sum()
    return obs_share / cell_share, (expected / e[in_scope].sum()) / cell_share, e[in_unit].sum() / expected


# --------------------------------------------------------------------------------------
# known answers
# --------------------------------------------------------------------------------------


def test_composition_explains_everything(composition_only):
    scores, cells = composition_only
    st = d28.group_sums(scores, cells)
    units = cells.units()
    assert set(units) == {"P02", "P03", "P04", "P09", d28.HP_UNIT}
    for weighting in d28.WEIGHTINGS:
        e = d28.energy(st, weighting)
        for reference in d28.REFERENCES:
            res = d28.allocate(e, st["n"], units, scope="all", strata="celltype", reference=reference)
            for unit, r in res.items():
                # every patient's cells of a type are copies of one template -> residual exactly 1
                assert r["resid_ratio"][0] == pytest.approx(1.0, abs=1e-12), (weighting, reference, unit)
                assert r["obs_ratio"][0] == pytest.approx(r["comp_ratio"][0], abs=1e-12)
    # the mixes differ and tA carries more energy, so the observed ratios are not 1
    r = d28.allocate(d28.energy(st, "ss_weighted"), st["n"], units, scope="all", strata="celltype", reference="cohort")
    assert r["P02"]["obs_ratio"][0] > 1.5 and r["P04"]["obs_ratio"][0] < 0.7


def test_patient_dispersion_survives_adjustment():
    rng = np.random.default_rng(1)
    t = _templates(rng, 3)
    # recentre the templates so that scaling P03 spreads its cells around the cohort centre
    pooled = np.vstack([t[k] for k in TYPES])
    t = {k: v - pooled.mean(axis=0) for k, v in t.items()}
    scores, pats, types, mal = _cohort_from_copies(t, COPIES, scale={"P03": 1.5})
    cells = d28.make_cells(pats, types, mal)
    st = d28.group_sums(scores, cells)
    res = d28.allocate(d28.energy(st, "ss_weighted"), st["n"], cells.units(), scope="all", strata="celltype",
                       reference="loo")
    assert res["P03"]["resid_ratio"][0] > 1.5  # P03's cells are more dispersed than same-type cells elsewhere
    for p in ("P02", "P04", "P09"):
        assert res[p]["resid_ratio"][0] < 1.0


def test_group_sums_match_brute_force():
    rng = np.random.default_rng(2)
    n = 600
    pats = rng.choice(["P02", "P03", "P04", "P07", "P09"], size=n, p=[0.1, 0.15, 0.1, 0.25, 0.4])
    types = rng.choice(list(TYPES) + ["tD"], size=n)
    types[pats == "P07"] = "tD"  # P07 only has tD cells
    types[(types == "tD") & (pats != "P07")] = "tA"  # so tD is P07-only -> loo fallback
    mal = rng.random(n) < 0.1
    scores = rng.normal(size=(n, 5)) * rng.uniform(0.5, 3, size=5)
    scores[types == "tA"] *= 1.7
    scores[:, 2] = 1.0  # degenerate factor
    cells = d28.make_cells(pats, types, mal)
    st = d28.group_sums(scores, cells)
    assert st["nondegenerate"].tolist() == [True, True, False, True, True]
    units = cells.units()
    for weighting in d28.WEIGHTINGS:
        e = d28.energy(st, weighting)
        for scope in ("all", "normal"):
            for reference in d28.REFERENCES:
                res = d28.allocate(e, st["n"], units, scope=scope, strata="celltype", reference=reference)
                for unit, idx in units.items():
                    obs, comp, resid = _brute_force(scores, cells, idx, scope=scope, reference=reference, weighting=weighting)
                    r = res[unit]
                    assert r["obs_ratio"][0] == pytest.approx(obs, rel=1e-10)
                    assert r["comp_ratio"][0] == pytest.approx(comp, rel=1e-10)
                    assert r["resid_ratio"][0] == pytest.approx(resid, rel=1e-10)
    # P07 holds every tD cell: its loo reference falls back to the cohort average for all its cells
    res = d28.allocate(d28.energy(st, "ss_weighted"), st["n"], units, scope="all", strata="celltype", reference="loo")
    assert res["P07"]["n_fallback_cells"] == int((pats == "P07").sum())
    assert res["P07"]["resid_ratio"][0] == pytest.approx(1.0)


def test_identities_and_d16_consistency():
    rng = np.random.default_rng(3)
    n = 800
    pats = rng.choice(["P01", "P02", "P03", "P04", "P09"], size=n)
    types = rng.choice(TYPES, size=n)
    mal = rng.random(n) < 0.15
    scores = rng.normal(size=(n, 6)) * np.arange(1, 7)
    scores[pats == "P01"] *= 1.3
    cells = d28.make_cells(pats, types, mal)
    st = d28.group_sums(scores, cells)
    key = {"stage0_panel_id": "panel_x", "short_panel_label": "panel x", "stage1_method": "pca", "k": 6}
    rp = d28.rep_patient_rows(key, st, cells)
    # 5 patients + the group, 2 weightings, 3 scope/strata variants, 2 references
    assert len(rp) == 6 * 2 * 3 * 2
    np.testing.assert_allclose(rp["obs_ratio"], rp["comp_ratio"] * rp["resid_ratio"], rtol=1e-12)
    coh = rp.loc[rp["reference"].eq("cohort") & ~rp["is_group"]]
    np.testing.assert_allclose(coh.groupby(["scope", "weighting", "strata"])["exp_share"].sum(), 1.0, rtol=1e-12)
    # observed ratios equal the D16 module's (all cells, normal cells, group, standardized = mean enrichment)
    cohort = fpm.Cohort(pats, mal, supported=(), high_prevalence=fpm.HIGH_PREVALENCE)
    long = fpm.allocation_table(scores, cohort)
    for c, v in key.items():
        long.insert(0, c, v)
    rps = fpm.representation_patient_summary(long).set_index("patient")
    prim = rp.loc[rp["weighting"].eq("ss_weighted") & rp["reference"].eq("cohort") & rp["strata"].eq("celltype")]
    a = prim.loc[prim["scope"].eq("all") & ~prim["is_group"]].set_index("unit")
    nrm = prim.loc[prim["scope"].eq("normal") & ~prim["is_group"]].set_index("unit")
    np.testing.assert_allclose(a["obs_ratio"], rps.loc[a.index, "weighted_enrichment"], rtol=1e-12)
    np.testing.assert_allclose(nrm["obs_ratio"], rps.loc[nrm.index, "weighted_enrichment_normal"], rtol=1e-12)
    std = rp.loc[rp["weighting"].eq("standardized") & rp["scope"].eq("all") & rp["reference"].eq("cohort")
                 & rp["strata"].eq("celltype") & ~rp["is_group"]].set_index("unit")
    np.testing.assert_allclose(std["obs_ratio"], long.groupby("patient")["enrichment"].mean().loc[std.index], rtol=1e-12)
    fsum = fpm.factor_summary(long)
    rsum = fpm.representation_summary(fsum, fpm.representation_patient_summary(long))
    grp = prim.loc[prim["is_group"] & prim["scope"].eq("all"), "obs_ratio"].iloc[0]
    assert grp == pytest.approx(rsum["hp_enrichment_weighted"].iloc[0], rel=1e-12)
    # per-factor observed enrichment equals D16's per-factor enrichment
    fl = d28.factor_rows({k: key[k] for k in ("stage0_panel_id", "stage1_method", "k")}, st, cells)
    f = fl.loc[fl["scope"].eq("all") & fl["unit"].ne(d28.HP_UNIT)].set_index(["factor_index", "unit"])
    lg = long.set_index(["factor_index", "patient"])["enrichment"]
    np.testing.assert_allclose(f["obs_enrichment"], lg.loc[f.index].to_numpy(), rtol=1e-12)
    # residual = expected-energy-weighted average of the within-type ratios; composition = mix x level
    ct = d28.celltype_rows({k: key[k] for k in ("stage0_panel_id", "stage1_method", "k")}, st, cells)
    for scope in ("all", "normal"):
        c = ct.loc[ct["scope"].eq(scope) & ct["reference"].eq("cohort")]
        w = (c["weight"] * c["within_ratio"]).groupby(c["unit"]).sum()
        m = (c["comp_share"] * c["type_energy_level"]).groupby(c["unit"]).sum()
        pr = prim.loc[prim["scope"].eq(scope)].set_index("unit")
        np.testing.assert_allclose(pr["resid_ratio"], w.loc[pr.index], rtol=1e-12)
        np.testing.assert_allclose(pr["comp_ratio"], m.loc[pr.index], rtol=1e-12)


def test_bad_inputs():
    cells = d28.make_cells(["P01", "P02"], ["a", "b"], [False, True])
    with pytest.raises(ValueError):
        d28.group_sums(np.zeros((3, 2)), cells)
    st = d28.group_sums(np.array([[0.0, 1.0], [1.0, 2.0]]), cells)
    with pytest.raises(ValueError):
        d28.allocate(d28.energy(st, "ss_weighted"), st["n"], cells.units(), scope="normal", strata="celltype_label",
                     reference="cohort")
    with pytest.raises(ValueError):
        d28.energy(st, "unknown")


# --------------------------------------------------------------------------------------
# runner end to end on a synthetic experiment dir (outputs under --out-root)
# --------------------------------------------------------------------------------------


def _write_fake_exp(root: Path, scores_by_unit: dict, patient, malignant, celltype) -> Path:
    exp = root / "exp_fake"
    for (panel, method, k), scores in scores_by_unit.items():
        d = exp / "stage1_dr" / f"panel_id={panel}" / f"method={method}" / f"k={k}" / "seed=42"
        d.mkdir(parents=True)
        np.save(d / "scores.npy", scores.astype(np.float32))
        (d / "metadata.json").write_text(json.dumps({"effective_k": k, "n_genes": 50}))
    meta_dir = exp / "stage2_supervised" / "multiobjective" / "discovery_full_cohort"
    meta_dir.mkdir(parents=True)
    pd.DataFrame({"cell_id": [f"c{i}" for i in range(len(patient))], "patient": patient,
                  "CN.label": np.where(malignant, "cancer", "normal"), "predicted.annotation": celltype}
                 ).to_csv(meta_dir / "cell_metadata.csv.gz", index=False)
    atlas = exp / "analysis" / "scorecards" / "stage2_diagnostic"
    atlas.mkdir(parents=True)
    panels = sorted({u[0] for u in scores_by_unit})
    pd.DataFrame({"stage0_panel_id": panels, "short_panel_label": [p.replace("_", " ") for p in panels],
                  "atlas_row_block": "biological", "n_covered_genes": 50}
                 ).to_csv(atlas / "old34_d1_atlas_representation_table.csv", index=False)
    return exp


def test_runner_resume_merge_and_out_root(tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    patient = np.repeat(["P01", "P02", "P03", "P04", "P09"], [120, 60, 60, 60, 100])
    malignant = np.zeros(400, dtype=bool)
    for p in ("P02", "P03", "P04", "P09"):
        malignant[np.flatnonzero(patient == p)[:12]] = True
    celltype = rng.choice(["tA", "tB", "tC"], size=400)
    units = {
        ("panel_a", "pca", 5): rng.normal(size=(400, 5)),
        ("panel_a", "fa", 5): np.column_stack([rng.normal(size=(400, 4)), np.zeros(400)]),
        ("panel_b", "pca", 5): rng.normal(size=(400, 5)),
    }
    exp = _write_fake_exp(tmp_path, units, patient, malignant, celltype)
    out = tmp_path / "scratch_out"
    monkeypatch.setattr(d28, "N_CELLS", 400)
    monkeypatch.setattr(d28, "study_config", lambda *a, **k: {})
    base = ["--experiment-dir", str(exp), "--out-root", str(out), "--panel-ids", "panel_a,panel_b", "--ks", "5,10",
            "--no-figures"]
    assert d28.main(base) == 0
    dirs = d28.out_dirs(out)
    parts = sorted(dirs["parts"].glob("*.npz"))
    assert len(parts) == 3
    # nothing was written into the fake EXP's analysis tree for this slug
    assert not (exp / "analysis" / "cache" / "stage2_diagnostic" / d28.SLUG).exists()
    # a second plain run refuses to overwrite finished parts
    assert d28.main(base) == 2
    # an empty part and a truncated part count as unfinished and are recomputed by --resume
    parts[0].write_bytes(b"")
    parts[1].write_bytes(parts[2].read_bytes()[:50])
    cells, _ = d28.load_cells(exp)
    assert d28.read_part(parts[0], cells) is None and d28.read_part(parts[1], cells) is None
    assert d28.main(base + ["--resume"]) == 0
    assert all(d28.read_part(p, cells) is not None for p in parts)
    assert d28.main(["--experiment-dir", str(exp), "--out-root", str(out), "--merge-only"]) == 0  # with figures
    td = dirs["tables"]
    rp = pd.read_csv(td / f"{d28.PREFIX}_representation_patient.csv.gz")
    assert len(rp) == 3 * 6 * 12  # 3 bases x (5 patients + group) x 12 variants
    checks = pd.read_csv(td / f"{d28.PREFIX}_regression_checks.csv")
    assert (checks.loc[checks["check_id"].str.startswith("identity"), "status"] == "pass").all()
    manifest = json.loads((td / f"{d28.PREFIX}_manifest.json").read_text())
    assert manifest["units"]["n_merged"] == 3
    assert manifest["units"]["missing_inputs"] == ["panel_b|fa|K5"]
    assert manifest["output_base"] == "<out-root>"
    ledger = pd.read_csv(td / f"{d28.PREFIX}_artifact_ledger.csv")
    assert not ledger["artifact_relpath"].str.startswith("/").any()
    assert ledger["kind"].eq("figure").sum() == 6  # D28A-C, png + pdf
    text = (td / f"{d28.PREFIX}_findings.md").read_text()
    assert "Residual ratio" in text and "not loading leverage" in text
    # a corrupt part makes --merge-only fail loudly and leaves merge_failed.json
    parts[2].write_bytes(b"")
    with pytest.raises(RuntimeError):
        d28.main(["--experiment-dir", str(exp), "--out-root", str(out), "--merge-only", "--no-figures"])
    assert (dirs["cache"] / "merge_failed.json").is_file()


# --------------------------------------------------------------------------------------
# regression against the canonical experiment (skipped without EXP)
# --------------------------------------------------------------------------------------

EXP = d28.EXP_DEFAULT
HAS_EXP = (EXP / d28.CELL_METADATA_DEFAULT).is_file() and (
    d28.d16.score_dir(EXP, d28.FOCUS_PANEL, "pca", 40) / "scores.npy"
).is_file()
needs_exp = pytest.mark.skipif(not HAS_EXP, reason="canonical experiment dir not available")


@pytest.fixture(scope="module")
def exp_cells():
    meta = pd.read_csv(EXP / d28.CELL_METADATA_DEFAULT, usecols=["patient", "CN.label", d28.CELLTYPE_COLUMN])
    return d28.make_cells(meta["patient"], meta[d28.CELLTYPE_COLUMN], meta["CN.label"].eq("cancer"))


@needs_exp
def test_regression_ifna_k40_against_d16(exp_cells):
    rps_path = EXP / d28.D16_TABLES / "factor_patient_matrix_representation_patient_summary.csv"
    rs_path = EXP / d28.D16_TABLES / "factor_patient_matrix_representation_summary.csv"
    if not rps_path.is_file() or not rs_path.is_file():
        pytest.skip("D16 tables absent")
    rps = pd.read_csv(rps_path)
    rs = pd.read_csv(rs_path)
    for method, hp in (("pca", 1.11), ("fa", 1.02)):
        scores = np.load(d28.d16.score_dir(EXP, d28.FOCUS_PANEL, method, 40) / "scores.npy")
        st = d28.group_sums(scores, exp_cells)
        key = {"stage0_panel_id": d28.FOCUS_PANEL, "short_panel_label": "ifna", "stage1_method": method, "k": 40}
        rp = d28.rep_patient_rows(key, st, exp_cells)
        prim = rp.loc[rp["weighting"].eq("ss_weighted") & rp["reference"].eq("cohort") & rp["strata"].eq("celltype")]
        ref = rps.loc[rps["stage0_panel_id"].eq(d28.FOCUS_PANEL) & rps["stage1_method"].eq(method) & rps["k"].eq(40)]
        ref = ref.set_index("patient")
        for scope, col in (("all", "weighted_enrichment"), ("normal", "weighted_enrichment_normal")):
            ours = prim.loc[prim["scope"].eq(scope) & ~prim["is_group"]].set_index("unit")["obs_ratio"]
            assert len(ours) == 13
            np.testing.assert_allclose(ours, ref.loc[ours.index, col], atol=1e-9)
        grp = prim.loc[prim["scope"].eq("all") & prim["is_group"], "obs_ratio"].iloc[0]
        r = rs.loc[rs["stage0_panel_id"].eq(d28.FOCUS_PANEL) & rs["stage1_method"].eq(method) & rs["k"].eq(40)]
        assert grp == pytest.approx(r["hp_enrichment_weighted"].iloc[0], abs=1e-9)
        assert abs(grp - hp) <= d28.ROUND_TOL  # the D16 devlog's rounded headline
        coh = prim.loc[~prim["is_group"]].groupby("scope")["exp_share"].sum()
        np.testing.assert_allclose(coh, 1.0, atol=1e-12)


@needs_exp
def test_saved_outputs_consistent():
    td = d28.out_dirs(EXP / "analysis")["tables"]
    checks_path = td / f"{d28.PREFIX}_regression_checks.csv"
    if not checks_path.is_file():
        pytest.skip("D28 outputs not written yet")
    checks = pd.read_csv(checks_path)
    assert (checks["status"] == "pass").all(), checks.loc[checks["status"] != "pass"]
    manifest = json.loads((td / f"{d28.PREFIX}_manifest.json").read_text())
    assert manifest["units"]["n_merged"] == 320
    dist = pd.read_csv(td / f"{d28.PREFIX}_distribution_summary.csv")
    p01 = dist.loc[dist["unit"].eq("P01") & dist["scope"].eq("all") & dist["weighting"].eq("ss_weighted")
                   & dist["reference"].eq("cohort") & dist["strata"].eq("celltype")].iloc[0]
    assert abs(p01["obs_ratio_median"] - 1.24) <= d28.ROUND_TOL  # D16 devlog headline
    assert p01["n_representations"] == 320
