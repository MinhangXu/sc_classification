"""Tests for the read-only factor explorer (scripts/mrd_stage0_2/apps/old34_factor_explorer.py, W7.2).

Every test except the last two uses a tiny synthetic factor_cards_v1 bundle that follows the bundle
schema (schema_version 1).  The real-bundle tests are skipped when the D18 bundle is not present.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

APPS = Path(__file__).resolve().parents[1] / "scripts" / "mrd_stage0_2" / "apps"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, APPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fx = load_module("old34_factor_explorer")

# --------------------------------------------------------------------------------------
# Synthetic bundle
# --------------------------------------------------------------------------------------

SUPPORTED = ["P01", "P02", "P03"]
UNSUPPORTED = ["P04", "P05"]
FOLDS = SUPPORTED + UNSUPPORTED
PANEL_A = "single_geneset__alpha"
PANEL_B = "hvg_top_requested_500__available_500"
OUTSIDE = "single_geneset__omega__pca__k40__f001"  # matched card that is not in the index
PREVALENCE = {"P01": 0.01, "P02": 0.20, "P03": 0.30}

# (panel, method, K) -> factor -> spec
CARDS = {
    (PANEL_A, "pca", 5): {
        1: dict(sign=-1, status="oriented", coefs={"P02": -0.5, "P03": -0.3, "P04": -0.2}, driver=True, lift=[0.0, 0.15, 0.10]),
        2: dict(sign=0, status="unoriented_all_zero", coefs={}, driver=False, lift=[0.0, 0.08, 0.01]),
        3: dict(sign=0, status="unoriented_sign_flip", coefs={"P02": 0.2, "P03": -0.1}, driver=True, lift=[0.0, 0.02, -0.01]),
        4: dict(sign=0, status="unoriented_all_zero", coefs={}, driver=False, lift=None, degenerate=True),
        5: dict(sign=1, status="oriented", coefs={"P02": 0.4}, driver=True, lift=[0.0, 0.05, 0.02]),
    },
    (PANEL_A, "fa", 5): {
        f: dict(sign=1, status="oriented", coefs={"P01": 0.1 * f, "P02": 0.1 * f, "P03": 0.1 * f}, driver=True, lift=[0.01, 0.02, 0.03],
                determinacy=0.9 - 0.1 * f)
        for f in range(1, 6)
    },
    (PANEL_B, "factosig_promax", 3): {
        f: dict(sign=1, status="oriented", coefs={"P01": 0.3, "P02": 0.3, "P03": 0.3}, driver=False, lift=[0.0, 0.01, 0.0], promax=True)
        for f in range(1, 4)
    },
}
ZERO_FOLDS = {(PANEL_A, "pca", 5): {"P01"}}  # the whole PCA model is all-zero when P01 is held out


def _flags(spec: dict, rep) -> str:
    flags = ["transductive_basis", "ablation_is_attribution"]
    if spec["status"] != "oriented":
        flags.append(spec["status"])
    if spec["status"] == "unoriented_all_zero":
        flags.append("l1_selected_absent")
    if ZERO_FOLDS.get(rep):
        flags.append("has_all_zero_folds")
    if spec.get("degenerate"):
        flags.append("zero_variance")
    if spec.get("determinacy", 1.0) < 0.707:
        flags.append("low_determinacy")
    return ";".join(flags)


def build_bundle(root: Path, *, experiment_layout: bool = True, mutate=None) -> Path:
    """Write a schema-v1 bundle under root (experiment layout) or flat in root; returns root."""
    out = root / fx.BUNDLE_SUBDIR if experiment_layout else root
    out.mkdir(parents=True, exist_ok=True)
    index, genes, lift, folds, score, adt, celltype, phi = [], [], [], [], [], [], [], []
    for rep, factors in CARDS.items():
        panel, method, k = rep
        rep_id = fx.representation_id(panel, method, k)
        for f, spec in factors.items():
            cid = fx.card_id_for(panel, method, k, f)
            active = [p for p in SUPPORTED if spec["coefs"].get(p, 0) != 0]
            signs = {np.sign(spec["coefs"][p]) for p in active}
            matched = [fx.card_id_for(PANEL_A, "fa", 5, 1), OUTSIDE] if cid == fx.card_id_for(PANEL_A, "pca", 5, 1) else []
            lifts = spec["lift"] or [np.nan] * 3
            index.append(dict(
                card_id=cid, panel_id=panel, method=method, k=k, factor=f, is_selected_driver=spec["driver"],
                n_active_folds=len(active), sign_consistency=(1.0 if len(signs) == 1 else 0.5) if active else np.nan,
                orientation_sign=spec["sign"], eta2_normal=0.01 * f, eta2_malignant=0.02 * f,
                degenerate_flag=bool(spec.get("degenerate")), effective_k=k - (1 if rep == (PANEL_A, "pca", 5) else 0),
                determinacy=spec.get("determinacy", np.nan), matched_cards=";".join(matched), caveat_flags=_flags(spec, rep),
                representation_id=rep_id, panel_label=f"label {panel[-5:]}", n_genes=50, shortlist_reasons="synthetic",
                orientation_status=spec["status"], n_folds=len(SUPPORTED), fit_penalty="l1", fit_C=1e-4, fit_dense=method == "fa",
                n_all_zero_folds=len(ZERO_FOLDS.get(rep, ())), max_univariate_lift=max(lifts) if spec["lift"] else np.nan,
                median_univariate_lift=float(np.median(lifts)) if spec["lift"] else np.nan,
                n_patients_univariate_lift_ge_0p05=sum(v >= 0.05 for v in spec["lift"] or []),
                score_sd=1.0, eigengap_rel=0.2, loading_semantics="pca_component_weights" if method == "pca" else "generative",
                adt_isotype_max_abs_rho_normal=0.05, adt_isotype_max_abs_rho_malignant=0.09, card_md=f"cards/{cid}.md",
            ))
            if not spec.get("degenerate"):
                oriented = spec["sign"] != 0
                for ranked_by in ("loading", "structure") if spec.get("promax") else ("loading",):
                    for pole, sign in (("malignant", 1), ("normal", -1)) if oriented else (("positive", 1), ("negative", -1)):
                        for rank in (1, 2, 3):
                            loading = spec["sign"] * sign * (1.0 - 0.1 * rank) if oriented else sign * (1.0 - 0.1 * rank)
                            genes.append(dict(
                                card_id=cid, gene=f"{pole[:3].upper()}{f}{rank}", loading=loading,
                                oriented_loading=spec["sign"] * loading if oriented else np.nan, rank=rank, pole=pole,
                                structure_corr=0.5 * sign if spec.get("promax") else np.nan, ranked_by=ranked_by,
                            ))
                for j, p in enumerate(SUPPORTED):
                    lift.append(dict(
                        card_id=cid, patient=p, univariate_lift=lifts[j], univariate_auroc=0.5 + lifts[j],
                        n_malignant=[12, 200, 700][j], prevalence=PREVALENCE[p], univariate_lift_sign_agnostic=abs(lifts[j]) + 0.01,
                        fitted_single_factor_lift=lifts[j] / 2, fitted_ap_drop=lifts[j] / 3, fitted_coef=spec["coefs"].get(p, 0.0),
                    ))
                    for label in ("normal", "malignant"):
                        q = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) + (0.5 if label == "malignant" else 0.0)
                        score.append(dict(card_id=cid, factor=f, patient=p, label=label, n_cells=100, oriented=spec["sign"] != 0,
                                          mean=q[2], sd=1.0, q10=q[0], q25=q[1], q50=q[2], q75=q[3], q90=q[4]))
                for label in ("normal", "malignant"):
                    for rank, marker in enumerate(("ADT-CD34", "ADT-CD99"), 1):
                        adt.append(dict(card_id=cid, label=label, rank=rank, adt_marker=marker, is_isotype=False,
                                        spearman_pooled=0.3 / rank, spearman_patient_stratified=0.25 / rank, n_cells=300, oriented=True))
                    for t, ct in enumerate(("HSC", "GMP", "B cell")):
                        celltype.append(dict(card_id=cid, factor=f, label=label, cell_type=ct, n_cells=50, mean_oriented_z=0.4 - 0.4 * t))
            for p in FOLDS:
                coef = spec["coefs"].get(p, 0.0)
                zero = p in ZERO_FOLDS.get(rep, set())
                folds.append(dict(
                    card_id=cid, heldout_patient=p, penalty="l1", C=1e-4, l1_ratio=np.nan, coef=coef, is_nonzero=coef != 0,
                    supported_fold=p in SUPPORTED, fold_all_zero=zero, fold_n_active=0 if zero else 2,
                    fold_model_lift=0.0 if zero else 0.05,
                ))
        if method == "factosig_promax":
            for i in range(1, k + 1):
                for j in range(i + 1, k + 1):
                    phi.append(dict(representation_id=rep_id, factor_i=i, factor_j=j, score_r=0.2))
    a1, fa1 = fx.card_id_for(PANEL_A, "pca", 5, 1), fx.card_id_for(PANEL_A, "fa", 5, 1)
    matches = [
        dict(card_id=a1, factor=1, relation="hungarian_same_k", other_method="fa", other_k=5, matched_card_id=fa1, score_r=-0.8,
             abs_r=0.8, r2_many_to_one=0.9, angle_deg=20.0, tucker_phi_aligned=0.85,
             other_representation_id=fx.representation_id(PANEL_A, "fa", 5)),
        dict(card_id=a1, factor=1, relation="best_abs_r", other_method="pca", other_k=40, matched_card_id=OUTSIDE, score_r=0.6,
             abs_r=0.6, r2_many_to_one=0.7, angle_deg=40.0, tucker_phi_aligned=0.6, other_representation_id="single_geneset__omega|dr|pca|40|42"),
    ]
    tables = {
        "factor_cards_index.csv": pd.DataFrame(index), "factor_cards_genes.csv": pd.DataFrame(genes),
        "factor_cards_patient_lift.csv": pd.DataFrame(lift), "factor_cards_fold_coefs.csv": pd.DataFrame(folds),
        "factor_cards_matches.csv": pd.DataFrame(matches), "factor_cards_score_dist.csv": pd.DataFrame(score),
        "factor_cards_adt.csv": pd.DataFrame(adt), "factor_cards_celltype.csv": pd.DataFrame(celltype),
        "factor_cards_phi.csv": pd.DataFrame(phi),
        "factor_cards_shortlist.csv": pd.DataFrame({"representation_id": [fx.representation_id(*rep) for rep in CARDS]}),
        "factor_cards_regression_checks.csv": pd.DataFrame({"check_id": ["synthetic"], "description": ["x"], "passed": [True]}),
    }
    if mutate:
        mutate(tables)
    counts = {name: len(frame) for name, frame in tables.items() if name not in ("factor_cards_shortlist.csv", "factor_cards_regression_checks.csv")}
    counts.update(n_cards=len(tables["factor_cards_index.csv"]), n_representations=len(CARDS))
    manifest = dict(
        schema_version=1, slug="factor_cards_v1", inputs={}, git_state={"head": "abc1234", "dirty": True},
        created_by="scripts/mrd_stage0_2/workstreams/W5_factor_interpretation/run_d18_factor_cards.py",
        created_utc="2026-10-08T00:00:00+00:00", parameters={"supported_patients": SUPPORTED, "c_policy": "global_selected"},
        counts=counts, runtime_seconds=1.0,
    )
    if mutate:
        mutate(manifest)
    for name, frame in tables.items():
        frame.to_csv(out / name, index=False)
    (out / fx.MANIFEST).write_text(json.dumps(manifest, indent=2))
    (out / fx.SCHEMA_DOC).write_text("# factor_cards_v1 bundle schema (schema_version 1)\n")
    cards_dir = out / "cards"
    cards_dir.mkdir(exist_ok=True)
    (cards_dir / f"{a1}.md").write_text("# not read by the app\n")
    return root


@pytest.fixture()
def exp_dir(tmp_path):
    return build_bundle(tmp_path / "exp")


@pytest.fixture()
def bundle(exp_dir):
    return fx.CardBundle(exp_dir)


A1 = fx.card_id_for(PANEL_A, "pca", 5, 1)


def _snapshot(root: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------------------
# Loading and navigation
# --------------------------------------------------------------------------------------


def test_only_the_index_is_read_at_start_then_tables_load_lazily(bundle, exp_dir):
    assert bundle.mode == "experiment" and bundle.root == exp_dir / fx.BUNDLE_SUBDIR
    assert bundle.loaded == ["index"]
    assert len(bundle.index) == sum(len(v) for v in CARDS.values()) and len(bundle.reps) == 3
    assert bundle.reps[(PANEL_A, "pca", 5)] == [fx.card_id_for(PANEL_A, "pca", 5, f) for f in range(1, 6)]
    fx.gene_poles(bundle, A1)
    assert bundle.loaded == ["index", "genes"]
    assert bundle.supported_patients == SUPPORTED
    assert bundle.methods_for(PANEL_A) == ["pca", "fa"] and bundle.ks_for(PANEL_B, "factosig_promax") == [3]


def test_navigation_resolves_deep_links_and_reports_uncovered_representations(bundle):
    nav = fx.navigate(bundle, PANEL_A, "pca", "5")
    assert nav.status == "ok" and nav.rep == (PANEL_A, "pca", 5) and nav.card_id is None
    assert fx.navigate(bundle, PANEL_A, "pca", 5, "f001").card_id == A1
    raw = fx.parse_search(f"?card={A1}")
    assert fx.navigate(bundle, raw["panel"], raw["method"], raw["k"], raw["factor"]).card_id == A1
    missing_factor = fx.navigate(bundle, PANEL_A, "pca", 5, 9)
    assert missing_factor.status == "ok" and missing_factor.factor is None and "factor 9" in missing_factor.message
    uncovered = fx.navigate(bundle, PANEL_A, "pca", 40)
    assert uncovered.status == "not_covered" and "not in the D18 shortlist" in uncovered.message
    assert fx.navigate(bundle, PANEL_A, "factosig", 5).status == "not_covered"
    relaxed = fx.navigate(bundle, PANEL_A, "factosig", 40, strict=False)  # dropdowns: first carded representation
    assert relaxed.status == "ok" and relaxed.rep == (PANEL_A, "pca", 5)
    assert fx.navigate(bundle, PANEL_A).rep == (PANEL_A, "pca", 5)
    assert fx.navigate(bundle, "nope", "pca", 5).status == "unknown_panel"
    assert fx.navigate(bundle, PANEL_A, "pca", "abc").status == "invalid"
    assert fx.navigate(bundle, None).status == "home" and fx.Nav().query() == ""
    query = fx.navigate(bundle, PANEL_A, "pca", 5, 1).query()
    assert query == f"?panel={PANEL_A}&method=pca&k=5&factor=1"
    assert fx.navigate(bundle, **{k: v for k, v in fx.parse_search(query).items()}).card_id == A1
    assert fx.parse_search("?panel=a+b&method=fa&k=40") == {"panel": "a b", "method": "fa", "k": "40"}


def test_atlas_deep_link_round_trip(bundle):
    atlas_app = load_module("old34_d1_atlas_explorer")
    href = atlas_app.factor_explorer_href(PANEL_A, "pca|K=5", "http://127.0.0.1:8051/")
    assert href.startswith("http://127.0.0.1:8051/?")
    raw = fx.parse_search(href.split("/", 3)[3].lstrip("/"))
    assert fx.navigate(bundle, raw["panel"], raw["method"], raw["k"], raw.get("factor")).rep == (PANEL_A, "pca", 5)


def test_missing_bundle_and_wrong_schema_give_actionable_errors(tmp_path):
    with pytest.raises(fx.DataError, match="factor_cards_v1 manifest not found"):
        fx.CardBundle(tmp_path)
    with pytest.raises(fx.DataError, match="unsupported schema_version 2"):
        fx.CardBundle(build_bundle(tmp_path / "v2", mutate=lambda t: t.update(schema_version=2) if "schema_version" in t else None))

    def drop_column(tables):
        if "factor_cards_index.csv" in tables:
            tables["factor_cards_index.csv"] = tables["factor_cards_index.csv"].drop(columns="caveat_flags")

    with pytest.raises(fx.DataError, match="missing required columns"):
        fx.CardBundle(build_bundle(tmp_path / "nocol", mutate=drop_column))


# --------------------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------------------


def test_representation_table_lists_factors_and_filters(bundle):
    frame = fx.representation_table(bundle, PANEL_A, "pca", 5)
    assert frame["factor"].tolist() == [1, 2, 3, 4, 5]
    assert frame["orientation"].tolist() == ["-", "0", "0", "0", "+"]
    assert frame.loc[0, "active folds"] == "2/3" and frame.loc[0, "max lift pp"] == pytest.approx(15.0)
    assert frame.loc[0, "patients >= 5 pp"] == 2 and bool(frame.loc[3, "degenerate"])
    assert fx.representation_table(bundle, PANEL_A, "pca", 5, drivers_only=True)["factor"].tolist() == [1, 3, 5]
    assert fx.representation_table(bundle, PANEL_A, "pca", 5, hide_degenerate=True)["factor"].tolist() == [1, 2, 3, 5]
    head = fx.representation_header(bundle, PANEL_A, "pca", 5)
    assert head["n_cards"] == 5 and head["n_selected_drivers"] == 3 and head["fit_penalty"] == "l1" and head["effective_k"] == 4


def test_fold_strip_orders_folds_and_badges_all_zero_supported_folds(bundle):
    strip = fx.fold_strip(bundle, A1)
    assert [s["patient"] for s in strip] == FOLDS and [s["supported"] for s in strip] == [True] * 3 + [False] * 2
    status = {s["patient"]: s["status"] for s in strip}
    assert status == {"P01": "model_zero", "P02": "negative", "P03": "negative", "P04": "negative", "P05": "not_selected"}
    assert [s["patient"] for s in strip if s["badge"]] == ["P01"] and "lift = 0 by construction" in strip[0]["title"]
    flip = {s["patient"]: s["status"] for s in fx.fold_strip(bundle, fx.card_id_for(PANEL_A, "pca", 5, 3))}
    assert flip["P02"] == "positive" and flip["P03"] == "negative"
    assert "unsupported" in strip[3]["title"]


def test_gene_poles_follow_orientation_and_promax_structure(bundle):
    poles = fx.gene_poles(bundle, A1)
    assert list(poles) == ["malignant pole (ranked by loading)", "normal pole (ranked by loading)"]
    assert poles["malignant pole (ranked by loading)"]["gene"].tolist() == ["MAL11", "MAL12", "MAL13"]
    assert (poles["malignant pole (ranked by loading)"]["oriented_loading"] > 0).all()
    unoriented = fx.gene_poles(bundle, fx.card_id_for(PANEL_A, "pca", 5, 2))
    assert list(unoriented) == ["positive pole (ranked by loading)", "negative pole (ranked by loading)"]
    assert fx.gene_poles(bundle, fx.card_id_for(PANEL_A, "pca", 5, 4)) == {}  # degenerate: no genes
    promax = fx.gene_poles(bundle, fx.card_id_for(PANEL_B, "factosig_promax", 3, 1))
    assert "malignant pole (ranked by structure)" in promax and "structure_corr" in promax["malignant pole (ranked by structure)"].columns


def test_card_chips_and_caveat_badges(bundle):
    chips = dict(fx.card_chips(bundle, A1))
    assert chips["orientation"].startswith("oriented, beta sign - (malignant pole = negative raw scores)")
    assert chips["active supported folds"] == "2/3" and chips["max univariate lift"] == "+15.00 pp"
    unoriented = dict(fx.card_chips(bundle, fx.card_id_for(PANEL_A, "pca", 5, 3)))
    assert unoriented["orientation"] == "unoriented: beta sign flips across supported folds"
    assert "degenerate" in dict(fx.card_chips(bundle, fx.card_id_for(PANEL_A, "pca", 5, 4)))
    badges = dict(fx.caveat_badges("transductive_basis;has_all_zero_folds;made_up_code"))
    assert "label transfer" in badges["transductive_basis"] and "0 by construction" in badges["has_all_zero_folds"]
    assert badges["made_up_code"] == "(code not in the v1 vocabulary)"
    assert set(fx.CAVEATS) >= {c for v in bundle.index["caveat_flags"] for c in fx.split_list(v)}


def test_patient_lift_and_score_figures(bundle):
    fig = fx.patient_lift_figure(bundle, A1)
    names = [t.name for t in fig.data]
    assert names[0].startswith("univariate lift") and any("sign-agnostic" in n for n in names) and any("D8" in n for n in names)
    assert list(fig.data[0].y) == pytest.approx([0.0, 15.0, 10.0])
    assert "P02<br>π 20.0%<br>200 mal" in list(fig.data[0].x)
    degenerate = fx.patient_lift_figure(bundle, fx.card_id_for(PANEL_A, "pca", 5, 4))
    assert len(degenerate.data) == 0
    box = fx.score_dist_figure(bundle, A1)
    assert [t.name for t in box.data] == ["normal cells", "malignant cells"] and list(box.data[1].median) == [0.5] * 3
    assert "oriented score" in box.layout.yaxis.title.text
    assert "raw score" in fx.score_dist_figure(bundle, fx.card_id_for(PANEL_A, "pca", 5, 2)).layout.yaxis.title.text


def test_matches_link_only_cards_in_this_bundle_and_side_tables(bundle):
    rows = fx.match_rows(bundle, A1)
    assert [(r["card"], r["linked"]) for r in rows] == [(fx.card_id_for(PANEL_A, "fa", 5, 1), True), (OUTSIDE, False)]
    assert fx.card_href(OUTSIDE) == "?panel=single_geneset__omega&method=pca&k=40&factor=1" and fx.card_href("bad") == ""
    adt = fx.adt_tables(bundle, A1)
    assert list(adt) == ["normal", "malignant"] and adt["normal"]["adt_marker"].tolist() == ["ADT-CD34", "ADT-CD99"]
    cts = fx.celltype_table(bundle, A1)
    assert cts["malignant"]["cell_type"].tolist()[0] == "HSC"


# --------------------------------------------------------------------------------------
# --check, export, read-only
# --------------------------------------------------------------------------------------


def _check(root: Path):
    bundle = fx.CardBundle(root)
    bundle_root, _ = fx.resolve_bundle_root(root)
    before = fx.bundle_fingerprint(bundle_root)
    bundle.load_all()
    return fx.run_checks(bundle, before, fx.bundle_fingerprint(bundle_root))


def test_check_passes_on_a_schema_conforming_bundle(exp_dir):
    rows = _check(exp_dir)
    status = {name: s for name, s, _ in rows}
    assert not [r for r in rows if r[1] == "FAIL"], rows
    assert status["IFN-alpha PC8 / FA7 anchor cards"] == "SKIP"  # experiment-specific anchors are optional
    assert status["deep links resolve (?panel=&method=&k=)"] == "PASS" and status["views render (first card of every representation)"] == "PASS"
    assert rows[-1][0] == "inputs unmodified by loading" and rows[-1][1] == "PASS"


@pytest.mark.parametrize(
    "name, mutate, failing",
    [
        ("count", lambda t: t.get("counts", {}).update({"factor_cards_genes.csv": 1}) if "counts" in t else None, "row counts match the manifest"),
        ("orphan", lambda t: t.update({"factor_cards_genes.csv": pd.concat([t["factor_cards_genes.csv"], t["factor_cards_genes.csv"].head(1).assign(card_id="x__pca__k5__f001")])}) if "factor_cards_genes.csv" in t else None,
         "per-card tables reference indexed cards only"),
        ("bad_id", lambda t: t.update({"factor_cards_index.csv": t["factor_cards_index.csv"].assign(factor=lambda f: f["factor"].where(f.index != 0, 2))}) if "factor_cards_index.csv" in t else None,
         "index card ids, factors and orientation"),
        ("producer", lambda t: t.update({"factor_cards_regression_checks.csv": t["factor_cards_regression_checks.csv"].assign(passed=False)}) if "factor_cards_regression_checks.csv" in t else None,
         "producer regression checks passed"),
        ("patients", lambda t: t.update({"factor_cards_patient_lift.csv": t["factor_cards_patient_lift.csv"].assign(patient=lambda f: f["patient"].replace("P03", "P09"))}) if "factor_cards_patient_lift.csv" in t else None,
         "patient lift covers supported patients only"),
    ],
)
def test_check_catches_inconsistent_bundles(tmp_path, name, mutate, failing):
    rows = {n: (s, d) for n, s, d in _check(build_bundle(tmp_path / name, mutate=mutate))}
    assert rows[failing][0] == "FAIL", rows[failing]


def test_export_bundle_gzip_round_trip_and_limits(exp_dir, tmp_path):
    with pytest.raises(fx.DataError, match="inside the data directory"):
        fx.export_bundle(exp_dir, exp_dir / "analysis" / "out")
    with pytest.raises(fx.DataError, match="above the"):
        fx.export_bundle(exp_dir, tmp_path / "tiny", limit_bytes=100)
    assert not (tmp_path / "tiny").exists()
    before = _snapshot(exp_dir)
    out = tmp_path / "shared"
    written = fx.export_bundle(exp_dir, out)
    names = {p.name for p in written}
    assert fx.MANIFEST in names and fx.EXPORT_MANIFEST in names and "factor_cards_index.csv.gz" in names
    assert not any(n.endswith(".md") and n != fx.SCHEMA_DOC for n in names) and not (out / "cards").exists()
    assert sum(p.stat().st_size for p in written) <= fx.EXPORT_LIMIT_BYTES
    export = json.loads((out / fx.EXPORT_MANIFEST).read_text())
    assert export["schema_version"] == 1 and "/" not in export["source"].split(" of experiment ")[-1]
    raw = gzip.decompress((out / "factor_cards_genes.csv.gz").read_bytes())
    assert raw == (exp_dir / fx.BUNDLE_SUBDIR / "factor_cards_genes.csv").read_bytes()
    assert _snapshot(exp_dir) == before
    with pytest.raises(fx.DataError, match="refusing to overwrite"):
        fx.export_bundle(exp_dir, out)
    shared = fx.CardBundle(out)
    assert shared.mode == "bundle" and shared.index.equals(fx.CardBundle(exp_dir).index)
    assert not [r for r in _check(out) if r[1] == "FAIL"]
    again = fx.export_bundle(out, tmp_path / "again")  # an exported bundle exports again (gz in, gz out)
    assert {p.name for p in again} == names


def test_flat_bundle_layout_and_read_only_rendering(tmp_path):
    flat = build_bundle(tmp_path / "flat", experiment_layout=False)
    before = _snapshot(flat)
    bundle = fx.CardBundle(flat)
    assert bundle.mode == "bundle"
    for rep, cards in bundle.reps.items():
        fx.representation_table(bundle, *rep)
        for cid in cards:
            fx.card_chips(bundle, cid), fx.fold_strip(bundle, cid), fx.gene_poles(bundle, cid), fx.patient_lift_figure(bundle, cid)
            fx.score_dist_figure(bundle, cid), fx.match_rows(bundle, cid), fx.adt_tables(bundle, cid), fx.celltype_table(bundle, cid)
    assert _snapshot(flat) == before


def test_app_does_not_import_the_atlas_app():
    source = (APPS / "old34_factor_explorer.py").read_text()
    assert "import old34_d1_atlas_explorer" not in source and "from old34_d1_atlas_explorer" not in source
    # no absolute or home-relative paths (AGENTS R9); the pattern is split so lint E07 does not flag this line
    assert not re.search(r"/(home|Users)/|~" r"/", source)


# --------------------------------------------------------------------------------------
# Dash wiring (needs dash; exercised through Flask's test client, no server)
# --------------------------------------------------------------------------------------


def _route(app, search="", panel=None, method=None, k=None, factor=None, changed="url.search", drivers=(), degenerate=()):
    key = next(k_ for k_ in app.callback_map if "content.children" in k_)
    parts = key.strip(".").split("...")
    payload = {
        "output": key,
        "outputs": [{"id": p.rsplit(".", 1)[0], "property": p.rsplit(".", 1)[1]} for p in parts],
        "inputs": [
            {"id": "url", "property": "search", "value": search}, {"id": "panel", "property": "value", "value": panel},
            {"id": "method", "property": "value", "value": method}, {"id": "k", "property": "value", "value": k},
            {"id": "factor", "property": "value", "value": factor},
            {"id": "drivers-only", "property": "value", "value": list(drivers)},
            {"id": "hide-degenerate", "property": "value", "value": list(degenerate)},
        ],
        "state": [], "changedPropIds": [changed] if changed else [],
    }
    response = app.server.test_client().post("/_dash-update-component", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200
    return response.get_json()["response"]


def test_dash_layout_has_the_persistent_caveat_banner(bundle):
    pytest.importorskip("dash")
    app = fx.create_app(bundle)
    banner = app.layout["banner"]
    text = json.dumps(banner.to_plotly_json(), default=str)
    assert "transductive" in text and "label transfer, not inductive deployment" in text and "not absence of signal" in text
    page = app.server.test_client().get("/")
    assert page.status_code == 200 and b"Old34 factor explorer" in page.data


def test_dash_deep_link_card_view_and_dropdown_sync(bundle):
    pytest.importorskip("dash")
    app = fx.create_app(bundle)
    body = _route(app, f"?panel={PANEL_A}&method=pca&k=5")
    assert body["k"]["value"] == 5 and body["method"]["value"] == "pca" and body["factor"]["value"] is None
    text = json.dumps(body["content"])
    assert "Factors (5 shown)" in text and "permalink to this view" in text
    assert [o["value"] for o in body["method"]["options"]] == ["pca", "fa"]
    body = _route(app, f"?panel={PANEL_A}&method=pca&k=5&factor=1")
    text = json.dumps(body["content"])
    assert "MAL11" in text and "NOR11" in text and "lift = 0 by construction: P01" in text and OUTSIDE in text
    assert "Factor 1 of PCA K=5" in text and "transductive_basis" in text
    uncovered = _route(app, f"?panel={PANEL_A}&method=pca&k=40")
    text = json.dumps(uncovered["content"])
    assert "not in the D18 shortlist" in text and "FA K=5 (5 cards)" in text and uncovered["k"]["value"] is None
    assert "has no D18 factor cards" in json.dumps(_route(app, "?panel=nope")["content"])
    moved = _route(app, f"?panel={PANEL_A}&method=pca&k=5&factor=1", panel=PANEL_B, method="pca", k=5, factor=1, changed="panel.value")
    assert moved["url"]["search"] == f"?panel={PANEL_B}&method=factosig_promax&k=3" and moved["factor"]["value"] is None
    filtered = _route(app, f"?panel={PANEL_A}&method=pca&k=5", panel=PANEL_A, method="pca", k=5, changed="drivers-only.value", drivers=["on"])
    assert "Factors (3 shown)" in json.dumps(filtered["content"])
    home = _route(app, "")
    assert home["url"]["search"] == "" and "Pick a panel" in json.dumps(home["content"])


# --------------------------------------------------------------------------------------
# Real bundle (skipped until D18 has landed)
# --------------------------------------------------------------------------------------


def _real_root() -> Path | None:
    experiment = fx.default_data_dir()
    if experiment is None or not (experiment / fx.BUNDLE_SUBDIR / fx.MANIFEST).is_file():
        return None
    return experiment


def test_real_bundle_check_passes():
    experiment = _real_root()
    if experiment is None:
        pytest.skip("D18 factor_cards_v1 bundle is not available here")
    rows = _check(experiment)
    assert not [r for r in rows if r[1] == "FAIL"], rows
    bundle = fx.CardBundle(experiment)
    pc8 = "single_geneset__hallmark_interferon_alpha_response__pca__k40__f008"
    assert [s["patient"] for s in fx.fold_strip(bundle, pc8) if s["badge"]] == ["P01", "P09"]
    nav = fx.navigate(bundle, *fx.parse_search(
        "?panel=single_geneset__hallmark_interferon_alpha_response&method=fa&k=40&factor=7").values())
    assert nav.card_id == "single_geneset__hallmark_interferon_alpha_response__fa__k40__f007"


def test_real_bundle_exports_under_20_mb(tmp_path):
    experiment = _real_root()
    if experiment is None:
        pytest.skip("D18 factor_cards_v1 bundle is not available here")
    written = fx.export_bundle(experiment, tmp_path / "real")
    total = sum(p.stat().st_size for p in written)
    assert total <= fx.EXPORT_LIMIT_BYTES, total
    assert len(fx.CardBundle(tmp_path / "real").index) == len(fx.CardBundle(experiment).index)
