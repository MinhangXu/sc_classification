from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BUILDER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "knowledge_driven_embedding"
    / "expanded_stage0_genesets"
    / "build_expanded_stage0_bundle.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("build_expanded_stage0_bundle", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_gmt_deduplicates_and_uppercases(tmp_path):
    builder = load_builder()
    gmt = tmp_path / "mini.gmt"
    gmt.write_text("SetA\tdesc\tgene1\tGENE1\tna\tGene2\n", encoding="utf-8")

    rows = builder.parse_gmt(gmt, "Test")

    assert rows[0]["source_collection"] == "Test"
    assert rows[0]["original_name"] == "SetA"
    assert rows[0]["genes"] == ["GENE1", "GENE2"]


def test_local_resource_metadata_has_reproducible_sha256(tmp_path):
    builder = load_builder()
    resource = tmp_path / "resource.gmt"
    resource.write_text("SetA\tdesc\tA\tB\n", encoding="utf-8")
    spec = builder.ResourceSpec("LOCAL", "v1", "", "resource.gmt", local_path=str(resource))

    path, meta = builder.materialize_resource(spec, tmp_path)

    assert path == resource
    assert meta["sha256"] == builder.sha256_file(resource)
    assert meta["file_size_bytes"] == resource.stat().st_size
    assert meta["retrieval_status"] == "local_existing"


def test_build_bundle_filters_no_overlap_and_builds_unions(tmp_path, monkeypatch):
    builder = load_builder()
    gmt = tmp_path / "mini.gmt"
    gmt.write_text(
        "\n".join(
            [
                "GO_RESPONSE_TO_INTERFERON_GAMMA\tdesc\tA\tB\tB\tC",
                "GO_CELLULAR_QUIESCENCE\tdesc\tD\tE",
                "GO_NO_OVERLAP\tdesc\tX\tY",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    selector = builder.SelectorSpec(
        "mini_selector",
        "ifng_decomposed_response",
        "ifng_stat_irf_targets",
        "target_footprint",
        "core",
        ("MSigDB_C5_GO_BP",),
        ("INTERFERON_GAMMA", "NO_OVERLAP"),
        "test selector",
    )
    monkeypatch.setattr(builder, "SELECTORS", (selector,))
    monkeypatch.setattr(
        builder,
        "CORE_ANCHORS",
        {
            "ifng_decomposed_response": {
                "subfamily": "core_ifng",
                "genes": ["A", "B", "IFNG"],
                "layer": "target_footprint",
                "rationale": "test anchors",
            }
        },
    )
    resources = [
        {
            "source_collection": "MSigDB_C5_GO_BP",
            "source_version": "v1",
            "url": "https://example.test/mini.gmt",
            "local_path": str(gmt),
        }
    ]

    manifest, dropped, provenance = builder.build_bundle(
        resources=resources,
        gene_universe={"A", "B", "C", "IFNG"},
        old_bundle_genes={"A"},
        hvg_genes={"B"},
        min_present=2,
        broad_threshold=750,
        very_broad_threshold=1000,
        max_atomic_per_selector=0,
    )

    assert "mini_selector" in provenance
    assert set(manifest["interpretation_layer"]) == {"target_footprint", "family_union", "core_anchor"}
    assert manifest.loc[manifest["original_name"].eq("GO_RESPONSE_TO_INTERFERON_GAMMA"), "n_present"].item() == 3
    assert dropped.loc[dropped["original_name"].eq("GO_NO_OVERLAP"), "reason"].item() == "n_present_lt_2"
    union = manifest[manifest["interpretation_layer"].eq("family_union")].iloc[0]
    assert {"A", "B", "C"}.issubset(set(union["genes_raw"].split(";")))


def test_write_final_gmt_from_manifest(tmp_path, monkeypatch):
    builder = load_builder()
    gmt = tmp_path / "mini.gmt"
    gmt.write_text("GO_INTERFERON_GAMMA\tdesc\tA\tB\tC\n", encoding="utf-8")
    selector = builder.SelectorSpec(
        "mini_selector",
        "ifng_decomposed_response",
        "ifng_stat_irf_targets",
        "target_footprint",
        "core",
        ("MSigDB_C5_GO_BP",),
        ("INTERFERON_GAMMA",),
        "test selector",
    )
    monkeypatch.setattr(builder, "SELECTORS", (selector,))
    monkeypatch.setattr(builder, "CORE_ANCHORS", {})

    manifest, dropped, _ = builder.build_bundle(
        resources=[
            {
                "source_collection": "MSigDB_C5_GO_BP",
                "source_version": "v1",
                "url": "https://example.test/mini.gmt",
                "local_path": str(gmt),
            }
        ],
        gene_universe={"A", "B", "C"},
        old_bundle_genes=set(),
        hvg_genes=set(),
        min_present=1,
        broad_threshold=750,
        very_broad_threshold=1000,
        max_atomic_per_selector=0,
    )
    assert dropped.empty

    out_gmt = tmp_path / "final_bundle.gmt"
    builder.write_gmt(out_gmt, manifest)

    parsed = builder.parse_gmt(out_gmt, "final")
    assert {row["original_name"] for row in parsed} == set(manifest["set_id"])
