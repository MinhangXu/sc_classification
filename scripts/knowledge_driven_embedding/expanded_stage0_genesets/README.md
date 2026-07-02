# Expanded Stage 0 Gene-Set Bundle

This bundle expands the old-34 MSigDB Hallmark/Reactome/KEGG panel with public and local-prior gene sets for MRD malignant-vs-normal classification.

## Biological Axes

- `antigen_presentation_ciita_hla_escape`: 32 sets
- `cd54_cd244_adhesion_niche`: 67 sets
- `hematopoietic_lineage_controls`: 57 sets
- `hspc_lsc_stemness`: 79 sets
- `ifng_decomposed_response`: 64 sets
- `mitochondrial_metabolism_fine`: 63 sets
- `quiescence_dormancy_growth_arrest`: 32 sets
- `regulon_ifn_nfkb_ap1_gata_myc_e2f_p53_smad`: 38 sets
- `senescence_sasp_control`: 18 sets
- `tgfb_niche_quiescence`: 21 sets
- `tnf_nfkb_decomposed_survival_adhesion`: 70 sets
- `typeI_ifn_inflammasome_noncanonical_nfkb_contrast`: 65 sets

## Reproducibility

- MSigDB release: `2026.1.Hs`.
- Minimum expression-universe overlap: `n_present >= 10`.
- Regex selectors are frozen in `selector_provenance.json`; exact resolved names are frozen in `final_manifest.tsv`.
- Public prior gene sets are primary evaluation resources; expert anchors are support/interpretation panels.
- Same-cohort DE-derived sets are not used. Local huCIRA cytokine resources are treated as external perturbation priors when present.

## Outputs

- `final_manifest.tsv` / `final_manifest.csv`: full manifest with Stage 0 compatibility columns.
- `final_bundle.gmt`: GMT keyed by `set_id`.
- `resource_metadata.csv` / `.json`: URL/local path, retrieval date, file size, and SHA256.
- `qc/`: set-size, overlap, coverage, novelty, and dropped-set reports.

## Interpretation Guardrails

Classifier performance should be described as a transcriptional footprint consistent with a biological process, not as causal evidence. Pooled-cell performance alone is not biological validation; sharedness and patient-specific results should be reviewed separately.

## Resource Summary

- `MSigDB_H`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_h.all.symbols.gmt` (`sha256=eecaf6dad908334ae885406ec72bdc0646d8917588ed7c219fac92fc5363f596`)
- `MSigDB_C2_REACTOME`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c2.cp.reactome.symbols.gmt` (`sha256=5d61f289a2400cddfbb3a3353829fd2284a360bbe50f2093b566c4b7bea93341`)
- `MSigDB_C2_KEGG_LEGACY`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c2.cp.kegg_legacy.symbols.gmt` (`sha256=4a87c8d1260e637ec174f5960885519d46fa29da650e70d3a3641a4d41c4fa5e`)
- `MSigDB_C5_GO_BP`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c5.go.bp.symbols.gmt` (`sha256=9be09dd06d6652566eb52eed530d62e6dfecc4365c1e81afd6f0b7f2e86dd4f9`)
- `MSigDB_C5_GO_CC`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c5.go.cc.symbols.gmt` (`sha256=1ee846c446a87c1b6cc2b097dc415bed1c2f6b539312d2611f946038c5f7e9f8`)
- `MSigDB_C5_GO_MF`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c5.go.mf.symbols.gmt` (`sha256=72da03f5438f1005566ecd5c85bf32e81f343910241948da603d2fafdf47f555`)
- `MSigDB_C3_TFT_GTRD`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c3.tft.gtrd.symbols.gmt` (`sha256=1c981780b00173ee7e1f86d61f0cf1a9fa80afc2bb28bfb010efc5eea3aaad6c`)
- `MSigDB_C3_TFT`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c3.tft.symbols.gmt` (`sha256=426b3c4f863d14bb6c80d5d00fb740d407c5d263526af0757fb5d82b10ae09c6`)
- `MSigDB_C7_IMMUNESIGDB`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c7.immunesigdb.symbols.gmt` (`sha256=facfeff5feca6bcef83abf7bde88043096c441b718e74c862b6248404cf97f5d`)
- `MSigDB_C8_CELL_TYPE`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c8.all.symbols.gmt` (`sha256=715a1af9f0d36e1dfd631e9daa204157e741e2a07dce90abb18f2f0aa98f8d04`)
- `MSigDB_C2_CGP`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c2.cgp.symbols.gmt` (`sha256=170609526c4a52d89f3ce3719b616b199b3b13d8785dc672a593fabb90e72bf2`)
- `MSigDB_C2_WIKIPATHWAYS`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c2.cp.wikipathways.symbols.gmt` (`sha256=42c7a6dedea6a7abbc37f5d77d1ab84cc8fdaad44d7f664026047d23cc78f52c`)
- `MSigDB_C4_3CA`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c4.3ca.symbols.gmt` (`sha256=718fc509b9891f1db00e624c571d55d593a54a1a0d9cb9d48848e3caa34e7b28`)
- `MSigDB_C6_ONCOGENIC`: `/home/minhang/mds_project/data/resource_cache/stage0_expanded_genesets/msigdb_2026.1.Hs_c6.all.symbols.gmt` (`sha256=d127b735a03a85fdf90c1ab91d49d3cab2c4c507454f769b85e2df5ce3bb3561`)
- `huCIRA_CYTOKINE_UP`: `/home/minhang/mds_project/data/hucira_reference/cytokine_gene_sets_up.gmt` (`sha256=16cb90e073be293c9151e877b122711bfade5a3dbb6ea526e273f93beeb1c096`)
- `huCIRA_CYTOKINE_DOWN`: `/home/minhang/mds_project/data/hucira_reference/cytokine_gene_sets_down.gmt` (`sha256=99c09d5dd63a46505cc32bacbec31d17abc6ade07b371f6559c576c27be02abc`)
