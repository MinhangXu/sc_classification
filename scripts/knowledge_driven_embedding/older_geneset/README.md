# Older Generation Gene Set Bundle (MSigDB)

This directory contains a uniform MSigDB-based bundle for requested older-generation gene sets.

## Requested set list (exact names)

### Hallmark (H)
- `HALLMARK_INTERFERON_GAMMA_RESPONSE`
- `HALLMARK_INTERFERON_ALPHA_RESPONSE`
- `HALLMARK_TNFA_SIGNALING_VIA_NFKB`
- `HALLMARK_INFLAMMATORY_RESPONSE`
- `HALLMARK_IL6_JAK_STAT3_SIGNALING`
- `HALLMARK_IL2_STAT5_SIGNALING`
- `HALLMARK_COMPLEMENT`
- `HALLMARK_ALLOGRAFT_REJECTION`
- `HALLMARK_APOPTOSIS`
- `HALLMARK_P53_PATHWAY`
- `HALLMARK_HYPOXIA`
- `HALLMARK_UNFOLDED_PROTEIN_RESPONSE`
- `HALLMARK_DNA_REPAIR`
- `HALLMARK_E2F_TARGETS`
- `HALLMARK_G2M_CHECKPOINT`
- `HALLMARK_MYC_TARGETS_V1`
- `HALLMARK_MYC_TARGETS_V2`
- `HALLMARK_OXIDATIVE_PHOSPHORYLATION`
- `HALLMARK_GLYCOLYSIS`
- `HALLMARK_MTORC1_SIGNALING`
- `HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY`
- `HALLMARK_TGF_BETA_SIGNALING`
- `HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION`

### Reactome (C2:CP:REACTOME)
- `REACTOME_INTERFERON_SIGNALING`
- `REACTOME_INTERFERON_GAMMA_SIGNALING`
- `REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM`
- `REACTOME_NF_KB_ACTIVATION`
- `REACTOME_TNFR1_INDUCED_NFKB_SIGNALING_PATHWAY`
- `REACTOME_ANTIGEN_PRESENTATION_FOLDING_ASSEMBLY_AND_PEPTIDE_LOADING_OF_CLASS_I_MHC`
- `REACTOME_MHC_CLASS_I_ANTIGEN_PRESENTATION`
- `REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION`

### KEGG (C2:CP:KEGG)
- `KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION`
- `KEGG_JAK_STAT_SIGNALING_PATHWAY`
- `KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY`

## How sets were retrieved

- Source: MSigDB release `2026.1.Hs` GMT files downloaded over HTTPS.
- Files used:
  - Hallmark: `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/h.all.v2026.1.Hs.symbols.gmt`
  - Reactome: `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/c2.cp.reactome.v2026.1.Hs.symbols.gmt`
  - KEGG: `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2026.1.Hs/c2.cp.kegg_legacy.v2026.1.Hs.symbols.gmt`
- Exact-name output preserved; for 3 Reactome entries, current canonical MSigDB names were aliased:
  - `REACTOME_NF_KB_ACTIVATION` <= `REACTOME_NF_KB_IS_ACTIVATED_AND_SIGNALS_SURVIVAL`
  - `REACTOME_TNFR1_INDUCED_NFKB_SIGNALING_PATHWAY` <= `REACTOME_TNFR1_INDUCED_NF_KAPPA_B_SIGNALING_PATHWAY`
  - `REACTOME_MHC_CLASS_I_ANTIGEN_PRESENTATION` <= `REACTOME_CLASS_I_MHC_MEDIATED_ANTIGEN_PROCESSING_PRESENTATION`

## Gene symbol harmonization rules

- Uppercase symbols.
- Remove empty / NA symbols.
- Deduplicate symbols within each set.
- Sort symbols alphabetically within each set.
- Keep sets with fewer than 5 genes and flag them as warnings.

## QA

- Number of sets written: **34** (expected 34).
- Gene counts summary (min / median / max): **13 / 199.5 / 792**.

### Gene counts per set

| geneset_name | n_genes |
|---|---:|
| `HALLMARK_INTERFERON_GAMMA_RESPONSE` | 200 |
| `HALLMARK_INTERFERON_ALPHA_RESPONSE` | 97 |
| `HALLMARK_TNFA_SIGNALING_VIA_NFKB` | 200 |
| `HALLMARK_INFLAMMATORY_RESPONSE` | 200 |
| `HALLMARK_IL6_JAK_STAT3_SIGNALING` | 87 |
| `HALLMARK_IL2_STAT5_SIGNALING` | 199 |
| `HALLMARK_COMPLEMENT` | 200 |
| `HALLMARK_ALLOGRAFT_REJECTION` | 200 |
| `HALLMARK_APOPTOSIS` | 161 |
| `HALLMARK_P53_PATHWAY` | 200 |
| `HALLMARK_HYPOXIA` | 200 |
| `HALLMARK_UNFOLDED_PROTEIN_RESPONSE` | 113 |
| `HALLMARK_DNA_REPAIR` | 150 |
| `HALLMARK_E2F_TARGETS` | 200 |
| `HALLMARK_G2M_CHECKPOINT` | 200 |
| `HALLMARK_MYC_TARGETS_V1` | 200 |
| `HALLMARK_MYC_TARGETS_V2` | 58 |
| `HALLMARK_OXIDATIVE_PHOSPHORYLATION` | 200 |
| `HALLMARK_GLYCOLYSIS` | 200 |
| `HALLMARK_MTORC1_SIGNALING` | 200 |
| `HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY` | 49 |
| `HALLMARK_TGF_BETA_SIGNALING` | 54 |
| `HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION` | 200 |
| `REACTOME_INTERFERON_SIGNALING` | 289 |
| `REACTOME_INTERFERON_GAMMA_SIGNALING` | 98 |
| `REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM` | 792 |
| `REACTOME_NF_KB_ACTIVATION` | 13 |
| `REACTOME_TNFR1_INDUCED_NFKB_SIGNALING_PATHWAY` | 33 |
| `REACTOME_ANTIGEN_PRESENTATION_FOLDING_ASSEMBLY_AND_PEPTIDE_LOADING_OF_CLASS_I_MHC` | 29 |
| `REACTOME_MHC_CLASS_I_ANTIGEN_PRESENTATION` | 373 |
| `REACTOME_MHC_CLASS_II_ANTIGEN_PRESENTATION` | 126 |
| `KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION` | 88 |
| `KEGG_JAK_STAT_SIGNALING_PATHWAY` | 155 |
| `KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY` | 102 |

### Warnings

- None.

## Run metadata

- Build timestamp (UTC): `2026-03-02 19:57:43 UTC`
- Python: `3.11.5`
- Platform: `Linux-5.4.0-205-generic-x86_64-with-glibc2.31`

### pip freeze (first 30 lines)

```text
anndata==0.10.8
anyio @ file:///work/ci_py311/anyio_1676823771847/work/dist
archspec @ file:///croot/archspec_1697725767277/work
argon2-cffi @ file:///opt/conda/conda-bld/argon2-cffi_1645000214183/work
argon2-cffi-bindings @ file:///work/ci_py311/argon2-cffi-bindings_1676823553406/work
array_api_compat==1.8
asttokens==2.4.1
async-lru @ file:///croot/async-lru_1699554519285/work
attrs @ file:///croot/attrs_1695717823297/work
Babel @ file:///work/ci_py311/babel_1676825020543/work
beautifulsoup4 @ file:///croot/beautifulsoup4-split_1681493039619/work
bleach @ file:///opt/conda/conda-bld/bleach_1641577558959/work
boltons @ file:///work/ci_py311/boltons_1677685195580/work
Brotli @ file:///work/ci_py311/brotli-split_1676830125088/work
certifi @ file:///home/conda/feedstock_root/build_artifacts/certifi_1700303426725/work/certifi
cffi @ file:///croot/cffi_1700254295673/work
charset-normalizer @ file:///tmp/build/80754af9/charset-normalizer_1630003229654/work
comm==0.2.1
conda @ file:///home/conda/feedstock_root/build_artifacts/conda_1701731617055/work
conda-content-trust @ file:///croot/conda-content-trust_1693490622020/work
conda-libmamba-solver @ file:///croot/conda-libmamba-solver_1702997573971/work/src
conda-package-handling @ file:///croot/conda-package-handling_1690999929514/work
conda_package_streaming @ file:///croot/conda-package-streaming_1690987966409/work
contourpy==1.2.1
cryptography @ file:///croot/cryptography_1702070282333/work
cycler==0.12.1
debugpy==1.8.0
decorator @ file:///opt/conda/conda-bld/decorator_1643638310831/work
defusedxml @ file:///tmp/build/80754af9/defusedxml_1615228127516/work
distro @ file:///croot/distro_1701455004953/work
```

## Future extension note

Tier 2 cytokine-dictionary gene sets should live in `knowledge_driven_embedding/cytokine_dictionary/`.
