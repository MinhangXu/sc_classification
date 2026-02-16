# FactoSig pipeline lineage (legacy)

This folder contains the FactoSig experiment flow used before the current
`scripts/comprehensive_run/` workflow.

## Files

- `run_factosig_only.py`: runs FactoSig-only DR with ExperimentManager outputs.
- `compare_dr_factosig_vs_sklearn.py`: compares sklearn FA and FactoSig DR caches.
- `elastic_net_from_dr_cache.py`: elastic-net classifier evaluation from cached DR embeddings.

These scripts are preserved for reproducibility and historical comparison.

