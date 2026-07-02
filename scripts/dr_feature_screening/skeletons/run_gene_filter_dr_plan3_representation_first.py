#!/usr/bin/env python
"""
Plan 3 runner (skeleton): representation-first evaluation.

Source plan: `sc_classification/scripts/comprehensive_run/plans/later_plans2_4.md`

Intent:
- Evaluate preprocess × DR choices primarily by representation quality + stability,
  and report CN.label separability as a secondary metric.

Candidate primary metrics (method-dependent; not yet wired):
- stability of loadings across seeds / bootstraps (alignment score)
- reconstruction / likelihood proxy (PCA/FA/FactoSig/NMF specific)
- interpretability proxy (sparsity post-rotation; enrichment consistency)

Not implemented yet:
- metric definitions + caching format
- bootstrapping protocol
- aggregation/summarization outputs
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Plan 3 (skeleton): representation-first evaluation.")
    p.add_argument("--input-h5ad", required=True)
    p.add_argument("--experiments-dir", default="experiments")
    p.add_argument("--timepoint-filter", default="MRD")
    p.add_argument("--tech-filter", default="CITE")

    # Mirror Plan 1 knobs so we can reuse caches later
    p.add_argument("--preprocess-set", default="hvg,all_filtered,deg_weak_screen,hybrid")
    p.add_argument("--dr-methods", default="pca,fa,nmf,factosig,cnmf")
    p.add_argument("--k-by-method", default="pca=60,fa=60,nmf=60,factosig=60,cnmf=60")
    p.add_argument("--seeds", default="1,2,3,4,5")

    args = p.parse_args()
    _ = args

    raise NotImplementedError(
        "Plan 3 runner skeleton only. Next steps: define representation metrics, "
        "reuse per-seed caches (Plan 0/1), and write a single summary table per grid."
    )


if __name__ == "__main__":
    main()

