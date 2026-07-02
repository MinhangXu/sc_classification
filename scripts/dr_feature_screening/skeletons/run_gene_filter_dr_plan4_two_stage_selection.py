#!/usr/bin/env python
"""
Plan 4 runner (skeleton): two-stage gene selection (unsup prefilter → supervised top-up).

Source plan: `sc_classification/scripts/comprehensive_run/plans/later_plans2_4.md`

Intent:
- Stage 1: broad unsupervised prefilter (e.g. HVG, variance, detection filter)
- Stage 2: within training only (later hardening), add a small supervised top-up:
  - weak DEGs or "rescue" genes (rare but malignant-enriched)
- Evaluate via the Plan 1 metric suite.

Not implemented yet:
- train-only protocol (avoid leakage)
- implementing a 2-stage pipeline inside PreprocessingPipeline config
  (some pieces exist: `hvg_plus_rescue_union`; this plan is about generalizing it)
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Plan 4 (skeleton): two-stage gene selection.")
    p.add_argument("--input-h5ad", required=True)
    p.add_argument("--experiments-dir", default="experiments")
    p.add_argument("--timepoint-filter", default="MRD")
    p.add_argument("--tech-filter", default="CITE")

    # Stage 1 (unsupervised)
    p.add_argument("--stage1", default="hvg", help="Stage-1 unsupervised prefilter (e.g. hvg).")
    p.add_argument("--stage1-hvg-n", type=int, default=3000)

    # Stage 2 (supervised top-up)
    p.add_argument("--stage2", default="rescue", help="Stage-2 top-up method (e.g. rescue or deg_weak).")
    p.add_argument("--rescue-min-frac", type=float, default=0.01)
    p.add_argument("--rescue-ratio", type=float, default=20.0)
    p.add_argument("--deg-p", type=float, default=0.1)
    p.add_argument("--deg-lfc", type=float, default=0.05)

    # Downstream eval knobs (mirror Plan 1)
    p.add_argument("--dr-methods", default="pca,fa,nmf,factosig,cnmf")
    p.add_argument("--k-by-method", default="pca=60,fa=60,nmf=60,factosig=60,cnmf=60")
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--cv-repeats", type=int, default=10)

    args = p.parse_args()
    _ = args

    raise NotImplementedError(
        "Plan 4 runner skeleton only. Next steps: implement a generic two-stage gene selection "
        "config, then reuse Plan 1 grid evaluation + (optional) train-only hardening."
    )


if __name__ == "__main__":
    main()

