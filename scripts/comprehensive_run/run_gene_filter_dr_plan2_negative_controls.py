#!/usr/bin/env python
"""
Plan 2 runner (skeleton): negative controls / leakage checks.

Source plan: `sc_classification/scripts/comprehensive_run/plans/later_plans2_4.md`

Intent:
- Run a "mirrored" version of Plan 1, but with CN.label permuted within strata
  (e.g. within patient × timepoint_type), so we can detect leakage/confounding.

Design notes:
- This should *not* inflate storage: keep one main artifact per preprocess method,
  and write permutation-run metrics under a distinct directory name.
- Prefer reusing the Plan 1 preprocessing + DR code paths to avoid drift.

Not implemented yet:
- actual stratified permutation
- wiring to the ExperimentManager layout
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def main() -> None:
    p = argparse.ArgumentParser(description="Plan 2 (skeleton): negative controls / leakage checks.")
    p.add_argument("--input-h5ad", required=True)
    p.add_argument("--experiments-dir", default="experiments")

    # Match Plan 0/1 filters for comparability
    p.add_argument("--timepoint-filter", default="MRD")
    p.add_argument("--tech-filter", default="CITE")

    # Permutation protocol
    p.add_argument("--target-col", default="CN.label")
    p.add_argument("--strata-cols", default="patient,timepoint_type", help="Comma-separated strata columns.")
    p.add_argument("--n-permutations", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)

    # Mirror Plan 1 knobs (kept minimal for skeleton)
    p.add_argument("--preprocess-set", default="hvg,all_filtered,deg_weak_screen,hybrid")
    p.add_argument("--dr-methods", default="pca,fa,nmf,factosig,cnmf")
    p.add_argument("--k-by-method", default="pca=60,fa=60,nmf=60,factosig=60,cnmf=60")
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--cv-repeats", type=int, default=10)

    args = p.parse_args()

    exp_root = Path(args.experiments_dir)
    _ = exp_root  # placeholder so linters don't complain in skeleton mode

    raise NotImplementedError(
        "Plan 2 runner skeleton only. Next steps: implement within-strata label permutation, "
        "then reuse Plan 1 preprocessing/DR + classification under a dedicated output subtree."
    )


if __name__ == "__main__":
    main()

