#!/usr/bin/env python
"""
Build `analysis/plan0/k_selection_summary.csv` for a Plan 0 experiment.

Why:
- The plan0 runner normally writes this at the end of a successful run.
- If plan0 crashes mid-way (e.g., during cNMF), FA/FactoSig caches still exist but
  `k_selection_summary.csv` won't.
- This script reconstructs the summary from cached JSON/arrays on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {"_raw": obj}


def _try_variance_proxy(method: str, extras: Dict[str, Any]) -> Optional[float]:
    m = method.lower().strip()
    if m == "pca":
        evr = np.asarray(extras.get("explained_variance_ratio", []), dtype=float)
        return float(np.sum(evr)) if evr.size else None
    if m in ("fa", "factosig", "nmf"):
        v = extras.get("sum_factor_score_variances", None)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None
    return None


def build_summary(exp_dir: Path, seed: int = 1) -> pd.DataFrame:
    exp = exp_dir.resolve()
    stab_root = exp / "analysis" / "plan0" / "stability"
    if not stab_root.exists():
        raise FileNotFoundError(f"Missing stability directory: {stab_root}")

    rows: List[Dict[str, Any]] = []
    for method_dir in sorted([p for p in stab_root.iterdir() if p.is_dir()]):
        method = method_dir.name
        for k_dir in sorted([p for p in method_dir.iterdir() if p.is_dir() and p.name.startswith("k_")]):
            k = int(k_dir.name.split("_", 1)[1])

            pairwise_p = k_dir / "pairwise_stability_summary.json"
            pairwise = _read_json(pairwise_p) if pairwise_p.exists() else {}

            # count replicates available
            reps_dir = k_dir / "replicates"
            n_runs = None
            if reps_dir.exists():
                n_runs = len([p for p in reps_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])

            extras_p = k_dir / "replicates" / f"seed_{int(seed)}" / "extras.json"
            extras = _read_json(extras_p) if extras_p.exists() else {}

            cons_p = k_dir / "consensus_cache" / "consensus_metrics.json"
            cons = _read_json(cons_p) if cons_p.exists() else {}

            rows.append(
                {
                    "method": method,
                    "k": k,
                    "n_seeds": n_runs,
                    "stability_best_a_median": pairwise.get("best_a_median_over_pairs", None),
                    "stability_best_a_mean": pairwise.get("best_a_mean_over_pairs", None),
                    "stability_frac_lt_0p3": pairwise.get("best_a_frac_lt_0p3_over_pairs", None),
                    "consensus_silhouette": cons.get("silhouette", None),
                    "variance_proxy": _try_variance_proxy(method, extras),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["method", "k"]).reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild Plan 0 k_selection_summary.csv from cached outputs.")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--seed", type=int, default=1, help="Which seed's extras.json to use for variance_proxy.")
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    df = build_summary(exp, seed=int(args.seed))

    out = exp / "analysis" / "plan0" / "k_selection_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

