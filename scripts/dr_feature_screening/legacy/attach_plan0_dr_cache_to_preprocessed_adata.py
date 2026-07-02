#!/usr/bin/env python
"""
Attach Plan 0 DR cache outputs back onto the Plan 0 preprocessed AnnData.

Why:
- Plan 0 caches DR results as arrays under:
  analysis/plan0/stability/<method>/k_<K>/replicates/seed_<seed>/{scores.npy,loadings.npy,extras.json}
- Sometimes it's convenient to rehydrate these into the preprocessed AnnData
  (`preprocessing/adata_processed.h5ad`) for downstream diagnosis/plotting.

Important:
- `preprocessing/adata_processed.h5ad` is the *standardized* Plan 0 AnnData (pre_std).
  It's correct to attach PCA/FA/FactoSig results there.
- NMF/cNMF in Plan 0 are fit on the *non-standardized* view (pre_nstd / counts-derived),
  so attaching them to the standardized `.X` is possible but can be misleading; we default
  to only attaching {pca,fa,factosig}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scanpy as sc


def _split_list_arg(v: str) -> List[str]:
    s = str(v).replace(",", " ")
    return [t for t in s.split() if t]


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {"_raw": obj}


def _method_keys(method: str) -> Dict[str, str]:
    """
    Map method name to AnnData keys consistent with sc_classification wrappers.
    """
    m = method.lower().strip()
    if m == "pca":
        return {"obsm": "X_pca", "varm": "PCA_loadings", "uns": "pca"}
    if m == "fa":
        return {"obsm": "X_fa", "varm": "FA_loadings", "uns": "fa"}
    if m == "factosig":
        return {"obsm": "X_factosig", "varm": "FACTOSIG_loadings", "uns": "factosig"}
    if m == "nmf":
        return {"obsm": "X_nmf", "varm": "NMF_components", "uns": "nmf"}
    raise ValueError(f"Unknown method: {method}")


def attach_plan0_cache(
    *,
    experiment_dir: Path,
    methods: List[str],
    k: int,
    seed: int,
    attach_consensus: bool,
    output_h5ad: Path,
) -> None:
    exp = experiment_dir.resolve()
    adata_path = exp / "preprocessing" / "adata_processed.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"Missing preprocessed AnnData: {adata_path}")

    ad = sc.read_h5ad(str(adata_path))

    for method in methods:
        m = method.lower().strip()
        keys = _method_keys(m)
        rep_dir = exp / "analysis" / "plan0" / "stability" / m / f"k_{int(k)}" / "replicates" / f"seed_{int(seed)}"
        scores_p = rep_dir / "scores.npy"
        loadings_p = rep_dir / "loadings.npy"
        extras_p = rep_dir / "extras.json"
        if not (scores_p.exists() and loadings_p.exists() and extras_p.exists()):
            raise FileNotFoundError(f"Missing cache files for {m} at {rep_dir}")

        scores = np.load(scores_p)
        loadings = np.load(loadings_p)
        extras = _read_json(extras_p)

        # shape checks
        if scores.shape[0] != ad.n_obs:
            raise ValueError(f"{m} scores n_obs mismatch: scores={scores.shape}, adata={ad.shape}")
        if scores.shape[1] != int(k):
            raise ValueError(f"{m} scores k mismatch: scores={scores.shape}, expected k={k}")
        if loadings.shape[0] != ad.n_vars or loadings.shape[1] != int(k):
            raise ValueError(f"{m} loadings shape mismatch: loadings={loadings.shape}, adata_vars={ad.n_vars}, k={k}")

        ad.obsm[keys["obsm"]] = np.asarray(scores, dtype=np.float32)
        ad.varm[keys["varm"]] = np.asarray(loadings, dtype=np.float32)
        # store extras under the method's conventional .uns slot
        if isinstance(extras, dict):
            ad.uns[keys["uns"]] = extras

        if attach_consensus and m in ("fa", "factosig"):
            cons_dir = exp / "analysis" / "plan0" / "stability" / m / f"k_{int(k)}" / "consensus_cache"
            cons_p = cons_dir / "consensus_loadings.npy"
            if cons_p.exists():
                cons = np.load(cons_p)  # (k, genes) as saved by runner
                if cons.shape[0] == int(k) and cons.shape[1] == ad.n_vars:
                    # store as (genes, k) to match varm convention
                    ad.varm[f"{keys['varm']}_consensus"] = np.asarray(cons.T, dtype=np.float32)

    ad.uns.setdefault("plan0_cache_attach", {})
    ad.uns["plan0_cache_attach"].update(
        {"k": int(k), "seed": int(seed), "methods": [m.lower().strip() for m in methods]}
    )

    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(str(output_h5ad))


def main() -> None:
    p = argparse.ArgumentParser(description="Attach Plan 0 DR caches onto preprocessing/adata_processed.h5ad")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--methods", default="pca,fa,factosig", help="Comma-separated list (default: pca,fa,factosig).")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--attach-consensus", action="store_true", help="Also attach FA/FactoSig consensus loadings if cached.")
    p.add_argument(
        "--output-h5ad",
        default="",
        help="Output path. Default: <experiment-dir>/preprocessing/adata_processed_with_plan0_dr_k<K>_seed<SEED>.h5ad",
    )
    args = p.parse_args()

    exp = Path(args.experiment_dir)
    methods = [m.strip() for m in _split_list_arg(args.methods)]
    out = (
        Path(args.output_h5ad)
        if args.output_h5ad.strip()
        else (exp / "preprocessing" / f"adata_processed_with_plan0_dr_k{int(args.k)}_seed{int(args.seed)}.h5ad")
    )

    attach_plan0_cache(
        experiment_dir=exp,
        methods=methods,
        k=int(args.k),
        seed=int(args.seed),
        attach_consensus=bool(args.attach_consensus),
        output_h5ad=out,
    )


if __name__ == "__main__":
    main()

