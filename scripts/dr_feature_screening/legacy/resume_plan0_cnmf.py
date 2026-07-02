#!/usr/bin/env python
"""
Resume cNMF for an existing Plan 0 experiment directory.

Use-case:
- You ran `run_gene_filter_dr_grid.py plan0` and it crashed mid-cNMF due to an API mismatch.
- FA / FactoSig / PCA outputs were already cached.
- This script finishes the cNMF part *in the same experiment directory*.

Compatible with cnmf>=1.7.0 (uses `cNMF.factorize`). Will fall back to `run_nmf` if present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(p: Path, obj: Any) -> None:
    _ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _split_list_arg(v: str) -> List[str]:
    s = str(v).replace(",", " ")
    return [t for t in s.split() if t]


def _try_read_plan0_ks(exp_dir: Path) -> Optional[List[int]]:
    """
    Best-effort: pull ks from analysis/plan0/config.json if it exists.
    """
    p = exp_dir / "analysis" / "plan0" / "config.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        ks = obj.get("plan0", {}).get("ks", None)
        if ks is None:
            return None
        return [int(x) for x in ks]
    except Exception:
        return None


def resume_cnmf_plan0(
    *,
    experiment_dir: Path,
    ks: List[int],
    cnmf_n_iter: Optional[int],
    cnmf_dt: str,
    name: str = "plan0_cnmf",
    models_subdir: str = "models/cnmf_plan0",
    skip_completed_runs: bool = True,
) -> None:
    try:
        from cnmf import cNMF  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Could not import cnmf. Are you in the right environment? error={e}") from e

    exp_dir = experiment_dir.resolve()
    output_dir = exp_dir / models_subdir
    if not output_dir.exists():
        raise FileNotFoundError(f"Expected cNMF output_dir not found: {output_dir}")

    # Instantiate cNMF pointing at existing output tree
    cn = cNMF(output_dir=str(output_dir), name=name)

    # cnmf consensus() expects a TPM h5ad on disk. Our original Plan 0 code saved norm_counts
    # but not the TPM file, so for salvage we create it if missing.
    try:
        tpm_path = Path(cn.paths["tpm"])
        norm_counts_path = Path(cn.paths["normalized_counts"])
        # If a previous attempt created a broken symlink, `Path.exists()` is False
        # but the path still "lexists" and must be removed before re-linking.
        if os.path.lexists(tpm_path) and not tpm_path.exists():
            try:
                tpm_path.unlink()
            except Exception as e:
                _write_json(exp_dir / "analysis" / "plan0" / "cnmf_tpm_prepare_error.json", {"error": f"failed_unlink_broken_tpm: {e}"})
        if not tpm_path.exists() and norm_counts_path.exists():
            _ensure_dir(tpm_path.parent)
            try:
                # Prefer a symlink to avoid copying a large file.
                tpm_path.symlink_to(norm_counts_path.resolve())
                _write_json(
                    exp_dir / "analysis" / "plan0" / "cnmf_tpm_link_created.json",
                    {"tpm_path": str(tpm_path), "linked_to": str(norm_counts_path)},
                )
            except Exception:
                # Fallback: copy if symlinks not supported
                import shutil

                shutil.copy2(norm_counts_path, tpm_path)
                _write_json(
                    exp_dir / "analysis" / "plan0" / "cnmf_tpm_copied.json",
                    {"tpm_path": str(tpm_path), "copied_from": str(norm_counts_path)},
                )
    except Exception as e:
        _write_json(exp_dir / "analysis" / "plan0" / "cnmf_tpm_prepare_error.json", {"error": str(e)})

    # cnmf==1.7.0 consensus() also expects tpm_stats (computed during cn.prepare()).
    # Older Plan 0 runs may not have created it; reconstruct from tpm.h5ad.
    try:
        tpm_stats_path = Path(cn.paths.get("tpm_stats", ""))
        if tpm_stats_path and not tpm_stats_path.exists():
            _ensure_dir(tpm_stats_path.parent)
            import numpy as np
            import pandas as pd
            import scanpy as sc
            import scipy.sparse as sp

            try:
                # Use cnmf's own helpers if present (matches file format exactly).
                from cnmf.cnmf import get_mean_var, save_df_to_npz  # type: ignore
            except Exception:  # pragma: no cover
                get_mean_var = None
                save_df_to_npz = None

            tpm = sc.read(str(Path(cn.paths["tpm"])))
            if sp.issparse(tpm.X):
                if get_mean_var is None:
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler(with_mean=False)
                    scaler.fit(tpm.X)
                    gene_tpm_mean = scaler.mean_
                    gene_tpm_stddev = np.sqrt(scaler.var_)
                else:
                    gene_tpm_mean, gene_tpm_var = get_mean_var(tpm.X)
                    gene_tpm_stddev = np.sqrt(gene_tpm_var)
            else:
                gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
                gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)

            input_tpm_stats = pd.DataFrame(
                [gene_tpm_mean, gene_tpm_stddev],
                index=["__mean", "__std"],
                columns=tpm.var.index,
            ).T

            if save_df_to_npz is not None:
                save_df_to_npz(input_tpm_stats, str(tpm_stats_path))
            else:  # pragma: no cover
                np.savez_compressed(
                    str(tpm_stats_path),
                    data=input_tpm_stats.values,
                    index=input_tpm_stats.index.values,
                    columns=input_tpm_stats.columns.values,
                )
            _write_json(
                exp_dir / "analysis" / "plan0" / "cnmf_tpm_stats_created.json",
                {"tpm_stats_path": str(tpm_stats_path), "from_tpm": str(cn.paths["tpm"])},
            )
    except Exception as e:
        _write_json(exp_dir / "analysis" / "plan0" / "cnmf_tpm_stats_prepare_error.json", {"error": str(e)})

    # Optional sanity: warn if iter params don't match desired n_iter
    # (We don't rewrite params here; we just run factorize using what was saved.)
    if cnmf_n_iter is not None:
        _write_json(
            exp_dir / "analysis" / "plan0" / "cnmf_resume_request.json",
            {"requested_n_iter": int(cnmf_n_iter), "note": "This resume script does not rewrite nmf params."},
        )

    # Factorize (API depends on cnmf version)
    if hasattr(cn, "factorize"):
        # Important for cnmf>=1.7: `skip_completed_runs` relies on the ledger being updated.
        # The nmf_params table stores a `completed` column that may remain False even when
        # output files exist on disk. `update_nmf_iter_params()` scans outputs and updates it.
        if skip_completed_runs and hasattr(cn, "update_nmf_iter_params"):
            try:
                cn.update_nmf_iter_params()
            except Exception as e:
                _write_json(exp_dir / "analysis" / "plan0" / "cnmf_update_nmf_iter_params_error.json", {"error": str(e)})
        cn.factorize(worker_i=0, total_workers=1, skip_completed_runs=bool(skip_completed_runs))
    else:  # older cnmf
        cn.run_nmf(worker_i=0, total_workers=1)  # type: ignore[attr-defined]

    # Combine + consensus for each K
    diag_dir = _ensure_dir(exp_dir / "analysis" / "plan0")
    for k in ks:
        cn.combine_nmf(int(k))
        # cnmf API compatibility: density threshold argument name changed across versions.
        # cnmf==1.7.0 uses `density_threshold` (float). Some older variants used `density_threshold_str` (string).
        import inspect

        sig = inspect.signature(cn.consensus)
        kwargs: Dict[str, Any] = {}
        if "density_threshold_str" in sig.parameters:
            kwargs["density_threshold_str"] = str(cnmf_dt)
        elif "density_threshold" in sig.parameters:
            kwargs["density_threshold"] = float(cnmf_dt)
        elif "dt" in sig.parameters:
            kwargs["dt"] = float(cnmf_dt)
        if "show_clustering" in sig.parameters:
            kwargs["show_clustering"] = True

        stats = cn.consensus(int(k), **kwargs)
        try:
            stats_obj: Dict[str, Any] = stats.to_dict()  # type: ignore[assignment]
        except Exception:
            stats_obj = {"stats_repr": str(stats)}
        _write_json(diag_dir / "cnmf" / f"k_{int(k)}" / "consensus_stats.json", stats_obj)

    # K-selection plot (best effort)
    try:
        cn.k_selection_plot(close_fig=True)
    except Exception as e:
        _write_json(diag_dir / "cnmf_k_selection_plot_error.json", {"error": str(e)})


def main() -> None:
    p = argparse.ArgumentParser(description="Resume cNMF portion of a Plan 0 experiment directory.")
    p.add_argument(
        "--experiment-dir",
        required=True,
        help="Path to an existing Plan 0 experiment directory (contains analysis/plan0 and models/cnmf_plan0).",
    )
    p.add_argument(
        "--ks",
        default="",
        help="K list (e.g. '20,40,60'). If omitted, tries to read from analysis/plan0/config.json.",
    )
    p.add_argument("--cnmf-dt", default="0.5")
    p.add_argument(
        "--cnmf-n-iter",
        type=int,
        default=None,
        help="Optional: record desired n_iter in a resume log (does not rewrite saved params).",
    )
    p.add_argument("--name", default="plan0_cnmf", help="cNMF run name (default matches run_gene_filter_dr_grid.py).")
    p.add_argument("--models-subdir", default="models/cnmf_plan0", help="Relative subdir under experiment-dir.")
    p.add_argument("--skip-completed-runs", action="store_true", help="If set, skip completed nmf replicates.")

    args = p.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    ks: Optional[List[int]] = None
    if args.ks.strip():
        ks = [int(x) for x in _split_list_arg(args.ks)]
    else:
        ks = _try_read_plan0_ks(exp_dir)

    if not ks:
        raise ValueError("No ks provided and could not infer ks from analysis/plan0/config.json")

    resume_cnmf_plan0(
        experiment_dir=exp_dir,
        ks=ks,
        cnmf_n_iter=args.cnmf_n_iter,
        cnmf_dt=str(args.cnmf_dt),
        name=str(args.name),
        models_subdir=str(args.models_subdir),
        skip_completed_runs=bool(args.skip_completed_runs),
    )


if __name__ == "__main__":
    main()

