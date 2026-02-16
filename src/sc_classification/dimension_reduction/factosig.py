import numpy as np
from .base import DimensionReductionMethod


class FactoSigDR(DimensionReductionMethod):
    """Wrapper for the external FactoSig package with consistent AnnData outputs."""

    def fit_transform(
        self,
        adata,
        n_components: int = 50,
        random_state: int = 0,
        row_weights=None,
        device: str = "cpu",
        lr: float = 1e-2,
        max_iter: int = 300,
        verbose: bool = True,
        save_fitted_models: bool = False,
        model_save_dir: str = None,
        order_factors_by: str | None = "ss_loadings",
    ):
        """
        Perform FactoSig on the provided AnnData.

        Returns:
        - AnnData updated with:
          obsm['X_factosig'] (cells × k),
          varm['FACTOSIG_loadings'] (genes × k),
          var['FACTOSIG_psi'] if available,
          uns['factosig'] (metadata).
        """
        # Robust import:
        # In this monorepo, the project directory `mds_project/factosig/` can shadow the
        # installed `factosig` package (namespace package without `FactoSig` exported).
        # If the import fails, try adding the local package root to sys.path.
        try:
            from factosig import FactoSig  # type: ignore
        except Exception:
            try:
                import sys
                import importlib
                from pathlib import Path

                here = Path(__file__).resolve()
                for parent in here.parents:
                    candidate = parent / "factosig" / "factosig" / "__init__.py"
                    if candidate.exists():
                        sys.path.insert(0, str(parent / "factosig"))
                        break

                # If a namespace package named 'factosig' was already imported from the repo root,
                # clear it so Python can re-resolve using the injected sys.path.
                sys.modules.pop("factosig", None)
                FactoSig = getattr(importlib.import_module("factosig"), "FactoSig")
            except Exception as e:
                raise ImportError(
                    "Failed to import 'factosig'. Install it (e.g. `pip install -e ./factosig`) "
                    "and ensure your working directory is not shadowing the package."
                ) from e

        X = self.preprocess_data(adata.X, row_weights)

        fs = FactoSig(
            n_factors=n_components,
            device=device,
            random_state=random_state,
            lr=lr,
            max_iter=max_iter,
            verbose=verbose,
            order_factors_by=order_factors_by,
        )
        fs.fit(X)

        loadings = np.asarray(fs.loadings_)
        scores = np.asarray(fs.scores_)
        psi = np.asarray(fs.psi_) if getattr(fs, "psi_", None) is not None else None

        adata.obsm["X_factosig"] = scores.astype(np.float32, copy=False)
        adata.varm["FACTOSIG_loadings"] = loadings.astype(np.float32, copy=False)
        if psi is not None and psi.size == adata.n_vars:
            adata.var["FACTOSIG_psi"] = psi.astype(np.float32, copy=False)

        # Factor score variance and SS loadings per factor
        factor_score_variances = np.var(scores, axis=0)
        ss_loadings_per_factor = np.sum(loadings**2, axis=0)

        adata.uns["factosig"] = {
            "n_factors": int(n_components),
            "random_state": int(random_state),
            "device": device,
            "lr": float(lr),
            "max_iter": int(max_iter),
            "rotation": "varimax",
            "order_factors_by": order_factors_by,
            "factor_score_variances": factor_score_variances,
            "sum_factor_score_variances": float(np.sum(factor_score_variances)),
            "ss_loadings_per_factor": ss_loadings_per_factor,
            "has_psi": psi is not None,
        }

        # Stash model for upstream save routines
        adata.uns["_temp_factosig_model_obj"] = fs

        # Optional save path
        if save_fitted_models and model_save_dir is not None:
            import os
            import pickle

            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f"factosig_model_{n_components}factors.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(fs, f)
            adata.uns["factosig"]["saved_model_paths"] = {"factosig_model": model_path}

        return adata


