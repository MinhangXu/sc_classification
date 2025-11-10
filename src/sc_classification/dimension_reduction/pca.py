import numpy as np
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from .base import DimensionReductionMethod


class PCA(DimensionReductionMethod):
    """Principal Component Analysis wrapper with consistent AnnData outputs."""

    def fit_transform(
        self,
        adata,
        n_components: int = 50,
        random_state: int = 0,
        row_weights=None,
        standardize_input: bool = True,
        svd_solver: str = "auto",
        whiten: bool = False,
        save_fitted_models: bool = False,
        model_save_dir: str = None,
    ):
        """
        Perform PCA on the provided AnnData.

        Parameters:
        - adata: AnnData
        - n_components: number of PCs
        - random_state: seed
        - standardize_input: if True, z-score genes prior to PCA
        - svd_solver: sklearn PCA svd_solver ('auto', 'full', 'arpack', 'randomized')
        - whiten: if True, scale PC scores by 1/singular_values so output PCs have unit variance
        - save_fitted_models: save PCA model (and scaler if used)
        - model_save_dir: directory to save models

        Returns:
        - AnnData updated with:
          obsm['X_pca'] (cells × k), varm['PCA_loadings'] (genes × k), uns['pca'] (metadata)
        """
        X = self.preprocess_data(adata.X, row_weights)
        current_scaler = None

        if standardize_input:
            current_scaler = StandardScaler(with_mean=True)
            X_proc = current_scaler.fit_transform(X)
        else:
            X_proc = X

        pca = SklearnPCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=whiten,
            random_state=random_state,
        )

        scores = pca.fit_transform(X_proc)  # (n_cells, k)
        # sklearn components_ has shape (k, p); loadings as (p, k)
        loadings = pca.components_.T

        adata.obsm["X_pca"] = scores
        adata.varm["PCA_loadings"] = loadings

        # Communality proxy: sum of squared loadings per gene (well-defined if input standardized)
        communality = np.sum(loadings**2, axis=1)
        adata.var["communality"] = communality

        ss_loadings_per_factor = np.sum(pca.components_**2, axis=1)  # per factor
        factor_score_variances = np.var(scores, axis=0)

        adata.uns["pca"] = {
            "n_components": int(n_components),
            "random_state": int(random_state),
            "svd_solver": svd_solver,
            "whiten": bool(whiten),
            "standardized_input_in_class_call": bool(standardize_input),
            "explained_variance": np.asarray(pca.explained_variance_),
            "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_),
            "singular_values": np.asarray(pca.singular_values_),
            "factor_score_variances": factor_score_variances,
            "sum_factor_score_variances": float(np.sum(factor_score_variances)),
            "ss_loadings_per_factor": ss_loadings_per_factor,
        }

        # Temporarily stash model/scaler for upstream save routines
        adata.uns["_temp_pca_model_obj"] = pca
        adata.uns["_temp_scaler_obj"] = current_scaler

        # Optional: save models via base helper
        if save_fitted_models and model_save_dir is not None:
            import os

            os.makedirs(model_save_dir, exist_ok=True)
            pca_model_path = self.save_sklearn_model(pca, "pca_model", model_save_dir, n_components)
            adata.uns["pca"]["saved_model_paths"] = {"pca_model": pca_model_path}
            if current_scaler is not None:
                scaler_path = self.save_sklearn_model(current_scaler, "scaler", model_save_dir, n_components)
                adata.uns["pca"]["saved_model_paths"]["scaler"] = scaler_path

        return adata


