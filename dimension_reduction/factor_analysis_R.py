import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle
import tempfile

from .base import DimensionReductionMethod

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False

class FactorAnalysis_R(DimensionReductionMethod):
    """
    Factor Analysis using R's 'psych' package with bootstrapped confidence intervals.
    """

    def __init__(self):
        if not RPY2_AVAILABLE:
            raise ImportError("rpy2 is not installed. Please install it to use FactorAnalysis_R.")
        
        try:
            self.psych = importr('psych')
        except Exception as e:
            raise ImportError(f"R package 'psych' not found. Please install it in your R environment. Details: {e}")

    def fit_transform(self, adata, n_components=100, random_state=0,
                      standardize_input=True,
                      save_fitted_models=False,
                      model_save_dir=None,
                      row_weights=None,
                      fm="ml", rotate="varimax", n_iter=100):
        """
        Perform FA using R's psych::fa.ci to get bootstrapped standard errors.
        """
        X_orig = self.preprocess_data(adata.X, row_weights)
        current_scaler = None

        if standardize_input:
            print("Standardizing input data before R Factor Analysis.")
            current_scaler = StandardScaler(with_mean=True)
            X_scaled = current_scaler.fit_transform(X_orig)
        else:
            X_scaled = X_orig

        X_df = pd.DataFrame(X_scaled, index=adata.obs_names, columns=adata.var_names)

        with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=True) as tmp_file:
            print(f"Writing data to temporary file for R: {tmp_file.name}")
            X_df.to_csv(tmp_file.name)
            
            ro.r(f"r_df <- read.csv('{tmp_file.name}', row.names=1, check.names=FALSE)")

            print(f"Running Factor Analysis with R 'psych' (fm='{fm}', n_iter={n_iter})...")
            ro.r(f'set.seed({random_state})')
            
            fa_results_r = self.psych.fa_ci(ro.r['r_df'], nfactors=n_components, n_iter=n_iter, rotate=rotate, fm=fm)
            fa_original_results = fa_results_r.rx2('fa')

        with localconverter(ro.default_converter + pandas2ri.converter):
            loadings_df = ro.conversion.rpy2py(fa_results_r.rx2('loadings'))
            loadings_se = ro.conversion.rpy2py(fa_results_r.rx2('se'))
            scores = ro.conversion.rpy2py(fa_original_results.rx2('scores'))
            weights = ro.conversion.rpy2py(fa_original_results.rx2('weights'))

            if scores is None:
                print("Scores not returned by R, calculating from weights.")
                scores = X_scaled @ np.array(weights)

            communality = ro.conversion.rpy2py(fa_original_results.rx2('communality'))
            uniquenesses = ro.conversion.rpy2py(fa_original_results.rx2('uniquenesses'))

        factor_names = [f'Factor{i+1}' for i in range(n_components)]
        adata.obsm['X_fa'] = pd.DataFrame(scores, index=adata.obs_names, columns=factor_names)
        adata.varm['FA_loadings'] = pd.DataFrame(np.array(loadings_df), index=adata.var_names, columns=factor_names)
        adata.var['communality'] = pd.Series(communality, index=adata.var_names)
        adata.var['uniqueness'] = pd.Series(uniquenesses, index=adata.var_names)
        
        model_to_save = {
            'loadings': pd.DataFrame(np.array(loadings_df), index=adata.var_names, columns=factor_names),
            'loadings_se': pd.DataFrame(np.array(loadings_se), index=adata.var_names, columns=factor_names),
            'score_weights': pd.DataFrame(np.array(weights), index=adata.var_names, columns=factor_names),
            'gene_means': pd.Series(current_scaler.mean_, index=adata.var_names) if current_scaler else pd.Series(np.mean(X_orig, axis=0), index=adata.var_names),
            'gene_stds': pd.Series(current_scaler.scale_, index=adata.var_names) if current_scaler else pd.Series(np.std(X_orig, axis=0), index=adata.var_names),
            'communality': pd.Series(communality, index=adata.var_names),
            'uniqueness': pd.Series(uniquenesses, index=adata.var_names),
            'n_components': n_components,
            'fm': fm, 'rotate': rotate, 'n_iter': n_iter
        }
        
        adata.uns['fa_r'] = {
            'n_factors': n_components, 'random_state': random_state,
            'fm': fm, 'rotate': rotate, 'n_iter': n_iter,
            'standardized_input': standardize_input
        }

        adata.uns['_temp_fa_r_model_obj'] = model_to_save
        adata.uns['_temp_scaler_obj'] = current_scaler
        
        if save_fitted_models and model_save_dir:
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, f"fa_r_model_{n_components}factors.pkl")
            with open(model_path, 'wb') as f: pickle.dump(model_to_save, f)
            adata.uns['fa_r']['saved_model_path'] = model_path
            
            if current_scaler:
                scaler_path = os.path.join(model_save_dir, f"scaler_{n_components}factors.pkl")
                with open(scaler_path, 'wb') as f: pickle.dump(current_scaler, f)
                adata.uns['fa_r']['saved_scaler_path'] = scaler_path

        return adata
