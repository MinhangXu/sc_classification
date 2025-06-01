# dimension_reduction/factor_analysis.py
import numpy as np
from sklearn.decomposition import FactorAnalysis as SklearnFA
from .base import DimensionReductionMethod
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import pickle

class FactorAnalysis(DimensionReductionMethod):
    """Factor Analysis implementation for single-cell data."""
    
    def fit_transform(self, adata, n_components=50, random_state=0, 
                      standardize_input=True, 
                      svd_method='lapack',
                      save_fitted_models=False, # New: To control saving of fa & scaler
                      model_save_dir=None,     # New: Directory to save models
                      row_weights=None): # Removed save_model, save_dir from here
        """
        Perform Factor Analysis on scRNA-seq data.
        ... (docstring params as before) ...
        - save_fitted_models: If True, save the FA model and scaler object.
        - model_save_dir: Directory to save model objects if save_fitted_models is True.
        """
        X_orig = self.preprocess_data(adata.X, row_weights) 
        current_scaler = None # Initialize scaler

        if standardize_input:
            print("Standardizing input data (genes to mean=0, variance=1) before Factor Analysis.")
            current_scaler = StandardScaler(with_mean=True) 
            X_scaled = current_scaler.fit_transform(X_orig)
        else:
            X_scaled = X_orig

        print(f"Using SVD method: {svd_method} for SklearnFA.")
        fa = SklearnFA(n_components=n_components, random_state=random_state, svd_method=svd_method)
        
        adata.obsm['X_fa'] = fa.fit_transform(X_scaled) 
        adata.varm['FA_loadings'] = fa.components_.T 
        communalities = np.sum(np.square(adata.varm['FA_loadings']), axis=1)
        adata.var['communality'] = communalities
        ss_loadings_per_factor = np.sum(np.square(fa.components_), axis=1)

        # Store model objects directly in .uns for the pipeline to access and save
        # These won't be saved with adata.write_h5ad() typically.
        model_details_for_saving = {
            'fa_model_obj': fa,
            'scaler_obj': current_scaler,
            'hvg_var_names_input_order': adata.var_names.tolist() # Order of genes FA was trained on
        }
        
        # Storing metadata in .uns is fine for h5ad
        adata.uns['fa'] = {
            'model_mean_on_input': fa.mean_, # Mean of (potentially scaled) input features used by FA
            'n_factors': n_components,
            'random_state': random_state,
            'standardized_input_in_class_call': standardize_input, # What this specific call did
            'svd_method_used': svd_method,
            'factor_score_variances': np.var(adata.obsm['X_fa'], axis=0),
            'sum_factor_score_variances': np.sum(np.var(adata.obsm['X_fa'], axis=0)),
            'communalities_per_gene_summary': {
                'mean': np.mean(communalities), 'median': np.median(communalities),
                'min': np.min(communalities), 'max': np.max(communalities),
                'std': np.std(communalities)
            },
            'ss_loadings_per_factor': ss_loadings_per_factor,
            'noise_variance_per_feature': fa.noise_variance_,
            # Add a placeholder for paths if saved by pipeline
            'saved_model_paths': {} 
        }
        
        # The pipeline script will handle the actual saving using these objects.
        # This allows the FactorAnalysis class to not depend on specific saving paths
        # directly passed into fit_transform, but rather through dedicated params.
        if save_fitted_models:
            if model_save_dir is None:
                print("Warning: save_fitted_models is True, but model_save_dir is None. Models not saved.")
            else:
                os.makedirs(model_save_dir, exist_ok=True)
                fa_model_path = self.save_sklearn_model(fa, "fa_model", model_save_dir, n_components)
                adata.uns['fa']['saved_model_paths']['fa_model'] = fa_model_path
                if current_scaler:
                    scaler_path = self.save_sklearn_model(current_scaler, "scaler", model_save_dir, n_components) # n_components here is just for filename consistency
                    adata.uns['fa']['saved_model_paths']['scaler'] = scaler_path
                
                # HVG list can be saved by pipeline, but storing order here is useful
                adata.uns['fa']['hvg_var_names_input_order'] = adata.var_names.tolist()


        # Temporary storage of actual model objects for immediate use by pipeline if needed
        # before saving anndata (which won't save these complex objects effectively)
        adata.uns['_temp_fa_model_obj'] = fa 
        adata.uns['_temp_scaler_obj'] = current_scaler

        return adata