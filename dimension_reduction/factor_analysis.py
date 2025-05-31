# dimension_reduction/factor_analysis.py
import numpy as np
from sklearn.decomposition import FactorAnalysis as SklearnFA
from .base import DimensionReductionMethod
from sklearn.preprocessing import StandardScaler

class FactorAnalysis(DimensionReductionMethod):
    """Factor Analysis implementation for single-cell data."""
    
    def fit_transform(self, adata, n_components=50, random_state=0, 
                      standardize_input=True, # New parameter
                      row_weights=None, save_model=False, save_dir=None):
        """
        Perform Factor Analysis on scRNA-seq data.
        
        Parameters:
        - adata: AnnData object to perform FA on.
        - n_components: Number of latent factors to extract.
        - random_state: Random seed for reproducibility.
        - standardize_input: If True, standardize input data (genes) to unit variance.
        - row_weights: Optional 1D np.array for cell weights (not used by SklearnFA directly).
        - save_model: Whether to save the FA model.
        - save_dir: Directory to save the model.
        
        Returns:
        - Updated AnnData with Factor Analysis results.
        """
        # Preprocess data (basic handling of sparse/dense)
        X_orig = self.preprocess_data(adata.X, row_weights) 

        if standardize_input:
            print("Standardizing input data (genes to mean=0, variance=1) before Factor Analysis.")
            scaler = StandardScaler(with_mean=True) # with_mean=True is fine for dense corrected data
            X_scaled = scaler.fit_transform(X_orig)
            # Store scaler info if you might need to inverse_transform later (optional)
            # adata.uns['fa_scaler'] = scaler 
        else:
            X_scaled = X_orig
        
        # Initialize and fit Factor Analysis model
        fa = SklearnFA(n_components=n_components, random_state=random_state, svd_method='lapack')
        
        # X_fa are the factor scores for each cell
        adata.obsm['X_fa'] = fa.fit_transform(X_scaled) 
        
        # fa.components_ are the factor loadings (n_components, n_features/genes)
        # Store as (n_features, n_components) in .varm
        adata.varm['FA_loadings'] = fa.components_.T 
        
        # --- New Metrics ---
        # Communalities: sum of squared loadings for each gene
        # h_g^2 = sum_j (lambda_gj^2)
        communalities = np.sum(np.square(adata.varm['FA_loadings']), axis=1)
        adata.var['communality'] = communalities
        
        # Sum of Squared Loadings (SS Loadings) per factor:
        # V_j = sum_g (lambda_gj^2)
        ss_loadings_per_factor = np.sum(np.square(fa.components_), axis=1) # Sum across genes for each factor

        # --- Store results in AnnData.uns ---
        adata.uns['fa'] = {
            'model_mean_on_input': fa.mean_, # Mean of (potentially scaled) input features
            'n_factors': n_components,
            'random_state': random_state,
            'standardized_input': standardize_input,
            
            # Metrics related to factor scores
            'factor_score_variances': np.var(adata.obsm['X_fa'], axis=0),
            'sum_factor_score_variances': np.sum(np.var(adata.obsm['X_fa'], axis=0)),
            
            # Metrics related to original variables (genes)
            'communalities_per_gene_summary': {
                'mean': np.mean(communalities),
                'median': np.median(communalities),
                'min': np.min(communalities),
                'max': np.max(communalities),
                'std': np.std(communalities)
            },
            'ss_loadings_per_factor': ss_loadings_per_factor, # V_j values
            'noise_variance_per_feature': fa.noise_variance_ # Uniqueness for each gene
        }
        
        if save_model and save_dir is not None:
            patient_id = None
            if 'patient' in adata.obs.columns: # Assuming 'patient' column exists
                patient_values = adata.obs['patient'].unique()
                if len(patient_values) == 1:
                    patient_id = patient_values[0]
            model_path = self.save_model(fa, save_dir, patient_id)
            print(f"Saved FA model (fit on potentially scaled data) to {model_path}")
        
        return adata