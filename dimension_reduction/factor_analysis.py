# dimension_reduction/factor_analysis.py
import numpy as np
from sklearn.decomposition import FactorAnalysis as SklearnFA
from .base import DimensionReductionMethod

class FactorAnalysis(DimensionReductionMethod):
    """Factor Analysis implementation for single-cell data."""
    
    def fit_transform(self, adata, n_components=50, random_state=0, row_weights=None, save_model=False, save_dir=None):
        """
        Perform Factor Analysis on scRNA-seq data.
        
        Parameters:
        - adata: AnnData object to perform FA on
        - n_components: Number of latent factors to extract
        - random_state: Random seed for reproducibility
        - row_weights: Optional 1D np.array for cell weights (if None, no weighting)
        - save_model: Whether to save the FA model for later projection
        - save_dir: Directory to save the model
        
        Returns:
        - Updated AnnData with Factor Analysis results
        """
        # Preprocess data
        X = self.preprocess_data(adata.X, row_weights)
        
        # Initialize and fit Factor Analysis model
        fa = SklearnFA(n_components=n_components, random_state=random_state)
        X_fa = fa.fit_transform(X)
        
        # Store results in AnnData
        adata.obsm['X_fa'] = X_fa
        adata.varm['FA_loadings'] = fa.components_.T
        
        # Calculate and store explained variance
        explained_variance = np.var(X_fa, axis=0)
        total_variance = np.var(X, axis=0).sum()
        explained_variance_ratio = explained_variance / total_variance
        
        # Store FA parameters including the mean vector
        adata.uns['fa'] = {
            'mean': fa.mean_,
            'variance': explained_variance,
            'variance_ratio': explained_variance_ratio,
            'n_factors': n_components
        }
        
        # Save the full FA model if requested
        if save_model and save_dir is not None:
            # Extract patient ID if the adata o
            patient_id = None
            if 'patient' in adata.obs.columns:
                patient_values = adata.obs['patient'].unique()
                if len(patient_values) == 1:
                    patient_id = patient_values[0]
            
            # Save the model
            model_path = self.save_model(fa, save_dir, patient_id)
            print(f"Saved FA model to {model_path}")
        
        return adata