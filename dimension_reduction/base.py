# dimension_reduction/base.py
from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import os
import pickle
from anndata import AnnData

class DimensionReductionMethod(ABC):
    """Base class for all dimension reduction methods."""
    
    @abstractmethod
    def fit_transform(self, adata, n_components=50, random_state=0, row_weights=None):
        """
        Fit the dimension reduction model and transform the data.
        
        Parameters:
        - adata: AnnData object to perform dimension reduction on
        - n_components: Number of components to extract
        - random_state: Random seed for reproducibility
        - row_weights: Optional 1D np.array for cell weights (if None, no weighting)
        
        Returns:
        - Updated AnnData with dimension reduction results
        """
        pass
    
    def preprocess_data(self, X_data, row_weights=None):
        # Basic preprocessing, can be overridden or made more complex
        if hasattr(X_data, 'toarray'): # Check if sparse
            X_data = X_data.toarray()
        # Further preprocessing like scaling could go here if not handled elsewhere
        return X_data

    def save_model(self, model, save_dir, patient_id=None):
        # Placeholder for saving model logic if needed
        import pickle
        import os
        model_filename = "fa_model.pkl"
        if patient_id:
            model_filename = f"fa_model_{patient_id}.pkl"
        model_path = os.path.join(save_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path