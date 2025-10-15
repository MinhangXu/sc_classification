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
        if hasattr(X_data, 'toarray'):
            X_data = X_data.toarray()
        return X_data

    def save_sklearn_model(self, model, model_type, save_dir, n_factors, patient_id=None):
        # Generalized model saving
        filename_parts = [model_type, f"{n_factors}factors"]
        if patient_id: # This might not be relevant for global FA models/scalers
            filename_parts.append(patient_id)
        model_filename = "_".join(filename_parts) + ".pkl"
        
        model_path = os.path.join(save_dir, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {model_type} model to {model_path}")
        return model_path