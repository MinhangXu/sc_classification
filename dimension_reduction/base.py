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
    
    def save_model(self, model, save_dir, patient_id=None):
        """
        Save the fitted model to disk.
        
        Parameters:
        - model: The fitted model object
        - save_dir: Directory to save the model
        - patient_id: Optional patient ID for model filename
        
        Returns:
        - Path to the saved model file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine filename based on patient ID
        if patient_id is not None:
            model_path = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_model_{patient_id}.pkl")
        else:
            model_path = os.path.join(save_dir, f"{self.__class__.__name__.lower()}_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model_path
    
    def preprocess_data(self, X, row_weights=None):
        """
        Preprocess data for dimension reduction.
        
        Parameters:
        - X: Data matrix
        - row_weights: Optional 1D np.array for cell weights
        
        Returns:
        - Preprocessed data matrix
        """
        # Ensure data is dense
        if sp.issparse(X):
            X = X.toarray()
        
        # Apply row-scaling if weights provided
        if row_weights is not None:
            w_sqrt = np.sqrt(row_weights)
            X = X * w_sqrt[:, None]
        
        return X