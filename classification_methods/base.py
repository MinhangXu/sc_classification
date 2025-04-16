# classification/base.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Classifier(ABC):
    """Base class for all classifiers."""
    
    def __init__(self, adata, target_col='CN.label', target_value='cancer'):
        """
        Initialize the classifier.
        
        Parameters:
        - adata: AnnData object
        - target_col: Column in adata.obs to be used as the target variable
        - target_value: The value in target_col to be treated as the positive class
        """
        self.adata = adata
        self.target_col = target_col
        self.target_value = target_value
    
    @abstractmethod
    def prepare_data(self, use_factorized=True, factorization_method='X_fa', selected_features=None):
        """
        Prepare the feature matrix (X), target labels (y), and filtered original feature matrix.
        
        Parameters:
        - use_factorized: Whether to use dimensionality-reduced features
        - factorization_method: The key in adata.obsm for the reduced features
        - selected_features: List of feature names to use (if not using factorized)
        
        Returns:
        - X: The processed feature matrix
        - y: The binary target vector
        - feature_names: A list of feature names
        - valid_X_original: The original feature matrix filtered for valid rows
        """
        pass
    
    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        """
        Fit the classifier to the data.
        
        Parameters:
        - X: Feature matrix
        - y: Target vector
        - *args, **kwargs: Additional arguments
        
        Returns:
        - Fitted classifier
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the fitted classifier.
        
        Parameters:
        - X: Feature matrix
        
        Returns:
        - Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X, y, *args, **kwargs):
        """
        Evaluate the classifier.
        
        Parameters:
        - X: Feature matrix
        - y: True labels
        - *args, **kwargs: Additional arguments
        
        Returns:
        - Evaluation metrics
        """
        pass