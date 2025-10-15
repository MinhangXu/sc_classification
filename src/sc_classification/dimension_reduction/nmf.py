# dimension_reduction/nmf.py
import numpy as np
import scipy.sparse as sp
import os
import pickle
from sklearn.decomposition import NMF as SklearnNMF
from .base import DimensionReductionMethod
import scanpy as sc
import pandas as pd
from typing import Dict, Any, Optional


class NMF(DimensionReductionMethod):
    """Non-negative Matrix Factorization implementation for single-cell data."""
    
    def fit_transform(self, adata, n_components=50, random_state=0, row_weights=None, 
                      save_model=False, save_dir=None, use_hvg=True, n_top_genes=2000,
                      beta_loss='kullback-leibler', standardize_input=False,
                      handle_negative_values='clip'):
        """
        Perform Non-negative Matrix Factorization on scRNA-seq data.
        
        Parameters:
        - adata: AnnData object to perform NMF on
        - n_components: Number of components to extract
        - random_state: Random seed for reproducibility
        - row_weights: Optional 1D np.array for cell weights (if None, no weighting)
        - save_model: Whether to save the NMF model for later projection
        - save_dir: Directory to save the model
        - use_hvg: Whether to use only highly variable genes (if True, expects HVGs to be pre-selected)
        - n_top_genes: Number of highly variable genes to select if calculation is needed (deprecated)
        - beta_loss: The beta divergence to use as the objective function
        - standardize_input: Whether input data is already standardized
        - handle_negative_values: How to handle negative values ('clip', 'error')
        
        Returns:
        - Updated AnnData with NMF results
        """
        # Make a working copy to avoid modifying the original
        adata_work = adata.copy()
        
        # Store original gene names for later mapping
        original_genes = adata_work.var_names.tolist()
        
        # Handle HVG selection
        if use_hvg:
            # Check if HVGs are already selected (from preprocessing)
            if 'highly_variable' in adata_work.var:
                n_hvgs = adata_work.var['highly_variable'].sum()
                print(f"Using {n_hvgs} pre-selected highly variable genes")
                # Filter for HVGs
                adata_work = adata_work[:, adata_work.var['highly_variable']].copy()
                print(f"Filtered data shape: {adata_work.shape}")
            else:
                raise ValueError("use_hvg=True but no 'highly_variable' annotation found in adata.var. "
                               "Please ensure HVGs are selected during preprocessing.")
        else:
            print(f"Using all {adata_work.n_vars} genes for NMF")
        
        # Store the genes used for NMF
        genes_used = adata_work.var_names.tolist()
        
        # Get the data matrix
        X = self.preprocess_data(adata_work.X, row_weights)
        
        # Handle negative values
        X = self._handle_negative_values(X, handle_negative_values)
        
        # Initialize and fit NMF model
        print(f"Fitting NMF model with {n_components} components...")
        nmf = SklearnNMF(
            n_components=n_components,
            init='nndsvda',  # Good initialization for sparse data
            beta_loss=beta_loss,  # Better for count data
            solver='mu',  # Multiplicative Update - compatible with KL divergence
            max_iter=500,
            random_state=random_state,
            tol=1e-4
        )
        
        X_nmf = nmf.fit_transform(X)
        
        # Store results in the original AnnData
        adata.obsm['X_nmf'] = X_nmf
        
        # Create components matrix for all genes
        # Map back the components to the original gene space
        all_components = np.zeros((len(original_genes), n_components))
        
        if use_hvg:
            # Get indices of HVG in the original adata
            hvg_indices = [i for i, gene in enumerate(original_genes) if gene in genes_used]
            all_components[hvg_indices, :] = nmf.components_.T
        else:
            all_components = nmf.components_.T
        
        adata.varm['NMF_components'] = all_components
        
        # Calculate and store explained variance (approximation for NMF)
        reconstruction = X_nmf @ nmf.components_
        residuals = X - reconstruction
        total_variance = np.var(X, axis=0).sum()
        residual_variance = np.var(residuals, axis=0).sum()
        explained_variance_ratio = 1 - (residual_variance / total_variance)
        
        # Store NMF parameters and metadata
        adata.uns['nmf'] = {
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'solver': 'mu',
            'beta_loss': beta_loss,
            'use_hvg': use_hvg,
            'genes_used': genes_used,
            'original_genes': original_genes,
            'standardized_input': standardize_input,
            'random_state': random_state,
            'handle_negative_values': handle_negative_values,
            'model_params': {
                'init': 'nndsvda',
                'max_iter': 500,
                'tol': 1e-4
            }
        }
        
        # Store factor score variances for consistency with FA
        factor_score_variances = np.var(X_nmf, axis=0)
        adata.uns['nmf']['factor_score_variances'] = factor_score_variances
        adata.uns['nmf']['sum_factor_score_variances'] = np.sum(factor_score_variances)
        
        # Calculate communalities (how much variance each gene explains)
        communalities = np.sum(all_components**2, axis=1)
        adata.var['communality'] = communalities
        
        # Store communality statistics
        adata.uns['nmf']['communalities_per_gene_summary'] = {
            'mean': np.mean(communalities),
            'std': np.std(communalities),
            'min': np.min(communalities),
            'max': np.max(communalities)
        }
        
        # Store SS loadings per factor
        ss_loadings_per_factor = np.sum(all_components**2, axis=0)
        adata.uns['nmf']['ss_loadings_per_factor'] = ss_loadings_per_factor
        
        # Save the full NMF model if requested
        if save_model and save_dir is not None:
            # Create model info dictionary
            model_info = {
                'model': nmf,
                'genes_used': genes_used,
                'original_genes': original_genes,
                'use_hvg': use_hvg,
                'n_components': n_components,
                'explained_variance_ratio': explained_variance_ratio,
                'beta_loss': beta_loss,
                'standardized_input': standardize_input,
                'handle_negative_values': handle_negative_values
            }
            
            # Save the model
            model_path = self.save_model(model_info, save_dir)
            print(f"Saved NMF model to {model_path}")
            
            # Store path in AnnData for later reference
            adata.uns['nmf']['saved_model_paths'] = {'nmf_model': model_path}
        
        print("NMF completed successfully")
        return adata
    
    def _handle_negative_values(self, X, method='error'):
        """
        Handle negative values in the data matrix for NMF.
        
        Parameters:
        - X: Input data matrix
        - method: Method to handle negative values
            - 'clip': Clip negative values to 0
            - 'error': Raise error if negative values found (default)
        
        Returns:
        - Processed data matrix
        """
        if method == 'error':
            if np.any(X < 0):
                raise ValueError("Negative values found in data. NMF requires non-negative input. "
                               "This is often caused by standardization. Try running the pipeline "
                               "with the --no-std flag or set handle_negative_values='clip'.")
            return X
        
        elif method == 'clip':
            if np.any(X < 0):
                n_negative = np.sum(X < 0)
                print(f"Clipping {n_negative} negative values to 0")
                X = np.clip(X, 0, None)
            return X
        
        else:
            raise ValueError(f"Unknown method '{method}' for handling negative values. "
                           "Use 'clip' or 'error'.")
    
    def save_model(self, model_info, save_dir, patient_id=None):
        """
        Save the NMF model and related information to disk.
        
        Parameters:
        - model_info: Dictionary containing the model and related information
        - save_dir: Directory to save the model
        - patient_id: Optional patient ID for model filename
        
        Returns:
        - Path to the saved model file
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Determine filename based on patient ID
        if patient_id is not None:
            model_path = os.path.join(save_dir, f"nmf_model_{patient_id}.pkl")
        else:
            model_path = os.path.join(save_dir, "nmf_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        return model_path
    
    def project_new_data(self, adata_new, model_path, use_hvg=True):
        """
        Project new data onto the NMF components learned from a reference dataset.
        
        Parameters:
        - adata_new: New AnnData object to project
        - model_path: Path to the saved NMF model
        - use_hvg: Whether to use only highly variable genes for projection
        
        Returns:
        - Updated AnnData with NMF projection results
        """
        # Load the model and related information
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        
        nmf_model = model_info['model']
        genes_used = model_info['genes_used']
        handle_negative_values = model_info.get('handle_negative_values', 'clip')
        
        # Make a working copy of the data
        adata_work = adata_new.copy()
        
        # Filter for the genes used in the original NMF model
        common_genes = [gene for gene in genes_used if gene in adata_work.var_names]
        adata_work = adata_work[:, common_genes].copy()
        
        print(f"Projecting new data with {len(common_genes)} genes from the original model")
        
        # Get the data matrix
        X = adata_work.X
        if sp.issparse(X):
            X = X.toarray()
        
        # Apply the same negative value handling as during training
        X = self._handle_negative_values(X, handle_negative_values)
        
        # Project the new data
        X_nmf_new = nmf_model.transform(X)
        
        # Store the projection in the original AnnData
        adata_new.obsm['X_nmf'] = X_nmf_new
        
        # Store the model information
        adata_new.uns['nmf_projection'] = {
            'n_components': model_info['n_components'],
            'genes_used': common_genes,
            'original_genes_used': genes_used,
            'handle_negative_values': handle_negative_values,
            'projection_date': pd.Timestamp.now().isoformat()
        }
        
        return adata_new
    
    def generate_model_summary(self, adata) -> str:
        """
        Generate a comprehensive summary of the NMF model.
        
        Parameters:
        - adata: AnnData object with NMF results
        
        Returns:
        - String summary of the model
        """
        nmf_info = adata.uns.get('nmf', {})
        
        summary_lines = [
            f"NMF Model Summary",
            "=" * 50,
            f"Number of components: {nmf_info.get('n_components', 'N/A')}",
            f"Explained variance ratio: {nmf_info.get('explained_variance_ratio', 'N/A'):.4f}",
            f"Beta loss function: {nmf_info.get('beta_loss', 'N/A')}",
            f"Solver: {nmf_info.get('solver', 'N/A')}",
            f"Used HVGs: {nmf_info.get('use_hvg', 'N/A')}",
            f"Number of genes used: {len(nmf_info.get('genes_used', []))}",
            f"Standardized input: {nmf_info.get('standardized_input', 'N/A')}",
            f"Negative value handling: {nmf_info.get('handle_negative_values', 'N/A')}",
            "",
            "Factor Score Statistics:",
            f"  Sum of factor score variances: {nmf_info.get('sum_factor_score_variances', 'N/A'):.4f}",
            f"  Mean factor score variance: {np.mean(nmf_info.get('factor_score_variances', [])):.4f}",
            f"  Min factor score variance: {np.min(nmf_info.get('factor_score_variances', [])):.4f}",
            f"  Max factor score variance: {np.max(nmf_info.get('factor_score_variances', [])):.4f}",
            "",
            "Communality Statistics:",
        ]
        
        comm_summary = nmf_info.get('communalities_per_gene_summary', {})
        for stat, value in comm_summary.items():
            summary_lines.append(f"  {stat.capitalize()}: {value:.4f}")
        
        summary_lines.extend([
            "",
            "SS Loadings per Factor:",
            f"  Sum of SS loadings: {np.sum(nmf_info.get('ss_loadings_per_factor', [])):.4f}",
            f"  Mean SS loading: {np.mean(nmf_info.get('ss_loadings_per_factor', [])):.4f}",
        ])
        
        return "\n".join(summary_lines)