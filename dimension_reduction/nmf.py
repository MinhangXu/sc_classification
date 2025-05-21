# dimension_reduction/nmf.py
import numpy as np
import scipy.sparse as sp
import os
import pickle
from sklearn.decomposition import NMF as SklearnNMF
from .base import DimensionReductionMethod
import scanpy as sc

class NMF(DimensionReductionMethod):
    """Non-negative Matrix Factorization implementation for single-cell data."""
    
    def fit_transform(self, adata, n_components=50, random_state=0, row_weights=None, 
                      save_model=False, save_dir=None, use_hvg=True, n_top_genes=2000,
                      beta_loss='kullback-leibler'):
        """
        Perform Non-negative Matrix Factorization on scRNA-seq data.
        
        Parameters:
        - adata: AnnData object to perform NMF on
        - n_components: Number of components to extract
        - random_state: Random seed for reproducibility
        - row_weights: Optional 1D np.array for cell weights (if None, no weighting)
        - save_model: Whether to save the NMF model for later projection
        - save_dir: Directory to save the model
        - use_hvg: Whether to use only highly variable genes for the analysis
        - n_top_genes: Number of highly variable genes to select if calculation is needed
        - beta_loss: The beta divergence to use as the objective function
        
        Returns:
        - Updated AnnData with NMF results
        """
        # Make a working copy to avoid modifying the original
        adata_work = adata.copy()
        
        # Filter for highly variable genes if requested
        if use_hvg:
            if 'highly_variable' in adata_work.var:
                print(f"Using {adata_work.var['highly_variable'].sum()} existing highly variable genes")
            else:
                print(f"Calculating {n_top_genes} highly variable genes")
                sc.pp.highly_variable_genes(adata_work, n_top_genes=n_top_genes)
                print(f"Identified {adata_work.var['highly_variable'].sum()} highly variable genes")
            
            # Filter for HVGs
            adata_work = adata_work[:, adata_work.var['highly_variable']].copy()
            print(f"Filtered data shape: {adata_work.shape}")
        
        # Store the genes used for NMF
        genes_used = adata_work.var_names.tolist()
        
        # Get the data matrix
        X = self.preprocess_data(adata_work.X, row_weights)
        
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
        all_components = np.zeros((adata.n_vars, n_components))
        
        if use_hvg:
            # Get indices of HVG in the original adata
            hvg_indices = np.where(adata.var['highly_variable'])[0]
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
        
        # Store NMF parameters
        adata.uns['nmf'] = {
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'solver': 'mu',
            'beta_loss': beta_loss,
            'use_hvg': use_hvg,
            'genes_used': genes_used
        }
        
        # Save the full NMF model if requested
        if save_model and save_dir is not None:
            # Extract patient ID if available
            patient_id = None
            if 'patient' in adata.obs.columns:
                patient_values = adata.obs['patient'].unique()
                if len(patient_values) == 1:
                    patient_id = patient_values[0]
            
            # Create the model info dictionary
            model_info = {
                'model': nmf,
                'genes_used': genes_used,
                'use_hvg': use_hvg,
                'n_components': n_components,
                'explained_variance_ratio': explained_variance_ratio
            }
            
            # Save the model
            model_path = self.save_model(model_info, save_dir, patient_id)
            print(f"Saved NMF model to {model_path}")
        
        print("NMF completed successfully")
        return adata
    
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
        
        # Project the new data
        X_nmf_new = nmf_model.transform(X)
        
        # Store the projection in the original AnnData
        adata_new.obsm['X_nmf'] = X_nmf_new
        
        # Store the model information
        adata_new.uns['nmf_projection'] = {
            'n_components': model_info['n_components'],
            'genes_used': common_genes
        }
        
        return adata_new