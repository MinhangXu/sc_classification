# pipeline/all_patients.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..dimension_reduction.factor_analysis import FactorAnalysis
from ..dimension_reduction.nmf import NMF
from ..classification.lr_lasso import LRLasso
from ..evaluation.visualization import plot_metrics_and_coefficients

def run_all_patients_pipeline(
    adata, 
    dr_method='factor_analysis', 
    n_components=100,
    use_row_weights=False,
    row_weights_beta=0.25,
    alphas=None,
    output_dir="results/all_patients"
):
    """
    Run the dimension reduction and classification pipeline for all patients combined.
    
    Parameters:
    - adata: AnnData object with all patients' data
    - dr_method: Dimension reduction method ('factor_analysis' or 'nmf')
    - n_components: Number of components to extract
    - use_row_weights: Whether to use row weights
    - row_weights_beta: Beta parameter for row weights calculation
    - alphas: Array of regularization strengths (if None, default values are used)
    - output_dir: Directory to save results
    
    Returns:
    - Dictionary of results
    """
    # Create output directories
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Calculate row weights if requested
    row_weights = None
    if use_row_weights:
        row_weights = compute_row_weights_by_group(adata, beta=row_weights_beta, mode='patient')
        adata.obs['row_weights'] = row_weights
    
    # Run dimension reduction
    if dr_method == 'factor_analysis':
        dr = FactorAnalysis()
        adata_dr = dr.fit_transform(
            adata, 
            n_components=n_components, 
            random_state=0, 
            row_weights=row_weights,
            save_model=True,
            save_dir=model_dir
        )
        dr_obsm_key = 'X_fa'
    elif dr_method == 'nmf':
        dr = NMF()
        adata_dr = dr.fit_transform(
            adata, 
            n_components=n_components, 
            random_state=0, 
            row_weights=row_weights,
            save_model=True,
            save_dir=model_dir
        )
        dr_obsm_key = 'X_nmf'
    else:
        raise ValueError(f"Unsupported dimension reduction method: {dr_method}")
    
    # Save the dimension-reduced data
    dr_output_path = os.path.join(output_dir, f"adata_all_patients_{dr_method}.h5ad")
    adata_dr.write_h5ad(dr_output_path)
    
    # Run LR-Lasso classification
    classifier = LRLasso(adata_dr, target_col='CN.label', target_value='cancer')
    X, y, feature_names, valid_X_original = classifier.prepare_data(
        use_factorized=True, 
        factorization_method=dr_obsm_key, 
        selected_features=None
    )
    
    # Set default alphas if not provided
    if alphas is None:
        alphas = np.logspace(-4, 5, 20)
    
    # Fit along regularization path
    print("Running LR-Lasso classification for all patients...")
    results = classifier.fit_along_regularization_path(
        X, y, feature_names, alphas=alphas, metrics_grouping='patient'
    )
    
    # Save coefficient results
    coefs_df = pd.DataFrame(
        results['coefs'],
        index=feature_names,
        columns=alphas
    )
    coefs_filename = os.path.join(output_dir, f"results_coefs_{dr_method}_all_patients.csv")
    coefs_df.to_csv(coefs_filename, index=True)
    
    # Save metrics results
    metrics_df = pd.DataFrame(results['group_metrics_path'])
    metrics_filename = os.path.join(output_dir, f"results_metrics_{dr_method}_all_patients.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    
    # Generate visualizations
    plt.figure(figsize=(12, 8))
    plot_metrics_and_coefficients(
        metrics_df, 
        coefs_df, 
        sample_name="all_samples", 
        top_n=10
    )
    plt.savefig(os.path.join(output_dir, f"metrics_and_coefs_{dr_method}_all_patients.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create per-patient visualizations
    for patient_id in adata.obs['patient'].unique():
        plt.figure(figsize=(12, 8))
        plot_metrics_and_coefficients(
            metrics_df, 
            coefs_df, 
            sample_name=patient_id, 
            top_n=10
        )
        plt.savefig(os.path.join(output_dir, f"metrics_and_coefs_{dr_method}_patient_{patient_id}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return {
        'dr_method': dr_method,
        'adata_dr': adata_dr,
        'results': results,
        'coefs_df': coefs_df,
        'metrics_df': metrics_df
    }

def compute_row_weights_by_group(adata, beta=1.0, mode='patient'):
    """
    Compute per-cell weights for an AnnData object using a group-level metric.
    
    Parameters:
    - adata: AnnData object with adata.obs containing mode and 'CN.label'
    - beta: Weighting parameter in [0,1] for normal cells
    - mode: Either 'patient' or 'sample', specifying the grouping level
    
    Returns:
    - row_weights: 1D numpy array of shape (n_obs,), with each cell's computed weight
    """
    row_weights = np.zeros(adata.n_obs)
    
    if mode not in ['patient', 'sample']:
        raise ValueError("Mode must be either 'patient' or 'sample'")
    
    # Group cells by the specified grouping (patient or sample)
    for group, group_df in adata.obs.groupby(mode, observed=False):
        # Create a boolean mask for the current group
        if mode == 'patient':
            mask = adata.obs['patient'] == group
        else:  # mode == 'sample'
            mask = adata.obs['sample'] == group
        
        # Identify malignant cells (CN.label == 'cancer') in the current group
        malignant_mask = adata.obs.loc[mask, 'CN.label'] == 'cancer'
        M = malignant_mask.sum()       # Number of malignant cells
        N = (~malignant_mask).sum()    # Number of normal cells
        
        # Calculate denominator, with beta scaling the normal cell count
        denominator = beta * N + M
        if denominator == 0:
            denominator = 1.0  # Fallback if group is empty or undefined
        
        # Compute weight for every cell in the group
        weight = 1.0 / denominator
        row_weights[mask] = weight
        
    return row_weights