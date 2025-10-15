# pipeline/single_patient.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..dimension_reduction.factor_analysis import FactorAnalysis
from ..dimension_reduction.nmf import NMF
from ..classification.lr_lasso import LRLasso
from ..evaluation.visualization import plot_metrics_and_coefficients

def run_single_patient_pipeline(
    adata, 
    patient_id, 
    dr_method='factor_analysis', 
    n_components=50,
    use_row_weights=False,
    row_weights_beta=0.25,
    alphas=None,
    output_dir="results/single_patient"
):
    """
    Run the dimension reduction and classification pipeline for a single patient.
    
    Parameters:
    - adata: AnnData object with all patients' data
    - patient_id: ID of the patient to analyze
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
    patient_dir = os.path.join(output_dir, f"patient_{patient_id}")
    model_dir = os.path.join(patient_dir, "models")
    os.makedirs(patient_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Filter data for the specific patient
    patient_adata = adata[adata.obs['patient'] == patient_id].copy()
    print(f"Analyzing patient {patient_id}: {patient_adata.n_obs} cells")
    
    # Calculate row weights if requested
    row_weights = None
    if use_row_weights:
        row_weights = compute_row_weights(patient_adata, beta=row_weights_beta)
        patient_adata.obs['row_weights'] = row_weights
    
    # Run dimension reduction
    if dr_method == 'factor_analysis':
        dr = FactorAnalysis()
        patient_adata_dr = dr.fit_transform(
            patient_adata, 
            n_components=n_components, 
            random_state=0, 
            row_weights=row_weights,
            save_model=True,
            save_dir=model_dir
        )
        dr_obsm_key = 'X_fa'
    elif dr_method == 'nmf':
        dr = NMF()
        patient_adata_dr = dr.fit_transform(
            patient_adata, 
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
    dr_output_path = os.path.join(patient_dir, f"adata_{patient_id}_{dr_method}.h5ad")
    patient_adata_dr.write_h5ad(dr_output_path)
    
    # Run LR-Lasso classification
    classifier = LRLasso(patient_adata_dr, target_col='CN.label', target_value='cancer')
    X, y, feature_names, valid_X_original = classifier.prepare_data(
        use_factorized=True, 
        factorization_method=dr_obsm_key, 
        selected_features=None
    )
    
    # Check if we have enough data
    if len(np.unique(y)) < 2:
        print(f"Warning: Patient {patient_id} does not have both normal and cancer cells. Skipping classification.")
        return {
            'patient_id': patient_id,
            'dr_method': dr_method,
            'adata_dr': patient_adata_dr,
            'classification_skipped': True
        }
    
    # Set default alphas if not provided
    if alphas is None:
        alphas = np.logspace(-4, 5, 20)
    
    # Fit along regularization path
    print(f"Running LR-Lasso classification for patient {patient_id}...")
    results = classifier.fit_along_regularization_path(
        X, y, feature_names, alphas=alphas, metrics_grouping='sample'
    )
    
    # Save coefficient results
    coefs_df = pd.DataFrame(
        results['coefs'],
        index=feature_names,
        columns=alphas
    )
    coefs_filename = os.path.join(patient_dir, f"results_coefs_{dr_method}_{patient_id}.csv")
    coefs_df.to_csv(coefs_filename, index=True)
    
    # Save metrics results
    metrics_df = pd.DataFrame(results['group_metrics_path'])
    metrics_filename = os.path.join(patient_dir, f"results_metrics_{dr_method}_{patient_id}.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    
    # Generate visualizations
    plt.figure(figsize=(12, 8))
    plot_metrics_and_coefficients(
        metrics_df, 
        coefs_df, 
        sample_name="all_samples", 
        top_n=10
    )
    plt.savefig(os.path.join(patient_dir, f"metrics_and_coefs_{dr_method}_{patient_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'patient_id': patient_id,
        'dr_method': dr_method,
        'adata_dr': patient_adata_dr,
        'results': results,
        'coefs_df': coefs_df,
        'metrics_df': metrics_df
    }

def compute_row_weights(adata, beta=1.0):
    """
    Compute per-cell weights for an AnnData object using a sample-level metric.
    
    Parameters:
    - adata: AnnData object with adata.obs containing 'sample' and 'CN.label'
    - beta: Weighting parameter in [0,1] for normal cells
    
    Returns:
    - row_weights: 1D numpy array of shape (n_obs,), with each cell's computed weight
    """
    row_weights = np.zeros(adata.n_obs)
    
    # Group cells by sample
    for sample, group in adata.obs.groupby('sample', observed=False):
        # Create mask for the current sample
        mask = adata.obs['sample'] == sample
        
        # Identify malignant cells (CN.label == 'cancer') in the current sample
        malignant_mask = adata.obs.loc[mask, 'CN.label'] == 'cancer'
        M = malignant_mask
        # pipeline/single_patient.py (continued)
        M = malignant_mask.sum()       # Number of malignant cells
        N = (~malignant_mask).sum()    # Number of normal cells
        
        # Calculate denominator, with beta scaling the normal cell count
        denominator = beta * N + M
        if denominator == 0:
            denominator = 1.0  # Fallback if sample is empty or undefined
        
        # Compute weight for every cell in the sample
        weight = 1.0 / denominator
        row_weights[mask] = weight
        
    return row_weights