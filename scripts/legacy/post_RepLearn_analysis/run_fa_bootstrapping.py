#!/usr/bin/env python
"""
Factor Analysis Bootstrapping for Significance Testing

This script performs bootstrapping on a completed Factor Analysis experiment
to calculate standard errors and confidence intervals for the gene loadings.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from sklearn.decomposition import FactorAnalysis as SklearnFA
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_manager import ExperimentManager

def run_bootstrapping(experiment_id: str, n_iterations: int = 100):
    """
    Loads a completed experiment, runs bootstrapping on the FA model,
    and saves the significance results.
    """
    print(f"--- Starting bootstrapping for experiment: {experiment_id} ---")
    print(f"--- Running {n_iterations} iterations ---")

    # Load experiment
    manager = ExperimentManager("experiments")
    try:
        experiment = manager.load_experiment(experiment_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Check if this was an 'fa' experiment
    dr_method = experiment.config.get('dimension_reduction.method')
    if dr_method != 'fa':
        print(f"Error: This script is for 'fa' experiments, but this experiment used '{dr_method}'.")
        return

    # 1. Load the preprocessed data used for the FA
    preprocessed_data_path = experiment.get_path('preprocessed_data')
    if not preprocessed_data_path.exists():
        print(f"Error: Preprocessed data not found at {preprocessed_data_path}")
        return
    
    print(f"Loading preprocessed data from {preprocessed_data_path}...")
    adata = sc.read_h5ad(preprocessed_data_path)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    # 2. Get the original FA model configuration
    dr_config = experiment.config.get('dimension_reduction')
    n_components = dr_config.get('n_components')
    random_state_base = dr_config.get('random_state', 42)

    # Load the original FA model to use as a reference for alignment
    fa_model_path = experiment.get_path('dr_model', dr_method=dr_method, n_components=n_components)
    model_dir = fa_model_path.parent
    try:
        model_file = next(model_dir.glob('fa_model_*.pkl'))
        import pickle
        with open(model_file, 'rb') as f:
            original_fa = pickle.load(f)
        original_loadings = original_fa.components_.T
    except (StopIteration, FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Warning: Could not load the original FA model from {model_dir}. Error: {e}")
        print("Alignment will not be performed. Using unaligned bootstrap results.")
        original_loadings = None

    # 3. Run bootstrapping loop serially
    n_samples = X.shape[0]
    all_loadings_dfs = []
    
    print("Starting bootstrapping iterations serially...")

    for i in tqdm(range(n_iterations), desc="Bootstrapping Progress"):
        # Resample cells with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_resampled = X[indices, :]
        
        # Fit FA model on the resampled data
        fa_boot = SklearnFA(n_components=n_components, random_state=random_state_base + i, svd_method='lapack')
        bootstrap_loadings = fa_boot.fit(X_resampled).components_.T
        
        # Align factors to the original loadings if a reference is provided
        aligned_loadings = np.zeros_like(bootstrap_loadings)
        if original_loadings is not None:
            # Calculate correlation matrix between original and bootstrap factors
            correlation_matrix = np.corrcoef(original_loadings.T, bootstrap_loadings.T)
            corr_block = correlation_matrix[:n_components, n_components:]
            
            # Find optimal alignment using the Hungarian algorithm
            cost_matrix = 1 - np.abs(corr_block)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Reorder and flip signs
            for orig_factor_idx, boot_factor_idx in zip(row_ind, col_ind):
                sign = np.sign(corr_block[orig_factor_idx, boot_factor_idx])
                aligned_loadings[:, orig_factor_idx] = bootstrap_loadings[:, boot_factor_idx] * sign
        else:
            aligned_loadings = bootstrap_loadings

        # Append aligned loadings as a DataFrame
        factor_names = [f'Factor{i+1}' for i in range(n_components)]
        all_loadings_dfs.append(pd.DataFrame(aligned_loadings, index=adata.var_names, columns=factor_names))

    # 4. Calculate statistics using pandas for robust index handling
    print("Calculating statistics from bootstrap results...")
    # Concatenate all bootstrap results into a single multi-indexed DataFrame
    # The panel has a (gene, factor) MultiIndex on the columns
    panel = pd.concat(all_loadings_dfs, axis=1, keys=range(n_iterations))
    
    # To calculate stats per gene/factor, we can group by column level 1 (the factor names)
    # and then calculate stats over the iterations (axis=1)
    mean_loadings = panel.groupby(level=1, axis=1).mean()
    se_loadings = panel.groupby(level=1, axis=1).std()
    lower_ci = panel.groupby(level=1, axis=1).quantile(0.025)
    upper_ci = panel.groupby(level=1, axis=1).quantile(0.975)

    # 5. Use the original loadings DataFrame if available, otherwise use the mean
    if original_loadings is not None:
        factor_names = [f'Factor{i+1}' for i in range(n_components)]
        original_loadings_df = pd.DataFrame(original_loadings, index=adata.var_names, columns=factor_names)
    else:
        # This case is triggered if the original model file could not be loaded
        original_loadings_df = mean_loadings
        
    # 6. Assemble and save the results
    # Reshape the DataFrames from wide to long format for the final CSV
    results_df = original_loadings_df.stack().reset_index()
    results_df.columns = ['gene', 'factor', 'original_loading']
    
    results_df['se'] = se_loadings.stack().values
    results_df['ci_lower_2.5'] = lower_ci.stack().values
    results_df['ci_upper_97.5'] = upper_ci.stack().values
    
    save_dir = experiment.get_path('factor_interpretation')
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "bootstrapped_loadings_stats.csv"
    
    results_df.to_csv(save_path, index=False)
    print(f"\n--- Bootstrapping complete. ---")
    print(f"Significance results saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run bootstrapping on a Factor Analysis experiment.")
    parser.add_argument("experiment_id", type=str, help="The ID of the experiment to process.")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of bootstrap iterations to run.")
    
    args = parser.parse_args()
    
    run_bootstrapping(args.experiment_id, args.n_iter)
