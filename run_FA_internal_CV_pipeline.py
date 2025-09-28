#!/usr/bin/env python
"""
Runner Script for Standardized Pipeline with Internal Cross-Validation

This script runs the Factor Analysis pipeline with repeated stratified
cross-validation enabled for the per-patient classification step.
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.standardized_pipeline import StandardizedPipeline
from utils.experiment_manager import ExperimentManager, create_standard_config

def main():
    """Configure and run the FA pipeline with internal CV."""
    parser = argparse.ArgumentParser(description="Run FA pipeline with internal cross-validation.")
    parser.add_argument("--n-components", type=int, default=100, help="Number of components for Factor Analysis.")
    parser.add_argument("--downsampling-method", type=str, default="random", choices=["random", "none"], help="Downsampling method for donor cells.")
    parser.add_argument("--target-donor-fraction", type=float, default=0.7, help="Target fraction of donor cells after random downsampling.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds for cross-validation. Set to 0 or 1 to disable CV.")
    parser.add_argument("--cv-repeats", type=int, default=10, help="Number of repeats for cross-validation.")
    args = parser.parse_args()
    
    # --- 1. Setup Experiment ---
    experiment_manager = ExperimentManager("experiments")
    
    # --- 2. Create Configuration ---
    config = create_standard_config(
        dr_method='fa',
        n_components=args.n_components,
        downsampling_method=args.downsampling_method,
        target_donor_fraction=args.target_donor_fraction
    )
    
    # --- Key CV Parameters ---
    # fold-capping logic in classification_methods/lr_lasso.py
    config.config['classification']['cv_folds'] = args.cv_folds
    config.config['classification']['cv_repeats'] = args.cv_repeats
    
    # --- Other Parameters ---
    config.config['preprocessing']['standardize'] = True
    config.config['downsampling']['target_donor_recipient_ratio_threshold'] = 10.0
    config.config['downsampling']['min_cells_per_type'] = 20
    
    # --- 3. Run Pipeline ---
    pipeline = StandardizedPipeline(experiment_manager)
    
    # Define the input data path
    input_path = "/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad"
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
        
    print("--- Starting FA Pipeline with Internal Cross-Validation ---")
    print(f"Experiment configuration:")
    print(f"  - DR Method: {config.get('dimension_reduction.method')}")
    print(f"  - Components: {config.get('dimension_reduction.n_components')}")
    print(f"  - Downsampling: {config.get('downsampling.method')}")
    print(f"  - CV Folds: {config.get('classification.cv_folds')}")
    print(f"  - CV Repeats: {config.get('classification.cv_repeats')}")
    print("-" * 50)
    
    try:
        experiment = pipeline.run_pipeline(config, input_path)
        print(f"\n--- Pipeline Completed ---")
        print(f"Successfully completed experiment: {experiment.config.experiment_id}")
        print(f"Results saved in: {experiment.experiment_dir}")
        
    except Exception as e:
        print(f"\n--- Pipeline Failed ---")
        print(f"An error occurred during the pipeline run: {e}")
        # The pipeline automatically logs the error in the experiment's metadata.json

if __name__ == '__main__':
    main()