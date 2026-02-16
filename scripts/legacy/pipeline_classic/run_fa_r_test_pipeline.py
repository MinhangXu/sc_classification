#!/usr/bin/env python
"""
Test Pipeline Runner for R-based Factor Analysis

This script runs a single experiment to validate the FactorAnalysis_R implementation.
It uses the same preprocessing configuration as the 'supervised_all_filtered'
experiments, focusing on CITE-seq data from the MRD timepoint.
"""

import os
import sys
import copy
from pathlib import Path

from sc_classification.utils.experiment_manager import ExperimentManager, ExperimentConfig
from sc_classification.pipeline.standardized_pipeline import StandardizedPipeline

def run_fa_r_validation_experiment(input_data_path: str, n_components: int = 100):
    """
    Sets up and runs a validation experiment for the R-based Factor Analysis.
    """
    print("--- Starting R-based Factor Analysis (fa_r) Validation Experiment ---")
    
    if not os.path.exists(input_data_path):
        print(f"Error: Input data file not found at '{input_data_path}'")
        print("Please update the 'input_data_path' variable in the main block of this script.")
        return

    experiment_manager = ExperimentManager("experiments")
    
    # --- Base Configuration ---
    # This configuration is based on the aug19_rep_learn_supervised_filtering.py script
    base_config_dict = {
        'preprocessing': {
            'standardize': True,
            'timepoint_filter': 'MRD',
            'target_column': 'CN.label',
            'positive_class': 'cancer',
            'negative_class': 'normal',
            'tech_filter': 'CITE',
            'gene_selection_pipeline': [
                {
                    'method': 'all_filtered',
                    'min_cells_fraction': 0.01,
                    'malignant_enrichment_ratio': 20
                }
            ]
        },
        'dimension_reduction': {
            'method': 'fa_r',  # Use the new R-based Factor Analysis
            'n_components': n_components,
            'random_state': 42,
            # 'fm': 'ml', # 'ml' (Maximum Likelihood) can be extremely slow on datasets with many genes.
            'fm': 'minres',  # 'minres' (Minimum Residual) is a much faster alternative and often gives similar results.
            'rotate': 'varimax'
        },
        'classification': {
            'method': 'lr_lasso',
            'alphas': 'logspace(-4, 5, 20)',
            'cv_folds': 5,
            'random_state': 42
        },
        'downsampling': {
            'method': 'random',
            'target_donor_fraction': 0.7,
            'target_donor_recipient_ratio_threshold': 10.0,
            'min_cells_per_type': 20,
            'cell_type_column': 'predicted.annotation',
            'donor_recipient_column': 'source'
        }
    }

    # --- Create and Run Experiment ---
    exp_config = ExperimentConfig(copy.deepcopy(base_config_dict))
    pipeline = StandardizedPipeline(experiment_manager)
    
    print("\n--- Running FA-R Validation Pipeline ---")
    print(f"Configuration ID: {exp_config.experiment_id}")
    
    experiment = pipeline.run_pipeline(exp_config, input_data_path)
    
    print(f"\n--- FA-R Validation Experiment Complete. ---")
    print(f"Results saved in: {experiment.experiment_dir}")
    print("Please check the logs and output files to ensure everything ran as expected.")

if __name__ == '__main__':
    # --- Configuration ---
    # Please verify this path points to your input data file.
    input_data_path = "/home/minhang/mds_project/data/cohort_adata/adata_mrd_preprocessed_aug14.h5ad"
    NUM_FACTORS = 100
    
    run_fa_r_validation_experiment(input_data_path, n_components=NUM_FACTORS)
