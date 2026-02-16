#!/usr/bin/env python
"""
Driver script for CITE-seq only MRD classification using advanced supervised filtering
WITH INTERNAL CROSS-VALIDATION.

This script builds on previous experiments by incorporating more sophisticated,
supervised filtering methods during preprocessing to retain potentially
important biological signals from rare malignant cell populations, and adds
internal cross-validation to the classification stage for more robust
performance metrics.

The script will execute three main experiments on the CITE-seq subset:
1. Pan-Patient (5-fold x 10 repeats CV): Uses a supervised 'all_filtered' method.
2. Per-Patient (5-fold x 10 repeats CV): Uses the same supervised 'all_filtered' method as Exp 1.
3. Per-Patient (5-fold x 10 repeats CV): Uses a relaxed weak DEG screen.
"""

import os
import sys
import copy
from pathlib import Path

from sc_classification.utils.experiment_manager import ExperimentManager, ExperimentConfig
from sc_classification.pipeline.standardized_pipeline import StandardizedPipeline

def run_supervised_filtering_experiments_with_cv(input_data_path: str, n_components: int = 100):
    """
    Sets up and runs a series of classification experiments with supervised filtering and CV.
    """
    print("--- Starting Supervised Filtering Experiments with Internal CV for CITE-seq MRD ---")

    experiment_manager = ExperimentManager("experiments")
    
    # --- Base Config ---
    base_config_dict = {
        'preprocessing': {
            'standardize': True, 'timepoint_filter': 'MRD', 'target_column': 'CN.label',
            'positive_class': 'cancer', 'negative_class': 'normal',
            'tech_filter': 'CITE'
        },
        'dimension_reduction': {
            'method': 'fa', 'n_components': n_components, 'random_state': 42, 'svd_method': 'randomized'
        },
        'classification': {
            'method': 'lr_lasso', 'alphas': 'logspace(-4, 5, 20)', 
            'cv_folds': 5,      # Enable 5-fold CV
            'cv_repeats': 10,   # With 10 repeats
            'random_state': 42
        },
        'downsampling': {
            'method': 'random', 'target_donor_fraction': 0.7,
            'target_donor_recipient_ratio_threshold': 10.0,
            'min_cells_per_type': 20, 'cell_type_column': 'predicted.annotation',
            'donor_recipient_column': 'source'
        }
    }

    # --- Experiment 1 & 2: Supervised 'all_filtered' Method ---
    print("\n--- Configuring Exp 1 & 2: Supervised 'all_filtered' Method with CV ---")
    config_supervised_all_filtered = copy.deepcopy(base_config_dict)
    config_supervised_all_filtered['preprocessing']['gene_selection_pipeline'] = [
        {
            'method': 'all_filtered',
            'min_cells_fraction': 0.01,  # Filter genes in < 1% of cells
            'malignant_enrichment_ratio': 50  # But rescue if enrichment > 50
        }
    ]
    
    # --- Experiment 1: Pan-Patient, Supervised All Filtered ---
    config1 = copy.deepcopy(config_supervised_all_filtered)
    config1['downsampling']['method'] = 'none'
    config1['classification']['patient_column'] = 'pan_patient_group'
    config1['classification']['metrics_grouping'] = 'patient'
    
    exp_config_1 = ExperimentConfig(config1)
    pipeline1 = StandardizedPipeline(experiment_manager)

    def run_with_pan_patient_mods(pipe_instance, config, data_path):
        adata = sc.read_h5ad(data_path)
        adata.obs['pan_patient_group'] = 'all_patients'
        temp_path = f"/tmp/pan_patient_temp_adata_{os.getpid()}.h5ad"
        adata.write_h5ad(temp_path)
        experiment = pipe_instance.run_pipeline(config, temp_path)
        os.remove(temp_path)
        return experiment

    print("--- Running Exp 1: Pan-Patient, Supervised All Filtered with CV ---")
    exp1 = run_with_pan_patient_mods(pipeline1, exp_config_1, input_data_path)
    print(f"--- Exp 1 Complete. Results saved in: {exp1.experiment_dir} ---")

    # --- Experiment 2: Per-Patient, Supervised All Filtered ---
    exp_config_2 = ExperimentConfig(copy.deepcopy(config_supervised_all_filtered))
    pipeline2 = StandardizedPipeline(experiment_manager)
    print("\n--- Running Exp 2: Per-Patient, Supervised All Filtered with CV ---")
    exp2 = pipeline2.run_pipeline(exp_config_2, input_data_path)
    print(f"--- Exp 2 Complete. Results saved in: {exp2.experiment_dir} ---")

    # --- Experiment 3: Per-Patient, Relaxed Weak DEGs ---
    print("\n--- Configuring Exp 3: Relaxed Weak DEG Screen with CV ---")
    config3 = copy.deepcopy(base_config_dict)
    config3['preprocessing']['gene_selection_pipeline'] = [
        {
            'method': 'deg_weak_screen', 'deg_test_method': 'wilcoxon',
            'use_adj_pvals': True,
            'pval_threshold': 0.1,       # Relaxed p-value
            'lfc_threshold': 0.05,       # Relaxed LFC
            'min_n_genes': None,         # No topping up
        }
    ]
    exp_config_3 = ExperimentConfig(config3)
    pipeline3 = StandardizedPipeline(experiment_manager)
    print("--- Running Exp 3: Per-Patient, Relaxed Weak DEGs with CV ---")
    exp3 = pipeline3.run_pipeline(exp_config_3, input_data_path)
    print(f"--- Exp 3 Complete. Results saved in: {exp3.experiment_dir} ---")
    
    print("\n--- All Supervised Filtering Experiments with CV Complete ---")

def main():
    input_data_path = "/home/minhang/mds_project/data/cohort_adata/adata_mrd_preprocessed_aug14.h5ad"
    if not os.path.exists(input_data_path):
        print(f"Error: Input file not found: {input_data_path}")
        return

    run_supervised_filtering_experiments_with_cv(input_data_path=input_data_path)

if __name__ == "__main__":
    import scanpy as sc
    main()
