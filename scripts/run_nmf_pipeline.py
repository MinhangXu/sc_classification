#!/usr/bin/env python
"""
Simple script to run NMF pipeline using the standardized workflow.

This script demonstrates how to run the NMF + LR-Lasso pipeline
and compare it with your existing FA results.
"""

import os
import sys
from pathlib import Path

from sc_classification.utils.experiment_manager import ExperimentManager, create_standard_config
from sc_classification.pipeline.standardized_pipeline import StandardizedPipeline


def run_nmf_pipeline(input_data_path: str, n_components: int = 100, 
                    downsampling: str = 'random', default_target_fraction: float = 0.7,
                    target_ratio_threshold: float = 10.0,
                    min_cells_per_type: int = 20):
    """
    Run NMF pipeline with specified parameters.
    
    Parameters:
    - input_data_path: Path to input AnnData file
    - n_components: Number of NMF components
    - downsampling: Downsampling method ('random' or 'none')
    - default_target_fraction: Default target donor fraction for downsampling
    - target_ratio_threshold: Target donor-recipient ratio threshold
    - min_cells_per_type: The minimum number of donor cells to keep per cell type.
    """
    
    print(f"Running NMF pipeline with {n_components} components...")
    print(f"Input data: {input_data_path}")
    print(f"Downsampling: {downsampling} (default fraction: {default_target_fraction})")
    print(f"Target ratio threshold: {target_ratio_threshold}")
    
    # Create experiment manager
    experiment_manager = ExperimentManager("experiments")
    
    # Create NMF configuration
    config = create_standard_config(
        dr_method='nmf',
        n_components=n_components,
        downsampling_method=downsampling,
        target_donor_fraction=default_target_fraction
    )
    
    # Explicitly disable standardization for NMF runs
    config.config['preprocessing']['standardize'] = False
    
    # Update downsampling configuration with additional parameters
    config.config['downsampling']['target_donor_recipient_ratio_threshold'] = target_ratio_threshold
    config.config['downsampling']['min_cells_per_type'] = min_cells_per_type
    
    # Update NMF-specific configuration
    config.config['dimension_reduction']['handle_negative_values'] = 'error'
    
    # Create and run pipeline
    pipeline = StandardizedPipeline(experiment_manager)
    experiment = pipeline.run_pipeline(config, input_data_path)
    
    print(f"\nNMF pipeline completed successfully!")
    print(f"Experiment ID: {experiment.config.experiment_id}")
    print(f"Results saved in: {experiment.experiment_dir}")
    
    return experiment


def compare_fa_nmf(fa_experiment_id: str, nmf_experiment_id: str):
    """
    Compare FA and NMF experiment results.
    
    Parameters:
    - fa_experiment_id: ID of the FA experiment
    - nmf_experiment_id: ID of the NMF experiment
    """
    
    from sc_classification.utils.experiment_analysis import ExperimentAnalyzer
    
    print(f"\nComparing FA vs NMF experiments...")
    print(f"FA Experiment: {fa_experiment_id}")
    print(f"NMF Experiment: {nmf_experiment_id}")
    
    # Create experiment manager and analyzer
    experiment_manager = ExperimentManager("experiments")
    analyzer = ExperimentAnalyzer(experiment_manager)
    
    # Compare experiments
    comparison_df = analyzer.compare_experiments([fa_experiment_id, nmf_experiment_id])
    print("\nExperiment Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Compare performance
    performance_df = analyzer.compare_classification_performance([fa_experiment_id, nmf_experiment_id])
    if not performance_df.empty:
        print("\nPerformance Comparison:")
        print(performance_df.groupby('experiment_id')[['auc', 'precision', 'recall', 'f1_score']].mean())
    
    # Create comparison plots
    plots_dir = "fa_nmf_comparison_plots"
    analyzer.create_performance_comparison_plots([fa_experiment_id, nmf_experiment_id], plots_dir)
    print(f"\nComparison plots saved to: {plots_dir}")
    
    # Generate HTML report
    report_path = "fa_nmf_comparison_report.html"
    analyzer.generate_experiment_summary_report([fa_experiment_id, nmf_experiment_id], report_path)
    print(f"Comparison report saved to: {report_path}")


def main():
    """Main function to run NMF pipeline and comparison."""
    
    # Configuration
    input_data_path = "/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad"
    n_components = 100
    downsampling = 'random'
    # --- CHANGED: Default fraction updated to 0.7 ---
    default_target_fraction = 0.7
    target_ratio_threshold = 10.0
    min_cells_per_type = 20
    
    # Check if input file exists
    if not os.path.exists(input_data_path):
        print(f"Error: Input file not found: {input_data_path}")
        print("Please update the input_data_path variable in this script.")
        return
    
    # Run NMF pipeline
    nmf_experiment = run_nmf_pipeline(
        input_data_path=input_data_path,
        n_components=n_components,
        downsampling=downsampling,
        default_target_fraction=default_target_fraction,
        target_ratio_threshold=target_ratio_threshold,
        min_cells_per_type=min_cells_per_type
    )
    
    # Ask user for FA experiment ID for comparison
    print("\n" + "="*60)
    print("NMF pipeline completed!")

if __name__ == "__main__":
    main() 