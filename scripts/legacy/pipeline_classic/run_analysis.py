#!/usr/bin/env python
"""
Command-line interface for running downstream analysis on standardized pipeline experiments.

This script provides easy access to:
1. LR-Lasso result analysis and plotting
2. Factor interpretation analysis
3. Projection validation analysis
"""

import argparse
import sys
from pathlib import Path

from sc_classification.utils.experiment_manager import ExperimentManager
from sc_classification.utils.experiment_analysis import ExperimentAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Run downstream analysis on standardized pipeline experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LR-Lasso analysis (saves to experiment's analysis/summary_plots)
  python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type lr_lasso

  # Run factor interpretation analysis (saves to experiment's analysis/factor_interpretation)
  python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type factor_interpretation

  # Run projection validation analysis (saves to experiment's analysis/projections)
  python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type projection \
    --validation-data /path/to/full_data.h5ad

  # Run all analyses (saves to experiment's analysis directories)
  python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type all \
    --validation-data /path/to/full_data.h5ad

  # Use custom output directory
  python run_analysis.py \
    --experiment-id 20250620_154919_nmf_100_random_d9c1dec9 \
    --analysis-type lr_lasso \
    --output-dir custom_analysis_results
        """
    )
    
    parser.add_argument(
        "--experiment-id", 
        required=True,
        help="ID of the experiment to analyze"
    )
    
    parser.add_argument(
        "--analysis-type",
        choices=["lr_lasso", "factor_interpretation", "projection", "all"],
        required=True,
        help="Type of analysis to run"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Custom output directory (defaults to experiment's analysis directories)"
    )
    
    parser.add_argument(
        "--validation-data",
        help="Path to full AnnData with all timepoints (required for projection analysis)"
    )
    
    parser.add_argument(
        "--factor-interpretation-method",
        choices=["overall", "loading", "both"],
        default="both",
        help="Method for factor interpretation analysis (default: both)"
    )
    
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Directory containing experiments (default: experiments)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.analysis_type == "projection" and not args.validation_data:
        parser.error("--validation-data is required for projection analysis")
    
    # Initialize experiment manager and analyzer
    experiment_manager = ExperimentManager(args.experiments_dir)
    analyzer = ExperimentAnalyzer(experiment_manager)
    
    # Check if experiment exists
    try:
        experiment = experiment_manager.load_experiment(args.experiment_id)
        print(f"Loaded experiment: {args.experiment_id}")
        print(f"  DR Method: {experiment.config.get('dimension_reduction.method')}")
        print(f"  N Components: {experiment.config.get('dimension_reduction.n_components')}")
        print(f"  Downsampling: {experiment.config.get('downsampling.method')}")
    except Exception as e:
        print(f"Error loading experiment {args.experiment_id}: {e}")
        sys.exit(1)
    
    # Run requested analysis
    if args.analysis_type in ["lr_lasso", "all"]:
        print("\n=== Running LR-Lasso Analysis ===")
        try:
            # Use custom output dir if specified, otherwise use experiment's summary_plots
            output_dir = args.output_dir if args.output_dir else None
            analyzer.generate_lr_lasso_analysis_plots(
                args.experiment_id, 
                output_dir
            )
            if output_dir:
                print(f"LR-Lasso analysis results saved to: {output_dir}")
            else:
                print(f"LR-Lasso analysis results saved to experiment's summary_plots directory")
        except Exception as e:
            print(f"Error in LR-Lasso analysis: {e}")
    
    if args.analysis_type in ["factor_interpretation", "all"]:
        print("\n=== Running Factor Interpretation Analysis ===")
        try:
            # Use custom output dir if specified, otherwise use experiment's factor_interpretation
            output_dir = args.output_dir if args.output_dir else None
            analyzer.generate_factor_interpretation_analysis(
                args.experiment_id,
                output_dir,
                args.factor_interpretation_method
            )
            if output_dir:
                print(f"Factor interpretation results saved to: {output_dir}")
            else:
                print(f"Factor interpretation results saved to experiment's factor_interpretation directory")
        except Exception as e:
            print(f"Error in factor interpretation analysis: {e}")
    
    if args.analysis_type in ["projection", "all"]:
        print("\n=== Running Projection Validation Analysis ===")
        try:
            # Use custom output dir if specified, otherwise use experiment's projections
            output_dir = args.output_dir if args.output_dir else None
            analyzer.generate_projection_validation_analysis(
                args.experiment_id,
                args.validation_data,
                output_dir
            )
            if output_dir:
                print(f"Projection validation results saved to: {output_dir}")
            else:
                print(f"Projection validation results saved to experiment's projections directory")
        except Exception as e:
            print(f"Error in projection validation analysis: {e}")
    
    print(f"\nAnalysis complete!")
    if args.output_dir:
        print(f"Results saved to: {args.output_dir}")
    else:
        print(f"Results saved to experiment's analysis directories")


if __name__ == "__main__":
    main() 