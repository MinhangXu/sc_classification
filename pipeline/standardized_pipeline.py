#!/usr/bin/env python
"""
Standardized Pipeline for Single-Cell Classification

This script provides a unified pipeline that can run both Factor Analysis and NMF
with consistent preprocessing, classification, and result tracking.
"""

import os
import sys
import time
import numpy as np
import json
import pandas as pd
import scanpy as sc
from pathlib import Path
import logging
import warnings

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.experiment_manager import ExperimentManager, create_standard_config
from utils.preprocessing import PreprocessingPipeline, create_preprocessing_config, random_downsample_stratified
from dimension_reduction.factor_analysis import FactorAnalysis
from dimension_reduction.nmf import NMF
from classification_methods.lr_lasso import LRLasso

warnings.filterwarnings("ignore", category=FutureWarning, module="anndata.utils")

class StandardizedPipeline:
    """Standardized pipeline for single-cell classification."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger("standardized_pipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_pipeline(self, config, input_data_path: str):
        """
        Run the complete standardized pipeline.
        
        Parameters:
        - config: ExperimentConfig object
        - input_data_path: Path to input AnnData file
        """
        self.logger.info(f"Starting standardized pipeline for experiment: {config.experiment_id}")
        
        # Create experiment instance
        experiment = self.experiment_manager.create_experiment(config)
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1: Loading and preprocessing data...")
            preprocessing_results = self._run_preprocessing(experiment, input_data_path)
            
            # Step 2: Dimension reduction
            self.logger.info("Step 2: Running dimension reduction...")
            dr_results = self._run_dimension_reduction(experiment, preprocessing_results)
            
            # Step 3: Classification
            self.logger.info("Step 3: Running classification...")
            classification_results = self._run_classification(experiment, dr_results)
            
            # Step 4: Update final status
            experiment.update_metadata({
                'status': 'completed',
                'completion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_runtime': time.time() - experiment.load_metadata().get('start_time', time.time())
            })
            
            self.logger.info(f"Pipeline completed successfully for experiment: {config.experiment_id}")
            return experiment
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            experiment.update_metadata({
                'status': 'failed',
                'error': str(e),
                'failure_time': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            raise
    
    def _run_preprocessing(self, experiment, input_data_path: str):
        """Run preprocessing step."""
        # Load raw data
        self.logger.info(f"Loading data from: {input_data_path}")
        adata_raw = sc.read_h5ad(input_data_path)
        
        # Create preprocessing configuration
        preproc_config = create_preprocessing_config(
            n_top_genes=experiment.config.get('preprocessing.n_top_genes', 3000),
            standardize=experiment.config.get('preprocessing.standardize', True),
            timepoint_filter=experiment.config.get('preprocessing.timepoint_filter', 'MRD'),
            target_column=experiment.config.get('preprocessing.target_column', 'CN.label'),
            positive_class=experiment.config.get('preprocessing.positive_class', 'cancer'),
            negative_class=experiment.config.get('preprocessing.negative_class', 'normal')
        )
        
        # Run preprocessing
        preproc_pipeline = PreprocessingPipeline(preproc_config)
        results = preproc_pipeline.run_preprocessing(adata_raw)
        
        # Save preprocessing results
        experiment.save_preprocessing_results(
            results['adata'],
            results['hvg_list'],
            results['scaler'],
            results['summary']
        )
        
        return results
    
    def _run_dimension_reduction(self, experiment, preprocessing_results):
        """Run dimension reduction step."""
        adata = preprocessing_results['adata']
        dr_config = experiment.config.get('dimension_reduction', {})
        
        # Get DR method from config, raise error if not specified
        dr_method = dr_config.get('method')
        if not dr_method:
            raise ValueError("Dimension reduction 'method' must be specified in the config (e.g., 'fa' or 'nmf').")

        n_components = dr_config.get('n_components', 100)
        random_state = dr_config.get('random_state', 42)
        
        self.logger.info(f"Running {dr_method.upper()} with {n_components} components...")
        
        # Initialize DR method and save directory
        if dr_method.lower() == 'fa':
            dr_model = FactorAnalysis()
            save_dir = experiment.experiment_dir / "models" / f"fa_{n_components}"
        elif dr_method.lower() == 'nmf':
            dr_model = NMF()
            save_dir = experiment.experiment_dir / "models" / f"nmf_{n_components}"
        else:
            raise ValueError(f"Unsupported dimension reduction method: {dr_method}")
        
        # Run dimension reduction
        if dr_method.lower() == 'fa':
            dr_model_instance = FactorAnalysis()
            save_dir = experiment.experiment_dir / "models" / f"fa_{n_components}"
            fa_params = {
                'n_components': n_components,
                'random_state': random_state,
                'standardize_input': False, # Already done in preprocessing
                'svd_method': dr_config.get('svd_method', 'lapack'),
                'save_fitted_models': True,
                'model_save_dir': str(save_dir)
            }
            adata_transformed = dr_model_instance.fit_transform(adata, **fa_params)
            model = adata_transformed.uns.get('_temp_fa_model_obj')
            
            # --- FIX: Clean the AnnData object before saving ---
            # The h5ad format cannot store complex Python objects like sklearn models.
            # We must remove them from .uns before writing the file to disk.
            # The model object itself is saved separately via pickle in save_dr_results.
            if '_temp_fa_model_obj' in adata_transformed.uns:
                del adata_transformed.uns['_temp_fa_model_obj']
            if '_temp_scaler_obj' in adata_transformed.uns:
                del adata_transformed.uns['_temp_scaler_obj']
            # --- END FIX ---
            
            fa_info = adata_transformed.uns.get('fa', {})
            summary = "Factor Analysis Summary:\n" + json.dumps(fa_info, indent=2, default=str)
            
        elif dr_method.lower() == 'nmf':
            input_standardized = experiment.config.get('preprocessing.standardize', True)
            handle_negative_values = dr_config.get('handle_negative_values', 'error')

            nmf_params = {
                'n_components': n_components,
                'random_state': random_state,
                'save_model': True,
                'save_dir': str(save_dir),
                'use_hvg': True,
                'standardize_input': input_standardized,
                'handle_negative_values': handle_negative_values
            }
            adata_transformed = dr_model.fit_transform(adata, **nmf_params)
            
            model_path = save_dir / "nmf_model.pkl"
            with open(model_path, 'rb') as f:
                import pickle
                model_info = pickle.load(f)
                model = model_info['model']
        
        # Generate summary
        if dr_method.lower() == 'fa':
            # Create summary for FA
            fa_info = adata_transformed.uns.get('fa', {})
            summary_lines = [
                f"Factor Analysis Summary",
                "=" * 50,
                f"Number of components: {fa_info.get('n_factors', 'N/A')}",
                f"Random state: {fa_info.get('random_state', 'N/A')}",
                f"Standardized input: {fa_info.get('standardized_input_in_class_call', 'N/A')}",
                f"SVD method: {fa_info.get('svd_method_used', 'N/A')}",
                "",
                "Factor Score Statistics:",
                f"  Sum of factor score variances: {fa_info.get('sum_factor_score_variances', 'N/A'):.4f}",
                f"  Mean factor score variance: {np.mean(fa_info.get('factor_score_variances', [])):.4f}",
                f"  Min factor score variance: {np.min(fa_info.get('factor_score_variances', [])):.4f}",
                f"  Max factor score variance: {np.max(fa_info.get('factor_score_variances', [])):.4f}",
                "",
                "Communality Statistics:",
            ]
            
            comm_summary = fa_info.get('communalities_per_gene_summary', {})
            for stat, value in comm_summary.items():
                summary_lines.append(f"  {stat.capitalize()}: {value:.4f}")
            
            summary_lines.extend([
                "",
                "SS Loadings per Factor:",
                f"  Sum of SS loadings: {np.sum(fa_info.get('ss_loadings_per_factor', [])):.4f}",
                f"  Mean SS loading: {np.mean(fa_info.get('ss_loadings_per_factor', [])):.4f}",
            ])
            
            summary = "\n".join(summary_lines)
        elif dr_method.lower() == 'nmf':
            summary = dr_model.generate_model_summary(adata_transformed)
        else:
            summary = "Summary generation not implemented for this method."
        
        # Save DR results
        experiment.save_dr_results(
            model,
            adata_transformed,
            dr_method,
            n_components,
            summary
        )
        
        return {
            'adata': adata_transformed,
            'model': model,
            'method': dr_method,
            'n_components': n_components
        }
    
    def _run_classification(self, experiment, dr_results):
        """Run classification step."""
        adata = dr_results['adata']
        dr_method = dr_results['method']
        n_components = dr_results['n_components']
        
        # Get patient information
        patient_col = experiment.config.get('classification.patient_column', 'patient')
        target_col = experiment.config.get('preprocessing.target_column', 'CN.label')
        donor_recipient_col = experiment.config.get('downsampling.donor_recipient_column', 'source')
        
        patients = sorted(adata.obs[patient_col].unique())
        self.logger.info(f"Running classification for {len(patients)} patients...")
        
        # Get classification parameters
        alphas = np.logspace(-4, 5, 20)  # Default alpha range
        cv_folds = experiment.config.get('classification.cv_folds', 5)
        random_state = experiment.config.get('classification.random_state', 42)
        
        # Get downsampling parameters
        downsampling_config = experiment.config.get('downsampling', {})
        downsampling_method = downsampling_config.get('method', 'none')
        default_target_fraction = downsampling_config.get('target_donor_fraction', 0.5)
        target_ratio_threshold = downsampling_config.get('target_donor_recipient_ratio_threshold', 10.0)
        min_cells_per_type = downsampling_config.get('min_cells_per_type', 20)
        cell_type_col = downsampling_config.get('cell_type_column', 'predicted.annotation')
        
        all_patient_metrics = []
        all_patient_coefs = []

        for patient in patients:
            self.logger.info(f"Processing patient: {patient}")
            
            adata_patient = adata[adata.obs[patient_col] == patient].copy()
            
            # --- MODIFIED: Capture the returned downsampling_info dictionary ---
            downsampling_info = {'downsampling_method': 'none'}
            if downsampling_method != 'none':
                adata_patient, downsampling_info = self._apply_dynamic_downsampling(
                    adata_patient, downsampling_config, donor_recipient_col,
                    cell_type_col=target_col, 
                    default_target_fraction=default_target_fraction,
                    target_ratio_threshold=target_ratio_threshold
                )
            
            # --- Classification ---
            classifier = LRLasso(
                adata=adata_patient,
                target_col=target_col,
                target_value=experiment.config.get('preprocessing.positive_class', 'cancer')
            )
            
            # Prepare data and check for sufficient classes
            X, y, feature_names, _ = classifier.prepare_data(
                use_factorized=True,
                factorization_method=f'X_{dr_method}'
            )

            if len(np.unique(y)) < 2:
                self.logger.warning(f"Patient {patient} has only one class label. Skipping classification.")
                continue

            results = classifier.fit_along_regularization_path(
                X, y, feature_names, alphas=alphas, metrics_grouping='patient'
            )
            
            patient_metrics_df = pd.DataFrame(results['group_metrics_path'])
            patient_coefs_df = pd.DataFrame(
                results['coefs'],
                index=feature_names,
                columns=[f'alpha_{a:.2e}' for a in alphas]
            )
            
            # Save results and update metadata for the current patient ---
            experiment.save_classification_results(
                patient_id=patient,
                coefficients=patient_coefs_df,
                metrics=patient_metrics_df,
                downsampling_info=downsampling_info
            )
            
            all_patient_metrics.append(patient_metrics_df)
            all_patient_coefs.append(patient_coefs_df)
            
            self.logger.info(f"Finished processing and saved results for patient: {patient}")

        # --- Final summary (optional) ---
        if all_patient_metrics:
            # You could save an aggregated summary here if needed
            self.logger.info("Classification complete for all patients.")
        else:
            self.logger.warning("No classification results were generated.")
            
        return {
            'metrics_all_patients': pd.concat(all_patient_metrics, ignore_index=True) if all_patient_metrics else pd.DataFrame(),
            'coefficients_all_patients': pd.concat(all_patient_coefs, ignore_index=True) if all_patient_coefs else pd.DataFrame()
        }
    
    def _apply_dynamic_downsampling(self, adata_patient, downsampling_config, 
                                   donor_recipient_col, cell_type_col, 
                                   default_target_fraction, target_ratio_threshold):
        """
        Apply dynamic downsampling and return the processed AnnData and a detailed info dictionary.
        """
        downsampling_method = downsampling_config.get('method', 'none')
        # --- MODIFIED: Initialize detailed info dictionary ---
        downsampling_details = {
            'downsampling_method': downsampling_method,
            'scenario': 'not_applicable',
            'n_donor_original': 0,
            'n_recipient_original': 0,
            'final_fraction_used': 0.0,
            'n_donor_after_downsampling': 0
        }

        if downsampling_method == 'none':
            return adata_patient, downsampling_details
        
        donor_mask = adata_patient.obs[donor_recipient_col] == 'donor'
        n_donor_original = int(donor_mask.sum())
        n_recipient_original = int((~donor_mask).sum())
        
        downsampling_details.update({
            'n_donor_original': n_donor_original,
            'n_recipient_original': n_recipient_original
        })

        if n_donor_original == 0:
            self.logger.info(f"  No donor cells found for patient. Skipping downsampling.")
            downsampling_details['scenario'] = 'no_donor_cells'
            downsampling_details['n_donor_after_downsampling'] = 0
            return adata_patient, downsampling_details

        patient_target_fraction = default_target_fraction
        scenario = "default_fallback"
        
        if n_recipient_original > 0:
            current_ratio = n_donor_original / n_recipient_original
            if current_ratio <= target_ratio_threshold:
                patient_target_fraction = 1.0
                scenario = "ratio_below_threshold"
                self.logger.info(f"  Scenario: {scenario}. Donor/recipient ratio ({current_ratio:.2f}) is below threshold ({target_ratio_threshold}). Keeping all {n_donor_original} donor cells.")
            else:
                target_n_donors = int(n_recipient_original * target_ratio_threshold)
                patient_target_fraction = target_n_donors / n_donor_original
                patient_target_fraction = max(0, min(1.0, patient_target_fraction))
                scenario = "ratio_above_threshold"
                self.logger.info(f"  Scenario: {scenario}. Donor/recipient ratio ({current_ratio:.2f}) is above threshold. Adjusting target fraction to {patient_target_fraction:.3f} to reach target ratio.")
        else:
             scenario = "no_recipient_cells"
             self.logger.info(f"  Scenario: {scenario}. No recipient cells found. Using default target fraction {default_target_fraction:.3f} for {n_donor_original} donor cells.")
        
        self.logger.info(f"  Applying '{downsampling_method}' downsampling with final target fraction: {patient_target_fraction:.3f}")
        
        adata_donor = adata_patient[donor_mask].copy()
        adata_recipient = adata_patient[~donor_mask].copy()
        
        min_cells_per_type = downsampling_config.get('min_cells_per_type', 1)
        
        # --- FIX: Use the correct cell_type_col for stratification ---
        stratification_col = downsampling_config.get('cell_type_column', 'predicted.annotation')
        self.logger.info(f"  Stratifying downsampling by column: '{stratification_col}'")
        
        adata_donor_downsampled, per_stratum_log = random_downsample_stratified(
            adata_donor,
            strata_col=stratification_col,
            target_fraction=patient_target_fraction,
            min_cells_per_stratum=min_cells_per_type,
            random_state=42
        )
        # --- END FIX ---
        
        n_kept = adata_donor_downsampled.n_obs
        
        # --- MODIFIED: Update detailed info dictionary ---
        downsampling_details.update({
            'scenario': scenario,
            'final_fraction_used': patient_target_fraction,
            'n_donor_after_downsampling': n_kept,
            'per_cell_type_counts': per_stratum_log
        })
        
        return adata_patient_processed, downsampling_details


def main():
    """Example of running the pipeline."""
    # This is for demonstration. Use runner scripts for actual runs.
    
    # Create experiment manager
    experiment_manager = ExperimentManager("experiments")
    
    # Create configuration
    config = create_standard_config(
        dr_method='nmf',
        n_components=100,
        downsampling_method='random',
        target_donor_fraction=0.5
    )
    config.config['preprocessing']['standardize'] = False
    
    # Create and run pipeline
    pipeline = StandardizedPipeline(experiment_manager)
    
    input_path = "/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad"
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    experiment = pipeline.run_pipeline(config, input_path)
    print(f"Completed experiment: {experiment.config.experiment_id}")


if __name__ == '__main__':
    main() 