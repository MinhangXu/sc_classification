#!/usr/bin/env python
"""
Standardized Preprocessing Module for Single-Cell Classification Pipeline

This module provides consistent preprocessing steps:
1. Data filtering and subsetting
2. Highly Variable Gene selection
3. Data standardization
4. Result tracking and validation
"""

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
import logging


class PreprocessingPipeline:
    """Standardized preprocessing pipeline for single-cell data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for preprocessing steps."""
        logger = logging.getLogger(f"preprocessing_{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def filter_data(self, adata: sc.AnnData) -> sc.AnnData:
        """Filter data based on configuration parameters."""
        self.logger.info("Starting data filtering...")
        
        # Get configuration parameters
        timepoint_col = self.config.get('timepoint_col', 'timepoint_type')
        timepoint_filter = self.config.get('timepoint_filter', 'MRD')
        target_col = self.config.get('target_column', 'CN.label')
        positive_class = self.config.get('positive_class', 'cancer')
        negative_class = self.config.get('negative_class', 'normal')
        
        # Filter by timepoint
        if timepoint_col in adata.obs.columns:
            timepoint_mask = adata.obs[timepoint_col] == timepoint_filter
            adata_filtered = adata[timepoint_mask].copy()
            self.logger.info(f"Filtered for {timepoint_filter} timepoint: {adata_filtered.n_obs} cells")
        else:
            self.logger.warning(f"Timepoint column '{timepoint_col}' not found. Skipping timepoint filtering.")
            adata_filtered = adata.copy()
        
        # Filter by target labels
        if target_col in adata_filtered.obs.columns:
            valid_targets = [positive_class, negative_class]
            target_mask = adata_filtered.obs[target_col].isin(valid_targets) & \
                         ~adata_filtered.obs[target_col].isna()
            adata_filtered = adata_filtered[target_mask].copy()
            self.logger.info(f"Filtered for valid targets {valid_targets}: {adata_filtered.n_obs} cells")
        else:
            self.logger.warning(f"Target column '{target_col}' not found. Skipping target filtering.")
        
        # Check if we have enough data
        if adata_filtered.n_obs == 0:
            raise ValueError("No cells remaining after filtering!")
        
        self.logger.info(f"Data filtering complete. Final shape: {adata_filtered.shape}")
        return adata_filtered
    
    def select_hvgs(self, adata: sc.AnnData) -> Tuple[sc.AnnData, List[str]]:
        """Select highly variable genes."""
        self.logger.info("Starting HVG selection...")
        
        n_top_genes = self.config.get('n_top_genes', 3000)
        
        # Check if we need to calculate HVGs
        if 'highly_variable' in adata.var.columns:
            n_hvgs = adata.var['highly_variable'].sum()
            self.logger.info(f"Using existing HVG annotation: {n_hvgs} genes")
        else:
            self.logger.info(f"Calculating {n_top_genes} highly variable genes...")
            
            # Determine if data is suitable for seurat_v3
            try:
                if np.any(adata.X < 0) or np.any(adata.X % 1 != 0):
                    self.logger.info("Data contains negative/decimal values. Using simple variance for HVG selection.")
                    gene_variances = np.var(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, axis=0)
                    hvg_indices = np.argsort(gene_variances)[-n_top_genes:]
                    adata.var['highly_variable'] = False
                    adata.var.iloc[hvg_indices, adata.var.columns.get_loc('highly_variable')] = True
                else:
                    self.logger.info("Attempting HVG selection with flavor 'seurat_v3'.")
                    sc.pp.highly_variable_genes(
                        adata,
                        n_top_genes=n_top_genes,
                        flavor='seurat_v3'
                    )
            except Exception as e:
                self.logger.warning(f"Error during HVG selection: {e}. Falling back to simple variance.")
                gene_variances = np.var(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, axis=0)
                hvg_indices = np.argsort(gene_variances)[-n_top_genes:]
                adata.var['highly_variable'] = False
                adata.var.iloc[hvg_indices, adata.var.columns.get_loc('highly_variable')] = True
        
        # Subset to HVGs
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
        hvg_list = adata_hvg.var_names.tolist()
        
        self.logger.info(f"HVG selection complete. Selected {len(hvg_list)} genes.")
        return adata_hvg, hvg_list
    
    def standardize_data(self, adata: sc.AnnData) -> Tuple[sc.AnnData, StandardScaler]:
        """Standardize the data."""
        self.logger.info("Starting data standardization...")
        
        # Check if standardization is requested
        if not self.config.get('standardize', True):
            self.logger.info("Standardization disabled. Returning original data.")
            return adata, None
        
        # Prepare data for scaling
        if hasattr(adata.X, 'toarray'):
            X_for_scaling = adata.X.toarray()
        else:
            X_for_scaling = adata.X.copy()
        
        # Perform standardization
        scaler = StandardScaler(with_mean=True)
        X_scaled = scaler.fit_transform(X_for_scaling)
        
        # Create new AnnData with scaled data
        adata_scaled = adata.copy()
        adata_scaled.X = X_scaled
        
        # Validate standardization
        mean_check = np.abs(np.mean(adata_scaled.X, axis=0)).max()
        var_check = np.abs(np.var(adata_scaled.X, axis=0) - 1.0).max()
        
        self.logger.info(f"Standardization complete. Max mean deviation: {mean_check:.6f}, Max variance deviation: {var_check:.6f}")
        
        return adata_scaled, scaler
    
    def run_preprocessing(self, adata: sc.AnnData) -> Dict[str, Any]:
        """Run the complete preprocessing pipeline."""
        self.logger.info("Starting complete preprocessing pipeline...")
        
        # Track preprocessing steps
        preprocessing_info = {
            'original_shape': adata.shape,
            'steps_completed': [],
            'warnings': []
        }
        
        try:
            # Step 1: Filter data
            adata_filtered = self.filter_data(adata)
            preprocessing_info['filtered_shape'] = adata_filtered.shape
            preprocessing_info['steps_completed'].append('filtering')
            
            # Step 2: Select HVGs
            adata_hvg, hvg_list = self.select_hvgs(adata_filtered)
            preprocessing_info['hvg_shape'] = adata_hvg.shape
            preprocessing_info['n_hvgs'] = len(hvg_list)
            preprocessing_info['steps_completed'].append('hvg_selection')
            
            # Step 3: Standardize data
            adata_final, scaler = self.standardize_data(adata_hvg)
            preprocessing_info['final_shape'] = adata_final.shape
            preprocessing_info['steps_completed'].append('standardization')
            
            # Generate summary
            summary = self._generate_summary(preprocessing_info)
            
            self.logger.info("Preprocessing pipeline completed successfully!")
            
            return {
                'adata': adata_final,
                'hvg_list': hvg_list,
                'scaler': scaler,
                'summary': summary,
                'info': preprocessing_info
            }
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise
    
    def _generate_summary(self, info: Dict[str, Any]) -> str:
        """Generate a text summary of preprocessing results."""
        summary_lines = [
            "Preprocessing Pipeline Summary",
            "=" * 40,
            f"Original data shape: {info['original_shape']}",
            f"After filtering: {info['filtered_shape']}",
            f"After HVG selection: {info['hvg_shape']}",
            f"Final shape: {info['final_shape']}",
            f"Number of HVGs: {info['n_hvgs']}",
            "",
            "Steps completed:",
        ]
        
        for step in info['steps_completed']:
            summary_lines.append(f"  - {step}")
        
        if info.get('warnings'):
            summary_lines.extend(["", "Warnings:"])
            for warning in info['warnings']:
                summary_lines.append(f"  - {warning}")
        
        return "\n".join(summary_lines)
    
    def validate_preprocessing(self, adata: sc.AnnData) -> Dict[str, Any]:
        """Validate preprocessing results."""
        validation_results = {
            'passed': True,
            'checks': {}
        }
        
        # Check for NaN values
        if np.any(np.isnan(adata.X)):
            validation_results['checks']['no_nans'] = False
            validation_results['passed'] = False
        else:
            validation_results['checks']['no_nans'] = True
        
        # Check for infinite values
        if np.any(np.isinf(adata.X)):
            validation_results['checks']['no_infs'] = False
            validation_results['passed'] = False
        else:
            validation_results['checks']['no_infs'] = True
        
        # Check data range (for standardized data)
        if self.config.get('standardize', True):
            data_range = np.ptp(adata.X, axis=0)
            if np.any(data_range > 10):  # Reasonable range for standardized data
                validation_results['checks']['reasonable_range'] = False
                validation_results['warnings'] = ['Some features have unusually large ranges']
            else:
                validation_results['checks']['reasonable_range'] = True
        
        # Check minimum cell count
        if adata.n_obs < 100:
            validation_results['checks']['sufficient_cells'] = False
            validation_results['passed'] = False
        else:
            validation_results['checks']['sufficient_cells'] = True
        
        # Check minimum gene count
        if adata.n_vars < 100:
            validation_results['checks']['sufficient_genes'] = False
            validation_results['passed'] = False
        else:
            validation_results['checks']['sufficient_genes'] = True
        
        return validation_results


def create_preprocessing_config(n_top_genes: int = 3000, 
                              standardize: bool = True,
                              timepoint_filter: str = 'MRD',
                              target_column: str = 'CN.label',
                              positive_class: str = 'cancer',
                              negative_class: str = 'normal') -> Dict[str, Any]:
    """Create a standard preprocessing configuration."""
    
    return {
        'n_top_genes': n_top_genes,
        'standardize': standardize,
        'timepoint_col': 'timepoint_type',
        'timepoint_filter': timepoint_filter,
        'target_column': target_column,
        'positive_class': positive_class,
        'negative_class': negative_class
    }

def random_downsample_stratified(adata_donor, strata_col, 
                                 target_fraction=0.5, min_cells_per_stratum=1, random_state=42):
    """ 
    Randomly downsamples cells, stratified by a given column.
    
    Returns a tuple of:
    1. A new, downsampled AnnData object.
    2. A dictionary logging the counts per stratum.
    """
    np.random.seed(random_state)
    target_fraction = max(0, min(1.0, target_fraction))
    
    if target_fraction == 1.0:
        # --- MODIFIED: Return counts even if no downsampling occurs ---
        per_stratum_counts = {
            group: {'before': adata_donor[adata_donor.obs[strata_col] == group].n_obs, 'after': adata_donor[adata_donor.obs[strata_col] == group].n_obs}
            for group in adata_donor.obs[strata_col].unique()
        }
        return adata_donor.copy(), per_stratum_counts
        
    kept_indices = []
    # --- MODIFIED: Initialize dictionary to log per-stratum counts ---
    per_stratum_counts = {}
    
    for group in adata_donor.obs[strata_col].unique():
        strata_adata = adata_donor[adata_donor.obs[strata_col] == group]
        n_cells_strata = strata_adata.n_obs
        
        n_to_keep_frac = int(n_cells_strata * target_fraction)
        n_to_keep = max(min_cells_per_stratum, n_to_keep_frac)
        n_to_keep = min(n_to_keep, n_cells_strata)

        # --- MODIFIED: Log the before and after counts for this stratum ---
        per_stratum_counts[group] = {'before': n_cells_strata, 'after': n_to_keep}

        if n_to_keep > 0:
            chosen_indices = np.random.choice(strata_adata.obs.index, size=n_to_keep, replace=False)
            kept_indices.extend(chosen_indices.tolist())
            
    # --- MODIFIED: Return the downsampled data AND the logging dictionary ---
    return adata_donor[kept_indices].copy(), per_stratum_counts