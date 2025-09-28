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
from scipy import sparse


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
        tech_filter = self.config.get('tech_filter', None)
        
        adata_filtered = adata.copy()

        # Filter by technology
        if tech_filter and 'Tech' in adata_filtered.obs.columns:
            tech_mask = adata_filtered.obs['Tech'] == tech_filter
            adata_filtered = adata_filtered[tech_mask].copy()
            self.logger.info(f"Filtered for '{tech_filter}' technology: {adata_filtered.n_obs} cells remaining.")
        elif tech_filter:
            self.logger.warning(f"Technology column 'Tech' not found. Skipping technology filtering.")

        # Filter by timepoint
        if timepoint_col in adata_filtered.obs.columns:
            timepoint_mask = adata_filtered.obs[timepoint_col] == timepoint_filter
            adata_filtered = adata_filtered[timepoint_mask].copy()
            self.logger.info(f"Filtered for {timepoint_filter} timepoint: {adata_filtered.n_obs} cells remaining.")
        else:
            self.logger.warning(f"Timepoint column '{timepoint_col}' not found. Skipping timepoint filtering.")
        
        # Filter by target labels
        if target_col in adata_filtered.obs.columns:
            valid_targets = [positive_class, negative_class]
            target_mask = adata_filtered.obs[target_col].isin(valid_targets) & \
                         ~adata_filtered.obs[target_col].isna()
            adata_filtered = adata_filtered[target_mask].copy()
            self.logger.info(f"Filtered for valid targets {valid_targets}: {adata_filtered.n_obs} cells remaining.")
        else:
            self.logger.warning(f"Target column '{target_col}' not found. Skipping target filtering.")
        
        # Check if we have enough data
        if adata_filtered.n_obs == 0:
            raise ValueError("No cells remaining after filtering!")
        
        self.logger.info(f"Data filtering complete. Final shape: {adata_filtered.shape}")
        return adata_filtered
    
    def run_gene_selection_pipeline(self, adata: sc.AnnData) -> Tuple[sc.AnnData, Dict[str, Any]]:
        """
        Run a sequence of gene selection steps defined in the configuration.
        """
        self.logger.info("Starting gene selection pipeline...")
        
        # Default to standard HVG selection if no pipeline is specified
        selection_pipeline = self.config.get('gene_selection_pipeline', 
                                             [{'method': 'hvg', 'n_top_genes': self.config.get('n_top_genes', 3000)}])
        
        adata_current = adata.copy()
        
        for i, step_config in enumerate(selection_pipeline):
            method = step_config.get('method')
            self.logger.info(f"Step {i+1}/{len(selection_pipeline)}: Applying '{method}' selection.")
            
            if method == 'hvg':
                adata_current, gene_log = self._select_hvgs(adata_current, step_config)
            elif method == 'all_filtered':
                adata_current, gene_log = self._select_all_genes_filtered(adata_current, step_config)
            elif method == 'deg_weak_screen':
                adata_current, gene_log = self._select_degs_weak_screen(adata_current, step_config)
            else:
                self.logger.warning(f"Unknown gene selection method '{method}'. Skipping step.")
                continue
                
        final_gene_list = adata_current.var_names.tolist()
        self.logger.info(f"Gene selection pipeline complete. Final selected genes: {len(final_gene_list)}")
        
        return adata_current, gene_log

    def _select_all_genes_filtered(self, adata: sc.AnnData, gs_config: Dict[str, Any]) -> Tuple[sc.AnnData, Dict[str, List[str]]]:
        """
        Keep all genes except low-expression and low-mean/high-variance outliers.
        Includes a supervised step to "rescue" lowly-expressed genes that are
        highly specific to the positive class.
        """
        self.logger.info(f"Applying 'all_filtered' gene selection with config: {gs_config}")
        
        # --- Initialize log dictionary ---
        gene_log = {
            'initial_genes': adata.var_names.tolist(),
            'genes_removed_min_cells': [],
            'genes_rescued_enrichment': [],
            'genes_removed_noise': [],
            'final_gene_set': []
        }
        
        # --- Get Parameters ---
        min_cells = gs_config.get('min_cells', None)
        min_cells_fraction = gs_config.get('min_cells_fraction', 0.05)
        
        # Supervised rescue parameters
        enrichment_ratio_threshold = gs_config.get('malignant_enrichment_ratio', None)
        target_col = self.config.get('target_column', 'CN.label')
        positive_class = self.config.get('positive_class', 'cancer')
        
        X = adata.X

        if sparse.issparse(X):
            expressed_counts = np.asarray((X > 0).sum(axis=0)).ravel()
        else:
            expressed_counts = np.sum(X > 0, axis=0)

        if min_cells is None:
            min_cells = max(1, int(min_cells_fraction * adata.n_obs))

        initial_keep_mask = expressed_counts >= min_cells
        self.logger.info(f"Filtering genes expressed in fewer than {min_cells} cells.")
        self.logger.info(f"  {np.sum(initial_keep_mask)} genes remaining after min cell filter.")
        
        # Log removed genes
        gene_log['genes_removed_min_cells'] = adata.var_names[~initial_keep_mask].tolist()
        
        # --- Supervised Rescue Step ---
        genes_to_rescue = []
        if enrichment_ratio_threshold is not None:
            # Identify genes that were just filtered out
            genes_to_check_mask = ~initial_keep_mask
            genes_to_check_names = adata.var_names[genes_to_check_mask]
            
            self.logger.info(f"Checking {len(genes_to_check_names)} lowly expressed genes for malignant enrichment...")
            
            # Calculate enrichment ratio for these specific genes
            adata_check = adata[:, genes_to_check_names].copy()
            malignant_mask = adata_check.obs[target_col] == positive_class
            normal_mask = ~malignant_mask # More robust than specifying negative class
            
            n_malignant = np.sum(malignant_mask)
            n_normal = np.sum(normal_mask)
            
            if n_malignant > 0 and n_normal > 0:
                X_check = adata_check.X
                if sparse.issparse(X_check):
                    expr_in_mal = np.asarray((X_check[malignant_mask, :] > 0).sum(axis=0)).ravel()
                    expr_in_norm = np.asarray((X_check[normal_mask, :] > 0).sum(axis=0)).ravel()
                else:
                    expr_in_mal = np.sum(X_check[malignant_mask, :] > 0, axis=0)
                    expr_in_norm = np.sum(X_check[normal_mask, :] > 0, axis=0)

                prop_mal = expr_in_mal / n_malignant
                prop_norm = expr_in_norm / n_normal
                
                enrichment_ratio = prop_mal / (prop_norm + 1e-9)
                
                rescue_mask = enrichment_ratio > enrichment_ratio_threshold
                genes_to_rescue = genes_to_check_names[rescue_mask].tolist()
                self.logger.info(f"  Rescuing {len(genes_to_rescue)} genes with enrichment ratio > {enrichment_ratio_threshold}.")
                gene_log['genes_rescued_enrichment'] = genes_to_rescue

        # Combine initial keep mask with rescued genes
        final_keep_mask = initial_keep_mask.copy()
        rescue_indices = np.where(np.isin(adata.var_names, genes_to_rescue))[0]
        final_keep_mask[rescue_indices] = True
        self.logger.info(f"  Total genes after rescue step: {np.sum(final_keep_mask)}.")

        if gs_config.get('variance_filter', True):
            var_pct = gs_config.get('variance_percentile_high', 99.5)
            low_mean_q = gs_config.get('low_mean_quantile', 0.1)
            
            # Calculate metrics only on the genes we are keeping so far
            X_filtered = adata.X[:, final_keep_mask]
            
            if sparse.issparse(X_filtered):
                gene_means = np.asarray(X_filtered.mean(axis=0)).ravel()
                sq_means = np.asarray(X_filtered.power(2).mean(axis=0)).ravel()
                gene_vars = sq_means - np.square(gene_means)
            else:
                gene_means = np.asarray(X_filtered.mean(axis=0)).ravel()
                gene_vars = np.asarray(X_filtered.var(axis=0)).ravel()
                
            var_cut = np.percentile(gene_vars, var_pct)
            mean_cut = np.quantile(gene_means, low_mean_q)
            high_var_low_mean = (gene_vars > var_cut) & (gene_means < mean_cut)
            
            self.logger.info(f"Filtering {np.sum(high_var_low_mean)} noisy genes (var > {var_pct:.1f}th percentile and mean < {low_mean_q:.1f}th quantile).")
            
            # Create a mask of the same size as the original final_keep_mask
            noise_mask_full = np.zeros_like(final_keep_mask)
            # Place the noise results into the correct positions
            noise_mask_full[final_keep_mask] = high_var_low_mean
            
            # Log the noisy genes that are being removed
            noisy_genes_to_remove_mask = final_keep_mask & noise_mask_full
            gene_log['genes_removed_noise'] = adata.var_names[noisy_genes_to_remove_mask].tolist()
            
            final_keep_mask = final_keep_mask & (~noise_mask_full)
            self.logger.info(f"  {np.sum(final_keep_mask)} genes remaining after noise filter.")

        adata_sel = adata[:, final_keep_mask].copy()
        gene_log['final_gene_set'] = adata_sel.var_names.tolist()
        
        self.logger.info(f"Kept {adata_sel.n_vars} genes after 'all_filtered' selection.")
        return adata_sel, gene_log

    def _select_degs_weak_screen(self, adata: sc.AnnData, gs_config: Dict[str, Any]) -> Tuple[sc.AnnData, Dict[str, List[str]]]:
        """
        Univariate DEG weak screen between positive and negative classes.
        """
        self.logger.info(f"Applying 'deg_weak_screen' with config: {gs_config}")
        target_col = self.config.get('target_column', 'CN.label')
        pos_class = self.config.get('positive_class', 'cancer')
        neg_class = self.config.get('negative_class', 'normal')
        method = gs_config.get('deg_test_method', 'wilcoxon')
        pval_thresh = gs_config.get('pval_threshold', 0.1)
        use_adj_pvals = gs_config.get('use_adj_pvals', False)
        min_n_genes = gs_config.get('min_n_genes', 3000)
        if min_n_genes is not None:
            min_n_genes = int(min_n_genes)
            
        lfc_threshold = gs_config.get('lfc_threshold', 0) # New parameter, defaults to no filtering
        pval_key = 'pvals_adj' if use_adj_pvals else 'pvals'
        lfc_key = 'logfoldchanges'

        ad_temp = adata.copy()
        sc.tl.rank_genes_groups(ad_temp, groupby=target_col, groups=[pos_class], reference=neg_class,
                                method=method, use_raw=False, n_genes=adata.n_vars)

        df = sc.get.rank_genes_groups_df(ad_temp, group=pos_class)
        df = df.dropna(subset=[pval_key, lfc_key])
        
        # Apply both p-value and LFC thresholds
        selection_mask = (df[pval_key] < pval_thresh) & (df[lfc_key].abs() > lfc_threshold)
        selected_genes = df[selection_mask]['names'].tolist()
        self.logger.info(f"Found {len(selected_genes)} genes with {pval_key} < {pval_thresh} and |LFC| > {lfc_threshold}.")

        if min_n_genes is not None and len(selected_genes) < min_n_genes:
            self.logger.info(f"Fewer than {min_n_genes} genes passed threshold. Topping up to {min_n_genes} using lowest p-values.")
            # Top up with genes that pass p-value cut, sorted by LFC, then p-value
            df_sorted_for_topup = df.reindex(df[lfc_key].abs().sort_values(ascending=False).index)
            additional_genes = df_sorted_for_topup['names'].tolist()
            for gene in additional_genes:
                if gene not in selected_genes:
                    selected_genes.append(gene)
                if len(selected_genes) >= min_n_genes:
                    break
        
        # Ensure genes are actually in the original AnnData's var_names
        selected_genes = [g for g in selected_genes if g in adata.var_names]
        adata_sel = adata[:, selected_genes].copy()
        
        self.logger.info(f"Kept {adata_sel.n_vars} genes after 'deg_weak_screen' selection.")
        # For compatibility, return a placeholder log
        gene_log = {'final_gene_set': selected_genes}
        return adata_sel, gene_log

    def _select_hvgs(self, adata: sc.AnnData, gs_config: Dict[str, Any]) -> Tuple[sc.AnnData, Dict[str, List[str]]]:
        """Select highly variable genes."""
        self.logger.info(f"Applying 'hvg' selection with config: {gs_config}")
        
        n_top_genes = gs_config.get('n_top_genes', self.config.get('n_top_genes', 3000))
        
        # Check if we need to calculate HVGs
        if 'highly_variable' in adata.var.columns and adata.var['highly_variable'].sum() >= n_top_genes:
            n_hvgs = adata.var['highly_variable'].sum()
            self.logger.info(f"Using existing HVG annotation found for {n_hvgs} genes.")
        else:
            self.logger.info(f"Calculating {n_top_genes} highly variable genes...")
            
            # Use seurat_v3 flavor as it's common for UMI-based counts, with a fallback
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_top_genes,
                    flavor='seurat_v3'
                )
            except Exception as e:
                self.logger.warning(f"HVG selection with 'seurat_v3' failed: {e}. Falling back to simple variance.")
                gene_variances = np.var(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X, axis=0)
                hvg_indices = np.argsort(gene_variances)[-n_top_genes:]
                adata.var['highly_variable'] = False
                adata.var.iloc[hvg_indices, adata.var.columns.get_loc('highly_variable')] = True
        
        # Subset to HVGs
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
        self.logger.info(f"Kept {adata_hvg.n_vars} genes after 'hvg' selection.")
        # For compatibility, return a placeholder log
        gene_log = {'final_gene_set': adata_hvg.var_names.tolist()}
        return adata_hvg, gene_log
    
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
            'log': [],
            'warnings': []
        }
        
        try:
            # Step 1: Filter data
            adata_filtered = self.filter_data(adata)
            preprocessing_info['filtered_shape'] = adata_filtered.shape
            preprocessing_info['steps_completed'].append('filtering')
            
            # Step 2: Select genes by running the configured pipeline
            adata_selected, gene_log = self.run_gene_selection_pipeline(adata_filtered)
            preprocessing_info['gene_selection_shape'] = adata_selected.shape
            preprocessing_info['n_selected_genes'] = len(gene_log['final_gene_set'])
            preprocessing_info['gene_log'] = gene_log # Add the detailed log
            preprocessing_info['steps_completed'].append('gene_selection')
            
            # Step 3: Standardize data
            adata_final, scaler = self.standardize_data(adata_selected)
            preprocessing_info['final_shape'] = adata_final.shape
            preprocessing_info['steps_completed'].append('standardization')
            
            # Generate summary
            summary = self._generate_summary(preprocessing_info)
            
            self.logger.info("Preprocessing pipeline completed successfully!")
            
            return {
                'adata': adata_final,
                'hvg_list': gene_log['final_gene_set'], # Kept name for compatibility
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
            f"After cell filtering: {info['filtered_shape']}",
            f"After gene selection: {info.get('gene_selection_shape', 'N/A')}",
            f"Final shape: {info['final_shape']}",
            f"Number of selected genes: {info.get('n_selected_genes', 'N/A')}",
            "",
            "Steps completed:",
        ]
        
        for step in info['steps_completed']:
            summary_lines.append(f"  - {step}")
        
        # --- Gene Selection Logging ---
        gene_log = info.get('gene_log', {})
        if gene_log:
            summary_lines.extend([
                "",
                "Gene Selection Details:",
                f"  Initial gene count: {len(gene_log.get('initial_genes', []))}",
                f"  Genes removed by min cell filter: {len(gene_log.get('genes_removed_min_cells', []))}",
                f"  Genes rescued by enrichment: {len(gene_log.get('genes_rescued_enrichment', []))}",
                f"  Genes removed by noise filter: {len(gene_log.get('genes_removed_noise', []))}",
                f"  Final gene set size: {len(gene_log.get('final_gene_set', []))}",
            ])

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
                              negative_class: str = 'normal',
                              tech_filter: Optional[str] = None,
                              gene_selection_pipeline: Optional[List[Dict[str, Any]]] = None
                              ) -> Dict[str, Any]:
    """Create a standard preprocessing configuration."""
    
    config = {
        'n_top_genes': n_top_genes, # Keep for general reference / backward compatibility
        'standardize': standardize,
        'timepoint_col': 'timepoint_type',
        'timepoint_filter': timepoint_filter,
        'target_column': target_column,
        'positive_class': positive_class,
        'negative_class': negative_class,
        'tech_filter': tech_filter
    }

    if gene_selection_pipeline is not None:
        config['gene_selection_pipeline'] = gene_selection_pipeline
    else:
        # Default behavior is the classic HVG selection
        config['gene_selection_pipeline'] = [{'method': 'hvg', 'n_top_genes': n_top_genes}]
        
    return config


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
    # Initialize dictionary to log per-stratum counts ---
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