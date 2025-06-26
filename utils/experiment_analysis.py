import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import scanpy as sc
import pickle
from .experiment_manager import ExperimentManager


class ExperimentAnalyzer:
    """Analyzer for comparing and summarizing experiments."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments and return summary statistics."""
        comparison_data = []

        for exp_id in experiment_ids:
            try:
                exp = self.experiment_manager.load_experiment(exp_id)
                metadata = exp.load_metadata()

                # Basic experiment info
                exp_info = {
                    'experiment_id': exp_id,
                    'dr_method': exp.config.get('dimension_reduction.method'),
                    'n_components': exp.config.get('dimension_reduction.n_components'),
                    'downsampling': exp.config.get('downsampling.method'),
                    'target_donor_fraction': exp.config.get('downsampling.target_donor_fraction'),
                    'creation_date': metadata.get('creation_date'),
                    'status': metadata.get('status', 'unknown'),
                    'stages_completed': ','.join(metadata.get('stages_completed', []))
                }

                # Add preprocessing info if available
                if 'preprocessing' in metadata:
                    exp_info.update({
                        'n_cells': metadata['preprocessing'].get('n_cells'),
                        'n_genes': metadata['preprocessing'].get('n_genes'),
                        'n_hvgs': metadata['preprocessing'].get('n_hvgs')
                    })

                # Add classification info if available
                if 'patients_completed' in metadata:
                    exp_info['n_patients'] = len(metadata['patients_completed'])
                    exp_info['patients_completed'] = ','.join(metadata['patients_completed'])

                comparison_data.append(exp_info)

            except Exception as e:
                print(f"Error loading experiment {exp_id}: {e}")

        return pd.DataFrame(comparison_data)

    def extract_classification_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Extracts classification results.
        Returns a dictionary containing 'metrics' (a single long-format DataFrame)
        and 'coefficients' (a dictionary of patient-keyed wide-format DataFrames).
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        metadata = exp.load_metadata()
        patients = metadata.get('patients_completed', [])

        if not patients:
            print("Warning: 'patients_completed' not found in metadata.json. Attempting to discover patients by scanning the classification directory.")
            classification_dir = exp.experiment_dir / "models" / "classification"
            if classification_dir.exists():
                patients = sorted([p.name for p in classification_dir.iterdir() if p.is_dir()])
                print(f"Discovered patients by directory scan: {patients}")
            else:
                print(f"Error: Classification directory does not exist at {classification_dir}")
                return {}

        if not patients:
            print(f"Warning: No completed patients found for experiment {experiment_id}.")
            return {}

        all_coefficients = {}
        all_metrics_list = []

        for patient_id in patients:
            try:
                coef_path = exp.get_path('patient_coefficients', patient_id=patient_id)
                if coef_path.exists():
                    all_coefficients[patient_id] = pd.read_csv(coef_path, index_col=0)
                else:
                    print(f"DEBUG: Coefficients file not found for patient {patient_id} at {coef_path}")

                metrics_path = exp.get_path('patient_metrics', patient_id=patient_id)
                if metrics_path.exists():
                    metrics_df = pd.read_csv(metrics_path)
                    if 'group' not in metrics_df.columns:
                        metrics_df['group'] = patient_id
                    all_metrics_list.append(metrics_df)
                else:
                    print(f"DEBUG: Metrics file not found for patient {patient_id} at {metrics_path}")

            except Exception as e:
                print(f"Error loading results for patient {patient_id}: {e}")

        results = {}
        if all_coefficients:
            results['coefficients'] = all_coefficients
        if all_metrics_list:
            results['metrics'] = pd.concat(all_metrics_list, ignore_index=True)

        return results

    def compare_classification_performance(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare classification performance across experiments."""
        all_metrics = []

        for exp_id in experiment_ids:
            try:
                results = self.extract_classification_results(exp_id)
                if 'metrics' in results:
                    metrics_df = results['metrics']
                    metrics_df['experiment_id'] = exp_id

                    # Add experiment metadata
                    exp = self.experiment_manager.load_experiment(exp_id)
                    metrics_df['dr_method'] = exp.config.get('dimension_reduction.method')
                    metrics_df['n_components'] = exp.config.get('dimension_reduction.n_components')
                    metrics_df['downsampling'] = exp.config.get('downsampling.method')

                    all_metrics.append(metrics_df)

            except Exception as e:
                print(f"Error extracting metrics for experiment {exp_id}: {e}")

        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        else:
            return pd.DataFrame()
    def _plot_metrics_and_coefficients(self, metrics_df, coefs_df, patient_id_for_filtering="all_samples", 
                                      display_patient_id=None, top_n=10, alpha_idx=None, 
                                      n_factors_info=None):
        # This is the user's function, integrated into the class
        if display_patient_id is None:
            display_patient_id = patient_id_for_filtering

        metrics_results = metrics_df[metrics_df['group'] == patient_id_for_filtering]
        
        if metrics_results.empty:
            print(f"No metrics found for patient_id_for_filtering (group) '{patient_id_for_filtering}'")
            return None

        try:
            alphas = coefs_df.columns.astype(float).values
        except ValueError:
            print("Error: Could not convert coefficient DataFrame columns to float for alpha values.")
            return None
            
        coef_results_arr = np.array(coefs_df)
        feature_names = coefs_df.index

        overall_acc = metrics_results['overall_accuracy'].values
        mal_accuracy = metrics_results['mal_accuracy'].values
        norm_accuracy = metrics_results['norm_accuracy'].values
        has_roc = 'roc_auc' in metrics_results.columns
        roc_auc = metrics_results['roc_auc'].values if has_roc else None
        surviving_features_count = (coefs_df != 0).sum(axis=0).values
        majority_num = metrics_results['majority_num'].values[0] if 'majority_num' in metrics_results.columns and len(metrics_results['majority_num'].values)>0 else "N/A"
        minority_num = metrics_results['minority_num'].values[0] if 'minority_num' in metrics_results.columns and len(metrics_results['minority_num'].values)>0 else "N/A"

        non_zero_counts_per_feature = (coefs_df != 0).sum(axis=1)
        actual_top_n = min(top_n, len(feature_names))
        if actual_top_n == 0:
            top_features_idx = []
        else:
            top_features_idx = np.argsort(non_zero_counts_per_feature)[-actual_top_n:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 3]})
        fig.patch.set_facecolor('white')

        log_alphas = np.log10(alphas)

        ax1.plot(log_alphas, overall_acc, 'o-', label="Overall Accuracy", color='skyblue', linewidth=1.5, alpha=0.8, markersize=5)
        ax1.plot(log_alphas, mal_accuracy, '^-', label="Cancer Cell Accuracy", color='darkblue', linewidth=1.5, alpha=0.8, markersize=5)
        ax1.plot(log_alphas, norm_accuracy, 's-', label="Normal Cell Accuracy", color='green', linewidth=1.5, alpha=0.8, markersize=5)

        if "trivial_accuracy" in metrics_results.columns and len(metrics_results['trivial_accuracy'].values) > 0:
            trivial_acc = metrics_results['trivial_accuracy'].values
            if not np.all(np.isnan(trivial_acc)):
                ax1.plot(log_alphas, trivial_acc, '--', label=f"Trivial (Majority) Acc = {trivial_acc[~np.isnan(trivial_acc)][0]:.3f}", color='red', linewidth=2, alpha=0.7)

        if roc_auc is not None:
            ax1.plot(log_alphas, roc_auc, 'd-', label="ROC AUC", color='purple', linewidth=1.5, alpha=0.8, markersize=5)

        ax1.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=12)
        ax1.set_ylabel("Accuracy / AUC", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlim(np.min(log_alphas) - 0.5, np.max(log_alphas) + 0.5)
        
        ax1_2 = ax1.twinx()
        surviving_features_percent = (surviving_features_count / len(feature_names) * 100) if len(feature_names) > 0 else np.zeros_like(surviving_features_count)
        ax1_2.plot(log_alphas, surviving_features_percent, 'p-', color='orange', label="Surviving Features (%)", alpha=0.8, markersize=5)
        ax1_2.set_ylabel("Surviving Features (%)", fontsize=12)
        ax1_2.set_ylim([-5, 105])

        title_str = f"Classification Metrics & Feature Survival vs. Regularization (Patient: {display_patient_id})"
        if n_factors_info:
            title_str += f"\n{n_factors_info}"
        ax1.set_title(title_str, fontsize=14, pad=20)

        if alpha_idx is not None and 0 <= alpha_idx < len(alphas):
            ax1.axvline(x=log_alphas[alpha_idx], color='dimgray', linestyle='-.', linewidth=2.5, label=f"Selected $\lambda$={alphas[alpha_idx]:.2e}", alpha=0.7)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_2.get_legend_handles_labels()
        extra_labels_list = [f"Normal Cells (Maj.): {majority_num}", f"Cancer Cells (Min.): {minority_num}"]
        dummy_lines = [plt.Line2D([0], [0], linestyle="none", c=c, marker='o') for c in ['green', 'darkblue']]
        ax1.legend(dummy_lines + lines_1 + lines_2, extra_labels_list + labels_1 + labels_2, loc='center left', bbox_to_anchor=(1.20, 0.5), fontsize=10, frameon=True)

        if len(top_features_idx) > 0:
            colors = plt.cm.get_cmap('tab10' if actual_top_n <= 10 else 'viridis', actual_top_n)
            for i, idx in enumerate(top_features_idx):
                ax2.plot(log_alphas, coef_results_arr[idx], label=feature_names[idx], alpha=0.85, linewidth=1.5, color=colors(i))
        else:
            ax2.text(0.5, 0.5, "No features to plot.", ha='center', va='center', transform=ax2.transAxes)

        if alpha_idx is not None and 0 <= alpha_idx < len(alphas):
            ax2.axvline(x=log_alphas[alpha_idx], color='dimgray', linestyle='-.', linewidth=2.5, alpha=0.7)

        ax2.set_xlim(np.min(log_alphas) - 0.5, np.max(log_alphas) + 0.5)
        ax2.set_xlabel(r"$\log_{10}(\lambda)$  ($\lambda$ = Lasso Regularization Strength)", fontsize=12)
        ax2.set_ylabel("Coefficient Value", fontsize=12)
        ax2.set_title(f"Top-{actual_top_n} Factor Coefficient Paths", fontsize=14, pad=10)
        ax2.axhline(0, color='black', linestyle=':', lw=1.5)
        if len(top_features_idx) > 0:
            ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        return fig
    
    def generate_lr_lasso_analysis_plots(self, experiment_id: str, output_dir: str = None):
        """
        Generate LR-Lasso analysis plots using the user's detailed plotting functions.
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')

        if output_dir is None:
            output_path = exp.get_path('summary_plots')
        else:
            output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        results = self.extract_classification_results(experiment_id)
        if 'metrics' not in results or 'coefficients' not in results:
            print(f"No classification results found for experiment {experiment_id}. Cannot generate plots.")
            return

        metrics_df = results['metrics']
        coefficients_dict = results['coefficients']
        
        for patient_id, coefs_df in coefficients_dict.items():
            print(f"Generating plot for patient: {patient_id}")
            
            # The plotting function expects alpha values as floats in the column headers
            try:
                # Store original columns before modification
                original_cols = coefs_df.columns
                coefs_df.columns = [float(col.split('_')[1]) for col in coefs_df.columns]
            except (ValueError, IndexError) as e:
                 print(f"  Warning: Could not parse alpha values from column headers for patient {patient_id}. Error: {e}")
                 continue # Skip this patient if columns are not in the expected format

            fig = self._plot_metrics_and_coefficients(
                metrics_df=metrics_df,
                coefs_df=coefs_df,
                patient_id_for_filtering=patient_id,
                n_factors_info=f"{n_components} {dr_method.upper()} Factors"
            )

            # Restore original column names if needed elsewhere, though not necessary here
            coefs_df.columns = original_cols

            if fig:
                save_filepath = output_path / f"patient_{patient_id}_metrics_and_coefficients.png"
                fig.savefig(save_filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved plot to {save_filepath}")
                plt.close(fig)

        if dr_method and dr_method.lower() == 'fa':
            self._create_fa_specific_plots(exp, n_components, output_path)
            
    def _create_fa_specific_plots(self, exp, n_factors, output_path):
        """Create FA-specific plots by calling the integrated helper methods."""
        print("Attempting to generate FA-specific diagnostic plots...")
        # Note: These functions rely on files that may not be generated by the current pipeline.
        # They are included here to match your previous workflow.
        
        # Construct path to a potential summary file from the experiment directory
        fa_summary_file_path = exp.experiment_dir / "models" / f"fa_{n_factors}" / "model_summary.txt"
        self._plot_factor_ss_loadings(fa_summary_file_path, n_factors, output_path)

        # Construct path to a potential communalities file
        communalities_csv_path = exp.experiment_dir / "models" / f"fa_{n_factors}" / "gene_communalities.csv"
        self._plot_communality_distribution(communalities_csv_path, n_factors, output_path)

    def _plot_factor_ss_loadings(self, fa_summary_file_path, n_factors, output_path):
            """Parses fa_summary.txt for SS Loadings per factor and plots them as a bar chart."""
            try:
                with open(fa_summary_file_path, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"Could not generate SS Loadings plot: Summary file not found at {fa_summary_file_path}")
                return None

            ss_loadings_match = re.search(
                r"Sum of Squared Loadings \(SS Loadings\) per Factor \(V_j\):\s*(\[[\s\S]*?\])", 
                content, 
                re.MULTILINE
            )

            if not ss_loadings_match:
                print(f"Could not find/parse SS Loadings block in FA summary file: {fa_summary_file_path}")
                return None

            ss_loadings_block_str = ss_loadings_match.group(1)
            numbers_str = ss_loadings_block_str.strip()[1:-1]
            try:
                ss_loadings = np.array([float(v) for v in re.split(r'\s+', numbers_str) if v])
            except ValueError as e:
                print(f"Error converting SS loadings to numbers after regex match: {e}. String was: '{numbers_str}'")
                return None

            if len(ss_loadings) == 0:
                print(f"No SS loading values were parsed from {fa_summary_file_path}")
                return None

            actual_n_factors_parsed = len(ss_loadings)
            if actual_n_factors_parsed != n_factors:
                print(f"Warning: Parsed {actual_n_factors_parsed} SS loadings, but expected {n_factors} factors. Plotting parsed values.")

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('white')
            factor_indices = np.arange(1, actual_n_factors_parsed + 1)
            
            sorted_indices = np.argsort(ss_loadings)[::-1]
            sorted_ss_loadings = ss_loadings[sorted_indices]
            
            ax.bar(factor_indices, sorted_ss_loadings, color='darkcyan', alpha=0.9, edgecolor='black')
            ax.plot(factor_indices, sorted_ss_loadings, marker='o', color='orangered', linestyle='-', linewidth=1.5)

            ax.set_xlabel("Factor Index (Sorted by SS Loadings)", fontsize=12)
            ax.set_ylabel("Sum of Squared Loadings ($V_j$)", fontsize=12)
            total_ss_loadings = np.sum(ss_loadings)
            title = f"FA: Sum of Squared Loadings per Factor ({actual_n_factors_parsed} Factors, Sorted)"
            subtitle = f"Total Sum of Squared Loadings (Total Communality): {total_ss_loadings:.3f}"
            ax.set_title(f"{title}\n{subtitle}", fontsize=14)
            
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins='auto'))
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_path = output_path / f"fa_factor_ss_loadings_{n_factors}factors.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved factor SS loadings plot to {save_path}")
            plt.close(fig)

    def _plot_communality_distribution(self, communalities_csv_path, n_factors, output_path):
            """Plots a histogram of gene communalities from a CSV file."""
            try:
                comm_df = pd.read_csv(communalities_csv_path)
                communalities = comm_df['communality'].values
            except FileNotFoundError:
                print(f"Could not generate communality plot: File not found at {communalities_csv_path}")
                return None
            except KeyError:
                print(f"Could not generate communality plot: 'communality' column not found in {communalities_csv_path}")
                return None
                
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('white')
            ax.hist(communalities, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.8)
            ax.set_xlabel("Communality ($h^2$)", fontsize=12)
            ax.set_ylabel("Number of Genes", fontsize=12)
            ax.set_title(f"Distribution of Gene Communalities ({n_factors} Factors)", fontsize=14)
            
            mean_comm = np.mean(communalities)
            median_comm = np.median(communalities)
            ax.axvline(mean_comm, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_comm:.3f}')
            ax.axvline(median_comm, color='purple', linestyle='dotted', linewidth=1.5, label=f'Median: {median_comm:.3f}')
            ax.legend(fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_path = output_path / f"fa_gene_communalities_dist_{n_factors}factors.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved communality distribution plot to {save_path}")
            plt.close(fig) 
            
    def _create_overall_factor_interpretation(self, exp, model, coef_df, adata, output_path):
        """Create overall factor interpretation (mapping LR coefficients with factor loadings)."""
        # This would implement the overall interpretation method
        # Similar to the approach in multiVI_FA_LR_eval_may31.ipynb
        
        # For now, create a basic mapping
        patients = coef_df['patient_id'].unique()
        
        for patient in patients:
            patient_coefs = coef_df[coef_df['patient_id'] == patient]
            if patient_coefs.empty:
                continue
            
            # Get coefficients for a specific alpha (e.g., middle of the range)
            alpha_cols = [col for col in patient_coefs.columns if col.startswith('alpha_')]
            if not alpha_cols:
                continue
            
            # Use middle alpha for analysis
            mid_alpha = alpha_cols[len(alpha_cols)//2]
            coefs = patient_coefs[mid_alpha].values
            
            # Create factor importance plot
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(coefs)), np.abs(coefs))
            plt.xlabel('Factor Number')
            plt.ylabel('|Coefficient Value|')
            plt.title(f'Patient {patient} - Factor Importance (alpha={mid_alpha})')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / f"patient_{patient}_factor_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_loading_based_interpretation(self, exp, model, adata, output_path):
        """Create loading-based factor interpretation."""
        # This would implement the loading-based interpretation
        # Similar to gsea_factor_loading_gene_space_jun11.ipynb
        
        if hasattr(model, 'components_'):
            # For NMF
            components = model.components_
        elif hasattr(model, 'loadings_'):
            # For FA
            components = model.loadings_
        else:
            print("Model does not have components or loadings attribute")
            return
        
        # Create loading heatmap for top factors
        n_top_factors = min(20, components.shape[0])
        n_top_genes = min(50, components.shape[1])
        
        # Get top genes for each factor
        top_genes_per_factor = []
        for i in range(n_top_factors):
            top_indices = np.argsort(np.abs(components[i]))[-n_top_genes:]
            top_genes_per_factor.append(top_indices)
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        heatmap_data = components[:n_top_factors, :]
        
        sns.heatmap(heatmap_data, 
                   xticklabels=False, 
                   yticklabels=[f'Factor_{i+1}' for i in range(n_top_factors)],
                   cmap='RdBu_r', 
                   center=0)
        plt.title('Factor Loading Heatmap (Top Factors)')
        plt.xlabel('Genes')
        plt.ylabel('Factors')
        plt.savefig(output_path / "factor_loading_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_projection_validation_analysis(self, experiment_id: str,
                                              validation_data_path: str,
                                              output_dir: str = None):
        """
        Generate projection validation analysis across timepoints.
        
        Args:
            experiment_id: ID of the experiment to analyze
            validation_data_path: Path to full AnnData with all timepoints
            output_dir: Output directory for results (defaults to experiment's projections dir)
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        
        # Use experiment's projections directory as default
        if output_dir is None:
            output_path = exp.get_path('projections')
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True)
        
        # Load validation data
        print(f"Loading validation data from {validation_data_path}")
        adata_full = sc.read_h5ad(validation_data_path)
        
        # Load trained objects
        preproc_path = exp.get_path('preprocessed_data')
        model_path = exp.get_path('dr_model', dr_method=dr_method, n_components=n_components)
        
        if not preproc_path.exists() or not model_path.exists():
            print("Required files not found")
            return
        
        # Load preprocessed data and model
        adata_mrd = sc.read_h5ad(preproc_path)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load preprocessing objects
        hvg_path = exp.get_path('hvg_list')
        scaler_path = exp.get_path('scaler')
        
        if not hvg_path.exists() or not scaler_path.exists():
            print("Preprocessing objects not found")
            return
        
        with open(hvg_path, 'rb') as f:
            hvg_list = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Extract classification results to get selected factors
        results = self.extract_classification_results(experiment_id)
        if 'coefficients' not in results:
            print("No coefficient results found")
            return
        
        coef_df = results['coefficients']
        
        # Perform projection analysis
        self._perform_projection_analysis(exp, adata_full, adata_mrd, model, 
                                        hvg_list, scaler, coef_df, output_path)
    
    def _perform_projection_analysis(self, exp, adata_full, adata_mrd, model, 
                                   hvg_list, scaler, coef_df, output_path):
        """Perform the actual projection analysis."""
        # This would implement the projection validation logic
        # Similar to QE_projection_MRD_factor_val.ipynb
        
        # Get patient and timepoint information
        patient_col = exp.config.get('classification.patient_column', 'patient')
        timepoint_col = exp.config.get('preprocessing.timepoint_filter', 'timepoint_type')
        
        patients = adata_full.obs[patient_col].unique()
        timepoints = adata_full.obs[timepoint_col].unique()
        
        print(f"Found {len(patients)} patients and {len(timepoints)} timepoints")
        
        # For each patient, project to different timepoints
        for patient in patients:
            print(f"Processing patient {patient}")
            
            # Get selected factors for this patient (simplified - use middle alpha)
            patient_coefs = coef_df[coef_df['patient_id'] == patient]
            if patient_coefs.empty:
                continue
            
            alpha_cols = [col for col in patient_coefs.columns if col.startswith('alpha_')]
            if not alpha_cols:
                continue
            
            # Use middle alpha
            mid_alpha = alpha_cols[len(alpha_cols)//2]
            coefs = patient_coefs[mid_alpha].values
            
            # Get selected factor indices
            selected_factors = np.where(np.abs(coefs) > 0)[0]
            
            if len(selected_factors) == 0:
                continue
            
            # Project to each timepoint
            for timepoint in timepoints:
                # Filter data for this patient and timepoint
                mask = (adata_full.obs[patient_col] == patient) & \
                      (adata_full.obs[timepoint_col] == timepoint)
                
                if not mask.any():
                    continue
                
                adata_subset = adata_full[mask].copy()
                
                # Preprocess and project
                try:
                    projected_data = self._preprocess_and_project(
                        adata_subset, hvg_list, scaler, model
                    )
                    
                    if projected_data is not None:
                        # Create visualization
                        self._create_projection_visualization(
                            projected_data, selected_factors, patient, timepoint, output_path
                        )
                        
                except Exception as e:
                    print(f"Error projecting {patient} {timepoint}: {e}")
    
    def _preprocess_and_project(self, adata, hvg_list, scaler, model):
        """Preprocess data and project using the model."""
        # Subset to available HVGs
        available_hvgs = [hvg for hvg in hvg_list if hvg in adata.var_names]
        if len(available_hvgs) == 0:
            return None
        
        adata_hvg = adata[:, available_hvgs].copy()
        
        # Align with full HVG list
        if adata_hvg.n_vars != len(hvg_list):
            # Create aligned matrix
            X_df = pd.DataFrame(adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X,
                              index=adata_hvg.obs_names,
                              columns=adata_hvg.var_names)
            aligned_X_df = X_df.reindex(columns=hvg_list, fill_value=0)
            X_for_scaling = aligned_X_df.values
        else:
            X_for_scaling = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X
        
        # Scale and project
        X_scaled = scaler.transform(X_for_scaling)
        X_projected = model.transform(X_scaled)
        
        # Create result AnnData
        result_adata = adata.copy()
        result_adata.obsm['X_projected'] = X_projected
        
        return result_adata
    
    def _create_projection_visualization(self, adata, selected_factors, patient, timepoint, output_path):
        """Create visualization for projected data."""
        if len(selected_factors) == 0:
            return
        
        # Select projected factors
        X_selected = adata.obsm['X_projected'][:, selected_factors]
        
        # Create UMAP if enough factors
        if len(selected_factors) > 2:
            adata_temp = adata.copy()
            adata_temp.obsm['X_selected'] = X_selected
            
            sc.pp.neighbors(adata_temp, use_rep='X_selected', n_neighbors=min(15, adata_temp.n_obs-1))
            sc.tl.umap(adata_temp)
            
            sc.pl.umap(adata_temp, color='CN.label', 
                      title=f'Patient {patient} - {timepoint}',
                      show=False,
                      save=f"_patient_{patient}_{timepoint}_umap.png")
            plt.close()
        
        # Create scatter plot for first two factors
        elif len(selected_factors) >= 2:
            plt.figure(figsize=(10, 8))
            colors = {'cancer': 'red', 'normal': 'blue'}
            cell_colors = [colors.get(label, 'grey') for label in adata.obs['CN.label']]
            
            plt.scatter(X_selected[:, 0], X_selected[:, 1], c=cell_colors, alpha=0.7)
            plt.xlabel(f'Factor {selected_factors[0]+1}')
            plt.ylabel(f'Factor {selected_factors[1]+1}')
            plt.title(f'Patient {patient} - {timepoint}')
            plt.legend(['Cancer', 'Normal'])
            plt.savefig(output_path / f"patient_{patient}_{timepoint}_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main function for experiment analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and compare experiments")
    parser.add_argument("--experiments-dir", default="experiments", help="Directory containing experiments")
    parser.add_argument("--experiment-ids", nargs='+', required=True, help="Experiment IDs to analyze")
    parser.add_argument("--output-dir", default="analysis_results", help="Output directory for analysis")
    parser.add_argument("--generate-report", action='store_true', help="Generate HTML report")
    parser.add_argument("--create-plots", action='store_true', help="Create comparison plots")
    parser.add_argument("--export-results", action='store_true', help="Export results for downstream analysis")
    
    args = parser.parse_args()
    
    # Create experiment manager and analyzer
    experiment_manager = ExperimentManager(args.experiments_dir)
    analyzer = ExperimentAnalyzer(experiment_manager)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate comparison table
    comparison_df = analyzer.compare_experiments(args.experiment_ids)
    comparison_path = os.path.join(args.output_dir, "experiment_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Experiment comparison saved to: {comparison_path}")
    
    # Generate performance comparison
    performance_df = analyzer.compare_classification_performance(args.experiment_ids)
    if not performance_df.empty:
        performance_path = os.path.join(args.output_dir, "performance_comparison.csv")
        performance_df.to_csv(performance_path, index=False)
        print(f"Performance comparison saved to: {performance_path}")
    
    # Generate HTML report
    if args.generate_report:
        report_path = os.path.join(args.output_dir, "experiment_report.html")
        analyzer.generate_experiment_summary_report(args.experiment_ids, report_path)
    
    # Create comparison plots
    if args.create_plots:
        plots_dir = os.path.join(args.output_dir, "comparison_plots")
        analyzer.create_performance_comparison_plots(args.experiment_ids, plots_dir)
    
    # Export results for downstream analysis
    if args.export_results:
        for exp_id in args.experiment_ids:
            export_dir = os.path.join(args.output_dir, f"export_{exp_id}")
            analyzer.export_results_for_downstream_analysis(exp_id, export_dir)


if __name__ == "__main__":
    main() 