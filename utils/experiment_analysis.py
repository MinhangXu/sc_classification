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
import gseapy
from anndata import AnnData


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
    
    def generate_lasso_path_2panels_report(self, experiment_id: str, output_dir: str = None):
        """
        Generate a standard set of analysis plots for an experiment.
        
        This includes:
        - Per-patient classification metrics and coefficient paths.
        - For Factor Analysis runs, diagnostic plots for SS Loadings and Communalities.
        """
        print(f"--- Generating Standard Analysis Report for Experiment: {experiment_id} ---")
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

        # For FA experiments, add model diagnostic plots
        if dr_method and dr_method.lower() == 'fa':
            print("\nGenerating FA-specific diagnostic plots...")
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

    def generate_cross_patient_summary_plot(self, experiment_id: str, patient_alphas: Dict[str, float], output_dir: str = None):
        """
        Generates a summary plot comparing performance and coefficients across all
        patients at their hand-picked optimal alpha values.

        Args:
            experiment_id: The ID of the experiment to analyze.
            patient_alphas: A dictionary mapping patient IDs to their chosen alpha value.
                            e.g., {'P01': 0.01, 'P02': 0.05}
            output_dir: Custom output directory. Defaults to the experiment's summary_plots dir.
        """
        print(f"--- Generating Cross-Patient Summary Plot for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        if output_dir is None:
            output_path = exp.get_path('summary_plots')
        else:
            output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        results = self.extract_classification_results(experiment_id)
        if 'metrics' not in results or 'coefficients' not in results:
            print("No classification results found. Cannot generate plot.")
            return

        all_metrics = results['metrics']
        all_coefficients = results['coefficients']

        # Filter the results to get only the data for the selected alpha for each patient
        filtered_metrics = {}
        filtered_coeffs = {}

        for patient_id, selected_alpha in patient_alphas.items():
            if patient_id not in all_coefficients:
                print(f"Warning: Patient {patient_id} not found in experiment results. Skipping.")
                continue

            patient_coefs_df = all_coefficients[patient_id]
            patient_metrics_df = all_metrics[all_metrics['group'] == patient_id]

            # Find the closest alpha in the results
            alpha_cols_numeric = pd.to_numeric(patient_coefs_df.columns.str.replace('alpha_', ''))
            closest_alpha_idx = (np.abs(alpha_cols_numeric - selected_alpha)).argmin()
            closest_alpha_val = alpha_cols_numeric[closest_alpha_idx]
            
            # Store the single series of coefficients
            filtered_coeffs[patient_id] = patient_coefs_df.iloc[:, closest_alpha_idx]
            
            # Store the single row of metrics
            metric_row = patient_metrics_df[np.isclose(patient_metrics_df['alpha'], closest_alpha_val)]
            if not metric_row.empty:
                filtered_metrics[patient_id] = metric_row.iloc[0].to_dict()
            else:
                print(f"Warning: Could not find metrics for patient {patient_id} at alpha ~{closest_alpha_val:.2e}")

        if not filtered_metrics or not filtered_coeffs:
            print("No data available after filtering for selected alphas. Aborting plot generation.")
            return
            
        # Generate the plot using the helper function adapted from your notebook
        fig = self._create_enhanced_summary_plot(filtered_metrics, filtered_coeffs)
        
        if fig:
            n_components = exp.config.get('dimension_reduction.n_components')
            save_path = output_path / f"cross_patient_summary_{n_components}factors.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cross-patient summary plot to {save_path}")
            plt.close(fig)

    def _create_enhanced_summary_plot(self, patient_metrics: Dict, patient_coefficients: Dict):
        """
        Creates an enhanced visualization summarizing results across patients.
        Adapted from FA_result_eval.ipynb.
        """
        from matplotlib import gridspec, patches as mpatches
        
        # Sort patients in numerical order
        patients = sorted(patient_coefficients.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))
        
        metrics_df = pd.DataFrame(index=patients)
        feature_counts = {}
        balanced_accuracies = {}
        trivial_accuracies = {}
        class_balance_info = {}

        for p in patients:
            if p not in patient_metrics: continue
            metrics = patient_metrics[p]
            
            # Feature counts
            non_zero_count = np.sum(np.abs(patient_coefficients[p]) > 1e-10)
            total_count = len(patient_coefficients[p])
            feature_counts[p] = {'non_zero': non_zero_count, 'total': total_count, 'percentage': (non_zero_count / total_count) * 100}

            # Balanced accuracy
            tp = metrics.get('tp', 0); fp = metrics.get('fp', 0); tn = metrics.get('tn', 0); fn = metrics.get('fn', 0)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_accuracies[p] = (sensitivity + specificity) / 2

            # Other metrics
            metrics_df.loc[p, 'ROC AUC'] = metrics.get('roc_auc', 0.5)
            metrics_df.loc[p, 'Overall Accuracy'] = metrics.get('overall_accuracy', 0.5)
            metrics_df.loc[p, 'Cancer Cell Accuracy'] = metrics.get('mal_accuracy', 0.5)
            metrics_df.loc[p, 'Normal Cell Accuracy'] = metrics.get('norm_accuracy', 0.5)
            metrics_df.loc[p, 'Balanced Accuracy'] = balanced_accuracies[p]
            trivial_accuracies[p] = metrics.get('trivial_accuracy', 0.5)

            # Class balance
            majority_num = metrics.get('majority_num', 0); minority_num = metrics.get('minority_num', 0)
            total_cells = majority_num + minority_num
            class_balance_info[p] = {'majority_num': majority_num, 'minority_num': minority_num, 'total_cells': total_cells}

        # Create plot
        fig = plt.figure(figsize=(max(15, len(patients) * 2.5), 14))
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 1.5, 4.5], hspace=0.4)
        ax_features = fig.add_subplot(gs[0]); ax_metrics = fig.add_subplot(gs[1]); ax_heatmap = fig.add_subplot(gs[2])

        # Metrics Panel
        n_metrics = 5; bar_width = 0.15; group_width = bar_width * n_metrics
        group_positions = np.arange(len(patients))
        colors = {'ROC AUC': 'darkgreen', 'Overall Accuracy': 'royalblue', 'Cancer Cell Accuracy': 'firebrick', 'Normal Cell Accuracy': 'darkorange', 'Balanced Accuracy': 'purple'}
        
        for i, (metric, color) in enumerate(colors.items()):
            pos = group_positions - (group_width / 2) + (i * bar_width) + (bar_width / 2)
            ax_metrics.bar(pos, metrics_df[metric], width=bar_width, color=color, label=metric, alpha=0.85)
            for j, value in enumerate(metrics_df[metric]):
                if value > 0.55: ax_metrics.text(pos[j], value - 0.05, f'{value:.2f}', ha='center', va='center', color='white', fontsize=7, fontweight='bold')
        
        for i, p in enumerate(patients):
            trivial_acc = trivial_accuracies[p]
            line_start, line_end = group_positions[i] - group_width / 2, group_positions[i] + group_width / 2
            ax_metrics.plot([line_start, line_end], [trivial_acc, trivial_acc], linestyle='--', color='gray', linewidth=1.5)
            info = class_balance_info[p]
            ax_metrics.text(group_positions[i], trivial_acc + 0.02, f"{info['majority_num']}/{info['total_cells']}={trivial_acc:.2f}", ha='center', va='bottom', color='gray', fontsize=7)

        ax_metrics.set_ylim(0.5, 1.05); ax_metrics.set_ylabel('Metric Value')
        ax_metrics.set_title('Performance Metrics at Optimal Regularization Strength'); ax_metrics.set_xticks(group_positions); ax_metrics.set_xticklabels(patients)
        ax_metrics.legend(loc='lower right', ncol=3, fontsize='small'); ax_metrics.grid(axis='y', linestyle='--', alpha=0.3)

        # Features Panel
        for i, p in enumerate(patients):
            counts = feature_counts[p]
            ax_features.bar(i, counts['non_zero'], color='darkred', width=0.6, alpha=0.7)
            ax_features.bar(i, counts['total'] - counts['non_zero'], bottom=counts['non_zero'], color='lightgray', width=0.6, alpha=0.6)
            ax_features.text(i, counts['non_zero'] / 2, f"{counts['non_zero']}", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        ax_features.set_ylabel('Feature Count'); ax_features.set_title('Active vs. Inactive Features at Selected Alpha')
        ax_features.set_xticks(np.arange(len(patients))); ax_features.set_xticklabels(patients)
        ax_features.legend(handles=[mpatches.Patch(color='darkred', alpha=0.7, label='Active'), mpatches.Patch(color='lightgray', alpha=0.6, label='Inactive')], loc='upper right', fontsize='small')

        # Heatmap Panel
        coef_df = pd.DataFrame({p: patient_coefficients[p] for p in patients}).fillna(0)
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.3])
        sns.heatmap(coef_df.astype(float), cmap="coolwarm", center=0, ax=ax_heatmap, cbar_ax=cbar_ax, vmin=-1.5, vmax=1.5, cbar_kws={'label': 'Factor Coefficient Value'})
        ax_heatmap.set_xticks(np.arange(len(patients)) + 0.5); ax_heatmap.set_xticklabels(patients)
        ax_heatmap.set_title('Factor Coefficients at Optimal Sparsity-Performance Trade-off'); ax_heatmap.set_xlabel('Patient'); ax_heatmap.set_ylabel('Factor')

        plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
        return fig

    def prepare_projection_environment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Loads all necessary artifacts from an experiment for data projection.

        This is a convenience function to gather the DR model, pre-processing scaler,
        HVG list, and classification coefficients.

        Args:
            experiment_id: The ID of the experiment to load artifacts from.

        Returns:
            A dictionary containing the loaded 'model', 'scaler', 'hvg_list', 
            'coefficients', and 'config'. Returns an empty dictionary on failure.
        """
        try:
            exp = self.experiment_manager.load_experiment(experiment_id)
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            
            # Load DR model
            model_path = exp.get_path('dr_model', dr_method=dr_method, n_components=n_components)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler
            scaler_path = exp.get_path('scaler')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load HVG list
            hvg_path = exp.get_path('hvg_list')
            with open(hvg_path, 'rb') as f:
                hvg_list = pickle.load(f)
            
            # Load coefficients
            results = self.extract_classification_results(experiment_id)
            if 'coefficients' not in results:
                raise FileNotFoundError("Could not find or load classification coefficients.")
            
            return {
                "model": model,
                "scaler": scaler,
                "hvg_list": hvg_list,
                "coefficients": results['coefficients'],
                "config": exp.config
            }
        except Exception as e:
            print(f"Error preparing projection environment for experiment {experiment_id}: {e}")
            return {}

    def get_factors_for_alpha(self, patient_coefficients: pd.DataFrame, alpha_value: float) -> np.ndarray:
        """
        Gets the indices of factors with non-zero coefficients for a specific alpha.

        Args:
            patient_coefficients: A DataFrame of coefficients for a single patient,
                                  where columns are alphas and rows are factors.
            alpha_value: The regularization strength (alpha) to select. The function
                         will find the closest matching column.

        Returns:
            A numpy array of integer indices for the selected factors.
        """
        # Find the column name that is closest to the provided alpha_value
        alpha_cols = patient_coefficients.columns.str.replace('alpha_', '').astype(float)
        closest_alpha_col_idx = (np.abs(alpha_cols - alpha_value)).argmin()
        target_col = patient_coefficients.columns[closest_alpha_col_idx]
        
        print(f"Selected alpha {alpha_value}. Closest match found: {alpha_cols[closest_alpha_col_idx]:.2e} (column: {target_col})")
        
        # Get coefficients and find non-zero ones
        coefs = patient_coefficients[target_col]
        selected_factor_indices = np.where(coefs != 0)[0]
        
        return selected_factor_indices

    def project_new_data(self, adata_new: sc.AnnData, model: Any, hvg_list: List[str], scaler: Any) -> sc.AnnData:
        """
        Preprocesses and projects new data using a trained model and scaler.

        Args:
            adata_new: The new AnnData object to project.
            model: The trained dimension reduction model (e.g., FA or NMF).
            hvg_list: The list of highly variable genes the model was trained on.
            scaler: The trained StandardScaler object.

        Returns:
            The AnnData object with projected data in `adata.obsm['X_projected']`, 
            or None if projection is not possible.
        """
        print(f"Projecting {adata_new.n_obs} cells onto {len(hvg_list)} HVGs...")
        
        # Subset to HVGs that are available in the new data
        available_hvgs = [hvg for hvg in hvg_list if hvg in adata_new.var_names]
        if not available_hvgs:
            print("Warning: No matching HVGs found in the new data. Cannot project.")
            return None
        
        adata_hvg = adata_new[:, available_hvgs].copy()
        
        # Align the data matrix to the exact HVG list, filling missing genes with 0
        X_df = pd.DataFrame(
            adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X,
            index=adata_hvg.obs_names,
            columns=adata_hvg.var_names
        )
        aligned_X_df = X_df.reindex(columns=hvg_list, fill_value=0)
        X_for_scaling = aligned_X_df.values
        
        # Standardize data using the pre-fitted scaler
        X_scaled = scaler.transform(X_for_scaling)
        
        # Project data using the pre-fitted model
        X_projected = model.transform(X_scaled)
        
        # Store projected data in the AnnData object
        adata_projected = adata_new.copy()
        adata_projected.obsm['X_projected'] = X_projected
        
        print("Projection complete.")
        return adata_projected
    
    def generate_projection_validation_analysis(self, experiment_id: str,
                                              validation_data_path: str,
                                              output_dir: str = None):
        """
        Generate projection validation analysis across timepoints.
        
        This function serves as a high-level wrapper and example of how to use
        the more granular projection methods like `prepare_projection_environment`
        and `project_new_data`.

        Args:
            experiment_id: ID of the experiment to analyze.
            validation_data_path: Path to full AnnData with all timepoints.
            output_dir: Output directory for results (defaults to experiment's projections dir).
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        # Use experiment's projections directory as default
        if output_dir is None:
            output_path = exp.get_path('projections')
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. Prepare projection environment
        print("--- Preparing projection environment ---")
        proj_env = self.prepare_projection_environment(experiment_id)
        if not proj_env:
            print("Failed to prepare projection environment. Aborting.")
            return
            
        model = proj_env['model']
        scaler = proj_env['scaler']
        hvg_list = proj_env['hvg_list']
        coefficients_dict = proj_env['coefficients']
        config = proj_env['config']

        # Load validation data
        print(f"--- Loading validation data from {validation_data_path} ---")
        adata_full = sc.read_h5ad(validation_data_path)

        # Get relevant column names from config
        patient_col = config.get('classification.patient_column', 'patient')
        timepoint_col = config.get('preprocessing.timepoint_column', 'timepoint_type')
        
        # Iterate through patients present in the classification results
        for patient_id, patient_coefs_df in coefficients_dict.items():
            print(f"\n--- Processing patient: {patient_id} ---")

            # For this example, we automatically select the middle alpha.
            # A user could manually select an alpha and call get_factors_for_alpha.
            alpha_cols_str = patient_coefs_df.columns[patient_coefs_df.columns.str.startswith('alpha_')]
            if len(alpha_cols_str) == 0:
                print(f"  No alpha columns found for patient {patient_id}. Skipping.")
                continue
            
            # For demonstration, pick a middle alpha
            middle_alpha_str = alpha_cols_str[len(alpha_cols_str) // 2]
            middle_alpha_val = float(middle_alpha_str.replace('alpha_', ''))
            
            selected_factors = self.get_factors_for_alpha(patient_coefs_df, middle_alpha_val)
            
            if len(selected_factors) == 0:
                print(f"  No factors selected for alpha ~{middle_alpha_val:.2e}. Skipping projection for this patient.")
                continue
                
            print(f"  Selected {len(selected_factors)} factors: {selected_factors}")

            # Project to each timepoint found in the full data
            available_timepoints = adata_full.obs[timepoint_col].unique()
            for timepoint in available_timepoints:
                print(f"  - Projecting onto timepoint: {timepoint}")
                
                # Filter data for this patient and timepoint
                mask = (adata_full.obs[patient_col] == patient_id) & (adata_full.obs[timepoint_col] == timepoint)
                
                if not mask.any():
                    print(f"    No cells found for patient {patient_id} at timepoint {timepoint}.")
                    continue
                
                adata_subset = adata_full[mask].copy()
                
                # 2. Project the new data
                adata_projected = self.project_new_data(adata_subset, model, hvg_list, scaler)
                
                if adata_projected:
                    # 3. Visualize the projection
                    self._create_projection_visualization(
                        adata_projected, selected_factors, patient_id, timepoint, output_path
                    )
    
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


    def _plot_classification_umap_panels(self, adata, patient_id, cn_label_palette, status_palette, suptitle):
        """Helper function to generate the standard 3-panel UMAP plot."""
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(suptitle, fontsize=16)

        sc.pl.umap(adata, color='predicted.annotation', ax=axes[0], show=False, title=f'Cell Types (Patient {patient_id})')
        sc.pl.umap(adata, color='CN.label', palette=cn_label_palette, ax=axes[1], show=False, title='Ground Truth (Cancer/Normal)')
        sc.pl.umap(adata, color='classification_status', palette=status_palette, ax=axes[2], show=False, title='Classification Status')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _plot_1d_factor_distribution(self, adata, factor_name, cn_label_palette, status_palette, suptitle):
        """Helper function to plot the distribution of a single factor when UMAP isn't possible."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(suptitle, fontsize=16)

        # Split by ground truth
        sc.pl.violin(adata, keys=factor_name, groupby='CN.label', palette=cn_label_palette, ax=axes[0], show=False)
        axes[0].set_title(f'{factor_name} Distribution by Ground Truth')

        # Split by classification status
        sc.pl.violin(adata, keys=factor_name, groupby='classification_status', palette=status_palette, ax=axes[1], show=False)
        axes[1].set_title(f'{factor_name} Distribution by Classification Status')
        axes[1].tick_params(axis='x', labelrotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def generate_classification_umap_report(self, experiment_id: str, 
                                          patient_reg_strength_indices: Dict[str, List[int]], 
                                          static_umap_rep: str = 'X_multivi'):
        """
        Generates a report with UMAPs visualizing classification status for specified patients and regularization strengths.

        For each specified alpha index, this method produces two sets of plots saved into distinct subdirectories:
        
        1.  **Static UMAPs**: A UMAP is calculated once using a static, global representation (e.g., 'X_multivi').
            The classification status from different alpha values is then projected onto this single layout.
            - Output: `analysis/classification_umaps/static_umap/`

        2.  **Dynamic UMAPs**: A new UMAP is calculated for each alpha value, using only the factors that
            the LASSO model selected (i.e., had non-zero coefficients) at that specific regularization strength.
            This visualizes the feature space as the classifier sees it.
            - Output: `analysis/classification_umaps/dynamic_umap/`

        Args:
            experiment_id: The ID of the experiment to analyze.
            patient_reg_strength_indices: A dictionary mapping patient IDs to a list of 1-based 
                                          regularization strength indices to inspect.
                                          e.g., {'P01': [12, 13], 'P02': [15]}
            static_umap_rep: The key in `adata.obsm` to use for the static UMAP plots.
        """
        print(f"--- Generating Classification UMAP Report for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        # Create separate output directories for static and dynamic plots
        base_output_dir = exp.experiment_dir / "analysis" / "classification_umaps"
        static_output_dir = base_output_dir / "static_umap"
        dynamic_output_dir = base_output_dir / "dynamic_umap"
        static_output_dir.mkdir(exist_ok=True, parents=True)
        dynamic_output_dir.mkdir(exist_ok=True, parents=True)

        # --- Load common data ---
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        transformed_adata_path = exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components)
        transformed_adata = sc.read_h5ad(transformed_adata_path)

        # --- Define Color Palettes ---
        positive_class = exp.config.get('preprocessing.positive_class', 'cancer')
        cn_label_palette = {positive_class: 'orange', 'normal': 'lightgreen'}
        status_palette = {
            f'Correct {positive_class.capitalize()}': 'orange', 'Correct Normal': 'lightgreen',
            'Incorrect (False Negative)': 'red', 'Incorrect (False Positive)': 'darkgreen'
        }

        for patient_id, indices in patient_reg_strength_indices.items():
            print(f"\nProcessing patient: {patient_id}")
            
            try:
                correctness_df = pd.read_csv(exp.get_path('patient_classification_correctness', patient_id=patient_id), index_col=0)
                coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
                adata_patient = transformed_adata[correctness_df.index].copy()
            except FileNotFoundError as e:
                print(f"  Warning: Could not load necessary file for patient {patient_id}. Skipping. Error: {e}")
                continue

            # --- 1. Generate Static UMAP plots ---
            if static_umap_rep in adata_patient.obsm:
                print(f"  Generating Static UMAPs based on: '{static_umap_rep}'")
                sc.pp.neighbors(adata_patient, use_rep=static_umap_rep, random_state=42)
                sc.tl.umap(adata_patient, min_dist=0.5, spread=1.0, random_state=42)
                
                for idx in indices:
                    alpha_idx_0based = idx - 1
                    alpha_col_name = correctness_df.columns[alpha_idx_0based]
                    alpha_val = float(alpha_col_name.split('_')[1])
                    self._add_classification_status(adata_patient, correctness_df, alpha_col_name, positive_class)
                    suptitle = f"Patient {patient_id} - Static UMAP ({static_umap_rep}) @ alpha={alpha_val:.2e}"
                    fig = self._plot_classification_umap_panels(adata_patient, patient_id, cn_label_palette, status_palette, suptitle)
                    save_path = static_output_dir / f"{patient_id}_alpha_idx_{idx}_UMAP_{static_umap_rep}.png"
                    fig.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                print(f"  Finished static UMAPs for patient {patient_id}.")

            # --- 2. Generate Dynamic UMAP plots ---
            print(f"  Generating Dynamic UMAPs based on selected '{dr_method}' factors")
            for idx in indices:
                alpha_idx_0based = idx - 1
                coef_col_name = coef_df.columns[alpha_idx_0based]
                alpha_val = float(coef_df.columns[alpha_idx_0based].split('_')[1])
                active_factors_mask = coef_df[coef_col_name] != 0
                n_active_factors = active_factors_mask.sum()
                self._add_classification_status(adata_patient, correctness_df, correctness_df.columns[alpha_idx_0based], positive_class)
                
                if n_active_factors >= 2:
                    adata_patient.obsm['X_selected_factors'] = adata_patient.obsm[f'X_{dr_method}'][:, active_factors_mask]
                    sc.pp.neighbors(adata_patient, use_rep='X_selected_factors', random_state=42)
                    sc.tl.umap(adata_patient, min_dist=0.5, spread=1.0, random_state=42)
                    suptitle = f"Patient {patient_id} - Dynamic UMAP ({n_active_factors} factors) @ alpha={alpha_val:.2e}"
                    fig = self._plot_classification_umap_panels(adata_patient, patient_id, cn_label_palette, status_palette, suptitle)
                    save_path = dynamic_output_dir / f"{patient_id}_alpha_idx_{idx}_UMAP_selected_{dr_method}.png"
                elif n_active_factors == 1:
                    print(f"    Index {idx}: Only 1 active factor. Attempting fallback.")
                    fallback_alpha_idx = alpha_idx_0based - 1
                    if fallback_alpha_idx >= 0:
                        fallback_coef_col = coef_df.columns[fallback_alpha_idx]
                        fallback_coefs = coef_df[fallback_coef_col]
                        factor_to_plot_name = fallback_coefs.abs().idxmax()
                        adata_patient.obs[factor_to_plot_name] = adata_patient.obsm[f'X_{dr_method}'][:, coef_df.index.get_loc(factor_to_plot_name)]
                        suptitle = f"Patient {patient_id} - 1D Dist. of Fallback Factor '{factor_to_plot_name}' from alpha_idx {idx-1}"
                        fig = self._plot_1d_factor_distribution(adata_patient, factor_to_plot_name, cn_label_palette, status_palette, suptitle)
                        save_path = dynamic_output_dir / f"{patient_id}_alpha_idx_{idx}_1D_fallback_{dr_method}.png"
                    else:
                        print(f"    Index {idx}: Cannot fall back, already at the first alpha. Skipping.")
                        continue
                else: 
                    print(f"    Index {idx}: No active factors. Skipping.")
                    continue
                
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            print(f"  Finished dynamic UMAPs for patient {patient_id}.")
        print("\n--- Report generation complete. ---")
        
    def _add_classification_status(self, adata, correctness_df, alpha_col_name, positive_class):
        """Helper to compute and add the 4-category classification status to adata.obs."""
        correctness = correctness_df[alpha_col_name]
        ground_truth_labels = adata.obs['CN.label']
        
        conditions = [
            (correctness) & (ground_truth_labels == positive_class),
            (correctness) & (ground_truth_labels != positive_class),
            (~correctness) & (ground_truth_labels == positive_class),
            (~correctness) & (ground_truth_labels != positive_class)
        ]
        choices = [
            f'Correct {positive_class.capitalize()}',
            'Correct Normal',
            'Incorrect (False Negative)',
            'Incorrect (False Positive)'
        ]
        adata.obs['classification_status'] = np.select(conditions, choices, default='Unknown')
        adata.obs['classification_status'] = adata.obs['classification_status'].astype('category')

    def analyze_discovery_factor_signatures(self, experiment_id: str, patient_id: str,
                                          transition_indices: tuple,
                                          protein_data_path: str):
        """
        Analyzes the multi-omic signatures of factors that are "discovered" during a classification transition.

        A transition is defined by two alpha indices: one where a set of cells is misclassified,
        and one (at a weaker regularization) where they become correctly classified. This function
        identifies the factors responsible for this "flip" and correlates their activity with
        cell types, pseudotime, and protein markers.

        Args:
            experiment_id: The ID of the experiment to analyze.
            patient_id: The patient to focus on.
            transition_indices: A tuple of two 1-based indices (misclassified_idx, classified_idx).
                                e.g., (11, 12) to see what changed between index 11 and 12.
            protein_data_path: Path to the corrected protein expression data in Parquet format.
        """
        print(f"--- Analyzing Discovery Factor Signatures for Patient {patient_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        # --- 1. Load All Necessary Data ---
        try:
            # Load classification and model data
            correctness_df = pd.read_csv(exp.get_path('patient_classification_correctness', patient_id=patient_id), index_col=0)
            coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
            
            # Load the transformed adata to get obs metadata (cell type, pseudotime) and X_fa
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            adata_patient = transformed_adata[correctness_df.index].copy()

            # Load protein data from parquet and merge with adata.obs
            print(f"Loading protein data from: {protein_data_path}")
            protein_df = pd.read_parquet(protein_data_path)
            # Ensure we only use protein data for the cells in our patient anndata
            adata_patient.obs = adata_patient.obs.join(protein_df, how='left')

        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Aborting.")
            return None
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            return None

        # --- 2. Identify the Transition and Discovery Factors ---
        misclassified_idx, classified_idx = transition_indices[0] - 1, transition_indices[1] - 1
        
        # Get factors at each point
        factors_at_misclassified_pt = set(coef_df.index[coef_df.iloc[:, misclassified_idx] != 0])
        factors_at_classified_pt = set(coef_df.index[coef_df.iloc[:, classified_idx] != 0])
        
        discovery_factors = list(factors_at_classified_pt - factors_at_misclassified_pt)
        
        if not discovery_factors:
            print("No new factors were discovered in this transition. Cannot proceed.")
            return None
            
        print(f"Found {len(discovery_factors)} discovery factors: {discovery_factors}")

        # Add factor scores to .obs for easy access
        for factor in discovery_factors:
            factor_idx = coef_df.index.get_loc(factor)
            adata_patient.obs[factor] = adata_patient.obsm['X_fa'][:, factor_idx]

        # --- 3. Run Correlation and Enrichment Analyses ---
        results = {'discovery_factors': discovery_factors}
        protein_names = [c for c in adata_patient.obs.columns if c.startswith('ADT-')]
        
        # A. Continuous Variable Correlation (Proteins & Pseudotime)
        continuous_vars = ['predicted.pseudotime'] + protein_names
        correlation_results = adata_patient.obs[discovery_factors + continuous_vars].corr(method='spearman')
        results['continuous_correlation'] = correlation_results.loc[discovery_factors, continuous_vars]
        
        # B. Categorical Variable Enrichment (Cell Type)
        # Using a simple mean score difference for demonstration; could be replaced with statistical tests
        enrichment_results = adata_patient.obs.groupby('predicted.annotation')[discovery_factors].mean()
        results['cell_type_enrichment'] = enrichment_results
        
        print("\n--- Top 5 Protein Correlations per Discovery Factor ---")
        for factor in discovery_factors:
            top_proteins = results['continuous_correlation'].loc[factor, protein_names].abs().nlargest(5)
            print(f"\nFactor: {factor}")
            print(top_proteins)
            
        print("\n--- Cell Type Enrichment (Mean Factor Score) ---")
        print(results['cell_type_enrichment'])
        
        # (Optional) Here you could add plotting logic to visualize these results
        
        return results

    def generate_unsupervised_fa_report(self, experiment_id: str):
        """
        Generates unsupervised diagnostic plots for a Factor Analysis experiment.

        This includes:
        1. A bar plot of the Sum of Squared Loadings (SS Loadings) per factor,
           indicating the variance explained by each factor.
        2. A histogram of gene communalities, showing how well the variance of
           each gene is captured by the model.

        Args:
            experiment_id: The ID of the Factor Analysis experiment.
        """
        print(f"--- Generating Unsupervised FA Report for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        dr_method = exp.config.get('dimension_reduction.method')
        if dr_method.lower() != 'fa':
            print(f"Warning: This report is for Factor Analysis runs. Experiment is '{dr_method}'. Skipping.")
            return

        # Create a dedicated output directory
        output_dir = exp.experiment_dir / "analysis" / "unsupervised_dr_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)

        # --- Load Data ---
        try:
            n_components = exp.config.get('dimension_reduction.n_components')
            transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            
            if 'FA_loadings' not in transformed_adata.varm:
                print("Error: 'FA_loadings' not found in transformed_adata.varm. Cannot proceed.")
                return
            
            loadings = transformed_adata.varm['FA_loadings']
            # Create a DataFrame for easier handling
            loading_df = pd.DataFrame(loadings, index=transformed_adata.var_names, 
                                      columns=[f'X_fa_{i+1}' for i in range(loadings.shape[1])])

        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Aborting.")
            return

        # --- Calculate SS Loadings and Communality ---
        ss_loadings = (loading_df ** 2).sum(axis=0).sort_values(ascending=False)
        communalities = (loading_df ** 2).sum(axis=1)

        # --- Generate Plots ---
        # 1. SS Loadings Plot
        plt.figure(figsize=(16, 6))
        sns.barplot(x=ss_loadings.index, y=ss_loadings.values, color='skyblue')
        plt.title('Sum of Squared Loadings (Variance Explained) per Factor', fontsize=16)
        plt.ylabel('Sum of Squared Loadings')
        plt.xlabel('Factor')
        plt.xticks(rotation=90, fontsize=8)
        # Show only every 5th tick label to avoid clutter
        ax = plt.gca()
        ticks = ax.get_xticks()
        labels = ax.get_xticklabels()
        ax.set_xticks(ticks[::5])
        ax.set_xticklabels(labels[::5])
        
        plt.tight_layout()
        ss_loadings_path = output_dir / "ss_loadings_per_factor.png"
        plt.savefig(ss_loadings_path, dpi=300)
        plt.close()
        print(f"Saved SS Loadings plot to: {ss_loadings_path}")
        
        # 2. Communality Plot
        plt.figure(figsize=(10, 6))
        sns.histplot(communalities, bins=50, kde=True)
        plt.title('Distribution of Gene Communalities', fontsize=16)
        plt.xlabel("Communality (Proportion of Gene's Variance Explained by Model)")
        plt.ylabel('Number of Genes')
        plt.tight_layout()
        communality_path = output_dir / "gene_communality_distribution.png"
        plt.savefig(communality_path, dpi=300)
        plt.close()
        print(f"Saved Communality plot to: {communality_path}")

    def run_gsea_on_factors(self, experiment_id: str, 
                          gene_sets_path: str = '/home/minhang/mds_project/data/cohort_adata/gene_sets/h.all.v2024.1.Hs.symbols.gmt',
                          rescale_loadings: bool = False,
                          debug_factor: str = None):
        """
        Runs GSEA on all factors, with an option to rescale loadings and debug the input.
        ...
        Args:
            ...
            debug_factor: If set to a factor name (e.g., 'X_fa_95'), saves the ranked gene list for that factor to a CSV for inspection.
        """
        mode = "Rescaled" if rescale_loadings else "Original"
        print(f"--- Running GSEA on All Factors ({mode} Loadings) for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        output_dir_name = f"gsea_on_factors_{mode.lower()}"
        output_dir = exp.experiment_dir / "analysis" / "factor_interpretation" / output_dir_name
        output_dir.mkdir(exist_ok=True, parents=True)

        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
        loading_df = pd.DataFrame(transformed_adata.varm['FA_loadings'], 
                                  index=transformed_adata.var_names, 
                                  columns=[f'X_fa_{i+1}' for i in range(n_components)])

        if rescale_loadings:
            print("  Rescaling factor loadings to common [-1, 1] scale...")
            for factor in loading_df.columns:
                max_abs_loading = loading_df[factor].abs().max()
                if max_abs_loading > 0:
                    loading_df[factor] /= max_abs_loading

        # --- SANITY CHECK DEBUG LOGIC ---
        if debug_factor and debug_factor in loading_df.columns:
            debug_ranked_genes = loading_df[debug_factor].sort_values(ascending=False).dropna()
            debug_path = output_dir / f"DEBUG_RANKED_LIST_{debug_factor}_{mode}.csv"
            debug_ranked_genes.to_csv(debug_path, header=['score'])
            print(f"  >>>> DEBUG: Saved ranked list for {debug_factor} to {debug_path}")
        # --- END DEBUG LOGIC ---

        significant_hits_per_factor = {}
        for factor in loading_df.columns:
            print(f"  Running GSEA on factor: {factor}")
            ranked_genes = loading_df[factor].sort_values(ascending=False).dropna()
            factor_out_dir = output_dir / factor
            
            try:
                prerank_res = gseapy.prerank(
                    rnk=ranked_genes, gene_sets=gene_sets_path, outdir=str(factor_out_dir),
                    min_size=5, max_size=1000, seed=42, no_plot=True
                )
                
                results_csv_path = factor_out_dir / "gseapy.gene_set.prerank.report.csv"
                # Generate summary barplot for this factor
                plot_path = factor_out_dir / f"GSEA_summary_barplot_{factor}.png"
                self._plot_gsea_results(results_csv_path, factor, gene_sets_path, plot_path)

                # Count significant hits for the summary plot
                gsea_results_df = pd.read_csv(results_csv_path)
                significant_hits_per_factor[factor] = (gsea_results_df['FDR q-val'] < 0.05).sum()

            except Exception as e:
                print(f"    ERROR running GSEA for {factor}: {e}")
                significant_hits_per_factor[factor] = 0
        
        # --- Generate final summary plot comparing all factors ---
        if significant_hits_per_factor:
            hits_series = pd.Series(significant_hits_per_factor).sort_index()
            plt.figure(figsize=(20, 7))
            sns.barplot(x=hits_series.index, y=hits_series.values, color='cornflowerblue')
            
            ax = plt.gca()
            num_factors = len(hits_series)
            tick_spacing = max(1, num_factors // 20)
            ax.set_xticks(ax.get_xticks()[::tick_spacing])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
            
            plt.title(f'Number of Significant Pathways (FDR < 0.05) per Factor\n({mode} Loadings)', fontsize=16)
            plt.ylabel('Number of Significant Pathways')
            plt.xlabel('Factor')
            plt.tight_layout()

            summary_plot_path = output_dir / f"summary_significant_hits_{mode.lower()}.png"
            plt.savefig(summary_plot_path, dpi=300)
            plt.close()
            print(f"\nGenerated summary plot of significant hits: {summary_plot_path}")

    def plot_ora_results(self, csv_path: str, output_path: str, top_n: int = 15, pval_threshold: float = 0.05):
        """Creates a summary bar plot from an ORA (Enrichr) results CSV file."""
        try:
            results_df = pd.read_csv(csv_path)
            # Filter for significant terms and sort by Combined Score
            significant = results_df[results_df['Adjusted P-value'] < pval_threshold]
            if significant.empty:
                print(f"  - No significant terms found at p < {pval_threshold} in {os.path.basename(csv_path)}")
                return
            
            plot_data = significant.nlargest(top_n, 'Combined Score')

            plt.figure(figsize=(10, 8))
            ax = sns.barplot(x='Combined Score', y='Term', data=plot_data, color='#3498db')
            
            plt.title(f'ORA Top Pathways for {os.path.basename(csv_path)}', fontsize=16)
            plt.xlabel('Combined Score (Enrichr)', fontsize=12)
            plt.ylabel('Term', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"  - Saved ORA plot to {output_path}")

        except Exception as e:
            print(f"  - ERROR creating ORA plot for {csv_path}: {e}")

    def analyze_communality_extremes(self, experiment_id: str, n_genes: int = 100, 
                                   gene_sets_path: str = '/home/minhang/mds_project/data/cohort_adata/gene_sets/h.all.v2024.1.Hs.symbols.gmt'):
        """
        Visualizes top/bottom genes by communality and runs ORA on these gene lists.

        Args:
            experiment_id: The ID of the experiment.
            n_genes: The number of top and bottom genes to analyze.
            gene_sets_path: Path to the gene sets GMT file for ORA.
        """
        print(f"--- Analyzing Communality Extremes for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        output_dir = exp.experiment_dir / "analysis" / "unsupervised_dr_analysis"
        output_dir.mkdir(exist_ok=True)

        # Load loadings to calculate communality
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
        loadings = transformed_adata.varm['FA_loadings']
        loading_df = pd.DataFrame(loadings, index=transformed_adata.var_names)
        communalities = (loading_df ** 2).sum(axis=1).sort_values(ascending=False)

        # --- Get gene lists ---
        top_genes_list = communalities.head(n_genes)
        bottom_genes_list = communalities.tail(n_genes)

        # --- Visualize ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 12))
        
        # Plot for Top 100 Genes
        sns.barplot(y=top_genes_list.index, x=top_genes_list.values, ax=axes[0], orient='h', color='green')
        axes[0].set_title(f'Top {n_genes} Genes (Best Explained)', fontsize=14)
        axes[0].set_xlabel('Communality Score')
        axes[0].set_ylabel('')  # Remove the y-axis label
        axes[0].tick_params(axis='y', labelsize=7) # Adjust gene name font size
        
        # Plot for Bottom 100 Genes
        sns.barplot(y=bottom_genes_list.index, x=bottom_genes_list.values, ax=axes[1], orient='h', color='red')
        axes[1].set_title(f'Bottom {n_genes} Genes (Worst Explained)', fontsize=14)
        axes[1].set_xlabel('Communality Score')
        axes[1].set_ylabel('') # Remove the y-axis label
        axes[1].tick_params(axis='y', labelsize=7) # Adjust gene name font size
        
        # Set the main title for the figure
        plt.suptitle(f'Gene Communality Score for {experiment_id}', fontsize=18)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for the suptitle
        plot_path = output_dir / "top_bottom_communality_genes.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved communality bar plot to {plot_path}")

        # --- Run ORA ---
        print("\nRunning Over-Representation Analysis (ORA)...")
        if not os.path.exists(gene_sets_path):
            print(f"  ERROR: ORA gene sets file not found at: {gene_sets_path}. Skipping ORA.")
            return
            
        for gene_list, name in [(top_genes_list, "Top"), (bottom_genes_list, "Bottom")]:
            try:
                # UPDATED: Using gene_sets_path for the query
                enr_res = gseapy.enrichr(gene_list=list(gene_list.index), gene_sets=gene_sets_path)
                out_path = output_dir / f"ORA_results_{name}_{n_genes}_genes.csv"
                enr_res.results.to_csv(out_path)
                print(f"  Saved ORA results for {name} {n_genes} genes to {out_path}")
            except Exception as e:
                print(f"  ERROR running ORA for {name} genes: {e}")

    def analyze_classification_transitions(self, experiment_id: str, patient_id: str, indices_to_check: list):
        """
        Analyzes how cell classification status changes across a range of regularization strengths.

        For each step in the alpha path (from stronger to weaker regularization), this function identifies:
        1. The "Discovery Factors" that were activated in that step.
        2. The barcodes of cells that "flipped" from incorrect to correct.
        3. The barcodes of cells that "flipped" from correct to incorrect.
        
        The results are saved to a JSON file in the patient's classification directory.

        Args:
            experiment_id: The ID of the experiment.
            patient_id: The patient to analyze.
            indices_to_check: A list of 1-based alpha indices to analyze transitions between.
                              The list will be sorted to ensure transitions are analyzed correctly.
        """
        print(f"--- Analyzing Classification Transitions for Patient {patient_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        try:
            correctness_df = pd.read_csv(exp.get_path('patient_classification_correctness', patient_id=patient_id), index_col=0)
            coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
        except FileNotFoundError as e:
            print(f"  ERROR: Could not load required file for {patient_id}. Skipping. Details: {e}")
            return

        # Sort indices descending to move from stronger to weaker regularization
        sorted_indices = sorted(indices_to_check, reverse=True)
        all_transitions = []

        for i in range(len(sorted_indices) - 1):
            idx_from = sorted_indices[i]
            idx_to = sorted_indices[i+1]
            
            print(f"  Analyzing transition from index {idx_from} to {idx_to}...")

            # Get 0-based indices for DataFrame access
            idx_from_0based = idx_from - 1
            idx_to_0based = idx_to - 1

            # Get column names
            col_from = coef_df.columns[idx_from_0based]
            col_to = coef_df.columns[idx_to_0based]

            # 1. Find Discovery Factors
            factors_from = set(coef_df.index[coef_df[col_from] != 0])
            factors_to = set(coef_df.index[coef_df[col_to] != 0])
            discovery_factors = sorted(list(factors_to - factors_from))

            # 2. Find Flipping Cells
            correct_from = correctness_df.iloc[:, idx_from_0based]
            correct_to = correctness_df.iloc[:, idx_to_0based]

            flip_to_correct_mask = (~correct_from) & (correct_to)
            flip_to_incorrect_mask = (correct_from) & (~correct_to)
            
            flipped_to_correct_barcodes = correctness_df.index[flip_to_correct_mask].tolist()
            flipped_to_incorrect_barcodes = correctness_df.index[flip_to_incorrect_mask].tolist()

            # 3. Store results
            transition_summary = {
                "transition_indices": [idx_from, idx_to],
                "discovery_factors": discovery_factors,
                "flipped_to_correct_barcodes": flipped_to_correct_barcodes,
                "flipped_to_incorrect_barcodes": flipped_to_incorrect_barcodes
            }
            all_transitions.append(transition_summary)

        # 4. Save results to JSON
        output_path = exp.get_path('patient_classification_transitions', patient_id=patient_id)
        with open(output_path, 'w') as f:
            json.dump(all_transitions, f, indent=2)
            
        print(f"\nSuccessfully saved transition analysis to: {output_path}")

    def plot_classification_transitions(self, experiment_id: str, patient_id: str, static_umap_rep: str = 'X_multivi'):
        """
        Generates a UMAP plot highlighting cells that flip from incorrect to correct at different transitions.
        This plot is generated using only the cells that participated in the classification (i.e., post-downsampling).
        Args:
            experiment_id: The ID of the experiment.
            patient_id: The patient to analyze.
            static_umap_rep: The key in `adata.obsm` to use for the static UMAP layout.
        """
        print(f"--- Plotting Classification Transitions for Patient {patient_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        try:
            transitions = json.load(open(exp.get_path('patient_classification_transitions', patient_id=patient_id)))
            
            # Load correctness_df to get the definitive list of cells used in classification
            correctness_df = pd.read_csv(exp.get_path('patient_classification_correctness', patient_id=patient_id), index_col=0)
            barcodes_for_classification = correctness_df.index
            
            # Load the full transformed adata and subset it to the correct cells
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            adata_patient = adata[barcodes_for_classification].copy()

        except FileNotFoundError as e:
            print(f"  ERROR: Could not load required file. Please run `analyze_classification_transitions` first. Details: {e}")
            return

        # Create the static UMAP layout
        if static_umap_rep not in adata_patient.obsm:
            print(f"  ERROR: Static representation '{static_umap_rep}' not found. Cannot generate plot.")
            return
        sc.pp.neighbors(adata_patient, use_rep=static_umap_rep, random_state=42)
        sc.tl.umap(adata_patient, min_dist=0.5, spread=1.0, random_state=42)

        adata_patient.obs['transition_flip'] = 'Did not flip'
        colors = plt.cm.get_cmap('tab10', len(transitions))
        color_map = {'Did not flip': 'lightgrey'}

        for i, transition in enumerate(transitions):
            barcodes = transition.get('flipped_to_correct_barcodes', [])
            if not barcodes:
                continue
            idx_from, idx_to = transition['transition_indices']
            label = f'Flip at {idx_from} -> {idx_to}'
            adata_patient.obs.loc[barcodes, 'transition_flip'] = label
            color_map[label] = colors(i)

        adata_patient.obs['transition_flip'] = adata_patient.obs['transition_flip'].astype('category')

        plt.figure(figsize=(12, 10))
        # CHANGED: Moved legend to the side
        sc.pl.umap(adata_patient, color='transition_flip', palette=color_map, 
                   title=f'Cells Flipping to Correct by Transition Step (Patient {patient_id})',
                   legend_loc='right margin', size=50)

        output_path = exp.experiment_dir / "analysis" / "classification_umaps"
        plot_path = output_path / f"{patient_id}_transition_flip_map.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved transition flip map to: {plot_path}")

    def report_on_classification_transitions(self, experiment_id: str, patient_id: str):
        """
        For each transition, runs GSEA on discovery factors and reports on flipped cell properties.
        """
        print(f"--- Reporting on Classification Transitions for Patient {patient_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        try:
            transitions = json.load(open(exp.get_path('patient_classification_transitions', patient_id=patient_id)))
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            adata_patient = adata[adata.obs['patient'] == patient_id].copy()
        except FileNotFoundError as e:
            print(f"  ERROR: Could not load required file. Details: {e}")
            return

        # Create a master output directory for this analysis
        base_output_dir = exp.experiment_dir / "analysis" / "transition_analysis" / patient_id
        base_output_dir.mkdir(exist_ok=True, parents=True)

        for transition in transitions:
            idx_from, idx_to = transition['transition_indices']
            discovery_factors = transition.get('discovery_factors', [])
            barcodes = transition.get('flipped_to_correct_barcodes', [])

            print("\n" + "="*50)
            print(f"ANALYZING TRANSITION: Index {idx_from} -> {idx_to}")
            print(f"Discovery Factors: {discovery_factors}")
            print(f"Cells Flipped to Correct: {len(barcodes)}")
            
            if not barcodes:
                print("No cells flipped to correct in this transition. Skipping further analysis.")
                continue

            # --- GSEA on Discovery Factors ---
            if discovery_factors:
                print("\n  Running GSEA on discovery factors...")
                # Note: This calls the existing GSEA function
                self.run_gsea_on_factors(
                    experiment_id=experiment_id,
                    factor_names=discovery_factors
                )
            
            # --- Analyze Properties of Flipped Cells ---
            flipped_cells_adata = adata_patient[barcodes, :]
            
            # Cell Type Composition
            print("\n  Cell Type Composition of Flipped Cells:")
            cell_type_counts = flipped_cells_adata.obs['predicted.annotation'].value_counts()
            print(cell_type_counts)
            
            # Pseudotime Distribution
            print("\n  Pseudotime Distribution of Flipped Cells:")
            pseudotime_stats = flipped_cells_adata.obs['predicted.pseudotime'].describe()
            print(pseudotime_stats)

        print("\n" + "="*50)
        print("Transition reporting complete.")

    def plot_single_classification_transition(self, experiment_id: str, patient_id: str, transition: dict, static_umap_rep: str = 'X_multivi'):
        """
        Generates a dynamic UMAP for a single transition, highlighting flipping cells.

        Args:
            experiment_id: The ID of the experiment.
            patient_id: The patient to analyze.
            transition: A single transition dictionary from the transitions JSON file.
            static_umap_rep: The obsm key to use if a dynamic UMAP cannot be made.
        """
        print(f"--- Plotting single transition for Patient {patient_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        try:
            # Load the necessary data
            correctness_df = pd.read_csv(exp.get_path('patient_classification_correctness', patient_id=patient_id), index_col=0)
            coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
            adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=exp.config.get('dimension_reduction.method'), n_components=exp.config.get('dimension_reduction.n_components')))
            adata_patient = adata[correctness_df.index].copy()
        except FileNotFoundError as e:
            print(f"  ERROR: Could not load required file. Details: {e}")
            return

        idx_from, idx_to = transition['transition_indices']
        
        # --- Create Dynamic UMAP ---
        factors_at_end = set(coef_df.index[coef_df.iloc[:, idx_to - 1] != 0])
        if len(factors_at_end) >= 2:
            active_factors_mask = coef_df.index.isin(factors_at_end)
            adata_patient.obsm['X_selected_factors'] = adata_patient.obsm[f"X_{exp.config.get('dimension_reduction.method')}"][:, active_factors_mask]
            sc.pp.neighbors(adata_patient, use_rep='X_selected_factors', random_state=42)
            sc.tl.umap(adata_patient, min_dist=0.5, spread=1.0, random_state=42)
            umap_type = f"Dynamic ({len(factors_at_end)} factors)"
        else:
            # Fallback to static UMAP if not enough factors
            sc.pp.neighbors(adata_patient, use_rep=static_umap_rep, random_state=42)
            sc.tl.umap(adata_patient, min_dist=0.5, spread=1.0, random_state=42)
            umap_type = f"Static ({static_umap_rep})"

        # --- Color cells by flip type ---
        positive_class = exp.config.get('preprocessing.positive_class', 'cancer')
        adata_patient.obs['flip_type'] = 'Did not flip'
        
        # Color flips to correct
        flipped_correct_barcodes = transition.get('flipped_to_correct_barcodes', [])
        if flipped_correct_barcodes:
            ground_truth = adata_patient.obs.loc[flipped_correct_barcodes, 'CN.label']
            malignant_flips = ground_truth[ground_truth == positive_class].index
            normal_flips = ground_truth[ground_truth != positive_class].index
            adata_patient.obs.loc[malignant_flips, 'flip_type'] = f'{positive_class.capitalize()} -> Correct'
            adata_patient.obs.loc[normal_flips, 'flip_type'] = 'Normal -> Correct'

        color_map = {
            'Did not flip': 'lightgrey',
            f'{positive_class.capitalize()} -> Correct': 'darkred',
            'Normal -> Correct': 'darkgreen'
        }
        
        plt.figure(figsize=(12, 10))
        sc.pl.umap(adata_patient, color='flip_type', palette=color_map,
                   title=f'Cells Flipping at Reg Strength {idx_from} -> {idx_to} (Patient {patient_id})\nUMAP based on {umap_type}',
                   legend_loc='right margin', size=50)

        output_path = exp.experiment_dir / "analysis" / "classification_umaps" / "transition_plots"
        output_path.mkdir(exist_ok=True, parents=True)
        plot_path = output_path / f"{patient_id}_transition_{idx_from}_to_{idx_to}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved single transition plot to {plot_path}")

    def plot_factor_loading_distributions(self, experiment_id: str, factors_to_compare: list = None, show_fliers=False):
        """
        Generates a boxplot comparing the distribution of gene loading values for specified factors.

        If no factors are specified, it plots all factors to show the overall trend.

        Args:
            experiment_id: The ID of the experiment.
            factors_to_compare: A list of factor names to compare. If None, all factors are used.
            show_fliers: Whether to show the outlier points on the boxplot.
        """
        print(f"--- Plotting Factor Loading Distributions for {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        output_dir = exp.experiment_dir / "analysis" / "unsupervised_dr_analysis"
        output_dir.mkdir(exist_ok=True)

        # Load loadings
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
        all_factors = [f'X_fa_{i+1}' for i in range(n_components)]
        loading_df = pd.DataFrame(transformed_adata.varm['FA_loadings'], index=transformed_adata.var_names, columns=all_factors)
        
        if factors_to_compare is None:
            factors_to_compare = all_factors
        
        plot_data = loading_df[factors_to_compare].melt(var_name='Factor', value_name='Loading Value')

        plt.figure(figsize=(20, 7))
        sns.boxplot(x='Factor', y='Loading Value', data=plot_data, showfliers=show_fliers)
        
        # --- Clean up X-axis ---
        num_factors = len(factors_to_compare)
        ax = plt.gca()
        if num_factors > 20: # Only thin out labels if there are many factors
            tick_spacing = max(1, num_factors // 20) # Aim for ~20 ticks
            ax.set_xticks(ax.get_xticks()[::tick_spacing])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

        plt.title(f'Distribution of Gene Loadings for {num_factors} Factors', fontsize=16)
        plt.axhline(0, color='red', linestyle='--', lw=1)
        plt.tight_layout()

        plot_path = output_dir / "factor_loading_distributions.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved factor loading distribution plot to: {plot_path}")

    def run_gsea_on_predictive_loading(self, experiment_id: str, patient_id: str, alpha_index: int,
                                     gene_sets_path: str = '/home/minhang/mds_project/data/cohort_adata/gene_sets/h.all.v2024.1.Hs.symbols.gmt'):
        """
        Runs GSEA on a "Predictive Loading" vector for a specific patient at a specific alpha.

        The Predictive Loading is a supervised gene ranking calculated by taking the dot product
        of the factor loadings matrix (L) and the LASSO coefficient vector (β) for a given alpha.
        PredictiveLoading = L @ β

        This boosts the signal for genes in low-variance but highly predictive factors.

        Args:
            experiment_id: The ID of the experiment.
            patient_id: The patient to analyze.
            alpha_index: The 1-based index of the regularization strength to use.
            gene_sets_path: Path to the gene sets GMT file.
        """
        print(f"--- Running GSEA on Predictive Loading for Patient {patient_id} at alpha index {alpha_index} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        # Create a dedicated output directory
        output_dir = exp.experiment_dir / "analysis" / "supervised_gsea" / patient_id
        output_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Load unsupervised factor loadings
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            loading_df = pd.DataFrame(transformed_adata.varm['FA_loadings'], 
                                      index=transformed_adata.var_names, 
                                      columns=[f'X_fa_{i+1}' for i in range(n_components)])
            
            # Load supervised LASSO coefficients
            coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
            coef_col_name = coef_df.columns[alpha_index - 1]
            lasso_coeffs = coef_df[coef_col_name]

        except FileNotFoundError as e:
            print(f"  ERROR: Could not load required file. Details: {e}")
            return
        
        # --- Calculate Predictive Loading ---
        print("  Calculating predictive loading vector...")
        predictive_loadings = loading_df.dot(lasso_coeffs).sort_values(ascending=False)

        # --- Run GSEA on the new ranked list ---
        print("  Running GSEA...")
        gsea_out_dir = output_dir / f"alpha_idx_{alpha_index}"
        try:
            prerank_res = gseapy.prerank(
                rnk=predictive_loadings,
                gene_sets=gene_sets_path,
                outdir=str(gsea_out_dir),
                min_size=5,
                max_size=1000,
                seed=42,
                no_plot=True
            )
            
            # Generate our custom summary plot
            results_csv = gsea_out_dir / "gseapy.gene_set.prerank.report.csv"
            plot_path = gsea_out_dir / "supervised_GSEA_summary_barplot.png"
            self._plot_gsea_results(results_csv, f"Predictive Loading (Patient {patient_id})", gene_sets_path, plot_path)
            print(f"  Successfully ran supervised GSEA. Results in {gsea_out_dir}")

        except Exception as e:
            print(f"  ERROR running supervised GSEA: {e}")

    def _plot_gsea_results(self, csv_path: str, factor_name: str, gene_set: str, output_path: str, top_n: int = 10):
        """Creates a summary bar plot from a GSEA results CSV file."""
        try:
            results_df = pd.read_csv(csv_path)
            if results_df.empty:
                print(f"    - GSEA results file is empty for {factor_name}. Skipping summary plot.")
                return

            # --- Select top N pathways (balanced between up and down) ---
            top_up = results_df[results_df['NES'] > 0].nsmallest(top_n, 'FDR q-val')
            top_down = results_df[results_df['NES'] < 0].nsmallest(top_n, 'FDR q-val')
            
            combined = pd.concat([top_up, top_down]).nsmallest(top_n, 'FDR q-val').sort_values('NES', ascending=False)
            
            if combined.empty:
                print(f"    - No significant pathways to plot for {factor_name}.")
                return

            plt.figure(figsize=(12, 8))
            colors = ['#d62728' if x > 0 else '#1f77b4' for x in combined['NES']]
            ax = sns.barplot(x='NES', y='Term', data=combined, palette=colors, orient='h')
            
            for i in range(len(combined)):
                row = combined.iloc[i]
                bar = ax.patches[i]
                q_val = row['FDR q-val']
                asterisks = ''
                if q_val < 0.001: asterisks = '***'
                elif q_val < 0.01: asterisks = '**'
                elif q_val < 0.05: asterisks = '*'
                
                if not asterisks:
                    continue

                x_pos = bar.get_width()
                y_pos = bar.get_y() + bar.get_height() / 2
                
                ha = 'left' if row['NES'] > 0 else 'right'
                offset = 0.02 if row['NES'] > 0 else -0.02
                
                ax.text(x_pos + offset, y_pos, asterisks, va='center', ha=ha, fontsize=14, color='black')

            plt.title(f'GSEA Top Pathways for {factor_name}\n(Gene Set: {os.path.basename(gene_set)})', fontsize=16)
            plt.xlabel('Normalized Enrichment Score (NES)', fontsize=12)
            plt.ylabel('Term', fontsize=12)
            plt.axvline(0, color='black', linewidth=0.8)
            
            legend_text = 'Significance:\n***: FDR q-val < 0.001\n **: FDR q-val < 0.01\n  *: FDR q-val < 0.05'
            ax.text(0.97, 0.03, legend_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='ghostwhite', alpha=0.7))

            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300)
            plt.close()

        except KeyError as e:
            print(f"    - ERROR creating summary plot for {csv_path}: A required column is missing. {e}")
        except Exception as e:
            print(f"    - ERROR creating summary plot for {csv_path}: {e}")

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