import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import scanpy as sc
import pickle
from .experiment_manager import ExperimentManager
import gseapy
from anndata import AnnData
import logging
import anndata
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_auc_score


class ExperimentAnalyzer:
    """Analyzer for comparing and summarizing experiments."""
    
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.logger = logging.getLogger(__name__)

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
            # Concatenate all metrics first
            raw_metrics_df = pd.concat(all_metrics_list, ignore_index=True)

            # --- Handle Pan-Patient Aggregation ---
            if len(patients) == 1 and patients[0] == 'all_patients':
                self.logger.info("Pan-patient experiment detected. Aggregating metrics...")
                all_patients_dir = exp.experiment_dir / "models" / "classification" / "all_patients"
                is_cv_run = 'roc_auc_mean' in raw_metrics_df.columns

                if is_cv_run:
                    agg_path = all_patients_dir / "all_patients_aggregated_cv_metrics.csv"
                    self.logger.info(f"CV run detected. Aggregating metrics across patients and saving to {agg_path}")
                    
                    metrics_to_agg = [
                        'overall_accuracy_mean', 'overall_accuracy_std',
                        'mal_accuracy_mean', 'mal_accuracy_std',
                        'norm_accuracy_mean', 'norm_accuracy_std',
                        'roc_auc_mean', 'roc_auc_std'
                    ]
                    agg_metrics_df = raw_metrics_df.groupby('alpha')[metrics_to_agg].agg('mean').reset_index()
                    agg_metrics_df['group'] = 'all_patients'
                    agg_metrics_df.to_csv(agg_path, index=False)
                    results['metrics'] = agg_metrics_df
                else:
                    agg_path = all_patients_dir / "all_patients_metrics.csv"
                    self.logger.info(f"Non-CV run detected. Aggregating metrics and saving to {agg_path}")
                    agg_metrics_df = self._calculate_aggregated_metrics(raw_metrics_df)
                    agg_metrics_df.to_csv(agg_path, index=False)
                    results['metrics'] = agg_metrics_df
            else:
                # For per-patient runs, just use the concatenated data as is.
                results['metrics'] = raw_metrics_df

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

    def _plot_metrics_and_coefficients(self, metrics_df: pd.DataFrame, coefs_df: pd.DataFrame, 
                                       patient_id_for_filtering: str, n_factors_info: str, best_alpha: Optional[float] = None) -> plt.Figure:
        """
        Internal helper to create the detailed 2-panel stacked plot for classification metrics.
        This is the restored original plotting function.
        """
        metrics_for_patient = metrics_df[metrics_df['group'] == patient_id_for_filtering].copy()
        if metrics_for_patient.empty:
            self.logger.warning(f"No metrics found for patient {patient_id_for_filtering} in the provided DataFrame.")
            return None

        is_cv_run = 'roc_auc_std' in metrics_for_patient.columns
        
        # Define the color palette
        color_palette = {
            'Normal Cells': 'darkgreen',
            'Cancer Cells': 'darkblue',
            'Overall Accuracy': 'skyblue',
            'ROC AUC': 'purple',
            'Surviving Features': 'gold'
        }
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # --- Top Panel: Metrics vs. Regularization ---
        ax1 = axes[0]
        alphas = metrics_for_patient['alpha']
        log_alphas = np.log10(alphas)
        
        # Plot main metrics
        if is_cv_run:
            # For CV runs, metrics have _mean and _std suffixes. We plot error bands for all.
            # Plot Overall Accuracy with std dev band
            ax1.plot(log_alphas, metrics_for_patient['overall_accuracy_mean'], 'o-', color=color_palette['Overall Accuracy'], label='Overall Accuracy')
            ax1.fill_between(log_alphas,
                             metrics_for_patient['overall_accuracy_mean'] - metrics_for_patient['overall_accuracy_std'],
                             metrics_for_patient['overall_accuracy_mean'] + metrics_for_patient['overall_accuracy_std'],
                             color=color_palette['Overall Accuracy'], alpha=0.2)
            # Plot Cancer Cell Accuracy with std dev band
            ax1.plot(log_alphas, metrics_for_patient['mal_accuracy_mean'], 'o-', color=color_palette['Cancer Cells'], label='Cancer Cell Accuracy')
            ax1.fill_between(log_alphas,
                             metrics_for_patient['mal_accuracy_mean'] - metrics_for_patient['mal_accuracy_std'],
                             metrics_for_patient['mal_accuracy_mean'] + metrics_for_patient['mal_accuracy_std'],
                             color=color_palette['Cancer Cells'], alpha=0.2)
            # Plot Normal Cell Accuracy with std dev band
            ax1.plot(log_alphas, metrics_for_patient['norm_accuracy_mean'], 'o-', color=color_palette['Normal Cells'], label='Normal Cell Accuracy')
            ax1.fill_between(log_alphas,
                             metrics_for_patient['norm_accuracy_mean'] - metrics_for_patient['norm_accuracy_std'],
                             metrics_for_patient['norm_accuracy_mean'] + metrics_for_patient['norm_accuracy_std'],
                             color=color_palette['Normal Cells'], alpha=0.2)
            # Plot ROC AUC with std dev band
            ax1.plot(log_alphas, metrics_for_patient['roc_auc_mean'], 'o-', color=color_palette['ROC AUC'], label='ROC AUC (Mean)')
            ax1.fill_between(log_alphas, 
                             metrics_for_patient['roc_auc_mean'] - metrics_for_patient['roc_auc_std'],
                             metrics_for_patient['roc_auc_mean'] + metrics_for_patient['roc_auc_std'],
                             color=color_palette['ROC AUC'], alpha=0.2, label='ROC AUC (CV std. dev.)')
        else:
            # Add cell counts to legend labels
            majority_num = metrics_for_patient['majority_num'].iloc[0]
            minority_num = metrics_for_patient['minority_num'].iloc[0]
            
            # --- Create custom legend handles ---
            # Main metric lines
            line1, = ax1.plot(log_alphas, metrics_for_patient['norm_accuracy'], 'o-', color=color_palette['Normal Cells'], label=f'Normal Cell Accuracy')
            line2, = ax1.plot(log_alphas, metrics_for_patient['mal_accuracy'], 's-', color=color_palette['Cancer Cells'], label=f'Cancer Cell Accuracy')
            line3, = ax1.plot(log_alphas, metrics_for_patient['overall_accuracy'], 'o-', color=color_palette['Overall Accuracy'], label='Overall Accuracy')
            line4, = ax1.plot(log_alphas, metrics_for_patient['roc_auc'], '^-', color=color_palette['ROC AUC'], label='ROC AUC')
            
            # Trivial accuracy line
            trivial_acc = metrics_for_patient['trivial_accuracy'].iloc[0]
            line5 = ax1.axhline(y=trivial_acc, color='r', linestyle='--', label=f"Trivial (Majority) Acc = {trivial_acc:.3f}")
            
            # Dummy plots for cell counts in legend
            dummy_norm = plt.Line2D([0], [0], marker='o', color='w', label=f'Normal Cells (Maj.): {majority_num}',
                                  markerfacecolor=color_palette['Normal Cells'], markersize=10)
            dummy_mal = plt.Line2D([0], [0], marker='s', color='w', label=f'Cancer Cells (Min.): {minority_num}',
                                 markerfacecolor=color_palette['Cancer Cells'], markersize=10)

        ax1.set_xlabel('')
        ax1.set_ylabel('Accuracy / AUC')
        
        # Secondary y-axis for feature survival
        ax2 = ax1.twinx()
        survival_series = 100 * (coefs_df != 0).sum(axis=0) / len(coefs_df)
        line_survival, = ax2.plot(log_alphas, survival_series, 'o-', color=color_palette['Surviving Features'], label='Surviving Features (%)')
        ax2.set_ylabel('Surviving Features (%)', color='black')

        # --- Combined Legend ---
        if not is_cv_run:
            handles = [dummy_norm, dummy_mal, line3, line2, line1, line5, line4, line_survival]
            ax1.legend(handles=handles, bbox_to_anchor=(1.15, 1), loc='upper left', borderaxespad=0.)
        else:
            # Standard legend for CV case
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.15, 1), loc='upper left')

        title_prefix = "Cross-Validated " if is_cv_run else ""
        ax1.set_title(f"{title_prefix}Classification Metrics vs. Regularization (Patient: {patient_id_for_filtering})\n{n_factors_info}")

        # --- Bottom Panel: Coefficient Paths of Last Surviving Factors (Updated Logic) ---
        ax3 = axes[1]
        
        # New logic: Find the last 10 factors to be regularized to zero
        last_surviving_factors = set()
        # Iterate backwards through regularization strengths (from strongest to weakest)
        for alpha_col in reversed(coefs_df.columns):
            non_zero_at_this_alpha = coefs_df.index[coefs_df[alpha_col] != 0]
            
            # Add these factors to our set
            for factor in non_zero_at_this_alpha:
                last_surviving_factors.add(factor)
                if len(last_surviving_factors) >= 10:
                    break
            
            if len(last_surviving_factors) >= 10:
                # If we've found 10 or more, take the 10 with largest magnitude at this alpha
                last_survivors_at_alpha = coefs_df.loc[list(last_surviving_factors), alpha_col].abs().nlargest(10).index
                factors_to_plot = list(last_survivors_at_alpha)
                break
        else:
            # If the loop finishes without finding 10 factors (unlikely), plot what we have
            factors_to_plot = list(last_surviving_factors)
        
        if factors_to_plot:
            coefs_df_to_plot = coefs_df.loc[factors_to_plot]
            for factor in coefs_df_to_plot.index:
                ax3.plot(log_alphas, coefs_df_to_plot.loc[factor], label=factor)
        
        ax3.axhline(0, color='black', linestyle='dotted')
        ax3.set_xlabel('log10(λ) (λ = Lasso Regularization Strength)')
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('Top-10 Most Robust Factor Coefficient Paths')
        ax3.legend(title="Factor", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def generate_lasso_path_2panels_report(self, experiment_id: str, save_format: str = 'png', dpi: int = 300):
        """
        Generates and saves a 2-panel plot for each patient/group in an experiment.
        This function intelligently handles both per-patient and pan-patient runs.
        """
        self.logger.info(f"--- Generating 2-Panel LASSO Reports for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        classification_dir = exp.experiment_dir / "models" / "classification"
        if not classification_dir.exists():
            self.logger.warning(f"Classification directory not found for experiment {experiment_id}")
            return

        patient_ids = [p.name for p in classification_dir.iterdir() if p.is_dir()]
        n_factors_info = f"{exp.config.get('dimension_reduction.n_components')} {exp.config.get('dimension_reduction.method').upper()} Factors"
        
        # Handle Pan-Patient case intelligently
        if patient_ids == ['all_patients']:
            self.logger.info("  Detected pan-patient run. Aggregating metrics for plotting...")
            
            all_patients_dir = classification_dir / "all_patients"
            metrics_path = all_patients_dir / "metrics.csv"
            agg_metrics_path = all_patients_dir / "all_patients_metrics.csv"
            
            if not metrics_path.exists():
                self.logger.error(f"  Could not find metrics.csv for 'all_patients' group.")
                return

            per_patient_metrics_df = pd.read_csv(metrics_path)
            agg_metrics_df = self._calculate_aggregated_metrics(per_patient_metrics_df)
            agg_metrics_df.to_csv(agg_metrics_path, index=False)
            self.logger.info(f"  Saved aggregated pan-patient metrics to {agg_metrics_path}")
            
            # Now, plot using this single aggregated file
            metrics_df = pd.read_csv(agg_metrics_path)
            coef_df = pd.read_csv(all_patients_dir / "coefficients.csv", index_col=0)
            
            fig = self._plot_metrics_and_coefficients(metrics_df, coef_df, 'all_patients', n_factors_info)
            if fig:
                save_path = exp.get_path('summary_plots') / f"pan_patient_aggregated_metrics.{save_format}"
                fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"  Saved aggregated plot to {save_path}")

        else:
            # Original behavior for per-patient runs
            self.logger.info("Plotting summary for each patient...")
            metrics_path = classification_dir / patient_ids[0] / "metrics.csv" # Path logic might need adjustment if metrics are split
            
            try:
                # Assuming metrics for all patients are in one file, which might not be the case.
                # Let's load them per patient.
                for patient_id in patient_ids:
                    self.logger.info(f"  Generating plot for patient: {patient_id}")
                    metrics_df = pd.read_csv(exp.get_path('patient_metrics', patient_id=patient_id))
                    coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
                    
                    fig = self._plot_metrics_and_coefficients(metrics_df, coef_df, patient_id, n_factors_info)
                    if fig:
                        save_path = exp.get_path('summary_plots') / f"patient_{patient_id}_metrics_and_coefficients.{save_format}"
                        fig.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
                        plt.close(fig)
                        self.logger.info(f"  Saved plot to {save_path}")

            except FileNotFoundError:
                 self.logger.error(f"Could not find metrics/coefficients for patient {patient_id}. Ensure all per-patient files exist.")

    def _calculate_aggregated_metrics(self, per_patient_metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates per-patient metrics from a pan-patient run into a single pan-patient metric DataFrame.
        """
        is_cv_run = any('_mean' in col for col in per_patient_metrics_df.columns)

        if is_cv_run:
            # --- Robust aggregation for CV runs ---
            # Define all possible columns we want to aggregate
            possible_agg_cols = {
                'overall_accuracy_mean': 'mean',
                'mal_accuracy_mean': 'mean',
                'norm_accuracy_mean': 'mean',
                'roc_auc_mean': 'mean',
                'roc_auc_std': lambda x: np.sqrt(np.mean(x**2)), # Pool std dev
                'trivial_accuracy_mean': 'first' # Trivial accuracy should be constant
            }
            # Find which of these columns actually exist in the dataframe
            agg_cols_to_use = {k: v for k, v in possible_agg_cols.items() if k in per_patient_metrics_df.columns}
            
            if not agg_cols_to_use:
                self.logger.error("  No recognizable CV metric columns found for aggregation.")
                return pd.DataFrame()

            # For CV runs, we focus on aggregating the mean and std of key metrics
            agg_metrics = per_patient_metrics_df.groupby('alpha').agg(agg_cols_to_use).reset_index()

            # Rename for consistency with the plotting function
            rename_dict = {
                'roc_auc_mean': 'roc_auc',
                'overall_accuracy_mean': 'overall_accuracy',
                'mal_accuracy_mean': 'mal_accuracy',
                'norm_accuracy_mean': 'norm_accuracy',
                'trivial_accuracy_mean': 'trivial_accuracy'
            }
            # Only rename columns that exist
            agg_metrics.rename(columns={k: v for k, v in rename_dict.items() if k in agg_metrics.columns}, inplace=True)

        else:
            # For non-CV runs, we sum fundamental counts and recalculate all metrics
            agg_metrics = per_patient_metrics_df.groupby('alpha')[['tp', 'fp', 'tn', 'fn', 'majority_num', 'minority_num']].sum().reset_index()
            with np.errstate(divide='ignore', invalid='ignore'):
                # --- Recalculate ALL metrics from aggregated counts ---
                agg_metrics['overall_accuracy'] = (agg_metrics['tp'] + agg_metrics['tn']) / \
                                                  (agg_metrics['tp'] + agg_metrics['tn'] + agg_metrics['fp'] + agg_metrics['fn'])
                
                # Malignant (Positive Class) Metrics
                agg_metrics['mal_accuracy'] = agg_metrics['tp'] / (agg_metrics['tp'] + agg_metrics['fn']) # Recall
                agg_metrics['mal_precision'] = agg_metrics['tp'] / (agg_metrics['tp'] + agg_metrics['fp'])
                agg_metrics['mal_recall'] = agg_metrics['mal_accuracy'] # Recall is the same as accuracy for this class
                agg_metrics['mal_f1'] = 2 * (agg_metrics['mal_precision'] * agg_metrics['mal_recall']) / \
                                        (agg_metrics['mal_precision'] + agg_metrics['mal_recall'])

                # Normal (Negative Class) Metrics
                agg_metrics['norm_accuracy'] = agg_metrics['tn'] / (agg_metrics['tn'] + agg_metrics['fp']) # Specificity
                agg_metrics['norm_precision'] = agg_metrics['tn'] / (agg_metrics['tn'] + agg_metrics['fn'])
                agg_metrics['norm_recall'] = agg_metrics['norm_accuracy'] # Recall for the negative class is specificity
                agg_metrics['norm_f1'] = 2 * (agg_metrics['norm_precision'] * agg_metrics['norm_recall']) / \
                                       (agg_metrics['norm_precision'] + agg_metrics['norm_recall'])
                
                agg_metrics['trivial_accuracy'] = agg_metrics['majority_num'] / (agg_metrics['majority_num'] + agg_metrics['minority_num'])
                
                # ROC AUC cannot be perfectly recalculated from TP/FP rates, so we average the patient-level AUCs
                agg_metrics['roc_auc'] = per_patient_metrics_df.groupby('alpha')['roc_auc'].mean().values

        agg_metrics['group'] = 'all_patients'
        # Fill any potential NaN values that result from division by zero
        agg_metrics.fillna(0, inplace=True)
        return agg_metrics

    def generate_lasso_path_2panels_report_internal_CV(self, experiment_id: str, output_dir: str = None, save_format: str = 'png', dpi: int = 300):
        """
        Generate a standard set of analysis plots for an experiment that used internal CV.
        This version plots mean metrics with standard deviation bands.
        """
        print(f"--- Generating Internal CV Analysis Report for Experiment: {experiment_id} ---")
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
        
        # The extract_classification_results function now correctly aggregates pan-patient CV runs,
        # so we can use a single, unified loop for plotting.
        for patient_id, coefs_df in coefficients_dict.items():
            print(f"Generating plot for patient: {patient_id}")
            
            fig = self._plot_metrics_and_coefficients(
                metrics_df=metrics_df,
                coefs_df=coefs_df,
                patient_id_for_filtering=patient_id,
                n_factors_info=f"{n_components} {dr_method.upper()} Factors"
            )

            if fig:
                # Create a filename that works for both pan-patient and per-patient
                if patient_id == 'all_patients':
                    filename = f"pan_patient_CV_metrics_and_coefficients.{save_format}"
                else:
                    filename = f"patient_{patient_id}_CV_metrics_and_coefficients.{save_format}"
                
                save_filepath = output_path / filename
                fig.savefig(save_filepath, format=save_format, dpi=dpi, bbox_inches='tight')
                print(f"  Saved plot to {save_filepath}")
                plt.close(fig)

    def _create_fa_specific_plots(self, exp, n_factors, output_path):
        """Creates FA-specific diagnostic plots."""
        dr_results_dir = exp.experiment_dir / "models" / f"fa_{n_factors}"
        if not dr_results_dir.exists():
            print(f"  WARNING: FA results directory not found for {n_factors} factors. Skipping FA-specific plots.")
            return

        # Construct path to a potential summary file from the experiment directory
        fa_summary_file_path = dr_results_dir / "model_summary.txt"
        self._plot_factor_ss_loadings(fa_summary_file_path, n_factors, output_path)

        # Construct path to a potential communalities file
        communalities_csv_path = dr_results_dir / "gene_communalities.csv"
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

    def plot_accuracy_vs_n_factors_summary(self, experiment_id: str, output_dir: str = None, metric: str = 'overall_accuracy'):
        """
        Generates a summary plot comparing classification accuracy against the number of selected factors for all patients.

        Args:
            experiment_id: The ID of the experiment to analyze.
            output_dir: Custom output directory. Defaults to the experiment's summary_plots dir.
            metric: The accuracy metric to plot. e.g., 'overall_accuracy', 'mal_accuracy', 'roc_auc'.
                    The function will look for '{metric}_mean' and fallback to '{metric}'.
        """
        print(f"--- Generating Accuracy vs. Number of Factors Summary for Experiment: {experiment_id} ---")
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

        all_metrics_df = results['metrics']
        all_coefficients = results['coefficients']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        patients = sorted(all_coefficients.keys())
        colors = plt.cm.get_cmap('tab10', len(patients))

        for i, patient_id in enumerate(patients):
            patient_metrics = all_metrics_df[all_metrics_df['group'] == patient_id].copy()
            patient_coefs = all_coefficients[patient_id].copy()
            
            if patient_metrics.empty or patient_coefs.empty:
                print(f"Warning: Missing metrics or coefficients for patient {patient_id}. Skipping.")
                continue

            # --- Dynamically select metric columns based on availability ---
            is_cv_data = f"{metric}_mean" in patient_metrics.columns
            metric_col = f"{metric}_mean" if is_cv_data else metric
            y_label_suffix = " (CV Mean)" if is_cv_data else ""

            if metric_col not in patient_metrics.columns:
                print(f"Warning: Metric '{metric_col}' not found for patient {patient_id}. Skipping.")
                continue
            # --- End dynamic selection ---

            # Ensure metrics are sorted by alpha to make a proper line plot
            patient_metrics.sort_values('alpha', inplace=True)
            
            # Align number of features with metrics
            try:
                # Handle cases where columns might not be in 'alpha_...' format
                try:
                    coef_alphas = np.array([float(col.split('_')[1]) for col in patient_coefs.columns])
                except (ValueError, IndexError):
                    coef_alphas = patient_coefs.columns.astype(float)
                    
                metric_alphas = patient_metrics['alpha'].values
                
                surviving_features_count = (patient_coefs != 0).sum(axis=0).values
                
                # Interpolate to align feature counts with metric alphas
                aligned_features = np.interp(metric_alphas, coef_alphas, surviving_features_count)
            except Exception as e:
                print(f"Warning: Could not align features for patient {patient_id}. Error: {e}")
                continue

            metric_values = patient_metrics[metric_col].values
            
            # Sort by number of features for a clean plot
            sort_indices = np.argsort(aligned_features)
            ax.plot(aligned_features[sort_indices], metric_values[sort_indices], 'o-', label=patient_id, color=colors(i), alpha=0.8)

        ax.set_xlabel("Number of Selected Factors", fontsize=12)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}{y_label_suffix}", fontsize=12)
        ax.set_title(f"Model Performance vs. Number of Factors ({experiment_id})", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(title="Patient ID", loc='lower right')
        plt.tight_layout()

        save_path = output_path / f"summary_accuracy_vs_n_factors_{metric}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
        plt.close(fig)

    def plot_factors_for_good_accuracy_summary(self, experiment_id: str, output_dir: str = None, margin: float = 0.0):
        """
        Generates a bar plot showing the minimum number of factors required to achieve "good" accuracy for each patient.
        
        "Good" accuracy is defined as the first point in the regularization path where both cancer and normal
        cell accuracies exceed the trivial accuracy (plus an optional margin). Handles both CV and non-CV results.

        Args:
            experiment_id: The ID of the experiment to analyze.
            output_dir: Custom output directory. Defaults to the experiment's summary_plots dir.
            margin: An optional margin to add to the trivial accuracy threshold (e.g., 0.1 for 10% better).
        """
        print(f"--- Generating 'Factors for Good Accuracy' Summary for Experiment: {experiment_id} ---")
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

        all_metrics_df = results['metrics']
        all_coefficients = results['coefficients']
        
        factors_for_good_accuracy = {}
        patients = sorted(all_coefficients.keys())

        for patient_id in patients:
            patient_metrics = all_metrics_df[all_metrics_df['group'] == patient_id].copy()
            patient_coefs = all_coefficients[patient_id].copy()
            
            if patient_metrics.empty or patient_coefs.empty:
                print(f"Warning: Missing data for patient {patient_id}. Skipping.")
                continue

            # --- Dynamically select metric columns based on availability ---
            is_cv_data = 'trivial_accuracy_mean' in patient_metrics.columns
            trivial_col = 'trivial_accuracy_mean' if is_cv_data else 'trivial_accuracy'
            mal_col = 'mal_accuracy_mean' if is_cv_data else 'mal_accuracy'
            norm_col = 'norm_accuracy_mean' if is_cv_data else 'norm_accuracy'

            required_cols = [trivial_col, mal_col, norm_col]
            if not all(col in patient_metrics.columns for col in required_cols):
                print(f"Warning: Missing one of required columns {required_cols} for patient {patient_id}. Skipping.")
                continue
            # --- End dynamic selection ---
            
            # Sort by alpha from largest to smallest (strongest to weakest regularization)
            # This finds the most parsimonious model (fewest factors) that meets the criteria.
            patient_metrics.sort_values('alpha', inplace=True, ascending=False)
            
            good_enough_point = None
            for _, row in patient_metrics.iterrows():
                trivial_acc_threshold = row[trivial_col] + margin
                if row[mal_col] > trivial_acc_threshold and row[norm_col] > trivial_acc_threshold:
                    good_enough_point = row
                    break
            
            if good_enough_point is not None:
                target_alpha = good_enough_point['alpha']
                
                # Find the closest alpha in coefficient data
                try:
                    # Handle cases where columns might not be in 'alpha_...' format
                    try:
                        coef_alphas = np.array([float(col.split('_')[1]) for col in patient_coefs.columns])
                    except (ValueError, IndexError):
                        coef_alphas = patient_coefs.columns.astype(float)

                    closest_alpha_idx = (np.abs(coef_alphas - target_alpha)).argmin()
                    target_col_name = patient_coefs.columns[closest_alpha_idx]
                    
                    n_factors = (patient_coefs[target_col_name] != 0).sum()
                    factors_for_good_accuracy[patient_id] = n_factors
                except Exception as e:
                    print(f"Warning: Could not process coefficients for patient {patient_id}. Error: {e}")
                    factors_for_good_accuracy[patient_id] = np.nan
            else:
                print(f"Warning: No point found meeting 'good accuracy' criteria for patient {patient_id}.")
                factors_for_good_accuracy[patient_id] = np.nan

        # Generate the bar plot
        if not factors_for_good_accuracy:
            print("No data to plot.")
            return
            
        fig, ax = plt.subplots(figsize=(max(10, len(patients) * 0.8), 6))
        
        factor_series = pd.Series(factors_for_good_accuracy).dropna()
        
        if factor_series.empty:
            print("No patients met the criteria for plotting.")
            return

        factor_series.plot(kind='bar', ax=ax, color='steelblue', alpha=0.9)
        
        ax.set_ylabel("Number of Factors for 'Good' Accuracy", fontsize=12)
        ax.set_xlabel("Patient ID", fontsize=12)
        ax.set_title(f"Factors Needed for Good Performance (>{margin*100:.0f}% above Trivial)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, val in enumerate(factor_series.values):
            ax.text(i, val + 0.5, str(int(val)), ha='center', va='bottom')

        plt.tight_layout()
        save_path = output_path / "summary_factors_for_good_accuracy.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary bar plot to {save_path}")
        plt.close(fig)

    def plot_patient_cluster_w_factor_coefs_at_best_reg_strength(self, experiment_id: str, output_dir: str = None, margin: float = 0.0):
        """
        Clusters patients based on their factor coefficients at their optimal regularization strength.

        For each patient, it first identifies the most parsimonious model (strongest regularization)
        that achieves "good performance". It then extracts the full coefficient vector for that model.
        Finally, it performs hierarchical clustering on the resulting patient-factor matrix and
        visualizes the result as a clustered heatmap.

        Args:
            experiment_id: The ID of the experiment to analyze.
            output_dir: Custom output directory. Defaults to the experiment's summary_plots dir.
            margin: An optional margin to add to the trivial accuracy threshold for defining "good performance".
        """
        print(f"--- Clustering Patients by Factor Coefficients for Experiment: {experiment_id} ---")
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

        all_metrics_df = results['metrics']
        all_coefficients = results['coefficients']

        optimal_coef_vectors = {}
        patients = sorted(all_coefficients.keys())

        for patient_id in patients:
            patient_metrics = all_metrics_df[all_metrics_df['group'] == patient_id].copy()
            patient_coefs = all_coefficients[patient_id].copy()

            if patient_metrics.empty or patient_coefs.empty:
                continue

            is_cv_data = 'trivial_accuracy_mean' in patient_metrics.columns
            trivial_col = 'trivial_accuracy_mean' if is_cv_data else 'trivial_accuracy'
            mal_col = 'mal_accuracy_mean' if is_cv_data else 'mal_accuracy'
            norm_col = 'norm_accuracy_mean' if is_cv_data else 'norm_accuracy'

            required_cols = [trivial_col, mal_col, norm_col]
            if not all(col in patient_metrics.columns for col in required_cols):
                continue

            patient_metrics.sort_values('alpha', inplace=True, ascending=False)

            good_enough_point = None
            for _, row in patient_metrics.iterrows():
                trivial_acc_threshold = row[trivial_col] + margin
                if row[mal_col] > trivial_acc_threshold and row[norm_col] > trivial_acc_threshold:
                    good_enough_point = row
                    break
            
            if good_enough_point is not None:
                target_alpha = good_enough_point['alpha']
                try:
                    try:
                        coef_alphas = np.array([float(col.split('_')[1]) for col in patient_coefs.columns])
                    except (ValueError, IndexError):
                        coef_alphas = patient_coefs.columns.astype(float)

                    closest_alpha_idx = (np.abs(coef_alphas - target_alpha)).argmin()
                    target_col_name = patient_coefs.columns[closest_alpha_idx]
                    
                    optimal_coef_vectors[patient_id] = patient_coefs[target_col_name]
                except Exception as e:
                    print(f"Warning: Could not process coefficients for patient {patient_id}. Error: {e}")
            else:
                 print(f"Warning: No 'good performance' point found for patient {patient_id}. Skipping.")


        if not optimal_coef_vectors:
            print("Could not determine optimal coefficients for any patient. Aborting.")
            return

        # Create the patient-factor matrix
        patient_factor_df = pd.DataFrame(optimal_coef_vectors).T.fillna(0)
        
        # Remove factors that are zero for all patients to clean up the heatmap
        patient_factor_df = patient_factor_df.loc[:, (patient_factor_df != 0).any(axis=0)]

        if patient_factor_df.empty:
            print("No factors with non-zero coefficients found across all patients. Aborting heatmap generation.")
            return

        # Create the clustermap
        try:
            g = sns.clustermap(
                patient_factor_df,
                method='ward',      # Clustering method
                metric='euclidean', # Distance metric
                cmap='RdBu_r',      # Color map, centered at zero
                center=0,
                figsize=(max(15, patient_factor_df.shape[1] * 0.2), 
                         max(8, patient_factor_df.shape[0] * 0.5)),
                xticklabels=True,
                yticklabels=True,
                cbar_kws={'label': 'Coefficient Value'}
            )
            
            # Use suptitle for a figure-level title to avoid overlap
            g.fig.suptitle(f"Patient Clustering based on Factor Coefficients\n(at Optimal Regularization, >{margin*100:.0f}% above Trivial)", fontsize=14)
            
            g.ax_heatmap.set_xlabel("Factors", fontsize=10)
            g.ax_heatmap.set_ylabel("Patients", fontsize=10)
            
            # Adjust layout to make space for the super title
            g.fig.tight_layout(rect=[0, 0, 1, 0.96])

            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)

            save_path = output_path / "patient_factor_coefficient_clustermap.png"
            g.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved patient clustering heatmap to {save_path}")
            plt.close()

        except Exception as e:
            print(f"Error generating clustermap: {e}")

    def visualize_gene_in_factor_space(self, experiment_id: str, genes_of_interest: List[str], output_dir: str = None):
        """
        Visualizes how a specific list of genes is represented in the factor space.

        This function performs two main tasks:
        1. Calculates and prints the communality score for each gene, which indicates how
           well the gene's variance is explained by the factor model as a whole.
        2. Generates a heatmap of the factor loadings for the specified genes, showing
           which specific factors each gene is associated with.

        Args:
            experiment_id: The ID of the experiment to analyze.
            genes_of_interest: A list of gene symbols to visualize (e.g., ['PTH2R', 'GNG11']).
            output_dir: Custom output directory. Defaults to the experiment's unsupervised_dr_analysis dir.
        """
        print(f"--- Visualizing Genes in Factor Space for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        if output_dir is None:
            output_dir = exp.experiment_dir / "analysis" / "unsupervised_dr_analysis"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Load the transformed adata to get loadings and gene names
        try:
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            transformed_adata_path = exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components)
            transformed_adata = sc.read_h5ad(transformed_adata_path)

            if f'{dr_method.upper()}_loadings' not in transformed_adata.varm:
                print(f"Error: '{dr_method.upper()}_loadings' not found in transformed_adata.varm. Cannot proceed.")
                return

            loadings = transformed_adata.varm[f'{dr_method.upper()}_loadings']
            loading_df = pd.DataFrame(loadings, 
                                      index=transformed_adata.var_names, 
                                      columns=[f'X_{dr_method}_{i+1}' for i in range(loadings.shape[1])])
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading required data: {e}. Aborting.")
            return

        # 1. Calculate and report communality
        print("\n--- Communality Scores ---")
        communalities = (loading_df ** 2).sum(axis=1)
        
        valid_genes = []
        for gene in genes_of_interest:
            if gene in communalities.index:
                print(f"  - {gene}: {communalities[gene]:.4f}")
                valid_genes.append(gene)
            else:
                print(f"  - {gene}: Not found in the model's gene list.")
        
        if not valid_genes:
            print("\nNone of the specified genes were found. Cannot generate heatmap.")
            return

        # 2. Generate loading heatmap
        gene_loadings_df = loading_df.loc[valid_genes]

        # Filter out factors where all specified genes have near-zero loadings to make the heatmap cleaner
        significant_factors = gene_loadings_df.loc[:, (gene_loadings_df.abs() > 0.01).any(axis=0)]

        if significant_factors.empty:
            print("\nNo factors with significant loadings for the specified genes. Heatmap not generated.")
            return
        
        # Only show annotations if the plot is likely to be readable
        show_annotations = significant_factors.shape[1] <= 25

        plt.figure(figsize=(max(15, significant_factors.shape[1] * 0.3), 
                            max(6, significant_factors.shape[0] * 0.5)))
        
        sns.heatmap(significant_factors, cmap='coolwarm', center=0, annot=show_annotations, fmt=".2f", linewidths=.5)
        
        plt.title(f"Factor Loadings for Genes of Interest", fontsize=16)
        plt.xlabel("Factors", fontsize=12)
        plt.ylabel("Genes", fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        save_path = output_dir / f"gene_loadings_heatmap_{'_'.join(valid_genes)}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nSaved gene loading heatmap to: {save_path}")

    def plot_patient_cluster_in_gene_space(self, experiment_id: str, output_dir: str = None, margin: float = 0.0, n_top_genes: int = 100):
        """
        Calculates a "predictive loading" for each patient and clusters them in gene space.

        This function first calculates a patient-specific gene signature by taking the dot product
        of the factor loading matrix (L) and the patient's optimal LASSO coefficient vector (β).
        It then clusters patients based on these gene-level signatures to find biologically
        interpretable subgroups.

        Args:
            experiment_id: The ID of the experiment to analyze.
            output_dir: Custom output directory. Defaults to the experiment's summary_plots dir.
            margin: An optional margin to add to the trivial accuracy threshold for defining "good performance".
            n_top_genes: The number of most variable genes (across patient signatures) to use for the heatmap.
        """
        print(f"--- Clustering Patients in Gene Space for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        if output_dir is None:
            output_path = exp.get_path('summary_plots')
        else:
            output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 1. Get optimal coefficient vectors (β) for each patient
        results = self.extract_classification_results(experiment_id)
        if 'metrics' not in results or 'coefficients' not in results:
            print("No classification results found. Cannot proceed.")
            return
        
        all_metrics_df = results['metrics']
        all_coefficients = results['coefficients']
        optimal_coef_vectors = {}
        patients = sorted(all_coefficients.keys())

        for patient_id in patients:
            # This logic is duplicated from the previous function, could be refactored
            patient_metrics = all_metrics_df[all_metrics_df['group'] == patient_id].copy()
            patient_coefs = all_coefficients[patient_id].copy()
            if patient_metrics.empty or patient_coefs.empty: continue
            is_cv_data = 'trivial_accuracy_mean' in patient_metrics.columns
            trivial_col, mal_col, norm_col = ('trivial_accuracy_mean', 'mal_accuracy_mean', 'norm_accuracy_mean') if is_cv_data else ('trivial_accuracy', 'mal_accuracy', 'norm_accuracy')
            if not all(col in patient_metrics.columns for col in [trivial_col, mal_col, norm_col]): continue
            patient_metrics.sort_values('alpha', inplace=True, ascending=False)
            good_enough_point = None
            for _, row in patient_metrics.iterrows():
                if row[mal_col] > row[trivial_col] + margin and row[norm_col] > row[trivial_col] + margin:
                    good_enough_point = row
                    break
            if good_enough_point is not None:
                target_alpha = good_enough_point['alpha']
                try:
                    coef_alphas = patient_coefs.columns.astype(float)
                except ValueError:
                    coef_alphas = np.array([float(c.split('_')[1]) for c in patient_coefs.columns])
                closest_alpha_idx = (np.abs(coef_alphas - target_alpha)).argmin()
                optimal_coef_vectors[patient_id] = patient_coefs.iloc[:, closest_alpha_idx]

        if not optimal_coef_vectors:
            print("Could not determine optimal coefficients for any patient. Aborting.")
            return
        patient_factor_df = pd.DataFrame(optimal_coef_vectors).T.fillna(0)

        # 2. Get Factor Loading matrix (L)
        try:
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            loadings = transformed_adata.varm[f'{dr_method.upper()}_loadings']
            loading_df = pd.DataFrame(loadings, index=transformed_adata.var_names)
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading factor loadings: {e}. Aborting.")
            return

        # 3. Calculate patient-specific gene signatures (S = L @ β.T)
        # Ensure factor orders match between loadings and coefficients
        patient_factor_df.columns = loading_df.columns # Ensure column names match for dot product
        predictive_loadings = patient_factor_df @ loading_df.T
        
        # 4. Select top variable genes for visualization
        gene_variance = predictive_loadings.var(axis=0)
        top_genes = gene_variance.nlargest(n_top_genes).index
        plot_df = predictive_loadings[top_genes]

        # 5. Generate clustermap
        print(f"Clustering patients based on the top {n_top_genes} most variable signature genes.")
        g = sns.clustermap(
            plot_df,
            method='ward', metric='euclidean',
            cmap='viridis', z_score=0, # z_score along rows (genes) to see pattern, not magnitude
            figsize=(max(15, plot_df.shape[1] * 0.15), 
                     max(8, plot_df.shape[0] * 0.5)),
            xticklabels=True, yticklabels=True
        )

        g.fig.suptitle(f"Patient Clustering in Gene Space\n(Top {n_top_genes} Genes)", fontsize=16)
        g.ax_heatmap.set_xlabel("Genes", fontsize=10)
        g.ax_heatmap.set_ylabel("Patients", fontsize=10)
        g.fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = output_path / "patient_gene_space_clustermap.png"
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gene-space clustering heatmap to: {save_path}")
        plt.close()

    def analyze_factor_similarity(self, experiment_id: str, top_n_genes: int = 50, output_dir: str = None):
        """
        Analyzes the similarity between factors in a model using two methods:
        1.  Factor-Factor Correlation: Calculates the Pearson correlation between factor loading vectors.
        2.  Top Gene Overlap: Calculates the Jaccard similarity of the top N genes for each factor.

        This helps identify redundant factors or groups of factors representing similar biological programs.

        Args:
            experiment_id: The ID of the experiment to analyze.
            top_n_genes: The number of top genes (by absolute loading) to use for the overlap calculation.
            output_dir: Custom output directory. Defaults to the experiment's unsupervised_dr_analysis dir.
        """
        print(f"--- Analyzing Factor Similarity for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        if output_dir is None:
            output_path = exp.experiment_dir / "analysis" / "unsupervised_dr_analysis"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 1. Load Factor Loading matrix (L)
        try:
            dr_method = exp.config.get('dimension_reduction.method')
            n_components = exp.config.get('dimension_reduction.n_components')
            transformed_adata = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
            loadings = transformed_adata.varm[f'{dr_method.upper()}_loadings']
            loading_df = pd.DataFrame(loadings, 
                                      index=transformed_adata.var_names, 
                                      columns=[f'X_{dr_method}_{i+1}' for i in range(loadings.shape[1])])
        except (FileNotFoundError, KeyError) as e:
            print(f"Error loading factor loadings: {e}. Aborting.")
            return

        # 2. Method 1: Factor-Factor Correlation
        print("  Calculating factor-factor correlation...")
        factor_corr = loading_df.corr(method='pearson')

        g_corr = sns.clustermap(
            factor_corr,
            cmap='vlag', # A diverging colormap is good for correlations
            center=0,
            figsize=(14, 12)
        )
        g_corr.fig.suptitle('Factor-Factor Correlation based on Gene Loadings', fontsize=16)
        g_corr.ax_heatmap.set_xlabel("Factors")
        g_corr.ax_heatmap.set_ylabel("Factors")
        g_corr.fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        corr_save_path = output_path / "factor_loading_correlation_heatmap.png"
        g_corr.savefig(corr_save_path, dpi=300)
        print(f"  Saved factor correlation heatmap to: {corr_save_path}")
        plt.close()

        # 3. Method 2: Top Gene Overlap
        print(f"  Calculating top {top_n_genes} gene overlap (Jaccard Index)...")
        top_genes_per_factor = {}
        for factor in loading_df.columns:
            top_genes = loading_df[factor].abs().nlargest(top_n_genes).index
            top_genes_per_factor[factor] = set(top_genes)

        overlap_matrix = pd.DataFrame(index=loading_df.columns, columns=loading_df.columns, dtype=float)
        for factor1 in loading_df.columns:
            for factor2 in loading_df.columns:
                set1 = top_genes_per_factor[factor1]
                set2 = top_genes_per_factor[factor2]
                jaccard_index = len(set1.intersection(set2)) / len(set1.union(set2))
                overlap_matrix.loc[factor1, factor2] = jaccard_index
                
        g_overlap = sns.clustermap(
            overlap_matrix,
            cmap='viridis',
            figsize=(14, 12)
        )
        g_overlap.fig.suptitle(f'Factor Similarity based on Top {top_n_genes} Gene Overlap (Jaccard Index)', fontsize=16)
        g_overlap.ax_heatmap.set_xlabel("Factors")
        g_overlap.ax_heatmap.set_ylabel("Factors")
        g_overlap.fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        overlap_save_path = output_path / "factor_top_gene_overlap_heatmap.png"
        g_overlap.savefig(overlap_save_path, dpi=300)
        print(f"  Saved gene overlap heatmap to: {overlap_save_path}")
        plt.close()

    def generate_fa_diagnostic_plots(self, experiment_id: str):
        """
        Generates and saves diagnostic plots specific to a Factor Analysis model.
        """
        print("--- Generating FA-specific diagnostic plots ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        if exp.config.get('dimension_reduction.method') != 'fa':
            print("  Skipping: Experiment is not a Factor Analysis run.")
            return

        output_dir = exp.experiment_dir / "analysis" / "dimension_reduction_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        adata_path = exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components)
        if not adata_path.exists():
            print(f"  Could not find transformed data at {adata_path}")
            return
            
        adata = sc.read_h5ad(adata_path)

        # Plot SS Loadings
        ss_loadings = adata.uns.get('fa', {}).get('ss_loadings_per_factor')
        if ss_loadings is not None:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(ss_loadings)), ss_loadings)
            plt.title(f"SS Loadings per Factor\n({exp.config.experiment_id})")
            plt.xlabel("Factor Index")
            plt.ylabel("SS Loadings")
            save_path = output_dir / "ss_loadings_per_factor.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved SS Loadings plot to {save_path}")
        else:
            print("  Could not find 'ss_loadings_per_factor' in AnnData .uns field.")

        # Plot Communality
        if 'communality' in adata.var.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(adata.var['communality'], bins=50, kde=True)
            plt.title(f"Gene Communality Distribution\n({exp.config.experiment_id})")
            plt.xlabel("Gene Communality")
            plt.ylabel("Density")
            save_path = output_dir / "gene_communality_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved Gene Communality plot to {save_path}")
        else:
            print("  Could not find 'communality' in AnnData .var field.")

    def run_predictive_loading_gsea(self, experiment_id: str, patient_reg_strength_indices: dict, 
                                    gene_sets_path: str = '/home/minhang/mds_project/data/cohort_adata/gene_sets/h.all.v2024.1.Hs.symbols.gmt'):
        """
        For each patient and specified regularization strength, calculates the "predictive loadings"
        and runs GSEA to find enriched pathways driving the classification.
        """
        print(f"--- Running Predictive Loading GSEA for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        # Load the base data needed for all patients
        dr_method = exp.config.get('dimension_reduction.method')
        n_components = exp.config.get('dimension_reduction.n_components')
        adata_transformed = sc.read_h5ad(exp.get_path('transformed_data', dr_method=dr_method, n_components=n_components))
        factor_loadings = pd.DataFrame(adata_transformed.varm['FA_loadings'], 
                                       index=adata_transformed.var_names, 
                                       columns=[f'X_fa_{i+1}' for i in range(n_components)])

        for patient_id, indices in patient_reg_strength_indices.items():
            print(f"\n  Processing Patient: {patient_id}")
            
            # Load patient-specific coefficients
            try:
                coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id), index_col=0)
            except FileNotFoundError:
                print(f"    WARNING: Coefficient file not found for patient {patient_id}. Skipping.")
                continue

            for alpha_idx in indices:
                # 0-based index for DataFrame access
                alpha_idx_0based = alpha_idx - 1 
                if alpha_idx_0based >= len(coef_df.columns):
                    print(f"    WARNING: Alpha index {alpha_idx} is out of bounds for patient {patient_id}. Skipping.")
                    continue
                
                col_name = coef_df.columns[alpha_idx_0based]
                print(f"    Running GSEA for alpha index {alpha_idx} ({col_name})...")

                # Get the coefficient vector for this alpha
                factor_coefficients = coef_df.iloc[:, alpha_idx_0based]
                
                # Calculate predictive loadings: (gene x factor) @ (factor x 1) -> (gene x 1)
                predictive_loadings = factor_loadings.dot(factor_coefficients)
                
                # Rank genes by their predictive loading score
                ranked_genes = predictive_loadings.sort_values(ascending=False).dropna()

                # Define output directory
                output_dir = exp.experiment_dir / "analysis" / "predictive_loading_gsea" / patient_id / f"alpha_idx_{alpha_idx}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Run GSEA Prerank
                    prerank_res = gseapy.prerank(
                        rnk=ranked_genes, gene_sets=gene_sets_path, outdir=str(output_dir),
                        min_size=5, max_size=1000, seed=42, no_plot=True
                    )
                    
                    # Generate and save the summary barplot
                    results_csv_path = output_dir / "gseapy.gene_set.prerank.report.csv"
                    plot_title = f"Predictive Loading GSEA - {patient_id} @ alpha_idx {alpha_idx}"
                    plot_path = output_dir / "GSEA_summary_barplot.png"
                    self._plot_gsea_results(results_csv_path, plot_title, gene_sets_path, plot_path)
                    print(f"      GSEA results saved to: {output_dir}")

                except Exception as e:
                    print(f"      ERROR running GSEA for {patient_id} at index {alpha_idx}: {e}")

    def plot_roc_auc_vs_alpha(self, metrics_df: pd.DataFrame, ax, group_name: str, plot_std: bool = False):
        """Plots ROC AUC vs. Alpha on a given matplotlib axis."""
        ax.plot(metrics_df['alpha'], metrics_df['roc_auc'], 'o-', label=group_name)
        if plot_std and 'roc_auc_std' in metrics_df.columns:
            ax.fill_between(
                metrics_df['alpha'],
                metrics_df['roc_auc'] - metrics_df['roc_auc_std'],
                metrics_df['roc_auc'] + metrics_df['roc_auc_std'],
                alpha=0.2
            )
        ax.set_xscale('log')
        ax.set_xlabel("Alpha (Regularization Strength)")
        ax.set_ylabel("ROC AUC")
        ax.set_title("Performance vs. Regularization")
        ax.grid(True)

    def plot_coefficient_paths(self, coef_df: pd.DataFrame, alphas: pd.Series, ax):
        """Plots coefficient paths on a given matplotlib axis."""
        for factor_idx, row in coef_df.iterrows():
            ax.plot(alphas, row.values, label=factor_idx if coef_df.shape[0] < 20 else None)
        ax.set_xscale('log')
        ax.set_xlabel("Alpha (Regularization Strength)")
        ax.set_ylabel("Coefficient Value")
        ax.set_title("Factor Coefficients vs. Regularization")
        if coef_df.shape[0] < 20:
            ax.legend(title="Factor", bbox_to_anchor=(1.05, 1), loc='upper left')

    def plot_patient_lasso_metrics(self, experiment_id: str, patient_id_for_filtering: str, 
                                   metrics_file_override: Optional[str] = None):
        """
        Generates and saves a 2-panel plot for a single patient (or patient group) showing
        the relationship between regularization strength (alpha) and classification metrics.
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        
        # --- REFACTOR: Allow overriding the metrics file path ---
        if metrics_file_override:
            metrics_path = Path(metrics_file_override)
            patient_id_for_filtering = 'all_patients' # Ensure we filter for the correct group name
        else:
            metrics_path = exp.get_path('patient_metrics', patient_id=patient_id_for_filtering)

        output_dir = exp.get_path('summary_plots')
        output_dir.mkdir(exist_ok=True, parents=True)

        try:
            metrics_df_full = pd.read_csv(metrics_path)
            coef_df = pd.read_csv(exp.get_path('patient_coefficients', patient_id=patient_id_for_filtering), index_col=0)
        except FileNotFoundError:
            print(f"  Could not find metrics or coefficients for patient {patient_id_for_filtering}. Skipping.")
            return

        # Filter metrics for the specific patient/group if the file contains multiple
        metrics_df = metrics_df_full[metrics_df_full['group'] == patient_id_for_filtering].copy()

        if metrics_df.empty:
            print(f"No metrics found for patient_id_for_filtering (group) '{patient_id_for_filtering}'")
            return

        # Determine if this is a CV run to plot error bars
        plot_std = 'roc_auc_std' in metrics_df.columns

        # --- REVERT TO STACKED PLOT LAYOUT ---
        fig, axes = plt.subplots(2, 1, figsize=(16, 12)) # 2 rows, 1 column
        title_group = 'Pan-Patient (Aggregated)' if metrics_file_override else f'Patient: {patient_id_for_filtering}'
        fig.suptitle(f"{title_group}\nExperiment: {experiment_id}", fontsize=16)

        # --- Top Panel: Metrics vs. Regularization ---
        self.plot_roc_auc_vs_alpha(metrics_df, ax=axes[0], group_name=patient_id_for_filtering, plot_std=plot_std)
        # Add secondary y-axis for feature survival
        ax2 = axes[0].twinx()
        survival_series = 100 * (coef_df != 0).sum(axis=0) / len(coef_df)
        ax2.plot(metrics_df['alpha'], survival_series, 'o-', color='purple', label='Surviving Features (%)')
        ax2.set_ylabel('Surviving Features (%)')
        ax2.legend(loc='center right')
        axes[0].legend(loc='center left')

        # --- Bottom Panel: Top-10 Factor Coefficient Paths ---
        # Identify top 10 factors by max absolute coefficient value
        top_10_factors = coef_df.abs().max(axis=1).nlargest(10).index
        coef_df_top10 = coef_df.loc[top_10_factors]
        self.plot_coefficient_paths(coef_df_top10, metrics_df['alpha'], ax=axes[1])
        axes[1].set_title("Top-10 Factor Coefficient Paths")


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_name = 'pan_patient_aggregated' if metrics_file_override else f'patient_{patient_id_for_filtering}'
        save_path = output_dir / f"{save_name}_metrics_and_coefficients.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  Saved plot to {save_path}")

    def find_best_tradeoff_alpha(self, patient_metrics_df: pd.DataFrame, patient_coefficients_df: pd.DataFrame, performance_metric: str = 'roc_auc') -> float:
        """
        (New, more robust version)
        Finds the optimal alpha by maximizing the difference between performance and the fraction of surviving features.

        This algorithm calculates a simple score for each regularization strength:
        Trade-off Score = (Performance Score) - (Fraction of Surviving Features)
        
        It joins metrics and coefficients on the actual 'alpha' values, rounding to
        handle floating-point precision issues, making it robust to data ordering.
        """
        perf_col = performance_metric

        if perf_col not in patient_metrics_df.columns:
            raise KeyError(f"Performance column '{perf_col}' not found. Available: {patient_metrics_df.columns.tolist()}")

        # --- 1. Calculate Sparsity from Coefficients ---
        n_total_features = patient_coefficients_df.shape[0]
        n_nonzero_coefs = (patient_coefficients_df != 0).sum(axis=0)
        surviving_pct = (n_nonzero_coefs / n_total_features) * 100.0
        
        # Robustly extract alpha values from column names
        try:
            coef_alphas = np.array([float(c.split('_')[1]) for c in patient_coefficients_df.columns])
        except (ValueError, IndexError):
            coef_alphas = patient_coefficients_df.columns.astype(float)

        sparsity_df = pd.DataFrame({
            'alpha': coef_alphas,
            'surviving_features_pct': surviving_pct.values
        })

        # --- 2. Merge on Alpha using a nearest-value approach for robustness ---
        # Sort both dataframes by alpha to enable efficient merging
        metrics_sorted = patient_metrics_df.sort_values('alpha').reset_index()
        sparsity_sorted = sparsity_df.sort_values('alpha').reset_index()

        # `merge_asof` is ideal for joining ordered data with near-matching keys
        merged_df = pd.merge_asof(
            metrics_sorted,
            sparsity_sorted,
            on='alpha',
            direction='nearest', # Finds the closest alpha in sparsity_df for each alpha in metrics_df
            suffixes=('_metrics', '_coefs')
        )

        if merged_df.empty:
            raise ValueError("Could not merge performance and sparsity data.")

        # --- Filter out trivial models with zero features ---
        non_trivial_df = merged_df[merged_df['surviving_features_pct'] > 0].copy()

        if non_trivial_df.empty:
            raise ValueError("All regularization strengths resulted in a model with zero features.")

        # --- Prioritize models that have performed some feature selection ---
        regularized_df = non_trivial_df[non_trivial_df['surviving_features_pct'] < 100].copy()
        
        df_to_score = regularized_df if not regularized_df.empty else non_trivial_df
        # --- DEBUG: Print the dataframe before scoring ---
        # print("\n--- Debugging Dataframe ---")
        # print(df_to_score[['alpha', perf_col, 'surviving_features_pct']].to_string())
        # print("---------------------------\n")
        # ---------------------------------------------
        
        # --- 3. Calculate the Trade-off Score (Performance - Cost) ---
        perf_score = df_to_score[perf_col]
        feature_cost = df_to_score['surviving_features_pct'] / 100.0
        df_to_score['tradeoff_score'] = perf_score - feature_cost
        print(f'{df_to_score["tradeoff_score"].max()} is the trade-off score of the best point')
        # --- 4. Find the Alpha with the Maximum Score ---
        best_point = df_to_score.loc[df_to_score['tradeoff_score'].idxmax()]
        
        # Return the original alpha value from the metrics dataframe
        return best_point['alpha']

    def _parse_gsea_results(self, exp, patient_id, alpha_idx, fdr_threshold=0.01):
        """
        Parses a GSEA result file and returns a set of significant pathways.
        """
        try:
            exp_dir = exp.experiment_dir
            gsea_path = exp_dir / "analysis" / "supervised_gsea" / patient_id / f"alpha_idx_{alpha_idx}" / "gseapy.gene_set.prerank.report.csv"

            if not gsea_path.exists():
                return set()

            gsea_df = pd.read_csv(gsea_path)
            
            if 'FDR q-val' not in gsea_df.columns or 'Term' not in gsea_df.columns:
                return set()

            # Convert to numeric, coercing errors to NaN
            numeric_fdr = pd.to_numeric(gsea_df['FDR q-val'], errors='coerce')
            
            significant_pathways = gsea_df[numeric_fdr < fdr_threshold]
            
            return set(significant_pathways['Term'])

        except Exception:
            # This is a silent failure, which is acceptable for this analysis function
            return set()

    def analyze_gsea_stability(self, experiment_id, patient_reg_strength_indices):
        """
        Quantifies the stability of top GSEA pathways across varying regularization strengths.

        Args:
            experiment_id (str): The ID of the experiment.
            patient_reg_strength_indices (dict): A dictionary where keys are patient IDs
                                                 and values are lists of alpha indices to analyze.

        Returns:
            dict: A dictionary containing stability analysis results for each patient.
                  Each patient's entry contains:
                  - 'jaccard_scores' (dict): Jaccard similarity scores between consecutive alpha indices.
                  - 'overlap_fractions' (dict): Overlap fractions between consecutive alpha indices.
                  - 'all_pathways_union' (set): The union of all significant pathways across all indices.
                  - 'pathway_stability' (pd.Series): The fraction of times each pathway appeared as significant.
        """
        print(f"--- Starting GSEA Stability Analysis for Experiment: {experiment_id} ---")
        
        # Load the experiment object once to avoid redundant loads and print statements
        try:
            exp = self.experiment_manager.load_experiment(experiment_id)
        except Exception as e:
            print(f"  ERROR: Could not load experiment {experiment_id}. Aborting analysis. Details: {e}")
            return {}

        stability_results = {}

        for patient_id, alpha_indices in patient_reg_strength_indices.items():
            print(f"\nAnalyzing patient: {patient_id}")
            
            # --- 1. Collect Significant Pathways for Each Alpha ---
            pathway_sets = {}
            for idx in sorted(alpha_indices):
                pathway_sets[idx] = self._parse_gsea_results(exp, patient_id, idx)

            # --- 2. Quantify Stability Between Consecutive Strengths ---
            jaccard_scores = {}
            overlap_fractions = {}
            sorted_indices = sorted(alpha_indices)

            for i in range(len(sorted_indices) - 1):
                idx1 = sorted_indices[i]
                idx2 = sorted_indices[i+1]
                
                set1 = pathway_sets[idx1]
                set2 = pathway_sets[idx2]

                intersection_size = len(set1.intersection(set2))
                union_size = len(set1.union(set2))
                
                # Jaccard Similarity
                if union_size == 0:
                    jaccard = 1.0 if not set1 and not set2 else 0.0
                else:
                    jaccard = intersection_size / union_size
                jaccard_scores[f'{idx1}_vs_{idx2}'] = jaccard
                
                # Overlap Fraction (relative to the smaller set)
                min_set_size = min(len(set1), len(set2))
                if min_set_size == 0:
                    overlap = 1.0 if not set1 and not set2 else 0.0
                else:
                    overlap = intersection_size / min_set_size
                overlap_fractions[f'{idx1}_vs_{idx2}'] = overlap

            # --- 3. Overall Pathway Stability (Voting Scheme) ---
            all_pathways_list = [pathway for idx in sorted_indices for pathway in pathway_sets[idx]]
            
            if not all_pathways_list:
                print(f"  No significant pathways found for patient {patient_id} across any alpha.")
                pathway_stability_series = pd.Series(dtype=float)
                all_pathways_union = set()
            else:
                pathway_counts = pd.Series(all_pathways_list).value_counts()
                pathway_stability_series = pathway_counts / len(sorted_indices)
                all_pathways_union = set(all_pathways_list)

            # --- 4. Store Results ---
            stability_results[patient_id] = {
                'jaccard_scores': jaccard_scores,
                'overlap_fractions': overlap_fractions,
                'all_pathways_union': all_pathways_union,
                'pathway_stability': pathway_stability_series.sort_values(ascending=False)
            }
            
            # Print summary for the patient
            print(f"  - Found {len(all_pathways_union)} unique significant pathways across {len(sorted_indices)} regularization strengths.")
            if jaccard_scores:
                avg_jaccard = np.mean(list(jaccard_scores.values()))
                print(f"  - Average Jaccard similarity between consecutive strengths: {avg_jaccard:.3f}")
            if pathway_stability_series.any():
                top_5_stable = pathway_stability_series.head(5)
                print("  - Top 5 most stable pathways:")
                for term, stability in top_5_stable.items():
                    print(f"    - {term} (appeared in {stability:.2%} of runs)")

        print("\n--- GSEA Stability Analysis Complete ---")
        return stability_results

    def calculate_cell_type_activity_score(
        self, 
        adata: AnnData, 
        factors: list, 
        cell_type_col: str = 'predicted.annotation'
    ) -> pd.DataFrame:
        """
        Calculates the mean activity score for specified factors across different cell types.

        Args:
            adata: An AnnData object containing factor scores in .obs and cell types in .obs.
            factors: A list of factor names to analyze (e.g., ['X_fa_36', 'X_fa_46']).
            cell_type_col: The name of the .obs column containing cell type annotations.

        Returns:
            A pandas DataFrame where rows are cell types, columns are factors,
            and values are the mean activity scores.
        """
        if cell_type_col not in adata.obs.columns:
            raise KeyError(f"Cell type column '{cell_type_col}' not found in adata.obs.")
        
        # Factor scores are in adata.obsm['X_fa']. We need to construct a temporary DataFrame.
        # The factor names in the `factors` list (e.g., 'X_fa_36') need to be mapped to column indices.
        try:
            factor_indices = [int(f.split('_')[-1]) for f in factors]
        except (ValueError, IndexError):
            raise ValueError(f"Could not parse factor indices from factor names: {factors}. Expected format 'X_fa_N'.")

        if 'X_fa' not in adata.obsm:
            raise KeyError("Factor scores 'X_fa' not found in adata.obsm.")
            
        # Create a DataFrame for the factor scores
        fa_scores_df = pd.DataFrame(adata.obsm['X_fa'][:, factor_indices], index=adata.obs.index, columns=factors)
        
        # Combine with cell type information
        data_to_agg = pd.concat([adata.obs[[cell_type_col]], fa_scores_df], axis=1)
        
        # Calculate the mean score for each factor, grouped by cell type
        activity_scores = data_to_agg.groupby(cell_type_col).mean()
        
        return activity_scores

    def calculate_cell_type_activity_score_grouped(self, adata: anndata.AnnData, factors: list, cell_type_col: str, grouping_col: str) -> tuple[dict, dict]:
        """
        Calculates the mean factor activity score, grouped by cell type and another category (e.g., malignancy).

        Args:
            adata: Transformed AnnData object with factor scores in .obsm['X_fa'].
            factors: List of factor names to analyze.
            cell_type_col: The column in adata.obs to use for cell type labels.
            grouping_col: The column in adata.obs to use for the primary grouping (e.g., 'CN.label').

        Returns:
            A tuple containing two dictionaries:
            - scores_by_group: {group: DataFrame_of_scores}
            - counts_by_group: {group: Series_of_counts}
        """
        if cell_type_col not in adata.obs.columns:
            raise KeyError(f"Cell type column '{cell_type_col}' not found in adata.obs.")
        if grouping_col not in adata.obs.columns:
            raise KeyError(f"Grouping column '{grouping_col}' not found in adata.obs.")

        factor_indices = [int(f.split('_')[-1]) for f in factors]
        fa_scores_df = pd.DataFrame(adata.obsm['X_fa'][:, factor_indices], index=adata.obs.index, columns=factors)
        
        data_to_agg = pd.concat([adata.obs[[cell_type_col, grouping_col]], fa_scores_df], axis=1)
        
        scores_by_group = {}
        counts_by_group = {}
        
        all_cell_types = adata.obs[cell_type_col].unique()

        for group_name, group_df in data_to_agg.groupby(grouping_col):
            # Calculate mean scores
            mean_scores = group_df.groupby(cell_type_col)[factors].mean().reindex(all_cell_types)
            scores_by_group[group_name] = mean_scores
            
            # Calculate cell counts
            counts = group_df[cell_type_col].value_counts().reindex(all_cell_types, fill_value=0)
            counts_by_group[group_name] = counts
            
        return scores_by_group, counts_by_group

    def analyze_unsupervised_gsea_overlap(self, experiment_id: str, factors: list, pathway_name: str) -> dict:
        """
        Analyzes GSEA results from unsupervised factor loadings to find leading-edge genes for a specific pathway and computes their overlap.

        Args:
            experiment_id: The ID of the experiment to analyze.
            factors: A list of factor names (e.g., ['X_fa_36', 'X_fa_46']).
            pathway_name: The exact name of the pathway to look for in the GSEA reports.

        Returns:
            A dictionary where keys are factor names and values are the sets of leading-edge genes for the specified pathway.
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        exp_path = exp.experiment_dir
        gsea_base_path = exp_path / "analysis" / "factor_interpretation" / "gsea_on_factors_original"
        
        leading_edge_genes = {}
        
        print(f"--- Analyzing GSEA results for pathway: '{pathway_name}' ---")
        print(f"Source directory: {gsea_base_path}")

        for factor in factors:
            gsea_report_path = gsea_base_path / factor / "gseapy.gene_set.prerank.report.csv"
            
            if not gsea_report_path.exists():
                print(f"  - WARNING: GSEA report not found for {factor} at {gsea_report_path}")
                leading_edge_genes[factor] = set()
                continue
            
            try:
                gsea_df = pd.read_csv(gsea_report_path)
                pathway_row = gsea_df[gsea_df['Term'] == pathway_name]
                
                if pathway_row.empty:
                    print(f"  - INFO: Pathway '{pathway_name}' not found or not significant in {factor}.")
                    leading_edge_genes[factor] = set()
                    continue
                
                # The column name is 'Lead_genes' based on your provided `head` output
                genes_str = pathway_row.iloc[0]['Lead_genes']
                if isinstance(genes_str, str) and genes_str:
                    genes = set(genes_str.split(';'))
                    leading_edge_genes[factor] = genes
                    print(f"  - Found {len(genes)} leading-edge genes for {factor}.")
                else:
                    leading_edge_genes[factor] = set()

            except Exception as e:
                print(f"  - ERROR: Could not process file for {factor}. Reason: {e}")
                leading_edge_genes[factor] = set()

        return leading_edge_genes

    def analyze_unsupervised_gsea_overlap_signed(self, experiment_id: str, factors: list, pathway_name: str) -> tuple[dict, dict]:
        """
        Analyzes GSEA results to find leading-edge genes, separated by positive or negative enrichment.

        Args:
            experiment_id: The ID of the experiment.
            factors: A list of factor names.
            pathway_name: The name of the pathway to analyze.

        Returns:
            A tuple containing two dictionaries:
            - positive_genes: {factor: {genes}} for pathways with positive NES.
            - negative_genes: {factor: {genes}} for pathways with negative NES.
        """
        exp = self.experiment_manager.load_experiment(experiment_id)
        gsea_base_path = exp.experiment_dir / "analysis" / "factor_interpretation" / "gsea_on_factors_original"
        
        positive_genes = {}
        negative_genes = {}
        
        print(f"--- Analyzing Signed GSEA results for pathway: '{pathway_name}' ---")
        print(f"Source directory: {gsea_base_path}")

        for factor in factors:
            gsea_report_path = gsea_base_path / factor / "gseapy.gene_set.prerank.report.csv"
            
            positive_genes[factor] = set()
            negative_genes[factor] = set()

            if not gsea_report_path.exists():
                print(f"  - WARNING: GSEA report not found for {factor} at {gsea_report_path}")
                continue
            
            try:
                gsea_df = pd.read_csv(gsea_report_path)
                pathway_row = gsea_df[gsea_df['Term'] == pathway_name]
                
                if pathway_row.empty:
                    print(f"  - INFO: Pathway '{pathway_name}' not found in {factor}.")
                    continue
                
                nes_score = pathway_row.iloc[0]['NES']
                genes_str = pathway_row.iloc[0]['Lead_genes']
                
                if isinstance(genes_str, str) and genes_str:
                    genes = set(genes_str.split(';'))
                    if nes_score > 0:
                        positive_genes[factor] = genes
                        print(f"  - Found {len(genes)} positively associated genes for {factor} (NES: {nes_score:.2f}).")
                    else:
                        negative_genes[factor] = genes
                        print(f"  - Found {len(genes)} negatively associated genes for {factor} (NES: {nes_score:.2f}).")

            except Exception as e:
                print(f"  - ERROR: Could not process file for {factor}. Reason: {e}")

        return positive_genes, negative_genes

    def calculate_factor_separation_scores(self, experiment_id: str) -> pd.DataFrame:
        """
        Calculates the distinguishing power of each factor for separating malignant vs. healthy cells
        on a per-patient basis.

        The distinguishing power is measured as a signed AUC-ROC score: 2 * (AUC - 0.5).
        - A score of +1 means the factor perfectly separates cells, with high values indicating malignancy.
        - A score of -1 means perfect separation, with high values indicating health.
        - A score of 0 means the factor has no distinguishing power.

        Args:
            experiment_id: The ID of the experiment to analyze.

        Returns:
            A pandas DataFrame where rows are patients, columns are factors, and values
            are the signed AUC-ROC separation scores.
        """
        self.logger.info(f"--- Calculating Factor Separation Scores for Experiment: {experiment_id} ---")
        exp = self.experiment_manager.load_experiment(experiment_id)

        # Path to the data after dimension reduction
        dr_method = exp.config.get('dimension_reduction.method', 'fa')
        n_components = exp.config.get('dimension_reduction.n_components', 100)
        adata_path = exp.get_path(
            'transformed_data',
            dr_method=dr_method,
            n_components=n_components
        )
        if not adata_path.exists():
            self.logger.error(f"Could not find transformed_data.h5ad for experiment {experiment_id} at {adata_path}.")
            return pd.DataFrame()

        adata = sc.read_h5ad(adata_path)

        patient_col = exp.config.get('classification.patient_column', 'patient')
        target_col = exp.config.get('preprocessing.target_column', 'CN.label')
        positive_class = exp.config.get('preprocessing.positive_class', 'cancer')

        patients = sorted(adata.obs[patient_col].unique())
        n_factors = adata.obsm['X_fa'].shape[1]
        factor_names = [f'fa_{i+1}' for i in range(n_factors)]
        
        all_scores = {}

        for patient_id in patients:
            self.logger.info(f"  Processing patient: {patient_id}")
            adata_patient = adata[adata.obs[patient_col] == patient_id].copy()

            # Prepare target variable y (0s and 1s)
            y_true = (adata_patient.obs[target_col] == positive_class).astype(int)

            if len(y_true.unique()) < 2:
                self.logger.warning(f"  Skipping {patient_id}: only one class present.")
                continue

            patient_scores = []
            X_patient = adata_patient.obsm['X_fa']

            for i in range(n_factors):
                y_scores = X_patient[:, i]
                try:
                    auc = roc_auc_score(y_true, y_scores)
                    signed_auc = 2 * (auc - 0.5)
                    patient_scores.append(signed_auc)
                except ValueError:
                    patient_scores.append(np.nan)
            
            all_scores[patient_id] = patient_scores

        if not all_scores:
            self.logger.warning("No scores were calculated for any patient.")
            return pd.DataFrame()

        scores_df = pd.DataFrame.from_dict(all_scores, orient='index', columns=factor_names)
        
        self.logger.info("Factor separation score calculation complete.")
        return scores_df

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