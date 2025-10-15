#!/usr/bin/env python
"""
Pipeline script for running Factor Analysis on PREPROCESSED (HVG-selected, standardized)
MRD cancer/normal cells, followed by per-patient LR-Lasso classification.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# --- Configuration ---
# **NEW**: Path to the PREPROCESSED AnnData object
PREPROCESSED_ADATA_PATH = '/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_mrd_hvg_std_may31.h5ad' 
BASE_OUTPUT_DIR = "results_fa_lrlasso_mrd_pipeline_prestd_hvg_may31" # Reflects pre-standardized input

PATIENT_COL = 'patient'
# TIMEPOINT_COL, MRD_LABEL, TARGET_COL etc. are less critical here as data is pre-filtered,
# but PATIENT_COL is still needed for per-patient analysis.
# TARGET_COL is needed for LR-Lasso.
TARGET_COL = 'CN.label' 
TARGET_VALUE_POSITIVE = 'cancer'
TARGET_VALUE_NEGATIVE = 'normal'


# Factor Analysis parameters
N_COMPONENTS_LIST_FOR_FA = [20, 50, 100] # Or whatever you used
FA_RANDOM_STATE = 0
FA_SVD_METHOD = 'lapack' # From previous discussion

# LR-Lasso parameters
ALPHAS_LASSO = np.logspace(-4, 5, 20)

# --- Setup Paths for Imports (same as before) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) 
if not os.path.isdir(os.path.join(SCRIPT_DIR, "sc_classification")) and os.path.isdir(os.path.join(PROJECT_ROOT, "sc_classification")):
    sys.path.insert(0, PROJECT_ROOT)
else: 
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

try:
    # Ensure your FactorAnalysis class is the updated one that accepts svd_method
    from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis 
    from sc_classification.classification_methods.lr_lasso import LRLasso
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print(f"Starting FA (on pre-std HVGs, {FA_SVD_METHOD} SVD) + LR-Lasso pipeline.")
    start_time_pipeline = time.time()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {BASE_OUTPUT_DIR}")

    # --- Step 1: Load PREPROCESSED Data ---
    print(f"\nLoading PREPROCESSED AnnData from: {PREPROCESSED_ADATA_PATH}")
    try:
        # This data is already filtered for MRD, HVG-selected, and standardized
        adata_mrd_preprocessed = sc.read_h5ad(PREPROCESSED_ADATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Preprocessed AnnData file not found: {PREPROCESSED_ADATA_PATH}. Exiting.")
        return
    print(f"Preprocessed AnnData loaded: {adata_mrd_preprocessed.shape[0]} cells, {adata_mrd_preprocessed.shape[1]} features (HVGs).")

    if adata_mrd_preprocessed.n_obs == 0:
        print("Preprocessed AnnData is empty. Exiting.")
        return
    
    # The base anndata for FA is now the fully preprocessed one
    adata_for_fa_base = adata_mrd_preprocessed 

    if PATIENT_COL not in adata_for_fa_base.obs.columns:
        print(f"ERROR: Patient column '{PATIENT_COL}' not found. Exiting.")
        return
        
    unique_patients_for_classification = sorted(list(adata_for_fa_base.obs[PATIENT_COL].unique()))
    print(f"Found {len(unique_patients_for_classification)} unique patients for FA on {adata_for_fa_base.n_vars} pre-standardized HVGs: {unique_patients_for_classification}")

    for n_factors in N_COMPONENTS_LIST_FOR_FA:
        print(f"\n--- Processing for {n_factors} factors ---")
        current_output_dir_n_factors = os.path.join(BASE_OUTPUT_DIR, f"n_factors_{n_factors}")
        os.makedirs(current_output_dir_n_factors, exist_ok=True)
        
        fa_processing_start_time = time.time()
        # Use a copy of the preprocessed data for each FA run
        adata_for_current_fa = adata_for_fa_base.copy() 
        
        print(f"Running Factor Analysis with {n_factors} components on {adata_for_current_fa.n_obs} cells and {adata_for_current_fa.n_vars} pre-standardized HVGs...")
        fa_model_instance = FactorAnalysis()
        
        adata_fa_transformed = fa_model_instance.fit_transform(
            adata_for_current_fa, 
            n_components=n_factors, 
            random_state=FA_RANDOM_STATE,
            standardize_input=False, # CRITICAL: Input is ALREADY standardized
            svd_method=FA_SVD_METHOD, # Pass the SVD method
            save_model=False
        )
        
        # --- Write enhanced FA summary ---
        fa_summary_path = os.path.join(current_output_dir_n_factors, "fa_summary.txt")
        with open(fa_summary_path, "w") as f:
            f.write(f"Factor Analysis Summary for n_components = {n_factors} on {adata_fa_transformed.n_vars} HVGs\n")
            f.write(f"Input AnnData was pre-standardized externally.\n") # Note about pre-standardization
            f.write(f"Internal standardization by FactorAnalysis class call: {adata_fa_transformed.uns['fa']['standardized_input']}\n") # Should be False
            f.write(f"SVD Method Used (in SklearnFA): {adata_fa_transformed.uns['fa'].get('svd_method_used', FA_SVD_METHOD)}\n")
            
            # ... (rest of the fa_summary.txt writing, same as before, using fields from adata_fa_transformed.uns['fa']) ...
            f.write("\n--- Factor Score Variances ---\n")
            f.write(f"Individual Factor Score Variances:\n{adata_fa_transformed.uns['fa']['factor_score_variances']}\n")
            f.write(f"Sum of Factor Score Variances: {adata_fa_transformed.uns['fa']['sum_factor_score_variances']:.4f}\n")
            
            f.write("\n--- Gene-Level Metrics ---\n")
            comm_summary = adata_fa_transformed.uns['fa']['communalities_per_gene_summary']
            f.write("Communalities per Gene Summary:\n")
            for key, val in comm_summary.items():
                f.write(f"  {key.capitalize()}: {val:.4f}\n")
            
            f.write("\n--- Factor-Level Metrics ---\n")
            f.write(f"Sum of Squared Loadings (SS Loadings) per Factor (V_j):\n{adata_fa_transformed.uns['fa']['ss_loadings_per_factor']}\n")
            
            f.write("\n--- Noise/Unique Variance ---\n")
            f.write(f"Mean Noise Variance per Feature (Uniqueness): {np.mean(adata_fa_transformed.uns['fa']['noise_variance_per_feature']):.4f}\n")

        print(f"Saved enhanced FA summary to {fa_summary_path}")

        communalities_df = pd.DataFrame({
            'gene': adata_fa_transformed.var_names,
            'communality': adata_fa_transformed.var['communality']
        })
        communalities_csv_path = os.path.join(current_output_dir_n_factors, f"gene_communalities_factors_{n_factors}.csv")
        communalities_df.to_csv(communalities_csv_path, index=False)
        print(f"Saved per-gene communalities to {communalities_csv_path}")

        fa_duration = time.time() - fa_processing_start_time
        print(f"Factor Analysis for {n_factors} factors completed in {fa_duration:.2f} seconds.")

        # --- Per-Patient LR-Lasso (loop remains the same) ---
        # ... (Ensure this part is copied correctly from your working version / previous script) ...
        for patient_id in unique_patients_for_classification:
            patient_classification_start_time = time.time()
            patient_output_dir = os.path.join(current_output_dir_n_factors, f"patient_{patient_id}")
            os.makedirs(patient_output_dir, exist_ok=True)
            adata_patient_with_fa = adata_fa_transformed[adata_fa_transformed.obs[PATIENT_COL] == patient_id].copy()
            
            print(f"\n  Processing Patient ID: {patient_id} (for {n_factors} factors)")
            # ... (rest of the patient processing code from your previous script) ...
            print(f"  Number of cells for patient {patient_id}: {adata_patient_with_fa.n_obs}")

            class_counts = adata_patient_with_fa.obs[TARGET_COL].value_counts()
            print(f"  Class distribution for patient {patient_id}: {class_counts.to_dict()}")

            if not (TARGET_VALUE_POSITIVE in class_counts and TARGET_VALUE_NEGATIVE in class_counts and \
                    class_counts[TARGET_VALUE_POSITIVE] > 0 and class_counts[TARGET_VALUE_NEGATIVE] > 0):
                print(f"  Skipping patient {patient_id} (for {n_factors} factors): insufficient data for both classes.")
                continue
            
            classifier = LRLasso(adata_patient_with_fa, target_col=TARGET_COL, target_value=TARGET_VALUE_POSITIVE)
            print(f"  Preparing data for LR-Lasso for patient {patient_id}...")
            X_patient_factors, y_patient_labels, feature_names_factors, _ = classifier.prepare_data(
                use_factorized=True, factorization_method='X_fa' 
            )
            
            if X_patient_factors is None or y_patient_labels is None or X_patient_factors.shape[0] == 0:
                print(f"  Data preparation yielded no data for patient {patient_id}. Skipping.")
                continue
            if len(np.unique(y_patient_labels)) < 2:
                print(f"  Patient {patient_id} (after LRLasso prepare_data) no longer has both classes. Skipping.")
                continue

            print(f"  Running LR-Lasso for patient {patient_id} using {X_patient_factors.shape[1]} factors...")
            classification_results = classifier.fit_along_regularization_path(
                X_patient_factors, y_patient_labels, feature_names_factors, 
                alphas=ALPHAS_LASSO, metrics_grouping=PATIENT_COL 
            )
            
            coefs_df = pd.DataFrame(classification_results['coefs'], index=feature_names_factors, columns=ALPHAS_LASSO)
            coefs_filename = os.path.join(patient_output_dir, f"results_coefs_{patient_id}_factors_{n_factors}.csv")
            coefs_df.to_csv(coefs_filename, index=True)
            print(f"    Saved coefficients to {coefs_filename}")
            
            metrics_df = pd.DataFrame(classification_results['group_metrics_path'])
            metrics_filename = os.path.join(patient_output_dir, f"results_metrics_{patient_id}_factors_{n_factors}.csv")
            metrics_df.to_csv(metrics_filename, index=False)
            print(f"    Saved metrics to {metrics_filename}")
            
            patient_duration = time.time() - patient_classification_start_time
            print(f"  Completed processing for patient {patient_id} ({n_factors} factors) in {patient_duration:.2f} seconds.")

    pipeline_duration = time.time() - start_time_pipeline
    print(f"\n--- Pipeline completed in {pipeline_duration:.2f} seconds ---")
    print(f"All results saved in subdirectories under: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()