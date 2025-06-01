#!/usr/bin/env python
"""
Pipeline script for:
1. Running Factor Analysis on PREPROCESSED (MRD, HVG, standardized) data.
2. For each patient:
   a. Subsetting the FA-transformed data.
   b. Performing PER-PATIENT donor cell downsampling (Random method implemented here).
   c. Running LR-Lasso classification using the (downsampled) donor factors 
      and all recipient factors.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import pickle

# --- Configuration ---
PREPROCESSED_ADATA_PATH = '/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_mrd_hvg_std_may31.h5ad' 
BASE_OUTPUT_DIR = "results_fa_lrlasso_mrd_pipeline_donor_downsampled_projection_ready_may31" 

PATIENT_COL = 'patient'
TARGET_COL = 'CN.label' 
TARGET_VALUE_POSITIVE = 'cancer' # Malignant/recipient cells
# Assuming 'normal' cells in CN.label correspond to donor cells for classification target
# And that donor_recipient_col helps identify donor cells for downsampling
DONOR_RECIPIENT_COL = 'source' # Your column: 'donor', 'recipient'
DONOR_LABEL = 'donor'
RECIPIENT_LABEL = 'recipient' # Not strictly needed if CN.label handles classification y

# Factor Analysis parameters
N_COMPONENTS_LIST_FOR_FA = [20, 50, 100] 
FA_RANDOM_STATE = 0
FA_SVD_METHOD = 'lapack' 

# LR-Lasso parameters
ALPHAS_LASSO = np.logspace(-4, 5, 20)

# Per-Patient Donor Downsampling Parameters (Random Method Example)
DOWNSAMPLE_METHOD = 'random' # Options: 'random', 'centroid', 'density', or 'none'
# Parameters for Random Downsampling (adapt if using other methods)
DEFAULT_TARGET_DONOR_FRACTION = 0.5 
TARGET_DONOR_RECIPIENT_RATIO_THRESHOLD = 10.0 
MIN_CELLS_PER_TYPE_AFTER_DOWNSAMPLE = 20 # For downsampling function
CELL_TYPE_COL_FOR_DOWNSAMPLING = 'predicted.annotation' # Cell type column for stratified downsampling


# --- Setup Paths for Imports ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..')) 
if not os.path.isdir(os.path.join(SCRIPT_DIR, "sc_classification")) and os.path.isdir(os.path.join(PROJECT_ROOT, "sc_classification")):
    sys.path.insert(0, PROJECT_ROOT)
else: 
    sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

try:
    from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis 
    from sc_classification.classification_methods.lr_lasso import LRLasso
    # You'll need your downsampling functions available here too if called directly
    # For simplicity, this example implements random downsampling inline.
    # If using your more complex ones, ensure they are importable or defined.
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Inline Random Downsampling Function (or import your preferred ones) ---
def random_downsample_donor_cells(adata_donor_subset, cell_type_col, 
                                  target_fraction=0.5, min_cells_per_type=5, random_state=42):
    """ Randomly downsamples cells within each cell type in adata_donor_subset.
        Returns a boolean mask aligned with adata_donor_subset.obs.index. """
    np.random.seed(random_state)
    kept_indices = []
    for cell_type in adata_donor_subset.obs[cell_type_col].unique():
        type_adata = adata_donor_subset[adata_donor_subset.obs[cell_type_col] == cell_type]
        n_cells_type = type_adata.n_obs
        n_to_keep_frac = int(n_cells_type * target_fraction)
        n_to_keep = max(min_cells_per_type, n_to_keep_frac)
        n_to_keep = min(n_to_keep, n_cells_type) if n_cells_type > 0 else 0

        if n_to_keep > 0:
            chosen_indices = np.random.choice(type_adata.obs.index, size=n_to_keep, replace=False)
            kept_indices.extend(chosen_indices.tolist())
            
    return pd.Series(adata_donor_subset.obs.index.isin(kept_indices), index=adata_donor_subset.obs.index)

def main():
    print(f"Starting FA (on pre-std HVGs) + Per-Patient Donor Downsampling + LR-Lasso pipeline.")
    print(f"Donor downsampling method: {DOWNSAMPLE_METHOD}")
    start_time_pipeline = time.time()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {BASE_OUTPUT_DIR}")

    print(f"\nLoading PREPROCESSED AnnData from: {PREPROCESSED_ADATA_PATH}")
    try:
        adata_mrd_preprocessed = sc.read_h5ad(PREPROCESSED_ADATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Preprocessed AnnData file not found: {PREPROCESSED_ADATA_PATH}. Exiting.")
        return
    print(f"Preprocessed AnnData loaded: {adata_mrd_preprocessed.shape[0]} cells, {adata_mrd_preprocessed.shape[1]} HVGs.")

    if adata_mrd_preprocessed.n_obs == 0:
        print("Preprocessed AnnData is empty. Exiting.")
        return
    
    # Save the HVG list from the preprocessed AnnData (used for ALL FA runs)
    # This assumes adata_mrd_preprocessed is already subsetted to HVGs
    #hvg_list_global = adata_mrd_preprocessed.var_names.tolist()
    #hvg_output_dir = os.path.join(BASE_OUTPUT_DIR, "global_hvg_info") # Save it at the base level
    #os.makedirs(hvg_output_dir, exist_ok=True)
    #hvg_list_path = os.path.join(hvg_output_dir, "hvg_gene_list.csv")
    #pd.DataFrame(hvg_list_global, columns=['gene']).to_csv(hvg_list_path, index=False)
    #print(f"Saved global HVG list (order for FA) to {hvg_list_path}")
    
    # --- Global Factor Analysis (once on all preprocessed MRD cells) ---
    # Loop over n_factors for FA will be outside the patient loop
    for n_factors in N_COMPONENTS_LIST_FOR_FA:
        print(f"\n--- GLOBAL FACTOR ANALYSIS: Processing for {n_factors} factors ---")
        current_output_dir_n_factors = os.path.join(BASE_OUTPUT_DIR, f"n_factors_{n_factors}")
        os.makedirs(current_output_dir_n_factors, exist_ok=True)
        
        fa_processing_start_time = time.time()
        # Use a copy of the preprocessed data for each FA model with different n_factors
        adata_for_current_fa_model = adata_mrd_preprocessed.copy() 
        
        print(f"Running Factor Analysis with {n_factors} components on {adata_for_current_fa_model.n_obs} cells and {adata_for_current_fa_model.n_vars} pre-standardized HVGs...")
        fa_model_instance = FactorAnalysis()
        
        # This adata_fa_transformed contains ALL cells with their factor scores
        adata_fa_transformed = fa_model_instance.fit_transform(
            adata_for_current_fa_model, 
            n_components=n_factors, 
            random_state=FA_RANDOM_STATE,
            standardize_input=False, # Input is ALREADY standardized
            svd_method=FA_SVD_METHOD,
            save_fitted_models=False # Not saving the model object itself for now
        )
        
        # --- Save FA model, scaler, and precise HVG list used for this FA run ---
        model_objects_save_dir = os.path.join(current_output_dir_n_factors, "projection_model_objects")
        os.makedirs(model_objects_save_dir, exist_ok=True)

        # Retrieve from temporary storage in .uns
        fa_model_to_save = adata_fa_transformed.uns.get('_temp_fa_model_obj')
        # hvg_list_for_this_fa = adata_fa_transformed.var_names.tolist() # Genes in this specific anndata after FA

        if fa_model_to_save:
            fa_model_path = os.path.join(model_objects_save_dir, f"fa_model_{n_factors}factors.pkl")
            with open(fa_model_path, 'wb') as f:
                pickle.dump(fa_model_to_save, f)
            print(f"  Saved FA model object to {fa_model_path}")
            adata_fa_transformed.uns['fa']['saved_model_paths']['fa_model'] = fa_model_path


        # --- Write FA summary (same logic as before) ---
        fa_summary_path = os.path.join(current_output_dir_n_factors, "fa_summary.txt")
        # ... (content of fa_summary.txt writing - refer to previous script version) ...
        with open(fa_summary_path, "w") as f:
            f.write(f"Factor Analysis Summary for n_components = {n_factors} on {adata_fa_transformed.n_vars} HVGs\n")
            f.write(f"Input AnnData was pre-standardized externally.\n")
            # f.write(f"Internal standardization by FactorAnalysis class call: {adata_fa_transformed.uns['fa']['standardized_input']}\n")
            # f.write(f"SVD Method Used (in SklearnFA): {adata_fa_transformed.uns['fa'].get('svd_method_used', FA_SVD_METHOD)}\n")
            f.write("\n--- Factor Score Variances ---\n") # ... and so on for other metrics
            f.write(f"Sum of Factor Score Variances: {adata_fa_transformed.uns['fa']['sum_factor_score_variances']:.4f}\n")
            comm_summary = adata_fa_transformed.uns['fa']['communalities_per_gene_summary']
            f.write("Communalities per Gene Summary (Mean): {comm_summary['mean']:.4f}\n")
            f.write(f"SS Loadings per Factor (V_j) sum: {np.sum(adata_fa_transformed.uns['fa']['ss_loadings_per_factor']):.4f}\n")

        print(f"Saved FA summary to {fa_summary_path}")
        # Communalities CSV also saved once per n_factors setting
        communalities_df = pd.DataFrame({'gene': adata_fa_transformed.var_names, 'communality': adata_fa_transformed.var['communality']})
        communalities_csv_path = os.path.join(current_output_dir_n_factors, f"gene_communalities_factors_{n_factors}.csv")
        communalities_df.to_csv(communalities_csv_path, index=False)
        print(f"Saved per-gene communalities to {communalities_csv_path}")

        fa_duration = time.time() - fa_processing_start_time
        print(f"Global Factor Analysis for {n_factors} factors completed in {fa_duration:.2f} seconds.")

        # --- Per-Patient LR-Lasso with Donor Downsampling ---
        unique_patients = sorted(list(adata_fa_transformed.obs[PATIENT_COL].unique())) # From FA transformed data

        for patient_id in unique_patients:
            patient_classification_start_time = time.time()
            # Create output dir for this patient UNDER the current n_factors directory
            patient_output_dir = os.path.join(current_output_dir_n_factors, f"patient_{patient_id}")
            os.makedirs(patient_output_dir, exist_ok=True)

            # Subset the globally FA-transformed data for the current patient
            adata_patient_with_fa = adata_fa_transformed[adata_fa_transformed.obs[PATIENT_COL] == patient_id].copy()
            
            print(f"\n  Processing Patient ID: {patient_id} (for {n_factors} factors)")
            print(f"  Original cells for patient {patient_id}: {adata_patient_with_fa.n_obs}")

            # --- **NEW**: Per-Patient Donor Cell Downsampling ---
            # Identify donor cells within this patient's subset
            donor_mask_in_patient = adata_patient_with_fa.obs[DONOR_RECIPIENT_COL] == DONOR_LABEL
            adata_donor_patient_subset = adata_patient_with_fa[donor_mask_in_patient].copy()
            
            n_donor_patient_original = adata_donor_patient_subset.n_obs
            n_recipient_patient_original = adata_patient_with_fa.n_obs - n_donor_patient_original
            print(f"    Original donor: {n_donor_patient_original}, recipient: {n_recipient_patient_original} for patient {patient_id}")

            adata_for_classifier = adata_patient_with_fa # Default to using all cells for this patient

            if DOWNSAMPLE_METHOD != 'none' and n_donor_patient_original > 0:
                # Determine target fraction for this patient
                patient_target_fraction = DEFAULT_TARGET_DONOR_FRACTION
                if n_recipient_patient_original > 0:
                    current_ratio = n_donor_patient_original / n_recipient_patient_original
                    if current_ratio <= TARGET_DONOR_RECIPIENT_RATIO_THRESHOLD:
                        patient_target_fraction = 1.0 # Keep all or most
                    else:
                        target_n_donors = int(n_recipient_patient_original * TARGET_DONOR_RECIPIENT_RATIO_THRESHOLD)
                        if n_donor_patient_original > 0:
                             patient_target_fraction = target_n_donors / n_donor_patient_original
                             patient_target_fraction = max(0, min(1.0, patient_target_fraction))
                elif n_donor_patient_original > 0: # No recipients, only donors
                    patient_target_fraction = DEFAULT_TARGET_DONOR_FRACTION
                
                print(f"    Applying '{DOWNSAMPLE_METHOD}' downsampling to {n_donor_patient_original} donor cells. Target fraction: {patient_target_fraction:.2f}")

                if DOWNSAMPLE_METHOD == 'random':
                    # This mask is aligned with adata_donor_patient_subset.obs.index
                    kept_donor_mask_on_donor_subset = random_downsample_donor_cells(
                        adata_donor_patient_subset, 
                        CELL_TYPE_COL_FOR_DOWNSAMPLING, # Use the appropriate cell type column
                        target_fraction=patient_target_fraction,
                        min_cells_per_type=MIN_CELLS_PER_TYPE_AFTER_DOWNSAMPLE
                    )
                # Add elif for 'centroid', 'density' if you import/define those functions
                # elif DOWNSAMPLE_METHOD == 'centroid':
                #     kept_donor_mask_on_donor_subset = centroid_percentile_downsample(...)
                else:
                    print(f"    Warning: Unknown DOWNSAMPLE_METHOD '{DOWNSAMPLE_METHOD}'. Using all donor cells.")
                    kept_donor_mask_on_donor_subset = pd.Series(True, index=adata_donor_patient_subset.obs.index)

                # Get the indices of donor cells that were kept
                kept_donor_indices = adata_donor_patient_subset.obs.index[kept_donor_mask_on_donor_subset]
                
                # Combine kept donors with all recipients for this patient
                recipient_indices = adata_patient_with_fa.obs.index[~donor_mask_in_patient]
                final_cell_indices_for_classifier = recipient_indices.union(kept_donor_indices)
                
                adata_for_classifier = adata_patient_with_fa[final_cell_indices_for_classifier].copy()
                print(f"    After downsampling, using {adata_for_classifier.obs[donor_mask_in_patient[final_cell_indices_for_classifier]].shape[0]} donor cells and {adata_for_classifier.obs[~donor_mask_in_patient[final_cell_indices_for_classifier]].shape[0]} recipient cells for classifier.")
            
            # --- LR-Lasso on (potentially downsampled) patient data ---
            if adata_for_classifier.n_obs < 2 or len(adata_for_classifier.obs[TARGET_COL].unique()) < 2:
                 print(f"  Skipping patient {patient_id} (for {n_factors} factors): not enough cells or classes after potential downsampling.")
                 continue
            
            # Ensure TARGET_COL has at least two unique values for classification
            class_counts_classifier = adata_for_classifier.obs[TARGET_COL].value_counts()
            if not (TARGET_VALUE_POSITIVE in class_counts_classifier and class_counts_classifier.get(TARGET_VALUE_POSITIVE, 0) > 0 and \
                    any(val > 0 for key, val in class_counts_classifier.items() if key != TARGET_VALUE_POSITIVE)): # Check for at least one other class
                print(f"  Skipping patient {patient_id} (for {n_factors} factors): does not have at least two distinct classes with positive counts in '{TARGET_COL}' for classifier input.")
                continue

            classifier = LRLasso(adata_for_classifier, target_col=TARGET_COL, target_value=TARGET_VALUE_POSITIVE)
            print(f"  Preparing data for LR-Lasso for patient {patient_id} from {adata_for_classifier.n_obs} cells...")
            X_patient_factors, y_patient_labels, feature_names_factors, _ = classifier.prepare_data(
                use_factorized=True, factorization_method='X_fa' 
            )
            
            if X_patient_factors is None or y_patient_labels is None or X_patient_factors.shape[0] == 0 or len(np.unique(y_patient_labels)) < 2:
                print(f"  Data preparation for LR-Lasso yielded insufficient data for patient {patient_id}. Skipping.")
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
            print(f"  Completed processing for patient {patient_id} ({n_factors} factors, downsampling: {DOWNSAMPLE_METHOD}) in {patient_duration:.2f} seconds.")

    pipeline_duration = time.time() - start_time_pipeline
    print(f"\n--- Pipeline completed in {pipeline_duration:.2f} seconds ---")
    print(f"All results saved in subdirectories under: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()