import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sc_classification.classification_methods.lr_lasso import LRLasso

def run_single_patient_classification_logic(
    adata_patient_mrd, # This is already subsetted for patient and MRD, and NaNs in target_col are handled
    patient_id,
    target_col='CN.label',
    target_value_positive='cancer',
    latent_representation_key='X_multivi',
    alphas=None,
    output_dir_patient_specific="" # Full path for this patient's output
):
    """
    Core classification logic for a single patient's pre-filtered MRD AnnData.
    """
    # Check for at least two classes (should be guaranteed by caller if this function is used internally)
    if len(adata_patient_mrd.obs[target_col].unique()) < 2:
        print(f"  Patient {patient_id} at MRD does not have at least two unique classes in '{target_col}'. Values: {adata_patient_mrd.obs[target_col].unique()}. Skipping classification logic.")
        return {
            'patient_id': patient_id,
            'classification_skipped_one_class_in_logic': True
        }

    classifier = LRLasso(adata_patient_mrd, target_col=target_col, target_value=target_value_positive)
    
    print(f"  Preparing data for patient {patient_id} classifier...")
    X, y, feature_names, _ = classifier.prepare_data(
        use_factorized=True,
        factorization_method=latent_representation_key 
    )
    
    if X is None or y is None or X.shape[0] == 0:
        print(f"  Data preparation yielded no data for patient {patient_id}. Skipping.")
        return {'patient_id': patient_id, 'classification_skipped_no_data_post_prep': True}
    if len(np.unique(y)) < 2:
        print(f"  Patient {patient_id} (after LRLasso prepare_data) does not have both normal and cancer cells. Values: {np.unique(y)}. Skipping.")
        return {'patient_id': patient_id, 'classification_skipped_one_class_post_prep': True}
        
    if alphas is None:
        alphas = np.logspace(-4, 4, 20) # Default alphas for LR-Lasso
    
    print(f"  Running LR-Lasso for patient {patient_id} using {len(feature_names)} features ({latent_representation_key})...")
    results = classifier.fit_along_regularization_path(
        X, y, feature_names, alphas=alphas, metrics_grouping='patient' 
    )
    
    print(f"  Saving classification results for patient {patient_id}...")
    coefs_df = pd.DataFrame(results['coefs'], index=feature_names, columns=alphas)
    coefs_filename = os.path.join(output_dir_patient_specific, f"results_coefs_{latent_representation_key}_{patient_id}.csv")
    coefs_df.to_csv(coefs_filename, index=True)
    print(f"    Coefficients saved to: {coefs_filename}")
    
    metrics_df = pd.DataFrame(results['group_metrics_path'])
    metrics_filename = os.path.join(output_dir_patient_specific, f"results_metrics_{latent_representation_key}_{patient_id}.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"    Metrics saved to: {metrics_filename}")

    
    return {
        'patient_id': patient_id,
        'results_summary': f"Processed successfully, metrics and coefs saved."
    }

def main_classification_pipeline(
    adata_corrected_path,
    timepoint_type_col='timepoint_type',
    mrd_label='MRD',
    target_col='CN.label',
    output_dir_base="results_multivi_classification"
):
    """
    Main pipeline to load data and run classification for all eligible patients.
    """
    print(f"--- Starting Main Classification Pipeline using MULTIVI Latent Space ---")
    print(f"Loading full corrected AnnData from: {adata_corrected_path}")
    try:
        adata_full = sc.read_h5ad(adata_corrected_path)
    except FileNotFoundError:
        print(f"ERROR: AnnData file not found at {adata_corrected_path}. Exiting.")
        return
    print(f"Full corrected AnnData loaded: {adata_full.shape}")

    # --- Pre-filter for MRD timepoints and valid CN.labels ---
    print(f"\nFiltering for '{mrd_label}' timepoints and non-NaN '{target_col}' labels globally...")
    mrd_mask = adata_full.obs[timepoint_type_col] == mrd_label
    valid_label_mask = ~adata_full.obs[target_col].isna()
    
    adata_mrd_all_patients = adata_full[mrd_mask & valid_label_mask].copy()

    if adata_mrd_all_patients.n_obs == 0:
        print(f"No cells found for timepoint '{mrd_label}' with valid '{target_col}' labels. Exiting.")
        return
    print(f"Total cells for MRD analysis across all patients: {adata_mrd_all_patients.n_obs}")

    unique_patients = sorted(list(adata_mrd_all_patients.obs['patient'].unique()))
    print(f"Found {len(unique_patients)} unique patients with MRD data and valid labels: {unique_patients}")

    all_results_summary = []

    for patient_id in unique_patients:
        print(f"\n>>> Processing Patient: {patient_id} <<<")
        
        adata_patient_mrd = adata_mrd_all_patients[adata_mrd_all_patients.obs['patient'] == patient_id].copy()
        print(f"  Patient {patient_id} MRD data: {adata_patient_mrd.n_obs} cells")

        if adata_patient_mrd.n_obs == 0:
            print(f"  Skipping patient {patient_id}: No cells after patient-specific MRD subsetting (should not happen if previous filter was correct).")
            all_results_summary.append({'patient_id': patient_id, 'status': 'skipped_no_cells_in_subset'})
            continue
            
        cn_counts = adata_patient_mrd.obs[target_col].value_counts()
        print(f"  CN.label distribution for patient {patient_id}: {cn_counts.to_dict()}")
        
        if not ('cancer' in cn_counts.index and 'normal' in cn_counts.index and cn_counts['cancer'] > 0 and cn_counts['normal'] > 0):
            print(f"  Skipping patient {patient_id}: does not have both 'cancer' and 'normal' cells with counts > 0 in MRD data.")
            all_results_summary.append({'patient_id': patient_id, 'status': 'skipped_missing_both_classes'})
            continue
        
        patient_specific_output_dir = os.path.join(output_dir_base, f"patient_{patient_id}")
        os.makedirs(patient_specific_output_dir, exist_ok=True)

        result = run_single_patient_classification_logic(
            adata_patient_mrd=adata_patient_mrd,
            patient_id=patient_id,
            target_col=target_col, # Passed through
            output_dir_patient_specific=patient_specific_output_dir
            # alphas can be passed here if specific values are desired per patient or globally
        )
        all_results_summary.append(result)

    print("\n--- Main Classification Pipeline Finished ---")
    print("Summary of processing:")
    for res_sum in all_results_summary:
        print(f"  Patient {res_sum.get('patient_id')}: {res_sum.get('status') or res_sum.get('results_summary') or 'No specific result message'}")

# --- Example Main Execution Logic (Call this from your main script/notebook) ---
if __name__ == '__main__':
    # Define the path to your AnnData object that contains:
    # - .X with batch-corrected RNA (can be dense or sparse)
    # - .obsm['X_multivi'] with the 23-dimensional latent space
    # - .obs with 'patient', 'timepoint_type', and 'CN.label'
    adata_corrected_path = '/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad' # Update this path
    
    if not os.path.exists(adata_corrected_path):
        print(f"FATAL ERROR: Input AnnData file for classification not found at {adata_corrected_path}")
        print("Please ensure you have generated this file (containing corrected RNA, protein, and X_multivi) and that the path is correct.")
        print("If you ran the sparsification script, make sure to use the output path from that script (e.g., ending in _final_sparseRNA.h5ad).")
    else:
        main_classification_pipeline(adata_corrected_path=adata_corrected_path)