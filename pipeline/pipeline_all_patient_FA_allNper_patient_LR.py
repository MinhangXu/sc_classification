#!/usr/bin/env python
"""
Pipeline script for running Factor Analysis on all patients followed by
LR_lasso classification on both all patients and individual patients.

This script imports the necessary modules from the sc_classification package
and follows the structure of the repository.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

# Add the project root to Python path for absolute imports
# Assuming this script is in sc_classification/pipeline/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use absolute imports instead of relative imports
from sc_classification.dimension_reduction.factor_analysis import FactorAnalysis
from sc_classification.classification_methods.lr_lasso import LRLasso
from sc_classification.pipeline.all_patients_DR_classification import run_all_patients_pipeline

# Define paths
ADATA_PATH = "/home/minhang/mds_project/data/cohort_adata/mrd_malignant_healthy_scRNA_adata_may1.h5ad"
OUTPUT_DIR = "results/fa_lrlasso_pipeline_may1"
ALL_PATIENTS_DIR = os.path.join(OUTPUT_DIR, "all_patients")
PER_PATIENT_DIR = os.path.join(OUTPUT_DIR, "per_patient")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ALL_PATIENTS_DIR, exist_ok=True)
os.makedirs(PER_PATIENT_DIR, exist_ok=True)

def run_lrlasso_per_patient(adata_fa, alphas=None):
    """
    Run LR-LASSO analysis for each patient separately using factors from all patients.
    
    Parameters:
    - adata_fa: AnnData object with FA results from all patients
    - alphas: Array of regularization strengths
    
    Returns:
    - Dictionary with results for each patient
    """
    print("\nRunning LR-LASSO analysis for each patient separately...")
    
    # Set default alphas if not provided
    if alphas is None:
        alphas = np.logspace(-4, 5, 20)
    
    # Get unique patients
    unique_patients = adata_fa.obs['patient'].unique()
    print(f"Found {len(unique_patients)} unique patients")
    
    # Store metrics for all patients
    all_patient_metrics = []
    
    # Process each patient
    for patient in unique_patients:
        print(f"\nProcessing patient: {patient}")
        
        # Create patient directory
        patient_dir = os.path.join(PER_PATIENT_DIR, f"patient_{patient}")
        os.makedirs(patient_dir, exist_ok=True)
        
        # Subset to current patient
        sub_adata = adata_fa[adata_fa.obs['patient'] == patient].copy()
        print(f"Subsetting patient {patient}: {sub_adata.n_obs} cells")
        
        # Check class distribution
        cn_counts = sub_adata.obs['CN.label'].value_counts()
        print(f"CN.label distribution: {cn_counts}")
        
        # Skip if both classes not present
        if 'cancer' not in cn_counts.index or 'normal' not in cn_counts.index:
            print(f"Skipping patient {patient}: missing one or more classes in CN.label")
            continue
        
        # Initialize LRLasso classifier
        classifier = LRLasso(sub_adata, target_col='CN.label', target_value='cancer')
        
        # Extract factors (X_fa) and target (y)
        X_fa, y, feature_names, valid_X_original = classifier.prepare_data(
            use_factorized=True, factorization_method='X_fa', selected_features=None
        )
        print(f"Patient {patient} data -- X_fa: {X_fa.shape}, y: {y.shape}")
        
        # Skip if only one class present after filtering
        if len(np.unique(y)) < 2:
            print(f"Skipping patient {patient}: only one class present after filtering")
            continue
        
        # Run LR-LASSO classification
        results = classifier.fit_along_regularization_path(
            X_fa, y, feature_names, alphas=alphas, metrics_grouping='patient'
        )
        
        # Save coefficients for this patient
        coefs_df = pd.DataFrame(
            results['coefs'],
            index=feature_names,
            columns=alphas
        )
        coefs_filename = os.path.join(patient_dir, f"results_coefs_{patient}.csv")
        coefs_df.to_csv(coefs_filename, index=True)
        print(f"Saved coefficients to {coefs_filename}")
        
        # Save metrics for this patient
        metrics_df = pd.DataFrame(results['group_metrics_path'])
        metrics_filename = os.path.join(patient_dir, f"results_metrics_{patient}.csv")
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Saved metrics to {metrics_filename}")
        
        # Extract summary metrics for this patient
        all_samples_metrics = next((m for m in results['group_metrics_path'] if m['group'] == 'all_samples'), None)
        if all_samples_metrics:
            patient_metrics = {
                'patient': patient,
                'n_cells': X_fa.shape[0],
                'n_cancer': np.sum(y),
                'n_normal': np.sum(1-y)
            }
            
            # Add metrics for each alpha value
            for key, value in all_samples_metrics.items():
                if key not in ['group', 'fpr', 'tpr', 'thresholds']:
                    patient_metrics[f"{key}"] = value
            
            # Store this patient's metrics
            all_patient_metrics.append(patient_metrics)
    
    # Combine all patient metrics
    if all_patient_metrics:
        all_metrics_df = pd.DataFrame(all_patient_metrics)
        metrics_path = os.path.join(PER_PATIENT_DIR, "all_patient_summary.csv")
        all_metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved combined metrics for all patients to {metrics_path}")
    
    return all_patient_metrics

def generate_best_alpha_summary():
    """
    Generate a summary of best alpha values per patient based on AUC.
    """
    print("\nGenerating summary of optimal alpha values per patient")
    
    # Get patient directories
    patient_dirs = [d for d in os.listdir(PER_PATIENT_DIR) if d.startswith("patient_")]
    
    if patient_dirs:
        all_summaries = []
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.replace("patient_", "")
            metrics_file = os.path.join(PER_PATIENT_DIR, patient_dir, f"results_metrics_{patient_id}.csv")
            
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
                patient_metrics = metrics_df[metrics_df['group'] == 'all_samples']
                
                if 'roc_auc' in patient_metrics.columns and not patient_metrics.empty:
                    # Find alpha with best AUC
                    best_idx = patient_metrics['roc_auc'].idxmax()
                    best_alpha = patient_metrics.loc[best_idx, 'alpha']
                    best_auc = patient_metrics.loc[best_idx, 'roc_auc']
                    best_accuracy = patient_metrics.loc[best_idx, 'overall_accuracy']
                    
                    # Add to summary
                    all_summaries.append({
                        'patient': patient_id,
                        'best_alpha': best_alpha,
                        'best_auc': best_auc,
                        'best_accuracy': best_accuracy,
                        'n_cancer': patient_metrics.loc[best_idx, 'minority_num'] if 'minority_num' in patient_metrics.columns else None,
                        'n_normal': patient_metrics.loc[best_idx, 'majority_num'] if 'majority_num' in patient_metrics.columns else None
                    })
        
        # Save summary
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            summary_path = os.path.join(OUTPUT_DIR, "best_alphas_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved best alpha summary to {summary_path}")
            
            # Create a pivot table comparing AUC values across patients
            pivot_auc = summary_df.pivot(index='patient', columns=['best_alpha'], values='best_auc')
            pivot_path = os.path.join(OUTPUT_DIR, "patient_auc_comparison.csv")
            pivot_auc.to_csv(pivot_path)
            print(f"Saved patient AUC comparison to {pivot_path}")

def main():
    """Main pipeline execution function."""
    print("Starting Factor Analysis + LR_lasso pipeline")
    print(f"Reading data from: {ADATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Step 1: Read input data
    start_time = time.time()
    adata = sc.read_h5ad(ADATA_PATH)
    print(f"Read AnnData: {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"Patient distribution: {adata.obs['patient'].value_counts().shape[0]} unique patients")
    
    # Step 2 & 3: Run all patients pipeline (FA + LR on all patients)
    # Using the existing function from all_patients_DR_classification.py
    print("\nRunning Factor Analysis + LR_lasso on all patients combined...")
    all_patients_results = run_all_patients_pipeline(
        adata,
        dr_method='factor_analysis',
        n_components=100,
        use_row_weights=False,  # No row weighting as specified
        alphas=np.logspace(-4, 5, 20),
        output_dir=ALL_PATIENTS_DIR
    )
    
    # Step 4: Run LR-LASSO for each patient separately using the FA results
    adata_fa = all_patients_results['adata_dr']
    per_patient_metrics = run_lrlasso_per_patient(adata_fa, alphas=np.logspace(-4, 5, 20))
    
    # Step 5: Generate summary of best alpha values per patient
    generate_best_alpha_summary()
    
    elapsed = time.time() - start_time
    print(f"\nPipeline completed in {elapsed:.2f} seconds")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()