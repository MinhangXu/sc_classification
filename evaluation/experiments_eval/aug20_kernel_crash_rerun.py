import sys
from pathlib import Path

# --- Add project root to system path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.experiment_manager import ExperimentManager
from utils.experiment_analysis import ExperimentAnalyzer

def run_gsea_analysis(analyzer, exp_id, patient_indices_dict):
    """Helper function to run only the GSEA analysis."""
    print("\n--- 3. Running GSEA on predictive factor loadings ---")
    try:
        for patient_id, indices_to_check in patient_indices_dict.items():
            print(f"  Running GSEA for patient: {patient_id}")
            for alpha_idx in indices_to_check:
                print(f"    - GSEA for alpha index: {alpha_idx}")
                analyzer.run_gsea_on_predictive_loading(
                    experiment_id=exp_id,
                    patient_id=patient_id,
                    alpha_index=alpha_idx
                )
        print("  GSEA analysis complete.")
    except Exception as e:
        print(f"  ERROR during GSEA for {exp_id}: {e}")

def run_full_downstream_analysis(analyzer, exp_id, patient_indices_dict):
    """Helper function to run the full downstream analysis suite."""
    # --- 1. Analyze Classification Transitions ---
    print("\n--- 1. Analyzing classification transitions ---")
    try:
        for patient_id, indices_to_check in patient_indices_dict.items():
            print(f"  Analyzing transitions for patient: {patient_id}")
            analyzer.analyze_classification_transitions(
                experiment_id=exp_id,
                patient_id=patient_id,
                indices_to_check=indices_to_check
            )
        print("  Transition analysis complete.")
    except Exception as e:
        print(f"  ERROR during transition analysis for {exp_id}: {e}")

    # --- 2. Generate Classification UMAP Reports ---
    print("\n--- 2. Generating classification UMAP reports ---")
    try:
        analyzer.generate_classification_umap_report(
            experiment_id=exp_id,
            patient_reg_strength_indices=patient_indices_dict,
            static_umap_rep='X_multivi'
        )
        print("  UMAP report generation complete.")
    except Exception as e:
        print(f"  ERROR during UMAP report generation for {exp_id}: {e}")
    
    # --- 3. Run GSEA on Predictive Loadings ---
    run_gsea_analysis(analyzer, exp_id, patient_indices_dict)


def main():
    """
    Reruns failed/pending analyses from the Aug 20 notebook.
    - Reruns GSEA for the 'per_patient_all_filtered' experiment.
    - Runs all downstream analyses for the 'per_patient_deg_weak' experiment.
    """
    print("--- Setting up analysis environment ---")

    BASE_EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

    experiments_to_run = {
        "per_patient_all_filtered": {
            "id": "20250819_232404_fa_100_random_all_filtered_f85f0e07",
            "label": "Per-Patient (All Filtered Genes)",
            "indices": {
                'P01': list(range(10, 14)), 'P02': list(range(12, 17)),
                'P03': list(range(12, 17)), 'P04': list(range(11, 16)),
                'P05': list(range(10, 14)), 'P06': list(range(9, 14)),
                'P07': [10, 11, 12, 13], 'P09': list(range(12, 18)),
                'P13': list(range(11, 14))
            },
            "analysis_type": "gsea_only"
        },
        "per_patient_deg_weak": {
            "id": "20250820_024407_fa_100_random_deg_weak_screen_bf7e9669",
            "label": "Per-Patient (DEG Weak Screen)",
            "indices": {
                'P01': list(range(9, 14)), 'P02': list(range(11, 17)),
                'P03': list(range(10, 17)), 'P04': list(range(10, 15)),
                'P05': list(range(10, 14)), 'P06': [10, 11, 12, 13],
                'P07': [10, 11, 12, 13], 'P09': list(range(12, 18)),
                'P13': [10, 11, 12, 13]
            },
            "analysis_type": "full"
        }
    }

    manager = ExperimentManager(BASE_EXPERIMENT_DIR)
    analyzer = ExperimentAnalyzer(manager)

    print("Setup complete. Starting analysis...")

    for key, exp_data in experiments_to_run.items():
        exp_id = exp_data["id"]
        patient_indices_dict = exp_data["indices"]
        
        print(f"\n{'='*80}")
        print(f"--- Processing: {exp_data['label']} ({exp_id}) ---")
        print(f"--- Analysis Type: {exp_data['analysis_type']} ---")
        print(f"{'='*80}")

        if exp_data['analysis_type'] == 'gsea_only':
            run_gsea_analysis(analyzer, exp_id, patient_indices_dict)
        elif exp_data['analysis_type'] == 'full':
            run_full_downstream_analysis(analyzer, exp_id, patient_indices_dict)

    print(f"\n{'='*80}")
    print("--- All Analyses Finished ---")
    print("Check the 'analysis' directory within each experiment folder for results.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
