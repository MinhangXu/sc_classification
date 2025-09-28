import sys
from pathlib import Path

# --- Add project root to system path ---
# The script is in evaluation/experiments_eval/, so we need to go up 3 levels
# to reach the project root 'sc_classification/'.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.experiment_manager import ExperimentManager
from utils.experiment_analysis import ExperimentAnalyzer

def main():
    """
    Runs the downstream analysis pipeline (transitions, UMAPs, GSEA)
    for the two per-patient experiments.
    """
    print("--- Setting up analysis environment ---")

    # --- Base Directory ---
    BASE_EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

    # --- Experiment Metadata for Per-Patient Runs ---
    per_patient_experiments = {
        "per_patient_all_filtered": {
            "id": "20250815_004248_fa_100_random_all_filtered_f77df82f",
            "label": "Per-Patient (All Filtered Genes)",
            "indices": {
                'P01': list(range(8, 15)), 'P02': list(range(12, 17)),
                'P03': [17], 'P04': list(range(9, 15)),
                'P05': list(range(10, 14)), 'P06': list(range(9, 14)),
                'P07': [11, 12, 13], 'P09': list(range(12, 17)),
                'P13': list(range(10, 14))
            }
        },
        "per_patient_deg_weak": {
            "id": "20250815_005818_fa_100_random_deg_weak_screen_5b5a8e81",
            "label": "Per-Patient (DEG Weak Screen)",
            "indices": {
                'P01': list(range(10, 15)), 'P02': list(range(13, 17)),
                'P03': list(range(11, 18)), 'P04': list(range(10, 15)),
                'P05': list(range(9, 15)), 'P06': [10, 11, 12, 13],
                'P07': [10, 11, 12, 13], 'P09': list(range(11, 18)),
                'P13': [10, 11, 12, 13]
            }
        }
    }

    # --- Initialize Managers ---
    manager = ExperimentManager(BASE_EXPERIMENT_DIR)
    analyzer = ExperimentAnalyzer(manager)

    print("Setup complete. Starting analysis...")

    # --- Main Analysis Loop ---
    for key, exp_data in per_patient_experiments.items():
        exp_id = exp_data["id"]
        patient_indices_dict = exp_data["indices"]
        
        print(f"\n{'='*80}")
        print(f"--- Starting Downstream Analysis for: {exp_data['label']} ({exp_id}) ---")
        print(f"{'='*80}")

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
                static_umap_rep='X_fa'  # Corrected to use FA embedding
            )
            print("  UMAP report generation complete.")
        except Exception as e:
            print(f"  ERROR during UMAP report generation for {exp_id}: {e}")

        # --- 3. Run GSEA on Predictive Loadings ---
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

    print(f"\n{'='*80}")
    print("--- All Analyses Finished ---")
    print("Check the 'analysis' directory within each experiment folder for results.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
