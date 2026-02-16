import pandas as pd
import numpy as np
import semopy
import os
import sys
from pathlib import Path
import scanpy as sc
import re
from scipy.stats import chi2
from typing import Optional

def sanitize_gene_name(name: str) -> str:
    """Replaces characters invalid for semopy model formulas (like '-') with '_'."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

# Add project root to path to allow importing utility scripts
# TODO: This is a bit of a hack, consider a proper package structure
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from utils.experiment_manager import ExperimentManager
from utils.experiment_analysis import ExperimentAnalyzer

# --- Configuration ---
# This name will define the subdirectory where results are cached.
# Change this for different hypotheses (e.g., when running on a data subset).
HYPOTHESIS_NAME = "IFN_3factor_vs_1factor_all_cells"
EXPERIMENT_NAME = "20250819_232404_fa_100_random_all_filtered_f85f0e07"
FACTORS_TO_ANALYZE = {
    'IFN_Core': ['X_fa_36'],
    'IFN_Quiescent': ['X_fa_46'],
    'IFN_ProInflammatory': ['X_fa_88']
}
PATHWAY_NAME = 'HALLMARK_INTERFERON_ALPHA_RESPONSE' # Assuming this is the context
EXPERIMENTS_BASE_DIR = project_root / "experiments"

# --- Data Subsetting Configuration ---
# Define the different analysis scenarios to run sequentially.
# Each dictionary defines a specific hypothesis to test.
ANALYSIS_SCENARIOS = [
    {
        "name": "all_cells_and_patients",
        "patient_subset": None,
        "cell_type_subset": None,
    },
    {
        "name": "relevant_patients_all_cells",
        # Combined list of patients relevant to factors 36, 46, 88
        "patient_subset": sorted(list(set(['P02', 'P13', 'P05', 'P07', 'P01', 'P06']))),
        "cell_type_subset": None,
    },
    {
        "name": "relevant_patients_and_cell_types",
        "patient_subset": sorted(list(set(['P02', 'P13', 'P05', 'P07', 'P01', 'P06']))),
        "cell_type_subset": ['HSC/MPP1', 'HSC/MPP2', 'LMPP'],
    }
]


def load_data_for_cfa(analyzer: ExperimentAnalyzer, 
                      experiment_id: str, 
                      factors: list, 
                      pathway: str,
                      patient_subset: Optional[list] = None,
                      cell_type_subset: Optional[list] = None
                      ) -> tuple[pd.DataFrame, dict, tuple]:
    """
    Loads and subsets the necessary data for CFA.
    1. Leading edge genes for each factor from GSEA results.
    2. The preprocessed gene expression matrix for the relevant genes.
    Returns the expression dataframe, the filtered gene sets, and the final data shape.
    """
    print("--- Step 1: Loading Data ---")
    
    # 1. Get leading edge genes for each factor using the analyzer
    print(f"Fetching leading edge genes for factors: {', '.join(factors)}...")
    leading_edge_genes = analyzer.analyze_unsupervised_gsea_overlap(
        experiment_id=experiment_id,
        factors=factors,
        pathway_name=pathway
    )

    if not any(leading_edge_genes.values()):
        print("ERROR: Could not find any leading edge genes. Aborting.")
        return pd.DataFrame(), {}, (0, 0)

    # 2. Get the preprocessed gene expression data
    print("Fetching preprocessed gene expression data...")
    try:
        exp = analyzer.experiment_manager.load_experiment(experiment_id)
        preprocessed_data_path = exp.get_path('preprocessed_data')
        
        if not preprocessed_data_path.exists():
            print(f"ERROR: Preprocessed data file not found at {preprocessed_data_path}")
            return pd.DataFrame(), {}, (0, 0)
            
        adata = sc.read_h5ad(preprocessed_data_path)

        # --- Subsetting Data (Optional) ---
        if patient_subset:
            print(f"\nSubsetting data to {len(patient_subset)} patients...")
            adata = adata[adata.obs['patient'].isin(patient_subset)].copy()
        if cell_type_subset:
            print(f"\nSubsetting data to {len(cell_type_subset)} cell types...")
            # Assuming 'predicted.annotation' is the column for cell types
            adata = adata[adata.obs['predicted.annotation'].isin(cell_type_subset)].copy()
        
        final_shape = adata.shape
        print(f"Final data shape after subsetting: {final_shape[0]} cells, {final_shape[1]} genes")

        # --- Sanitize Gene Names for Model Compatibility ---
        # It's crucial to sanitize names in both the expression data and the GSEA gene
        # lists to ensure they match perfectly. The primary issue is that libraries like
        # semopy can misinterpret characters like '-' in model formulas.

        # Sanitize AnnData var_names (in-place)
        original_var_names = adata.var.index.tolist()
        adata.var.index = [sanitize_gene_name(name) for name in original_var_names]
        adata.var_names_make_unique() # Ensures no name collisions after sanitization

        # Sanitize the gene sets derived from GSEA
        sanitized_leading_edge_genes = {}
        for factor, gene_set in leading_edge_genes.items():
            sanitized_leading_edge_genes[factor] = {sanitize_gene_name(gene) for gene in gene_set}
        # --- End Sanitization ---

        # Consolidate all unique genes needed for the models (using sanitized names)
        all_genes = set()
        for gene_set in sanitized_leading_edge_genes.values():
            all_genes.update(gene_set)
        
        # --- DIAGNOSTIC STEP & FILTERING ---
        # Now we compare the sanitized lists to find overlaps
        valid_genes = all_genes.intersection(adata.var_names)
        
        if not valid_genes:
            print("ERROR: None of the leading edge genes were found in the expression data after sanitization.")
            return pd.DataFrame(), {}, (0, 0)
            
        # Filter the sanitized gene sets to only include the valid, overlapping genes
        filtered_gene_sets = {}
        for factor, gene_set in sanitized_leading_edge_genes.items():
            valid_genes_for_factor = gene_set.intersection(valid_genes)
            if valid_genes_for_factor:
                filtered_gene_sets[factor] = valid_genes_for_factor
            else:
                print(f"WARNING: No valid genes found for factor {factor} after filtering. It will be excluded from the model.")
        
        if not filtered_gene_sets:
            print("ERROR: No factors have any overlapping genes with the expression data. Aborting.")
            return pd.DataFrame(), {}, (0, 0)
            
        # Subset the AnnData object to only the genes we will actually use
        adata_subset = adata[:, list(valid_genes)]
        
        # Convert to a pandas DataFrame for semopy
        expression_df = adata_subset.to_df()
        
        print(f"Successfully loaded and filtered expression data for {expression_df.shape[0]} cells and {expression_df.shape[1]} genes.")
        
    except Exception as e:
        print(f"ERROR: Failed to load or process gene expression data. Details: {e}")
        return pd.DataFrame(), {}, (0, 0)
    
    print("Data loading complete.\n")
    return expression_df, filtered_gene_sets, final_shape

def generate_cfa_model_string(factor_gene_map: dict, single_factor_name: str = "IFN_General") -> tuple[str, str]:
    """
    Generates model strings for a multi-factor (A) and single-factor (B) model.
    
    Args:
        factor_gene_map: A dictionary mapping latent factor names to lists of sanitized gene names.
        single_factor_name: The name to use for the combined latent factor in Model B.
        
    Returns:
        A tuple containing the model string for Model A and Model B.
    """
    
    # --- Model A: Multi-Factor Correlated Model ---
    measurement_parts = []
    for factor_name, gene_list in factor_gene_map.items():
        if gene_list:
            genes_str = '+'.join(sorted(list(gene_list)))
            measurement_parts.append(f"        {factor_name} =~ {genes_str}")
    
    factor_names = list(factor_gene_map.keys())
    covariance_parts = []
    if len(factor_names) > 1:
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                covariance_parts.append(f"        {factor_names[i]} ~~ {factor_names[j]}")

    model_a_str = "# Model A: {num_factors}-Factor Correlated Model\n".format(num_factors=len(factor_names))
    model_a_str += "        # Measurement Model\n"
    model_a_str += '\n'.join(measurement_parts)
    if covariance_parts:
        model_a_str += "\n\n        # Latent Factor Covariances\n"
        model_a_str += '\n'.join(covariance_parts)

    # --- Model B: Single-Factor Model ---
    all_genes = set()
    for gene_list in factor_gene_map.values():
        all_genes.update(gene_list)
    
    model_b_str = "# Model B: 1-Factor Model\n"
    model_b_str += "        # Measurement Model\n"
    model_b_str += "        {name} =~ {genes}".format(
        name=single_factor_name,
        genes='+'.join(sorted(list(all_genes)))
    )
    
    return model_a_str, model_b_str


def run_and_evaluate_model(model_str: str, data: pd.DataFrame, model_name: str, cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Fits a CFA model using semopy, prints fit indices, and caches the statistics.
    Returns the stats DataFrame on success, otherwise None.
    """
    print(f"--- Evaluating {model_name} ---")
    
    # Check for cached results first
    if cache_path.exists():
        print(f"Found cached stats for {model_name}. Loading from file.")
        stats = pd.read_csv(cache_path, index_col=0)
        return stats

    if not model_str.strip() or "=~" not in model_str:
        print("Model string is empty or invalid. Skipping.")
        return None
        
    try:
        model = semopy.Model(model_str)
        print(f"Fitting {model_name}... (this may take several minutes depending on data size)")
        model.fit(data)
        
        stats = semopy.calc_stats(model)
        print(f"\nFit Indices for {model_name}:")
        print(stats.T)
        
        # Save the stats to the cache file
        print(f"\nSaving stats to cache: {cache_path}")
        stats.to_csv(cache_path)
        
        return stats
    except Exception as e:
        print(f"ERROR: Could not fit {model_name}. Details: {e}")
        return None

def main():
    """
    Main function to orchestrate the CFA fitting process.
    This function will loop through all defined ANALYSIS_SCENARIOS.
    """
    # --- Initialize Managers ---
    manager = ExperimentManager(base_dir=str(EXPERIMENTS_BASE_DIR))
    analyzer = ExperimentAnalyzer(experiment_manager=manager)
    
    # --- Find Experiment ID ---
    exp_id = None
    for eid in manager.list_experiments():
        if EXPERIMENT_NAME in eid:
            exp_id = eid
            break
            
    if not exp_id:
        print(f"FATAL: Experiment containing name '{EXPERIMENT_NAME}' not found.")
        return
        
    print(f"--- Found Experiment ID: {exp_id} ---")

    # --- Loop Through and Run All Scenarios ---
    for i, scenario in enumerate(ANALYSIS_SCENARIOS):
        hypothesis_name = f"IFN_factors_{scenario['name']}"
        patient_subset = scenario['patient_subset']
        cell_type_subset = scenario['cell_type_subset']
        
        print(f"\n{'='*80}")
        print(f"--- Running Scenario {i+1}/{len(ANALYSIS_SCENARIOS)}: {hypothesis_name} ---")
        print(f"{'='*80}")
        
        # Define cache paths within a hypothesis-specific directory
        exp = manager.load_experiment(exp_id)
        analysis_dir = exp.experiment_dir / "analysis" / "cfa" / hypothesis_name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be cached in: {analysis_dir}")
        
        cache_path_a = analysis_dir / "model_a_stats.csv"
        cache_path_b = analysis_dir / "model_b_stats.csv"
        summary_path = analysis_dir / "hypothesis_summary.txt"

        # Step 1: Load and subset data
        all_factors_to_load = [f for factors in FACTORS_TO_ANALYZE.values() for f in factors]
        expression_df, leading_edge_gene_sets, _ = load_data_for_cfa(
            analyzer, 
            exp_id, 
            all_factors_to_load, 
            PATHWAY_NAME,
            patient_subset=patient_subset,
            cell_type_subset=cell_type_subset
        )

        if expression_df.empty or not leading_edge_gene_sets:
            print(f"Could not load data for scenario '{hypothesis_name}'. Skipping.")
            continue
            
        # Map the loaded genes back to our desired factor structure
        factor_gene_map = {}
        for factor_name, original_factors in FACTORS_TO_ANALYZE.items():
            gene_set = set()
            for original_factor in original_factors:
                gene_set.update(leading_edge_gene_sets.get(sanitize_gene_name(original_factor), set()))
            factor_gene_map[factor_name] = gene_set
            
        # Step 2: Generate Models
        print("\n--- Step 2: Generating CFA Models ---")
        model_a_str, model_b_str = generate_cfa_model_string(factor_gene_map)
        
        # --- Save Hypothesis Summary ---
        # Capture the final, correct data shape from the dataframe that will be used in the model
        final_data_shape = expression_df.shape
        print(f"Saving hypothesis summary to: {summary_path}")
        with open(summary_path, 'w') as f:
            f.write(f"--- CFA Hypothesis Summary ---\n\n")
            f.write(f"Hypothesis Name: {hypothesis_name}\n\n")
            f.write(f"Patient Subset: {patient_subset if patient_subset else 'All'}\n")
            f.write(f"Cell Type Subset: {cell_type_subset if cell_type_subset else 'All'}\n")
            f.write(f"Final Data Shape for Model Fitting: {final_data_shape[0]} cells x {final_data_shape[1]} genes\n\n")
            f.write(f"--- Model A (Multi-Factor) ---\n{model_a_str}\n\n")
            f.write(f"--- Model B (Single-Factor) ---\n{model_b_str}\n")
        
        # Step 3: Run and Evaluate Models
        stats_a = run_and_evaluate_model(model_a_str, expression_df, "Model A (Multi-Factor)", cache_path_a)
        stats_b = run_and_evaluate_model(model_b_str, expression_df, "Model B (Single-Factor)", cache_path_b)
        
        print(f"\n--- Scenario '{hypothesis_name}' complete. ---")
        if stats_a is None or stats_b is None:
            print("Warning: One or both models failed to fit in this scenario.")

    print(f"\n{'='*80}")
    print("All CFA fitting scenarios are complete.")
    print("Results have been saved and can now be analyzed in a separate script or notebook.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
