import scvi
import torch
import os
import numpy as np
import pandas as pd # Though not directly used for DataFrame creation here, often good to have
import anndata
import scanpy as sc # For reading AnnData, good practice

def generate_and_add_latent_space():
    """
    Loads a pre-trained scvi-tools MULTIVI model and the original AnnData object
    it was trained on. Generates the latent representation using a manual
    GPU tensor movement loop. Then, loads the AnnData object containing
    batch-corrected RNA and adds this latent representation to it.
    """
    print("Starting script to generate latent space and add to corrected RNA AnnData...")

    # --- CUDA and System Checks ---
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        print("Error: CUDA not available. This script requires GPU.")
        return

    # --- scvi-tools Settings ---
    try:
        # Match the setting used in your previous successful script if it mattered
        # For latent space generation, num_workers=0 might be simpler as it avoids multiprocessing complexities
        # if data loading isn't the bottleneck. Test what works best for stability/speed.
        scvi.settings.dl_num_workers = 8 
        print(f"Set scvi.settings.dl_num_workers to: {scvi.settings.dl_num_workers}")
    except Exception as e:
        print(f"Could not set scvi.settings.dl_num_workers: {e}. Using default.")

    # --- File Paths ---
    base_dir = '/home/minhang/mds_project/data/cohort_adata/multiVI_model'
    original_adata_path = os.path.join(base_dir, 'adata.h5ad') # Original adata used for model training
    model_pt_path = os.path.join(base_dir, 'model.pt')
    # Path to the AnnData file you created that has the corrected RNA
    corrected_rna_adata_path = os.path.join(base_dir, 'adata_multivi_corrected_rna.h5ad')

    print(f"Original AnnData path (for model setup and latent generation): {original_adata_path}")
    print(f"Model path: {model_pt_path}")
    print(f"Corrected RNA AnnData path (to update): {corrected_rna_adata_path}")

    # --- Load Original AnnData (for model context) ---
    print("Loading original AnnData object for model setup...")
    adata_original_for_model = sc.read_h5ad(original_adata_path)
    adata_original_for_model.var_names_make_unique()
    print("Original AnnData loaded successfully.")

    # --- Setup AnnData for scvi-tools (using the original AnnData) ---
    print("Setting up original AnnData for MULTIVI...")
    # This step is crucial so the model_shell is initialized correctly
    # and the AnnDataManager within the model refers to this adata_original_for_model
    scvi.model.MULTIVI.setup_anndata(
        adata_original_for_model, # Use the original adata here
        batch_key="Tech",
        protein_expression_obsm_key="ADT",
        categorical_covariate_keys=["sample"]
    )
    print("Original AnnData setup complete.")

    # --- Initialize Model Shell (using the original AnnData) ---
    print("Initializing MULTIVI model shell...")
    # Parameters will be inferred from adata_original_for_model.uns
    n_genes_val = (adata_original_for_model.var["modality"] == "Gene Expression").sum()
    n_regions_val = (adata_original_for_model.var["modality"] == "peaks").sum()
    model_shell = scvi.model.MULTIVI(
        adata_original_for_model, # Use the original adata here
        n_genes=n_genes_val,
        n_regions=n_regions_val,
    )
    print("MULTIVI model shell created. Inferred/default parameters:")
    print(model_shell)

    # --- Load Model State ---
    print(f"Loading model state from: {model_pt_path}")
    loaded_full_checkpoint = torch.load(model_pt_path, map_location='cpu', weights_only=False)
    actual_state_dict = loaded_full_checkpoint['model_state_dict']
    model_shell.module.load_state_dict(actual_state_dict)
    model_shell.is_trained_ = True
    model = model_shell
    print("Model state loaded into shell successfully.")

    # --- Move Model to GPU ---
    target_device = "cuda:0"
    print(f"Moving model to device: {target_device}...")
    model.to_device(target_device)
    print(f"Model is now on device: {model.device}")

    # --- Manual Latent Space Retrieval ---
    print("\nStarting manual retrieval of latent representation (X_multivi)...")
    batch_size_latent = 512  # Adjust based on your GPU memory
    all_latent_list = []

    # Create a DataLoader using the adata_original_for_model (which model is set up with)
    print("Creating DataLoader for latent space generation...")
    latent_scdl = model._make_data_loader(
        adata=adata_original_for_model, # Use the AnnData model is set up with
        shuffle=False, # CRITICAL: Keep original cell order
        batch_size=batch_size_latent,
    )
    print("DataLoader created.")
    num_batches = len(latent_scdl)
    print(f"Processing {adata_original_for_model.n_obs} cells in batches of {batch_size_latent}...")

    with torch.no_grad():
        model.module.eval()
        for i, tensors_cpu in enumerate(latent_scdl):
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == num_batches: # Print progress
                 print(f"  Processing batch {i+1}/{num_batches} for latent space...")

            tensors_gpu = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in tensors_cpu.items()
            }
            inference_inputs = model.module._get_inference_input(tensors_gpu)
            # n_samples=1 for a deterministic latent representation (posterior mean of z)
            inference_outputs = model.module.inference(**inference_inputs, n_samples=1)

            if "z" in inference_outputs:
                all_latent_list.append(inference_outputs["z"].cpu().numpy())
            else:
                print(f"ERROR: 'z' (latent space) not found in inference_outputs for batch {i+1}. Available keys: {inference_outputs.keys()}")
                print("Aborting latent space retrieval.")
                return

    print("All batches processed for latent space.")

    if not all_latent_list:
        print("No latent representations were generated. Exiting.")
        return

    final_latent_representation = np.concatenate(all_latent_list, axis=0)
    print(f"Shape of concatenated latent representation: {final_latent_representation.shape}")

    if final_latent_representation.shape[0] != adata_original_for_model.n_obs:
        print(f"ERROR: Number of cells in latent representation ({final_latent_representation.shape[0]}) " \
              f"does not match original AnnData ({adata_original_for_model.n_obs}).")
        return

    # --- Load the Corrected RNA AnnData and Add Latent Space ---
    print(f"\nLoading existing corrected RNA AnnData from: {corrected_rna_adata_path}")
    if not os.path.exists(corrected_rna_adata_path):
        print(f"ERROR: Corrected RNA AnnData file not found at {corrected_rna_adata_path}. Please ensure it was created by the previous script.")
        return
        
    corrected_rna_adata_loaded = sc.read_h5ad(corrected_rna_adata_path)
    print(f"Loaded corrected RNA AnnData with shape: {corrected_rna_adata_loaded.shape}")

    # Validate cell order (should match as both originate from adata_original_for_model.obs_names)
    if not all(corrected_rna_adata_loaded.obs_names == adata_original_for_model.obs_names):
        print("ERROR: Cell order mismatch between loaded corrected RNA AnnData and original AnnData used for latent space generation.")
        print("Attempting to re-index corrected_rna_adata_loaded.obs if cell sets are identical...")
        if set(corrected_rna_adata_loaded.obs_names) == set(adata_original_for_model.obs_names):
            corrected_rna_adata_loaded = corrected_rna_adata_loaded[adata_original_for_model.obs_names, :].copy()
            print("Re-indexed corrected_rna_adata_loaded.obs to match original AnnData order.")
            if not all(corrected_rna_adata_loaded.obs_names == adata_original_for_model.obs_names):
                 print("ERROR: Re-indexing failed. Cannot safely add latent space.")
                 return
        else:
            print("ERROR: Different sets of cells. Cannot safely add latent space.")
            return
    
    print("Cell order verified between corrected RNA AnnData and latent space source.")
    corrected_rna_adata_loaded.obsm['X_multivi'] = final_latent_representation
    print("Added 'X_multivi' to the .obsm slot of the corrected RNA AnnData.")

    # --- Save the Updated AnnData ---
    print(f"Saving updated corrected RNA AnnData (now with X_multivi) to: {corrected_rna_adata_path}")
    try:
        # Overwrite the previous file
        corrected_rna_adata_loaded.write_h5ad(corrected_rna_adata_path, compression="gzip")
        print("Successfully saved updated AnnData with latent space.")
    except Exception as e:
        print(f"Error saving updated AnnData: {e}")

    print("Script finished.")

if __name__ == "__main__":
    generate_and_add_latent_space()