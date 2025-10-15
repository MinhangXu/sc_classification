import scvi
import torch
import os
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import h5py # For chunked writing of large ATAC data

def get_corrected_protein_atac_chunked():
    print("Starting script for batch-corrected Protein and ATAC retrieval (ATAC chunked)...")

    # --- Configuration ---
    PROCESS_PROTEIN = True
    PROCESS_ATAC = False
    # If True, existing HDF5 file for ATAC will be overwritten.
    # Set to False if you want to resume or are cautious (script will exit if file exists).
    OVERWRITE_ATAC_HDF5 = True 

    # --- CUDA and System Checks ---
    # ... (same as your previous script) ...
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Error: CUDA not available. This script requires GPU.")
        return
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")


    # --- scvi-tools Settings ---
    try:
        scvi.settings.dl_num_workers = 0 # Using 0 for simplicity, was not the cause of OOM
        print(f"Set scvi.settings.dl_num_workers to: {scvi.settings.dl_num_workers}")
    except Exception as e:
        print(f"Could not set scvi.settings.dl_num_workers: {e}. Using default.")

    # --- File Paths ---
    base_dir = '/home/minhang/mds_project/data/cohort_adata/multiVI_model'
    original_adata_path = os.path.join(base_dir, 'adata.h5ad')
    model_pt_path = os.path.join(base_dir, 'model.pt')
    
    output_protein_parquet_path = os.path.join(base_dir, 'corrected_protein_expression.parquet.gz')
    # For ATAC, we'll save to an HDF5 file first, then optionally convert later
    output_atac_hdf5_path = os.path.join(base_dir, 'corrected_atac_accessibility.h5')

    print(f"Original AnnData path: {original_adata_path}")
    print(f"Model path: {model_pt_path}")
    if PROCESS_PROTEIN:
        print(f"Output Protein Parquet path: {output_protein_parquet_path}")
    if PROCESS_ATAC:
        print(f"Output ATAC HDF5 path: {output_atac_hdf5_path}")


    # --- Load Original AnnData ---
    print("Loading original AnnData object...")
    adata_mvi_original = sc.read_h5ad(original_adata_path)
    adata_mvi_original.var_names_make_unique()
    print("Original AnnData loaded successfully.")
    n_obs = adata_mvi_original.n_obs
    accessibility_mask = (adata_mvi_original.var["modality"] == "peaks")
    peak_names = adata_mvi_original.var_names[accessibility_mask].tolist()
    n_peaks = len(peak_names)


    # --- Setup AnnData, Initialize Model, Load State, Move to GPU ---
    # ... (This part is identical to your previous script) ...
    print("Setting up original AnnData for MULTIVI model...")
    scvi.model.MULTIVI.setup_anndata(
        adata_mvi_original, batch_key="Tech",
        protein_expression_obsm_key="ADT", categorical_covariate_keys=["sample"]
    )
    print("Initializing MULTIVI model shell...")
    n_genes_val = (adata_mvi_original.var["modality"] == "Gene Expression").sum()
    n_regions_val = n_peaks # Already calculated
    model_shell = scvi.model.MULTIVI(
        adata_mvi_original, n_genes=n_genes_val, n_regions=n_regions_val,
    )
    print(f"Model shell created: {model_shell}")
    print(f"Loading model state from: {model_pt_path}")
    loaded_full_checkpoint = torch.load(model_pt_path, map_location='cpu', weights_only=False)
    actual_state_dict = loaded_full_checkpoint['model_state_dict']
    model_shell.module.load_state_dict(actual_state_dict)
    model_shell.is_trained_ = True
    model = model_shell
    print("Model state loaded.")
    target_device = "cuda:0"
    print(f"Moving model to device: {target_device}...")
    model.to_device(target_device)
    print(f"Model is now on device: {model.device}")


    # --- Manual Retrieval Loop ---
    print("\nStarting manual retrieval of corrected Protein and/or ATAC data...")
    batch_size = 256
    
    all_protein_list = [] if PROCESS_PROTEIN else None
    
    # For ATAC: HDF5 file setup
    atac_hdf5_file = None
    atac_hdf5_dataset = None
    if PROCESS_ATAC:
        if os.path.exists(output_atac_hdf5_path):
            if OVERWRITE_ATAC_HDF5:
                print(f"Overwriting existing ATAC HDF5 file: {output_atac_hdf5_path}")
                os.remove(output_atac_hdf5_path)
            else:
                print(f"ERROR: ATAC HDF5 file already exists: {output_atac_hdf5_path}. Set OVERWRITE_ATAC_HDF5=True to overwrite.")
                PROCESS_ATAC = False # Do not process ATAC
        
        if PROCESS_ATAC: # Check again in case it was set to False
            print(f"Initializing HDF5 file for ATAC data at {output_atac_hdf5_path}")
            atac_hdf5_file = h5py.File(output_atac_hdf5_path, 'w', libver='latest')
            # Create dataset with full dimensions, chunking for efficient writing
            # Choose a chunk shape, e.g., (batch_size, n_peaks) or smaller like (128, 10000)
            # Let's use a chunk shape that aligns with batches for simplicity here
            chunk_shape = (min(batch_size, n_obs), n_peaks) 
            atac_hdf5_dataset = atac_hdf5_file.create_dataset(
                'corrected_atac_probs',
                shape=(n_obs, n_peaks),
                dtype='float32', # pa_probs are usually float32
                chunks=chunk_shape,
                compression="gzip" # Optional: add compression
            )
            print(f"HDF5 dataset 'corrected_atac_probs' created with shape {atac_hdf5_dataset.shape} and chunks {atac_hdf5_dataset.chunks}")


    print("Creating DataLoader...")
    full_scdl = model._make_data_loader(
        adata=adata_mvi_original, shuffle=False, batch_size=batch_size,
    )
    num_batches = len(full_scdl)
    print(f"DataLoader created. Processing {n_obs} cells in {num_batches} batches of size up to {batch_size}...")

    processed_cell_count = 0
    with torch.no_grad():
        model.module.eval()
        for i, tensors_cpu in enumerate(full_scdl):
            current_batch_size = tensors_cpu['X'].shape[0] # Actual size of this batch
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == num_batches:
                 print(f"  Processing batch {i+1}/{num_batches} (cells {processed_cell_count+1}-{processed_cell_count+current_batch_size})...")

            tensors_gpu = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in tensors_cpu.items()}
            inference_kwargs = {"n_samples": 1}
            inference_inputs = model.module._get_inference_input(tensors_gpu)
            inference_outputs = model.module.inference(**inference_inputs, **inference_kwargs)
            generative_kwargs = {"use_z_mean": True}
            generative_inputs = model.module._get_generative_input(tensors_gpu, inference_outputs)
            generative_outputs_dict = model.module.generative(**generative_inputs, **generative_kwargs)

            if PROCESS_PROTEIN and all_protein_list is not None:
                if "py_" in generative_outputs_dict and isinstance(generative_outputs_dict['py_'], dict) and "rate_fore" in generative_outputs_dict['py_']:
                    batch_protein_gpu = generative_outputs_dict['py_']["rate_fore"]
                    all_protein_list.append(batch_protein_gpu.cpu().numpy())
                else:
                    print(f"ERROR: Protein data ('py_['rate_fore']') not found in batch {i+1}. Disabling further protein processing.")
                    PROCESS_PROTEIN = False 
                    all_protein_list = None # Ensure no further appends

            if PROCESS_ATAC and atac_hdf5_dataset is not None:
                if "p" in generative_outputs_dict:
                    batch_atac_gpu = generative_outputs_dict["p"]
                    batch_atac_cpu_np = batch_atac_gpu.cpu().numpy()
                    # Write this batch to the correct slice in the HDF5 file
                    atac_hdf5_dataset[processed_cell_count : processed_cell_count + current_batch_size, :] = batch_atac_cpu_np
                else:
                    print(f"ERROR: ATAC data ('p') not found in batch {i+1}. Disabling further ATAC processing.")
                    PROCESS_ATAC = False
                    if atac_hdf5_file: # Close file if error occurs
                        atac_hdf5_file.close()
                        atac_hdf5_file = None 
                    atac_hdf5_dataset = None
            
            processed_cell_count += current_batch_size

    print("All batches processed.")

    # Close HDF5 file for ATAC if it was opened
    if atac_hdf5_file is not None:
        print(f"Closing ATAC HDF5 file: {output_atac_hdf5_path}")
        atac_hdf5_file.close()

    # --- Concatenate and Save Protein Data ---
    if PROCESS_PROTEIN and all_protein_list: # Check all_protein_list as it might have been set to None
        print("\nConcatenating protein results...")
        # ... (Same protein saving logic as your previous script, using output_protein_parquet_path) ...
        final_protein_np = np.concatenate(all_protein_list, axis=0)
        if 'ADT' not in adata_mvi_original.obsm or not hasattr(adata_mvi_original.obsm['ADT'], 'columns'):
            print("ERROR: adata_mvi_original.obsm['ADT'] not found or not a DataFrame. Cannot get protein names.")
        elif final_protein_np.shape[0] != n_obs or final_protein_np.shape[1] != adata_mvi_original.obsm['ADT'].shape[1]:
            print(f"ERROR: Protein data shape mismatch. Got {final_protein_np.shape}, expected ({n_obs}, {adata_mvi_original.obsm['ADT'].shape[1]})")
        else:
            protein_names = adata_mvi_original.obsm['ADT'].columns.tolist()
            protein_df = pd.DataFrame(final_protein_np, index=adata_mvi_original.obs_names, columns=protein_names)
            print("Successfully generated DataFrame for corrected protein expression.")
            print(f"Saving corrected protein DataFrame to: {output_protein_parquet_path}")
            try:
                protein_df.to_parquet(output_protein_parquet_path, compression='gzip', engine='pyarrow')
                print(f"Protein DataFrame saved to Parquet: {output_protein_parquet_path}")
            except Exception as e:
                print(f"Error saving protein Parquet: {e}. Trying compressed CSV...")
                output_protein_csv_path = output_protein_parquet_path.replace(".parquet.gz", ".csv.gz")
                try:
                    protein_df.to_csv(output_protein_csv_path, compression='gzip')
                    print(f"Protein DataFrame saved to CSV (gzipped): {output_protein_csv_path}")
                except Exception as e_csv:
                    print(f"Error saving protein to compressed CSV: {e_csv}")

    elif PROCESS_PROTEIN: # If it was True initially but list is None due to error
        print("\nProtein data processing was attempted but encountered an error or no data was generated.")

    if PROCESS_ATAC: # Check if ATAC processing was intended and HDF5 file was created
        if os.path.exists(output_atac_hdf5_path) and os.path.getsize(output_atac_hdf5_path) > 0 : # Check if file exists and is not empty
             print(f"\nCorrected ATAC data (probabilities) saved to HDF5 file: {output_atac_hdf5_path}")
             print(f"  Dataset name inside HDF5: 'corrected_atac_probs'")
             print(f"  Shape: ({n_obs}, {n_peaks})")
             print("You can load this HDF5 dataset later for analysis, e.g., using h5py or by converting to AnnData (in chunks if needed).")
        elif not os.path.exists(output_atac_hdf5_path) and PROCESS_ATAC: # If PROCESS_ATAC was true but file doesn't exist (e.g. error before creation)
             print("\nATAC HDF5 file was not created due to an earlier error or processing was disabled.")


    print("\nScript finished.")

if __name__ == "__main__":
    get_corrected_protein_atac_chunked()