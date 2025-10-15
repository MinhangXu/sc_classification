import scvi
import torch
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata # Often imported with scanpy, but good to be explicit
import pyarrow

def get_batch_corrected_expression():
    """
    Loads a pre-trained scvi-tools MULTIVI model and an AnnData object,
    then manually performs inference to get batch-corrected/normalized
    gene expression, handling potential device mismatches by moving data
    to the model's device.
    """
    print("Starting script for batch-corrected expression retrieval...")

    # --- CUDA and System Checks ---
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        # Set default device for PyTorch (optional, as model.to_device() is more specific)
        # torch.set_default_tensor_type('torch.cuda.FloatTensor') # Be cautious with global settings
    else:
        print("Warning: CUDA not available. This script expects to run on GPU.")
        # Consider exiting if GPU is mandatory:
        # return

    # --- scvi-tools Settings ---
    # Using more workers can speed up CPU-bound parts of data loading.
    # However, the primary device issue workaround is manual tensor movement.
    try:
        scvi.settings.dl_num_workers = 8
        print(f"Set scvi.settings.dl_num_workers to: {scvi.settings.dl_num_workers}")
    except Exception as e:
        print(f"Could not set scvi.settings.dl_num_workers: {e}. Using default.")


    # --- File Paths ---
    base_dir = '/home/minhang/mds_project/data/cohort_adata/multiVI_model'
    scvi_model_dir = base_dir # Assuming model.pt is in this directory
    adata_path = os.path.join(base_dir, 'adata.h5ad')
    model_pt_path = os.path.join(scvi_model_dir, 'model.pt')
    output_csv_path = os.path.join(base_dir, 'batch_corrected_normalized_expression.csv')

    print(f"AnnData path: {adata_path}")
    print(f"Model path: {model_pt_path}")
    print(f"Output CSV path: {output_csv_path}")

    # --- Load AnnData ---
    print("Loading AnnData object...")
    adata_mvi = sc.read_h5ad(adata_path)
    adata_mvi.var_names_make_unique()
    print("AnnData loaded successfully.")

    # Display feature counts and metadata to verify
    print("\n--- AnnData Overview ---")
    print("Feature counts by modality:")
    if "modality" in adata_mvi.var.columns:
        print(adata_mvi.var["modality"].value_counts())
    else:
        print("Warning: adata_mvi.var['modality'] column not found.")
    print(f"Shape of adata_mvi: {adata_mvi.shape}")
    if "ADT" in adata_mvi.obsm_keys():
        print(f"Shape of ADT in obsm: {adata_mvi.obsm['ADT'].shape}")
    if "Tech" in adata_mvi.obs.columns:
        print(f"Batch key 'Tech' categories: {adata_mvi.obs['Tech'].unique().tolist()}")
    if "sample" in adata_mvi.obs.columns:
        print(f"Categorical covariate 'sample' categories (first 5): {adata_mvi.obs['sample'].unique().tolist()[:5]}")
    print("------------------------\n")

    # --- Setup AnnData for scvi-tools ---
    print("Setting up AnnData for MULTIVI...")
    scvi.model.MULTIVI.setup_anndata(
        adata_mvi,
        batch_key="Tech",
        protein_expression_obsm_key="ADT",
        categorical_covariate_keys=["sample"]
    )
    print("AnnData setup complete.")

    # --- Initialize Model Shell ---
    # Parameters will be inferred from adata_mvi.uns if saved there by collaborator,
    # which is good for ensuring architectural consistency.
    print("Initializing MULTIVI model shell...")
    n_genes_val = (adata_mvi.var["modality"] == "Gene Expression").sum()
    n_regions_val = (adata_mvi.var["modality"] == "peaks").sum()

    model_shell = scvi.model.MULTIVI(
        adata_mvi,
        n_genes=n_genes_val,
        n_regions=n_regions_val,
    )
    print("MULTIVI model shell created. Inferred/default parameters:")
    print(model_shell) # This will show the actual parameters used (n_hidden, n_latent, etc.)

    # --- Load Model State ---
    print(f"Loading model state from: {model_pt_path}")
    # map_location='cpu' is good practice for loading, then move to GPU
    loaded_full_checkpoint = torch.load(model_pt_path, map_location='cpu', weights_only=False)
    print("Model checkpoint loaded successfully as a dictionary.")
    # print("Top-level keys in loaded file:", list(loaded_full_checkpoint.keys())) # For debugging

    actual_state_dict = loaded_full_checkpoint['model_state_dict']
    # print("Extracted 'model_state_dict'. First 5 keys:", list(actual_state_dict.keys())[:5]) # For debugging

    model_shell.module.load_state_dict(actual_state_dict)
    model_shell.is_trained_ = True  # Manually mark the model as trained
    model = model_shell
    print("Model state loaded into shell successfully.")

    # --- Move Model to GPU ---
    if torch.cuda.is_available():
        target_device = "cuda:0" # Or simply "cuda"
        print(f"Moving model to device: {target_device}...")
        model.to_device(target_device)
        print(f"Model is now on device: {model.device}")
        # Check model parameter device
        if hasattr(model.module, 'parameters') and next(model.module.parameters(), None) is not None:
             print(f"First model parameter is on device: {next(model.module.parameters()).device}")
        else:
            print("Could not check model parameter device.")

    else:
        print("CUDA not available. Model remains on CPU. This script requires GPU for efficient execution.")
        return # Exit if no GPU

    # --- Manual Normalized Expression Retrieval ---
    print("\nStarting manual retrieval of normalized expression...")
    print(f"Using scvi.settings.dl_num_workers: {scvi.settings.dl_num_workers}")

    batch_size = 512  # Adjust based on your GPU memory
    all_norm_exp_list = []

    # Create a DataLoader for the full AnnData object.
    # shuffle=False is CRITICAL to maintain original cell order.
    print("Creating DataLoader for the full dataset...")
    full_scdl = model._make_data_loader(
        adata=adata_mvi,
        shuffle=False,
        batch_size=batch_size,
        # num_workers will be taken from scvi.settings.dl_num_workers
    )
    print("DataLoader created.")

    print(f"Processing {adata_mvi.n_obs} cells in batches of {batch_size}...")
    num_batches = len(full_scdl)

    with torch.no_grad():
        model.module.eval()

        for i, tensors_cpu in enumerate(full_scdl):
            print(f"Processing batch {i+1}/{num_batches}...")
            # 1. Manually move tensors to the model's device
            tensors_gpu = {
                k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                for k, v in tensors_cpu.items()
            }

            # 2. Perform inference
            inference_kwargs = {"n_samples": 1} # Get a deterministic latent state
            inference_inputs = model.module._get_inference_input(tensors_gpu)
            inference_outputs = model.module.inference(**inference_inputs, **inference_kwargs)

            # 3. Perform generation of normalized expression
            generative_kwargs = {"use_z_mean": True} # Use mean of latent space for stable normalized expression
            generative_inputs = model.module._get_generative_input(tensors_gpu, inference_outputs)
            generative_outputs_dict = model.module.generative(**generative_inputs, **generative_kwargs)

            # 4. Extract RNA normalized expression ('px_scale')
            if "px_scale" in generative_outputs_dict:
                batch_norm_exp_gpu = generative_outputs_dict["px_scale"]
                all_norm_exp_list.append(batch_norm_exp_gpu.cpu().numpy())
            else:
                print(f"ERROR: 'px_scale' not found in generative_outputs_dict for batch {i+1}. Available keys: {generative_outputs_dict.keys()}")
                print("Aborting expression retrieval.")
                return # Exit if critical data is missing

    print("All batches processed.")

    # 5. Concatenate results and create DataFrame
    if all_norm_exp_list:
        print("Concatenating results...")
        final_norm_exp_np = np.concatenate(all_norm_exp_list, axis=0)
        print(f"Shape of concatenated normalized expression numpy array: {final_norm_exp_np.shape}")

        gene_expression_mask = (adata_mvi.var["modality"] == "Gene Expression")
        gene_names = adata_mvi.var_names[gene_expression_mask].tolist()

        if final_norm_exp_np.shape[0] != adata_mvi.n_obs:
            print(f"ERROR: Number of cells in output ({final_norm_exp_np.shape[0]}) " \
                  f"does not match input AnnData ({adata_mvi.n_obs}). Check batch processing.")
            # return # Keep this if you want the function to exit on error

        if final_norm_exp_np.shape[1] == len(gene_names):
            norm_exp_df = pd.DataFrame(
                final_norm_exp_np,
                index=adata_mvi.obs_names,
                columns=gene_names
            )
            print("\nSuccessfully generated DataFrame for normalized expression.")
            print("Head of the DataFrame:")
            print(norm_exp_df.head())

            # --- MODIFIED SAVING SECTION ---
            base_output_filename = "batch_corrected_normalized_expression"
            
            # Option 1: Save as Parquet with gzip compression (Recommended)
            
            output_parquet_path = os.path.join(scvi_model_dir, f"{base_output_filename}.parquet.gz")
            try:
                print(f"\nAttempting to save normalized expression DataFrame to Parquet (gzipped): {output_parquet_path}")
                # You might need to install pyarrow: pip install pyarrow
                norm_exp_df.to_parquet(output_parquet_path, compression='gzip', engine='pyarrow')
                print(f"DataFrame successfully saved to Parquet: {output_parquet_path}")
            except ImportError:
                print("Warning: 'pyarrow' library not found for Parquet export. Falling back to compressed CSV.")
                # Option 2: Save as Compressed CSV (gzip) as a fallback
                output_csv_gz_path = os.path.join(scvi_model_dir, f"{base_output_filename}.csv.gz")
                try:
                    print(f"\nAttempting to save normalized expression DataFrame to CSV (gzipped): {output_csv_gz_path}")
                    norm_exp_df.to_csv(output_csv_gz_path, compression='gzip')
                    print(f"DataFrame successfully saved to CSV (gzipped): {output_csv_gz_path}")
                except OSError as e:
                    print(f"OSError during compressed CSV saving: {e}")
                    print("Saving failed. Please check available disk space and permissions.")
                except Exception as e:
                    print(f"An unexpected error occurred during compressed CSV saving: {e}")
            except OSError as e:
                print(f"OSError during Parquet saving: {e}")
                if "[Errno 28]" in str(e):
                    print("Still no space left on device, even with Parquet compression.")
                print("Consider saving to a different disk or ensuring more space is available.")
                print("You could also try saving as AnnData H5AD with compression (see validation script).")
            except Exception as e:
                print(f"An unexpected error occurred during Parquet saving: {e}")
            

            # Option 3: Save as AnnData object (as discussed in the validation step)
            # This would typically be done after validation.
            #output_adata_path = os.path.join(scvi_model_dir, f"{base_output_filename}_rna.h5ad")
            #print(f"\nTo save as AnnData (H5AD), you can use the validation script's logic or adapt it here:")
            #print(f"Example: corrected_rna_adata = anndata.AnnData(X=norm_exp_df, obs=adata_mvi.obs.copy(), var=adata_mvi.var[gene_expression_mask].copy())")
            # print(f"corrected_rna_adata.write_h5ad(output_adata_path, compression='gzip')")
            # print(f"This would save to: {output_adata_path}")

            # --- END OF MODIFIED SAVING SECTION ---

        else:
            print(f"\nShape Mismatch Error during DataFrame creation. Cannot save.")
            # ... (your existing error message) ...
    else:
        print("No normalized expression data was generated (all_norm_exp_list is empty). Cannot save.")

    print("Script finished.")

if __name__ == "__main__":
    get_batch_corrected_expression()
