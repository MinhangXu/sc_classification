import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix, vstack as sparse_vstack
import gc
import os
import math

def sparsify_adata_x_chunked(adata_path, output_path, threshold=1e-6, chunk_size=10000):
    """
    Loads an AnnData object, sparsifies its .X matrix (if dense NumPy array)
    in chunks to manage memory, applies a threshold, and saves the updated AnnData object.
    """
    print(f"Loading AnnData from: {adata_path}")
    try:
        adata = sc.read_h5ad(adata_path)
    except Exception as e:
        print(f"Failed to load AnnData object: {e}")
        return
    
    print(f"Loaded AnnData. Original .X type: {type(adata.X)}, shape: {adata.X.shape if hasattr(adata.X, 'shape') else 'N/A'}")

    if not isinstance(adata.X, np.ndarray):
        print(f".X is already of type {type(adata.X)}, not a dense NumPy array. Skipping sparsification.")
        # Optionally save if you want to ensure consistent output path/compression
        # print(f"Saving AnnData (as is for .X) to: {output_path}")
        # adata.write_h5ad(output_path, compression="gzip")
        # print("AnnData saved.")
        return

    print(f"Attempting to sparsify .X with threshold {threshold} using chunk_size {chunk_size}...")
    
    n_obs, n_vars = adata.X.shape
    sparse_chunks_list = []
    
    try:
        num_chunks = math.ceil(n_obs / chunk_size)
        print(f"Processing {n_obs} cells in {num_chunks} chunks of up to {chunk_size} cells each.")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_obs)
            
            print(f"  Processing chunk {i+1}/{num_chunks} (cells {start_idx}-{end_idx-1})...")
            
            dense_chunk = adata.X[start_idx:end_idx, :]
            
            # Apply threshold: set values smaller than threshold (in magnitude) to 0
            # This creates a temporary boolean mask for the chunk only
            dense_chunk[np.abs(dense_chunk) < threshold] = 0.0
            
            # Convert the (now more zero-filled) dense chunk to CSR sparse matrix
            sparse_chunk = csr_matrix(dense_chunk)
            sparse_chunks_list.append(sparse_chunk)
            
            # Clean up to save memory within the loop (though Python's GC should handle dense_chunk)
            del dense_chunk
            del sparse_chunk
            if (i + 1) % 5 == 0: # Collect garbage every 5 chunks
                gc.collect()
        
        print("All chunks processed. Vertically stacking sparse chunks...")
        if not sparse_chunks_list:
            print("No sparse chunks generated. This should not happen if there was data.")
            return

        final_sparse_X = sparse_vstack(sparse_chunks_list)
        gc.collect()
        
        adata.X = final_sparse_X
        
        new_percent_zeros = (1 - (adata.X.nnz / adata.X.size)) * 100
        print(f"  New percentage of zero values after thresholding: {new_percent_zeros:.2f}%")
        print(f"  Type of .X after sparsification: {type(adata.X)}")
        print("Sparsification complete.")

    except MemoryError:
        print("ERROR: A MemoryError occurred even during chunked sparsification.")
        print("Try reducing chunk_size further, or ensure more system RAM is available.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during chunked sparsification: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nSaving updated AnnData with sparsified .X to: {output_path}")
    try:
        adata.write_h5ad(output_path, compression="gzip")
        print("Updated AnnData saved successfully.")
    except Exception as e:
        print(f"Error saving updated AnnData: {e}")

if __name__ == '__main__':
    # Path to the AnnData object that has the DENSE corrected RNA in .X 
    # (and potentially already protein and latent space)
    adata_to_sparsify_path = '/home/minhang/mds_project/data/cohort_adata/multiVI_model/adata_multivi_corrected_rna.h5ad' 
    
    # Path for the output AnnData with sparsified .X
    sparsified_output_path = adata_to_sparsify_path.replace('.h5ad', '_final_sparseRNA.h5ad') 
    
    print(f"Input AnnData for sparsification: {adata_to_sparsify_path}")
    print(f"Output AnnData after sparsification: {sparsified_output_path}")
    
    # Adjust chunk_size based on your available RAM.
    # If 10000 still crashes, try 5000, 2000, or 1000.
    # A smaller chunk_size means more iterations but lower peak RAM per iteration.
    sparsify_adata_x_chunked(adata_to_sparsify_path, sparsified_output_path, threshold=1e-6, chunk_size=5000)