import os
import json
import pandas as pd
import csv
from typing import Dict, Set, List, Union 

# --- Import refactored configuration tools ---
# Assuming current script is in project root, utils is one level below root
try:
    from utils.gene_config_utils import get_config_path
except ImportError:
    # Compatibility for cases without subdirectory structure
    print("Error: Unable to import utils.gene_config_utils. Please check the directory structure.")
    # Keep the original function structure for execution fallback
    def get_config_path(config_filename: str, prefix_key: str = "prefix1") -> Union[str, None]:
        config_path = os.path.join(os.getcwd(), config_filename)
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                prefix1_path = config_data.get(prefix_key)
                if not prefix1_path:
                    return None
                return prefix1_path
        except Exception:
            return None


# --- Configuration ---
CONFIG_FILE = "config.txt"
CONFIG_KEY = "prefix1" 
GENE_MATRIX_SUBPATH = "processed/cell_gene_matrix.csv"
MATCHED_CSV_FILENAME = "processed/matched_nuclei_filtered.csv" # Used for cell count statistics
MASK_FILE = "mask.csv"
# Final merged information file
INFO_FILE = "all_info.json" 

def get_cell_count(csv_path: str) -> int:
    """Read the number of cell IDs (row count) from matched_nuclei_filtered.csv."""
    count = 0
    try:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # Skip header
            except StopIteration:
                return 0
            
            for _ in reader:
                count += 1
        return count
    except Exception as e:
        # Return 0 on failure to avoid excessive output
        return 0

def process_gene_masks(config_filename: str = CONFIG_FILE):
    """
    Main function: Reads dataset directories, extracts gene information and cell counts, 
    generates mask.csv, and merges statistics into all_info.json (including min_num).
    """
    # Use the imported configuration tool function
    prefix1_path = get_config_path(config_filename, CONFIG_KEY) 
    
    if prefix1_path is None:
        print(f"Error: Unable to get '{CONFIG_KEY}' path, exiting.")
        return

    if not os.path.isdir(prefix1_path):
        print(f"Error: Path '{prefix1_path}' does not exist or is not a directory.")
        return

    # Get all top-level subdirectory names
    try:
        all_dirs = [
            item for item in os.listdir(prefix1_path) 
            if os.path.isdir(os.path.join(prefix1_path, item))
        ]
    except PermissionError:
        print(f"Error: No permission to access directory '{prefix1_path}'.")
        return
    
    if not all_dirs:
        print(f"Warning: No subdirectories found under '{prefix1_path}'.")
        return
    
    print(f"Found {len(all_dirs)} dataset directories, starting extraction of genes and cell counts...")

    # Used to store gene sets for each directory
    dataset_genes: Dict[str, Set[str]] = {}
    all_genes_union: Set[str] = set()
    
    # Used to store cell counts for all valid datasets
    all_valid_cell_counts: List[int] = []
    
    # Final statistics dictionary (for JSON writing)
    final_info_data: Dict[str, Union[int, Dict[str, int]]] = {}
    
    # 2. Iterate through each directory, read gene names and cell counts
    for dataset_name in all_dirs:
        gene_matrix_path = os.path.join(prefix1_path, dataset_name, GENE_MATRIX_SUBPATH)
        matched_csv_path = os.path.join(prefix1_path, dataset_name, MATCHED_CSV_FILENAME)

        gene_count = 0
        cell_count = 0
        
        print(f"\n--- Processing dataset: {dataset_name} ---")

        # --- Gene Statistics ---
        if not os.path.exists(gene_matrix_path):
            print(f"Warning: Skipping {dataset_name} gene stats: Gene matrix file '{GENE_MATRIX_SUBPATH}' not found")
        else:
            try:
                df = pd.read_csv(gene_matrix_path, nrows=0)
                genes = set(df.columns[1:].tolist())
                gene_count = len(genes)
                
                if gene_count > 0:
                    dataset_genes[dataset_name] = genes
                    all_genes_union.update(genes)
                
                print(f"    - Gene count: {gene_count}")

            except Exception as e:
                print(f"Error: Gene statistics failed for {dataset_name}: {e}")
        
        # --- Cell Statistics ---
        if not os.path.exists(matched_csv_path):
            print(f"Warning: Skipping {dataset_name} cell stats: Matched CSV file '{MATCHED_CSV_FILENAME}' not found")
        else:
            cell_count = get_cell_count(matched_csv_path)
            print(f"    - Cell count: {cell_count}")
        
        # Record statistics results (Dataset Level)
        final_info_data[dataset_name] = {
            "gene_count": gene_count,
            "cell_count": cell_count
        }

        # Collect valid cell counts
        if cell_count > 0:
            all_valid_cell_counts.append(cell_count)


    # 3. Calculate global statistics
    union_count = len(all_genes_union)
    min_cell_num = min(all_valid_cell_counts) if all_valid_cell_counts else 0

    print(f"\nGlobal gene union size: {union_count} genes.")
    print(f"Minimum cell count among all valid datasets: {min_cell_num}")

    # 4. Generate and save mask.csv
    if not dataset_genes:
        print("\nNo gene information successfully processed, unable to generate mask.csv.")
    else:
        # Sort genes to ensure consistent column ordering
        all_genes_sorted = sorted(list(all_genes_union))
        
        mask_data: List[List[Union[str, int]]] = []
        # Use final_info_data.keys() to ensure consistent dataset order
        for dataset_name in final_info_data.keys(): 
            if dataset_name in dataset_genes:
                current_genes = dataset_genes[dataset_name]
                mask_row: List[Union[str, int]] = [dataset_name]
                for gene in all_genes_sorted:
                    mask_row.append(1 if gene in current_genes else 0)
                mask_data.append(mask_row)
            
        mask_columns = ['dataset_name'] + all_genes_sorted
        mask_df = pd.DataFrame(mask_data, columns=mask_columns)
        
        mask_output_path = os.path.join(prefix1_path, MASK_FILE)
        try:
            mask_df.to_csv(mask_output_path, index=False)
            print(f"\nSuccessfully generated mask.csv file!")
            print(f"Path: {mask_output_path}")
            print(f"Shape: {mask_df.shape} (Rows: {mask_df.shape[0]} datasets, Cols: {mask_df.shape[1]-1} genes)")
        except Exception as e:
            print(f"Error: Failed to save mask.csv, reason: {e}")
    
    # 5. Save final all_info.json file
    
    # Dictionary to store global statistics
    global_stats = {
        "union_gene_count": union_count,
        "min_num": min_cell_num
    }
    
    # Merge global stats and dataset-level stats
    final_json_output = {}
    final_json_output.update(global_stats)
    
    # Place dataset information under the 'dataset_info' key
    final_json_output["dataset_info"] = {
        k: v for k, v in final_info_data.items()
        if isinstance(v, dict) and 'gene_count' in v # Filter out non-dataset items
    }


    info_output_path = os.path.join(prefix1_path, INFO_FILE)
    try:
        with open(info_output_path, 'w') as f:
            json.dump(final_json_output, f, indent=4) 
        print(f"\nSuccessfully generated merged information file ({INFO_FILE})!")
        print(f"Path: {info_output_path}")
    except Exception as e:
        print(f"Error: Failed to save {INFO_FILE}, reason: {e}")

if __name__ == "__main__":
    print("=== Global Gene Union Mask and Statistics Generation Tool ===")
    process_gene_masks()