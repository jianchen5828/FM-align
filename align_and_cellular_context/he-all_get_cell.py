import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm
import tifffile
import json
from typing import Union, Dict, Set, List, Tuple

# Import configuration tools
from utils.he_config import get_prefix_path

# ==============================================================================
# 0. Global Configuration and Paths
# ==============================================================================

# Configuration filename
CONFIG_FILE = "config.txt"
CONFIG_KEY = "prefix1" # Key name for the configuration file

# -------------------- File Path Subpaths (Relative to each dataset directory) --------------------
HE_FILENAME = 'he_image.tif'                                # Original HE image filename
MASK_FILENAME = 'processed/he_image_nuclei_seg.tif'         # Original high-bit nuclear mask (ID map)
CSV_FILENAME = 'processed/matched_nuclei_filtered.csv'      # CSV filename containing cell IDs to process
OUTPUT_MASK_FILENAME = 'processed/cell_mask_dilated.tif'    # Dilated mask save filename
OUTPUT_CELLS_DIR_NAME = "processed/cropped_cells_zoomed"    # Saving directory for cropped cell images
DONE_FLAG_FILENAME = "processed/crop_done"                  # Completion flag filename

# -------------------- Processing Parameters --------------------
TARGET_SIZE = 128                                          # Unified target size (e.g., 128x128)
KERNEL_SIZE = 17                                           # Recommended dilation kernel size for 10x Breast Cancer (~8 pixel expansion)

# ==============================================================================
# 1. Helper Functions: Removed, replaced by utils.he_config.get_prefix_path
# ==============================================================================

# ==============================================================================
# 2. Core Function: Process Single Dataset
# ==============================================================================

def process_single_dataset(dataset_path: str):
    """Processes a single dataset directory, including mask dilation and cell cropping."""
    
    dataset_name = os.path.basename(dataset_path)
    
    # Path combinations
    MASK_FILE_PATH = os.path.join(dataset_path, MASK_FILENAME)
    HE_FILE_PATH = os.path.join(dataset_path, HE_FILENAME)
    CSV_FILE_PATH = os.path.join(dataset_path, CSV_FILENAME)
    OUTPUT_MASK_PATH = os.path.join(dataset_path, OUTPUT_MASK_FILENAME)
    OUTPUT_CELLS_DIR = os.path.join(dataset_path, OUTPUT_CELLS_DIR_NAME)
    DONE_FLAG_PATH = os.path.join(dataset_path, DONE_FLAG_FILENAME)

    # -------------------- Check Completion Flag --------------------
    if os.path.exists(DONE_FLAG_PATH):
        print(f"\n=======================================================")
        print(f"Skipping dataset: {dataset_name} (Detected {DONE_FLAG_FILENAME} flag)")
        print(f"=======================================================")
        return
    
    print(f"\n=======================================================")
    print(f"Processing dataset: {dataset_name}")
    print(f"=======================================================")

    # Check for basic input files
    if not os.path.exists(HE_FILE_PATH) or not os.path.exists(MASK_FILE_PATH) or not os.path.exists(CSV_FILE_PATH):
        print(f"Warning: Skipping {dataset_name}: Missing necessary input files (HE, MASK, or CSV).")
        print(f"   HE: {HE_FILE_PATH}")
        print(f"   MASK: {MASK_FILE_PATH}")
        print(f"   CSV: {CSV_FILE_PATH}")
        return

    # --- Phase 1: Generate dilated cell mask ---
    print(f"\n--- Phase 1: Generating dilated cell mask (KERNEL_SIZE={KERNEL_SIZE}) ---")

    try:
        # Use tifffile to read large files
        nucleus_mask = tifffile.imread(MASK_FILE_PATH)
    except Exception as e:
        print(f"Error: Could not load mask image {MASK_FILE_PATH}. Info: {e}")
        return

    print(f"Original mask dimensions: {nucleus_mask.shape}, Max ID: {nucleus_mask.max()}")

    cell_mask_dilated = np.zeros_like(nucleus_mask)
    props = regionprops(nucleus_mask)
    
    # Check if nuclei exist
    if not props:
        print("Warning: No nuclear instances found in the original mask. Skipping dilation and cropping.")
        return
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    print(f"Starting dilation for all {len(props)} cell instances...")

    for prop in tqdm(props, desc="Dilating nuclei"):
        cell_id = prop.label
        ymin, xmin, ymax, xmax = prop.bbox
        
        local_mask_slice = nucleus_mask[ymin:ymax, xmin:xmax]
        # Keep only the current cell_id and convert to uint8 (0 or 255)
        temp_mask = (local_mask_slice == cell_id).astype(np.uint8) * 255
        
        # Execute dilation
        local_dilated_mask = cv2.dilate(temp_mask, kernel, iterations=1)
        dilated_indices = np.nonzero(local_dilated_mask)
        
        # Write dilated area into result map
        global_y = ymin + dilated_indices[0]
        global_x = xmin + dilated_indices[1]
        
        cell_mask_dilated[global_y, global_x] = cell_id

    # Create save directory
    os.makedirs(os.path.dirname(OUTPUT_MASK_PATH), exist_ok=True)
    
    # Save dilated mask using tifffile
    tifffile.imwrite(OUTPUT_MASK_PATH, cell_mask_dilated)
    print(f"Dilated cell mask saved to: {OUTPUT_MASK_PATH}")

    # ==============================================================================
    # Phase 2: CSV filtering, mask reading, cropping, scaling, and saving
    # ==============================================================================

    print("\n--- Phase 2: CSV filtering, cropping, resizing, and saving cell images ---")

    # 1. Read CSV and create HE ID -> Gene ID mapping
    try:
        df_matched = pd.read_csv(CSV_FILE_PATH)
        # Assume 1st column is HE mask ID, 2nd column is Gene ID
        id_map_df = df_matched.iloc[:, [0, 1]].copy()
        
        # Ensure ID columns are integers
        id_map_df.iloc[:, 0] = id_map_df.iloc[:, 0].astype(int) 
        
        # Create mapping dictionary {HE_Mask_ID: Gene_ID}
        HE_ID_TO_GENE_ID = id_map_df.set_index(id_map_df.columns[0])[id_map_df.columns[1]].to_dict()
        
        CELL_IDS_TO_PROCESS = set(HE_ID_TO_GENE_ID.keys())
        
        if not CELL_IDS_TO_PROCESS:
            print("Error: No valid cell IDs found in CSV for processing.")
            return
            
        print(f"Successfully loaded {len(CELL_IDS_TO_PROCESS)} target cell IDs from CSV.")
    except Exception as e:
        print(f"Error: Could not load or parse CSV file {CSV_FILE_PATH}. Info: {e}")
        return


    # 2. Check and create output directory
    if not os.path.exists(OUTPUT_CELLS_DIR):
        os.makedirs(OUTPUT_CELLS_DIR)
        print(f"Created output directory: {OUTPUT_CELLS_DIR}")

    # 3. Read Images
    try:
        # Use tifffile to read HE image
        he_image_rgb = tifffile.imread(HE_FILE_PATH) 
        if he_image_rgb.ndim == 3 and he_image_rgb.shape[2] == 3:
            # Convert to BGR order (OpenCV default)
            he_image = cv2.cvtColor(he_image_rgb, cv2.COLOR_RGB2BGR)
        else:
            he_image = he_image_rgb
    except Exception:
        # Fallback to cv2.imread
        he_image = cv2.imread(HE_FILE_PATH, cv2.IMREAD_COLOR) 
        if he_image is None:
            print(f"Error: Could not load HE image {HE_FILE_PATH}. Please check file.")
            return
    
    # 4. Prepare mask attributes
    try:
        # Reload dilated mask
        cell_mask_dilated = tifffile.imread(OUTPUT_MASK_PATH)
    except Exception as e:
        print(f"Error: Could not load dilated mask {OUTPUT_MASK_PATH}.")
        return

    if he_image.shape[:2] != cell_mask_dilated.shape[:2]:
        print("Error: HE image and dilated mask dimensions mismatch!")
        return

    # Identify all individual cell instances using dilated mask
    all_props = regionprops(cell_mask_dilated) 
    safety_margin = 5
    total_processed_cells = 0
    prop_map = {prop.label: prop for prop in all_props}

    print(f"Starting cropping for {len(CELL_IDS_TO_PROCESS)} matched cell instances...")

    # 5. Iterate, crop, and save
    for cell_id in tqdm(CELL_IDS_TO_PROCESS, desc="Cropping, scaling, and padding"):
        
        if cell_id not in prop_map:
            continue 
        
        prop = prop_map[cell_id]
        
        ymin, xmin, ymax, xmax = prop.bbox
        
        # 1. Apply safety margin and ensure boundary constraints
        x_start = max(0, xmin - safety_margin)
        y_start = max(0, ymin - safety_margin)
        x_end = min(he_image.shape[1], xmax + safety_margin)
        y_end = min(he_image.shape[0], ymax + safety_margin)
        
        w_safe = x_end - x_start
        h_safe = y_end - y_start

        if w_safe > 0 and h_safe > 0:
            # 2. Crop from HE image
            cell_patch = he_image[y_start:y_end, x_start:x_end]
            
            # 3. Resize and pad to unified size
            current_w, current_h = w_safe, h_safe
            
            # --- Unified scaling logic ---
            
            if current_w > TARGET_SIZE or current_h > TARGET_SIZE:
                # Downscale: longest edge fits TARGET_SIZE
                scale = min(TARGET_SIZE / current_w, TARGET_SIZE / current_h)
                interpolation_method = cv2.INTER_AREA 
            else:
                # Upscale: longest edge fits TARGET_SIZE
                scale = max(TARGET_SIZE / current_w, TARGET_SIZE / current_h)
                interpolation_method = cv2.INTER_CUBIC 

            new_w = int(current_w * scale)
            new_h = int(current_h * scale)
            
            # Hard limit dimensions to TARGET_SIZE
            new_w = min(new_w, TARGET_SIZE)
            new_h = min(new_h, TARGET_SIZE)
            
            if new_w <= 0 or new_h <= 0: continue
            
            # Execute scaling
            cell_patch = cv2.resize(cell_patch, (new_w, new_h), interpolation=interpolation_method)
            current_w, current_h = new_w, new_h

            # 4. Calculate padding for centering
            top = (TARGET_SIZE - current_h) // 2
            bottom = TARGET_SIZE - current_h - top
            left = (TARGET_SIZE - current_w) // 2
            right = TARGET_SIZE - current_w - left

            # Constant padding (128, 128, 128)
            padded_cell = cv2.copyMakeBorder(
                cell_patch, top, bottom, left, right, 
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            
            # 5. Get Gene ID and save as PNG
            gene_id = HE_ID_TO_GENE_ID.get(cell_id, 'UNKNOWN_GENE') 
            # Filename format: cell_id_{HE_Mask_ID}_gene_{Gene_ID}.png
            output_filename = os.path.join(OUTPUT_CELLS_DIR, f"cell_id_{cell_id}_gene_{gene_id}.png")
            cv2.imwrite(output_filename, padded_cell)
            total_processed_cells += 1

    print(f" {dataset_name} complete! Successfully cropped, processed, and saved {total_processed_cells} matched cell images to {OUTPUT_CELLS_DIR}")
    
    # -------------------- Create Completion Flag --------------------
    try:
        with open(DONE_FLAG_PATH, 'w') as f:
            f.write(f"Cropping completed successfully on {pd.Timestamp.now()}")
        print(f"Created completion flag file: {DONE_FLAG_FILENAME}")
    except Exception as e:
        print(f"Warning: Could not create completion flag file {DONE_FLAG_PATH}. Error: {e}")


# ==============================================================================
# 3. Main Entry: Directory Traversal
# ==============================================================================

def main():
    
    # 1. Get prefix1 path using configuration tool
    prefix1_path = get_prefix_path(CONFIG_FILE, CONFIG_KEY)
    if not prefix1_path: 
        print(f"Error: prefix1 path is invalid or config.txt format error. Exiting.")
        return

    # 2. Traverse all dataset directories
    
    all_dirs = [
        item for item in os.listdir(prefix1_path) 
        if os.path.isdir(os.path.join(prefix1_path, item)) and not item.startswith('.') # Exclude hidden directories
    ]
    
    if not all_dirs:
        print(f"Warning: No dataset subdirectories found in '{prefix1_path}' for processing.")
        return

    print(f"\n>>> Found {len(all_dirs)} dataset directories, starting batch processing.")

    for dataset_name in sorted(all_dirs): # Sort for consistent progress tracking
        dataset_path = os.path.join(prefix1_path, dataset_name)
        process_single_dataset(dataset_path)

    print("\nAll datasets processed successfully.")

if __name__ == "__main__":
    main()