import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm
import tifffile
from pathlib import Path
import sys

# -------------------- Path Configuration --------------------
INPUT_HE_DIR = Path("./he0")
INPUT_MASK_DIR = Path("./cell_mask0")
OUTPUT_CELLS_ROOT_DIR = Path("./cell0")
CSV_LIST_PATH = Path("./secondhover2.csv")
DONE_FLAG_FILENAME = "crop_done" # Completion flag filename

# -------------------- Processing Parameters --------------------
TARGET_SIZE = 128    # Unified target size
KERNEL_SIZE = 17     # Dilation kernel size (simulating cytoplasm range)
SAFETY_MARGIN = 5    # Safety margin for cropping boundaries

# ==============================================================================
# 1. Core Function: Process a single HE/Mask pair
# ==============================================================================

def process_single_pair(he_file_path: Path, mask_file_path: Path):
    """Processes all non-zero instances within a single HE image and its corresponding nuclear mask."""
    
    he_stem = he_file_path.stem 
    OUTPUT_CELLS_DIR = OUTPUT_CELLS_ROOT_DIR / he_stem
    DONE_FLAG_PATH = OUTPUT_CELLS_DIR / DONE_FLAG_FILENAME

    # 1. Check for breakpoint continuation flag
    if os.path.exists(DONE_FLAG_PATH):
        return 

    print(f"Starting crop: {he_stem}")

    # 2. Read image and mask
    try:
        # A. Read nuclear mask - use tifffile to avoid pixel limits
        nucleus_mask = tifffile.imread(str(mask_file_path))
        
        # B. Read HE image - use tifffile to bypass OpenCV's validateInputImageSize limits
        # tifffile can stably handle ultra-high resolution images like TCGA
        he_image_raw = tifffile.imread(str(he_file_path))
        
        if he_image_raw is None:
            raise ValueError("Failed to load image data via tifffile")

        # Convert to OpenCV format (RGB -> BGR) to ensure correct colors in cv2.imwrite
        if he_image_raw.ndim == 3 and he_image_raw.shape[2] == 3:
            he_image = cv2.cvtColor(he_image_raw, cv2.COLOR_RGB2BGR)
        else:
            he_image = he_image_raw # Grayscale or special formats

        if he_image.shape[:2] != nucleus_mask.shape[:2]:
            print(f"Dimension mismatch: {he_stem} (HE: {he_image.shape[:2]}, Mask: {nucleus_mask.shape[:2]})")
            return
        
        os.makedirs(OUTPUT_CELLS_DIR, exist_ok=True)

    except Exception as e:
        print(f"Failed to read file {he_stem}: {e}")
        return

    # 3. Get all cell instances
    all_props = regionprops(nucleus_mask) 
    if not all_props:
        print(f"Mask is empty: {he_stem}")
        _create_done_flag(DONE_FLAG_PATH, 0, he_stem)
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    total_processed = 0

    # 4. Loop through each cell
    img_h, img_w = he_image.shape[:2]
    
    for prop in tqdm(all_props, desc=f"Progress", leave=False):
        cell_id = prop.label
        ymin, xmin, ymax, xmax = prop.bbox
        
        # Local dilation (simulating cytoplasm)
        local_mask_slice = nucleus_mask[ymin:ymax, xmin:xmax]
        temp_mask = (local_mask_slice == cell_id).astype(np.uint8) * 255
        local_dilated_mask = cv2.dilate(temp_mask, kernel, iterations=1)
        
        dilated_y, dilated_x = np.nonzero(local_dilated_mask)
        if len(dilated_y) == 0: continue
        
        # Calculate global cropping coordinates (with safety margin)
        y_start = max(0, ymin - SAFETY_MARGIN)
        x_start = max(0, xmin - SAFETY_MARGIN)
        y_end = min(img_h, ymax + SAFETY_MARGIN)
        x_end = min(img_w, xmax + SAFETY_MARGIN)

        cell_patch = he_image[y_start:y_end, x_start:x_end]
        if cell_patch.size == 0: continue

        # Unified scaling to TARGET_SIZE
        h, w = cell_patch.shape[:2]
        max_dim = max(h, w)
        if max_dim == 0: continue
        
        scale = TARGET_SIZE / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Select interpolation method
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        cell_patch = cv2.resize(cell_patch, (new_w, new_h), interpolation=interp)
        
        # Pad to square (128x128)
        top = (TARGET_SIZE - new_h) // 2
        bottom = TARGET_SIZE - new_h - top
        left = (TARGET_SIZE - new_w) // 2
        right = TARGET_SIZE - new_w - left
        
        padded_cell = cv2.copyMakeBorder(
            cell_patch, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
        
        # Save individual cell image
        cv2.imwrite(str(OUTPUT_CELLS_DIR / f"{cell_id}.png"), padded_cell)
        total_processed += 1

    _create_done_flag(DONE_FLAG_PATH, total_processed, he_stem)
    print(f"Complete: {he_stem} (Saved {total_processed} cells)")

def _create_done_flag(done_flag_path: Path, count: int, name: str):
    """Creates a completion flag file."""
    try:
        with open(done_flag_path, 'w') as f:
            f.write(f"Done. Cells: {count}")
    except: pass

# ==============================================================================
# 2. Main Entry Point
# ==============================================================================

def main():
    # 1. Permissions and list check
    if not CSV_LIST_PATH.exists():
        print(f"Cannot find list file: {CSV_LIST_PATH}")
        return

    df = pd.read_csv(CSV_LIST_PATH)
    if 'he_name' not in df.columns or 'status' not in df.columns:
        print("CSV must contain 'he_name' and 'status' columns")
        return

    # Get list of Qualified status names
    qualified_names = set(df[df['status'] == 'Qualified']['he_name'].astype(str).tolist())
    print(f"List Statistics: {len(df)} total records, including {len(qualified_names)} Qualified samples")

    # 2. Scan files
    he_files = sorted(list(INPUT_HE_DIR.glob("*.tif")) + list(INPUT_HE_DIR.glob("*.png")))
    
    if not he_files:
        print(f"Warning: No .tif or .png files found in directory {INPUT_HE_DIR}")
        return

    # 3. Execution
    processed_count = 0
    for he_path in he_files:
        he_stem = he_path.stem
        
        # Filter A: Skip if not in the list or not Qualified
        if he_stem not in qualified_names:
            continue
            
        # Filter B: Check if already processed (breakpoint continuation)
        if (OUTPUT_CELLS_ROOT_DIR / he_stem / DONE_FLAG_FILENAME).exists():
            print(f"Skipping completed sample: {he_stem}")
            continue

        # Filter C: Check if Mask exists
        mask_path = INPUT_MASK_DIR / f"{he_stem}_he_image_nuclei_seg.tif"
        if not mask_path.exists():
            print(f"Warning: Missing Mask: {he_stem} (Expected path: {mask_path})")
            continue

        process_single_pair(he_path, mask_path)
        processed_count += 1

    print(f"\nAll tasks complete! A total of {processed_count} Qualified samples were processed in this run.")

if __name__ == "__main__":
    # Ensure output root directory exists
    os.makedirs(OUTPUT_CELLS_ROOT_DIR, exist_ok=True)
    main()