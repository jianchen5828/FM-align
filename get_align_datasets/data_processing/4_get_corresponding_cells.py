import numpy as np
import glob
import natsort
import tifffile
import os
import cv2
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
import argparse
import sys
import json # Used for reading Xenium JSON configuration


# Global variable declarations (used for multiprocessing Pool)
seg_hist = None
seg_xenium = None

def process_nucleus(nuc_id):
    """
    Calculate the overlap between a single H&E nucleus (nuc_id) and Xenium nuclei.
    """
    # seg_hist and seg_xenium are now in the global scope, registered and transformed
    nuc_coords = np.where(seg_hist == nuc_id)
    cells = seg_xenium[nuc_coords[0], nuc_coords[1]]
    cells_unique, counts = np.unique(cells, return_counts=True)
    
    # Exclude background (ID 0)
    nonzero_mask = cells_unique != 0
    matches = list(cells_unique[nonzero_mask])
    nonzero_counts = counts[nonzero_mask]
    
    n_overlap = len(matches)
    
    if n_overlap > 0:
        # Total pixel size of the H&E nucleus
        orig_size = [len(nuc_coords[0])] * n_overlap
        orig_id = [nuc_id] * n_overlap
        return orig_id, matches, nonzero_counts, orig_size
    else:
        return [], [], [], []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fp_seg_hist",
        default="he_image_nuclei_seg.tif",
        type=str,
        help="file path to nuclei segmented from H&E image",
    )
    parser.add_argument(
        "--fp_seg_xenium",
        default="xenium_nuclei.tif",
        type=str,
        help="file path to nuclei segmented from DAPI image",
    )
    parser.add_argument(
        "--fp_cgm",
        default="cell_gene_matrix.csv",
        type=str,
        help="file path to Xenium cell gene matrix from 4_get_xenium_cell_gene_matrix.py",
    )
    parser.add_argument(
        "--fp_out_matched_nuclei",
        default="matched_nuclei.csv",
        type=str,
        help="file path of output csv",
    )
    parser.add_argument(
        "--n_processes",
        default=24,
        type=int,
        help="max number of cpus to use",
    )
    parser.add_argument(
        "--changepath",
        default="./",
        type=str,
        help="file path to matrix transformation files (he_image_alignment.csv and experiment.xenium.json)",
    )
    parser.add_argument(
        "--min_overlap_fraction",
        default=0.3,
        type=float,
        help="min overlap fraction of H&E nucleus area between 2 nuclei",
    )
    parser.add_argument("--dir_output", default="data_processing", type=str)

    config = parser.parse_args()

    dir_output = config.dir_output
    if not os.path.exists(dir_output):
        sys.exit(f"Data not found in --dir_output: {dir_output}")

    # --------------------------------------------------------
    # I. Read images and registration parameters
    # --------------------------------------------------------
    
    fp_seg_xenium = os.path.join(config.dir_output, config.fp_seg_xenium)
    seg_xenium_full = tifffile.imread(fp_seg_xenium) # Full Xenium mask

    fp_seg_hist = os.path.join(config.dir_output, config.fp_seg_hist)
    seg_hist_raw = tifffile.imread(fp_seg_hist) # Raw H&E mask

    h_h, h_w = seg_hist_raw.shape
    x_h, x_w = seg_xenium_full.shape
    
    print(f"--- Original Image Dimensions ---")
    print(f"H&E mask raw size (H, W): {h_h}, {h_w}")
    print(f"Xenium mask full size (H, W): {x_h}, {x_w}")
    print("--------------------")
    
    # --------------------------------------------------------
    # II. Geometric Registration (Cropping Removed)
    # --------------------------------------------------------
    changepath = config.changepath
    
    # 1. Read 3x3 transformation matrix M (H&E pixels -> microns)
    fp_alignment = os.path.join(changepath, "he_image_alignment.csv")
    try:
        M_3x3_raw = np.loadtxt(fp_alignment, delimiter=',')
        if M_3x3_raw.shape != (3, 3):
            sys.exit(f"Registration file {fp_alignment} format error, expected 3x3 matrix, got {M_3x3_raw.shape}.")
    except Exception as e:
        sys.exit(f"Failed to read registration file {fp_alignment}: {e}")

    # 2. Read Xenium pixel size P (microns/pixel)
    fp_xenium_json = os.path.join(changepath, "experiment.xenium")
    if not os.path.exists(fp_xenium_json):
        fp_xenium_json = os.path.join(changepath, "experiment.xenium.json")

    try:
        with open(fp_xenium_json, 'r') as f:
            xenium_config = json.load(f)
            pixel_size = xenium_config.get("pixel_size", None)
            if pixel_size is None:
                 sys.exit("Could not find 'pixel_size' in Xenium JSON file.")
    except Exception as e:
        sys.exit(f"Failed to read Xenium JSON file: {e}")
        
    # 3. Calculate final pixel-to-pixel transformation matrix M_final
    P_factor = pixel_size
    print(f"Using scaling factor P = pixel_size = {P_factor}")
    
    S_3x3 = np.array([
        [P_factor, 0, 0],
        [0, P_factor, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    M_final_3x3 = np.dot(S_3x3, M_3x3_raw)
    M_final_2x3 = M_final_3x3[:2]
    
    # Using the full Xenium image as the target
    seg_xenium = seg_xenium_full # No cropping
    M_final_2x3_adjusted = M_final_2x3.copy() # No translation adjustment

    target_width = seg_xenium.shape[1]
    target_height = seg_xenium.shape[0]

    print(f"--- Target Dimensions ---")
    print(f"Final target size (H, W) using full Xenium mask size: {target_height}, {target_width}")
    print("--------------------")

    # --------------------------------------------------------
    # 6. Apply tiled affine transformation to H&E mask
    # --------------------------------------------------------
    
    # TILE_HEIGHT must be smaller than SHRT_MAX (32767)
    TILE_HEIGHT = 16000  
    
    # Initialize target array
    seg_hist_transformed = np.zeros_like(seg_xenium, dtype=np.uint32)
    
    if h_h > 32767:
        print(f"H&E raw height {h_h} > 32767, performing tiled transformation along height (Tile height: {TILE_HEIGHT}).")
    else:
        # Single tile transformation for small H&E
        TILE_HEIGHT = h_h
        print("H&E dimensions within safe range, performing single tiled transformation.")
    
    # Iterating through vertical tiles of the raw H&E image
    for y_start in tqdm(range(0, h_h, TILE_HEIGHT), desc="Processing H&E Y-tiles"):
        y_end = min(y_start + TILE_HEIGHT, h_h)
        
        # 1. Crop vertical slice of H&E mask
        src_tile = seg_hist_raw[y_start:y_end, :] 

        # 2. Adjust transformation matrix M_tile_adjusted
        # New_M[1, 2] = M[1, 2] + M[1, 1] * y_start
        M_tile_adjusted = M_final_2x3_adjusted.copy()
        M_tile_adjusted[1, 2] += M_final_2x3_adjusted[1, 1] * y_start
        
        # 3. Apply transformation to H&E slice
        warped_tile = cv2.warpAffine(
            src_tile.astype(np.float32), 
            M_tile_adjusted, 
            (target_width, target_height), 
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        
        # 4. Superimpose results onto final image
        seg_hist_transformed = np.where(warped_tile > 0, warped_tile, seg_hist_transformed)

    seg_hist = seg_hist_transformed.astype(np.uint32)
    print("\nH&E mask aligned to full Xenium space via tiled affine transformation.")
    
    if seg_xenium.shape != seg_hist.shape:
         sys.exit("Error: Transformed H&E dimensions do not match full Xenium dimensions.")

    # --------------------------------------------------------
    # III. Overlap calculation and filtering
    # --------------------------------------------------------

    nuc_unique_ids = np.unique(seg_hist)

    # Determine the number of CPUs to use
    num_cpus = min(
        cpu_count() - 2, config.n_processes
    )
    print("Processing using CPUs:", num_cpus)

    # Multiprocessing
    with Pool(processes=num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(process_nucleus, nuc_unique_ids),
                total=len(nuc_unique_ids),
            )
        )
    
    original_ids = []
    matching_ids = []
    overlap_amount = []
    original_size = []

    for orig_id, match_id, overlap_count, orig_size in results:
        original_ids.extend(orig_id)
        matching_ids.extend(match_id)
        overlap_amount.extend(overlap_count)
        original_size.extend(orig_size)

    original_ids = np.array(original_ids)
    matching_ids = np.array(matching_ids)
    overlap_amount = np.array(overlap_amount)
    original_size = np.array(original_size)

    combined = np.vstack((original_ids, matching_ids, overlap_amount, original_size)).T
    df = pd.DataFrame(
        combined, columns=["id_histology", "id_xenium", "overlap", "size_pix_histology"]
    )

    # 1. H&E Nuclei to Xenium Nuclei: select maximum overlap
    max_overlap_indices = df.groupby("id_histology")["overlap"].idxmax()

    df_match = df.loc[max_overlap_indices].copy()

    # remove background
    df_match = df_match[df_match["id_histology"] != 0]

    # Save the DataFrame to a CSV file (unfiltered)
    fp_out_matched_nuclei = os.path.join(dir_output, config.fp_out_matched_nuclei)
    df_match.to_csv(fp_out_matched_nuclei, index=False)

    # 2. Overlap threshold filtering
    df_match = df_match[
        df_match["overlap"]
        >= config.min_overlap_fraction * df_match["size_pix_histology"]
    ]

    # 3. Xenium Nuclei to H&E Nuclei: select maximum overlap (resolving many-to-one)
    df_match = df_match.sort_values(
        by=["id_xenium", "overlap"], ascending=[True, False]
    )
    df_match = df_match.drop_duplicates(subset="id_xenium", keep="first")

    mapping_dict = dict(zip(df_match["id_xenium"], df_match["id_histology"]))

    df_match_sorted = df_match.sort_values(by="id_histology")

    fp_out_matched_nuclei_f = fp_out_matched_nuclei.replace('.csv', '_filtered.csv')
    df_match_sorted.to_csv(fp_out_matched_nuclei_f, index=False)
    print("Saved filtered matched nuclei list:", fp_out_matched_nuclei_f)


    # --------------------------------------------------------
    # IV. Filter expression matrix
    # --------------------------------------------------------

    # Get expression matrix and filter cells
    fp_cgm = os.path.join(dir_output, config.fp_cgm)
    df_expr = pd.read_csv(fp_cgm, index_col=0)
    if 'cell_id' not in df_expr.columns:
        df_expr['cell_id'] = df_expr.index

    # IDs not in the dict will be nan
    df_expr["cell_id"] = df_expr["cell_id"].map(mapping_dict)
    df_expr.set_index("cell_id", inplace=True, drop=False)
    df_expr.index.name = None
    df_expr.dropna(inplace=True)
    df_expr.drop('cell_id', axis=1, inplace=True)
    df_expr.index = df_expr.index.astype(int)

    fp_out_expr_mat = fp_cgm.replace('.csv', '_filtered.csv')
    df_expr.to_csv(fp_out_expr_mat)
    print("Saved filtered expression matrix:", fp_out_expr_mat)