import os
import pandas as pd
import openslide
from tqdm import tqdm
from PIL import Image
import glob

# --- Global Configuration ---
CSV_PATH = './BRCA_input.csv'        # Input file
SOURCE_DIR = '/BRCA/slide'  # Source directory containing multiple folders
TARGET_DIR = './he'
OUTPUT_CSV_PATH = './data_list_exist.csv'

# Global control: which level to extract
TARGET_LEVEL = 1 

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

def process_data():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found: {CSV_PATH}")
        return

    # 1. Read data and get case_id from the first column
    df = pd.read_csv(CSV_PATH)
    # Assume the first column is the ID; use df.iloc[:, 0] to ensure accuracy
    case_ids = df.iloc[:, 0].astype(str).unique()
    
    # 2. Pre-build file index (accelerate matching to avoid repeated disk traversal in the loop)
    print("Scanning source directory files, please wait...")
    all_svs_files = glob.glob(os.path.join(SOURCE_DIR, "**/*.svs"), recursive=True)
    # Build a mapping from ID to file path
    path_map = {}
    for path in all_svs_files:
        filename = os.path.basename(path)
        # Build mapping if the filename starts with a case_id
        for cid in case_ids:
            if filename.startswith(cid):
                path_map[cid] = path
                break
    
    print(f"Index complete, found {len(path_map)} matching SVS files.")

    existing_rows = []
    skip_count = 0 
    
    # 3. Start processing
    pbar = tqdm(df.iterrows(), total=len(df), desc=f"Extracting Level {TARGET_LEVEL}")
    
    for index, row in pbar:
        case_id = str(row.iloc[0]) # Get first column ID
        target_name = f"{case_id}.tif"
        target_path = os.path.join(TARGET_DIR, target_name)
        
        pbar.set_postfix(case=case_id[:12], skipped=skip_count)
        
        # Check if file already exists
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
            existing_rows.append(row)
            skip_count += 1
            continue

        # Get the matched source path
        src_path = path_map.get(case_id)
        
        if src_path and os.path.exists(src_path):
            try:
                slide = openslide.OpenSlide(src_path)
                
                if slide.level_count > TARGET_LEVEL:
                    dims = slide.level_dimensions[TARGET_LEVEL]
                    # Read Region
                    img = slide.read_region((0, 0), TARGET_LEVEL, dims)
                    img = img.convert("RGB")
                    
                    # Save TIFF
                    img.save(target_path, compression="tiff_lzw")
                    existing_rows.append(row)
                else:
                    tqdm.write(f"Warning: Skipping {case_id} due to insufficient levels ({slide.level_count} levels)")
                
                slide.close()
                
            except Exception as e:
                tqdm.write(f"Error processing {case_id}: {e}")
        else:
            # Missing files log can be enabled if needed
            pass

    # 4. Save results
    if existing_rows:
        new_df = pd.DataFrame(existing_rows)
        new_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nProcessing complete! Successfully exported: {len(existing_rows)}")
    else:
        print("\nFailed to match or process any files.")

if __name__ == "__main__":
    process_data()