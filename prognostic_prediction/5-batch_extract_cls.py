import torch
import torch.nn as nn
import tifffile
import numpy as np
import os
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm
import glob
import random

# ==============================================================================
# 1. Global Configuration
# ==============================================================================
MASK_DIR = "./cell_mask"      
HE_PT_DIR = "./hept"   
MODEL_PATH = "../best_multi_model_union.pth"  
OUTPUT_DIR = "./clspt_filtered"           
STATS_CSV_PATH = "./patch_density_stats_full2.csv"

FEAT_DIM = 64
NUM_CELLS = 32           # Transformer window size
MAX_RUNS_PER_PATCH = 15  # Maximum runs per patch (32 * 15 = 480 cells)
PATCH_SIZE = 256 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_CELLS_PER_PATCH = 32 # Qualification threshold
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# ==============================================================================
# 2. Model Definition
# ==============================================================================
class CellContextModule(nn.Module):
    def __init__(self, dim, num_cells):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True, dropout=0.0)
        self.aggregator = nn.TransformerEncoder(layer, num_layers=2)
        
    def forward(self, x, mask):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat((cls_tokens, x), dim=1)
        cls_mask_fill = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat((cls_mask_fill, mask), dim=1)
        return self.aggregator(x_with_cls, src_key_padding_mask=full_mask)

# ==============================================================================
# 3. Execution Logic
# ==============================================================================
def get_cls_features():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = CellContextModule(FEAT_DIM, NUM_CELLS).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['context'] if 'context' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    pt_files = sorted(glob.glob(os.path.join(HE_PT_DIR, "*.pt")))
    stats_records = []

    for pt_path in tqdm(pt_files, desc="Processing Files"):
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        # Note: Matching based on the filename rules provided
        mask_path = os.path.join(MASK_DIR, f"{base_name}_he_image_nuclei_seg_microns.tif")
        
        if not os.path.exists(mask_path):
            continue
            
        try:
            all_vectors = torch.load(pt_path, map_location='cpu')
            mask_img = tifffile.imread(mask_path)
            h, w = mask_img.shape[:2]
            cols, rows = int(np.ceil(w / PATCH_SIZE)), int(np.ceil(h / PATCH_SIZE))
            
            # --- Core Fix: Establish a safe ID mapping ---
            props = regionprops(mask_img)
            # Get all Labels present in the current image and sort them
            present_labels = sorted(list(set([p.label for p in props])))
            # Establish Label -> Feature matrix row index mapping
            label_to_idx = {label: i for i, label in enumerate(present_labels)}
            
            patch_cells = {}
            visited_labels = set() # Used to handle cases where multiple blocks merge into one cell

            for prop in props:
                label = prop.label
                # If this cell ID hasn't been processed and is in the mapping table
                if label in label_to_idx and label not in visited_labels:
                    idx_in_vec = label_to_idx[label]
                    
                    # Safety check: index cannot exceed the length of the feature matrix
                    if idx_in_vec < all_vectors.size(0):
                        y, x = prop.centroid
                        r, c = int(y // PATCH_SIZE), int(x // PATCH_SIZE)
                        r, c = min(r, rows - 1), min(c, cols - 1)
                        p_idx = r * cols + c
                        
                        if p_idx not in patch_cells: 
                            patch_cells[p_idx] = []
                        
                        patch_cells[p_idx].append(all_vectors[idx_in_vec])
                        visited_labels.add(label) # Mark the cell as assigned to a Patch

            valid_feats, valid_p_indices = [], []

            for p_idx in sorted(patch_cells.keys()):
                cells = patch_cells[p_idx]
                k = len(cells)
                
                if k >= MIN_CELLS_PER_PATCH:
                    # Sampling limit logic
                    max_allowed = MAX_RUNS_PER_PATCH * NUM_CELLS
                    if k > max_allowed:
                        cells = random.sample(cells, max_allowed)
                        k = len(cells)
                    else:
                        random.shuffle(cells)
                    
                    num_runs = int(np.ceil(k / NUM_CELLS))
                    sub_outputs = []
                    
                    for r in range(num_runs):
                        he_in = torch.zeros((1, NUM_CELLS, FEAT_DIM), device=DEVICE)
                        p_mask = torch.ones((1, NUM_CELLS), dtype=torch.bool, device=DEVICE)
                        
                        start, end = r * NUM_CELLS, (r + 1) * NUM_CELLS
                        chunk = cells[start:end]
                        
                        for i, vec in enumerate(chunk):
                            he_in[0, i] = vec.to(DEVICE)
                            p_mask[0, i] = False
                        
                        with torch.no_grad():
                            context_out = model(he_in, p_mask)
                            sub_outputs.append(context_out[0, 0, :].cpu())
                    
                    if sub_outputs:
                        patch_feat = torch.stack(sub_outputs).mean(dim=0)
                        valid_feats.append(patch_feat)
                        valid_p_indices.append(p_idx)

            if valid_feats:
                torch.save({
                    "features": torch.stack(valid_feats),
                    "p_indices": torch.tensor(valid_p_indices),
                    "grid_size": (rows, cols)
                }, os.path.join(OUTPUT_DIR, f"{base_name}.pt"))
                
                stats_records.append({
                    "he_name": base_name,
                    "valid_patches": len(valid_p_indices),
                    "total_cells_processed": len(visited_labels)
                })

        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    pd.DataFrame(stats_records).to_csv(STATS_CSV_PATH, index=False)
    print(f"Processing complete. Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    get_cls_features()