# Obtain embedding vectors for all cells
import torch
import cv2
import numpy as np
import os
import pandas as pd
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path
from typing import Union, Any
import json

# --- Import model structure ---
models_dir = Path(__file__).resolve().parents[1] / "align_utils"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))
try:
    from alignment_backbone import ModifiedResNet2, load_alignment_backbone
except ImportError:
    print(f"Error: Could not import alignment_backbone.py from {models_dir}")
    exit()

# ==============================================================================
#                               Global Configuration
# ==============================================================================
# 1. Path Configuration
CONFIG_FILE = "config.txt"
CONFIG_KEY_PREFIX = "down_prefix"

def get_config_value(config_filename: str, key: str) -> Union[Any, None]:
    """General function: Read specific key value from the configuration file."""
    config_path = os.path.join(os.getcwd(), config_filename) 
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get(key)
    except:
        return None

PREFIX = get_config_value(CONFIG_FILE, CONFIG_KEY_PREFIX)

HE_FILE = os.path.join(os.path.dirname(PREFIX), 'he_image.tif')
MASK_FILE = os.path.join(PREFIX, 'he_image_nuclei_seg.tif')
CSV_FILE = os.path.join(PREFIX, 'matched_nuclei_filtered.csv')
CKPT_PATH = os.environ.get("FMALIGN_CHECKPOINT", str(Path(__file__).resolve().parents[1] / "align_and_cellular_context" / "best_joint_contrastive_model_fixed_heads_MILatt.pth"))
SAVE_PATH = os.path.join(PREFIX, 'cell_vec.pt')

# 2. Core Parameters
TARGET_SIZE = 128
SAFETY_MARGIN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512

# 3. Preprocessing Transformations
extraction_transform = transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
#                               Model Loading
# ==============================================================================
def load_extractor(ckpt_path):
    model = ModifiedResNet2(
        layers=(3, 4, 6, 3), 
        output_dim=64, 
        heads=8, 
        input_resolution=(TARGET_SIZE, TARGET_SIZE), 
        width=64
    ).to(DEVICE)
    if os.path.exists(ckpt_path):
        load_alignment_backbone(model, ckpt_path, map_location=DEVICE)
        model.eval()
        print(f"Model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"Weight file not found")

# ==============================================================================
#                               Main Processing Logic
# ==============================================================================
def main():
    # 1. Read IDs strictly according to CSV order
    df = pd.read_csv(CSV_FILE)
    csv_ordered_ids = df.iloc[:, 0].astype(int).tolist()
    unique_ids = set(csv_ordered_ids)
    print(f"--- Target cell count: {len(csv_ordered_ids)} (CSV order locked) ---")

    # 2. Read Large Images
    print("Loading image data...")
    he_image_rgb = tifffile.imread(HE_FILE)
    he_image = cv2.cvtColor(he_image_rgb, cv2.COLOR_RGB2BGR) if he_image_rgb.ndim == 3 else he_image_rgb
    mask = tifffile.imread(MASK_FILE)
    
    # 3. Establish ID Index (only process cells appearing in CSV)
    print("Analyzing cell spatial positions...")
    props = regionprops(mask)
    id_to_prop = {p.label: p for p in props if p.label in unique_ids}
    
    # 4. Initialize Model
    extractor = load_extractor(CKPT_PATH)
    
    # 5. Feature Collection Map (for reordering after out-of-order inference)
    feat_map = {}

    # 6. Batch Feature Extraction
    process_list = list(id_to_prop.keys())
    
    print(f"Starting feature extraction...")
    for i in tqdm(range(0, len(process_list), BATCH_SIZE), desc="Inference Progress"):
        batch_ids = process_list[i : i + BATCH_SIZE]
        batch_tensors = []
        valid_batch_ids = []

        for cell_id in batch_ids:
            prop = id_to_prop[cell_id]
            ymin, xmin, ymax, xmax = prop.bbox
            
            # Cropping
            y_s, y_e = max(0, ymin-SAFETY_MARGIN), min(he_image.shape[0], ymax+SAFETY_MARGIN)
            x_s, x_e = max(0, xmin-SAFETY_MARGIN), min(he_image.shape[1], xmax+SAFETY_MARGIN)
            
            cell_patch = he_image[y_s:y_e, x_s:x_e]
            if cell_patch.size == 0: continue

            # Preprocessing
            cell_patch_rgb = cv2.cvtColor(cell_patch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cell_patch_rgb)
            batch_tensors.append(extraction_transform(pil_img))
            valid_batch_ids.append(cell_id)

        if not batch_tensors: continue

        # Inference
        inputs = torch.stack(batch_tensors).to(DEVICE)
        with torch.no_grad():
            outputs = extractor(inputs).cpu()

        # Store in map
        for idx, cid in enumerate(valid_batch_ids):
            feat_map[cid] = outputs[idx]

    # 7. Core: Reassemble Tensor according to original CSV order
    print("Aligning vectors based on CSV order...")
    # Using list comprehension to extract features in the order of csv_ordered_ids
    final_ordered_list = [feat_map[cid] for cid in csv_ordered_ids]
    
    # Merge into a single large tensor [N, 64]
    final_tensor = torch.stack(final_ordered_list)
    
    # 8. Save
    torch.save(final_tensor, SAVE_PATH)
    
    print(f"\nProcessing complete!")
    print(f"Final feature tensor shape: {final_tensor.shape}")
    print(f"This order is perfectly consistent with {os.path.basename(CSV_FILE)}.")
    print(f"Output path: {SAVE_PATH}")

if __name__ == "__main__":
    main()