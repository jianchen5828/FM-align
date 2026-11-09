import torch
import torch.nn as nn
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# -------------------- Import Path Configuration --------------------
# Adapt to the new model path
models_dir = Path(__file__).resolve().parents[1] / "align_utils"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

try:
    from alignment_backbone import ModifiedResNet2, load_alignment_backbone
except ImportError:
    print(f"Error: Could not import alignment_backbone.py from {models_dir}")
    sys.exit()

# ==========================================
# 1. Environment and Path Configuration
# ==========================================
DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
INPUT_CELL_ROOT = Path("./cell0")
OUTPUT_PT_DIR = Path("./hept0")
# Using the new checkpoint path
STAGE1_CHECKPOINT = os.environ.get("FMALIGN_CHECKPOINT", str(Path(__file__).resolve().parents[1] / "align_and_cellular_context" / "best_joint_contrastive_model_fixed_heads_MILatt.pth"))

# Optimized preprocessing for Stage 1 (using provided Norm parameters)
extraction_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SimpleImageDataset(Dataset):
    def __init__(self, img_paths): 
        self.img_paths = img_paths
    def __len__(self): 
        return len(self.img_paths)
    def __getitem__(self, i):
        try:
            img = Image.open(self.img_paths[i]).convert('RGB')
            return extraction_transform(img)
        except Exception as e:
            # Batch error handling needs to ensure consistent tensor shapes
            return torch.zeros((3, 128, 128))

# ==========================================
# 2. New Feature Extraction Engine (Stage 1 Adaptation)
# ==========================================
class FeatureExtractor:
    def __init__(self, ckpt_path):
        print(f"Loading Stage 1 model (Path: {ckpt_path})...")
        # Keep parameters consistent with Stage 1 training
        self.model = ModifiedResNet2(
            layers=(3, 4, 6, 3), 
            output_dim=64, 
            heads=8, 
            input_resolution=(128, 128), 
            width=64
        ).to(DEVICE)
        
        if os.path.exists(ckpt_path):
            load_alignment_backbone(self.model, ckpt_path, map_location=DEVICE)
            print("Alignment visual encoder loaded with strict=True")
        else:
            raise FileNotFoundError(f"Error: Weight file not found: {ckpt_path}")
            
        self.model.eval()

    @torch.no_grad()
    def process_folder(self, folder_path, batch_size=512):
        img_paths = sorted(list(Path(folder_path).glob("*.png")))
        if not img_paths:
            return None
        
        dataset = SimpleImageDataset(img_paths)
        # Use multi-threading to speed up reading
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
        
        all_vectors = []
        for batch in loader:
            # Stage 1 extracts features directly through the backbone
            feats = self.model(batch.to(DEVICE))
            all_vectors.append(feats.cpu()) 
        
        return torch.cat(all_vectors, dim=0)

# ==========================================
# 3. Execution
# ==========================================
def main():
    if not OUTPUT_PT_DIR.exists():
        OUTPUT_PT_DIR.mkdir(parents=True)

    # Initialize new engine
    try:
        extractor = FeatureExtractor(STAGE1_CHECKPOINT)
    except Exception as e:
        print(f"Initialization failed: {e}")
        return
    
    he_folders = [f for f in INPUT_CELL_ROOT.iterdir() if f.is_dir()]
    print(f"Starting Stage 1 feature extraction, {len(he_folders)} samples total...")

    for folder in tqdm(he_folders, desc="Extraction progress"):
        he_name = folder.name
        save_path = OUTPUT_PT_DIR / f"{he_name}.pt"
        
        if save_path.exists():
            continue

        bag_tensor = extractor.process_folder(folder)
        
        if bag_tensor is not None:
            # Save as [N, 64] or defined training dimension
            torch.save(bag_tensor, save_path)
        else:
            print(f"Warning: Skipping empty folder: {he_name}")

    print(f"\nAll complete! Features saved to: {OUTPUT_PT_DIR}")

if __name__ == "__main__":
    main()