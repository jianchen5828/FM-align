import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from tqdm import tqdm
import math
from typing import Union, List, Tuple, Dict, Any, Set
from collections import deque
import random
import copy 
import pandas as pd
from PIL import Image
import json
import sys

# ==============================================================================
# 0. Unified Configuration and Path Definitions
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration file paths
CONFIG_FILE = "config.txt"
CONFIG_KEY_PREFIX = "prefix1"
CONFIG_KEY_LIMIT = "CELL_LIMIT"
MASK_CSV_FILENAME = "mask.csv" 
UNION_INFO_FILENAME = "all_info.json" 

# Dataset internal paths (relative paths)
MATCHED_CSV_FILENAME = "processed/matched_nuclei_filtered.csv"
GENE_MATRIX_SUBPATH = "processed/cell_gene_matrix.csv"
IMAGE_FOLDER_SUBPATH = "processed/cropped_cells_zoomed"
MODEL_SAVE_PATH = "best_joint_contrastive_model_fixed_heads_MILatt.pth"

# Image configuration 
IMAGE_OUT_DIM = 64; IMAGE_PROJ_DIM = 128; TARGET_SIZE = 128       

# RNA configuration
D_MODEL = 512; RNA_N_HEAD = 8; RNA_NUM_LAYERS = 6
RNA_INPUT_DIM: Union[int, None] = None # Dynamically loaded (e.g., 510)
RNA_PROJ_DIM = 256      

# RNA augmentation configuration
RNA_GAUSSIAN_STD = 0.05    # Standard deviation for Gaussian noise
RNA_MASK_RATIO = 0.2       # Random mask ratio

# Cross-modal contrastive learning configuration 
CONTRAST_OUT_DIM = 64; CONTRAST_TEMP = 0.6; NEGATIVES_N = 512; MEMORY_BANK_SIZE = 4096 

# Training configuration
BATCH_SIZE = 64; LEARNING_RATE = 5e-5; WEIGHT_DECAY = 1e-5; EPOCHS = 500; NUM_WORKERS = 4
VAL_RATIO = 0.1; MAX_NON_IMPROVING_STREAK = 6; DELTA = 0.0             

PREFIX1: Union[str, None] = None
CELL_LIMIT: Union[int, None] = None

# Loss weight control (L1, L2, L3)
LOSS_WEIGHTS = [1.0, 1.0, 1.0] # Corresponding to [L_Img (NTXent), L_RNA (NTXent), L_CL (Gated InfoNCE)]

# ==============================================================================
# 1. Helper Functions (Config reading, Paths, Data Loading)
# ==============================================================================

def get_config_value(config_filename: str, key: str) -> Union[Any, None]:
    """General function: Read specific key value from the configuration file."""
    config_path = os.path.join(os.getcwd(), config_filename) 
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get(key)
    except:
        return None

def get_union_gene_count(prefix: str) -> int:
    """Read union_gene_count from all_info.json."""
    info_path = os.path.join(prefix, UNION_INFO_FILENAME)
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        gene_count = info.get("union_gene_count")
        if gene_count is None or not isinstance(gene_count, int):
            raise ValueError("Could not find 'union_gene_count' key in all_info.json or the value is not a valid integer.")
        return gene_count
    except Exception as e:
        print(f"Critical Error: Failed to read or parse the union gene info file {info_path}. Exception: {e}")
        sys.exit(1)

def get_full_gene_list(prefix: str, mask_csv_filename: str) -> Union[List[str], None]:
    """Read the full gene list (union gene list) from the column names of mask.csv."""
    mask_path = os.path.join(prefix, mask_csv_filename)
    try:
        df_mask_header = pd.read_csv(mask_path, nrows=0) 
        if len(df_mask_header.columns) > 1:
            # Assume first column is dataset_name, gene names start from the second column
            full_gene_list = df_mask_header.columns[1:].tolist()
            return full_gene_list
        else:
            print(f"Error: Insufficient columns in mask.csv to extract full gene list.")
            return None
    except Exception as e:
        print(f"Error: Could not read full gene list from {mask_csv_filename}: {e}")
        return None

def get_dataset_paths(prefix: str, dataset_name: str) -> Dict[str, str]:
    """Generate data paths for a single dataset based on configuration."""
    root_dir = os.path.join(prefix, dataset_name)
    return {
        'matched_csv': os.path.join(root_dir, MATCHED_CSV_FILENAME),
        'gene_matrix': os.path.join(root_dir, GENE_MATRIX_SUBPATH),
        'image_root': os.path.join(root_dir, IMAGE_FOLDER_SUBPATH),
    }

def load_single_dataset(dataset_name: str, prefix: str, limit: Union[int, None], full_gene_list: List[str], expected_gene_count: int) -> Tuple[List[str], np.ndarray]:
    """Load samples from a single dataset and expand gene expression matrix to full gene dimension (padding with 0)."""
    paths = get_dataset_paths(prefix, dataset_name)
    
    # 1. Read two IDs from CSV
    try:
        df_matched = pd.read_csv(paths['matched_csv'], header=0)
        he_ids = df_matched.iloc[:, 0].astype(str).str.strip().str.lower()
        cell_ids = df_matched.iloc[:, 1].astype(str).str.strip().str.lower() 
        combined_ids = [f"{he_id}#{cell_id}" for he_id, cell_id in zip(he_ids, cell_ids)]
        rna_matrix_keys = he_ids.tolist() 

        if limit is not None:
            combined_ids = combined_ids[:limit]
            rna_matrix_keys = rna_matrix_keys[:limit]
            print(f"CELL_LIMIT applied: Limiting number of cells to {len(combined_ids)}")
            
    except Exception as e:
        raise ValueError(f"Could not read or process matched CSV: {paths['matched_csv']}. Exception: {e}")
    
    # 2. Read Gene Matrix
    try:
        df_gene_matrix = pd.read_csv(paths['gene_matrix'], index_col=0) 
        rna_index_series = df_gene_matrix.index.astype(str).to_series().str.strip().str.lower() 
        rna_index_set = set(rna_index_series.tolist())
        
        valid_indices = [i for i, key in enumerate(rna_matrix_keys) if key in rna_index_set]
        final_combined_ids = [combined_ids[i] for i in valid_indices]
        final_rna_matrix_keys = [rna_matrix_keys[i] for i in valid_indices]

        if not final_combined_ids:
            return [], np.array([])
            
        df_gene_matrix.index = rna_index_series.tolist()
        dataset_rna_matrix = df_gene_matrix.loc[final_rna_matrix_keys].values.astype(np.float32)
        dataset_gene_names = df_gene_matrix.columns.tolist() 
        
    except Exception as e:
        raise ValueError(f"Could not read or process gene matrix CSV: {paths['gene_matrix']}. Exception: {e}")

    # 3. Gene dimension alignment and zero padding
    num_cells = dataset_rna_matrix.shape[0]
    num_union_genes = len(full_gene_list)
    padded_rna_matrix = np.zeros((num_cells, num_union_genes), dtype=np.float32)
    gene_to_union_index = {gene: full_gene_list.index(gene) 
                           for gene in dataset_gene_names 
                           if gene in full_gene_list}
                           
    if not gene_to_union_index:
        raise ValueError("Critical Error: No overlap between dataset gene names and union gene list.")

    dataset_gene_indices = [dataset_gene_names.index(gene) for gene in gene_to_union_index.keys()]
    union_gene_indices = list(gene_to_union_index.values())
    padded_rna_matrix[:, union_gene_indices] = dataset_rna_matrix[:, dataset_gene_indices]
    
    print(f"Data loaded successfully: cells={len(final_combined_ids)}, genes={padded_rna_matrix.shape[1]} (expanded from {dataset_rna_matrix.shape[1]})")
    
    return final_combined_ids, padded_rna_matrix

def load_all_datasets(prefix: str, mask_csv_filename: str, limit: Union[int, None], expected_gene_count: int) -> Tuple[List[str], np.ndarray]:
    mask_path = os.path.join(prefix, mask_csv_filename)
    all_unique_cell_ids = [] 
    all_rna_matrices = []
    
    try:
        df_mask = pd.read_csv(mask_path)
        dataset_names = df_mask.iloc[:, 0].astype(str).tolist()
    except Exception as e:
        print(f"Error: Could not read or process MASK CSV ({mask_path}): {e}")
        return [], np.array([])

    print(f"Successfully read {len(dataset_names)} dataset directory names from {mask_csv_filename}.")

    full_gene_list = get_full_gene_list(prefix, mask_csv_filename)
    if full_gene_list is None or len(full_gene_list) != expected_gene_count:
        raise SystemExit(f"Critical Error: Full gene list loading failed or count ({len(full_gene_list) if full_gene_list else 0}) does not match RNA_INPUT_DIM ({expected_gene_count}).")
    
    for dataset_name in dataset_names:
        print(f"\n--- Loading Dataset: {dataset_name} ---")
        try:
            combined_ids, padded_rna_matrix = load_single_dataset(
                dataset_name=dataset_name, 
                prefix=prefix, 
                limit=limit, 
                full_gene_list=full_gene_list, 
                expected_gene_count=expected_gene_count
            )
            
            if not combined_ids: continue
            
            unique_cell_ids = [f"{dataset_name}/{cid}" for cid in combined_ids]
            all_unique_cell_ids.extend(unique_cell_ids)
            all_rna_matrices.append(padded_rna_matrix)

        except ValueError as e:
            print(f"Error: Exception occurred while processing dataset {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"Error: Unknown exception occurred while processing dataset {dataset_name}: {e}")
            continue

    if not all_rna_matrices:
        print("Error: All datasets failed to load.")
        return [], np.array([])

    final_rna_matrix = np.concatenate(all_rna_matrices, axis=0)
    
    print(f"\nAll datasets aggregated: total cells={len(all_unique_cell_ids)}, genes={final_rna_matrix.shape[1]}")
    return all_unique_cell_ids, final_rna_matrix


# ==============================================================================
# 2. Model Structure Definition
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class GeneTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int = D_MODEL, nhead: int = RNA_N_HEAD, num_layers: int = RNA_NUM_LAYERS, dropout: float = 0.1):
        super().__init__()
        self.gene_projection = nn.Linear(1, d_model) 
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.final_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, rna_data: torch.Tensor) -> torch.Tensor:
        # rna_data shape: (batch_size, input_dim)
        x = self.gene_projection(rna_data.unsqueeze(-1)) # (B, L, 1) -> (B, L, D_MODEL)
        x = x.transpose(0, 1) # (L, B, D_MODEL) - Required by Transformer
        x = self.pos_encoder(x)
        x_encoded = self.transformer_encoder(x)
        x_encoded = x_encoded.transpose(0, 1) # (B, L, D_MODEL)
        F_RNA_tokens = self.final_projection(x_encoded) # (B, L, output_dim)
        F_RNA = F_RNA_tokens.mean(dim=1) # (B, output_dim)
        return F_RNA 
        
class Bottleneck(nn.Module): 
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ModifiedResNet2(nn.Module):
    # ResNet-50 Architecture
    def __init__(self, layers: List[int], output_dim: int, heads: int, input_resolution: Tuple[int, int] = (TARGET_SIZE, TARGET_SIZE), width: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.inplanes = width
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        def _make_layer(block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)
        
        self.layer1 = _make_layer(Bottleneck, width, layers[0])
        self.layer2 = _make_layer(Bottleneck, width * 2, layers[1], stride=2)
        self.layer3 = _make_layer(Bottleneck, width * 4, layers[2], stride=2)
        self.layer4 = _make_layer(Bottleneck, width * 8, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * Bottleneck.expansion, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h_cell = self.fc(x.flatten(1)) 
        return h_cell

class ProjectionHead(nn.Module):
    # Simple MLP Projection Head
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossModalGatedHeads(nn.Module):
    """
    Cross-modal head based on CHIEF (MIL) gated attention mechanism.
    Uses image features as a Gate to modulate RNA features via Tanh and Sigmoid branches.
    """
    def __init__(self, gater_dim: int, anchor_dim: int, hidden_dim: int = CONTRAST_OUT_DIM):
        super().__init__()
        
        # CHIEF core: Gated Attention structure
        # Branch 1: V (Tanh) - Extracts non-linear candidate information
        self.attention_V = nn.Sequential(
            nn.Linear(gater_dim, 512),
            nn.Tanh()
        )
        
        # Branch 2: U (Sigmoid) - Gating/Weight filtering
        self.attention_U = nn.Sequential(
            nn.Linear(gater_dim, 512),
            nn.Sigmoid()
        )
        
        # Attention weighting layer (corresponding to w in CHIEF)
        # Maps the gated 512-dim features back to anchor_dim (gene dimension) to form a dynamic mask
        self.attention_w = nn.Linear(512, anchor_dim)

        # Projection head (consistent with original contrastive learning flow)
        self.gated_anchor_projector = nn.Sequential(
            nn.Linear(anchor_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )
        
        self.positive_projector = nn.Sequential(
            nn.Linear(gater_dim, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )
        
    def forward(self, anchor_feature: torch.Tensor, gater_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        anchor_feature: RNA features [B, RNA_INPUT_DIM]
        gater_feature: Image features [B, IMAGE_OUT_DIM]
        """
        # 1. Calculate CHIEF-style Gated Attention Score
        a_v = self.attention_V(gater_feature)  # [B, 512]
        a_u = self.attention_U(gater_feature)  # [B, 512]
        
        # Element-wise multiplication (Gate mechanism)
        gated_context = a_v * a_u             # [B, 512]
        
        # Generate mask for RNA dimension
        gating_mask = torch.sigmoid(self.attention_w(gated_context)) # [B, anchor_dim]
        
        # 2. Modulate RNA features (Anchor)
        gated_anchor_feature = anchor_feature * gating_mask
        
        # 3. Project to contrastive space
        anchor = self.gated_anchor_projector(gated_anchor_feature)
        positive = self.positive_projector(gater_feature)
        
        return anchor, positive

class JointContrastiveModel(nn.Module):
    def __init__(self, rna_input_dim: int, image_output_dim: int = IMAGE_OUT_DIM):
        super().__init__()
        # 1. Encoders
        self.rna_encoder = GeneTransformer(input_dim=rna_input_dim, output_dim=rna_input_dim) 
        self.image_backbone = ModifiedResNet2(layers=[3, 4, 6, 3], output_dim=image_output_dim, heads=8) 
        
        # 2. Independent projection heads (for L1, L2 contrastive loss)
        self.image_projection_head = ProjectionHead(image_output_dim, IMAGE_PROJ_DIM, image_output_dim) 
        self.rna_projection_head = ProjectionHead(rna_input_dim, RNA_PROJ_DIM, rna_input_dim) 
        
        # 3. Cross-modal gated heads (for L3 cross-modal contrastive loss)
        self.rna_to_cell_heads = CrossModalGatedHeads(
            gater_dim=image_output_dim, anchor_dim=rna_input_dim, hidden_dim=CONTRAST_OUT_DIM
        )
        
    # forward only receives one image view and one RNA view
    def forward(self, v_i: torch.Tensor, rna_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Encoder outputs
        h_cell = self.image_backbone(v_i)       # Cell Feature (IMAGE_OUT_DIM)
        rna_feature = self.rna_encoder(rna_data) # RNA Feature (RNA_INPUT_DIM)
        
        # 2. Intra-modal projections (for L1, L2)
        h_cell_proj = self.image_projection_head(h_cell)
        rna_feature_proj = self.rna_projection_head(rna_feature)
        
        # 3. Cross-modal projections (for L3)
        rna_cl_proj, cell_cl_proj = self.rna_to_cell_heads(rna_feature, h_cell)
        
        return {
            "h_cell": h_cell, 
            "rna_feature": rna_feature,
            "h_cell_proj": h_cell_proj,
            "rna_feature_proj": rna_feature_proj,
            "rna_cl_proj": rna_cl_proj, 
            "cell_cl_proj": cell_cl_proj, 
        }

# ==============================================================================
# 3. RNA Augmentation Functions
# ==============================================================================

def rna_transform_A(rna_data: torch.Tensor) -> torch.Tensor:
    """RNA Augmentation View A: Gaussian noise + Random mask"""
    # 1. Random mask
    mask = torch.rand_like(rna_data) > RNA_MASK_RATIO
    masked_rna = rna_data * mask
    
    # 2. Gaussian noise
    noise = torch.randn_like(masked_rna) * RNA_GAUSSIAN_STD
    augmented_rna = masked_rna + noise
    
    return augmented_rna

def rna_transform_B(rna_data: torch.Tensor) -> torch.Tensor:
    """RNA Augmentation View B: Different random mask + Different Gaussian noise"""
    # 1. Different random mask
    current_mask_ratio = min(RNA_MASK_RATIO + 0.1, 0.5) 
    mask = torch.rand_like(rna_data) > current_mask_ratio 
    masked_rna = rna_data * mask
    
    # 2. Different Gaussian noise
    current_std = RNA_GAUSSIAN_STD * 1.5
    noise = torch.randn_like(masked_rna) * current_std 
    augmented_rna = masked_rna + noise
    
    return augmented_rna


# ==============================================================================
# 4. JointDataset Class (Modified for dual augmentation views)
# ==============================================================================

class JointDataset(Dataset):
    """Joint dataset for training and validation"""
    def __init__(self, unique_cell_ids: List[str], rna_data: np.ndarray, prefix_path: str, transform_A: transforms.Compose, transform_B: transforms.Compose):
        self.unique_cell_ids = unique_cell_ids 
        self.rna_data = torch.from_numpy(rna_data).float() 
        self.prefix_path = prefix_path 
        self.transform_A = transform_A # Augmentation for image view A
        self.transform_B = transform_B # Augmentation for image view B
        
    def __len__(self):
        return len(self.unique_cell_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_id = self.unique_cell_ids[index]
        
        try:
            dataset_name, combined_id = unique_id.split('/')
            he_id, cell_id_rna = combined_id.split('#') 
        except ValueError:
            raise Exception(f"Internal Error: Cell ID format error: {unique_id}")
            
        # 1. Image path construction and lookup
        image_dir = os.path.join(self.prefix_path, dataset_name, IMAGE_FOLDER_SUBPATH)
        exact_filename = f"cell_id_{he_id}_gene_{cell_id_rna}.png" 
        image_path = os.path.join(image_dir, exact_filename) 

        found_image_path = None
        if os.path.exists(image_path):
            found_image_path = image_path
        else:
            for ext in ['.tif', '.tiff', '.jpg', '.jpeg']:
                filename_other_ext = f"cell_id_{he_id}_gene_{cell_id_rna}{ext}"
                path_other_ext = os.path.join(image_dir, filename_other_ext)
                if os.path.exists(path_other_ext):
                    found_image_path = path_other_ext
                    break
        
        if not found_image_path:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(found_image_path).convert('RGB')
            # Dual Image Augmentation (L1)
            v_i_A = self.transform_A(image) 
            v_i_B = self.transform_B(image) 
        except Exception as e:
            raise RuntimeError(f"Error loading/processing image {unique_id}: {e}")

        # 2. RNA Data
        rna_data_orig = self.rna_data[index] # Original RNA Data (for L3)
        # Dual RNA Augmentation (L2)
        rna_data_A = rna_transform_A(rna_data_orig.clone()) 
        rna_data_B = rna_transform_B(rna_data_orig.clone()) 
        
        idx = torch.tensor(index, dtype=torch.long)
        
        # Returns: Image A, Image B, RNA A, RNA B, Original RNA (for L3), index
        return v_i_A, v_i_B, rna_data_A, rna_data_B, rna_data_orig, idx


# ==============================================================================
# 5. Training Helper Classes and Functions
# ==============================================================================

class MemoryBank:
    """Memory bank used for storing negative sample features"""
    def __init__(self, size: int, dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.bank = torch.randn(size, dim, device=device)
        self.bank = F.normalize(self.bank, dim=1)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor):
        batch_size = features.shape[0]
        ptr_end = self.ptr + batch_size
        if ptr_end < self.size:
            self.bank[self.ptr:ptr_end] = features
            self.ptr = ptr_end
        else:
            overflow = ptr_end - self.size
            self.bank[self.ptr:] = features[:self.size - self.ptr]
            self.bank[:overflow] = features[self.size - self.ptr:]
            self.ptr = overflow

    def dequeue(self, n: int) -> torch.Tensor:
        """Get n negative samples (approximate random sampling)"""
        indices = torch.randint(0, self.size, (n,), device=self.device)
        return self.bank[indices]

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    def __init__(self, temperature: float, device: torch.device):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        features = torch.cat([z_i, z_j], dim=0) 
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        logits = similarity_matrix[~mask].view(2 * batch_size, -1) 
        
        simclr_labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=self.device) - 1,
            torch.arange(batch_size, device=self.device)
        ]).long()
        
        return F.cross_entropy(logits, simclr_labels)

def stable_contrastive_loss(query: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, temp: float) -> torch.Tensor:
    """Calculate contrastive loss based on memory bank sampling (InfoNCE)"""
    if query.shape[0] == 0: return torch.tensor(0.0, device=query.device)

    query = F.normalize(query, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)
    
    l_pos = torch.einsum('bd,bd->b', query, positive).unsqueeze(1) / temp
    l_neg = torch.einsum('bd,nd->bn', query, negatives) / temp
    
    logits = torch.cat([l_pos, l_neg], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
    
    return F.cross_entropy(logits, labels)

class EarlyStopping:
    """Class for early stopping"""
    def __init__(self, patience=7, verbose=False, delta=DELTA, path=MODEL_SAVE_PATH):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==============================================================================
# 6. Main Training and Validation Functions
# ==============================================================================

def train_joint_contrastive(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, memory_bank_cell: MemoryBank, ntxent_loss: NTXentLoss) -> float:
    """Joint training function"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    weight_img, weight_rna, weight_cl = LOSS_WEIGHTS
    
    # Unpack 5 data tensors
    for v_i_A, v_i_B, rna_A, rna_B, rna_orig, _ in tqdm(dataloader, desc="Training"):
        v_i_A = v_i_A.to(DEVICE); v_i_B = v_i_B.to(DEVICE)
        rna_A = rna_A.to(DEVICE); rna_B = rna_B.to(DEVICE)
        rna_orig = rna_orig.to(DEVICE)
        
        optimizer.zero_grad()
        
        # 1. Calculate features needed for View A and L3 (using v_i_A and rna_orig)
        output_A = model(v_i_A, rna_orig)
        
        # 2. Calculate features needed for View B (run encoder and projection heads only)
        # Needed for L1: Image B projection features
        h_cell_B = model.image_backbone(v_i_B)
        h_cell_proj_B = model.image_projection_head(h_cell_B)
        
        # Needed for L2: RNA B projection features
        rna_feature_B = model.rna_encoder(rna_B)
        rna_feature_proj_B = model.rna_projection_head(rna_feature_B)

        # L1 intra-modal loss: Image (A vs B)
        loss_img = ntxent_loss(output_A['h_cell_proj'], h_cell_proj_B) 
        
        # L2 intra-modal loss: RNA (A vs B)
        loss_rna = ntxent_loss(output_A['rna_feature_proj'], rna_feature_proj_B)
        
        # L3 cross-modal loss: RNA(Orig)->Cell(A) Gated Contrastive Loss
        negatives_cell = memory_bank_cell.dequeue(NEGATIVES_N)
        loss_cl_rna_to_cell = stable_contrastive_loss(
            query=output_A['rna_cl_proj'], 
            positive=output_A['cell_cl_proj'], 
            negatives=negatives_cell, 
            temp=CONTRAST_TEMP
        )
        
        # 4. Update memory bank (using View A's Cell CL features)
        memory_bank_cell.enqueue(output_A['cell_cl_proj'].detach())
        
        # Total loss L_Total = w1*L1 + w2*L2 + w3*L3
        total_loss_batch = (
            weight_img * loss_img + 
            weight_rna * loss_rna + 
            weight_cl * loss_cl_rna_to_cell
        )
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        
    return total_loss / num_batches

@torch.no_grad()
def validate_model(model: nn.Module, dataloader: DataLoader, memory_bank_cell: MemoryBank, ntxent_loss: NTXentLoss) -> Tuple[float, float]:
    """Validation function"""
    model.eval()
    total_loss = 0
    num_batches = 0
    total_correct_cl = 0
    total_samples = 0
    
    weight_img, weight_rna, weight_cl = LOSS_WEIGHTS
    val_memory_bank_cell = copy.deepcopy(memory_bank_cell)

    # Unpack 5 data tensors
    for v_i_A, v_i_B, rna_A, rna_B, rna_orig, _ in tqdm(dataloader, desc="Validation"):
        v_i_A = v_i_A.to(DEVICE); v_i_B = v_i_B.to(DEVICE)
        rna_A = rna_A.to(DEVICE); rna_B = rna_B.to(DEVICE)
        rna_orig = rna_orig.to(DEVICE)
        
        # 1. Calculate features needed for View A and L3
        output_A = model(v_i_A, rna_orig)
        batch_size = v_i_A.shape[0]

        # 2. Calculate features needed for View B
        h_cell_B = model.image_backbone(v_i_B)
        h_cell_proj_B = model.image_projection_head(h_cell_B)
        rna_feature_B = model.rna_encoder(rna_B)
        rna_feature_proj_B = model.rna_projection_head(rna_feature_B)

        # L1, L2, L3 loss calculations
        loss_img = ntxent_loss(output_A['h_cell_proj'], h_cell_proj_B) 
        loss_rna = ntxent_loss(output_A['rna_feature_proj'], rna_feature_proj_B) 
        
        negatives_cell = val_memory_bank_cell.dequeue(NEGATIVES_N)
        loss_cl_rna_to_cell = stable_contrastive_loss(
            query=output_A['rna_cl_proj'], 
            positive=output_A['cell_cl_proj'], 
            negatives=negatives_cell, 
            temp=CONTRAST_TEMP
        )
        
        total_loss_batch = (
            weight_img * loss_img + 
            weight_rna * loss_rna + 
            weight_cl * loss_cl_rna_to_cell
        )
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        # Accuracy calculation (L3 Acc)
        query = F.normalize(output_A['rna_cl_proj'], dim=1)
        positive = F.normalize(output_A['cell_cl_proj'], dim=1)
        negatives = val_memory_bank_cell.dequeue(NEGATIVES_N)
        negatives = F.normalize(negatives, dim=1)

        logits_pos = torch.einsum('bd,bd->b', query, positive).unsqueeze(1) / CONTRAST_TEMP
        logits_neg = torch.einsum('bd,nd->bn', query, negatives) / CONTRAST_TEMP
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == 0).sum().item()
        
        total_correct_cl += correct
        total_samples += batch_size
        
        # Update validation memory bank
        val_memory_bank_cell.enqueue(output_A['cell_cl_proj'])

    avg_val_loss = total_loss / num_batches
    avg_val_acc_cl = total_correct_cl / total_samples if total_samples > 0 else 0.0
    
    return avg_val_loss, avg_val_acc_cl

# ==============================================================================
# 7. Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    
    print("=== Cellular Cross-Modal Joint Contrastive Learning (Joint Gated Contrastive Learning) ===")
    
    # 1. Config Loading 
    PREFIX1 = get_config_value(CONFIG_FILE, CONFIG_KEY_PREFIX)
    CELL_LIMIT = get_config_value(CONFIG_FILE, CONFIG_KEY_LIMIT)
    
    if PREFIX1 is None or CELL_LIMIT is None:
        raise SystemExit(f"Startup Failed: Could not read {CONFIG_KEY_PREFIX} or {CONFIG_KEY_LIMIT} from {CONFIG_FILE}.")

    try:
        RNA_INPUT_DIM = get_union_gene_count(PREFIX1)
    except SystemExit:
        sys.exit(1)
    
    print(f"--- Training Configuration ---")
    print(f"Root Directory: {PREFIX1}, CELL_LIMIT: {CELL_LIMIT}, Union Gene Count (RNA_INPUT_DIM): {RNA_INPUT_DIM}")
    print(f"Loss Weights (L_Img/L_RNA/L_CL): {LOSS_WEIGHTS}")
    print(f"RNA Augmentation: Gaussian Std={RNA_GAUSSIAN_STD}, Mask Ratio={RNA_MASK_RATIO}")
    print(f"========================================================\n")
    
    # 2. Data Loading
    all_unique_cell_ids, all_rna_matrix = load_all_datasets(
        prefix=PREFIX1, 
        mask_csv_filename=MASK_CSV_FILENAME, 
        limit=CELL_LIMIT, 
        expected_gene_count=RNA_INPUT_DIM
    )
    
    if not all_unique_cell_ids:
        raise SystemExit("Critical Error: No valid cell data loaded.")

    # 3. Image augmentation pipelines (for L1)
    image_transform_A = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.2, 1.0)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),   
        transforms.RandomRotation(degrees=15), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_transform_B = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.2, 1.0)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15), 
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Increased color jitter
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Dataset Partitioning
    full_dataset_size = len(all_unique_cell_ids)
    val_size = int(VAL_RATIO * full_dataset_size)
    train_size = full_dataset_size - val_size
    print(f"\nDataset Partitioning: Total={full_dataset_size}, Train={train_size}, Validation={val_size}")
    
    # Passing two augmentation pipelines to JointDataset
    full_dataset = JointDataset(
        unique_cell_ids=all_unique_cell_ids, 
        rna_data=all_rna_matrix, 
        prefix_path=PREFIX1, 
        transform_A=image_transform_A,
        transform_B=image_transform_B
    )
    
    joint_train_dataset, joint_val_dataset = random_split(full_dataset, [train_size, val_size])
    
    joint_train_dataloader = DataLoader(joint_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    joint_val_dataloader = DataLoader(joint_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    
    # 5. Model Initialization and Training
    joint_model = JointContrastiveModel(rna_input_dim=RNA_INPUT_DIM, image_output_dim=IMAGE_OUT_DIM).to(DEVICE)
    optimizer = optim.AdamW(joint_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    memory_bank_cell = MemoryBank(size=MEMORY_BANK_SIZE, dim=CONTRAST_OUT_DIM, device=DEVICE) 
    ntxent_loss = NTXentLoss(temperature=CONTRAST_TEMP, device=DEVICE)

    early_stopping = EarlyStopping(patience=MAX_NON_IMPROVING_STREAK, verbose=True, path=MODEL_SAVE_PATH)

    print(f"\nStarting Joint Training: L_Total = {LOSS_WEIGHTS[0]}*L_Img (L1) + {LOSS_WEIGHTS[1]}*L_RNA (L2) + {LOSS_WEIGHTS[2]}*L_RNA->Cell Gated CL (L3)")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n======== Epoch {epoch}/{EPOCHS} ========")
        
        avg_train_loss = train_joint_contrastive(
            joint_model, joint_train_dataloader, optimizer, memory_bank_cell, ntxent_loss
        )
        print(f"Epoch {epoch} Training Average Loss: {avg_train_loss:.4f}")
        
        avg_val_loss, avg_val_acc_cl = validate_model(
            joint_model, joint_val_dataloader, memory_bank_cell, ntxent_loss
        )
        print(f"Epoch {epoch} Validation Total Loss: {avg_val_loss:.4f}, L3 Acc (RNA->Cell): {avg_val_acc_cl:.4f}")
        
        # Early stopping logic based on total validation loss
        early_stopping(avg_val_loss, joint_model)
        if early_stopping.early_stop:
            print(f"Early Stopping Triggered! Training stopped at epoch {epoch}.")
            break
            
    print("\nScript execution completed. Best model saved.")