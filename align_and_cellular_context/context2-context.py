import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import os
import tifffile
from skimage.measure import regionprops
from tqdm import tqdm
import random

# ==========================================
# 1. Global Configuration
# ==========================================
CANCER_PATHS = [
    "./data/liver/",
    "./data/Breast_Cancer/"
]

FEAT_DIM = 64
CONTRAST_DIM = 32
NUM_CELLS = 32
PATCH_SIZE = 256
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 30  # Early stopping threshold: stop if no improvement for 30 consecutive epochs
VAL_SPLIT = 0.1
INIT_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "best_multi_model_union.pth"

# ==========================================
# 2. Model Definition (Unchanged)
# ==========================================

class CellContextModule(nn.Module):
    def __init__(self, dim, num_cells):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True, dropout=0.2)
        self.aggregator = nn.TransformerEncoder(layer, num_layers=2)
        
    def forward(self, x, mask):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_with_cls = torch.cat((cls_tokens, x), dim=1)
        cls_mask_fill = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat((cls_mask_fill, mask), dim=1)
        return self.aggregator(x_with_cls, src_key_padding_mask=full_mask)

class CrossModalGatedHeads(nn.Module):
    def __init__(self, he_dim, rna_raw_dim, hidden_dim):
        super().__init__()
        self.attention_V = nn.Linear(he_dim, 512)
        self.attention_U = nn.Linear(he_dim, 512)
        self.attention_w = nn.Linear(512, rna_raw_dim)
        self.rna_projector = nn.Sequential(
            nn.Linear(rna_raw_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.he_projector = nn.Sequential(
            nn.Linear(he_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6593)

    def forward(self, rna_raw, he_enhanced):
        a_v = torch.tanh(self.attention_V(he_enhanced))
        a_u = torch.sigmoid(self.attention_U(he_enhanced))
        gating_mask = torch.sigmoid(self.attention_w(a_v * a_u))
        return self.rna_projector(rna_raw * gating_mask), self.he_projector(he_enhanced)

# ==========================================
# 3. Data Processing (Union Padding Logic)
# ==========================================

def prepare_multi_data():
    print("Step 1: Building union gene list...")
    union_genes = set()
    valid_paths = []
    for path in CANCER_PATHS:
        rna_file = os.path.join(path, 'processed', 'cell_gene_matrix_filtered.csv')
        if os.path.exists(rna_file):
            valid_paths.append(path)
            df_cols = pd.read_csv(rna_file, index_col=0, nrows=0)
            union_genes.update(df_cols.columns)
    
    all_genes = sorted(list(union_genes))
    rna_dim = len(all_genes)
    print(f"Total union genes: {rna_dim}")

    all_data = []
    for path in valid_paths:
        cancer_name = os.path.basename(path.strip('/'))
        print(f"Loading and aligning data: {cancer_name}")
        proc_dir = os.path.join(path, 'processed')
        cell_vecs = torch.load(os.path.join(proc_dir, 'cell_vec.pt'), weights_only=False)
        df_bridge = pd.read_csv(os.path.join(proc_dir, 'matched_nuclei_filtered.csv'))
        
        # Use reindex for automatic zero padding of missing genes
        df_rna_raw = pd.read_csv(os.path.join(proc_dir, 'cell_gene_matrix_filtered.csv'), index_col=0)
        df_rna = df_rna_raw.reindex(columns=all_genes, fill_value=0.0)
        
        mask_path = os.path.join(proc_dir, "he_image_nuclei_seg.tif")
        if not os.path.exists(mask_path): mask_path = os.path.join(path, "he_image_nuclei_seg.tif")
        mask_img = tifffile.imread(mask_path)
        id_to_coord = {prop.label: prop.centroid for prop in regionprops(mask_img)}
        
        h, w = mask_img.shape[:2]
        rows, cols = (h + PATCH_SIZE - 1) // PATCH_SIZE, (w + PATCH_SIZE - 1) // PATCH_SIZE
        patch_cells = {i: [] for i in range(rows * cols)}
        
        rna_ids = set(df_rna.index.astype(int))
        for idx in range(len(cell_vecs)):
            heid = int(df_bridge.iloc[idx, 0])
            if heid in id_to_coord and heid in rna_ids:
                y, x = id_to_coord[heid]
                p_idx = min(int(y // PATCH_SIZE), rows - 1) * cols + min(int(x // PATCH_SIZE), cols - 1)
                patch_cells[p_idx].append({
                    'he': cell_vecs[idx],
                    'rna': torch.tensor(df_rna.loc[heid].values, dtype=torch.float32)
                })
        all_data.extend([v for v in patch_cells.values() if len(v) > 0])
    
    random.shuffle(all_data)
    split = int(len(all_data) * VAL_SPLIT)
    return all_data[split:], all_data[:split], rna_dim

# ==========================================
# 4. Training Logic (with Early Stopping)
# ==========================================

def run_batch(batch, context_net, align_net, rna_dim, optimizer=None):
    B = len(batch)
    he_in = torch.zeros((B, NUM_CELLS, FEAT_DIM)).to(DEVICE)
    rna_in = torch.zeros((B, NUM_CELLS, rna_dim)).to(DEVICE)
    p_mask = torch.ones((B, NUM_CELLS), dtype=torch.bool).to(DEVICE)

    for i, cells in enumerate(batch):
        selected = random.sample(cells, min(len(cells), NUM_CELLS))
        for j, cell in enumerate(selected):
            he_in[i, j], rna_in[i, j], p_mask[i, j] = cell['he'], cell['rna'], False

    context_out = context_net(he_in, p_mask)
    he_features = context_out[:, 1:, :]
    valid_idx = ~p_mask.view(-1)
    he_flat = he_features.reshape(-1, FEAT_DIM)[valid_idx]
    rna_flat = rna_in.reshape(-1, rna_dim)[valid_idx]

    rna_emb, he_emb = align_net(rna_flat, he_flat)
    rna_emb, he_emb = F.normalize(rna_emb, dim=-1), F.normalize(he_emb, dim=-1)
    
    logits = torch.matmul(rna_emb, he_emb.T) * align_net.logit_scale.exp()
    labels = torch.arange(len(logits)).to(DEVICE)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    
    if optimizer:
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item(), torch.diagonal(torch.matmul(rna_emb, he_emb.T)).mean().item()

def train():
    tr_data, val_data, rna_dim = prepare_multi_data()
    context_net = CellContextModule(FEAT_DIM, NUM_CELLS).to(DEVICE)
    align_net = CrossModalGatedHeads(FEAT_DIM, rna_dim, CONTRAST_DIM).to(DEVICE)
    
    optimizer = optim.AdamW(list(context_net.parameters()) + list(align_net.parameters()), lr=INIT_LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_sim = -1.0
    patience_counter = 0  # Initialize counter

    print(f"Starting training | Early stopping threshold: {PATIENCE}")
    
    for epoch in range(EPOCHS):
        context_net.train(); align_net.train()
        t_losses, t_sims = [] , []
        pbar = tqdm(range(0, len(tr_data), BATCH_SIZE), desc=f"Epoch {epoch+1}")
        for i in pbar:
            l, s = run_batch(tr_data[i:i+BATCH_SIZE], context_net, align_net, rna_dim, optimizer)
            t_losses.append(l); t_sims.append(s)
        
        context_net.eval(); align_net.eval()
        v_sims = []
        with torch.no_grad():
            for i in range(0, len(val_data), BATCH_SIZE):
                _, s = run_batch(val_data[i:i+BATCH_SIZE], context_net, align_net, rna_dim)
                v_sims.append(s)
        
        avg_v_sim = np.mean(v_sims)
        print(f"Epoch {epoch+1} Results | Loss: {np.mean(t_losses):.4f} | Val Sim: {avg_v_sim:.4f}")
        
        # Early stopping logic
        if avg_v_sim > best_sim:
            best_sim = avg_v_sim
            patience_counter = 0  # Reset counter
            torch.save({'context': context_net.state_dict(), 'align': align_net.state_dict()}, SAVE_PATH)
            print(f"Model updated (Best Sim: {best_sim:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement count: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered, training ended.")
                break
        
        scheduler.step()

if __name__ == "__main__":
    train()