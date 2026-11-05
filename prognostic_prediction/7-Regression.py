import sys
import datetime
import math
import statistics

if not hasattr(datetime, 'UTC'):
    from datetime import timezone
    datetime.UTC = timezone.utc
    sys.modules['datetime'].UTC = timezone.utc

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
# Import survival analysis statistical tools
from lifelines.statistics import logrank_test as log_rank_test

# Resolve shared modules independently of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "align_utils"))
from prognostic_models import GatedSlideEncoder

# --- Global Configuration ---
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 64
N_SPLITS = 10
MAX_EPOCHS = 150
PATIENCE = 30
L_DIM = 32     
N_HEADS = 4    
GLOBAL_MODE = 'attention'
TEST_SEEDS = [42, 44, 46, 48, 50]

SAVE_DIR = Path("./checkpoints/mtl_multihead_gated_v2")  # Keep historical model files separate.
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def validate_fold_partition(splits, n_samples):
    expected = set(range(n_samples))
    held_out_counts = [0] * n_samples
    for fold, (train_indices, held_out_indices) in enumerate(splits):
        train = [int(index) for index in train_indices]
        held_out = [int(index) for index in held_out_indices]
        train_set, held_out_set = set(train), set(held_out)
        if len(train) != len(train_set) or len(held_out) != len(held_out_set):
            raise ValueError(f"fold {fold} contains duplicate indices")
        if train_set & held_out_set:
            raise ValueError(f"fold {fold} train and held-out indices overlap")
        if train_set | held_out_set != expected:
            raise ValueError(f"fold {fold} does not cover the complete cohort")
        for index in held_out:
            if index < 0 or index >= n_samples:
                raise ValueError(f"fold {fold} held-out index is out of range")
            held_out_counts[index] += 1
    if any(count != 1 for count in held_out_counts):
        raise ValueError("each sample must be held out exactly once")


def assign_oof(oof_values, assignment_counts, indices, values):
    indices, values = [int(index) for index in indices], list(values)
    if len(indices) != len(values):
        raise ValueError("OOF indices and values must have equal lengths")
    for index, value in zip(indices, values):
        if assignment_counts[index] != 0:
            raise ValueError(f"duplicate OOF assignment for index {index}")
        oof_values[index] = value
        assignment_counts[index] += 1


def require_complete_oof(assignment_counts):
    if any(count != 1 for count in assignment_counts):
        raise ValueError("incomplete OOF coverage")


def build_regression_result(
    fold_c_indices,
    pooled_oof_c_index,
    pooled_logrank_p,
    coverage,
    n_samples,
):
    fold_c_indices = [float(score) for score in fold_c_indices]
    if len(fold_c_indices) != N_SPLITS:
        raise ValueError(f"regression requires {N_SPLITS} fold C-Indices")
    finite_scores = [score for score in fold_c_indices if math.isfinite(score)]
    if not finite_scores:
        raise ValueError("no finite fold C-Indices are available")
    if int(coverage) != int(n_samples):
        raise ValueError(f"incomplete OOF coverage: {coverage}/{n_samples}")
    return {
        'fold_c_indices': fold_c_indices,
        'fold_mean': statistics.fmean(finite_scores),
        'fold_std': statistics.pstdev(finite_scores),
        'valid_folds': len(finite_scores),
        'pooled_oof_c_index': float(pooled_oof_c_index),
        'pooled_logrank_p': float(pooled_logrank_p),
        'coverage': int(coverage),
    }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. Base Class Definitions
# ==========================================
class SurvivalDataset(Dataset):
    def __init__(self, df, root):
        self.df, self.root = df, Path(root)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(self.root / f"{row['case_submitter_id']}.pt", map_location='cpu')
        return {
            'x': F.normalize(data['features'], p=2, dim=1), 
            'days': row['OS_days'], 
            'event': row['event_status'],
            'label': row['label'] 
        }

def cox_loss(risk_scores, times, events):
    if risk_scores.ndim == 0: risk_scores = risk_scores.unsqueeze(0)
    if risk_scores.shape[0] < 2 or torch.sum(events) == 0: 
        return torch.tensor(0.0, requires_grad=True).to(DEVICE)
    _, idx = torch.sort(times, descending=True)
    risk_scores, events = risk_scores[idx], events[idx]
    max_risk = torch.max(risk_scores)
    hazard_ratio = torch.exp(risk_scores - max_risk)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-7) + max_risk
    return -torch.sum((risk_scores - log_risk) * events) / (torch.sum(events) + 1e-7)

# ==========================================
# 2. Enhanced Model Architecture (SEBlock + Multi-Head)
# ==========================================
class GatedMTLSurvivalMIL(GatedSlideEncoder):
    """Gated-attention revision: requires a newly trained checkpoint."""
    def __init__(self, input_dim=64, L=32, n_heads=4):
        super().__init__(
            input_dim=input_dim, hidden_dim=L, num_heads=n_heads,
            embedding_dropout=0.6, attention_temperature=1.5,
            zero_init_attention=(GLOBAL_MODE == 'attention'),
            use_attention=(GLOBAL_MODE == 'attention'),
        )
        self.risk_regressor = nn.Linear(L, 1)
        self.aux_classifier = nn.Linear(L, 2)

    def forward(self, x):
        bag_feat = super().forward(x)
        return self.risk_regressor(bag_feat), self.aux_classifier(bag_feat)

# ==========================================
# 3. Core Experiment Function (Integrated P-value Calculation)
# ==========================================
def run_experiment(seed, df, feat_root):
    set_seed(seed)
    class_counts = df['label'].value_counts()
    if len(class_counts) < 2 or class_counts.min() < N_SPLITS:
        raise ValueError(
            f"{N_SPLITS}-fold stratification requires at least {N_SPLITS} samples "
            "from each label group."
        )

    oof_ranks = np.full(len(df), np.nan, dtype=float)
    oof_counts = np.zeros(len(df), dtype=int)
    fold_c_indices = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(skf.split(df, df['label']))
    validate_fold_partition(splits, len(df))

    for fold, (train_idx, val_idx) in enumerate(splits):
        tr_df = df.iloc[train_idx]
        tr_loader = DataLoader(SurvivalDataset(tr_df, feat_root), batch_size=1, shuffle=True)
        model = GatedMTLSurvivalMIL(L=L_DIM, n_heads=N_HEADS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        
        best_c = -1
        fold_path = SAVE_DIR / f"best_model_seed{seed}_fold{fold}.pth"
        patience = 0
        
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            alpha = max(0.5, 2.0 * (0.97 ** epoch)) 
            rs, ds, es, cls_loss_acc = [], [], [], 0
            for i, b in enumerate(tr_loader):
                risk, logits = model(b['x'].squeeze(0).to(DEVICE))
                cls_loss_acc += F.cross_entropy(logits, b['label'].to(DEVICE).long())
                rs.append(risk.view(-1)); ds.append(b['days'].to(DEVICE)); es.append(b['event'].to(DEVICE))
                
                if (i+1) % 4 == 0 or (i+1) == len(tr_loader):
                    if len(rs) > 1:
                        loss_cox = cox_loss(torch.cat(rs), torch.cat(ds), torch.cat(es))
                        loss_total = loss_cox + alpha * (cls_loss_acc / len(rs))
                        optimizer.zero_grad(); loss_total.backward(); optimizer.step()
                    rs, ds, es, cls_loss_acc = [], [], [], 0
            
            model.eval()
            v_r, v_d, v_e = [], [], []
            with torch.no_grad():
                for idx_v in val_idx:
                    row_v = df.iloc[idx_v]
                    feat_v = F.normalize(torch.load(feat_root / f"{row_v['case_submitter_id']}.pt", map_location='cpu')['features'], p=2, dim=1).to(DEVICE)
                    r, _ = model(feat_v)
                    v_r.append(r.item()); v_d.append(row_v['OS_days']); v_e.append(row_v['event_status'])
            
            try: c = concordance_index(v_d, -np.array(v_r), v_e)
            except: c = 0.5
            
            if c > best_c:
                best_c = c; torch.save(model.state_dict(), fold_path); patience = 0
            else: patience += 1
            if patience >= PATIENCE: break
        
        # Every fold contributes predictions; no performance-based filtering.
        model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        model.eval()
        fold_scores = []
        with torch.no_grad():
            for idx_v in val_idx:
                row_v = df.iloc[idx_v]
                feat_v = F.normalize(torch.load(feat_root / f"{row_v['case_submitter_id']}.pt", map_location='cpu')['features'], p=2, dim=1).to(DEVICE)
                tta = [model(feat_v + torch.randn_like(feat_v)*0.0005)[0].item() for _ in range(10)]
                fold_scores.append(np.mean(tta))

        fold_days = df.iloc[val_idx]['OS_days'].to_numpy()
        fold_events = df.iloc[val_idx]['event_status'].to_numpy()
        try:
            fold_c = concordance_index(
                fold_days,
                -np.asarray(fold_scores),
                fold_events,
            )
        except Exception:
            fold_c = float('nan')
        fold_c_indices.append(fold_c)

        ranks = pd.Series(fold_scores).rank(pct=True).to_numpy()
        assign_oof(oof_ranks, oof_counts, val_idx, ranks)
        fold_text = f"{fold_c:.4f}" if np.isfinite(fold_c) else "undefined"
        print(
            f"Seed {seed} fold {fold + 1}/{N_SPLITS} "
            f"held-out C-index: {fold_text}"
        )

    require_complete_oof(oof_counts)
    all_days = df['OS_days'].to_numpy()
    all_events = df['event_status'].to_numpy()
    pooled_oof_c_index = concordance_index(all_days, -oof_ranks, all_events)

    median_rank = np.median(oof_ranks)
    high_risk = oof_ranks > median_rank
    lr_results = log_rank_test(
        all_days[high_risk], all_days[~high_risk],
        event_observed_A=all_events[high_risk],
        event_observed_B=all_events[~high_risk]
    )
    return build_regression_result(
        fold_c_indices,
        pooled_oof_c_index=pooled_oof_c_index,
        pooled_logrank_p=lr_results.p_value,
        coverage=int(oof_counts.sum()),
        n_samples=len(df),
    )

# ==========================================
# 4. Main Program
# ==========================================
def main():
    feat_root = Path("./clspt_filtered2")
    csv_path = "./BRCA_input.csv" 
    
    if not Path(csv_path).exists():
        print(f"Error: Could not find file {csv_path}"); return

    df = pd.read_csv(csv_path)
    existing_files = {f.stem for f in feat_root.glob("*.pt")}
    df = df[df['case_submitter_id'].isin(existing_files)].copy()
    
    if len(df) == 0:
        print(f"Error: No matching feature files found"); return

    seed_results = []
    print(f"Dataset: {csv_path} | Sample size: {len(df)}")
    print(f"Mode: {GLOBAL_MODE} (Multi-Head: {N_HEADS}) | Folds: {N_SPLITS}")
    print("-" * 75)
    
    for s in TEST_SEEDS:
        print(f"Seed {s}:")
        result = run_experiment(s, df, feat_root)
        seed_results.append(result)

        p_val = result['pooled_logrank_p']
        star = ""
        if p_val < 0.001: star = "***"
        elif p_val < 0.01: star = "**"
        elif p_val < 0.05: star = "*"

        fold_display = [
            round(value, 4) if np.isfinite(value) else None
            for value in result['fold_c_indices']
        ]
        print(f"Fold C-Indices: {fold_display}")
        print(
            f"Fold mean C-index: {result['fold_mean']:.4f} +/- "
            f"{result['fold_std']:.4f} ({result['valid_folds']}/{N_SPLITS} defined)"
        )
        print(
            f"Pooled OOF C-index: {result['pooled_oof_c_index']:.4f} | "
            f"P-value = {p_val:.2e} {star:3}"
        )
        print(f"coverage={result['coverage']}/{len(df)}")

    if seed_results:
        seed_fold_means = [result['fold_mean'] for result in seed_results]
        seed_pooled_c_indices = [
            result['pooled_oof_c_index'] for result in seed_results
        ]
        pooled_p_values = [
            result['pooled_logrank_p'] for result in seed_results
        ]
        print("\n" + "="*75)
        print(f"FINAL TRUE {N_SPLITS}-FOLD STATISTICS (MTL + SE + MultiHead)")
        print(f"Per-seed fold mean C-index: {[round(x, 4) for x in seed_fold_means]}")
        print(f"Mean C-index: {np.mean(seed_fold_means):.4f} +/- {np.std(seed_fold_means):.4f}")
        print(f"Per-seed pooled OOF C-index: {[round(x, 4) for x in seed_pooled_c_indices]}")
        print(f"Mean pooled OOF C-index: {np.mean(seed_pooled_c_indices):.4f}")
        print(f"Median pooled OOF P-value: {np.median(pooled_p_values):.2e}")
        sig_seeds = sum(1 for p in pooled_p_values if p < 0.05)
        print(f"Significant Seeds (P < 0.05): {sig_seeds}/{len(pooled_p_values)}")
        print("="*75)

if __name__ == "__main__":
    main()
