import os
import sys
import math
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Resolve shared modules independently of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "align_utils"))
from prognostic_models import GatedSlideEncoder

# ==========================================
# 1. Environment Configuration and Hyperparameters
# ==========================================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 64
N_SPLITS = 6 
MONITOR_MODE = "ACC" 
SAMPLE_SIZE = 256
CHECKPOINT_ROOT = Path("./checkpoints_brca_mil_gated_v2")  # Keep historical model files separate.
TEST_SEEDS = [44, 46, 48, 50, 52]  # Define 5 seeds


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


def build_classification_result(fold_aurocs, pooled_oof_auroc, coverage, n_samples):
    fold_aurocs = [float(score) for score in fold_aurocs]
    if len(fold_aurocs) != N_SPLITS or not all(
        math.isfinite(score) for score in fold_aurocs
    ):
        raise ValueError(f"classification requires {N_SPLITS} finite fold AUROCs")
    if int(coverage) != int(n_samples):
        raise ValueError(f"incomplete OOF coverage: {coverage}/{n_samples}")
    return {
        'fold_aurocs': fold_aurocs,
        'fold_mean': statistics.fmean(fold_aurocs),
        'fold_std': statistics.pstdev(fold_aurocs),
        'pooled_oof_auroc': float(pooled_oof_auroc),
        'coverage': int(coverage),
    }

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. Model and Data Classes
# ==========================================
class SelfCalibrationMIL(GatedSlideEncoder):
    """Gated-attention revision: requires a newly trained checkpoint."""
    def __init__(self, input_dim=64, L=128, n_heads=4):
        super().__init__(
            input_dim=input_dim, hidden_dim=L, num_heads=n_heads,
            embedding_dropout=0.0, attention_temperature=1.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(L, L), nn.ReLU(), nn.Dropout(0.5), nn.Linear(L, 2)
        )

    def forward(self, x):
        return self.classifier(super().forward(x))

class SprintDataset(Dataset):
    def __init__(self, df, root, is_train=False, sample_size=256):
        self.df, self.root, self.is_train, self.sample_size = df, Path(root), is_train, sample_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Core change: match the file reading logic of the first code
        feat_path = self.root / f"{row['case_submitter_id']}.pt"
        data = torch.load(feat_path, map_location='cpu')
        
        # Compatible with dictionary format {'features': ...} or direct tensor format
        feat = data['features'] if isinstance(data, dict) else data
        
        n = feat.shape[0]
        if self.is_train:
            # Training set: random sampling to increase robustness
            indices = torch.randperm(n)[:self.sample_size] if n >= self.sample_size else torch.randint(0, n, (self.sample_size,))
            feat = feat[indices]
        else:
            # Validation/Test set: fixed crop of first N
            feat = feat[:self.sample_size]
        return feat, int(row['label'])

# ==========================================
# 3. Core Training Function (Complete OOF K-fold evaluation)
# ==========================================
def train_and_validate(seed, df, feat_root):
    set_seed(seed)

    class_counts = df['label'].value_counts()
    if len(class_counts) != 2 or class_counts.min() < N_SPLITS:
        raise ValueError(
            f"{N_SPLITS}-fold stratification requires at least {N_SPLITS} samples "
            "from each class."
        )

    seed_dir = CHECKPOINT_ROOT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    X = df['case_submitter_id'].values
    y = df['label'].values
    splits = list(skf.split(X, y))
    validate_fold_partition(splits, len(df))

    fold_aurocs = []
    oof_probs = np.full(len(df), np.nan, dtype=float)
    oof_counts = np.zeros(len(df), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(splits):
        tr_df, v_df = df.iloc[train_idx], df.iloc[val_idx]
        
        # Sample weight balancing
        counts = np.bincount(tr_df['label'])
        weights = 1. / torch.tensor(counts, dtype=torch.float)
        samples_weights = weights[tr_df['label'].values]
        sampler = WeightedRandomSampler(samples_weights, len(tr_df), replacement=True)

        train_loader = DataLoader(SprintDataset(tr_df, feat_root, True, SAMPLE_SIZE), batch_size=1, sampler=sampler)
        val_loader = DataLoader(SprintDataset(v_df, feat_root, False, SAMPLE_SIZE), batch_size=1)
        
        model = SelfCalibrationMIL(input_dim=INPUT_DIM).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_m = 0
        fold_path = seed_dir / f"model_fold_{fold}.pth"
        
        for epoch in range(1, 101):
            model.train()
            for bags, label in train_loader:
                logits = model(bags.squeeze(0).to(DEVICE))
                loss = criterion(logits, label.to(DEVICE))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            model.eval()
            v_probs, v_preds, v_labels = [], [], []
            with torch.no_grad():
                for bags, label in val_loader:
                    logits = model(bags.squeeze(0).to(DEVICE))
                    v_probs.append(F.softmax(logits, dim=1)[:, 1].item())
                    v_preds.append(torch.argmax(logits, 1).item())
                    v_labels.append(label.item())
            
            acc = accuracy_score(v_labels, v_preds)
            try: auc = roc_auc_score(v_labels, v_probs)
            except: auc = 0.5
            
            curr_m = auc if MONITOR_MODE == "AUC" else acc
            if curr_m >= best_val_m:
                best_val_m = curr_m
                torch.save(model.state_dict(), fold_path)

        # Load the best model and predict this fold's held-out samples once.
        model.load_state_dict(torch.load(fold_path, map_location=DEVICE))
        model.eval()
        fold_val_probs = []
        with torch.no_grad():
            for bags, _ in val_loader:
                logits = model(bags.squeeze(0).to(DEVICE))
                fold_val_probs.append(F.softmax(logits, dim=1)[:, 1].item())

        fold_auc = roc_auc_score(v_df['label'].values, fold_val_probs)
        fold_aurocs.append(fold_auc)
        assign_oof(oof_probs, oof_counts, val_idx, fold_val_probs)
        print(
            f"Seed {seed} fold {fold + 1}/{N_SPLITS} "
            f"held-out AUROC: {fold_auc:.4f}"
        )

    require_complete_oof(oof_counts)
    pooled_oof_auroc = roc_auc_score(y, oof_probs)
    return build_classification_result(
        fold_aurocs,
        pooled_oof_auroc=pooled_oof_auroc,
        coverage=int(oof_counts.sum()),
        n_samples=len(df),
    )

# ==========================================
# 4. Main Program: Data Loading and Experiment Loop
# ==========================================
if __name__ == "__main__":
    # Path configuration
    csv_path = "./BRCA_input.csv" 
    feat_root = Path("./clspt_sampling_avg")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
    else:
        # Data preprocessing and path existence validation
        df = pd.read_csv(csv_path)
        existing = {f.stem for f in feat_root.glob("*.pt")}
        df = df[df['case_submitter_id'].isin(existing)].copy()
        
        print(f"Data loaded successfully: {len(df)} valid matching samples found.")
        
        seed_results = []
        print(f"Starting Multi-Seed Experiment (N={len(TEST_SEEDS)})")
        
        for s in TEST_SEEDS:
            print(f"\n>> Running Seed: {s} | Monitoring: {MONITOR_MODE}")
            result = train_and_validate(s, df, feat_root)
            seed_results.append(result)
            print(f">> Seed {s} fold AUROCs: {[round(x, 4) for x in result['fold_aurocs']]}")
            print(
                f">> Seed {s} fold mean AUROC: {result['fold_mean']:.4f} "
                f"+/- {result['fold_std']:.4f}"
            )
            print(f">> Seed {s} pooled OOF AUROC: {result['pooled_oof_auroc']:.4f}")
            print(f">> Seed {s} coverage={result['coverage']}/{len(df)}")

        # Primary statistics are based on each seed's six held-out fold metrics.
        seed_fold_means = [result['fold_mean'] for result in seed_results]
        seed_pooled_aurocs = [result['pooled_oof_auroc'] for result in seed_results]
        final_mean = np.mean(seed_fold_means)
        final_std = np.std(seed_fold_means)

        print("\n" + "*" * 40)
        print(f"FINAL TRUE {N_SPLITS}-FOLD ROBUSTNESS REPORT")
        print(f"Seeds: {TEST_SEEDS}")
        print(f"Per-seed fold mean AUROC: {[round(x, 4) for x in seed_fold_means]}")
        print(f"Mean AUROC: {final_mean:.4f}")
        print(f"Std Dev:    {final_std:.4f}")
        print(f"Per-seed pooled OOF AUROC: {[round(x, 4) for x in seed_pooled_aurocs]}")
        print(f"Mean pooled OOF AUROC: {np.mean(seed_pooled_aurocs):.4f}")
        print(f"Checkpoints saved in: {CHECKPOINT_ROOT}")
        print("*" * 40)
