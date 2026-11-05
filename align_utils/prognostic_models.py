"""Shared slide-level modules for the gated-attention model revision.

This architecture revision requires newly trained checkpoints; it is not a
checkpoint-compatible replacement for the historical prognostic models.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x.mean(dim=0, keepdim=True))


class MultiHeadGatedAttention(nn.Module):
    """Per-head scores w_k^T[tanh(V_k h) * sigmoid(U_k h)]."""

    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        if min(input_dim, hidden_dim, num_heads) < 1:
            raise ValueError("Attention dimensions and head count must be positive")
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # Each head owns a separate block of rows in both projections.
        self.tanh_projection = nn.Linear(input_dim, num_heads * hidden_dim)
        self.sigmoid_projection = nn.Linear(input_dim, num_heads * hidden_dim)
        self.score_weight = nn.Parameter(torch.empty(num_heads, hidden_dim))
        nn.init.xavier_uniform_(self.score_weight)

    def forward(self, h):
        n = h.shape[0]
        v = torch.tanh(self.tanh_projection(h).reshape(n, self.num_heads, self.hidden_dim))
        u = torch.sigmoid(self.sigmoid_projection(h).reshape(n, self.num_heads, self.hidden_dim))
        return torch.einsum("nkd,kd->nk", v * u, self.score_weight)


class GatedSlideEncoder(nn.Module):
    """SE -> global mean / gated multi-head focal pooling -> gamma fusion."""

    def __init__(self, input_dim=64, hidden_dim=32, num_heads=4,
                 embedding_dropout=0.0, attention_temperature=1.0,
                 zero_init_attention=False, use_attention=True):
        super().__init__()
        if attention_temperature <= 0:
            raise ValueError("Attention temperature must be positive")
        self.input_dim = input_dim
        self.n_heads = num_heads
        self.attention_temperature = attention_temperature
        self.use_attention = use_attention
        self.se = SEBlock(input_dim)
        embedding = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        if embedding_dropout:
            embedding.append(nn.Dropout(embedding_dropout))
        self.embedding = nn.Sequential(*embedding)
        self.attention = MultiHeadGatedAttention(hidden_dim, max(1, hidden_dim // 2), num_heads)
        self.head_fusion = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )
        if zero_init_attention:
            # Preserve the regression model's initially uniform patch attention.
            nn.init.zeros_(self.attention.score_weight)

    def forward(self, x):
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected a non-empty [patches, {self.input_dim}] feature bag")
        h = self.embedding(self.se(x))
        g_feat = h.mean(dim=0, keepdim=True)
        if self.use_attention:
            scores = self.attention(h).transpose(0, 1)
            weights = F.softmax(scores / self.attention_temperature, dim=1)
            per_head_features = weights @ h
            l_feat = self.head_fusion(per_head_features.reshape(1, -1))
        else:
            l_feat = g_feat
        gamma = self.fusion_gate(torch.cat([l_feat, g_feat], dim=1))
        return gamma * l_feat + (1 - gamma) * g_feat
