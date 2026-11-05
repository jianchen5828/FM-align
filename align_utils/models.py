import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Tuple, List, Optional

# ==============================================================================
#                             Image Model Components (ResNet)
# ==============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # 1x1 conv, BN, ReLU
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1  = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 3x3 conv, BN, ReLU
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        
        # AvgPool for downsampling (ResNet-D style)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 1x1 conv, BN (output dimension expansion)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3  = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride
        
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor):
        identity = x
        
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu3(out)
        return out

class ModifiedResNet2(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=(128, 128), width=64):
        super().__init__()
        self.output_dim = output_dim
        self.inplanes = width
        
        self.conv1 = nn.Conv2d(3, width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        
        # Calculate feature map size after layers
        # Input: 128x128
        # Conv1: 64x64
        # MaxPool: 32x32
        # Layer1: 32x32
        # Layer2: 16x16
        # Layer3: 8x8
        # Layer4: 4x4
        
        # Attention Pooling
        # Assuming input 128x128 -> 4x4 feature map
        # If input size changes, this needs to be adaptive or calculated
        # For simplicity, we use AdaptiveAvgPool to 1x1 then flatten, OR use the AttentionPool from the reference
        # The reference used AttentionPool2d2. Let's implement a simple AvgPool + Linear for robustness if AttentionPool is complex to port without full context
        # But let's try to stick to the reference.
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * Bottleneck.expansion, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride))
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(z_i.device)

        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * batch_size, 1)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        negative_samples = sim[~mask].reshape(2 * batch_size, -1)

        labels = torch.zeros(2 * batch_size).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)

# ==============================================================================
#                             Gene Model Components (Transformer)
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class GeneTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, output_dim: int = None):
        super(GeneTransformer, self).__init__()
        
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.d_model = d_model
        
        self.gene_projection = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_dim + 1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, activation=F.gelu
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Output projection to match desired output dimension (e.g. 280 in reference)
        # The reference used attention weights as output, but here we might want a feature vector.
        # If we want to match the reference exactly:
        # Reference: "final_output = np.vstack(all_outputs)" where all_outputs comes from attention weights.
        # "output_matrix = cls_attn_to_all.mean(dim=1).cpu().numpy()"
        # So the output is the attention weights of CLS token to all genes.
        # Dimension = num_genes.
        
        # However, for a trainable model, we usually use the CLS token output.
        # But let's stick to the reference's idea: The output is related to genes.
        # Wait, if the output is attention weights, it's [Batch, Num_Genes].
        # If we use CLS token, it's [Batch, d_model].
        
        # In `contrast_with_gate.py`:
        # TRANSFORMER_DIM = transformer_output.shape[1] # 280 (which matches gene count in some datasets)
        # So the output dimension is indeed the number of genes (or whatever the attention size is).
        
        # Let's implement a flexible head.
        # If we want to use the CLS token embedding:
        self.to_out = nn.Linear(d_model, self.output_dim)

    def forward(self, rna_data: torch.Tensor):
        # rna_data: [Batch, Num_Genes]
        batch_size = rna_data.size(0)
        
        x = rna_data.unsqueeze(-1) # [Batch, Num_Genes, 1]
        x = self.gene_projection(x) # [Batch, Num_Genes, d_model]
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) # [Batch, Num_Genes+1, d_model]
        
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        x = self.encoder(x) # [Batch, Num_Genes+1, d_model]
        
        # Take CLS token output
        cls_out = x[:, 0, :] # [Batch, d_model]
        
        # Project to output dim
        out = self.to_out(cls_out)
        
        return out

class GeneMLP(nn.Module):
    """
    A simple MLP encoder for gene expression data.
    Often more effective than Transformers for tabular/sparse gene data.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# ==============================================================================
#                             Gated Contrastive Model
# ==============================================================================

class GatedStableContrastiveModel(nn.Module):
    def __init__(self, transformer_dim, cell_dim, hidden_dim=16):
        super().__init__()
        
        # Gating Generator: Cell -> Gate (for RNA)
        self.gating_generator = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, transformer_dim),
            nn.Sigmoid() 
        )

        # Gated Anchor Projector: Gated RNA -> Output
        self.gated_anchor_projector = nn.Sequential(
            nn.Linear(transformer_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim), 
        )
        
        # Cell Encoder: Cell -> Output
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
        )
        
    def forward(self, transformer_input, cell_input):
        # transformer_input: RNA features
        # cell_input: Image features
        
        # 1. Generate Gate
        gamma = self.gating_generator(cell_input) 
        
        # 2. Apply Gate
        gated_transformer_input = transformer_input * gamma 
        
        # 3. Project RNA
        trans_out = self.gated_anchor_projector(gated_transformer_input)
        
        # 4. Project Cell
        cell_out = self.cell_encoder(cell_input)

        # 5. Normalize
        trans_out = F.normalize(trans_out, p=2, dim=1)
        cell_out = F.normalize(cell_out, p=2, dim=1)
        
        return trans_out, cell_out

class StableMemoryBank:
    def __init__(self, size, feature_dim, device):
        self.size = size
        self.feature_dim = feature_dim
        self.device = device
        self.bank = torch.zeros(size, feature_dim, dtype=torch.float32, device=device)
        self.ptr = 0
        
    def update(self, features):
        B = features.size(0)
        if B > self.size:
            features = features[:self.size]
            B = self.size
        
        if self.ptr + B <= self.size:
            self.bank[self.ptr:self.ptr + B] = features.detach()
            self.ptr += B
        else:
            R = self.size - self.ptr
            self.bank[self.ptr:self.size] = features[:R].detach()
            self.bank[0:B - R] = features[R:].detach()
            self.ptr = B - R
        
    def sample_negatives(self, num_negatives):
        indices = torch.randint(0, self.size, (num_negatives,), device=self.device)
        negatives = self.bank[indices]
        return negatives

def stable_contrastive_loss(anchor, positive, negatives, temperature=0.1):
    batch_size = anchor.size(0)
    
    pos_similarity = torch.sum(anchor * positive, dim=1).unsqueeze(1)
    neg_similarity = torch.matmul(anchor, negatives.T)
    
    all_similarities = torch.cat([pos_similarity, neg_similarity], dim=1)
    all_similarities = all_similarities / temperature
    
    labels = torch.zeros(batch_size, dtype=torch.long).to(anchor.device)
    
    loss = F.cross_entropy(all_similarities, labels)
    
    with torch.no_grad():
        predictions = torch.argmax(all_similarities, dim=1)
        accuracy = (predictions == labels).float().mean()
    
    return loss, accuracy
