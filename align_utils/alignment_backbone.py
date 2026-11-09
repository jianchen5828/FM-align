"""Visual encoder matching the current align.py; guarded by parity tests."""
from typing import List, Tuple
import torch
from torch import nn

TARGET_SIZE = 128

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


def load_alignment_backbone(model, checkpoint_path, map_location="cpu"):
    """Load the visual encoder from align.py's full state_dict, strictly.

    Mixed-model checkpoints using img_backbone are intentionally unsupported.
    """
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Expected the full state_dict saved by align.py")
    prefix = "image_backbone."
    visual = {key[len(prefix):]: value for key, value in state.items()
              if isinstance(key, str) and key.startswith(prefix)}
    if not visual:
        raise ValueError("No image_backbone.* parameters found. Use the checkpoint "
                         "saved by the current align.py, not an img_backbone/mixed checkpoint.")
    return model.load_state_dict(visual, strict=True)
