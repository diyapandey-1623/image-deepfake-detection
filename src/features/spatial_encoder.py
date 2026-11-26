"""
spatial_encoder.py
------------------
Spatial RGB Face CNN Encoder.

Design decisions and viva defense points:
1. Architectural Design:
   A 5-stage convolutional feature extractor trained completely from scratch:
   - ConvBlock 1: 3 -> 32 channels (receptive field 3x3)
   - ConvBlock 2: 32 -> 64 channels (receptive field 7x7)
   - ConvBlock 3: 64 -> 128 channels (receptive field 15x15)
   - ConvBlock 4: 128 -> 256 channels (receptive field 31x31)
   - ConvBlock 5: 256 -> 128 channels (receptive field 63x63)
   - Adaptive Global Average Pooling -> Linear projection [128].
   This covers the 112x112 input resolution while limiting total parameters to ~450k,
   avoiding the severe overfitting inherent to oversized models trained from scratch.
2. Ground Rule Compliance:
   - `use_pretrained_backbone: false` by default.
   - If enabled via config, only generic torchvision ImageNet backbones (e.g., resnet18)
     are loaded; NEVER any deepfake-specific weights.
"""

from typing import Optional
import torch
import torch.nn as nn
from src.models.baselines import ConvBlock


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        use_pretrained_backbone: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pretrained_backbone = use_pretrained_backbone

        if use_pretrained_backbone:
            # Generic torchvision ImageNet backbone only (never deepfake-specific weights)
            import torchvision.models as models
            try:
                weights = models.ResNet18_Weights.DEFAULT
                resnet = models.resnet18(weights=weights)
            except Exception:
                # If network access is unavailable, initialize architecture structure without external weights
                resnet = models.resnet18(weights=None)
            modules = list(resnet.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.proj = nn.Linear(resnet.fc.in_features, embed_dim)
        else:
            # Custom 5-stage CNN trained from scratch
            self.backbone = nn.Sequential(
                ConvBlock(in_channels, 32, pool=True),    # 112 -> 56
                ConvBlock(32, 64, pool=True),             # 56 -> 28
                ConvBlock(64, 128, pool=True),            # 28 -> 14
                ConvBlock(128, 256, pool=True),           # 14 -> 7
                ConvBlock(256, embed_dim, pool=False)     # 7 -> 7
            )
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input RGB tensor [B, 3, 112, 112], normalized to [0, 1] or standard ImageNet stats.
        Returns:
            embedding: Tensor [B, embed_dim]
        """
        if self.use_pretrained_backbone:
            feat = self.backbone(x)  # [B, 512, 1, 1]
            embed = self.proj(feat.flatten(1))
        else:
            feat_map = self.backbone(x)
            pooled = self.global_pool(feat_map).flatten(1)
            embed = self.proj(pooled)
            
        return embed
