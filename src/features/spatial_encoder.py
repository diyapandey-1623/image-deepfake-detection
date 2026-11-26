"""
spatial_encoder.py
------------------
Spatial RGB Face CNN Encoder.
"""

import torch
import torch.nn as nn
from src.features.blocks import make_conv_backbone


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        use_pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pretrained_backbone = use_pretrained_backbone

        if use_pretrained_backbone:
            # Generic ImageNet backbone only (never deepfake-specific weights)
            import torchvision.models as models

            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except Exception:
                resnet = models.resnet18(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.proj = nn.Linear(resnet.fc.in_features, embed_dim)
        else:
            self.backbone = make_conv_backbone(in_channels, embed_dim)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained_backbone:
            feat = self.backbone(x)
            return self.proj(feat.flatten(1))
        feat_map = self.backbone(x)
        pooled = self.global_pool(feat_map).flatten(1)
        return self.proj(pooled)
