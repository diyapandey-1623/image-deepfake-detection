"""
frequency_encoder.py
--------------------
Frequency Domain 2D FFT Spectral CNN Encoder.
"""

import torch
import torch.nn as nn
from src.features.blocks import make_conv_backbone


class FrequencyEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = make_conv_backbone(in_channels, embed_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        pooled = self.global_pool(feat_map).flatten(1)
        return self.proj(pooled)
