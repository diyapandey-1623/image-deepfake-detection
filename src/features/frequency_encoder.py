"""
frequency_encoder.py
--------------------
Frequency Domain 2D FFT Spectral CNN Encoder.

Design decisions and viva defense points:
1. Input representation:
   Accepts 1-channel 2D log-magnitude Fourier power spectrum [B, 1, 112, 112].
   The DC zero-frequency component is centered, with high frequencies radiating outward.
2. Architecture:
   4-stage convolutional extractor with BatchNorm and LeakyReLU designed to learn
   spectral fingerprints:
   - Identifies periodic high-frequency grid artifacts (characteristic of StyleGAN upsampling)
   - Detects abnormal azimuthal spectral power decay (typical of Diffusion Models)
   - Outputs a normalized 128-dimensional frequency embedding [B, 128].
"""

import torch
import torch.nn as nn
from src.models.baselines import ConvBlock


class FrequencyEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 4-stage convolutional encoder trained strictly from scratch
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
            x: Input 2D FFT spectrum tensor [B, 1, 112, 112], values in [0, 1].
        Returns:
            embedding: Tensor [B, embed_dim]
        """
        feat_map = self.backbone(x)
        pooled = self.global_pool(feat_map).flatten(1)
        embed = self.proj(pooled)
        return embed
