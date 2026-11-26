"""
blocks.py
---------
Shared convolutional building blocks for spatial and frequency encoders.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU(0.1) -> optional MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


def make_conv_backbone(in_channels: int, embed_dim: int = 128) -> nn.Sequential:
    """5-stage CNN backbone mapping 112x112 input to 7x7 feature maps."""
    return nn.Sequential(
        ConvBlock(in_channels, 32, pool=True),
        ConvBlock(32, 64, pool=True),
        ConvBlock(64, 128, pool=True),
        ConvBlock(128, 256, pool=True),
        ConvBlock(256, embed_dim, pool=False),
    )
