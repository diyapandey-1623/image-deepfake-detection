"""
fusion_model.py
---------------
Multimodal Cross-Attention Fusion Architecture for Image Deepfake Detection.

Design decisions and viva defense points:
1. Cross-Domain Inconsistency Detection:
   A critical vulnerability of modern generative deepfakes is the inconsistency between
   spatial appearance and frequency/spectral domains:
   - High visual fidelity in spatial domain vs periodic grid anomalies in Fourier domain.
   - Smooth facial skin vs sharp compression discrepancies in Error Level Analysis.
   By employing Bidirectional Cross-Attention (Spatial attends to Frequency, and Frequency
   attends to Spatial), the model directly queries and highlights discrepancies between domains.
2. Auxiliary Multi-Task Heads:
   In addition to the primary fused logit, the model incorporates auxiliary unimodal heads:
   - Aux Spatial Head (RGB face only)
   - Aux Frequency Head (2D FFT spectrum only)
   This multi-task supervision forces the individual encoders to learn domain-discriminative
   representations while also directly powering the per-modality breakdown in the user interface.
3. Compact Capacity (<3M parameters):
   Total parameter count is ~1.1M, strictly preventing over-parameterization and empirical overfitting.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.spatial_encoder import SpatialEncoder
from src.features.frequency_encoder import FrequencyEncoder


class MultiModalCrossAttentionFusion(nn.Module):
    def __init__(
        self,
        spatial_dim: int = 128,
        frequency_dim: int = 128,
        handcrafted_dim: int = 32,
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.35,
        use_pretrained_backbone: bool = False
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.frequency_dim = frequency_dim
        self.fusion_dim = fusion_dim
        self.dropout_rate = dropout

        # 1. Domain Encoders
        self.spatial_encoder = SpatialEncoder(
            in_channels=3,
            embed_dim=spatial_dim,
            use_pretrained_backbone=use_pretrained_backbone
        )
        self.frequency_encoder = FrequencyEncoder(
            in_channels=1,
            embed_dim=frequency_dim
        )

        # 2. Handcrafted Feature Projection
        self.handcrafted_proj = nn.Sequential(
            nn.Linear(handcrafted_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(64, 64),
            nn.LayerNorm(64)
        )

        # 3. Domain Linear Projections to common fusion dimension
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.proj_freq = nn.Sequential(
            nn.Linear(frequency_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # 4. Bidirectional Cross-Attention
        # Query = Spatial, Key/Value = Frequency
        self.cross_attn_s2f = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Query = Frequency, Key/Value = Spatial
        self.cross_attn_f2s = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm_s = nn.LayerNorm(fusion_dim)
        self.norm_f = nn.LayerNorm(fusion_dim)

        # 5. Multimodal Fusion MLP Classifier
        # Concatenated features:
        # Spatial-fused (128) + Frequency-fused (128) + Handcrafted (64) + Discrepancy (1) = 321
        fused_in_dim = fusion_dim * 2 + 64 + 1
        self.classifier = nn.Sequential(
            nn.Linear(fused_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)  # Main Fusion Binary Logit
        )

        # 6. Auxiliary Unimodal Classifiers (for UI breakdown and multi-task gradient flow)
        self.aux_spatial_head = nn.Sequential(
            nn.Linear(spatial_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)
        )
        self.aux_freq_head = nn.Sequential(
            nn.Linear(frequency_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        face_crop: torch.Tensor,
        fft_spectrum: torch.Tensor,
        handcrafted: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            face_crop: [B, 3, 112, 112]
            fft_spectrum: [B, 1, 112, 112]
            handcrafted: [B, 32]
        Returns:
            dict containing logits, calibrated probabilities, and inconsistency scores.
        """
        batch_size = face_crop.size(0)

        # 1. Extract unimodal representations
        z_spatial = self.spatial_encoder(face_crop)        # [B, 128]
        z_freq = self.frequency_encoder(fft_spectrum)      # [B, 128]
        z_handcrafted = self.handcrafted_proj(handcrafted) # [B, 64]

        # 2. Auxiliary unimodal predictions
        spatial_logit = self.aux_spatial_head(z_spatial).squeeze(-1)   # [B]
        freq_logit = self.aux_freq_head(z_freq).squeeze(-1)           # [B]

        # 3. Project to shared fusion dimension and reshape for MultiheadAttention [B, 1, D]
        h_s = self.proj_spatial(z_spatial).unsqueeze(1)    # [B, 1, 128]
        h_f = self.proj_freq(z_freq).unsqueeze(1)          # [B, 1, 128]

        # 4. Bidirectional Cross-Attention
        # Spatial queries Frequency: "What spectral artifacts correspond to this visual region?"
        attn_s, _ = self.cross_attn_s2f(query=h_s, key=h_f, value=h_f)
        h_s_fused = self.norm_s(h_s + attn_s).squeeze(1)   # [B, 128]

        # Frequency queries Spatial: "What visual regions explain this spectral pattern?"
        attn_f, _ = self.cross_attn_f2s(query=h_f, key=h_s, value=h_s)
        h_f_fused = self.norm_f(h_f + attn_f).squeeze(1)   # [B, 128]

        # 5. Domain Inconsistency Discrepancy Metric (Cosine distance: 1 - cos_sim)
        cos_sim = F.cosine_similarity(h_s.squeeze(1), h_f.squeeze(1), dim=1, eps=1e-8)
        inconsistency = (1.0 - cos_sim).unsqueeze(1)       # [B, 1]

        # 6. Multimodal Feature Aggregation
        fused_features = torch.cat([h_s_fused, h_f_fused, z_handcrafted, inconsistency], dim=1) # [B, 321]

        # 7. Main Fused Classification Logit
        fusion_logit = self.classifier(fused_features).squeeze(-1) # [B]

        # Probabilities via sigmoid
        fusion_prob = torch.sigmoid(fusion_logit)
        spatial_prob = torch.sigmoid(spatial_logit)
        freq_prob = torch.sigmoid(freq_logit)

        return {
            "fusion_logit": fusion_logit,
            "spatial_logit": spatial_logit,
            "frequency_logit": freq_logit,
            "fusion_prob": fusion_prob,
            "spatial_prob": spatial_prob,
            "frequency_prob": freq_prob,
            "inconsistency_score": inconsistency.squeeze(-1)
        }
