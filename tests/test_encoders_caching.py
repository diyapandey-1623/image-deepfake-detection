"""
test_encoders_caching.py
------------------------
Unit tests for SpatialEncoder, FrequencyEncoder, and loading cached features.
"""

import unittest
from pathlib import Path
import torch

from src.features.spatial_encoder import SpatialEncoder
from src.features.frequency_encoder import FrequencyEncoder


class TestEncodersAndCaching(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_spatial_encoder_scratch(self):
        """Test SpatialEncoder trained from scratch."""
        model = SpatialEncoder(in_channels=3, embed_dim=128, use_pretrained_backbone=False)
        x = torch.randn(2, 3, 112, 112)
        out = model(x)
        self.assertEqual(out.shape, (2, 128))
        self.assertFalse(torch.isnan(out).any())

    def test_spatial_encoder_pretrained_flag(self):
        """Test SpatialEncoder with torchvision backbone when flag is set."""
        model = SpatialEncoder(in_channels=3, embed_dim=128, use_pretrained_backbone=True)
        x = torch.randn(2, 3, 112, 112)
        out = model(x)
        self.assertEqual(out.shape, (2, 128))

    def test_frequency_encoder(self):
        """Test FrequencyEncoder forward pass on 2D FFT spectrum."""
        model = FrequencyEncoder(in_channels=1, embed_dim=128)
        x = torch.rand(2, 1, 112, 112)
        out = model(x)
        self.assertEqual(out.shape, (2, 128))
        self.assertFalse(torch.isnan(out).any())

    def test_cached_file_structure(self):
        """Verify cached .pt file structure and tensor dimensions when cache exists."""
        cache_dir = Path("data/cache")
        pt_files = list(cache_dir.glob("sample_*.pt"))
        if not pt_files:
            self.skipTest("No cached .pt files in data/cache.")

        data = torch.load(pt_files[0], weights_only=False)
        self.assertIn("face_crop", data)
        self.assertIn("fft_spectrum", data)
        self.assertIn("handcrafted", data)
        self.assertIn("label", data)
        self.assertIn("subject_id", data)

        self.assertEqual(data["face_crop"].shape, (3, 112, 112))
        self.assertEqual(data["fft_spectrum"].shape, (1, 112, 112))
        self.assertEqual(data["handcrafted"].shape, (32,))
        self.assertIn(data["label"], [0, 1])


if __name__ == "__main__":
    unittest.main()
