"""
test_handcrafted_baseline.py
----------------------------
Tests for handcrafted forensic feature extraction, classical ML baselines (RF/SVM),
and single-modality PyTorch baselines.
"""

import unittest
import numpy as np
import torch

from src.features.handcrafted import (
    extract_all_handcrafted_features,
    extract_spectral_features,
    extract_texture_edge_features,
    extract_ela_features,
    extract_color_chrominance_features,
    HANDCRAFTED_FEATURE_DIM
)
from src.models.baselines import (
    ClassicalForensicBaseline,
    SpatialOnlyCNN,
    FrequencyOnlyCNN,
    compute_classification_metrics
)
from tests.generate_test_samples import create_synthetic_portrait


class TestHandcraftedAndBaselines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.sample_real = create_synthetic_portrait(is_fake=False)
        self.sample_fake = create_synthetic_portrait(is_fake=True)

    def test_feature_dimensions(self):
        """Verify each forensic feature extractor returns exact expected dimensions."""
        spec = extract_spectral_features(self.sample_real)
        self.assertEqual(len(spec), 10)

        texture = extract_texture_edge_features(self.sample_real)
        self.assertEqual(len(texture), 8)

        ela = extract_ela_features(self.sample_real)
        self.assertEqual(len(ela), 6)

        color = extract_color_chrominance_features(self.sample_real)
        self.assertEqual(len(color), 8)

        total = extract_all_handcrafted_features(self.sample_real)
        self.assertEqual(len(total), HANDCRAFTED_FEATURE_DIM)
        self.assertEqual(total.dtype, np.float32)
        self.assertFalse(np.any(np.isnan(total)))
        self.assertFalse(np.any(np.isinf(total)))

    def test_classical_forensic_baseline(self):
        """Verify Random Forest and SVM training, prediction, and metric computation."""
        # Create a mini synthetic feature matrix (16 samples, 32 features)
        X_train = np.random.randn(16, HANDCRAFTED_FEATURE_DIM).astype(np.float32)
        y_train = np.array([0, 1] * 8)
        X_test = np.random.randn(8, HANDCRAFTED_FEATURE_DIM).astype(np.float32)
        y_test = np.array([0, 1] * 4)

        for model_type in ["rf", "svm"]:
            clf = ClassicalForensicBaseline(model_type=model_type, random_state=42)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)
            
            self.assertEqual(len(probs), 8)
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

            metrics = clf.evaluate(X_test, y_test)
            self.assertIn("accuracy", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1", metrics)
            self.assertIn("roc_auc", metrics)
            self.assertIn("confusion_matrix", metrics)

    def test_spatial_only_cnn(self):
        """Verify SpatialOnlyCNN forward pass and shape."""
        model = SpatialOnlyCNN(in_channels=3, embed_dim=128)
        x = torch.randn(4, 3, 112, 112)
        out = model(x)
        self.assertEqual(out.shape, (4,))

        embed = model.extract_embedding(x)
        self.assertEqual(embed.shape, (4, 128))

    def test_frequency_only_cnn(self):
        """Verify FrequencyOnlyCNN forward pass and shape."""
        model = FrequencyOnlyCNN(in_channels=1, embed_dim=128)
        x = torch.randn(4, 1, 112, 112)
        out = model(x)
        self.assertEqual(out.shape, (4,))

        embed = model.extract_embedding(x)
        self.assertEqual(embed.shape, (4, 128))


if __name__ == "__main__":
    unittest.main()
