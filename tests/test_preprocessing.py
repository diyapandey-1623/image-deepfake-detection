"""
test_preprocessing.py
---------------------
Unit tests for extract_media, face_crop, and frequency_transform.
"""

import os
import unittest
import numpy as np
from pathlib import Path

from src.preprocessing.extract_media import load_image, validate_image_file, MediaValidationError
from src.preprocessing.face_crop import FaceDetector, NoFaceDetectedError
from src.preprocessing.frequency_transform import (
    compute_fft_magnitude_spectrum,
    compute_ela_map,
    compute_laplacian_sharpness_map
)
from tests.generate_test_samples import create_synthetic_portrait


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.detector = FaceDetector(target_size=112, margin_factor=0.20)
        self.sample_real_rgb = create_synthetic_portrait(is_fake=False)
        self.sample_fake_rgb = create_synthetic_portrait(is_fake=True)

    def test_face_detection_and_crop(self):
        """Verify face is detected, cropped with margins, and standardized to 112x112."""
        face_crop, meta = self.detector.extract_face(self.sample_real_rgb)
        
        self.assertEqual(face_crop.shape, (112, 112, 3))
        self.assertEqual(face_crop.dtype, np.uint8)
        self.assertTrue(meta["face_detected"])
        self.assertGreater(meta["total_faces"], 0)
        self.assertEqual(len(meta["bbox"]), 4)

    def test_no_face_detected_error(self):
        """Verify NoFaceDetectedError is raised when an image contains no faces."""
        blank_img = np.zeros((200, 200, 3), dtype=np.uint8)
        with self.assertRaises(NoFaceDetectedError):
            self.detector.extract_face(blank_img, allow_fallback_center_crop=False)

    def test_mouth_extraction(self):
        """Verify mouth / lower-face region is extracted at specified resolution."""
        face_crop, _ = self.detector.extract_face(self.sample_real_rgb)
        mouth = self.detector.extract_mouth_region(face_crop, mouth_size=64)
        self.assertEqual(mouth.shape, (64, 64, 3))

    def test_fft_magnitude_spectrum(self):
        """Verify 2D FFT computation produces normalized 112x112x1 spectrum."""
        spectrum = compute_fft_magnitude_spectrum(self.sample_fake_rgb, target_size=112)
        self.assertEqual(spectrum.shape, (112, 112, 1))
        self.assertEqual(spectrum.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(spectrum)), 0.0)
        self.assertLessEqual(float(np.max(spectrum)), 1.0)

    def test_ela_computation(self):
        """Verify Error Level Analysis map computation."""
        ela = compute_ela_map(self.sample_real_rgb, quality=90, scale=15.0)
        self.assertEqual(ela.shape, self.sample_real_rgb.shape)
        self.assertEqual(ela.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(ela)), 0.0)
        self.assertLessEqual(float(np.max(ela)), 1.0)

    def test_laplacian_sharpness(self):
        """Verify Laplacian edge gradient map computation."""
        sharpness = compute_laplacian_sharpness_map(self.sample_real_rgb)
        self.assertEqual(sharpness.shape, (self.sample_real_rgb.shape[0], self.sample_real_rgb.shape[1]))
        self.assertGreaterEqual(float(np.min(sharpness)), 0.0)
        self.assertLessEqual(float(np.max(sharpness)), 1.0)


if __name__ == "__main__":
    unittest.main()
