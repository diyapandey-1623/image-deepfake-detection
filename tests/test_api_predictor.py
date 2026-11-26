"""
test_api_predictor.py
---------------------
Tests for DeepfakePredictor and FastAPI endpoint handlers.
"""

import io
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException

from backend.main import health_check, analyze_image
from src.inference.predictor import DeepfakePredictor
from tests.generate_test_samples import create_synthetic_portrait


class TestApiAndPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a test image in memory
        real_rgb = create_synthetic_portrait(is_fake=False)
        pil_img = Image.fromarray(real_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)
        cls.real_jpeg_bytes = buf.getvalue()

        # Non-face image
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        buf_blank = io.BytesIO()
        Image.fromarray(blank).save(buf_blank, format="JPEG")
        buf_blank.seek(0)
        cls.blank_jpeg_bytes = buf_blank.getvalue()

    def test_predictor_direct_inference(self):
        """Test DeepfakePredictor direct python method."""
        pred = DeepfakePredictor()
        res = pred.predict(self.real_jpeg_bytes)
        
        self.assertIn(res["verdict"], ["REAL", "DEEPFAKE", "UNCERTAIN"])
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertTrue(0.0 <= res["probability_fake"] <= 1.0)
        self.assertIn("scores", res)
        self.assertIn("fusion", res["scores"])
        self.assertIn("spatial", res["scores"])
        self.assertIn("frequency", res["scores"])
        self.assertIn("forensic", res["scores"])
        self.assertTrue(res["face_detected"])
        self.assertIn("bbox", res["metadata"])

    def test_health_endpoint_direct(self):
        """Test GET /api/health endpoint handler."""
        res = health_check()
        self.assertEqual(res.status, "healthy")
        self.assertTrue(res.model_loaded)

    def test_sample_endpoint_direct(self):
        """Test GET /api/sample/{kind} endpoint handler."""
        from backend.main import get_sample_image
        res_real = get_sample_image("real")
        self.assertTrue(Path(res_real.path).is_file())
        res_fake = get_sample_image("fake")
        self.assertTrue(Path(res_fake.path).is_file())

    def test_analyze_endpoint_valid_face(self):
        """Test POST /api/analyze endpoint with valid portrait."""
        upload = UploadFile(
            file=io.BytesIO(self.real_jpeg_bytes),
            filename="portrait.jpg",
            headers={"content-type": "image/jpeg"}
        )
        res = analyze_image(upload)
        self.assertIn(res.verdict, ["fake", "real", "uncertain"])
        self.assertTrue(0.0 <= res.fake_probability <= 1.0)
        self.assertTrue(res.checks.face_detected)
        self.assertTrue(res.report_id.startswith("rep_"))

    def test_analyze_endpoint_no_face(self):
        """Test POST /api/analyze handles image with no face by fallback to full-image."""
        upload = UploadFile(
            file=io.BytesIO(self.blank_jpeg_bytes),
            filename="blank.jpg",
            headers={"content-type": "image/jpeg"}
        )
        res = analyze_image(upload)
        self.assertIn(res.verdict, ["fake", "real", "uncertain"])
        self.assertFalse(res.checks.face_detected)

    def test_analyze_endpoint_invalid_mime(self):
        """Test POST /api/analyze returns 400 for non-image MIME."""
        upload = UploadFile(
            file=io.BytesIO(b"sample text"),
            filename="notes.txt",
            headers={"content-type": "text/plain"}
        )
        with self.assertRaises(HTTPException) as ctx:
            analyze_image(upload)
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
