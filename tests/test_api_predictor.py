"""
test_api_predictor.py
---------------------
Tests for DeepfakePredictor and FastAPI endpoint handlers.
"""

import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import backend.database as db_mod
import backend.main as main_mod
from backend.main import app
from backend.model.samples import create_synthetic_portrait
from src.inference.predictor import DeepfakePredictor


class TestApiAndPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)
        cls._orig_db_path = db_mod.DB_PATH
        cls._orig_samples_dir = main_mod.SAMPLES_DIR
        db_mod.DB_PATH = tmp_path / "test_scans.db"
        db_mod._db_initialized = False
        main_mod.SAMPLES_DIR = tmp_path / "samples"

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

        real_rgb = create_synthetic_portrait(is_fake=False)
        buf = io.BytesIO()
        Image.fromarray(real_rgb).save(buf, format="JPEG")
        cls.real_jpeg_bytes = buf.getvalue()

        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        buf_blank = io.BytesIO()
        Image.fromarray(blank).save(buf_blank, format="JPEG")
        cls.blank_jpeg_bytes = buf_blank.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)
        db_mod.DB_PATH = cls._orig_db_path
        db_mod._db_initialized = False
        main_mod.SAMPLES_DIR = cls._orig_samples_dir
        cls._tmpdir.cleanup()

    def test_predictor_direct_inference(self):
        pred = DeepfakePredictor()
        res = pred.predict(self.real_jpeg_bytes)

        self.assertIn(res["verdict"], ["REAL", "DEEPFAKE", "UNCERTAIN"])
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertTrue(0.0 <= res["probability_fake"] <= 1.0)
        self.assertIn("fusion", res["scores"])
        self.assertIn("spatial", res["scores"])
        self.assertIn("frequency", res["scores"])
        self.assertIn("forensic", res["scores"])
        self.assertTrue(res["face_detected"])
        self.assertIn("bbox", res["metadata"])

    def test_sample_endpoint(self):
        for kind in ("real", "fake"):
            res = self.client.get(f"/api/sample/{kind}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers["content-type"], "image/jpeg")

        bad = self.client.get("/api/sample/invalid")
        self.assertEqual(bad.status_code, 400)

    def test_analyze_endpoint_valid_face(self):
        res = self.client.post(
            "/api/analyze",
            files={"file": ("portrait.jpg", self.real_jpeg_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn(body["verdict"], ["fake", "real"])
        self.assertTrue(0.0 <= body["fake_probability"] <= 1.0)
        self.assertTrue(body["checks"]["face_detected"])
        self.assertTrue(body["report_id"].startswith("rep_"))

    def test_analyze_endpoint_no_face(self):
        res = self.client.post(
            "/api/analyze",
            files={"file": ("blank.jpg", self.blank_jpeg_bytes, "image/jpeg")},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn(body["verdict"], ["fake", "real"])
        self.assertFalse(body["checks"]["face_detected"])

    def test_analyze_endpoint_invalid_mime(self):
        res = self.client.post(
            "/api/analyze",
            files={"file": ("notes.txt", b"sample text", "text/plain")},
        )
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
