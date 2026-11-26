"""
tests/test_backend_integration.py
---------------------------------
Integration tests for FastAPI endpoints, CustomPyTorchModel, Grad-CAM, and forensic checks.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN

from backend.main import app
from backend.model.checks import check_ela, check_metadata, detect_and_crop_faces
from backend.model.inference import CustomPyTorchModel
from src.class_config import get_verdict


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    import backend.database as db_mod
    import backend.main as main_mod

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "test_scans.db")
    monkeypatch.setattr(db_mod, "_db_initialized", False)
    monkeypatch.setattr(main_mod, "SAMPLES_DIR", tmp_path / "samples")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_real_image():
    arr = np.random.randint(100, 220, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_real_image):
    buf = io.BytesIO()
    sample_real_image.save(buf, format="JPEG")
    return buf.getvalue()


class TestVerdictAndModel:
    def test_verdict_binary_threshold(self):
        assert get_verdict(0.85) == "fake"
        assert get_verdict(0.50) == "fake"
        assert get_verdict(0.20) == "real"

    def test_custom_pytorch_predict_and_gradcam(self, sample_real_image):
        model = CustomPyTorchModel(architecture="efficientnet_b0", device="cpu")
        res = model.predict(sample_real_image)

        assert 0.0 <= res["fake_probability"] <= 1.0
        assert res["verdict"] in ("fake", "real")
        assert "Real" in res["raw_output"]
        assert "Fake" in res["raw_output"]

        heatmap, regions, rationale = model.generate_heatmap_and_regions(
            sample_real_image,
            verdict=res["verdict"],
            fake_prob=res["fake_probability"],
        )
        assert heatmap is not None and heatmap.startswith("data:image/png;base64,")
        assert len(regions) == 5
        assert rationale


class TestForensicChecks:
    def test_check_metadata_flags_suspicious_software(self):
        from PIL.PngImagePlugin import PngInfo

        img = Image.new("RGB", (64, 64), color=(120, 120, 120))
        meta = PngInfo()
        meta.add_text("Software", "Adobe Photoshop 2024")
        buf = io.BytesIO()
        img.save(buf, format="PNG", pnginfo=meta)
        buf.seek(0)
        reloaded = Image.open(buf)

        has_flags, flags = check_metadata(reloaded)
        assert has_flags is True
        assert any("photoshop" in f.lower() for f in flags)

    def test_check_ela_detects_threshold(self, sample_real_image):
        ela_flag, mean_err = check_ela(sample_real_image, quality=90)
        assert isinstance(ela_flag, bool)
        assert mean_err >= 0.0

    def test_detect_and_crop_faces_blank_image(self):
        detector = MTCNN(keep_all=True, device="cpu")
        blank = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
        faces, detected = detect_and_crop_faces(blank, detector)
        assert detected is False
        assert faces == []


class TestApiPipeline:
    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert body["model_backend"] in ("huggingface", "custom_pytorch")

    def test_analyze_history_and_pdf_flow(self, client, sample_image_bytes):
        res = client.post(
            "/api/analyze",
            files={"file": ("scan_for_pdf.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["report_id"].startswith("rep_")
        assert body["verdict"] in ("fake", "real")
        assert 0.0 <= body["fake_probability"] <= 1.0
        assert body["processing_time_ms"] > 0

        hist_res = client.get("/api/history?limit=10")
        assert hist_res.status_code == 200
        matching = [item for item in hist_res.json() if item["report_id"] == body["report_id"]]
        assert len(matching) == 1
        assert matching[0]["filename"] == "scan_for_pdf.jpg"

        pdf_res = client.get(f"/api/report/{body['report_id']}/pdf")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"

    def test_analyze_rejects_invalid_and_empty_uploads(self, client):
        bad_ext = client.post(
            "/api/analyze",
            files={"file": ("document.pdf", b"dummy", "application/pdf")},
        )
        assert bad_ext.status_code == 400
        assert "Unsupported format" in bad_ext.json()["detail"]

        empty = client.post(
            "/api/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert empty.status_code == 400
        assert "empty" in empty.json()["detail"].lower()

        corrupt = client.post(
            "/api/analyze",
            files={"file": ("corrupt.jpg", b"\xFF\xD8\xFFcorrupt", "image/jpeg")},
        )
        assert corrupt.status_code == 400
        assert "Corrupted or invalid image file" in corrupt.json()["detail"]

        oversized = client.post(
            "/api/analyze",
            files={"file": ("oversized.png", b"0" * (11 * 1024 * 1024), "image/png")},
        )
        assert oversized.status_code == 413

    def test_analyze_no_face_fallback_and_multi_face(self, client):
        solid = np.zeros((200, 200, 3), dtype=np.uint8)
        solid[:, :] = [40, 100, 200]
        buf = io.BytesIO()
        Image.fromarray(solid).save(buf, format="JPEG")
        res = client.post(
            "/api/analyze",
            files={"file": ("landscape.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert res.status_code == 200
        assert res.json()["checks"]["face_detected"] is False
        assert res.json()["faces"] == []

        img = Image.new("RGB", (320, 160), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)
        for offset in (0, 160):
            draw.ellipse([20 + offset, 20, 120 + offset, 140], fill=(235, 195, 170))
            draw.ellipse([45 + offset, 55, 60 + offset, 68], fill=(30, 30, 30))
            draw.ellipse([80 + offset, 55, 95 + offset, 68], fill=(30, 30, 30))
        buf2 = io.BytesIO()
        img.save(buf2, format="JPEG")
        res2 = client.post(
            "/api/analyze",
            files={"file": ("group.jpg", buf2.getvalue(), "image/jpeg")},
        )
        assert res2.status_code == 200
        assert res2.json()["verdict"] in ("fake", "real")
