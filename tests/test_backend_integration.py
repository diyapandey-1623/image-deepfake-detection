"""
tests/test_backend_integration.py
---------------------------------
Pytest test suite for model-agnostic FastAPI deepfake detection backend.
Tests both HuggingFace and Custom PyTorch backends, MTCNN cropping, Grad-CAM,
forensic checks, and FastAPI endpoints.
"""

import io
import pytest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from fastapi import UploadFile, HTTPException

from backend.config import FAKE_THRESHOLD, REAL_THRESHOLD, MAX_UPLOAD_MB
from backend.model.inference import (
    CustomPyTorchModel,
    HuggingFaceModel,
    get_verdict,
    BaseDeepfakeModel
)
from backend.model.checks import check_metadata, check_ela, detect_and_crop_faces
from backend.main import app, health_check, analyze_image
from facenet_pytorch import MTCNN


@pytest.fixture
def sample_real_image():
    """Generates a simple RGB portrait image for testing."""
    arr = np.random.randint(100, 220, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_real_image):
    buf = io.BytesIO()
    sample_real_image.save(buf, format="JPEG")
    return buf.getvalue()


class TestConfigAndVerdict:
    def test_verdict_thresholds(self):
        assert get_verdict(0.85, fake_thresh=0.6, real_thresh=0.4) == "fake"
        assert get_verdict(0.20, fake_thresh=0.6, real_thresh=0.4) == "real"
        assert get_verdict(0.50, fake_thresh=0.6, real_thresh=0.4) == "uncertain"


class TestCustomPyTorchModel:
    def test_custom_pytorch_predict_and_gradcam(self, sample_real_image):
        model = CustomPyTorchModel(architecture="efficientnet_b0", device="cpu")
        res = model.predict(sample_real_image)

        assert "fake_probability" in res
        assert 0.0 <= res["fake_probability"] <= 1.0
        assert res["verdict"] in ["fake", "real", "uncertain"]
        assert "raw_output" in res
        assert "fake" in res["raw_output"]
        assert "real" in res["raw_output"]
        assert "model_name" in res

        # Grad-CAM heatmap should be generated for CNN
        heatmap = model.generate_heatmap(sample_real_image)
        assert heatmap is not None
        assert heatmap.startswith("data:image/png;base64,")


class TestDynamicLabelMatching:
    def test_label_matching_keywords(self):
        # Test class matching logic independently
        class MockConfig:
            model_type = "cnn"
            id2label = {0: "authentic_human", 1: "ai_generated_deepfake"}

        # Simulate label resolver
        hf_dummy = HuggingFaceModel.__new__(HuggingFaceModel)
        hf_dummy.FAKE_KEYWORDS = {"fake", "deepfake", "ai", "artificial", "synthetic", "manipulated", "generated", "forgery"}
        hf_dummy.REAL_KEYWORDS = {"real", "authentic", "genuine", "pristine", "original", "natural", "human"}
        
        idx = hf_dummy._resolve_fake_label_id(MockConfig.id2label)
        assert idx == 1

    def test_label_matching_failure_raises_clear_error(self):
        hf_dummy = HuggingFaceModel.__new__(HuggingFaceModel)
        hf_dummy.FAKE_KEYWORDS = {"fake", "deepfake", "ai", "artificial"}
        hf_dummy.REAL_KEYWORDS = {"real", "authentic"}
        
        bad_id2label = {0: "cat", 1: "dog", 2: "car"}
        with pytest.raises(ValueError) as exc:
            hf_dummy._resolve_fake_label_id(bad_id2label)
        assert "Could not identify a 'fake' or 'deepfake' class" in str(exc.value)
        assert "cat" in str(exc.value)


class TestForensicChecks:
    def test_check_metadata(self, sample_real_image):
        has_flags, flags = check_metadata(sample_real_image)
        assert isinstance(has_flags, bool)
        assert isinstance(flags, list)

    def test_check_ela(self, sample_real_image):
        ela_flag, mean_err = check_ela(sample_real_image, quality=90)
        assert isinstance(ela_flag, bool)
        assert isinstance(mean_err, float)

    def test_detect_and_crop_faces_fallback(self):
        detector = MTCNN(keep_all=True, device="cpu")
        # Empty uniform gray image will have no face
        blank = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
        faces, detected = detect_and_crop_faces(blank, detector)
        assert detected is False
        assert len(faces) == 0


class TestApiPipeline:
    def test_health_endpoint(self):
        # Direct handler test
        res = health_check()
        assert res.status == "healthy"
        assert res.model_backend in ["huggingface", "custom_pytorch"]

    def test_analyze_pipeline(self, sample_image_bytes):
        if getattr(app.state, "model", None) is None:
            app.state.model = CustomPyTorchModel(architecture="efficientnet_b0", device="cpu")
        if getattr(app.state, "mtcnn", None) is None:
            app.state.mtcnn = MTCNN(keep_all=True, device="cpu")

        upload = UploadFile(
            file=io.BytesIO(sample_image_bytes),
            filename="test_portrait.jpg",
            headers={"content-type": "image/jpeg"}
        )

        res = analyze_image(upload)
        assert res.report_id.startswith("rep_")
        assert res.verdict in ["fake", "real", "uncertain"]
        assert 0.0 <= res.fake_probability <= 1.0
        assert hasattr(res.checks, "face_detected")
        assert hasattr(res.checks, "metadata_flags")
        assert hasattr(res.checks, "ela_flag")
        assert res.created_at is not None

    def test_analyze_invalid_extension(self):
        upload = UploadFile(
            file=io.BytesIO(b"dummy text content"),
            filename="document.pdf",
            headers={"content-type": "application/pdf"}
        )
        with pytest.raises(HTTPException) as exc:
            analyze_image(upload)
        assert exc.value.status_code == 400
        assert "Unsupported format" in exc.value.detail

    def test_analyze_empty_file(self):
        upload = UploadFile(
            file=io.BytesIO(b""),
            filename="empty.jpg",
            headers={"content-type": "image/jpeg"}
        )
        with pytest.raises(HTTPException) as exc:
            analyze_image(upload)
        assert exc.value.status_code == 400
        assert "Uploaded file is empty" in exc.value.detail


class TestEdgeCases:
    """
    Mandatory edge-case verification suite:
    1. Valid image
    2. Non-image file
    3. Oversized file (>10MB)
    4. Image with no face
    5. Image with multiple faces
    6. Corrupted image file
    7. History and PDF generation
    """

    def test_edge_case_valid_image(self, sample_image_bytes):
        upload = UploadFile(
            file=io.BytesIO(sample_image_bytes),
            filename="valid_photo.jpg",
            headers={"content-type": "image/jpeg"}
        )
        res = analyze_image(upload)
        assert res.report_id.startswith("rep_")
        assert res.verdict in ["fake", "real", "uncertain"]
        assert 0.0 <= res.fake_probability <= 1.0
        assert res.processing_time_ms > 0

    def test_edge_case_non_image_file(self):
        upload = UploadFile(
            file=io.BytesIO(b"Plain text file contents not an image"),
            filename="malicious.txt",
            headers={"content-type": "text/plain"}
        )
        with pytest.raises(HTTPException) as exc:
            analyze_image(upload)
        assert exc.value.status_code == 400
        assert "Unsupported format" in exc.value.detail

    def test_edge_case_oversized_file(self):
        # 11 Megabytes (exceeds default MAX_UPLOAD_MB = 10)
        oversized_data = b"0" * (11 * 1024 * 1024)
        upload = UploadFile(
            file=io.BytesIO(oversized_data),
            filename="oversized.png",
            headers={"content-type": "image/png"}
        )
        with pytest.raises(HTTPException) as exc:
            analyze_image(upload)
        assert exc.value.status_code == 413
        assert "exceeds maximum limit" in exc.value.detail

    def test_edge_case_image_with_no_face(self):
        # Uniform solid blue landscape with no human faces
        solid = np.zeros((200, 200, 3), dtype=np.uint8)
        solid[:, :] = [40, 100, 200]
        buf = io.BytesIO()
        Image.fromarray(solid).save(buf, format="JPEG")
        buf.seek(0)

        upload = UploadFile(
            file=buf,
            filename="landscape_no_face.jpg",
            headers={"content-type": "image/jpeg"}
        )
        res = analyze_image(upload)
        assert res.checks.face_detected is False
        assert len(res.faces) == 0
        assert res.verdict in ["fake", "real", "uncertain"]

    def test_edge_case_image_with_multiple_faces(self):
        from facenet_pytorch import MTCNN
        # Create a synthetic image with 2 side-by-side face ovals
        img = Image.new("RGB", (320, 160), color=(220, 220, 220))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Face 1 (left)
        draw.ellipse([20, 20, 120, 140], fill=(235, 195, 170))
        draw.ellipse([45, 55, 60, 68], fill=(30, 30, 30))
        draw.ellipse([80, 55, 95, 68], fill=(30, 30, 30))
        draw.line([70, 75, 70, 95], fill=(180, 120, 90), width=2)
        draw.chord([55, 105, 85, 120], 0, 180, fill=(180, 60, 60))

        # Face 2 (right)
        draw.ellipse([180, 20, 280, 140], fill=(235, 195, 170))
        draw.ellipse([205, 55, 220, 68], fill=(30, 30, 30))
        draw.ellipse([240, 55, 255, 68], fill=(30, 30, 30))
        draw.line([230, 75, 230, 95], fill=(180, 120, 90), width=2)
        draw.chord([215, 105, 245, 120], 0, 180, fill=(180, 60, 60))

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        upload = UploadFile(
            file=buf,
            filename="group_faces.jpg",
            headers={"content-type": "image/jpeg"}
        )
        res = analyze_image(upload)
        assert res.report_id.startswith("rep_")
        assert res.verdict in ["fake", "real", "uncertain"]

    def test_edge_case_corrupted_file(self):
        # File has .jpg extension but corrupt non-image bytes
        corrupted_bytes = b"\xFF\xD8\xFF" + b"Corrupted garbage data string not valid image" * 20
        upload = UploadFile(
            file=io.BytesIO(corrupted_bytes),
            filename="corrupted_portrait.jpg",
            headers={"content-type": "image/jpeg"}
        )
        with pytest.raises(HTTPException) as exc:
            analyze_image(upload)
        assert exc.value.status_code == 400
        assert "Corrupted or invalid image file" in exc.value.detail

    def test_history_and_pdf_generation(self, sample_image_bytes):
        from backend.main import get_history, download_pdf_report
        # Perform a scan first
        upload = UploadFile(
            file=io.BytesIO(sample_image_bytes),
            filename="scan_for_pdf.jpg",
            headers={"content-type": "image/jpeg"}
        )
        res = analyze_image(upload)

        # Verify history endpoint
        history = get_history(limit=10)
        assert isinstance(history, list)
        assert len(history) > 0
        matching = [item for item in history if item["report_id"] == res.report_id]
        assert len(matching) == 1
        assert matching[0]["filename"] == "scan_for_pdf.jpg"

        # Verify PDF report generation endpoint
        pdf_response = download_pdf_report(res.report_id)
        assert pdf_response.media_type == "application/pdf"
        assert f"deepfake_report_{res.report_id}.pdf" in pdf_response.headers["Content-Disposition"]
