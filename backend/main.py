"""
backend/main.py
---------------
FastAPI REST Application for Pretrained Deepfake Detection.

Features:
- Model-agnostic integration (Hugging Face / Custom PyTorch) configured via backend/config.py
- Face detection & 20% margin cropping via MTCNN (facenet-pytorch)
- Fallback full-image analysis when no human faces are detected
- Grad-CAM explainability heatmaps on convolutional layers
- Metadata / EXIF modification tags and Error Level Analysis (ELA) integrity checks
- Lifespan model caching, CORS middleware, and safe temporary file cleanup
"""

import io
import os
import uuid
import time
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Ensure project root is always in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.config import (
    MODEL_BACKEND,
    MODEL_ID,
    DEVICE,
    FAKE_THRESHOLD,
    REAL_THRESHOLD,
    MAX_UPLOAD_MB,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    CORS_ORIGINS,
)
from backend.schemas import AnalyzeResponse, HealthStatus, FaceDetectionResult, ForensicChecks
from backend.model.inference import create_model_loader, BaseDeepfakeModel, get_verdict
from backend.model.checks import check_metadata, check_ela, detect_and_crop_faces
from backend.database import init_db, insert_scan, get_all_scans, get_scan_by_report_id
from backend.pdf_report import generate_pdf_report


# Global lifespan context manager for one-time model initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n[Lifespan Startup] Initializing Deepfake Detection System...")
    print(f"  Backend: {MODEL_BACKEND}")
    print(f"  Model ID: {MODEL_ID}")
    print(f"  Device: {DEVICE}")

    # Initialize SQLite database schema
    init_db()

    # 1. Initialize MTCNN Face Detector once on startup
    from facenet_pytorch import MTCNN
    try:
        app.state.mtcnn = MTCNN(keep_all=True, device=DEVICE)
        print("[Lifespan Startup] MTCNN Face Detector initialized successfully.")
    except Exception as e:
        print(f"[Lifespan Startup] MTCNN initialization warning: {e}. Falling back to CPU.")
        app.state.mtcnn = MTCNN(keep_all=True, device="cpu")

    # 2. Initialize Model Loader once on startup
    try:
        app.state.model = create_model_loader()
        print(f"[Lifespan Startup] Model loaded successfully: '{app.state.model.model_name}'")
    except Exception as e:
        print(f"[Lifespan Startup Error] Primary model loading failed: {e}")
        # Fallback to custom_pytorch with local architecture if HuggingFace network is blocked
        if MODEL_BACKEND == "huggingface":
            print("[Lifespan Startup] Falling back to local 'custom_pytorch' backend for offline resiliency...")
            from backend.model.inference import CustomPyTorchModel
            app.state.model = CustomPyTorchModel(device=DEVICE)
        else:
            raise e

    yield

    # Teardown logic on shutdown
    print("[Lifespan Shutdown] Unloading models and clearing resources.")
    app.state.model = None
    app.state.mtcnn = None


# Create FastAPI instance with lifespan event handler
app = FastAPI(
    title="Model-Agnostic Deepfake Detection API",
    description="Deepfake detection REST API supporting interchangeable Hugging Face and custom PyTorch backends.",
    version="2.0.0",
    lifespan=lifespan
)

# Helper functions for robust access/lazy-loading outside lifespan context (e.g. testing)
def get_active_model() -> BaseDeepfakeModel:
    model: Optional[BaseDeepfakeModel] = getattr(app.state, "model", None)
    if model is None:
        try:
            model = create_model_loader()
        except Exception:
            from backend.model.inference import CustomPyTorchModel
            model = CustomPyTorchModel(device=DEVICE)
        app.state.model = model
    return model


def get_active_mtcnn():
    mtcnn = getattr(app.state, "mtcnn", None)
    if mtcnn is None:
        from facenet_pytorch import MTCNN
        try:
            mtcnn = MTCNN(keep_all=True, device=DEVICE)
        except Exception:
            mtcnn = MTCNN(keep_all=True, device="cpu")
        app.state.mtcnn = mtcnn
    return mtcnn


# CORS Middleware for React frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # Allows localhost:5173, localhost:3000, and wildcard in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthStatus)
@app.get("/api/health", response_model=HealthStatus)
def health_check():
    """
    Health check endpoint reporting loaded model details, active backend, and compute device.
    """
    model = get_active_model()
    return HealthStatus(
        status="healthy",
        model_loaded=model is not None,
        model_name=model.model_name if model else "uninitialized",
        model_backend=MODEL_BACKEND,
        device=DEVICE
    )


@app.get("/api/sample/{kind}")
def get_sample_image(kind: str):
    """Returns sample real or fake image for testing and interactive previews."""
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / f"sample_{kind}.jpg"
    if not sample_path.exists():
        from tests.generate_test_samples import create_synthetic_portrait
        arr = create_synthetic_portrait(is_fake=(kind == "fake"))
        Image.fromarray(arr).save(sample_path, format="JPEG")
    return FileResponse(str(sample_path), media_type="image/jpeg")


@app.post("/analyze", response_model=AnalyzeResponse)
@app.post("/api/analyze", response_model=AnalyzeResponse)
@app.post("/predict", response_model=AnalyzeResponse)
@app.post("/api/predict", response_model=AnalyzeResponse)
def analyze_image(file: UploadFile = File(...)):
    """
    Complete analysis pipeline:
    1. Validation: file format, size <= MAX_UPLOAD_MB, PIL integrity verification.
    2. MTCNN face detection: crop with 20% margin. If no face, analyze full image (face_detected=False).
    3. Multi-crop inference: run predict() on each face crop (or full image).
    4. Explainability: generate Grad-CAM heatmap on convolutional layers (null for ViT).
    5. Forensics: EXIF/metadata modification tags and Error Level Analysis (ELA).
    6. Returns structured JSON report with categorical verdict.
    """
    # 1. Validate file extension & MIME type upfront
    filename = file.filename or "upload.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{file.content_type}'. Allowed MIME types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        )

    model = get_active_model()
    mtcnn = get_active_mtcnn()
    if model is None or mtcnn is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is still initializing.")

    # 2. Validate file size (MAX_UPLOAD_MB)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    temp_file_path = None
    start_time = time.perf_counter()

    try:
        contents = file.file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(contents) / (1024*1024):.1f}MB) exceeds maximum limit of {MAX_UPLOAD_MB}MB."
            )

        # 3. Real image validation with PIL
        try:
            image_buffer = io.BytesIO(contents)
            with Image.open(image_buffer) as img_check:
                img_check.verify()
            # Reopen fresh instance after verify()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupted or invalid image file: {str(e)}"
            )

        # Performance optimization: resize excessively large images before heavy CNN/MTCNN inference
        if max(pil_image.size) > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        # 4. Detect faces with MTCNN and crop with margin
        try:
            face_crops_with_bbox, face_detected = detect_and_crop_faces(pil_image, mtcnn, margin_percent=0.25)
        except Exception as e:
            print(f"[Face Detection Error] {e}")
            face_crops_with_bbox, face_detected = [], False

        faces_results: List[FaceDetectionResult] = []
        primary_target_image: Image.Image = pil_image

        if face_detected and len(face_crops_with_bbox) > 0:
            # Sort crops by bounding box area to identify primary face
            face_crops_with_bbox.sort(key=lambda c: c[1][2] * c[1][3], reverse=True)
            primary_crop, primary_bbox = face_crops_with_bbox[0]

            # Check if input is ALREADY a cropped face portrait (>55% of image area)
            img_w, img_h = pil_image.size
            bbox_area_ratio = (primary_bbox[2] * primary_bbox[3]) / max(1, (img_w * img_h))

            # If already a portrait crop, using full image preserves natural hairline & boundary
            if bbox_area_ratio > 0.55:
                primary_target_image = pil_image
            else:
                primary_target_image = primary_crop

            for crop_img, bbox in face_crops_with_bbox:
                eval_img = pil_image if (bbox_area_ratio > 0.55 and bbox == primary_bbox) else crop_img
                pred = model.predict(eval_img)
                faces_results.append(FaceDetectionResult(
                    bbox=bbox,
                    fake_probability=pred["fake_probability"]
                ))
            # Primary face probability drives overall verdict, avoiding noisy multi-detection bias
            overall_fake_prob = faces_results[0].fake_probability
        else:
            # Fallback: run on full image when no faces detected
            pred = model.predict(pil_image)
            overall_fake_prob = pred["fake_probability"]

        # 5. Categorical verdict calculation
        verdict = get_verdict(overall_fake_prob, fake_thresh=FAKE_THRESHOLD, real_thresh=REAL_THRESHOLD)

        # 6. Compute Grad-CAM heatmap overlay & regional anatomical influence breakdown
        heatmap_base64, regional_influence, forensic_rationale = model.generate_heatmap_and_regions(
            primary_target_image,
            verdict=verdict,
            fake_prob=overall_fake_prob
        )

        # 7. Forensic checks: Metadata / EXIF and Error Level Analysis (ELA)
        has_meta_flags, meta_flags = check_metadata(pil_image)
        ela_flag, ela_score = check_ela(primary_target_image, quality=90)

        # 8. Record processing latency and database history
        processing_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        report_id = f"rep_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            insert_scan(
                report_id=report_id,
                filename=filename,
                verdict=verdict,
                fake_probability=round(overall_fake_prob, 4),
                face_detected=face_detected,
                total_faces=len(faces_results),
                metadata_flags=meta_flags,
                ela_flag=ela_flag,
                model_name=model.model_name,
                processing_time_ms=processing_time_ms,
                created_at=created_at,
                heatmap_base64=heatmap_base64,
                regional_influence=regional_influence,
                forensic_rationale=forensic_rationale
            )
        except Exception as db_err:
            print(f"[Database Error] Failed to persist scan history: {db_err}")

        # 9. Assemble structured JSON response
        report = AnalyzeResponse(
            report_id=report_id,
            verdict=verdict,
            fake_probability=round(overall_fake_prob, 4),
            faces=faces_results,
            heatmap_base64=heatmap_base64,
            gradcam_url=heatmap_base64,
            forensic_rationale=forensic_rationale,
            regional_influence=regional_influence,
            checks=ForensicChecks(
                face_detected=face_detected,
                metadata_flags=meta_flags,
                ela_flag=ela_flag
            ),
            model_name=model.model_name,
            processing_time_ms=processing_time_ms,
            created_at=created_at
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error occurred: {str(e)}"
        )
    finally:
        # Guarantee cleanup of any temporary disk files
        if temp_file_path and Path(temp_file_path).exists():
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


@app.get("/history")
@app.get("/api/history")
def get_history(limit: int = 50):
    """Retrieves recent scan records from SQLite database."""
    return get_all_scans(limit=limit)


@app.get("/report/{report_id}/pdf")
@app.get("/api/report/{report_id}/pdf")
def download_pdf_report(report_id: str):
    """Generates and downloads a forensic PDF report for a given report ID."""
    scan = get_scan_by_report_id(report_id)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan report '{report_id}' not found."
        )

    pdf_buffer = generate_pdf_report(scan)
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="deepfake_report_{report_id}.pdf"'
        }
    )


# Serve frontend static build if available (SPA Fallback support)
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        target = frontend_dist / full_path
        if target.is_file():
            return FileResponse(str(target))
        index_file = frontend_dist / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        return {"message": "Frontend build index.html not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
