"""
backend/main.py
---------------
FastAPI REST Application for Deepfake Detection & Facial Forensics.
"""

import io
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from backend.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    CUSTOM_MODEL_PATH,
    DEVICE,
    FACE_MARGIN_PERCENT,
    FRONTEND_DIST_DIR,
    MAX_UPLOAD_MB,
    MODEL_BACKEND,
    MODEL_ID,
    SAMPLES_DIR,
)
from backend.database import get_all_scans, get_scan_by_report_id, init_db, insert_scan
from backend.model.checks import check_ela, check_metadata, detect_and_crop_faces
from backend.model.inference import BaseDeepfakeModel, CustomPyTorchModel, create_model_loader
from backend.model.samples import create_synthetic_portrait
from backend.pdf_report import generate_pdf_report
from backend.schemas import AnalyzeResponse, FaceDetectionResult, ForensicChecks, HealthStatus
from src.class_config import get_verdict


def _init_mtcnn():
    from facenet_pytorch import MTCNN

    try:
        return MTCNN(keep_all=True, device=DEVICE)
    except Exception as exc:
        print(f"[Startup] MTCNN initialization warning on {DEVICE}: {exc}. Falling back to CPU.")
        return MTCNN(keep_all=True, device="cpu")


def _init_model() -> BaseDeepfakeModel:
    try:
        return create_model_loader()
    except Exception as exc:
        if MODEL_BACKEND == "huggingface" and Path(CUSTOM_MODEL_PATH).is_file():
            print(f"[Startup] Hugging Face load failed ({exc}); falling back to trained CustomPyTorchModel.")
            return CustomPyTorchModel(device=DEVICE)
        raise RuntimeError(
            f"Primary model '{MODEL_ID}' failed to load ({exc}) and no trained fallback weights exist at '{CUSTOM_MODEL_PATH}'."
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.seed import seed_everything
    seed_everything(42)

    print(f"[Startup] Backend={MODEL_BACKEND} | Model={MODEL_ID} | Device={DEVICE}")
    init_db()
    app.state.mtcnn = _init_mtcnn()
    app.state.model = _init_model()
    yield
    app.state.model = None
    app.state.mtcnn = None


app = FastAPI(
    title="Model-Agnostic Deepfake Detection API",
    description="Deepfake detection REST API supporting Hugging Face and custom PyTorch backends.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_active_model() -> BaseDeepfakeModel:
    model: Optional[BaseDeepfakeModel] = getattr(app.state, "model", None)
    if model is None:
        model = _init_model()
        app.state.model = model
    return model


def get_active_mtcnn():
    mtcnn = getattr(app.state, "mtcnn", None)
    if mtcnn is None:
        mtcnn = _init_mtcnn()
        app.state.mtcnn = mtcnn
    return mtcnn


@app.get("/health", response_model=HealthStatus)
@app.get("/api/health", response_model=HealthStatus)
def health_check():
    model = get_active_model()
    return HealthStatus(
        status="healthy",
        model_loaded=model is not None,
        model_name=model.model_name if model else "uninitialized",
        model_backend=MODEL_BACKEND,
        device=DEVICE,
    )


@app.get("/api/sample/{kind}")
def get_sample_image(kind: str):
    if kind not in ("real", "fake"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Sample kind must be 'real' or 'fake'.")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SAMPLES_DIR / f"sample_{kind}.jpg"
    if not sample_path.exists():
        arr = create_synthetic_portrait(is_fake=(kind == "fake"))
        Image.fromarray(arr).save(sample_path, format="JPEG")
    return FileResponse(str(sample_path), media_type="image/jpeg")


@app.post("/analyze", response_model=AnalyzeResponse)
@app.post("/api/analyze", response_model=AnalyzeResponse)
@app.post("/predict", response_model=AnalyzeResponse)
@app.post("/api/predict", response_model=AnalyzeResponse)
def analyze_image(file: UploadFile = File(...)):
    filename = file.filename or "upload.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{file.content_type}'. Allowed MIME types: {', '.join(sorted(ALLOWED_MIME_TYPES))}",
        )

    model = get_active_model()
    mtcnn = get_active_mtcnn()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    start_time = time.perf_counter()

    try:
        contents = file.file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size ({len(contents) / (1024 * 1024):.1f}MB) exceeds maximum limit of {MAX_UPLOAD_MB}MB.",
            )

        try:
            with Image.open(io.BytesIO(contents)) as img_check:
                img_check.verify()
            raw_image = Image.open(io.BytesIO(contents))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupted or invalid image file: {exc}",
            )

        # Inspect EXIF/metadata and ELA on the unresized original before RGB conversion/downscaling
        _, meta_flags = check_metadata(raw_image)
        ela_flag, _ = check_ela(raw_image, quality=90)

        # Apply EXIF orientation before RGB conversion so portrait camera/phone photos aren't fed sideways
        pil_image = ImageOps.exif_transpose(raw_image).convert("RGB")
        if max(pil_image.size) > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            # Normalize high-res camera sensor grain to standard web JPEG (matching ViT training distribution)
            norm_buf = io.BytesIO()
            pil_image.save(norm_buf, format="JPEG", quality=90)
            norm_buf.seek(0)
            pil_image = Image.open(norm_buf).convert("RGB")

        try:
            face_crops_with_bbox, face_detected = detect_and_crop_faces(
                pil_image,
                mtcnn,
                margin_percent=FACE_MARGIN_PERCENT,
            )
        except Exception as exc:
            print(f"[Face Detection Error] {exc}")
            face_crops_with_bbox, face_detected = [], False

        faces_results: List[FaceDetectionResult] = []
        primary_target_image: Image.Image = pil_image

        if face_detected and face_crops_with_bbox:
            face_crops_with_bbox.sort(key=lambda c: c[1][2] * c[1][3], reverse=True)
            primary_crop, primary_bbox = face_crops_with_bbox[0]

            img_w, img_h = pil_image.size
            bbox_area_ratio = (primary_bbox[2] * primary_bbox[3]) / max(1, img_w * img_h)
            # If the uploaded image is already a tight portrait crop, keep full frame for hairline/jawline context
            use_full_for_primary = bbox_area_ratio > 0.55
            primary_target_image = pil_image if use_full_for_primary else primary_crop

            for idx, (crop_img, bbox) in enumerate(face_crops_with_bbox):
                eval_img = pil_image if (use_full_for_primary and idx == 0) else crop_img
                pred = model.predict(eval_img)
                faces_results.append(FaceDetectionResult(bbox=bbox, fake_probability=pred["fake_probability"]))
            overall_fake_prob = faces_results[0].fake_probability
        else:
            overall_fake_prob = model.predict(pil_image)["fake_probability"]

        verdict = get_verdict(overall_fake_prob)
        heatmap_base64, regional_influence, forensic_rationale = model.generate_heatmap_and_regions(
            primary_target_image,
            verdict=verdict,
            fake_prob=overall_fake_prob,
        )

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
                forensic_rationale=forensic_rationale,
                faces=[f.model_dump() for f in faces_results],
            )
        except Exception as db_err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist scan report: {db_err}",
            ) from db_err

        print(
            f"[Analyze] file='{filename}' | faces={len(faces_results)} (detected={face_detected}) "
            f"| verdict={verdict} (p_fake={overall_fake_prob:.4f}) | ela={ela_flag} | {processing_time_ms:.1f}ms"
        )

        return AnalyzeResponse(
            report_id=report_id,
            verdict=verdict,
            fake_probability=round(overall_fake_prob, 4),
            faces=faces_results,
            heatmap_base64=heatmap_base64,
            forensic_rationale=forensic_rationale,
            regional_influence=regional_influence,
            checks=ForensicChecks(
                face_detected=face_detected,
                metadata_flags=meta_flags,
                ela_flag=ela_flag,
            ),
            model_name=model.model_name,
            processing_time_ms=processing_time_ms,
            created_at=created_at,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error occurred: {exc}",
        ) from exc


@app.get("/history")
@app.get("/api/history")
def get_history(limit: int = 50):
    return get_all_scans(limit=limit)


@app.get("/report/{report_id}/pdf")
@app.get("/api/report/{report_id}/pdf")
def download_pdf_report(report_id: str):
    scan = get_scan_by_report_id(report_id)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan report '{report_id}' not found.",
        )

    return StreamingResponse(
        generate_pdf_report(scan),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="deepfake_report_{report_id}.pdf"'},
    )


if FRONTEND_DIST_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        target = FRONTEND_DIST_DIR / full_path
        if target.is_file():
            return FileResponse(str(target))
        index_file = FRONTEND_DIST_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        return {"message": "Frontend build index.html not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
