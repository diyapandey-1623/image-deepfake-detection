"""
backend/schemas.py
------------------
Pydantic response models for the deepfake detection REST API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FaceDetectionResult(BaseModel):
    bbox: List[int] = Field(..., description="Bounding box [x, y, w, h]")
    fake_probability: float = Field(..., description="Probability of manipulation for this face (0.0 - 1.0)")


class ForensicChecks(BaseModel):
    face_detected: bool = Field(..., description="Whether MTCNN located at least one human face")
    metadata_flags: List[str] = Field(default_factory=list, description="EXIF or metadata modification indicators")
    ela_flag: bool = Field(..., description="Error Level Analysis compression discrepancy flag")


class RegionalInfluenceItem(BaseModel):
    region: str = Field(..., description="Anatomical facial region name")
    influence_level: str = Field(..., description="High, Moderate, or Low")
    color: str = Field(..., description="Color indicator: Red, Yellow, or Blue")
    hex_color: str = Field(..., description="Hex color code (e.g. #FF4D4D)")
    score: float = Field(..., description="Relative influence percentage (0 - 100)")
    description: Optional[str] = Field(default=None, description="Diagnostic observation")


class AnalyzeResponse(BaseModel):
    report_id: str = Field(..., description="Unique investigation report UUID")
    verdict: str = Field(..., description="'fake' (>= 0.50) or 'real' (< 0.50)")
    fake_probability: float = Field(..., description="Overall calculated fake probability (0.0 - 1.0)")
    faces: List[FaceDetectionResult] = Field(default_factory=list, description="Per-face crop detection results")
    heatmap_base64: Optional[str] = Field(default=None, description="Grad-CAM visual heatmap overlay")
    forensic_rationale: Optional[str] = Field(default=None, description="Technical rationale explaining verdict")
    regional_influence: Optional[List[RegionalInfluenceItem]] = Field(default=None, description="Facial region influence breakdown")
    checks: ForensicChecks = Field(..., description="Integrity check outcomes")
    model_name: str = Field(..., description="Active model identifier")
    processing_time_ms: float = Field(default=0.0, description="End-to-end analysis latency in milliseconds")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")


class HealthStatus(BaseModel):
    status: str = Field(..., description="System health ('healthy')")
    model_loaded: bool = Field(..., description="Whether the neural network is initialized in memory")
    model_name: str = Field(..., description="Name or identifier of the loaded model")
    model_backend: str = Field(..., description="Active backend: 'huggingface' or 'custom_pytorch'")
    device: str = Field(..., description="Active compute device ('cpu' or 'cuda')")
