"""
backend/config.py
-----------------
Centralized application configuration with environment variable overrides.
"""

import os
from pathlib import Path
import torch

from src.vit_utils import resolve_vit_model_source

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_BACKEND: str = os.getenv("MODEL_BACKEND", "huggingface").lower()
MODEL_ID: str = os.getenv("MODEL_ID", resolve_vit_model_source(project_root=PROJECT_ROOT))
DEVICE: str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
FACE_MARGIN_PERCENT: float = 0.20

DB_PATH: Path = Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "scans.db")))
SAMPLES_DIR: Path = PROJECT_ROOT / "data" / "samples"
FRONTEND_DIST_DIR: Path = PROJECT_ROOT / "frontend" / "dist"

CUSTOM_MODEL_PATH: str = os.getenv(
    "CUSTOM_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "model.pth"),
)
CUSTOM_ARCHITECTURE: str = os.getenv("CUSTOM_ARCHITECTURE", "efficientnet_b0")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
