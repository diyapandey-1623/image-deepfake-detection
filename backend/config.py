"""
backend/config.py
-----------------
Centralized application configuration with environment variable overrides.
Enables model-agnostic swapping between Hugging Face pipelines and custom PyTorch architectures.
"""

import os
from pathlib import Path
import torch


def get_default_device() -> str:
    """Auto-detects CUDA GPU if available, otherwise falls back to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# Candidate paths for pretrained local Hugging Face ViT model
_vit_paths = [
    Path(__file__).resolve().parent.parent / "models" / "best_model_vit",
    Path(__file__).resolve().parent.parent / "models" / "pretrained_vit",
    Path(__file__).resolve().parent / "models" / "pretrained_vit"
]
DEFAULT_MODEL_ID = next((str(p) for p in _vit_paths if p.exists()), "dima806/deepfake_vs_real_image_detection")

# Model backend selection: "huggingface" (default for ViT) or "custom_pytorch"
MODEL_BACKEND: str = os.getenv("MODEL_BACKEND", "huggingface").lower()

# Hugging Face Model ID or local path
MODEL_ID: str = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)

# Compute Device
DEVICE: str = os.getenv("DEVICE", get_default_device())

from backend.class_config import DECISION_THRESHOLD

# Classification Decision Threshold (clean binary 0.50)
FAKE_THRESHOLD: float = DECISION_THRESHOLD
REAL_THRESHOLD: float = DECISION_THRESHOLD

# Upload Size Limit (in Megabytes)
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))

# Custom PyTorch Model Configuration (if MODEL_BACKEND == "custom_pytorch")
_candidate_paths = [
    Path(__file__).resolve().parent.parent / "models" / "model.pth",
    Path(__file__).resolve().parent / "model" / "model.pth"
]
DEFAULT_CUSTOM_MODEL_PATH = next((str(p) for p in _candidate_paths if p.exists()), str(_candidate_paths[0]))
CUSTOM_MODEL_PATH: str = os.getenv("CUSTOM_MODEL_PATH", DEFAULT_CUSTOM_MODEL_PATH)
CUSTOM_ARCHITECTURE: str = os.getenv("CUSTOM_ARCHITECTURE", "efficientnet_b0")

# Allowed Image Formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# CORS Origins
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
