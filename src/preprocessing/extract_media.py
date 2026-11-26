"""
extract_media.py
----------------
Handles image ingestion, validation, and normalization.

Design rationale for academic viva:
Deepfake detection pipelines in production face diverse input qualities, varied color
spaces, and adversarial or corrupted uploads. Standardizing input validation (mime check,
size limits, dimension checks, color channel expansion) ensures robustness against malformed
inputs and security exploits (e.g. decompression bombs).
"""

import os
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np


# Allowed image extensions and safety limits
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit
MIN_IMAGE_DIM = 64                      # Minimum width/height for meaningful forensic analysis
MAX_IMAGE_DIM = 8192                    # Prevent decompression bombs


class MediaValidationError(Exception):
    """Raised when an uploaded image fails validation checks."""
    pass


def validate_image_file(file_path: Union[str, Path]) -> Path:
    """
    Validates file existence, extension, and size.
    """
    path = Path(file_path)
    if not path.is_file():
        raise MediaValidationError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise MediaValidationError(
            f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise MediaValidationError("File is empty (0 bytes).")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise MediaValidationError(
            f"File size ({file_size / (1024*1024):.1f} MB) exceeds limit of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB."
        )

    return path


def load_image(file_source: Union[str, Path, bytes]) -> np.ndarray:
    """
    Loads an image from file path or raw bytes, converts to RGB uint8 array [H, W, 3].
    
    Returns:
        np.ndarray: RGB image array of shape (H, W, 3), dtype uint8.
    """
    if isinstance(file_source, (str, Path)):
        validated_path = validate_image_file(file_source)
        # Using cv2.imdecode to handle international and Unicode paths cleanly on Windows
        try:
            with open(validated_path, "rb") as f:
                file_bytes = f.read()
            img_array = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise MediaValidationError(f"Error reading image file: {str(e)}")
    elif isinstance(file_source, bytes):
        if len(file_source) > MAX_FILE_SIZE_BYTES:
            raise MediaValidationError("Uploaded bytes exceed maximum allowed file size.")
        img_array = np.frombuffer(file_source, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        raise TypeError(f"Unsupported file_source type: {type(file_source)}")

    if img_bgr is None:
        raise MediaValidationError("Failed to decode image data. File may be corrupted.")

    h, w = img_bgr.shape[:2]
    if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
        raise MediaValidationError(f"Image resolution too small ({w}x{h}). Minimum required is {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}.")
    if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
        raise MediaValidationError(f"Image resolution ({w}x{h}) exceeds maximum permitted dimension ({MAX_IMAGE_DIM}).")

    # OpenCV loads BGR by default; convert to standard RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
