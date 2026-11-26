"""
backend/model/checks.py
-----------------------
Forensic integrity checks:
1. Face Detection and 20% margin cropping via MTCNN (facenet-pytorch)
2. EXIF & Metadata Tag Inspector (detects editing software, AI generator signatures)
3. Error Level Analysis (ELA) compression discrepancy detection
"""

import io
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from PIL import Image, ExifTags


# Editing software & AI generator signature keywords
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "midjourney", "stable diffusion",
    "dall-e", "automatic1111", "comfyui", "civitai", "novelai",
    "faceapp", "deepface", "faceswap", "reface", "pixlr", "snapseed"
]


def check_metadata(image: Image.Image) -> Tuple[bool, List[str]]:
    """
    Inspects image EXIF tags and format metadata for editing software traces
    or synthetic generation markers.

    Returns:
        (has_flags: bool, flag_descriptions: List[str])
    """
    flags: List[str] = []

    # 1. Check PIL image.info dictionary (e.g. PNG text chunks, comments)
    for key, val in image.info.items():
        val_str = str(val).lower()
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in val_str:
                flags.append(f"Metadata tag '{key}' references editing/synthesis software: '{kw}'")

    # 2. Check EXIF tags
    try:
        exif = image._getexif()
        if exif:
            exif_dict = {
                ExifTags.TAGS.get(tag_id, str(tag_id)): val
                for tag_id, val in exif.items()
            }

            software = str(exif_dict.get("Software", "")).lower()
            for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
                if kw in software:
                    flags.append(f"EXIF Software tag indicates modification: '{exif_dict.get('Software')}'")

            artist = str(exif_dict.get("Artist", "")).lower()
            user_comment = str(exif_dict.get("UserComment", "")).lower()
            for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
                if kw in artist or kw in user_comment:
                    flags.append(f"EXIF metadata references synthetic tool: '{kw}'")

            # Check for lack of standard camera hardware tags on JPEG images
            has_camera = "Make" in exif_dict or "Model" in exif_dict
            if not has_camera and image.format == "JPEG":
                flags.append("EXIF present but lacks standard camera manufacturer (Make/Model) tags.")
        else:
            # If JPEG has zero EXIF data, note as common web/synthetic image trait
            if image.format == "JPEG":
                flags.append("No camera EXIF metadata detected (common in stripped or AI-generated imagery).")
    except Exception:
        pass

    has_flags = len(flags) > 0
    return has_flags, flags


def check_ela(image: Image.Image, quality: int = 90, threshold: float = 15.0) -> Tuple[bool, float]:
    """
    Error Level Analysis (ELA): Computes compression error delta at designated JPEG quality.
    A high residual delta indicates differential compression history (e.g. compositing or deepfake splicing).

    Returns:
        (ela_flag: bool, mean_error: float)
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save to memory buffer at JPEG quality
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    # Re-open and calculate absolute difference
    compressed = Image.open(buf)
    orig_arr = np.array(image, dtype=np.float32)
    comp_arr = np.array(compressed, dtype=np.float32)

    diff = np.abs(orig_arr - comp_arr)
    mean_error = float(np.mean(diff))

    # Higher compression delta triggers the ELA flag
    ela_flag = bool(mean_error > threshold)
    return ela_flag, round(mean_error, 2)


def detect_and_crop_faces(
    image: Image.Image,
    mtcnn_detector,
    margin_percent: float = 0.20
) -> Tuple[List[Tuple[Image.Image, List[int]]], bool]:
    """
    Detects faces with MTCNN and crops each face with a 20% margin.

    Returns:
        (face_crops_with_bbox, face_detected: bool)
        where face_crops_with_bbox is a list of tuples: (cropped_pil_image, [x, y, w, h])
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_w, img_h = image.size

    # Detect faces via MTCNN
    try:
        boxes, probs = mtcnn_detector.detect(image)
    except Exception as e:
        print(f"[MTCNN] Face detection error: {e}")
        return [], False

    if boxes is None or len(boxes) == 0:
        return [], False

    face_crops = []
    for i, box in enumerate(boxes):
        prob = probs[i] if probs is not None and len(probs) > i else 1.0
        if prob is not None and prob < 0.70:
            continue

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Apply 20% margin
        mx = int(bw * margin_percent)
        my = int(bh * margin_percent)

        crop_x1 = max(0, int(x1 - mx))
        crop_y1 = max(0, int(y1 - my))
        crop_x2 = min(img_w, int(x2 + mx))
        crop_y2 = min(img_h, int(y2 + my))

        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        bbox = [crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1]
        face_crops.append((crop, bbox))

    if len(face_crops) == 0:
        return [], False

    return face_crops, True
