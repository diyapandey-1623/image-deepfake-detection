"""
backend/model/checks.py
-----------------------
Forensic integrity checks: MTCNN face cropping, EXIF/metadata inspection, and ELA.
"""

import io
from typing import List, Tuple
import numpy as np
from PIL import ExifTags, Image

SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "midjourney", "stable diffusion",
    "dall-e", "automatic1111", "comfyui", "civitai", "novelai",
    "faceapp", "deepface", "faceswap", "reface", "pixlr", "snapseed",
]


def check_metadata(image: Image.Image) -> Tuple[bool, List[str]]:
    """Inspects image info chunks and EXIF tags for editing/synthesis software markers."""
    flags: List[str] = []

    for key, val in image.info.items():
        val_str = str(val).lower()
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in val_str:
                flags.append(f"Metadata tag '{key}' references editing/synthesis software: '{kw}'")

    try:
        exif = image.getexif()
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
    except Exception as exc:
        print(f"[Metadata Check Warning] Failed to parse EXIF: {exc}")

    return len(flags) > 0, flags


def check_ela(image: Image.Image, quality: int = 90, threshold: float = 15.0) -> Tuple[bool, float]:
    """Computes Error Level Analysis residual delta at the given JPEG quality."""
    rgb = image.convert("RGB") if image.mode != "RGB" else image

    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)

    compressed = Image.open(buf)
    diff = np.abs(np.array(rgb, dtype=np.float32) - np.array(compressed, dtype=np.float32))
    mean_error = float(np.mean(diff))
    return bool(mean_error > threshold), round(mean_error, 2)


def detect_and_crop_faces(
    image: Image.Image,
    mtcnn_detector,
    margin_percent: float = 0.20,
) -> Tuple[List[Tuple[Image.Image, List[int]]], bool]:
    """Detects faces via MTCNN and crops each face with an expanded boundary margin."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_w, img_h = image.size

    try:
        boxes, probs = mtcnn_detector.detect(image)
    except Exception as exc:
        print(f"[MTCNN] Face detection error: {exc}")
        return [], False

    if boxes is None or len(boxes) == 0:
        return [], False

    face_crops: List[Tuple[Image.Image, List[int]]] = []
    for i, box in enumerate(boxes):
        prob = probs[i] if probs is not None and len(probs) > i else 1.0
        if prob is not None and prob < 0.70:
            continue

        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin_percent), int(bh * margin_percent)

        crop_x1 = max(0, int(x1 - mx))
        crop_y1 = max(0, int(y1 - my))
        crop_x2 = min(img_w, int(x2 + mx))
        crop_y2 = min(img_h, int(y2 + my))

        crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        bbox = [crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1]
        face_crops.append((crop, bbox))

    return face_crops, len(face_crops) > 0
