"""
scripts/download_model.py
-------------------------
Downloads and caches the pinned Vision Transformer (ViT) deepfake detection model
from Hugging Face Hub into the local `models/best_model_vit` directory.

Model: dima806/deepfake_vs_real_image_detection
Revision: 29e4cf9efc543845610045f6ba7e88e5cf9d9301
"""

import os
import sys
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "dima806/deepfake_vs_real_image_detection"
REVISION = "29e4cf9efc543845610045f6ba7e88e5cf9d9301"
OUTPUT_DIR = PROJECT_ROOT / "models" / "best_model_vit"

def download_pinned_model():
    print("=" * 65)
    print("        DOWNLOADING PINNED VISION TRANSFORMER MODEL")
    print("=" * 65)
    print(f"Hugging Face ID : {MODEL_ID}")
    print(f"Pinned Revision : {REVISION}")
    print(f"Destination     : {OUTPUT_DIR}")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Downloading image processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
    processor.save_pretrained(OUTPUT_DIR)
    print("  Image processor saved successfully.")

    print("\n[2/2] Downloading model weights and configuration...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=REVISION)
    model.save_pretrained(OUTPUT_DIR)
    print("  Model weights saved successfully.")

    print("\n" + "=" * 65)
    print("SUCCESS: Pinned model ready for deterministic inference.")
    print("=" * 65)

if __name__ == "__main__":
    download_pinned_model()
