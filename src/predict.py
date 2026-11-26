"""CLI prediction script for single-image Real vs Fake classification."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from PIL import Image

import torch
from transformers import AutoModelForImageClassification

from src.class_config import get_verdict, resolve_fake_label_id
from src.vit_transforms import get_vit_transform
from src.vit_utils import resolve_vit_model_source


def predict_single_image(image_path, model_path=None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_vit_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model_source = resolve_vit_model_source(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    fake_idx = resolve_fake_label_id(getattr(model.config, "id2label", None))
    real_idx = 1 - fake_idx if len(probs) == 2 else 0
    prob_real = float(probs[real_idx].item())
    prob_fake = float(probs[fake_idx].item())
    verdict = get_verdict(prob_fake)
    confidence = prob_fake if verdict == "fake" else prob_real

    return verdict, confidence, prob_real, prob_fake


def main():
    parser = argparse.ArgumentParser(description="Predict whether a face image is Real or Fake.")
    parser.add_argument("image_path", type=str, help="Path to input image file (JPG, PNG, etc.)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint directory")
    args = parser.parse_args()

    try:
        verdict, confidence, prob_real, prob_fake = predict_single_image(args.image_path, args.model_path)
        print("\n" + "=" * 40)
        print(f"File:       {os.path.basename(args.image_path)}")
        print(f"Prediction: {verdict.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Detail:     Real: {prob_real * 100:.2f}% | Fake: {prob_fake * 100:.2f}%")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
