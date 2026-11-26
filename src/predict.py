"""Inference script for Deepfake Detection (Real vs Fake Faces)
Takes a single image path and prints 'real' or 'fake' with a confidence score.
"""

import os
import sys
import argparse
from PIL import Image

import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


def predict_single_image(image_path, model_path=None):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ViT preprocessing: Resize to 224x224, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    # Model resolution
    if model_path and os.path.exists(model_path):
        model_source = model_path
    else:
        candidates = [
            os.path.join(os.getcwd(), 'models', 'pretrained_vit'),
            os.path.join(os.getcwd(), 'models', 'best_model'),
            os.path.join(os.getcwd(), 'models'),
            os.path.join(os.getcwd(), 'pretrained_model')
        ]
        model_source = None
        for cand in candidates:
            if os.path.exists(cand) and os.path.exists(os.path.join(cand, 'model.safetensors')):
                model_source = cand
                break
        if model_source is None:
            model_source = 'dima806/deepfake_vs_real_image_detection'

    model = AutoModelForImageClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # Label mapping: 0 = Real, 1 = Fake (aligned with dima806 config)
    prob_real = float(probs[0].item())
    prob_fake = float(probs[1].item())

    if prob_fake >= 0.5:
        verdict = "fake"
        confidence = prob_fake
    else:
        verdict = "real"
        confidence = prob_real

    return verdict, confidence, prob_real, prob_fake


def main():
    parser = argparse.ArgumentParser(description="Predict whether a face image is Real or Fake.")
    parser.add_argument('image_path', type=str, help="Path to input image file (JPG, PNG, etc.)")
    parser.add_argument('--model_path', type=str, default=None, help="Path to model checkpoint directory")
    args = parser.parse_args()

    try:
        verdict, confidence, prob_real, prob_fake = predict_single_image(args.image_path, args.model_path)
        print("\n" + "=" * 40)
        print(f"File:       {os.path.basename(args.image_path)}")
        print(f"Prediction: {verdict.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Detail:     Real: {prob_real*100:.2f}% | Fake: {prob_fake*100:.2f}%")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
