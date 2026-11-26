"""Fresh evaluation script for the backend HuggingFace ViT model on data/small/test."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model.inference import HuggingFaceModel
from src.class_config import DECISION_THRESHOLD
from src.vit_utils import resolve_vit_model_source


def main() -> None:
    test_dir = PROJECT_ROOT / "data" / "small" / "test"
    real_files = sorted((test_dir / "real").iterdir())
    fake_files = sorted((test_dir / "fake").iterdir())

    model_path = resolve_vit_model_source(project_root=PROJECT_ROOT)
    print(f"Loading ViT model from: {model_path}")
    model = HuggingFaceModel(model_id=model_path, device="cpu")

    y_true, y_pred, probs_real, probs_fake = [], [], [], []

    for label, files, probs_bucket in ((0, real_files, probs_real), (1, fake_files, probs_fake)):
        for img_path in files:
            with Image.open(img_path).convert("RGB") as img:
                p = model.predict(img)["fake_probability"]
            y_true.append(label)
            y_pred.append(1 if p >= DECISION_THRESHOLD else 0)
            probs_bucket.append(p)

    acc = accuracy_score(y_true, y_pred)
    prec_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Total Test Images: {len(y_true)} ({len(real_files)} Real, {len(fake_files)} Fake)")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Real Recall:       {rec_real * 100:.2f}% | Fake Recall: {rec_fake * 100:.2f}%")
    print(f"Fake Precision:    {prec_fake * 100:.2f}% | Fake F1: {f1_fake * 100:.2f}%")
    print(f"Avg P(fake) Real:  {np.mean(probs_real):.4f} (std {np.std(probs_real):.4f})")
    print(f"Avg P(fake) Fake:  {np.mean(probs_fake):.4f} (std {np.std(probs_fake):.4f})")
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
