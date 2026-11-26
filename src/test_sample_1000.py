"""Evaluation script for the 1,000-image balanced dataset (500 Real, 500 Fake)."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import csv
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.class_config import IDX_TO_LABEL, resolve_fake_label_id
from src.vit_transforms import get_vit_transform
from src.vit_utils import RealFakeDataset, resolve_vit_model_source, plot_confusion_matrix


def run_evaluation(data_dir=None, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Using device: {device.type.upper()}")

    if data_dir is None:
        data_dir = "data/sample_1000" if os.path.exists("data/sample_1000") else "data/small/test"

    dataset = RealFakeDataset(data_dir, transform=get_vit_transform(), return_meta=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model_path = resolve_vit_model_source()
    print(f"[Model] Loading pretrained weights: {model_path}")
    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
    fake_idx = resolve_fake_label_id(getattr(model.config, "id2label", None))
    real_idx = 1 - fake_idx

    all_preds, all_targets, all_probs, all_paths, all_fnames = [], [], [], [], []
    start_time = time.time()

    with torch.inference_mode():
        for images, labels, paths, fnames in loader:
            logits = model(images.to(device)).logits
            probs = torch.softmax(logits, dim=1)
            preds = (probs[:, fake_idx] >= 0.50).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(torch.stack([probs[:, real_idx], probs[:, fake_idx]], dim=1).cpu().numpy().tolist())
            all_paths.extend(paths)
            all_fnames.extend(fnames)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    errors_csv = os.path.join(output_dir, "errors_sample_1000.csv")
    error_count = 0
    with open(errors_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "filepath", "true_label", "predicted_label", "confidence", "prob_real", "prob_fake"])
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                error_count += 1
                p_real, p_fake = all_probs[i][0], all_probs[i][1]
                writer.writerow([
                    all_fnames[i],
                    all_paths[i],
                    IDX_TO_LABEL[y_true[i]].upper(),
                    IDX_TO_LABEL[y_pred[i]].upper(),
                    f"{max(p_real, p_fake):.4f}",
                    f"{p_real:.4f}",
                    f"{p_fake:.4f}",
                ])

    plot_confusion_matrix(cm, ["REAL", "FAKE"], os.path.join(output_dir, "confusion_matrix_sample_1000.png"), title=f"Confusion Matrix ({data_dir})")

    metrics_path = os.path.join(output_dir, "metrics_sample_1000.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": data_dir,
            "total_samples": len(y_true),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "total_errors": error_count,
            "inference_time_sec": round(time.time() - start_time, 2),
        }, f, indent=2)

    print(f"Accuracy: {acc * 100:.2f}% | Precision: {prec * 100:.2f}% | Recall: {rec * 100:.2f}% | F1: {f1 * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"], digits=4))


if __name__ == "__main__":
    run_evaluation()
