"""Evaluation script for Deepfake Detection (Real vs Fake Faces)
Model: dima806/deepfake_vs_real_image_detection (ViTForImageClassification)

Evaluates on data_small/test and reports:
- Accuracy, Precision, Recall, F1
- Confusion Matrix (printed and saved as heatmap PNG)
- Misclassified images exported to /output/errors.csv
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.class_config import CLASS_TO_IDX, IDX_TO_LABEL
from src.vit_transforms import get_vit_transform


class RealFakeDataset(Dataset):
    """Custom dataset using shared class mapping: 'real' -> 0, 'fake' -> 1."""
    CLASS_MAP = CLASS_TO_IDX
    INDEX_TO_NAME = {0: 'real', 1: 'fake'}

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name, label in self.CLASS_MAP.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for fname in sorted(os.listdir(class_folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(class_folder, fname), label, fname))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, fname = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, path, fname


def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size='xx-large', weight='bold')

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.title('Confusion Matrix (data_small/test)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model on data/small/test or sample_1000.")
    default_test = 'data/small/test' if os.path.exists('data/small/test') else ('data/sample_1000' if os.path.exists('data/sample_1000') else 'data_small/test')
    parser.add_argument('--test_dir', type=str, default=default_test, help="Path to test set containing real/ and fake/")
    parser.add_argument('--model_path', type=str, default=None, help="Path to model directory or Hugging Face model ID")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save errors.csv and metrics")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    # Hardware detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using compute hardware: {device.type.upper()}")

    # ViT 224x224 normalization via shared transform
    eval_transform = get_vit_transform()

    test_dataset = RealFakeDataset(args.test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[Data] Loaded {len(test_dataset)} test images ({sum(1 for _, l, _ in test_dataset.samples if l == 0)} real, {sum(1 for _, l, _ in test_dataset.samples if l == 1)} fake)")

    # Model resolution
    if args.model_path:
        model_source = args.model_path
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

    print(f"[Model] Loading model from: {model_source}")
    model = AutoModelForImageClassification.from_pretrained(model_source)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_paths = []
    all_fnames = []
    all_probs = []

    print("[Evaluation] Running inference over test set...")
    with torch.no_grad():
        for batch_idx, (images, labels, paths, fnames) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images).logits
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(paths)
            all_fnames.extend(fnames)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Metrics computation
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Export errors to /output/errors.csv
    os.makedirs(args.output_dir, exist_ok=True)
    errors_csv_path = os.path.join(args.output_dir, 'errors.csv')

    error_count = 0
    with open(errors_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'filepath', 'true_label', 'predicted_label', 'confidence', 'prob_real', 'prob_fake'])
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                error_count += 1
                t_label = RealFakeDataset.INDEX_TO_NAME[y_true[i]]
                p_label = RealFakeDataset.INDEX_TO_NAME[y_pred[i]]
                p_real = all_probs[i][0]
                p_fake = all_probs[i][1]
                conf = max(p_real, p_fake)
                writer.writerow([all_fnames[i], all_paths[i], t_label, p_label, f"{conf:.4f}", f"{p_real:.4f}", f"{p_fake:.4f}"])

    print(f"[Errors] Saved {error_count} misclassified images to: {errors_csv_path}")

    # Plot confusion matrix
    cm_plot_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, ['Real', 'Fake'], cm_plot_path)

    # Save JSON metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_summary.json')
    metrics = {
        'total_samples': len(y_true),
        'accuracy': float(acc),
        'precision_fake': float(prec),
        'recall_fake': float(rec),
        'f1_score_fake': float(f1),
        'total_errors': error_count,
        'confusion_matrix': {
            'true_real_pred_real': int(cm[0, 0]),
            'true_real_pred_fake': int(cm[0, 1]),
            'true_fake_pred_real': int(cm[1, 0]),
            'true_fake_pred_fake': int(cm[1, 1])
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Console Report
    print("\n" + "=" * 55)
    print("           MODEL TEST EVALUATION SUMMARY          ")
    print("=" * 55)
    print(f"Total Test Images:   {len(y_true)} (250 Real, 250 Fake)")
    print(f"Accuracy:            {acc * 100:.2f}%")
    print(f"Precision (Fake):    {prec * 100:.2f}%")
    print(f"Recall (Fake):       {rec * 100:.2f}%")
    print(f"F1-Score (Fake):     {f1 * 100:.2f}%")
    print(f"Misclassified Count: {error_count} / {len(y_true)}")
    print("-" * 55)
    print("Confusion Matrix:")
    print(f"  Real -> Predicted Real: {cm[0, 0]} | Predicted Fake: {cm[0, 1]}")
    print(f"  Fake -> Predicted Real: {cm[1, 0]} | Predicted Fake: {cm[1, 1]}")
    print("-" * 55)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'], digits=4))
    print("=" * 55)


if __name__ == '__main__':
    main()
