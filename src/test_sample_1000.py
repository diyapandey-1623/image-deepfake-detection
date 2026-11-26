"""Evaluation script for testing the existing pretrained deepfake detection model
strictly on dataset/sample_1000 (500 REAL, 500 FAKE).
"""

import os
import sys
import json
import csv
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


class Sample1000Dataset(Dataset):
    """Dataset dedicated strictly to dataset/sample_1000."""
    LABEL_MAP = {'real': 0, 'fake': 1}
    INDEX_TO_LABEL = {0: 'REAL', 1: 'FAKE'}

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name, label in self.LABEL_MAP.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                raise FileNotFoundError(f"Missing expected class folder: {class_folder}")
            files = sorted(os.listdir(class_folder))
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(class_folder, fname), label, fname))

        if len(self.samples) != 1000:
            print(f"[Warning] Expected exactly 1,000 samples, but found {len(self.samples)}")

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
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va='center', ha='center', size='xx-large', weight='bold')

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12, fontweight='bold')
    ax.set_yticklabels(class_names, fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('Actual (Ground Truth) Label', fontsize=12, labelpad=10)
    plt.title('Confusion Matrix (dataset/sample_1000)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[Plot] Confusion matrix saved: {output_path}")


def run_evaluation(data_dir=None, output_dir=r'outputs'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Hardware] Using device: {device.type.upper()}")

    # Resolve default data directory: prefer data/sample_1000 then data/small/test
    if data_dir is None:
        if os.path.exists('data/sample_1000'):
            data_dir = 'data/sample_1000'
        elif os.path.exists('data/small/test'):
            data_dir = 'data/small/test'
        else:
            data_dir = 'dataset/sample_1000'

    # ViT preprocessing: 224x224, normalized with mean 0.5, std 0.5
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = Sample1000Dataset(data_dir, transform=preprocess)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Locate pretrained model: prefer models/pretrained_vit, then pretrained_model, then HF
    candidates = [
        os.path.join(os.getcwd(), 'models', 'pretrained_vit'),
        os.path.join(os.getcwd(), 'models'),
        os.path.join(os.getcwd(), 'pretrained_model')
    ]
    model_path = None
    for cand in candidates:
        if os.path.exists(cand) and os.path.exists(os.path.join(cand, 'model.safetensors')):
            model_path = cand
            break
    if model_path is None:
        model_path = 'dima806/deepfake_vs_real_image_detection'

    print(f"[Model] Loading existing pretrained weights: {model_path}")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []
    all_paths = []
    all_fnames = []

    print(f"[Inference] Processing {len(dataset)} images from '{data_dir}'...")
    start_time = time.time()

    with torch.inference_mode():
        for batch_idx, (images, labels, paths, fnames) in enumerate(loader):
            images = images.to(device)
            logits = model(images).logits
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(paths)
            all_fnames.extend(fnames)

    inference_time = time.time() - start_time
    print(f"[Inference] Completed in {inference_time:.2f}s ({inference_time/len(dataset)*1000:.1f}ms/image)")

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    correct_mask = (y_true == y_pred)
    num_correct = int(np.sum(correct_mask))
    num_incorrect = int(len(y_true) - num_correct)
    num_real = int(np.sum(y_true == 0))
    num_fake = int(np.sum(y_true == 1))

    # 1. Save detailed predictions CSV
    csv_path = os.path.join(output_dir, 'sample_1000_predictions.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_filename', 'actual_label', 'predicted_label', 'confidence_score', 'status', 'prob_real', 'prob_fake', 'file_path'])
        for i in range(len(y_true)):
            fname = all_fnames[i]
            act = Sample1000Dataset.INDEX_TO_LABEL[y_true[i]]
            pred = Sample1000Dataset.INDEX_TO_LABEL[y_pred[i]]
            p_real = all_probs[i][0]
            p_fake = all_probs[i][1]
            conf = max(p_real, p_fake)
            status = 'Correct' if y_true[i] == y_pred[i] else 'Incorrect'
            writer.writerow([fname, act, pred, f"{conf*100:.2f}%", status, f"{p_real:.4f}", f"{p_fake:.4f}", all_paths[i]])
    print(f"[CSV] Predictions written: {csv_path}")

    # 2. Confusion matrix plot
    cm_plot_path = os.path.join(output_dir, 'sample_1000_confusion_matrix.png')
    plot_confusion_matrix(cm, ['REAL', 'FAKE'], cm_plot_path)

    # 3. Save Summary Report (Markdown)
    summary_md_path = os.path.join(output_dir, 'sample_1000_summary.md')
    summary_md = f"""# Deepfake Detection Pretrained Model Evaluation Summary

**Target Dataset:** `{data_dir}` (Balanced Sample)  
**Model Architecture:** Vision Transformer (`ViTForImageClassification` - `dima806/deepfake_vs_real_image_detection`)  
**Hardware Compute Device:** `{device.type.upper()}`  
**Inference Latency:** {inference_time:.2f} seconds ({inference_time/len(dataset)*1000:.1f} ms/image)  

---

## 1. Overall Performance Statistics

| Metric | Value |
| :--- | :--- |
| **Total Images Tested** | **{len(y_true):,}** |
| **REAL Images Tested** | **{num_real:,}** |
| **FAKE Images Tested** | **{num_fake:,}** |
| **Correct Predictions** | **{num_correct:,}** |
| **Incorrect Predictions** | **{num_incorrect:,}** |
| **Overall Accuracy** | **{acc * 100:.2f}%** |
| **Precision (FAKE)** | **{prec_fake * 100:.2f}%** |
| **Recall (FAKE)** | **{rec_fake * 100:.2f}%** |
| **F1-Score (FAKE)** | **{f1_fake * 100:.2f}%** |

---

## 2. Confusion Matrix

| Ground Truth \\ Predicted | Predicted REAL | Predicted FAKE | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual REAL** | **{cm[0, 0]}** (True Real) | **{cm[0, 1]}** (False Fake) | **{num_real}** |
| **Actual FAKE** | **{cm[1, 0]}** (False Real) | **{cm[1, 1]}** (True Fake) | **{num_fake}** |
| **Total Predicted** | **{cm[0, 0] + cm[1, 0]}** | **{cm[0, 1] + cm[1, 1]}** | **{len(y_true)}** |

---

## 3. Detailed Scikit-Learn Classification Report

```
{classification_report(y_true, y_pred, target_names=['REAL (0)', 'FAKE (1)'], digits=4)}
```

---

## 4. Generated Artifacts
- **Detailed CSV Log:** [`sample_1000_predictions.csv`](file:///{os.path.abspath(csv_path).replace('\\', '/')})
- **Confusion Matrix Plot:** [`sample_1000_confusion_matrix.png`](file:///{os.path.abspath(cm_plot_path).replace('\\', '/')})
"""
    with open(summary_md_path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"[Report] Summary report written: {summary_md_path}")

    # Print summary to console
    print("\n" + "=" * 55)
    print("        PRETRAINED MODEL EVALUATION REPORT        ")
    print("=" * 55)
    print(f"Total Images Tested:    {len(y_true):,}")
    print(f"  - Real Images Tested: {num_real:,}")
    print(f"  - Fake Images Tested: {num_fake:,}")
    print(f"Correct Predictions:    {num_correct:,} ({acc*100:.2f}%)")
    print(f"Incorrect Predictions:  {num_incorrect:,} ({(1-acc)*100:.2f}%)")
    print("-" * 55)
    print(f"Accuracy:               {acc * 100:.2f}%")
    print(f"Precision (Fake):       {prec_fake * 100:.2f}%")
    print(f"Recall (Fake):          {rec_fake * 100:.2f}%")
    print(f"F1-Score (Fake):        {f1_fake * 100:.2f}%")
    print("-" * 55)
    print("Confusion Matrix:")
    print(f"  Actual REAL -> Predicted REAL: {cm[0, 0]} | Predicted FAKE: {cm[0, 1]}")
    print(f"  Actual FAKE -> Predicted REAL: {cm[1, 0]} | Predicted FAKE: {cm[1, 1]}")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'outputs'
    run_evaluation(data_path, out_path)
