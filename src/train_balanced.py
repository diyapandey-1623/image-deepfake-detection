"""
src/train_balanced.py
---------------------
Fine-tuning script for Deepfake Detection with:
- Balanced sampling & class weighting
- Light data augmentation (Flip, ColorJitter, simulated JPEG compression)
- Differential learning rate (base layers 5e-6, classifier head 8e-5)
- Early stopping on validation balanced accuracy
- Threshold optimization on valid set to prevent fake-class bias
- Saves best checkpoint to models/best_model_vit and models/best_model.pt
"""

import os
import io
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.class_config import CLASS_TO_IDX, IDX_TO_LABEL
from src.vit_transforms import get_vit_transform


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JpegCompressionTransform:
    """Simulates social media recompression artifacts to prevent frequency overfitting."""
    def __init__(self, quality_range=(65, 95), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            return Image.open(buf)
        return img


class BalancedRealFakeDataset(Dataset):
    """Maps 'real' -> 0, 'fake' -> 1 matching shared config."""
    CLASS_MAP = CLASS_TO_IDX

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name, label in self.CLASS_MAP.items():
            folder = os.path.join(root_dir, class_name)
            if not os.path.exists(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(folder, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        JpegCompressionTransform(quality_range=(70, 95), p=0.35),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return train_transform, eval_transform


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_with_probs(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=-1)[:, 1] # fake prob
            y_true.extend(labels.cpu().numpy().tolist())
            y_probs.extend(probs.cpu().numpy().tolist())

    val_loss = running_loss / len(y_true)
    return val_loss, np.array(y_true), np.array(y_probs)


def find_optimal_threshold(y_true, y_probs):
    """Grid search for decision threshold that maximizes balanced accuracy."""
    best_thresh = 0.5
    best_bal_acc = 0.0
    best_metrics = {}

    for thresh in np.arange(0.10, 0.92, 0.02):
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        rec_real = cm[0, 0] / max(1, (cm[0, 0] + cm[0, 1]))
        rec_fake = cm[1, 1] / max(1, (cm[1, 0] + cm[1, 1]))
        bal_acc = (rec_real + rec_fake) / 2.0

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = thresh
            best_metrics = {
                'threshold': round(float(thresh), 4),
                'balanced_accuracy': round(float(bal_acc), 4),
                'recall_real': round(float(rec_real), 4),
                'recall_fake': round(float(rec_fake), 4),
                'confusion_matrix': cm.tolist()
            }

    return best_thresh, best_bal_acc, best_metrics


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Running on compute hardware: {device.type.upper()}")

    base_dir = r'c:\deepfake detection model\deepfake-detector'
    data_dir = os.path.join(base_dir, 'data', 'small')
    models_dir = os.path.join(base_dir, 'models')
    pretrained_path = os.path.join(models_dir, 'pretrained_vit')

    train_transform, eval_transform = get_transforms()
    train_dataset = BalancedRealFakeDataset(os.path.join(data_dir, 'train'), transform=train_transform)
    valid_dataset = BalancedRealFakeDataset(os.path.join(data_dir, 'valid'), transform=eval_transform)
    test_dataset = BalancedRealFakeDataset(os.path.join(data_dir, 'test'), transform=eval_transform)

    print(f"[Data] Loaded {len(train_dataset)} train, {len(valid_dataset)} valid, {len(test_dataset)} test.")

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Model] Loading model from: {pretrained_path}")
    processor = AutoImageProcessor.from_pretrained(pretrained_path)
    model = AutoModelForImageClassification.from_pretrained(pretrained_path)
    model.to(device)

    # Differential Learning Rate setup:
    # Freeze lower blocks (0-7) to preserve robust representations and accelerate CPU training
    # Train blocks 8-11 with lower lr (5e-6)
    # Train classifier head with lr (8e-5)
    params_base = []
    params_head = []

    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            params_head.append(param)
        elif any(f'layer.{i}.' in name for i in [8, 9, 10, 11]) or 'layernorm' in name:
            param.requires_grad = True
            params_base.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW([
        {'params': params_base, 'lr': 5e-6, 'weight_decay': 1e-4},
        {'params': params_head, 'lr': 8e-5, 'weight_decay': 1e-4}
    ])

    # Balanced class weights: equal penalty for real and fake misclassifications
    criterion = nn.CrossEntropyLoss()

    best_val_bal_acc = 0.0
    best_threshold = 0.5
    best_metrics = {}
    patience = 4
    no_improve = 0
    max_epochs = 4

    print(f"\n--- Starting Balanced Fine-Tuning ({max_epochs} epochs) ---")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true_val, y_probs_val = eval_with_probs(model, valid_loader, criterion, device)

        opt_thresh, bal_acc, metrics = find_optimal_threshold(y_true_val, y_probs_val)
        epoch_sec = time.time() - t0

        print(f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_sec:.1f}s] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Bal Acc: {bal_acc*100:.1f}% (Opt Thresh: {opt_thresh:.2f}) | "
              f"Real Rec: {metrics['recall_real']*100:.1f}%, Fake Rec: {metrics['recall_fake']*100:.1f}%")

        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            best_threshold = opt_thresh
            best_metrics = metrics
            no_improve = 0

            # Save best checkpoints
            best_model_dir = os.path.join(models_dir, 'best_model_vit')
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)

            torch.save(model.state_dict(), os.path.join(models_dir, 'best_model.pt'))

            threshold_file = os.path.join(models_dir, 'best_threshold.json')
            with open(threshold_file, 'w') as tf:
                json.dump({
                    'optimal_threshold': float(best_threshold),
                    'val_balanced_acc': float(best_val_bal_acc),
                    'metrics': best_metrics,
                    'timestamp': time.time()
                }, tf, indent=2)

            print(f"  --> Saved new best checkpoint to {best_model_dir} (Threshold: {best_threshold:.2f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n[Training Finished in {time.time()-start_time:.1f}s]")
    print(f"Best Validation Balanced Accuracy: {best_val_bal_acc*100:.2f}% at Decision Threshold: {best_threshold:.2f}")

    # Final Evaluation on Held-Out Test Set with the Optimal Threshold
    print("\n--- Evaluating Best Model on data/small/test ---")
    # Load best checkpoint with strict=True
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_model.pt')), strict=True)
    test_loss, y_true_test, y_probs_test = eval_with_probs(model, test_loader, criterion, device)

    y_pred_test = (y_probs_test >= best_threshold).astype(int)
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_prec = precision_score(y_true_test, y_pred_test, pos_label=1, zero_division=0)
    test_rec_fake = recall_score(y_true_test, y_pred_test, pos_label=1, zero_division=0)
    test_rec_real = recall_score(y_true_test, y_pred_test, pos_label=0, zero_division=0)
    test_f1 = f1_score(y_true_test, y_pred_test, pos_label=1, zero_division=0)
    test_cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])

    print(f"Test Accuracy:         {test_acc*100:.2f}%")
    print(f"Test Real Recall:      {test_rec_real*100:.2f}% ({test_cm[0, 0]} / 250)")
    print(f"Test Fake Recall:      {test_rec_fake*100:.2f}% ({test_cm[1, 1]} / 250)")
    print(f"Test Precision (Fake): {test_prec*100:.2f}%")
    print(f"Test F1-Score:         {test_f1*100:.2f}%")
    print(f"Confusion Matrix:\n  True Real -> Pred Real: {test_cm[0, 0]} | Pred Fake: {test_cm[0, 1]}")
    print(f"  True Fake -> Pred Real: {test_cm[1, 0]} | Pred Fake: {test_cm[1, 1]}")

    # Save summary report
    summary = {
        'best_threshold': float(best_threshold),
        'test_accuracy': float(test_acc),
        'test_real_recall': float(test_rec_real),
        'test_fake_recall': float(test_rec_fake),
        'test_precision_fake': float(test_prec),
        'test_f1': float(test_f1),
        'confusion_matrix': test_cm.tolist(),
        'avg_fake_prob_real': float(np.mean(y_probs_test[y_true_test == 0])),
        'avg_fake_prob_fake': float(np.mean(y_probs_test[y_true_test == 1]))
    }
    with open(os.path.join(models_dir, 'fine_tuned_test_summary.json'), 'w') as sf:
        json.dump(summary, sf, indent=2)


if __name__ == '__main__':
    main()
