"""Fine-tuning script for Deepfake Detection (Real vs Fake Faces)
Model: dima806/deepfake_vs_real_image_detection (ViTForImageClassification)
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RealFakeDataset(Dataset):
    """Custom dataset mapping 'real' -> 0 and 'fake' -> 1 to match dima806 config."""
    CLASS_MAP = {'real': 0, 'fake': 1}

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
                    self.samples.append((os.path.join(class_folder, fname), label))

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
    # ViT expects 224x224, normalized to [-1, 1] with mean=0.5, std=0.5
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return train_transform, eval_transform


def train_one_epoch(model, loader, optimizer, criterion, device):
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
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_metrics(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Valid Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'b-o', label='Train Accuracy (%)', linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'r-s', label='Valid Accuracy (%)', linewidth=2)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT model for Real vs Fake image detection.")
    parser.add_argument('--data_dir', type=str, default='data_small', help="Path to data directory containing train/ and valid/")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save checkpoints and plots")
    parser.add_argument('--model_id', type=str, default='dima806/deepfake_vs_real_image_detection', help="Hugging Face model ID or local directory")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size (auto-configured if not specified)")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # Auto-detect compute hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using compute device: {device.type.upper()}")

    # Determine batch size: CPU benefits from smaller batch size (8 or 16)
    if args.batch_size is None:
        batch_size = 32 if device.type == 'cuda' else 8
    else:
        batch_size = args.batch_size
    print(f"[Config] Batch size: {batch_size} | Epochs: {args.epochs} | Learning rate: {args.lr}")

    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')

    train_transform, eval_transform = get_transforms()
    train_dataset = RealFakeDataset(train_dir, transform=train_transform)
    valid_dataset = RealFakeDataset(valid_dir, transform=eval_transform)

    print(f"[Data] Training samples: {len(train_dataset)} | Validation samples: {len(valid_dataset)}")

    # On Windows, num_workers=0 avoids multiprocessing spawn deadlocks
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Check local pretrained_model folder fallback first
    local_pretrained = os.path.join(os.getcwd(), 'pretrained_model')
    model_source = local_pretrained if (os.path.exists(local_pretrained) and os.path.exists(os.path.join(local_pretrained, 'model.safetensors'))) else args.model_id

    print(f"[Model] Loading ViT model from: {model_source}")
    processor = AutoImageProcessor.from_pretrained(model_source)
    model = AutoModelForImageClassification.from_pretrained(model_source)
    model.to(device)

    # Ensure label mapping is consistent
    model.config.id2label = {0: "Real", 1: "Fake"}
    model.config.label2id = {"Real": 0, "Fake": 1}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -1.0
    best_checkpoint_dir = os.path.join(args.output_dir, 'best_model')
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n--- Starting Fine-Tuning ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_time:.1f}s] - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  --> Best model updated (Val Acc: {val_acc*100:.2f}%) -> Saving to {best_checkpoint_dir}")
            model.save_pretrained(best_checkpoint_dir)
            processor.save_pretrained(best_checkpoint_dir)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, os.path.join(args.output_dir, 'best_checkpoint.pt'))

    total_time = time.time() - start_time
    print(f"\n[Finished] Training completed in {total_time:.1f}s. Best Val Accuracy: {best_val_acc*100:.2f}%")

    plot_metrics(history, args.output_dir)


if __name__ == '__main__':
    main()
