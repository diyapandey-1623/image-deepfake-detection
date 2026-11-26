r"""
train.py
--------
Training pipeline for Multimodal Cross-Attention Deepfake Detector.

Design decisions and viva defense points:
1. Multi-Task Objective Loss:
   $L_{total} = L_{fusion} + \lambda_{aux} \cdot (L_{spatial} + L_{frequency})$
   Supervising auxiliary unimodal heads alongside the cross-attention fusion head guarantees
   that individual encoders preserve domain-specific forensic features rather than degenerating.
2. Positive Class Weighting:
   $w_{pos} = N_{real} / N_{fake}$ dynamically balances gradient magnitudes in imbalanced datasets.
3. Checkpoint Tracking:
   Models are checkpointed strictly according to Validation ROC-AUC (not accuracy), ensuring
   optimal discrimination thresholding on unseen test data.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.fusion_model import MultiModalCrossAttentionFusion
from src.models.baselines import compute_classification_metrics
from src.training.dataset import create_dataloaders


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates model on a dataloader, returning full forensic metrics.
    """
    model.eval()
    all_targets = []
    all_fusion_probs = []
    all_spatial_probs = []
    all_freq_probs = []

    with torch.no_grad():
        for batch in loader:
            faces = batch["face_crop"].to(device)
            ffts = batch["fft_spectrum"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs = model(faces, ffts, handcrafted)
            all_targets.extend(labels)
            all_fusion_probs.extend(outputs["fusion_prob"].cpu().numpy())
            all_spatial_probs.extend(outputs["spatial_prob"].cpu().numpy())
            all_freq_probs.extend(outputs["frequency_prob"].cpu().numpy())

    y_true = np.array(all_targets)
    y_prob = np.array(all_fusion_probs)
    
    metrics = compute_classification_metrics(y_true, y_prob)
    metrics["spatial_metrics"] = compute_classification_metrics(y_true, np.array(all_spatial_probs))
    metrics["frequency_metrics"] = compute_classification_metrics(y_true, np.array(all_freq_probs))
    return metrics


def train_detector(
    config_path: str = "configs/default.yaml",
    epochs: Optional[int] = None,
    subset: Optional[int] = None,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    set_seed(seed)

    # Determine compute device
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Create DataLoaders
    train_loader, val_loader, test_loader, pos_weight = create_dataloaders(
        cache_dir=config["data"].get("cache_dir", "data/cache"),
        batch_size=config["data"].get("batch_size", 16),
        val_split=config["data"].get("val_split", 0.15),
        test_split=config["data"].get("test_split", 0.15),
        seed=seed,
        num_workers=config["data"].get("num_workers", 0)
    )

    print(f"Dataset split sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"Computed positive class weight (handling imbalance): {pos_weight:.3f}")

    # Instantiate Fusion Model
    m_cfg = config["model"]
    model = MultiModalCrossAttentionFusion(
        spatial_dim=m_cfg.get("spatial_dim", 128),
        frequency_dim=m_cfg.get("frequency_dim", 128),
        handcrafted_dim=m_cfg.get("handcrafted_dim", 32),
        fusion_dim=m_cfg.get("fusion_dim", 128),
        num_heads=m_cfg.get("num_heads", 4),
        dropout=m_cfg.get("dropout", 0.35),
        use_pretrained_backbone=m_cfg.get("use_pretrained_backbone", False)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params:,} trainable parameters.")

    # Optimizer & Scheduler
    t_cfg = config["training"]
    lr = t_cfg.get("learning_rate", 0.0005)
    weight_decay = t_cfg.get("weight_decay", 0.0001)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # Loss function with positive class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    aux_weight = t_cfg.get("aux_loss_weight", 0.3)

    num_epochs = epochs if epochs is not None else t_cfg.get("epochs", 25)
    patience = t_cfg.get("early_stopping_patience", 6)

    best_val_auc = -1.0
    patience_counter = 0
    save_dir = Path(t_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = save_dir / "best_fusion_model.pt"

    history = []

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            faces = batch["face_crop"].to(device)
            ffts = batch["fft_spectrum"].to(device)
            handcrafted = batch["handcrafted"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(faces, ffts, handcrafted)

            # Combined multi-task objective
            l_fusion = criterion(outputs["fusion_logit"], labels)
            l_spatial = criterion(outputs["spatial_logit"], labels)
            l_freq = criterion(outputs["frequency_logit"], labels)
            loss = l_fusion + aux_weight * (l_spatial + l_freq)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * faces.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device)
        val_auc = val_metrics["roc_auc"]
        val_f1 = val_metrics["f1"]
        scheduler.step(val_auc)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_auc": round(val_auc, 4),
            "val_f1": round(val_f1, 4),
            "val_acc": round(val_metrics["accuracy"], 4)
        }
        history.append(epoch_record)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")

        # Checkpoint if best validation ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_metrics": val_metrics,
                "config": config
            }, best_checkpoint_path)
            print(f"  [Checkpoint] New best model saved to {best_checkpoint_path} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Early Stopping] Triggered after {epoch} epochs (patience={patience}).")
                break

    # Load best checkpoint and evaluate on Test set
    print(f"\nLoading best checkpoint for final evaluation on Test set...")
    checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device)

    print("\n================ FINAL TEST RESULTS ================")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1 Score:  {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")
    print("====================================================")

    # Save results to json
    results_dir = Path(config["training"].get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "training_metrics.json"
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "test_metrics": test_metrics,
            "best_val_auc": best_val_auc,
            "checkpoint_path": str(best_checkpoint_path.as_posix())
        }, f, indent=2)

    print(f"Saved complete metrics log to {results_file}")
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal Deepfake Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    train_detector(config_path=args.config, epochs=args.epochs, device_name=args.device)
