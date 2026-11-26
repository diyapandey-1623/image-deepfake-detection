r"""
evaluate.py
-----------
Comprehensive Evaluation, Ablation Study, and Robustness Stress Testing Suite.

Design decisions and viva defense points:
1. Rigorous Multi-Metric Evaluation:
   Accuracy in isolation is severely misleading in forensic contexts where class distributions
   are skewed or asymmetric costs exist between False Positives (falsely accusing authentic imagery)
   and False Negatives (missing malicious deepfakes). We compute ROC-AUC, Precision, Recall, and F1.
2. Ablation Study:
   Ablation isolates the distinct contribution of each architectural component:
   - Handcrafted-only (RF/SVM on 32-dim forensic features)
   - Spatial-only (RGB face CNN)
   - Frequency-only (2D Fourier spectrum CNN)
   - Full Multimodal Cross-Attention Fusion
   This provides empirical proof of whether cross-domain fusion adds value over unimodal signals.
3. Robustness Stress Testing:
   Real-world deepfakes circulated on social media platforms undergo post-processing
   (aggressive JPEG recompression, messaging app downsizing, transmission noise).
   Re-evaluating under controlled perturbations reveals real-world generalization limits.
"""

import os
import io
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.fusion_model import MultiModalCrossAttentionFusion
from src.models.baselines import (
    ClassicalForensicBaseline,
    SpatialOnlyCNN,
    FrequencyOnlyCNN,
    compute_classification_metrics
)
from src.training.dataset import load_cached_splits, DeepfakeCachedDataset


def run_ablation_study(
    config: Dict[str, Any],
    cache_dir: str = "data/cache",
    device_name: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes 4-way ablation comparison:
    1. Handcrafted-Only (Random Forest)
    2. Spatial-Only CNN
    3. Frequency-Only CNN
    4. Multimodal Fusion Model
    """
    device = torch.device(device_name)
    train_samples, val_samples, test_samples, _ = load_cached_splits(
        cache_dir=cache_dir,
        val_split=config["data"].get("val_split", 0.15),
        test_split=config["data"].get("test_split", 0.15),
        seed=config["data"].get("seed", 42)
    )

    # Prepare datasets
    train_ds = DeepfakeCachedDataset(train_samples, cache_dir=cache_dir, is_training=False)
    test_ds = DeepfakeCachedDataset(test_samples, cache_dir=cache_dir, is_training=False)

    # 1. Handcrafted-Only Baseline
    print("\n--- [Ablation 1/4] Evaluating Handcrafted-Only Baseline (Random Forest) ---")
    X_train = np.array([train_ds[i]["handcrafted"].numpy() for i in range(len(train_ds))])
    y_train = np.array([int(train_ds[i]["label"].item()) for i in range(len(train_ds))])
    X_test = np.array([test_ds[i]["handcrafted"].numpy() for i in range(len(test_ds))])
    y_test = np.array([int(test_ds[i]["label"].item()) for i in range(len(test_ds))])

    rf_baseline = ClassicalForensicBaseline(model_type="rf", random_state=42)
    rf_baseline.fit(X_train, y_train)
    rf_metrics = rf_baseline.evaluate(X_test, y_test)

    # 2. Spatial-Only CNN Baseline
    print("--- [Ablation 2/4] Evaluating Spatial-Only CNN Baseline ---")
    spatial_model = SpatialOnlyCNN(in_channels=3, embed_dim=128).to(device)
    # Quick train on train_ds for fair comparison
    sp_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    sp_opt = torch.optim.AdamW(spatial_model.parameters(), lr=1e-3)
    spatial_model.train()
    for _ in range(5):
        for b in sp_loader:
            sp_opt.zero_grad()
            out = spatial_model(b["face_crop"].to(device))
            loss = nn.BCEWithLogitsLoss()(out, b["label"].to(device))
            loss.backward()
            sp_opt.step()
            
    spatial_model.eval()
    with torch.no_grad():
        sp_probs = [torch.sigmoid(spatial_model(test_ds[i]["face_crop"].unsqueeze(0).to(device))).item() for i in range(len(test_ds))]
    spatial_metrics = compute_classification_metrics(y_test, np.array(sp_probs))

    # 3. Frequency-Only CNN Baseline
    print("--- [Ablation 3/4] Evaluating Frequency-Only CNN Baseline ---")
    freq_model = FrequencyOnlyCNN(in_channels=1, embed_dim=128).to(device)
    fr_opt = torch.optim.AdamW(freq_model.parameters(), lr=1e-3)
    freq_model.train()
    for _ in range(5):
        for b in sp_loader:
            fr_opt.zero_grad()
            out = freq_model(b["fft_spectrum"].to(device))
            loss = nn.BCEWithLogitsLoss()(out, b["label"].to(device))
            loss.backward()
            fr_opt.step()

    freq_model.eval()
    with torch.no_grad():
        fr_probs = [torch.sigmoid(freq_model(test_ds[i]["fft_spectrum"].unsqueeze(0).to(device))).item() for i in range(len(test_ds))]
    freq_metrics = compute_classification_metrics(y_test, np.array(fr_probs))

    # 4. Multimodal Fusion Model
    print("--- [Ablation 4/4] Evaluating Full Multimodal Fusion Model ---")
    checkpoint_path = Path("checkpoints/best_fusion_model.pt")
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, weights_only=False)
        m_cfg = config["model"]
        fusion_model = MultiModalCrossAttentionFusion(
            spatial_dim=m_cfg.get("spatial_dim", 128),
            frequency_dim=m_cfg.get("frequency_dim", 128),
            handcrafted_dim=m_cfg.get("handcrafted_dim", 32),
            fusion_dim=m_cfg.get("fusion_dim", 128)
        ).to(device)
        fusion_model.load_state_dict(ckpt["model_state_dict"])
        fusion_model.eval()
        with torch.no_grad():
            fu_probs = []
            for i in range(len(test_ds)):
                out = fusion_model(
                    test_ds[i]["face_crop"].unsqueeze(0).to(device),
                    test_ds[i]["fft_spectrum"].unsqueeze(0).to(device),
                    test_ds[i]["handcrafted"].unsqueeze(0).to(device)
                )
                fu_probs.append(out["fusion_prob"].item())
        fusion_metrics = compute_classification_metrics(y_test, np.array(fu_probs))
    else:
        fusion_metrics = {"accuracy": 0.0, "roc_auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": []}

    ablation_table = {
        "Handcrafted_Only (RF)": rf_metrics,
        "Spatial_Only (CNN)": spatial_metrics,
        "Frequency_Only (CNN)": freq_metrics,
        "Multimodal_Fusion": fusion_metrics
    }

    print("\n================== ABLATION STUDY RESULTS ==================")
    header = f"{'Model Architecture':<26} | {'Accuracy':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'Precision':<9} | {'Recall':<8}"
    print(header)
    print("-" * len(header))
    for name, m in ablation_table.items():
        print(f"{name:<26} | {m['accuracy']:<8.4f} | {m['f1']:<8.4f} | {m['roc_auc']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<8.4f}")
    print("============================================================")

    return ablation_table


def apply_jpeg_compression(tensor_img: torch.Tensor, quality: int = 50) -> torch.Tensor:
    """
    Simulates real-world JPEG compression on a [3, H, W] tensor in [0, 1].
    """
    arr = (tensor_img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = np.array(Image.open(buf), dtype=np.float32) / 255.0
    return torch.from_numpy(compressed).permute(2, 0, 1).float()


def apply_gaussian_noise(tensor_img: torch.Tensor, sigma: float = 0.08) -> torch.Tensor:
    """
    Simulates additive sensor noise.
    """
    noise = torch.randn_like(tensor_img) * sigma
    return torch.clamp(tensor_img + noise, 0.0, 1.0)


def apply_gaussian_blur(tensor_img: torch.Tensor, ksize: int = 5) -> torch.Tensor:
    """
    Simulates social media transmission blur.
    """
    arr = (tensor_img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return torch.from_numpy(blurred.astype(np.float32) / 255.0).permute(2, 0, 1).float()


def run_robustness_test(
    config: Dict[str, Any],
    cache_dir: str = "data/cache",
    device_name: str = "cpu"
) -> Dict[str, Any]:
    """
    Stress-tests model against real-world perturbations:
    1. Unperturbed Baseline (clean)
    2. Moderate JPEG Compression (Quality 60)
    3. Severe JPEG Compression (Quality 35)
    4. Sensor Additive Noise
    5. Social Media Gaussian Blur
    """
    device = torch.device(device_name)
    checkpoint_path = Path("checkpoints/best_fusion_model.pt")
    if not checkpoint_path.is_file():
        print("No checkpoint found for robustness test.")
        return {}

    ckpt = torch.load(checkpoint_path, weights_only=False)
    m_cfg = config["model"]
    model = MultiModalCrossAttentionFusion(
        spatial_dim=m_cfg.get("spatial_dim", 128),
        frequency_dim=m_cfg.get("frequency_dim", 128),
        handcrafted_dim=m_cfg.get("handcrafted_dim", 32),
        fusion_dim=m_cfg.get("fusion_dim", 128)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_samples, _ = load_cached_splits(cache_dir=cache_dir, seed=config["data"].get("seed", 42))
    test_ds = DeepfakeCachedDataset(test_samples, cache_dir=cache_dir, is_training=False)
    y_test = np.array([int(test_ds[i]["label"].item()) for i in range(len(test_ds))])

    perturbations = {
        "Pristine (Clean Test Set)": lambda x: x,
        "JPEG Compression (Q=60)": lambda x: apply_jpeg_compression(x, quality=60),
        "JPEG Compression (Q=35)": lambda x: apply_jpeg_compression(x, quality=35),
        "Additive Gaussian Noise (sigma=0.08)": lambda x: apply_gaussian_noise(x, sigma=0.08),
        "Transmission Blur (k=5)": lambda x: apply_gaussian_blur(x, ksize=5)
    }

    robustness_results = {}
    print("\n================== ROBUSTNESS STRESS TEST ==================")
    header = f"{'Perturbation Condition':<36} | {'Accuracy':<8} | {'F1':<8} | {'ROC-AUC':<8}"
    print(header)
    print("-" * len(header))

    with torch.no_grad():
        for pert_name, pert_fn in perturbations.items():
            probs = []
            for i in range(len(test_ds)):
                # Apply perturbation to face crop
                perturbed_face = pert_fn(test_ds[i]["face_crop"])
                out = model(
                    perturbed_face.unsqueeze(0).to(device),
                    test_ds[i]["fft_spectrum"].unsqueeze(0).to(device),
                    test_ds[i]["handcrafted"].unsqueeze(0).to(device)
                )
                probs.append(out["fusion_prob"].item())

            metrics = compute_classification_metrics(y_test, np.array(probs))
            robustness_results[pert_name] = metrics
            print(f"{pert_name:<36} | {metrics['accuracy']:<8.4f} | {metrics['f1']:<8.4f} | {metrics['roc_auc']:<8.4f}")
    print("============================================================")

    return robustness_results


def evaluate_suite(config_path: str = "configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["training"].get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ablation Study
    ablation_results = run_ablation_study(config)
    with open(results_dir / "ablation_study.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # 2. Robustness Stress Testing
    robustness_results = run_robustness_test(config)
    with open(results_dir / "robustness_results.json", "w", encoding="utf-8") as f:
        json.dump(robustness_results, f, indent=2)

    print(f"\n[Done] All evaluation artifacts saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation and Robustness Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    args = parser.parse_args()

    evaluate_suite(args.config)
