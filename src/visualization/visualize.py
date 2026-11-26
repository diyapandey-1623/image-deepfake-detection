"""
visualize.py
------------
Forensic Visual Analytics: 2D Fourier Spectrum Inspector, ELA Heatmap,
ROC Curve, and Confusion Matrix Generators.

Design decisions and viva defense points:
1. Visual Explainability:
   Forensic examiners and viva committees need visual evidence, not just opaque probability scores.
   - Spectral Heatmap: False-color representation (Inferno colormap) of 2D FFT magnitude spectrum
     visually exposes the high-frequency periodic grid spikes characteristic of generative upsampling.
   - Error Level Analysis (ELA): Amplified compression deltas visually reveal spliced boundaries.
   - Laplacian Sharpness Map: Visualizes over-smoothed skin textures vs unnaturally sharp boundaries.
2. Direct Base64 Web Delivery:
   Images are compressed to compact PNG base64 strings in-memory for instant visual inspection in the React UI.
"""

import io
import os
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np

# Set writable temporary directory for matplotlib configuration on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.preprocessing.frequency_transform import compute_fft_magnitude_spectrum, compute_ela_map, compute_laplacian_sharpness_map


def encode_image_to_base64_png(img_bgr_or_rgb: np.ndarray, is_bgr: bool = True) -> str:
    """
    Encodes uint8 image array to data:image/png;base64 string.
    """
    img_bgr = img_bgr_or_rgb if is_bgr else cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def generate_forensic_visuals_b64(face_crop_rgb: np.ndarray) -> Dict[str, str]:
    """
    Generates base64 data URLs for:
    1. Face Crop (standardized 112x112)
    2. 2D FFT Magnitude Spectrum (color-mapped with Inferno)
    3. Error Level Analysis (ELA) map
    4. Laplacian gradient map (color-mapped with Viridis)
    """
    # 1. Face Crop
    face_b64 = encode_image_to_base64_png(face_crop_rgb, is_bgr=False)

    # 2. 2D FFT Spectrum [112, 112, 1] float32 in [0, 1]
    fft_spec = compute_fft_magnitude_spectrum(face_crop_rgb, target_size=112)
    fft_uint8 = (fft_spec.squeeze(-1) * 255.0).astype(np.uint8)
    fft_colored = cv2.applyColorMap(fft_uint8, cv2.COLORMAP_INFERNO)
    fft_b64 = encode_image_to_base64_png(fft_colored, is_bgr=True)

    # 3. ELA Map
    ela = compute_ela_map(face_crop_rgb, quality=90, scale=20.0)
    ela_uint8 = (np.clip(ela, 0.0, 1.0) * 255.0).astype(np.uint8)
    ela_b64 = encode_image_to_base64_png(ela_uint8, is_bgr=False)

    # 4. Laplacian Sharpness Map
    lap = compute_laplacian_sharpness_map(face_crop_rgb)
    lap_uint8 = (np.clip(lap, 0.0, 1.0) * 255.0).astype(np.uint8)
    lap_colored = cv2.applyColorMap(lap_uint8, cv2.COLORMAP_VIRIDIS)
    lap_b64 = encode_image_to_base64_png(lap_colored, is_bgr=True)

    return {
        "face_crop": face_b64,
        "fft_spectrum": fft_b64,
        "ela_map": ela_b64,
        "laplacian_map": lap_b64
    }


def save_forensic_panel(
    face_crop_rgb: np.ndarray,
    output_path: str = "results/forensic_inspection.png",
    title_suffix: str = ""
):
    """
    Generates a publication-quality 4-panel forensic inspection figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fft_spec = compute_fft_magnitude_spectrum(face_crop_rgb, target_size=112).squeeze(-1)
    ela_map = compute_ela_map(face_crop_rgb, quality=90, scale=20.0)
    lap_map = compute_laplacian_sharpness_map(face_crop_rgb)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=150)
    
    # 1. Face Crop
    axes[0].imshow(face_crop_rgb)
    axes[0].set_title("1. Facial Crop (RGB)\n112x112 ROI", fontsize=11, fontweight="bold", color="#38bdf8")
    axes[0].axis("off")

    # 2. 2D Fourier Spectrum
    im1 = axes[1].imshow(fft_spec, cmap="inferno")
    axes[1].set_title("2. 2D FFT Spectrum\nPeriodic Grid / Decay", fontsize=11, fontweight="bold", color="#f43f5e")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. ELA Map
    axes[2].imshow(ela_map)
    axes[2].set_title("3. Error Level Analysis (ELA)\nJPEG Q=90 Compression Delta", fontsize=11, fontweight="bold", color="#a855f7")
    axes[2].axis("off")

    # 4. Laplacian Gradient
    im3 = axes[3].imshow(lap_map, cmap="viridis")
    axes[3].set_title("4. Laplacian Sharpness\nTexture Boundary Variance", fontsize=11, fontweight="bold", color="#10b981")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.suptitle(f"Multimodal Forensic Inspection Report {title_suffix}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="#0b0f17")
    plt.close()
    print(f"[Done] Forensic report panel saved to {output_path}")


def plot_roc_curve_and_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str = "results"
):
    """
    Plots dark-mode ROC Curve and Confusion Matrix figures.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.plot(fpr, tpr, color="#38bdf8", lw=2.5, label=f"Multimodal Fusion (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#64748b", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=10, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.8)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.png", facecolor="#0b0f17")
    plt.close()

    # 2. Confusion Matrix
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix (Decision Threshold = 0.50)", fontsize=11, fontweight="bold", pad=12)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Real (0)", "Fake (1)"], fontsize=10)
    ax.set_yticklabels(["Real (0)", "Fake (1)"], fontsize=10)
    ax.set_ylabel("True Ground Truth", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")

    # Print values inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "#94a3b8",
                    fontweight="bold", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", facecolor="#0b0f17")
    plt.close()

    print(f"[Done] ROC curve and confusion matrix saved to {out_path}")
