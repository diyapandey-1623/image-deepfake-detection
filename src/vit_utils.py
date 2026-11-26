"""
src/vit_utils.py
----------------
Shared dataset, model resolution, and confusion-matrix plotting utilities
for Vision Transformer training, evaluation, and inference scripts.
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

from src.class_config import CLASS_TO_IDX


DEFAULT_HF_MODEL_ID = "vishvjeet1623/image-deepfake-detection"
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def resolve_vit_model_source(
    model_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> str:
    """Resolves an explicit path/HF model ID, a local checkpoint directory, or the default HF ID."""
    if model_path:
        return model_path

    root = project_root or Path(__file__).resolve().parent.parent
    candidates = [
        root / "models" / "best_model_vit",
        root / "models" / "pretrained_vit",
        root / "models" / "best_model",
        root / "models",
        root / "pretrained_model",
    ]
    for cand in candidates:
        if cand.is_dir() and (cand / "model.safetensors").is_file():
            return str(cand)

    return DEFAULT_HF_MODEL_ID


class RealFakeDataset(Dataset):
    """Directory-backed dataset mapping real/ -> 0 and fake/ -> 1."""

    CLASS_MAP = CLASS_TO_IDX

    def __init__(self, root_dir: str, transform=None, return_meta: bool = False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_meta = return_meta
        self.samples = []

        for class_name, label in self.CLASS_MAP.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for fname in sorted(os.listdir(class_folder)):
                if fname.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(class_folder, fname), label, fname))

        if not self.samples:
            raise RuntimeError(f"No valid images found in {root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, fname = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_meta:
            return image, label, path, fname
        return image, label


def plot_confusion_matrix(cm, class_names, output_path: str, title: str = "Confusion Matrix") -> None:
    """Renders and saves a confusion matrix heatmap PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", size="xx-large", weight="bold")

    fig.colorbar(cax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10)
    plt.ylabel("True Label", fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
