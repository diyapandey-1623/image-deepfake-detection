"""
baselines.py
------------
Baseline models for ablation and comparative evaluation:
1. Classical Machine Learning Baseline: Random Forest & SVM on Handcrafted Forensic Features
2. Unimodal Deep Learning Baselines:
   - Spatial-Only CNN (face appearance only)
   - Frequency-Only CNN (2D Fourier spectrum only)

Design decisions and viva defense points:
To prove scientifically that cross-domain multimodal fusion provides a statistically
significant advantage, we must evaluate against strong unimodal baselines under identical
data splits. Reporting fusion results without baselines is scientifically incomplete.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import torch
import torch.nn as nn


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes complete evaluation metrics. Never reports accuracy alone.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle edge case where only one class is present in mini-eval
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": cm  # [[TN, FP], [FN, TP]]
    }


class ClassicalForensicBaseline:
    """
    Classical Machine Learning Baseline operating on 32-dim handcrafted forensic features.
    Supports Random Forest and Support Vector Machine (SVM) with Platt scaling probability calibration.
    """
    def __init__(self, model_type: str = "rf", random_state: int = 42):
        self.model_type = model_type.lower()
        self.scaler = StandardScaler()
        self.random_state = random_state

        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=3,
                class_weight="balanced",
                random_state=random_state
            )
        elif self.model_type == "svm":
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self.model = SVC(
                    kernel="rbf",
                    C=1.0,
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state
                )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Expected 'rf' or 'svm'.")

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the scaler and classical classifier on training features."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability of DEEPFAKE (class 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_scaled = self.scaler.transform(X)
        # Return probability of positive class (label 1: deepfake)
        return self.model.predict_proba(X_scaled)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluates model on test features and returns comprehensive forensic metrics."""
        probs = self.predict_proba(X)
        return compute_classification_metrics(y, probs)

    def save(self, file_path: str):
        """Saves model and scaler bundle."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "type": self.model_type}, file_path)

    @classmethod
    def load(cls, file_path: str):
        """Loads fitted model and scaler bundle."""
        data = joblib.load(file_path)
        instance = cls(model_type=data["type"])
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.is_fitted = True
        return instance


class ConvBlock(nn.Module):
    """
    Standard Convolutional building block trained strictly from scratch:
    Conv2d -> BatchNorm2d -> LeakyReLU(0.1) -> MaxPool2d
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SpatialOnlyCNN(nn.Module):
    """
    Unimodal baseline: Classifies images using spatial RGB face appearance alone.
    Architecture: 5-block CNN from scratch with global average pooling -> binary logit.
    Total parameters: ~450k (compact, fast training, no external weights).
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),    # 112 -> 56
            ConvBlock(32, 64, pool=True),             # 56 -> 28
            ConvBlock(64, 128, pool=True),            # 28 -> 14
            ConvBlock(128, 256, pool=True),           # 14 -> 7
            ConvBlock(256, embed_dim, pool=False)     # 7 -> 7
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)  # Binary logit
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        embed = self.global_pool(feat_map).flatten(1)
        return embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.extract_embedding(x)
        return self.classifier(embed).squeeze(-1)


class FrequencyOnlyCNN(nn.Module):
    """
    Unimodal baseline: Classifies images using 2D Fourier power spectrum alone.
    Architecture: 5-block CNN from scratch on 1-channel log-magnitude spectrum.
    Total parameters: ~450k.
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
            ConvBlock(256, embed_dim, pool=False)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        embed = self.global_pool(feat_map).flatten(1)
        return embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.extract_embedding(x)
        return self.classifier(embed).squeeze(-1)
