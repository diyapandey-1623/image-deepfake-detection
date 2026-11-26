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

from pathlib import Path
from typing import Dict, Any
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

from src.features.blocks import make_conv_backbone


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


class _UnimodalCNN(nn.Module):
    """Shared 5-block CNN baseline for spatial RGB or 2D FFT inputs."""

    def __init__(self, in_channels: int, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = make_conv_backbone(in_channels, embed_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_embedding(x)).squeeze(-1)


class SpatialOnlyCNN(_UnimodalCNN):
    """Unimodal baseline classifying images from spatial RGB face appearance."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__(in_channels=in_channels, embed_dim=embed_dim, dropout=dropout)


class FrequencyOnlyCNN(_UnimodalCNN):
    """Unimodal baseline classifying images from 2D Fourier power spectrum."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__(in_channels=in_channels, embed_dim=embed_dim, dropout=dropout)
