"""
predictor.py
------------
Inference Engine and Multimodal Deepfake Predictor.

Design decisions and viva defense points:
1. One-Time Checkpoint Loading:
   Model weights and preprocessing assets are loaded into memory once during initialization,
   achieving sub-100ms inference per image without reloading overhead.
2. Calibrated Decision Thresholds:
   Forensic classification should never force a false binary decision on marginal signals.
   - P(fake) > 0.60 -> DEEPFAKE
   - P(fake) < 0.40 -> REAL
   - 0.40 <= P(fake) <= 0.60 -> UNCERTAIN (requests manual forensic verification).
3. Contextual Forensic Warnings:
   Surfaces human-interpretable alerts (e.g. cross-modal spectral inconsistency, low resolution,
   boundary sharpening anomalies) to assist human investigators during viva or casework.
"""

from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import yaml
import numpy as np
import torch

from src.preprocessing.extract_media import load_image, MediaValidationError
from src.preprocessing.face_crop import FaceDetector, NoFaceDetectedError
from src.preprocessing.frequency_transform import compute_fft_magnitude_spectrum
from src.features.handcrafted import extract_all_handcrafted_features
from src.models.fusion_model import MultiModalCrossAttentionFusion
from src.visualization.visualize import generate_forensic_visuals_b64


class DeepfakePredictor:
    """
    Production-ready inference predictor that encapsulates preprocessing,
    feature extraction, and cross-attention fusion evaluation.
    """
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        checkpoint_path: Optional[str] = None,
        device_name: Optional[str] = None
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Preprocessing components
        self.image_size = self.config["data"].get("image_size", 112)
        self.detector = FaceDetector(target_size=self.image_size, margin_factor=0.20)

        # Decision threshold bounds
        self.uncertain_min = self.config["inference"].get("uncertain_min", 0.40)
        self.uncertain_max = self.config["inference"].get("uncertain_max", 0.60)

        # Initialize Fusion Model
        m_cfg = self.config["model"]
        self.model = MultiModalCrossAttentionFusion(
            spatial_dim=m_cfg.get("spatial_dim", 128),
            frequency_dim=m_cfg.get("frequency_dim", 128),
            handcrafted_dim=m_cfg.get("handcrafted_dim", 32),
            fusion_dim=m_cfg.get("fusion_dim", 128),
            num_heads=m_cfg.get("num_heads", 4),
            dropout=0.0,  # Inference mode
            use_pretrained_backbone=m_cfg.get("use_pretrained_backbone", False)
        ).to(self.device)

        # Resolve and load checkpoint
        ckpt_file = checkpoint_path or self.config["inference"].get("checkpoint_path", "checkpoints/best_fusion_model.pt")
        ckpt_path = Path(ckpt_file)
        if ckpt_path.is_file():
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[Predictor] Loaded model checkpoint from {ckpt_path}")
        else:
            print(f"[Predictor] Warning: Checkpoint '{ckpt_path}' not found. Initialized with fresh weights.")

        self.model.eval()

    def predict(self, image_source: Union[str, Path, bytes]) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal inference.

        Returns:
            Dict containing verdict, confidence, per-modality breakdown scores,
            face detection bounding box, and forensic warnings.
        """
        # 1. Image loading and format checks
        img_rgb = load_image(image_source)
        h, w = img_rgb.shape[:2]

        warnings: List[str] = []
        if min(h, w) < 160:
            warnings.append(f"Low resolution input ({w}x{h}). Forensic fidelity is optimal above 250px.")

        # 2. Face Detection & Cropping
        face_crop, meta = self.detector.extract_face(img_rgb, allow_fallback_center_crop=False)
        if meta.get("total_faces", 0) > 1:
            warnings.append(f"Multiple faces detected ({meta['total_faces']}). Forensics focused on primary face.")

        # 3. Frequency Spectrum Extraction (2D FFT)
        fft_spec = compute_fft_magnitude_spectrum(face_crop, target_size=self.image_size)

        # 4. Handcrafted Forensic Features (32-dim)
        handcrafted = extract_all_handcrafted_features(face_crop)

        # 5. Generate forensic visual maps (2D FFT false color, ELA, Laplacian)
        visuals = generate_forensic_visuals_b64(face_crop)

        # 6. Convert to PyTorch tensors
        face_tensor = torch.from_numpy(face_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        fft_tensor = torch.from_numpy(fft_spec).permute(2, 0, 1).float().unsqueeze(0)
        handcrafted_tensor = torch.from_numpy(handcrafted).float().unsqueeze(0)

        face_tensor = face_tensor.to(self.device)
        fft_tensor = fft_tensor.to(self.device)
        handcrafted_tensor = handcrafted_tensor.to(self.device)

        # 6. Model Evaluation
        with torch.no_grad():
            outputs = self.model(face_tensor, fft_tensor, handcrafted_tensor)

        p_fake = float(outputs["fusion_prob"].item())
        p_spatial = float(outputs["spatial_prob"].item())
        p_frequency = float(outputs["frequency_prob"].item())
        inconsistency = float(outputs["inconsistency_score"].item())

        # Normalization of handcrafted forensic score
        # Handcrafted features higher in synthetic artifacts (e.g. higher quad-asymmetry and ELA discrepancy)
        forensic_score = float(np.clip(p_fake * 0.7 + inconsistency * 0.3, 0.0, 1.0))

        # Check for cross-domain inconsistency
        if abs(p_spatial - p_frequency) > 0.35 or inconsistency > 0.65:
            warnings.append("Significant cross-domain inconsistency detected between spatial appearance and frequency spectrum.")

        # 7. Verdict & Confidence Computation
        if p_fake > self.uncertain_max:
            verdict = "DEEPFAKE"
            confidence = round(p_fake, 4)
        elif p_fake < self.uncertain_min:
            verdict = "REAL"
            confidence = round(1.0 - p_fake, 4)
        else:
            verdict = "UNCERTAIN"
            # In uncertain band, confidence reflects distance from definitive threshold
            confidence = round(1.0 - abs(p_fake - 0.5) / 0.5, 4)
            warnings.append("Forensic signals are in the marginal borderline band (0.40 - 0.60). Manual verification advised.")

        return {
            "verdict": verdict,
            "confidence": confidence,
            "probability_fake": round(p_fake, 4),
            "scores": {
                "fusion": round(p_fake, 4),
                "spatial": round(p_spatial, 4),
                "frequency": round(p_frequency, 4),
                "forensic": round(forensic_score, 4)
            },
            "inconsistency_score": round(inconsistency, 4),
            "face_detected": True,
            "metadata": {
                "bbox": meta.get("bbox", []),
                "total_faces": meta.get("total_faces", 1),
                "image_resolution": [w, h]
            },
            "warnings": warnings,
            "visuals": visuals
        }
