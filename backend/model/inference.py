"""
backend/model/inference.py
--------------------------
Model-agnostic inference engine for deepfake detection.
Supports Hugging Face vision transformers and custom PyTorch CNNs,
with Grad-CAM explainability and anatomical facial region analysis.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from backend.config import (
    CUSTOM_ARCHITECTURE,
    CUSTOM_MODEL_PATH,
    DEVICE,
    MODEL_BACKEND,
    MODEL_ID,
)
from src.class_config import (
    FAKE_CLASS_IDX,
    REAL_CLASS_IDX,
    build_forensic_rationale,
    get_verdict,
    resolve_fake_label_id,
)
from src.vit_transforms import get_vit_transform


def analyze_cam_regions(
    grayscale_cam: np.ndarray,
    verdict: str = "real",
    fake_prob: float = 0.5,
) -> Tuple[List[Dict[str, Any]], str]:
    """Segments a 2D Grad-CAM map into 5 facial regions with Red/Yellow/Blue influence tiers."""
    h, w = grayscale_cam.shape
    zones = [
        {
            "region": "Forehead & Hairline",
            "slice": grayscale_cam[0:int(h * 0.25), :],
            "desc": "Upper facial boundary and hairline blend integrity.",
        },
        {
            "region": "Eyes & Periorbital Region",
            "slice": grayscale_cam[int(h * 0.22):int(h * 0.48), int(w * 0.12):int(w * 0.88)],
            "desc": "Iris symmetry, corneal reflection consistency, and eyelid margins.",
        },
        {
            "region": "Nose & Central Midface",
            "slice": grayscale_cam[int(h * 0.42):int(h * 0.68), int(w * 0.30):int(w * 0.70)],
            "desc": "Nasal ridge diffusion and pore texture consistency.",
        },
        {
            "region": "Mouth & Lips",
            "slice": grayscale_cam[int(h * 0.65):int(h * 0.85), int(w * 0.25):int(w * 0.75)],
            "desc": "Vermilion border, tooth alignment, and lip texture fidelity.",
        },
        {
            "region": "Cheeks & Jawline",
            "slice": np.concatenate([
                grayscale_cam[int(h * 0.45):int(h * 0.85), :int(w * 0.25)].flatten(),
                grayscale_cam[int(h * 0.45):int(h * 0.85), int(w * 0.75):].flatten(),
                grayscale_cam[int(h * 0.82):, :].flatten(),
            ]),
            "desc": "Jawline boundary sharpness and peripheral blending artifacts.",
        },
    ]

    regional_influence: List[Dict[str, Any]] = []
    for z in zones:
        score = float(np.mean(z["slice"])) if z["slice"].size > 0 else 0.0
        if score >= 0.50:
            level, color, hex_color = "High", "Red", "#FF4D4D"
        elif score >= 0.25:
            level, color, hex_color = "Moderate", "Yellow", "#FFD700"
        else:
            level, color, hex_color = "Low", "Blue", "#3B82F6"

        regional_influence.append({
            "region": z["region"],
            "influence_level": level,
            "color": color,
            "hex_color": hex_color,
            "score": round(score * 100, 1),
            "description": z["desc"],
        })

    rationale = build_forensic_rationale(verdict, fake_prob, regional_influence)
    return regional_influence, rationale


class BaseDeepfakeModel(ABC):
    """Abstract base class defining the unified interface for all detection backends."""

    def __init__(self, model_name: str, device: str = DEVICE):
        self.model_name = model_name
        self.device = torch.device(device)

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_heatmap_and_regions(
        self,
        image: Image.Image,
        verdict: str = "real",
        fake_prob: float = 0.5,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        pass

    def _render_cam_overlay_and_regions(
        self,
        grayscale_cam: np.ndarray,
        image: Image.Image,
        verdict: str,
        fake_prob: float,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        from pytorch_grad_cam.utils.image import show_cam_on_image

        h, w = grayscale_cam.shape
        rgb_img = np.array(image.resize((w, h))).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        success, buffer = cv2.imencode(".png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}" if success else None

        regional_influence, rationale = analyze_cam_regions(
            grayscale_cam,
            verdict=verdict,
            fake_prob=fake_prob,
        )
        return heatmap_b64, regional_influence, rationale


class ViTWrapper(nn.Module):
    """Exposes raw logits directly for pytorch-grad-cam compatibility."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(x).logits


class HuggingFaceModel(BaseDeepfakeModel):
    """Hugging Face image classification loader with dynamic id2label resolution."""

    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE):
        super().__init__(model_name=model_id, device=device)
        print(f"[Model Loader] Loading Hugging Face model: '{model_id}' on {self.device}...")

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        try:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                self.model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True).to(self.device)
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(f"Failed to load Hugging Face model '{model_id}': {exc}") from exc

        self.id2label = self.model.config.id2label or {0: "real", 1: "fake"}
        self.fake_label_id = resolve_fake_label_id(self.id2label)
        print(f"[Model Loader] Identified fake label index: {self.fake_label_id} ('{self.id2label[self.fake_label_id]}')")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]

        fake_prob = float(probs[self.fake_label_id].item())
        return {
            "fake_probability": round(fake_prob, 4),
            "verdict": get_verdict(fake_prob),
            "raw_output": {
                str(self.id2label.get(i, f"class_{i}")): float(probs[i].item())
                for i in range(len(probs))
            },
            "model_name": self.model_name,
        }

    def generate_heatmap_and_regions(
        self,
        image: Image.Image,
        verdict: str = "real",
        fake_prob: float = 0.5,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            if image.mode != "RGB":
                image = image.convert("RGB")

            input_tensor = self.processor(images=image, return_tensors="pt").to(self.device)["pixel_values"]

            if hasattr(self.model, "vit"):
                # ViT-base-patch16-224 emits 196 patch tokens (14x14) after the [CLS] token
                def reshape_transform(tensor, height=14, width=14):
                    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                    return result.transpose(2, 3).transpose(1, 2)

                if hasattr(self.model.vit, "layers"):
                    target_layers = [self.model.vit.layers[-1].layernorm_before]
                elif hasattr(self.model.vit, "encoder") and hasattr(self.model.vit.encoder, "layer"):
                    target_layers = [self.model.vit.encoder.layer[-1].layernorm_before]
                else:
                    target_layers = [self.model.vit.layernorm]

                cam = GradCAM(model=ViTWrapper(self.model), target_layers=target_layers, reshape_transform=reshape_transform)
            else:
                conv_layers = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
                if not conv_layers:
                    return None, [], "Grad-CAM target layer unavailable."
                cam = GradCAM(model=self.model, target_layers=[conv_layers[-1]])

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(self.fake_label_id)],
            )[0]
            return self._render_cam_overlay_and_regions(grayscale_cam, image, verdict, fake_prob)
        except Exception as exc:
            print(f"[Grad-CAM Error] {exc}")
            return None, [], build_forensic_rationale(verdict, fake_prob)


class CustomPyTorchModel(BaseDeepfakeModel):
    """Custom PyTorch CNN loader supporting efficientnet_b0 and resnet18."""

    def __init__(
        self,
        weights_path: str = CUSTOM_MODEL_PATH,
        architecture: str = CUSTOM_ARCHITECTURE,
        device: str = DEVICE,
    ):
        super().__init__(model_name=f"custom_pytorch ({architecture})", device=device)
        self.architecture_name = architecture
        self.weights_path = Path(weights_path)
        self.model, self.target_layer = self._build_architecture(architecture)
        self.model.to(self.device)

        if self.weights_path.is_file():
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"[Model Loader] Loaded weights from {self.weights_path}")
        else:
            print(f"[Model Loader] Warning: Weights file '{self.weights_path}' not found. Using untrained weights.")

        self.model.eval()
        self.transform = get_vit_transform()

    def _build_architecture(self, architecture: str) -> Tuple[nn.Module, nn.Module]:
        from torchvision import models

        if architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None, num_classes=2)
            return model, model.features[-1]
        if architecture == "resnet18":
            model = models.resnet18(weights=None, num_classes=2)
            return model, model.layer4[-1]
        raise ValueError(f"Unsupported custom architecture: {architecture}. Supported: ['efficientnet_b0', 'resnet18']")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(tensor), dim=-1)[0]

        fake_prob = float(probs[FAKE_CLASS_IDX].item())
        return {
            "fake_probability": round(fake_prob, 4),
            "verdict": get_verdict(fake_prob),
            "raw_output": {
                "Real": round(float(probs[REAL_CLASS_IDX].item()), 4),
                "Fake": round(float(probs[FAKE_CLASS_IDX].item()), 4),
            },
            "model_name": self.model_name,
        }

    def generate_heatmap_and_regions(
        self,
        image: Image.Image,
        verdict: str = "real",
        fake_prob: float = 0.5,
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            if image.mode != "RGB":
                image = image.convert("RGB")

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            cam = GradCAM(model=self.model, target_layers=[self.target_layer])
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(FAKE_CLASS_IDX)],
            )[0]
            return self._render_cam_overlay_and_regions(grayscale_cam, image, verdict, fake_prob)
        except Exception as exc:
            print(f"[Grad-CAM] Error generating heatmap: {exc}")
            return None, [], build_forensic_rationale(verdict, fake_prob)


def create_model_loader(
    backend_type: Optional[str] = None,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> BaseDeepfakeModel:
    backend = (backend_type or MODEL_BACKEND).lower()
    active_device = device or DEVICE

    if backend == "huggingface":
        return HuggingFaceModel(model_id=model_id or MODEL_ID, device=active_device)
    if backend == "custom_pytorch":
        return CustomPyTorchModel(
            weights_path=CUSTOM_MODEL_PATH,
            architecture=CUSTOM_ARCHITECTURE,
            device=active_device,
        )
    raise ValueError(f"Unsupported MODEL_BACKEND '{backend}'. Must be 'huggingface' or 'custom_pytorch'.")
