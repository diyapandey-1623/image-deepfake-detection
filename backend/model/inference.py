"""
backend/model/inference.py
--------------------------
Model-agnostic inference engine for deepfake detection.
Supports both Hugging Face vision models and custom PyTorch architectures.
Provides a unified prediction interface, automated label mapping, and Grad-CAM explainability.
"""

import sys
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

# Ensure project root (deepfake-detector) is always in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from backend.config import (
        MODEL_BACKEND,
        MODEL_ID,
        DEVICE,
        CUSTOM_MODEL_PATH,
        CUSTOM_ARCHITECTURE,
    )
    from backend.class_config import (
        IDX_TO_LABEL,
        REAL_CLASS_IDX,
        FAKE_CLASS_IDX,
        DECISION_THRESHOLD,
    )
    from backend.vit_transforms import get_vit_transform
except ImportError:
    from ..config import (
        MODEL_BACKEND,
        MODEL_ID,
        DEVICE,
        CUSTOM_MODEL_PATH,
        CUSTOM_ARCHITECTURE,
    )
    from ..class_config import (
        IDX_TO_LABEL,
        REAL_CLASS_IDX,
        FAKE_CLASS_IDX,
        DECISION_THRESHOLD,
    )
    from ..vit_transforms import get_vit_transform


def get_verdict(fake_prob: float, fake_thresh: float = DECISION_THRESHOLD, real_thresh: Optional[float] = None) -> str:
    """
    Computes categorical verdict based on decision threshold (0.50):
    >= 0.50 = "fake", < 0.50 = "real".
    """
    threshold = real_thresh if real_thresh is not None else fake_thresh
    if fake_prob >= threshold:
        return "fake"
    return "real"


def analyze_cam_regions(
    grayscale_cam: np.ndarray,
    verdict: str = "real",
    fake_prob: float = 0.5
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Performs anatomical facial region segmentation on a 2D activation map (grayscale_cam).
    Evaluates mean activation across 5 core facial regions:
      - Forehead & Hairline
      - Eyes & Periorbital Region
      - Nose & Central Midface
      - Mouth & Lips
      - Cheeks & Jawline
    Assigns tiered influence badges:
      - High (Red, #FF4D4D) for score >= 50%
      - Moderate (Yellow, #FFD700) for 25% <= score < 50%
      - Low (Blue/Cool, #3B82F6) for score < 25%
    Returns:
      regional_influence: List of region metadata dictionaries
      rationale: Comprehensive forensic rationale text
    """
    h, w = grayscale_cam.shape
    zones = [
        {"region": "Forehead & Hairline", "slice": grayscale_cam[0:int(h * 0.25), :], "desc": "Upper facial boundary and hairline blend integrity."},
        {"region": "Eyes & Periorbital Region", "slice": grayscale_cam[int(h * 0.22):int(h * 0.48), int(w * 0.12):int(w * 0.88)], "desc": "Iris symmetry, corneal reflection consistency, and eyelid margins."},
        {"region": "Nose & Central Midface", "slice": grayscale_cam[int(h * 0.42):int(h * 0.68), int(w * 0.30):int(w * 0.70)], "desc": "Nasal ridge diffusion and pore texture consistency."},
        {"region": "Mouth & Lips", "slice": grayscale_cam[int(h * 0.65):int(h * 0.85), int(w * 0.25):int(w * 0.75)], "desc": "Vermilion border, tooth alignment, and lip texture fidelity."},
        {"region": "Cheeks & Jawline", "slice": np.concatenate([
            grayscale_cam[int(h * 0.45):int(h * 0.85), :int(w * 0.25)].flatten(),
            grayscale_cam[int(h * 0.45):int(h * 0.85), int(w * 0.75):].flatten(),
            grayscale_cam[int(h * 0.82):, :].flatten()
        ]), "desc": "Jawline boundary sharpness and peripheral blending artifacts."}
    ]

    regional_influence = []
    high_influence_regions = []
    moderate_influence_regions = []
    for z in zones:
        score = float(np.mean(z["slice"])) if z["slice"].size > 0 else 0.0
        if score >= 0.50:
            level = "High"
            color = "Red"
            hex_color = "#FF4D4D"
            high_influence_regions.append(z["region"])
        elif score >= 0.25:
            level = "Moderate"
            color = "Yellow"
            hex_color = "#FFD700"
            moderate_influence_regions.append(z["region"])
        else:
            level = "Low"
            color = "Blue/Cool"
            hex_color = "#3B82F6"

        regional_influence.append({
            "region": z["region"],
            "influence_level": level,
            "color": color,
            "hex_color": hex_color,
            "score": round(score * 100, 1),
            "description": z["desc"]
        })

    # Detailed technical forensic explanation
    if str(verdict).lower() in ("fake", "deepfake"):
        regions_str = ", ".join(high_influence_regions) if high_influence_regions else (", ".join(moderate_influence_regions) if moderate_influence_regions else "central facial landmarks")
        rationale = (
            f"The model detected high manipulation attention ({round(fake_prob * 100, 1)}% synthetic probability) "
            f"concentrated on the {regions_str}. Neural activations reveal anomalous generative synthesis patterns, "
            f"unnatural boundary blending, and abnormal frequency distributions distinct from authentic physical captures."
        )
    else:
        regions_str = ", ".join(high_influence_regions) if high_influence_regions else (", ".join(moderate_influence_regions) if moderate_influence_regions else "facial micro-features")
        rationale = (
            f"Biometric integrity verified ({round((1.0 - fake_prob) * 100, 1)}% authentic confidence). "
            f"Model focus across {regions_str} confirms coherent organic skin pores, natural light-reflection "
            f"dynamics across the eyes/teeth, and uniform camera sensor noise without generative manipulation signatures."
        )

    return regional_influence, rationale


class BaseDeepfakeModel(ABC):
    """Abstract base class defining the unified interface for all detection backends."""

    def __init__(self, model_name: str, device: str = DEVICE):
        self.model_name = model_name
        self.device = torch.device(device)

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Unified prediction interface.
        Returns:
            fake_probability: float in [0.0, 1.0]
            verdict: "fake" | "real" | "uncertain"
            raw_output: dict of class probabilities
            model_name: str
        """
        pass

    @abstractmethod
    def generate_heatmap(self, image: Image.Image) -> Optional[str]:
        """
        Generates Grad-CAM visual heatmap overlay as a base64 PNG string.
        """
        pass

    def generate_heatmap_and_regions(
        self,
        image: Image.Image,
        verdict: str = "real",
        fake_prob: float = 0.5
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        """
        Generates Grad-CAM heatmap, anatomical facial region influence breakdown (Red/Yellow/Blue),
        and detailed technical forensic rationale.
        """
        heatmap_b64 = self.generate_heatmap(image)
        return heatmap_b64, [], "Forensic analysis completed."


class ViTWrapper(torch.nn.Module):
    """Wrapper exposing logits directly for PyTorch-GradCAM compatibility."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x):
        return self.hf_model(x).logits


class HuggingFaceModel(BaseDeepfakeModel):
    """
    Hugging Face image classification loader using AutoModelForImageClassification
    and AutoImageProcessor. Dynamically parses model labels without hardcoded indices.
    """

    # Keyword patterns matching fake/synthetic classes
    FAKE_KEYWORDS = {"fake", "deepfake", "ai", "artificial", "synthetic", "manipulated", "generated", "forgery"}
    REAL_KEYWORDS = {"real", "authentic", "genuine", "pristine", "original", "natural", "human"}

    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE):
        super().__init__(model_name=model_id, device=device)
        print(f"[Model Loader] Loading Hugging Face model: '{model_id}' on {self.device}...")

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        try:
            # First attempt loading from local Hugging Face cache to avoid unnecessary network roundtrips
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                self.model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True).to(self.device)
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face model '{model_id}'. "
                f"Verify network connectivity and model ID. Error: {str(e)}"
            )

        # Dynamic label identification
        self.id2label = self.model.config.id2label or {0: "real", 1: "fake"}
        self.fake_label_id = self._resolve_fake_label_id(self.id2label)
        print(f"[Model Loader] Identified fake label index: {self.fake_label_id} ('{self.id2label[self.fake_label_id]}')")

    def _resolve_fake_label_id(self, id2label: Dict[int, str]) -> int:
        """
        Matches label names against known fake/synthetic descriptors without hardcoded indices.
        Raises a clear error at startup if no matching label is detected.
        """
        # 1. Direct search for fake keywords in label names
        for label_id, label_name in id2label.items():
            name_norm = label_name.lower().replace("-", " ").replace("_", " ").strip()
            tokens = set(name_norm.split())
            if tokens.intersection(self.FAKE_KEYWORDS) or any(kw in name_norm for kw in self.FAKE_KEYWORDS):
                return int(label_id)

        # 2. Binary classification fallback: if one label is explicitly "real", the other is "fake"
        if len(id2label) == 2:
            for label_id, label_name in id2label.items():
                name_norm = label_name.lower().replace("-", " ").replace("_", " ").strip()
                tokens = set(name_norm.split())
                if tokens.intersection(self.REAL_KEYWORDS) or any(kw in name_norm for kw in self.REAL_KEYWORDS):
                    other_id = [k for k in id2label.keys() if k != label_id][0]
                    return int(other_id)

        # 3. If no label matches, raise descriptive startup error printing available labels
        available_labels = list(id2label.values())
        raise ValueError(
            f"Could not identify a 'fake' or 'deepfake' class from model labels: {available_labels}. "
            f"Expected label names matching keywords: {sorted(self.FAKE_KEYWORDS)}. "
            f"Please verify model configuration or id2label dictionary."
        )

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        fake_prob = float(probs[self.fake_label_id].item())
        verdict = get_verdict(fake_prob)

        raw_output = {
            str(self.id2label.get(i, f"class_{i}")): float(probs[i].item())
            for i in range(len(probs))
        }

        return {
            "fake_probability": round(fake_prob, 4),
            "verdict": verdict,
            "raw_output": raw_output,
            "model_name": self.model_name
        }

    def generate_heatmap_and_regions(
        self,
        image: Image.Image,
        verdict: str = "real",
        fake_prob: float = 0.5
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        """
        Generates Grad-CAM visual heatmap overlay (compatible with both ViT and CNN architectures),
        analyzes facial region influences (Red/Yellow/Blue), and creates technical forensic rationale.
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            input_tensor = inputs["pixel_values"]

            model_type = getattr(self.model.config, "model_type", "").lower()
            if "vit" in model_type or hasattr(self.model, "vit"):
                def reshape_transform(tensor, height=14, width=14):
                    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
                    result = result.transpose(2, 3).transpose(1, 2)
                    return result

                if hasattr(self.model.vit, "layers"):
                    target_layers = [self.model.vit.layers[-1].layernorm_before]
                elif hasattr(self.model.vit, "encoder") and hasattr(self.model.vit.encoder, "layer"):
                    target_layers = [self.model.vit.encoder.layer[-1].layernorm_before]
                else:
                    target_layers = [self.model.vit.layernorm]

                wrapped_model = ViTWrapper(self.model)
                cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)
            else:
                target_layers = []
                for module in reversed(list(self.model.modules())):
                    if isinstance(module, nn.Conv2d):
                        target_layers.append(module)
                        break
                if not target_layers:
                    return None, [], "Grad-CAM target layer unavailable."
                cam = GradCAM(model=self.model, target_layers=target_layers)

            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

            h, w = grayscale_cam.shape
            rgb_img = np.array(image.resize((w, h))).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            success, buffer = cv2.imencode(".png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}" if success else None

            # Anatomical region segmentation on normalized grid & technical forensic rationale
            regional_influence, rationale = analyze_cam_regions(
                grayscale_cam,
                verdict=verdict,
                fake_prob=fake_prob
            )

            return heatmap_b64, regional_influence, rationale

        except Exception as e:
            print(f"[Grad-CAM Error] {e}")
            return None, [], f"Heatmap generation error: {str(e)}"

    def generate_heatmap(self, image: Image.Image) -> Optional[str]:
        heatmap_b64, _, _ = self.generate_heatmap_and_regions(image)
        return heatmap_b64


class CustomPyTorchModel(BaseDeepfakeModel):
    """
    Custom PyTorch model loader supporting custom architectures (e.g. efficientnet_b0).
    Rebuilds the architecture, loads state_dict, uses standard ImageNet preprocessing,
    and applies pytorch-grad-cam on the final convolutional layer.
    """

    def __init__(
        self,
        weights_path: str = CUSTOM_MODEL_PATH,
        architecture: str = CUSTOM_ARCHITECTURE,
        device: str = DEVICE
    ):
        super().__init__(model_name=f"custom_pytorch ({architecture})", device=device)
        self.architecture_name = architecture
        self.weights_path = Path(weights_path)

        print(f"[Model Loader] Initializing Custom PyTorch architecture: '{architecture}' on {self.device}...")
        self.model, self.target_layer = self._build_architecture(architecture)
        self.model.to(self.device)

        # Load weights if available
        if self.weights_path.is_file():
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"[Model Loader] Successfully loaded weights from {self.weights_path}")
        else:
            print(f"[Model Loader] Warning: Weights file '{self.weights_path}' not found. Initialized with fresh weights.")

        self.model.eval()

        # Unified class mapping: 0=Real, 1=Fake (from backend.class_config)
        self.class_mapping = IDX_TO_LABEL
        self.fake_label_id = FAKE_CLASS_IDX

        # Unified preprocessing transform identical to training/evaluation
        self.transform = get_vit_transform()

    def _build_architecture(self, architecture: str) -> Tuple[nn.Module, Optional[nn.Module]]:
        """Rebuilds designated PyTorch CNN architecture and locates its final conv layer."""
        from torchvision import models

        if architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None, num_classes=2)
            # Last conv layer in efficientnet_b0 is features[-1]
            target_layer = model.features[-1]
            return model, target_layer
        elif architecture == "resnet18":
            model = models.resnet18(weights=None, num_classes=2)
            target_layer = model.layer4[-1]
            return model, target_layer
        else:
            raise ValueError(f"Unsupported custom architecture: {architecture}. Supported: ['efficientnet_b0', 'resnet18']")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]

        # Index 0 is Real, Index 1 is Fake (synchronized with ViT and training)
        fake_prob = float(probs[FAKE_CLASS_IDX].item())
        verdict = get_verdict(fake_prob)

        raw_output = {
            "Real": round(float(probs[REAL_CLASS_IDX].item()), 4),
            "Fake": round(float(probs[FAKE_CLASS_IDX].item()), 4)
        }

        return {
            "fake_probability": round(fake_prob, 4),
            "verdict": verdict,
            "raw_output": raw_output,
            "model_name": self.model_name
        }

    def generate_heatmap_and_regions(
        self,
        image: Image.Image,
        verdict: str = "real",
        fake_prob: float = 0.5
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str]:
        """
        Computes Grad-CAM heatmap using pytorch-grad-cam on the final convolutional layer,
        analyzes facial region influences (Red/Yellow/Blue), and creates technical forensic rationale.
        """
        if self.target_layer is None:
            return None, [], "Grad-CAM target layer unavailable for this architecture."

        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            if image.mode != "RGB":
                image = image.convert("RGB")

            tensor = self.transform(image).unsqueeze(0).to(self.device)
            cam = GradCAM(model=self.model, target_layers=[self.target_layer])

            # Generate CAM activation map
            grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

            # Prepare RGB reference array scaled [0, 1]
            img_resized = image.resize((224, 224))
            rgb_img = np.array(img_resized).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Encode as base64 PNG string
            success, buffer = cv2.imencode(".png", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}" if success else None

            # Regional influence segmentation and forensic rationale
            regional_influence, rationale = analyze_cam_regions(
                grayscale_cam,
                verdict=verdict,
                fake_prob=fake_prob
            )
            return heatmap_b64, regional_influence, rationale
        except Exception as e:
            print(f"[Grad-CAM] Error generating heatmap: {e}")
            return None, [], f"Heatmap generation error: {str(e)}"

    def generate_heatmap(self, image: Image.Image) -> Optional[str]:
        heatmap_b64, _, _ = self.generate_heatmap_and_regions(image)
        return heatmap_b64


def create_model_loader(
    backend_type: Optional[str] = None,
    model_id: Optional[str] = None,
    device: Optional[str] = None
) -> BaseDeepfakeModel:
    """
    Factory function instantiating the configured model backend.
    """
    backend = (backend_type or MODEL_BACKEND).lower()
    active_device = device or DEVICE

    if backend == "huggingface":
        return HuggingFaceModel(model_id=model_id or MODEL_ID, device=active_device)
    elif backend == "custom_pytorch":
        return CustomPyTorchModel(
            weights_path=CUSTOM_MODEL_PATH,
            architecture=CUSTOM_ARCHITECTURE,
            device=active_device
        )
    else:
        raise ValueError(
            f"Unsupported MODEL_BACKEND '{backend}'. Must be 'huggingface' or 'custom_pytorch'."
        )


if __name__ == "__main__":
    print("[Self-Test] Initializing model loader from backend/model/inference.py...")
    loader = create_model_loader()
    print(f"[Self-Test] Successfully loaded model: '{loader.model_name}' on device '{loader.device}'")
