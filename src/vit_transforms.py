"""
src/vit_transforms.py
---------------------
Shared PyTorch transform for Vision Transformer (ViT) deepfake detection.
Ensures identical preprocessing across training, evaluation, and backend inference.
"""

from torchvision import transforms

def get_vit_transform():
    """
    Standard ViT image preprocessing:
    - Resize to (224, 224) using Bilinear interpolation
    - Convert PIL image [0, 255] to PyTorch Tensor [C, H, W] in [0.0, 1.0]
    - Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] to [-1.0, 1.0]
    """
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
