"""
generate_test_samples.py
-------------------------
Generates sample synthetic portraits and pristine reference images for unit tests,
smoke tests, and end-to-end verification without requiring internet downloads.
"""

import os
from pathlib import Path
import cv2
import numpy as np


def create_synthetic_portrait(
    width: int = 320,
    height: int = 320,
    is_fake: bool = False,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Synthesizes a human portrait image using procedural computer vision primitives.
    
    If is_fake is True, injects specific deepfake artifacts:
      1. Periodic 2D high-frequency grid interference (mimicking GAN checkerboard / upsampling)
      2. Inconsistent boundary blending and color cast
      3. Over-smoothed skin texture (Laplacian reduction)
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 230  # Light studio background
    
    # Add subtle gradient to background
    for y in range(height):
        img[y, :, :] = np.clip(img[y, :, :].astype(np.int32) - int(y * 40 / height), 0, 255)

    center_x = width // 2
    center_y = int(height * 0.48)
    face_w = int(width * 0.28)
    face_h = int(height * 0.36)

    # Hair / silhouette
    cv2.ellipse(img, (center_x, int(center_y * 0.85)), (int(face_w * 1.3), int(face_h * 1.2)), 0, 0, 360, (40, 35, 30), -1)

    # Head oval (skin tone: natural warm peach)
    skin_color = (195, 215, 245) if not is_fake else (180, 210, 255)  # RGB
    cv2.ellipse(img, (center_x, center_y), (face_w, face_h), 0, 0, 360, (skin_color[2], skin_color[1], skin_color[0]), -1)

    # Eyes
    eye_offset_x = int(face_w * 0.48)
    eye_y = int(center_y - face_h * 0.12)
    for sign in [-1, 1]:
        ex = center_x + sign * eye_offset_x
        # Sclera (white)
        cv2.ellipse(img, (ex, eye_y), (14, 8), 0, 0, 360, (250, 250, 250), -1)
        # Iris (brown)
        cv2.circle(img, (ex, eye_y), 6, (40, 60, 90), -1)
        # Pupil (black)
        cv2.circle(img, (ex, eye_y), 3, (10, 10, 10), -1)
        # Eyebrow
        cv2.ellipse(img, (ex, eye_y - 12), (15, 3), 0, 0, 180, (40, 35, 30), 2)

    # Nose bridge & tip
    nose_y = int(center_y + face_h * 0.15)
    cv2.line(img, (center_x, eye_y + 5), (center_x, nose_y), (160, 180, 210), 2)
    cv2.circle(img, (center_x, nose_y), 4, (150, 170, 200), -1)

    # Mouth / Lips
    mouth_y = int(center_y + face_h * 0.45)
    lip_color = (130, 130, 210) if not is_fake else (110, 120, 230)
    cv2.ellipse(img, (center_x, mouth_y), (22, 9), 0, 0, 360, lip_color, -1)
    cv2.line(img, (center_x - 18, mouth_y), (center_x + 18, mouth_y), (80, 80, 160), 2)

    # If FAKE: inject GAN/Diffusion manipulation artifacts
    if is_fake:
        # Artifact 1: Periodic 2D spectral grid (StyleGAN / deconvolution checkerboard artifact)
        grid_pattern = np.zeros((height, width), dtype=np.float32)
        freq = 8.0  # high frequency periodic component
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        grid_pattern = np.sin(x_coords * (2 * np.pi / freq)) * np.cos(y_coords * (2 * np.pi / freq))
        # Mask artifact strictly inside face region
        face_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(face_mask, (center_x, center_y), (face_w, face_h), 0, 0, 360, 255, -1)
        
        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            ch[face_mask > 0] += grid_pattern[face_mask > 0] * 18.0
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # Artifact 2: Skin over-smoothing (bilateral filtering inside face mask)
        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        for c in range(3):
            img[:, :, c] = np.where(face_mask > 0, smoothed[:, :, c], img[:, :, c])

        # Artifact 3: Boundary blending inconsistency (seam along jawline)
        seam_mask = cv2.Canny(face_mask, 100, 200)
        seam_dilated = cv2.dilate(seam_mask, np.ones((3, 3), np.uint8))
        img[seam_dilated > 0] = np.clip(img[seam_dilated > 0].astype(np.int32) + 25, 0, 255).astype(np.uint8)
    else:
        # Natural sensor noise
        gaussian_noise = np.random.normal(0, 3.0, img.shape)
        img = np.clip(img.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def generate_sample_dataset(output_dir: str = "data/raw", num_samples: int = 12):
    """
    Generates a starter dataset with balanced real/fake subjects for immediate testing.
    """
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} sample images into {base_path}...")
    np.random.seed(42)

    for i in range(num_samples):
        is_fake = (i % 2 != 0)
        subject_id = f"subject_{i // 2 + 1:02d}"
        kind = "fake_gan" if is_fake else "real"
        
        folder = base_path / subject_id / kind
        folder.mkdir(parents=True, exist_ok=True)
        
        img_rgb = create_synthetic_portrait(is_fake=is_fake)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        file_path = folder / f"sample_{i:03d}.jpg"
        cv2.imwrite(str(file_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
    print("[Done] Generated sample dataset.")


if __name__ == "__main__":
    generate_sample_dataset()
