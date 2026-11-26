"""
backend/model/samples.py
------------------------
Procedural synthetic portrait generator for sample endpoints and offline tests.
"""

import cv2
import numpy as np


def create_synthetic_portrait(
    width: int = 320,
    height: int = 320,
    is_fake: bool = False,
) -> np.ndarray:
    """Synthesizes a human portrait RGB array, optionally with GAN/smoothing artifacts."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 230
    for y in range(height):
        img[y, :, :] = np.clip(img[y, :, :].astype(np.int32) - int(y * 40 / height), 0, 255)

    center_x = width // 2
    center_y = int(height * 0.48)
    face_w = int(width * 0.28)
    face_h = int(height * 0.36)

    cv2.ellipse(img, (center_x, int(center_y * 0.85)), (int(face_w * 1.3), int(face_h * 1.2)), 0, 0, 360, (40, 35, 30), -1)
    skin_color = (195, 215, 245) if not is_fake else (180, 210, 255)
    cv2.ellipse(img, (center_x, center_y), (face_w, face_h), 0, 0, 360, (skin_color[2], skin_color[1], skin_color[0]), -1)

    eye_offset_x = int(face_w * 0.48)
    eye_y = int(center_y - face_h * 0.12)
    for sign in (-1, 1):
        ex = center_x + sign * eye_offset_x
        cv2.ellipse(img, (ex, eye_y), (14, 8), 0, 0, 360, (250, 250, 250), -1)
        cv2.circle(img, (ex, eye_y), 6, (40, 60, 90), -1)
        cv2.circle(img, (ex, eye_y), 3, (10, 10, 10), -1)
        cv2.ellipse(img, (ex, eye_y - 12), (15, 3), 0, 0, 180, (40, 35, 30), 2)

    nose_y = int(center_y + face_h * 0.15)
    cv2.line(img, (center_x, eye_y + 5), (center_x, nose_y), (160, 180, 210), 2)
    cv2.circle(img, (center_x, nose_y), 4, (150, 170, 200), -1)

    mouth_y = int(center_y + face_h * 0.45)
    lip_color = (130, 130, 210) if not is_fake else (110, 120, 230)
    cv2.ellipse(img, (center_x, mouth_y), (22, 9), 0, 0, 360, lip_color, -1)
    cv2.line(img, (center_x - 18, mouth_y), (center_x + 18, mouth_y), (80, 80, 160), 2)

    if is_fake:
        freq = 8.0
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        grid_pattern = np.sin(x_coords * (2 * np.pi / freq)) * np.cos(y_coords * (2 * np.pi / freq))
        face_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(face_mask, (center_x, center_y), (face_w, face_h), 0, 0, 360, 255, -1)

        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            ch[face_mask > 0] += grid_pattern[face_mask > 0] * 18.0
            img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        for c in range(3):
            img[:, :, c] = np.where(face_mask > 0, smoothed[:, :, c], img[:, :, c])

        seam_mask = cv2.Canny(face_mask, 100, 200)
        seam_dilated = cv2.dilate(seam_mask, np.ones((3, 3), np.uint8))
        img[seam_dilated > 0] = np.clip(img[seam_dilated > 0].astype(np.int32) + 25, 0, 255).astype(np.uint8)
    else:
        gaussian_noise = np.random.normal(0, 3.0, img.shape)
        img = np.clip(img.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
