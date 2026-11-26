"""
face_crop.py
------------
Face detection, bounding-box margin expansion, and standardized cropping.

Design rationale for academic viva:
1. Deepfake manipulation artifacts (blending seams, boundary color shifts, warp shears)
   cluster along the facial perimeter, hairline, and jawline. An expansive margin (+20%)
   ensures these critical boundary zones are captured rather than cropped out.
2. In academic evaluation, guessing a label on an image without a detectable face leads to
   invalid forensic conclusions. Therefore, we explicitly raise NoFaceDetectedError.
3. Cropping to a standardized 112x112 resolution balances fine architectural detail with
   computationally lightweight from-scratch model training on CPU/Kaggle/Colab.
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np


class NoFaceDetectedError(Exception):
    """Raised when no human face can be detected with sufficient confidence."""
    pass


class FaceDetector:
    """
    Robust Face Detector leveraging Haar cascades with fallback multi-scale parameters.
    Fully self-contained: relies solely on OpenCV, requiring no external weights.
    """
    def __init__(self, target_size: int = 112, margin_factor: float = 0.20):
        self.target_size = target_size
        self.margin_factor = margin_factor
        
        # Load frontal face Haar cascade from OpenCV distribution
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {cascade_path}")
            
        # Auxiliary eye cascade for landmark / eye proxy detection
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def detect_faces(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Detects faces in RGB image, returning bounding boxes [x, y, w, h].
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale detection: coarse to fine
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # If no face found, retry with slightly more permissive parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(24, 24)
            )
            
        return faces

    def extract_face(
        self,
        img_rgb: np.ndarray,
        allow_fallback_center_crop: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detects the largest face, applies margin expansion, crops and resizes to target_size.

        Returns:
            face_crop (np.ndarray): RGB uint8 array [target_size, target_size, 3]
            metadata (dict): Bounding box coordinates, face count, detection confidence flag
        """
        h, w = img_rgb.shape[:2]
        faces = self.detect_faces(img_rgb)

        if len(faces) == 0:
            if allow_fallback_center_crop:
                # Square center crop as fallback when explicitly permitted (e.g. pre-aligned datasets)
                side = min(h, w)
                y1 = (h - side) // 2
                x1 = (w - side) // 2
                crop = img_rgb[y1:y1 + side, x1:x1 + side]
                resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                return resized, {
                    "face_detected": False,
                    "bbox": [x1, y1, side, side],
                    "total_faces": 0,
                    "is_fallback_crop": True
                }
            raise NoFaceDetectedError(
                "No human face detected in the image. Deepfake forensic analysis requires a visible face."
            )

        # Select largest face by area (w * h)
        largest_face = max(faces, key=lambda b: b[2] * b[3])
        fx, fy, fw, fh = largest_face

        # Apply margin expansion to include blending boundary and hairline
        mx = int(fw * self.margin_factor)
        my = int(fh * self.margin_factor)

        x1 = max(0, fx - mx)
        y1 = max(0, fy - my)
        x2 = min(w, fx + fw + mx)
        y2 = min(h, fy + fh + my)

        face_crop = img_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            raise NoFaceDetectedError("Cropped face area is empty.")

        # Standardize resolution
        face_resized = cv2.resize(
            face_crop,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA
        )

        metadata = {
            "face_detected": True,
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "total_faces": int(len(faces)),
            "is_fallback_crop": False
        }

        return face_resized, metadata

    def extract_mouth_region(self, face_crop: np.ndarray, mouth_size: int = 64) -> np.ndarray:
        """
        Extracts the lower-third region of the face crop (anatomical location of mouth/jaw).
        Used for localized facial-artifact inspection.
        """
        h, w = face_crop.shape[:2]
        # Lower 35% of the face contains the mouth and chin
        y_start = int(h * 0.65)
        x_start = int(w * 0.20)
        x_end = int(w * 0.80)
        
        mouth_roi = face_crop[y_start:h, x_start:x_end]
        if mouth_roi.size == 0:
            mouth_roi = face_crop[y_start:h, :]
            
        mouth_resized = cv2.resize(mouth_roi, (mouth_size, mouth_size), interpolation=cv2.INTER_AREA)
        return mouth_resized
