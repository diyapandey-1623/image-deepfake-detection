"""
frequency_transform.py
----------------------
Transforms spatial domain images into the 2D Frequency Domain (Fourier Transform)
and computes Error Level Analysis (ELA) compression maps.

Design rationale for academic viva:
1. Deep learning generative models (GANs such as StyleGAN, and Diffusion Models such as
   Latent Diffusion) rely on transposed convolutions or latent upsampling operations.
   These induce periodic spectral spikes, grid artifacts, and unnatural high-frequency
   spectral roll-off that are clearly identifiable in the 2D Fourier power spectrum,
   even when imperceptible to the human eye in the spatial domain (Frank et al., Durall et al.).
2. Error Level Analysis (ELA) detects differences in JPEG compression error levels. When a
   synthetic or altered face is spliced into an existing photograph, the resaved compression
   profile across the boundary shows a sharp divergence in residual error.
"""

import io
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def compute_fft_magnitude_spectrum(img_rgb: np.ndarray, target_size: int = 112) -> np.ndarray:
    """
    Computes the 2D Discrete Fourier Transform (DFT) log-magnitude power spectrum.

    Pipeline:
      RGB -> Grayscale -> 2D FFT -> Shift low frequencies to center -> Log Magnitude -> Normalization

    Returns:
        np.ndarray: Centered log-magnitude spectrum of shape (target_size, target_size, 1),
                    normalized to range [0.0, 1.0], dtype float32.
    """
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb.copy()

    # Ensure float32 for FFT calculation
    f_transform = np.fft.fft2(gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)

    # Compute magnitude and apply logarithmic compression: log(1 + |F(u, v)|)
    # Log compression expands dynamic range of subtle high frequencies
    magnitude = np.abs(f_shift)
    log_magnitude = np.log1p(magnitude)

    # Normalize to [0, 1] range
    mag_min = float(np.min(log_magnitude))
    mag_max = float(np.max(log_magnitude))
    if mag_max > mag_min:
        norm_spectrum = (log_magnitude - mag_min) / (mag_max - mag_min)
    else:
        norm_spectrum = np.zeros_like(log_magnitude)

    # Resize to standardized target dimension
    resized_spectrum = cv2.resize(
        norm_spectrum,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )

    # Return shape (target_size, target_size, 1) float32
    return np.expand_dims(resized_spectrum.astype(np.float32), axis=-1)


def compute_ela_map(img_rgb: np.ndarray, quality: int = 90, scale: float = 15.0) -> np.ndarray:
    """
    Computes Error Level Analysis (ELA) map by resaving the image at a designated JPEG quality
    level and calculating the absolute error residual between original and compressed representations.

    Parameters:
        img_rgb (np.ndarray): Original RGB image array.
        quality (int): JPEG resave quality level (standard forensic benchmark is 90).
        scale (float): Amplification factor to enhance visibility of subtle compression differences.

    Returns:
        np.ndarray: ELA residual image array of shape (H, W, 3), dtype float32, in range [0.0, 1.0].
    """
    pil_img = Image.fromarray(img_rgb)
    
    # Save to memory buffer with specific JPEG compression quality
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    
    # Reload compressed image
    compressed_pil = Image.open(buffer)
    compressed_img = np.array(compressed_pil, dtype=np.float32)
    original_img = img_rgb.astype(np.float32)

    # Compute absolute difference
    diff = np.abs(original_img - compressed_img) * scale
    
    # Clip and normalize to [0, 1]
    ela_map = np.clip(diff / 255.0, 0.0, 1.0).astype(np.float32)
    return ela_map


def compute_laplacian_sharpness_map(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes spatial gradient / edge sharpness via 2D Laplacian operator.
    Synthetic faces frequently demonstrate either over-smoothed skin textures
    or boundary ringing from sharpening filters.
    """
    if img_rgb.ndim == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb
        
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_abs = np.abs(laplacian)
    
    max_val = np.max(laplacian_abs)
    if max_val > 0:
        norm_lap = laplacian_abs / max_val
    else:
        norm_lap = np.zeros_like(laplacian_abs)
        
    return norm_lap
