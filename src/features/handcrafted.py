"""
handcrafted.py
--------------
Extraction of forensic and frequency domain handcrafted features from face images.

Design decisions and viva defense points:
1. Deep neural networks sometimes overfit to high-level semantic shortcuts. Handcrafted
   features provide an orthogonal, physics- and signal-grounded perspective:
   - Spectral Energy Distribution: Natural cameras obey natural 1/f^alpha image statistics.
     Generative models (GANs/diffusion) violate this with periodic upsampling spikes and
     anomalous high-frequency roll-offs.
   - Spatial Texture & Gradient Discontinuities: Blending a fake face into a pristine photo
     creates edge blurring or sharpening discrepancies between boundary and center.
   - Error Level Analysis (ELA): Compression artifacts differ when synthetic pixels with
     no prior JPEG history are saved alongside existing JPEG backgrounds.
   - Chrominance Inconsistency: Generative models frequently introduce unnatural color channel
     cross-correlations and skin-chrominance deviations in YCbCr/HSV color spaces.

Output: Standardized 1D vector of exactly 32 forensic features.
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np
from scipy import stats

from src.preprocessing.frequency_transform import compute_fft_magnitude_spectrum, compute_ela_map


HANDCRAFTED_FEATURE_DIM = 32


def extract_spectral_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes 10 frequency-domain forensic features from 2D FFT and DCT.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # 2D FFT log-magnitude
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    power_spectrum = magnitude ** 2
    total_power = np.sum(power_spectrum) + 1e-10

    # Calculate radial distance matrix from center (DC component)
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    r_matrix = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    # Radial frequency bands: Low (<15%), Mid (15%-50%), High (>50%)
    low_mask = r_matrix < (0.15 * max_r)
    mid_mask = (r_matrix >= (0.15 * max_r)) & (r_matrix < (0.50 * max_r))
    high_mask = r_matrix >= (0.50 * max_r)

    low_energy_ratio = float(np.sum(power_spectrum[low_mask]) / total_power)
    mid_energy_ratio = float(np.sum(power_spectrum[mid_mask]) / total_power)
    high_energy_ratio = float(np.sum(power_spectrum[high_mask]) / total_power)

    # Azimuthal / Radial average profile for spectral decay slope
    num_bins = min(32, int(max_r))
    r_int = (r_matrix * (num_bins - 1) / max_r).astype(np.int32)
    radial_profile = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        mask = (r_int == b)
        if np.any(mask):
            radial_profile[b] = np.mean(power_spectrum[mask])

    # Power law decay slope: log(P) = -alpha * log(f)
    freqs = np.arange(1, num_bins)
    p_vals = radial_profile[1:] + 1e-10
    slope, intercept, r_val, _, _ = stats.linregress(np.log(freqs), np.log(p_vals))
    spectral_decay_slope = float(slope)
    decay_fit_r2 = float(r_val ** 2)

    # Spectral roll-off: frequency index where 85% of power is accumulated
    sorted_r = np.argsort(r_matrix.ravel())
    sorted_power = power_spectrum.ravel()[sorted_r]
    cumsum_power = np.cumsum(sorted_power)
    roll_off_idx = np.searchsorted(cumsum_power, 0.85 * total_power)
    spectral_rolloff = float(roll_off_idx / len(sorted_r))

    # Quadrant symmetry ratio (GAN checkerboards break natural quad-symmetry)
    q1 = np.sum(magnitude[:cy, :cx])
    q2 = np.sum(magnitude[:cy, cx:])
    q3 = np.sum(magnitude[cy:, :cx])
    q4 = np.sum(magnitude[cy:, cx:])
    q_vals = [q1, q2, q3, q4]
    quad_asymmetry = float(np.std(q_vals) / (np.mean(q_vals) + 1e-10))

    # Discrete Cosine Transform (DCT) energy compactness
    dct = cv2.dct(gray)
    dct_power = dct ** 2
    dct_total = np.sum(dct_power) + 1e-10
    # Top 8x8 low-frequency DCT coefficients energy ratio
    dct_low_energy = float(np.sum(dct_power[:min(8, h), :min(8, w)]) / dct_total)
    dct_high_energy = float(1.0 - dct_low_energy)

    return np.array([
        low_energy_ratio,
        mid_energy_ratio,
        high_energy_ratio,
        spectral_decay_slope,
        decay_fit_r2,
        spectral_rolloff,
        quad_asymmetry,
        dct_low_energy,
        dct_high_energy,
        float(np.mean(magnitude) / (np.std(magnitude) + 1e-10))  # spectral flatness proxy
    ], dtype=np.float32)


def extract_texture_edge_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes 8 spatial texture and edge gradient features.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Laplacian of entire face (global sharpness)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var_global = float(np.var(lap))

    # Boundary vs center sharpness ratio
    margin_y, margin_x = int(h * 0.15), int(w * 0.15)
    center_lap = lap[margin_y:h-margin_y, margin_x:w-margin_x]
    
    # Boundary mask
    boundary_mask = np.ones((h, w), dtype=bool)
    boundary_mask[margin_y:h-margin_y, margin_x:w-margin_x] = False
    boundary_lap = lap[boundary_mask]

    lap_var_center = float(np.var(center_lap)) if center_lap.size > 0 else lap_var_global
    lap_var_boundary = float(np.var(boundary_lap)) if boundary_lap.size > 0 else lap_var_global
    boundary_to_center_sharpness = float(lap_var_boundary / (lap_var_center + 1e-6))

    # Sobel gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))
    grad_p90 = float(np.percentile(grad_mag, 90))

    # Local variance (texture homogeneity)
    local_mean = cv2.blur(gray.astype(np.float32), (5, 5))
    local_sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (5, 5))
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)
    local_var_mean = float(np.mean(local_var))
    local_var_std = float(np.std(local_var))

    return np.array([
        lap_var_global,
        lap_var_center,
        lap_var_boundary,
        boundary_to_center_sharpness,
        grad_mean,
        grad_std,
        grad_p90,
        float(local_var_mean / (local_var_std + 1e-6))
    ], dtype=np.float32)


def extract_ela_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes 6 Error Level Analysis (ELA) forensic statistics.
    """
    ela = compute_ela_map(img_rgb, quality=90, scale=15.0)
    h, w, c = ela.shape

    ela_mean = float(np.mean(ela))
    ela_std = float(np.std(ela))
    ela_max = float(np.max(ela))
    ela_p95 = float(np.percentile(ela, 95))

    # Per-channel discrepancy (synthetic generators often exhibit color channel compression imbalances)
    ela_r_mean = float(np.mean(ela[:, :, 0]))
    ela_g_mean = float(np.mean(ela[:, :, 1]))
    ela_b_mean = float(np.mean(ela[:, :, 2]))
    channel_discrepancy = float(np.std([ela_r_mean, ela_g_mean, ela_b_mean]))

    # Boundary vs center ELA residual discrepancy
    my, mx = int(h * 0.15), int(w * 0.15)
    center_ela = np.mean(ela[my:h-my, mx:w-mx])
    boundary_ela = (np.sum(ela) - np.sum(ela[my:h-my, mx:w-mx])) / (h * w - (h - 2*my) * (w - 2*mx) + 1e-6)
    ela_boundary_discrepancy = float(abs(boundary_ela - center_ela))

    return np.array([
        ela_mean,
        ela_std,
        ela_max,
        ela_p95,
        channel_discrepancy,
        ela_boundary_discrepancy
    ], dtype=np.float32)


def extract_color_chrominance_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Computes 8 color distribution and chrominance consistency features.
    """
    # YCbCr color space
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    
    cb_std = float(np.std(cb))
    cr_std = float(np.std(cr))
    cb_cr_corr = float(np.corrcoef(cb.ravel(), cr.ravel())[0, 1])
    if np.isnan(cb_cr_corr):
        cb_cr_corr = 0.0

    # HSV color space
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s_std = float(np.std(hsv[:, :, 1]))
    v_std = float(np.std(hsv[:, :, 2]))

    # Cross-channel RGB Pearson correlations
    r = img_rgb[:, :, 0].ravel().astype(np.float32)
    g = img_rgb[:, :, 1].ravel().astype(np.float32)
    b = img_rgb[:, :, 2].ravel().astype(np.float32)
    
    corr_rg = float(np.corrcoef(r, g)[0, 1])
    corr_rb = float(np.corrcoef(r, b)[0, 1])
    corr_gb = float(np.corrcoef(g, b)[0, 1])

    return np.array([
        cb_std,
        cr_std,
        cb_cr_corr,
        s_std,
        v_std,
        corr_rg if not np.isnan(corr_rg) else 0.0,
        corr_rb if not np.isnan(corr_rb) else 0.0,
        corr_gb if not np.isnan(corr_gb) else 0.0
    ], dtype=np.float32)


def extract_all_handcrafted_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts complete 32-dimensional forensic feature vector.
    
    Returns:
        np.ndarray: 1D float32 array of shape (32,) with clean NaN/Inf handling.
    """
    spec_feats = extract_spectral_features(img_rgb)          # 10
    texture_feats = extract_texture_edge_features(img_rgb)    # 8
    ela_feats = extract_ela_features(img_rgb)                # 6
    color_feats = extract_color_chrominance_features(img_rgb)# 8

    feature_vector = np.concatenate([spec_feats, texture_feats, ela_feats, color_feats])
    
    # Sanitize any accidental numerical instability (NaN or Inf)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return feature_vector.astype(np.float32)
