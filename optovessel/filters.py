"""Multiscale Hessian vesselness filtering (Frangi vessel filter) and contrast enhancement."""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

def compute_hessian_2d(image: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute smoothed 2D Hessian matrix components [I_xx, I_xy, I_yy] at scale sigma."""
    # Smooth image
    smoothed = gaussian_filter(image, sigma=sigma, mode='reflect')
    # Gradients
    gy, gx = np.gradient(smoothed)
    # Second derivatives
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    # Scale-normalized Hessian (Lindeberg gamma=2)
    scale = sigma ** 2
    return gxx * scale, 0.5 * (gxy + gyx) * scale, gyy * scale

def frangi_vesselness_2d(
    image: np.ndarray,
    sigmas: Sequence[float] = (1.0, 2.0, 3.0),
    beta: float = 0.5,
    c: float = 15.0,
    black_ridges: bool = False
) -> np.ndarray:
    """Compute multiscale Frangi vesselness filter for 2D en-face angiograms.
    
    Args:
        image: 2D input image (Height, Width)
        sigmas: Scales over which to detect vascular structures
        beta: Sensitivity to vessel blobness vs tubularity (typically 0.5)
        c: Sensitivity to background noise contrast
        black_ridges: False for bright vessels on dark background (OCTA standard)
        
    Returns:
        Enhanced vessel probability map of shape (Height, Width) in [0, 1].
    """
    H, W = image.shape
    max_vesselness = np.zeros((H, W), dtype=np.float64)
    
    for sigma in sigmas:
        Ixx, Ixy, Iyy = compute_hessian_2d(image, sigma=sigma)
        
        # Eigenvalues of 2x2 symmetric matrix: [[Ixx, Ixy], [Ixy, Iyy]]
        # det = Ixx*Iyy - Ixy^2, trace = Ixx + Iyy
        # lambda = 0.5 * (trace +/- sqrt(trace^2 - 4*det))
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        discriminant = np.maximum(trace**2 - 4 * det, 0.0)
        sqrt_disc = np.sqrt(discriminant)
        
        lambda1 = 0.5 * (trace - sqrt_disc)
        lambda2 = 0.5 * (trace + sqrt_disc)
        
        # Sort eigenvalues such that |lambda1| <= |lambda2|
        swap_mask = np.abs(lambda1) > np.abs(lambda2)
        l1 = np.where(swap_mask, lambda2, lambda1)
        l2 = np.where(swap_mask, lambda1, lambda2)
        
        # Blobness measure Rb = |lambda1| / |lambda2|
        eps = 1e-10
        Rb = np.abs(l1) / (np.abs(l2) + eps)
        
        # Second-order structureness S = sqrt(lambda1^2 + lambda2^2)
        S2 = l1**2 + l2**2
        
        # For bright vessels on dark background, tubular structures have negative l2
        # Vesselness V = exp(-Rb^2 / (2*beta^2)) * (1 - exp(-S^2 / (2*c^2)))
        if not black_ridges:
            condition = l2 < 0.0
        else:
            condition = l2 > 0.0
            
        vesselness = np.where(
            condition,
            np.exp(-(Rb**2) / (2 * (beta**2))) * (1.0 - np.exp(-S2 / (2 * (c**2)))),
            0.0
        )
        
        max_vesselness = np.maximum(max_vesselness, vesselness)
        
    # Normalize output to [0, 1]
    v_max = np.max(max_vesselness)
    if v_max > 0:
        max_vesselness /= v_max
        
    return max_vesselness
