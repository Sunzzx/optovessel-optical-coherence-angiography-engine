"""Physiologically grounded retinal microvascular hemodynamic phantom synthesis."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from optovessel.types import OCTAScanMetadata, PlexusBoundaries

def generate_synthetic_retinal_vasculature(
    width: int = 128,
    height: int = 128,
    seed: int = 42,
    include_faz: bool = True,
    faz_radius: float = 14.0
) -> np.ndarray:
    """Synthesize 2D fractal-like bifurcating retinal vascular tree with central FAZ.
    
    Args:
        width: Width in pixels
        height: Height in pixels
        seed: RNG seed for reproducible generation
        include_faz: Whether to hollow out a central foveal avascular zone
        faz_radius: Radius in pixels of the FAZ
        
    Returns:
        2D binary or continuous blood vessel flow map in [0, 1].
    """
    rng = np.random.RandomState(seed)
    vessel_map = np.zeros((height, width), dtype=np.float64)
    cy, cx = height // 2, width // 2
    
    # Radial grid
    y_grid, x_grid = np.ogrid[:height, :width]
    dist_from_fovea = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
    
    # Primary arcade branches radiating outward
    num_arcades = 4
    angles = [0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi]
    for base_theta in angles:
        # Trace arcades
        steps = int(min(width, height) * 0.45)
        theta = base_theta
        r = faz_radius + 2.0
        for s in range(steps):
            # random angular walk
            theta += rng.normal(0, 0.05)
            r += 1.0
            py = int(cy + r * np.sin(theta))
            px = int(cx + r * np.cos(theta))
            if 0 <= py < height and 0 <= px < width:
                # Primary vessel thickness (3 pixels)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            vessel_map[ny, nx] = 1.0
                            
            # Bifurcate daughter capillaries
            if s % 12 == 0 and s > 8:
                sub_theta = theta + rng.choice([-0.6, 0.6])
                sub_r = r
                for sub_s in range(15):
                    sub_r += 0.8
                    sub_theta += rng.normal(0, 0.08)
                    spy = int(cy + sub_r * np.sin(sub_theta))
                    spx = int(cx + sub_r * np.cos(sub_theta))
                    if 0 <= spy < height and 0 <= spx < width:
                        vessel_map[spy, spx] = 0.8
                        
    # Interconnected capillary meshwork
    noise = rng.uniform(0, 1, size=(height, width))
    capillaries = (noise > 0.955).astype(np.float64)
    vessel_map = np.maximum(vessel_map, capillaries)
    
    # Apply FAZ mask
    if include_faz:
        faz_mask = dist_from_fovea <= faz_radius
        vessel_map[faz_mask] = 0.0
        
    return vessel_map

def generate_synthetic_oct_interferograms(
    vessel_slice: np.ndarray,
    repeats: int = 4,
    depth_k: int = 64,
    doppler_shift: float = 0.8,
    noise_sigma: float = 0.08,
    seed: int = 123
) -> np.ndarray:
    """Generate repeated spectral-domain OCT interferograms across time for a B-scan slice.
    
    Args:
        vessel_slice: 2D slice (Depth, Width) indicating presence of moving RBCs
        repeats: Number of B-scan repeats (BM-scan)
        depth_k: Spectral k-space resolution
        doppler_shift: Phase shift introduced by erythrocyte transit velocity
        noise_sigma: Additive complex gaussian noise
        seed: RNG seed
        
    Returns:
        Complex array of shape (repeats, depth_k, Width)
    """
    rng = np.random.RandomState(seed)
    D, W = vessel_slice.shape
    interferograms = np.zeros((repeats, depth_k, W), dtype=np.complex128)
    
    # Static tissue spatial reflectance profile
    static_reflectance = np.zeros((depth_k, W), dtype=np.float64)
    # Anatomical layers
    static_reflectance[int(depth_k * 0.15):int(depth_k * 0.75), :] = 0.4
    static_reflectance[int(depth_k * 0.70):int(depth_k * 0.75), :] = 0.9 # RPE peak
    
    # Interpolate vessel slice to depth_k
    vessel_layer = np.zeros((depth_k, W), dtype=np.float64)
    d_start = int(depth_k * 0.20)
    d_end = int(depth_k * 0.45)
    vessel_resized = vessel_slice[:min(D, d_end - d_start), :]
    vessel_layer[d_start:d_start + vessel_resized.shape[0], :] = vessel_resized
    
    for rep in range(repeats):
        # Dynamic phase shift from moving red blood cells (erythrocyte hemodynamics)
        flow_phase = (rep * doppler_shift) % (2 * np.pi)
        dynamic_signal = vessel_layer * np.exp(1j * flow_phase)
        
        # Total spatial reflectivity profile
        total_spatial = static_reflectance + dynamic_signal
        
        # Additive speckle & measurement noise
        noise = (rng.normal(0, noise_sigma, size=(depth_k, W)) +
                 1j * rng.normal(0, noise_sigma, size=(depth_k, W)))
        noisy_spatial = total_spatial + noise
        
        # Optical interferogram is the Fourier transform of the axial spatial structure
        interferograms[rep] = np.fft.fft(noisy_spatial, axis=0)
        
    return interferograms
