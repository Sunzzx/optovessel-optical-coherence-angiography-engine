"""Decorrelation and motion contrast algorithms for OCT Angiography."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.signal import windows

def complex_decorrelation(repeated_complex_bscan: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Complex Cross-Decorrelation across repeated B-scans.
    
    Args:
        repeated_complex_bscan: Complex array of shape (N_repeats, Depth, Width)
        eps: Small regularization term to avoid division by zero
        
    Returns:
        Decorrelation array of shape (Depth, Width) with values in [0, 1].
    """
    N, D, W = repeated_complex_bscan.shape
    if N < 2:
        raise ValueError("At least 2 repeated B-scans are required for decorrelation")
    
    decorr_sum = np.zeros((D, W), dtype=np.float64)
    num_pairs = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            s_i = repeated_complex_bscan[i]
            s_j = repeated_complex_bscan[j]
            real_part = np.real(s_i * np.conj(s_j))
            norm_factor = 0.5 * (np.abs(s_i)**2 + np.abs(s_j)**2) + eps
            r_ij = real_part / norm_factor
            d_ij = 1.0 - r_ij
            decorr_sum += np.clip(d_ij, 0.0, 1.0)
            num_pairs += 1
            
    return decorr_sum / num_pairs

def amplitude_decorrelation(repeated_amplitude: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute standard amplitude decorrelation across repeated B-scans."""
    N, D, W = repeated_amplitude.shape
    if N < 2:
        raise ValueError("At least 2 repeated B-scans are required")
    
    decorr_sum = np.zeros((D, W), dtype=np.float64)
    num_pairs = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            a_i = repeated_amplitude[i]
            a_j = repeated_amplitude[j]
            numerator = a_i * a_j
            denominator = 0.5 * (a_i**2 + a_j**2) + eps
            d_ij = 1.0 - (numerator / denominator)
            decorr_sum += np.clip(d_ij, 0.0, 1.0)
            num_pairs += 1
            
    return decorr_sum / num_pairs

def split_spectrum_amplitude_decorrelation(
    complex_interferogram: np.ndarray,
    num_splits: int = 4,
    bandwidth_fraction: float = 0.35,
    eps: float = 1e-12
) -> np.ndarray:
    """Split-Spectrum Amplitude-Decorrelation Angiography (SSADA) algorithm (Jia et al.).
    
    Splits the spectral interferogram along the axial wavenumber (k-space) axis into multiple
    overlapping sub-bands to reduce axial motion decorrelation noise while maintaining lateral
    microvascular resolution.
    
    Args:
        complex_interferogram: Complex spectral or spatial-frequency data (N_repeats, K_axial, Width)
        num_splits: Number of split spectral sub-bands
        bandwidth_fraction: Relative spectral width of each gaussian window
        eps: Regularization factor
        
    Returns:
        SSADA decorrelation signal of shape (Depth, Width) in [0, 1]
    """
    N, K, W = complex_interferogram.shape
    if N < 2:
        raise ValueError("At least 2 repeated B-scans are required for SSADA")
        
    # Sub-band centers along spectral axis
    centers = np.linspace(K * 0.2, K * 0.8, num_splits)
    sigma = (K * bandwidth_fraction) / 2.355  # FWHM relation
    
    band_decorrelations = []
    k_indices = np.arange(K)
    
    for c in centers:
        window = np.exp(-0.5 * ((k_indices - c) / max(sigma, 1.0)) ** 2)[:, np.newaxis]
        # Window each repeat and take inverse FFT to obtain spatial depth profile
        windowed_spectra = complex_interferogram * window[np.newaxis, :, :]
        spatial_bands = np.fft.ifft(windowed_spectra, axis=1)
        amp_bands = np.abs(spatial_bands)
        
        # Amplitude decorrelation across repeats for this sub-band
        band_d = amplitude_decorrelation(amp_bands, eps=eps)
        band_decorrelations.append(band_d)
        
    # Average decorrelation across split spectra
    ssada_d = np.mean(band_decorrelations, axis=0)
    return np.clip(ssada_d, 0.0, 1.0)

def phase_variance_angiography(repeated_complex_bscan: np.ndarray) -> np.ndarray:
    """Phase-Variance Angiography (PVA) based on temporal phase fluctuations.
    
    Args:
        repeated_complex_bscan: Complex array of shape (N_repeats, Depth, Width)
        
    Returns:
        Normalized phase variance map of shape (Depth, Width)
    """
    N, D, W = repeated_complex_bscan.shape
    if N < 2:
        raise ValueError("At least 2 repeated B-scans required")
        
    # Extract instantaneous phases
    phases = np.angle(repeated_complex_bscan)
    # Circular/directional variance
    mean_cos = np.mean(np.cos(phases), axis=0)
    mean_sin = np.mean(np.sin(phases), axis=0)
    r = np.sqrt(mean_cos**2 + mean_sin**2)
    # Circular variance in [0, 1]
    circ_var = 1.0 - r
    return np.clip(circ_var, 0.0, 1.0)
