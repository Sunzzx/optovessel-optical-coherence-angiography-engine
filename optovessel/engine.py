"""Autonomous High-Throughput OCTA End-to-End Processing Engine."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from optovessel.types import OCTAScanMetadata, PlexusBoundaries, VascularBiomarkers
from optovessel.decorrelation import complex_decorrelation, split_spectrum_amplitude_decorrelation, phase_variance_angiography
from optovessel.plexus import segment_retinal_boundaries, project_en_face_plexus
from optovessel.filters import frangi_vesselness_2d
from optovessel.quantification import compute_vascular_biomarkers

class OCTAngiographyEngine:
    """Full-stack OCTA processing pipeline integrating SSADA, depth segmentation, and biomarker morphometry."""
    
    def __init__(self, metadata: Optional[OCTAScanMetadata] = None) -> None:
        if metadata is None:
            self.metadata = OCTAScanMetadata(
                width_pixels=128,
                height_pixels=128,
                depth_pixels=64,
                repeats_per_bmscan=4,
                field_of_view_mm=(3.0, 3.0)
            )
        else:
            self.metadata = metadata
            
    def process_bscan_decorrelation(
        self,
        repeated_spectral_data: np.ndarray,
        method: str = "ssada"
    ) -> np.ndarray:
        """Compute 2D cross-sectional angiogram for a single repeated B-scan.
        
        Args:
            repeated_spectral_data: Complex array (repeats, depth_k, width)
            method: 'ssada', 'complex_decorr', or 'phase_variance'
            
        Returns:
            2D decorrelation map (depth, width) in [0, 1]
        """
        if method == "ssada":
            return split_spectrum_amplitude_decorrelation(repeated_spectral_data)
        elif method == "complex_decorr":
            # First transform to spatial depth
            spatial_complex = np.fft.ifft(repeated_spectral_data, axis=1)
            return complex_decorrelation(spatial_complex)
        elif method == "phase_variance":
            spatial_complex = np.fft.ifft(repeated_spectral_data, axis=1)
            return phase_variance_angiography(spatial_complex)
        else:
            raise ValueError(f"Unsupported decorrelation method: {method}")
            
    def process_volume(
        self,
        spectral_volume: np.ndarray,
        method: str = "ssada",
        enhance_vessels: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, VascularBiomarkers], PlexusBoundaries]:
        """Process full 4D OCT dataset (Height, Repeats, Depth, Width) into en-face angiograms & biomarkers.
        
        Args:
            spectral_volume: Array of shape (H, N_repeats, D, W)
            method: Decorrelation algorithm ('ssada', 'complex_decorr')
            enhance_vessels: Whether to apply Frangi multiscale filtering
            
        Returns:
            (en_face_maps, biomarkers_per_plexus, segmented_boundaries)
        """
        H, N, D, W = spectral_volume.shape
        octa_volume = np.zeros((H, D, W), dtype=np.float64)
        structural_volume = np.zeros((H, D, W), dtype=np.float64)
        
        for h in range(H):
            bscan_repeats = spectral_volume[h]
            # Compute angiographic decorrelation
            octa_volume[h] = self.process_bscan_decorrelation(bscan_repeats, method=method)
            
            # Structural intensity is the mean amplitude
            spatial = np.fft.ifft(bscan_repeats, axis=1)
            structural_volume[h] = np.mean(np.abs(spatial), axis=0)
            
        # Retinal boundary segmentation
        boundaries = segment_retinal_boundaries(structural_volume)
        
        en_face_maps = {}
        biomarkers = {}
        
        # Extract Superficial and Deep Plexuses
        for plexus in ["superficial", "deep", "avascular"]:
            en_face = project_en_face_plexus(octa_volume, boundaries, plexus=plexus, projection_mode="max")
            
            if enhance_vessels and plexus in ["superficial", "deep"]:
                en_face = frangi_vesselness_2d(en_face, sigmas=(1.0, 1.5, 2.0))
                
            en_face_maps[plexus] = en_face
            
            bm, _ = compute_vascular_biomarkers(
                en_face,
                field_of_view_mm=self.metadata.field_of_view_mm,
                plexus_name=plexus
            )
            biomarkers[plexus] = bm
            
        return en_face_maps, biomarkers, boundaries
