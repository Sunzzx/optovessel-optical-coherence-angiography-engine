"""Depth-resolved retinal plexus segmentation and en-face projection."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from optovessel.types import PlexusBoundaries

def segment_retinal_boundaries(structural_oct_volume: np.ndarray) -> PlexusBoundaries:
    """Autonomous segmentation of anatomical retinal layer boundaries from structural OCT intensity.
    
    Args:
        structural_oct_volume: 3D intensity array (Height, Depth, Width) or (Depth, Width)
        
    Returns:
        PlexusBoundaries containing (H, W) boundary depth indices.
    """
    if structural_oct_volume.ndim == 2:
        # Expand single B-scan to 3D with H=1
        volume = structural_oct_volume[np.newaxis, :, :]
    else:
        volume = structural_oct_volume
        
    H, D, W = volume.shape
    ilm = np.zeros((H, W), dtype=int)
    ipl = np.zeros((H, W), dtype=int)
    opl = np.zeros((H, W), dtype=int)
    rpe = np.zeros((H, W), dtype=int)
    
    for h in range(H):
        for w in range(W):
            profile = volume[h, :, w]
            # Smooth profile axially
            smooth_profile = gaussian_filter1d(profile, sigma=2.0)
            # Maximum axial gradient indicates vitreo-retinal boundary (ILM)
            diff = np.diff(smooth_profile)
            if len(diff) == 0:
                ilm[h, w] = int(D * 0.15)
                rpe[h, w] = int(D * 0.75)
            else:
                # First prominent positive peak is ILM
                ilm_cand = np.argmax(diff[:int(D * 0.5)]) if int(D * 0.5) > 1 else int(D * 0.15)
                ilm[h, w] = max(ilm_cand, int(D * 0.05))
                
                # Hyper-reflective band near bottom is RPE
                rpe_cand = int(D * 0.4) + np.argmax(smooth_profile[int(D * 0.4):int(D * 0.95)])
                rpe[h, w] = min(max(rpe_cand, ilm[h, w] + 15), D - 2)
                
            # Physiological fractions between ILM and RPE for IPL and OPL
            retinal_thickness = max(rpe[h, w] - ilm[h, w], 10)
            ipl[h, w] = int(ilm[h, w] + 0.35 * retinal_thickness)
            opl[h, w] = int(ilm[h, w] + 0.65 * retinal_thickness)
            
    return PlexusBoundaries(
        ilm_indices=ilm if structural_oct_volume.ndim == 3 else ilm[0],
        ipl_indices=ipl if structural_oct_volume.ndim == 3 else ipl[0],
        opl_indices=opl if structural_oct_volume.ndim == 3 else opl[0],
        rpe_indices=rpe if structural_oct_volume.ndim == 3 else rpe[0],
    )

def project_en_face_plexus(
    octa_volume: np.ndarray,
    boundaries: PlexusBoundaries,
    plexus: str = "superficial",
    projection_mode: str = "max"
) -> np.ndarray:
    """Generate en-face vascular projection for superficial, deep, or avascular retinal plexuses.
    
    Args:
        octa_volume: 3D OCTA volume (Height/SlowAxis, Depth, Width/FastAxis)
        boundaries: Segmented boundary depths
        plexus: 'superficial' (ILM -> IPL), 'deep' (IPL -> OPL), 'avascular' (OPL -> RPE), 'choriocapillaris'
        projection_mode: 'max' for Maximum Intensity Projection (MIP), 'mean' for mean projection
        
    Returns:
        2D en-face angiogram array (Height, Width) in [0, 1]
    """
    if octa_volume.ndim != 3:
        raise ValueError("octa_volume must have 3 dimensions: (Height, Depth, Width)")
        
    H, D, W = octa_volume.shape
    en_face = np.zeros((H, W), dtype=np.float64)
    
    ilm = np.atleast_2d(boundaries.ilm_indices)
    ipl = np.atleast_2d(boundaries.ipl_indices)
    opl = np.atleast_2d(boundaries.opl_indices)
    rpe = np.atleast_2d(boundaries.rpe_indices)
    
    for h in range(H):
        for w in range(W):
            if plexus == "superficial":
                z_start = max(0, ilm[h, w])
                z_end = max(z_start + 1, min(D, ipl[h, w]))
            elif plexus == "deep":
                z_start = max(0, ipl[h, w])
                z_end = max(z_start + 1, min(D, opl[h, w]))
            elif plexus == "avascular":
                z_start = max(0, opl[h, w])
                z_end = max(z_start + 1, min(D, rpe[h, w]))
            elif plexus == "full_retina":
                z_start = max(0, ilm[h, w])
                z_end = max(z_start + 1, min(D, rpe[h, w]))
            else:
                raise ValueError(f"Unknown plexus type: {plexus}")
                
            slab = octa_volume[h, z_start:z_end, w]
            if len(slab) == 0:
                en_face[h, w] = 0.0
            elif projection_mode == "max":
                en_face[h, w] = np.max(slab)
            elif projection_mode == "mean":
                en_face[h, w] = np.mean(slab)
            else:
                raise ValueError(f"Unknown projection mode: {projection_mode}")
                
    return en_face
