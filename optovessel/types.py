"""Core data types and structures for OptoVessel OCTA engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class OCTAScanMetadata:
    width_pixels: int
    height_pixels: int
    depth_pixels: int
    repeats_per_bmscan: int
    field_of_view_mm: Tuple[float, float] = (3.0, 3.0)
    axial_resolution_um: float = 3.5
    lateral_resolution_um: float = 12.0
    wavelength_nm: float = 840.0

@dataclass
class PlexusBoundaries:
    ilm_indices: np.ndarray        # Inner limiting membrane depth (H, W)
    ipl_indices: np.ndarray        # Inner plexiform layer depth (H, W)
    opl_indices: np.ndarray        # Outer plexiform layer depth (H, W)
    rpe_indices: np.ndarray        # Retinal pigment epithelium depth (H, W)
    bruchs_indices: Optional[np.ndarray] = None # Bruch's membrane

@dataclass
class VascularBiomarkers:
    vessel_density: float
    vessel_length_density_mm_inv: float
    branching_node_density_mm2: float
    mean_vessel_caliper_um: float
    faz_area_mm2: float
    faz_perimeter_mm: float
    faz_circularity_index: float
    non_perfusion_area_ratio: float
    plexus_name: str = "superficial"
    extra_metrics: Dict[str, float] = field(default_factory=dict)
