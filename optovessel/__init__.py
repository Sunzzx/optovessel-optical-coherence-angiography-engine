"""OptoVessel: Autonomous Optical Coherence Tomography Angiography (OCTA) Microvascular Hemodynamics & Capillary Perfusion Engine."""
from __future__ import annotations

from optovessel.types import OCTAScanMetadata, PlexusBoundaries, VascularBiomarkers
from optovessel.decorrelation import (
    complex_decorrelation,
    amplitude_decorrelation,
    split_spectrum_amplitude_decorrelation,
    phase_variance_angiography,
)
from optovessel.plexus import segment_retinal_boundaries, project_en_face_plexus
from optovessel.filters import frangi_vesselness_2d, compute_hessian_2d
from optovessel.skeleton import zhang_suen_thinning, analyze_skeleton_nodes, estimate_vessel_calipers
from optovessel.quantification import (
    compute_vascular_biomarkers,
    segment_foveal_avascular_zone,
    detect_non_perfusion_areas,
)
from optovessel.simulation import (
    generate_synthetic_retinal_vasculature,
    generate_synthetic_oct_interferograms,
)
from optovessel.engine import OCTAngiographyEngine

__version__ = "0.1.0"
__all__ = [
    "OCTAScanMetadata",
    "PlexusBoundaries",
    "VascularBiomarkers",
    "complex_decorrelation",
    "amplitude_decorrelation",
    "split_spectrum_amplitude_decorrelation",
    "phase_variance_angiography",
    "segment_retinal_boundaries",
    "project_en_face_plexus",
    "frangi_vesselness_2d",
    "compute_hessian_2d",
    "zhang_suen_thinning",
    "analyze_skeleton_nodes",
    "estimate_vessel_calipers",
    "compute_vascular_biomarkers",
    "segment_foveal_avascular_zone",
    "detect_non_perfusion_areas",
    "generate_synthetic_retinal_vasculature",
    "generate_synthetic_oct_interferograms",
    "OCTAngiographyEngine",
]
