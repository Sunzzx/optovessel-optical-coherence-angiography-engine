"""Quantification of vascular biomarkers, FAZ morphometry, and non-perfusion areas (NPA)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import label, binary_fill_holes, distance_transform_edt
from optovessel.types import VascularBiomarkers
from optovessel.skeleton import zhang_suen_thinning, analyze_skeleton_nodes, estimate_vessel_calipers

def segment_foveal_avascular_zone(
    en_face_angiogram: np.ndarray,
    vessel_mask: np.ndarray,
    center_prior: Optional[Tuple[int, int]] = None,
    max_faz_radius_fraction: float = 0.35
) -> Tuple[np.ndarray, float, float, float]:
    """Segment the Foveal Avascular Zone (FAZ) and compute morphometric indices.
    
    Args:
        en_face_angiogram: 2D en-face angiogram (Height, Width)
        vessel_mask: 2D binary vessel mask (1 for vessels, 0 for avascular)
        center_prior: Optional (cy, cx) pixel coordinates of foveal center
        max_faz_radius_fraction: Maximum expected radius of FAZ relative to FOV
        
    Returns:
        (faz_mask, area_pixels, perimeter_pixels, circularity_index)
    """
    H, W = vessel_mask.shape
    if center_prior is None:
        cy, cx = H // 2, W // 2
    else:
        cy, cx = center_prior
        
    # Invert vessel mask to identify avascular spaces
    avascular_regions = (~vessel_mask.astype(bool))
    labeled_regions, num_features = label(avascular_regions)
    
    # Find component closest to foveal center
    target_label = labeled_regions[cy, cx]
    if target_label == 0:
        # Center is on a vessel pixel; find nearest avascular component
        dist = distance_transform_edt(vessel_mask)
        # Mask of region within central radius
        y_grid, x_grid = np.ogrid[:H, :W]
        central_roi = ((y_grid - cy)**2 + (x_grid - cx)**2) <= (min(H, W) * max_faz_radius_fraction)**2
        cand_labels = labeled_regions[central_roi & (labeled_regions > 0)]
        if len(cand_labels) > 0:
            # Pick largest candidate near center
            counts = np.bincount(cand_labels)
            target_label = int(np.argmax(counts))
            
    if target_label > 0:
        faz_mask = (labeled_regions == target_label)
    else:
        faz_mask = np.zeros((H, W), dtype=bool)
        
    # Calculate morphometrics
    area_pixels = float(np.sum(faz_mask))
    
    # Perimeter: border pixels of FAZ mask (4-connected)
    padded = np.pad(faz_mask, 1, mode='constant', constant_values=False)
    eroded = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    boundary = faz_mask & (~eroded)
    perimeter_pixels = float(np.sum(boundary))
    
    # Circularity Index = 4 * pi * Area / (Perimeter^2)
    if perimeter_pixels > 0 and area_pixels > 0:
        circularity = (4.0 * np.pi * area_pixels) / (perimeter_pixels ** 2)
        circularity = min(circularity, 1.0) # Clamp numerical discrete error
    else:
        circularity = 0.0
        
    return faz_mask, area_pixels, perimeter_pixels, circularity

def detect_non_perfusion_areas(
    vessel_mask: np.ndarray,
    faz_mask: np.ndarray,
    min_npa_area_pixels: int = 50
) -> Tuple[np.ndarray, float]:
    """Detect Non-Perfusion Areas (NPA) outside the physiological FAZ.
    
    Pathological capillary dropout in diabetic retinopathy, retinal vein occlusions, and macular ischemia.
    
    Args:
        vessel_mask: 2D binary vessel mask
        faz_mask: Segmented physiological FAZ mask
        min_npa_area_pixels: Minimum pixel size to qualify as a non-perfusion defect
        
    Returns:
        (npa_mask, npa_ratio_excluding_faz)
    """
    # Regions with no vessels and not part of the normal FAZ
    avascular = (~vessel_mask.astype(bool)) & (~faz_mask.astype(bool))
    labeled_npa, num_npa = label(avascular)
    
    npa_mask = np.zeros_like(vessel_mask, dtype=bool)
    for i in range(1, num_npa + 1):
        component_mask = (labeled_npa == i)
        if np.sum(component_mask) >= min_npa_area_pixels:
            npa_mask |= component_mask
            
    # Ratio of retina with ischemic dropout relative to non-FAZ retinal area
    non_faz_area = np.sum(~faz_mask.astype(bool))
    if non_faz_area > 0:
        npa_ratio = float(np.sum(npa_mask) / non_faz_area)
    else:
        npa_ratio = 0.0
        
    return npa_mask, npa_ratio

def compute_vascular_biomarkers(
    en_face_angiogram: np.ndarray,
    vessel_threshold: Optional[float] = None,
    field_of_view_mm: Tuple[float, float] = (3.0, 3.0),
    plexus_name: str = "superficial"
) -> Tuple[VascularBiomarkers, Dict[str, np.ndarray]]:
    """Complete automated quantitative biomarker profiling of retinal microvasculature.
    
    Args:
        en_face_angiogram: 2D angiogram array (Height, Width) in [0, 1]
        vessel_threshold: Optional binarization threshold (Otsu-like if None)
        field_of_view_mm: Physical scan size (Height_mm, Width_mm)
        plexus_name: Retinal plexus identifier
        
    Returns:
        (VascularBiomarkers, dict_of_feature_maps)
    """
    H, W = en_face_angiogram.shape
    fov_h, fov_w = field_of_view_mm
    total_area_mm2 = fov_h * fov_w
    pixel_area_mm2 = total_area_mm2 / (H * W)
    pixel_size_um = (fov_w / W) * 1000.0
    
    # Adaptive thresholding if not provided
    if vessel_threshold is None:
        # Huang/Otsu proxy: mean + 0.3 * std
        vessel_threshold = float(np.mean(en_face_angiogram) + 0.35 * np.std(en_face_angiogram))
        
    binary_vessels = en_face_angiogram > vessel_threshold
    
    # Vessel Density (VD) = area of vessels / total scan area
    vessel_density = float(np.sum(binary_vessels) / (H * W))
    
    # Skeletonization and Vessel Length Density (VLD)
    skeleton = zhang_suen_thinning(binary_vessels)
    vessel_length_pixels = float(np.sum(skeleton))
    vessel_length_mm = vessel_length_pixels * (fov_w / W)
    vessel_length_density_mm_inv = vessel_length_mm / total_area_mm2
    
    # Branching / junction analysis
    branching_mask, endpoints_mask, n_branching, n_endpoints = analyze_skeleton_nodes(skeleton)
    branching_density_mm2 = float(n_branching / total_area_mm2)
    
    # Mean Caliper
    mean_caliper_um = estimate_vessel_calipers(binary_vessels, skeleton, pixel_size_um=pixel_size_um)
    
    # FAZ segmentation
    faz_mask, faz_area_px, faz_perim_px, faz_circularity = segment_foveal_avascular_zone(
        en_face_angiogram, binary_vessels
    )
    faz_area_mm2 = faz_area_px * pixel_area_mm2
    faz_perim_mm = faz_perim_px * (fov_w / W)
    
    # Non-perfusion area
    npa_mask, npa_ratio = detect_non_perfusion_areas(binary_vessels, faz_mask)
    
    biomarkers = VascularBiomarkers(
        vessel_density=vessel_density,
        vessel_length_density_mm_inv=vessel_length_density_mm_inv,
        branching_node_density_mm2=branching_density_mm2,
        mean_vessel_caliper_um=mean_caliper_um,
        faz_area_mm2=faz_area_mm2,
        faz_perimeter_mm=faz_perim_mm,
        faz_circularity_index=faz_circularity,
        non_perfusion_area_ratio=npa_ratio,
        plexus_name=plexus_name,
        extra_metrics={
            "num_endpoints": float(n_endpoints),
            "num_branching_nodes": float(n_branching),
            "total_vessel_length_mm": vessel_length_mm,
        }
    )
    
    maps = {
        "binary_vessels": binary_vessels,
        "skeleton": skeleton,
        "branching_nodes": branching_mask,
        "faz_mask": faz_mask,
        "npa_mask": npa_mask,
    }
    
    return biomarkers, maps
