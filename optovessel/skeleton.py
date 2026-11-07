"""Topological thinning, skeletonization, and vascular graph analysis."""
from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt

def zhang_suen_thinning(binary_image: np.ndarray) -> np.ndarray:
    """Zhang-Suen topological thinning algorithm for 2D binary images.
    
    Produces a 1-pixel wide medial skeleton while preserving connectivity and topology.
    
    Args:
        binary_image: 2D boolean or {0, 1} array
        
    Returns:
        2D boolean array representing the morphological skeleton
    """
    skeleton = (binary_image > 0).astype(np.uint8)
    changed = True
    
    while changed:
        changed = False
        for step in (1, 2):
            # Pad image to handle borders
            padded = np.pad(skeleton, ((1, 1), (1, 1)), mode='constant', constant_values=0)
            
            # 8-neighbors (P2 to P9 clockwise starting from north)
            # P9 P2 P3
            # P8 P1 P4
            # P7 P6 P5
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            p1 = padded[1:-1, 1:-1]
            
            # Condition 1: 2 <= B(P1) <= 6 (non-zero neighbors)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            c1 = (b >= 2) & (b <= 6)
            
            # Condition 2: A(P1) == 1 (0->1 transitions in clockwise order)
            # Sequence: P2, P3, P4, P5, P6, P7, P8, P9, P2
            t1 = (p2 == 0) & (p3 == 1)
            t2 = (p3 == 0) & (p4 == 1)
            t3 = (p4 == 0) & (p5 == 1)
            t4 = (p5 == 0) & (p6 == 1)
            t5 = (p6 == 0) & (p7 == 1)
            t6 = (p7 == 0) & (p8 == 1)
            t7 = (p8 == 0) & (p9 == 1)
            t8 = (p9 == 0) & (p2 == 1)
            a = t1.astype(int) + t2 + t3 + t4 + t5 + t6 + t7 + t8
            c2 = (a == 1)
            
            if step == 1:
                # Step 1: P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0
                c3 = (p2 * p4 * p6 == 0)
                c4 = (p4 * p6 * p8 == 0)
            else:
                # Step 2: P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0
                c3 = (p2 * p4 * p8 == 0)
                c4 = (p2 * p6 * p8 == 0)
                
            to_delete = (p1 == 1) & c1 & c2 & c3 & c4
            if np.any(to_delete):
                skeleton[to_delete] = 0
                changed = True
                
    return skeleton.astype(bool)

def analyze_skeleton_nodes(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Analyze 8-connected skeleton nodes to identify endpoints and branching junctions.
    
    Args:
        skeleton: 2D boolean array
        
    Returns:
        (branching_nodes_mask, endpoints_mask, num_branching_nodes, num_endpoints)
    """
    padded = np.pad(skeleton.astype(np.uint8), ((1, 1), (1, 1)), mode='constant', constant_values=0)
    # Count 8-neighbors
    neighbor_count = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2]                    + padded[1:-1, 2:] +
        padded[2:, :-2]  + padded[2:, 1:-1]  + padded[2:, 2:]
    )
    
    skel_mask = skeleton.astype(bool)
    # Endpoints have exactly 1 neighbor
    endpoints = skel_mask & (neighbor_count == 1)
    # Branching junctions have >= 3 neighbors
    branching = skel_mask & (neighbor_count >= 3)
    
    return branching, endpoints, int(np.sum(branching)), int(np.sum(endpoints))

def estimate_vessel_calipers(binary_mask: np.ndarray, skeleton: np.ndarray, pixel_size_um: float = 10.0) -> float:
    """Compute mean vessel caliper (diameter in um) using Euclidean Distance Transform along skeleton."""
    if not np.any(skeleton):
        return 0.0
    edt = distance_transform_edt(binary_mask)
    # Radius in pixels along skeleton, diameter = 2 * radius
    radii_pixels = edt[skeleton]
    diameters_um = 2.0 * radii_pixels * pixel_size_um
    return float(np.mean(diameters_um))
