"""Comprehensive test suite for OptoVessel OCTA engine."""
from __future__ import annotations
import pytest
import numpy as np

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
    segment_foveal_avascular_zone,
    detect_non_perfusion_areas,
    compute_vascular_biomarkers,
)
from optovessel.simulation import (
    generate_synthetic_retinal_vasculature,
    generate_synthetic_oct_interferograms,
)
from optovessel.engine import OCTAngiographyEngine


def test_complex_decorrelation_invariance():
    # Identical repeats should yield 0 decorrelation
    static_scan = np.ones((3, 32, 32), dtype=np.complex128)
    d = complex_decorrelation(static_scan)
    assert d.shape == (32, 32)
    assert np.allclose(d, 0.0, atol=1e-5)

    # Orthogonal repeats should yield maximum decorrelation (~1.0)
    ortho_scan = np.zeros((2, 16, 16), dtype=np.complex128)
    ortho_scan[0, :, :] = 1.0
    ortho_scan[1, :, :] = 1j
    d_ortho = complex_decorrelation(ortho_scan)
    assert np.allclose(d_ortho, 1.0, atol=1e-5)


def test_amplitude_decorrelation_bounds():
    rng = np.random.RandomState(42)
    amp = np.abs(rng.normal(1.0, 0.2, size=(4, 20, 20)))
    d = amplitude_decorrelation(amp)
    assert d.shape == (20, 20)
    assert np.all(d >= 0.0) and np.all(d <= 1.0)


def test_split_spectrum_amplitude_decorrelation():
    # Synthetic interferogram with 4 repeats
    interferograms = np.ones((4, 32, 32), dtype=np.complex128)
    ssada_d = split_spectrum_amplitude_decorrelation(interferograms, num_splits=3)
    assert ssada_d.shape == (32, 32)
    assert np.all(ssada_d >= 0.0) and np.all(ssada_d <= 1.0)


def test_phase_variance_angiography():
    # Constant phase should have zero circular variance
    const_phase = np.exp(1j * 0.5 * np.ones((4, 16, 16)))
    pv = phase_variance_angiography(const_phase)
    assert np.allclose(pv, 0.0, atol=1e-5)

    # Random distributed phase should have high variance
    rng = np.random.RandomState(42)
    rand_phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(20, 16, 16)))
    pv_rand = phase_variance_angiography(rand_phase)
    assert np.mean(pv_rand) > 0.5


def test_retinal_boundary_segmentation():
    # Synthetic structural volume with distinct high reflectivity RPE
    vol = np.zeros((10, 40, 10))
    vol[:, 5:10, :] = 0.5    # ILM transition
    vol[:, 25:30, :] = 1.0   # RPE layer
    boundaries = segment_retinal_boundaries(vol)
    assert boundaries.ilm_indices.shape == (10, 10)
    assert boundaries.rpe_indices.shape == (10, 10)
    assert np.all(boundaries.ilm_indices < boundaries.ipl_indices)
    assert np.all(boundaries.ipl_indices <= boundaries.opl_indices)
    assert np.all(boundaries.opl_indices < boundaries.rpe_indices)


def test_en_face_projection():
    octa_vol = np.zeros((8, 30, 8))
    octa_vol[:, 5:12, :] = 0.8 # Superficial signal
    octa_vol[:, 15:20, :] = 0.4 # Deep signal
    
    boundaries = PlexusBoundaries(
        ilm_indices=np.full((8, 8), 4),
        ipl_indices=np.full((8, 8), 13),
        opl_indices=np.full((8, 8), 22),
        rpe_indices=np.full((8, 8), 28),
    )
    
    sup_proj = project_en_face_plexus(octa_vol, boundaries, plexus="superficial", projection_mode="max")
    deep_proj = project_en_face_plexus(octa_vol, boundaries, plexus="deep", projection_mode="max")
    assert sup_proj.shape == (8, 8)
    assert np.allclose(sup_proj, 0.8)
    assert np.allclose(deep_proj, 0.4)


def test_frangi_vesselness_filter():
    img = np.zeros((30, 30))
    # Create a vertical vessel line in the middle
    img[:, 15] = 1.0
    vesselness = frangi_vesselness_2d(img, sigmas=(1.0, 2.0))
    assert vesselness.shape == (30, 30)
    assert np.all(vesselness >= 0.0) and np.all(vesselness <= 1.0)
    # Response along the line should be significantly higher than background
    assert np.mean(vesselness[:, 15]) > np.mean(vesselness[:, 0])


def test_zhang_suen_thinning_topology():
    # Solid vertical bar of width 5
    bar = np.zeros((20, 20), dtype=bool)
    bar[3:17, 8:13] = True
    skel = zhang_suen_thinning(bar)
    assert skel.dtype == bool
    # Skeleton should be 1-pixel wide
    assert np.sum(skel) < np.sum(bar)
    assert np.all(np.sum(skel, axis=1)[3:17] == 1)


def test_skeleton_node_analysis():
    # T-junction shape
    t_shape = np.zeros((15, 15), dtype=bool)
    t_shape[7, 3:12] = True # Horizontal bar
    t_shape[7:13, 7] = True # Vertical stem downwards
    
    branching, endpoints, n_branch, n_end = analyze_skeleton_nodes(t_shape)
    assert n_branch >= 1
    assert n_end == 3 # 3 tips
    assert branching[7, 7] # Center is the branch point


def test_faz_segmentation_morphometry():
    # Vessel image with central circular hole
    vessel_mask = np.ones((40, 40), dtype=bool)
    y, x = np.ogrid[:40, :40]
    faz_disk = ((y - 20)**2 + (x - 20)**2) <= 36 # radius 6
    vessel_mask[faz_disk] = False
    
    faz_mask, area_px, perim_px, circ = segment_foveal_avascular_zone(vessel_mask.astype(float), vessel_mask)
    assert np.sum(faz_mask) > 0
    assert area_px == pytest.approx(np.sum(faz_disk), rel=0.05)
    assert circ > 0.6  # High circularity for disk


def test_non_perfusion_area_detection():
    vessel_mask = np.ones((50, 50), dtype=bool)
    faz_mask = np.zeros((50, 50), dtype=bool)
    faz_mask[20:30, 20:30] = True
    vessel_mask[faz_mask] = False
    
    # Add ischemic capillary dropout patch (NPA)
    vessel_mask[5:15, 5:15] = False # 100 pixels
    npa_mask, ratio = detect_non_perfusion_areas(vessel_mask, faz_mask, min_npa_area_pixels=40)
    assert np.sum(npa_mask) >= 100
    assert ratio > 0.0


def test_biomarker_pipeline_end_to_end():
    en_face = np.zeros((32, 32))
    en_face[16, :] = 1.0
    en_face[:, 16] = 1.0
    bm, maps = compute_vascular_biomarkers(en_face, vessel_threshold=0.5, field_of_view_mm=(3.0, 3.0))
    assert isinstance(bm, VascularBiomarkers)
    assert bm.vessel_density > 0.0
    assert bm.vessel_length_density_mm_inv > 0.0
    assert "binary_vessels" in maps
    assert "skeleton" in maps


def test_hemodynamic_phantom_generator():
    vasc = generate_synthetic_retinal_vasculature(width=64, height=64, seed=42, include_faz=True)
    assert vasc.shape == (64, 64)
    assert np.all(vasc >= 0.0) and np.all(vasc <= 1.0)
    # Center FAZ should be 0
    assert vasc[32, 32] == 0.0


def test_oct_interferogram_simulation():
    v_slice = np.zeros((1, 32))
    v_slice[0, 16] = 1.0
    interf = generate_synthetic_oct_interferograms(v_slice, repeats=3, depth_k=24, seed=99)
    assert interf.shape == (3, 24, 32)
    assert np.iscomplexobj(interf)


def test_octa_engine_full_volume_execution():
    # Mini volume: 4 B-scans, 2 repeats, 16 depths, 16 widths
    vol = np.zeros((4, 2, 16, 16), dtype=np.complex128)
    vol[:, :, 4:12, :] = 1.0 + 0.5j
    engine = OCTAngiographyEngine()
    en_face, bm, boundaries = engine.process_volume(vol, method="ssada", enhance_vessels=False)
    assert "superficial" in en_face
    assert "deep" in en_face
    assert "avascular" in en_face
    assert isinstance(bm["superficial"], VascularBiomarkers)
    assert boundaries.ilm_indices.shape == (4, 16)
