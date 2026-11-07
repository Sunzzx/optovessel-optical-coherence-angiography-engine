"""Interactive terminal CLI demo for OptoVessel OCTA engine."""
from __future__ import annotations
import time
import numpy as np
from optovessel.simulation import generate_synthetic_retinal_vasculature, generate_synthetic_oct_interferograms
from optovessel.engine import OCTAngiographyEngine
from optovessel.types import OCTAScanMetadata

def render_ascii_image(image: np.ndarray, width: int = 50, height: int = 25) -> str:
    """Render 2D intensity map into compact ASCII art visualization."""
    chars = " .:-=+*#%@"
    H, W = image.shape
    # Downsample
    y_idx = (np.linspace(0, H - 1, height)).astype(int)
    x_idx = (np.linspace(0, W - 1, width)).astype(int)
    sampled = image[np.ix_(y_idx, x_idx)]
    
    # Normalize
    s_min, s_max = np.min(sampled), np.max(sampled)
    if s_max > s_min:
        norm = (sampled - s_min) / (s_max - s_min)
    else:
        norm = np.zeros_like(sampled)
        
    lines = []
    for row in norm:
        line = "".join(chars[min(int(val * (len(chars) - 1)), len(chars) - 1)] for val in row)
        lines.append(line)
    return "\n".join(lines)

def main() -> None:
    print("=" * 70)
    print("👁️  OPTOVESSEL: OPTICAL COHERENCE TOMOGRAPHY ANGIOGRAPHY (OCTA) ENGINE")
    print("    Autonomous Microvascular Hemodynamics & Capillary Perfusion Analysis")
    print("=" * 70)
    
    print("\n[1/5] Synthesizing physiological retinal microvascular ground truth (128x128)...")
    gt_vasculature = generate_synthetic_retinal_vasculature(width=128, height=128, seed=42, include_faz=True)
    print(f"      Generated arcade vessels with central Foveal Avascular Zone (FAZ).")
    
    print("\n[2/5] Simulating repeated 4D spectral OCT interferogram volume (BM-scan)...")
    start_sim = time.time()
    # Synthesize volume: 16 B-scans along slow axis, 4 repeats each, 64 k-space depths, 128 fast axis
    H, N_rep, D, W = 16, 4, 64, 128
    spectral_vol = np.zeros((H, N_rep, D, W), dtype=np.complex128)
    for h in range(H):
        slice_y = int((h / H) * 128)
        v_slice = gt_vasculature[slice_y:slice_y + 1, :] # 1x128
        spectral_vol[h] = generate_synthetic_oct_interferograms(
            v_slice, repeats=N_rep, depth_k=D, doppler_shift=0.9, noise_sigma=0.06, seed=h + 100
        )
    print(f"      Generated 4D spectral volume ({H}x{N_rep}x{D}x{W}) in {time.time() - start_sim:.2f}s")
    
    print("\n[3/5] Executing Split-Spectrum Amplitude-Decorrelation (SSADA) & Layer Segmentation...")
    meta = OCTAScanMetadata(
        width_pixels=W,
        height_pixels=H,
        depth_pixels=D,
        repeats_per_bmscan=N_rep,
        field_of_view_mm=(3.0, 3.0)
    )
    engine = OCTAngiographyEngine(metadata=meta)
    start_proc = time.time()
    en_face_maps, biomarkers, boundaries = engine.process_volume(spectral_vol, method="ssada", enhance_vessels=True)
    print(f"      OCTA Volume processed in {time.time() - proc_time if 'proc_time' in locals() else time.time() - start_proc:.2f}s")
    
    sup_bm = biomarkers["superficial"]
    print("\n[4/5] Quantitative Retinal Biomarker Morphometry Results (Superficial Plexus):")
    print(f"      • Vessel Density (VD):              {sup_bm.vessel_density * 100:.2f} %")
    print(f"      • Vessel Length Density (VLD):     {sup_bm.vessel_length_density_mm_inv:.2f} mm^-1")
    print(f"      • Branching Node Density:          {sup_bm.branching_node_density_mm2:.2f} junctions/mm²")
    print(f"      • Mean Vessel Caliper:             {sup_bm.mean_vessel_caliper_um:.2f} µm")
    print(f"      • FAZ Area:                        {sup_bm.faz_area_mm2:.3f} mm²")
    print(f"      • FAZ Circularity Index:           {sup_bm.faz_circularity_index:.3f}")
    print(f"      • Non-Perfusion Area (NPA) Ratio:  {sup_bm.non_perfusion_area_ratio * 100:.2f} %")
    
    print("\n[5/5] ASCII Visualization: Superficial Capillary Plexus En-Face Angiogram:")
    print("-" * 52)
    ascii_vis = render_ascii_image(en_face_maps["superficial"], width=50, height=14)
    print(ascii_vis)
    print("-" * 52)
    print("\n✨ OptoVessel autonomous pipeline executed successfully with 0 errors.")

if __name__ == "__main__":
    main()
