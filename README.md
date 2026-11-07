# OptoVessel: Optical Coherence Tomography Angiography (OCTA) Microvascular Hemodynamics & Capillary Perfusion Engine

[![CI](https://github.com/Sunzzx/optovessel-optical-coherence-angiography-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/optovessel-optical-coherence-angiography-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **OptoVessel** is an autonomous, high-throughput computational engine for **Optical Coherence Tomography Angiography (OCTA)**. It provides end-to-end processing of repeated spectral-domain/swept-source OCT scans into depth-resolved retinal microvascular angiograms, multi-scale Frangi vesselness enhancements, topological skeletonization, and automated quantitative biomarker profiling (Vessel Density, Vessel Length Density, Foveal Avascular Zone morphometry, and Ischemic Non-Perfusion Areas).

---

## 1. Humanity & Real-World Clinical Impact

Retinal microvascular compromise is the earliest clinical hallmark of blinding neurodegenerative and systemic vascular diseases:
1. **Diabetic Retinopathy (DR):** Capillary non-perfusion, microaneurysms, and foveal avascular zone (FAZ) enlargement precede irreversible vision loss.
2. **Age-Related Macular Degeneration (AMD):** Choroidal neovascularization (CNV) disrupts normal choriocapillaris architecture.
3. **Retinal Vein Occlusion (RVO) & Glaucoma:** Microvascular dropout directly correlates with retinal nerve fiber layer (RNFL) axonal death.
4. **Systemic Biomarkers (Alzheimer's & Cardiovascular Disease):** Retinal microvasculature serves as the sole non-invasive in-vivo optical window into cerebral and systemic microvascular health.

**OptoVessel** eliminates reliance on expensive proprietary imaging workstations ($150,000+ per scanner) and closed vendor lock-in by implementing an open, mathematically rigorous, reproducible framework for high-resolution retinal capillary hemodynamics at zero infrastructure cost.

---

## 2. Theoretical Mathematical Foundations

### A. Split-Spectrum Amplitude-Decorrelation Angiography (SSADA)
Traditional amplitude decorrelation across repeated B-scans suffers from axial motion artifacts (bulk eye motion) which degrade microvascular SNR. SSADA decomposes the full spectral interferogram $I(k, x)$ into $M$ overlapping Gaussian spectral sub-bands:

$$I_m(k, x) = I(k, x) \cdot \exp \left(-\frac{(k - k_{0,m})^2}{2 \sigma_k^2}\right)$$

Taking the inverse Fourier transform yields spatial amplitude profiles $A_m(z, x, t_n)$ for repeat $n \in \{1, \dots, N\}$:

$$D_m(z, x) = 1 - \frac{1}{N-1} \sum_{n=1}^{N-1} \frac{A_m(z, x, t_n) A_m(z, x, t_{n+1})}{\frac{1}{2} \left(A_m(z, x, t_n)^2 + A_m(z, x, t_{n+1})^2\right) + \epsilon}$$

The final decorrelation signal is obtained by averaging across all split sub-bands:

$$\bar{D}(z, x) = \frac{1}{M} \sum_{m=1}^M D_m(z, x)$$

### B. Multiscale Frangi Vesselness Filter
To enhance tubular capillary structures while suppressing speckle background noise, we compute the scale-space Hessian matrix $H_\sigma(x, y)$:

$$H_\sigma = \begin{bmatrix} I_{xx} * G_\sigma & I_{xy} * G_\sigma \\ I_{xy} * G_\sigma & I_{yy} * G_\sigma \end{bmatrix}$$

With eigenvalues $|\lambda_1| \le |\lambda_2|$, the 2D vesselness function is given by:

$$V_\sigma = \begin{cases} 0 & \text{if } \lambda_2 > 0 \\ \exp\left(-\frac{R_B^2}{2\beta^2}\right) \left(1 - \exp\left(-\frac{S^2}{2c^2}\right)\right) & \text{otherwise} \end{cases}$$

where $R_B = \frac{|\lambda_1|}{|\lambda_2|}$ measures circular blobness and $S = \sqrt{\lambda_1^2 + \lambda_2^2}$ measures second-order structureness.

### C. Foveal Avascular Zone (FAZ) Morphometry
Circularity index measures how closely the FAZ contour resembles a perfect unit circle:

$$C_{\text{FAZ}} = \frac{4 \pi \cdot \text{Area}_{\text{FAZ}}}{\text{Perimeter}_{\text{FAZ}}^2}$$

---

## 3. Computational Architecture

```
+----------------------------------------------------------------------+
|                  OptoVessel Autonomous Pipeline                      |
+----------------------------------------------------------------------+
      |
      | [4D Spectral Volume: B-scan repeats (H x N_rep x Depth_k x W)]
      v
+----------------------------------------------------------------------+
| 1. Motion Contrast & Hemodynamic Decorrelation                       |
|    - SSADA (Split-Spectrum Gaussian sub-bands)                       |
|    - Complex Decorrelation & Phase-Variance Angiography              |
+----------------------------------------------------------------------+
      |
      | [3D OCTA Volume + 3D Structural Intensity Volume]
      v
+----------------------------------------------------------------------+
| 2. Anatomical Layer Boundary Segmentation                            |
|    - ILM (Inner Limiting Membrane)                                   |
|    - IPL / OPL (Inner & Outer Plexiform Layers)                      |
|    - RPE (Retinal Pigment Epithelium)                                |
+----------------------------------------------------------------------+
      |
      | [Depth-Resolved Plexus En-Face Slabs]
      v
+----------------------------------------------------------------------+
| 3. En-Face Slabs & Multiscale Hessian Filtering                      |
|    - Superficial Capillary Plexus (SCP: ILM -> IPL)                  |
|    - Deep Capillary Plexus (DCP: IPL -> OPL)                         |
|    - Avascular Retinal Zone (OPL -> RPE)                             |
+----------------------------------------------------------------------+
      |
      | [Denoised High-Contrast Retinal Angiograms]
      v
+----------------------------------------------------------------------+
| 4. Topological Skeletonization & Biomarker Extraction                |
|    - Zhang-Suen Thinning & Branching Junction Analysis               |
|    - Vessel Density (VD) & Vessel Length Density (VLD)               |
|    - FAZ Area, Perimeter & Circularity Index                         |
|    - Ischemic Non-Perfusion Area (NPA) Detection                     |
+----------------------------------------------------------------------+
```

---

## 4. Formal Complexity Analysis

| Pipeline Component | Time Complexity | Space Complexity | Description |
| :--- | :--- | :--- | :--- |
| **SSADA Decomposition** | $O(M \cdot N \cdot D \log D \cdot W)$ | $O(M \cdot D \cdot W)$ | Multi-band IFFT across $M$ sub-bands and $N$ repeats |
| **Retinal Boundary Segmentation** | $O(H \cdot D \cdot W)$ | $O(H \cdot W)$ | Axial 1D gaussian gradient peak tracking |
| **En-Face MIP Projection** | $O(H \cdot D_{\text{slab}} \cdot W)$ | $O(H \cdot W)$ | Slab-restricted maximum intensity search |
| **Multiscale Frangi Filter** | $O(K_{\sigma} \cdot H \cdot W)$ | $O(H \cdot W)$ | Scale-space Hessian decomposition and eigensolving |
| **Zhang-Suen Skeletonization** | $O(I_{\text{iter}} \cdot H \cdot W)$ | $O(H \cdot W)$ | 8-connected topological thinning ($I \ll \min(H, W)$) |

---

## 5. Quickstart & Interactive Terminal Demo

### Installation
```bash
git clone https://github.com/Sunzzx/optovessel-optical-coherence-angiography-engine.git
cd optovessel-optical-coherence-angiography-engine
uv pip install -e ".[dev]"
```

### Running the Full Pipeline Demo
```bash
uv run python -m optovessel.demo
```

### Running the Test Suite
```bash
uv run pytest -v
```

---

## 6. Commercial Licensing & Sponsorship

This project is distributed under the **MIT License** for academic, clinical research, and open-source applications.

- **Commercial Inquiries & Custom Hardware Integrations:** Contact `ayushkandwal2007@gmail.com`
- **Support the Research:** [GitHub Sponsors](https://github.com/sponsors/Sunzzx)
