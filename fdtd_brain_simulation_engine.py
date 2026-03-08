"""
3D FDTD brain segmentation simulation engine.

Same implementation, pattern, and parameters as 3D_FDTD_brain_segmentation.py.
Stores all simulation videos and visualizations inside the `results/` folder.

Usage:
  # From existing segmentation NIfTI:
  python fdtd_brain_simulation_engine.py --seg brain_segmentation.nii
  python fdtd_brain_simulation_engine.py brain_segmentation.nii   # positional still supported

  # From 4 BraTS modality NIfTIs (runs 3D U-Net segmentation first):
  python fdtd_brain_simulation_engine.py --modalities flair.nii t1.nii t1ce.nii t2.nii [--checkpoint path.pth]
  # From a single folder containing the four modalities (auto-detects filenames):
  python fdtd_brain_simulation_engine.py --modalities-dir dataset/validation_data/001 [--checkpoint path.pth]

  # Antenna optimization (4-quadrant APA, Houle Ch.6 style; maximizes SAR tumor/healthy ratio):
  # With --optimize-antenna, the main FDTD run uses the optimized 4-quadrant source (Option A);
  # all SAR, temperature, E-frames, and animations are from this optimized configuration.
  python fdtd_brain_simulation_engine.py --seg brain_segmentation.nii --optimize-antenna [--f0 100e6 ...]

CLI arguments:
  seg (positional, optional)
      Path to BraTS-style segmentation NIfTI (0,1,2,3). Omit if using --modalities or --modalities-dir.
  --modalities FLAIR T1 T1CE T2
      Paths to 4 BraTS NIfTI files. Runs 3D U-Net segmentation then FDTD.
  --modalities-dir DIR
      Folder containing four BraTS modalities (flair.nii, t1.nii, t1ce.nii, t2.nii or *_flair.nii, etc.).
  --checkpoint PATH
      Path to 3D U-Net checkpoint .pth (default: best_model.pth).
  --no-normal-brain
      When using --modalities, do not add normal brain tissue (label 4); keep only tumor classes 1–3 and background 0.

  Antenna optimization:
  --optimize-antenna
      Run 4-quadrant APA antenna optimization to maximize J = meanSAR_tumor / meanSAR_healthy.
  --f0 FREQ
      Operating frequency in Hz for optimization CW source (default: 100e6).
  --opt-time-steps N
      FDTD time steps per unit-quadrant run (default: 700).
  --opt-phase-steps N
      Phase grid points per quadrant in coarse sweep (default: 24).
  --opt-amp-steps N
      Amplitude grid points per quadrant in coarse sweep (default: 9).
  --opt-amp-min, --opt-amp-max
      Amplitude bounds (default: 0.2, 2.5).
  --opt-refine-iters N
      Coordinate-descent refinement iterations (default: 8).
  --opt-multi-start N
      Number of random multi-start phase offsets (default: 3).
  --opt-freq-sweep FREQ [FREQ ...]
      Frequencies to sweep; best is auto-selected (e.g. --opt-freq-sweep 70e6 100e6 130e6 170e6 200e6).
  --opt-geom-offsets OFFSET [OFFSET ...]
      Source ring offsets (cells from PML) to sweep; small offsets keep applicator in air.
      Default: 8 10 12.
  --opt-geom-zplanes Z [Z ...]
      Z-plane indices for source placement sweep (e.g. --opt-geom-zplanes 30 41 50).
  --opt-penalty-weight W
      Penalty weight for healthy P95 SAR hotspot in objective (default: 0.1).
  --opt-source-scale S
      Global scale for optimized source in final FDTD run; SAR scales with S² (default: 1.0).
  --opt-parallel N
      Parallel workers for antenna optimization (frequency/geometry sweep, multi-start). Default: 1.
  --stream-frames
      Stream E and SAR frames to disk during FDTD (no in-memory accumulation). Use with
      --stream-frame-interval for full or dense timesteps; build animations separately from saved frames.
      After the run, build_animations_from_streamed_frames.py is invoked to build MP4s; use --slice-timestep-images to also generate per-(slice, timestep) PNGs (E/SAR/T) for the dashboard.
  --no-stream-frames
      Keep E and SAR frames in memory instead of streaming to disk (disables default streaming).
  --stream-frame-interval N
      Save a frame every N timesteps when --stream-frames (default: 1 = every step).
  --skip-animations
      Do not build or save MP4 animations; frames are still saved. Use build_animations_from_streamed_frames.py later.
  --slice-timestep-images
      When using --stream-frames: generate per-(slice, timestep) PNGs for the dashboard (skipped by default).

  Standard run (pulse types: gaussian, sinusoid, sinusoid_no_ramp, modulated_gaussian, cw):
  --time-steps N
      FDTD time steps for standard run (default: 500). Ignored when --optimize-antenna.
  --max-dim N
      Maximum grid dimension; segmentation downsampled if larger (default: 120).
  --pulse-type TYPE
      Source waveform: gaussian, cw, modulated_gaussian, sinusoid, sinusoid_no_ramp (default: gaussian).
  --prop-direction DIR
      Plane-wave direction for gaussian/modulated_gaussian: +x, -x, +y, -y, +z, -z (default: +y).
  --source-x, --source-y, --source-z
      Grid indices for point source 1 (default: grid center). For cw/sinusoid/sinusoid_no_ramp.
  --pulse-amplitude A
      Amplitude of the source pulse (default: 100). SAR and temperature scale with A².
  --pulse-freq FREQ
      Frequency in Hz for modulated_gaussian, sinusoid, sinusoid_no_ramp, cw (default: 100e6).
  --cw-periods N
      For cw/sinusoid_no_ramp: set time_steps to N periods, SAR from period 10. Unset: cw/sinusoid use min 15 periods.
  --pulse-ramp-width N
      Gaussian ramp-up width (time steps) for CW and sinusoid soft start (default: 30). Ignored for sinusoid_no_ramp.
  --use-source-2, --use-source-3
      Enable 2nd/3rd point source at antenna-like positions (cw/sinusoid/sinusoid_no_ramp).
  --source-x-2, --source-y-2, --source-z-2
      Grid indices for second point source (when --use-source-2). Default: antenna-like position.
  --source-x-3, --source-y-3, --source-z-3
      Grid indices for third point source (when --use-source-3). Default: antenna-like position.
  --source-ring-offset N
      Cells from boundary for default antenna-like positions of source 2/3 (default: 10).

  Grid resolution:
  --dx-mm MM
      Grid resolution (voxel size) in mm (default: 10). Stored in meters for FDTD, SAR, thermal solver, and NIfTI affine.
      Paper recommends 1–5 mm for anatomical detail; default 10 mm preserves backward compatibility.
  --courant-factor F
      Safety factor for time step: dt = F * dt_courant, where dt_courant follows the Courant stability condition (paper Sec. 3.2). Default: 0.99.

Input: BraTS-style segmentation (0=background, 1=necrotic, 2=edema, 3=enhancing; optional 4=normal brain).
Output: results/{timestamp}/{data|images|animations}/{base}_*.png, *.mp4, *.npy, *.json.
  Performance and scalability JSONs include a ``backend`` field (e.g. ``numpy_numba``).

When using --modalities, segmentation is performed by the 3D U-Net from
BrainTumorSegmentation-3DUNet-StreamlitApp (Apache License 2.0). See
brain_tumor_segmentation_model.py for attribution.

Benchmark mode (Objective 5 scalability):
  --benchmark-grid-sizes N [N ...]
      Run minimal FDTD-only for each grid size N³, collect timing and memory, write scalability JSON.
  --benchmark-grid-sizes-range A B S
      Grid sizes as range: min A, max B, step S (e.g. 50 200 50 → 50,100,150,200). Alternative to --benchmark-grid-sizes.
  --benchmark-time-steps N
      FDTD time steps per benchmark run (default: 500).
"""

# Benchmark config (Objective 5: performance evaluation)
BENCHMARK_GRID_SIZES_DEFAULT = [100, 200, 300]
BENCHMARK_TIME_STEPS_DEFAULT = 500

from math import exp, sqrt, cos, sin
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
import numba

try:
    import resource
except ImportError:
    resource = None  # Windows: no resource module; peak memory will be omitted
import numpy as np
import nibabel as nib
from scipy import ndimage
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _get_peak_memory_mb():
    """Return peak resident set size in MB, or None if unavailable (e.g. on Windows)."""
    if resource is None:
        return None
    try:
        import platform

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss: Linux = KB, macOS = bytes (see getrusage(2))
        if platform.system() == "Darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0  # Linux: KB -> MB
    except Exception:
        return None


def _run_minimal_fdtd_benchmark(
    nx, ny, nz, time_steps, dx_mm=10.0, courant_factor=0.99
):
    """
    Run a minimal FDTD-only loop (air, no source) for scalability benchmarking.
    Returns dict with grid_shape, number_of_voxels, time_steps, total_wall_time_s,
    time_per_step_ms, peak_memory_MB for inclusion in scalability JSON.
    """
    dx = dx_mm * 1e-3
    c_light = 2.99792458e8
    dy = dz = dx
    dt_courant = 1.0 / (
        c_light * sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    )
    dt = courant_factor * dt_courant
    npml = max(4, min(16, min(nx, ny, nz) // 10))
    (
        gi1,
        gi2,
        gi3,
        fi1,
        fi2,
        fi3,
        gj1,
        gj2,
        gj3,
        fj1,
        fj2,
        fj3,
        gk1,
        gk2,
        gk3,
        fk1,
        fk2,
        fk3,
    ) = calculate_pml_parameters(npml, nx, ny, nz)
    # Air: eps_r=1, sigma=0 -> eps = 1, conductivity = 0
    eps_x = np.ones((nx, ny, nz))
    eps_y = np.ones((nx, ny, nz))
    eps_z = np.ones((nx, ny, nz))
    conductivity_x = np.zeros((nx, ny, nz))
    conductivity_y = np.zeros((nx, ny, nz))
    conductivity_z = np.zeros((nx, ny, nz))
    Dx = np.zeros((nx, ny, nz))
    Dy = np.zeros((nx, ny, nz))
    Dz = np.zeros((nx, ny, nz))
    iDx = np.zeros((nx, ny, nz))
    iDy = np.zeros((nx, ny, nz))
    iDz = np.zeros((nx, ny, nz))
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Ix = np.zeros((nx, ny, nz))
    Iy = np.zeros((nx, ny, nz))
    Iz = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    iHx = np.zeros((nx, ny, nz))
    iHy = np.zeros((nx, ny, nz))
    iHz = np.zeros((nx, ny, nz))
    t0 = time.perf_counter()
    for _ in range(1, time_steps + 1):
        Dx, iDx = calculate_dx_field(
            nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
        )
        Dy, iDy = calculate_dy_field(
            nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
        )
        Dz, iDz = calculate_dz_field(
            nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
        )
        Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
            nx,
            ny,
            nz,
            Dx,
            Dy,
            Dz,
            eps_x,
            eps_y,
            eps_z,
            conductivity_x,
            conductivity_y,
            conductivity_z,
            Ex,
            Ey,
            Ez,
            Ix,
            Iy,
            Iz,
        )
        Hx, iHx = calculate_hx_field(
            nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
        )
        Hy, iHy = calculate_hy_field(
            nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
        )
        Hz, iHz = calculate_hz_field(
            nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3
        )
    t1 = time.perf_counter()
    total_wall_time_s = t1 - t0
    time_per_step_ms = 1000.0 * total_wall_time_s / time_steps if time_steps else None
    number_of_voxels = nx * ny * nz
    peak_memory_MB = _get_peak_memory_mb()
    return {
        "grid_shape": [nx, ny, nz],
        "number_of_voxels": number_of_voxels,
        "time_steps": time_steps,
        "total_wall_time_s": round(total_wall_time_s, 6),
        "time_per_step_ms": (
            round(time_per_step_ms, 6) if time_per_step_ms is not None else None
        ),
        "peak_memory_MB": (
            round(peak_memory_MB, 2) if peak_memory_MB is not None else None
        ),
    }


# functions for main FDTD loop
def calculate_pml_parameters(
    npml, simulation_size_x, simulation_size_y, simulation_size_z
):
    """Calculate and return the PML parameters"""
    gi1 = np.zeros(simulation_size_x)
    gi2 = np.ones(simulation_size_x)
    gi3 = np.ones(simulation_size_x)
    fi1 = np.zeros(simulation_size_x)
    fi2 = np.ones(simulation_size_x)
    fi3 = np.ones(simulation_size_x)
    gj1 = np.zeros(simulation_size_y)
    gj2 = np.ones(simulation_size_y)
    gj3 = np.ones(simulation_size_y)
    fj1 = np.zeros(simulation_size_y)
    fj2 = np.ones(simulation_size_y)
    fj3 = np.ones(simulation_size_y)
    gk1 = np.zeros(simulation_size_z)
    gk2 = np.ones(simulation_size_z)
    gk3 = np.ones(simulation_size_z)
    fk1 = np.zeros(simulation_size_z)
    fk2 = np.ones(simulation_size_z)
    fk3 = np.ones(simulation_size_z)

    for n in range(npml):
        xxn = (npml - n) / npml
        xn = 0.33 * (xxn**3)
        fi1[n] = xn
        fi1[simulation_size_x - n - 1] = xn
        gi2[n] = 1 / (1 + xn)
        gi2[simulation_size_x - 1 - n] = 1 / (1 + xn)

        gi3[n] = (1 - xn) / (1 + xn)
        gi3[simulation_size_x - 1 - n] = (1 - xn) / (1 + xn)
        fj1[n] = xn
        fj1[simulation_size_y - n - 1] = xn
        gj2[n] = 1 / (1 + xn)
        gj2[simulation_size_y - 1 - n] = 1 / (1 + xn)
        gj3[n] = (1 - xn) / (1 + xn)
        gj3[simulation_size_y - 1 - n] = (1 - xn) / (1 + xn)
        fk1[n] = xn
        fk1[simulation_size_z - n - 1] = xn
        gk2[n] = 1 / (1 + xn)
        gk2[simulation_size_z - 1 - n] = 1 / (1 + xn)
        gk3[n] = (1 - xn) / (1 + xn)
        gk3[simulation_size_z - 1 - n] = (1 - xn) / (1 + xn)
        xxn = (npml - n - 0.5) / npml
        xn = 0.33 * (xxn**3)
        gi1[n] = xn
        gi1[simulation_size_x - 1 - n] = xn
        fi2[n] = 1 / (1 + xn)
        fi2[simulation_size_x - 1 - n] = 1 / (1 + xn)
        fi3[n] = (1 - xn) / (1 + xn)
        fi3[simulation_size_x - 1 - n] = (1 - xn) / (1 + xn)
        gj1[n] = xn
        gj1[simulation_size_y - 1 - n] = xn
        fj2[n] = 1 / (1 + xn)
        fj2[simulation_size_y - 1 - n] = 1 / (1 + xn)
        fj3[n] = (1 - xn) / (1 + xn)
        fj3[simulation_size_y - 1 - n] = (1 - xn) / (1 + xn)
        gk1[n] = xn
        gk1[simulation_size_z - 1 - n] = xn
        fk2[n] = 1 / (1 + xn)
        fk2[simulation_size_z - 1 - n] = 1 / (1 + xn)
        fk3[n] = (1 - xn) / (1 + xn)
        fk3[simulation_size_z - 1 - n] = (1 - xn) / (1 + xn)

    return (
        gi1,
        gi2,
        gi3,
        fi1,
        fi2,
        fi3,
        gj1,
        gj2,
        gj3,
        fj1,
        fj2,
        fj3,
        gk1,
        gk2,
        gk3,
        fk1,
        fk2,
        fk3,
    )


@numba.jit(nopython=True)
def calculate_dx_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dx,
    iDx,
    Hy,
    Hz,
    gj3,
    gk3,
    gj2,
    gk2,
    gi1,
):
    """Calculate the Dx Field"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hz[i, j, k] - Hz[i, j - 1, k] - Hy[i, j, k] + Hy[i, j, k - 1]
                iDx[i, j, k] = iDx[i, j, k] + curl_h
                Dx[i, j, k] = gj3[j] * gk3[k] * Dx[i, j, k] + gj2[j] * gk2[k] * (
                    0.5 * curl_h + gi1[i] * iDx[i, j, k]
                )
    return Dx, iDx


@numba.jit(nopython=True)
def calculate_dy_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dy,
    iDy,
    Hx,
    Hz,
    gi3,
    gk3,
    gi2,
    gk2,
    gj1,
):
    """Calculate the Dy Field"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hx[i, j, k] - Hx[i, j, k - 1] - Hz[i, j, k] + Hz[i - 1, j, k]
                iDy[i, j, k] = iDy[i, j, k] + curl_h
                Dy[i, j, k] = gi3[i] * gk3[k] * Dy[i, j, k] + gi2[i] * gk2[k] * (
                    0.5 * curl_h + gj1[j] * iDy[i, j, k]
                )
    return Dy, iDy


@numba.jit(nopython=True)
def calculate_dz_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dz,
    iDz,
    Hx,
    Hy,
    gi3,
    gj3,
    gi2,
    gj2,
    gk1,
):
    """Calculate the Dz Field"""
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hy[i, j, k] - Hy[i - 1, j, k] - Hx[i, j, k] + Hx[i, j - 1, k]
                iDz[i, j, k] = iDz[i, j, k] + curl_h
                Dz[i, j, k] = gi3[i] * gj3[j] * Dz[i, j, k] + gi2[i] * gj2[j] * (
                    0.5 * curl_h + gk1[k] * iDz[i, j, k]
                )
    return Dz, iDz


@numba.jit(nopython=True)
def calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy, hx_inc):
    """Calculate the incident Dy Field"""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Dy[i, j, ka] = Dy[i, j, ka] - 0.5 * hx_inc[j]
            Dy[i, j, kb + 1] = Dy[i, j, kb + 1] + 0.5 * hx_inc[j]
    return Dy


@numba.jit(nopython=True)
def calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz, hx_inc):
    """Calculate the incident Dz Field"""
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Dz[i, ja, k] = Dz[i, ja, k] + 0.5 * hx_inc[ja - 1]
            Dz[i, jb, k] = Dz[i, jb, k] - 0.5 * hx_inc[jb]
    return Dz


@numba.jit(nopython=True)
def calculate_e_fields(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dx,
    Dy,
    Dz,
    eps_x,
    eps_y,
    eps_z,
    conductivity_x,
    conductivity_y,
    conductivity_z,
    Ex,
    Ey,
    Ez,
    Ix,
    Iy,
    Iz,
):
    """Calculate the E field from the D field"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex[i, j, k] = eps_x[i, j, k] * (Dx[i, j, k] - Ix[i, j, k])
                Ix[i, j, k] = Ix[i, j, k] + conductivity_x[i, j, k] * Ex[i, j, k]
                Ey[i, j, k] = eps_y[i, j, k] * (Dy[i, j, k] - Iy[i, j, k])
                Iy[i, j, k] = Iy[i, j, k] + conductivity_y[i, j, k] * Ey[i, j, k]
                Ez[i, j, k] = eps_z[i, j, k] * (Dz[i, j, k] - Iz[i, j, k])
                Iz[i, j, k] = Iz[i, j, k] + conductivity_z[i, j, k] * Ez[i, j, k]
    return Ex, Ey, Ez, Ix, Iy, Iz


@numba.jit(nopython=True)
def calculate_fourier_transform_ex(
    simulation_size_x,
    simulation_size_y,
    number_of_frequencies,
    real_pt,
    imag_pt,
    Ez,
    arg,
    time_step,
    source_z,
):
    """Calculate the Fourier transform of Ex"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for m in range(0, number_of_frequencies):
                real_pt[m, i, j] = (
                    real_pt[m, i, j] + cos(arg[m] * time_step) * Ez[i, j, source_z]
                )
                imag_pt[m, i, j] = (
                    imag_pt[m, i, j] - sin(arg[m] * time_step) * Ez[i, j, source_z]
                )
    return real_pt, imag_pt


@numba.jit(nopython=True)
def calculate_hx_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hx,
    iHx,
    Ey,
    Ez,
    fi1,
    fj2,
    fk2,
    fj3,
    fk3,
):
    """Calculate the Hx field"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y - 1):
            for k in range(0, simulation_size_z - 1):
                curl_e = Ey[i, j, k + 1] - Ey[i, j, k] - Ez[i, j + 1, k] + Ez[i, j, k]
                iHx[i, j, k] = iHx[i, j, k] + curl_e
                Hx[i, j, k] = fj3[j] * fk3[k] * Hx[i, j, k] + fj2[j] * fk2[k] * 0.5 * (
                    curl_e + fi1[i] * iHx[i, j, k]
                )
    return Hx, iHx


@numba.jit(nopython=True)
def calculate_hy_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hy,
    iHy,
    Ex,
    Ez,
    fj1,
    fi2,
    fk2,
    fi3,
    fk3,
):
    """Calculate the Hy field"""
    for i in range(0, simulation_size_x - 1):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z - 1):
                curl_e = Ez[i + 1, j, k] - Ez[i, j, k] - Ex[i, j, k + 1] + Ex[i, j, k]
                iHy[i, j, k] = iHy[i, j, k] + curl_e
                Hy[i, j, k] = fi3[i] * fk3[k] * Hy[i, j, k] + fi2[i] * fk2[k] * 0.5 * (
                    curl_e + fj1[j] * iHy[i, j, k]
                )
    return Hy, iHy


@numba.jit(nopython=True)
def calculate_hz_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hz,
    iHz,
    Ex,
    Ey,
    fk1,
    fi2,
    fj2,
    fi3,
    fj3,
):
    """Calculate the Hz field"""
    for i in range(0, simulation_size_x - 1):
        for j in range(0, simulation_size_y - 1):
            for k in range(0, simulation_size_z):
                curl_e = Ex[i, j + 1, k] - Ex[i, j, k] - Ey[i + 1, j, k] + Ey[i, j, k]
                iHz[i, j, k] = iHz[i, j, k] + curl_e
                Hz[i, j, k] = fi3[i] * fj3[j] * Hz[i, j, k] + fi2[i] * fj2[j] * 0.5 * (
                    curl_e + fk1[k] * iHz[i, j, k]
                )
    return Hz, iHz


@numba.jit(nopython=True)
def calculate_hx_inc(simulation_size_y, hx_inc, ez_inc):
    """Calculate incident Hx field"""
    for j in range(0, simulation_size_y - 1):
        hx_inc[j] = hx_inc[j] + 0.5 * (ez_inc[j] - ez_inc[j + 1])
    return hx_inc


@numba.jit(nopython=True)
def calculate_hx_with_incident_field(ia, ib, ja, jb, ka, kb, Hx, ez_inc):
    """Calculate Hx with incident Ez"""
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Hx[i, ja - 1, k] = Hx[i, ja - 1, k] + 0.5 * ez_inc[ja]
            Hx[i, jb, k] = Hx[i, jb, k] - 0.5 * ez_inc[jb]
    return Hx


@numba.jit(nopython=True)
def calculate_hy_with_incident_field(ia, ib, ja, jb, ka, kb, Hy, ez_inc):
    """Calculate Hy with incident Ez"""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Hy[ia - 1, j, k] = Hy[ia - 1, j, k] - 0.5 * ez_inc[j]
            Hy[ib, j, k] = Hy[ib, j, k] + 0.5 * ez_inc[j]
    return Hy


# ----- Incident field helpers for plane-wave propagation along x (E_z, H_y) -----
@numba.jit(nopython=True)
def update_ez_inc_x(simulation_size_x, ez_inc_x, hy_inc_x):
    """1D incident Ez update for propagation along x (dEz/dt from dHy/dx)."""
    for i in range(1, simulation_size_x - 1):
        ez_inc_x[i] = ez_inc_x[i] + 0.5 * (hy_inc_x[i - 1] - hy_inc_x[i])
    return ez_inc_x


@numba.jit(nopython=True)
def calculate_hy_inc_x(simulation_size_x, hy_inc_x, ez_inc_x):
    """1D incident Hy update for propagation along x (dHy/dt from dEz/dx)."""
    for i in range(0, simulation_size_x - 1):
        hy_inc_x[i] = hy_inc_x[i] + 0.5 * (ez_inc_x[i] - ez_inc_x[i + 1])
    return hy_inc_x


@numba.jit(nopython=True)
def calculate_inc_dz_field_x(ia, ib, ja, jb, ka, kb, Dz, hy_inc_x):
    """Add incident field (x propagation) to Dz at planes i=ia and i=ib+1."""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Dz[ia, j, k] = Dz[ia, j, k] + 0.5 * hy_inc_x[ia - 1]
            Dz[ib + 1, j, k] = Dz[ib + 1, j, k] - 0.5 * hy_inc_x[ib]
    return Dz


@numba.jit(nopython=True)
def calculate_hy_with_incident_field_x(ia, ib, ja, jb, ka, kb, Hy, ez_inc_x):
    """Add incident Ez (x propagation) to Hy at planes i=ia-1 and i=ib."""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Hy[ia - 1, j, k] = Hy[ia - 1, j, k] - 0.5 * ez_inc_x[ia]
            Hy[ib, j, k] = Hy[ib, j, k] + 0.5 * ez_inc_x[ib]
    return Hy


# ----- Incident field helpers for plane-wave propagation along z (E_z, H_x) -----
@numba.jit(nopython=True)
def update_ez_inc_z(simulation_size_z, ez_inc_z, hx_inc_z):
    """1D incident Ez update for propagation along z (dEz/dt from dHx/dz)."""
    for k in range(1, simulation_size_z - 1):
        ez_inc_z[k] = ez_inc_z[k] + 0.5 * (hx_inc_z[k - 1] - hx_inc_z[k])
    return ez_inc_z


@numba.jit(nopython=True)
def calculate_hx_inc_z(simulation_size_z, hx_inc_z, ez_inc_z):
    """1D incident Hx update for propagation along z (dHx/dt from dEz/dz)."""
    for k in range(0, simulation_size_z - 1):
        hx_inc_z[k] = hx_inc_z[k] + 0.5 * (ez_inc_z[k] - ez_inc_z[k + 1])
    return hx_inc_z


@numba.jit(nopython=True)
def calculate_inc_dz_field_z(ia, ib, ja, jb, ka, kb, Dz, hx_inc_z):
    """Add incident field (z propagation) to Dz at planes k=ka and k=kb+1."""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Dz[i, j, ka] = Dz[i, j, ka] + 0.5 * hx_inc_z[ka - 1]
            Dz[i, j, kb + 1] = Dz[i, j, kb + 1] - 0.5 * hx_inc_z[kb]
    return Dz


@numba.jit(nopython=True)
def calculate_hx_with_incident_field_z(ia, ib, ja, jb, ka, kb, Hx, ez_inc_z):
    """Add incident Ez (z propagation) to Hx at planes k=ka-1 and k=kb."""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Hx[i, j, ka - 1] = Hx[i, j, ka - 1] + 0.5 * ez_inc_z[ka]
            Hx[i, j, kb] = Hx[i, j, kb] - 0.5 * ez_inc_z[kb]
    return Hx


@numba.jit(nopython=True)
def accumulate_e_field_squared(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex,
    Ey,
    Ez,
    Ex_sq_sum,
    Ey_sq_sum,
    Ez_sq_sum,
):
    """Accumulate squared electric field components for SAR calculation"""
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex_sq_sum[i, j, k] = Ex_sq_sum[i, j, k] + Ex[i, j, k] ** 2
                Ey_sq_sum[i, j, k] = Ey_sq_sum[i, j, k] + Ey[i, j, k] ** 2
                Ez_sq_sum[i, j, k] = Ez_sq_sum[i, j, k] + Ez[i, j, k] ** 2
    return Ex_sq_sum, Ey_sq_sum, Ez_sq_sum


@numba.jit(nopython=True)
def compute_instantaneous_sar(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex,
    Ey,
    Ez,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
):
    """
    Compute instantaneous SAR from current E-field values
    SAR = σ|E|² / (2ρ)
    Units: W/kg
    """
    sar = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                # Instantaneous magnitude squared of electric field
                e_mag_sq = Ex[i, j, k] ** 2 + Ey[i, j, k] ** 2 + Ez[i, j, k] ** 2

                # Average conductivity (isotropic assumption)
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0

                # SAR computation
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_mag_sq) / (2.0 * rho[i, j, k])
                else:
                    sar[i, j, k] = 0.0

    return sar


@numba.jit(nopython=True)
def compute_sar(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex_sq_sum,
    Ey_sq_sum,
    Ez_sq_sum,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
    n_samples,
):
    """
    Compute Specific Absorption Rate (SAR) using RMS values
    SAR = σ|E_rms|² / (2ρ)
    Units: W/kg
    """
    sar = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                # RMS magnitude squared of electric field
                e_rms_sq = (
                    Ex_sq_sum[i, j, k] + Ey_sq_sum[i, j, k] + Ez_sq_sum[i, j, k]
                ) / n_samples

                # Average conductivity (isotropic assumption)
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0

                # SAR computation
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_rms_sq) / (2.0 * rho[i, j, k])
                else:
                    sar[i, j, k] = 0.0

    return sar


def solve_steady_bioheat_3d(
    nx, ny, nz, k_3d, Q_3d, dx, T_boundary=37.0, max_iter=50000, tol=1e-6
):
    """
    Solve simplified Pennes steady-state (no perfusion): ∇·(k∇T) + Q = 0.
    Q = SAR·ρ (W/m³). Dirichlet T = T_boundary at domain boundaries.
    Central-difference Laplacian; Gauss-Seidel iteration. Only updates voxels with k > 0.
    tol=1e-6 so that small Q still converges to the correct temperature rise (not stop at first iteration).
    """
    T = np.full((nx, ny, nz), T_boundary, dtype=np.float64)
    dx2 = dx * dx
    for _ in range(max_iter):
        T_old = T.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    if k_3d[i, j, k] <= 0:
                        T[i, j, k] = T_boundary
                        continue
                    kc = k_3d[i, j, k]
                    q = Q_3d[i, j, k]
                    # 6-neighbor stencil; boundary nodes stay T_boundary
                    s = (
                        T[i + 1, j, k]
                        + T[i - 1, j, k]
                        + T[i, j + 1, k]
                        + T[i, j - 1, k]
                        + T[i, j, k + 1]
                        + T[i, j, k - 1]
                    )
                    T[i, j, k] = (s + q * dx2 / kc) / 6.0
        err = np.max(np.abs(T - T_old))
        if err < tol:
            break
    return T


# ===========================================================================
# Antenna optimization framework (Objective 4)
# Houle Ch.6 style: 4-quadrant APA with superposition + J-ratio objective.
# ===========================================================================


def build_quadrant_sources(
    nx, ny, nz, npml, dipole_half_len=9, ring_offset=None, z_plane=None
):
    """
    Build 4-quadrant dipole source positions for an annular phased array (APA).
    Quadrants are placed on +x, +y, -x, -y faces of the domain (inside PML).
    Each dipole is oriented along z and has a gap at center.

    Parameters
    ----------
    ring_offset : int or None
        Distance (cells) from PML boundary to dipole gap. Default: npml + 2.
    z_plane : int or None
        Z-index for dipole gap center. Default: nz // 2.

    Returns list of 4 dicts, each with 'gap' (i,j,k) and 'arm' voxels.
    """
    cx, cy = nx // 2, ny // 2
    cz = z_plane if z_plane is not None else nz // 2
    # Place dipoles a few cells inside PML on each face
    offset = ring_offset if ring_offset is not None else (npml + 2)
    quadrant_positions = [
        (cx, offset, cz),  # Q1: near y-low face
        (nx - offset - 1, cy, cz),  # Q2: near x-high face
        (cx, ny - offset - 1, cz),  # Q3: near y-high face
        (offset, cy, cz),  # Q4: near x-low face
    ]
    sources = []
    for qi, (gi, gj, gk) in enumerate(quadrant_positions):
        arm_voxels = []
        for dz in range(-dipole_half_len, dipole_half_len + 1):
            kk = gk + dz
            if 0 <= kk < nz and dz != 0:
                arm_voxels.append((gi, gj, kk))
        sources.append(
            {
                "quadrant": qi + 1,
                "gap": (gi, gj, gk),
                "arms": arm_voxels,
            }
        )
    return sources


def run_unit_quadrant_fdtd(
    quadrant_source,
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    eps_x,
    eps_y,
    eps_z,
    conductivity_x,
    conductivity_y,
    conductivity_z,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
    dx,
    dt,
    epsz,
    npml,
    f0,
    time_steps_opt,
):
    """
    Run a single FDTD simulation with one quadrant's dipole active at frequency f0.
    Uses a soft sinusoidal CW source at the dipole gap, with a Gaussian ramp-up.
    Collects DFT of Ez at f0 over the full grid -> returns complex Ez field (amplitude, phase).
    Returns: complex_Ez array of shape (nx, ny, nz) at frequency f0.
    """
    nx, ny, nz = simulation_size_x, simulation_size_y, simulation_size_z
    # Allocate fields (fresh per quadrant)
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Ix = np.zeros((nx, ny, nz))
    Iy = np.zeros((nx, ny, nz))
    Iz = np.zeros((nx, ny, nz))
    Dx = np.zeros((nx, ny, nz))
    Dy = np.zeros((nx, ny, nz))
    Dz = np.zeros((nx, ny, nz))
    iDx = np.zeros((nx, ny, nz))
    iDy = np.zeros((nx, ny, nz))
    iDz = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    iHx = np.zeros((nx, ny, nz))
    iHy = np.zeros((nx, ny, nz))
    iHz = np.zeros((nx, ny, nz))

    # PML
    (
        gi1,
        gi2,
        gi3,
        fi1,
        fi2,
        fi3,
        gj1,
        gj2,
        gj3,
        fj1,
        fj2,
        fj3,
        gk1,
        gk2,
        gk3,
        fk1,
        fk2,
        fk3,
    ) = calculate_pml_parameters(npml, nx, ny, nz)

    # Set dipole arms to metal (ga=0 -> short circuit in E-update is modeled
    # by zeroing eps for those voxels; for simplicity we just don't excite them,
    # the soft source at the gap is sufficient for this framework).

    # DFT accumulators at f0
    omega_dt = 2.0 * np.pi * f0 * dt
    real_Ez = np.zeros((nx, ny, nz))
    imag_Ez = np.zeros((nx, ny, nz))

    gap_i, gap_j, gap_k = quadrant_source["gap"]
    ramp_width = 30.0  # Gaussian ramp-up width in time steps

    for t in range(1, time_steps_opt + 1):
        # D-field updates
        Dx, iDx = calculate_dx_field(
            nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
        )
        Dy, iDy = calculate_dy_field(
            nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
        )
        Dz, iDz = calculate_dz_field(
            nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
        )

        # Soft source: sinusoidal CW at f0 with Gaussian ramp-up (unit amplitude)
        ramp = 1.0 - exp(-0.5 * (t / ramp_width) ** 2)
        source_val = ramp * sin(2.0 * np.pi * f0 * t * dt)
        # Soft source: ADD to Dz at the gap (not overwrite)
        Dz[gap_i, gap_j, gap_k] += source_val

        # E-field from D-field
        Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
            nx,
            ny,
            nz,
            Dx,
            Dy,
            Dz,
            eps_x,
            eps_y,
            eps_z,
            conductivity_x,
            conductivity_y,
            conductivity_z,
            Ex,
            Ey,
            Ez,
            Ix,
            Iy,
            Iz,
        )

        # H-field updates
        Hx, iHx = calculate_hx_field(
            nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
        )
        Hy, iHy = calculate_hy_field(
            nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
        )
        Hz, iHz = calculate_hz_field(
            nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3
        )

        # DFT accumulation (running Fourier transform at f0)
        cos_val = cos(omega_dt * t)
        sin_val = sin(omega_dt * t)
        real_Ez += cos_val * Ez
        imag_Ez -= sin_val * Ez

    # Complex Ez = real + j*imag (phasor at f0)
    complex_Ez = real_Ez + 1j * imag_Ez
    return complex_Ez


def synthesize_total_field(complex_fields, alphas, thetas):
    """
    Superpose complex Ez fields from N quadrants with amplitude/phase controls.
    complex_fields: list of N complex arrays (nx, ny, nz)
    alphas: array of N amplitudes
    thetas: array of N phase shifts (radians)
    Returns: complex E_total array (nx, ny, nz)
    """
    E_total = np.zeros_like(complex_fields[0])
    for q in range(len(complex_fields)):
        E_total += alphas[q] * complex_fields[q] * np.exp(1j * thetas[q])
    return E_total


def compute_sar_from_complex_field(E_total, sigma_avg, rho):
    """
    SAR = sigma * |E_total|^2 / (2 * rho), voxel-wise.
    sigma_avg, rho: 3D arrays. Returns SAR array.
    """
    E_mag_sq = np.abs(E_total) ** 2
    sar = np.zeros_like(E_mag_sq)
    tissue = rho > 0
    sar[tissue] = sigma_avg[tissue] * E_mag_sq[tissue] / (2.0 * rho[tissue])
    return sar


def compute_j_ratio(sar, tumor_mask, healthy_mask):
    """
    J = mean SAR in tumor / mean SAR in healthy tissue.
    Returns J (float), mean_tumor, mean_healthy.
    """
    mean_tumor = np.mean(sar[tumor_mask]) if np.any(tumor_mask) else 0.0
    mean_healthy = np.mean(sar[healthy_mask]) if np.any(healthy_mask) else 1e-30
    if mean_healthy < 1e-30:
        mean_healthy = 1e-30
    J = mean_tumor / mean_healthy
    return J, mean_tumor, mean_healthy


def compute_robust_objective(sar, tumor_mask, healthy_mask, penalty_weight=0.0):
    """
    Robust objective: J_eff = J - penalty_weight * P95_healthy / mean_tumor.
    When penalty_weight=0, this is identical to plain J.
    The penalty discourages solutions with healthy-tissue SAR hotspots.
    Returns J_eff, J_plain, mean_tumor, mean_healthy, p95_healthy.
    """
    mean_tumor = np.mean(sar[tumor_mask]) if np.any(tumor_mask) else 0.0
    mean_healthy = np.mean(sar[healthy_mask]) if np.any(healthy_mask) else 1e-30
    if mean_healthy < 1e-30:
        mean_healthy = 1e-30
    J_plain = mean_tumor / mean_healthy
    p95_healthy = (
        float(np.percentile(sar[healthy_mask], 95)) if np.any(healthy_mask) else 0.0
    )
    if penalty_weight > 0 and mean_tumor > 0:
        penalty = penalty_weight * p95_healthy / mean_tumor
    else:
        penalty = 0.0
    J_eff = J_plain - penalty
    return J_eff, J_plain, mean_tumor, mean_healthy, p95_healthy


def _evaluate_candidate(
    complex_fields,
    alphas,
    thetas,
    sigma_avg,
    rho,
    tumor_mask,
    healthy_mask,
    penalty_weight=0.0,
):
    """Helper: synthesize field, compute SAR, return (J_eff, J_plain, mean_tumor, mean_healthy, p95)."""
    E_total = synthesize_total_field(complex_fields, alphas, thetas)
    sar = compute_sar_from_complex_field(E_total, sigma_avg, rho)
    return compute_robust_objective(sar, tumor_mask, healthy_mask, penalty_weight)


# ---------------------------------------------------------------------------
# Top-level picklable worker functions for multiprocessing (--opt-parallel)
# These must be module-level so ProcessPoolExecutor can pickle them.
# ---------------------------------------------------------------------------


def _eval_frequency_one(
    f0,
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    time_steps_opt,
    eps_x,
    eps_y,
    eps_z,
    conductivity_x,
    conductivity_y,
    conductivity_z,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
    dx,
    dt,
    epsz,
    tumor_mask,
    healthy_mask,
):
    """
    Worker: run 4 unit FDTD sims at frequency f0 with default geometry and return baseline J.
    Used by frequency sweep parallel branch.
    """
    npml = max(
        4, min(16, min(simulation_size_x, simulation_size_y, simulation_size_z) // 10)
    )
    quad_srcs = build_quadrant_sources(
        simulation_size_x, simulation_size_y, simulation_size_z, npml=npml
    )
    fields = []
    for qs in quad_srcs:
        cplx = run_unit_quadrant_fdtd(
            quadrant_source=qs,
            simulation_size_x=simulation_size_x,
            simulation_size_y=simulation_size_y,
            simulation_size_z=simulation_size_z,
            eps_x=eps_x,
            eps_y=eps_y,
            eps_z=eps_z,
            conductivity_x=conductivity_x,
            conductivity_y=conductivity_y,
            conductivity_z=conductivity_z,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            rho=rho,
            dx=dx,
            dt=dt,
            epsz=epsz,
            npml=npml,
            f0=f0,
            time_steps_opt=time_steps_opt,
        )
        fields.append(cplx)
    sigma_avg = (sigma_x + sigma_y + sigma_z) / 3.0
    E_bl = synthesize_total_field(fields, np.ones(4), np.zeros(4))
    sar_bl = compute_sar_from_complex_field(E_bl, sigma_avg, rho)
    J_bl, _, _ = compute_j_ratio(sar_bl, tumor_mask, healthy_mask)
    return {"f0": f0, "J_baseline": J_bl}


def _eval_geom_one(
    g_off,
    g_z,
    f0,
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    time_steps_opt,
    eps_x,
    eps_y,
    eps_z,
    conductivity_x,
    conductivity_y,
    conductivity_z,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
    dx,
    dt,
    epsz,
    tumor_mask,
    healthy_mask,
):
    """
    Worker: run 4 unit FDTD sims for one (ring_offset, z_plane) geometry and return baseline J.
    Used by geometry sweep parallel branch.
    """
    npml = max(
        4, min(16, min(simulation_size_x, simulation_size_y, simulation_size_z) // 10)
    )
    quad_srcs = build_quadrant_sources(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        npml=npml,
        ring_offset=g_off,
        z_plane=g_z,
    )
    fields = []
    for qs in quad_srcs:
        cplx = run_unit_quadrant_fdtd(
            quadrant_source=qs,
            simulation_size_x=simulation_size_x,
            simulation_size_y=simulation_size_y,
            simulation_size_z=simulation_size_z,
            eps_x=eps_x,
            eps_y=eps_y,
            eps_z=eps_z,
            conductivity_x=conductivity_x,
            conductivity_y=conductivity_y,
            conductivity_z=conductivity_z,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            rho=rho,
            dx=dx,
            dt=dt,
            epsz=epsz,
            npml=npml,
            f0=f0,
            time_steps_opt=time_steps_opt,
        )
        fields.append(cplx)
    sigma_avg = (sigma_x + sigma_y + sigma_z) / 3.0
    E_bl = synthesize_total_field(fields, np.ones(4), np.zeros(4))
    sar_bl = compute_sar_from_complex_field(E_bl, sigma_avg, rho)
    J_bl, _, _ = compute_j_ratio(sar_bl, tumor_mask, healthy_mask)
    gaps = [qs["gap"] for qs in quad_srcs]
    return {"ring_offset": g_off, "z_plane": g_z, "J_baseline": J_bl, "gaps": gaps}


def _run_one_multistart(
    ms_idx,
    start_offset,
    complex_fields,
    sigma_avg,
    rho,
    tumor_mask,
    healthy_mask,
    penalty_weight,
    phase_values,
    amp_values,
    amp_range,
    n_quadrants,
    refine_iterations,
    refine_factor,
    phase_steps,
    amp_steps,
):
    """
    Worker: run one full multi-start iteration (phase sweep + amplitude sweep + refinement).
    Returns dict with best_J, best_alphas, best_thetas, trace, eval_count, ms_idx.
    """
    from itertools import product as iter_product

    trace = []
    eval_count = 0
    best_J = -1.0
    best_thetas = np.zeros(n_quadrants)
    best_alphas = np.ones(n_quadrants)

    # Phase sweep
    for combo in iter_product(phase_values, repeat=n_quadrants - 1):
        thetas = np.array([0.0] + list(combo)) + start_offset
        alphas = np.ones(n_quadrants)
        J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
            complex_fields,
            alphas,
            thetas,
            sigma_avg,
            rho,
            tumor_mask,
            healthy_mask,
            penalty_weight,
        )
        eval_count += 1
        if J_eff > best_J:
            best_J = J_eff
            best_thetas = thetas.copy()
            best_alphas = alphas.copy()
            trace.append(
                {
                    "eval": eval_count,
                    "phase": f"phase_sweep_ms{ms_idx}",
                    "J_eff": float(J_eff),
                    "J": float(J_plain),
                    "alphas": alphas.tolist(),
                    "thetas": thetas.tolist(),
                    "p95_healthy": float(p95),
                }
            )

    # Amplitude sweep
    for combo in iter_product(amp_values, repeat=n_quadrants - 1):
        alphas = np.array([1.0] + list(combo))
        J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
            complex_fields,
            alphas,
            best_thetas,
            sigma_avg,
            rho,
            tumor_mask,
            healthy_mask,
            penalty_weight,
        )
        eval_count += 1
        if J_eff > best_J:
            best_J = J_eff
            best_alphas = alphas.copy()
            trace.append(
                {
                    "eval": eval_count,
                    "phase": f"amp_sweep_ms{ms_idx}",
                    "J_eff": float(J_eff),
                    "J": float(J_plain),
                    "alphas": alphas.tolist(),
                    "thetas": best_thetas.tolist(),
                    "p95_healthy": float(p95),
                }
            )

    # Coordinate-descent refinement
    phase_delta = (2 * np.pi / phase_steps) * refine_factor
    amp_delta = ((amp_range[1] - amp_range[0]) / amp_steps) * refine_factor
    for r_iter in range(refine_iterations):
        improved = False
        for q in range(1, n_quadrants):
            for direction in [-1, 1]:
                trial_thetas = best_thetas.copy()
                trial_thetas[q] += direction * phase_delta
                J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                    complex_fields,
                    best_alphas,
                    trial_thetas,
                    sigma_avg,
                    rho,
                    tumor_mask,
                    healthy_mask,
                    penalty_weight,
                )
                eval_count += 1
                if J_eff > best_J:
                    best_J = J_eff
                    best_thetas = trial_thetas.copy()
                    improved = True
                    trace.append(
                        {
                            "eval": eval_count,
                            "phase": f"refine_{r_iter}_ms{ms_idx}",
                            "J_eff": float(J_eff),
                            "J": float(J_plain),
                            "alphas": best_alphas.tolist(),
                            "thetas": best_thetas.tolist(),
                            "p95_healthy": float(p95),
                        }
                    )
        for q in range(1, n_quadrants):
            for direction in [-1, 1]:
                trial_alphas = best_alphas.copy()
                trial_alphas[q] = np.clip(
                    trial_alphas[q] + direction * amp_delta, amp_range[0], amp_range[1]
                )
                J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                    complex_fields,
                    trial_alphas,
                    best_thetas,
                    sigma_avg,
                    rho,
                    tumor_mask,
                    healthy_mask,
                    penalty_weight,
                )
                eval_count += 1
                if J_eff > best_J:
                    best_J = J_eff
                    best_alphas = trial_alphas.copy()
                    improved = True
                    trace.append(
                        {
                            "eval": eval_count,
                            "phase": f"refine_{r_iter}_ms{ms_idx}",
                            "J_eff": float(J_eff),
                            "J": float(J_plain),
                            "alphas": best_alphas.tolist(),
                            "thetas": best_thetas.tolist(),
                            "p95_healthy": float(p95),
                        }
                    )
        phase_delta *= refine_factor
        amp_delta *= refine_factor
        if not improved:
            break

    _, J_plain_final, _, _, _ = _evaluate_candidate(
        complex_fields,
        best_alphas,
        best_thetas,
        sigma_avg,
        rho,
        tumor_mask,
        healthy_mask,
        0.0,
    )
    return {
        "ms_idx": ms_idx,
        "best_J": best_J,
        "best_J_plain": J_plain_final,
        "best_alphas": best_alphas,
        "best_thetas": best_thetas,
        "trace": trace,
        "eval_count": eval_count,
    }


def optimize_quadrant_controls(
    complex_fields,
    sigma_avg,
    rho,
    tumor_mask,
    healthy_mask,
    n_quadrants=4,
    phase_steps=24,
    amp_steps=9,
    amp_range=(0.2, 2.5),
    refine_iterations=8,
    refine_factor=0.5,
    multi_start=3,
    penalty_weight=0.0,
    parallel_workers=1,
):
    """
    Optimize amplitudes and phases of N quadrants to maximize J = SAR_tumor / SAR_healthy.
    Strategy: multi-start coarse grid search on phases (with Q1 phase fixed at 0),
    then amplitude sweep, then local coordinate-descent refinement.
    When penalty_weight > 0, uses robust objective (J minus healthy P95 penalty).
    When parallel_workers > 1, multi-start iterations run in parallel processes.
    Returns: best_alphas, best_thetas, best_J, trace (list of dicts), eval_count.
    """
    from itertools import product as iter_product

    trace = []
    eval_count = 0
    global_best_J = -1.0
    global_best_Jplain = -1.0
    global_best_thetas = np.zeros(n_quadrants)
    global_best_alphas = np.ones(n_quadrants)

    # Generate multi-start initial phase offsets (first is zero = standard start)
    np.random.seed(42)
    start_offsets = [np.zeros(n_quadrants)]
    for _ in range(multi_start - 1):
        offset = np.zeros(n_quadrants)
        offset[1:] = np.random.uniform(-np.pi, np.pi, n_quadrants - 1)
        start_offsets.append(offset)

    phase_values = np.linspace(-np.pi, np.pi, phase_steps, endpoint=False)
    amp_values = np.linspace(amp_range[0], amp_range[1], amp_steps)

    # -----------------------------------------------------------------------
    # Parallel multi-start: each start runs phase sweep + amp sweep + refinement
    # -----------------------------------------------------------------------
    if parallel_workers > 1 and len(start_offsets) > 1:
        n_ms_workers = min(parallel_workers, len(start_offsets))
        print(
            f"\n  Running {len(start_offsets)} multi-starts in parallel ({n_ms_workers} workers)..."
        )
        with ProcessPoolExecutor(max_workers=n_ms_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_multistart,
                    ms_idx,
                    start_offset,
                    complex_fields,
                    sigma_avg,
                    rho,
                    tumor_mask,
                    healthy_mask,
                    penalty_weight,
                    phase_values,
                    amp_values,
                    amp_range,
                    n_quadrants,
                    refine_iterations,
                    refine_factor,
                    phase_steps,
                    amp_steps,
                ): ms_idx
                for ms_idx, start_offset in enumerate(start_offsets)
            }
            for future in as_completed(futures):
                ms_idx = futures[future]
                try:
                    result = future.result()
                    trace.extend(result["trace"])
                    eval_count += result["eval_count"]
                    print(
                        f"  Multi-start {result['ms_idx'] + 1}/{len(start_offsets)} done: "
                        f"J_eff={result['best_J']:.6f}, J_plain={result['best_J_plain']:.6f}"
                    )
                    if result["best_J"] > global_best_J:
                        global_best_J = result["best_J"]
                        global_best_Jplain = result["best_J_plain"]
                        global_best_alphas = result["best_alphas"].copy()
                        global_best_thetas = result["best_thetas"].copy()
                except Exception as exc:
                    print(f"  Multi-start {ms_idx + 1} error: {exc}")
        print(
            f"\n  Optimization complete: best J_eff = {global_best_J:.6f}, "
            f"best J_plain = {global_best_Jplain:.6f} ({eval_count} total evals)"
        )
        return (
            global_best_alphas,
            global_best_thetas,
            global_best_Jplain,
            trace,
            eval_count,
        )

    # -----------------------------------------------------------------------
    # Serial multi-start (parallel_workers == 1 or single start)
    # -----------------------------------------------------------------------
    for ms_idx, start_offset in enumerate(start_offsets):
        print(
            f"\n  --- Multi-start {ms_idx + 1}/{len(start_offsets)} "
            f"(offset: [{', '.join(f'{np.degrees(o):.0f}°' for o in start_offset)}]) ---"
        )

        # --- Phase 1: coarse phase sweep (all amplitudes = 1.0) ---
        n_phase_combos = phase_steps ** (n_quadrants - 1)
        print(
            f"  Phase sweep: {phase_steps} steps/quadrant ({n_phase_combos} combos)..."
        )
        best_J = -1.0
        best_thetas = np.zeros(n_quadrants)
        best_alphas = np.ones(n_quadrants)

        for combo in iter_product(phase_values, repeat=n_quadrants - 1):
            thetas = np.array([0.0] + list(combo)) + start_offset
            alphas = np.ones(n_quadrants)
            J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                complex_fields,
                alphas,
                thetas,
                sigma_avg,
                rho,
                tumor_mask,
                healthy_mask,
                penalty_weight,
            )
            eval_count += 1
            if J_eff > best_J:
                best_J = J_eff
                best_thetas = thetas.copy()
                best_alphas = alphas.copy()
                trace.append(
                    {
                        "eval": eval_count,
                        "phase": f"phase_sweep_ms{ms_idx}",
                        "J_eff": float(J_eff),
                        "J": float(J_plain),
                        "alphas": alphas.tolist(),
                        "thetas": thetas.tolist(),
                        "p95_healthy": float(p95),
                    }
                )

        print(f"    Best J_eff after phase sweep: {best_J:.6f} ({eval_count} evals)")

        # --- Phase 2: amplitude sweep around best phases ---
        print(f"  Amplitude sweep: {amp_steps} steps/quadrant...")
        for combo in iter_product(amp_values, repeat=n_quadrants - 1):
            alphas = np.array([1.0] + list(combo))
            J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                complex_fields,
                alphas,
                best_thetas,
                sigma_avg,
                rho,
                tumor_mask,
                healthy_mask,
                penalty_weight,
            )
            eval_count += 1
            if J_eff > best_J:
                best_J = J_eff
                best_alphas = alphas.copy()
                trace.append(
                    {
                        "eval": eval_count,
                        "phase": f"amp_sweep_ms{ms_idx}",
                        "J_eff": float(J_eff),
                        "J": float(J_plain),
                        "alphas": alphas.tolist(),
                        "thetas": best_thetas.tolist(),
                        "p95_healthy": float(p95),
                    }
                )

        print(
            f"    Best J_eff after amplitude sweep: {best_J:.6f} ({eval_count} evals)"
        )

        # --- Phase 3: local coordinate-descent refinement ---
        print(f"  Coordinate refinement ({refine_iterations} iterations)...")
        phase_delta = (2 * np.pi / phase_steps) * refine_factor
        amp_delta = ((amp_range[1] - amp_range[0]) / amp_steps) * refine_factor

        for r_iter in range(refine_iterations):
            improved = False
            # Refine phases (Q2..Q4)
            for q in range(1, n_quadrants):
                for direction in [-1, 1]:
                    trial_thetas = best_thetas.copy()
                    trial_thetas[q] += direction * phase_delta
                    J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                        complex_fields,
                        best_alphas,
                        trial_thetas,
                        sigma_avg,
                        rho,
                        tumor_mask,
                        healthy_mask,
                        penalty_weight,
                    )
                    eval_count += 1
                    if J_eff > best_J:
                        best_J = J_eff
                        best_thetas = trial_thetas.copy()
                        improved = True
                        trace.append(
                            {
                                "eval": eval_count,
                                "phase": f"refine_{r_iter}_ms{ms_idx}",
                                "J_eff": float(J_eff),
                                "J": float(J_plain),
                                "alphas": best_alphas.tolist(),
                                "thetas": best_thetas.tolist(),
                                "p95_healthy": float(p95),
                            }
                        )
            # Refine amplitudes (Q2..Q4)
            for q in range(1, n_quadrants):
                for direction in [-1, 1]:
                    trial_alphas = best_alphas.copy()
                    trial_alphas[q] = np.clip(
                        trial_alphas[q] + direction * amp_delta,
                        amp_range[0],
                        amp_range[1],
                    )
                    J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                        complex_fields,
                        trial_alphas,
                        best_thetas,
                        sigma_avg,
                        rho,
                        tumor_mask,
                        healthy_mask,
                        penalty_weight,
                    )
                    eval_count += 1
                    if J_eff > best_J:
                        best_J = J_eff
                        best_alphas = trial_alphas.copy()
                        improved = True
                        trace.append(
                            {
                                "eval": eval_count,
                                "phase": f"refine_{r_iter}_ms{ms_idx}",
                                "J_eff": float(J_eff),
                                "J": float(J_plain),
                                "alphas": best_alphas.tolist(),
                                "thetas": best_thetas.tolist(),
                                "p95_healthy": float(p95),
                            }
                        )
            phase_delta *= refine_factor
            amp_delta *= refine_factor
            if not improved:
                print(
                    f"    Refinement iteration {r_iter}: no improvement, stopping early."
                )
                break

        # Check if this multi-start beat the global best
        # Compute actual plain J for comparison
        _, J_plain_final, _, _, _ = _evaluate_candidate(
            complex_fields,
            best_alphas,
            best_thetas,
            sigma_avg,
            rho,
            tumor_mask,
            healthy_mask,
            0.0,
        )
        print(
            f"  Multi-start {ms_idx + 1} result: J_eff={best_J:.6f}, J_plain={J_plain_final:.6f}"
        )
        if best_J > global_best_J:
            global_best_J = best_J
            global_best_Jplain = J_plain_final
            global_best_thetas = best_thetas.copy()
            global_best_alphas = best_alphas.copy()

    print(
        f"\n  Optimization complete: best J_eff = {global_best_J:.6f}, "
        f"best J_plain = {global_best_Jplain:.6f} ({eval_count} total evals)"
    )
    return global_best_alphas, global_best_thetas, global_best_Jplain, trace, eval_count


# ----- Load brain segmentation: from file OR from 4 modalities (run U-Net) -----
# (RESULTS_DIR and subdirs are created inside __main__ so worker processes do not create extra dirs.)


def _write_progress(phase, message, percent, phases_done=None, extra=None):
    """Write progress JSON for dashboard. phases_done = list of completed phase names."""
    try:
        os.makedirs(PROGRESS_DIR, exist_ok=True)
        payload = {
            "phase": phase,
            "message": message,
            "percent": min(100, max(0, percent)),
            "phases_done": list(phases_done) if phases_done else [],
            "updated_at": datetime.now().isoformat(),
        }
        if extra:
            payload.update(extra)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
    except Exception:
        pass


_DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_model.pth",
)


def _find_modalities_in_dir(dir_path):
    """
    Find FLAIR, T1, T1CE, T2 NIfTI files in a directory.
    Supports: exact names (flair.nii, t1.nii, t1ce.nii, t2.nii) or
    BraTS-style (*_flair.nii, *_t1.nii, *_t1ce.nii, *_t2.nii).
    Returns (flair_path, t1_path, t1ce_path, t2_path).
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Modalities directory not found: {dir_path}")
    names = {}
    for f in os.listdir(dir_path):
        if not f.endswith(".nii") and not f.endswith(".nii.gz"):
            continue
        fp = os.path.join(dir_path, f)
        if not os.path.isfile(fp):
            continue
        low = f.lower()
        if low == "flair.nii" or low.endswith("_flair.nii"):
            names["flair"] = fp
        elif low == "t1ce.nii" or low.endswith("_t1ce.nii"):
            names["t1ce"] = fp
        elif (low == "t1.nii" or low.endswith("_t1.nii")) and "t1ce" not in low:
            names["t1"] = fp
        elif low == "t2.nii" or low.endswith("_t2.nii"):
            names["t2"] = fp
    for key in ("flair", "t1", "t1ce", "t2"):
        if key not in names:
            raise FileNotFoundError(
                f"Missing modality in {dir_path}: no file matching {key} (e.g. {key}.nii or *_t1ce.nii)"
            )
    return names["flair"], names["t1"], names["t1ce"], names["t2"]


def _load_segmentation_for_benchmark(args):
    """Load segmentation (from --seg, --modalities, or --modalities-dir) for full-pipeline benchmark. Returns labels_3d (int32, 3D)."""
    use_modalities = (args.modalities is not None) or (args.modalities_dir is not None)
    if use_modalities:
        if args.modalities_dir is not None:
            flair_path, t1_path, t1ce_path, t2_path = _find_modalities_in_dir(
                args.modalities_dir
            )
        else:
            flair_path, t1_path, t1ce_path, t2_path = tuple(args.modalities)
        from brain_tumor_segmentation_model import run_segmentation_from_modalities

        labels_3d = run_segmentation_from_modalities(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            args.checkpoint,
            extend_with_normal_brain=not args.no_normal_brain,
        )
    else:
        seg_path = args.seg or os.environ.get(
            "BRAIN_SEGMENTATION_NII", "brain_segmentation.nii"
        )
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(
                f"Segmentation file not found: {seg_path}. "
                "Use --seg path.nii, --modalities F T1 T1CE T2, or --modalities-dir DIR"
            )
        img = nib.load(seg_path)
        labels_3d = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
        if labels_3d.ndim != 3:
            raise ValueError(f"Expected 3D segmentation, got shape {labels_3d.shape}")
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(labels_3d, 0, 4)
    return labels_3d


parser = argparse.ArgumentParser(
    description="3D FDTD brain segmentation simulation. Provide either a segmentation NIfTI, 4 BraTS modality paths, or a modalities directory."
)
parser.add_argument(
    "seg",
    nargs="?",
    default=None,
    help="Path to BraTS-style segmentation NIfTI (0,1,2,3). Omit if using --modalities or --modalities-dir.",
)
parser.add_argument(
    "--modalities",
    nargs=4,
    metavar=("FLAIR", "T1", "T1CE", "T2"),
    help="Paths to 4 BraTS NIfTI files (FLAIR, T1, T1CE, T2). Runs 3D U-Net segmentation then FDTD.",
)
parser.add_argument(
    "--modalities-dir",
    metavar="DIR",
    help="Path to a folder containing the four BraTS modalities (flair.nii, t1.nii, t1ce.nii, t2.nii or *_flair.nii, *_t1.nii, etc.). Auto-loads and runs segmentation.",
)
parser.add_argument(
    "--checkpoint",
    default=_DEFAULT_CHECKPOINT,
    help=f"Path to 3D U-Net checkpoint .pth (default: {_DEFAULT_CHECKPOINT})",
)
parser.add_argument(
    "--no-normal-brain",
    action="store_true",
    help="When using --modalities, do not add normal brain tissue (label 4); keep only tumor classes 1–3 and background 0.",
)
# --- Antenna optimization mode (Objective 4) ---
parser.add_argument(
    "--optimize-antenna",
    action="store_true",
    help="Run 4-quadrant APA antenna optimization to maximize SAR tumor/healthy ratio (Houle Ch.6 style).",
)
parser.add_argument(
    "--f0",
    type=float,
    default=100e6,
    help="Operating frequency in Hz for antenna optimization CW source (default: 100 MHz).",
)
parser.add_argument(
    "--opt-time-steps",
    type=int,
    default=700,
    help="Number of FDTD time steps per unit-quadrant run during optimization (default: 700).",
)
parser.add_argument(
    "--opt-phase-steps",
    type=int,
    default=24,
    help="Number of phase grid points per quadrant in coarse sweep (default: 24).",
)
parser.add_argument(
    "--opt-amp-steps",
    type=int,
    default=9,
    help="Number of amplitude grid points per quadrant in coarse sweep (default: 9).",
)
parser.add_argument(
    "--opt-amp-min",
    type=float,
    default=0.2,
    help="Minimum amplitude bound for optimization (default: 0.2).",
)
parser.add_argument(
    "--opt-amp-max",
    type=float,
    default=2.5,
    help="Maximum amplitude bound for optimization (default: 2.5).",
)
parser.add_argument(
    "--opt-refine-iters",
    type=int,
    default=8,
    help="Number of coordinate-descent refinement iterations (default: 8).",
)
parser.add_argument(
    "--opt-multi-start",
    type=int,
    default=3,
    help="Number of random multi-start initial phase offsets for optimization (default: 3). "
    "Helps escape local optima.",
)
parser.add_argument(
    "--opt-freq-sweep",
    nargs="+",
    type=float,
    default=None,
    help="List of frequencies (Hz) to sweep before full optimization. "
    "The best f0 is auto-selected. Example: --opt-freq-sweep 70e6 100e6 130e6 170e6 200e6",
)
parser.add_argument(
    "--opt-geom-offsets",
    nargs="+",
    type=int,
    default=None,
    help="List of source ring offsets (cells from PML) to sweep for geometry optimization. "
    "Small offsets (e.g. 8 10 12) keep the applicator ring in the air region surrounding the head. "
    "When --optimize-antenna is used and neither this nor --opt-geom-zplanes is set, defaults to 8 10 12.",
)
parser.add_argument(
    "--opt-geom-zplanes",
    nargs="+",
    type=int,
    default=None,
    help="List of z-plane indices for source placement. "
    "Example: --opt-geom-zplanes 30 41 50  (default: tumor centroid z).",
)
parser.add_argument(
    "--opt-penalty-weight",
    type=float,
    default=0.1,
    help="Penalty weight for healthy-tissue P95 SAR hotspot in objective. "
    "Effective objective = J - weight * P95_healthy_SAR / mean_tumor_SAR. "
    "Default: 0.1 to limit P95 healthy SAR; use 0 for pure J maximization.",
)
parser.add_argument(
    "--opt-parallel",
    type=int,
    default=1,
    help="Number of parallel workers for antenna optimization (frequency sweep, geometry sweep, multi-start). "
    "Default: 1 (serial). Use 2–4 for sweeps on 16 GB RAM; multi-start can use more (e.g. 8 on M3).",
)
parser.add_argument(
    "--opt-source-scale",
    type=float,
    default=1.0,
    help="Global scale factor for the optimized 4-quadrant source in the final FDTD run. "
    "SAR scales with scale². Use >1 for non-trivial temperature rise (e.g. 1e3--1e4). Default: 1.0.",
)
parser.add_argument(
    "--pulse-amplitude",
    type=float,
    default=100.0,
    help="Amplitude of the Gaussian pulse in the standard (non-optimized) FDTD run. "
    "SAR and temperature scale with amplitude squared. (default: 100.0)",
)
parser.add_argument(
    "--time-steps",
    type=int,
    default=500,
    help="Number of FDTD time steps for standard (non-optimized) run (default: 500). "
    "Ignored when --optimize-antenna is used (steps derived from frequency).",
)
parser.add_argument(
    "--max-dim",
    type=int,
    default=120,
    help="Maximum grid dimension; segmentation is downsampled if larger (default: 120).",
)
parser.add_argument(
    "--dx-mm",
    type=float,
    default=10.0,
    help="Grid resolution (voxel size) in mm (default: 10). Converted to meters internally. "
    "Paper recommends 1–5 mm for anatomical detail; 10 mm is used for backward compatibility.",
)
parser.add_argument(
    "--courant-factor",
    type=float,
    default=0.99,
    help="Safety factor for time step: dt = factor * dt_courant, where dt_courant is from the Courant stability condition (default: 0.99).",
)
parser.add_argument(
    "--benchmark-grid-sizes",
    nargs="*",
    type=int,
    default=None,
    metavar="N",
    help="Benchmark mode: run minimal FDTD for each N³ grid; write scalability JSON. "
    "Example: --benchmark-grid-sizes 100 200 300. Default grid sizes: 100 200 300.",
)
parser.add_argument(
    "--benchmark-time-steps",
    type=int,
    default=BENCHMARK_TIME_STEPS_DEFAULT,
    help=f"FDTD time steps per benchmark run (default: {BENCHMARK_TIME_STEPS_DEFAULT}).",
)
parser.add_argument(
    "--benchmark-grid-sizes-range",
    nargs=3,
    type=int,
    default=None,
    metavar=("A", "B", "S"),
    help="Benchmark grid sizes as range: min A, max B, step S (e.g. 50 200 50 → 50,100,150,200). "
    "Alternative to --benchmark-grid-sizes.",
)
parser.add_argument(
    "--stream-frames",
    action="store_true",
    default=True,
    dest="stream_frames",
    help="Stream E and SAR frames to disk during FDTD instead of keeping in memory (default: True). "
    "Use with --stream-frame-interval to control density. Enables full (or dense) timestep "
    "saving without OOM; animations can be built separately from saved frames.",
)
parser.add_argument(
    "--no-stream-frames",
    action="store_false",
    dest="stream_frames",
    help="Keep E and SAR frames in memory instead of streaming to disk (disables default streaming).",
)
parser.add_argument(
    "--stream-frame-interval",
    type=int,
    default=1,
    help="Save a frame every N timesteps when streaming (default: 1 = every step).",
)
parser.add_argument(
    "--skip-animations",
    action="store_true",
    help="Do not build or save MP4 animations (2D/3D). Frames are still saved to E_frames/, SAR_frames/, "
    "Temperature_frames/ when applicable; use build_animations_from_streamed_frames.py later to build videos.",
)
parser.add_argument(
    "--slice-timestep-images",
    action="store_true",
    help="When using --stream-frames: generate per-(slice, timestep) PNGs (E/SAR/T) for the dashboard. By default they are skipped.",
)
# Standard (non-optimized) run: pulse type and excitation location (ignored when --optimize-antenna)
parser.add_argument(
    "--pulse-type",
    type=str,
    default="gaussian",
    choices=["gaussian", "cw", "modulated_gaussian", "sinusoid", "sinusoid_no_ramp"],
    help="Source waveform: gaussian, cw, modulated_gaussian, sinusoid (ramp+sin), or sinusoid_no_ramp (default: gaussian).",
)
parser.add_argument(
    "--prop-direction",
    type=str,
    default="+y",
    choices=["+x", "-x", "+y", "-y", "+z", "-z"],
    help="Plane-wave propagation direction for gaussian/modulated_gaussian (default: +y). Ignored for cw/sinusoid/sinusoid_no_ramp.",
)
parser.add_argument(
    "--source-x",
    type=int,
    default=None,
    help="Grid index i for CW point source (standard run, pulse-type cw). Default: grid center. Clamped to interior.",
)
parser.add_argument(
    "--source-y",
    type=int,
    default=None,
    help="Grid index j for CW point source (standard run, pulse-type cw). Default: grid center.",
)
parser.add_argument(
    "--source-z",
    type=int,
    default=None,
    help="Grid index k for point source 1 (cw/sinusoid/sinusoid_no_ramp). Default: grid center.",
)
parser.add_argument(
    "--use-source-2",
    action="store_true",
    help="Enable second point source at antenna-like position (for cw/sinusoid/sinusoid_no_ramp). Use --source-x-2 etc. to override.",
)
parser.add_argument(
    "--use-source-3",
    action="store_true",
    help="Enable third point source at antenna-like position (for cw/sinusoid/sinusoid_no_ramp). Use --source-x-3 etc. to override.",
)
parser.add_argument(
    "--source-x-2",
    type=int,
    default=None,
    help="Grid index i for second point source. Used when --use-source-2; default antenna-like position if unset.",
)
parser.add_argument(
    "--source-y-2",
    type=int,
    default=None,
    help="Grid index j for second point source.",
)
parser.add_argument(
    "--source-z-2",
    type=int,
    default=None,
    help="Grid index k for second point source.",
)
parser.add_argument(
    "--source-x-3",
    type=int,
    default=None,
    help="Grid index i for third point source. Used when --use-source-3; default antenna-like if unset.",
)
parser.add_argument(
    "--source-y-3",
    type=int,
    default=None,
    help="Grid index j for third point source.",
)
parser.add_argument(
    "--source-z-3",
    type=int,
    default=None,
    help="Grid index k for third point source.",
)
parser.add_argument(
    "--source-ring-offset",
    type=int,
    default=10,
    help="Cells from boundary for default antenna-like positions of source 2/3 when --use-source-2/3 (default: 10).",
)
parser.add_argument(
    "--pulse-freq",
    type=float,
    default=100e6,
    help="Frequency (Hz) for modulated_gaussian, sinusoid, sinusoid_no_ramp, and cw (default: 100e6).",
)
parser.add_argument(
    "--cw-periods",
    type=int,
    default=None,
    help="When pulse-type is cw or sinusoid_no_ramp: set time_steps to this many periods and SAR from period 10. If unset, cw/sinusoid use min 15 periods; sinusoid_no_ramp uses --time-steps as-is.",
)
parser.add_argument(
    "--pulse-ramp-width",
    type=float,
    default=30.0,
    help="Gaussian ramp-up width (time steps) for CW and sinusoid (ramp+sin) soft start (default: 30). Ignored for sinusoid_no_ramp.",
)
args = parser.parse_args()

if __name__ == "__main__":
    # Create single timestamped results directory (only in main process; workers do not run this)
    # When BENCHMARK_RESULTS_DIR is set (by parent benchmark), use it so all subprocess output goes in one dir
    now = datetime.now()
    timestamp_str = now.strftime("%d%m%y-%H%M%S")  # DDMMYY-HHMMSS
    if os.environ.get("BENCHMARK_RESULTS_DIR"):
        RESULTS_DIR = os.path.abspath(os.environ["BENCHMARK_RESULTS_DIR"])
    else:
        RESULTS_DIR = os.path.join("results", f"{timestamp_str}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    PROGRESS_DIR = os.path.join("results", "uploads")
    PROGRESS_FILE = os.path.join(PROGRESS_DIR, "last_run_progress.json")
    # When BENCHMARK_GRID_SIZE is set (benchmark subprocess), put output in data/N, images/N, animations/N
    benchmark_grid_subdir = os.environ.get("BENCHMARK_GRID_SIZE")
    if benchmark_grid_subdir is not None:
        subdir = str(benchmark_grid_subdir)
        DATA_DIR = os.path.join(RESULTS_DIR, "data", subdir)
        IMAGES_DIR = os.path.join(RESULTS_DIR, "images", subdir)
        ANIMATIONS_DIR = os.path.join(RESULTS_DIR, "animations", subdir)
    else:
        DATA_DIR = os.path.join(RESULTS_DIR, "data")
        IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
        ANIMATIONS_DIR = os.path.join(RESULTS_DIR, "animations")
    E_FRAMES_DIR = os.path.join(DATA_DIR, "E_frames")
    SAR_FRAMES_DIR = os.path.join(DATA_DIR, "SAR_frames")
    TEMPERATURE_FRAMES_DIR = os.path.join(DATA_DIR, "Temperature_frames")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(E_FRAMES_DIR, exist_ok=True)
    os.makedirs(SAR_FRAMES_DIR, exist_ok=True)
    os.makedirs(TEMPERATURE_FRAMES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(ANIMATIONS_DIR, exist_ok=True)
    _write_progress("setup", "Output directories created", 0, [])
    print(f"Output directory: {RESULTS_DIR}/")
    print(f"  Data: {DATA_DIR}/")
    print(f"  E-frames: {E_FRAMES_DIR}/")
    print(f"  Images: {IMAGES_DIR}/")
    print(f"  Animations: {ANIMATIONS_DIR}/")

    # ----- Benchmark mode (Objective 5: scalability) -----
    in_benchmark_mode = (
        args.benchmark_grid_sizes is not None
        or getattr(args, "benchmark_grid_sizes_range", None) is not None
    )
    if in_benchmark_mode:
        # Benchmark runs skip animations (subprocess gets --skip-animations; no animations in minimal FDTD path)
        setattr(args, "skip_animations", True)
        # Build sizes from --benchmark-grid-sizes-range A B S or --benchmark-grid-sizes list
        if getattr(args, "benchmark_grid_sizes_range", None) is not None:
            a, b, s = args.benchmark_grid_sizes_range
            sizes = list(range(a, b + 1, s))
            if not sizes:
                sizes = BENCHMARK_GRID_SIZES_DEFAULT
        elif (
            args.benchmark_grid_sizes is not None and len(args.benchmark_grid_sizes) > 0
        ):
            sizes = args.benchmark_grid_sizes
        else:
            sizes = BENCHMARK_GRID_SIZES_DEFAULT
        steps = getattr(args, "benchmark_time_steps", BENCHMARK_TIME_STEPS_DEFAULT)
        has_anatomy = (
            (args.seg and os.path.isfile(args.seg))
            or (args.modalities is not None)
            or (args.modalities_dir is not None)
        )
        if has_anatomy:
            print(
                f"\nBenchmark mode (full pipeline with anatomy): grid sizes {sizes}, "
                f"{steps} time steps each."
            )
        else:
            print(
                f"\nBenchmark mode (FDTD-only, no anatomy): grid sizes {sizes}, "
                f"{steps} time steps each."
            )
        results = []

        if has_anatomy:
            # Full-pipeline benchmark: load segmentation once, resample to each N, run engine in subprocess
            labels_3d = _load_segmentation_for_benchmark(args)
            sx, sy, sz = labels_3d.shape
            engine_dir = os.path.dirname(os.path.abspath(__file__))
            engine_script = os.path.join(engine_dir, "fdtd_brain_simulation_engine.py")
            for N in sizes:
                print(f"  Running full pipeline N={N} ({N**3} voxels)...")
                zoom_factors = (N / sx, N / sy, N / sz)
                labels_n = ndimage.zoom(
                    labels_3d.astype(np.float32),
                    zoom_factors,
                    order=0,
                    mode="nearest",
                )
                labels_n = np.round(labels_n).astype(np.int32)
                labels_n = np.clip(labels_n, 0, 4)
                base_name = f"benchmark_anatomy_{N}"
                fd, temp_nii = tempfile.mkstemp(
                    suffix=".nii.gz", prefix=base_name + "_", dir=engine_dir
                )
                os.close(fd)
                try:
                    affine = np.diag([1.0, 1.0, 1.0, 1.0])
                    nib.save(
                        nib.Nifti1Image(labels_n.astype(np.int32), affine),
                        temp_nii,
                    )
                    cmd = [
                        sys.executable,
                        engine_script,
                        temp_nii,  # seg is positional
                        "--max-dim",
                        str(N),
                        "--time-steps",
                        str(steps),
                        "--skip-animations",
                    ]
                    t_before = time.time()
                    subprocess_env = os.environ.copy()
                    subprocess_env["BENCHMARK_RESULTS_DIR"] = os.path.abspath(
                        RESULTS_DIR
                    )
                    subprocess_env["BENCHMARK_GRID_SIZE"] = str(N)
                    subprocess.run(cmd, cwd=engine_dir, check=True, env=subprocess_env)
                finally:
                    try:
                        os.remove(temp_nii)
                    except OSError:
                        pass
                # Subprocess wrote to data/N/; find its performance JSON (mtime after start)
                perf_files = glob.glob(
                    os.path.join(DATA_DIR, str(N), "*_performance.json")
                )
                perf_files = [
                    p for p in perf_files if os.path.getmtime(p) >= t_before - 5
                ]
                perf_files.sort(key=os.path.getmtime, reverse=True)
                if perf_files:
                    with open(perf_files[0], "r") as f:
                        pm = json.load(f)
                    phases = pm.get("phases_s") or {}
                    run_metrics = {
                        "grid_shape": pm["grid_shape"],
                        "number_of_voxels": pm["number_of_voxels"],
                        "time_steps": pm["time_steps"],
                        "total_wall_time_s": round(
                            pm.get("total_simulation_time_s")
                            or pm.get("total_wall_time_s")
                            or 0,
                            6,
                        ),
                        "time_per_step_ms": pm.get("time_per_step_ms"),
                        "peak_memory_MB": pm.get("peak_memory_MB"),
                        "time_fdtd_s": phases.get("fdtd_simulation"),
                        "time_sar_s": phases.get("sar_computation"),
                        "time_thermal_s": phases.get("thermal_solver"),
                    }
                    results.append(run_metrics)
                    print(
                        f"    wall_time_s={run_metrics['total_wall_time_s']:.3f}, "
                        f"time_per_step_ms={run_metrics.get('time_per_step_ms')}, "
                        f"peak_memory_MB={run_metrics.get('peak_memory_MB')}"
                    )
                else:
                    print(f"    [WARN] No performance JSON found for N={N}")
        else:
            for N in sizes:
                print(f"  Running N={N} ({N**3} voxels)...")
                run_metrics = _run_minimal_fdtd_benchmark(
                    N,
                    N,
                    N,
                    steps,
                    dx_mm=getattr(args, "dx_mm", 10.0),
                    courant_factor=getattr(args, "courant_factor", 0.99),
                )
                results.append(run_metrics)
                print(
                    f"    wall_time_s={run_metrics['total_wall_time_s']:.3f}, "
                    f"time_per_step_ms={run_metrics['time_per_step_ms']:.4f}, "
                    f"peak_memory_MB={run_metrics['peak_memory_MB']}"
                )

        scalability_path = os.path.join(DATA_DIR, "scalability_benchmark_results.json")
        out = {
            "benchmark_time_steps": steps,
            "benchmark_full_pipeline": has_anatomy,
            "backend": "numpy_numba",
            "runs": results,
        }
        with open(scalability_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nScalability results written to {scalability_path}")

        # Automatically run scalability plotting script
        plot_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "plot_scalability_benchmark.py",
        )
        if os.path.isfile(plot_script):
            print(f"Running {plot_script}...")
            rc = subprocess.run(
                [sys.executable, plot_script, scalability_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if rc.returncode != 0:
                print(
                    f"  [WARN] plot_scalability_benchmark.py exited with code {rc.returncode}"
                )
        else:
            print(f"  [INFO] Plot script not found: {plot_script}")

        sys.exit(0)

    modality_paths = args.modalities
    use_modalities = (modality_paths is not None) or (args.modalities_dir is not None)
    t_start_pipeline = time.perf_counter()
    t_start_segmentation = t_start_pipeline
    volume_4d_ds = None  # downsampled modalities, used for previews when available
    if use_modalities:
        if args.modalities_dir is not None:
            flair_path, t1_path, t1ce_path, t2_path = _find_modalities_in_dir(
                args.modalities_dir
            )
        else:
            flair_path, t1_path, t1ce_path, t2_path = tuple(args.modalities)
        from brain_tumor_segmentation_model import run_segmentation_from_modalities

        labels_3d = run_segmentation_from_modalities(
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            args.checkpoint,
            extend_with_normal_brain=not args.no_normal_brain,
        )
        # Output base from first modality filename (or directory name when using --modalities-dir)
        if args.modalities_dir is not None:
            OUTPUT_BASE = os.path.basename(os.path.normpath(args.modalities_dir))
        else:
            OUTPUT_BASE = os.path.splitext(os.path.basename(flair_path))[0]
            if OUTPUT_BASE.endswith(".nii"):
                OUTPUT_BASE = os.path.splitext(OUTPUT_BASE)[0]
        print(f"Segmentation shape: {labels_3d.shape}")
    t_end_segmentation = time.perf_counter()
    _write_progress(
        "segmentation", "Segmentation complete", 10, ["setup", "segmentation"]
    )

    if not use_modalities:
        SEG_PATH = args.seg or os.environ.get(
            "BRAIN_SEGMENTATION_NII", "brain_segmentation.nii"
        )
        if not os.path.isfile(SEG_PATH):
            raise FileNotFoundError(
                f"Segmentation file not found: {SEG_PATH}. "
                "Use --seg path.nii, --modalities F T1 T1CE T2, or --modalities-dir DIR"
            )
        OUTPUT_BASE = os.path.splitext(os.path.basename(SEG_PATH))[0]
        if OUTPUT_BASE.endswith(".nii"):
            OUTPUT_BASE = os.path.splitext(OUTPUT_BASE)[0]
        img = nib.load(SEG_PATH)
        labels_3d = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
        if labels_3d.ndim != 3:
            raise ValueError(f"Expected 3D segmentation, got shape {labels_3d.shape}")
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(
            labels_3d, 0, 4
        )  # 0=bg, 1–3=tumor, 4=normal brain (if present)
        t_end_segmentation = time.perf_counter()

    # Downsample if too large (keep simulation tractable)
    max_dim = args.max_dim
    nx, ny, nz = labels_3d.shape
    orig_shape = labels_3d.shape  # for volume downsampling when use_modalities
    if max(nx, ny, nz) > max_dim:
        scale = max_dim / max(nx, ny, nz)
        order = 0  # nearest-neighbor for labels
        labels_3d = ndimage.zoom(
            labels_3d, (scale, scale, scale), order=order, mode="nearest"
        )
        labels_3d = np.round(labels_3d).astype(np.int32)
        labels_3d = np.clip(labels_3d, 0, 4)
        nx, ny, nz = labels_3d.shape
        print(f"Downsampled segmentation to ({nx}, {ny}, {nz}) (max_dim={max_dim})")
    _write_progress(
        "setup",
        f"Grid shape {nx}×{ny}×{nz}",
        12,
        ["setup", "segmentation"],
        {"grid_shape": [int(nx), int(ny), int(nz)]},
    )

    # 10 slices with biggest tumor area (axial, for slice-specific FDTD outputs and preview)
    from brain_tumor_segmentation_model import select_slices_biggest_tumor

    top_10_slice_indices = select_slices_biggest_tumor(labels_3d, n_slices=10, axis=2)
    print(f"Top 10 axial slices by tumor area (indices): {top_10_slice_indices}")

    # When using modalities: load volume, downsample to match labels_3d, save 10 individual slice previews
    if use_modalities:
        from brain_tumor_segmentation_model import (
            load_patient_volume_from_paths,
            create_slice_preview_figure_streamlit_style,
        )

        volume_4d = load_patient_volume_from_paths(
            flair_path, t1_path, t1ce_path, t2_path
        )
        # Downsample volume to (4, nx, ny, nz) to match labels_3d
        zoom_factors = (
            1,
            nx / orig_shape[0],
            ny / orig_shape[1],
            nz / orig_shape[2],
        )
        volume_4d_ds = ndimage.zoom(volume_4d, zoom_factors, order=1, mode="nearest")
        for k in top_10_slice_indices:
            fig = create_slice_preview_figure_streamlit_style(
                volume_4d_ds, labels_3d, k, slice_axis=2
            )
            if fig is not None:
                preview_path = os.path.join(
                    IMAGES_DIR, f"{OUTPUT_BASE}_tumor_preview_slice_{k}.png"
                )
                fig.savefig(preview_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  Slice {k} preview saved to {preview_path}")
        print(f"10 individual slice previews saved to {IMAGES_DIR}/")

    # Save slice indices for the "10 simulations" (one FDTD run, 10 slice-specific outputs)
    slice_indices_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_slice_indices.txt")
    with open(slice_indices_path, "w") as f:
        f.write("\n".join(map(str, top_10_slice_indices)))
    print(f"Slice indices saved to {slice_indices_path}")

    simulation_size_x = int(nx)
    simulation_size_y = int(ny)
    simulation_size_z = int(nz)
    mid_z_index = simulation_size_z // 2
    print(
        f"Simulation grid: {simulation_size_x} x {simulation_size_y} x {simulation_size_z}"
    )

    # ----- Render and visualize tumor before FDTD -----
    # Tumor = labels 1 (necrotic), 2 (edema), 3 (enhancing)
    tumor_mask = (labels_3d >= 1) & (labels_3d <= 3)

    # RGB overlay: 0=black, 1=red, 2=green, 3=blue, 4=normal brain (BraTS + normal tissue)
    def labels_to_rgb(lab):
        lab = np.clip(lab, 0, 4).astype(np.int32)
        rgb = np.zeros((*lab.shape, 3))
        rgb[lab == 0] = [0.15, 0.15, 0.15]  # background: dark gray
        rgb[lab == 1] = [1, 0.2, 0.2]  # necrotic: red
        rgb[lab == 2] = [0.2, 0.8, 0.2]  # edema: green
        rgb[lab == 3] = [0.2, 0.2, 1]  # enhancing: blue
        rgb[lab == 4] = [0.6, 0.55, 0.5]  # normal brain: tan/light gray
        return rgb

    fig_viz = plt.figure(figsize=(14, 10))
    ax1 = fig_viz.add_subplot(2, 2, 1)
    ax2 = fig_viz.add_subplot(2, 2, 2)
    ax3 = fig_viz.add_subplot(2, 2, 3)
    ax4 = fig_viz.add_subplot(2, 2, 4, projection="3d")

    # Mid slices with label coloring
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    slice_axial = labels_to_rgb(labels_3d[:, :, cz])
    slice_sagittal = labels_to_rgb(labels_3d[cx, :, :])
    slice_coronal = labels_to_rgb(labels_3d[:, cy, :])

    ax1.imshow(slice_axial, origin="lower")
    ax1.set_title(f"Axial (z={cz})")
    ax1.axis("off")

    ax2.imshow(slice_sagittal, origin="lower")
    ax2.set_title(f"Sagittal (x={cx})")
    ax2.axis("off")

    ax3.imshow(slice_coronal, origin="lower")
    ax3.set_title(f"Coronal (y={cy})")
    ax3.axis("off")

    # 3D tumor voxels (downsample for speed)
    step = max(1, min(nx, ny, nz) // 30)
    ii, jj, kk = np.where(tumor_mask)
    ii, jj, kk = ii[::step], jj[::step], kk[::step]
    if len(ii) > 0:
        lab = labels_3d[ii, jj, kk]
        colors_3d = np.array([[1, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 1]])[
            np.clip(lab.astype(int) - 1, 0, 2)
        ]
        ax4.scatter(ii, jj, kk, c=colors_3d, s=2, alpha=0.6)
    ax4.set_title("3D tumor (red=necrotic, green=edema, blue=enhancing)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.view_init(elev=20, azim=45)

    fig_viz.suptitle(
        "Brain segmentation – tumor visualization (before FDTD)", fontsize=12
    )
    plt.tight_layout()
    preview_path = os.path.join(IMAGES_DIR, f"{OUTPUT_BASE}_tumor_preview.png")
    plt.savefig(preview_path, dpi=120, bbox_inches="tight")
    print(f"Tumor preview saved to {preview_path}")
    plt.close(fig_viz)

    # PML thickness: scale with smallest grid dimension (4–16 cells)
    npml = max(
        4, min(16, min(simulation_size_x, simulation_size_y, simulation_size_z) // 10)
    )
    # Source 1 default: X-low face with offset (antenna-like); source 2/3 use center or other faces
    source_x = npml + 2
    source_y = simulation_size_y // 2
    source_z = simulation_size_z // 2
    ia = npml
    ja = npml
    ka = npml
    ib = simulation_size_x - npml - 1
    jb = simulation_size_y - npml - 1
    kb = simulation_size_z - npml - 1

    # step size (m), time step (before allocating arrays that depend on dt/epsz)
    # dx from CLI --dx-mm (default 10 mm); paper recommends 1–5 mm for anatomical detail
    dx = args.dx_mm * 1e-3  # mm -> m
    # Uniform grid: dy = dz = dx. Courant condition (paper Sec. 3.2):
    # dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)); c = speed of light in vacuum
    c_light = 2.99792458e8  # m/s
    dy, dz = dx, dx
    dt_courant = 1.0 / (
        c_light * sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    )
    dt = args.courant_factor * dt_courant
    epsz = 8.854e-12

    # Tissue properties at ~100 MHz.
    # BraTS labels: 0=background, 1=necrotic, 2=edema, 3=enhancing; 4=normal brain (optional).
    # Map 0 -> free space (air). Format: (eps_r, sigma (S/m), rho (kg/m³)).
    # Dielectric/thermal values from literature (ICNIRP, Gabriel et al., ~100 MHz).
    TISSUE_TABLE = {
        0: (1.0, 0.0, 0.0),  # background = free space (air); no SAR
        1: (60.0, 0.8, 1050.0),  # necrotic tumor
        2: (60.0, 0.8, 1050.0),  # edema
        3: (60.0, 0.8, 1050.0),  # enhancing tumor
        4: (50.0, 0.6, 1046.0),  # normal brain (parenchyma-like average GM/WM)
    }
    # Thermal conductivity k (W/(m·K)) for steady-state bioheat
    K_TISSUE = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
    T_BOUNDARY_CELSIUS = 37.0  # Dirichlet at domain boundaries (paper)

    # field values
    Ex = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ey = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ez = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ix = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Iy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Iz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Dx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Dy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Dz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iDx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iDy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iDz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Hx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Hy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Hz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iHx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iHy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iHz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    # permittivity, conductivity, Hx/Ez incident
    eps_x = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
    eps_y = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
    eps_z = np.ones((simulation_size_x, simulation_size_y, simulation_size_z))
    conductivity_x = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    conductivity_y = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    conductivity_z = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    sigma_x = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    sigma_y = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    sigma_z = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    rho = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    k_3d = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    hx_inc = np.zeros(simulation_size_y)
    ez_inc = np.zeros(simulation_size_y)

    # Fill material arrays from segmentation (label at each voxel -> eps, sigma, rho, k)
    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                lab = int(labels_3d[i, j, k])
                if lab not in TISSUE_TABLE:
                    lab = 0
                eps_r, sigma_val, rho_val = TISSUE_TABLE[lab]
                k_3d[i, j, k] = K_TISSUE.get(lab, 0.0)
                denom = eps_r + (sigma_val * dt / epsz)
                c = 1.0 / denom if denom > 0 else 1.0
                cond = sigma_val * dt / epsz
                eps_x[i, j, k] = c
                eps_y[i, j, k] = c
                eps_z[i, j, k] = c
                conductivity_x[i, j, k] = cond
                conductivity_y[i, j, k] = cond
                conductivity_z[i, j, k] = cond
                sigma_x[i, j, k] = sigma_val
                sigma_y[i, j, k] = sigma_val
                sigma_z[i, j, k] = sigma_val
                rho[i, j, k] = rho_val

    # Sanity check: verify tissue/tumor geometry is in the simulation
    n_air = np.sum(labels_3d == 0)
    n_tumor = np.sum((labels_3d >= 1) & (labels_3d <= 3))
    n_normal_brain = np.sum(labels_3d == 4)
    print(
        f"\nMaterial fill: voxels air (0): {n_air}, tumor (1–3): {n_tumor}, normal brain (4): {n_normal_brain}"
    )
    print(f"  rho: min={np.min(rho):.1f}, max={np.max(rho):.1f} kg/m³")
    print(f"  eps_x (inverse): min={np.min(eps_x):.6f}, max={np.max(eps_x):.6f}")
    # Save a 2D slice of the geometry used in FDTD (same view as later max-projection)
    geom_slice = labels_3d[:, :, simulation_size_z // 2]
    fig_check = plt.figure(figsize=(6, 5))
    ax_check = fig_check.add_subplot(111)
    ax_check.imshow(labels_to_rgb(geom_slice), origin="lower")
    ax_check.set_title("FDTD geometry (mid-Z): dark=air, R/G/B=tumor, tan=normal brain")
    ax_check.set_xlabel("Y (cells)")
    ax_check.set_ylabel("X (cells)")
    fig_check.savefig(
        os.path.join(IMAGES_DIR, f"{OUTPUT_BASE}_fdtd_geometry_slice.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig_check)
    print(
        f"  Saved FDTD geometry slice to {IMAGES_DIR}/{OUTPUT_BASE}_fdtd_geometry_slice.png"
    )

    # Hold optimized antenna parameters for Option A: full time-domain FDTD with 4 quadrants
    # (set at end of optimization block; used later in main FDTD branch)
    opt_quad_sources = None
    opt_alphas = None
    opt_thetas = None
    opt_f0 = None

    # ===========================================================================
    # ANTENNA OPTIMIZATION MODE (--optimize-antenna)
    # 4-quadrant APA, Houle Ch.6 style superposition + J-ratio optimization
    # Enhanced: frequency sweep, geometry sweep, multi-start, robust objective
    # ===========================================================================
    antenna_optimization_s = None
    do_geom_sweep = False  # set True inside optimize_antenna when doing geometry sweep
    if args.optimize_antenna:
        import json as _json_opt

        _write_progress(
            "antenna_optimization",
            "Antenna optimization starting...",
            15,
            ["setup", "segmentation"],
        )
        print("\n" + "=" * 72)
        print("ANTENNA OPTIMIZATION MODE (4-quadrant APA, Houle Ch.6 style)")
        print("=" * 72)
        t_opt_start = time.perf_counter()

        # --- Build tissue masks for J-ratio ---
        opt_tumor_mask = (labels_3d >= 1) & (labels_3d <= 3)
        opt_healthy_mask = labels_3d == 4  # normal brain
        if not np.any(opt_healthy_mask):
            opt_healthy_mask = (labels_3d == 0) & (rho > 0)
        if not np.any(opt_healthy_mask):
            opt_healthy_mask = (labels_3d == 0) & (rho > 0)
        print(f"  Tumor voxels for J:   {np.sum(opt_tumor_mask)}")
        print(f"  Healthy voxels for J: {np.sum(opt_healthy_mask)}")

        # Average conductivity array for SAR computation
        sigma_avg_opt = (sigma_x + sigma_y + sigma_z) / 3.0

        # Tumor centroid (for geometry sweep defaults)
        _tumor_coords = np.argwhere(opt_tumor_mask)
        tumor_centroid_z = (
            int(np.mean(_tumor_coords[:, 2]))
            if len(_tumor_coords) > 0
            else simulation_size_z // 2
        )
        print(f"  Tumor centroid z-index: {tumor_centroid_z}")

        # --- Helper: run 4 unit FDTD sims for a given geometry and frequency ---
        def _run_unit_fields(f0_val, quad_srcs):
            fields = []
            for qi, qs in enumerate(quad_srcs):
                print(
                    f"    Quadrant {qs['quadrant']} (f0={f0_val/1e6:.1f} MHz, "
                    f"{args.opt_time_steps} steps, gap={qs['gap']})..."
                )
                t_q = time.perf_counter()
                cplx = run_unit_quadrant_fdtd(
                    quadrant_source=qs,
                    simulation_size_x=simulation_size_x,
                    simulation_size_y=simulation_size_y,
                    simulation_size_z=simulation_size_z,
                    eps_x=eps_x,
                    eps_y=eps_y,
                    eps_z=eps_z,
                    conductivity_x=conductivity_x,
                    conductivity_y=conductivity_y,
                    conductivity_z=conductivity_z,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    sigma_z=sigma_z,
                    rho=rho,
                    dx=dx,
                    dt=dt,
                    epsz=epsz,
                    npml=npml,
                    f0=f0_val,
                    time_steps_opt=args.opt_time_steps,
                )
                fields.append(cplx)
                print(
                    f"      Done in {time.perf_counter()-t_q:.1f}s, peak |Ez|={np.max(np.abs(cplx)):.6g}"
                )
            return fields

        # --- Helper: quick baseline J for a set of complex fields ---
        def _quick_baseline_J(fields):
            E_bl = synthesize_total_field(fields, np.ones(4), np.zeros(4))
            sar_bl = compute_sar_from_complex_field(E_bl, sigma_avg_opt, rho)
            J_bl, _, _ = compute_j_ratio(sar_bl, opt_tumor_mask, opt_healthy_mask)
            return J_bl

        # ===================================================================
        # STEP 1: Frequency sweep (if --opt-freq-sweep provided)
        # ===================================================================
        n_workers_sweep = min(args.opt_parallel, 4) if args.opt_parallel > 1 else 1
        if args.opt_parallel > 1:
            print(
                f"\n  Using {args.opt_parallel} parallel workers (sweeps capped at {n_workers_sweep} for memory)."
            )
        freq_sweep_list = args.opt_freq_sweep
        if freq_sweep_list is not None and len(freq_sweep_list) > 1:
            print(f"\n  FREQUENCY SWEEP over {len(freq_sweep_list)} frequencies...")
            if n_workers_sweep > 1:
                with ProcessPoolExecutor(max_workers=n_workers_sweep) as executor:
                    futures = {
                        executor.submit(
                            _eval_frequency_one,
                            f0_cand,
                            simulation_size_x,
                            simulation_size_y,
                            simulation_size_z,
                            args.opt_time_steps,
                            eps_x,
                            eps_y,
                            eps_z,
                            conductivity_x,
                            conductivity_y,
                            conductivity_z,
                            sigma_x,
                            sigma_y,
                            sigma_z,
                            rho,
                            dx,
                            dt,
                            epsz,
                            opt_tumor_mask,
                            opt_healthy_mask,
                        ): f0_cand
                        for f0_cand in freq_sweep_list
                    }
                    freq_results = []
                    for future in as_completed(futures):
                        f0_cand = futures[future]
                        try:
                            r = future.result()
                            freq_results.append(r)
                            print(
                                f"  [f0={f0_cand/1e6:.1f} MHz] Baseline J = {r['J_baseline']:.6f}"
                            )
                        except Exception as e:
                            print(f"  [f0={f0_cand/1e6:.1f} MHz] Error: {e}")
            else:
                freq_results = []
                for f0_cand in freq_sweep_list:
                    print(f"\n  --- f0 = {f0_cand/1e6:.1f} MHz ---")
                    qs_default = build_quadrant_sources(
                        simulation_size_x,
                        simulation_size_y,
                        simulation_size_z,
                        npml=npml,
                    )
                    fields_cand = _run_unit_fields(f0_cand, qs_default)
                    J_cand = _quick_baseline_J(fields_cand)
                    freq_results.append({"f0": f0_cand, "J_baseline": J_cand})
                    print(f"    Baseline J = {J_cand:.6f}")
            # Pick best frequency
            freq_results.sort(key=lambda x: x["J_baseline"], reverse=True)
            best_f0 = freq_results[0]["f0"]
            print(
                f"\n  Frequency sweep result: best f0 = {best_f0/1e6:.1f} MHz "
                f"(baseline J = {freq_results[0]['J_baseline']:.6f})"
            )
            print(
                f"  All results: {[(f'{r['f0']/1e6:.0f}MHz', f'J={r['J_baseline']:.4f}') for r in freq_results]}"
            )
            # Save frequency sweep results
            freq_sweep_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_freq_sweep.json")
            with open(freq_sweep_path, "w") as f:
                _json_opt.dump(freq_results, f, indent=2)
            args.f0 = best_f0  # Override f0 for subsequent steps
        elif freq_sweep_list is not None and len(freq_sweep_list) == 1:
            args.f0 = freq_sweep_list[0]

        # ===================================================================
        # STEP 2: Geometry sweep (if --opt-geom-offsets or --opt-geom-zplanes)
        # Restrict to small offsets (8, 10, 12) by default so applicator stays in air.
        # ===================================================================
        if args.opt_geom_offsets is not None:
            geom_offsets = args.opt_geom_offsets
        elif args.opt_geom_zplanes is not None:
            geom_offsets = [8, 10, 12]  # default small offsets when only z-planes set
        else:
            geom_offsets = [
                8,
                10,
                12,
            ]  # default when optimizing: small offsets (applicator in air)
        geom_zplanes = (
            args.opt_geom_zplanes if args.opt_geom_zplanes is not None else [None]
        )
        do_geom_sweep = (
            (args.opt_geom_offsets is not None)
            or (args.opt_geom_zplanes is not None)
            or args.optimize_antenna
        )  # when optimizing, always do at least one sweep (with default geom)

    # Defaults: set by geometry sweep when do_geom_sweep is True; safe fallbacks otherwise.
    _best_offset = None
    _best_zplane = None

    # ===================================================================
    # STEP 1: Frequency sweep (if --opt-freq-sweep provided)
    # ===================================================================
    freq_sweep_list = args.opt_freq_sweep
    if freq_sweep_list is not None and len(freq_sweep_list) > 1:
        print(f"\n  FREQUENCY SWEEP over {len(freq_sweep_list)} frequencies...")
        freq_results = []
        for f0_cand in freq_sweep_list:
            print(f"\n  --- f0 = {f0_cand/1e6:.1f} MHz ---")
            qs_default = build_quadrant_sources(
                simulation_size_x, simulation_size_y, simulation_size_z, npml=npml
            )
            fields_cand = _run_unit_fields(f0_cand, qs_default)
            J_cand = _quick_baseline_J(fields_cand)
            freq_results.append({"f0": f0_cand, "J_baseline": J_cand})
            print(f"    Baseline J = {J_cand:.6f}")
        # Pick best frequency
        freq_results.sort(key=lambda x: x["J_baseline"], reverse=True)
        best_f0 = freq_results[0]["f0"]
        print(
            f"\n  FINAL OPTIMIZATION (f0={args.f0/1e6:.1f} MHz, "
            f"offset={_best_offset or 'default'}, z={_best_zplane or 'default'})"
        )

        # Build sources with best geometry
        quad_sources = build_quadrant_sources(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            npml=npml,
            ring_offset=_best_offset,
            z_plane=_best_zplane,
        )
        for qs in quad_sources:
            print(
                f"  Quadrant {qs['quadrant']}: gap={qs['gap']}, arm voxels={len(qs['arms'])}"
            )

        # Run 4 unit-excitation FDTD simulations
        print(f"\n  Running 4 unit-excitation FDTD sims...")
        complex_fields = _run_unit_fields(args.f0, quad_sources)

    if do_geom_sweep:
        geom_combos = [(g_off, g_z) for g_off in geom_offsets for g_z in geom_zplanes]
        print(
            f"\n  GEOMETRY SWEEP: offsets={geom_offsets}, z-planes={geom_zplanes} "
            f"({len(geom_combos)} combos)"
        )
        geom_results = []
        if n_workers_sweep > 1 and len(geom_combos) > 1:
            print(
                f"  Running geometry sweep in parallel ({n_workers_sweep} workers)..."
            )
            with ProcessPoolExecutor(max_workers=n_workers_sweep) as executor:
                futures = {
                    executor.submit(
                        _eval_geom_one,
                        g_off,
                        g_z,
                        args.f0,
                        simulation_size_x,
                        simulation_size_y,
                        simulation_size_z,
                        args.opt_time_steps,
                        eps_x,
                        eps_y,
                        eps_z,
                        conductivity_x,
                        conductivity_y,
                        conductivity_z,
                        sigma_x,
                        sigma_y,
                        sigma_z,
                        rho,
                        dx,
                        dt,
                        epsz,
                        opt_tumor_mask,
                        opt_healthy_mask,
                    ): (g_off, g_z)
                    for g_off, g_z in geom_combos
                }
                for future in as_completed(futures):
                    g_off, g_z = futures[future]
                    try:
                        r = future.result()
                        geom_results.append(r)
                        print(
                            f"  [offset={g_off}, z={g_z}] Baseline J = {r['J_baseline']:.6f}"
                        )
                    except Exception as exc:
                        print(f"  [offset={g_off}, z={g_z}] Error: {exc}")
        else:
            for g_off, g_z in geom_combos:
                label = f"offset={g_off or 'default'}, z={g_z or 'default'}"
                print(f"\n  --- Geometry: {label} ---")
                qs_cand = build_quadrant_sources(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    npml=npml,
                    ring_offset=g_off,
                    z_plane=g_z,
                )
                for qs in qs_cand:
                    print(f"    Q{qs['quadrant']}: gap={qs['gap']}")
                fields_cand = _run_unit_fields(args.f0, qs_cand)
                J_cand = _quick_baseline_J(fields_cand)
                geom_results.append(
                    {
                        "ring_offset": g_off,
                        "z_plane": g_z,
                        "J_baseline": J_cand,
                        "gaps": [qs["gap"] for qs in qs_cand],
                    }
                )
                print(f"    Baseline J = {J_cand:.6f}")
        geom_results.sort(key=lambda x: x["J_baseline"], reverse=True)
        best_geom = geom_results[0]
        _best_offset = best_geom["ring_offset"]
        _best_zplane = best_geom["z_plane"]
        print(
            f"  Best geometry: offset={_best_offset or 'default'}, z={_best_zplane or 'default'}, "
            f"J_baseline={best_geom['J_baseline']:.6f}"
        )

        # Build sources for best geometry and run 4 unit FDTDs (needed for optimize_quadrant_controls)
        quad_sources = build_quadrant_sources(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            npml=npml,
            ring_offset=_best_offset,
            z_plane=_best_zplane,
        )
        print(f"\n  Running 4 unit-excitation FDTD sims (best geometry)...")
        complex_fields = _run_unit_fields(args.f0, quad_sources)

        # Run optimization (multi-start + robust objective)
        print(
            f"\n  Starting optimization (phase_steps={args.opt_phase_steps}, "
            f"amp_steps={args.opt_amp_steps}, refine_iters={args.opt_refine_iters}, "
            f"multi_start={args.opt_multi_start}, penalty_weight={args.opt_penalty_weight})..."
        )
        best_alphas, best_thetas, best_J, opt_trace, total_evals = (
            optimize_quadrant_controls(
                complex_fields=complex_fields,
                sigma_avg=sigma_avg_opt,
                rho=rho,
                tumor_mask=opt_tumor_mask,
                healthy_mask=opt_healthy_mask,
                n_quadrants=4,
                phase_steps=args.opt_phase_steps,
                amp_steps=args.opt_amp_steps,
                amp_range=(args.opt_amp_min, args.opt_amp_max),
                refine_iterations=args.opt_refine_iters,
                multi_start=args.opt_multi_start,
                penalty_weight=args.opt_penalty_weight,
                parallel_workers=args.opt_parallel,
            )
        )

    if args.optimize_antenna:
        # Run 4 unit-excitation FDTD simulations (skip if already done in geometry sweep)
        if not do_geom_sweep:
            print(f"\n  Running 4 unit-excitation FDTD sims...")
            complex_fields = _run_unit_fields(args.f0, quad_sources)

        # Baseline: equal amplitude, zero phase
        baseline_alphas = np.ones(4)
        baseline_thetas = np.zeros(4)
        E_baseline = synthesize_total_field(
            complex_fields, baseline_alphas, baseline_thetas
        )
        sar_baseline = compute_sar_from_complex_field(E_baseline, sigma_avg_opt, rho)
        J_baseline, mt_bl, mh_bl = compute_j_ratio(
            sar_baseline, opt_tumor_mask, opt_healthy_mask
        )
        _, _, _, _, p95_bl = compute_robust_objective(
            sar_baseline, opt_tumor_mask, opt_healthy_mask
        )
        print(f"\n  Baseline (equal amp, zero phase): J = {J_baseline:.6f}")
        print(
            f"    mean SAR tumor = {mt_bl:.6g}, mean SAR healthy = {mh_bl:.6g}, P95 healthy = {p95_bl:.6g}"
        )

        # Run optimization (multi-start + robust objective); skip if already done in geometry sweep
        if not do_geom_sweep:
            print(
                f"\n  Starting optimization (phase_steps={args.opt_phase_steps}, "
                f"amp_steps={args.opt_amp_steps}, refine_iters={args.opt_refine_iters}, "
                f"multi_start={args.opt_multi_start}, penalty_weight={args.opt_penalty_weight})..."
            )
            best_alphas, best_thetas, best_J, opt_trace, total_evals = (
                optimize_quadrant_controls(
                    complex_fields=complex_fields,
                    sigma_avg=sigma_avg_opt,
                    rho=rho,
                    tumor_mask=opt_tumor_mask,
                    healthy_mask=opt_healthy_mask,
                    n_quadrants=4,
                    phase_steps=args.opt_phase_steps,
                    amp_steps=args.opt_amp_steps,
                    amp_range=(args.opt_amp_min, args.opt_amp_max),
                    refine_iterations=args.opt_refine_iters,
                    multi_start=args.opt_multi_start,
                    penalty_weight=args.opt_penalty_weight,
                    parallel_workers=args.opt_parallel,
                )
            )
        E_optimized = synthesize_total_field(complex_fields, best_alphas, best_thetas)
        sar_optimized = compute_sar_from_complex_field(E_optimized, sigma_avg_opt, rho)
        J_opt, mt_opt, mh_opt = compute_j_ratio(
            sar_optimized, opt_tumor_mask, opt_healthy_mask
        )
        _, _, _, _, p95_opt = compute_robust_objective(
            sar_optimized, opt_tumor_mask, opt_healthy_mask
        )
        t_opt_end = time.perf_counter()
        antenna_optimization_s = round(float(t_opt_end - t_opt_start), 4)

        # --- Summary ---
        print("\n" + "-" * 60)
        print("ANTENNA OPTIMIZATION RESULTS")
        print("-" * 60)
        print(f"  Operating frequency:  {args.f0/1e6:.1f} MHz")
        print(
            f"  Geometry: offset={_best_offset or 'default'}, z_plane={_best_zplane or 'default'}"
        )
        print(f"  Baseline  J = {J_baseline:.6f}  (equal amp, zero phase)")
        print(
            f"  Optimized J = {J_opt:.6f}  (improvement: {(J_opt/max(J_baseline,1e-30) - 1)*100:.1f}%)"
        )
        print(f"  Best amplitudes: {best_alphas.tolist()}")
        print(f"  Best phases (rad): {best_thetas.tolist()}")
        print(f"  Best phases (deg): {[f'{np.degrees(t):.1f}' for t in best_thetas]}")
        print(f"  Mean SAR tumor (optimized):   {mt_opt:.6g} W/kg")
        print(f"  Mean SAR healthy (optimized):  {mh_opt:.6g} W/kg")
        print(f"  P95 SAR healthy (baseline):    {p95_bl:.6g} W/kg")
        print(f"  P95 SAR healthy (optimized):   {p95_opt:.6g} W/kg")
        print(f"  Total evaluations: {total_evals}")
        print(f"  Total optimization time: {t_opt_end - t_opt_start:.1f}s")

        # --- Validation: sanity checks ---
        print("\n  Validation checks:")
        if J_opt >= J_baseline:
            print(
                f"    [PASS] Optimized J ({J_opt:.4f}) >= baseline J ({J_baseline:.4f})"
            )
        else:
            print(
                f"    [WARN] Optimized J ({J_opt:.4f}) < baseline J ({J_baseline:.4f}) – unexpected"
            )
        if np.all(best_alphas >= args.opt_amp_min) and np.all(
            best_alphas <= args.opt_amp_max
        ):
            print(
                f"    [PASS] All amplitudes within [{args.opt_amp_min}, {args.opt_amp_max}]"
            )
        else:
            print(f"    [WARN] Some amplitudes out of bounds: {best_alphas.tolist()}")
        if np.all(np.abs(best_thetas) <= 2 * np.pi + 0.01):
            print(f"    [PASS] All phases within [-2π, 2π]")
        else:
            print(f"    [WARN] Some phases out of range: {best_thetas.tolist()}")
        if mt_opt > 0:
            print(f"    [PASS] Mean tumor SAR > 0 ({mt_opt:.6g})")
        else:
            print(f"    [WARN] Mean tumor SAR is zero or negative")
        # Check if healthy P95 reduced
        if p95_opt <= p95_bl:
            print(f"    [PASS] P95 healthy SAR reduced ({p95_bl:.6g} -> {p95_opt:.6g})")
        else:
            print(
                f"    [INFO] P95 healthy SAR increased ({p95_bl:.6g} -> {p95_opt:.6g})"
            )
        # Check if amplitudes hit bounds (warns user to widen)
        at_lower = np.sum(np.abs(best_alphas - args.opt_amp_min) < 0.01)
        at_upper = np.sum(np.abs(best_alphas - args.opt_amp_max) < 0.01)
        if at_lower > 0 or at_upper > 0:
            print(
                f"    [INFO] {at_lower} amp(s) at lower bound, {at_upper} at upper bound. "
                f"Consider widening --opt-amp-min/--opt-amp-max."
            )
        else:
            print(f"    [PASS] No amplitudes at bounds – search space sufficient.")

        # --- Save optimization artifacts ---
        opt_results = {
            "f0_Hz": args.f0,
            "time_steps_per_quadrant": args.opt_time_steps,
            "n_quadrants": 4,
            "quadrant_gaps": [qs["gap"] for qs in quad_sources],
            "geometry": {"ring_offset": _best_offset, "z_plane": _best_zplane},
            "baseline": {
                "alphas": baseline_alphas.tolist(),
                "thetas_rad": baseline_thetas.tolist(),
                "J": float(J_baseline),
                "mean_sar_tumor": float(mt_bl),
                "mean_sar_healthy": float(mh_bl),
                "p95_sar_healthy": float(p95_bl),
            },
            "optimized": {
                "alphas": best_alphas.tolist(),
                "thetas_rad": best_thetas.tolist(),
                "thetas_deg": np.degrees(best_thetas).tolist(),
                "J": float(J_opt),
                "mean_sar_tumor": float(mt_opt),
                "mean_sar_healthy": float(mh_opt),
                "p95_sar_healthy": float(p95_opt),
                "improvement_pct": float((J_opt / max(J_baseline, 1e-30) - 1) * 100),
            },
            "search_config": {
                "phase_steps": args.opt_phase_steps,
                "amp_steps": args.opt_amp_steps,
                "amp_range": [args.opt_amp_min, args.opt_amp_max],
                "refine_iterations": args.opt_refine_iters,
                "multi_start": args.opt_multi_start,
                "penalty_weight": args.opt_penalty_weight,
                "opt_source_scale": args.opt_source_scale,
            },
            "total_evaluations": total_evals,
            "total_time_s": float(t_opt_end - t_opt_start),
        }
        opt_results_path = os.path.join(
            DATA_DIR, f"{OUTPUT_BASE}_antenna_optimization.json"
        )
        with open(opt_results_path, "w") as f:
            _json_opt.dump(opt_results, f, indent=2)
        print(f"\n  Optimization results saved to {opt_results_path}")

        # Save optimization trace
        opt_trace_path = os.path.join(
            DATA_DIR, f"{OUTPUT_BASE}_optimization_trace.json"
        )
        with open(opt_trace_path, "w") as f:
            _json_opt.dump(opt_trace, f, indent=2)
        print(f"  Optimization trace saved to {opt_trace_path}")

        # Save optimized SAR as NumPy
        np.save(
            os.path.join(DATA_DIR, f"{OUTPUT_BASE}_sar_optimized.npy"), sar_optimized
        )
        np.save(os.path.join(DATA_DIR, f"{OUTPUT_BASE}_sar_baseline.npy"), sar_baseline)
        print(f"  Optimized and baseline SAR arrays saved to {DATA_DIR}/")

        # --- Comparative visualization: baseline vs optimized SAR (mid-Z slice) ---
        mid_z = simulation_size_z // 2
        fig_comp, axes_comp = plt.subplots(1, 3, figsize=(18, 5))

        ax_bl = axes_comp[0]
        sar_bl_slice = sar_baseline[:, :, mid_z]
        im_bl = ax_bl.imshow(
            sar_bl_slice,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=max(np.max(sar_bl_slice), 1e-12),
        )
        ax_bl.set_title(f"Baseline SAR (z={mid_z})\nJ = {J_baseline:.4f}")
        ax_bl.set_xlabel("Y (cells)")
        ax_bl.set_ylabel("X (cells)")
        plt.colorbar(im_bl, ax=ax_bl, label="SAR (W/kg)")

        ax_opt = axes_comp[1]
        sar_opt_slice = sar_optimized[:, :, mid_z]
        im_opt = ax_opt.imshow(
            sar_opt_slice,
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=max(np.max(sar_opt_slice), 1e-12),
        )
        ax_opt.set_title(f"Optimized SAR (z={mid_z})\nJ = {J_opt:.4f}")
        ax_opt.set_xlabel("Y (cells)")
        ax_opt.set_ylabel("X (cells)")
        plt.colorbar(im_opt, ax=ax_opt, label="SAR (W/kg)")

        ax_diff = axes_comp[2]
        sar_diff_slice = sar_opt_slice - sar_bl_slice
        vabs = max(np.max(np.abs(sar_diff_slice)), 1e-12)
        im_diff = ax_diff.imshow(
            sar_diff_slice, origin="lower", cmap="RdBu_r", vmin=-vabs, vmax=vabs
        )
        ax_diff.set_title(f"Difference (Opt - Baseline)\nz={mid_z}")
        ax_diff.set_xlabel("Y (cells)")
        ax_diff.set_ylabel("X (cells)")
        plt.colorbar(im_diff, ax=ax_diff, label="ΔSAR (W/kg)")

        fig_comp.suptitle(
            f"Antenna Optimization: Baseline vs Optimized SAR (f0={args.f0/1e6:.0f} MHz)",
            fontsize=13,
        )
        plt.tight_layout()
        comp_path = os.path.join(
            IMAGES_DIR, f"{OUTPUT_BASE}_antenna_opt_comparison.png"
        )
        fig_comp.savefig(comp_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison figure saved to {comp_path}")
        plt.close(fig_comp)

        # --- Optimization trace plot ---
        if len(opt_trace) > 1:
            fig_trace, ax_trace = plt.subplots(figsize=(10, 5))
            trace_evals = [t["eval"] for t in opt_trace]
            trace_js = [t.get("J", t.get("J_eff", 0)) for t in opt_trace]
            ax_trace.plot(trace_evals, trace_js, "o-", markersize=2, linewidth=0.8)
            ax_trace.axhline(
                y=J_baseline,
                color="red",
                linestyle="--",
                label=f"Baseline J={J_baseline:.4f}",
            )
            ax_trace.axhline(
                y=1.0,
                color="green",
                linestyle=":",
                alpha=0.5,
                label="J=1.0 (selectivity threshold)",
            )
            ax_trace.set_xlabel("Evaluation number")
            ax_trace.set_ylabel("J (SAR_tumor / SAR_healthy)")
            ax_trace.set_title("Antenna Optimization Convergence Trace")
            ax_trace.legend()
            ax_trace.grid(True, alpha=0.3)
            plt.tight_layout()
            trace_fig_path = os.path.join(
                IMAGES_DIR, f"{OUTPUT_BASE}_optimization_trace.png"
            )
            fig_trace.savefig(trace_fig_path, dpi=150, bbox_inches="tight")
            print(f"  Optimization trace plot saved to {trace_fig_path}")
            plt.close(fig_trace)

        # Store for Option A: main FDTD will run 4-quadrant time-domain with these parameters
        opt_quad_sources = quad_sources
        opt_alphas = best_alphas.copy()
        opt_thetas = best_thetas.copy()
        opt_f0 = args.f0

        _write_progress(
            "antenna_optimization",
            "Antenna optimization complete",
            25,
            ["setup", "segmentation", "antenna_optimization"],
        )
        print(
            "\nAntenna optimization complete. Running full time-domain FDTD with optimized 4-quadrant source...\n"
        )

    # 2D footprints for overlaying anatomy on E-field/SAR animations (max projection along Z)
    tumor_footprint_2d = np.max((labels_3d >= 1).astype(np.float32), axis=2)
    # Z-index of tumor centroid for 2D combined animation (efield_sar_temp_2d): show slice at tumor plane only
    _tumor_mask_anim = (labels_3d >= 1) & (labels_3d <= 3)
    if np.any(_tumor_mask_anim):
        _tz = np.argwhere(_tumor_mask_anim)[:, 2]
        tumor_centroid_z_2d = int(round(np.mean(_tz)))
        tumor_centroid_z_2d = max(0, min(tumor_centroid_z_2d, simulation_size_z - 1))
    else:
        tumor_centroid_z_2d = simulation_size_z // 2
    # Precompute tumor contour segments for 3D floor overlay (contour at 0.5 in array coords)
    _fig_c = plt.figure()
    _ax_c = _fig_c.add_subplot(111)
    _cs = _ax_c.contour(tumor_footprint_2d, levels=[0.5], origin="lower")
    tumor_contour_segments = _cs.allsegs[0] if len(_cs.allsegs) > 0 else []
    plt.close(_fig_c)

    # frequencies
    number_of_frequencies = 3
    freq = np.array((50e6, 200e6, 500e6))
    arg = 2 * np.pi * freq * dt
    real_in = np.zeros(number_of_frequencies)
    imag_in = np.zeros(number_of_frequencies)
    real_pt = np.zeros(
        (number_of_frequencies, simulation_size_x, simulation_size_y, simulation_size_z)
    )
    imag_pt = np.zeros(
        (number_of_frequencies, simulation_size_x, simulation_size_y, simulation_size_z)
    )
    amp = np.zeros((number_of_frequencies, simulation_size_y))

    # Pulse Parameters (standard run; --pulse-amplitude scales E-field and thus SAR/T)
    pulse_width = 8
    pulse_delay = 20
    pulse_amplitude = args.pulse_amplitude
    pulse_type = getattr(args, "pulse_type", "gaussian")
    prop_direction = getattr(args, "prop_direction", "+y")
    pulse_freq = getattr(args, "pulse_freq", 100e6)
    pulse_ramp_width = getattr(args, "pulse_ramp_width", 30.0)

    # Calculate the PML parameters (npml set above, scaled with grid size)
    (
        gi1,
        gi2,
        gi3,
        fi1,
        fi2,
        fi3,
        gj1,
        gj2,
        gj3,
        fj1,
        fj2,
        fj3,
        gk1,
        gk2,
        gk3,
        fk1,
        fk2,
        fk3,
    ) = calculate_pml_parameters(
        npml, simulation_size_x, simulation_size_y, simulation_size_z
    )
    boundary_low = [0, 0]
    boundary_high = [0, 0]
    time_steps = args.time_steps
    sar_start_step = int(time_steps * 0.7)  # Start accumulating after 70% of simulation

    # For optimized 4-quadrant CW run: use many more steps for steady-state propagation
    # At 100 MHz, ~600 steps/period; need 10+ periods for steady state, then average over several cycles
    if args.optimize_antenna and opt_quad_sources is not None:
        steps_per_period = max(1, round(1.0 / (opt_f0 * dt)))
        time_steps = 15 * steps_per_period  # 15 periods total
        sar_start_step = (
            10 * steps_per_period
        )  # Start after 10 periods (steady state reached)
        print(
            f"  CW run: {time_steps} steps (~{time_steps // steps_per_period} periods), "
            f"SAR accumulation from step {sar_start_step} ({time_steps - sar_start_step} samples)"
        )
    else:
        # Standard run: optional source position (for CW point source) and CW steady-state timing
        if getattr(args, "source_x", None) is not None:
            source_x = max(npml, min(simulation_size_x - npml - 1, args.source_x))
        if getattr(args, "source_y", None) is not None:
            source_y = max(npml, min(simulation_size_y - npml - 1, args.source_y))
        if getattr(args, "source_z", None) is not None:
            source_z = max(npml, min(simulation_size_z - npml - 1, args.source_z))
        if pulse_type == "cw" and getattr(args, "cw_periods", None) is not None:
            steps_per_period = max(1, round(1.0 / (pulse_freq * dt)))
            time_steps = args.cw_periods * steps_per_period
            sar_start_step = 10 * steps_per_period
            print(
                f"  Standard CW run: {time_steps} steps (~{args.cw_periods} periods), "
                f"SAR accumulation from step {sar_start_step}"
            )
        elif (
            pulse_type == "sinusoid_no_ramp"
            and getattr(args, "cw_periods", None) is not None
        ):
            # sinusoid_no_ramp with --cw-periods: use N periods like CW
            steps_per_period = max(1, round(1.0 / (pulse_freq * dt)))
            time_steps = args.cw_periods * steps_per_period
            sar_start_step = 10 * steps_per_period
            print(
                f"  sinusoid_no_ramp (--cw-periods): {time_steps} steps (~{args.cw_periods} periods), "
                f"SAR accumulation from step {sar_start_step}"
            )
        elif pulse_type == "sinusoid_no_ramp":
            # sinusoid_no_ramp by default: use --time-steps as-is; accumulate SAR for all timesteps
            sar_start_step = 0
        elif pulse_type in ("cw", "sinusoid"):
            # CW/sinusoid: SAR must be averaged over full periods after steady state
            steps_per_period = max(1, round(1.0 / (pulse_freq * dt)))
            min_periods_total = 15  # 10 to reach steady state + 5 full periods for SAR
            min_steps = min_periods_total * steps_per_period
            if time_steps < min_steps:
                time_steps = min_steps
                print(
                    f"  {pulse_type}: time_steps increased to {time_steps} (~{min_periods_total} periods) "
                    "for valid SAR (full-period average after steady state)"
                )
            sar_start_step = 10 * steps_per_period
            n_sar_steps = time_steps - sar_start_step
            print(
                f"  SAR accumulation from step {sar_start_step} ({n_sar_steps} samples, "
                f"~{n_sar_steps // steps_per_period} full periods)"
            )
        # Direction-specific incident buffers for plane-wave (gaussian/modulated_gaussian)
        if prop_direction in ("+x", "-x"):
            ez_inc_x = np.zeros(simulation_size_x)
            hy_inc_x = np.zeros(simulation_size_x)
            boundary_low_x = [0.0, 0.0]
            boundary_high_x = [0.0, 0.0]
        else:
            ez_inc_x = hy_inc_x = None
            boundary_low_x = boundary_high_x = None
        if prop_direction in ("+z", "-z"):
            ez_inc_z = np.zeros(simulation_size_z)
            hx_inc_z = np.zeros(simulation_size_z)
            boundary_low_z = [0.0, 0.0]
            boundary_high_z = [0.0, 0.0]
        else:
            ez_inc_z = hx_inc_z = None
            boundary_low_z = boundary_high_z = None

    # Arrays for SAR calculation (accumulate E-field squared)
    Ex_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ey_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ez_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    n_sar_samples = 0  # Count samples for averaging

    E_frames = []
    SAR_frames = []  # Store instantaneous SAR frames for animation
    Temperature_frames = []  # Store instantaneous temperature frames for animation
    frame_interval = 1  # Steps between saved frames; set to 1 in standard run, overwritten in optimized run
    streamed_n_frames = 0  # When --stream-frames: total frames written to disk (E_frames/SAR_frames stay empty)
    STREAM_CHUNK_SIZE = (
        20  # Frames per file when streaming (match E_FRAMES_CHUNK_SIZE used later)
    )

    # ----- Performance metrics (objective 5: computational performance and scalability) -----
    t_start_total = time.perf_counter()
    _phases_before_fdtd = ["setup", "segmentation"]
    if args.optimize_antenna and opt_quad_sources is not None:
        _phases_before_fdtd = ["setup", "segmentation", "antenna_optimization"]
    _write_progress(
        "fdtd_simulation",
        f"FDTD running (0 / {time_steps} steps)",
        25,
        _phases_before_fdtd,
        {
            "time_steps": time_steps,
            "grid_shape": [simulation_size_x, simulation_size_y, simulation_size_z],
        },
    )

    # Point source positions (used for cw/sinusoid/sinusoid_no_ramp; in scope for metadata)
    point_source_positions = [(source_x, source_y, source_z)]

    # Branch: Option A = full time-domain FDTD with optimized 4-quadrant source; else = standard single-source
    if args.optimize_antenna and opt_quad_sources is not None:
        # ----- Optimized 4-quadrant time-domain FDTD (Option A) -----
        # All main results (E_frames, SAR, temperature, animations) come from this run
        print(
            "\nRunning full time-domain FDTD with optimized 4-quadrant antenna (f0={:.1f} MHz)...".format(
                opt_f0 / 1e6
            )
        )
        ramp_width_opt = 30.0
        frame_interval = (
            args.stream_frame_interval
            if args.stream_frames
            else max(1, time_steps // 350)
        )  # When streaming, use --stream-frame-interval (e.g. 1 = every step)
        if args.stream_frames:
            E_buffer, SAR_buffer = [], []
            stream_part = 0
            print(
                f"  Streaming E/SAR frames to disk (interval={frame_interval}, chunk={STREAM_CHUNK_SIZE} frames)"
            )
        for time_step in range(1, time_steps + 1):
            # D-field updates (curl of H)
            Dx, iDx = calculate_dx_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dx,
                iDx,
                Hy,
                Hz,
                gj3,
                gk3,
                gj2,
                gk2,
                gi1,
            )
            Dy, iDy = calculate_dy_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dy,
                iDy,
                Hx,
                Hz,
                gi3,
                gk3,
                gi2,
                gk2,
                gj1,
            )
            Dz, iDz = calculate_dz_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dz,
                iDz,
                Hx,
                Hy,
                gi3,
                gj3,
                gi2,
                gj2,
                gk1,
            )
            # 4 soft sources at quadrant gaps: CW at opt_f0 with ramp and optimized amplitude/phase
            # args.opt_source_scale multiplies amplitude so SAR scales with scale² (for temperature rise)
            for q, qs in enumerate(opt_quad_sources):
                gi, gj, gk = qs["gap"]
                ramp = 1.0 - exp(-0.5 * (time_step / ramp_width_opt) ** 2)
                src_val = (
                    args.opt_source_scale
                    * opt_alphas[q]
                    * ramp
                    * sin(2.0 * np.pi * opt_f0 * time_step * dt + opt_thetas[q])
                )
                Dz[gi, gj, gk] += src_val

            # E-field from D-field
            Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dx,
                Dy,
                Dz,
                eps_x,
                eps_y,
                eps_z,
                conductivity_x,
                conductivity_y,
                conductivity_z,
                Ex,
                Ey,
                Ez,
                Ix,
                Iy,
                Iz,
            )

            # Accumulate E-field squared for SAR (after field stabilizes)
            if time_step >= sar_start_step:
                Ex_sq_sum, Ey_sq_sum, Ez_sq_sum = accumulate_e_field_squared(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Ex,
                    Ey,
                    Ez,
                    Ex_sq_sum,
                    Ey_sq_sum,
                    Ez_sq_sum,
                )
                n_sar_samples += 1

            # H-field updates (curl of E; no incident field in optimized path)
            Hx, iHx = calculate_hx_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Hx,
                iHx,
                Ey,
                Ez,
                fi1,
                fj2,
                fk2,
                fj3,
                fk3,
            )
            Hy, iHy = calculate_hy_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Hy,
                iHy,
                Ex,
                Ez,
                fj1,
                fi2,
                fk2,
                fi3,
                fk3,
            )
            Hz, iHz = calculate_hz_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Hz,
                iHz,
                Ex,
                Ey,
                fk1,
                fi2,
                fj2,
                fi3,
                fj3,
            )

            if time_step % 500 == 0:
                print("  opt FDTD step", time_step)
                _pct = 25 + 45 * time_step / time_steps
                _write_progress(
                    "fdtd_simulation",
                    f"FDTD step {time_step} / {time_steps}",
                    _pct,
                    _phases_before_fdtd,
                    {"time_step": time_step, "time_steps": time_steps},
                )

            if time_step % frame_interval == 0:
                sar_instant = compute_instantaneous_sar(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Ex,
                    Ey,
                    Ez,
                    sigma_x,
                    sigma_y,
                    sigma_z,
                    rho,
                )
                if args.stream_frames:
                    E_buffer.append(Ez.copy())
                    SAR_buffer.append(sar_instant.copy())
                    streamed_n_frames += 1
                    if len(E_buffer) >= STREAM_CHUNK_SIZE:
                        part_path_e = os.path.join(
                            E_FRAMES_DIR,
                            f"{OUTPUT_BASE}_E_frames_part{stream_part}.npz",
                        )
                        part_path_sar = os.path.join(
                            SAR_FRAMES_DIR,
                            f"{OUTPUT_BASE}_SAR_frames_part{stream_part}.npz",
                        )
                        np.savez_compressed(
                            part_path_e,
                            E_frames=np.array(E_buffer, dtype=np.float32),
                        )
                        np.savez_compressed(
                            part_path_sar,
                            SAR_frames=np.array(SAR_buffer, dtype=np.float32),
                        )
                        E_buffer.clear()
                        SAR_buffer.clear()
                        stream_part += 1
                else:
                    E_frames.append(Ez.copy())
                    SAR_frames.append(sar_instant.copy())

        if args.stream_frames and E_buffer:
            part_path_e = os.path.join(
                E_FRAMES_DIR, f"{OUTPUT_BASE}_E_frames_part{stream_part}.npz"
            )
            part_path_sar = os.path.join(
                SAR_FRAMES_DIR, f"{OUTPUT_BASE}_SAR_frames_part{stream_part}.npz"
            )
            np.savez_compressed(
                part_path_e, E_frames=np.array(E_buffer, dtype=np.float32)
            )
            np.savez_compressed(
                part_path_sar, SAR_frames=np.array(SAR_buffer, dtype=np.float32)
            )
            E_buffer.clear()
            SAR_buffer.clear()

        t_end_fdtd = time.perf_counter()
        _write_progress(
            "fdtd_simulation",
            "FDTD complete",
            70,
            _phases_before_fdtd + ["fdtd_simulation"],
        )
        print("  Optimized 4-quadrant FDTD complete.")

    else:
        # ----- Standard main FDTD loop (pulse_type: gaussian | cw | modulated_gaussian; prop_direction or point source) -----
        if args.stream_frames:
            E_buffer, SAR_buffer = [], []
            stream_part = 0
            print(
                f"  Streaming E/SAR frames to disk (interval={args.stream_frame_interval}, chunk={STREAM_CHUNK_SIZE} frames)"
            )
        # Injection indices for plane wave (low end for +axis, high end for -axis)
        inj_y = 3 if prop_direction == "+y" else simulation_size_y - 4
        inj_x = 3 if prop_direction == "+x" else simulation_size_x - 4
        inj_z = 3 if prop_direction == "+z" else simulation_size_z - 4

        # Point source positions (1–3) for cw/sinusoid/sinusoid_no_ramp; antenna-like defaults for 2nd/3rd
        point_source_positions = [(source_x, source_y, source_z)]
        if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
            ring_off = max(npml, getattr(args, "source_ring_offset", 10))
            cx, cy, cz = (
                simulation_size_x // 2,
                simulation_size_y // 2,
                simulation_size_z // 2,
            )
            if getattr(args, "use_source_2", False):
                sx2 = cx if args.source_x_2 is None else args.source_x_2
                sy2 = ring_off if args.source_y_2 is None else args.source_y_2
                sz2 = cz if args.source_z_2 is None else args.source_z_2
                sx2 = max(npml, min(simulation_size_x - npml - 1, sx2))
                sy2 = max(npml, min(simulation_size_y - npml - 1, sy2))
                sz2 = max(npml, min(simulation_size_z - npml - 1, sz2))
                point_source_positions.append((sx2, sy2, sz2))
            if getattr(args, "use_source_3", False):
                sx3 = (
                    (simulation_size_x - ring_off - 1)
                    if args.source_x_3 is None
                    else args.source_x_3
                )
                sy3 = cy if args.source_y_3 is None else args.source_y_3
                sz3 = cz if args.source_z_3 is None else args.source_z_3
                sx3 = max(npml, min(simulation_size_x - npml - 1, sx3))
                sy3 = max(npml, min(simulation_size_y - npml - 1, sy3))
                sz3 = max(npml, min(simulation_size_z - npml - 1, sz3))
                point_source_positions.append((sx3, sy3, sz3))
            if len(point_source_positions) > 1:
                print(
                    f"  Point sources: {len(point_source_positions)} positions {point_source_positions}"
                )

        for time_step in range(1, time_steps + 1):
            t_dt = time_step * dt
            # Compute pulse value from waveform type
            if pulse_type == "gaussian":
                pulse = pulse_amplitude * exp(
                    -0.5 * ((pulse_delay - time_step) / pulse_width) ** 2
                )
            elif pulse_type == "modulated_gaussian":
                pulse = (
                    pulse_amplitude
                    * exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
                    * sin(2.0 * np.pi * pulse_freq * t_dt)
                )
            elif pulse_type == "sinusoid":
                ramp = (
                    1.0
                    if time_step >= 2 * pulse_ramp_width
                    else (1.0 - exp(-0.5 * (time_step / pulse_ramp_width) ** 2))
                )
                pulse = ramp * pulse_amplitude * sin(2.0 * np.pi * pulse_freq * t_dt)
            elif pulse_type == "sinusoid_no_ramp":
                pulse = pulse_amplitude * sin(2.0 * np.pi * pulse_freq * t_dt)
            else:  # cw
                ramp = (
                    1.0
                    if time_step >= 2 * pulse_ramp_width
                    else (1.0 - exp(-0.5 * (time_step / pulse_ramp_width) ** 2))
                )
                pulse = ramp * pulse_amplitude * sin(2.0 * np.pi * pulse_freq * t_dt)

            if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
                # CW point source: no incident buffer; use pulse for incident Fourier (avoid amp_in=0)
                for m in range(number_of_frequencies):
                    real_in[m] = real_in[m] + cos(arg[m] * time_step) * pulse
                    imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * pulse
            else:
                # Plane wave: update 1D incident buffer and apply ABCs for chosen direction
                if prop_direction in ("+y", "-y"):
                    for j in range(1, simulation_size_y - 1):
                        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])
                    ez_inc[0] = boundary_low.pop(0)
                    boundary_low.append(ez_inc[1])
                    ez_inc[simulation_size_y - 1] = boundary_high.pop(0)
                    boundary_high.append(ez_inc[simulation_size_y - 2])
                    ez_inc[inj_y] = pulse
                elif prop_direction in ("+x", "-x"):
                    ez_inc_x = update_ez_inc_x(simulation_size_x, ez_inc_x, hy_inc_x)
                    ez_inc_x[0] = boundary_low_x.pop(0)
                    boundary_low_x.append(ez_inc_x[1])
                    ez_inc_x[simulation_size_x - 1] = boundary_high_x.pop(0)
                    boundary_high_x.append(ez_inc_x[simulation_size_x - 2])
                    ez_inc_x[inj_x] = pulse
                else:  # +z, -z
                    ez_inc_z = update_ez_inc_z(simulation_size_z, ez_inc_z, hx_inc_z)
                    ez_inc_z[0] = boundary_low_z.pop(0)
                    boundary_low_z.append(ez_inc_z[1])
                    ez_inc_z[simulation_size_z - 1] = boundary_high_z.pop(0)
                    boundary_high_z.append(ez_inc_z[simulation_size_z - 2])
                    ez_inc_z[inj_z] = pulse

                # Fourier transform of the incident field (plane wave only)
                if prop_direction in ("+y", "-y"):
                    for m in range(number_of_frequencies):
                        real_in[m] = (
                            real_in[m] + cos(arg[m] * time_step) * ez_inc[ja - 1]
                        )
                        imag_in[m] = (
                            imag_in[m] - sin(arg[m] * time_step) * ez_inc[ja - 1]
                        )
                elif prop_direction in ("+x", "-x"):
                    for m in range(number_of_frequencies):
                        real_in[m] = (
                            real_in[m] + cos(arg[m] * time_step) * ez_inc_x[ia - 1]
                        )
                        imag_in[m] = (
                            imag_in[m] - sin(arg[m] * time_step) * ez_inc_x[ia - 1]
                        )
                else:
                    for m in range(number_of_frequencies):
                        real_in[m] = (
                            real_in[m] + cos(arg[m] * time_step) * ez_inc_z[ka - 1]
                        )
                        imag_in[m] = (
                            imag_in[m] - sin(arg[m] * time_step) * ez_inc_z[ka - 1]
                        )

            # Calculate the D Fields
            Dx, iDx = calculate_dx_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dx,
                iDx,
                Hy,
                Hz,
                gj3,
                gk3,
                gj2,
                gk2,
                gi1,
            )
            Dy, iDy = calculate_dy_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dy,
                iDy,
                Hx,
                Hz,
                gi3,
                gk3,
                gi2,
                gk2,
                gj1,
            )
            Dz, iDz = calculate_dz_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dz,
                iDz,
                Hx,
                Hy,
                gi3,
                gj3,
                gi2,
                gj2,
                gk1,
            )

            # Add the source: plane wave via incident coupling; cw/sinusoid/sinusoid_no_ramp as soft point source(s)
            if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
                pass  # add to Ez after E from D
            elif prop_direction in ("+y", "-y"):
                Dy = calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy, hx_inc)
                Dz = calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz, hx_inc)
            elif prop_direction in ("+x", "-x"):
                Dz = calculate_inc_dz_field_x(ia, ib, ja, jb, ka, kb, Dz, hy_inc_x)
            else:  # +z, -z
                Dz = calculate_inc_dz_field_z(ia, ib, ja, jb, ka, kb, Dz, hx_inc_z)

            # Calculate the E field from the D field
            Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Dx,
                Dy,
                Dz,
                eps_x,
                eps_y,
                eps_z,
                conductivity_x,
                conductivity_y,
                conductivity_z,
                Ex,
                Ey,
                Ez,
                Ix,
                Iy,
                Iz,
            )

            # Point source(s): soft source at each position (cw, sinusoid, sinusoid_no_ramp)
            if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
                for sx, sy, sz in point_source_positions:
                    Ez[sx, sy, sz] = Ez[sx, sy, sz] + pulse

            # Accumulate E-field squared for SAR calculation (after field stabilizes)
            if time_step >= sar_start_step:
                Ex_sq_sum, Ey_sq_sum, Ez_sq_sum = accumulate_e_field_squared(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Ex,
                    Ey,
                    Ez,
                    Ex_sq_sum,
                    Ey_sq_sum,
                    Ez_sq_sum,
                )
                n_sar_samples += 1

            # Calculate the Fourier transform of Ex
            real_pt, imag_pt = calculate_fourier_transform_ex(
                simulation_size_x,
                simulation_size_y,
                number_of_frequencies,
                real_pt,
                imag_pt,
                Ez,
                arg,
                time_step,
                source_z,
            )

            # Calculate the H fields (with incident coupling for plane wave; full update for point sources)
            if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
                Hx, iHx = calculate_hx_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hx,
                    iHx,
                    Ey,
                    Ez,
                    fi1,
                    fj2,
                    fk2,
                    fj3,
                    fk3,
                )
                Hy, iHy = calculate_hy_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hy,
                    iHy,
                    Ex,
                    Ez,
                    fj1,
                    fi2,
                    fk2,
                    fi3,
                    fk3,
                )
            elif prop_direction in ("+y", "-y"):
                hx_inc = calculate_hx_inc(simulation_size_y, hx_inc, ez_inc)
                Hx, iHx = calculate_hx_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hx,
                    iHx,
                    Ey,
                    Ez,
                    fi1,
                    fj2,
                    fk2,
                    fj3,
                    fk3,
                )
                Hx = calculate_hx_with_incident_field(
                    ia, ib, ja, jb, ka, kb, Hx, ez_inc
                )
                Hy, iHy = calculate_hy_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hy,
                    iHy,
                    Ex,
                    Ez,
                    fj1,
                    fi2,
                    fk2,
                    fi3,
                    fk3,
                )
                Hy = calculate_hy_with_incident_field(
                    ia, ib, ja, jb, ka, kb, Hy, ez_inc
                )
            elif prop_direction in ("+x", "-x"):
                hy_inc_x = calculate_hy_inc_x(simulation_size_x, hy_inc_x, ez_inc_x)
                Hx, iHx = calculate_hx_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hx,
                    iHx,
                    Ey,
                    Ez,
                    fi1,
                    fj2,
                    fk2,
                    fj3,
                    fk3,
                )
                Hy, iHy = calculate_hy_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hy,
                    iHy,
                    Ex,
                    Ez,
                    fj1,
                    fi2,
                    fk2,
                    fi3,
                    fk3,
                )
                Hy = calculate_hy_with_incident_field_x(
                    ia, ib, ja, jb, ka, kb, Hy, ez_inc_x
                )
            else:  # +z, -z
                hx_inc_z = calculate_hx_inc_z(simulation_size_z, hx_inc_z, ez_inc_z)
                Hx, iHx = calculate_hx_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hx,
                    iHx,
                    Ey,
                    Ez,
                    fi1,
                    fj2,
                    fk2,
                    fj3,
                    fk3,
                )
                Hx = calculate_hx_with_incident_field_z(
                    ia, ib, ja, jb, ka, kb, Hx, ez_inc_z
                )
                Hy, iHy = calculate_hy_field(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Hy,
                    iHy,
                    Ex,
                    Ez,
                    fj1,
                    fi2,
                    fk2,
                    fi3,
                    fk3,
                )
            Hz, iHz = calculate_hz_field(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                Hz,
                iHz,
                Ex,
                Ey,
                fk1,
                fi2,
                fj2,
                fi3,
                fj3,
            )
            if time_step % 5 == 0:
                print(time_step)
                print(np.min(Ez), np.max(Ez))
            if time_step % 500 == 0:
                _pct = 25 + 45 * time_step / time_steps
                _write_progress(
                    "fdtd_simulation",
                    f"FDTD step {time_step} / {time_steps}",
                    _pct,
                    _phases_before_fdtd,
                    {"time_step": time_step, "time_steps": time_steps},
                )

            # Store E-field and SAR frame (every step when not streaming; else per --stream-frame-interval)
            if args.stream_frames:
                if time_step % args.stream_frame_interval == 0:
                    sar_instant = compute_instantaneous_sar(
                        simulation_size_x,
                        simulation_size_y,
                        simulation_size_z,
                        Ex,
                        Ey,
                        Ez,
                        sigma_x,
                        sigma_y,
                        sigma_z,
                        rho,
                    )
                    E_buffer.append(Ez.copy())
                    SAR_buffer.append(sar_instant.copy())
                    streamed_n_frames += 1
                    if len(E_buffer) >= STREAM_CHUNK_SIZE:
                        part_path_e = os.path.join(
                            E_FRAMES_DIR,
                            f"{OUTPUT_BASE}_E_frames_part{stream_part}.npz",
                        )
                        part_path_sar = os.path.join(
                            SAR_FRAMES_DIR,
                            f"{OUTPUT_BASE}_SAR_frames_part{stream_part}.npz",
                        )
                        np.savez_compressed(
                            part_path_e,
                            E_frames=np.array(E_buffer, dtype=np.float32),
                        )
                        np.savez_compressed(
                            part_path_sar,
                            SAR_frames=np.array(SAR_buffer, dtype=np.float32),
                        )
                        E_buffer.clear()
                        SAR_buffer.clear()
                        stream_part += 1
            else:
                E_frames.append(Ez.copy())
                sar_instant = compute_instantaneous_sar(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    Ex,
                    Ey,
                    Ez,
                    sigma_x,
                    sigma_y,
                    sigma_z,
                    rho,
                )
                SAR_frames.append(sar_instant.copy())

        if args.stream_frames and E_buffer:
            part_path_e = os.path.join(
                E_FRAMES_DIR, f"{OUTPUT_BASE}_E_frames_part{stream_part}.npz"
            )
            part_path_sar = os.path.join(
                SAR_FRAMES_DIR, f"{OUTPUT_BASE}_SAR_frames_part{stream_part}.npz"
            )
            np.savez_compressed(
                part_path_e, E_frames=np.array(E_buffer, dtype=np.float32)
            )
            np.savez_compressed(
                part_path_sar, SAR_frames=np.array(SAR_buffer, dtype=np.float32)
            )
            E_buffer.clear()
            SAR_buffer.clear()

        t_end_fdtd = time.perf_counter()
        _write_progress(
            "fdtd_simulation",
            "FDTD complete",
            70,
            _phases_before_fdtd + ["fdtd_simulation"],
        )

    # Fourier amplitude (standard run only; skip when using optimized 4-quadrant run)
    if not args.optimize_antenna:
        # Calculate the Fourier amplitude of the incident pulse
        amp_in = np.sqrt(real_in**2 + imag_in**2)
        # Calculate the Fourier amplitude of the total field
        for m in range(number_of_frequencies):
            for j in range(ja, jb + 1):
                if eps_z[source_x, j, source_z] < 1:
                    amp[m, j] = (
                        1
                        / (amp_in[m])
                        * sqrt(
                            real_pt[m, source_x, j, source_z] ** 2
                            + imag_pt[m, source_x, j, source_z] ** 2
                        )
                    )

    # Compute SAR (Specific Absorption Rate)
    _write_progress(
        "sar_computation",
        "Computing SAR...",
        72,
        _phases_before_fdtd + ["fdtd_simulation"],
    )
    print("\nComputing SAR distribution...")
    print(f"SAR samples collected: {n_sar_samples}")
    t_start_sar = time.perf_counter()
    if n_sar_samples > 0:
        SAR = compute_sar(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            Ex_sq_sum,
            Ey_sq_sum,
            Ez_sq_sum,
            sigma_x,
            sigma_y,
            sigma_z,
            rho,
            n_sar_samples,
        )
        sar_max = np.max(SAR)
        # Region masks from segmentation (labels 1–3 = tumor, 4 = normal brain)
        tumor_region = (labels_3d >= 1) & (labels_3d <= 3)
        non_tumor_region = labels_3d == 4
        print(f"SAR statistics:")
        print(f"  Max SAR: {sar_max:.6g} W/kg")
        if np.any(rho > 0):
            print(f"  Mean SAR (tissue only): {np.mean(SAR[rho > 0]):.6g} W/kg")
        # Region masks from segmentation (labels 1–3 = tumor, 4 = normal brain)
        tumor_region = (labels_3d >= 1) & (labels_3d <= 3)
        non_tumor_region = labels_3d == 4
        if np.any(tumor_region):
            sar_t = SAR[tumor_region]
            print(
                f"  SAR in tumor: min={np.min(sar_t):.6g}, max={np.max(sar_t):.6g}, mean={np.mean(sar_t):.6g} W/kg"
            )
        if np.any(non_tumor_region):
            sar_nt = SAR[non_tumor_region]
            print(
                f"  SAR in non-tumor tissue: min={np.min(sar_nt):.6g}, max={np.max(sar_nt):.6g}, mean={np.mean(sar_nt):.6g} W/kg"
            )
    else:
        print("Warning: No SAR samples collected. SAR calculation skipped.")
        SAR = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
        tumor_region = (labels_3d >= 1) & (labels_3d <= 3)
        non_tumor_region = labels_3d == 4
    t_end_sar = time.perf_counter()
    _write_progress(
        "sar_computation",
        "SAR complete",
        75,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation"],
    )

    # ----- Thermal modeling (simplified Pennes, steady-state, no perfusion) -----
    # ∇·(k∇T) + SAR·ρ = 0; Q = SAR·ρ (W/m³). One-way coupling from SAR.
    _write_progress(
        "thermal_solver",
        "Thermal modeling...",
        77,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation"],
    )
    print("\nThermal modeling (steady-state bioheat, no perfusion)...")
    t_start_thermal = time.perf_counter()
    Q_heat = SAR * rho
    Q_heat[rho <= 0] = 0.0
    T_temp = solve_steady_bioheat_3d(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        k_3d,
        Q_heat,
        dx,
        T_boundary=T_BOUNDARY_CELSIUS,
        max_iter=50000,
        tol=1e-6,
    )
    if np.any(k_3d > 0):
        T_tissue = T_temp[k_3d > 0]
        print(
            f"  Temperature in tissue: min={np.min(T_tissue):.4f} °C, max={np.max(T_tissue):.4f} °C"
        )
    if np.any(tumor_region):
        T_t = T_temp[tumor_region]
        print(
            f"  Temperature in tumor: min={np.min(T_t):.4f} °C, max={np.max(T_t):.4f} °C"
        )
    if np.any(non_tumor_region):
        T_nt = T_temp[non_tumor_region]
        print(
            f"  Temperature in non-tumor tissue: min={np.min(T_nt):.4f} °C, max={np.max(T_nt):.4f} °C"
        )
    # Region stats for metadata (SAR and T min/max/mean by tumor vs non-tumor)
    region_stats = {}
    if np.any(tumor_region):
        region_stats["sar_tumor_W_per_kg"] = {
            "min": float(np.min(SAR[tumor_region])),
            "max": float(np.max(SAR[tumor_region])),
            "mean": float(np.mean(SAR[tumor_region])),
        }
        region_stats["temperature_tumor_C"] = {
            "min": float(np.min(T_temp[tumor_region])),
            "max": float(np.max(T_temp[tumor_region])),
        }
    if np.any(non_tumor_region):
        region_stats["sar_non_tumor_tissue_W_per_kg"] = {
            "min": float(np.min(SAR[non_tumor_region])),
            "max": float(np.max(SAR[non_tumor_region])),
            "mean": float(np.mean(SAR[non_tumor_region])),
        }
        region_stats["temperature_non_tumor_tissue_C"] = {
            "min": float(np.min(T_temp[non_tumor_region])),
            "max": float(np.max(T_temp[non_tumor_region])),
        }
    t_end_thermal = time.perf_counter()
    _write_progress(
        "thermal_solver",
        "Thermal complete",
        80,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation", "thermal_solver"],
    )

    # Compute temperature frames from SAR frames for animation (simplified proportional model)
    # T_instant = T_boundary + (SAR_instant / SAR_max) * (T_max - T_boundary)
    # When streaming: read SAR from disk in chunks, compute T, write to disk (no full RAM).
    print("\nComputing temperature frames for animation...")
    T_max = np.max(T_temp) if np.any(k_3d > 0) else T_BOUNDARY_CELSIUS
    SAR_max_final = np.max(SAR) if np.max(SAR) > 0 else 1.0
    tissue_mask = rho > 0

    if streamed_n_frames > 0:
        # Streamed run: SAR is on disk; compute Temperature_frames per chunk and write to disk
        n_stream_parts = (
            streamed_n_frames + STREAM_CHUNK_SIZE - 1
        ) // STREAM_CHUNK_SIZE
        for part in range(n_stream_parts):
            start = part * STREAM_CHUNK_SIZE
            end = min(start + STREAM_CHUNK_SIZE, streamed_n_frames)
            part_path_sar = os.path.join(
                SAR_FRAMES_DIR, f"{OUTPUT_BASE}_SAR_frames_part{part}.npz"
            )
            with np.load(part_path_sar) as z:
                sar_chunk = z["SAR_frames"]
            temp_chunk = np.full_like(sar_chunk, T_BOUNDARY_CELSIUS, dtype=np.float32)
            for i in range(sar_chunk.shape[0]):
                if np.any(tissue_mask) and SAR_max_final > 0:
                    sar_normalized = np.clip(sar_chunk[i] / SAR_max_final, 0, 1).astype(
                        np.float32
                    )
                    temp_chunk[i][tissue_mask] = T_BOUNDARY_CELSIUS + sar_normalized[
                        tissue_mask
                    ] * (T_max - T_BOUNDARY_CELSIUS)
            part_path_t = os.path.join(
                TEMPERATURE_FRAMES_DIR,
                f"{OUTPUT_BASE}_Temperature_frames_part{part}.npz",
            )
            np.savez_compressed(part_path_t, Temperature_frames=temp_chunk)
        print(
            f"  Computed and saved {streamed_n_frames} temperature frames (streamed, {n_stream_parts} parts)"
        )
    elif len(SAR_frames) > 0:
        for sar_frame in SAR_frames:
            temp_frame = np.full_like(sar_frame, T_BOUNDARY_CELSIUS, dtype=np.float32)
            if np.any(tissue_mask) and SAR_max_final > 0:
                sar_normalized = np.clip(sar_frame / SAR_max_final, 0, 1)
                temp_frame[tissue_mask] = T_BOUNDARY_CELSIUS + sar_normalized[
                    tissue_mask
                ] * (T_max - T_BOUNDARY_CELSIUS)
            Temperature_frames.append(temp_frame)
        print(f"  Computed {len(Temperature_frames)} temperature frames")
    else:
        print(
            "  Warning: No SAR frames available, skipping temperature frame computation"
        )

    # Scalar time series (per-frame max/mean SAR and temperature) for dashboard plots
    time_series_data = None
    if len(SAR_frames) > 0 and len(Temperature_frames) == len(SAR_frames):
        n_frames_ts = len(SAR_frames)
        time_step_indices = [(i + 1) * frame_interval for i in range(n_frames_ts)]
        tissue_mask = rho > 0
        max_sar_list = []
        mean_sar_list = []
        max_temp_list = []
        mean_temp_list = []
        for i in range(n_frames_ts):
            sf = SAR_frames[i]
            tf = Temperature_frames[i]
            max_sar_list.append(float(np.max(sf)))
            mean_sar_list.append(
                float(np.mean(sf[tissue_mask])) if np.any(tissue_mask) else 0.0
            )
            max_temp_list.append(float(np.max(tf)))
            mean_temp_list.append(
                float(np.mean(tf[tissue_mask])) if np.any(tissue_mask) else 0.0
            )
        time_series_data = {
            "time_step": time_step_indices,
            "max_sar_W_per_kg": max_sar_list,
            "mean_sar_W_per_kg": mean_sar_list,
            "max_temperature_C": max_temp_list,
            "mean_temperature_C": mean_temp_list,
        }
        print(
            f"  Scalar time series: {n_frames_ts} frames (frame_interval={frame_interval})"
        )

    # Aggregate performance metrics (for thesis objective 5 and metadata)
    total_wall_time_s = time.perf_counter() - t_start_total
    time_fdtd_s = t_end_fdtd - t_start_total
    time_sar_s = t_end_sar - t_start_sar
    time_thermal_s = t_end_thermal - t_start_thermal
    setup_s = t_start_total - t_end_segmentation
    segmentation_s = t_end_segmentation - t_start_pipeline
    number_of_voxels = simulation_size_x * simulation_size_y * simulation_size_z
    peak_memory_MB = _get_peak_memory_mb()

    # Phased timing: segmentation -> setup -> [antenna_opt] -> FDTD -> SAR -> thermal -> saving_and_animations (set at end)
    performance_metrics = {
        "total_simulation_time_s": None,  # set at end of script (segmentation through animations)
        "phases_s": {
            "segmentation": round(segmentation_s, 4),
            "setup": round(setup_s, 4),
            "fdtd_simulation": round(time_fdtd_s, 4),
            "sar_computation": round(time_sar_s, 4),
            "thermal_solver": round(time_thermal_s, 4),
            "saving_and_animations": None,  # set at end of script
        },
        "time_steps": time_steps,
        "grid_shape": [simulation_size_x, simulation_size_y, simulation_size_z],
        "number_of_voxels": number_of_voxels,
        "time_per_step_ms": (
            round(1000.0 * time_fdtd_s / time_steps, 4) if time_steps else None
        ),
        "peak_memory_MB": (
            round(peak_memory_MB, 2) if peak_memory_MB is not None else None
        ),
        "backend": "numpy_numba",
        "dt_s": round(float(dt), 12),
        "dt_courant_s": round(float(dt_courant), 12),
        "courant_factor": float(args.courant_factor),
    }
    if antenna_optimization_s is not None:
        performance_metrics["phases_s"]["antenna_optimization"] = antenna_optimization_s
    # Backward compatibility
    performance_metrics["total_wall_time_s"] = round(total_wall_time_s, 4)
    performance_metrics["time_fdtd_s"] = round(time_fdtd_s, 4)
    performance_metrics["time_sar_s"] = round(time_sar_s, 4)
    performance_metrics["time_thermal_s"] = round(time_thermal_s, 4)

    print("\nPerformance metrics (computational efficiency):")
    print(f"  Segmentation: {segmentation_s:.2f} s")
    print(f"  Setup: {setup_s:.2f} s")
    if antenna_optimization_s is not None:
        print(f"  Antenna optimization: {antenna_optimization_s:.2f} s")
    print(f"  FDTD loop: {time_fdtd_s:.2f} s")
    print(f"  SAR computation: {time_sar_s:.2f} s")
    print(f"  Thermal solver: {time_thermal_s:.2f} s")
    print(f"  Total so far: {total_wall_time_s:.2f} s")
    if peak_memory_MB is not None:
        print(f"  Peak memory: {peak_memory_MB:.1f} MB")
    print(f"  Time per FDTD step: {performance_metrics['time_per_step_ms']} ms")

    # ----- Save SAR, temperature, metadata, then E-field (partial save safe; animations unchanged) -----
    _write_progress(
        "saving_and_animations",
        "Saving NIfTI, metadata, frames...",
        85,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation", "thermal_solver"],
    )
    t_start_save = time.perf_counter()
    print("\nSaving simulation data (NIfTI, NumPy, JSON)...")
    affine = np.diag([float(dx), float(dx), float(dx), 1.0])  # voxel size in meters

    # 1) SAR and temperature first (small; always succeed)
    sar_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_SAR.nii.gz")
    nib.save(
        nib.Nifti1Image(SAR.astype(np.float32), affine),
        sar_path,
    )
    print(f"  SAR saved to {sar_path}")

    temperature_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_temperature.nii.gz")
    nib.save(
        nib.Nifti1Image(T_temp.astype(np.float32), affine),
        temperature_path,
    )
    print(f"  Temperature saved to {temperature_path}")

    segmentation_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_segmentation.nii.gz")
    nib.save(
        nib.Nifti1Image(labels_3d.astype(np.int32), affine),
        segmentation_path,
    )
    print(f"  Segmentation (labels) saved to {segmentation_path}")

    # 2) Metadata next so partial data (SAR + T + metadata) is complete even if E_frames save fails
    metadata_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_metadata.json")
    # When --stream-frames was used, E_frames/SAR_frames/Temperature_frames are empty; use streamed count
    n_frames = streamed_n_frames if streamed_n_frames > 0 else len(E_frames)
    n_sar_frames = streamed_n_frames if streamed_n_frames > 0 else len(SAR_frames)
    n_temp_frames = (
        streamed_n_frames if streamed_n_frames > 0 else len(Temperature_frames)
    )
    E_FRAMES_CHUNK_SIZE = 20  # small chunks to avoid OOM when building each part (match STREAM_CHUNK_SIZE)
    n_parts = (
        (n_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        if n_frames > 0
        else 0
    )
    # Tissue/dielectric and thermal table for metadata (serializable for dashboard)
    LABEL_NAMES = {
        0: "background (air)",
        1: "necrotic tumor",
        2: "edema",
        3: "enhancing tumor",
        4: "normal brain",
    }
    tissue_properties = []
    for lab in sorted(TISSUE_TABLE.keys()):
        eps_r, sigma_val, rho_val = TISSUE_TABLE[lab]
        k_val = K_TISSUE.get(lab, 0.0)
        tissue_properties.append(
            {
                "label": lab,
                "name": LABEL_NAMES.get(lab, f"label_{lab}"),
                "eps_r": float(eps_r),
                "sigma_S_per_m": float(sigma_val),
                "rho_kg_per_m3": float(rho_val),
                "k_W_per_mK": float(k_val),
            }
        )

    SAR_frames_n_parts = (
        (n_sar_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        if n_sar_frames > 0
        else 0
    )
    Temperature_frames_n_parts = (
        (n_temp_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        if n_temp_frames > 0
        else 0
    )

    metadata = {
        "output_base": OUTPUT_BASE,
        "grid_shape": [simulation_size_x, simulation_size_y, simulation_size_z],
        "voxel_size_m": float(dx),
        "time_step_s": float(dt),
        "dt_courant_s": float(dt_courant),
        "courant_factor": float(args.courant_factor),
        "time_steps": time_steps,
        "frame_interval": frame_interval,
        "stream_frames": args.stream_frames,
        "stream_frame_interval": getattr(args, "stream_frame_interval", 1),
        "pulse_amplitude": float(pulse_amplitude),
        "n_frames": n_frames,
        "E_frames_chunk_size": E_FRAMES_CHUNK_SIZE,
        "E_frames_n_parts": n_parts,
        "SAR_frames_n_parts": SAR_frames_n_parts,
        "Temperature_frames_n_parts": Temperature_frames_n_parts,
        "time_series_file": (
            f"{OUTPUT_BASE}_time_series.json" if time_series_data else None
        ),
        "frequencies_Hz": freq.tolist(),
        "top_10_slice_indices": top_10_slice_indices,
        "T_boundary_C": T_BOUNDARY_CELSIUS,
        "tissue_properties": tissue_properties,
        "performance": performance_metrics,
        "region_stats": region_stats,
    }
    if args.optimize_antenna and opt_f0 is not None:
        metadata["antenna_optimized"] = True
        metadata["optimized_f0_Hz"] = float(opt_f0)
        metadata["optimized_alphas"] = opt_alphas.tolist()
        metadata["optimized_thetas_rad"] = opt_thetas.tolist()
        metadata["optimized_quadrant_gaps"] = [qs["gap"] for qs in opt_quad_sources]
        metadata["opt_source_scale"] = args.opt_source_scale
    else:
        metadata["pulse_type"] = pulse_type
        metadata["prop_direction"] = prop_direction
        metadata["source_x"] = int(source_x)
        metadata["source_y"] = int(source_y)
        metadata["source_z"] = int(source_z)
        if len(point_source_positions) > 1:
            metadata["point_source_positions"] = [
                [int(sx), int(sy), int(sz)] for (sx, sy, sz) in point_source_positions
            ]
        metadata["pulse_freq_Hz"] = float(pulse_freq)
        if getattr(args, "cw_periods", None) is not None:
            metadata["cw_periods"] = args.cw_periods
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")

    # Save performance metrics separately for scalability comparisons (e.g. multiple grid sizes)
    performance_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_performance.json")
    with open(performance_path, "w") as f:
        json.dump(performance_metrics, f, indent=2)
    print(f"  Performance metrics saved to {performance_path}")

    # 2b) Scalar time series (time step vs max/mean SAR and temperature)
    if time_series_data is not None:
        time_series_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_time_series.json")
        with open(time_series_path, "w") as f:
            json.dump(time_series_data, f, indent=2)
        print(f"  Time series saved to {time_series_path}")

    # 3) E_frames: from memory only when not streamed (when streamed, already written during FDTD)
    if n_frames > 0 and not args.stream_frames:
        n_parts_actual = (n_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        for part in range(n_parts_actual):
            start = part * E_FRAMES_CHUNK_SIZE
            end = min(start + E_FRAMES_CHUNK_SIZE, n_frames)
            chunk = np.array(E_frames[start:end], dtype=np.float32)
            part_path = os.path.join(
                DATA_DIR, "E_frames", f"{OUTPUT_BASE}_E_frames_part{part}.npz"
            )
            np.savez_compressed(part_path, E_frames=chunk)
            del chunk
        print(
            f"  E-field time series saved to {DATA_DIR}/E_frames/ ({n_parts_actual} parts, "
            f"shape ({n_frames}, {simulation_size_x}, {simulation_size_y}, {simulation_size_z}))"
        )
    elif n_frames > 0 and args.stream_frames:
        print(f"  E-field time series already streamed to disk ({n_frames} frames)")
    else:
        print("  E_frames empty, skipping E-field save")

    # 4) SAR_frames: from memory only when not streamed (when streamed, already written during FDTD)
    if n_sar_frames > 0 and not args.stream_frames:
        for part in range(SAR_frames_n_parts):
            start = part * E_FRAMES_CHUNK_SIZE
            end = min(start + E_FRAMES_CHUNK_SIZE, n_sar_frames)
            chunk = np.array(SAR_frames[start:end], dtype=np.float32)
            part_path = os.path.join(
                SAR_FRAMES_DIR, f"{OUTPUT_BASE}_SAR_frames_part{part}.npz"
            )
            np.savez_compressed(part_path, SAR_frames=chunk)
            del chunk
        print(
            f"  SAR time series saved to {SAR_FRAMES_DIR}/ ({SAR_frames_n_parts} parts)"
        )
    elif n_sar_frames > 0 and args.stream_frames:
        print(f"  SAR time series already streamed to disk ({n_sar_frames} frames)")
    else:
        print("  SAR_frames empty, skipping SAR frames save")

    # 5) Temperature_frames: from memory only when not streamed (when streamed, already written above)
    if n_temp_frames > 0 and not args.stream_frames:
        for part in range(Temperature_frames_n_parts):
            start = part * E_FRAMES_CHUNK_SIZE
            end = min(start + E_FRAMES_CHUNK_SIZE, n_temp_frames)
            chunk = np.array(Temperature_frames[start:end], dtype=np.float32)
            part_path = os.path.join(
                TEMPERATURE_FRAMES_DIR,
                f"{OUTPUT_BASE}_Temperature_frames_part{part}.npz",
            )
            np.savez_compressed(part_path, Temperature_frames=chunk)
            del chunk
        print(
            f"  Temperature time series saved to {TEMPERATURE_FRAMES_DIR}/ "
            f"({Temperature_frames_n_parts} parts)"
        )
    elif n_temp_frames > 0 and args.stream_frames:
        print(
            f"  Temperature time series already streamed to disk ({n_temp_frames} frames)"
        )
    else:
        print("  Temperature_frames empty, skipping temperature frames save")

    print("✓ Simulation data save complete")
    t_end_save_data = time.perf_counter()
    saving_data_s = t_end_save_data - t_start_save

    _complete_phases = [
        "setup",
        "segmentation",
        "fdtd_simulation",
        "sar_computation",
        "thermal_solver",
        "saving_and_animations",
        "complete",
    ]
    if args.optimize_antenna and opt_quad_sources is not None:
        _complete_phases.insert(2, "antenna_optimization")
    _write_progress("complete", "Simulation complete", 100, _complete_phases)

    # ----- Visualize total SAR pattern distribution -----
    print("\nSaving total SAR pattern distribution...")
    cx, cy, cz = (
        simulation_size_x // 2,
        simulation_size_y // 2,
        simulation_size_z // 2,
    )
    sar_slice_z = np.max(SAR, axis=2)
    sar_max_val = np.max(SAR) if np.max(SAR) > 0 else 1.0

    fig_sar = plt.figure(figsize=(14, 10))
    ax1 = fig_sar.add_subplot(2, 2, 1)
    im1 = ax1.imshow(
        SAR[:, :, cz], origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax1.set_title(f"SAR axial (z={cz})")
    ax1.set_xlabel("Y (cells)")
    ax1.set_ylabel("X (cells)")
    plt.colorbar(im1, ax=ax1, label="SAR (W/kg)")
    ax1.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )

    ax2 = fig_sar.add_subplot(2, 2, 2)
    im2 = ax2.imshow(
        SAR[cx, :, :], origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax2.set_title(f"SAR sagittal (x={cx})")
    ax2.set_xlabel("Z (cells)")
    ax2.set_ylabel("Y (cells)")
    plt.colorbar(im2, ax=ax2, label="SAR (W/kg)")

    ax3 = fig_sar.add_subplot(2, 2, 3)
    im3 = ax3.imshow(
        SAR[:, cy, :], origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax3.set_title(f"SAR coronal (y={cy})")
    ax3.set_xlabel("Z (cells)")
    ax3.set_ylabel("X (cells)")
    plt.colorbar(im3, ax=ax3, label="SAR (W/kg)")

    ax4 = fig_sar.add_subplot(2, 2, 4)
    im4 = ax4.imshow(
        sar_slice_z, origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax4.set_title("SAR max projection (along Z)")
    ax4.set_xlabel("Y (cells)")
    ax4.set_ylabel("X (cells)")
    plt.colorbar(im4, ax=ax4, label="SAR (W/kg)")
    ax4.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )

    fig_sar.suptitle("Total SAR pattern distribution (post-FDTD)", fontsize=12)
    plt.tight_layout()
    fig_sar.savefig(
        os.path.join(IMAGES_DIR, f"{OUTPUT_BASE}_SAR_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_sar)
    print(f"  Saved {OUTPUT_BASE}_SAR_distribution.png to {IMAGES_DIR}/")

    # ----- Visualize temperature distribution -----
    print("Saving temperature distribution...")
    T_max_val = np.max(T_temp)
    T_min_val = np.min(T_temp)
    if T_max_val <= T_min_val:
        T_max_val = T_min_val + 0.1

    fig_T = plt.figure(figsize=(14, 10))
    ax1 = fig_T.add_subplot(2, 2, 1)
    im1 = ax1.imshow(
        T_temp[:, :, cz],
        origin="lower",
        cmap="coolwarm",
        vmin=T_min_val,
        vmax=T_max_val,
    )
    ax1.set_title(f"Temperature axial (z={cz})")
    ax1.set_xlabel("Y (cells)")
    ax1.set_ylabel("X (cells)")
    plt.colorbar(im1, ax=ax1, label="T (°C)")
    ax1.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )

    ax2 = fig_T.add_subplot(2, 2, 2)
    im2 = ax2.imshow(
        T_temp[cx, :, :],
        origin="lower",
        cmap="coolwarm",
        vmin=T_min_val,
        vmax=T_max_val,
    )
    ax2.set_title(f"Temperature sagittal (x={cx})")
    ax2.set_xlabel("Z (cells)")
    ax2.set_ylabel("Y (cells)")
    plt.colorbar(im2, ax=ax2, label="T (°C)")

    ax3 = fig_T.add_subplot(2, 2, 3)
    im3 = ax3.imshow(
        T_temp[:, cy, :],
        origin="lower",
        cmap="coolwarm",
        vmin=T_min_val,
        vmax=T_max_val,
    )
    ax3.set_title(f"Temperature coronal (y={cy})")
    ax3.set_xlabel("Z (cells)")
    ax3.set_ylabel("X (cells)")
    plt.colorbar(im3, ax=ax3, label="T (°C)")

    ax4 = fig_T.add_subplot(2, 2, 4)
    T_proj = np.max(T_temp, axis=2)
    im4 = ax4.imshow(
        T_proj, origin="lower", cmap="coolwarm", vmin=T_min_val, vmax=T_max_val
    )
    ax4.set_title("Temperature max projection (along Z)")
    ax4.set_xlabel("Y (cells)")
    ax4.set_ylabel("X (cells)")
    plt.colorbar(im4, ax=ax4, label="T (°C)")
    ax4.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )

    fig_T.suptitle(
        "Temperature distribution (Pennes steady-state, no perfusion)", fontsize=12
    )
    plt.tight_layout()
    fig_T.savefig(
        os.path.join(IMAGES_DIR, f"{OUTPUT_BASE}_temperature_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig_T)
    print(f"  Saved {OUTPUT_BASE}_temperature_distribution.png to {IMAGES_DIR}/")

    # ----- Per-slice: side-by-side Anatomy (FLAIR+seg or seg) | SAR | Temperature (top 10 tumor slices) -----
    print(
        "\nSaving side-by-side comparison (Anatomy | SAR | Temperature) for top 10 tumor slices..."
    )
    from matplotlib.colors import ListedColormap

    if volume_4d_ds is not None:
        from brain_tumor_segmentation_model import SEG_COLORS_STREAMLIT

    for k in top_10_slice_indices:
        if not (0 <= k < simulation_size_z):
            continue
        fig_sk = plt.figure(figsize=(14, 5))
        ax_anat = fig_sk.add_subplot(1, 3, 1)
        ax_sar = fig_sk.add_subplot(1, 3, 2)
        ax_temp = fig_sk.add_subplot(1, 3, 3)

        # Panel 1: Anatomy — FLAIR + segmentation overlay (if modalities) or segmentation only
        if volume_4d_ds is not None:
            flair_slice = volume_4d_ds[0, :, :, k]
            seg_slice = labels_3d[:, :, k].astype(np.int32)
            seg_vmax = 4 if np.max(labels_3d) >= 4 else 3
            seg_cmap_local = ListedColormap(SEG_COLORS_STREAMLIT[: seg_vmax + 1])
            ax_anat.imshow(flair_slice, cmap="gray", origin="lower", alpha=0.7)
            mask_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
            ax_anat.imshow(
                mask_overlay,
                cmap=seg_cmap_local,
                origin="lower",
                vmin=0,
                vmax=seg_vmax,
                alpha=0.8,
                interpolation="nearest",
            )
            ax_anat.set_title(f"FLAIR + segmentation (z={k})")
        else:
            rgb_slice = labels_to_rgb(labels_3d[:, :, k])
            ax_anat.imshow(rgb_slice, origin="lower")
            ax_anat.set_title(f"Segmentation (z={k})")
        ax_anat.set_xlabel("Y (cells)")
        ax_anat.set_ylabel("X (cells)")
        ax_anat.axis("on")

        # Panel 2: SAR
        sar_max_k = np.max(SAR) if np.max(SAR) > 0 else 1.0
        im_sar = ax_sar.imshow(
            SAR[:, :, k], origin="lower", cmap="gray", vmin=0, vmax=sar_max_k
        )
        ax_sar.set_title(f"SAR axial (z={k})")
        ax_sar.set_xlabel("Y (cells)")
        ax_sar.set_ylabel("X (cells)")
        plt.colorbar(im_sar, ax=ax_sar, label="SAR (W/kg)")
        ax_sar.contour(
            tumor_footprint_2d,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1,
            origin="lower",
        )

        # Panel 3: Temperature
        T_max_k = np.max(T_temp)
        T_min_k = np.min(T_temp)
        if T_max_k <= T_min_k:
            T_max_k = T_min_k + 0.1
        im_temp = ax_temp.imshow(
            T_temp[:, :, k],
            origin="lower",
            cmap="coolwarm",
            vmin=T_min_k,
            vmax=T_max_k,
        )
        ax_temp.set_title(f"Temperature axial (z={k})")
        ax_temp.set_xlabel("Y (cells)")
        ax_temp.set_ylabel("X (cells)")
        plt.colorbar(im_temp, ax=ax_temp, label="T (°C)")
        ax_temp.contour(
            tumor_footprint_2d,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1,
            origin="lower",
        )

        fig_sk.suptitle(
            f"Slice {k} (top 10 by tumor area): Anatomy | SAR | Temperature",
            fontsize=12,
        )
        plt.tight_layout()
        fig_sk.savefig(
            os.path.join(
                IMAGES_DIR, f"{OUTPUT_BASE}_slice_{k}_anatomy_SAR_temperature.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_sk)
    print(
        f"  Saved {len(top_10_slice_indices)} side-by-side anatomy/SAR/temperature figures to {IMAGES_DIR}/"
    )

    # ----- 3x15 tumor previews: FLAIR+segmentation, SAR, Temperature (top 15 slices) -----
    if use_modalities and volume_4d_ds is not None:
        try:
            from brain_tumor_segmentation_model import create_3x15_tumor_previews
        except ImportError:
            create_3x15_tumor_previews = None

        if create_3x15_tumor_previews is not None:
            print(
                "\nSaving 3x15 tumor previews (FLAIR+segmentation, SAR, Temperature) for top slices..."
            )
            # Use the steady-state SAR and temperature fields as 3D volumes
            sar_3d = SAR
            temperature_3d = T_temp
            try:
                create_3x15_tumor_previews(
                    volume_4d_ds,
                    labels_3d,
                    sar_3d,
                    temperature_3d,
                    output_dir=IMAGES_DIR,
                    case_name=OUTPUT_BASE,
                    n_slices=15,
                )
                print(
                    f"  Saved 3x15 grid previews (FLAIR+seg, SAR, Temperature) to {IMAGES_DIR}/"
                )
            except Exception as e:
                print(f"  Warning: failed to create 3x15 tumor previews: {e}")

    # Function to prepare 3D voxel data for a frame
    def prepare_3d_voxel_data(data, step=3, threshold_ratio=0.3):
        """Prepare 3D voxel data for visualization"""
        # Normalize data
        data_norm = np.abs(data)
        data_max = np.max(data_norm)
        if data_max == 0:
            return None, None, None, None, None

        threshold = data_max * threshold_ratio

        # Downsample data
        data_sub = data_norm[::step, ::step, ::step]
        nx, ny, nz = data_sub.shape

        # Create coordinate arrays - voxels expects coordinates to be one element larger
        x_edges = np.arange(nx + 1) * step
        y_edges = np.arange(ny + 1) * step
        z_edges = np.arange(nz + 1) * step
        X_sub, Y_sub, Z_sub = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

        # Create voxel plot for values above threshold
        voxels = data_sub > threshold

        # Create color array based on data values
        colors = np.empty(voxels.shape + (4,))
        data_normalized = np.clip(data_sub / data_max, 0, 1)

        return X_sub, Y_sub, Z_sub, voxels, colors, data_normalized, data_max

    # Function to create 3D isometric view (static)
    def create_3d_isometric_view(data, title, cmap="jet", threshold_ratio=0.3):
        """Create a 3D isometric view using isosurfaces"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        result = prepare_3d_voxel_data(data, step=3, threshold_ratio=threshold_ratio)
        if result[0] is None:
            print(f"Warning: {title} has zero data, skipping 3D view")
            return fig, ax

        X_sub, Y_sub, Z_sub, voxels, colors, data_normalized, data_max = result

        if not np.any(voxels):
            print(f"Warning: No voxels above threshold for {title}")
            return fig, ax

        # Use colormap-like coloring (red to yellow for hot, blue to red for jet)
        if cmap == "hot":
            colors[..., 0] = 1.0  # Red
            colors[..., 1] = data_normalized  # Green (yellow when max)
            colors[..., 2] = 0.0  # Blue
        else:  # jet-like
            colors[..., 0] = np.clip(1.0 - 2 * data_normalized, 0, 1)  # Red
            colors[..., 1] = np.clip(2 * data_normalized, 0, 1)  # Green
            colors[..., 2] = np.clip(2 * (data_normalized - 0.5), 0, 1)  # Blue
        # Alpha based on magnitude (only for voxels that are True)
        colors[..., 3] = np.where(voxels, np.clip(data_normalized, 0.2, 0.8), 0.0)

        ax.voxels(
            X_sub, Y_sub, Z_sub, voxels, facecolors=colors, edgecolor="none", alpha=0.6
        )

        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")
        ax.set_zlabel("Z (cells)")
        ax.set_title(title)

        # Set equal aspect ratio
        max_range = np.array([data.shape[0], data.shape[1], data.shape[2]]).max() / 2.0
        mid_x = data.shape[0] / 2.0
        mid_y = data.shape[1] / 2.0
        mid_z = data.shape[2] / 2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig, ax

    # Animated 3D isometric views side-by-side using ArtistAnimation
    t_start_anim = time.perf_counter()
    t_end_anim = t_start_anim  # set at end of whichever branch runs
    if args.skip_animations:
        print("\nSkipping animations (--skip-animations).")
    else:
        print(
            "\nCreating animated 3D isometric views (E-field, SAR, and Temperature)..."
        )
        if len(E_frames) > 0 and len(SAR_frames) > 0:
            # Re-implement using ArtistAnimation (similar to reference file)
            # Create 2D projections for each frame (works better with ArtistAnimation)
            # Include temperature if available
            has_temperature = len(Temperature_frames) > 0
            n_cols = 3 if has_temperature else 2
            fig_anim = plt.figure(figsize=(20 if has_temperature else 20, 10))
            ax_e = fig_anim.add_subplot(1, n_cols, 1)
            ax_sar = fig_anim.add_subplot(1, n_cols, 2)
            if has_temperature:
                ax_temp = fig_anim.add_subplot(1, n_cols, 3)

            # Find global min/max for consistent color scaling
            e_max = max(np.max(np.abs(frame)) for frame in E_frames)
            sar_max = (
                max(np.max(frame) for frame in SAR_frames)
                if len(SAR_frames) > 0
                else 1.0
            )
            temp_min = T_BOUNDARY_CELSIUS
            temp_max = T_BOUNDARY_CELSIUS + 1.0
            if has_temperature:
                temp_min = (
                    min(np.min(frame) for frame in Temperature_frames)
                    if Temperature_frames
                    else T_BOUNDARY_CELSIUS
                )
                temp_max = (
                    max(np.max(frame) for frame in Temperature_frames)
                    if Temperature_frames
                    else T_BOUNDARY_CELSIUS + 1.0
                )

            # 2D combined animation: max-projection over z for E-field, SAR, and temperature
            # Prepare frames for ArtistAnimation
            frames = []

            for frame_idx in range(len(E_frames)):
                # Get current frame data
                e_data = np.abs(E_frames[frame_idx])
                sar_data = SAR_frames[frame_idx]
                if has_temperature:
                    temp_data = Temperature_frames[frame_idx]

                # Max-projection over z (2D overview)
                e_projection = np.max(e_data, axis=2)
                sar_projection = np.max(sar_data, axis=2)
                if has_temperature:
                    temp_projection = np.max(temp_data, axis=2)

                # Plot E-field (max proj. over z)
                im_e = ax_e.imshow(
                    e_projection,
                    cmap="jet",
                    origin="lower",
                    vmin=0,
                    vmax=e_max,
                    animated=True,
                )
                ax_e.set_title(
                    f"E-field (Ez) max proj. (z) - Frame {frame_idx + 1}/{len(E_frames)}"
                )
                ax_e.set_xlabel("Y (cells)")
                ax_e.set_ylabel("X (cells)")
                # Overlay tumor outline so anatomy is visible in simulation output
                c_e = ax_e.contour(
                    tumor_footprint_2d,
                    levels=[0.5],
                    colors=["lime"],
                    linewidths=1.5,
                    origin="lower",
                )

                # Plot SAR projection
                im_sar = ax_sar.imshow(
                    sar_projection,
                    cmap="coolwarm",
                    origin="lower",
                    vmin=0,
                    vmax=sar_max,
                    animated=True,
                )
                ax_sar.set_title(
                    f"SAR max proj. (z) - Frame {frame_idx + 1}/{len(SAR_frames)}"
                )
                ax_sar.set_xlabel("Y (cells)")
                ax_sar.set_ylabel("X (cells)")
                # Overlay tumor outline
                c_sar = ax_sar.contour(
                    tumor_footprint_2d,
                    levels=[0.5],
                    colors=["cyan"],
                    linewidths=1.5,
                    origin="lower",
                )

                # Plot Temperature projection if available
                if has_temperature:
                    im_temp = ax_temp.imshow(
                        temp_projection,
                        cmap="coolwarm",
                        origin="lower",
                        vmin=temp_min,
                        vmax=temp_max,
                        animated=True,
                    )
                    ax_temp.set_title(
                        f"Temperature max proj. (z) - Frame {frame_idx + 1}/{len(Temperature_frames)}"
                    )
                    ax_temp.set_xlabel("Y (cells)")
                    ax_temp.set_ylabel("X (cells)")
                    # Overlay tumor outline
                    c_temp = ax_temp.contour(
                        tumor_footprint_2d,
                        levels=[0.5],
                        colors=["cyan"],
                        linewidths=1.5,
                        origin="lower",
                    )

                # Build frame artists: contour sets may not have .collections in all matplotlib versions
                def _contour_artists(c):
                    return getattr(c, "collections", [c])

                frame_artists = (
                    [im_e, im_sar]
                    + list(_contour_artists(c_e))
                    + list(_contour_artists(c_sar))
                )
                if has_temperature:
                    frame_artists.append(im_temp)
                    frame_artists.extend(list(_contour_artists(c_temp)))
                frames.append(frame_artists)

            fig_anim.tight_layout()

            # Create animation using ArtistAnimation (blit=False when contours are included for compatibility)
            ani = animation.ArtistAnimation(
                fig_anim, frames, interval=20, blit=False, repeat_delay=1000
            )

            # Save animation as video (similar to reference file)
            print("\nSaving 2D animation as video...")
            animation_name = (
                f"{OUTPUT_BASE}_efield_sar_temp_2d.mp4"
                if has_temperature
                else f"{OUTPUT_BASE}_sar_2d.mp4"
            )
            ani.save(
                os.path.join(ANIMATIONS_DIR, animation_name),
                writer="ffmpeg",
                fps=60,
            )
            print(f"✓ 2D animation saved ({animation_name})")

            # Now create 3D isometric view animations using FuncAnimation
            print("\nCreating 3D isometric view animations...")

            # 3D E-field animation
            fig_3d_e = plt.figure(figsize=(12, 10))
            ax_3d_e = fig_3d_e.add_subplot(111, projection="3d")

            # Prepare coordinate arrays for 3D surface plots
            sample_data = E_frames[0]
            step_3d = 2  # Downsample for performance
            nx, ny = sample_data.shape[0], sample_data.shape[1]
            x_coords = np.arange(0, nx, step_3d)
            y_coords = np.arange(0, ny, step_3d)
            X_3d, Y_3d = np.meshgrid(y_coords, x_coords)

            # Set fixed axis limits
            x_min, x_max = 0, ny
            y_min, y_max = 0, nx
            z_min, z_max = 0, e_max

            def update_3d_e(frame_num):
                ax_3d_e.clear()
                e_data = np.abs(E_frames[frame_num])
                e_projection = np.max(e_data, axis=2)
                e_projection_3d = e_projection[::step_3d, ::step_3d]

                surf_e = ax_3d_e.plot_surface(
                    X_3d,
                    Y_3d,
                    e_projection_3d,
                    cmap="jet",
                    vmin=0,
                    vmax=e_max,
                    alpha=0.9,
                    linewidth=0,
                    antialiased=True,
                )
                # Tumor outline on floor (z=0) so anatomy is visible
                for seg in tumor_contour_segments:
                    if len(seg) > 1:
                        ax_3d_e.plot(
                            seg[:, 0],
                            seg[:, 1],
                            np.zeros(seg.shape[0]),
                            "lime",
                            linewidth=1.5,
                        )
                ax_3d_e.set_xlabel("Y (cells)")
                ax_3d_e.set_ylabel("X (cells)")
                ax_3d_e.set_zlabel("Magnitude")
                ax_3d_e.set_title(
                    f"E-field (Ez) 3D Isometric - Frame {frame_num + 1}/{len(E_frames)}"
                )
                ax_3d_e.view_init(elev=30, azim=45)  # Isometric view angle
                # Set fixed axis limits
                ax_3d_e.set_xlim(x_min, x_max)
                ax_3d_e.set_ylim(y_min, y_max)
                ax_3d_e.set_zlim(z_min, z_max)

            ani_3d_e = animation.FuncAnimation(
                fig_3d_e,
                update_3d_e,
                frames=len(E_frames),
                interval=20,
                blit=False,
                repeat=True,
                repeat_delay=1000,
            )

            print("Saving 3D E-field animation as video...")
            ani_3d_e.save(
                os.path.join(ANIMATIONS_DIR, f"{OUTPUT_BASE}_efield_3d.mp4"),
                writer="ffmpeg",
                fps=60,
            )
            print("✓ 3D E-field animation saved")

            # 3D SAR animation
            fig_3d_sar = plt.figure(figsize=(12, 10))
            ax_3d_sar = fig_3d_sar.add_subplot(111, projection="3d")

            # Set fixed axis limits for SAR
            sar_z_min, sar_z_max = 0, sar_max

            def update_3d_sar(frame_num):
                ax_3d_sar.clear()
                sar_data = SAR_frames[frame_num]
                sar_projection = np.max(sar_data, axis=2)
                sar_projection_3d = sar_projection[::step_3d, ::step_3d]

                surf_sar = ax_3d_sar.plot_surface(
                    X_3d,
                    Y_3d,
                    sar_projection_3d,
                    cmap="coolwarm",
                    vmin=0,
                    vmax=sar_max,
                    alpha=0.9,
                    linewidth=0,
                    antialiased=True,
                )
                # Tumor outline on floor (z=0) so anatomy is visible
                for seg in tumor_contour_segments:
                    if len(seg) > 1:
                        ax_3d_sar.plot(
                            seg[:, 0],
                            seg[:, 1],
                            np.zeros(seg.shape[0]),
                            "cyan",
                            linewidth=1.5,
                        )
                ax_3d_sar.set_xlabel("Y (cells)")
                ax_3d_sar.set_ylabel("X (cells)")
                ax_3d_sar.set_zlabel("SAR (W/kg)")
                ax_3d_sar.set_title(
                    f"SAR Distribution 3D Isometric - Frame {frame_num + 1}/{len(SAR_frames)}"
                )
                ax_3d_sar.view_init(elev=30, azim=45)  # Isometric view angle
                # Set fixed axis limits
                ax_3d_sar.set_xlim(x_min, x_max)
                ax_3d_sar.set_ylim(y_min, y_max)
                ax_3d_sar.set_zlim(sar_z_min, sar_z_max)

            ani_3d_sar = animation.FuncAnimation(
                fig_3d_sar,
                update_3d_sar,
                frames=len(SAR_frames),
                interval=20,
                blit=False,
                repeat=True,
                repeat_delay=1000,
            )

            print("Saving 3D SAR animation as video...")
            ani_3d_sar.save(
                os.path.join(ANIMATIONS_DIR, f"{OUTPUT_BASE}_sar_3d.mp4"),
                writer="ffmpeg",
                fps=60,
            )
            print("✓ 3D SAR animation saved")

            # 3D Temperature animation (if temperature frames available)
            if len(Temperature_frames) > 0:
                fig_3d_temp = plt.figure(figsize=(12, 10))
                ax_3d_temp = fig_3d_temp.add_subplot(111, projection="3d")

                # Set fixed axis limits for Temperature
                temp_z_min, temp_z_max = temp_min, temp_max

                def update_3d_temp(frame_num):
                    ax_3d_temp.clear()
                    temp_data = Temperature_frames[frame_num]
                    temp_projection = np.max(temp_data, axis=2)
                    temp_projection_3d = temp_projection[::step_3d, ::step_3d]

                    surf_temp = ax_3d_temp.plot_surface(
                        X_3d,
                        Y_3d,
                        temp_projection_3d,
                        cmap="coolwarm",
                        vmin=temp_min,
                        vmax=temp_max,
                        alpha=0.9,
                        linewidth=0,
                        antialiased=True,
                    )
                    # Tumor outline on floor (z=0) so anatomy is visible
                    for seg in tumor_contour_segments:
                        if len(seg) > 1:
                            ax_3d_temp.plot(
                                seg[:, 0],
                                seg[:, 1],
                                np.zeros(seg.shape[0]),
                                "cyan",
                                linewidth=1.5,
                            )
                    ax_3d_temp.set_xlabel("Y (cells)")
                    ax_3d_temp.set_ylabel("X (cells)")
                    ax_3d_temp.set_zlabel("Temperature (°C)")
                    ax_3d_temp.set_title(
                        f"Temperature Distribution 3D Isometric - Frame {frame_num + 1}/{len(Temperature_frames)}"
                    )
                    ax_3d_temp.view_init(elev=30, azim=45)  # Isometric view angle
                    # Set fixed axis limits
                    ax_3d_temp.set_xlim(x_min, x_max)
                    ax_3d_temp.set_ylim(y_min, y_max)
                    ax_3d_temp.set_zlim(temp_z_min, temp_z_max)

                ani_3d_temp = animation.FuncAnimation(
                    fig_3d_temp,
                    update_3d_temp,
                    frames=len(Temperature_frames),
                    interval=20,
                    blit=False,
                    repeat=True,
                    repeat_delay=1000,
                )

                print("Saving 3D Temperature animation as video...")
                ani_3d_temp.save(
                    os.path.join(ANIMATIONS_DIR, f"{OUTPUT_BASE}_temperature_3d.mp4"),
                    writer="ffmpeg",
                    fps=60,
                )
                print("✓ 3D Temperature animation saved")
            t_end_anim = time.perf_counter()

        elif len(E_frames) > 0:
            # Only E-field available
            print("Warning: SAR frames not available, showing only E-field animation")
            # Re-implement using ArtistAnimation (similar to reference file)
            fig_anim = plt.figure(figsize=(12, 10))
            ax_e = fig_anim.add_subplot(1, 1, 1)

            # Find global min/max for consistent color scaling
            e_max = max(np.max(np.abs(frame)) for frame in E_frames)

            # Prepare frames for ArtistAnimation
            frames = []

            for frame_idx in range(len(E_frames)):
                e_data = np.abs(E_frames[frame_idx])

                # Create maximum intensity projection (isometric-like view)
                e_projection = np.max(e_data, axis=2)

                im_e = ax_e.imshow(
                    e_projection,
                    cmap="jet",
                    origin="lower",
                    vmin=0,
                    vmax=e_max,
                    animated=True,
                )
                ax_e.set_title(f"E-field (Ez) - Frame {frame_idx + 1}/{len(E_frames)}")
                ax_e.set_xlabel("Y (cells)")
                ax_e.set_ylabel("X (cells)")

                frames.append([im_e])

            # Create animation using ArtistAnimation (similar to reference file)
            ani = animation.ArtistAnimation(
                fig_anim, frames, interval=20, blit=True, repeat_delay=1000
            )

            # Save animation as video (similar to reference file)
            print("\nSaving 2D animation as video...")
            ani.save(
                os.path.join(ANIMATIONS_DIR, f"{OUTPUT_BASE}_efield_2d.mp4"),
                writer="ffmpeg",
                fps=60,
            )
            print("✓ 2D animation saved")

            # Now create 3D isometric view animation using FuncAnimation
            print("\nCreating 3D isometric view animation...")

            fig_3d_e = plt.figure(figsize=(12, 10))
            ax_3d_e = fig_3d_e.add_subplot(111, projection="3d")

            # Prepare coordinate arrays for 3D surface plots
            sample_data = E_frames[0]
            step_3d = 2  # Downsample for performance
            nx, ny = sample_data.shape[0], sample_data.shape[1]
            x_coords = np.arange(0, nx, step_3d)
            y_coords = np.arange(0, ny, step_3d)
            X_3d, Y_3d = np.meshgrid(y_coords, x_coords)

            # Set fixed axis limits
            x_min, x_max = 0, ny
            y_min, y_max = 0, nx
            z_min, z_max = 0, e_max

            def update_3d_e(frame_num):
                ax_3d_e.clear()
                e_data = np.abs(E_frames[frame_num])
                e_projection = np.max(e_data, axis=2)
                e_projection_3d = e_projection[::step_3d, ::step_3d]

                surf_e = ax_3d_e.plot_surface(
                    X_3d,
                    Y_3d,
                    e_projection_3d,
                    cmap="jet",
                    vmin=0,
                    vmax=e_max,
                    alpha=0.9,
                    linewidth=0,
                    antialiased=True,
                )
                # Tumor outline on floor (z=0) so anatomy is visible
                for seg in tumor_contour_segments:
                    if len(seg) > 1:
                        ax_3d_e.plot(
                            seg[:, 0],
                            seg[:, 1],
                            np.zeros(seg.shape[0]),
                            "lime",
                            linewidth=1.5,
                        )
                ax_3d_e.set_xlabel("Y (cells)")
                ax_3d_e.set_ylabel("X (cells)")
                ax_3d_e.set_zlabel("Magnitude")
                ax_3d_e.set_title(
                    f"E-field (Ez) 3D Isometric - Frame {frame_num + 1}/{len(E_frames)}"
                )
                ax_3d_e.view_init(elev=30, azim=45)  # Isometric view angle
                # Set fixed axis limits
                ax_3d_e.set_xlim(x_min, x_max)
                ax_3d_e.set_ylim(y_min, y_max)
                ax_3d_e.set_zlim(z_min, z_max)

            ani_3d_e = animation.FuncAnimation(
                fig_3d_e,
                update_3d_e,
                frames=len(E_frames),
                interval=20,
                blit=False,
                repeat=True,
                repeat_delay=1000,
            )

            print("Saving 3D E-field animation as video...")
            ani_3d_e.save(
                os.path.join(ANIMATIONS_DIR, f"{OUTPUT_BASE}_efield_3d.mp4"),
                writer="ffmpeg",
                fps=60,
            )
            print("✓ 3D E-field animation saved")
            t_end_anim = time.perf_counter()

        elif args.stream_frames and streamed_n_frames > 0:
            print(
                f"  Frames were streamed to disk ({streamed_n_frames} frames). "
                "Building animations from streamed frames (this may take a while)..."
            )
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _anim_script = os.path.join(
                _script_dir, "build_animations_from_streamed_frames.py"
            )
            if os.path.isfile(_anim_script):
                anim_argv = [
                    sys.executable,
                    _anim_script,
                    os.path.abspath(RESULTS_DIR),
                    os.path.abspath(ANIMATIONS_DIR),
                ]
                if args.slice_timestep_images:
                    anim_argv.append("--generate-slice-timestep-images")
                proc = subprocess.run(
                    anim_argv,
                    cwd=_script_dir,
                    stdin=subprocess.DEVNULL,
                )
                if proc.returncode == 0:
                    print(
                        f"  build_animations_from_streamed_frames.py finished (saved to {ANIMATIONS_DIR})."
                    )
                else:
                    print(
                        f"  Warning: build_animations_from_streamed_frames.py exited with code {proc.returncode}."
                    )
            else:
                print(
                    f"  Warning: {_anim_script} not found; run build_animations_from_streamed_frames.py manually."
                )
        t_end_anim = time.perf_counter()

    # Final pipeline timing (segmentation → end of animations) and update performance metadata
    animations_s = t_end_anim - t_start_anim
    performance_metrics["phases_s"]["saving_data"] = round(saving_data_s, 4)
    performance_metrics["phases_s"]["animations"] = round(animations_s, 4)
    # Remove lumped phase so dashboard shows saving_data and animations separately (no double-count)
    performance_metrics["phases_s"].pop("saving_and_animations", None)
    t_end_pipeline = time.perf_counter()
    performance_metrics["total_simulation_time_s"] = round(
        t_end_pipeline - t_start_pipeline, 4
    )
    with open(metadata_path, "r") as f:
        _meta = json.load(f)
    _meta["performance"] = performance_metrics
    with open(metadata_path, "w") as f:
        json.dump(_meta, f, indent=2)
    with open(performance_path, "w") as f:
        json.dump(performance_metrics, f, indent=2)
    print(
        f"\nTotal simulation time (segmentation → animations): {performance_metrics['total_simulation_time_s']:.2f} s"
    )
    print(f"  Saving data: {saving_data_s:.2f} s  |  Animations: {animations_s:.2f} s")
