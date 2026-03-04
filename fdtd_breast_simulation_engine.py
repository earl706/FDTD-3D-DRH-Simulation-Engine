"""
3D FDTD breast simulation engine using voxelized segmentation from BC_MRI_SEG.

Runs directly from image data (e.g. data/ISPY1/images_std/) by calling BC_MRI_SEG/breast_tumor_segmentation_model.py
to produce segmentations, then FDTD, then build_breast_animations_from_streamed_frames.py for animations.

Usage:
  # Single image .npy (runs segmentation → FDTD → animations):
  python fdtd_breast_simulation_engine.py -i BC_MRI_SEG/data/ISPY1/images_std/ispy1_12.npy [--max-dim 64] [--time-steps 500]
  # From image directory:
  python fdtd_breast_simulation_engine.py --data-dir data/ISPY1 [--max-cases 1]
  # From precomputed segmentation:
  python fdtd_breast_simulation_engine.py --seg path/to/segmentation.npy [--max-dim 60] [--time-steps 150]

Output: results/breast_fdtd_{timestamp}/ with data/ (E_frames/, SAR_frames/, metadata, segmentation.npy),
        images/, animations/ (unless --no-animations). --data-dir uses subdir seg_data/ for segmentation output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from math import exp

import numba
import numpy as np
from scipy import ndimage

# Optional: matplotlib for static images and in-script animations
try:
    from matplotlib import pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Project root and BC_MRI_SEG for segmentation
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BC_MRI_SEG_DIR = os.path.join(_SCRIPT_DIR, "BC_MRI_SEG")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _BC_MRI_SEG_DIR not in sys.path:
    sys.path.insert(0, _BC_MRI_SEG_DIR)

# Breast segmentation labels (must match breast_tumor_segmentation_model.py)
LABEL_BACKGROUND = 0
LABEL_TUMOR = 1
LABEL_HEALTHY_BREAST = 2

STREAM_CHUNK_SIZE = 20
DEFAULT_MAX_DIM = 60
DEFAULT_TIME_STEPS = 150


# ----- FDTD Yee solver (same as fdtd_brain_simulation_engine): PML, D/E/H updates, incident field -----
def _calculate_pml_parameters(npml, nx, ny, nz):
    gi1 = np.zeros(nx)
    gi2 = np.ones(nx)
    gi3 = np.ones(nx)
    fi1 = np.zeros(nx)
    fi2 = np.ones(nx)
    fi3 = np.ones(nx)
    gj1 = np.zeros(ny)
    gj2 = np.ones(ny)
    gj3 = np.ones(ny)
    fj1 = np.zeros(ny)
    fj2 = np.ones(ny)
    fj3 = np.ones(ny)
    gk1 = np.zeros(nz)
    gk2 = np.ones(nz)
    gk3 = np.ones(nz)
    fk1 = np.zeros(nz)
    fk2 = np.ones(nz)
    fk3 = np.ones(nz)
    for n in range(npml):
        xxn = (npml - n) / npml
        xn = 0.33 * (xxn**3)
        fi1[n] = xn
        fi1[nx - n - 1] = xn
        gi2[n] = 1 / (1 + xn)
        gi2[nx - 1 - n] = 1 / (1 + xn)
        gi3[n] = (1 - xn) / (1 + xn)
        gi3[nx - 1 - n] = (1 - xn) / (1 + xn)
        fj1[n] = xn
        fj1[ny - n - 1] = xn
        gj2[n] = 1 / (1 + xn)
        gj2[ny - 1 - n] = 1 / (1 + xn)
        gj3[n] = (1 - xn) / (1 + xn)
        gj3[ny - 1 - n] = (1 - xn) / (1 + xn)
        fk1[n] = xn
        fk1[nz - n - 1] = xn
        gk2[n] = 1 / (1 + xn)
        gk2[nz - 1 - n] = 1 / (1 + xn)
        gk3[n] = (1 - xn) / (1 + xn)
        gk3[nz - 1 - n] = (1 - xn) / (1 + xn)
        xxn = (npml - n - 0.5) / npml
        xn = 0.33 * (xxn**3)
        gi1[n] = xn
        gi1[nx - 1 - n] = xn
        fi2[n] = 1 / (1 + xn)
        fi2[nx - 1 - n] = 1 / (1 + xn)
        fi3[n] = (1 - xn) / (1 + xn)
        fi3[nx - 1 - n] = (1 - xn) / (1 + xn)
        gj1[n] = xn
        gj1[ny - 1 - n] = xn
        fj2[n] = 1 / (1 + xn)
        fj2[ny - 1 - n] = 1 / (1 + xn)
        fj3[n] = (1 - xn) / (1 + xn)
        fj3[ny - 1 - n] = (1 - xn) / (1 + xn)
        gk1[n] = xn
        gk1[nz - 1 - n] = xn
        fk2[n] = 1 / (1 + xn)
        fk2[nz - 1 - n] = 1 / (1 + xn)
        fk3[n] = (1 - xn) / (1 + xn)
        fk3[nz - 1 - n] = (1 - xn) / (1 + xn)
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
def _fdtd_dx_field(nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1):
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                curl_h = Hz[i, j, k] - Hz[i, j - 1, k] - Hy[i, j, k] + Hy[i, j, k - 1]
                iDx[i, j, k] += curl_h
                Dx[i, j, k] = gj3[j] * gk3[k] * Dx[i, j, k] + gj2[j] * gk2[k] * (
                    0.5 * curl_h + gi1[i] * iDx[i, j, k]
                )
    return Dx, iDx


@numba.jit(nopython=True)
def _fdtd_dy_field(nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1):
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                curl_h = Hx[i, j, k] - Hx[i, j, k - 1] - Hz[i, j, k] + Hz[i - 1, j, k]
                iDy[i, j, k] += curl_h
                Dy[i, j, k] = gi3[i] * gk3[k] * Dy[i, j, k] + gi2[i] * gk2[k] * (
                    0.5 * curl_h + gj1[j] * iDy[i, j, k]
                )
    return Dy, iDy


@numba.jit(nopython=True)
def _fdtd_dz_field(nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1):
    for i in range(1, nx):
        for j in range(1, ny):
            for k in range(1, nz):
                curl_h = Hy[i, j, k] - Hy[i - 1, j, k] - Hx[i, j, k] + Hx[i, j - 1, k]
                iDz[i, j, k] += curl_h
                Dz[i, j, k] = gi3[i] * gj3[j] * Dz[i, j, k] + gi2[i] * gj2[j] * (
                    0.5 * curl_h + gk1[k] * iDz[i, j, k]
                )
    return Dz, iDz


@numba.jit(nopython=True)
def _fdtd_inc_dy(ia, ib, ja, jb, ka, kb, Dy, hx_inc):
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Dy[i, j, ka] -= 0.5 * hx_inc[j]
            Dy[i, j, kb + 1] += 0.5 * hx_inc[j]
    return Dy


@numba.jit(nopython=True)
def _fdtd_inc_dz(ia, ib, ja, jb, ka, kb, Dz, hx_inc):
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Dz[i, ja, k] += 0.5 * hx_inc[ja - 1]
            Dz[i, jb, k] -= 0.5 * hx_inc[jb]
    return Dz


@numba.jit(nopython=True)
def _fdtd_e_fields(
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
):
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                Ex[i, j, k] = eps_x[i, j, k] * (Dx[i, j, k] - Ix[i, j, k])
                Ix[i, j, k] += conductivity_x[i, j, k] * Ex[i, j, k]
                Ey[i, j, k] = eps_y[i, j, k] * (Dy[i, j, k] - Iy[i, j, k])
                Iy[i, j, k] += conductivity_y[i, j, k] * Ey[i, j, k]
                Ez[i, j, k] = eps_z[i, j, k] * (Dz[i, j, k] - Iz[i, j, k])
                Iz[i, j, k] += conductivity_z[i, j, k] * Ez[i, j, k]
    return Ex, Ey, Ez, Ix, Iy, Iz


@numba.jit(nopython=True)
def _fdtd_hx_inc(ny, hx_inc, ez_inc):
    for j in range(ny - 1):
        hx_inc[j] += 0.5 * (ez_inc[j] - ez_inc[j + 1])
    return hx_inc


@numba.jit(nopython=True)
def _fdtd_hx_with_inc(ia, ib, ja, jb, ka, kb, Hx, ez_inc):
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Hx[i, ja - 1, k] += 0.5 * ez_inc[ja]
            Hx[i, jb, k] -= 0.5 * ez_inc[jb]
    return Hx


@numba.jit(nopython=True)
def _fdtd_hy_with_inc(ia, ib, ja, jb, ka, kb, Hy, ez_inc):
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Hy[ia - 1, j, k] -= 0.5 * ez_inc[j]
            Hy[ib, j, k] += 0.5 * ez_inc[j]
    return Hy


@numba.jit(nopython=True)
def _fdtd_hx_field(nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3):
    for i in range(nx):
        for j in range(ny - 1):
            for k in range(nz - 1):
                curl_e = Ey[i, j, k + 1] - Ey[i, j, k] - Ez[i, j + 1, k] + Ez[i, j, k]
                iHx[i, j, k] += curl_e
                Hx[i, j, k] = fj3[j] * fk3[k] * Hx[i, j, k] + fj2[j] * fk2[k] * 0.5 * (
                    curl_e + fi1[i] * iHx[i, j, k]
                )
    return Hx, iHx


@numba.jit(nopython=True)
def _fdtd_hy_field(nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3):
    for i in range(nx - 1):
        for j in range(ny):
            for k in range(nz - 1):
                curl_e = Ez[i + 1, j, k] - Ez[i, j, k] - Ex[i, j, k + 1] + Ex[i, j, k]
                iHy[i, j, k] += curl_e
                Hy[i, j, k] = fi3[i] * fk3[k] * Hy[i, j, k] + fi2[i] * fk2[k] * 0.5 * (
                    curl_e + fj1[j] * iHy[i, j, k]
                )
    return Hy, iHy


@numba.jit(nopython=True)
def _fdtd_hz_field(nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3):
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz):
                curl_e = Ex[i, j + 1, k] - Ex[i, j, k] - Ey[i + 1, j, k] + Ey[i, j, k]
                iHz[i, j, k] += curl_e
                Hz[i, j, k] = fi3[i] * fj3[j] * Hz[i, j, k] + fi2[i] * fj2[j] * 0.5 * (
                    curl_e + fk1[k] * iHz[i, j, k]
                )
    return Hz, iHz


@numba.jit(nopython=True)
def _fdtd_sar_instant(nx, ny, nz, Ex, Ey, Ez, sigma_x, sigma_y, sigma_z, rho):
    sar = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                e_mag_sq = Ex[i, j, k] ** 2 + Ey[i, j, k] ** 2 + Ez[i, j, k] ** 2
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_mag_sq) / (2.0 * rho[i, j, k])
    return sar


def _run_segmentation_from_data_dir(data_dir, max_cases, results_dir):
    """
    Run BC_MRI_SEG breast_tumor_segmentation_model to produce segmentations.
    data_dir: e.g. data/ISPY1 (must contain images_std/ or images/ and masks/).
    Writes to results_dir/data/ and results_dir/images/. Returns list of (seg_path, output_base).
    """
    import breast_tumor_segmentation_model as seg_module  # noqa: PLC0415  # type: ignore

    seg_module.run_segmentation(
        data_dir=data_dir,
        max_cases=max_cases,
        results_dir=results_dir,
    )
    data_subdir = os.path.join(results_dir, "data")
    seg_files = sorted(
        [f for f in os.listdir(data_subdir) if f.endswith("_segmentation.npy")]
    )
    if not seg_files:
        raise FileNotFoundError(f"No *_segmentation.npy found in {data_subdir}")
    out = []
    for f in seg_files:
        base = f.replace("_segmentation.npy", "")
        out.append((os.path.join(data_subdir, f), base))
    return out


def _run_segmentation_single_image(image_path, results_dir):
    """
    Run BC_MRI_SEG segment_single_file on one image .npy. Mask path inferred from image path.
    Writes to results_dir/data/. Returns (seg_path, output_base).
    """
    import breast_tumor_segmentation_model as seg_module  # noqa: PLC0415  # type: ignore

    seg_module.segment_single_file(npy_path=image_path, results_dir=results_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]
    data_subdir = os.path.join(results_dir, "data")
    seg_path = os.path.join(data_subdir, f"{base}_segmentation.npy")
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    return (seg_path, base)


def load_breast_segmentation(seg_path):
    """Load 3-class segmentation (H,W,D) int32: 0=bg, 1=tumor, 2=healthy breast."""
    seg = np.load(seg_path).astype(np.int32)
    if seg.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got shape {seg.shape}")
    return seg


def downsample_segmentation(labels_3d, max_dim):
    """Downsample to grid with max dimension max_dim. Nearest-neighbor."""
    nx, ny, nz = labels_3d.shape
    scale = min(max_dim / nx, max_dim / ny, max_dim / nz, 1.0)
    if scale >= 1.0:
        return labels_3d
    out = ndimage.zoom(
        labels_3d.astype(np.float32),
        (scale, scale, scale),
        order=0,
        mode="nearest",
    )
    out = np.round(out).astype(np.int32)
    out = np.clip(out, 0, 2)
    return out


def run_fdtd_breast(
    labels_3d,
    time_steps,
    stream_frames=True,
    stream_interval=1,
    pulse_amplitude=100.0,
    pulse_width=8,
    pulse_delay=20,
    data_dir=None,
    output_base="breast_fdtd",
):
    """
    Full 3D Yee FDTD on breast grid (same as fdtd_brain_simulation_engine).
    Plane wave +y, Gaussian pulse only. Material arrays from labels: 0=air, 1=tumor, 2=healthy breast.
    Returns (n_frames, grid_shape) and writes E_frames_part*.npz (Ez), SAR_frames_part*.npz.
    """
    nx, ny, nz = labels_3d.shape
    dx = 0.01
    dt = dx / 6e8
    epsz = 8.854e-12

    # Tissue at ~100 MHz: (eps_r, sigma S/m, rho kg/m³)
    TISSUE_TABLE = {
        0: (1.0, 0.0, 0.0),
        1: (60.0, 0.8, 1050.0),  # tumor
        2: (50.0, 0.5, 1040.0),  # healthy breast
    }

    eps_x = np.ones((nx, ny, nz))
    eps_y = np.ones((nx, ny, nz))
    eps_z = np.ones((nx, ny, nz))
    conductivity_x = np.zeros((nx, ny, nz))
    conductivity_y = np.zeros((nx, ny, nz))
    conductivity_z = np.zeros((nx, ny, nz))
    sigma_x = np.zeros((nx, ny, nz))
    sigma_y = np.zeros((nx, ny, nz))
    sigma_z = np.zeros((nx, ny, nz))
    rho = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                lab = int(labels_3d[i, j, k])
                lab = min(max(lab, 0), 2)
                eps_r, sigma_val, rho_val = TISSUE_TABLE[lab]
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

    # Field arrays (Yee)
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

    npml = 8
    ia, ib = npml, nx - npml - 1
    ja, jb = npml, ny - npml - 1
    ka, kb = npml, nz - npml - 1
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
    ) = _calculate_pml_parameters(npml, nx, ny, nz)

    # Plane wave +y: 1D incident buffers (Ez, Hx along y)
    ez_inc = np.zeros(ny)
    hx_inc = np.zeros(ny)
    boundary_low = [0.0, 0.0]
    boundary_high = [0.0, 0.0]
    inj_y = 3  # inject at low-y

    E_frames_dir = os.path.join(data_dir, "E_frames")
    SAR_frames_dir = os.path.join(data_dir, "SAR_frames")
    os.makedirs(E_frames_dir, exist_ok=True)
    os.makedirs(SAR_frames_dir, exist_ok=True)
    E_buffer, SAR_buffer = [], []
    stream_part = 0
    n_frames_saved = 0

    for time_step in range(1, time_steps + 1):
        pulse = pulse_amplitude * exp(
            -0.5 * ((pulse_delay - time_step) / pulse_width) ** 2
        )

        # 1D incident Ez update for +y propagation (dEz/dt from dHx/dy)
        for j in range(1, ny - 1):
            ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])
        ez_inc[0] = boundary_low.pop(0)
        boundary_low.append(ez_inc[1])
        ez_inc[ny - 1] = boundary_high.pop(0)
        boundary_high.append(ez_inc[ny - 2])
        ez_inc[inj_y] = pulse

        # D-field updates (curl of H)
        Dx, iDx = _fdtd_dx_field(nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1)
        Dy, iDy = _fdtd_dy_field(nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1)
        Dz, iDz = _fdtd_dz_field(nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1)

        # Add incident field for plane wave +y
        Dy = _fdtd_inc_dy(ia, ib, ja, jb, ka, kb, Dy, hx_inc)
        Dz = _fdtd_inc_dz(ia, ib, ja, jb, ka, kb, Dz, hx_inc)

        # E from D
        Ex, Ey, Ez, Ix, Iy, Iz = _fdtd_e_fields(
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

        # Instantaneous SAR for streaming
        sar = _fdtd_sar_instant(nx, ny, nz, Ex, Ey, Ez, sigma_x, sigma_y, sigma_z, rho)

        # H-field updates: incident first, then curl of E
        hx_inc = _fdtd_hx_inc(ny, hx_inc, ez_inc)
        Hx, iHx = _fdtd_hx_field(nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3)
        Hx = _fdtd_hx_with_inc(ia, ib, ja, jb, ka, kb, Hx, ez_inc)
        Hy, iHy = _fdtd_hy_field(nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3)
        Hy = _fdtd_hy_with_inc(ia, ib, ja, jb, ka, kb, Hy, ez_inc)
        Hz, iHz = _fdtd_hz_field(nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3)

        if stream_frames and (time_step % stream_interval == 0):
            E_buffer.append(Ez.copy().astype(np.float32))
            SAR_buffer.append(sar.astype(np.float32))
            n_frames_saved += 1
            if len(E_buffer) >= STREAM_CHUNK_SIZE:
                np.savez_compressed(
                    os.path.join(
                        E_frames_dir, f"{output_base}_E_frames_part{stream_part}.npz"
                    ),
                    E_frames=np.array(E_buffer, dtype=np.float32),
                )
                np.savez_compressed(
                    os.path.join(
                        SAR_frames_dir,
                        f"{output_base}_SAR_frames_part{stream_part}.npz",
                    ),
                    SAR_frames=np.array(SAR_buffer, dtype=np.float32),
                )
                E_buffer.clear()
                SAR_buffer.clear()
                stream_part += 1

    if E_buffer:
        np.savez_compressed(
            os.path.join(E_frames_dir, f"{output_base}_E_frames_part{stream_part}.npz"),
            E_frames=np.array(E_buffer, dtype=np.float32),
        )
        np.savez_compressed(
            os.path.join(
                SAR_frames_dir, f"{output_base}_SAR_frames_part{stream_part}.npz"
            ),
            SAR_frames=np.array(SAR_buffer, dtype=np.float32),
        )

    return n_frames_saved, (nx, ny, nz)


def _run_one_case(seg_path, results_dir, data_dir, images_dir, output_base, args):
    """Load seg, downsample, run FDTD, write metadata and geometry; optionally build animations."""
    print(f"\n--- Case: {output_base} ---")
    print("Loading breast segmentation...")
    labels_3d = load_breast_segmentation(seg_path)
    print(f"  Shape: {labels_3d.shape}")

    print(f"Downsampling to max_dim={args.max_dim}...")
    labels_3d = downsample_segmentation(labels_3d, args.max_dim)
    nx, ny, nz = labels_3d.shape
    print(f"  Grid: ({nx}, {ny}, {nz})")

    np.save(os.path.join(data_dir, f"{output_base}_segmentation.npy"), labels_3d)

    print("Running FDTD...")
    t0 = time.perf_counter()
    n_frames, grid_shape = run_fdtd_breast(
        labels_3d,
        time_steps=args.time_steps,
        stream_frames=args.stream_frames,
        stream_interval=args.stream_frame_interval,
        data_dir=data_dir,
        output_base=output_base,
    )
    t1 = time.perf_counter()
    print(f"  Done. {n_frames} frames in {t1 - t0:.1f} s")

    n_parts = (n_frames + STREAM_CHUNK_SIZE - 1) // STREAM_CHUNK_SIZE if n_frames else 0
    metadata = {
        "output_base": output_base,
        "grid_shape": list(grid_shape),
        "n_frames": n_frames,
        "E_frames_chunk_size": STREAM_CHUNK_SIZE,
        "time_steps": args.time_steps,
        "segmentation_path": seg_path,
        "segmentation_labels": {"0": "background", "1": "tumor", "2": "healthy_breast"},
    }
    meta_path = os.path.join(data_dir, f"{output_base}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        rgb = np.zeros((*labels_3d.shape[:2], 3))
        rgb[labels_3d[:, :, nz // 2] == LABEL_TUMOR] = [1, 0, 0]
        rgb[labels_3d[:, :, nz // 2] == LABEL_HEALTHY_BREAST] = [0.3, 0.5, 0.8]
        ax.imshow(rgb, origin="lower")
        ax.set_title(f"Breast FDTD geometry (mid-z) — {output_base}")
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        fig.savefig(
            os.path.join(images_dir, f"{output_base}_fdtd_geometry_slice.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved geometry slice to {images_dir}/")

    if not args.no_animations and n_frames > 0:
        try:
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        _SCRIPT_DIR, "build_breast_animations_from_streamed_frames.py"
                    ),
                    results_dir,  # positional results_dir (script has no --results-dir)
                    "--data-dir",
                    data_dir,
                    "--output-base",
                    output_base,
                ],
                check=True,
                cwd=_SCRIPT_DIR,
            )
        except Exception as e:
            print(f"  Animation build skipped: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="FDTD breast simulation from BC_MRI_SEG voxelized segmentation."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a single image .npy (e.g. BC_MRI_SEG/data/ISPY1/images_std/ispy1_12.npy). Mask inferred from path; runs segmentation then FDTD.",
    )
    parser.add_argument(
        "--seg",
        default=None,
        help="Path to a single segmentation .npy (0=bg, 1=tumor, 2=healthy breast). Omit when using -i or --data-dir.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory with images_std/ or images/ and masks/ (e.g. data/ISPY1). Runs BC_MRI_SEG then FDTD per case.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="When using --data-dir, limit number of cases (default: all).",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_MAX_DIM,
        help=f"Max grid dimension (default: {DEFAULT_MAX_DIM}).",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=DEFAULT_TIME_STEPS,
        help=f"FDTD time steps (default: {DEFAULT_TIME_STEPS}).",
    )
    parser.add_argument(
        "--stream-frames",
        action="store_true",
        default=True,
        help="Stream E and SAR frames to disk (default: True).",
    )
    parser.add_argument(
        "--stream-frame-interval",
        type=int,
        default=1,
        help="Save a frame every N steps (default: 1).",
    )
    parser.add_argument(
        "--no-animations",
        action="store_true",
        help="Do not build MP4 animations; use build_breast_animations_from_streamed_frames.py later.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override results directory (default: results/DDMMYY-HHMMSS).",
    )
    args = parser.parse_args()

    if not args.seg and not args.data_dir and not args.input:
        print(
            "Error: provide one of -i/--input <image.npy>, --seg <segmentation.npy>, or --data-dir <dir>.",
            file=sys.stderr,
        )
        sys.exit(1)
    if sum(bool(x) for x in (args.seg, args.data_dir, args.input)) > 1:
        print(
            "Error: provide only one of -i/--input, --seg, or --data-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    timestamp_str = datetime.now().strftime("%d%m%y-%H%M%S")
    results_dir = args.output_dir or os.path.join(
        _SCRIPT_DIR, "results", f"breast_fdtd_{timestamp_str}"
    )
    data_dir = os.path.join(results_dir, "data")
    images_dir = os.path.join(results_dir, "images")
    animations_dir = os.path.join(results_dir, "animations")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(animations_dir, exist_ok=True)

    cases = []  # list of (seg_path, output_base)

    if args.input:
        image_path = os.path.abspath(args.input)
        if not os.path.isfile(image_path):
            print(f"Error: image not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        print("Running segmentation from single image (BC_MRI_SEG)...")
        seg_path, output_base = _run_segmentation_single_image(image_path, results_dir)
        cases = [(seg_path, output_base)]
        print(f"  Case {output_base} ready for FDTD.")
    elif args.data_dir:
        # Run BC_MRI_SEG segmentation into this run's results dir so segmentations live in data/
        seg_results_dir = results_dir
        print("Running segmentation from image data (BC_MRI_SEG)...")
        cases = _run_segmentation_from_data_dir(
            data_dir=(
                os.path.abspath(args.data_dir)
                if not os.path.isabs(args.data_dir)
                else args.data_dir
            ),
            max_cases=args.max_cases,
            results_dir=seg_results_dir,
        )
        print(f"  {len(cases)} case(s) ready for FDTD.")
    else:
        seg_path = os.path.abspath(args.seg)
        if not os.path.isfile(seg_path):
            print(f"Error: segmentation not found: {seg_path}", file=sys.stderr)
            sys.exit(1)
        output_base = os.path.splitext(os.path.basename(seg_path))[0].replace(
            "_segmentation", ""
        )
        cases = [(seg_path, output_base)]

    for seg_path, output_base in cases:
        if not os.path.isfile(seg_path):
            print(
                f"Skip {output_base}: segmentation not found at {seg_path}",
                file=sys.stderr,
            )
            continue
        _run_one_case(seg_path, results_dir, data_dir, images_dir, output_base, args)

    print(f"\nResults: {results_dir}/")
    print("  data/: segmentation.npy, E_frames/, SAR_frames/, metadata.json")
    print(
        f"  Build more animations: python build_breast_animations_from_streamed_frames.py {results_dir}"
    )


if __name__ == "__main__":
    main()
