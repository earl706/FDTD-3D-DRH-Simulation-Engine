"""
Practice copy of the minimal FDTD benchmark (see core.fdtd.loops._run_minimal_fdtd_benchmark).

Uses kernels from practice.py. Tests compare this against core.fdtd.loops.
"""

from math import sqrt
import time
import numpy as np

import practice as pk

try:
    from performance_logging import get_peak_memory_mb as _get_peak_memory_mb
except ImportError:
    _get_peak_memory_mb = None


def _run_minimal_fdtd_benchmark(
    nx, ny, nz, time_steps, dx_mm=10.0, courant_factor=0.99
):
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
    ) = pk.calculate_pml_parameters(npml, nx, ny, nz)
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
        Dx, iDx = pk.calculate_dx_field(
            nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
        )
        Dy, iDy = pk.calculate_dy_field(
            nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
        )
        Dz, iDz = pk.calculate_dz_field(
            nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
        )
        Ex, Ey, Ez, Ix, Iy, Iz = pk.calculate_e_fields(
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
        Hx, iHx = pk.calculate_hx_field(
            nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
        )
        Hy, iHy = pk.calculate_hy_field(
            nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
        )
        Hz, iHz = pk.calculate_hz_field(
            nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3
        )
    t1 = time.perf_counter()
    total_wall_time_s = t1 - t0
    time_per_step_ms = 1000.0 * total_wall_time_s / time_steps if time_steps else None
    number_of_voxels = nx * ny * nz
    peak_memory_MB = _get_peak_memory_mb() if _get_peak_memory_mb is not None else None
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
