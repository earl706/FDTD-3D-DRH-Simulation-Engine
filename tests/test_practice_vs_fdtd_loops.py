"""Compare practice_loops._run_minimal_fdtd_benchmark to core.fdtd.loops."""

from __future__ import annotations

import numpy as np
import pytest

from core.fdtd.loops import _run_minimal_fdtd_benchmark as ref_benchmark

from tests.conftest import load_optional_practice_module

mod = load_optional_practice_module("practice_loops.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice_loops.py with _run_minimal_fdtd_benchmark using your practice kernels.",
        allow_module_level=True,
    )


def test_minimal_benchmark_metadata_matches():
    assert hasattr(mod, "_run_minimal_fdtd_benchmark")
    nx, ny, nz, steps = 8, 8, 8, 6
    r = ref_benchmark(nx, ny, nz, steps)
    p = mod._run_minimal_fdtd_benchmark(nx, ny, nz, steps)
    assert r["grid_shape"] == p["grid_shape"]
    assert r["number_of_voxels"] == p["number_of_voxels"]
    assert r["time_steps"] == p["time_steps"]
    assert set(r.keys()) == set(p.keys())


def test_minimal_benchmark_deterministic_field_evolution():
    """Same kernels should yield identical final fields for identical steps (no sources)."""
    from core.fdtd import kernels as ref_k

    from tests.conftest import load_practice_kernels

    pk = load_practice_kernels()
    nx, ny, nz, steps = 6, 7, 8, 4
    npml = max(4, min(16, min(nx, ny, nz) // 10))

    def run_pair(kmod):
        coef = kmod.calculate_pml_parameters(npml, nx, ny, nz)
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
        ) = coef
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
        for _ in range(1, steps + 1):
            Dx, iDx = kmod.calculate_dx_field(
                nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
            )
            Dy, iDy = kmod.calculate_dy_field(
                nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
            )
            Dz, iDz = kmod.calculate_dz_field(
                nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
            )
            Ex, Ey, Ez, Ix, Iy, Iz = kmod.calculate_e_fields(
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
            Hx, iHx = kmod.calculate_hx_field(
                nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
            )
            Hy, iHy = kmod.calculate_hy_field(
                nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
            )
            Hz, iHz = kmod.calculate_hz_field(
                nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3
            )
        return Ez

    Ez_ref = run_pair(ref_k)
    Ez_p = run_pair(pk)
    np.testing.assert_allclose(Ez_p, Ez_ref, rtol=1e-9, atol=1e-12)
