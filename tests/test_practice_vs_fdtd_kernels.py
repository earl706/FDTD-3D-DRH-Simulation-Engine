"""Compare practice.py implementations to core.fdtd.kernels."""

from __future__ import annotations

import numpy as np
from core.fdtd import kernels as ref

from tests.conftest import assert_allclose_rtol, clone_ndarray


def test_calculate_pml_parameters(pk):
    npml, nx, ny, nz = 4, 16, 18, 20
    out_r = ref.calculate_pml_parameters(npml, nx, ny, nz)
    out_p = pk.calculate_pml_parameters(npml, nx, ny, nz)
    assert len(out_r) == len(out_p) == 18
    for a, b in zip(out_r, out_p):
        assert_allclose_rtol(a, b)


def test_calculate_dx_field(pk):
    nx, ny, nz = 10, 11, 12
    npml = 4
    coef = ref.calculate_pml_parameters(npml, nx, ny, nz)
    gj3, gk3, gj2, gk2, gi1 = coef[8], coef[14], coef[7], coef[13], coef[0]
    Hy = np.random.default_rng(0).standard_normal((nx, ny, nz))
    Hz = np.random.default_rng(1).standard_normal((nx, ny, nz))
    Dx_r, iDx_r = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
    Dx_p, iDx_p = clone_ndarray(Dx_r), clone_ndarray(iDx_r)
    ref.calculate_dx_field(nx, ny, nz, Dx_r, iDx_r, Hy, Hz, gj3, gk3, gj2, gk2, gi1)
    pk.calculate_dx_field(nx, ny, nz, Dx_p, iDx_p, Hy, Hz, gj3, gk3, gj2, gk2, gi1)
    assert_allclose_rtol(Dx_p, Dx_r)
    assert_allclose_rtol(iDx_p, iDx_r)


def test_calculate_dy_field(pk):
    nx, ny, nz = 10, 11, 12
    npml = 4
    coef = ref.calculate_pml_parameters(npml, nx, ny, nz)
    gi3, gk3, gi2, gk2, gj1 = coef[2], coef[14], coef[1], coef[13], coef[6]
    rng = np.random.default_rng(2)
    Hx = rng.standard_normal((nx, ny, nz))
    Hz = rng.standard_normal((nx, ny, nz))
    Dy_r, iDy_r = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
    Dy_p, iDy_p = clone_ndarray(Dy_r), clone_ndarray(iDy_r)
    ref.calculate_dy_field(nx, ny, nz, Dy_r, iDy_r, Hx, Hz, gi3, gk3, gi2, gk2, gj1)
    pk.calculate_dy_field(nx, ny, nz, Dy_p, iDy_p, Hx, Hz, gi3, gk3, gi2, gk2, gj1)
    assert_allclose_rtol(Dy_p, Dy_r)
    assert_allclose_rtol(iDy_p, iDy_r)


def test_calculate_dz_field(pk):
    nx, ny, nz = 10, 11, 12
    npml = 4
    coef = ref.calculate_pml_parameters(npml, nx, ny, nz)
    gi3, gj3, gi2, gj2, gk1 = coef[2], coef[8], coef[1], coef[7], coef[12]
    rng = np.random.default_rng(3)
    Hx = rng.standard_normal((nx, ny, nz))
    Hy = rng.standard_normal((nx, ny, nz))
    Dz_r, iDz_r = np.zeros((nx, ny, nz)), np.zeros((nx, ny, nz))
    Dz_p, iDz_p = clone_ndarray(Dz_r), clone_ndarray(iDz_r)
    ref.calculate_dz_field(nx, ny, nz, Dz_r, iDz_r, Hx, Hy, gi3, gj3, gi2, gj2, gk1)
    pk.calculate_dz_field(nx, ny, nz, Dz_p, iDz_p, Hx, Hy, gi3, gj3, gi2, gj2, gk1)
    assert_allclose_rtol(Dz_p, Dz_r)
    assert_allclose_rtol(iDz_p, iDz_r)


def test_calculate_e_fields(pk):
    nx, ny, nz = 8, 9, 10
    rng = np.random.default_rng(4)
    Dx = rng.standard_normal((nx, ny, nz))
    Dy = rng.standard_normal((nx, ny, nz))
    Dz = rng.standard_normal((nx, ny, nz))
    eps_x = np.abs(rng.standard_normal((nx, ny, nz))) + 0.5
    eps_y = np.abs(rng.standard_normal((nx, ny, nz))) + 0.5
    eps_z = np.abs(rng.standard_normal((nx, ny, nz))) + 0.5
    cx = rng.random((nx, ny, nz)) * 0.1
    cy = rng.random((nx, ny, nz)) * 0.1
    cz = rng.random((nx, ny, nz)) * 0.1
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Ix = np.zeros((nx, ny, nz))
    Iy = np.zeros((nx, ny, nz))
    Iz = np.zeros((nx, ny, nz))
    args = (
        nx,
        ny,
        nz,
        Dx,
        Dy,
        Dz,
        eps_x,
        eps_y,
        eps_z,
        cx,
        cy,
        cz,
        Ex,
        Ey,
        Ez,
        Ix,
        Iy,
        Iz,
    )
    Ex_p, Ey_p, Ez_p = clone_ndarray(Ex), clone_ndarray(Ey), clone_ndarray(Ez)
    Ix_p, Iy_p, Iz_p = clone_ndarray(Ix), clone_ndarray(Iy), clone_ndarray(Iz)
    ref.calculate_e_fields(*args)
    pk.calculate_e_fields(
        nx,
        ny,
        nz,
        Dx,
        Dy,
        Dz,
        eps_x,
        eps_y,
        eps_z,
        cx,
        cy,
        cz,
        Ex_p,
        Ey_p,
        Ez_p,
        Ix_p,
        Iy_p,
        Iz_p,
    )
    assert_allclose_rtol(Ex_p, Ex)
    assert_allclose_rtol(Ey_p, Ey)
    assert_allclose_rtol(Ez_p, Ez)
    assert_allclose_rtol(Ix_p, Ix)
    assert_allclose_rtol(Iy_p, Iy)
    assert_allclose_rtol(Iz_p, Iz)


def test_calculate_fourier_transform_ex(pk):
    nx, ny = 6, 7
    nf = 3
    rng = np.random.default_rng(5)
    Ez = rng.standard_normal((nx, ny, 14))
    source_z = 5
    arg = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    time_step = 1.23e-12
    real_r = np.zeros((nf, nx, ny))
    imag_r = np.zeros((nf, nx, ny))
    real_p = clone_ndarray(real_r)
    imag_p = clone_ndarray(imag_r)
    ref.calculate_fourier_transform_ex(
        nx, ny, nf, real_r, imag_r, Ez, arg, time_step, source_z
    )
    pk.calculate_fourier_transform_ex(
        nx, ny, nf, real_p, imag_p, Ez, arg, time_step, source_z
    )
    assert_allclose_rtol(real_p, real_r)
    assert_allclose_rtol(imag_p, imag_r)


def test_calculate_hx_field(pk):
    nx, ny, nz = 10, 11, 12
    npml = 4
    coef = ref.calculate_pml_parameters(npml, nx, ny, nz)
    fi1, fj2, fk2, fj3, fk3 = coef[3], coef[10], coef[16], coef[11], coef[17]
    rng = np.random.default_rng(6)
    Ey = rng.standard_normal((nx, ny, nz))
    Ez = rng.standard_normal((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    iHx = np.zeros((nx, ny, nz))
    Hx_p, iHx_p = clone_ndarray(Hx), clone_ndarray(iHx)
    ref.calculate_hx_field(nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3)
    pk.calculate_hx_field(nx, ny, nz, Hx_p, iHx_p, Ey, Ez, fi1, fj2, fk2, fj3, fk3)
    assert_allclose_rtol(Hx_p, Hx)
    assert_allclose_rtol(iHx_p, iHx)


def test_calculate_hy_field(pk):
    nx, ny, nz = 10, 11, 12
    npml = 4
    coef = ref.calculate_pml_parameters(npml, nx, ny, nz)
    fj1, fi2, fk2, fi3, fk3 = coef[6], coef[4], coef[16], coef[5], coef[17]
    rng = np.random.default_rng(7)
    Ex = rng.standard_normal((nx, ny, nz))
    Ez = rng.standard_normal((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    iHy = np.zeros((nx, ny, nz))
    Hy_p, iHy_p = clone_ndarray(Hy), clone_ndarray(iHy)
    ref.calculate_hy_field(nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3)
    pk.calculate_hy_field(nx, ny, nz, Hy_p, iHy_p, Ex, Ez, fj1, fi2, fk2, fi3, fk3)
    assert_allclose_rtol(Hy_p, Hy)
    assert_allclose_rtol(iHy_p, iHy)


def test_calculate_hz_field(pk):
    nx, ny, nz = 10, 11, 12
    npml = 4
    coef = ref.calculate_pml_parameters(npml, nx, ny, nz)
    fk1, fi2, fj2, fi3, fj3 = coef[12], coef[4], coef[10], coef[5], coef[11]
    rng = np.random.default_rng(8)
    Ex = rng.standard_normal((nx, ny, nz))
    Ey = rng.standard_normal((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    iHz = np.zeros((nx, ny, nz))
    Hz_p, iHz_p = clone_ndarray(Hz), clone_ndarray(iHz)
    ref.calculate_hz_field(nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3)
    pk.calculate_hz_field(nx, ny, nz, Hz_p, iHz_p, Ex, Ey, fk1, fi2, fj2, fi3, fj3)
    assert_allclose_rtol(Hz_p, Hz)
    assert_allclose_rtol(iHz_p, iHz)


def test_accumulate_e_field_squared(pk):
    nx, ny, nz = 8, 9, 10
    rng = np.random.default_rng(9)
    Ex = rng.standard_normal((nx, ny, nz))
    Ey = rng.standard_normal((nx, ny, nz))
    Ez = rng.standard_normal((nx, ny, nz))
    Exs = np.zeros((nx, ny, nz))
    Eys = np.zeros((nx, ny, nz))
    Ezs = np.zeros((nx, ny, nz))
    Exs_p, Eys_p, Ezs_p = clone_ndarray(Exs), clone_ndarray(Eys), clone_ndarray(Ezs)
    ref.accumulate_e_field_squared(nx, ny, nz, Ex, Ey, Ez, Exs, Eys, Ezs)
    pk.accumulate_e_field_squared(nx, ny, nz, Ex, Ey, Ez, Exs_p, Eys_p, Ezs_p)
    assert_allclose_rtol(Exs_p, Exs)
    assert_allclose_rtol(Eys_p, Eys)
    assert_allclose_rtol(Ezs_p, Ezs)


def test_public_api_parity(pk):
    """practice.py should expose the same top-level function names as kernels."""
    ref_names = {n for n in dir(ref) if not n.startswith("_")}
    pk_names = {n for n in dir(pk) if not n.startswith("_")}
    for name in ref_names:
        if callable(getattr(ref, name)):
            assert name in pk_names, f"practice.py missing {name!r}"
