"""Compare practice_boundaries.py to core.fdtd.boundaries (optional drill file)."""

from __future__ import annotations

import numpy as np
import pytest

from core.fdtd import boundaries as ref

from tests.conftest import (
    assert_allclose_rtol,
    clone_ndarray,
    load_optional_practice_module,
)

mod = load_optional_practice_module("practice.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice_boundaries.py and reimplement core.fdtd.boundaries from memory.",
        allow_module_level=True,
    )

_BOUNDARY_FUNCS = [
    "calculate_inc_dy_field",
    "calculate_inc_dz_field",
    "calculate_hx_inc",
    "calculate_hx_with_incident_field",
    "calculate_hy_with_incident_field",
    "update_ez_inc_x",
    "calculate_hy_inc_x",
    "calculate_inc_dz_field_x",
    "calculate_hy_with_incident_field_x",
    "update_ez_inc_z",
    "calculate_hx_inc_z",
    "calculate_inc_dz_field_z",
    "calculate_hx_with_incident_field_z",
]


@pytest.mark.parametrize("name", _BOUNDARY_FUNCS)
def test_function_exported(name):
    assert hasattr(mod, name), f"practice_boundaries.py missing {name!r}"
    assert callable(getattr(mod, name))


def test_calculate_inc_dy_field():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Dy_r = np.random.default_rng(0).standard_normal((nx, ny, nz))
    Dy_p = clone_ndarray(Dy_r)
    hx = np.random.default_rng(1).standard_normal(ny)
    ref.calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy_r, hx)
    mod.calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy_p, hx)
    assert_allclose_rtol(Dy_p, Dy_r)


def test_calculate_inc_dz_field():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Dz_r = np.random.default_rng(2).standard_normal((nx, ny, nz))
    Dz_p = clone_ndarray(Dz_r)
    hx = np.random.default_rng(3).standard_normal(ny)
    ref.calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz_r, hx)
    mod.calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz_p, hx)
    assert_allclose_rtol(Dz_p, Dz_r)


def test_calculate_hx_inc():
    sy = 20
    hx_r = np.random.default_rng(4).standard_normal(sy)
    hx_p = clone_ndarray(hx_r)
    ez = np.random.default_rng(5).standard_normal(sy)
    ref.calculate_hx_inc(sy, hx_r, ez)
    mod.calculate_hx_inc(sy, hx_p, ez)
    assert_allclose_rtol(hx_p, hx_r)


def test_calculate_hx_with_incident_field():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Hx_r = np.random.default_rng(6).standard_normal((nx, ny, nz))
    Hx_p = clone_ndarray(Hx_r)
    ez_inc = np.random.default_rng(7).standard_normal(ny)
    ref.calculate_hx_with_incident_field(ia, ib, ja, jb, ka, kb, Hx_r, ez_inc)
    mod.calculate_hx_with_incident_field(ia, ib, ja, jb, ka, kb, Hx_p, ez_inc)
    assert_allclose_rtol(Hx_p, Hx_r)


def test_calculate_hy_with_incident_field():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Hy_r = np.random.default_rng(8).standard_normal((nx, ny, nz))
    Hy_p = clone_ndarray(Hy_r)
    ez_inc = np.random.default_rng(9).standard_normal(ny)
    ref.calculate_hy_with_incident_field(ia, ib, ja, jb, ka, kb, Hy_r, ez_inc)
    mod.calculate_hy_with_incident_field(ia, ib, ja, jb, ka, kb, Hy_p, ez_inc)
    assert_allclose_rtol(Hy_p, Hy_r)


def test_update_ez_inc_x():
    sx = 24
    ez_r = np.random.default_rng(10).standard_normal(sx)
    ez_p = clone_ndarray(ez_r)
    hy = np.random.default_rng(11).standard_normal(sx)
    ref.update_ez_inc_x(sx, ez_r, hy)
    mod.update_ez_inc_x(sx, ez_p, hy)
    assert_allclose_rtol(ez_p, ez_r)


def test_calculate_hy_inc_x():
    sx = 24
    hy_r = np.random.default_rng(12).standard_normal(sx)
    hy_p = clone_ndarray(hy_r)
    ez = np.random.default_rng(13).standard_normal(sx)
    ref.calculate_hy_inc_x(sx, hy_r, ez)
    mod.calculate_hy_inc_x(sx, hy_p, ez)
    assert_allclose_rtol(hy_p, hy_r)


def test_calculate_inc_dz_field_x():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Dz_r = np.random.default_rng(14).standard_normal((nx, ny, nz))
    Dz_p = clone_ndarray(Dz_r)
    hy_inc = np.random.default_rng(15).standard_normal(nx)
    ref.calculate_inc_dz_field_x(ia, ib, ja, jb, ka, kb, Dz_r, hy_inc)
    mod.calculate_inc_dz_field_x(ia, ib, ja, jb, ka, kb, Dz_p, hy_inc)
    assert_allclose_rtol(Dz_p, Dz_r)


def test_calculate_hy_with_incident_field_x():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Hy_r = np.random.default_rng(16).standard_normal((nx, ny, nz))
    Hy_p = clone_ndarray(Hy_r)
    ez_x = np.random.default_rng(17).standard_normal(nx)
    ref.calculate_hy_with_incident_field_x(ia, ib, ja, jb, ka, kb, Hy_r, ez_x)
    mod.calculate_hy_with_incident_field_x(ia, ib, ja, jb, ka, kb, Hy_p, ez_x)
    assert_allclose_rtol(Hy_p, Hy_r)


def test_update_ez_inc_z():
    sz = 20
    ez_r = np.random.default_rng(18).standard_normal(sz)
    ez_p = clone_ndarray(ez_r)
    hx = np.random.default_rng(19).standard_normal(sz)
    ref.update_ez_inc_z(sz, ez_r, hx)
    mod.update_ez_inc_z(sz, ez_p, hx)
    assert_allclose_rtol(ez_p, ez_r)


def test_calculate_hx_inc_z():
    sz = 20
    hx_r = np.random.default_rng(20).standard_normal(sz)
    hx_p = clone_ndarray(hx_r)
    ez = np.random.default_rng(21).standard_normal(sz)
    ref.calculate_hx_inc_z(sz, hx_r, ez)
    mod.calculate_hx_inc_z(sz, hx_p, ez)
    assert_allclose_rtol(hx_p, hx_r)


def test_calculate_inc_dz_field_z():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Dz_r = np.random.default_rng(22).standard_normal((nx, ny, nz))
    Dz_p = clone_ndarray(Dz_r)
    hx_z = np.random.default_rng(23).standard_normal(nz)
    ref.calculate_inc_dz_field_z(ia, ib, ja, jb, ka, kb, Dz_r, hx_z)
    mod.calculate_inc_dz_field_z(ia, ib, ja, jb, ka, kb, Dz_p, hx_z)
    assert_allclose_rtol(Dz_p, Dz_r)


def test_calculate_hx_with_incident_field_z():
    nx, ny, nz = 16, 16, 16
    ia, ib, ja, jb, ka, kb = 3, 10, 3, 10, 2, 7
    Hx_r = np.random.default_rng(24).standard_normal((nx, ny, nz))
    Hx_p = clone_ndarray(Hx_r)
    ez_z = np.random.default_rng(25).standard_normal(nz)
    ref.calculate_hx_with_incident_field_z(ia, ib, ja, jb, ka, kb, Hx_r, ez_z)
    mod.calculate_hx_with_incident_field_z(ia, ib, ja, jb, ka, kb, Hx_p, ez_z)
    assert_allclose_rtol(Hx_p, Hx_r)
