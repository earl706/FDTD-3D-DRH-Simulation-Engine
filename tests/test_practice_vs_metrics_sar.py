"""Compare practice_sar.py to core.metrics.sar."""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import sar as ref

from tests.conftest import assert_allclose_rtol, load_optional_practice_module

mod = load_optional_practice_module("practice.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice_sar.py and reimplement core.metrics.sar from memory.",
        allow_module_level=True,
    )

_SAR_FUNCS = [
    "compute_instantaneous_sar",
    "compute_sar",
    "compute_sar_from_complex_field",
    "compute_j_ratio",
    "compute_robust_objective",
]


@pytest.mark.parametrize("name", _SAR_FUNCS)
def test_exported(name):
    assert hasattr(mod, name), f"practice_sar.py missing {name!r}"
    assert callable(getattr(mod, name))


def test_compute_instantaneous_sar():
    nx, ny, nz = 6, 7, 8
    rng = np.random.default_rng(0)
    Ex, Ey, Ez = (
        rng.standard_normal((nx, ny, nz)),
        rng.standard_normal((nx, ny, nz)),
        rng.standard_normal((nx, ny, nz)),
    )
    sx = np.abs(rng.standard_normal((nx, ny, nz))) + 0.01
    sy = np.abs(rng.standard_normal((nx, ny, nz))) + 0.01
    sz = np.abs(rng.standard_normal((nx, ny, nz))) + 0.01
    rho = np.abs(rng.standard_normal((nx, ny, nz))) * 500 + 100
    sar_r = ref.compute_instantaneous_sar(nx, ny, nz, Ex, Ey, Ez, sx, sy, sz, rho)
    sar_p = mod.compute_instantaneous_sar(nx, ny, nz, Ex, Ey, Ez, sx, sy, sz, rho)
    assert_allclose_rtol(sar_p, sar_r)


def test_compute_sar():
    nx, ny, nz = 6, 7, 8
    rng = np.random.default_rng(1)
    Exs = rng.random((nx, ny, nz))
    Eys = rng.random((nx, ny, nz))
    Ezs = rng.random((nx, ny, nz))
    sx = np.abs(rng.standard_normal((nx, ny, nz))) + 0.01
    sy = np.abs(rng.standard_normal((nx, ny, nz))) + 0.01
    sz = np.abs(rng.standard_normal((nx, ny, nz))) + 0.01
    rho = np.abs(rng.standard_normal((nx, ny, nz))) * 500 + 100
    n_samples = 100
    sar_r = ref.compute_sar(nx, ny, nz, Exs, Eys, Ezs, sx, sy, sz, rho, n_samples)
    sar_p = mod.compute_sar(nx, ny, nz, Exs, Eys, Ezs, sx, sy, sz, rho, n_samples)
    assert_allclose_rtol(sar_p, sar_r)


def test_compute_sar_from_complex_field():
    rng = np.random.default_rng(2)
    shape = (5, 6, 7)
    E_total = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    sigma_avg = np.abs(rng.standard_normal(shape)) + 0.01
    rho = np.abs(rng.standard_normal(shape)) * 400 + 50
    sar_r = ref.compute_sar_from_complex_field(E_total, sigma_avg, rho)
    sar_p = mod.compute_sar_from_complex_field(E_total, sigma_avg, rho)
    assert_allclose_rtol(sar_p, sar_r)


def test_compute_j_ratio():
    rng = np.random.default_rng(3)
    sar = np.abs(rng.standard_normal((8, 8, 8)))
    tumor = np.zeros(sar.shape, dtype=bool)
    tumor[2:5, 2:5, 2:5] = True
    healthy = np.zeros(sar.shape, dtype=bool)
    healthy[6:8, 6:8, 6:8] = True
    jr = ref.compute_j_ratio(sar, tumor, healthy)
    jp = mod.compute_j_ratio(sar, tumor, healthy)
    assert len(jr) == len(jp) == 3
    for a, b in zip(jr, jp):
        assert_allclose_rtol(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-12)


def test_compute_robust_objective():
    rng = np.random.default_rng(4)
    sar = np.abs(rng.standard_normal((8, 8, 8)))
    tumor = np.zeros(sar.shape, dtype=bool)
    tumor[2:5, 2:5, 2:5] = True
    healthy = np.zeros(sar.shape, dtype=bool)
    healthy[6:8, 6:8, 6:8] = True
    wr = ref.compute_robust_objective(sar, tumor, healthy, penalty_weight=0.2)
    wp = mod.compute_robust_objective(sar, tumor, healthy, penalty_weight=0.2)
    assert len(wr) == len(wp) == 5
    for a, b in zip(wr, wp):
        assert_allclose_rtol(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-12)
