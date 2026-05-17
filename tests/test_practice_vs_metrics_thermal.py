"""Compare practice_thermal.py to core.metrics.thermal."""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import thermal as ref

from tests.conftest import assert_allclose_rtol, load_optional_practice_module

mod = load_optional_practice_module("practice.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice_thermal.py and reimplement core.metrics.thermal from memory.",
        allow_module_level=True,
    )


def test_solve_steady_bioheat_3d():
    assert hasattr(mod, "solve_steady_bioheat_3d")
    nx, ny, nz = 8, 9, 10
    rng = np.random.default_rng(42)
    k_3d = np.zeros((nx, ny, nz), dtype=np.float64)
    k_3d[2:-2, 2:-2, 2:-2] = 0.5 + 0.1 * rng.random((nx - 4, ny - 4, nz - 4))
    Q_3d = rng.random((nx, ny, nz)) * 1e3
    Q_3d[k_3d <= 0] = 0.0
    dx = 1e-3
    T_r = ref.solve_steady_bioheat_3d(
        nx, ny, nz, k_3d, Q_3d, dx, T_boundary=37.0, max_iter=8000, tol=1e-5
    )
    T_p = mod.solve_steady_bioheat_3d(
        nx, ny, nz, k_3d, Q_3d, dx, T_boundary=37.0, max_iter=8000, tol=1e-5
    )
    assert_allclose_rtol(T_p, T_r, rtol=1e-6, atol=1e-5)
