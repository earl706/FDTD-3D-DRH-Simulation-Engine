"""Compare practice_voxel_model.py to core.materials.voxel_model."""

from __future__ import annotations

import numpy as np
import pytest

from core.materials import voxel_model as ref

from tests.conftest import assert_allclose_rtol, load_optional_practice_module

mod = load_optional_practice_module("practice_voxel_model.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice_voxel_model.py and reimplement core.materials.voxel_model from memory.",
        allow_module_level=True,
    )


def test_build_material_arrays():
    assert hasattr(mod, "build_material_arrays")
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=(6, 7, 8))
    dt = 1e-12
    epsz = 8.8541878128e-12
    out_r = ref.build_material_arrays(labels, dt, epsz)
    out_p = mod.build_material_arrays(labels, dt, epsz)
    assert len(out_r) == len(out_p) == 11
    for a, b in zip(out_r, out_p):
        assert_allclose_rtol(a, b)
