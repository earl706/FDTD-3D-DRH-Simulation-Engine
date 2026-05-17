"""Compare practice_sources.py to core.fdtd.sources."""

from __future__ import annotations

import pytest

from core.fdtd import sources as ref

from tests.conftest import load_optional_practice_module

mod = load_optional_practice_module("practice.py")
if mod is None:
    pytest.skip(
        "Add CODE/practice_sources.py and reimplement core.fdtd.sources from memory.",
        allow_module_level=True,
    )


def test_build_quadrant_sources():
    assert hasattr(mod, "build_quadrant_sources")
    nx, ny, nz, npml = 40, 40, 32, 8
    out_r = ref.build_quadrant_sources(nx, ny, nz, npml)
    out_p = mod.build_quadrant_sources(nx, ny, nz, npml)
    assert len(out_r) == len(out_p) == 4
    for a, b in zip(out_r, out_p):
        assert a.keys() == b.keys()
        assert a["quadrant"] == b["quadrant"]
        assert a["gap"] == b["gap"]
        assert a["arms"] == b["arms"]
