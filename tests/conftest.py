"""Pytest configuration: repo root on path and helpers for practice-vs-reference tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


def load_practice_kernels():
    """Load CODE/practice.py as a module (avoids clashing with a practice/ package)."""
    path = CODE_ROOT / "practice.py"
    if not path.is_file():
        pytest.fail(f"Expected {path}")
    name = "practice_kernels_drill"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load practice module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_optional_practice_module(filename: str):
    """
    Load an optional sibling of practice.py (e.g. practice_boundaries.py).
    Returns None if the file does not exist.
    """
    path = CODE_ROOT / filename
    if not path.is_file():
        return None
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_allclose_rtol(
    actual,
    desired,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
):
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def clone_ndarray(a: np.ndarray) -> np.ndarray:
    return np.array(a, copy=True, order="C")


@pytest.fixture(scope="session")
def pk():
    """Loaded practice.py module (same kernels as core.fdtd.kernels drill)."""
    return load_practice_kernels()
