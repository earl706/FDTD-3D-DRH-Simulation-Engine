"""Unit tests for load_optimized_config replay helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from load_optimized_config import (
    LoadOptimizedError,
    load_optimized_antenna_params,
    resolve_optimization_artifact_path,
)


def test_resolve_prefers_antenna_optimization_in_dir(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    ant = d / "case_antenna_optimization.json"
    meta = d / "case_metadata.json"
    ant.write_text("{}", encoding="utf-8")
    meta.write_text("{}", encoding="utf-8")
    assert resolve_optimization_artifact_path(d) == ant


def test_resolve_unique_metadata(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    meta = d / "run_metadata.json"
    meta.write_text("{}", encoding="utf-8")
    assert resolve_optimization_artifact_path(d) == meta


def test_load_antenna_optimization_json(tmp_path: Path) -> None:
    p = tmp_path / "x_antenna_optimization.json"
    payload = {
        "f0_Hz": 100e6,
        "geometry": {"ring_offset": 10, "z_plane": 120},
        "optimized": {
            "alphas": [1.0, 1.1, 0.9, 1.0],
            "thetas_rad": [0.0, 1.0, 2.0, 3.0],
        },
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = load_optimized_antenna_params(p)
    assert out["f0_hz"] == 100e6
    assert out["ring_offset"] == 10
    assert out["z_plane"] == 120
    assert len(out["alphas"]) == 4


def test_load_metadata_optimized(tmp_path: Path) -> None:
    p = tmp_path / "seg_metadata.json"
    payload = {
        "output_base": "seg",
        "antenna_optimized": True,
        "optimized_f0_Hz": 85e6,
        "optimized_alphas": [1, 2, 3, 4],
        "optimized_thetas_rad": [0, 0, 0, 0],
        "optimized_quadrant_gaps": [[10, 12, 140], [180, 100, 140], [10, 188, 140], [12, 100, 140]],
        "opt_source_scale": 1.5,
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = load_optimized_antenna_params(p)
    assert out["f0_hz"] == 85e6
    assert out["ring_offset"] == 12
    assert out["z_plane"] == 140
    assert out["opt_source_scale"] == 1.5


def test_ambiguous_dir_raises(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    (d / "a_antenna_optimization.json").write_text("{}", encoding="utf-8")
    (d / "b_antenna_optimization.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LoadOptimizedError):
        resolve_optimization_artifact_path(d)
