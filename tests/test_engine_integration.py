"""
Integration tests: full `fdtd_brain_simulation_engine` subprocess runs on tiny synthetic data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]

try:
    import nibabel as nib
except ImportError:
    nib = None


pytestmark = pytest.mark.skipif(nib is None, reason="nibabel required for engine integration tests")


def _write_tiny_brain_seg_nifti(path: Path) -> None:
    """20³ BraTS-style labels: air (0), tumor (1), healthy brain (4) for objective J."""
    lab = np.zeros((20, 20, 20), dtype=np.int32)
    lab[6:14, 6:14, 6:10] = 1
    lab[6:14, 6:14, 10:14] = 4
    affine = np.eye(4, dtype=np.float64)
    img = nib.Nifti1Image(lab, affine)
    nib.save(img, str(path))


def _run_engine(cwd: Path, argv: list[str]) -> subprocess.CompletedProcess:
    engine = cwd / "fdtd_brain_simulation_engine.py"
    return subprocess.run(
        [sys.executable, str(engine), *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.fixture
def tiny_seg_path(tmp_path: Path) -> Path:
    p = tmp_path / "tiny_brain_seg.nii.gz"
    _write_tiny_brain_seg_nifti(p)
    return p


@pytest.mark.integration
def test_engine_pipeline_outputs_metadata_sar_objective(tiny_seg_path, tmp_path):
    """Tiny grid, few steps: metadata JSON, SAR NIfTI, objective JSON must exist."""
    results_dir = tmp_path / "run_out"
    argv = [
        str(tiny_seg_path),
        "--max-dim",
        "20",
        "--time-steps",
        "50",
        "--results-dir",
        str(results_dir),
        "--skip-animations",
        "--quadrant-air-margin-cells",
        "0",
        "--no-stream-frames",
    ]
    proc = _run_engine(CODE_ROOT, argv)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    data_dir = results_dir / "data"
    metas = sorted(data_dir.glob("*_metadata.json"))
    assert metas, f"expected *_metadata.json under {data_dir}"
    stem = metas[0].stem
    base = stem[: -len("_metadata")]

    assert (data_dir / f"{base}_SAR.nii.gz").is_file()
    assert (data_dir / f"{base}_objective.json").is_file()

    with open(data_dir / f"{base}_objective.json", encoding="utf-8") as f:
        obj = json.load(f)
    assert "definition" in obj


@pytest.mark.integration
@pytest.mark.slow
def test_engine_optimize_antenna_smoke(tiny_seg_path, tmp_path):
    """Fast smoke: optimization path completes (tiny grid, minimal opt grid)."""
    results_dir = tmp_path / "opt_out"
    argv = [
        "--optimize-antenna",
        str(tiny_seg_path),
        "--max-dim",
        "20",
        "--results-dir",
        str(results_dir),
        "--skip-animations",
        "--no-stream-frames",
        "--quadrant-air-margin-cells",
        "0",
        "--opt-time-steps",
        "40",
        "--opt-phase-steps",
        "3",
        "--opt-amp-steps",
        "3",
        "--opt-refine-iters",
        "0",
        "--opt-multi-start",
        "1",
        "--opt-geom-offsets",
        "8",
        "--opt-parallel",
        "1",
    ]
    proc = _run_engine(CODE_ROOT, argv)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    data_dir = results_dir / "data"
    metas = sorted(data_dir.glob("*_metadata.json"))
    assert metas
    stem = metas[0].stem
    base = stem[: -len("_metadata")]
    assert (data_dir / f"{base}_metadata.json").is_file()
    assert (data_dir / f"{base}_antenna_optimization.json").is_file()
    assert (data_dir / f"{base}_SAR.nii.gz").is_file()
