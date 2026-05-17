"""
Load 4-quadrant APA parameters from a prior ``--optimize-antenna`` run for replay (no optimization).

Supported artifacts:
  - ``*_antenna_optimization.json`` (preferred; includes geometry.ring_offset / z_plane)
  - ``*_metadata.json`` from an optimized or quadrant_fixed run (geometry from optimized_quadrant_gaps)

A directory path resolves to a single ``*_antenna_optimization.json``, or else a single ``*_metadata.json``.

``*_freq_sweep.json`` lists baseline J vs frequency only and is not used for replay.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class LoadOptimizedError(ValueError):
    """Invalid path, ambiguous directory, or unrecognized JSON schema."""


def resolve_optimization_artifact_path(path: str | Path) -> Path:
    """Resolve a concrete JSON file: direct file, or unique match inside a directory."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()
    if p.is_file():
        return p
    if p.is_dir():
        opts = sorted(p.glob("*_antenna_optimization.json"))
        if len(opts) == 1:
            return opts[0]
        if len(opts) > 1:
            raise LoadOptimizedError(
                f"Directory {p} contains multiple *_antenna_optimization.json files; "
                "pass one file path explicitly."
            )
        metas = sorted(p.glob("*_metadata.json"))
        if len(metas) == 1:
            return metas[0]
        if len(metas) > 1:
            raise LoadOptimizedError(
                f"Directory {p} contains multiple *_metadata.json files; "
                "pass one file path explicitly."
            )
        raise FileNotFoundError(
            f"No *_antenna_optimization.json or *_metadata.json under {p}"
        )
    raise FileNotFoundError(f"Not a file or directory: {path}")


def _params_from_antenna_optimization_json(data: dict[str, Any]) -> dict[str, Any]:
    opt = data.get("optimized") or {}
    geom = data.get("geometry") or {}
    alphas = opt.get("alphas")
    thetas_rad = opt.get("thetas_rad")
    if not alphas or len(alphas) != 4:
        raise LoadOptimizedError("antenna_optimization.json: missing optimized.alphas (length 4)")
    if not thetas_rad or len(thetas_rad) != 4:
        raise LoadOptimizedError(
            "antenna_optimization.json: missing optimized.thetas_rad (length 4)"
        )
    f0 = data.get("f0_Hz")
    if f0 is None:
        raise LoadOptimizedError("antenna_optimization.json: missing f0_Hz")
    ring = geom.get("ring_offset")
    zpl = geom.get("z_plane")
    if ring is None or zpl is None:
        raise LoadOptimizedError(
            "antenna_optimization.json: missing geometry.ring_offset or geometry.z_plane"
        )
    return {
        "f0_hz": float(f0),
        "alphas": [float(x) for x in alphas],
        "thetas_rad": [float(x) for x in thetas_rad],
        "ring_offset": int(ring),
        "z_plane": int(zpl),
    }


def _params_from_run_metadata_json(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("antenna_optimized"):
        alphas = data.get("optimized_alphas")
        thetas_rad = data.get("optimized_thetas_rad")
        f0 = data.get("optimized_f0_Hz")
        gaps = data.get("optimized_quadrant_gaps")
    elif data.get("quadrant_fixed"):
        alphas = data.get("fixed_alphas")
        thetas_rad = data.get("fixed_thetas_rad")
        f0 = data.get("fixed_f0_Hz")
        gaps = data.get("fixed_quadrant_gaps")
    else:
        raise LoadOptimizedError(
            "metadata.json must be from an optimized or quadrant_fixed run "
            "(antenna_optimized or quadrant_fixed)."
        )
    if not alphas or len(alphas) != 4:
        raise LoadOptimizedError("metadata.json: missing quadrant alphas (length 4)")
    if not thetas_rad or len(thetas_rad) != 4:
        raise LoadOptimizedError("metadata.json: missing quadrant thetas_rad (length 4)")
    if f0 is None:
        raise LoadOptimizedError(
            "metadata.json: missing optimized_f0_Hz / fixed_f0_Hz"
        )
    ring: int | None = None
    zpl: int | None = None
    if gaps and len(gaps) >= 1:
        g0 = gaps[0]
        if isinstance(g0, (list, tuple)) and len(g0) >= 3:
            # Q1 gap (cx, ring_offset, z_plane) from build_quadrant_sources
            ring = int(g0[1])
            zpl = int(g0[2])
    if ring is None or zpl is None:
        raise LoadOptimizedError(
            "metadata.json: could not infer ring_offset / z_plane from quadrant gaps; "
            "use *_antenna_optimization.json instead."
        )
    out: dict[str, Any] = {
        "f0_hz": float(f0),
        "alphas": [float(x) for x in alphas],
        "thetas_rad": [float(x) for x in thetas_rad],
        "ring_offset": ring,
        "z_plane": zpl,
    }
    scale = data.get("opt_source_scale")
    if scale is not None:
        out["opt_source_scale"] = float(scale)
    ap = data.get("antenna_parameters")
    if isinstance(ap, dict):
        dip = ap.get("dipole_half_length_cells")
        if dip is not None:
            out["dipole_half_length_cells"] = int(dip)
    return out


def load_optimized_antenna_params(path: str | Path) -> dict[str, Any]:
    """
    Load replay parameters from a JSON file or from a directory containing one artifact.

    Returns a dict with f0_hz, alphas, thetas_rad, ring_offset, z_plane, and optionally
    opt_source_scale, dipole_half_length_cells, plus _source_path.
    """
    resolved = resolve_optimization_artifact_path(path)
    with open(resolved, encoding="utf-8") as f:
        data = json.load(f)

    fname = resolved.name
    if "antenna_optimization" in fname:
        params = _params_from_antenna_optimization_json(data)
    elif fname.endswith("_metadata.json") or (
        "output_base" in data
        and ("optimized_f0_Hz" in data or "fixed_f0_Hz" in data)
    ):
        params = _params_from_run_metadata_json(data)
    elif (
        isinstance(data.get("optimized"), dict)
        and "geometry" in data
        and "baseline" in data
    ):
        params = _params_from_antenna_optimization_json(data)
    else:
        raise LoadOptimizedError(
            f"Unrecognized JSON schema in {resolved}; expected "
            "*_antenna_optimization.json or *_metadata.json from an optimized run."
        )

    params["_source_path"] = str(resolved)
    return params


def apply_load_optimized_from_to_args(args: Any) -> None:
    """
    If ``args.load_optimized_from`` is set: disable optimization, enable quadrant-fixed,
    fill f0 / quadrant / geometry from file, and set replay_loaded_quadrant_geometry so the
    pipeline does not re-search ring offset (faithful replay).
    """
    path = getattr(args, "load_optimized_from", None)
    if not path:
        return

    params = load_optimized_antenna_params(path)
    args.optimize_antenna = False
    args.quadrant_fixed = True
    args.f0 = params["f0_hz"]
    args.fixed_quadrant_alphas = params["alphas"]
    args.fixed_quadrant_phases_deg = [math.degrees(t) for t in params["thetas_rad"]]
    args.fixed_quadrant_ring_offset = params["ring_offset"]
    args.fixed_quadrant_z_plane = params["z_plane"]
    if params.get("opt_source_scale") is not None:
        args.opt_source_scale = params["opt_source_scale"]
    if params.get("dipole_half_length_cells") is not None:
        args.fixed_quadrant_dipole_half_len = params["dipole_half_length_cells"]

    args.replay_loaded_quadrant_geometry = True
    args._loaded_optimized_source_path = params.get("_source_path")
