"""
Benchmark workflow: scalability runs (full pipeline or FDTD-only) per grid size.

Writes scalability_benchmark_results.json and optionally runs plot_scalability_benchmark.py.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import nibabel as nib
from scipy import ndimage

from hermes_drh.cli.parser import BENCHMARK_GRID_SIZES_DEFAULT, BENCHMARK_TIME_STEPS_DEFAULT
from hermes_drh.segmentation.loader import load_segmentation_for_benchmark
from hermes_drh.compat.fdtd_solver import _run_minimal_fdtd_benchmark


def _cleanup_grid_frame_dirs(grid_data_dir):
    """
    Remove per-grid frame stacks that are not needed for scalability benchmarking.
    Keeps metadata/performance JSON and aggregate benchmark outputs.
    """
    frame_dirs = ("E_frames", "SAR_frames", "Temperature_frames")
    removed = []
    for name in frame_dirs:
        d = os.path.join(grid_data_dir, name)
        if not os.path.isdir(d):
            continue
        try:
            shutil.rmtree(d)
            removed.append(name)
        except OSError as e:
            print(f"    [WARN] Could not remove {d}: {e}")
    if removed:
        print(
            f"    Cleaned frame directories for benchmark data retention: {', '.join(removed)}"
        )


def run_benchmark(args, paths, write_progress_cb=None):
    """
    Run scalability benchmark: one run per grid size (full pipeline or FDTD-only).

    paths: object with results_dir, data_dir (and optionally other dirs).
    Caller must create dirs. BENCHMARK_RESULTS_DIR and BENCHMARK_GRID_SIZE are set for
    child engine runs.
    """
    results_dir = paths.results_dir
    data_dir = paths.data_dir

    # Benchmark runs skip animations (subprocess gets --skip-animations; no animations in minimal FDTD path)
    setattr(args, "skip_animations", True)

    if getattr(args, "benchmark_grid_sizes_range", None) is not None:
        a, b, s = args.benchmark_grid_sizes_range
        sizes = list(range(a, b + 1, s))
        if not sizes:
            sizes = BENCHMARK_GRID_SIZES_DEFAULT
    elif args.benchmark_grid_sizes is not None and len(args.benchmark_grid_sizes) > 0:
        sizes = args.benchmark_grid_sizes
    else:
        sizes = BENCHMARK_GRID_SIZES_DEFAULT

    steps = getattr(args, "benchmark_time_steps", BENCHMARK_TIME_STEPS_DEFAULT)
    has_anatomy = (
        (args.seg and os.path.isfile(args.seg))
        or (args.modalities is not None)
        or (args.modalities_dir is not None)
    )

    if has_anatomy:
        print(
            f"\nBenchmark mode (full pipeline with anatomy): grid sizes {sizes}, "
            f"{steps} time steps each."
        )
    else:
        print(
            f"\nBenchmark mode (FDTD-only, no anatomy): grid sizes {sizes}, "
            f"{steps} time steps each."
        )

    results = []
    engine_module = "hermes_drh.cli.main"

    if has_anatomy:
        labels_3d = load_segmentation_for_benchmark(args)
        sx, sy, sz = labels_3d.shape
        for N in sizes:
            print(f"  Running full pipeline N={N} ({N**3} voxels)...")
            zoom_factors = (N / sx, N / sy, N / sz)
            labels_n = ndimage.zoom(
                labels_3d.astype(np.float32),
                zoom_factors,
                order=0,
                mode="nearest",
            )
            labels_n = np.round(labels_n).astype(np.int32)
            labels_n = np.clip(labels_n, 0, 4)
            base_name = f"benchmark_anatomy_{N}"
            fd, temp_nii = tempfile.mkstemp(suffix=".nii.gz", prefix=base_name + "_")
            os.close(fd)
            try:
                affine = np.diag([1.0, 1.0, 1.0, 1.0])
                nib.save(
                    nib.Nifti1Image(labels_n.astype(np.int32), affine),
                    temp_nii,
                )
                cmd = [
                    sys.executable,
                    "-m",
                    engine_module,
                    temp_nii,
                    "--max-dim",
                    str(N),
                    "--time-steps",
                    str(steps),
                    "--skip-animations",
                ]
                t_before = time.time()
                subprocess_env = os.environ.copy()
                subprocess_env["BENCHMARK_RESULTS_DIR"] = os.path.abspath(results_dir)
                subprocess_env["BENCHMARK_GRID_SIZE"] = str(N)
                subprocess.run(cmd, check=True, env=subprocess_env)
            finally:
                try:
                    os.remove(temp_nii)
                except OSError:
                    pass
            perf_files = glob.glob(os.path.join(data_dir, str(N), "*_performance.json"))
            perf_files = [p for p in perf_files if os.path.getmtime(p) >= t_before - 5]
            perf_files.sort(key=os.path.getmtime, reverse=True)
            if perf_files:
                with open(perf_files[0], "r") as f:
                    pm = json.load(f)
                phases = pm.get("phases_s") or {}
                run_metrics = {
                    "grid_shape": pm["grid_shape"],
                    "number_of_voxels": pm["number_of_voxels"],
                    "time_steps": pm["time_steps"],
                    "total_wall_time_s": round(
                        pm.get("total_simulation_time_s")
                        or pm.get("total_wall_time_s")
                        or 0,
                        6,
                    ),
                    "time_per_step_ms": pm.get("time_per_step_ms"),
                    "peak_memory_MB": pm.get("peak_memory_MB"),
                    "time_fdtd_s": phases.get("fdtd_simulation"),
                    "time_sar_s": phases.get("sar_computation"),
                    "time_thermal_s": phases.get("thermal_solver"),
                }
                results.append(run_metrics)
                print(
                    f"    wall_time_s={run_metrics['total_wall_time_s']:.3f}, "
                    f"time_per_step_ms={run_metrics.get('time_per_step_ms')}, "
                    f"peak_memory_MB={run_metrics.get('peak_memory_MB')}"
                )
            else:
                print(f"    [WARN] No performance JSON found for N={N}")
            _cleanup_grid_frame_dirs(os.path.join(data_dir, str(N)))
    else:
        for N in sizes:
            print(f"  Running N={N} ({N**3} voxels)...")
            run_metrics = _run_minimal_fdtd_benchmark(
                N,
                N,
                N,
                steps,
                dx_mm=getattr(args, "dx_mm", 10.0),
                courant_factor=getattr(args, "courant_factor", 0.99),
            )
            results.append(run_metrics)
            print(
                f"    wall_time_s={run_metrics['total_wall_time_s']:.3f}, "
                f"time_per_step_ms={run_metrics['time_per_step_ms']:.4f}, "
                f"peak_memory_MB={run_metrics['peak_memory_MB']}"
            )

    scalability_path = os.path.join(data_dir, "scalability_benchmark_results.json")
    out = {
        "benchmark_time_steps": steps,
        "benchmark_full_pipeline": has_anatomy,
        "backend": "numpy_numba",
        "runs": results,
    }
    with open(scalability_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nScalability results written to {scalability_path}")

    print(
        "  [INFO] Scalability plot is thesis-only (plot_scalability_benchmark.py); "
        f"JSON written to {scalability_path}"
    )
