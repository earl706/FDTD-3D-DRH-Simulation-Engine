"""
Scalability analysis for FDTD benchmark results (Objective 5, task 3.4).

Loads scalability_benchmark_results.json, plots:
  (a) Runtime vs N³ (number_of_voxels)
  (b) Time per step vs N³ (and runtime vs N_t via caption)
  (c) Memory vs N³
Fits and comments on O(N³) and O(N_t) behavior. Saves plot images with matplotlib.

Usage:
  python plot_scalability_benchmark.py [path_to_scalability_benchmark_results.json]
  python plot_scalability_benchmark.py   # auto-finds latest results/data/scalability_benchmark_results.json
  python plot_scalability_benchmark.py results/081226-123456/data/scalability_benchmark_results.json

Output:
  Saves images in the same directory as the JSON (or current dir if path given):
  - scalability_runtime_vs_N3.png
  - scalability_time_per_step_vs_N3.png
  - scalability_memory_vs_N3.png
  - scalability_summary.png (all three in one figure)
"""

import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def find_latest_scalability_json():
    """Return path to most recent results/.../data/scalability_benchmark_results.json."""
    candidates = glob.glob(
        os.path.join("results", "*", "data", "scalability_benchmark_results.json")
    )
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    if not runs:
        raise ValueError(f"No 'runs' in {path}")
    return data, runs


def fit_power_law(x, y):
    """Fit y = a * x^b in log space; return (a, b), R²."""
    mask = (x > 0) & (y > 0)
    if np.sum(mask) < 2:
        return None, None, None
    logx = np.log(x[mask].astype(float))
    logy = np.log(y[mask].astype(float))
    (b, loga), res, _, _, _ = np.polyfit(logx, logy, 1, full=True)
    a = np.exp(loga)
    ss_res = res[0] if res.size else np.sum((logy - (loga + b * logx)) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return a, b, r2


def _benchmark_label(data):
    """Return 'FDTD+SAR+Temperature' or 'FDTD-only' for titles."""
    return (
        "FDTD+SAR+Temperature" if data.get("benchmark_full_pipeline") else "FDTD-only"
    )


def _format_voxels(n):
    """Format voxel count for axis labels (e.g. 512000 -> '512e3', 1e6 -> '1e6')."""
    n = int(round(float(n)))
    if n >= 1e9:
        return f"{n / 1e9:.1f}e9"
    if n >= 1e6:
        return f"{n / 1e6:.1f}e6"
    if n >= 1e3:
        return f"{n / 1e3:.0f}e3"
    return str(n)


def plot_runtime_vs_N3(runs, out_dir, data=None, suffix=""):
    data = data or {}
    n3 = np.array([r["number_of_voxels"] for r in runs], dtype=float)
    rt = np.array([r["total_wall_time_s"] for r in runs], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(n3, rt, s=60, color="C0", label="Benchmark runs", zorder=2)
    a, b, r2 = fit_power_law(n3, rt)
    if a is not None:
        n3_fit = np.linspace(n3.min(), n3.max(), 100)
        rt_fit = a * (n3_fit**b)
        ax.plot(
            n3_fit,
            rt_fit,
            "--",
            color="C1",
            label=rf"Fit: $t \propto N^{{{b:.2f}}}$ (R²={r2:.3f})",
        )
    ax.set_xlabel(r"Number of voxels $N^3$")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime vs grid size ({_benchmark_label(data)} benchmark)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_voxels(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scalability_runtime_vs_N3{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_per_step_vs_N3(runs, out_dir, data=None, suffix=""):
    data = data or {}
    n3 = np.array([r["number_of_voxels"] for r in runs], dtype=float)
    tps = np.array([r["time_per_step_ms"] for r in runs], dtype=float)
    valid = ~np.isnan(tps) & (tps > 0)
    n3, tps = n3[valid], tps[valid]
    if len(n3) == 0:
        n3 = np.array([r["number_of_voxels"] for r in runs], dtype=float)
        tps = np.array([r["time_per_step_ms"] for r in runs], dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(n3, tps, s=60, color="C0", label="Time per step", zorder=2)
    a, b, r2 = fit_power_law(n3, tps)
    if a is not None:
        n3_fit = np.linspace(n3.min(), n3.max(), 100)
        tps_fit = a * (n3_fit**b)
        ax.plot(
            n3_fit,
            tps_fit,
            "--",
            color="C1",
            label=rf"Fit: $\Delta t_{{\mathrm{{step}}}} \propto N^{{{b:.2f}}}$ (R²={r2:.3f})",
        )
    ax.set_xlabel(r"Number of voxels $N^3$")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title(
        f"Time per FDTD step vs grid size ({_benchmark_label(data)}, runtime ∝ N_t × this)"
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_voxels(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scalability_time_per_step_vs_N3{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_memory_vs_N3(runs, out_dir, data=None, suffix=""):
    data = data or {}
    n3 = np.array([r["number_of_voxels"] for r in runs], dtype=float)
    mem = np.array([r["peak_memory_MB"] for r in runs], dtype=float)
    valid = ~np.isnan(mem) & (mem > 0)
    if not np.any(valid):
        # e.g. Windows without resource module
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(
            0.5, 0.5, "No peak_memory_MB data (e.g. Windows)", ha="center", va="center"
        )
        path = os.path.join(out_dir, f"scalability_memory_vs_N3{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    n3 = n3[valid]
    mem = mem[valid]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(n3, mem, s=60, color="C0", label="Peak memory", zorder=2)
    a, b, r2 = fit_power_law(n3, mem)
    if a is not None:
        n3_fit = np.linspace(n3.min(), n3.max(), 100)
        mem_fit = a * (n3_fit**b)
        ax.plot(
            n3_fit,
            mem_fit,
            "--",
            color="C1",
            label=rf"Fit: memory $\propto N^{{{b:.2f}}}$ (R²={r2:.3f})",
        )
    ax.set_xlabel(r"Number of voxels $N^3$")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Peak memory vs grid size ({_benchmark_label(data)})")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_voxels(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"scalability_memory_vs_N3{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_runtime_breakdown(runs, out_dir):
    """Stacked bar: FDTD, SAR, Thermal runtime per grid size (full-pipeline only)."""
    n3 = np.array([r["number_of_voxels"] for r in runs], dtype=float)
    fdtd = np.array([r.get("time_fdtd_s") or 0 for r in runs], dtype=float)
    sar = np.array([r.get("time_sar_s") or 0 for r in runs], dtype=float)
    thermal = np.array([r.get("time_thermal_s") or 0 for r in runs], dtype=float)
    if np.all(fdtd == 0) and np.all(sar == 0) and np.all(thermal == 0):
        return None
    x = np.arange(len(n3))
    w = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar(x - w / 2, fdtd, w, label="FDTD", color="C0")
    ax.bar(x - w / 2, sar, w, bottom=fdtd, label="SAR", color="C1")
    ax.bar(x - w / 2, thermal, w, bottom=fdtd + sar, label="Thermal", color="C2")
    ax.set_xticks(x)
    labels = [_format_voxels(r["number_of_voxels"]) for r in runs]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(r"Grid size $N^3$ (voxels)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime breakdown: FDTD + SAR + Temperature (full pipeline)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(
        out_dir, "scalability_runtime_breakdown_FDTD_SAR_Temperature.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_summary(data, runs, out_dir, suffix=""):
    """Single figure with (a) runtime vs N³, (b) time per step vs N³, (c) memory vs N³."""
    n3 = np.array([r["number_of_voxels"] for r in runs], dtype=float)
    rt = np.array([r["total_wall_time_s"] for r in runs], dtype=float)
    tps = np.array([r["time_per_step_ms"] for r in runs], dtype=float)
    mem = np.array([r["peak_memory_MB"] for r in runs], dtype=float)
    n_t = data.get("benchmark_time_steps", runs[0].get("time_steps"))
    label = _benchmark_label(data)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # (a) Runtime vs N³
    ax = axes[0]
    ax.scatter(n3, rt, s=50, color="C0", zorder=2)
    a, b, r2 = fit_power_law(n3, rt)
    if a is not None:
        n3_fit = np.linspace(n3.min(), n3.max(), 100)
        ax.plot(
            n3_fit,
            a * (n3_fit**b),
            "--",
            color="C1",
            label=rf"$t \propto N^{{{b:.2f}}}$",
        )
    ax.set_xlabel(r"$N^3$ (voxels)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"(a) Runtime vs grid size ({label})")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_voxels(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Time per step vs N³ (runtime = N_t × time_per_step; so O(N_t) for runtime)
    ax = axes[1]
    ax.scatter(n3, tps, s=50, color="C0", zorder=2)
    a, b, r2 = fit_power_law(n3, tps)
    if a is not None:
        n3_fit = np.linspace(n3.min(), n3.max(), 100)
        ax.plot(
            n3_fit,
            a * (n3_fit**b),
            "--",
            color="C1",
            label=rf"$\Delta t_{{\mathrm{{step}}}} \propto N^{{{b:.2f}}}$",
        )
    ax.set_xlabel(r"$N^3$ (voxels)")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title(f"(b) Time per step vs N³ (runtime ∝ N_t={n_t})")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_voxels(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Memory vs N³
    ax = axes[2]
    valid = ~np.isnan(mem) & (mem > 0)
    if np.any(valid):
        ax.scatter(n3[valid], mem[valid], s=50, color="C0", zorder=2)
        a, b, r2 = fit_power_law(n3[valid], mem[valid])
        if a is not None:
            n3_fit = np.linspace(n3[valid].min(), n3[valid].max(), 100)
            ax.plot(
                n3_fit,
                a * (n3_fit**b),
                "--",
                color="C1",
                label=rf"mem $\propto N^{{{b:.2f}}}$",
            )
    else:
        ax.text(
            0.5, 0.5, "No memory data", ha="center", va="center", transform=ax.transAxes
        )
    ax.set_xlabel(r"$N^3$ (voxels)")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"(c) Memory vs grid size ({label})")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_voxels(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Scalability: O(N³) spatial, O(N_t) temporal ({label} benchmark)", fontsize=11
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f"scalability_summary{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
    else:
        path = find_latest_scalability_json()
        if path is None:
            print(
                "No scalability_benchmark_results.json found. Run the engine with --benchmark-grid-sizes first, or pass the JSON path.",
                file=sys.stderr,
            )
            sys.exit(1)
    out_dir = os.path.dirname(os.path.abspath(path))
    data, runs = load_results(path)
    print(f"Loaded {len(runs)} runs from {path}")
    print(f"Output directory: {out_dir}")

    paths = []
    paths.append(plot_runtime_vs_N3(runs, out_dir, data))
    paths.append(plot_time_per_step_vs_N3(runs, out_dir, data))
    paths.append(plot_memory_vs_N3(runs, out_dir, data))
    paths.append(plot_summary(data, runs, out_dir))
    if data.get("benchmark_full_pipeline"):
        p = plot_runtime_breakdown(runs, out_dir)
        if p:
            paths.append(p)
        # FDTD-only scalability summary (same metrics, runtime = time_fdtd_s)
        runs_fdtd = [
            {
                **r,
                "total_wall_time_s": r.get("time_fdtd_s") or r.get("total_wall_time_s"),
            }
            for r in runs
        ]
        data_fdtd = {**data, "benchmark_full_pipeline": False}
        paths.append(
            plot_runtime_vs_N3(runs_fdtd, out_dir, data_fdtd, suffix="_FDTD_only")
        )
        paths.append(
            plot_time_per_step_vs_N3(runs, out_dir, data_fdtd, suffix="_FDTD_only")
        )
        paths.append(plot_memory_vs_N3(runs, out_dir, data_fdtd, suffix="_FDTD_only"))
        paths.append(plot_summary(data_fdtd, runs_fdtd, out_dir, suffix="_FDTD_only"))

    for p in paths:
        print(f"  Saved {p}")
    print("Done.")


if __name__ == "__main__":
    main()
