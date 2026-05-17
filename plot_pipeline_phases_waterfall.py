#!/usr/bin/env python3
"""
Stacked horizontal bar chart of pipeline phase times from {base}_performance.json (phases_s).

Usage:
  python plot_pipeline_phases_waterfall.py --performance-json PATH/to/*_performance.json [--out PATH]

Suitable for thesis figures: segmentation, setup, FDTD, SAR, thermal, saving, animations, etc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ordered_phases(phases: dict) -> list[tuple[str, float]]:
    """Stable order for thesis readability."""
    preferred = [
        "segmentation",
        "setup",
        "antenna_optimization",
        "fdtd_simulation",
        "sar_computation",
        "thermal_solver",
        "saving_data",
        "animations",
    ]
    out = []
    for k in preferred:
        if k in phases and phases[k] is not None:
            try:
                v = float(phases[k])
                if v > 0:
                    out.append((k.replace("_", " ").title(), v))
            except (TypeError, ValueError):
                continue
    for k, v in phases.items():
        if k in preferred or v is None:
            continue
        try:
            vf = float(v)
            if vf > 0:
                label = k.replace("_", " ").title()
                if (label, vf) not in out and not any(x[0] == label for x in out):
                    out.append((label, vf))
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--performance-json",
        type=Path,
        required=True,
        help="Path to *_performance.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: same dir as JSON, pipeline_phases_waterfall.png)",
    )
    args = ap.parse_args()
    perf_path = args.performance_json
    if not perf_path.is_file():
        raise SystemExit(f"Not found: {perf_path}")
    with open(perf_path, encoding="utf-8") as f:
        perf = json.load(f)
    phases = perf.get("phases_s") or {}
    rows = _ordered_phases(phases)
    if not rows:
        raise SystemExit("No positive phases_s entries in performance JSON")

    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows], dtype=float)
    total = float(np.sum(values))
    if total <= 0:
        raise SystemExit("Phase sum is zero")

    out_path = args.out
    if out_path is None:
        out_path = perf_path.parent / "pipeline_phases_waterfall.png"

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.35 * len(labels))), dpi=150)
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(labels)))
    left = 0.0
    for i, (lab, sec) in enumerate(rows):
        ax.barh(
            0,
            sec,
            left=left,
            height=0.65,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
        )
        mid = left + sec / 2
        pct = 100.0 * sec / total
        ax.text(
            mid,
            0,
            f"{lab}\n{sec:.2f}s ({pct:.1f}%)",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            fontweight="medium",
        )
        left += sec

    ax.set_yticks([])
    ax.set_xlabel("Cumulative time (seconds)")
    ax.set_title(
        f"Pipeline phase time breakdown (total {total:.2f} s)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, left * 1.02)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
