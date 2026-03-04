#!/usr/bin/env python3
"""
Build animations from streamed E_frames and SAR_frames saved by fdtd_breast_simulation_engine.py.

Usage:
  python build_breast_animations_from_streamed_frames.py --results-dir results/breast_fdtd_240226-120000
  python build_breast_animations_from_streamed_frames.py --results-dir results/breast_fdtd_240226-120000 --subsample 5 --fps 30

Loads frame data from data/E_frames/ and data/SAR_frames/, segmentation from data/{output_base}_segmentation.npy.
Writes 2D combined (E-field, SAR) animation to results_dir/animations/.
"""

import argparse
import json
import os
import sys

import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

DEFAULT_CHUNK_SIZE = 20


def parse_args():
    p = argparse.ArgumentParser(
        description="Build breast FDTD animations from streamed E and SAR frames."
    )
    p.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to results directory (e.g. results/breast_fdtd_240226-120000).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: results_dir/data).",
    )
    p.add_argument(
        "--animations-dir",
        default=None,
        help="Output directory for MP4 (default: results_dir/animations).",
    )
    p.add_argument(
        "--output-base",
        default=None,
        help="Override output base name (default: from E_frames_part0.npz).",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for MP4 (default: 30).",
    )
    return p.parse_args()


def find_metadata_and_output_base(data_dir, output_base_arg):
    if output_base_arg:
        meta_path = os.path.join(data_dir, f"{output_base_arg}_metadata.json")
        if os.path.isfile(meta_path):
            return meta_path, output_base_arg
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    e_dir = os.path.join(data_dir, "E_frames")
    if not os.path.isdir(e_dir):
        raise FileNotFoundError(f"E_frames directory not found: {e_dir}")
    for f in sorted(os.listdir(e_dir)):
        if f.endswith("_E_frames_part0.npz"):
            output_base = f.replace("_E_frames_part0.npz", "")
            meta_path = os.path.join(data_dir, f"{output_base}_metadata.json")
            if os.path.isfile(meta_path):
                return meta_path, output_base
            break
    raise FileNotFoundError(
        f"No metadata or E_frames_part0.npz in {e_dir}. Use --output-base."
    )


def load_metadata(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)


def load_segmentation(data_dir, output_base):
    seg_path = os.path.join(data_dir, f"{output_base}_segmentation.npy")
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    labels_3d = np.load(seg_path).astype(np.int32)
    return labels_3d


def tumor_footprint_and_contour(labels_3d):
    """Tumor = label 1. Max projection and contour for overlay."""
    tumor_footprint_2d = np.max((labels_3d == 1).astype(np.float32), axis=2)
    fig_c = plt.figure()
    ax_c = fig_c.add_subplot(111)
    cs = ax_c.contour(tumor_footprint_2d, levels=[0.5], origin="lower")
    tumor_contour_segments = cs.allsegs[0] if len(cs.allsegs) > 0 else []
    plt.close(fig_c)
    return tumor_footprint_2d, tumor_contour_segments


class FrameLoader:
    """Load E and SAR frames by index from part files."""

    def __init__(self, data_dir, output_base, chunk_size, n_frames):
        self.data_dir = data_dir
        self.output_base = output_base
        self.chunk_size = chunk_size
        self.n_frames = n_frames
        self.e_dir = os.path.join(data_dir, "E_frames")
        self.sar_dir = os.path.join(data_dir, "SAR_frames")
        self._e_part = self._sar_part = None
        self._e_part_idx = self._sar_part_idx = -1

    def _load_part(self, kind, part_idx):
        if kind == "E":
            path = os.path.join(self.e_dir, f"{self.output_base}_E_frames_part{part_idx}.npz")
        else:
            path = os.path.join(self.sar_dir, f"{self.output_base}_SAR_frames_part{part_idx}.npz")
        with np.load(path) as z:
            key = "E_frames" if kind == "E" else "SAR_frames"
            return z[key][:]

    def get_frame(self, frame_idx):
        part = frame_idx // self.chunk_size
        idx = frame_idx % self.chunk_size
        if self._e_part_idx != part:
            self._e_part = self._load_part("E", part)
            self._e_part_idx = part
        if self._sar_part_idx != part:
            self._sar_part = self._load_part("SAR", part)
            self._sar_part_idx = part
        e = np.abs(self._e_part[idx])
        sar = self._sar_part[idx]
        return e, sar


def compute_global_limits(data_dir, output_base, n_frames, chunk_size):
    e_max, sar_max = 0.0, 0.0
    n_parts = (n_frames + chunk_size - 1) // chunk_size
    e_dir = os.path.join(data_dir, "E_frames")
    sar_dir = os.path.join(data_dir, "SAR_frames")
    for part in range(n_parts):
        with np.load(os.path.join(e_dir, f"{output_base}_E_frames_part{part}.npz")) as z:
            e_max = max(e_max, float(np.max(np.abs(z["E_frames"]))))
        with np.load(os.path.join(sar_dir, f"{output_base}_SAR_frames_part{part}.npz")) as z:
            sar_max = max(sar_max, float(np.max(z["SAR_frames"])))
    if sar_max <= 0:
        sar_max = 1.0
    return e_max, sar_max


def main():
    args = parse_args()
    if not args.results_dir and not args.data_dir:
        print("Error: provide results_dir or --data-dir", file=sys.stderr)
        sys.exit(1)
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
        results_dir = os.path.dirname(data_dir)
    else:
        results_dir = os.path.abspath(args.results_dir)
        data_dir = os.path.join(results_dir, "data")
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        meta_path, output_base = find_metadata_and_output_base(data_dir, args.output_base)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    meta = load_metadata(meta_path)
    if args.output_base:
        output_base = args.output_base
    n_frames = meta.get("n_frames", 0)
    if n_frames <= 0:
        print("Error: n_frames is 0 or missing in metadata.", file=sys.stderr)
        sys.exit(1)
    grid_shape = meta["grid_shape"]
    chunk_size = meta.get("E_frames_chunk_size", DEFAULT_CHUNK_SIZE)
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2]

    print("Loading segmentation and building tumor contour...")
    labels_3d = load_segmentation(data_dir, output_base)
    tumor_footprint_2d, tumor_contour_segments = tumor_footprint_and_contour(labels_3d)

    print(f"Computing global limits over {n_frames} frames...")
    e_max, sar_max = compute_global_limits(data_dir, output_base, n_frames, chunk_size)

    frame_indices = list(range(0, n_frames, args.subsample))
    n_anim_frames = len(frame_indices)
    print(f"Animation: {n_anim_frames} frames (subsample={args.subsample})")

    loader = FrameLoader(data_dir, output_base, chunk_size, n_frames)
    animations_dir = args.animations_dir or os.path.join(results_dir, "animations")
    os.makedirs(animations_dir, exist_ok=True)
    print(f"Output: {animations_dir}")

    # 2D combined: E-field max proj, SAR max proj
    fig_anim = plt.figure(figsize=(14, 6))
    ax_e = fig_anim.add_subplot(1, 2, 1)
    ax_sar = fig_anim.add_subplot(1, 2, 2)

    sm_e = ScalarMappable(norm=Normalize(0, e_max), cmap="jet")
    sm_e.set_array([])
    div_e = make_axes_locatable(ax_e)
    cax_e = div_e.append_axes("right", size="6%", pad=0.15)
    fig_anim.colorbar(sm_e, cax=cax_e, label="|E| (a.u.)")

    sm_sar = ScalarMappable(norm=Normalize(0, sar_max), cmap="coolwarm")
    sm_sar.set_array([])
    div_sar = make_axes_locatable(ax_sar)
    cax_sar = div_sar.append_axes("right", size="6%", pad=0.15)
    fig_anim.colorbar(sm_sar, cax=cax_sar, label="SAR (W/kg)")

    fig_anim.tight_layout()

    def update_2d(anim_frame_idx):
        frame_idx = frame_indices[anim_frame_idx]
        e_data, sar_data = loader.get_frame(frame_idx)
        e_proj = np.max(np.abs(e_data), axis=2)
        sar_proj = np.max(sar_data, axis=2)
        ax_e.clear()
        ax_e.imshow(e_proj, cmap="jet", origin="lower", vmin=0, vmax=e_max)
        ax_e.set_title(f"E-field max proj. — Frame {frame_idx + 1}/{n_frames}")
        ax_e.set_xlabel("Y"); ax_e.set_ylabel("X")
        ax_e.contour(tumor_footprint_2d, levels=[0.5], colors=["lime"], linewidths=1.5, origin="lower")
        ax_sar.clear()
        ax_sar.imshow(sar_proj, cmap="coolwarm", origin="lower", vmin=0, vmax=sar_max)
        ax_sar.set_title(f"SAR max proj. — Frame {frame_idx + 1}/{n_frames}")
        ax_sar.set_xlabel("Y"); ax_sar.set_ylabel("X")
        ax_sar.contour(tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1.5, origin="lower")

    ani_2d = animation.FuncAnimation(
        fig_anim, update_2d, frames=n_anim_frames, interval=20, blit=False, repeat=True, repeat_delay=1000
    )
    out_path = os.path.join(animations_dir, f"{output_base}_efield_sar_2d.mp4")
    print(f"Saving 2D animation to {out_path}...")
    ani_2d.save(out_path, writer="ffmpeg", fps=args.fps)
    plt.close(fig_anim)
    print("Done.")


if __name__ == "__main__":
    main()
