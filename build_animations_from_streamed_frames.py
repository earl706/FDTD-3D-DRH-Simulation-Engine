#!/usr/bin/env python3
"""
Build complete animations from streamed E_frames, SAR_frames, and Temperature_frames
saved by fdtd_brain_simulation_engine.py when run with --stream-frames.

Usage:
  python build_animations_from_streamed_frames.py --results-dir results/210226-142342
  python build_animations_from_streamed_frames.py --results-dir results/210226-142342 --subsample 10 --skip-3d

Loads frame data in chunks from disk (no full load into RAM). Writes 2D combined
(E-field, SAR, Temperature) and optional 3D isometric animations to results_dir/animations/.
"""

import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt, animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Default chunk size (must match STREAM_CHUNK_SIZE / E_FRAMES_CHUNK_SIZE in engine)
DEFAULT_CHUNK_SIZE = 20
T_BOUNDARY_CELSIUS = 37.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Build animations from streamed FDTD frames (E, SAR, Temperature)."
    )
    p.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Path to results directory (e.g. results/210226-142342). Must contain data/ with metadata and frame parts.",
    )
    p.add_argument(
        "animations_dir",
        nargs="?",
        default=None,
        help="Path to animations output directory (default: results_dir/animations). Pass same as simulation's animations dir to store there.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override: path to data directory (default: results_dir/data).",
    )
    p.add_argument(
        "--output-base",
        default=None,
        help="Override output base name (default: from metadata).",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Use every Nth frame for animation (default: 1 = all frames).",
    )
    p.add_argument(
        "--skip-3d",
        action="store_true",
        help="Only build 2D combined animation; skip 3D isometric animations.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for output video (default: 60).",
    )
    p.add_argument(
        "--generate-slice-timestep-images",
        action="store_true",
        help="Generate one combined PNG (E-field, SAR, Temperature) per (slice, timestep) for dashboard viewer.",
    )
    p.add_argument(
        "--slice-timestep-images-dir",
        default=None,
        help="Output directory for slice-timestep images (default: data_dir/slice_timestep_images).",
    )
    return p.parse_args()


def find_metadata_and_output_base(data_dir, output_base_arg):
    """Resolve metadata path and output_base from data_dir."""
    if output_base_arg:
        meta_path = os.path.join(data_dir, f"{output_base_arg}_metadata.json")
        if os.path.isfile(meta_path):
            return meta_path, output_base_arg
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    # Infer from first E_frames part file
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
        f"No metadata or E_frames_part0.npz found in {e_dir}. Specify --output-base."
    )


def load_metadata(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)


def load_segmentation(data_dir, output_base):
    seg_path = os.path.join(data_dir, f"{output_base}_segmentation.nii.gz")
    if not os.path.isfile(seg_path):
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")
    img = nib.load(seg_path)
    labels_3d = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
    labels_3d = np.round(labels_3d).astype(np.int32)
    return labels_3d


def tumor_footprint_and_contour(labels_3d):
    tumor_footprint_2d = np.max((labels_3d >= 1).astype(np.float32), axis=2)
    fig_c = plt.figure()
    ax_c = fig_c.add_subplot(111)
    cs = ax_c.contour(tumor_footprint_2d, levels=[0.5], origin="lower")
    tumor_contour_segments = cs.allsegs[0] if len(cs.allsegs) > 0 else []
    plt.close(fig_c)
    return tumor_footprint_2d, tumor_contour_segments


class FrameLoader:
    """Load E, SAR, Temperature frames by index from part files (with part caching)."""

    def __init__(self, data_dir, output_base, chunk_size, n_frames):
        self.data_dir = data_dir
        self.output_base = output_base
        self.chunk_size = chunk_size
        self.n_frames = n_frames
        self.e_dir = os.path.join(data_dir, "E_frames")
        self.sar_dir = os.path.join(data_dir, "SAR_frames")
        self.t_dir = os.path.join(data_dir, "Temperature_frames")
        self._e_part = self._sar_part = self._t_part = None
        self._e_part_idx = self._sar_part_idx = self._t_part_idx = -1

    def _load_part(self, kind, part_idx):
        if kind == "E":
            path = os.path.join(
                self.e_dir, f"{self.output_base}_E_frames_part{part_idx}.npz"
            )
        elif kind == "SAR":
            path = os.path.join(
                self.sar_dir, f"{self.output_base}_SAR_frames_part{part_idx}.npz"
            )
        else:
            path = os.path.join(
                self.t_dir, f"{self.output_base}_Temperature_frames_part{part_idx}.npz"
            )
        with np.load(path) as z:
            key = (
                "E_frames"
                if kind == "E"
                else "SAR_frames" if kind == "SAR" else "Temperature_frames"
            )
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
        if self._t_part_idx != part:
            self._t_part = self._load_part("T", part)
            self._t_part_idx = part
        e = np.abs(self._e_part[idx])
        sar = self._sar_part[idx]
        temp = self._t_part[idx]
        return e, sar, temp


def compute_global_limits(data_dir, output_base, n_frames, chunk_size, has_temperature):
    """One pass over part files to get e_max, sar_max, temp_min, temp_max."""
    e_max = 0.0
    sar_max = 0.0
    temp_min = T_BOUNDARY_CELSIUS
    temp_max = T_BOUNDARY_CELSIUS + 1.0
    n_parts = (n_frames + chunk_size - 1) // chunk_size
    e_dir = os.path.join(data_dir, "E_frames")
    sar_dir = os.path.join(data_dir, "SAR_frames")
    t_dir = os.path.join(data_dir, "Temperature_frames")

    for part in range(n_parts):
        with np.load(
            os.path.join(e_dir, f"{output_base}_E_frames_part{part}.npz")
        ) as z:
            e_chunk = z["E_frames"]
        e_max = max(e_max, float(np.max(np.abs(e_chunk))))
        with np.load(
            os.path.join(sar_dir, f"{output_base}_SAR_frames_part{part}.npz")
        ) as z:
            sar_chunk = z["SAR_frames"]
        sar_max = max(sar_max, float(np.max(sar_chunk)))
        if has_temperature:
            with np.load(
                os.path.join(t_dir, f"{output_base}_Temperature_frames_part{part}.npz")
            ) as z:
                t_chunk = z["Temperature_frames"]
            temp_min = min(temp_min, float(np.min(t_chunk)))
            temp_max = max(temp_max, float(np.max(t_chunk)))
    if sar_max <= 0:
        sar_max = 1.0
    return e_max, sar_max, temp_min, temp_max


def generate_slice_timestep_images(
    data_dir,
    output_base,
    output_dir,
    loader,
    labels_3d,
    n_frames,
    nz,
    e_max,
    sar_max,
    temp_min,
    temp_max,
    has_temperature,
):
    """Generate one PNG per (frame, slice): 3 panels (E-field, SAR, Temperature) axial slice."""
    os.makedirs(output_dir, exist_ok=True)
    n_cols = 3 if has_temperature else 2
    total = n_frames * nz
    done = 0
    for frame_idx in range(n_frames):
        e_data, sar_data, temp_data = loader.get_frame(frame_idx)
        for slice_idx in range(nz):
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            if n_cols == 2:
                ax_e, ax_sar = axes
                ax_temp = None
            else:
                ax_e, ax_sar, ax_temp = axes
            # E-field axial slice (x, y at z=slice_idx)
            e_slice = np.abs(e_data[:, :, slice_idx])
            ax_e.imshow(e_slice, cmap="jet", origin="lower", vmin=0, vmax=e_max)
            ax_e.set_title(f"E-field (z={slice_idx})")
            ax_e.set_xlabel("Y")
            ax_e.set_ylabel("X")
            # SAR slice
            sar_slice = sar_data[:, :, slice_idx]
            ax_sar.imshow(
                sar_slice, cmap="coolwarm", origin="lower", vmin=0, vmax=sar_max
            )
            ax_sar.set_title(f"SAR (z={slice_idx})")
            ax_sar.set_xlabel("Y")
            ax_sar.set_ylabel("X")
            if has_temperature and ax_temp is not None:
                temp_slice = temp_data[:, :, slice_idx]
                ax_temp.imshow(
                    temp_slice,
                    cmap="coolwarm",
                    origin="lower",
                    vmin=temp_min,
                    vmax=temp_max,
                )
                ax_temp.set_title(f"Temperature (z={slice_idx})")
                ax_temp.set_xlabel("Y")
                ax_temp.set_ylabel("X")
            fig.suptitle(f"Frame {frame_idx}  Slice z={slice_idx}", fontsize=10)
            fig.tight_layout()
            out_path = os.path.join(
                output_dir,
                f"frame_{frame_idx:05d}_slice_{slice_idx:03d}.png",
            )
            fig.savefig(out_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  Slice-timestep images: {done}/{total}")
    print(f"  Wrote {total} images to {output_dir}")


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
        meta_path, output_base = find_metadata_and_output_base(
            data_dir, args.output_base
        )
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
    T_boundary = meta.get("T_boundary_C", T_BOUNDARY_CELSIUS)
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2]

    # Check Temperature_frames exist
    t_dir = os.path.join(data_dir, "Temperature_frames")
    has_temperature = os.path.isfile(
        os.path.join(t_dir, f"{output_base}_Temperature_frames_part0.npz")
    )

    print(f"Loading segmentation and building tumor contour...")
    labels_3d = load_segmentation(data_dir, output_base)
    tumor_footprint_2d, tumor_contour_segments = tumor_footprint_and_contour(labels_3d)
    # Z-index of tumor centroid for 2D combined animation only (slice at tumor plane)
    tumor_mask_2d = (labels_3d >= 1) & (labels_3d <= 3)
    if np.any(tumor_mask_2d):
        tz = np.argwhere(tumor_mask_2d)[:, 2]
        tumor_centroid_z_2d = int(round(np.mean(tz)))
        tumor_centroid_z_2d = max(0, min(tumor_centroid_z_2d, nz - 1))
    else:
        tumor_centroid_z_2d = nz // 2
    print(
        f"  2D combined animation uses slice at z={tumor_centroid_z_2d} (tumor centroid)."
    )

    print(f"Computing global min/max over {n_frames} frames...")
    e_max, sar_max, temp_min, temp_max = compute_global_limits(
        data_dir, output_base, n_frames, chunk_size, has_temperature
    )

    frame_indices = list(range(0, n_frames, args.subsample))
    n_anim_frames = len(frame_indices)
    print(f"Animation: {n_anim_frames} frames (subsample={args.subsample})")

    loader = FrameLoader(data_dir, output_base, chunk_size, n_frames)
    # Use explicit animations_dir if provided (e.g. by engine spawn), else results_dir/animations
    animations_dir = (
        os.path.abspath(args.animations_dir)
        if args.animations_dir
        else os.path.join(results_dir, "animations")
    )
    os.makedirs(animations_dir, exist_ok=True)
    print(f"Animations output: {animations_dir}")

    if args.generate_slice_timestep_images:
        slice_timestep_dir = (
            os.path.abspath(args.slice_timestep_images_dir)
            if args.slice_timestep_images_dir
            else os.path.join(data_dir, "slice_timestep_images")
        )
        print("Generating slice-timestep images (E, SAR, T per frame per slice)...")
        generate_slice_timestep_images(
            data_dir=data_dir,
            output_base=output_base,
            output_dir=slice_timestep_dir,
            loader=loader,
            labels_3d=labels_3d,
            n_frames=n_frames,
            nz=nz,
            e_max=e_max,
            sar_max=sar_max,
            temp_min=temp_min,
            temp_max=temp_max,
            has_temperature=has_temperature,
        )

    # ---- 2D combined animation (E, SAR, Temperature) ----
    n_cols = 3 if has_temperature else 2
    fig_anim = plt.figure(figsize=(20, 10))
    ax_e = fig_anim.add_subplot(1, n_cols, 1)
    ax_sar = fig_anim.add_subplot(1, n_cols, 2)
    if has_temperature:
        ax_temp = fig_anim.add_subplot(1, n_cols, 3)

    # Colorbars for magnitude (fixed scale; explicit cax so they appear in saved video)
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

    if has_temperature:
        sm_temp = ScalarMappable(norm=Normalize(temp_min, temp_max), cmap="coolwarm")
        sm_temp.set_array([])
        div_temp = make_axes_locatable(ax_temp)
        cax_temp = div_temp.append_axes("right", size="6%", pad=0.15)
        fig_anim.colorbar(sm_temp, cax=cax_temp, label="Temperature (°C)")

    fig_anim.tight_layout()

    def update_2d(anim_frame_idx):
        frame_idx = frame_indices[anim_frame_idx]
        e_data, sar_data, temp_data = loader.get_frame(frame_idx)
        # 2D combined: max-projection over z (2D overview)
        e_proj = np.max(np.abs(e_data), axis=2)
        sar_proj = np.max(sar_data, axis=2)
        ax_e.clear()
        ax_e.imshow(e_proj, cmap="jet", origin="lower", vmin=0, vmax=e_max)
        ax_e.set_title(f"E-field (Ez) max proj. (z) - Frame {frame_idx + 1}/{n_frames}")
        ax_e.set_xlabel("Y (cells)")
        ax_e.set_ylabel("X (cells)")
        ax_e.contour(
            tumor_footprint_2d,
            levels=[0.5],
            colors=["lime"],
            linewidths=1.5,
            origin="lower",
        )
        ax_sar.clear()
        ax_sar.imshow(sar_proj, cmap="coolwarm", origin="lower", vmin=0, vmax=sar_max)
        ax_sar.set_title(f"SAR max proj. (z) - Frame {frame_idx + 1}/{n_frames}")
        ax_sar.set_xlabel("Y (cells)")
        ax_sar.set_ylabel("X (cells)")
        ax_sar.contour(
            tumor_footprint_2d,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1.5,
            origin="lower",
        )
        if has_temperature:
            temp_proj = np.max(temp_data, axis=2)
            ax_temp.clear()
            ax_temp.imshow(
                temp_proj, cmap="coolwarm", origin="lower", vmin=temp_min, vmax=temp_max
            )
            ax_temp.set_title(
                f"Temperature max proj. (z) - Frame {frame_idx + 1}/{n_frames}"
            )
            ax_temp.set_xlabel("Y (cells)")
            ax_temp.set_ylabel("X (cells)")
            ax_temp.contour(
                tumor_footprint_2d,
                levels=[0.5],
                colors=["cyan"],
                linewidths=1.5,
                origin="lower",
            )

    ani_2d = animation.FuncAnimation(
        fig_anim,
        update_2d,
        frames=n_anim_frames,
        interval=20,
        blit=False,
        repeat=True,
        repeat_delay=1000,
    )
    plt.tight_layout()
    out_name = (
        f"{output_base}_efield_sar_temp_2d.mp4"
        if has_temperature
        else f"{output_base}_sar_2d.mp4"
    )
    out_path = os.path.join(animations_dir, out_name)
    print(f"Saving 2D animation to {out_path}...")
    ani_2d.save(out_path, writer="ffmpeg", fps=args.fps)
    plt.close(fig_anim)
    print("✓ 2D animation saved")

    if args.skip_3d:
        print("Skipping 3D animations (--skip-3d).")
        return

    # ---- 3D isometric animations (E, SAR, optional Temperature) ----
    step_3d = 2
    x_coords = np.arange(0, nx, step_3d)
    y_coords = np.arange(0, ny, step_3d)
    X_3d, Y_3d = np.meshgrid(y_coords, x_coords)
    x_min, x_max = 0, ny
    y_min, y_max = 0, nx

    def make_update_3d_e(ax_3d_e):
        def update(frame_num):
            ax_3d_e.clear()
            anim_idx = frame_num
            frame_idx = frame_indices[anim_idx]
            e_data, _, _ = loader.get_frame(frame_idx)
            e_proj = np.max(np.abs(e_data), axis=2)
            e_proj_3d = e_proj[::step_3d, ::step_3d]
            ax_3d_e.plot_surface(
                X_3d,
                Y_3d,
                e_proj_3d,
                cmap="jet",
                vmin=0,
                vmax=e_max,
                alpha=0.9,
                linewidth=0,
                antialiased=True,
            )
            for seg in tumor_contour_segments:
                if len(seg) > 1:
                    ax_3d_e.plot(
                        seg[:, 0],
                        seg[:, 1],
                        np.zeros(seg.shape[0]),
                        "lime",
                        linewidth=1.5,
                    )
            ax_3d_e.set_xlabel("Y (cells)")
            ax_3d_e.set_ylabel("X (cells)")
            ax_3d_e.set_zlabel("Magnitude")
            ax_3d_e.set_title(f"E-field (Ez) 3D - Frame {frame_idx + 1}/{n_frames}")
            ax_3d_e.view_init(elev=30, azim=45)
            ax_3d_e.set_xlim(x_min, x_max)
            ax_3d_e.set_ylim(y_min, y_max)
            ax_3d_e.set_zlim(0, e_max)

        return update

    fig_3d_e = plt.figure(figsize=(12, 10))
    ax_3d_e = fig_3d_e.add_subplot(111, projection="3d")
    ani_3d_e = animation.FuncAnimation(
        fig_3d_e,
        make_update_3d_e(ax_3d_e),
        frames=n_anim_frames,
        interval=20,
        blit=False,
        repeat=True,
        repeat_delay=1000,
    )
    out_3d_e = os.path.join(animations_dir, f"{output_base}_efield_3d.mp4")
    print(f"Saving 3D E-field animation to {out_3d_e}...")
    ani_3d_e.save(out_3d_e, writer="ffmpeg", fps=args.fps)
    plt.close(fig_3d_e)
    print("✓ 3D E-field animation saved")

    def make_update_3d_sar(ax_3d_sar):
        def update(frame_num):
            ax_3d_sar.clear()
            frame_idx = frame_indices[frame_num]
            _, sar_data, _ = loader.get_frame(frame_idx)
            sar_proj = np.max(sar_data, axis=2)
            sar_proj_3d = sar_proj[::step_3d, ::step_3d]
            ax_3d_sar.plot_surface(
                X_3d,
                Y_3d,
                sar_proj_3d,
                cmap="coolwarm",
                vmin=0,
                vmax=sar_max,
                alpha=0.9,
                linewidth=0,
                antialiased=True,
            )
            for seg in tumor_contour_segments:
                if len(seg) > 1:
                    ax_3d_sar.plot(
                        seg[:, 0],
                        seg[:, 1],
                        np.zeros(seg.shape[0]),
                        "cyan",
                        linewidth=1.5,
                    )
            ax_3d_sar.set_xlabel("Y (cells)")
            ax_3d_sar.set_ylabel("X (cells)")
            ax_3d_sar.set_zlabel("SAR (W/kg)")
            ax_3d_sar.set_title(f"SAR 3D - Frame {frame_idx + 1}/{n_frames}")
            ax_3d_sar.view_init(elev=30, azim=45)
            ax_3d_sar.set_xlim(x_min, x_max)
            ax_3d_sar.set_ylim(y_min, y_max)
            ax_3d_sar.set_zlim(0, sar_max)

        return update

    fig_3d_sar = plt.figure(figsize=(12, 10))
    ax_3d_sar = fig_3d_sar.add_subplot(111, projection="3d")
    ani_3d_sar = animation.FuncAnimation(
        fig_3d_sar,
        make_update_3d_sar(ax_3d_sar),
        frames=n_anim_frames,
        interval=20,
        blit=False,
        repeat=True,
        repeat_delay=1000,
    )
    out_3d_sar = os.path.join(animations_dir, f"{output_base}_sar_3d.mp4")
    print(f"Saving 3D SAR animation to {out_3d_sar}...")
    ani_3d_sar.save(out_3d_sar, writer="ffmpeg", fps=args.fps)
    plt.close(fig_3d_sar)
    print("✓ 3D SAR animation saved")

    if has_temperature:

        def make_update_3d_temp(ax_3d_temp):
            def update(frame_num):
                ax_3d_temp.clear()
                frame_idx = frame_indices[frame_num]
                _, _, temp_data = loader.get_frame(frame_idx)
                temp_proj = np.max(temp_data, axis=2)
                temp_proj_3d = temp_proj[::step_3d, ::step_3d]
                ax_3d_temp.plot_surface(
                    X_3d,
                    Y_3d,
                    temp_proj_3d,
                    cmap="coolwarm",
                    vmin=temp_min,
                    vmax=temp_max,
                    alpha=0.9,
                    linewidth=0,
                    antialiased=True,
                )
                for seg in tumor_contour_segments:
                    if len(seg) > 1:
                        ax_3d_temp.plot(
                            seg[:, 0],
                            seg[:, 1],
                            np.zeros(seg.shape[0]),
                            "cyan",
                            linewidth=1.5,
                        )
                ax_3d_temp.set_xlabel("Y (cells)")
                ax_3d_temp.set_ylabel("X (cells)")
                ax_3d_temp.set_zlabel("Temperature (°C)")
                ax_3d_temp.set_title(
                    f"Temperature 3D - Frame {frame_idx + 1}/{n_frames}"
                )
                ax_3d_temp.view_init(elev=30, azim=45)
                ax_3d_temp.set_xlim(x_min, x_max)
                ax_3d_temp.set_ylim(y_min, y_max)
                ax_3d_temp.set_zlim(temp_min, temp_max)

            return update

        fig_3d_temp = plt.figure(figsize=(12, 10))
        ax_3d_temp = fig_3d_temp.add_subplot(111, projection="3d")
        ani_3d_temp = animation.FuncAnimation(
            fig_3d_temp,
            make_update_3d_temp(ax_3d_temp),
            frames=n_anim_frames,
            interval=20,
            blit=False,
            repeat=True,
            repeat_delay=1000,
        )
        out_3d_temp = os.path.join(animations_dir, f"{output_base}_temperature_3d.mp4")
        print(f"Saving 3D Temperature animation to {out_3d_temp}...")
        ani_3d_temp.save(out_3d_temp, writer="ffmpeg", fps=args.fps)
        plt.close(fig_3d_temp)
        print("✓ 3D Temperature animation saved")

    print("Done.")


if __name__ == "__main__":
    main()
