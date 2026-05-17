"""
Data Analysis and Validation (thesis: Data Analysis and Validation).

Figure generation and saving: geometry slice, tumor preview, SAR/temperature/anatomy
panels, optimization comparison and trace. Also: save_simulation_data (NIfTI, JSON, NPZ),
build_and_save_animations (2D/3D MP4). Progress writing is in progress.py; re-exported here for backward compatibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import animation, pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, ScalarFormatter

from hermes_drh.io.progress import write_progress

try:
    import nibabel as nib
except ImportError:
    nib = None


# Default BraTS label→RGB when label_colors is not provided (backward compatible)
_LABEL_COLORS_BRATS = {
    0: (0.15, 0.15, 0.15),
    1: (1.0, 0.2, 0.2),
    2: (0.2, 0.8, 0.2),
    3: (0.2, 0.2, 1.0),
    4: (0.6, 0.55, 0.5),
}
_DEFAULT_UNSEEN_COLOR = (0.5, 0.5, 0.5)


def _tumor_centroid_z_from_labels(labels_3d, nz):
    """Axial index of mean tumor (labels 1–3) z; fallback mid-grid."""
    if labels_3d is None:
        return int(nz // 2)
    m = (labels_3d >= 1) & (labels_3d <= 3)
    if not np.any(m):
        return int(nz // 2)
    tz = np.argwhere(m)[:, 2]
    z = int(round(float(np.mean(tz))))
    return max(0, min(z, nz - 1))


def _contour_tumor_footprint(ax, tumor_footprint_2d, color):
    return ax.contour(
        tumor_footprint_2d,
        levels=[0.5],
        colors=[color],
        linewidths=1.5,
        origin="lower",
    )


def _axial_sagittal_coronal_max(volume_3d):
    """Return max projections along z, x, and y axes."""
    axial = np.max(volume_3d, axis=2)  # x-y plane, max along z
    sagittal = np.max(volume_3d, axis=0)  # y-z plane, max along x
    coronal = np.max(volume_3d, axis=1)  # x-z plane, max along y
    return axial, sagittal, coronal


def _orient_sagittal_for_display(arr_2d):
    """Rotate sagittal counter-clockwise, then flip vertically."""
    return np.flipud(np.rot90(arr_2d, k=1))


def _orient_coronal_for_display(arr_2d):
    """Rotate coronal clockwise once for display."""
    return np.rot90(arr_2d, k=-1)


def labels_to_rgb(lab, label_colors=None):
    """Convert label array to RGB. If label_colors is None, use BraTS mapping (0–4). Else use provided dict; unseen labels use gray."""
    lab = lab.astype(np.int32)
    colors = label_colors if label_colors is not None else _LABEL_COLORS_BRATS
    if label_colors is None:
        lab = np.clip(lab, 0, 4)
    rgb = np.zeros((*lab.shape, 3))
    for idx, val in enumerate(np.unique(lab)):
        val = int(val)
        r, g, b = colors.get(val, _DEFAULT_UNSEEN_COLOR)
        rgb[lab == val] = [r, g, b]
    return rgb


def save_geometry_slice(labels_3d, output_base, images_dir, nz, viz_config=None):
    """Save 2D mid-Z slice of FDTD geometry (segmentation labels). If viz_config is set, use its geometry_slice_title."""
    geom_slice = labels_3d[:, :, nz // 2]
    label_colors = viz_config.label_colors if viz_config else None
    title = (
        viz_config.geometry_slice_title
        if viz_config
        else "FDTD geometry (mid-Z): dark=air, R/G/B=tumor, tan=normal brain"
    )
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(labels_to_rgb(geom_slice, label_colors=label_colors), origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Y (cells)")
    ax.set_ylabel("X (cells)")
    path = os.path.join(images_dir, f"{output_base}_fdtd_geometry_slice.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_tumor_preview(
    labels_3d, tumor_mask, output_base, images_dir, nx, ny, nz, viz_config=None
):
    """Save tumor visualization (axial/sagittal/coronal + 3D scatter) before FDTD. If viz_config is set, use its titles and label colors."""
    label_colors = viz_config.label_colors if viz_config else None
    suptitle = (
        viz_config.tumor_preview_suptitle
        if viz_config
        else "Brain segmentation – tumor visualization (before FDTD)"
    )
    title_3d = (
        viz_config.tumor_3d_title
        if viz_config
        else "3D tumor (red=necrotic, green=edema, blue=enhancing)"
    )
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    slice_axial = labels_to_rgb(labels_3d[:, :, cz], label_colors=label_colors)
    slice_sagittal = labels_to_rgb(labels_3d[cx, :, :], label_colors=label_colors)
    slice_coronal = labels_to_rgb(labels_3d[:, cy, :], label_colors=label_colors)
    ax1.imshow(slice_axial, origin="lower")
    ax1.set_title(f"Axial (z={cz})")
    ax1.axis("off")
    ax2.imshow(slice_sagittal, origin="lower")
    ax2.set_title(f"Sagittal (x={cx})")
    ax2.axis("off")
    ax3.imshow(slice_coronal, origin="lower")
    ax3.set_title(f"Coronal (y={cy})")
    ax3.axis("off")
    step = max(1, min(nx, ny, nz) // 30)
    ii, jj, kk = np.where(tumor_mask)
    ii, jj, kk = ii[::step], jj[::step], kk[::step]
    if len(ii) > 0:
        lab = labels_3d[ii, jj, kk]
        if label_colors is not None:
            colors_3d = np.array(
                [label_colors.get(int(l), _DEFAULT_UNSEEN_COLOR) for l in lab]
            )
        else:
            colors_3d = np.array([[1, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 1]])[
                np.clip(lab.astype(int) - 1, 0, 2)
            ]
        ax4.scatter(ii, jj, kk, c=colors_3d, s=2, alpha=0.6)
    ax4.set_title(title_3d)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.view_init(elev=20, azim=45)
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    path = os.path.join(images_dir, f"{output_base}_tumor_preview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_sar_distribution(
    SAR,
    output_base,
    images_dir,
    tumor_footprint_2d,
    cx,
    cy,
    cz,
    tumor_mask_3d=None,
):
    """Save 3-panel SAR max-projection figure (axial, sagittal, coronal)."""
    sar_axial, sar_sagittal, sar_coronal = _axial_sagittal_coronal_max(SAR)
    sar_max_val = np.max(SAR) if np.max(SAR) > 0 else 1.0
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(
        sar_axial, origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax1.set_title("SAR axial max projection (along Z)")
    ax1.set_xlabel("Y (cells)")
    ax1.set_ylabel("X (cells)")
    plt.colorbar(im1, ax=ax1, label="SAR (W/kg)")
    ax1.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )

    if tumor_mask_3d is not None:
        tum_axial, tum_sagittal, tum_coronal = _axial_sagittal_coronal_max(
            tumor_mask_3d.astype(np.float32)
        )
    else:
        tum_axial = tumor_footprint_2d
        tum_sagittal = None
        tum_coronal = None

    sar_sagittal_view = _orient_sagittal_for_display(sar_sagittal)
    tum_sagittal_view = (
        _orient_sagittal_for_display(tum_sagittal) if tum_sagittal is not None else None
    )
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(
        sar_sagittal_view, origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax2.set_title("SAR sagittal max projection (along X)")
    ax2.set_xlabel("Y (cells) [oriented]")
    ax2.set_ylabel("Z (cells) [oriented]")
    plt.colorbar(im2, ax=ax2, label="SAR (W/kg)")
    if tum_sagittal_view is not None:
        ax2.contour(
            tum_sagittal_view,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1,
            origin="lower",
        )

    sar_coronal_view = _orient_coronal_for_display(sar_coronal)
    tum_coronal_view = (
        _orient_coronal_for_display(tum_coronal) if tum_coronal is not None else None
    )
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(
        sar_coronal_view, origin="lower", cmap="coolwarm", vmin=0, vmax=sar_max_val
    )
    ax3.set_title("SAR coronal max projection (along Y)")
    ax3.set_xlabel("Z (cells) [oriented]")
    ax3.set_ylabel("X (cells) [oriented]")
    plt.colorbar(im3, ax=ax3, label="SAR (W/kg)")
    if tum_coronal_view is not None:
        ax3.contour(
            tum_coronal_view,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1,
            origin="lower",
        )

    fig.suptitle("Total SAR pattern distribution (post-FDTD)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(images_dir, f"{output_base}_SAR_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _distribution_temperature_bounds(
    T_vol: np.ndarray,
    *projection_2d: np.ndarray,
) -> tuple[float, float]:
    """
    ``vmin`` / ``vmax`` for the 3-panel max-projection figure.

    Prefer min/max over the **three projected images** (same spirit as the slice
    triptych, which uses min/max on the 2D slice).

    If those projections are **numerically flat** (often after float32 NIfTI
    round-off so every displayed pixel is exactly ``37``), fall back to
    **full-volume** ``min(T)`` / ``max(T)`` so a real ~1e-13 °C span is not
    replaced by the flat guard ``vmax = vmin + 0.1`` (which forces a 0.1 °C
    bar and hides sub-millikelvin variation).

    If still flat, use ``vmax = vmin + 0.1`` (matches ``save_slice_anatomy_sar_temperature``).
    """
    projs = [np.asarray(a, dtype=np.float64) for a in projection_2d]
    Tv = np.asarray(T_vol, dtype=np.float64).ravel()
    d_min = min(float(np.min(a)) for a in projs) if projs else float(np.min(Tv))
    d_max = max(float(np.max(a)) for a in projs) if projs else float(np.max(Tv))
    v_min = float(np.min(Tv))
    v_max = float(np.max(Tv))
    span_d = d_max - d_min
    span_v = v_max - v_min
    eps = max(
        np.finfo(np.float64).eps * max(abs(d_max), abs(d_min), 37.0, 1.0),
        1e-24,
    )
    if span_d > eps:
        lo, hi = d_min, d_max
    elif span_v > eps:
        lo, hi = v_min, v_max
    else:
        lo = v_min
        hi = v_max if v_max > v_min else v_min + 0.1
    if hi <= lo:
        hi = lo + 0.1
    return lo, hi


def _temperature_colorbar_absolute_T(
    mappable,
    ax,
    *,
    vmin_t: float,
    vmax_t: float,
    extend=None,
):
    """
    Colorbar for absolute temperature ``T`` (°C).

    Wide span: ticks are absolute ``T`` with a plain scalar formatter.

    Narrow span (typical bioheat maps near 37 °C): tick labels show
    ``ΔT = T - T_{\mathrm{ref}}`` in **scientific notation** with
    ``T_{\mathrm{ref}} = T_{\min}`` on the colorbar label, so long runs of
    leading zeros are avoided.
    """
    cb = plt.colorbar(mappable, ax=ax, extend=extend or "neither")
    span = float(vmax_t) - float(vmin_t)
    if not np.isfinite(span):
        span = 0.0

    # Wide range: absolute T on ticks.
    if span >= 0.25:
        cb.set_label(r"$T$ ($^\circ$C)")
        cb.locator = MaxNLocator(nbins=8, min_n_ticks=4)
        cb.update_ticks()
        fmt = ScalarFormatter(useMathText=False, useOffset=False)
        fmt.set_powerlimits((-12, 12))
        cb.ax.yaxis.set_major_formatter(fmt)
        return cb

    lo, hi = float(vmin_t), float(vmax_t)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-12, 1e-15)
        span = hi - lo
    t_ref = lo
    ticks = np.linspace(lo, hi, 7, dtype=np.float64)
    cb.locator = FixedLocator(ticks)
    cb.update_ticks()

    def _fmt_delta(x, _pos):
        if not np.isfinite(x):
            return ""
        d = float(x) - t_ref
        if abs(d) < 1e-300:
            return "0"
        return f"{d:.2e}"

    cb.ax.yaxis.set_major_formatter(FuncFormatter(_fmt_delta))
    cb.set_label(
        r"$\Delta T = T - T_{\mathrm{ref}}$ ($^\circ$C)"
        + "\n"
        + rf"$T_{{\mathrm{{ref}}}} = {t_ref:.10g}$"
    )
    return cb


def save_temperature_distribution(
    T_temp,
    output_base,
    images_dir,
    tumor_footprint_2d,
    cx,
    cy,
    cz,
    tumor_mask_3d=None,
):
    """Save 3-panel temperature max-projection figure (axial, sagittal, coronal)."""
    T_work = np.asarray(T_temp, dtype=np.float64)
    temp_axial, temp_sagittal, temp_coronal = _axial_sagittal_coronal_max(T_work)
    temp_sagittal_view = _orient_sagittal_for_display(temp_sagittal.astype(np.float64))
    temp_coronal_view = _orient_coronal_for_display(temp_coronal.astype(np.float64))
    # vmin/vmax: prefer range on the three projections; if flat (float32 crush), use
    # full-volume min/max so ~1e-13 °C is not replaced by vmin+0.1 (see helper docstring).
    vmin_plot, vmax_plot = _distribution_temperature_bounds(
        T_work,
        temp_axial,
        temp_sagittal_view,
        temp_coronal_view,
    )
    cb_extend = "neither"
    temp_norm = mcolors.Normalize(vmin=vmin_plot, vmax=vmax_plot)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(
        temp_axial.astype(np.float64),
        origin="lower",
        cmap="coolwarm",
        norm=temp_norm,
    )
    ax1.set_title("Temperature axial max projection (along Z)")
    ax1.set_xlabel("Y (cells)")
    ax1.set_ylabel("X (cells)")
    _temperature_colorbar_absolute_T(
        im1,
        ax1,
        vmin_t=vmin_plot,
        vmax_t=vmax_plot,
        extend=cb_extend,
    )
    ax1.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )

    if tumor_mask_3d is not None:
        tum_axial, tum_sagittal, tum_coronal = _axial_sagittal_coronal_max(
            tumor_mask_3d.astype(np.float32)
        )
    else:
        tum_axial = tumor_footprint_2d
        tum_sagittal = None
        tum_coronal = None

    tum_sagittal_view = (
        _orient_sagittal_for_display(tum_sagittal) if tum_sagittal is not None else None
    )
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(
        temp_sagittal_view,
        origin="lower",
        cmap="coolwarm",
        norm=temp_norm,
    )
    ax2.set_title("Temperature sagittal max projection (along X)")
    ax2.set_xlabel("Y (cells) [oriented]")
    ax2.set_ylabel("Z (cells) [oriented]")
    _temperature_colorbar_absolute_T(
        im2,
        ax2,
        vmin_t=vmin_plot,
        vmax_t=vmax_plot,
        extend=cb_extend,
    )
    if tum_sagittal_view is not None:
        ax2.contour(
            tum_sagittal_view,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1,
            origin="lower",
        )

    tum_coronal_view = (
        _orient_coronal_for_display(tum_coronal) if tum_coronal is not None else None
    )
    ax3 = fig.add_subplot(1, 3, 3)
    im3 = ax3.imshow(
        temp_coronal_view,
        origin="lower",
        cmap="coolwarm",
        norm=temp_norm,
    )
    ax3.set_title("Temperature coronal max projection (along Y)")
    ax3.set_xlabel("Z (cells) [oriented]")
    ax3.set_ylabel("X (cells) [oriented]")
    _temperature_colorbar_absolute_T(
        im3,
        ax3,
        vmin_t=vmin_plot,
        vmax_t=vmax_plot,
        extend=cb_extend,
    )
    if tum_coronal_view is not None:
        ax3.contour(
            tum_coronal_view,
            levels=[0.5],
            colors=["cyan"],
            linewidths=1,
            origin="lower",
        )
    fig.suptitle(
        "Temperature distribution (Pennes steady-state, no perfusion)", fontsize=12
    )
    plt.tight_layout()
    path = os.path.join(images_dir, f"{output_base}_temperature_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def regenerate_run_figures_from_disk(results_dir: str | Path, output_base: str) -> bool:
    """
    Rebuild geometry, SAR/temperature distribution, and multiview plates under
    ``{results_dir}/images`` from saved NIfTI. If ``E_frames_part*.npz`` exist,
    also rebuilds timeline montages via :func:`multiview_visualization.export_all_multiview_static`.

    Returns True if core NIfTI were found and distribution figures were written.
    """
    results_dir = Path(results_dir).resolve()
    data_dir = results_dir / "data"
    images_dir = results_dir / "images"
    if not data_dir.is_dir():
        print(f"[WARN] refresh: missing data directory: {data_dir}")
        return False
    images_dir.mkdir(parents=True, exist_ok=True)
    if nib is None:
        print("[WARN] refresh: nibabel not installed; cannot regenerate from NIfTI")
        return False
    seg = data_dir / f"{output_base}_segmentation.nii.gz"
    sar_p = data_dir / f"{output_base}_SAR.nii.gz"
    tmp_p = data_dir / f"{output_base}_temperature.nii.gz"
    if not seg.is_file() or not sar_p.is_file():
        print(
            f"[WARN] refresh: need segmentation + SAR NIfTI under {data_dir} "
            f"(base={output_base!r})"
        )
        return False
    labels_3d = np.asarray(nib.load(seg).get_fdata(), dtype=np.float32).squeeze()
    labels_3d = np.round(labels_3d).astype(np.int32)
    SAR = np.asarray(nib.load(sar_p).get_fdata(), dtype=np.float32).squeeze()
    if tmp_p.is_file():
        # float64: float32 NIfTI cannot represent ~1e-13 °C ΔT above 37 °C; prefer saving
        # thermal volumes as float64 in the pipeline when sub-µK contrast matters.
        T_temp = np.asarray(nib.load(tmp_p).get_fdata(), dtype=np.float64).squeeze()
        has_temperature = True
    else:
        T_temp = np.zeros_like(SAR, dtype=np.float64)
        has_temperature = False
    tumor_region = (labels_3d >= 1) & (labels_3d <= 3)
    tumor_footprint_2d = np.max(tumor_region.astype(np.float32), axis=2)
    nx, ny, nz = labels_3d.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    save_geometry_slice(labels_3d, output_base, str(images_dir), nz, viz_config=None)
    save_sar_distribution(
        SAR,
        output_base,
        str(images_dir),
        tumor_footprint_2d,
        cx,
        cy,
        cz,
        tumor_mask_3d=tumor_region,
    )
    if has_temperature:
        save_temperature_distribution(
            T_temp,
            output_base,
            str(images_dir),
            tumor_footprint_2d,
            cx,
            cy,
            cz,
            tumor_mask_3d=tumor_region.astype(np.float32),
        )
    meta_path = data_dir / f"{output_base}_metadata.json"
    e0 = data_dir / f"{output_base}_E_frames_part0.npz"
    try:
        if e0.is_file() and meta_path.is_file():
            from hermes_drh.visualization.animations import FrameLoader
            from hermes_drh.visualization.multiview import export_all_multiview_static

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            n_frames_meta = int(meta.get("n_frames", 0))
            if n_frames_meta <= 0:
                raise ValueError("metadata n_frames missing or zero")
            chunk_size = int(meta.get("E_frames_chunk_size", 20))
            loader = FrameLoader(str(data_dir), output_base, chunk_size, n_frames_meta)
            saved = meta.get("saved_frame_timesteps")
            if not saved:
                tm = int(meta.get("time_steps", n_frames_meta))
                sf = max(1, int(meta.get("stream_frame_interval", 1)))
                saved = [t for t in range(1, tm + 1) if t % sf == 0][:n_frames_meta]
            export_all_multiview_static(
                get_frame=loader.get_frame,
                saved_timesteps=saved,
                time_steps=int(meta.get("time_steps", len(saved))),
                labels_3d=labels_3d,
                sar_3d=SAR,
                temp_3d=T_temp,
                output_base=output_base,
                images_dir=str(images_dir),
                has_temperature=has_temperature,
            )
        else:
            from hermes_drh.visualization.multiview import export_multiview_plates_from_volumes

            export_multiview_plates_from_volumes(
                SAR,
                T_temp,
                labels_3d,
                output_base,
                str(images_dir),
                has_temperature=has_temperature,
            )
    except Exception as e:
        print(f"[WARN] refresh multiview failed ({e}); trying plates-only")
        try:
            from hermes_drh.visualization.multiview import export_multiview_plates_from_volumes

            export_multiview_plates_from_volumes(
                SAR,
                T_temp,
                labels_3d,
                output_base,
                str(images_dir),
                has_temperature=has_temperature,
            )
        except Exception as e2:
            print(f"[WARN] refresh multiview plates-only failed: {e2}")
    print(
        f"[INFO] Refreshed run figures from disk → {images_dir} (base={output_base!r})"
    )
    return True


def save_optimization_comparison(
    sar_baseline,
    sar_optimized,
    J_baseline,
    J_opt,
    mid_z,
    output_base,
    images_dir,
    f0_mhz,
):
    """Save baseline vs optimized SAR comparison (3 panels: baseline, optimized, difference)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sar_bl_slice = sar_baseline[:, :, mid_z]
    sar_opt_slice = sar_optimized[:, :, mid_z]
    ax_bl, ax_opt, ax_diff = axes[0], axes[1], axes[2]
    im_bl = ax_bl.imshow(
        sar_bl_slice,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=max(np.max(sar_bl_slice), 1e-12),
    )
    ax_bl.set_title(f"Baseline SAR (z={mid_z})\nJ = {J_baseline:.4f}")
    ax_bl.set_xlabel("Y (cells)")
    ax_bl.set_ylabel("X (cells)")
    plt.colorbar(im_bl, ax=ax_bl, label="SAR (W/kg)")
    im_opt = ax_opt.imshow(
        sar_opt_slice,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=max(np.max(sar_opt_slice), 1e-12),
    )
    ax_opt.set_title(f"Optimized SAR (z={mid_z})\nJ = {J_opt:.4f}")
    ax_opt.set_xlabel("Y (cells)")
    ax_opt.set_ylabel("X (cells)")
    plt.colorbar(im_opt, ax=ax_opt, label="SAR (W/kg)")
    sar_diff_slice = sar_opt_slice - sar_bl_slice
    vabs = max(np.max(np.abs(sar_diff_slice)), 1e-12)
    im_diff = ax_diff.imshow(
        sar_diff_slice, origin="lower", cmap="RdBu_r", vmin=-vabs, vmax=vabs
    )
    ax_diff.set_title(f"Difference (Opt - Baseline)\nz={mid_z}")
    ax_diff.set_xlabel("Y (cells)")
    ax_diff.set_ylabel("X (cells)")
    plt.colorbar(im_diff, ax=ax_diff, label="ΔSAR (W/kg)")
    fig.suptitle(
        f"Antenna Optimization: Baseline vs Optimized SAR (f0={f0_mhz:.0f} MHz)",
        fontsize=13,
    )
    plt.tight_layout()
    path = os.path.join(images_dir, f"{output_base}_antenna_opt_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_optimization_trace(opt_trace, J_baseline, output_base, images_dir):
    """Save optimization convergence trace plot."""
    if len(opt_trace) <= 1:
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    trace_evals = [t["eval"] for t in opt_trace]
    trace_js = [t.get("J", t.get("J_eff", 0)) for t in opt_trace]
    ax.plot(trace_evals, trace_js, "o-", markersize=2, linewidth=0.8)
    ax.axhline(
        y=J_baseline, color="red", linestyle="--", label=f"Baseline J={J_baseline:.4f}"
    )
    ax.axhline(
        y=1.0,
        color="green",
        linestyle=":",
        alpha=0.5,
        label="J=1.0 (selectivity threshold)",
    )
    ax.set_xlabel("Evaluation number")
    ax.set_ylabel("J (SAR_tumor / SAR_healthy)")
    ax.set_title("Antenna Optimization Convergence Trace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(images_dir, f"{output_base}_optimization_trace.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_slice_anatomy_sar_temperature(
    k,
    labels_3d,
    SAR,
    T_temp,
    tumor_footprint_2d,
    output_base,
    images_dir,
    volume_4d_ds=None,
    viz_config=None,
):
    """Save one slice: Anatomy | SAR | Temperature. Optional FLAIR overlay if volume_4d_ds given. If viz_config is set, use its titles and label colors."""
    fig = plt.figure(figsize=(14, 5))
    ax_anat = fig.add_subplot(1, 3, 1)
    ax_sar = fig.add_subplot(1, 3, 2)
    ax_temp = fig.add_subplot(1, 3, 3)
    if volume_4d_ds is not None:
        try:
            from matplotlib.colors import ListedColormap
            from hermes_drh.segmentation.brain import SEG_COLORS_STREAMLIT

            flair_slice = volume_4d_ds[0, :, :, k]
            seg_slice = labels_3d[:, :, k].astype(np.int32)
            seg_vmax = 4 if np.max(labels_3d) >= 4 else 3
            seg_cmap_local = ListedColormap(SEG_COLORS_STREAMLIT[: seg_vmax + 1])
            ax_anat.imshow(flair_slice, cmap="gray", origin="lower", alpha=0.7)
            mask_overlay = np.ma.masked_where(seg_slice == 0, seg_slice)
            ax_anat.imshow(
                mask_overlay,
                cmap=seg_cmap_local,
                origin="lower",
                vmin=0,
                vmax=seg_vmax,
                alpha=0.8,
                interpolation="nearest",
            )
            ax_anat.set_title(f"FLAIR + segmentation (z={k})")
        except ImportError:
            label_colors = viz_config.label_colors if viz_config else None
            ax_anat.imshow(
                labels_to_rgb(labels_3d[:, :, k], label_colors=label_colors),
                origin="lower",
            )
            ax_anat.set_title(f"Segmentation (z={k})")
    else:
        label_colors = viz_config.label_colors if viz_config else None
        ax_anat.imshow(
            labels_to_rgb(labels_3d[:, :, k], label_colors=label_colors), origin="lower"
        )
        ax_anat.set_title(f"Segmentation (z={k})")
    ax_anat.set_xlabel("Y (cells)")
    ax_anat.set_ylabel("X (cells)")
    ax_anat.axis("on")
    sar_max_k = np.max(SAR) if np.max(SAR) > 0 else 1.0
    im_sar = ax_sar.imshow(
        SAR[:, :, k], origin="lower", cmap="gray", vmin=0, vmax=sar_max_k
    )
    ax_sar.set_title(f"SAR axial (z={k})")
    ax_sar.set_xlabel("Y (cells)")
    ax_sar.set_ylabel("X (cells)")
    plt.colorbar(im_sar, ax=ax_sar, label="SAR (W/kg)")
    ax_sar.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )
    T_max_k = np.max(T_temp)
    T_min_k = np.min(T_temp)
    if T_max_k <= T_min_k:
        T_max_k = T_min_k + 0.1
    im_temp = ax_temp.imshow(
        T_temp[:, :, k],
        origin="lower",
        cmap="coolwarm",
        vmin=T_min_k,
        vmax=T_max_k,
    )
    ax_temp.set_title(f"Temperature axial (z={k})")
    ax_temp.set_xlabel("Y (cells)")
    ax_temp.set_ylabel("X (cells)")
    _temperature_colorbar_absolute_T(
        im_temp, ax_temp, vmin_t=float(T_min_k), vmax_t=float(T_max_k)
    )
    ax_temp.contour(
        tumor_footprint_2d, levels=[0.5], colors=["cyan"], linewidths=1, origin="lower"
    )
    if viz_config and viz_config.slice_panel_suffix:
        suptitle = (
            f"Slice {k} ({viz_config.slice_panel_suffix}): Anatomy | SAR | Temperature"
        )
    else:
        suptitle = (
            f"Slice {k} (top 10 by tumor area): Anatomy | SAR | Temperature"
            if viz_config is None
            else f"Slice {k}: Anatomy | SAR | Temperature"
        )
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    path = os.path.join(
        images_dir, f"{output_base}_slice_{k}_anatomy_SAR_temperature.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# write_progress imported from hermes_drh.io.progress.py (single responsibility: dashboard progress)


def save_simulation_data(
    output_base,
    data_dir,
    sar_frames_dir,
    temperature_frames_dir,
    e_frames_dir,
    SAR,
    T_temp,
    labels_3d,
    affine,
    E_frames,
    SAR_frames,
    Temperature_frames,
    streamed_n_frames,
    stream_frames,
    metadata,
    performance_metrics,
    time_series_data=None,
    e_frames_chunk_size=20,
):
    """
    Save simulation outputs: SAR/temperature/segmentation NIfTI, metadata JSON,
    performance JSON, optional time_series JSON, and E/SAR/T frame NPZ (when not streamed).
    """
    if nib is None:
        raise ImportError("nibabel is required for save_simulation_data")
    n_frames = streamed_n_frames if streamed_n_frames > 0 else len(E_frames)
    n_sar_frames = streamed_n_frames if streamed_n_frames > 0 else len(SAR_frames)
    n_temp_frames = (
        streamed_n_frames if streamed_n_frames > 0 else len(Temperature_frames)
    )
    grid_shape = metadata.get("grid_shape", [0, 0, 0])
    simulation_size_x, simulation_size_y, simulation_size_z = (
        grid_shape[0],
        grid_shape[1],
        grid_shape[2],
    )

    # 1) NIfTI
    sar_path = os.path.join(data_dir, f"{output_base}_SAR.nii.gz")
    nib.save(nib.Nifti1Image(SAR.astype(np.float32), affine), sar_path)
    print(f"  SAR saved to {sar_path}")
    temperature_path = os.path.join(data_dir, f"{output_base}_temperature.nii.gz")
    # float64: float32 cannot resolve ~1e-13 °C ΔT around 37 °C (plots look uniform).
    nib.save(
        nib.Nifti1Image(np.asarray(T_temp, dtype=np.float64), affine), temperature_path
    )
    print(f"  Temperature saved to {temperature_path}")
    segmentation_path = os.path.join(data_dir, f"{output_base}_segmentation.nii.gz")
    nib.save(nib.Nifti1Image(labels_3d.astype(np.int32), affine), segmentation_path)
    print(f"  Segmentation (labels) saved to {segmentation_path}")

    # 2) Metadata and performance JSON
    metadata_path = os.path.join(data_dir, f"{output_base}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")
    performance_path = os.path.join(data_dir, f"{output_base}_performance.json")
    with open(performance_path, "w") as f:
        json.dump(performance_metrics, f, indent=2)
    print(f"  Performance metrics saved to {performance_path}")

    if time_series_data is not None:
        time_series_path = os.path.join(data_dir, f"{output_base}_time_series.json")
        with open(time_series_path, "w") as f:
            json.dump(time_series_data, f, indent=2)
        print(f"  Time series saved to {time_series_path}")

    # 3) E_frames NPZ (from memory only when not streamed)
    if n_frames > 0 and not stream_frames:
        n_parts_actual = (n_frames + e_frames_chunk_size - 1) // e_frames_chunk_size
        for part in range(n_parts_actual):
            start = part * e_frames_chunk_size
            end = min(start + e_frames_chunk_size, n_frames)
            chunk = np.array(E_frames[start:end], dtype=np.float32)
            part_path = os.path.join(
                e_frames_dir, f"{output_base}_E_frames_part{part}.npz"
            )
            np.savez_compressed(part_path, E_frames=chunk)
            del chunk
        print(
            f"  E-field time series saved to {e_frames_dir}/ ({n_parts_actual} parts, "
            f"shape ({n_frames}, {simulation_size_x}, {simulation_size_y}, {simulation_size_z}))"
        )
    elif n_frames > 0 and stream_frames:
        print(f"  E-field time series already streamed to disk ({n_frames} frames)")
    else:
        print("  E_frames empty, skipping E-field save")

    # 4) SAR_frames NPZ
    sar_frames_n_parts = (
        (n_sar_frames + e_frames_chunk_size - 1) // e_frames_chunk_size
        if n_sar_frames > 0
        else 0
    )
    if n_sar_frames > 0 and not stream_frames:
        for part in range(sar_frames_n_parts):
            start = part * e_frames_chunk_size
            end = min(start + e_frames_chunk_size, n_sar_frames)
            chunk = np.array(SAR_frames[start:end], dtype=np.float32)
            part_path = os.path.join(
                sar_frames_dir, f"{output_base}_SAR_frames_part{part}.npz"
            )
            np.savez_compressed(part_path, SAR_frames=chunk)
            del chunk
        print(
            f"  SAR time series saved to {sar_frames_dir}/ ({sar_frames_n_parts} parts)"
        )
    elif n_sar_frames > 0 and stream_frames:
        print(f"  SAR time series already streamed to disk ({n_sar_frames} frames)")
    else:
        print("  SAR_frames empty, skipping SAR frames save")

    # 5) Temperature_frames NPZ
    temp_frames_n_parts = (
        (n_temp_frames + e_frames_chunk_size - 1) // e_frames_chunk_size
        if n_temp_frames > 0
        else 0
    )
    if n_temp_frames > 0 and not stream_frames:
        for part in range(temp_frames_n_parts):
            start = part * e_frames_chunk_size
            end = min(start + e_frames_chunk_size, n_temp_frames)
            chunk = np.array(Temperature_frames[start:end], dtype=np.float32)
            part_path = os.path.join(
                temperature_frames_dir,
                f"{output_base}_Temperature_frames_part{part}.npz",
            )
            np.savez_compressed(part_path, Temperature_frames=chunk)
            del chunk
        print(
            f"  Temperature time series saved to {temperature_frames_dir}/ "
            f"({temp_frames_n_parts} parts)"
        )
    elif n_temp_frames > 0 and stream_frames:
        print(
            f"  Temperature time series already streamed to disk ({n_temp_frames} frames)"
        )
    else:
        print("  Temperature_frames empty, skipping temperature frames save")
    print(
        "  Simulation data save complete (SAR, temperature, metadata, E/SAR/T frames)."
    )


def build_and_save_animations(
    output_base,
    animations_dir,
    E_frames,
    SAR_frames,
    Temperature_frames,
    tumor_footprint_2d,
    tumor_contour_segments,
    T_BOUNDARY_CELSIUS,
    skip_animations,
    stream_frames,
    streamed_n_frames,
    script_dir,
    results_dir_abs,
    animations_dir_abs,
    subsample=1,
    slice_timestep_images=False,
    volume_4d_ds=None,
    labels_3d=None,
    sar_3d=None,
    temperature_3d=None,
    images_dir=None,
    case_name=None,
    saved_frame_timesteps=None,
    time_steps=None,
    data_dir_abs=None,
):
    """
    Build static outputs and (optionally) save 2D/3D MP4 animations.
    If skip_animations is True, MP4 encoding is skipped but static multiview exports
    are still generated when inputs are available.
    If stream_frames and streamed_n_frames>0 with skip_animations=False, invoke
    build_animations_from_streamed_frames.py.
    In-memory path: multiview static PNGs run before 3x15 previews and before FuncAnimation saves.
    Returns duration in seconds (animations_s).
    """
    t_start_anim = time.perf_counter()
    subsample = max(1, int(subsample))
    if stream_frames and streamed_n_frames > 0:
        if skip_animations:
            print(
                f"  Frames were streamed to disk ({streamed_n_frames} frames). "
                "Skipping MP4 encoding (--skip-animations) and exporting static multiview figures only..."
            )
            try:
                from hermes_drh.visualization.animations import FrameLoader
                from hermes_drh.visualization.multiview import export_all_multiview_static

                if nib is None:
                    raise ImportError("nibabel is required for streamed static export")

                data_dir = (
                    os.path.abspath(data_dir_abs)
                    if data_dir_abs is not None
                    else os.path.join(os.path.abspath(results_dir_abs), "data")
                )
                meta_path = os.path.join(data_dir, f"{output_base}_metadata.json")
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                n_frames_meta = int(meta.get("n_frames", streamed_n_frames))
                chunk_size = int(meta.get("E_frames_chunk_size", 20))
                loader = FrameLoader(data_dir, output_base, chunk_size, n_frames_meta)

                saved = meta.get("saved_frame_timesteps")
                if not saved:
                    tm = int(meta.get("time_steps", n_frames_meta))
                    sf = max(1, int(meta.get("stream_frame_interval", 1)))
                    saved = [t for t in range(1, tm + 1) if t % sf == 0][:n_frames_meta]

                labels_path = os.path.join(
                    data_dir, f"{output_base}_segmentation.nii.gz"
                )
                sar_path = os.path.join(data_dir, f"{output_base}_SAR.nii.gz")
                temp_path = os.path.join(data_dir, f"{output_base}_temperature.nii.gz")

                labels_3d_stream = np.asarray(
                    nib.load(labels_path).get_fdata(), dtype=np.float32
                ).squeeze()
                labels_3d_stream = np.round(labels_3d_stream).astype(np.int32)
                sar_3d_stream = np.asarray(
                    nib.load(sar_path).get_fdata(), dtype=np.float32
                ).squeeze()
                if os.path.isfile(temp_path):
                    temp_3d_stream = np.asarray(
                        nib.load(temp_path).get_fdata(), dtype=np.float32
                    ).squeeze()
                    has_temp_stream = True
                else:
                    temp_3d_stream = np.zeros_like(sar_3d_stream, dtype=np.float32)
                    has_temp_stream = False

                images_dir_stream = (
                    images_dir
                    if images_dir is not None
                    else os.path.join(os.path.abspath(results_dir_abs), "images")
                )
                os.makedirs(images_dir_stream, exist_ok=True)
                export_all_multiview_static(
                    get_frame=loader.get_frame,
                    saved_timesteps=saved,
                    time_steps=int(meta.get("time_steps", len(saved))),
                    labels_3d=labels_3d_stream,
                    sar_3d=sar_3d_stream,
                    temp_3d=temp_3d_stream,
                    output_base=output_base,
                    images_dir=images_dir_stream,
                    has_temperature=has_temp_stream,
                )
                print(f"  Multiview static PNGs saved to {images_dir_stream}/")
            except Exception as e:
                print(f"  Warning: streamed static multiview export failed: {e}")
            return time.perf_counter() - t_start_anim

        print(
            f"  Frames were streamed to disk ({streamed_n_frames} frames). "
            "Building animations from streamed frames (this may take a while)..."
        )
        anim_argv = [
            sys.executable,
            "-m",
            "hermes_drh.visualization.animations",
            results_dir_abs,
            animations_dir_abs,
            "--subsample",
            str(subsample),
        ]
        if slice_timestep_images:
            anim_argv.append("--generate-slice-timestep-images")
        proc = subprocess.run(anim_argv, stdin=subprocess.DEVNULL)
        if proc.returncode == 0:
            print(
                f"  hermes-build-animations finished (saved to {animations_dir_abs})."
            )
        else:
            print(
                f"  Warning: animation builder exited with code {proc.returncode}."
            )
        return time.perf_counter() - t_start_anim
    if len(E_frames) == 0:
        return time.perf_counter() - t_start_anim

    # Static multiview PNGs before create_3x15 previews or any MP4 encoding
    if (
        saved_frame_timesteps is not None
        and time_steps is not None
        and labels_3d is not None
        and sar_3d is not None
        and temperature_3d is not None
        and images_dir is not None
    ):
        try:
            from hermes_drh.visualization.multiview import export_all_multiview_static

            n_fr = len(E_frames)

            def _get_mem_frame(i):
                return (
                    E_frames[i],
                    SAR_frames[i],
                    Temperature_frames[i] if Temperature_frames else None,
                )

            export_all_multiview_static(
                get_frame=_get_mem_frame,
                saved_timesteps=list(saved_frame_timesteps)[:n_fr],
                time_steps=int(time_steps),
                labels_3d=labels_3d,
                sar_3d=np.asarray(sar_3d, dtype=np.float32),
                temp_3d=np.asarray(temperature_3d, dtype=np.float32),
                output_base=output_base,
                images_dir=images_dir,
                has_temperature=len(Temperature_frames) > 0,
            )
            print(f"  Multiview static PNGs saved to {images_dir}/")
        except Exception as e:
            print(f"  Warning: multiview static export failed: {e}")

    if (
        volume_4d_ds is not None
        and labels_3d is not None
        and sar_3d is not None
        and temperature_3d is not None
        and images_dir is not None
        and case_name is not None
    ):
        try:
            from hermes_drh.segmentation.brain import create_3x15_tumor_previews

            create_3x15_tumor_previews(
                volume_4d_ds,
                labels_3d,
                sar_3d,
                temperature_3d,
                output_dir=images_dir,
                case_name=case_name,
                n_slices=15,
            )
            print(
                f"  Saved 3x15 grid previews (FLAIR+seg, SAR, Temperature) to {images_dir}/"
            )
        except Exception as e:
            print(f"  Warning: failed to create 3x15 tumor previews: {e}")
        except ImportError:
            pass

    if skip_animations:
        print("\nSkipping MP4 animation encoding (--skip-animations).")
        return time.perf_counter() - t_start_anim

    has_temperature = len(Temperature_frames) > 0
    e_max = max(np.max(np.abs(f)) for f in E_frames)
    sar_max = max(np.max(f) for f in SAR_frames) if SAR_frames else 1.0
    temp_min = 37.0
    temp_max = 37.0 + 1.0
    if has_temperature and Temperature_frames:
        temp_max = max(np.max(f) for f in Temperature_frames)
    temp_max = max(temp_max, temp_min + 1e-3)
    step_3d = 2

    if len(SAR_frames) > 0:
        from hermes_drh.visualization.multiview import (
            make_update_6panel_efield_sar,
            tumor_footprints_three_views,
        )

        _lab = labels_3d
        if _lab is None:
            _lab = np.zeros(E_frames[0].shape, dtype=np.int32)
        fp_ax, fp_sag, fp_cor = tumor_footprints_three_views(_lab)

        print(
            "\nCreating 2D 2×3 animation (E / SAR × axial / sagittal / coronal, p99.5; no temperature row)..."
        )

        frame_indices = list(range(0, len(E_frames), subsample))
        n_anim_frames = len(frame_indices)
        print(f"  Animation frame subsample: {subsample} (frames={n_anim_frames})")

        fig_2d, axes_2d = plt.subplots(2, 3, figsize=(16, 9))

        def _loader_mem(i):
            return (
                E_frames[i],
                SAR_frames[i],
                Temperature_frames[i] if has_temperature else None,
            )

        _update_2d_multipanel = make_update_6panel_efield_sar(
            axes_2d,
            fig_2d,
            _loader_mem,
            frame_indices,
            fp_ax,
            fp_sag,
            fp_cor,
            _contour_tumor_footprint,
            len(E_frames),
        )

        animation_name = f"{output_base}_efield_sar_2d.mp4"
        fig_2d.tight_layout()
        ani_2d = animation.FuncAnimation(
            fig_2d,
            _update_2d_multipanel,
            frames=n_anim_frames,
            interval=20,
            blit=False,
            repeat=True,
            repeat_delay=1000,
        )
        print("\nSaving 2D 3×3 animation as video...")
        ani_2d.save(
            os.path.join(animations_dir, animation_name),
            writer="ffmpeg",
            fps=60,
        )
        plt.close(fig_2d)
        print(f"  Saved 2D animation ({animation_name})")
        sample_data = E_frames[0]

        print(
            "\nCreating animated 3D isometric views (E-field, SAR, and Temperature)..."
        )
        nx_3d, ny_3d = sample_data.shape[0], sample_data.shape[1]
        x_coords = np.arange(0, nx_3d, step_3d)
        y_coords = np.arange(0, ny_3d, step_3d)
        X_3d, Y_3d = np.meshgrid(y_coords, x_coords)
        x_min, x_max = 0, ny_3d
        y_min, y_max = 0, nx_3d
        z_min, z_max = 0, e_max
        sar_z_min, sar_z_max = 0, sar_max
        temp_z_min, temp_z_max = temp_min, temp_max

        def update_3d_e(anim_idx):
            ax_3d_e.clear()
            fn = frame_indices[anim_idx]
            e_data = np.abs(E_frames[fn])
            e_projection = np.max(e_data, axis=2)
            e_projection_3d = e_projection[::step_3d, ::step_3d]
            ax_3d_e.plot_surface(
                X_3d,
                Y_3d,
                e_projection_3d,
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
            ax_3d_e.set_title(
                f"E-field (Ez) 3D Isometric - Frame {fn + 1}/{len(E_frames)}"
            )
            ax_3d_e.view_init(elev=30, azim=45)
            ax_3d_e.set_xlim(x_min, x_max)
            ax_3d_e.set_ylim(y_min, y_max)
            ax_3d_e.set_zlim(z_min, z_max)

        fig_3d_e = plt.figure(figsize=(12, 10))
        ax_3d_e = fig_3d_e.add_subplot(111, projection="3d")
        ani_3d_e = animation.FuncAnimation(
            fig_3d_e,
            update_3d_e,
            frames=n_anim_frames,
            interval=20,
            blit=False,
            repeat=True,
            repeat_delay=1000,
        )
        print("Saving 3D E-field animation as video...")
        ani_3d_e.save(
            os.path.join(animations_dir, f"{output_base}_efield_3d.mp4"),
            writer="ffmpeg",
            fps=60,
        )
        plt.close(fig_3d_e)

        def update_3d_sar(anim_idx):
            ax_3d_sar.clear()
            fn = frame_indices[anim_idx]
            sar_data = SAR_frames[fn]
            sar_projection = np.max(sar_data, axis=2)
            sar_projection_3d = sar_projection[::step_3d, ::step_3d]
            ax_3d_sar.plot_surface(
                X_3d,
                Y_3d,
                sar_projection_3d,
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
            ax_3d_sar.set_title(
                f"SAR Distribution 3D Isometric - Frame {fn + 1}/{len(SAR_frames)}"
            )
            ax_3d_sar.view_init(elev=30, azim=45)
            ax_3d_sar.set_xlim(x_min, x_max)
            ax_3d_sar.set_ylim(y_min, y_max)
            ax_3d_sar.set_zlim(sar_z_min, sar_z_max)

        fig_3d_sar = plt.figure(figsize=(12, 10))
        ax_3d_sar = fig_3d_sar.add_subplot(111, projection="3d")
        ani_3d_sar = animation.FuncAnimation(
            fig_3d_sar,
            update_3d_sar,
            frames=n_anim_frames,
            interval=20,
            blit=False,
            repeat=True,
            repeat_delay=1000,
        )
        print("Saving 3D SAR animation as video...")
        ani_3d_sar.save(
            os.path.join(animations_dir, f"{output_base}_sar_3d.mp4"),
            writer="ffmpeg",
            fps=60,
        )
        plt.close(fig_3d_sar)
        if has_temperature and Temperature_frames:

            def update_3d_temp(anim_idx):
                ax_3d_temp.clear()
                fn = frame_indices[anim_idx]
                temp_data = Temperature_frames[fn]
                temp_projection = np.max(temp_data, axis=2)
                temp_projection_3d = temp_projection[::step_3d, ::step_3d]
                ax_3d_temp.plot_surface(
                    X_3d,
                    Y_3d,
                    temp_projection_3d,
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
                    f"Temperature 3D Isometric - Frame {fn + 1}/{len(Temperature_frames)}"
                )
                ax_3d_temp.view_init(elev=30, azim=45)
                ax_3d_temp.set_xlim(x_min, x_max)
                ax_3d_temp.set_ylim(y_min, y_max)
                ax_3d_temp.set_zlim(temp_z_min, temp_z_max)

            fig_3d_temp = plt.figure(figsize=(12, 10))
            ax_3d_temp = fig_3d_temp.add_subplot(111, projection="3d")
            ani_3d_temp = animation.FuncAnimation(
                fig_3d_temp,
                update_3d_temp,
                frames=n_anim_frames,
                interval=20,
                blit=False,
                repeat=True,
                repeat_delay=1000,
            )
            print("Saving 3D Temperature animation as video...")
            ani_3d_temp.save(
                os.path.join(animations_dir, f"{output_base}_temperature_3d.mp4"),
                writer="ffmpeg",
                fps=60,
            )
            plt.close(fig_3d_temp)
    else:
        fig_anim = plt.figure(figsize=(12, 10))
        ax_e = fig_anim.add_subplot(1, 1, 1)
        frames_art = []
        frame_indices = list(range(0, len(E_frames), subsample))
        n_anim_frames = len(frame_indices)
        print(f"  Animation frame subsample: {subsample} (frames={n_anim_frames})")
        for anim_idx, frame_idx in enumerate(frame_indices):
            e_data = np.abs(E_frames[frame_idx])
            e_projection = np.max(e_data, axis=2)
            im_e = ax_e.imshow(
                e_projection,
                cmap="jet",
                origin="lower",
                vmin=0,
                vmax=e_max,
                animated=True,
            )
            ax_e.set_title(f"E-field (Ez) - Frame {frame_idx + 1}/{len(E_frames)}")
            ax_e.set_xlabel("Y (cells)")
            ax_e.set_ylabel("X (cells)")
            frames_art.append([im_e])
        ani = animation.ArtistAnimation(
            fig_anim, frames_art, interval=20, blit=True, repeat_delay=1000
        )
        print("\nSaving 2D animation as video...")
        ani.save(
            os.path.join(animations_dir, f"{output_base}_efield_2d.mp4"),
            writer="ffmpeg",
            fps=60,
        )
        plt.close(fig_anim)
        nx_3d, ny_3d = E_frames[0].shape[0], E_frames[0].shape[1]
        x_coords = np.arange(0, nx_3d, step_3d)
        y_coords = np.arange(0, ny_3d, step_3d)
        X_3d, Y_3d = np.meshgrid(y_coords, x_coords)
        x_min, x_max = 0, ny_3d
        y_min, y_max = 0, nx_3d
        z_min, z_max = 0, e_max

        def update_3d_e(anim_idx):
            ax_3d_e.clear()
            fn = frame_indices[anim_idx]
            e_data = np.abs(E_frames[fn])
            e_projection = np.max(e_data, axis=2)
            e_projection_3d = e_projection[::step_3d, ::step_3d]
            ax_3d_e.plot_surface(
                X_3d,
                Y_3d,
                e_projection_3d,
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
            ax_3d_e.set_title(
                f"E-field (Ez) 3D Isometric - Frame {fn + 1}/{len(E_frames)}"
            )
            ax_3d_e.view_init(elev=30, azim=45)
            ax_3d_e.set_xlim(x_min, x_max)
            ax_3d_e.set_ylim(y_min, y_max)
            ax_3d_e.set_zlim(z_min, z_max)

        fig_3d_e = plt.figure(figsize=(12, 10))
        ax_3d_e = fig_3d_e.add_subplot(111, projection="3d")
        ani_3d_e = animation.FuncAnimation(
            fig_3d_e,
            update_3d_e,
            frames=n_anim_frames,
            interval=20,
            blit=False,
            repeat=True,
            repeat_delay=1000,
        )
        print("Saving 3D E-field animation as video...")
        ani_3d_e.save(
            os.path.join(animations_dir, f"{output_base}_efield_3d.mp4"),
            writer="ffmpeg",
            fps=60,
        )
        plt.close(fig_3d_e)
    return time.perf_counter() - t_start_anim
