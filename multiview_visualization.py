"""
Multi-view FDTD visualization: 15-timestep Ez/SAR timeline montages, unified 3×3
SAR/temperature/geometry plate, per-view SAR and temperature max-projection maps, and
2×3 E/SAR propagation animations (axial / sagittal / coronal).

Uses 99.5th-percentile scaling per 2D max-projection map (per frame for animations).
Static timeline montages are generated for E-field and SAR only (not temperature).
"""

from __future__ import annotations

import os
from typing import Callable, Sequence

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

N_TIME_SNAPSHOTS = 15
T_DISPLAY_MIN_C = 37.0


def _temperature_absolute_colormap_bounds(
    temp_3d: np.ndarray, *, T_ref: float = T_DISPLAY_MIN_C
) -> tuple[float, float, str]:
    """
    vmin/vmax in °C for absolute ``T`` maps (matches ``save_temperature_distribution``).
    Caps rare hot voxels via excess-at-reference p99.5 when needed.
    """
    if not np.size(temp_3d):
        z = T_ref
        return z, z + max(abs(z) * 1e-12, 1e-15), "neither"
    tw = np.asarray(temp_3d, dtype=np.float64).ravel()
    T_lo = float(np.min(tw))
    T_hi = float(np.max(tw))
    heat = np.maximum(tw - T_ref, 0.0)
    hpos = heat[heat > 0]
    peak_ex = float(np.max(heat)) if heat.size else 0.0
    if hpos.size >= 64:
        e995 = float(np.percentile(hpos, 99.5))
    elif hpos.size:
        e995 = float(np.max(hpos))
    else:
        e995 = peak_ex
    if peak_ex > e995 * 1.5 and e995 > 0:
        vmax = T_ref + e995
        ext = "max"
    else:
        vmax = T_hi
        ext = "neither"
    vmin = T_lo
    if vmax <= vmin:
        vmax = vmin + max(abs(vmin) * 1e-12, 1e-15)
    return vmin, vmax, ext


def _colorbar_ticks_absolute_T(cb, vmin_t: float, vmax_t: float) -> None:
    """Narrow span: ΔT = T − T_ref in sci notation on ticks; wide span: absolute T."""
    from matplotlib import ticker

    span = float(vmax_t) - float(vmin_t)
    if not np.isfinite(span):
        span = 0.0

    if span >= 0.25:
        cb.locator = ticker.MaxNLocator(nbins=8, min_n_ticks=4)
        cb.update_ticks()
        fmt = ticker.ScalarFormatter(useMathText=False, useOffset=False)
        fmt.set_powerlimits((-12, 12))
        cb.ax.yaxis.set_major_formatter(fmt)
        return

    lo, hi = float(vmin_t), float(vmax_t)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-12, 1e-15)
        span = hi - lo
    t_ref = lo
    ticks = np.linspace(lo, hi, 7, dtype=np.float64)
    cb.locator = ticker.FixedLocator(ticks)
    cb.update_ticks()

    def _fmt_delta(x, _p):
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

# Formalized multiview artifact names (brain pipeline / build_animations static export):
#   {base}_timeline15_Ez_{axial_maxz|sagittal_maxx|coronal_maxy}.png
#   {base}_timeline15_SAR_{...}.png
#   {base}_sar_maxproj_p995_{...}.png
#   {base}_temp_maxproj_p995_{...}.png  (when temperature is available)
VIEW_SUFFIXES: tuple[str, ...] = (
    "axial_maxz",
    "sagittal_maxx",
    "coronal_maxy",
)


def equal_spaced_target_timesteps(
    time_steps: int, n: int = N_TIME_SNAPSHOTS
) -> list[int]:
    """Return ``n`` timesteps in [1, time_steps], approximately equally spaced (integer, unique)."""
    if time_steps < 1:
        return []
    if n < 1:
        return []
    raw = np.linspace(1, float(time_steps), num=n)
    idx = np.unique(np.clip(np.round(raw).astype(np.int64), 1, time_steps))
    out = idx.tolist()
    # Ensure length n by adding/removing at end if linspace collapsed (tiny T)
    while len(out) < n and len(out) < time_steps:
        cand = min(time_steps, out[-1] + 1) if out else 1
        if cand in out:
            break
        out.append(cand)
    while len(out) > n:
        out.pop(len(out) // 2)
    return out[:n]


def saved_timesteps_for_run(
    time_steps: int,
    n_frames: int,
    stream_frames: bool,
    stream_frame_interval: int,
    frame_interval: int,
    use_quadrant_loop: bool,
) -> list[int]:
    """
    Reconstruct FDTD timestep index for each saved frame (1-based), matching core loops.

    Quadrant / optimized path uses ``frame_interval`` (streaming or ``max(1, T//350)``).
    Standard path uses ``stream_frame_interval`` when streaming, else every step.
    """
    if n_frames <= 0 or time_steps < 1:
        return []
    if use_quadrant_loop:
        interval = max(1, int(frame_interval))
        ts = [t for t in range(1, time_steps + 1) if t % interval == 0]
    elif stream_frames:
        interval = max(1, int(stream_frame_interval))
        ts = [t for t in range(1, time_steps + 1) if t % interval == 0]
    else:
        ts = list(range(1, time_steps + 1))
    if len(ts) > n_frames:
        ts = ts[:n_frames]
    if len(ts) < n_frames:
        while len(ts) < n_frames and ts:
            ts.append(ts[-1])
    return ts[:n_frames]


def map_target_timesteps_to_frame_indices(
    target_timesteps: Sequence[int], saved_timesteps: Sequence[int]
) -> list[int]:
    """For each target, pick saved frame index whose timestep is closest."""
    saved = np.asarray(saved_timesteps, dtype=np.int64)
    if saved.size == 0:
        return [0] * len(target_timesteps)
    out: list[int] = []
    for t in target_timesteps:
        j = int(np.argmin(np.abs(saved - int(t))))
        out.append(j)
    return out


def _max_axial(vol: np.ndarray) -> np.ndarray:
    return np.max(vol, axis=2)


def _max_sagittal(vol: np.ndarray) -> np.ndarray:
    return np.max(vol, axis=0)


def _max_coronal(vol: np.ndarray) -> np.ndarray:
    return np.max(vol, axis=1)


def _orient_sagittal(arr: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(arr, k=1))


def _orient_coronal(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=-1)


def p995_vmax(arr: np.ndarray, floor: float = 1e-20) -> float:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return floor
    return max(float(np.percentile(a, 99.5)), floor)


def _imshow_p995(
    ax,
    arr: np.ndarray,
    cmap: str,
    title: str,
    *,
    floor: float = 0.0,
    cbar_label: str,
    clip_floor: bool = True,
):
    vmax = p995_vmax(arr, floor=1e-20 if floor == 0 else floor)
    vmin = floor
    im = ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    ax.set_title(title, fontsize=9)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4.5%", pad=0.04)
    plt.colorbar(im, cax=cax, label=cbar_label)
    return im


def _tumor_mask_maxproj(labels: np.ndarray, axis: int) -> np.ndarray:
    m = ((labels >= 1) & (labels <= 3)).astype(np.float32)
    return np.max(m, axis=axis)


def write_nine_timeline_montages(
    get_frame: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_indices: list[int],
    target_timesteps: list[int],
    saved_timesteps: list[int],
    output_base: str,
    images_dir: str,
    *,
    has_temperature: bool,
):
    """
    Six PNGs: for each modality (E, SAR) × projection (axial z-max, sagittal x-max,
    coronal y-max), a 3×5 grid of 15 timesteps (p99.5 scaling per subplot).
    """
    os.makedirs(images_dir, exist_ok=True)
    # has_temperature kept for callers (export_all_multiview_static); T timelines not emitted.
    _ = has_temperature
    names = [
        ("E", "Ez", "V/m (stored Ez magnitude)"),
        ("SAR", "SAR", "SAR (W/kg)"),
    ]
    for mi, (tag, fname_part, cbar_e) in enumerate(names):
        for pi, (proj_name, proj_fn, orient_fn, cbar_label) in enumerate(
            [
                (
                    "axial_maxz",
                    _max_axial,
                    None,
                    "|Ez| (p99.5)" if tag == "E" else "SAR (p99.5)",
                ),
                ("sagittal_maxx", _max_sagittal, _orient_sagittal, cbar_e),
                ("coronal_maxy", _max_coronal, _orient_coronal, cbar_e),
            ]
        ):
            fig, axes = plt.subplots(3, 5, figsize=(18, 10))
            axes = np.atleast_2d(axes)
            for k, t_step in enumerate(target_timesteps):
                fi = frame_indices[k] if k < len(frame_indices) else frame_indices[-1]
                e, sar, _temp = get_frame(fi)
                vol = np.abs(e) if tag == "E" else sar
                proj = proj_fn(vol)
                if orient_fn is not None:
                    proj = orient_fn(proj)
                row, col = k // 5, k % 5
                ax = axes[row, col]
                vmax = p995_vmax(proj)
                im = ax.imshow(
                    proj,
                    cmap="jet" if tag == "E" else "coolwarm",
                    vmin=0.0,
                    vmax=vmax,
                    origin="lower",
                )
                ax.set_title(
                    f"target t={target_timesteps[k]}\nsaved t={saved_timesteps[fi] if fi < len(saved_timesteps) else fi}",
                    fontsize=7,
                )
                div = make_axes_locatable(ax)
                cax = div.append_axes("right", size="3%", pad=0.02)
                plt.colorbar(im, cax=cax, fraction=1.0)
                ax.axis("off")
            for k in range(len(target_timesteps), 15):
                row, col = k // 5, k % 5
                axes[row, col].axis("off")
            st0 = target_timesteps[0] if target_timesteps else 0
            st1 = target_timesteps[-1] if target_timesteps else 0
            fig.suptitle(
                f"{output_base} — {fname_part} — {proj_name} — 15 timesteps ({st0}…{st1})",
                fontsize=11,
            )
            plt.tight_layout()
            out = os.path.join(
                images_dir,
                f"{output_base}_timeline15_{fname_part}_{proj_name}.png",
            )
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)


def write_unified_sar_temp_geometry_3x3(
    sar_3d: np.ndarray,
    temp_3d: np.ndarray,
    labels_3d: np.ndarray,
    output_base: str,
    images_dir: str,
    *,
    has_temperature: bool,
):
    """One 3×3 figure: rows SAR, Temperature, tumor geometry; cols axial, sagittal, coronal."""
    os.makedirs(images_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    def _sar_temp_row(row: int, vol: np.ndarray, is_temp: bool):
        projs = (_max_axial(vol), _max_sagittal(vol), _max_coronal(vol))
        orients = (None, _orient_sagittal, _orient_coronal)
        titles = (
            "SAR axial (max z, p99.5)",
            "SAR sagittal (max x, p99.5)",
            "SAR coronal (max y, p99.5)",
        )
        if is_temp:
            titles = (
                "T axial (max z)",
                "T sagittal (max x)",
                "T coronal (max y)",
            )
        vmin_t = vmax_t = 1.0
        cb_ex = "neither"
        if is_temp:
            vmin_t, vmax_t, cb_ex = _temperature_absolute_colormap_bounds(vol)
        span_t = vmax_t - vmin_t if is_temp else 1.0
        if is_temp and span_t > 0 and span_t < 0.25:
            t_norm: mcolors.Normalize | None = mcolors.PowerNorm(
                gamma=0.35,
                vmin=vmin_t,
                vmax=vmax_t,
                clip=(cb_ex == "max"),
            )
        elif is_temp:
            t_norm = mcolors.Normalize(
                vmin=vmin_t, vmax=vmax_t, clip=(cb_ex == "max")
            )
        else:
            t_norm = None
        for c in range(3):
            p = projs[c]
            if orients[c] is not None:
                p = orients[c](p)
            ax = axes[row, c]
            if is_temp:
                im = ax.imshow(
                    p.astype(np.float64),
                    cmap="coolwarm",
                    norm=t_norm,
                    origin="lower",
                )
            else:
                vmax = p995_vmax(p)
                im = ax.imshow(p, cmap="coolwarm", vmin=0.0, vmax=vmax, origin="lower")
            ax.set_title(titles[c], fontsize=9)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="3.5%", pad=0.03)
            lab = r"$T$ ($^\circ$C)" if is_temp else "SAR (W/kg)"
            cb_panel = plt.colorbar(
                im,
                cax=cax,
                label=lab,
                extend=cb_ex if is_temp else "neither",
            )
            if is_temp:
                _colorbar_ticks_absolute_T(cb_panel, vmin_t, vmax_t)

    _sar_temp_row(0, sar_3d, is_temp=False)
    if has_temperature:
        _sar_temp_row(1, temp_3d, is_temp=True)
    else:
        for c in range(3):
            axes[1, c].text(0.5, 0.5, "No temperature", ha="center", va="center")
            axes[1, c].axis("off")

    for c, (axis, orient, ttl) in enumerate(
        [
            (2, None, "Geometry axial (tumor max-z)"),
            (0, _orient_sagittal, "Geometry sagittal (tumor max-x)"),
            (1, _orient_coronal, "Geometry coronal (tumor max-y)"),
        ]
    ):
        m = _tumor_mask_maxproj(labels_3d, axis=axis)
        # Orient first (may swap H/W), then allocate RGB to match.
        if orient is not None:
            m = orient(m)
        pic = np.zeros((*m.shape, 3), dtype=np.float32)
        pic[:, :, 1] = m  # green tumor mask
        ax = axes[2, c]
        ax.imshow(pic, origin="lower")
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"{output_base} — SAR | Temperature | Tumor geometry (max projections)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(images_dir, f"{output_base}_unified_sar_temp_geometry_3x3.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _colorbar_whole_ticks_sci_offset(cbar) -> None:
    """Prefer integer-like ticks; enable offset/scientific notation when Matplotlib chooses it."""
    from matplotlib import ticker

    cbar.ax.yaxis.set_major_locator(
        ticker.MaxNLocator(nbins=8, min_n_ticks=3, integer=True)
    )
    fmt = ticker.ScalarFormatter(useMathText=True, useOffset=True)
    fmt.set_powerlimits((-2, 4))
    cbar.ax.yaxis.set_major_formatter(fmt)


def write_sar_temp_maxproj_views(
    sar_3d: np.ndarray,
    temp_3d: np.ndarray,
    output_base: str,
    images_dir: str,
    *,
    has_temperature: bool,
) -> None:
    """
    Max-projection maps per anatomical view: SAR with p99.5 display cap; temperature
    in absolute °C with vmin/vmax from the full 3D volume (consistent with distribution PNGs).
    """
    os.makedirs(images_dir, exist_ok=True)
    views = [
        (VIEW_SUFFIXES[0], _max_axial, None),
        (VIEW_SUFFIXES[1], _max_sagittal, _orient_sagittal),
        (VIEW_SUFFIXES[2], _max_coronal, _orient_coronal),
    ]
    for view_name, proj_fn, orient_fn in views:
        sar_p = proj_fn(sar_3d)
        if orient_fn is not None:
            sar_p = orient_fn(sar_p)
        vmax = p995_vmax(sar_p)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(sar_p, cmap="coolwarm", vmin=0.0, vmax=vmax, origin="lower")
        ax.set_title(
            f"SAR max projection — {view_name}\n(p99.5 display cap)", fontsize=10
        )
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.06)
        cb = plt.colorbar(im, cax=cax, label="SAR (W/kg)")
        _colorbar_whole_ticks_sci_offset(cb)
        fig.suptitle(output_base, fontsize=11, y=1.02)
        plt.tight_layout()
        fig.savefig(
            os.path.join(images_dir, f"{output_base}_sar_maxproj_p995_{view_name}.png"),
            dpi=140,
            bbox_inches="tight",
        )
        plt.close(fig)

        if not has_temperature:
            continue
        temp_p = proj_fn(temp_3d)
        if orient_fn is not None:
            temp_p = orient_fn(temp_p)
        vmin_g, vmax_g, cb_ex = _temperature_absolute_colormap_bounds(temp_3d)
        span_g = vmax_g - vmin_g
        if span_g > 0 and span_g < 0.25:
            tnorm = mcolors.PowerNorm(
                gamma=0.35,
                vmin=vmin_g,
                vmax=vmax_g,
                clip=(cb_ex == "max"),
            )
        else:
            tnorm = mcolors.Normalize(
                vmin=vmin_g, vmax=vmax_g, clip=(cb_ex == "max")
            )
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(
            temp_p.astype(np.float64),
            cmap="coolwarm",
            norm=tnorm,
            origin="lower",
        )
        ax.set_title(
            f"Temperature max projection — {view_name}\n"
            r"(same absolute $T$ scale as full volume)",
            fontsize=10,
        )
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="4%", pad=0.06)
        cb = plt.colorbar(
            im,
            cax=cax,
            label=r"$T$ ($^\circ$C)",
            extend=cb_ex,
        )
        _colorbar_ticks_absolute_T(cb, vmin_g, vmax_g)
        fig.suptitle(output_base, fontsize=11, y=1.02)
        plt.tight_layout()
        fig.savefig(
            os.path.join(
                images_dir, f"{output_base}_temp_maxproj_p995_{view_name}.png"
            ),
            dpi=140,
            bbox_inches="tight",
        )
        plt.close(fig)


def make_update_6panel_efield_sar(
    axes_2x3,
    fig,
    loader_frame_getter: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    frame_indices: list[int],
    tumor_footprint_2d,
    tumor_footprint_sagittal,
    tumor_footprint_coronal,
    contour_fn,
    n_frames_total: int,
):
    """
    Build FuncAnimation update callable: 2×3 panels — rows |Ez| and SAR;
    columns axial / sagittal / coronal max projections (p99.5 for both).
    """

    def _foot(ax, fp2d, color):
        if fp2d is not None and np.any(fp2d):
            contour_fn(ax, fp2d, color)

    def update(anim_idx: int):
        frame_idx = frame_indices[anim_idx]
        e, sar, _temp = loader_frame_getter(frame_idx)
        e_abs = np.abs(e)
        projs_e = (_max_axial(e_abs), _max_sagittal(e_abs), _max_coronal(e_abs))
        projs_sar = (_max_axial(sar), _max_sagittal(sar), _max_coronal(sar))
        footprints = (
            tumor_footprint_2d,
            tumor_footprint_sagittal,
            tumor_footprint_coronal,
        )
        orients = (None, _orient_sagittal, _orient_coronal)
        names = ("|Ez|", "SAR")
        projs_rows = (projs_e, projs_sar)
        cmaps = ("jet", "coolwarm")

        for r in range(2):
            for c in range(3):
                ax = axes_2x3[r, c]
                ax.clear()
                proj = projs_rows[r][c]
                if orients[c] is not None:
                    proj = orients[c](proj)
                fp = footprints[c]
                vmax = p995_vmax(proj)
                im = ax.imshow(
                    proj,
                    cmap=cmaps[r],
                    vmin=0.0,
                    vmax=max(vmax, 1e-20),
                    origin="lower",
                )
                _foot(ax, fp, "lime" if r == 0 else "cyan")
                ttl = (
                    f"{names[r]} axial (max z) p99.5",
                    f"{names[r]} sagittal (max x) p99.5",
                    f"{names[r]} coronal (max y) p99.5",
                )[c]
                ax.set_title(
                    f"{ttl}\nframe {frame_idx + 1}/{n_frames_total}", fontsize=8
                )
        fig.suptitle(
            "E (|Ez|) / SAR — axial (max z), sagittal (max x), coronal (max y); p99.5",
            fontsize=10,
            y=0.99,
        )
        return []

    return update


def tumor_footprints_three_views(labels_3d: np.ndarray):
    """Binary tumor max projections for contour on axial / sagittal / coronal panels."""
    m = ((labels_3d >= 1) & (labels_3d <= 3)).astype(np.float32)
    axial = np.max(m, axis=2)
    sag = _orient_sagittal(np.max(m, axis=0))
    cor = _orient_coronal(np.max(m, axis=1))
    return axial, sag, cor


def export_multiview_plates_from_volumes(
    sar_3d: np.ndarray,
    temp_3d: np.ndarray,
    labels_3d: np.ndarray,
    output_base: str,
    images_dir: str,
    *,
    has_temperature: bool,
) -> None:
    """
    Unified 3×3 and SAR/temperature max-projection PNGs only (no 15-frame timelines).
    Used when refreshing paper figures from NIfTI without E-frame NPZs.
    """
    os.makedirs(images_dir, exist_ok=True)
    write_unified_sar_temp_geometry_3x3(
        sar_3d,
        temp_3d,
        labels_3d,
        output_base,
        images_dir,
        has_temperature=has_temperature,
    )
    write_sar_temp_maxproj_views(
        sar_3d,
        temp_3d,
        output_base,
        images_dir,
        has_temperature=has_temperature,
    )


def export_all_multiview_static(
    *,
    get_frame: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    saved_timesteps: list[int],
    time_steps: int,
    labels_3d: np.ndarray,
    sar_3d: np.ndarray,
    temp_3d: np.ndarray,
    output_base: str,
    images_dir: str,
    has_temperature: bool,
):
    """Ez/SAR 15-timeline montages, unified 3×3, SAR/temperature max-projection views."""
    targets = equal_spaced_target_timesteps(time_steps, N_TIME_SNAPSHOTS)
    fidx = map_target_timesteps_to_frame_indices(targets, saved_timesteps)
    write_nine_timeline_montages(
        get_frame,
        fidx,
        targets,
        saved_timesteps,
        output_base,
        images_dir,
        has_temperature=has_temperature,
    )
    write_unified_sar_temp_geometry_3x3(
        sar_3d,
        temp_3d,
        labels_3d,
        output_base,
        images_dir,
        has_temperature=has_temperature,
    )
    write_sar_temp_maxproj_views(
        sar_3d,
        temp_3d,
        output_base,
        images_dir,
        has_temperature=has_temperature,
    )
