"""
Brain simulation workflow: segmentation → grid/materials → optional antenna optimization
→ FDTD → SAR → thermal → save/validate/animations.
"""

from math import sqrt
import json
import os
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sar_computation import compute_robust_objective, compute_sar
from thermal_solver import solve_steady_bioheat_3d
from config.anatomy_config import get_anatomy_config
from voxel_model import K_TISSUE, TISSUE_TABLE, build_material_arrays
from antenna_optimization import run_antenna_optimization_block
from performance_logging import get_peak_memory_mb
from segmentation_loader import (
    load_segmentation_and_output_base,
    _find_modalities_in_dir,
)
from multiview_visualization import saved_timesteps_for_run
from data_analysis_validation import (
    build_and_save_animations,
    save_geometry_slice,
    save_sar_distribution,
    save_simulation_data,
    save_slice_anatomy_sar_temperature,
    save_temperature_distribution,
    save_tumor_preview,
)
from core.fdtd.sources import build_quadrant_sources
from fdtd_solver import (
    calculate_pml_parameters,
    run_fdtd_optimized_loop,
    run_fdtd_standard_loop,
)


def build_antenna_parameters_metadata(
    args,
    pulse_amplitude,
    pulse_type,
    prop_direction,
    pulse_freq,
    pulse_ramp_width,
    source_x,
    source_y,
    source_z,
    point_source_positions,
    opt_f0,
    opt_alphas,
    opt_thetas,
    opt_quad_sources,
):
    """
    Unified excitation/antenna record written to metadata for every run (dashboard + reproducibility).
    """
    out = {"pulse_amplitude": float(pulse_amplitude)}
    if args.optimize_antenna and opt_f0 is not None:
        out["excitation_mode"] = "four_quadrant_optimized"
        out["carrier_frequency_Hz"] = float(opt_f0)
        out["opt_source_scale"] = float(getattr(args, "opt_source_scale", 1.0))
        if opt_alphas is not None:
            out["quadrant_amplitudes"] = np.asarray(opt_alphas).astype(float).tolist()
        if opt_thetas is not None:
            out["quadrant_phases_rad"] = np.asarray(opt_thetas).astype(float).tolist()
        if opt_quad_sources:
            out["quadrant_gap_cells"] = [qs["gap"] for qs in opt_quad_sources]
        out["dipole_half_length_cells"] = int(
            getattr(args, "fixed_quadrant_dipole_half_len", 9)
        )
        return out
    if getattr(args, "quadrant_fixed", False) and opt_f0 is not None:
        out["excitation_mode"] = "four_quadrant_fixed"
        out["carrier_frequency_Hz"] = float(opt_f0)
        out["opt_source_scale"] = float(getattr(args, "opt_source_scale", 1.0))
        if opt_alphas is not None:
            out["quadrant_amplitudes"] = np.asarray(opt_alphas).astype(float).tolist()
        if opt_thetas is not None:
            out["quadrant_phases_rad"] = np.asarray(opt_thetas).astype(float).tolist()
        if opt_quad_sources:
            out["quadrant_gap_cells"] = [qs["gap"] for qs in opt_quad_sources]
        out["dipole_half_length_cells"] = int(
            getattr(args, "fixed_quadrant_dipole_half_len", 9)
        )
        return out

    out["excitation_mode"] = "volume_pulse"
    out["pulse_type"] = pulse_type
    out["propagation_direction"] = prop_direction
    out["pulse_frequency_Hz"] = float(pulse_freq)
    out["pulse_ramp_width"] = float(pulse_ramp_width)
    out["primary_source_voxel"] = [int(source_x), int(source_y), int(source_z)]
    if len(point_source_positions) > 1:
        out["point_source_positions"] = [
            [int(sx), int(sy), int(sz)] for (sx, sy, sz) in point_source_positions
        ]
    if getattr(args, "cw_periods", None) is not None:
        out["cw_periods"] = int(args.cw_periods)
    ro = getattr(args, "source_ring_offset", None)
    if ro is not None:
        out["source_ring_offset_cells"] = int(ro)
    if getattr(args, "use_source_2", False):
        out["use_second_ring_source"] = True
    return out


def run_simulation(args, paths, write_progress_cb, anatomy="brain"):
    """
    Single pipeline for all anatomies (brain, breast, cervix).
    Uses get_anatomy_config(anatomy) for tissue tables, visualization, and target/healthy labels.
    Segmentation loading is anatomy-specific (brain uses load_segmentation_and_output_base;
    breast/cervix use their loaders when wired).
    """
    RESULTS_DIR = paths.results_dir
    DATA_DIR = paths.data_dir
    IMAGES_DIR = paths.images_dir
    ANIMATIONS_DIR = paths.animations_dir
    E_FRAMES_DIR = paths.e_frames_dir
    SAR_FRAMES_DIR = paths.sar_frames_dir
    TEMPERATURE_FRAMES_DIR = paths.temperature_frames_dir
    _write_progress = write_progress_cb

    t_start_pipeline = time.perf_counter()
    t_start_segmentation = t_start_pipeline
    volume_4d_ds = None

    if anatomy == "brain":
        use_modalities = (args.modalities is not None) or (
            args.modalities_dir is not None
        )
        labels_3d, OUTPUT_BASE, t_end_segmentation = load_segmentation_and_output_base(
            args, use_modalities
        )
        if use_modalities:
            print(f"Segmentation shape: {labels_3d.shape}")
        _write_progress(
            "segmentation", "Segmentation complete", 10, ["setup", "segmentation"]
        )
        max_dim = args.max_dim
        nx, ny, nz = labels_3d.shape
        orig_shape = labels_3d.shape
        if max(nx, ny, nz) > max_dim:
            scale = max_dim / max(nx, ny, nz)
            order = 0
            labels_3d = ndimage.zoom(
                labels_3d, (scale, scale, scale), order=order, mode="nearest"
            )
            labels_3d = np.round(labels_3d).astype(np.int32)
            labels_3d = np.clip(labels_3d, 0, 4)
            nx, ny, nz = labels_3d.shape
            print(f"Downsampled segmentation to ({nx}, {ny}, {nz}) (max_dim={max_dim})")
        _write_progress(
            "setup",
            f"Grid shape {nx}×{ny}×{nz}",
            12,
            ["setup", "segmentation"],
            {"grid_shape": [int(nx), int(ny), int(nz)]},
        )
        from brain_tumor_segmentation_model import select_slices_biggest_tumor

        top_10_slice_indices = select_slices_biggest_tumor(
            labels_3d, n_slices=10, axis=2
        )
        print(
            f"Voxel-based tissue grid: top 10 axial slices by tumor area: {top_10_slice_indices}"
        )
        if use_modalities:
            from brain_tumor_segmentation_model import (
                load_patient_volume_from_paths,
                save_fifteen_slice_modality_montages,
            )

            if args.modalities_dir is not None:
                flair_path, t1_path, t1ce_path, t2_path = _find_modalities_in_dir(
                    args.modalities_dir
                )
            else:
                flair_path, t1_path, t1ce_path, t2_path = tuple(args.modalities)

            volume_4d = load_patient_volume_from_paths(
                flair_path, t1_path, t1ce_path, t2_path
            )
            zoom_factors = (
                1,
                nx / orig_shape[0],
                ny / orig_shape[1],
                nz / orig_shape[2],
            )
            volume_4d_ds = ndimage.zoom(
                volume_4d, zoom_factors, order=1, mode="nearest"
            )
            top_preview_slices = select_slices_biggest_tumor(
                labels_3d, n_slices=15, axis=2
            )
            montage_paths = save_fifteen_slice_modality_montages(
                volume_4d_ds,
                labels_3d,
                OUTPUT_BASE,
                IMAGES_DIR,
                top_preview_slices,
            )
            if montage_paths:
                print(
                    f"  Saved {len(montage_paths)} modality montages (3×5) to {IMAGES_DIR}/"
                )
    else:
        # breast or cervix: use anatomy-specific loader (already downsampled)
        use_modalities = False
        if anatomy == "cervix":
            from segmentation_loader import load_cervix_segmentation_for_pipeline

            labels_3d, OUTPUT_BASE, t_end_segmentation = (
                load_cervix_segmentation_for_pipeline(args)
            )
        else:
            from segmentation_loader import load_breast_segmentation_for_pipeline

            labels_3d, OUTPUT_BASE, t_end_segmentation = (
                load_breast_segmentation_for_pipeline(args)
            )
        _write_progress(
            "segmentation", "Segmentation complete", 10, ["setup", "segmentation"]
        )
        nx, ny, nz = labels_3d.shape
        _write_progress(
            "setup",
            f"Grid shape {nx}×{ny}×{nz}",
            12,
            ["setup", "segmentation"],
            {"grid_shape": [int(nx), int(ny), int(nz)]},
        )
        ac_temp = get_anatomy_config(anatomy)
        target_mask = np.isin(labels_3d, ac_temp.target_labels)
        n_per_slice = np.sum(target_mask, axis=(0, 1))
        top_10_slice_indices = np.argsort(n_per_slice)[-10:][::-1].tolist()
        print(f"Top 10 axial slices by target area: {top_10_slice_indices}")

    air_padding_cells = max(0, int(getattr(args, "air_padding_cells", 0)))
    if air_padding_cells > 0:
        labels_3d = np.pad(
            labels_3d,
            (
                (air_padding_cells, air_padding_cells),
                (air_padding_cells, air_padding_cells),
                (air_padding_cells, air_padding_cells),
            ),
            mode="constant",
            constant_values=0,
        )
        if use_modalities and volume_4d_ds is not None:
            # Keep anatomy preview overlays aligned with padded segmentation.
            volume_4d_ds = np.pad(
                volume_4d_ds,
                (
                    (0, 0),
                    (air_padding_cells, air_padding_cells),
                    (air_padding_cells, air_padding_cells),
                    (air_padding_cells, air_padding_cells),
                ),
                mode="constant",
                constant_values=0.0,
            )
        print(
            f"Applied air padding: {air_padding_cells} cells/side "
            f"-> padded grid shape {labels_3d.shape[0]}x{labels_3d.shape[1]}x{labels_3d.shape[2]}"
        )

    # Any geometry padding changes the simulation domain shape and slice indexing.
    nx, ny, nz = labels_3d.shape
    ac_post_pad = get_anatomy_config(anatomy)
    target_mask_post_pad = np.isin(labels_3d, ac_post_pad.target_labels)
    n_per_slice_post_pad = np.sum(target_mask_post_pad, axis=(0, 1))
    top_10_slice_indices = np.argsort(n_per_slice_post_pad)[-10:][::-1].tolist()

    slice_indices_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_slice_indices.txt")
    with open(slice_indices_path, "w") as f:
        f.write("\n".join(map(str, top_10_slice_indices)))
    print(f"Slice indices saved to {slice_indices_path}")

    simulation_size_x = int(nx)
    simulation_size_y = int(ny)
    simulation_size_z = int(nz)
    mid_z_index = simulation_size_z // 2
    print(
        f"Simulation grid: {simulation_size_x} x {simulation_size_y} x {simulation_size_z}"
    )

    anatomy_config = get_anatomy_config(anatomy)
    tissue_table = anatomy_config.tissue_table
    k_tissue = anatomy_config.k_tissue
    viz_config = anatomy_config.visualization_config
    tumor_mask = np.isin(labels_3d, anatomy_config.target_labels)
    tumor_region = tumor_mask
    non_tumor_region = np.isin(labels_3d, anatomy_config.healthy_labels)

    save_tumor_preview(
        labels_3d,
        tumor_mask,
        OUTPUT_BASE,
        IMAGES_DIR,
        nx,
        ny,
        nz,
        viz_config=viz_config,
    )
    print(f"  Tumor preview (pre-FDTD) saved: {OUTPUT_BASE}_tumor_preview.png")

    npml = max(
        4, min(16, min(simulation_size_x, simulation_size_y, simulation_size_z) // 10)
    )
    source_x = npml + 2
    source_y = simulation_size_y // 2
    source_z = simulation_size_z // 2
    ia = npml
    ja = npml
    ka = npml
    ib = simulation_size_x - npml - 1
    jb = simulation_size_y - npml - 1
    kb = simulation_size_z - npml - 1

    dx = args.dx_mm * 1e-3
    c_light = 2.99792458e8
    dy, dz = dx, dx
    dt_courant = 1.0 / (
        c_light * sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    )
    dt = args.courant_factor * dt_courant
    epsz = 8.854e-12

    (
        eps_x,
        eps_y,
        eps_z,
        conductivity_x,
        conductivity_y,
        conductivity_z,
        sigma_x,
        sigma_y,
        sigma_z,
        rho,
        k_3d,
    ) = build_material_arrays(
        labels_3d, dt, epsz, tissue_table=tissue_table, k_tissue=k_tissue
    )
    T_BOUNDARY_CELSIUS = 37.0

    Ex = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ey = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ez = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ix = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Iy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Iz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Dx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Dy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Dz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iDx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iDy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iDz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Hx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Hy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Hz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iHx = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iHy = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    iHz = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    hx_inc = np.zeros(simulation_size_y)
    ez_inc = np.zeros(simulation_size_y)

    n_air = np.sum(labels_3d == 0)
    n_tumor = np.sum(tumor_region)
    n_normal_brain = np.sum(non_tumor_region)
    print(
        f"\nMaterial fill (ε, σ, ρ per voxel): air={n_air}, tumor={n_tumor}, normal brain={n_normal_brain}"
    )
    print(
        f"  ρ: [{np.min(rho):.1f}, {np.max(rho):.1f}] kg/m³  |  ε⁻¹ (eff.): [{np.min(eps_x):.6f}, {np.max(eps_x):.6f}]"
    )
    save_geometry_slice(
        labels_3d, OUTPUT_BASE, IMAGES_DIR, simulation_size_z, viz_config=viz_config
    )
    print(f"  Geometry slice (mid-Z) saved: {OUTPUT_BASE}_fdtd_geometry_slice.png")

    opt_quad_sources = None
    opt_alphas = None
    opt_thetas = None
    opt_f0 = None

    antenna_optimization_s = None
    do_geom_sweep = False
    if args.optimize_antenna:
        (
            opt_quad_sources,
            opt_alphas,
            opt_thetas,
            opt_f0,
            antenna_optimization_s,
            do_geom_sweep,
        ) = run_antenna_optimization_block(
            args,
            labels_3d,
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            npml,
            dx,
            dt,
            epsz,
            rho,
            sigma_x,
            sigma_y,
            sigma_z,
            eps_x,
            eps_y,
            eps_z,
            conductivity_x,
            conductivity_y,
            conductivity_z,
            OUTPUT_BASE,
            DATA_DIR,
            IMAGES_DIR,
            write_progress_cb=_write_progress,
            target_labels=anatomy_config.target_labels,
            healthy_labels=anatomy_config.healthy_labels,
            quadrant_air_margin_cells=int(
                getattr(args, "quadrant_air_margin_cells", 18)
            ),
        )
        print(
            "\nAntenna optimization complete. Running full time-domain FDTD with optimized 4-quadrant source...\n"
        )

    tumor_footprint_2d = np.max(tumor_region.astype(np.float32), axis=2)
    _tumor_mask_anim = tumor_region
    if np.any(_tumor_mask_anim):
        _tz = np.argwhere(_tumor_mask_anim)[:, 2]
        tumor_centroid_z_2d = int(round(np.mean(_tz)))
        tumor_centroid_z_2d = max(0, min(tumor_centroid_z_2d, simulation_size_z - 1))
    else:
        tumor_centroid_z_2d = simulation_size_z // 2

    if getattr(args, "quadrant_fixed", False):
        opt_f0 = float(args.f0)
        if getattr(args, "replay_loaded_quadrant_geometry", False):
            # Faithful replay from --load-optimized-from: use stored ring / z_plane exactly.
            _ring = int(getattr(args, "fixed_quadrant_ring_offset") or (npml + 2))
            _zpl = getattr(args, "fixed_quadrant_z_plane", None)
            if _zpl is None:
                _zpl = tumor_centroid_z_2d
            _zpl = int(_zpl)
            src = getattr(args, "_loaded_optimized_source_path", None)
            print(
                "\nReplay: using exact loaded quadrant geometry "
                f"(ring_offset={_ring}, z_plane={_zpl}"
                + (f", from {src}" if src else "")
                + ")."
            )
        else:
            _ring = getattr(args, "fixed_quadrant_ring_offset", None)
            if _ring is None:
                _ring = npml + 2
            _zpl = getattr(args, "fixed_quadrant_z_plane", None)
            if _zpl is None:
                _zpl = tumor_centroid_z_2d

            quadrant_air_margin_cells = int(
                getattr(args, "quadrant_air_margin_cells", 18)
            )
            if quadrant_air_margin_cells > 0:
                tissue_mask = labels_3d != 0
                air_mask = ~tissue_mask
                # Distance (in voxels) from each air voxel to the nearest tissue voxel.
                air_to_tissue_dist = ndimage.distance_transform_edt(air_mask)

                cx, cy = simulation_size_x // 2, simulation_size_y // 2
                z_plane_idx = int(_zpl)
                start_offset = int(_ring)
                offset_min = max(0, npml + 1)
                offset_max = min(
                    simulation_size_x - npml - 2, simulation_size_y - npml - 2
                )
                offset_max = max(offset_min, int(offset_max))

                best_offset = start_offset
                best_min_dist = -1.0
                best_abs_to_start = 10**9
                found_meeting = False

                for off in range(offset_min, offset_max + 1):
                    gaps = (
                        (cx, off, z_plane_idx),
                        (simulation_size_x - off - 1, cy, z_plane_idx),
                        (cx, simulation_size_y - off - 1, z_plane_idx),
                        (off, cy, z_plane_idx),
                    )
                    # Skip any configuration that would fall outside the grid.
                    valid = True
                    for gi, gj, gk in gaps:
                        if gi < 0 or gj < 0 or gk < 0:
                            valid = False
                            break
                        if (
                            gi >= simulation_size_x
                            or gj >= simulation_size_y
                            or gk >= simulation_size_z
                        ):
                            valid = False
                            break
                    if not valid:
                        continue

                    min_dist = min(
                        float(air_to_tissue_dist[gi, gj, gk]) for gi, gj, gk in gaps
                    )
                    meets = min_dist >= quadrant_air_margin_cells
                    if meets:
                        found_meeting = True
                        abs_to_start = abs(off - start_offset)
                        if abs_to_start < best_abs_to_start or (
                            abs_to_start == best_abs_to_start and min_dist > best_min_dist
                        ):
                            best_abs_to_start = abs_to_start
                            best_min_dist = min_dist
                            best_offset = off
                    elif min_dist > best_min_dist:
                        # Best-effort fallback if no offsets fully satisfy.
                        best_min_dist = min_dist
                        best_offset = off

                if not found_meeting:
                    print(
                        "\nWarning: could not satisfy quadrant air margin for all 4 gaps "
                        f"(requested={quadrant_air_margin_cells} cells). "
                        f"Using best offset={best_offset} with min_air_to_tissue_dist={best_min_dist:.2f}."
                    )
                _ring = int(best_offset)

        opt_quad_sources = build_quadrant_sources(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            npml,
            dipole_half_len=int(getattr(args, "fixed_quadrant_dipole_half_len", 9)),
            ring_offset=_ring,
            z_plane=int(_zpl),
        )
        _fa = getattr(args, "fixed_quadrant_alphas", None)
        if _fa is None:
            opt_alphas = np.ones(4, dtype=np.float64)
        else:
            opt_alphas = np.array(_fa, dtype=np.float64)
        _fp = getattr(args, "fixed_quadrant_phases_deg", None)
        if _fp is None:
            opt_thetas = np.zeros(4, dtype=np.float64)
        else:
            opt_thetas = np.deg2rad(np.array(_fp, dtype=np.float64))
        print(
            "\n4-quadrant APA (fixed, no optimization): "
            f"f0={opt_f0 / 1e6:.1f} MHz, ring_offset={_ring}, z_plane={_zpl}, "
            f"alphas={opt_alphas.tolist()}, phases_deg={np.rad2deg(opt_thetas).tolist()}"
        )

    _fig_c = plt.figure()
    _ax_c = _fig_c.add_subplot(111)
    _cs = _ax_c.contour(tumor_footprint_2d, levels=[0.5], origin="lower")
    tumor_contour_segments = _cs.allsegs[0] if len(_cs.allsegs) > 0 else []
    plt.close(_fig_c)

    number_of_frequencies = 3
    freq = np.array((50e6, 200e6, 500e6))
    arg = 2 * np.pi * freq * dt
    real_in = np.zeros(number_of_frequencies)
    imag_in = np.zeros(number_of_frequencies)
    real_pt = np.zeros(
        (number_of_frequencies, simulation_size_x, simulation_size_y, simulation_size_z)
    )
    imag_pt = np.zeros(
        (number_of_frequencies, simulation_size_x, simulation_size_y, simulation_size_z)
    )
    amp = np.zeros((number_of_frequencies, simulation_size_y))

    pulse_width = 8
    pulse_delay = 20
    pulse_amplitude = args.pulse_amplitude
    pulse_type = getattr(args, "pulse_type", "gaussian")
    prop_direction = getattr(args, "prop_direction", "+y")
    pulse_freq = getattr(args, "pulse_freq", 100e6)
    pulse_ramp_width = getattr(args, "pulse_ramp_width", 30.0)

    (
        gi1,
        gi2,
        gi3,
        fi1,
        fi2,
        fi3,
        gj1,
        gj2,
        gj3,
        fj1,
        fj2,
        fj3,
        gk1,
        gk2,
        gk3,
        fk1,
        fk2,
        fk3,
    ) = calculate_pml_parameters(
        npml, simulation_size_x, simulation_size_y, simulation_size_z
    )
    boundary_low = [0, 0]
    boundary_high = [0, 0]
    time_steps = args.time_steps
    sar_start_step = int(time_steps * 0.7)

    if opt_quad_sources is not None and (
        args.optimize_antenna or getattr(args, "quadrant_fixed", False)
    ):
        steps_per_period = max(1, round(1.0 / (opt_f0 * dt)))
        time_steps = 15 * steps_per_period
        sar_start_step = 10 * steps_per_period
        print(
            f"  CW run: {time_steps} steps (~{time_steps // steps_per_period} periods), "
            f"SAR accumulation from step {sar_start_step} ({time_steps - sar_start_step} samples)"
        )
    else:
        if getattr(args, "source_x", None) is not None:
            source_x = max(npml, min(simulation_size_x - npml - 1, args.source_x))
        if getattr(args, "source_y", None) is not None:
            source_y = max(npml, min(simulation_size_y - npml - 1, args.source_y))
        if getattr(args, "source_z", None) is not None:
            source_z = max(npml, min(simulation_size_z - npml - 1, args.source_z))
        if pulse_type == "cw" and getattr(args, "cw_periods", None) is not None:
            steps_per_period = max(1, round(1.0 / (pulse_freq * dt)))
            time_steps = args.cw_periods * steps_per_period
            sar_start_step = 10 * steps_per_period
            print(
                f"  Standard CW run: {time_steps} steps (~{args.cw_periods} periods), "
                f"SAR accumulation from step {sar_start_step}"
            )
        elif (
            pulse_type == "sinusoid_no_ramp"
            and getattr(args, "cw_periods", None) is not None
        ):
            steps_per_period = max(1, round(1.0 / (pulse_freq * dt)))
            time_steps = args.cw_periods * steps_per_period
            sar_start_step = 10 * steps_per_period
            print(
                f"  sinusoid_no_ramp (--cw-periods): {time_steps} steps (~{args.cw_periods} periods), "
                f"SAR accumulation from step {sar_start_step}"
            )
        elif pulse_type == "sinusoid_no_ramp":
            sar_start_step = 0
        elif pulse_type in ("cw", "sinusoid"):
            steps_per_period = max(1, round(1.0 / (pulse_freq * dt)))
            min_periods_total = 15
            min_steps = min_periods_total * steps_per_period
            if time_steps < min_steps:
                time_steps = min_steps
                print(
                    f"  {pulse_type}: time_steps increased to {time_steps} (~{min_periods_total} periods) "
                    "for valid SAR (full-period average after steady state)"
                )
            sar_start_step = 10 * steps_per_period
            n_sar_steps = time_steps - sar_start_step
            print(
                f"  SAR accumulation from step {sar_start_step} ({n_sar_steps} samples, "
                f"~{n_sar_steps // steps_per_period} full periods)"
            )
        if prop_direction in ("+x", "-x"):
            ez_inc_x = np.zeros(simulation_size_x)
            hy_inc_x = np.zeros(simulation_size_x)
            boundary_low_x = [0.0, 0.0]
            boundary_high_x = [0.0, 0.0]
        else:
            ez_inc_x = hy_inc_x = None
            boundary_low_x = boundary_high_x = None
        if prop_direction in ("+z", "-z"):
            ez_inc_z = np.zeros(simulation_size_z)
            hx_inc_z = np.zeros(simulation_size_z)
            boundary_low_z = [0.0, 0.0]
            boundary_high_z = [0.0, 0.0]
        else:
            ez_inc_z = hx_inc_z = None
            boundary_low_z = boundary_high_z = None

    Ex_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ey_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    Ez_sq_sum = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))
    n_sar_samples = 0

    E_frames = []
    SAR_frames = []
    Temperature_frames = []
    frame_interval = 1
    streamed_n_frames = 0
    STREAM_CHUNK_SIZE = 20

    t_start_total = time.perf_counter()
    _phases_before_fdtd = ["setup", "segmentation"]
    if args.optimize_antenna and opt_quad_sources is not None:
        _phases_before_fdtd = ["setup", "segmentation", "antenna_optimization"]
    _write_progress(
        "fdtd_simulation",
        f"FDTD running (0 / {time_steps} steps)",
        25,
        _phases_before_fdtd,
        {
            "time_steps": time_steps,
            "grid_shape": [simulation_size_x, simulation_size_y, simulation_size_z],
        },
    )

    point_source_positions = [(source_x, source_y, source_z)]

    if opt_quad_sources is not None and (
        args.optimize_antenna or getattr(args, "quadrant_fixed", False)
    ):
        _quad_mode = "optimized" if args.optimize_antenna else "fixed"
        print(
            "\nFDTD loop ({} 4-quadrant source): f0={:.1f} MHz, D→E,I→H per step.".format(
                _quad_mode,
                opt_f0 / 1e6,
            )
        )
        frame_interval = (
            args.stream_frame_interval
            if args.stream_frames
            else max(1, time_steps // 350)
        )
        fdtd_state = SimpleNamespace(
            simulation_size_x=simulation_size_x,
            simulation_size_y=simulation_size_y,
            simulation_size_z=simulation_size_z,
            time_steps=time_steps,
            frame_interval=frame_interval,
            sar_start_step=sar_start_step,
            dt=dt,
            opt_quad_sources=opt_quad_sources,
            opt_alphas=opt_alphas,
            opt_thetas=opt_thetas,
            opt_f0=opt_f0,
            opt_source_scale=args.opt_source_scale,
            stream_frames=args.stream_frames,
            STREAM_CHUNK_SIZE=STREAM_CHUNK_SIZE,
            E_frames=E_frames,
            SAR_frames=SAR_frames,
            write_progress_cb=_write_progress,
            _phases_before_fdtd=_phases_before_fdtd,
            args=args,
            Dx=Dx,
            Dy=Dy,
            Dz=Dz,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            Ix=Ix,
            Iy=Iy,
            Iz=Iz,
            Hx=Hx,
            Hy=Hy,
            Hz=Hz,
            iDx=iDx,
            iDy=iDy,
            iDz=iDz,
            iHx=iHx,
            iHy=iHy,
            iHz=iHz,
            gi1=gi1,
            gj1=gj1,
            gk1=gk1,
            gi2=gi2,
            gj2=gj2,
            gk2=gk2,
            gi3=gi3,
            gj3=gj3,
            gk3=gk3,
            fi1=fi1,
            fj1=fj1,
            fk1=fk1,
            fi2=fi2,
            fj2=fj2,
            fk2=fk2,
            fi3=fi3,
            fj3=fj3,
            fk3=fk3,
            eps_x=eps_x,
            eps_y=eps_y,
            eps_z=eps_z,
            conductivity_x=conductivity_x,
            conductivity_y=conductivity_y,
            conductivity_z=conductivity_z,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            rho=rho,
            Ex_sq_sum=Ex_sq_sum,
            Ey_sq_sum=Ey_sq_sum,
            Ez_sq_sum=Ez_sq_sum,
            E_FRAMES_DIR=E_FRAMES_DIR,
            SAR_FRAMES_DIR=SAR_FRAMES_DIR,
            OUTPUT_BASE=OUTPUT_BASE,
        )
        n_sar_samples, E_frames, SAR_frames, streamed_n_frames, time_fdtd_s = (
            run_fdtd_optimized_loop(fdtd_state)
        )
        t_end_fdtd = t_start_total + time_fdtd_s
        print(
            "  {} 4-quadrant FDTD complete.".format(
                "Optimized" if args.optimize_antenna else "Fixed"
            )
        )

    else:
        if args.stream_frames:
            print(
                f"  Streaming E/SAR frames to disk (interval={args.stream_frame_interval}, chunk={STREAM_CHUNK_SIZE} frames)"
            )
        inj_y = 3 if prop_direction == "+y" else simulation_size_y - 4
        inj_x = 3 if prop_direction == "+x" else simulation_size_x - 4
        inj_z = 3 if prop_direction == "+z" else simulation_size_z - 4

        point_source_positions = [(source_x, source_y, source_z)]
        if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
            ring_off = max(npml, getattr(args, "source_ring_offset", 10))
            cx, cy, cz = (
                simulation_size_x // 2,
                simulation_size_y // 2,
                simulation_size_z // 2,
            )
            if getattr(args, "use_source_2", False):
                sx2 = cx if args.source_x_2 is None else args.source_x_2
                sy2 = ring_off if args.source_y_2 is None else args.source_y_2
                sz2 = cz if args.source_z_2 is None else args.source_z_2
                sx2 = max(npml, min(simulation_size_x - npml - 1, sx2))
                sy2 = max(npml, min(simulation_size_y - npml - 1, sy2))
                sz2 = max(npml, min(simulation_size_z - npml - 1, sz2))
                point_source_positions.append((sx2, sy2, sz2))
            if getattr(args, "use_source_3", False):
                sx3 = (
                    (simulation_size_x - ring_off - 1)
                    if args.source_x_3 is None
                    else args.source_x_3
                )
                sy3 = cy if args.source_y_3 is None else args.source_y_3
                sz3 = cz if args.source_z_3 is None else args.source_z_3
                sx3 = max(npml, min(simulation_size_x - npml - 1, sx3))
                sy3 = max(npml, min(simulation_size_y - npml - 1, sy3))
                sz3 = max(npml, min(simulation_size_z - npml - 1, sz3))
                point_source_positions.append((sx3, sy3, sz3))
            if len(point_source_positions) > 1:
                print(
                    f"  Point sources: {len(point_source_positions)} positions {point_source_positions}"
                )

        fdtd_standard_state = SimpleNamespace(
            simulation_size_x=simulation_size_x,
            simulation_size_y=simulation_size_y,
            simulation_size_z=simulation_size_z,
            time_steps=time_steps,
            sar_start_step=sar_start_step,
            dt=dt,
            pulse_type=pulse_type,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_delay=pulse_delay,
            pulse_freq=pulse_freq,
            pulse_ramp_width=pulse_ramp_width,
            number_of_frequencies=number_of_frequencies,
            arg=arg,
            real_in=real_in,
            imag_in=imag_in,
            real_pt=real_pt,
            imag_pt=imag_pt,
            source_z=source_z,
            source_x=source_x,
            prop_direction=prop_direction,
            point_source_positions=point_source_positions,
            ia=ia,
            ja=ja,
            ka=ka,
            ib=ib,
            jb=jb,
            kb=kb,
            boundary_low=boundary_low,
            boundary_high=boundary_high,
            ez_inc=ez_inc,
            hx_inc=hx_inc,
            inj_y=inj_y,
            Dx=Dx,
            Dy=Dy,
            Dz=Dz,
            iDx=iDx,
            iDy=iDy,
            iDz=iDz,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            Ix=Ix,
            Iy=Iy,
            Iz=Iz,
            Hx=Hx,
            Hy=Hy,
            Hz=Hz,
            iHx=iHx,
            iHy=iHy,
            iHz=iHz,
            gi1=gi1,
            gj1=gj1,
            gk1=gk1,
            gi2=gi2,
            gj2=gj2,
            gk2=gk2,
            gi3=gi3,
            gj3=gj3,
            gk3=gk3,
            fi1=fi1,
            fj1=fj1,
            fk1=fk1,
            fi2=fi2,
            fj2=fj2,
            fk2=fk2,
            fi3=fi3,
            fj3=fj3,
            fk3=fk3,
            eps_x=eps_x,
            eps_y=eps_y,
            eps_z=eps_z,
            conductivity_x=conductivity_x,
            conductivity_y=conductivity_y,
            conductivity_z=conductivity_z,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            rho=rho,
            Ex_sq_sum=Ex_sq_sum,
            Ey_sq_sum=Ey_sq_sum,
            Ez_sq_sum=Ez_sq_sum,
            stream_frames=args.stream_frames,
            stream_frame_interval=args.stream_frame_interval,
            STREAM_CHUNK_SIZE=STREAM_CHUNK_SIZE,
            E_FRAMES_DIR=E_FRAMES_DIR,
            SAR_FRAMES_DIR=SAR_FRAMES_DIR,
            OUTPUT_BASE=OUTPUT_BASE,
            E_frames=E_frames,
            SAR_frames=SAR_frames,
            write_progress_cb=_write_progress,
            _phases_before_fdtd=_phases_before_fdtd,
        )
        if prop_direction in ("+x", "-x"):
            fdtd_standard_state.boundary_low_x = boundary_low_x
            fdtd_standard_state.boundary_high_x = boundary_high_x
            fdtd_standard_state.ez_inc_x = ez_inc_x
            fdtd_standard_state.hy_inc_x = hy_inc_x
            fdtd_standard_state.inj_x = inj_x
        if prop_direction in ("+z", "-z"):
            fdtd_standard_state.boundary_low_z = boundary_low_z
            fdtd_standard_state.boundary_high_z = boundary_high_z
            fdtd_standard_state.ez_inc_z = ez_inc_z
            fdtd_standard_state.hx_inc_z = hx_inc_z
            fdtd_standard_state.inj_z = inj_z

        n_sar_samples, E_frames, SAR_frames, streamed_n_frames, time_fdtd_s = (
            run_fdtd_standard_loop(fdtd_standard_state)
        )
        t_end_fdtd = t_start_total + time_fdtd_s
        real_in = fdtd_standard_state.real_in
        imag_in = fdtd_standard_state.imag_in
        real_pt = fdtd_standard_state.real_pt
        imag_pt = fdtd_standard_state.imag_pt

    if not args.optimize_antenna and not getattr(args, "quadrant_fixed", False):
        amp_in = np.sqrt(real_in**2 + imag_in**2)
        for m in range(number_of_frequencies):
            for j in range(ja, jb + 1):
                if eps_z[source_x, j, source_z] < 1:
                    amp[m, j] = (
                        1
                        / (amp_in[m])
                        * sqrt(
                            real_pt[m, source_x, j, source_z] ** 2
                            + imag_pt[m, source_x, j, source_z] ** 2
                        )
                    )

    _write_progress(
        "sar_computation",
        "Computing SAR...",
        72,
        _phases_before_fdtd + ["fdtd_simulation"],
    )
    print("\nSAR computation (σ|E_rms|²/(2ρ) voxel-wise):")
    print(f"  Samples used for RMS: {n_sar_samples}")
    t_start_sar = time.perf_counter()
    if n_sar_samples > 0:
        SAR = compute_sar(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            Ex_sq_sum,
            Ey_sq_sum,
            Ez_sq_sum,
            sigma_x,
            sigma_y,
            sigma_z,
            rho,
            n_sar_samples,
        )
        sar_max = np.max(SAR)
        print(f"  SAR statistics: max={sar_max:.6g} W/kg")
        if np.any(rho > 0):
            print(f"  Mean SAR (tissue only): {np.mean(SAR[rho > 0]):.6g} W/kg")
        if np.any(tumor_region):
            sar_t = SAR[tumor_region]
            print(
                f"  SAR in tumor: min={np.min(sar_t):.6g}, max={np.max(sar_t):.6g}, mean={np.mean(sar_t):.6g} W/kg"
            )
        if np.any(non_tumor_region):
            sar_nt = SAR[non_tumor_region]
            print(
                f"  SAR in non-tumor tissue: min={np.min(sar_nt):.6g}, max={np.max(sar_nt):.6g}, mean={np.mean(sar_nt):.6g} W/kg"
            )
    else:
        print("  Warning: no E² samples collected; SAR set to zero.")
        SAR = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    # Objective J = mean(SAR|tumor) / mean(SAR|healthy) — every run (same label masks as optimization)
    penalty_w = float(getattr(args, "opt_penalty_weight", 0.0))
    objective_record = {
        "penalty_weight": penalty_w,
        "target_labels": list(anatomy_config.target_labels),
        "healthy_labels": list(anatomy_config.healthy_labels),
        "definition": (
            "J = mean(SAR in tumor voxels) / mean(SAR in healthy-label voxels); "
            "J_eff = J - penalty_weight * (P95 healthy SAR) / mean(tumor SAR) when penalty_weight > 0"
        ),
    }
    if np.any(tumor_region) and np.any(non_tumor_region):
        J_eff, J_plain, m_tumor, m_healthy, p95_h = compute_robust_objective(
            SAR, tumor_region, non_tumor_region, penalty_weight=penalty_w
        )
        objective_record.update(
            {
                "valid": True,
                "J": float(J_plain),
                "J_eff": float(J_eff),
                "mean_sar_tumor_W_per_kg": float(m_tumor),
                "mean_sar_healthy_W_per_kg": float(m_healthy),
                "p95_sar_healthy_W_per_kg": float(p95_h),
            }
        )
        print(
            f"  Objective (final SAR field): J={J_plain:.6f}"
            + (f"  J_eff={J_eff:.6f}" if penalty_w > 0 else "")
            + f"  (mean tumor {m_tumor:.6g} W/kg, mean healthy {m_healthy:.6g} W/kg)"
        )
    else:
        reason = (
            "no_tumor_voxels" if not np.any(tumor_region) else "no_healthy_label_voxels"
        )
        objective_record.update(
            {
                "valid": False,
                "reason": reason,
                "J": None,
                "J_eff": None,
                "mean_sar_tumor_W_per_kg": None,
                "mean_sar_healthy_W_per_kg": None,
                "p95_sar_healthy_W_per_kg": None,
            }
        )
        print(f"  Objective J: not defined ({reason}).")
    objective_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_objective.json")
    with open(objective_path, "w") as f:
        json.dump(objective_record, f, indent=2)
    print(f"  Objective record: {objective_path}")

    t_end_sar = time.perf_counter()
    _write_progress(
        "sar_computation",
        "SAR complete",
        75,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation"],
    )

    _write_progress(
        "thermal_solver",
        "Thermal modeling...",
        77,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation"],
    )
    print(
        "\nThermal solver (steady-state Pennes, no perfusion): Q = SAR·ρ, Gauss–Seidel."
    )
    t_start_thermal = time.perf_counter()
    Q_heat = SAR * rho
    Q_heat[rho <= 0] = 0.0
    T_temp = solve_steady_bioheat_3d(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        k_3d,
        Q_heat,
        dx,
        T_boundary=T_BOUNDARY_CELSIUS,
        max_iter=50000,
        tol=1e-6,
    )
    if np.any(k_3d > 0):
        T_tissue = T_temp[k_3d > 0]
        print(
            f"  Temperature in tissue: min={np.min(T_tissue):.4f} °C, max={np.max(T_tissue):.4f} °C"
        )
    if np.any(tumor_region):
        T_t = T_temp[tumor_region]
        print(
            f"  Temperature in tumor: min={np.min(T_t):.4f} °C, max={np.max(T_t):.4f} °C"
        )
    if np.any(non_tumor_region):
        T_nt = T_temp[non_tumor_region]
        print(
            f"  Temperature in non-tumor tissue: min={np.min(T_nt):.4f} °C, max={np.max(T_nt):.4f} °C"
        )
    region_stats = {}
    if np.any(tumor_region):
        region_stats["sar_tumor_W_per_kg"] = {
            "min": float(np.min(SAR[tumor_region])),
            "max": float(np.max(SAR[tumor_region])),
            "mean": float(np.mean(SAR[tumor_region])),
        }
        region_stats["temperature_tumor_C"] = {
            "min": float(np.min(T_temp[tumor_region])),
            "max": float(np.max(T_temp[tumor_region])),
        }
    if np.any(non_tumor_region):
        region_stats["sar_non_tumor_tissue_W_per_kg"] = {
            "min": float(np.min(SAR[non_tumor_region])),
            "max": float(np.max(SAR[non_tumor_region])),
            "mean": float(np.mean(SAR[non_tumor_region])),
        }
        region_stats["temperature_non_tumor_tissue_C"] = {
            "min": float(np.min(T_temp[non_tumor_region])),
            "max": float(np.max(T_temp[non_tumor_region])),
        }
    t_end_thermal = time.perf_counter()
    _write_progress(
        "thermal_solver",
        "Thermal complete",
        80,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation", "thermal_solver"],
    )

    print("\nTemperature frames for animation (proportional to SAR frames)...")
    T_max = np.max(T_temp) if np.any(k_3d > 0) else T_BOUNDARY_CELSIUS
    SAR_max_final = np.max(SAR) if np.max(SAR) > 0 else 1.0
    tissue_mask = rho > 0

    if streamed_n_frames > 0:
        n_stream_parts = (
            streamed_n_frames + STREAM_CHUNK_SIZE - 1
        ) // STREAM_CHUNK_SIZE
        for part in range(n_stream_parts):
            start = part * STREAM_CHUNK_SIZE
            end = min(start + STREAM_CHUNK_SIZE, streamed_n_frames)
            part_path_sar = os.path.join(
                SAR_FRAMES_DIR, f"{OUTPUT_BASE}_SAR_frames_part{part}.npz"
            )
            with np.load(part_path_sar) as z:
                sar_chunk = z["SAR_frames"]
            temp_chunk = np.full_like(sar_chunk, T_BOUNDARY_CELSIUS, dtype=np.float32)
            for i in range(sar_chunk.shape[0]):
                if np.any(tissue_mask) and SAR_max_final > 0:
                    sar_normalized = np.clip(sar_chunk[i] / SAR_max_final, 0, 1).astype(
                        np.float32
                    )
                    temp_chunk[i][tissue_mask] = T_BOUNDARY_CELSIUS + sar_normalized[
                        tissue_mask
                    ] * (T_max - T_BOUNDARY_CELSIUS)
            part_path_t = os.path.join(
                TEMPERATURE_FRAMES_DIR,
                f"{OUTPUT_BASE}_Temperature_frames_part{part}.npz",
            )
            np.savez_compressed(part_path_t, Temperature_frames=temp_chunk)
        print(
            f"  Computed and saved {streamed_n_frames} temperature frames (streamed, {n_stream_parts} parts)"
        )
    elif len(SAR_frames) > 0:
        for sar_frame in SAR_frames:
            temp_frame = np.full_like(sar_frame, T_BOUNDARY_CELSIUS, dtype=np.float32)
            if np.any(tissue_mask) and SAR_max_final > 0:
                sar_normalized = np.clip(sar_frame / SAR_max_final, 0, 1)
                temp_frame[tissue_mask] = T_BOUNDARY_CELSIUS + sar_normalized[
                    tissue_mask
                ] * (T_max - T_BOUNDARY_CELSIUS)
            Temperature_frames.append(temp_frame)
        print(f"  Computed {len(Temperature_frames)} temperature frames")
    else:
        print(
            "  Warning: No SAR frames available, skipping temperature frame computation"
        )

    n_frames_effective = (
        streamed_n_frames
        if streamed_n_frames > 0
        else (len(SAR_frames) if SAR_frames else 0)
    )
    use_quad_block = opt_quad_sources is not None and (
        args.optimize_antenna or getattr(args, "quadrant_fixed", False)
    )
    saved_frame_timesteps_list = saved_timesteps_for_run(
        time_steps,
        n_frames_effective,
        args.stream_frames,
        int(getattr(args, "stream_frame_interval", 1)),
        int(frame_interval),
        use_quad_block,
    )

    time_series_data = None
    if len(SAR_frames) > 0 and len(Temperature_frames) == len(SAR_frames):
        n_frames_ts = len(SAR_frames)
        if len(saved_frame_timesteps_list) == n_frames_ts:
            time_step_indices = list(saved_frame_timesteps_list)
        else:
            time_step_indices = [
                saved_frame_timesteps_list[min(i, len(saved_frame_timesteps_list) - 1)]
                for i in range(n_frames_ts)
            ]
            if len(saved_frame_timesteps_list) != n_frames_ts:
                print(
                    f"  Note: saved_frame_timesteps ({len(saved_frame_timesteps_list)}) "
                    f"!= SAR frames ({n_frames_ts}); time_series uses nearest mapping."
                )
        tissue_mask = rho > 0
        max_sar_list = []
        mean_sar_list = []
        max_temp_list = []
        mean_temp_list = []
        for i in range(n_frames_ts):
            sf = SAR_frames[i]
            tf = Temperature_frames[i]
            max_sar_list.append(float(np.max(sf)))
            mean_sar_list.append(
                float(np.mean(sf[tissue_mask])) if np.any(tissue_mask) else 0.0
            )
            max_temp_list.append(float(np.max(tf)))
            mean_temp_list.append(
                float(np.mean(tf[tissue_mask])) if np.any(tissue_mask) else 0.0
            )
        time_series_data = {
            "time_step": time_step_indices,
            "max_sar_W_per_kg": max_sar_list,
            "mean_sar_W_per_kg": mean_sar_list,
            "max_temperature_C": max_temp_list,
            "mean_temperature_C": mean_temp_list,
        }
        print(
            f"  Scalar time series: {n_frames_ts} frames (timesteps from saved_frame_timesteps)"
        )

    total_wall_time_s = time.perf_counter() - t_start_total
    time_fdtd_s = t_end_fdtd - t_start_total
    time_sar_s = t_end_sar - t_start_sar
    time_thermal_s = t_end_thermal - t_start_thermal
    setup_s = t_start_total - t_end_segmentation
    segmentation_s = t_end_segmentation - t_start_pipeline
    number_of_voxels = simulation_size_x * simulation_size_y * simulation_size_z
    peak_memory_MB = get_peak_memory_mb()

    performance_metrics = {
        "total_simulation_time_s": None,
        "phases_s": {
            "segmentation": round(segmentation_s, 4),
            "setup": round(setup_s, 4),
            "fdtd_simulation": round(time_fdtd_s, 4),
            "sar_computation": round(time_sar_s, 4),
            "thermal_solver": round(time_thermal_s, 4),
            "saving_and_animations": None,
        },
        "time_steps": time_steps,
        "grid_shape": [simulation_size_x, simulation_size_y, simulation_size_z],
        "number_of_voxels": number_of_voxels,
        "time_per_step_ms": (
            round(1000.0 * time_fdtd_s / time_steps, 4) if time_steps else None
        ),
        "peak_memory_MB": (
            round(peak_memory_MB, 2) if peak_memory_MB is not None else None
        ),
        "backend": "numpy_numba",
        "dt_s": round(float(dt), 12),
        "dt_courant_s": round(float(dt_courant), 12),
        "courant_factor": float(args.courant_factor),
    }
    if antenna_optimization_s is not None:
        performance_metrics["phases_s"]["antenna_optimization"] = antenna_optimization_s
    performance_metrics["total_wall_time_s"] = round(total_wall_time_s, 4)
    performance_metrics["time_fdtd_s"] = round(time_fdtd_s, 4)
    performance_metrics["time_sar_s"] = round(time_sar_s, 4)
    performance_metrics["time_thermal_s"] = round(time_thermal_s, 4)

    print("\nPerformance metrics (paper Sec. 3.2.6):")
    print(f"  Segmentation: {segmentation_s:.2f} s  |  Setup: {setup_s:.2f} s")
    if antenna_optimization_s is not None:
        print(f"  Antenna optimization: {antenna_optimization_s:.2f} s")
    print(
        f"  FDTD: {time_fdtd_s:.2f} s  |  SAR: {time_sar_s:.2f} s  |  Thermal: {time_thermal_s:.2f} s"
    )
    print(f"  Total (pre-save): {total_wall_time_s:.2f} s")
    if peak_memory_MB is not None:
        print(f"  Peak memory: {peak_memory_MB:.1f} MB")
    print(f"  Time per FDTD step: {performance_metrics['time_per_step_ms']} ms")

    _write_progress(
        "saving_and_animations",
        "Saving NIfTI, metadata, frames...",
        85,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation", "thermal_solver"],
    )
    t_start_save = time.perf_counter()
    print("\nSaving simulation data (NIfTI, NumPy, JSON)...")
    affine = np.diag([float(dx), float(dx), float(dx), 1.0])
    E_FRAMES_CHUNK_SIZE = 20
    n_frames = streamed_n_frames if streamed_n_frames > 0 else len(E_frames)
    n_sar_frames = streamed_n_frames if streamed_n_frames > 0 else len(SAR_frames)
    n_temp_frames = (
        streamed_n_frames if streamed_n_frames > 0 else len(Temperature_frames)
    )
    n_parts = (
        (n_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        if n_frames > 0
        else 0
    )
    LABEL_NAMES = {
        0: "background (air)",
        1: "necrotic tumor",
        2: "edema",
        3: "enhancing tumor",
        4: "normal brain",
    }
    tissue_properties = []
    for lab in sorted(TISSUE_TABLE.keys()):
        eps_r, sigma_val, rho_val = TISSUE_TABLE[lab]
        k_val = K_TISSUE.get(lab, 0.0)
        tissue_properties.append(
            {
                "label": lab,
                "name": LABEL_NAMES.get(lab, f"label_{lab}"),
                "eps_r": float(eps_r),
                "sigma_S_per_m": float(sigma_val),
                "rho_kg_per_m3": float(rho_val),
                "k_W_per_mK": float(k_val),
            }
        )
    SAR_frames_n_parts = (
        (n_sar_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        if n_sar_frames > 0
        else 0
    )
    Temperature_frames_n_parts = (
        (n_temp_frames + E_FRAMES_CHUNK_SIZE - 1) // E_FRAMES_CHUNK_SIZE
        if n_temp_frames > 0
        else 0
    )
    metadata = {
        "output_base": OUTPUT_BASE,
        "grid_shape": [simulation_size_x, simulation_size_y, simulation_size_z],
        "air_padding_cells": int(getattr(args, "air_padding_cells", 0)),
        "voxel_size_m": float(dx),
        "time_step_s": float(dt),
        "dt_courant_s": float(dt_courant),
        "courant_factor": float(args.courant_factor),
        "time_steps": time_steps,
        "frame_interval": frame_interval,
        "stream_frames": args.stream_frames,
        "stream_frame_interval": getattr(args, "stream_frame_interval", 1),
        "pulse_amplitude": float(pulse_amplitude),
        "n_frames": n_frames,
        "E_frames_chunk_size": E_FRAMES_CHUNK_SIZE,
        "E_frames_n_parts": n_parts,
        "SAR_frames_n_parts": SAR_frames_n_parts,
        "Temperature_frames_n_parts": Temperature_frames_n_parts,
        "saved_frame_timesteps": saved_frame_timesteps_list,
        "time_series_file": (
            f"{OUTPUT_BASE}_time_series.json" if time_series_data else None
        ),
        "frequencies_Hz": freq.tolist(),
        "top_10_slice_indices": top_10_slice_indices,
        "T_boundary_C": T_BOUNDARY_CELSIUS,
        "tissue_properties": tissue_properties,
        "performance": performance_metrics,
        "region_stats": region_stats,
        "objective": objective_record,
    }
    if args.optimize_antenna and opt_f0 is not None:
        metadata["antenna_optimized"] = True
        metadata["optimized_f0_Hz"] = float(opt_f0)
        metadata["optimized_alphas"] = opt_alphas.tolist()
        metadata["optimized_thetas_rad"] = opt_thetas.tolist()
        metadata["optimized_quadrant_gaps"] = [qs["gap"] for qs in opt_quad_sources]
        metadata["opt_source_scale"] = args.opt_source_scale
    elif getattr(args, "quadrant_fixed", False) and opt_f0 is not None:
        metadata["quadrant_fixed"] = True
        metadata["fixed_f0_Hz"] = float(opt_f0)
        metadata["fixed_alphas"] = opt_alphas.tolist()
        metadata["fixed_thetas_rad"] = opt_thetas.tolist()
        metadata["fixed_quadrant_gaps"] = [qs["gap"] for qs in opt_quad_sources]
        metadata["opt_source_scale"] = args.opt_source_scale
    else:
        metadata["pulse_type"] = pulse_type
        metadata["prop_direction"] = prop_direction
        metadata["source_x"] = int(source_x)
        metadata["source_y"] = int(source_y)
        metadata["source_z"] = int(source_z)
        if len(point_source_positions) > 1:
            metadata["point_source_positions"] = [
                [int(sx), int(sy), int(sz)] for (sx, sy, sz) in point_source_positions
            ]
        metadata["pulse_freq_Hz"] = float(pulse_freq)
        if getattr(args, "cw_periods", None) is not None:
            metadata["cw_periods"] = args.cw_periods
    metadata["antenna_parameters"] = build_antenna_parameters_metadata(
        args,
        pulse_amplitude,
        pulse_type,
        prop_direction,
        pulse_freq,
        pulse_ramp_width,
        source_x,
        source_y,
        source_z,
        point_source_positions,
        opt_f0,
        opt_alphas,
        opt_thetas,
        opt_quad_sources,
    )
    save_simulation_data(
        OUTPUT_BASE,
        DATA_DIR,
        SAR_FRAMES_DIR,
        TEMPERATURE_FRAMES_DIR,
        E_FRAMES_DIR,
        SAR,
        T_temp,
        labels_3d,
        affine,
        E_frames,
        SAR_frames,
        Temperature_frames,
        streamed_n_frames,
        args.stream_frames,
        metadata,
        performance_metrics,
        time_series_data=time_series_data,
        e_frames_chunk_size=E_FRAMES_CHUNK_SIZE,
    )
    metadata_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_metadata.json")
    performance_path = os.path.join(DATA_DIR, f"{OUTPUT_BASE}_performance.json")
    t_end_save_data = time.perf_counter()
    saving_data_s = t_end_save_data - t_start_save

    print("\nSaving SAR and temperature distribution figures...")
    cx, cy, cz = (
        simulation_size_x // 2,
        simulation_size_y // 2,
        simulation_size_z // 2,
    )
    save_sar_distribution(
        SAR,
        OUTPUT_BASE,
        IMAGES_DIR,
        tumor_footprint_2d,
        cx,
        cy,
        cz,
        tumor_mask_3d=tumor_region,
    )
    print(f"  SAR distribution figure: {OUTPUT_BASE}_SAR_distribution.png")
    save_temperature_distribution(
        T_temp,
        OUTPUT_BASE,
        IMAGES_DIR,
        tumor_footprint_2d,
        cx,
        cy,
        cz,
        tumor_mask_3d=tumor_region,
    )
    print(
        f"  Temperature distribution figure: {OUTPUT_BASE}_temperature_distribution.png"
    )

    print("\nSaving per-slice validation (Anatomy | SAR | Temperature)...")
    for k in top_10_slice_indices:
        if not (0 <= k < simulation_size_z):
            continue
        save_slice_anatomy_sar_temperature(
            k,
            labels_3d,
            SAR,
            T_temp,
            tumor_footprint_2d,
            OUTPUT_BASE,
            IMAGES_DIR,
            volume_4d_ds=volume_4d_ds,
            viz_config=viz_config,
        )
    print(f"  Per-slice figures saved: {len(top_10_slice_indices)} slices.")
    _write_progress(
        "saving_and_animations",
        "Building animations...",
        90,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation", "thermal_solver"],
    )

    _script_dir = getattr(
        paths, "script_dir", os.path.dirname(os.path.abspath(__file__))
    )
    animations_s = build_and_save_animations(
        OUTPUT_BASE,
        ANIMATIONS_DIR,
        E_frames,
        SAR_frames,
        Temperature_frames,
        tumor_footprint_2d,
        tumor_contour_segments,
        T_BOUNDARY_CELSIUS,
        args.skip_animations,
        args.stream_frames,
        streamed_n_frames,
        _script_dir,
        os.path.abspath(RESULTS_DIR),
        os.path.abspath(ANIMATIONS_DIR),
        subsample=max(1, int(getattr(args, "subsample", 1))),
        slice_timestep_images=getattr(args, "slice_timestep_images", False),
        volume_4d_ds=volume_4d_ds if use_modalities else None,
        labels_3d=labels_3d,
        sar_3d=SAR if use_modalities else None,
        temperature_3d=T_temp if use_modalities else None,
        images_dir=IMAGES_DIR if use_modalities else None,
        case_name=OUTPUT_BASE if use_modalities else None,
        saved_frame_timesteps=saved_frame_timesteps_list,
        time_steps=time_steps,
        data_dir_abs=DATA_DIR,
    )
    _write_progress(
        "saving_and_animations",
        "Animations complete",
        99,
        _phases_before_fdtd + ["fdtd_simulation", "sar_computation", "thermal_solver"],
    )

    performance_metrics["phases_s"]["saving_data"] = round(saving_data_s, 4)
    performance_metrics["phases_s"]["animations"] = round(animations_s, 4)
    performance_metrics["phases_s"].pop("saving_and_animations", None)
    t_end_pipeline = time.perf_counter()
    performance_metrics["total_simulation_time_s"] = round(
        t_end_pipeline - t_start_pipeline, 4
    )
    with open(metadata_path, "r") as f:
        _meta = json.load(f)
    _meta["performance"] = performance_metrics
    with open(metadata_path, "w") as f:
        json.dump(_meta, f, indent=2)
    with open(performance_path, "w") as f:
        json.dump(performance_metrics, f, indent=2)
    _complete_phases = [
        "setup",
        "segmentation",
        "fdtd_simulation",
        "sar_computation",
        "thermal_solver",
        "saving_and_animations",
        "complete",
    ]
    if args.optimize_antenna and opt_quad_sources is not None:
        _complete_phases.insert(2, "antenna_optimization")
    _write_progress("complete", "Simulation complete", 100, _complete_phases)
    print(
        f"\nPipeline complete (paper Sec. 3.2): segmentation → FDTD → SAR → thermal → validation."
    )
    print(
        f"  Total time: {performance_metrics['total_simulation_time_s']:.2f} s  |  Save: {saving_data_s:.2f} s  |  Animations: {animations_s:.2f} s"
    )


def run_brain_simulation(args, paths, write_progress_cb):
    """Run full brain simulation. Delegates to run_simulation(..., anatomy=\"brain\")."""
    return run_simulation(args, paths, write_progress_cb, anatomy="brain")
