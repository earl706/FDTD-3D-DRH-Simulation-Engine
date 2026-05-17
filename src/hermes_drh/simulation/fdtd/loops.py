"""
FDTD time-loop control (thesis: FDTD Solver Implementation).

Minimal benchmark runner, optimized 4-quadrant loop, and standard (multi-source/pulse) loop.
Single responsibility: solver control logic and time stepping.
"""

from math import exp, sqrt, cos, sin
import os
import time
import numpy as np

from hermes_drh.simulation.fdtd import kernels
from hermes_drh.simulation.fdtd import boundaries

try:
    from hermes_drh.compat.sar_computation import compute_instantaneous_sar
except ImportError:
    try:
        from hermes_drh.simulation.metrics.sar import compute_instantaneous_sar
    except ImportError:
        compute_instantaneous_sar = None

try:
    from hermes_drh.io.performance import get_peak_memory_mb as _get_peak_memory_mb
except ImportError:
    _get_peak_memory_mb = None

# Kernel and boundary helpers (local refs for readability)
calculate_pml_parameters = kernels.calculate_pml_parameters
calculate_dx_field = kernels.calculate_dx_field
calculate_dy_field = kernels.calculate_dy_field
calculate_dz_field = kernels.calculate_dz_field
calculate_e_fields = kernels.calculate_e_fields
calculate_fourier_transform_ex = kernels.calculate_fourier_transform_ex
calculate_hx_field = kernels.calculate_hx_field
calculate_hy_field = kernels.calculate_hy_field
calculate_hz_field = kernels.calculate_hz_field
accumulate_e_field_squared = kernels.accumulate_e_field_squared
calculate_inc_dy_field = boundaries.calculate_inc_dy_field
calculate_inc_dz_field = boundaries.calculate_inc_dz_field
calculate_hx_inc = boundaries.calculate_hx_inc
calculate_hx_with_incident_field = boundaries.calculate_hx_with_incident_field
calculate_hy_with_incident_field = boundaries.calculate_hy_with_incident_field
update_ez_inc_x = boundaries.update_ez_inc_x
calculate_hy_inc_x = boundaries.calculate_hy_inc_x
calculate_inc_dz_field_x = boundaries.calculate_inc_dz_field_x
calculate_hy_with_incident_field_x = boundaries.calculate_hy_with_incident_field_x
update_ez_inc_z = boundaries.update_ez_inc_z
calculate_hx_inc_z = boundaries.calculate_hx_inc_z
calculate_inc_dz_field_z = boundaries.calculate_inc_dz_field_z
calculate_hx_with_incident_field_z = boundaries.calculate_hx_with_incident_field_z


def _run_minimal_fdtd_benchmark(
    nx, ny, nz, time_steps, dx_mm=10.0, courant_factor=0.99
):
    """
    Run a minimal FDTD-only loop (air, no source) for scalability benchmarking.
    Returns dict with grid_shape, number_of_voxels, time_steps, total_wall_time_s,
    time_per_step_ms, peak_memory_MB for inclusion in scalability JSON.
    """
    dx = dx_mm * 1e-3
    c_light = 2.99792458e8
    dy = dz = dx
    dt_courant = 1.0 / (
        c_light * sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz))
    )
    dt = courant_factor * dt_courant
    npml = max(4, min(16, min(nx, ny, nz) // 10))
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
    ) = calculate_pml_parameters(npml, nx, ny, nz)
    eps_x = np.ones((nx, ny, nz))
    eps_y = np.ones((nx, ny, nz))
    eps_z = np.ones((nx, ny, nz))
    conductivity_x = np.zeros((nx, ny, nz))
    conductivity_y = np.zeros((nx, ny, nz))
    conductivity_z = np.zeros((nx, ny, nz))
    Dx = np.zeros((nx, ny, nz))
    Dy = np.zeros((nx, ny, nz))
    Dz = np.zeros((nx, ny, nz))
    iDx = np.zeros((nx, ny, nz))
    iDy = np.zeros((nx, ny, nz))
    iDz = np.zeros((nx, ny, nz))
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Ix = np.zeros((nx, ny, nz))
    Iy = np.zeros((nx, ny, nz))
    Iz = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    iHx = np.zeros((nx, ny, nz))
    iHy = np.zeros((nx, ny, nz))
    iHz = np.zeros((nx, ny, nz))
    t0 = time.perf_counter()
    for _ in range(1, time_steps + 1):
        Dx, iDx = calculate_dx_field(
            nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
        )
        Dy, iDy = calculate_dy_field(
            nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
        )
        Dz, iDz = calculate_dz_field(
            nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
        )
        Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
            nx,
            ny,
            nz,
            Dx,
            Dy,
            Dz,
            eps_x,
            eps_y,
            eps_z,
            conductivity_x,
            conductivity_y,
            conductivity_z,
            Ex,
            Ey,
            Ez,
            Ix,
            Iy,
            Iz,
        )
        Hx, iHx = calculate_hx_field(
            nx, ny, nz, Hx, iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
        )
        Hy, iHy = calculate_hy_field(
            nx, ny, nz, Hy, iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
        )
        Hz, iHz = calculate_hz_field(
            nx, ny, nz, Hz, iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3
        )
    t1 = time.perf_counter()
    total_wall_time_s = t1 - t0
    time_per_step_ms = 1000.0 * total_wall_time_s / time_steps if time_steps else None
    number_of_voxels = nx * ny * nz
    peak_memory_MB = _get_peak_memory_mb() if _get_peak_memory_mb is not None else None
    return {
        "grid_shape": [nx, ny, nz],
        "number_of_voxels": number_of_voxels,
        "time_steps": time_steps,
        "total_wall_time_s": round(total_wall_time_s, 6),
        "time_per_step_ms": (
            round(time_per_step_ms, 6) if time_per_step_ms is not None else None
        ),
        "peak_memory_MB": (
            round(peak_memory_MB, 2) if peak_memory_MB is not None else None
        ),
    }


def run_fdtd_optimized_loop(state):
    """
    Run the optimized 4-quadrant FDTD time loop (Option A).
    Mutates state in place. Returns (n_sar_samples, E_frames, SAR_frames, streamed_n_frames, time_fdtd_s).
    """
    if compute_instantaneous_sar is None:
        raise ImportError("sar_computation is required for run_fdtd_optimized_loop")
    from math import sin as _sin, exp as _exp

    nx = state.simulation_size_x
    ny = state.simulation_size_y
    nz = state.simulation_size_z
    time_steps = state.time_steps
    frame_interval = state.frame_interval
    sar_start_step = state.sar_start_step
    dt = state.dt
    opt_quad_sources = state.opt_quad_sources
    opt_alphas = state.opt_alphas
    opt_thetas = state.opt_thetas
    opt_f0 = state.opt_f0
    opt_source_scale = state.opt_source_scale
    stream_frames = state.stream_frames
    STREAM_CHUNK_SIZE = state.STREAM_CHUNK_SIZE
    ramp_width_opt = 30.0
    E_frames = state.E_frames
    SAR_frames = state.SAR_frames
    streamed_n_frames = 0
    n_sar_samples = 0
    write_progress_cb = getattr(state, "write_progress_cb", None)
    _phases_before_fdtd = getattr(state, "_phases_before_fdtd", [])
    args = state.args
    if stream_frames:
        E_buffer, SAR_buffer = [], []
        stream_part = 0
    t0 = time.perf_counter()
    for time_step in range(1, time_steps + 1):
        Dx, iDx = calculate_dx_field(
            nx,
            ny,
            nz,
            state.Dx,
            state.iDx,
            state.Hy,
            state.Hz,
            state.gj3,
            state.gk3,
            state.gj2,
            state.gk2,
            state.gi1,
        )
        state.Dx, state.iDx = Dx, iDx
        Dy, iDy = calculate_dy_field(
            nx,
            ny,
            nz,
            state.Dy,
            state.iDy,
            state.Hx,
            state.Hz,
            state.gi3,
            state.gk3,
            state.gi2,
            state.gk2,
            state.gj1,
        )
        state.Dy, state.iDy = Dy, iDy
        Dz, iDz = calculate_dz_field(
            nx,
            ny,
            nz,
            state.Dz,
            state.iDz,
            state.Hx,
            state.Hy,
            state.gi3,
            state.gj3,
            state.gi2,
            state.gj2,
            state.gk1,
        )
        state.Dz, state.iDz = Dz, iDz
        for q, qs in enumerate(opt_quad_sources):
            gi, gj, gk = qs["gap"]
            ramp = 1.0 - _exp(-0.5 * (time_step / ramp_width_opt) ** 2)
            src_val = (
                opt_source_scale
                * opt_alphas[q]
                * ramp
                * _sin(2.0 * np.pi * opt_f0 * time_step * dt + opt_thetas[q])
            )
            state.Dz[gi, gj, gk] += src_val
        Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
            nx,
            ny,
            nz,
            state.Dx,
            state.Dy,
            state.Dz,
            state.eps_x,
            state.eps_y,
            state.eps_z,
            state.conductivity_x,
            state.conductivity_y,
            state.conductivity_z,
            state.Ex,
            state.Ey,
            state.Ez,
            state.Ix,
            state.Iy,
            state.Iz,
        )
        state.Ex, state.Ey, state.Ez = Ex, Ey, Ez
        state.Ix, state.Iy, state.Iz = Ix, Iy, Iz
        if time_step >= sar_start_step:
            state.Ex_sq_sum, state.Ey_sq_sum, state.Ez_sq_sum = (
                accumulate_e_field_squared(
                    nx,
                    ny,
                    nz,
                    state.Ex,
                    state.Ey,
                    state.Ez,
                    state.Ex_sq_sum,
                    state.Ey_sq_sum,
                    state.Ez_sq_sum,
                )
            )
            n_sar_samples += 1
        Hx, iHx = calculate_hx_field(
            nx,
            ny,
            nz,
            state.Hx,
            state.iHx,
            state.Ey,
            state.Ez,
            state.fi1,
            state.fj2,
            state.fk2,
            state.fj3,
            state.fk3,
        )
        state.Hx, state.iHx = Hx, iHx
        Hy, iHy = calculate_hy_field(
            nx,
            ny,
            nz,
            state.Hy,
            state.iHy,
            state.Ex,
            state.Ez,
            state.fj1,
            state.fi2,
            state.fk2,
            state.fi3,
            state.fk3,
        )
        state.Hy, state.iHy = Hy, iHy
        Hz, iHz = calculate_hz_field(
            nx,
            ny,
            nz,
            state.Hz,
            state.iHz,
            state.Ex,
            state.Ey,
            state.fk1,
            state.fi2,
            state.fj2,
            state.fi3,
            state.fj3,
        )
        state.Hz, state.iHz = Hz, iHz
        if time_step % 500 == 0 and write_progress_cb:
            _pct = 25 + 45 * time_step / time_steps
            write_progress_cb(
                "fdtd_simulation",
                f"FDTD step {time_step} / {time_steps}",
                _pct,
                _phases_before_fdtd,
                {"time_step": time_step, "time_steps": time_steps},
            )
        if time_step % frame_interval == 0:
            sar_instant = compute_instantaneous_sar(
                nx,
                ny,
                nz,
                state.Ex,
                state.Ey,
                state.Ez,
                state.sigma_x,
                state.sigma_y,
                state.sigma_z,
                state.rho,
            )
            if stream_frames:
                E_buffer.append(state.Ez.copy())
                SAR_buffer.append(sar_instant.copy())
                streamed_n_frames += 1
                if len(E_buffer) >= STREAM_CHUNK_SIZE:
                    part_path_e = os.path.join(
                        state.E_FRAMES_DIR,
                        f"{state.OUTPUT_BASE}_E_frames_part{stream_part}.npz",
                    )
                    part_path_sar = os.path.join(
                        state.SAR_FRAMES_DIR,
                        f"{state.OUTPUT_BASE}_SAR_frames_part{stream_part}.npz",
                    )
                    np.savez_compressed(
                        part_path_e, E_frames=np.array(E_buffer, dtype=np.float32)
                    )
                    np.savez_compressed(
                        part_path_sar, SAR_frames=np.array(SAR_buffer, dtype=np.float32)
                    )
                    E_buffer.clear()
                    SAR_buffer.clear()
                    stream_part += 1
            else:
                E_frames.append(state.Ez.copy())
                SAR_frames.append(sar_instant.copy())
    if stream_frames and E_buffer:
        part_path_e = os.path.join(
            state.E_FRAMES_DIR, f"{state.OUTPUT_BASE}_E_frames_part{stream_part}.npz"
        )
        part_path_sar = os.path.join(
            state.SAR_FRAMES_DIR,
            f"{state.OUTPUT_BASE}_SAR_frames_part{stream_part}.npz",
        )
        np.savez_compressed(part_path_e, E_frames=np.array(E_buffer, dtype=np.float32))
        np.savez_compressed(
            part_path_sar, SAR_frames=np.array(SAR_buffer, dtype=np.float32)
        )
    t1 = time.perf_counter()
    if write_progress_cb:
        write_progress_cb(
            "fdtd_simulation",
            "FDTD complete",
            70,
            _phases_before_fdtd + ["fdtd_simulation"],
        )
    return (n_sar_samples, state.E_frames, state.SAR_frames, streamed_n_frames, t1 - t0)


def run_fdtd_standard_loop(state):
    """
    Run the standard FDTD time loop (single or multi source, pulse_type, prop_direction).
    Mutates state in place. Returns (n_sar_samples, E_frames, SAR_frames, streamed_n_frames, time_fdtd_s).
    """
    if compute_instantaneous_sar is None:
        raise ImportError("sar_computation is required for run_fdtd_standard_loop")
    s = state
    nx = s.simulation_size_x
    ny = s.simulation_size_y
    nz = s.simulation_size_z
    time_steps = s.time_steps
    sar_start_step = s.sar_start_step
    dt = s.dt
    pulse_type = s.pulse_type
    pulse_amplitude = s.pulse_amplitude
    pulse_width = s.pulse_width
    pulse_delay = s.pulse_delay
    pulse_freq = s.pulse_freq
    pulse_ramp_width = s.pulse_ramp_width
    number_of_frequencies = s.number_of_frequencies
    arg = s.arg
    real_in = s.real_in
    imag_in = s.imag_in
    real_pt = s.real_pt
    imag_pt = s.imag_pt
    source_z = s.source_z
    prop_direction = s.prop_direction
    point_source_positions = s.point_source_positions
    ia, ja, ka = s.ia, s.ja, s.ka
    ib, jb, kb = s.ib, s.jb, s.kb
    Dx, Dy, Dz = s.Dx, s.Dy, s.Dz
    iDx, iDy, iDz = s.iDx, s.iDy, s.iDz
    Ex, Ey, Ez = s.Ex, s.Ey, s.Ez
    Ix, Iy, Iz = s.Ix, s.Iy, s.Iz
    Hx, Hy, Hz = s.Hx, s.Hy, s.Hz
    eps_x, eps_y, eps_z = s.eps_x, s.eps_y, s.eps_z
    conductivity_x = s.conductivity_x
    conductivity_y = s.conductivity_y
    conductivity_z = s.conductivity_z
    sigma_x, sigma_y, sigma_z = s.sigma_x, s.sigma_y, s.sigma_z
    rho = s.rho
    Ex_sq_sum, Ey_sq_sum, Ez_sq_sum = s.Ex_sq_sum, s.Ey_sq_sum, s.Ez_sq_sum
    gi1, gj1, gk1 = s.gi1, s.gj1, s.gk1
    gi2, gj2, gk2 = s.gi2, s.gj2, s.gk2
    gi3, gj3, gk3 = s.gi3, s.gj3, s.gk3
    fi1, fj1, fk1 = s.fi1, s.fj1, s.fk1
    fi2, fj2, fk2 = s.fi2, s.fj2, s.fk2
    fi3, fj3, fk3 = s.fi3, s.fj3, s.fk3
    E_frames = s.E_frames
    SAR_frames = s.SAR_frames
    stream_frames = s.stream_frames
    stream_frame_interval = s.stream_frame_interval
    STREAM_CHUNK_SIZE = s.STREAM_CHUNK_SIZE
    E_FRAMES_DIR = s.E_FRAMES_DIR
    SAR_FRAMES_DIR = s.SAR_FRAMES_DIR
    OUTPUT_BASE = s.OUTPUT_BASE
    write_progress_cb = s.write_progress_cb
    _phases_before_fdtd = s._phases_before_fdtd

    boundary_low = s.boundary_low
    boundary_high = s.boundary_high
    ez_inc = s.ez_inc
    hx_inc = s.hx_inc
    inj_y = s.inj_y
    boundary_low_x = getattr(s, "boundary_low_x", None)
    boundary_high_x = getattr(s, "boundary_high_x", None)
    ez_inc_x = getattr(s, "ez_inc_x", None)
    hy_inc_x = getattr(s, "hy_inc_x", None)
    inj_x = getattr(s, "inj_x", None)
    boundary_low_z = getattr(s, "boundary_low_z", None)
    boundary_high_z = getattr(s, "boundary_high_z", None)
    ez_inc_z = getattr(s, "ez_inc_z", None)
    hx_inc_z = getattr(s, "hx_inc_z", None)
    inj_z = getattr(s, "inj_z", None)

    n_sar_samples = 0
    streamed_n_frames = 0
    if stream_frames:
        E_buffer, SAR_buffer = [], []
        stream_part = 0

    t0 = time.perf_counter()
    for time_step in range(1, time_steps + 1):
        t_dt = time_step * dt
        if pulse_type == "gaussian":
            pulse = pulse_amplitude * exp(
                -0.5 * ((pulse_delay - time_step) / pulse_width) ** 2
            )
        elif pulse_type == "modulated_gaussian":
            pulse = (
                pulse_amplitude
                * exp(-0.5 * ((pulse_delay - time_step) / pulse_width) ** 2)
                * sin(2.0 * np.pi * pulse_freq * t_dt)
            )
        elif pulse_type == "sinusoid":
            ramp = (
                1.0
                if time_step >= 2 * pulse_ramp_width
                else (1.0 - exp(-0.5 * (time_step / pulse_ramp_width) ** 2))
            )
            pulse = ramp * pulse_amplitude * sin(2.0 * np.pi * pulse_freq * t_dt)
        elif pulse_type == "sinusoid_no_ramp":
            pulse = pulse_amplitude * sin(2.0 * np.pi * pulse_freq * t_dt)
        else:
            ramp = (
                1.0
                if time_step >= 2 * pulse_ramp_width
                else (1.0 - exp(-0.5 * (time_step / pulse_ramp_width) ** 2))
            )
            pulse = ramp * pulse_amplitude * sin(2.0 * np.pi * pulse_freq * t_dt)

        if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
            for m in range(number_of_frequencies):
                real_in[m] = real_in[m] + cos(arg[m] * time_step) * pulse
                imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * pulse
        else:
            if prop_direction in ("+y", "-y"):
                for j in range(1, ny - 1):
                    ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j - 1] - hx_inc[j])
                ez_inc[0] = boundary_low.pop(0)
                boundary_low.append(ez_inc[1])
                ez_inc[ny - 1] = boundary_high.pop(0)
                boundary_high.append(ez_inc[ny - 2])
                ez_inc[inj_y] = pulse
            elif prop_direction in ("+x", "-x"):
                ez_inc_x = update_ez_inc_x(nx, ez_inc_x, hy_inc_x)
                ez_inc_x[0] = boundary_low_x.pop(0)
                boundary_low_x.append(ez_inc_x[1])
                ez_inc_x[nx - 1] = boundary_high_x.pop(0)
                boundary_high_x.append(ez_inc_x[nx - 2])
                ez_inc_x[inj_x] = pulse
                s.ez_inc_x = ez_inc_x
            else:
                ez_inc_z = update_ez_inc_z(nz, ez_inc_z, hx_inc_z)
                ez_inc_z[0] = boundary_low_z.pop(0)
                boundary_low_z.append(ez_inc_z[1])
                ez_inc_z[nz - 1] = boundary_high_z.pop(0)
                boundary_high_z.append(ez_inc_z[nz - 2])
                ez_inc_z[inj_z] = pulse
                s.ez_inc_z = ez_inc_z

            if prop_direction in ("+y", "-y"):
                for m in range(number_of_frequencies):
                    real_in[m] = real_in[m] + cos(arg[m] * time_step) * ez_inc[ja - 1]
                    imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * ez_inc[ja - 1]
            elif prop_direction in ("+x", "-x"):
                for m in range(number_of_frequencies):
                    real_in[m] = real_in[m] + cos(arg[m] * time_step) * ez_inc_x[ia - 1]
                    imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * ez_inc_x[ia - 1]
            else:
                for m in range(number_of_frequencies):
                    real_in[m] = real_in[m] + cos(arg[m] * time_step) * ez_inc_z[ka - 1]
                    imag_in[m] = imag_in[m] - sin(arg[m] * time_step) * ez_inc_z[ka - 1]

        Dx, iDx = calculate_dx_field(
            nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
        )
        Dy, iDy = calculate_dy_field(
            nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
        )
        Dz, iDz = calculate_dz_field(
            nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
        )
        if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
            pass
        elif prop_direction in ("+y", "-y"):
            Dy = calculate_inc_dy_field(ia, ib, ja, jb, ka, kb, Dy, hx_inc)
            Dz = calculate_inc_dz_field(ia, ib, ja, jb, ka, kb, Dz, hx_inc)
        elif prop_direction in ("+x", "-x"):
            Dz = calculate_inc_dz_field_x(ia, ib, ja, jb, ka, kb, Dz, hy_inc_x)
        else:
            Dz = calculate_inc_dz_field_z(ia, ib, ja, jb, ka, kb, Dz, hx_inc_z)

        Ex, Ey, Ez, Ix, Iy, Iz = calculate_e_fields(
            nx,
            ny,
            nz,
            Dx,
            Dy,
            Dz,
            eps_x,
            eps_y,
            eps_z,
            conductivity_x,
            conductivity_y,
            conductivity_z,
            Ex,
            Ey,
            Ez,
            Ix,
            Iy,
            Iz,
        )
        s.Ex, s.Ey, s.Ez = Ex, Ey, Ez

        if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
            for sx, sy, sz in point_source_positions:
                Ez[sx, sy, sz] = Ez[sx, sy, sz] + pulse
        s.Ex, s.Ey, s.Ez = Ex, Ey, Ez

        if time_step >= sar_start_step:
            Ex_sq_sum, Ey_sq_sum, Ez_sq_sum = accumulate_e_field_squared(
                nx,
                ny,
                nz,
                Ex,
                Ey,
                Ez,
                Ex_sq_sum,
                Ey_sq_sum,
                Ez_sq_sum,
            )
            n_sar_samples += 1
            s.Ex_sq_sum, s.Ey_sq_sum, s.Ez_sq_sum = Ex_sq_sum, Ey_sq_sum, Ez_sq_sum

        real_pt, imag_pt = calculate_fourier_transform_ex(
            nx,
            ny,
            number_of_frequencies,
            real_pt,
            imag_pt,
            Ez,
            arg,
            time_step,
            source_z,
        )
        s.real_pt, s.imag_pt = real_pt, imag_pt

        if pulse_type in ("cw", "sinusoid", "sinusoid_no_ramp"):
            Hx, s.iHx = calculate_hx_field(
                nx, ny, nz, Hx, s.iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
            )
            Hy, s.iHy = calculate_hy_field(
                nx, ny, nz, Hy, s.iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
            )
        elif prop_direction in ("+y", "-y"):
            hx_inc = calculate_hx_inc(ny, hx_inc, ez_inc)
            s.hx_inc = hx_inc
            Hx, s.iHx = calculate_hx_field(
                nx, ny, nz, Hx, s.iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
            )
            Hx = calculate_hx_with_incident_field(ia, ib, ja, jb, ka, kb, Hx, ez_inc)
            Hy, s.iHy = calculate_hy_field(
                nx, ny, nz, Hy, s.iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
            )
            Hy = calculate_hy_with_incident_field(ia, ib, ja, jb, ka, kb, Hy, ez_inc)
        elif prop_direction in ("+x", "-x"):
            hy_inc_x = calculate_hy_inc_x(nx, hy_inc_x, ez_inc_x)
            s.hy_inc_x = hy_inc_x
            Hx, s.iHx = calculate_hx_field(
                nx, ny, nz, Hx, s.iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
            )
            Hy, s.iHy = calculate_hy_field(
                nx, ny, nz, Hy, s.iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
            )
            Hy = calculate_hy_with_incident_field_x(
                ia, ib, ja, jb, ka, kb, Hy, ez_inc_x
            )
        else:
            hx_inc_z = calculate_hx_inc_z(nz, hx_inc_z, ez_inc_z)
            s.hx_inc_z = hx_inc_z
            Hx, s.iHx = calculate_hx_field(
                nx, ny, nz, Hx, s.iHx, Ey, Ez, fi1, fj2, fk2, fj3, fk3
            )
            Hx = calculate_hx_with_incident_field_z(
                ia, ib, ja, jb, ka, kb, Hx, ez_inc_z
            )
            Hy, s.iHy = calculate_hy_field(
                nx, ny, nz, Hy, s.iHy, Ex, Ez, fj1, fi2, fk2, fi3, fk3
            )
        s.Hx, s.Hy = Hx, Hy
        Hz, s.iHz = calculate_hz_field(
            nx, ny, nz, Hz, s.iHz, Ex, Ey, fk1, fi2, fj2, fi3, fj3
        )
        s.Hz = Hz

        if time_step % 500 == 0 and write_progress_cb is not None:
            _pct = 25 + 45 * time_step / time_steps
            write_progress_cb(
                "fdtd_simulation",
                f"FDTD step {time_step} / {time_steps}",
                _pct,
                _phases_before_fdtd,
                {"time_step": time_step, "time_steps": time_steps},
            )

        if stream_frames:
            if time_step % stream_frame_interval == 0:
                sar_instant = compute_instantaneous_sar(
                    nx,
                    ny,
                    nz,
                    Ex,
                    Ey,
                    Ez,
                    sigma_x,
                    sigma_y,
                    sigma_z,
                    rho,
                )
                E_buffer.append(Ez.copy())
                SAR_buffer.append(sar_instant.copy())
                streamed_n_frames += 1
                if len(E_buffer) >= STREAM_CHUNK_SIZE:
                    part_path_e = os.path.join(
                        E_FRAMES_DIR, f"{OUTPUT_BASE}_E_frames_part{stream_part}.npz"
                    )
                    part_path_sar = os.path.join(
                        SAR_FRAMES_DIR,
                        f"{OUTPUT_BASE}_SAR_frames_part{stream_part}.npz",
                    )
                    np.savez_compressed(
                        part_path_e, E_frames=np.array(E_buffer, dtype=np.float32)
                    )
                    np.savez_compressed(
                        part_path_sar, SAR_frames=np.array(SAR_buffer, dtype=np.float32)
                    )
                    E_buffer.clear()
                    SAR_buffer.clear()
                    stream_part += 1
        else:
            E_frames.append(Ez.copy())
            sar_instant = compute_instantaneous_sar(
                nx,
                ny,
                nz,
                Ex,
                Ey,
                Ez,
                sigma_x,
                sigma_y,
                sigma_z,
                rho,
            )
            SAR_frames.append(sar_instant.copy())

    if stream_frames and E_buffer:
        part_path_e = os.path.join(
            E_FRAMES_DIR, f"{OUTPUT_BASE}_E_frames_part{stream_part}.npz"
        )
        part_path_sar = os.path.join(
            SAR_FRAMES_DIR, f"{OUTPUT_BASE}_SAR_frames_part{stream_part}.npz"
        )
        np.savez_compressed(part_path_e, E_frames=np.array(E_buffer, dtype=np.float32))
        np.savez_compressed(
            part_path_sar, SAR_frames=np.array(SAR_buffer, dtype=np.float32)
        )

    time_fdtd_s = time.perf_counter() - t0
    if write_progress_cb is not None:
        write_progress_cb(
            "fdtd_simulation",
            "FDTD complete",
            70,
            _phases_before_fdtd + ["fdtd_simulation"],
        )
    return (n_sar_samples, E_frames, SAR_frames, streamed_n_frames, time_fdtd_s)
