"""
Antenna Optimization (thesis: Antenna Optimization).

4-quadrant APA: unit FDTD runs, field synthesis, J/J_eff, frequency/geometry sweeps,
multi-start, optimize_quadrant_controls.
"""

from math import cos, exp, sin
from itertools import product as iter_product
import json as _json_opt
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage

from hermes_drh.compat.fdtd_solver import (
    calculate_pml_parameters,
    calculate_dx_field,
    calculate_dy_field,
    calculate_dz_field,
    calculate_e_fields,
    calculate_hx_field,
    calculate_hy_field,
    calculate_hz_field,
)
from hermes_drh.compat.sar_computation import (
    compute_j_ratio,
    compute_robust_objective,
    compute_sar_from_complex_field,
)
from hermes_drh.compat.sources import build_quadrant_sources

try:
    from hermes_drh.io.validation import (
        save_optimization_comparison,
        save_optimization_trace,
    )
except ImportError:
    save_optimization_comparison = None
    save_optimization_trace = None


def run_unit_quadrant_fdtd(
    quadrant_source,
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
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
    dx,
    dt,
    epsz,
    npml,
    f0,
    time_steps_opt,
):
    """
    Run a single FDTD simulation with one quadrant's dipole active at frequency f0.
    Uses a soft sinusoidal CW source at the dipole gap, with a Gaussian ramp-up.
    Collects DFT of Ez at f0 over the full grid -> returns complex Ez field (amplitude, phase).
    Returns: complex_Ez array of shape (nx, ny, nz) at frequency f0.
    """
    nx, ny, nz = simulation_size_x, simulation_size_y, simulation_size_z
    Ex = np.zeros((nx, ny, nz))
    Ey = np.zeros((nx, ny, nz))
    Ez = np.zeros((nx, ny, nz))
    Ix = np.zeros((nx, ny, nz))
    Iy = np.zeros((nx, ny, nz))
    Iz = np.zeros((nx, ny, nz))
    Dx = np.zeros((nx, ny, nz))
    Dy = np.zeros((nx, ny, nz))
    Dz = np.zeros((nx, ny, nz))
    iDx = np.zeros((nx, ny, nz))
    iDy = np.zeros((nx, ny, nz))
    iDz = np.zeros((nx, ny, nz))
    Hx = np.zeros((nx, ny, nz))
    Hy = np.zeros((nx, ny, nz))
    Hz = np.zeros((nx, ny, nz))
    iHx = np.zeros((nx, ny, nz))
    iHy = np.zeros((nx, ny, nz))
    iHz = np.zeros((nx, ny, nz))

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

    omega_dt = 2.0 * np.pi * f0 * dt
    real_Ez = np.zeros((nx, ny, nz))
    imag_Ez = np.zeros((nx, ny, nz))

    gap_i, gap_j, gap_k = quadrant_source["gap"]
    ramp_width = 30.0

    for t in range(1, time_steps_opt + 1):
        Dx, iDx = calculate_dx_field(
            nx, ny, nz, Dx, iDx, Hy, Hz, gj3, gk3, gj2, gk2, gi1
        )
        Dy, iDy = calculate_dy_field(
            nx, ny, nz, Dy, iDy, Hx, Hz, gi3, gk3, gi2, gk2, gj1
        )
        Dz, iDz = calculate_dz_field(
            nx, ny, nz, Dz, iDz, Hx, Hy, gi3, gj3, gi2, gj2, gk1
        )
        ramp = 1.0 - exp(-0.5 * (t / ramp_width) ** 2)
        source_val = ramp * sin(2.0 * np.pi * f0 * t * dt)
        Dz[gap_i, gap_j, gap_k] += source_val

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
        cos_val = cos(omega_dt * t)
        sin_val = sin(omega_dt * t)
        real_Ez += cos_val * Ez
        imag_Ez -= sin_val * Ez

    complex_Ez = real_Ez + 1j * imag_Ez
    return complex_Ez


def synthesize_total_field(complex_fields, alphas, thetas):
    """
    Superpose complex Ez fields from N quadrants with amplitude/phase controls.
    """
    E_total = np.zeros_like(complex_fields[0])
    for q in range(len(complex_fields)):
        E_total += alphas[q] * complex_fields[q] * np.exp(1j * thetas[q])
    return E_total


def _evaluate_candidate(
    complex_fields,
    alphas,
    thetas,
    sigma_avg,
    rho,
    tumor_mask,
    healthy_mask,
    penalty_weight=0.0,
):
    """Helper: synthesize field, compute SAR, return (J_eff, J_plain, mean_tumor, mean_healthy, p95)."""
    E_total = synthesize_total_field(complex_fields, alphas, thetas)
    sar = compute_sar_from_complex_field(E_total, sigma_avg, rho)
    return compute_robust_objective(sar, tumor_mask, healthy_mask, penalty_weight)


def _eval_frequency_one(
    f0,
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    time_steps_opt,
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
    dx,
    dt,
    epsz,
    tumor_mask,
    healthy_mask,
    ring_offset,
    z_plane,
):
    """Worker: run 4 unit FDTD sims at frequency f0 with default geometry and return baseline J."""
    npml = max(
        4, min(16, min(simulation_size_x, simulation_size_y, simulation_size_z) // 10)
    )
    quad_srcs = build_quadrant_sources(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        npml=npml,
        ring_offset=ring_offset,
        z_plane=z_plane,
    )
    fields = []
    for qs in quad_srcs:
        cplx = run_unit_quadrant_fdtd(
            quadrant_source=qs,
            simulation_size_x=simulation_size_x,
            simulation_size_y=simulation_size_y,
            simulation_size_z=simulation_size_z,
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
            dx=dx,
            dt=dt,
            epsz=epsz,
            npml=npml,
            f0=f0,
            time_steps_opt=time_steps_opt,
        )
        fields.append(cplx)
    sigma_avg_arr = (sigma_x + sigma_y + sigma_z) / 3.0
    E_bl = synthesize_total_field(fields, np.ones(4), np.zeros(4))
    sar_bl = compute_sar_from_complex_field(E_bl, sigma_avg_arr, rho)
    J_bl, _, _ = compute_j_ratio(sar_bl, tumor_mask, healthy_mask)
    return {"f0": f0, "J_baseline": J_bl}


def _eval_geom_one(
    g_off,
    g_z,
    f0,
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    time_steps_opt,
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
    dx,
    dt,
    epsz,
    tumor_mask,
    healthy_mask,
):
    """Worker: run 4 unit FDTD sims for one (ring_offset, z_plane) geometry and return baseline J."""
    npml = max(
        4, min(16, min(simulation_size_x, simulation_size_y, simulation_size_z) // 10)
    )
    quad_srcs = build_quadrant_sources(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        npml=npml,
        ring_offset=g_off,
        z_plane=g_z,
    )
    fields = []
    for qs in quad_srcs:
        cplx = run_unit_quadrant_fdtd(
            quadrant_source=qs,
            simulation_size_x=simulation_size_x,
            simulation_size_y=simulation_size_y,
            simulation_size_z=simulation_size_z,
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
            dx=dx,
            dt=dt,
            epsz=epsz,
            npml=npml,
            f0=f0,
            time_steps_opt=time_steps_opt,
        )
        fields.append(cplx)
    sigma_avg_arr = (sigma_x + sigma_y + sigma_z) / 3.0
    E_bl = synthesize_total_field(fields, np.ones(4), np.zeros(4))
    sar_bl = compute_sar_from_complex_field(E_bl, sigma_avg_arr, rho)
    J_bl, _, _ = compute_j_ratio(sar_bl, tumor_mask, healthy_mask)
    gaps = [qs["gap"] for qs in quad_srcs]
    return {"ring_offset": g_off, "z_plane": g_z, "J_baseline": J_bl, "gaps": gaps}


def _run_one_multistart(
    ms_idx,
    start_offset,
    complex_fields,
    sigma_avg,
    rho,
    tumor_mask,
    healthy_mask,
    penalty_weight,
    phase_values,
    amp_values,
    amp_range,
    n_quadrants,
    refine_iterations,
    refine_factor,
    phase_steps,
    amp_steps,
):
    """Worker: run one full multi-start iteration. Returns dict with best_J, best_alphas, best_thetas, trace, eval_count, ms_idx."""
    trace = []
    eval_count = 0
    best_J = -1.0
    best_thetas = np.zeros(n_quadrants)
    best_alphas = np.ones(n_quadrants)

    for combo in iter_product(phase_values, repeat=n_quadrants - 1):
        thetas = np.array([0.0] + list(combo)) + start_offset
        alphas = np.ones(n_quadrants)
        J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
            complex_fields,
            alphas,
            thetas,
            sigma_avg,
            rho,
            tumor_mask,
            healthy_mask,
            penalty_weight,
        )
        eval_count += 1
        if J_eff > best_J:
            best_J = J_eff
            best_thetas = thetas.copy()
            best_alphas = alphas.copy()
            trace.append(
                {
                    "eval": eval_count,
                    "phase": f"phase_sweep_ms{ms_idx}",
                    "J_eff": float(J_eff),
                    "J": float(J_plain),
                    "alphas": alphas.tolist(),
                    "thetas": thetas.tolist(),
                    "p95_healthy": float(p95),
                }
            )

    for combo in iter_product(amp_values, repeat=n_quadrants - 1):
        alphas = np.array([1.0] + list(combo))
        J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
            complex_fields,
            alphas,
            best_thetas,
            sigma_avg,
            rho,
            tumor_mask,
            healthy_mask,
            penalty_weight,
        )
        eval_count += 1
        if J_eff > best_J:
            best_J = J_eff
            best_alphas = alphas.copy()
            trace.append(
                {
                    "eval": eval_count,
                    "phase": f"amp_sweep_ms{ms_idx}",
                    "J_eff": float(J_eff),
                    "J": float(J_plain),
                    "alphas": alphas.tolist(),
                    "thetas": best_thetas.tolist(),
                    "p95_healthy": float(p95),
                }
            )

    phase_delta = (2 * np.pi / phase_steps) * refine_factor
    amp_delta = ((amp_range[1] - amp_range[0]) / amp_steps) * refine_factor
    for r_iter in range(refine_iterations):
        improved = False
        for q in range(1, n_quadrants):
            for direction in [-1, 1]:
                trial_thetas = best_thetas.copy()
                trial_thetas[q] += direction * phase_delta
                J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                    complex_fields,
                    best_alphas,
                    trial_thetas,
                    sigma_avg,
                    rho,
                    tumor_mask,
                    healthy_mask,
                    penalty_weight,
                )
                eval_count += 1
                if J_eff > best_J:
                    best_J = J_eff
                    best_thetas = trial_thetas.copy()
                    improved = True
                    trace.append(
                        {
                            "eval": eval_count,
                            "phase": f"refine_{r_iter}_ms{ms_idx}",
                            "J_eff": float(J_eff),
                            "J": float(J_plain),
                            "alphas": best_alphas.tolist(),
                            "thetas": best_thetas.tolist(),
                            "p95_healthy": float(p95),
                        }
                    )
        for q in range(1, n_quadrants):
            for direction in [-1, 1]:
                trial_alphas = best_alphas.copy()
                trial_alphas[q] = np.clip(
                    trial_alphas[q] + direction * amp_delta, amp_range[0], amp_range[1]
                )
                J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                    complex_fields,
                    trial_alphas,
                    best_thetas,
                    sigma_avg,
                    rho,
                    tumor_mask,
                    healthy_mask,
                    penalty_weight,
                )
                eval_count += 1
                if J_eff > best_J:
                    best_J = J_eff
                    best_alphas = trial_alphas.copy()
                    improved = True
                    trace.append(
                        {
                            "eval": eval_count,
                            "phase": f"refine_{r_iter}_ms{ms_idx}",
                            "J_eff": float(J_eff),
                            "J": float(J_plain),
                            "alphas": best_alphas.tolist(),
                            "thetas": best_thetas.tolist(),
                            "p95_healthy": float(p95),
                        }
                    )
        phase_delta *= refine_factor
        amp_delta *= refine_factor
        if not improved:
            break

    _, J_plain_final, _, _, _ = _evaluate_candidate(
        complex_fields,
        best_alphas,
        best_thetas,
        sigma_avg,
        rho,
        tumor_mask,
        healthy_mask,
        0.0,
    )
    return {
        "ms_idx": ms_idx,
        "best_J": best_J,
        "best_J_plain": J_plain_final,
        "best_alphas": best_alphas,
        "best_thetas": best_thetas,
        "trace": trace,
        "eval_count": eval_count,
    }


def optimize_quadrant_controls(
    complex_fields,
    sigma_avg,
    rho,
    tumor_mask,
    healthy_mask,
    n_quadrants=4,
    phase_steps=24,
    amp_steps=9,
    amp_range=(0.2, 2.5),
    refine_iterations=8,
    refine_factor=0.5,
    multi_start=3,
    penalty_weight=0.0,
    parallel_workers=1,
):
    """
    Optimize amplitudes and phases of N quadrants to maximize J = SAR_tumor / SAR_healthy.
    Returns: best_alphas, best_thetas, best_J, trace (list of dicts), eval_count.
    """
    trace = []
    eval_count = 0
    global_best_J = -1.0
    global_best_Jplain = -1.0
    global_best_thetas = np.zeros(n_quadrants)
    global_best_alphas = np.ones(n_quadrants)

    np.random.seed(42)
    start_offsets = [np.zeros(n_quadrants)]
    for _ in range(multi_start - 1):
        offset = np.zeros(n_quadrants)
        offset[1:] = np.random.uniform(-np.pi, np.pi, n_quadrants - 1)
        start_offsets.append(offset)

    phase_values = np.linspace(-np.pi, np.pi, phase_steps, endpoint=False)
    amp_values = np.linspace(amp_range[0], amp_range[1], amp_steps)

    if parallel_workers > 1 and len(start_offsets) > 1:
        n_ms_workers = min(parallel_workers, len(start_offsets))
        print(
            f"\n  Running {len(start_offsets)} multi-starts in parallel ({n_ms_workers} workers)..."
        )
        with ProcessPoolExecutor(max_workers=n_ms_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_multistart,
                    ms_idx,
                    start_offset,
                    complex_fields,
                    sigma_avg,
                    rho,
                    tumor_mask,
                    healthy_mask,
                    penalty_weight,
                    phase_values,
                    amp_values,
                    amp_range,
                    n_quadrants,
                    refine_iterations,
                    refine_factor,
                    phase_steps,
                    amp_steps,
                ): ms_idx
                for ms_idx, start_offset in enumerate(start_offsets)
            }
            for future in as_completed(futures):
                ms_idx = futures[future]
                try:
                    result = future.result()
                    trace.extend(result["trace"])
                    eval_count += result["eval_count"]
                    print(
                        f"  Multi-start {result['ms_idx'] + 1}/{len(start_offsets)} done: "
                        f"J_eff={result['best_J']:.6f}, J_plain={result['best_J_plain']:.6f}"
                    )
                    if result["best_J"] > global_best_J:
                        global_best_J = result["best_J"]
                        global_best_Jplain = result["best_J_plain"]
                        global_best_alphas = result["best_alphas"].copy()
                        global_best_thetas = result["best_thetas"].copy()
                except Exception as exc:
                    print(f"  Multi-start {ms_idx + 1} error: {exc}")
        print(
            f"\n  Optimization complete: best J_eff = {global_best_J:.6f}, "
            f"best J_plain = {global_best_Jplain:.6f} ({eval_count} total evals)"
        )
        return (
            global_best_alphas,
            global_best_thetas,
            global_best_Jplain,
            trace,
            eval_count,
        )

    for ms_idx, start_offset in enumerate(start_offsets):
        print(
            f"\n  --- Multi-start {ms_idx + 1}/{len(start_offsets)} "
            f"(offset: [{', '.join(f'{np.degrees(o):.0f}°' for o in start_offset)}]) ---"
        )
        n_phase_combos = phase_steps ** (n_quadrants - 1)
        print(
            f"  Phase sweep: {phase_steps} steps/quadrant ({n_phase_combos} combos)..."
        )
        best_J = -1.0
        best_thetas = np.zeros(n_quadrants)
        best_alphas = np.ones(n_quadrants)

        for combo in iter_product(phase_values, repeat=n_quadrants - 1):
            thetas = np.array([0.0] + list(combo)) + start_offset
            alphas = np.ones(n_quadrants)
            J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                complex_fields,
                alphas,
                thetas,
                sigma_avg,
                rho,
                tumor_mask,
                healthy_mask,
                penalty_weight,
            )
            eval_count += 1
            if J_eff > best_J:
                best_J = J_eff
                best_thetas = thetas.copy()
                best_alphas = alphas.copy()
                trace.append(
                    {
                        "eval": eval_count,
                        "phase": f"phase_sweep_ms{ms_idx}",
                        "J_eff": float(J_eff),
                        "J": float(J_plain),
                        "alphas": alphas.tolist(),
                        "thetas": thetas.tolist(),
                        "p95_healthy": float(p95),
                    }
                )
        print(f"    Best J_eff after phase sweep: {best_J:.6f} ({eval_count} evals)")

        print(f"  Amplitude sweep: {amp_steps} steps/quadrant...")
        for combo in iter_product(amp_values, repeat=n_quadrants - 1):
            alphas = np.array([1.0] + list(combo))
            J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                complex_fields,
                alphas,
                best_thetas,
                sigma_avg,
                rho,
                tumor_mask,
                healthy_mask,
                penalty_weight,
            )
            eval_count += 1
            if J_eff > best_J:
                best_J = J_eff
                best_alphas = alphas.copy()
                trace.append(
                    {
                        "eval": eval_count,
                        "phase": f"amp_sweep_ms{ms_idx}",
                        "J_eff": float(J_eff),
                        "J": float(J_plain),
                        "alphas": alphas.tolist(),
                        "thetas": best_thetas.tolist(),
                        "p95_healthy": float(p95),
                    }
                )
        print(
            f"    Best J_eff after amplitude sweep: {best_J:.6f} ({eval_count} evals)"
        )

        print(f"  Coordinate refinement ({refine_iterations} iterations)...")
        phase_delta = (2 * np.pi / phase_steps) * refine_factor
        amp_delta = ((amp_range[1] - amp_range[0]) / amp_steps) * refine_factor

        for r_iter in range(refine_iterations):
            improved = False
            for q in range(1, n_quadrants):
                for direction in [-1, 1]:
                    trial_thetas = best_thetas.copy()
                    trial_thetas[q] += direction * phase_delta
                    J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                        complex_fields,
                        best_alphas,
                        trial_thetas,
                        sigma_avg,
                        rho,
                        tumor_mask,
                        healthy_mask,
                        penalty_weight,
                    )
                    eval_count += 1
                    if J_eff > best_J:
                        best_J = J_eff
                        best_thetas = trial_thetas.copy()
                        improved = True
                        trace.append(
                            {
                                "eval": eval_count,
                                "phase": f"refine_{r_iter}_ms{ms_idx}",
                                "J_eff": float(J_eff),
                                "J": float(J_plain),
                                "alphas": best_alphas.tolist(),
                                "thetas": best_thetas.tolist(),
                                "p95_healthy": float(p95),
                            }
                        )
            for q in range(1, n_quadrants):
                for direction in [-1, 1]:
                    trial_alphas = best_alphas.copy()
                    trial_alphas[q] = np.clip(
                        trial_alphas[q] + direction * amp_delta,
                        amp_range[0],
                        amp_range[1],
                    )
                    J_eff, J_plain, mt, mh, p95 = _evaluate_candidate(
                        complex_fields,
                        trial_alphas,
                        best_thetas,
                        sigma_avg,
                        rho,
                        tumor_mask,
                        healthy_mask,
                        penalty_weight,
                    )
                    eval_count += 1
                    if J_eff > best_J:
                        best_J = J_eff
                        best_alphas = trial_alphas.copy()
                        improved = True
                        trace.append(
                            {
                                "eval": eval_count,
                                "phase": f"refine_{r_iter}_ms{ms_idx}",
                                "J_eff": float(J_eff),
                                "J": float(J_plain),
                                "alphas": best_alphas.tolist(),
                                "thetas": best_thetas.tolist(),
                                "p95_healthy": float(p95),
                            }
                        )
            phase_delta *= refine_factor
            amp_delta *= refine_factor
            if not improved:
                print(
                    f"    Refinement iteration {r_iter}: no improvement, stopping early."
                )
                break

        _, J_plain_final, _, _, _ = _evaluate_candidate(
            complex_fields,
            best_alphas,
            best_thetas,
            sigma_avg,
            rho,
            tumor_mask,
            healthy_mask,
            0.0,
        )
        print(
            f"  Multi-start {ms_idx + 1} result: J_eff={best_J:.6f}, J_plain={J_plain_final:.6f}"
        )
        if best_J > global_best_J:
            global_best_J = best_J
            global_best_Jplain = J_plain_final
            global_best_thetas = best_thetas.copy()
            global_best_alphas = best_alphas.copy()

    print(
        f"\n  Optimization complete: best J_eff = {global_best_J:.6f}, "
        f"best J_plain = {global_best_Jplain:.6f} ({eval_count} total evals)"
    )
    return global_best_alphas, global_best_thetas, global_best_Jplain, trace, eval_count


def run_antenna_optimization_block(
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
    output_base,
    data_dir,
    images_dir,
    write_progress_cb=None,
    target_labels=None,
    healthy_labels=None,
    quadrant_air_margin_cells=18,
):
    """
    Run the full antenna optimization flow: masks, frequency/geometry sweep,
    optimize_quadrant_controls, validation, save artifacts.
    Returns (opt_quad_sources, opt_alphas, opt_thetas, opt_f0, antenna_optimization_s, do_geom_sweep).

    If target_labels and healthy_labels are provided, masks are built as np.isin(labels_3d, ...).
    Otherwise defaults to brain: target=[1,2,3], healthy=[4].
    If quadrant_air_margin_cells > 0, geometry candidates are filtered so that the
    4 quadrant GAP voxels stay at least this many voxel cells away from any tissue
    voxel (labels_3d != 0).
    """
    if write_progress_cb is None:

        def write_progress_cb(*args, **kwargs):
            pass

    t_opt_start = time.perf_counter()
    if target_labels is not None and healthy_labels is not None:
        opt_tumor_mask = np.isin(labels_3d, target_labels)
        opt_healthy_mask = np.isin(labels_3d, healthy_labels)
    else:
        opt_tumor_mask = (labels_3d >= 1) & (labels_3d <= 3)
        opt_healthy_mask = labels_3d == 4
    if not np.any(opt_healthy_mask):
        opt_healthy_mask = (labels_3d == 0) & (rho > 0)
    sigma_avg_opt = (sigma_x + sigma_y + sigma_z) / 3.0
    _tumor_coords = np.argwhere(opt_tumor_mask)
    tumor_centroid_z = (
        int(np.mean(_tumor_coords[:, 2]))
        if len(_tumor_coords) > 0
        else simulation_size_z // 2
    )

    margin_cells = int(quadrant_air_margin_cells)
    baseline_z_plane = simulation_size_z // 2
    baseline_ring_offset = npml + 2
    air_to_tissue_dist = None

    def _gap_min_air_to_tissue_dist(g_off, z_plane_idx):
        """Min distance-to-tissue for the 4 GAP voxels for a given geometry."""
        if air_to_tissue_dist is None:
            return float("inf")
        cx, cy = simulation_size_x // 2, simulation_size_y // 2
        gaps = (
            (cx, g_off, z_plane_idx),
            (simulation_size_x - g_off - 1, cy, z_plane_idx),
            (cx, simulation_size_y - g_off - 1, z_plane_idx),
            (g_off, cy, z_plane_idx),
        )
        for gi, gj, gk in gaps:
            if gi < 0 or gi >= simulation_size_x:
                return -1.0
            if gj < 0 or gj >= simulation_size_y:
                return -1.0
            if gk < 0 or gk >= simulation_size_z:
                return -1.0
        return min(float(air_to_tissue_dist[gi, gj, gk]) for gi, gj, gk in gaps)

    if margin_cells > 0:
        tissue_mask = labels_3d != 0
        air_mask = ~tissue_mask
        # Distance from each AIR voxel to the nearest TISSUE voxel.
        air_to_tissue_dist = ndimage.distance_transform_edt(air_mask)

        offset_min = max(0, npml + 1)
        offset_max = min(simulation_size_x - npml - 2, simulation_size_y - npml - 2)
        offset_max = max(offset_min, int(offset_max))

        start_offset = npml + 2
        best_offset = start_offset
        best_meeting_abs = 10**9
        best_meeting_min_dist = -1.0
        found_meeting = False

        best_any_min_dist = -1.0
        best_any_offset = start_offset

        for off in range(offset_min, offset_max + 1):
            min_dist = _gap_min_air_to_tissue_dist(off, baseline_z_plane)
            if min_dist < 0:
                continue
            if min_dist >= margin_cells:
                found_meeting = True
                abs_to_start = abs(off - start_offset)
                if abs_to_start < best_meeting_abs or (
                    abs_to_start == best_meeting_abs
                    and min_dist > best_meeting_min_dist
                ):
                    best_meeting_abs = abs_to_start
                    best_meeting_min_dist = min_dist
                    best_offset = off
            if min_dist > best_any_min_dist:
                best_any_min_dist = min_dist
                best_any_offset = off

        if found_meeting:
            baseline_ring_offset = int(best_offset)
        else:
            baseline_ring_offset = int(best_any_offset)
            print(
                f"\nWarning: could not satisfy quadrant air margin for all 4 gaps "
                f"(requested={margin_cells} cells). "
                f"Using best baseline ring_offset={baseline_ring_offset} with "
                f"min_air_to_tissue_dist={best_any_min_dist:.2f}."
            )

    def _run_unit_fields(f0_val, quad_srcs):
        fields = []
        for qs in quad_srcs:
            cplx = run_unit_quadrant_fdtd(
                quadrant_source=qs,
                simulation_size_x=simulation_size_x,
                simulation_size_y=simulation_size_y,
                simulation_size_z=simulation_size_z,
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
                dx=dx,
                dt=dt,
                epsz=epsz,
                npml=npml,
                f0=f0_val,
                time_steps_opt=args.opt_time_steps,
            )
            fields.append(cplx)
        return fields

    def _quick_baseline_J(fields):
        E_bl = synthesize_total_field(fields, np.ones(4), np.zeros(4))
        sar_bl = compute_sar_from_complex_field(E_bl, sigma_avg_opt, rho)
        J_bl, _, _ = compute_j_ratio(sar_bl, opt_tumor_mask, opt_healthy_mask)
        return J_bl

    write_progress_cb(
        "antenna_optimization",
        "Antenna optimization starting...",
        15,
        ["setup", "segmentation"],
    )
    print("\n" + "=" * 72)
    print("ANTENNA OPTIMIZATION MODE (4-quadrant APA, Houle Ch.6 style)")
    print("=" * 72)
    print(f"  Tumor voxels for J:   {np.sum(opt_tumor_mask)}")
    print(f"  Healthy voxels for J: {np.sum(opt_healthy_mask)}")
    print(f"  Tumor centroid z-index: {tumor_centroid_z}")

    n_workers_sweep = min(args.opt_parallel, 4) if args.opt_parallel > 1 else 1
    freq_sweep_list = getattr(args, "opt_freq_sweep", None)
    if freq_sweep_list is not None and len(freq_sweep_list) > 1:
        from concurrent.futures import as_completed

        print(f"\n  FREQUENCY SWEEP over {len(freq_sweep_list)} frequencies...")
        if n_workers_sweep > 1:
            with ProcessPoolExecutor(max_workers=n_workers_sweep) as executor:
                futures = {
                    executor.submit(
                        _eval_frequency_one,
                        f0_cand,
                        simulation_size_x,
                        simulation_size_y,
                        simulation_size_z,
                        args.opt_time_steps,
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
                        dx,
                        dt,
                        epsz,
                        opt_tumor_mask,
                        opt_healthy_mask,
                        baseline_ring_offset,
                        baseline_z_plane,
                    ): f0_cand
                    for f0_cand in freq_sweep_list
                }
                freq_results = []
                for future in as_completed(futures):
                    f0_cand = futures[future]
                    try:
                        r = future.result()
                        freq_results.append(r)
                        print(
                            f"  [f0={f0_cand/1e6:.1f} MHz] Baseline J = {r['J_baseline']:.6f}"
                        )
                    except Exception as e:
                        print(f"  [f0={f0_cand/1e6:.1f} MHz] Error: {e}")
        else:
            freq_results = []
            for f0_cand in freq_sweep_list:
                print(f"\n  --- f0 = {f0_cand/1e6:.1f} MHz ---")
                qs_default = build_quadrant_sources(
                    simulation_size_x,
                    simulation_size_y,
                    simulation_size_z,
                    npml=npml,
                    ring_offset=baseline_ring_offset,
                    z_plane=baseline_z_plane,
                )
                fields_cand = _run_unit_fields(f0_cand, qs_default)
                J_cand = _quick_baseline_J(fields_cand)
                freq_results.append({"f0": f0_cand, "J_baseline": J_cand})
                print(f"    Baseline J = {J_cand:.6f}")
        freq_results.sort(key=lambda x: x["J_baseline"], reverse=True)
        best_f0 = freq_results[0]["f0"]
        print(f"\n  Frequency sweep result: best f0 = {best_f0/1e6:.1f} MHz")
        freq_sweep_path = os.path.join(data_dir, f"{output_base}_freq_sweep.json")
        with open(freq_sweep_path, "w") as f:
            _json_opt.dump(freq_results, f, indent=2)
        args.f0 = best_f0
    elif freq_sweep_list is not None and len(freq_sweep_list) == 1:
        args.f0 = freq_sweep_list[0]

    if getattr(args, "opt_geom_offsets", None) is not None:
        geom_offsets = args.opt_geom_offsets
    elif getattr(args, "opt_geom_zplanes", None) is not None:
        geom_offsets = [8, 10, 12]
    else:
        geom_offsets = [8, 10, 12]
    geom_zplanes = (
        getattr(args, "opt_geom_zplanes", None)
        if getattr(args, "opt_geom_zplanes", None) is not None
        else [None]
    )
    do_geom_sweep = (
        getattr(args, "opt_geom_offsets", None) is not None
        or getattr(args, "opt_geom_zplanes", None) is not None
        or getattr(args, "optimize_antenna", False)
    )
    _best_offset = None
    _best_zplane = None
    quad_sources = build_quadrant_sources(
        simulation_size_x,
        simulation_size_y,
        simulation_size_z,
        npml=npml,
        ring_offset=baseline_ring_offset,
    )
    complex_fields = None

    if do_geom_sweep:
        geom_combos = [(g_off, g_z) for g_off in geom_offsets for g_z in geom_zplanes]
        if margin_cells > 0 and air_to_tissue_dist is not None:
            filtered = []
            for g_off, g_z in geom_combos:
                z_val = baseline_z_plane if g_z is None else int(g_z)
                min_dist = _gap_min_air_to_tissue_dist(g_off, z_val)
                if min_dist >= margin_cells:
                    filtered.append((g_off, g_z))
            if filtered:
                geom_combos = filtered
            else:
                print(
                    "\nWarning: margin filtering removed all geometry candidates for the "
                    f"4 quadrant GAPs (requested={margin_cells} cells). "
                    "Proceeding with unfiltered candidates."
                )
        print(
            f"\n  GEOMETRY SWEEP: offsets={geom_offsets}, z-planes={geom_zplanes} ({len(geom_combos)} combos)"
        )
        geom_results = []
        for g_off, g_z in geom_combos:
            qs_cand = build_quadrant_sources(
                simulation_size_x,
                simulation_size_y,
                simulation_size_z,
                npml=npml,
                ring_offset=g_off,
                z_plane=g_z,
            )
            fields_cand = _run_unit_fields(args.f0, qs_cand)
            J_cand = _quick_baseline_J(fields_cand)
            geom_results.append(
                {
                    "ring_offset": g_off,
                    "z_plane": g_z,
                    "J_baseline": J_cand,
                    "gaps": [qs["gap"] for qs in qs_cand],
                }
            )
            print(f"    [offset={g_off}, z={g_z}] Baseline J = {J_cand:.6f}")
        geom_results.sort(key=lambda x: x["J_baseline"], reverse=True)
        best_geom = geom_results[0]
        _best_offset = best_geom["ring_offset"]
        _best_zplane = best_geom["z_plane"]
        quad_sources = build_quadrant_sources(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            npml=npml,
            ring_offset=_best_offset,
            z_plane=_best_zplane,
        )
        print(f"\n  Running 4 unit-excitation FDTD sims (best geometry)...")
        complex_fields = _run_unit_fields(args.f0, quad_sources)
        best_alphas, best_thetas, best_J, opt_trace, total_evals = (
            optimize_quadrant_controls(
                complex_fields=complex_fields,
                sigma_avg=sigma_avg_opt,
                rho=rho,
                tumor_mask=opt_tumor_mask,
                healthy_mask=opt_healthy_mask,
                n_quadrants=4,
                phase_steps=args.opt_phase_steps,
                amp_steps=args.opt_amp_steps,
                amp_range=(args.opt_amp_min, args.opt_amp_max),
                refine_iterations=args.opt_refine_iters,
                multi_start=args.opt_multi_start,
                penalty_weight=args.opt_penalty_weight,
                parallel_workers=args.opt_parallel,
            )
        )
    else:
        quad_sources = build_quadrant_sources(
            simulation_size_x,
            simulation_size_y,
            simulation_size_z,
            npml=npml,
            ring_offset=baseline_ring_offset,
        )
        print(f"\n  Running 4 unit-excitation FDTD sims...")
        complex_fields = _run_unit_fields(args.f0, quad_sources)
        best_alphas, best_thetas, best_J, opt_trace, total_evals = (
            optimize_quadrant_controls(
                complex_fields=complex_fields,
                sigma_avg=sigma_avg_opt,
                rho=rho,
                tumor_mask=opt_tumor_mask,
                healthy_mask=opt_healthy_mask,
                n_quadrants=4,
                phase_steps=args.opt_phase_steps,
                amp_steps=args.opt_amp_steps,
                amp_range=(args.opt_amp_min, args.opt_amp_max),
                refine_iterations=args.opt_refine_iters,
                multi_start=args.opt_multi_start,
                penalty_weight=args.opt_penalty_weight,
                parallel_workers=args.opt_parallel,
            )
        )

    baseline_alphas = np.ones(4)
    baseline_thetas = np.zeros(4)
    E_baseline = synthesize_total_field(
        complex_fields, baseline_alphas, baseline_thetas
    )
    sar_baseline = compute_sar_from_complex_field(E_baseline, sigma_avg_opt, rho)
    J_baseline, mt_bl, mh_bl = compute_j_ratio(
        sar_baseline, opt_tumor_mask, opt_healthy_mask
    )
    _, _, _, _, p95_bl = compute_robust_objective(
        sar_baseline, opt_tumor_mask, opt_healthy_mask
    )
    E_optimized = synthesize_total_field(complex_fields, best_alphas, best_thetas)
    sar_optimized = compute_sar_from_complex_field(E_optimized, sigma_avg_opt, rho)
    J_opt, mt_opt, mh_opt = compute_j_ratio(
        sar_optimized, opt_tumor_mask, opt_healthy_mask
    )
    _, _, _, _, p95_opt = compute_robust_objective(
        sar_optimized, opt_tumor_mask, opt_healthy_mask
    )
    t_opt_end = time.perf_counter()
    antenna_optimization_s = round(float(t_opt_end - t_opt_start), 4)

    print("\n" + "-" * 60)
    print("ANTENNA OPTIMIZATION RESULTS")
    print("-" * 60)
    print(f"  Baseline  J = {J_baseline:.6f}   Optimized J = {J_opt:.6f}")

    opt_results = {
        "f0_Hz": args.f0,
        "time_steps_per_quadrant": args.opt_time_steps,
        "n_quadrants": 4,
        "quadrant_gaps": [qs["gap"] for qs in quad_sources],
        "geometry": {"ring_offset": _best_offset, "z_plane": _best_zplane},
        "baseline": {
            "alphas": baseline_alphas.tolist(),
            "thetas_rad": baseline_thetas.tolist(),
            "J": float(J_baseline),
        },
        "optimized": {
            "alphas": best_alphas.tolist(),
            "thetas_rad": best_thetas.tolist(),
            "J": float(J_opt),
        },
        "total_evaluations": total_evals,
        "total_time_s": float(t_opt_end - t_opt_start),
    }
    opt_results_path = os.path.join(
        data_dir, f"{output_base}_antenna_optimization.json"
    )
    with open(opt_results_path, "w") as f:
        _json_opt.dump(opt_results, f, indent=2)
    opt_trace_path = os.path.join(data_dir, f"{output_base}_optimization_trace.json")
    with open(opt_trace_path, "w") as f:
        _json_opt.dump(opt_trace, f, indent=2)
    np.save(os.path.join(data_dir, f"{output_base}_sar_optimized.npy"), sar_optimized)
    np.save(os.path.join(data_dir, f"{output_base}_sar_baseline.npy"), sar_baseline)
    if save_optimization_comparison and save_optimization_trace:
        mid_z = simulation_size_z // 2
        save_optimization_comparison(
            sar_baseline,
            sar_optimized,
            J_baseline,
            J_opt,
            mid_z,
            output_base,
            images_dir,
            args.f0 / 1e6,
        )
        save_optimization_trace(opt_trace, J_baseline, output_base, images_dir)
    write_progress_cb(
        "antenna_optimization",
        "Antenna optimization complete",
        25,
        ["setup", "segmentation", "antenna_optimization"],
    )
    return (
        quad_sources,
        best_alphas,
        best_thetas,
        args.f0,
        antenna_optimization_s,
        do_geom_sweep,
    )
