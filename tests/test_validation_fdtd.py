import math
from types import SimpleNamespace

import numpy as np


def _courant_dt(dx_m: float, courant_factor: float = 0.99) -> float:
    c_light = 2.99792458e8
    dt_courant = 1.0 / (c_light * math.sqrt(3.0 / (dx_m * dx_m)))
    return courant_factor * dt_courant


def _build_uniform_fdtd_state(
    *,
    nx: int,
    ny: int,
    nz: int,
    dx_m: float,
    courant_factor: float,
    f0_hz: float,
    eps_r: float,
    sigma_s_per_m: float,
    rho_kg_per_m3: float = 1000.0,
    pulse_type: str = "modulated_gaussian",
    time_steps: int = 600,
    prop_direction: str = "+y",
):
    """
    Build a minimal SimpleNamespace compatible with core.fdtd.loops.run_fdtd_standard_loop.

    Notes:
    - Uses the plane-wave injection path (TFSF-like) when pulse_type is not in
      ("cw", "sinusoid", "sinusoid_no_ramp").
    - Uses uniform material properties (single medium).
    """
    from core.fdtd.kernels import calculate_pml_parameters

    dt = _courant_dt(dx_m, courant_factor=courant_factor)
    npml = max(4, min(16, min(nx, ny, nz) // 10))
    ia = ja = ka = npml
    ib, jb, kb = nx - npml - 1, ny - npml - 1, nz - npml - 1

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

    # Material arrays in the solver are "effective" coefficients (ADE-style) and are computed in the
    # real pipeline via core.materials.voxel_model.build_material_arrays:
    #   denom = eps_r + (sigma * dt / eps0)
    #   eps_coeff = 1 / denom
    #   conductivity_coeff = sigma * dt / eps0
    #
    # We mirror that mapping here so loss in the EM update is actually represented.
    eps0 = 8.854e-12
    denom = eps_r + (sigma_s_per_m * dt / eps0)
    eps_coeff = 1.0 / denom if denom > 0 else 1.0
    cond_coeff = sigma_s_per_m * dt / eps0

    eps_x = np.full((nx, ny, nz), eps_coeff, dtype=np.float64)
    eps_y = np.full((nx, ny, nz), eps_coeff, dtype=np.float64)
    eps_z = np.full((nx, ny, nz), eps_coeff, dtype=np.float64)
    conductivity_x = np.full((nx, ny, nz), cond_coeff, dtype=np.float64)
    conductivity_y = np.full((nx, ny, nz), cond_coeff, dtype=np.float64)
    conductivity_z = np.full((nx, ny, nz), cond_coeff, dtype=np.float64)

    sigma_x = np.full((nx, ny, nz), sigma_s_per_m, dtype=np.float64)
    sigma_y = np.full((nx, ny, nz), sigma_s_per_m, dtype=np.float64)
    sigma_z = np.full((nx, ny, nz), sigma_s_per_m, dtype=np.float64)
    rho = np.full((nx, ny, nz), rho_kg_per_m3, dtype=np.float64)

    Dx = np.zeros((nx, ny, nz), dtype=np.float64)
    Dy = np.zeros((nx, ny, nz), dtype=np.float64)
    Dz = np.zeros((nx, ny, nz), dtype=np.float64)
    iDx = np.zeros((nx, ny, nz), dtype=np.float64)
    iDy = np.zeros((nx, ny, nz), dtype=np.float64)
    iDz = np.zeros((nx, ny, nz), dtype=np.float64)

    Ex = np.zeros((nx, ny, nz), dtype=np.float64)
    Ey = np.zeros((nx, ny, nz), dtype=np.float64)
    Ez = np.zeros((nx, ny, nz), dtype=np.float64)
    Ix = np.zeros((nx, ny, nz), dtype=np.float64)
    Iy = np.zeros((nx, ny, nz), dtype=np.float64)
    Iz = np.zeros((nx, ny, nz), dtype=np.float64)

    Hx = np.zeros((nx, ny, nz), dtype=np.float64)
    Hy = np.zeros((nx, ny, nz), dtype=np.float64)
    Hz = np.zeros((nx, ny, nz), dtype=np.float64)
    iHx = np.zeros((nx, ny, nz), dtype=np.float64)
    iHy = np.zeros((nx, ny, nz), dtype=np.float64)
    iHz = np.zeros((nx, ny, nz), dtype=np.float64)

    # Incident-field buffers (for +y propagation path)
    hx_inc = np.zeros(ny, dtype=np.float64)
    ez_inc = np.zeros(ny, dtype=np.float64)
    boundary_low = [0.0, 0.0]
    boundary_high = [0.0, 0.0]
    inj_y = 3 if prop_direction == "+y" else ny - 4

    # Frequency-domain accumulation arrays (the pipeline uses 3 freqs; we keep 1 for tests).
    number_of_frequencies = 1
    arg = np.array([2.0 * np.pi * f0_hz * dt], dtype=np.float64)
    real_in = np.zeros(number_of_frequencies, dtype=np.float64)
    imag_in = np.zeros(number_of_frequencies, dtype=np.float64)
    # Keep the pipeline’s 4D shape to match existing behavior with numba indexing.
    real_pt = np.zeros((number_of_frequencies, nx, ny, nz), dtype=np.float64)
    imag_pt = np.zeros((number_of_frequencies, nx, ny, nz), dtype=np.float64)

    source_x = npml + 2
    source_y = ny // 2
    source_z = nz // 2
    point_source_positions = [(source_x, source_y, source_z)]

    state = SimpleNamespace(
        simulation_size_x=nx,
        simulation_size_y=ny,
        simulation_size_z=nz,
        time_steps=time_steps,
        sar_start_step=int(0.7 * time_steps),
        dt=dt,
        pulse_type=pulse_type,
        pulse_amplitude=1.0,
        pulse_width=8,
        pulse_delay=20,
        pulse_freq=f0_hz,
        pulse_ramp_width=30.0,
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
        Ex_sq_sum=np.zeros((nx, ny, nz), dtype=np.float64),
        Ey_sq_sum=np.zeros((nx, ny, nz), dtype=np.float64),
        Ez_sq_sum=np.zeros((nx, ny, nz), dtype=np.float64),
        stream_frames=False,
        stream_frame_interval=1,
        STREAM_CHUNK_SIZE=8,
        E_FRAMES_DIR="",
        SAR_FRAMES_DIR="",
        OUTPUT_BASE="validation",
        E_frames=[],
        SAR_frames=[],
        write_progress_cb=None,
        _phases_before_fdtd=[],
    )
    return state, npml


def _estimate_wavenumber_from_dft(complex_line: np.ndarray, dx_m: float) -> float:
    """
    Estimate spatial wavenumber k (rad/m) from a complex phasor sampled along a line.
    Uses phase unwrapping and a least-squares slope.
    """
    phase = np.unwrap(np.angle(complex_line))
    y = np.arange(phase.size, dtype=np.float64) * dx_m
    slope, _ = np.polyfit(y, phase, 1)
    return float(slope)


def test_plane_wave_vacuum_phase_velocity_reasonable():
    """
    Vacuum (eps_r=1, sigma=0): estimate k from DFT phase slope and check v = omega/k ~ c.
    """
    from core.fdtd.loops import run_fdtd_standard_loop

    f0 = 100e6
    dx_m = 0.01
    state, npml = _build_uniform_fdtd_state(
        nx=60,
        ny=80,
        nz=40,
        dx_m=dx_m,
        courant_factor=0.99,
        f0_hz=f0,
        eps_r=1.0,
        sigma_s_per_m=0.0,
        pulse_type="modulated_gaussian",
        time_steps=700,
        prop_direction="+y",
    )
    run_fdtd_standard_loop(state)

    cx = state.simulation_size_x // 2
    z = state.source_z
    y0 = npml + 8
    y1 = state.simulation_size_y - npml - 8
    phasor = state.real_pt[0, cx, y0:y1, z] + 1j * state.imag_pt[0, cx, y0:y1, z]
    k_hat = _estimate_wavenumber_from_dft(phasor, dx_m=dx_m)
    omega = 2.0 * np.pi * f0
    v_hat = omega / k_hat if abs(k_hat) > 1e-12 else float("inf")

    c_light = 2.99792458e8
    assert np.isfinite(v_hat)
    # Loose tolerance: coarse grid + PML + transient source.
    assert 0.7 * c_light < abs(v_hat) < 1.3 * c_light


def test_plane_wave_lossy_attenuation_trend():
    """
    Lossy uniform medium: amplitude should decay with distance (positive attenuation).
    We fit log|E| vs y and assert the fitted slope corresponds to alpha > 0.
    """
    from core.fdtd.loops import run_fdtd_standard_loop

    f0 = 100e6
    dx_m = 0.01
    eps_r = 60.0
    sigma = 0.2
    state, npml = _build_uniform_fdtd_state(
        nx=60,
        ny=90,
        nz=40,
        dx_m=dx_m,
        courant_factor=0.99,
        f0_hz=f0,
        eps_r=eps_r,
        sigma_s_per_m=sigma,
        pulse_type="modulated_gaussian",
        time_steps=900,
        prop_direction="+y",
    )
    run_fdtd_standard_loop(state)

    cx = state.simulation_size_x // 2
    z = state.source_z
    y0 = npml + 10
    y1 = state.simulation_size_y - npml - 10
    phasor = state.real_pt[0, cx, y0:y1, z] + 1j * state.imag_pt[0, cx, y0:y1, z]
    amp = np.abs(phasor)

    # Avoid the near-source growth region: fit only after the peak amplitude.
    peak_idx = int(np.argmax(amp))
    fit_start = min(peak_idx + 5, amp.size - 3)
    amp_fit = amp[fit_start:]
    y_fit = (np.arange(amp_fit.size, dtype=np.float64) + fit_start) * dx_m

    amp_fit = np.clip(amp_fit, 1e-30, None)
    slope, _ = np.polyfit(y_fit, np.log(amp_fit), 1)

    # log|E| ~ -alpha y + const => slope negative for alpha > 0.
    assert slope < 0.0


def test_energy_does_not_blow_up_under_cfl():
    """
    Basic stability check: with courant_factor<1, fields should remain finite/bounded.
    """
    from core.fdtd.loops import run_fdtd_standard_loop

    f0 = 100e6
    dx_m = 0.01
    state, _ = _build_uniform_fdtd_state(
        nx=50,
        ny=70,
        nz=30,
        dx_m=dx_m,
        courant_factor=0.99,
        f0_hz=f0,
        eps_r=1.0,
        sigma_s_per_m=0.0,
        pulse_type="modulated_gaussian",
        time_steps=500,
        prop_direction="+y",
    )
    run_fdtd_standard_loop(state)

    for arr in (state.Ex, state.Ey, state.Ez, state.Hx, state.Hy, state.Hz):
        assert np.all(np.isfinite(arr))

    emax = float(np.max(np.abs(state.Ez)))
    assert np.isfinite(emax)
    assert emax < 1e6  # sanity bound; should be far below for normalized sources


def test_grid_refinement_moves_wavenumber_toward_theory():
    """
    Run the same vacuum scenario at two dx values and ensure the estimated k gets closer
    to k_theory = omega / c as dx decreases.
    """
    from core.fdtd.loops import run_fdtd_standard_loop

    f0 = 100e6
    c_light = 2.99792458e8
    omega = 2.0 * np.pi * f0
    k_theory = omega / c_light

    def run(dx_m, nx, ny, nz, steps):
        state, npml = _build_uniform_fdtd_state(
            nx=nx,
            ny=ny,
            nz=nz,
            dx_m=dx_m,
            courant_factor=0.99,
            f0_hz=f0,
            eps_r=1.0,
            sigma_s_per_m=0.0,
            pulse_type="modulated_gaussian",
            time_steps=steps,
            prop_direction="+y",
        )
        run_fdtd_standard_loop(state)
        cx = state.simulation_size_x // 2
        z = state.source_z
        y0 = npml + 8
        y1 = state.simulation_size_y - npml - 8
        phasor = state.real_pt[0, cx, y0:y1, z] + 1j * state.imag_pt[0, cx, y0:y1, z]
        k_hat = _estimate_wavenumber_from_dft(phasor, dx_m=dx_m)
        return abs(k_hat - k_theory)

    # Keep physical size roughly comparable.
    err_coarse = run(dx_m=0.01, nx=60, ny=80, nz=40, steps=700)
    err_fine = run(dx_m=0.005, nx=90, ny=120, nz=60, steps=900)
    assert err_fine < err_coarse


def test_thermal_solver_matches_small_linear_system():
    """
    Verify solve_steady_bioheat_3d against a direct linear solve on a tiny grid.
    """
    from core.metrics.thermal import solve_steady_bioheat_3d

    nx = ny = nz = 6
    dx = 0.01
    Tb = 37.0
    k_val = 0.5
    Q_val = 1000.0

    k_3d = np.full((nx, ny, nz), k_val, dtype=np.float64)
    Q_3d = np.full((nx, ny, nz), Q_val, dtype=np.float64)

    T_gs = solve_steady_bioheat_3d(
        nx,
        ny,
        nz,
        k_3d=k_3d,
        Q_3d=Q_3d,
        dx=dx,
        T_boundary=Tb,
        max_iter=20000,
        tol=1e-10,
    )

    # Build linear system for interior points with Dirichlet boundaries at Tb.
    interior = [
        (i, j, k)
        for i in range(1, nx - 1)
        for j in range(1, ny - 1)
        for k in range(1, nz - 1)
    ]
    n = len(interior)
    idx = {p: ii for ii, p in enumerate(interior)}
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    rhs = -(Q_val * dx * dx) / k_val

    for p in interior:
        r = idx[p]
        A[r, r] = -6.0
        b[r] = rhs
        i, j, k = p
        for nb in (
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ):
            if nb in idx:
                A[r, idx[nb]] = 1.0
            else:
                # Boundary neighbor contributes known Tb.
                b[r] -= Tb

    x = np.linalg.solve(A, b)
    T_lin = np.full((nx, ny, nz), Tb, dtype=np.float64)
    for p, ii in idx.items():
        T_lin[p] = x[ii]

    max_err = float(np.max(np.abs(T_gs - T_lin)))
    assert max_err < 1e-6
