"""
FDTD boundary and incident-field helpers (thesis: Source Injection, TFSF).

Incident field updates and coupling for plane-wave (y, x, z propagation).
Single responsibility: TFSF/ABC incident field logic.
"""

import numpy as np
import numba


@numba.jit(nopython=True)
def calculate_inc_dy_field(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Dy,
    hx_inc,
):
    """Add incident field to Dy at TFSF planes (y propagation)."""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Dy[i, j, ka] = Dy[i, j, ka] - 0.5 * hx_inc[j]
            Dy[i, j, kb + 1] = Dy[i, j, kb + 1] + 0.5 * hx_inc[j]
    return Dy


@numba.jit(nopython=True)
def calculate_inc_dz_field(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Dz,
    hx_inc,
):
    """Add incident field to Dz at TFSF planes (y propagation)."""
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Dz[i, ja, k] = Dz[i, ja, k] + 0.5 * hx_inc[ja - 1]
            Dz[i, jb, k] = Dz[i, jb, k] - 0.5 * hx_inc[jb]
    return Dz


@numba.jit(nopython=True)
def calculate_hx_inc(
    simulation_size_y,
    hx_inc,
    ez_inc,
):
    """1D incident Hx update for propagation along y."""
    for j in range(0, simulation_size_y - 1):
        hx_inc[j] = hx_inc[j] + 0.5 * (ez_inc[j] - ez_inc[j + 1])
    return hx_inc


@numba.jit(nopython=True)
def calculate_hx_with_incident_field(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Hx,
    ez_inc,
):
    """Add incident Ez (y propagation) to Hx at TFSF planes."""
    for i in range(ia, ib + 1):
        for k in range(ka, kb + 1):
            Hx[i, ja - 1, k] = Hx[i, ja - 1, k] + 0.5 * ez_inc[ja]
            Hx[i, jb, k] = Hx[i, jb, k] - 0.5 * ez_inc[jb]
    return Hx


@numba.jit(nopython=True)
def calculate_hy_with_incident_field(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Hy,
    ez_inc,
):
    """Add incident Ez (y propagation) to Hy at TFSF planes."""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Hy[ia - 1, j, k] = Hy[ia - 1, j, k] - 0.5 * ez_inc[j - 1]
            Hy[ib, j, k] = Hy[ib, j, k] + 0.5 * ez_inc[j]
    return Hy


@numba.jit(nopython=True)
def update_ez_inc_x(
    simulation_size_x,
    ez_inc_x,
    hy_inc_x,
):
    """1D incident Ez update for propagation along x (dEz/dt from dHy/dx)."""
    for i in range(1, simulation_size_x - 1):
        ez_inc_x[i] = ez_inc_x[i] + 0.5 * (hy_inc_x[i - 1] - hy_inc_x[i])
    return ez_inc_x


@numba.jit(nopython=True)
def calculate_hy_inc_x(
    simulation_size_x,
    hy_inc_x,
    ez_inc_x,
):
    """1D incident Hy update for propagation along x (dHy/dt from dEz/dx)."""
    for i in range(0, simulation_size_x - 1):
        hy_inc_x[i] = hy_inc_x[i] + 0.5 * (ez_inc_x[i] - ez_inc_x[i + 1])
    return hy_inc_x


@numba.jit(nopython=True)
def calculate_inc_dz_field_x(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Dz,
    hy_inc_x,
):
    """Add incident field (x propagation) to Dz at planes i=ia and i=ib+1."""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Dz[ia, j, k] = Dz[ia, j, k] + 0.5 * hy_inc_x[ia - 1]
            Dz[ib + 1, j, k] = Dz[ib + 1, j, k] - 0.5 * hy_inc_x[ib]
    return Dz


@numba.jit(nopython=True)
def calculate_hy_with_incident_field_x(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Hy,
    ez_inc_x,
):
    """Add incident Ez (x propagation) to Hy at planes i=ia-1 and i=ib."""
    for j in range(ja, jb + 1):
        for k in range(ka, kb + 1):
            Hy[ia - 1, j, k] = Hy[ia - 1, j, k] - 0.5 * ez_inc_x[ia]
            Hy[ib, j, k] = Hy[ib, j, k] + 0.5 * ez_inc_x[ib]
    return Hy


@numba.jit(nopython=True)
def update_ez_inc_z(
    simulation_size_z,
    ez_inc_z,
    hx_inc_z,
):
    """1D incident Ez update for propagation along z (dEz/dt from dHx/dz)."""
    for k in range(1, simulation_size_z - 1):
        ez_inc_z[k] = ez_inc_z[k] + 0.5 * (hx_inc_z[k - 1] - hx_inc_z[k])
    return ez_inc_z


@numba.jit(nopython=True)
def calculate_hx_inc_z(
    simulation_size_z,
    hx_inc_z,
    ez_inc_z,
):
    """1D incident Hx update for propagation along z (dHx/dt from dEz/dz)."""
    for k in range(0, simulation_size_z - 1):
        hx_inc_z[k] = hx_inc_z[k] + 0.5 * (ez_inc_z[k] - ez_inc_z[k + 1])
    return hx_inc_z


@numba.jit(nopython=True)
def calculate_inc_dz_field_z(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Dz,
    hx_inc_z,
):
    """Add incident field (z propagation) to Dz at planes k=ka and k=kb+1."""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Dz[i, j, ka] = Dz[i, j, ka] + 0.5 * hx_inc_z[ka - 1]
            Dz[i, j, kb + 1] = Dz[i, j, kb + 1] - 0.5 * hx_inc_z[kb]
    return Dz


@numba.jit(nopython=True)
def calculate_hx_with_incident_field_z(
    ia,
    ib,
    ja,
    jb,
    ka,
    kb,
    Hx,
    ez_inc_z,
):
    """Add incident Ez (z propagation) to Hx at planes k=ka-1 and k=kb."""
    for i in range(ia, ib + 1):
        for j in range(ja, jb + 1):
            Hx[i, j, ka - 1] = Hx[i, j, ka - 1] + 0.5 * ez_inc_z[ka]
            Hx[i, j, kb] = Hx[i, j, kb] - 0.5 * ez_inc_z[kb]
    return Hx
