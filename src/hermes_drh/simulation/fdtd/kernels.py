"""
FDTD Yee/PML kernels (thesis: FDTD Solver Implementation).

PML coefficient calculation; D, E, H field updates; E² accumulation for SAR;
Fourier transform of Ez for frequency-domain output.
Single responsibility: numerical discretization of Maxwell update equations.
"""

from math import cos, sin
import numpy as np
import numba


def calculate_pml_parameters(
    npml, simulation_size_x, simulation_size_y, simulation_size_z
):
    gi1 = np.zeros(simulation_size_x)
    gi2 = np.ones(simulation_size_x)
    gi3 = np.ones(simulation_size_x)
    fi1 = np.zeros(simulation_size_x)
    fi2 = np.ones(simulation_size_x)
    fi3 = np.ones(simulation_size_x)
    gj1 = np.zeros(simulation_size_y)
    gj2 = np.ones(simulation_size_y)
    gj3 = np.ones(simulation_size_y)
    fj1 = np.zeros(simulation_size_y)
    fj2 = np.ones(simulation_size_y)
    fj3 = np.ones(simulation_size_y)
    gk1 = np.zeros(simulation_size_z)
    gk2 = np.ones(simulation_size_z)
    gk3 = np.ones(simulation_size_z)
    fk1 = np.zeros(simulation_size_z)
    fk2 = np.ones(simulation_size_z)
    fk3 = np.ones(simulation_size_z)

    for n in range(npml):
        xxn = (npml - n) / npml
        xn = 0.33 * (xxn**3)
        fi1[n] = xn
        fi1[simulation_size_x - 1 - n] = xn
        gi2[n] = 1 / (1 + xn)
        gi2[simulation_size_x - 1 - n] = 1 / (1 + xn)
        gi3[n] = (1 - xn) / (1 + xn)
        gi3[simulation_size_x - 1 - n] = (1 - xn) / (1 + xn)
        fj1[n] = xn
        fj1[simulation_size_y - n - 1] = xn
        gj2[n] = 1 / (1 + xn)
        gj2[simulation_size_y - 1 - n] = 1 / (1 + xn)
        gj3[n] = (1 - xn) / (1 + xn)
        gj3[simulation_size_y - 1 - n] = (1 - xn) / (1 + xn)
        fk1[n] = xn
        fk1[simulation_size_z - n - 1] = xn
        gk2[n] = 1 / (1 + xn)
        gk2[simulation_size_z - 1 - n] = 1 / (1 + xn)
        gk3[n] = (1 - xn) / (1 + xn)
        gk3[simulation_size_z - 1 - n] = (1 - xn) / (1 + xn)
        xxn = (npml - n - 0.5) / npml
        xn = 0.33 * (xxn**3)
        gi1[n] = xn
        gi1[simulation_size_x - 1 - n] = xn
        fi2[n] = 1 / (1 + xn)
        fi2[simulation_size_x - 1 - n] = 1 / (1 + xn)
        fi3[n] = (1 - xn) / (1 + xn)
        fi3[simulation_size_x - 1 - n] = (1 - xn) / (1 + xn)
        gj1[n] = xn
        gj1[simulation_size_y - 1 - n] = xn
        fj2[n] = 1 / (1 + xn)
        fj2[simulation_size_y - 1 - n] = 1 / (1 + xn)
        fj3[n] = (1 - xn) / (1 + xn)
        fj3[simulation_size_y - 1 - n] = (1 - xn) / (1 + xn)
        gk1[n] = xn
        gk1[simulation_size_z - 1 - n] = xn
        fk2[n] = 1 / (1 + xn)
        fk2[simulation_size_z - 1 - n] = 1 / (1 + xn)
        fk3[n] = (1 - xn) / (1 + xn)
        fk3[simulation_size_z - 1 - n] = (1 - xn) / (1 + xn)

    return (
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
    )


@numba.jit(nopython=True)
def calculate_dx_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dx,
    iDx,
    Hy,
    Hz,
    gj3,
    gk3,
    gj2,
    gk2,
    gi1,
):
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hz[i, j, k] - Hz[i, j - 1, k] - Hy[i, j, k] + Hy[i, j, k - 1]
                iDx[i, j, k] = iDx[i, j, k] + curl_h
                Dx[i, j, k] = gj3[j] * gk3[k] * Dx[i, j, k] + gj2[j] * gk2[k] * (
                    0.5 * curl_h + gi1[i] * iDx[i, j, k]
                )
    return Dx, iDx


@numba.jit(nopython=True)
def calculate_dy_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dy,
    iDy,
    Hx,
    Hz,
    gi3,
    gk3,
    gi2,
    gk2,
    gj1,
):
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hx[i, j, k] - Hx[i, j, k - 1] - Hz[i, j, k] + Hz[i - 1, j, k]
                iDy[i, j, k] = iDy[i, j, k] + curl_h
                Dy[i, j, k] = gi3[i] * gk3[k] * Dy[i, j, k] + gi2[i] * gk2[k] * (
                    0.5 * curl_h + gj1[j] * iDy[i, j, k]
                )
    return Dy, iDy


@numba.jit(nopython=True)
def calculate_dz_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Dz,
    iDz,
    Hx,
    Hy,
    gi3,
    gj3,
    gi2,
    gj2,
    gk1,
):
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z):
                curl_h = Hy[i, j, k] - Hy[i - 1, j, k] - Hx[i, j, k] + Hx[i, j - 1, k]
                iDz[i, j, k] = iDz[i, j, k] + curl_h
                Dz[i, j, k] = gi3[i] * gj3[j] * Dz[i, j, k] + gi2[i] * gj2[j] * (
                    0.5 * curl_h + gk1[k] * iDz[i, j, k]
                )
    return Dz, iDz


@numba.jit(nopython=True)
def calculate_e_fields(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
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
):
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex[i, j, k] = eps_x[i, j, k] * (Dx[i, j, k] - Ix[i, j, k])
                Ix[i, j, k] = Ix[i, j, k] + conductivity_x[i, j, k] * Ex[i, j, k]
                Ey[i, j, k] = eps_y[i, j, k] * (Dy[i, j, k] - Iy[i, j, k])
                Iy[i, j, k] = Iy[i, j, k] + conductivity_y[i, j, k] * Ey[i, j, k]
                Ez[i, j, k] = eps_z[i, j, k] * (Dz[i, j, k] - Iz[i, j, k])
                Iz[i, j, k] = Iz[i, j, k] + conductivity_z[i, j, k] * Ez[i, j, k]
    return Ex, Ey, Ez, Ix, Iy, Iz


@numba.jit(nopython=True)
def calculate_fourier_transform_ex(
    simulation_size_x,
    simulation_size_y,
    number_of_frequencies,
    real_pt,
    imag_pt,
    Ez,
    arg,
    time_step,
    source_z,
):
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for m in range(0, number_of_frequencies):
                real_pt[m, i, j] = (
                    real_pt[m, i, j] + cos(arg[m] * time_step) * Ez[i, j, source_z]
                )
                imag_pt[m, i, j] = (
                    imag_pt[m, i, j] - sin(arg[m] * time_step) * Ez[i, j, source_z]
                )
    return real_pt, imag_pt


@numba.jit(nopython=True)
def calculate_hx_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hx,
    iHx,
    Ey,
    Ez,
    fi1,
    fj2,
    fk2,
    fj3,
    fk3,
):
    for i in range(1, simulation_size_x):
        for j in range(1, simulation_size_y - 1):
            for k in range(1, simulation_size_z - 1):
                curl_e = Ey[i, j, k + 1] - Ey[i, j, k] - Ez[i, j + 1, k] + Ez[i, j, k]
                iHx[i, j, k] = iHx[i, j, k] + curl_e
                Hx[i, j, k] = fj3[j] * fk3[k] * Hx[i, j, k] + fj2[j] * fk2[k] * 0.5 * (
                    curl_e + fi1[i] * iHx[i, j, k]
                )
    return Hx, iHx


@numba.jit(nopython=True)
def calculate_hy_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hy,
    iHy,
    Ex,
    Ez,
    fj1,
    fi2,
    fk2,
    fi3,
    fk3,
):
    for i in range(1, simulation_size_x - 1):
        for j in range(1, simulation_size_y):
            for k in range(1, simulation_size_z - 1):
                curl_e = Ez[i + 1, j, k] - Ez[i, j, k] - Ex[i, j, k + 1] + Ex[i, j, k]
                iHy[i, j, k] = iHy[i, j, k] + curl_e
                Hy[i, j, k] = fi3[i] * fk3[k] * Hy[i, j, k] + fi2[i] * fk2[k] * 0.5 * (
                    curl_e + fj1[j] * iHy[i, j, k]
                )
    return Hy, iHy


@numba.jit(nopython=True)
def calculate_hz_field(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Hz,
    iHz,
    Ex,
    Ey,
    fk1,
    fi2,
    fj2,
    fi3,
    fj3,
):
    for i in range(1, simulation_size_x - 1):
        for j in range(1, simulation_size_y - 1):
            for k in range(1, simulation_size_z):
                curl_e = Ex[i, j + 1, k] - Ex[i, j, k] - Ey[i + 1, j, k] + Ey[i, j, k]
                iHz[i, j, k] = iHz[i, j, k] + curl_e
                Hz[i, j, k] = fi3[i] * fj3[j] * Hz[i, j, k] + fi2[i] * fj2[j] * 0.5 * (
                    curl_e + fk1[k] * iHz[i, j, k]
                )
    return Hz, iHz


@numba.jit(nopython=True)
def accumulate_e_field_squared(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex,
    Ey,
    Ez,
    Ex_sq_sum,
    Ey_sq_sum,
    Ez_sq_sum,
):
    for i in range(0, simulation_size_x):
        for j in range(0, simulation_size_y):
            for k in range(0, simulation_size_z):
                Ex_sq_sum[i, j, k] = Ex_sq_sum[i, j, k] + Ex[i, j, k] ** 2
                Ey_sq_sum[i, j, k] = Ey_sq_sum[i, j, k] + Ey[i, j, k] ** 2
                Ez_sq_sum[i, j, k] = Ez_sq_sum[i, j, k] + Ez[i, j, k] ** 2
    return Ex_sq_sum, Ey_sq_sum, Ez_sq_sum
