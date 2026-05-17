"""
SAR computation (thesis: SAR Computation).

Instantaneous SAR, RMS SAR from E² sums, phasor SAR from complex field.
J-ratio and penalized objective for antenna optimization.
Single responsibility: SAR formulas and optimization metrics.
"""

import numpy as np
import numba


@numba.jit(nopython=True)
def compute_instantaneous_sar(
    simulation_size_x,
    simulation_size_y,
    simulation_size_z,
    Ex,
    Ey,
    Ez,
    sigma_x,
    sigma_y,
    sigma_z,
    rho,
):
    """
    Compute instantaneous SAR from current E-field values
    SAR = σ|E|² / (2ρ)
    Units: W/kg
    """
    sar = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                e_mag_sq = Ex[i, j, k] ** 2 + Ey[i, j, k] ** 2 + Ez[i, j, k] ** 2
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_mag_sq) / (2.0 * rho[i, j, k])
                else:
                    sar[i, j, k] = 0.0
    return sar


@numba.jit(nopython=True)
def compute_sar(
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
    n_samples,
):
    """
    Compute Specific Absorption Rate (SAR) using RMS values
    SAR = σ|E_rms|² / (2ρ)
    Units: W/kg
    """
    sar = np.zeros((simulation_size_x, simulation_size_y, simulation_size_z))

    for i in range(simulation_size_x):
        for j in range(simulation_size_y):
            for k in range(simulation_size_z):
                e_rms_sq = (
                    Ex_sq_sum[i, j, k] + Ey_sq_sum[i, j, k] + Ez_sq_sum[i, j, k]
                ) / n_samples
                sigma_avg = (
                    sigma_x[i, j, k] + sigma_y[i, j, k] + sigma_z[i, j, k]
                ) / 3.0
                if rho[i, j, k] > 0:
                    sar[i, j, k] = (sigma_avg * e_rms_sq) / (2.0 * rho[i, j, k])
                else:
                    sar[i, j, k] = 0.0
    return sar


def compute_sar_from_complex_field(E_total, sigma_avg, rho):
    """
    SAR = sigma * |E_total|^2 / (2 * rho), voxel-wise.
    sigma_avg, rho: 3D arrays. Returns SAR array.
    """
    E_mag_sq = np.abs(E_total) ** 2
    sar = np.zeros_like(E_mag_sq)
    tissue = rho > 0
    sar[tissue] = sigma_avg[tissue] * E_mag_sq[tissue] / (2.0 * rho[tissue])
    return sar


def compute_j_ratio(sar, tumor_mask, healthy_mask):
    """
    J = mean SAR in tumor / mean SAR in healthy tissue.
    Returns J (float), mean_tumor, mean_healthy.
    """
    mean_tumor = np.mean(sar[tumor_mask]) if np.any(tumor_mask) else 0.0
    mean_healthy = np.mean(sar[healthy_mask]) if np.any(healthy_mask) else 1e-30
    if mean_healthy < 1e-30:
        mean_healthy = 1e-30
    J = mean_tumor / mean_healthy
    return J, mean_tumor, mean_healthy


def compute_robust_objective(sar, tumor_mask, healthy_mask, penalty_weight=0.0):
    """
    Robust objective: J_eff = J - penalty_weight * P95_healthy / mean_tumor.
    Returns J_eff, J_plain, mean_tumor, mean_healthy, p95_healthy.
    """
    mean_tumor = np.mean(sar[tumor_mask]) if np.any(tumor_mask) else 0.0
    mean_healthy = np.mean(sar[healthy_mask]) if np.any(healthy_mask) else 1e-30
    if mean_healthy < 1e-30:
        mean_healthy = 1e-30
    J_plain = mean_tumor / mean_healthy
    p95_healthy = (
        float(np.percentile(sar[healthy_mask], 95)) if np.any(healthy_mask) else 0.0
    )
    if penalty_weight > 0 and mean_tumor > 0:
        penalty = penalty_weight * p95_healthy / mean_tumor
    else:
        penalty = 0.0
    J_eff = J_plain - penalty
    return J_eff, J_plain, mean_tumor, mean_healthy, p95_healthy
