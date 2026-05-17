"""
Thermal solver (thesis: Thermal Solver Implementation).

Simplified Pennes steady-state (no perfusion): ∇·(k∇T) + Q = 0.
Gauss-Seidel iteration; Dirichlet boundary conditions.
Single responsibility: steady-state bioheat PDE.
"""

import numpy as np


def solve_steady_bioheat_3d(
    nx,
    ny,
    nz,
    k_3d,
    Q_3d,
    dx,
    T_boundary=37.0,
    max_iter=50000,
    tol=1e-6,
):
    """
    Solve simplified Pennes steady-state (no perfusion): ∇·(k∇T) + Q = 0.
    Q = SAR·ρ (W/m³). Dirichlet T = T_boundary at domain boundaries.
    Central-difference Laplacian; Gauss-Seidel iteration. Only updates voxels with k > 0.
    """
    T = np.full((nx, ny, nz), T_boundary, dtype=np.float64)
    dx2 = dx * dx
    for _ in range(max_iter):
        T_old = T.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    if k_3d[i, j, k] <= 0:
                        T[i, j, k] = T_boundary
                        continue
                    kc = k_3d[i, j, k]
                    q = Q_3d[i, j, k]
                    s = (
                        T[i + 1, j, k]
                        + T[i - 1, j, k]
                        + T[i, j + 1, k]
                        + T[i, j - 1, k]
                        + T[i, j, k + 1]
                        + T[i, j, k - 1]
                    )
                    T[i, j, k] = (s + q * dx2 / kc) / 6.0
        err = np.max(np.abs(T - T_old))
        if err < tol:
            break
    return T
