"""
Voxel-based tissue modeling (thesis: Voxel-based Tissue Modeling).

Tissue property tables and material array construction from segmentation labels.
BraTS labels: 0=background, 1=necrotic, 2=edema, 3=enhancing, 4=normal brain.
Single responsibility: label -> eps/sigma/rho/k mapping.
"""

import numpy as np


# Tissue properties at ~100 MHz. Format: (eps_r, sigma (S/m), rho (kg/m³)).
TISSUE_TABLE = {
    0: (1.0, 0.0, 0.0),  # background = free space (air); no SAR
    1: (60.0, 0.8, 1050.0),  # necrotic tumor
    2: (60.0, 0.8, 1050.0),  # edema
    3: (60.0, 0.8, 1050.0),  # enhancing tumor
    4: (50.0, 0.6, 1046.0),  # normal brain (parenchyma-like average GM/WM)
}

# Thermal conductivity k (W/(m·K)) for steady-state bioheat
K_TISSUE = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}


def build_material_arrays(labels_3d, dt, epsz, tissue_table=None, k_tissue=None):
    """
    Build FDTD/ADE material arrays from segmentation labels.

    Parameters
    ----------
    labels_3d : array
        3D int array of tissue labels.
    dt, epsz : float
        Time step and vacuum permittivity.
    tissue_table : dict, optional
        label -> (eps_r, sigma_S_per_m, rho_kg_per_m3). If None, use module-level TISSUE_TABLE (BraTS).
    k_tissue : dict, optional
        label -> k_W_per_mK. If None, use module-level K_TISSUE.

    Returns
    -------
    tuple
        (eps_x, eps_y, eps_z, conductivity_x, conductivity_y, conductivity_z,
         sigma_x, sigma_y, sigma_z, rho, k_3d). All 3D arrays matching labels_3d shape.
    """
    tbl = tissue_table if tissue_table is not None else TISSUE_TABLE
    kt = k_tissue if k_tissue is not None else K_TISSUE

    nx, ny, nz = labels_3d.shape
    eps_x = np.ones((nx, ny, nz))
    eps_y = np.ones((nx, ny, nz))
    eps_z = np.ones((nx, ny, nz))
    conductivity_x = np.zeros((nx, ny, nz))
    conductivity_y = np.zeros((nx, ny, nz))
    conductivity_z = np.zeros((nx, ny, nz))
    sigma_x = np.zeros((nx, ny, nz))
    sigma_y = np.zeros((nx, ny, nz))
    sigma_z = np.zeros((nx, ny, nz))
    rho = np.zeros((nx, ny, nz))
    k_3d = np.zeros((nx, ny, nz))

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                lab = int(labels_3d[i, j, k])
                if lab not in tbl:
                    lab = 0
                eps_r, sigma_val, rho_val = tbl[lab]
                k_3d[i, j, k] = kt.get(lab, 0.0)
                denom = eps_r + (sigma_val * dt / epsz)
                c = 1.0 / denom if denom > 0 else 1.0
                cond = sigma_val * dt / epsz
                eps_x[i, j, k] = c
                eps_y[i, j, k] = c
                eps_z[i, j, k] = c
                conductivity_x[i, j, k] = cond
                conductivity_y[i, j, k] = cond
                conductivity_z[i, j, k] = cond
                sigma_x[i, j, k] = sigma_val
                sigma_y[i, j, k] = sigma_val
                sigma_z[i, j, k] = sigma_val
                rho[i, j, k] = rho_val

    return (
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
    )
