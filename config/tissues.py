"""
Tissue property tables per anatomy (eps_r, sigma, rho at ~100 MHz; k for thermal).

Used by build_material_arrays when called with optional tissue_table / k_tissue.
Format: tissue_table[label] = (eps_r, sigma_S_per_m, rho_kg_per_m3); k_tissue[label] = k_W_per_mK.
"""

# Brain (BraTS): 0=background, 1=necrotic, 2=edema, 3=enhancing, 4=normal brain
BRAIN_TISSUE_TABLE = {
    0: (1.0, 0.0, 0.0),
    1: (60.0, 0.8, 1050.0),  # necrotic tumor
    2: (60.0, 0.8, 1050.0),  # edema
    3: (60.0, 0.8, 1050.0),  # enhancing tumor
    4: (50.0, 0.6, 1046.0),  # normal brain
}
BRAIN_K_TISSUE = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}

# Breast: 0=air, 1=tumor, 2=healthy breast
BREAST_TISSUE_TABLE = {
    0: (1.0, 0.0, 0.0),
    1: (60.0, 0.8, 1050.0),  # tumor
    2: (50.0, 0.5, 1040.0),  # healthy breast
}
BREAST_K_TISSUE = {0: 0.0, 1: 0.5, 2: 0.5}

# Cervix: 0=air, 1=tumor, 2=healthy cervix (placeholder literature-based)
CERVIX_TISSUE_TABLE = {
    0: (1.0, 0.0, 0.0),
    1: (60.0, 0.8, 1050.0),  # tumor
    2: (50.0, 0.5, 1040.0),  # healthy cervix
}
CERVIX_K_TISSUE = {0: 0.0, 1: 0.5, 2: 0.5}


def get_tissue_tables(anatomy: str):
    """
    Return (tissue_table, k_tissue) for the given anatomy.

    anatomy: "brain" | "breast" | "cervix"
    Returns: (dict, dict) where tissue_table[label] = (eps_r, sigma, rho), k_tissue[label] = k.
    """
    a = anatomy.lower().strip()
    if a == "brain":
        return (dict(BRAIN_TISSUE_TABLE), dict(BRAIN_K_TISSUE))
    if a == "breast":
        return (dict(BREAST_TISSUE_TABLE), dict(BREAST_K_TISSUE))
    if a == "cervix":
        return (dict(CERVIX_TISSUE_TABLE), dict(CERVIX_K_TISSUE))
    raise ValueError(f"Unknown anatomy: {anatomy!r}. Use 'brain', 'breast', or 'cervix'.")
