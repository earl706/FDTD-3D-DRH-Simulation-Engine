"""
Source geometry for FDTD (thesis: Source Injection and Antenna Modeling).

4-quadrant dipole source positions for annular phased array (APA).
Single responsibility: source position and geometry.
"""


def build_quadrant_sources(
    nx, ny, nz, npml, dipole_half_len=9, ring_offset=None, z_plane=None
):
    """
    Build 4-quadrant dipole source positions for an annular phased array (APA).
    Quadrants are placed on +x, +y, -x, -y faces of the domain (inside PML).
    Each dipole is oriented along z and has a gap at center.

    Returns list of 4 dicts, each with 'gap' (i,j,k) and 'arms' voxels.
    """
    cx, cy = nx // 2, ny // 2
    cz = z_plane if z_plane is not None else nz // 2
    offset = ring_offset if ring_offset is not None else (npml + 2)
    quadrant_positions = [
        (cx, offset, cz),  # Q1: near y-low face
        (nx - offset - 1, cy, cz),  # Q2: near x-high face
        (cx, ny - offset - 1, cz),  # Q3: near y-high face
        (offset, cy, cz),  # Q4: near x-low face
    ]
    sources = []
    for qi, (gi, gj, gk) in enumerate(quadrant_positions):
        arm_voxels = []
        for dz in range(-dipole_half_len, dipole_half_len + 1):
            kk = gk + dz
            if 0 <= kk < nz and dz != 0:
                arm_voxels.append((gi, gj, kk))
        sources.append(
            {
                "quadrant": qi + 1,
                "gap": (gi, gj, gk),
                "arms": arm_voxels,
            }
        )
    return sources
