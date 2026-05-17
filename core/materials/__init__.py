"""
Material and tissue modeling: label-to-parameter mapping for FDTD and thermal.
"""

from core.materials.voxel_model import (
    K_TISSUE,
    TISSUE_TABLE,
    build_material_arrays,
)

__all__ = ["K_TISSUE", "TISSUE_TABLE", "build_material_arrays"]
