"""
Voxel-based tissue modeling facade (implementation in core.materials.voxel_model).

Re-exports from hermes_drh.simulation.materials.voxel_model for backward compatibility.
See code_principles.md Section 5 architecture.
"""

from hermes_drh.simulation.materials.voxel_model import (
    K_TISSUE,
    TISSUE_TABLE,
    build_material_arrays,
)

__all__ = ["K_TISSUE", "TISSUE_TABLE", "build_material_arrays"]
