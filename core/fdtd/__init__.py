"""
FDTD subpackage: kernels, boundaries, sources, and time-loop control.
"""

from core.fdtd.kernels import (
    calculate_pml_parameters,
    calculate_dx_field,
    calculate_dy_field,
    calculate_dz_field,
    calculate_e_fields,
    calculate_fourier_transform_ex,
    calculate_hx_field,
    calculate_hy_field,
    calculate_hz_field,
    accumulate_e_field_squared,
)
from core.fdtd.boundaries import (
    calculate_inc_dy_field,
    calculate_inc_dz_field,
    calculate_hx_inc,
    calculate_hx_with_incident_field,
    calculate_hy_with_incident_field,
    update_ez_inc_x,
    calculate_hy_inc_x,
    calculate_inc_dz_field_x,
    calculate_hy_with_incident_field_x,
    update_ez_inc_z,
    calculate_hx_inc_z,
    calculate_inc_dz_field_z,
    calculate_hx_with_incident_field_z,
)
from core.fdtd.sources import build_quadrant_sources
from core.fdtd.loops import (
    _run_minimal_fdtd_benchmark,
    run_fdtd_optimized_loop,
    run_fdtd_standard_loop,
)

__all__ = [
    "calculate_pml_parameters",
    "calculate_dx_field",
    "calculate_dy_field",
    "calculate_dz_field",
    "calculate_e_fields",
    "calculate_fourier_transform_ex",
    "calculate_hx_field",
    "calculate_hy_field",
    "calculate_hz_field",
    "accumulate_e_field_squared",
    "calculate_inc_dy_field",
    "calculate_inc_dz_field",
    "calculate_hx_inc",
    "calculate_hx_with_incident_field",
    "calculate_hy_with_incident_field",
    "update_ez_inc_x",
    "calculate_hy_inc_x",
    "calculate_inc_dz_field_x",
    "calculate_hy_with_incident_field_x",
    "update_ez_inc_z",
    "calculate_hx_inc_z",
    "calculate_inc_dz_field_z",
    "calculate_hx_with_incident_field_z",
    "build_quadrant_sources",
    "_run_minimal_fdtd_benchmark",
    "run_fdtd_optimized_loop",
    "run_fdtd_standard_loop",
]
