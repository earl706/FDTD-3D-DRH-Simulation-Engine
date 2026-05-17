"""
FDTD solver facade (implementation in core.fdtd).

Re-exports from hermes_drh.simulation.fdtd for backward compatibility.
See code_principles.md Section 5 architecture.
"""

from hermes_drh.simulation.fdtd import (
    accumulate_e_field_squared,
    calculate_dx_field,
    calculate_dy_field,
    calculate_dz_field,
    calculate_e_fields,
    calculate_fourier_transform_ex,
    calculate_hx_field,
    calculate_hx_inc,
    calculate_hx_with_incident_field,
    calculate_hx_with_incident_field_z,
    calculate_hy_field,
    calculate_hy_inc_x,
    calculate_hy_with_incident_field,
    calculate_hy_with_incident_field_x,
    calculate_inc_dy_field,
    calculate_inc_dz_field,
    calculate_inc_dz_field_x,
    calculate_inc_dz_field_z,
    calculate_hz_field,
    calculate_pml_parameters,
    update_ez_inc_x,
    update_ez_inc_z,
    calculate_hx_inc_z,
    _run_minimal_fdtd_benchmark,
    run_fdtd_optimized_loop,
    run_fdtd_standard_loop,
)

__all__ = [
    "accumulate_e_field_squared",
    "calculate_dx_field",
    "calculate_dy_field",
    "calculate_dz_field",
    "calculate_e_fields",
    "calculate_fourier_transform_ex",
    "calculate_hx_field",
    "calculate_hx_inc",
    "calculate_hx_with_incident_field",
    "calculate_hx_with_incident_field_z",
    "calculate_hy_field",
    "calculate_hy_inc_x",
    "calculate_hy_with_incident_field",
    "calculate_hy_with_incident_field_x",
    "calculate_inc_dy_field",
    "calculate_inc_dz_field",
    "calculate_inc_dz_field_x",
    "calculate_inc_dz_field_z",
    "calculate_hz_field",
    "calculate_pml_parameters",
    "update_ez_inc_x",
    "update_ez_inc_z",
    "calculate_hx_inc_z",
    "_run_minimal_fdtd_benchmark",
    "run_fdtd_optimized_loop",
    "run_fdtd_standard_loop",
]
