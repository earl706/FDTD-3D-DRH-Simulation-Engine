"""
Derived physics metrics: SAR (instantaneous, RMS, phasor) and thermal PDE.
"""

from hermes_drh.simulation.metrics.sar import (
    compute_instantaneous_sar,
    compute_sar,
    compute_sar_from_complex_field,
    compute_j_ratio,
    compute_robust_objective,
)
from hermes_drh.simulation.metrics.thermal import solve_steady_bioheat_3d

__all__ = [
    "compute_instantaneous_sar",
    "compute_sar",
    "compute_sar_from_complex_field",
    "compute_j_ratio",
    "compute_robust_objective",
    "solve_steady_bioheat_3d",
]
