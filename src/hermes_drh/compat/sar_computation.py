"""
SAR computation facade (implementation in core.metrics.sar).

Re-exports from hermes_drh.simulation.metrics.sar for backward compatibility.
See code_principles.md Section 5 architecture.
"""

from hermes_drh.simulation.metrics.sar import (
    compute_instantaneous_sar,
    compute_sar,
    compute_sar_from_complex_field,
    compute_j_ratio,
    compute_robust_objective,
)

__all__ = [
    "compute_instantaneous_sar",
    "compute_sar",
    "compute_sar_from_complex_field",
    "compute_j_ratio",
    "compute_robust_objective",
]
