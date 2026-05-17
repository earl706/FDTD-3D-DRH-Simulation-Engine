"""
Thermal solver facade (implementation in core.metrics.thermal).

Re-exports from hermes_drh.simulation.metrics.thermal for backward compatibility.
See code_principles.md Section 5 architecture.
"""

from hermes_drh.simulation.metrics.thermal import solve_steady_bioheat_3d

__all__ = ["solve_steady_bioheat_3d"]
