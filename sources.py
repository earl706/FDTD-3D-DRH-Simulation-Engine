"""
Source geometry facade (implementation in core.fdtd.sources).

Re-exports from core.fdtd.sources for backward compatibility.
See code_principles.md Section 5 architecture.
"""

from core.fdtd.sources import build_quadrant_sources

__all__ = ["build_quadrant_sources"]
