"""
Practice drill file for thesis tests.

- `parse_args` and benchmark defaults match `cli.py` (see tests/test_practice_vs_cli.py).
- FDTD kernel callables are re-exported from `core.fdtd.kernels` for parity drills
  (see tests/test_practice_vs_fdtd_kernels.py).
"""

from cli import (
    BENCHMARK_GRID_SIZES_DEFAULT,
    BENCHMARK_TIME_STEPS_DEFAULT,
    parse_args,
)
from core.fdtd.kernels import *  # noqa: F403
