"""
YAML simulation configuration for the FDTD brain engine CLI.

Keys in the YAML file must match argparse destination names (snake_case), e.g.
``max_dim``, ``time_steps``, ``pulse_type``, ``optimize_antenna``. Nested
mappings are flattened (inner keys merge into the top-level namespace).

Usage::

    python fdtd_brain_simulation_engine.py --config configs/simulation_example.yaml
    python fdtd_brain_simulation_engine.py --config sim.yaml --max-dim 99  # CLI overrides YAML
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any


def flatten_config_tree(data: Any) -> dict[str, Any]:
    """
    Flatten nested dicts by merging keys into one namespace.
    Later duplicate keys overwrite earlier ones. Keys starting with ``_`` are skipped.
    """
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"YAML root must be a mapping (dict), not {type(data).__name__}"
        )
    out: dict[str, Any] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            inner = flatten_config_tree(v)
            for ik, iv in inner.items():
                out[ik] = iv
        else:
            out[k] = v
    return out


def load_simulation_config(path: str | Path) -> dict[str, Any]:
    """Load and flatten a simulation YAML file."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for --config. Install with: pip install PyYAML"
        ) from e

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return flatten_config_tree(raw)


def argv_without_config_option(argv: list[str]) -> list[str]:
    """Remove ``--config PATH`` and ``--config=PATH`` from argv (keep other args)."""
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--config":
            i += 2
            continue
        if a.startswith("--config="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def validated_defaults_for_parser(
    parser: argparse.ArgumentParser, flat: dict[str, Any]
) -> dict[str, Any]:
    """
    Keep only keys that match a parser action ``dest``.
    Warn once per unknown key.
    """
    dests: set[str] = set()
    for action in parser._actions:
        d = getattr(action, "dest", None)
        if d and d != argparse.SUPPRESS:
            dests.add(d)

    out: dict[str, Any] = {}
    unknown: list[str] = []
    for k, v in flat.items():
        if k not in dests:
            unknown.append(k)
            continue
        out[k] = v

    for u in unknown:
        warnings.warn(
            f"Unknown simulation config key ignored: {u!r}",
            UserWarning,
            stacklevel=2,
        )
    return out
