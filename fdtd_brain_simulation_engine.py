#!/usr/bin/env python3
"""Backward-compatible shim — use ``hermes-simulate`` or ``python -m hermes_drh.cli.main``."""

from hermes_drh.cli.main import main

if __name__ == "__main__":
    main()
