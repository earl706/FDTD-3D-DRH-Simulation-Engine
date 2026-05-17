"""Smoke tests for installable hermes-drh package."""

from importlib import resources
from importlib.metadata import version

import pytest


def test_version():
    import hermes_drh

    assert hermes_drh.__version__ == version("hermes-drh")


def test_bundled_example_config_exists():
    ref = resources.files("hermes_drh").joinpath("configs", "simulation_example.yaml")
    assert ref.is_file()


def test_paper_scripts_not_shipped_in_package():
    import hermes_drh
    from pathlib import Path

    root = Path(hermes_drh.__file__).resolve().parent
    assert not (root / "generate_paper_results_section.py").exists()
    assert not (root / "run_paper_bundle.py").exists()


def test_cli_main_importable():
    from hermes_drh.cli.main import main

    assert callable(main)
