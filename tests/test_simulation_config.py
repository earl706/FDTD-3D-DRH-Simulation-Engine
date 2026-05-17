"""Tests for YAML simulation config (--config) integration."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import cli


def test_parse_args_config_overrides_defaults(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            max_dim: 77
            time_steps: 888
            pulse_type: cw
            """
        ),
        encoding="utf-8",
    )
    args = cli.parse_args(["--config", str(cfg)])
    assert args.max_dim == 77
    assert args.time_steps == 888
    assert args.pulse_type == "cw"


def test_parse_args_cli_overrides_yaml(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text("max_dim: 50\n", encoding="utf-8")
    args = cli.parse_args(["--config", str(cfg), "--max-dim", "99"])
    assert args.max_dim == 99


def test_parse_args_nested_yaml_flattened(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            grid:
              max_dim: 60
              dx_mm: 5.0
            """
        ),
        encoding="utf-8",
    )
    args = cli.parse_args(["--config", str(cfg)])
    assert args.max_dim == 60
    assert args.dx_mm == 5.0


def test_flatten_duplicate_inner_wins(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            max_dim: 40
            nested:
              max_dim: 55
            """
        ),
        encoding="utf-8",
    )
    args = cli.parse_args(["--config", str(cfg)])
    assert args.max_dim == 55


def test_unknown_config_key_warns(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text("not_a_real_cli_key: 1\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="Unknown simulation config key"):
        cli.parse_args(["--config", str(cfg)])


def test_config_equals_form(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text("max_dim: 42\n", encoding="utf-8")
    args = cli.parse_args([f"--config={cfg}"])
    assert args.max_dim == 42


def test_modalities_list_in_yaml(tmp_path: Path):
    cfg = tmp_path / "sim.yaml"
    cfg.write_text("modalities: [a.nii, b.nii, c.nii, d.nii]\n", encoding="utf-8")
    args = cli.parse_args(["--config", str(cfg)])
    assert args.modalities == ["a.nii", "b.nii", "c.nii", "d.nii"]
