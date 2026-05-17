"""Package paths and bundled config resolution."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def configs_dir() -> Path:
    return package_dir() / "configs"


def bundled_config(name: str) -> Path:
    """Resolve a YAML file shipped under ``hermes_drh/configs/``."""
    ref = resources.files("hermes_drh").joinpath("configs", name)
    with resources.as_file(ref) as p:
        return Path(p)


def default_checkpoint_path() -> str | None:
    env = os.environ.get("HERMES_CHECKPOINT")
    if env:
        return os.path.abspath(env)
    local = package_dir().parent.parent.parent / "best_model.pth"
    if local.is_file():
        return str(local.resolve())
    return None
