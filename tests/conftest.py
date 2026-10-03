"""Shared fixtures: a minimal valid config on disk, loaded, and wrapped in LiveConfig."""

from pathlib import Path

import pytest

from birdy.config import load_config
from birdy.live import LiveConfig

MINIMAL_TOML = """
[site]
name = "test"

[motion]
roi = [[40, 60], [280, 60], [280, 220], [40, 220]]

[storage]
data_dir = "data"
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "birdy.toml"
    p.write_text(MINIMAL_TOML)
    return p


@pytest.fixture
def config(config_path):
    return load_config(config_path)


@pytest.fixture
def live(config):
    return LiveConfig(config)
