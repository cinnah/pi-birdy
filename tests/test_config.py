from pathlib import Path

import pytest

from birdy.config import ConfigError, load_config

MINIMAL = """
[site]
name = "test"

[motion]
roi = [[0, 0], [10, 0], [10, 10], [0, 10]]
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "birdy.toml"
    p.write_text(text)
    return p


def test_defaults_fill_in(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.site.name == "test"
    assert cfg.site.lat is None and cfg.site.lon is None
    assert cfg.camera.main_size == (3280, 2464)
    assert cfg.camera.lores_size == (320, 240)
    assert cfg.motion.min_area == 150
    assert cfg.motion.cooldown_seconds == 3.0
    assert cfg.daylight.margin_minutes == 30
    assert cfg.classifier.threshold == 0.5
    assert cfg.retention.non_bird_days == 7
    assert cfg.web.port == 8000


def test_roi_is_tuple_of_points(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.motion.roi == ((0, 0), (10, 0), (10, 10), (0, 10))


def test_relative_paths_resolve_against_config_dir(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[storage]\ndata_dir = "data"\n'))
    assert cfg.storage.data_dir == tmp_path / "data"
    assert cfg.classifier.model == tmp_path / "models" / "mobilenet_v2_1.0_224_quant.tflite"


def test_lat_lon_parsed(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL.replace('name = "test"', 'name = "test"\nlat = 38.7\nlon = -9.1')))
    assert cfg.site.lat == 38.7 and cfg.site.lon == -9.1


def test_roi_needs_three_points(tmp_path):
    with pytest.raises(ConfigError, match="roi"):
        load_config(write(tmp_path, MINIMAL.replace("[[0, 0], [10, 0], [10, 10], [0, 10]]", "[[0, 0], [1, 1]]")))


def test_threshold_range(tmp_path):
    with pytest.raises(ConfigError, match="threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[classifier]\nthreshold = 1.5\n"))


def test_only_one_of_lat_lon_is_error(tmp_path):
    with pytest.raises(ConfigError, match="lat"):
        load_config(write(tmp_path, MINIMAL.replace('name = "test"', 'name = "test"\nlat = 38.7')))
