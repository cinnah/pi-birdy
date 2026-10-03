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


def _with_site(extra: str) -> str:
    return MINIMAL.replace('name = "test"', 'name = "test"\n' + extra)


@pytest.mark.parametrize(
    "text, key",
    [
        (MINIMAL + "background_alpha = 0\n", "motion.background_alpha"),
        (MINIMAL + "background_alpha = 1.5\n", "motion.background_alpha"),
        (MINIMAL + "pixel_threshold = -1\n", "motion.pixel_threshold"),
        (MINIMAL + "min_area = -1\n", "motion.min_area"),
        (MINIMAL + "cooldown_seconds = -0.5\n", "motion.cooldown_seconds"),
        (MINIMAL + "\n[camera]\njpeg_quality = 0\n", "camera.jpeg_quality"),
        (MINIMAL + "\n[camera]\njpeg_quality = 101\n", "camera.jpeg_quality"),
        (MINIMAL + "\n[retention]\nnon_bird_days = 0\n", "retention.non_bird_days"),
        (MINIMAL + "\n[daylight]\nmargin_minutes = -1\n", "daylight.margin_minutes"),
        (_with_site("lat = 91\nlon = 0"), "site.lat"),
        (_with_site("lat = 0\nlon = -181"), "site.lon"),
        (MINIMAL + "\n[web]\nport = 0\n", "web.port"),
        (MINIMAL + "\n[web]\nport = 65536\n", "web.port"),
        (MINIMAL.replace("[10, 10]", "[320, 10]"), "motion.roi point .* outside camera.lores_size"),
        (MINIMAL.replace("[0, 10]]", "[0, 240]]"), "motion.roi point .* outside camera.lores_size"),
        (MINIMAL.replace("[0, 0], [10, 0]", "[-1, 0], [10, 0]"), "motion.roi point .* outside camera.lores_size"),
        (MINIMAL + "\n[camera]\nlores_size = [8, 8]\n", "motion.roi point .* outside camera.lores_size"),
    ],
)
def test_range_checks(tmp_path, text, key):
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize(
    "text, key",
    [
        (MINIMAL + '\n[classifier]\nmodel = 5\n', "classifier.model"),
        (MINIMAL + '\n[web]\nport = "80a"\n', "web.port"),
        (MINIMAL + '\n[camera]\njpeg_quality = "high"\n', "camera.jpeg_quality"),
    ],
)
def test_wrong_types_raise_config_error(tmp_path, text, key):
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, text))


def test_absolute_data_dir_kept(tmp_path):
    abs_dir = tmp_path / "elsewhere" / "data"
    cfg = load_config(write(tmp_path, MINIMAL + f'\n[storage]\ndata_dir = "{abs_dir}"\n'))
    assert cfg.storage.data_dir == abs_dir


def test_example_config_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "birdy.toml.example")
    assert cfg.site.name == "home"
    assert len(cfg.motion.roi) == 4


@pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
def test_float_must_be_finite(tmp_path, value):
    p = write(tmp_path, MINIMAL.replace("[motion]", f"[motion]\ncooldown_seconds = {value}"))
    with pytest.raises(ConfigError, match="motion.cooldown_seconds must be a finite number"):
        load_config(p)


def test_roi_rejects_booleans(tmp_path):
    p = write(tmp_path, '[site]\nname = "t"\n[motion]\nroi = [[true, 0], [10, 0], [10, 10]]\n')
    with pytest.raises(ConfigError, match="motion.roi"):
        load_config(p)


def test_size_rejects_booleans(tmp_path):
    p = write(tmp_path, MINIMAL + "\n[camera]\nlores_size = [true, 240]\n")
    with pytest.raises(ConfigError, match="camera.lores_size"):
        load_config(p)


def test_species_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.species.model == tmp_path / "models" / "mobilenet_v2_1.0_224_inat_bird_quant.tflite"
    assert cfg.species.labels == tmp_path / "models" / "inat_bird_labels.txt"
    assert cfg.species.threshold == 0.5 and cfg.species.rescue_threshold == 0.7


def test_species_rescue_must_not_be_below_threshold(tmp_path):
    with pytest.raises(ConfigError, match="species.rescue_threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[species]\nthreshold = 0.8\nrescue_threshold = 0.6\n"))


@pytest.mark.parametrize("key", ["threshold", "rescue_threshold"])
def test_species_thresholds_range(tmp_path, key):
    with pytest.raises(ConfigError, match=f"species.{key}"):
        load_config(write(tmp_path, MINIMAL + f"\n[species]\n{key} = 1.5\n"))
