import dataclasses
from pathlib import Path

import pytest

from birdy.config import ConfigError, load_config
from birdy.settings import HEADER, SettingsPatch, config_to_dict, parse_settings_form, save_settings


def default_patch(config, **overrides) -> SettingsPatch:
    base = dict(
        roi=config.motion.roi,
        pixel_threshold=config.motion.pixel_threshold,
        min_area=config.motion.min_area,
        background_alpha=config.motion.background_alpha,
        cooldown_seconds=config.motion.cooldown_seconds,
        classifier_threshold=config.classifier.threshold,
        lat=config.site.lat,
        lon=config.site.lon,
        margin_minutes=config.daylight.margin_minutes,
        non_bird_days=config.retention.non_bird_days,
    )
    base.update(overrides)
    return SettingsPatch(**base)


def test_config_to_dict_roundtrips_through_load_config(config, tmp_path):
    import tomli_w

    d = config_to_dict(config)
    p = tmp_path / "rt.toml"
    p.write_text(tomli_w.dumps(d))
    assert load_config(p) == config


def test_config_to_dict_omits_lat_lon_when_unset(config):
    d = config_to_dict(config)
    assert "lat" not in d["site"] and "lon" not in d["site"]


def test_save_writes_patch_and_returns_new_config(config_path, config):
    new = save_settings(config_path, config, default_patch(config, min_area=777, lat=38.7, lon=-9.1, roi=((0, 0), (50, 0), (50, 50))))
    assert new.motion.min_area == 777
    assert new.site.lat == 38.7 and new.site.lon == -9.1
    assert new.motion.roi == ((0, 0), (50, 0), (50, 50))
    assert load_config(config_path) == new
    assert config_path.read_text().startswith(HEADER)
    assert not config_path.with_suffix(".toml.tmp").exists()


def test_save_preserves_non_editable_keys(config_path, config):
    config_path.write_text(config_path.read_text() + '\n[web]\nport = 8123\n[camera]\njpeg_quality = 77\n')
    cfg = load_config(config_path)
    new = save_settings(config_path, cfg, default_patch(cfg, min_area=1))
    assert new.web.port == 8123 and new.camera.jpeg_quality == 77


def test_save_rejects_invalid_patch_and_leaves_file_untouched(config_path, config):
    before = config_path.read_bytes()
    with pytest.raises(ConfigError, match="motion.roi"):
        save_settings(config_path, config, default_patch(config, roi=((0, 0), (1, 1))))
    assert config_path.read_bytes() == before
    assert not config_path.with_suffix(".toml.tmp").exists()


def test_save_rejects_lat_without_lon(config_path, config):
    with pytest.raises(ConfigError, match="site.lat"):
        save_settings(config_path, config, default_patch(config, lat=38.7, lon=None))


def test_save_clears_lat_lon(config_path, config):
    cfg = save_settings(config_path, config, default_patch(config, lat=38.7, lon=-9.1))
    cfg2 = save_settings(config_path, cfg, default_patch(cfg, lat=None, lon=None))
    assert cfg2.site.lat is None and cfg2.site.lon is None


def test_save_writes_absolute_paths(config_path, config):
    save_settings(config_path, config, default_patch(config))
    text = config_path.read_text()
    assert str(config.classifier.model) in text
    assert str(config.storage.data_dir) in text


FORM = {
    "roi": "[[40, 60], [280, 60], [280, 220], [40, 220]]",
    "pixel_threshold": "25",
    "min_area": "150",
    "background_alpha": "0.05",
    "cooldown_seconds": "3",
    "classifier_threshold": "0.5",
    "lat": "",
    "lon": "",
    "margin_minutes": "30",
    "non_bird_days": "7",
}


def test_parse_form_ok():
    p = parse_settings_form(FORM)
    assert p.roi == ((40, 60), (280, 60), (280, 220), (40, 220))
    assert p.pixel_threshold == 25 and p.min_area == 150
    assert p.background_alpha == 0.05 and p.cooldown_seconds == 3.0
    assert p.classifier_threshold == 0.5
    assert p.lat is None and p.lon is None
    assert p.margin_minutes == 30 and p.non_bird_days == 7


def test_parse_form_lat_lon():
    p = parse_settings_form({**FORM, "lat": "38.72", "lon": "-9.14"})
    assert p.lat == 38.72 and p.lon == -9.14


@pytest.mark.parametrize(
    "field, value, key",
    [
        ("min_area", "abc", "motion.min_area"),
        ("background_alpha", "", "motion.background_alpha"),
        ("roi", "not json", "motion.roi"),
        ("roi", "[[1, 2], [3]]", "motion.roi"),
        ("roi", '[["a", 1], [2, 3], [4, 5]]', "motion.roi"),
        ("lat", "north", "site.lat"),
        ("non_bird_days", "7.5", "retention.non_bird_days"),
    ],
)
def test_parse_form_errors_name_the_config_key(field, value, key):
    with pytest.raises(ConfigError, match=key):
        parse_settings_form({**FORM, field: value})


def test_parse_form_missing_field():
    form = dict(FORM)
    del form["min_area"]
    with pytest.raises(ConfigError, match="motion.min_area"):
        parse_settings_form(form)
