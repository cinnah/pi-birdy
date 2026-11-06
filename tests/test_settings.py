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
        species_threshold=config.species.threshold,
        species_include=config.species.include,
        species_exclude=config.species.exclude,
        detector_threshold=config.detector.threshold,
        lat=config.site.lat,
        lon=config.site.lon,
        margin_minutes=config.daylight.margin_minutes,
        non_bird_days=config.retention.non_bird_days,
        watchlist=config.telegram.watchlist,
        alert_new_species=config.telegram.alert_new_species,
        cooldown_minutes=config.telegram.cooldown_minutes,
        publish_min_confidence=config.publish.min_confidence,
        per_species_per_day=config.publish.per_species_per_day,
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
    "species_threshold": "0.5",
    "species_include": "",
    "species_exclude": "",
    "detector_threshold": "0.2",
    "lat": "",
    "lon": "",
    "margin_minutes": "30",
    "non_bird_days": "7",
    "watchlist": "",
    "alert_new_species": "on",
    "cooldown_minutes": "30",
    "publish_min_confidence": "0.6",
    "per_species_per_day": "3",
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


FULL_TOML = """
[site]
name = "full"
lat = 38.72
lon = -9.14

[camera]
main_size = [1920, 1440]
lores_size = [640, 480]
jpeg_quality = 80

[motion]
roi = [[10, 20], [600, 20], [600, 400]]
pixel_threshold = 31
min_area = 222
background_alpha = 0.2
cooldown_seconds = 7.5

[daylight]
margin_minutes = 45

[classifier]
model = "m/model.tflite"
labels = "m/labels.txt"
bird_classes = "m/birds.txt"
threshold = 0.65

[species]
model = "models/other_species.tflite"
threshold = 0.4
bank = "models/other-birds.npz"
region_radius_km = 80
include = ["Hoopoe"]
exclude = ["Rock Pigeon"]

[detector]
model = "models/other_detector.tflite"
labels = "models/other_coco_labels.txt"
threshold = 0.35

[retention]
non_bird_days = 14

[storage]
data_dir = "store"

[web]
host = "127.0.0.1"
port = 9001

[telegram]
enabled = true
bot_token = "123:ABC"
chat_id = "42"
base_url = "http://example.test"
watchlist = ["Hoopoe"]
alert_new_species = false
cooldown_minutes = 5

[publish]
enabled = true
host = "gallery.example"
user = "pub"
remote_dir = "/srv/gallery"
site_id = "site-2"
ssh_key = "keys/publish"
min_confidence = 0.8
per_species_per_day = 7
"""


def test_full_config_roundtrip(tmp_path):
    from birdy.config import Config

    path = tmp_path / "birdy.toml"
    path.write_text(FULL_TOML)
    cfg = load_config(path)
    # sanity: every section really differs from the defaults
    defaults = load_config(_write_minimal(tmp_path / "min.toml"))
    for f in dataclasses.fields(Config):
        assert getattr(cfg, f.name) != getattr(defaults, f.name), f.name
    save_settings(path, cfg, default_patch(cfg))
    assert load_config(path) == cfg


def _write_minimal(p: Path) -> Path:
    p.write_text('[site]\nname = "full"\n[motion]\nroi = [[0, 0], [1, 0], [1, 1]]\n')
    return p


def test_config_to_dict_covers_every_field(tmp_path):
    from birdy.config import Config

    path = tmp_path / "birdy.toml"
    path.write_text(FULL_TOML)
    for cfg in (load_config(path), load_config(_write_minimal(tmp_path / "min.toml"))):
        d = config_to_dict(cfg)
        for section in dataclasses.fields(Config):
            sub = getattr(cfg, section.name)
            for f in dataclasses.fields(sub):
                if getattr(sub, f.name) is None:
                    assert f.name not in d[section.name]
                else:
                    assert f.name in d[section.name], f"{section.name}.{f.name}"


def test_save_cleans_tmp_on_oserror(config_path, config, monkeypatch):
    import birdy.settings

    def boom(path):
        raise OSError("disk on fire")

    monkeypatch.setattr(birdy.settings, "load_config", boom)
    before = config_path.read_bytes()
    with pytest.raises(OSError):
        save_settings(config_path, config, default_patch(config, min_area=1))
    assert config_path.read_bytes() == before
    assert not config_path.with_suffix(".toml.tmp").exists()


def test_save_follows_symlink(tmp_path, config):
    real = tmp_path / "real" / "birdy.toml"
    real.parent.mkdir()
    real.write_text("placeholder")
    link = tmp_path / "birdy-link.toml"
    link.symlink_to(real)
    save_settings(link, config, default_patch(config, min_area=5))
    assert link.is_symlink()
    assert load_config(real).motion.min_area == 5


def test_save_preserves_file_mode(config_path, config):
    config_path.chmod(0o640)
    save_settings(config_path, config, default_patch(config, min_area=5))
    assert config_path.stat().st_mode & 0o777 == 0o640


def test_parse_form_strips_whitespace():
    p = parse_settings_form({**FORM, "non_bird_days": " 7 ", "background_alpha": " 0.1 "})
    assert p.non_bird_days == 7 and p.background_alpha == 0.1


def test_parse_form_error_wording():
    with pytest.raises(ConfigError, match="motion.min_area must be an integer"):
        parse_settings_form({**FORM, "min_area": "abc"})
    with pytest.raises(ConfigError, match="site.lat must be a number"):
        parse_settings_form({**FORM, "lat": "north"})


def test_parse_form_rejects_inf(config_path, config):
    patch = parse_settings_form({**FORM, "cooldown_seconds": "inf"})
    before = config_path.read_bytes()
    with pytest.raises(ConfigError, match="motion.cooldown_seconds"):
        save_settings(config_path, config, patch)
    assert config_path.read_bytes() == before


def test_species_thresholds_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, species_threshold=0.3))
    assert new.species.threshold == 0.3


def test_parse_form_species_fields():
    p = parse_settings_form({**FORM, "species_threshold": "0.35"})
    assert p.species_threshold == 0.35


def test_telegram_fields_roundtrip_and_token_preserved(config_path, config):
    config_path.write_text(config_path.read_text() + '\n[telegram]\nbot_token = "SECRET"\nchat_id = "7"\n')
    cfg = load_config(config_path)
    new = save_settings(config_path, cfg, default_patch(cfg, watchlist=("Hoopoe", "Blue Tit"), alert_new_species=False, cooldown_minutes=10))
    assert new.telegram.watchlist == ("Hoopoe", "Blue Tit")
    assert new.telegram.alert_new_species is False and new.telegram.cooldown_minutes == 10
    assert new.telegram.bot_token == "SECRET" and new.telegram.chat_id == "7"  # restart-only keys survive a save


def test_parse_form_watchlist_textarea_and_checkbox():
    p = parse_settings_form({**FORM, "watchlist": "Hoopoe\n\n  Blue Tit  \n", "alert_new_species": "on"})
    assert p.watchlist == ("Hoopoe", "Blue Tit") and p.alert_new_species is True
    form = dict(FORM)
    del form["alert_new_species"]  # unchecked checkboxes are absent from the POST
    assert parse_settings_form(form).alert_new_species is False


def test_parse_form_cooldown_minutes_int():
    with pytest.raises(ConfigError, match="telegram.cooldown_minutes"):
        parse_settings_form({**FORM, "cooldown_minutes": "x"})


def test_save_creates_private_file(config_path, config):
    config_path.unlink()
    save_settings(config_path, config, default_patch(config))
    assert (config_path.stat().st_mode & 0o777) == 0o600


def test_parse_form_watchlist_crlf():
    p = parse_settings_form({**FORM, "watchlist": "Hoopoe\r\n\r\nBlue Tit\r\n"})
    assert p.watchlist == ("Hoopoe", "Blue Tit")


def test_parse_form_publish_fields():
    patch = parse_settings_form({**FORM, "publish_min_confidence": "0.75", "per_species_per_day": "5"})
    assert patch.publish_min_confidence == 0.75 and patch.per_species_per_day == 5


def test_parse_form_publish_requires_integer():
    with pytest.raises(ConfigError, match="publish.per_species_per_day must be an integer"):
        parse_settings_form({**FORM, "per_species_per_day": "2.5"})


def test_save_applies_publish_fields_and_keeps_restart_only(config_path):
    config_path.write_text(config_path.read_text() + '\n[publish]\nenabled = false\nhost = "h"\nsite_id = "abc"\n')
    cfg = load_config(config_path)
    patch = parse_settings_form({**FORM, "publish_min_confidence": "0.8", "per_species_per_day": "4"})
    new = save_settings(config_path, cfg, patch)
    assert new.publish.min_confidence == 0.8 and new.publish.per_species_per_day == 4
    assert new.publish.host == "h" and new.publish.site_id == "abc"


def test_detector_threshold_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, detector_threshold=0.35))
    assert new.detector.threshold == 0.35
    assert load_config(config_path).detector.threshold == 0.35


def test_parse_form_detector_threshold():
    assert parse_settings_form(FORM).detector_threshold == 0.2


def test_species_include_exclude_roundtrip(config_path, config):
    new = save_settings(config_path, config, default_patch(config, species_include=("Hoopoe", "Blue Tit"), species_exclude=("Rock Pigeon",)))
    assert new.species.include == ("Hoopoe", "Blue Tit") and new.species.exclude == ("Rock Pigeon",)
    assert load_config(config_path).species.exclude == ("Rock Pigeon",)


def test_parse_form_species_lists():
    p = parse_settings_form({**FORM, "species_include": "Hoopoe\r\n\r\n Blue Tit \n", "species_exclude": "Rock Pigeon"})
    assert p.species_include == ("Hoopoe", "Blue Tit") and p.species_exclude == ("Rock Pigeon",)


def test_save_drops_legacy_rescue_key_and_writes_new_species_keys(tmp_path):
    path = tmp_path / "birdy.toml"
    path.write_text(
        '[site]\nname = "test"\n\n[motion]\nroi = [[40, 60], [280, 60], [280, 220], [40, 220]]\n'
        "\n[species]\nthreshold = 0.5\nrescue_threshold = 1.0\n"
    )
    cfg = load_config(path)
    save_settings(path, cfg, default_patch(cfg))
    text = path.read_text()
    assert "rescue_threshold" not in text
    assert "bank" in text
    assert load_config(path).species.region_radius_km == 50
