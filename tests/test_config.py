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
        (MINIMAL + "\n[camera]\nmain_size = [8, 8]\nlores_size = [8, 8]\n", "motion.roi point .* outside camera.lores_size"),
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
    assert cfg.species.model == tmp_path / "models" / "bioclip2-vision-int8.onnx"
    assert cfg.species.bank == tmp_path / "models" / "bioclip2-birds.npz"
    assert cfg.species.labels == tmp_path / "models" / "inat_bird_labels.txt"   # removed in Task 7
    assert cfg.species.threshold == 0.5
    assert cfg.species.region_radius_km == 50
    assert cfg.species.include == () and cfg.species.exclude == ()
    assert not hasattr(cfg.species, "rescue_threshold")


def test_species_threshold_range(tmp_path):
    with pytest.raises(ConfigError, match="species.threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[species]\nthreshold = 1.5\n"))


def test_species_unknown_key_ignored(tmp_path):
    old = (
        '\n[species]\nmodel = "models/old.tflite"\nlabels = "models/old.txt"\n'
        "threshold = 0.6\nrescue_threshold = 1.0\n"
    )
    cfg = load_config(write(tmp_path, MINIMAL + old))
    assert cfg.species.threshold == 0.6
    assert cfg.species.region_radius_km == 50
    assert cfg.species.include == ()


@pytest.mark.parametrize("text", ["region_radius_km = 0", "region_radius_km = 501", 'region_radius_km = "x"'])
def test_species_radius_range(tmp_path, text):
    with pytest.raises(ConfigError, match="species.region_radius_km"):
        load_config(write(tmp_path, MINIMAL + f"\n[species]\n{text}\n"))


def test_species_include_exclude_lists(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[species]\ninclude = [" Hoopoe ", ""]\nexclude = ["Rock Pigeon"]\n'))
    assert cfg.species.include == ("Hoopoe",) and cfg.species.exclude == ("Rock Pigeon",)
    with pytest.raises(ConfigError, match="species.include"):
        load_config(write(tmp_path, MINIMAL + "\n[species]\ninclude = [1]\n"))


def test_telegram_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    t = cfg.telegram
    assert t.enabled is False and t.bot_token == "" and t.chat_id == ""
    assert t.base_url == "http://birdy.local:8000"
    assert t.watchlist == () and t.alert_new_species is True and t.cooldown_minutes == 30


def test_telegram_watchlist_trimmed_and_empty_dropped(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[telegram]\nwatchlist = [" Hoopoe ", "", "Blue Tit"]\n'))
    assert cfg.telegram.watchlist == ("Hoopoe", "Blue Tit")


def test_telegram_watchlist_must_be_strings(tmp_path):
    with pytest.raises(ConfigError, match="telegram.watchlist"):
        load_config(write(tmp_path, MINIMAL + "\n[telegram]\nwatchlist = [1, 2]\n"))


@pytest.mark.parametrize("missing", ["bot_token", "chat_id"])
def test_telegram_enabled_requires_token_and_chat(tmp_path, missing):
    other = "chat_id" if missing == "bot_token" else "bot_token"
    with pytest.raises(ConfigError, match=f"telegram.{missing}"):
        load_config(write(tmp_path, MINIMAL + f'\n[telegram]\nenabled = true\n{other} = "x"\n'))


def test_telegram_cooldown_non_negative(tmp_path):
    with pytest.raises(ConfigError, match="telegram.cooldown_minutes"):
        load_config(write(tmp_path, MINIMAL + "\n[telegram]\ncooldown_minutes = -1\n"))


def test_telegram_enabled_must_be_bool(tmp_path):
    with pytest.raises(ConfigError, match="telegram.enabled"):
        load_config(write(tmp_path, MINIMAL + '\n[telegram]\nenabled = "yes"\n'))


def test_telegram_token_hidden_from_repr(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[telegram]\nbot_token = "SECRET"\n'))
    assert cfg.telegram.bot_token == "SECRET"
    assert "SECRET" not in repr(cfg)


def test_telegram_chat_id_accepts_int(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + "\n[telegram]\nchat_id = 1115872433\n"))
    assert cfg.telegram.chat_id == "1115872433"


def test_telegram_base_url_trailing_slash_stripped(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[telegram]\nbase_url = "http://example.test/"\n'))
    assert cfg.telegram.base_url == "http://example.test"



def test_publish_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    pub = cfg.publish
    assert pub.enabled is False
    assert pub.host == ""
    assert pub.user == "birdy"
    assert pub.remote_dir == "/"
    assert pub.site_id == "home"
    assert pub.ssh_key == Path("~/.ssh/birdy_publish").expanduser()
    assert pub.min_confidence == 0.6
    assert pub.per_species_per_day == 3


def test_publish_enabled_requires_host(tmp_path):
    with pytest.raises(ConfigError, match="publish.host is required"):
        load_config(write(tmp_path, MINIMAL + "\n[publish]\nenabled = true\n"))


@pytest.mark.parametrize("line, msg", [
    ('user = ""', "publish.user is required"),
    ('remote_dir = ""', "publish.remote_dir must be an absolute path"),
    ('remote_dir = "rel"', "publish.remote_dir must be an absolute path"),
    ('host = "-x"', "publish.host must be a plain hostname or IP"),
    ('host = "a b"', "publish.host must be a plain hostname or IP"),
    ('user = "a@b"', "publish.user must be a plain user name"),
])
def test_publish_enabled_rejects_unsafe_target(tmp_path, line, msg):
    host = "" if line.startswith("host") else 'host = "gallery.example.org"\n'
    with pytest.raises(ConfigError, match=msg):
        load_config(write(tmp_path, MINIMAL + f"\n[publish]\nenabled = true\n{host}{line}\n"))


def test_publish_target_is_stripped(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[publish]\nenabled = true\nhost = " gallery.example.org "\n'
                                                'user = " birdy "\nremote_dir = " / "\n'))
    assert (cfg.publish.host, cfg.publish.user, cfg.publish.remote_dir) == ("gallery.example.org", "birdy", "/")


@pytest.mark.parametrize("site_id", ["", "Home", "a b", "x" * 33, "home/../x"])
def test_publish_site_id_pattern(tmp_path, site_id):
    with pytest.raises(ConfigError, match="publish.site_id"):
        load_config(write(tmp_path, MINIMAL + f'\n[publish]\nsite_id = "{site_id}"\n'))


@pytest.mark.parametrize("line, key", [
    ("min_confidence = 1.5", "publish.min_confidence"),
    ("min_confidence = -0.1", "publish.min_confidence"),
    ("per_species_per_day = 0", "publish.per_species_per_day"),
    ("per_species_per_day = 21", "publish.per_species_per_day"),
    ("per_species_per_day = 2.5", "publish.per_species_per_day"),
    ("enabled = 1", "publish.enabled"),
])
def test_publish_ranges(tmp_path, line, key):
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, MINIMAL + f"\n[publish]\n{line}\n"))


def test_publish_whole_float_count_accepted(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + "\n[publish]\nper_species_per_day = 3.0\n"))
    assert cfg.publish.per_species_per_day == 3 and isinstance(cfg.publish.per_species_per_day, int)


def test_int_key_rejects_fractional_float(tmp_path):
    with pytest.raises(ConfigError, match="web.port must be an integer"):
        load_config(write(tmp_path, MINIMAL + "\n[web]\nport = 80.5\n"))


def test_publish_full_section(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[publish]\nenabled = true\nhost = "gallery.example.org"\n'
                                                'user = "birdy"\nremote_dir = "/"\nsite_id = "sintra-1"\n'
                                                'ssh_key = "keys/pub"\nmin_confidence = 0.75\nper_species_per_day = 5\n'))
    assert cfg.publish.enabled and cfg.publish.host == "gallery.example.org"
    assert cfg.publish.site_id == "sintra-1"
    assert cfg.publish.ssh_key == tmp_path.resolve() / "keys/pub"   # relative to the config file, like _path
    assert cfg.publish.min_confidence == 0.75 and cfg.publish.per_species_per_day == 5


def test_camera_streams_must_share_aspect_ratio(tmp_path):
    bad = MINIMAL + "\n[camera]\nmain_size = [1920, 1080]\nlores_size = [320, 240]\n"
    with pytest.raises(ConfigError, match="aspect ratio"):
        load_config(write(tmp_path, bad))


def test_default_camera_streams_share_aspect_ratio(tmp_path):
    ok = MINIMAL + "\n[camera]\nmain_size = [1640, 1232]\nlores_size = [320, 240]\n"
    assert load_config(write(tmp_path, ok)).camera.main_size == (1640, 1232)


def test_camera_size_must_be_positive(tmp_path):
    bad = MINIMAL + "\n[camera]\nmain_size = [0, 2464]\n"
    with pytest.raises(ConfigError, match="must be positive"):
        load_config(write(tmp_path, bad))


def test_detector_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.detector.model == tmp_path / "models" / "efficientdet_lite0_320_ptq.tflite"
    assert cfg.detector.labels == tmp_path / "models" / "coco_labels.txt"
    assert cfg.detector.threshold == 0.2


def test_detector_threshold_range(tmp_path):
    with pytest.raises(ConfigError, match="detector.threshold"):
        load_config(write(tmp_path, MINIMAL + "\n[detector]\nthreshold = 1.5\n"))
