import logging
import os
import time

from birdy.watcher import ConfigWatcher


def bump_mtime(path):
    # Guarantee a different mtime even on coarse filesystems.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def test_no_change_no_reload(config_path, live):
    w = ConfigWatcher(config_path, live)
    assert w.check_once() is False
    assert live.version == 0


def test_change_reloads(config_path, live):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text().replace("[motion]", "[motion]\nmin_area = 321"))
    bump_mtime(config_path)
    assert w.check_once() is True
    assert live.version == 1
    assert live.get().motion.min_area == 321


def test_same_mtime_different_size_reloads(config_path, live):
    w = ConfigWatcher(config_path, live)
    st = config_path.stat()
    config_path.write_text(config_path.read_text().replace("[motion]", "[motion]\nmin_area = 321"))
    os.utime(config_path, ns=(st.st_atime_ns, st.st_mtime_ns))  # rewrite within mtime granularity
    assert config_path.stat().st_mtime_ns == st.st_mtime_ns
    assert w.check_once() is True
    assert live.get().motion.min_area == 321


def test_touch_without_content_change_does_not_reload(config_path, live):
    w = ConfigWatcher(config_path, live)
    bump_mtime(config_path)
    assert w.check_once() is False
    assert live.version == 0


def test_invalid_file_logs_and_keeps_running_config(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text("this is not toml = [")
    bump_mtime(config_path)
    with caplog.at_level(logging.ERROR):
        assert w.check_once() is False
    assert live.version == 0
    assert "invalid" in caplog.text
    # not re-logged every tick
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert w.check_once() is False
    assert caplog.text == ""


def test_config_error_logs_and_keeps_running_config(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text().replace("[motion]", "[motion]\nmin_area = -5"))
    bump_mtime(config_path)
    with caplog.at_level(logging.ERROR):
        assert w.check_once() is False
    assert live.version == 0 and "motion.min_area" in caplog.text


def test_restored_file_reloads(config_path, live):
    w = ConfigWatcher(config_path, live)
    good = config_path.read_text()
    config_path.write_text("broken = [")
    bump_mtime(config_path)
    assert w.check_once() is False
    config_path.write_text(good.replace("[motion]", "[motion]\nmin_area = 42"))
    bump_mtime(config_path)
    assert w.check_once() is True
    assert live.get().motion.min_area == 42


def test_missing_file_is_ignored(config_path, live):
    w = ConfigWatcher(config_path, live)
    config_path.unlink()
    assert w.check_once() is False


def test_thread_picks_up_change(config_path, live):
    w = ConfigWatcher(config_path, live, interval=0.05)
    w.start()
    try:
        config_path.write_text(config_path.read_text().replace("[motion]", "[motion]\nmin_area = 9"))
        bump_mtime(config_path)
        end = time.monotonic() + 3
        while time.monotonic() < end and live.version == 0:
            time.sleep(0.02)
        assert live.get().motion.min_area == 9
    finally:
        w.stop()


def test_restart_only_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + "\n[web]\nport = 8123\n")
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert live.get().web.port == 8123
    assert "restart" in caplog.text


def test_species_model_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[species]\nmodel = "models/other.onnx"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert live.get().species.model.name == "other.onnx"
    assert "restart birdy to apply them" in caplog.text


def test_save_settings_then_watch_is_noop(config_path, live):
    from birdy.settings import SettingsPatch, save_settings

    w = ConfigWatcher(config_path, live)
    cur = live.get()
    patch = SettingsPatch(
        roi=cur.motion.roi,
        pixel_threshold=cur.motion.pixel_threshold,
        min_area=200,
        background_alpha=cur.motion.background_alpha,
        cooldown_seconds=cur.motion.cooldown_seconds,
        classifier_threshold=cur.classifier.threshold,
        species_threshold=cur.species.threshold,
        species_include=cur.species.include, species_exclude=cur.species.exclude,
        detector_threshold=cur.detector.threshold,
        visits_gap_seconds=cur.visits.gap_seconds,
        lat=cur.site.lat,
        lon=cur.site.lon,
        margin_minutes=cur.daylight.margin_minutes,
        non_bird_days=cur.retention.non_bird_days,
        watchlist=cur.telegram.watchlist,
        alert_new_species=cur.telegram.alert_new_species,
        cooldown_minutes=cur.telegram.cooldown_minutes,
        publish_min_confidence=cur.publish.min_confidence,
        per_species_per_day=cur.publish.per_species_per_day,
    )
    new = save_settings(config_path, cur, patch)
    live.replace(new)  # what the web handler does
    bump_mtime(config_path)  # make sure the watcher sees a new mtime
    assert w.check_once() is False
    assert live.version == 1
    assert live.get().motion.min_area == 200


def test_telegram_token_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[telegram]\nbot_token = "new"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart" in caplog.text and live.get().telegram.bot_token == "new"


def test_publish_restart_only_keys_warn(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[publish]\nsite_id = "other"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart" in caplog.text and live.get().publish.site_id == "other"


def test_detector_model_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[detector]\nmodel = "models/other.tflite"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert live.get().detector.model.name == "other.tflite"
    assert "restart birdy to apply them" in caplog.text


def test_species_bank_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[species]\nbank = "models/other.npz"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart birdy to apply them" in caplog.text
