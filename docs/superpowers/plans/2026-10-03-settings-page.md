# Web Settings Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `/settings` page on the Pi's LAN web app where the ROI polygon is drawn on the live frame and the tuning settings are edited, saved to `birdy.toml`, and applied to the running service without a restart.

**Architecture:** A `LiveConfig` holder (lock + `Config` + version) replaces the static `Config` at every consumer that reads a tunable value. `settings.save_settings()` validates and atomically rewrites `birdy.toml`; a `ConfigWatcher` thread picks up external (SSH) edits. The capture loop rebuilds its detector/gate when the version changes; pipeline, classifier threshold and retention read the live value on use. The web gains `GET/POST /settings` and `GET /frame.jpg`.

**Tech Stack:** existing stack + `tomli-w` (TOML writer) + `python-multipart` (form parsing for Starlette). Vanilla JS on a `<canvas>`; no build step.

**Spec:** `docs/superpowers/specs/2026-10-03-settings-page-design.md`

**Model dispatch hint:** `[opus-ok]` tasks are mechanical; others need more care.

**Branch:** `settings-page` (already created, spec committed).

---

## File structure

| File | Responsibility |
|---|---|
| `birdy/live.py` (new) | `LiveConfig`: thread-safe current `Config` + version |
| `birdy/settings.py` (new) | `SettingsPatch`, `config_to_dict`, `parse_settings_form`, `save_settings` (validate + atomic write) |
| `birdy/watcher.py` (new) | `ConfigWatcher` thread: reload `birdy.toml` on external change |
| `birdy/classifier.py` (modify) | `classify(..., threshold=None)` override |
| `birdy/pipeline.py` (modify) | takes `live`; ROI bbox + threshold per event |
| `birdy/retention.py` (modify) | takes `live`; days per run |
| `birdy/capture.py` (modify) | takes `live`; rebuild detector/gate on version change |
| `birdy/cli.py` (modify) | builds `LiveConfig`, starts watcher, passes `live` + config path |
| `birdy/web/app.py` (modify) | `create_app(store, state, live, config_path, queue_depth)`; `/frame.jpg`, `/settings` |
| `birdy/web/templates/settings.html` (new) | ROI canvas editor + form |
| `birdy/web/templates/index.html` (modify) | header link to `/settings` |
| `tests/conftest.py` (new) | `config_path`, `config`, `live` fixtures |
| `tests/test_live.py`, `tests/test_settings.py`, `tests/test_watcher.py` (new); `tests/test_pipeline.py`, `tests/test_retention.py`, `tests/test_web.py`, `tests/test_classifier.py` (modify) | |
| `pyproject.toml` (modify) | add `tomli-w`, `python-multipart` |
| `CLAUDE.md` (modify) | document live config + settings page |

---

### Task 1: `LiveConfig` + shared test fixtures `[opus-ok]`

**Files:**
- Create: `birdy/live.py`, `tests/conftest.py`, `tests/test_live.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/test_live.py`:
```python
import dataclasses
import threading

from birdy.live import LiveConfig


def test_get_returns_initial_config(config):
    live = LiveConfig(config)
    assert live.get() is config
    assert live.version == 0


def test_replace_bumps_version_and_swaps(config):
    live = LiveConfig(config)
    new = dataclasses.replace(config, motion=dataclasses.replace(config.motion, min_area=999))
    live.replace(new)
    assert live.version == 1
    assert live.get().motion.min_area == 999
    assert live.get() is new


def test_concurrent_readers_and_writers_never_see_torn_state(config):
    live = LiveConfig(config)
    stop = threading.Event()
    errors: list[Exception] = []

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            live.replace(dataclasses.replace(config, motion=dataclasses.replace(config.motion, min_area=i)))

    def reader():
        try:
            while not stop.is_set():
                cfg = live.get()
                assert cfg.motion.min_area >= 0
                _ = live.version
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    threading.Event().wait(0.2)
    stop.set()
    for t in threads:
        t.join(2)
    assert errors == []
    assert live.version > 0
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_live.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.live'` (from conftest).

- [ ] **Step 4: Write `birdy/live.py`**

```python
"""Thread-safe holder for the current Config, with a version counter.

Consumers call `get()` whenever they need a tunable value and compare
`version` to decide whether to rebuild derived objects (e.g. the motion
detector). `Config` is frozen, so a returned snapshot never changes.
"""

from __future__ import annotations

import threading

from birdy.config import Config


class LiveConfig:
    def __init__(self, config: Config) -> None:
        self._lock = threading.Lock()
        self._config = config
        self._version = 0

    def get(self) -> Config:
        with self._lock:
            return self._config

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def replace(self, config: Config) -> None:
        with self._lock:
            self._config = config
            self._version += 1
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_live.py -q` → `3 passed`. Then `.venv/bin/pytest -q` → all previous tests still pass (conftest must not break anything).

- [ ] **Step 6: Commit**

```bash
git add birdy/live.py tests/conftest.py tests/test_live.py
git commit -m "Add LiveConfig holder and shared config fixtures"
```

---

### Task 2: `settings.py` — patch, form parsing, atomic save `[opus-ok]`

**Files:**
- Modify: `pyproject.toml` (dependencies)
- Create: `birdy/settings.py`, `tests/test_settings.py`

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `[project] dependencies`, add two lines after `"jinja2>=3.1",`:
```toml
    "tomli-w>=1.0",
    "python-multipart>=0.0.9",
```
Run: `.venv/bin/pip install -q -e '.[dev]' && .venv/bin/python -c "import tomli_w, multipart; print('deps ok')"` → `deps ok`.

- [ ] **Step 2: Write the failing tests**

`tests/test_settings.py`:
```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_settings.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.settings'`.

- [ ] **Step 4: Write `birdy/settings.py`**

```python
"""Editable settings: form parsing, config serialisation and atomic save."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from birdy.config import Config, ConfigError, load_config

Point = tuple[int, int]

HEADER = "# Written by the birdy settings page. Comments live in birdy.toml.example.\n\n"

# form field -> config key (used for error messages and field highlighting)
FIELD_KEYS = {
    "roi": "motion.roi",
    "pixel_threshold": "motion.pixel_threshold",
    "min_area": "motion.min_area",
    "background_alpha": "motion.background_alpha",
    "cooldown_seconds": "motion.cooldown_seconds",
    "classifier_threshold": "classifier.threshold",
    "lat": "site.lat",
    "lon": "site.lon",
    "margin_minutes": "daylight.margin_minutes",
    "non_bird_days": "retention.non_bird_days",
}


@dataclass(frozen=True)
class SettingsPatch:
    """Exactly the keys the settings page may change."""

    roi: tuple[Point, ...]
    pixel_threshold: int
    min_area: int
    background_alpha: float
    cooldown_seconds: float
    classifier_threshold: float
    lat: float | None
    lon: float | None
    margin_minutes: int
    non_bird_days: int


def config_to_dict(config: Config) -> dict:
    """Every key of the config as plain TOML-serialisable values (paths absolute, no None)."""
    site: dict = {"name": config.site.name}
    if config.site.lat is not None and config.site.lon is not None:
        site["lat"] = config.site.lat
        site["lon"] = config.site.lon
    return {
        "site": site,
        "camera": {
            "main_size": list(config.camera.main_size),
            "lores_size": list(config.camera.lores_size),
            "jpeg_quality": config.camera.jpeg_quality,
        },
        "motion": {
            "roi": [list(p) for p in config.motion.roi],
            "pixel_threshold": config.motion.pixel_threshold,
            "min_area": config.motion.min_area,
            "background_alpha": config.motion.background_alpha,
            "cooldown_seconds": config.motion.cooldown_seconds,
        },
        "daylight": {"margin_minutes": config.daylight.margin_minutes},
        "classifier": {
            "model": str(config.classifier.model),
            "labels": str(config.classifier.labels),
            "bird_classes": str(config.classifier.bird_classes),
            "threshold": config.classifier.threshold,
        },
        "retention": {"non_bird_days": config.retention.non_bird_days},
        "storage": {"data_dir": str(config.storage.data_dir)},
        "web": {"host": config.web.host, "port": config.web.port},
    }


def apply_patch(data: dict, patch: SettingsPatch) -> dict:
    data["motion"].update(
        roi=[list(p) for p in patch.roi],
        pixel_threshold=patch.pixel_threshold,
        min_area=patch.min_area,
        background_alpha=patch.background_alpha,
        cooldown_seconds=patch.cooldown_seconds,
    )
    data["classifier"]["threshold"] = patch.classifier_threshold
    data["site"].pop("lat", None)
    data["site"].pop("lon", None)
    # Write whichever is given; load_config rejects one-without-the-other with a clear message.
    if patch.lat is not None:
        data["site"]["lat"] = patch.lat
    if patch.lon is not None:
        data["site"]["lon"] = patch.lon
    data["daylight"]["margin_minutes"] = patch.margin_minutes
    data["retention"]["non_bird_days"] = patch.non_bird_days
    return data


def save_settings(path: Path, current: Config, patch: SettingsPatch) -> Config:
    """Validate `current` + `patch`, write `path` atomically, return the new Config.

    On ConfigError nothing is written and the temp file is removed.
    """
    path = Path(path)
    data = apply_patch(config_to_dict(current), patch)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(HEADER + tomli_w.dumps(data))
    try:
        new = load_config(tmp)
    except ConfigError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return new


def _num(form: Mapping[str, str], field: str, cast: type, optional: bool = False):
    key = FIELD_KEYS[field]
    raw = form.get(field)
    if raw is None or raw.strip() == "":
        if optional:
            return None
        raise ConfigError(f"{key} is required")
    try:
        if cast is int:
            # int("7.5") raises; int(float("7.5")) would silently truncate, so go via str check
            return int(raw.strip())
        return float(raw.strip())
    except ValueError:
        raise ConfigError(f"{key} must be a {cast.__name__}") from None


def _roi(form: Mapping[str, str]) -> tuple[Point, ...]:
    key = FIELD_KEYS["roi"]
    raw = form.get("roi", "")
    try:
        pts = json.loads(raw)
    except json.JSONDecodeError:
        raise ConfigError(f"{key} must be JSON like [[x, y], ...]") from None
    if not isinstance(pts, list):
        raise ConfigError(f"{key} must be a list of [x, y] points")
    out = []
    for p in pts:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in p)):
            raise ConfigError(f"{key} points must be [x, y] integers")
        out.append((p[0], p[1]))
    return tuple(out)


def parse_settings_form(form: Mapping[str, str]) -> SettingsPatch:
    """Turn the POSTed form into a SettingsPatch; raises ConfigError naming the config key."""
    return SettingsPatch(
        roi=_roi(form),
        pixel_threshold=_num(form, "pixel_threshold", int),
        min_area=_num(form, "min_area", int),
        background_alpha=_num(form, "background_alpha", float),
        cooldown_seconds=_num(form, "cooldown_seconds", float),
        classifier_threshold=_num(form, "classifier_threshold", float),
        lat=_num(form, "lat", float, optional=True),
        lon=_num(form, "lon", float, optional=True),
        margin_minutes=_num(form, "margin_minutes", int),
        non_bird_days=_num(form, "non_bird_days", int),
    )
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_settings.py -q` → `19 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml birdy/settings.py tests/test_settings.py
git commit -m "Add settings patch parsing and atomic config save"
```

---

### Task 3: `ConfigWatcher` `[opus-ok]`

**Files:**
- Create: `birdy/watcher.py`, `tests/test_watcher.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_watcher.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_watcher.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.watcher'`.

- [ ] **Step 3: Write `birdy/watcher.py`**

```python
"""Reload birdy.toml into LiveConfig when it changes on disk (e.g. edited over SSH)."""

from __future__ import annotations

import logging
import threading
import tomllib
from pathlib import Path

from birdy.config import ConfigError, load_config
from birdy.live import LiveConfig

log = logging.getLogger(__name__)


class ConfigWatcher:
    def __init__(self, path: Path, live: LiveConfig, interval: float = 5.0) -> None:
        self._path = Path(path)
        self._live = live
        self._interval = interval
        self._last_mtime_ns = self._mtime_ns()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="config-watcher", daemon=True)

    def _mtime_ns(self) -> int | None:
        try:
            return self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def check_once(self) -> bool:
        """Reload if the file changed and differs from the running config. Returns True on reload."""
        mtime = self._mtime_ns()
        if mtime is None or mtime == self._last_mtime_ns:
            return False
        self._last_mtime_ns = mtime  # whatever happens, don't re-process this mtime
        try:
            cfg = load_config(self._path)
        except (ConfigError, tomllib.TOMLDecodeError, OSError) as e:
            log.error("%s changed but is invalid (%s); keeping running settings", self._path, e)
            return False
        if cfg == self._live.get():
            return False  # e.g. the settings page just wrote it
        self._live.replace(cfg)
        log.info("settings reloaded from %s (v%d)", self._path, self._live.version)
        return True

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check_once()
            except Exception:
                log.exception("config watcher failed")
```

Note: the spec described a content hash; comparing the loaded `Config` to the running one is equivalent for our purpose (frozen dataclass equality) and avoids keeping a hash in sync with the web's own writes.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_watcher.py -q` → `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/watcher.py tests/test_watcher.py
git commit -m "Add ConfigWatcher for external birdy.toml edits"
```

---

### Task 4: Classifier threshold override + pipeline reads live config `[opus-ok]`

**Files:**
- Modify: `birdy/classifier.py` (Protocol + `classify`), `birdy/pipeline.py`, `tests/test_classifier.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classifier.py` inside `class TestRealModel` (uses the existing `clf` fixture):
```python
    def test_threshold_override(self, clf):
        p_default = clf.classify(FIXTURES / "bird.jpg", roi_bbox=None)
        p_strict = clf.classify(FIXTURES / "bird.jpg", roi_bbox=None, threshold=0.9999)
        assert p_default.is_bird is True
        assert p_strict.is_bird is False
        assert p_default.confidence == p_strict.confidence
```

In `tests/test_pipeline.py`:
1. Change `FakeClassifier.classify` to record everything but keep `calls` as a list of paths:
```python
class FakeClassifier:
    def __init__(self, fail_on: set[str] = ()):
        self.fail_on = set(fail_on)
        self.calls: list[Path] = []
        self.bboxes: list = []
        self.thresholds: list = []

    def classify(self, image_path, roi_bbox, threshold=None):
        self.calls.append(image_path)
        self.bboxes.append(roi_bbox)
        self.thresholds.append(threshold)
        if image_path.name in self.fail_on:
            raise RuntimeError("boom")
        return Prediction(is_bird=True, confidence=0.9)
```
2. Every `Pipeline(store, <clf>, roi_bbox=...)` call becomes `Pipeline(store, <clf>, live, ...)` — add `live` to each test's parameters (it's a conftest fixture). Remove the now-unused `BBox` import if nothing else uses it.
3. Add:
```python
def test_roi_and_threshold_come_from_live_config(store, live, config):
    import dataclasses

    from birdy.motion import roi_bbox

    clf = FakeClassifier()
    p = Pipeline(store, clf, live)
    ev1 = add(store, 0)
    p.process(ev1.capture_id, ev1.path)
    assert clf.bboxes[-1] == roi_bbox(config.motion.roi)
    assert clf.thresholds[-1] == 0.5

    new = dataclasses.replace(
        config,
        motion=dataclasses.replace(config.motion, roi=((0, 0), (10, 0), (10, 10))),
        classifier=dataclasses.replace(config.classifier, threshold=0.9),
    )
    live.replace(new)
    ev2 = add(store, 1)
    p.process(ev2.capture_id, ev2.path)
    assert clf.bboxes[-1] == roi_bbox(new.motion.roi)
    assert clf.thresholds[-1] == 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_classifier.py -q`
Expected: FAIL — `TypeError` on the `Pipeline(...)` calls (`live` not accepted) and `classify() got an unexpected keyword argument 'threshold'`.

- [ ] **Step 3: Modify `birdy/classifier.py`**

Protocol:
```python
class Classifier(Protocol):
    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction: ...
```
`ImageNetBirdClassifier.classify`:
```python
    def classify(self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None) -> Prediction:
        """`threshold` overrides the constructor default so it can be tuned live."""
        t = self._threshold if threshold is None else threshold
        conf = bird_probability(self._probs(image_path, roi_bbox), self._bird_indices)
        return Prediction(is_bird=conf >= t, confidence=conf)
```

- [ ] **Step 4: Modify `birdy/pipeline.py`**

Imports: replace `from birdy.motion import BBox` with `from birdy.motion import roi_bbox` and add `from birdy.live import LiveConfig`.

Constructor:
```python
    def __init__(
        self,
        store: Store,
        classifier: Classifier,
        live: LiveConfig,
        maxsize: int = 50,
        sweep_interval: float = 300.0,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._live = live
        ...
```
(remove `self._roi_bbox`.)

`process`, first statement:
```python
        cfg = self._live.get()
        try:
            pred = self._classifier.classify(path, roi_bbox(cfg.motion.roi), threshold=cfg.classifier.threshold)
        except Exception:
```
(the rest unchanged.)

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_classifier.py -q` → `10 passed` + `15 passed` (counts: pipeline 9→10, classifier 14→15). Then `.venv/bin/pytest -q`: `tests/test_cli.py` and `tests/test_web.py` still pass (cli constructs `Pipeline` only inside `cmd_run`, which Task 7 updates; if `test_cli` fails on `Pipeline(...)` at import time, it doesn't — it's inside a function).

- [ ] **Step 6: Commit**

```bash
git add birdy/classifier.py birdy/pipeline.py tests/test_classifier.py tests/test_pipeline.py
git commit -m "Pipeline reads ROI and threshold from LiveConfig"
```

---

### Task 5: Retention reads live config `[opus-ok]`

**Files:**
- Modify: `birdy/retention.py`, `tests/test_retention.py`

- [ ] **Step 1: Update the tests**

In `tests/test_retention.py`, `test_job_thread_runs_periodically` becomes:
```python
def test_job_thread_runs_periodically(store, live):
    add_non_bird(store, T0 - timedelta(days=10))
    job = RetentionJob(store, live, interval=0.1)
    job.start()
    try:
        end = time.monotonic() + 3
        while time.monotonic() < end and store.count("all") != 0:
            time.sleep(0.02)
        assert store.count("all") == 0
    finally:
        job.stop()
```
Add:
```python
def test_job_uses_current_days(store, live, config):
    import dataclasses

    add_non_bird(store, T0 - timedelta(days=3))  # younger than the default 7 days
    live.replace(dataclasses.replace(config, retention=dataclasses.replace(config.retention, non_bird_days=1)))
    job = RetentionJob(store, live, interval=0.1)
    job.start()
    try:
        end = time.monotonic() + 3
        while time.monotonic() < end and store.count("all") != 0:
            time.sleep(0.02)
        assert store.count("all") == 0
    finally:
        job.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_retention.py -q` → FAIL (`RetentionJob.__init__()` signature).

- [ ] **Step 3: Modify `birdy/retention.py`**

```python
from birdy.live import LiveConfig
...
class RetentionJob:
    def __init__(self, store: Store, live: LiveConfig, interval: float = 3600.0) -> None:
        self._store = store
        self._live = live
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="retention", daemon=True)
    ...
    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                run_once(self._store, self._live.get().retention.non_bird_days)
            except Exception:
                log.exception("retention run failed")
```
(`run_once(store, non_bird_days, now)` unchanged — the CLI `purge` still calls it with a plain int.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_retention.py -q` → `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/retention.py tests/test_retention.py
git commit -m "Retention reads days from LiveConfig"
```

---

### Task 6: Web — `/frame.jpg`, `GET/POST /settings`, template

**Files:**
- Modify: `birdy/web/app.py`, `birdy/web/templates/index.html`, `tests/test_web.py`
- Create: `birdy/web/templates/settings.html`

- [ ] **Step 1: Update existing web tests for the new `create_app` signature**

In `tests/test_web.py`: delete the module-level `ROI = ...` constant; change the client fixture to:
```python
@pytest.fixture
def client(store, state, live, config_path):
    return TestClient(create_app(store, state, live, config_path))
```
(`live` and `config_path` come from conftest; the conftest ROI is the same polygon the old constant had.)

- [ ] **Step 2: Add the failing settings tests** (append to `tests/test_web.py`)

```python
import json
import re

from birdy.config import load_config


def post_settings(client, **overrides):
    form = {
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
    form.update(overrides)
    return client.post("/settings", data=form, follow_redirects=False)


def test_frame_jpg_404_without_frame(client):
    assert client.get("/frame.jpg").status_code == 404


def test_frame_jpg_returns_jpeg(client, state):
    state.update(latest_lores=np.full((240, 320), 100, np.uint8))
    r = client.get("/frame.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.headers["cache-control"] == "no-store"
    assert r.content[:2] == b"\xff\xd8"


def test_index_links_to_settings(client):
    assert 'href="/settings"' in client.get("/").text


def test_settings_page_shows_current_values(client, config):
    r = client.get("/settings")
    assert r.status_code == 200
    assert 'name="min_area"' in r.text and 'value="150"' in r.text
    assert json.dumps([list(p) for p in config.motion.roi]) in r.text
    assert "lores_w = 320" in r.text or "320" in r.text
    assert "edit birdy.toml" in r.text  # read-only block
    assert str(config.camera.main_size[0]) in r.text


def test_settings_post_saves_applies_and_redirects(client, config_path, live):
    r = post_settings(client, min_area="400", roi="[[10, 10], [100, 10], [100, 100]]", lat="38.72", lon="-9.14")
    assert r.status_code == 303
    assert r.headers["location"] == "/settings?saved=1"
    assert live.version == 1
    cfg = live.get()
    assert cfg.motion.min_area == 400 and cfg.motion.roi == ((10, 10), (100, 10), (100, 100))
    assert cfg.site.lat == 38.72
    assert load_config(config_path) == cfg
    assert "applied live" in client.get("/settings?saved=1").text


@pytest.mark.parametrize(
    "overrides, key",
    [
        ({"roi": "[[0, 0], [1, 1]]"}, "motion.roi"),
        ({"min_area": "-1"}, "motion.min_area"),
        ({"lat": "38.7"}, "site.lat"),
        ({"background_alpha": "abc"}, "motion.background_alpha"),
    ],
)
def test_settings_post_invalid_renders_error_and_writes_nothing(client, config_path, live, overrides, key):
    before = config_path.read_bytes()
    r = post_settings(client, **overrides)
    assert r.status_code == 422
    assert key in r.text
    assert re.search(r'class="[^"]*error[^"]*"', r.text)
    assert config_path.read_bytes() == before
    assert live.version == 0


def test_settings_post_keeps_submitted_values_on_error(client):
    r = post_settings(client, min_area="-1", pixel_threshold="77")
    assert 'value="77"' in r.text
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: FAIL — `create_app()` signature `TypeError` in the client fixture (all tests error).

- [ ] **Step 4: Modify `birdy/web/app.py`**

Replace the imports block's `from fastapi.responses import HTMLResponse, Response` with:
```python
from fastapi.responses import HTMLResponse, RedirectResponse, Response
```
Add imports:
```python
from birdy.config import ConfigError
from birdy.live import LiveConfig
from birdy.settings import FIELD_KEYS, parse_settings_form, save_settings
```
Remove `from collections.abc import Callable, Sequence` → `from collections.abc import Callable`.

New signature and body changes:
```python
def create_app(
    store: Store,
    state: RuntimeState,
    live: LiveConfig,
    config_path: Path,
    queue_depth: Callable[[], int] = lambda: 0,
) -> FastAPI:
```
In `roi_png`, replace `np.array(roi, ...)` with `np.array(live.get().motion.roi, dtype=np.int32).reshape((-1, 1, 2))`.

Add, before `return app`:
```python
    @app.get("/frame.jpg")
    def frame_jpg():
        frame = state.lores_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=jpg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def settings_context(request: Request, form: dict | None = None, error: tuple[str | None, str] | None = None, saved: bool = False) -> dict:
        cfg = live.get()
        values = form if form is not None else {
            "roi": [list(p) for p in cfg.motion.roi],
            "pixel_threshold": cfg.motion.pixel_threshold,
            "min_area": cfg.motion.min_area,
            "background_alpha": cfg.motion.background_alpha,
            "cooldown_seconds": cfg.motion.cooldown_seconds,
            "classifier_threshold": cfg.classifier.threshold,
            "lat": "" if cfg.site.lat is None else cfg.site.lat,
            "lon": "" if cfg.site.lon is None else cfg.site.lon,
            "margin_minutes": cfg.daylight.margin_minutes,
            "non_bird_days": cfg.retention.non_bird_days,
        }
        if isinstance(values["roi"], str):
            # re-rendering a failed POST: keep the raw string the browser sent
            roi_json = values["roi"]
        else:
            roi_json = json.dumps(values["roi"])
        return {
            "v": values,
            "roi_json": roi_json,
            "lores_w": cfg.camera.lores_size[0],
            "lores_h": cfg.camera.lores_size[1],
            "readonly": {
                "camera.main_size": list(cfg.camera.main_size),
                "camera.lores_size": list(cfg.camera.lores_size),
                "camera.jpeg_quality": cfg.camera.jpeg_quality,
                "web.host": cfg.web.host,
                "web.port": cfg.web.port,
                "storage.data_dir": str(cfg.storage.data_dir),
                "classifier.model": str(cfg.classifier.model),
            },
            "error_field": None if error is None else error[0],
            "error_message": None if error is None else error[1],
            "saved": saved,
            "version": live.version,
            "status": status_payload(),
        }

    @app.get("/settings", response_class=HTMLResponse)
    def settings_get(request: Request, saved: int = 0):
        return templates.TemplateResponse(request, "settings.html", settings_context(request, saved=bool(saved)))

    @app.post("/settings")
    async def settings_post(request: Request):
        form = {k: str(v) for k, v in (await request.form()).items()}
        try:
            patch = parse_settings_form(form)
            new = save_settings(config_path, live.get(), patch)
        except ConfigError as e:
            msg = str(e)
            field = next((f for f, key in FIELD_KEYS.items() if msg.startswith(key)), None)
            ctx = settings_context(request, form=form, error=(field, msg))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=422)
        except OSError as e:
            ctx = settings_context(request, form=form, error=(None, f"could not write {config_path}: {e}"))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=500)
        live.replace(new)
        return RedirectResponse("/settings?saved=1", status_code=303)
```
Add `import json` at the top.

- [ ] **Step 5: Add the header link in `birdy/web/templates/index.html`**

In the `<div class="status">`, change `<span><a href="/roi.png">roi</a></span>` to:
```html
    <span><a href="/settings">settings</a></span>
```

- [ ] **Step 6: Write `birdy/web/templates/settings.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>birdy settings</title>
<style>
  :root { color-scheme: light dark; --bg: #fafaf7; --fg: #222; --muted: #777; --card: #fff; --line: #ddd; --ok: #2a7; --bad: #c66; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #eee; --muted: #999; --card: #222; --line: #333; } }
  body { margin: 0; padding: 16px; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); max-width: 720px; }
  a { color: var(--fg); }
  h1 { font-size: 20px; margin: 0 0 12px; } h2 { font-size: 15px; margin: 20px 0 8px; }
  canvas { width: 100%; max-width: 640px; aspect-ratio: {{ lores_w }} / {{ lores_h }}; background: #333; border: 1px solid var(--line); touch-action: none; display: block; }
  .hint { color: var(--muted); font-size: 12px; margin: 6px 0; }
  .readout { margin: 8px 0; } .readout b { font-size: 18px; } .readout b.hot { color: var(--ok); }
  .banner { padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; }
  .banner.ok { background: #2a72; border: 1px solid var(--ok); } .banner.bad { background: #c662; border: 1px solid var(--bad); }
  .row { display: grid; grid-template-columns: 160px 1fr 90px; gap: 8px; align-items: center; margin: 6px 0; }
  .row.error label { color: var(--bad); font-weight: bold; }
  .row input[type=number], .row input[type=text] { width: 100%; box-sizing: border-box; padding: 4px 6px; background: var(--card); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; }
  .ro { color: var(--muted); font-size: 12px; } .ro code { color: var(--fg); }
  button { padding: 8px 16px; border-radius: 6px; border: 1px solid var(--line); background: var(--card); color: var(--fg); cursor: pointer; }
  button.primary { background: var(--ok); color: #fff; border-color: var(--ok); font-weight: bold; }
</style>
</head>
<body>
<h1><a href="/">birdy</a> › settings</h1>

{% if saved %}<div class="banner ok">Saved — applied live (v{{ version }}).</div>{% endif %}
{% if error_message %}<div class="banner bad">{{ error_message }}</div>{% endif %}

<form method="post" action="/settings" id="form">
<h2>Region of interest</h2>
<canvas id="roi" width="{{ lores_w * 2 }}" height="{{ lores_h * 2 }}"></canvas>
<div class="hint">Click to add a corner, drag to move it, click a corner to delete it (min 3). Coordinates are in the {{ lores_w }}×{{ lores_h }} analysis frame.</div>
<div class="readout">motion score <b id="score">–</b> <span class="hint">(green when ≥ min_area; state <span id="state">{{ status.state }}</span>)</span>
  <button type="button" id="reset">Reset polygon</button></div>
<input type="hidden" name="roi" id="roi-field" value='{{ roi_json }}'>
<div id="roi-hint" class="hint {{ 'error' if error_field == 'roi' else '' }}">{{ error_message if error_field == 'roi' else '' }}</div>

<h2>Motion</h2>
{% macro num(name, label, min, max, step) -%}
<div class="row {{ 'error' if error_field == name else '' }}">
  <label for="{{ name }}">{{ label }}</label>
  <input type="range" min="{{ min }}" max="{{ max }}" step="{{ step }}" value="{{ v[name] }}" data-pair="{{ name }}">
  <input type="number" id="{{ name }}" name="{{ name }}" min="{{ min }}" max="{{ max }}" step="{{ step }}" value="{{ v[name] }}">
</div>
{%- endmacro %}
{{ num("pixel_threshold", "pixel_threshold", 0, 100, 1) }}
{{ num("min_area", "min_area", 0, 20000, 10) }}
{{ num("background_alpha", "background_alpha", 0.01, 1, 0.01) }}
{{ num("cooldown_seconds", "cooldown_seconds", 0, 60, 0.5) }}

<h2>Classifier</h2>
{{ num("classifier_threshold", "threshold (P bird)", 0, 1, 0.01) }}

<h2>Site &amp; daylight</h2>
<div class="row {{ 'error' if error_field in ('lat', 'lon') else '' }}">
  <label for="lat">lat / lon</label>
  <input type="text" id="lat" name="lat" placeholder="38.72 (empty = gate off)" value="{{ v['lat'] }}">
  <input type="text" id="lon" name="lon" placeholder="-9.14" value="{{ v['lon'] }}">
</div>
{{ num("margin_minutes", "margin_minutes", 0, 180, 5) }}

<h2>Retention</h2>
{{ num("non_bird_days", "non_bird_days", 1, 365, 1) }}

<p><button class="primary" type="submit">Save</button> <span class="hint">writes birdy.toml and applies immediately</span></p>
</form>

<h2>Read-only (edit birdy.toml and restart)</h2>
<div class="ro">
{% for k, val in readonly.items() %}<div><code>{{ k }}</code> = {{ val }}</div>{% endfor %}
</div>

<script>
(function () {
  const W = {{ lores_w }}, H = {{ lores_h }}, SCALE = 2;
  const canvas = document.getElementById('roi'), ctx = canvas.getContext('2d');
  const field = document.getElementById('roi-field'), hint = document.getElementById('roi-hint');
  const saved = JSON.parse(field.value);
  let pts = JSON.parse(JSON.stringify(saved));
  let img = null, dragging = null, moved = false;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2; ctx.strokeStyle = '#0f0'; ctx.fillStyle = '#0f0';
    ctx.beginPath();
    pts.forEach((p, i) => { const x = p[0] * SCALE, y = p[1] * SCALE; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    if (pts.length > 2) ctx.closePath();
    ctx.stroke();
    pts.forEach(p => { ctx.beginPath(); ctx.arc(p[0] * SCALE, p[1] * SCALE, 6, 0, Math.PI * 2); ctx.fill(); });
    field.value = JSON.stringify(pts);
  }
  function toLores(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * W / r.width, y = (e.clientY - r.top) * H / r.height;
    return [Math.max(0, Math.min(W - 1, Math.round(x))), Math.max(0, Math.min(H - 1, Math.round(y)))];
  }
  function hit(p) {
    const r = canvas.getBoundingClientRect();
    const radius = 12 * W / r.width;  // 12 CSS px, in lores units
    return pts.findIndex(q => Math.hypot(q[0] - p[0], q[1] - p[1]) <= radius);
  }
  canvas.addEventListener('pointerdown', e => {
    e.preventDefault();
    const p = toLores(e), i = hit(p);
    moved = false; hint.textContent = '';
    if (i >= 0) { dragging = i; canvas.setPointerCapture(e.pointerId); }
    else { pts.push(p); draw(); }
  });
  canvas.addEventListener('pointermove', e => {
    if (dragging === null) return;
    const p = toLores(e);
    if (p[0] !== pts[dragging][0] || p[1] !== pts[dragging][1]) { pts[dragging] = p; moved = true; draw(); }
  });
  canvas.addEventListener('pointerup', e => {
    if (dragging === null) return;
    if (!moved) {
      if (pts.length > 3) pts.splice(dragging, 1);
      else hint.textContent = 'need at least 3 points';
    }
    dragging = null; draw();
  });
  document.getElementById('reset').onclick = () => { pts = JSON.parse(JSON.stringify(saved)); draw(); };

  function refreshFrame() {
    const i = new Image();
    i.onload = () => { img = i; draw(); };
    i.src = '/frame.jpg?t=' + Date.now();
  }
  async function poll() {
    try {
      const s = await (await fetch('/status')).json();
      const el = document.getElementById('score');
      el.textContent = s.motion_score;
      el.className = s.motion_score >= +document.getElementById('min_area').value ? 'hot' : '';
      document.getElementById('state').textContent = s.state;
    } catch (e) { /* page still usable without live status */ }
  }
  document.querySelectorAll('[data-pair]').forEach(range => {
    const num = document.getElementById(range.dataset.pair);
    range.oninput = () => { num.value = range.value; };
    num.oninput = () => { range.value = num.value; };
  });
  draw(); refreshFrame(); poll();
  setInterval(refreshFrame, 1000); setInterval(poll, 1000);
})();
</script>
</body>
</html>
```

- [ ] **Step 7: Run tests**

Run: `.venv/bin/pytest tests/test_web.py -q` → `12 + 10 = 22 passed` (8 original + 4 added earlier + 10 new). Then `.venv/bin/pytest -q` — `tests/test_cli.py` must still pass (CLI not yet updated; its tests don't construct the app).

- [ ] **Step 8: Manual look (Mac)**

Serve the app against a seeded temp store to eyeball the page:
```bash
cd /Users/yassine/www/pi-birdy && .venv/bin/python - <<'EOF'
import threading, numpy as np, uvicorn
from pathlib import Path
from birdy.config import load_config
from birdy.live import LiveConfig
from birdy.state import RuntimeState
from birdy.store import Store
from birdy.web.app import create_app
root = Path("/private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/webdemo"); root.mkdir(parents=True, exist_ok=True)
cfg_path = root / "birdy.toml"
cfg_path.write_text('[site]\nname = "demo"\n[motion]\nroi = [[40, 60], [280, 60], [280, 220], [40, 220]]\n[storage]\ndata_dir = "data"\n')
live = LiveConfig(load_config(cfg_path)); state = RuntimeState()
y, x = np.mgrid[0:240, 0:320]; state.update(state="capturing", motion_score=12, latest_lores=((x + y) % 256).astype(np.uint8))
uvicorn.run(create_app(Store(root / "data"), state, live, cfg_path), host="127.0.0.1", port=8777)
EOF
```
Open http://127.0.0.1:8777/settings: the gradient frame shows with the green polygon; click/drag/delete corners; sliders sync; Save → banner "Saved — applied live (v1)"; `cat` the demo `birdy.toml` to see the rewrite. Ctrl-C to stop.

- [ ] **Step 9: Commit**

```bash
git add birdy/web/ tests/test_web.py
git commit -m "Web: settings page with ROI editor, /frame.jpg"
```

---

### Task 7: Capture hot-reload + CLI wiring

**Files:**
- Modify: `birdy/capture.py` (`run_capture`), `birdy/cli.py` (`cmd_run`, `main`), `tests/test_cli.py`

- [ ] **Step 1: Rewrite `run_capture` in `birdy/capture.py`**

Replace the import `from birdy.config import Config` with:
```python
from birdy.config import Config
from birdy.live import LiveConfig
```
Add a helper above `run_capture`:
```python
def _build_tunables(cfg: Config) -> tuple[MotionDetector, DaylightGate]:
    # MotionDetector's default warmup_frames (10) suppresses triggers while
    # auto-exposure settles after the camera session starts, and again for ~1 s
    # after a settings change rebuilds the detector.
    detector = MotionDetector(
        frame_size=cfg.camera.lores_size,
        roi=cfg.motion.roi,
        pixel_threshold=cfg.motion.pixel_threshold,
        min_area=cfg.motion.min_area,
        background_alpha=cfg.motion.background_alpha,
        cooldown_seconds=cfg.motion.cooldown_seconds,
    )
    gate = DaylightGate(cfg.site.lat, cfg.site.lon, cfg.daylight.margin_minutes)
    return detector, gate
```
Replace the whole `run_capture` with:
```python
def run_capture(
    live: LiveConfig,
    store: Store,
    pipeline: Pipeline,
    state: RuntimeState,
    stop: threading.Event,
) -> None:
    global _low_disk_warned
    paused_logged = False
    version = live.version
    config = live.get()  # camera settings are read once per session
    detector, gate = _build_tunables(config)

    def refresh_tunables() -> None:
        nonlocal version, detector, gate
        if live.version != version:
            version = live.version
            detector, gate = _build_tunables(live.get())
            log.info("settings applied (v%d)", version)

    while not stop.is_set():
        refresh_tunables()
        if not gate.is_active(datetime.now().astimezone()):
            state.update(state="daylight_paused", motion_score=0)
            if not paused_logged:
                log.info("outside daylight window; next change %s", gate.next_change(datetime.now().astimezone()))
                paused_logged = True
            stop.wait(GATE_RECHECK)
            continue
        if paused_logged:
            log.info("daylight window open; starting camera")
            paused_logged = False

        config = live.get()
        detector, gate = _build_tunables(config)
        version = live.version
        try:
            with camera_session(config) as picam2:
                state.update(state="capturing")
                _low_disk_warned = False
                last_gate_check = time.monotonic()
                consecutive_errors = 0
                while not stop.is_set():
                    refresh_tunables()
                    if time.monotonic() - last_gate_check > GATE_RECHECK:
                        last_gate_check = time.monotonic()
                        if not gate.is_active(datetime.now().astimezone()):
                            break
                    t0 = time.monotonic()
                    try:
                        frame = lores_grey(picam2, config.camera.lores_size)
                        result = detector.process(frame)
                        # No copy needed: capture_buffer returns a fresh array each call and
                        # RuntimeState.lores_frame() copies on read.
                        state.update(motion_score=result.score, latest_lores=frame)
                        if result.triggered:
                            _capture_still(picam2, store, pipeline, result.score)
                        consecutive_errors = 0
                    except Exception:
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                            log.error("%d consecutive frame failures; giving up on this camera session", consecutive_errors)
                            raise RuntimeError("camera stopped delivering frames")
                        log.exception("frame processing failed; continuing")
                    elapsed = time.monotonic() - t0
                    if elapsed < FRAME_INTERVAL:
                        stop.wait(FRAME_INTERVAL - elapsed)
        except Exception:
            state.update(state="camera_error")
            log.exception("camera session failed")
            raise
```
(`camera_session(config)` and `grab_lores_frame(config)` keep taking a `Config`.)

Run: `.venv/bin/python -m py_compile birdy/capture.py && .venv/bin/python -c "import birdy.capture; print('import ok')"` → `import ok`.

- [ ] **Step 2: Modify `birdy/cli.py`**

`cmd_run` takes the config path and wires everything through `live`:
```python
def cmd_run(config: Config, config_path: Path) -> int:
    import uvicorn

    from birdy.capture import run_capture
    from birdy.live import LiveConfig
    from birdy.pipeline import Pipeline
    from birdy.retention import RetentionJob
    from birdy.state import RuntimeState
    from birdy.watcher import ConfigWatcher
    from birdy.web.app import create_app

    live = LiveConfig(config)
    store = Store(config.storage.data_dir)
    store.set_site(config.site.name, config.site.lat, config.site.lon)
    state = RuntimeState()

    pipeline = Pipeline(store, _classifier(config), live)
    pipeline.start()
    retention = RetentionJob(store, live)
    retention.start()
    watcher = ConfigWatcher(config_path, live)
    watcher.start()

    app = create_app(store, state, live, config_path, queue_depth=pipeline.queue_depth)
    server = uvicorn.Server(uvicorn.Config(app, host=config.web.host, port=config.web.port, log_level="warning"))
    web_thread = threading.Thread(target=server.run, name="web", daemon=True)
    web_thread.start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    log.info(
        "birdy starting: site=%s daylight_gate=%s web=%s:%d",
        config.site.name, config.site.lat is not None, config.web.host, config.web.port,
    )
    try:
        # If the port is taken, uvicorn sys.exit(1)s inside its thread; without this
        # check the service would run headless forever. Fail fast so systemd restarts us.
        for _ in range(50):
            if server.started or not web_thread.is_alive():
                break
            time.sleep(0.1)
        if not server.started:
            log.error("web server failed to start on %s:%d", config.web.host, config.web.port)
            return 1
        run_capture(live, store, pipeline, state, stop)
        return 0
    except Exception:
        # capture.py already logged the traceback via log.exception.
        return 1
    finally:
        server.should_exit = True
        watcher.stop()
        pipeline.stop()
        retention.stop()
        store.close()
```
Remove the now-unused `from birdy.daylight import DaylightGate` import inside `cmd_run` and `from birdy.motion import roi_bbox` at the top if nothing else uses it (check with pyflakes). In `main()`, the `run` branch becomes `return cmd_run(config, Path(args.config))`.

- [ ] **Step 3: Add a CLI test for the live wiring**

Append to `tests/test_cli.py`:
```python
def test_run_wires_live_config_and_exits_without_camera(tmp_path):
    """On the Mac picamera2 is missing: `run` must start everything, then exit 1 cleanly."""
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n[storage]\ndata_dir = "data"\n[web]\nport = 8791\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
    )
    proc = subprocess.run(
        [sys.executable, "-m", "birdy.cli", "--config", str(cfg), "run"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 1, proc.stderr
    assert "birdy starting" in proc.stderr
    assert "picamera2" in proc.stderr
```
(Skip it with the same `pytest.skip` guard as `test_classify_command_prints_prediction` if the model is missing.)

- [ ] **Step 4: Run everything**

Run: `.venv/bin/pytest -q` → all pass (expected ≈ 150). `.venv/bin/python -m pyflakes birdy tests` → no output.

- [ ] **Step 5: Commit**

```bash
git add birdy/capture.py birdy/cli.py tests/test_cli.py
git commit -m "Capture rebuilds tunables on settings change; CLI wires LiveConfig and watcher"
```

---

### Task 8: Docs, deploy, verify on the Pi

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-10-03-settings-page-design.md` (watcher note)

- [ ] **Step 1: Update `CLAUDE.md`**

In **Architecture**, add after the `web/app.py` bullet:
```markdown
- `live.py` — `LiveConfig` holds the current `Config` + a version. Tunables (`motion.*`, `classifier.threshold`, site/daylight, retention) are read through `live.get()` on use; the capture loop rebuilds its detector/gate when `live.version` changes. Camera/web/storage/model keys are read once at startup — changing them needs a restart.
- `settings.py` + `watcher.py` — the `/settings` page validates and atomically rewrites `birdy.toml` (comments are lost; `birdy.toml.example` keeps the docs) and applies it live; `ConfigWatcher` reloads the file when it's edited over SSH (invalid edits are logged and ignored until restart).
```
In **Commands**, replace the `/roi.png` note with: "Tune ROI/thresholds at `http://birdy.local:8000/settings` (live frame, click-to-draw polygon, applies without restart)."
In **Conventions**, add: "Adding a tunable key: `config.py` dataclass + validation, `birdy.toml.example`, `settings.py` (`SettingsPatch`, `config_to_dict`, `apply_patch`, `FIELD_KEYS`, `parse_settings_form`), `settings.html` form row, and the consumer must read it via `live.get()`."

In the spec, under `ConfigWatcher`, replace the content-hash sentence with: "After loading, the watcher compares the new `Config` to `live.get()`; equal configs (e.g. the page's own write) are not re-applied."

Commit: `git add CLAUDE.md docs/ && git commit -m "Document live config and settings page"`

- [ ] **Step 2: Deploy**

```bash
cd /Users/yassine/www/pi-birdy && deploy/deploy.sh
```
Expected: `active` and journal lines without errors. (`deploy.sh` runs `pip install -e .` on the Pi, which pulls `tomli-w` and `python-multipart` from PyPI — needs internet on the Pi.)

- [ ] **Step 3: Verify on the Pi**

1. `curl -s http://birdy.local:8000/settings | head -5` → HTML. Open it in a browser (in daylight to see the frame; at night the frame is black but the editor still works).
2. Change `min_area` to 999, Save → banner "Saved — applied live (v1)". `ssh yassine@birdy.local 'journalctl -u birdy -n 3 --no-pager -o cat'` shows `settings applied (v1)`. `ssh yassine@birdy.local 'grep min_area ~/pi-birdy/birdy.toml'` → `min_area = 999`.
3. SSH edit: `ssh yassine@birdy.local "sed -i 's/min_area = 999/min_area = 150/' ~/pi-birdy/birdy.toml"`; within 5 s the journal shows `settings reloaded ... (v2)` and `settings applied (v2)`; the page shows 150 on reload.
4. Invalid SSH edit: set `min_area = -1` the same way → journal shows `changed but is invalid ... keeping running settings`; page still shows 150. Restore.
5. Set lat/lon on the page (user-provided) → at night the state flips to `daylight_paused` within 60 s; the journal shows `outside daylight window`.
6. In daylight: draw the real ROI around the feeder, Save, watch `motion score` react.

- [ ] **Step 4: Finish**

Full suite green, pyflakes clean, service running on the new code → `superpowers:finishing-a-development-branch`.
