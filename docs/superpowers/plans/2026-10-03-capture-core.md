# Birdy Capture Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Pi 4 + Camera v2.1 behind a window photographs birds at a feeder during daylight, filters non-birds with a local TFLite model, stores photos + metadata in SQLite, and serves a LAN web page for review.

**Architecture:** One Python package `birdy` running as one systemd service. `capture` (the only module that imports `picamera2`) runs a frame-differencing loop on a low-res stream and, on a trigger, saves a full-res still and hands a `MotionEvent` to `pipeline`, which classifies it on a worker thread and writes to `store`. `web` reads `store` and a shared `RuntimeState`. All modules except `capture` are tested on the Mac.

**Tech Stack:** Python 3.11 (Homebrew `python3.11` on the Mac, system Python on Bookworm), `picamera2`, `opencv`, `numpy`, `ai-edge-litert` (TFLite interpreter, wheels for macOS arm64 + Linux aarch64), `astral`, `fastapi` + `uvicorn` + `jinja2`, `sqlite3`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-03-capture-core-design.md`

**Model dispatch hint:** tasks marked `[opus-ok]` are mechanical and can go to a cheaper/faster model; unmarked tasks need more care.

---

## File structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, `birdy` console script, pytest config |
| `birdy/config.py` | `birdy.toml` → frozen dataclasses with validation |
| `birdy/motion.py` | Pure numpy/OpenCV motion detector + `BBox`, `roi_bbox()` |
| `birdy/daylight.py` | `DaylightGate`: is it daytime, when does that change |
| `birdy/store.py` | SQLite schema, photo/thumb paths, queries, purge |
| `birdy/classifier.py` | `Prediction`, `Classifier` protocol, `ImageNetBirdClassifier` |
| `birdy/state.py` | `RuntimeState` shared between capture and web |
| `birdy/pipeline.py` | `MotionEvent`, queue + worker thread + backlog sweep |
| `birdy/retention.py` | Hourly purge + disk-space warning thread |
| `birdy/web/app.py`, `birdy/web/templates/index.html` | FastAPI app |
| `birdy/capture.py` | picamera2 loop; the only Pi-only module |
| `birdy/cli.py` | `birdy run|roi|classify|purge` |
| `models/download.sh`, `models/imagenet_bird_classes.txt`, `models/README.md` | Model assets |
| `deploy/install.sh`, `deploy/deploy.sh`, `deploy/birdy.service` | Pi install + deploy |
| `birdy.toml.example` | Documented config |
| `CLAUDE.md` | Repo guidance |
| `tests/…` | One test file per module |

---

### Task 1: Project scaffold `[opus-ok]`

**Files:**
- Create: `pyproject.toml`, `birdy/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

- [ ] **Step 1: Create the venv with Python 3.11**

Run:
```bash
cd /Users/yassine/www/pi-birdy
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
```
Expected: `.venv/bin/python --version` prints `Python 3.11.x`.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "birdy"
version = "0.1.0"
description = "Bird feeder camera for Raspberry Pi"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "ai-edge-litert>=2.0",
    "astral>=3.2",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "jinja2>=3.1",
]

[project.optional-dependencies]
dev = [
    "numpy",
    "opencv-python-headless>=4.8",
    "pytest>=8",
    "httpx>=0.27",
]

[project.scripts]
birdy = "birdy.cli:main"

[tool.setuptools.packages.find]
include = ["birdy*"]

[tool.setuptools.package-data]
birdy = ["web/templates/*.html"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

NumPy is deliberately unpinned: on the Pi (Trixie, NumPy 2.2 from apt) the venv uses `--system-site-packages`, and a pip-installed different NumPy would break the apt `cv2`.

- [ ] **Step 3: Create the package and a smoke test**

`birdy/__init__.py`:
```python
"""Birdy: bird feeder camera for Raspberry Pi."""

__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

`tests/test_smoke.py`:
```python
import birdy


def test_version():
    assert birdy.__version__ == "0.1.0"
```

- [ ] **Step 4: Install and run**

Run:
```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml birdy/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "Scaffold birdy package"
```

---

### Task 2: Config loader `[opus-ok]`

**Files:**
- Create: `birdy/config.py`, `tests/test_config.py`, `birdy.toml.example`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.config'`.

- [ ] **Step 3: Write `birdy/config.py`**

```python
"""Load and validate birdy.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Point = tuple[int, int]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SiteConfig:
    name: str
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class CameraConfig:
    main_size: tuple[int, int]
    lores_size: tuple[int, int]
    jpeg_quality: int


@dataclass(frozen=True)
class MotionConfig:
    roi: tuple[Point, ...]
    pixel_threshold: int
    min_area: int
    background_alpha: float
    cooldown_seconds: float


@dataclass(frozen=True)
class DaylightConfig:
    margin_minutes: int


@dataclass(frozen=True)
class ClassifierConfig:
    model: Path
    labels: Path
    bird_classes: Path
    threshold: float


@dataclass(frozen=True)
class RetentionConfig:
    non_bird_days: int


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int


@dataclass(frozen=True)
class Config:
    site: SiteConfig
    camera: CameraConfig
    motion: MotionConfig
    daylight: DaylightConfig
    classifier: ClassifierConfig
    retention: RetentionConfig
    storage: StorageConfig
    web: WebConfig


def _size(v: Any, name: str) -> tuple[int, int]:
    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(i, int) for i in v)):
        raise ConfigError(f"{name} must be [width, height]")
    return (v[0], v[1])


def _roi(v: Any) -> tuple[Point, ...]:
    if not isinstance(v, list) or len(v) < 3:
        raise ConfigError("motion.roi must be a list of at least 3 [x, y] points")
    pts = []
    for p in v:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(i, int) for i in p)):
            raise ConfigError("motion.roi points must be [x, y] integers")
        pts.append((p[0], p[1]))
    return tuple(pts)


def _path(base: Path, v: str) -> Path:
    p = Path(v)
    return p if p.is_absolute() else base / p


def load_config(path: Path) -> Config:
    path = Path(path)
    base = path.parent.resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    site = raw.get("site", {})
    if "name" not in site:
        raise ConfigError("site.name is required")
    lat, lon = site.get("lat"), site.get("lon")
    if (lat is None) != (lon is None):
        raise ConfigError("site.lat and site.lon must both be set or both omitted")

    cam = raw.get("camera", {})
    mot = raw.get("motion", {})
    if "roi" not in mot:
        raise ConfigError("motion.roi is required")
    day = raw.get("daylight", {})
    cls = raw.get("classifier", {})
    threshold = float(cls.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("classifier.threshold must be between 0 and 1")
    ret = raw.get("retention", {})
    sto = raw.get("storage", {})
    web = raw.get("web", {})

    return Config(
        site=SiteConfig(name=str(site["name"]), lat=None if lat is None else float(lat), lon=None if lon is None else float(lon)),
        camera=CameraConfig(
            main_size=_size(cam.get("main_size", [3280, 2464]), "camera.main_size"),
            lores_size=_size(cam.get("lores_size", [320, 240]), "camera.lores_size"),
            jpeg_quality=int(cam.get("jpeg_quality", 90)),
        ),
        motion=MotionConfig(
            roi=_roi(mot["roi"]),
            pixel_threshold=int(mot.get("pixel_threshold", 25)),
            min_area=int(mot.get("min_area", 150)),
            background_alpha=float(mot.get("background_alpha", 0.05)),
            cooldown_seconds=float(mot.get("cooldown_seconds", 3.0)),
        ),
        daylight=DaylightConfig(margin_minutes=int(day.get("margin_minutes", 30))),
        classifier=ClassifierConfig(
            model=_path(base, cls.get("model", "models/mobilenet_v2_1.0_224_quant.tflite")),
            labels=_path(base, cls.get("labels", "models/imagenet_labels.txt")),
            bird_classes=_path(base, cls.get("bird_classes", "models/imagenet_bird_classes.txt")),
            threshold=threshold,
        ),
        retention=RetentionConfig(non_bird_days=int(ret.get("non_bird_days", 7))),
        storage=StorageConfig(data_dir=_path(base, sto.get("data_dir", "data"))),
        web=WebConfig(host=str(web.get("host", "0.0.0.0")), port=int(web.get("port", 8000))),
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Write `birdy.toml.example`**

```toml
# Birdy configuration. Copy to birdy.toml and edit. Relative paths are
# resolved against the directory containing this file.

[site]
name = "home"
# Optional. Both or neither. With both set, capture pauses outside
# sunrise-margin .. sunset+margin. Without them, capture runs 24/7.
# lat = 38.72
# lon = -9.14

[camera]
main_size = [3280, 2464]   # full-res stills. Drop to [1640, 1232] if the
                           # camera fails to allocate buffers (CMA memory).
lores_size = [320, 240]    # analysis stream; ROI coordinates use this size
jpeg_quality = 90

[motion]
# Polygon around the feeder, in lores_size pixel coordinates.
# Run `birdy roi` to save a frame and pick the points.
roi = [[40, 60], [280, 60], [280, 220], [40, 220]]
pixel_threshold = 25       # per-pixel grey-level change to count as "changed"
min_area = 150             # changed pixels inside ROI needed to trigger
background_alpha = 0.05    # 0..1; higher = background adapts faster
cooldown_seconds = 3.0     # minimum gap between stills

[daylight]
margin_minutes = 30

[classifier]
model = "models/mobilenet_v2_1.0_224_quant.tflite"
labels = "models/imagenet_labels.txt"
bird_classes = "models/imagenet_bird_classes.txt"
threshold = 0.5            # P(bird) at or above this = bird

[retention]
non_bird_days = 7          # non-bird photos older than this are deleted hourly

[storage]
data_dir = "data"          # photos/, thumbs/, birdy.db live here

[web]
host = "0.0.0.0"
port = 8000
```

- [ ] **Step 6: Commit**

```bash
git add birdy/config.py tests/test_config.py birdy.toml.example
git commit -m "Add config loader"
```

---

### Task 3: Motion detector

**Files:**
- Create: `birdy/motion.py`, `tests/test_motion.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_motion.py`:
```python
import numpy as np

from birdy.motion import BBox, MotionDetector, roi_bbox

W, H = 320, 240
ROI = ((40, 60), (280, 60), (280, 220), (40, 220))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make_detector(clock, **overrides):
    kw = dict(frame_size=(W, H), roi=ROI, pixel_threshold=25, min_area=150, background_alpha=0.05, cooldown_seconds=3.0, clock=clock)
    kw.update(overrides)
    return MotionDetector(**kw)


def blank():
    return np.full((H, W), 50, dtype=np.uint8)


def with_blob(x, y, size=30):
    f = blank()
    f[y:y + size, x:x + size] = 255
    return f


def test_first_frame_initialises_background_without_trigger():
    d = make_detector(FakeClock())
    r = d.process(blank())
    assert r.score == 0 and r.triggered is False


def test_static_scene_never_triggers():
    d = make_detector(FakeClock())
    for _ in range(20):
        r = d.process(blank())
    assert r.score == 0 and r.triggered is False


def test_blob_inside_roi_triggers_with_score():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100))
    assert r.triggered is True
    assert 800 <= r.score <= 900  # 30x30 blob, blur softens edges


def test_blob_outside_roi_does_not_trigger():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(0, 0))
    assert r.score == 0 and r.triggered is False


def test_cooldown_blocks_second_trigger():
    clock = FakeClock()
    d = make_detector(clock)
    d.process(blank())
    assert d.process(with_blob(100, 100)).triggered
    clock.t = 1.0
    assert d.process(with_blob(150, 150)).triggered is False
    clock.t = 3.5
    assert d.process(with_blob(200, 100)).triggered is True


def test_stationary_blob_fades_into_background():
    clock = FakeClock()
    d = make_detector(clock)
    d.process(blank())
    for i in range(100):
        clock.t = i * 10.0
        r = d.process(with_blob(100, 100))
    assert r.score == 0


def test_small_blob_below_min_area_does_not_trigger():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100, size=5))
    assert 0 < r.score < 150 and r.triggered is False


def test_roi_bbox():
    assert roi_bbox(ROI) == BBox(x=40, y=60, w=240, h=160)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_motion.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.motion'`.

- [ ] **Step 3: Write `birdy/motion.py`**

```python
"""Frame-differencing motion detector. Pure numpy/OpenCV; no camera here."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

Point = tuple[int, int]


class BBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class MotionResult:
    score: int
    triggered: bool


def roi_bbox(roi: Sequence[Point]) -> BBox:
    xs = [p[0] for p in roi]
    ys = [p[1] for p in roi]
    return BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


class MotionDetector:
    """Compares each greyscale frame to a running-average background.

    `process()` must be called with frames of shape (h, w) uint8 where
    (w, h) == frame_size. The ROI polygon is in the same coordinates.
    """

    def __init__(
        self,
        frame_size: tuple[int, int],
        roi: Sequence[Point],
        pixel_threshold: int,
        min_area: int,
        background_alpha: float,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        w, h = frame_size
        self._mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(self._mask, [np.array(roi, dtype=np.int32)], 1)
        self._pixel_threshold = pixel_threshold
        self._min_area = min_area
        self._alpha = background_alpha
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._bg: np.ndarray | None = None
        self._last_trigger = float("-inf")

    def process(self, frame: np.ndarray) -> MotionResult:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32)
        if self._bg is None:
            self._bg = blurred
            return MotionResult(score=0, triggered=False)

        diff = np.abs(blurred - self._bg)
        changed = (diff > self._pixel_threshold).astype(np.uint8) * self._mask
        score = int(changed.sum())

        self._bg = (1.0 - self._alpha) * self._bg + self._alpha * blurred

        now = self._clock()
        triggered = score >= self._min_area and (now - self._last_trigger) >= self._cooldown
        if triggered:
            self._last_trigger = now
        return MotionResult(score=score, triggered=triggered)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_motion.py -q`
Expected: `8 passed`. If `test_blob_inside_roi_triggers_with_score` fails on the score bounds, print `r.score` and widen the bounds by ±50 — the exact count depends on OpenCV's blur border handling, not on correctness.

- [ ] **Step 5: Commit**

```bash
git add birdy/motion.py tests/test_motion.py
git commit -m "Add motion detector"
```

---

### Task 4: Daylight gate `[opus-ok]`

**Files:**
- Create: `birdy/daylight.py`, `tests/test_daylight.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_daylight.py`:
```python
from datetime import datetime, timezone

from birdy.daylight import DaylightGate

LISBON = dict(lat=38.72, lon=-9.14)


def utc(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_disabled_without_coordinates():
    g = DaylightGate(lat=None, lon=None, margin_minutes=30)
    assert g.enabled is False
    assert g.is_active(utc(2026, 6, 21, 2)) is True
    assert g.next_change(utc(2026, 6, 21, 2)) is None


def test_midday_is_active():
    g = DaylightGate(**LISBON, margin_minutes=30)
    assert g.is_active(utc(2026, 6, 21, 13)) is True


def test_night_is_inactive():
    g = DaylightGate(**LISBON, margin_minutes=30)
    assert g.is_active(utc(2026, 6, 21, 2)) is False
    assert g.is_active(utc(2026, 6, 21, 23)) is False


def test_margin_extends_window():
    # Lisbon sunrise on 2026-06-21 is ~05:12 UTC. 04:55 UTC is inside a 30-min margin, outside a 0-min one.
    assert DaylightGate(**LISBON, margin_minutes=30).is_active(utc(2026, 6, 21, 4, 55)) is True
    assert DaylightGate(**LISBON, margin_minutes=0).is_active(utc(2026, 6, 21, 4, 55)) is False


def test_next_change_before_sunrise_is_today_start():
    g = DaylightGate(**LISBON, margin_minutes=0)
    nxt = g.next_change(utc(2026, 6, 21, 2))
    assert nxt.date() == utc(2026, 6, 21, 2).date() and 4 <= nxt.hour <= 6


def test_next_change_during_day_is_today_end():
    g = DaylightGate(**LISBON, margin_minutes=0)
    nxt = g.next_change(utc(2026, 6, 21, 13))
    assert nxt.hour >= 19


def test_next_change_after_sunset_is_tomorrow_start():
    g = DaylightGate(**LISBON, margin_minutes=0)
    nxt = g.next_change(utc(2026, 6, 21, 23))
    assert nxt.day == 22 and 4 <= nxt.hour <= 6
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_daylight.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.daylight'`.

- [ ] **Step 3: Write `birdy/daylight.py`**

```python
"""Decide whether capture should run based on sunrise/sunset."""

from __future__ import annotations

from datetime import datetime, timedelta

from astral import Observer
from astral.sun import sun


class DaylightGate:
    def __init__(self, lat: float | None, lon: float | None, margin_minutes: int) -> None:
        self.enabled = lat is not None and lon is not None
        self._observer = Observer(latitude=lat, longitude=lon) if self.enabled else None
        self._margin = timedelta(minutes=margin_minutes)

    def _window(self, now: datetime, day_offset: int = 0) -> tuple[datetime, datetime]:
        assert self._observer is not None
        date = (now + timedelta(days=day_offset)).date()
        s = sun(self._observer, date=date, tzinfo=now.tzinfo)
        return s["sunrise"] - self._margin, s["sunset"] + self._margin

    def is_active(self, now: datetime) -> bool:
        """`now` must be timezone-aware."""
        if not self.enabled:
            return True
        start, end = self._window(now)
        return start <= now < end

    def next_change(self, now: datetime) -> datetime | None:
        """When the active/inactive state next flips, or None if the gate is disabled."""
        if not self.enabled:
            return None
        start, end = self._window(now)
        if now < start:
            return start
        if now < end:
            return end
        return self._window(now, day_offset=1)[0]
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_daylight.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/daylight.py tests/test_daylight.py
git commit -m "Add daylight gate"
```

---

### Task 5: Store `[opus-ok]`

**Files:**
- Create: `birdy/store.py`, `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:
```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def write_jpeg(path: Path, w=800, h=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), 120, dtype=np.uint8))


def add(store: Store, when: datetime, score=200) -> int:
    abs_path, rel_path = store.photo_path_for(when)
    write_jpeg(abs_path)
    return store.add_capture(when, rel_path, score)


# Local-time aware, because photo paths use local time.
T0 = datetime(2026, 10, 3, 10, 0, 0, 123000).astimezone()


def test_creates_dirs_and_db(store, tmp_path):
    assert (tmp_path / "data" / "birdy.db").exists()
    assert store.photos_dir.is_dir() and store.thumbs_dir.is_dir()


def test_photo_path_layout(store):
    abs_path, rel_path = store.photo_path_for(T0)
    assert rel_path == Path("2026/10/03/100000_123.jpg")
    assert abs_path == store.photos_dir / rel_path
    assert abs_path.parent.is_dir()


def test_add_capture_creates_thumbnail_and_pending_row(store):
    cid = add(store, T0)
    c = store.get(cid)
    assert c.status == "pending" and c.is_bird is None
    assert c.motion_score == 200
    thumb = store.thumbs_dir / c.thumb_path
    assert thumb.exists()
    assert cv2.imread(str(thumb)).shape[1] == 400


def test_update_classification(store):
    cid = add(store, T0)
    store.update_classification(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None)
    c = store.get(cid)
    assert c.status == "done" and c.is_bird is True and c.confidence == 0.9


def test_mark_error(store):
    cid = add(store, T0)
    store.mark_error(cid)
    assert store.get(cid).status == "error"


def test_list_filters_and_orders_newest_first(store):
    a = add(store, T0)
    b = add(store, T0 + timedelta(minutes=1))
    c = add(store, T0 + timedelta(minutes=2))
    store.update_classification(a, True, 0.9, None, None)
    store.update_classification(b, False, 0.1, None, None)
    assert [x.id for x in store.list_captures("all")] == [c, b, a]
    assert [x.id for x in store.list_captures("birds")] == [a]
    assert [x.id for x in store.list_captures("non-birds")] == [b]
    assert store.count("all") == 3 and store.count("birds") == 1


def test_list_pagination(store):
    ids = [add(store, T0 + timedelta(minutes=i)) for i in range(5)]
    page = store.list_captures("all", limit=2, offset=2)
    assert [x.id for x in page] == [ids[2], ids[1]]


def test_pending_lists_only_pending(store):
    a = add(store, T0)
    b = add(store, T0 + timedelta(minutes=1))
    store.update_classification(a, True, 0.9, None, None)
    assert [x.id for x in store.pending()] == [b]


def test_purge_deletes_only_old_non_birds(store):
    old = T0 - timedelta(days=10)
    old_bird = add(store, old)
    old_junk = add(store, old + timedelta(minutes=1))
    new_junk = add(store, T0)
    store.update_classification(old_bird, True, 0.9, None, None)
    store.update_classification(old_junk, False, 0.1, None, None)
    store.update_classification(new_junk, False, 0.1, None, None)
    junk = store.get(old_junk)

    deleted = store.purge_non_birds(older_than_days=7, now=T0)

    assert deleted == 1
    assert store.get(old_junk) is None
    assert not (store.photos_dir / junk.path).exists()
    assert not (store.thumbs_dir / junk.thumb_path).exists()
    assert store.get(old_bird) is not None and store.get(new_junk) is not None


def test_site_roundtrip(store):
    store.set_site("home", 38.7, -9.1)
    assert store.site() == {"name": "home", "lat": 38.7, "lon": -9.1}
    store.set_site("home2", None, None)
    assert store.site() == {"name": "home2", "lat": None, "lon": None}


def test_last_capture_at(store):
    assert store.last_capture_at() is None
    add(store, T0)
    assert store.last_capture_at() == T0


def test_disk_free_is_positive(store):
    assert store.disk_free_bytes() > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_store.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.store'`.

- [ ] **Step 3: Write `birdy/store.py`**

```python
"""SQLite + on-disk photo storage. One Store instance is shared by all threads."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

log = logging.getLogger(__name__)

THUMB_WIDTH = 400

SCHEMA = """
CREATE TABLE IF NOT EXISTS site (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  name TEXT NOT NULL,
  lat REAL,
  lon REAL
);
CREATE TABLE IF NOT EXISTS captures (
  id INTEGER PRIMARY KEY,
  captured_at TEXT NOT NULL,
  path TEXT NOT NULL,
  thumb_path TEXT,
  motion_score INTEGER NOT NULL,
  is_bird INTEGER,
  confidence REAL,
  species TEXT,
  species_confidence REAL,
  status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS captures_captured_at ON captures(captured_at DESC);
CREATE INDEX IF NOT EXISTS captures_is_bird ON captures(is_bird, captured_at DESC);
"""

FILTERS = {
    "all": "1 = 1",
    "birds": "is_bird = 1",
    "non-birds": "is_bird = 0",
}


@dataclass(frozen=True)
class Capture:
    id: int
    captured_at: datetime
    path: str
    thumb_path: str | None
    motion_score: int
    is_bird: bool | None
    confidence: float | None
    species: str | None
    species_confidence: float | None
    status: str


def _to_db_time(dt: datetime) -> str:
    """Store as UTC ISO 8601 so string comparison == chronological comparison."""
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _from_db_time(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _row_to_capture(row: sqlite3.Row) -> Capture:
    return Capture(
        id=row["id"],
        captured_at=_from_db_time(row["captured_at"]),
        path=row["path"],
        thumb_path=row["thumb_path"],
        motion_score=row["motion_score"],
        is_bird=None if row["is_bird"] is None else bool(row["is_bird"]),
        confidence=row["confidence"],
        species=row["species"],
        species_confidence=row["species_confidence"],
        status=row["status"],
    )


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.photos_dir = self.data_dir / "photos"
        self.thumbs_dir = self.data_dir / "thumbs"
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "birdy.db"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    # --- site -------------------------------------------------------------

    def set_site(self, name: str, lat: float | None, lon: float | None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO site (id, name, lat, lon) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name = excluded.name, lat = excluded.lat, lon = excluded.lon",
                (name, lat, lon),
            )

    def site(self) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT name, lat, lon FROM site WHERE id = 1").fetchone()
        return None if row is None else {"name": row["name"], "lat": row["lat"], "lon": row["lon"]}

    # --- captures ---------------------------------------------------------

    def photo_path_for(self, captured_at: datetime) -> tuple[Path, Path]:
        """Return (absolute path, path relative to photos_dir). Creates the day directory."""
        local = captured_at.astimezone()
        rel = Path(local.strftime("%Y/%m/%d")) / f"{local.strftime('%H%M%S')}_{local.microsecond // 1000:03d}.jpg"
        abs_path = self.photos_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        return abs_path, rel

    def _make_thumbnail(self, rel_path: Path) -> Path | None:
        img = cv2.imread(str(self.photos_dir / rel_path))
        if img is None:
            log.warning("could not read %s for thumbnail", rel_path)
            return None
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
        out = self.thumbs_dir / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return rel_path

    def add_capture(self, captured_at: datetime, rel_path: Path, motion_score: int) -> int:
        thumb = self._make_thumbnail(rel_path)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO captures (captured_at, path, thumb_path, motion_score, status) VALUES (?, ?, ?, ?, 'pending')",
                (_to_db_time(captured_at), str(rel_path), None if thumb is None else str(thumb), motion_score),
            )
            return int(cur.lastrowid)

    def update_classification(
        self, capture_id: int, is_bird: bool, confidence: float, species: str | None, species_confidence: float | None
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET is_bird = ?, confidence = ?, species = ?, species_confidence = ?, status = 'done' WHERE id = ?",
                (int(is_bird), confidence, species, species_confidence, capture_id),
            )

    def mark_error(self, capture_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET status = 'error' WHERE id = ?", (capture_id,))

    def get(self, capture_id: int) -> Capture | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
        return None if row is None else _row_to_capture(row)

    def list_captures(self, filter: str = "all", limit: int = 50, offset: int = 0) -> list[Capture]:
        where = FILTERS[filter]
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM captures WHERE {where} ORDER BY captured_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def count(self, filter: str = "all") -> int:
        where = FILTERS[filter]
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM captures WHERE {where}").fetchone()[0])

    def pending(self) -> list[Capture]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM captures WHERE status = 'pending' ORDER BY id").fetchall()
        return [_row_to_capture(r) for r in rows]

    def last_capture_at(self) -> datetime | None:
        with self._lock:
            row = self._conn.execute("SELECT MAX(captured_at) AS t FROM captures").fetchone()
        return None if row["t"] is None else _from_db_time(row["t"])

    # --- maintenance ------------------------------------------------------

    def purge_non_birds(self, older_than_days: int, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        cutoff = _to_db_time(now - timedelta(days=older_than_days))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, path, thumb_path FROM captures WHERE is_bird = 0 AND captured_at < ?", (cutoff,)
            ).fetchall()
        deleted = 0
        for row in rows:
            for p in (self.photos_dir / row["path"], self.thumbs_dir / row["thumb_path"] if row["thumb_path"] else None):
                if p is not None:
                    p.unlink(missing_ok=True)
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM captures WHERE id = ?", (row["id"],))
            deleted += 1
        return deleted

    def disk_free_bytes(self) -> int:
        return shutil.disk_usage(self.data_dir).free

    def close(self) -> None:
        with self._lock:
            self._conn.close()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_store.py -q`
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/store.py tests/test_store.py
git commit -m "Add SQLite store"
```

---

### Task 6: Model assets + classifier

**Files:**
- Create: `models/download.sh`, `models/imagenet_bird_classes.txt`, `models/README.md`, `birdy/classifier.py`, `tests/test_classifier.py`, `tests/fixtures/download.sh`

- [ ] **Step 1: Write `models/download.sh` and run it**

```bash
#!/usr/bin/env bash
# Downloads the MobileNetV2 ImageNet TFLite model and labels into this directory.
set -euo pipefail
cd "$(dirname "$0")"
tmp="$(mktemp -d)"
curl -sSL -o "$tmp/model.tgz" \
  https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar -xzf "$tmp/model.tgz" -C "$tmp"
cp "$tmp/mobilenet_v2_1.0_224_quant.tflite" .
curl -sSL -o imagenet_labels.txt \
  https://raw.githubusercontent.com/tensorflow/tensorflow/v2.15.0/tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt
rm -rf "$tmp"
echo "model: $(du -h mobilenet_v2_1.0_224_quant.tflite | cut -f1), labels: $(wc -l < imagenet_labels.txt) lines"
```

Run: `chmod +x models/download.sh && models/download.sh`
Expected: `model: 3.4M, labels: 1001 lines` (first label is `background`).

Then verify the model's tensor shapes:
```bash
.venv/bin/python -c "
from ai_edge_litert.interpreter import Interpreter
i = Interpreter(model_path='models/mobilenet_v2_1.0_224_quant.tflite'); i.allocate_tensors()
print(i.get_input_details()[0]['shape'], i.get_input_details()[0]['dtype'])
print(i.get_output_details()[0]['shape'], i.get_output_details()[0]['quantization'])"
```
Expected: `[  1 224 224   3] <class 'numpy.uint8'>` and `[   1 1001] (0.00390625, 0)`.

- [ ] **Step 2: Write `models/imagenet_bird_classes.txt`**

Output indices (1001-class model, index 0 = background, so ImageNet class n is at n+1) covering all ImageNet bird classes:

```
# Output indices of bird classes in the 1001-class MobileNet labels file
# (index 0 = background). Ranges are inclusive. ImageNet classes 7-24,
# 80-100 and 127-146 are birds; +1 for the background offset.
8-25
81-101
128-147
```

- [ ] **Step 3: Write `models/README.md`**

```markdown
# Models

`mobilenet_v2_1.0_224_quant.tflite` and `imagenet_labels.txt` are from the
TensorFlow hosted models (Apache 2.0). Run `./download.sh` to fetch them.
They are committed so the Pi needs no network at install time.

`imagenet_bird_classes.txt` lists which output indices are birds; the
classifier sums their probabilities to get P(bird).
```

- [ ] **Step 4: Download test fixtures**

`tests/fixtures/download.sh`:
```bash
#!/usr/bin/env bash
# Two public-domain/CC photos used only by the classifier integration test.
set -euo pipefail
cd "$(dirname "$0")"
UA="birdy-tests/0.1 (github.com; contact via repo)"
curl -sSL -A "$UA" -o bird.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/640px-Eopsaltria_australis_-_Mogo_Campground.jpg"
curl -sSL -A "$UA" -o cat.jpg \
  "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg"
ls -la bird.jpg cat.jpg
```

Run: `chmod +x tests/fixtures/download.sh && tests/fixtures/download.sh`
Expected: both files listed, each > 20 KB. If Wikimedia refuses, save any clear bird photo as `tests/fixtures/bird.jpg` and any cat photo as `tests/fixtures/cat.jpg` by hand; the integration test skips if they're missing.

- [ ] **Step 5: Write the failing tests**

`tests/test_classifier.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import ImageNetBirdClassifier, Prediction, bird_probability, crop_to_bbox, load_bird_indices
from birdy.motion import BBox

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite"
LABELS = ROOT / "models" / "imagenet_labels.txt"
BIRD_CLASSES = ROOT / "models" / "imagenet_bird_classes.txt"
FIXTURES = Path(__file__).parent / "fixtures"


def test_load_bird_indices_parses_ranges_and_comments(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("# comment\n3\n\n10-12\n")
    assert load_bird_indices(p) == frozenset({3, 10, 11, 12})


def test_load_bird_indices_real_file():
    idx = load_bird_indices(BIRD_CLASSES)
    assert 8 in idx and 25 in idx and 147 in idx
    assert 7 not in idx and 26 not in idx
    assert len(idx) == 18 + 21 + 20


def test_bird_probability_sums_bird_indices():
    probs = np.zeros(10, dtype=np.float32)
    probs[2] = 0.3
    probs[5] = 0.4
    probs[9] = 0.3
    assert bird_probability(probs, frozenset({2, 5})) == pytest.approx(0.7)


def test_crop_to_bbox_scales_from_lores_to_main_and_pads():
    img = np.zeros((2464, 3280, 3), dtype=np.uint8)
    # bbox in 320x240 coords; scale is 10.25 x ~10.27
    out = crop_to_bbox(img, BBox(x=40, y=60, w=240, h=160), lores_size=(320, 240), pad=0.1)
    h, w = out.shape[:2]
    assert 2800 <= w <= 3000 and 1900 <= h <= 2100


def test_crop_to_bbox_clamps_to_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    out = crop_to_bbox(img, BBox(x=0, y=0, w=320, h=240), lores_size=(320, 240), pad=0.5)
    assert out.shape[:2] == (240, 320)


@pytest.mark.skipif(not MODEL.exists(), reason="model not downloaded")
class TestRealModel:
    @pytest.fixture(scope="class")
    def clf(self):
        return ImageNetBirdClassifier(MODEL, LABELS, BIRD_CLASSES, threshold=0.5, lores_size=(320, 240))

    @pytest.mark.skipif(not (FIXTURES / "bird.jpg").exists(), reason="fixture missing")
    def test_bird_is_bird(self, clf):
        p = clf.classify(FIXTURES / "bird.jpg", roi_bbox=None)
        assert isinstance(p, Prediction)
        assert p.is_bird and p.confidence > 0.5

    @pytest.mark.skipif(not (FIXTURES / "cat.jpg").exists(), reason="fixture missing")
    def test_cat_is_not_bird(self, clf):
        p = clf.classify(FIXTURES / "cat.jpg", roi_bbox=None)
        assert not p.is_bird and p.confidence < 0.2

    @pytest.mark.skipif(not (FIXTURES / "bird.jpg").exists(), reason="fixture missing")
    def test_top_k_returns_labels(self, clf):
        top = clf.top_k(FIXTURES / "bird.jpg", roi_bbox=None, k=3)
        assert len(top) == 3
        assert all(isinstance(label, str) and 0 <= prob <= 1 for label, prob in top)

    def test_grey_image_is_not_bird(self, clf, tmp_path):
        p = tmp_path / "grey.jpg"
        cv2.imwrite(str(p), np.full((480, 640, 3), 128, dtype=np.uint8))
        assert clf.classify(p, roi_bbox=None).is_bird is False

    def test_unreadable_image_raises(self, clf, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not a jpeg")
        with pytest.raises(ValueError):
            clf.classify(p, roi_bbox=None)
```

- [ ] **Step 6: Run to verify failure**

Run: `.venv/bin/pytest tests/test_classifier.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.classifier'`.

- [ ] **Step 7: Write `birdy/classifier.py`**

```python
"""Bird / not-bird classification with a TFLite ImageNet model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from birdy.motion import BBox


@dataclass(frozen=True)
class Prediction:
    is_bird: bool
    confidence: float  # P(bird), 0..1
    species: str | None = None
    species_confidence: float | None = None


class Classifier(Protocol):
    def classify(self, image_path: Path, roi_bbox: BBox | None) -> Prediction: ...


def _load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # older Pi installs
        from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path=str(model_path))


def load_bird_indices(path: Path) -> frozenset[int]:
    out: set[int] = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "-" in line:
            a, b = line.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(line))
    return frozenset(out)


def bird_probability(probs: np.ndarray, bird_indices: frozenset[int]) -> float:
    return float(sum(probs[i] for i in bird_indices if i < len(probs)))


def crop_to_bbox(img: np.ndarray, bbox: BBox, lores_size: tuple[int, int], pad: float) -> np.ndarray:
    """Crop `img` (main-stream size) to `bbox` given in lores coordinates, padded by `pad` fraction."""
    img_h, img_w = img.shape[:2]
    sx = img_w / lores_size[0]
    sy = img_h / lores_size[1]
    x0 = (bbox.x - bbox.w * pad) * sx
    y0 = (bbox.y - bbox.h * pad) * sy
    x1 = (bbox.x + bbox.w * (1 + pad)) * sx
    y1 = (bbox.y + bbox.h * (1 + pad)) * sy
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(img_w, int(round(x1))), min(img_h, int(round(y1)))
    return img[y0:y1, x0:x1]


class ImageNetBirdClassifier:
    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        bird_classes_path: Path,
        threshold: float,
        lores_size: tuple[int, int],
        pad: float = 0.1,
    ) -> None:
        self._interp = _load_interpreter(model_path)
        self._interp.allocate_tensors()
        self._input = self._interp.get_input_details()[0]
        self._output = self._interp.get_output_details()[0]
        _, self._in_h, self._in_w, _ = self._input["shape"]
        self._labels = Path(labels_path).read_text().splitlines() if Path(labels_path).exists() else []
        self._bird_indices = load_bird_indices(bird_classes_path)
        self._threshold = threshold
        self._lores_size = lores_size
        self._pad = pad

    def _probs(self, image_path: Path, roi_bbox: BBox | None) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image {image_path}")
        if roi_bbox is not None:
            img = crop_to_bbox(img, roi_bbox, self._lores_size, self._pad)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._in_w, self._in_h), interpolation=cv2.INTER_AREA)

        if self._input["dtype"] == np.uint8:
            tensor = img[np.newaxis].astype(np.uint8)
        else:
            tensor = (img[np.newaxis].astype(np.float32) / 127.5) - 1.0
        self._interp.set_tensor(self._input["index"], tensor)
        self._interp.invoke()
        out = self._interp.get_tensor(self._output["index"])[0]

        scale, zero_point = self._output["quantization"]
        if scale:
            return (out.astype(np.float32) - zero_point) * scale
        return out.astype(np.float32)

    def classify(self, image_path: Path, roi_bbox: BBox | None) -> Prediction:
        conf = bird_probability(self._probs(image_path, roi_bbox), self._bird_indices)
        return Prediction(is_bird=conf >= self._threshold, confidence=conf)

    def top_k(self, image_path: Path, roi_bbox: BBox | None, k: int = 5) -> list[tuple[str, float]]:
        probs = self._probs(image_path, roi_bbox)
        order = np.argsort(probs)[::-1][:k]
        return [(self._labels[i] if i < len(self._labels) else str(i), float(probs[i])) for i in order]
```

- [ ] **Step 8: Run tests**

Run: `.venv/bin/pytest tests/test_classifier.py -q`
Expected: `10 passed`. If `test_cat_is_not_bird` fails with confidence between 0.2 and 0.5, loosen it to `< 0.5` — it's the threshold that matters.

- [ ] **Step 9: Commit**

```bash
git add models/ birdy/classifier.py tests/test_classifier.py tests/fixtures/
git commit -m "Add TFLite bird classifier and model assets"
```

---

### Task 7: Runtime state + pipeline `[opus-ok]`

**Files:**
- Create: `birdy/state.py`, `birdy/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write `birdy/state.py`** (no logic worth a unit test; it's exercised by the web tests)

```python
"""Mutable runtime status shared between the capture loop and the web app."""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "starting"  # starting | capturing | daylight_paused | camera_error
        self.motion_score = 0
        self.last_capture_at: datetime | None = None
        self.latest_lores: np.ndarray | None = None  # greyscale (h, w) uint8
        self.started_at = time.monotonic()

    def update(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                if not hasattr(self, k):
                    raise AttributeError(k)
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "motion_score": self.motion_score,
                "last_capture_at": None if self.last_capture_at is None else self.last_capture_at.isoformat(),
                "uptime_seconds": int(time.monotonic() - self.started_at),
            }

    def lores_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self.latest_lores is None else self.latest_lores.copy()
```

- [ ] **Step 2: Write the failing pipeline tests**

`tests/test_pipeline.py`:
```python
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import Prediction
from birdy.motion import BBox
from birdy.pipeline import MotionEvent, Pipeline
from birdy.store import Store

# Local-time aware: the fail_on filename below depends on local HHMMSS.
T0 = datetime(2026, 10, 3, 10, 0).astimezone()


class FakeClassifier:
    def __init__(self, fail_on: set[str] = ()):
        self.fail_on = set(fail_on)
        self.calls: list[Path] = []

    def classify(self, image_path, roi_bbox):
        self.calls.append(image_path)
        if image_path.name in self.fail_on:
            raise RuntimeError("boom")
        return Prediction(is_bird=True, confidence=0.9)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def add(store, i):
    when = T0.replace(minute=i)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    return MotionEvent(capture_id=cid, path=abs_path, captured_at=when, motion_score=100)


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_worker_classifies_submitted_event(store):
    clf = FakeClassifier()
    p = Pipeline(store, clf, roi_bbox=BBox(0, 0, 10, 10))
    p.start()
    try:
        ev = add(store, 0)
        assert p.submit(ev) is True
        assert wait_until(lambda: store.get(ev.capture_id).status == "done")
        c = store.get(ev.capture_id)
        assert c.is_bird is True and c.confidence == 0.9
        assert clf.calls == [ev.path]
    finally:
        p.stop()


def test_classifier_error_marks_row_error_and_continues(store):
    p = Pipeline(store, FakeClassifier(fail_on={"100000_000.jpg"}), roi_bbox=None)
    p.start()
    try:
        bad = add(store, 0)
        good = add(store, 1)
        p.submit(bad)
        p.submit(good)
        assert wait_until(lambda: store.get(good.capture_id).status == "done")
        assert store.get(bad.capture_id).status == "error"
    finally:
        p.stop()


def test_submit_returns_false_when_full(store):
    p = Pipeline(store, FakeClassifier(), roi_bbox=None, maxsize=1)
    # not started: nothing drains the queue
    assert p.submit(add(store, 0)) is True
    assert p.submit(add(store, 1)) is False
    assert p.queue_depth() == 1


def test_sweep_classifies_pending_rows(store):
    p = Pipeline(store, FakeClassifier(), roi_bbox=None)
    add(store, 0)
    add(store, 1)
    assert len(store.pending()) == 2
    p.sweep()
    assert store.pending() == []
    assert store.count("birds") == 2


def test_periodic_sweep_runs(store):
    p = Pipeline(store, FakeClassifier(), roi_bbox=None, sweep_interval=0.1)
    add(store, 0)
    p.start()
    try:
        assert wait_until(lambda: store.pending() == [])
    finally:
        p.stop()
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.pipeline'`.

- [ ] **Step 4: Write `birdy/pipeline.py`**

```python
"""Queue between capture and classification, plus a backlog sweep."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from birdy.classifier import Classifier
from birdy.motion import BBox
from birdy.store import Store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionEvent:
    capture_id: int
    path: Path
    captured_at: datetime
    motion_score: int


class Pipeline:
    def __init__(
        self,
        store: Store,
        classifier: Classifier,
        roi_bbox: BBox | None,
        maxsize: int = 50,
        sweep_interval: float = 300.0,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._roi_bbox = roi_bbox
        self._queue: queue.Queue[MotionEvent] = queue.Queue(maxsize=maxsize)
        self._sweep_interval = sweep_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def submit(self, event: MotionEvent) -> bool:
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            log.warning("pipeline queue full; capture %d left pending", event.capture_id)
            return False

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def process(self, capture_id: int, path: Path) -> None:
        try:
            pred = self._classifier.classify(path, self._roi_bbox)
        except Exception:
            log.exception("classification failed for %s", path)
            self._store.mark_error(capture_id)
            return
        self._store.update_classification(capture_id, pred.is_bird, pred.confidence, pred.species, pred.species_confidence)
        log.info("capture %d: bird=%s p=%.2f", capture_id, pred.is_bird, pred.confidence)

    def sweep(self) -> None:
        for c in self._store.pending():
            self.process(c.id, self._store.photos_dir / c.path)

    def _run(self) -> None:
        next_sweep = time.monotonic() + self._sweep_interval
        while not self._stop.is_set():
            try:
                ev = self._queue.get(timeout=0.5)
            except queue.Empty:
                ev = None
            if ev is not None:
                self.process(ev.capture_id, ev.path)
            if time.monotonic() >= next_sweep:
                try:
                    self.sweep()
                except Exception:
                    log.exception("sweep failed")
                next_sweep = time.monotonic() + self._sweep_interval
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add birdy/state.py birdy/pipeline.py tests/test_pipeline.py
git commit -m "Add runtime state and classification pipeline"
```

---

### Task 8: Retention job `[opus-ok]`

**Files:**
- Create: `birdy/retention.py`, `tests/test_retention.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_retention.py`:
```python
import logging
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from birdy.retention import RetentionJob, run_once
from birdy.store import Store

T0 = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def add_non_bird(store, when):
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, False, 0.1, None, None)
    return cid


def test_run_once_purges_and_logs(store, caplog):
    old = add_non_bird(store, T0 - timedelta(days=10))
    new = add_non_bird(store, T0)
    with caplog.at_level(logging.INFO):
        deleted = run_once(store, non_bird_days=7, now=T0)
    assert deleted == 1
    assert store.get(old) is None and store.get(new) is not None
    assert "disk free" in caplog.text


def test_low_disk_warns(store, caplog, monkeypatch):
    monkeypatch.setattr(store, "disk_free_bytes", lambda: 500 * 1024 * 1024)
    with caplog.at_level(logging.WARNING):
        run_once(store, non_bird_days=7, now=T0)
    assert "low disk" in caplog.text


def test_job_thread_runs_periodically(store):
    add_non_bird(store, T0 - timedelta(days=10))
    job = RetentionJob(store, non_bird_days=7, interval=0.1)
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

Run: `.venv/bin/pytest tests/test_retention.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.retention'`.

- [ ] **Step 3: Write `birdy/retention.py`**

```python
"""Hourly deletion of old non-bird captures and a disk-space check."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from birdy.store import Store

log = logging.getLogger(__name__)

LOW_DISK_BYTES = 1024 * 1024 * 1024  # 1 GB


def run_once(store: Store, non_bird_days: int, now: datetime | None = None) -> int:
    deleted = store.purge_non_birds(older_than_days=non_bird_days, now=now)
    free = store.disk_free_bytes()
    log.info("retention: deleted %d non-bird captures; disk free %.1f GB", deleted, free / 1e9)
    if free < LOW_DISK_BYTES:
        log.warning("low disk: %.0f MB free", free / 1e6)
    return deleted


class RetentionJob:
    def __init__(self, store: Store, non_bird_days: int, interval: float = 3600.0) -> None:
        self._store = store
        self._days = non_bird_days
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="retention", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                run_once(self._store, self._days)
            except Exception:
                log.exception("retention run failed")
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_retention.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/retention.py tests/test_retention.py
git commit -m "Add retention job"
```

---

### Task 9: Web app `[opus-ok]`

**Files:**
- Create: `birdy/web/__init__.py`, `birdy/web/app.py`, `birdy/web/templates/index.html`, `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_web.py`:
```python
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from birdy.state import RuntimeState
from birdy.store import Store
from birdy.web.app import create_app

T0 = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)
ROI = ((40, 60), (280, 60), (280, 220), (40, 220))


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


@pytest.fixture
def state():
    s = RuntimeState()
    s.update(state="capturing", motion_score=42)
    return s


@pytest.fixture
def client(store, state):
    return TestClient(create_app(store, state, roi=ROI))


def add(store, i, is_bird):
    when = T0 + timedelta(minutes=i)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.full((60, 80, 3), 90, np.uint8))
    cid = store.add_capture(when, rel, 100 + i)
    store.update_classification(cid, is_bird, 0.9 if is_bird else 0.1, "Robin" if is_bird else None, 0.8 if is_bird else None)
    return store.get(cid)


def test_index_renders_captures(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/")
    assert r.status_code == 200
    assert f"/thumbs/{bird.thumb_path}" in r.text
    assert f"/photos/{bird.path}" in r.text
    assert "Robin" in r.text
    assert f"/thumbs/{junk.thumb_path}" in r.text


def test_index_filter_birds(client, store):
    bird = add(store, 0, True)
    junk = add(store, 1, False)
    r = client.get("/?filter=birds")
    assert bird.thumb_path in r.text and junk.thumb_path not in r.text


def test_index_bad_filter_is_400(client):
    assert client.get("/?filter=cats").status_code == 400


def test_index_pagination(client, store):
    for i in range(60):
        add(store, i, True)
    r = client.get("/?page=2")
    assert r.status_code == 200
    assert "page=1" in r.text  # link back


def test_status_json(client, store):
    add(store, 0, True)
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "capturing"
    assert body["motion_score"] == 42
    assert body["queue_depth"] == 0
    assert body["pending_count"] == 0
    assert body["disk_free_bytes"] > 0
    assert body["last_capture_at"] is not None
    assert body["uptime_seconds"] >= 0


def test_photos_and_thumbs_are_served(client, store):
    c = add(store, 0, True)
    assert client.get(f"/photos/{c.path}").status_code == 200
    assert client.get(f"/thumbs/{c.thumb_path}").status_code == 200


def test_roi_png_404_without_frame(client):
    assert client.get("/roi.png").status_code == 404


def test_roi_png_renders_frame(client, state):
    state.update(latest_lores=np.full((240, 320), 100, np.uint8))
    r = client.get("/roi.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.web'`.

- [ ] **Step 3: Write `birdy/web/__init__.py`** (empty) **and `birdy/web/app.py`**

```python
"""Read-only LAN web page over the store."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from birdy.state import RuntimeState
from birdy.store import FILTERS, Store

PAGE_SIZE = 50


def _local_time(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def create_app(
    store: Store,
    state: RuntimeState,
    roi: Sequence[tuple[int, int]],
    queue_depth: Callable[[], int] = lambda: 0,
) -> FastAPI:
    app = FastAPI(title="birdy")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    templates.env.filters["local_time"] = _local_time

    app.mount("/photos", StaticFiles(directory=str(store.photos_dir)), name="photos")
    app.mount("/thumbs", StaticFiles(directory=str(store.thumbs_dir)), name="thumbs")

    def status_payload() -> dict:
        snap = state.snapshot()
        last = store.last_capture_at()
        snap.update(
            queue_depth=queue_depth(),
            pending_count=len(store.pending()),
            disk_free_bytes=store.disk_free_bytes(),
            last_capture_at=None if last is None else last.isoformat(),
        )
        return snap

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, filter: str = "all", page: int = 1):
        if filter not in FILTERS:
            raise HTTPException(status_code=400, detail="filter must be one of all, birds, non-birds")
        page = max(1, page)
        total = store.count(filter)
        captures = store.list_captures(filter, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
        pages = max(1, -(-total // PAGE_SIZE))
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "captures": captures,
                "filter": filter,
                "page": page,
                "pages": pages,
                "total": total,
                "status": status_payload(),
            },
        )

    @app.get("/status")
    def status():
        return status_payload()

    @app.get("/roi.png")
    def roi_png():
        frame = state.lores_frame()
        if frame is None:
            raise HTTPException(status_code=404, detail="no frame yet")
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=png.tobytes(), media_type="image/png")

    return app
```

- [ ] **Step 4: Write `birdy/web/templates/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>birdy</title>
<style>
  :root { --bg: #fafaf7; --fg: #222; --muted: #777; --card: #fff; --line: #ddd; --bird: #2a7; --junk: #c66; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #eee; --muted: #999; --card: #222; --line: #333; } }
  body { margin: 0; padding: 16px; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: baseline; margin-bottom: 16px; }
  h1 { font-size: 20px; margin: 0; }
  .status span { margin-right: 16px; color: var(--muted); }
  .status b { color: var(--fg); }
  nav a { margin-right: 12px; } nav a.active { font-weight: bold; text-decoration: none; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .tile { background: var(--card); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .tile img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; }
  .meta { padding: 6px 8px; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; }
  .tag { font-weight: bold; } .tag.bird { color: var(--bird); } .tag.junk { color: var(--junk); }
  .pager { margin-top: 16px; } .pager a, .pager span { margin-right: 12px; }
</style>
</head>
<body>
<header>
  <h1>birdy</h1>
  <div class="status">
    <span>state <b>{{ status.state }}</b></span>
    <span>motion <b>{{ status.motion_score }}</b></span>
    <span>pending <b>{{ status.pending_count }}</b></span>
    <span>disk free <b>{{ (status.disk_free_bytes / 1e9) | round(1) }} GB</b></span>
    <span>last capture <b>{{ status.last_capture_at or "–" }}</b></span>
    <span><a href="/roi.png">roi</a></span>
  </div>
</header>

<nav>
  {% for f in ["all", "birds", "non-birds"] %}
    <a href="/?filter={{ f }}" class="{{ 'active' if f == filter else '' }}">{{ f }}</a>
  {% endfor %}
  <span>({{ total }})</span>
</nav>

<div class="grid">
  {% for c in captures %}
  <a class="tile" href="/photos/{{ c.path }}">
    {% if c.thumb_path %}<img src="/thumbs/{{ c.thumb_path }}" alt="" loading="lazy">{% endif %}
    <div class="meta">
      <span>{{ c.captured_at | local_time }}</span>
      <span>
        {% if c.is_bird is none %}<span class="tag">{{ c.status }}</span>
        {% elif c.is_bird %}<span class="tag bird">bird {{ (c.confidence * 100) | round(0) | int }}%</span>
        {% else %}<span class="tag junk">no {{ (c.confidence * 100) | round(0) | int }}%</span>{% endif %}
      </span>
    </div>
    <div class="meta">
      <span>motion {{ c.motion_score }}</span>
      {% if c.species %}<span>{{ c.species }}</span>{% endif %}
    </div>
  </a>
  {% else %}
  <p>No captures yet.</p>
  {% endfor %}
</div>

<div class="pager">
  {% if page > 1 %}<a href="/?filter={{ filter }}&page={{ page - 1 }}">← page {{ page - 1 }}</a>{% endif %}
  <span>page {{ page }} / {{ pages }}</span>
  {% if page < pages %}<a href="/?filter={{ filter }}&page={{ page + 1 }}">page {{ page + 1 }} →</a>{% endif %}
</div>
</body>
</html>
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/test_web.py -q`
Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add birdy/web/
git add tests/test_web.py
git commit -m "Add LAN web page"
```

---

### Task 10: Capture loop (Pi-only)

**Files:**
- Create: `birdy/capture.py`

No Mac tests — `picamera2` isn't importable here. Verified on the Pi in Task 13. Keep it thin.

- [ ] **Step 1: Write `birdy/capture.py`**

```python
"""Camera loop: lores motion detection, full-res still on trigger. Pi only."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from birdy.config import Config
from birdy.daylight import DaylightGate
from birdy.motion import MotionDetector
from birdy.pipeline import MotionEvent, Pipeline
from birdy.state import RuntimeState
from birdy.store import Store

log = logging.getLogger(__name__)

FRAME_INTERVAL = 0.1  # ~10 fps analysis
GATE_RECHECK = 60.0


@contextmanager
def camera_session(config: Config):
    from picamera2 import Picamera2  # Pi only

    picam2 = Picamera2()
    cam_cfg = picam2.create_preview_configuration(
        main={"size": tuple(config.camera.main_size), "format": "RGB888"},
        lores={"size": tuple(config.camera.lores_size), "format": "YUV420"},
        buffer_count=2,
    )
    picam2.configure(cam_cfg)
    picam2.options["quality"] = config.camera.jpeg_quality
    picam2.start()
    time.sleep(1.0)  # let AE/AWB settle
    try:
        yield picam2
    finally:
        picam2.stop()
        picam2.close()


def lores_grey(picam2, lores_size: tuple[int, int]):
    """Y plane of the lores stream as an (h, w) uint8 array, stride-safe."""
    w, h = lores_size
    stride = picam2.stream_configuration("lores")["stride"]
    buf = picam2.capture_buffer("lores")
    # ascontiguousarray: the [:, :w] slice is strided and OpenCV wants a contiguous input.
    return np.ascontiguousarray(buf[: stride * h].reshape(h, stride)[:, :w])


def grab_lores_frame(config: Config):
    """One-shot helper for `birdy roi`."""
    with camera_session(config) as picam2:
        return lores_grey(picam2, config.camera.lores_size)


def _capture_still(picam2, store: Store, pipeline: Pipeline, state: RuntimeState, score: int) -> None:
    now = datetime.now().astimezone()
    abs_path, rel_path = store.photo_path_for(now)
    picam2.capture_file(str(abs_path))
    capture_id = store.add_capture(now, rel_path, score)
    state.update(last_capture_at=now)
    pipeline.submit(MotionEvent(capture_id=capture_id, path=abs_path, captured_at=now, motion_score=score))
    log.info("captured %s (motion %d)", rel_path, score)


def run_capture(
    config: Config,
    store: Store,
    pipeline: Pipeline,
    state: RuntimeState,
    gate: DaylightGate,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        if not gate.is_active(datetime.now().astimezone()):
            state.update(state="daylight_paused", motion_score=0)
            log.info("outside daylight window; next change %s", gate.next_change(datetime.now().astimezone()))
            stop.wait(GATE_RECHECK)
            continue

        detector = MotionDetector(
            frame_size=config.camera.lores_size,
            roi=config.motion.roi,
            pixel_threshold=config.motion.pixel_threshold,
            min_area=config.motion.min_area,
            background_alpha=config.motion.background_alpha,
            cooldown_seconds=config.motion.cooldown_seconds,
        )
        try:
            with camera_session(config) as picam2:
                state.update(state="capturing")
                last_gate_check = time.monotonic()
                while not stop.is_set():
                    if time.monotonic() - last_gate_check > GATE_RECHECK:
                        last_gate_check = time.monotonic()
                        if not gate.is_active(datetime.now().astimezone()):
                            break
                    t0 = time.monotonic()
                    try:
                        frame = lores_grey(picam2, config.camera.lores_size)
                        result = detector.process(frame)
                        state.update(motion_score=result.score, latest_lores=frame.copy())
                        if result.triggered:
                            _capture_still(picam2, store, pipeline, state, result.score)
                    except Exception:
                        log.exception("frame processing failed; continuing")
                    elapsed = time.monotonic() - t0
                    if elapsed < FRAME_INTERVAL:
                        stop.wait(FRAME_INTERVAL - elapsed)
        except Exception:
            state.update(state="camera_error")
            log.exception("camera session failed")
            raise
```

- [ ] **Step 2: Byte-compile to catch syntax errors**

Run: `.venv/bin/python -m py_compile birdy/capture.py && echo ok`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add birdy/capture.py
git commit -m "Add picamera2 capture loop"
```

---

### Task 11: CLI and service wiring

**Files:**
- Create: `birdy/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
import subprocess
import sys
from pathlib import Path

import pytest

from birdy.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def test_parser_subcommands():
    p = build_parser()
    assert p.parse_args(["run"]).command == "run"
    assert p.parse_args(["roi", "--out", "x.png"]).out == "x.png"
    assert p.parse_args(["classify", "a.jpg"]).image == "a.jpg"
    assert p.parse_args(["purge"]).command == "purge"


def test_default_config_path():
    assert build_parser().parse_args(["run"]).config == "birdy.toml"


def test_classify_command_prints_prediction(tmp_path):
    bird = FIXTURES / "bird.jpg"
    if not bird.exists() or not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("fixtures/model missing")
    cfg = tmp_path / "birdy.toml"
    cfg.write_text(
        '[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10],[0,10]]\n'
        f'[classifier]\nmodel = "{ROOT}/models/mobilenet_v2_1.0_224_quant.tflite"\n'
        f'labels = "{ROOT}/models/imagenet_labels.txt"\nbird_classes = "{ROOT}/models/imagenet_bird_classes.txt"\n'
    )
    out = subprocess.run(
        [sys.executable, "-m", "birdy.cli", "--config", str(cfg), "classify", str(bird)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "bird=True" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'birdy.cli'`.

- [ ] **Step 3: Write `birdy/cli.py`**

```python
"""Command line entry point: birdy run | roi | classify | purge."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from birdy.classifier import ImageNetBirdClassifier
from birdy.config import Config, load_config
from birdy.motion import roi_bbox
from birdy.store import Store

log = logging.getLogger("birdy")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="birdy")
    p.add_argument("--config", default="birdy.toml", help="path to birdy.toml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run the capture service")
    roi = sub.add_parser("roi", help="save one lores frame (with ROI drawn) for picking coordinates")
    roi.add_argument("--out", default="roi.png")
    cls = sub.add_parser("classify", help="classify a single image file")
    cls.add_argument("image")
    sub.add_parser("purge", help="run the retention purge once")
    return p


def _classifier(config: Config) -> ImageNetBirdClassifier:
    return ImageNetBirdClassifier(
        config.classifier.model,
        config.classifier.labels,
        config.classifier.bird_classes,
        threshold=config.classifier.threshold,
        lores_size=config.camera.lores_size,
    )


def cmd_run(config: Config) -> int:
    import uvicorn

    from birdy.capture import run_capture
    from birdy.daylight import DaylightGate
    from birdy.pipeline import Pipeline
    from birdy.retention import RetentionJob
    from birdy.state import RuntimeState
    from birdy.web.app import create_app

    store = Store(config.storage.data_dir)
    store.set_site(config.site.name, config.site.lat, config.site.lon)
    state = RuntimeState()

    pipeline = Pipeline(store, _classifier(config), roi_bbox(config.motion.roi))
    pipeline.start()
    retention = RetentionJob(store, config.retention.non_bird_days)
    retention.start()

    app = create_app(store, state, config.motion.roi, queue_depth=pipeline.queue_depth)
    server = uvicorn.Server(uvicorn.Config(app, host=config.web.host, port=config.web.port, log_level="warning"))
    threading.Thread(target=server.run, name="web", daemon=True).start()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    gate = DaylightGate(config.site.lat, config.site.lon, config.daylight.margin_minutes)
    log.info("birdy starting: site=%s daylight_gate=%s web=%s:%d", config.site.name, gate.enabled, config.web.host, config.web.port)
    try:
        run_capture(config, store, pipeline, state, gate, stop)
        return 0
    except Exception:
        return 1
    finally:
        server.should_exit = True
        pipeline.stop()
        retention.stop()
        store.close()


def cmd_roi(config: Config, out: str) -> int:
    import cv2
    import numpy as np

    from birdy.capture import grab_lores_frame

    frame = grab_lores_frame(config)
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    pts = np.array(config.motion.roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
    cv2.imwrite(out, img)
    print(f"saved {out} ({img.shape[1]}x{img.shape[0]}); current roi={list(map(list, config.motion.roi))}")
    return 0


def cmd_classify(config: Config, image: str) -> int:
    clf = _classifier(config)
    pred = clf.classify(Path(image), roi_bbox=None)
    print(f"bird={pred.is_bird} p_bird={pred.confidence:.3f}")
    for label, prob in clf.top_k(Path(image), roi_bbox=None, k=5):
        print(f"  {prob:.3f}  {label}")
    return 0


def cmd_purge(config: Config) -> int:
    from birdy.retention import run_once

    store = Store(config.storage.data_dir)
    deleted = run_once(store, config.retention.non_bird_days)
    print(f"deleted {deleted}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    if args.command == "run":
        return cmd_run(config)
    if args.command == "roi":
        return cmd_roi(config, args.out)
    if args.command == "classify":
        return cmd_classify(config, args.image)
    if args.command == "purge":
        return cmd_purge(config)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Run the service locally with a fake camera to check wiring**

Create a throwaway config in the scratchpad and run the web + pipeline without the camera:

```bash
cat > /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/birdy.toml <<'EOF'
[site]
name = "mac"
[motion]
roi = [[40, 60], [280, 60], [280, 220], [40, 220]]
[classifier]
model = "/Users/yassine/www/pi-birdy/models/mobilenet_v2_1.0_224_quant.tflite"
labels = "/Users/yassine/www/pi-birdy/models/imagenet_labels.txt"
bird_classes = "/Users/yassine/www/pi-birdy/models/imagenet_bird_classes.txt"
[storage]
data_dir = "data"
EOF
.venv/bin/birdy --config /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/birdy.toml run
```
Expected: logs `birdy starting`, then a traceback `ModuleNotFoundError: No module named 'picamera2'` and exit code 1 — proving everything before the camera wires up. (Ctrl-C if it hangs; it shouldn't.)

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: all passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add birdy/cli.py tests/test_cli.py
git commit -m "Add CLI and service wiring"
```

---

### Task 12: Deploy scripts, service unit, CLAUDE.md `[opus-ok]`

**Files:**
- Create: `deploy/install.sh`, `deploy/deploy.sh`, `deploy/birdy.service`, `CLAUDE.md`

- [ ] **Step 1: Write `deploy/birdy.service`**

```ini
[Unit]
Description=Birdy bird feeder camera
After=network-online.target

[Service]
User=yassine
WorkingDirectory=/home/yassine/pi-birdy
ExecStart=/home/yassine/pi-birdy/.venv/bin/birdy --config /home/yassine/pi-birdy/birdy.toml run
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write `deploy/install.sh`** (run once, on the Pi)

```bash
#!/usr/bin/env bash
# One-time setup on the Pi. Run from the repo root: deploy/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy python3-venv rsync

# --system-site-packages: picamera2 and cv2 come from apt, not pip.
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

[ -f birdy.toml ] || { cp birdy.toml.example birdy.toml; echo "created birdy.toml - edit site/motion settings"; }
mkdir -p data

sudo cp deploy/birdy.service /etc/systemd/system/birdy.service
sudo systemctl daemon-reload
sudo systemctl enable birdy
echo "installed. start with: sudo systemctl start birdy"
```

- [ ] **Step 3: Write `deploy/deploy.sh`** (run from the Mac)

```bash
#!/usr/bin/env bash
# Sync the repo to the Pi and restart the service. Usage: deploy/deploy.sh
set -euo pipefail
HOST="${BIRDY_HOST:-yassine@birdy.local}"
cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude data/ --exclude .venv/ --exclude birdy.toml --exclude .git/ \
  --exclude '__pycache__/' --exclude .pytest_cache/ --exclude roi.png \
  ./ "$HOST:~/pi-birdy/"

ssh "$HOST" 'cd ~/pi-birdy && .venv/bin/pip install -q -e . && sudo systemctl restart birdy && sleep 3 && systemctl is-active birdy && journalctl -u birdy -n 5 --no-pager'
```

Run: `chmod +x deploy/install.sh deploy/deploy.sh`

- [ ] **Step 4: Patch the existing `CLAUDE.md`**

`CLAUDE.md` already exists (committed in `194a886`) and matches the spec. Do not rewrite it; make these three edits so it matches the code:

1. In **Commands**, replace the venv line
   `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
   with
   `/opt/homebrew/bin/python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # 3.11 matches the Pi`
2. In **Conventions and gotchas**, replace
   `- Timestamps are ISO 8601 local time with offset.`
   with
   `- \`captured_at\` is stored as **UTC** ISO 8601 (so string \`<\` comparisons are chronological) and converted to local time in the template. Photo *paths* use local time.`
3. In **Conventions and gotchas**, after the `--system-site-packages` bullet, add:
   `- The Pi runs Raspberry Pi OS Trixie (Debian 13, Python 3.13, NumPy 2.x, OpenCV 4.10 from apt). Never pin \`numpy\` in \`pyproject.toml\`: with \`--system-site-packages\` the apt NumPy must satisfy it, or pip installs a second NumPy that breaks the apt \`cv2\`. \`opencv-python-headless\` is a dev-only extra for the Mac.`

- [ ] **Step 5: Commit**

```bash
git add deploy/ CLAUDE.md
git commit -m "Add deploy scripts, systemd unit and CLAUDE.md"
```

---

### Task 13: Deploy to the Pi and verify end-to-end

Manual steps; the user runs interactive ones with the `!` prefix.

- [ ] **Step 1: SSH key auth** (user runs; password typed once)

```bash
ssh-copy-id yassine@birdy.local
ssh yassine@birdy.local 'echo ok && python3 --version && libcamera-hello --list-cameras 2>/dev/null | head -5 || rpicam-hello --list-cameras | head -5'
```
Expected: `ok`, `Python 3.13.x` (Trixie), camera listed as `imx219`.

- [ ] **Step 2: Initial sync + install**

```bash
rsync -az --exclude data/ --exclude .venv/ --exclude .git/ --exclude '__pycache__/' ./ yassine@birdy.local:~/pi-birdy/
ssh -t yassine@birdy.local 'cd ~/pi-birdy && deploy/install.sh'
```
Expected: ends with `installed. start with: sudo systemctl start birdy`. Then verify the TFLite runtime imported fine:
```bash
ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/python -c "import cv2, numpy, picamera2; from ai_edge_litert.interpreter import Interpreter; print(cv2.__version__, numpy.__version__)"'
```
Expected: versions printed, no ImportError. If `ai_edge_litert` fails to install (no wheel), run `.venv/bin/pip install tflite-runtime` — the classifier falls back to it.

- [ ] **Step 3: Classify a fixture on the Pi** (proves the model runs on aarch64)

```bash
scp tests/fixtures/bird.jpg yassine@birdy.local:/tmp/bird.jpg
ssh yassine@birdy.local 'cd ~/pi-birdy && time .venv/bin/birdy classify /tmp/bird.jpg'
```
Expected: `bird=True p_bird=0.9xx`, real time well under 1 s.

- [ ] **Step 4: Configure site and ROI**

Edit `birdy.toml` on the Pi: set `site.lat`/`lon` (ask the user for their approximate coordinates). Then:
```bash
ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy roi --out roi.png'
scp yassine@birdy.local:~/pi-birdy/roi.png /private/tmp/claude-501/-Users-yassine-www-pi-birdy/a0942696-7d59-46f4-bfae-4ac8e8affaac/scratchpad/roi.png
```
View the PNG (Read tool), pick a polygon around the feeder in 320×240 coordinates, write it into `motion.roi` on the Pi.

If `birdy roi` fails with a buffer/CMA allocation error, set `camera.main_size = [1640, 1232]` in `birdy.toml` and retry.

- [ ] **Step 5: Start the service and watch**

```bash
ssh yassine@birdy.local 'sudo systemctl start birdy && sleep 5 && systemctl is-active birdy && journalctl -u birdy -n 20 --no-pager'
```
Expected: `active`, log lines `birdy starting`, and either captures (`captured 2026/...`) or `outside daylight window` if it's night.

Open `http://birdy.local:8000` in a browser: status line shows `capturing`, motion score changes when you wave a hand in front of the feeder, tiles appear with `bird`/`no` tags. `/roi.png` shows the live frame with the polygon.

- [ ] **Step 6: Tune**

Watch `motion` on the status line while nothing is happening — if it hovers above ~50, raise `pixel_threshold`; if a landing bird scores below `min_area`, lower `min_area`. Edit `birdy.toml` on the Pi and `sudo systemctl restart birdy`. Check `non-birds` filter after a day to see what the classifier rejected.

- [ ] **Step 7: Commit any tuned defaults**

If the defaults in `birdy.toml.example` / `config.py` changed, update both and commit:
```bash
git add birdy.toml.example birdy/config.py
git commit -m "Tune default motion settings from first deployment"
```
