# Motion Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify the square area around what moved instead of the whole ROI box, so a bird fills the model's 224×224 input (Plan A / Part 1 of `docs/superpowers/specs/2026-10-05-recognition-improvements-design.md`).

**Architecture:** `MotionDetector` additionally returns the bounding box of the changed-pixel blobs. The capture loop stores it on the row, normalised to 0..1 of the lores frame. One pure function, `crop_box()` in `birdy/motion.py`, turns a stored box + the current ROI into the crop to classify (a padded square around the motion, or today's ROI crop as fallback). The pipeline, `birdy classify --capture` and `/captures/{id}/crop.jpg` all use it, so the three always agree.

**Tech Stack:** Python 3.11 (Mac) / 3.13 (Pi), numpy, OpenCV, SQLite, FastAPI + Jinja2, pytest.

**Conventions:** Run tests with `.venv/bin/pytest`. Never import `picamera2` outside `birdy/capture.py`. Commit after each task with the trailer:

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AgaiTFaRCqUA3WSx8TK4W4
```

## File map

| File | Change |
|---|---|
| `birdy/motion.py` | `blob_bbox`, `MotionResult.bbox`, `motion_crop`, `normalise_bbox`/`denormalise_bbox`, `CropBox`, `crop_box` |
| `birdy/config.py` | main/lores aspect-ratio validation |
| `birdy/store.py` | `motion_x/y/w/h` columns, `add_capture(..., motion_box=)`, `Capture.motion_box` |
| `birdy/classifier.py` | `pad` override on `CropLoader.load` / `classify`; `ImageNetBirdClassifier.top_k_image` |
| `birdy/capture.py` | pass the normalised box from the detector into `_capture_still` → `add_capture` |
| `birdy/pipeline.py` | `lores_size` param; classify with `crop_box(...)` (live, sweep, backfill) |
| `birdy/cli.py` | `classify --capture <id>`; pass `lores_size` to `Pipeline` |
| `birdy/web/app.py`, `birdy/web/templates/index.html` | `GET /captures/{id}/crop.jpg`, "Model view" link per tile |
| `CLAUDE.md` | document the motion crop |
| tests | `test_motion.py`, `test_config.py`, `test_store.py`, `test_classifier.py`, `test_capture.py`, `test_pipeline.py`, `test_cli.py`, `test_web.py` |

---

### Task 1: Blob bounding box in the motion detector

**Files:**
- Modify: `birdy/motion.py`
- Test: `tests/test_motion.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_motion.py`)

```python
from birdy.motion import blob_bbox


def test_trigger_reports_blob_bbox():
    d = make_detector(FakeClock())
    d.process(blank())
    r = d.process(with_blob(100, 100))
    assert r.bbox is not None
    # 30x30 blob, plus the blur halo and the 3x3 dilation: a few px larger on each side.
    assert 95 <= r.bbox.x <= 100 and 95 <= r.bbox.y <= 100
    assert 30 <= r.bbox.w <= 36 and 30 <= r.bbox.h <= 36


def test_no_motion_has_no_bbox():
    d = make_detector(FakeClock())
    d.process(blank())
    assert d.process(blank()).bbox is None


def test_blob_outside_roi_has_no_bbox():
    d = make_detector(FakeClock())
    d.process(blank())
    assert d.process(with_blob(0, 0)).bbox is None


def test_blob_bbox_unions_blobs_and_drops_specks():
    changed = np.zeros((H, W), np.uint8)
    changed[100:110, 100:110] = 1     # blob A
    changed[150:160, 200:210] = 1     # blob B
    changed[50, 50] = 1               # 1 px speck: dropped (< MIN_BLOB_AREA original pixels)
    b = blob_bbox(changed)
    assert b is not None
    assert b.x <= 100 and b.y <= 100
    assert b.x + b.w >= 210 and b.y + b.h >= 160
    assert b.x >= 98 and b.y >= 98    # the speck at (50, 50) did not widen it


def test_blob_bbox_none_for_specks_only():
    changed = np.zeros((H, W), np.uint8)
    changed[50, 50] = 1
    changed[200, 300] = 1
    assert blob_bbox(changed) is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_motion.py -q`
Expected: ImportError for `blob_bbox` (collection error).

- [ ] **Step 3: Implement** in `birdy/motion.py`

Add below `class BBox`:

```python
MIN_BLOB_AREA = 4  # changed (pre-dilation) pixels; smaller blobs are noise specks
_JOIN_KERNEL = np.ones((3, 3), np.uint8)
```

Change `MotionResult`:

```python
@dataclass(frozen=True)
class MotionResult:
    score: int
    triggered: bool
    # Union box of the changed-pixel blobs inside the ROI, lores px; None when nothing moved.
    bbox: BBox | None = None
```

Add after `roi_bbox`:

```python
def blob_bbox(changed: np.ndarray, min_area: int = MIN_BLOB_AREA) -> BBox | None:
    """Union box of the blobs in a 0/1 changed-pixel mask.

    A 3x3 dilation joins a bird's parts (head, tail, wing) into one blob; blobs with fewer
    than `min_area` changed pixels *before* dilation are dropped as noise.
    """
    joined = cv2.dilate(changed, _JOIN_KERNEL)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if n <= 1:
        return None
    original = np.bincount(labels[changed > 0], minlength=n)
    keep = [i for i in range(1, n) if original[i] >= min_area]
    if not keep:
        return None
    x0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in keep)
    y0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in keep)
    x1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in keep)
    y1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in keep)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
```

In `MotionDetector.process`, after `score = int(changed.sum())` add:

```python
        bbox = blob_bbox(changed) if score else None
```

and change the final `return` to:

```python
        return MotionResult(score=score, triggered=triggered, bbox=bbox)
```

(The first-frame early return keeps `MotionResult(score=0, triggered=False)`; `bbox` defaults to None.)

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_motion.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/motion.py tests/test_motion.py
git commit -m "Motion: detector reports the bounding box of what moved"
```

---

### Task 2: Crop selection (`motion_crop`, `crop_box`, normalised boxes)

**Files:**
- Modify: `birdy/motion.py`
- Test: `tests/test_motion.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_motion.py`)

```python
from birdy.motion import CropBox, crop_box, denormalise_bbox, motion_crop, normalise_bbox

FRAME = (W, H)


def test_motion_crop_is_square_twice_the_bird():
    b = motion_crop(BBox(100, 100, 30, 20), ROI, FRAME)
    assert b == BBox(85, 80, 60, 60)  # side 2*30, centred on (115, 110)


def test_motion_crop_has_minimum_side():
    b = motion_crop(BBox(150, 150, 6, 4), ROI, FRAME)
    assert b.w == b.h == 40
    assert b == BBox(133, 132, 40, 40)  # centred on (153, 152)


def test_motion_crop_is_shifted_inside_the_frame():
    b = motion_crop(BBox(300, 225, 10, 10), ((200, 150), (319, 150), (319, 239), (200, 239)), FRAME)
    assert b == BBox(280, 200, 40, 40)
    b = motion_crop(BBox(0, 0, 8, 8), ((0, 0), (100, 0), (100, 100), (0, 100)), FRAME)
    assert b == BBox(0, 0, 40, 40)


def test_motion_crop_capped_at_frame_short_side():
    roi = ((0, 0), (319, 0), (319, 239), (0, 239))
    b = motion_crop(BBox(100, 50, 150, 100), roi, FRAME)  # 15000 px < half of 319*239
    assert b.w == b.h == 240
    assert 0 <= b.x <= W - 240 and b.y == 0


def test_motion_crop_falls_back_for_whole_roi_motion():
    rb = roi_bbox(ROI)  # 240 x 160
    assert motion_crop(BBox(rb.x, rb.y, rb.w, rb.h // 2 + 2), ROI, FRAME) is None
    assert motion_crop(None, ROI, FRAME) is None


def test_normalise_round_trip():
    b = BBox(100, 50, 30, 20)
    n = normalise_bbox(b, FRAME)
    assert n == (100 / 320, 50 / 240, 30 / 320, 20 / 240)
    assert denormalise_bbox(n, FRAME) == b
    # a box stored at 320x240 read back at 640x480 scales with the frame
    assert denormalise_bbox(n, (640, 480)) == BBox(200, 100, 60, 40)


def test_crop_box_motion_and_fallback():
    m = crop_box(normalise_bbox(BBox(100, 100, 30, 20), FRAME), ROI, FRAME)
    assert m == CropBox(bbox=BBox(85, 80, 60, 60), pad=0.0, kind="motion")
    r = crop_box(None, ROI, FRAME)
    assert r == CropBox(bbox=roi_bbox(ROI), pad=0.1, kind="roi")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_motion.py -q`
Expected: ImportError for `CropBox`.

- [ ] **Step 3: Implement** — append to `birdy/motion.py`

```python
MIN_CROP_SIDE = 40         # lores px (~410 px of a 3280-wide still)
MAX_MOTION_FRACTION = 0.5  # motion bigger than this share of the ROI box = lighting/feeder, not a bird
ROI_PAD = 0.1              # the ROI fallback crop's padding (today's behaviour)

NormBox = tuple[float, float, float, float]  # x, y, w, h as fractions of the lores frame


@dataclass(frozen=True)
class CropBox:
    bbox: BBox   # lores px
    pad: float   # fraction added on each side by `crop_to_bbox`
    kind: str    # 'motion' | 'roi'


def normalise_bbox(bbox: BBox, frame_size: tuple[int, int]) -> NormBox:
    fw, fh = frame_size
    return (bbox.x / fw, bbox.y / fh, bbox.w / fw, bbox.h / fh)


def denormalise_bbox(box: NormBox, frame_size: tuple[int, int]) -> BBox:
    fw, fh = frame_size
    x, y, w, h = box
    return BBox(x=round(x * fw), y=round(y * fh), w=round(w * fw), h=round(h * fh))


def motion_crop(bbox: BBox | None, roi: Sequence[Point], frame_size: tuple[int, int]) -> BBox | None:
    """Square crop around the motion box, or None to fall back to the ROI crop.

    Side = 2x the box's longer side (the still is taken a few hundred ms after the trigger
    frame, so the bird may have moved), at least MIN_CROP_SIDE, at most the frame's short
    side; shifted, never shrunk, to stay inside the frame. Square so the model's 224x224
    resize doesn't distort the bird.
    """
    if bbox is None:
        return None
    rb = roi_bbox(roi)
    if bbox.w * bbox.h > MAX_MOTION_FRACTION * rb.w * rb.h:
        return None
    fw, fh = frame_size
    side = min(max(2 * max(bbox.w, bbox.h), MIN_CROP_SIDE), fw, fh)
    x = round(bbox.x + bbox.w / 2 - side / 2)
    y = round(bbox.y + bbox.h / 2 - side / 2)
    x = min(max(x, 0), fw - side)
    y = min(max(y, 0), fh - side)
    return BBox(x=x, y=y, w=side, h=side)


def crop_box(motion_box: NormBox | None, roi: Sequence[Point], frame_size: tuple[int, int]) -> CropBox:
    """The crop to classify for a capture: around its stored motion box, else today's ROI crop."""
    bbox = None if motion_box is None else denormalise_bbox(motion_box, frame_size)
    square = motion_crop(bbox, roi, frame_size)
    if square is not None:
        return CropBox(bbox=square, pad=0.0, kind="motion")
    return CropBox(bbox=roi_bbox(roi), pad=ROI_PAD, kind="roi")
```

Check the expected values while implementing: `BBox(100, 100, 30, 20)` → side 60, centre (115, 110) → x = 85, y = 80. `BBox(150, 150, 6, 4)` → side 40, centre (153, 152) → (133, 132). Python's `round` is banker's rounding on .5; every expected value above has integer centres ± integer half-sides, so no .5 occurs.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_motion.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/motion.py tests/test_motion.py
git commit -m "Motion: square crop around the motion box, ROI fallback"
```

---

### Task 3: Main and lores streams must share an aspect ratio

A lores square is only a main-stream square when both streams have the same aspect ratio (`crop_to_bbox` scales x and y separately).

**Files:**
- Modify: `birdy/config.py` (after `lores_size = _size(...)`, ~line 212)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_config.py`; `write`, `MINIMAL` and `load_config` already exist there — check the imports at the top and add `import pytest` / `ConfigError` if missing)

```python
def test_camera_streams_must_share_aspect_ratio(tmp_path):
    bad = MINIMAL + "\n[camera]\nmain_size = [1920, 1080]\nlores_size = [320, 240]\n"
    with pytest.raises(ConfigError, match="aspect ratio"):
        load_config(write(tmp_path, bad))


def test_default_camera_streams_share_aspect_ratio(tmp_path):
    ok = MINIMAL + "\n[camera]\nmain_size = [1640, 1232]\nlores_size = [320, 240]\n"
    assert load_config(write(tmp_path, ok)).camera.main_size == (1640, 1232)
```

If `MINIMAL` already contains a `[camera]` table, build the TOML for these two tests from scratch instead (copy `MINIMAL`'s other tables).

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -q -k aspect`
Expected: `test_camera_streams_must_share_aspect_ratio` FAILS (no ConfigError raised).

- [ ] **Step 3: Implement** — in `birdy/config.py`, right after the `lores_size = ...` line:

```python
    (mw, mh), (lw0, lh0) = main_size, lores_size
    _check(
        abs(mw / mh - lw0 / lh0) <= 0.01 * (mw / mh),
        "camera.main_size and camera.lores_size must have the same aspect ratio (the motion crop is drawn in lores coordinates)",
    )
```

(3280×2464 = 1.331 and 320×240 = 1.333 differ by 0.2 %, inside the 1 % tolerance.)

- [ ] **Step 4: Run the config tests**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/config.py tests/test_config.py
git commit -m "Config: main and lores streams must share an aspect ratio"
```

---

### Task 4: Store the motion box on the capture row

**Files:**
- Modify: `birdy/store.py` (`Capture`, `_row_to_capture`, the column-migration loop in `Store.__init__`, `add_capture`)
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_store.py`; it already imports `Store`, `T0` and has a `store` fixture — reuse them)

```python
def test_add_capture_stores_motion_box(store):
    _, rel = store.photo_path_for(T0)
    cid = store.add_capture(T0, rel, 100, motion_box=(0.25, 0.5, 0.125, 0.0625))
    assert store.get(cid).motion_box == (0.25, 0.5, 0.125, 0.0625)


def test_add_capture_without_motion_box(store):
    _, rel = store.photo_path_for(T0)
    cid = store.add_capture(T0, rel, 100)
    assert store.get(cid).motion_box is None


def test_old_database_gets_motion_columns(tmp_path):
    import sqlite3

    data = tmp_path / "old"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.execute(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending')"
    )
    conn.execute("INSERT INTO captures (captured_at, path, motion_score) VALUES ('2026-10-01T10:00:00.000+00:00', 'a.jpg', 5)")
    conn.commit()
    conn.close()
    s = Store(data)
    assert s.get(1).motion_box is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_store.py -q -k motion`
Expected: TypeError (`add_capture` got an unexpected keyword `motion_box`) / AttributeError.

- [ ] **Step 3: Implement** in `birdy/store.py`

`Capture` — add a last field:

```python
    # Motion box at the trigger frame as fractions (x, y, w, h) of the lores frame; None = unknown.
    motion_box: tuple[float, float, float, float] | None = None
```

`_row_to_capture` — add the argument:

```python
        motion_box=None if row["motion_x"] is None else (row["motion_x"], row["motion_y"], row["motion_w"], row["motion_h"]),
```

Migration loop in `Store.__init__` — extend the tuple:

```python
            for name, decl in (
                ("species_checked", "INTEGER NOT NULL DEFAULT 0"),
                ("published", "TEXT"),
                ("web_path", "TEXT"),
                ("motion_x", "REAL"),
                ("motion_y", "REAL"),
                ("motion_w", "REAL"),
                ("motion_h", "REAL"),
            ):
```

`add_capture`:

```python
    def add_capture(
        self,
        captured_at: datetime,
        rel_path: Path,
        motion_score: int,
        motion_box: tuple[float, float, float, float] | None = None,
    ) -> int:
        thumb = self._make_thumbnail(rel_path)
        mx, my, mw, mh = motion_box if motion_box is not None else (None, None, None, None)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO captures (captured_at, path, thumb_path, motion_score, status, motion_x, motion_y, motion_w, motion_h)"
                " VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
                (_to_db_time(captured_at), str(rel_path), None if thumb is None else str(thumb), motion_score, mx, my, mw, mh),
            )
            return int(cur.lastrowid)
```

Do not add the columns to the `SCHEMA` `CREATE TABLE` — the migration loop adds them on new and old databases alike, like `published`/`web_path`.

- [ ] **Step 4: Run the store tests, then the full suite** (every `SELECT *` reader now sees the new columns)

Run: `.venv/bin/pytest tests/test_store.py -q && .venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/store.py tests/test_store.py
git commit -m "Store: keep each capture's motion box (normalised)"
```

---

### Task 5: Crop padding override in the classifier

A motion crop is already padded, so it must be cut with `pad = 0`; the ROI crop keeps 0.1.

**Files:**
- Modify: `birdy/classifier.py` (`Classifier` protocol, `CropLoader.load`, `ImageNetBirdClassifier.classify/top_k`, `TwoStageClassifier.classify`)
- Modify: `tests/test_pipeline.py` (the four fake classifiers' signatures)
- Test: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_classifier.py`; add `import cv2`, `import numpy as np` and `from birdy.classifier import CropLoader` / `from birdy.motion import BBox` if not already imported)

```python
def test_crop_loader_pad_override(tmp_path):
    p = tmp_path / "img.jpg"
    cv2.imwrite(str(p), np.zeros((480, 640, 3), np.uint8))
    loader = CropLoader((320, 240))            # main is 2x lores
    box = BBox(10, 20, 40, 40)
    assert loader.load(p, box).shape[:2] == (96, 96)          # default pad 0.1: (40 + 2*4) * 2
    assert loader.load(p, box, pad=0.0).shape[:2] == (80, 80)  # exact box
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_classifier.py -q -k pad_override`
Expected: TypeError (unexpected keyword `pad`).

- [ ] **Step 3: Implement** in `birdy/classifier.py`

Protocol:

```python
class Classifier(Protocol):
    def classify(
        self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None, pad: float | None = None
    ) -> Prediction: ...
```

`CropLoader.load`:

```python
    def load(self, image_path: Path, roi_bbox: BBox | None, pad: float | None = None) -> np.ndarray:
        """`pad` overrides the constructor's padding (a motion crop is already padded: 0)."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cannot read image {image_path}")
        if roi_bbox is not None:
            crop = crop_to_bbox(img, roi_bbox, self._lores_size, self._pad if pad is None else pad)
```

(rest of the method unchanged).

`ImageNetBirdClassifier` — split `top_k` so the CLI can rank an already-cropped image, and thread `pad` through:

```python
    def classify(
        self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None, pad: float | None = None
    ) -> Prediction:
        """`threshold` overrides the constructor default so it can be tuned live."""
        t = self._threshold if threshold is None else threshold
        conf = self.predict(self._crops.load(image_path, roi_bbox, pad))
        return Prediction(is_bird=conf >= t, confidence=conf)

    def top_k(self, image_path: Path, roi_bbox: BBox | None, k: int = 5) -> list[tuple[str, float]]:
        return self.top_k_image(self._crops.load(image_path, roi_bbox), k)

    def top_k_image(self, img_rgb: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        probs = self._probs(img_rgb)
        order = np.argsort(probs)[::-1][:k]
        return [(self._labels[i] if i < len(self._labels) else str(i), float(probs[i])) for i in order]
```

`TwoStageClassifier.classify`:

```python
    def classify(
        self, image_path: Path, roi_bbox: BBox | None, threshold: float | None = None, pad: float | None = None
    ) -> Prediction:
        return self.classify_image(self._crops.load(image_path, roi_bbox, pad), threshold)[0]
```

In `tests/test_pipeline.py`, update every fake `classify` to accept and record `pad`:
- line ~25 (`FakeClassifier`): `def classify(self, image_path, roi_bbox, threshold=None, pad=None):` and add `self.pads: list = []` in `__init__` plus `self.pads.append(pad)` next to `self.bboxes.append(roi_bbox)`;
- lines ~186, ~240, ~296: same signature; where they call `super().classify(image_path, roi_bbox, threshold)` pass `pad` too: `super().classify(image_path, roi_bbox, threshold, pad)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_classifier.py tests/test_two_stage.py tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/classifier.py tests/test_classifier.py tests/test_pipeline.py
git commit -m "Classifier: crop padding can be overridden per call"
```

---

### Task 6: Capture loop stores the motion box

**Files:**
- Modify: `birdy/capture.py` (`_capture_still`, the trigger call in `run_capture`)
- Test: `tests/test_capture.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_capture.py`; add `import cv2`, `from birdy.store import Store` and `from birdy import capture` if not imported)

```python
class _StillCamera:
    def capture_file(self, path):
        cv2.imwrite(path, np.zeros((60, 80, 3), np.uint8))


class _Sink:
    def __init__(self):
        self.events = []

    def submit(self, ev):
        self.events.append(ev)
        return True


def test_capture_still_stores_motion_box(tmp_path):
    store = Store(tmp_path / "data")
    sink = _Sink()
    capture._capture_still(_StillCamera(), store, sink, 321, (0.25, 0.5, 0.125, 0.0625))
    c = store.get(sink.events[0].capture_id)
    assert c.motion_score == 321
    assert c.motion_box == (0.25, 0.5, 0.125, 0.0625)
```

Also change the existing stub at line ~60 so it accepts the new argument:

```python
    monkeypatch.setattr(capture, "_capture_still", lambda picam2, store, pipeline, score, motion_box=None: stills.append(score))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_capture.py -q -k motion_box`
Expected: TypeError (`_capture_still` takes 4 positional arguments).

- [ ] **Step 3: Implement** in `birdy/capture.py`

Import: `from birdy.motion import MotionDetector, normalise_bbox`.

`_capture_still`:

```python
def _capture_still(picam2, store: Store, pipeline: Pipeline, score: int, motion_box=None) -> None:
```

and its `add_capture` call:

```python
    capture_id = store.add_capture(now, rel_path, score, motion_box=motion_box)
```

In `run_capture`, replace `_capture_still(picam2, store, pipeline, result.score)` with:

```python
                        if result.triggered:
                            box = None if result.bbox is None else normalise_bbox(result.bbox, config.camera.lores_size)
                            _capture_still(picam2, store, pipeline, result.score, box)
```

(`config.camera` is the session's pinned camera config, the same size the detector was built with.)

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_capture.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/capture.py tests/test_capture.py
git commit -m "Capture: store the motion box with each still"
```

---

### Task 7: Pipeline classifies the motion crop

**Files:**
- Modify: `birdy/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline.py`)

```python
from birdy.motion import BBox, roi_bbox


def add_with_box(store, i, box):
    when = T0.replace(minute=i)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100, motion_box=box)
    return MotionEvent(capture_id=cid, path=abs_path, captured_at=when, motion_score=100)


def test_live_capture_is_classified_on_its_motion_crop(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    p.start()
    try:
        p.submit(add_with_box(store, 1, (100 / 320, 100 / 240, 20 / 320, 20 / 240)))
        assert wait_until(lambda: clf.calls)
    finally:
        p.stop()
    assert clf.bboxes[-1] == BBox(90, 90, 40, 40)  # 20x20 at (100,100) -> 40 px square
    assert clf.pads[-1] == 0.0


def test_capture_without_box_uses_roi_crop(store, live, config):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    ev = add(store, 2)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert clf.bboxes[-1] == roi_bbox(config.motion.roi)
    assert clf.pads[-1] == 0.1


def test_sweep_uses_the_stored_box(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    add_with_box(store, 3, (100 / 320, 100 / 240, 20 / 320, 20 / 240))  # pending, never submitted
    p.sweep()
    assert clf.bboxes[0] == BBox(90, 90, 40, 40)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -q -k "motion_crop or without_box or stored_box"`
Expected: TypeError (unexpected keyword `lores_size`).

- [ ] **Step 3: Implement** in `birdy/pipeline.py`

Import: replace `from birdy.motion import roi_bbox` with `from birdy.motion import NormBox, crop_box`.

`__init__` — new parameter after `publisher` and stored:

```python
        lores_size: tuple[int, int] = (320, 240),
```
```python
        # The process's pinned lores size (camera keys are restart-only): motion boxes and the
        # ROI are in this frame.
        self._lores_size = lores_size
```

`process`:

```python
    def process(self, capture_id: int, path: Path, captured_at: datetime, motion_box: NormBox | None = None) -> None:
        cfg = self._live.get()
        crop = crop_box(motion_box, cfg.motion.roi, self._lores_size)
        try:
            pred = self._classifier.classify(path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad)
```

(rest unchanged).

`sweep` — pass the stored box:

```python
            self.process(c.id, self._store.photos_dir / c.path, c.captured_at, current.motion_box)
```

`_backfill_species` — replace the classify line:

```python
            crop = crop_box(c.motion_box, cfg.motion.roi, self._lores_size)
            pred = self._classifier.classify(path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad)
```

`_run` — pass the row's box:

```python
                        self.process(ev.capture_id, ev.path, ev.captured_at, current.motion_box)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: all pass (the existing ROI-crop tests still see `roi_bbox(...)`, since their rows have no box).

- [ ] **Step 5: Commit**

```bash
git add birdy/pipeline.py tests/test_pipeline.py
git commit -m "Pipeline: classify the motion crop when the capture has one"
```

---

### Task 8: CLI — `classify --capture`, and wire `lores_size` into the service

**Files:**
- Modify: `birdy/cli.py` (`build_parser`, `cmd_classify`, `main`, `cmd_run`'s `Pipeline(...)` call ~line 95)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli.py`)

```python
def test_parser_classify_capture():
    args = build_parser().parse_args(["classify", "--capture", "7"])
    assert args.capture == 7 and args.image is None


def test_classify_needs_image_or_capture(config_path):
    from birdy import cli

    assert cli.main(["--config", str(config_path), "classify"]) == 2


def test_classify_capture_uses_stored_crop(tmp_path):
    import cv2

    from birdy.store import Store

    bird = FIXTURES / "bird.jpg"
    if not bird.exists() or not (ROOT / "models" / "mobilenet_v2_1.0_224_quant.tflite").exists():
        pytest.skip("fixtures/model missing")
    data = tmp_path / "data"
    store = Store(data)
    when = datetime(2026, 10, 3, 10, 0).astimezone()
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), cv2.imread(str(bird)))
    with_box = store.add_capture(when, rel, 100, motion_box=(2 / 320, 2 / 240, 5 / 320, 5 / 240))
    without = store.add_capture(when, rel, 100)
    cfg = _classify_config(tmp_path, f"{ROOT}/models/mobilenet_v2_1.0_224_inat_bird_quant.tflite")
    cfg.write_text(cfg.read_text() + f'[storage]\ndata_dir = "{data}"\n')

    proc = _run_cli("--config", str(cfg), "classify", "--capture", str(with_box))
    assert proc.returncode == 0, proc.stderr
    assert "motion crop" in proc.stdout and "gate (ImageNet):" in proc.stdout

    proc = _run_cli("--config", str(cfg), "classify", "--capture", str(without))
    assert proc.returncode == 0, proc.stderr
    assert "roi crop" in proc.stdout

    proc = _run_cli("--config", str(cfg), "classify", "--capture", "999")
    assert proc.returncode == 1
```

(The `_classify_config` ROI is the 10×10 square at the origin; a 5×5 box is a quarter of it, so the motion crop applies.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -q -k "classify"`
Expected: the three new tests fail (argparse error: unrecognised `--capture` / missing `image`).

- [ ] **Step 3: Implement** in `birdy/cli.py`

Imports: add `from birdy.motion import crop_box`.

`build_parser`:

```python
    cls = sub.add_parser("classify", help="classify an image file, or a stored capture with its crop")
    cls.add_argument("image", nargs="?", help="image file (classified whole, no crop)")
    cls.add_argument("--capture", type=int, default=None, help="capture id: classify its photo with the crop the service uses")
```

`cmd_classify` (replaces the current function):

```python
def cmd_classify(config: Config, image: str | None, capture_id: int | None = None) -> int:
    """Classify one image and print both models' views.

    With `--capture`, the stored photo is cropped exactly as the pipeline crops it (motion
    crop, else the ROI crop). `clf.gate.predict` / `top_k_image` bypass the classifier's lock:
    fine for this one-shot CLI, never do that in the service.
    """
    if (image is None) == (capture_id is None):
        print("classify: give an image file or --capture <id>", file=sys.stderr)
        return 2
    bbox, pad = None, None
    if capture_id is not None:
        store = Store(config.storage.data_dir)
        c = store.get(capture_id)
        if c is None:
            print(f"no capture {capture_id}", file=sys.stderr)
            return 1
        path = store.photos_dir / c.path
        crop = crop_box(c.motion_box, config.motion.roi, config.camera.lores_size)
        bbox, pad = crop.bbox, crop.pad
        b = crop.bbox
        print(f"capture {c.id} {c.path}: {crop.kind} crop x={b.x} y={b.y} w={b.w} h={b.h} (lores px)")
    else:
        path = Path(image)
    live = LiveConfig(config)
    clf = _classifier(config, live)
    try:
        img = CropLoader(config.camera.lores_size).load(path, bbox, pad)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    p_gate = clf.gate.predict(img)
    pred, top = clf.classify_image(img)
    threshold = config.classifier.threshold
    rescued = ", rescued by species" if pred.is_bird and p_gate < threshold else ""
    print(
        f"bird={pred.is_bird} p_bird={pred.confidence:.3f} (gate {p_gate:.3f}{rescued}) "
        f"species={pred.species or '-'}"
    )
    print("gate (ImageNet):")
    for label, prob in clf.gate.top_k_image(img, k=5):
        print(f"  {prob:.3f}  {label}")
    print("species (iNat):")
    for label, prob in top:
        print(f"  {prob:.3f}  {label.common} ({label.latin})")
    return 0
```

`main`:

```python
        if args.command == "classify":
            return cmd_classify(config, args.image, args.capture)
```

`cmd_run` (~line 95):

```python
    pipeline = Pipeline(store, clf, live, notifier=notifier, publisher=publisher, lores_size=config.camera.lores_size)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_cli.py -q`
Expected: all pass (the existing `test_parser_subcommands` still gets `.image == "a.jpg"`).

- [ ] **Step 5: Commit**

```bash
git add birdy/cli.py tests/test_cli.py
git commit -m "CLI: classify --capture shows the crop the service uses"
```

---

### Task 9: `/captures/{id}/crop.jpg` and the tile's "Model view" link

**Files:**
- Modify: `birdy/web/app.py`, `birdy/web/templates/index.html`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_web.py`; it already has `store`, `client`, `T0`)

```python
def _add_photo(store, box=None):
    abs_path, rel = store.photo_path_for(T0)
    cv2.imwrite(str(abs_path), np.full((480, 640, 3), 90, np.uint8))   # 2x the 320x240 lores frame
    return store.add_capture(T0, rel, 100, motion_box=box)


def test_crop_jpg_is_the_motion_crop(client, store):
    cid = _add_photo(store, (100 / 320, 100 / 240, 20 / 320, 20 / 240))
    r = client.get(f"/captures/{cid}/crop.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (80, 80)   # 40 lores px square, x2


def test_crop_jpg_falls_back_to_roi(client, store, config):
    from birdy.motion import roi_bbox

    cid = _add_photo(store)
    img = cv2.imdecode(np.frombuffer(client.get(f"/captures/{cid}/crop.jpg").content, np.uint8), cv2.IMREAD_COLOR)
    rb = roi_bbox(config.motion.roi)   # 240 x 160 lores, padded 10 % each side, x2
    assert img.shape[1] == round((rb.w * 1.2) * 2) and img.shape[0] == round((rb.h * 1.2) * 2)


def test_crop_jpg_unknown_capture(client):
    assert client.get("/captures/999/crop.jpg").status_code == 404


def test_grid_tile_links_model_view(client, store):
    cid = _add_photo(store)
    store.update_classification(cid, True, 0.9, None, None)
    assert f'href="/captures/{cid}/crop.jpg"' in client.get("/?filter=all").text
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_web.py -q -k "crop_jpg or model_view"`
Expected: 404 / assertion failures.

- [ ] **Step 3: Implement**

`birdy/web/app.py` — imports: `from birdy.classifier import CropLoader` and `from birdy.motion import crop_box` (check what `app.py` already imports from these modules). Inside `create_app`, after the `StaticFiles` mounts:

```python
    # Camera keys are restart-only: crop with the size the running process started with.
    lores_size = live.get().camera.lores_size

    def _crop_jpeg(c) -> bytes | None:
        """The exact crop the classifier sees for this capture (BGR JPEG); None if unreadable."""
        crop = crop_box(c.motion_box, live.get().motion.roi, lores_size)
        try:
            rgb = CropLoader(lores_size).load(store.photos_dir / c.path, crop.bbox, crop.pad)
        except ValueError:
            return None
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes() if ok else None
```

Route, next to `/frame.jpg`:

```python
    @app.get("/captures/{capture_id}/crop.jpg")
    async def capture_crop(capture_id: int):
        """What the model saw: the motion crop, or the ROI crop for captures without a motion box."""
        c = await run_in_threadpool(store.get, capture_id)
        if c is None:
            raise HTTPException(status_code=404, detail="unknown capture")
        jpg = await run_in_threadpool(_crop_jpeg, c)   # decodes the full-size still: off the event loop
        if jpg is None:
            raise HTTPException(status_code=404, detail="photo unreadable")
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
```

`birdy/web/templates/index.html` — in the tile's `<div class="acts">`, just before the Delete `<form>`:

```html
    <a class="pb view" href="/captures/{{ c.id }}/crop.jpg" title="The crop the model classified">Model view</a>
```

and in the `<style>` block, after the `.acts .pb.del:hover` rule:

```css
  .acts a.pb { display: inline-flex; align-items: center; justify-content: center; text-decoration: none; }
  .acts .pb.view { flex: 0 1 auto; color: #a3aaa5; }
```

- [ ] **Step 4: Run the web tests, then the full suite**

Run: `.venv/bin/pytest tests/test_web.py -q && .venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/web/app.py birdy/web/templates/index.html tests/test_web.py
git commit -m "Web: /captures/{id}/crop.jpg and a Model view link per tile"
```

---

### Task 10: Docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md`**

- Architecture, `birdy/motion.py` bullet: append "It also returns the union box of the changed-pixel blobs (`MotionResult.bbox`); `crop_box()` turns a capture's stored box + the current ROI into the crop to classify: a square 2× the motion box (min 40 lores px, shifted inside the frame), or the ROI box + 10 % when there is no box or the motion covers > 50 % of the ROI box."
- Conventions, the ROI bullet ("ROI polygon coordinates are in **lores** …"): append "Each capture's motion box is stored as `motion_x/y/w/h`, fractions of the lores frame (NULL for captures from before the motion crop, which keep the ROI crop). `camera.main_size` and `lores_size` must share an aspect ratio (validated)."
- Conventions, the `birdy classify <file>` bullet: append "`birdy classify --capture <id>` classifies a stored capture with the crop the service uses and prints which crop (`motion`/`roi`); the grid's **Model view** link (`/captures/{id}/crop.jpg`) shows the same crop."
- Commands block: add `.venv/bin/birdy classify --capture <id>  # a stored capture, with the service's crop`.

- [ ] **Step 2: Run the full suite one last time**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Docs: motion crop"
```

---

## After the plan (manual, on the Pi)

1. `deploy/deploy.sh`.
2. Wait for a few bird captures, open a tile's **Model view**: the bird should fill a good part of the square.
3. `ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy classify --capture <id>'` on a few new captures and compare the species scores with older (ROI-crop) captures.
