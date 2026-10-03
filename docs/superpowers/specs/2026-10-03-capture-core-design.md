# Birdy — Capture Core Design

**Date:** 2026-10-03
**Status:** approved for implementation
**Sub-project:** 1 of 5 (capture core). Later: species recognition, publishing, friends/collab, e-ink.

## Goal

A Raspberry Pi 4 with a Camera Module v2.1, placed indoors behind a window and pointed at a bird feeder, photographs birds automatically during daylight, filters out non-bird motion with a local TFLite model, stores photos + metadata locally, and shows them on a LAN-only web page for review and tuning.

Inspired by [thjkral/BirdDetection](https://github.com/thjkral/BirdDetection), rebuilt for Raspberry Pi OS Trixie (64-bit Lite) and the `picamera2`/libcamera stack.

## Hardware & environment

- Raspberry Pi 4, Raspberry Pi OS Trixie (Debian 13, Python 3.13) 64-bit Lite, fresh install, networked, camera verified working.
- Camera Module v2.1 (8 MP, fixed focus ~1 m → ∞, no night capability).
- Placement: indoors, lens flush against the window, feeder ≥ 1 m away. Outdoor enclosure is a possible later move; only the ROI mask would need retuning.
- Host: `yassine@birdy.local`, SSH key auth (no passwords stored anywhere).
- Development on macOS; deploy to the Pi via rsync + systemd restart.

## Decisions made

| Question | Decision |
|---|---|
| Trigger | Camera-based motion detection (frame differencing on a low-res stream) |
| What to keep | Bird photos forever; non-bird photos auto-deleted after N days (default 7) |
| Viewing | Minimal LAN web page (grid, filter, status) |
| Hours | Daylight only (sunrise/sunset ± margin, from site lat/lon); 24/7 if no lat/lon configured |
| Architecture | One Python process, one systemd service, modules with strict interfaces |
| Bird/not-bird model | MobileNetV2 ImageNet TFLite, bird classes aggregated — no training needed |
| Species | Not in this sub-project, but schema fields exist (`species`, `species_confidence`) |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ birdy (one process, systemd service)                        │
│                                                             │
│  capture ──MotionEvent──▶ pipeline ──▶ classifier           │
│  (picamera2,              (queue +      (ai-edge-litert,    │
│   motion, ROI,             worker        bird vs not-bird)  │
│   daylight gate)           thread)            │             │
│                               │               ▼             │
│                               └──────▶ store (SQLite +      │
│                                         photos on disk,     │
│                                         retention job)      │
│                                               ▲             │
│  web (FastAPI, read-only LAN page) ───────────┘             │
└─────────────────────────────────────────────────────────────┘
```

Module responsibilities (each is importable and testable alone):

| Module | Owns | Depends on | Pi-only? |
|---|---|---|---|
| `birdy/config.py` | Load/validate `birdy.toml` into a dataclass | — | no |
| `birdy/motion.py` | Pure-numpy motion detector: background model, ROI mask, threshold, cooldown | numpy, opencv | no |
| `birdy/capture.py` | Camera setup, lores/main streams, motion loop glue, daylight gate, emits `MotionEvent` | picamera2, motion, daylight | **yes** (only module importing picamera2) |
| `birdy/daylight.py` | Is it daytime now, when is the next transition | astral | no |
| `birdy/pipeline.py` | Queue + worker thread; classify → store; error isolation; backlog sweep | classifier, store | no |
| `birdy/classifier.py` | `Classifier` interface + `ImageNetBirdClassifier` (TFLite) | ai-edge-litert (`tflite-runtime` fallback import), opencv | no |
| `birdy/store.py` | SQLite schema, `add()`, `query()`, thumbnails, retention | sqlite3, opencv | no |
| `birdy/web/` | FastAPI app, one Jinja template, `/status` | fastapi, store | no |
| `birdy/cli.py` | `birdy run`, `birdy roi`, `birdy classify <file>`, `birdy purge` | all | partly |

Config is a single `birdy.toml` (gitignored on the Pi; `birdy.toml.example` in the repo documents every key).

## Capture & motion detection

**Camera (`picamera2`)**
- One configuration with two streams: `main` 3280×2464 RGB → JPEG stills; `lores` 320×240 YUV420 for analysis. Same sensor exposure, no mode switch on trigger.
- AE/AWB on. Still saved as JPEG quality 90 via `capture_file()` on the `main` stream.

**Motion loop** (~10 fps on lores; target < 10 % of one core)
1. Y plane of the lores frame → Gaussian blur (5×5) to suppress sensor noise.
2. Multiply by the **ROI mask** (polygon from `motion.roi`, in lores coordinates, rasterised once at start). Everything outside is ignored — this is the defence against window reflections, curtains and branches.
3. Running background: `bg = (1-α)·bg + α·frame`, α = `motion.background_alpha` (default 0.05). Diff = `|frame − bg|`, binarised at `motion.pixel_threshold` (default 25). `motion_score` = count of changed pixels inside the ROI.
4. Trigger when `motion_score ≥ motion.min_area` (default 150 px). Then capture still, emit `MotionEvent(path, captured_at, motion_score)`, and enter cooldown `motion.cooldown_seconds` (default 3.0).
5. Background keeps updating during cooldown so a stationary bird fades into the background and stops re-triggering.

**Daylight gate**
- `daylight.py` uses `astral` with `site.lat`/`site.lon`. Active window = sunrise − margin … sunset + margin (`daylight.margin_minutes`, default 30). Outside it, `capture` stops the camera and sleeps until the next transition (re-evaluated every 60 s so clock changes don't strand it). If lat/lon are missing, the gate is disabled and capture runs continuously.

**Tuning aids**
- `birdy roi` saves one lores frame to `roi.png` for picking polygon coordinates on the Mac.
- The web `/status` endpoint exposes the live `motion_score`, so `min_area` can be tuned by watching it.

**Error handling**
- Camera init failure: log, exit code 1; systemd `Restart=on-failure`, `RestartSec=10`.
- Exception on a single frame: log with traceback, skip frame, continue.
- Still-capture failure: log, skip; the detector's cooldown still applies, so a broken camera can't re-fire on every frame.

## Classifier

Interface:

```python
@dataclass(frozen=True)
class Prediction:
    is_bird: bool
    confidence: float          # P(bird), 0..1
    species: str | None = None
    species_confidence: float | None = None

class Classifier(Protocol):
    def classify(self, image_path: Path, roi_bbox: BBox | None) -> Prediction: ...
```

`ImageNetBirdClassifier`:
- MobileNetV2 1.0 224, ImageNet, uint8-quantized TFLite (~3.5 MB), from the TensorFlow hosted-models list. Checked into `models/` with its license and labels file.
- Pre-processing: crop the still to the ROI bounding box scaled from lores to main coordinates (plus 10 % padding), resize to 224×224. Without the crop a small bird is too few pixels.
- `confidence` = sum of softmax probabilities over ImageNet bird class indices (7–24, 80–100, 127–146; the exact list lives in `models/imagenet_bird_classes.txt`). `is_bird = confidence ≥ classifier.threshold` (default 0.5).
- Expected latency on Pi 4: 50–80 ms. Loaded once at startup; thread-safe by giving the worker thread its own interpreter.

Sub-project 2 will add a species classifier behind the same interface (planned: Google AIY Birds V1 TFLite, ~965 species) and populate `species` fields. Not built here.

## Pipeline

- `queue.Queue(maxsize=50)` between the capture thread and one worker thread.
- Capture does `put_nowait`; on `queue.Full` it logs a warning and leaves the still on disk. The store row is created by capture immediately with `status='pending'`, so nothing is ever lost — only delayed.
- Worker loop: get event → `classifier.classify()` → `store.update_classification()` (`status='done'`). Any exception → log with path → `status='error'`; loop continues.
- Backlog sweep every 5 min: classify rows still `pending` (covers queue overflow and restarts mid-queue).

## Store

SQLite (stdlib `sqlite3`), file `data/birdy.db`, WAL mode so the web reader never blocks the writer. One `Store` instance holds a single write connection (`check_same_thread=False`) guarded by a lock; capture, pipeline and retention threads all write through it, and the web thread reads through the same locked connection (reads are short; a second connection buys nothing at this scale).

```sql
CREATE TABLE site (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  name TEXT NOT NULL,
  lat REAL, lon REAL
);

CREATE TABLE captures (
  id INTEGER PRIMARY KEY,
  captured_at TEXT NOT NULL,          -- ISO 8601, UTC
  path TEXT NOT NULL,                 -- relative to data/photos
  thumb_path TEXT,                    -- relative to data/thumbs
  motion_score INTEGER NOT NULL,
  is_bird INTEGER,                    -- NULL until classified
  confidence REAL,
  species TEXT,
  species_confidence REAL,
  status TEXT NOT NULL DEFAULT 'pending'  -- pending | done | error
);
CREATE INDEX captures_captured_at ON captures(captured_at DESC);
CREATE INDEX captures_is_bird ON captures(is_bird, captured_at DESC);
```

- `site` has exactly one row, seeded from `birdy.toml` at startup (updated if config changes). It exists now so that multi-site sharing in sub-project 4 has a stable identity to key on.
- Photos: `data/photos/YYYY/MM/DD/HHMMSS_mmm.jpg`. Thumbnails: `data/thumbs/<same path>`, 400 px wide, generated at insert time; the web page never reads full-res files except on click.
- Store API: `add_capture()`, `update_classification()`, `list_captures(filter, limit, offset)`, `pending()`, `purge_non_birds(older_than_days)`, `disk_free_bytes()`.
- **Retention job** (thread in the main process, hourly): deletes non-bird (`is_bird = 0`) and `status = 'error'` captures older than `retention.non_bird_days` (default 7) — files, thumbs and rows. `is_bird = 1` rows are never auto-deleted. Logs disk free; warns below 1 GB.

## Web

- FastAPI + one Jinja2 template, served by uvicorn in a thread of the main process on `0.0.0.0:8000`. LAN only, no auth, read-only.
- `GET /` — thumbnail grid newest-first, filter `all | birds | non-birds` (query param), pagination (50/page). Tile: time, `confidence`, `motion_score`, species if present. Click → full-res JPEG.
- `GET /status` (JSON, also rendered at the top of `/`): `state` (`capturing` | `daylight_paused` | `camera_error`), `motion_score` (latest), `queue_depth`, `pending_count`, `disk_free_bytes`, `last_capture_at`, `uptime_seconds`.
- `GET /roi.png` — latest lores frame with the ROI polygon drawn on it, for tuning.

## Configuration (`birdy.toml`)

```toml
[site]
name = "home"
lat = 0.0        # optional; omit both to disable the daylight gate
lon = 0.0

[camera]
main_size = [3280, 2464]
lores_size = [320, 240]
jpeg_quality = 90

[motion]
roi = [[40, 60], [280, 60], [280, 220], [40, 220]]   # lores coords
pixel_threshold = 25
min_area = 150
background_alpha = 0.05
cooldown_seconds = 3.0

[daylight]
margin_minutes = 30

[classifier]
model = "models/mobilenet_v2_1.0_224_quant.tflite"
labels = "models/imagenet_labels.txt"
bird_classes = "models/imagenet_bird_classes.txt"
threshold = 0.5

[retention]
non_bird_days = 7

[storage]
data_dir = "/home/yassine/pi-birdy/data"

[web]
host = "0.0.0.0"
port = 8000
```

## Repo layout

```
pi-birdy/
  birdy/
    __init__.py  config.py  motion.py  daylight.py  capture.py
    pipeline.py  classifier.py  store.py  cli.py
    web/  (app.py, templates/index.html, static/)
  models/        mobilenet_v2_1.0_224_quant.tflite, imagenet_labels.txt,
                 imagenet_bird_classes.txt, LICENSE
  deploy/        birdy.service, install.sh, deploy.sh
  tests/         test_motion.py test_daylight.py test_classifier.py
                 test_store.py test_pipeline.py test_web.py fixtures/
  docs/superpowers/specs/
  birdy.toml.example
  pyproject.toml
  CLAUDE.md
```

## Deploy

- `deploy/install.sh` (run once on the Pi via ssh): `apt install python3-picamera2 python3-opencv python3-numpy`; `python3 -m venv --system-site-packages .venv`; `pip install -e .` (pulls `ai-edge-litert` — `tflite-runtime` is a fallback import, `fastapi`, `uvicorn`, `jinja2`, `astral`); copy `birdy.toml.example` → `birdy.toml` if absent; install and enable `deploy/birdy.service`.
- `deploy/birdy.service`: `ExecStart=/home/yassine/pi-birdy/.venv/bin/birdy run`, `WorkingDirectory=/home/yassine/pi-birdy`, `Restart=on-failure`, `RestartSec=10`, `User=yassine`.
- `deploy/deploy.sh` (from the Mac): `rsync -az --delete --exclude data/ --exclude .venv/ --exclude birdy.toml ./ birdy.local:~/pi-birdy/` then `ssh birdy.local sudo systemctl restart birdy`.
- Logs: `ssh birdy.local journalctl -u birdy -f`.
- `picamera2` and OpenCV come from apt (pip wheels for them are unreliable on the Pi); hence `--system-site-packages`.

## Testing

All tests run on the Mac with `pytest`; nothing imports `picamera2` outside `capture.py`.

- `motion.py`: synthetic 320×240 frames — static scene → no trigger; blob inside ROI → trigger with expected score; same blob outside ROI → none; second blob within cooldown → none; stationary blob fades into background after N frames.
- `daylight.py`: fixed lat/lon and dates → known active/inactive; missing lat/lon → always active.
- `classifier.py`: real TFLite model against `tests/fixtures/{bird,cat,empty_feeder}.jpg` → `is_bird` as expected; ROI-bbox crop produces a 224×224 input.
- `store.py`: temp DB — add/update/list/filter; thumbnail created; `purge_non_birds` deletes only old non-birds and old `error` rows, and their files.
- `pipeline.py`: fake classifier (raises on one path) → that row becomes `error`, next row `done`; backlog sweep picks up `pending`.
- `web`: `TestClient` against a seeded temp store — grid renders, filter works, `/status` JSON shape.
- `capture.py`: manual verification on the Pi (`birdy run` under journalctl; `birdy roi`).

## Out of scope (later sub-projects)

- Species identification (2), public publishing to hosting (3), multi-site/friends (4), e-ink display (5), audio/BirdNET, outdoor enclosure, Coral accelerator, auth on the web page, retraining the bird/not-bird gate on own photos.
