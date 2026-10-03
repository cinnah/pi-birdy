# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Birdy: a Raspberry Pi 4 + Camera Module v2.1 behind a window photographs birds at a feeder, filters non-bird motion with a local TFLite model, stores photos + metadata in SQLite, and serves a LAN-only review page.

The authoritative design is `docs/superpowers/specs/2026-10-03-capture-core-design.md` (approved, sub-project 1 of 5), together with `docs/superpowers/specs/2026-10-03-settings-page-design.md` for the settings page and live reload (sub-project 1b). Read it before implementing anything; module interfaces, config keys, DB schema and defaults are all specified there. Later sub-projects (species ID, publishing, multi-site, e-ink) are out of scope unless asked.

## Commands

Development happens on the Mac; the Pi (`yassine@birdy.local`, SSH key auth) only runs the service.

```bash
# Mac
/opt/homebrew/bin/python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # 3.11 is the closest Homebrew match to the Pi's 3.13
.venv/bin/pytest                          # full suite (runs entirely on the Mac)
.venv/bin/pytest tests/test_motion.py     # one module
.venv/bin/pytest tests/test_motion.py -k "cooldown"   # one test
deploy/deploy.sh                # rsync to the Pi + restart the systemd service

# First-time bootstrap (from the Mac)
deploy/deploy.sh                # syncs the repo; its pip step fails on a fresh Pi - expected
ssh -t yassine@birdy.local 'cd ~/pi-birdy && deploy/install.sh'   # -t required: sudo prompts interactively

# Pi (via ssh, from ~/pi-birdy - --config defaults to ./birdy.toml)
deploy/install.sh               # one-time: apt deps, venv, birdy.toml, systemd unit (run via ssh -t)
.venv/bin/birdy run             # what the service runs
.venv/bin/birdy classify <file> # run the classifier on a single image
.venv/bin/birdy purge           # run the retention job now
journalctl -u birdy -f          # logs
# Tuning: ROI and thresholds at http://birdy.local:8000/settings (live frame, click-to-draw
# polygon, applies without restart). No auth - LAN only, no CSRF token by design.
```

## Architecture

One Python process, one systemd service, five threads talking through narrow interfaces:

```
capture ──MotionEvent──▶ pipeline (queue + worker) ──▶ classifier
   │                          │
   └──────────────────────────┴──▶ store (SQLite WAL + photos/thumbs on disk)
                                     ▲
web (FastAPI + /settings) ───────────┘      retention (hourly thread) ──▶ store

config watcher (5 s stat of birdy.toml) ──▶ LiveConfig ◀── web POST /settings
```

- `birdy/capture.py` is the **only** module that imports `picamera2`. Everything else must import and test on the Mac. Keep it that way; put any new camera-touching code there.
- `birdy/motion.py` is a pure-numpy/OpenCV detector (ROI mask → running background → diff → threshold → cooldown) so it can be tested with synthetic 320×240 frames. No camera code in it.
- `birdy/classifier.py` exposes a `Classifier` protocol returning `Prediction(is_bird, confidence, species, species_confidence)`. The current implementation aggregates ImageNet bird classes from MobileNetV2; a species model will later sit behind the same interface. The `species*` fields exist in the schema now and stay `NULL`.
- `birdy/store.py` owns the single write connection (locked, `check_same_thread=False`); capture, pipeline and retention all write through it. The web thread reads through the same locked connection. The only other writer is the one-shot `birdy purge` CLI, which is safe under WAL.
- Nothing is dropped on overload: capture inserts the row as `status='pending'` before enqueueing, and the pipeline's 5-minute backlog sweep reclassifies anything still pending.
- Retention deletes `is_bird = 0` and `status = 'error'` rows older than `retention.non_bird_days`; bird photos are never auto-deleted.
- `birdy/web/app.py` serves the capture grid, `/status`, `/frame.jpg`, `/roi.png` (ROI overlay; still served, `birdy roi` points to it while the service holds the camera) and the `/settings` page; `POST /settings` is its only write (to `birdy.toml`, never to the DB).
- `birdy/live.py` — `LiveConfig` holds the current `Config` + a version. Tunables (`motion.*`, `classifier.threshold`, site/daylight, retention) are read through `live.get()` on use; the capture loop rebuilds its detector/gate when `live.version` changes (and `motion`, `site.lat/lon` or `daylight` differ). Camera/web/storage/model keys are read once at startup — changing them needs a restart (the watcher logs a warning).
- `birdy/settings.py` + `birdy/watcher.py` — the `/settings` page validates and atomically rewrites `birdy.toml` (comments are lost; `birdy.toml.example` keeps the docs) and applies it live; `ConfigWatcher` reloads the file when it's edited over SSH (invalid edits are logged and ignored until restart).
- A settings-page save writes the running config: it overwrites SSH edits made in the last few seconds that the watcher hasn't applied yet, and drops comments and unknown keys.

## Conventions and gotchas

- Config is `birdy.toml` (gitignored). Every key must be documented in `birdy.toml.example`; add new keys to both the dataclass in `config.py` and the example file.
- ROI polygon coordinates are in **lores** (320×240) space. The classifier scales that bbox to main-stream coordinates before cropping.
- `picamera2` and OpenCV come from apt on the Pi, so the Pi venv is created with `--system-site-packages`. Do not add them as pip dependencies in `pyproject.toml`.
- The Pi runs Raspberry Pi OS Trixie (Debian 13, Python 3.13, NumPy 2.x, OpenCV 4.10 from apt). Never pin `numpy` in `pyproject.toml`: with `--system-site-packages` the apt NumPy must satisfy it, or pip installs a second NumPy that breaks the apt `cv2`. `opencv-python-headless` is a dev-only extra for the Mac.
- Photos live at `data/photos/YYYY/MM/DD/HHMMSS_mmm.jpg` with thumbs mirrored under `data/thumbs/`; paths in the DB are relative to those roots. `data/`, `.venv/` and `birdy.toml` are excluded from deploy rsync.
- `captured_at` is stored as **UTC** ISO 8601 (so string `<` comparisons are chronological) and converted to local time in the template. Photo *paths* use local time.
- Models in `models/` are checked in with their license and labels; `imagenet_bird_classes.txt` is the source of truth for which class indices count as "bird".
- The hosted MobileNetV2 TFLite model outputs **logits** (no softmax layer); `ImageNetBirdClassifier` applies softmax itself. Don't "fix" the dequantized sum ≠ 1.
- `MotionDetector` suppresses triggers for its first `warmup_frames` (default 10) so camera auto-exposure settling at session start doesn't produce junk captures.
- Adding a tunable key: `config.py` dataclass + validation, `birdy.toml.example`, `settings.py` (`SettingsPatch`, `config_to_dict`, `apply_patch`, `FIELD_KEYS`, `parse_settings_form`), `settings.html` form row, and the consumer must read it via `live.get()`. `test_config_to_dict_covers_every_field` fails if `config_to_dict` is forgotten.
