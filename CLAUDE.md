# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Birdy: a Raspberry Pi 4 + Camera Module v2.1 behind a window photographs birds at a feeder, filters non-bird motion with a local TFLite model, stores photos + metadata in SQLite, and serves a LAN-only review page.

The authoritative design is `docs/superpowers/specs/2026-10-03-capture-core-design.md` (approved, sub-project 1 of 5). Read it before implementing anything; module interfaces, config keys, DB schema and defaults are all specified there. Later sub-projects (species ID, publishing, multi-site, e-ink) are out of scope unless asked.

## Commands

Development happens on the Mac; the Pi (`yassine@birdy.local`, SSH key auth) only runs the service.

```bash
# Mac
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest                          # full suite (runs entirely on the Mac)
pytest tests/test_motion.py     # one module
pytest tests/test_motion.py -k "cooldown"   # one test
deploy/deploy.sh                # rsync to the Pi + restart the systemd service

# Pi (via ssh)
deploy/install.sh               # one-time: apt deps, venv, birdy.toml, systemd unit
birdy run                       # what the service runs
birdy roi                       # dump one lores frame to roi.png for ROI tuning
birdy classify <file>           # run the classifier on a single image
birdy purge                     # run the retention job now
journalctl -u birdy -f          # logs
```

## Architecture

One Python process, one systemd service, four threads talking through narrow interfaces:

```
capture ──MotionEvent──▶ pipeline (queue + worker) ──▶ classifier
   │                          │
   └──────────────────────────┴──▶ store (SQLite WAL + photos/thumbs on disk)
                                     ▲
web (FastAPI, read-only) ────────────┘      retention (hourly thread) ──▶ store
```

- `birdy/capture.py` is the **only** module that imports `picamera2`. Everything else must import and test on the Mac. Keep it that way; put any new camera-touching code there.
- `birdy/motion.py` is a pure-numpy/OpenCV detector (ROI mask → running background → diff → threshold → cooldown) so it can be tested with synthetic 320×240 frames. No camera code in it.
- `birdy/classifier.py` exposes a `Classifier` protocol returning `Prediction(is_bird, confidence, species, species_confidence)`. The current implementation aggregates ImageNet bird classes from MobileNetV2; a species model will later sit behind the same interface. The `species*` fields exist in the schema now and stay `NULL`.
- `birdy/store.py` owns the single write connection (locked, `check_same_thread=False`); capture, pipeline and retention all write through it. The web app opens its own read-only connection. Never open a second writer.
- Nothing is dropped on overload: capture inserts the row as `status='pending'` before enqueueing, and the pipeline's 5-minute backlog sweep reclassifies anything still pending.
- Retention deletes only `is_bird = 0` rows older than `retention.non_bird_days`; bird photos are never auto-deleted.

## Conventions and gotchas

- Config is `birdy.toml` (gitignored). Every key must be documented in `birdy.toml.example`; add new keys to both the dataclass in `config.py` and the example file.
- ROI polygon coordinates are in **lores** (320×240) space. The classifier scales that bbox to main-stream coordinates before cropping.
- `picamera2` and OpenCV come from apt on the Pi, so the Pi venv is created with `--system-site-packages`. Do not add them as pip dependencies in `pyproject.toml`.
- Photos live at `data/photos/YYYY/MM/DD/HHMMSS_mmm.jpg` with thumbs mirrored under `data/thumbs/`; paths in the DB are relative to those roots. `data/`, `.venv/` and `birdy.toml` are excluded from deploy rsync.
- Timestamps are ISO 8601 local time with offset.
- Models in `models/` are checked in with their license and labels; `imagenet_bird_classes.txt` is the source of truth for which class indices count as "bird".
