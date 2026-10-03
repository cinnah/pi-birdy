# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Birdy: a Raspberry Pi 4 + Camera Module v2.1 behind a window photographs birds at a feeder, filters non-bird motion with a local TFLite model, stores photos + metadata in SQLite, and serves a LAN-only review page.

The authoritative design is `docs/superpowers/specs/2026-10-03-capture-core-design.md` (sub-project 1), together with `docs/superpowers/specs/2026-10-03-settings-page-design.md` for the settings page and live reload (1b), `docs/superpowers/specs/2026-10-03-species-recognition-design.md` for species ID (2) and `docs/superpowers/specs/2026-10-03-telegram-alerts-design.md` for Telegram alerts (6). Read them before implementing anything; module interfaces, config keys, DB schema and defaults are all specified there. Sub-projects 3–5 (publishing, multi-site, e-ink) are out of scope unless asked.

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
.venv/bin/birdy classify <file> # gate + species verdict, then both models' top-5
.venv/bin/birdy purge           # run the retention job now
.venv/bin/birdy telegram-chat-id   # print chat ids that have messaged the bot (--token overrides [telegram] bot_token)
journalctl -u birdy -f          # logs
# Tuning: ROI and thresholds at http://birdy.local:8000/settings (live frame, click-to-draw
# polygon, applies without restart). No auth - LAN only, no CSRF token by design.
```

Telegram bot setup (one-time; the token never leaves the Pi):

1. Telegram → @BotFather → `/newbot` → put the token in `[telegram] bot_token` in `~/pi-birdy/birdy.toml` on the Pi (`install.sh` chmods it 0600 on every run).
2. Open the bot in Telegram and send it any message.
3. `ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy telegram-chat-id'` → put the id in `chat_id`, set `enabled = true`, then `sudo systemctl restart birdy`.
4. Settings page → **Send test alert**.

## Architecture

One Python process, one systemd service, five threads (six with Telegram alerts enabled) talking through narrow interfaces:

```
capture ──MotionEvent──▶ pipeline (queue + worker) ──▶ classifier
   │                          │
   └──────────────────────────┴──▶ store (SQLite WAL + photos/thumbs on disk)
                                     ▲
web (FastAPI + /settings) ───────────┘      retention (hourly thread) ──▶ store

config watcher (5 s stat of birdy.toml) ──▶ LiveConfig ◀── web POST /settings

telegram notifier (queue + worker) ◀── pipeline (live only)
```

- `birdy/capture.py` is the **only** module that imports `picamera2`. Everything else must import and test on the Mac. Keep it that way; put any new camera-touching code there.
- `birdy/motion.py` is a pure-numpy/OpenCV detector (ROI mask → running background → diff → threshold → cooldown) so it can be tested with synthetic 320×240 frames. No camera code in it.
- `birdy/classifier.py` exposes a `Classifier` protocol returning `Prediction(is_bird, confidence, species, species_confidence)`. The implementation is `TwoStageClassifier`: an ImageNet gate (`ImageNetBirdClassifier.predict` → P(bird), summed over ImageNet bird classes from MobileNetV2) plus `InatBirdClassifier` (`birdy/species.py`, 965 iNat bird species), both run on one decoded crop (`CropLoader`). Species is reported when its probability ≥ `species.threshold`; a gate-rejected frame is *rescued* (`is_bird=True`, `confidence = max(gate, species)`) when species ≥ `species.rescue_threshold`.
- The pipeline worker and the web upload box share **one** `TwoStageClassifier` (built in `cmd_run`); it serialises inference under its own lock because TFLite interpreters aren't thread-safe. Don't build a second one for the web.
- `species_checked` marks rows the species model has seen. Each pipeline sweep backfills up to 50 older `done` rows (`_backfill_species`): it never demotes a bird to non-bird, never flips a `done` row to `error` on failure (it marks it checked instead), yields as soon as a live event is queued, and logs one summary line per sweep.
- `birdy/store.py` owns the single write connection (locked, `check_same_thread=False`); capture, pipeline and retention all write through it. The web thread reads through the same locked connection. The only other writer is the one-shot `birdy purge` CLI, which is safe under WAL.
- Nothing is dropped on overload: capture inserts the row as `status='pending'` before enqueueing, and the pipeline's 5-minute backlog sweep reclassifies anything still pending.
- Retention deletes `is_bird = 0` and `status = 'error'` rows older than `retention.non_bird_days`; bird photos are never auto-deleted.
- `birdy/web/app.py` serves the capture grid, `/status`, `/frame.jpg`, `/roi.png` (ROI overlay; still served, `birdy roi` points to it while the service holds the camera) the `/settings` page (plus `POST /settings/test-alert`, which sends a Telegram message and writes nothing), `/species` (per-species summary + an upload box that `POST /classify`s a photo and shows both models' verdict; it runs on the full uploaded image with no ROI crop and nothing is stored), a species dropdown filter on `/` (returns birds only: `is_bird = 1`), and `last_species` in `/status`. `POST /settings` is its only write (to `birdy.toml`, never to the DB).
- `birdy/live.py` — `LiveConfig` holds the current `Config` + a version. Tunables (`motion.*`, `classifier.threshold`, `species.threshold`/`species.rescue_threshold`, `telegram.watchlist/alert_new_species/cooldown_minutes`, site/daylight, retention) are read through `live.get()` on use; `telegram.enabled` is also checked per alert, so `enabled = false` silences alerts live but `enabled = true` needs a restart (the notifier isn't built otherwise); the capture loop rebuilds its detector/gate when `live.version` changes (and `motion`, `site.lat/lon` or `daylight` differ). Camera/web/storage/model keys and `telegram.enabled/bot_token/chat_id/base_url` are read once at startup — changing them needs a restart (the watcher logs a warning).
- `birdy/settings.py` + `birdy/watcher.py` — the `/settings` page validates and atomically rewrites `birdy.toml` (comments are lost; `birdy.toml.example` keeps the docs) and applies it live; `ConfigWatcher` reloads the file when it's edited over SSH (invalid edits are logged and ignored until restart).
- `birdy/notify.py` — `AlertPolicy` is pure: watchlist match is case-insensitive, a first-ever species gives reason `"new"` (bypasses the cooldown), otherwise a per-species cooldown held in memory. `TelegramNotifier` has a bounded queue (20; newest dropped with a warning when full) and one worker thread that `sendPhoto`s the full-res JPEG, retries after 5/15/45 s (not on 400/401/403, which can't succeed) then logs once and drops, and falls back to a text message if the photo was purged. It is fed only by `Pipeline.process()` after a live classification is stored — never by the species backfill — and a `consider()` failure is logged without touching the row. "First-ever" means first row with that species in the DB at alert time: if the backfill labels an older photo with a species before a live capture of it, that species never gets a "new" alert (it was seen before, just not live). Queued alerts are dropped on shutdown. The web shares the same instance for `POST /settings/test-alert` (`send_test()`, synchronous; 502 with Telegram's description on error, 503 when alerts are disabled). When `telegram.enabled = false` the notifier isn't constructed at all.
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
- The iNat species model's output is the opposite: already softmax-scaled (uint8 at 1/256). **Never re-softmax it**; its probabilities sum to ~0.8–0.95 because quantisation truncates the long tail.
- Both models must get the **same RGB** array (`CropLoader` converts once). Feeding BGR silently degrades results — a blue tit comes out as a hummingbird.
- `birdy classify <file>` prints the final verdict with the raw gate probability (and whether species rescued it), then both models' top-5 — the first thing to run when a capture looks misclassified.
- `MotionDetector` suppresses triggers for its first `warmup_frames` (default 10) so camera auto-exposure settling at session start doesn't produce junk captures.
- `telegram.bot_token` is a secret. It lives only in the Pi's `birdy.toml` (0600, gitignored, rsync-excluded), is written back by every settings-page save (`config_to_dict` includes it, so a save never wipes it), is excluded from `repr(Config)` (`field(repr=False)`), is never logged, and every `TelegramError` message is redacted (`_redact`; request URLs are never included). Never put it in git, docs, tests, memory or log lines. The settings page shows only "set"/"not set".
- `telegram.enabled`, `bot_token`, `chat_id`, `base_url` are restart-only (the watcher warns); `watchlist`, `alert_new_species`, `cooldown_minutes` are live. Cooldown state is in memory, so a restart may re-alert a species once; "new species" is derived from the DB (`store.count("birds", species=...) == 1`) and survives restarts.
- Adding a tunable key: `config.py` dataclass + validation, `birdy.toml.example`, `settings.py` (`SettingsPatch`, `config_to_dict`, `apply_patch`, `FIELD_KEYS`, `parse_settings_form`), `settings.html` form row, and the consumer must read it via `live.get()`. `test_config_to_dict_covers_every_field` fails if `config_to_dict` is forgotten.
