# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Birdy: a Raspberry Pi 4 + Camera Module v2.1 behind a window photographs birds at a feeder, filters non-bird motion with a local TFLite model, stores photos + metadata in SQLite, and serves a LAN-only review page.

The authoritative design is `docs/superpowers/specs/2026-10-03-capture-core-design.md` (sub-project 1), together with `docs/superpowers/specs/2026-10-03-settings-page-design.md` for the settings page and live reload (1b), `docs/superpowers/specs/2026-10-03-species-recognition-design.md` for species ID (2) and `docs/superpowers/specs/2026-10-03-telegram-alerts-design.md` for Telegram alerts (6) and `docs/superpowers/specs/2026-10-04-publishing-design.md` for the public gallery (3). Read them before implementing anything; module interfaces, config keys, DB schema and defaults are all specified there. Sub-projects 4–5 (multi-site onboarding, e-ink) are out of scope unless asked.

## Commands

Development happens on the Mac; the Pi (`yassine@birdy.local`, SSH key auth) only runs the service.

```bash
# Mac
/opt/homebrew/bin/python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # 3.11 is the closest Homebrew match to the Pi's 3.13
.venv/bin/pytest                          # full suite (runs entirely on the Mac)
.venv/bin/pytest tests/test_motion.py     # one module
.venv/bin/pytest tests/test_motion.py -k "cooldown"   # one test
deploy/deploy.sh                # rsync to the Pi + restart the systemd service
deploy/gallery.sh               # push gallery/ pages to the server (never sites/ or sites.json)

# First-time bootstrap (from the Mac)
deploy/deploy.sh                # syncs the repo; its pip step fails on a fresh Pi - expected
ssh -t yassine@birdy.local 'cd ~/pi-birdy && deploy/install.sh'   # -t required: sudo prompts interactively

# Pi (via ssh, from ~/pi-birdy - --config defaults to ./birdy.toml)
deploy/install.sh               # one-time: apt deps, venv, birdy.toml, systemd unit (run via ssh -t)
.venv/bin/birdy run             # what the service runs
.venv/bin/birdy classify <file> # gate + species verdict, then both models' top-5
.venv/bin/birdy purge           # run the retention job now
.venv/bin/birdy telegram-chat-id   # print chat ids that have messaged the bot (--token overrides [telegram] bot_token)
.venv/bin/birdy publish           # build the gallery mirror and rsync it to the server once
journalctl -u birdy -f          # logs
# Tuning: ROI and thresholds at http://birdy.local:8000/settings (live frame, click-to-draw
# polygon, applies without restart). No auth - LAN only, no CSRF token by design.
```

Telegram bot setup (one-time; the token never leaves the Pi):

1. Telegram → @BotFather → `/newbot` → put the token in `[telegram] bot_token` in `~/pi-birdy/birdy.toml` on the Pi (`install.sh` chmods it 0600 on every run).
2. Open the bot in Telegram and send it any message.
3. `ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy telegram-chat-id'` → put the id in `chat_id`, set `enabled = true`, then `sudo systemctl restart birdy`.
4. Settings page → **Send test alert**.

Gallery setup (one-time; public at `https://birdy.akibeo.com`; the server runs no code, Caddy serves `/srv/birdy`). DNS is already in place: `birdy.akibeo.com A 91.98.229.209` at Combell.

1. On the server (`ssh root@91.98.229.209`):
   ```bash
   adduser --system --group --home /srv/birdy/sites --shell /usr/sbin/nologin birdy
   mkdir -p /srv/birdy/sites/home /srv/birdy/sites/.ssh && chmod 700 /srv/birdy/sites/.ssh
   echo '[{"id":"home","name":"Yassine","place":"Sintra, PT"}]' > /srv/birdy/sites.json   # or copy gallery/sites.example.json (deploy/gallery.sh puts it in /srv/birdy/)
   chown -R birdy:birdy /srv/birdy/sites
   ```
   `authorized_keys` for the `birdy` user is `/srv/birdy/sites/.ssh/authorized_keys` (mode 600, owner birdy). One line per Pi; `install.sh` prints it on every run:
   `command="/usr/bin/rrsync /srv/birdy/sites/<site_id>",restrict ssh-ed25519 AAAA… birdy-publish@birdy`
   `rrsync` pins the key to that folder: every path the Pi sends is taken relative to it, so the key can't reach other sites (and `remote_dir = "/"` lands in `sites/<site_id>/`). `restrict` disables forwarding and ptys.
2. Caddy lives in the user's **cartaveo** stack (its Caddy owns 80/443). The user applies these two edits; Claude does not edit cartaveo files. Add to its Caddyfile
   ```
   birdy.akibeo.com {
       root * /srv/birdy
       file_server
       encode gzip
       @pages path / /*.html
       header @pages Cache-Control "no-cache"
       header /sites/* Cache-Control "public, max-age=300"
       header /*.json Cache-Control "no-cache"
   }
   ```
   and `- /srv/birdy:/srv/birdy:ro` under the `web` service's `volumes`; then `docker compose up -d web`.
3. Mac: `deploy/gallery.sh` pushes the pages (it rewrites the `style.css`/`app.js` links with a content hash, so pushes are never masked by a browser cache; the HTML and photo caching depends on the Caddy headers above — as of 2026-10-04 the live server sends no `Cache-Control` for pages or photos, so check that block if a push doesn't show). Pi: `[publish] enabled = true`, `host = "91.98.229.209"` in `birdy.toml`, `sudo systemctl restart birdy`, then `.venv/bin/birdy publish` to push immediately (restart the service before any manual `birdy publish`, or use `--rerank`, so history gets ranked) (or wait: the service's first run is right after start, then ≤ 15 min). `birdy publish` is safe while the service runs: both take an `flock` on `data/publish/.lock`; whichever comes second skips its run with a log line (exit 1 for the CLI — just rerun).

## Architecture

One Python process, one systemd service, five threads (six with Telegram alerts enabled, seven with publishing enabled) talking through narrow interfaces:

```
capture ──MotionEvent──▶ pipeline (queue + worker) ──▶ classifier
   │                          │
   └──────────────────────────┴──▶ store (SQLite WAL + photos/thumbs on disk)
                                     ▲
web (FastAPI + /settings) ───────────┘      retention (hourly thread) ──▶ store

config watcher (5 s stat of birdy.toml) ──▶ LiveConfig ◀── web POST /settings

telegram notifier (queue + worker) ◀── pipeline (live only)

publisher (mirror + rsync) ◀── pipeline / web publish buttons
```

- `birdy/capture.py` is the **only** module that imports `picamera2`. Everything else must import and test on the Mac. Keep it that way; put any new camera-touching code there.
- `birdy/motion.py` is a pure-numpy/OpenCV detector (ROI mask → running background → diff → threshold → cooldown) so it can be tested with synthetic 320×240 frames. No camera code in it.
- `birdy/classifier.py` exposes a `Classifier` protocol returning `Prediction(is_bird, confidence, species, species_confidence)`. The implementation is `TwoStageClassifier`: an ImageNet gate (`ImageNetBirdClassifier.predict` → P(bird), summed over ImageNet bird classes from MobileNetV2) plus `InatBirdClassifier` (`birdy/species.py`, 965 iNat bird species), both run on one decoded crop (`CropLoader`). Species is reported when its probability ≥ `species.threshold`; a gate-rejected frame is *rescued* (`is_bird=True`, `confidence = max(gate, species)`) when species ≥ `species.rescue_threshold`.
- The pipeline worker and the web upload box share **one** `TwoStageClassifier` (built in `cmd_run`); it serialises inference under its own lock because TFLite interpreters aren't thread-safe. Don't build a second one for the web.
- `species_checked` marks rows the species model has seen. Each pipeline sweep backfills up to 50 older `done` rows (`_backfill_species`): it never demotes a bird to non-bird, never flips a `done` row to `error` on failure (it marks it checked instead), yields as soon as a live event is queued, and logs one summary line per sweep.
- `birdy/store.py` owns the single write connection (locked, `check_same_thread=False`); capture, pipeline and retention all write through it. The web thread reads through the same locked connection. The only other writer is the one-shot `birdy purge` CLI, which is safe under WAL.
- Nothing is dropped on overload: capture inserts the row as `status='pending'` before enqueueing, and the pipeline's 5-minute backlog sweep reclassifies anything still pending.
- Manual delete: every grid tile has a Delete button (`POST /captures/{id}/delete` → `store.delete_capture`: row first, then photo + thumb; works on birds too, 404 for an unknown id), and the non-birds view has a bulk "Delete these N" (`POST /captures/delete-non-birds` with the page's `ids`; `store.delete_non_birds` only deletes non-bird/error rows, so a bird id posted there is ignored). Both take the grid's `next` (only `/` or `/?…`), use a two-tap confirm in the page, and call `publisher.forget(capture)` so a deleted published photo's slot goes to the next-best photo and the gallery drops it on the next sync.
- Retention deletes `is_bird = 0` and `status = 'error'` rows older than `retention.non_bird_days`; rows with any `published` state are kept, and bird photos are never auto-deleted.
- `birdy/web/app.py` serves the capture grid, `/status`, `/frame.jpg`, `/roi.png` (ROI overlay; still served, `birdy roi` points to it while the service holds the camera) the `/settings` page (plus `POST /settings/test-alert`, which sends a Telegram message and writes nothing), `/species` (per-species summary + an upload box that `POST /classify`s a photo and shows both models' verdict; it runs on the full uploaded image with no ROI crop and nothing is stored), a species dropdown filter on `/` (returns birds only: `is_bird = 1`), and `last_species` in `/status`. `POST /settings` writes `birdy.toml` (never the DB). When publishing is enabled the grid also gets publish badges, Publish/Hide/Auto buttons (`POST /captures/{id}/publish` with `state=forced|hidden|auto` → `publisher.set_manual`; 404 unknown id, 503 when publishing is disabled; 303 to the form's `next`, accepted only if it is `/` or starts with `/?`, else `/`) and a `filter=published` option.
- Web UI look: all pages extend `templates/_base.html` (OSD strip; the one `/status` poll, re-broadcast as a `birdy:status` event) and share `web/static/osd.css` plus the self-hosted Martian Mono font (no remote assets). Visual rules live in `DESIGN.md`, product context in `PRODUCT.md`. The grid at `/` defaults to `filter=birds`.
- `birdy/live.py` — `LiveConfig` holds the current `Config` + a version. Tunables (`motion.*`, `classifier.threshold`, `species.threshold`/`species.rescue_threshold`, `telegram.watchlist/alert_new_species/cooldown_minutes`, site/daylight, retention) are read through `live.get()` on use; `telegram.enabled` is also checked per alert, so `enabled = false` silences alerts live but `enabled = true` needs a restart (the notifier isn't built otherwise); the capture loop rebuilds its detector/gate when `live.version` changes (and `motion`, `site.lat/lon` or `daylight` differ). Camera/web/storage/model keys and `telegram.enabled/bot_token/chat_id/base_url` are read once at startup — changing them needs a restart (the watcher logs a warning).
- Camera switch: `POST /camera` (`mode=on|off|auto`, `next` = `/`, `/?…` or `/settings`) sets `RuntimeState.set_override`, shown in the grid header and at the top of `/settings`. On/off holds until the next daylight change at the time it was pressed (`DaylightGate.next_change`; with no lat/lon, until switched back), then the schedule resumes; `auto` returns to the schedule now. The capture loop asks `state.camera_wanted(gate, now)` and wakes within ~1 s on a change (`override_version`). It's in memory only: a restart drops it. State `manual_off` = switched off by hand.
- `birdy/settings.py` + `birdy/watcher.py` — the `/settings` page validates and atomically rewrites `birdy.toml` (comments are lost; `birdy.toml.example` keeps the docs) and applies it live; `ConfigWatcher` reloads the file when it's edited over SSH (invalid edits are logged and ignored until restart).
- `birdy/notify.py` — `AlertPolicy` is pure: watchlist match is case-insensitive, a first-ever species gives reason `"new"` (bypasses the cooldown), otherwise a per-species cooldown held in memory. `TelegramNotifier` has a bounded queue (20; newest dropped with a warning when full) and one worker thread that `sendPhoto`s the full-res JPEG, retries after 5/15/45 s (not on 400/401/403, which can't succeed) then logs once and drops, and falls back to a text message if the photo was purged. It is fed only by `Pipeline.process()` after a live classification is stored — never by the species backfill — and a `consider()` failure is logged without touching the row. "First-ever" means first row with that species in the DB at alert time: if the backfill labels an older photo with a species before a live capture of it, that species never gets a "new" alert (it was seen before, just not live). Queued alerts are dropped on shutdown. The web shares the same instance for `POST /settings/test-alert` (`send_test()`, synchronous; 502 with Telegram's description on error, 503 when alerts are disabled). When `telegram.enabled = false` the notifier isn't constructed at all.
- `birdy/publish.py` — `rank_day` is pure: per species and local day, `forced`/`hidden` win (forced doesn't use a slot), then the best `publish.per_species_per_day` photos at or above `publish.min_confidence` (ties: lower id) get `auto`. `Publisher.evaluate(capture)` re-ranks that species/day whenever the pipeline stores a species (live **and** backfill — historical photos may be published, unlike alerts); `set_manual(id, 'forced'|'hidden'|None)` is the review page's Publish/Hide/Auto. The first time publishing is enabled (no `data/publish/` yet) the publisher ranks all history; `birdy publish --rerank` redoes that with the current thresholds. Its thread keeps `data/publish/` in sync with the DB — web copies ≤ 1600 px (JPEG 85, OpenCV `INTER_AREA`, never upscaled) written through a temp name + `os.replace`, thumbs copied as is, `manifest.json` written atomically (per-pid temp) — then runs `rsync -az --delete --delete-after --delay-updates` (excluding `.lock` and `*.tmp*`; `--timeout=120` and `ConnectTimeout=20` bound stalls, a 1800 s wall-clock cap bounds the whole run — 3600 s for the `birdy publish` one-shot) to `user@host:remote_dir`: debounced 2 min after a change, at least every 15 min, first run right after start (reconcile). `stop()` interrupts a mirror build between photos; that run then pushes nothing and returns False. An unreadable original (0-byte, deleted) is set `hidden` and its day re-ranked so the next-best photo takes the slot; a mirror write failure (disk full) is retried next run and never hides. A run logs at info only when the published set changed (debug otherwise); an rsync failure is one warning per run. The gallery pages in `gallery/` (vanilla JS) render in the browser from `sites.json` + each site's manifest; nothing runs on the server. They are designed as a printed recording sheet (home = the daily record, species × feeders; `site.html` = a feeder's monthly sheet, species × days, from `stats.species[].days`; `species.html` = a species card); fonts are self-hosted in `gallery/fonts/` (Libre Franklin, Courier Prime, OFL), and the look is recorded in `gallery/DESIGN.md`, not the Pi UI's `DESIGN.md`.
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
- `publish.enabled/host/user/remote_dir/site_id/ssh_key` are restart-only; `min_confidence`/`per_species_per_day` are live but only affect days ranked from then on; `birdy publish --rerank` re-ranks history with the current thresholds, and that runs automatically the first time publishing is enabled. `published` states: `NULL`, `auto`, `forced`, `hidden`; the published set is `auto`+`forced`. `web_path` is the mirror copy's relative path (same as `path`). The manifest's `captured_at` is local time with offset and `site.utc_offset_minutes` lets the page compute the site's "today". `stats.species[].days` (`{local day: visits}` inside the 30-day window) feeds the monthly sheet; the pages fall back to counting published photos when a manifest (an older Pi) lacks it, and say so in the sheet's corner label.
- `FILTERS["published"]` exists; the nav shows it only when a publisher is configured.
- `remote_dir = "/"` is only safe with the rrsync-restricted key. With an unrestricted key, `rsync --delete` to `/` would be destructive — use the full path (`/srv/birdy/sites/<site_id>`).
- The publisher's ssh uses `StrictHostKeyChecking=accept-new`: if the server is re-keyed, every push fails with exit 255 until the Pi's `known_hosts` entry is removed (`ssh-keygen -R <host>` as the service user).
- `Publisher.stop()` doesn't flush: evaluations made right before shutdown are pushed by the reconcile run at the next start.
- Adding a tunable key: `config.py` dataclass + validation, `birdy.toml.example`, `settings.py` (`SettingsPatch`, `config_to_dict`, `apply_patch`, `FIELD_KEYS`, `parse_settings_form`), `settings.html` form row, and the consumer must read it via `live.get()`. `test_config_to_dict_covers_every_field` fails if `config_to_dict` is forgotten.
