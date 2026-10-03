# Birdy — Web Settings Page Design

**Date:** 2026-10-03
**Status:** approved for implementation
**Sub-project:** 1b (follows the capture core; precedes species recognition)
**Depends on:** capture core on `master` (`db65de5`), running on the Pi.

## Goal

Tune the running system from the LAN web page instead of editing `birdy.toml` over SSH and restarting: draw the ROI polygon on the live camera frame, adjust motion/classifier/site/retention settings with the live motion score visible, and have changes apply within a second without restarting the camera.

## Decisions made

| Question | Decision |
|---|---|
| Editable settings | Tuning subset only: `motion.*`, `classifier.threshold`, `site.lat/lon`, `daylight.margin_minutes`, `retention.non_bird_days`. Camera sizes, JPEG quality, web host/port, data dir, model paths stay TOML-only (restart required; shown greyed on the page). |
| ROI input | Click-to-place polygon on the live lores frame; drag to move, click a corner to delete, min 3 points. |
| Auth | None (LAN only, same as the read-only page). Revisit with the publishing sub-project. |
| Source of truth | `birdy.toml` remains the single config. The web writes it; the process hot-reloads. No settings in SQLite. |
| Apply semantics | One **Save**: validate → write file atomically → apply live. No separate "try" mode; the live score shows the effect. |

## Architecture

No module boundaries change. Three additions and small edits at the consumers:

```
web (GET/POST /settings, GET /frame.jpg)
      │ save_settings()            ┌──────────────┐
      ▼                            │ ConfigWatcher│ (stat birdy.toml every 5 s)
 settings.py ──writes──▶ birdy.toml│   thread     │
      │                            └──────┬───────┘
      │ live.replace(cfg)                 │ live.replace(cfg) on external edit
      ▼                                   ▼
            LiveConfig (lock, Config, version)
      ▲            ▲             ▲              ▲
   capture      pipeline     classifier     retention
 (rebuild on   (roi_bbox    (threshold      (non_bird_days
  version Δ)    per event)   per classify)   per run)
```

### `birdy/live.py` — `LiveConfig`
- `LiveConfig(config: Config)`; `get() -> Config` (frozen snapshot); `replace(config) -> None` bumps `version`; `version: int` property. Lock-protected; trivially cheap to call per frame.
- `cmd_run` constructs it once and passes it (instead of `Config`) to capture, pipeline, retention and the web app. Modules keep their current signatures for `Config` values they read once at startup (camera sizes, paths); only the tunable reads go through `live.get()`.

### `birdy/settings.py` — `save_settings(path, current, patch) -> Config`
- `patch` is a `SettingsPatch` dataclass with exactly the editable fields (`roi: tuple[Point, ...]`, `pixel_threshold`, `min_area`, `background_alpha`, `cooldown_seconds`, `classifier_threshold`, `lat`, `lon`, `margin_minutes`, `non_bird_days`). Unknown fields can't exist by construction.
- Builds a plain dict from `current` (every section, including non-editable keys, so nothing is lost), overlays the patch, serialises with `tomli-w`, writes `<path>.tmp`, validates it by calling `load_config(tmp)`, then `os.replace(tmp, path)`. On `ConfigError` the temp file is deleted and the error re-raised; the original file is untouched.
- The written file starts with a comment: `# Written by the birdy settings page. Comments live in birdy.toml.example.`
- Paths in the written file are written exactly as they were loaded (absolute) — `load_config` resolves relative paths against the config directory, so writing absolute paths is safe and unambiguous.

### `birdy/watcher.py` — `ConfigWatcher`
- Thread: every 5 s, `stat()` the file; if mtime changed → `load_config()`; on `ConfigError`/`TOMLDecodeError`/`OSError` → `log.error`, keep the running config. After loading, the watcher compares the new `Config` to `live.get()`; equal configs (e.g. the page's own write) are not re-applied. If camera/web/storage/model keys changed it logs that a restart is required (they are still loaded into `live`, but the capture loop keeps the camera config it started with). Otherwise `live.replace()` + `log.info`.
- Same start/stop pattern as `RetentionJob`.

### Hot-reload at the consumers
| Consumer | Change |
|---|---|
| `capture.run_capture` | Takes `live: LiveConfig`. Camera config (`camera.*`) is fixed per process: read once when `run_capture` starts and reused for every camera session. At the top of each frame compares `live.version` to the version last seen; if different, and the `motion`/`site`/`daylight` sections differ from those the current `MotionDetector`/`DaylightGate` were built from, rebuilds both from `live.get()` (fresh warm-up ≈ 1 s) and logs "settings applied (vN)". A version bump that only touched other sections (e.g. retention) does not reset the detector. |
| `Pipeline` | Takes `live`; computes `roi_bbox(live.get().motion.roi)` per event instead of storing it. |
| `ImageNetBirdClassifier.classify` | Gains a `threshold: float | None = None` argument overriding the constructor default; `Pipeline` passes `live.get().classifier.threshold`. (`Classifier` Protocol updated accordingly.) |
| `RetentionJob` | Takes `live`; reads `non_bird_days` per run. |
| `DaylightGate` | Unchanged; rebuilt by capture on version change. |

## Web

### `GET /settings`
One page (`settings.html`), linked from the header of `/`. Plain HTML + ~100 lines of vanilla JS, no framework, no build step.

**ROI editor**
- `<canvas>` sized to the lores frame × 2 (640×480 by default; scales down to the viewport width on phones, keeping aspect). Background: `/frame.jpg`, re-fetched every 1 s. Overlay: the polygon and its corners (12 px hit radius).
- Interactions (pointer events, so mouse and touch): click empty space → append corner; drag corner → move; click corner (no drag) → delete (refused below 3 points, with a hint); **Reset** → reload saved polygon. A press that moves less than 3 lores px is a tap (delete), not a drag. When corners overlap, the nearest corner within the hit radius wins. `pointercancel` (e.g. the browser taking over a touch for scrolling) ends the drag.
- Coordinates are kept in lores pixels (integers) and submitted as JSON in a hidden `roi` field.
- Live readout under the canvas: `motion score` and `min_area` (score polled from `/status` every 1 s), colour turns green when score ≥ min_area.

**Form** (groups mirror the TOML)
- Motion: `pixel_threshold` 0–100 step 1; `min_area` 0–20000 step 10; `background_alpha` 0.01–1 step 0.01; `cooldown_seconds` 0–60 step 0.5 — `<input type=range>` paired with a numeric input, kept in sync.
- Classifier: `threshold` 0–1 step 0.01.
- Site / daylight: `lat`, `lon` (numeric, both optional — both empty disables the gate; one empty is a validation error), `margin_minutes` 0–180.
- Retention: `non_bird_days` 1–365.
- Read-only block listing the non-editable keys with their current values and the note "edit birdy.toml and restart".
- **Save** → `POST /settings`. The form has `novalidate`: values hand-set in `birdy.toml` may be off the slider step or range, and the server's validation (with field-level messages) is the authority.

### `POST /settings`
- Form-encoded. Parses fields → `SettingsPatch` (type errors → 422 with a message on that field) → `save_settings()` → `live.replace()` → 303 redirect to `/settings?saved=1`, which shows a "Saved — applied live (vN)" banner.
- `ConfigError` → 422, page re-rendered with submitted values. The message always appears in the error banner at the top of the page, and additionally beside the field whose key the message names (numeric fields, ROI, lat/lon).
- `OSError` on write → 500 page with the OS error; running config unchanged.

### `GET /frame.jpg`
- Latest lores frame from `RuntimeState.lores_frame()` as JPEG (quality 80), no overlay. 404 before the first frame. `Cache-Control: no-store`.

## Configuration

No new keys. `tomli-w` added to `pyproject.toml` dependencies.

## Error handling summary

| Case | Behaviour |
|---|---|
| Invalid value / polygon from the page | 422, nothing written, field-level message |
| Disk/permission error writing the file | 500, running config unchanged |
| External edit produces a bad file | Logged, running config kept; a restart would exit 2 (existing behaviour) |
| Camera-size change via an external (SSH) edit | Watcher logs that a restart is required; the new config is loaded into `live`, but the running capture loop ignores the camera change and keeps its startup camera config until restart. (Edits from the page cannot change camera sizes, and `load_config` validates ROI against `lores_size`.) |

## Testing (Mac)

- `settings.py`: round-trip (save then `load_config` equals merged config); invalid patch → `ConfigError`, temp file removed, original byte-identical; absolute paths preserved; header comment present; `lat` without `lon` rejected.
- `live.py`: `get()` returns latest; `replace()` increments `version`; thread-safety smoke (writers/readers).
- `watcher.py`: modify file → `live` updated once; rewrite with identical content → no reload; corrupt file → error logged, `live` unchanged; restored file → reload.
- Web: `GET /settings` shows current values and polygon JSON and the read-only block; `POST` valid → file updated, `live.version` bumped, 303; `POST` with 3 invalid cases (2-point polygon, `min_area=-1`, `lat` only) → 422 with field message and file untouched; `GET /frame.jpg` 404 then JPEG magic bytes.
- `pipeline`: threshold and ROI from `live` are passed through (fake classifier records the threshold it was called with; change `live`, next event sees the new value).
- `classifier.classify(threshold=)` override.
- Capture-loop rebuild: Pi-only; verified by saving a new `min_area` and watching the score colour flip on the page, and by `journalctl` showing "settings applied (vN)".

## Out of scope

Auth/PIN, editing camera/web/storage/model keys, undo history, multi-user conflict handling, preview of unsaved settings, species settings (sub-project 2).
