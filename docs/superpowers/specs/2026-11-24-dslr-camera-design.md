# DSLR second camera (Canon EOS 50D) — design

Date: 2026-11-24. Builds on the capture core, the detector gate (`2026-11-06-detector-gate-design.md`), BioCLIP species (`2026-11-06-bioclip-species-design.md`) and visits (`2026-11-07-visits-and-corrections-design.md`).

## Why

The Pi Camera v2.1 photographs through a window at 8 MP with a fixed wide lens; a bird at the feeder is a small part of the frame. A Canon EOS 50D (15 MP, real lens) sits at the same window on AC power and is connected to the Pi over USB.

Measured on the Pi on 2026-11-24 with `gphoto2` from apt, camera in Av, manual focus, Medium Fine JPEG:

| Test | Result |
|---|---|
| `--capture-image-and-download` (3456×2304, ~3 MB) | 1.1–1.3 s per invocation |
| 3 frames in one invocation (`-F 3 -I 1`) | 3.1 s |
| `--capture-preview` (live view, 1024×680) | ~1.8 s |
| any command's connection overhead (`--get-config iso`) | 0.33 s |

Live view is slower than a real shot and far smaller, so it is not used. The shutter counter read 22,764 after the tests (rating ~100,000). Over the previous 14 days the Pi stored 1–14 bird photos a day (3.2 per visit on average, 16 at most) and ~80 motion triggers a day. Firing on every bird verdict is ≤ ~5,000 frames a year; firing on every motion trigger would be ~30,000.

**Decisions** (taken with the user):

1. The Pi camera keeps motion detection and the bird gate. The 50D fires **once per live bird verdict**, never on plain motion.
2. When a bird is found in the 50D frame, that frame is the capture's **main photo** (grid, lightbox, gallery, Telegram) and the species model runs on it. Otherwise the Pi photo stays main.
3. Everything runs **sequentially on the existing pipeline thread**. A second fast thread is deferred (see *Later*).

## Scope

In: a `gphoto2` wrapper, a staged classifier API (verdict, then name), the pipeline's DSLR step, stored DSLR columns, a "main photo" rule for every reader, retention of unused DSLR frames, a `[dslr]` config section, a `birdy dslr` CLI, status display, install/docs.

Added after the first version (2026-11-24): a **Take a test frame** button on `/settings` with a draggable region box, and a manual **Take a photo** always fires the DSLR. Also added that day: `dslr.auto_shutter` — the shutter speed is picked before each shot from the Pi camera's lux reading and a self-correcting exposure constant (details in `CLAUDE.md`), because a fixed 1/500 s measured far too dark at sunset (brightness 12 of 255 at f/4, ISO 1600).

Out: manually promoting a bird-less DSLR frame, bursts, live view, setting exposure/ISO/size from birdy (set on the camera body), RAW files, a second worker thread.

With `dslr.enabled = false` (the default) the service behaves exactly as before: the pipeline takes its Pi-only code path and `gphoto2` is never run. `dslr.enabled` is a live switch on `/settings` (changed 2026-11-24, after the first version made it restart-only): the `Dslr` object is always built, the pipeline reads the switch per capture, and switching it on from the page calls `info()` once so the status line is current.

## Camera wrapper — `birdy/dslr.py`

The only module that runs `gphoto2`. No other module imports `subprocess` for the camera.

```python
class DslrError(Exception): ...

@dataclass(frozen=True)
class DslrStatus:
    state: str            # 'ready' | 'not_connected' | 'error' | 'cap_reached' | 'unknown' (before the first check)
    shots_today: int
    last_error: str | None

class Dslr:
    def __init__(self, store: Store, live: LiveConfig, runner=subprocess.run, now=datetime.now) -> None: ...
    def capture(self, dest: Path) -> float: ...   # seconds taken; raises DslrError
    def info(self) -> dict[str, str]: ...         # config values by name; raises DslrError
    def status(self) -> DslrStatus: ...
```

- `capture(dest)` runs `gphoto2 --capture-image-and-download --filename <dest>.tmp --force-overwrite` with a 15 s timeout, then `os.replace`s the temp file onto `dest`. It succeeds only when the exit code is 0 **and** the temp file exists and is non-empty (gphoto2 can exit 0 after printing an error). On any failure the temp file is removed and `DslrError` carries gphoto2's last stderr line.
- Daily cap: before running, `capture` raises `DslrError("daily cap reached")` when `store.count_dslr_shots(since=local midnight) >= dslr.daily_cap`. The count is read from the DB, so it survives restarts.
- `info()` runs one `gphoto2` invocation with `--get-config` for `capturetarget`, `imageformat`, `autopoweroff`, `focusmode`, `autoexposuremode`, `iso`, `shuttercounter` and parses each `Current:` line.
- `status()` is in memory: `ready` after a successful capture or `info()`; `not_connected` when gphoto2 reported no camera (`*** Error: No camera found` / "Could not detect any camera"); `cap_reached`; `error` otherwise. It starts from one `info()` call at service start.
- Logging: one warning per **distinct** error message (like the detector and species failures), one info line when the camera comes back.
- Single caller: the pipeline thread. The `birdy dslr` CLI may run while the service does; when both claim the USB device at once, one fails with `DslrError` and its capture falls back to the Pi photo.

## Staged classifier — `birdy/classifier.py`

`TwoStageClassifier` gains three methods; `classify_image` is rewritten as `verdict` + `name` and keeps its behaviour and signature (web upload, CLI and backfill are unchanged).

```python
@dataclass(frozen=True)
class Verdict:
    is_bird: bool
    confidence: float                     # max(p_gate, p_det)
    detector_confidence: float
    detector_box: NormBox | None

def load_crop(self, image_path, roi_bbox, pad=None) -> np.ndarray      # CropLoader.load, RGB
def verdict(self, img_rgb, threshold=None) -> Verdict                  # gate + detector, under the lock
def locate(self, img_rgb) -> Detection | None                          # detector only, under the lock
def name(self, img_rgb) -> tuple[list[tuple[SpeciesLabel, float]], np.ndarray | None]   # species top-5 + embedding, under the lock
```

The lock is released between the stages (the DSLR capture happens outside it), so a web upload can run in between. `name` does the candidate refresh and the once-per-error handling that `classify_image` does today.

## Pipeline — `birdy/pipeline.py`

`Pipeline.__init__` gains `dslr: Dslr | None = None`. With `dslr is None`, `process` is today's code. Otherwise, for a live event:

```
crop_rgb = classifier.load_crop(path, crop.bbox, crop.pad)       # the Pi crop, as today
v = classifier.verdict(crop_rgb, threshold)
frame = None
if v.is_bird and now - captured_at <= FRESH (10 s):
    try: dslr.capture(photos_dir / dslr_rel)                      # ~1.2 s
    except DslrError: pass                                        # logged by Dslr
    else: frame = RGB decode of the file
box = None
if frame is not None:
    region_rgb = frame cropped to cfg.dslr.region
    det = classifier.locate(region_rgb)
    if det and det.score >= cfg.detector.threshold:
        box = det.box mapped from the region to fractions of the full frame
if v.is_bird:
    species_rgb = square crop of the frame around box (below) if box else crop_rgb
    top, emb = classifier.name(species_rgb)
store.store_verdict(..., dslr_path=dslr_rel if frame is not None else None, dslr_box=box)
visits.photo_classified(...)                                      # unchanged
```

- `dslr_rel` is the Pi photo's relative path with `_eos` before the extension (`2026/11/24/101502_123_eos.jpg`), so it lives beside the Pi photo and `/photos` serves it.
- **Species crop on the DSLR frame:** a square centred on the box, side = 2 × the box's longer side in pixels, at least 224 px, shifted to stay inside the frame (the motion-crop rule, in DSLR pixels).
- `FRESH = 10 s` keeps the backlog sweep, restarts and the backfill from firing the camera at an empty feeder. A manual **Take a photo** shot fires the DSLR whatever the verdict (`MotionEvent.manual`); its frame is the capture's main photo even with no bird in it (`dslr_main = 1`; changed the same day, after first storing it as an unused frame).
- A DSLR frame that cannot be decoded counts as a failed capture (file removed, Pi photo stays main).
- When a row is promoted (`box` set), `store_verdict` also makes the thumbnail from the DSLR frame and points `thumb_path` at it (the Pi thumbnail file is removed).
- Log line: `capture 812: 50D 1.21 s, bird 0.71` or `capture 812: 50D 1.19 s, no bird in frame`.
- **The main photo never changes after first classification.** The DSLR only fires on the live path, and promotion is written with the verdict, before `Visits` ranks or publishes anything. So no publish slot or alert ever moves because of it.

### Backfill

`_backfill` re-gates on the Pi crop as today. For a promoted row (`dslr_box` set) whose species must be recomputed (`species_checked = 0`, e.g. a model swap), the species and embedding come from the DSLR species crop instead of the Pi crop. If the DSLR file is unreadable, it falls back to the Pi crop. The backfill never fires the camera.

## Store — `birdy/store.py`

New `captures` columns (added by the existing `ALTER TABLE` loop):

| Column | Meaning |
|---|---|
| `dslr_path TEXT` | relative path of the DSLR frame under `photos/`; NULL = no shot |
| `dslr_x, dslr_y, dslr_w, dslr_h REAL` | the bird's box as fractions of the full DSLR frame; NULL = no bird found |

- `Capture` gains `dslr_path`, `dslr_box` and a property `photo`: `dslr_path` when `dslr_box` is set, else `path`. `path` always stays the Pi photo (motion crop, re-gating).
- `store_verdict` takes `dslr_path` and `dslr_box` (default: leave unchanged).
- `count_dslr_shots(since)` counts rows with `dslr_path IS NOT NULL` and `captured_at >= since`.
- `delete_capture`, `delete_non_birds` and `purge_non_birds` also remove the DSLR file and its thumbnail.
- `purge_dslr_unused(cutoff)` deletes the DSLR file of rows with `dslr_path` set, `dslr_x IS NULL` and `captured_at < cutoff`, and sets `dslr_path` to NULL. `RetentionJob` and `birdy purge` call it with the `retention.non_bird_days` cutoff. Promoted frames are never auto-deleted (they are bird photos).
- `SpeciesRow.best_path` uses the same rule in SQL (`CASE WHEN dslr_x IS NOT NULL THEN dslr_path ELSE path END`).

## Readers of the main photo

Each of these switches from `c.path` to `c.photo`:

- **Grid and lightbox** (`web/app.py`, templates): tile link, lightbox image. A promoted tile shows a muted `50D` stamp and a **Pi photo** link to the Pi photo; a row with an unused DSLR frame shows a **50D frame** link to it.
- **Model view** (`/captures/{id}/crop.jpg`) and `birdy classify --capture`: for a promoted row they show the DSLR species crop (crop kind `dslr`); otherwise unchanged.
- **Publisher** (`publish.py`): the mirror copy is made from `c.photo` and `web_path` is that relative path. The ≤ 1600 px downscale already applies.
- **Telegram** (`visits.py` → `notify.py`): the alert sends `c.photo`. Telegram's `sendPhoto` rejects files over 10 MB or with width + height over 10,000, so when the file is over 8 MB or its sides sum to more than 10,000 the notifier sends a ≤ 2560 px JPEG-85 copy made in memory.

## Config — `[dslr]`

| Key | Default | Applies | Meaning |
|---|---|---|---|
| `enabled` | `false` | live | use the staged path and fire the DSLR; a switch on `/settings` |
| `region` | `[0.0, 0.0, 1.0, 1.0]` | live | `x, y, w, h` fractions of the DSLR frame the detector searches |
| `daily_cap` | `200` | live | maximum DSLR shots per local day (1–2000) |

Validation: `region` values in [0, 1], `w` and `h` > 0, `x + w` and `y + h` ≤ 1. All three follow the "adding a tunable key" checklist (`config.py`, `birdy.toml.example`, `settings.py`, `settings.html`); on `/settings`, `enabled` is a switch, `daily_cap` a number field and `region` one text field of four fractions.

The region exists because the detector resizes its input to 320×320: a bird that is a few percent of a 15 MP frame would vanish. It must frame the feeder tightly enough that a bird is at least ~10 % of the region's width.

## CLI — `birdy dslr`

- `birdy dslr info` prints the `info()` values and warns when `capturetarget` is not `Internal RAM`, `autopoweroff` is not `0`, or `focusmode` is not `Manual`.
- `birdy dslr shot [--out FILE]` takes one frame (refused when the daily cap is reached; test frames are not counted, since the cap counts stored captures), saves it (default `data/dslr-test.jpg`) plus a copy with the region drawn (`…-region.jpg`), and prints the capture time, the frame size and the detector's result inside the region. This is the tool for aiming, focusing and choosing `region`.

Both work while the service runs.

## Status

`/status` gains `dslr: {state, shots_today}` (absent when disabled). `/settings` shows one line under the camera switch: `50D: ready · 4 shots today` / `not connected` / `daily cap reached`.

## Camera setup (on the body, once)

AC adapter; auto power off **Off**; lens on **MF**, focused on the feeder; Large Fine JPEG (no RAW); a shutter-priority or manual exposure fast enough for birds (1/500 s or faster) with Auto ISO. No card is needed: `capturetarget = Internal RAM` downloads straight to the Pi. On a Pi OS with a desktop, `gvfs-gphoto2-volume-monitor` can claim the camera; the Pi runs Lite, where this does not happen.

## Install and docs

`deploy/install.sh` adds `gphoto2` to its apt packages. `CLAUDE.md` gets the module, the main-photo rule, the config keys and the CLI. `birdy.toml.example` documents `[dslr]`.

## Tests (all on the Mac)

- `test_dslr.py`: a fake `runner` — success writes the file atomically; exit ≠ 0, exit 0 with no file, an empty file and a timeout each raise `DslrError` and leave no temp file; "no camera" maps to `not_connected`; the daily cap blocks before the runner is called; `info()` parsing; one warning per distinct error.
- `test_classifier.py`: `verdict` + `name` give the same `Prediction` as `classify_image`; `locate` returns the detector's best bird.
- `test_pipeline.py` with a fake `Dslr`: non-bird never fires; a fresh bird fires once; a stale bird (> 10 s) does not; bird found → `dslr_box` set, species crop from the DSLR frame, thumbnail replaced; no bird in the frame → `dslr_path` set, `dslr_box` NULL, species from the Pi crop; `DslrError` and an undecodable file → today's result; `dslr=None` → today's calls only; backfill of a promoted row names from the DSLR crop and never fires.
- `test_store.py`: migration adds the columns; `Capture.photo`; `count_dslr_shots`; delete and purge remove DSLR files; `purge_dslr_unused`.
- Web, publish, notify: a promoted row serves, mirrors and alerts with the DSLR file; the Telegram downscale rule.
- Settings: the three keys round-trip (`test_config_to_dict_covers_every_field`), region validation errors.

## Rollout

1. Deploy with `enabled = false`; nothing changes.
2. Mount and aim the 50D, set it up as above, then `birdy dslr info` and `birdy dslr shot` until the region frames the feeder.
3. Switch **Use the 50D** on at `/settings` (no restart).
4. After a week, compare `dslr_path IS NOT NULL` with `dslr_x IS NOT NULL` over that week: the share of shots with no bird found.

## Later

If more than about half of the shots have no bird in the frame, the bird is leaving before the shutter fires. The fix is a second worker thread that runs verdict + DSLR ahead of the species queue (every shot then lands ~1.5–2 s after its Pi photo, also mid-visit, where the sequential design can take 4–5 s). It needs split model locks and competes with BioCLIP for the Pi's cores, so it waits for that measurement.

## Tiled search (added 2026-11-25)

The detector works on a 320 px input. With a region of about 2900 x 2000 px of a Large frame, a
bird on the dish is about 20 x 40 px after the resize and was missed (capture 5981: "no bird in
frame" with a Black Redstart in plain view). `Pipeline._find_bird` therefore searches the whole
region first and, when that finds nothing at `detector.threshold`, overlapping squares of about
960 px (`dslr.search_tiles`, 25 % overlap); the best tile's box is the bird. Measured on the Pi on
13 stored frames: 12 tiles take 2.1 s (only when the whole-region pass fails), the ten empty
frames score at most 0.15, the missed frame 0.69. A capture's main photo is still fixed at first
classification, so frames from before this change are not promoted.

Since 2026-11-27 a bird in the DSLR frame needs a score of at least `DSLR_BIRD_MIN = 0.4`
(`max(detector.threshold, 0.4)`), in the whole region and in the tiles alike: capture 6077's frame
was empty (the bird had left in the ~1.1 s between the motion trigger and the shutter), a leaf
scored 0.21 and the empty frame became the main photo. On the 31 frames stored that day the 11
empty ones score at most 0.21 and the 20 with a bird at least 0.61 in their best tile; four real
birds score only 0.26-0.39 in the whole region, so those frames now go through the tiles too.
