# DSLR Second Camera (EOS 50D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fire a USB-connected Canon EOS 50D once per live "bird" verdict and use its frame as the capture's main photo (and species source) when a bird is found in it.

**Architecture:** A new `birdy/dslr.py` wraps the `gphoto2` command and holds the pure crop geometry. `TwoStageClassifier` gains a staged API (`verdict` → `locate` → `name`) so the pipeline can fire the camera between the bird verdict and the 3.5 s species run. The store keeps the Pi photo in `path` and adds `dslr_path` / `dslr_x..h`; `Capture.photo` is what every display reader uses. With `dslr.enabled = false` nothing changes.

**Tech Stack:** Python 3.11 (Mac) / 3.13 (Pi), OpenCV, SQLite, FastAPI + Jinja, pytest, `gphoto2` CLI (apt) on the Pi only.

**Spec:** `docs/superpowers/specs/2026-11-24-dslr-camera-design.md` — read it first.

**Conventions for every task**

- Run tests with `.venv/bin/pytest` from the repo root. The whole suite runs on the Mac; no camera is needed.
- Never import `subprocess` for the camera outside `birdy/dslr.py`. Never import `picamera2` outside `birdy/capture.py`.
- All image arrays handed to models are **RGB**.
- Commit after each task with the message given. End every commit message with the line `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## File map

| File | Change |
|---|---|
| `birdy/config.py` | `DslrConfig`, `[dslr]` parsing |
| `birdy/settings.py`, `birdy/web/templates/settings.html`, `birdy/web/app.py` | `dslr.region`, `dslr.daily_cap` on `/settings`; status line; main-photo readers |
| `birdy/watcher.py` | `dslr.enabled` is restart-only |
| `birdy.toml.example` | `[dslr]` docs |
| `birdy/store.py` | columns, `Capture.photo`, `store_verdict` DSLR args, counts, deletes, `purge_dslr_unused` |
| `birdy/dslr.py` (new) | `Dslr`, `DslrError`, `DslrStatus`, geometry helpers |
| `birdy/classifier.py` | `Verdict`, `load_crop`, `verdict`, `locate`, `name`, `prediction` |
| `birdy/pipeline.py` | DSLR step in `process`, promoted rows in `_backfill` |
| `birdy/retention.py` | purge unused DSLR frames |
| `birdy/publish.py`, `birdy/visits.py`, `birdy/notify.py` | use `c.photo`; Telegram size rule |
| `birdy/cli.py` | `birdy dslr info|shot`, wiring in `cmd_run`, `classify --capture` |
| `deploy/install.sh`, `CLAUDE.md` | `gphoto2`, docs |
| `tests/test_dslr.py` (new) + existing test modules | tests per task |

---

### Task 1: `[dslr]` config, settings page fields, watcher

**Files:**
- Modify: `birdy/config.py`, `birdy/settings.py`, `birdy/watcher.py`, `birdy/web/app.py` (`settings_context`), `birdy/web/templates/settings.html`, `birdy.toml.example`
- Test: `tests/test_config.py`, `tests/test_settings.py`, `tests/test_web.py`, `tests/test_watcher.py`

- [ ] **Step 1: Write the failing config tests** — append to `tests/test_config.py` (it already has `write(tmp_path, text)` and `MINIMAL`):

```python
def test_dslr_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.dslr.enabled is False
    assert cfg.dslr.region == (0.0, 0.0, 1.0, 1.0)
    assert cfg.dslr.daily_cap == 200


def test_dslr_values(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + "\n[dslr]\nenabled = true\nregion = [0.2, 0.1, 0.5, 0.6]\ndaily_cap = 50\n"))
    assert cfg.dslr.enabled is True and cfg.dslr.region == (0.2, 0.1, 0.5, 0.6) and cfg.dslr.daily_cap == 50


@pytest.mark.parametrize("line", [
    "region = [0.2, 0.1, 0.5]",          # three values
    "region = [0.6, 0.1, 0.5, 0.6]",     # x + w > 1
    "region = [0.2, 0.1, 0.0, 0.6]",     # zero width
    "region = [-0.1, 0.1, 0.5, 0.6]",    # negative
    'region = "all"',
    "daily_cap = 0",
    "daily_cap = 2001",
    "daily_cap = 1.5",
    'enabled = "yes"',
])
def test_dslr_rejects(tmp_path, line):
    with pytest.raises(ConfigError, match="dslr\\."):
        load_config(write(tmp_path, MINIMAL + f"\n[dslr]\n{line}\n"))
```

- [ ] **Step 2: Run them** — `.venv/bin/pytest tests/test_config.py -k dslr -v`. Expected: FAIL (`Config` has no `dslr`).

- [ ] **Step 3: Implement in `birdy/config.py`.** Add after `PublishConfig`:

```python
@dataclass(frozen=True)
class DslrConfig:
    enabled: bool                               # restart-only; the second camera (gphoto2 over USB)
    region: tuple[float, float, float, float]   # live; x, y, w, h fractions of the DSLR frame the detector searches
    daily_cap: int                              # live; maximum DSLR shots per local day
```

Add `dslr: DslrConfig` as the last field of `Config`. Add this helper next to `_roi`:

```python
def _region(v: Any, name: str) -> tuple[float, float, float, float]:
    if not (isinstance(v, list) and len(v) == 4
            and all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in v)):
        raise ConfigError(f"{name} must be [x, y, w, h] as fractions of the frame")
    x, y, w, h = (float(i) for i in v)
    ok = all(math.isfinite(i) for i in (x, y, w, h)) and x >= 0 and y >= 0 and w > 0 and h > 0 \
        and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
    if not ok:
        raise ConfigError(f"{name} must lie inside the frame: values 0..1, w and h > 0, x + w <= 1, y + h <= 1")
    return (x, y, w, h)
```

In `load_config`, before `return Config(`:

```python
    ds = raw.get("dslr", {})
    dslr_enabled = _bool(ds, "dslr", "enabled", False)
    dslr_region = _region(ds.get("region", [0.0, 0.0, 1.0, 1.0]), "dslr.region")
    dslr_cap = _num(ds, "dslr", "daily_cap", 200, int)
    _check(1 <= dslr_cap <= 2000, "dslr.daily_cap must be between 1 and 2000")
```

and add `dslr=DslrConfig(enabled=dslr_enabled, region=dslr_region, daily_cap=dslr_cap),` as the last argument of `Config(...)`.

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_config.py -v`. Expected: the dslr tests PASS; `tests/test_settings.py::test_config_to_dict_covers_every_field` will fail until Step 6 (that is expected).

- [ ] **Step 5: Write the failing settings tests.** In `tests/test_settings.py`: add to `default_patch`'s `base` dict

```python
        dslr_region=config.dslr.region,
        dslr_daily_cap=config.dslr.daily_cap,
```

add to the module's `FORM` dict (near `"per_species_per_day": "3",`)

```python
    "dslr_region": "0, 0, 1, 1",
    "dslr_daily_cap": "200",
```

and append:

```python
def test_dslr_fields_parse_and_save(config_path, config):
    patch = parse_settings_form({**FORM, "dslr_region": "0.2, 0.1 0.5,0.6", "dslr_daily_cap": "40"})
    assert patch.dslr_region == (0.2, 0.1, 0.5, 0.6) and patch.dslr_daily_cap == 40
    new = save_settings(config_path, config, patch)
    assert new.dslr.region == (0.2, 0.1, 0.5, 0.6) and new.dslr.daily_cap == 40 and new.dslr.enabled is False


@pytest.mark.parametrize("value", ["", "0.2, 0.1, 0.5", "a, b, c, d"])
def test_dslr_region_form_errors(value):
    with pytest.raises(ConfigError, match="dslr.region must be four numbers"):
        parse_settings_form({**FORM, "dslr_region": value})


def test_dslr_region_out_of_frame_is_rejected_on_save(config_path, config):
    patch = parse_settings_form({**FORM, "dslr_region": "0.8, 0, 0.5, 1"})
    with pytest.raises(ConfigError, match="dslr.region must lie inside the frame"):
        save_settings(config_path, config, patch)
```

In `tests/test_web.py`, add the same two entries to the dict returned by `_base_form()` and append:

```python
def test_settings_page_has_dslr_fields(client):
    html = client.get("/settings").text
    assert 'name="dslr_region"' in html and 'name="dslr_daily_cap"' in html


def test_settings_post_saves_dslr(client, live):
    r = post_settings(client, dslr_region="0.1, 0.2, 0.3, 0.4", dslr_daily_cap="25")
    assert r.status_code in (200, 303)
    assert live.get().dslr.region == (0.1, 0.2, 0.3, 0.4) and live.get().dslr.daily_cap == 25
```

In `tests/test_watcher.py`, find the helper near line 146 that builds a `SettingsPatch` from the current config and add `dslr_region=cur.dslr.region, dslr_daily_cap=cur.dslr.daily_cap,`.

- [ ] **Step 6: Implement in `birdy/settings.py`.**

`FIELD_KEYS`: add `"dslr_region": "dslr.region",` and `"dslr_daily_cap": "dslr.daily_cap",`.

`SettingsPatch`: add at the end

```python
    dslr_region: tuple[float, float, float, float]
    dslr_daily_cap: int
```

`config_to_dict`: add after the `"publish"` entry

```python
        "dslr": {
            "enabled": config.dslr.enabled,
            "region": list(config.dslr.region),
            "daily_cap": config.dslr.daily_cap,
        },
```

`apply_patch`: add before `return data`

```python
    data.setdefault("dslr", {}).update(region=list(patch.dslr_region), daily_cap=patch.dslr_daily_cap)
```

Add next to `_lines`:

```python
def _region(form: Mapping[str, str]) -> tuple[float, float, float, float]:
    parts = form.get("dslr_region", "").replace(",", " ").split()
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        vals = ()
    if len(vals) != 4:
        raise ConfigError(f"{FIELD_KEYS['dslr_region']} must be four numbers: x, y, w, h")
    return vals
```

`parse_settings_form`: add `dslr_region=_region(form),` and `dslr_daily_cap=_num(form, "dslr_daily_cap", int),`.

- [ ] **Step 7: Web context and template.** In `birdy/web/app.py` `settings_context`, add to the `values` dict:

```python
                "dslr_region": ", ".join(f"{x:g}" for x in cfg.dslr.region),
                "dslr_daily_cap": cfg.dslr.daily_cap,
```

and to `"readonly"`: `"dslr.enabled": cfg.dslr.enabled,`.

In `birdy/web/templates/settings.html`, insert this section directly before the `<section class="group">` whose heading is `System`:

```html
  <section class="group">
    <header><h2 class="head">50D</h2><p>The second camera, fired once per bird</p></header>
    <div class="row {{ 'error' if error_field == 'dslr_region' else '' }}">
      <label for="dslr_region">Feeder region</label>
      <span class="val"><input type="text" id="dslr_region" name="dslr_region" value="{{ v['dslr_region'] }}" placeholder="0, 0, 1, 1"></span>
      {{ explain("The part of the 50D frame searched for the bird: x, y, width, height as fractions of the frame. Frame the feeder tightly; check it with birdy dslr shot.", "dslr.region") }}
    </div>
    {% if error_field == 'dslr_region' %}<div class="hint error">{{ error_message }}</div>{% endif %}
    {{ num("dslr_daily_cap", "dslr.daily_cap", "Shots per day, at most",
           "The 50D stops firing for the day after this many shots; the Pi camera carries on.", 1, 2000, 1, "shots") }}
  </section>
```

- [ ] **Step 8: Watcher.** In `birdy/watcher.py` `_restart_only`, add `cfg.dslr.enabled,` as the last element of the returned tuple, and change the warning text to `"%s: camera/web/storage/model/telegram/publish/dslr settings changed; restart birdy to apply them"`. Append to `tests/test_watcher.py`:

```python
def test_dslr_enabled_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + "\n[dslr]\nenabled = true\n")
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart birdy to apply them" in caplog.text
```

- [ ] **Step 9: Document in `birdy.toml.example`** — append:

```toml
[dslr]
# A second camera: a Canon EOS 50D (or another gphoto2 camera) on USB, at the same window.
# The Pi camera still detects motion and decides "bird"; the DSLR then fires once, and its
# frame becomes the capture's photo when a bird is found in it. Needs `gphoto2` (install.sh).
enabled = false                   # restart-only
region = [0.0, 0.0, 1.0, 1.0]     # live; x, y, w, h (fractions of the DSLR frame) searched for the bird.
                                  # Frame the feeder tightly: a bird should be >= ~10 % of the region's width.
                                  # `birdy dslr shot` saves a test frame with this region drawn.
daily_cap = 200                   # live; DSLR shots per local day (1-2000), a guard for the shutter
```

- [ ] **Step 10: Run** — `.venv/bin/pytest tests/test_config.py tests/test_settings.py tests/test_web.py tests/test_watcher.py`. Expected: all PASS.

- [ ] **Step 11: Commit** — `git add -A && git commit -m "Config: [dslr] section (enabled, region, daily_cap) with settings-page fields"`

---

### Task 2: Store — DSLR columns, `Capture.photo`, counts and deletes

**Files:**
- Modify: `birdy/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_store.py` (it has `store`, `write_jpeg`, `add`, `T0`):

```python
def _eos_rel(store, cid) -> str:
    c = store.get(cid)
    rel = c.path.replace(".jpg", "_eos.jpg")
    write_jpeg(store.photos_dir / rel, w=1200, h=800)
    return rel


def _verdict(store, cid, **kw):
    store.store_verdict(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None, **kw)


def test_photo_is_the_pi_photo_by_default(store):
    cid = add(store, T0)
    _verdict(store, cid)
    c = store.get(cid)
    assert c.dslr_path is None and c.dslr_box is None and c.photo == c.path


def test_promoted_capture_uses_the_dslr_photo_and_thumb(store):
    cid = add(store, T0)
    old_thumb = store.thumbs_dir / store.get(cid).thumb_path
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    c = store.get(cid)
    assert c.dslr_path == rel and c.dslr_box == (0.4, 0.4, 0.1, 0.1)
    assert c.photo == rel and c.path != rel
    assert c.thumb_path == rel and (store.thumbs_dir / rel).exists()
    assert not old_thumb.exists()


def test_dslr_frame_without_a_bird_is_not_the_photo(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=None)
    c = store.get(cid)
    assert c.dslr_path == rel and c.dslr_box is None and c.photo == c.path
    assert c.thumb_path == c.path


def test_a_later_verdict_keeps_the_dslr_columns(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    _verdict(store, cid)                      # the backfill: no dslr arguments
    assert store.get(cid).photo == rel


def test_count_dslr_shots(store):
    a, b = add(store, T0), add(store, T0 + timedelta(hours=1))
    _verdict(store, a, dslr_path=_eos_rel(store, a), dslr_box=None)
    _verdict(store, b, dslr_path=_eos_rel(store, b), dslr_box=(0.1, 0.1, 0.2, 0.2))
    assert store.count_dslr_shots(T0 - timedelta(minutes=1)) == 2
    assert store.count_dslr_shots(T0 + timedelta(minutes=30)) == 1


def test_delete_capture_removes_the_dslr_file(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    store.delete_capture(cid)
    assert not (store.photos_dir / rel).exists() and not (store.thumbs_dir / rel).exists()


def test_purge_dslr_unused(store):
    old, new, promoted = add(store, T0), add(store, T0 + timedelta(days=9)), add(store, T0 + timedelta(seconds=5))
    rels = {}
    for cid, box in ((old, None), (new, None), (promoted, (0.4, 0.4, 0.1, 0.1))):
        rels[cid] = _eos_rel(store, cid)
        _verdict(store, cid, dslr_path=rels[cid], dslr_box=box)
    n = store.purge_dslr_unused(older_than_days=7, now=T0 + timedelta(days=10))
    assert n == 1
    assert store.get(old).dslr_path is None and not (store.photos_dir / rels[old]).exists()
    assert store.get(new).dslr_path == rels[new]                       # too recent
    assert store.get(promoted).photo == rels[promoted]                 # a bird photo: kept
    assert (store.photos_dir / store.get(old).path).exists()           # the Pi photo is untouched


def test_species_summary_best_photo_is_the_main_photo(store):
    cid = add(store, T0)
    rel = _eos_rel(store, cid)
    _verdict(store, cid, dslr_path=rel, dslr_box=(0.4, 0.4, 0.1, 0.1))
    store.write_final(cid, True, 0.9, "Robin", 0.8)
    assert store.species_summary()[0].best_path == rel
```

Make sure `timedelta` is imported at the top of the test file.

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_store.py -k "dslr or promoted or photo_is or main_photo" -v`. Expected: FAIL (`store_verdict() got an unexpected keyword argument 'dslr_path'`).

- [ ] **Step 3: Implement in `birdy/store.py`.**

(a) `Capture`: add after `has_embedding`

```python
    # The DSLR frame taken for this capture (relative to photos/), and the bird's box in it as
    # fractions (x, y, w, h) of that frame. A box means the frame is the capture's main photo.
    dslr_path: str | None = None
    dslr_box: tuple[float, float, float, float] | None = None

    @property
    def photo(self) -> str:
        """The photo to show, publish and alert with: the DSLR frame when a bird was found in it, else the Pi photo."""
        return self.dslr_path if self.dslr_path is not None and self.dslr_box is not None else self.path
```

(b) `_row_to_capture`: add

```python
        dslr_path=row["dslr_path"],
        dslr_box=None if row["dslr_x"] is None else (row["dslr_x"], row["dslr_y"], row["dslr_w"], row["dslr_h"]),
```

(c) In `Store.__init__`'s column loop add, after `("embedding", "BLOB"),`:

```python
                ("dslr_path", "TEXT"),
                ("dslr_x", "REAL"),
                ("dslr_y", "REAL"),
                ("dslr_w", "REAL"),
                ("dslr_h", "REAL"),
```

(d) Replace `store_verdict` with:

```python
    def store_verdict(
        self,
        capture_id: int,
        *,
        is_bird: bool,
        confidence: float,
        species: str | None,
        species_confidence: float | None,
        det_confidence: float = 0.0,
        det_box: tuple[float, float, float, float] | None = None,
        embedding: np.ndarray | None = None,
        dslr_path: str | None = None,
        dslr_box: tuple[float, float, float, float] | None = None,
    ) -> None:
        """The classifier's verdict for a photo (model_*); the final columns are written by Visits.
        A None embedding keeps the stored one. `dslr_path` records a DSLR frame taken for this
        capture (None = leave the DSLR columns as they are); with a `dslr_box` the frame becomes the
        main photo and the thumbnail is remade from it."""
        bx = (None, None, None, None) if det_box is None else det_box
        blob = None if embedding is None else np.asarray(embedding, np.float16).tobytes()
        # Outside the lock: decoding a 15 MP frame takes a moment.
        new_thumb = self._make_thumbnail(Path(dslr_path)) if dslr_path is not None and dslr_box is not None else None
        old_thumb = None
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE captures SET model_is_bird = ?, model_confidence = ?, model_species = ?, model_species_confidence = ?,"
                " det_confidence = ?, det_x = ?, det_y = ?, det_w = ?, det_h = ?,"
                " embedding = COALESCE(?, embedding), species_checked = 1, detector_checked = 1, status = 'done'"
                " WHERE id = ?",
                (int(is_bird), confidence, species, species_confidence, det_confidence, *bx, blob, capture_id),
            )
            if dslr_path is not None:
                dx = (None, None, None, None) if dslr_box is None else dslr_box
                self._conn.execute(
                    "UPDATE captures SET dslr_path = ?, dslr_x = ?, dslr_y = ?, dslr_w = ?, dslr_h = ? WHERE id = ?",
                    (dslr_path, *dx, capture_id),
                )
                if new_thumb is not None:
                    row = self._conn.execute("SELECT thumb_path FROM captures WHERE id = ?", (capture_id,)).fetchone()
                    old_thumb = None if row is None else row["thumb_path"]
                    self._conn.execute("UPDATE captures SET thumb_path = ? WHERE id = ?", (str(new_thumb), capture_id))
        if old_thumb and old_thumb != str(new_thumb):
            (self.thumbs_dir / old_thumb).unlink(missing_ok=True)
```

(e) Add after `count_pending`:

```python
    def count_dslr_shots(self, since: datetime) -> int:
        """Captures with a DSLR frame taken at or after `since` (the daily shutter cap)."""
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) FROM captures WHERE dslr_path IS NOT NULL AND captured_at >= ?", (_to_db_time(since),)
            ).fetchone()[0])
```

(f) `species_summary`: change the inner query to

```python
                    "SELECT id, CASE WHEN dslr_x IS NOT NULL THEN dslr_path ELSE path END AS photo, thumb_path"
                    " FROM captures WHERE species = ? AND is_bird = 1"
                    " ORDER BY species_confidence DESC, captured_at DESC LIMIT 1",
```

and `best_path=best["photo"],`.

(g) Deletes. Change `_delete_row_and_files` to

```python
    def _delete_row_and_files(self, capture_id: int, path: str, thumb_path: str | None, dslr_path: str | None = None) -> None:
        # Row first: an orphan file is harmless, a row without its file is a broken tile.
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM captures WHERE id = ?", (capture_id,))
        files = [self.photos_dir / path, self.thumbs_dir / path]
        if thumb_path:
            files.append(self.thumbs_dir / thumb_path)
        if dslr_path:
            files += [self.photos_dir / dslr_path, self.thumbs_dir / dslr_path]
        for p in files:
            p.unlink(missing_ok=True)
```

In `purge_non_birds` select `id, path, thumb_path, visit_id, dslr_path` and call `self._delete_row_and_files(row["id"], row["path"], row["thumb_path"], row["dslr_path"])`. In `delete_capture` and `delete_non_birds` pass `c.dslr_path` as the fourth argument.

(h) Add after `purge_non_birds`:

```python
    def purge_dslr_unused(self, older_than_days: int, now: datetime | None = None) -> int:
        """Delete DSLR frames no bird was found in, older than the cutoff; the capture and its Pi
        photo stay. Returns the number of files dropped."""
        now = now or datetime.now(timezone.utc)
        cutoff = _to_db_time(now - timedelta(days=older_than_days))
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT id, dslr_path FROM captures WHERE dslr_path IS NOT NULL AND dslr_x IS NULL AND captured_at < ?",
                (cutoff,),
            ).fetchall()
            self._conn.executemany("UPDATE captures SET dslr_path = NULL WHERE id = ?", [(r["id"],) for r in rows])
        for r in rows:
            (self.photos_dir / r["dslr_path"]).unlink(missing_ok=True)
        return len(rows)
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_store.py`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Store: DSLR frame columns, Capture.photo, shot count, unused-frame purge"`

---

### Task 3: `birdy/dslr.py` — gphoto2 wrapper and crop geometry

**Files:**
- Create: `birdy/dslr.py`, `tests/test_dslr.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_dslr.py`:

```python
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.dslr import Dslr, DslrError, box_in_frame, dslr_rel, region_crop, species_crop
from birdy.store import Store

NOW = datetime(2026, 11, 24, 12, 0).astimezone()


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


class Runner:
    """Stands in for subprocess.run: records calls and plays a scripted gphoto2."""

    def __init__(self, returncode=0, stdout="", stderr="", write=b"jpegbytes", raises=None):
        self.returncode, self.stdout, self.stderr, self.write, self.raises = returncode, stdout, stderr, write, raises
        self.calls: list[list[str]] = []

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(cmd)
        if self.raises is not None:
            raise self.raises
        if "--filename" in cmd and self.write is not None:
            Path(cmd[cmd.index("--filename") + 1]).write_bytes(self.write)
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def make(store, live, runner):
    return Dslr(store, live, runner=runner, now=lambda: NOW)


def tmp_files(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.name.endswith(".tmp")]


def test_capture_writes_the_file_atomically(store, live, tmp_path):
    r = Runner()
    dest = tmp_path / "a_eos.jpg"
    took = make(store, live, r).capture(dest)
    assert dest.read_bytes() == b"jpegbytes" and took >= 0 and not tmp_files(tmp_path)
    assert r.calls[0][:2] == ["gphoto2", "--capture-image-and-download"] and "--force-overwrite" in r.calls[0]


@pytest.mark.parametrize("runner, state", [
    (Runner(returncode=1, write=None, stderr="*** Error: No camera found. ***\n"), "not_connected"),
    (Runner(returncode=1, stderr="*** Error ***\nCould not claim the USB device\n"), "error"),
    (Runner(returncode=0, write=None), "error"),                       # exit 0 but no file
    (Runner(returncode=0, write=b""), "error"),                        # empty file
    (Runner(raises=subprocess.TimeoutExpired("gphoto2", 15)), "error"),
    (Runner(raises=FileNotFoundError("gphoto2")), "error"),
])
def test_capture_failures_raise_and_leave_nothing(store, live, tmp_path, runner, state):
    d = make(store, live, runner)
    dest = tmp_path / "a_eos.jpg"
    with pytest.raises(DslrError):
        d.capture(dest)
    assert not dest.exists() and not tmp_files(tmp_path)
    assert d.status().state == state


def test_daily_cap_blocks_before_gphoto2_runs(store, live, tmp_path):
    for i in range(live.get().dslr.daily_cap):
        when = NOW - timedelta(minutes=1 + i)
        abs_path, rel = store.photo_path_for(when)
        cv2.imwrite(str(abs_path), np.zeros((6, 8, 3), np.uint8))
        cid = store.add_capture(when, rel, 1)
        store.store_verdict(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None,
                            dslr_path=str(rel), dslr_box=None)
    r = Runner()
    d = make(store, live, r)
    with pytest.raises(DslrError, match="daily cap"):
        d.capture(tmp_path / "a_eos.jpg")
    assert r.calls == [] and d.status().state == "cap_reached"
    assert d.status().shots_today == live.get().dslr.daily_cap


def test_one_warning_per_distinct_error_and_recovery_is_logged(store, live, tmp_path, caplog):
    bad = Runner(returncode=1, write=None, stderr="*** Error: No camera found. ***\n")
    d = make(store, live, bad)
    with caplog.at_level(logging.INFO, logger="birdy.dslr"):
        for _ in range(3):
            with pytest.raises(DslrError):
                d.capture(tmp_path / "a_eos.jpg")
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
        d._runner = Runner()
        d.capture(tmp_path / "a_eos.jpg")
        assert any("ready" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert d.status().state == "ready"


INFO_OUT = """Label: Capture Target
Readonly: 0
Type: RADIO
Current: Internal RAM
Choice: 0 Internal RAM
END
Label: Auto Power Off
Readonly: 0
Type: TEXT
Current: 0
END
"""


def test_info_parses_current_values(store, live):
    r = Runner(stdout=INFO_OUT)
    d = make(store, live, r)
    assert d.info(("capturetarget", "autopoweroff")) == {"capturetarget": "Internal RAM", "autopoweroff": "0"}
    assert r.calls[0] == ["gphoto2", "--get-config", "capturetarget", "--get-config", "autopoweroff"]
    assert d.status().state == "ready"


def test_info_failure(store, live):
    d = make(store, live, Runner(returncode=1, stderr="*** Error: No camera found. ***\n"))
    with pytest.raises(DslrError):
        d.info()
    assert d.status().state == "not_connected"


def test_dslr_rel():
    assert dslr_rel(Path("2026/11/24/101502_123.jpg")) == Path("2026/11/24/101502_123_eos.jpg")


def test_region_crop_and_box_mapping():
    frame = np.zeros((200, 400, 3), np.uint8)
    region = (0.25, 0.5, 0.5, 0.5)
    assert region_crop(frame, region).shape[:2] == (100, 200)
    # A box in the middle of the region, as fractions of the region -> fractions of the frame.
    assert box_in_frame((0.5, 0.5, 0.2, 0.2), region) == pytest.approx((0.5, 0.75, 0.1, 0.1))


def test_species_crop_is_a_square_twice_the_box():
    frame = np.zeros((2000, 3000, 3), np.uint8)
    crop = species_crop(frame, (0.5, 0.5, 0.1, 0.05))        # box 300 x 100 px -> side 600
    assert crop.shape[:2] == (600, 600)


def test_species_crop_has_a_minimum_and_stays_in_the_frame():
    frame = np.zeros((2000, 3000, 3), np.uint8)
    assert species_crop(frame, (0.0, 0.0, 0.01, 0.01)).shape[:2] == (224, 224)   # tiny box in the corner
    assert species_crop(frame, (0.1, 0.1, 0.8, 0.8)).shape[:2] == (2000, 2000)   # capped by the frame height
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_dslr.py -v`. Expected: FAIL (`No module named 'birdy.dslr'`).

- [ ] **Step 3: Implement** — `birdy/dslr.py`:

```python
"""The second camera: a DSLR (Canon EOS 50D) driven with the `gphoto2` command over USB.

The only module that runs gphoto2. `Dslr` takes one frame per call; the pure helpers below turn
a detector box in the DSLR frame into the crop the species model sees."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from birdy.live import LiveConfig
from birdy.store import Store

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 15
MIN_SPECIES_SIDE = 224
INFO_KEYS = ("capturetarget", "imageformat", "autopoweroff", "focusmode", "autoexposuremode", "iso", "shuttercounter")

NormBox = tuple[float, float, float, float]


class DslrError(Exception):
    pass


@dataclass(frozen=True)
class DslrStatus:
    state: str            # 'ready' | 'not_connected' | 'error' | 'cap_reached' | 'unknown'
    shots_today: int
    last_error: str | None


def _message(proc: subprocess.CompletedProcess) -> str:
    """gphoto2's most telling output line: the first one naming an error, else the last one."""
    lines = [ln.strip() for ln in ((proc.stderr or "") + "\n" + (proc.stdout or "")).splitlines() if ln.strip()]
    for ln in lines:
        if "error" in ln.casefold() and ln.strip("* ").casefold() != "error":
            return ln.strip("* ")
    return lines[-1].strip("* ") if lines else f"gphoto2 exit {proc.returncode}"


def _state_for(message: str) -> str:
    m = message.casefold()
    return "not_connected" if "no camera found" in m or "could not detect any camera" in m else "error"


class Dslr:
    def __init__(
        self,
        store: Store,
        live: LiveConfig,
        runner: Callable = subprocess.run,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    ) -> None:
        self._store = store
        self._live = live
        self._runner = runner
        self._now = now
        self._state = "unknown"
        self._last_error: str | None = None
        self._warned: set[str] = set()

    # --- gphoto2 ----------------------------------------------------------
    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            return self._runner(["gphoto2", *args], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise DslrError(f"gphoto2 timed out after {TIMEOUT_SECONDS} s") from None
        except OSError as e:
            raise DslrError(f"gphoto2 could not be run: {e}") from None

    def _fail(self, state: str, message: str) -> None:
        self._state, self._last_error = state, message
        if message not in self._warned:       # one warning per distinct failure, not per bird
            self._warned.add(message)
            log.warning("50D: %s", message)
        raise DslrError(message)

    def _ok(self) -> None:
        if self._state != "ready":
            log.info("50D ready")
        self._state, self._last_error = "ready", None
        self._warned.clear()

    def _midnight(self) -> datetime:
        return self._now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)

    # --- public -----------------------------------------------------------
    def capture(self, dest: Path) -> float:
        """Take one frame and download it to `dest` (written through a temp name). Returns the
        seconds it took; raises DslrError (camera off, cap reached, timeout...), leaving no file."""
        cap = self._live.get().dslr.daily_cap
        midnight = self._midnight()
        if self._store.count_dslr_shots(midnight) >= cap:
            self._fail("cap_reached", f"daily cap reached ({cap} shots on {midnight:%d/%m}); Pi photos only until midnight")
        dest = Path(dest)
        tmp = dest.with_name(dest.name + ".tmp")
        t0 = time.monotonic()
        try:
            proc = self._run(["--capture-image-and-download", "--filename", str(tmp), "--force-overwrite"])
            # gphoto2 can exit 0 after printing an error: the file is the proof.
            if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
                raise DslrError(_message(proc))
            os.replace(tmp, dest)
        except DslrError as e:
            tmp.unlink(missing_ok=True)
            self._fail(_state_for(str(e)), str(e))
        self._ok()
        return time.monotonic() - t0

    def info(self, keys: tuple[str, ...] = INFO_KEYS) -> dict[str, str]:
        """The camera's current value for each config key, in one gphoto2 call."""
        args: list[str] = []
        for k in keys:
            args += ["--get-config", k]
        try:
            proc = self._run(args)
            if proc.returncode != 0:
                raise DslrError(_message(proc))
        except DslrError as e:
            self._fail(_state_for(str(e)), str(e))
        out: dict[str, str] = {}
        blocks = [b for b in (proc.stdout or "").split("END") if b.strip()]
        for key, block in zip(keys, blocks):
            for line in block.splitlines():
                if line.startswith("Current:"):
                    out[key] = line.split(":", 1)[1].strip()
        self._ok()
        return out

    def status(self) -> DslrStatus:
        shots = self._store.count_dslr_shots(self._midnight())
        state = self._state
        if state == "cap_reached" and shots < self._live.get().dslr.daily_cap:
            state = "ready"                   # a new day, or the cap was raised
        return DslrStatus(state=state, shots_today=shots, last_error=self._last_error)


# --- geometry (pure) --------------------------------------------------------

def dslr_rel(pi_rel: Path) -> Path:
    """Where a capture's DSLR frame lives: beside the Pi photo, `_eos` before the extension."""
    pi_rel = Path(pi_rel)
    return pi_rel.with_name(f"{pi_rel.stem}_eos{pi_rel.suffix}")


def region_crop(frame: np.ndarray, region: NormBox) -> np.ndarray:
    """The part of the frame the detector searches; `region` is (x, y, w, h) in fractions."""
    h, w = frame.shape[:2]
    x, y, rw, rh = region
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = min(w, max(x0 + 1, round((x + rw) * w))), min(h, max(y0 + 1, round((y + rh) * h)))
    return frame[y0:y1, x0:x1]


def box_in_frame(box: NormBox, region: NormBox) -> NormBox:
    """A box given as fractions of `region` -> fractions of the whole frame."""
    bx, by, bw, bh = box
    rx, ry, rw, rh = region
    return (rx + bx * rw, ry + by * rh, bw * rw, bh * rh)


def species_crop(frame: np.ndarray, box: NormBox) -> np.ndarray:
    """A square around the bird: side = 2 x the box's longer side (at least MIN_SPECIES_SIDE px),
    capped by the frame and shifted to stay inside it. `box` is in fractions of the frame."""
    h, w = frame.shape[:2]
    bx, by, bw, bh = box
    side = max(2 * max(bw * w, bh * h), MIN_SPECIES_SIDE)
    side = int(min(round(side), w, h))
    cx, cy = (bx + bw / 2) * w, (by + bh / 2) * h
    x0 = int(min(max(round(cx - side / 2), 0), w - side))
    y0 = int(min(max(round(cy - side / 2), 0), h - side))
    return frame[y0:y0 + side, x0:x0 + side]
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_dslr.py -v`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "dslr: gphoto2 wrapper (capture, info, status, daily cap) and species-crop geometry"`

---

### Task 4: Staged classifier API

**Files:**
- Modify: `birdy/classifier.py` (`TwoStageClassifier`)
- Test: `tests/test_two_stage.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_two_stage.py` (it has `two_stage`, `IMG`, `BLUE_TIT`):

```python
def test_verdict_then_name_equals_classify_image(live):
    clf = two_stage(live, 0.9, 0.8, p_det=0.4)
    v = clf.verdict(IMG)
    assert v.is_bird and v.confidence == 0.9 and v.detector_confidence == 0.4
    assert v.detector_box == (0.1, 0.2, 0.3, 0.4)
    top, emb = clf.name(IMG)
    assert clf.prediction(v, top, emb) == clf.classify_image(IMG)[0]


def test_verdict_does_not_run_the_species_model(live):
    clf = two_stage(live, 0.9, 0.8)
    clf.verdict(IMG)
    assert clf.species.calls == 0


def test_prediction_without_a_name_is_a_bird_with_no_species(live):
    clf = two_stage(live, 0.9, 0.8)
    pred = clf.prediction(clf.verdict(IMG), [], None)
    assert pred.is_bird and pred.species is None and pred.species_confidence is None and pred.embedding is None


def test_locate_returns_the_detectors_best_bird(live):
    assert two_stage(live, 0.0, 0.0, p_det=0.6).locate(IMG).score == 0.6
    assert two_stage(live, 0.0, 0.0, p_det=0.0).locate(IMG) is None
    assert two_stage(live, 0.0, 0.0, det_exc=RuntimeError("x")).locate(IMG) is None


def test_load_crop_returns_rgb(live, tmp_path):
    import cv2
    p = tmp_path / "a.jpg"
    bgr = np.zeros((60, 80, 3), np.uint8)
    bgr[:, :, 0] = 250                                    # blue in BGR
    cv2.imwrite(str(p), bgr)
    rgb = two_stage(live, 0.9, 0.8).load_crop(p, None)
    assert rgb.shape == (60, 80, 3) and rgb[30, 40, 2] > 200 and rgb[30, 40, 0] < 50
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_two_stage.py -k "verdict or prediction_without or locate or load_crop" -v`. Expected: FAIL (`'TwoStageClassifier' object has no attribute 'verdict'`).

- [ ] **Step 3: Implement in `birdy/classifier.py`.** Add after the `Prediction` dataclass (import `Detection` from `birdy.detector` beside `BirdDetector`):

```python
@dataclass(frozen=True)
class Verdict:
    """Bird or not, from the gate and the detector; no species yet."""
    is_bird: bool
    confidence: float                                         # max(gate P(bird), detector score)
    detector_confidence: float
    detector_box: tuple[float, float, float, float] | None
```

Replace `TwoStageClassifier.classify_image` with these methods (the docstring of the class stays; `_refresh_candidates`, `apply_candidates` and `classify` are unchanged):

```python
    def load_crop(self, image_path: Path, roi_bbox: BBox | None, pad: float | None = None) -> np.ndarray:
        """Decode a still and crop it like `classify` does; RGB."""
        return self._crops.load(image_path, roi_bbox, pad)

    def _detect(self, img_rgb: np.ndarray) -> Detection | None:
        """The detector's best bird, None when there is none or it failed. Caller holds the lock."""
        if self.detector is None:
            return None
        try:
            return self.detector.best_bird(img_rgb)
        except Exception as e:
            key = f"{type(e).__name__}: {e}"
            if key not in self._detector_errors:  # log each distinct failure once, not per capture
                log.exception("detector failed (%s); continuing without it", key)
                self._detector_errors.add(key)
            return None

    def verdict(self, img_rgb: np.ndarray, threshold: float | None = None) -> Verdict:
        """Stage 1: gate + detector. Fast (well under a second on the Pi)."""
        cfg = self._live.get()
        t_gate = cfg.classifier.threshold if threshold is None else threshold
        with self._lock:
            p_bird = self.gate.predict(img_rgb)
            det = self._detect(img_rgb)
        p_det = det.score if det is not None else 0.0
        return Verdict(
            is_bird=p_bird >= t_gate or p_det >= cfg.detector.threshold,
            confidence=max(p_bird, p_det),
            detector_confidence=p_det,
            detector_box=det.box if det is not None else None,
        )

    def locate(self, img_rgb: np.ndarray) -> Detection | None:
        """The detector alone, on any image (the DSLR frame's feeder region)."""
        with self._lock:
            return self._detect(img_rgb)

    def name(self, img_rgb: np.ndarray) -> tuple[list[tuple[SpeciesLabel, float]], np.ndarray | None]:
        """Stage 2: the species model's top-5 and image embedding ([], None when it failed).
        Only call it for a bird: the species model has no "nothing here" answer (14 % of empty
        frames get a confident name)."""
        cfg = self._live.get()
        with self._lock:
            try:
                self._refresh_candidates(cfg)
                fn = getattr(self.species, "top_with_embedding", None)
                return fn(img_rgb, k=5) if fn else (self.species.top(img_rgb, k=5), None)
            except Exception as e:
                key = f"{type(e).__name__}: {e}"
                if key not in self._species_errors:
                    log.exception("species model failed (%s); continuing without a species", key)
                    self._species_errors.add(key)
                return [], None

    def prediction(self, v: Verdict, top: list[tuple[SpeciesLabel, float]], emb: np.ndarray | None) -> Prediction:
        """Verdict + species result -> Prediction; a species is reported at or above `species.threshold`."""
        label, p_sp = top[0] if top else (None, 0.0)
        species = label.common if label is not None and p_sp >= self._live.get().species.threshold else None
        return Prediction(
            is_bird=v.is_bird,
            confidence=v.confidence,
            species=species,
            species_confidence=p_sp if species is not None else None,
            detector_confidence=v.detector_confidence,
            detector_box=v.detector_box,
            embedding=emb,
        )

    def classify_image(
        self, img_rgb: np.ndarray, threshold: float | None = None, known_bird: bool = False
    ) -> tuple[Prediction, list[tuple[SpeciesLabel, float]]]:
        v = self.verdict(img_rgb, threshold)
        # `known_bird`: a stored bird being re-checked always gets a species run.
        top, emb = self.name(img_rgb) if (v.is_bird or known_bird) else ([], None)
        return self.prediction(v, top, emb), top
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_two_stage.py tests/test_classifier.py tests/test_cli.py tests/test_web.py`. Expected: all PASS (the existing `classify_image` tests prove the refactor kept its behaviour).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Classifier: staged API (verdict, locate, name, prediction); classify_image built on it"`

---

### Task 5: Pipeline — fire the DSLR on a live bird

**Files:**
- Modify: `birdy/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_pipeline.py`:

```python
from birdy.classifier import Verdict
from birdy.detector import Detection
from birdy.dslr import DslrError


class StagedClassifier:
    """The staged API the DSLR path uses; records which image each stage saw."""

    def __init__(self, bird=True, dslr_bird=0.7):
        self.bird, self.dslr_bird = bird, dslr_bird
        self.located: list[tuple] = []
        self.named: list[tuple] = []

    def classify(self, *a, **k):
        raise AssertionError("the DSLR path must use the staged API")

    def load_crop(self, image_path, roi_bbox, pad=None):
        return np.zeros((48, 64, 3), np.uint8)

    def verdict(self, img, threshold=None):
        return Verdict(self.bird, 0.9 if self.bird else 0.1, 0.0, None)

    def locate(self, img):
        self.located.append(img.shape[:2])
        return Detection("bird", self.dslr_bird, (0.5, 0.5, 0.2, 0.2)) if self.dslr_bird else None

    def name(self, img):
        self.named.append(img.shape[:2])
        return [], np.ones(8, np.float32) / np.sqrt(8)

    def prediction(self, v, top, emb):
        return Prediction(is_bird=v.is_bird, confidence=v.confidence, embedding=emb)


class FakeDslr:
    def __init__(self, fail=False, garbage=False):
        self.fail, self.garbage, self.shots = fail, garbage, []

    def capture(self, dest):
        self.shots.append(dest)
        if self.fail:
            raise DslrError("no camera")
        if self.garbage:
            dest.write_bytes(b"not a jpeg")
        else:
            cv2.imwrite(str(dest), np.zeros((800, 1200, 3), np.uint8))
        return 1.2


def fresh():
    return datetime.now().astimezone()


def run_one(store, live, clf, dslr, when):
    ev = add_at(store, when)
    p = Pipeline(store, clf, live, dslr=dslr)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    return store.get(ev.capture_id)


def test_dslr_fires_once_for_a_fresh_bird_and_becomes_the_photo(store, live):
    clf, dslr = StagedClassifier(), FakeDslr()
    c = run_one(store, live, clf, dslr, fresh())
    assert len(dslr.shots) == 1
    assert c.status == "done" and c.dslr_path == c.path.replace(".jpg", "_eos.jpg")
    assert c.photo == c.dslr_path and c.thumb_path == c.dslr_path
    # default region = whole frame, so the box maps one to one
    assert c.dslr_box == pytest.approx((0.5, 0.5, 0.2, 0.2))
    assert clf.located == [(800, 1200)]
    assert clf.named == [(480, 480)]          # 2 x the box's longer side (0.2 * 1200), from the DSLR frame
    assert c.has_embedding


def test_dslr_never_fires_for_a_non_bird(store, live):
    clf, dslr = StagedClassifier(bird=False), FakeDslr()
    c = run_one(store, live, clf, dslr, fresh())
    assert dslr.shots == [] and clf.named == [] and c.dslr_path is None and c.is_bird is False


def test_dslr_does_not_fire_for_a_stale_bird(store, live):
    clf, dslr = StagedClassifier(), FakeDslr()
    c = run_one(store, live, clf, dslr, datetime.now().astimezone() - timedelta(seconds=30))
    assert dslr.shots == [] and c.dslr_path is None
    assert clf.named == [(48, 64)]            # the species still runs, on the Pi crop


def test_dslr_frame_without_a_bird_keeps_the_pi_photo(store, live):
    clf, dslr = StagedClassifier(dslr_bird=0.0), FakeDslr()
    c = run_one(store, live, clf, dslr, fresh())
    assert c.dslr_path is not None and c.dslr_box is None and c.photo == c.path
    assert clf.named == [(48, 64)]


def test_a_weak_detection_in_the_dslr_frame_does_not_promote(store, live):
    clf = StagedClassifier(dslr_bird=live.get().detector.threshold - 0.05)
    c = run_one(store, live, clf, FakeDslr(), fresh())
    assert c.dslr_box is None and c.photo == c.path


def test_dslr_failure_falls_back_to_the_pi_photo(store, live):
    clf = StagedClassifier()
    c = run_one(store, live, clf, FakeDslr(fail=True), fresh())
    assert c.status == "done" and c.dslr_path is None and c.is_bird is True
    assert clf.named == [(48, 64)]


def test_an_undecodable_dslr_file_is_removed(store, live):
    clf, dslr = StagedClassifier(), FakeDslr(garbage=True)
    c = run_one(store, live, clf, dslr, fresh())
    assert c.dslr_path is None and not dslr.shots[0].exists() and c.status == "done"


def test_the_dslr_region_limits_the_search_and_maps_the_box(store, live, config):
    live.replace(dataclasses.replace(config, dslr=dataclasses.replace(config.dslr, region=(0.5, 0.0, 0.5, 1.0))))
    clf = StagedClassifier()
    c = run_one(store, live, clf, FakeDslr(), fresh())
    assert clf.located == [(800, 600)]
    assert c.dslr_box == pytest.approx((0.75, 0.5, 0.1, 0.2))


def test_without_a_dslr_the_pipeline_calls_classify_only(store, live):
    clf = FakeClassifier()
    ev = add_at(store, fresh())
    Pipeline(store, clf, live).process(ev.capture_id, ev.path, ev.captured_at)
    assert clf.calls == [ev.path] and store.get(ev.capture_id).dslr_path is None
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_pipeline.py -k dslr -v`. Expected: FAIL (`Pipeline.__init__() got an unexpected keyword argument 'dslr'`).

- [ ] **Step 3: Implement in `birdy/pipeline.py`.**

Imports: add `import cv2`, `from datetime import timezone` (beside the existing `datetime` import), `from birdy.dslr import DslrError, box_in_frame, dslr_rel, region_crop, species_crop`, and under `TYPE_CHECKING` `from birdy.dslr import Dslr`. Add the constant near `TICK_INTERVAL`:

```python
FRESH_SECONDS = 10.0   # the DSLR only fires for a Pi photo this recent: never for the backlog or after a restart
```

`Pipeline.__init__`: add the parameter `dslr: "Dslr | None" = None` after `visits` and store `self._dslr = dslr`.

Replace the first `try` block of `process` (the one that computes `pred`) and the `store_verdict` call with:

```python
        cfg = self._live.get()
        dslr_path, dslr_box = None, None
        try:
            # Inside the try: a bad stored box marks the row 'error' instead of retrying it forever.
            crop = crop_box(motion_box, cfg.motion.roi, self._lores_size)
            if self._dslr is None:
                pred = self._classifier.classify(path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad)
            else:
                pred, dslr_path, dslr_box = self._classify_with_dslr(capture_id, path, captured_at, crop, cfg)
        except Exception:
            log.exception("classification failed for %s", path)
            try:
                self._store.mark_error(capture_id)
            except Exception:
                log.exception("mark_error failed for capture %d; left pending", capture_id)
            return
        try:
            self._store.store_verdict(
                capture_id, is_bird=pred.is_bird, confidence=pred.confidence,
                species=pred.species, species_confidence=pred.species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box, embedding=pred.embedding,
                dslr_path=dslr_path, dslr_box=dslr_box,
            )
```

(the `except` after it and the rest of `process` stay as they are). Add these two methods after `process`:

```python
    def _classify_with_dslr(self, capture_id: int, path: Path, captured_at: datetime, crop, cfg):
        """Verdict on the Pi crop; for a fresh bird, one DSLR frame; the species from that frame
        when the detector finds the bird in it, else from the Pi crop.
        Returns (prediction, DSLR path relative to photos/ or None, bird box in the DSLR frame or None)."""
        clf = self._classifier
        crop_rgb = clf.load_crop(path, crop.bbox, crop.pad)
        v = clf.verdict(crop_rgb, cfg.classifier.threshold)
        if not v.is_bird:
            return clf.prediction(v, [], None), None, None
        rel, frame, box = None, None, None
        if (datetime.now(timezone.utc) - captured_at).total_seconds() <= FRESH_SECONDS:
            rel = dslr_rel(path.relative_to(self._store.photos_dir))
            frame, took = self._dslr_frame(capture_id, rel)
        species_rgb = crop_rgb
        if frame is not None:
            det = clf.locate(region_crop(frame, cfg.dslr.region))
            if det is not None and det.score >= cfg.detector.threshold:
                box = box_in_frame(det.box, cfg.dslr.region)
                species_rgb = species_crop(frame, box)
                log.info("capture %d: 50D %.2f s, bird %.2f", capture_id, took, det.score)
            else:
                log.info("capture %d: 50D %.2f s, no bird in frame", capture_id, took)
        top, emb = clf.name(species_rgb)
        return clf.prediction(v, top, emb), (str(rel) if frame is not None else None), box

    def _dslr_frame(self, capture_id: int, rel: Path) -> tuple[np.ndarray | None, float]:
        """Fire the DSLR; the decoded RGB frame and the seconds taken, or (None, 0) when it failed."""
        dest = self._store.photos_dir / rel
        try:
            took = self._dslr.capture(dest)
        except DslrError:
            return None, 0.0                  # Dslr logged it (once per distinct error)
        bgr = cv2.imread(str(dest))
        if bgr is None:
            dest.unlink(missing_ok=True)
            log.warning("capture %d: the 50D frame could not be decoded; using the Pi photo", capture_id)
            return None, 0.0
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), took
```

Add `import numpy as np` if the module does not import it yet.

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_pipeline.py`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Pipeline: fire the DSLR on a fresh bird verdict; name the bird from its frame when found there"`

---

### Task 6: Backfill — promoted rows are renamed from the DSLR crop

**Files:**
- Modify: `birdy/pipeline.py` (`_backfill`)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_pipeline.py`:

```python
def test_backfill_of_a_promoted_row_names_from_the_dslr_crop_and_never_fires(store, live):
    clf, dslr = StagedClassifier(), FakeDslr()
    ev = add_at(store, fresh())
    p = Pipeline(store, clf, live, dslr=dslr)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).dslr_box is not None
    # A species-model swap puts the row back in the backlog.
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0 WHERE id = ?", (ev.capture_id,))
    clf.named.clear()
    p.sweep()
    assert clf.named == [(480, 480)]          # the DSLR species crop again, not the Pi crop
    assert len(dslr.shots) == 1               # the backfill never fires the camera
    assert store.get(ev.capture_id).photo == store.get(ev.capture_id).dslr_path


def test_backfill_of_a_promoted_row_falls_back_to_the_pi_crop_when_the_dslr_file_is_gone(store, live):
    clf = StagedClassifier()
    ev = add_at(store, fresh())
    p = Pipeline(store, clf, live, dslr=FakeDslr())
    p.process(ev.capture_id, ev.path, ev.captured_at)
    c = store.get(ev.capture_id)
    (store.photos_dir / c.dslr_path).unlink()
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0 WHERE id = ?", (ev.capture_id,))
    clf.named.clear()
    p.sweep()
    assert clf.named == [(48, 64)] and store.get(ev.capture_id).status == "done"
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_pipeline.py -k "backfill_of_a_promoted" -v`. Expected: FAIL with `AssertionError: the DSLR path must use the staged API` (the backfill calls `classify`).

- [ ] **Step 3: Implement.** In `_backfill`, replace

```python
            pred = self._classifier.classify(
                path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad, known_bird=known
            )
```

with

```python
            if c.dslr_box is not None and hasattr(self._classifier, "verdict"):
                pred = self._reclassify_promoted(c, path, crop, cfg, known)
            else:
                pred = self._classifier.classify(
                    path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad, known_bird=known
                )
```

and add after `_backfill`:

```python
    def _reclassify_promoted(self, c: Capture, path: Path, crop, cfg, known: bool):
        """A row whose main photo is the DSLR frame: bird/non-bird is re-checked on the Pi crop,
        the species comes from the stored DSLR crop (the Pi crop when that file is unreadable)."""
        clf = self._classifier
        crop_rgb = clf.load_crop(path, crop.bbox, crop.pad)
        v = clf.verdict(crop_rgb, cfg.classifier.threshold)
        if not (v.is_bird or known):
            return clf.prediction(v, [], None)
        bgr = cv2.imread(str(self._store.photos_dir / c.dslr_path))
        img = crop_rgb if bgr is None else species_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), c.dslr_box)
        top, emb = clf.name(img)
        return clf.prediction(v, top, emb)
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_pipeline.py`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Backfill: rename promoted rows from the DSLR crop; never fire the camera"`

---

### Task 7: Retention — drop unused DSLR frames

**Files:**
- Modify: `birdy/retention.py`
- Test: `tests/test_retention.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_retention.py` (match the imports already in that file; `cv2`, `np`, `datetime`, `timedelta` are needed):

```python
def test_run_once_drops_unused_dslr_frames(store, caplog):
    now = datetime(2026, 11, 24, 12, 0).astimezone()
    when = now - timedelta(days=10)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((6, 8, 3), np.uint8))
    cid = store.add_capture(when, rel, 1)
    eos = str(rel).replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), np.zeros((6, 8, 3), np.uint8))
    store.store_verdict(cid, is_bird=True, confidence=0.9, species=None, species_confidence=None, dslr_path=eos)
    store.write_final(cid, True, 0.9, None, None)
    with caplog.at_level(logging.INFO, logger="birdy.retention"):
        run_once(store, 7, now=now)
    c = store.get(cid)
    assert c is not None and c.dslr_path is None and not (store.photos_dir / eos).exists()
    assert "1 unused 50D frames" in caplog.text
```


- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_retention.py -k dslr -v`. Expected: FAIL (`dslr_path` still set).

- [ ] **Step 3: Implement.** In `run_once`, after the `visits` loop and before `free = ...`:

```python
    frames = store.purge_dslr_unused(older_than_days=non_bird_days, now=now)
```

and change the info line to

```python
    log.info("retention: deleted %d non-bird captures, %d unused 50D frames; disk free %.1f GB", deleted, frames, free / 1e9)
```

Update the function's docstring to add: "Also drops DSLR frames no bird was found in after the same number of days (the capture stays)."

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_retention.py tests/test_cli.py`. Expected: all PASS (fix any test that asserts the old log text by matching the new text).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Retention: drop DSLR frames with no bird after non_bird_days"`

---

### Task 8: Readers of the main photo — grid, Model view, publisher, alert

**Files:**
- Modify: `birdy/web/templates/index.html`, `birdy/web/app.py` (`_crop_jpeg`), `birdy/publish.py`, `birdy/visits.py`
- Test: `tests/test_web.py`, `tests/test_publish.py`, `tests/test_visits.py`

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_web.py` (its `add(store, i, is_bird)` helper stores a classified capture and returns the `Capture`):

```python
def _promote(store, cid, box=(0.4, 0.4, 0.2, 0.2)):
    c = store.get(cid)
    eos = c.path.replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), np.full((800, 1200, 3), 90, np.uint8))
    store.store_verdict(cid, is_bird=True, confidence=0.9, species="Robin", species_confidence=0.8,
                        dslr_path=eos, dslr_box=box)
    return eos


def test_grid_links_the_dslr_photo_for_a_promoted_capture(client, store):
    cid = add(store, 0, True).id
    eos = _promote(store, cid)
    html = client.get("/?filter=all").text
    assert f'class="photo" href="/photos/{eos}"' in html
    assert f'href="/photos/{store.get(cid).path}"' in html and ">Pi photo<" in html
    assert '<span class="tag">50D</span>' in html


def test_grid_offers_the_unused_dslr_frame(client, store):
    cid = add(store, 0, True).id
    eos = _promote(store, cid, box=None)
    html = client.get("/?filter=all").text
    assert f'class="photo" href="/photos/{store.get(cid).path}"' in html
    assert f'href="/photos/{eos}"' in html and ">50D frame<" in html


def test_crop_jpg_of_a_promoted_capture_is_the_dslr_species_crop(client, store):
    cid = add(store, 0, True).id
    _promote(store, cid)
    r = client.get(f"/captures/{cid}/crop.jpg")
    assert r.status_code == 200
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (480, 480)        # 2 x (0.2 * 1200)
```

Append to `tests/test_publish.py` (it has `make`, `bird`, `T0`):

```python
def test_mirror_uses_the_dslr_photo_of_a_promoted_capture(live, store, config):
    pub = make(live, store, config)
    c = bird(store, T0, "Robin", 0.9)
    eos = c.path.replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), np.full((2000, 3000, 3), 90, np.uint8))
    store.store_verdict(c.id, is_bird=True, confidence=0.9, species="Robin", species_confidence=0.9,
                        dslr_path=eos, dslr_box=(0.4, 0.4, 0.2, 0.2))
    store.set_published(c.id, "forced")
    assert pub.run_once() is True
    c = store.get(c.id)
    assert c.web_path == eos
    web = cv2.imread(str(pub.photos_dir / eos))
    assert max(web.shape[:2]) == 1600
    assert (pub.thumbs_dir / eos).exists()
```

Append to `tests/test_visits.py` (it has `add`, `e`, `T0`, and a `FakeNotifier` on `visits._notifier`):

```python
def test_alert_sends_the_main_photo(store, visits):
    paths = []
    consider = visits._notifier.consider

    def recording(capture_id, photo_path, *rest):
        paths.append(photo_path)
        return consider(capture_id, photo_path, *rest)

    visits._notifier.consider = recording
    a = add(store, 0, emb=e(0))
    eos = store.get(a).path.replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), np.zeros((80, 120, 3), np.uint8))
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE captures SET dslr_path = ?, dslr_x = 0.4, dslr_y = 0.4, dslr_w = 0.2, dslr_h = 0.2 WHERE id = ?", (eos, a))
    visits.photo_classified(store.get(a), live=True)
    visits.tick(T0 + timedelta(seconds=71))
    assert paths == [store.photos_dir / eos]
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_web.py tests/test_publish.py tests/test_visits.py -k "dslr or promoted" -v`. Expected: FAIL.

- [ ] **Step 3: Implement.**

`birdy/web/templates/index.html`:

- Tile link (line ~181): change `href="/photos/{{ c.path }}"` to `href="/photos/{{ c.photo }}"`.
- In the second `.stamp` of the tile, after the `<span class="mot">…</span>` line, add: `{% if c.dslr_box %}<span class="tag">50D</span>{% endif %}`
- After the `Model view` link (line ~219) add (the class is `aux`, not `view`: the lightbox script binds the tile's first `a.view`):

```html
    {% if c.dslr_box %}<a class="pb aux" href="/photos/{{ c.path }}" title="The Pi camera's photo that triggered the 50D">Pi photo</a>
    {% elif c.dslr_path %}<a class="pb aux" href="/photos/{{ c.dslr_path }}" title="The 50D frame: no bird was found in it">50D frame</a>{% endif %}
```

- CSS: change `.acts .pb.view { flex: 0 1 auto; color: #a3aaa5; }` to `.acts .pb.view, .acts .pb.aux { flex: 0 1 auto; color: #a3aaa5; }`, and in the phone media query `.acts .pb.view { flex: 1 1 auto; }` to `.acts .pb.view, .acts .pb.aux { flex: 1 1 auto; }`.

`birdy/web/app.py` — import `from birdy.dslr import species_crop` and start `_crop_jpeg` with:

```python
        if c.dslr_box is not None:
            # The species model saw a square around the bird in the DSLR frame.
            frame = cv2.imread(str(store.photos_dir / c.dslr_path))
            if frame is None:
                return None
            ok, jpg = cv2.imencode(".jpg", species_crop(frame, c.dslr_box), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpg.tobytes() if ok else None
```

Update its docstring: "…; for a capture whose main photo is the DSLR frame, the species crop of that frame."

`birdy/publish.py` — in the mirror loop change `rel = c.path` to `rel = c.photo`.

`birdy/visits.py` — in `_alert` change `self._store.photos_dir / best.path` to `self._store.photos_dir / best.photo`.

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_web.py tests/test_publish.py tests/test_visits.py`. Expected: all PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Grid, Model view, gallery mirror and alerts use the capture's main photo (the 50D frame when a bird is in it)"`

---

### Task 9: Telegram — photos Telegram would reject are sent smaller

**Files:**
- Modify: `birdy/notify.py`
- Test: `tests/test_notify.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_notify.py`:

```python
def _jpeg(path, w, h):
    import cv2
    import numpy as np
    cv2.imwrite(str(path), np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))
    return path


def test_telegram_jpeg_passes_a_normal_photo_through(tmp_path):
    from birdy.notify import _telegram_jpeg
    p = _jpeg(tmp_path / "a.jpg", 3280, 2464)
    assert _telegram_jpeg(p) == p.read_bytes()


def test_telegram_jpeg_shrinks_a_photo_with_oversized_sides(tmp_path):
    import cv2
    import numpy as np
    from birdy.notify import _telegram_jpeg
    p = _jpeg(tmp_path / "a.jpg", 6000, 4100)            # sides sum to 10,100
    out = cv2.imdecode(np.frombuffer(_telegram_jpeg(p), np.uint8), cv2.IMREAD_COLOR)
    assert max(out.shape[:2]) == 2560


def test_telegram_jpeg_shrinks_a_heavy_file(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import birdy.notify as notify
    p = _jpeg(tmp_path / "a.jpg", 3000, 2000)
    monkeypatch.setattr(notify, "TG_MAX_BYTES", 1000)
    out = cv2.imdecode(np.frombuffer(notify._telegram_jpeg(p), np.uint8), cv2.IMREAD_COLOR)
    assert max(out.shape[:2]) == 2560


def test_telegram_jpeg_missing_file_raises_file_not_found(tmp_path):
    from birdy.notify import _telegram_jpeg
    with pytest.raises(FileNotFoundError):
        _telegram_jpeg(tmp_path / "gone.jpg")
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_notify.py -k telegram_jpeg -v`. Expected: FAIL (`cannot import name '_telegram_jpeg'`).

- [ ] **Step 3: Implement in `birdy/notify.py`.** Add `import cv2` and `import numpy as np` to the imports, and near the other constants:

```python
# Telegram's sendPhoto rejects files over 10 MB and photos whose width + height exceed 10,000.
TG_MAX_BYTES = 8 * 1024 * 1024
TG_MAX_SIDE_SUM = 10_000
TG_SEND_SIDE = 2560


def _telegram_jpeg(path: Path) -> bytes:
    """The photo's bytes as Telegram accepts them: the file itself, or a smaller JPEG when it is
    too heavy or too large (a 15 MP DSLR frame can be). FileNotFoundError propagates."""
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return data
    h, w = img.shape[:2]
    if len(data) <= TG_MAX_BYTES and w + h <= TG_MAX_SIDE_SUM:
        return data
    scale = TG_SEND_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes() if ok else data
```

In `_multipart`, replace `data = path.read_bytes()  # FileNotFoundError propagates to the caller` with `data = _telegram_jpeg(path)  # FileNotFoundError propagates to the caller`.

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_notify.py`. Expected: all PASS (`test_http_transport_send_photo_multipart` writes a small fake photo; if it writes non-JPEG bytes, `_telegram_jpeg` returns them unchanged, so it still passes).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Telegram: send a 2560 px copy when the photo is over 8 MB or its sides exceed Telegram's limit"`

---

### Task 10: CLI and service wiring — `birdy dslr`, `cmd_run`, status

**Files:**
- Modify: `birdy/cli.py`, `birdy/web/app.py`, `birdy/web/templates/settings.html`
- Test: `tests/test_cli.py`, `tests/test_web.py`

- [ ] **Step 1: Write the failing tests.**

Append to `tests/test_cli.py` (add `import cv2` at the top; `config_path` is the conftest fixture, whose `data_dir` lands under `tmp_path`):

```python
class _FakeDslr:
    def __init__(self, *a, **k): pass

    def info(self, keys=None):
        return {"capturetarget": "Memory card", "imageformat": "Large Fine JPEG", "autopoweroff": "30",
                "focusmode": "Manual", "autoexposuremode": "TV", "iso": "Auto", "shuttercounter": "22764"}

    def capture(self, dest):
        cv2.imwrite(str(dest), np.zeros((800, 1200, 3), np.uint8))
        return 1.25


class _FakeDetector:
    def __init__(self, *a, **k): pass

    def best_bird(self, img):
        return None


def test_dslr_info_prints_values_and_warnings(config_path, monkeypatch, capsys):
    from birdy import cli
    monkeypatch.setattr("birdy.cli.Dslr", _FakeDslr)
    assert cli.main(["--config", str(config_path), "dslr", "info"]) == 0
    out = capsys.readouterr().out
    assert "shuttercounter: 22764" in out
    assert "warning: capturetarget is 'Memory card'" in out and "warning: autopoweroff is '30'" in out
    assert "warning: focusmode" not in out


def test_dslr_shot_saves_the_frame_and_the_region_copy(config_path, tmp_path, monkeypatch, capsys):
    from birdy import cli
    monkeypatch.setattr("birdy.cli.Dslr", _FakeDslr)
    monkeypatch.setattr("birdy.cli.BirdDetector", _FakeDetector)
    out_file = tmp_path / "shot.jpg"
    assert cli.main(["--config", str(config_path), "dslr", "shot", "--out", str(out_file)]) == 0
    assert out_file.exists() and (tmp_path / "shot-region.jpg").exists()
    out = capsys.readouterr().out
    assert "1200x800" in out and "1.25 s" in out and "no bird" in out


def test_dslr_error_is_reported(config_path, monkeypatch, capsys):
    from birdy import cli
    from birdy.dslr import DslrError

    class Broken(_FakeDslr):
        def info(self, keys=None):
            raise DslrError("No camera found.")

    monkeypatch.setattr("birdy.cli.Dslr", Broken)
    assert cli.main(["--config", str(config_path), "dslr", "info"]) == 1
    assert "No camera found." in capsys.readouterr().err
```

Append to `tests/test_web.py`:

```python
class _StatusDslr:
    def status(self):
        from birdy.dslr import DslrStatus
        return DslrStatus(state="ready", shots_today=4, last_error=None)


def test_status_and_settings_show_the_dslr(store, state, live, config_path):
    c = TestClient(create_app(store, state, live, config_path, dslr=_StatusDslr()))
    assert c.get("/status").json()["dslr"] == {"state": "ready", "shots_today": 4}
    assert "50D: ready · 4 shots today" in c.get("/settings").text


def test_status_has_no_dslr_when_disabled(client):
    assert "dslr" not in client.get("/status").json()
    assert "50D:" not in client.get("/settings").text
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_cli.py tests/test_web.py -k dslr -v`. Expected: FAIL.

- [ ] **Step 3: Implement the web part.** `create_app` gains the parameter `dslr: Any | None = None` (document it in the docstring: "`dslr` has `status()`; normally a `birdy.dslr.Dslr`; None = no second camera."). In `status_payload`, before `return snap`:

```python
        if dslr is not None:
            st = dslr.status()
            snap["dslr"] = {"state": st.state, "shots_today": st.shots_today}
```

In `settings.html`, directly after the `<div class="cambar">…</div>` line:

```html
{% if status.dslr %}<p class="hint">50D: {{ {'ready': 'ready', 'not_connected': 'not connected', 'cap_reached': 'daily cap reached', 'error': 'error', 'unknown': 'not checked yet'}[status.dslr.state] }} · {{ status.dslr.shots_today }} shots today</p>{% endif %}
```

- [ ] **Step 4: Implement the CLI.** In `birdy/cli.py` add the imports `from birdy.detector import BirdDetector` (if not already imported) and `from birdy.dslr import Dslr, DslrError, region_crop`.

`build_parser`, after the `species-list` parser:

```python
    ds = sub.add_parser("dslr", help="the second camera: print its settings, or take a test frame")
    ds.add_argument("action", choices=("info", "shot"))
    ds.add_argument("--out", default=None, help="shot: where to save the frame (default data/dslr-test.jpg)")
```

`main`, with the other dispatch lines: `if args.command == "dslr": return cmd_dslr(config, args.action, args.out)`.

Add:

```python
_DSLR_EXPECTED = {"capturetarget": "Internal RAM", "autopoweroff": "0", "focusmode": "Manual"}


def cmd_dslr(config: Config, action: str, out: str | None = None) -> int:
    """`info`: the camera's settings, with a warning for each one birdy needs different.
    `shot`: one test frame, a copy with `dslr.region` drawn, and what the detector finds in the
    region. Works while the service runs (when both reach for the camera at once, one fails)."""
    store = Store(config.storage.data_dir)
    try:
        dslr = Dslr(store, LiveConfig(config))
        try:
            if action == "info":
                values = dslr.info()
                for k, v in values.items():
                    print(f"{k}: {v}")
                for k, want in _DSLR_EXPECTED.items():
                    if k in values and values[k] != want:
                        print(f"warning: {k} is '{values[k]}', birdy expects '{want}'")
                return 0
            dest = Path(out) if out else config.storage.data_dir / "dslr-test.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            took = dslr.capture(dest)
        except DslrError as e:
            print(f"50D: {e}", file=sys.stderr)
            return 1
        frame = cv2.imread(str(dest))
        if frame is None:
            print(f"50D: {dest} could not be decoded", file=sys.stderr)
            return 1
        h, w = frame.shape[:2]
        x, y, rw, rh = config.dslr.region
        marked = frame.copy()
        cv2.rectangle(marked, (int(x * w), int(y * h)), (int((x + rw) * w), int((y + rh) * h)), (0, 176, 255), max(2, w // 400))
        region_file = dest.with_name(f"{dest.stem}-region{dest.suffix}")
        cv2.imwrite(str(region_file), marked)
        print(f"saved {dest} ({w}x{h}) in {took:.2f} s; region drawn in {region_file}")
        region_rgb = cv2.cvtColor(region_crop(frame, config.dslr.region), cv2.COLOR_BGR2RGB)
        det = BirdDetector(config.detector.model, config.detector.labels).best_bird(region_rgb)
        if det is None:
            print(f"detector in region {list(config.dslr.region)}: no bird")
        else:
            bx, by, bw, bh = det.box
            print(f"detector in region {list(config.dslr.region)}: bird {det.score:.3f} "
                  f"box x={bx:.2f} y={by:.2f} w={bw:.2f} h={bh:.2f} ({bw * 100:.0f} % of the region's width)")
        return 0
    finally:
        store.close()
```

`cmd_run`: after the `publisher` block add

```python
    dslr = None
    if config.dslr.enabled:
        dslr = Dslr(store, live)
        try:
            info = dslr.info()
            log.info("50D connected: %s", ", ".join(f"{k}={v}" for k, v in info.items()))
        except DslrError:
            pass   # logged by Dslr; captures fall back to the Pi photo until it answers
```

change the pipeline line to `pipeline = Pipeline(store, clf, live, lores_size=config.camera.lores_size, visits=visits, dslr=dslr)`, pass `dslr=dslr` to `create_app(...)`, and extend the "birdy starting" log line with `dslr=%s` / `config.dslr.enabled`.

`cmd_classify`: for `--capture` on a promoted row, classify what the service classified. After `c` is loaded and before the `crop = crop_box(...)` lines, add:

```python
        if c.dslr_box is not None:
            frame = cv2.imread(str(store.photos_dir / c.dslr_path))
            if frame is None:
                print(f"cannot read {c.dslr_path}", file=sys.stderr)
                return 1
            dslr_img = species_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), c.dslr_box)
            print(f"capture {c.id} {c.dslr_path}: dslr crop {dslr_img.shape[1]}x{dslr_img.shape[0]} px around the bird")
```

initialise `dslr_img = None` before the `if capture_id is not None:` block, guard the existing `crop = crop_box(...)`/`print` lines with `else:` (they still run for non-promoted captures; `path` must still be set in both branches), and change the load to:

```python
    try:
        img = dslr_img if dslr_img is not None else CropLoader(config.camera.lores_size).load(path, bbox, pad)
```

Import `species_crop` from `birdy.dslr`. In `tests/test_cli.py`, extend `test_classify_capture_uses_stored_crop` (it skips without the real models) — before its `999` case add:

```python
    eos = str(rel).replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), cv2.imread(str(bird)))
    store.store_verdict(without, is_bird=True, confidence=0.9, species=None, species_confidence=None,
                        dslr_path=eos, dslr_box=(0.25, 0.25, 0.5, 0.5))
    proc = _run_cli("--config", str(cfg), "classify", "--capture", str(without))
    assert proc.returncode == 0, proc.stderr
    assert "dslr crop" in proc.stdout
```

- [ ] **Step 5: Run** — `.venv/bin/pytest tests/test_cli.py tests/test_web.py`. Expected: all PASS.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "CLI: birdy dslr info|shot; service builds the Dslr when enabled; 50D state on /status and /settings"`

---

### Task 11: Install script and docs

**Files:**
- Modify: `deploy/install.sh`, `CLAUDE.md`, `docs/superpowers/specs/2026-11-24-dslr-camera-design.md`

- [ ] **Step 1: `deploy/install.sh`** — change the apt line to

```bash
sudo apt-get install -y python3-picamera2 python3-opencv python3-numpy python3-venv rsync gphoto2
```

- [ ] **Step 2: Spec correction.** In the spec's CLI section, replace "takes one frame (it counts against the daily cap)" with "takes one frame (refused when the daily cap is reached; test frames are not counted, since the cap counts stored captures)".

- [ ] **Step 3: `CLAUDE.md`.**

- In the first paragraph's spec list, add: `` `docs/superpowers/specs/2026-11-24-dslr-camera-design.md` for the DSLR second camera ``.
- In the Pi commands block, add:

```bash
.venv/bin/birdy dslr info         # the 50D's settings over USB, with warnings for the ones birdy needs different
.venv/bin/birdy dslr shot         # one test frame + a copy with dslr.region drawn; the tool for aiming and focusing
```

- In Architecture, add this bullet after the `birdy/classifier.py` bullets:

```markdown
- `birdy/dslr.py` is the **only** module that runs `gphoto2` (a Canon EOS 50D on USB, same window as the Pi camera). With `dslr.enabled = true` the pipeline uses the classifier's staged API: `verdict` (gate + detector on the Pi crop) → for a bird whose Pi photo is ≤ 10 s old (`FRESH_SECONDS`; never the backlog or backfill) `Dslr.capture` (~1.2 s, one frame, `…_eos.jpg` beside the Pi photo) → `locate` (the detector on `dslr.region` of that frame) → `name` (BioCLIP) on a square around the bird in the DSLR frame (`species_crop`: 2× the box's longer side, ≥ 224 px), or on the Pi crop when no bird was found or the camera failed. `captures.path` always stays the Pi photo; `dslr_path` + `dslr_x/y/w/h` (the bird's box, fractions of the DSLR frame) record the frame, and `Capture.photo` — the DSLR frame when the box is set, else the Pi photo — is what the grid, lightbox, Model view, gallery mirror and Telegram use. The main photo never changes after first classification. A DSLR frame with no bird is kept `retention.non_bird_days` (`store.purge_dslr_unused`), linked from the tile as **50D frame**. Failures (camera off, timeout, `dslr.daily_cap` reached — counted from the DB per local day) are one warning per distinct error and the capture carries on with the Pi photo. Everything is sequential on the pipeline thread: mid-visit shots can land 4–5 s late behind a BioCLIP run; the spec's *Later* section has the measurement that would justify a second thread.
```

- In Conventions and gotchas, add:

```markdown
- `dslr.enabled` is restart-only (the watcher warns); `dslr.region` and `dslr.daily_cap` are live and on `/settings`. The 50D's exposure, ISO, image size and focus are set on the camera body, never by birdy; it must be on AC power with auto power off **Off**, lens on **MF**, `capturetarget = Internal RAM` (`birdy dslr info` warns otherwise). `gphoto2` comes from apt (`install.sh`). Tests never need it: `Dslr` takes a `runner` in place of `subprocess.run`.
- Telegram rejects photos over 10 MB or with width + height over 10,000: `notify._telegram_jpeg` sends a 2560 px copy for a file over 8 MB or past that size.
```

- Update the thread-count sentence only if it mentions modules; the thread count is unchanged (the DSLR runs on the pipeline thread).

- [ ] **Step 4: Full suite** — `.venv/bin/pytest`. Expected: all PASS, no new warnings.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "DSLR: gphoto2 in install.sh; CLAUDE.md and spec updated"`

---

### Task 12: Rollout on the Pi (needs the user)

These steps touch the Pi and the camera; run them with the user, not unattended.

- [ ] **Step 1: Deploy with the feature off** — `deploy/deploy.sh`, then `ssh yassine@birdy.local 'journalctl -u birdy -n 30 --no-pager'`. Expected: the service starts, `dslr=False` in the "birdy starting" line, no errors.

- [ ] **Step 2: Check the camera** — `ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy dslr info'`. Expected: seven values, no warnings. Then, with the 50D switched off, run it again and confirm the error text is reported and (in `tests/test_dslr.py` terms) maps to "not connected": if gphoto2's wording differs from `No camera found` / `Could not detect any camera`, add its wording to `_state_for` and a test case, and redeploy.

- [ ] **Step 3: Aim and set the region** — with the 50D mounted and focused on the feeder: `ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy dslr shot'`, copy `data/dslr-test-region.jpg` to the Mac and look at it. Set `dslr.region` on `/settings` so the box frames the feeder tightly; repeat until it does.

- [ ] **Step 4: Enable** — on the Pi set `[dslr] enabled = true` in `~/pi-birdy/birdy.toml`, `sudo systemctl restart birdy` (the user runs sudo), and check the log for `50D connected: …`.

- [ ] **Step 5: Verify live** — wait for a bird (or wave a printed bird photo at the feeder). Expected in the log: `capture N: 50D 1.2x s, bird 0.xx`; on the grid the tile shows the 50D photo, a `50D` tag and a `Pi photo` link.

- [ ] **Step 6: After a week** — on the Pi:

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('file:data/birdy.db?mode=ro', uri=True)
print(c.execute('select count(*), sum(dslr_x is not null) from captures where dslr_path is not null').fetchone())"
```

(shots, shots with a bird found). If fewer than half have a bird, the spec's *Later* section applies. Note that frames purged by retention after `non_bird_days` are no longer counted, so run this within that window or raise the window first.
