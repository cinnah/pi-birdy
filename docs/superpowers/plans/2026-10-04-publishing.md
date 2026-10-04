# Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Pi publishes its best bird photos to a public static gallery at https://birdy.akibeo.com (home with a column per site, a dashboard per site, a page per species), rule-based with manual overrides from the review page.

**Architecture:** A `published` state per capture in SQLite, set by a pure ranking policy (`rank_day`) whenever a bird gets a species, and by publish/hide buttons on the review page. A `Publisher` thread keeps a local mirror (`data/publish/`: web-sized photos, thumbs, `manifest.json`) and rsyncs it over SSH to `birdy@<host>:<remote_dir>` where an `rrsync`-restricted key pins it to `/srv/birdy/sites/<site_id>/`. Fixed pages in `gallery/` (vanilla HTML/JS/CSS) render everything in the browser from `sites.json` plus every site's manifest. The server runs no code; its Caddy serves the folder.

**Tech Stack:** Python 3.11/3.13, SQLite, OpenCV (resize), rsync/ssh (subprocess), FastAPI/Jinja2, vanilla JS. No new Python dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-publishing-design.md`. Two small deviations, both recorded in Task 10: the manifest carries `site.utc_offset_minutes` instead of a zone name (Python can't name the Pi's zone portably), and photo entries have no `w`/`h` (reading every file each run is too slow; CSS handles aspect).

**Conventions (read `CLAUDE.md` first):** run tests with `.venv/bin/pytest`; only `birdy/capture.py` imports picamera2; every config key goes in `config.py` + `birdy.toml.example` + `settings.py` when live; commit after each task with the attribution trailer the session shows. Never put real hosts' secrets in the repo (there are none in this feature; the SSH private key lives only on the Pi).

---

## File structure

| File | Responsibility |
|---|---|
| `birdy/config.py` | `PublishConfig` dataclass + validation (Task 1) |
| `birdy/settings.py`, `birdy/watcher.py` | two live keys on the settings page; six restart-only keys (Tasks 1, 6) |
| `birdy/store.py` | `published`/`web_path` columns, candidate/published/stats queries (Task 2) |
| `birdy/publish.py` | **new**: `PublishSettings`, `rank_day`, `local_day_bounds`, `Publisher` (mirror, manifest, rsync, thread) (Tasks 3, 4) |
| `birdy/pipeline.py` | calls `publisher.evaluate` after live and backfill classification (Task 5) |
| `birdy/web/app.py`, `templates/index.html`, `templates/settings.html` | badges + buttons + `POST /captures/{id}/publish`, `filter=published`, Publishing settings group (Tasks 6, 7) |
| `birdy/cli.py`, `deploy/install.sh`, `deploy/gallery.sh` | wiring, `birdy publish`, key generation, page deploy (Task 8) |
| `gallery/index.html`, `site.html`, `species.html`, `app.js`, `style.css` | **new**: the static gallery (Task 9) |
| `CLAUDE.md`, spec | docs, server setup, Pi verification (Task 10) |
| `tests/test_config.py`, `test_settings.py`, `test_store.py`, `test_publish.py` (new), `test_pipeline.py`, `test_web.py`, `test_cli.py`, `test_gallery.py` (new), `test_watcher.py` | tests |

---

### Task 1: `[publish]` config section

**Files:**
- Modify: `birdy/config.py` (dataclass after `TelegramConfig`, parsing after the telegram block, `Config` field)
- Modify: `birdy/watcher.py:16-24` (`_restart_only`)
- Modify: `birdy/settings.py` (`config_to_dict` only; the live fields come in Task 6)
- Modify: `birdy.toml.example` (append section)
- Test: `tests/test_config.py`, `tests/test_watcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def _write(tmp_path, extra):
    p = tmp_path / "birdy.toml"
    p.write_text('[site]\nname = "t"\n[motion]\nroi = [[0,0],[10,0],[10,10]]\n' + extra)
    return p


def test_publish_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, ""))
    pub = cfg.publish
    assert pub.enabled is False
    assert pub.host == ""
    assert pub.user == "birdy"
    assert pub.remote_dir == "/"
    assert pub.site_id == "home"
    assert pub.ssh_key == Path("~/.ssh/birdy_publish").expanduser()
    assert pub.min_confidence == 0.6
    assert pub.per_species_per_day == 3


def test_publish_enabled_requires_host(tmp_path):
    with pytest.raises(ConfigError, match="publish.host is required"):
        load_config(_write(tmp_path, '[publish]\nenabled = true\n'))


@pytest.mark.parametrize("site_id", ["", "Home", "a b", "x" * 33, "home/../x"])
def test_publish_site_id_pattern(tmp_path, site_id):
    with pytest.raises(ConfigError, match="publish.site_id"):
        load_config(_write(tmp_path, f'[publish]\nsite_id = "{site_id}"\n'))


@pytest.mark.parametrize("line, key", [
    ("min_confidence = 1.5", "publish.min_confidence"),
    ("min_confidence = -0.1", "publish.min_confidence"),
    ("per_species_per_day = 0", "publish.per_species_per_day"),
    ("per_species_per_day = 21", "publish.per_species_per_day"),
    ("per_species_per_day = 2.5", "publish.per_species_per_day"),
    ("enabled = 1", "publish.enabled"),
])
def test_publish_ranges(tmp_path, line, key):
    with pytest.raises(ConfigError, match=key):
        load_config(_write(tmp_path, f"[publish]\n{line}\n"))


def test_publish_full_section(tmp_path):
    cfg = load_config(_write(tmp_path, '[publish]\nenabled = true\nhost = "91.98.229.209"\nuser = "birdy"\n'
                                       'remote_dir = "/"\nsite_id = "sintra-1"\nssh_key = "keys/pub"\n'
                                       'min_confidence = 0.75\nper_species_per_day = 5\n'))
    assert cfg.publish.enabled and cfg.publish.host == "91.98.229.209"
    assert cfg.publish.site_id == "sintra-1"
    assert cfg.publish.ssh_key == Path("keys/pub")   # relative paths are left as given (expanduser only)
    assert cfg.publish.min_confidence == 0.75 and cfg.publish.per_species_per_day == 5
```

Make sure `tests/test_config.py` imports `Path`, `pytest`, `ConfigError`, `load_config` (it already imports most; add `from pathlib import Path` if missing).

Append to `tests/test_watcher.py` (look at `test_restart_only_change_warns` for the pattern: it rewrites the file and calls `check_once`):

```python
def test_publish_restart_only_keys_warn(config_path, live, caplog):
    from birdy.watcher import ConfigWatcher
    import logging
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[publish]\nsite_id = "other"\n')
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart birdy" in caplog.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py tests/test_watcher.py -q -k "publish"`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'publish'`.

- [ ] **Step 3: Implement**

`birdy/config.py` — add `import re` at the top; add after `TelegramConfig`:

```python
@dataclass(frozen=True)
class PublishConfig:
    enabled: bool              # restart-only; the publisher thread is only built when true
    host: str                  # restart-only; gallery server
    user: str                  # restart-only; SSH user on the server
    remote_dir: str            # restart-only; rsync target as the server key sees it ("/" with rrsync)
    site_id: str               # restart-only; folder under sites/ and id in sites.json
    ssh_key: Path              # restart-only; private key (~ expanded)
    min_confidence: float      # live
    per_species_per_day: int   # live
```

Add `publish: PublishConfig` as the last field of `Config`. In `load_config`, after the telegram block:

```python
    pub = raw.get("publish", {})
    pub_enabled = _bool(pub, "publish", "enabled", False)
    pub_host = _str(pub, "publish", "host", "")
    pub_user = _str(pub, "publish", "user", "birdy")
    pub_remote = _str(pub, "publish", "remote_dir", "/")
    pub_site = _str(pub, "publish", "site_id", "home")
    _check(re.fullmatch(r"[a-z0-9-]{1,32}", pub_site) is not None, "publish.site_id must match [a-z0-9-]{1,32}")
    pub_key = Path(_str(pub, "publish", "ssh_key", "~/.ssh/birdy_publish")).expanduser()
    pub_min = _num(pub, "publish", "min_confidence", 0.6, float)
    _check(0.0 <= pub_min <= 1.0, "publish.min_confidence must be between 0 and 1")
    pub_n = _num(pub, "publish", "per_species_per_day", 3, int)
    _check(1 <= pub_n <= 20, "publish.per_species_per_day must be between 1 and 20")
    if pub_enabled:
        _check(bool(pub_host), "publish.host is required when publish.enabled = true")
        _check(bool(pub_user), "publish.user is required when publish.enabled = true")
        _check(bool(pub_remote), "publish.remote_dir is required when publish.enabled = true")
```

and in the `Config(...)` constructor:

```python
        publish=PublishConfig(
            enabled=pub_enabled, host=pub_host, user=pub_user, remote_dir=pub_remote, site_id=pub_site,
            ssh_key=pub_key, min_confidence=pub_min, per_species_per_day=pub_n,
        ),
```

Note `_num(..., int)` with `2.5` raises because `int(2.5)` would truncate silently — check: `int(2.5)` returns 2 in Python. So the existing `_num` **does not** reject `2.5` for int. Fix `_num` so it does: before the `cast(v)` call add

```python
    if cast is int and isinstance(v, float) and not v.is_integer():
        raise ConfigError(f"{name}.{key} must be an integer")
```

(`test_publish_ranges[per_species_per_day = 2.5]` matches `publish.per_species_per_day` either way.) Run the whole suite after this change; no existing test depends on truncation.

`birdy/watcher.py` — extend `_restart_only`:

```python
def _restart_only(cfg: Config) -> tuple:
    """Settings only read at startup; changing them on disk needs a restart."""
    c = cfg.classifier
    p = cfg.publish
    return (
        cfg.camera, cfg.web, cfg.storage, c.model, c.labels, c.bird_classes,
        cfg.species.model, cfg.species.labels,
        cfg.telegram.enabled, cfg.telegram.bot_token, cfg.telegram.chat_id, cfg.telegram.base_url,
        p.enabled, p.host, p.user, p.remote_dir, p.site_id, p.ssh_key,
    )
```

and change the warning text to `"%s: camera/web/storage/model/telegram/publish settings changed; restart birdy to apply them"`.

`birdy/settings.py` — add to the dict returned by `config_to_dict`, after `"telegram"`:

```python
        "publish": {
            "enabled": config.publish.enabled,
            "host": config.publish.host,
            "user": config.publish.user,
            "remote_dir": config.publish.remote_dir,
            "site_id": config.publish.site_id,
            "ssh_key": str(config.publish.ssh_key),
            "min_confidence": config.publish.min_confidence,
            "per_species_per_day": config.publish.per_species_per_day,
        },
```

`birdy.toml.example` — append:

```toml
[publish]
# Public gallery (https://birdy.akibeo.com). The Pi rsyncs web-sized copies of its best
# bird photos to the server; see CLAUDE.md "Gallery setup".
enabled = false                   # restart-only; when false nothing is published
host = ""                         # restart-only; the gallery server, e.g. "91.98.229.209"
user = "birdy"                    # restart-only; SSH user on the server
remote_dir = "/"                  # restart-only; rsync target as the server key sees it: "/" with the
                                  # rrsync-restricted key, or the full path if the key is unrestricted
site_id = "home"                  # restart-only; this site's id: folder under sites/ and id in sites.json; [a-z0-9-]{1,32}
ssh_key = "~/.ssh/birdy_publish"  # restart-only; private key (install.sh generates it)
min_confidence = 0.6              # live; species confidence needed to be auto-published (0-1)
per_species_per_day = 3           # live; best N photos per species per local day (1-20)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -q`
Expected: all pass, including `test_config_to_dict_covers_every_field` (it iterates `Config` fields, so the new section must be in `config_to_dict`).

- [ ] **Step 5: Commit**

```bash
git add birdy/config.py birdy/watcher.py birdy/settings.py birdy.toml.example tests/test_config.py tests/test_watcher.py
git commit -m "Config: [publish] section"
```

---

### Task 2: Store — `published`/`web_path` columns and publish queries

**Files:**
- Modify: `birdy/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py` (it has a `store` fixture and uses `datetime`; add imports as needed: `sqlite3`, `timedelta`, `timezone`, `Path`):

```python
def _bird(store, when, species, conf, is_bird=True):
    _, rel = store.photo_path_for(when)
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, is_bird, 0.9, species, conf)
    return cid


def test_fresh_db_has_publish_columns(store):
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(captures)")}
    assert {"published", "web_path"} <= cols


def test_migration_adds_publish_columns_to_old_db(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    conn = sqlite3.connect(data / "birdy.db")
    conn.executescript(
        "CREATE TABLE captures (id INTEGER PRIMARY KEY, captured_at TEXT NOT NULL, path TEXT NOT NULL,"
        " thumb_path TEXT, motion_score INTEGER NOT NULL, is_bird INTEGER, confidence REAL, species TEXT,"
        " species_confidence REAL, status TEXT NOT NULL DEFAULT 'pending', species_checked INTEGER NOT NULL DEFAULT 0);"
        "INSERT INTO captures (captured_at, path, motion_score, status) VALUES ('2026-10-01T10:00:00.000+00:00', 'a.jpg', 1, 'done');"
    )
    conn.commit()
    conn.close()
    s = Store(data)
    c = s.get(1)
    assert c.published is None and c.web_path is None


def test_set_published_and_manual_states(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    a = _bird(store, t, "Robin", 0.9)
    b = _bird(store, t + timedelta(minutes=1), "Robin", 0.8)
    c = _bird(store, t + timedelta(minutes=2), "Robin", 0.7)
    store.set_published(a, "auto")
    store.set_published(b, "forced")
    store.set_published(c, "hidden")
    assert [x.id for x in store.published()] == [b, a]          # newest first, hidden excluded
    assert store.manual_states() == {b: "forced", c: "hidden"}
    store.set_published(a, None)
    assert store.get(a).published is None
    with pytest.raises(ValueError):
        store.set_published(a, "maybe")


def test_species_day_candidates_window_and_order(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    low = _bird(store, t, "Robin", 0.5)
    high = _bird(store, t + timedelta(minutes=1), "Robin", 0.9)
    _bird(store, t, "Wren", 0.9)                           # other species
    _bird(store, t + timedelta(days=1), "Robin", 0.95)     # outside the window
    nb = _bird(store, t + timedelta(minutes=2), "Robin", 0.99, is_bird=False)  # non-bird with a label
    pend = store.add_capture(t + timedelta(minutes=3), store.photo_path_for(t + timedelta(minutes=3))[1], 1)  # pending
    got = store.species_day_candidates("Robin", t, t + timedelta(days=1))
    assert [c.id for c in got] == [high, low]
    assert nb not in [c.id for c in got] and pend not in [c.id for c in got]


def test_set_web_path_and_stale_web_paths(store):
    t = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
    a = _bird(store, t, "Robin", 0.9)
    b = _bird(store, t + timedelta(minutes=1), "Robin", 0.9)
    store.set_published(a, "auto")
    store.set_web_path(a, "2026/10/04/x.jpg")
    store.set_web_path(b, "2026/10/04/y.jpg")   # has a web copy but is not published
    assert store.get(a).web_path == "2026/10/04/x.jpg"
    assert store.stale_web_paths() == [(b, "2026/10/04/y.jpg")]
    store.set_published(a, "hidden")
    assert sorted(store.stale_web_paths()) == [(a, "2026/10/04/x.jpg"), (b, "2026/10/04/y.jpg")]


def test_publish_stats(store):
    now = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc).astimezone()
    today9 = now.replace(hour=9, minute=0)
    _bird(store, today9, "Robin", 0.9)
    _bird(store, today9 + timedelta(minutes=5), "Robin", 0.8)
    _bird(store, today9 - timedelta(days=1), "Wren", 0.9)
    _bird(store, today9 - timedelta(days=40), "Wren", 0.9)   # outside per_day but counted in totals
    _bird(store, today9, None, None)                          # bird without species
    _bird(store, today9, "Cat", 0.9, is_bird=False)           # not a bird: ignored everywhere
    s = store.publish_stats(now)
    assert s["total_birds"] == 5
    assert s["species_count"] == 2
    assert s["first_capture"] == (today9 - timedelta(days=40)).date().isoformat()
    assert s["last_capture_at"] == (today9 + timedelta(minutes=5)).isoformat(timespec="seconds")
    assert len(s["per_day"]) == 30 and s["per_day"][-1] == {"day": now.date().isoformat(), "birds": 3}
    assert s["per_day"][-2]["birds"] == 1
    assert len(s["per_hour"]) == 24 and s["per_hour"][9] == 4
    assert s["species"][0] == {
        "name": "Robin", "count": 2,
        "first": today9.date().isoformat(), "last": (today9 + timedelta(minutes=5)).isoformat(timespec="seconds"),
    }
    assert s["species"][1]["name"] == "Wren" and s["species"][1]["count"] == 2


def test_publish_stats_empty(store):
    s = store.publish_stats(datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc))
    assert s["total_birds"] == 0 and s["species"] == [] and s["first_capture"] is None
    assert len(s["per_day"]) == 30 and sum(d["birds"] for d in s["per_day"]) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_store.py -q -k "publish or web_path or candidates"`
Expected: FAIL (`AttributeError`/`AssertionError` on missing columns and methods).

- [ ] **Step 3: Implement**

In `birdy/store.py`:

Add to `FILTERS`:

```python
    "published": "published IN ('auto', 'forced')",
```

Add `PUBLISHED_STATES = (None, "auto", "forced", "hidden")` under `THUMB_WIDTH`.

Add to `Capture` (at the end, with defaults so existing constructions keep working):

```python
    published: str | None = None   # None | 'auto' | 'forced' | 'hidden'
    web_path: str | None = None    # relative path of the web-sized copy under data/publish/photos
```

In `_row_to_capture` add `published=row["published"], web_path=row["web_path"],`.

Replace the migration block in `__init__` with a loop:

```python
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(captures)")}
            for name, decl in (
                ("species_checked", "INTEGER NOT NULL DEFAULT 0"),
                ("published", "TEXT"),
                ("web_path", "TEXT"),
            ):
                if name not in cols:
                    try:
                        self._conn.execute(f"ALTER TABLE captures ADD COLUMN {name} {decl}")
                    except sqlite3.OperationalError as e:
                        # Another process (service vs `birdy purge`) migrated between our check and ALTER.
                        if "duplicate column" not in str(e):
                            raise
            self._conn.execute("CREATE INDEX IF NOT EXISTS captures_published ON captures(published)")
```

Add a `# --- publishing ---` block before `# --- maintenance ---`:

```python
    def set_published(self, capture_id: int, state: str | None) -> None:
        if state not in PUBLISHED_STATES:
            raise ValueError(f"bad published state {state!r}")
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET published = ? WHERE id = ?", (state, capture_id))

    def set_web_path(self, capture_id: int, web_path: str | None) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE captures SET web_path = ? WHERE id = ?", (web_path, capture_id))

    def species_day_candidates(self, species: str, start: datetime, end: datetime) -> list[Capture]:
        """Classified birds of one species captured in [start, end), best confidence first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM captures WHERE is_bird = 1 AND status = 'done' AND species = ?"
                " AND captured_at >= ? AND captured_at < ? ORDER BY species_confidence DESC, id ASC",
                (species, _to_db_time(start), _to_db_time(end)),
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def published(self) -> list[Capture]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM captures WHERE published IN ('auto', 'forced') ORDER BY captured_at DESC, id DESC"
            ).fetchall()
        return [_row_to_capture(r) for r in rows]

    def manual_states(self) -> dict[int, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, published FROM captures WHERE published IN ('forced', 'hidden')"
            ).fetchall()
        return {int(r["id"]): r["published"] for r in rows}

    def stale_web_paths(self) -> list[tuple[int, str]]:
        """Captures that still have a web copy but are no longer published."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, web_path FROM captures WHERE web_path IS NOT NULL"
                " AND (published IS NULL OR published = 'hidden') ORDER BY id"
            ).fetchall()
        return [(int(r["id"]), r["web_path"]) for r in rows]

    def publish_stats(self, now: datetime | None = None, days: int = 30) -> dict:
        """Aggregate numbers for the gallery dashboard, over ALL bird captures (published or not).

        Times are local (the Pi's zone); `per_day` has `days` entries ending today, `per_hour` 24.
        """
        now = (now or datetime.now(timezone.utc)).astimezone()
        with self._lock:
            rows = self._conn.execute("SELECT captured_at, species FROM captures WHERE is_bird = 1").fetchall()
        today = now.date()
        per_day = {today - timedelta(days=i): 0 for i in range(days)}
        per_hour = [0] * 24
        by_species: dict[str, list] = {}  # name -> [count, first, last]
        first = last = None
        for r in rows:
            t = _from_db_time(r["captured_at"]).astimezone()
            d = t.date()
            if d in per_day:
                per_day[d] += 1
            per_hour[t.hour] += 1
            first = t if first is None or t < first else first
            last = t if last is None or t > last else last
            if r["species"]:
                s = by_species.setdefault(r["species"], [0, t, t])
                s[0] += 1
                s[1] = min(s[1], t)
                s[2] = max(s[2], t)
        species = [
            {"name": n, "count": c, "first": f.date().isoformat(), "last": l.isoformat(timespec="seconds")}
            for n, (c, f, l) in by_species.items()
        ]
        species.sort(key=lambda s: (-s["count"], s["name"]))
        return {
            "total_birds": len(rows),
            "species_count": len(by_species),
            "first_capture": None if first is None else first.date().isoformat(),
            "last_capture_at": None if last is None else last.isoformat(timespec="seconds"),
            "species": species,
            "per_day": [{"day": d.isoformat(), "birds": per_day[d]} for d in sorted(per_day)],
            "per_hour": per_hour,
        }
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -q`
Expected: all pass. (`test_web` has a `filter must be one of` 400 test that lists `all, birds, non-birds`; if it asserts on the exact message, leave the message for now — Task 7 updates it.)

- [ ] **Step 5: Commit**

```bash
git add birdy/store.py tests/test_store.py
git commit -m "Store: published/web_path columns and publish queries"
```

---

### Task 3: Publish policy (`rank_day`)

**Files:**
- Create: `birdy/publish.py` (policy part; the `Publisher` class comes in Task 4)
- Test: `tests/test_publish.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_publish.py`:

```python
"""Publish policy and Publisher (mirror, manifest, rsync, thread)."""

from datetime import datetime, timedelta, timezone

from birdy.publish import PublishSettings, local_day_bounds, rank_day
from birdy.store import Capture

T0 = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
S = PublishSettings(min_confidence=0.6, per_species_per_day=2)


def cap(i, conf, published=None, species="Robin"):
    return Capture(
        id=i, captured_at=T0 + timedelta(minutes=i), path=f"p{i}.jpg", thumb_path=None, motion_score=1,
        is_bird=True, confidence=0.9, species=species, species_confidence=conf, status="done",
        species_checked=True, published=published,
    )


def test_rank_day_top_n_by_confidence_then_id():
    got = rank_day([cap(1, 0.7), cap(2, 0.9), cap(3, 0.9), cap(4, 0.8)], {}, S)
    assert got == {2: "auto", 3: "auto", 4: None, 1: None}


def test_rank_day_threshold():
    got = rank_day([cap(1, 0.59), cap(2, 0.6), cap(3, None)], {}, S)
    assert got == {1: None, 2: "auto", 3: None}


def test_rank_day_demotes_weaker_auto():
    got = rank_day([cap(1, 0.7, "auto"), cap(2, 0.8, "auto"), cap(3, 0.95)], {}, S)
    assert got == {3: "auto", 2: "auto", 1: None}


def test_rank_day_manual_states_win_and_forced_does_not_take_a_slot():
    manual = {1: "forced", 2: "hidden"}
    got = rank_day([cap(1, 0.1), cap(2, 0.99), cap(3, 0.7), cap(4, 0.65), cap(5, 0.61)], manual, S)
    assert got == {1: "forced", 2: "hidden", 3: "auto", 4: "auto", 5: None}


def test_rank_day_n_bounds():
    caps = [cap(i, 0.9) for i in range(1, 25)]
    assert sum(v == "auto" for v in rank_day(caps, {}, PublishSettings(0.6, 1)).values()) == 1
    assert sum(v == "auto" for v in rank_day(caps, {}, PublishSettings(0.6, 20)).values()) == 20


def test_rank_day_empty():
    assert rank_day([], {}, S) == {}


def test_local_day_bounds():
    when = datetime(2026, 10, 4, 23, 30, tzinfo=timezone.utc)
    start, end = local_day_bounds(when)
    local = when.astimezone()
    assert start == local.replace(hour=0, minute=0, second=0, microsecond=0)
    assert end - start == timedelta(days=1)
    assert start.tzinfo is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_publish.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'birdy.publish'`.

- [ ] **Step 3: Implement**

Create `birdy/publish.py`:

```python
"""Publishing: which bird photos go to the public gallery, the local mirror, the manifest and the rsync push.

`rank_day` is pure. `Publisher` (further down) owns the DB state changes, the mirror folder
`data/publish/` and a worker thread that pushes it to the server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from birdy.config import Config
from birdy.store import Capture

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishSettings:
    """The live-tunable subset, read from live.get().publish on each evaluation."""

    min_confidence: float
    per_species_per_day: int

    @classmethod
    def from_config(cls, cfg: Config) -> "PublishSettings":
        return cls(cfg.publish.min_confidence, cfg.publish.per_species_per_day)


def local_day_bounds(when: datetime) -> tuple[datetime, datetime]:
    """[start, end) of the local calendar day containing `when` (the Pi's zone), tz-aware."""
    start = when.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def rank_day(candidates: list[Capture], manual: dict[int, str], settings: PublishSettings) -> dict[int, str | None]:
    """Desired `published` value for every candidate of one species on one local day.

    Manual states ('forced'/'hidden') are kept as they are and 'forced' does not use a slot.
    Of the rest, those at or above `min_confidence`, best confidence first (ties: lower id first),
    the first `per_species_per_day` get 'auto'; the others None.
    """
    desired: dict[int, str | None] = {}
    eligible: list[Capture] = []
    for c in candidates:
        m = manual.get(c.id)
        if m in ("forced", "hidden"):
            desired[c.id] = m
        elif c.species_confidence is not None and c.species_confidence >= settings.min_confidence:
            eligible.append(c)
        else:
            desired[c.id] = None
    eligible.sort(key=lambda c: (-(c.species_confidence or 0.0), c.id))
    for i, c in enumerate(eligible):
        desired[c.id] = "auto" if i < settings.per_species_per_day else None
    return desired
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_publish.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add birdy/publish.py tests/test_publish.py
git commit -m "Publish: rank_day policy"
```

---

### Task 4: `Publisher` — evaluate, mirror, manifest, rsync, thread

**Files:**
- Modify: `birdy/publish.py`
- Test: `tests/test_publish.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_publish.py`:

```python
import dataclasses
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.publish import Publisher, rsync_command
from birdy.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def enabled(config, **kw):
    base = dict(enabled=True, host="gallery.example", user="birdy", remote_dir="/", site_id="home",
                ssh_key=Path("/keys/pub"), min_confidence=0.6, per_species_per_day=2)
    base.update(kw)
    return dataclasses.replace(config, publish=dataclasses.replace(config.publish, **base))


class FakeRunner:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self.rc


def bird(store, when, species, conf, size=(2464, 3280), is_bird=True, write=True):
    abs_path, rel = store.photo_path_for(when)
    if write:
        cv2.imwrite(str(abs_path), np.full((*size, 3), 120, np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, is_bird, 0.9, species, conf)
    return store.get(cid)


def make(live, store, config, runner=None, **kw):
    live.replace(enabled(config, **kw))
    return Publisher(live, store, store.data_dir / "publish", runner=runner or FakeRunner(), debounce=0.05, max_interval=0.5)


# --- evaluate / set_manual -------------------------------------------------

def test_evaluate_publishes_top_n_and_demotes(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.7)
    assert pub.evaluate(a) is True
    assert store.get(a.id).published == "auto"
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.8)
    assert pub.evaluate(b) is True
    c = bird(store, T0 + timedelta(minutes=2), "Robin", 0.9)
    assert pub.evaluate(c) is True
    assert [store.get(x.id).published for x in (a, b, c)] == [None, "auto", "auto"]
    assert pub.evaluate(c) is False  # nothing changed


def test_evaluate_ignores_non_birds_and_no_species(live, store, config):
    pub = make(live, store, config)
    assert pub.evaluate(bird(store, T0, None, None)) is False
    assert pub.evaluate(bird(store, T0, "Cat", 0.9, is_bird=False)) is False


def test_evaluate_reads_live_settings(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.7)
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.8)
    pub.evaluate(a)
    pub.evaluate(b)
    assert store.get(a.id).published is None
    live.replace(enabled(config, per_species_per_day=2))
    pub.evaluate(b)
    assert store.get(a.id).published == "auto"


def test_set_manual_forced_hidden_auto(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.9)
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.7)
    pub.evaluate(b)
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)
    pub.set_manual(a.id, "hidden")           # frees the slot -> b becomes auto
    assert (store.get(a.id).published, store.get(b.id).published) == ("hidden", "auto")
    pub.set_manual(b.id, "forced")           # forced doesn't use a slot, but a is hidden
    assert (store.get(a.id).published, store.get(b.id).published) == ("hidden", "forced")
    pub.set_manual(a.id, None)               # back to the rule -> a takes the slot
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", "forced")
    pub.set_manual(b.id, None)
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)


def test_set_manual_without_species_and_unknown_id(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, None, None)
    pub.set_manual(a.id, "forced")
    assert store.get(a.id).published == "forced"
    pub.set_manual(a.id, None)
    assert store.get(a.id).published is None
    with pytest.raises(KeyError):
        pub.set_manual(999, "forced")
    with pytest.raises(ValueError):
        pub.set_manual(a.id, "auto")


# --- run_once: mirror + manifest + rsync -------------------------------------

def test_run_once_builds_mirror_and_manifest_and_rsyncs(live, store, config):
    runner = FakeRunner()
    pub = make(live, store, config, runner=runner)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    assert pub.run_once(now=T0 + timedelta(hours=1)) is True

    web = pub.photos_dir / a.path
    assert web.exists()
    img = cv2.imread(str(web))
    assert max(img.shape[:2]) == 1600 and img.shape[0] < img.shape[1]
    assert (pub.thumbs_dir / a.path).exists()
    assert store.get(a.id).web_path == a.path

    m = json.loads((pub.mirror_dir / "manifest.json").read_text())
    assert m["version"] == 1 and m["site"]["id"] == "home" and m["site"]["name"] == "test"
    assert m["site"]["lat"] is None and isinstance(m["site"]["utc_offset_minutes"], int)
    assert m["site"]["since"] == T0.astimezone().date().isoformat()
    assert m["photos"] == [{
        "id": a.id, "captured_at": a.captured_at.astimezone().isoformat(timespec="seconds"),
        "species": "Robin", "confidence": 0.9, "photo": f"photos/{a.path}", "thumb": f"thumbs/{a.path}", "state": "auto",
    }]
    assert m["stats"]["total_birds"] == 1 and len(m["stats"]["per_day"]) == 30
    assert m["generated_at"].startswith((T0 + timedelta(hours=1)).astimezone().date().isoformat())

    assert runner.calls == [rsync_command(pub.mirror_dir, live.get().publish)]
    cmd = runner.calls[0]
    assert cmd[:3] == ["rsync", "-az", "--delete"]
    assert cmd[-2] == f"{pub.mirror_dir}/" and cmd[-1] == "birdy@gallery.example:/"
    assert "-i /keys/pub" in cmd[cmd.index("-e") + 1] and "BatchMode=yes" in cmd[cmd.index("-e") + 1]


def test_run_once_never_upscales(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9, size=(60, 80))
    pub.evaluate(a)
    pub.run_once()
    assert cv2.imread(str(pub.photos_dir / a.path)).shape[:2] == (60, 80)


def test_run_once_removes_unpublished_files(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    pub.run_once()
    pub.set_manual(a.id, "hidden")
    pub.run_once()
    assert not (pub.photos_dir / a.path).exists() and not (pub.thumbs_dir / a.path).exists()
    assert store.get(a.id).web_path is None
    assert json.loads((pub.mirror_dir / "manifest.json").read_text())["photos"] == []


def test_run_once_hides_unreadable_original(live, store, config, caplog):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9, write=False)
    (store.photos_dir / a.path).write_bytes(b"")   # power-cut leftover
    pub.evaluate(a)
    with caplog.at_level(logging.WARNING):
        pub.run_once()
    assert store.get(a.id).published == "hidden"
    assert "hiding capture" in caplog.text
    assert json.loads((pub.mirror_dir / "manifest.json").read_text())["photos"] == []


def test_run_once_rsync_failure_logs_and_returns_false(live, store, config, caplog):
    pub = make(live, store, config, runner=FakeRunner(rc=255))
    pub.evaluate(bird(store, T0, "Robin", 0.9))
    with caplog.at_level(logging.WARNING):
        assert pub.run_once() is False
    assert "rsync" in caplog.text and "exit 255" in caplog.text


def test_run_once_thumb_falls_back_to_photo(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    (store.thumbs_dir / a.thumb_path).unlink()
    pub.evaluate(a)
    pub.run_once()
    m = json.loads((pub.mirror_dir / "manifest.json").read_text())
    assert m["photos"][0]["thumb"] == f"photos/{a.path}"


# --- thread -------------------------------------------------------------------

def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_thread_runs_on_start_and_coalesces_notifies(live, store, config):
    runner = FakeRunner()
    pub = make(live, store, config, runner=runner)
    pub.start()
    try:
        assert wait_until(lambda: len(runner.calls) == 1)      # reconcile on start
        pub.notify()
        pub.notify()
        pub.notify()
        assert wait_until(lambda: len(runner.calls) == 2)      # one run for the burst
        time.sleep(0.15)
        assert len(runner.calls) == 2
        assert wait_until(lambda: len(runner.calls) >= 3, timeout=2.0)   # max_interval tick
    finally:
        pub.stop()


def test_thread_survives_run_errors(live, store, config, caplog):
    class Boom:
        calls = 0

        def __call__(self, cmd):
            self.calls += 1
            raise RuntimeError("no")

    runner = Boom()
    pub = make(live, store, config, runner=runner)
    pub.start()
    try:
        with caplog.at_level(logging.ERROR):
            assert wait_until(lambda: runner.calls >= 1)
            pub.notify()
            assert wait_until(lambda: runner.calls >= 2)
        assert "publish run failed" in caplog.text
    finally:
        pub.stop()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_publish.py -q`
Expected: FAIL with `ImportError: cannot import name 'Publisher'`.

- [ ] **Step 3: Implement**

Append to `birdy/publish.py` (and extend the imports at the top: `import json, os, shlex, shutil, subprocess, threading, time`, `from datetime import timezone`, `from pathlib import Path`, `from typing import Callable`, `import cv2`, `from birdy.live import LiveConfig`, `from birdy.store import Capture, Store`):

```python
WEB_MAX_SIDE = 1600
WEB_JPEG_QUALITY = 85
MANIFEST_VERSION = 1
RSYNC_TIMEOUT = 300.0

Runner = Callable[[list[str]], int]


def rsync_command(mirror_dir: Path, cfg) -> list[str]:
    """`cfg` is a PublishConfig. Trailing slash on the source: sync the folder's contents."""
    ssh = f"ssh -i {shlex.quote(str(cfg.ssh_key))} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    return ["rsync", "-az", "--delete", "--timeout=120", "-e", ssh, f"{mirror_dir}/", f"{cfg.user}@{cfg.host}:{cfg.remote_dir}"]


def _default_runner(cmd: list[str]) -> int:
    try:
        return subprocess.call(cmd, stdin=subprocess.DEVNULL, timeout=RSYNC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return -1
    except OSError as e:  # rsync/ssh not installed
        log.error("cannot run rsync: %s", e)
        return -2


class Publisher:
    """Decides what is published (DB), keeps the mirror folder and pushes it to the server."""

    def __init__(
        self,
        live: LiveConfig,
        store: Store,
        mirror_dir: Path,
        runner: Runner | None = None,
        debounce: float = 120.0,
        max_interval: float = 900.0,
    ) -> None:
        self._live = live
        self._store = store
        self.mirror_dir = Path(mirror_dir)
        self.photos_dir = self.mirror_dir / "photos"
        self.thumbs_dir = self.mirror_dir / "thumbs"
        self._runner = runner or _default_runner
        self._debounce = debounce
        self._max_interval = max_interval
        self._lock = threading.Lock()  # evaluate/set_manual come from the pipeline and web threads
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="publisher", daemon=True)

    # --- decisions -----------------------------------------------------------

    def evaluate(self, capture: Capture | None) -> bool:
        """Re-rank the species/day of `capture`; returns True if any published state changed."""
        if capture is None or not (capture.is_bird and capture.species and capture.status == "done"):
            return False
        with self._lock:
            changed = self._rank(capture.species, capture.captured_at)
        if changed:
            self.notify()
        return changed

    def _rank(self, species: str, when: datetime) -> bool:
        start, end = local_day_bounds(when)
        candidates = self._store.species_day_candidates(species, start, end)
        desired = rank_day(candidates, self._store.manual_states(), PublishSettings.from_config(self._live.get()))
        changed = False
        for c in candidates:
            if desired[c.id] != c.published:
                self._store.set_published(c.id, desired[c.id])
                changed = True
        return changed

    def set_manual(self, capture_id: int, state: str | None) -> None:
        """'forced' | 'hidden' | None (back to the rule). Raises KeyError for an unknown capture."""
        if state not in (None, "forced", "hidden"):
            raise ValueError(f"bad manual state {state!r}")
        c = self._store.get(capture_id)
        if c is None:
            raise KeyError(capture_id)
        with self._lock:
            self._store.set_published(capture_id, state)
            if c.is_bird and c.species and c.status == "done":
                self._rank(c.species, c.captured_at)  # a freed or taken slot moves other photos
        self.notify()

    def notify(self) -> None:
        self._wake.set()

    # --- mirror + manifest + push -----------------------------------------------

    def run_once(self, now: datetime | None = None) -> bool:
        cfg = self._live.get().publish
        photos = self._sync_mirror()
        self._write_manifest(photos, now)
        rc = self._runner(rsync_command(self.mirror_dir, cfg))
        if rc != 0:
            log.warning("publish: rsync to %s@%s failed (exit %d); will retry", cfg.user, cfg.host, rc)
            return False
        log.info("publish: %d photos synced to %s", len(photos), cfg.host)
        return True

    def _sync_mirror(self) -> list[Capture]:
        self.photos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        keep: set[str] = set()
        out: list[Capture] = []
        for c in self._store.published():
            if c.web_path is None or not (self.photos_dir / c.web_path).exists():
                if not self._make_web_copy(c):
                    self._store.set_published(c.id, "hidden")
                    log.warning("publish: cannot read %s; hiding capture %d", c.path, c.id)
                    continue
                self._store.set_web_path(c.id, c.path)
                c = self._store.get(c.id)
            keep.add(c.web_path)
            out.append(c)
        for root in (self.photos_dir, self.thumbs_dir):
            for f in root.rglob("*.jpg"):
                if f.relative_to(root).as_posix() not in keep:
                    f.unlink()
        for cid, _ in self._store.stale_web_paths():
            self._store.set_web_path(cid, None)
        return out

    def _make_web_copy(self, c: Capture) -> bool:
        img = cv2.imread(str(self._store.photos_dir / c.path))
        if img is None:
            return False
        h, w = img.shape[:2]
        scale = WEB_MAX_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        out = self.photos_dir / c.path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, WEB_JPEG_QUALITY]):
            return False
        src_thumb = None if c.thumb_path is None else self._store.thumbs_dir / c.thumb_path
        if src_thumb is not None and src_thumb.exists():
            dst = self.thumbs_dir / c.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_thumb, dst)
        return True

    def _photo_entry(self, c: Capture) -> dict:
        thumb = f"thumbs/{c.web_path}" if (self.thumbs_dir / c.web_path).exists() else f"photos/{c.web_path}"
        return {
            "id": c.id,
            "captured_at": c.captured_at.astimezone().isoformat(timespec="seconds"),
            "species": c.species,
            "confidence": None if c.species_confidence is None else round(c.species_confidence, 3),
            "photo": f"photos/{c.web_path}",
            "thumb": thumb,
            "state": c.published,
        }

    def build_manifest(self, photos: list[Capture], now: datetime | None = None) -> dict:
        cfg = self._live.get()
        now = (now or datetime.now(timezone.utc)).astimezone()
        stats = self._store.publish_stats(now)
        site = self._store.site() or {"name": cfg.site.name, "lat": cfg.site.lat, "lon": cfg.site.lon}
        return {
            "version": MANIFEST_VERSION,
            "generated_at": now.isoformat(timespec="seconds"),
            "site": {
                "id": cfg.publish.site_id,
                "name": site["name"],
                "lat": None if site["lat"] is None else round(site["lat"], 2),
                "lon": None if site["lon"] is None else round(site["lon"], 2),
                "since": stats["first_capture"],
                "utc_offset_minutes": int(now.utcoffset().total_seconds() // 60),
            },
            "photos": [self._photo_entry(c) for c in photos],
            "stats": stats,
        }

    def _write_manifest(self, photos: list[Capture], now: datetime | None) -> None:
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        target = self.mirror_dir / "manifest.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.build_manifest(photos, now), ensure_ascii=False, indent=1))
        os.replace(tmp, target)

    # --- thread -----------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        last_run = time.monotonic() - self._max_interval  # first run right after start: reconcile
        while not self._stop.is_set():
            remaining = last_run + self._max_interval - time.monotonic()
            woke = self._wake.wait(max(0.0, remaining))
            if self._stop.is_set():
                break
            if woke:
                # Coalesce a burst (a backfill sweep can publish many photos in a few seconds).
                if self._stop.wait(self._debounce):
                    break
            self._wake.clear()
            try:
                self.run_once()
            except Exception:
                log.exception("publish run failed")
            last_run = time.monotonic()
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_publish.py -q`
Expected: all pass. If `test_thread_runs_on_start_and_coalesces_notifies` is flaky on the max_interval tick, raise its last `timeout` to 3.0 (the fixture uses `max_interval=0.5`).

- [ ] **Step 5: Commit**

```bash
git add birdy/publish.py tests/test_publish.py
git commit -m "Publish: Publisher (mirror, manifest, rsync, worker thread)"
```

---

### Task 5: Pipeline calls the publisher

**Files:**
- Modify: `birdy/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py` (it already has `store`, `live`, `FakeClassifier`, `add`, `wait_until`; look at how the existing notifier tests build a classifier that returns a species — if there is a `SpeciesClassifier` helper reuse it, otherwise use the class below):

```python
class SpeciesFake:
    def __init__(self, species="Robin", conf=0.8, raise_exc=None):
        self.species, self.conf, self.raise_exc = species, conf, raise_exc

    def classify(self, image_path, roi_bbox, threshold=None):
        return Prediction(is_bird=True, confidence=0.9, species=self.species, species_confidence=self.conf)


class FakePublisher:
    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.evaluated = []

    def evaluate(self, capture):
        self.evaluated.append(capture)
        if self.raise_exc:
            raise self.raise_exc
        return True


def test_process_calls_publisher_with_stored_capture(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert [c.id for c in pub.evaluated] == [ev.capture_id]
    assert pub.evaluated[0].species == "Robin" and pub.evaluated[0].status == "done"


def test_publisher_error_does_not_mark_row_error(store, live, caplog):
    pub = FakePublisher(raise_exc=RuntimeError("boom"))
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    with caplog.at_level(logging.ERROR):
        p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).status == "done"
    assert "publish decision failed" in caplog.text


def test_backfill_calls_publisher(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    with store._lock:
        store._conn.execute("UPDATE captures SET species_checked = 0")
        store._conn.commit()
    p.sweep()
    assert [c.id for c in pub.evaluated] == [ev.capture_id]
    assert pub.evaluated[0].species == "Robin"


def test_backfill_without_species_does_not_call_publisher(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(species=None, conf=None), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    with store._lock:
        store._conn.execute("UPDATE captures SET species_checked = 0")
        store._conn.commit()
    p.sweep()
    assert pub.evaluated == []
```

Add `import logging` at the top of the test file if missing.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -q -k publisher`
Expected: FAIL with `TypeError: Pipeline.__init__() got an unexpected keyword argument 'publisher'`.

- [ ] **Step 3: Implement**

`birdy/pipeline.py`:

- In the `TYPE_CHECKING` block add `from birdy.publish import Publisher`.
- `__init__` gains `publisher: "Publisher | None" = None` after `notifier`, stored as `self._publisher = publisher`.
- At the end of `process()` (after the notifier block):

```python
        if self._publisher is not None:
            try:
                self._publisher.evaluate(self._store.get(capture_id))
            except Exception:
                log.exception("publish decision failed for capture %d", capture_id)
```

- Replace `_backfill_species` so the publisher call happens outside the classify/store `try`:

```python
    def _backfill_species(self, c: Capture) -> tuple[bool, bool]:
        """Re-classify an already-classified row; returns (got_species, rescued)."""
        cfg = self._live.get()
        path = self._store.photos_dir / c.path
        try:
            pred = self._classifier.classify(path, roi_bbox(cfg.motion.roi), threshold=cfg.classifier.threshold)
            # Never demote: a bird stays a bird. Confidence comes from the new verdict when it
            # is a bird (gate or rescue); the old confidence is kept only when it says non-bird.
            is_bird = bool(c.is_bird) or pred.is_bird
            confidence = pred.confidence if pred.is_bird else c.confidence
            self._store.update_classification(c.id, is_bird, confidence, pred.species, pred.species_confidence)
        except Exception:
            # Unlike live processing, a failure here must not flip a `done` row to `error`.
            log.exception("species backfill failed for capture %d; marking checked", c.id)
            try:
                self._store.mark_species_checked(c.id)
            except Exception:
                log.exception("mark_species_checked failed for capture %d", c.id)
            return False, False
        # Historical photos may be published too (unlike alerts, which are for live events only).
        if self._publisher is not None and pred.species:
            try:
                self._publisher.evaluate(self._store.get(c.id))
            except Exception:
                log.exception("publish decision failed for capture %d", c.id)
        return bool(pred.species), (not c.is_bird) and pred.is_bird
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/pipeline.py tests/test_pipeline.py
git commit -m "Pipeline: evaluate publishing after live and backfill classification"
```

---

### Task 6: Settings page — Publishing group

**Files:**
- Modify: `birdy/settings.py` (`FIELD_KEYS`, `SettingsPatch`, `apply_patch`, `parse_settings_form`)
- Modify: `birdy/web/app.py` (`settings_context` values + readonly)
- Modify: `birdy/web/templates/settings.html` (new group before the Save button)
- Test: `tests/test_settings.py`, `tests/test_web.py`

- [ ] **Step 1: Update the form fixtures and write the failing tests**

Every test that POSTs the full settings form must now include the two new fields. Run `grep -n "cooldown_minutes" tests/test_settings.py tests/test_web.py` and, next to every dict entry `"cooldown_minutes": ...`, add:

```python
        "publish_min_confidence": "0.6",
        "per_species_per_day": "3",
```

(Where the dict holds Python values rather than strings, use `0.6` and `3`.)

Append to `tests/test_settings.py`:

```python
def test_parse_form_publish_fields(form):
    patch = parse_settings_form({**form, "publish_min_confidence": "0.75", "per_species_per_day": "5"})
    assert patch.publish_min_confidence == 0.75 and patch.per_species_per_day == 5


def test_parse_form_publish_requires_integer(form):
    with pytest.raises(ConfigError, match="publish.per_species_per_day must be an integer"):
        parse_settings_form({**form, "per_species_per_day": "2.5"})


def test_save_applies_publish_fields_and_keeps_restart_only(tmp_path, config_path):
    config_path.write_text(config_path.read_text() + '\n[publish]\nenabled = false\nhost = "h"\nsite_id = "abc"\n')
    cfg = load_config(config_path)
    patch = parse_settings_form({**FORM, "publish_min_confidence": "0.8", "per_species_per_day": "4"})
    new = save_settings(config_path, cfg, patch)
    assert new.publish.min_confidence == 0.8 and new.publish.per_species_per_day == 4
    assert new.publish.host == "h" and new.publish.site_id == "abc"
```

Adapt `form`/`FORM` to whatever fixture name the file already uses for a full valid form dict (look at the existing telegram tests in that file and copy their pattern exactly).

Append to `tests/test_web.py`:

```python
def test_settings_page_shows_publishing_group(client):
    html = client.get("/settings").text
    assert "Publishing" in html
    assert 'name="publish_min_confidence"' in html and 'name="per_species_per_day"' in html
    assert "publish.site_id" in html and "publish.enabled" in html


def test_settings_post_saves_publish_fields(client, live, config_path):
    form = {**SETTINGS_FORM, "publish_min_confidence": "0.9", "per_species_per_day": "1"}
    r = client.post("/settings", data=form, follow_redirects=False)
    assert r.status_code == 303
    assert live.get().publish.min_confidence == 0.9 and live.get().publish.per_species_per_day == 1
```

(`SETTINGS_FORM`: use the name of the full-form dict that `test_settings_post_saves_applies_and_redirects` uses.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_settings.py tests/test_web.py -q -k "publish"`
Expected: FAIL (`TypeError` on unknown `SettingsPatch` field / missing inputs in HTML).

- [ ] **Step 3: Implement**

`birdy/settings.py`:

- `FIELD_KEYS` gains `"publish_min_confidence": "publish.min_confidence", "per_species_per_day": "publish.per_species_per_day",`.
- `SettingsPatch` gains `publish_min_confidence: float` and `per_species_per_day: int` (at the end).
- `apply_patch` gains, before `return data`:

```python
    data["publish"].update(
        min_confidence=patch.publish_min_confidence,
        per_species_per_day=patch.per_species_per_day,
    )
```

- `parse_settings_form` gains `publish_min_confidence=_num(form, "publish_min_confidence", float), per_species_per_day=_num(form, "per_species_per_day", int),`.

`birdy/web/app.py` `settings_context`: add to the default `values` dict:

```python
                "publish_min_confidence": cfg.publish.min_confidence,
                "per_species_per_day": cfg.publish.per_species_per_day,
```

and to `readonly`:

```python
                "publish.enabled": cfg.publish.enabled,
                "publish.host": cfg.publish.host or "–",
                "publish.user": cfg.publish.user,
                "publish.remote_dir": cfg.publish.remote_dir,
                "publish.site_id": cfg.publish.site_id,
```

`birdy/web/templates/settings.html`: after the `{{ num("cooldown_minutes", ...) }}` line and before the `<p><button class="primary" ...>Save</button>` paragraph, insert:

```html
<h2>Publishing</h2>
{{ num("publish_min_confidence", "min_confidence", 0, 1, 0.05) }}
{{ num("per_species_per_day", "per_species_per_day", 1, 20, 1) }}
<div class="hint">applies to days ranked from now on; earlier days keep their picks. Publish/Hide buttons on the home page override the rule per photo.</div>
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add birdy/settings.py birdy/web/app.py birdy/web/templates/settings.html tests/test_settings.py tests/test_web.py
git commit -m "Settings page: Publishing group (min_confidence, per_species_per_day)"
```

---

### Task 7: Review page — badges, Publish/Hide buttons, `filter=published`

**Files:**
- Modify: `birdy/web/app.py` (`create_app(publisher=)`, index context, new POST route)
- Modify: `birdy/web/templates/index.html`
- Test: `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web.py`:

```python
class FakePublisher:
    def __init__(self):
        self.calls = []

    def set_manual(self, capture_id, state):
        self.calls.append((capture_id, state))
        if capture_id == 999:
            raise KeyError(capture_id)


@pytest.fixture
def publisher():
    return FakePublisher()


@pytest.fixture
def pclient(store, state, live, config_path, classifier, notifier, publisher):
    return TestClient(create_app(store, state, live, config_path, classifier=classifier, notifier=notifier, publisher=publisher))


def test_index_without_publisher_has_no_buttons(client, store):
    add(store, 0, True)
    html = client.get("/").text
    assert "/publish" not in html and "filter=published" not in html


def test_index_renders_badges_and_buttons(pclient, store):
    a = add(store, 0, True)
    b = add(store, 1, True)
    c = add(store, 2, True)
    store.set_published(a.id, "auto")
    store.set_published(b.id, "forced")
    store.set_published(c.id, "hidden")
    html = pclient.get("/").text
    assert html.count('class="badge') == 3
    assert ">published<" in html and ">forced<" in html and ">hidden<" in html
    assert f'action="/captures/{a.id}/publish"' in html
    assert "filter=published" in html


def test_publish_post_sets_state_and_redirects_back(pclient, store, publisher):
    a = add(store, 0, True)
    r = pclient.post(f"/captures/{a.id}/publish", data={"state": "hidden", "next": "/?filter=birds&page=2"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?filter=birds&page=2"
    assert publisher.calls == [(a.id, "hidden")]
    r = pclient.post(f"/captures/{a.id}/publish", data={"state": "auto"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert publisher.calls[-1] == (a.id, None)


def test_publish_post_rejects_bad_input(pclient, store, publisher):
    a = add(store, 0, True)
    assert pclient.post(f"/captures/{a.id}/publish", data={"state": "maybe"}).status_code == 400
    assert pclient.post("/captures/999/publish", data={"state": "forced"}).status_code == 404
    r = pclient.post(f"/captures/{a.id}/publish", data={"state": "forced", "next": "//evil.example/x"}, follow_redirects=False)
    assert r.headers["location"] == "/"


def test_publish_post_503_without_publisher(client, store):
    a = add(store, 0, True)
    assert client.post(f"/captures/{a.id}/publish", data={"state": "forced"}).status_code == 503


def test_index_filter_published(pclient, store):
    a = add(store, 0, True)
    add(store, 1, True)
    store.set_published(a.id, "auto")
    html = pclient.get("/?filter=published").text
    assert html.count('class="tile"') == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_web.py -q -k "publish"`
Expected: FAIL with `TypeError: create_app() got an unexpected keyword argument 'publisher'`.

- [ ] **Step 3: Implement**

`birdy/web/app.py`:

- `create_app(..., notifier: Any | None = None, publisher: Any | None = None)`; docstring: "`publisher` has `set_manual(capture_id, state)`; None = publishing disabled."
- `index`: change the filter check to `if filter not in FILTERS: raise HTTPException(status_code=400, detail="filter must be one of " + ", ".join(FILTERS))` and add `"publishing": publisher is not None,` to the template context. (Update any test asserting the old message text.)
- New route after `settings_test_alert`:

```python
    @app.post("/captures/{capture_id}/publish")
    async def capture_publish(capture_id: int, request: Request):
        if publisher is None:
            raise HTTPException(status_code=503, detail="publishing is disabled: set [publish] in birdy.toml and restart")
        form = await request.form()
        state = str(form.get("state", ""))
        if state not in ("forced", "hidden", "auto"):
            raise HTTPException(status_code=400, detail="state must be forced, hidden or auto")
        nxt = str(form.get("next", "/"))
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        try:
            await run_in_threadpool(publisher.set_manual, capture_id, None if state == "auto" else state)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown capture") from None
        return RedirectResponse(nxt, status_code=303)
```

`birdy/web/templates/index.html`:

- Add CSS inside `<style>`:

```css
  .tile a.photo { display: block; color: inherit; text-decoration: none; }
  .pub { display: flex; gap: 6px; align-items: center; padding: 6px 8px; border-top: 1px solid var(--line); font-size: 12px; }
  .pub button { font-size: 12px; padding: 2px 8px; background: var(--card); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; cursor: pointer; }
  .badge { font-weight: bold; color: var(--bird); margin-right: auto; } .badge.off { color: var(--junk); }
```

- Nav filter list: `{% for f in (["all", "birds", "non-birds", "published"] if publishing else ["all", "birds", "non-birds"]) %}`.
- Replace the tile markup (`<a class="tile" ...>` … `</a>`) with:

```html
  <div class="tile">
    <a class="photo" href="/photos/{{ c.path }}">
      {% if c.thumb_path %}<img src="/thumbs/{{ c.thumb_path }}" alt="" loading="lazy">{% else %}<div class="nothumb">no thumbnail</div>{% endif %}
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
    {% if publishing %}
    <form method="post" action="/captures/{{ c.id }}/publish" class="pub">
      <input type="hidden" name="next" value="{{ request.url.path }}{% if request.url.query %}?{{ request.url.query }}{% endif %}">
      {% if c.published == 'auto' %}<span class="badge">published</span>
      {% elif c.published == 'forced' %}<span class="badge">forced</span>
      {% elif c.published == 'hidden' %}<span class="badge off">hidden</span>
      {% else %}<span class="badge" style="visibility:hidden">–</span>{% endif %}
      {% if c.published != 'forced' %}<button name="state" value="forced">Publish</button>{% endif %}
      {% if c.published != 'hidden' %}<button name="state" value="hidden">Hide</button>{% endif %}
      {% if c.published in ('forced', 'hidden') %}<button name="state" value="auto">Auto</button>{% endif %}
    </form>
    {% endif %}
  </div>
```

The invisible placeholder badge keeps the three-badge count test honest: adjust the test's `html.count('class="badge')` expectation to count only visible ones, i.e. assert `html.count('class="badge">') + html.count('class="badge off">') == 3` — or simpler, drop the placeholder and keep the row aligned with `margin-left: auto` on the first button. **Pick the second** (no placeholder; add `.pub button:first-of-type { margin-left: auto; }`) so the test stays as written.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest -q`
Expected: all pass. Check that `/?filter=non-birds` and the species dropdown still render (existing tests cover this).

- [ ] **Step 5: Commit**

```bash
git add birdy/web/app.py birdy/web/templates/index.html tests/test_web.py
git commit -m "Review page: publish/hide buttons, badges, published filter"
```

---

### Task 8: CLI wiring, `birdy publish`, key generation, gallery deploy script

**Files:**
- Modify: `birdy/cli.py`
- Modify: `deploy/install.sh`
- Create: `deploy/gallery.sh`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Look at `tests/test_cli.py` for how `cmd_run` wiring is tested (the Telegram task added a test that patches `TelegramNotifier`/`Pipeline`/`create_app` with fakes and checks the kwargs). Add, following that pattern:

```python
def test_cmd_run_builds_publisher_when_enabled(config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    cfg = dataclasses.replace(cfg, publish=dataclasses.replace(cfg.publish, enabled=True, host="h"))
    built = {}

    class FakePublisher:
        def __init__(self, live, store, mirror_dir):
            built["mirror_dir"] = mirror_dir
            built["self"] = self

        def start(self):
            built["started"] = True

        def stop(self, timeout=10.0):
            built["stopped"] = True

    monkeypatch.setattr(cli, "Publisher", FakePublisher)
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    seen = {}
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: seen.update(pipeline=pipeline))
    import birdy.web.app as webapp

    real_create_app = webapp.create_app
    monkeypatch.setattr(webapp, "create_app", lambda *a, **k: seen.update(publisher=k.get("publisher")) or real_create_app(*a, **k))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert built["mirror_dir"] == cfg.storage.data_dir / "publish" and built["started"] and built["stopped"]
    assert seen["publisher"] is built["self"] and seen["pipeline"]._publisher is built["self"]


def test_cmd_run_no_publisher_when_disabled(config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    monkeypatch.setattr(cli, "Publisher", lambda *a, **k: pytest.fail("Publisher built while disabled"))
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    seen = {}
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: seen.update(pipeline=pipeline))
    import birdy.web.app as webapp

    real_create_app = webapp.create_app
    monkeypatch.setattr(webapp, "create_app", lambda *a, **k: seen.update(publisher=k.get("publisher")) or real_create_app(*a, **k))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert seen["publisher"] is None and seen["pipeline"]._publisher is None


def test_publish_command_runs_once(tmp_path, monkeypatch, capsys):
    import birdy.cli as cli

    class FakePublisher:
        instances = []

        def __init__(self, live, store, mirror_dir, **kw):
            self.mirror_dir = mirror_dir
            FakePublisher.instances.append(self)

        def run_once(self):
            return True

    monkeypatch.setattr("birdy.cli.Publisher", FakePublisher)
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n[publish]\nenabled=true\nhost="h"\n')
    assert cli.main(["--config", str(cfg), "publish"]) == 0
    assert FakePublisher.instances[0].mirror_dir == tmp_path / "data" / "publish"
    assert "published" in capsys.readouterr().out


def test_publish_command_exit_2_when_disabled(tmp_path, capsys):
    import birdy.cli as cli
    cfg = tmp_path / "birdy.toml"
    cfg.write_text('[site]\nname="t"\n[motion]\nroi=[[0,0],[9,0],[9,9]]\n[storage]\ndata_dir="data"\n')
    assert cli.main(["--config", str(cfg), "publish"]) == 2
    assert "publish" in capsys.readouterr().err
```

(`_free_port` already exists in `tests/test_cli.py`; `test_cmd_run_builds_notifier_when_enabled` is the model for these two.) `test_publish_command_runs_once` patches `birdy.publish.Publisher`, but `cmd_publish` uses the name imported into `birdy.cli`, so patch `birdy.cli.Publisher` there instead (`monkeypatch.setattr("birdy.cli.Publisher", FakePublisher)`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli.py -q -k publish`
Expected: FAIL (`argparse` error "invalid choice: 'publish'", `AttributeError` on `birdy.cli.Publisher`).

- [ ] **Step 3: Implement**

`birdy/cli.py`:

- Module docstring: `birdy run | roi | classify | purge | telegram-chat-id | publish`.
- Import: `from birdy.publish import Publisher` (top-level; it imports cv2, which is available wherever the CLI runs).
- `build_parser`: `sub.add_parser("publish", help="build the gallery mirror and rsync it to the server once")`.
- `cmd_run`, after the notifier block:

```python
    publisher = None
    if config.publish.enabled:
        publisher = Publisher(live, store, config.storage.data_dir / "publish")
        publisher.start()
    pipeline = Pipeline(store, clf, live, notifier=notifier, publisher=publisher)
```

  pass `publisher=publisher` to `create_app(...)`; in the startup log add `publish=%s` with `config.publish.enabled`; in `finally`, after `pipeline.stop()` and before `notifier.stop()`:

```python
        if publisher is not None:
            publisher.stop()
```

- New command:

```python
def cmd_publish(config: Config) -> int:
    if not config.publish.enabled:
        print("publishing is disabled: set [publish] enabled = true (with host) in birdy.toml", file=sys.stderr)
        return 2
    store = Store(config.storage.data_dir)
    try:
        ok = Publisher(LiveConfig(config), store, config.storage.data_dir / "publish").run_once()
    finally:
        store.close()
    if ok:
        print(f"published to {config.publish.user}@{config.publish.host}:{config.publish.remote_dir}")
        return 0
    print(f"rsync failed (see log above); key: {config.publish.ssh_key}", file=sys.stderr)
    return 1
```

  and in `main`: `if args.command == "publish": return cmd_publish(config)`.

`deploy/install.sh` — before the `sudo cp deploy/birdy.service` line:

```bash
# SSH key the publisher uses to rsync the gallery mirror to the server (see CLAUDE.md "Gallery setup").
if [ ! -f ~/.ssh/birdy_publish ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/birdy_publish -C "birdy-publish@$(hostname)" >/dev/null
  echo "created ~/.ssh/birdy_publish; add this line to the gallery server's authorized_keys (see CLAUDE.md):"
  echo "command=\"/usr/bin/rrsync /srv/birdy/sites/home\",restrict $(cat ~/.ssh/birdy_publish.pub)"
fi
```

Create `deploy/gallery.sh` (chmod +x):

```bash
#!/usr/bin/env bash
# Push the fixed gallery pages (gallery/) to the server's /srv/birdy/. Never touches sites/ or sites.json.
# Usage: deploy/gallery.sh            (override with GALLERY_HOST=user@host GALLERY_ROOT=/srv/birdy)
set -euo pipefail
HOST="${GALLERY_HOST:-root@91.98.229.209}"
ROOT="${GALLERY_ROOT:-/srv/birdy}"
cd "$(dirname "$0")/.."
rsync -az --exclude sites/ --exclude sites.json gallery/ "$HOST:$ROOT/"
echo "gallery pages synced to $HOST:$ROOT/"
```

- [ ] **Step 4: Run the tests and pyflakes**

Run: `.venv/bin/pytest -q && .venv/bin/python -m pyflakes birdy tests`
Expected: all pass, pyflakes silent.

- [ ] **Step 5: Commit**

```bash
chmod +x deploy/gallery.sh
git add birdy/cli.py deploy/install.sh deploy/gallery.sh tests/test_cli.py
git commit -m "CLI: wire Publisher, add birdy publish; install.sh makes the publish key; deploy/gallery.sh"
```

---

### Task 9: Gallery pages (`gallery/`)

**Files:**
- Create: `gallery/index.html`, `gallery/site.html`, `gallery/species.html`, `gallery/app.js`, `gallery/style.css`, `gallery/sites.example.json`
- Test: `tests/test_gallery.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gallery.py`:

```python
"""The static gallery has no build step; check its files reference each other consistently."""

import json
import re
from pathlib import Path

GALLERY = Path(__file__).resolve().parents[1] / "gallery"
PAGES = ["index.html", "site.html", "species.html"]


def test_pages_exist():
    for name in PAGES + ["app.js", "style.css", "sites.example.json"]:
        assert (GALLERY / name).exists(), name


def test_pages_reference_existing_local_assets():
    for name in PAGES:
        html = (GALLERY / name).read_text()
        for ref in re.findall(r'(?:src|href)="([^"#?]+)', html):
            if "://" in ref:
                continue
            assert (GALLERY / ref).exists(), f"{name} -> {ref}"


def test_pages_name_a_renderer_that_exists():
    js = (GALLERY / "app.js").read_text()
    for name in PAGES:
        m = re.search(r'<body data-page="(\w+)"', (GALLERY / name).read_text())
        assert m, f"{name} has no data-page"
        assert f"async function {m.group(1)}(" in js, f"{name}: no renderer {m.group(1)}"


def test_app_only_fetches_gallery_data():
    js = (GALLERY / "app.js").read_text()
    args = re.findall(r"fetch\(([^,)]+)", js)
    assert args, "app.js never fetches"
    for a in args:
        assert "sites.json" in a or "sites/" in a, a


def test_pages_are_responsive_and_titled():
    for name in PAGES:
        html = (GALLERY / name).read_text()
        assert '<meta name="viewport"' in html and "<title>" in html


def test_sites_example_is_valid():
    sites = json.loads((GALLERY / "sites.example.json").read_text())
    assert sites and all({"id", "name", "place"} <= set(s) for s in sites)
    assert all(re.fullmatch(r"[a-z0-9-]{1,32}", s["id"]) for s in sites)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_gallery.py -q`
Expected: FAIL (`gallery/` does not exist).

- [ ] **Step 3: Create the files**

`gallery/sites.example.json`:

```json
[
  {"id": "home", "name": "Yassine", "place": "Sintra, PT"}
]
```

`gallery/style.css`:

```css
:root { color-scheme: light dark; --bg: #fafaf7; --fg: #222; --muted: #777; --card: #fff; --line: #ddd; --accent: #2a7; --bar: #4b9; }
@media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #eee; --muted: #999; --card: #222; --line: #333; --bar: #3a8; } }
* { box-sizing: border-box; }
body { margin: 0; padding: 16px; font: 15px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
a { color: inherit; }
header { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: baseline; margin-bottom: 16px; }
header h1 { font-size: 22px; margin: 0; }
header nav a { margin-right: 12px; color: var(--muted); }
h2 { font-size: 17px; margin: 24px 0 8px; }
.muted { color: var(--muted); }
.err { color: #c55; }
/* home: one column per site */
.columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; align-items: start; }
.col { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.col h2 { margin: 0 0 2px; font-size: 18px; }
.col .place { margin: 0 0 10px; }
.col .empty { color: var(--muted); margin: 8px 0; }
.col .past { opacity: .5; }
/* cards and grids */
.cards { display: grid; gap: 10px; }
.card { display: block; text-decoration: none; border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--card); }
.card img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; }
.card .cap { display: flex; justify-content: space-between; gap: 8px; padding: 6px 8px; font-size: 13px; }
.card .cap b { font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }
/* dashboard */
.nums { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; }
.num { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.num .v { font-size: 26px; font-weight: 600; line-height: 1.1; }
.num .l { color: var(--muted); font-size: 13px; }
.chart { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.chart svg { width: 100%; height: 120px; display: block; }
.chart rect { fill: var(--bar); }
.chart text { fill: var(--muted); font-size: 10px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); font-size: 14px; }
th { color: var(--muted); font-weight: 500; }
select { padding: 4px 6px; background: var(--card); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; }
/* lightbox */
dialog { border: 0; padding: 0; background: #000; max-width: 96vw; max-height: 96vh; }
dialog::backdrop { background: rgba(0,0,0,.8); }
dialog img { display: block; max-width: 96vw; max-height: 90vh; }
dialog p { margin: 0; padding: 8px 12px; color: #ddd; font-size: 13px; }
@media (max-width: 700px) { body { padding: 12px; } .columns { grid-template-columns: 1fr; } }
```

`gallery/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>birdy — today's visitors</title>
<link rel="stylesheet" href="style.css">
</head>
<body data-page="home">
<header>
  <h1><a href="index.html" style="text-decoration:none">birdy</a></h1>
  <nav><span class="muted">today's visitors at every feeder</span></nav>
</header>
<main id="main"><p class="muted">Loading…</p></main>
<dialog id="lightbox"><img alt=""><p></p></dialog>
<script src="app.js"></script>
</body>
</html>
```

`gallery/site.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>birdy — site</title>
<link rel="stylesheet" href="style.css">
</head>
<body data-page="site">
<header>
  <h1><a href="index.html" style="text-decoration:none">birdy</a></h1>
  <nav><a href="index.html">all feeders</a></nav>
</header>
<main id="main"><p class="muted">Loading…</p></main>
<dialog id="lightbox"><img alt=""><p></p></dialog>
<script src="app.js"></script>
</body>
</html>
```

`gallery/species.html`: identical to `site.html` except `<title>birdy — species</title>` and `<body data-page="species">`.

`gallery/app.js`:

```js
/* birdy gallery: renders index/site/species pages from sites.json + sites/<id>/manifest.json. No build step. */
const BIRDY = (() => {
  const main = () => document.getElementById("main");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const q = (name) => new URLSearchParams(location.search).get(name);
  const day = (iso) => iso.slice(0, 10);          // captured_at carries the site's offset: the local date is the prefix
  const hm = (iso) => iso.slice(11, 16);
  const pct = (c) => (c == null ? "" : Math.round(c * 100) + "%");
  const name = (p) => p.species || "Unidentified bird";
  const siteToday = (site) => new Date(Date.now() + (site.utc_offset_minutes || 0) * 60000).toISOString().slice(0, 10);
  const speciesHref = (s) => "species.html?name=" + encodeURIComponent(s);
  const siteHref = (id) => "site.html?id=" + encodeURIComponent(id);

  async function loadSites() {
    const r = await fetch("sites.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("sites.json: HTTP " + r.status);
    return r.json();
  }
  async function loadManifest(id) {
    try {
      const r = await fetch("sites/" + encodeURIComponent(id) + "/manifest.json", { cache: "no-cache" });
      return r.ok ? await r.json() : null;
    } catch { return null; }
  }
  async function loadAll() {
    const sites = await loadSites();
    const manifests = await Promise.all(sites.map((s) => loadManifest(s.id)));
    return sites.map((s, i) => ({ ...s, manifest: manifests[i] }));
  }

  // --- rendering helpers ------------------------------------------------------
  function photoCard(siteId, p, extra = "") {
    const base = "sites/" + encodeURIComponent(siteId) + "/";
    return `<a class="card" href="${base + p.photo}" data-caption="${esc(name(p))} · ${esc(day(p.captured_at))} ${esc(hm(p.captured_at))}">
      <img src="${base + p.thumb}" alt="${esc(name(p))}" loading="lazy">
      <div class="cap"><b>${esc(name(p))}</b><span class="muted">${esc(hm(p.captured_at))} ${extra}</span></div></a>`;
  }
  function lightbox() {
    const dlg = document.getElementById("lightbox");
    if (!dlg) return;
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a.card");
      if (!a) return;
      e.preventDefault();
      dlg.querySelector("img").src = a.getAttribute("href");
      dlg.querySelector("p").textContent = a.dataset.caption || "";
      dlg.showModal();
    });
    dlg.addEventListener("click", () => dlg.close());
  }
  function bars(values, labels, every) {
    const W = 300, H = 110, pad = 14, max = Math.max(1, ...values);
    const bw = (W - pad) / values.length;
    const rects = values.map((v, i) => {
      const h = Math.round((v / max) * (H - pad - 4));
      return `<rect x="${(pad + i * bw).toFixed(1)}" y="${H - pad - h}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${h}"><title>${labels[i]}: ${v}</title></rect>`;
    });
    const texts = labels.map((l, i) => (i % every ? "" : `<text x="${(pad + i * bw).toFixed(1)}" y="${H - 2}">${esc(l)}</text>`));
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><text x="0" y="10">${max}</text>${rects.join("")}${texts.join("")}</svg>`;
  }
  function groupByDay(photos, d) {
    const groups = new Map();
    for (const p of photos.filter((p) => day(p.captured_at) === d)) {
      const g = groups.get(name(p)) || { best: p, count: 0, latest: p.captured_at };
      g.count += 1;
      if ((p.confidence ?? 0) > (g.best.confidence ?? 0)) g.best = p;
      if (p.captured_at > g.latest) g.latest = p.captured_at;
      groups.set(name(p), g);
    }
    return [...groups.values()].sort((a, b) => (a.latest < b.latest ? 1 : -1));
  }
  function unavailable(site) {
    return `<section class="col"><h2>${esc(site.name)}</h2><p class="place muted">${esc(site.place || "")}</p><p class="empty">unavailable</p></section>`;
  }

  // --- pages ----------------------------------------------------------------------
  async function home() {
    const sites = await loadAll();
    main().innerHTML = `<div class="columns">${sites.map((s) => {
      const m = s.manifest;
      if (!m) return unavailable(s);
      const today = siteToday(m.site);
      let groups = groupByDay(m.photos, today), heading = "", past = "";
      if (!groups.length) {
        const last = m.photos.length ? day(m.photos[0].captured_at) : null;
        heading = `<p class="empty">No visitors yet today</p>`;
        if (last) { groups = groupByDay(m.photos, last); past = " past"; heading += `<p class="muted">last visit ${esc(last)}</p>`; }
      }
      const cards = groups.map((g) => photoCard(s.id, g.best, `· ${g.count} today`)).join("");
      return `<section class="col"><h2><a href="${siteHref(s.id)}">${esc(s.name)}</a></h2>
        <p class="place muted">${esc(s.place || "")} · <a href="${siteHref(s.id)}">dashboard</a></p>${heading}<div class="cards${past}">${cards}</div></section>`;
    }).join("")}</div>`;
    lightbox();
  }

  async function site() {
    const id = q("id");
    const sites = await loadSites();
    const s = sites.find((x) => x.id === id);
    if (!s) throw new Error("unknown site");
    const m = await loadManifest(id);
    if (!m) throw new Error("manifest unavailable");
    document.title = `birdy — ${s.name}`;
    const st = m.stats, pd = st.per_day, today = pd[pd.length - 1]?.birds ?? 0;
    const week = pd.slice(-7).reduce((a, d) => a + d.birds, 0);
    const nums = [[st.species_count, "species seen"], [today, "birds today"], [week, "last 7 days"], [st.total_birds, "birds in total"],
      [st.last_capture_at ? hm(st.last_capture_at) : "–", st.last_capture_at ? "last visit " + day(st.last_capture_at) : "no visits yet"]];
    const top = st.species.map((x) => `<tr><td><a href="${speciesHref(x.name)}">${esc(x.name)}</a></td><td>${x.count}</td><td>${esc(x.first)}</td><td>${esc(day(x.last))}</td></tr>`).join("");
    const options = [...new Set(m.photos.map(name))].sort().map((n) => `<option>${esc(n)}</option>`).join("");
    main().innerHTML = `
      <h2 style="margin-top:0">${esc(s.name)} <span class="muted" style="font-weight:normal">${esc(s.place || "")}${m.site.since ? " · watching since " + esc(m.site.since) : ""}</span></h2>
      <div class="nums">${nums.map(([v, l]) => `<div class="num"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("")}</div>
      <div class="charts" style="margin-top:12px">
        <div class="chart"><div class="muted">birds per day, last 30 days</div>${bars(pd.map((d) => d.birds), pd.map((d) => d.day.slice(5)), 5)}</div>
        <div class="chart"><div class="muted">visits by hour of day</div>${bars(st.per_hour, st.per_hour.map((_, h) => String(h)), 3)}</div>
      </div>
      <h2>species</h2>
      <table><tr><th>species</th><th>visits</th><th>first</th><th>last</th></tr>${top || "<tr><td colspan=4 class=muted>none yet</td></tr>"}</table>
      <h2>published photos <select id="flt"><option value="">all species</option>${options}</select></h2>
      <div class="grid" id="grid"></div>`;
    const grid = document.getElementById("grid");
    const draw = (f) => { grid.innerHTML = m.photos.filter((p) => !f || name(p) === f).map((p) => photoCard(id, p, day(p.captured_at))).join("") || "<p class=muted>nothing published yet</p>"; };
    document.getElementById("flt").addEventListener("change", (e) => draw(e.target.value));
    draw("");
    lightbox();
  }

  async function species() {
    const sp = q("name");
    if (!sp) throw new Error("no species given");
    document.title = `birdy — ${sp}`;
    const sites = await loadAll();
    const rows = [], cards = [];
    for (const s of sites) {
      if (!s.manifest) continue;
      const st = s.manifest.stats.species.find((x) => x.name === sp);
      if (st) rows.push(`<tr><td><a href="${siteHref(s.id)}">${esc(s.name)}</a></td><td>${st.count}</td><td>${esc(st.first)}</td><td>${esc(day(st.last))}</td></tr>`);
      for (const p of s.manifest.photos.filter((p) => p.species === sp)) cards.push({ s, p });
    }
    cards.sort((a, b) => (a.p.captured_at < b.p.captured_at ? 1 : -1));
    main().innerHTML = `<h2 style="margin-top:0">${esc(sp)}</h2>
      <table><tr><th>feeder</th><th>visits</th><th>first</th><th>last</th></tr>${rows.join("") || "<tr><td colspan=4 class=muted>not seen yet</td></tr>"}</table>
      <h2>photos</h2><div class="grid">${cards.map(({ s, p }) => photoCard(s.id, p, `${esc(day(p.captured_at))} · ${esc(s.name)}`)).join("") || "<p class=muted>none published</p>"}</div>`;
    lightbox();
  }

  return { home, site, species };
})();

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  const run = BIRDY[page];
  if (run) run().catch((e) => { document.getElementById("main").innerHTML = `<p class="err">${e.message}</p>`; });
});
```

- [ ] **Step 4: Run the tests, then look at the pages in a browser**

Run: `.venv/bin/pytest tests/test_gallery.py -q` → 6 passed.

Manual check with sample data (uses the test fixtures as photos):

```bash
cd /Users/yassine/www/pi-birdy && rm -rf /tmp/gal && mkdir -p /tmp/gal/sites/home/photos /tmp/gal/sites/home/thumbs && cp gallery/* /tmp/gal/ && cp gallery/sites.example.json /tmp/gal/sites.json
cp tests/fixtures/bird.jpg /tmp/gal/sites/home/photos/a.jpg && cp tests/fixtures/blue_tit.jpg /tmp/gal/sites/home/photos/b.jpg && cp /tmp/gal/sites/home/photos/*.jpg /tmp/gal/sites/home/thumbs/
.venv/bin/python - <<'EOF'
import json, datetime as dt
now = dt.datetime.now().astimezone()
d = now.date().isoformat()
m = {"version": 1, "generated_at": now.isoformat(timespec="seconds"),
     "site": {"id": "home", "name": "home", "lat": 38.8, "lon": -9.39, "since": "2026-10-03", "utc_offset_minutes": int(now.utcoffset().total_seconds()//60)},
     "photos": [{"id": 2, "captured_at": f"{d}T10:12:00{now.isoformat()[-6:]}", "species": "Eurasian Blue Tit", "confidence": 0.81, "photo": "photos/b.jpg", "thumb": "thumbs/b.jpg", "state": "auto"},
                {"id": 1, "captured_at": f"{d}T09:01:00{now.isoformat()[-6:]}", "species": "Common Hoopoe", "confidence": 0.93, "photo": "photos/a.jpg", "thumb": "thumbs/a.jpg", "state": "auto"}],
     "stats": {"total_birds": 412, "species_count": 2, "first_capture": "2026-10-03", "last_capture_at": f"{d}T10:12:00{now.isoformat()[-6:]}",
               "species": [{"name": "Common Hoopoe", "count": 300, "first": "2026-10-03", "last": f"{d}T09:01:00+01:00"}, {"name": "Eurasian Blue Tit", "count": 112, "first": "2026-10-03", "last": f"{d}T10:12:00+01:00"}],
               "per_day": [{"day": (now.date()-dt.timedelta(days=29-i)).isoformat(), "birds": (i*7)%40} for i in range(30)],
               "per_hour": [0,0,0,0,0,0,2,14,31,40,33,20,12,15,22,30,25,10,3,0,0,0,0,0]}}
json.dump(m, open("/tmp/gal/sites/home/manifest.json", "w"))
EOF
(cd /tmp/gal && python3 -m http.server 8765 >/dev/null 2>&1 &) ; echo "open http://localhost:8765/  http://localhost:8765/site.html?id=home  http://localhost:8765/species.html?name=Common%20Hoopoe"
```

Open the three URLs; check: cards render with thumbs, lightbox opens, the dashboard shows 5 number cards, two bar charts, the species table and the filterable grid; narrow the window below 700 px and confirm a single column. Then `pkill -f "http.server 8765"`.

- [ ] **Step 5: Commit**

```bash
git add gallery tests/test_gallery.py
git commit -m "Gallery: static pages (home columns, site dashboard, species)"
```

---

### Task 10: Docs, server setup, Pi deployment and verification

> **Public URL:** `https://birdy.akibeo.com`. The user controls akibeo.com's DNS at Combell and adds the A record `birdy.akibeo.com → 91.98.229.209` there. The site block below goes into the cartaveo Caddyfile (that Caddy owns ports 80/443); everything else about the gallery is separate from cartaveo.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-10-04-publishing-design.md` (implementation notes)

- [ ] **Step 1: CLAUDE.md**

- Intro sentence: add `` `docs/superpowers/specs/2026-10-04-publishing-design.md` for the public gallery (3) `` to the list of specs, and change "Sub-projects 3–5 (publishing, multi-site, e-ink) are out of scope" to "Sub-projects 4–5 (multi-site onboarding, e-ink) are out of scope".
- Commands block: add `.venv/bin/birdy publish           # build the gallery mirror and rsync it to the server once` and, under the Mac section, `deploy/gallery.sh               # push gallery/ pages to the server (never sites/ or sites.json)`.
- New section after "Telegram bot setup":

````markdown
Gallery setup (one-time; the server runs no code, Caddy serves `/srv/birdy`):

1. On the server (`ssh root@91.98.229.209`):
   ```bash
   adduser --system --group --home /srv/birdy/sites --shell /usr/sbin/nologin birdy
   mkdir -p /srv/birdy/sites/home /srv/birdy/sites/.ssh && chmod 700 /srv/birdy/sites/.ssh
   cp ~/pi-birdy-gallery/sites.example.json /srv/birdy/sites.json   # or write it by hand: [{"id":"home","name":"Yassine","place":"Sintra, PT"}]
   chown -R birdy:birdy /srv/birdy/sites
   ```
   `authorized_keys` for the `birdy` user is `/srv/birdy/sites/.ssh/authorized_keys` (mode 600, owner birdy). One line per Pi, printed by `install.sh`:
   `command="/usr/bin/rrsync /srv/birdy/sites/<site_id>",restrict ssh-ed25519 AAAA… birdy-publish@birdy`
   `rrsync` makes every path the Pi sends relative to that folder, so the key can't reach other sites.
2. Caddy (cartaveo stack): add to the Caddyfile
   ```
   birdy.akibeo.com {
       root * /srv/birdy
       file_server
       encode gzip
       header /sites/* Cache-Control "public, max-age=300"
       header /*.json Cache-Control "no-cache"
   }
   ```
   and `- /srv/birdy:/srv/birdy:ro` under the `web` service's `volumes`; `docker compose up -d web`. DNS: `birdy.akibeo.com A 91.98.229.209`.
3. Mac: `deploy/gallery.sh` pushes the pages. Pi: `[publish] enabled = true`, `host = "91.98.229.209"` in `birdy.toml`, `sudo systemctl restart birdy`, then `.venv/bin/birdy publish` to push immediately (or wait ≤ 15 min).
````

- Architecture diagram: add `publisher (mirror + rsync) ◀── pipeline / web publish buttons` and "seven with publishing enabled".
- Architecture bullets, after the notify bullet:

```markdown
- `birdy/publish.py` — `rank_day` is pure: per species and local day, `forced`/`hidden` win, then the best `publish.per_species_per_day` photos at or above `publish.min_confidence` get `auto`. `Publisher.evaluate(capture)` re-ranks that species/day whenever the pipeline stores a species (live **and** backfill — historical photos may be published, unlike alerts); `set_manual(id, 'forced'|'hidden'|None)` is the review page's Publish/Hide/Auto. Its thread keeps `data/publish/` (web copies ≤ 1600 px via OpenCV, thumbs, `manifest.json`) in sync with the DB and `rsync --delete`s it to `user@host:remote_dir` (debounced 2 min, at least every 15 min, first run at start). Unreadable originals are set `hidden`. The gallery pages in `gallery/` render in the browser from `sites.json` + each site's manifest; nothing runs on the server.
```

- Gotchas:

```markdown
- `publish.enabled/host/user/remote_dir/site_id/ssh_key` are restart-only; `min_confidence`/`per_species_per_day` are live but only affect days ranked from then on. `published` states: `NULL`, `auto`, `forced`, `hidden`; the published set is `auto`+`forced`. `web_path` is the mirror copy's relative path (same as `path`). The manifest's `captured_at` is local time with offset and `site.utc_offset_minutes` lets the page compute the site's "today".
- `FILTERS["published"]` exists; the nav shows it only when a publisher is configured.
```

- [ ] **Step 2: Spec implementation notes**

Append to the spec before "## Out of scope":

```markdown
## Implementation notes (as merged)

- The manifest's `site` block carries `utc_offset_minutes` (offset at generation time) instead of a zone name; photo entries have no `w`/`h` (the page uses `object-fit`). `captured_at` strings carry the site's offset, so a photo's local date is its first 10 characters.
- `remote_dir` replaces `root`: the Pi's rsync target is `user@host:remote_dir` and the server-side `rrsync` restriction maps it onto `sites/<site_id>/`.
- The publisher's first run happens right after start (reconcile), then on change (debounced 120 s) and at least every 900 s.
- `_num(..., int)` in `config.py` now rejects non-integral floats (`2.5`) instead of truncating.
```

- [ ] **Step 3: Full suite, pyflakes, commit**

```bash
.venv/bin/pytest -q && .venv/bin/python -m pyflakes birdy tests
git add CLAUDE.md docs/superpowers/specs/2026-10-04-publishing-design.md
git commit -m "Docs: publishing / gallery setup"
```

- [ ] **Step 4: Server setup (the user applies the production-stack edits; Claude does the rest)**

Claude, from the Mac (read-only checks are fine; creating the `birdy` user and `/srv/birdy` is the agreed server-side work):

```bash
ssh root@91.98.229.209 'id birdy 2>/dev/null || adduser --system --group --home /srv/birdy/sites --shell /usr/sbin/nologin birdy; mkdir -p /srv/birdy/sites/home /srv/birdy/sites/.ssh; chmod 700 /srv/birdy/sites/.ssh; chown -R birdy:birdy /srv/birdy/sites; which rrsync || ls /usr/bin/rrsync'
deploy/gallery.sh
ssh root@91.98.229.209 '[ -f /srv/birdy/sites.json ] || cp /srv/birdy/sites.example.json /srv/birdy/sites.json; ls -la /srv/birdy'
```

Then hand the user the two cartaveo edits (Caddy block + bind mount, verbatim from CLAUDE.md) and the DNS record, and wait for them to apply and reload. Do not edit the cartaveo files.

- [ ] **Step 5: Pi deployment and verification**

```bash
deploy/deploy.sh
ssh -t yassine@birdy.local 'cd ~/pi-birdy && deploy/install.sh'       # generates ~/.ssh/birdy_publish and prints the authorized_keys line
```

Append that printed line to `/srv/birdy/sites/.ssh/authorized_keys` on the server (`chmod 600`, owner `birdy`). Then on the Pi, add to `birdy.toml`:

```toml
[publish]
enabled = true
host = "91.98.229.209"
```

(`user`, `remote_dir`, `site_id`, `ssh_key` keep their defaults), `sudo systemctl restart birdy`, and:

```bash
ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/birdy publish'                       # expect "published to birdy@91.98.229.209:/"
ssh root@91.98.229.209 'ls /srv/birdy/sites/home; python3 -c "import json;m=json.load(open(\"/srv/birdy/sites/home/manifest.json\"));print(len(m[\"photos\"]), m[\"stats\"][\"species_count\"])"'
curl -sI https://birdy.akibeo.com/ | head -1                                             # HTTP/2 200 once DNS + Caddy are in place
ssh yassine@birdy.local 'journalctl -u birdy --since "5 min ago" --no-pager -o cat | grep -E "publish|starting"'
```

Then in the browser: open https://birdy.akibeo.com (column with today's visitors), the dashboard, a species page; on http://birdy.local:8000 hide one published photo and confirm it disappears from the gallery within ~2 minutes (journal: `publish: N photos synced`).

- [ ] **Step 6: Finish**

Final whole-branch review (spec coverage, cross-module consistency, secret hygiene: no server credentials in the repo, the Pi's private key never leaves the Pi), then `superpowers:finishing-a-development-branch`.
