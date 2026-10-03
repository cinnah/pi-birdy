# Telegram Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send a Telegram photo message when a watchlisted species or a never-before-seen species is captured, with a per-species cooldown, configurable on the settings page and verifiable with a test button.

**Architecture:** `birdy/notify.py` holds a pure `AlertPolicy` (watchlist / first-seen / cooldown), a `TelegramTransport` protocol with an `urllib` implementation, and a `TelegramNotifier` (bounded queue + one worker thread with retries). The pipeline calls `notifier.consider(...)` after storing a live classification; backfill never does. Config gains `[telegram]`; the settings page gains an Alerts group and a test-alert button; the CLI gains `telegram-chat-id`.

**Tech Stack:** existing stack; `urllib` only (no new dependency).

**Spec:** `docs/superpowers/specs/2026-10-03-telegram-alerts-design.md`
**Branch:** `telegram` (created, spec committed).
**Model dispatch hint:** `[opus-ok]` = mechanical.

---

## File structure

| File | Responsibility |
|---|---|
| `birdy/notify.py` (new) | `TelegramError`, `TelegramSettings`, `Alert`, `AlertPolicy`, `caption_for`, `TelegramTransport`, `HttpTelegramTransport`, `TelegramNotifier`, `fetch_chat_ids` |
| `birdy/config.py` (modify) | `TelegramConfig` + `[telegram]` parsing/validation |
| `birdy/settings.py` (modify) | three live fields (`watchlist`, `alert_new_species`, `cooldown_minutes`) |
| `birdy/watcher.py` (modify) | restart-only keys |
| `birdy/pipeline.py` (modify) | `process(capture_id, path, captured_at)` + notifier hook |
| `birdy/web/app.py`, `templates/settings.html` (modify) | Alerts group, read-only entries, `POST /settings/test-alert` |
| `birdy/cli.py` (modify) | wiring; `telegram-chat-id` |
| `birdy.toml.example`, `CLAUDE.md` (modify) | |
| `tests/test_notify.py` (new); `test_config.py`, `test_settings.py`, `test_watcher.py`, `test_pipeline.py`, `test_web.py`, `test_cli.py` (modify) | |

---

### Task 1: `[telegram]` config, settings fields, watcher keys `[opus-ok]`

**Files:**
- Modify: `birdy/config.py`, `birdy.toml.example`, `birdy/settings.py`, `birdy/watcher.py`, `tests/test_config.py`, `tests/test_settings.py`, `tests/test_watcher.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:
```python
def test_telegram_defaults(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    t = cfg.telegram
    assert t.enabled is False and t.bot_token == "" and t.chat_id == ""
    assert t.base_url == "http://birdy.local:8000"
    assert t.watchlist == () and t.alert_new_species is True and t.cooldown_minutes == 30


def test_telegram_watchlist_trimmed_and_empty_dropped(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL + '\n[telegram]\nwatchlist = [" Hoopoe ", "", "Blue Tit"]\n'))
    assert cfg.telegram.watchlist == ("Hoopoe", "Blue Tit")


def test_telegram_watchlist_must_be_strings(tmp_path):
    with pytest.raises(ConfigError, match="telegram.watchlist"):
        load_config(write(tmp_path, MINIMAL + "\n[telegram]\nwatchlist = [1, 2]\n"))


@pytest.mark.parametrize("missing", ["bot_token", "chat_id"])
def test_telegram_enabled_requires_token_and_chat(tmp_path, missing):
    other = "chat_id" if missing == "bot_token" else "bot_token"
    with pytest.raises(ConfigError, match=f"telegram.{missing}"):
        load_config(write(tmp_path, MINIMAL + f'\n[telegram]\nenabled = true\n{other} = "x"\n'))


def test_telegram_cooldown_non_negative(tmp_path):
    with pytest.raises(ConfigError, match="telegram.cooldown_minutes"):
        load_config(write(tmp_path, MINIMAL + "\n[telegram]\ncooldown_minutes = -1\n"))


def test_telegram_enabled_must_be_bool(tmp_path):
    with pytest.raises(ConfigError, match="telegram.enabled"):
        load_config(write(tmp_path, MINIMAL + '\n[telegram]\nenabled = "yes"\n'))
```

In `tests/test_settings.py`: `default_patch` gains `watchlist=config.telegram.watchlist, alert_new_species=config.telegram.alert_new_species, cooldown_minutes=config.telegram.cooldown_minutes`; `FORM` gains `"watchlist": "", "alert_new_species": "on", "cooldown_minutes": "30"`; `FULL_TOML` gains:
```toml
[telegram]
enabled = true
bot_token = "123:ABC"
chat_id = "42"
base_url = "http://example.test"
watchlist = ["Hoopoe"]
alert_new_species = false
cooldown_minutes = 5
```
Add:
```python
def test_telegram_fields_roundtrip_and_token_preserved(config_path, config):
    config_path.write_text(config_path.read_text() + '\n[telegram]\nbot_token = "SECRET"\nchat_id = "7"\n')
    cfg = load_config(config_path)
    new = save_settings(config_path, cfg, default_patch(cfg, watchlist=("Hoopoe", "Blue Tit"), alert_new_species=False, cooldown_minutes=10))
    assert new.telegram.watchlist == ("Hoopoe", "Blue Tit")
    assert new.telegram.alert_new_species is False and new.telegram.cooldown_minutes == 10
    assert new.telegram.bot_token == "SECRET" and new.telegram.chat_id == "7"  # restart-only keys survive a save


def test_parse_form_watchlist_textarea_and_checkbox():
    p = parse_settings_form({**FORM, "watchlist": "Hoopoe\n\n  Blue Tit  \n", "alert_new_species": "on"})
    assert p.watchlist == ("Hoopoe", "Blue Tit") and p.alert_new_species is True
    form = dict(FORM)
    del form["alert_new_species"]  # unchecked checkboxes are absent from the POST
    assert parse_settings_form(form).alert_new_species is False


def test_parse_form_cooldown_minutes_int():
    with pytest.raises(ConfigError, match="telegram.cooldown_minutes"):
        parse_settings_form({**FORM, "cooldown_minutes": "x"})
```
Append to `tests/test_watcher.py`:
```python
def test_telegram_token_change_warns(config_path, live, caplog):
    w = ConfigWatcher(config_path, live)
    config_path.write_text(config_path.read_text() + '\n[telegram]\nbot_token = "new"\n')
    bump_mtime(config_path)
    with caplog.at_level(logging.WARNING):
        assert w.check_once() is True
    assert "restart" in caplog.text and live.get().telegram.bot_token == "new"
```

- [ ] **Step 2: Run to verify failure**

`.venv/bin/pytest tests/test_config.py tests/test_settings.py tests/test_watcher.py -q` → `AttributeError: 'Config' object has no attribute 'telegram'` etc.

- [ ] **Step 3: `birdy/config.py`**

Add after `WebConfig`:
```python
@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool              # restart-only
    bot_token: str             # restart-only, secret
    chat_id: str               # restart-only
    base_url: str              # restart-only; link in the alert caption
    watchlist: tuple[str, ...] # live
    alert_new_species: bool    # live
    cooldown_minutes: int      # live
```
Add `telegram: TelegramConfig` as the last field of `Config`. Helpers (next to `_num`):
```python
def _bool(section: dict[str, Any], name: str, key: str, default: bool) -> bool:
    v = section.get(key, default)
    if not isinstance(v, bool):
        raise ConfigError(f"{name}.{key} must be true or false")
    return v


def _str(section: dict[str, Any], name: str, key: str, default: str) -> str:
    v = section.get(key, default)
    if not isinstance(v, str):
        raise ConfigError(f"{name}.{key} must be a string")
    return v


def _str_list(section: dict[str, Any], name: str, key: str) -> tuple[str, ...]:
    v = section.get(key, [])
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise ConfigError(f"{name}.{key} must be a list of strings")
    return tuple(s.strip() for s in v if s.strip())
```
In `load_config`, after the web block:
```python
    tg = raw.get("telegram", {})
    tg_enabled = _bool(tg, "telegram", "enabled", False)
    tg_token = _str(tg, "telegram", "bot_token", "")
    tg_chat = _str(tg, "telegram", "chat_id", "")
    tg_base = _str(tg, "telegram", "base_url", "http://birdy.local:8000").rstrip("/")
    tg_watch = _str_list(tg, "telegram", "watchlist")
    tg_new = _bool(tg, "telegram", "alert_new_species", True)
    tg_cooldown = _num(tg, "telegram", "cooldown_minutes", 30, int)
    _check(tg_cooldown >= 0, "telegram.cooldown_minutes must be >= 0")
    if tg_enabled:
        _check(bool(tg_token), "telegram.bot_token is required when telegram.enabled = true")
        _check(bool(tg_chat), "telegram.chat_id is required when telegram.enabled = true")
```
and in the `Config(...)` construction, last:
```python
        telegram=TelegramConfig(
            enabled=tg_enabled, bot_token=tg_token, chat_id=tg_chat, base_url=tg_base,
            watchlist=tg_watch, alert_new_species=tg_new, cooldown_minutes=tg_cooldown,
        ),
```
`birdy.toml.example`, at the end:
```toml
[telegram]
enabled = false            # restart-only; set true once bot_token and chat_id are filled in
bot_token = ""             # restart-only, SECRET — from @BotFather; this file is gitignored
chat_id = ""               # restart-only — `birdy telegram-chat-id` prints it after you message the bot
base_url = "http://birdy.local:8000"   # link in the alert caption
watchlist = []             # species common names (as on /species), e.g. ["Eurasian Hoopoe"]
alert_new_species = true   # one alert the first time any species is ever seen
cooldown_minutes = 30      # at most one alert per species per this many minutes
```

- [ ] **Step 4: `birdy/settings.py`**

`FIELD_KEYS`: add `"watchlist": "telegram.watchlist", "alert_new_species": "telegram.alert_new_species", "cooldown_minutes": "telegram.cooldown_minutes"`. `SettingsPatch`: add `watchlist: tuple[str, ...]`, `alert_new_species: bool`, `cooldown_minutes: int` (after `non_bird_days`). `config_to_dict`: add
```python
        "telegram": {
            "enabled": config.telegram.enabled,
            "bot_token": config.telegram.bot_token,
            "chat_id": config.telegram.chat_id,
            "base_url": config.telegram.base_url,
            "watchlist": list(config.telegram.watchlist),
            "alert_new_species": config.telegram.alert_new_species,
            "cooldown_minutes": config.telegram.cooldown_minutes,
        },
```
`apply_patch`: `data["telegram"].update(watchlist=list(patch.watchlist), alert_new_species=patch.alert_new_species, cooldown_minutes=patch.cooldown_minutes)`. Add:
```python
def _watchlist(form: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(s.strip() for s in form.get("watchlist", "").splitlines() if s.strip())
```
and in `parse_settings_form`: `watchlist=_watchlist(form), alert_new_species=form.get("alert_new_species") is not None, cooldown_minutes=_num(form, "cooldown_minutes", int)`.

- [ ] **Step 5: `birdy/watcher.py`** `_restart_only`: append `cfg.telegram.enabled, cfg.telegram.bot_token, cfg.telegram.chat_id, cfg.telegram.base_url` to the tuple.

- [ ] **Step 6: Run tests + commit**

`.venv/bin/pytest tests/test_config.py tests/test_settings.py tests/test_watcher.py -q` → all pass. `.venv/bin/pytest -q`: `tests/test_web.py` settings POSTs will now 422 (missing `cooldown_minutes`) — expected until Task 4; report the count but don't fix web here.
```bash
git add birdy/config.py birdy.toml.example birdy/settings.py birdy/watcher.py tests/test_config.py tests/test_settings.py tests/test_watcher.py
git commit -m "Add [telegram] config, live alert settings, watcher keys"
```

---

### Task 2: `birdy/notify.py` — policy, transport, notifier

**Files:**
- Create: `birdy/notify.py`, `tests/test_notify.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_notify.py`:
```python
import dataclasses
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import Prediction
from birdy.notify import (
    Alert, AlertPolicy, HttpTelegramTransport, TelegramError, TelegramNotifier, TelegramSettings,
    caption_for, fetch_chat_ids,
)

NOW = datetime(2026, 10, 3, 14, 32, tzinfo=timezone.utc)
SET = TelegramSettings(watchlist=("Eurasian Hoopoe", " blue tit "), alert_new_species=True, cooldown_minutes=30)


# --- policy ---------------------------------------------------------------

def test_watchlist_hit_case_insensitive():
    p = AlertPolicy()
    assert p.decide("Blue Tit", False, SET, NOW) == "watchlist"
    assert p.decide("eurasian hoopoe", False, SET, NOW) == "watchlist"


def test_not_watchlisted_not_new_is_none():
    assert AlertPolicy().decide("Robin", False, SET, NOW) is None


def test_new_species_alerts_once_and_bypasses_cooldown():
    p = AlertPolicy()
    assert p.decide("Blue Tit", True, SET, NOW) == "new"
    assert p.decide("Blue Tit", False, SET, NOW + timedelta(minutes=1)) is None  # cooldown from the "new" alert
    p2 = AlertPolicy()
    assert p2.decide("Blue Tit", False, SET, NOW) == "watchlist"
    assert p2.decide("Blue Tit", True, SET, NOW + timedelta(minutes=1)) == "new"  # new ignores cooldown


def test_new_species_disabled():
    s = dataclasses.replace(SET, alert_new_species=False)
    assert AlertPolicy().decide("Robin", True, s, NOW) is None
    assert AlertPolicy().decide("Blue Tit", True, s, NOW) == "watchlist"


def test_cooldown_suppresses_then_reallows():
    p = AlertPolicy()
    assert p.decide("Blue Tit", False, SET, NOW) == "watchlist"
    assert p.decide("Blue Tit", False, SET, NOW + timedelta(minutes=29)) is None
    assert p.decide("Blue Tit", False, SET, NOW + timedelta(minutes=30)) == "watchlist"


def test_cooldown_zero_never_suppresses():
    s = dataclasses.replace(SET, cooldown_minutes=0)
    p = AlertPolicy()
    assert p.decide("Blue Tit", False, s, NOW) == "watchlist"
    assert p.decide("Blue Tit", False, s, NOW) == "watchlist"


def test_cooldown_is_per_species():
    p = AlertPolicy()
    assert p.decide("Blue Tit", False, SET, NOW) == "watchlist"
    assert p.decide("Eurasian Hoopoe", False, SET, NOW) == "watchlist"


# --- caption --------------------------------------------------------------

def test_caption_format():
    a = Alert(1, Path("x.jpg"), "Eurasian Blue Tit", 0.823, NOW, "new")
    text = caption_for(a, "http://birdy.local:8000")
    first, link = text.split("\n")
    assert first.startswith("🐦 Eurasian Blue Tit (82%) · ")
    assert first.endswith(" · new species!")
    assert link == "http://birdy.local:8000/?species=Eurasian%20Blue%20Tit"
    assert "new species" not in caption_for(dataclasses.replace(a, reason="watchlist"), "http://x")


# --- transport ------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_transport_send_photo_multipart(tmp_path):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"\xff\xd8JPEGDATA")
    seen = {}

    def urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["ctype"] = req.get_header("Content-type")
        seen["body"] = req.data
        seen["timeout"] = timeout
        return FakeResponse({"ok": True, "result": {}})

    t = HttpTelegramTransport("TOKEN123", urlopen=urlopen)
    t.send_photo("42", photo, "hello")
    assert seen["url"].endswith("/botTOKEN123/sendPhoto") and seen["timeout"] == 20.0
    assert seen["ctype"].startswith("multipart/form-data; boundary=")
    body = seen["body"]
    assert b'name="chat_id"\r\n\r\n42\r\n' in body
    assert b'name="caption"\r\n\r\nhello\r\n' in body
    assert b'name="photo"; filename="p.jpg"' in body and b"JPEGDATA" in body


def test_http_transport_send_text_and_error_hides_token():
    def urlopen(req, timeout):
        return FakeResponse({"ok": False, "description": "Unauthorized"})

    t = HttpTelegramTransport("TOKEN123", urlopen=urlopen)
    with pytest.raises(TelegramError) as ei:
        t.send_text("42", "hi")
    assert "Unauthorized" in str(ei.value) and "TOKEN123" not in str(ei.value)


def test_http_transport_missing_photo_raises_file_not_found(tmp_path):
    t = HttpTelegramTransport("T", urlopen=lambda req, timeout: FakeResponse({"ok": True}))
    with pytest.raises(FileNotFoundError):
        t.send_photo("42", tmp_path / "missing.jpg", "x")


def test_fetch_chat_ids_parses_updates():
    payload = {"ok": True, "result": [
        {"message": {"chat": {"id": 11, "first_name": "Yazz"}, "text": "/start"}},
        {"message": {"chat": {"id": 11, "first_name": "Yazz"}, "text": "test"}},
        {"edited_message": {"chat": {"id": 99}}},
    ]}
    chats = fetch_chat_ids("T", urlopen=lambda req, timeout: FakeResponse(payload))
    assert chats == [(11, "Yazz", "test")]


# --- notifier -------------------------------------------------------------

class FakeTransport:
    def __init__(self, fail_times: int = 0, missing_photo: bool = False):
        self.fail_times = fail_times
        self.missing_photo = missing_photo
        self.photos: list[tuple[str, Path, str]] = []
        self.texts: list[tuple[str, str]] = []

    def send_photo(self, chat_id, photo, caption):
        if self.fail_times:
            self.fail_times -= 1
            raise TelegramError("Bad Gateway", 502)
        if self.missing_photo:
            raise FileNotFoundError(photo)
        self.photos.append((chat_id, photo, caption))

    def send_text(self, chat_id, text):
        self.texts.append((chat_id, text))


def enabled_config(config, **tg):
    base = dict(enabled=True, bot_token="T", chat_id="42", watchlist=("Blue Tit",), alert_new_species=True, cooldown_minutes=30)
    base.update(tg)
    return dataclasses.replace(config, telegram=dataclasses.replace(config.telegram, **base))


@pytest.fixture
def store(tmp_path):
    from birdy.store import Store

    return Store(tmp_path / "data")


def add_bird(store, species, when=NOW):
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, True, 0.9, species, 0.8)
    return cid, abs_path


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_consider_ignores_non_birds_unknown_species_and_disabled(live, config, store):
    n = TelegramNotifier(live, store, FakeTransport())
    pred = Prediction(True, 0.9, "Blue Tit", 0.8)
    assert n.consider(1, Path("x"), pred, NOW) is None  # telegram disabled in the default config
    live.replace(enabled_config(config))
    assert n.consider(1, Path("x"), Prediction(False, 0.1, "Blue Tit", 0.8), NOW) is None
    assert n.consider(1, Path("x"), Prediction(True, 0.9, None, None), NOW) is None


def test_consider_queues_and_worker_sends_photo(live, config, store):
    live.replace(enabled_config(config))
    cid, path = add_bird(store, "Blue Tit")
    t = FakeTransport()
    n = TelegramNotifier(live, store, t)
    n.start()
    try:
        assert n.consider(cid, path, Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW) == "new"  # first ever Blue Tit
        assert wait_until(lambda: t.photos)
        chat, photo, caption = t.photos[0]
        assert chat == "42" and photo == path
        assert caption.startswith("🐦 Blue Tit (80%)") and "new species!" in caption
        assert "/?species=Blue%20Tit" in caption
    finally:
        n.stop()


def test_first_seen_uses_store_count(live, config, store):
    live.replace(enabled_config(config, watchlist=()))
    cid1, p1 = add_bird(store, "Robin")
    n = TelegramNotifier(live, store, FakeTransport())
    assert n.consider(cid1, p1, Prediction(True, 0.9, "Robin", 0.8), NOW, now=NOW) == "new"
    cid2, p2 = add_bird(store, "Robin", NOW + timedelta(minutes=1))
    assert n.consider(cid2, p2, Prediction(True, 0.9, "Robin", 0.8), NOW, now=NOW + timedelta(hours=2)) is None


def test_missing_photo_falls_back_to_text(live, config, store):
    live.replace(enabled_config(config))
    t = FakeTransport(missing_photo=True)
    n = TelegramNotifier(live, store, t)
    n.start()
    try:
        n.consider(1, Path("gone.jpg"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW)
        assert wait_until(lambda: t.texts)
        assert t.texts[0][1].startswith("🐦 Blue Tit")
    finally:
        n.stop()


def test_retries_then_drops(live, config, store, caplog):
    live.replace(enabled_config(config))
    sleeps = []
    t = FakeTransport(fail_times=4)
    n = TelegramNotifier(live, store, t, sleep=sleeps.append)
    n.start()
    try:
        with caplog.at_level(logging.WARNING):
            n.consider(1, Path("x.jpg"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW)
            assert wait_until(lambda: "alert dropped" in caplog.text)
        assert sleeps == [5.0, 15.0, 45.0]
        assert t.photos == []
    finally:
        n.stop()


def test_retry_succeeds_on_second_attempt(live, config, store):
    live.replace(enabled_config(config))
    sleeps = []
    t = FakeTransport(fail_times=1)
    n = TelegramNotifier(live, store, t, sleep=sleeps.append)
    n.start()
    try:
        n.consider(1, Path("x.jpg"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW)
        assert wait_until(lambda: t.photos)
        assert sleeps == [5.0]
    finally:
        n.stop()


def test_queue_full_drops_with_warning(live, config, store, caplog):
    live.replace(enabled_config(config, cooldown_minutes=0))
    n = TelegramNotifier(live, store, FakeTransport(), maxsize=1)  # not started: nothing drains
    with caplog.at_level(logging.WARNING):
        assert n.consider(1, Path("a"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW) == "watchlist"
        assert n.consider(2, Path("b"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW) is None
    assert "queue full" in caplog.text


def test_worker_survives_transport_crash(live, config, store):
    live.replace(enabled_config(config, cooldown_minutes=0))

    class Crashy(FakeTransport):
        def send_photo(self, chat_id, photo, caption):
            if not self.photos:
                self.photos.append(("crash", photo, caption))
                raise RuntimeError("boom")
            super().send_photo(chat_id, photo, caption)

    t = Crashy()
    n = TelegramNotifier(live, store, t)
    n.start()
    try:
        n.consider(1, Path("a"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW)
        n.consider(2, Path("b"), Prediction(True, 0.9, "Blue Tit", 0.8), NOW, now=NOW)
        assert wait_until(lambda: len(t.photos) == 2)
    finally:
        n.stop()


def test_send_test_uses_site_name_and_propagates_errors(live, config, store):
    live.replace(enabled_config(config))
    t = FakeTransport()
    n = TelegramNotifier(live, store, t)
    n.send_test()
    assert t.texts == [("42", f"birdy test alert from {config.site.name}")]

    class Bad(FakeTransport):
        def send_text(self, chat_id, text):
            raise TelegramError("Unauthorized", 401)

    with pytest.raises(TelegramError):
        TelegramNotifier(live, store, Bad()).send_test()
```

- [ ] **Step 2: Run to verify failure**

`.venv/bin/pytest tests/test_notify.py -q` → `ModuleNotFoundError: No module named 'birdy.notify'`.

- [ ] **Step 3: Write `birdy/notify.py`**

```python
"""Telegram alerts: decide (AlertPolicy), queue, and send (TelegramNotifier)."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from birdy.classifier import Prediction
from birdy.config import Config
from birdy.live import LiveConfig
from birdy.store import Store

log = logging.getLogger(__name__)

RETRY_DELAYS = (5.0, 15.0, 45.0)
TELEGRAM_API = "https://api.telegram.org"


class TelegramError(Exception):
    """Telegram refused or could not be reached. Never contains the bot token."""

    def __init__(self, description: str, status: int | None = None) -> None:
        super().__init__(description)
        self.description = description
        self.status = status


@dataclass(frozen=True)
class TelegramSettings:
    """The live-tunable subset of [telegram]."""

    watchlist: tuple[str, ...]
    alert_new_species: bool
    cooldown_minutes: int

    @classmethod
    def from_config(cls, cfg: Config) -> TelegramSettings:
        t = cfg.telegram
        return cls(t.watchlist, t.alert_new_species, t.cooldown_minutes)


@dataclass(frozen=True)
class Alert:
    capture_id: int
    photo_path: Path
    species: str
    species_confidence: float
    captured_at: datetime
    reason: str  # "watchlist" | "new"


class AlertPolicy:
    """Pure decision logic. State: last alert time per species (in memory; lost on restart)."""

    def __init__(self) -> None:
        self._last: dict[str, datetime] = {}

    def decide(self, species: str, first_seen: bool, settings: TelegramSettings, now: datetime) -> str | None:
        key = species.casefold()
        if first_seen and settings.alert_new_species:
            self._last[key] = now
            return "new"
        if key not in {w.strip().casefold() for w in settings.watchlist}:
            return None
        last = self._last.get(key)
        if last is not None and now - last < timedelta(minutes=settings.cooldown_minutes):
            return None
        self._last[key] = now
        return "watchlist"


def caption_for(alert: Alert, base_url: str) -> str:
    pct = int(round(alert.species_confidence * 100))
    text = f"🐦 {alert.species} ({pct}%) · {alert.captured_at.astimezone():%H:%M}"
    if alert.reason == "new":
        text += " · new species!"
    return f"{text}\n{base_url}/?species={urllib.parse.quote(alert.species)}"


class TelegramTransport(Protocol):
    def send_photo(self, chat_id: str, photo: Path, caption: str) -> None: ...
    def send_text(self, chat_id: str, text: str) -> None: ...


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    data = path.read_bytes()  # FileNotFoundError propagates to the caller
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n".encode() + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _telegram_call(token: str, method: str, body: bytes, content_type: str, urlopen, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{TELEGRAM_API}/bot{token}/{method}", data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            desc = json.loads(e.read()).get("description", "")
        except Exception:
            desc = ""
        # Never include the URL (it carries the token) in the message.
        raise TelegramError(desc or f"HTTP {e.code}", e.code) from None
    except urllib.error.URLError as e:
        raise TelegramError(f"network error: {e.reason}") from None
    if not payload.get("ok"):
        raise TelegramError(str(payload.get("description", "unknown error")))
    return payload


class HttpTelegramTransport:
    def __init__(self, token: str, timeout: float = 20.0, urlopen: Callable = urllib.request.urlopen) -> None:
        self._token = token
        self._timeout = timeout
        self._urlopen = urlopen

    def send_text(self, chat_id: str, text: str) -> None:
        body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        _telegram_call(self._token, "sendMessage", body, "application/x-www-form-urlencoded", self._urlopen, self._timeout)

    def send_photo(self, chat_id: str, photo: Path, caption: str) -> None:
        body, ctype = _multipart({"chat_id": chat_id, "caption": caption}, "photo", photo)
        _telegram_call(self._token, "sendPhoto", body, ctype, self._urlopen, self._timeout)


def fetch_chat_ids(token: str, urlopen: Callable = urllib.request.urlopen, timeout: float = 20.0) -> list[tuple[int, str, str]]:
    """(chat id, name, last text) for every chat that has messaged the bot (getUpdates)."""
    payload = _telegram_call(token, "getUpdates", b"", "application/x-www-form-urlencoded", urlopen, timeout)
    chats: dict[int, tuple[int, str, str]] = {}
    for upd in payload.get("result", []):
        msg = upd.get("message")
        if not msg:
            continue
        chat = msg["chat"]
        name = chat.get("first_name") or chat.get("title") or chat.get("username") or ""
        chats[chat["id"]] = (chat["id"], name, msg.get("text", ""))
    return list(chats.values())


class TelegramNotifier:
    """Bounded queue + one worker thread. `consider()` never blocks the caller."""

    def __init__(
        self,
        live: LiveConfig,
        store: Store,
        transport: TelegramTransport,
        policy: AlertPolicy | None = None,
        maxsize: int = 20,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._live = live
        self._store = store
        self._transport = transport
        self._policy = policy or AlertPolicy()
        self._queue: queue.Queue[Alert] = queue.Queue(maxsize=maxsize)
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telegram", daemon=True)

    def consider(self, capture_id: int, photo_path: Path, pred: Prediction, captured_at: datetime, now: datetime | None = None) -> str | None:
        if not pred.is_bird or not pred.species:
            return None
        cfg = self._live.get()
        if not cfg.telegram.enabled:
            return None
        first_seen = self._store.count("birds", species=pred.species) == 1  # this capture is the only one
        reason = self._policy.decide(pred.species, first_seen, TelegramSettings.from_config(cfg), now or datetime.now().astimezone())
        if reason is None:
            return None
        alert = Alert(capture_id, photo_path, pred.species, pred.species_confidence or 0.0, captured_at, reason)
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            log.warning("alert queue full; dropping alert for capture %d", capture_id)
            return None
        return reason

    def send_test(self) -> None:
        """Synchronous; raises TelegramError/OSError to the caller (the settings page shows it)."""
        cfg = self._live.get()
        self._transport.send_text(cfg.telegram.chat_id, f"birdy test alert from {cfg.site.name}")

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._deliver(alert)
            except Exception:
                log.exception("alert delivery crashed for capture %d; dropping", alert.capture_id)

    def _deliver(self, alert: Alert) -> None:
        cfg = self._live.get()
        chat_id = cfg.telegram.chat_id
        caption = caption_for(alert, cfg.telegram.base_url)
        last: Exception | None = None
        for attempt, delay in enumerate((0.0, *RETRY_DELAYS), start=1):
            if delay:
                self._sleep(delay)
            try:
                try:
                    self._transport.send_photo(chat_id, alert.photo_path, caption)
                except FileNotFoundError:
                    self._transport.send_text(chat_id, caption)  # photo purged meanwhile: text-only
                log.info("alert sent: %s (%s) capture %d", alert.species, alert.reason, alert.capture_id)
                return
            except (TelegramError, OSError) as e:
                last = e
                log.warning("alert send failed (attempt %d): %s", attempt, e)
        log.error("alert dropped after %d attempts: %s", 1 + len(RETRY_DELAYS), last)
```

- [ ] **Step 4: Run tests**

`.venv/bin/pytest tests/test_notify.py -q` → `21 passed`.

- [ ] **Step 5: Commit**

```bash
git add birdy/notify.py tests/test_notify.py
git commit -m "Add Telegram alert policy, transport and notifier"
```

---

### Task 3: Pipeline hook `[opus-ok]`

**Files:**
- Modify: `birdy/pipeline.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Update tests**

Every call `p.process(ev.capture_id, ev.path)` in `tests/test_pipeline.py` gains `ev.captured_at` as a third argument. Append:
```python
class FakeNotifier:
    def __init__(self, raise_exc=None):
        self.calls = []
        self.raise_exc = raise_exc

    def consider(self, capture_id, photo_path, pred, captured_at):
        self.calls.append((capture_id, photo_path, pred.species, captured_at))
        if self.raise_exc:
            raise self.raise_exc
        return "watchlist"


def test_live_process_notifies(store, live):
    n = FakeNotifier()
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert n.calls == [(ev.capture_id, ev.path, "Robin", ev.captured_at)]
    assert store.get(ev.capture_id).status == "done"


def test_notifier_exception_does_not_mark_error(store, live):
    p = Pipeline(store, SpeciesFake(), live, notifier=FakeNotifier(raise_exc=RuntimeError("boom")))
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).status == "done"


def test_backfill_never_notifies(store, live):
    n = FakeNotifier()
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    p.sweep()
    assert store.get(ev.capture_id).species == "Robin" and n.calls == []


def test_queued_event_passes_captured_at(store, live):
    n = FakeNotifier()
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    p.start()
    try:
        ev = add(store, 0)
        p.submit(ev)
        assert wait_until(lambda: n.calls)
        assert n.calls[0][3] == ev.captured_at
    finally:
        p.stop()
```

- [ ] **Step 2: Run to verify failure** → `TypeError: Pipeline.__init__() got an unexpected keyword argument 'notifier'` / `process() takes 3 positional arguments`.

- [ ] **Step 3: Modify `birdy/pipeline.py`**

Constructor: add `notifier: "TelegramNotifier | None" = None` after `live` (import `from typing import TYPE_CHECKING`; `if TYPE_CHECKING: from birdy.notify import TelegramNotifier`), store as `self._notifier`. `process` becomes `process(self, capture_id: int, path: Path, captured_at: datetime)`; after the success log line add:
```python
        if self._notifier is not None:
            try:
                reason = self._notifier.consider(capture_id, path, pred, captured_at)
                if reason:
                    log.info("alert queued for capture %d (%s)", capture_id, reason)
            except Exception:
                log.exception("alert decision failed for capture %d", capture_id)
```
`sweep`: `self.process(c.id, self._store.photos_dir / c.path, c.captured_at)`. `_run`: `self.process(ev.capture_id, ev.path, ev.captured_at)`. `_backfill_species` unchanged (never notifies).

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_pipeline.py -q` → all pass (17 + 4). **Commit**: `git add birdy/pipeline.py tests/test_pipeline.py && git commit -m "Pipeline: hand live classifications to the Telegram notifier"`.

---

### Task 4: Settings page — Alerts group, read-only entries, test button `[opus-ok]`

**Files:**
- Modify: `birdy/web/app.py`, `birdy/web/templates/settings.html`, `tests/test_web.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_web.py`: `post_settings()`'s form gains `"watchlist": "", "alert_new_species": "on", "cooldown_minutes": "30"`. Add a `notifier` fixture and pass it to the client:
```python
from birdy.notify import TelegramError


class FakeNotifier:
    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.tests = 0

    def send_test(self):
        self.tests += 1
        if self.raise_exc:
            raise self.raise_exc

    def consider(self, *a, **k):
        return None


@pytest.fixture
def notifier():
    return FakeNotifier()
```
and change the `client` fixture to `create_app(store, state, live, config_path, classifier=classifier, notifier=notifier)` (add `notifier` to its parameters). Tests:
```python
def test_settings_alerts_group_renders(client, store, live, config):
    import dataclasses

    add(store, 0, True)  # a "Robin" bird → species hint
    live.replace(dataclasses.replace(config, telegram=dataclasses.replace(config.telegram, watchlist=("Hoopoe",), bot_token="SECRET123", chat_id="42")))
    r = client.get("/settings").text
    assert '<textarea id="watchlist" name="watchlist"' in r and ">Hoopoe</textarea>" in r
    assert 'name="alert_new_species"' in r and "checked" in r
    assert 'name="cooldown_minutes"' in r
    assert "Robin" in r  # species hint
    assert "telegram.bot_token" in r and "set" in r and "SECRET123" not in r
    assert "Send test alert" in r


def test_settings_post_saves_alert_fields(client, live, config_path):
    r = post_settings(client, watchlist="Hoopoe\nBlue Tit", cooldown_minutes="10")
    assert r.status_code == 303
    t = live.get().telegram
    assert t.watchlist == ("Hoopoe", "Blue Tit") and t.cooldown_minutes == 10 and t.alert_new_species is True
    data = _base_form()
    data.pop("alert_new_species")  # an unchecked checkbox is simply absent from the POST
    r2 = client.post("/settings", data=data, follow_redirects=False)
    assert r2.status_code == 303 and live.get().telegram.alert_new_species is False


def test_settings_post_invalid_cooldown_highlights_row(client):
    r = post_settings(client, cooldown_minutes="-5")
    assert r.status_code == 422 and "telegram.cooldown_minutes" in r.text
    assert re.search(r'<div class="row error">\s*<label for="cooldown_minutes"', r.text)


def test_test_alert_success(client, notifier):
    r = client.post("/settings/test-alert")
    assert r.status_code == 200 and "test alert sent" in r.text and notifier.tests == 1


def test_test_alert_telegram_error(store, state, live, config_path, classifier):
    bad = FakeNotifier(raise_exc=TelegramError("Unauthorized", 401))
    c = TestClient(create_app(store, state, live, config_path, classifier=classifier, notifier=bad))
    r = c.post("/settings/test-alert")
    assert r.status_code == 502 and "Unauthorized" in r.text


def test_test_alert_disabled_503(store, state, live, config_path, classifier):
    c = TestClient(create_app(store, state, live, config_path, classifier=classifier))
    r = c.post("/settings/test-alert")
    assert r.status_code == 503 and "disabled" in r.text
    assert "disabled" in c.get("/settings").text  # button state explained
```
Refactor `post_settings` so its form dict comes from a module-level `_base_form()` function (returning a fresh dict) that the second test can reuse without the checkbox.

- [ ] **Step 2: Run to verify failure** → `TypeError: create_app() got an unexpected keyword argument 'notifier'` (+ 422s from the missing `cooldown_minutes` until the helper is updated).

- [ ] **Step 3: `birdy/web/app.py`**

- `create_app(..., classifier=None, notifier: Any | None = None)`; import `from birdy.notify import TelegramError` and `from starlette.concurrency import run_in_threadpool` (already imported for `/classify`).
- `settings_context(form=None, error=None, saved=False, test: tuple[bool, str] | None = None)`: defaults dict gains `"watchlist": "\n".join(cfg.telegram.watchlist), "alert_new_species": cfg.telegram.alert_new_species, "cooldown_minutes": cfg.telegram.cooldown_minutes`; context gains `"species_seen": [r.species for r in store.species_summary()]`, `"alerts_enabled": notifier is not None`, `"test_ok": None if test is None else test[0]`, `"test_result": None if test is None else test[1]`; `readonly` gains `"telegram.enabled": cfg.telegram.enabled, "telegram.chat_id": cfg.telegram.chat_id or "–", "telegram.bot_token": "set" if cfg.telegram.bot_token else "not set", "telegram.base_url": cfg.telegram.base_url`.
- New route:
```python
    @app.post("/settings/test-alert", response_class=HTMLResponse)
    async def settings_test_alert(request: Request):
        if notifier is None:
            ctx = settings_context(test=(False, "alerts are disabled: set [telegram] enabled = true (with bot_token and chat_id) in birdy.toml and restart"))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=503)
        try:
            await run_in_threadpool(notifier.send_test)
        except TelegramError as e:
            ctx = settings_context(test=(False, f"Telegram error: {e.description}"))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=502)
        except OSError as e:
            ctx = settings_context(test=(False, f"network error: {e}"))
            return templates.TemplateResponse(request, "settings.html", ctx, status_code=502)
        return templates.TemplateResponse(request, "settings.html", settings_context(test=(True, "test alert sent — check Telegram")))
```

- [ ] **Step 4: `settings.html`**

After the Retention group (inside the main form, before the Save button):
```html
<h2>Alerts (Telegram)</h2>
<div class="row {{ 'error' if error_field == 'watchlist' else '' }}">
  <label for="watchlist">watchlist</label>
  <textarea id="watchlist" name="watchlist" rows="4" placeholder="one species per line">{{ v['watchlist'] }}</textarea>
  <span class="hint">seen so far: {{ species_seen | join(', ') if species_seen else '–' }}</span>
</div>
{% if error_field == 'watchlist' %}<div class="hint error">{{ error_message }}</div>{% endif %}
<div class="row {{ 'error' if error_field == 'alert_new_species' else '' }}">
  <label for="alert_new_species">alert on new species</label>
  <input type="checkbox" id="alert_new_species" name="alert_new_species" value="on" {{ 'checked' if v['alert_new_species'] else '' }}>
  <span class="hint">first sighting of any species</span>
</div>
{{ num("cooldown_minutes", "cooldown_minutes", 0, 1440, 5) }}
```
After the main `</form>`:
```html
<form method="post" action="/settings/test-alert" class="inline">
  <button type="submit" {{ '' if alerts_enabled else 'disabled' }}>Send test alert</button>
  <span class="hint">{{ 'sends "birdy test alert" to your chat' if alerts_enabled else 'alerts disabled — set [telegram] in birdy.toml and restart' }}</span>
</form>
{% if test_result %}<div class="banner {{ 'ok' if test_ok else 'bad' }}">{{ test_result }}</div>{% endif %}
```
CSS: `.row textarea { width: 100%; box-sizing: border-box; padding: 4px 6px; background: var(--card); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; font: inherit; }` and `form.inline { margin-top: 16px; }`.

Note: `v['alert_new_species']` is a bool on GET and the raw string `"on"`/missing on a failed POST re-render; both are truthy/falsy correctly.

- [ ] **Step 5: Run** `.venv/bin/pytest tests/test_web.py -q` → all pass. Full suite green except `tests/test_cli.py` possibly (Task 5). **Commit**: `git add birdy/web/ tests/test_web.py && git commit -m "Settings page: Telegram alerts group and test-alert button"`.

---

### Task 5: CLI wiring + `telegram-chat-id` `[opus-ok]`

**Files:**
- Modify: `birdy/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Tests** (append to `tests/test_cli.py`)

```python
def test_telegram_chat_id_prints_ids(capsys, config, monkeypatch):
    import dataclasses

    from birdy import cli

    monkeypatch.setattr(cli, "fetch_chat_ids", lambda token: [(11, "Yazz", "test")])
    cfg = dataclasses.replace(config, telegram=dataclasses.replace(config.telegram, bot_token="T"))
    assert cli.cmd_telegram_chat_id(cfg, None) == 0
    assert "11\tYazz\ttest" in capsys.readouterr().out


def test_telegram_chat_id_without_token_exits_2(capsys, config):
    from birdy import cli

    assert cli.cmd_telegram_chat_id(config, None) == 2
    assert "bot_token" in capsys.readouterr().err


def test_telegram_chat_id_no_chats_exits_1(capsys, config, monkeypatch):
    from birdy import cli

    monkeypatch.setattr(cli, "fetch_chat_ids", lambda token: [])
    assert cli.cmd_telegram_chat_id(config, "T") == 1
    assert "send" in capsys.readouterr().err.lower()


def test_parser_telegram_chat_id():
    a = build_parser().parse_args(["telegram-chat-id", "--token", "x"])
    assert a.command == "telegram-chat-id" and a.token == "x"


def test_cmd_run_builds_notifier_when_enabled(tmp_path, config_path, monkeypatch):
    import dataclasses

    from birdy import cli
    from birdy.config import load_config

    cfg = load_config(config_path)
    cfg = dataclasses.replace(cfg, telegram=dataclasses.replace(cfg.telegram, enabled=True, bot_token="T", chat_id="42"))
    built = {}

    class FakeNotifier:
        def __init__(self, live, store, transport):
            built["transport"] = transport
            built["self"] = self
        def start(self): built["started"] = True
        def stop(self, timeout=5.0): built["stopped"] = True

    monkeypatch.setattr(cli, "TelegramNotifier", FakeNotifier)
    monkeypatch.setattr(cli, "HttpTelegramTransport", lambda token: ("transport", token))
    monkeypatch.setattr(cli, "_classifier", lambda c, l: object())
    seen = {}
    monkeypatch.setattr("birdy.capture.run_capture", lambda live, store, pipeline, state, stop: seen.update(pipeline=pipeline))
    import birdy.web.app as webapp
    real_create_app = webapp.create_app
    monkeypatch.setattr(webapp, "create_app", lambda *a, **k: seen.update(notifier=k.get("notifier")) or real_create_app(*a, **k))
    cfg = dataclasses.replace(cfg, web=dataclasses.replace(cfg.web, port=_free_port()))
    assert cli.cmd_run(cfg, config_path) == 0
    assert built["transport"] == ("transport", "T") and built["started"] and built["stopped"]
    assert seen["notifier"] is built["self"] and seen["pipeline"]._notifier is built["self"]
```
(`_free_port` already exists in the file. `test_cmd_run_passes_live_everywhere` keeps working: the default config has `enabled = false`, so no notifier is built; add `assert seen_create_app_kwargs.get("notifier") is None` there if the recording is easy, otherwise leave it.)

- [ ] **Step 2: Run to verify failure** → `AttributeError: module 'birdy.cli' has no attribute 'cmd_telegram_chat_id'`.

- [ ] **Step 3: `birdy/cli.py`**

Imports: `from birdy.notify import HttpTelegramTransport, TelegramError, TelegramNotifier, fetch_chat_ids`. Parser: `tg = sub.add_parser("telegram-chat-id", help="print the chat ids that have messaged the bot (bot setup)"); tg.add_argument("--token", default=None, help="override [telegram] bot_token")`.

`cmd_run`: after `clf = _classifier(config, live)`:
```python
    notifier = None
    if config.telegram.enabled:
        notifier = TelegramNotifier(live, store, HttpTelegramTransport(config.telegram.bot_token))
        notifier.start()
    pipeline = Pipeline(store, clf, live, notifier=notifier)
```
`create_app(..., classifier=clf, notifier=notifier)`; the startup log gains `alerts=%s` (`config.telegram.enabled`); in `finally`, after `pipeline.stop()`: `if notifier is not None: notifier.stop()`.

```python
def cmd_telegram_chat_id(config: Config, token_override: str | None) -> int:
    token = token_override or config.telegram.bot_token
    if not token:
        print("no bot token: set [telegram] bot_token in birdy.toml or pass --token", file=sys.stderr)
        return 2
    try:
        chats = fetch_chat_ids(token)
    except (TelegramError, OSError) as e:
        print(f"telegram error: {e}", file=sys.stderr)
        return 1
    if not chats:
        print("no chats yet: open your bot in Telegram, send it any message, then run this again", file=sys.stderr)
        return 1
    for chat_id, name, text in chats:
        print(f"{chat_id}\t{name}\t{text}")
    return 0
```
`main()`: `if args.command == "telegram-chat-id": return cmd_telegram_chat_id(config, args.token)` (outside the `ModelLoadError` try).

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_cli.py -q` → all pass; full suite green; pyflakes silent. **Commit**: `git add birdy/cli.py tests/test_cli.py && git commit -m "CLI: wire Telegram notifier; add telegram-chat-id"`.

---

### Task 6: Docs, deploy, enable on the Pi, verify

- [ ] **Step 1: Docs**. `CLAUDE.md`: architecture bullet for `notify.py` (policy/queue/worker, what triggers, cooldown, never from backfill, token never logged or rendered); the Commands block gains `birdy telegram-chat-id`; a "Bot setup" mini-section (BotFather → token into the Pi's `birdy.toml` → message the bot → `telegram-chat-id` → `enabled = true` → restart → "Send test alert"). Gotcha: `telegram.bot_token` is written back by every settings-page save (so it survives), but must never appear in git/docs/logs. Commit: `git add CLAUDE.md && git commit -m "Docs: Telegram alerts"`.

- [ ] **Step 2: Deploy + enable** (from the Mac; the token and chat id are known to the session — the chat id is `1115872433`; the token is NOT in the repo and is written only to the Pi's `birdy.toml`):
```bash
deploy/deploy.sh
ssh yassine@birdy.local 'cd ~/pi-birdy && .venv/bin/python - <<EOF
import tomllib, tomli_w, pathlib
p = pathlib.Path("birdy.toml"); d = tomllib.load(p.open("rb"))
d.setdefault("telegram", {}).update(enabled=True, bot_token="<TOKEN>", chat_id="1115872433", watchlist=["Eurasian Hoopoe", "Eurasian Blue Tit"])
p.write_text(tomli_w.dumps(d))
EOF
sudo -n systemctl restart birdy'
```
(Insert the real token; it must not be committed anywhere.)

- [ ] **Step 3: Verify**
1. `journalctl -u birdy -n 5` shows `birdy starting … alerts=True`, no `config error`.
2. `curl -s -o /dev/null -w "%{http_code}\n" -X POST http://birdy.local:8000/settings/test-alert` → 200, and a "birdy test alert from home" message arrives in Telegram.
3. `curl -s http://birdy.local:8000/settings | grep -c SECRET` → must be 0 for the real token string (grep for its first 8 chars) and the read-only block shows `telegram.bot_token = set`.
4. Show the phone a hoopoe → within ~5 s a photo alert `🐦 Eurasian Hoopoe (NN%) · HH:MM` with the link; a second showing within 30 min → no alert (cooldown); journal shows `alert queued` / `alert sent`.
5. Show a species never seen before (any new bird photo) → `… · new species!` alert even though it's not on the watchlist.

- [ ] **Step 4: Finish** → `superpowers:finishing-a-development-branch`.
