import dataclasses
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

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
    def __init__(self, fail_times: int = 0, missing_photo: bool = False, fail_status: int = 502):
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.missing_photo = missing_photo
        self.photos: list[tuple[str, Path, str]] = []
        self.texts: list[tuple[str, str]] = []

    def send_photo(self, chat_id, photo, caption):
        if self.fail_times:
            self.fail_times -= 1
            raise TelegramError("Bad Gateway", self.fail_status)
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


def test_consider_ignores_unknown_species_and_disabled(live, config, store):
    n = TelegramNotifier(live, FakeTransport())
    assert n.consider(1, Path("x"), "Blue Tit", 0.8, NOW, True) is None  # telegram disabled in the default config
    live.replace(enabled_config(config))
    assert n.consider(1, Path("x"), None, 0.0, NOW, True) is None


def test_consider_queues_and_worker_sends_photo(live, config, store):
    live.replace(enabled_config(config))
    cid, path = add_bird(store, "Blue Tit")
    t = FakeTransport()
    n = TelegramNotifier(live, t)
    n.start()
    try:
        assert n.consider(cid, path, "Blue Tit", 0.8, NOW, True, now=NOW) == "new"  # first ever Blue Tit
        assert wait_until(lambda: t.photos)
        chat, photo, caption = t.photos[0]
        assert chat == "42" and photo == path
        assert caption.startswith("🐦 Blue Tit (80%)") and "new species!" in caption
        assert "/?species=Blue%20Tit" in caption
    finally:
        n.stop()


def test_consider_takes_first_seen_from_caller(live, config, store):
    live.replace(enabled_config(config, watchlist=()))
    n = TelegramNotifier(live, FakeTransport())
    assert n.consider(1, Path("a"), "Robin", 0.9, NOW, first_seen=True, now=NOW) == "new"
    assert n.consider(2, Path("b"), "Robin", 0.9, NOW, first_seen=False, now=NOW + timedelta(hours=2)) is None


def test_missing_photo_falls_back_to_text(live, config, store):
    live.replace(enabled_config(config))
    t = FakeTransport(missing_photo=True)
    n = TelegramNotifier(live, t)
    n.start()
    try:
        n.consider(1, Path("gone.jpg"), "Blue Tit", 0.8, NOW, False, now=NOW)
        assert wait_until(lambda: t.texts)
        assert t.texts[0][1].startswith("🐦 Blue Tit")
    finally:
        n.stop()


def test_retries_then_drops(live, config, store, caplog):
    live.replace(enabled_config(config))
    sleeps = []
    t = FakeTransport(fail_times=4)
    n = TelegramNotifier(live, t, sleep=sleeps.append)
    n.start()
    try:
        with caplog.at_level(logging.WARNING):
            n.consider(1, Path("x.jpg"), "Blue Tit", 0.8, NOW, False, now=NOW)
            assert wait_until(lambda: "alert dropped" in caplog.text)
        assert sleeps == [5.0, 15.0, 45.0]
        assert t.photos == []
    finally:
        n.stop()


@pytest.mark.parametrize("status", [400, 401, 403])
def test_permanent_errors_are_not_retried(live, config, store, caplog, status):
    live.replace(enabled_config(config))
    sleeps = []
    t = FakeTransport(fail_times=4, fail_status=status)
    n = TelegramNotifier(live, t, sleep=sleeps.append)
    n.start()
    try:
        with caplog.at_level(logging.WARNING):
            n.consider(1, Path("x.jpg"), "Blue Tit", 0.8, NOW, False, now=NOW)
            assert wait_until(lambda: "alert dropped" in caplog.text)
        assert sleeps == []
        assert t.fail_times == 3  # exactly one attempt
    finally:
        n.stop()


def test_retry_succeeds_on_second_attempt(live, config, store):
    live.replace(enabled_config(config))
    sleeps = []
    t = FakeTransport(fail_times=1)
    n = TelegramNotifier(live, t, sleep=sleeps.append)
    n.start()
    try:
        n.consider(1, Path("x.jpg"), "Blue Tit", 0.8, NOW, False, now=NOW)
        assert wait_until(lambda: t.photos)
        assert sleeps == [5.0]
    finally:
        n.stop()


def test_queue_full_drops_with_warning(live, config, store, caplog):
    live.replace(enabled_config(config, cooldown_minutes=0))
    n = TelegramNotifier(live, FakeTransport(), maxsize=1)  # not started: nothing drains
    with caplog.at_level(logging.WARNING):
        assert n.consider(1, Path("a"), "Blue Tit", 0.8, NOW, False, now=NOW) == "watchlist"
        assert n.consider(2, Path("b"), "Blue Tit", 0.8, NOW, False, now=NOW) is None
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
    n = TelegramNotifier(live, t)
    n.start()
    try:
        n.consider(1, Path("a"), "Blue Tit", 0.8, NOW, False, now=NOW)
        n.consider(2, Path("b"), "Blue Tit", 0.8, NOW, False, now=NOW)
        assert wait_until(lambda: len(t.photos) == 2)
    finally:
        n.stop()


def test_send_test_uses_site_name_and_propagates_errors(live, config, store):
    live.replace(enabled_config(config))
    t = FakeTransport()
    n = TelegramNotifier(live, t)
    n.send_test()
    assert t.texts == [("42", f"birdy test alert from {config.site.name}")]

    class Bad(FakeTransport):
        def send_text(self, chat_id, text):
            raise TelegramError("Unauthorized", 401)

    with pytest.raises(TelegramError):
        TelegramNotifier(live, Bad()).send_test()


# --- review follow-ups: every transport failure is a TelegramError ------------

def _http_error(code: int, body: bytes):
    import io
    import urllib.error

    return urllib.error.HTTPError("https://api.telegram.org/botT/x", code, "err", {}, io.BytesIO(body))


def test_http_error_with_json_body_gives_description_and_status():
    def urlopen(req, timeout):
        raise _http_error(400, b'{"ok": false, "description": "Bad Request: chat not found"}')

    with pytest.raises(TelegramError) as ei:
        HttpTelegramTransport("TOKEN123", urlopen=urlopen).send_text("42", "hi")
    assert ei.value.description == "Bad Request: chat not found" and ei.value.status == 400
    assert "TOKEN123" not in str(ei.value)


def test_url_error_is_telegram_error():
    import urllib.error

    def urlopen(req, timeout):
        raise urllib.error.URLError("Name or service not known")

    with pytest.raises(TelegramError, match="network error"):
        HttpTelegramTransport("T", urlopen=urlopen).send_text("42", "hi")


def test_timeout_is_telegram_error():
    def urlopen(req, timeout):
        raise TimeoutError("timed out")

    with pytest.raises(TelegramError, match="TimeoutError"):
        HttpTelegramTransport("123456:ABCdef", urlopen=urlopen).send_text("42", "hi")  # a 1-char token would be redacted out of "TimeoutError"


def test_non_json_200_is_telegram_error():
    class Html(FakeResponse):
        def read(self):
            return b"<html>502 Bad Gateway</html>"

    with pytest.raises(TelegramError, match="network error"):
        HttpTelegramTransport("T", urlopen=lambda req, timeout: Html({})).send_text("42", "hi")


def test_multipart_body_details(tmp_path):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"\xff\xd8JPEGDATA")
    seen = {}

    def urlopen(req, timeout):
        seen["body"], seen["ctype"] = req.data, req.get_header("Content-type")
        return FakeResponse({"ok": True})

    HttpTelegramTransport("T", urlopen=urlopen).send_photo("42", photo, "🐦 Blue Tit · new species!")
    body = seen["body"]
    boundary = seen["ctype"].split("boundary=", 1)[1].encode()
    assert "🐦 Blue Tit · new species!".encode("utf-8") in body
    assert b'name="photo"; filename="p.jpg"\r\nContent-Type: image/jpeg\r\n\r\n\xff\xd8JPEGDATA\r\n' in body
    assert body.endswith(b"--" + boundary + b"--\r\n")


def test_text_fallback_failure_retries(live, config, store):
    live.replace(enabled_config(config))
    sleeps = []

    class TextFlaky(FakeTransport):
        def __init__(self):
            super().__init__(missing_photo=True)
            self.text_fail = 1

        def send_text(self, chat_id, text):
            if self.text_fail:
                self.text_fail -= 1
                raise TelegramError("Too Many Requests", 429)
            super().send_text(chat_id, text)

    t = TextFlaky()
    n = TelegramNotifier(live, t, sleep=sleeps.append)
    n.start()
    try:
        n.consider(1, Path("gone.jpg"), "Blue Tit", 0.8, NOW, False, now=NOW)
        assert wait_until(lambda: t.texts)
        assert sleeps == [5.0] and t.photos == []
    finally:
        n.stop()


def test_stop_interrupts_retry_sleep(live, config, store):
    live.replace(enabled_config(config))
    t = FakeTransport(fail_times=1)
    n = TelegramNotifier(live, t)  # default sleep: Event.wait, so stop() cuts the 5 s backoff short
    n.start()
    n.consider(1, Path("x.jpg"), "Blue Tit", 0.8, NOW, False, now=NOW)
    t0 = time.monotonic()
    n.stop()
    assert time.monotonic() - t0 < 1.0
    assert not n._thread.is_alive() and t.photos == []


def test_invalid_url_error_redacts_token():
    import http.client

    def urlopen(req, timeout):
        raise http.client.InvalidURL("URL can't contain control characters. '/botSECRET TOKEN/sendMessage'")

    with pytest.raises(TelegramError) as ei:
        HttpTelegramTransport("SECRET TOKEN", urlopen=urlopen).send_text("42", "hi")
    assert "***" in str(ei.value) and "SECRET" not in str(ei.value)


def test_transport_strips_token():
    seen = {}

    def urlopen(req, timeout):
        seen["url"] = req.full_url
        return FakeResponse({"ok": True})

    HttpTelegramTransport("  TOKEN123\n", urlopen=urlopen).send_text("42", "hi")
    assert seen["url"].endswith("/botTOKEN123/sendMessage")


def _jpeg(path, w, h, noise=True):
    import cv2
    import numpy as np
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8) if noise else np.full((h, w, 3), 120, np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_telegram_jpeg_passes_a_normal_photo_through(tmp_path):
    from birdy.notify import _telegram_jpeg
    p = _jpeg(tmp_path / "a.jpg", 3280, 2464, noise=False)   # a Pi still; pure noise this size is 9.5 MB
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
