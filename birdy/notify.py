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

    def consider(
        self, capture_id: int, photo_path: Path, pred: Prediction, captured_at: datetime, now: datetime | None = None
    ) -> str | None:
        if not pred.is_bird or not pred.species:
            return None
        cfg = self._live.get()
        if not cfg.telegram.enabled:
            return None
        first_seen = self._store.count("birds", species=pred.species) == 1  # this capture is the only one
        reason = self._policy.decide(
            pred.species, first_seen, TelegramSettings.from_config(cfg), now or datetime.now().astimezone()
        )
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
