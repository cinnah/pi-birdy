"""Telegram alerts: decide (AlertPolicy), queue, and send (TelegramNotifier)."""

from __future__ import annotations

import http.client
import json
import logging
import queue
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from birdy.config import Config
from birdy.live import LiveConfig

log = logging.getLogger(__name__)

RETRY_DELAYS = (5.0, 15.0, 45.0)
NO_RETRY_STATUSES = frozenset({400, 401, 403})  # bad request / bad token / blocked bot: retrying cannot help
TELEGRAM_API = "https://api.telegram.org"

# Telegram's sendPhoto rejects files over 10 MB and photos whose width + height exceed 10,000.
TG_MAX_BYTES = 8 * 1024 * 1024
TG_MAX_SIDE_SUM = 10_000
TG_SEND_SIDE = 2560


class TelegramError(Exception):
    """Telegram refused or could not be reached. Never contains the bot token."""

    def __init__(self, description: str, status: int | None = None) -> None:
        super().__init__(description)
        self.description = description
        self.status = status


@dataclass(frozen=True)
class TelegramSettings:
    """The live-tunable subset of [telegram]."""

    no_alert: tuple[str, ...]
    cooldown_minutes: int

    @classmethod
    def from_config(cls, cfg: Config) -> TelegramSettings:
        t = cfg.telegram
        return cls(t.no_alert, t.cooldown_minutes)


@dataclass(frozen=True)
class Alert:
    capture_id: int
    photo_path: Path
    species: str
    species_confidence: float
    captured_at: datetime
    reason: str  # "visit" | "new"


class AlertPolicy:
    """Pure decision logic. State: last alert time per species (in memory; lost on restart)."""

    def __init__(self) -> None:
        self._last: dict[str, datetime] = {}

    def decide(self, species: str, first_seen: bool, settings: TelegramSettings, now: datetime) -> str | None:
        # every species alerts unless it is on the no-alert list; a first sighting skips the cooldown
        key = species.casefold()
        if key in {w.strip().casefold() for w in settings.no_alert}:
            return None
        if first_seen:
            self._last[key] = now
            return "new"
        last = self._last.get(key)
        if last is not None and now - last < timedelta(minutes=settings.cooldown_minutes):
            return None
        self._last[key] = now
        return "visit"


def caption_for(alert: Alert, base_url: str) -> str:
    pct = int(round(alert.species_confidence * 100))
    text = f"🐦 {alert.species} ({pct}%) · {alert.captured_at.astimezone():%H:%M}"
    if alert.reason == "new":
        text += " · new species!"
    return f"{text}\n{base_url}/?species={urllib.parse.quote(alert.species)}"


class TelegramTransport(Protocol):
    def send_photo(self, chat_id: str, photo: Path, caption: str) -> None: ...
    def send_text(self, chat_id: str, text: str) -> None: ...


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


def _multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    data = _telegram_jpeg(path)  # FileNotFoundError propagates to the caller
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


def _redact(text: str, token: str) -> str:
    """Replace the bot token (raw or URL-encoded) with *** so it can't leak via error messages."""
    if not token:
        return text
    for form in {token, urllib.parse.quote(token), urllib.parse.quote_plus(token)}:
        text = text.replace(form, "***")
    return text


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
        raise TelegramError(_redact(desc or f"HTTP {e.code}", token), e.code) from None
    except urllib.error.URLError as e:
        raise TelegramError(_redact(f"network error: {e.reason}", token)) from None
    except (OSError, http.client.HTTPException, ValueError) as e:
        # Timeouts, socket errors, truncated responses, non-JSON bodies: all one error type for callers.
        # InvalidURL (e.g. whitespace in the token) quotes the request path, hence the redaction.
        raise TelegramError(_redact(f"network error: {type(e).__name__}: {e}", token)) from None
    if not payload.get("ok"):
        raise TelegramError(_redact(str(payload.get("description", "unknown error")), token))
    return payload


class HttpTelegramTransport:
    def __init__(self, token: str | Callable[[], str], timeout: float = 20.0, urlopen: Callable = urllib.request.urlopen) -> None:
        """`token`: the bot token, or a function returning the current one (it is a live setting)."""
        self._token_source = token
        self._timeout = timeout
        self._urlopen = urlopen

    @property
    def _token(self) -> str:
        t = self._token_source
        return (t() if callable(t) else t).strip()

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
        transport: TelegramTransport,
        policy: AlertPolicy | None = None,
        maxsize: int = 20,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self._live = live
        self._transport = transport
        self._policy = policy or AlertPolicy()
        self._queue: queue.Queue[Alert] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait  # Event.wait: a retry sleep ends early on stop()
        self._thread = threading.Thread(target=self._run, name="telegram", daemon=True)

    def consider(
        self,
        capture_id: int,
        photo_path: Path,
        species: str | None,
        species_confidence: float,
        captured_at: datetime,
        first_seen: bool,
        now: datetime | None = None,
    ) -> str | None:
        """Fed by `Visits` once per visit (its best photo). `first_seen`: this is the first visit with
        that species. Returns the alert reason when one is queued, else None."""
        if not species:
            return None
        cfg = self._live.get()
        if not cfg.telegram.enabled:
            return None
        reason = self._policy.decide(
            species, first_seen, TelegramSettings.from_config(cfg), now or datetime.now().astimezone()
        )
        if reason is None:
            return None
        alert = Alert(capture_id, photo_path, species, species_confidence or 0.0, captured_at, reason)
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            log.warning("alert queue full; dropping alert for capture %d", capture_id)
            return None
        return reason

    def send_test(self) -> None:
        """Synchronous; raises TelegramError to the caller (the settings page shows it)."""
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
        tg = self._live.get().telegram      # chat and link are live settings: read per message
        chat_id = tg.chat_id
        caption = caption_for(alert, tg.base_url)
        last: Exception | None = None
        for attempt, delay in enumerate((0.0, *RETRY_DELAYS), start=1):
            if delay:
                self._sleep(delay)
                if self._stop.is_set():
                    log.debug("stopping; dropping alert for capture %d mid-retry", alert.capture_id)
                    return
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
                if isinstance(e, TelegramError) and e.status in NO_RETRY_STATUSES:
                    break
        log.error("alert dropped after %d attempt(s): %s", attempt, last)
