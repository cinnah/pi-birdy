"""Reload birdy.toml into LiveConfig when it changes on disk (e.g. edited over SSH)."""

from __future__ import annotations

import logging
import threading
import tomllib
from pathlib import Path

from birdy.config import ConfigError, load_config
from birdy.live import LiveConfig

log = logging.getLogger(__name__)


class ConfigWatcher:
    def __init__(self, path: Path, live: LiveConfig, interval: float = 5.0) -> None:
        self._path = Path(path)
        self._live = live
        self._interval = interval
        self._last_mtime_ns = self._mtime_ns()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="config-watcher", daemon=True)

    def _mtime_ns(self) -> int | None:
        try:
            return self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def check_once(self) -> bool:
        """Reload if the file changed and differs from the running config. Returns True on reload."""
        mtime = self._mtime_ns()
        if mtime is None or mtime == self._last_mtime_ns:
            return False
        self._last_mtime_ns = mtime  # whatever happens, don't re-process this mtime
        try:
            cfg = load_config(self._path)
        except (ConfigError, tomllib.TOMLDecodeError, OSError) as e:
            log.error("%s changed but is invalid (%s); keeping running settings", self._path, e)
            return False
        if cfg == self._live.get():
            return False  # e.g. the settings page just wrote it
        self._live.replace(cfg)
        log.info("settings reloaded from %s (v%d)", self._path, self._live.version)
        return True

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check_once()
            except Exception:
                log.exception("config watcher failed")
