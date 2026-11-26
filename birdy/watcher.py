"""Reload birdy.toml into LiveConfig when it changes on disk (e.g. edited over SSH)."""

from __future__ import annotations

import logging
import threading
import tomllib
from pathlib import Path

from birdy.config import Config, ConfigError, load_config
from birdy.live import LiveConfig

log = logging.getLogger(__name__)


def _restart_only(cfg: Config) -> tuple:
    """Settings only read at startup; changing them on disk needs a restart."""
    c = cfg.classifier
    return (
        cfg.camera, cfg.web, cfg.storage, c.model, c.labels, c.bird_classes,
        cfg.species.model, cfg.species.bank,
        cfg.detector.model, cfg.detector.labels,
    )


class ConfigWatcher:
    def __init__(self, path: Path, live: LiveConfig, interval: float = 5.0) -> None:
        self._path = Path(path)
        self._live = live
        self._interval = interval
        self._last_sig = self._signature()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="config-watcher", daemon=True)

    def _signature(self) -> tuple[int, int, int] | None:
        """(mtime, size, inode): catches rewrites that land within the filesystem's mtime granularity
        and atomic replaces (new inode) that keep the old mtime."""
        try:
            st = self._path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def check_once(self) -> bool:
        """Reload if the file changed and differs from the running config. Returns True on reload."""
        sig = self._signature()
        if sig is None or sig == self._last_sig:
            return False
        self._last_sig = sig  # whatever happens, don't re-process this version of the file
        try:
            cfg = load_config(self._path)
        except (ConfigError, tomllib.TOMLDecodeError, OSError) as e:
            log.error("%s changed but is invalid (%s); keeping running settings", self._path, e)
            return False
        if cfg == self._live.get():
            return False  # e.g. the settings page just wrote it
        if _restart_only(cfg) != _restart_only(self._live.get()):
            log.warning("%s: camera/web/storage/model settings changed; restart birdy to apply them", self._path)
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
