"""Hourly deletion of old non-bird captures and a disk-space check."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from birdy.store import Store

log = logging.getLogger(__name__)

LOW_DISK_BYTES = 1024 * 1024 * 1024  # 1 GB


def run_once(store: Store, non_bird_days: int, now: datetime | None = None) -> int:
    deleted = store.purge_non_birds(older_than_days=non_bird_days, now=now)
    free = store.disk_free_bytes()
    log.info("retention: deleted %d non-bird captures; disk free %.1f GB", deleted, free / 1e9)
    if free < LOW_DISK_BYTES:
        log.warning("low disk: %.0f MB free", free / 1e6)
    return deleted


class RetentionJob:
    def __init__(self, store: Store, non_bird_days: int, interval: float = 3600.0) -> None:
        self._store = store
        self._days = non_bird_days
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="retention", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                run_once(self._store, self._days)
            except Exception:
                log.exception("retention run failed")
