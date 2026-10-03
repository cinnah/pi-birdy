"""Mutable runtime status shared between the capture loop and the web app."""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "starting"  # starting | capturing | daylight_paused | camera_error
        self.motion_score = 0
        self.last_capture_at: datetime | None = None
        self.latest_lores: np.ndarray | None = None  # greyscale (h, w) uint8
        self.started_at = time.monotonic()

    def update(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                if not hasattr(self, k):
                    raise AttributeError(k)
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "motion_score": self.motion_score,
                "last_capture_at": None if self.last_capture_at is None else self.last_capture_at.isoformat(),
                "uptime_seconds": int(time.monotonic() - self.started_at),
            }

    def lores_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self.latest_lores is None else self.latest_lores.copy()
