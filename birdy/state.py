"""Mutable runtime status shared between the capture loop and the web app."""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np

from birdy.daylight import DaylightGate

OVERRIDES = ("on", "off")


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = "starting"  # starting | capturing | daylight_paused | manual_off | camera_error
        self.motion_score = 0
        self.latest_lores: np.ndarray | None = None  # greyscale (h, w) uint8
        self.started_at = time.monotonic()
        # Manual camera switch from the web page: "on"/"off" until `override_until`
        # (the next daylight change when it was set; None = until cleared). In memory
        # only: a restart goes back to the schedule.
        self.override: str | None = None
        self.override_until: datetime | None = None
        self.override_version = 0  # bumped on every set, so the capture loop wakes promptly
        # "Take a photo" from the web page: the capture loop takes it on its next frame.
        self.capture_requested = False

    def update(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                if not hasattr(self, k):
                    raise AttributeError(k)
                setattr(self, k, v)

    def set_override(self, mode: str | None, until: datetime | None = None) -> None:
        if mode is not None and mode not in OVERRIDES:
            raise ValueError(f"override must be one of {OVERRIDES} or None")
        with self._lock:
            self.override = mode
            self.override_until = None if mode is None else until
            self.override_version += 1

    def request_capture(self) -> None:
        with self._lock:
            self.capture_requested = True

    def take_capture_request(self) -> bool:
        """Consume a pending "Take a photo" request (True at most once per press)."""
        with self._lock:
            pending = self.capture_requested
            self.capture_requested = False
        return pending

    def camera_wanted(self, gate: DaylightGate, now: datetime) -> bool:
        """Whether the camera should run now: an unexpired override wins, else the daylight gate.

        An override that has reached its `override_until` is cleared here.
        """
        with self._lock:
            if self.override is not None and self.override_until is not None and now >= self.override_until:
                self.override = None
                self.override_until = None
            mode = self.override
        if mode is not None:
            return mode == "on"
        return gate.is_active(now)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "motion_score": self.motion_score,
                "uptime_seconds": int(time.monotonic() - self.started_at),
                "override": self.override,
                "override_until": None if self.override_until is None else self.override_until.isoformat(),
            }

    def lores_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self.latest_lores is None else self.latest_lores.copy()
