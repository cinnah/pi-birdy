"""Thread-safe holder for the current Config, with a version counter.

Consumers call `get()` whenever they need a tunable value and compare
`version` to decide whether to rebuild derived objects (e.g. the motion
detector). `Config` is frozen, so a returned snapshot never changes.
"""

from __future__ import annotations

import threading

from birdy.config import Config


class LiveConfig:
    def __init__(self, config: Config) -> None:
        self._lock = threading.Lock()
        self._config = config
        self._version = 0

    def get(self) -> Config:
        with self._lock:
            return self._config

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def replace(self, config: Config) -> None:
        with self._lock:
            self._config = config
            self._version += 1
