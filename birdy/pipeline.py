"""Queue between capture and classification, plus a backlog sweep."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from birdy.classifier import Classifier
from birdy.motion import BBox
from birdy.store import Store

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionEvent:
    capture_id: int
    path: Path
    captured_at: datetime
    motion_score: int


class Pipeline:
    def __init__(
        self,
        store: Store,
        classifier: Classifier,
        roi_bbox: BBox | None,
        maxsize: int = 50,
        sweep_interval: float = 300.0,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._roi_bbox = roi_bbox
        self._queue: queue.Queue[MotionEvent] = queue.Queue(maxsize=maxsize)
        self._sweep_interval = sweep_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def submit(self, event: MotionEvent) -> bool:
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            log.warning("pipeline queue full; capture %d left pending", event.capture_id)
            return False

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def process(self, capture_id: int, path: Path) -> None:
        try:
            pred = self._classifier.classify(path, self._roi_bbox)
        except Exception:
            log.exception("classification failed for %s", path)
            self._store.mark_error(capture_id)
            return
        try:
            self._store.update_classification(capture_id, pred.is_bird, pred.confidence, pred.species, pred.species_confidence)
        except Exception:
            log.exception("storing classification failed for capture %d", capture_id)
            try:
                self._store.mark_error(capture_id)
            except Exception:
                log.exception("mark_error failed for capture %d; left pending", capture_id)
            return
        log.info("capture %d: bird=%s p=%.2f", capture_id, pred.is_bird, pred.confidence)

    def sweep(self) -> None:
        for c in self._store.pending():
            current = self._store.get(c.id)
            if current is None or current.status != "pending":
                continue
            self.process(c.id, self._store.photos_dir / c.path)

    def _run(self) -> None:
        next_sweep = time.monotonic() + self._sweep_interval
        while not self._stop.is_set():
            try:
                ev = self._queue.get(timeout=0.5)
            except queue.Empty:
                ev = None
            if ev is not None:
                try:
                    self.process(ev.capture_id, ev.path)
                except Exception:
                    log.exception("processing capture %d failed", ev.capture_id)
            if time.monotonic() >= next_sweep:
                try:
                    self.sweep()
                except Exception:
                    log.exception("sweep failed")
                next_sweep = time.monotonic() + self._sweep_interval
