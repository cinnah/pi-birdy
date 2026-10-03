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
from birdy.live import LiveConfig
from birdy.motion import roi_bbox
from birdy.store import Capture, Store

log = logging.getLogger(__name__)

SPECIES_BACKLOG_PER_SWEEP = 50


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
        live: LiveConfig,
        maxsize: int = 50,
        sweep_interval: float = 300.0,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._live = live
        self._queue: queue.Queue[MotionEvent] = queue.Queue(maxsize=maxsize)
        self._sweep_interval = sweep_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("pipeline worker did not stop within %.1fs", timeout)

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
        cfg = self._live.get()
        try:
            pred = self._classifier.classify(path, roi_bbox(cfg.motion.roi), threshold=cfg.classifier.threshold)
        except Exception:
            log.exception("classification failed for %s", path)
            try:
                self._store.mark_error(capture_id)
            except Exception:
                log.exception("mark_error failed for capture %d; left pending", capture_id)
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
            if self._stop.is_set():
                break
            current = self._store.get(c.id)
            if current is None or current.status != "pending":
                continue
            self.process(c.id, self._store.photos_dir / c.path)
        # Rows classified before species recognition existed: re-run the (two-stage)
        # classifier so they get a species. Bounded per sweep to keep the Pi responsive,
        # and yields to live events (the remainder drains on the next sweep).
        processed = with_species = rescued = 0
        for c in self._store.species_backlog(limit=SPECIES_BACKLOG_PER_SWEEP):
            if self._stop.is_set() or self._queue.qsize():
                break
            got_species, was_rescued = self._backfill_species(c)
            processed += 1
            with_species += got_species
            rescued += was_rescued
        if processed:
            remaining = self._store.count_species_backlog()
            log.info("backfill: %d rows, %d with species, %d rescued, %d remaining", processed, with_species, rescued, remaining)

    def _backfill_species(self, c: Capture) -> tuple[bool, bool]:
        """Re-classify an already-classified row; returns (got_species, rescued)."""
        cfg = self._live.get()
        path = self._store.photos_dir / c.path
        try:
            pred = self._classifier.classify(path, roi_bbox(cfg.motion.roi), threshold=cfg.classifier.threshold)
            # Never demote: a bird stays a bird. Confidence comes from the new verdict when it
            # is a bird (gate or rescue); the old confidence is kept only when it says non-bird.
            is_bird = bool(c.is_bird) or pred.is_bird
            confidence = pred.confidence if pred.is_bird else c.confidence
            self._store.update_classification(c.id, is_bird, confidence, pred.species, pred.species_confidence)
            return bool(pred.species), (not c.is_bird) and pred.is_bird
        except Exception:
            # Unlike live processing, a failure here must not flip a `done` row to `error`.
            log.exception("species backfill failed for capture %d; marking checked", c.id)
            try:
                self._store.mark_species_checked(c.id)
            except Exception:
                log.exception("mark_species_checked failed for capture %d", c.id)
            return False, False

    def _run(self) -> None:
        next_sweep = time.monotonic()  # sweep backlog left pending by a restart on the first tick
        while not self._stop.is_set():
            try:
                ev = self._queue.get(timeout=0.5)
            except queue.Empty:
                ev = None
            if ev is not None:
                try:
                    current = self._store.get(ev.capture_id)
                    if current is None or current.status != "pending":
                        log.debug("capture %d no longer pending; skipping", ev.capture_id)
                    else:
                        self.process(ev.capture_id, ev.path)
                except Exception:
                    log.exception("processing capture %d failed", ev.capture_id)
            if time.monotonic() >= next_sweep:
                try:
                    self.sweep()
                except Exception:
                    log.exception("sweep failed")
                next_sweep = time.monotonic() + self._sweep_interval
