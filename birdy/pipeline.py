"""Queue between capture and classification, plus a backlog sweep."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from birdy.classifier import Classifier
from birdy.live import LiveConfig
from birdy.motion import NormBox, crop_box
from birdy.store import Capture, Store

if TYPE_CHECKING:
    from birdy.notify import TelegramNotifier
    from birdy.publish import Publisher

log = logging.getLogger(__name__)

BACKLOG_PER_SWEEP = 50


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
        notifier: "TelegramNotifier | None" = None,
        publisher: "Publisher | None" = None,
        lores_size: tuple[int, int] = (320, 240),
        maxsize: int = 50,
        sweep_interval: float = 300.0,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._live = live
        self._notifier = notifier
        self._publisher = publisher
        # The process's pinned lores size (camera keys are restart-only): motion boxes and the
        # ROI are in this frame.
        self._lores_size = lores_size
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

    def process(self, capture_id: int, path: Path, captured_at: datetime, motion_box: NormBox | None = None) -> None:
        cfg = self._live.get()
        try:
            # Inside the try: a bad stored box marks the row 'error' instead of retrying it forever.
            crop = crop_box(motion_box, cfg.motion.roi, self._lores_size)
            pred = self._classifier.classify(path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad)
        except Exception:
            log.exception("classification failed for %s", path)
            try:
                self._store.mark_error(capture_id)
            except Exception:
                log.exception("mark_error failed for capture %d; left pending", capture_id)
            return
        try:
            self._store.update_classification(
                capture_id, pred.is_bird, pred.confidence, pred.species, pred.species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box,
            )
        except Exception:
            log.exception("storing classification failed for capture %d", capture_id)
            try:
                self._store.mark_error(capture_id)
            except Exception:
                log.exception("mark_error failed for capture %d; left pending", capture_id)
            return
        log.info("capture %d: bird=%s p=%.2f", capture_id, pred.is_bird, pred.confidence)
        if self._notifier is not None:
            try:
                reason = self._notifier.consider(capture_id, path, pred, captured_at)
                if reason:
                    log.info("alert queued for capture %d (%s)", capture_id, reason)
            except Exception:
                log.exception("alert decision failed for capture %d", capture_id)
        if self._publisher is not None and pred.is_bird and pred.species:
            try:
                self._publisher.evaluate(self._store.get(capture_id))
            except Exception:
                log.exception("publish decision failed for capture %d", capture_id)

    def sweep(self) -> None:
        for c in self._store.pending():
            if self._stop.is_set():
                break
            current = self._store.get(c.id)
            if current is None or current.status != "pending":
                continue
            self.process(c.id, self._store.photos_dir / c.path, c.captured_at, current.motion_box)
        # Rows classified before species recognition or the detector existed: re-run the
        # classifier so they get a species and a second look at bird/not-bird. Bounded per
        # sweep to keep the Pi responsive, and yields to live events.
        processed = with_species = promoted = 0
        for c in self._store.classification_backlog(limit=BACKLOG_PER_SWEEP):
            if self._stop.is_set() or self._queue.qsize():
                break
            got_species, was_promoted = self._backfill(c)
            processed += 1
            with_species += got_species
            promoted += was_promoted
        if processed:
            remaining = self._store.count_classification_backlog()
            log.info("backfill: %d rows, %d with species, %d promoted, %d remaining", processed, with_species, promoted, remaining)

    def _backfill(self, c: Capture) -> tuple[bool, bool]:
        """Re-classify an already-classified row; returns (got_species, promoted)."""
        cfg = self._live.get()
        path = self._store.photos_dir / c.path
        try:
            crop = crop_box(c.motion_box, cfg.motion.roi, self._lores_size)
            pred = self._classifier.classify(path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad)
            # Bird/non-bird: never demote; promote when the new verdict says bird (gate
            # or detector). Confidence comes from the new verdict when it says bird, else
            # the old one is kept. det_* always records the latest detector run.
            promoted = (not c.is_bird) and pred.is_bird
            is_bird = bool(c.is_bird) or pred.is_bird
            confidence = pred.confidence if pred.is_bird else c.confidence
            species, species_confidence = c.species, c.species_confidence
            # Species: a row the species model hasn't seen with the current model (species_checked = 0)
            # takes the new verdict, even "no species" (its old name came from another model); a seen
            # row keeps its label (the crop may have changed with the ROI; rewriting would move publish
            # slots for nothing). A promoted non-bird gets one like a new row.
            write_species = (not c.species_checked) or promoted
            if write_species:
                species, species_confidence = pred.species, pred.species_confidence
            self._store.update_classification(
                c.id, is_bird, confidence, species, species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box,
            )
        except Exception:
            # Unlike live processing, a failure here must not flip a `done` row to `error`.
            log.exception("backfill failed for capture %d; marking checked", c.id)
            try:
                self._store.mark_checked(c.id)
            except Exception:
                log.exception("mark_checked failed for capture %d", c.id)
            return False, False
        # Historical photos may be published too (unlike alerts, which are for live events only).
        # `c` is the snapshot from before the write: forgetting it re-ranks the old species/day,
        # where this row (now carrying the new species) is no longer a candidate.
        renamed = write_species and bool(c.species) and c.species != species
        if self._publisher is not None and renamed:
            try:
                self._publisher.forget(c)          # the old species/day loses this row's slot
            except Exception:
                log.exception("publish forget failed for capture %d", c.id)
        got_species = write_species and bool(pred.species)
        if self._publisher is not None and got_species:
            try:
                self._publisher.evaluate(self._store.get(c.id))
            except Exception:
                log.exception("publish decision failed for capture %d", c.id)
        return got_species, promoted

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
                        self.process(ev.capture_id, ev.path, ev.captured_at, current.motion_box)
                except Exception:
                    log.exception("processing capture %d failed", ev.capture_id)
            if time.monotonic() >= next_sweep:
                try:
                    self.sweep()
                except Exception:
                    log.exception("sweep failed")
                next_sweep = time.monotonic() + self._sweep_interval
