"""Queue between capture and classification, plus a backlog sweep and the visit tick.

The worker thread classifies queued events, sweeps the backlog every `sweep_interval`, and calls
`Visits.tick()` every `tick_interval` (also between backlog rows) so quiet visits close and alert."""

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
from birdy.visits import Visits

if TYPE_CHECKING:
    from birdy.notify import TelegramNotifier
    from birdy.publish import Publisher

log = logging.getLogger(__name__)

BACKLOG_PER_SWEEP = 50
TICK_INTERVAL = 5.0     # seconds between visit close/alert checks


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
        visits: Visits | None = None,
        lores_size: tuple[int, int] = (320, 240),
        maxsize: int = 50,
        sweep_interval: float = 300.0,
        tick_interval: float = TICK_INTERVAL,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._live = live
        self._notifier = notifier      # kept for inspection; Visits alerts and publishes
        self._publisher = publisher
        # No Visits given: model-species majority voting (no bank, no embeddings needed).
        self.visits = visits if visits is not None else Visits(
            store, live, bank=None, species_list=None, notifier=notifier, publisher=publisher
        )
        # The process's pinned lores size (camera keys are restart-only): motion boxes and the
        # ROI are in this frame.
        self._lores_size = lores_size
        self._queue: queue.Queue[MotionEvent] = queue.Queue(maxsize=maxsize)
        self._sweep_interval = sweep_interval
        self._tick_interval = tick_interval
        self._next_tick = time.monotonic()
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
            self._store.store_verdict(
                capture_id, is_bird=pred.is_bird, confidence=pred.confidence,
                species=pred.species, species_confidence=pred.species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box, embedding=pred.embedding,
            )
        except Exception:
            log.exception("storing classification failed for capture %d", capture_id)
            try:
                self._store.mark_error(capture_id)
            except Exception:
                log.exception("mark_error failed for capture %d; left pending", capture_id)
            return
        # Visits writes the final columns (the visit's label), re-ranks publishing, and alerts
        # when the visit closes (see `tick`). A failure leaves the row for the unlabelled sweep.
        final = None
        try:
            self.visits.photo_classified(self._store.get(capture_id), live=True)
            final = self._store.get(capture_id)
        except Exception:
            log.exception("visit update failed for capture %d", capture_id)
        if final is not None and final.species:
            log.info("capture %d: bird=%s p=%.2f species=%s", capture_id, pred.is_bird, pred.confidence, final.species)
        else:
            log.info("capture %d: bird=%s p=%.2f", capture_id, pred.is_bird, pred.confidence)

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
        processed = with_species = promoted = renamed = 0
        for c in self._store.classification_backlog(limit=BACKLOG_PER_SWEEP):
            if self._stop.is_set() or self._queue.qsize():
                break
            got_species, was_promoted, was_renamed = self._backfill(c)
            self._maybe_tick()          # a long sweep must not hold back visit closes and alerts
            processed += 1
            with_species += got_species
            promoted += was_promoted
            renamed += was_renamed
        if processed:
            remaining = self._store.count_classification_backlog()
            log.info(
                "backfill: %d rows, %d with species, %d promoted, %d renamed, %d remaining",
                processed, with_species, promoted, renamed, remaining,
            )
        # Rows with a verdict but no final label (Visits failed or the process died in between):
        # resolve them now. No model runs; bounded and yields to live events like the backfill.
        resolved = 0
        for c in self._store.unresolved_done(limit=BACKLOG_PER_SWEEP):
            if self._stop.is_set() or self._queue.qsize():
                break
            try:
                # live=True: these rows came from the live path; the LATE rule keeps old ones from alerting.
                self.visits.photo_classified(c, live=True)
                resolved += 1
            except Exception:
                log.exception("resolving unlabelled capture %d failed", c.id)
            self._maybe_tick()
        if resolved:
            log.info("resolved %d unlabelled rows", resolved)

    def _backfill(self, c: Capture) -> tuple[bool, bool, bool]:
        """Re-classify an already-classified row; returns (got_species, promoted, renamed)."""
        cfg = self._live.get()
        path = self._store.photos_dir / c.path
        try:
            crop = crop_box(c.motion_box, cfg.motion.roi, self._lores_size)
            # known_bird: a stored bird (final or model verdict) always gets a species run, so a
            # stricter gate can't strip its name.
            known = bool(c.is_bird or c.model_is_bird)
            pred = self._classifier.classify(
                path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad, known_bird=known
            )
            # Bird/non-bird: never demote; promote when the new verdict says bird (gate or
            # detector). Visits decides membership from model_is_bird, so the merged verdict is
            # what gets stored. Confidence comes from the new verdict when it says bird, else the
            # old one is kept. det_* always records the latest detector run.
            is_bird = known or pred.is_bird
            confidence = pred.confidence if pred.is_bird else (c.model_confidence or c.confidence or 0.0)
            if c.model_is_bird is None:     # a row from before the verdict/final split: final = model
                species, species_confidence = c.species, c.species_confidence
            else:
                species, species_confidence = c.model_species, c.model_species_confidence
            # Species: a row the species model hasn't seen with the current model (species_checked = 0)
            # takes the new verdict, even "no species" (its old name came from another model); a seen
            # row keeps its label (the crop may have changed with the ROI; rewriting would move publish
            # slots for nothing). A promoted non-bird gets one like a new row.
            write_species = (not c.species_checked) or ((not known) and pred.is_bird)
            if write_species:
                species, species_confidence = pred.species, pred.species_confidence
            # The embedding drives the visit vote: a detector-only pass on a seen row keeps the
            # stored one (None = keep), so it can't move the visit's label.
            embedding = pred.embedding if write_species or not c.has_embedding else None
            self._store.store_verdict(
                c.id, is_bird=is_bird, confidence=confidence, species=species, species_confidence=species_confidence,
                det_confidence=pred.detector_confidence, det_box=pred.detector_box, embedding=embedding,
            )
        except Exception:
            # Unlike live processing, a failure here must not flip a `done` row to `error`.
            log.exception("backfill failed for capture %d; marking checked", c.id)
            try:
                self._store.mark_checked(c.id)
            except Exception:
                log.exception("mark_checked failed for capture %d", c.id)
            return False, False, False
        # Visits writes the final columns and re-ranks publishing for every species change (old and
        # new day; a renamed `auto` row loses its slot). Historical visits never alert.
        try:
            changes = self.visits.photo_classified(self._store.get(c.id), live=False)
        except Exception:
            # The row keeps its previous final columns; the next resolve of its visit heals it.
            log.exception("visit update failed for capture %d", c.id)
            changes = []
        after = self._store.get(c.id)
        got_species = after is not None and bool(after.species)
        promoted = (not c.is_bird) and after is not None and bool(after.is_bird)
        renamed = any(ch.capture_id == c.id and ch.old_species for ch in changes)
        return got_species, promoted, renamed

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
            self._maybe_tick()

    def _maybe_tick(self) -> None:
        """Close quiet visits and alert, at most once per `tick_interval`."""
        if time.monotonic() < self._next_tick:
            return
        try:
            self.visits.tick()
        except Exception:
            log.exception("visit tick failed")
        self._next_tick = time.monotonic() + self._tick_interval
