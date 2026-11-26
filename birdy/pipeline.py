"""Queue between capture and classification, plus a backlog sweep and the visit tick.

The worker thread classifies queued events, sweeps the backlog every `sweep_interval`, and calls
`Visits.tick()` every `tick_interval` (also between backlog rows) so quiet visits close and alert."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from birdy.classifier import Classifier
from birdy.dslr import (
    EXPOSURE_K_DEFAULT, DslrError, adapt_exposure_k, box_in_frame, dslr_rel, frame_brightness, pick_shutter,
    region_crop, search_tiles, species_crop,
)
from birdy.live import LiveConfig
from birdy.motion import NormBox, crop_box
from birdy.store import Capture, Store
from birdy.visits import Visits

if TYPE_CHECKING:
    from birdy.dslr import Dslr
    from birdy.notify import TelegramNotifier
    from birdy.publish import Publisher

log = logging.getLogger(__name__)

BACKLOG_PER_SWEEP = 50
TICK_INTERVAL = 5.0     # seconds between visit close/alert checks
FRESH_SECONDS = 10.0   # the DSLR only fires for a Pi photo this recent: never for the backlog or after a restart


@dataclass(frozen=True)
class MotionEvent:
    capture_id: int
    path: Path
    captured_at: datetime
    motion_score: int
    manual: bool = False    # "Take a photo" on the web page: the extra camera fires whatever the verdict
    lux: float | None = None   # the Pi camera's light reading for this still (None = unknown)


class Pipeline:
    def __init__(
        self,
        store: Store,
        classifier: Classifier,
        live: LiveConfig,
        notifier: "TelegramNotifier | None" = None,
        publisher: "Publisher | None" = None,
        visits: Visits | None = None,
        dslr: "Dslr | None" = None,
        lores_size: tuple[int, int] = (320, 240),
        maxsize: int = 50,
        sweep_interval: float = 300.0,
        tick_interval: float = TICK_INTERVAL,
    ) -> None:
        self._store = store
        self._classifier = classifier
        self._live = live
        self._dslr = dslr
        # Visits alerts and publishes; `notifier`/`publisher` only feed the default one built here
        # when none is given (model-species majority voting: no bank, no embeddings needed).
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

    def process(
        self, capture_id: int, path: Path, captured_at: datetime, motion_box: NormBox | None = None, manual: bool = False,
        lux: float | None = None,
    ) -> None:
        cfg = self._live.get()
        dslr_path, dslr_box = None, None
        try:
            # Inside the try: a bad stored box marks the row 'error' instead of retrying it forever.
            crop = crop_box(motion_box, cfg.motion.roi, self._lores_size)
            if self._dslr is None or not cfg.dslr.enabled:   # the switch is live
                pred = self._classifier.classify(path, crop.bbox, threshold=cfg.classifier.threshold, pad=crop.pad)
            else:
                pred, dslr_path, dslr_box = self._classify_with_dslr(capture_id, path, captured_at, crop, cfg, manual, lux)
        except Exception:
            log.exception("classification failed for %s", path)
            self._drop_dslr_frame(path)
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
                dslr_path=dslr_path, dslr_box=dslr_box, dslr_main=manual and dslr_path is not None,
            )
        except Exception:
            log.exception("storing classification failed for capture %d", capture_id)
            self._drop_dslr_frame(path)
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

    def _drop_dslr_frame(self, path: Path) -> None:
        """Remove the DSLR frame taken for this Pi photo when its row will not record it (the
        capture is being marked 'error'): no row would point at the file, so nothing could purge it."""
        if self._dslr is None:
            return
        try:
            (self._store.photos_dir / dslr_rel(path.relative_to(self._store.photos_dir))).unlink(missing_ok=True)
        except (OSError, ValueError):
            log.exception("could not remove the extra-camera frame of %s", path)

    def _classify_with_dslr(
        self, capture_id: int, path: Path, captured_at: datetime, crop, cfg, manual: bool = False, lux: float | None = None
    ):
        """Verdict on the Pi crop; for a fresh bird, one DSLR frame; the species from that frame
        when the detector finds the bird in it, else from the Pi crop. A photo taken by hand
        (`manual`) fires the DSLR whatever the verdict; its frame is the capture's main photo even with no
        bird in it (`process` stores `dslr_main`).
        Returns (prediction, DSLR path relative to photos/ or None, bird box in the DSLR frame or None)."""
        clf = self._classifier
        crop_rgb = clf.load_crop(path, crop.bbox, crop.pad)
        v = clf.verdict(crop_rgb, cfg.classifier.threshold)
        if not (v.is_bird or manual):
            return clf.prediction(v, [], None), None, None
        rel, frame, box = None, None, None
        if (datetime.now(timezone.utc) - captured_at).total_seconds() <= FRESH_SECONDS:
            rel = dslr_rel(path.relative_to(self._store.photos_dir))
            frame, took = self._dslr_frame(capture_id, rel, cfg, lux)
        if not v.is_bird:
            if frame is not None:
                log.info("capture %d: extra camera %.2f s (by hand, not a bird)", capture_id, took)
            return clf.prediction(v, [], None), (str(rel) if frame is not None else None), None
        species_rgb = crop_rgb
        if frame is not None:
            score, box = self._find_bird(frame, cfg.dslr.region, cfg.detector.threshold)
            if box is not None:
                species_rgb = species_crop(frame, box)
                log.info("capture %d: extra camera %.2f s, bird %.2f", capture_id, took, score)
            else:
                log.info("capture %d: extra camera %.2f s, no bird in frame", capture_id, took)
        top, emb = clf.name(species_rgb)
        return clf.prediction(v, top, emb), (str(rel) if frame is not None else None), box

    def _find_bird(self, frame: np.ndarray, region, threshold: float):
        """The bird in the DSLR frame: (score, box in fractions of the frame), or (0, None).
        The whole region first; when that finds nothing, overlapping tiles of it (a small bird
        vanishes when a large region is squeezed into the detector's 320 px), best tile wins."""
        clf = self._classifier
        det = clf.locate(region_crop(frame, region))
        if det is not None and det.score >= threshold:
            return det.score, box_in_frame(det.box, region)
        best, best_box = 0.0, None
        for tile in search_tiles(frame.shape, region):
            det = clf.locate(region_crop(frame, tile))
            if det is not None and det.score >= threshold and det.score > best:
                best, best_box = det.score, box_in_frame(det.box, tile)
        return best, best_box

    def _exposure_k(self) -> float:
        """The learned exposure constant (lux x seconds), kept in the meta table across restarts."""
        try:
            return float(self._store.get_meta("dslr_exposure_k") or EXPOSURE_K_DEFAULT)
        except ValueError:
            return EXPOSURE_K_DEFAULT

    def _dslr_frame(self, capture_id: int, rel: Path, cfg=None, lux: float | None = None) -> tuple[np.ndarray | None, float]:
        """Fire the DSLR; the decoded RGB frame and the seconds taken, or (None, 0) when it failed.
        With `dslr.auto_shutter` and a light reading, the shutter speed is picked first and the
        frame's brightness then corrects the exposure constant for the next shot."""
        dest = self._store.photos_dir / rel
        shutter, k = None, None
        if cfg is not None and cfg.dslr.auto_shutter and lux is not None:
            k = self._exposure_k()
            shutter = pick_shutter(lux, k)
        try:
            took = self._dslr.capture(dest, shutter) if shutter else self._dslr.capture(dest)
        except DslrError:
            return None, 0.0                  # Dslr logged it (once per distinct error)
        bgr = cv2.imread(str(dest))
        if bgr is None:
            dest.unlink(missing_ok=True)
            log.warning("capture %d: the extra-camera frame could not be decoded; using the Pi photo", capture_id)
            return None, 0.0
        if shutter is not None:
            brightness = frame_brightness(bgr, cfg.dslr.region)
            new_k = adapt_exposure_k(k, brightness, shutter)
            log.info("capture %d: extra camera at %s for %.0f lux, brightness %.0f (k %.2f -> %.2f)",
                     capture_id, shutter, lux, brightness, k, new_k)
            if new_k != k:
                try:
                    self._store.set_meta("dslr_exposure_k", f"{new_k:.4f}")
                except Exception:
                    log.exception("could not store the exposure constant")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), took

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
            if c.dslr_box is not None and hasattr(self._classifier, "verdict"):
                pred = self._reclassify_promoted(c, path, crop, cfg, known)
            else:
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

    def _reclassify_promoted(self, c: Capture, path: Path, crop, cfg, known: bool):
        """A row whose main photo is the DSLR frame: bird/non-bird is re-checked on the Pi crop,
        the species comes from the stored DSLR crop (the Pi crop when that file is unreadable)."""
        clf = self._classifier
        crop_rgb = clf.load_crop(path, crop.bbox, crop.pad)
        v = clf.verdict(crop_rgb, cfg.classifier.threshold)
        if not (v.is_bird or known):
            return clf.prediction(v, [], None)
        bgr = cv2.imread(str(self._store.photos_dir / c.dslr_path))
        img = crop_rgb if bgr is None else species_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), c.dslr_box)
        top, emb = clf.name(img)
        return clf.prediction(v, top, emb)

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
                        self.process(ev.capture_id, ev.path, ev.captured_at, current.motion_box, manual=ev.manual, lux=ev.lux)
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
