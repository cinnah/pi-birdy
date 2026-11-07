"""Visits: one bird's photos within `visits.gap_seconds`, one label, one alert, one correction.

All decisions here are arithmetic on stored embeddings; no model runs. One RLock serialises
assign/merge/resolve/close/correct across the pipeline and web threads."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from birdy.live import LiveConfig
from birdy.species import Bank, score_candidates
from birdy.store import Capture, Store, Visit

log = logging.getLogger(__name__)

KEEP_OWN = 0.8            # a photo keeps its own species over the vote at or above this (second bird in frame)
ALERT_AFTER = 300         # seconds a visit may stay open before it alerts with the vote so far
LATE = timedelta(minutes=10)   # a visit whose last photo is older than this at decision time never alerts


@dataclass(frozen=True)
class Vote:
    species: str
    p: float
    pooled: np.ndarray


@dataclass(frozen=True)
class Change:
    capture_id: int
    old_species: str | None
    new_species: str | None
    captured_at: datetime


def vote(embeddings: list[np.ndarray], bank: Bank, cand_rows: np.ndarray, threshold: float) -> Vote | None:
    """Pooled (mean, renormalised) embedding scored over the candidates; None below `threshold`."""
    if not embeddings:
        return None
    pooled = np.mean(np.stack(embeddings), axis=0)
    n = float(np.linalg.norm(pooled))
    if n == 0.0:
        return None
    pooled = pooled / n
    scored = score_candidates(bank, cand_rows, pooled)
    if not scored or scored[0][1] < threshold:
        return None
    row, p = scored[0]
    return Vote(bank.common[row] or bank.latin[row], p, pooled)


def final_label(*, manual_is_bird: bool | None, manual_species: str | None, own: tuple[str, float] | None,
                vote_species: str | None) -> tuple[bool, str | None]:
    """(is_bird, species) for one photo. Priority: correction > keep-own > vote > unnamed."""
    if manual_is_bird is False:
        return False, None
    if manual_is_bird is True:
        return True, manual_species
    if own is not None and vote_species is not None and own[0] != vote_species and own[1] >= KEEP_OWN:
        return True, own[0]
    return True, vote_species


class Visits:
    def __init__(self, store: Store, live: LiveConfig, bank: Bank | None, species_list, notifier=None, publisher=None) -> None:
        self._store, self._live, self._bank, self._list = store, live, bank, species_list
        self._notifier, self._publisher = notifier, publisher
        self._lock = threading.RLock()

    # --- candidates ---------------------------------------------------------
    def _cand_rows(self) -> np.ndarray:
        if self._bank is None:
            return np.array([], np.int64)
        ids = self._list.candidates(self._live.get().species) if self._list is not None else None
        if ids is None:
            return np.arange(len(self._bank.ids))
        return np.array([i for i, t in enumerate(self._bank.ids) if int(t) in ids], np.int64)

    def _name_of_row(self, row: int) -> str:
        return self._bank.common[row] or self._bank.latin[row]

    def _own(self, emb: np.ndarray | None, cand: np.ndarray) -> list[tuple[int, float]]:
        return [] if emb is None or self._bank is None else score_candidates(self._bank, cand, emb)

    # --- entry points ----------------------------------------------------------
    def photo_classified(self, c: Capture, live: bool) -> list[Change]:
        """After `store_verdict`: a bird joins/creates a visit and the visit is resolved; a non-bird
        is resolved alone (final = model, no species)."""
        with self._lock:
            if not self._is_member(c):
                if c.visit_id is None:
                    return self.resolve_single(c.id)
                return self.resolve(c.visit_id)
            vid = c.visit_id if c.visit_id is not None else self.assign(c, live=live)
            return self.resolve(vid)

    def _is_member(self, c: Capture) -> bool:
        if c.manual_is_bird is not None:
            return bool(c.manual_is_bird)
        return bool(c.model_is_bird) or bool(c.is_bird)

    def assign(self, c: Capture, live: bool) -> int:
        """Put a bird photo in a visit: a new one, the one within `gap`, or several merged into the lowest id."""
        with self._lock:
            gap = timedelta(seconds=self._live.get().visits.gap_seconds)
            found = [v for v in self._store.visits_overlapping(c.captured_at - gap, c.captured_at + gap)
                     if self._store.bird_members(v.id)]
            if not found:
                vid = self._store.create_visit(c.captured_at, c.captured_at, alert="pending" if live else "skipped")
            else:
                vid = found[0].id
                if len(found) > 1:
                    states = {v.alert for v in found}     # read before the merge deletes the others
                    self._store.merge_visits(vid, [v.id for v in found[1:]])
                    self._store.set_visit_alert(vid, "sent" if "sent" in states else "skipped" if "skipped" in states else "pending")
            self._store.attach_to_visit(c.id, vid, extend=True)
            # inherit the visit's correction, if any
            corrected = next((p for p in self._store.visit_photos(vid) if p.manual_at is not None and p.id != c.id), None)
            if corrected is not None:
                self._store.set_manual(c.id, is_bird=bool(corrected.manual_is_bird), species=corrected.manual_species, at=corrected.manual_at)
            return vid

    def resolve_single(self, capture_id: int) -> list[Change]:
        """A photo with no visit: final = its correction, else the model's verdict (species only for a bird)."""
        with self._lock:
            c = self._store.get(capture_id)
            if c is None:
                return []
            conf = c.model_confidence or 0.0
            if c.manual_is_bird is not None:
                is_bird = bool(c.manual_is_bird)
                species = c.manual_species if is_bird else None
                sp_conf = conf if species else None
            else:
                is_bird = bool(c.model_is_bird)
                species = c.model_species if is_bird else None
                sp_conf = c.model_species_confidence if species else None
            changes = []
            if c.species != species:
                changes.append(Change(c.id, c.species, species, c.captured_at))
                if c.published == "auto":
                    self._store.set_published(c.id, None)
            self._store.write_final(c.id, is_bird, conf, species, sp_conf)
            self._publish(changes)
            return changes

    def resolve(self, visit_id: int) -> list[Change]:
        with self._lock:
            photos = self._store.visit_photos(visit_id)
            if not photos:
                self._store.delete_visit(visit_id)
                return []
            cfg = self._live.get()
            cand = self._cand_rows()
            members = [p for p in photos if self._is_member(p)]
            embs = self._store.embeddings([p.id for p in members])
            v = vote([embs[p.id] for p in members if p.id in embs], self._bank, cand, cfg.species.threshold) if self._bank is not None else None
            if v is None and members and not embs:
                v = self._majority(members)
            vote_species, vote_p = (v.species, v.p) if v is not None else (None, None)
            changes: list[Change] = []
            for p in photos:
                own_scored = self._own(embs.get(p.id), cand)
                own = (self._name_of_row(own_scored[0][0]), own_scored[0][1]) if own_scored else (
                    (p.model_species, p.model_species_confidence or 0.0) if p.model_species else None)
                is_bird, species = final_label(manual_is_bird=p.manual_is_bird, manual_species=p.manual_species, own=own, vote_species=vote_species)
                if not is_bird:
                    conf = p.model_confidence or 0.0; sp_conf = None
                elif p.manual_is_bird is True and species is not None:
                    conf = p.model_confidence or 0.0; sp_conf = conf
                else:
                    conf = p.model_confidence or 0.0
                    sp_conf = self._p_for(own_scored, species, p) if species else None
                if (p.is_bird, p.species) != (is_bird, species) or p.species_confidence != sp_conf or p.confidence != conf:
                    if p.species != species:
                        changes.append(Change(p.id, p.species, species, p.captured_at))
                        if p.published == "auto":
                            self._store.set_published(p.id, None)
                    self._store.write_final(p.id, is_bird, conf, species, sp_conf)
            self._store.set_visit_result(visit_id, *self._visit_result(photos, vote_species, vote_p))
            self._publish(changes)
            return changes

    def _visit_result(self, photos: list[Capture], vote_species: str | None, vote_p: float | None) -> tuple[str | None, float | None]:
        """The visit's (species, p): a person's correction wins over the vote. A corrected species
        carries the best photo's model confidence (what the photos' species_confidence holds)."""
        corrected = next((p for p in photos if p.manual_is_bird is not None), None)
        if corrected is None:
            return vote_species, vote_p
        if not corrected.manual_is_bird or not corrected.manual_species:
            return None, None
        return corrected.manual_species, max((p.model_confidence or 0.0) for p in photos
                                             if p.manual_is_bird and p.manual_species == corrected.manual_species)

    def _majority(self, members: list[Capture]) -> Vote | None:
        """No embeddings yet (migration window): the most common model species, ties by confidence."""
        named = [p for p in members if p.model_species]
        if not named:
            return None
        counts = Counter(p.model_species for p in named)
        best = max(counts, key=lambda s: (counts[s], max(p.model_species_confidence or 0 for p in named if p.model_species == s)))
        return Vote(best, max(p.model_species_confidence or 0 for p in named if p.model_species == best), np.zeros(1))

    def _p_for(self, scored: list[tuple[int, float]], species: str, p: Capture) -> float | None:
        for row, prob in scored:
            if self._name_of_row(row) == species:
                return prob
        return p.model_species_confidence if p.model_species == species else None

    def _publish(self, changes: list[Change]) -> None:
        if self._publisher is None:
            return
        for ch in changes:
            for sp in (ch.old_species, ch.new_species):
                if sp:
                    try:
                        self._publisher.evaluate_day(sp, ch.captured_at)
                    except Exception:
                        log.exception("publish re-rank failed for %s", sp)

    # --- close / alert -------------------------------------------------------------
    def tick(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        with self._lock:
            gap = timedelta(seconds=self._live.get().visits.gap_seconds)
            for v in self._store.open_visits():
                if v.ended_at < now - gap:
                    self.resolve(v.id)
                    self._store.close_visit(v.id)
                    v = self._store.get_visit(v.id)
                    if v is not None and v.alert == "pending":
                        self._alert(v, now)
                elif v.alert == "pending" and now - v.started_at >= timedelta(seconds=ALERT_AFTER):
                    self.resolve(v.id)
                    v = self._store.get_visit(v.id)
                    if v is not None:
                        self._alert(v, now)

    def _alert(self, v: Visit, now: datetime) -> None:
        late = now - v.ended_at > LATE
        if late or not v.species:
            self._store.set_visit_alert(v.id, "skipped")
            log.info("visit %d: alert skipped (%s)", v.id, "late" if late else "no species")
            return
        best = self._best_photo(v)
        if best is None or self._notifier is None:
            self._store.set_visit_alert(v.id, "skipped")
            return
        first_seen = self._store.count_visits_with_species(v.species) == 1
        try:
            reason = self._notifier.consider(best.id, self._store.photos_dir / best.path, v.species,
                                             best.species_confidence or v.species_confidence or 0.0, best.captured_at, first_seen)
        except Exception:
            log.exception("alert decision failed for visit %d", v.id)
            reason = None
        self._store.set_visit_alert(v.id, "sent" if reason else "skipped")
        if reason:
            log.info("alert queued for visit %d (%s, %s)", v.id, v.species, reason)

    def _best_photo(self, v: Visit) -> Capture | None:
        photos = [p for p in self._store.visit_photos(v.id) if p.is_bird and p.species == v.species]
        return max(photos, key=lambda p: (p.species_confidence or 0.0, -p.id), default=None)

    # --- corrections -----------------------------------------------------------------
    def correct(self, capture_id: int, label: str) -> None:
        """`not_bird` | `bird_unknown` | `species:<name>` | `clear`; ValueError for anything else."""
        with self._lock:
            c = self._store.get(capture_id)
            if c is None:
                raise KeyError(capture_id)
            if label == "clear":
                if c.visit_id is None:
                    self._store.clear_manual_single(c.id); self.resolve_single(c.id)
                else:
                    self._store.clear_manual(c.visit_id); self.resolve(c.visit_id)
                return
            if label == "not_bird":
                is_bird, species = False, None
            elif label == "bird_unknown":
                is_bird, species = True, None
            elif label.startswith("species:"):
                name = label[len("species:"):].strip()
                tid = self._bank.resolve(name) if self._bank is not None else None
                if tid is None:
                    raise ValueError(f"unknown species {name!r}")
                row = int(np.nonzero(self._bank.ids == tid)[0][0])
                is_bird, species = True, self._name_of_row(row)
            else:
                raise ValueError(f"bad label {label!r}")
            now = datetime.now(timezone.utc)
            if is_bird and c.visit_id is None:
                self._store.set_manual(c.id, is_bird=True, species=species, at=now)
                vid = self.assign(self._store.get(c.id), live=False)
            else:
                vid = c.visit_id
            targets = [p.id for p in self._store.visit_photos(vid)] if vid is not None else [c.id]
            for pid in targets:
                self._store.set_manual(pid, is_bird=is_bird, species=species, at=now)
            if vid is not None:
                self.resolve(vid)
            else:
                self.resolve_single(c.id)
            log.info("capture %d: corrected to %s (%d photos)", capture_id, label, len(targets))

    # --- deletes ------------------------------------------------------------------------
    def photo_removed(self, visit_id: int | None) -> None:
        if visit_id is None:
            return
        with self._lock:
            self.resolve(visit_id)   # deletes the visit when empty
