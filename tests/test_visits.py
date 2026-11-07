from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from birdy.species import Bank, load_bank
from birdy.store import Store
from birdy.visits import ALERT_AFTER, KEEP_OWN, Visits, final_label, vote

T0 = datetime(2026, 11, 7, 9, 0, tzinfo=timezone.utc)
D = 4


def bank(tmp_path) -> Bank:
    # unit vectors: row 0 = Black Redstart (e1), row 1 = Robin (e2), row 2 = Hoopoe (e3)
    p = tmp_path / "bank.npz"
    np.savez(p, ids=np.array([1, 2, 3], np.int32), latin=np.array(["Phoenicurus ochruros", "Erithacus rubecula", "Upupa epops"]),
             common=np.array(["Black Redstart", "European Robin", "Eurasian Hoopoe"]), emb=np.eye(3, D, dtype=np.float16),
             logit_scale=np.float32(100.0), model=np.str_("x"), prompt=np.str_("p"))
    return load_bank(p)


def e(i, noise=0.0):
    v = np.eye(D, dtype=np.float32)[i] + noise
    return v / np.linalg.norm(v)


class FakeList:
    version = 1
    def candidates(self, cfg): return None          # whole bank


class FakeNotifier:
    def __init__(self): self.calls = []
    def consider(self, capture_id, photo_path, species, species_confidence, captured_at, first_seen):
        self.calls.append((capture_id, species, first_seen)); return "watchlist"


class FakePublisher:
    def __init__(self): self.days = []
    def evaluate_day(self, species, when): self.days.append((species, when.date()))


@pytest.fixture
def store(tmp_path): return Store(tmp_path / "data")


@pytest.fixture
def visits(store, live, tmp_path):
    return Visits(store, live, bank=bank(tmp_path), species_list=FakeList(), notifier=FakeNotifier(), publisher=FakePublisher())


def add(store, seconds, *, bird=True, species=None, p=None, emb=None, live_verdict=True):
    when = T0 + timedelta(seconds=seconds)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.store_verdict(cid, is_bird=bird, confidence=0.9 if bird else 0.1, species=species, species_confidence=p, embedding=emb)
    return cid


# --- pure functions -------------------------------------------------------

def test_vote_pools_embeddings(tmp_path):
    b = bank(tmp_path); cand = np.arange(3)
    out = vote([e(0), e(0, 0.1), e(1)], b, cand, threshold=0.5)
    assert out is not None and out.species == "Black Redstart" and out.p > 0.5
    assert vote([], b, cand, threshold=0.5) is None
    assert vote([e(0) * 0 + np.full(D, 0.5)], b, cand, threshold=0.99) is None     # flat -> below threshold


def test_final_label_priority():
    assert final_label(manual_is_bird=False, manual_species=None, own=("Robin", 0.9), vote_species="Hoopoe") == (False, None)
    assert final_label(manual_is_bird=True, manual_species="Hoopoe", own=("Robin", 0.9), vote_species="Robin") == (True, "Hoopoe")
    assert final_label(manual_is_bird=True, manual_species=None, own=("Robin", 0.9), vote_species="Robin") == (True, None)
    assert final_label(manual_is_bird=None, manual_species=None, own=("Robin", KEEP_OWN), vote_species="Hoopoe") == (True, "Robin")
    assert final_label(manual_is_bird=None, manual_species=None, own=("Robin", 0.5), vote_species="Hoopoe") == (True, "Hoopoe")
    assert final_label(manual_is_bird=None, manual_species=None, own=None, vote_species=None) == (True, None)


# --- grouping ---------------------------------------------------------------

def test_new_join_and_gap(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 30, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    c = add(store, 200, emb=e(1)); visits.photo_classified(store.get(c), live=True)
    assert store.get(a).visit_id == store.get(b).visit_id != store.get(c).visit_id
    assert store.get(a).species == "Black Redstart" and store.get(b).species == "Black Redstart" and store.get(c).species == "European Robin"
    assert store.get_visit(store.get(a).visit_id).alert == "pending"


def test_backfill_visits_never_alert(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=False)
    assert store.get_visit(store.get(a).visit_id).alert == "skipped"


def test_merge_bridging_photo(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    c = add(store, 100, emb=e(0)); visits.photo_classified(store.get(c), live=True)
    assert store.get(a).visit_id != store.get(c).visit_id
    store.set_visit_alert(store.get(c).visit_id, "sent")
    b = add(store, 50, emb=e(0)); visits.photo_classified(store.get(b), live=True)   # within 60 s of both
    vid = store.get(a).visit_id
    assert store.get(b).visit_id == vid == store.get(c).visit_id and store.get_visit(vid).alert == "sent"
    assert len(store.visit_photos(vid)) == 3


def test_not_bird_visit_is_not_joined(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "not_bird")
    b = add(store, 20, emb=e(1)); visits.photo_classified(store.get(b), live=True)
    assert store.get(b).visit_id != store.get(a).visit_id and store.get(b).manual_at is None


def test_joining_photo_inherits_correction(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "species:Eurasian Hoopoe")
    b = add(store, 20, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    assert store.get(b).visit_id == store.get(a).visit_id and store.get(b).species == "Eurasian Hoopoe" and store.get(b).manual_species == "Eurasian Hoopoe"


def test_non_bird_gets_no_visit_and_resolve_single(store, visits):
    a = add(store, 0, bird=False, species="Robin", p=0.9); visits.photo_classified(store.get(a), live=True)
    c = store.get(a)
    assert c.visit_id is None and c.is_bird is False and c.species is None and c.model_species == "Robin"


# --- resolve ---------------------------------------------------------------

def test_keep_own_second_bird(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    c = add(store, 20, emb=e(1)); visits.photo_classified(store.get(c), live=True)    # own p ~1.0 for Robin
    assert store.get(c).species == "European Robin" and store.get(a).species == "Black Redstart"


def test_resolve_moves_publish_slots(store, visits):
    pub = visits._publisher
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    store.set_published(a, "auto")
    visits.correct(a, "species:European Robin")
    assert store.get(a).published is None and store.get(a).species == "European Robin"
    assert ("Black Redstart", T0.date()) in pub.days and ("European Robin", T0.date()) in pub.days


def test_species_confidence_is_own_p_or_model_confidence_when_corrected(store, visits):
    a = add(store, 0, emb=e(0, 10.0)); visits.photo_classified(store.get(a), live=True)   # own p ~0.985 (logit_scale 100 saturates small noise to 1.0)
    assert 0.5 < store.get(a).species_confidence < 1.0
    visits.correct(a, "species:Eurasian Hoopoe")
    assert store.get(a).species_confidence == store.get(a).model_confidence


# --- tick / alerts -----------------------------------------------------------

def test_tick_closes_and_alerts_once(store, visits):
    n = visits._notifier
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(0, 0.3)); visits.photo_classified(store.get(b), live=True)
    vid = store.get(a).visit_id
    visits.tick(T0 + timedelta(seconds=30))
    assert store.get_visit(vid).closed is False and n.calls == []
    visits.tick(T0 + timedelta(seconds=71))
    v = store.get_visit(vid)
    assert v.closed is True and v.alert == "sent" and n.calls == [(a, "Black Redstart", True)]   # best photo = highest own p
    visits.tick(T0 + timedelta(seconds=200))
    assert len(n.calls) == 1


def test_tick_alerts_long_open_visit_early(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    vid = store.get(a).visit_id
    for s in range(50, ALERT_AFTER + 60, 50):
        store.attach_to_visit(add(store, s, emb=e(0)), vid, extend=True)
    visits.tick(T0 + timedelta(seconds=ALERT_AFTER + 1))
    v = store.get_visit(vid)
    assert v.closed is False and v.alert == "sent"


def test_tick_skips_late_and_unnamed(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.tick(T0 + timedelta(minutes=20))
    assert store.get_visit(store.get(a).visit_id).alert == "skipped" and visits._notifier.calls == []
    b = add(store, 3600, emb=np.full(D, 0.5) / np.linalg.norm(np.full(D, 0.5))); visits.photo_classified(store.get(b), live=True)
    visits.tick(T0 + timedelta(seconds=3600 + 61))
    assert store.get_visit(store.get(b).visit_id).alert == "skipped"


def test_first_seen_is_first_visit(store, visits):
    n = visits._notifier
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.tick(T0 + timedelta(seconds=61))
    b = add(store, 600, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    visits.tick(T0 + timedelta(seconds=661))
    assert [c[2] for c in n.calls] == [True, False]


# --- corrections ---------------------------------------------------------------

def test_correct_labels(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(0)); visits.photo_classified(store.get(b), live=True)
    visits.correct(a, "not_bird")
    assert store.get(b).is_bird is False and store.get(b).species is None and store.get(b).manual_is_bird is False
    visits.correct(b, "bird_unknown")
    assert store.get(a).is_bird is True and store.get(a).species is None and store.get(a).manual_is_bird is True
    visits.correct(a, "species:phoenicurus ochruros")
    assert store.get(b).species == "Black Redstart" and store.get(b).manual_species == "Black Redstart"
    visits.correct(a, "clear")
    assert store.get(a).manual_at is None and store.get(b).species == "Black Redstart"
    with pytest.raises(ValueError):
        visits.correct(a, "species:Dodo")
    with pytest.raises(ValueError):
        visits.correct(a, "nonsense")


def test_correct_non_bird_to_species_creates_visit_without_alert(store, visits):
    a = add(store, 0, bird=False); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "species:European Robin")
    c = store.get(a)
    assert c.visit_id is not None and c.is_bird is True and c.species == "European Robin"
    assert store.get_visit(c.visit_id).alert == "skipped" and visits._notifier.calls == []


def test_corrections_never_alert(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    visits.correct(a, "species:European Robin")
    visits.tick(T0 + timedelta(seconds=61))
    assert visits._notifier.calls == [(a, "European Robin", True)]   # the close alert, with the corrected name, once


# --- deletes -----------------------------------------------------------------

def test_photo_removed_reresolves_or_drops_visit(store, visits):
    a = add(store, 0, emb=e(0)); visits.photo_classified(store.get(a), live=True)
    b = add(store, 10, emb=e(1)); visits.photo_classified(store.get(b), live=True)
    vid = store.get(a).visit_id
    store.delete_capture(b); visits.photo_removed(vid)
    assert store.get_visit(vid) is not None
    store.delete_capture(a); visits.photo_removed(vid)
    assert store.get_visit(vid) is None and store.count_visits_with_species("Black Redstart") == 0


# --- without embeddings (migration window) ----------------------------------

def test_vote_falls_back_to_model_species_majority(store, visits):
    a = add(store, 0, species="Robin", p=0.7); visits.photo_classified(store.get(a), live=False)
    b = add(store, 10, species="Robin", p=0.6); visits.photo_classified(store.get(b), live=False)
    c = add(store, 20, species="Hoopoe", p=0.75); visits.photo_classified(store.get(c), live=False)   # most confident, below KEEP_OWN
    assert store.get(c).species == "Robin" and store.get_visit(store.get(a).visit_id).species == "Robin"
