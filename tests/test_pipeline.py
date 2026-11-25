import dataclasses
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import Prediction, Verdict
from birdy.detector import Detection
from birdy.dslr import DslrError
from birdy.motion import BBox, roi_bbox
from birdy.pipeline import MotionEvent, Pipeline
from birdy.publish import Publisher
from birdy.store import Store

# Local-time aware: the fail_on filename below depends on local HHMMSS.
T0 = datetime(2026, 10, 3, 10, 0).astimezone()


class FakeClassifier:
    def __init__(self, fail_on: set[str] = ()):
        self.fail_on = set(fail_on)
        self.calls: list[Path] = []
        self.bboxes: list = []
        self.thresholds: list = []
        self.pads: list = []

    def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
        self.calls.append(image_path)
        self.bboxes.append(roi_bbox)
        self.pads.append(pad)
        self.thresholds.append(threshold)
        if image_path.name in self.fail_on:
            raise RuntimeError("boom")
        return Prediction(is_bird=True, confidence=0.9)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def add(store, i):
    return add_at(store, T0.replace(minute=i))


def recent():
    """A whole-second time 2 min ago: its visit is quiet (gap 60 s) but not too late to alert."""
    return (datetime.now().astimezone() - timedelta(minutes=2)).replace(microsecond=0)


def add_at(store, when):
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    return MotionEvent(capture_id=cid, path=abs_path, captured_at=when, motion_score=100)


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_worker_classifies_submitted_event(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live)
    p.start()
    try:
        ev = add(store, 0)
        assert p.submit(ev) is True
        # is_bird, not status: the final columns are written by Visits a moment after the verdict
        assert wait_until(lambda: store.get(ev.capture_id).is_bird is not None)
        c = store.get(ev.capture_id)
        assert c.status == "done" and c.is_bird is True and c.confidence == 0.9
        assert clf.calls == [ev.path]
    finally:
        p.stop()


def test_classifier_error_marks_row_error_and_continues(store, live):
    p = Pipeline(store, FakeClassifier(fail_on={"100000_000.jpg"}), live)
    p.start()
    try:
        bad = add(store, 0)
        good = add(store, 1)
        p.submit(bad)
        p.submit(good)
        assert wait_until(lambda: store.get(good.capture_id).status == "done")
        assert store.get(bad.capture_id).status == "error"
    finally:
        p.stop()


def test_submit_returns_false_when_full(store, live):
    p = Pipeline(store, FakeClassifier(), live, maxsize=1)
    # not started: nothing drains the queue
    assert p.submit(add(store, 0)) is True
    assert p.submit(add(store, 1)) is False
    assert p.queue_depth() == 1


def test_sweep_classifies_pending_rows(store, live):
    p = Pipeline(store, FakeClassifier(), live)
    add(store, 0)
    add(store, 1)
    assert len(store.pending()) == 2
    p.sweep()
    assert store.pending() == []
    assert store.count("birds") == 2


def test_periodic_sweep_runs(store, live):
    p = Pipeline(store, FakeClassifier(), live, sweep_interval=0.1)
    add(store, 0)
    p.start()
    try:
        assert wait_until(lambda: store.pending() == [])
    finally:
        p.stop()


def test_store_error_does_not_kill_worker(store, live, monkeypatch):
    real_update = store.store_verdict
    calls = {"n": 0}

    def flaky_update(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db write failed")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(store, "store_verdict", flaky_update)
    p = Pipeline(store, FakeClassifier(), live)
    p.start()
    try:
        first = add(store, 0)
        second = add(store, 1)
        p.submit(first)
        p.submit(second)
        assert wait_until(lambda: store.get(second.capture_id).status == "done")
        assert store.get(first.capture_id).status == "error"
        assert p._thread.is_alive()
    finally:
        p.stop()


def test_queued_event_already_classified_is_skipped(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.1, None, None)
    p.submit(ev)
    p.start()
    try:
        # Once the worker has taken the event off the queue, stop() joins the
        # thread, so any processing of it has finished before we assert.
        assert wait_until(lambda: p.queue_depth() == 0)
    finally:
        p.stop()
    assert clf.calls == []
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.is_bird is False


def test_store_get_error_in_worker_does_not_kill_thread(store, live, monkeypatch):
    import threading

    real_get = store.get
    raised = {"done": False}

    def flaky_get(capture_id):
        # Only break the worker's lookup; the test's own polling uses get too.
        if threading.current_thread().name == "pipeline" and not raised["done"]:
            raised["done"] = True
            raise RuntimeError("db read failed")
        return real_get(capture_id)

    monkeypatch.setattr(store, "get", flaky_get)
    clf = FakeClassifier()
    p = Pipeline(store, clf, live)
    p.start()
    try:
        first = add(store, 0)
        p.submit(first)
        assert wait_until(lambda: raised["done"])
        second = add(store, 1)
        p.submit(second)
        assert wait_until(lambda: store.get(second.capture_id).status == "done")
        assert p._thread.is_alive()
    finally:
        p.stop()


class SlowClassifier(FakeClassifier):
    def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
        time.sleep(0.3)
        return super().classify(image_path, roi_bbox, threshold, pad)


def test_sweep_stops_promptly(store, live):
    for i in range(10):
        add(store, i)
    clf = SlowClassifier()
    p = Pipeline(store, clf, live)
    p.start()  # startup sweep begins after the first empty queue tick
    assert wait_until(lambda: len(clf.calls) >= 1)
    time.sleep(0.1)
    t = time.monotonic()
    p.stop(timeout=1.0)
    assert time.monotonic() - t < 1.0
    assert not p._thread.is_alive()
    assert 1 <= len(clf.calls) < 10


def test_roi_and_threshold_come_from_live_config(store, live, config):
    import dataclasses

    from birdy.motion import roi_bbox

    clf = FakeClassifier()
    p = Pipeline(store, clf, live)
    ev1 = add(store, 0)
    p.process(ev1.capture_id, ev1.path, ev1.captured_at)
    assert clf.bboxes[-1] == roi_bbox(config.motion.roi)
    assert clf.thresholds[-1] == 0.5

    new = dataclasses.replace(
        config,
        motion=dataclasses.replace(config.motion, roi=((0, 0), (10, 0), (10, 10))),
        classifier=dataclasses.replace(config.classifier, threshold=0.9),
    )
    live.replace(new)
    ev2 = add(store, 1)
    p.process(ev2.capture_id, ev2.path, ev2.captured_at)
    assert clf.bboxes[-1] == roi_bbox(new.motion.roi)
    assert clf.thresholds[-1] == 0.9


def unchecked(store, ids):
    with store._lock, store._conn:
        store._conn.execute(f"UPDATE captures SET species_checked = 0 WHERE id IN ({','.join('?' * len(ids))})", ids)


class SpeciesFake(FakeClassifier):
    def __init__(self, species="Robin", conf=0.8, **kwargs):
        super().__init__(**kwargs)
        self.species, self.conf = species, conf

    def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
        super().classify(image_path, roi_bbox, threshold, pad)
        return Prediction(is_bird=True, confidence=0.9, species=self.species, species_confidence=self.conf)


def test_sweep_drains_classification_backlog(store, live):
    evs = [add(store, i) for i in range(3)]
    for ev in evs:
        store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id for ev in evs])
    p = Pipeline(store, SpeciesFake(), live)
    p.sweep()
    assert store.classification_backlog() == []
    assert all(store.get(ev.capture_id).species == "Robin" for ev in evs)
    assert all(store.get(ev.capture_id).status == "done" for ev in evs)


def test_backlog_failure_is_marked_checked_not_error(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p = Pipeline(store, FakeClassifier(fail_on={ev.path.name}), live)
    p.sweep()
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.species_checked is True and c.species is None


def test_backlog_respects_stop(store, live):
    evs = [add(store, i) for i in range(2)]
    for ev in evs:
        store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id for ev in evs])
    p = Pipeline(store, SpeciesFake(), live)
    p._stop.set()
    p.sweep()
    assert len(store.classification_backlog()) == 2


def classified_unchecked(store, n, is_bird=True):
    evs = [add(store, i) for i in range(n)]
    for ev in evs:
        store.update_classification(ev.capture_id, is_bird, 0.9, None, None)
    unchecked(store, [ev.capture_id for ev in evs])
    return evs


def test_backlog_limit_per_sweep(store, live):
    from birdy.pipeline import BACKLOG_PER_SWEEP

    evs = classified_unchecked(store, BACKLOG_PER_SWEEP + 10)
    Pipeline(store, SpeciesFake(), live).sweep()
    left = store.classification_backlog(limit=1000)
    assert sorted(c.id for c in left) == sorted(ev.capture_id for ev in evs[:10])


class NotBirdFake(FakeClassifier):
    def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
        super().classify(image_path, roi_bbox, threshold, pad)
        return Prediction(is_bird=False, confidence=0.2)


def test_backlog_never_demotes(store, live):
    (ev,) = classified_unchecked(store, 1, is_bird=True)
    Pipeline(store, NotBirdFake(), live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.9 and c.species_checked is True


def test_backlog_rescues_non_bird(store, live):
    (ev,) = classified_unchecked(store, 1, is_bird=False)
    Pipeline(store, SpeciesFake(), live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.species == "Robin"


def test_backlog_yields_to_queue(store, live):
    classified_unchecked(store, 5)
    p = Pipeline(store, SpeciesFake(), live)
    live_ev = add(store, 30)  # pending row with a queued event
    assert p.submit(live_ev)
    p.sweep()  # the pending loop handles live_ev's row; the queue still holds its event
    assert store.get(live_ev.capture_id).status == "done"
    assert len(store.classification_backlog()) == 5


class FakeNotifier:
    def __init__(self, raise_exc=None):
        self.calls = []
        self.raise_exc = raise_exc

    def consider(self, capture_id, photo_path, species, species_confidence, captured_at, first_seen):
        self.calls.append((capture_id, photo_path, species, captured_at))
        if self.raise_exc:
            raise self.raise_exc
        return "watchlist"


def test_live_process_does_not_alert_until_the_visit_closes(store, live):
    n = FakeNotifier()
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert n.calls == []
    assert store.get(ev.capture_id).status == "done"
    p.visits.tick(T0 + timedelta(seconds=61))
    assert len(n.calls) == 1
    assert n.calls[0][0] == ev.capture_id and n.calls[0][2] == "Robin"


def test_notifier_exception_does_not_mark_error(store, live):
    p = Pipeline(store, SpeciesFake(), live, notifier=FakeNotifier(raise_exc=RuntimeError("boom")))
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    p.visits.tick(T0 + timedelta(seconds=61))
    assert store.get(ev.capture_id).status == "done"


def test_backfill_never_notifies(store, live):
    n = FakeNotifier()
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    p.sweep()
    p.visits.tick(T0 + timedelta(hours=1))
    assert store.get(ev.capture_id).species == "Robin" and n.calls == []


def test_queued_event_passes_captured_at(store, live):
    n = FakeNotifier()
    p = Pipeline(store, SpeciesFake(), live, notifier=n, tick_interval=0.1)
    p.start()
    try:
        ev = add_at(store, recent())
        p.submit(ev)
        assert wait_until(lambda: n.calls, timeout=3)     # the worker's tick closes the visit
        assert n.calls[0][3] == ev.captured_at
    finally:
        p.stop()


class FakePublisher:
    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.evaluated = []
        self.forgotten: list[int] = []
        self.order: list[str] = []
        self.days: list = []

    def evaluate(self, capture):
        self.evaluated.append(capture)
        self.order.append("evaluate")
        if self.raise_exc:
            raise self.raise_exc
        return True

    def forget(self, c):
        self.forgotten.append(c.id)
        self.order.append("forget")

    def evaluate_day(self, species, when):
        self.order.append(("day", species))
        self.days.append((species, when.astimezone().date()))
        if self.raise_exc:
            raise self.raise_exc
        return True


def test_process_calls_publisher_with_stored_capture(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert pub.days == [("Robin", T0.date())]
    c = store.get(ev.capture_id)
    assert c.species == "Robin" and c.status == "done"


def test_publisher_error_does_not_mark_row_error(store, live, caplog):
    pub = FakePublisher(raise_exc=RuntimeError("boom"))
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    with caplog.at_level(logging.ERROR):
        p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).status == "done"
    assert "publish re-rank failed" in caplog.text


def test_backfill_calls_publisher(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p.sweep()
    assert pub.days == [("Robin", T0.date())]
    assert store.get(ev.capture_id).species == "Robin"


def test_backfill_without_species_does_not_call_publisher(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(species=None, conf=None), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p.sweep()
    assert pub.days == []
    assert store.classification_backlog() == []


def test_process_skips_publisher_without_species(store, live):
    pub = FakePublisher()
    p = Pipeline(store, FakeClassifier(), live, publisher=pub)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert pub.days == []
    assert store.get(ev.capture_id).status == "done"


def test_backfill_publisher_error_keeps_row_done_and_checked(store, live, caplog):
    pub = FakePublisher(raise_exc=RuntimeError("boom"))
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    with caplog.at_level(logging.ERROR):
        p.sweep()
    assert "publish re-rank failed" in caplog.text
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.species_checked is True and c.species == "Robin"
    assert store.classification_backlog() == []


def test_failed_backfill_does_not_call_publisher(store, live):
    pub = FakePublisher()
    ev = add(store, 0)
    p = Pipeline(store, SpeciesFake(fail_on={ev.path.name}), live, publisher=pub)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p.sweep()
    assert pub.days == []
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.species_checked is True
    assert store.classification_backlog() == []


def add_with_box(store, i, box):
    when = T0.replace(minute=i)
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100, motion_box=box)
    return MotionEvent(capture_id=cid, path=abs_path, captured_at=when, motion_score=100)


def test_live_capture_is_classified_on_its_motion_crop(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    p.start()
    try:
        p.submit(add_with_box(store, 1, (100 / 320, 100 / 240, 20 / 320, 20 / 240)))
        assert wait_until(lambda: clf.calls)
    finally:
        p.stop()
    assert clf.bboxes[-1] == BBox(90, 90, 40, 40)  # 20x20 at (100,100) -> 40 px square
    assert clf.pads[-1] == 0.0


def test_capture_without_box_uses_roi_crop(store, live, config):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    ev = add(store, 2)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert clf.bboxes[-1] == roi_bbox(config.motion.roi)
    assert clf.pads[-1] == 0.1


def test_sweep_uses_the_stored_box(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    add_with_box(store, 3, (100 / 320, 100 / 240, 20 / 320, 20 / 240))  # pending, never submitted
    p.sweep()
    assert clf.bboxes[0] == BBox(90, 90, 40, 40)


def test_backfill_uses_the_stored_box(store, live):
    clf = FakeClassifier()
    p = Pipeline(store, clf, live, lores_size=(320, 240))
    ev = add_with_box(store, 4, (100 / 320, 100 / 240, 20 / 320, 20 / 240))
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p.sweep()
    assert clf.bboxes == [BBox(90, 90, 40, 40)] and clf.pads == [0.0]


def test_corrupt_stored_box_marks_error(store, live):
    p = Pipeline(store, FakeClassifier(), live, lores_size=(320, 240))
    # Inf is stored as is (NaN would be stored as NULL and read back as "no box").
    ev = add_with_box(store, 5, (float("inf"), 0.1, 0.1, 0.1))
    p.process(ev.capture_id, ev.path, ev.captured_at, store.get(ev.capture_id).motion_box)
    assert store.get(ev.capture_id).status == "error"


class DetectorFake(FakeClassifier):
    """What the real classifier returns for a bird the gate missed but the detector found."""

    def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
        super().classify(image_path, roi_bbox, threshold, pad)
        return Prediction(is_bird=True, confidence=0.55, detector_confidence=0.55, detector_box=(0.1, 0.2, 0.3, 0.4))


def test_process_stores_detector_fields(store, live):
    ev = add(store, 0)
    Pipeline(store, DetectorFake(), live).process(ev.capture_id, ev.path, ev.captured_at)
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.55
    assert c.det_confidence == 0.55 and c.det_box == (0.1, 0.2, 0.3, 0.4)


def detector_unchecked(store, ids):
    with store._lock, store._conn:
        store._conn.execute(f"UPDATE captures SET detector_checked = 0 WHERE id IN ({','.join('?' * len(ids))})", ids)


def test_backfill_promotes_non_bird_the_detector_finds(store, live, caplog):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.05, None, None)
    detector_unchecked(store, [ev.capture_id])
    with caplog.at_level(logging.INFO):
        Pipeline(store, DetectorFake(), live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.55 and c.det_box == (0.1, 0.2, 0.3, 0.4)
    assert store.classification_backlog() == []
    assert "1 rows, 0 with species, 1 promoted, 0 renamed" in caplog.text


def test_backfill_never_demotes_on_detector_pass(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    detector_unchecked(store, [ev.capture_id])
    Pipeline(store, NotBirdFake(), live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.confidence == 0.9 and store.classification_backlog() == []
    assert c.det_confidence == 0.0


def test_backfill_failure_marks_detector_checked(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.05, None, None)
    detector_unchecked(store, [ev.capture_id])
    Pipeline(store, FakeClassifier(fail_on={ev.path.name}), live).sweep()
    c = store.get(ev.capture_id)
    assert c.status == "done" and c.is_bird is False and store.classification_backlog() == []


class ReturnsFake(FakeClassifier):
    """Returns a fixed prediction."""

    def __init__(self, pred, **kw):
        super().__init__(**kw)
        self.pred = pred

    def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
        super().classify(image_path, roi_bbox, threshold, pad)
        return self.pred


def test_backfill_keeps_a_birds_species_when_the_rerun_has_none(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Robin", 0.8)
    detector_unchecked(store, [ev.capture_id])
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9)), live).sweep()
    c = store.get(ev.capture_id)
    assert c.species == "Robin" and c.species_confidence == 0.8
    assert store.classification_backlog() == []


def test_backfill_keeps_a_seen_rows_species_even_if_the_rerun_differs(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Robin", 0.8)  # species_checked = 1
    detector_unchecked(store, [ev.capture_id])
    pred = Prediction(is_bird=True, confidence=0.9, species="Blackbird", species_confidence=0.7)
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(pred), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.species == "Robin" and c.species_confidence == 0.8
    assert c.model_species == "Robin" and pub.days == []
    assert store.classification_backlog() == []


def test_backfill_writes_species_for_unseen_rows(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    pred = Prediction(is_bird=True, confidence=0.9, species="Robin", species_confidence=0.8)
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(pred), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.species == "Robin" and c.species_confidence == 0.8
    assert pub.days == [("Robin", T0.date())]
    assert store.classification_backlog() == []


def test_backfill_renames_a_known_bird_the_gate_now_rejects(store, live):
    class KnownBirdFake(ReturnsFake):
        def __init__(self, pred):
            super().__init__(pred)
            self.known = []

        def classify(self, image_path, roi_bbox, threshold=None, pad=None, known_bird=None):
            self.known.append(known_bird)
            return super().classify(image_path, roi_bbox, threshold, pad)

    bird, other = add(store, 0), add(store, 1)
    store.update_classification(bird.capture_id, True, 0.9, "Black Phoebe", 0.7)
    store.update_classification(other.capture_id, False, 0.1, None, None)
    unchecked(store, [bird.capture_id, other.capture_id])
    clf = KnownBirdFake(Prediction(is_bird=False, confidence=0.1, species="Black Redstart", species_confidence=0.9))
    Pipeline(store, clf, live).sweep()
    c = store.get(bird.capture_id)
    assert c.is_bird is True and c.species == "Black Redstart" and c.species_confidence == 0.9
    assert sorted(clf.known) == [False, True]          # only the stored bird is shown as known


def test_backfill_promotion_with_species_evaluates_publisher_and_never_notifies(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.05, None, None)
    detector_unchecked(store, [ev.capture_id])
    pred = Prediction(
        is_bird=True, confidence=0.55, species="Robin", species_confidence=0.8,
        detector_confidence=0.55, detector_box=(0.1, 0.2, 0.3, 0.4),
    )
    n, pub = FakeNotifier(), FakePublisher()
    p = Pipeline(store, ReturnsFake(pred), live, notifier=n, publisher=pub)
    p.sweep()
    p.visits.tick(T0 + timedelta(hours=1))
    assert pub.days == [("Robin", T0.date())]
    assert n.calls == []
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.species == "Robin"


def test_backfill_rename_reranks_old_and_new_days(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Black Phoebe", 0.8)
    unchecked(store, [ev.capture_id])                      # species pass pending (new model)
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9, species="Black Redstart", species_confidence=0.95)), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.species == "Black Redstart" and c.species_confidence == 0.95
    assert pub.days == [("Black Phoebe", T0.date()), ("Black Redstart", T0.date())]


def test_backfill_rename_to_nothing_forgets_old_species(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Black Phoebe", 0.8)
    unchecked(store, [ev.capture_id])
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9)), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.species is None and c.species_confidence is None
    assert pub.days == [("Black Phoebe", T0.date())]


def test_backfill_same_species_does_not_forget(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Robin", 0.8)
    unchecked(store, [ev.capture_id])
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9, species="Robin", species_confidence=0.85)), live, publisher=pub).sweep()
    # Same name, new confidence: that day is re-ranked once.
    assert pub.days == [("Robin", T0.date())]
    c = store.get(ev.capture_id)
    assert c.species == "Robin" and c.species_confidence == 0.85


def test_backfill_detector_pass_keeps_species_and_does_not_forget(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Robin", 0.8)
    detector_unchecked(store, [ev.capture_id])             # species_checked stays 1
    pub = FakePublisher()
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9, species="Blackbird", species_confidence=0.7)), live, publisher=pub).sweep()
    assert store.get(ev.capture_id).species == "Robin" and pub.days == []


def real_publisher(store, live, **kw):
    cfg = live.get()
    live.replace(dataclasses.replace(cfg, publish=dataclasses.replace(cfg.publish, **kw)))
    return Publisher(live, store, store.data_dir / "publish")      # never started: no rsync


def test_backfill_rename_to_nothing_clears_auto_publish_state(store, live):
    pub = real_publisher(store, live)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Black Phoebe", 0.8)
    store.set_published(ev.capture_id, "auto")
    unchecked(store, [ev.capture_id])
    Pipeline(store, ReturnsFake(Prediction(is_bird=True, confidence=0.9)), live, publisher=pub).sweep()
    c = store.get(ev.capture_id)
    assert c.published is None and c.species is None


def test_backfill_rename_hands_the_old_slot_to_the_next_best_photo(store, live):
    pub = real_publisher(store, live, per_species_per_day=1)
    a, b = add(store, 0), add(store, 1)
    store.update_classification(a.capture_id, True, 0.9, "Black Phoebe", 0.9)
    store.update_classification(b.capture_id, True, 0.9, "Black Phoebe", 0.6)
    store.set_published(a.capture_id, "auto")
    unchecked(store, [a.capture_id])
    pred = Prediction(is_bird=True, confidence=0.9, species="Black Redstart", species_confidence=0.95)
    Pipeline(store, ReturnsFake(pred), live, publisher=pub).sweep()
    ca, cb = store.get(a.capture_id), store.get(b.capture_id)
    assert cb.species == "Black Phoebe" and cb.published == "auto"
    assert ca.species == "Black Redstart" and ca.published == "auto"


def test_run_loop_ticks_visits(store, live):
    p = Pipeline(store, SpeciesFake(), live, sweep_interval=100, tick_interval=0.1)
    p.start()
    try:
        ev = add_at(store, recent())
        assert p.submit(ev)
        assert wait_until(lambda: store.get(ev.capture_id).status == "done")
        assert store.get(ev.capture_id).visit_id is not None
        assert wait_until(lambda: store.open_visits() == [], timeout=3)
    finally:
        p.stop()


def test_sweep_resolves_unlabelled_done_rows(store, live, caplog):
    ev = add(store, 0)
    store.store_verdict(ev.capture_id, is_bird=True, confidence=0.9, species="Robin", species_confidence=0.8)
    assert store.get(ev.capture_id).is_bird is None
    clf = FakeClassifier()
    with caplog.at_level(logging.INFO):
        Pipeline(store, clf, live).sweep()
    c = store.get(ev.capture_id)
    assert c.is_bird is True and c.species == "Robin" and c.visit_id is not None
    assert clf.calls == []                       # no model runs: the stored verdict is resolved
    assert store.unresolved_done() == []
    assert "resolved 1 unlabelled rows" in caplog.text


def test_detector_only_backfill_keeps_the_stored_embedding(store, live):
    ev = add(store, 0)
    old = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    store.store_verdict(ev.capture_id, is_bird=True, confidence=0.9, species="Robin", species_confidence=0.8, embedding=old)
    Pipeline(store, FakeClassifier(), live).sweep()            # resolves the row (unlabelled sweep)
    detector_unchecked(store, [ev.capture_id])                 # species_checked stays 1
    new = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    pred = Prediction(is_bird=True, confidence=0.9, species="Blackbird", species_confidence=0.7, embedding=new)
    Pipeline(store, ReturnsFake(pred), live).sweep()
    assert np.allclose(store.embeddings([ev.capture_id])[ev.capture_id], old)
    c = store.get(ev.capture_id)
    assert c.species == "Robin" and store.classification_backlog() == []
    unchecked(store, [ev.capture_id])                          # a species pass does take the new one
    Pipeline(store, ReturnsFake(pred), live).sweep()
    assert np.allclose(store.embeddings([ev.capture_id])[ev.capture_id], new)


class StagedClassifier:
    """The staged API the DSLR path uses; records which image each stage saw."""

    def __init__(self, bird=True, dslr_bird=0.7):
        self.bird, self.dslr_bird = bird, dslr_bird
        self.located: list[tuple] = []
        self.named: list[tuple] = []

    def classify(self, *a, **k):
        raise AssertionError("the DSLR path must use the staged API")

    def load_crop(self, image_path, roi_bbox, pad=None):
        return np.zeros((48, 64, 3), np.uint8)

    def verdict(self, img, threshold=None):
        return Verdict(self.bird, 0.9 if self.bird else 0.1, 0.0, None)

    def locate(self, img):
        self.located.append(img.shape[:2])
        return Detection("bird", self.dslr_bird, (0.5, 0.5, 0.2, 0.2)) if self.dslr_bird else None

    def name(self, img):
        self.named.append(img.shape[:2])
        return [], np.ones(8, np.float32) / np.sqrt(8)

    def prediction(self, v, top, emb):
        return Prediction(is_bird=v.is_bird, confidence=v.confidence, embedding=emb)


class FakeDslr:
    def __init__(self, fail=False, garbage=False):
        self.fail, self.garbage, self.shots = fail, garbage, []

    def capture(self, dest, shutter=None):
        self.shots.append(dest)
        self.shutters = getattr(self, "shutters", []) + [shutter]
        if self.fail:
            raise DslrError("no camera")
        if self.garbage:
            dest.write_bytes(b"not a jpeg")
        else:
            cv2.imwrite(str(dest), np.full((800, 1200, 3), getattr(self, 'level', 0), np.uint8))
        return 1.2


def fresh():
    return datetime.now().astimezone()


def dslr_on(live):
    cfg = live.get()
    live.replace(dataclasses.replace(cfg, dslr=dataclasses.replace(cfg.dslr, enabled=True)))


def run_one(store, live, clf, dslr, when):
    if not live.get().dslr.enabled:
        dslr_on(live)
    ev = add_at(store, when)
    p = Pipeline(store, clf, live, dslr=dslr)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    return store.get(ev.capture_id)


def test_dslr_fires_once_for_a_fresh_bird_and_becomes_the_photo(store, live):
    clf, dslr = StagedClassifier(), FakeDslr()
    c = run_one(store, live, clf, dslr, fresh())
    assert len(dslr.shots) == 1
    assert c.status == "done" and c.dslr_path == c.path.replace(".jpg", "_eos.jpg")
    assert c.photo == c.dslr_path and c.thumb_path == c.dslr_path
    # default region = whole frame, so the box maps one to one
    assert c.dslr_box == pytest.approx((0.5, 0.5, 0.2, 0.2))
    assert clf.located == [(800, 1200)]
    assert clf.named == [(480, 480)]          # 2 x the box's longer side (0.2 * 1200), from the DSLR frame
    assert c.has_embedding


def test_dslr_never_fires_for_a_non_bird(store, live):
    clf, dslr = StagedClassifier(bird=False), FakeDslr()
    c = run_one(store, live, clf, dslr, fresh())
    assert dslr.shots == [] and clf.named == [] and c.dslr_path is None and c.is_bird is False


def test_dslr_does_not_fire_for_a_stale_bird(store, live):
    clf, dslr = StagedClassifier(), FakeDslr()
    c = run_one(store, live, clf, dslr, datetime.now().astimezone() - timedelta(seconds=30))
    assert dslr.shots == [] and c.dslr_path is None
    assert clf.named == [(48, 64)]            # the species still runs, on the Pi crop


def test_dslr_frame_without_a_bird_keeps_the_pi_photo(store, live):
    clf, dslr = StagedClassifier(dslr_bird=0.0), FakeDslr()
    c = run_one(store, live, clf, dslr, fresh())
    assert c.dslr_path is not None and c.dslr_box is None and c.photo == c.path
    assert clf.named == [(48, 64)]


def test_a_weak_detection_in_the_dslr_frame_does_not_promote(store, live):
    clf = StagedClassifier(dslr_bird=live.get().detector.threshold - 0.05)
    c = run_one(store, live, clf, FakeDslr(), fresh())
    assert c.dslr_box is None and c.photo == c.path


def test_dslr_failure_falls_back_to_the_pi_photo(store, live):
    clf = StagedClassifier()
    c = run_one(store, live, clf, FakeDslr(fail=True), fresh())
    assert c.status == "done" and c.dslr_path is None and c.is_bird is True
    assert clf.named == [(48, 64)]


def test_an_undecodable_dslr_file_is_removed(store, live):
    clf, dslr = StagedClassifier(), FakeDslr(garbage=True)
    c = run_one(store, live, clf, dslr, fresh())
    assert c.dslr_path is None and not dslr.shots[0].exists() and c.status == "done"


class SmallBirdClassifier(StagedClassifier):
    """Sees nothing in the whole region; finds the bird only in tiles, best in the second."""

    def locate(self, img):
        self.located.append(img.shape[:2])
        n = len(self.located)
        if n == 1:
            return None
        return Detection("bird", 0.3 if n == 2 else 0.6, (0.5, 0.5, 0.25, 0.25))


def test_a_small_bird_is_found_in_a_tile_of_the_dslr_frame(store, live):
    clf = SmallBirdClassifier()
    c = run_one(store, live, clf, FakeDslr(), fresh())
    # the 1200 x 800 frame whole, then two overlapping 960 x 800 tiles; the better tile wins
    assert clf.located == [(800, 1200), (800, 960), (800, 960)]
    assert c.dslr_box == pytest.approx((0.2 + 0.5 * 0.8, 0.5, 0.25 * 0.8, 0.25))
    assert c.photo == c.dslr_path


def test_tiles_are_not_searched_when_the_whole_region_has_the_bird(store, live):
    clf = StagedClassifier()
    run_one(store, live, clf, FakeDslr(), fresh())
    assert len(clf.located) == 1


def test_the_dslr_region_limits_the_search_and_maps_the_box(store, live, config):
    live.replace(dataclasses.replace(config, dslr=dataclasses.replace(config.dslr, enabled=True, region=(0.5, 0.0, 0.5, 1.0))))
    clf = StagedClassifier()
    c = run_one(store, live, clf, FakeDslr(), fresh())
    assert clf.located == [(800, 600)]
    assert c.dslr_box == pytest.approx((0.75, 0.5, 0.1, 0.2))


def test_without_a_dslr_the_pipeline_calls_classify_only(store, live):
    clf = FakeClassifier()
    ev = add_at(store, fresh())
    Pipeline(store, clf, live).process(ev.capture_id, ev.path, ev.captured_at)
    assert clf.calls == [ev.path] and store.get(ev.capture_id).dslr_path is None


def test_backfill_of_a_promoted_row_names_from_the_dslr_crop_and_never_fires(store, live):
    clf, dslr = StagedClassifier(), FakeDslr()
    ev = add_at(store, fresh())
    dslr_on(live)
    p = Pipeline(store, clf, live, dslr=dslr)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).dslr_box is not None
    # A species-model swap puts the row back in the backlog.
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0 WHERE id = ?", (ev.capture_id,))
    clf.named.clear()
    p.sweep()
    assert clf.named == [(480, 480)]          # the DSLR species crop again, not the Pi crop
    assert len(dslr.shots) == 1               # the backfill never fires the camera
    assert store.get(ev.capture_id).photo == store.get(ev.capture_id).dslr_path


def test_backfill_of_a_promoted_row_falls_back_to_the_pi_crop_when_the_dslr_file_is_gone(store, live):
    clf = StagedClassifier()
    ev = add_at(store, fresh())
    dslr_on(live)
    p = Pipeline(store, clf, live, dslr=FakeDslr())
    p.process(ev.capture_id, ev.path, ev.captured_at)
    c = store.get(ev.capture_id)
    (store.photos_dir / c.dslr_path).unlink()
    with store._lock, store._conn:
        store._conn.execute("UPDATE captures SET species_checked = 0 WHERE id = ?", (ev.capture_id,))
    clf.named.clear()
    p.sweep()
    assert clf.named == [(48, 64)] and store.get(ev.capture_id).status == "done"


def test_a_dslr_frame_is_removed_when_its_verdict_cannot_be_stored(store, live, monkeypatch):
    dslr = FakeDslr()
    ev = add_at(store, fresh())

    def boom(*a, **k):
        raise RuntimeError("db is gone")

    monkeypatch.setattr(store, "store_verdict", boom)
    dslr_on(live)
    Pipeline(store, StagedClassifier(), live, dslr=dslr).process(ev.capture_id, ev.path, ev.captured_at)
    assert len(dslr.shots) == 1 and not dslr.shots[0].exists()
    assert store.get(ev.capture_id).status == "error"
    assert ev.path.exists()                       # the Pi photo is untouched


def test_the_dslr_switch_is_live(store, live):
    """Off: the pipeline is the Pi-only path even though a Dslr exists. On: the next bird fires it."""
    dslr = FakeDslr()
    off = add_at(store, fresh())
    clf = FakeClassifier()
    Pipeline(store, clf, live, dslr=dslr).process(off.capture_id, off.path, off.captured_at)
    assert dslr.shots == [] and clf.calls == [off.path]
    dslr_on(live)
    on = add_at(store, fresh() + timedelta(milliseconds=5))
    Pipeline(store, StagedClassifier(), live, dslr=dslr).process(on.capture_id, on.path, on.captured_at)
    assert len(dslr.shots) == 1


def test_a_photo_taken_by_hand_fires_the_dslr_even_for_a_non_bird(store, live):
    dslr_on(live)
    clf, dslr = StagedClassifier(bird=False), FakeDslr()
    ev = add_at(store, fresh())
    Pipeline(store, clf, live, dslr=dslr).process(ev.capture_id, ev.path, ev.captured_at, manual=True)
    c = store.get(ev.capture_id)
    assert len(dslr.shots) == 1 and c.is_bird is False
    # taken by hand: the 50D frame is the main photo even with no bird in it
    assert c.dslr_path is not None and c.dslr_box is None and c.dslr_main is True
    assert c.photo == c.dslr_path and c.thumb_path == c.dslr_path
    assert clf.named == [] and clf.located == []                                  # no species for a non-bird


def test_a_bird_photo_taken_by_hand_is_handled_like_any_bird(store, live):
    dslr_on(live)
    clf, dslr = StagedClassifier(), FakeDslr()
    ev = add_at(store, fresh())
    Pipeline(store, clf, live, dslr=dslr).process(ev.capture_id, ev.path, ev.captured_at, manual=True)
    c = store.get(ev.capture_id)
    assert len(dslr.shots) == 1 and c.photo == c.dslr_path


def test_the_worker_passes_the_manual_flag(store, live):
    dslr_on(live)
    dslr = FakeDslr()
    p = Pipeline(store, StagedClassifier(bird=False), live, dslr=dslr)
    p.start()
    try:
        ev = dataclasses.replace(add_at(store, fresh()), manual=True)
        p.submit(ev)
        assert wait_until(lambda: store.get(ev.capture_id).is_bird is not None)
        assert len(dslr.shots) == 1
    finally:
        p.stop()


def test_an_automatic_bird_with_no_bird_in_the_dslr_frame_is_not_promoted_by_hand_rule(store, live):
    dslr_on(live)
    c = run_one(store, live, StagedClassifier(dslr_bird=0.0), FakeDslr(), fresh())
    assert c.dslr_main is False and c.photo == c.path


def auto_shutter_on(live):
    cfg = live.get()
    live.replace(dataclasses.replace(cfg, dslr=dataclasses.replace(cfg.dslr, enabled=True, auto_shutter=True)))


def shoot(store, live, dslr, lux, i=0):
    ev = add_at(store, fresh() + timedelta(milliseconds=i))
    Pipeline(store, StagedClassifier(), live, dslr=dslr).process(ev.capture_id, ev.path, ev.captured_at, lux=lux)
    return store.get(ev.capture_id)


def test_auto_shutter_picks_the_speed_from_the_light_and_learns_from_a_dark_frame(store, live):
    auto_shutter_on(live)
    dslr = FakeDslr()
    dslr.level = 50                                   # half the target brightness
    shoot(store, live, dslr, lux=300)                 # k 1.5 -> needs 1/200
    assert dslr.shutters == ["1/200"]
    assert float(store.get_meta("dslr_exposure_k")) == pytest.approx(3.0)
    shoot(store, live, dslr, lux=300, i=5)            # k 3.0 -> needs 1/100
    assert dslr.shutters[-1] == "1/100"


def test_auto_shutter_creeps_faster_after_a_bright_frame(store, live):
    auto_shutter_on(live)
    dslr = FakeDslr()
    dslr.level = 120
    shoot(store, live, dslr, lux=300)
    assert float(store.get_meta("dslr_exposure_k")) == pytest.approx(1.5 * 0.95)


def test_auto_shutter_off_or_no_light_reading_leaves_the_camera_alone(store, live):
    dslr_on(live)
    dslr = FakeDslr()
    shoot(store, live, dslr, lux=300)                 # switch off
    auto_shutter_on(live)
    shoot(store, live, dslr, lux=None, i=5)           # switch on, no reading
    assert dslr.shutters == [None, None] and store.get_meta("dslr_exposure_k") is None


def test_the_worker_passes_the_light_reading(store, live):
    auto_shutter_on(live)
    dslr = FakeDslr()
    p = Pipeline(store, StagedClassifier(), live, dslr=dslr)
    p.start()
    try:
        ev = dataclasses.replace(add_at(store, fresh()), lux=5000.0)
        p.submit(ev)
        assert wait_until(lambda: store.get(ev.capture_id).is_bird is not None)
        assert dslr.shutters == ["1/500"]
    finally:
        p.stop()
