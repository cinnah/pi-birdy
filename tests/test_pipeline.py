import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import Prediction
from birdy.motion import BBox, roi_bbox
from birdy.pipeline import MotionEvent, Pipeline
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

    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
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
    when = T0.replace(minute=i)
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
        assert wait_until(lambda: store.get(ev.capture_id).status == "done")
        c = store.get(ev.capture_id)
        assert c.is_bird is True and c.confidence == 0.9
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
    real_update = store.update_classification
    calls = {"n": 0}

    def flaky_update(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db write failed")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(store, "update_classification", flaky_update)
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
    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
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

    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
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
    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
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

    def consider(self, capture_id, photo_path, pred, captured_at):
        self.calls.append((capture_id, photo_path, pred.species, captured_at))
        if self.raise_exc:
            raise self.raise_exc
        return "watchlist"


def test_live_process_notifies(store, live):
    n = FakeNotifier()
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert n.calls == [(ev.capture_id, ev.path, "Robin", ev.captured_at)]
    assert store.get(ev.capture_id).status == "done"


def test_notifier_exception_does_not_mark_error(store, live):
    p = Pipeline(store, SpeciesFake(), live, notifier=FakeNotifier(raise_exc=RuntimeError("boom")))
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).status == "done"


def test_backfill_never_notifies(store, live):
    n = FakeNotifier()
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    p.sweep()
    assert store.get(ev.capture_id).species == "Robin" and n.calls == []


def test_queued_event_passes_captured_at(store, live):
    n = FakeNotifier()
    p = Pipeline(store, SpeciesFake(), live, notifier=n)
    p.start()
    try:
        ev = add(store, 0)
        p.submit(ev)
        assert wait_until(lambda: n.calls)
        assert n.calls[0][3] == ev.captured_at
    finally:
        p.stop()


class FakePublisher:
    def __init__(self, raise_exc=None):
        self.raise_exc = raise_exc
        self.evaluated = []

    def evaluate(self, capture):
        self.evaluated.append(capture)
        if self.raise_exc:
            raise self.raise_exc
        return True


def test_process_calls_publisher_with_stored_capture(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert [c.id for c in pub.evaluated] == [ev.capture_id]
    assert pub.evaluated[0].species == "Robin" and pub.evaluated[0].status == "done"


def test_publisher_error_does_not_mark_row_error(store, live, caplog):
    pub = FakePublisher(raise_exc=RuntimeError("boom"))
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    with caplog.at_level(logging.ERROR):
        p.process(ev.capture_id, ev.path, ev.captured_at)
    assert store.get(ev.capture_id).status == "done"
    assert "publish decision failed" in caplog.text


def test_backfill_calls_publisher(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p.sweep()
    assert [c.id for c in pub.evaluated] == [ev.capture_id]
    assert pub.evaluated[0].species == "Robin"


def test_backfill_without_species_does_not_call_publisher(store, live):
    pub = FakePublisher()
    p = Pipeline(store, SpeciesFake(species=None, conf=None), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    p.sweep()
    assert pub.evaluated == []
    assert store.classification_backlog() == []


def test_process_skips_publisher_without_species(store, live):
    pub = FakePublisher()
    p = Pipeline(store, FakeClassifier(), live, publisher=pub)
    ev = add(store, 0)
    p.process(ev.capture_id, ev.path, ev.captured_at)
    assert pub.evaluated == []
    assert store.get(ev.capture_id).status == "done"


def test_backfill_publisher_error_keeps_row_done_and_checked(store, live, caplog):
    pub = FakePublisher(raise_exc=RuntimeError("boom"))
    p = Pipeline(store, SpeciesFake(), live, publisher=pub)
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, None, None)
    unchecked(store, [ev.capture_id])
    with caplog.at_level(logging.ERROR):
        p.sweep()
    assert "publish decision failed" in caplog.text
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
    assert pub.evaluated == []
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

    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
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
    assert "1 rows, 0 with species, 1 promoted" in caplog.text


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

    def classify(self, image_path, roi_bbox, threshold=None, pad=None):
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


def test_backfill_new_species_overwrites_a_birds_species(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, True, 0.9, "Robin", 0.8)
    detector_unchecked(store, [ev.capture_id])
    pred = Prediction(is_bird=True, confidence=0.9, species="Blackbird", species_confidence=0.7)
    Pipeline(store, ReturnsFake(pred), live).sweep()
    c = store.get(ev.capture_id)
    assert c.species == "Blackbird" and c.species_confidence == 0.7
    assert store.classification_backlog() == []


def test_backfill_promotion_with_species_evaluates_publisher_and_never_notifies(store, live):
    ev = add(store, 0)
    store.update_classification(ev.capture_id, False, 0.05, None, None)
    detector_unchecked(store, [ev.capture_id])
    pred = Prediction(
        is_bird=True, confidence=0.55, species="Robin", species_confidence=0.8,
        detector_confidence=0.55, detector_box=(0.1, 0.2, 0.3, 0.4),
    )
    n, pub = FakeNotifier(), FakePublisher()
    Pipeline(store, ReturnsFake(pred), live, notifier=n, publisher=pub).sweep()
    assert [c.id for c in pub.evaluated] == [ev.capture_id]
    assert n.calls == []
    assert store.get(ev.capture_id).is_bird is True
