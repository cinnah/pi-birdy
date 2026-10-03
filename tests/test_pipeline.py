import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.classifier import Prediction
from birdy.motion import BBox
from birdy.pipeline import MotionEvent, Pipeline
from birdy.store import Store

# Local-time aware: the fail_on filename below depends on local HHMMSS.
T0 = datetime(2026, 10, 3, 10, 0).astimezone()


class FakeClassifier:
    def __init__(self, fail_on: set[str] = ()):
        self.fail_on = set(fail_on)
        self.calls: list[Path] = []

    def classify(self, image_path, roi_bbox):
        self.calls.append(image_path)
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


def test_worker_classifies_submitted_event(store):
    clf = FakeClassifier()
    p = Pipeline(store, clf, roi_bbox=BBox(0, 0, 10, 10))
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


def test_classifier_error_marks_row_error_and_continues(store):
    p = Pipeline(store, FakeClassifier(fail_on={"100000_000.jpg"}), roi_bbox=None)
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


def test_submit_returns_false_when_full(store):
    p = Pipeline(store, FakeClassifier(), roi_bbox=None, maxsize=1)
    # not started: nothing drains the queue
    assert p.submit(add(store, 0)) is True
    assert p.submit(add(store, 1)) is False
    assert p.queue_depth() == 1


def test_sweep_classifies_pending_rows(store):
    p = Pipeline(store, FakeClassifier(), roi_bbox=None)
    add(store, 0)
    add(store, 1)
    assert len(store.pending()) == 2
    p.sweep()
    assert store.pending() == []
    assert store.count("birds") == 2


def test_periodic_sweep_runs(store):
    p = Pipeline(store, FakeClassifier(), roi_bbox=None, sweep_interval=0.1)
    add(store, 0)
    p.start()
    try:
        assert wait_until(lambda: store.pending() == [])
    finally:
        p.stop()


def test_store_error_does_not_kill_worker(store, monkeypatch):
    real_update = store.update_classification
    calls = {"n": 0}

    def flaky_update(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db write failed")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(store, "update_classification", flaky_update)
    p = Pipeline(store, FakeClassifier(), roi_bbox=None)
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


def test_queued_event_already_classified_is_skipped(store):
    clf = FakeClassifier()
    p = Pipeline(store, clf, roi_bbox=None)
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


def test_store_get_error_in_worker_does_not_kill_thread(store, monkeypatch):
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
    p = Pipeline(store, clf, roi_bbox=None)
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
    def classify(self, image_path, roi_bbox):
        time.sleep(0.3)
        return super().classify(image_path, roi_bbox)


def test_sweep_stops_promptly(store):
    for i in range(10):
        add(store, i)
    clf = SlowClassifier()
    p = Pipeline(store, clf, roi_bbox=None)
    p.start()  # startup sweep begins after the first empty queue tick
    assert wait_until(lambda: len(clf.calls) >= 1)
    time.sleep(0.1)
    t = time.monotonic()
    p.stop(timeout=1.0)
    assert time.monotonic() - t < 0.5
    assert not p._thread.is_alive()
    assert 1 <= len(clf.calls) < 10
