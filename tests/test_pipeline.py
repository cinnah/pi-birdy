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
