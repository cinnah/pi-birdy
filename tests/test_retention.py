import logging
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from birdy.retention import RetentionJob, run_once
from birdy.store import Store

T0 = datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def add_non_bird(store, when):
    abs_path, rel = store.photo_path_for(when)
    cv2.imwrite(str(abs_path), np.zeros((60, 80, 3), np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, False, 0.1, None, None)
    return cid


def test_run_once_purges_and_logs(store, caplog):
    old = add_non_bird(store, T0 - timedelta(days=10))
    new = add_non_bird(store, T0)
    with caplog.at_level(logging.INFO):
        deleted = run_once(store, non_bird_days=7, now=T0)
    assert deleted == 1
    assert store.get(old) is None and store.get(new) is not None
    assert "disk free" in caplog.text


def test_low_disk_warns(store, caplog, monkeypatch):
    monkeypatch.setattr(store, "disk_free_bytes", lambda: 500 * 1024 * 1024)
    with caplog.at_level(logging.WARNING):
        run_once(store, non_bird_days=7, now=T0)
    assert "low disk" in caplog.text


def test_job_thread_runs_periodically(store, live):
    add_non_bird(store, T0 - timedelta(days=10))
    job = RetentionJob(store, live, interval=0.1)
    job.start()
    try:
        end = time.monotonic() + 3
        while time.monotonic() < end and store.count("all") != 0:
            time.sleep(0.02)
        assert store.count("all") == 0
    finally:
        job.stop()


def test_job_uses_current_days(store, live, config):
    import dataclasses

    add_non_bird(store, datetime.now(timezone.utc) - timedelta(days=3))
    run_once(store, 7)  # default 7 days: the 3-day-old row must survive
    assert store.count("all") == 1
    live.replace(dataclasses.replace(config, retention=dataclasses.replace(config.retention, non_bird_days=1)))
    job = RetentionJob(store, live, interval=0.1)
    job.start()
    try:
        end = time.monotonic() + 3
        while time.monotonic() < end and store.count("all") != 0:
            time.sleep(0.02)
        assert store.count("all") == 0
    finally:
        job.stop()


class SpyVisits:
    def __init__(self, fail=False):
        self.removed = []
        self.fail = fail

    def photo_removed(self, visit_id):
        self.removed.append(visit_id)
        if self.fail:
            raise RuntimeError("boom")


def test_run_once_re_resolves_the_visits_of_deleted_rows(store):
    vid = store.create_visit(T0 - timedelta(days=10), T0 - timedelta(days=10))
    in_visit = add_non_bird(store, T0 - timedelta(days=10))
    in_visit_too = add_non_bird(store, T0 - timedelta(days=10, seconds=-5))
    add_non_bird(store, T0 - timedelta(days=9))                 # no visit
    store.attach_to_visit(in_visit, vid)
    store.attach_to_visit(in_visit_too, vid)
    spy = SpyVisits()
    assert run_once(store, non_bird_days=7, now=T0, visits=spy) == 3
    assert spy.removed == [vid]                                  # once per visit, never None


def test_run_once_logs_a_visits_failure_and_carries_on(store, caplog):
    for days in (10, 11):
        vid = store.create_visit(T0 - timedelta(days=days), T0 - timedelta(days=days))
        store.attach_to_visit(add_non_bird(store, T0 - timedelta(days=days)), vid)
    spy = SpyVisits(fail=True)
    with caplog.at_level(logging.ERROR):
        assert run_once(store, non_bird_days=7, now=T0, visits=spy) == 2
    assert len(spy.removed) == 2 and "re-resolving visit" in caplog.text


def test_job_passes_visits(store, live):
    add_non_bird(store, T0 - timedelta(days=10))
    vid = store.create_visit(T0, T0)
    store.attach_to_visit(add_non_bird(store, datetime.now(timezone.utc) - timedelta(days=30)), vid)
    spy = SpyVisits()
    job = RetentionJob(store, live, interval=0.1, visits=spy)
    job.start()
    try:
        end = time.monotonic() + 3
        while time.monotonic() < end and not spy.removed:
            time.sleep(0.02)
    finally:
        job.stop()
    assert spy.removed == [vid]
