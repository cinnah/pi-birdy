"""Publish policy and Publisher (mirror, manifest, rsync, thread)."""

import dataclasses
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from birdy.publish import PublishSettings, Publisher, local_day_bounds, rank_day, rsync_command
from birdy.store import Capture, Store

T0 = datetime(2026, 10, 4, 9, 0, tzinfo=timezone.utc)
S = PublishSettings(min_confidence=0.6, per_species_per_day=2)


def cap(i, conf, published=None, species="Robin"):
    return Capture(
        id=i, captured_at=T0 + timedelta(minutes=i), path=f"p{i}.jpg", thumb_path=None, motion_score=1,
        is_bird=True, confidence=0.9, species=species, species_confidence=conf, status="done",
        species_checked=True, published=published,
    )


def test_rank_day_top_n_by_confidence_then_id():
    got = rank_day([cap(1, 0.7), cap(2, 0.9), cap(3, 0.9), cap(4, 0.8)], {}, S)
    assert got == {2: "auto", 3: "auto", 4: None, 1: None}


def test_rank_day_threshold():
    got = rank_day([cap(1, 0.59), cap(2, 0.6), cap(3, None)], {}, S)
    assert got == {1: None, 2: "auto", 3: None}


def test_rank_day_demotes_weaker_auto():
    got = rank_day([cap(1, 0.7, "auto"), cap(2, 0.8, "auto"), cap(3, 0.95)], {}, S)
    assert got == {3: "auto", 2: "auto", 1: None}


def test_rank_day_manual_states_win_and_forced_does_not_take_a_slot():
    manual = {1: "forced", 2: "hidden"}
    got = rank_day([cap(1, 0.1), cap(2, 0.99), cap(3, 0.7), cap(4, 0.65), cap(5, 0.61)], manual, S)
    assert got == {1: "forced", 2: "hidden", 3: "auto", 4: "auto", 5: None}


def test_rank_day_n_bounds():
    caps = [cap(i, 0.9) for i in range(1, 25)]
    assert sum(v == "auto" for v in rank_day(caps, {}, PublishSettings(0.6, 1)).values()) == 1
    assert sum(v == "auto" for v in rank_day(caps, {}, PublishSettings(0.6, 20)).values()) == 20


def test_rank_day_empty():
    assert rank_day([], {}, S) == {}


def test_rank_day_manual_entry_for_unknown_id_is_ignored():
    got = rank_day([cap(1, 0.7)], {99: "hidden"}, S)
    assert got == {1: "auto"}


def test_rank_day_stray_manual_auto_is_ranked_normally():
    got = rank_day([cap(1, 0.9), cap(2, 0.8), cap(3, 0.7)], {1: "auto"}, S)
    assert got == {1: "auto", 2: "auto", 3: None}


def test_rank_day_hidden_takes_no_slot():
    manual = {1: "hidden", 2: "hidden"}
    got = rank_day([cap(1, 0.99), cap(2, 0.98), cap(3, 0.7), cap(4, 0.8), cap(5, 0.65)], manual, S)
    assert got == {1: "hidden", 2: "hidden", 4: "auto", 3: "auto", 5: None}


@pytest.fixture
def lisbon(monkeypatch):
    monkeypatch.setenv("TZ", "Europe/Lisbon")
    time.tzset()
    try:
        yield
    finally:
        monkeypatch.undo()
        time.tzset()


def test_local_day_bounds(lisbon):
    when = datetime(2026, 10, 4, 23, 30, tzinfo=timezone.utc)  # 00:30 on the 5th in Lisbon (WEST, UTC+1)
    start, end = local_day_bounds(when)
    assert start == datetime(2026, 10, 5, 0, 0, tzinfo=timezone(timedelta(hours=1)))
    assert start.utcoffset() == timedelta(hours=1)
    assert end - start == timedelta(days=1)


# --- Publisher ----------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data")


def enabled(config, **kw):
    base = dict(enabled=True, host="gallery.example", user="birdy", remote_dir="/", site_id="home",
                ssh_key=Path("/keys/pub"), min_confidence=0.6, per_species_per_day=2)
    base.update(kw)
    return dataclasses.replace(config, publish=dataclasses.replace(config.publish, **base))


class FakeRunner:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        return self.rc


def bird(store, when, species, conf, size=(2464, 3280), is_bird=True, write=True):
    abs_path, rel = store.photo_path_for(when)
    if write:
        cv2.imwrite(str(abs_path), np.full((*size, 3), 120, np.uint8))
    cid = store.add_capture(when, rel, 100)
    store.update_classification(cid, is_bird, 0.9, species, conf)
    return store.get(cid)


def make(live, store, config, runner=None, **kw):
    live.replace(enabled(config, **kw))
    return Publisher(live, store, store.data_dir / "publish", runner=runner or FakeRunner(), debounce=0.05, max_interval=0.5)


# --- evaluate / set_manual -------------------------------------------------

def test_evaluate_publishes_top_n_and_demotes(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.7)
    assert pub.evaluate(a) is True
    assert store.get(a.id).published == "auto"
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.8)
    assert pub.evaluate(b) is True
    c = bird(store, T0 + timedelta(minutes=2), "Robin", 0.9)
    assert pub.evaluate(c) is True
    assert [store.get(x.id).published for x in (a, b, c)] == [None, "auto", "auto"]
    assert pub.evaluate(c) is False  # nothing changed


def test_evaluate_ignores_non_birds_and_no_species(live, store, config):
    pub = make(live, store, config)
    assert pub.evaluate(bird(store, T0, None, None)) is False
    assert pub.evaluate(bird(store, T0, "Cat", 0.9, is_bird=False)) is False


def test_evaluate_reads_live_settings(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.7)
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.8)
    pub.evaluate(a)
    pub.evaluate(b)
    assert store.get(a.id).published is None
    live.replace(enabled(config, per_species_per_day=2))
    pub.evaluate(b)
    assert store.get(a.id).published == "auto"


def test_set_manual_forced_hidden_auto(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.9)
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.7)
    pub.evaluate(b)
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)
    pub.set_manual(a.id, "hidden")           # frees the slot -> b becomes auto
    assert (store.get(a.id).published, store.get(b.id).published) == ("hidden", "auto")
    pub.set_manual(b.id, "forced")           # forced doesn't use a slot, but a is hidden
    assert (store.get(a.id).published, store.get(b.id).published) == ("hidden", "forced")
    pub.set_manual(a.id, None)               # back to the rule -> a takes the slot
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", "forced")
    pub.set_manual(b.id, None)
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)


def test_set_manual_without_species_and_unknown_id(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, None, None)
    pub.set_manual(a.id, "forced")
    assert store.get(a.id).published == "forced"
    pub.set_manual(a.id, None)
    assert store.get(a.id).published is None
    with pytest.raises(KeyError):
        pub.set_manual(999, "forced")
    with pytest.raises(ValueError):
        pub.set_manual(a.id, "auto")


# --- run_once: mirror + manifest + rsync -------------------------------------

def test_run_once_builds_mirror_and_manifest_and_rsyncs(live, store, config):
    runner = FakeRunner()
    pub = make(live, store, config, runner=runner)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    assert pub.run_once(now=T0 + timedelta(hours=1)) is True

    web = pub.photos_dir / a.path
    assert web.exists()
    img = cv2.imread(str(web))
    assert max(img.shape[:2]) == 1600 and img.shape[0] < img.shape[1]
    assert (pub.thumbs_dir / a.path).exists()
    assert store.get(a.id).web_path == a.path

    m = json.loads((pub.mirror_dir / "manifest.json").read_text())
    assert m["version"] == 1 and m["site"]["id"] == "home" and m["site"]["name"] == "test"
    assert m["site"]["lat"] is None and isinstance(m["site"]["utc_offset_minutes"], int)
    assert m["site"]["since"] == T0.astimezone().date().isoformat()
    assert m["photos"] == [{
        "id": a.id, "captured_at": a.captured_at.astimezone().isoformat(timespec="seconds"),
        "species": "Robin", "confidence": 0.9, "photo": f"photos/{a.path}", "thumb": f"thumbs/{a.path}", "state": "auto",
    }]
    assert m["stats"]["total_birds"] == 1 and len(m["stats"]["per_day"]) == 30
    assert m["generated_at"].startswith((T0 + timedelta(hours=1)).astimezone().date().isoformat())

    assert runner.calls == [rsync_command(pub.mirror_dir, live.get().publish)]
    cmd = runner.calls[0]
    assert cmd[:3] == ["rsync", "-az", "--delete"]
    assert cmd[-2] == f"{pub.mirror_dir}/" and cmd[-1] == "birdy@gallery.example:/"
    assert "-i /keys/pub" in cmd[cmd.index("-e") + 1] and "BatchMode=yes" in cmd[cmd.index("-e") + 1]


def test_run_once_never_upscales(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9, size=(60, 80))
    pub.evaluate(a)
    pub.run_once()
    assert cv2.imread(str(pub.photos_dir / a.path)).shape[:2] == (60, 80)


def test_run_once_removes_unpublished_files(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    pub.run_once()
    pub.set_manual(a.id, "hidden")
    pub.run_once()
    assert not (pub.photos_dir / a.path).exists() and not (pub.thumbs_dir / a.path).exists()
    assert store.get(a.id).web_path is None
    assert json.loads((pub.mirror_dir / "manifest.json").read_text())["photos"] == []


def test_run_once_hides_unreadable_original(live, store, config, caplog):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9, write=False)
    (store.photos_dir / a.path).write_bytes(b"")   # power-cut leftover
    pub.evaluate(a)
    with caplog.at_level(logging.WARNING):
        pub.run_once()
    assert store.get(a.id).published == "hidden"
    assert "hiding capture" in caplog.text
    assert json.loads((pub.mirror_dir / "manifest.json").read_text())["photos"] == []


def test_run_once_rsync_failure_logs_and_returns_false(live, store, config, caplog):
    pub = make(live, store, config, runner=FakeRunner(rc=255))
    pub.evaluate(bird(store, T0, "Robin", 0.9))
    with caplog.at_level(logging.WARNING):
        assert pub.run_once() is False
    assert "rsync" in caplog.text and "exit 255" in caplog.text


def test_run_once_thumb_falls_back_to_photo(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    (store.thumbs_dir / a.thumb_path).unlink()
    pub.evaluate(a)
    pub.run_once()
    m = json.loads((pub.mirror_dir / "manifest.json").read_text())
    assert m["photos"][0]["thumb"] == f"photos/{a.path}"


# --- thread -------------------------------------------------------------------

def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_thread_runs_on_start_and_coalesces_notifies(live, store, config):
    runner = FakeRunner()
    pub = make(live, store, config, runner=runner)
    pub.start()
    try:
        assert wait_until(lambda: len(runner.calls) == 1)      # reconcile on start
        pub.notify()
        pub.notify()
        pub.notify()
        assert wait_until(lambda: len(runner.calls) == 2)      # one run for the burst
        time.sleep(0.15)
        assert len(runner.calls) == 2
        assert wait_until(lambda: len(runner.calls) >= 3, timeout=3.0)   # max_interval tick
    finally:
        pub.stop()


def test_thread_survives_run_errors(live, store, config, caplog):
    class Boom:
        calls = 0

        def __call__(self, cmd):
            self.calls += 1
            raise RuntimeError("no")

    runner = Boom()
    pub = make(live, store, config, runner=runner)
    pub.start()
    try:
        with caplog.at_level(logging.ERROR):
            assert wait_until(lambda: runner.calls >= 1)
            pub.notify()
            assert wait_until(lambda: runner.calls >= 2)
        assert "publish run failed" in caplog.text
    finally:
        pub.stop()
