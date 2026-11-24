"""Publish policy and Publisher (mirror, manifest, rsync, thread)."""

import dataclasses
import json
import logging
import os
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
    try:
        with monkeypatch.context() as m:
            m.setenv("TZ", "Europe/Lisbon")
            time.tzset()
            yield
    finally:
        time.tzset()  # back to the process zone


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


def make(live, store, config, runner=None, debounce=0.1, max_interval=0.5, **kw):
    live.replace(enabled(config, **kw))
    return Publisher(
        live, store, store.data_dir / "publish", runner=runner or FakeRunner(), debounce=debounce, max_interval=max_interval
    )


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


def test_evaluate_all_ranks_every_species_day(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    day1 = [bird(store, T0 + timedelta(minutes=i), "Robin", c) for i, c in enumerate((0.7, 0.9))]
    day2 = [bird(store, T0 + timedelta(days=1, minutes=i), "Robin", c) for i, c in enumerate((0.95, 0.8))]
    wren = bird(store, T0, "Wren", 0.65)
    low = bird(store, T0, "Jay", 0.3)                      # below min_confidence
    assert pub.evaluate_all() == 3
    assert [store.get(c.id).published for c in day1] == [None, "auto"]
    assert [store.get(c.id).published for c in day2] == ["auto", None]
    assert store.get(wren.id).published == "auto"
    assert store.get(low.id).published is None
    assert pub._wake.is_set()
    assert pub.evaluate_all() == 0


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
    assert "--delete-after" in cmd and "--delay-updates" in cmd   # server never lists a photo before it arrives
    assert cmd[cmd.index(".lock") - 1] == "--exclude" and cmd[cmd.index("*.tmp*") - 1] == "--exclude"
    assert cmd[-2] == f"{pub.mirror_dir}/" and cmd[-1] == "birdy@gallery.example:/"
    ssh = cmd[cmd.index("-e") + 1]
    assert ssh.startswith("ssh -i /keys/pub ")
    for opt in ("BatchMode=yes", "ConnectTimeout=20", "IdentitiesOnly=yes", "StrictHostKeyChecking=accept-new"):
        assert f"-o {opt}" in ssh


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


def test_run_once_unreadable_original_frees_the_slot(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.9, write=False)
    (store.photos_dir / a.path).write_bytes(b"")
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.7)
    pub.evaluate(b)
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)
    pub.run_once()
    assert (store.get(a.id).published, store.get(b.id).published) == ("hidden", "auto")
    assert pub._wake.is_set()   # the next run mirrors b


def test_run_once_write_failure_keeps_state_and_retries(live, store, config, caplog, monkeypatch):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    with monkeypatch.context() as m, caplog.at_level(logging.WARNING):
        m.setattr(cv2, "imwrite", lambda *args, **kw: False)   # disk full
        pub.run_once()
    assert store.get(a.id).published == "auto"
    assert store.get(a.id).web_path is None
    assert "cannot write web copy" in caplog.text
    assert json.loads((pub.mirror_dir / "manifest.json").read_text())["photos"] == []
    pub.run_once()   # next run succeeds
    assert json.loads((pub.mirror_dir / "manifest.json").read_text())["photos"][0]["id"] == a.id


def test_run_once_truncated_write_leaves_no_half_file(live, store, config, monkeypatch):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)

    def truncated_imwrite(path, img, params=None):
        Path(path).write_bytes(b"\xff\xd8\xff")   # disk full after a few bytes
        return False

    with monkeypatch.context() as m:
        m.setattr(cv2, "imwrite", truncated_imwrite)
        pub.run_once()
    final = pub.photos_dir / a.path
    assert not final.exists()
    assert not list(pub.photos_dir.rglob("*.tmp.jpg"))
    pub.run_once()
    img = cv2.imread(str(final))
    assert img is not None and max(img.shape[:2]) == 1600


def test_run_once_recopies_missing_thumb_and_prunes_empty_dirs(live, store, config):
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    pub.run_once()
    (pub.thumbs_dir / a.path).unlink()
    pub.run_once()
    assert (pub.thumbs_dir / a.path).exists()
    pub.set_manual(a.id, "hidden")
    pub.run_once()
    assert not any(pub.photos_dir.iterdir()) and not any(pub.thumbs_dir.iterdir())


def test_run_once_remakes_empty_mirror_files(live, store, config):
    # a power cut before the data reached the disk leaves 0-byte files behind a completed rename
    pub = make(live, store, config)
    a = bird(store, T0, "Robin", 0.9)
    pub.evaluate(a)
    pub.run_once()
    (pub.photos_dir / a.path).write_bytes(b"")
    (pub.thumbs_dir / a.path).write_bytes(b"")
    pub.run_once()
    img = cv2.imread(str(pub.photos_dir / a.path))
    assert img is not None and max(img.shape[:2]) == 1600
    assert (pub.thumbs_dir / a.path).stat().st_size > 0


def test_write_mirror_file_fsyncs_before_rename(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])
    target = tmp_path / "x.jpg"
    Publisher._write_mirror_file(target, lambda tmp: tmp.write_bytes(b"jpeg"))
    assert target.read_bytes() == b"jpeg"
    assert calls == ["fsync", "replace"]


def test_run_once_rsync_failure_logs_and_returns_false(live, store, config, caplog):
    pub = make(live, store, config, runner=FakeRunner(rc=255))
    pub.evaluate(bird(store, T0, "Robin", 0.9))
    with caplog.at_level(logging.WARNING):
        assert pub.run_once() is False
    assert "rsync" in caplog.text and "exit 255" in caplog.text


def test_run_once_skips_while_another_process_holds_the_lock(live, store, config, caplog):
    import fcntl
    import os

    runner = FakeRunner()
    pub = make(live, store, config, runner=runner)
    pub.mirror_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(pub.mirror_dir / ".lock", os.O_RDWR | os.O_CREAT)  # a separate open file description,
    try:                                                            # so flock conflicts like another process
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with caplog.at_level(logging.WARNING):
            assert pub.run_once() is False
        assert "another process is publishing" in caplog.text and runner.calls == []
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert pub.run_once() is True and len(runner.calls) == 1


def test_manifest_temp_name_is_per_process(live, store, config, monkeypatch):
    import os

    import birdy.publish as publish

    sources = []
    real_replace = os.replace
    monkeypatch.setattr(publish.os, "replace", lambda src, dst: sources.append(Path(src).name) or real_replace(src, dst))
    pub = make(live, store, config, runner=FakeRunner())
    assert pub.run_once() is True
    assert f"manifest.json.tmp.{os.getpid()}" in sources
    names = {p.name for p in pub.mirror_dir.iterdir()}
    assert "manifest.json" in names and not any(".tmp" in n for n in names)


def test_stop_warns_when_thread_outlives_timeout(live, store, config, caplog):
    import threading

    release = threading.Event()

    class Slow:
        def __call__(self, cmd):
            release.wait(5)
            return 0

    pub = make(live, store, config, runner=Slow())
    pub.start()
    pub.notify()
    time.sleep(0.2)
    with caplog.at_level(logging.WARNING):
        pub.stop(timeout=0.1)
    release.set()
    assert "did not stop within 0.1s" in caplog.text


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
        for _ in range(3):                                      # a burst, within the debounce
            pub.notify()
            time.sleep(0.02)
        assert wait_until(lambda: len(runner.calls) >= 2)      # one run for the burst
        n = len(runner.calls)
        time.sleep(0.15)
        assert len(runner.calls) == n                           # no extra run for the burst
        assert wait_until(lambda: len(runner.calls) > n, timeout=3.0)   # max_interval tick
    finally:
        pub.stop()


def test_thread_first_start_ranks_history_once(live, store, config):
    runner = FakeRunner()
    old = bird(store, T0, "Robin", 0.9)
    pub = make(live, store, config, runner=runner, max_interval=60)
    assert not pub.mirror_dir.exists()
    pub.start()
    try:
        assert wait_until(lambda: len(runner.calls) >= 1)
    finally:
        pub.stop()
    assert store.get(old.id).published == "auto"
    manifest = json.loads((pub.mirror_dir / "manifest.json").read_text())
    assert [p["id"] for p in manifest["photos"]] == [old.id]

    newer = bird(store, T0 + timedelta(days=1), "Wren", 0.9)   # unranked, mirror dir now exists
    runner2 = FakeRunner()
    pub2 = make(live, store, config, runner=runner2, max_interval=60)
    pub2.start()
    try:
        assert wait_until(lambda: len(runner2.calls) >= 1)
    finally:
        pub2.stop()
    assert store.get(newer.id).published is None


def test_thread_notify_during_run_causes_another_run(live, store, config):
    class Runner:
        calls = 0

        def __call__(self, cmd):
            self.calls += 1
            if self.calls == 1:
                pub.notify()   # e.g. _sync_mirror hid a photo and re-ranked
            return 0

    runner = Runner()
    pub = make(live, store, config, runner=runner, max_interval=60)
    pub.start()
    try:
        assert wait_until(lambda: runner.calls >= 2)
    finally:
        pub.stop()


def test_thread_stop_returns_promptly(live, store, config):
    runner = FakeRunner()
    pub = make(live, store, config, runner=runner, max_interval=60)
    pub.start()
    assert wait_until(lambda: len(runner.calls) == 1)
    t = time.monotonic()
    pub.stop()
    assert time.monotonic() - t < 1.0
    assert not pub._thread.is_alive()


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


def test_run_once_stops_mid_mirror_build_without_pushing(live, store, config, monkeypatch):
    runner = FakeRunner()
    pub = make(live, store, config, runner=runner, per_species_per_day=10)
    for i in range(5):
        pub.evaluate(bird(store, T0 + timedelta(minutes=i), "Robin", 0.9, size=(60, 80)))
    real_imwrite = cv2.imwrite

    def imwrite_then_stop(*a, **k):
        pub._stop.set()
        return real_imwrite(*a, **k)

    monkeypatch.setattr("birdy.publish.cv2.imwrite", imwrite_then_stop)
    assert pub.run_once() is False
    assert runner.calls == []
    assert not (pub.mirror_dir / "manifest.json").exists()
    assert len(list(pub.photos_dir.rglob("*.jpg"))) <= 1


def test_default_runner_uses_configured_timeout(live, store, config, monkeypatch):
    import subprocess

    from birdy import publish

    assert publish.RSYNC_TIMEOUT == 1800.0
    seen = {}
    monkeypatch.setattr(subprocess, "call", lambda cmd, **kw: seen.update(kw) or 0)
    live.replace(enabled(config))
    pub = Publisher(live, store, store.data_dir / "publish", rsync_timeout=42.0)
    assert pub._runner(["rsync"]) == 0
    assert seen["timeout"] == 42.0
    Publisher(live, store, store.data_dir / "publish")._runner(["rsync"])
    assert seen["timeout"] == publish.RSYNC_TIMEOUT


def test_default_runner_timeout_returns_minus_one_and_logs_value(live, store, config, monkeypatch, caplog):
    import subprocess

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(subprocess, "call", boom)
    pub = Publisher(live, store, store.data_dir / "publish", rsync_timeout=7.0)
    with caplog.at_level(logging.WARNING):
        assert pub._runner(["rsync"]) == -1
    assert "7 s" in caplog.text


def test_write_manifest_removes_tmp_on_failure(live, store, config, monkeypatch):
    pub = make(live, store, config)
    pub.mirror_dir.mkdir(parents=True)

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("birdy.publish.os.replace", fail)
    with pytest.raises(OSError):
        pub._write_manifest([], None)
    assert list(pub.mirror_dir.glob("manifest.json*")) == []


def test_forget_deleted_capture_frees_its_slot(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.9)
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.7)
    pub.evaluate(b)
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)
    deleted = store.delete_capture(a.id)
    pub.forget(deleted)                     # the next-best photo takes the freed slot
    assert store.get(b.id).published == "auto"
    pub.forget(None)                        # nothing to do, no error


def test_evaluate_day_reranks_by_name_and_time(live, store, config):
    pub = make(live, store, config, per_species_per_day=1)
    a = bird(store, T0, "Robin", 0.9)
    b = bird(store, T0 + timedelta(minutes=1), "Robin", 0.7)
    assert pub.evaluate_day("Robin", T0 + timedelta(hours=2)) is True
    assert (store.get(a.id).published, store.get(b.id).published) == ("auto", None)
    assert pub._wake.is_set()
    assert pub.evaluate_day("Robin", T0) is False
    assert pub.evaluate_day("Nobody", T0) is False
    c = bird(store, T0 + timedelta(minutes=2), "Robin", 0.95)
    assert pub.evaluate(c) is True          # evaluate goes through evaluate_day
    assert [store.get(x.id).published for x in (a, b, c)] == [None, None, "auto"]
    assert pub.evaluate(c) is False
    assert pub.evaluate_day(c.species, c.captured_at) is False


def test_mirror_uses_the_dslr_photo_of_a_promoted_capture(live, store, config):
    pub = make(live, store, config)
    c = bird(store, T0, "Robin", 0.9)
    eos = c.path.replace(".jpg", "_eos.jpg")
    cv2.imwrite(str(store.photos_dir / eos), np.full((2000, 3000, 3), 90, np.uint8))
    store.store_verdict(c.id, is_bird=True, confidence=0.9, species="Robin", species_confidence=0.9,
                        dslr_path=eos, dslr_box=(0.4, 0.4, 0.2, 0.2))
    store.set_published(c.id, "forced")
    assert pub.run_once() is True
    c = store.get(c.id)
    assert c.web_path == eos
    web = cv2.imread(str(pub.photos_dir / eos))
    assert max(web.shape[:2]) == 1600
    assert (pub.thumbs_dir / eos).exists()
