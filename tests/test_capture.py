"""run_capture is Pi-only at runtime, but its tunable handling is testable with a stubbed camera."""

import dataclasses
import logging
import threading
import time
from contextlib import contextmanager

import numpy as np

from birdy import capture
from birdy.capture import _build_tunables, run_capture
from birdy.state import RuntimeState


def test_build_tunables_uses_pinned_lores_size_not_config(config):
    cfg = dataclasses.replace(config, camera=dataclasses.replace(config.camera, lores_size=(640, 480)))
    detector, _gate = _build_tunables(cfg, (160, 120))
    assert detector._mask.shape == (120, 160)


class RecordingState(RuntimeState):
    def __init__(self):
        super().__init__()
        self.scores: list[int] = []

    def update(self, **fields):
        if "motion_score" in fields:
            self.scores.append(fields["motion_score"])
        super().update(**fields)


def _wait(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_run_capture_pins_camera_and_rebuilds_only_when_relevant(live, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="birdy.capture")
    frames = {"n": 0}

    @contextmanager
    def fake_session(cfg):
        assert cfg.camera.lores_size == (320, 240)  # pinned startup camera, even after replace()
        yield object()

    def fake_lores_grey(picam2, lores_size):
        assert lores_size == (320, 240)
        frames["n"] += 1
        # Alternate brightness so the score is non-zero once the background exists.
        return np.full((240, 320), 200 if frames["n"] % 2 else 0, np.uint8)

    stills: list[int] = []
    monkeypatch.setattr(capture, "camera_session", fake_session)
    monkeypatch.setattr(capture, "lores_grey", fake_lores_grey)
    monkeypatch.setattr(capture, "_capture_still", lambda picam2, store, pipeline, score: stills.append(score))
    monkeypatch.setattr(capture, "FRAME_INTERVAL", 0.02)

    state = RecordingState()
    stop = threading.Event()
    store = pipeline = None  # only ever passed to the stubbed _capture_still
    t = threading.Thread(target=run_capture, args=(live, store, pipeline, state, stop), daemon=True)
    t.start()
    try:
        assert _wait(lambda: frames["n"] >= 3)
        base = live.get()
        # Relevant change (motion) + a camera change that must NOT reach the running session.
        live.replace(
            dataclasses.replace(
                base,
                camera=dataclasses.replace(base.camera, lores_size=(160, 120)),
                motion=dataclasses.replace(base.motion, min_area=10**9),
            )
        )
        assert _wait(lambda: any("v1" in r.message for r in caplog.records))
        n_before = frames["n"]
        assert _wait(lambda: frames["n"] >= n_before + 5)
        # Irrelevant change: only the classifier threshold.
        cur = live.get()
        live.replace(dataclasses.replace(cur, classifier=dataclasses.replace(cur.classifier, threshold=0.99)))
        assert _wait(lambda: any("v2" in r.message for r in caplog.records))
        # site.name is cosmetic: must not rebuild (and re-warm) the detector.
        cur = live.get()
        live.replace(dataclasses.replace(cur, site=dataclasses.replace(cur.site, name="elsewhere")))
        assert _wait(lambda: any("v3" in r.message for r in caplog.records))
    finally:
        stop.set()
        t.join(timeout=2.0)
    assert not t.is_alive()

    messages = [r.message for r in caplog.records]
    assert not any("frame processing failed" in m for m in messages), messages
    assert not any("camera session failed" in m for m in messages), messages
    assert "settings applied (v1) (detector rebuilt)" in messages
    assert "settings applied (v2)" in messages
    assert "settings applied (v3)" in messages
    assert state.state == "capturing"
    assert state.scores and max(state.scores) > 0


class FakeGate:
    """Active only when no site is set, so tests can pause/unpause via lat/lon."""

    def __init__(self, lat, lon, margin_minutes):
        self.lat = lat

    def is_active(self, now):
        return self.lat is None

    def next_change(self, now):
        return None


def test_paused_loop_wakes_promptly_on_settings_change(live, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="birdy.capture")

    @contextmanager
    def fake_session(cfg):
        yield object()

    monkeypatch.setattr(capture, "DaylightGate", FakeGate)
    monkeypatch.setattr(capture, "camera_session", fake_session)
    monkeypatch.setattr(capture, "lores_grey", lambda picam2, lores_size: np.zeros((240, 320), np.uint8))
    monkeypatch.setattr(capture, "FRAME_INTERVAL", 0.02)
    base = live.get()
    live.replace(dataclasses.replace(base, site=dataclasses.replace(base.site, lat=38.72, lon=-9.14)))

    state = RuntimeState()
    stop = threading.Event()
    t = threading.Thread(target=run_capture, args=(live, None, None, state, stop), daemon=True)
    t.start()
    try:
        assert _wait(lambda: state.state == "daylight_paused")
        cur = live.get()
        live.replace(dataclasses.replace(cur, site=dataclasses.replace(cur.site, lat=None, lon=None)))
        assert _wait(lambda: state.state == "capturing", timeout=3.0)  # not GATE_RECHECK (60 s)
        # ...and pausing again inside a session is immediate too.
        cur = live.get()
        live.replace(dataclasses.replace(cur, site=dataclasses.replace(cur.site, lat=38.72, lon=-9.14)))
        assert _wait(lambda: state.state == "daylight_paused", timeout=3.0)
    finally:
        stop.set()
        t.join(timeout=3.0)
    assert not t.is_alive()
